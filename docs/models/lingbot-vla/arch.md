# LingBot-VLA 2.0 架构分析

**LingBot-VLA 2.0** 是面向真实机器人的 Vision-Language-Action（VLA）基础模型：输入多相机图像 + 语言指令 + 本体状态，输出一段动作 chunk。相对 1.0，重点在跨本体泛化、更大动作空间，以及用深度/视频预测做辅助监督。

源码位置：`edgeLLM/lingbot-vla-v2/`  
论文：[arXiv:2607.06403](https://arxiv.org/pdf/2607.06403)

---

## 1. 功能定位

| 能力 | 含义 |
|------|------|
| 感知 | 多视角 RGB → Qwen3-VL 视觉塔 |
| 理解 | 语言指令 → Qwen3-VL 文本侧 |
| 决策 | Action Expert（Qwen2 + MoE）用 Flow Matching 生成动作 |
| 辅助任务 | Dual-Query：蒸馏 LingBot-Depth（几何）+ DINO-Video（时序语义） |

预训练约 6 万小时（5 万机器人轨迹 + 1 万 egocentric），统一到 **55 维** canonical state/action，覆盖臂、末端、夹爪、灵巧手、腰、头、底盘等。

相对 1.0 的三项核心提升：

- **跨任务 / 跨本体泛化**：重设计数据管线，约 60,000 小时预训练数据（含 20 种机器人配置）
- **扩展动作空间**：统一表示支持臂、末端、夹爪、灵巧手、腰、头、移动底盘，而非仅标准双臂
- **预测动力学建模**：未来预测作为 proxy task；DINO-Video 提供语义时序先验，LingBot-Depth 提供几何线索

---

## 2. 总体架构

```text
观测 (多相机图像 + 语言 + state)
        │
        ▼
┌───────────────────────────────────────┐
│  Prefix（VLM 侧，算一次可缓存 KV）      │
│  Qwen3-VL Vision + Language           │
│  + Current/Future Query tokens        │
│    （Depth / Video 蒸馏用）            │
└──────────────────┬────────────────────┘
                   │ KV cache
                   ▼
┌───────────────────────────────────────┐
│  Suffix（Action Expert，多步去噪）      │
│  state_emb + noisy_action + time_emb  │
│  Qwen2 layers（部分层 Token-MoE）      │
│  → action_out_proj → velocity v_t     │
└──────────────────┬────────────────────┘
                   │ Flow Matching ODE
                   ▼
            动作 chunk (T × D)
```

### 核心类关系

| 类 | 职责 |
|----|------|
| `LingbotVlaV2Policy` | 对外 Policy（训练 `forward` / 推理 `sample_actions`） |
| `FlowMatchingV2` | Flow Matching + prefix/suffix 组装 + 蒸馏头 |
| `QwenvlWithExpertV2Model` | Qwen3-VL + Qwen2 Action Expert 联合前向（共享层间 attention） |
| `LingbotVLAV2Config` | V2 配置（默认 `flex_cached` attention、Qwen3-VL 相关开关） |

相对 V1：VLM 从 Qwen2.5-VL 换到 **Qwen3-VL**，attention 默认 `flex_cached`，Action Expert 头数更大（32 Q / 8 KV），并强化 MoE 与双 query 蒸馏。

---

## 3. 核心原理

### 3.1 Flow Matching 动作生成

**训练**：对真值动作 \(a\) 加噪

\[
x_t = t \cdot \varepsilon + (1 - t) \cdot a
\]

网络预测速度场 \(v_t\)，损失为 MSE / L1（`fm` / `L1_fm`）。

**推理**：从噪声出发，约 **10 步** Euler 积分

\[
x \leftarrow x + dt \cdot v_t
\]

（`num_steps=10`），得到动作 chunk。Prefix 只算一次并缓存 KV，每步只跑 suffix。

### 3.2 Prefix / Suffix 双塔式联合注意力

- **Prefix**：图像 token（含 vision start/end）+ 语言 + alignment query
- **Suffix**：`state_proj(state)` + `action_time_mlp(noisy_action, time)`
- **层间**：VLM 与 Expert 同步过层；suffix 可 attend prefix（可用 mask 挡住 future depth → action，避免泄漏）

#### 联合前向机制（与 π₀ / π₀.₅ 同构）

`QwenvlWithExpertV2Model` 的联合前向**本质上与 π₀.₅ 同一套路**：层对齐双塔 + concat attention，而不是独立的 `CrossAttention(q=action, kv=vl)` 模块。

每层大致为：

```text
VLM layer_i:    h_prefix → Q_p, K_p, V_p   （Qwen3-VL 权重）
Expert layer_i: h_suffix → Q_s, K_s, V_s   （Qwen2 权重）

Q = cat([Q_p, Q_s], seq)
K = cat([K_p, K_s], seq)
V = cat([V_p, V_s], seq)

Attn(Q, K, V)  ← 一次联合 attention
再按长度切回各自 residual / MLP
```

对应实现：对 `inputs_embeds=[prefix, suffix]` 分别 `compute_kqv`，再 `torch.cat` 后做 attention——与 π₀ / π₀.₅ 的 `PaliGemmaWithExpert.compute_layer_complete` 同构。

#### 推理时为何看起来像 cross-attn

推理拆成两段：

1. **Prefill**：`inputs_embeds=[prefix, None]`，只跑 VLM，写入 `past_key_values`
2. **Denoise × N**：`inputs_embeds=[None, suffix]`，suffix 的 Q 与 **cached prefix K/V + 当前 suffix K/V** 一起算

去噪步里，action 对 VL 条件的注入语义上就是 **action Q × VL（prefix）K/V**，再加 suffix 内部 self-attn。这与 Chamleon 文档中「suffix Q cross-attend prefix KV cache」是一回事。

训练时通常 prefix+suffix 一起过（联合 self-attn），不一定走 cache；mask 保证 **prefix 不读 suffix**，**suffix 可读 prefix**。

#### 与「独立 CrossAttention」的差别

| | 说法 | 实际 |
|--|------|------|
| 模块形态 | 不是 `nn.MultiheadAttention(q=action, kv=vl)` | 同层 concat QKV + 2D/block mask |
| 信息流 | 等价于 action → VL 的 cross | 同时还有 VL 自注意力、action 自注意力 |
| 权重 | VL / Expert **各自** QKV、MLP | 不是共享一套 transformer |

**一句话**：信息流与 π₀.₅ 同类；实现是联合 self-attn + mask，不是 DiT 式独立 cross-attn 层。

### 3.3 统一动作表示（55 维）

| 维度 | 含义 |
|------|------|
| 14 | 臂关节位置 |
| 14 | 末端位姿 |
| 2 | 夹爪位置 |
| 12 | 手关节位置 |
| 4 | 腰位置 |
| 2 | 头位置 |
| 3 | 移动底盘信号 |
| 4 | 预留 |

通过 `configs/robot_configs/*.yaml` 把各机器人原始 state/action 映射到统一特征（如 RoboTwin：双臂关节 + gripper）。`FeatureTransform` 负责归一化 / 反归一化。

### 3.4 MoE Action Expert

MoE **只装在 Action Expert 的 FFN 上**，不碰 Qwen3-VL。目的：在固定激活算力下，让不同本体/任务走不同专家。

#### 用在哪

```text
每层 Action Expert DecoderLayer:
  Attention（与 VLM concat 联合 attn）  ← 仍是 dense
  → post_attn_norm
  → MLP  ← 这里换成 Qwen2TokenMoeBlock（MoE）
```

安装点：`QwenvlWithExpertV2Model._install_moe_blocks`，把 `qwen_expert.model.layers[i].mlp` 替换成 MoE。RoboTwin / real_robot 配置下 **36 层全开**（`token_moe_layers: 0..35`）。

过 MoE 的 token 只有 **suffix**（state + noisy action + time），即 denoise 路径上的 action 侧 hidden；prefix VL token 走 VLM 自己的 dense MLP。

#### 单层结构（典型配置）

| 项 | 值（robotwin.yaml） | 含义 |
|----|---------------------|------|
| `token_num_experts` | 32 | 路由专家数 |
| `token_top_k` | 4 | 每 token 激活 4 个 |
| `token_moe_intermediate_size` | 512 | 路由专家 FFN 宽度（细粒度） |
| `token_shared_intermediate_size` | 704 | 共享专家宽度 |
| `router_activation` | sigmoid | 路由分数 |
| `routed_scaling_factor` | 4.0 | 路由输出缩放 |
| `use_shared_expert_gate` | false | 共享专家不加 sigmoid gate |
| `moe_implementation` | fused | group GEMM |

前向：

```text
h (B, T_suffix, D)
  → gate(h) → scores (sigmoid/softmax)
  → + e_score_correction_bias → top-k 选专家
  → 加权求和 routed experts
  → + shared_expert(h)          # 每个 token 必算
  → 输出
```

设计意图（fine-grained + shared isolation）：

- **Routed experts**：细、多，专学 embodiment/task 差异
- **Shared expert**：所有 token 共享，保留通用动作先验
- 激活量约 `top_k/E = 4/32`，容量大、算力可控

#### 在 Action 训练 / 推理中的路径

**训练（联合 forward）**：每层 Expert 先算 QKV → 与 VLM concat attn → 切回 suffix → **MoE MLP**。路由只看 action 侧 token。

**推理**：

1. Prefill：只跑 VLM，**不进 MoE**
2. Denoise × N：每步 suffix 过 36 层 Expert，**每层都路由一次**；同一观测下不同去噪步 / 不同 action token 可选不同专家

#### 负载均衡

| 机制 | 作用 |
|------|------|
| **Loss-free bias**（`bias_update_speed`） | optimizer pre-hook 按负载更新 `e_score_correction_bias`（DeepSeek 式）；post-train 常设 `0`，改用辅助损失 |
| **sequence-wise balance loss**（`1e-3`） | 每个 sample 的 T 个 token 上专家负载均衡 |
| **router z-loss**（`1e-4`） | 抑制 router logits 过大 |
| **expert LR scale** | `use_moe_expert_lr: true`，routed 专家 LR × √(E/top_k) |

#### 与 π₀.₅ 对比

```text
π₀.₅ Expert layer:  Attn → Dense MLP
LingBot Expert:     Attn → MoE(MLP) = Shared + TopK(Routed×32)
```

π₀.₅ Action Expert 是 dense FFN；LingBot 把同一位置换成 Token-MoE，专门服务跨本体 action 建模。Attention 仍是双塔 concat，MoE 只扩 FFN 容量。

#### 代码结构

关键文件：

| 文件 | 内容 |
|------|------|
| `qwen2_action_expert.py` | `Qwen2DecoderLayer` / `Qwen2TokenMoeBlock` / `Qwen2FusedExperts` |
| `modeling_lingbot_vla_v2.py` | 双塔联合 forward + `_install_moe_blocks` |

**对象树：**

```text
LingbotVlaV2Policy
└── FlowMatchingV2
    └── QwenvlWithExpertV2Model
        ├── qwenvl          # Qwen3-VL（dense，无 MoE）
        └── qwen_expert     # Qwen2ForCausalLM ← Action Expert
            └── model       # Qwen2Model
                └── layers[0..35]   # Qwen2DecoderLayer × 36
                    ├── self_attn   # dense
                    └── mlp         # 默认 Qwen2MLP；use_moe 后换成 Qwen2TokenMoeBlock
                        ├── gate
                        ├── experts          # 32 个 routed（fused 3D 权重 or ModuleList）
                        ├── shared_expert
                        └── e_score_correction_bias
```

**安装：** 先建成普通 Qwen2（每层 `mlp = Qwen2MLP`），再由 `_install_moe_blocks` 按 `token_moe_layers` 替换为 `Qwen2TokenMoeBlock`——只改 `qwen_expert`，不动 `qwenvl`。

**一层 Decoder 两拍调用**（配合联合 attention）：

```text
拍 1 compute_kqv=True:
  suffix_h → input_norm → q/k/v_proj → 返回 Q_s, K_s, V_s
  （与 VLM 的 QKV cat → joint Attn）

拍 2 output_atten=True:
  att_output[:, start:end] → o_proj → + residual
  → post_attn_norm → mlp(=MoE) → + residual
  → 返回 (out_emb, router_logits)
```

外层 `QwenvlWithExpertV2Model.forward` 中 `i==0` 为 VLM、`i==1` 为 Expert；仅 Expert 分支收集 `router_logits`。

**`Qwen2TokenMoeBlock` 内部：**

```text
Qwen2TokenMoeBlock
├── gate: Linear(D → 32)              # token 级路由（fp32）
├── experts:
│   ├── fused: Qwen2FusedExperts      # gate/up/down 各 [E, ...]
│   └── 或 eager: ModuleList[MLP×32]
├── shared_expert: SwiGLU MLP         # 全 token 必过
├── shared_expert_gate?               # 配置里常关掉
└── e_score_correction_bias [E]       # loss-free 负载均衡
```

Routed / Shared 均为 `down(act(gate(x)) * up(x))`。Fused 权重形状便于 group GEMM / FSDP：

```text
gate_proj: [32, 512, D]
up_proj:   [32, 512, D]
down_proj: [32, D, 512]
```

Forward：`[B,T,D] → flatten → gate → top-k → experts → + shared → reshape`，并返回 `router_logits` 供 seq-wise / z-loss。

**代码视角对照：**

| 位置 | π₀.₅ / dense | LingBot MoE |
|------|--------------|-------------|
| `layer.mlp` | 单个 `MLP(D→I→D)` | `Qwen2TokenMoeBlock` |
| Attention | 双塔 concat | 同 |
| 谁进 MoE | — | 只有 suffix `i==1` 的 hidden |
| 额外输出 | 无 | `router_logits` → 辅助损失 |

**串起来：** Action Expert = 36 层 `Qwen2DecoderLayer`；Attn dense，`mlp` 被换成 MoE；联合 forward 每层两拍（`compute_kqv` → joint attn → `output_atten` 跑 MoE）；只处理 action/state suffix token。

**一句话**：MoE 是 Action Expert 的稀疏 FFN——只对 action/state token 做 token 级 top-4/32 路由，加 shared expert；VLM 保持 dense，用更大参数容量学跨本体动作，而不成倍增加 denoise 激活算力。

### 3.5 Dual-Query Distillation

在视觉/文本后追加 learnable query：

- **Current depth query** ← LingBot-Depth / MoRGBD 几何特征
- **Future depth / video query** ← 未来帧几何 + DINO-Video 时序语义

用 query 对齐损失逼 VLM 表征同时编码「当前几何」和「未来演化」，再服务动作预测。

---

## 4. 代码与数据流

```text
lingbot-vla-v2/
├── lingbotvla/models/vla/lingbot_vla/
│   ├── modeling_lingbot_vla_v2.py   # V2 主模型
│   ├── modeling_lingbot_vla.py      # V1 + FlowMatching 基类 / 蒸馏头
│   ├── qwen3vl_in_vla.py            # Qwen3-VL 适配
│   ├── qwen2_action_expert.py       # Action Expert + MoE
│   └── configuration_lingbot_vla.py # LingbotVLAV2Config
├── lingbotvla/data/vla_data/        # LeRobot 数据、特征变换
├── tasks/vla/train_lingbotvla.py    # 训练入口
├── deploy/lingbot_vla_v2_policy.py  # 真机 / websocket 推理
├── configs/robot_configs/           # 本体特征映射
└── configs/vla/                     # 训练 YAML
```

**训练一步**：图像 / 语言 / state / action → Flow Matching 主损失 +（可选）depth/video 蒸馏 + MoE 辅助损失。

**推理一步**：`select_action` → `sample_actions`（prefix KV → 10 步去噪）→ `FeatureTransform.unapply` 还原到机器人原始动作空间。RTX 4090D 约 **130 ms / 次**（10 步）。

---

## 5. 与 π₀.₅ / Chamleon 的关系

结构同属「VLM prefix + Action Expert + Flow Matching」；联合 attention 骨架与 π 系同构（见 §3.2），差异主要在 backbone 与附加模块：

| | π₀.₅ / Chamleon | LingBot-VLA 2.0 |
|--|-----------------|-----------------|
| VLM | PaliGemma | Qwen3-VL-4B |
| Expert | Gemma | Qwen2 + Token MoE |
| 条件注入 | suffix Q cross-attend prefix KV（concat attn + mask） | **同构** |
| State 位置 | π₀.₅ 常把 state 离散进 **prefix 语言**；π₀ 用 suffix `state_proj` | 多在 **suffix** 用 `state_proj`（更像 π₀） |
| 额外监督 | 较少 | Depth + Video query 蒸馏 |
| 动作空间 | 多为双臂 | 55 维多本体统一表示 |

Chamleon 里的 stage 切分（`vit` / `llm_prefix` / `action_expert`）与这里的 Vision / Prefix LLM / Action Expert 一一对应，后续若做边缘部署可按同样方式拆 stage（prefix prefill 1 次 + denoise 只跑 expert）。

---

## 6. 一句话总结

**LingBot-VLA 2.0 = Qwen3-VL 感知理解 + MoE Action Expert 的 Flow Matching 动作生成 + Depth/Video 双 query 蒸馏**，用统一 55 维动作空间把大规模异构数据接到真实机器人应用上。
