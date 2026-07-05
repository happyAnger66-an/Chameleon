# Cosmos3 Policy vs π₀.₅：架构对比

> 对比 **Cosmos3-Nano-Policy-DROID**（diffusers / MoT DiT）与 **Physical Intelligence π₀.₅**（openpi / PaliGemma + Action Expert）在机器人 policy 上的设计异同。  
> 相关文档：[cosmos3_policy_detail.md](./cosmos3_policy_detail.md)、[cosmos3_arch_dit.md](./cosmos3_arch_dit.md)、[cosmos3_arch_vae.md](./cosmos3_arch_vae.md)。

**源码：**

| 系统 | 路径 |
|------|------|
| Cosmos3 | `diffusers/.../pipeline_cosmos3_omni.py`、`transformer_cosmos3.py` |
| π₀ / π₀.₅ | `openpi/src/openpi/models_pytorch/pi0_pytorch.py`、`gemma_pytorch.py` |

---

## 一、一句话对照

| | π₀.₅ | Cosmos3 Policy |
|---|------|----------------|
| **定位** | VLM prefill + 小 Action Expert **只 denoise action** | 单塔 MoT DiT **联合 denoise video latent + action** |
| **条件注入** | suffix Q cross-attend **prefix KV cache** | gen Q full-attn **und K/V + gen K/V** |
| **推理优化** | prefix **算 1 次**，denoise N 步只跑 ~300M expert | 每 denoising step **16B 全序列 forward** |
| **视觉** | SigLIP → LLM token | Wan VAE → latent → DiT patch token |

两者思路同属 **Prefix-LM 条件生成 + flow matching**；π₀.₅ 把 cache 与模型拆分做到极致（低延迟 policy），Cosmos3 把联合 world model 做到极致（视频 + action 同训同推）。

---

## 二、共同骨架

### 2.1 「理解段 + 生成段」

| 概念 | π₀ / π₀.₅ | Cosmos3 MoT |
|------|-----------|-------------|
| **理解段** | SigLIP 图像 token + 语言 token（π₀.₅：state 离散化进 prompt） | 文本 token（`und_seq`，因果 self-attn） |
| **生成段** | Action chunk token | Vision latent token + action token（+ 可选 sound） |
| **条件 → 生成** | suffix 的 Q attend prefix 的 K/V | gen 的 Q attend **und + gen** 的 K/V |
| **生成 → 条件** | **blocked**（prefix 不 attend action） | **blocked**（und 不 attend gen） |
| **去噪算法** | Flow matching（预测 velocity `u_t = noise - action`） | 同（UniPC + flow matching） |
| **时间步** | π₀.₅：adaRMSNorm 注入；π₀：time 与 action concat 过 MLP | `time_embedder` scatter 到 noisy token |

### 2.2 信息流向（抽象）

```text
π₀.₅:
  [图像 + 语言 (+ state@π₀.₅)]  ──prefill──►  prefix K/V cache（1 次）
                        ▲
  [noisy action chunk]  ──denoise×N──►  suffix Q 读 prefix KV

Cosmos3:
  [文本]  ──und 通路──►  因果 self-attn（每 step 仍在网内）
  [VAE latent + action 噪声]  ──gen 通路──►  full attn → und KV + gen KV
                        （每 denoising step 整网 forward）
```

### 2.3 DROID Policy 下的直观类比

Policy 模式下 packed 序列切分为：

```text
[ JSON caption × und_len ] [ vision latent tokens ] [ action noise × chunk_size ]
└──────── und_seq ────────┘ └──────────────── gen_seq ────────────────────────┘
```

与 π₀.₅ 的对应关系（抽象层面 **非常接近**）：

| 步骤 | π₀.₅ | Cosmos3 Policy |
|------|------|----------------|
| 理解段 | VLM：**图像 + 语言**（+ state）→ prefix | **und：仅文本**（`embed_tokens`） |
| 生成段 | suffix：**noisy action** | **gen：vision latent + noisy action** |
| 条件注入 | suffix **Q** cross-attend **prefix K/V cache** | gen **Q** full-attn **[und K/V \| gen K/V]** |
| 反向 blocked | prefix 不读 action | und 不读 gen（vision/action） |

```text
π₀.₅:
  [SigLIP img + text] ──prefill 1×──► prefix K/V cache
  [action noise]      ──denoise×N──► Q=action,  K/V=cache

Cosmos3 Policy:
  [text]              ──und──► Q/K/V + 因果 self-attn（每 step 仍在网内）
  [VAE z + action]    ──gen──► Q_gen full-attn [K_und | K_vision | K_action]
                               ↑ 文本条件      ↑ vision↔action 互相可见
```

**三个易混点：**

1. **π₀.₅ 把 vision 放在 prefix，Cosmos3 把 vision 放在 gen**——Cosmos3 的 vision latent 与未来帧一起在扩散环里更新（policy 下 z₀ 为 condition，z₁…₄ 每步变），不是「encode 一次就固定」的 ViT 特征。
2. **π₀.₅ 有显式 prefix KV cache，Cosmos3 没有**——语义上等价于「gen 读 und K/V」，但工程上每 denoise step 仍 **16B 全序列 forward**，文本 K/V 每步重算。
3. **Cosmos3 gen 内部还有 vision ↔ action 联合建模**——π₀.₅ suffix 只有 action；Cosmos3 允许 gen token 彼此 full-attn，action 与 rollout video latent **绑定去噪**。

> **一句话**：π₀ suffix cross-attend prefix KV ≈ Cosmos3 gen cross-attend und KV；Cosmos3 额外把 vision 放进扩散、且每步跑全 16B、无 prefix cache。

---

## 三、架构差异

### 3.1 总体拓扑

```text
π₀ / π₀.₅                              Cosmos3 Policy
─────────────────                      ─────────────────
SigLIP ──► PaliGemma ~2B               像素 ──► Wan VAE ──► latent
     ╲                                      ╲
      ╲── prefix K/V cache (prefill 1次)      ├── gen_seq (vision + action)
           ▲                                 │
Gemma Action Expert ~300M                    Cosmos3OmniTransformer ~16B (MoT)
  suffix Q ──cross──► prefix KV              gen Q ──full attn──► und KV + gen KV
  (每 denoise step 仅 expert)                (每 denoise step 全 36 层)
```

| 维度 | π₀ / π₀.₅ | Cosmos3 |
|------|-----------|---------|
| 模型数量 | **两个** Gemma（VLM + Action Expert） | **一个** MoT DiT |
| Attention 实现 | 同层 concat Q/K/V + 2D mask | 双通路投影 + 两种 attn 规则 |
| 推理 KV cache | **显式** prefix cache | **无** 典型 prefix cache |
| 视觉编码 | SigLIP patch token（LLM 隐空间） | Wan VAE latent + DiT 2×2 patchify |
| 输出 | **仅** action chunk | action + rollout 视频 latent |
| 参数量 | ~2B + ~300M | ~16B + VAE (~150M encoder) |

### 3.2 π₀ 与 π₀.₅ 的差异（openpi 内部）

来自 `pi0_config.py` 注释：

1. **π₀.₅**：robot state 作为 **离散语言 token** 进入 prefix，而非 π₀ 的连续 `state_proj` suffix token。
2. **π₀.₅**：Action Expert 用 **adaRMSNorm** 注入 flow matching 时间步；π₀ 用 time 与 action embedding concat 过 MLP。
3. **π₀.₅**：`max_token_len=200`（π₀ 为 48）。

Cosmos3 无直接对应：本体差异靠 `domain_id` + `DomainAwareLinear`，时间靠 `time_embedder`。

### 3.3 MoT「双塔」参数量：不是两个独立 8B

常见误解：MoT = und 塔 8B + gen 塔 8B = 16B，两个结构相同、权重不同的子模型串行推理。

**实际：**

- **一个 ~16B checkpoint**（`Cosmos3OmniTransformer`），一次 load、一次 forward。
- 36 层内 **Attn + MLP 近似复制一份**给 gen（`to_q/k/v` vs `add_q/k/v_proj`，`mlp` vs `mlp_moe_gen`），故 und 侧约占 **~一半权重（~8B 量级）**，但不是可单独拆出的 8B 子网。
- **不能**两个塔分开推理：gen 的 Q 必须 attend und 的 K/V，拆开即破坏条件注入。
- 外围模块 **不对半**：und 侧有 `embed_tokens`、`lm_head`；gen 侧有 `proj_in/out`、`action_proj_*` 等。

| 层级 | und 通路 | gen 通路 | 权重关系 |
|------|----------|----------|----------|
| 36×Attn | `to_q/k/v/out` + QK-Norm | `add_q/k/v_proj` + `to_add_out` | **结构对称，张量独立**，联合训练 |
| 36×MLP | `mlp` | `mlp_moe_gen` | 同上 |
| 36×LN | `input/post_attention_layernorm` | `*_moe_gen` | 不共享 |
| 头 | `embed_tokens`、`lm_head`（policy 不用 lm_head 出文本） | `proj_in/out`、`action_proj_*` | 模态专用 |

Chamleon stats 中 `dit Weight ≈ 30 GB`（bf16）≈ **15B 参数**，与官方 ~16B 一致；这是 **整网** dit，不是「文本子模型 8B」。

---

## 四、视觉编码：SigLIP ViT vs Wan VAE

> 详尽的 VAE 结构见 [cosmos3_arch_vae.md](./cosmos3_arch_vae.md)。本节只对比 **视觉 front-end** 的设计差异。

两类编码解决的是不同问题：**π₀.₅ 的 SigLIP ViT 做「看懂」**，**Wan VAE 做「压像素、可生成」**——不是同一类模块的简单替换。

### 4.1 本质定位

| | π₀.₅ SigLIP ViT | Cosmos3 Wan VAE |
|---|-----------------|-----------------|
| **类型** | 2D Vision Transformer（PaliGemma `vision_tower`） | 3D 因果 KL-VAE（`AutoencoderKLWan`） |
| **训练目标** | 图像-文本对比 / VLM 对齐（**语义理解**） | 视频重建 + KL（**生成式 codec**） |
| **输出** | LLM 空间 token：`[num_patches, embed_dim]` | Latent 网格：`[48, T', H/16, W/16]` |
| **可 decode 回像素** | **否**（单向特征） | **是**（对称 Decoder） |
| **在 policy 里的角色** | 观测 → **固定条件**（prefill 1 次） | 观测 → **条件 latent z₀** + **未来 latent 可扩散** |

π₀.₅ 代码路径（`pi0_pytorch.py`）：

```python
# SigLIP vision_tower → multi_modal_projector → prefix token
return self.paligemma_with_expert.embed_image(img)
```

Cosmos3：像素 → VAE encode → per-channel 归一化 → DiT 内 2×2 patchify → `proj_in`。

### 4.2 结构与数据流

```text
π₀.₅（每相机 1 帧 RGB 224×224）:
  RGB ──► SigLIP ViT ──► projector ──► [~256 tokens × N 相机] ──► PaliGemma prefix
         （2D patch，无时间维）

Cosmos3 Policy:
  RGB 视频 [1,3,17,H,W] ──► WanEncoder3d（3D 因果 CNN）──► [1,48,5,h,w] latent
                              │
                              └──► 可 decode 回像素；t=1…4 可 flow matching 去噪
```

| 维度 | SigLIP ViT | Wan VAE |
|------|------------|---------|
| **输入** | 单帧 `224×224`（openpi `IMAGE_RESOLUTION`），多相机各编码 | 视频 clip（Policy：17 帧窗口，仅首帧为真实观测） |
| **时间** | **无**；多帧 = 多相机或多张独立图 | **原生 3D**；时间 `/4` 压缩 → 5 latent 步 |
| **空间** | patch → ~256 token/图（224/14 patch 量级） | 空间 `/16`（相对原像素；含 patchify×2 + encoder×8） |
| **算子** | ViT Self-Attention | CausalConv3d + ResBlock + 下采样 |
| **参数量** | 含在 PaliGemma ~2B 内（SigLIP So400m 量级） | Encoder ~150M（Wan 2.2） |

### 4.3 核心差异（为何不能互换）

#### （1）语义理解 vs 生成 codec

- **SigLIP**：为「图里有什么、与语言如何对齐」优化，特征在 **PaliGemma 语义空间**，适合 **条件化 action**。
- **VAE**：为「像素 ↔ latent 可逆、latent 适合扩散」优化，特征在 **生成 latent 空间**，适合 **预测未来视频 + 联合 action**。

#### （2）单帧 2D token vs 时空 latent 网格

- π₀.₅：例如 3 相机 × ~256 token ≈ **768 视觉 token**，全是 **当前时刻** 的 2D 语义。
- Cosmos3：一个 **5×h×w 的时空 latent 网格**；DiT 对 t=1…4 **去噪成未来 rollout**。SigLIP 路径 **不预测未来像素/latent**。

#### （3）推理是否参与 denoising 迭代

- π₀.₅：ViT **只跑 1 次**（prefill），denoise 环只更新 action。
- Cosmos3：VAE encode **1 次**，但 **vision latent 每步 denoising 都变**；VAE decode 在 loop 结束后 **再跑 1 次**。VAE 提供的是 latent **空间**与 **起点**，不是全程固定的 ViT 特征。

#### （4）分辨率策略

- SigLIP：**固定 224×224**，细节在 patch 化时已 trade-off，靠 VLM 预训练补语义。
- Cosmos3：**resolution tier**（如 480p canvas）+ 高倍压缩；空间低于原图，但保留 **layout / 运动** 供 world model；Policy 会裁 padding 区域。

### 4.4 优缺点

#### SigLIP ViT（π₀.₅）

**优点**

- **轻、快**：单帧 2D ViT，encode 1 次，适配 prefill + KV cache。
- **语义强**：大规模 image-text 预训练，语言 grounding、物体/任务理解好。
- **与 LLM 同空间**：projector 后进 Gemma，**无需 VAE↔DiT 分布对齐**。
- **多相机自然**：每路独立 encode，concat 为 prefix token。
- **部署成熟**：224 输入、ViT 量化 / TensorRT 路径多。

**不足**

- **无 native 视频**：时间不建模；运动靠多相机 + 语言隐式推断。
- **不可重建像素**：无「想象未来画面」的 decode 路径。
- **固定分辨率**：224 对精细 manipulation（小物体、腕部细节）可能不够。
- **与 action 解耦**：视觉只做条件，不联合优化 future video。

#### Wan VAE（Cosmos3）

**优点**

- **生成式 codec**：encode/decode 对称，**同一 latent 空间**可 rollout 未来视频。
- **3D 因果**：时间 `/4` 压缩，适合 video policy / world model；chunk + `feat_cache` 可流式 encode。
- **高分辨率 pipeline**：可与 tier canvas 配合，不绑死 224。
- **与 DiT 联合训练**：视觉 latent 与 action **同一 flow matching**，时空一致性强。

**不足**

- **更重**：3D CNN encoder + 因果分块，比单帧 ViT forward 贵。
- **语义非 VLM 级**：优化目标是重建；**物体/语言对齐**靠后面 16B DiT + 文本，不靠 VAE 本身。
- **有损压缩**：`/16` 空间 + `/4` 时间，细纹理与快速运动有损失。
- **Policy 浪费**：只要 action 时仍 encode 17 帧窗口、联合 denoise video latent。
- **栈复杂**：latent mean/std、VAE patchify 与 DiT patchify 两层 token 化，调试成本高。

### 4.5 视觉编码选型（Policy 场景）

| 需求 | 更合适 |
|------|--------|
| 低延迟、边缘 GPU、只要 action | **SigLIP（π₀.₅）** |
| 当前帧语义 + 语言条件 | **SigLIP** 通常更直接 |
| 未来视频 rollout / 世界模型 | **Wan VAE** |
| 多相机静态观测 | 两者均可；π₀.₅ 多路 ViT 更常见 |
| 长 horizon 运动、时间一致 future | **Wan VAE + DiT** |

**一句话：**

> π₀.₅ ViT = 把「现在看清什么」编码成 **语义 token**，给 LLM **读 1 次**。  
> Wan VAE = 把「现在（和未来）的视频」压成 **可扩散 latent 网格**，给 DiT **反复改**，还能 **decode 成像素**。

---

## 五、Attention 机制对照

### 5.1 π₀：Prefix-LM mask

`make_att_2d_masks` 支持 prefix-lm 模式（`pi0_pytorch.py`）：

```text
[[0 0 0 1 1 1]]: prefix-lm attention.
  前 3 token 互相 full attention；
  后 3 token 因果，且可看前 3 token。
```

- **prefix**（图 + 文）：bidirectional
- **suffix**（action）：causal + 可读 prefix
- **prefix 不可读 suffix**

推理时 `denoise_step` 显式拼接 mask：suffix 对 prefix 全 1，suffix 内部因果。

### 5.2 Cosmos3：MoT 双通路

`Cosmos3AttnProcessor`（`transformer_cosmos3.py`）：

| 通路 | Token | Attention | 可见范围 |
|------|-------|-----------|----------|
| **und** | 文本 | `is_causal=True` | 仅文本自身 |
| **gen** | vision / action / sound | `is_causal=False` | **全部 und K/V + 全部 gen K/V** |

```python
# gen 通路：拼接 und 与 gen 的 K/V
all_k = torch.cat([k_und, k_gen], dim=0)
all_v = torch.cat([v_und, v_gen], dim=0)
full_out = dispatch_attention_fn(q_gen, all_k, all_v, is_causal=False, ...)
```

**等价关系：**

- π₀ suffix cross-attend prefix KV ≈ Cosmos3 gen cross-attend und KV
- Cosmos3 **额外**允许 gen 内部 full self-attn（vision ↔ action 联合建模）

### 5.3 「双通路是否同时算？」

**是。** 每一层、每个 denoising step，und 与 gen **同一次 forward** 完成；「先后」体现在 **mask**（gen 读 und，und 不读 gen），不是两个串行模型。详见 [cosmos3_arch_dit.md §2.3](./cosmos3_arch_dit.md)。

### 5.4 Policy 下 und / gen 各做什么

DROID policy 每个 denoise step、每一层 Attention 的实际分工（`Cosmos3AttnProcessor`）：

```text
und（文本）:
  Q_und, K_und, V_und  ← und 专用投影（to_q/k/v）
  Q_und ──causal──► K_und, V_und          （仅文本内部，从左到右）

gen（vision + action）:
  Q_gen, K_gen, V_gen  ← gen 专用投影（add_q/k/v_proj）
  Q_gen ──full──► [K_und | K_gen], [V_und | V_gen]
                  ↑ 注入文本条件   ↑ vision↔action 互相可见
```

| 说法 | 对错 |
|------|------|
| und 处理文本 Q/K/V，gen 处理 vision + action noise | ✅ |
| 然后「两塔一起做 full attention」 | ❌ **只有 gen 的 Q 做 full attn**；und 是因果 self-attn |
| gen 读完 und 再单独跑第二个模型 | ❌ 同一层、同一次 forward 内并行 |

und 每步仍在网内跑满 36 层，产出 **K_und/V_und 供 gen 读取**——功能上必要；policy 下 caption **静态不变**，K/V **本可 cache 却未做**（见 §6.3、§10）。

---

## 六、推理路径对比

### 6.1 π₀.₅ `sample_actions`

```text
1. embed_prefix(图像, 语言) 
2. paligemma.forward(prefix, use_cache=True)  → past_key_values   【1 次】
3. x_t = noise
4. for t in timesteps:
       denoise_step(x_t, past_key_values)     → 仅 gemma_expert + 读 cache
       x_t += dt * v_t
5. return x_t
```

关键代码（`pi0_pytorch.py`）：

```python
# Prefill：只跑 PaliGemma prefix
_, past_key_values = self.paligemma_with_expert.forward(
    inputs_embeds=[prefix_embs, None], use_cache=True, ...
)

# 每步 denoise：只跑 Action Expert suffix，复用 cache
outputs_embeds, _ = self.paligemma_with_expert.forward(
    past_key_values=past_key_values,
    inputs_embeds=[None, suffix_embs], use_cache=False, ...
)
```

### 6.2 Cosmos3 Policy `__call__`

```text
1. JSON caption + Qwen2 tokenize                          【1 次】
2. 视觉 preprocess + Wan VAE encode + latent 归一化       【1 次】
3. condition mask：z₀=观测，z₁₋₄ + action = 噪声
4. 联合序列打包 + mRoPE                                   【1 次】
5. for step in num_inference_steps:
       transformer.forward(文本 + noisy vision + noisy action)  【每步 16B×36 层】
       UniPC scheduler.step
6. VAE decode → rollout 视频；输出 action chunk
```

Policy 典型：`guidance_scale=1.0`，无 CFG，每步 1 次 transformer forward。

### 6.3 为何 policy 里 und ~8B 权重「像浪费」？

**直觉成立（部署视角）**：DROID policy 只用一段 **静态 JSON caption**，却背着 **~一半 16B 参数（und 通路）**，且 **每个 denoise step 重跑 36 层 und**，无 π₀.₅ 式 prefix cache——相对 π₀.₅「VLM prefill 1 次 + 小 expert 去噪」，显存与延迟都不划算。

但需区分 **训练设计动机** 与 **policy 推理冗余**：

| 维度 | 说明 |
|------|------|
| **不是 policy 专用设计** | Cosmos3-Nano 首要目标是 **统一世界模型**（T2V、forward/inverse dynamics、policy 等同栈）；DROID 是 **post-train**，**不改架构**。 |
| **训练时 und 不「闲」** | 长 JSON caption、因果 LM、CFG 双路文本、与 gen 联合 flow matching——und 在 **text-to-video 主任务** 里承担真实语义建模，不是摆设。 |
| **为何不用小 text encoder** | gen 读 **in-model und K/V**，需同 hidden_size、同 36 层、**联合 co-train** 的表示；外接 T5/CLIP 需额外对齐，且与 Cosmos 视频栈、Qwen 词表一体。 |
| **权重 vs 算力** | und 侧 **~8B 参数全程加载**（浪费感主因）；单步 **FLOPs** 往往 gen 主导（vision token ~1500+ vs 文本 ~几百），但 und **每步仍重算**且无 cache。 |

```text
训练目标（合理）:  16B MoT 统一建模 text + video + action
Policy 推理（冗余）: 静态 text 每 step 过 36 层 und → K/V 给 gen（可 cache 未做）
对比 π₀.₅:        ~2B VLM prefill 1× → ~300M expert × N 步
```

**功能上 und 每步不可删**（gen 需要 und 的 K/V 注入语言条件）；**工程上可优化**（prefix KV cache、action-only 蒸馏等，见 §10）。文档 §9.2 所称「Policy 算力浪费」主要指 **inference 路径**，而非训练时 und 无意义。

---

## 七、任务与数据形态

| | π₀.₅ | Cosmos3 Policy-DROID |
|---|------|----------------------|
| 观测 | 多相机 RGB（SigLIP） | concat 多相机单图 → 17 帧窗口（仅首帧有效） |
| 语言 | 自然语言 +（π₀.₅）离散 state | plain English → JSON caption |
| Action | `[horizon, action_dim]`，DROID fine-tune 常用 32D pad | `[chunk_size, 10]` DROID 10D |
| 未来视频 | **不预测** | **联合**预测 latent rollout（可视化/世界模型） |
| 时间对齐 | action horizon 一步到底 | 5 latent 步 vs 16 action 步（**非 1:1**） |

---

## 八、从 action 生成看世界模型差异

一句话分水岭:**π₀.₅ 是「无模型策略」,Cosmos3 是「基于世界模型的策略」**。

### 8.1 action 是「反应式映射」还是「与未来共生成」

**π₀.₅（reactive policy，model-free）：**

```text
action = f(当前观测, 语言)
```

- Action Expert 把 noisy action chunk 去噪，**条件**是 prefix（vision+lang KV）。
- 模型**只输出 action**，不预测环境会变成什么样。
- 本质学策略函数 `π(a | obs, lang)`——「看到这个就这么动」，不显式理解动作后果。

**Cosmos3（action 与未来视频联合去噪）：**

```text
(未来视频 latent, action) ~ p( · | 当前观测, 语言)
```

- gen 通路里 **vision latent 与 action token 互相 full-attention**，一起 flow matching。
- 模型**同时想象**「未来画面 z₁…z₄」和「达成它的动作 a₁…a₁₆」。
- Action 与预测出的未来轨迹**绑定**——动作是「为了让世界变成想象中的样子」而生成的。

> π₀.₅ 的视觉只是**静态条件**；Cosmos3 的视觉是**被生成的未来**，并反过来约束 action。

### 8.2 是否显式建模「动作 → 后果」

| | π₀.₅ | Cosmos3 |
|---|------|---------|
| 预测未来观测 | **否** | **是**（video rollout latent） |
| 动作后果建模 | 隐式（藏在 VLM 表征） | **显式**（forward dynamics） |
| 给动作、推画面 | 不能 | 能（`forward_dynamics` 模式） |
| 给画面、推动作 | 唯一能做的事 | 能（`inverse_dynamics` 模式） |
| policy 模式 | 唯一形态 | 世界模型的**一个特例** |

Cosmos3 同一套权重支持 **forward / inverse dynamics / policy** 三种模式——它学的是**环境动力学的联合分布**，policy 只是「给定观测下同时采样未来+动作」的一种查询。

### 8.3 为什么 Cosmos3 被称为「世界模型」

**世界模型**定义：学习环境**动力学**，能根据当前状态（和动作）**预测/模拟未来状态**。Cosmos3 满足全部要件：

1. **预测未来观测**：核心训练目标是视频生成/预测（Wan VAE latent 上扩散）——即学「世界如何演化」。
2. **可条件于动作**：`forward_dynamics` 给定 `raw_actions` 预测未来视频 = 标准 learned simulator `s_{t+1}=World(s_t, a_t)`。
3. **可 rollout/想象**：能在 latent 空间「脑补」未来画面，用于规划、数据增广、仿真。
4. **物理/时序先验**：3D 因果 VAE + 时空 latent，天然建模运动与因果，而非单帧语义。

π₀.₅ **不预测未来观测**，没有「世界会怎样」的显式模型，只有「我该怎么动」的策略——故为 **model-free policy**，不是世界模型。

类比：

> - **π₀.₅** = 熟练司机的肌肉记忆：看到路况直接打方向，不在脑中放电影。
> - **Cosmos3** = 先在脑中「预演」接下来几秒画面，再选出能实现该画面的动作——预演能力本身就是世界模型。

### 8.4 对 action 质量 / 代价的影响

| 维度 | π₀.₅ | Cosmos3 |
|------|------|---------|
| Action 一致性 | 靠 VLM 表征，短期反应快 | 与想象未来对齐，长程一致性潜力更好 |
| 可解释性 | 低（黑盒策略） | 高（能看到它「想象」的 rollout） |
| 泛化 / 规划 | 依赖数据覆盖 | 可借世界模型做 imagination/planning |
| 推理代价 | **低**（只 denoise action） | **高**（要联合 denoise video latent） |
| 边缘部署 | 友好 | 重（video 分支对纯 action 是额外开销） |

**代价本质**：Cosmos3 为了「有世界模型」，即使只想要 action，每个 denoising step 也要把**未来视频 latent 一起去噪**——这是它比 π₀.₅ 贵得多的根本原因。

### 8.5 一句话总结

> **π₀.₅**：直接学 `π(a|obs, lang)`，是**反应式策略**。
> **Cosmos3**：学 `p(未来视频, a | obs, lang)`，先**想象世界演化**再从中导出动作，是**世界模型 + 内嵌策略**。

差异根源不在网络结构（都是 prefix-条件 + flow matching），而在**输出目标**：π₀.₅ 只生成动作；Cosmos3 把动作嵌入「对未来的生成」里。

---

## 九、优缺点

### 9.1 π₀.₅

**优点**

- **推理极省**：prefix prefill 一次，denoise N 步主要跑 ~300M Action Expert。
- **结构清晰**：VLM 负责感知理解，expert 负责动作，模块边界明确。
- **专精 manipulation policy**：无视频扩散，延迟/显存更适合真机与边缘设备。
- **π₀.₅ 改进**：state 语言化 + adaRMS 时间注入，条件表达更统一。

**不足**

- **无联合 world model**：不预测未来像素/latent，物理想象依赖 VLM 隐式表征。
- **视觉为 2D token**：无 3D 因果 VAE，长时序视频不在设计目标内。
- **双塔训练**：VLM 与 expert 层对齐 concat attn，工程与权重管理较繁。

### 9.2 Cosmos3 Policy

**优点**

- **单塔多模态**：文本、video latent、action 在同一 DiT **联合 flow matching**，vision ↔ action 可互相约束。
- **生成式 world model**：同一套权重可 rollout 未来视频，利于 sim/visualization/数据增广。
- **VAE latent 时空压缩**：适合长 horizon 视频生成管线（与 Cosmos 视频栈一致）。
- **MoT 双权重**：und/gen 分投影与 MLP，表达力优于简单 concat。

**不足**

- **推理贵**：每 denoising step 全量 ~16B forward，无 π₀ 式 prefix KV cache。
- **栈重**：Wan VAE + 16B DiT，Jetson Thor 等设备部署压力大。
- **Policy 算力浪费**：DROID 控制主路径只需 action，video latent 去噪仍占大量计算；**静态文本 und 通路每 step 全量重算、~8B 权重常驻**（见 §6.3）。
- **时间轴不对齐**：5 latent 步 vs 16 action 步，理解与调试成本更高。

---

## 十、部署与优化启示（Chameleon / Thor）

若要在边缘侧借鉴 π₀.₅ 思路优化 Cosmos3：

| 方向 | 说明 |
|------|------|
| **Prefix KV cache** | **最高优先级、最接近 π₀.₅**：und 的 per-layer K/V（乃至 hidden）对静态 caption **prefill 1 次**，denoise 环 **只更新 gen 段**；条件 latent z₀ 若不变也可一并 cache |
| **拆分推理** | Policy 部署可跳过 video latent 去噪，仅保留 action 分支（需蒸馏或专用 checkpoint） |
| **VAE 与 DiT 分离 profiling** | 见 Chamleon `cosmos3_real_stats.yaml`；VAE 因果分块已相对成熟 |
| **对比 baseline** | 同 DROID 任务上 π₀.₅ 延迟/显存通常显著低于 Cosmos3 Policy |

**与 π₀.₅ 的工程差距（简要）**：差异根源不只在「多 denoise 了 video latent」，还在于 Cosmos3 **未把「理解段算一次、生成段反复算」拆到推理图里**——π₀.₅ 用双 Gemma + `past_key_values` 显式实现；Cosmos3 用 MoT 联合 forward 隐式实现相同 **gen Q 读 prefix/und KV** 的语义，但 **每步仍付 16B 全栈代价**。

---

## 十一、选型建议（简要）

| 场景 | 更倾向 |
|------|--------|
| 真机低延迟控制、边缘 GPU | **π₀.₅** |
| 需要未来视频 rollout / 世界模型 / 与 Cosmos 视频栈统一 | **Cosmos3** |
| 仅要 action、算力有限 | **π₀.₅** 或 Cosmos3 action-only 蒸馏版（非官方） |
| 研究联合 video-action 生成 | **Cosmos3** |

---

## 十二、关键源码索引

| 主题 | π₀.₅ | Cosmos3 |
|------|------|---------|
| 推理入口 | `pi0_pytorch.py` → `sample_actions` | `pipeline_cosmos3_omni.py` → `__call__` |
| Prefill / cache | `sample_actions` L390–400 | 无等价；文本 segment 静态打包 |
| Denoise step | `denoise_step` L422–462 | denoising loop L1620+ |
| Attention mask | `make_att_2d_masks` | `Cosmos3AttnProcessor` |
| 双模型 forward | `gemma_pytorch.py` → `PaliGemmaWithExpertModel` | `Cosmos3OmniTransformer.forward` |
| π₀ vs π₀.₅ 配置 | `pi0_config.py` L28–31 | — |
| 视觉 encode | `pi0_pytorch.py` → `embed_prefix` / `embed_image` | `pipeline_cosmos3_omni.py` → `_encode_video`；[cosmos3_arch_vae.md](./cosmos3_arch_vae.md) |
