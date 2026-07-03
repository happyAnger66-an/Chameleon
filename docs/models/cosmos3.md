# Cosmos3 模型架构与推理流程分析

> 基于三个仓库源码的分析：
> - `cosmos`（文档/cookbooks/部署指南）
> - `transformers`（Reasoner 塔）
> - `diffusers`（Generator 塔 + VAE + Pipeline）

---

## 一、Cosmos3 是什么：一个 Omni 世界基础模型

Cosmos3 把上一代 Cosmos 里分开的 **Predict（世界生成）/ Transfer（受控生成）/ Reason（场景理解）/ Action（动作策略）** 统一成**一个模型**，核心是 **Mixture-of-Transformers (MoT)** 架构：在同一个前向里同时跑「理解（Reasoner）」和「生成（Generator）」两条计算路径。

它由两座"塔"组成：

| 塔 | 职责 | 代码归属 |
|---|---|---|
| **Reasoner 塔** | 多模态理解（图/视频→文本推理），自回归 LM | `transformers` 仓库 |
| **Generator 塔** | 扩散式生成 video/sound/action 隐变量 | `diffusers` 仓库 |

两座塔**共享输入 embedding 和 hidden_size/层数**，但 **Q/K/V/O 投影、MLP、RMSNorm 全部各自一套**（这正是 MoT 而非 MoE 的关键）。统一的 checkpoint（`nvidia/Cosmos3-Nano` / `Cosmos3-Super`）同时含两座塔，各框架按需只加载自己那部分。

---

## 二、三个仓库的分工

### 1. `transformers` —— Reasoner 塔（理解路径）

`transformers/src/transformers/models/cosmos3_omni/modular_cosmos3_omni.py`：

```python
class Cosmos3OmniForConditionalGeneration(Qwen3VLForConditionalGeneration):
    pass
```

- **Reasoner 在架构上完全等同于 Qwen3-VL**，`Cosmos3OmniConfig` 继承 `Qwen3VLConfig`，只是改了 `model_type="cosmos3_omni"` 来匹配 Hub 上的 ckpt。
- 加载时**只取 Reasoner 塔**，把 Generator/Sound/Action 相关权重全部丢弃（`_keys_to_ignore_on_load_unexpected`）：

```python
_COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS = [
    # Generator (image / video diffusion) MoT expert + cross-modal projections
    r"\.add_q_proj\.", r"\.add_k_proj\.", r"\.add_v_proj\.", r"\.to_add_out\.",
    r"moe_gen", r"^proj_out\.", r"^proj_in\.", r"^time_embedder\.",
    # Sound tower / Action tower
    r"^audio_proj_out\.", r"^action_proj_out\.", ...
]
```

> 即 `transformers` / vLLM / NIM 这条链路只做**图像/视频理解（Reasoner）**，对应 cookbook 里的 "reasoner" 用法（`apply_chat_template` → `generate`）。

### 2. `diffusers` —— Generator 塔（生成路径）+ VAE + Pipeline

这是生成能力的核心，三个文件：
- `models/transformers/transformer_cosmos3.py` —— MoT DiT 主干 `Cosmos3OmniTransformer`
- `models/autoencoders/autoencoder_cosmos3_audio.py` —— 声音 tokenizer
- `pipelines/cosmos/pipeline_cosmos3_omni.py` —— `Cosmos3OmniPipeline` 推理编排

### 3. `cosmos` —— 文档/cookbooks/部署指南

无 Python 源码，提供环境搭建、各后端（Cosmos Framework / Diffusers / Transformers / vLLM / vLLM-Omni / NIM）的运行说明和提示词 upsampling 流程。

---

## 三、MoT 双路径架构（`transformer_cosmos3.py` 重点）

### 双路径注意力 `Cosmos3AttnProcessor`

模型把一条**联合序列（joint sequence）**拆成两段：前缀是文本（understanding=und），后面是生成 token（vision + 可选 sound/action = gen）。每层做两条不同的注意力：

```python
# Causal pathway (understanding): und tokens self-attend with causal masking.
causal_out = dispatch_attention_fn(q_und, k_und, v_und, is_causal=True, ...)
# Full pathway (generation): gen tokens cross-attend to all (und + gen) keys/values.
all_k = torch.cat([k_und, k_gen], dim=0)
all_v = torch.cat([v_und, v_gen], dim=0)
full_out = dispatch_attention_fn(q_gen, all_k, all_v, is_causal=False, ...)
```

- **und 路径**：文本 token 之间**因果自注意力**（像普通 LM）。
- **gen 路径**：生成 token **双向**地 cross-attend 到 **und+gen 全部 KV**（既看文本条件，也看彼此）。

两路用**各自独立**的 `to_q/to_k/to_v/to_out`（und）和 `add_q_proj/add_k_proj/add_v_proj/to_add_out`（gen），RMSNorm 也各一套。每个 decoder layer（`Cosmos3VLTextMoTDecoderLayer`）里 MLP 也分两套：`mlp`（und）与 `mlp_moe_gen`（gen），LayerNorm 同理分 `*_moe_gen`：

```python
und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, rotary_emb)
residual_und = und_seq + und_attn_out
residual_gen = gen_seq + gen_attn_out
mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
return residual_und + mlp_out_und, residual_gen + mlp_out_gen
```

### 3D 多模态 RoPE (mRoPE)

`Cosmos3VLTextRotaryEmbedding` 用三轴（时间 T / 高 H / 宽 W）频率交错（`apply_interleaved_mrope`）。文本三轴共用同一递增位置；视频隐变量按 T/H/W 网格分配；声音/动作复用时间轴。还有 **FPS 调制**（`base_fps=24`，按真实 fps 缩放时间位置），使时间步长与帧率解耦。位置矩阵乘法强制在 fp32 下算，避免 bf16 把 >256 的整数位置坍缩。

### 模态投影头（生成塔的输入/输出适配器）

```python
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)      # 文本
self.proj_in  = nn.Linear(patch_latent_dim, hidden_size)          # vision 入
self.proj_out = nn.Linear(hidden_size, patch_latent_dim)          # vision 出
self.time_embedder = TimestepEmbedding(...)                       # 扩散时间步
# action_gen 时: action_proj_in/out 是 DomainAwareLinear（每个具身域一套权重）
# sound_gen 时: audio_proj_in/out + audio_modality_embed
```

- **Vision**：隐变量 patchify（`latent_patch_size=2`）成 token，`proj_in`→hidden，加 timestep embedding（只加在 noisy 帧），过 backbone 后 `proj_out` 还原。
- **Action**：`DomainAwareLinear` 用 `nn.Embedding` 给**每个具身域（domain）**一组独立 weight/bias，按 `domain_id` 选择。
- **Sound**：`audio_proj_in/out` + 一个可学习的模态 embedding。

`forward()` 一次完整前向：把 text/vision/sound/action 全部 embed 进一个零初始化的 `hidden_states` buffer 的各自位置 → 算 mRoPE → 切成 und/gen 两半逐层跑 → 分模态 `proj_out` 解码出**各模态的速度预测（velocity）**。

### VAE

- **视频**：复用 `AutoencoderKLWan`（Wan 2.2 VAE），时间压缩 4×、空间压缩 16×，latent 通道 `latent_channel=48`。pipeline 里 `_encode_video` / decode 时做 mean/std 归一化。
- **声音**：`Cosmos3AVAEAudioTokenizer`（可选组件）。

---

## 四、推理流程（`Cosmos3OmniPipeline.__call__`）

Pipeline 组件：`transformer`（Generator）+ `text_tokenizer`（Qwen2 分词器）+ `vae`（Wan）+ `scheduler`（UniPC）+ 可选 `sound_tokenizer` + `safety_checker`。

注意：**Generator 路径不跑独立文本编码器**，而是直接把 Qwen2 token id 喂进联合 transformer 的 und 段。

整体步骤：

1. **模式选择**（由输入推断）：`num_frames==1`→文生图；给 `image`→图生视频；给 `video`→视频生视频；给 `action`→动作条件生成；否则文生视频。action 时 `num_frames=chunk_size+1`、分辨率由 `resolution_tier` 选 bin。

2. **安全检查（文本）**：`CosmosSafetyChecker.check_text_safety`，不过则 `ValueError`。

3. **Tokenize prompt（`tokenize_prompt`）**：
   - 套 Qwen2 chat template（含 system prompt），末尾追加 `eos` + `<|vision_start|>`（生成起点）。
   - 追加**元数据模板**句子：时长 `"The video is 7.9 seconds long and is of 24 FPS."`、分辨率 `"This video is of 720x1280 resolution."`；负向 prompt 用**反义模板**（"is not ..."）。
   - action 模式则把 prompt 转成训练时用的**结构化 JSON caption**（含视角 framing 句）。
   - 产出 `cond_input_ids` 和 `uncond_input_ids`（CFG 两支）。

4. **预打包静态段**：文本段在去噪步间不变，提前 pack 一次（`_prepare_text_segment`）；再 pack vision/sound/action 段，拼出联合 `position_ids`（mRoPE）和 `sequence_length`。cond/uncond 各一份。

5. **prepare_latents**：按模式采样初始高斯噪声，并构建 **condition_mask**（图生视频锚定 frame 0、视频生视频锚定 `condition_frame_indexes_vision=[0,1]` 等保持"干净"的帧），通过 VAE 编码条件帧。

6. **设置 timesteps**：`UniPCMultistepScheduler`，默认 35 步、`flow_shift=10.0`。sound/action 各 `deepcopy` 一份独立 scheduler（各自维护 step 状态）。

7. **去噪循环**（每步）——这是流匹配（flow-matching velocity）扩散：
   - 把当前 latents 转 bf16，构造每模态的 per-token timestep。
   - **Conditional pass**：调 `transformer(...)` 得 `(preds_vision, preds_sound, preds_action)`。
   - `_mask_velocity_predictions`：用 `1 - condition_mask` 把**条件位置的速度清零**（只在 noisy 位置保留有意义的预测），action 还把 padding 维度清零。
   - **Unconditional pass**：CFG 时再跑一遍 uncond。
   - **CFG 合成**：`velocity = uncond + guidance_scale*(cond - uncond)`（`guidance_scale=1.0` 时跳过 uncond）。
   - 每模态各自 `scheduler.step` 更新 latents。

```python
velocity_vision = uncond_v_vision + guidance_scale * (cond_v_vision - uncond_v_vision)
...
latents = self.scheduler.step(velocity_vision.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)
```

8. **解码 + 后处理**：
   - vision latents 反归一化 → `vae.decode` → `postprocess_video`。
   - sound：`decode_sound`（sound_tokenizer 解码成波形）。
   - action：取 `action_latents` 前 `raw_action_dim` 维作为预测动作序列。
   - **视频安全检查**：guardrail 对解码帧做人脸打码/拒绝。
   - 返回 `Cosmos3OmniPipelineOutput(video, sound, action)`。

---

## 五、一张图概括数据流

```
                            统一 Cosmos3 Checkpoint (Nano / Super)
                          ┌──────────────┴──────────────┐
              Reasoner 塔（=Qwen3-VL）            Generator 塔（MoT DiT）
              [transformers 仓库]                 [diffusers 仓库]
              图/视频+文本 → 文本推理               扩散生成 video/sound/action

Generator 推理（diffusers Pipeline）:
  prompt ──Qwen2 tokenizer──► 文本token(und段) ─┐
  image/video ──Wan VAE.encode──► 条件latent ──┤
  noise ───────────────────────► noisy latent ─┤(gen段: vision/sound/action)
                                                 ▼
                          联合序列 + 3D mRoPE(T/H/W, FPS调制)
                                                 ▼
            ┌──── 每层 MoT 双路径 (×num_hidden_layers) ────┐
            │ und: 因果自注意力 + mlp                       │
            │ gen: 双向cross-attn(und+gen) + mlp_moe_gen   │  ← 投影/MLP/Norm 两套独立
            └──────────────────────────────────────────────┘
                                                 ▼
                   proj_out/audio_proj_out/action_proj_out → 各模态velocity
                                                 ▼
                CFG 合成 → UniPC scheduler.step (循环 N 步)
                                                 ▼
              VAE.decode→视频 / sound_tokenizer→音频 / action 序列 + Guardrail
```

---

## 六、关键结论

1. **一套权重两座塔**：理解（Reasoner=Qwen3-VL，自回归）与生成（Generator=MoT 扩散 DiT）共享 embedding 与骨架尺寸，但注意力/MLP/Norm 投影完全分离 → 这是 **Mixture-of-Transformers** 的本质。
2. **仓库分工**：`transformers` 只加载 Reasoner 做理解；`diffusers` 加载 Generator + Wan VAE + sound tokenizer 做生成；`cosmos` 是文档与部署指南。
3. **生成是流匹配扩散**：联合序列里文本走因果、视觉/声音/动作走双向，靠 3D mRoPE 对齐时空，条件帧通过 mask 保持干净，UniPC 多步去噪，CFG 引导。
4. **Omni**：同一 Generator 前向可并行产出 video + sound + action（多 modality 各有 proj 头与独立 scheduler），action 还按具身域用 `DomainAwareLinear` 选权重。

---

## 七、Chameleon 集成

Cosmos3 已按 pi05 的「架构规格 + ModelAdapter + Orchestrator + Deploy + Evaluate + YAML」范式接入 Chameleon，覆盖 **action 策略生成** 与 **video/world 生成** 两种模式，并打通 reference 冒烟、real 真权重端到端、TRT 分阶段部署三条链路。

### 7.1 Stage 映射

pi05 的 `vit → llm_prefix → action_expert` 对应 cosmos3 的四段式：

```text
vae_encode（Wan VAE 条件编码，跑一次）
  → text_embed（文本理解前缀，跑一次）
    → dit（MoT 联合 transformer，flow-matching 去噪环，每步整模型 forward + CFG ×2）
      → vae_decode（隐变量 → 视频帧；action 模式直接取 dit 输出的 action chunk）
```

关键差异：pi05 每个去噪步只跑一个轻量 cross-attn expert；cosmos3 每步是对联合序列（文本 und + 视觉/声音/动作 gen）的**一次完整 MoT forward**。`dit` 是部署/优化的热点。

- 架构规格：`chameleon/architectures/cosmos3.py`（`COSMOS3_SPEC`，orchestrator=`cosmos3`）。
- 适配器：`chameleon/models/cosmos3/adapter.py`（`Cosmos3Adapter`，reference / diffusers 双路 + graceful fallback）。
- 参考模型：`chameleon/models/cosmos3/reference.py`（小尺寸 MoT + VAE surrogate，CPU 可跑）。

### 7.2 两种生成模式

由 `task.generate.mode` 选择（也可用 `model_overrides.mode`）：

- `mode=action`：对齐 pi05 VLA，输出 action chunk `[B, H, D]`（真实权重经 `CosmosActionCondition`）。
- `mode=video`：text/image/video-to-video，输出视频张量 `[B, T, C, H, W]`（可选联合 sound）。

### 7.3 三条链路

| 链路 | 选择方式 | 编排器 | 说明 |
|------|----------|--------|------|
| reference 冒烟 | `model_overrides.use_reference=true`（默认） | `cosmos3` | CPU 端到端，无外部权重 |
| real 真权重 | `use_reference=false` + `model_id`/`checkpoint` | `cosmos3_real` | diffusers `Cosmos3OmniPipeline` 端到端 |
| TRT 分阶段 | `deploy.backend=cosmos3` + 各 stage build_cfg | `cosmos3_trt` | ONNX 导出 → TRT engine → TRT 去噪环 |

- real 编排器：`chameleon/runtime/orchestrators/cosmos3/real.py`。
- TRT 管线 / 编排器：`chameleon/runtime/cosmos3_trt/{pipeline,orchestrator}.py`。
- 部署导出/构建：`chameleon/deploy/cosmos3/`（loader/vae/dit/sound/export/shapes）+ `chameleon/deploy/cosmos3_diffusers.py`。reference 子图可在 CPU 上离线导出 ONNX。
- 评测：`chameleon/evaluate/cosmos3_runner.py`（policy_runner=`cosmos3`），action 与 ground-truth 比对，video 做 smoke。

### 7.4 配置示例

| 配置 | 用途 |
|------|------|
| `configs/cosmos3_cpu.yaml` | reference 冒烟（CPU） |
| `configs/cosmos3_video_realweights.yaml` | 真实权重视频生成 |
| `configs/cosmos3_action_realweights.yaml` | 真实权重 action 策略 |
| `configs/cosmos3_trt_deploy.yaml` | export → compile → infer 闭环 |

依赖：真实权重路径需 `pip install -e ".[cosmos3]"`（diffusers / transformers / accelerate）。

```bash
# reference 冒烟（video 与 action 各产出张量）
chameleon infer --config configs/cosmos3_cpu.yaml

# TRT 部署闭环（dry-run 仅打印计划；实际 build 需 TensorRT + GPU）
chameleon workflow --config configs/cosmos3_trt_deploy.yaml --dry-run
```
