# Cosmos3 DROID Policy 推理流程详解（diffusers 原生）

> 本文只分析 **`Cosmos3-Nano-Policy-DROID` + `mode="policy"` + `domain_name="droid_lerobot"`** 在 diffusers 中的端到端推理路径。接入示例与参数速查见 [cosmos3_policy_droid.md](./cosmos3_policy_droid.md)；视频生成见 [cosmos3_flow.md](./cosmos3_flow.md)。

**代码入口：** `diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py`（`Cosmos3OmniPipeline.__call__`）

---

## 一、Policy 在做什么

**Policy 模式** = 给定 **当前观测（首帧）+ 语言任务**，模型 **联合去噪** 输出：

- **action chunk**：未来 `chunk_size` 步机器人动作（DROID 为 10D）
- **rollout 视频**：与动作对齐的未来帧（可选，用于可视化，非控制主路径）

与另外两种 action 模式的区别：

| 模式 | 输入 | 输出 |
|------|------|------|
| **`policy`** | 首帧观测 + 语言 | **预测 action** + 联合 rollout 视频 |
| `forward_dynamics` | 首帧 + 给定 `raw_actions` | 预测未来视频 |
| `inverse_dynamics` | 完整视频 | 反推动作轨迹 |

Policy **不传入 `raw_actions`**，动作全靠 flow matching 扩散生成。

---

## 二、最小调用形态

```python
result = pipe(
    prompt="Pick up the red block and place it in the bowl.",  # plain English
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="droid_lerobot",      # domain_id=8, raw_action_dim=10
        resolution_tier=480,
        image=concat_view_pil,            # 多相机横向 concat 后的单张图
        view_point="concat_view",
    ),
    fps=15,
    num_inference_steps=30,
    guidance_scale=1.0,                   # policy 通常不用 CFG
    use_system_prompt=False,
    enable_safety_checker=False,
)
actions = result.action[0]   # [16, 10]，模型归一化空间
video   = result.video       # rollout 视频
```

**API 约束（`check_inputs` 强制）：**

- 不能传顶层 `image` / `video` / `height` / `width` / `num_frames`
- checkpoint 必须 `transformer.config.action_gen=True`（Policy-DROID 权重）
- `action.image` 与 `action.video` 二选一；policy 实际只使用 **第一帧**

---

## 三、整体数据流

```mermaid
flowchart TB
    subgraph in["输入"]
        IMG[DROID concat 观测图]
        TXT[plain 任务描述]
    end

    subgraph prep["预处理（各 1 次）"]
        T1[JSON caption + tokenize]
        T2[视觉 padding + VAE encode]
        T3[action 噪声初始化]
        T4[联合序列打包 + mRoPE]
    end

    subgraph loop["去噪 × num_steps"]
        TR[MoT Transformer forward]
        TR --> VA[velocity: vision + action]
        VA --> SCH[UniPC scheduler.step]
    end

    subgraph out["输出"]
        ACT[action[:, :10]]
        VID[VAE decode rollout]
    end

    IMG --> T2
    TXT --> T1
    T1 --> T4
    T2 --> T4
    T3 --> T4
    T4 --> loop --> out
```

---

## 四、时间轴语义（像素 / latent / action）

Policy 最容易混淆的是 **`num_frames = chunk_size + 1` 到底指什么**。结论先说：

| 问题 | 答案 |
|------|------|
| 需要输入 17 张不同的图吗？ | **不需要**。只提供 **1 张** 当前观测（`action.image`）。 |
| 未来帧要提前放进 pipeline 吗？ | **不要**。帧 1～16 是 **扩散生成的输出**，不是用户输入。 |
| `num_frames=17` 是什么？ | 模型内部 rollout 的 **像素时间窗口长度**，与训练格式对齐。 |
| 为何 latent 只有 5 步、却说「frame 1～4」？ | VAE **时间 4× 压缩**：17 像素帧 → 5 latent 步；条件占 1 步，**4 步待生成**。 |

### 4.1 用户侧：输入 vs 输出

以 `chunk_size=16` 为例：

```text
【输入】
  1 张 concat 观测图（action.image）
  1 条 plain 语言指令（prompt）

【输出】
  action: [16, 10]     — 16 步转移（f0→f1, …, f15→f16）
  video:  17 帧像素     — f0≈观测 + f1～f16 为生成的 rollout（可选）
```

**16 步 action 与 17 帧视频的关系：**

```text
视频帧:   f0   f1   f2  ...  f16        （共 17 帧）
          │    └──── 生成的未来 ────┘
          当前观测（唯一真实输入）

action:      a0   a1  ...  a15           （共 16 步）
             f0→f1  f1→f2  ... f15→f16
```

每个 action 表示 **相邻两帧之间的转移**；因此 `chunk_size` 个 action 对应 `chunk_size + 1` 帧视频。

### 4.2 内部 canvas：repeat-pad 不是「把未来当输入」

Pipeline 会把 **1 帧观测** repeat-pad 成 **17 帧像素张量**，仅为 **凑齐与训练一致的 shape**：

```text
vision_tensor 像素维:
  frame 0:     真实观测（preprocess 后）
  frame 1~16:  第 0 帧的复制（占位，不是未来真值）
```

在 `prepare_latents` 初始化时，**非条件 latent 位置直接用纯噪声**，不用 repeat-pad 的像素内容：

```python
vision_condition_frames = [0]   # policy 只锁 latent 第 0 步

latents = mask * x0_encoded + (1 - mask) * Gaussian_noise
#         latent z0 = 观测 encode   latent z1~z4 = 纯噪声 → 去噪后才是未来视频
```

因此：**repeat-pad 是内部 shape 占位；条件信息只锁在 latent z0；z1～z4 从噪声生成，属于输出。**

### 4.3 像素帧 ↔ latent 帧（VAE 时间 4× 压缩）

Wan VAE 的 `scale_factor_temporal=4`，像素时间与 latent 时间 **不是 1:1**：

```text
T_lat = (num_frames - 1) // 4 + 1
      = (17 - 1) // 4 + 1 = 5
```

```text
像素时间:  f0  f1  f2  f3  f4  f5  f6  f7  ...  f16   （17 帧）
              \_________/  \_________/  \___ ... ___/
latent 时间:   z0         z1         z2    z3    z4     （5 步）

z0  ← 锚定当前观测（clean condition，不去噪）
z1~z4 ← 初始纯噪声，多步去噪 → 承载 f1~f16 的未来视频（输出）
       （4 个 latent 步编码/生成整段未来，不是 16 个 latent 步）
```

**为何文档里写「latent frame 1～4」而不是 1～16？**

- **16 帧未来像素** 经 VAE 压缩后由 **4 个 latent 时间步**（z1～z4）表示；
- DiT 去噪的对象是 **latent 张量**，索引 0～4，不是像素索引 0～16；
- VAE decode 之后才会变回 **17 帧像素视频** 输出。

### 4.4 三条时间轴对照（chunk_size=16）

| 轴 | 长度 | 索引 0 | 索引 1～末尾 | 谁提供 |
|----|------|--------|--------------|--------|
| **像素帧** | 17 | 观测（输入） | f1～f16（输出） | 用户 1 张图 + repeat-pad |
| **latent 帧** | 5 | z0 clean（输入） | z1～z4 noisy→去噪（输出） | VAE encode + 扩散 |
| **action 步** | 16 | — | 全部 noisy→去噪（输出） | 独立 action token 扩散 |

Action 在 **action 分支** 上用 16 个 token 联合去噪，与 vision latent **并行**、不在 vision 的 5 个 latent 步上逐帧一一对应。

---

## 五、逐步展开

### Phase A：尺寸与帧数解析

policy 进入 `__call__` 后自动设定：

```python
num_frames = action.chunk_size + 1   # 内部像素 canvas 长度，不是「上传 17 张图」
# height/width 由 resolution_tier + 输入宽高比决定
```

`resolution_tier=480` 时，从 `_ACTION_RESOLUTION_BINS["480"]` 选最接近的 canvas：

| 宽高比 | canvas (H×W) |
|--------|--------------|
| ~1.73（DROID concat 常见） | **832 × 480** |
| 1.0 | 640 × 640 |
| 0.58 | 480 × 832 |
| 1.35 | 736 × 544 |

规则：**只 downscale，不 upscale**；内容区居中 padding（reflect/replicate）。

---

### Phase B：视觉预处理（`_prepare_action_video_conditioning`）

#### B0. 重要：pipeline **不会**做多相机 concat

多路 DROID 相机 **concat 在 pipeline 外完成**，再作为 **一张** `PIL.Image` 传入 `action.image`：

```text
【Pipeline 外 — 你必须做】
  cam_ext | cam_wrist | cam_side  →  np.concatenate(axis=宽度)
  → 一张宽图 PIL  →  action.image
  view_point="concat_view"  →  仅写入 JSON 文本 caption，不在图像代码里 concat

【Pipeline 内 — 本函数】
  单张图 → 归一化 → tier canvas → repeat-pad 17 帧 → 输出 5D 张量
```

调用链（policy + 单图）：

```python
conditioning_clip = [action.image]   # 一张已 concat 的 PIL，不是 [cam1, cam2, cam3]
vision_tensor, image_size, height, width = _prepare_action_video_conditioning(
    conditioning_clip, resolution_tier, num_frames=chunk_size+1, ...
)
```

#### B1. 逐步处理与 shape（例：chunk=16, tier=480, 输入 360×1280 concat 图）

| 步骤 | 操作 | Shape |
|------|------|-------|
| 输入 | `action.image`（单张 PIL） | `(H_src, W_src, 3)` |
| 1. `preprocess_video` | RGB、对齐 16 倍数、归一化 **[-1,1]**，**不 resize 到 tier** | `[1, 3, 1, H', W']` |
| 2. 选 canvas | `classify_height_width_bin(H', W', tier=480)` | `target_h×target_w`（如 480×832） |
| 3. 时间 repeat-pad | 1 帧 → 复制到 `num_frames=17` | `[1, 3, 17, H', W']` |
| 4. bicubic 缩放 | `scale=min(target_w/W', target_h/H', 1.0)`，**只缩小** | `[17, 3, content_h, content_w]` |
| 5. 右/下 padding | 内容贴 **左上角**，右侧/底部 pad | `[1, 3, 17, target_h, target_w]` |
| 输出 | `vision_tensor`, `image_size` | 见下 |

**Step 3 repeat-pad 语义：** 17 帧像素 **内容相同**（全是当前观测的复制），仅为满足 `(chunk_size+1)` 的 tensor shape；**不是**用户提供 17 张未来图。非条件 latent 在 `prepare_latents` 中会用 **纯噪声** 覆盖，见上文 §4.2。

**Step 5 padding 布局（俯视图，H 向下）：**

```text
┌──────────────────────── target_w ────────────────────────┐
│  content 区（缩放后的 concat 图）                         │  ← 贴左上角
│  padding 区（reflect/replicate 填充）                    │
└──────────────────────── target_h ────────────────────────┘
```

**返回值：**

```python
vision_tensor   # [1, 3, 17, target_h, target_w]
image_size      # [target_h, target_w, content_h, content_w]
target_h, target_w  # 用于 JSON caption 的 resolution 字段
```

VAE encode 后会用 `image_size` 裁掉 padding 对应 latent（`_remove_action_video_padding_from_latent`），避免 pad 区进入 DiT。

**本例数值（360×1280 → tier 480）：**

```text
scale = min(832/1280, 480/360, 1.0) = 0.65
content: 234 × 832  →  canvas: 480 × 832
latent content 区: 14 × 52  （÷16 空间压缩）
```

---

### Phase C：文本 → JSON caption → tokens（`tokenize_prompt`）

policy **不用** flat 分辨率/时长模板，而是把 plain prompt 包成训练时的 **JSON 结构化 caption**（`_build_action_json_prompt`）：

```json
{
  "cinematography": {
    "framing": "This video contains concatenated views from multiple camera perspectives."
  },
  "actions": [{"time": "0:00-0:01", "description": "Pick up the red block..."}],
  "duration": "1s",
  "fps": 15.0,
  "resolution": {"H": 480, "W": 832},
  "aspect_ratio": "..."
}
```

- `view_point="concat_view"` → 填入 `cinematography.framing`（与训练对齐，见 `_ACTION_VIEWPOINT_TEMPLATES`）
- 再套 Qwen2 chat template + 特殊 token（`<|vision_start|>` 等）
- 产出 `cond_input_ids` / `uncond_input_ids`（negative 默认空串）

`_prepare_text_segment` 打包一次，去噪环内复用。

---

### Phase D：Latent 初始化（`prepare_latents`，policy 核心）

#### D1. Vision latent（Wan VAE encode + 噪声初始化）

**目标：** 把 Phase B 的像素张量 `vision_tensor [1,3,T,H,W]` 变成去噪环的起点 **`latents [1,C,T_lat,H_lat,W_lat]`**。

**重要：** D1 **只使用 VAE（`AutoencoderKLWan`）**，**不经过 16B MoT DiT**。DiT 在去噪环里对 `latents` 做 patchify + `proj_in` 后才参与（见 [D1.5](#d15-与-mot-dit-的衔接去噪环内)）。

##### D1.0 四步总览

```text
vision_tensor [1,3,17,H,W]   Phase B 输出，值域 [-1,1]
        │
        ▼ ① VAE encode（AutoencoderKLWan / WanEncoder3d）
x0_raw [1, 16, 5, H_lat, W_lat]     # 高斯 posterior 的 mode（μ），确定性
        │
        ▼ ② 数据集级归一化（latents_mean / latents_std）
x0_tokens_vision
        │
        ▼ ③ 裁掉 action padding 区（policy 专用）
x0_tokens_vision [1, 16, 5, H_lat', W_lat']
        │
        ▼ ④ condition mask + 高斯噪声混合
latents = mask * x0 + (1 - mask) * noise
        │
        └── 作为去噪状态变量，进入 Phase F
```

代码入口（`prepare_latents`）：

```python
x0_tokens_vision = self._encode_video(vision_tensor).contiguous().float()
if action_image_size is not None:
    x0_tokens_vision = self._remove_action_video_padding_from_latent(
        x0_tokens_vision, action_image_size)
vision_condition_frames = [0]   # policy
latents = vision_condition_mask * x0_tokens_vision + (1 - mask) * pure_noise
```

##### D1.1 模型结构：`AutoencoderKLWan`（Wan 2.1/2.2 3D VAE）

Pipeline 中 `pipe.vae` 即此类，Cosmos3-Nano / Policy-DROID 与视频生成 **共用同一 VAE**。

```text
AutoencoderKLWan                          # autoencoder_kl_wan.py
├── WanEncoder3d                            # 3D 因果卷积编码器（D1 使用）
│   ├── WanCausalConv3d                   # 时间因果 3D conv
│   ├── WanResidualBlock / WanResidualDownBlock
│   ├── WanResample                       # 2D 或 3D 下采样
│   └── WanMidBlock
├── quant_conv                            # 1×1×1 因果 conv → μ / logσ
├── DiagonalGaussianDistribution
├── post_quant_conv                       # decode 用（D1 不用）
└── WanDecoder3d                          # Phase G decode 用
```

**checkpoint 典型配置（`vae/config.json`）：**

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `z_dim` | **16** | latent 通道数 C |
| `scale_factor_temporal` | **4** | 时间压缩比 |
| `scale_factor_spatial` | **16** | 空间压缩比（H/16, W/16） |
| `temperal_downsample` | `[False, True, True]` | 哪些 stage 做时间下采样 |
| `latents_mean` / `latents_std` | 各 16 个 float | encode/decode 归一化统计量 |

##### D1.2 VAE encode：分块因果编码

17 帧像素 **不是** 一次性过 encoder，而是按 Wan 时间块 **因果** 编码（`AutoencoderKLWan._encode`）：

```python
iter_ = 1 + (num_frame - 1) // 4   # 17 帧 → iter_=5
# i=0: 编码像素帧 0
# i=1..4: 每次编码后续 4 帧，feat_cache 传递跨块因果状态
```

因此 **17 像素帧 → 5 个 latent 时间步**，且时间维有 **因果依赖**（不是每帧独立 encode）。

Pipeline 封装 `_encode_video`：

```449:457:diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py
    def _encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,T,H,W] → normalized latents [B,z_dim,T//4,H//16,W//16]. ..."""
        raw_mu = retrieve_latents(self.vae.encode(x.to(dtype)), sample_mode="argmax")
        ...
        return ((raw_mu - mean) * inv_std).to(in_dtype)
```

要点：

- **`sample_mode="argmax"`**：取 posterior 的 **mode（μ）**，不随机采样，推理确定性。
- **关闭 autocast**：与 Wan 训练一致（注释：`WanVAE was trained with is_amp=False`）。
- 返回值已是 **归一化后** latent，不是 raw μ。

**Shape 示例（chunk=16, canvas 480×832，裁 pad 后 content 234×832）：**

```text
输入 vision_tensor:  [1, 3, 17, 480, 832]
encode 后（裁 pad 前）: [1, 16,  5,  30,  52]    # T_lat=(17-1)//4+1
裁 pad 后 x0:         [1, 16,  5,  14,  52]    # content_h=234 → 234//16=14
```

时间 / 空间压缩公式：

```text
T_lat = (num_frames - 1) // scale_factor_temporal + 1
H_lat = canvas_h // scale_factor_spatial        # 裁 pad 前
W_lat = canvas_w // scale_factor_spatial
H_lat' = content_h // scale_factor_spatial      # 裁 pad 后（policy）
```

##### D1.3 Latent 归一化

Wan VAE 输出的 μ 与 DiT 训练空间不一致，Cosmos3 做 **逐通道仿射变换**：

```text
encode:  z_norm = (μ - latents_mean) / latents_std
decode:  μ = z_norm / inv_std + mean          # Phase G 反变换
```

`latents_mean/std` 在 pipeline `__init__` 时从 `vae.config` 加载。**DiT 始终在归一化 latent 空间去噪**。

##### D1.4 裁 padding：`_remove_action_video_padding_from_latent`

Phase B 在右/下 padding 凑 tier canvas；encode 后按 **有效内容区** 裁 latent，避免 pad 区进入 DiT：

```696:703:diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py
        content_h_latent = max(content_h // self.vae_scale_factor_spatial, 1)
        content_w_latent = max(content_w // self.vae_scale_factor_spatial, 1)
        return latents[:, :, :, :content_h_latent, :content_w_latent]
```

`image_size = [target_h, target_w, content_h, content_w]` 由 Phase B 传入。

##### D1.5 Condition mask + 噪声初始化（policy）

`vision_condition_mask` shape：**`[T_lat, 1, 1]`**（按 **latent 时间步**，不是像素帧索引）

```python
vision_condition_frames = [0]   # policy：仅 z0 为条件
vision_condition_mask[0, 0, 0] = 1.0

latents = mask * x0_tokens_vision + (1 - mask) * pure_noise
```

| latent 步 | mask | 初始化 | 去噪 |
|-----------|------|--------|------|
| **z0** | 1 | `x0_encoded`（观测 encode，clean） | **不更新**（velocity 被 mask 为 0） |
| **z1～z4** | 0 | **纯高斯噪声** | 多步去噪 → 未来视频（**输出**） |

**repeat-pad 的 16 帧像素** 虽会参与 VAE 整段 encode 算 `x0`，但 **z1～z4 初始化时被 mask 丢弃**，改用纯噪声——未来视频 **不是**「把 repeat 像素当输入」。

##### D1.6 与 MoT DiT 的衔接（去噪环内，非 D1）

D1 产出 5D 张量 `latents`；**每一步** denoise 在 `Cosmos3OmniTransformer.forward` 中：

```text
latents [1, C, T_lat, H_lat, W_lat]
  → squeeze → [C, T_lat, H_lat, W_lat]
  → patchify（latent_patch_size=2：2×2 空间块 → 1 token）
  → token dim = 2×2×C（如 C=16 → 64，以 checkpoint 为准）
  → proj_in: Linear(patch_latent_dim → hidden_size 4096)
  → + timestep embedding（仅 noisy 帧 token）
  → 写入 joint sequence 的 vision 位置
  → MoT layers（und 文本 + gen vision/action）
  → proj_out → velocity → scheduler.step 更新 latents
```

**模块分工：**

| 模块 | 何时运行 | 作用 |
|------|----------|------|
| **`AutoencoderKLWan`** | D1 encode 1 次；Phase G decode 1 次 | 像素 ↔ latent |
| **`Cosmos3OmniTransformer`（MoT DiT）** | 去噪环 × `num_inference_steps` | latent 空间预测 velocity、更新 z1～z4 |
| **`UniPCMultistepScheduler`** | 每 denoise 步 | flow matching 积分 |

VAE **不在去噪环内重复 encode**；去噪全程在 **latent 空间**；最后 **decode 一次** 得 17 帧像素 `result.video`。

##### D1.7 数据流一图（policy, chunk=16）

```text
                    AutoencoderKLWan (WanEncoder3d)
[1,3,17,480,832] ──────────────────────────────────► [1,16,5,14,52]  x0
  像素 [-1,1]              encode+norm+crop                归一化 latent
                                                              │
                    mask: z0=x0, z1~z4=noise                │
                                                              ▼
                                                         latents（去噪状态）
                                                              │
                    Cosmos3OmniTransformer × N steps         │
                    patchify(2×2) → proj_in → MoT            │
                                                              ▼
                                                    latents'（去噪后）
                                                              │
                    AutoencoderKLWan (WanDecoder3d)          │
                    denorm + decode ◄─────────────────────────┘
                              │
                              ▼
                    [1,3,17,H,W] → result.video
```

源码：`pipeline_cosmos3_omni.py`（`_encode_video`, `prepare_latents`）；`autoencoder_kl_wan.py`（`AutoencoderKLWan`）；`transformer_cosmos3.py`（`_patchify_and_pack_latents`, `proj_in`）。

#### D2. Action latent

```python
# policy 分支：不从 raw_actions 初始化
x0_tokens_action = zeros(chunk_size, action_dim)

action_condition_frames = []        # policy 下为空
action_condition_mask = 全 0        # chunk_size 个 token 全部 noisy
action_latents = 纯高斯噪声
action_latents[:, raw_action_dim:] = 0   # droid: raw_action_dim=10 之后清零
```

**Domain 选择：**

```python
domain_name="droid_lerobot" → domain_id=8  (_EMBODIMENT_TO_DOMAIN_ID)
raw_action_dim=10                          (_EMBODIMENT_TO_RAW_ACTION_DIM)
# 9D EE pose (平移3 + 6D rotation) + 1D gripper
```

Transformer 内 `DomainAwareLinear`（`transformer_cosmos3.py`）按 `domain_id=8` 选 DROID 专用 `action_proj_in` / `action_proj_out` 权重。

#### D3. 三种 action 模式 condition 对比

| 模式 | vision 条件（latent 时间步） | action 条件 |
|------|------------------------------|-------------|
| **policy** | 仅 z0 | 全无（全 noisy） |
| `forward_dynamics` | z0 | 给定 `raw_actions` 为 clean |
| `inverse_dynamics` | 全部 latent 步 | 全 noisy（反推） |

---

### Phase E：联合序列打包

三路 token 拼进 **一条 joint sequence**：

```text
[ Text (und, causal) | Vision tokens (gen) | Action tokens (gen) ]
```

| 段 | 函数 | policy 下规模（例 chunk=16, 832×480） |
|----|------|--------------------------------------|
| Text | `_prepare_text_segment` | ~100–300 tokens |
| Vision | `_prepare_vision_segment` | T_lat=5, patch 后 ~5×15×27 ≈ **2025** tokens |
| Action | `_prepare_action_segment` | **16** tokens |

每段有独立 **3D mRoPE** 位置编码，再 `torch.cat` 成 `position_ids [3, seq_len]`。

**MoT 结构（`Cosmos3OmniTransformer`）：**

- **und 路径**：文本，causal self-attention
- **gen 路径**：vision + action，full attention，可 attend 到全部 und+gen

---

### Phase F：去噪环（算力主体）

```python
scheduler.set_timesteps(num_inference_steps)  # 通常 ~30
# guidance_scale=1.0 → 每步只 1 次 forward（无 CFG）
```

每步 `t`：

```text
1. vision_tokens = 当前 latents
   action_tokens  = 当前 action_latents

2. transformer.forward(..., action_domain_ids=[8])
   ├─ embed_tokens(text)
   ├─ patchify + proj_in(vision) + timestep embed（仅 noisy 帧）
   ├─ action_proj_in(action, domain_id=8) + timestep embed（全 noisy）
   ├─ MoT layers × N
   ├─ proj_out → velocity_vision
   └─ action_proj_out(domain_id=8) → velocity_action

3. _mask_velocity_predictions
   └─ 条件帧 velocity 置 0（vision frame 0 不更新）

4. scheduler.step(velocity, t, latents)           # 更新 vision latent
   action_scheduler.step(velocity_action, ...)   # 更新 action latent
   action_latents[:, 10:] = 0                    # padding 维保持为 0
```

**30 步 ≈ 30 次完整 16B MoT forward**（policy 无 CFG，比 text2video 的 CFG=6.0 省约一半 DiT 调用）。

---

### Phase G：输出解码

#### Action（控制用）

```python
# policy / inverse_dynamics 模式
action_output = action_latents[:, :raw_action_dim]  # [chunk_size, 10]
result.action = [action_output.detach().cpu()]
```

**10 维含义（`droid_lerobot`）：**

```text
[ tx, ty, tz, rot_6d(6), gripper(1) ]
```

仍在 **模型训练时的归一化空间**；上 Franka 前需按 [cosmos-framework DROID post-train 文档](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md) 做 **反归一化 + 坐标系转换**。diffusers pipeline **不包含** 真机后处理。

#### Video（rollout，可选）

```text
latents 反归一化 → vae.decode → [1, 3, T, H, W] → postprocess → PIL 列表
```

模型「想象」的未来视频，用于 debug/可视化；真机控制环通常只取 `result.action`。

---

## 六、真机控制环

```text
每控制周期:
  1. 采集 DROID 多相机帧 → concat_view 单图
  2. pipe(..., action=CosmosActionCondition(mode="policy", ...))
  3. 得到 [chunk_size, 10] action chunk
  4. 反归一化 → 发送前 k 步到控制器（k ≤ chunk_size）
  5. 重新观测 → 重复（chunk-based receding horizon）
```

与 pi0.5-DROID 类似是 **chunk VLA**，但 action 空间、预处理、权重 **不通用**。

---

## 七、DROID 关键参数对照

| 参数 | 推荐值 | 代码位置 |
|------|--------|----------|
| `domain_name` | `droid_lerobot` | `_EMBODIMENT_TO_DOMAIN_ID` → 8 |
| `view_point` | `concat_view` | `_ACTION_VIEWPOINT_TEMPLATES` |
| `resolution_tier` | `480` | `_ACTION_RESOLUTION_BINS["480"]` |
| `chunk_size` | 16 或 32 | 决定 action 步数 & `num_frames=chunk+1` |
| `fps` | 15 | JSON caption + mRoPE 时间调制 |
| `guidance_scale` | **1.0** | 关闭 CFG |
| `num_inference_steps` | ~30 | 比 stats 短视频用的 4 步多得多 |
| `use_system_prompt` | False | 文档示例推荐 |
| 权重 | **Policy-DROID** | 必须 `action_gen=True` |

---

## 八、张量尺寸速查（chunk=16, tier=480, 宽屏 concat）

详见上文 §4.4。

```text
像素视频 canvas:     [1, 3, 17, 480, 832]     # 内部 canvas；用户只输入 1 张图
VAE latent:          [1, 16,  5,  30,  52]     # T_lat=(17-1)//4+1；裁 pad 后 [1,16,5,14,52]
Vision condition:    latent z0 clean，z1~z4 noisy → 去噪输出
Action latents:      [16, action_dim]          # 16 步全 noisy → 去噪输出
输出 action:         [16, 10]
输出 video:          17 帧像素（f0 + 生成的 f1~f16）
DiT forward 次数:    num_inference_steps × 1   # guidance_scale=1 时
```

Patch 网格（`latent_patch_size=2`）：H_patches=⌈30/2⌉=15，W_patches=⌈52/2⌉=27 → 每 latent 帧 405 vision tokens，共 5 帧 ≈ **2025** tokens。

---

## 九、Pipeline 组件与模块映射

| 阶段 | diffusers 模块 | policy 执行次数 |
|------|----------------|-----------------|
| 视觉预处理 | `_prepare_action_video_conditioning` | 1 |
| VAE encode | `pipe._encode_video` / `vae.encode` | 1 |
| 文本 tokenize | `tokenize_prompt` + `_prepare_text_segment` | 1 |
| Latent 初始化 | `prepare_latents` | 1 |
| 序列打包 | `_prepare_vision/action_segment` | 1（静态部分） |
| **DiT forward** | `Cosmos3OmniTransformer` | **num_steps × 1**（无 CFG） |
| VAE decode | `vae.decode` | 1 |
| Action 输出 | 直接取 `action_latents[:, :10]` | 1 |

Chameleon stats 若按 video 四 stage 拆分，dit 只计单次 cond forward；**真实 policy infer 的 dit 次数 = num_inference_steps**（无 CFG 倍乘）。

---

## 十、常见坑

1. **权重错用**：`Cosmos3-Nano`（video）不能当 policy；必须 **Policy-DROID** checkpoint。
2. **预处理不一致**：未做 concat_view、tier 用 720、或 upscale 与训练不符 → 质量下降。
3. **`result.action` 不能直接发关节**：需 cosmos-framework 反归一化与坐标映射。
4. **`prompt` 不要手写 JSON**：传 plain 英文，pipeline 自动构造 caption。
5. **显存/延迟**：16B MoT × 30 步在 Thor 上仍很重；可降 `chunk_size` / steps，需权衡质量。
6. **顶层 API 混用**：action 模式下勿传 `image`/`height`/`num_frames` 等 video 参数。
7. **误解 `num_frames=17`**：不是输入 17 张图；repeat-pad 是内部占位，f1~f16 / z1~z4 均为 **输出**（见 §四）。
8. **误解 latent「frame 1～4」**：指 **latent 时间步** z1～z4（VAE 4× 时间压缩），不是像素帧 1～4，更不是 16 个 latent 步。

---

## 十一、相关文档与代码

| 资源 | 链接 |
|------|------|
| 接入指南（示例代码 / Chameleon） | [cosmos3_policy_droid.md](./cosmos3_policy_droid.md) |
| 视频生成流程 | [cosmos3_flow.md](./cosmos3_flow.md) |
| Pipeline 源码 | `diffusers/.../pipeline_cosmos3_omni.py` |
| Wan VAE 源码 | `diffusers/.../autoencoders/autoencoder_kl_wan.py` |
| Transformer 源码 | `diffusers/.../transformers/transformer_cosmos3.py` |
| Policy-DROID 权重 | [nvidia/Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) |
| DROID 后处理 | [cosmos-framework action_policy_droid_posttrain.md](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md) |
