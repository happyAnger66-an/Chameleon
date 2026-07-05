# Cosmos3-Nano-Policy-DROID 使用指南（机器人 / VLA）

> 面向机器人开发的 Policy-DROID 接入说明。**逐步推理详解**见 [cosmos3_policy_detail.md](./cosmos3_policy_detail.md)。视频生成流程见 [cosmos3_flow.md](./cosmos3_flow.md)；模型背景见 [cosmos3.md](./cosmos3.md)。

---

## 一、模型是什么

**Cosmos3-Nano-Policy-DROID** 是在 [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano)（16B Mixture-of-Transformers）基础上，用 [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) 数据 post-train 的 **视觉-语言-动作策略模型**。

| 项目 | 说明 |
|------|------|
| 输入 | 自然语言任务 + DROID 平台视觉观测（图像或视频首帧） |
| 输出 | **动作 chunk**（`sample_action.json` / `result.action`）+ 可选未来视频 rollout |
| 权重 | [nvidia/Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)（diffusers 格式） |
| 与 Nano 关系 | 结构相同（Wan VAE + MoT Transformer），action head 针对 DROID 微调 |

**不要**用通用 `Cosmos3-Nano` video 权重做 robot policy；必须加载 **Policy-DROID** checkpoint。

---

## 二、Policy 模式在做什么

Cosmos3 支持三种 action 模式（`CosmosActionCondition.mode`）：

| 模式 | 输入 | 输出 |
|------|------|------|
| **`policy`** | 首帧观测 + 语言指令 | 预测 **action chunk** + 联合 rollout 视频 |
| `forward_dynamics` | 首帧 + 给定 action 序列 | 预测未来视频 |
| `inverse_dynamics` | 完整视频 | 反推动作轨迹 |

Policy-DROID 使用的是 **`policy`**：从当前观测出发，预测接下来若干步机器人动作。

---

## 三、推理数据流（机器人视角）

```text
DROID 相机帧（多视角 concat 后单张图）
        +
自然语言任务（plain English，如 "Pick up the cup..."）
        │
        ▼
CosmosActionCondition(
    mode="policy",
    domain_name="droid_lerobot",
    view_point="concat_view",
    ...
)
        │
        ▼
Cosmos3OmniPipeline
  ① tokenize_prompt（system + 时长/分辨率模板）
  ② prepare_latents：VAE encode 条件帧 → latent + 噪声初始化
  ③ MoT Transformer 去噪环 × num_inference_steps（flow matching）
  ④ 输出 action_latents → 切片为 [chunk_size, raw_action_dim]
        │
        ▼
result.action   [chunk_size, 10]   # 模型归一化空间，需后处理
result.video    可选 rollout 视频
```

**典型控制环**：每周期采集当前帧 → 调用 policy → 执行 chunk 前几步 → 再观测、再推理（chunk-based VLA，类似 pi0.5，但 action 空间与预处理不同）。

---

## 四、DROID 关键参数

diffusers 中 embodiment 通过 `domain_name` 选择 `DomainAwareLinear` 权重：

| 参数 | Policy-DROID 推荐值 | 说明 |
|------|---------------------|------|
| `domain_name` | **`droid_lerobot`** | DROID 专用域（domain_id=8） |
| `view_point` | **`concat_view`** | 多视角横向拼接（DROID 训练设定） |
| `resolution_tier` | **`480`** | 条件画布档位；训练约 640×360，勿用 720p tier 硬凑 |
| `chunk_size` | **`16` 或 `32`** | 动作步数；SFT 用过 32，可先试 16 再对照 model card |
| `fps` | **`15`** | DROID 数据常用帧率 |
| `guidance_scale` | **`1.0`** | policy 通常不用 video 的 6.0 |
| `num_inference_steps` | **`30` 左右** | 比短视频 stats 的 4 步多得多 |
| 条件输入 | `image` 或 `video` | **policy 只使用第一帧** |

**Action 维度（diffusers 接口）：**

- `droid_lerobot` → **10D**：9D 末端位姿（平移 3 + 6D 旋转）+ 1D 夹爪
- 输出在 **模型归一化空间**；上真机前需按 cosmos-framework / Cosmos3-DROID 文档做 **反归一化 + 关节/笛卡尔映射**

**视觉预处理（DROID）：**

- 多路相机 → **横向 concat**（与 `concat_view` 一致）
- 坐标系 to-OpenCV、归一化与训练 pipeline 对齐（见 [cosmos-framework DROID post-train 文档](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md)）

**可用的 `domain_name` 列表（节选）：** `droid_lerobot`, `bridge_orig_lerobot`, `umi`, `av`, `galbot`, … — 见 `pipeline_cosmos3_omni.py` 中 `_EMBODIMENT_TO_DOMAIN_ID`。

---

## 五、三种接入方式

### 5.1 diffusers 直接推理（最快验证）

**依赖：**

```bash
pip install -e ".[cosmos3]"   # Chameleon 仓库
# 或安装含 Cosmos3OmniPipeline 的 diffusers
```

**下载权重：**

```bash
huggingface-cli download nvidia/Cosmos3-Nano-Policy-DROID \
  --local-dir ./Cosmos3-Nano-Policy-DROID
# ModelScope 镜像路径亦可，作为 from_pretrained 的本地目录
```

**示例代码：**

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition, UniPCMultistepScheduler
from diffusers.utils import load_image

pipe = Cosmos3OmniPipeline.from_pretrained(
    "./Cosmos3-Nano-Policy-DROID",
    torch_dtype=torch.bfloat16,
    enable_safety_checker=False,  # 本地调试；上线需 cosmos_guardrail
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)
pipe.to("cuda")

obs = load_image("/path/to/droid_concat_view.jpg")

result = pipe(
    prompt="Pick up the red block and place it in the bowl.",
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="droid_lerobot",
        resolution_tier=480,
        image=obs,
        view_point="concat_view",
    ),
    fps=15,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
)

actions = result.action[0]  # [chunk_size, 10]
with open("sample_action.json", "w") as f:
    json.dump(actions.tolist(), f)
```

**注意：**

- `prompt` 写 **plain 任务描述**，不要手写 JSON caption；pipeline 会自动构造训练用结构化 caption。
- `height` / `width` / `num_frames` 在 action 模式下由 `CosmosActionCondition` 决定（`chunk_size + 1` 条件帧），不要与 video 模式混用。

**CLI  smoke test（diffusers 仓库）：**

```bash
python examples/cosmos3/inference_cosmos3.py \
  --model nano \
  --prompt "Pick up the red block and place it in the bowl." \
  --vision-path /path/to/droid_frame.jpg \
  --action-mode policy \
  --action-chunk-size 16 \
  --domain-name droid_lerobot \
  --view-point concat_view \
  --resolution-tier 480 --fps 15 \
  --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 \
  --output results/cosmos3_policy_droid
```

（需将 `--model` 指向 Policy-DROID 本地路径，或改脚本中的 `HF_REPOS`。）

---

### 5.2 Chameleon 接入

**示例配置**（可存为 `configs/cosmos3_action_droid.yaml`）：

```yaml
architecture: cosmos3
model: cosmos3
platform: nvidia_thor   # Jetson Thor / Orin；桌面 GPU 用 nvidia_ada

actions:
  - infer

model_overrides:
  use_reference: false
  checkpoint: /path/to/Cosmos3-Nano-Policy-DROID
  precision: bfloat16
  mode: action
  enable_safety_checker: false

generate:
  mode: action
  prompt: "Pick up the cup and place it on the plate."
  num_inference_steps: 30
  guidance_scale: 1.0
  output_type: pt
  action:
    mode: policy
    chunk_size: 16
    domain_name: droid_lerobot
    resolution_tier: 480
    view_point: concat_view
    image: /path/to/current_frame.jpg   # 或 video: /path/to/clip.mp4

infer:
  batch_size: 1
  torch_device: cuda
```

**运行：**

```bash
chameleon infer --config configs/cosmos3_action_droid.yaml
```

**Chameleon 内部路径：**

- 适配器：`chameleon/models/cosmos3/adapter.py` → 加载 `Cosmos3OmniPipeline`
- 编排器：`chameleon/runtime/orchestrators/cosmos3/real.py` → `Cosmos3RealOrchestrator.infer()`
- 构造 `CosmosActionCondition` 并调用 `pipe(...)`；主返回值是 action 张量，完整输出在 `orchestrator.last_output`

**Chameleon 现状（机器人开发者需知）：**

| 能力 | 状态 |
|------|------|
| 真实 Policy-DROID 权重端到端 infer | 已支持 |
| 相机采集 + DROID concat 预处理 | **需自研**（evaluate runner 仍用 synthetic obs） |
| action 反归一化 → 机器人 SDK | **需对照 cosmos-framework** |
| TRT 分 stage 部署 action | bring-up 中（参考 `cosmos3_trt_deploy.yaml`） |
| stats 算力评估（action real） | 偏 video 路径，policy 建议用 infer + profiler |

---

### 5.3 cosmos-framework Policy Server（仿真 / 真机闭环）

NVIDIA 官方推荐的 **Server–Client** 部署：Server 跑 Policy-DROID，Client 驱动仿真或真机。

1. 克隆 [cosmos-framework](https://github.com/NVIDIA/cosmos-framework)
2. 启动 policy server：

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab --port 8000
```

3. 仿真客户端（[RoboLab](https://github.com/NVlabs/RoboLab)）：

```bash
python policies/cosmos3/run.py --task BananaInBowlTask
```

文档：[cosmos/cookbooks/.../run_policy_with_cosmos_framework.md](https://github.com/nvidia/Cosmos/blob/main/cookbooks/cosmos3/generator/action/run_policy_with_cosmos_framework.md)

真机：自写 Client — 采图 → HTTP/gRPC 请求 → 收 action chunk → 后处理 → 下发控制。

---

## 六、与 Chameleon Stage 的对应关系

Chameleon 将 Cosmos3 拆为四段（action 与 video 共用同一 MoT，但 `generate.mode=action`）：

```text
vae_encode   → Wan VAE encode 条件帧（policy：chunk_size+1 帧 canvas）
text_embed   → transformer.embed_tokens（文本理解前缀）
dit          → Cosmos3OmniTransformer 去噪环 × num_steps [×2 CFG 若 guidance>1]
vae_decode   → Wan VAE decode（policy 仍可能生成 rollout video）
```

**与 video 推理的差异：**

| 维度 | video (t2v) | policy (DROID) |
|------|---------------|----------------|
| 条件 | 零 canvas / 可选 image | **当前观测首帧**（concat_view） |
| `guidance_scale` | 常 6.0 | **常 1.0** |
| `num_inference_steps` | 35（全尺寸） | **~30** |
| 主输出 | 视频 | **action chunk** |
| stats CFG | dit 只计 cond 1 次 | 真实 infer 若 CFG>1 则 dit ×2 |

---

## 七、与 pi0.5 / openpi 的对比

| | pi0.5 (openpi) | Cosmos3-Nano-Policy-DROID |
|--|----------------|---------------------------|
| 范式 | Flow matching VLA | MoT 扩散 world model + action modality |
| 输入 | 多相机 + proprio state + 语言 | concat 图像 + 语言 |
| 输出 | `[horizon, action_dim]` 关节空间 | `[chunk_size, 10]` 归一化 EE+gripper |
| 典型延迟 | 较低（轻量 action expert 环） | 高（16B 全模型 denoise） |
| Chameleon 成熟度 | RTX 4070 真权重 E2E 已验证 | infer 已通；TRT / 真机 I/O 在 bring-up |
| checkpoint | openpi `.safetensors` | diffusers Policy-DROID 目录 |

二者 **不能共用** checkpoint、norm_stats 或数据 transform；DROID 上需单独做 Cosmos3 I/O 对齐。

---

## 八、部署与算力提示

- **精度**：官方仅充分测试 **bf16**；Thor / Orin 上注意显存（16B + 激活，参考 video stats **~40GB+ 峰值**量级）。
- **延迟优化**：减小 `chunk_size`、`num_inference_steps`；Thor 上先用小分辨率 `resolution_tier=480`；长期考虑量化 / TRT 分 stage。
- **安全**：生产环境安装 `cosmos_guardrail` 并设 `enable_safety_checker=true`（OpenMDW 许可要求）。

**Thor 快速摸底（仅 video stats，非 policy）：**

```bash
chameleon stats --config configs/cosmos3_stats_thor.yaml
```

Policy 算力请用 `actions: [infer]` + 实测 wall time，或 datacenter GPU 上 `--measured`。

---

## 九、推荐落地步骤

1. 下载 **Cosmos3-Nano-Policy-DROID**（HF 或 ModelScope）。
2. 用 **diffusers** 跑通 `policy` + `droid_lerobot` + `concat_view`，确认 `sample_action.json` 有输出。
3. 对照 **cosmos-framework** DROID 后处理，将 normalized action 转为机器人命令。
4. 在 Chameleon 写 `configs/cosmos3_action_droid.yaml`，`chameleon infer` 与 diffusers 结果对齐。
5. 仿真闭环：**cosmos-framework policy server + RoboLab**。
6. 真机：自写 Client，按训练时 concat / 归一化协议预处理相机数据。

---

## 十、参考链接

| 资源 | URL |
|------|-----|
| Policy-DROID 权重 | https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID |
| Cosmos3-DROID 数据集 | https://huggingface.co/datasets/nvidia/Cosmos3-DROID |
| DROID SFT / 后处理 | https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md |
| Policy Server + RoboLab | https://github.com/nvidia/Cosmos/blob/main/cookbooks/cosmos3/generator/action/run_policy_with_cosmos_framework.md |
| diffusers Cosmos3 API | https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3 |
| Chameleon action 配置示例 | `configs/cosmos3_action_realweights.yaml` |
| Chameleon 视频推理流程 | [cosmos3_flow.md](./cosmos3_flow.md) |
