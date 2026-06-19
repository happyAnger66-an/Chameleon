# Chameleon

[English](README.md) | 中文

跨平台**端侧 VLA**（Vision-Language-Action）工具包，支持**量化、编译、推理与自定义算子**。基于单一模型定义，面向 NVIDIA（Jetson/DRIVE）、AMD、Intel、通用 CPU 与地平线 BPU 等多平台部署，采用**统一前端抽象 + 可插拔原生编译后端**策略。

MVP 模型为 **pi0.5**（`pi05`），即 openpi 的 flow-matching VLA 策略。

## 设计动机

端侧 VLA 推理具有以下特点，直接决定了框架设计：

- **三段式计算图**：视觉编码器（`vit`）→ 前缀/KV（`llm_prefix`，只算一次）→ 动作专家去噪环（`action_expert`，迭代 `num_steps` 次并复用 prefix KV）。去噪环是延迟热点路径。
- **静态、可预分配**：batch=1、固定 action horizon 与去噪步数 → 适合 AOT 编译 + CUDA Graph 式执行，而非服务端的 paged / continuous batching。
- **跨平台差异集中在两层**：算子 kernel，以及编译工具链（NVIDIA→TensorRT、Intel→OpenVINO、地平线→BPU SDK、AMD/CPU→TVM）。

## 架构概览

```
core/         PlatformSpec + 泛型 Registry + Artifact/Manifest + 上下文
architectures/ ArchitectureSpec / StageSpec（pi05 = vit | llm_prefix | action_expert）
models/       ModelAdapter（pi05 包装 openpi；内置可运行参考模型）
frontend/     GraphCapture（ONNX）→ 平台中性计算图
quantization/ QuantMethod + Calibrator + QuantMetadata 契约（封装 modelopt）
compile/      CompilerBackend：tensorrt（首发）+ openvino/tvm/horizon（脚手架）
kernels/      OpSpec + 各平台 KernelImpl（三段式：stub → 图节点 → 后端）
runtime/      RuntimeBackend / Engine + VLAOrchestrator（KV 传递 + 去噪环）
workflows/    由 TaskConfig 驱动的 WorkflowRunner（quantize → compile → infer）
config/       pydantic + YAML 任务配置
cli.py        chameleon 命令行入口
```

所有插件通过 import 时注册到类型化 Registry，`import chameleon` 即可完成发现。新增平台 / 量化方法 / 算子均为**自包含、增量式**改动。

更完整的架构说明见 [docs/arch.md](docs/arch.md)（中文，后续将提供英文版 `docs/arch.en.md`）。

## 安装

```bash
pip install -e .
# 按目标平台可选安装：
pip install -e ".[nvidia]"   # TensorRT + modelopt
pip install -e ".[intel]"    # OpenVINO + NNCF
```

## 使用

```bash
# 查看已注册组件
chameleon platforms
chameleon architectures
chameleon info

# 在 CPU 上经 orchestrator 端到端运行 pi05（参考模型）
chameleon infer --config configs/pi05_cpu.yaml

# 预览 / 运行完整 NVIDIA 任务（非目标设备上会优雅降级）
chameleon workflow --config configs/pi05_nvidia.yaml --dry-run
chameleon workflow --config configs/pi05_nvidia.yaml

# compile→infer 闭环：将各 stage 编译为 TensorRT engine，并真正在这些 engine 上推理
#（非 PyTorch 参考路径）。仅 NVIDIA + CUDA。
chameleon workflow --config configs/pi05_nvidia_trt.yaml

# 真实 openpi 权重 + RTX 4070：PyTorch 端到端推理（见下文配置说明）
PYTHONPATH=. /path/to/openpi/.venv/bin/python -m chameleon.cli infer \
  --config configs/pi05_rtx4070_realweights.yaml

# 测量推理延迟
chameleon profile --config configs/pi05_cpu.yaml --runs 20
```

编程接口：

```python
import chameleon
from chameleon.api import run_infer

task = chameleon.TaskConfig.load("configs/pi05_cpu.yaml")
actions = run_infer(task)   # [B, action_horizon, action_dim]
```

## 配置文件说明

Chameleon 任务由 YAML 描述，经 `TaskConfig.load()` 加载并由 pydantic 校验。所有示例配置位于 `configs/` 目录。

| 配置文件 | 用途 |
|----------|------|
| `pi05_cpu.yaml` | CPU 参考模型，仅 `infer`，冒烟测试 |
| `pi05_nvidia.yaml` | NVIDIA 全流程（quantize → compile → infer），参考模型 + PyTorch 运行时 |
| `pi05_nvidia_trt.yaml` | 参考模型 compile→infer 闭环，推理跑在 TRT engine 上 |
| `pi05_realweights.yaml` | 真实 openpi 权重，默认仅 `quantize` |
| **`pi05_rtx4070_realweights.yaml`** | **RTX 4070 + 真实 openpi 权重，PyTorch 端到端推理** |

完整 schema 定义见 `chameleon/config/schema.py`。

### `configs/pi05_rtx4070_realweights.yaml`（RTX 4070 + 真实权重）

该配置用于在 **NVIDIA RTX 4070（Ada Lovelace, sm_89）** 上加载真实 openpi pi05 checkpoint，并通过 `Pi05RealOrchestrator` 调用 `PI0Pytorch.sample_actions` 做 **PyTorch 端到端推理**（不拆 stage、不编译 TRT engine）。

#### 前置条件

1. **Chameleon**：`pip install -e ".[nvidia]"`（或 `PYTHONPATH` 指向仓库根目录）。
2. **openpi 运行环境**：需能 `import openpi`，且已安装 `transformers==4.53.2` 与 `transformers_replace`（`PI0Pytorch` 初始化会校验）。
3. **Checkpoint**：openpi 导出的 PyTorch 权重，通常为目录下的 `model.safetensors` + `config.json`（本仓库示例路径：`models/openpi/pytorch/`）。
4. **GPU**：CUDA 可用；12GB 显存建议 `precision: bfloat16`（选择性 bf16，约 7–8GB 显存）。

#### 运行示例

若 Chameleon 安装在 openpi 的 uv 虚拟环境中，可直接：

```bash
chameleon infer --config configs/pi05_rtx4070_realweights.yaml
```

若使用独立 openpi 环境（未 `pip install` chameleon），需注入 `PYTHONPATH`：

```bash
cd /path/to/Chamleon
PYTHONPATH=. /path/to/openpi/.venv/bin/python -m chameleon.cli infer \
  --config configs/pi05_rtx4070_realweights.yaml
```

输出动作为 `[batch_size, action_horizon, action_dim]`，例如 `(1, 10, 32)`。

#### 字段说明

**顶层任务**

| 字段 | 示例值 | 含义与配置要点 |
|------|--------|----------------|
| `architecture` | `pi05` | 架构注册名，决定 stage 划分与默认编排器。pi05 固定为 `pi05`。 |
| `model` | `pi05` | 模型适配器注册名，对应 `Pi05Adapter`。 |
| `platform` | `nvidia_ada` | 部署平台；4070 使用 `nvidia_ada`（`kernel_tag=sm_89`，支持 fp8）。别名：`rtx4070`、`ada`。 |
| `output_dir` | `output/pi05_rtx4070_realweights` | 工作目录：写入 `chameleon_manifest.json` 及 quantize/compile 产物。 |
| `actions` | `[infer]` | 按顺序执行的动作：`quantize` / `compile` / `infer`。真实权重端到端推理至少包含 `infer`；仅量化改为 `[quantize]` 或 `[quantize, infer]`。 |

**`model_overrides`**（传入 `Pi05Config`，覆盖适配器默认项）

| 字段 | 示例值 | 含义与配置要点 |
|------|--------|----------------|
| `use_reference` | `false` | **`false`**：加载真实 openpi 权重并启用 `Pi05RealOrchestrator`；**`true`**：使用内置轻量参考模型（无需 checkpoint）。 |
| `checkpoint` | `.../model.safetensors` | 权重路径，支持 `.safetensors`（推荐）、`.pt`、`.pth`。须与 `use_reference: false` 同时使用。 |
| `action_dim` | `32` | 动作维度，**必须与 checkpoint 的 `config.json` 一致**。 |
| `action_horizon` | `10` | 动作序列长度（flow-matching 输出步长），**必须与 checkpoint 一致**（常见为 `10`，不是参考模型默认的 `50`）。 |
| `paligemma_variant` | `gemma_2b` | PaliGemma 规模，与训练/checkpoint 一致。 |
| `action_expert_variant` | `gemma_300m` | 动作专家 Gemma 规模，与训练/checkpoint 一致。 |
| `precision` | `bfloat16` | 真实模型精度：`bfloat16` 为 openpi 同款**选择性 bf16**（部分 layernorm/embedding 保持 fp32），12GB 显存可加载 ~3.6B 参数；`float32` 约需 14GB+ 显存。 |

**`infer`**（推理阶段）

| 字段 | 示例值 | 含义与配置要点 |
|------|--------|----------------|
| `batch_size` | `1` | 批大小；端侧 VLA 固定为 1。 |
| `num_steps` | `10` | flow-matching **去噪迭代次数**，传给 `sample_actions(..., num_steps=...)`。可与 `action_horizon` 不同。 |
| `torch_device` | `cuda` | PyTorch 设备；4070 上设为 `cuda`。无 GPU 时 API 会降级到 CPU 并告警。 |
| `use_compiled_engines` | `false` | **`true`**：infer 使用 compile 产出的 TRT engine（参考模型路径）。**真实权重当前须保持 `false`**（真实子模块 ONNX/QDQ 导出尚未打通）。 |
| `cuda_graph` | `false` | TRT 运行时是否捕获 CUDA Graph；仅 `use_compiled_engines: true` 且静态 shape 时有意义。 |

**`quantize`**（可选；仅当 `actions` 含 `quantize` 时执行）

| 字段 | 示例值 | 含义与配置要点 |
|------|--------|----------------|
| `stage` | `action_expert` | 要量化的 stage 名（`vit` / `llm_prefix` / `action_expert`）。 |
| `method` | `fp8` | 量化方法注册名（`int8` / `fp8` / `int4_awq` 等，见 `chameleon info`）。 |
| `weight_dtype` | `fp8` | 权重量化 dtype 声明。 |
| `kv_cache_dtype` | `fp8` | KV cache dtype（可选）；Ada 支持 FP8 E4M3。 |
| `activation_dtype` | — | 激活 dtype（可选）；未设则由 method 默认。 |

> **注意**：配置文件中列出的 `quantize` 块在 `actions: [infer]` 时**不会执行**；需要量化时把 `actions` 改为包含 `quantize`。

#### 常见改法

```yaml
# 仅改 checkpoint 路径（其余与 checkpoint config.json 对齐）
model_overrides:
  use_reference: false
  checkpoint: /your/path/to/model.safetensors
  action_dim: 32
  action_horizon: 10      # 以 checkpoint config.json 为准
  precision: bfloat16

# 先量化 action_expert，再推理
actions:
  - quantize
  - infer

# 在 CPU 上调试加载逻辑（极慢，仅验证能否 import openpi）
platform: generic_cpu
infer:
  torch_device: cpu
```

#### 当前能力边界

| 能力 | 本配置 |
|------|--------|
| 真实 checkpoint 加载 | 支持 |
| PyTorch 端到端 infer（`sample_actions`） | 支持 |
| 按 stage 量化（`quantize`） | 支持（需 modelopt） |
| compile → TRT engine → infer | **不支持**（保持 `use_compiled_engines: false`） |
| 与 openpi Policy 相同的 norm_stats / 数据 transforms | **未接入**（当前为合成 observation，用于验证计算图） |

## 当前状态

- **已可用**：核心抽象与注册表、pi05 参考模型、PyTorch 运行时、VLA orchestrator（真实 flow-matching 去噪环）、配置系统、CLI、workflow runner。
- **NVIDIA 路径（阶段二，已本机验证）**：TensorRT 编译与运行时，含声明式 `TensorRegistry`、位置绑定、持久化设备缓冲、`enqueueV3` 与可选 CUDA Graph；**compile→infer 闭环已打通并数值校验**（TRT FP16 vs PyTorch `cosine=1.0`、`max_abs≈1.25e-3`）；prefill/decode 双 optimization profile；`fmha_d256` 升级为真实 `torch.library` custom op + ONNX symbolic；真实 openpi checkpoint 加载；**RTX 4070 上真实权重 PyTorch 端到端 infer 已验证**（`configs/pi05_rtx4070_realweights.yaml`）。
- **已知后续 bring-up**：modelopt 量化模块的 ONNX QDQ 导出、真实模型 compile→TRT engine、Orin/Thor 实测与 CuTe DSL plugin 构建——目前均优雅降级。
- **占位（`NotImplementedError` + 集成说明）**：OpenVINO / TVM / 地平线编译后端。

分阶段路线图见 [docs/arch.md](docs/arch.md) 第 8 节。

## 文档

| 文档 | 语言 | 说明 |
|------|------|------|
| [README.md](README.md) | English | 项目简介与快速上手 |
| [README.zh-CN.md](README.zh-CN.md) | 中文 | 本文档 |
| [docs/arch.md](docs/arch.md) | 中文 | 架构设计、业界调研与扩展指南 |

后续新增文档将同时提供中/英文版本，文件名约定：

- 中文：`<name>.md` 或 `<name>.zh-CN.md`
- 英文：`<name>.en.md`
