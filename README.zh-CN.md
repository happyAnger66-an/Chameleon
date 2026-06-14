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

## 当前状态

- **已可用**：核心抽象与注册表、pi05 参考模型、PyTorch 运行时、VLA orchestrator（真实 flow-matching 去噪环）、配置系统、CLI、workflow runner。
- **NVIDIA 路径（阶段二，已本机验证）**：TensorRT 编译与运行时，含声明式 `TensorRegistry`、位置绑定、持久化设备缓冲、`enqueueV3` 与可选 CUDA Graph；**compile→infer 闭环已打通并数值校验**（TRT FP16 vs PyTorch `cosine=1.0`、`max_abs≈1.25e-3`）；prefill/decode 双 optimization profile；`fmha_d256` 升级为真实 `torch.library` custom op + ONNX symbolic；真实 openpi checkpoint 加载。
- **已知后续 bring-up**：modelopt 量化模块的 ONNX QDQ 导出、真实模型经编排器端到端、Orin/Thor 实测与 CuTe DSL plugin 构建——目前均优雅降级。
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
