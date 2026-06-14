# Chameleon

English | [中文](README.zh-CN.md)

A cross-platform **edge VLA** (Vision-Language-Action) toolkit for **quantization,
compilation, inference and custom operators**. It targets NVIDIA (Jetson/DRIVE),
AMD, Intel, generic CPU and Horizon BPU from a single model definition, using a
**unified frontend abstraction + pluggable native compile backends** strategy.

The MVP model is **pi0.5** (`pi05`), the openpi flow-matching VLA policy.

## Why

Edge VLA inference has distinctive characteristics that shape the design:

- **Three-stage graph**: vision encoder (`vit`) -> prefix/KV (`llm_prefix`, run
  once) -> action expert denoise loop (`action_expert`, run `num_steps` times,
  reusing the prefix KV). The denoise loop is the latency hot path.
- **Static, pre-allocatable**: batch=1, fixed action horizon and step count ->
  AOT compilation + CUDA-graph style execution rather than server-side paged /
  continuous batching.
- **Cross-platform differences live in two layers**: operator kernels and the
  compile toolchain (NVIDIA->TensorRT, Intel->OpenVINO, Horizon->BPU SDK,
  AMD/CPU->TVM).

## Architecture

```
core/         PlatformSpec + generic Registry + Artifact/Manifest + contexts
architectures/ ArchitectureSpec / StageSpec (pi05 = vit | llm_prefix | action_expert)
models/       ModelAdapter (pi05 wraps openpi; ships a runnable reference model)
frontend/     GraphCapture (ONNX) -> platform-neutral graph
quantization/ QuantMethod + Calibrator + QuantMetadata contract (modelopt-backed)
compile/      CompilerBackend: tensorrt (first-class) + openvino/tvm/horizon (scaffold)
kernels/      OpSpec + per-vendor KernelImpl (3-stage: stub -> graph node -> backend)
runtime/      RuntimeBackend / Engine + VLAOrchestrator (KV handoff + denoise loop)
workflows/    TaskConfig-driven WorkflowRunner (quantize -> compile -> infer)
config/       pydantic + YAML task schema
cli.py        chameleon CLI
```

Every plugin is discovered through a typed registry populated at import time, so
adding a platform/quant-method/kernel is a self-contained, additive change.

For the full architecture write-up, see [docs/arch.md](docs/arch.md) (Chinese; English version `docs/arch.en.md` planned).

## Install

```bash
pip install -e .
# optional, per target:
pip install -e ".[nvidia]"   # TensorRT + modelopt
pip install -e ".[intel]"    # OpenVINO + NNCF
```

## Usage

```bash
# Inspect what's registered.
chameleon platforms
chameleon architectures
chameleon info

# Run pi05 end-to-end on CPU through the orchestrator (reference model).
chameleon infer --config configs/pi05_cpu.yaml

# Preview / run the full NVIDIA task (degrades gracefully off-device).
chameleon workflow --config configs/pi05_nvidia.yaml --dry-run
chameleon workflow --config configs/pi05_nvidia.yaml

# Closed compile->infer loop: compile each stage to a TensorRT engine, then run
# inference ON those engines (not the PyTorch reference path). NVIDIA + CUDA only.
chameleon workflow --config configs/pi05_nvidia_trt.yaml

# Measure latency.
chameleon profile --config configs/pi05_cpu.yaml --runs 20
```

Programmatic API:

```python
import chameleon
from chameleon.api import run_infer

task = chameleon.TaskConfig.load("configs/pi05_cpu.yaml")
actions = run_infer(task)   # [B, action_horizon, action_dim]
```

## Status

- **Functional**: core abstractions, registries, pi05 reference model, PyTorch
  runtime, the VLA orchestrator (real flow-matching denoise loop), config, CLI,
  workflow runner.
- **NVIDIA path (Phase 2, verified on-box)**: TensorRT compile + runtime with a
  declarative `TensorRegistry`, positional binding, persistent device buffers,
  `enqueueV3` and optional CUDA Graph; the **compile->infer loop is closed and
  numerically validated** (TRT FP16 vs PyTorch `cosine=1.0`, `max_abs≈1.25e-3`).
  Dual prefill/decode optimization profiles; `fmha_d256` as a real
  `torch.library` custom op with ONNX symbolic; real-openpi checkpoint loading.
- **Known bring-up items**: ONNX QDQ export of modelopt-quantized modules, real
  model end-to-end through the orchestrator, on-device Orin/Thor + CuTe DSL
  plugin build. These degrade gracefully today.
- **Stubs (informative `NotImplementedError`)**: OpenVINO / TVM / Horizon
  compile backends.

See the phased roadmap in [docs/arch.md](docs/arch.md) (section 8).

## Documentation

| Document | Language | Description |
|----------|----------|-------------|
| [README.md](README.md) | English | Project overview and quick start (this file) |
| [README.zh-CN.md](README.zh-CN.md) | 中文 | Chinese README |
| [docs/arch.md](docs/arch.md) | 中文 | Architecture design, industry survey, extension guide |

New documentation will be provided in both Chinese and English. Naming convention:

- Chinese: `<name>.md` or `<name>.zh-CN.md`
- English: `<name>.en.md`
