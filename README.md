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

# Real openpi weights on RTX 4070: PyTorch end-to-end infer (see config section below).
PYTHONPATH=. /path/to/openpi/.venv/bin/python -m chameleon.cli infer \
  --config configs/pi05_rtx4070_realweights.yaml

# Measure latency.
chameleon profile --config configs/pi05_cpu.yaml --runs 20

# Estimate compute (MACs/FLOPs) and memory traffic for one full inference.
chameleon stats --config configs/pi05_cpu.yaml
chameleon stats --config configs/pi05_libero_trt_deploy.yaml --dry-run
PYTHONPATH=. models/openpi/.venv/bin/python -m chameleon.cli stats \
  --config configs/pi05_libero_trt_deploy.yaml --format json --output output/stats.json
```

Programmatic API:

```python
import chameleon
from chameleon.api import run_infer

task = chameleon.TaskConfig.load("configs/pi05_cpu.yaml")
actions = run_infer(task)   # [B, action_horizon, action_dim]
```

## Configuration files

Tasks are described in YAML, loaded via `TaskConfig.load()` and validated by pydantic. Example configs live under `configs/`.

| File | Purpose |
|------|---------|
| `pi05_cpu.yaml` | CPU reference model, infer-only smoke test |
| `pi05_nvidia.yaml` | Full NVIDIA pipeline (quantize → compile → infer), reference model + PyTorch runtime |
| `pi05_nvidia_trt.yaml` | Reference model compile→infer loop on TRT engines |
| `pi05_realweights.yaml` | Real openpi weights, quantize-only by default |
| **`pi05_rtx4070_realweights.yaml`** | **RTX 4070 + real openpi weights, PyTorch end-to-end infer** |

Full schema: `chameleon/config/schema.py`.

### `configs/pi05_rtx4070_realweights.yaml` (RTX 4070 + real weights)

Loads a real openpi pi05 checkpoint on **NVIDIA RTX 4070 (Ada, sm_89)** and runs **PyTorch end-to-end inference** via `Pi05RealOrchestrator` → `PI0Pytorch.sample_actions` (whole model, no stage split, no TRT compile).

#### Prerequisites

1. **Chameleon**: `pip install -e ".[nvidia]"` (or set `PYTHONPATH` to the repo root).
2. **openpi runtime**: `import openpi` must work; `transformers==4.53.2` + `transformers_replace` installed (`PI0Pytorch` checks this at init).
3. **Checkpoint**: PyTorch export from openpi — typically `model.safetensors` + `config.json` (example in this repo: `models/openpi/pytorch/`).
4. **GPU**: CUDA required; for 12GB VRAM use `precision: bfloat16` (selective bf16, ~7–8GB allocated).

#### Run

If Chameleon is installed in the openpi venv:

```bash
chameleon infer --config configs/pi05_rtx4070_realweights.yaml
```

Otherwise inject `PYTHONPATH`:

```bash
cd /path/to/Chamleon
PYTHONPATH=. /path/to/openpi/.venv/bin/python -m chameleon.cli infer \
  --config configs/pi05_rtx4070_realweights.yaml
```

Output shape: `[batch_size, action_horizon, action_dim]`, e.g. `(1, 10, 32)`.

#### Field reference

**Top-level**

| Field | Example | Meaning |
|-------|---------|---------|
| `architecture` | `pi05` | Architecture registry key (stage layout + default orchestrator). |
| `model` | `pi05` | Model adapter key (`Pi05Adapter`). |
| `platform` | `nvidia_ada` | Deployment target; RTX 4070 → `nvidia_ada` (`kernel_tag=sm_89`, fp8). Aliases: `rtx4070`, `ada`. |
| `output_dir` | `output/pi05_rtx4070_realweights` | Work dir for manifest and artifacts. |
| `actions` | `[infer]` | Ordered steps: `quantize` / `compile` / `infer`. Real-weight E2E needs at least `infer`. |

**`model_overrides`** (passed to `Pi05Config`)

| Field | Example | Meaning |
|-------|---------|---------|
| `use_reference` | `false` | **`false`**: load real openpi weights + `Pi05RealOrchestrator`. **`true`**: lightweight reference model. |
| `checkpoint` | `.../model.safetensors` | Weight path (`.safetensors` recommended, or `.pt`/`.pth`). Required when `use_reference: false`. |
| `action_dim` | `32` | Must match checkpoint `config.json`. |
| `action_horizon` | `10` | Action sequence length; **must match checkpoint** (often `10`, not reference default `50`). |
| `paligemma_variant` | `gemma_2b` | PaliGemma size; match training/checkpoint. |
| `action_expert_variant` | `gemma_300m` | Action expert size; match training/checkpoint. |
| `precision` | `bfloat16` | Real model precision: selective bf16 (openpi-style); ~7–8GB on 4070. `float32` needs ~14GB+ VRAM. |

**`infer`**

| Field | Example | Meaning |
|-------|---------|---------|
| `batch_size` | `1` | Batch size (edge VLA: always 1). |
| `num_steps` | `10` | Flow-matching denoise iterations → `sample_actions(..., num_steps=...)`. |
| `torch_device` | `cuda` | PyTorch device; use `cuda` on 4070. |
| `use_compiled_engines` | `false` | **`true`**: infer on TRT engines (reference path). Real weights: keep **`false`** (ONNX/QDQ export not ready). |
| `cuda_graph` | `false` | TRT CUDA Graph capture; only relevant with compiled engines. |

**`quantize`** (optional; runs only if `actions` includes `quantize`)

| Field | Example | Meaning |
|-------|---------|---------|
| `stage` | `action_expert` | Stage to quantize: `vit` / `llm_prefix` / `action_expert`. |
| `method` | `fp8` | Quant method key (`chameleon info`). |
| `weight_dtype` | `fp8` | Weight dtype contract. |
| `kv_cache_dtype` | `fp8` | Optional KV cache dtype; Ada supports FP8 E4M3. |

> The `quantize` block in the YAML is **ignored** when `actions: [infer]` only.

#### Common tweaks

```yaml
model_overrides:
  use_reference: false
  checkpoint: /your/path/to/model.safetensors
  action_dim: 32
  action_horizon: 10      # from checkpoint config.json
  precision: bfloat16

actions: [quantize, infer]   # quantize then infer

platform: generic_cpu       # debug load only (slow)
infer:
  torch_device: cpu
```

#### Capability matrix (this config)

| Capability | Supported |
|------------|-----------|
| Real checkpoint load | Yes |
| PyTorch E2E infer (`sample_actions`) | Yes |
| Per-stage quantize | Yes (needs modelopt) |
| compile → TRT → infer | No (`use_compiled_engines: false`) |
| openpi Policy norm_stats / data transforms | Not wired (synthetic obs today) |

## Status

- **Functional**: core abstractions, registries, pi05 reference model, PyTorch
  runtime, the VLA orchestrator (real flow-matching denoise loop), config, CLI,
  workflow runner.
- **NVIDIA path (Phase 2, verified on-box)**: TensorRT compile + runtime with a
  declarative `TensorRegistry`, positional binding, persistent device buffers,
  `enqueueV3` and optional CUDA Graph; the **compile->infer loop is closed and
  numerically validated** (TRT FP16 vs PyTorch `cosine=1.0`, `max_abs≈1.25e-3`).
  Dual prefill/decode optimization profiles; `fmha_d256` as a real
  `torch.library` custom op with ONNX symbolic; real-openpi checkpoint loading;
  **RTX 4070 real-weight PyTorch E2E infer verified**
  (`configs/pi05_rtx4070_realweights.yaml`).
- **Known bring-up items**: ONNX QDQ export of modelopt-quantized modules, real
  model compile→TRT engine, on-device Orin/Thor + CuTe DSL plugin build. These
  degrade gracefully today.
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
