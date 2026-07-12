# Chameleon YAML 任务配置说明

本文说明 `TaskConfig`（`chameleon/config/schema.py`）驱动的 YAML 写法，按用途分章：

| 章节 | 典型文件 | CLI |
|------|----------|-----|
| [通用字段](#1-通用字段) | 所有配置共用 | — |
| [Deploy](#2-deploy导出与编译) | `configs/pi05/pi05_libero_trt_deploy.yaml` | `workflow` / `export` / `compile` |
| [Eval](#3-eval精度评测) | `configs/pi05/pi05_libero_trt_eval.yaml`、`pi05_libero_tvm_eval.yaml` | `eval` |
| [Bench](#4-bench阶段延迟对比) | `configs/pi05/pi05_libero_bench.yaml` | `bench` |

Schema 单一来源：`chameleon/config/schema.py`。加载：`TaskConfig.load(path)`。

---

## 1. 通用字段

几乎所有任务 YAML 都以如下顶层键开头。

### 1.1 身份与输出

| 字段 | 含义 | 示例 |
|------|------|------|
| `architecture` | 架构注册名，决定 stage 划分（pi05：`vit` / `llm` / `expert` / `denoise`） | `pi05` |
| `model` | 模型适配器注册名（通常与 architecture 同名或为变体） | `pi05` |
| `platform` | 部署平台（影响默认 runtime / compiler） | `nvidia_ada`、`generic_cpu` |
| `output_dir` | 本任务产物根目录（日志、bench JSON、默认 onnx/engines 等） | `output/pi05_libero_bench` |

### 1.2 `actions`

有序动作列表，决定 `chameleon workflow` 执行哪些步骤；子命令（`eval` / `bench` / `export`）也会读 YAML，但以各自 CLI 为主。

常见取值：

| action | 作用 |
|--------|------|
| `export` | ONNX 导出 |
| `compile` | TRT（等）engine 构建 |
| `trt_profile` | trtexec layer profile |
| `infer` | 单次冒烟推理 |
| `eval` | 数据集精度评测（通常直接用 `chameleon eval`） |
| `bench` | 分阶段延迟对比（通常直接用 `chameleon bench`） |
| `quantize` | 量化（pi05 ModelOpt / ASR Edge-LLM 等） |
| `stream` | ASR 流式 demo |

示例：

```yaml
actions:
  - export
  - compile
```

### 1.3 `model_overrides`

传给 `ModelAdapter.make_config()` 的覆盖项，**不同架构字段不同**。pi05 常用：

| 字段 | 含义 |
|------|------|
| `use_reference` | `false`：走真实 openpi 权重，不用内置 toy 参考网 |
| `checkpoint` | `model.safetensors` 路径（文件或所在目录由解析逻辑处理） |
| `action_dim` / `action_horizon` | 动作维度 / chunk 长度（LIBERO pi05 一般为 32 / 10） |
| `paligemma_variant` | 视觉语言模型规格，如 `gemma_2b` |
| `action_expert_variant` | 动作专家规格，如 `gemma_300m` |
| `precision` | 权重/推理精度标签（如 `bfloat16`），影响 openpi 加载与 TRT 挂载语义 |
| `tvm_dtype` | **仅 TVM 路径**：mlc-vla 编译/推理 dtype（推荐 `float16`，见 TVM eval 注释） |
| `tvm_loop` | **仅 TVM**：`true` = 图内 `denoise_loop_kv`；`false` = 宿主逐步 `denoise_step_kv` |
| `tvm_cuda_graph` | **仅 TVM**：整段去噪环 CUDA Graph（需 `tvm_loop=true`） |

### 1.4 `deploy`

部署路径与产物目录（export/compile/eval/bench 都会读 `engine_dir`）。

| 字段 | 含义 |
|------|------|
| `backend` | `pi05` / `cosmos3` / `qwen3_asr` / `reference` 等 |
| `checkpoint_dir` | 含 `model.safetensors` 的目录 |
| `train_config` | openpi TrainConfig 名（如 `pi05_libero`） |
| `export_dir` | ONNX 输出目录 |
| `engine_dir` | TRT engine 目录（`vit.engine` / `llm.engine` / `denoise.engine`…） |
| `use_cudagraph` | build 时是否开 TRT `BuilderFlag.CUDA_GRAPH` |
| `build_cfg_dir` / `build_cfgs` | 覆盖各 stage 的 build_cfg `.py` |
| `edgellm_home` | Edge-LLM 根目录（ASR 用；也可设环境变量 `TENSORRT_EDGELLM_HOME`） |

### 1.5 `infer`

推理冒烟 / denoise 步数等。

| 字段 | 含义 |
|------|------|
| `batch_size` | batch（edge VLA 通常为 1） |
| `num_steps` | flow-matching Euler 步数（默认与模型 `num_denoise_steps` 对齐，LIBERO 常为 10） |
| `torch_device` | PyTorch 侧设备提示（如 `cuda`） |
| `use_compiled_engines` | workflow infer 是否消费刚 compile 的 engine |
| `cuda_graph` | runtime 侧 CUDA Graph（与 evaluate.trt_cuda_graph 不同入口） |

### 1.6 `data`

真实数据集（eval / bench 取 observation 用）。

| 字段 | 含义 |
|------|------|
| `dataset` | dataloader 注册名，如 `pi05_libero`、`librispeech_test_clean` |
| `openpi_config` | openpi DataConfig / TrainConfig 名（repack、默认 repo） |
| `repo_id` | 覆盖 LeRobot `repo_id` |
| `dataset_root` | 本地数据集根；空则用 HF cache |
| `start_index` | 起始帧下标 |
| `num_samples` | 从 `start_index` 起最多暴露多少帧（bench 常设 1） |

### 1.7 `evaluate`

精度评测与 runner 选择；**bench 也会复用**其中的 checkpoint / noise / engine_dir / device。

详见 [§3](#3-eval精度评测)。

### 1.8 `bench`

仅 stage 延迟对比使用。详见 [§4](#4-bench阶段延迟对比)。

---

## 2. Deploy（导出与编译）

**目的**：把 openpi 权重导出为分 stage ONNX，再编译成 TensorRT engine，供后续 eval / bench 加载。

**示例**：[`configs/pi05/pi05_libero_trt_deploy.yaml`](../../configs/pi05/pi05_libero_trt_deploy.yaml)

### 2.1 启动

```bash
PYTHONPATH=. models/openpi/.venv/bin/python -m chameleon.cli workflow \
    --config configs/pi05/pi05_libero_trt_deploy.yaml
```

仅导出（可加 `--dry-run` 看计划）：

```bash
PYTHONPATH=. models/openpi/.venv/bin/python -m chameleon.cli export \
    --config configs/pi05/pi05_libero_trt_deploy.yaml
```

### 2.2 关键段落

```yaml
actions:
  - export
  - compile
  - trt_profile   # 可选：对各 engine 做 trtexec layer profile
```

| 段落 | 作用 |
|------|------|
| `export` | 列表，每项一个 `stage`（`vit` / `llm` / `expert` / `denoise`） |
| `compile` | 每项指定 `stage` + `options.build_cfg`（指向 `configs/build_configs/*.py`） |
| `trt_profile` | 要对哪些 stage 跑 trtexec；全局参数在 `profile:` |
| `profile` | `profile_dir`、`iterations`、`viewer`（`static`/`webui`/`both`）、`trtexec_path` 等 |

**注意**：

- LIBERO 三相机 prefix 长度须与 build_cfg 一致（通常 968）；改长度需重新 export + compile。
- `compile` 用的 TensorRT 版本须与运行时 / `trtexec` 一致（可用 `profile.trtexec_path` 指定）。
- deploy 产出的 `engine_dir` 会被 eval / bench 的 `deploy.engine_dir` 或 `evaluate.engine_dir` 引用。

### 2.3 典型产物

```text
{export_dir}/          # ONNX + sidecar
{engine_dir}/
  vit.engine
  llm.engine
  expert.engine        # 部分流程单独 build；运行时 denoise 常已内嵌 expert
  denoise.engine
{profile_dir}/         # 若跑了 trt_profile
```

---

## 3. Eval（精度评测）

**目的**：在 LeRobot 真实数据上对比预测动作与 ground-truth（或 PT vs TRT / PT vs TVM 双路）。

**示例**：

- TRT 单路：[`configs/pi05/pi05_libero_trt_eval.yaml`](../../configs/pi05/pi05_libero_trt_eval.yaml)
- TVM 单路：[`configs/pi05/pi05_libero_tvm_eval.yaml`](../../configs/pi05/pi05_libero_tvm_eval.yaml)
- 双路对比：`pi05_libero_trt_compare.yaml`、`pi05_libero_tvm_compare.yaml`

### 3.1 启动

```bash
PYTHONPATH=. models/openpi/.venv/bin/python -m chameleon.cli eval \
    --config configs/pi05/pi05_libero_trt_eval.yaml
```

TVM 额外需要环境变量：`TVM_HOME`、`MLC_VLA_HOME`、`MLC_VLA_PY`（及通常的 `TVM_LIBRARY_PATH`）。

### 3.2 `evaluate` 字段详解

| 字段 | 含义 |
|------|------|
| `checkpoint_dir` | openpi Policy 权重目录 |
| `norm_stats_dir` | 含 `{asset_id}/norm_stats.json` 的 assets 根 |
| `asset_id` | 如 `physical-intelligence/libero` |
| `policy_runner` | 策略后端，见下表 |
| `compare_mode` | `true` 时双路对比（需 `pt_trt_compare` / `pt_tvm_compare`） |
| `engine_dir` | TRT engine 目录（可覆盖 `deploy.engine_dir`） |
| `trt_engines` | 可选，覆盖各 stage 文件名 |
| `noise` | `random` \| `fixed`；双路对比建议 `fixed` |
| `noise_seed` | `fixed` 时的基础 seed |
| `precision` | TRT 挂载精度提示（如 `bf16`） |
| `trt_cuda_graph` | TRT 路 CUDA Graph；compare 模式常自动关 |
| `num_samples` | 评测帧数 |
| `stride` | 跳帧步长 |
| `compare_horizon` | 只对比前 N 个动作步；`null` = 全程 |
| `device` | 推理设备（`cuda`） |
| `pytorch_load_device` | 构建 Policy 时权重先放哪（默认 `cpu` 省显存） |
| `viewer` | `console` \| `webui` \| `both` \| `none` |
| `webui_*` | WebSocket 宿主、端口、路径、历史长度、JPEG 等 |

**`policy_runner` 常用值**：

| 值 | 含义 |
|----|------|
| `openpi` | 纯 PyTorch openpi |
| `trt_only` | 仅 TRT（vit+llm+denoise） |
| `tvm_only` | vit=TRT，llm prefill+denoise=TVM |
| `pt_trt_compare` | PT vs TRT 双路 |
| `pt_tvm_compare` | PT vs TVM 双路 |
| `chameleon` | 框架 Orchestrator 路径 |

### 3.3 `data` 与 `evaluate.num_samples`

- `data.num_samples`：限制 dataloader 暴露的帧窗口。
- `evaluate.num_samples`：实际评测循环使用的样本数（可再截断）。

两者可不同；评测以 `evaluate` 为准，并会与 `data` 同步部分字段（见 `sync_eval_num_samples`）。

### 3.4 TRT vs TVM eval 差异（`model_overrides`）

TVM eval 额外关心：

```yaml
model_overrides:
  tvm_dtype: float16      # 精度/速度权衡；fp16 通常优于 bf16
  tvm_loop: true          # 图内整段 Euler
  tvm_cuda_graph: true    # 需 tvm_loop=true
evaluate:
  policy_runner: tvm_only
```

TRT eval：

```yaml
evaluate:
  policy_runner: trt_only
  trt_cuda_graph: false   # 按需开启
```

---

## 4. Bench（阶段延迟对比）

**目的**：同一 observation + fixed noise，对 TRT / TVM 做 warmup + N 次计时，输出分阶段 p50/p90 对比表与 JSON。用于定位「TVM 比 TRT 慢在哪一段」。

**主配置（逐项说明）**：[`configs/pi05/pi05_libero_bench.yaml`](../../configs/pi05/pi05_libero_bench.yaml)  
**逐步 denoise 变体**：[`configs/pi05/pi05_libero_bench_steps.yaml`](../../configs/pi05/pi05_libero_bench_steps.yaml)

### 4.1 前置与启动

前置：

1. TRT engines 已 build（至少 `vit` / `llm` / `denoise`，见 deploy 配置）。
2. TVM：`TVM_HOME`、`MLC_VLA_HOME`、`MLC_VLA_PY`。

```bash
export TVM_HOME=/path/to/tvm
export MLC_VLA_HOME=/path/to/mlc-vla
export MLC_VLA_PY=/usr/bin/python3.12

PYTHONPATH=. models/openpi/.venv/bin/python -m chameleon.cli bench \
    --config configs/pi05/pi05_libero_bench.yaml -v
```

CLI 可覆盖：`--runs`、`--warmup`、`--backends trt,tvm`、`--tvm-loop true|false`、`--output`。

若双后端串行仍 OOM，可分开跑：

```bash
... bench --backends trt
... bench --backends tvm
```

### 4.2 全文结构导读

下面按 `pi05_libero_bench.yaml` 分段解释。

#### （1）顶层身份

```yaml
architecture: pi05
model: pi05
platform: nvidia_ada
output_dir: output/pi05_libero_bench
```

- 声明这是 pi05 + Ada 类 NVIDIA 平台。
- 报告默认写在 `output_dir` 下（也可被 `bench.output` 覆盖）。

#### （2）`actions`

```yaml
actions:
  - bench
```

标记本文件用途；实际执行靠 `chameleon bench`（不必走 `workflow`）。

#### （3）`model_overrides`

```yaml
model_overrides:
  use_reference: false
  checkpoint: models/openpi/pytorch/model.safetensors
  action_dim: 32
  action_horizon: 10
  paligemma_variant: gemma_2b
  action_expert_variant: gemma_300m
  precision: bfloat16
  tvm_dtype: float16
  tvm_loop: true
  tvm_cuda_graph: true
```

| 项 | 在 bench 中的作用 |
|----|-------------------|
| `checkpoint` 等 | 构建 openpi Policy（语言 embedding 仍走 PT；大权重会释放） |
| `tvm_dtype` | TVM worker 编译/运行精度 |
| `tvm_loop: true` | 默认测「图内整段 denoise」（与线上稳态路径一致） |
| `tvm_cuda_graph: true` | 在 loop 模式下启用 CUDA Graph；首次 warmup 可能较慢 |

逐步对齐 TRT denoise×N 时，用 `bench.tvm_loop: false` 或 `--tvm-loop false`（见 steps YAML）。

#### （4）`deploy`

```yaml
deploy:
  backend: pi05
  checkpoint_dir: models/openpi/pytorch
  train_config: pi05_libero
  export_dir: output/pi05_libero_trt/onnx
  engine_dir: output/pi05_libero_trt/engines
```

bench **不重新 export/compile**，只读取已有 `engine_dir`：

- TRT 后端：加载 `vit` / `llm` / `denoise`
- TVM 后端：只加载 `vit`，llm+denoise 在 3.12 worker 内

`export_dir` 在 bench 中可忽略，保留是为了与 deploy/eval 配置对齐、方便复制。

#### （5）`infer`

```yaml
infer:
  num_steps: 10
  torch_device: cuda
```

- `num_steps`：Euler 步数，须与模型 / TVM 编译期 `num_denoise_steps` 一致才能走 `tvm_loop`。
- `torch_device`：Policy / TRT 推理设备提示。

#### （6）`data`

```yaml
data:
  dataset: pi05_libero
  openpi_config: pi05_libero
  start_index: 0
  num_samples: 1
```

bench 从数据集取 **一帧** observation（由 `bench.sample_index` 指定下标，默认 0）：

- `num_samples: 1` 只打开最小窗口，加快加载。
- 不需要整库评测；要换样本改 `sample_index` / `start_index`。

#### （7）`evaluate`（bench 复用）

```yaml
evaluate:
  checkpoint_dir: models/openpi/pytorch
  norm_stats_dir: ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets
  asset_id: physical-intelligence/libero
  policy_runner: trt_only          # 会被 bench 按 backends 临时改成 trt_only / tvm_only
  engine_dir: output/pi05_libero_trt/engines
  noise: fixed
  noise_seed: 0
  precision: bf16
  trt_cuda_graph: false
  device: cuda
  pytorch_load_device: cpu
  viewer: console
```

| 项 | 含义 |
|----|------|
| `norm_stats_dir` / `asset_id` | openpi 输入归一化；缺失会跳过并打日志 |
| `noise: fixed` + `noise_seed` | TRT/TVM **同一噪声**，延迟可比、数值也可抽查 |
| `engine_dir` | 覆盖/确认 TRT 路径 |
| `trt_cuda_graph` | 本配置默认关；与 `model_overrides.tvm_cuda_graph` 独立 |
| `pytorch_load_device: cpu` | 先 CPU 加载再释放大模块，减轻与 TVM 切换时的显存压力 |
| `policy_runner` | 文件里的值会被 `run_bench` 按当前 backend 覆盖，写 `trt_only` 仅作占位 |

#### （8）`bench`（核心）

```yaml
bench:
  backends: [trt, tvm]
  warmup: 3
  runs: 20
  sync: cuda_event
  sample_index: 0
  output: output/pi05_libero_bench/bench.json
```

| 字段 | 含义 |
|------|------|
| `backends` | 串行测试的后端列表：`trt`、`tvm`（顺序执行，中间会 `close` 释放显存） |
| `warmup` | 不计时的预热次数（含 CUDA Graph 捕获） |
| `runs` | 正式计次数；报告 mean / p50 / p90 |
| `sync` | `cuda_event`（GPU 段用 CUDA Event）或 `host`（纯 `perf_counter`） |
| `sample_index` | 在 `data` 打开的窗口内取第几帧 |
| `output` | JSON 报告路径 |
| `tvm_loop` | 可选；覆盖 `model_overrides.tvm_loop`（steps 配置里设为 `false`） |
| `stages` | 可选；控制台对比表优先显示的 stage 顺序 |

### 4.3 计时 stage 含义

| stage | TRT | TVM |
|-------|-----|-----|
| `preprocess` | openpi `_preprocess_observation` | 同左 |
| `vit` | TRT SigLIP（多相机累加） | 同左（共用） |
| `lang_embed` | PT `embed_language_tokens` | 同左 |
| `prefix_prep` | pad + attention mask | pad + `.cpu().numpy()` |
| `llm_prefill` | TRT `llm.engine` | worker 内 `prefill`（P1 已拆分） |
| `denoise_total` | 全部 denoise 步总和 | worker 内 `denoise_loop_kv` 或逐步总和 |
| `denoise_step_mean` | — | 仅 `tvm_loop=false` |
| `tvm_worker` | — | prefill+denoise 合计（worker 内） |
| `ipc` | — | 父进程往返 − `tvm_worker` |
| `e2e` | 单次 `policy.infer` 墙钟 | 同左 |

读报告时优先看：

- `vit` 是否接近（验证计时可信）
- `llm_prefill` / `denoise_total` 的 delta（慢在哪段）
- `ipc` 是否可忽略
- `delta.core_llm_denoise`（两侧都有拆分时）

### 4.4 图内环 vs 逐步环

| 配置 | `tvm_loop` | 适用 |
|------|------------|------|
| `pi05_libero_bench.yaml` | `true`（默认） | 对齐线上稳态（图内 + 可选 CUDA Graph） |
| `pi05_libero_bench_steps.yaml` | `false` | 与 TRT「host 环 ×N 次 denoise」粒度更接近；有 `denoise_step_mean` |

同一主配置也可：

```bash
... bench --config configs/pi05/pi05_libero_bench.yaml --tvm-loop false
```

### 4.5 输出示例

控制台：

```text
stage                  trt_p50     tvm_p50       delta
------------------------------------------------------
vit                      26.20       26.09       -0.11
llm_prefill              85.58      xxx.xx     +xxx.xx
denoise_total            51.57      xxx.xx     +xxx.xx
e2e                     181.59      476.16     +294.57
```

JSON：`bench.output`，含 `meta`、`backends.*.{mean,p50,p90}`、`delta`。

---

## 5. 配置选型速查

| 你想做的事 | 用哪个 YAML | 命令 |
|------------|-------------|------|
| 导出 + 编 TRT | `pi05_libero_trt_deploy.yaml` | `workflow` |
| TRT 精度 vs GT | `pi05_libero_trt_eval.yaml` | `eval` |
| TVM 精度 vs GT | `pi05_libero_tvm_eval.yaml` | `eval` |
| PT vs TRT/TVM 双路 | `*_compare.yaml` | `eval` |
| TRT vs TVM 延迟拆分 | `pi05_libero_bench.yaml` | `bench` |
| 逐步 denoise 延迟拆分 | `pi05_libero_bench_steps.yaml` | `bench` |
| TRT llm/denoise layer 基线 | `pi05_libero_trt_profile.yaml` | `trt-profile` |
| **Thor 编 TRT 引擎** | `pi05_libero_trt_deploy_thor.yaml` | `workflow` |
| **Thor TRT vs TVM 延迟** | `pi05_libero_bench_thor.yaml` | `bench` |

Jetson Thor（sm_101）：引擎设备相关，须在 Thor 本机 build 后再 bench。先
`source scripts/tvm_thor.sh && export MLC_VLA_PY=<thor python3.12>`，再：

```bash
bash scripts/profile_pi05_trt_tvm.sh --run thor          # build 引擎 + bench
# 或分步：--run thor-deploy / --run thor-bench
```

注意先确认 Thor 的 TVM 带 cuBLAS 扩展（`$MLC_VLA_PY -c "import tvm;
print(bool(tvm.get_global_func('relax.ext.cublas', True)))"`），否则 mlc-vla 自动回退 dlight，
TVM 会明显慢于 TRT。

推荐流水线：

```text
deploy → eval(trt_only / tvm_only) → bench（定位差距）
  → mlc_vla.bench_kv / nsys（TVM kernel）+ trt-profile（TRT layer）
  → 优化
```

深层 profiling 命令（`bench_kv` / `nsys` / `trt-profile`）见
[`docs/optimizer/pi05/trt_tvm_profile.md`](../optimizer/pi05/trt_tvm_profile.md)；
一键脚本：`bash scripts/profile_pi05_trt_tvm.sh`（加 `--run` 才真正执行）。

---

## 6. 相关代码入口

| 模块 | 路径 |
|------|------|
| Schema | `chameleon/config/schema.py` |
| Bench | `chameleon/profile/bench.py`、`commands/bench.py` |
| Stage 计时 | `chameleon/profile/stage_timer.py` |
| TRT 管线 | `chameleon/runtime/pi05_trt/pipeline.py` |
| TVM 管线 / worker | `chameleon/runtime/pi05_tvm/` |
| Eval | `chameleon/commands/eval.py`、`evaluate/*_runner.py` |
| Deploy | `chameleon/deploy/`、`workflows/actions.py` |
