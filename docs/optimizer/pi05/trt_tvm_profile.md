# pi05 LIBERO：TRT vs TVM 深层 Profiling

承接 `chameleon bench` 的 stage 结论（TVM `llm_prefill` ~2.4×、`denoise_total` ~4.4× 于 TRT），
本页给出 **可复现命令**：用 `mlc_vla.bench_kv` / `nsys` 拆 TVM kernel，用 `trt-profile` 拿 TRT layer 基线。

| 层 | 工具 | 回答什么 |
|----|------|----------|
| Stage（已完成） | `chameleon bench` | e2e / prefill / denoise 墙钟差距 |
| 单算子延迟 | `python -m mlc_vla.bench_kv` | 无 Vit/IPC 的 `prefill` / `denoise_step_kv` |
| Kernel 时间线 | `nsys` | launch 间隙、Graph、HtoD、主机空泡 |
| Layer 基线 | `chameleon trt-profile` | TRT `llm` / `denoise` 各层 ms |

形状与 bench 一致：`prefix_len=968`（3×256 + 200）、`action_horizon=10`、`num_steps=10`、TVM `float16`。

---

## 0. 环境

```bash
cd /path/to/Chamleon   # 或板端 /srcs/Chamleon
source scripts/tvm_env.sh   # 设置 TVM_HOME / MLC_VLA_HOME / MLC_VLA_PY / PYTHONPATH

# openpi 评测 venv（bench / trt-profile）
CHAM_PY=${CHAM_PY:-models/openpi/.venv/bin/python}
export PYTHONPATH=.

# 产出目录
OUT=${OUT:-output/pi05_libero_profile}
mkdir -p "$OUT"/{nsys,bench_kv,trt}
```

自检：

```bash
$MLC_VLA_PY -c "import tvm_ffi, tvm, mlc_vla; print('tvm ok', tvm.__file__)"
$CHAM_PY -c "import chameleon; print('cham ok')"
which nsys trtexec
```

---

## 1. TVM：`mlc_vla.bench_kv`（首选对照）

随机权重、无 Vit、无跨进程；延迟与权重数值无关，形状与 LIBERO 默认 `Pi0Config` 一致（968）。

### 1.1 宿主逐步（对齐 `tvm_loop=false` / `denoise_step_mean`）

```bash
$MLC_VLA_PY -m mlc_vla.bench_kv \
  --target cuda --dtype float16 --steps 10 --iters 50 \
  | tee "$OUT/bench_kv/fp16_steps.txt"
```

期望输出字段：

| 字段 | 应对齐的 Chamleon bench |
|------|-------------------------|
| `prefill` | `llm_prefill`（TVM） |
| `denoise_step_kv(M1)` | `denoise_step_mean` |
| `10-step total M1` | `llm_prefill + denoise_total`（steps 模式） |

### 1.2 CUDA Graph 单步（对齐生产 `tvm_cuda_graph`）

```bash
$MLC_VLA_PY -m mlc_vla.bench_kv \
  --target cuda --dtype float16 --steps 10 --iters 50 --cuda-graph \
  | tee "$OUT/bench_kv/fp16_steps_cg.txt"
```

若 `--cuda-graph` 与默认差距很小，说明 Graph 未吃掉主要开销，瓶颈在 kernel 本身（与 bench loop≈steps 结论一致）。

### 1.3 与 Chamleon steps bench 并读

```bash
$CHAM_PY -m chameleon.cli bench \
  --config configs/pi05/pi05_libero_bench_steps.yaml -v \
  | tee "$OUT/bench_kv/cham_steps.txt"
```

判读：

1. `bench_kv.prefill` ≈ `bench.llm_prefill(TVM)` → 差距在 LLM 图内，不在 Vit/IPC。
2. `bench_kv.denoise_step_kv × 10` ≈ `bench.denoise_total(TVM)` → 同上。
3. 若 Chamleon 明显大于 `bench_kv` → 再查 worker 同步、pickle、TRT Vit。

---

## 2. TVM：`nsys` 包 `bench_kv`

只采稳态（跳过编译与 warmup 较难自动切；先整段采，再在 GUI/`stats` 里盯后半段重复 kernel）。

### 2.1 逐步路径

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  -s none \
  --force-overwrite=true \
  -o "$OUT/nsys/tvm_bench_kv_fp16" \
  -- $MLC_VLA_PY -m mlc_vla.bench_kv \
       --target cuda --dtype float16 --steps 10 --iters 20
```

### 2.2 开 CUDA Graph

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  -s none \
  --force-overwrite=true \
  -o "$OUT/nsys/tvm_bench_kv_fp16_cg" \
  -- $MLC_VLA_PY -m mlc_vla.bench_kv \
       --target cuda --dtype float16 --steps 10 --iters 20 --cuda-graph
```

### 2.3 汇总

```bash
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  "$OUT/nsys/tvm_bench_kv_fp16.nsys-rep" \
  | tee "$OUT/nsys/tvm_bench_kv_fp16_stats.txt"

nsys analyze -r gpu_gaps,gpu_time_util \
  "$OUT/nsys/tvm_bench_kv_fp16.nsys-rep" \
  | tee "$OUT/nsys/tvm_bench_kv_fp16_analyze.txt"
```

关注：

| 现象 | 含义 |
|------|------|
| Top kernel 多为 GEMM / attn | 算子级优化（dlight / CUTLASS / 量化） |
| 大量短 kernel + 高 `cudaLaunchKernel` | launch 开销；Graph / fuse 更有价值 |
| `gpu_gaps` 大、queue 近 0 | 主机侧饿 GPU（同步、Python） |
| Graph 开后仍接近逐步 | 与 bench 一致：优化目标转向 kernel |

可选：对 e2e worker 再包一层（噪声大，仅当怀疑 IPC）：

```bash
nsys profile -t cuda,nvtx,osrt -s none -o "$OUT/nsys/tvm_bench_e2e" \
  -- $CHAM_PY -m chameleon.cli bench \
       --config configs/pi05/pi05_libero_bench_steps.yaml --backends tvm
```

---

## 3. TRT：`trt-profile`（llm + denoise layer 基线）

**不重建 engine**；形状/plugin 必须与 compile 一致。`denoise` 已含 expert 子图，**不要**把 `expert` 再与 denoise 相加。

### 3.1 专用配置（仅 profile）

```bash
$CHAM_PY -m chameleon.cli trt-profile \
  --config configs/pi05/pi05_libero_trt_profile.yaml -v
```

产物（默认）：

```text
output/pi05_libero_trt/profiles/
  llm.profile.json
  denoise.profile.json
  index.html
```

查看：

```bash
$CHAM_PY -m chameleon.cli draw profile --config configs/pi05/pi05_libero_trt_profile.yaml
# 或单文件
$CHAM_PY -m chameleon.cli draw profile output/pi05_libero_trt/profiles/llm.profile.json
```

### 3.2 与 bench 对齐读数

| TRT profile | 换算 | 对比 |
|-------------|------|------|
| `llm` 各层合计（≈ 单次 enqueue） | ×1 | `bench.llm_prefill(TRT)` |
| `denoise` 各层合计 | **× `num_steps`(10)** | `bench.denoise_total(TRT)` |

Layer 表里重点看：Attention / MatMul / Softmax / LayerNorm（或 Myelin 融合块名）。

### 3.3 可选：`nsys` 包单次 `trtexec`

与 `docs/hw/nvidia/shared_mem.md` 同思路；plugin / shapes 从对应 `build_cfg` 取。

```bash
# 示例：llm（按本机 trtexec / plugin 路径改）
ENGINE=output/pi05_libero_trt/engines/llm.engine
SHAPES="inputs_embeds:1x968x2048,attention_mask:1x1x968x968,position_ids:1x968"

nsys profile -t cuda,nvtx -s none \
  -o "$OUT/nsys/trt_llm" --force-overwrite=true \
  -- trtexec --loadEngine="$ENGINE" \
       --shapes="$SHAPES" \
       --warmUp=200 --iterations=20 --noDataTransfers
```

`denoise` 同理换 engine 与 `denoise_step_build_cfg.py` 的 shape 串；比较时仍 ×10。

---

## 4. 对照表（填实测）

跑完后填空，决定下一优化方向：

| 指标 | TRT | TVM | 来源 |
|------|-----|-----|------|
| `llm_prefill` ms | | | `chameleon bench` |
| `denoise_total` ms | | | `chameleon bench` |
| `denoise_step_mean` ms | ≈ total/10 | | steps bench / TRT×1 |
| `bench_kv.prefill` | — | | `mlc_vla.bench_kv` |
| `bench_kv.denoise_step_kv` | — | | 同上 |
| Top-5 TRT llm layers | | — | `*.profile.json` |
| Top-5 TRT denoise layers | | — | 同上 |
| nsys Top kernels (TVM) | — | | `cuda_gpu_kern_sum` |

决策简表：

1. **TVM prefill ≫ TRT llm layer 合计** → 优先 LLM attention/GEMM（dlight / BYOC / 量化）。
2. **TVM step ≫ TRT denoise×1** → 优先 suffix-only attn / expert FFN。
3. **bench_kv ≈ Chamleon，但远慢于 TRT** → 图编译与 schedule，不是 IPC。
4. **nsys 显示 launch/gap 主导** → Graph / fuse / 减 sync；否则上 ncu 看 SOL%。

下一层（本页不做）：`ncu` 对 Top kernel 做 roofline（见 `docs/hw/nvidia/shared_mem.md` §9）。

---

## 5. 一键脚本

```bash
bash scripts/profile_pi05_trt_tvm.sh           # 打印并可选执行
bash scripts/profile_pi05_trt_tvm.sh --run     # 实际跑（需 GPU + engine + TVM）
bash scripts/profile_pi05_trt_tvm.sh --run kv  # 只跑 bench_kv
bash scripts/profile_pi05_trt_tvm.sh --run trt # 只跑 trt-profile
bash scripts/profile_pi05_trt_tvm.sh --run nsys # 只跑 nsys(bench_kv)
```

---

## 6. 相关入口

| 路径 | 作用 |
|------|------|
| `configs/pi05/pi05_libero_bench.yaml` | Stage 对比（loop） |
| `configs/pi05/pi05_libero_bench_steps.yaml` | Stage 对比（逐步） |
| `configs/pi05/pi05_libero_trt_profile.yaml` | 仅 llm/denoise trtexec profile |
| `mlc-vla/python/mlc_vla/bench_kv.py` | 孤立 M1 测速 |
| `chameleon/deploy/trt_profile.py` | trtexec 组装 |
| `docs/manual/yaml_config.md` §4–5 | Bench YAML 说明 |
