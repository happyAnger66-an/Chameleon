# Jetson Thor Roofline 模型

本文档说明 **Jetson Thor（T5000 为主）** 的 Roofline 性能模型：硬件顶、拐点、kernel 判读方法，以及与 π0.5 LLM / TensorRT FP8 推理的关系。

相关硬件背景见 [shared_mem.md](../hw/nvidia/shared_mem.md)。

---

## 1. Roofline 是什么

Roofline 用一张图同时看 **算力上限** 和 **带宽上限**：

```text
Performance (FLOP/s)
    ▲
    │     ╱────────────  Compute Roof（算力顶）
    │    ╱
    │   ╱  ← 斜线：Memory Roof（带宽顶）
    │  ╱
    └──────────────────────► Operational Intensity (FLOP/Byte)
                              每从内存读 1 Byte，做多少 FLOP
```

### 1.1 核心公式

**Operational Intensity（运算强度）**：

\[
AI = \frac{\text{有效 FLOPs}}{\text{访问的字节数（通常指 DRAM）}}
\]

**Memory-bound 区（斜线以下）**：

\[
\text{Achievable FLOP/s} \le B_{\text{mem}} \times AI
\]

**Compute-bound 区（水平顶）**：

\[
\text{Achievable FLOP/s} \le P_{\text{peak}}
\]

**拐点（Ridge Point）**：

\[
AI_{\text{ridge}} = \frac{P_{\text{peak}}}{B_{\text{mem}}}
\]

| 相对拐点位置 | 含义 |
|--------------|------|
| \(AI < AI_{\text{ridge}}\) | 更偏 **memory-bound**（带宽主导） |
| \(AI > AI_{\text{ridge}}\) | 更偏 **compute-bound**（算力主导） |

---

## 2. Jetson Thor T5000 硬件参数

来源：NVIDIA Jetson T5000 Modules Data Sheet（DS-11945）。

### 2.1 Memory Roof（内存顶）

| 项目 | 值 |
|------|-----|
| 类型 | 128 GB LPDDR5X，256-bit |
| **峰值带宽** | **273 GB/s** |
| 特性 | **统一内存**（CPU / GPU 共享，无独立 GDDR） |

Roofline 输入：

\[
B_{\text{mem}} = 273 \times 10^9\ \text{Byte/s}
\]

> 实际 sustained 带宽通常低于 273 GB/s（CPU 访问、拷贝、多 client 争用）。ncu 中用 `dram__throughput.avg.pct_of_peak_sustained_elapsed` 衡量离顶多远。

### 2.2 Compute Roof（算力顶）

按 **精度 / 执行路径** 选择不同的水平「顶」：

| 路径 | MAXN（~130W） | 120W | 说明 |
|------|---------------|------|------|
| **FP32 CUDA Core** | **8.064 TFLOP/s** | **7.096 TFLOP/s** | 通用 CUDA kernel |
| **FP8 Tensor Core（Dense）** | **517 TFLOP/s** | **455 TFLOP/s** | `llm_fp8.engine` 应走这条 |
| **FP8 Tensor Core（Sparse）** | 1035 TFLOP/s | 910 TFLOP/s | 需稀疏模式，不作默认顶 |
| **FP4 Tensor Core（Dense）** | 1035 TFLOP/s | 910 TFLOP/s | NVFP4 等 |

GPU 规模（DevKit 常见）：**20 SM**，2560 CUDA cores，Blackwell sm_100 系，GPU 名 `NVIDIA Thor`。

Datasheet **未单独列出 FP16/BF16 峰值**。BF16 TRT 路径可先用 ncu 的 `sm__inst_executed_pipe_tensor` 反推实际算力利用率。

### 2.3 功耗与频率

| 模式 | GPU 频率（up to） | FP32 顶 | FP8 Dense 顶 |
|------|-------------------|---------|--------------|
| MAXN | 1.575 GHz | 8.064 TFLOP/s | 517 TFLOP/s |
| 120W | 1.386 GHz | 7.096 TFLOP/s | 455 TFLOP/s |

Profiling 前建议锁频，避免 roof 与 kernel 点漂移：

```bash
sudo jetson_clocks   # 或 nvpmodel 固定 MAXN
nvidia-smi --query-gpu=name,compute_cap,clocks.max.graphics --format=csv
```

---

## 3. Ridge Point（拐点）

\[
AI_{\text{ridge}} = \frac{P_{\text{peak}}}{273\ \text{GB/s}}
\]

| Compute Roof | \(P_{\text{peak}}\) | \(AI_{\text{ridge}}\) | 典型用途 |
|--------------|---------------------|------------------------|----------|
| FP32 CUDA（MAXN） | 8.064 TFLOP/s | **~29.5 FLOP/B** | 通用 CUDA |
| FP32 CUDA（120W） | 7.096 TFLOP/s | **~26.0 FLOP/B** | 降功耗 |
| **FP8 TC Dense（MAXN）** | 517 TFLOP/s | **~1894 FLOP/B** | **llm_fp8.engine** |
| **FP8 TC Dense（120W）** | 455 TFLOP/s | **~1667 FLOP/B** | 降功耗 FP8 |
| FP8 TC Sparse（MAXN） | 1035 TFLOP/s | ~3791 FLOP/B | 仅 sparse workload |

### 3.1 对 π0.5 LLM prefill 的含义

典型配置：`batch=1`，`seq_len=968`，TRT FP8 engine。

- GEMM 理论 AI 随 M/N/K 变化；小 batch、大 K 时常在 **几十 FLOP/B** 量级
- 相对 FP8 ridge **~1894 FLOP/B**，绝大多数 kernel **远在拐点左侧 → DRAM memory-bound**
- CUTLASS 大 tile（227 KiB shared/block）是在 **带宽受限** 下提高 **achieved FLOP/s**，低 occupancy 不一定是 bug

示意（FP8 Dense MAXN，横轴 log scale）：

```text
Performance
    │
517T │────────────────────────────  FP8 Tensor Core 顶
    │              ╱
    │            ╱  ← 斜率 = 273 GB/s
    │          ╱
    │        ╱   ★ 典型 LLM GEMM（batch=1）
    │      ╱
    └──────────────────► AI (FLOP/Byte)
         1   10  100  1K  10K
              ↑ ridge ≈ 1894
```

---

## 4. 多层 Roofline

Roofline 不只有 DRAM 一条斜线，还可叠加 **L2 / L1**（数据已在 cache 内时）：

```text
Performance
    │
    │──────── FP8 Compute Roof
    │      ╱
    │     ╱ L2 roof（更高斜率，容量有限）
    │    ╱
    │   ╱ DRAM roof（273 GB/s）
    └────────────────► AI
```

| 层级 | ncu 指标 | 说明 |
|------|----------|------|
| **DRAM** | `dram__bytes_*`，`dram__throughput.*` | 主 roof；Thor 统一内存 273 GB/s |
| **L2** | `lts__t_sectors.sum`，`lts__throughput.*` | Blackwell L2 更大；reuse 好时可抬离 DRAM 斜线 |
| **L1 / Shared** | `l1tex__*` | 大 shared GEMM 在 SM 内 tile；权重/激活仍从 DRAM 进入 |

ncu **Speed of Light** 页的 **Memory Throughput % vs Compute Throughput %** 是简化版 roofline 判读。

---

## 5. 用 ncu 把 kernel 标到 Roofline 上

### 5.1 采集命令（单热点 kernel）

```bash
export ENGINE=/tmp/pi05/build/llm/llm.engine
export SHAPES="inputs_embeds:1x968x2048,attention_mask:1x1x968x968,position_ids:1x968"

sudo ncu \
  --launch-count 1 \
  --kernel-name-base demangled \
  --kernel-name "myl_Fc|cutlass" \
  --metrics \
    dram__bytes_read.sum,dram__bytes_write.sum,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,\
    gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
  -f -o /tmp/roofline_kernel \
  /opt/tensorrt/bin/trtexec \
    --loadEngine="${ENGINE}" \
    --shapes="${SHAPES}" \
    --warmUp=0 \
    --iterations=1 \
    --noDataTransfers
```

整趟 forward 多 kernel 采集见 [shared_mem.md §9](../hw/nvidia/shared_mem.md#9-ncu只采集一次-forward-的分层内存--计算分析)（NVTX capture-range，**不要** `--launch-count 1`，**不要** `--useCudaGraph`）。

### 5.2 计算 AI 与 Achieved FLOP/s

```text
Bytes   ≈ dram__bytes_read.sum + dram__bytes_write.sum   （该 kernel 一次 launch）
FLOPs   ≈ MatMul: 2 × M × N × K
        或 ncu --section ComputeWorkloadAnalysis

AI              = FLOPs / Bytes
Achieved FLOP/s = FLOPs / kernel_duration
```

在 Thor FP8 roof 上标点 **(AI, Achieved FLOP/s)**：

| 观察 | 解读 |
|------|------|
| 点贴近 DRAM 斜线下方 | memory-bound；优化带宽、融合、量化、layout |
| `pipe_tensor` 低、`dram` 高 | Tensor Core 未吃满；layout / tile / occupancy |
| 点贴近 compute 顶且 AI 很大 | compute-bound（大 batch GEMM 才常见） |

### 5.3 整趟 LLM forward 的分层分析

| 层级 | Roofline 用法 |
|------|----------------|
| 整 engine ~63 ms | 总 FLOPs / 总 DRAM bytes → 宏观 AI |
| 单个 `myl_Fc` GEMM | 找最胖的点，看是否贴 DRAM 斜线 |
| LayerNorm 小 kernel | AI 低、FLOPs 小 → 常 **latency-bound**，非 roofline 主矛盾 |

---

## 6. Thor 特有注意点

### 6.1 统一内存

- **273 GB/s 由 CPU + GPU + 拷贝共享**，不是 GPU 独占带宽
- `trtexec --noDataTransfers` 去掉边界 H2D/DtoH，更接近 **纯 kernel DRAM roof**
- webui 真实路径若 CPU 频繁喂数，effective \(B_{\text{mem}}\) 低于 273 GB/s

### 6.2 选对 Compute Roof

```text
llm_fp8.engine + Tensor Core GEMM
  → Compute Roof = 517 TFLOP/s（Dense FP8, MAXN）
  → Memory Roof  = 273 GB/s
  → Ridge        ≈ 1894 FLOP/Byte

不要用：
  - FP4 Sparse 2070（除非真开 sparse）
  - FP32 8 TFLOP/s（除非 kernel 走 CUDA Core 而非 Tensor Core）
```

### 6.3 与 shared / occupancy 的关系

大 shared GEMM（227 KiB/block → 每 SM 仅 1 block）详见 [shared_mem.md](../hw/nvidia/shared_mem.md)：

- **低 occupancy** 不一定劣于高 occupancy；roofline 上看 **achieved FLOP/s 是否贴近 \(B \times AI\)**
- **273 GB/s 是硬顶**：π0.5 prefill 优化空间常在 **算子融合、FP8、少读写、FMHA plugin**，而非单纯堆 occupancy

---

## 7. 快速心算表（T5000，DRAM = 273 GB/s）

### 7.1 带宽利用率

| Achieved 带宽 | 占 273 GB/s |
|---------------|-------------|
| 136 GB/s | ~50% |
| 205 GB/s | ~75% |
| 245 GB/s | ~90% |

### 7.2 FP8 Tensor Core 利用率（Dense MAXN，517 TFLOP/s）

| Achieved 算力 | 占比 |
|---------------|------|
| 130 TFLOP/s | ~25% |
| 260 TFLOP/s | ~50% |
| 415 TFLOP/s | ~80% |

LLM prefill 粗经验（以 ncu 实测为准）：**DRAM 50–85% 峰值，Tensor Core 30–70% 峰值**。

### 7.3 Ridge Point 速查

```text
AI_ridge(FP8 Dense, MAXN)  = 517e12 / 273e9  ≈ 1894 FLOP/Byte
AI_ridge(FP32 CUDA, MAXN)  = 8.064e12 / 273e9 ≈ 29.5 FLOP/Byte
```

---

## 8. GEMM 运算强度估算（参考）

单精度 MatMul \(C = A \times B\)，\(A \in \mathbb{R}^{M \times K}\)，\(B \in \mathbb{R}^{K \times N}\)：

\[
\text{FLOPs} = 2MNK
\]

若各矩阵各读一次（最简模型）：

\[
\text{Bytes} \approx (MK + KN + MN) \times \text{sizeof(dtype)}
\]

\[
AI \approx \frac{2MNK}{(MK + KN + MN) \times \text{bytes}}
\]

例：\(M=968, N=16384, K=2048\)，FP8（1 byte）：

```text
FLOPs ≈ 2 × 968 × 16384 × 2048 ≈ 6.5×10^11
Bytes ≈ (968×2048 + 2048×16384 + 968×16384) ≈ 3.6×10^7
AI    ≈ 18000 FLOP/Byte（理论 upper；实际因 cache/reuse 更高 effective AI）
```

实际 kernel 的 AI 以 ncu **`dram__bytes_*`** 为准；cache hit 会使 achieved 性能 **高于** 仅用 DRAM bytes 估的斜线。

---

## 9. 常见误解

| 误解 | 正确理解 |
|------|----------|
| 离 FP8 compute 顶远 = 性能差 | batch=1 LLM 多在 memory-bound 区，应看 **是否贴近 DRAM 斜线** |
| 用 FP4 Sparse 2070 TFLOP/s 作顶 | 仅 sparse 模式；Dense FP8 应用 **517 TFLOP/s** |
| 273 GB/s 是 GPU 独占 | 统一内存，CPU / 拷贝共享总线 |
| occupancy 低 = roofline 差 | 大 tile GEMM 故意低 occupancy；看 achieved FLOP/s |
| `--noDataTransfers` 不测内存 | 只关 Host↔Device 边界拷贝；kernel DRAM 访问仍在 |

---

## 10. 一句话总结

**Jetson Thor 的 Roofline = 273 GB/s 的 DRAM 斜线 + 按精度选的 Compute 水平线。**

对 `llm_fp8.engine`：

- **Compute Roof**：517 TFLOP/s（FP8 Dense Tensor Core @ MAXN）
- **Memory Roof**：273 GB/s
- **Ridge Point**：≈ **1894 FLOP/Byte**

π0.5 batch=1 prefill 的 GEMM/FMHA 多在拐点 **左侧（memory-bound）**；优化应优先 **DRAM 流量与 Tensor Core 利用率**，而非单纯提高 occupancy。
