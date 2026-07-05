# Cosmos3 Wan VAE — Engine 文件大小 vs 运行时 Context Memory

本文档解释 **Cosmos3 Policy-DROID TRT 部署** 中 `vae_encode.engine` 的：

1. **磁盘体积 vs execution context memory**（§1–§5）
2. **trtexec layer profile 延迟热点**（§8）
3. **数值格式与热点算子原理**（Conv3D / RMS Norm / GEMM）（§9–§10）

以及如何估算、验证与优化。

相关背景：

- TRT 分 stage 部署：[`docs/models/cosmos3_trt_deploy.md`](../../models/cosmos3_trt_deploy.md)
- 显存/memory 模型：[`docs/research/edge-llm-flow.md`](../../research/edge-llm-flow.md) §3.6.2
- VAE 结构：[`docs/models/cosmos3_arch_vae.md`](../../models/cosmos3_arch_vae.md)
- build 配置：`configs/build_configs/cosmos3_policy_droid_vae_encode_build_cfg.py`

---

## 1. 现象（Jetson Thor 实测）

Policy-DROID 固定 profile 下，四 stage engine 磁盘大小与 `vae_encode` profiling 日志如下：

```text
engines/
  dit.engine         28G
  text_embed.engine  1.2G
  vae_decode.engine  1.1G
  vae_encode.engine  297M

trtexec --loadEngine=.../vae_encode.engine --exportProfile=./vae_profile.json ...
  Input:  video         1×3×17×480×832
  Output: vision_latent 1×48×5×30×52
  Created execution context with device memory size: 2358.28 MiB
```

**疑问**：engine 文件只有 ~300MB，为何 context memory 需要 ~2.3GB？

---

## 2. 两个数字分别是什么

| 指标 | vae_encode 实测 | 含义 |
|------|-----------------|------|
| **Engine 文件 ~297MB** | 磁盘序列化产物 | 主要是 **BF16 权重** + TensorRT 编译后的 layer / kernel 元数据 |
| **Context memory ~2358 MiB** | `IExecutionContext` 分配大小 | GPU 上 **单次 forward** 所需运行时显存：中间 **激活（activation）** + conv **scratch workspace** + 部分持久 buffer |

TensorRT 日志中的 **2358 MiB 通常不含 engine 权重本身**（权重在 `deserializeCudaEngine` 时另外占用 GPU）。

实际跑 `vae_encode` 时，峰值显存粗算为：

```text
总显存 ≈ engine 权重 (~300MB) + context (~2.3GB) + I/O binding (~40MB) ≈ 2.6GB+
```

**结论：297MB vs 2.3GB 并不矛盾**——前者是「模型有多重」，后者是「一次 forward 中间要开多大的临时空间」。

---

## 3. 为什么 vae_encode 的激活远大于权重

### 3.1 输入 / 输出体积极不对称

Policy-DROID profile（`shapes.POLICY_DROID`）：

```text
video:         [1, 3, 17, 480, 832]   ≈  40 MB (BF16)
vision_latent: [1, 48,  5,  30,  52]  ≈  0.7 MB (BF16)
```

输出 latent 极小，但 **encoder 必须在全分辨率、多帧的 3D 特征图上做因果卷积**。中间激活随 `T × H × W × C` 增长，与最终输出体积无关。

### 3.2 Wan VAE Encoder 结构

`AutoencoderKLWan` / `WanEncoder3d`（diffusers）典型配置：

- `base_dim=96`，`dim_mult=[1, 2, 4, 4]` → 通道 96 → 192 → 384 → 384
- 空间下采样 ×4（`scale_factor_spatial=16`），时间下采样（`temperal_downsample`）
- 残差块 + 因果 3D 卷积（`WanCausalConv3d`）

**前几层**仍在 **17 帧 × 480 × 832** 上计算。单层 BF16 激活粗算：

```text
96  × 17 × 480 × 832 × 2B  ≈  1.2 GB   # 单张 feature map
384 × 17 × 240 × 416 × 2B  ≈  1.2 GB   # 下采样后仍很大
384 × 17 × 120 × 208 × 2B  ≈  0.3 GB
```

### 3.3 峰值显存的叠加因素

1. **残差连接**：同一时刻需保留 `x` 与 `x + f(x)`，峰值接近 2×。
2. **TRT 层融合有限**：部分中间 tensor 并行存活，峰值 = 同时存活 tensor 之和，而非单 tensor 大小。
3. **Conv3D workspace**：build 时 `workspace_mb=8192`（见 build_cfg）允许 TRT 选择更耗显存、更快的 cuDNN 算法；runtime scratch 被烘焙进 context size。
4. **Profiler 开销**：`trtexec --exportProfile` 会引入额外同步与 profiler buffer（日志会提示 e2e timing 不准确）。

因此 **输入 40MB → 中间激活 1~2GB+** 对 3D CNN 是预期行为，不是 leak 或 build 错误。

---

## 4. 与其他 stage 对比

| Stage | Engine 文件 | 主导因素 | Context 特点 |
|-------|-------------|----------|--------------|
| `vae_encode` | ~297MB | 权小、**激活大** | ~2.3GB（CNN 典型） |
| `vae_decode` | ~1.1GB | 权重 + 上采样激活 | 通常 ≥ encode |
| `text_embed` | ~1.2GB | 主要是权重 | 相对可控 |
| `dit` | ~28GB | **16B MoT 权重** | 权重主导；context 另计 |

dit 28GB 主要是 **MoT DiT 参数量**；vae_encode 权重小，但 **spatial-temporal 卷积激活随 H×W×T 超线性增长**，故 context 相对 engine 文件显得「不成比例地大」。

---

## 5. Cosmos3 四 stage 显存预算

Chamleon 串行调度下，context **可复用**同一块 device buffer；权重须分别加载：

```text
峰值显存 ≈ Σ(各 engine 权重) + max(各 stage context) + I/O buffer + host 开销
```

Policy-DROID 在 Thor 上的主要瓶颈通常是 **dit 28GB 权重**；vae_encode 的 ~2.6GB 总量在单次 policy infer 中只出现 **1 次**（去噪环内不再调用 vae_encode）。

内存类型划分（见 edge-llm-flow §3.6.2）：

| 类型 | vae_encode 对应物 | 生命周期 |
|------|-------------------|----------|
| 权重 | engine 内 constant | 进程级（deserialize 后常驻） |
| Context / scratch | 2.3GB device memory | 单次 `enqueue`；多 stage 可串行共享 |
| I/O binding | video / vision_latent buffer | stage 间 handoff |

---

## 6. 验证方法

```bash
# Layer 信息与 context size
trtexec --loadEngine=output/cosmos3_policy_droid_trt/engines/vae_encode.engine --dumpLayerInfo

# 不带 profiler 的纯推理（更接近生产显存）
trtexec --loadEngine=output/cosmos3_policy_droid_trt/engines/vae_encode.engine \
  --warmUp=10 --iterations=10
# 另开终端: nvidia-smi 观察实际占用
```

Chamleon workflow 的 `trt_profile` action 会导出 layer timing；若只关心显存，优先用无 `--exportProfile` 的 `trtexec` 或 `nvidia-smi`。

Layer 级延迟 breakdown 与优化优先级见 **§8**。

---

## 7. 优化方向（v2）

| 手段 | 效果 | 代价 |
|------|------|------|
| 降低 `workspace_mb`（如 8192→2048） | 减小 runtime scratch | 可能变慢；需 rebuild |
| 降低 canvas / 帧数 | 同时减 Conv FLOPs 与 RMS/SiLU 字节（收益最大，见 §10.3） | 需 re-export + rebuild 全 stage |
| VAE FP8 / NVFP4 量化 | **Conv：抬 Tensor Core 算力顶**（2×/4×）；RMS/SiLU：降字节 | ModelOpt 路径；需校准与精度验证 |
| 串行 stage + 共享 context memory | 峰值不叠加各 stage context | 已实现于编排设计；需 runtime 侧 `setDeviceMemory` |
| Gradient checkpointing（训练侧） | 不适用 TRT 推理 | — |

当前 v1 部署为 **bf16 固定 profile**；VAE 量化见 Model-Optimizer 与 [`docs/quantize/nvfp4.md`](../../quantize/nvfp4.md) 中的 Wan Conv3D 量化说明。

**延迟热点与 layer 级 breakdown** 见下文 **§8**（Thor 上 `trtexec --exportProfile` 实测）。

---

## 8. trtexec Layer Profile 分析（Jetson Thor）

### 8.1 数据来源

```bash
trtexec --loadEngine=output/cosmos3_policy_droid_trt/engines/vae_encode.engine \
  --exportProfile=./vae_profile.json --warmUp=10 --iterations=10
```

- Profile 文件：`vae_profile.json`（358 个 layer/kernel 条目）
- Engine：`vae_encode.engine`（BF16，build_cfg `workspace_mb=8192`）
- 输入 / 输出：`video [1,3,17,480,832]` → `vision_latent [1,48,5,30,52]`
- 迭代次数：**11**（JSON 首条 `{ "count": 11 }`）

> `--exportProfile` 会引入额外 GPU 同步，**e2e 计时略偏高**。精确 latency 请用
> `--separateProfileRun` 或去掉 `--exportProfile` 单独测（见 §8.7）。

### 8.2 总体性能

| 指标 | 数值 |
|------|------|
| Layer 时间总和（11 iter） | 5060 ms |
| **估算 e2e / iter**（layer 求和） | **~460 ms** |
| Profile 层 / kernel 数 | **358**（≈358 次 launch/iter） |
| Conv 条目 | 149 |
| 平均单次 Conv launch | ~1.65 ms |

Policy 链路中 vae_encode 只跑 **1 次**；dit 去噪环（~30 步）通常是算力主瓶颈，但单次 encode ~460ms 仍不可忽视。

### 8.3 按算子类型

| 类型 | 占比 | ms/iter | 识别特征 |
|------|------|---------|----------|
| **Conv3D** | **53.6%** | ~247 | `/encoder/.../Conv_myl*` |
| **RMS Norm** | **27.8%** | ~128 | `MulSumSqrtMax`、`TranReshMoveMean`（WanRMS_norm） |
| **Elemwise 融合** | **18.0%** | ~83 | `ReplDivMulMulAddSilu...`（SiLU + 残差 + scale） |
| Layout | 0.6% | ~3 | `Resh` / `Tran` / `Move` / `Slic` |
| Head / quant | <0.1% | <1 | `conv_out`、`quant_conv`、`SlicAddMul`（归一化已 bake 进 export） |

Head 与 latent 归一化几乎不占时间；热点在 **encoder 主体**。

### 8.4 按分辨率阶段

| 阶段 | 占比 | ms/iter | 空间尺度（约） |
|------|------|---------|----------------|
| **L0 `down_blocks_0`** | **23.6%** | ~109 | 全分辨率 480×832 |
| **Downsample RMS 融合** | **24.2%** | ~111 | 下采样路径上的 WanRMS_norm |
| **L1 `down_blocks_1`** | 17.8% | ~82 | 1/2 分辨率 |
| **ResBlock 融合（L0/L1）** | 16.8% | ~77 | SiLU + RMS + 残差 |
| L2 `down_blocks_2` | 7.4% | ~34 | 1/4 分辨率 |
| L3 / mid / conv_in | <5% | | 已充分下采样 |

**约 48% 时间耗在全分辨率阶段**（L0 Conv + 下采样 RMS），与 §3 中 GB 级 activation 分析一致。

Top RMS 层（各 ~23–28 ms/iter，合计 ~112 ms/iter ≈ 22%）：

```text
__myl_TranReshMoveMean_myl0_56   avg ≈ 28.1 ms
__myl_TranReshMoveMean_myl0_64   avg ≈ 26.7 ms
__myl_TranReshMoveMean_myl0_52   avg ≈ 24.5 ms
__myl_TranReshMoveMean_myl0_60   avg ≈ 22.9 ms
```

Top Conv 层（L0 resnet，每块 4 份 tiled conv 各 ~6 ms/iter）：

```text
/encoder/down_blocks_0/resnets_0/conv2_*   ~6.4 ms/iter × 4  ≈ 25.6 ms/iter（该 resblock）
/encoder/down_blocks_0/resnets_0/conv1_*   ~6.0 ms/iter × 4  ≈ 24.0 ms/iter
```

### 8.5 因果卷积时间展开（kernel 碎片化）

Wan VAE 使用 **因果 3D 卷积 + feat_cache**（`WanCausalConv3d`，`CACHE_T=2`，见 diffusers
`autoencoder_kl_wan.py`）。ONNX/TRT 导出时将 cache 逻辑**展开**，同一逻辑层对应多份 Conv：

```text
conv1/Conv                 ← 主路径
conv1_1/Conv … conv1_4/Conv   ← 时间维分块（17 帧 → 4–5 块）
```

Profile 统计：

- **28 组** Conv 各有 **4+** 个 tiled 变体
- 例：`down_blocks_0/resnets_0/conv2` 四份合计 **~282 ms / 11 iter ≈ 25.6 ms/iter**
- 带来 **358 次 kernel launch/iter** 的开销与算子碎片化

这是 TRT 处理动态 cache 的通用代价；自定义 CausalConv3d Plugin 是长期优化方向。

### 8.6 瓶颈性质：Conv 算力密集 + RMS/SiLU 访存密集（混合）

> **重要更正**：早期版本笼统写「整体 memory-bound」并不准确。经精确估算（见 §10.3），
> 卷积其实是 **compute-bound**；真正 memory-bound 的是 RMS norm 与 SiLU。大 activation
> 容量（2.3GB context）**≠** DRAM 带宽是瓶颈。

L0 单层 feature map（C=96, T=17, H=480, W=832, BF16）：

```text
输入 video：              1×3×17×480×832×2B  ≈  40 MB
单层 L0 feature map：     1×96×17×480×832×2B ≈ 1.3 GB
```

按算子类型区分瓶颈（数据见 §10.3、§10.6）：

| 算子 | Profile 占比 | 算术强度 AI | 真实瓶颈 |
|------|--------------|-------------|----------|
| **Conv3D** | ~54% | **~1588 FLOP/B**（> BF16 ridge 945） | **算力**（profile 反推 ~56% BF16 峰值利用） |
| **RMS norm** | ~28% | ~O(1) | **带宽**（FP32 reduce，ridge 仅 ~30） |
| **SiLU/elemwise** | ~18% | ~O(1) | **带宽** |

- 卷积 FLOPs 巨大（**整 encoder ~35 TFLOP/iter**），且高算术强度 → **compute-bound**。
- RMS + SiLU 近半占比、低算术强度 → **memory-bound**，是「访存密集」印象的真正来源。
- 2.3GB context（§2）是 **activation 容量**，不代表带宽受限；profile 期卷积仅用 ~33% 带宽。

### 8.7 延迟优化方案（按预期收益）

#### P0 — 输入侧（需 re-export，收益最大）

| 方案 | 预期 | 说明 |
|------|------|------|
| **降低 canvas 分辨率** | L0 近似按 H×W；边长 ÷2 → 时间约 ÷4 | 换 `resolution_tier`；需与训练分布对齐 |
| **减少 `num_frames`** | 随 T 近似线性 | Policy 主要用 frame0 条件；17 帧为 MoT 对齐；改 profile 需质量验证 |
| **Crop 替代 full canvas** | 显著减 activation | v1 为 TRT shape 恒定用 full-canvas；v2 可评估 |

#### P1 — TRT build / runtime（不改语义）

| 方案 | 预期 | 说明 |
|------|------|------|
| **CUDA Graph** | launch 开销，约 **5–15%** | `deploy.use_cudagraph: true` 后 rebuild；358 kernel 场景适用 |
| **重跑 build tactic** | 不定 | 对比 `workspace_mb` 4096/8192；或 builder optimization level |
| **FP16 engine A/B** | **通常不会**比 BF16 更快 | 两者均为 **16 bit/元素**，DRAM 流量相同；详见 **§9** |
| **FP32 → FP16/BF16** | 激活+权重字节数 **÷2** | 相对 FP32 才有「带宽减半」；当前 v1 已是 16-bit |
| **纯 TRT layer fusion** | 有限 | 已有大量 `ReplDivMulMulAddSilu` 融合；边际收益小 |

#### P2 — 模型 / 导出（工程量大）

| 方案 | 预期 | 说明 |
|------|------|------|
| **CausalConv3d TRT Plugin** | 消除 conv_1..4 展开 | 合并时间块，减 kernel 数与 buffer |
| **NVFP4 / INT8 量化 Conv3D** | 权重 4–8×、激活 2–4× 字节下降 | ModelOpt；需 calibration；**这才是相对 BF16 的真正降带宽手段** |
| **简化 VAE 结构** | 最大 | 改语义，仅研究方向 |

#### P3 — 部署策略

| 方案 | 说明 |
|------|------|
| **Cache z0** | 观测不变时跳过重复 vae_encode |
| **vae_encode ∥ text_embed** | 多 stream 并行（显存允许时） |
| **共享 context memory** | 不加速单次 encode，降峰值显存（§5） |

### 8.8 建议实验命令

```bash
# 1) 纯 latency（无 profiler 同步开销）
trtexec --loadEngine=output/cosmos3_policy_droid_trt/engines/vae_encode.engine \
  --shapes=video:1x3x17x480x832 --warmUp=20 --iterations=100

# 2) Layer profile 与 e2e 分离（推荐）
trtexec --loadEngine=.../vae_encode.engine \
  --exportProfile=./vae_profile.json --separateProfileRun \
  --warmUp=10 --iterations=10

# 3) CUDA Graph：deploy.yaml use_cudagraph: true → chameleon compile → 重复 (1)

# 4) 带宽验证（memory-bound 假设）
nsys profile trtexec --loadEngine=.../vae_encode.engine --iterations=10
```

**优先实验**：相同 shape 下对比 **CUDA Graph on/off** 的 e2e ms。若收益 <5%，主要矛盾在 L0 全分辨率 Conv3D 带宽，应转向 **降分辨率 / INT8·NVFP4 量化**（P0/P2），而非 BF16↔FP16 互换。

---

## 9. 数值格式与 DRAM 带宽

### 9.1 各 dtype 存储宽度

| 格式 | bit / 元素 | 字节 / 元素 | 相对 BF16 的激活+权重流量 |
|------|------------|-------------|---------------------------|
| FP32 | 32 | 4 B | **2×** |
| **BF16** | 16 | 2 B | **1×（基准）** |
| **FP16** | 16 | 2 B | **1×（与 BF16 相同）** |
| FP16/BF16 混合 | 16 | 2 B | 1× |
| INT8 | 8 | 1 B | **0.5×** |
| NVFP4 | 4 | 0.5 B | **0.25×** |

BF16 与 FP16 都是 **16 bit 浮点**，区别在 **指数/尾数分配**（BF16：8 bit 指数对齐 FP32，4 bit 尾数；FP16：5 bit 指数，10 bit 尾数）。**磁盘上的 engine 体积、DRAM 读写字节数、activation tensor 大小完全一致**。

因此文档 §8.7 中「BF16 → FP16 带宽减半」是**错误表述**（已修正）：**不存在**因位宽变化带来的 2× 加速。

### 9.2 BF16 ↔ FP16 A/B 还可能有什么差别？

在字节流量相同的前提下，换 precision flag Rebuild 后，最多出现**小幅、不确定**差异：

| 维度 | 说明 |
|------|------|
| TRT / cuDNN **kernel 选型** | 不同 dtype 标签可能走到不同 tactic（是否用 Tensor Core、tiling 策略） |
| 硬件 **峰值算力表** | 部分 GPU 对 FP16/BF16 Tensor Core 峰值不同；但 **memory-bound** 时算力常不是瓶颈 |
| **数值行为** | VAE 深链路上 BF16 动态范围更大；FP16 可能精度更差或偶然更快，需质量+A/B |
| WanRMS_norm | 源码中 BF16/FP16 输入会 **先 cast 到 FP32 做 normalize**（见 §10.2），两者内部 reduce 路径一样 |

**结论**：FP16 A/B 属于低优先级实验；**真正降带宽**靠 INT8/NVFP4（位宽变小）或减小 H×W×T。

### 9.3 当前 v1（BF16）与量化路径对照

```text
v1 bf16 engine     → 2 B/elem，profile ~460 ms/iter，context ~2.3 GB
FP16 rebuild       → 仍 2 B/elem，不应期待显著加速
INT8 Conv3D        → ~1 B/elem 权重+激活，理论 DRAM ~÷2（含 scale 开销）
NVFP4              → ~0.5 B/elem 权重，激活 8bit/4bit 混合，理论 DRAM 更低
```

---

## 10. 热点算子原理：Conv3D 与 RMS Norm

本节对应 §8 profile 中 **53.6% Conv3D** 与 **27.8% RMS Norm**，源码见 diffusers
`autoencoder_kl_wan.py`（`WanCausalConv3d`、`WanRMS_norm`、`WanResidualBlock`）。

### 10.1 张量约定（Policy-DROID encode）

全程 layout：**NCTHW** = `[B, C, T, H, W]`（batch, channel, time, height, width）。

| 位置 | 典型 shape（L0） | 说明 |
|------|------------------|------|
| 输入 `video` | `[1, 3, 17, 480, 832]` | RGB，17 帧 canvas |
| `conv_in` 后 | `[1, 96, 17, 480, 832]` | base_dim=96 |
| L0 resblock 内 | `[1, 96→384, 17, 480, 832]` | 通道渐增，空间仍全分辨率 |
| L1 下采样后 | `[1, 384, 17, 240, 416]` | 空间 ÷2 |
| 最终 latent | `[1, 48, 5, 30, 52]` | 时间+空间均下采样 |

通道阶梯（`dim_mult=[1,2,4,4]`，`base_dim=96`）：**96 → 192 → 384 → 384**。

### 10.2 WanRMS_norm：计算过程（非 GEMM）

#### 数学定义

对 **channel-first** 视频特征 `x ∈ R^{B×C×T×H×W}`，`WanRMS_norm` 在 **通道维 C** 上归一化（`dim=1`），再乘可学习缩放：

```text
# 对每个固定 (b, t, h, w)，令 v = x[b, :, t, h, w] ∈ R^C
v_norm = v / ||v||_2          # F.normalize，L2 归一化（非 LayerNorm 的减均值）
y[b,c,t,h,w] = v_norm[c] * sqrt(C) * gamma[c] + bias[c]
```

`gamma` shape 为 `(C, 1, 1, 1)`（`images=False` 时），在 T,H,W 上 broadcast。

与 **LayerNorm** 不同：RMS/L2 norm **不减均值**，只做范数缩放；**没有**全连接 GEMM。

#### 源码路径

```python
# autoencoder_kl_wan.py — WanRMS_norm.forward
needs_fp32_normalize = x.dtype in (float16, bfloat16, ...)
normalized = F.normalize(x.float() if needs_fp32_normalize else x, dim=1).to(x.dtype)
return normalized * self.scale * self.gamma + self.bias   # scale = sqrt(dim)
```

**BF16/FP16 推理的关键点**：normalize 在 **FP32** 中完成，再 cast 回 BF16。一次 RMS 的 DRAM 流量粗算：

```text
读 x (BF16)           C×T×H×W × 2B
写/读 FP32 临时       C×T×H×W × 4B   ← 隐式扩容
reduce/sqrt/div       遍历 C 维
写 y (BF16)           C×T×H×W × 2B
```

算术强度极低（每元素 O(1) FLOP，多次内存 pass），故 profile 中 RMS 占 **~28%** 且与 Conv 一样是 **memory-bound**。

#### TRT profile 中的名字

| Profile 子串 | 对应操作 |
|--------------|----------|
| `MulSumSqrtMax` / `MulSumSqrtMaxMin` | 平方 → 沿 C 求和 → sqrt → 归一化除法 |
| `TranReshMoveMean` | layout 变换 + reduce（大 tensor 上尤其慢，§8.4 Top 4 各 ~24–28 ms/iter） |
| `ReplDivMulMulAddSilu` | 常与 **norm + SiLU + 残差** 融合在同一 mega-kernel |

每个 `WanResidualBlock` 含 **norm1、norm2** 两次 RMS；encoder 堆叠数十次，RMS 总成本显著。

### 10.3 WanCausalConv3d：计算过程

#### 标准 Conv3D

给定输入 `x[B,C_in,T,H,W]`、权重 `W[C_out,C_in,kT,kH,kW]`、偏置 `b[C_out]`：

```text
y[b, c_out, t', h', w'] = b[c_out]
    + Σ_{c_in, kt, kh, kw}
      W[c_out, c_in, kt, kh, kw] · x[b, c_in, t'+·, h'+·, w'+·]
```

其中 `·` 由 stride / padding / dilation 决定输出坐标。Wan 主体 kernel 为 **3×3×3**（`kT=kH=kW=3`），`padding=1`（空间对称），**时间维因果**。

#### 因果（Causal）padding

`WanCausalConv3d` 将时间维 padding 改为 **只看过去帧**：

```text
标准 symmetric pad (T)  →  因果 pad：仅在 t 负方向补零 / 补 cache
self._padding = (W左, W右, H上, H下, 2×T_pad, 0)   # 过去侧 2×pad，未来侧 0
```

若提供 `cache_x`（上一时间块的末帧特征），则 **concat 到 t 维前面** 再 pad，保证 streaming 推理与整段 17 帧一次 encode **数学等价**。

Policy TRT **静态一次 encode** 时，export 仍把 cache 逻辑 **展开** 为 `conv_1 … conv_4` 多个固定 Conv 节点（§8.5），产生额外 launch，不是算法多算，而是 **图展开代价**。

#### Conv3D 与 GEMM 的关系

cuDNN / TensorRT 内部常把 Conv 化为 **GEMM**（通用矩阵乘）：

```text
im2col（或 implicit GEMM）:

  输入 patch 矩阵 A  :  M × K
    M = B · T' · H' · W'         （每个输出时空点一行）
    K = C_in · kT · kH · kW      （每个 patch 拉成向量）

  权重矩阵 B_w       :  K × N
    N = C_out

  输出               :  C = A · B_w     （M × N，即 y  reshape 后）
```

对 L0 一个 `conv1`（96→96，3³，T=17,H=480,W=832）粗算：

```text
GEMM 维度:
  M = C_out              = 96          （或把输出通道并入下式）
  M' = T·H·W             ≈ 6.79×10^6   （输出时空点数, B=1）
  K = C_in·k³ = 96·27    ≈ 2.59×10^3
  N = C_out              = 96

FLOPs = 2 · M'·K·N = 2 × 6.79e6 × 2592 × 96 ≈ 3.38×10^12  (~3.4 TFLOP)
激活读写(下界) ≈ 2 × 1.3 GB = 2.6 GB（输入+输出 feature map，BF16）
权重 ≈ 0.5 MB（可忽略）
```

> ⚠️ 早期版本误写「~330 GFLOP、AI≈130」——**漏乘了输出通道 `N=C_out=96`**。
> 正确值为 **~3.4 TFLOP**、AI **~1300**，见下。

**运算强度** AI = FLOPs / Bytes ≈ 3.38e12 / 2.6e9 ≈ **1300 FLOP/B**。

对比 Thor（273 GB/s）ridge point：

| 路径 | 峰值 | ridge | conv AI≈1300–1600 |
|------|------|-------|-------------------|
| FP32 CUDA | 8 TFLOP/s | ~30 | 远右 → compute |
| **BF16 TC（估 258T）** | 258 TFLOP/s | **~945** | **右侧 → compute-bound** |
| FP8 TC | 517 TFLOP/s | ~1894 | 接近平衡 |

**卷积是 compute-bound**（AI > BF16 ridge），不是 memory-bound。3×3×3 核每个输入像素被
`k³=27` 个输出复用、通道维 `C²` 进一步复用 → 天然高算术强度（与 LLM batch=1 瘦 GEMM 的
低 AI 相反）。

#### 整个 encoder 汇总（per iter，BF16，理论）

| 量 | 数值 |
|----|------|
| 卷积总 FLOPs | **≈ 35.4 TFLOP** |
| 卷积理论访存（下界） | ≈ 22.3 GB |
| 理论 AI | ≈ 1588 FLOP/B |

用 §8 profile 实测反推（conv ~247 ms/iter）：

```text
achieved = 35.4e12 / 0.247 ≈ 143 TFLOP/s  ≈ 56% BF16 峰值   ← 算力高利用
带宽     = 22.3 GB / 0.247 ≈ 90 GB/s       ≈ 33% 带宽顶     ← 远未打满带宽
L0 单独:  14.4 TFLOP/iter / 109ms ≈ 132 TFLOP/s (~51% 峰值)
```

**算力利用 ~56% 而带宽只 ~33% → 卷积时间花在乘加上，compute-bound 实锤。**

Tensor Core 路径会把 GEMM 放在 16×16×16 tile 上；im2col 展开会增大临时 buffer（计入
context memory §2），且浅通道（L0=96）/ kernel 碎片（§8.5）会拉低实际利用率，但主导仍是算力。

#### 输入 / 输出小结（单个 Conv3D 层）

| | Shape（L0 例） | 字节 (BF16) |
|--|----------------|-------------|
| 输入 x | `[1, 96, 17, 480, 832]` | ~1.3 GB |
| 权重 W | `[96, 96, 3, 3, 3]` | ~0.5 MB |
| 偏置 | `[96]` | 可忽略 |
| 输出 y | `[1, 96, 17, 480, 832]` | ~1.3 GB |
| workspace | cuDNN 算法相关 | 可达数百 MB（build 时 `workspace_mb` 上限） |

权重很小、激活巨大——这与 **engine 297MB vs context 2.3GB**（§1）一致。

### 10.4 WanResidualBlock：Conv 与 RMS 如何串联

```text
输入 x [B,C_in,T,H,W]
  │
  ├─ conv_shortcut (1×1×1 CausalConv3d) ──────────────────────────────┐
  │                                                                    │
  └─ norm1 (RMS) → SiLU → conv1 (3³ CausalConv3d)                     │
              → norm2 (RMS) → SiLU → Dropout → conv2 (3³)             │
                                                                    (+) → 输出
```

- **2× RMS + 2× Conv3D（3³）+ 1× shortcut Conv（可能 1³）** 每个 resblock
- SiLU：`x * sigmoid(x)`，elementwise，profile 中并入 `ReplDivMulMulAddSilu` 融合 kernel
- 残差 `x + h` 需 **同时持有** 主路径与 shortcut 输出 → 峰值 activation ≈ 2×（§3.3）

L0 每个 resblock 4 份 tiled conv（时间展开）× 2 个 conv 层 ≈ profile 中看到的 `conv1_1…conv1_4`、`conv2_1…conv2_4`。

### 10.5 下采样路径（Downsample RMS 为何特别慢）

`WanResample(mode="downsample3d")`：

```text
x [B,C,T,H,W]
  → permute+reshape → [B·T, C, H, W]
  → ZeroPad2d + Conv2d stride 2        # 空间 ÷2
  → reshape 回 [B,C,T,H/2,W/2]
  → time_conv: CausalConv3d stride (2,1,1)  # 时间维可选 ÷2
  → 前后各有 WanRMS_norm
```

空间下采样前后 feature map 仍大（例如 480×832→240×416，T 仍为 17），RMS 在 **大 H×W** 上全通道 reduce → §8.4 中 4 个 `TranReshMoveMean` 各 ~25 ms/iter，合计 **~22%** 总时间。

### 10.6 算子 ↔ Profile ↔ 优化映射

| 算子 | 是否 GEMM | Profile 占比 | AI | 主要瓶颈 | 有效优化 |
|------|-----------|--------------|----|----------|----------|
| CausalConv3d 3³ | 内部 im2col+GEMM | ~54% | ~1300–1600 | **算力**（~56% BF16 峰值） | 减 FLOPs：降分辨率/帧数；**FP8/NVFP4 抬算力顶**；减 kernel 碎片 |
| WanRMS_norm | **否**（reduce+scale） | ~28% | ~O(1) | **带宽**（FP32 双遍读写） | 量化降字节；降 spatial size；融合 |
| SiLU + 残差 | 否 | ~18% | ~O(1) | **带宽**（elementwise） | TRT 已融合；边际有限 |
| Layout | 否 | <1% | — | — | — |

**关键区分**：conv 是 compute-bound，**BF16→FP16 无用**（算力峰值相同）；FP8/NVFP4 有用是因为
**Tensor Core 峰值 2×/4×**（抬 compute roof），**不是**省带宽。RMS/SiLU 才靠降字节（量化/降分辨率）。

**mid_block** 含 `WanAttentionBlock`：reshape 为 `[B·T, C, H, W]` 后 `Conv2d 1×1` 做 QKV（**小 GEMM**）+ attention；Policy profile 中 mid 仅 **~1.7%**，不是热点。

### 10.7 与 Roofline 的直觉对照

参见 [`docs/optimizer/thor_roofline.md`](../thor_roofline.md)：

```text
Conv3D:     大 FLOPs、大 Bytes、AI≈1588  → ridge 右侧（BF16）→ compute roof 附近
RMS Norm:   极低 FLOPs、大 Bytes、AI≈O(1) → ridge 左侧 → memory roof（斜线）上
```

- **Conv**：受 **compute roof** 限制。降 H×W×T 减 FLOPs（时间下移）；FP8/NVFP4 **抬高 roof**（Tensor Core 峰值更高）。
- **RMS/SiLU**：受 **memory roof** 限制。降 H×W×T 或量化减 Bytes（沿斜线左移变快）。
- **BF16→FP16**：既不改 Bytes 也不改 16-bit 算力峰值，Roofline 上点基本不动 → 无收益。

---

## 11. VAE 3D 因果卷积 vs ViT / DiT 注意力

`vae_encode` 的 encoder（`WanEncoder3d`）是 **纯 3D CNN**（仅 mid_block 末尾一个轻量
attention），而其后的 `dit` 才是 ViT-like 的 patch + 自注意力。理解两者差异有助于
判断优化手段是否通用（结论：基本不通用）。

### 11.1 两条视觉处理路线

```text
video [1,3,17,480,832]                     latent [1,48,5,30,52]
  │  VAE encoder（CNN）                        │  DiT（Transformer）
  ▼                                            ▼
conv_in 3×3×3 CausalConv3d                  patchify 2×2 → token 序列
down_blocks ×4（残差 3D conv + 下采样）      [自注意力 + FFN] × N 层
mid_block（3D conv + 1 个 attention）        vision ↔ text ↔ action 跨模态
conv_out → latent                            → velocity
```

**同一 pipeline 里 VAE=CNN、DiT=Transformer** 是刻意分工（见 §11.5）。

### 11.2 符号含义（复杂度 / 输入形式公式）

下表汇总本节及 §10 公式中出现的所有符号：

| 符号 | 全称 | Policy-DROID L0 典型值 | 含义 |
|------|------|------------------------|------|
| `B` | Batch size | 1 | 一次处理几段视频 |
| `C` | Channels | 96 / 192 / 384（逐层增） | 每个时空点的特征向量维度（channel-first 放第 2 维） |
| `C_in` / `C_out` | 输入/输出通道 | 96→96、96→192… | 卷积前后通道数 |
| `T` | Time（帧数） | 17（latent 后 5） | 视频时间长度 |
| `H` | Height | 480（latent 后 30） | 特征图高 |
| `W` | Width | 832（latent 后 52） | 特征图宽 |
| `k` | Kernel size（每维） | 3 | 卷积核边长；3D 核体积 = `k×k×k` |
| `N` | Number of tokens | `T·H_p·W_p`（DiT 里数千） | 序列化后 token 个数 |
| `H_p` / `W_p` | Patch 网格高/宽 | `H/2`、`W/2`（patch=2） | patchify 后的空间网格尺寸 |

### 11.3 复杂度公式逐项拆解

#### Conv3D：`O(T·H·W · C² · k³)` — 线性于像素数

卷积输出有 `B·C_out·T·H·W` 个元素，算**一个**输出值需 `C_in·k³` 次乘加（MAC）：

```text
总 MAC = (输出元素数) × (每输出的乘加数)
       = B·C_out·T·H·W  ×  C_in·k³
令 B=1, C_in≈C_out=C:
       = O( T·H·W · C² · k³ )
           └像素数┘ └通道平方┘└核体积┘
```

| 因子 | 来源 |
|------|------|
| `T·H·W` | 输出时空点数——每个位置都卷一次 |
| `C²` | `C_in→C_out`，输出每通道对所有输入通道求和 → `C_in×C_out≈C²` |
| `k³` | 3D 核 `k×k×k` 个权重都参与乘加 |

**关键**：`C`、`k` 是固定小常数（k=3，C≤384），改分辨率时只有 `T·H·W` 变
→ **FLOPs ∝ 像素总数**，边长翻倍则 H·W×4、计算量×4（**线性**，不爆炸）。这就是
§8.7「降分辨率/帧数收益最大」的数学依据。

#### Attention：`O(N² · C)` — 二次于 token 数

自注意力算**每个 token 对每个 token** 的相关性：

```text
A = Q·Kᵀ   Q:[N,C], K:[N,C] → A:[N,N]   需 N×N 个内积，每个 C 次乘加 → O(N²·C)
A·V        [N,N]·[N,C] → [N,C]                                     → O(N²·C)
```

那个 `N×N` 矩阵是**二次项**来源：token 数翻倍，计算量与显存 ×4。

**为何不能在全分辨率像素上跑 attention**：`480×832×17 ≈ 6.8M` 个位置，`N² ≈ 4.6×10¹³`
直接爆炸——必须先用 VAE（线性复杂度）压成 latent，再在小 `N` 上做 attention。

### 11.4 输入形式：网格张量 vs 序列张量

| | `[B, C, T, H, W]`（卷积网格） | `[N_tokens, C]`（注意力序列） |
|--|-------------------------------|-------------------------------|
| 维度 | 5D | 2D（+batch=3D `[B,N,C]`） |
| 空间/时间邻接 | **显式保留**（相邻元素语义相邻） | 拍平后**丢失**，靠位置编码补回 |
| 位置信息 | 隐含在卷积局部结构 | 需显式 RoPE / 绝对位置编码 |
| 置换敏感 | 是（不能打乱，卷积需知邻接） | 否（attention 对 token 置换等变） |
| 谁在用 | VAE / CNN | ViT / DiT |
| 遍历单位 | 时空格点 `[b,:,t,h,w]` | token |

patchify 即两者的转换：

```text
[B, C, T, H, W]  ──每 2×2 空间块合成 1 token──►  [B, N, C']，N = T·(H/2)·(W/2)
```

### 11.5 核心区别对照

| 维度 | Wan VAE 3D 因果卷积 | ViT / DiT 自注意力 |
|------|---------------------|--------------------|
| 基本操作 | 局部卷积核滑窗 | 全局 token 两两注意力 |
| 感受野 | 局部（3×3×3），靠堆叠扩大 | 一层即全局 |
| 权重 | 卷积核**所有位置共享**（96×96×27 很小） | QKV 投影 + 每对 token 算分数 |
| 复杂度 | `O(T·H·W·C²·k³)`，线性于像素 | `O(N²·C)`，二次于 token |
| 输入形式 | 保持 `[B,C,T,H,W]` 网格 | 拉平成 `[N,C]` 序列 |
| 位置信息 | 隐含局部结构 | 需显式位置编码 |
| 时间因果 | **因果卷积**（padding 只补过去帧） | attention mask 屏蔽未来 |
| 归纳偏置 | 强（局部性、平移等变） | 弱（靠数据学） |
| 典型算子 | im2col→GEMM + reduce | 大 GEMM（QKV、AV、FFN） |

### 11.6 因果卷积特殊在哪

普通 3D 卷积时间维**对称**：算第 t 帧看 `[t-1,t,t+1]`（**看未来**）。世界模型要求
不能看未来帧，`WanCausalConv3d` 只在时间维**过去侧** padding：

```python
# autoencoder_kl_wan.py
self._padding = (W左, W右, H上, H下, 2*T_pad, 0)  # 时间维：过去 2×pad，未来 0
self.padding = (0, 0, 0)                          # 关掉 nn.Conv3d 自带对称 pad
```

```text
普通 Conv3D:  y[t] = f(x[t-1], x[t], x[t+1])   ← 看未来
因果 Conv3D:  y[t] = f(x[t-2], x[t-1], x[t])   ← 只看过去
```

**空间维（H,W）仍是对称卷积**，只有**时间维**因果。这与 DiT 用 attention mask 实现
因果是**同目的、不同手段**：VAE 在 padding 层面天然因果，无需 mask。

`cache_x` 参数支持 streaming：把上一时间块末帧 concat 到时间维前，分块编码与整段
一次编码**数学等价**（类比 LLM 的 KV cache，但缓存的是**卷积边界帧**）。Policy TRT
一次 encode 17 帧，export 把 cache 逻辑**静态展开**为 `conv_1…conv_4`（§8.5 kernel 碎片来源）。

### 11.7 为什么 VAE 用卷积、DiT 用注意力

```text
高分辨率像素   → CNN（VAE）压缩   → 低分辨率 latent → Transformer（DiT）建模
   局部冗余多       线性复杂度            token 少          二次复杂度可承受
```

- **VAE（CNN）压空间冗余**：图像局部强相关 → 局部卷积 + 权重共享高效；参数小、activation 大
- **DiT（Transformer）做全局建模**：低分辨率 latent 上才付得起 `O(N²)`；需跨模态长程交互
- 若在全分辨率像素上做 attention，`N≈6.8M`，`N²` 爆炸 → **先 VAE 降维是根本原因**

### 11.8 性能特征差异（呼应 profile）

| | VAE Conv3D（本 profile） | DiT attention |
|--|--------------------------|----------------|
| 热点 | 大 activation 读写（memory-bound） | 大 GEMM（compute-bound 更明显） |
| 瓶颈资源 | DRAM 带宽 | Tensor Core 算力 + 权重带宽 |
| 优化重点 | 降分辨率/帧数、量化激活 | KV cache、量化权重、flash-attention |
| 单次调用 | vae_encode **1 次** | dit **× ~30 步** |

两者优化手段基本**不通用**：VAE 围绕「减小 H×W×T 的 activation」，DiT 围绕「减少
attention/GEMM 的算力与权重访存」。

---

## 12. 小结

- **Engine 文件大小** ≈ 权重 + TRT plan；**Context memory** ≈ 激活 + scratch。
- Wan 3D VAE encode 在 `[1,3,17,480,832]` 上，**中间 feature map 可达 GB 级**，远大于 ~300MB 权重。
- Policy infer 中 vae_encode 只跑 **1 次**；dit 去噪环占 **算力与权重** 主导，vae 占 **激活峰值** 主导。
- Thor 上 `vae_encode` ~2.3GB context + ~300MB 权重 ≈ **2.6GB+** 为正常现象。
- **Layer profile（§8）**：单次 encode **~460 ms/iter**；**~48%** 在全分辨率 L0 + 下采样 RMS；**358 kernel/iter**。
- **瓶颈是混合的（§8.6、§10.3）**：**Conv3D（~54%）是 compute-bound**（总 ~35 TFLOP/iter，AI≈1588 > BF16 ridge，实测 ~56% 峰值利用）；**RMS+SiLU（~46%）才是 memory-bound**。大 activation 容量 ≠ 带宽瓶颈。
- **降带宽 vs 抬算力（§10.6）**：RMS/SiLU 靠量化/降分辨率减字节；Conv 靠 **FP8/NVFP4 抬 Tensor Core 峰值** 或减 FLOPs。**BF16↔FP16 两者都无收益**（§9）。
- **Conv3D（§10.3）** 内部 im2col+GEMM，高算术强度；**RMS（§10.2）** 非 GEMM、FP32 reduce 访存密集，两者共占 profile **~81%**。
- **VAE(CNN) vs DiT(Transformer)（§11）**：卷积**线性于像素**、保 `[B,C,T,H,W]` 网格、时间因果；注意力**二次于 token**、拍平成 `[N,C]` 序列。先 VAE 降维再 DiT 建模，二者优化手段不通用。
