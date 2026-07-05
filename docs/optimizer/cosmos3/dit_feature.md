# Cosmos3 MoT DiT — TRT Engine 与 trtexec Layer Profile 分析

本文档解释 **Cosmos3 Policy-DROID TRT 部署** 中 `dit.engine`（单次 denoise step）的：

1. **Engine 体积 vs 推理延迟**（§1–§3）
2. **trtexec layer profile 时间分布与瓶颈**（§4–§7）
3. **MoT 双通路与 profile 热点映射**（§8）
4. **优化方案与 π₀.₅ 对比**（§9–§10）

相关背景：

- TRT 分 stage 部署：[`docs/models/cosmos3_trt_deploy.md`](../../models/cosmos3_trt_deploy.md)
- MoT 结构：[`docs/models/cosmos3_mot.md`](../../models/cosmos3_mot.md)、[`docs/models/cosmos3_arch_dit.md`](../../models/cosmos3_arch_dit.md)
- 与 π₀.₅ 对比：[`docs/models/cosmos3_vs_pi05.md`](../../models/cosmos3_vs_pi05.md)
- VAE profile（对照）：[`vae_feature.md`](./vae_feature.md)
- build 配置：`configs/build_configs/cosmos3_policy_droid_dit_step_build_cfg.py`
- 固定 profile：`chameleon/deploy/cosmos3/shapes.py` → `POLICY_DROID`

**Profile 数据来源：** Jetson Thor 上 `trtexec --loadEngine=dit.engine --exportProfile=cosmos3_dit_profile.json`（101 次迭代，971 个 layer 条目）。

---

## 1. 现象与测量设定

Policy-DROID 固定 profile 下，dit stage 典型规模：

```text
engines/
  dit.engine         ~28 GB（BF16 权重 + TRT plan）

dit_step 动态输入（每 denoise step 1 次 forward）：
  vision_tokens     [1, 48, 5, 30, 52]
  vision_timesteps  [1560]              # noisy vision patch 数
  action_tokens     [16, 64]
  action_timesteps  [16]

trtexec profile 汇总：
  iterations(count) = 101
  layer entries     = 971
  单次 forward      ≈ 393 ms/step
```

**注意：**

- `dit.engine` 表示 **一次 flow-matching denoise step**，不是完整去噪环；host 侧需 `num_inference_steps` 次调用（deploy yaml 默认 30，stats 压测可用 4）。
- Profile JSON 中 `percentage` 字段 **已是百分比**（如 `0.85` = 0.85%），**不要再次 ×100**。
- 下文时间均以 **`timeMs` 聚合** 为准。

---

## 2. dit.engine 28GB 是什么

| 指标 | 实测/配置 | 含义 |
|------|-----------|------|
| **Engine 文件 ~28GB** | 磁盘序列化 | **~16B MoT 参数 × BF16（2B）** + TensorRT kernel/plan 元数据 |
| **单次 step 延迟 ~393ms** | trtexec profile | 36 层 MoT **整网 forward**（und + gen 双通路） |
| **Context memory** | 另计（未在本 profile 中单列） | 激活 + workspace；权重在 deserialize 时单独占 GPU |

Chamleon stats（`FlopCounterMode`）对同 profile 的理论估算：

```text
dit × 1 step:  MACs ≈ 25.9 T,  TFLOPs ≈ 51.8,  Weight ≈ 30.3 GB
```

与 engine ~28GB、profile ~393ms/step **数量级一致**。

---

## 3. 端到端延迟预算（Policy）

```text
单次 dit forward（1 denoise step）     ≈ 393 ms
num_inference_steps = 30（默认 yaml）  → dit 环 ≈ 11.8 s
num_inference_steps = 4（stats 压测）  → dit 环 ≈ 1.57 s
```

对比同机 VAE encode profile（~460 ms/iter，见 `vae_feature.md` §8）：

| Stage | 单次 | ×30 steps | 占比直觉 |
|-------|------|-----------|----------|
| `vae_encode` | ~0.46 s | ×1 | 一次性 |
| **`dit` step** | **~0.39 s** | **×30** | **pipeline 绝对主导** |
| `vae_decode` | 更重（decode FLOPs 更大） | ×1 | 次主导 |

**结论：** Policy 延迟瓶颈在 **dit 去噪环 × steps**，不是 VAE encode。

---

## 4. trtexec Layer Profile：整体时间分布

### 4.1 按算子类别（ms/step）

| 类别 | ms/step | 占比 | kernel 条目 |
|------|--------:|-----:|------------:|
| **Fc / Linear（attention Q/K/V/Out，und+gen 两套）** | 259.4 | **66.0%** | 256 |
| Attention MHA（`_gemm_mha_v2`） | 34.3 | 8.7% | 72 |
| MLP up+gate（SwiGLU，gen `mlp_moe_gen`） | 30.3 | 7.7% | 36 |
| RMSNorm | 23.8 | 6.0% | 288 |
| SiLU / SiLU×Mul | 21.2 | 5.4% | 74 |
| MLP down（gen） | 15.1 | 3.9% | 36 |
| RoPE（Cos/Sin/Slice/Concat 等） | 5.3 | 1.3% | 108 |
| Layout / Scatter / Gather / Move | 3.5 | 0.9% | 97 |
| 其他小 GEMM（time_embedder、action_proj 等） | 0.3 | 0.1% | 4 |

**GEMM 合计（Fc + MLP + 小投影）≈ 305 ms/step，占 77.6%。**

### 4.2 归一化算子族（跨 36 层累加）

| 归一化名称 | 总耗时占比 | 实例数 | 含义 |
|------------|----------:|-------:|------|
| `__myl_Fc` | 66.0% | 256 | MoT 每层 und/gen 的 Q/K/V/Out 线性投影 |
| `_gemm_mha_v2` | 8.7% | 72 | 36 层 × 2 通路（und 因果 + gen full） |
| `layers_N/mlp/up+gate MatMul` | 7.7% | 36 | gen SwiGLU 上行 |
| `__myl_SiluMul` | 5.4% | 72 | SwiGLU 激活 |
| `__myl_AddMulMeanAddSqrtDivMulMul` | 4.4% | 142 | RMSNorm 融合链 |
| `layers_N/mlp/down MatMul` | 3.9% | 36 | gen MLP 下行 |

### 4.3 Kernel 碎片化

- **971 kernels / step**（对比 VAE encode ~358 kernels/iter）。
- RMSNorm（288）+ SiLU（74）+ RoPE（108）占 **~11% 时间**，算术强度低、launch 开销不可忽视。
- Scatter/Gather（`ScatterElements`、pack/unpack 序列写入）合计 ~0.9%，不是主瓶颈。

---

## 5. 瓶颈判定：GEMM-bound、Weight-bound

### 5.1 不是 Attention-bound

`_gemm_mha_v2` 仅 **8.7%**。FlashAttention / fused MHA 类优化 **上限有限**（最多改善 ~9% 量级，且 TRT 已用 `_gemm_mha_v2`）。

真正耗时在 **Attention 前后的线性投影**（`__myl_Fc`，66%）和 **gen MLP**（~12%）。

### 5.2 小 batch × 大权重 → Weight Memory-bound

Policy-DROID 联合序列长度（静态打包，`text_prefix_len=256`）：

```text
und（文本）     ≈ 256 token
gen（vision）   ≈ 5 × patch_h × patch_w = 5 × 15 × 26 = 1950 token
gen（action）   = 16 token
合计            ≈ 2222 token / step
```

单 step 理论计算量 ~51.8 TFLOP，实测 393 ms → **有效 ~132 TFLOP/s**。

MoT 权重 BF16 ≈ 30 GB，**每 step 需完整读取全部线性层权重**（M 维 = seq_len ~2000，属「小 token × 大矩阵」）。在 Thor LPDDR5X 带宽下：

- 光读 30 GB 权重 ≈ 数十 ms 量级（与 profile 同阶）
- 实际 GEMM 难以吃满 Tensor Core 峰值 → **典型的 weight memory-bound**

**含义：**

- **降 bitwidth（FP8 / NVFP4 权重量化）** 直接减权重带宽 → 对 77% GEMM 时间 **近似线性收益**。
- **BF16 → FP16 A/B 无带宽收益**（同为 2B/element，见 `vae_feature.md` §9）。

### 5.3 与 VAE 瓶颈对比

| | VAE encode | DiT step |
|---|------------|----------|
| 主算子 | Conv3D + RMS | Linear/GEMM + MHA |
| 主导资源 | Conv **compute-bound**；RMS **memory-bound** | **GEMM weight-bound** |
| 第一优化杠杆 | 降输入 T/H/W；FP8/NVFP4 抬 compute roof | **权重量化**；减 steps |
| steps 倍数 | ×1 | **× num_inference_steps** |

---

## 6. Token 构成：Vision 去噪是「世界模型税」

gen 段 token 中 **vision >> action**：

```text
vision tokens : action tokens ≈ 1950 : 16 ≈ 99% : 1%
```

Policy 控制环 **主要关心 action**，但 dit 每 step 仍对 **全部 vision latent patch** 做 und+gen 全套 GEMM + MHA。

这与 [`cosmos3_vs_pi05.md`](../../models/cosmos3_vs_pi05.md) §8、§9.2 一致：

> Cosmos3 为联合 world model；即使只要 action，video latent 去噪仍占大量计算。

**π₀.₅** suffix 只有 action token，prefix（vision+text）prefill 1 次 cache —— 这是 Cosmos3 policy **结构性更贵** 的根因，而非 TRT build 配置失误。

---

## 7. MoT 双通路与 Profile 热点映射

### 7.1 每层（×36）在 profile 里的体现

`Cosmos3VLTextMoTDecoderLayer` 每层包含：

| 模块 | 通路 | Profile 中典型条目 |
|------|------|-------------------|
| Q/K/V/Out 投影 | und | `__myl_Fc`（`to_q/k/v/out`） |
| Q/K/V/Out 投影 | gen | `__myl_Fc`（`add_q/k/v_proj`、`to_add_out`） |
| Self-Attention | und | `_gemm_mha_v2`（`is_causal=True`） |
| Cross/Full-Attn | gen | `_gemm_mha_v2`（Q_gen → [K_und\|K_gen]） |
| SwiGLU MLP | und | `__myl_Fc` + `SiluMul`（`mlp`） |
| SwiGLU MLP | gen | `layers_N/mlp/up+gate/down MatMul` + `SiluMul`（`mlp_moe_gen`） |
| RMSNorm | 双通路 | `AddMulMeanAddSqrtDivMulMul`、`ReshMulMean` |
| RoPE | 双通路 | `SlicSlicNegConc...`、`CastMulReshMulCosSin...` |

72 个 `_gemm_mha_v2` = **36 层 × 2 通路**，与 MoT 结构吻合。

### 7.2 Policy 下 und 通路的「浪费感」

- 文本 caption **静态**（`text_prefix_len=256`），但 **每 step 仍跑满 36 层 und**（因果 self-attn + und MLP），产出 K/V 供 gen 读取。
- und token 仅 ~256，**单步 FLOPs 占比低于 gen**，但 **~一半权重（~8B 量级）仍常驻并参与内存读**。
- **未做 prefix KV cache**（π₀.₅ 的 `past_key_values` 等价优化）→ 详见 [`cosmos3_vs_pi05.md` §6.3、§10。

---

## 8. 算子级说明（热点 GEMM / Norm）

### 8.1 `__myl_Fc`（66%）

TensorRT 对 Linear 的融合命名，对应 MoT 中：

- und：`to_q`, `to_k`, `to_v`, `to_out`（4096 ↔ 4096/1024，GQA 32Q/8KV）
- gen：`add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`
- und MLP：`gate_proj`, `up_proj`, `down_proj`（4096 ↔ 12288 SwiGLU）

**hidden_size=4096, intermediate_size=12288, 36 layers, 双通路** → Fc 条目多、单次 forward 权重大。

### 8.2 `_gemm_mha_v2`（8.7%）

TRT 内置 fused MHA GEMM。seq ~2222 时 attention 矩阵规模有限，**非二次爆炸瓶颈**（对比超长上下文 LLM）。

### 8.3 RMSNorm + SiLU（~11%）

- 低算术强度，memory-bound。
- 288 条 RMS 相关 kernel / step → 融合进相邻 GEMM epilogue 或 LayerNorm plugin 有 **边际收益**。

### 8.4 前置 Scatter / 序列打包（<1%）

`ScatterElements`、time_embedder、`action_proj_in` 等在 **整 step 393ms 中可忽略**；主成本在 36 层 backbone。

---

## 9. 优化方案（按优先级）

### P0 — 最高 ROI

| 方案 | 目标 | 预期 | 备注 |
|------|------|------|------|
| **权重量化 FP8 / NVFP4** | GEMM 77% | **~1.5–2×** step 加速 | weight-bound；Thor Blackwell 支持 FP4 TC；`modelopt` PTQ + 重建 engine |
| **减少 `num_inference_steps`** | 整环线性 | 30→10 即 ~3× | 需质量评估；可配合蒸馏 scheduler |
| **降低 vision token 数** | GEMM + MHA | 近似线性于 token | 降 `resolution_tier` / canvas；减 `num_frames`（需质量验证） |

### P1 — 结构性（接近 π₀.₅）

| 方案 | 目标 | 预期 | 备注 |
|------|------|------|------|
| **und prefix KV cache** | 静态文本重复计算 | und 部分每 step 节省 | gen 仍占主导；需改 runtime / 导出图 |
| **Action-only 推理** | vision 去噪 | **数量级** 潜力 | 跳过/冻结 video latent 分支；需蒸馏或非官方 checkpoint |
| **分离 text_embed + 缓存 K/V** | 与 cache 配合 | 减重复 embed | 已有 `text_embed.engine`（~1.2GB），可 prefill |

### P2 — 工程微调

| 方案 | 目标 | 预期 | 备注 |
|------|------|------|------|
| **CUDA Graph**（`use_cudagraph: true`） | 971 launch/step | 个位数 %～10% | deploy yaml 默认 false；静态 shape 已满足 |
| **RMSNorm+SiLU epilogue 融合** | 11% memory op | 边际 | 依赖 TRT / 自定义 plugin |
| **增大 workspace** | 选更快 GEMM algo | 不确定 | build_cfg 已 `workspace_mb=16384` |
| **BF16→FP16** | 带宽 | **~0** | 同 2B/element，勿指望加速 |

### P3 — 不推荐作为首选

| 方案 | 原因 |
|------|------|
| 仅换 FlashAttention | MHA 已仅 8.7%；TRT 已有 `_gemm_mha_v2` |
| 仅优化 Scatter | <1% 占比 |

---

## 10. 与 π₀.₅ 及 Chamleon stats 对照

### 10.1 π₀.₅（反应式 policy）

```text
prefill ~2B VLM 1×  →  KV cache
denoise ~300M expert × N  →  Q=action, KV=cache
```

Cosmos3 dit step：

```text
每 step ~16B MoT 全 forward  →  Q_gen full-attn [K_und | K_vision | K_action]
```

等价语义、**差一个数量级的工程实现**（见 [`cosmos3_vs_pi05.md`](../../models/cosmos3_vs_pi05.md) §2.3、§6.3）。

### 10.2 Chamleon stats（理论 vs profiler）

| 指标 | FlopCounterMode（可信） | torch.profiler `--measured` |
|------|-------------------------|------------------------------|
| dit TFLOPs/step | **51.8** | **51.8（+0.0%）** ✓ |
| dit Weight | 30.3 GB | — |
| profiler_mem | — | 高估（allocator churn，勿与理论 Total GB 比） |

dit 的 **理论 FLOPs 与 profiler 完美吻合**（matmul 为主）；VAE 则 profiler 漏计 Conv3D（见 `vae_feature.md` 或 stats 说明）。

---

## 11. 小结

| 问题 | 结论 |
|------|------|
| dit 慢在哪？ | **77% GEMM（主要是 attention 线性投影 + gen MLP）**，非 MHA |
| 瓶颈类型？ | **Weight memory-bound**（~30GB 权重/step，seq ~2k） |
| 为何比 π₀.₅ 贵？ | 16B 全网 × steps；vision token 占 99%；无 prefix KV cache |
| 第一优化？ | **FP8/NVFP4 量化** + **减 steps** +（可选）**缩 vision token** |
| 28GB engine 正常吗？ | 是，~16B BF16 MoT 参数量 |

**一句话：** `dit.engine` 单次 step ~393ms 是 **16B MoT 在小 batch、大权重下的 GEMM 带宽瓶颈**；Attention 只占 ~9%。Policy 要接近 π₀.₅ 延迟，需 **量化 + 减 steps**，并从架构上 **削减 vision 去噪或 cache 静态 und**——单靠 TRT BF16 build 微调难以 order-of-magnitude 提速。

---

## 12. 附录：Profile 复现命令

```bash
trtexec --loadEngine=output/cosmos3_policy_droid_trt/engines/dit.engine \
  --exportProfile=cosmos3_dit_profile.json \
  --iterations=100 --warmUp=10

# 聚合分析（示例）
python3 - <<'PY'
import json
from collections import defaultdict
data = json.load(open("cosmos3_dit_profile.json"))
n = data[0]["count"]
rows = [d for d in data[1:] if "name" in d]
total = sum(d["timeMs"] for d in rows)
print(f"per-iter: {total/n:.1f} ms, entries: {len(rows)}")
PY
```

Profile 文件路径（本分析）：`/home/zhangxa/nsys/wm/cosmos3_dit_profile.json`
