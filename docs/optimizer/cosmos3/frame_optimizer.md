# Cosmos3 推理优化 — cosmos-framework 实现梳理

本文档总结 **cosmos-framework** 在 `cosmos_framework/inference/` 及底层模型代码中，为 **Cosmos3 推理** 做的工程与算法优化。侧重 **离线批量推理 / 多卡 serving** 路径（`python -m cosmos_framework.scripts.inference`），与 Chameleon 侧 TRT 分 stage 部署（见 [`dit_feature.md`](./dit_feature.md)、[`vae_feature.md`](./vae_feature.md)）互补。

相关背景：

- 推理 CLI 与 preset：cosmos-framework [`docs/inference.md`](https://github.com/NVIDIA/Cosmos/blob/main/docs/inference.md)
- MoT 结构：[`docs/models/cosmos3_mot.md`](../../models/cosmos3_mot.md)
- Policy 推理细节：[`docs/models/cosmos3_policy_detail.md`](../../models/cosmos3_policy_detail.md)
- TRT 部署：[`docs/models/cosmos3_trt_deploy.md`](../../models/cosmos3_trt_deploy.md)

**代码仓库：** [cosmos-framework](https://github.com/NVIDIA/Cosmos)（下文路径均相对于该仓库根目录）

---

## 1. 优化总览

| 维度 | 主要手段 | 典型收益 |
|------|----------|----------|
| **显存** | FSDP 分片、`inference_mode`、分块 VAE、selective activation checkpointing | 32B Super 可上 8×80GB |
| **单样本延迟** | Context Parallel、CFG Parallel、CUDA Graphs、`torch.compile` 静态 shape | `latency` preset |
| **批量吞吐** | `throughput` preset、token packing、varlen attention、DP replicate | 离线批处理 |
| **计算复用** | KV cache（AR）、TaylorSeer 特征缓存、`guidance_interval` | 少算重复前向 |
| **流水线** | VAE 独立 GPU + CUDA stream 重叠 | DiT 与 decode 并行 |
| **采样** | UniPC 少步求解器、蒸馏 FixedStep、bf16 | 默认 ~30 step → 可更少 |

---

## 2. 并行策略（FSDP / CP / CFGP）

推理将并行拆成四个独立轴，通过 **`latency` / `throughput`** 两个 preset 自动配置。

| 轴 | CLI 参数 | 作用 |
|----|----------|------|
| **FSDP** | `--dp-shard-size` | 权重分片，大模型 fit 显存 |
| **DP replicate** | `--dp-replicate-size` | 多副本并行处理不同 batch |
| **Context Parallel** | `--cp-size` | 长序列（video token）切分到多卡 |
| **CFG Parallel** | `--cfgp-size` | cond / uncond 两条前向分到不同 GPU |

**Preset 行为**（`cosmos_framework/inference/args.py` → `_build_context_parallelism`）：

| preset | cp | cfgp | 适用 |
|--------|----|------|------|
| **`latency`** | `min(MAX_CP, world_size // cfgp)` | `min(2, world_size)` | 单样本墙钟时间最小（在线 / Ray Serve） |
| **`throughput`** | 1 | 1 | 全卡 FSDP + batch，离线批量最大吞吐 |

Ray 在线服务默认 `parallelism_preset: latency`（`cosmos_framework/inference/ray/configs/latency.yaml`）。

**推理模式开关：** 创建 pipeline 时设置 `enable_inference_mode=True`（`cosmos_framework/inference/inference.py` → `OmniInference._get_parallelism_config`），关闭 autograd 记录，降低显存与 Python 开销。

```python
# cosmos_framework/inference/inference.py
ParallelismConfig(
    enable_inference_mode=True,
    data_parallel_shard_degree=setup_args.dp_shard_size,
    context_parallel_shard_degree=setup_args.cp_size,
    cfg_parallel_shard_degree=setup_args.cfgp_size,
)
```

---

## 3. torch.compile + CUDA Graphs

配置定义：`cosmos_framework/configs/base/defaults/compile.py` → `CompileConfig`。

推理侧默认（`cosmos_framework/inference/common/args.py` → `ParallelismOverrides`）：

| 开关 | 推理默认 | 说明 |
|------|----------|------|
| `use_torch_compile` | `True` | 编译 MoT 去噪环 |
| `use_cuda_graphs` | `True` | 捕获固定 CUDA 执行图 |
| `compiled_region` | `"all"` | 可改为 `"language"` 仅编译 Transformer 层 |
| `compile_dynamic` | `True`（可关） | 推理形状稳定时设 `False` 更快、少重编译 |

**CUDA Graph 启用条件**（`inference.py` → `_get_compile_config`）：仅当 `dp_shard × cp × cfgp == 1`（单卡无并行分片）时开启，避免多 rank 图捕获不一致。

**其它编译细节：**

- `before_load_model()` 设 `use_duck_shape=False`（`inference/model.py`），减少 duck shape 触发的重编译。
- 每步去噪前调用 `torch.compiler.cudagraph_mark_step_begin()`（`omni_mot_model.py` → `velocity_fn`），配合 CUDA Graph 复用。
- 可选 `max_autotune_pointwise` / `coordinate_descent_tuning` 对 RMSNorm 等 pointwise kernel 做 autotune（首步编译更慢、稳态更快）。

Nano 模型 config 中训练侧已 `compile.enabled: true`、`compiled_region: language`（`inference/configs/model/Cosmos3-Nano.yaml`）；推理 CLI 可覆盖为 `all` + cuda graphs。

---

## 4. CFG（Classifier-Free Guidance）计算优化

文本 CFG 默认使每步去噪跑 **cond + uncond 两次前向**。`velocity_fn`（`omni_mot_model.py`）做了三层优化：

### 4.1 单前向快路径

`guidance == 1.0` 且无 `velocity_postprocess` 时只跑一次 cond（Policy / Action 默认无 CFG）：

```python
if not _any_needs_text_cfg and velocity_postprocess is None:
    return _single_velocity_fn(cond_tokens, skip_text_tokens=False)
```

### 4.2 guidance_interval

仅在指定 σ 区间内做 CFG，区间外退回单前向（`guidance_interval: [t_lo, t_hi]`）。

### 4.3 FSDP 序列对齐

多卡 `throughput` preset 下，各 rank 样本可能 CFG 决策或 `num_steps` 不一致，会导致 FSDP `allgather` 死锁。通过对 **dp_shard 组** 做：

1. `all_reduce` CFG 决策：任一 rank 需要 CFG 则全体跑双前向（本地不需要 CFG 的 rank 仍返回 `cond_v`）。
2. `all_reduce` `num_steps`：步数不足的 rank 用 dummy sampler 调用补齐 collective 序列。

### 4.4 CFG Parallel（cfgp）

`cfgp_size=2` 时 cond/uncond 可分到两张卡并行（`_run_classifier_free_guidance`，`omni_mot_model.py:2145`），与 `latency` preset 配合。

**注意**：cfgp 依赖 P2P `isend/irecv` 在两卡间交换 velocity（`omni_mot_model.py:2185-2192`）。当 `cfgp_enabled == False`（单设备）时直接退回**串行两次前向**：

```python
# omni_mot_model.py:2167-2171
if self.parallel_dims is None or not self.parallel_dims.cfgp_enabled:
    return (
        single_velocity_fn(cond_tokens, False),                       # cond forward
        single_velocity_fn(uncond_tokens, skip_text_tokens_for_cfg),  # uncond forward
    )
```

---

## 4′. Jetson Thor 端侧场景下的 CFG 专项分析

前面 §4 描述的是通用（多卡）路径。**端侧单设备（Jetson Thor）** 的约束完全不同，本节展开。

### 4′.1 端侧 CFG = 严格 2× 串行，无并行摊薄

多卡服务器可用 `cfgp_size=2` 把 cond / uncond 分到两张卡**并行**，CFG 的墙钟开销接近 1×。但 Jetson Thor 是**单 device**：

- `cfgp_enabled == False` → 走 `omni_mot_model.py:2168` 分支，cond 与 uncond **顺序执行两次完整 MoT forward**。
- `_dp_shard_group is None` → §4.3 的 FSDP `all_reduce` 对齐逻辑**全部短路**（`needs_text_cfg` 直接本地判定），端侧无此开销也无此收益。
- 端侧 CFG 是**纯 2× 惩罚**，没有任何并行/通信手段能摊薄。

### 4′.2 延迟预算：CFG 是端侧最贵的乘数

以 `dit_feature.md` §3 的 Thor 实测（单次 dit step ≈ 393 ms）为基准：

| 任务 | guidance | 每 step forward | 每 step 延迟 | ×30 step 环 |
|------|----------|-----------------|-------------|-------------|
| **Policy / Action** | `1.0` | 1（cond） | ~393 ms | ~11.8 s |
| **T2V / 世界模型生成** | `>1`（如 6.0） | 2（cond+uncond） | ~786 ms | **~23.6 s** |

CFG 直接把 **dit 去噪环翻倍**，是端侧继 `num_inference_steps` 之后**第二大延迟乘数**，且与步数**相乘**（`总 forward 数 = num_steps × (1 or 2)`）。

### 4′.3 两类端侧任务的分野

| 场景 | 端侧行为 | 说明 |
|------|----------|------|
| **Policy（DROID/LIBERO/Bridge）** | 默认 `guidance_scale=1.0` → §4.1 单前向快路径 | 见 [`cosmos3_policy_detail.md`](../../models/cosmos3_policy_detail.md)：`dit.engine` 即单次 cond forward，TRT 部署天然免 CFG |
| **T2V / I2V 世界生成** | 高 guidance（默认可达 6.0）→ 双前向 | 端侧成本近 2×，是 T2V 上 Thor 的主要痛点 |

**结论**：Chameleon 端侧 Policy 部署（`dit.engine`）本就跑在无 CFG 快路径上，这也是 `dit_feature.md` 中 profile 只体现**单次 cond forward** 的原因；真正需要为 CFG 优化的是 **T2V/世界模型生成** 场景。

### 4′.4 端侧 CFG 降本手段（按 ROI）

| 手段 | 机制 | 端侧收益 | 代价 / 备注 |
|------|------|----------|-------------|
| **引导蒸馏（guidance distillation）** | 把 CFG 融进单前向的蒸馏模型 | **2× → 1×**，且常配合极少步 | 需蒸馏 checkpoint；配 `FixedStepSampler`（§9），端侧首选 |
| **`guidance_interval=[lo,hi]`** | 仅中间 σ 区间做双前向，两端退单前向（`omni_mot_model.py:2446-2449`） | 双前向步数按区间线性下降 | 需质量评估；区间过窄掉指导效果 |
| **降 `num_inference_steps`** | 与 CFG 正交但相乘 | 30→10 直接 ~3×（叠加 CFG 后基数减半） | 需质量评估 |
| **`normalize_cfg`** | velocity 范数归一（`:2479`） | 不省算力 | 仅稳定数值 / 抑制过曝，不是加速项 |

### 4′.5 TRT 端侧的关键洞察：batch=2 打包 vs 串行 2×

PyTorch 路径单设备只能**串行**跑 cond、uncond（每次都要完整读一遍 ~30 GB BF16 权重）。但 `dit_feature.md` §5.2 已判定 dit 是 **weight memory-bound**（小 token × 大权重），这带来一个 TRT 侧优化空间：

> **把 cond + uncond 拼成 `batch=2` 一次 forward**：权重只需**读一次**，两条样本共享权重带宽。在 weight-bound 下，batch=2 的延迟远小于 2× 单样本串行（增量主要是激活与 MHA，权重带宽被摊薄）。

因此若端侧 T2V 必须开 CFG：

- **优先** 用引导蒸馏模型（彻底消除第二次前向）；
- **次选** 导出支持 `batch=2` profile 的 `dit.engine`，把 cond/uncond 打包，用「权重复用」换取接近 1× 的权重读取成本，而非 §4′.1 的 2× 串行；
- cfgp 多卡并行方案在单 Thor **不适用**（无第二张卡做 P2P）。

### 4′.6 与量化 / 减步的叠加关系

端侧三大乘数相互独立、可叠乘：

```text
总 dit 时间 ≈ 单forward权重带宽成本
            × num_inference_steps        （减步 / 蒸馏 scheduler）
            × CFG倍数(1 或 ~2)            （蒸馏→1；或 batch=2 打包→~1.x）
            × 权重精度因子                 （FP8/NVFP4 量化，见 dit_feature.md §5.2 P0）
```

端侧 T2V 的推荐组合：**引导蒸馏（CFG→1 + 少步）× 权重量化（FP8/NVFP4）**，可望在 Thor 上把 T2V 从数十秒级压到个位数秒级；Policy 场景因 `guidance=1.0` 已省掉 CFG 这一维，重心回到量化与减步。

---

## 5. TaylorSeer — 去噪步特征缓存

实现：`cosmos_framework/model/generator/utils/taylorseer.py`（移植自 [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)）。

### 5.1 原理

Flow-matching / diffusion 的相邻去噪步之间，DiT 内部特征随 σ **平滑变化**。TaylorSeer 把「特征关于去噪步的轨迹」用**泰勒级数**建模：在少数 **full** 步真算 DiT 并缓存各阶导数，其余 **Taylor** 步直接用级数外推，跳过整网前向。

- `derivative_approximation()`（`taylorseer.py:12`）：在 full 步用有限差分递推估计 0…`max_order` 阶导数，存进 `cache_dic`。
- `taylor_formula()`（`:38`）：Taylor 步按 \( f(t)=\sum_i \frac{1}{i!} f^{(i)} \cdot x^i \) 外推，`x = step - 上一个 full step`。
- `cal_type()`（`:93`）：每步判定 `full` / `Taylor`（也支持 `ToCa` / `Delta-DiT`）。
- `cache_init()`（`:134`）：按层建缓存，默认见下表。

| 参数 | 默认 | 含义 |
|------|------|------|
| `taylor_cache` | `True` | 开启泰勒缓存分支 |
| `max_order` | 6 | 展开阶数 |
| `first_enhance` | 5 | 前 5 步强制 full（预热轨迹） |
| `fresh_threshold` | 3 | full 步之间的间隔（每 3 步刷新一次 full） |

**收益直觉**：30 步里若只有 ~10 步是 full、其余 Taylor，则 DiT 前向次数近似 **降到 1/3**（外推极廉价）。这属于「减少有效 forward 数」这一维，与量化、CFG 正交。

### 5.2 关键事实：当前 cosmos-framework 中它是**未接线的脚手架**

务必注意（基于源码核对）：

- `cal_type` / `taylor_formula` / `derivative_approximation` / `cache_init` 在整个仓库里**除定义外没有任何调用点**。
- `unified_mot.py:845` 的 `_impl_init_taylorseer` 只设置 `self.enable_taylorseer = True` / `self.cache_dic` / `self.current`，但 `init_taylorseer` **本身从未被调用**，`enable_taylorseer` 也**未在 forward 中被读取**。

**结论**：在当前开源版本里，TaylorSeer 只是**预留接口**，去噪环并未真正启用 full/Taylor 分支。**开箱即用地"打开"它并不存在**——无论服务器还是端侧，都需要自己把 `cal_type`/`taylor_formula` 接进 MoT forward 才能生效。

### 5.3 DROID Policy 端侧能用吗？

分三层看：

**(1) 功能上——目前不能"直接开"。** 如 §5.2，framework 未接线；TRT 端侧的 `dit.engine` 更是**单步静态图**（`dit_feature.md` §1），天然没有「跨步缓存 + 每步 full/Taylor 分支」的控制流。要在 TRT 上用，必须**在 host 侧编排**：engine 仍只做 full 前向，host 决定哪些 step 跳过并改用外推——等于自己重写调度。

**(2) 算法上——对 DROID policy 收益存疑。**

| 因素 | 对 policy 的影响 |
|------|------------------|
| **步数少** | Policy 常用 ~30 步，但 stats 压测只 4 步；`first_enhance=5` 已 ≥ 4，**预热都没跑完就结束**，Taylor 步几乎为 0 → 无收益 |
| **动作精度敏感** | Policy 关心的是 16 个 action token（`dit_feature.md` §6），外推误差直接作用到控制指令，比 T2V 的像素误差更不可接受 |
| **weight-bound** | full 步仍要读满 ~30 GB 权重；Taylor 步虽跳过 GEMM，但省的是「已经 weight-bound 的那次」，需足够多 Taylor 步才摊得回 |

**(3) 更适合 TaylorSeer 的是 T2V/长序列世界生成**：步数多（几十步）、vision token 占 99%、对细微特征误差更宽容——正是 TaylorSeer 原论文的目标场景。

**给 DROID policy 端侧的建议**：**不优先上 TaylorSeer**。端侧 policy 降本应先走 `dit_feature.md` §9 的 P0：**权重量化（FP8/NVFP4）+ 减 `num_inference_steps`（配蒸馏 scheduler / `FixedStepSampler`）**。这两者对 weight-bound 的 dit 是近线性收益、且不引入外推误差；若将来要用特征缓存类方法，也应在**多步 T2V**而非**少步 policy**上验证。

---

## 6. KV Cache — compile 边界外的 MemoryState

实现：`cosmos_framework/model/generator/utils/memory.py`；forward 接线：`unified_mot.py` → `_impl_forward` / `Cosmos3VLTextMoTDecoderLayer`。

### 6.1 设计动机：为何 KV cache 必须在 compile 外

`torch.compile` + CUDA Graph 要求 compiled region 内**无可变状态写入**。若 KV cache 在 compiled layer 内 `update()`，会导致：

- 图捕获因 tensor 地址/形状变化而**断裂**；
- 每步触发**重编译**（比不算 cache 更慢）。

Cosmos3 的解法是 **读写分离**：

```text
MemoryState（可变 Python 对象，compile 外）
  init(hidden_states)                         # 每步/每帧初始化
  read_for_layer(i)  → MemoryValue            # 只读 tensor 快照，可传入 compiled layer
  write_for_layer(i, kv_to_store)             # compile 外写回 (gen_k, gen_v, und_k, und_v)

_impl_forward 每层循环：
  memory_value = memory.read_for_layer(i)     # compile 外
  hidden, lbl, kv = decoder_layer(..., memory_value=memory_value, gen_only=...)
  memory.write_for_layer(i, kv)               # compile 外
```

`is_gen_only() == True` 时，decoder layer 走 **gen_only 快路径**（`unified_mot.py:1070`）：跳过 und 通路的 layernorm / self-attn / MLP，und K/V 从 cache 读取，只算 gen（vision/action）token。

这与 LLM 的 `past_key_values` 思路一致，但 MoT 额外区分 **und K/V**（文本理解段）和 **gen K/V**（生成段），每层存 4 个 tensor。

### 6.2 两种 KV cache 场景（不要混为一谈）

| 场景 | 触发条件 | cache 什么 | 省什么 |
|------|----------|-----------|--------|
| **A. AR 逐帧生成**（T2V 长视频） | `autoregressive=true`，逐 chunk 生成 | 第 0 帧后 cache 全部 und K/V + 历史 gen K/V | 第 1+ 帧跳过 text 重算，`gen_only=True` |
| **B. Prefix KV cache**（Policy / 固定序列去噪） | 静态 caption + 固定 condition latent | 去噪环外 prefill 1 次 und K/V（+ 可选 z₀ K/V） | 30 个 denoise step 不再重跑 und 36 层 |

框架里的 `MemoryState` 抽象**主要面向场景 A**（AR 推理 + 训练时 KV-cache BPTT）。场景 B 是 Chameleon 文档（`cosmos3_vs_pi05.md` §10、`dit_feature.md` §7.2）讨论的 **π₀.₅ 式 prefix cache**，语义等价但**尚未实现**。

### 6.3 当前开源代码的接线状态

| 组件 | 状态 |
|------|------|
| `MemoryState` / `MemoryValue` 抽象 | ✅ 已定义（`memory.py`） |
| `_impl_forward` read/write 循环 | ✅ 已接线（`unified_mot.py:912-934`） |
| `gen_only` decoder 快路径 | ✅ 已实现（`:1070-1145`） |
| `kv_to_store` 产出（4-tuple K/V） | ✅ 已实现（`:674-687`） |
| **`ARMemoryState` / `ARMemoryValue` 具体类** | ❌ **未开源**（仅 docstring 引用） |
| **`OmniMoTCausalModel.build_memory_state()`** | ❌ **未开源**（base 返回 `None`，`:651-670`） |
| **`attention_AR_gen_only` 注意力核** | ❌ **未开源**（comment 引用，无函数体） |
| `dispatch_attention` + `memory_value` | ❌ base 路径 `assert memory_value is None`（`attention.py:443`） |
| diffusers-cosmos3 `gen_only` | ❌ `# TODO: support gen_only = True`（`transformer.py:532`） |
| Policy `sample_args.json` | `"autoregressive": false` |

**结论**：KV cache 的 **compile 边界架构 + gen_only 快路径已写好**，但 **AR 推理的 MemoryState 实现与 attention 核未放出**，当前任何推理路径（含 Policy）**实际不启用** KV cache。

### 6.4 DROID Policy 端侧：能用吗？

**短答：当前不能直接用；概念上可做 prefix cache，但需自研 runtime，且收益有限。**

#### 6.4.1 为什么 Policy 不走 AR cache（场景 A）

Policy 推理是 **flow-matching 去噪环**，不是 AR 逐帧生成：

```text
# cosmos3_vs_pi05.md §6.2 — Policy 实际路径
for step in num_inference_steps:          # 默认 30 步
    transformer.forward(文本 + noisy vision + noisy action)   # 每步完整 16B forward
    scheduler.step
```

- `autoregressive: false`（`inference/defaults/policy/sample_args.json`）
- 每步 **vision / action latent 的噪声水平不同** → gen 段 K/V **每步都变**，不能跨步 cache gen K/V
- 唯一静态的是 **text und 段**（~256 token）→ 对应场景 B（prefix cache），不是 AR MemoryState

#### 6.4.2 关键架构问题：und K/V 生成是否需要与 gen suffix 联合计算？

**短答：不需要。** und K/V 可以像 π₀.₅ 那样 **prefill 一次**，再供 gen 反复读取；耦合是 **单向** 的。

##### 注意力拓扑（单向、可拆分）

MoT 每层 attention 在 `three_way_attention()`（`attention.py:151`）里拆成三路，**und 与 gen 的计算图可分离**：

```text
① causal（und self-attn）:
   Q_und ──causal──► K_und, V_und          （仅文本 token，und 不读 gen）

② full_sa（gen self-attn）:
   Q_gen ──full──► K_gen, V_gen            （vision + action 内部互相可见）

③ full_ca（gen → und cross-attn）:
   Q_gen ──full──► K_und, V_und            （gen 读 und，单向注入语言条件）

最终 gen 输出 = merge(full_sa, full_ca)     （attention.py:259）
```

对应 mask 语义（[`cosmos3_vs_pi05.md`](../../models/cosmos3_vs_pi05.md) §5.2）：

| 通路 | 可见范围 |
|------|----------|
| **und** | 仅 und 自身（causal） |
| **gen** | 全部 und K/V + 全部 gen K/V |

**π₀.₅ suffix cross-attend prefix KV ≈ Cosmos3 ③ full_ca**；Cosmos3 额外有 ② gen 内部 full self-attn。

##### und K/V 的输入依赖链（与 gen 无关）

`PackedAttentionMoT.forward`（`unified_mot.py:594-609`）中 und K/V 的投影 **只读 und token**：

```python
k_und_in = self.k_proj(get_und_seq(pack))   # 只用 und hidden
v_und_in = self.v_proj(get_und_seq(pack))   # 只用 und hidden
```

跨层递推也是 **und 独立残差流**（`:1117-1118, 1148-1149`）：

```text
Layer L:   und_L  →  und self-attn  →  und MLP  →  und_{L+1}
           gen_L  →  gen attn(读 und KV + gen KV) → gen MLP → gen_{L+1}
                    ↑ gen 读 und，und 不读 gen
```

因此，对 **静态 caption** 的 DROID policy：

- Layer 0：`und_0 = embed_tokens(text)` — 不依赖 gen
- Layer L：`und_K/V` 只依赖 `und_{L-1}` — **全程与 vision/action 噪声无关**
- 30 个 denoise step 间 caption 不变 → **每层 und K/V 完全相同**，可 prefill 1 次复用

框架的 `gen_only=True` 快路径（`:1070-1107`）正是为此设计：跳过 und 计算，gen 只算 Q/K/V，und K/V 从 `MemoryState` cache 读取。

##### 什么不能 prefix cache（与 π₀.₅ 的差异）

| Token | 通路 | 跨 denoise step 是否静态 | 能否 prefix cache |
|-------|------|--------------------------|-------------------|
| **文本 caption** | und | ✅ 静态 | ✅ **可以**（与 π₀ prefix 语义一致） |
| **观测 z₀**（condition vision） | gen | latent 值静态，但… | ❌ **不能**（见下） |
| **未来 z₁…z₄ + action** | gen | ❌ 每步噪声 σ 变化 | ❌ 不能 |

z₀ 虽为 clean condition，但它在 **gen 通路**内，每步仍参与 gen full self-attn（②）。z₁…z₄ 与 action 的 K/V 每步变化 → z₀ 的 gen hidden 通过 ② 被邻居「污染」→ **z₀ 的 gen K/V 也每步变化**。

这与 π₀.₅ 的根本差异：

```text
π₀.₅ prefix = SigLIP 图像 + 语言  →  prefill 1 次 cache（vision 在 prefix）
Cosmos3 und = 仅文本              →  可 prefill cache
Cosmos3 gen = VAE vision + action →  每步重算（vision 在 gen，且 gen 内部 full-attn 耦合）
```

**Cosmos3 只能 cache 文本 und K/V，不能把观测 vision 像 π₀ 那样放进 prefix cache。**

##### gen-only denoise step 的正确计算流程

```text
Prefill（1 次，无 gen token 或 dummy gen）:
  for layer in 0..35:
    und K/V[layer] = f_und(text)        # 36 层 × 256 token，~36 MB

Denoise（×30 step，gen_only=True）:
  for layer in 0..35:
    Q_gen, K_gen, V_gen = f_gen(noisy_vision, noisy_action)   # 每步重算
    out_gen = merge(
      full_sa(Q_gen, K_gen, V_gen),                          # gen 内部
      full_ca(Q_gen, und_K_cached[layer], und_V_cached[layer])  # 读 cache
    )
    gen_{L+1} = MLP_gen(out_gen + residual_gen)
```

这与 π₀.₅ 的 `denoise_step(x_t, past_key_values)` **在文本条件注入层面等价**；差别是 Cosmos3 gen 还要额外算 vision+action 的 ② full_sa，且 vision 不在 prefix 里。

##### 对 Thor 部署的含义

| 问题 | 结论 |
|------|------|
| und K/V 能否 prefill？ | ✅ **算法上完全可行**，与 gen suffix **无需联合计算** |
| 框架是否已支持？ | ❌ `build_memory_state()` 返回 `None`，推理不传 `memory` |
| TRT 如何实现？ | prefill engine（und only）+ gen-only dit engine + host 存 ~36 MB K/V |
| 加速上限？ | ~15–25%（und 占 ~11% token；但 ~8B und 权重带宽可省） |
| 能否 cache z₀ vision？ | ❌ gen 内部 full-attn 耦合，z₀ gen K/V 每步变 |

#### 6.4.3 Prefix KV cache（场景 B）在 Thor 上的潜力

Policy 每 step 的 token 构成（`dit_feature.md` §6-7）：

```text
und（文本）     ≈ 256 token    ← 静态，可 cache
gen（vision）   ≈ 1950 token   ← 每步变化（去噪）
gen（action）   = 16 token     ← 每步变化（去噪）
```

若实现 prefix cache（prefill 1 次 → 30 步复用 und K/V）：

| 维度 | 分析 |
|------|------|
| **跳过什么** | 36 层 und 的 Q/K/V proj + causal self-attn + und SwiGLU MLP |
| **仍算什么** | 36 层 gen 全套（vision+action 去噪），gen Q full-attn `[K_und_cached \| K_gen]` |
| **FLOPs 节省** | und token 仅占 ~11%，und FLOPs 低于 gen；**粗估 step 时间节省 15–25%** |
| **权重带宽** | dit weight-bound（§5.2）；跳过 und 线性层可减少 ~8B 权重读取 → 与 FLOPs 节省同量级 |
| **显存开销** | und K/V cache：36 层 × 256 token × 8 KV heads × 128 dim × 2(K+V) × 2B ≈ **~36 MB**，Thor 可忽略 |

**对比 π₀.₅**：π₀ prefill ~2B VLM 1 次 + ~300M expert × N 步；Cosmos3 若做 prefix cache 仍要 **16B gen 通路 × 30 步**，差距仍在数量级。

#### 6.4.4 TRT 端侧落地的工程代价

当前 Chameleon TRT 部署（`dit.engine`）是 **单步静态图**，无跨步状态：

```text
text_embed.engine  →  文本 embedding（1 次）
vae_encode.engine  →  观测 latent（1 次）
dit.engine         →  单次 denoise step（×30，每步独立调用，无 cache）
vae_decode.engine  →  decode（1 次）
```

要在 Thor 上启用 prefix KV cache，需要：

1. **Prefill 阶段**：新增 prefill pass（或复用 `text_embed.engine` + 部分 dit 层），产出 36 层 und K/V；
2. **Host 侧存储**：~36 MB K/V tensor（Thor LPDDR 上可行）；
3. **Gen-only dit engine**：导出 `gen_only=True` 的 TRT 图（只含 gen 通路 + 读 cached und K/V 的 cross-attn），或 host 侧拼 cached K/V 到输入 binding；
4. **去噪环改造**：30 步 loop 从「每步 full forward」变为「prefill 1× + gen-only ×30」。

这是 **P1 结构性优化**（`dit_feature.md` §9），工程量大，且 **gen 仍占 99% token**——不是 order-of-magnitude 提速。

#### 6.4.5 与 CUDA Graph / compile 的交互

`gen_only` 路径与 CUDA Graph **部分兼容**：

- `memory.py` 设计刻意把读写放在 compile 外 → 不破坏 graph capture；
- 但 `ARReplicatedIODispatch` 对 `for_cuda_graphs=True` 的 ARMemoryState **直接 raise**（`parallelize_unified_mot.py:201`）；
- Policy TRT 路径当前 `use_cudagraph: false`（`dit_feature.md` §9 P2）。

端侧若做 prefix cache + gen-only engine，CUDA Graph 可在 gen-only 子图上重新评估（静态 shape 已满足）。

### 6.5 端侧 KV cache 决策表

| 场景 | 能否直接用 framework KV cache | Thor 上值得做吗 | 推荐替代 |
|------|------------------------------|----------------|----------|
| **DROID Policy** | ❌ 未接线；且非 AR 模式 | Prefix cache 可做，**~15–25% 加速**，工程量大 | **P0：FP8/NVFP4 量化 + 减步**（近线性 ~1.5–3×） |
| **T2V 长视频 AR** | ❌ ARMemoryState 未开源 | AR cache 对多帧收益大，但需等 framework 放出 | 短片段用非 AR；长片段等官方 AR 路径 |
| **Action-only 蒸馏** | 不适用 | 跳过 vision 去噪 → **数量级**潜力 | 需专用 checkpoint，见 `dit_feature.md` §9 P1 |

### 6.6 与 π₀.₅ 的对照（为何 Policy 端侧更贵）

```text
π₀.₅ Policy:
  prefill ~2B VLM 1×  →  prefix KV cache（~36 MB 量级）
  denoise ~300M expert × N  →  suffix Q cross-attend cached KV

Cosmos3 Policy（当前）:
  每 step ~16B MoT full forward  →  text K/V 每步重算，~8B und 权重每步读取

Cosmos3 Policy（若做 prefix cache）:
  prefill und 1×  →  cache ~36 MB K/V
  gen-only × 30 step  →  仍读 ~8B gen 权重 + 算 vision/action 去噪
```

**一句话**：MemoryState / KV cache 是 Cosmos3 为 **AR 长视频生成** 预留的 compile-friendly 架构，**DROID policy 端侧当前不可用**；policy 端侧真正需要的是 **prefix KV cache（场景 B）**，概念可行、显存可承受，但 TRT runtime 需自研，且 gen 去噪仍主导延迟——**优先量化 + 减步，prefix cache 作为 P1 结构性优化**。

---

## 7. VAE 编解码优化

### 7.1 独立 GPU + Stream 重叠

`--use-separate-pipeline-vision-decode-gpu`（`inference.py` → `OmniInference._create`）：

- 将 Wan VAE 放到 spare GPU（index = `cp_size × cfgp_size`）。
- 用独立 `torch.cuda.Stream` + `Event`：DiT 去噪在主卡跑完后，异步在 VAE 卡 decode。

```text
主卡：DiT denoise step N
VAE 卡（并行）：decode step N-1 的 latent
```

### 7.2 分块与缓存（模型 config）

`Cosmos3-Nano.yaml` → `tokenizer`：

| 字段 | 作用 |
|------|------|
| `use_streaming_encode` | 流式编码，降低峰值显存 |
| `keep_decoder_cache` | 解码器 cache 复用 |
| `encode_chunk_frames` / `encode_exact_durations` | 按分辨率分块编码 |

Chameleon TRT 侧 VAE profile 见 [`vae_feature.md`](./vae_feature.md)。

---

## 8. 批处理与变长注意力

### 8.1 Token packing

`_iter_packed_batches`（`inference.py`）按 **`max_num_seqs`** 或 **`max_model_len`** 贪心打包多样本：

- 单遍扫描、保序、不丢样本；
- 超预算则 flush 当前 batch；
- 默认 `max_num_seqs=1`（单样本）；`throughput` 可提高。

### 8.2 Varlen attention

打包后序列 **batch=1、无 padding 拼接**；`model/attention/frontend.py` 的 varlen API（`seqlens_Q/KV`、`cumulative_seqlen_*`）避免 padding 浪费算力。

### 8.3 注意力后端

`BACKEND_MAP`：`flash2` / `flash3` / `natten`（视频时序局部窗口，`sequence_packing/natten.py`）。

### 8.4 Selective activation checkpointing

推理 config 中 `activation_checkpointing.mode: selective`、`save_ops_regex: [fmma]`（`Cosmos3-Nano.yaml`）——训练/大模型路径下用重算换显存；纯推理通常配合 `inference_mode` 走前向无梯度。

---

## 9. 采样器与精度（UniPC / FixedStep / bf16）

Policy 端侧延迟 ≈ **`num_inference_steps × dit_step 延迟`**（Thor 实测 ~393 ms/step，`dit_feature.md` §3）。采样器决定 **每步 σ 调度** 与 **latent 更新公式**；精度决定 **权重带宽与 Tensor Core 利用率**。二者与 CFG、KV cache **正交**，是端侧 **P0 减步 + 量化** 的主战场。

相关配置：

- Policy 默认：`cosmos_framework/inference/defaults/policy/sample_args.json`（`num_steps=30`, `shift=10.0`, `guidance=1.0`）
- Chameleon TRT：`configs/cosmos3_policy_droid_trt_deploy.yaml`（`num_inference_steps: 30`, `precision: bfloat16`）
- dit build：`configs/build_configs/cosmos3_policy_droid_dit_step_build_cfg.py`（`precision: "bf16"`）

---

### 9.1 去噪环在做什么（Rectified Flow）

Cosmos3 用 **Rectified Flow Matching**：模型每步预测 velocity `v_θ(x_t, t)`，scheduler 沿 σ 轨迹把 noisy latent 积分为 clean sample。

```text
初始化：x_T ~ N(0,I)（policy 下 vision z₀ 为 condition 锁 clean，z₁…z₄ + action 为噪声）
for t in timesteps:                    # 默认 30 步
    v = MoT.forward(x_t, t)             # dit.engine ×1（~393 ms on Thor）
    x_{t-1} = scheduler.step(v, x_t, t) # host 侧，毫秒级
输出：action_latents → postprocess → 控制指令
```

**dit 占绝对主导**；scheduler 在 CPU/GPU 上极轻。优化采样 = 优化 **有效 dit forward 次数** 与 **每步权重精度**。

---

### 9.2 UniPC — 默认多步 ODE 求解器

#### 9.2.1 原理

**UniPC**（Unified Predictor-Corrector）是 training-free 高阶 ODE 求解器，专为 diffusion / flow-matching **少步采样**设计。Cosmos3 封装：

- 调度器：`FlowUniPCMultistepScheduler`（`fm_solvers_unipc.py`，移植 diffusers UniPC）
- 采样环：`UniPCSampler.forward`（`unipc.py:32`）

每步循环：

```python
for timestep in timesteps:
    velocity_pred = velocity_fn(latent, timestep)   # 1× MoT forward
    latent = sample_scheduler.step(velocity_pred, timestep, latent)  # 高阶 multistep 更新
```

相对简单 Euler（`x += dt * v`），UniPC 用 **历史若干步 velocity** 做 predictor-corrector 外推，**同样 30 步下质量更好**，或 **更少步数达到可接受质量**。

#### 9.2.2 σ 调度与 shift

`set_timesteps(num_steps, shift=shift)`（`fm_solvers_unipc.py:180-188`）：

```python
sigmas = linspace(sigma_max, sigma_min, num_steps)   # 默认 [1.0 → 1/1000]
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # flow-matching warp
timesteps = sigmas * num_train_timesteps              # ×1000
```

| 参数 | Policy 默认 | 含义 |
|------|------------|------|
| `num_steps` | **30** | 去噪步数 = dit forward 次数 |
| `shift` | **10.0**（sample_args） | σ 曲线 warp；越大高 σ 区步长越密 |
| `num_train_timesteps` | 1000 | 连续时间离散化基数 |

训练 config 按分辨率另有 shift（`Cosmos3-Nano.yaml`：`256→3, 480→5, 720→10`）；Policy-DROID 480 tier 的 sample_args 用 `shift=10.0`，与训练 480 档的 `5` 不同——**推理 shift 以 sample_args / CLI 为准**，改 shift 会改变 σ 轨迹，需与步数一起做质量评估。

#### 9.2.3 Chameleon TRT 端侧怎么用

`Cosmos3PolicyTrtPipeline`（`chameleon/runtime/cosmos3_trt/pipeline.py:207-238`）：

```text
host（diffusers pipe）:
  pipe.scheduler.set_timesteps(steps)     # UniPC，与 PyTorch 推理一致
  for t in timesteps:
      v = dit.engine(...)                 # TRT，~393 ms
      latents = pipe.scheduler.step(v, t, latents)
      action_latents = action_scheduler.step(v_action, t, action_latents)
```

- **dit.engine** 只做单步 velocity；**UniPC 在 host** 跑（极轻）。
- vision / action 各有一份 scheduler 深拷贝，**timesteps 相同**，可独立 mask velocity。
- `guidance=1.0` → 每步 1 次 dit（无 CFG）。

#### 9.2.4 端侧减步策略（base Policy-DROID）

| 手段 | 机制 | Thor 预期 | 风险 |
|------|------|-----------|------|
| **减 `num_steps`** | 30→10→4，dit 环线性缩短 | 30→10 ≈ **3×** | 动作抖动、长程一致性下降 |
| **调 `shift`** | 改变 σ 分布，少步时补偿 | 需 grid search | 与训练分布偏移 |
| **UniPC 阶数** | `solver_order`（默认 2） | 边际 | 主要影响质量/步数 tradeoff |

**粗算**（`dit_feature.md` §3）：

```text
dit 环延迟 ≈ num_steps × 393 ms
  30 step → ~11.8 s
  10 step → ~3.9 s
   4 step → ~1.6 s
```

减步 **不改 engine、不增显存**，是端侧 **零工程成本** 的第一刀；但 base 16B 模型 **未针对极少步训练**，4 步质量需实测。

---

### 9.3 FixedStep — 蒸馏模型的固定 σ 调度

#### 9.3.1 与 UniPC 的本质区别

`FixedStepSampler`（`fixed_step.py`）面向 **DMD2 蒸馏 student**：

```text
UniPC（base 模型）:  模型是 smooth velocity field → 高阶 ODE 求解器逐步积分
FixedStep（蒸馏）:   模型是 discrete σ 上的 one-shot denoiser → 固定 σ 列表 + Euler 一步
```

文档明确（`fixed_step.py:11-13`）：

> DMD2 students are trained as one-shot denoisers at specific discrete sigmas, **not** as smooth score functions. **Incompatible with UniPC / EDM.**

#### 9.3.2 默认配置

`FixedStepSamplerConfig`（`model_config.py:109-122`）：

| 字段 | 默认 | 含义 |
|------|------|------|
| `t_list` | `[0.999, 0.75, 0.5, 0.25]` + 自动补 `0.0` | **4 步**固定 σ |
| `sample_type` | `"ode"` | Euler：`x_next = x_t + (σ_next - σ_cur) * v` |
| 备选 | `"sde"` | 每步 re-noise（随机性） |

每步（`:126-159`）：

```python
v_pred = velocity_fn(latent, sigma_cur * 1000)
x0_pred = latent - sigma_cur * v_pred
latent_next = latent + (sigma_next - sigma_cur) * v_pred   # ODE
# condition_mask：policy 下 z₀ 等 clean token 每步强制回写 reference
```

#### 9.3.3 推理如何选中

`inference.py:1351-1357`：

```python
is_distilled = model.config.fixed_step_sampler_config is not None
if is_distilled:
    sampler = model.fixed_step_sampler
    guidance = 1.0   # 蒸馏模型强制无 CFG
```

**Cosmos3-Nano / Policy-DROID 开源 checkpoint**：`fixed_step_sampler_config: null` → **走 UniPC，不走 FixedStep**。

#### 9.3.4 端侧含义

| 问题 | 结论 |
|------|------|
| 现有 Policy-DROID 能直接开 FixedStep 吗？ | ❌ 需 **蒸馏 checkpoint** + config 里填 `t_list` |
| 步数能到多少？ | 典型 **4 步**（默认 t_list），配合 CFG 蒸馏可达 1 步 |
| TRT 要改吗？ | dit.engine 仍单步；host 换 FixedStep 环 + 改 timesteps binding |
| 与减步 UniPC 比？ | 蒸馏 **算法上**为少步训练，4 步质量通常优于 base 模型硬减到 4 步 |

**端侧推荐路径**：若要 **4 步级**延迟，优先 **DMD2 蒸馏 + FixedStep**，而非 base 模型 UniPC 30→4。

---

### 9.4 bf16 精度

#### 9.4.1 框架默认

| 层级 | 配置 | 说明 |
|------|------|------|
| 模型权重 / 激活 | `precision: bfloat16`（`Cosmos3-Nano.yaml:76`） | MoT 主计算 |
| FSDP master | `fsdp_master_dtype: float32` | 训练 optimizer 用；推理无 |
| TRT engine | `build_cfg precision: "bf16"` | `dit.engine` ~28 GB = 16B × 2B |

Chameleon deploy yaml（`cosmos3_policy_droid_trt_deploy.yaml:26`）与 dit build cfg 均为 **bf16 固定 profile**——Thor 端侧 **已在 bf16 路径上**。

#### 9.4.2 bf16 解决了什么、没解决什么

| 维度 | bf16 效果 |
|------|-----------|
| **vs fp32** | 权重减半（~30 GB→~15 GB 理论；实测 engine ~28 GB 含 plan） |
| **vs fp16** | 同 2B/element，**带宽相同**（`dit_feature.md` §5.2 P2：BF16→FP16 **~0 加速**） |
| **数值** | 动态范围与 fp32 同指数位，policy 通常足够 |
| **Thor weight-bound** | bf16 已是最低「无损」精度；进一步加速靠 **FP8/NVFP4 权重量化** |

#### 9.4.3 端侧精度路线

```text
fp32  →  太慢，仅调试
bf16  →  ✅ 当前 TRT 默认（v1 deploy）
fp16  →  无带宽收益，不推荐
FP8   →  P0：weight-bound GEMM ~1.5–2×（需 modelopt PTQ + 重建 engine）
NVFP4 →  Thor Blackwell 进一步压缩（质量需验证）
```

bf16 **不是可调旋钮**（已默认），而是 **量化前的基线**。

---

### 9.5 三维优化如何叠加（Thor Policy）

```text
总 dit 时间 ≈ num_steps × 单步延迟
            ≈ num_steps × (权重读取成本 / 精度因子)
            ≈ num_steps × (393 ms / 量化加速比)
```

| 组合 | num_steps | 求解器 | 精度 | 粗算 dit 环 | 备注 |
|------|-----------|--------|------|------------|------|
| **当前默认** | 30 | UniPC | bf16 | ~11.8 s | deploy yaml |
| **减步** | 10 | UniPC | bf16 | ~3.9 s | 零工程，需测质量 |
| **蒸馏** | 4 | FixedStep | bf16 | ~1.6 s | 需蒸馏 ckpt |
| **量化** | 30 | UniPC | FP8 | ~6–8 s | 重建 engine |
| **量化+减步** | 10 | UniPC | FP8 | ~2–3 s | P0 组合 |
| **量化+蒸馏** | 4 | FixedStep | FP8 | ~0.8–1 s | 端侧理想目标 |

VAE encode/decode 与 dit 环 **相加**（encode ~0.46 s 一次性，`dit_feature.md` §3），不是乘数。

---

### 9.6 端侧决策表

| 目标 | 推荐 | 不推荐 |
|------|------|--------|
| **立刻降延迟、可接受质量风险** | UniPC `num_steps` 30→10～15 | 盲目 4 步（base 模型） |
| **4 步级、要稳质量** | 等/训 **DMD2 蒸馏 ckpt** + FixedStep | 硬减 UniPC 步数 |
| **单步再快 ~2×** | FP8/NVFP4 dit.engine | BF16→FP16 |
| **改 σ 曲线** | 调 `shift` + 固定步数 A/B | 只改 shift 不改步数 |
| **TRT 侧** | host 保留 UniPC/FixedStep，dit 仍单步 engine | 把 scheduler 塞进 engine（无必要） |

---

### 9.7 关键源码索引

| 主题 | 路径 |
|------|------|
| UniPC 采样环 | `cosmos_framework/model/generator/diffusion/samplers/unipc.py` |
| Flow UniPC 调度 | `cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py` |
| FixedStep 蒸馏采样 | `cosmos_framework/model/generator/diffusion/samplers/fixed_step.py` |
| 采样器选择 | `cosmos_framework/model/generator/omni_mot_model.py:468-489, 2519-2562` |
| 蒸馏检测 | `cosmos_framework/inference/inference.py:1351-1357` |
| FixedStep 配置 | `cosmos_framework/configs/base/defaults/model_config.py:109-122` |
| Policy sample_args | `cosmos_framework/inference/defaults/policy/sample_args.json` |
| TRT policy 去噪环 | `chameleon/runtime/cosmos3_trt/pipeline.py:207-238` |
| TRT bf16 build | `configs/build_configs/cosmos3_policy_droid_dit_step_build_cfg.py` |

---

### 9.8 蒸馏 checkpoint + FixedStep：原理与实操

#### 9.8.1 开源仓库里有什么、没有什么

| 组件 | cosmos-framework 开源状态 |
|------|---------------------------|
| **FixedStepSampler**（推理环） | ✅ 完整实现 + 单测（`fixed_step.py`, `fixed_step_test.py`） |
| **`fixed_step_sampler_config`**（模型 config） | ✅ 定义于 `model_config.py:109-122` |
| **推理自动切换** | ✅ `fixed_step_sampler_config != null` → 用 FixedStep + 强制 `guidance=1.0`（`inference.py:1351`） |
| **导出蒸馏 ckpt** | ✅ `export_model.py` 剥 `net_teacher.*`，student 权重导出 HF |
| **DMD2 训练环**（`DistillationCoreMixin`） | ❌ **未在开源树**；见 [Cosmos Cookbook 蒸馏](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/distillation/overview.html) |
| **Policy-DROID 蒸馏 recipe** | ❌ 无官方 TOML / launch shell |
| **Cosmos3-Nano-Distilled** | 测试脚本引用（`inference/_test/distilled.sh`），**T2V 向**，非 Policy |

**结论**：开源侧 **推理管线已就绪**；要做 Policy 端侧 4 步，需 **自己训蒸馏 student**（或等 NVIDIA 放出 Policy 蒸馏 ckpt），不能给 base Policy-DROID 只改 scheduler。

#### 9.8.2 DMD2 步蒸馏在训什么（与 FixedStep 的对应关系）

DMD2（Improved Distribution Matching Distillation）把 **Teacher**（base 模型，30 步 UniPC + CFG）压成 **Student**（同架构，**离散 σ 上的 one-shot denoiser**）。

```text
Teacher（训练目标）:
  30 步 UniPC，guidance=6（T2V）或 1（Policy）
  → 学的是「光滑 velocity field」，可 ODE 多步积分

Student（蒸馏产物）:
  只在固定 σ ∈ t_list 上被训练成 one-shot denoiser
  → 不能用 UniPC（会假设光滑场）；必须用 FixedStep + Euler 一步
```

训练时交替两阶段（Cookbook / cosmos-predict2.5 `distillation.md`）：

| 阶段 | 冻结 / 解冻 | 做什么 |
|------|-------------|--------|
| **Generator（Student）** | 冻 critic；训 student | 从噪声做 **few-step backward simulation**（σ 沿 `t_list`）；re-noise 到随机 t；Teacher 跑 cond+uncond 得 CFG target；Critic 给 fake score；DMD2 loss 反传 student |
| **Critic（fake score net）** | 冻 student；训 critic | 用 student 生成的样本 re-noise；拟合 denoising target；可选 GAN discriminator |

额外收益：**CFG 蒸馏进 student** → 推理单前向、无 cond/uncond 双路（与 Policy `guidance=1` 天然一致）。

典型超参（Transfer 1 Edge 案例，Cookbook）：LR ~5e-7，global batch 64，多节点 + FSDP，student_update_freq=5，guidance_scale=5（训练 Teacher CFG），数千～数万 iter。

#### 9.8.3 FixedStep 推理环逐步拆解

默认 `t_list = [0.999, 0.75, 0.5, 0.25]` → 自动补 `0.0` → **4 次 dit forward**。

每步（`fixed_step.py:126-159`，ODE 模式）：

```python
timestep = sigma_cur * 1000          # 喂给 MoT 的离散 t
v_pred = velocity_fn(latent, t)      # 1× dit forward

if sigma_next > 0:
    latent_next = latent + (sigma_next - sigma_cur) * v_pred   # Euler ODE
else:
    latent_next = latent - sigma_cur * v_pred                  # 最后一步 = x0

# Policy：z₀ 等 condition token 每步强制写回
if condition_mask is not None:
    latent_next = mask * condition_reference + (1-mask) * latent_next
```

与 UniPC 对比：

| | UniPC | FixedStep |
|---|-------|-----------|
| 每步 dit 次数 | 1 | 1 |
| 总步数 | 30（可调） | **len(t_list)-1**（通常 4） |
| 步间更新 | 高阶 multistep（用历史 v） | **一阶 Euler**，σ 固定 |
| 适用模型 | base smooth field | **DMD2 student** |
| condition mask | velocity mask（policy pipe） | **latent 级写回**（`condition_reference`） |

`generate_samples_from_batch`（`omni_mot_model.py:2529-2543`）在 FixedStep 时自动传入 `condition_reference` / `condition_mask`（policy 的 z₀ clean latent）。

#### 9.8.4 推理侧如何启用蒸馏 ckpt

**1. 模型 config 必须带 `fixed_step_sampler_config`**

蒸馏 checkpoint 的 `config.json`（或 DCP metadata）中应包含，例如：

```yaml
fixed_step_sampler_config:
  t_list: [0.999, 0.75, 0.5, 0.25]   # 与训练时 backward simulation 一致
  sample_type: ode                     # policy 用确定性 ODE
```

`OmniMoTModel.__init__`（`omni_mot_model.py:481-487`）据此构造 `FixedStepSampler`。

**2. 推理 CLI 自动行为**

```python
# inference.py:1351-1357
is_distilled = model.config.fixed_step_sampler_config is not None
if is_distilled:
    sampler = model.fixed_step_sampler
    guidance = 1.0   # 强制无 CFG
```

`num_steps` 应与 `len(t_list)-1` 一致（或省略，由 t_list 决定）。

**3. 测试入口（T2V 蒸馏）**

```bash
# cosmos_framework/inference/_test/distilled.sh
python -m cosmos_framework.scripts.inference \
    --checkpoint-path Cosmos3-Nano-Distilled \
    -i inputs/omni/t2v.json -o outputs/ ...
```

**4. 导出 HF**

```bash
python -m cosmos_framework.scripts.export_model \
    --src <dcp_distilled_student> -o <hf_dir>
```

`export_model.py:49-63` 会把蒸馏训练用的子类 config 规整为 base `OmniMoTModel`，并去掉 `net_teacher.*` 权重。

#### 9.8.5 Policy-DROID 端侧要做蒸馏：推荐流程

当前 **无官方 Policy-DROID 蒸馏 ckpt**。若要在 Thor 上 4 步 policy，需自研 pipeline：

```text
阶段 0 — Teacher 就绪
  └─ Cosmos3-Nano-Policy-DROID（已有）= Teacher
  └─ 推理基线：UniPC 30 步，~11.8 s dit 环（Thor）

阶段 1 —（可选）Knowledge Distillation 预热
  └─ Teacher 在 DROID 数据上生成 (obs, lang) → (action, rollout) 对
  └─ Student 回归 loss 对齐 Teacher 输出（轻量，Cookbook KD 段）
  └─ 建立 student 与 teacher 的粗对齐

阶段 2 — DMD2 步蒸馏（主阶段）
  └─ Student 初始化 = Teacher 权重（同 16B MoT 架构）
  └─ 额外网络：fake score critic（+ 可选 discriminator）
  └─ 数据：DROID 真实视频 / 机器人轨迹（需多样、与部署域一致）
  └─ `t_list` 与推理 FixedStep **必须一致**（如 4 点：[0.999,0.75,0.5,0.25]）
  └─ `causal_training_strategy` 可设 `teacher_forcing_dcm`（`model_config.py:235`，离散一致性蒸馏，与 AR 长视频相关；policy 需验证是否启用）
  └─ 训练：交替 generator / critic step；FSDP 多卡（DMD2 显存 ≈ 2–4× 普通 SFT）
  └─ 参考实现：**Cosmos Cookbook** [Predict 2.5 案例](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/distillation/distilling_predict2.5.html)（`DistillationCoreMixin` 在 cosmos-predict2.5 等仓库，需移植到 `OmniMoTModel` policy 路径）

阶段 3 — 验证
  └─ PyTorch：`fixed_step_sampler_config` 非空 → `generate_samples_from_batch` 4 步
  └─ 真机 / sim：对比 Teacher 30 步的 success rate、action MSE、轨迹平滑度
  └─ 导出 DCP → HF → Chameleon ONNX/TRT

阶段 4 — Chameleon TRT 对接
  └─ **dit.engine 不变**（仍单步 velocity）
  └─ **改 host 去噪环**：`Cosmos3PolicyTrtPipeline` 当前用 `pipe.scheduler`（UniPC）
  └─ 需新增 `FixedStepPolicyTrtPipeline` 或分支：
      for (sigma_cur, sigma_next) in zip(t_list[:-1], t_list[1:]):
          v = dit.engine(vision_tokens, t=sigma_cur*1000, ...)
          latents = euler_step(latents, v, sigma_cur, sigma_next)
          latents = vcm * z0 + (1-vcm) * latents    # condition 写回
  └─ `configs/cosmos3_policy_droid_trt_deploy.yaml`：`num_inference_steps: 4`
```

#### 9.8.6 `t_list` 设计要点

| 考虑 | 建议 |
|------|------|
| **步数** | 4 步是 Cookbook 常见默认；1 步极限压缩、质量风险最大 |
| **σ 取值** | 不用 `1.0` 而用 `0.999`（`model_config.py:119` 注释：避免 σ=1 数值边界） |
| **与训练一致** | 推理 `t_list` **必须**等于蒸馏 `backward_simulation` 用的 schedule |
| **Policy shift** | FixedStep 默认走 **固定 t_list**；若传 `shift` 会动态生成 schedule（`fixed_step.py:41-47`），仅当蒸馏也用同一公式时才可用 |
| **sample_type** | Policy 部署用 **`ode`**（确定性）；`sde` 每步 re-noise，控制环不推荐 |

#### 9.8.7 常见误区

| 误区 | 事实 |
|------|------|
| base Policy-DROID 改 `num_steps=4` + 仍用 UniPC | ❌ 模型未为离散 σ 训练，质量通常崩 |
| 只改 config `fixed_step_sampler_config` 不训 student | ❌ 仍会加载 base 权重，FixedStep 语义不匹配 |
| 蒸馏后仍开 CFG | ❌ 推理强制 `guidance=1.0`；CFG 应在训练期蒸馏进 student |
| 把 FixedStep 塞进 dit.engine | ❌ scheduler 应在 host；engine 只管单步 `v(σ)` |
| 期望蒸馏省显存 | ❌ 训练期更吃显存（Teacher+Student+Critic）；**推理**才省 |

#### 9.8.8 外部参考

- [Cosmos Cookbook — 蒸馏总览](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/distillation/overview.html)
- [Distill Cosmos Predict 2.5（DMD2 案例）](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/distillation/distilling_predict2.5.html)
- [cosmos-predict2.5 `docs/distillation.md`](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/distillation.md)（`DistillationCoreMixin` API 说明）

---

## 10. 其它工程优化

| 优化 | 位置 / 说明 |
|------|-------------|
| **Warmup** | `warmup: N`（`common/args.py`）预热 compile / CUDA graph，benchmark 前丢弃 |
| **Checkpoint 加载** | DCP 分布式加载；`vlm_processor_from_checkpoint` 避免重复下 base 模型 |
| **AVAE 就地 materialize** | checkpoint  bundled `sound_tokenizer/` 直接用作音频 tokenizer |
| **Guardrail** | 可 `offload_guardrail_models` 到 CPU 或整体关闭 |
| **下载重试** | 指数退避 + jitter（`common/args.py`） |
| **分布式错误同步** | `sync_distributed_errors()`：任一 rank 失败则全体 abort，避免 hang |
| **CP 确定性 seed** | 无 seed 时用确定性 fallback，保证 CP/CFG 各 rank 噪声一致 |
| **跳过已生成** | 同 output_dir 已有结果自动 resume skip |
| **Benchmark / Profile** | `--benchmark`、`--profile` 写 `benchmark.json` / profiler 报告 |

---

## 11. 与 Chameleon TRT 部署的关系

| 路径 | 优化重心 |
|------|----------|
| **cosmos-framework PyTorch** | 多卡并行、compile、CFG/CP、TaylorSeer、KV cache、批 packing |
| **Chameleon TRT**（[`cosmos3_trt_deploy.md`](../../models/cosmos3_trt_deploy.md)） | 图拆分：text_embed / vae_encode / **dit_step** / vae_decode；BF16 engine；trtexec profile 调优 |

二者解决不同部署场景：

- **多卡服务器 / 研究复现** → cosmos-framework + `latency`/`throughput` preset。
- **边缘单卡（如 Jetson Thor）** → TRT 分 stage + 固定 shape profile（见 `dit_feature.md` §3 延迟预算）。

Policy 默认 `guidance=1.0` 时，PyTorch 与 TRT 路径均受益 **CFG 单前向快路径**；T2V 高 guidance 时 CFG 成本显著。端侧单卡无法用 `cfgp` 并行，详见 **§4′ Jetson Thor 端侧 CFG 专项分析**（引导蒸馏 / `guidance_interval` / batch=2 打包）。

---

## 12. 关键源码索引

| 主题 | 路径 |
|------|------|
| 并行 preset / CLI | `cosmos_framework/inference/args.py`, `common/args.py` |
| Pipeline 创建 | `cosmos_framework/inference/inference.py` → `OmniInference` |
| 模型加载 | `cosmos_framework/inference/model.py` → `Cosmos3OmniModel` |
| 去噪环 / velocity_fn | `cosmos_framework/model/generator/omni_mot_model.py` |
| TaylorSeer | `cosmos_framework/model/generator/utils/taylorseer.py` |
| KV Memory | `cosmos_framework/model/generator/utils/memory.py` |
| Compile 配置 | `cosmos_framework/configs/base/defaults/compile.py` |
| 注意力前端 | `cosmos_framework/model/attention/frontend.py` |
| Nano 默认 config | `cosmos_framework/inference/configs/model/Cosmos3-Nano.yaml` |
| Ray latency 配置 | `cosmos_framework/inference/ray/configs/latency.yaml` |

---

## 13. 快速对照：该开哪些开关

| 场景 | 建议 |
|------|------|
| 单卡低延迟 T2V/Policy | `--parallelism-preset=latency`，`guidance=1.0`（若可） |
| 8 卡批量 JSONL | `--parallelism-preset=throughput`，提高 `max_num_seqs` |
| 32B Super OOM | 全卡 FSDP（默认 `dp_shard=WORLD_SIZE`） |
| 有 spare GPU | `--use-separate-pipeline-vision-decode-gpu` |
| 首包延迟敏感 | `warmup=1`，`compile_dynamic=false` |
| 在线服务 | Ray Serve + `latency.yaml` preset |
