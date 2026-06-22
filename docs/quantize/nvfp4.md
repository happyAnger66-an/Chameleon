# NVFP4 量化与 `fp4qdq_to_2dq` 分析

本文聚焦 ModelOpt / model_optimizer 路径上 **NVFP4 W4A4** 从 PTQ、`set_dynamic_quant` 配置、PyTorch export，到 `fp4qdq_to_2dq` ONNX 后处理、TensorRT build 的完整链路。

---

## 1. 阶段分离（不要混为一谈）

| 阶段 | 做什么 | pi0.5 MO 路径 |
|------|--------|---------------|
| **PTQ**（`mtq.quantize`） | FP4 数值 + block/global scale 写入 checkpoint | ✅ 必做 |
| **`set_dynamic_quant`**（export 前） | 给 QuantLinear 打 ONNX export 标记：激活 dynamic、权重 static | ✅ NVFP4 必做 |
| **PyTorch repack**（export 前） | layout / view / MoE pack（如 `uint8→int8` view） | ❌ 一般不做 |
| **`fp4qdq_to_2dq`**（export 后） | ONNX 图改写 + 权重落盘为 `FLOAT4E2M1` | ✅ NVFP4 必做 |

```text
PTQ (mtq.quantize)
       │
       ▼
量化 checkpoint：FP4 数值 + block/global scale
       │
       ▼
set_dynamic_quant(model, dtype)   ◄── export 前配置 W4A4 策略（§3）
       │
       ▼
torch.onnx.export
  → 激活：dynamic → TRT_FP4DynamicQuantize + 两级 DQ（图内完整）
  → 权重：static  → TRT_FP4QDQ 占位节点，initializer 仍是 FP16/BF16
       │
       ▼
fp4qdq_to_2dq(onnx)          ◄── 仅处理 static 权重占位（§5 起）
  → 删 TRT_FP4QDQ，拆两级 DequantizeLinear
  → 新建 FLOAT4E2M1 weight + FP8/FP32 scale
  → 删旧 FP16/BF16 权重 initializer
       │
       ▼
build_engine / llm_build
```

**Edge-LLM 路径**若在 dynamo export 时已直接 emit 等价的两级 `DequantizeLinear`（不经过 `TRT_FP4QDQ`），则不必再跑 `fp4qdq_to_2dq`；PyTorch repack 在 `checkpoint/repacking.py::apply_all_repacking`，发生在 export **之前**，与本文函数无关。

---

## 2. 为什么需要 `fp4qdq_to_2dq`

ModelOpt 对 **静态权重** export 时，并不直接在 ONNX 里写 FP4 字节，而是插入占位自定义节点：

```python
# modelopt/torch/quantization/export_onnx.py
return g.op("trt::TRT_FP4QDQ", inputs, block_size_i=block_size)
```

此时 ONNX 里的 **权重 initializer 仍是 FP16/BF16**，`TRT_FP4QDQ` 只是标记「这里要做 NVFP4 静态双量化」。TensorRT 没有 `TRT_FP4QDQ` plugin，必须在 build 前展开为标准 `DequantizeLinear` 子图并写入 FP4 字节。

| 项 | 说明 |
|----|------|
| **函数签名** | `fp4qdq_to_2dq(onnx_model: onnx.ModelProto, verbose: bool = False) -> onnx.ModelProto` |
| **代码位置** | `Model-Optimizer/modelopt/onnx/quantization/qdq_utils.py:1508` |
| **状态** | Deprecated shim；新代码推荐 `NVFP4QuantExporter`（逻辑拆成 `compute_scales` → `compress_weights` → `post_process`） |
| **本函数特点** | 一步完成 scale 计算 + FP4 pack + 图改写（旧 API 兼容 Edge-LLM 0.6.1 及以前） |

---

## 3. Export 前配置：`set_dynamic_quant`

代码位置：`model_optimizer/src/model_optimizer/utils/utils.py`

```python
def set_dynamic_quant(model: nn.Module, dtype: str) -> None:
    for module in model.modules():
        if is_nvfp4_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half" if dtype == "fp16" else "BFloat16"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"
```

### 3.1 功能

在 **PTQ 完成之后、`torch.onnx.export` 之前**，遍历模型中所有 NVFP4 的 `QuantLinear`，给 `TensorQuantizer` 写入 **ONNX 导出专用私有属性**，实现 **W4A4** export 策略：

| 设置的属性 | 挂在谁身上 | 值 | 含义 |
|-----------|-----------|-----|------|
| `_onnx_quantizer_type` | `input_quantizer`（激活） | `"dynamic"` | export 时走 FP4 **动态量化** 子图 |
| `_onnx_quantizer_type` | `weight_quantizer`（权重） | `"static"` | export 时 emit **`TRT_FP4QDQ` 占位**，再由 `fp4qdq_to_2dq` 后处理 |
| `_trt_high_precision_dtype` | `input_quantizer` | `"Half"` / `"BFloat16"` | FP4 动态 Q/DQ 子图里 **高精度 Cast 的目标 dtype** |

**识别条件**（`is_nvfp4_linear`）：模块是 ModelOpt 量化 Linear，且 `input_quantizer.block_sizes["scale_bits"] == (4, 3)`。

**重要**：
- **不改权重数值**，只写 export 属性；PTQ 标定 / PyTorch 推理 **不读取** 这些字段。
- 对 **INT4 / FP8 / 纯 FP16** 模块是 **no-op**。
- docstring 写了 mxfp8，但 model_optimizer 这版 **只处理 NVFP4**；Edge-LLM 同名函数还覆盖 MXFP8。

### 3.2 谁消费、怎么用

**消费者**是 ModelOpt 的 ONNX symbolic 路径（`torch.onnx.export` trace 时）：

```text
torch.onnx.export trace
  → QuantLinear.forward
  → TensorQuantizer._fake_quantize
  → dynamic_block_quant(..., _trt_high_precision_dtype, _onnx_quantizer_type)
  → DynamicBlockQuantizationFunction.symbolic
  → export_fp4(..., trt_high_precision_dtype, onnx_quantizer_type)
```

`export_fp4` 按 `onnx_quantizer_type` 分支（`modelopt/torch/quantization/export_onnx.py`）：

| 量化对象 | `_onnx_quantizer_type` | export 产物 | 后续 |
|---------|------------------------|------------|------|
| **激活** | `"dynamic"` | `TRT_FP4DynamicQuantize` + 两级 `DequantizeLinear` | 一般 **不需要** `fp4qdq_to_2dq` |
| **权重** | `"static"` | `TRT_FP4QDQ` + FP16/BF16 initializer | **必须** `fp4qdq_to_2dq` |

`_trt_high_precision_dtype` 在 `_fp4_dynamic_quantize` 中：若 trace 输入 dtype 与目标不一致，先 **Cast 到 FP16/BF16** 再进动态量化（满足 TensorRT StronglyTyped 要求）。

### 3.3 pi0.5 调用时机与参数

典型顺序：

```text
quantize_model(...)              # PTQ 标定
       ↓
set_dynamic_quant(self, dtype)   # 仅设置 export 属性
       ↓
export(...)                      # torch.onnx.export
       ↓
[NVP4] fp4qdq_to_2dq             # 只处理 static 权重占位
```

`dtype` 约定：`"fp16"` → `"Half"`，其它（含 `"bf16"`）→ `"BFloat16"`。

| 调用位置 | `dtype` | 场景 |
|---------|---------|------|
| `pi05/llm.py` | `feature_config.quantize.dynamic_quant`（默认 `"bf16"`） | LLM stage 量化 + export |
| `pi05/dit.py`, `expert.py`, `vit.py`, `embed_prefix.py` | `"bf16"` | 各子 stage |
| `pi05/llm_with_cutedsl.py`, `llm_with_trtedgellm.py` | `"fp16"` | TRT / CuteDSL 路径 |
| `infer/server/policy_loader.py` | vit/llm/expert `"bf16"`，denoise `"fp16"` | PTQ bundle 推理加载 |
| `scripts/.../ptq_compare.py` | 同上 | PTQ 对比（不 export 时属性不会被读到） |

LLM stage 示例：

```python
# pi05/llm.py
dynamic_quant = self.feature_config.quantize.get("dynamic_quant", "bf16")
set_dynamic_quant(self, dynamic_quant)
self.export(export_dir, dynamo=dynamo)
```

### 3.4 与 ModelOpt 内置 context manager 的关系

ModelOpt 还提供 `configure_linear_module_onnx_quantizers`（对普通 `nn.Linear` 设 dynamic/static），但 pi0.5 走 **QuantLinear + 显式 `set_dynamic_quant`**，在 export 前按 NVFP4 模块精确打标，并指定 `_trt_high_precision_dtype`。

---

## 4. Export 前 ONNX 子图结构（static 权重）

典型单层 Linear 权重：

```text
linear.weight [FP16/BF16, shape=(N,K)]  ──initializer──┐
                                                       ▼
                                            TRT_FP4QDQ(block_size=16)
                                                       │
                                                       ▼
                                            fp4qdq_output [逻辑 FP16/BF16]
                                                       │
                                                       ▼
                                            MatMul(input, fp4qdq_output) → ...
```

**`TRT_FP4QDQ` 节点字段：**

| 字段 | 含义 |
|------|------|
| `node.input[0]` | 权重 initializer 名，如 `...q_proj.weight` |
| `node.output[0]` | 反量化后权重张量名；替换后由第二个 DQ **复用同名**，保证下游边不断 |
| `node.attribute["block_size"]` | NVFP4 block 大小，常见 **16** |

---

## 5. `fp4qdq_to_2dq` 整体流程

```text
1. 扫描 graph.node，收集所有 op_type == "TRT_FP4QDQ" 的节点
2. 检测 precision_dtype（Half / BFloat16），供后续 Cast 使用
3. 对每个 FP4QDQ 节点：
   a. 读 weight initializer → w32 (FP32)
   b. 计算 sw_f32_per_tensor（全局 scale）
   c. 计算 sw_f32_per_block（每 block scale）
   d. quantize → w_f32
   e. _cast_fp4 → w_f4 (packed bytes)
   f. _cast_fp8 → sw_f8_per_block
   g. _replace_fp4qdq_with_2dq（图改写）
   h. _cast_input_dtypes（给下游 MatMul 插 Cast）
4. 删除旧 FP16/BF16 weight initializer
5. return onnx_model
```

---

## 6. 函数内变量与 `onnx_model` 各字段作用

### 6.1 `onnx_model` 顶层结构

| 字段 | 本函数是否修改 | 作用 |
|------|----------------|------|
| `graph.node` | ✅ | 删 `TRT_FP4QDQ`；增 `DequantizeLinear`、`Cast` |
| `graph.initializer` | ✅ | 删旧 FP16/BF16 weight；增 FP4/FP8/FP32 scale |
| `graph.input` | ✅ | 新 initializer 也会挂到 `graph.input`（TRT 解析需要） |
| `graph.value_info` | ✅ | 更新 DQ 输出 dtype；新增 block scale 的 FP32 value_info |
| `graph.output` | ❌ | 不变 |

### 6.2 函数内建立的索引 / 辅助结构

| 变量 | 作用 |
|------|------|
| `initializers` | `graph.initializer` 的引用 |
| `initializers_to_delete` | 待删的旧权重名列表 |
| `initializer_indices` | initializer 名 → 列表下标 |
| `tensor_consumers` | 张量名 → 消费它的节点列表；用于找 FP4QDQ 下游 MatMul |
| `value_info_map` | 中间张量名 → `ValueInfoProto`；改 DQ 输出 elem_type |
| `graph_inputs` | 已有图输入名集合；避免新 initializer 重名 |
| `precision_dtype` | `"Half"` 或 `"BFloat16"`（扫描 initializer 是否含 BF16） |
| `fp4_qdq_nodes` | 所有 `TRT_FP4QDQ` 节点 |

### 6.3 循环内 numpy 中间量

| 变量 | shape / dtype | 含义 |
|------|---------------|------|
| `tensor` | `TensorProto` | 原始 weight initializer（FP16/BF16） |
| `w32` | `(N, K)` float32 | `read_f16_tensor_as_fp32(tensor)`，统一升到 FP32 |
| `sw_f32_per_tensor` | 标量 float32 | 全局 scale：`max(abs(w)) / 6.0 / 448.0` |
| `sw_f32_per_block` | `(N, K/block_size)` float32 | 每 block amax scale，再除以全局 scale |
| `w_f32` | `(N, K)` float32 | `quantize(...)` 后、cast FP4 前的 block-scaled 值 |
| `w_f4` | `(N/2, K)` uint8 | `_cast_fp4`：两枚 FP4 nibble pack 成 1 byte |
| `sw_f8_per_block` | 同 block scale shape | `_cast_fp8`：block scale 压成 FLOAT8E4M3FN |

**Scale 计算公式**（`modelopt/onnx/quantization/quant_utils.py`）：

```python
# 全局 per-tensor scale
sw_f32_per_tensor = max(abs(w)) / 6.0 / 448.0

# 每 block scale（reshape 为 (N, K/block_size, block_size) 后在最后一维取 amax）
per_block_scale = max(abs(block)) / 6.0
sw_f32_per_block = per_block_scale / sw_f32_per_tensor
```

**`quantize`** 按两级 scale 缩放权重：

```python
w_block = w.reshape(..., K/block_size, block_size)
w_f32 = w_block / (sw_f32_per_block[..., None] * sw_f32_per_tensor)
w_f32 = w_f32.reshape(N, K)
```

---

## 7. `_replace_fp4qdq_with_2dq` 图改写

代码位置：`modelopt/onnx/export/nvfp4_exporter.py`

对每个 FP4QDQ 节点，新建 **3 个 initializer + 2 个 DequantizeLinear**：

| 新 initializer 名 | ONNX 类型 | 内容 |
|-------------------|-----------|------|
| `{weight}_f4` | `FLOAT4E2M1` | packed FP4 字节（`raw=True`） |
| `{weight}_f8_scale` | `FLOAT8E4M3FN` | block scale（FP8 字节） |
| `{weight}_f8_scale_f32_scale` | FP32 | 全局 per-tensor scale（标量） |

**dims 注意**：`w_f4_proto.dims = [w_f4.shape[0] * 2, *w_f4.shape[1:]]` —— 物理存储第一维减半（pack），逻辑 shape 仍按未 pack 的 `(N, K)` 声明。

**替换后子图：**

```text
{weight}_f8_scale [FP8] ──┐
{weight}_f8_scale_f32_scale [FP32] ──┤
                                      ▼
                            DequantizeLinear  (DQ #1)
                                      │
                                      ▼
                            {weight}_f32_scale [FP32]
                                      │
{weight}_f4 [FLOAT4E2M1] ─────────────┤
                                      ▼
                            DequantizeLinear  (DQ #2, axis=-1, block_size=16)
                                      │
                                      ▼
                            {原 node.output[0]}  (dtype → FLOAT)
                                      │
                                      ▼
                            MatMul / Gemm ...
```

- **DQ #1**：`DequantizeLinear(sw_f8, sw_f32_global) → sw_f32`，把 FP8 block scale 还原成 FP32 block scale 张量
- **DQ #2**：`DequantizeLinear(w_f4, sw_f32, axis=-1, block_size=…)`，NVFP4 权重的 block dequant；输出名 **复用** 原 `node.output[0]`

然后删除原 `TRT_FP4QDQ` 节点，旧 `{weight}` initializer 加入删除列表。

---

## 8. `_cast_input_dtypes`：MatMul 输入 Cast

DQ #2 输出被标成 FLOAT，而 MatMul 期望 FP16/BF16。函数对 FP4QDQ 输出的 **第一个下游节点** 处理：

1. 若下游是 `Transpose`，再追到其后的 `MatMul`
2. 对 MatMul 的 **前两个输入**（activation + 权重路径，**不含 bias**）各插 `Cast → FP16/BF16`
3. Cast 输出名加 `_f16` 后缀，并替换 MatMul 对应 input

---

## 9. 变换前后对比（单个权重）

**Before：**

```text
graph.initializer:
  linear.weight          [FP16, N×K]

graph.node:
  TRT_FP4QDQ(input=linear.weight, block_size=16) → fp4qdq_output
  MatMul(input, fp4qdq_output) → output
```

**After：**

```text
graph.initializer:  (linear.weight 已删)
  linear.weight_f4                    [FLOAT4E2M1, packed]
  linear.weight_f8_scale              [FLOAT8E4M3FN]
  linear.weight_f8_scale_f32_scale    [FP32 scalar]

graph.node:
  DequantizeLinear(f8_scale, f32_scale) → linear.weight_f32_scale
  DequantizeLinear(weight_f4, f32_scale, axis=-1, block_size=16) → fp4qdq_output
  Cast(input → input_f16)
  Cast(... → ..._f16)    # MatMul 第二输入
  MatMul(input_f16, fp4qdq_output_f16) → output
```

---

## 10. 「INT8 → FLOAT4E2M1」澄清

容易混淆的三件事：

| 说法 | 实际含义 | 发生在哪 |
|------|----------|----------|
| **INT8 → FLOAT4E2M1（类型标签）** | 部分 Edge 路径把 packed 字节先用 INT8/UINT8 initializer 承载，再标成 ONNX 枚举 `FLOAT4E2M1` | Edge dynamo post-export |
| **`_cast_fp4` 输出 `uint8`** | 物理存储用 numpy `uint8`；写入 ONNX 时 `data_type=FLOAT4E2M1`，不是 INT8 量化语义 | `fp4qdq_to_2dq` 内部 |
| **从 FP16/BF16 生成 FP4** | 读高精度权重 → 按 block 重新量化 → pack → 写入新 initializer | **`fp4qdq_to_2dq` 主路径** |

**pi0.5 MO 主路径**：输入是 export 留下的 **FP16/BF16 initializer**，在本函数里重新量化并 pack，**不是** PTQ 后再走 INT8 整型权重。`FLOAT4E2M1` 是 ONNX 对 packed FP4 字节的类型名。

---

## 11. pi0.5 调用链与顺序

```text
quantize_model
  → set_dynamic_quant(model, dtype)          # export 前 W4A4 配置（§3）
  → export
  → [NVFP4] fp4qdq_to_2dq(onnx_model)
  → apply_denoise_onnx_post_export_patches   # MatMul/bias dtype 对齐，与本函数正交
  → build
```

典型入口（`model_optimizer`）：

- `src/model_optimizer/models/pi05/llm.py::_nvfp4_post_processing`
- `src/model_optimizer/models/model.py`（通用 stage）
- `llm.py` 内：`onnx.load` → `fp4qdq_to_2dq` → 删旧 external data → `save_as_external_data`

**激活侧**：`export_fp4` 在 `onnx_quantizer_type == "dynamic"` 时直接 emit FP4 dynamic quant + DQ 子图，**不**经过 `TRT_FP4QDQ`；`fp4qdq_to_2dq` 主要处理 **静态权重** 节点。

---

## 12. 使用注意

1. **必须有 `value_info`**：`_replace_fp4qdq_with_2dq` 要求 `node.output[0]` 在 `value_info_map` 中（export 后需跑 shape inference）。
2. **权重第一维需为偶数**：`_cast_fp4` 要求 `shape[0] % 2 == 0`（两 FP4 一字节 pack）。
3. **`K % block_size == 0`**：`get_weights_scaling_factor` 有 assert。
4. **与 PTQ 的关系**：PTQ 在 checkpoint 里已有量化参数；本函数从 export 留下的 **FP16/BF16 initializer 重新算 scale 并 pack**，数学与 NVFP4 recipe 一致，但是 export 后的 **再量化一步**。
5. **新 API**：推荐 `NVFP4QuantExporter`，可将 scale 计算与 weight compress 拆分缓存，post_process 阶段做图改写。

---

## 13. 关键代码索引

| 主题 | 路径 |
|------|------|
| `set_dynamic_quant` | `model_optimizer/src/model_optimizer/utils/utils.py` |
| `export_fp4` / `_fp4_dynamic_quantize` | `Model-Optimizer/modelopt/torch/quantization/export_onnx.py` |
| `fp4qdq_to_2dq` | `Model-Optimizer/modelopt/onnx/quantization/qdq_utils.py` |
| `_replace_fp4qdq_with_2dq` / `_cast_fp4` | `Model-Optimizer/modelopt/onnx/export/nvfp4_exporter.py` |
| Scale / quantize | `Model-Optimizer/modelopt/onnx/quantization/quant_utils.py` |
| Export 占位 `TRT_FP4QDQ` | `Model-Optimizer/modelopt/torch/quantization/export_onnx.py` |
| pi0.5 post-export | `model_optimizer/src/model_optimizer/models/pi05/llm.py` |
| Edge PyTorch repack | `TensorRT-Edge-LLM/tensorrt_edgellm/checkpoint/repacking.py` |
