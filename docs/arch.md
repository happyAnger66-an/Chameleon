# Chameleon 架构设计文档

[English](../README.md) | [中文](../README.zh-CN.md) | 架构文档（中文）

> 跨多平台(NVIDIA / AMD / Intel / CPU / 地平线)的端侧 VLA 模型 **量化 / 编译 / 推理 / 自定义算子** 框架。
> MVP 模型为 openpi 的 **pi0.5**(`pi05`)。
> 策略:**统一前端抽象 + 可插拔原生编译后端**。

本文是面向后续开发的设计总览,包含:端侧 VLA 推理特点分析、对标业界框架的调研结论、Chameleon 的分层架构与核心抽象、数据流、扩展指南,以及分阶段路线图。

---

## 1. 端侧 VLA(pi05)推理特点

源自对 `openpi/src/openpi/models_pytorch/pi0_pytorch.py` 的分析,这些特点是整个架构的设计前提:

- **三段式计算图**:`embed_prefix`(SigLIP 视觉 + Gemma 语言)→ `paligemma_with_expert`(LLM prefix)→ `denoise_step`(Gemma action expert)。可拆为三个独立编译单元:`vit` / `llm_prefix` / `action_expert`,KV / hidden state 跨单元传递。
- **去噪循环是延迟热点**:`sample_actions` 中 prefix 只前向一次并缓存 KV(`use_cache=True`),随后 `denoise_step` 按 flow-matching 迭代 `num_steps`(默认 10)次复用 prefix KV。优化重心在 `action_expert` 去噪环。
- **静态、可预分配**:batch=1、固定 `action_horizon` / 序列长度、固定去噪步数 → 适合静态 shape + AOT 编译 + CUDA Graph,而非服务端的 paged / continuous batching。
- **跨平台差异集中在两层**:
  1. 算子 kernel(attention / gemm / 量化)
  2. 编译工具链(NVIDIA→TensorRT、Intel→OpenVINO、地平线→BPU SDK、AMD/CPU→TVM)

这意味着:**上层模型定义、量化语义、编排逻辑应当平台无关,差异收敛到「编译后端」和「算子 kernel」两层**。

---

## 2. 业界框架调研结论(可借鉴点)

### 2.1 model_optimizer(已有实现,主要参考)

位置:`model_optimizer/src/model_optimizer`。

**做得好、值得保留的抽象:**

- `ArchitectureSpec` / `StageSpec`:模型无关的 stage 概念,每个 stage 声明 `supported_backends` / `quantizable`。
- Policy 与 Backend 分离:`PolicyAdapter` 对接 openpi,`BackendInstaller` 挂载运行时。
- 分阶段后端矩阵:`ServerConfig.resolve_stages()` 支持 vit / llm / denoise 混用不同后端。
- Feature 注册表 + Artifact manifest 血缘 + 薄 Workflow 编排(复用 CLI,不重复逻辑)。

**局限性(本框架要解决的):**

| 局限 | 表现 |
|------|------|
| 无 Platform 抽象 | TRT/ORT/Native 平行实现,新增 SNPE/CoreML 需大量 fork |
| NVIDIA 强绑定 | ModelOpt / TensorRT / CuTe DSL / CUDA Graph 全栈 NVIDIA |
| 注册键仅 `(architecture, backend)` | 缺 `platform` 维度 |
| Executor 接口不统一 | TRT / Native / ORT API 不一致 |
| 校准路径分裂 | pi05 shard / YOLO collector / generic 三套,无统一 `Calibrator` |
| 配置混杂 | `.py` + JSON + argparse,无统一 schema |

### 2.2 TVM / MLC-LLM(跨平台编译范式)

- **TVM**:`DeviceAPI` / `Target` / `CodeGen` 三层分离;Relax(图 IR)+ TIR(张量 IR)双层 IRModule;**BYOC**(Bring Your Own Codegen)是接入地平线 BPU 等专用 NPU 的标准路径。
- **MLC-LLM**:在 TVM 之上的垂直 LLM 编译器。关键模式:
  - **权重(跨平台共享)/ Model Library(平台相关)/ Runtime Engine(统一 API)三分离**。
  - `auto_target.py` 的 **PRESET + build_func**:用 preset 字符串绑定 Target 配置与产物打包方式。
  - 量化策略注册表(`q4f16_1` / `q4f16_awq` / `e4m3` …)+ `op.enable + fallback`(算子级 backend 切换)。

借鉴:`PlatformSpec` 即 TVM `Target` + MLC PRESET 的思想;地平线优先走 TVM BYOC;通用 CPU/AMD 走 TVM Relax + DLight。

### 2.3 TensorRT-Edge-LLM / sglang / vLLM(端侧推理 + 算子可插拔)

- **TensorRT-Edge-LLM(最相关,端侧范式)**:
  - 三阶段硬分离:**量化 → ONNX 导出 → Engine Build → C++ Runtime**,runtime 无 PyTorch 依赖。
  - `EngineExecutor`(薄 TRT 包装)/ `DecodingStrategy`(解码循环)/ Sampling 分离;`TensorRegistry` 声明式 I/O 绑定。
  - prefill / decode **双 optimization profile**;CUDA Graph;custom op 三段式(stub → ONNX schema → C++ plugin);kernel 按 SM 链接 artifact。
  - 已有 Alpamayo VLA 示例:VLM engine + action engine 链式,KV 跨 engine 传递。
- **sglang / vLLM(算子后端可插拔的注册模式)**:
  - `@register_attention_backend` / `AttentionBackendEnum` + selector:按 `platform + phase + dtype` 选 kernel。
  - `QuantizationConfig.get_quant_method(layer)`:按层类型分发量化方法。
  - KV 类型系统(`KVCacheSpec` / `KVQuantMode`),enum dispatch 而非字符串匹配。

借鉴:`Engine.run` 统一接口源自 `EngineExecutor`;`RuntimeRegistry` / `KernelRegistry` 源自 attention backend 注册;`QuantMetadata` 契约源自 Edge-LLM metadata + vLLM `get_quant_method`。

---

## 3. 整体架构

```mermaid
flowchart TB
    subgraph entry [入口/编排层]
        CLI[cli.py]
        WF[workflows WorkflowRunner]
        CFG[config TaskConfig pydantic+YAML]
        API[api.py 高层接口]
    end
    subgraph model [模型/架构层]
        ARCH[architectures ArchitectureSpec/StageSpec]
        MODEL[models ModelAdapter pi05]
    end
    subgraph optimize [优化/编译流水线]
        FE[frontend GraphCapture ONNX]
        QUANT[quantization QuantMethod+Calibrator+QuantMetadata]
        COMPILE[compile CompilerBackend 可插拔]
        KERNEL[kernels OpSpec + 平台 KernelImpl]
    end
    subgraph rt [运行时层]
        RTB[runtime RuntimeBackend/Engine 可插拔]
        ORCH[VLAOrchestrator 链式+KV handoff+去噪环]
    end
    subgraph plat [平台抽象层]
        PLAT[core PlatformSpec 能力描述]
    end
    CLI --> API --> WF --> CFG
    WF --> FE --> COMPILE
    QUANT --> COMPILE
    KERNEL -.算子注册.-> FE
    KERNEL -.平台kernel.-> COMPILE
    COMPILE -- Artifact+manifest --> RTB
    PLAT -.能力查询.-> QUANT
    PLAT -.能力查询.-> COMPILE
    PLAT -.能力查询.-> RTB
    RTB --> ORCH
    ARCH --> ORCH
    MODEL --> FE
    MODEL --> ORCH
```

### 分层职责

| 层 | 包 | 职责 |
|----|----|------|
| 平台抽象 | `core/platform.py` | `PlatformSpec` 描述 vendor/device/dtype/工具链/kernel_tag。量化/编译/运行时均查询它分流。**model_optimizer 缺失的核心抽象。** |
| 模型/架构 | `architectures/`、`models/` | `ArchitectureSpec`+`StageSpec`;`ModelAdapter` 暴露各 stage `nn.Module` 与 example inputs |
| 统一前端 | `frontend/` | `GraphCapture` 把 PyTorch stage 导出为平台中性图(ONNX,预留 torch.export) |
| 量化 | `quantization/` | `QuantMethod` 注册表 + `Calibrator` + `QuantMetadata` 契约 |
| 编译(核心) | `compile/` | `CompilerBackend.compile(graph, quant_meta, ctx) -> Artifact`,每平台一个实现 |
| 自定义算子 | `kernels/` | `OpSpec`(逻辑算子)+ 多平台 `KernelImpl`,三段式 |
| 运行时 | `runtime/` | `RuntimeBackend.load -> Engine.run`;`VLAOrchestrator` 驱动链式执行 + 去噪环 |
| 编排/配置 | `workflows/`、`config/`、`cli.py`、`api.py` | TaskConfig 驱动的 quantize→compile→infer |

---

## 4. 核心抽象与注册表

所有插件通过 **import 时副作用注册** 到泛型 `Registry`(`core/registry.py`),`import chameleon` 即填充全部注册表。

| 注册表 | 键 | 位置 |
|--------|----|----|
| `PLATFORM_REGISTRY` | `name` | `core/platform.py` |
| `ARCHITECTURE_REGISTRY` | `name` | `architectures/registry.py` |
| `MODEL_REGISTRY` | `name` | `models/base.py` |
| `GRAPH_CAPTURE_REGISTRY` | `name` | `frontend/base.py` |
| `QUANT_METHOD_REGISTRY` | `name` | `quantization/registry.py` |
| `CALIBRATOR_REGISTRY` | `(architecture, stage)` | `quantization/calibrate/base.py` |
| `COMPILER_REGISTRY` | `name`(= `PlatformSpec.compiler`) | `compile/base.py` |
| `RUNTIME_REGISTRY` | `name`(= `PlatformSpec.runtime`) | `runtime/base.py` |
| `KERNEL_REGISTRY` / `OP_REGISTRY` | `(op, vendor)` / `name` | `kernels/base.py` |
| `ORCHESTRATOR_REGISTRY` | `architecture` | `runtime/orchestrator.py` |

### 关键接口签名

```python
# core/platform.py
@dataclass(frozen=True)
class PlatformSpec:
    name: str; vendor: str; device: str
    dtypes: tuple[str, ...]
    compiler: str          # 默认 CompilerBackend key
    runtime: str           # 默认 RuntimeBackend key
    kernel_tag: str | None # "sm_87" / "bpu_j5"
    torch_device: str      # 参考运行时的 torch.device

# compile/base.py —— 可插拔编译后端(核心扩展点)
class CompilerBackend(ABC):
    name: str
    def available(self) -> bool: ...
    @abstractmethod
    def compile(self, graph: Artifact, quant_meta: QuantMetadata | None,
                ctx: CompileContext, cfg: dict | None) -> Artifact: ...

# quantization/base.py
class QuantMethod(ABC):
    name: str
    @abstractmethod
    def quantize(self, module, calibrator: Calibrator,
                 platform: PlatformSpec, config: QuantConfig
                 ) -> tuple[Any, QuantMetadata]: ...

# kernels/base.py —— 自定义算子三段式
class KernelImpl(ABC):
    op: str; platform_vendor: str
    def frontend_stub(self): ...       # torch.library custom op
    def graph_node(self, g, *ins): ... # ONNX symbolic / 图节点
    def backend_artifact(self, kernel_tag=None): ... # plugin .so / kernel lib
    @abstractmethod
    def reference(self, *args): ...    # 正确性参考(CPU/测试)

# runtime/base.py —— 统一执行接口
class Engine(ABC):
    @abstractmethod
    def run(self, inputs: dict) -> dict: ...
class RuntimeBackend(ABC):
    name: str
    @abstractmethod
    def load(self, artifact: Artifact, ctx: RunContext) -> Engine: ...
```

### QuantMetadata 契约

量化产出不仅是量化后的 module,还包含一份描述各组件数值格式的 `QuantMetadata`(`component_dtypes`:weight/activation/kv_cache…)。编译后端据此选 kernel / build flag,运行时据此 dispatch——避免字符串硬编码。

---

## 5. 数据流(pi05 端到端)

```
openpi checkpoint / 参考模型
   │  ModelAdapter.build()           → 三个 stage nn.Module
   ├─ quantize  QuantMethod.quantize(module, Calibrator, platform, QuantConfig)
   │                                  → 量化 module + QuantMetadata
   ├─ compile   GraphCapture.capture(module, example_inputs) → ONNX Artifact
   │            CompilerBackend.compile(onnx, quant_meta, ctx) → engine Artifact
   └─ infer     InferenceSession.build()  → 每 stage 按 stage_runtimes 选 RuntimeBackend.load → Engine
                VLAOrchestrator.infer(obs):
                   img_tokens   = vit_engine.run({"images": ...})
                   prefix_memory= llm_prefix_engine.run({...})        # KV,算一次
                   x_t = noise
                   for t in flow_matching_schedule(num_steps):        # 去噪热点
                       v_t = action_engine.run({state, prefix_memory, x_t, time_emb})
                       x_t = x_t + dt * v_t
                   return x_t                                         # [B, horizon, action_dim]
```

所有 stage 通过统一的 `Engine.run` 通信,因此可 **stage 级后端混用**(如 `vit=tensorrt, action_expert=pytorch`),由 `TaskConfig.stage_runtimes` 配置。`Manifest`(`chameleon_manifest.json`)记录每步 Artifact 血缘。

---

## 6. 当前实现状态

| 组件 | 状态 |
|------|------|
| `core` / `architectures` / `models(pi05)` / `runtime(pytorch)` / `VLAOrchestrator` / `config` / `cli` / `workflows` | **功能完整、可运行** |
| `frontend/onnx_export`(dynamo→legacy 回退 + modelopt 导出模式) | 完整 |
| `quantization`(int8/int8_sq/fp8/int4_awq/w4a8_awq/nvfp4,封装 ModelOpt) | 接口完整;有 modelopt 时真实插入量化器,缺失时降级为 metadata-only |
| `compile/tensorrt` | **可用**:三 stage 真实 build engine;支持插件预加载、FP16/INT8/FP8 flag、prefill/decode 双 optimization profile |
| `runtime/tensorrt`(`TensorRegistry` 声明式绑定 + 位置绑定 + 设备缓冲 + enqueueV3 + 可选 CUDA Graph) | **可用**:已验证 compile→infer 闭环,TRT vs PyTorch cosine=1.0(FP16 精度差 ~1e-3) |
| `kernels/fmha_d256` | 三段式:真实 `torch.library` custom op(`torch.ops.chameleon.fmha_d256`,eager=SDPA)+ ONNX symbolic + nvidia plugin 占位(按 kernel_tag 选 artifact) |
| `compile/openvino` / `compile/tvm` / `compile/horizon` | Stub,含集成方案说明,`NotImplementedError` |

**鲁棒性设计**:缺 modelopt / 特定工具链 / GPU 时,量化、编译、设备选择均优雅降级(记录 `compile_skipped` 血缘并继续),保证全链路在任意机器可跑。

### 阶段二已落地(NVIDIA 深化)

- **compile→infer 闭环**:`InferConfig.use_compiled_engines=true` 时,compile 产出的 engine 经 `stage_artifacts` 注入 `InferenceSession`,推理真实运行在 TRT engine 上(非 PyTorch 参考路径)。见 `configs/pi05_nvidia_trt.yaml`。
- **数值校验**:同权重下 TRT(FP16)与 PyTorch 输出 `cosine=1.000000`、`max_abs≈1.25e-3`。
- **TensorRT runtime**:`TensorRegistry` 发现 I/O、按位置绑定(规避 ONNX 名重命名)、持久化设备缓冲(去噪环复用)、`execute_async_v3`、可选 CUDA Graph 捕获/重放。
- **双 optimization profile**:编译器支持 `cfg["profiles"]`(context/prefill + generation/decode),runtime 按 `profile_index` 选择;静态 shape 的参考路径下为 no-op。
- **fmha_d256**:升级为真实 torch custom op + ONNX symbolic,nvidia 实现按 `kernel_tag`(sm_87/sm_101)解析 plugin artifact。
- **真实 openpi 权重**:`use_reference=false` + `checkpoint`(支持 .pt/.pth/.safetensors,partial-load 报告);真实模型按 `_OPENPI_STAGE_ATTR` 映射出三 stage 子模块。见 `configs/pi05_realweights.yaml`。

### 阶段二未尽事项(后续 bring-up)

- **量化模型 ONNX 导出**:modelopt 已量化模块(fake-quant 算子)经标准导出器翻译失败,当前优雅跳过 compile。需对齐 modelopt 的 ONNX QDQ 导出路径(版本敏感)。
- **真实模型编排**:真实 `PI0Pytorch` 经简化 `Pi05Orchestrator` 端到端推理需对齐 KV-cache plumbing(openpi `sample_actions` 的 prefix KV + adaRMS),当前仅支持按子模块量化/编译。
- **on-device**:Orin/Thor 实测、`fmha_d256` CuTe DSL 真实 plugin 构建与链接。

### 验证命令

```bash
chameleon platforms        # 列出 7 个平台
chameleon architectures    # pi05 三 stage
chameleon info             # 已注册的 compilers/runtimes/quant/kernels
chameleon infer    --config configs/pi05_cpu.yaml          # → action (1,50,32)
chameleon workflow --config configs/pi05_nvidia.yaml       # quantize→compile→infer(参考路径)
chameleon workflow --config configs/pi05_nvidia_trt.yaml   # compile→infer,推理跑在 TRT engine 上
chameleon profile  --config configs/pi05_cpu.yaml --runs 20
```

---

## 7. 扩展指南

### 新增一个平台

1. 在 `core/platform.py` 注册 `PlatformSpec`(指定 `compiler` / `runtime` / `dtypes` / `kernel_tag`)。
2. 实现并注册 `CompilerBackend`(`compile/<platform>/`)与 `RuntimeBackend`(`runtime/<platform>/`)。
3. (可选)为热点算子在 `kernels/` 注册该 vendor 的 `KernelImpl`。
4. 各 stage 在 `ArchitectureSpec.supported_platforms` 中声明可用即可。

### 新增一个量化方法

实现 `QuantMethod`,在 `quantization/methods/` 注册;`quantize()` 返回 `(module, QuantMetadata)`。

### 新增一个模型架构

1. 定义 `ArchitectureSpec`(stages + `orchestrator` key)。
2. 实现 `ModelAdapter`(`stage_module` / `example_observation` / `make_config`)。
3. 实现并注册对应 `Orchestrator`(若复用 pi05 的链式+去噪范式,可继承复用)。

---

## 8. 分阶段路线图

- **阶段一(已完成,本 MVP)**:全部核心抽象 + 注册表 + pi05 参考模型 + PyTorch 运行时 + VLAOrchestrator + TensorRT 编译路径,端到端跑通。
- **阶段二(NVIDIA 深化,主体已完成)**:TensorRT runtime 落地(`TensorRegistry` 声明式绑定 + prefill/decode 双 profile + 可选 CUDA Graph)、**compile→infer 闭环已打通并数值校验(cosine=1.0)**、`fmha_d256` 升级为真实 torch custom op + ONNX symbolic、真实 openpi 权重加载(多格式 + stage 子模块映射)。**未尽**:量化模型 ONNX QDQ 导出、真实模型经编排器端到端、Orin/Thor 实测与 CuTe DSL plugin 构建(见上节)。
- **阶段三(通用平台)**:接入 TVM(AMD GPU / 通用 CPU,Relax + DLight)与 Intel OpenVINO(+ NNCF INT8)。
- **阶段四(专用 NPU)**:地平线 BPU 经 TVM BYOC 或 `hb_mapper` 接入;跨平台 kernel 自动调度;权重 / Model Library / Runtime 三分离以支持 OTA。

---

## 9. 关键文件索引

| 职责 | 路径 |
|------|------|
| 平台抽象 | `chameleon/core/platform.py` |
| 架构定义 | `chameleon/architectures/pi05.py` |
| pi05 适配 + 参考模型 | `chameleon/models/pi05/{adapter,reference}.py` |
| ONNX 导出 | `chameleon/frontend/onnx_export.py` |
| 量化方法 | `chameleon/quantization/methods/modelopt_ptq.py` |
| TensorRT 编译 | `chameleon/compile/tensorrt/backend.py` |
| TensorRT 运行时(TensorRegistry/CUDA Graph) | `chameleon/runtime/tensorrt/backend.py` |
| 非 NVIDIA 编译 stub | `chameleon/compile/stubs.py` |
| 自定义算子示例 | `chameleon/kernels/fmha/fmha_d256.py` |
| 编排 + Session | `chameleon/runtime/orchestrator.py` |
| 高层 API | `chameleon/api.py` |
| 配置 schema | `chameleon/config/schema.py` |
| CLI | `chameleon/cli.py` |
