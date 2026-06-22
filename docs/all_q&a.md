# Chameleon 项目问题与修复汇总

本文档跟踪 **代码实现、架构演进、内存/性能优化、导出与编译路径** 等问题及修复方案。

不记录环境安装、依赖配置、pip/venv 类问题（这类问题见 README 或各模块注释）。

**后续项目相关的代码级问题总结均追加到本文档。**

---

## 2026-06 — pi05 TRT 部署（ONNX 导出 + engine build）

### Q1：支持真实 pi05 的 ONNX 导出与 TRT engine build

**背景**：Chameleon 原有 `run_compile()` 仅覆盖 reference 三 stage（vit / llm_prefix / action_expert），无法对接 openpi pi05 的五段式 TRT 路径（vit / llm / expert / denoise）。

**代码改动**：
- 新增 `chameleon/deploy/`：`pi05/` 分 stage exporter、`trt_build.py`、`paths.py`。
- 扩展 `TaskConfig`：`deploy`、`export`、`actions: export`。
- CLI：`chameleon export`；`deploy.backend=pi05` 时 `compile` 走 TRT build。
- 示例配置：`configs/pi05_libero_trt_deploy.yaml`；build_cfg：`configs/build_configs/`。

**入口**：
```bash
PYTHONPATH=. models/openpi/.venv/bin/python -m chameleon.cli export \
  --config configs/pi05_libero_trt_deploy.yaml
```

---

### Q2：`export` 报错 `SyntaxError: 'return' outside function`

**现象**：`chameleon/deploy/paths.py` 第 52 行 `return root` 在函数外。

**原因**：重构时误删 `resolve_model_optimizer_root()` 函数定义（该函数后在 Q3 中已移除）。

**修复**：补回函数定义；后续去 model_optimizer 依赖时改为 Chameleon 本地路径解析。

---

### Q3：去 model_optimizer 依赖，部署逻辑内化到 Chameleon

**需求**：导出与 build 不再 `import model_optimizer` 或依赖 sibling 仓库。

**代码改动**：
- `chameleon/deploy/pi05/` 内置 vit / llm / expert / denoise exporter。
- `chameleon/deploy/trt_build.py` 迁入 `build_engine`（去掉 artifact 记录）。
- `deploy.backend=pi05`（`pi05_openpi` 兼容别名）；build_cfg 默认 `configs/build_configs/`。

**相关文件**：`chameleon/deploy/pi05/*`、`trt_build.py`、`paths.py`

**待办**：`embed_prefix` 导出未实现；denoise ONNX post-export dtype 修补未移植。

---

### Q4：多 stage 导出 GPU OOM（12GB 显存）

**现象**：vit 导出成功后，llm 阶段 `torch.OutOfMemoryError`，GPU 已占 ~10GB。

**原因**：
1. 每 stage 单独 `load_pi05_model()` → `create_trained_policy` 默认上 CUDA，整模重复加载。
2. 上一 stage 子模块 `.cuda()` 后未迁回 CPU，显存未释放。

**修复（内存优化）**：
| 改动 | 说明 |
|------|------|
| CPU 加载 | `load_pi05_model(..., device="cpu")` + selective bf16 |
| 单次加载 | `export_pi05_stages()` 只 load 一次，顺序导出四 stage |
| stage 间回收 | `release_export_cuda_memory()`：`cpu()` + `gc` + `empty_cache()` |
| ViT dynamo 默认关 | `dynamo=False`，减少 dynamo 额外显存与 trace 开销 |

**相关文件**：`loader.py`、`export.py`、`memory.py`、`vit.py`

**补充**：仍紧张时可分 stage 导出（yaml `export` 只保留当前 stage）。

---

### Q5：expert 导出 AdaRMS 与 dynamo 不兼容

**现象**：
```
AttributeError: 'GemmaRMSNorm' object has no attribute 'weight'
```
堆栈：`GemmaRMSNorm.extra_repr` → `self.weight.shape`。

**原因**：expert 默认 `dynamo=True`；Dynamo 导出前 `repr(model)` 触发 HF `extra_repr`，而 pi05 expert 使用 **AdaRMS**（`dense` 调制，无 `weight`）。

**修复**：`expert.py` 默认 **`dynamo=False`**（与 vit / llm / denoise 一致）；导出后 `finally` 释放临时 tensor。

**相关文件**：`chameleon/deploy/pi05/expert.py`

---

### Q6：TRT build 报错 `NetworkDefinitionCreationFlag` 无 `EXPLICIT_BATCH`

**现象**：
```
AttributeError: ... NetworkDefinitionCreationFlag ... has no attribute 'EXPLICIT_BATCH'
```
发生在 `trt_build.build_engine` → `_network_creation_flags`。

**原因**：TensorRT **10.x** 移除 `EXPLICIT_BATCH`（网络默认为 explicit batch）；旧代码无条件设置该 flag。

**修复**：新增 `chameleon/compile/tensorrt/compat.py`：
- TRT 8/9：仍设置 `EXPLICIT_BATCH`（及可选 `STRONGLY_TYPED`）。
- TRT 10+：`create_network()` 无 flag，或仅加 `STRONGLY_TYPED`。
- `deploy/trt_build.py` 与 `compile/tensorrt/backend.py` 均改用 `create_onnx_network()`。

**相关文件**：`compile/tensorrt/compat.py`、`deploy/trt_build.py`、`compile/tensorrt/backend.py`

---

### Q7：TRT build 报错 `BuilderFlag` 无 `PREFER_PRECISION_CONSTRAINTS`

**现象**：
```
AttributeError: ... BuilderFlag ... has no attribute 'PREFER_PRECISION_CONSTRAINTS'
```
发生在 `build_engine` 默认 `precision_constraints="prefer"` 分支。

**原因**：部分 TensorRT 10.x Python 绑定未暴露 `PREFER_PRECISION_CONSTRAINTS` / `OBEY_PRECISION_CONSTRAINTS`（layer 精度约束策略为旧 API 能力）。

**修复**：`compat.set_builder_flag_if_present()` + `apply_precision_constraints_policy()`：flag 不存在时打 log 并跳过，不中断 build；`apply_builder_precision_flags` / `CUDA_GRAPH` 同样走安全设置。

**说明**：日志里 `could not open ... memory.limit_in_bytes` 为 TRT verbose 读 cgroup 失败，**无害**，与本次崩溃无关。

**相关文件**：`compile/tensorrt/compat.py`、`deploy/trt_build.py`

---

### Q8：`chameleon stats` — 整模型计算量与访存量统计

**需求**：在部署前估算一次完整推理的 MACs/FLOPs 与理论访存量，与 `chameleon profile`（仅延迟）互补。

**命令**：
```bash
chameleon stats --config configs/pi05_cpu.yaml
chameleon stats --config configs/pi05_libero_trt_deploy.yaml --dry-run
chameleon stats --config configs/pi05_libero_trt_deploy.yaml --measured --format json
```

**行为**：
| 项 | 说明 |
|----|------|
| 路径选择 | 自动：`deploy.backend=pi05` → deploy 路径；`use_reference=false` → real（stage 汇总）；否则 reference 三 stage |
| denoise/expert | 配置含 `denoise` 时去噪环只计 `denoise×num_steps`，跳过独立 `expert`（与 TRT 运行时一致） |
| 计算量 | PyTorch `FlopCounterMode`；`FLOPs = 2 × MACs` |
| 访存量 | 理论：权重 + 激活 + attention 中间张量；`--measured` 时用 profiler 校验（需 CUDA） |
| 形状 | deploy 路径读 `configs/build_configs/` 的 `opt_shapes`；reference 用 adapter 默认尺寸 |

**相关文件**：`chameleon/profile/compute_stats.py`、`execution_plan.py`、`counters.py`；`chameleon/deploy/pi05/stats.py`；`chameleon/commands/stats.py`

**限制**：理论访存未建模 cache reuse / TRT fusion；Softmax/LayerNorm 等小算子可能漏计；`embed_prefix` 未纳入。

### Q9：TRT layer profile（`trt_profile`）与 compile / stats 的区别

**compile**：从 ONNX **构建** TensorRT engine（Python TensorRT API / `trt_build.py`）。

**trt_profile**：对已有 `engines/{stage}.engine` 运行 **`trtexec --loadEngine --dumpProfile --exportProfile`**，产出各 layer 的 **实测 latency**（`profiles/{stage}.profile.json`），**不重建** engine。

**stats**（§Q8）：PyTorch 路径上的 **理论** MACs/FLOPs 与访存近似；与 trtexec layer timing 互补。

**形状 / plugin 对齐**：trtexec 的 `minShapes/optShapes/maxShapes` 与 `--plugins` 从各 stage 的 `configs/build_configs/*.py` 读取（与 compile 相同）。llm 若 build 时需要 AttentionPlugin，须在 `profile.plugin_lib_paths` 或 build_cfg 中配置相同 `.so` 路径。

**WebUI**：

| `profile.viewer` | 行为 |
|------------------|------|
| `static` | 写 `profiles/index.html` + `manifest.json` |
| `webui` | 仅阻塞起本地 HTTP 服务 |
| `both` | 先写静态文件，再起服务（workflow 末尾） |

与 eval WebUI（LeRobot WebSocket 相机流）**独立**，不复用其协议。

```bash
chameleon trt-profile --config configs/pi05_libero_trt_deploy.yaml
chameleon draw profile --config configs/pi05_libero_trt_deploy.yaml
chameleon draw profile output/.../vit.profile.json
```

**相关文件**：`chameleon/deploy/trt_profile.py`；`chameleon/draw/trt_profile_viewer.py`；`chameleon/commands/trt_profile.py`、`draw.py`

**限制**：需 `trtexec` 在 PATH；大 engine profile 可能数分钟；`denoise` 已含 expert，yaml 可只 profile vit/llm/denoise。

---

## 文档维护约定

**收录范围**：架构/功能实现、bug 修复、内存与性能优化、导出/build 路径、评测/推理逻辑等 **代码层** 问题。

**不收录**：pip 安装、venv、CUDA/TRT 版本匹配、环境变量调优等运维类问题。

新增条目模板：

```markdown
### Qn：问题标题

**现象**：…

**原因**：…

**修复 / 代码改动**：…

**相关文件**：…
```
