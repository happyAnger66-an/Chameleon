# Cosmos3 TRT 部署（Policy / action）操作文档

在 Chamleon 中按 pi05 的 `export → compile → TRT infer` 范式，落地 Cosmos3 真实权重的
TensorRT 推理。v1 目标：**bf16、固定 profile**；优先 **Policy-DROID**，并复用到
**Cosmos3-Nano action**。

> 相关设计：模型结构见 [`cosmos3_policy_detail.md`](./cosmos3_policy_detail.md)、
> [`cosmos3_mot.md`](./cosmos3_mot.md)；与 pi05 对比见 [`cosmos3_vs_pi05.md`](./cosmos3_vs_pi05.md)。

---

## 1. 流水线概览（对齐 pi05）

| pi05 | Cosmos3 TRT | 次数 |
|------|-------------|------|
| vit | `vae_encode`（观测→归一化 latent z0） | 1× / 帧 |
| llm prefill | `text_embed` + host 静态 pack | 1× / 帧 |
| denoise | `dit`（MoT 单步 velocity） | N× |
| — | `vae_decode`（可选 rollout 可视化） | 1× |
| host: Euler | host: **UniPC**（`pipe.scheduler`） | — |

Host 负责：JSON caption + tokenize、mRoPE 联合序列打包、`prepare_latents` 等价初始化、
velocity 掩码、scheduler.step。TRT engine 只承担热点子图。

```
观测图 → [TRT vae_encode] → z0 ┐
prompt → tokenize → 静态 pack ─┤→ 去噪环 ×N：[TRT dit] → mask → UniPC.step ┐
                              └ action 噪声初始化 ────────────────────────┴→ action chunk [16,10]
                                                                          └→ (可选)[TRT vae_decode] → video
```

---

## 2. 四个 stage engine（固定 profile）

固定 profile 定义在 [`chameleon/deploy/cosmos3/shapes.py`](../../chameleon/deploy/cosmos3/shapes.py)
（`POLICY_DROID` / `NANO_ACTION`）。**改任一 shape 需重新 export + build 全部 stage**
（与 pi05 `PI05_LIBERO_PREFIX_LEN` 同理）。

| stage | ONNX 输入 | 输出 | 模块 |
|-------|-----------|------|------|
| `vae_encode` | `video [1,3,17,480,832]` | `vision_latent [1,16,5,30,52]` | `WanVaeEncodeExport`（含归一化） |
| `text_embed` | `input_ids [256]` | `text_emb [256, hidden]` | `transformer.embed_tokens` |
| `dit` | `vision_tokens / vision_timesteps / action_tokens / action_timesteps` | `v_vision, v_action` | `Cosmos3DitStepExport` |
| `vae_decode` | `latent [1,16,5,30,52]` | `video [1,3,17,480,832]` | `WanVaeDecodeExport`（含反归一化） |

`dit` 是最大 engine：静态联合序列字段（`input_ids` / `position_ids` / `*_indexes` /
`token_shapes` / `noisy_frame_indexes` / `action_domain_ids`）在 `Cosmos3DitStepExport`
构造时 `register_buffer` 固化；forward 只接收逐 step 变化的 4 个动态张量。

> **Policy-DROID 全画布 latent**：v1 不做 padding 移除（`_remove_action_video_padding_from_latent`），
> 保证 `vae_encode` 输出与 `dit` 输入 shape 恒定。这是与 PyTorch 的一处已知差异，v2 可引入
> 动态 profile 或 content-crop 对齐。

---

## 3. 一键命令

### 3.1 Policy-DROID

```bash
# 打印计划（不需 GPU/权重）
PYTHONPATH=. python -m chameleon.cli workflow \
  --config configs/cosmos3_policy_droid_trt_deploy.yaml --dry-run

# export（ONNX，需 CUDA + 权重）+ compile（engine，需 TensorRT）+ trt_profile
PYTHONPATH=. python -m chameleon.cli workflow \
  --config configs/cosmos3_policy_droid_trt_deploy.yaml

# TRT-only 推理评测（host diffusers + TRT engine）
PYTHONPATH=. python -m chameleon.cli eval \
  --config configs/cosmos3_policy_droid_trt_eval.yaml

# PyTorch vs TRT 双路对比（max/mean diff）
PYTHONPATH=. python -m chameleon.cli eval \
  --config configs/cosmos3_policy_droid_pt_trt_compare.yaml
```

### 3.2 Cosmos3-Nano action（复用同一 export/build 代码）

```bash
PYTHONPATH=. python -m chameleon.cli workflow \
  --config configs/cosmos3_nano_action_trt_deploy.yaml
```

---

## 4. 代码结构

| 文件 | 作用 |
|------|------|
| `deploy/cosmos3/real/vae.py` | Wan VAE encode/decode + latent 归一化/反归一化 |
| `deploy/cosmos3/real/pack.py` | host 联合序列静态打包（`build_policy_pack`） |
| `deploy/cosmos3/real/dit_step.py` | `Cosmos3DitStepExport`（静态 buffer + 动态 4 输入） |
| `deploy/cosmos3/real/export_stages.py` | 4 stage 真实权重 ONNX 导出 |
| `deploy/cosmos3/export.py` | `use_reference=false` 时切到 real exporters |
| `deploy/cosmos3/shapes.py` | `Cosmos3Profile` + `POLICY_DROID` / `NANO_ACTION` |
| `configs/build_configs/cosmos3_{policy_droid,nano_action}_*_build_cfg.py` | 固定 profile build_cfg ×4 |
| `runtime/cosmos3_trt/engines.py` | 加载 4 stage TRT engine |
| `runtime/cosmos3_trt/pipeline.py` | `Cosmos3PolicyTrtPipeline`（policy 去噪内核） |
| `runtime/cosmos3_trt/adapter.py` | host pipeline 加载 + 观测→画布 |
| `evaluate/cosmos3_trt_runner.py` | `cosmos3_trt_only` runner |
| `evaluate/cosmos3_pt_trt_compare_runner.py` | `cosmos3_pt_trt_compare` runner |

---

## 5. v1 范围与 v2 路线

**v1（本次实现）**
- bf16 ONNX → TRT，固定单 profile
- `guidance_scale=1` 单路 dit，`enable_sound=False`
- Policy-DROID + Nano action 共用 export/build 代码

**v2（不在本次范围）**
- **CFG（guidance>1）**：dit 需 cond/uncond 双 forward 或双输入 dit profile
- **Quantize**（int8/fp8/nvfp4）：ModelOpt → QDQ ONNX，参考 [`docs/quantize/nvfp4.md`](../quantize/nvfp4.md)
- **prefix KV cache**：text + z0 静态 K/V，仅 gen 段 denoise（见 [`cosmos3_vs_pi05.md`](./cosmos3_vs_pi05.md) §9）
- **DROID dataloader + 反归一化**：真实观测 I/O 与动作反归一化（pipeline 不含）
- **16B 显存**：export/build 分 stage、加载后释放 PyTorch 权重、dit engine 单独 load
- **动态 profile / content-crop**：对齐 PyTorch 的 padding 移除

---

## 6. 验收（Done 定义）

- [x] `export → compile → trt_profile` 计划链（dry-run）跑通，列出 4 stage
- [x] 4 个 bf16 build_cfg（固定 480p / chunk16 profile），shape 与 export 一致
- [x] `cosmos3_trt_only` runner 输出 action `[16, 10]`（+ 可选 vae_decode rollout）
- [x] `cosmos3_pt_trt_compare` 报告 PT vs TRT max/mean diff
- [x] Nano action 复用同一套 export 代码（profile + build_cfg 不同）

> 真实 export/build/infer 需 CUDA + TensorRT + diffusers cosmos3 权重；无环境时 e2e
> smoke 测试（`tests/e2e/workflows/test_cosmos3_trt_real.py`）自动 skip 真实 export，
> 仅校验 profile / build_cfg / plan。
