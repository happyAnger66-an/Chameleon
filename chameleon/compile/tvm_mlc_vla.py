"""Chamleon ↔ mlc-vla 编译接缝（第一阶段：仅 denoise stage）。

作用：
    把 π0.5 的 ``denoise`` stage 交给独立的 [mlc-vla](../../mlc-vla) 包，用 TVM Relax
    的 nn 前端 + 编译 pipeline 产出 engine（.so + 打包权重）。权重复用 Chamleon 现有的
    openpi safetensors；可选做与 openpi "联合前向" 参考的 cosine 对拍（≥0.99）。

设计取舍（保持薄）：
    - 本阶段只闭合 denoise stage 的**编译 + 数值正确性**；vit / llm / 去噪环留后续。
    - mlc-vla / tvm 通过 sys.path 注入，默认取 edgeLLM 同级仓库；可用环境变量覆盖：
      ``MLC_VLA_HOME`` / ``TVM_HOME`` / ``TVM_LIBRARY_PATH``。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _edgellm_root() -> Path:
    # Chamleon/chameleon/compile/tvm_mlc_vla.py -> edgeLLM/
    return Path(__file__).resolve().parents[3]


def _ensure_paths() -> None:
    """把 tvm/python 与 mlc-vla/python 注入 sys.path（若尚未可用）。"""
    root = _edgellm_root()
    tvm_home = os.environ.get("TVM_HOME", str(root / "tvm"))
    mlc_home = os.environ.get("MLC_VLA_HOME", str(root / "mlc-vla"))
    for p in (str(Path(tvm_home) / "python"), str(Path(mlc_home) / "python")):
        if p not in sys.path and Path(p).is_dir():
            sys.path.insert(0, p)
    # TVM 运行库位置（若未显式配置）
    os.environ.setdefault("TVM_LIBRARY_PATH", str(Path(tvm_home) / "build" / "lib"))
    os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")


def available() -> bool:
    try:
        _ensure_paths()
        import tvm  # noqa: F401
        import mlc_vla  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# platform.device -> TVM target
_DEVICE_TO_TARGET = {"cuda": "cuda", "rocm": "rocm", "cpu": "llvm"}
# 各 target 的默认 dtype（cuda/rocm 用 bf16 避免 Gemma fp16 溢出；cpu 用 fp32）
_TARGET_DEFAULT_DTYPE = {"cuda": "bfloat16", "rocm": "bfloat16", "llvm": "float32"}


def resolve_target(platform_device: str, cfg: dict[str, Any] | None) -> str:
    if cfg and cfg.get("tvm_target"):
        return str(cfg["tvm_target"])
    return _DEVICE_TO_TARGET.get(platform_device, "llvm")


def compile_denoise(
    checkpoint_dir: str,
    output_dir: str,
    *,
    target: str = "llvm",
    dtype: str | None = None,
    verify: bool = False,
    threshold: float = 0.99,
) -> dict[str, Any]:
    """用 mlc-vla 编译 π0.5 ``denoise_step``，产出 engine + 打包权重。

    返回 metadata dict（engine 路径、参数路径、shape、可选 cosine）。
    """
    _ensure_paths()
    import numpy as np

    from mlc_vla.compile import compile_model
    from mlc_vla.model.pi0 import Pi0Config, Pi0Model
    from mlc_vla.model.pi0 import pi0_loader

    dtype = dtype or _TARGET_DEFAULT_DTYPE.get(target, "float32")
    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    ckpt_file = ckpt_dir / "model.safetensors"
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    config = Pi0Config.from_openpi_config(str(ckpt_dir), dtype=dtype)

    # 权重：Chamleon 现有 safetensors -> mlc-vla 参数名（带 shape 断言）
    model = Pi0Model(config)
    model.to(config.dtype)
    _, named_params, _ = model.export_tvm(
        spec=model.get_default_spec(functions=["denoise_step"]), allow_extern=True
    )
    src = pi0_loader.load_safetensors(str(ckpt_file), dtype="float32")
    params = pi0_loader.load_params(config, src, named_params=named_params, dtype=dtype)
    del src

    # 编译 denoise_step（仅此 stage，避开 SigLIP layer_norm 的 bf16 限制）
    ex, named_params = compile_model(config, target, functions=["denoise_step"])

    engine_path = out / "denoise_tvm.so"
    ex.export_library(str(engine_path))
    param_path = out / "denoise_tvm_params.npz"
    np.savez(str(param_path), **{name: params[name] for name, _ in named_params})

    meta: dict[str, Any] = {
        "backend": "tvm",
        "stage": "denoise",
        "target": target,
        "dtype": dtype,
        "engine": str(engine_path),
        "params": str(param_path),
        "prefix_len": config.prefix_len,
        "action_horizon": config.action_horizon,
        "action_dim": config.action_dim,
        "num_params": len(named_params),
    }

    if verify:
        from mlc_vla.compare import run_mode_b

        ok, cos = run_mode_b(
            config, target, threshold, seed=0, ckpt=str(ckpt_file),
            dtype=dtype, ref_dtype="bfloat16" if target != "llvm" else "float32",
            prefer_real=True,
        )
        meta["parity_cosine"] = cos
        meta["parity_pass"] = bool(ok)

    return meta


def compile_denoise_kv(
    checkpoint_dir: str,
    output_dir: str,
    *,
    target: str = "cuda",
    dtype: str | None = None,
    cuda_graph: bool = True,
    quant: str | None = None,
    verify: bool = False,
    threshold: float = 0.99,
) -> dict[str, Any]:
    """M1：编译 π0.5 的 ``prefill`` + ``denoise_step_kv`` 为单个 engine（含共享权重）。

    生产推理姿势：expert-0 对 prefix 只跑一次固化逐层 K/V，之后每个去噪步仅让 expert-1
    对 action token 前向、attend 缓存的 prefix K/V。宿主侧用 ``mlc_vla.sample.euler_loop``
    编排 10 步去噪。相较 M0（每步重算联合注意力）单步约 19x、端到端约 6.8x 提速。

    ``quant``（如 ``"q4bf16_1"``）：对 nn.Linear 做 group int4 量化（M1+）。约 2.5x 显存下降；
    注意本工况（10 token 计算轻）dequant 开销可能使单步略慢，收益主要在显存。

    返回 metadata dict（engine 路径、参数路径、shape、可选 cosine）。
    """
    _ensure_paths()
    import numpy as np

    from mlc_vla.model.pi0 import Pi0Config

    funcs = ["prefill", "denoise_step_kv"]
    dtype = dtype or _TARGET_DEFAULT_DTYPE.get(target, "float32")
    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    ckpt_file = ckpt_dir / "model.safetensors"
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if quant:
        # 量化预设的 model_dtype 决定 config.dtype（激活/KV 与反量化计算 dtype 对齐）
        from mlc_vla.quant import get_quant
        dtype = get_quant(quant).model_dtype

    config = Pi0Config.from_openpi_config(str(ckpt_dir), dtype=dtype)
    use_cg = cuda_graph and target == "cuda"
    vlm = config.vlm

    if quant:
        engine_path, param_path, num_params, q_bytes = _compile_quant(
            config, target, funcs, quant, use_cg, str(ckpt_file), out, np)
    else:
        engine_path, param_path, num_params, q_bytes = _compile_fp(
            config, target, funcs, dtype, use_cg, str(ckpt_file), out, np)

    meta: dict[str, Any] = {
        "backend": "tvm",
        "stage": "denoise",
        "mode": "M1",
        "quant": quant or "none",
        "functions": funcs,
        "target": target,
        "dtype": dtype,
        "cuda_graph": bool(use_cg),
        "engine": str(engine_path),
        "params": str(param_path),
        "param_bytes": int(q_bytes),
        "prefix_len": config.prefix_len,
        "action_horizon": config.action_horizon,
        "action_dim": config.action_dim,
        # denoise_step_kv 的 KV cache 输入 shape：[depth,1,kv,prefix_len,head_dim]
        "kv_shape": [vlm.depth, 1, vlm.num_kv_heads, config.prefix_len, vlm.head_dim],
        "num_params": num_params,
    }

    if verify:
        if quant:
            from mlc_vla.compare_quant import run as run_quant_parity

            ok = run_quant_parity(config, target, quant, seed=0, ckpt=str(ckpt_file))
            meta["parity_kind"] = "quant_vs_fp"
            meta["parity_pass"] = bool(ok)
        else:
            from mlc_vla.compare import run_mode_b

            ok, cos = run_mode_b(
                config, target, threshold, seed=0, ckpt=str(ckpt_file),
                dtype=dtype, ref_dtype="bfloat16" if target != "llvm" else "float32",
                prefer_real=True, use_kv=True,
            )
            meta["parity_kind"] = "tvm_vs_openpi_ref"
            meta["parity_cosine"] = cos
            meta["parity_pass"] = bool(ok)

    return meta


def _compile_fp(config, target, funcs, dtype, use_cg, ckpt_file, out, np):
    from mlc_vla.compile import compile_model
    from mlc_vla.model.pi0 import Pi0Model, pi0_loader

    model = Pi0Model(config)
    model.to(config.dtype)
    _, named_params, _ = model.export_tvm(
        spec=model.get_default_spec(functions=funcs), allow_extern=True)
    src = pi0_loader.load_safetensors(ckpt_file, dtype="float32")
    params = pi0_loader.load_params(config, src, named_params=named_params, dtype=dtype)
    del src

    ex, named_params = compile_model(config, target, functions=funcs, cuda_graph=use_cg)
    engine_path = out / "denoise_kv_tvm.so"
    ex.export_library(str(engine_path))
    param_path = out / "denoise_kv_tvm_params.npz"
    arrs = {name: params[name] for name, _ in named_params}
    np.savez(str(param_path), **arrs)
    return engine_path, param_path, len(named_params), sum(a.nbytes for a in arrs.values())


def _compile_quant(config, target, funcs, quant_name, use_cg, ckpt_file, out, np):
    from mlc_vla.compile_quant import compile_model_quant, fp_named_params, quantize_params
    from mlc_vla.model.pi0 import pi0_loader

    # fp 权重（float32）-> 量化 params
    fp_np = fp_named_params(config, funcs)
    src = pi0_loader.load_safetensors(ckpt_file, dtype="float32")
    src_fp = pi0_loader.load_params(config, src, named_params=fp_np, dtype="float32")
    del src

    ex, q_named_params, qmap, quant = compile_model_quant(
        config, target, funcs, quant_name, cuda_graph=use_cg)
    q_params = quantize_params(quant, src_fp, q_named_params, qmap)

    engine_path = out / f"denoise_kv_tvm_{quant_name}.so"
    ex.export_library(str(engine_path))
    param_path = out / f"denoise_kv_tvm_{quant_name}_params.npz"
    arrs = {name: q_params[name] for name, _ in q_named_params}
    np.savez(str(param_path), **arrs)
    return engine_path, param_path, len(q_named_params), sum(a.nbytes for a in arrs.values())
