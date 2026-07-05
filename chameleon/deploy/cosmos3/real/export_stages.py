"""真实权重分阶段 ONNX 导出 — vae_encode / text_embed / dit / vae_decode。

对标 ``deploy/pi05/*.py``：每个 stage 用真实 diffusers 子模块 + 固定 profile 样例输入
调用 ``torch.onnx.export``。由 ``deploy/cosmos3/export.py`` 在 ``use_reference=false`` 时
选用（reference 路径仍走 ``onnx_export.export_stage_module``）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

from chameleon.deploy.cosmos3.real.dit_step import Cosmos3DitStepExport
from chameleon.deploy.cosmos3.real.onnx_utils import (
    force_cosmos3_export_attention,
    force_nearest_interpolate,
)
from chameleon.deploy.cosmos3.real.pack import build_policy_pack
from chameleon.deploy.cosmos3.real.vae import WanVaeDecodeExport, WanVaeEncodeExport
from chameleon.deploy.cosmos3.shapes import Cosmos3Profile, get_profile

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "policy_droid"


def _resolve_dtype(adapter) -> torch.dtype:
    return torch.bfloat16 if adapter.config.precision == "bfloat16" else torch.float32


def _resolve_profile(options: dict[str, Any]) -> Cosmos3Profile:
    return get_profile(str(options.get("profile", DEFAULT_PROFILE)))


def _require_pipeline(adapter):
    if not getattr(adapter, "_is_real_diffusers", False) or adapter.pipeline is None:
        raise RuntimeError(
            "cosmos3 real export requires a loaded Cosmos3OmniPipeline "
            "(set model_overrides.use_reference=false and provide a valid model_id/checkpoint)."
        )
    return adapter.pipeline


def _onnx_export(
    module: torch.nn.Module,
    args: tuple,
    out_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None,
    opset_version: int = 19,
    dynamo: bool = False,
    do_constant_folding: bool = True,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    logger.info("Exporting cosmos3 real -> %s", out_path)
    with torch.inference_mode(), force_cosmos3_export_attention(), force_nearest_interpolate():
        torch.onnx.export(
            module,
            args,
            str(out_path),
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=dynamo,
            do_constant_folding=do_constant_folding,
            dynamic_axes=dynamic_axes,
        )
    logger.info("cosmos3 real export done in %.1fs (%s)", time.time() - start, out_path.name)
    if not out_path.is_file():
        raise FileNotFoundError(f"ONNX export finished but file missing: {out_path}")
    return out_path


def export_vae_encode(adapter, export_dir: str | Path, *, device: str = "cuda", **options) -> Path:
    pipe = _require_pipeline(adapter)
    profile = _resolve_profile(options)
    dtype = _resolve_dtype(adapter)
    module = WanVaeEncodeExport(pipe.vae).to(device).eval()
    video = torch.randn(
        1, 3, profile.num_frames, profile.canvas_h, profile.canvas_w, device=device, dtype=dtype
    )
    return _onnx_export(
        module,
        (video,),
        Path(export_dir) / "vae_encode.onnx",
        input_names=["video"],
        output_names=["vision_latent"],
        dynamic_axes={"video": {0: "batch"}, "vision_latent": {0: "batch"}},
        dynamo=bool(options.get("dynamo", False)),
    )


def export_vae_decode(adapter, export_dir: str | Path, *, device: str = "cuda", **options) -> Path:
    pipe = _require_pipeline(adapter)
    profile = _resolve_profile(options)
    dtype = _resolve_dtype(adapter)
    module = WanVaeDecodeExport(pipe.vae).to(device).eval()
    latent_channels = int(getattr(pipe.vae.config, "z_dim", profile.latent_channels))
    latent = torch.randn(
        1, latent_channels, profile.latent_t, profile.latent_h, profile.latent_w,
        device=device, dtype=dtype,
    )
    return _onnx_export(
        module,
        (latent,),
        Path(export_dir) / "vae_decode.onnx",
        input_names=["latent"],
        output_names=["video"],
        dynamic_axes={"latent": {0: "batch"}, "video": {0: "batch"}},
        dynamo=bool(options.get("dynamo", False)),
    )


def export_text_embed(adapter, export_dir: str | Path, *, device: str = "cuda", **options) -> Path:
    pipe = _require_pipeline(adapter)
    profile = _resolve_profile(options)
    module = pipe.transformer.embed_tokens.to(device).eval()
    input_ids = torch.zeros(profile.text_prefix_len, dtype=torch.long, device=device)
    return _onnx_export(
        module,
        (input_ids,),
        Path(export_dir) / "text_embed.onnx",
        input_names=["input_ids"],
        output_names=["text_emb"],
        dynamic_axes={"input_ids": {0: "und_len"}, "text_emb": {0: "und_len"}},
        dynamo=bool(options.get("dynamo", False)),
    )


def export_dit(adapter, export_dir: str | Path, *, device: str = "cuda", **options) -> Path:
    pipe = _require_pipeline(adapter)
    profile = _resolve_profile(options)
    dtype = _resolve_dtype(adapter)
    pack = build_policy_pack(pipe, profile, device=device, dtype=dtype)
    module = Cosmos3DitStepExport(pipe.transformer, pack.static).to(device).eval()
    ex = pack.dynamic_example
    args = (
        ex["vision_tokens"],
        ex["vision_timesteps"],
        ex["action_tokens"],
        ex["action_timesteps"],
    )
    # Fixed policy profile → no dynamic axes (seq_len / patch counts are locked).
    # MoT 的 packing/patchify 辅助里有若干在 CPU 上构造的 index/arange 常量（eager 运行
    # 时按需搬到 CUDA，但 ONNX 的 _jit_pass_onnx_constant_fold 会因 cuda/cpu 常量混用而
    # 报 device mismatch）。关掉 ONNX 侧常量折叠即可绕开该 pass；TRT build 时仍会做常量
    # 折叠，无运行期损失。可用 options.do_constant_folding=true 覆盖。
    return _onnx_export(
        module,
        args,
        Path(export_dir) / "dit.onnx",
        input_names=["vision_tokens", "vision_timesteps", "action_tokens", "action_timesteps"],
        output_names=["v_vision", "v_action"],
        dynamic_axes=None,
        dynamo=bool(options.get("dynamo", False)),
        do_constant_folding=bool(options.get("do_constant_folding", False)),
    )


REAL_EXPORTERS = {
    "vae_encode": export_vae_encode,
    "text_embed": export_text_embed,
    "dit": export_dit,
    "vae_decode": export_vae_decode,
}
