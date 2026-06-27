"""cosmos3 Wan VAE encode/decode ONNX 导出。"""

from __future__ import annotations

from pathlib import Path

from chameleon.deploy.cosmos3.onnx_export import export_stage_module


def export_vae_encode(adapter, export_dir: str | Path, *, device: str = "cpu", **kwargs) -> Path:
    return export_stage_module(adapter, "vae_encode", export_dir, "vae_encode.onnx", device=device, **kwargs)


def export_vae_decode(adapter, export_dir: str | Path, *, device: str = "cpu", **kwargs) -> Path:
    return export_stage_module(adapter, "vae_decode", export_dir, "vae_decode.onnx", device=device, **kwargs)
