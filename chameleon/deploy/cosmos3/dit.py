"""cosmos3 MoT 联合 transformer 单步 forward ONNX 导出（去噪热点）。

dit 是去噪环里每步都跑的整模型 forward；TRT 部署需固定联合序列长度 profile
（见 deploy/cosmos3/shapes.py）。文本 embedding（text_embed）单独导出，每次推理算
一次而非每步重算。
"""

from __future__ import annotations

from pathlib import Path

from chameleon.deploy.cosmos3.onnx_export import export_stage_module


def export_dit(adapter, export_dir: str | Path, *, device: str = "cpu", **kwargs) -> Path:
    return export_stage_module(adapter, "dit", export_dir, "dit.onnx", device=device, **kwargs)


def export_text_embed(adapter, export_dir: str | Path, *, device: str = "cpu", **kwargs) -> Path:
    return export_stage_module(
        adapter, "text_embed", export_dir, "text_embed.onnx", device=device, **kwargs
    )
