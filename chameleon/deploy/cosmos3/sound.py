"""cosmos3 sound tokenizer 解码 ONNX 导出（可选，联合音频生成时启用）。

参考路径不含 sound 头；真实权重路径需 checkpoint 提供 sound_tokenizer。当前作为
占位，待真实权重接入时补全（对照 vae/dit 的导出模式）。
"""

from __future__ import annotations

from pathlib import Path


def export_sound(adapter, export_dir: str | Path, *, device: str = "cpu", **kwargs) -> Path:  # noqa: ARG001
    raise NotImplementedError(
        "cosmos3 sound_decode ONNX export is not implemented yet; "
        "enable only with a checkpoint exposing a sound tokenizer."
    )
