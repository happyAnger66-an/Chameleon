"""Pi05 分阶段 ONNX 导出 registry。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chameleon.deploy.pi05.denoise import export_denoise
from chameleon.deploy.pi05.expert import export_expert
from chameleon.deploy.pi05.llm import export_llm
from chameleon.deploy.pi05.loader import load_pi05_model
from chameleon.deploy.pi05.vit import export_vit

logger = logging.getLogger(__name__)

PI05_STAGES = ("vit", "llm", "expert", "denoise", "embed_prefix")

_EXPORTERS: dict[str, Callable[..., Path]] = {
    "vit": export_vit,
    "llm": export_llm,
    "expert": export_expert,
    "denoise": export_denoise,
}


def export_stage(
    stage: str,
    *,
    checkpoint_dir: str,
    export_dir: str | Path,
    train_config: str | None = None,
    options: dict[str, Any] | None = None,
) -> Path:
    if stage == "embed_prefix":
        raise NotImplementedError(
            "embed_prefix export is not implemented yet in Chameleon deploy."
        )
    if stage not in _EXPORTERS:
        raise KeyError(f"Unknown pi05 export stage {stage!r}; expected one of {PI05_STAGES}.")

    options = dict(options or {})
    pi05_model = load_pi05_model(checkpoint_dir, train_config)
    exporter = _EXPORTERS[stage]
    return exporter(pi05_model, export_dir, **options)
