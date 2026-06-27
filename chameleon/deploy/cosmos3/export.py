"""cosmos3 分阶段 ONNX 导出 registry。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chameleon.deploy.cosmos3.dit import export_dit, export_text_embed
from chameleon.deploy.cosmos3.sound import export_sound
from chameleon.deploy.cosmos3.vae import export_vae_decode, export_vae_encode

logger = logging.getLogger(__name__)

COSMOS3_STAGES = ("vae_encode", "text_embed", "dit", "vae_decode", "sound_decode")

_EXPORTERS: dict[str, Callable[..., Path]] = {
    "vae_encode": export_vae_encode,
    "text_embed": export_text_embed,
    "dit": export_dit,
    "vae_decode": export_vae_decode,
    "sound_decode": export_sound,
}


def export_stage(
    stage: str,
    *,
    adapter,
    export_dir: str | Path,
    device: str = "cpu",
    options: dict[str, Any] | None = None,
) -> Path:
    if stage not in _EXPORTERS:
        raise KeyError(f"Unknown cosmos3 export stage {stage!r}; expected one of {COSMOS3_STAGES}.")
    exporter = _EXPORTERS[stage]
    return exporter(adapter, export_dir, device=device, **(options or {}))


def export_cosmos3_stages(
    stages: list[str],
    *,
    adapter,
    export_dir: str | Path,
    device: str = "cpu",
    stage_options: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Export each cosmos3 stage from a built adapter."""
    results: dict[str, Path] = {}
    for stage in stages:
        opts = dict((stage_options or {}).get(stage, {}))
        logger.info("Export pipeline cosmos3 stage=%s", stage)
        results[stage] = export_stage(
            stage, adapter=adapter, export_dir=export_dir, device=device, options=opts
        )
    return results
