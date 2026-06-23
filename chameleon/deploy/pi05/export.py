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
from chameleon.deploy.pi05.memory import release_export_cuda_memory
from chameleon.deploy.pi05.shapes import PI05_LIBERO_PREFIX_LEN
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
    pi05_model,
    export_dir: str | Path,
    options: dict[str, Any] | None = None,
) -> Path:
    if stage == "embed_prefix":
        raise NotImplementedError(
            "embed_prefix export is not implemented yet in Chameleon deploy."
        )
    if stage not in _EXPORTERS:
        raise KeyError(f"Unknown pi05 export stage {stage!r}; expected one of {PI05_STAGES}.")

    options = dict(options or {})
    if stage in ("llm", "denoise", "expert"):
        options.setdefault("seq_len" if stage == "llm" else "prefix_len", PI05_LIBERO_PREFIX_LEN)
    exporter = _EXPORTERS[stage]
    try:
        return exporter(pi05_model, export_dir, **options)
    finally:
        release_export_cuda_memory(pi05_model)


def export_pi05_stages(
    stages: list[str],
    *,
    checkpoint_dir: str,
    export_dir: str | Path,
    train_config: str | None = None,
    stage_options: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Load pi05 once on CPU, export each stage, release GPU memory between stages."""
    pi05_model = load_pi05_model(checkpoint_dir, train_config, device="cpu")
    results: dict[str, Path] = {}
    try:
        for stage in stages:
            opts = dict((stage_options or {}).get(stage, {}))
            logger.info("Export pipeline stage=%s", stage)
            results[stage] = export_stage(
                stage,
                pi05_model=pi05_model,
                export_dir=export_dir,
                options=opts,
            )
    finally:
        del pi05_model
        release_export_cuda_memory()
    return results
