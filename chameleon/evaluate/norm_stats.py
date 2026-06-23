"""norm_stats 路径解析 — openpi session / transform 共用。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def resolve_norm_stats_assets_dir(
    *,
    checkpoint_dir: str | Path,
    norm_stats_dir: str | Path | None,
) -> Path | None:
    """解析 norm_stats 根目录（``{assets_dir}/{asset_id}/norm_stats.json``）。"""
    if norm_stats_dir is not None:
        path = Path(norm_stats_dir).expanduser()
        if path.is_dir():
            return path.resolve()
        logger.warning("norm_stats_dir 不是有效目录: %s", path)
    candidate = Path(checkpoint_dir).expanduser() / "assets"
    if candidate.is_dir():
        return candidate.resolve()
    return None


def load_norm_stats_for_eval(
    *,
    checkpoint_dir: str | Path,
    norm_stats_dir: str | Path | None,
    asset_id: str | None,
    data_config: Any,
    checkpoints_mod: Any,
) -> Any | None:
    """加载 norm_stats；失败时记录 warning 并返回 ``None``。"""
    aid = asset_id or getattr(data_config, "asset_id", None)
    assets_dir = resolve_norm_stats_assets_dir(
        checkpoint_dir=checkpoint_dir,
        norm_stats_dir=norm_stats_dir,
    )
    if assets_dir is None or aid is None:
        return None
    try:
        return checkpoints_mod.load_norm_stats(str(assets_dir), aid)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "加载 norm_stats 失败（assets_dir=%s asset_id=%s）：%s",
            assets_dir,
            aid,
            exc,
        )
        return None
