"""openpi Policy 会话构建 — evaluate 各 runner 共用单点入口。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chameleon.config.schema import TaskConfig
from chameleon.deploy.paths import resolve_checkpoint_dir
from chameleon.evaluate.norm_stats import load_norm_stats_for_eval
from chameleon.evaluate.task_utils import resolve_eval_device, resolve_openpi_config

logger = logging.getLogger(__name__)


@dataclass
class OpenPiSession:
    policy: Any
    action_horizon: int
    action_dim: int
    openpi_config: str
    checkpoint_dir: Path
    device: str | None
    default_prompt: str | None


def build_openpi_session(
    task: TaskConfig,
    *,
    pytorch_device: str | None = None,
) -> OpenPiSession:
    """从 TaskConfig 构建 openpi Policy 及模型维度元信息。

    Args:
        pytorch_device: 覆盖 ``evaluate.device``；TRT 路径默认取 ``evaluate.pytorch_load_device``。
    """
    try:
        from openpi.policies import policy_config
        from openpi.training import checkpoints as _checkpoints
        from openpi.training import config as _config
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "evaluate 需要可 import 的 openpi（openpi.policies.policy_config / "
            "openpi.training.*）。请在 openpi 环境下运行。"
        ) from exc

    openpi_config = resolve_openpi_config(task)
    checkpoint_dir = resolve_checkpoint_dir(task)
    device = pytorch_device if pytorch_device is not None else resolve_eval_device(task)
    train_cfg = _config.get_config(openpi_config)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    norm_stats = load_norm_stats_for_eval(
        checkpoint_dir=checkpoint_dir,
        norm_stats_dir=task.evaluate.norm_stats_dir,
        asset_id=task.evaluate.asset_id,
        data_config=data_config,
        checkpoints_mod=_checkpoints,
    )

    logger.info(
        "OpenPiSession: config=%s checkpoint_dir=%s device=%s",
        openpi_config,
        checkpoint_dir,
        device,
    )
    policy = policy_config.create_trained_policy(
        train_cfg,
        str(checkpoint_dir),
        norm_stats=norm_stats,
        default_prompt=task.evaluate.default_prompt,
        pytorch_device=device,
    )
    return OpenPiSession(
        policy=policy,
        action_horizon=int(train_cfg.model.action_horizon),
        action_dim=int(getattr(train_cfg.model, "action_dim", 0) or 0),
        openpi_config=openpi_config,
        checkpoint_dir=checkpoint_dir,
        device=device,
        default_prompt=task.evaluate.default_prompt,
    )
