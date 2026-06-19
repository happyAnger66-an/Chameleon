"""openpi 策略运行器 — 封装真实 pi05 推理（含完整 transform 管线）。

作用：
    OpenPiPolicyRunner 复用 openpi ``policy_config.create_trained_policy``：
    自动完成 repack → 注入 prompt → 数据 transform → Normalize → tokenize →
    sample_actions → Unnormalize → 输出 transform，返回物理动作。

架构位置：
    工具层（evaluate）— 注册为 policy_runner ``openpi``；上游 evaluate_lerobot /
    api.run_eval；下游 openpi Policy。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from chameleon.config.schema import TaskConfig
from chameleon.evaluate.task_utils import (
    resolve_checkpoint_dir,
    resolve_eval_device,
    resolve_openpi_config,
)
from chameleon.evaluate.runner_base import PolicyRunner, register_policy_runner

logger = logging.getLogger(__name__)


class OpenPiPolicyRunner(PolicyRunner):
    """按 openpi config + checkpoint 构建真实 pi05 策略并逐帧推理。"""

    def __init__(
        self,
        *,
        openpi_config: str,
        checkpoint_dir: str | Path,
        norm_stats_dir: str | Path | None = None,
        asset_id: str | None = None,
        device: str | None = None,
        default_prompt: str | None = None,
    ) -> None:
        self.openpi_config = openpi_config
        self.checkpoint_dir = str(checkpoint_dir)
        self.norm_stats_dir = str(norm_stats_dir) if norm_stats_dir is not None else None
        self.asset_id = asset_id
        self.device = device
        self.default_prompt = default_prompt

        self._built = False
        self._policy: Any = None
        self._action_horizon: int = 0
        self._action_dim: int = 0

    @classmethod
    def from_task(cls, task: TaskConfig) -> "OpenPiPolicyRunner":
        return cls(
            openpi_config=resolve_openpi_config(task),
            checkpoint_dir=resolve_checkpoint_dir(task),
            norm_stats_dir=task.evaluate.norm_stats_dir,
            asset_id=task.evaluate.asset_id,
            device=resolve_eval_device(task),
            default_prompt=task.evaluate.default_prompt,
        )

    def build(self) -> "OpenPiPolicyRunner":
        if self._built:
            return self

        try:
            from openpi.policies import policy_config
            from openpi.training import checkpoints as _checkpoints
            from openpi.training import config as _config
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "evaluate 需要可 import 的 openpi（openpi.policies.policy_config / "
                "openpi.training.*）。请在 openpi 环境下运行。"
            ) from exc

        train_cfg = _config.get_config(self.openpi_config)
        data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
        norm_stats = self._maybe_load_norm_stats(data_config, _checkpoints)

        logger.info(
            "OpenPiPolicyRunner: config=%s checkpoint_dir=%s device=%s",
            self.openpi_config,
            self.checkpoint_dir,
            self.device,
        )
        self._policy = policy_config.create_trained_policy(
            train_cfg,
            self.checkpoint_dir,
            norm_stats=norm_stats,
            default_prompt=self.default_prompt,
            pytorch_device=self.device,
        )
        self._action_horizon = int(train_cfg.model.action_horizon)
        self._action_dim = int(getattr(train_cfg.model, "action_dim", 0) or 0)
        self._built = True
        return self

    def _maybe_load_norm_stats(self, data_config: Any, checkpoints_mod: Any) -> Any:
        asset_id = self.asset_id or getattr(data_config, "asset_id", None)
        assets_dir = self.norm_stats_dir
        if assets_dir is None:
            candidate = Path(self.checkpoint_dir) / "assets"
            if candidate.is_dir():
                assets_dir = str(candidate)
        if assets_dir is None or asset_id is None:
            return None
        try:
            return checkpoints_mod.load_norm_stats(assets_dir, asset_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "加载 norm_stats 失败（assets_dir=%s asset_id=%s）：%s",
                assets_dir,
                asset_id,
                exc,
            )
            return None

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> np.ndarray:
        if not self._built:
            self.build()
        out = self._policy.infer(dict(observation), noise=noise)
        return np.asarray(out["actions"])

    @property
    def action_horizon(self) -> int:
        if not self._built:
            self.build()
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        if not self._built:
            self.build()
        return self._action_dim

    @property
    def metadata(self) -> dict[str, Any]:
        if not self._built:
            self.build()
        base = dict(getattr(self._policy, "metadata", {}) or {})
        base["backend"] = "openpi"
        base["openpi_config"] = self.openpi_config
        base["checkpoint_dir"] = self.checkpoint_dir
        return base


register_policy_runner("openpi", OpenPiPolicyRunner, override=True)
