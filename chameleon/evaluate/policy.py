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
from typing import Any

import numpy as np

from chameleon.config.schema import TaskConfig
from chameleon.evaluate.openpi_session import OpenPiSession, build_openpi_session
from chameleon.evaluate.runner_base import PolicyRunner, register_policy_runner

logger = logging.getLogger(__name__)


class OpenPiPolicyRunner(PolicyRunner):
    """按 openpi config + checkpoint 构建真实 pi05 策略并逐帧推理。"""

    def __init__(
        self,
        session: OpenPiSession | None = None,
        *,
        task: TaskConfig | None = None,
    ) -> None:
        if session is None and task is None:
            raise ValueError("OpenPiPolicyRunner 需要 session 或 task。")
        self._session = session
        self._task = task
        self._built = session is not None

    @classmethod
    def from_task(cls, task: TaskConfig) -> "OpenPiPolicyRunner":
        return cls(task=task)

    def build(self) -> "OpenPiPolicyRunner":
        if self._built:
            return self
        if self._session is None:
            assert self._task is not None
            self._session = build_openpi_session(self._task)
        self._built = True
        return self

    @property
    def openpi_config(self) -> str:
        self.build()
        assert self._session is not None
        return self._session.openpi_config

    @property
    def checkpoint_dir(self) -> str:
        self.build()
        assert self._session is not None
        return str(self._session.checkpoint_dir)

    @property
    def device(self) -> str | None:
        self.build()
        assert self._session is not None
        return self._session.device

    @property
    def default_prompt(self) -> str | None:
        self.build()
        assert self._session is not None
        return self._session.default_prompt

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> np.ndarray:
        self.build()
        assert self._session is not None
        out = self._session.policy.infer(dict(observation), noise=noise)
        return np.asarray(out["actions"])

    @property
    def action_horizon(self) -> int:
        self.build()
        assert self._session is not None
        return self._session.action_horizon

    @property
    def action_dim(self) -> int:
        self.build()
        assert self._session is not None
        return self._session.action_dim

    @property
    def metadata(self) -> dict[str, Any]:
        self.build()
        assert self._session is not None
        base = dict(getattr(self._session.policy, "metadata", {}) or {})
        base["backend"] = "openpi"
        base["openpi_config"] = self._session.openpi_config
        base["checkpoint_dir"] = str(self._session.checkpoint_dir)
        return base


register_policy_runner("openpi", OpenPiPolicyRunner, override=True)
