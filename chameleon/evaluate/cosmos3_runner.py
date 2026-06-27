"""cosmos3 策略运行器 — evaluate 路径上的 action / video 推理。

action 模式（对齐 pi05 VLA）：经 Cosmos3 InferenceSession 生成 action chunk，与
ground-truth 比对；video 模式做 smoke（返回隐变量/视频张量的展平统计，验证 shape
与可生成性）。不依赖 openpi I/O transform，直接走 Cosmos3Adapter 的观测接口。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from chameleon.api import _run_context, build_adapter
from chameleon.config.schema import TaskConfig
from chameleon.evaluate.runner_base import PolicyRunner, register_policy_runner
from chameleon.runtime.orchestrator import InferenceSession

logger = logging.getLogger(__name__)


class Cosmos3PolicyRunner(PolicyRunner):
    """Reference / real Cosmos3 推理运行器（action 评测 + video smoke）。"""

    def __init__(self, task: TaskConfig, *, device: str) -> None:
        self._task = task
        self._device = device
        self._built = False
        self._adapter: Any = None
        self._session: InferenceSession | None = None
        self._mode = task.generate.mode

    @classmethod
    def from_task(cls, task: TaskConfig) -> "Cosmos3PolicyRunner":
        device = task.evaluate.device or task.infer.torch_device or "cpu"
        return cls(task, device=device)

    def build(self) -> "Cosmos3PolicyRunner":
        if self._built:
            return self
        ctx = _run_context(self._task)
        ctx.options["torch_device"] = self._device
        self._adapter = build_adapter(self._task, device=self._device)
        self._session = InferenceSession(
            self._adapter, ctx, stage_runtimes=self._task.stage_runtimes or None
        ).build()
        self._built = True
        logger.info("Cosmos3PolicyRunner ready: mode=%s device=%s", self._mode, self._device)
        return self

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> np.ndarray:
        if not self._built:
            self.build()
        assert self._session is not None
        obs = self._adapter.example_observation(1, device=self._device)
        obs["mode"] = self._mode
        out = self._session.infer(obs)
        arr = out.detach().to("cpu").float().numpy()
        if self._mode == "action":
            # [B, H, D] -> [H, D]
            return arr[0] if arr.ndim == 3 else arr
        # video smoke: collapse to [1, N] feature vector for the eval loop.
        return arr.reshape(1, -1)

    @property
    def action_horizon(self) -> int:
        return int(self._adapter.config.action_horizon) if self._adapter else int(
            self._task.model_overrides.get("action_horizon", 16)
        )

    @property
    def action_dim(self) -> int:
        return int(self._adapter.config.action_dim) if self._adapter else int(
            self._task.model_overrides.get("action_dim", 32)
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "cosmos3",
            "mode": self._mode,
            "device": self._device,
            "reference": getattr(getattr(self._adapter, "config", None), "use_reference", True),
        }


register_policy_runner("cosmos3", Cosmos3PolicyRunner, override=True)
