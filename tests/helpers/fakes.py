"""evaluate e2e 用的轻量 fake 实现（无 openpi / GPU 依赖）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from chameleon.dataloader.base import ChameleonSample
from chameleon.evaluate.lerobot_eval import EvalSummary
from chameleon.evaluate.runner_base import PolicyRunner
from chameleon.evaluate.viewers.base import EvalEventSink, EvalStepEvent


class FakeDataSource:
    """内存 LeRobot 替身，支持 chunk_eligible 与 evaluate 窗口语义。"""

    def __init__(
        self,
        *,
        length: int = 25,
        action_horizon: int = 10,
        action_dim: int = 7,
        start_index: int = 0,
        frame_count: int = 100,
    ) -> None:
        self._length = length
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._start = start_index
        self._frame_count = frame_count
        self._episode_ids = np.zeros(frame_count, dtype=np.int64)

    def build(self) -> FakeDataSource:
        return self

    def __len__(self) -> int:
        return self._length

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def start_index(self) -> int:
        return self._start

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def eval_end_exclusive(self) -> int:
        return self._start + self._length

    @property
    def episode_ids_per_frame(self) -> np.ndarray:
        return self._episode_ids

    @property
    def repo_id(self) -> str:
        return "test/repo"

    def __getitem__(self, index: int) -> ChameleonSample:
        global_index = self._start + index
        gt = np.linspace(0.0, 1.0, self._action_horizon * self._action_dim, dtype=np.float32)
        gt = gt.reshape(self._action_horizon, self._action_dim)
        obs = {
            "image": np.zeros((3, 8, 8), dtype=np.float32),
            "state": np.zeros(self._action_dim, dtype=np.float32),
        }
        return ChameleonSample(
            observation=obs,
            actions_gt=gt,
            prompt="pick up",
            index=global_index,
            episode_id=0,
        )


class FakePolicyRunner(PolicyRunner):
    """恒定预测，便于断言 compare 与 step 事件。"""

    def __init__(self, *, action_horizon: int = 10, action_dim: int = 7) -> None:
        self._ah = action_horizon
        self._ad = action_dim
        self._built = False

    @classmethod
    def from_task(cls, task) -> FakePolicyRunner:  # noqa: ANN001
        ah = int(task.model_overrides.get("action_horizon", 10))
        ad = int(task.model_overrides.get("action_dim", 7))
        return cls(action_horizon=ah, action_dim=ad)

    def build(self) -> FakePolicyRunner:
        self._built = True
        return self

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> np.ndarray:
        del observation, noise
        return np.full((self._ah, self._ad), 0.5, dtype=np.float32)

    @property
    def action_horizon(self) -> int:
        return self._ah

    @property
    def action_dim(self) -> int:
        return self._ad

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": "fake", "action_horizon": self._ah, "action_dim": self._ad}


class CaptureEventSink(EvalEventSink):
    def __init__(self) -> None:
        self.meta: dict[str, Any] | None = None
        self.steps: list[EvalStepEvent] = []
        self.summary: EvalSummary | None = None

    def on_run_start(self, meta: dict[str, Any]) -> None:
        self.meta = meta

    def on_step(self, event: EvalStepEvent) -> None:
        self.steps.append(event)

    def on_run_done(self, summary: EvalSummary) -> None:
        self.summary = summary
