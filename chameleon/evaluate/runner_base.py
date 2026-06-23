"""PolicyRunner 抽象 — evaluate 路径上的模型无关推理接口。

作用：
    定义 PolicyRunner ABC（infer / metadata / build）与 POLICY_RUNNER_REGISTRY。
    evaluate 循环只依赖本接口，具体实现可以是 openpi Policy（OpenPiPolicyRunner）
    或 Chameleon InferenceSession（ChameleonOrchestratorRunner），便于后续接
    TRT engine、其它 VLA 后端。

架构位置：
    工具层（evaluate）— 上游：api.run_eval / evaluate_lerobot；下游：各 Runner 实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np

from chameleon.config.schema import TaskConfig
from chameleon.core.registry import Registry


@runtime_checkable
class SupportsDualInfer(Protocol):
    """双路推理：PyTorch + TRT 并行输出。"""

    def infer_dual(
        self,
        observation: dict[str, Any],
        *,
        sample_index: int = 0,
        noise: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class SupportsFixedNoise(Protocol):
    """固定 flow-matching 噪声（``noise=fixed`` 评测）。"""

    def noise_for_sample(self, sample_index: int) -> np.ndarray | None: ...


class PolicyRunner(ABC):
    """evaluate 使用的策略运行器统一接口。"""

    @classmethod
    @abstractmethod
    def from_task(cls, task: TaskConfig) -> "PolicyRunner":
        """从 TaskConfig 构建 runner（各实现解析 checkpoint / device 等）。"""

    @abstractmethod
    def build(self) -> "PolicyRunner":
        """加载模型 / 会话（懒加载入口）。"""

    @abstractmethod
    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> np.ndarray:
        """对单帧 repack 后的 observation 推理，返回物理动作 ``[H, D_env]``。"""

    @property
    @abstractmethod
    def action_horizon(self) -> int:
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """至少应含 ``backend``；可含 action_horizon / checkpoint 等。"""


POLICY_RUNNER_REGISTRY: Registry[str, type[PolicyRunner]] = Registry("policy_runner")


def register_policy_runner(name: str, cls: type[PolicyRunner], *, override: bool = False):
    return POLICY_RUNNER_REGISTRY.register(name, cls, override=override)


def build_policy_runner(task: TaskConfig) -> PolicyRunner:
    """按 ``task.evaluate.policy_runner`` 构建策略运行器（默认 ``openpi``）。"""
    name = getattr(task.evaluate, "policy_runner", None) or "openpi"
    runner_cls = POLICY_RUNNER_REGISTRY.get(name)
    return runner_cls.from_task(task)


def list_policy_runners() -> list[str]:
    return POLICY_RUNNER_REGISTRY.keys()
