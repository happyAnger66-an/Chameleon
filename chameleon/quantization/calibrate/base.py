"""校准数据抽象 — PTQ 激活统计采集的统一接口。

作用：
    定义 Calibrator ABC（batches 迭代 + forward_loop 驱动模块前向）和
    TensorCalibrator（内存样本列表）。CALIBRATOR_REGISTRY 按
    (architecture, stage) 键注册工厂函数。

架构位置：
    优化/编译流水线 — 被 QuantMethod.quantize 调用，统一 model_optimizer
    原先分散的多套校准路径。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

import torch

from chameleon.core.registry import Registry


class Calibrator(ABC):
    """Provides calibration batches for a single stage."""

    @abstractmethod
    def batches(self) -> Iterator[Sequence[Any]]:
        """Yield positional-arg tuples suitable for ``module(*args)``."""

    def forward_loop(self, module: Any) -> None:
        """Default forward loop: run ``module`` over every calibration batch."""
        was_training = getattr(module, "training", False)
        if hasattr(module, "eval"):
            module.eval()
        try:
            with torch.no_grad():
                for batch in self.batches():
                    module(*batch)
        finally:
            if was_training and hasattr(module, "train"):
                module.train()


class TensorCalibrator(Calibrator):
    """Calibrator backed by an in-memory list of input tuples."""

    def __init__(self, samples: Sequence[Sequence[Any]]) -> None:
        self._samples = list(samples)

    def batches(self) -> Iterator[Sequence[Any]]:
        return iter(self._samples)


# Keyed by (architecture, stage); factories return a Calibrator given a module/adapter.
CALIBRATOR_REGISTRY: Registry[tuple[str, str], Any] = Registry("calibrator")


def register_calibrator(architecture: str, stage: str, factory: Any, *, override: bool = False):
    return CALIBRATOR_REGISTRY.register((architecture, stage), factory, override=override)


def get_calibrator_factory(architecture: str, stage: str):
    return CALIBRATOR_REGISTRY.get_or_none((architecture, stage))
