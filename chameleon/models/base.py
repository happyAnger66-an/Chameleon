"""ModelAdapter 抽象 — 外部模型到 Chameleon stage 世界的桥梁。

作用：
    定义 ModelAdapter 接口：build() 构建模型、stage_module() 暴露各 stage
    的 nn.Module、example_observation() 提供追踪/冒烟测试输入。维护
    MODEL_REGISTRY 供按名称查找适配器。

架构位置：
    模型/架构层 — 被 api.py、workflows、runtime/orchestrator 使用。
    上游对接 architectures（architecture 字段），下游对接 frontend /
    quantization / runtime 各 stage 模块。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from chameleon.core.registry import Registry


class ModelAdapter(ABC):
    """Adapts a concrete model into the Chameleon stage interface."""

    architecture: str

    def __init__(self, config: Any) -> None:
        self.config = config

    @classmethod
    def make_config(cls, overrides: dict[str, Any] | None = None) -> Any:
        """Build an adapter-specific config from a plain overrides dict.

        Subclasses with a structured config (e.g. a dataclass) should override
        this; the default simply returns the overrides dict.
        """
        return dict(overrides or {})

    @abstractmethod
    def build(self, device: str = "cpu") -> "ModelAdapter":
        """Construct (or load) the underlying model onto ``device``."""

    @abstractmethod
    def stage_module(self, stage: str):
        """Return the ``nn.Module`` implementing ``stage``."""

    @abstractmethod
    def example_observation(self, batch_size: int = 1, device: str = "cpu") -> dict[str, Any]:
        """Return a representative observation dict used for tracing / smoke tests."""

    # --- metadata used by the orchestrator -------------------------------
    @property
    def action_dim(self) -> int:
        return int(getattr(self.config, "action_dim"))

    @property
    def action_horizon(self) -> int:
        return int(getattr(self.config, "action_horizon"))

    @property
    def num_denoise_steps(self) -> int:
        return int(getattr(self.config, "num_denoise_steps"))


MODEL_REGISTRY: Registry[str, type[ModelAdapter]] = Registry("model")


def register_model(name: str, adapter_cls: type[ModelAdapter], *, override: bool = False):
    return MODEL_REGISTRY.register(name, adapter_cls, override=override)


def get_model_adapter(name: str) -> type[ModelAdapter]:
    return MODEL_REGISTRY.get(name)


def list_models() -> list[str]:
    return MODEL_REGISTRY.keys()
