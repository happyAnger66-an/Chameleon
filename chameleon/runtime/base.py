"""运行时后端抽象 — 统一 Engine.run 接口的可插拔执行层。

作用：
    定义 Engine ABC（run(inputs) → outputs）和 RuntimeBackend ABC
    （load(artifact, ctx) → Engine）。RUNTIME_REGISTRY 键匹配
    PlatformSpec.runtime 或 TaskConfig.stage_runtimes 覆盖。

架构位置：
    运行时层 — 统一 model_optimizer 原先不一致的 TRT/Native/ORT API。
    被 orchestrator 按 stage 加载，支持 stage 级后端混用（如 vit=TRT,
    action_expert=PyTorch）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.core.registry import Registry


class Engine(ABC):
    """A loaded, runnable stage."""

    stage: str | None = None

    @abstractmethod
    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the stage. ``inputs`` order matters for positional backends."""


class RuntimeBackend(ABC):
    """Loads an artifact into an :class:`Engine`."""

    name: str

    def available(self) -> bool:
        return True

    @abstractmethod
    def load(self, artifact: Artifact, ctx: RunContext) -> Engine:
        ...


RUNTIME_REGISTRY: Registry[str, RuntimeBackend] = Registry("runtime")


def register_runtime(backend: RuntimeBackend, *, override: bool = False) -> RuntimeBackend:
    return RUNTIME_REGISTRY.register(backend.name, backend, override=override)


def get_runtime(name: str) -> RuntimeBackend:
    return RUNTIME_REGISTRY.get(name)


def list_runtimes() -> list[str]:
    return RUNTIME_REGISTRY.keys()
