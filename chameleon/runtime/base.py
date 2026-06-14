"""Runtime backend abstraction.

A :class:`RuntimeBackend` loads a (compiled or reference) stage :class:`Artifact`
into an :class:`Engine`. Every engine exposes the same ``run(inputs) -> outputs``
contract regardless of platform -- unifying ``model_optimizer``'s inconsistent
TRT/Native/ORT executor APIs. Per-stage backend selection (e.g. ``vit`` on
TensorRT, ``action_expert`` on PyTorch) is driven by the orchestrator.
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
