"""Orchestrator 抽象与注册表 — 架构无关的控制流接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from chameleon.core.context import RunContext
from chameleon.core.registry import Registry
from chameleon.models.base import ModelAdapter
from chameleon.runtime.base import Engine


class Orchestrator(ABC):
    architecture: str

    requires_stage_engines: bool = True
    """是否需要 InferenceSession 预先按 ArchitectureSpec.stage_names 加载 Engine。"""

    def __init__(self, adapter: ModelAdapter, engines: dict[str, Engine], ctx: RunContext) -> None:
        self.adapter = adapter
        self.engines = engines
        self.ctx = ctx

    @abstractmethod
    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        """Run a full inference and return the action chunk ``[B, horizon, action_dim]``."""


ORCHESTRATOR_REGISTRY: Registry[str, type[Orchestrator]] = Registry("orchestrator")


def register_orchestrator(name: str, cls: type[Orchestrator], *, override: bool = False):
    return ORCHESTRATOR_REGISTRY.register(name, cls, override=override)
