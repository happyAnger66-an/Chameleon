"""架构注册表 — ArchitectureSpec 的插件发现与查询。

作用：
    维护 ARCHITECTURE_REGISTRY，提供 register_architecture /
    get_architecture / list_architectures 接口。

架构位置：
    模型/架构层 — 被 config、cli、runtime/orchestrator 查询架构元数据
    （stage 列表、orchestrator 键）。
"""

from __future__ import annotations

from chameleon.architectures.base import ArchitectureSpec
from chameleon.core.registry import Registry

ARCHITECTURE_REGISTRY: Registry[str, ArchitectureSpec] = Registry("architecture")


def register_architecture(spec: ArchitectureSpec, *, override: bool = False) -> ArchitectureSpec:
    return ARCHITECTURE_REGISTRY.register(spec.name, spec, override=override)


def get_architecture(name: str) -> ArchitectureSpec:
    return ARCHITECTURE_REGISTRY.get(name)


def list_architectures() -> list[str]:
    return ARCHITECTURE_REGISTRY.keys()
