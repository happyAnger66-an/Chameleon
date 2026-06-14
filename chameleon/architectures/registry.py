"""Architecture registry."""

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
