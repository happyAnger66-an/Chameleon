"""Model architecture specifications."""

from chameleon.architectures.base import ArchitectureSpec, StageSpec
from chameleon.architectures.registry import (
    ARCHITECTURE_REGISTRY,
    get_architecture,
    list_architectures,
    register_architecture,
)

# Import-time registration of built-in architectures.
from chameleon.architectures import pi05  # noqa: F401,E402

__all__ = [
    "ArchitectureSpec",
    "StageSpec",
    "ARCHITECTURE_REGISTRY",
    "get_architecture",
    "list_architectures",
    "register_architecture",
]
