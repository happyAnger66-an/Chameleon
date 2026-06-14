"""Core abstractions shared by all Chameleon subsystems."""

from chameleon.core.artifact import Artifact, Manifest
from chameleon.core.context import CompileContext, ProgressCallback, RunContext
from chameleon.core.platform import (
    PLATFORM_REGISTRY,
    PlatformSpec,
    get_platform,
    list_platforms,
    register_platform,
)
from chameleon.core.registry import Registry

__all__ = [
    "Artifact",
    "Manifest",
    "CompileContext",
    "RunContext",
    "ProgressCallback",
    "PlatformSpec",
    "PLATFORM_REGISTRY",
    "get_platform",
    "list_platforms",
    "register_platform",
    "Registry",
]
