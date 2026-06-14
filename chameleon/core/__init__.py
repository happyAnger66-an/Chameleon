"""核心抽象层包 — 导出全框架共享的基础类型与注册表。

作用：
    统一 re-export Artifact、Manifest、CompileContext、RunContext、
    PlatformSpec、Registry 等核心类型，供各子系统 import。

架构位置：
    平台抽象 + 基础设施层 — 被 quantization、compile、runtime、
    frontend 等所有子系统依赖，自身不依赖上层模块。
"""

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
