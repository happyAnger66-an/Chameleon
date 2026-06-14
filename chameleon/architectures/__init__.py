"""模型架构层包 — 导出 ArchitectureSpec / StageSpec 及注册表接口。

作用：
    re-export 架构抽象类型，并在 import 时加载 pi05 等内置架构定义。

架构位置：
    模型/架构层 — 位于 models 与 runtime/orchestrator 之间，定义"模型
    如何拆分为 stage"的静态规格，不持有具体 nn.Module。
"""

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
