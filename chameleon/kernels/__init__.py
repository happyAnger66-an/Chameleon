"""跨平台自定义算子框架包 — 导出 OpSpec / KernelImpl 与注册表。

作用：
    re-export 算子抽象，import 时加载 fmha 等内置算子实现。

架构位置：
    优化/编译流水线 + 运行时 — 横切 frontend（stub 追踪）、compile
    （plugin 预加载）、runtime（参考实现）三层。
"""

from chameleon.kernels.base import (
    KERNEL_REGISTRY,
    OP_REGISTRY,
    KernelImpl,
    OpSpec,
    get_kernel,
    list_kernels,
    register_kernel,
    register_op,
)

# Import-time registration of built-in kernels.
from chameleon.kernels import fmha  # noqa: F401,E402

__all__ = [
    "KERNEL_REGISTRY",
    "OP_REGISTRY",
    "KernelImpl",
    "OpSpec",
    "get_kernel",
    "list_kernels",
    "register_kernel",
    "register_op",
]
