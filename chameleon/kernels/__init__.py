"""Cross-platform custom operator / kernel framework."""

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
