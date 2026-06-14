"""Pluggable compiler backends."""

from chameleon.compile.base import (
    COMPILER_REGISTRY,
    CompilerBackend,
    get_compiler,
    list_compilers,
    register_compiler,
)

# Import-time registration of built-in backends.
from chameleon.compile import tensorrt  # noqa: F401,E402
from chameleon.compile import stubs  # noqa: F401,E402

__all__ = [
    "COMPILER_REGISTRY",
    "CompilerBackend",
    "get_compiler",
    "list_compilers",
    "register_compiler",
]
