"""Compiler backend abstraction (the core pluggable layer).

Each platform plugs in a :class:`CompilerBackend` that turns a neutral graph
:class:`Artifact` (plus an optional :class:`QuantMetadata` contract) into a
deployable engine :class:`Artifact`. All toolchain-specific differences --
TensorRT vs OpenVINO vs TVM vs Horizon BPU SDK -- are confined to this layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from chameleon.core.artifact import Artifact
from chameleon.core.context import CompileContext
from chameleon.core.registry import Registry
from chameleon.quantization.base import QuantMetadata


class CompilerBackend(ABC):
    """Lowers a neutral graph into a platform-specific engine."""

    name: str
    """Backend key, matched against :attr:`PlatformSpec.compiler`."""

    def available(self) -> bool:
        """Whether the backend's toolchain is importable in this environment."""
        return True

    @abstractmethod
    def compile(
        self,
        graph: Artifact,
        quant_meta: QuantMetadata | None,
        ctx: CompileContext,
        cfg: dict | None = None,
    ) -> Artifact:
        """Compile ``graph`` and return an engine :class:`Artifact`."""


COMPILER_REGISTRY: Registry[str, CompilerBackend] = Registry("compiler")


def register_compiler(backend: CompilerBackend, *, override: bool = False) -> CompilerBackend:
    return COMPILER_REGISTRY.register(backend.name, backend, override=override)


def get_compiler(name: str) -> CompilerBackend:
    return COMPILER_REGISTRY.get(name)


def list_compilers() -> list[str]:
    return COMPILER_REGISTRY.keys()
