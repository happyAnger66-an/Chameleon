"""编译后端抽象 — 平台中性图到可部署 engine 的核心可插拔层。

作用：
    定义 CompilerBackend ABC：compile(graph, quant_meta, ctx) → engine
    Artifact。COMPILER_REGISTRY 键匹配 PlatformSpec.compiler。

架构位置：
    优化/编译流水线 — 被 api.run_compile 调用。上游：frontend 的 ONNX
    Artifact + quantization 的 QuantMetadata；下游：runtime 加载 engine。
    TensorRT / OpenVINO / TVM / Horizon 差异均隔离在此层。
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
