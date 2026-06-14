"""跨平台自定义算子框架 — 逻辑算子与平台 Kernel 的三段式模式。

作用：
    定义 OpSpec（逻辑算子描述）和 KernelImpl ABC，实现三段式扩展：
    1) frontend_stub — torch.library custom op（追踪/导出）
    2) graph_node — ONNX symbolic / 图节点
    3) backend_artifact — 平台 plugin .so / kernel lib

架构位置：
    算子层 — 设计对标 model_optimizer 与 TensorRT-Edge-LLM custom op
    三段式。KERNEL_REGISTRY 键为 (op, vendor)，被 frontend 和 compile 引用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chameleon.core.registry import Registry


@dataclass(frozen=True)
class OpSpec:
    """Description of a logical (platform-independent) custom operator."""

    name: str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


class KernelImpl(ABC):
    """A platform-specific implementation of an :class:`OpSpec`."""

    op: str
    platform_vendor: str
    """Which vendor this kernel targets (``nvidia``/``amd``/``cpu``/...)."""

    def frontend_stub(self):
        """Return / register the torch custom op used during capture (optional)."""
        return None

    def graph_node(self, *args, **kwargs):
        """Emit the op into the neutral graph (e.g. ONNX symbolic). Optional."""
        raise NotImplementedError

    def backend_artifact(self, kernel_tag: str | None = None):
        """Return the executable artifact (plugin path / callable). Optional."""
        return None

    @abstractmethod
    def reference(self, *args, **kwargs):
        """A correctness reference (used by the PyTorch runtime and tests)."""


# Keyed by (op_name, vendor).
KERNEL_REGISTRY: Registry[tuple[str, str], KernelImpl] = Registry("kernel")
OP_REGISTRY: Registry[str, OpSpec] = Registry("op")


def register_op(spec: OpSpec, *, override: bool = False) -> OpSpec:
    return OP_REGISTRY.register(spec.name, spec, override=override)


def register_kernel(impl: KernelImpl, *, override: bool = False) -> KernelImpl:
    return KERNEL_REGISTRY.register((impl.op, impl.platform_vendor), impl, override=override)


def get_kernel(op: str, vendor: str) -> KernelImpl | None:
    return KERNEL_REGISTRY.get_or_none((op, vendor))


def list_kernels() -> list[tuple[str, str]]:
    return KERNEL_REGISTRY.keys()
