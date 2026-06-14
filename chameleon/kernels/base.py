"""Cross-platform custom operator framework.

A logical operator (:class:`OpSpec`, e.g. ``fmha_d256``) can have one
:class:`KernelImpl` per platform. Each implementation follows the three-stage
pattern used by both ``model_optimizer`` and TensorRT-Edge-LLM:

1. ``frontend_stub``  - a ``torch.library`` custom op so the model traces/exports.
2. ``graph_node``     - how the op appears in the neutral graph (ONNX symbolic).
3. ``backend_artifact`` - the platform kernel/plugin that actually executes it
   (e.g. a TRT plugin ``.so``, a CuTe DSL artifact tagged per ``kernel_tag``,
   or a pure-PyTorch reference for CPU).
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
