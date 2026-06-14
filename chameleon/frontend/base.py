"""Unified frontend: PyTorch stage module -> platform-neutral graph.

The frontend is the single contract between model definitions and every compile
backend. A :class:`GraphCapture` lowers an ``nn.Module`` stage into a neutral
graph artifact (ONNX by default; ``torch.export`` FX is reserved for backends
that prefer it). Compiler backends then consume the resulting :class:`Artifact`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from chameleon.core.artifact import Artifact
from chameleon.core.registry import Registry


class GraphCapture(ABC):
    """Lowers a PyTorch module into a platform-neutral graph artifact."""

    name: str

    @abstractmethod
    def capture(
        self,
        module: Any,
        example_inputs: Sequence[Any],
        *,
        stage: str,
        output_path: str | None = None,
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        dynamic_axes: dict[str, Any] | None = None,
    ) -> Artifact:
        """Capture ``module`` and return an :class:`Artifact` describing the graph."""


GRAPH_CAPTURE_REGISTRY: Registry[str, GraphCapture] = Registry("graph_capture")


def register_graph_capture(capture: GraphCapture, *, override: bool = False) -> GraphCapture:
    return GRAPH_CAPTURE_REGISTRY.register(capture.name, capture, override=override)


def get_graph_capture(name: str) -> GraphCapture:
    return GRAPH_CAPTURE_REGISTRY.get(name)
