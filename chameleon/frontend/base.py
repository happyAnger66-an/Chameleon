"""图捕获抽象 — PyTorch nn.Module stage 到平台中性图的统一接口。

作用：
    定义 GraphCapture ABC 及 GRAPH_CAPTURE_REGISTRY。capture() 将 stage
    模块 lower 为 Artifact（默认 ONNX；预留 torch.export FX）。

架构位置：
    优化/编译流水线 — 被 api.run_compile 调用，产出 ONNX Artifact 供
    compile/base.CompilerBackend 消费。自定义算子通过 kernels 在前端
    注册 stub 以参与追踪。
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
