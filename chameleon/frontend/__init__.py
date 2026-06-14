"""统一前端包 — 图捕获（PyTorch stage → 平台中性图）。

作用：
    re-export GraphCapture 抽象与注册表，import 时注册 ONNX 导出实现。

架构位置：
    优化/编译流水线 — 位于 models 与 compile 之间，是模型定义与各
    编译后端之间的唯一契约层。
"""

from chameleon.frontend.base import (
    GRAPH_CAPTURE_REGISTRY,
    GraphCapture,
    get_graph_capture,
    register_graph_capture,
)
from chameleon.frontend import onnx_export  # noqa: F401,E402

__all__ = [
    "GRAPH_CAPTURE_REGISTRY",
    "GraphCapture",
    "get_graph_capture",
    "register_graph_capture",
]
