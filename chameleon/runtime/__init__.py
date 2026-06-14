"""推理运行时包 — 导出 Engine / RuntimeBackend 抽象与 VLA 编排。

作用：
    re-export 运行时接口，import 时注册 PyTorch / TensorRT 运行时及
    Pi05Orchestrator。

架构位置：
    运行时层 — 加载 compile 产出的 engine 或 reference nn.Module，
    经 VLAOrchestrator 驱动链式执行与去噪环。
"""

from chameleon.runtime.base import (
    RUNTIME_REGISTRY,
    Engine,
    RuntimeBackend,
    get_runtime,
    list_runtimes,
    register_runtime,
)

# Import-time registration of built-in runtimes.
from chameleon.runtime import pytorch  # noqa: F401,E402
from chameleon.runtime import tensorrt  # noqa: F401,E402
from chameleon.runtime.orchestrator import (  # noqa: E402
    ORCHESTRATOR_REGISTRY,
    InferenceSession,
    Orchestrator,
    register_orchestrator,
)

__all__ = [
    "RUNTIME_REGISTRY",
    "Engine",
    "RuntimeBackend",
    "get_runtime",
    "list_runtimes",
    "register_runtime",
    "ORCHESTRATOR_REGISTRY",
    "InferenceSession",
    "Orchestrator",
    "register_orchestrator",
]
