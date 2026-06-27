"""推理运行时包 — Engine / RuntimeBackend 抽象与 VLA 编排注册。"""

from chameleon.runtime.base import (
    RUNTIME_REGISTRY,
    Engine,
    RuntimeBackend,
    get_runtime,
    list_runtimes,
    register_runtime,
)

# Import-time registration of built-in runtimes and orchestrators.
from chameleon.runtime import pytorch  # noqa: F401,E402
from chameleon.runtime import tensorrt  # noqa: F401,E402
from chameleon.runtime.orchestrator import (  # noqa: E402
    ORCHESTRATOR_REGISTRY,
    InferenceSession,
    Orchestrator,
    register_orchestrator,
)
from chameleon.runtime import orchestrators  # noqa: F401,E402
from chameleon.runtime.pi05_trt import orchestrator as _pi05_trt_orchestrator  # noqa: F401,E402
from chameleon.runtime.cosmos3_trt import orchestrator as _cosmos3_trt_orchestrator  # noqa: F401,E402

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
