"""Inference runtimes and VLA orchestration."""

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
