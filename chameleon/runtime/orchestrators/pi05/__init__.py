"""pi05 编排器注册 — import 时写入 ORCHESTRATOR_REGISTRY。"""

from chameleon.runtime.orchestrators.pi05 import real, reference
from chameleon.runtime.orchestrators.pi05.reference import (
    Pi05Orchestrator,
    Pi05ReferenceOrchestrator,
)
from chameleon.runtime.orchestrators.pi05.real import Pi05RealOrchestrator

__all__ = [
    "Pi05Orchestrator",
    "Pi05ReferenceOrchestrator",
    "Pi05RealOrchestrator",
    "reference",
    "real",
]
