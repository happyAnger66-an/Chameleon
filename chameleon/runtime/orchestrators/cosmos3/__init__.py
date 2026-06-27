"""cosmos3 编排器注册 — import 时写入 ORCHESTRATOR_REGISTRY。"""

from chameleon.runtime.orchestrators.cosmos3 import real, reference
from chameleon.runtime.orchestrators.cosmos3.real import Cosmos3RealOrchestrator
from chameleon.runtime.orchestrators.cosmos3.reference import (
    Cosmos3Orchestrator,
    Cosmos3ReferenceOrchestrator,
)

__all__ = [
    "Cosmos3Orchestrator",
    "Cosmos3ReferenceOrchestrator",
    "Cosmos3RealOrchestrator",
    "reference",
    "real",
]
