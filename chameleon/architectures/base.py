"""Architecture and stage abstractions.

An :class:`ArchitectureSpec` decomposes a model family into ordered *stages*.
Each :class:`StageSpec` is an independently quantizable / compilable unit and
declares which platforms it can target. This is carried over (and generalized
with a ``platform`` dimension) from ``model_optimizer``'s
``ArchitectureSpec`` / ``StageSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageSpec:
    """A single compile/quantize unit within an architecture."""

    name: str
    """e.g. ``vit`` / ``llm_prefix`` / ``action_expert``."""

    description: str = ""
    quantizable: bool = True
    supported_platforms: tuple[str, ...] = ()
    """Platform names this stage can target. Empty means "all registered platforms"."""

    def supports_platform(self, platform: str) -> bool:
        return not self.supported_platforms or platform in self.supported_platforms


@dataclass(frozen=True)
class ArchitectureSpec:
    """Describes a model family as an ordered list of stages."""

    name: str
    stages: tuple[StageSpec, ...]
    description: str = ""
    orchestrator: str = "sequential"
    """Key of the :class:`~chameleon.runtime.orchestrator.Orchestrator` to drive these stages."""

    metadata: dict = field(default_factory=dict)

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stages)

    def stage(self, name: str) -> StageSpec:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(f"Architecture {self.name!r} has no stage {name!r}.")
