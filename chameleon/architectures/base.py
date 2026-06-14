"""架构与 Stage 抽象 — 模型无关的分段编译单元定义。

作用：
    定义 StageSpec（单个可量化/可编译单元，声明 supported_platforms）和
    ArchitectureSpec（有序 stage 列表 + orchestrator 键），将 VLA 模型
    分解为 vit / llm_prefix / action_expert 等独立单元。

架构位置：
    模型/架构层 — 上游被 config 引用 stage 名，下游被 models（ModelAdapter
    按 stage 暴露模块）和 runtime/orchestrator（按 stage 链式执行）消费。
    继承自 model_optimizer 的 ArchitectureSpec / StageSpec 并增加平台维度。
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
