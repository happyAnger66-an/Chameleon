"""InferenceSession — 按 ArchitectureSpec 加载 stage engine 并驱动 Orchestrator。"""

from __future__ import annotations

from typing import Any

import torch

from chameleon.architectures.base import ArchitectureSpec
from chameleon.architectures.registry import get_architecture
from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.models.base import ModelAdapter
from chameleon.runtime.base import Engine, get_runtime
from chameleon.runtime.orchestrator.base import ORCHESTRATOR_REGISTRY, Orchestrator


class InferenceSession:
    """Builds per-stage engines and the architecture orchestrator, then runs inference.

    ``stage_runtimes`` maps each stage to a runtime backend name, enabling
    stage-level backend mixing (e.g. ``{"vit": "tensorrt", "action_expert": "pytorch"}``).
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        ctx: RunContext,
        stage_runtimes: dict[str, str] | None = None,
        stage_artifacts: dict[str, Artifact] | None = None,
    ) -> None:
        self.adapter = adapter
        self.ctx = ctx
        self.arch: ArchitectureSpec = get_architecture(adapter.architecture)
        self.stage_runtimes = stage_runtimes or {}
        self.stage_artifacts = stage_artifacts or {}
        self._engines: dict[str, Engine] = {}
        self._orchestrator: Orchestrator | None = None

    def _resolve_runtime(self, stage: str) -> str:
        return self.stage_runtimes.get(stage, self.ctx.platform.runtime)

    def _artifact_for(self, stage: str) -> Artifact:
        if stage in self.stage_artifacts:
            return self.stage_artifacts[stage]
        return Artifact(
            kind="reference",
            stage=stage,
            platform=self.ctx.platform.name,
            payload=self.adapter.stage_module(stage),
        )

    def build(self) -> "InferenceSession":
        orch_key = getattr(self.adapter, "orchestrator_key", None) or self.arch.orchestrator
        orch_cls = ORCHESTRATOR_REGISTRY.get(orch_key)
        if getattr(orch_cls, "requires_stage_engines", True):
            for stage in self.arch.stage_names:
                runtime_name = self._resolve_runtime(stage)
                backend = get_runtime(runtime_name)
                artifact = self._artifact_for(stage)
                self.ctx.on_progress(f"loading stage {stage} on {runtime_name}", 0.0)
                self._engines[stage] = backend.load(artifact, self.ctx)
        self._orchestrator = orch_cls(self.adapter, self._engines, self.ctx)
        return self

    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        if self._orchestrator is None:
            self.build()
        assert self._orchestrator is not None
        return self._orchestrator.infer(observation)
