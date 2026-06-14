"""VLA 编排器 — 架构特定的链式执行与去噪环控制流。

作用：
    定义 Orchestrator ABC 和 Pi05Orchestrator（vit → llm_prefix →
    action_expert 去噪环），以及 InferenceSession（按 stage 加载 Engine、
    构建编排器、执行 infer）。ORCHESTRATOR_REGISTRY 按 architecture 键注册。

架构位置：
    运行时层 — 框架控制流核心。上游：architectures（stage 顺序）、
    models（adapter 元数据）；下游：各 stage 的 Engine.run。KV handoff
    与去噪热点均在此层实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from chameleon.architectures.base import ArchitectureSpec
from chameleon.architectures.registry import get_architecture
from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.core.registry import Registry
from chameleon.models.base import ModelAdapter
from chameleon.models.pi05.reference import create_sinusoidal_pos_embedding
from chameleon.runtime.base import Engine, get_runtime


class Orchestrator(ABC):
    architecture: str

    def __init__(self, adapter: ModelAdapter, engines: dict[str, Engine], ctx: RunContext) -> None:
        self.adapter = adapter
        self.engines = engines
        self.ctx = ctx

    @abstractmethod
    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        """Run a full inference and return the action chunk ``[B, horizon, action_dim]``."""


ORCHESTRATOR_REGISTRY: Registry[str, type[Orchestrator]] = Registry("orchestrator")


def register_orchestrator(name: str, cls: type[Orchestrator], *, override: bool = False):
    return ORCHESTRATOR_REGISTRY.register(name, cls, override=override)


class Pi05Orchestrator(Orchestrator):
    architecture = "pi05"

    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        device = self.ctx.torch_device
        images = observation["images"].to(device)
        lang_tokens = observation["lang_tokens"].to(device)
        state = observation["state"].to(device)
        bsize = state.shape[0]

        # Stage 1: vision encoder.
        img_tokens = self.engines["vit"].run({"images": images})["output"]

        # Stage 2: prefix / KV memory (computed once and reused across denoise steps).
        prefix_memory = self.engines["llm_prefix"].run(
            {"img_tokens": img_tokens, "lang_tokens": lang_tokens}
        )["output"]

        # Stage 3: flow-matching denoise loop (Euler integration t: 1 -> 0).
        action_dim = self.adapter.action_dim
        horizon = self.adapter.action_horizon
        num_steps = int(self.ctx.options.get("num_steps", self.adapter.num_denoise_steps))
        time_dim = getattr(self.adapter, "time_embed_dim", action_dim)

        x_t = torch.randn(bsize, horizon, action_dim, device=device)
        dt = -1.0 / num_steps
        time = 1.0
        action_engine = self.engines["action_expert"]
        while time >= -dt / 2:
            t = torch.full((bsize,), time, dtype=torch.float32, device=device)
            time_emb = create_sinusoidal_pos_embedding(
                t, time_dim, min_period=4e-3, max_period=4.0
            )
            v_t = action_engine.run(
                {
                    "state": state,
                    "prefix_memory": prefix_memory,
                    "x_t": x_t,
                    "time_emb": time_emb,
                }
            )["output"]
            x_t = x_t + dt * v_t
            time += dt
        return x_t


register_orchestrator("pi05", Pi05Orchestrator, override=True)


class InferenceSession:
    """Builds per-stage engines and the architecture orchestrator, then runs inference.

    ``stage_runtimes`` maps each stage to a runtime backend name, enabling
    stage-level backend mixing (e.g. ``{"vit": "tensorrt", "action_expert": "pytorch"}``).
    For the reference path, every stage uses the ``pytorch`` runtime and is fed a
    reference artifact carrying the stage's ``nn.Module``.
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
        # Default: reference artifact wrapping the stage's nn.Module.
        return Artifact(
            kind="reference",
            stage=stage,
            platform=self.ctx.platform.name,
            payload=self.adapter.stage_module(stage),
        )

    def build(self) -> "InferenceSession":
        for stage in self.arch.stage_names:
            runtime_name = self._resolve_runtime(stage)
            backend = get_runtime(runtime_name)
            artifact = self._artifact_for(stage)
            self.ctx.on_progress(f"loading stage {stage} on {runtime_name}", 0.0)
            self._engines[stage] = backend.load(artifact, self.ctx)
        orch_cls = ORCHESTRATOR_REGISTRY.get(self.arch.orchestrator)
        self._orchestrator = orch_cls(self.adapter, self._engines, self.ctx)
        return self

    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        if self._orchestrator is None:
            self.build()
        assert self._orchestrator is not None
        return self._orchestrator.infer(observation)
