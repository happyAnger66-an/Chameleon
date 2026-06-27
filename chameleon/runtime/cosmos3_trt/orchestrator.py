"""cosmos3 TRT Orchestrator — 框架 Orchestrator 实现，委托 Cosmos3TrtPipeline。

注册 key ``cosmos3_trt``。``requires_stage_engines=True``：由 InferenceSession 按
ArchitectureSpec.stage_names（vae_encode / text_embed / dit / vae_decode）加载 engine，
``stage_runtimes`` 指向 tensorrt 时即为 TRT 推理；stage_artifacts 注入已编译 engine
（闭合 compile→infer）。实际 TRT 去噪控制流在 Cosmos3TrtPipeline。
"""

from __future__ import annotations

from typing import Any

import torch

from chameleon.core.context import RunContext
from chameleon.models.base import ModelAdapter
from chameleon.runtime.base import Engine
from chameleon.runtime.cosmos3_trt.pipeline import Cosmos3TrtPipeline
from chameleon.runtime.orchestrator.base import Orchestrator, register_orchestrator


class Cosmos3TrtOrchestrator(Orchestrator):
    architecture = "cosmos3"
    requires_stage_engines = True

    def __init__(self, adapter: ModelAdapter, engines: dict[str, Engine], ctx: RunContext) -> None:
        super().__init__(adapter, engines, ctx)
        num_steps = int(ctx.options.get("num_steps", adapter.num_denoise_steps))
        guidance = float(ctx.options.get("guidance_scale", getattr(adapter.config, "guidance_scale", 1.0)))
        self._pipeline = Cosmos3TrtPipeline(engines, num_steps=num_steps, guidance_scale=guidance)

    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        cfg = self.adapter.config
        mode = str(observation.get("mode", getattr(self.adapter, "mode", "video")))
        return self._pipeline.infer(
            observation,
            self.ctx.torch_device,
            mode=mode,
            token_dim=cfg.token_dim,
            num_video_tokens=cfg.num_video_tokens,
            action_horizon=cfg.action_horizon,
            time_dim=getattr(self.adapter, "time_embed_dim", cfg.token_dim),
        )


register_orchestrator("cosmos3_trt", Cosmos3TrtOrchestrator, override=True)
