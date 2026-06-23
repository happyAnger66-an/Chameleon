"""pi05 TRT Orchestrator — 框架 ``Orchestrator`` 实现，内部委托 ``Pi05TrtPipeline``。"""

from __future__ import annotations

from typing import Any

import torch

from chameleon.core.context import RunContext
from chameleon.models.base import ModelAdapter
from chameleon.runtime.base import Engine
from chameleon.runtime.orchestrator.base import Orchestrator, register_orchestrator
from chameleon.runtime.pi05_trt.pipeline import Pi05TrtPipeline


class Pi05TrtOrchestrator(Orchestrator):
    """注册于 ``ORCHESTRATOR_REGISTRY``（key ``pi05_trt``）的 pi05 TRT 编排器。

    Engine 由 workflow / evaluate 在 build 时注入 ``engines`` dict
    （vit / llm / denoise）；``requires_stage_engines=False`` 跳过 InferenceSession
    默认的 per-stage 加载。实际 TRT 控制流在 ``Pi05TrtPipeline``。
    """

    architecture = "pi05"
    requires_stage_engines = False

    def __init__(self, adapter: ModelAdapter, engines: dict[str, Engine], ctx: RunContext) -> None:
        super().__init__(adapter, engines, ctx)
        num_steps = int(ctx.options.get("num_steps", adapter.num_denoise_steps))
        self._pipeline = Pi05TrtPipeline(engines, num_steps=num_steps)

    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        if not getattr(self.adapter, "_is_real_openpi", False):
            raise RuntimeError(
                "Pi05TrtOrchestrator requires a built real openpi model."
            )
        device = self.ctx.torch_device
        num_steps = int(self.ctx.options.get("num_steps", self.adapter.num_denoise_steps))
        obs = self.adapter.to_openpi_observation(observation, device)
        with torch.no_grad():
            return self._pipeline.infer(
                self.adapter.model,
                device,
                obs,
                num_steps=num_steps,
            )


register_orchestrator("pi05_trt", Pi05TrtOrchestrator, override=True)
