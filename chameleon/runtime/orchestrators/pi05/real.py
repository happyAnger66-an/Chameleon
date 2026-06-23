"""pi05 真实 openpi 编排 — 整模型 PyTorch sample_actions 路径。"""

from __future__ import annotations

from typing import Any

import torch

from chameleon.runtime.orchestrator.base import Orchestrator, register_orchestrator


class Pi05RealOrchestrator(Orchestrator):
    """真实 openpi PI0Pytorch 端到端编排（不拆 stage / 不加载 per-stage engine）。"""

    architecture = "pi05"
    requires_stage_engines = False

    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        model = getattr(self.adapter, "model", None)
        if model is None or not getattr(self.adapter, "_is_real_openpi", False):
            raise RuntimeError(
                "Pi05RealOrchestrator requires a built real openpi model; "
                "use pi05 reference orchestrator for Pi05ReferenceModel."
            )
        device = self.ctx.torch_device
        num_steps = int(self.ctx.options.get("num_steps", self.adapter.num_denoise_steps))
        obs = self.adapter.to_openpi_observation(observation, device)
        with torch.no_grad():
            return model.sample_actions(device, obs, num_steps=num_steps)


register_orchestrator("pi05_real", Pi05RealOrchestrator, override=True)
