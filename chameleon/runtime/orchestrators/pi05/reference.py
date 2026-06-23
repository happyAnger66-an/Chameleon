"""pi05 参考模型编排 — vit → llm_prefix → action_expert 三段式 engine 链。"""

from __future__ import annotations

from typing import Any

import torch

from chameleon.models.pi05.reference import create_sinusoidal_pos_embedding
from chameleon.runtime.orchestrator.base import Orchestrator, register_orchestrator


class Pi05ReferenceOrchestrator(Orchestrator):
    """参考 Pi05ReferenceModel 的三段式 stage 链 + flow-matching 去噪环。"""

    architecture = "pi05"

    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        device = self.ctx.torch_device
        images = observation["images"].to(device)
        lang_tokens = observation["lang_tokens"].to(device)
        state = observation["state"].to(device)
        bsize = state.shape[0]

        img_tokens = self.engines["vit"].run({"images": images})["output"]
        prefix_memory = self.engines["llm_prefix"].run(
            {"img_tokens": img_tokens, "lang_tokens": lang_tokens}
        )["output"]

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


# Registry key matches ArchitectureSpec.orchestrator ("pi05").
Pi05Orchestrator = Pi05ReferenceOrchestrator

register_orchestrator("pi05", Pi05ReferenceOrchestrator, override=True)
