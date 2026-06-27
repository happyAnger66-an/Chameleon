"""cosmos3 参考模型编排 — vae_encode → text_embed → dit 去噪环 → vae_decode。

复现 Cosmos3OmniPipeline 的生成控制流：条件 VAE 编码 + 文本 embedding 各算一次，
dit 去噪步在 flow-matching 环内重复 num_steps 次（每步做 CFG cond/uncond 两遍并
合成 velocity）。action 模式输出 action chunk（对齐 pi05 VLA），video 模式经
vae_decode 输出视频帧。
"""

from __future__ import annotations

from typing import Any

import torch

from chameleon.models.cosmos3.reference import create_sinusoidal_pos_embedding
from chameleon.runtime.orchestrator.base import Orchestrator, register_orchestrator


class Cosmos3ReferenceOrchestrator(Orchestrator):
    """参考 Cosmos3ReferenceModel 的四段式 stage 链 + flow-matching 去噪环（含 CFG）。"""

    architecture = "cosmos3"

    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        device = self.ctx.torch_device
        cond_pixels = observation["cond_pixels"].to(device)
        lang_tokens = observation["lang_tokens"].to(device)
        neg_lang_tokens = observation.get("neg_lang_tokens")
        if neg_lang_tokens is None:
            neg_lang_tokens = torch.zeros_like(lang_tokens)
        neg_lang_tokens = neg_lang_tokens.to(device)
        bsize = cond_pixels.shape[0]

        mode = str(observation.get("mode", getattr(self.adapter, "mode", "video")))
        cfg = self.adapter.config
        token_dim = cfg.token_dim
        num_tokens = cfg.action_horizon if mode == "action" else cfg.num_video_tokens

        num_steps = int(self.ctx.options.get("num_steps", self.adapter.num_denoise_steps))
        guidance = float(self.ctx.options.get("guidance_scale", getattr(cfg, "guidance_scale", 1.0)))
        time_dim = getattr(self.adapter, "time_embed_dim", token_dim)

        cond_latent = self.engines["vae_encode"].run({"cond_pixels": cond_pixels})["output"]
        text_mem = self.engines["text_embed"].run({"lang_tokens": lang_tokens})["output"]
        neg_text_mem = self.engines["text_embed"].run({"lang_tokens": neg_lang_tokens})["output"]

        dit = self.engines["dit"]
        do_cfg = guidance != 1.0
        x_t = torch.randn(bsize, num_tokens, token_dim, device=device)
        dt = -1.0 / num_steps
        time = 1.0
        while time >= -dt / 2:
            t = torch.full((bsize,), time, dtype=torch.float32, device=device)
            time_emb = create_sinusoidal_pos_embedding(t, time_dim)
            v_cond = dit.run(
                {"text_mem": text_mem, "cond_latent": cond_latent, "x_t": x_t, "time_emb": time_emb}
            )["output"]
            if do_cfg:
                v_uncond = dit.run(
                    {
                        "text_mem": neg_text_mem,
                        "cond_latent": cond_latent,
                        "x_t": x_t,
                        "time_emb": time_emb,
                    }
                )["output"]
                v_t = v_uncond + guidance * (v_cond - v_uncond)
            else:
                v_t = v_cond
            x_t = x_t + dt * v_t
            time += dt

        if mode == "action":
            return x_t
        return self.engines["vae_decode"].run({"latent": x_t})["output"]


# Registry key matches ArchitectureSpec.orchestrator ("cosmos3").
Cosmos3Orchestrator = Cosmos3ReferenceOrchestrator

register_orchestrator("cosmos3", Cosmos3ReferenceOrchestrator, override=True)
