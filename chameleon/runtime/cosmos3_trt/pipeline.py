"""cosmos3 TRT 推理管线 — vae_encode → text_embed → dit 去噪环 → vae_decode。

对齐 Cosmos3OmniPipeline 的去噪控制流，但每个 stage 由 TRT engine 执行：条件 VAE
编码与文本 embedding 各算一次，dit 去噪步在 flow-matching 环内重复（每步 CFG
cond/uncond 两遍合成 velocity）。与参考编排器共享相同 engine-dict 接口（``run`` 返回
dict），因此可用 PyTorch / TensorRT 任一后端的 Engine 驱动。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from chameleon.models.cosmos3.reference import create_sinusoidal_pos_embedding
from chameleon.runtime.base import Engine

logger = logging.getLogger(__name__)


def _take(out: dict[str, Any], *names: str):
    """Pull a tensor from an engine output dict by known name, else first value."""
    for name in names:
        if name in out:
            return out[name]
    if "output" in out:
        return out["output"]
    return next(iter(out.values()))


class Cosmos3TrtPipeline:
    """Cosmos3 generator denoise kernel over a stage engine dict (TRT or PyTorch)."""

    def __init__(
        self,
        engines: dict[str, Engine],
        *,
        num_steps: int = 35,
        guidance_scale: float = 6.0,
    ) -> None:
        self._engines = engines
        self._num_steps = num_steps
        self._guidance = guidance_scale

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def infer(
        self,
        observation: dict[str, Any],
        device: str | torch.device,
        *,
        mode: str = "video",
        token_dim: int,
        num_video_tokens: int,
        action_horizon: int,
        time_dim: int,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        steps = int(num_steps if num_steps is not None else self._num_steps)
        guidance = float(guidance_scale if guidance_scale is not None else self._guidance)

        cond_pixels = observation["cond_pixels"].to(dev)
        lang_tokens = observation["lang_tokens"].to(dev)
        neg_lang_tokens = observation.get("neg_lang_tokens")
        if neg_lang_tokens is None:
            neg_lang_tokens = torch.zeros_like(lang_tokens)
        neg_lang_tokens = neg_lang_tokens.to(dev)
        bsize = cond_pixels.shape[0]
        num_tokens = action_horizon if mode == "action" else num_video_tokens

        cond_latent = _take(
            self._engines["vae_encode"].run({"cond_pixels": cond_pixels}), "cond_latent"
        )
        text_mem = _take(
            self._engines["text_embed"].run({"lang_tokens": lang_tokens}), "text_mem"
        )
        neg_text_mem = _take(
            self._engines["text_embed"].run({"lang_tokens": neg_lang_tokens}), "text_mem"
        )

        dit = self._engines["dit"]
        do_cfg = guidance != 1.0
        x_t = torch.randn(bsize, num_tokens, token_dim, device=dev)
        dt = -1.0 / steps
        time = 1.0
        while time >= -dt / 2:
            t = torch.full((bsize,), time, dtype=torch.float32, device=dev)
            time_emb = create_sinusoidal_pos_embedding(t, time_dim)
            v_cond = _take(
                dit.run(
                    {
                        "text_mem": text_mem,
                        "cond_latent": cond_latent,
                        "x_t": x_t,
                        "time_emb": time_emb,
                    }
                ),
                "v_t",
            )
            if do_cfg:
                v_uncond = _take(
                    dit.run(
                        {
                            "text_mem": neg_text_mem,
                            "cond_latent": cond_latent,
                            "x_t": x_t,
                            "time_emb": time_emb,
                        }
                    ),
                    "v_t",
                )
                v_t = v_uncond + guidance * (v_cond - v_uncond)
            else:
                v_t = v_cond
            x_t = x_t + dt * v_t
            time += dt

        if mode == "action":
            return x_t
        return _take(self._engines["vae_decode"].run({"latent": x_t}), "video")
