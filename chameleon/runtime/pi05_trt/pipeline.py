"""pi05 四段式 TRT 推理管线 — vit → llm → denoise 环（非框架 Orchestrator）。"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from chameleon.models.pi05.attention import make_att_2d_masks
from chameleon.runtime.base import Engine

logger = logging.getLogger(__name__)


def _embed_prefix_trt(
    model: Any,
    vit_engine: Engine,
    images: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SigLIP 走 TRT vit.engine；语言 embedding 仍用 PyTorch embed_tokens。"""
    embs: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []
    att_masks: list[int] = []

    for img, img_mask in zip(images, img_masks, strict=True):
        img_emb = vit_engine.run({"pixel_values": img})["output"]
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        att_masks.extend([0] * num_img_embs)

    pwe = model.paligemma_with_expert
    lang_emb = pwe.embed_language_tokens(lang_tokens)
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb = lang_emb * math.sqrt(lang_emb_dim)

    embs.append(lang_emb)
    pad_masks.append(lang_masks)
    att_masks.extend([0] * lang_emb.shape[1])

    embs_cat = torch.cat(embs, dim=1)
    pad_cat = torch.cat(pad_masks, dim=1)
    bsize = pad_cat.shape[0]
    att_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_cat.device)
    att_t = att_t[None, :].expand(bsize, att_t.shape[0])
    return embs_cat, pad_cat, att_t


class Pi05TrtPipeline:
    """对齐 openpi ``PI0Pytorch.sample_actions`` 的 TRT 推理内核。"""

    def __init__(
        self,
        engines: dict[str, Engine],
        *,
        num_steps: int = 10,
    ) -> None:
        self._engines = engines
        self._num_steps = num_steps

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def infer(
        self,
        model: Any,
        device: str | torch.device,
        observation: Any,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Run TRT vit → llm → denoise loop and return action chunk ``[B, H, D]``."""
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        steps = int(num_steps if num_steps is not None else self._num_steps)
        bsize = observation.state.shape[0]
        if noise is None:
            noise = model.sample_noise(
                (bsize, model.config.action_horizon, model.config.action_dim),
                dev,
            )

        images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(  # noqa: SLF001
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = _embed_prefix_trt(
            model,
            self._engines["vit"],
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = model._prepare_attention_masks_4d(prefix_att_2d)  # noqa: SLF001

        llm_out = self._engines["llm"].run(
            {
                "inputs_embeds": prefix_embs,
                "attention_mask": prefix_att_4d,
                "position_ids": prefix_position_ids,
            }
        )
        past_keys = llm_out.get("past_keys", llm_out["output"])
        past_values = llm_out["past_values"]

        dt = -1.0 / steps
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=dev)
        denoise = self._engines["denoise"]

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            step_out = denoise.run(
                {
                    "prefix_pad_masks": prefix_pad_masks,
                    "past_keys": past_keys,
                    "past_values": past_values,
                    "x_t": x_t,
                    "timestep": expanded_time,
                }
            )
            v_t = step_out.get("v_t", step_out["output"])
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t
