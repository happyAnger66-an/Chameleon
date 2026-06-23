"""TRT 接管后释放 PyTorch 大权重，保留 lang embed_tokens。"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def release_heavy_pytorch_weights(model: Any, *, embed_device: str | None = None) -> None:
    """TRT 接管后释放 vision / LLM 层 / expert 权重，仅保留 lang embed_tokens。"""
    pwe = model.paligemma_with_expert
    paligemma = pwe.paligemma.model

    embed_tokens = None
    if hasattr(paligemma, "language_model") and paligemma.language_model is not None:
        embed_tokens = getattr(paligemma.language_model, "embed_tokens", None)

    if hasattr(paligemma, "vision_tower"):
        logger.info("Releasing PyTorch vision_tower (TRT vit active)")
        del paligemma.vision_tower

    if hasattr(paligemma, "multi_modal_projector"):
        logger.info("Releasing PyTorch multi_modal_projector (included in TRT vit)")
        del paligemma.multi_modal_projector

    if hasattr(paligemma, "language_model") and paligemma.language_model is not None:
        logger.info("Releasing PyTorch language_model layers (TRT llm active)")
        del paligemma.language_model

    if embed_tokens is not None:
        if embed_device is not None:
            embed_tokens = embed_tokens.to(embed_device)
        pwe.embed_language_tokens = lambda tokens, et=embed_tokens: et(tokens)

    ge = pwe.gemma_expert
    if hasattr(ge, "model") and ge.model is not None:
        logger.info("Releasing PyTorch gemma_expert (TRT denoise active)")
        del ge.model
        ge.model = None
    if hasattr(ge, "lm_head") and ge.lm_head is not None:
        del ge.lm_head

    for attr in ("action_in_proj", "time_mlp_in", "time_mlp_out", "action_out_proj"):
        if hasattr(model, attr):
            delattr(model, attr)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
