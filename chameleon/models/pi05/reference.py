"""Lightweight reference implementation of the pi05 stages.

This module faithfully reproduces the *structure and algorithm* of openpi's
``pi0_pytorch.py`` (three stages + flow-matching denoise loop, sinusoidal time
embedding) using small, randomly-initialized layers. It lets the full Chameleon
pipeline -- orchestrator, KV handoff, denoise loop, stage-level backend mixing --
run end-to-end without the heavy openpi/transformers checkpoint dependencies.

The real weights are wired in via ``Pi05Adapter`` when a checkpoint and the
openpi runtime are available; both paths expose the identical stage interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class Pi05Config:
    """Configuration for the pi05 adapter.

    Defaults match openpi's ``Pi0Config(pi05=True)`` for the externally visible
    quantities (``action_dim`` / ``action_horizon``). The hidden widths are kept
    small for the reference path so the MVP runs quickly on CPU; set
    ``use_reference=False`` with a ``checkpoint`` to load the real model.
    """

    action_dim: int = 32
    action_horizon: int = 50
    num_denoise_steps: int = 10

    # Hidden sizes (reference path keeps these small for speed).
    prefix_width: int = 256
    expert_width: int = 256
    num_image_tokens: int = 64
    image_size: int = 224
    image_channels: int = 3
    vocab_size: int = 257152
    max_lang_len: int = 48

    # Model source selection.
    use_reference: bool = True
    checkpoint: str | None = None
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float
) -> Tensor:
    """Sine-cosine positional embedding for scalar positions.

    Mirrors ``pi0_pytorch.create_sinusoidal_pos_embedding``.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    device = time.device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


class Pi05ViT(nn.Module):
    """Stage ``vit``: image -> image tokens (SigLIP surrogate)."""

    def __init__(self, cfg: Pi05Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(cfg.image_channels, cfg.prefix_width, kernel_size=16, stride=16)
        self.pool = nn.AdaptiveAvgPool1d(cfg.num_image_tokens)
        self.norm = nn.LayerNorm(cfg.prefix_width)

    def forward(self, images: Tensor) -> Tensor:
        # images: [B, 3, H, W] -> tokens: [B, num_image_tokens, width]
        x = self.patch(images)  # [B, C, H/16, W/16]
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)  # [B, C, P]
        x = self.pool(x)  # [B, C, num_image_tokens]
        x = x.transpose(1, 2)  # [B, num_image_tokens, C]
        return self.norm(x)


class Pi05Prefix(nn.Module):
    """Stage ``llm_prefix``: image+language tokens -> prefix memory (KV surrogate)."""

    def __init__(self, cfg: Pi05Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.lang_embed = nn.Embedding(cfg.vocab_size, cfg.prefix_width)
        self.block = nn.TransformerEncoderLayer(
            d_model=cfg.prefix_width,
            nhead=8,
            dim_feedforward=cfg.prefix_width * 2,
            batch_first=True,
            activation="gelu",
        )
        self.norm = nn.LayerNorm(cfg.prefix_width)

    def forward(self, img_tokens: Tensor, lang_tokens: Tensor) -> Tensor:
        lang_emb = self.lang_embed(lang_tokens) * math.sqrt(self.cfg.prefix_width)
        prefix = torch.cat([img_tokens, lang_emb], dim=1)
        prefix = self.block(prefix)
        return self.norm(prefix)


class Pi05ActionExpert(nn.Module):
    """Stage ``action_expert``: one flow-matching denoise step.

    Consumes the prefix memory produced by :class:`Pi05Prefix` (the KV handoff)
    and predicts the flow velocity ``v_t`` for the noisy action chunk ``x_t``.
    """

    def __init__(self, cfg: Pi05Config) -> None:
        super().__init__()
        self.cfg = cfg
        w = cfg.expert_width
        self.action_in_proj = nn.Linear(cfg.action_dim, w)
        self.time_mlp_in = nn.Linear(w, w)
        self.time_mlp_out = nn.Linear(w, w)
        self.state_proj = nn.Linear(cfg.action_dim, w)
        self.prefix_proj = nn.Linear(cfg.prefix_width, w)
        self.cross_attn = nn.MultiheadAttention(w, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(w, w * 2), nn.SiLU(), nn.Linear(w * 2, w))
        self.norm = nn.LayerNorm(w)
        self.action_out_proj = nn.Linear(w, cfg.action_dim)

    def forward(
        self, state: Tensor, prefix_memory: Tensor, x_t: Tensor, time_emb: Tensor
    ) -> Tensor:
        # time MLP (adaRMS conditioning surrogate), mirrors pi05 time_mlp path.
        t = self.time_mlp_out(torch.nn.functional.silu(self.time_mlp_in(time_emb)))
        t = torch.nn.functional.silu(t)  # [B, w]

        action_emb = self.action_in_proj(x_t)  # [B, H, w]
        cond = (self.state_proj(state) + t)[:, None, :]  # [B, 1, w]
        query = action_emb + cond

        memory = self.prefix_proj(prefix_memory)  # [B, Tp, w]
        attn_out, _ = self.cross_attn(query, memory, memory)
        h = self.norm(query + attn_out)
        h = self.norm(h + self.ffn(h))
        return self.action_out_proj(h)  # [B, H, action_dim]


class Pi05ReferenceModel(nn.Module):
    """Container holding the three reference stages."""

    def __init__(self, cfg: Pi05Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.vit = Pi05ViT(cfg)
        self.llm_prefix = Pi05Prefix(cfg)
        self.action_expert = Pi05ActionExpert(cfg)

    def time_embedding(self, timestep: Tensor) -> Tensor:
        return create_sinusoidal_pos_embedding(
            timestep, self.cfg.expert_width, min_period=4e-3, max_period=4.0
        )
