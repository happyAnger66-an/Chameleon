"""Wan VAE encode/decode Export 模块 — 含数据集级 latent 归一化 / 反归一化。

reference 路径的 VAE surrogate 只做 conv；真实 ``AutoencoderKLWan`` 的 latent 在进入
MoT DiT 前须做 ``(mu - mean) * inv_std`` 归一化（``Cosmos3OmniPipeline._encode_video``），
解码前做 ``latent / inv_std + mean`` 反归一化。把这两步并入 encode/decode 子图，保证
TRT engine 的 I/O 与 pipeline 语义 bit 对齐（host 侧无需再算归一化）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _resolve_vae_norm_stats(vae: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(latents_mean, latents_inv_std)`` as 1-D tensors from the VAE config."""
    cfg = vae.config
    mean = torch.as_tensor(cfg.latents_mean, dtype=torch.float32)
    inv_std = 1.0 / torch.as_tensor(cfg.latents_std, dtype=torch.float32)
    return mean, inv_std


class WanVaeEncodeExport(nn.Module):
    """``video [B,3,T,H,W]`` → normalized latent ``[B,C,T_lat,H_lat,W_lat]``.

    Mirrors ``Cosmos3OmniPipeline._encode_video``: take the posterior mode
    (``argmax``) then apply dataset-level normalization. Normalization stats are
    registered as buffers so they are baked into the exported ONNX as constants.
    """

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae
        mean, inv_std = _resolve_vae_norm_stats(vae)
        self.register_buffer("latents_mean", mean.view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("latents_inv_std", inv_std.view(1, -1, 1, 1, 1), persistent=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        raw_mu = self.vae.encode(video).latent_dist.mode()
        mean = self.latents_mean.to(device=raw_mu.device, dtype=raw_mu.dtype)
        inv_std = self.latents_inv_std.to(device=raw_mu.device, dtype=raw_mu.dtype)
        return (raw_mu - mean) * inv_std


class WanVaeDecodeExport(nn.Module):
    """normalized latent ``[B,C,T_lat,H_lat,W_lat]`` → ``video [B,3,T,H,W]``.

    Inverse of :class:`WanVaeEncodeExport`: de-normalize with ``/ inv_std + mean``
    then run ``vae.decode`` (matches the ``output_type != 'latent'`` branch of
    ``Cosmos3OmniPipeline.__call__``).
    """

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae
        mean, inv_std = _resolve_vae_norm_stats(vae)
        self.register_buffer("latents_mean", mean.view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("latents_inv_std", inv_std.view(1, -1, 1, 1, 1), persistent=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(device=latent.device, dtype=latent.dtype)
        inv_std = self.latents_inv_std.to(device=latent.device, dtype=latent.dtype)
        z_raw = latent / inv_std + mean
        return self.vae.decode(z_raw).sample
