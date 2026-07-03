"""Cosmos3 轻量参考模型 — 无外部权重依赖的四阶段 MoT 实现。

作用：
    用小型随机初始化层复现 diffusers ``Cosmos3OmniTransformer`` 的关键结构与
    算法（VAE 条件编码 surrogate + 文本 embedding + MoT 双路径去噪步 +
    flow-matching velocity 头 + VAE 解码 surrogate），使全链路（action / video
    两模式）可在 CPU 上端到端运行。定义 Cosmos3Config 及四个 stage 模块。

架构位置：
    模型/架构层 — 被 Cosmos3Adapter（use_reference=True）与
    Cosmos3ReferenceOrchestrator 使用。与真实 diffusers 权重路径共享相同 stage
    接口（vae_encode / text_embed / dit / vae_decode）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class Cosmos3Config:
    """Configuration for the cosmos3 adapter.

    Hidden widths are kept small for the reference path so the MVP runs quickly
    on CPU; set ``use_reference=False`` with a ``checkpoint`` / ``model_id`` to
    load the real diffusers model.
    """

    # --- generation-mode externally visible quantities ---
    mode: str = "video"
    """``video`` | ``action`` — 默认生成模式（也可被 task.generate.mode 覆盖）。"""

    action_dim: int = 32
    action_horizon: int = 16
    num_denoise_steps: int = 35
    guidance_scale: float = 6.0

    # --- reference backbone widths (small for CPU smoke) ---
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4

    # video latent grid (tokens = latent_t * latent_h * latent_w); each token
    # decodes to a patch_size x patch_size RGB patch.
    latent_t: int = 2
    latent_h: int = 4
    latent_w: int = 4
    patch_size: int = 8
    out_channels: int = 3

    # conditioning-image surrogate input.
    image_channels: int = 3
    image_size: int = 64

    # text.
    vocab_size: int = 1024
    max_lang_len: int = 16

    # --- model source selection ---
    use_reference: bool = True
    checkpoint: str | None = None
    model_id: str = "nvidia/Cosmos3-Nano"
    precision: str = "bfloat16"
    # diffusers Cosmos3OmniPipeline enables cosmos_guardrail by default; disable for
    # local stats/infer unless ``pip install cosmos_guardrail`` is available.
    enable_safety_checker: bool = False

    @property
    def num_video_tokens(self) -> int:
        return self.latent_t * self.latent_h * self.latent_w

    @property
    def token_dim(self) -> int:
        """Per-token gen feature dim. Unified to ``action_dim`` so a single
        ``proj_in`` / ``proj_out`` serves both video latents and action tokens."""
        return self.action_dim


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float = 4e-3, max_period: float = 4.0
) -> Tensor:
    """Sine-cosine positional embedding for scalar diffusion timesteps.

    Mirrors the time embedding used by flow-matching policies (see pi05
    ``create_sinusoidal_pos_embedding``).
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    device = time.device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


class SimpleMHA(nn.Module):
    """Multi-head attention built from plain linears + matmul softmax.

    Avoids ``nn.MultiheadAttention`` whose fused kernel
    (``aten::_native_multi_head_attention``) has no ONNX export path.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = self._split(self.q_proj(query))  # [B, H, Nq, hd]
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)  # [B, H, Nq, hd]
        b, h, n, hd = ctx.shape
        ctx = ctx.transpose(1, 2).reshape(b, n, h * hd)
        return self.out_proj(ctx)


class Cosmos3VaeEncode(nn.Module):
    """Stage ``vae_encode``: conditioning image -> latent tokens (Wan VAE surrogate)."""

    def __init__(self, cfg: Cosmos3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(cfg.image_channels, cfg.hidden_size, kernel_size=16, stride=16)
        # Number of conv patches for a fixed-size conditioning image.
        side = cfg.image_size // 16
        self._num_patches = side * side
        # Fixed token mixer maps conv patches -> num_video_tokens (ONNX-friendly matmul,
        # avoids adaptive pooling which cannot up-sample under ONNX export).
        self.token_mix = nn.Linear(self._num_patches, cfg.num_video_tokens)
        self.proj = nn.Linear(cfg.hidden_size, cfg.token_dim)

    def forward(self, cond_pixels: Tensor) -> Tensor:
        # cond_pixels: [B, 3, H, W] -> cond_latent: [B, num_video_tokens, token_dim]
        x = self.patch(cond_pixels)  # [B, hidden, h, w]
        x = x.flatten(2)  # [B, hidden, num_patches]
        x = self.token_mix(x)  # [B, hidden, num_video_tokens]
        x = x.transpose(1, 2)  # [B, num_video_tokens, hidden]
        return self.proj(x)


class Cosmos3TextEmbed(nn.Module):
    """Stage ``text_embed``: language tokens -> text memory (understanding prefix surrogate)."""

    def __init__(self, cfg: Cosmos3Config) -> None:
        super().__init__()
        self.cfg = cfg
        w = cfg.hidden_size
        self.embed = nn.Embedding(cfg.vocab_size, w)
        self.attn = SimpleMHA(w, cfg.num_heads)
        self.ffn = nn.Sequential(nn.Linear(w, w * 2), nn.GELU(), nn.Linear(w * 2, w))
        self.norm1 = nn.LayerNorm(w)
        self.norm2 = nn.LayerNorm(w)

    def forward(self, lang_tokens: Tensor) -> Tensor:
        emb = self.embed(lang_tokens) * math.sqrt(self.cfg.hidden_size)
        attn_out = self.attn(emb, emb, emb)
        h = self.norm1(emb + attn_out)
        return self.norm2(h + self.ffn(h))


class Cosmos3Dit(nn.Module):
    """Stage ``dit``: one MoT flow-matching denoise step.

    Mirrors ``Cosmos3OmniTransformer`` dual-pathway structure at small scale: the
    text (und) memory and conditioning latents form the cross-attention context,
    and the noisy generation tokens (gen) bi-directionally attend over the full
    (und + gen) key/value set, predicting the flow velocity ``v_t``.
    """

    def __init__(self, cfg: Cosmos3Config) -> None:
        super().__init__()
        self.cfg = cfg
        w = cfg.hidden_size
        self.proj_in = nn.Linear(cfg.token_dim, w)
        self.cond_proj = nn.Linear(cfg.token_dim, w)
        self.time_in = nn.Linear(w, w)
        self.time_out = nn.Linear(w, w)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": SimpleMHA(w, cfg.num_heads),
                        "ffn": nn.Sequential(nn.Linear(w, w * 2), nn.SiLU(), nn.Linear(w * 2, w)),
                        "norm1": nn.LayerNorm(w),
                        "norm2": nn.LayerNorm(w),
                    }
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.proj_out = nn.Linear(w, cfg.token_dim)

    def forward(
        self, text_mem: Tensor, cond_latent: Tensor, x_t: Tensor, time_emb: Tensor
    ) -> Tensor:
        # time conditioning (adaRMS surrogate).
        t = F.silu(self.time_out(F.silu(self.time_in(time_emb))))  # [B, w]
        gen = self.proj_in(x_t) + t[:, None, :]  # [B, N, w]
        memory = torch.cat([text_mem, self.cond_proj(cond_latent)], dim=1)  # [B, L+Nv, w]

        h = gen
        for layer in self.layers:
            kv = torch.cat([memory, h], dim=1)  # gen attends to (und + gen)
            attn_out = layer["attn"](h, kv, kv)
            h = layer["norm1"](h + attn_out)
            h = layer["norm2"](h + layer["ffn"](h))
        return self.proj_out(h)  # [B, N, token_dim] velocity


class Cosmos3VaeDecode(nn.Module):
    """Stage ``vae_decode``: video latent tokens -> video frames (Wan VAE surrogate)."""

    def __init__(self, cfg: Cosmos3Config) -> None:
        super().__init__()
        self.cfg = cfg
        p = cfg.patch_size
        self.to_pixels = nn.Linear(cfg.token_dim, cfg.out_channels * p * p)

    def forward(self, latent: Tensor) -> Tensor:
        # latent: [B, num_video_tokens, token_dim] -> video: [B, T, C, H, W]
        cfg = self.cfg
        b = latent.shape[0]
        p = cfg.patch_size
        pixels = self.to_pixels(latent)  # [B, N, C*p*p]
        pixels = pixels.reshape(
            b, cfg.latent_t, cfg.latent_h, cfg.latent_w, cfg.out_channels, p, p
        )
        # [B, T, C, lh, p, lw, p] -> [B, T, C, lh*p, lw*p]
        pixels = pixels.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        return pixels.reshape(
            b, cfg.latent_t, cfg.out_channels, cfg.latent_h * p, cfg.latent_w * p
        )


class Cosmos3ReferenceModel(nn.Module):
    """Container holding the four reference stages."""

    def __init__(self, cfg: Cosmos3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.vae_encode = Cosmos3VaeEncode(cfg)
        self.text_embed = Cosmos3TextEmbed(cfg)
        self.dit = Cosmos3Dit(cfg)
        self.vae_decode = Cosmos3VaeDecode(cfg)

    def time_embedding(self, timestep: Tensor) -> Tensor:
        return create_sinusoidal_pos_embedding(timestep, self.cfg.hidden_size)
