"""Cosmos3 真实 diffusers 权重 — stage 级 stats 输入与模块准备。

Reference 路径使用 4D cond_pixels 与简化 dit；真实 ``Cosmos3OmniPipeline`` 的 Wan VAE
需要 ``[B, 3, T, H, W]``，MoT transformer 需要 pipeline 打包后的联合序列参数。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from chameleon.config.schema import TaskConfig
from chameleon.models.cosmos3.adapter import Cosmos3Adapter


class _WanVaeEncodeModule(nn.Module):
    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.mode()


class _WanVaeDecodeModule(nn.Module):
    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z).sample


class _Cosmos3TransformerStepModule(nn.Module):
    """单次 MoT denoise forward（参数在构造时固定，供 FLOPs 统计）。"""

    def __init__(self, transformer: nn.Module, kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.transformer = transformer
        self._kwargs = kwargs

    def forward(self) -> Any:
        return self.transformer(**self._kwargs)


def _resolve_dtype(adapter: Cosmos3Adapter) -> torch.dtype:
    if adapter.config.precision == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _video_profile(task: TaskConfig) -> dict[str, int | float | str]:
    gen = task.generate
    prompt = getattr(gen, "prompt", None) or "A test prompt for stats."
    return {
        "num_frames": int(getattr(gen, "num_frames", None) or task.model_overrides.get("num_frames", 189)),
        "height": int(getattr(gen, "height", None) or task.model_overrides.get("height", 720)),
        "width": int(getattr(gen, "width", None) or task.model_overrides.get("width", 1280)),
        "fps": float(getattr(gen, "fps", None) or task.model_overrides.get("fps", 24.0)),
        "prompt": str(prompt),
        "enable_sound": bool(getattr(gen, "enable_sound", False)),
    }


def _latent_grid(task: TaskConfig, pipe: Any) -> tuple[int, int, int, int]:
    profile = _video_profile(task)
    vae_cfg = pipe.vae.config
    t_scale = int(getattr(vae_cfg, "scale_factor_temporal", 4))
    s_scale = int(getattr(vae_cfg, "scale_factor_spatial", 16))
    latent_t = (int(profile["num_frames"]) - 1) // t_scale + 1
    latent_h = int(profile["height"]) // s_scale
    latent_w = int(profile["width"]) // s_scale
    z_dim = int(getattr(vae_cfg, "z_dim", 16))
    return z_dim, latent_t, latent_h, latent_w


def _tokenize_cond_ids(pipe: Any, task: TaskConfig) -> list[int]:
    profile = _video_profile(task)
    cond_ids, _ = pipe.tokenize_prompt(
        profile["prompt"],
        negative_prompt="",
        num_frames=int(profile["num_frames"]),
        height=int(profile["height"]),
        width=int(profile["width"]),
        fps=float(profile["fps"]),
    )
    return cond_ids


def _build_transformer_step_module(
    adapter: Cosmos3Adapter,
    task: TaskConfig,
    *,
    device: str,
) -> _Cosmos3TransformerStepModule:
    pipe = adapter.pipeline
    dtype = _resolve_dtype(adapter)
    profile = _video_profile(task)

    cond_ids = _tokenize_cond_ids(pipe, task)
    cond_text = pipe._prepare_text_segment(cond_ids, device)

    (
        latents,
        _sound_latents,
        _action_latents,
        fps_vision,
        _fps_sound,
        vision_condition_mask,
        _sound_condition_mask,
        _action_condition_mask,
        _action_domain_id,
        _action_image_size,
        _raw_action_dim,
        _action_condition_frame_indexes,
    ) = pipe.prepare_latents(
        num_frames=int(profile["num_frames"]),
        height=int(profile["height"]),
        width=int(profile["width"]),
        fps=float(profile["fps"]),
        device=device,
        dtype=dtype,
        enable_sound=bool(profile["enable_sound"]),
    )

    vision_condition_indexes = torch.nonzero(vision_condition_mask[:, 0, 0] > 0, as_tuple=False).flatten()
    vision_condition_indexes = [int(i.item()) for i in vision_condition_indexes]

    cond_vision = pipe._prepare_vision_segment(
        input_vision_tokens=latents,
        has_image_condition=bool(vision_condition_indexes),
        mrope_offset=cond_text["vision_start_temporal_offset"],
        vision_fps=fps_vision,
        curr=cond_text["und_len"],
        device=device,
        condition_frame_indexes=vision_condition_indexes,
    )

    packed = {
        **cond_text,
        **cond_vision,
        "position_ids": torch.cat(
            [cond_text["text_mrope_ids"], cond_vision["vision_mrope_ids"]],
            dim=1,
        ),
        "sequence_length": cond_text["und_len"] + cond_vision["num_vision_tokens"],
    }

    vision_timesteps = torch.full(
        (cond_vision["num_noisy_vision_tokens"],),
        500.0,
        device=device,
    )

    kwargs = {
        "input_ids": packed["input_ids"],
        "text_indexes": packed["text_indexes"],
        "position_ids": packed["position_ids"],
        "und_len": packed["und_len"],
        "sequence_length": packed["sequence_length"],
        "vision_tokens": [latents.to(device=device, dtype=dtype)],
        "vision_token_shapes": packed["vision_token_shapes"],
        "vision_sequence_indexes": packed["vision_sequence_indexes"],
        "vision_mse_loss_indexes": packed["vision_mse_loss_indexes"],
        "vision_timesteps": vision_timesteps,
        "vision_noisy_frame_indexes": packed["vision_noisy_frame_indexes"],
        "sound_tokens": None,
        "sound_token_shapes": None,
        "sound_sequence_indexes": None,
        "sound_mse_loss_indexes": None,
        "sound_timesteps": None,
        "sound_noisy_frame_indexes": None,
        "action_tokens": None,
        "action_token_shapes": None,
        "action_sequence_indexes": None,
        "action_mse_loss_indexes": None,
        "action_timesteps": None,
        "action_noisy_frame_indexes": None,
        "action_domain_ids": None,
    }
    return _Cosmos3TransformerStepModule(pipe.transformer, kwargs)


def prepare_real_cosmos3_stage(
    adapter: Cosmos3Adapter,
    stage: str,
    task: TaskConfig,
    *,
    device: str,
) -> tuple[nn.Module, tuple[Any, ...]]:
    if not getattr(adapter, "_is_real_diffusers", False) or adapter.pipeline is None:
        raise RuntimeError("prepare_real_cosmos3_stage requires a loaded Cosmos3OmniPipeline.")

    pipe = adapter.pipeline
    dtype = _resolve_dtype(adapter)
    profile = _video_profile(task)
    batch = task.infer.batch_size

    if stage == "vae_encode":
        module = _WanVaeEncodeModule(pipe.vae).eval()
        video = torch.randn(
            batch,
            3,
            int(profile["num_frames"]),
            int(profile["height"]),
            int(profile["width"]),
            device=device,
            dtype=dtype,
        )
        return module, (video,)

    if stage == "text_embed":
        module = pipe.transformer.embed_tokens.eval()
        cond_ids = _tokenize_cond_ids(pipe, task)
        input_ids = torch.tensor(cond_ids, dtype=torch.long, device=device)
        return module, (input_ids,)

    if stage == "dit":
        module = _build_transformer_step_module(adapter, task, device=device).eval()
        return module, ()

    if stage == "vae_decode":
        module = _WanVaeDecodeModule(pipe.vae).eval()
        z_dim, latent_t, latent_h, latent_w = _latent_grid(task, pipe)
        latent = torch.randn(
            batch,
            z_dim,
            latent_t,
            latent_h,
            latent_w,
            device=device,
            dtype=dtype,
        )
        return module, (latent,)

    raise KeyError(f"Unknown cosmos3 real stage {stage!r}.")
