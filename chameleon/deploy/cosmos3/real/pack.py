"""Host 侧联合序列静态打包 — policy（text + vision + action）。

MoT dit 的 forward 参数里，**只有** 逐 step 变化的 4 个张量是动态的
（``vision_tokens`` / ``vision_timesteps`` / ``action_tokens`` / ``action_timesteps``），
其余（``input_ids`` / ``position_ids`` / ``*_indexes`` / ``token_shapes`` /
``noisy_frame_indexes`` / ``action_domain_ids`` 等）在整个去噪环内恒定。

本模块用 pipeline 的 ``_prepare_text_segment`` / ``_prepare_vision_segment`` /
``_prepare_action_segment`` 复现 ``Cosmos3OmniPipeline.__call__`` 环前打包（policy /
guidance=1，仅 cond 通路），产出：

- :class:`Cosmos3PolicyPack.static` — 传给 :class:`Cosmos3DitStepExport` 的静态字段
- :class:`Cosmos3PolicyPack.dynamic_example` — 供 ONNX 导出的动态样例张量

这些字段只依赖 shape / mRoPE，不触碰 VAE 或视频预处理，因此可脱离真实观测确定性构造，
保证 export / build / runtime 三处 shape 完全一致（固定 profile）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from chameleon.deploy.cosmos3.shapes import Cosmos3Profile

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PROMPT = "Pick up the object and place it at the target location."


def _domain_id(domain_name: str) -> int:
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_DOMAIN_ID

    if domain_name not in _EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown cosmos3 action domain_name={domain_name!r}; "
            f"expected one of {sorted(_EMBODIMENT_TO_DOMAIN_ID)}."
        )
    return int(_EMBODIMENT_TO_DOMAIN_ID[domain_name])


def _fixed_length_ids(pipe: Any, profile: Cosmos3Profile, device: torch.device | str) -> list[int]:
    """Tokenize a representative policy caption, then pad/truncate to ``text_prefix_len``."""
    cond_ids, _uncond_ids = pipe.tokenize_prompt(
        _DEFAULT_POLICY_PROMPT,
        negative_prompt="",
        num_frames=profile.num_frames,
        height=profile.canvas_h,
        width=profile.canvas_w,
        fps=profile.fps,
        action_mode=profile.action_mode,
        action_view_point="ego_view",
    )
    target = profile.text_prefix_len
    if len(cond_ids) >= target:
        return list(cond_ids[:target])
    tok = getattr(pipe, "tokenizer", None)
    pad_id = int(getattr(tok, "pad_token_id", None) or 0) if tok is not None else 0
    return list(cond_ids) + [pad_id] * (target - len(cond_ids))


@dataclass
class Cosmos3PolicyPack:
    """静态打包字段 + 导出用动态样例张量（policy / guidance=1）。"""

    static: dict[str, Any]
    dynamic_example: dict[str, torch.Tensor]
    domain_id: torch.Tensor
    raw_action_dim: int
    action_condition_mask: torch.Tensor
    vision_condition_mask: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)


def build_policy_pack(
    pipe: Any,
    profile: Cosmos3Profile,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Cosmos3PolicyPack:
    """Build the static joint-sequence pack + example dynamic tensors for a policy profile.

    Reproduces the cond-pass pre-loop packing of ``Cosmos3OmniPipeline.__call__`` using
    synthetic latent shapes (full tier canvas, no padding removal) so that all shapes are
    deterministic and match the fixed TRT profile.
    """
    device = torch.device(device)
    cfg = pipe.transformer.config

    # Reconcile documented profile defaults against the real checkpoint config.
    latent_channels = int(getattr(pipe.vae.config, "z_dim", profile.latent_channels))
    scale_t = int(getattr(pipe.vae.config, "scale_factor_temporal", profile.scale_factor_temporal))
    scale_s = int(getattr(pipe.vae.config, "scale_factor_spatial", profile.scale_factor_spatial))
    if (latent_channels, scale_t, scale_s) != (
        profile.latent_channels,
        profile.scale_factor_temporal,
        profile.scale_factor_spatial,
    ):
        logger.warning(
            "cosmos3 profile %s VAE dims differ from checkpoint: profile(C=%d,t=%d,s=%d) vs "
            "checkpoint(C=%d,t=%d,s=%d); using checkpoint values for the pack.",
            profile.name,
            profile.latent_channels,
            profile.scale_factor_temporal,
            profile.scale_factor_spatial,
            latent_channels,
            scale_t,
            scale_s,
        )

    latent_t = (profile.num_frames - 1) // scale_t + 1
    latent_h = profile.canvas_h // scale_s
    latent_w = profile.canvas_w // scale_s
    action_dim = int(getattr(pipe.transformer, "action_dim", None) or cfg.action_dim)

    # --- text segment (fixed length) ---
    input_ids_list = _fixed_length_ids(pipe, profile, device)
    text_seg = pipe._prepare_text_segment(input_ids_list, device=device)

    # --- vision segment (synthetic latent grid; policy locks latent frame 0 clean) ---
    synth_latents = torch.zeros(
        1, latent_channels, latent_t, latent_h, latent_w, device=device, dtype=dtype
    )
    vision_seg = pipe._prepare_vision_segment(
        input_vision_tokens=synth_latents,
        has_image_condition=True,
        mrope_offset=text_seg["vision_start_temporal_offset"],
        vision_fps=profile.fps,
        curr=text_seg["und_len"],
        device=device,
        condition_frame_indexes=[0],
    )

    # --- action segment (chunk_size tokens, all noisy for policy) ---
    synth_action = torch.zeros(profile.chunk_size, action_dim, device=device, dtype=dtype)
    action_seg = pipe._prepare_action_segment(
        input_action_tokens=synth_action,
        condition_frame_indexes=[],
        mrope_offset=text_seg["vision_start_temporal_offset"],
        action_fps=profile.fps,
        curr=text_seg["und_len"] + vision_seg["num_vision_tokens"],
        device=device,
    )

    position_ids = torch.cat(
        [text_seg["text_mrope_ids"], vision_seg["vision_mrope_ids"], action_seg["action_mrope_ids"]],
        dim=1,
    )
    sequence_length = (
        text_seg["und_len"] + vision_seg["num_vision_tokens"] + action_seg["action_len"]
    )

    domain_id = torch.tensor([_domain_id(profile.domain_name)], dtype=torch.long, device=device)

    # Condition masks (host applies velocity masking after dit — see _mask_velocity_predictions).
    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)
    vision_condition_mask[0, 0, 0] = 1.0  # policy: latent frame 0 (observation) clean
    action_condition_mask = torch.zeros((profile.chunk_size, 1), device=device, dtype=dtype)

    static = {
        "input_ids": text_seg["input_ids"],
        "text_indexes": text_seg["text_indexes"],
        "position_ids": position_ids,
        "und_len": int(text_seg["und_len"]),
        "sequence_length": int(sequence_length),
        "vision_token_shapes": vision_seg["vision_token_shapes"],
        "vision_sequence_indexes": vision_seg["vision_sequence_indexes"],
        "vision_mse_loss_indexes": vision_seg["vision_mse_loss_indexes"],
        "vision_noisy_frame_indexes": vision_seg["vision_noisy_frame_indexes"],
        "action_token_shapes": action_seg["action_token_shapes"],
        "action_sequence_indexes": action_seg["action_sequence_indexes"],
        "action_mse_loss_indexes": action_seg["action_mse_loss_indexes"],
        "action_noisy_frame_indexes": action_seg["action_noisy_frame_indexes"],
        "action_domain_ids": domain_id,
    }

    dynamic_example = {
        "vision_tokens": synth_latents.clone(),
        "vision_timesteps": torch.full(
            (vision_seg["num_noisy_vision_tokens"],), 500.0, device=device, dtype=torch.float32
        ),
        "action_tokens": synth_action.clone(),
        "action_timesteps": torch.full(
            (action_seg["num_noisy_action_tokens"],), 500.0, device=device, dtype=torch.float32
        ),
    }

    meta = {
        "latent_shape": (1, latent_channels, latent_t, latent_h, latent_w),
        "action_shape": (profile.chunk_size, action_dim),
        "num_vision_tokens": int(vision_seg["num_vision_tokens"]),
        "num_noisy_vision_tokens": int(vision_seg["num_noisy_vision_tokens"]),
        "num_noisy_action_tokens": int(action_seg["num_noisy_action_tokens"]),
        "sequence_length": int(sequence_length),
        "und_len": int(text_seg["und_len"]),
    }
    logger.info("Built cosmos3 policy pack (%s): %s", profile.name, meta)

    return Cosmos3PolicyPack(
        static=static,
        dynamic_example=dynamic_example,
        domain_id=domain_id,
        raw_action_dim=profile.raw_action_dim,
        action_condition_mask=action_condition_mask,
        vision_condition_mask=vision_condition_mask,
        meta=meta,
    )
