"""cosmos3 TRT runtime 适配 — 加载 host diffusers pipeline + 观测预处理。

TRT 只承担热点子图（vae_encode / dit / vae_decode）；tokenize、mRoPE 打包、scheduler、
velocity 掩码等 host 逻辑复用真实 ``Cosmos3OmniPipeline``。本模块提供：

- :func:`load_cosmos3_host_pipeline` — 用 Cosmos3Adapter 加载真实 diffusers pipeline
- :func:`build_conditioning_canvas` — 把观测整理成固定 profile 的条件画布张量
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from chameleon.config.schema import TaskConfig
from chameleon.deploy.cosmos3.shapes import Cosmos3Profile

logger = logging.getLogger(__name__)


def load_cosmos3_host_pipeline(task: TaskConfig, device: str) -> Any:
    """Load the real diffusers ``Cosmos3OmniPipeline`` for host-side preprocessing."""
    from chameleon.models.cosmos3.adapter import Cosmos3Adapter

    config = Cosmos3Adapter.make_config(task.model_overrides)
    config.use_reference = False
    adapter = Cosmos3Adapter(config).build(device)
    if not getattr(adapter, "_is_real_diffusers", False) or adapter.pipeline is None:
        raise RuntimeError(
            "cosmos3 TRT runtime requires a real Cosmos3OmniPipeline "
            "(use_reference=false + valid model_id/checkpoint)."
        )
    return adapter.pipeline


def _normalize_pixels(t: torch.Tensor, *, was_int: bool) -> torch.Tensor:
    """Map pixels to the ``[-1, 1]`` range the Cosmos Wan VAE expects.

    Heuristic (datasets vary): integer / ``max > 1.5`` → treat as ``[0, 255]``;
    non-negative float ``≤ 1.5`` → treat as ``[0, 1]``; anything already in
    ``[-1, 1]`` (has negatives) is left untouched.
    """
    mx = float(t.abs().max()) if t.numel() else 0.0
    mn = float(t.min()) if t.numel() else 0.0
    if was_int or mx > 1.5:
        return t / 127.5 - 1.0
    if mn >= -0.01:
        return t * 2.0 - 1.0
    return t


def _prep_image_frame(img: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Any single frame (HWC / CHW, uint8 / float) → normalized ``[3, H, W]`` tensor."""
    t = img if isinstance(img, torch.Tensor) else torch.as_tensor(_np_asarray(img))
    was_int = not torch.is_floating_point(t)
    t = t.to(device=device).float()
    if t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] != 3:  # HWC -> CHW
        t = t.permute(2, 0, 1)
    t = _normalize_pixels(t, was_int=was_int)
    return t.to(dtype=dtype)


def _np_asarray(x: Any):
    import numpy as np

    return np.asarray(x)


def build_conditioning_canvas(
    pipe: Any,
    profile: Cosmos3Profile,
    observation: dict[str, Any],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a conditioning canvas ``[1, 3, num_frames, canvas_h, canvas_w]``.

    Accepts ``observation['video']`` (canvas tensor ``[3,T,H,W]`` / ``[1,3,T,H,W]``,
    resized if needed) or ``observation['image']`` (single frame, HWC/CHW & uint8/
    float, repeat-padded), else a zero canvas (smoke). Pixels are normalized to
    ``[-1, 1]``. The output shape is locked to the fixed profile so it matches the
    ``vae_encode`` TRT engine input.
    """
    device = torch.device(device)
    target = (profile.num_frames, profile.canvas_h, profile.canvas_w)

    vid = observation.get("video")
    if vid is not None:
        v = vid if isinstance(vid, torch.Tensor) else torch.as_tensor(_np_asarray(vid))
        was_int = not torch.is_floating_point(v)
        v = v.to(device=device).float()
        if v.ndim == 4:  # [3, T, H, W]
            v = v.unsqueeze(0)
        v = _normalize_pixels(v, was_int=was_int).to(dtype=dtype)
        return _resize_canvas(v, target, device, dtype)

    img = observation.get("image")
    if img is not None:
        frame = _prep_image_frame(img, device, dtype)  # [3, H, W]
        frame = frame.unsqueeze(0)  # [1, 3, H, W]
        frame = F.interpolate(
            frame, size=(profile.canvas_h, profile.canvas_w), mode="bilinear", align_corners=False
        )
        canvas = frame.unsqueeze(2).expand(-1, -1, profile.num_frames, -1, -1).contiguous()
        return canvas.to(device=device, dtype=dtype)

    logger.warning("cosmos3 TRT: no observation video/image; using a zero canvas (smoke only).")
    return torch.zeros(1, 3, *target, device=device, dtype=dtype)


def _resize_canvas(
    video: torch.Tensor,
    target: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_frames, canvas_h, canvas_w = target
    b, c, t, h, w = video.shape
    if (t, h, w) == (num_frames, canvas_h, canvas_w):
        return video
    # Resize spatially per-frame then repeat/trim temporally to the profile length.
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    if (h, w) != (canvas_h, canvas_w):
        frames = F.interpolate(frames, size=(canvas_h, canvas_w), mode="bilinear", align_corners=False)
    frames = frames.reshape(b, t, c, canvas_h, canvas_w).permute(0, 2, 1, 3, 4)
    if t < num_frames:
        pad = frames[:, :, -1:].expand(-1, -1, num_frames - t, -1, -1)
        frames = torch.cat([frames, pad], dim=2)
    elif t > num_frames:
        frames = frames[:, :, :num_frames]
    return frames.contiguous().to(device=device, dtype=dtype)
