"""cosmos3 conditioning canvas 归一化与 shape 单测。"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chameleon.deploy.cosmos3.shapes import get_profile
from chameleon.runtime.cosmos3_trt.adapter import build_conditioning_canvas


def _profile():
    return get_profile("policy_droid")


def test_canvas_from_hwc_uint8_image_shape_and_range() -> None:
    profile = _profile()
    img = np.random.randint(0, 256, size=(180, 320, 3), dtype=np.uint8)
    canvas = build_conditioning_canvas(
        None, profile, {"image": img}, device="cpu", dtype=torch.float32
    )
    assert tuple(canvas.shape) == (1, 3, profile.num_frames, profile.canvas_h, profile.canvas_w)
    assert float(canvas.min()) >= -1.0001
    assert float(canvas.max()) <= 1.0001


def test_canvas_repeats_single_frame_across_time() -> None:
    profile = _profile()
    img = np.full((16, 16, 3), 255, dtype=np.uint8)
    canvas = build_conditioning_canvas(
        None, profile, {"image": img}, device="cpu", dtype=torch.float32
    )
    # 单帧 repeat：各时间步相同
    first = canvas[:, :, 0]
    last = canvas[:, :, -1]
    assert torch.allclose(first, last)
    # 全白 255 -> 归一化到 +1
    np.testing.assert_allclose(float(canvas.max()), 1.0, atol=1e-3)


def test_canvas_from_chw_float_video() -> None:
    profile = _profile()
    # [3, T, H, W] float in [0,1]
    vid = torch.rand(3, 5, 60, 80)
    canvas = build_conditioning_canvas(
        None, profile, {"video": vid}, device="cpu", dtype=torch.float32
    )
    assert tuple(canvas.shape) == (1, 3, profile.num_frames, profile.canvas_h, profile.canvas_w)
    assert float(canvas.min()) >= -1.0001
    assert float(canvas.max()) <= 1.0001


def test_canvas_zero_fallback_when_empty() -> None:
    profile = _profile()
    canvas = build_conditioning_canvas(None, profile, {}, device="cpu", dtype=torch.float32)
    assert tuple(canvas.shape) == (1, 3, profile.num_frames, profile.canvas_h, profile.canvas_w)
    assert float(canvas.abs().max()) == 0.0
