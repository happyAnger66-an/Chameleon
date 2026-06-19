"""观测图像 JPEG 编码（在 asyncio 消费侧执行，不占用推理线程）。"""

from __future__ import annotations

import base64
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * img).clip(0.0, 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def encode_jpeg_b64(rgb_hwc_uint8: np.ndarray, *, quality: int = 85) -> str:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("缺少 opencv-python-headless，无法编码 WebUI JPEG。")
    img = np.asarray(rgb_hwc_uint8)
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != np.uint8:
        raise ValueError(f"expect uint8(H,W,3), got {img.dtype} {img.shape}")
    bgr = img[..., ::-1]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_observation_images(
    observation: dict[str, Any] | None,
    *,
    send_wrist: bool,
    jpeg_quality: int,
) -> dict[str, str] | None:
    if not observation:
        return None
    image = observation.get("image")
    if image is None and "observation/image" in observation:
        image = observation["observation/image"]
    if image is None:
        return None
    try:
        images: dict[str, str] = {
            "base_rgb_jpeg_b64": encode_jpeg_b64(to_hwc_uint8(np.asarray(image)), quality=jpeg_quality),
        }
        wrist = observation.get("wrist_image")
        if wrist is None:
            wrist = observation.get("observation/wrist_image")
        if send_wrist and wrist is not None:
            images["wrist_rgb_jpeg_b64"] = encode_jpeg_b64(
                to_hwc_uint8(np.asarray(wrist)),
                quality=jpeg_quality,
            )
        return images
    except Exception as exc:
        logger.warning("图像编码失败（继续只发数值）: %s", exc)
        return None
