"""pi05 注意力 mask 工具 — deploy 与 runtime TRT 编排共用。"""

from __future__ import annotations

import torch


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """由 pad / attention 1D mask 构造 2D 布尔 attention mask。"""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks
