"""pi05 注意力 mask 单元测试。"""

from __future__ import annotations

import torch

from chameleon.models.pi05.attention import make_att_2d_masks


def test_make_att_2d_masks_prefix_lm() -> None:
    pad = torch.ones(1, 4, dtype=torch.bool)
    att = torch.tensor([[0, 0, 0, 0]], dtype=torch.bool)
    mask = make_att_2d_masks(pad, att)
    assert mask.shape == (1, 4, 4)
    assert bool(mask[0, 0, 0]) is True
