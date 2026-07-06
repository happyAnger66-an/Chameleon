"""pi05 TRT pipeline — static prefix padding."""

from __future__ import annotations

import pytest
import torch

from chameleon.runtime.pi05_trt.pipeline import _pad_prefix_to_static_len


def test_pad_prefix_noop_when_lengths_match() -> None:
    embs = torch.randn(1, 4, 8)
    pad = torch.ones(1, 4, dtype=torch.bool)
    att = torch.zeros(1, 4, dtype=torch.bool)
    out = _pad_prefix_to_static_len(embs, pad, att, 4)
    assert out[0].shape == (1, 4, 8)
    assert torch.equal(out[1], pad)
    assert torch.equal(out[2], att)


def test_pad_prefix_extends_to_target_len() -> None:
    embs = torch.randn(1, 3, 8)
    pad = torch.tensor([[True, True, False]])
    att = torch.zeros(1, 3, dtype=torch.bool)
    out_embs, out_pad, out_att = _pad_prefix_to_static_len(embs, pad, att, 5)
    assert out_embs.shape == (1, 5, 8)
    assert out_pad.tolist() == [[True, True, False, False, False]]
    assert out_att.shape == (1, 5)
    assert out_embs[0, 3:].abs().max() == 0.0


def test_pad_prefix_raises_when_too_long() -> None:
    embs = torch.randn(1, 6, 8)
    pad = torch.ones(1, 6, dtype=torch.bool)
    att = torch.zeros(1, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="exceeds TRT static target"):
        _pad_prefix_to_static_len(embs, pad, att, 5)
