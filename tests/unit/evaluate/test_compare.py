"""compare_actions 单元测试。"""

from __future__ import annotations

import pytest
import torch

from chameleon.evaluate.compare import compare_actions


def test_identical_tensors_zero_diff() -> None:
    x = torch.ones(10, 7)
    diff = compare_actions(x, x)
    assert diff.max_abs == 0.0
    assert diff.mean_abs == 0.0
    assert diff.cosine == pytest.approx(1.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_actions(torch.zeros(10, 7), torch.zeros(5, 7))
