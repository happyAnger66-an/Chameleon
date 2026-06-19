"""counters 单元测试。"""

from __future__ import annotations

import torch
import torch.nn as nn

from chameleon.profile.counters import aggregate_stats, count_stage, estimate_attention_bytes


class TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 20)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_count_stage_linear_macs() -> None:
    module = TinyLinear().eval()
    x = torch.randn(2, 10)
    stats = count_stage(
        stage="tiny",
        repeat=1,
        module=module,
        inputs=(x,),
        shapes={},
        dtype_bytes=4,
        device="cpu",
    )
    assert stats.macs == 400
    assert stats.flops == 800
    assert stats.weight_bytes == 10 * 20 * 4 + 20 * 4  # weight + bias


def test_aggregate_stats_repeat() -> None:
    module = TinyLinear().eval()
    x = torch.randn(2, 10)
    s = count_stage(
        stage="tiny",
        repeat=3,
        module=module,
        inputs=(x,),
        shapes={},
        dtype_bytes=4,
        device="cpu",
    )
    totals = aggregate_stats([s])
    assert totals.macs == s.macs * 3
    assert totals.flops == s.flops * 3


def test_estimate_attention_bytes_with_past_keys() -> None:
    shapes = {
        "past_keys": (2, 1, 128, 64),
        "attention_mask": (1, 1, 10, 128),
    }
    b = estimate_attention_bytes(shapes, dtype_bytes=2)
    assert b > 0
