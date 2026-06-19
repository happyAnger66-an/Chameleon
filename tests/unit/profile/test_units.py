"""units 单元测试。"""

from __future__ import annotations

from chameleon.profile.units import format_bytes, format_ops


def test_format_ops_gflops() -> None:
    out = format_ops(636_068_560_896, kind="FLOP")
    assert out["raw"] == 636_068_560_896
    assert out["unit"] == "GFLOP"
    assert abs(out["value"] - 636.068561) < 0.01
    assert "636.069 GFLOPs" in out["display"]


def test_format_ops_tflops() -> None:
    out = format_ops(2_500_000_000_000, kind="FLOP")
    assert out["unit"] == "TFLOP"
    assert abs(out["value"] - 2.5) < 0.001
    assert "2.500 TFLOPs" in out["display"]


def test_format_bytes_gb() -> None:
    out = format_bytes(834_559_456)
    assert out["unit"] == "GB"
    assert abs(out["value"] - 0.834559) < 0.001
    assert "0.835 GB" in out["display"]


def test_format_bytes_tb() -> None:
    out = format_bytes(3_000_000_000_000)
    assert out["unit"] == "TB"
    assert abs(out["value"] - 3.0) < 0.001
