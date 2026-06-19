"""计算量与访存量的可读单位换算。"""

from __future__ import annotations

from typing import Any, Literal

OpKind = Literal["MAC", "FLOP"]
_BYTE_UNITS: tuple[tuple[float, str], ...] = (
    (1e12, "TB"),
    (1e9, "GB"),
    (1e6, "MB"),
    (1e3, "KB"),
)
_OP_UNITS: tuple[tuple[float, str], ...] = (
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "K"),
)


def _scale_value(raw: int | float, units: tuple[tuple[float, str], ...]) -> tuple[float, str]:
    value = float(raw)
    if value == 0:
        return 0.0, units[-1][1] if units is _BYTE_UNITS else units[-1][1]
    for scale, unit in units:
        if abs(value) >= scale:
            return value / scale, unit
    suffix = units[-1][1]
    return value / units[-1][0], suffix


def format_bytes(raw: int) -> dict[str, Any]:
    """将字节数格式化为 GB/TB 等可读字段（>=1MB 时用 GB 小数，便于阅读）。"""
    value = float(raw)
    if value >= 1e12:
        scaled, unit = value / 1e12, "TB"
    elif value >= 1e6:
        scaled, unit = value / 1e9, "GB"
    elif value >= 1e3:
        scaled, unit = value / 1e6, "MB"
    else:
        scaled, unit = value, "B"
    display = f"{scaled:.3f} {unit}"
    return {
        "raw": int(raw),
        "value": round(scaled, 6),
        "unit": unit,
        "display": display,
    }


def format_ops(raw: int, *, kind: OpKind) -> dict[str, Any]:
    """将 MAC/FLOP 计数格式化为 GMAC/TFLOP 等可读字段。"""
    value, prefix = _scale_value(raw, _OP_UNITS)
    unit = f"{prefix}{kind}"
    plural = f"{prefix}{kind}s"
    display = f"{value:.3f} {plural}"
    return {
        "raw": int(raw),
        "value": round(value, 6),
        "unit": unit,
        "display": display,
    }


def pick_ops_column_unit(max_raw: int) -> tuple[float, str, str]:
    """为表格列选择统一的 MAC/FLOP 单位（优先 TFLOPs / GFLOPs）。"""
    if max_raw >= 1e12:
        return 1e12, "T", "TFLOPs"
    if max_raw >= 1e9:
        return 1e9, "G", "GFLOPs"
    if max_raw >= 1e6:
        return 1e6, "M", "MFLOPs"
    return 1e3, "K", "KFLOPs"


def pick_bytes_column_unit(max_raw: int) -> tuple[float, str]:
    """为表格列选择统一的字节单位（优先 TB / GB）。"""
    if max_raw >= 1e12:
        return 1e12, "TB"
    return 1e9, "GB"
