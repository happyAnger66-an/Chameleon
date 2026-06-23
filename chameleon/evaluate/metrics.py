"""单步评测 metrics — pred vs GT、PT vs TRT 双路对比（WebUI 协议对齐）。"""

from __future__ import annotations

from typing import Any

import numpy as np


def _per_dim_lists(flat: np.ndarray) -> tuple[list[float], list[float]]:
    n = int(flat.size)
    return (
        [float(np.abs(flat[i])) for i in range(n)],
        [float(flat[i] * flat[i]) for i in range(n)],
    )


def _mse_mae(diff: np.ndarray) -> tuple[float, float]:
    return float(np.mean(diff**2)), float(np.mean(np.abs(diff)))


def step_metrics(gt_row: np.ndarray, pred_row: np.ndarray) -> dict[str, Any]:
    """单步 pred vs ground-truth 指标（WebUI ``mae_pt`` / ``mse_pt`` 键）。"""
    diff = np.asarray(pred_row, dtype=np.float64) - np.asarray(gt_row, dtype=np.float64)
    flat = np.ravel(diff)
    mse, mae = _mse_mae(diff)
    mae_per_dim, mse_per_dim = _per_dim_lists(flat)
    return {
        "mse": mse,
        "mae": mae,
        "mse_pt": mse,
        "mae_pt": mae,
        "mae_per_dim": mae_per_dim,
        "mse_per_dim": mse_per_dim,
    }


def metrics_pt_vs_gt(diff_pt: np.ndarray) -> dict[str, Any]:
    """从 pred−gt 差分向量构造逐步指标（兼容双路对比基线键）。"""
    dpt_flat = np.ravel(diff_pt.astype(np.float64))
    mse_pt, mae_pt = _mse_mae(diff_pt)
    mae_per_dim, mse_per_dim = _per_dim_lists(dpt_flat)
    return {
        "mse": mse_pt,
        "mae": mae_pt,
        "mse_pt": mse_pt,
        "mae_pt": mae_pt,
        "mae_per_dim": mae_per_dim,
        "mse_per_dim": mse_per_dim,
    }


def attach_pt_trt_pair_metrics(
    metrics: dict[str, Any],
    *,
    pred_pt_row: np.ndarray,
    pred_trt_row: np.ndarray,
    gt_row: np.ndarray,
) -> dict[str, Any]:
    """追加 TRT vs GT 与 PT vs TRT 逐步指标（``mae_pt_trt*`` 键供 WebUI 使用）。"""
    diff_trt = pred_trt_row - gt_row
    diff_pair = pred_pt_row - pred_trt_row
    mse_trt, mae_trt = _mse_mae(diff_trt)
    mse_pair, mae_pair = _mse_mae(diff_pair)
    dpair_flat = np.ravel(diff_pair.astype(np.float64))
    mae_pair_per_dim, mse_pair_per_dim = _per_dim_lists(dpair_flat)
    out = dict(metrics)
    out["mse_trt"] = mse_trt
    out["mae_trt"] = mae_trt
    out["mse_pt_trt"] = mse_pair
    out["mae_pt_trt"] = mae_pair
    out["mae_pt_trt_per_dim"] = mae_pair_per_dim
    out["mse_pt_trt_per_dim"] = mse_pair_per_dim
    return out
