"""metrics 单元测试。"""

from __future__ import annotations

import numpy as np

from chameleon.evaluate.metrics import attach_pt_trt_pair_metrics, metrics_pt_vs_gt, step_metrics


def test_metrics_pt_vs_gt_identical() -> None:
    gt = np.array([1.0, 2.0, 3.0])
    pred = gt.copy()
    m = metrics_pt_vs_gt(pred - gt)
    assert m["mae"] == 0.0
    assert m["mse"] == 0.0


def test_step_metrics_matches_pt_vs_gt() -> None:
    gt = np.array([0.0, 1.0])
    pred = np.array([0.1, 1.2])
    m1 = step_metrics(gt, pred)
    m2 = metrics_pt_vs_gt(pred - gt)
    assert m1["mae"] == m2["mae"]
    assert m1["mse"] == m2["mse"]


def test_attach_pt_trt_pair_metrics() -> None:
    gt = np.zeros(3)
    pred_pt = np.array([1.0, 0.0, 0.0])
    pred_trt = np.array([1.1, 0.0, 0.0])
    base = metrics_pt_vs_gt(pred_pt - gt)
    out = attach_pt_trt_pair_metrics(
        base, pred_pt_row=pred_pt, pred_trt_row=pred_trt, gt_row=gt
    )
    assert "mae_pt_trt" in out
    assert out["mae_pt_trt"] > 0.0
