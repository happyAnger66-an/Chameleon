"""lerobot_eval 对齐与 step 编码单元测试。"""

from __future__ import annotations

import json

import numpy as np

from chameleon.evaluate.lerobot_eval import _align_horizon
from chameleon.evaluate.viewers.base import EvalStepEvent
from chameleon.evaluate.viewers.webui.protocol import StepEvent, step_event_to_json


def test_align_horizon_squeezes_batch_dim() -> None:
    pred = np.ones((1, 10, 32), dtype=np.float32)
    gt = np.zeros((10, 32), dtype=np.float32)
    pred_a, gt_a = _align_horizon(pred, gt, None)
    assert pred_a.shape == (10, 32)
    assert gt_a.shape == (10, 32)


def test_step_json_includes_pred_action_trt() -> None:
    step = StepEvent(
        type="step",
        run_id="r1",
        episode_id=0,
        global_index=0,
        k_in_chunk=0,
        is_chunk_start=True,
        action_horizon=10,
        prompt="p",
        gt_action=[0.0, 1.0],
        pred_action=[0.1, 1.1],
        metrics={"mae_pt_trt": 0.01},
        images=None,
        server_timing={"infer_ms": 12.5},
        pred_action_trt=[0.2, 1.2],
    )
    payload = json.loads(step_event_to_json(step))
    assert payload["pred_action_trt"] == [0.2, 1.2]
    assert payload["metrics"]["mae_pt_trt"] == 0.01


def test_step_json_nan_becomes_null() -> None:
    ev = EvalStepEvent(
        run_id="r1",
        episode_id=0,
        global_index=0,
        k_in_chunk=0,
        is_chunk_start=True,
        action_horizon=1,
        prompt=None,
        gt_action=[0.0],
        pred_action=[0.1],
        metrics={},
        pred_action_trt=[float("nan")],
    )
    step = StepEvent(
        type="step",
        run_id=ev.run_id,
        episode_id=ev.episode_id,
        global_index=ev.global_index,
        k_in_chunk=ev.k_in_chunk,
        is_chunk_start=ev.is_chunk_start,
        action_horizon=ev.action_horizon,
        prompt=ev.prompt,
        gt_action=ev.gt_action,
        pred_action=ev.pred_action,
        metrics=ev.metrics,
        images=None,
        server_timing=None,
        pred_action_trt=ev.pred_action_trt,
    )
    payload = json.loads(step_event_to_json(step))
    assert payload["pred_action_trt"] == [None]
