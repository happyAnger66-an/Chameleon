"""WebUI 协议 JSON 单元测试。"""

from __future__ import annotations

import json
import math

from chameleon.evaluate.viewers.webui.protocol import event_to_json, step_event_to_json, StepEvent


def test_event_to_json_sanitizes_nan() -> None:
    raw = event_to_json({"type": "step", "metrics": {"mse": float("nan"), "mae": float("inf")}})
    assert "NaN" not in raw
    parsed = json.loads(raw)
    assert parsed["metrics"]["mse"] is None
    assert parsed["metrics"]["mae"] is None


def test_step_event_roundtrip() -> None:
    ev = StepEvent(
        type="step",
        run_id="abc",
        episode_id=0,
        global_index=10,
        k_in_chunk=0,
        is_chunk_start=True,
        action_horizon=10,
        prompt="hi",
        gt_action=[0.1, 0.2],
        pred_action=[0.3, 0.4],
        metrics={"mae": 0.1, "mse": 0.01},
        images=None,
        server_timing={"infer_ms": 12.5},
    )
    parsed = json.loads(step_event_to_json(ev))
    assert parsed["type"] == "step"
    assert parsed["global_index"] == 10
    assert parsed["server_timing"]["infer_ms"] == 12.5
