"""WebSocket 事件 JSON 序列化（协议 v1，兼容 model_optimizer webui_client）。"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Literal


@dataclasses.dataclass(frozen=True)
class StepEvent:
    type: Literal["step"]
    run_id: str
    episode_id: int
    global_index: int
    k_in_chunk: int
    is_chunk_start: bool
    action_horizon: int
    prompt: str | None
    gt_action: list[float]
    pred_action: list[float]
    metrics: dict[str, Any]
    images: dict[str, str] | None
    server_timing: dict[str, float] | None
    pred_action_trt: list[float] | None = None
    pred_action_ptq: list[float] | None = None


def _sanitize_json_value(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_value(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json_value(v) for v in obj]
    return obj


def event_to_json(event: dict[str, Any]) -> str:
    clean = _sanitize_json_value(event)
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def step_event_to_json(event: StepEvent) -> str:
    return event_to_json(dataclasses.asdict(event))
