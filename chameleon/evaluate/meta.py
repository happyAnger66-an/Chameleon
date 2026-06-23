"""评测 run meta 构建 — api 与 WebUI server 共用。"""

from __future__ import annotations

from typing import Any

from chameleon.config.schema import TaskConfig
from chameleon.evaluate.trt_eval_utils import should_attach_tensorrt_meta, tensorrt_meta


def build_eval_run_meta(
    task: TaskConfig,
    *,
    run_id: str,
    repo_id: str,
    action_horizon: int,
    action_dim: int,
    start_index: int,
    num_samples: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 ``on_run_start`` / WebUI handshake 使用的 meta 字典。"""
    meta: dict[str, Any] = {
        "type": "meta",
        "run_id": run_id,
        "repo_id": repo_id,
        "backend": task.evaluate.policy_runner,
        "compare_mode": bool(task.evaluate.compare_mode),
        "pred1_name": "PyTorch",
        "pred2_name": "TensorRT",
        "pair_name": "PT−TRT",
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "start_index": start_index,
        "end_index_exclusive": start_index + num_samples,
    }
    if should_attach_tensorrt_meta(task):
        meta["tensorrt"] = tensorrt_meta(task)
    if extra:
        meta.update(extra)
    return meta
