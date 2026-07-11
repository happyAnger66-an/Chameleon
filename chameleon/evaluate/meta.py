"""评测 run meta 构建 — api 与 WebUI server 共用。"""

from __future__ import annotations

from typing import Any

from chameleon.config.schema import TaskConfig
from chameleon.evaluate.trt_eval_utils import should_attach_tensorrt_meta, tensorrt_meta

_PT_TRT_COMPARE_RUNNERS = frozenset({"pt_trt_compare", "cosmos3_pt_trt_compare"})
_PT_TVM_COMPARE_RUNNERS = frozenset({"pt_tvm_compare"})
_DUAL_COMPARE_RUNNERS = _PT_TRT_COMPARE_RUNNERS | _PT_TVM_COMPARE_RUNNERS


def is_pt_trt_compare_runner(policy_runner: str | None) -> bool:
    return (policy_runner or "") in _PT_TRT_COMPARE_RUNNERS


def is_pt_tvm_compare_runner(policy_runner: str | None) -> bool:
    return (policy_runner or "") in _PT_TVM_COMPARE_RUNNERS


def is_dual_compare_runner(policy_runner: str | None) -> bool:
    return (policy_runner or "") in _DUAL_COMPARE_RUNNERS


def resolve_compare_mode(task: TaskConfig) -> bool:
    """WebUI 双路对比开关：显式 compare_mode 或双路 compare runner。"""
    ev = task.evaluate
    if bool(ev.compare_mode):
        return True
    return is_dual_compare_runner(ev.policy_runner)


def _compare_legend(policy_runner: str | None) -> tuple[str, str, str]:
    """返回 (pred1_name, pred2_name, pair_name)。"""
    if is_pt_tvm_compare_runner(policy_runner):
        return "PyTorch", "TVM", "PT−TVM"
    return "PyTorch", "TensorRT", "PT−TRT"


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
    pred1, pred2, pair = _compare_legend(task.evaluate.policy_runner)
    meta: dict[str, Any] = {
        "type": "meta",
        "run_id": run_id,
        "repo_id": repo_id,
        "backend": task.evaluate.policy_runner,
        "compare_mode": resolve_compare_mode(task),
        "pred1_name": pred1,
        "pred2_name": pred2,
        "pair_name": pair,
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
