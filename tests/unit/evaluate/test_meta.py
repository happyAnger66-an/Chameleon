"""evaluate meta 构建单元测试。"""

from __future__ import annotations

from chameleon.config.schema import EvaluateConfig, TaskConfig
from chameleon.evaluate.meta import build_eval_run_meta, resolve_compare_mode


def test_resolve_compare_mode_from_runner() -> None:
    task = TaskConfig()
    task.evaluate.policy_runner = "pt_trt_compare"
    task.evaluate.compare_mode = False
    assert resolve_compare_mode(task) is True


def test_resolve_compare_mode_explicit() -> None:
    task = TaskConfig()
    task.evaluate.policy_runner = "openpi"
    task.evaluate.compare_mode = True
    assert resolve_compare_mode(task) is True


def test_build_eval_run_meta_pt_trt_compare() -> None:
    task = TaskConfig()
    task.evaluate.policy_runner = "pt_trt_compare"
    meta = build_eval_run_meta(
        task,
        run_id="r1",
        repo_id="physical-intelligence/libero",
        action_horizon=10,
        action_dim=32,
        start_index=0,
        num_samples=5,
    )
    assert meta["compare_mode"] is True
    assert meta["backend"] == "pt_trt_compare"
