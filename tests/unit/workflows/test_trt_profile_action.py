"""workflow trt_profile action 单元测试。"""

from __future__ import annotations

import logging

from chameleon.config.schema import TaskConfig
from chameleon.workflows.runner import WorkflowRunner


def test_plan_includes_trt_profile(task_deploy_yaml) -> None:
    task = TaskConfig.load(task_deploy_yaml)
    task.actions = ["export", "compile", "trt_profile"]
    lines = WorkflowRunner(task).plan()
    joined = "\n".join(lines)
    assert "trt_profile" in joined
    assert "vit.profile.json" in joined
    assert "viewer=both" in joined


def test_plan_trt_profile_without_engine_mentions_paths(task_deploy_yaml, caplog) -> None:
    task = TaskConfig.load(task_deploy_yaml)
    task.actions = ["trt_profile"]
    with caplog.at_level(logging.INFO):
        WorkflowRunner(task).run(dry_run=True)
    assert any("trt_profile: stage=vit" in r.message for r in caplog.records)
