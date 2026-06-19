"""export / deploy workflow E2E（dry-run）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chameleon.config.schema import TaskConfig
from chameleon.workflows.runner import WorkflowRunner


@pytest.mark.e2e
class TestDeployWorkflowE2E:
    def test_export_dry_run(self, task_deploy_yaml: Path) -> None:
        task = TaskConfig.load(task_deploy_yaml)
        task.actions = ["export"]
        lines = WorkflowRunner(task).plan()
        text = "\n".join(lines)
        assert "export" in text
        assert "vit" in text or "default stages" in text

    def test_compile_dry_run_pi05_openpi(self, task_deploy_yaml: Path) -> None:
        task = TaskConfig.load(task_deploy_yaml)
        task.actions = ["compile"]
        lines = WorkflowRunner(task).plan()
        assert any("compile:" in line for line in lines)
