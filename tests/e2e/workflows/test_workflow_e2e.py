"""workflows 层 E2E — dry-run 编排。"""

from __future__ import annotations

import pytest

from chameleon.config.schema import TaskConfig
from chameleon.workflows.runner import WorkflowRunner


@pytest.mark.e2e
class TestWorkflowE2E:
    def test_dry_run_plan(self, configs_dir) -> None:
        task = TaskConfig.load(configs_dir / "pi05_cpu.yaml")
        runner = WorkflowRunner(task)
        lines = runner.plan()
        assert any("architecture=pi05" in line for line in lines)
        assert any("infer" in line for line in lines)

    def test_dry_run_returns_manifest(self, configs_dir) -> None:
        task = TaskConfig.load(configs_dir / "pi05_cpu.yaml")
        manifest = WorkflowRunner(task).run(dry_run=True)
        assert manifest is not None

    def test_cli_workflow_dry_run(self, cli_cmd, configs_dir) -> None:
        cfg = configs_dir / "pi05_cpu.yaml"
        result = cli_cmd("workflow", "--config", str(cfg), "--dry-run", "-v")
        assert result.returncode == 0
        assert "architecture=pi05" in result.stderr or "architecture=pi05" in result.stdout
