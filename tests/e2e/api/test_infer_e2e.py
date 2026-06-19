"""api 层 E2E — infer 参考路径（CPU smoke）。"""

from __future__ import annotations

import pytest

from chameleon.config.schema import TaskConfig
from chameleon.workflows.runner import WorkflowRunner


@pytest.mark.e2e
@pytest.mark.e2e_slow
class TestInferApiE2E:
    def test_pi05_cpu_infer_via_workflow(self, configs_dir) -> None:
        task = TaskConfig.load(configs_dir / "pi05_cpu.yaml")
        manifest = WorkflowRunner(task).run(dry_run=False)
        kinds = [a.kind for a in manifest.artifacts]
        assert "inference" in kinds

    def test_cli_infer_smoke(self, cli_cmd, configs_dir) -> None:
        cfg = configs_dir / "pi05_cpu.yaml"
        result = cli_cmd("infer", "--config", str(cfg), timeout=180)
        assert result.returncode == 0
