"""TaskConfig schema 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chameleon.config.schema import DeployConfig, EvaluateConfig, ExportStep, TaskConfig


def test_evaluate_defaults() -> None:
    ev = EvaluateConfig()
    assert ev.viewer == "console"
    assert ev.webui_port == 8765
    assert ev.policy_runner == "openpi"


def test_deploy_defaults() -> None:
    deploy = DeployConfig()
    assert deploy.backend == "reference"
    assert deploy.use_cudagraph is False


def test_export_step_roundtrip() -> None:
    step = ExportStep(stage="vit", options={"dynamo": True})
    assert step.stage == "vit"
    assert step.options["dynamo"] is True


def test_load_fixture_yaml(fixtures_dir: Path) -> None:
    task = TaskConfig.load(fixtures_dir / "eval_smoke.yaml")
    assert task.actions == ["eval"]
    assert task.evaluate.num_samples == 25
    assert task.data.dataset == "pi05_libero"


@pytest.mark.parametrize("name", ["pi05_cpu.yaml", "pi05_nvidia.yaml"])
def test_load_repo_configs(configs_dir: Path, name: str) -> None:
    task = TaskConfig.load(configs_dir / name)
    assert task.architecture == "pi05"
    assert task.actions
