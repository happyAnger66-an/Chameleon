"""TaskConfig schema 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chameleon.config.schema import DeployConfig, EvaluateConfig, ExportStep, TaskConfig, TrtProfileConfig


def test_evaluate_defaults() -> None:
    ev = EvaluateConfig()
    assert ev.viewer == "console"
    assert ev.webui_port == 8765
    assert ev.policy_runner == "openpi"
    assert ev.compare_mode is False
    assert ev.noise == "random"
    assert ev.pytorch_load_device == "cpu"


def test_load_trt_compare_yaml(configs_dir: Path) -> None:
    path = configs_dir / "pi05_libero_trt_compare.yaml"
    if not path.is_file():
        pytest.skip("pi05_libero_trt_compare.yaml not present")
    task = TaskConfig.load(path)
    assert task.actions == ["eval"]
    assert task.evaluate.compare_mode is True
    assert task.evaluate.policy_runner == "pt_trt_compare"
    assert task.evaluate.viewer == "both"
    assert task.evaluate.noise == "fixed"
    assert task.deploy.engine_dir == "output/pi05_libero_trt/engines"


def test_load_trt_eval_yaml(configs_dir: Path) -> None:
    path = configs_dir / "pi05_libero_trt_eval.yaml"
    if not path.is_file():
        pytest.skip("pi05_libero_trt_eval.yaml not present")
    task = TaskConfig.load(path)
    assert task.evaluate.policy_runner == "trt_only"
    assert task.evaluate.compare_mode is False
    assert task.evaluate.engine_dir == "output/pi05_libero_trt/engines"
    assert task.evaluate.pytorch_load_device == "cpu"


def test_deploy_defaults() -> None:
    deploy = DeployConfig()
    assert deploy.backend == "reference"
    assert deploy.use_cudagraph is False


def test_trt_profile_defaults() -> None:
    cfg = TrtProfileConfig()
    assert cfg.viewer == "static"
    assert cfg.iterations == 20
    assert cfg.webui_port == 8770


def test_load_deploy_yaml_has_trt_profile(task_deploy_yaml: Path) -> None:
    task = TaskConfig.load(task_deploy_yaml)
    assert "trt_profile" in task.actions
    assert task.profile.viewer == "both"
    assert len(task.trt_profile) == 4


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
