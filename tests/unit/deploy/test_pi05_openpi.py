"""deploy pi05 编排单元测试（mock exporter）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chameleon.config.schema import TaskConfig
from chameleon.core.artifact import Manifest
from chameleon.deploy.pi05_openpi import iter_export_steps, run_pi05_build, run_pi05_export


@pytest.fixture
def deploy_task(task_deploy_yaml: Path, tmp_path: Path) -> TaskConfig:
    task = TaskConfig.load(task_deploy_yaml)
    task.output_dir = str(tmp_path / "out")
    task.deploy.export_dir = str(tmp_path / "out" / "onnx")
    task.deploy.engine_dir = str(tmp_path / "out" / "engines")
    return task


def test_iter_export_steps_defaults() -> None:
    task = TaskConfig()
    steps = iter_export_steps(task)
    assert [s.stage for s in steps] == ["vit", "llm", "expert", "denoise"]


def test_run_pi05_export_records_manifest(deploy_task: TaskConfig, tmp_path: Path) -> None:
    export_dir = Path(deploy_task.deploy.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    def _fake_export(task, step, *, paths=None):
        onnx = export_dir / f"{step.stage}.onnx"
        onnx.write_bytes(b"onnx")
        return str(onnx)

    manifest = Manifest(deploy_task.output_dir)
    with patch("chameleon.deploy.pi05_openpi.export_pi05_stage", side_effect=_fake_export):
        artifacts = run_pi05_export(deploy_task, manifest)

    assert set(artifacts) == {"vit", "llm", "expert", "denoise"}
    assert all(a.kind == "onnx" for a in artifacts.values())
    assert len(manifest.artifacts) == 4


def test_run_pi05_build_records_manifest(deploy_task: TaskConfig, tmp_path: Path) -> None:
    export_dir = Path(deploy_task.deploy.export_dir)
    engine_dir = Path(deploy_task.deploy.engine_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)
    for stage in ("vit", "llm", "expert", "denoise"):
        (export_dir / f"{stage}.onnx").write_bytes(b"onnx")

    def _fake_build(task, stage, *, paths=None, build_cfg_path=None, use_cudagraph=None):
        engine = engine_dir / f"{stage}.engine"
        engine.write_bytes(b"engine")
        return str(engine)

    manifest = Manifest(deploy_task.output_dir)
    with patch("chameleon.deploy.pi05_openpi.build_pi05_stage_engine", side_effect=_fake_build):
        artifacts = run_pi05_build(deploy_task, manifest)

    assert set(artifacts) == {"vit", "llm", "expert", "denoise"}
    assert all(a.kind == "engine" for a in artifacts.values())
