"""deploy.paths 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chameleon.config.schema import TaskConfig
from chameleon.deploy.paths import (
    DEFAULT_BUILD_CFGS,
    resolve_build_cfg_path,
    resolve_deploy_paths,
    stage_engine_path,
    stage_onnx_path,
)


def test_default_build_cfg_mapping() -> None:
    assert DEFAULT_BUILD_CFGS["vit"] == "vit_build_cfg.py"


def test_resolve_deploy_paths(task_deploy_yaml: Path, build_configs_dir: Path) -> None:
    task = TaskConfig.load(task_deploy_yaml)
    paths = resolve_deploy_paths(task)
    assert paths.export_dir.name == "onnx"
    assert paths.engine_dir.name == "engines"
    assert paths.build_cfg_dir == build_configs_dir.resolve()
    assert stage_onnx_path(paths, "vit").name == "vit.onnx"
    assert stage_engine_path(paths, "llm").name == "llm.engine"


def test_resolve_build_cfg_path(task_deploy_yaml: Path, build_configs_dir: Path) -> None:
    task = TaskConfig.load(task_deploy_yaml)
    paths = resolve_deploy_paths(task)
    cfg = resolve_build_cfg_path(task, "vit", paths)
    assert cfg.is_file()
    assert cfg.name == "vit_build_cfg.py"


def test_resolve_checkpoint_dir_missing_raises() -> None:
    task = TaskConfig()
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_deploy_paths(task)
