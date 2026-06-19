"""deploy.build_cfg 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chameleon.deploy.build_cfg import load_build_cfg


def test_load_build_cfg_from_chameleon_configs(build_configs_dir: Path) -> None:
    path = build_configs_dir / "vit_build_cfg.py"
    cfg = load_build_cfg(path)
    assert cfg["precision"] == "bf16"
    assert "min_shapes" in cfg


def test_load_build_cfg_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_build_cfg("/nonexistent/build_cfg.py")
