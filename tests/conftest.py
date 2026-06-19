"""pytest 根配置 — 项目路径与共享 fixtures。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return CONFIGS


@pytest.fixture(scope="session")
def build_configs_dir(project_root: Path) -> Path:
    path = project_root / "configs" / "build_configs"
    if not path.is_dir():
        pytest.skip("configs/build_configs not present")
    return path.resolve()


@pytest.fixture(scope="session")
def task_deploy_yaml(configs_dir: Path) -> Path:
    path = configs_dir / "pi05_libero_trt_deploy.yaml"
    if not path.exists():
        pytest.skip("pi05_libero_trt_deploy.yaml not present")
    return path
