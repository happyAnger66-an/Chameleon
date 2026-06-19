"""E2E 层 fixtures — CLI 子进程与配置路径。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    return project_root / "configs"


@pytest.fixture
def cli_cmd(project_root: Path):
    """运行 chameleon CLI（开发模式 PYTHONPATH=项目根）。"""

    def _run(*args: str, cwd: str | None = None, env: dict | None = None, timeout: int = 120, check: bool = True):
        import shutil

        cli_bin = shutil.which("chameleon")
        if cli_bin:
            cmd = [cli_bin, *args]
        else:
            cmd = [sys.executable, "-m", "chameleon.cli", *args]
        full_env = os.environ.copy()
        root = str(project_root)
        full_env["PYTHONPATH"] = root + os.pathsep + full_env.get("PYTHONPATH", "")
        if env:
            full_env.update(env)
        result = subprocess.run(
            cmd,
            cwd=cwd or root,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            pytest.fail(
                f"CLI failed (rc={result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\nargs: {args}"
            )
        return result

    return _run
