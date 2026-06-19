"""config 层 E2E — YAML 加载与 evaluate 字段。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chameleon.config.schema import TaskConfig


@pytest.mark.e2e
class TestTaskConfigE2E:
    def test_fixture_eval_smoke(self, fixtures_dir: Path) -> None:
        task = TaskConfig.load(fixtures_dir / "eval_smoke.yaml")
        assert task.evaluate.viewer == "console"
        assert task.data.start_index == 0

    def test_libero_eval_yaml_has_webui_fields(self, configs_dir: Path) -> None:
        path = configs_dir / "pi05_libero_eval.yaml"
        if not path.exists():
            pytest.skip("pi05_libero_eval.yaml not present")
        task = TaskConfig.load(path)
        assert task.evaluate.viewer in ("console", "webui", "both")
        assert task.evaluate.webui_path.startswith("/")

    @pytest.mark.parametrize(
        "name",
        [
            "pi05_cpu.yaml",
            "pi05_nvidia.yaml",
            "pi05_nvidia_trt.yaml",
            "pi05_libero_trt_deploy.yaml",
        ],
    )
    def test_all_shipped_configs_validate(self, configs_dir: Path, name: str) -> None:
        task = TaskConfig.load(configs_dir / name)
        assert task.model == "pi05"
        assert isinstance(task.actions, list)
