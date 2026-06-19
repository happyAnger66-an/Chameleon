"""PolicyRunner registry E2E。"""

from __future__ import annotations

import pytest

from chameleon.config.schema import TaskConfig
from chameleon.evaluate.runner_base import POLICY_RUNNER_REGISTRY, build_policy_runner, list_policy_runners


@pytest.mark.e2e
class TestPolicyRunnerRegistryE2E:
    def test_registered_backends(self) -> None:
        names = list_policy_runners()
        assert "openpi" in names
        assert "chameleon" in names

    def test_build_openpi_runner_type(self, tmp_path) -> None:
        task = TaskConfig()
        task.evaluate.policy_runner = "openpi"
        task.evaluate.checkpoint_dir = str(tmp_path)
        runner = build_policy_runner(task)
        assert runner.__class__.__name__ == "OpenPiPolicyRunner"

    def test_build_chameleon_runner_type(self, tmp_path) -> None:
        cls = POLICY_RUNNER_REGISTRY.get("chameleon")
        assert cls.__name__ == "ChameleonOrchestratorRunner"
        try:
            __import__("openpi")
        except Exception:
            pytest.skip("chameleon from_task 需要可 import 的 openpi")
        task = TaskConfig()
        task.evaluate.policy_runner = "chameleon"
        task.evaluate.checkpoint_dir = str(tmp_path)
        runner = build_policy_runner(task)
        assert runner.__class__ is cls

    def test_unknown_runner_raises(self) -> None:
        task = TaskConfig()
        task.evaluate.policy_runner = "not_real"
        with pytest.raises(KeyError):
            build_policy_runner(task)
