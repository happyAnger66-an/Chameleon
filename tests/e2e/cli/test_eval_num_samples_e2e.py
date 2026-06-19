"""eval CLI num_samples 覆盖 E2E。"""

from __future__ import annotations

import pytest

from chameleon.commands.eval import eval_cli
from chameleon.config.schema import TaskConfig


@pytest.mark.e2e
def test_cli_num_samples_syncs_data_window(monkeypatch, fixtures_dir, tmp_path) -> None:
    captured: dict = {}

    def fake_run_eval(task: TaskConfig):
        captured["evaluate_n"] = task.evaluate.num_samples
        captured["data_n"] = task.data.num_samples

        class _Summary:
            def describe(self) -> str:
                return "ok"

        return _Summary()

    monkeypatch.setattr("chameleon.commands.eval.run_eval", fake_run_eval)

    cfg = fixtures_dir / "eval_smoke.yaml"
    rc = eval_cli(["--config", str(cfg), "--num-samples", "1000"])
    assert rc == 0
    assert captured["evaluate_n"] == 1000
    assert captured["data_n"] == 1000
