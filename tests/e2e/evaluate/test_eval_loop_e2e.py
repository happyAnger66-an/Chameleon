"""evaluate 层 E2E — 离线评测循环（fake runner，无 openpi）。"""

from __future__ import annotations

import pytest

from chameleon.evaluate.lerobot_eval import evaluate_lerobot
from tests.helpers.fakes import CaptureEventSink, FakeDataSource, FakePolicyRunner


@pytest.mark.e2e
class TestEvalLoopE2E:
    def test_evaluate_lerobot_summary(self) -> None:
        ds = FakeDataSource(length=25, action_horizon=10, action_dim=7)
        runner = FakePolicyRunner(action_horizon=10, action_dim=7)
        summary = evaluate_lerobot(ds, runner, num_samples=25, log_every=0)
        assert summary.num_samples == 25
        assert summary.mean_max_abs > 0
        assert summary.worst_index >= 0

    def test_step_events_only_on_aligned_chunks(self) -> None:
        sink = CaptureEventSink()
        ds = FakeDataSource(length=25, action_horizon=10)
        runner = FakePolicyRunner()
        evaluate_lerobot(
            ds,
            runner,
            num_samples=25,
            event_sink=sink,
            run_meta={"type": "meta", "run_id": "t1"},
            run_id="t1",
            log_every=0,
        )
        # idx 0 与 10 两个 chunk × 10 步 = 20 step
        assert len(sink.steps) == 20
        xs = [s.global_index for s in sink.steps]
        assert xs == list(range(20))
        assert all(s.is_chunk_start for s in sink.steps if s.k_in_chunk == 0)

    def test_global_index_monotonic_for_plotly(self) -> None:
        """WebUI 折线要求 global_index 严格递增，避免折返乱线。"""
        sink = CaptureEventSink()
        ds = FakeDataSource(length=30, action_horizon=10)
        runner = FakePolicyRunner()
        evaluate_lerobot(ds, runner, num_samples=30, event_sink=sink, log_every=0)
        xs = [s.global_index for s in sink.steps]
        assert xs == sorted(xs)
        assert len(set(xs)) == len(xs)

    def test_on_run_done_called(self) -> None:
        sink = CaptureEventSink()
        ds = FakeDataSource(length=5, action_horizon=10)
        runner = FakePolicyRunner()
        evaluate_lerobot(ds, runner, num_samples=5, event_sink=sink, log_every=0)
        assert sink.summary is not None
        assert sink.summary.num_samples == 5

    def test_step_metrics_include_cumulative_dim_means(self) -> None:
        sink = CaptureEventSink()
        ds = FakeDataSource(length=25, action_horizon=10, action_dim=7)
        runner = FakePolicyRunner(action_horizon=10, action_dim=7)
        evaluate_lerobot(ds, runner, num_samples=25, event_sink=sink, log_every=0)
        assert sink.steps
        last = sink.steps[-1].metrics
        assert "mae_dim_mean" in last
        assert "mse_dim_mean" in last
        assert len(last["mae_dim_mean"]) == 7
        assert last["diff_step_count"] == len(sink.steps)
