"""Unit tests for StageTimer (no GPU required)."""

from __future__ import annotations

import time

from chameleon.profile.stage_timer import StageTimer, format_comparison_table


def test_stage_timer_host_regions() -> None:
    timer = StageTimer(enabled=True, sync="host")
    timer.begin_run()
    with timer.region("vit", device=False):
        time.sleep(0.01)
    with timer.region("vit", device=False):
        time.sleep(0.01)
    with timer.region("llm_prefill", device=False):
        time.sleep(0.005)
    snap = timer.end_run()
    assert "e2e" in snap
    assert snap["vit"] >= snap["llm_prefill"]
    assert snap["e2e"] >= snap["vit"]

    timer.begin_run()
    with timer.region("vit", device=False):
        time.sleep(0.01)
    timer.end_run()

    summary = timer.summary()
    assert summary["vit"].count == 2
    assert summary["vit"].p50_ms > 0
    assert summary["llm_prefill"].count == 1


def test_stage_timer_add_external() -> None:
    timer = StageTimer(sync="host")
    timer.begin_run()
    timer.add("tvm_worker", 12.5)
    timer.add("ipc", 1.5)
    snap = timer.end_run()
    assert snap["tvm_worker"] == 12.5
    assert snap["ipc"] == 1.5


def test_format_comparison_table() -> None:
    timer_a = StageTimer(sync="host")
    timer_a.begin_run()
    timer_a.add("e2e", 40.0)
    timer_a.add("llm_prefill", 10.0)
    timer_a.add("denoise_total", 20.0)
    timer_a.end_run()
    timer_b = StageTimer(sync="host")
    timer_b.begin_run()
    timer_b.add("e2e", 55.0)
    timer_b.add("llm_prefill", 25.0)
    timer_b.add("denoise_total", 22.0)
    timer_b.add("tvm_worker", 47.0)
    timer_b.end_run()
    table = format_comparison_table(
        {"trt": timer_a.summary(), "tvm": timer_b.summary()},
        stages=["llm_prefill", "denoise_total", "tvm_worker", "e2e"],
    )
    assert "llm_prefill" in table
    assert "denoise_total" in table
    assert "delta" in table
