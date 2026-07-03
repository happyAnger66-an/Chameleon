"""compute_stats 端到端（reference 路径，无需 openpi）。"""

from __future__ import annotations

from chameleon.config.schema import TaskConfig
from chameleon.profile.compute_stats import format_stats_table, stats_infer, stats_result_to_dict


def test_stats_infer_reference_cpu() -> None:
    task = TaskConfig(
        model_overrides={"use_reference": True, "action_horizon": 10},
        infer={"num_steps": 3, "batch_size": 1},
    )
    result = stats_infer(task, measured=False, device="cpu")
    assert result.mode == "reference"
    assert len(result.stages) == 3
    assert result.totals.macs > 0
    assert result.totals.total_bytes > 0
    assert "action_expert" in [s.stage for s in result.stages]
    assert result.stages[-1].repeat == 3

    table = format_stats_table(result)
    assert "TOTAL" in table
    assert "Arithmetic intensity" in table

    payload = stats_result_to_dict(result)
    assert payload["totals"]["macs"]["raw"] == result.totals.macs
    assert payload["totals"]["flops"]["display"].endswith("FLOPs")
    assert payload["totals"]["total_bytes"]["unit"] in {"GB", "MB", "TB"}
    assert payload["measured_enabled"] is False
    assert "measured" not in payload["stages"][0]


def test_stats_infer_cosmos3_reference_cpu() -> None:
    task = TaskConfig(
        architecture="cosmos3",
        model="cosmos3",
        model_overrides={"use_reference": True, "mode": "video"},
        infer={"num_steps": 3, "batch_size": 1},
    )
    result = stats_infer(task, measured=False, device="cpu")
    assert result.mode == "reference"
    assert [s.stage for s in result.stages] == [
        "vae_encode",
        "text_embed",
        "dit",
        "vae_decode",
    ]
    assert result.stages[2].repeat == 3
    assert result.totals.flops > 0
    assert "dit" in result.execution_plan


def test_stats_infer_cosmos3_real_falls_back_without_diffusers() -> None:
    task = TaskConfig(
        architecture="cosmos3",
        model="cosmos3",
        model_overrides={"use_reference": False, "model_id": "nvidia/Cosmos3-Nano"},
        generate={"num_inference_steps": 4, "mode": "video"},
        infer={"batch_size": 1},
    )
    result = stats_infer(task, measured=False, device="cpu")
    assert result.mode == "real"
    assert len(result.stages) == 4
    assert any("diffusers pipeline unavailable" in w for w in result.warnings)


def test_measured_dict_includes_memory_comparison() -> None:
    from chameleon.profile.compute_stats import _measured_dict
    from chameleon.profile.counters import MeasuredStats, StageStats

    stage = StageStats(
        stage="vit",
        repeat=1,
        macs=100,
        flops=200,
        weight_bytes=800,
        activation_bytes=100,
        attention_bytes=0,
        total_bytes=900,
        measured=MeasuredStats(
            profiler_flops=150,
            profiler_device_memory=500,
            peak_device_memory=1024,
        ),
    )
    measured = _measured_dict(stage, theoretical_flops=200, theoretical_total_bytes=900)
    assert measured["status"] == "ok"
    assert measured["theoretical_total_bytes"]["raw"] == 900
    assert measured["profiler_device_memory"]["raw"] == 500
    assert measured["peak_device_memory"]["raw"] == 1024
    assert "memory_proxy_diff_pct" in measured
