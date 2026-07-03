"""整模型计算量与访存量统计 — 主入口。"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from chameleon.api import build_adapter
from chameleon.config.schema import TaskConfig
from chameleon.deploy.paths import resolve_deploy_paths
from chameleon.deploy.pi05.stats import prepare_pi05_stage
from chameleon.profile.counters import StatsResult, StatsTotals, aggregate_stats, count_stage
from chameleon.profile.execution_plan import ExecutionPlan, PlanMode, build_execution_plan
from chameleon.profile.cosmos3_real_stats import prepare_real_cosmos3_stage
from chameleon.profile.reference_stats import prepare_reference_stage
from chameleon.profile.shape_resolver import (
    precision_to_dtype_bytes,
    resolve_precision,
    resolve_stage_shapes,
    shapes_summary,
)
from chameleon.profile.units import format_bytes, format_ops, pick_bytes_column_unit, pick_ops_column_unit

logger = logging.getLogger(__name__)


def _stats_progress(msg: str, *, enabled: bool) -> None:
    if enabled:
        print(msg, file=sys.stderr, flush=True)


def _resolve_stats_device(task: TaskConfig, *, measured: bool) -> str:
    requested = task.infer.torch_device or "cpu"
    if measured and requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    if requested.startswith("cuda") and torch.cuda.is_available() and not measured:
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested if not requested.startswith("cuda") or torch.cuda.is_available() else "cpu"


def stats_infer(
    task: TaskConfig,
    *,
    measured: bool = False,
    device: str | None = None,
    progress: bool = True,
) -> StatsResult:
    plan = build_execution_plan(task)
    precision = resolve_precision(task)
    dtype_bytes = precision_to_dtype_bytes(precision)
    stats_device = device or _resolve_stats_device(task, measured=measured)

    result = StatsResult(
        execution_plan=plan.describe(),
        mode=plan.mode.value,
        batch_size=plan.batch_size,
        num_steps=plan.num_steps,
        precision=precision,
    )

    if measured and not stats_device.startswith("cuda"):
        result.warnings.append("--measured requested but CUDA unavailable; theoretical only.")

    stage_stats = []
    shapes_map: dict[str, dict[str, list[int]]] = {}
    total_stages = len(plan.stages)

    def _stage_progress_factory(index: int, stage_name: str) -> Callable[[str], None]:
        prefix = f"[stats {index}/{total_stages}] {stage_name}"

        def _report(substep: str) -> None:
            _stats_progress(f"{prefix}: {substep}...", enabled=progress)

        return _report

    if plan.mode == PlanMode.REFERENCE or task.architecture == "cosmos3":
        _stats_progress(
            f"Loading model adapter (device={stats_device}, measured={measured})...",
            enabled=progress,
        )
        t0 = time.perf_counter()
        adapter = build_adapter(task, device=stats_device)
        _stats_progress(
            f"Model adapter ready in {time.perf_counter() - t0:.1f}s",
            enabled=progress,
        )
        if (
            task.architecture == "cosmos3"
            and not bool(task.model_overrides.get("use_reference", True))
            and not getattr(adapter, "_is_real_diffusers", False)
        ):
            result.warnings.append(
                "cosmos3 real weights requested but diffusers pipeline unavailable; "
                "stats use reference surrogate (~0.68M params, not 16B/64B). "
                "Install diffusers with Cosmos3OmniPipeline support: pip install -e '.[cosmos3]'."
            )
        for idx, sr in enumerate(plan.stages, start=1):
            shapes = resolve_stage_shapes(task, sr.stage, plan)
            shapes_map[sr.stage] = shapes_summary(shapes)
            _stats_progress(
                f"[stats {idx}/{total_stages}] {sr.stage} (repeat×{sr.repeat}, "
                f"device={stats_device}): preparing inputs...",
                enabled=progress,
            )
            t_stage = time.perf_counter()
            if getattr(adapter, "_is_real_diffusers", False):
                module, inputs = prepare_real_cosmos3_stage(
                    adapter, sr.stage, task, device=stats_device
                )
            else:
                module, inputs = prepare_reference_stage(
                    adapter, sr.stage, shapes, plan=plan, device=stats_device
                )
            stage_stats.append(
                count_stage(
                    stage=sr.stage,
                    repeat=sr.repeat,
                    module=module,
                    inputs=inputs,
                    shapes=shapes,
                    dtype_bytes=dtype_bytes,
                    device=stats_device,
                    measured=measured,
                    on_progress=_stage_progress_factory(idx, sr.stage),
                )
            )
            _stats_progress(
                f"[stats {idx}/{total_stages}] {sr.stage}: done in "
                f"{time.perf_counter() - t_stage:.1f}s",
                enabled=progress,
            )
    else:
        from chameleon.deploy.pi05.loader import load_pi05_model

        paths = resolve_deploy_paths(task)
        try:
            pi05_model = load_pi05_model(
                str(paths.checkpoint_dir),
                paths.train_config,
                device="cpu",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load pi05 model for stats ({plan.mode.value} path): {exc}"
            ) from exc

        try:
            for idx, sr in enumerate(plan.stages, start=1):
                shapes = resolve_stage_shapes(task, sr.stage, plan)
                shapes_map[sr.stage] = shapes_summary(shapes)
                _stats_progress(
                    f"[stats {idx}/{total_stages}] {sr.stage} (repeat×{sr.repeat}): preparing...",
                    enabled=progress,
                )
                t_stage = time.perf_counter()
                module, inputs = prepare_pi05_stage(
                    sr.stage,
                    pi05_model,
                    shapes,
                    precision=precision,
                    device=stats_device,
                )
                stage_stats.append(
                    count_stage(
                        stage=sr.stage,
                        repeat=sr.repeat,
                        module=module,
                        inputs=inputs,
                        shapes=shapes,
                        dtype_bytes=dtype_bytes,
                        device=stats_device,
                        measured=measured,
                        on_progress=_stage_progress_factory(idx, sr.stage),
                    )
                )
                _stats_progress(
                    f"[stats {idx}/{total_stages}] {sr.stage}: done in "
                    f"{time.perf_counter() - t_stage:.1f}s",
                    enabled=progress,
                )
        finally:
            del pi05_model

    result.stages = stage_stats
    result.shapes = shapes_map
    result.totals = aggregate_stats(stage_stats)
    result.measured_enabled = measured
    if measured and stats_device.startswith("cuda"):
        failed = [s.stage for s in stage_stats if s.measured is None]
        if failed:
            result.warnings.append(
                f"--measured: CUDA profiler produced no data for stage(s): {', '.join(failed)}"
            )
        for s in stage_stats:
            if s.measured and s.measured.profiler_flops and s.flops > 0:
                diff_pct = abs(s.measured.profiler_flops - s.flops) / s.flops * 100
                if diff_pct > 10.0:
                    result.warnings.append(
                        f"--measured: stage {s.stage} theoretical flops vs profiler differ by {diff_pct:.0f}% "
                        f"(theoretical={s.flops}, profiler={s.measured.profiler_flops})"
                    )
    return result


def format_stats_table(result: StatsResult) -> str:
    max_macs = max((s.macs for s in result.stages), default=0)
    max_flops = max((s.flops for s in result.stages), default=0)
    max_bytes = max((s.total_bytes for s in result.stages), default=0)
    max_weight = max((s.weight_bytes for s in result.stages), default=0)
    max_act = max((s.activation_bytes + s.attention_bytes for s in result.stages), default=0)

    macs_scale, macs_prefix, _ = pick_ops_column_unit(max(max_macs, result.totals.macs))
    flops_scale, _, flops_label = pick_ops_column_unit(max(max_flops, result.totals.flops))
    bytes_scale, bytes_label = pick_bytes_column_unit(max(max_bytes, result.totals.total_bytes))
    weight_scale, weight_label = pick_bytes_column_unit(max(max_weight, result.totals.weight_bytes))
    act_scale, act_label = pick_bytes_column_unit(
        max(max_act, result.totals.activation_bytes + result.totals.attention_bytes)
    )

    header = (
        f"Execution plan: {result.execution_plan}  "
        f"batch={result.batch_size}  num_steps={result.num_steps}  precision={result.precision}\n"
        f"{'Stage':<16} {'Repeat':>6} "
        f"{f'MACs({macs_prefix})':>12} {flops_label:>12} "
        f"{f'Weight({weight_label})':>14} {f'Act({act_label})':>12} {f'Total({bytes_label})':>12}"
    )
    lines = [header, "-" * len(header.split("\n")[-1])]

    def _ops(value: int, scale: float) -> str:
        return f"{value / scale:.3f}"

    for s in result.stages:
        act_bytes = s.activation_bytes + s.attention_bytes
        lines.append(
            f"{s.stage:<16} {s.repeat:>6} "
            f"{_ops(s.macs, macs_scale):>12} {_ops(s.flops, flops_scale):>12} "
            f"{_ops(s.weight_bytes, weight_scale):>14} "
            f"{_ops(act_bytes, act_scale):>12} {_ops(s.total_bytes, bytes_scale):>12}"
        )

    t = result.totals
    total_act = t.activation_bytes + t.attention_bytes
    lines.append(
        f"{'TOTAL':<16} {'':>6} "
        f"{_ops(t.macs, macs_scale):>12} {_ops(t.flops, flops_scale):>12} "
        f"{_ops(t.weight_bytes, weight_scale):>14} "
        f"{_ops(total_act, act_scale):>12} {_ops(t.total_bytes, bytes_scale):>12}"
    )
    lines.append(f"Arithmetic intensity: {t.arithmetic_intensity:.2f} MAC/Byte")

    if result.measured_enabled:
        lines.append("")
        lines.append("Measured (CUDA profiler validation):")
        for s in result.stages:
            if s.measured is None:
                lines.append(f"  {s.stage}: profiler failed")
                continue
            parts = []
            if s.measured.profiler_flops is not None:
                diff = ""
                if s.flops > 0:
                    diff_pct = (s.measured.profiler_flops - s.flops) / s.flops * 100
                    diff = f", diff={diff_pct:+.1f}%"
                parts.append(
                    f"profiler_flops={format_ops(s.measured.profiler_flops, kind='FLOP')['display']}{diff}"
                )
            if s.measured.profiler_device_memory is not None:
                mem_diff = ""
                if s.total_bytes > 0:
                    mem_diff_pct = (
                        (s.measured.profiler_device_memory - s.total_bytes) / s.total_bytes * 100
                    )
                    mem_diff = f", mem_proxy_diff={mem_diff_pct:+.1f}%"
                parts.append(
                    "profiler_mem="
                    f"{format_bytes(s.measured.profiler_device_memory)['display']}"
                    f" vs theoretical={format_bytes(s.total_bytes)['display']}"
                    f"{mem_diff}"
                )
            if s.measured.peak_device_memory is not None:
                parts.append(
                    f"peak_device_mem={format_bytes(s.measured.peak_device_memory)['display']}"
                )
            lines.append(f"  {s.stage}: {'; '.join(parts) if parts else 'no profiler data'}")

    if result.warnings:
        lines.append("")
        lines.extend(f"WARN: {w}" for w in result.warnings)

    return "\n".join(lines)


def _measured_dict(s, *, theoretical_flops: int, theoretical_total_bytes: int) -> dict[str, Any]:
    if s.measured is None:
        return {"status": "failed", "reason": "profiler unavailable or returned no data"}

    out: dict[str, Any] = {
        "status": "ok",
        "note": (
            "theoretical_* fields are analytic traffic (weight+activation+attention). "
            "profiler_* memory fields are CUDA allocator deltas / peak VRAM — not exact DRAM bytes moved."
        ),
        "theoretical_total_bytes": format_bytes(theoretical_total_bytes),
        "theoretical_weight_bytes": format_bytes(s.weight_bytes),
        "theoretical_activation_bytes": format_bytes(s.activation_bytes),
        "theoretical_attention_bytes": format_bytes(s.attention_bytes),
    }
    if s.measured.profiler_flops is not None:
        out["profiler_flops"] = format_ops(s.measured.profiler_flops, kind="FLOP")
        if theoretical_flops > 0:
            diff = (s.measured.profiler_flops - theoretical_flops) / theoretical_flops * 100
            out["flops_diff_pct"] = round(diff, 2)
    if s.measured.profiler_cpu_memory is not None:
        out["profiler_cpu_memory"] = format_bytes(s.measured.profiler_cpu_memory)
    if s.measured.profiler_device_memory is not None:
        out["profiler_device_memory"] = format_bytes(s.measured.profiler_device_memory)
        if theoretical_total_bytes > 0:
            mem_diff = (
                (s.measured.profiler_device_memory - theoretical_total_bytes)
                / theoretical_total_bytes
                * 100
            )
            out["memory_proxy_diff_pct"] = round(mem_diff, 2)
    if s.measured.peak_device_memory is not None:
        out["peak_device_memory"] = format_bytes(s.measured.peak_device_memory)
    return out


def _stage_dict(s, *, measured_enabled: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "stage": s.stage,
        "repeat": s.repeat,
        "macs": format_ops(s.macs, kind="MAC"),
        "flops": format_ops(s.flops, kind="FLOP"),
        "weight_bytes": format_bytes(s.weight_bytes),
        "activation_bytes": format_bytes(s.activation_bytes),
        "attention_bytes": format_bytes(s.attention_bytes),
        "total_bytes_per_call": format_bytes(s.total_bytes),
        "total_macs": format_ops(s.total_macs, kind="MAC"),
        "total_flops": format_ops(s.total_flops, kind="FLOP"),
        "total_bytes": format_bytes(s.total_memory_bytes),
    }
    if measured_enabled:
        d["measured"] = _measured_dict(
            s,
            theoretical_flops=s.flops,
            theoretical_total_bytes=s.total_bytes,
        )
    return d


def stats_result_to_dict(result: StatsResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "execution_plan": result.execution_plan,
        "mode": result.mode,
        "batch_size": result.batch_size,
        "num_steps": result.num_steps,
        "precision": result.precision,
        "measured_enabled": result.measured_enabled,
        "shapes": result.shapes,
        "stages": [_stage_dict(s, measured_enabled=result.measured_enabled) for s in result.stages],
        "totals": {
            "macs": format_ops(result.totals.macs, kind="MAC"),
            "flops": format_ops(result.totals.flops, kind="FLOP"),
            "weight_bytes": format_bytes(result.totals.weight_bytes),
            "activation_bytes": format_bytes(result.totals.activation_bytes),
            "attention_bytes": format_bytes(result.totals.attention_bytes),
            "total_bytes": format_bytes(result.totals.total_bytes),
            "arithmetic_intensity": result.totals.arithmetic_intensity,
        },
        "warnings": result.warnings,
    }
    if result.measured_enabled:
        profiler_flops_total = sum(
            (s.measured.profiler_flops or 0) * s.repeat
            for s in result.stages
            if s.measured and s.measured.profiler_flops
        )
        profiler_mem_total = sum(
            (s.measured.profiler_device_memory or 0) * s.repeat
            for s in result.stages
            if s.measured and s.measured.profiler_device_memory
        )
        peak_mem_max = max(
            (s.measured.peak_device_memory or 0 for s in result.stages if s.measured),
            default=0,
        )
        summary: dict[str, Any] = {
            "theoretical_total_bytes": format_bytes(result.totals.total_bytes),
        }
        if profiler_flops_total > 0 and result.totals.flops > 0:
            summary["profiler_flops_total"] = format_ops(profiler_flops_total, kind="FLOP")
            summary["theoretical_flops_total"] = format_ops(result.totals.flops, kind="FLOP")
            summary["flops_diff_pct"] = round(
                (profiler_flops_total - result.totals.flops) / result.totals.flops * 100,
                2,
            )
        if profiler_mem_total > 0:
            summary["profiler_device_memory_total"] = format_bytes(profiler_mem_total)
            if result.totals.total_bytes > 0:
                summary["memory_proxy_diff_pct"] = round(
                    (profiler_mem_total - result.totals.total_bytes) / result.totals.total_bytes * 100,
                    2,
                )
        if peak_mem_max > 0:
            summary["peak_device_memory_max"] = format_bytes(peak_mem_max)
        if len(summary) > 1:
            payload["measured_summary"] = summary
    return payload


def write_stats_json(result: StatsResult, path: str | Path) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats_result_to_dict(result), indent=2))
