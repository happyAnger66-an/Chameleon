"""Stage 级 MACs/FLOPs 与理论访存量估算。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MeasuredStats:
    profiler_flops: int | None = None
    profiler_cpu_memory: int | None = None
    """Sum of profiler ``cpu_memory_usage`` (allocator delta proxy, not DRAM traffic)."""
    profiler_device_memory: int | None = None
    """Sum of ``|self_device_memory_usage|`` per op (CUDA allocator churn proxy)."""
    peak_device_memory: int | None = None
    """``torch.cuda.max_memory_allocated()`` high-water during the profiled forward."""


@dataclass
class StageStats:
    stage: str
    repeat: int
    macs: int
    flops: int
    weight_bytes: int
    activation_bytes: int
    attention_bytes: int
    total_bytes: int
    measured: MeasuredStats | None = None

    @property
    def total_macs(self) -> int:
        return self.macs * self.repeat

    @property
    def total_flops(self) -> int:
        return self.flops * self.repeat

    @property
    def total_weight_bytes(self) -> int:
        return self.weight_bytes * self.repeat

    @property
    def total_activation_bytes(self) -> int:
        return (self.activation_bytes + self.attention_bytes) * self.repeat

    @property
    def total_memory_bytes(self) -> int:
        return self.total_bytes * self.repeat


@dataclass
class StatsTotals:
    macs: int = 0
    flops: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0
    attention_bytes: int = 0
    total_bytes: int = 0
    arithmetic_intensity: float = 0.0


@dataclass
class StatsResult:
    execution_plan: str
    mode: str
    batch_size: int
    num_steps: int
    precision: str
    measured_enabled: bool = False
    stages: list[StageStats] = field(default_factory=list)
    totals: StatsTotals = field(default_factory=StatsTotals)
    shapes: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _tensor_bytes(shape: tuple[int, ...], dtype_bytes: int) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * dtype_bytes


def _activation_bytes_from_tensors(tensors: tuple[Any, ...], dtype_bytes: int) -> int:
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor):
            elem_bytes = t.element_size()
            total += t.numel() * elem_bytes
        elif isinstance(t, (tuple, list)):
            total += _activation_bytes_from_tensors(tuple(t), dtype_bytes)
    return total


def estimate_attention_bytes(shapes: dict[str, tuple[int, ...]], dtype_bytes: int) -> int:
    """估算 attention 中间张量额外访存（QK logits + PV）。"""
    if "past_keys" in shapes:
        past = shapes["past_keys"]
        num_layers = int(past[0])
        batch = int(past[1])
        seq_kv = int(past[2])
        head_dim = int(past[3])
    else:
        return 0

    if "attention_mask" in shapes and len(shapes["attention_mask"]) == 4:
        seq_q = int(shapes["attention_mask"][2])
    elif "inputs_embeds" in shapes:
        seq_q = int(shapes["inputs_embeds"][1])
    elif "x_t" in shapes:
        seq_q = int(shapes["x_t"][1])
    else:
        seq_q = 1

    num_heads = max(1, head_dim // 64)
    logits_bytes = batch * num_heads * seq_q * seq_kv * dtype_bytes
    qkv_bytes = batch * (seq_q * num_heads * head_dim + 2 * seq_kv * head_dim) * dtype_bytes
    per_layer = qkv_bytes + logits_bytes * 2
    return num_layers * per_layer


def _count_flops_forward(module: nn.Module, run_forward: Callable[[], Any]) -> int:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fcm:
            run_forward()
        return int(fcm.get_total_flops())
    except Exception:
        return _count_flops_hook(module, run_forward)


def _count_flops_hook(module: nn.Module, run_forward: Callable[[], Any]) -> int:
    macs = 0
    handles: list[Any] = []

    def _mm_hook(_module, inputs, outputs):
        nonlocal macs
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        a = inputs[0]
        if a.dim() == 2 and hasattr(_module, "weight"):
            macs += a.shape[0] * a.shape[1] * _module.weight.shape[0]
        elif a.dim() == 3 and hasattr(_module, "weight"):
            macs += a.shape[0] * a.shape[1] * a.shape[2] * _module.weight.shape[0]

    for m in module.modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(_mm_hook))

    try:
        with torch.inference_mode():
            run_forward()
    finally:
        for h in handles:
            h.remove()

    return macs * 2


def _profiler_memory_from_events(events) -> tuple[int, int]:
    """Sum profiler-reported CPU / device memory from key_averages events."""
    cpu_total = 0
    device_total = 0
    for event in events:
        cpu_total += abs(int(getattr(event, "cpu_memory_usage", 0) or 0))
        device_val = 0
        for attr in (
            "self_device_memory_usage",
            "self_cuda_memory_usage",
            "device_memory_usage",
            "cuda_memory_usage",
        ):
            if hasattr(event, attr):
                device_val = int(getattr(event, attr) or 0)
                if device_val:
                    break
        device_total += abs(device_val)
    return cpu_total, device_total


def _measure_with_profiler(module: nn.Module, run_forward: Callable[[], Any], device: str) -> MeasuredStats | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None

    dev = torch.device(device)
    try:
        module.to(dev)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
            profile_memory=True,
        ) as prof:
            with torch.inference_mode():
                run_forward()

        torch.cuda.synchronize(dev)
        peak_device_memory = int(torch.cuda.max_memory_allocated(dev))
        events = prof.key_averages()
        profiler_flops = sum(int(getattr(e, "flops", 0) or 0) for e in events if getattr(e, "flops", 0) > 0)
        cpu_mem, device_mem = _profiler_memory_from_events(events)

        if profiler_flops <= 0 and peak_device_memory <= 0 and device_mem <= 0:
            return None

        return MeasuredStats(
            profiler_flops=profiler_flops if profiler_flops > 0 else None,
            profiler_cpu_memory=cpu_mem if cpu_mem > 0 else None,
            profiler_device_memory=device_mem if device_mem > 0 else None,
            peak_device_memory=peak_device_memory if peak_device_memory > 0 else None,
        )
    except Exception as exc:
        logger.warning("Profiler measurement failed: %s", exc)
        return None


def count_stage(
    *,
    stage: str,
    repeat: int,
    module: nn.Module,
    inputs: tuple[Any, ...],
    shapes: dict[str, tuple[int, ...]],
    dtype_bytes: int,
    device: str = "cpu",
    measured: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> StageStats:
    """对单次 stage forward 统计 MACs/FLOPs 与理论访存。"""

    def _report(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    module = module.eval()
    if device != "cpu":
        module = module.to(device)
        inputs = tuple(
            t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs
        )

    def _run() -> Any:
        return module(*inputs)

    _report("counting theoretical FLOPs")
    flops = _count_flops_forward(module, _run)
    macs = flops // 2

    weight_bytes = _count_parameters(module) * dtype_bytes
    activation_bytes = _activation_bytes_from_tensors(inputs, dtype_bytes) * 2
    _report("counting activation traffic")
    with torch.inference_mode():
        outputs = _run()
    if isinstance(outputs, torch.Tensor):
        activation_bytes += outputs.numel() * outputs.element_size()
    elif isinstance(outputs, (tuple, list)):
        activation_bytes += _activation_bytes_from_tensors(tuple(outputs), dtype_bytes)

    attention_bytes = estimate_attention_bytes(shapes, dtype_bytes)
    total_bytes = weight_bytes + activation_bytes + attention_bytes

    measured_stats = None
    if measured:
        _report("running CUDA profiler (--measured)")
        measured_stats = _measure_with_profiler(module, _run, device)

    return StageStats(
        stage=stage,
        repeat=repeat,
        macs=macs,
        flops=flops,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        attention_bytes=attention_bytes,
        total_bytes=total_bytes,
        measured=measured_stats,
    )


def aggregate_stats(stage_stats: list[StageStats]) -> StatsTotals:
    totals = StatsTotals()
    for s in stage_stats:
        totals.macs += s.total_macs
        totals.flops += s.total_flops
        totals.weight_bytes += s.total_weight_bytes
        totals.activation_bytes += s.activation_bytes * s.repeat
        totals.attention_bytes += s.attention_bytes * s.repeat
        totals.total_bytes += s.total_memory_bytes

    if totals.total_bytes > 0:
        totals.arithmetic_intensity = totals.macs / totals.total_bytes
    return totals
