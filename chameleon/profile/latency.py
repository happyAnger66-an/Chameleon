"""Minimal latency profiler for an inference session."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from chameleon.api import _run_context, build_adapter
from chameleon.config.schema import TaskConfig
from chameleon.runtime.orchestrator import InferenceSession


@dataclass
class LatencyResult:
    runs: int
    mean_ms: float
    p50_ms: float
    p90_ms: float


def profile_infer(task: TaskConfig, runs: int = 20, warmup: int = 3) -> LatencyResult:
    ctx = _run_context(task)
    adapter = build_adapter(task, device=ctx.torch_device)
    obs = adapter.example_observation(task.infer.batch_size, device=ctx.torch_device)
    session = InferenceSession(adapter, ctx, stage_runtimes=task.stage_runtimes).build()

    for _ in range(warmup):
        session.infer(obs)

    samples_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        session.infer(obs)
        samples_ms.append((time.perf_counter() - start) * 1e3)

    t = torch.tensor(samples_ms)
    return LatencyResult(
        runs=runs,
        mean_ms=float(t.mean()),
        p50_ms=float(t.quantile(0.5)),
        p90_ms=float(t.quantile(0.9)),
    )
