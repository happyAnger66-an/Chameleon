"""推理延迟 profiler — 对 InferenceSession 做 warmup + 多次计时。

作用：
    profile_infer() 构建 session 后执行 warmup 和 runs 次 infer，
    返回 mean / p50 / p90 延迟（毫秒）。

架构位置：
    工具层 — 被 cli profile 子命令调用，基于 runtime/orchestrator
    InferenceSession 测量端到端延迟。
"""

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
