"""Stage wall-clock timer — CUDA Event（GPU 段）+ perf_counter（host/IPC）。

用于 pi05 TRT / TVM 分阶段延迟对比。一次 ``begin_run``…``end_run`` 对应一次
完整推理；同名 stage 在单次 run 内累加（如 vit×N 相机）。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class StageStats:
    name: str
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    samples_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
        }


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize(name: str, samples: list[float]) -> StageStats:
    if not samples:
        return StageStats(name=name, count=0, mean_ms=0.0, p50_ms=0.0, p90_ms=0.0)
    ordered = sorted(samples)
    mean = sum(samples) / float(len(samples))
    return StageStats(
        name=name,
        count=len(samples),
        mean_ms=mean,
        p50_ms=_percentile(ordered, 0.5),
        p90_ms=_percentile(ordered, 0.9),
        samples_ms=list(samples),
    )


class StageTimer:
    """Accumulate per-run stage timings across multiple inference runs."""

    def __init__(self, *, enabled: bool = True, sync: str = "cuda_event") -> None:
        self.enabled = enabled
        self.sync = sync  # cuda_event | host
        self._run_samples: list[dict[str, float]] = []
        self._current: dict[str, float] | None = None
        self._e2e_t0: float | None = None

    def begin_run(self) -> None:
        if not self.enabled:
            return
        self._current = {}
        self._e2e_t0 = time.perf_counter()

    def end_run(self) -> dict[str, float]:
        if not self.enabled or self._current is None:
            return {}
        if self._e2e_t0 is not None and "e2e" not in self._current:
            self._current["e2e"] = (time.perf_counter() - self._e2e_t0) * 1e3
        snap = dict(self._current)
        self._run_samples.append(snap)
        self._current = None
        self._e2e_t0 = None
        return snap

    def add(self, name: str, ms: float) -> None:
        """Accumulate external timing (e.g. worker-reported ms) into the current run."""
        if not self.enabled or self._current is None:
            return
        self._current[name] = self._current.get(name, 0.0) + float(ms)

    @contextmanager
    def region(self, name: str, *, device: bool = True) -> Iterator[None]:
        if not self.enabled or self._current is None:
            yield
            return
        use_cuda = (
            device
            and self.sync == "cuda_event"
            and _cuda_available()
        )
        if use_cuda:
            import torch

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                end.synchronize()
                self._current[name] = self._current.get(name, 0.0) + float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._current[name] = self._current.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def summary(self) -> dict[str, StageStats]:
        keys: set[str] = set()
        for snap in self._run_samples:
            keys.update(snap.keys())
        out: dict[str, StageStats] = {}
        for key in sorted(keys):
            samples = [snap[key] for snap in self._run_samples if key in snap]
            out[key] = _summarize(key, samples)
        return out

    def summary_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in self.summary().items()}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def format_comparison_table(
    backends: dict[str, dict[str, StageStats]],
    *,
    stages: list[str] | None = None,
    primary: str = "trt",
    secondary: str = "tvm",
) -> str:
    """Render a text table comparing two backends' p50 stage times."""
    if not backends:
        return "(no backend results)"
    stage_order = stages or _default_stage_order(backends)
    cols = list(backends.keys())
    header = f"{'stage':<18}" + "".join(f"{c + '_p50':>12}" for c in cols)
    if primary in backends and secondary in backends:
        header += f"{'delta':>12}"
    lines = [header, "-" * len(header)]
    for stage in stage_order:
        row = f"{stage:<18}"
        vals: dict[str, float] = {}
        for c in cols:
            st = backends[c].get(stage)
            v = st.p50_ms if st is not None else float("nan")
            vals[c] = v
            row += f"{v:12.2f}" if st is not None else f"{'—':>12}"
        if primary in vals and secondary in vals and stage in backends.get(primary, {}) and stage in backends.get(
            secondary, {}
        ):
            delta = vals[secondary] - vals[primary]
            row += f"{delta:+12.2f}"
        elif primary in backends and secondary in backends:
            row += f"{'—':>12}"
        lines.append(row)
    return "\n".join(lines)


def _default_stage_order(backends: dict[str, dict[str, StageStats]]) -> list[str]:
    preferred = [
        "preprocess",
        "vit",
        "lang_embed",
        "prefix_prep",
        "llm_prefill",
        "denoise_total",
        "denoise_step_mean",
        "tvm_worker",
        "ipc",
        "e2e",
    ]
    seen: set[str] = set()
    for b in backends.values():
        seen.update(b.keys())
    ordered = [s for s in preferred if s in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered
