"""pi05 stage-level latency bench — TRT vs TVM 分阶段对比。

对固定 observation + noise 做 warmup + N runs，经 ``StageTimer`` 收集各 stage
wall-clock，输出控制台对比表与 JSON 报告。
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chameleon.config.schema import TaskConfig
from chameleon.profile.stage_timer import (
    StageStats,
    StageTimer,
    format_comparison_table,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchReport:
    meta: dict[str, Any]
    backends: dict[str, dict[str, StageStats]] = field(default_factory=dict)
    delta: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "backends": {
                name: {k: v.to_dict() for k, v in stages.items()}
                for name, stages in self.backends.items()
            },
            "delta": self.delta,
        }

    def format_table(self, stages: list[str] | None = None) -> str:
        return format_comparison_table(
            self.backends,
            stages=stages,
            primary="trt",
            secondary="tvm",
        )


def _load_observation(task: TaskConfig, sample_index: int) -> dict[str, Any]:
    from chameleon.dataloader import build_dataset_from_config

    data_cfg = task.data
    if not getattr(data_cfg, "dataset", None):
        raise ValueError("bench 需要 task.data.dataset（如 pi05_libero）以取固定 observation。")
    source = build_dataset_from_config(data_cfg)
    source.build()
    idx = int(sample_index)
    if idx < 0 or idx >= len(source):
        raise IndexError(f"bench sample_index={idx} out of range [0, {len(source)})")
    sample = source[idx]
    # LeRobot / ChameleonSample: observation dict
    obs = getattr(sample, "observation", None)
    if obs is None and isinstance(sample, dict):
        obs = sample.get("observation") or sample
    if not isinstance(obs, dict):
        raise TypeError(f"unexpected sample type for bench: {type(sample)}")
    return dict(obs)


def _build_runner(task: TaskConfig, backend: str) -> Any:
    # Ensure runners are registered.
    import chameleon.evaluate  # noqa: F401
    from chameleon.evaluate.runner_base import build_policy_runner

    # Clone task with the right policy_runner name.
    saved = task.evaluate.policy_runner
    saved_loop = task.model_overrides.get("tvm_loop")
    try:
        if backend == "trt":
            task.evaluate.policy_runner = "trt_only"
        elif backend == "tvm":
            task.evaluate.policy_runner = "tvm_only"
            if task.bench.tvm_loop is not None:
                task.model_overrides["tvm_loop"] = bool(task.bench.tvm_loop)
        else:
            raise ValueError(f"unknown bench backend: {backend!r} (expected trt|tvm)")
        runner = build_policy_runner(task)
        runner.build()
        return runner
    finally:
        task.evaluate.policy_runner = saved
        if saved_loop is None:
            task.model_overrides.pop("tvm_loop", None)
        else:
            task.model_overrides["tvm_loop"] = saved_loop


def _free_runner(runner: Any) -> None:
    """Release runner GPU resources before switching backends (TRT → TVM)."""
    from chameleon.runtime.tensorrt.backend import memory_report

    logger.info("bench freeing runner… before=%s", memory_report())
    close = getattr(runner, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            logger.warning("runner.close failed", exc_info=True)
    # Drop references held by the local frame / policy closures.
    try:
        if hasattr(runner, "_pipeline"):
            runner._pipeline = None
        if hasattr(runner, "_session"):
            runner._session = None
    except Exception:  # noqa: BLE001
        pass
    del runner
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Encourage the allocator to return cached blocks to the driver.
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    logger.info("bench freed runner; after=%s", memory_report())


def _bench_one(
    runner: Any,
    observation: dict[str, Any],
    noise: Any,
    *,
    warmup: int,
    runs: int,
    sync: str,
) -> dict[str, StageStats]:
    set_timer = getattr(runner, "set_timer", None)
    timer = StageTimer(enabled=True, sync=sync)

    for i in range(max(0, int(warmup))):
        if callable(set_timer):
            set_timer(None)
        runner.infer(observation, noise=noise)
        logger.info("bench warmup %d/%d done", i + 1, warmup)

    if callable(set_timer):
        set_timer(timer)
    try:
        for i in range(max(1, int(runs))):
            timer.begin_run()
            runner.infer(observation, noise=noise)
            timer.end_run()
            logger.info("bench run %d/%d done", i + 1, runs)
    finally:
        if callable(set_timer):
            set_timer(None)
    return timer.summary()


def _compute_delta(
    backends: dict[str, dict[str, StageStats]],
    *,
    primary: str = "trt",
    secondary: str = "tvm",
) -> dict[str, float]:
    if primary not in backends or secondary not in backends:
        return {}
    a, b = backends[primary], backends[secondary]
    keys = set(a) | set(b)
    out: dict[str, float] = {}
    for k in sorted(keys):
        if k in a and k in b:
            out[k] = float(b[k].p50_ms - a[k].p50_ms)
    # Helpful aggregates beyond per-stage p50 delta
    if "llm_prefill" in a and "denoise_total" in a and "tvm_worker" in b:
        trt_core = a["llm_prefill"].p50_ms + a["denoise_total"].p50_ms
        out["core_llm_denoise_vs_tvm_worker"] = float(b["tvm_worker"].p50_ms - trt_core)
    if "llm_prefill" in a and "denoise_total" in a and "llm_prefill" in b and "denoise_total" in b:
        trt_core = a["llm_prefill"].p50_ms + a["denoise_total"].p50_ms
        tvm_core = b["llm_prefill"].p50_ms + b["denoise_total"].p50_ms
        out["core_llm_denoise"] = float(tvm_core - trt_core)
    return out


def run_bench(task: TaskConfig) -> BenchReport:
    """Run stage latency bench for configured backends; return structured report."""
    cfg = task.bench
    backends = [str(b).strip().lower() for b in (cfg.backends or ["trt", "tvm"])]
    observation = _load_observation(task, cfg.sample_index)

    meta: dict[str, Any] = {
        "architecture": task.architecture,
        "platform": task.platform,
        "backends": backends,
        "warmup": int(cfg.warmup),
        "runs": int(cfg.runs),
        "sync": cfg.sync,
        "sample_index": int(cfg.sample_index),
        "num_steps": int(task.infer.num_steps or task.model_overrides.get("num_denoise_steps") or 10),
        "tvm_loop": bool(
            task.bench.tvm_loop
            if task.bench.tvm_loop is not None
            else task.model_overrides.get("tvm_loop", True)
        ),
        "tvm_cuda_graph": bool(task.model_overrides.get("tvm_cuda_graph", False)),
        "tvm_dtype": str(task.model_overrides.get("tvm_dtype") or ""),
        "trt_cuda_graph": bool(task.evaluate.trt_cuda_graph),
        "noise": task.evaluate.noise,
        "noise_seed": int(task.evaluate.noise_seed),
    }

    results: dict[str, dict[str, StageStats]] = {}
    for backend in backends:
        logger.info("bench backend=%s building…", backend)
        runner = _build_runner(task, backend)
        try:
            noise = None
            if hasattr(runner, "noise_for_sample"):
                noise = runner.noise_for_sample(int(cfg.sample_index))
            stages = _bench_one(
                runner,
                observation,
                noise,
                warmup=int(cfg.warmup),
                runs=int(cfg.runs),
                sync=str(cfg.sync or "cuda_event"),
            )
            results[backend] = stages
        finally:
            _free_runner(runner)

    delta = _compute_delta(results)
    report = BenchReport(meta=meta, backends=results, delta=delta)

    out_path = Path(cfg.output or f"{task.output_dir}/bench.json").expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("bench report written: %s", out_path)
    report.meta["output"] = str(out_path)
    return report
