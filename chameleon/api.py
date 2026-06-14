"""High-level programmatic API used by the CLI and the workflow runner."""

from __future__ import annotations

import logging
from typing import Any

import torch

from chameleon.architectures.registry import get_architecture
from chameleon.config.schema import TaskConfig
from chameleon.core.artifact import Artifact, Manifest
from chameleon.core.context import CompileContext, RunContext
from chameleon.core.platform import get_platform
from chameleon.frontend.base import get_graph_capture
from chameleon.models.base import ModelAdapter, get_model_adapter
from chameleon.quantization.base import QuantConfig
from chameleon.quantization.calibrate.base import TensorCalibrator
from chameleon.quantization.registry import get_quant_method
from chameleon.runtime.orchestrator import InferenceSession

logger = logging.getLogger(__name__)


def build_adapter(task: TaskConfig, device: str = "cpu") -> ModelAdapter:
    adapter_cls = get_model_adapter(task.model)
    config = adapter_cls.make_config(task.model_overrides)
    return adapter_cls(config).build(device)


def _resolve_torch_device(requested: str) -> str:
    """Fall back to CPU when a requested accelerator is not available.

    Lets a config targeting e.g. ``nvidia_orin`` still run the PyTorch reference
    path on a CPU-only box.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available; running the reference path on CPU.")
        return "cpu"
    return requested


def _run_context(task: TaskConfig) -> RunContext:
    platform = get_platform(task.platform)
    options: dict[str, Any] = {}
    if task.infer.num_steps is not None:
        options["num_steps"] = task.infer.num_steps
    requested_device = task.infer.torch_device or platform.torch_device
    options["torch_device"] = _resolve_torch_device(requested_device)
    return RunContext(platform=platform, architecture=task.architecture, options=options)


def run_infer(task: TaskConfig) -> torch.Tensor:
    """Build a session and run a single inference, returning the action chunk."""
    ctx = _run_context(task)
    adapter = build_adapter(task, device=ctx.torch_device)
    observation = adapter.example_observation(task.infer.batch_size, device=ctx.torch_device)
    session = InferenceSession(adapter, ctx, stage_runtimes=task.stage_runtimes).build()
    return session.infer(observation)


def run_quantize(task: TaskConfig, adapter: ModelAdapter, manifest: Manifest) -> None:
    platform = get_platform(task.platform)
    for step in task.quantize:
        module = adapter.stage_module(step.stage)
        method = get_quant_method(step.method)
        # MVP calibrator: a couple of example observations exercising the stage.
        obs = adapter.example_observation(task.infer.batch_size)
        samples = _stage_calib_samples(adapter, step.stage, obs)
        calibrator = TensorCalibrator(samples)
        qcfg = QuantConfig(
            method=step.method,
            weight_dtype=step.weight_dtype,
            activation_dtype=step.activation_dtype,
            kv_cache_dtype=step.kv_cache_dtype,
            options=step.options,
        )
        _, meta = method.quantize(module, calibrator, platform, qcfg)
        manifest.add(
            Artifact(
                kind="quantized",
                stage=step.stage,
                platform=platform.name,
                metadata={"method": step.method, "dtypes": meta.component_dtypes},
            )
        )
        logger.info("Quantized stage %s with %s -> %s", step.stage, step.method, meta.component_dtypes)


def run_compile(task: TaskConfig, adapter: ModelAdapter, manifest: Manifest) -> None:
    from chameleon.compile.base import get_compiler

    platform = get_platform(task.platform)
    ctx = CompileContext(platform=platform, output_dir=task.output_dir, architecture=task.architecture)
    capture = get_graph_capture("onnx")
    compiler = get_compiler(platform.compiler)
    for step in task.compile:
        try:
            module = adapter.stage_module(step.stage)
            obs = adapter.example_observation(task.infer.batch_size)
            example_inputs = _stage_example_inputs(adapter, step.stage, obs)
            onnx_path = f"{task.output_dir}/{step.stage}.onnx"
            graph = capture.capture(module, example_inputs, stage=step.stage, output_path=onnx_path)
            manifest.add(graph)
            engine = compiler.compile(graph, None, ctx, step.options)
            manifest.add(engine)
            logger.info("Compiled stage %s -> %s", step.stage, engine.path)
        except Exception as exc:  # noqa: BLE001
            # Scaffold backends, partial toolchains and export edge-cases degrade
            # gracefully so the rest of the workflow (e.g. the reference infer
            # path) still runs. Bring-up phases will harden each stage.
            logger.warning(
                "Compile skipped for stage %s on %s: %s", step.stage, platform.compiler, exc
            )
            manifest.add(
                Artifact(
                    kind="compile_skipped",
                    stage=step.stage,
                    platform=platform.name,
                    metadata={"compiler": platform.compiler, "reason": str(exc)[:200]},
                )
            )


def _stage_example_inputs(adapter: ModelAdapter, stage: str, obs: dict[str, Any]):
    """Positional example inputs matching each pi05 stage's forward signature."""
    if stage == "vit":
        return (obs["images"],)
    if stage == "llm_prefix":
        img_tokens = adapter.stage_module("vit")(obs["images"])
        return (img_tokens, obs["lang_tokens"])
    if stage == "action_expert":
        cfg = adapter.config
        b = obs["state"].shape[0]
        img_tokens = adapter.stage_module("vit")(obs["images"])
        prefix = adapter.stage_module("llm_prefix")(img_tokens, obs["lang_tokens"])
        x_t = torch.randn(b, cfg.action_horizon, cfg.action_dim)
        time_emb = torch.randn(b, getattr(adapter, "time_embed_dim", cfg.action_dim))
        return (obs["state"], prefix, x_t, time_emb)
    raise KeyError(f"No example inputs defined for stage {stage!r}.")


def _stage_calib_samples(adapter: ModelAdapter, stage: str, obs: dict[str, Any], n: int = 2):
    return [_stage_example_inputs(adapter, stage, obs) for _ in range(n)]
