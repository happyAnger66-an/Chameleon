"""高层程序化 API — 量化 / 编译 / 推理的核心调度函数。

作用：
    提供 build_adapter、run_quantize、run_compile、run_infer 等函数，
    将 TaskConfig 翻译为各子系统的具体调用，并维护 per-stage I/O 命名约定。

架构位置：
    入口/编排层 — 位于 CLI / WorkflowRunner 与各子系统之间，是薄编排层
    的实际执行体。上游：config/schema（TaskConfig）、models（ModelAdapter）；
    下游：frontend（图捕获）、compile（编译）、quantization（量化）、
    runtime/orchestrator（推理会话）。
"""

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
    options["enable_cuda_graph"] = task.infer.cuda_graph
    return RunContext(platform=platform, architecture=task.architecture, options=options)


def run_infer(
    task: TaskConfig,
    *,
    stage_artifacts: dict[str, Artifact] | None = None,
    stage_runtimes: dict[str, str] | None = None,
) -> torch.Tensor:
    """Build a session and run a single inference, returning the action chunk.

    ``stage_artifacts`` injects compiled engines (closing the compile->infer loop);
    ``stage_runtimes`` overrides the per-stage runtime (e.g. ``tensorrt`` for
    stages that have an engine), otherwise the task's ``stage_runtimes`` is used.
    """
    ctx = _run_context(task)
    adapter = build_adapter(task, device=ctx.torch_device)
    observation = adapter.example_observation(task.infer.batch_size, device=ctx.torch_device)
    runtimes = stage_runtimes if stage_runtimes is not None else task.stage_runtimes
    session = InferenceSession(
        adapter, ctx, stage_runtimes=runtimes, stage_artifacts=stage_artifacts
    ).build()
    return session.infer(observation)


def run_eval(task: TaskConfig):
    """在真实 LeRobot 数据上评测 pi05：逐帧对比预测动作与 ground-truth。

    通过 ``evaluate.policy_runner`` 选择后端（``openpi`` / ``chameleon``），
    复用 dataloader + PolicyRunner registry。
    ``evaluate.viewer`` 为 ``webui`` / ``both`` 时启动 WebSocket 服务，
    推理线程仅非阻塞投递事件，JPEG 编码在 asyncio 消费侧完成。
    """
    from chameleon.evaluate.task_utils import sync_eval_num_samples

    sync_eval_num_samples(task)

    viewer = (task.evaluate.viewer or "console").strip().lower()
    if viewer in ("webui", "both"):
        from chameleon.evaluate.viewers.webui.server import run_eval_webui

        return run_eval_webui(task)

    from chameleon.dataloader import build_dataset_from_config
    from chameleon.evaluate import evaluate_lerobot
    from chameleon.evaluate.runner_base import build_policy_runner
    from chameleon.evaluate.viewers.base import build_eval_viewer

    data_cfg = task.data
    if not getattr(data_cfg, "dataset", None):
        raise ValueError("eval 需要 task.data.dataset（如 pi05_libero）。")

    data_source = build_dataset_from_config(data_cfg)
    data_source.build()
    runner = build_policy_runner(task)

    repo_id = data_cfg.repo_id or getattr(data_source, "repo_id", "") or ""
    action_horizon = int(getattr(data_source, "action_horizon", 10) or 10)
    action_dim = int(getattr(data_source, "action_dim", 7) or 7)
    start_index = int(getattr(data_cfg, "start_index", 0) or 0)
    num_samples = int(task.evaluate.num_samples)

    import uuid

    run_id = uuid.uuid4().hex[:12]
    meta = {
        "type": "meta",
        "run_id": run_id,
        "repo_id": repo_id,
        "backend": task.evaluate.policy_runner,
        "compare_mode": False,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "start_index": start_index,
        "end_index_exclusive": start_index + num_samples,
    }
    sink = build_eval_viewer(
        task,
        run_id=run_id,
        repo_id=repo_id,
        action_horizon=action_horizon,
        action_dim=action_dim,
        num_samples=num_samples,
        start_index=start_index,
    )

    return evaluate_lerobot(
        data_source,
        runner,
        num_samples=num_samples,
        stride=task.evaluate.stride,
        compare_horizon=task.evaluate.compare_horizon,
        event_sink=sink,
        run_meta=meta,
        run_id=run_id,
    )


def get_dataset_openpi_config(dataset_name: str) -> str:
    """从数据集 spec 取 openpi config 名（dataloader registry）。"""
    from chameleon.dataloader import get_dataset_spec

    return get_dataset_spec(dataset_name).openpi_config


def run_quantize(task: TaskConfig, adapter: ModelAdapter, manifest: Manifest) -> None:
    platform = get_platform(task.platform)
    device = getattr(adapter, "_device", "cpu")
    for step in task.quantize:
        module = adapter.stage_module(step.stage)
        method = get_quant_method(step.method)
        # MVP calibrator: a couple of example observations exercising the stage.
        obs = adapter.example_observation(task.infer.batch_size, device=device)
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


def run_export(task: TaskConfig, manifest: Manifest) -> dict[str, Artifact]:
    """Export ONNX graphs for deployment.

    ``deploy.backend=pi05`` uses Chameleon built-in pi05 stage exporters.
    """
    from chameleon.deploy.backends import is_pi05_deploy_backend

    backend = (task.deploy.backend or "reference").strip().lower()
    if is_pi05_deploy_backend(backend):
        from chameleon.deploy.pi05_openpi import run_pi05_export

        return run_pi05_export(task, manifest)
    if backend == "reference":
        raise NotImplementedError(
            "reference export is handled inside run_compile (capture -> ONNX). "
            "Use actions: [compile] with deploy.backend=reference, or set "
            "deploy.backend=pi05 for real pi05 ONNX export."
        )
    raise ValueError(f"Unknown deploy.backend {backend!r}.")


def run_deploy_build(task: TaskConfig, manifest: Manifest) -> dict[str, Artifact]:
    """Build TRT engines from exported ONNX (pi05 deploy path)."""
    from chameleon.deploy.backends import is_pi05_deploy_backend

    backend = (task.deploy.backend or "reference").strip().lower()
    if is_pi05_deploy_backend(backend):
        from chameleon.deploy.pi05_openpi import run_pi05_build

        return run_pi05_build(task, manifest)
    raise ValueError(
        f"run_deploy_build requires deploy.backend=pi05 (got {backend!r}). "
        "Use run_compile for the reference adapter path."
    )


def run_trt_profile(task: TaskConfig, manifest: Manifest) -> dict[str, Artifact]:
    """Profile compiled TRT engines with trtexec and write/serve layer profile WebUI."""
    from chameleon.deploy.trt_profile import run_trt_profile as _run

    return _run(task, manifest)


def run_compile(task: TaskConfig, adapter: ModelAdapter, manifest: Manifest) -> dict[str, Artifact]:
    """Compile each stage; returns the successfully built engine artifacts by stage."""
    from chameleon.compile.base import get_compiler

    platform = get_platform(task.platform)
    ctx = CompileContext(platform=platform, output_dir=task.output_dir, architecture=task.architecture)
    capture = get_graph_capture("onnx")
    compiler = get_compiler(platform.compiler)
    engines: dict[str, Artifact] = {}
    device = getattr(adapter, "_device", "cpu")
    for step in task.compile:
        try:
            module = adapter.stage_module(step.stage)
            obs = adapter.example_observation(task.infer.batch_size, device=device)
            example_inputs = _stage_example_inputs(adapter, step.stage, obs)
            input_names, output_names = _stage_io_names(step.stage)
            onnx_path = f"{task.output_dir}/{step.stage}.onnx"
            graph = capture.capture(
                module,
                example_inputs,
                stage=step.stage,
                output_path=onnx_path,
                input_names=input_names,
                output_names=output_names,
            )
            manifest.add(graph)
            engine = compiler.compile(graph, None, ctx, step.options)
            manifest.add(engine)
            engines[step.stage] = engine
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
    return engines


# Canonical per-stage I/O tensor names. The input order matches both the stage
# forward signature and the order the orchestrator passes values, so the TRT
# runtime can bind positionally.
_STAGE_IO_NAMES: dict[str, tuple[list[str], list[str]]] = {
    "vit": (["images"], ["output"]),
    "llm_prefix": (["img_tokens", "lang_tokens"], ["output"]),
    "action_expert": (["state", "prefix_memory", "x_t", "time_emb"], ["output"]),
}


def _stage_io_names(stage: str) -> tuple[list[str] | None, list[str] | None]:
    names = _STAGE_IO_NAMES.get(stage)
    return names if names is not None else (None, None)


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
        device = obs["state"].device
        img_tokens = adapter.stage_module("vit")(obs["images"])
        prefix = adapter.stage_module("llm_prefix")(img_tokens, obs["lang_tokens"])
        x_t = torch.randn(b, cfg.action_horizon, cfg.action_dim, device=device)
        time_emb = torch.randn(b, getattr(adapter, "time_embed_dim", cfg.action_dim), device=device)
        return (obs["state"], prefix, x_t, time_emb)
    raise KeyError(f"No example inputs defined for stage {stage!r}.")


def _stage_calib_samples(adapter: ModelAdapter, stage: str, obs: dict[str, Any], n: int = 2):
    return [_stage_example_inputs(adapter, stage, obs) for _ in range(n)]
