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
    # cosmos3 (and other generative) orchestrators read the generation config and
    # CFG scale off the run options; harmless for pi05 which ignores them.
    options["generate"] = task.generate
    options["mode"] = task.generate.mode
    options["guidance_scale"] = task.generate.guidance_scale
    return RunContext(platform=platform, architecture=task.architecture, options=options)


def _is_asr_task(task: TaskConfig) -> bool:
    """True when architecture/model/runner indicates ASR (qwen3_asr)."""
    arch = (task.architecture or "").strip().lower()
    model = (task.model or "").strip().lower()
    if arch.startswith("qwen3_asr") or model.startswith("qwen3_asr"):
        return True
    runner = getattr(task.evaluate, "policy_runner", None)
    if runner:
        from chameleon.evaluate import is_asr_runner_name

        return is_asr_runner_name(runner)
    return False


def run_infer(
    task: TaskConfig,
    *,
    stage_artifacts: dict[str, Artifact] | None = None,
    stage_runtimes: dict[str, str] | None = None,
) -> Any:
    """Build a session and run a single inference, returning the primary output.

    For action models (pi05, cosmos3 ``mode=action``) this is the action chunk
    ``[B, horizon, action_dim]``; for cosmos3 ``mode=video`` it is the generated
    video tensor ``[B, T, C, H, W]`` (or a latent proxy). For ``qwen3_asr`` this
    is a dict ``{text, language, raw_text, metrics}``.

    ``stage_artifacts`` injects compiled engines (closing the compile->infer loop);
    ``stage_runtimes`` overrides the per-stage runtime (e.g. ``tensorrt`` for
    stages that have an engine), otherwise the task's ``stage_runtimes`` is used.
    """
    if _is_asr_task(task):
        return _run_infer_asr(task)

    ctx = _run_context(task)
    adapter = build_adapter(task, device=ctx.torch_device)
    observation = adapter.example_observation(task.infer.batch_size, device=ctx.torch_device)
    runtimes = stage_runtimes if stage_runtimes is not None else task.stage_runtimes
    session = InferenceSession(
        adapter, ctx, stage_runtimes=runtimes, stage_artifacts=stage_artifacts
    ).build()
    return session.infer(observation)


def _run_infer_asr(task: TaskConfig) -> dict[str, Any]:
    """Single-utterance ASR via AsrRunner (Edge-LLM engines)."""
    from chameleon.evaluate import build_asr_runner

    audio = task.asr.audio or (task.model_overrides or {}).get("audio")
    if not audio:
        raise ValueError(
            "qwen3_asr infer needs asr.audio or model_overrides.audio (wav/mp3/flac path)."
        )
    runner = build_asr_runner(task)
    result = runner.transcribe(
        str(audio),
        context=task.asr.context,
        language=task.asr.language,
    )
    return {
        "text": result.text,
        "language": result.language,
        "raw_text": result.raw_text,
        "metrics": dict(result.metrics or {}),
    }


def run_eval(task: TaskConfig):
    """评测入口：ASR 走 WER；VLA 走 LeRobot 动作对比。

    通过 ``evaluate.policy_runner`` 先查 ``ASR_RUNNER_REGISTRY``，再查
    ``POLICY_RUNNER_REGISTRY``。VLA 路径下 ``evaluate.viewer`` 为
    ``webui`` / ``both`` 时启动 WebSocket 服务。
    """
    from chameleon.evaluate import is_asr_runner_name
    from chameleon.evaluate.task_utils import sync_eval_num_samples

    sync_eval_num_samples(task)

    runner_name = getattr(task.evaluate, "policy_runner", None)
    if _is_asr_task(task) or is_asr_runner_name(runner_name):
        return _run_eval_asr(task)

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

    from chameleon.evaluate.meta import build_eval_run_meta

    run_id = uuid.uuid4().hex[:12]
    meta = build_eval_run_meta(
        task,
        run_id=run_id,
        repo_id=repo_id,
        action_horizon=action_horizon,
        action_dim=action_dim,
        start_index=start_index,
        num_samples=num_samples,
    )
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


def _run_eval_asr(task: TaskConfig):
    """LibriSpeech / manifest ASR WER 评测。"""
    from chameleon.dataloader import build_dataset_from_config
    from chameleon.evaluate import build_asr_runner, evaluate_asr

    data_cfg = task.data
    if not getattr(data_cfg, "dataset", None):
        raise ValueError("ASR eval 需要 task.data.dataset（如 librispeech_test_clean）。")

    viewer = (task.evaluate.viewer or "console").strip().lower()
    if viewer in ("webui", "both"):
        logger.warning(
            "ASR eval viewer=%s: offline WER uses console; live text UI is via "
            "`chameleon stream` (fixed/pending regions).",
            viewer,
        )

    data_source = build_dataset_from_config(data_cfg)
    data_source.build()
    runner = build_asr_runner(task)
    return evaluate_asr(
        data_source,
        runner,
        num_samples=int(task.evaluate.num_samples),
        stride=int(task.evaluate.stride or 1),
    )


def run_stream(
    task: TaskConfig,
    *,
    on_update: Any | None = None,
) -> dict[str, Any]:
    """Streaming ASR V1: chunk re-feed + prefix rollback.

    Feeds ``asr.audio`` (or mic) in ``stream.chunk_size_sec`` windows, calling
    Edge-LLM on the accumulated wav each time. ``on_update`` receives dicts with
    ``fixed_text`` / ``pending_text`` for console or WebUI.
    """
    import tempfile
    from pathlib import Path

    from chameleon.evaluate import build_asr_runner
    from chameleon.runtime.edgellm.streaming import (
        AsrStreamingState,
        feed_pcm,
        finish_stream,
        load_audio_pcm,
    )

    if not _is_asr_task(task):
        raise ValueError("run_stream is only supported for architecture/model qwen3_asr")

    audio_path = task.asr.audio or (task.model_overrides or {}).get("audio")
    source = (task.stream.source or "file").strip().lower()
    if source == "mic":
        raise NotImplementedError(
            "stream.source=mic not implemented yet; use source=file with asr.audio"
        )
    if not audio_path:
        raise ValueError("stream needs asr.audio (wav path) when stream.source=file")

    runner = build_asr_runner(task)
    tokenizer = None
    ckpt = (task.model_overrides or {}).get("checkpoint")
    if ckpt:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("stream: tokenizer load failed (%s); whitespace rollback", exc)

    state = AsrStreamingState(
        unfixed_chunk_num=int(task.stream.unfixed_chunk_num),
        unfixed_token_num=int(task.stream.unfixed_token_num),
        chunk_size_sec=float(task.stream.chunk_size_sec),
        force_language=task.asr.language,
        base_context=task.asr.context or "",
        tokenizer=tokenizer,
    )

    def infer_fn(path: str, context: str, language: str | None) -> str:
        out = runner.transcribe(path, context=context, language=language)
        return out.raw_text or out.text

    pcm = load_audio_pcm(audio_path, sample_rate=state.sample_rate)
    # Feed in ~chunk-sized slices so buffer logic matches realtime arrival.
    step = state.chunk_size_samples
    with tempfile.TemporaryDirectory(prefix="chameleon_asr_stream_") as td:
        tmp = Path(td)
        for i in range(0, len(pcm), step):
            feed_pcm(state, pcm[i : i + step], infer_fn, tmp_dir=tmp, on_update=on_update)
        finish_stream(state, infer_fn, tmp_dir=tmp, on_update=on_update)

    return {
        "text": state.text,
        "language": state.language,
        "raw_text": state._raw_decoded,
        "chunk_id": state.chunk_id,
    }


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

    Dispatch is table-driven: ``deploy.backend`` resolves to a registered
    :class:`~chameleon.deploy.registry.DeployBackend` (pi05 / cosmos3 / ...).
    Adding a new architecture only requires registering a backend — no change here.
    """
    from chameleon.deploy.registry import resolve_deploy_backend

    return resolve_deploy_backend(task.deploy.backend).export(task, manifest)


def run_deploy_build(task: TaskConfig, manifest: Manifest) -> dict[str, Artifact]:
    """Build TRT engines from exported ONNX via the registered deploy backend."""
    from chameleon.deploy.registry import resolve_deploy_backend

    return resolve_deploy_backend(task.deploy.backend).build(task, manifest)


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
            input_names, output_names = _stage_io_names(step.stage, adapter)
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


def _stage_io_names(
    stage: str, adapter: ModelAdapter | None = None
) -> tuple[list[str] | None, list[str] | None]:
    # Model-specific adapters may declare their own stage I/O names (e.g. cosmos3).
    hook = getattr(adapter, "stage_io_names", None)
    if hook is not None:
        names = hook(stage)
        if names is not None and names[0] is not None:
            return names
    names = _STAGE_IO_NAMES.get(stage)
    return names if names is not None else (None, None)


def _stage_example_inputs(adapter: ModelAdapter, stage: str, obs: dict[str, Any]):
    """Positional example inputs matching each stage's forward signature."""
    # Defer to a model-specific hook when present (e.g. cosmos3 stages).
    hook = getattr(adapter, "stage_example_inputs", None)
    if hook is not None and stage not in ("vit", "llm_prefix", "action_expert"):
        return hook(stage, obs)
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
