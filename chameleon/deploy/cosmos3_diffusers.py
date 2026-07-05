"""cosmos3 ONNX 导出与 TensorRT engine 构建（Chameleon 内置实现）。

对照 deploy/pi05_openpi.py：run_cosmos3_export 导出 vae_encode / text_embed / dit /
vae_decode 子图，run_cosmos3_build 用各 stage 的 build_cfg 构建 TRT engine。导出
从 Cosmos3Adapter 取模块，reference 模型可在 CPU 上离线导出（无需 diffusers 权重）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from chameleon.config.schema import ExportStep, TaskConfig
from chameleon.core.artifact import Artifact
from chameleon.deploy.build_cfg import load_build_cfg
from chameleon.deploy.cosmos3.export import COSMOS3_STAGES, export_cosmos3_stages
from chameleon.deploy.cosmos3.loader import load_cosmos3_adapter
from chameleon.deploy.cosmos3.paths import (
    Cosmos3DeployPaths,
    resolve_build_cfg_path,
    resolve_cosmos3_paths,
    stage_engine_path,
    stage_onnx_path,
)

logger = logging.getLogger(__name__)

_DEFAULT_EXPORT_STAGES = ("vae_encode", "text_embed", "dit", "vae_decode")


def _tensorrt_version() -> str | None:
    try:
        import tensorrt as trt

        return str(getattr(trt, "__version__", None) or "")
    except ImportError:
        return None


def iter_export_steps(task: TaskConfig) -> list[ExportStep]:
    if task.export:
        return list(task.export)
    return [ExportStep(stage=s) for s in _DEFAULT_EXPORT_STAGES]


def run_cosmos3_export(task: TaskConfig, manifest) -> dict[str, Artifact]:
    paths = resolve_cosmos3_paths(task)
    steps = iter_export_steps(task)
    for step in steps:
        if step.stage not in COSMOS3_STAGES:
            raise ValueError(
                f"Unknown cosmos3 export stage {step.stage!r}; expected one of {COSMOS3_STAGES}."
            )

    # cosmos3 reference modules are ONNX-traceable on CPU; real diffusers submodules
    # (16B MoT + Wan VAE) require CUDA — honour the configured infer device for those.
    use_reference = bool(task.model_overrides.get("use_reference", True))
    if use_reference:
        device = "cpu"
    else:
        device = task.infer.torch_device or "cuda"
    adapter = load_cosmos3_adapter(task, device=device)
    is_real = bool(getattr(adapter, "_is_real_diffusers", False))
    logger.info(
        "cosmos3 export path: reference=%s device=%s (requested use_reference=%s)",
        adapter.config.use_reference,
        device,
        use_reference,
    )

    stage_options = {step.stage: dict(step.options) for step in steps}
    exported = export_cosmos3_stages(
        [step.stage for step in steps],
        adapter=adapter,
        export_dir=paths.export_dir,
        device=device,
        stage_options=stage_options,
    )

    artifacts: dict[str, Artifact] = {}
    for stage, out_path in exported.items():
        onnx_path = str(out_path)
        artifact = Artifact(
            kind="onnx",
            stage=stage,
            platform=task.platform,
            path=onnx_path,
            metadata={"backend": "cosmos3", "reference": adapter.config.use_reference},
        )
        manifest.add(artifact)
        artifacts[stage] = artifact
        logger.info("Exported cosmos3 stage %s -> %s", stage, onnx_path)
    return artifacts


def build_cosmos3_stage_engine(
    task: TaskConfig,
    stage: str,
    *,
    paths: Cosmos3DeployPaths | None = None,
    build_cfg_path: str | Path | None = None,
    use_cudagraph: bool | None = None,
) -> str:
    from chameleon.deploy.trt_build import build_engine, validate_precision_matches_onnx

    paths = paths or resolve_cosmos3_paths(task)
    paths.engine_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(build_cfg_path) if build_cfg_path else resolve_build_cfg_path(task, stage, paths)
    build_cfg = load_build_cfg(cfg_path)

    onnx_path = stage_onnx_path(paths, stage)
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX not found for cosmos3 stage {stage!r}: {onnx_path}. Run export first."
        )

    engine_path = stage_engine_path(paths, stage)
    cudagraph = task.deploy.use_cudagraph if use_cudagraph is None else use_cudagraph
    precision = build_cfg.get("precision", "bf16")
    validate_precision_matches_onnx(str(onnx_path), precision)

    logger.info(
        "Building cosmos3 TRT engine stage=%s onnx=%s engine=%s build_cfg=%s",
        stage,
        onnx_path,
        engine_path,
        cfg_path,
    )
    build_engine(str(onnx_path), str(engine_path), cudagraph, **build_cfg)
    if not engine_path.is_file():
        raise FileNotFoundError(f"TRT build finished but engine missing: {engine_path}")
    return str(engine_path)


def run_cosmos3_build(task: TaskConfig, manifest) -> dict[str, Artifact]:
    if not task.compile:
        raise ValueError("compile steps required for cosmos3 TRT build.")

    paths = resolve_cosmos3_paths(task)
    artifacts: dict[str, Artifact] = {}

    for step in task.compile:
        if step.stage not in COSMOS3_STAGES:
            raise ValueError(
                f"Unknown compile stage {step.stage!r}; expected one of {COSMOS3_STAGES}."
            )
        build_cfg = step.options.get("build_cfg")
        engine_path = build_cosmos3_stage_engine(
            task,
            step.stage,
            paths=paths,
            build_cfg_path=build_cfg,
            use_cudagraph=step.options.get("use_cudagraph"),
        )
        artifact = Artifact(
            kind="engine",
            stage=step.stage,
            platform=task.platform,
            path=engine_path,
            metadata={
                "backend": "cosmos3",
                "build_cfg": str(build_cfg or resolve_build_cfg_path(task, step.stage, paths)),
                "tensorrt_version": _tensorrt_version(),
            },
        )
        manifest.add(artifact)
        artifacts[step.stage] = artifact
        logger.info("Built cosmos3 engine stage %s -> %s", step.stage, engine_path)
    return artifacts
