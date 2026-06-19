"""真实 pi05 ONNX 导出与 TensorRT engine 构建（Chameleon 内置实现）。"""

from __future__ import annotations

import logging
from pathlib import Path

from chameleon.config.schema import ExportStep, TaskConfig
from chameleon.core.artifact import Artifact
from chameleon.deploy.build_cfg import load_build_cfg
from chameleon.deploy.paths import (
    DeployPaths,
    resolve_build_cfg_path,
    resolve_deploy_paths,
    stage_engine_path,
    stage_onnx_path,
)
from chameleon.deploy.pi05.export import PI05_STAGES, export_pi05_stages, export_stage
from chameleon.deploy.trt_build import build_engine, validate_precision_matches_onnx

logger = logging.getLogger(__name__)

PI05_OPENPI_STAGES = PI05_STAGES

_DEFAULT_EXPORT_STAGES = ("vit", "llm", "expert", "denoise")


def export_pi05_stage(
    task: TaskConfig,
    step: ExportStep,
    *,
    paths: DeployPaths | None = None,
    pi05_model=None,
) -> str:
    paths = paths or resolve_deploy_paths(task)
    paths.export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting pi05 stage %s -> %s (checkpoint=%s)",
        step.stage,
        paths.export_dir,
        paths.checkpoint_dir,
    )
    if pi05_model is None:
        from chameleon.deploy.pi05.loader import load_pi05_model

        pi05_model = load_pi05_model(
            str(paths.checkpoint_dir),
            paths.train_config,
            device="cpu",
        )
    out = export_stage(
        step.stage,
        pi05_model=pi05_model,
        export_dir=paths.export_dir,
        options=step.options,
    )
    onnx_path = Path(out)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX export finished but file missing: {onnx_path}")
    return str(onnx_path)


def build_pi05_stage_engine(
    task: TaskConfig,
    stage: str,
    *,
    paths: DeployPaths | None = None,
    build_cfg_path: str | Path | None = None,
    use_cudagraph: bool | None = None,
) -> str:
    paths = paths or resolve_deploy_paths(task)
    paths.engine_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(build_cfg_path) if build_cfg_path else resolve_build_cfg_path(task, stage, paths)
    build_cfg = load_build_cfg(cfg_path)

    onnx_path = stage_onnx_path(paths, stage)
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX not found for stage {stage!r}: {onnx_path}. Run export first."
        )

    engine_path = stage_engine_path(paths, stage)
    cudagraph = task.deploy.use_cudagraph if use_cudagraph is None else use_cudagraph
    precision = build_cfg.get("precision", "bf16")
    validate_precision_matches_onnx(str(onnx_path), precision)

    logger.info(
        "Building TRT engine stage=%s onnx=%s engine=%s build_cfg=%s",
        stage,
        onnx_path,
        engine_path,
        cfg_path,
    )
    build_engine(str(onnx_path), str(engine_path), cudagraph, **build_cfg)
    if not engine_path.is_file():
        raise FileNotFoundError(f"TRT build finished but engine missing: {engine_path}")
    return str(engine_path)


def iter_export_steps(task: TaskConfig) -> list[ExportStep]:
    if task.export:
        return list(task.export)
    return [ExportStep(stage=s) for s in _DEFAULT_EXPORT_STAGES]


def run_pi05_export(task: TaskConfig, manifest) -> dict[str, Artifact]:
    paths = resolve_deploy_paths(task)
    steps = iter_export_steps(task)
    for step in steps:
        if step.stage not in PI05_STAGES:
            raise ValueError(
                f"Unknown export stage {step.stage!r}; expected one of {PI05_STAGES}."
            )

    stage_options = {step.stage: dict(step.options) for step in steps}
    exported = export_pi05_stages(
        [step.stage for step in steps],
        checkpoint_dir=str(paths.checkpoint_dir),
        export_dir=paths.export_dir,
        train_config=paths.train_config,
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
            metadata={"backend": "pi05", "train_config": paths.train_config},
        )
        manifest.add(artifact)
        artifacts[stage] = artifact
        logger.info("Exported stage %s -> %s", stage, onnx_path)

    return artifacts


def run_pi05_build(task: TaskConfig, manifest) -> dict[str, Artifact]:
    if not task.compile:
        raise ValueError("compile steps required for pi05 TRT build.")

    paths = resolve_deploy_paths(task)
    artifacts: dict[str, Artifact] = {}

    for step in task.compile:
        if step.stage not in PI05_STAGES:
            raise ValueError(
                f"Unknown compile stage {step.stage!r}; expected one of {PI05_STAGES}."
            )
        build_cfg = step.options.get("build_cfg")
        engine_path = build_pi05_stage_engine(
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
                "backend": "pi05",
                "build_cfg": str(build_cfg or resolve_build_cfg_path(task, step.stage, paths)),
            },
        )
        manifest.add(artifact)
        artifacts[step.stage] = artifact
        logger.info("Built engine stage %s -> %s", step.stage, engine_path)

    return artifacts
