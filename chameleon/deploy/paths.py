"""部署路径解析 — ONNX / engine 输出目录与 build_cfg。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chameleon.config.schema import TaskConfig

PI05_ONNX_FILES: dict[str, str] = {
    "vit": "vit.onnx",
    "llm": "llm.onnx",
    "expert": "expert.onnx",
    "denoise": "denoise.onnx",
    "embed_prefix": "embed_prefix.onnx",
}

PI05_ENGINE_FILES: dict[str, str] = {
    "vit": "vit.engine",
    "llm": "llm.engine",
    "expert": "expert.engine",
    "denoise": "denoise.engine",
    "embed_prefix": "embed_prefix.engine",
}

DEFAULT_BUILD_CFGS: dict[str, str] = {
    "vit": "vit_build_cfg.py",
    "llm": "llm_build_cfg.py",
    "expert": "expert_build_cfg.py",
    "denoise": "denoise_step_build_cfg.py",
    "embed_prefix": "embed_prefix_build_cfg.py",
}


def _chameleon_project_root() -> Path:
    # chameleon/deploy/paths.py -> Chamleon/
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DeployPaths:
    export_dir: Path
    engine_dir: Path
    checkpoint_dir: Path
    build_cfg_dir: Path
    train_config: str | None


def resolve_checkpoint_dir(task: TaskConfig) -> Path:
    deploy = task.deploy
    if deploy.checkpoint_dir:
        return Path(deploy.checkpoint_dir).expanduser().resolve()

    ckpt = task.evaluate.checkpoint_dir
    if ckpt:
        return Path(ckpt).expanduser().resolve()

    override = task.model_overrides.get("checkpoint")
    if override:
        path = Path(str(override)).expanduser()
        if path.is_file():
            return path.parent.resolve()
        if path.is_dir():
            return path.resolve()

    raise ValueError(
        "deploy.checkpoint_dir / evaluate.checkpoint_dir / model_overrides.checkpoint "
        "required for pi05 export/build."
    )


def resolve_train_config(task: TaskConfig) -> str | None:
    deploy = task.deploy
    if deploy.train_config:
        return deploy.train_config
    if task.data.openpi_config:
        return task.data.openpi_config
    return None


def resolve_build_cfg_dir(task: TaskConfig) -> Path:
    if task.deploy.build_cfg_dir:
        return Path(task.deploy.build_cfg_dir).expanduser().resolve()
    return (_chameleon_project_root() / "configs" / "build_configs").resolve()


def resolve_deploy_paths(task: TaskConfig) -> DeployPaths:
    export_dir = Path(task.deploy.export_dir or f"{task.output_dir}/onnx").expanduser()
    engine_dir = Path(task.deploy.engine_dir or f"{task.output_dir}/engines").expanduser()

    return DeployPaths(
        export_dir=export_dir.resolve(),
        engine_dir=engine_dir.resolve(),
        checkpoint_dir=resolve_checkpoint_dir(task),
        build_cfg_dir=resolve_build_cfg_dir(task),
        train_config=resolve_train_config(task),
    )


def resolve_build_cfg_path(task: TaskConfig, stage: str, paths: DeployPaths) -> Path:
    deploy = task.deploy
    rel = deploy.build_cfgs.get(stage) or DEFAULT_BUILD_CFGS.get(stage)
    if not rel:
        raise KeyError(f"No default build_cfg for stage {stage!r}; set deploy.build_cfgs.")

    raw = Path(rel)
    if raw.is_absolute():
        if not raw.is_file():
            raise FileNotFoundError(f"build_cfg not found: {raw}")
        return raw.resolve()

    candidate = (paths.build_cfg_dir / raw).resolve()
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"build_cfg for stage {stage!r} not found: {candidate} "
        f"(build_cfg_dir={paths.build_cfg_dir})"
    )


def stage_onnx_path(paths: DeployPaths, stage: str) -> Path:
    name = PI05_ONNX_FILES.get(stage)
    if not name:
        raise KeyError(f"Unknown pi05 stage {stage!r}.")
    return paths.export_dir / name


def stage_engine_path(paths: DeployPaths, stage: str) -> Path:
    name = PI05_ENGINE_FILES.get(stage)
    if not name:
        raise KeyError(f"Unknown pi05 stage {stage!r}.")
    return paths.engine_dir / name


def stage_profile_path(profile_dir: Path, stage: str) -> Path:
    return profile_dir / f"{stage}.profile.json"


def stage_profile_log_path(profile_dir: Path, stage: str) -> Path:
    return profile_dir / f"{stage}.trtexec.log"


def stage_layer_info_path(profile_dir: Path, stage: str) -> Path:
    return profile_dir / f"{stage}.layer_info.json"


def stage_times_path(profile_dir: Path, stage: str) -> Path:
    return profile_dir / f"{stage}.times.json"


def resolve_profile_dir(task: TaskConfig) -> Path:
    if task.profile.profile_dir:
        return Path(task.profile.profile_dir).expanduser().resolve()
    return (Path(task.output_dir).expanduser().resolve() / "profiles")
