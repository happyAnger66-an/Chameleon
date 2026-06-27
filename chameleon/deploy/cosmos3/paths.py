"""cosmos3 部署路径解析 — ONNX / engine 文件名与 build_cfg。

对照 deploy/paths.py 的 pi05 版本，但 cosmos3 reference 导出不强制 checkpoint。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chameleon.config.schema import TaskConfig

COSMOS3_ONNX_FILES: dict[str, str] = {
    "vae_encode": "vae_encode.onnx",
    "text_embed": "text_embed.onnx",
    "dit": "dit.onnx",
    "vae_decode": "vae_decode.onnx",
}

COSMOS3_ENGINE_FILES: dict[str, str] = {
    "vae_encode": "vae_encode.engine",
    "text_embed": "text_embed.engine",
    "dit": "dit.engine",
    "vae_decode": "vae_decode.engine",
}

DEFAULT_BUILD_CFGS: dict[str, str] = {
    "vae_encode": "cosmos3_vae_encode_build_cfg.py",
    "text_embed": "cosmos3_text_embed_build_cfg.py",
    "dit": "cosmos3_dit_build_cfg.py",
    "vae_decode": "cosmos3_vae_decode_build_cfg.py",
}


def _chameleon_project_root() -> Path:
    # chameleon/deploy/cosmos3/paths.py -> Chamleon/
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Cosmos3DeployPaths:
    export_dir: Path
    engine_dir: Path
    build_cfg_dir: Path


def resolve_build_cfg_dir(task: TaskConfig) -> Path:
    if task.deploy.build_cfg_dir:
        return Path(task.deploy.build_cfg_dir).expanduser().resolve()
    return (_chameleon_project_root() / "configs" / "build_configs").resolve()


def resolve_engine_dir(task: TaskConfig) -> Path:
    if task.deploy.engine_dir:
        return Path(task.deploy.engine_dir).expanduser().resolve()
    return (Path(task.output_dir).expanduser().resolve() / "engines")


def resolve_cosmos3_paths(task: TaskConfig) -> Cosmos3DeployPaths:
    export_dir = Path(task.deploy.export_dir or f"{task.output_dir}/onnx").expanduser()
    return Cosmos3DeployPaths(
        export_dir=export_dir.resolve(),
        engine_dir=resolve_engine_dir(task),
        build_cfg_dir=resolve_build_cfg_dir(task),
    )


def stage_onnx_path(paths: Cosmos3DeployPaths, stage: str) -> Path:
    name = COSMOS3_ONNX_FILES.get(stage)
    if not name:
        raise KeyError(f"Unknown cosmos3 stage {stage!r}.")
    return paths.export_dir / name


def stage_engine_path(paths: Cosmos3DeployPaths, stage: str) -> Path:
    name = COSMOS3_ENGINE_FILES.get(stage)
    if not name:
        raise KeyError(f"Unknown cosmos3 stage {stage!r}.")
    return paths.engine_dir / name


def resolve_build_cfg_path(task: TaskConfig, stage: str, paths: Cosmos3DeployPaths) -> Path:
    rel = task.deploy.build_cfgs.get(stage) or DEFAULT_BUILD_CFGS.get(stage)
    if not rel:
        raise KeyError(f"No default build_cfg for cosmos3 stage {stage!r}; set deploy.build_cfgs.")
    raw = Path(rel)
    if raw.is_absolute():
        if not raw.is_file():
            raise FileNotFoundError(f"build_cfg not found: {raw}")
        return raw.resolve()
    candidate = (paths.build_cfg_dir / raw).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"build_cfg for stage {stage!r} not found: {candidate} (build_cfg_dir={paths.build_cfg_dir})"
    )
