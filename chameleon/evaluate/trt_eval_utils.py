"""pi05 TRT 评测公共工具 — engine 校验与 WebUI meta。"""

from __future__ import annotations

from pathlib import Path

from chameleon.config.schema import TaskConfig, TrtEngineNames
from chameleon.deploy.paths import PI05_ENGINE_FILES, resolve_engine_dir


def resolve_trt_engine_names(task: TaskConfig) -> TrtEngineNames:
    ev = task.evaluate
    if ev.trt_engines is not None:
        return ev.trt_engines
    return TrtEngineNames(
        vit=PI05_ENGINE_FILES["vit"],
        llm=PI05_ENGINE_FILES["llm"],
        expert=PI05_ENGINE_FILES["expert"],
        denoise=PI05_ENGINE_FILES["denoise"],
    )


def validate_engine_files(engine_dir: Path, engines: TrtEngineNames) -> None:
    if not engine_dir.is_dir():
        raise FileNotFoundError(
            f"TRT engine 目录不存在: {engine_dir}。"
            "请先运行 export + compile（如 configs/pi05_libero_trt_deploy.yaml）。"
        )
    for stage, name in (
        ("vit", engines.vit),
        ("llm", engines.llm),
        ("expert", engines.expert),
        ("denoise", engines.denoise),
    ):
        path = engine_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少 stage {stage!r} engine: {path}。请先完成 TRT build。"
            )


def resolve_trt_precision(task: TaskConfig) -> str:
    return str(task.evaluate.precision or task.model_overrides.get("precision") or "bf16")


def tensorrt_meta(task: TaskConfig) -> dict[str, str]:
    engines = resolve_trt_engine_names(task)
    return {
        "engine_path": str(resolve_engine_dir(task)),
        "vit_engine": engines.vit,
        "llm_engine": engines.llm,
        "expert_engine": engines.expert,
        "denoise_engine": engines.denoise,
    }


def should_attach_tensorrt_meta(task: TaskConfig) -> bool:
    ev = task.evaluate
    return bool(ev.compare_mode or ev.policy_runner in ("trt_only", "pt_trt_compare"))
