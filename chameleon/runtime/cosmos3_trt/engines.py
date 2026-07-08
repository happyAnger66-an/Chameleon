"""cosmos3 TRT stage engine 加载 — 对标 runtime/pi05_trt/engines.py。

加载 vae_encode / text_embed / dit / vae_decode 四个 TRT engine（固定 profile）。
engine 文件名取自 deploy/cosmos3/paths.py 的 COSMOS3_ENGINE_FILES。
"""

from __future__ import annotations

import logging
from pathlib import Path

from chameleon.config.schema import TaskConfig
from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.core.platform import get_platform
from chameleon.deploy.cosmos3.paths import COSMOS3_ENGINE_FILES
from chameleon.runtime.base import Engine
from chameleon.runtime.tensorrt.backend import TensorRTRuntime, memory_report

logger = logging.getLogger(__name__)


def load_trt_stage_engines(
    task: TaskConfig,
    *,
    engine_dir: Path,
    device: str,
    stages: tuple[str, ...] = ("vae_encode", "text_embed", "dit", "vae_decode"),
    enable_cuda_graph: bool = False,
) -> dict[str, Engine]:
    """加载 cosmos3 各 stage 的 TRT engine，返回 ``{stage: Engine}``。"""
    engine_dir = Path(engine_dir)
    platform = get_platform(task.platform)
    ctx = RunContext(
        platform=platform,
        architecture=task.architecture,
        options={
            "torch_device": device,
            "enable_cuda_graph": enable_cuda_graph,
        },
    )
    runtime = TensorRTRuntime()
    loaded: dict[str, Engine] = {}
    for stage in stages:
        fname = COSMOS3_ENGINE_FILES.get(stage)
        if fname is None:
            raise KeyError(f"Unknown cosmos3 stage {stage!r}; expected one of {sorted(COSMOS3_ENGINE_FILES)}.")
        path = engine_dir / fname
        size_gb = path.stat().st_size / 1e9 if path.is_file() else 0.0
        logger.warning(
            "Loading cosmos3 TRT engine stage=%s (%.1fGB plan) [%s]", stage, size_gb, memory_report()
        )
        artifact = Artifact(kind="engine", stage=stage, platform=platform.name, path=str(path))
        loaded[stage] = runtime.load(artifact, ctx)
        logger.warning("Loaded cosmos3 TRT engine stage=%s [%s]", stage, memory_report())
    return loaded


def validate_engine_files(engine_dir: Path, stages: tuple[str, ...]) -> None:
    """在构建 runner 前校验各 stage engine 文件存在。"""
    engine_dir = Path(engine_dir)
    missing = []
    for stage in stages:
        fname = COSMOS3_ENGINE_FILES.get(stage)
        if fname is None or not (engine_dir / fname).is_file():
            missing.append(f"{stage} ({fname})")
    if missing:
        raise FileNotFoundError(
            f"Missing cosmos3 TRT engine(s) in {engine_dir}: {', '.join(missing)}. Run export + compile first."
        )
