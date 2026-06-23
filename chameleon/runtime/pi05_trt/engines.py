"""pi05 TRT stage engine 加载。"""

from __future__ import annotations

import logging
from pathlib import Path

from chameleon.config.schema import TaskConfig, TrtEngineNames
from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.core.platform import get_platform
from chameleon.runtime.base import Engine
from chameleon.runtime.tensorrt.backend import TensorRTRuntime

logger = logging.getLogger(__name__)


def load_trt_stage_engines(
    task: TaskConfig,
    *,
    engine_dir: Path,
    engines: TrtEngineNames,
    device: str,
    enable_cuda_graph: bool = False,
) -> dict[str, Engine]:
    """加载 vit / llm / denoise TRT engine（expert 已含在 denoise 图中）。"""
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
    for stage, name in (
        ("vit", engines.vit),
        ("llm", engines.llm),
        ("denoise", engines.denoise),
    ):
        path = engine_dir / name
        artifact = Artifact(
            kind="engine",
            stage=stage,
            platform=platform.name,
            path=str(path),
        )
        loaded[stage] = runtime.load(artifact, ctx)
        logger.info("Loaded TRT engine stage=%s path=%s", stage, path)
    return loaded
