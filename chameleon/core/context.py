"""执行上下文 — 编译与运行时流水线的共享状态载体。

作用：
    定义 CompileContext（平台、输出目录、架构、进度回调）和
    RunContext（平台、架构、运行时选项如 num_steps / torch_device /
    cuda_graph），在各 Backend 调用间传递配置。

架构位置：
    核心抽象层 — 被 compile/base、runtime/base、api.py、
    workflows/runner.py 使用，连接平台抽象与各 Backend 实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from chameleon.core.platform import PlatformSpec

ProgressCallback = Callable[[str, float], None]
"""``(message, fraction_0_to_1) -> None`` progress reporter."""


def _noop_progress(message: str, fraction: float) -> None:  # pragma: no cover - trivial
    pass


@dataclass
class CompileContext:
    """Carries everything a compile/quantize step needs."""

    platform: PlatformSpec
    output_dir: Path
    architecture: str
    options: dict[str, Any] = field(default_factory=dict)
    on_progress: ProgressCallback = _noop_progress

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class RunContext:
    """Carries runtime configuration for a single inference session."""

    platform: PlatformSpec
    architecture: str
    options: dict[str, Any] = field(default_factory=dict)
    on_progress: ProgressCallback = _noop_progress

    @property
    def torch_device(self) -> str:
        # Honour an explicit override, otherwise fall back to the platform default.
        return str(self.options.get("torch_device", self.platform.torch_device))
