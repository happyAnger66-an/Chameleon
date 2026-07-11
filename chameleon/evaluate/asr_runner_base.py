"""ASR runner 抽象 — 与 PolicyRunner 平行，契约为音频→文本。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chameleon.config.schema import TaskConfig
from chameleon.core.registry import Registry


@dataclass
class AsrResult:
    language: str
    text: str
    raw_text: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class AsrRunner(ABC):
    @classmethod
    @abstractmethod
    def from_task(cls, task: TaskConfig) -> "AsrRunner":
        ...

    def build(self) -> "AsrRunner":
        return self

    @abstractmethod
    def transcribe(
        self,
        audio: str | np.ndarray,
        *,
        context: str = "",
        language: str | None = None,
        sample_rate: int | None = None,
    ) -> AsrResult:
        ...


ASR_RUNNER_REGISTRY: Registry[str, type[AsrRunner]] = Registry("asr_runner")


def register_asr_runner(name: str, cls: type[AsrRunner], *, override: bool = False):
    return ASR_RUNNER_REGISTRY.register(name, cls, override=override)


def build_asr_runner(task: TaskConfig) -> AsrRunner:
    name = getattr(task.evaluate, "policy_runner", None) or "qwen3_asr_edgellm"
    cls = ASR_RUNNER_REGISTRY.get(name)
    return cls.from_task(task).build()


def list_asr_runners() -> list[str]:
    return ASR_RUNNER_REGISTRY.keys()


def is_asr_runner_name(name: str | None) -> bool:
    return bool(name) and name in ASR_RUNNER_REGISTRY
