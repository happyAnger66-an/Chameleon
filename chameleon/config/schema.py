"""Unified task configuration (pydantic + YAML).

A single :class:`TaskConfig` describes a quantize -> compile -> infer task,
replacing ``model_optimizer``'s mix of ``.py`` / JSON / argparse configs. It is
loaded from YAML and validated up-front.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class QuantizeStep(BaseModel):
    stage: str
    method: str = "fp8"
    weight_dtype: str = "int8"
    activation_dtype: str | None = None
    kv_cache_dtype: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class CompileStep(BaseModel):
    stage: str
    options: dict[str, Any] = Field(default_factory=dict)


class InferConfig(BaseModel):
    batch_size: int = 1
    num_steps: int | None = None
    torch_device: str | None = None


class TaskConfig(BaseModel):
    architecture: str = "pi05"
    model: str = "pi05"
    platform: str = "generic_cpu"
    output_dir: str = "output/chameleon_run"

    actions: list[str] = Field(default_factory=lambda: ["infer"])
    """Ordered subset of ``quantize | compile | infer``."""

    model_overrides: dict[str, Any] = Field(default_factory=dict)
    """Overrides applied to the model adapter config (e.g. action_dim)."""

    stage_runtimes: dict[str, str] = Field(default_factory=dict)
    """Per-stage runtime backend, enabling stage-level backend mixing."""

    quantize: list[QuantizeStep] = Field(default_factory=list)
    compile: list[CompileStep] = Field(default_factory=list)
    infer: InferConfig = Field(default_factory=InferConfig)

    @classmethod
    def load(cls, path: str | Path) -> "TaskConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)
