"""统一任务配置 — pydantic + YAML 驱动的 quantize / compile / infer 描述。

作用：
    定义 TaskConfig 及子模型（QuantizeStep、CompileStep、InferConfig），
    描述 architecture / platform / actions / stage_runtimes / model_overrides
    等。TaskConfig.load() 从 YAML 加载并校验。

架构位置：
    入口/编排层 — 全框架配置的单一来源，被 cli.py、api.py、
    workflows/runner.py、profile/latency.py 消费。configs/*.yaml 为本
    schema 的实例。
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
    use_compiled_engines: bool = False
    """When true, infer consumes the engines produced by the compile step
    (per stage) via the platform runtime, instead of the PyTorch reference path."""
    cuda_graph: bool = False
    """Capture/replay a CUDA graph per engine (TensorRT runtime; static shapes)."""


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
