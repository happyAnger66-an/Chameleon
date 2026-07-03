"""Reference / adapter stage 统计 — 轻量参考模型与通用 ModelAdapter 路径。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from chameleon.models.base import ModelAdapter
from chameleon.models.pi05.reference import create_sinusoidal_pos_embedding
from chameleon.profile.execution_plan import ExecutionPlan


def _to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def build_reference_stage_module(adapter: ModelAdapter, stage: str) -> nn.Module:
    return adapter.stage_module(stage).eval()


def reference_stage_inputs(
    adapter: ModelAdapter,
    stage: str,
    shapes: dict[str, tuple[int, ...]],
    *,
    device: str,
) -> tuple[Any, ...]:
    if hasattr(adapter, "stage_example_inputs"):
        obs = adapter.example_observation(shapes[next(iter(shapes))][0], device=device)
        return adapter.stage_example_inputs(stage, obs)

    if stage == "vit":
        return (torch.randn(shapes["images"], device=device),)
    if stage == "llm_prefix":
        return (
            torch.randn(shapes["img_tokens"], device=device),
            torch.randint(0, 1000, shapes["lang_tokens"], device=device),
        )
    if stage == "action_expert":
        batch = shapes["state"][0]
        time_dim = adapter.time_embed_dim
        time_emb = create_sinusoidal_pos_embedding(
            torch.full((batch,), 1.0, device=device),
            time_dim,
            min_period=4e-3,
            max_period=4.0,
        )
        return (
            torch.randn(shapes["state"], device=device),
            torch.randn(shapes["prefix_memory"], device=device),
            torch.randn(shapes["x_t"], device=device),
            time_emb,
        )
    raise KeyError(f"Unknown reference stage {stage!r}.")


def prepare_reference_stage(
    adapter: ModelAdapter,
    stage: str,
    shapes: dict[str, tuple[int, ...]],
    *,
    plan: ExecutionPlan,
    device: str,
) -> tuple[nn.Module, tuple[Any, ...]]:
    module = build_reference_stage_module(adapter, stage)
    if device != "cpu":
        module = module.to(device)
    if hasattr(adapter, "stage_example_inputs"):
        obs = adapter.example_observation(plan.batch_size, device=device)
        inputs = adapter.stage_example_inputs(stage, obs)
    else:
        inputs = reference_stage_inputs(adapter, stage, shapes, device=device)
    inputs = _to_device(inputs, device)
    return module, inputs
