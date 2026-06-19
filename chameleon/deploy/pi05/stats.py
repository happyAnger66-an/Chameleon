"""pi05 deploy stage 统计 — 复用 Export wrapper 构建 forward。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from chameleon.deploy.pi05.denoise import Pi05DenoiseExport
from chameleon.deploy.pi05.expert import Pi05ExpertExport
from chameleon.deploy.pi05.llm import Pi05LlmExport
from chameleon.deploy.pi05.vit import Pi05VitExport
from chameleon.profile.shape_resolver import precision_to_dtype_bytes


def _rand_tensor(shape: tuple[int, ...], *, name: str, dtype_bytes: int, device: str) -> torch.Tensor:
    if "mask" in name or name == "prefix_pad_masks":
        return torch.ones(shape, dtype=torch.bool, device=device)
    if name in {"position_ids"}:
        return torch.randint(1, 1000, shape, dtype=torch.int64, device=device)
    if name == "timestep":
        return torch.full(shape, 1.0, dtype=torch.float32, device=device)
    if dtype_bytes == 2:
        dtype = torch.bfloat16
    elif dtype_bytes == 1:
        dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16
    else:
        dtype = torch.float32
    return torch.randn(shape, dtype=dtype, device=device)


def tensors_from_shapes(
    shapes: dict[str, tuple[int, ...]],
    *,
    dtype_bytes: int,
    device: str,
) -> dict[str, torch.Tensor]:
    return {
        name: _rand_tensor(tuple(shape), name=name, dtype_bytes=dtype_bytes, device=device)
        for name, shape in shapes.items()
    }


def build_pi05_stage_module(stage: str, pi05_model) -> nn.Module:
    if stage == "vit":
        return Pi05VitExport.from_pi05_model(pi05_model).eval()
    if stage == "llm":
        return Pi05LlmExport.from_pi05_model(pi05_model).eval()
    if stage == "expert":
        return Pi05ExpertExport.from_pi05_model(pi05_model).eval()
    if stage == "denoise":
        return Pi05DenoiseExport.from_pi05_model(pi05_model).eval()
    raise KeyError(f"Unknown pi05 deploy stage {stage!r}.")


def pi05_stage_inputs(stage: str, tensors: dict[str, torch.Tensor]) -> tuple[Any, ...]:
    if stage == "vit":
        return (tensors["pixel_values"],)
    if stage == "llm":
        return (
            tensors["inputs_embeds"],
            tensors["attention_mask"],
            tensors["position_ids"],
        )
    if stage == "expert":
        return (
            tensors["attention_mask"],
            tensors["position_ids"],
            tensors["inputs_embeds"],
            tensors["adarms_cond"],
            tensors["past_keys"],
            tensors["past_values"],
        )
    if stage == "denoise":
        return (
            tensors["prefix_pad_masks"],
            tensors["past_keys"],
            tensors["past_values"],
            tensors["x_t"],
            tensors["timestep"],
        )
    raise KeyError(f"Unknown pi05 deploy stage {stage!r}.")


def prepare_pi05_stage(
    stage: str,
    pi05_model,
    shapes: dict[str, tuple[int, ...]],
    *,
    precision: str,
    device: str,
) -> tuple[nn.Module, tuple[Any, ...]]:
    dtype_bytes = precision_to_dtype_bytes(precision)
    module = build_pi05_stage_module(stage, pi05_model)
    tensors = tensors_from_shapes(shapes, dtype_bytes=dtype_bytes, device=device)
    inputs = pi05_stage_inputs(stage, tensors)
    return module, inputs
