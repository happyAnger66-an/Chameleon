"""shape_resolver 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chameleon.config.schema import CompileStep, TaskConfig
from chameleon.profile.execution_plan import build_execution_plan
from chameleon.profile.shape_resolver import (
    precision_to_dtype_bytes,
    resolve_precision,
    resolve_stage_shapes,
)


def test_precision_to_dtype_bytes() -> None:
    assert precision_to_dtype_bytes("bf16") == 2
    assert precision_to_dtype_bytes("float32") == 4


def test_resolve_precision_from_model_overrides() -> None:
    task = TaskConfig(model_overrides={"precision": "float32"})
    assert resolve_precision(task) == "float32"


def test_resolve_stage_shapes_from_build_cfg(
    task_deploy_yaml: Path,
    build_configs_dir: Path,
) -> None:
    task = TaskConfig.load(task_deploy_yaml)
    task.deploy.build_cfg_dir = str(build_configs_dir)
    plan = build_execution_plan(task)

    vit_shapes = resolve_stage_shapes(task, "vit", plan)
    assert vit_shapes["pixel_values"][0] == 3  # _NUM_VIEWS from vit_build_cfg

    llm_shapes = resolve_stage_shapes(task, "llm", plan)
    assert llm_shapes["inputs_embeds"][1] == 818


def test_reference_shapes() -> None:
    task = TaskConfig(
        model_overrides={"action_dim": 16, "action_horizon": 8},
        infer={"batch_size": 1},
    )
    plan = build_execution_plan(task)
    shapes = resolve_stage_shapes(task, "action_expert", plan)
    assert shapes["x_t"] == (1, 8, 16)
