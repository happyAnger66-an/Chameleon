"""cosmos3 单元测试 — reference 冒烟、adapter/stage 映射、schema、ONNX 导出。"""

from __future__ import annotations

import torch

from chameleon.architectures.registry import get_architecture
from chameleon.config.schema import GenerateConfig, TaskConfig
from chameleon.models.cosmos3.adapter import Cosmos3Adapter
from chameleon.models.cosmos3.reference import Cosmos3Config


def _task(mode: str = "video", steps: int = 4) -> TaskConfig:
    return TaskConfig.model_validate(
        {
            "architecture": "cosmos3",
            "model": "cosmos3",
            "platform": "generic_cpu",
            "actions": ["infer"],
            "model_overrides": {"mode": mode, "guidance_scale": 6.0},
            "infer": {"batch_size": 1, "num_steps": steps},
            "generate": {"mode": mode},
        }
    )


def test_architecture_registered() -> None:
    spec = get_architecture("cosmos3")
    assert spec.orchestrator == "cosmos3"
    assert spec.stage_names == ("vae_encode", "text_embed", "dit", "vae_decode")


def test_generate_config_defaults() -> None:
    gen = GenerateConfig()
    assert gen.mode == "video"
    assert gen.guidance_scale == 6.0
    assert gen.action.mode == "policy"
    assert gen.action.chunk_size == 16


def test_adapter_stage_mapping() -> None:
    adapter = Cosmos3Adapter(Cosmos3Config()).build("cpu")
    for stage in ("vae_encode", "text_embed", "dit", "vae_decode"):
        module = adapter.stage_module(stage)
        assert isinstance(module, torch.nn.Module)
    assert adapter.orchestrator_key is None  # reference path uses default orchestrator


def test_reference_infer_video() -> None:
    from chameleon.api import run_infer

    out = run_infer(_task("video"))
    # [B, T, C, H, W]
    assert out.ndim == 5
    assert out.shape[0] == 1


def test_reference_infer_action() -> None:
    from chameleon.api import run_infer

    out = run_infer(_task("action"))
    cfg = Cosmos3Config()
    assert tuple(out.shape) == (1, cfg.action_horizon, cfg.action_dim)


import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cosmos3_cpu.yaml",
        "cosmos3_video_realweights.yaml",
        "cosmos3_action_realweights.yaml",
        "cosmos3_trt_deploy.yaml",
    ],
)
def test_load_repo_configs(configs_dir, name: str) -> None:
    task = TaskConfig.load(configs_dir / name)
    assert task.architecture == "cosmos3"
    assert task.generate.mode in ("video", "action")


def test_stage_example_inputs_and_io_names() -> None:
    adapter = Cosmos3Adapter(Cosmos3Config()).build("cpu")
    obs = adapter.example_observation(1, "cpu")
    for stage in ("vae_encode", "text_embed", "dit", "vae_decode"):
        inputs = adapter.stage_example_inputs(stage, obs)
        in_names, out_names = adapter.stage_io_names(stage)
        assert len(inputs) == len(in_names)
        assert out_names
