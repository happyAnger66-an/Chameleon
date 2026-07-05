"""Cosmos3 Policy TRT-only 运行器 — 真实权重 host + TRT 热点 engine。

对标 pi05 ``Pi05TrtOnlyRunner``：host 用 diffusers ``Cosmos3OmniPipeline`` 做 tokenize /
打包 / scheduler，TRT engine 跑 vae_encode / dit / vae_decode。输出 action chunk
``[chunk, raw_action_dim]``（policy）。注册 key ``cosmos3_trt_only``。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from chameleon.config.schema import TaskConfig
from chameleon.deploy.cosmos3.paths import resolve_engine_dir
from chameleon.deploy.cosmos3.shapes import Cosmos3Profile, get_profile
from chameleon.evaluate.runner_base import PolicyRunner, register_policy_runner
from chameleon.runtime.cosmos3_trt.adapter import build_conditioning_canvas, load_cosmos3_host_pipeline
from chameleon.runtime.cosmos3_trt.engines import load_trt_stage_engines, validate_engine_files
from chameleon.runtime.cosmos3_trt.pipeline import Cosmos3PolicyTrtPipeline

logger = logging.getLogger(__name__)

_STAGES = ("vae_encode", "text_embed", "dit", "vae_decode")


def resolve_cosmos3_profile(task: TaskConfig) -> Cosmos3Profile:
    """Resolve the fixed TRT profile from ``model_overrides.trt_profile`` or the action domain."""
    name = task.model_overrides.get("trt_profile")
    if name:
        return get_profile(str(name))
    domain = task.generate.action.domain_name
    return get_profile("policy_droid" if domain == "droid_lerobot" else "nano_action")


class Cosmos3TrtPolicyRunner(PolicyRunner):
    """diffusers host + Cosmos3PolicyTrtPipeline（policy 动作生成）。"""

    def __init__(self, task: TaskConfig) -> None:
        self._task = task
        self._device = task.infer.torch_device or "cuda"
        self._profile = resolve_cosmos3_profile(task)
        self._engine_dir = resolve_engine_dir(task)
        self._num_steps = int(task.infer.num_steps or self._profile.num_inference_steps)
        self._pipe: Any | None = None
        self._pipeline: Cosmos3PolicyTrtPipeline | None = None
        self._built = False

    @classmethod
    def from_task(cls, task: TaskConfig) -> "Cosmos3TrtPolicyRunner":
        validate_engine_files(resolve_engine_dir(task), _STAGES)
        return cls(task)

    def build(self) -> "Cosmos3TrtPolicyRunner":
        if self._built:
            return self
        self._pipe = load_cosmos3_host_pipeline(self._task, self._device)
        engines = load_trt_stage_engines(
            self._task,
            engine_dir=self._engine_dir,
            device=self._device,
            stages=_STAGES,
            enable_cuda_graph=bool(self._task.infer.cuda_graph),
        )
        self._pipeline = Cosmos3PolicyTrtPipeline(
            engines, self._pipe, self._profile, num_steps=self._num_steps
        )
        logger.info(
            "Cosmos3TrtPolicyRunner: profile=%s engines=%s device=%s steps=%d",
            self._profile.name,
            self._engine_dir,
            self._device,
            self._num_steps,
        )
        self._built = True
        return self

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> np.ndarray:
        if not self._built:
            self.build()
        assert self._pipe is not None and self._pipeline is not None
        import torch

        dtype = self._pipe.transformer.dtype
        video = build_conditioning_canvas(
            self._pipe, self._profile, observation, device=self._device, dtype=dtype
        )
        with torch.no_grad():
            action = self._pipeline.infer_policy(
                {"video": video}, self._device, num_steps=self._num_steps
            )
        return np.asarray(action.detach().float().cpu().numpy())

    @property
    def action_horizon(self) -> int:
        return int(self._profile.chunk_size)

    @property
    def action_dim(self) -> int:
        return int(self._profile.raw_action_dim)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "cosmos3_trt_only",
            "compare_mode": False,
            "profile": self._profile.name,
            "model_id": self._profile.model_id,
            "engine_dir": str(self._engine_dir),
            "num_steps": self._num_steps,
            "guidance_scale": self._profile.guidance_scale,
        }


register_policy_runner("cosmos3_trt_only", Cosmos3TrtPolicyRunner, override=True)
