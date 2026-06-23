"""openpi Policy ↔ Pi05TrtPipeline 桥接 — 替换 ``Policy._sample_actions``。"""

from __future__ import annotations

from typing import Any

from chameleon.runtime.pi05_trt.pipeline import Pi05TrtPipeline
from chameleon.runtime.pi05_trt.weight_release import release_heavy_pytorch_weights


def prepare_openpi_policy_for_trt(policy: Any, infer_device: str) -> None:
    """在加载 TRT engine 前释放 PyTorch 大权重，避免 GPU OOM。"""
    release_heavy_pytorch_weights(policy._model, embed_device=infer_device)
    policy._pytorch_device = infer_device


def attach_trt_to_policy(
    policy: Any,
    pipeline: Pi05TrtPipeline,
    *,
    release_weights: bool = True,
) -> None:
    """将 openpi Policy 的 ``_sample_actions`` 路由到 TRT 管线（不修改 model 类方法）。"""
    model = policy._model
    num_steps = pipeline.num_steps

    def _sample_actions_trt(device, observation, noise=None, **kwargs):
        ns = int(kwargs.get("num_steps", num_steps))
        return pipeline.infer(
            model,
            device,
            observation,
            noise=noise,
            num_steps=ns,
        )

    policy._sample_actions = _sample_actions_trt
    if release_weights:
        release_heavy_pytorch_weights(model)
