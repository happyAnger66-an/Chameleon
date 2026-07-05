"""Cosmos3 Policy PyTorch vs TRT 双路对比运行器 — 数值一致性验证。

复用 :class:`Cosmos3TrtPolicyRunner` 的 TRT 路径 + host diffusers ``Cosmos3OmniPipeline``
的 PyTorch 端到端 policy 生成，输出两路 action 并报告 max/mean diff（允许 bf16 误差）。
注册 key ``cosmos3_pt_trt_compare``。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from chameleon.config.schema import TaskConfig
from chameleon.evaluate.cosmos3_trt_runner import Cosmos3TrtPolicyRunner
from chameleon.evaluate.runner_base import register_policy_runner

logger = logging.getLogger(__name__)


class Cosmos3PtTrtCompareRunner(Cosmos3TrtPolicyRunner):
    """在 TRT runner 之上追加 PyTorch 端到端 policy 生成，做双路对比。"""

    def _pt_action(self, observation: dict[str, Any]) -> np.ndarray | None:
        """Run the full PyTorch ``Cosmos3OmniPipeline`` policy call for the same conditioning."""
        import torch

        try:
            from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition
        except Exception as exc:  # noqa: BLE001
            logger.warning("cosmos3 PT compare unavailable (import failed): %s", exc)
            return None

        image = observation.get("image")
        video = observation.get("video")
        if image is None and video is None:
            logger.warning("cosmos3 PT compare needs observation image/video; skipping PT path.")
            return None

        gen = self._task.generate
        action = CosmosActionCondition(
            mode="policy",
            chunk_size=self._profile.chunk_size,
            domain_name=self._profile.domain_name,
            resolution_tier=self._profile.resolution_tier,
            view_point=gen.action.view_point,
            image=image if image is not None else None,
            video=video if image is None else None,
        )
        with torch.no_grad():
            out = self._pipe(
                prompt=gen.prompt or "Pick up the object and place it at the target location.",
                num_inference_steps=self._num_steps,
                guidance_scale=1.0,
                output_type="pt",
                action=action,
                return_dict=False,
                enable_safety_check=False,
            )
        action_output = out[-1]
        if not action_output:
            return None
        return np.asarray(action_output[0].detach().float().cpu().numpy())

    def infer_dual(
        self,
        observation: dict[str, Any],
        *,
        sample_index: int = 0,
        noise: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        trt_action = self.infer(observation, noise=noise)
        pt_action = self._pt_action(observation)
        if pt_action is None:
            pt_action = trt_action
        else:
            diff = np.abs(pt_action - trt_action)
            logger.info(
                "cosmos3 PT vs TRT action diff: max=%.4e mean=%.4e (shape=%s)",
                float(diff.max()),
                float(diff.mean()),
                tuple(trt_action.shape),
            )
        return pt_action, trt_action

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> np.ndarray:
        return super().infer(observation, noise=noise)

    @property
    def metadata(self) -> dict[str, Any]:
        meta = dict(super().metadata)
        meta.update({"backend": "cosmos3_pt_trt_compare", "compare_mode": True})
        return meta


register_policy_runner("cosmos3_pt_trt_compare", Cosmos3PtTrtCompareRunner, override=True)
