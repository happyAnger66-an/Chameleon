"""Chameleon 编排器策略运行器 — 经 InferenceSession 推理（Pi05RealOrchestrator / TRT）。

作用：
    ChameleonOrchestratorRunner 在 evaluate 路径上走 Chameleon 主链：
    openpi I/O transform（与 OpenPiPolicyRunner 同源）→ ModelAdapter +
    InferenceSession（Pi05RealOrchestrator 或 stage engine 链）→ output transform。
    当前 LeRobot 离线 eval 仅支持 ``orchestrator_key=pi05_real`` 的真实 PyTorch
    整模型路径；TRT stage 链与 lerobot repack 数据的桥接留待后续。

架构位置：
    工具层（evaluate）— 注册为 policy_runner ``chameleon``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chameleon.api import _run_context, build_adapter
from chameleon.config.schema import TaskConfig
from chameleon.evaluate.openpi_transforms import (
    OpenPiEvalTransforms,
    apply_output_transform,
    build_openpi_eval_transforms,
)
from chameleon.evaluate.runner_base import PolicyRunner, register_policy_runner
from chameleon.evaluate.task_utils import (
    resolve_checkpoint_dir,
    resolve_eval_device,
    resolve_openpi_config,
    resolve_torch_device,
)
from chameleon.runtime.orchestrator import InferenceSession

logger = logging.getLogger(__name__)


def sample_actions_from_transformed(
    adapter: Any,
    inputs: dict[str, Any],
    *,
    device: str,
    num_steps: int,
    noise: np.ndarray | None = None,
) -> torch.Tensor:
    """对 post-input-transform 的 dict 调用 ``PI0Pytorch.sample_actions``。"""
    from openpi.models.model import Observation

    if not getattr(adapter, "_is_real_openpi", False):
        raise RuntimeError(
            "ChameleonOrchestratorRunner 当前仅支持真实 openpi 模型（use_reference=false）。"
        )

    torch_inputs: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        arr = np.asarray(value)
        tensor = torch.from_numpy(arr).to(device)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[0] != 1:
            tensor = tensor.unsqueeze(0)
        torch_inputs[key] = tensor

    observation = Observation.from_dict(torch_inputs)
    model = adapter.model
    kwargs: dict[str, Any] = {"num_steps": num_steps}
    if noise is not None:
        noise_t = torch.from_numpy(np.asarray(noise)).to(device)
        if noise_t.ndim == 2:
            noise_t = noise_t.unsqueeze(0)
        kwargs["noise"] = noise_t

    with torch.no_grad():
        return model.sample_actions(device, observation, **kwargs)


class ChameleonOrchestratorRunner(PolicyRunner):
    """evaluate 走 Chameleon InferenceSession + openpi I/O transform。"""

    def __init__(
        self,
        task: TaskConfig,
        *,
        transforms: OpenPiEvalTransforms,
        device: str,
    ) -> None:
        self._task = task
        self._transforms = transforms
        self._device = device
        self._built = False
        self._adapter: Any = None
        self._session: InferenceSession | None = None
        self._orch_key: str | None = None

    @classmethod
    def from_task(cls, task: TaskConfig) -> "ChameleonOrchestratorRunner":
        openpi_config = resolve_openpi_config(task)
        transforms = build_openpi_eval_transforms(
            openpi_config=openpi_config,
            checkpoint_dir=resolve_checkpoint_dir(task),
            norm_stats_dir=task.evaluate.norm_stats_dir,
            asset_id=task.evaluate.asset_id,
            default_prompt=task.evaluate.default_prompt,
        )
        device = resolve_eval_device(task) or "cpu"
        return cls(task, transforms=transforms, device=resolve_torch_device(device) or "cpu")

    def build(self) -> "ChameleonOrchestratorRunner":
        if self._built:
            return self

        ctx = _run_context(self._task)
        ctx.options["torch_device"] = self._device
        self._adapter = build_adapter(self._task, device=self._device)
        self._session = InferenceSession(
            self._adapter,
            ctx,
            stage_runtimes=self._task.stage_runtimes or None,
        ).build()
        self._orch_key = getattr(self._adapter, "orchestrator_key", None) or self._task.architecture
        logger.info(
            "ChameleonOrchestratorRunner ready: orchestrator=%s device=%s use_compiled=%s",
            self._orch_key,
            self._device,
            self._task.infer.use_compiled_engines,
        )
        self._built = True
        return self

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> np.ndarray:
        if not self._built:
            self.build()

        inputs = self._transforms.input_transform(dict(observation))
        num_steps = int(
            self._task.infer.num_steps
            or getattr(self._adapter.config, "num_denoise_steps", 10)
        )

        if self._orch_key == "pi05_real" and not self._task.infer.use_compiled_engines:
            actions_t = sample_actions_from_transformed(
                self._adapter,
                inputs,
                device=self._device,
                num_steps=num_steps,
                noise=noise,
            )
        elif self._session is not None and self._orch_key != "pi05_real":
            raise NotImplementedError(
                "LeRobot 离线 eval 暂不支持 staged orchestrator（TRT/分段 engine）。"
                "请使用 policy_runner=openpi，或 policy_runner=chameleon 且 "
                "use_compiled_engines=false + 真实 openpi 权重。"
            )
        else:
            raise RuntimeError(f"Unsupported orchestrator configuration: {self._orch_key!r}")

        return apply_output_transform(self._transforms.output_transform, actions_t, inputs)

    @property
    def action_horizon(self) -> int:
        return self._transforms.action_horizon

    @property
    def action_dim(self) -> int:
        return self._transforms.action_dim

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "chameleon_orchestrator",
            "orchestrator_key": self._orch_key,
            "openpi_config": self._transforms.openpi_config,
            "device": self._device,
            "use_compiled_engines": self._task.infer.use_compiled_engines,
        }


register_policy_runner("chameleon", ChameleonOrchestratorRunner, override=True)
