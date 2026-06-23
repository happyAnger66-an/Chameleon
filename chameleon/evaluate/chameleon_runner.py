"""Chameleon 编排器策略运行器 — 经 InferenceSession 或 Pi05TrtPipeline 推理。

作用：
    ChameleonOrchestratorRunner 在 evaluate 路径上走 Chameleon 主链：
    openpi I/O transform → ModelAdapter + InferenceSession（Pi05RealOrchestrator）
    或 Pi05TrtPipeline（TRT engine 可用时）→ output transform。

架构位置：
    工具层（evaluate）— 注册为 policy_runner ``chameleon``。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from chameleon.api import _run_context, build_adapter
from chameleon.config.schema import TaskConfig
from chameleon.deploy.paths import resolve_checkpoint_dir, resolve_engine_dir
from chameleon.evaluate.openpi_transforms import (
    OpenPiEvalTransforms,
    apply_output_transform,
    build_openpi_eval_transforms,
)
from chameleon.evaluate.runner_base import PolicyRunner, register_policy_runner
from chameleon.evaluate.task_utils import (
    resolve_eval_device,
    resolve_openpi_config,
    resolve_pytorch_load_device,
    resolve_torch_device,
)
from chameleon.evaluate.trt_eval_utils import resolve_trt_engine_names, validate_engine_files
from chameleon.runtime.orchestrator import InferenceSession
from chameleon.runtime.pi05_trt.engines import load_trt_stage_engines
from chameleon.runtime.pi05_trt.pipeline import Pi05TrtPipeline
from chameleon.runtime.pi05_trt.weight_release import release_heavy_pytorch_weights

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


def _trt_engines_available(task: TaskConfig) -> bool:
    try:
        validate_engine_files(resolve_engine_dir(task), resolve_trt_engine_names(task))
    except FileNotFoundError:
        return False
    return True


class ChameleonOrchestratorRunner(PolicyRunner):
    """evaluate 走 Chameleon InferenceSession / Pi05TrtPipeline + openpi I/O transform。"""

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
        self._trt_pipeline: Pi05TrtPipeline | None = None

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

    def _use_pi05_trt(self) -> bool:
        if self._task.infer.use_compiled_engines:
            return _trt_engines_available(self._task)
        return False

    def build(self) -> "ChameleonOrchestratorRunner":
        if self._built:
            return self

        ctx = _run_context(self._task)
        ctx.options["torch_device"] = self._device
        use_trt = self._use_pi05_trt()
        self._adapter = build_adapter(
            self._task,
            device=resolve_pytorch_load_device(self._task) if use_trt else self._device,
        )
        self._orch_key = getattr(self._adapter, "orchestrator_key", None) or self._task.architecture

        if use_trt:
            num_steps = int(
                self._task.infer.num_steps
                or getattr(self._adapter.config, "num_denoise_steps", 10)
            )
            engine_dir = resolve_engine_dir(self._task)
            engines = resolve_trt_engine_names(self._task)
            release_heavy_pytorch_weights(self._adapter.model, embed_device=self._device)
            trt_engines = load_trt_stage_engines(
                self._task,
                engine_dir=engine_dir,
                engines=engines,
                device=self._device,
                enable_cuda_graph=bool(self._task.evaluate.trt_cuda_graph),
            )
            self._trt_pipeline = Pi05TrtPipeline(trt_engines, num_steps=num_steps)
            self._orch_key = "pi05_trt"
            logger.info(
                "ChameleonOrchestratorRunner TRT path: engines=%s device=%s",
                engine_dir,
                self._device,
            )
        else:
            self._session = InferenceSession(
                self._adapter,
                ctx,
                stage_runtimes=self._task.stage_runtimes or None,
            ).build()
            logger.info(
                "ChameleonOrchestratorRunner ready: orchestrator=%s device=%s",
                self._orch_key,
                self._device,
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

        if self._trt_pipeline is not None:
            from openpi.models.model import Observation

            torch_inputs: dict[str, torch.Tensor] = {}
            for key, value in inputs.items():
                arr = np.asarray(value)
                tensor = torch.from_numpy(arr).to(self._device)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                elif tensor.shape[0] != 1:
                    tensor = tensor.unsqueeze(0)
                torch_inputs[key] = tensor
            obs = Observation.from_dict(torch_inputs)
            noise_t = None
            if noise is not None:
                noise_t = torch.from_numpy(np.asarray(noise)).to(self._device)
                if noise_t.ndim == 2:
                    noise_t = noise_t.unsqueeze(0)
            with torch.no_grad():
                actions_t = self._trt_pipeline.infer(
                    self._adapter.model,
                    self._device,
                    obs,
                    noise=noise_t,
                    num_steps=num_steps,
                )
        elif self._orch_key == "pi05_real" and not self._task.infer.use_compiled_engines:
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
                "请使用 policy_runner=openpi / trt_only，或 policy_runner=chameleon 且 "
                "use_compiled_engines=true + 已 build 的 pi05 TRT engine。"
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
