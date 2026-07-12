"""仅 TensorRT engine 策略运行器 — 单路推理 vs ground-truth。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from chameleon.config.schema import TaskConfig
from chameleon.deploy.paths import resolve_checkpoint_dir, resolve_engine_dir
from chameleon.evaluate.noise import flow_match_noise
from chameleon.evaluate.openpi_session import OpenPiSession, build_openpi_session
from chameleon.evaluate.runner_base import PolicyRunner, SupportsFixedNoise, register_policy_runner
from chameleon.evaluate.task_utils import resolve_eval_device, resolve_openpi_config, resolve_pytorch_load_device
from chameleon.evaluate.trt_eval_utils import (
    resolve_trt_engine_names,
    validate_engine_files,
)
from chameleon.runtime.pi05_trt.adapter import attach_trt_to_policy, prepare_openpi_policy_for_trt
from chameleon.runtime.pi05_trt.engines import load_trt_stage_engines
from chameleon.runtime.pi05_trt.pipeline import Pi05TrtPipeline
from chameleon.runtime.tensorrt.backend import memory_report as memory_hint

logger = logging.getLogger(__name__)


class Pi05TrtOnlyRunner(PolicyRunner, SupportsFixedNoise):
    """openpi Policy + Pi05TrtPipeline（``policy.infer`` 含 I/O transform）。"""

    def __init__(self, task: TaskConfig) -> None:
        self._task = task
        self._session: OpenPiSession | None = None
        self._built = False
        self._pipeline: Pi05TrtPipeline | None = None
        self._device = resolve_eval_device(task) or "cuda"
        self._pytorch_load_device = resolve_pytorch_load_device(task)
        self._engine_dir = resolve_engine_dir(task)
        self._engines_names = resolve_trt_engine_names(task)
        self._noise_mode = task.evaluate.noise
        self._noise_seed = int(task.evaluate.noise_seed)
        self._num_steps = int(
            task.infer.num_steps
            or task.model_overrides.get("num_denoise_steps")
            or 10
        )

    @classmethod
    def from_task(cls, task: TaskConfig) -> "Pi05TrtOnlyRunner":
        validate_engine_files(resolve_engine_dir(task), resolve_trt_engine_names(task))
        return cls(task)

    def build(self) -> "Pi05TrtOnlyRunner":
        if self._built:
            return self

        self._session = build_openpi_session(
            self._task,
            pytorch_device=self._pytorch_load_device,
        )
        prepare_openpi_policy_for_trt(self._session.policy, self._device)
        logger.info(
            "Pi05TrtOnlyRunner: checkpoint=%s engines=%s infer_device=%s pytorch_load_device=%s",
            self._session.checkpoint_dir,
            self._engine_dir,
            self._device,
            self._pytorch_load_device,
        )
        trt_engines = load_trt_stage_engines(
            self._task,
            engine_dir=self._engine_dir,
            engines=self._engines_names,
            device=self._device,
            enable_cuda_graph=bool(self._task.evaluate.trt_cuda_graph),
        )
        pipeline = Pi05TrtPipeline(trt_engines, num_steps=self._num_steps)
        attach_trt_to_policy(self._session.policy, pipeline, release_weights=False)
        self._pipeline = pipeline
        self._built = True
        return self

    def set_timer(self, timer: Any | None) -> None:
        if self._pipeline is not None:
            self._pipeline.set_timer(timer)

    def close(self) -> None:
        """Tear down TRT engines + openpi session so GPU mem can go to TVM."""
        if self._session is not None:
            policy = self._session.policy
            # Break attach_trt_to_policy closure that keeps engines alive.
            if hasattr(policy, "_sample_actions"):
                policy._sample_actions = None
            if hasattr(policy, "_chameleon_pipeline"):
                policy._chameleon_pipeline = None
        if self._pipeline is not None:
            try:
                self._pipeline.close()
            except Exception:  # noqa: BLE001
                logger.debug("pipeline.close failed", exc_info=True)
            self._pipeline = None
        if self._session is not None:
            try:
                policy = self._session.policy
                model = getattr(policy, "_model", None)
                if model is not None and hasattr(model, "cpu"):
                    model.cpu()
            except Exception:  # noqa: BLE001
                logger.debug("session model.cpu failed", exc_info=True)
            self._session = None
        self._built = False
        logger.info("Pi05TrtOnlyRunner closed (%s)", memory_hint())

    def noise_for_sample(self, sample_index: int) -> np.ndarray | None:
        return flow_match_noise(
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            sample_index=sample_index,
            noise_mode=self._noise_mode,
            noise_seed=self._noise_seed,
        )

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray | None = None) -> np.ndarray:
        if not self._built:
            self.build()
        assert self._session is not None
        out = self._session.policy.infer(dict(observation), noise=noise)
        return np.asarray(out["actions"])

    @property
    def action_horizon(self) -> int:
        if not self._built:
            self.build()
        assert self._session is not None
        return self._session.action_horizon

    @property
    def action_dim(self) -> int:
        if not self._built:
            self.build()
        assert self._session is not None
        return self._session.action_dim

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "trt_only",
            "compare_mode": False,
            "openpi_config": resolve_openpi_config(self._task),
            "checkpoint_dir": str(resolve_checkpoint_dir(self._task)),
            "engine_dir": str(self._engine_dir),
            "trt_engines": {
                "vit": self._engines_names.vit,
                "llm": self._engines_names.llm,
                "expert": self._engines_names.expert,
                "denoise": self._engines_names.denoise,
            },
            "noise": self._noise_mode,
            "noise_seed": self._noise_seed,
            "num_steps": self._num_steps,
            "pytorch_load_device": self._pytorch_load_device,
        }


register_policy_runner("trt_only", Pi05TrtOnlyRunner, override=True)
