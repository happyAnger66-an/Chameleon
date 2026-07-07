"""PyTorch vs TensorRT 双路对比策略运行器。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from chameleon.config.schema import TaskConfig
from chameleon.deploy.paths import resolve_engine_dir
from chameleon.evaluate.noise import flow_match_noise
from chameleon.evaluate.openpi_session import build_openpi_session
from chameleon.evaluate.policy import OpenPiPolicyRunner
from chameleon.evaluate.runner_base import (
    PolicyRunner,
    SupportsDualInfer,
    SupportsFixedNoise,
    register_policy_runner,
)
from chameleon.evaluate.trt_eval_utils import (
    resolve_trt_engine_names,
    validate_engine_files,
)
from chameleon.evaluate.task_utils import resolve_pytorch_load_device
from chameleon.runtime.pi05_trt.adapter import attach_trt_to_policy, prepare_openpi_policy_for_trt
from chameleon.runtime.pi05_trt.engines import load_trt_stage_engines
from chameleon.runtime.pi05_trt.pipeline import Pi05TrtPipeline

logger = logging.getLogger(__name__)


class Pi05PtTrtCompareRunner(PolicyRunner, SupportsDualInfer, SupportsFixedNoise):
    """evaluate 双路：PyTorch 浮点 vs Chameleon TRT engine。"""

    def __init__(
        self,
        task: TaskConfig,
        *,
        engine_dir: Path,
        engines,
        trt_cuda_graph: bool,
        noise_mode: str,
        noise_seed: int,
        num_steps: int,
    ) -> None:
        self._task = task
        self._pt = OpenPiPolicyRunner(task=task)
        self._policy_trt: Any | None = None
        self._engine_dir = engine_dir
        self._engines = engines
        self._trt_cuda_graph = trt_cuda_graph
        self._noise_mode = noise_mode
        self._noise_seed = noise_seed
        self._num_steps = num_steps
        self._built = False

    @classmethod
    def from_task(cls, task: TaskConfig) -> "Pi05PtTrtCompareRunner":
        if not task.evaluate.compare_mode:
            raise ValueError(
                "policy_runner=pt_trt_compare 需要 evaluate.compare_mode=true。"
            )
        engine_dir = resolve_engine_dir(task)
        engines = resolve_trt_engine_names(task)
        validate_engine_files(engine_dir, engines)
        cuda_graph = bool(task.evaluate.trt_cuda_graph)
        if cuda_graph:
            logger.warning(
                "compare_mode 下自动关闭 trt_cuda_graph（双策略同卡串行推理）。"
            )
            cuda_graph = False
        num_steps = int(
            task.infer.num_steps
            or task.model_overrides.get("num_denoise_steps")
            or 10
        )
        return cls(
            task,
            engine_dir=engine_dir,
            engines=engines,
            trt_cuda_graph=cuda_graph,
            noise_mode=task.evaluate.noise,
            noise_seed=int(task.evaluate.noise_seed),
            num_steps=num_steps,
        )

    def build(self) -> "Pi05PtTrtCompareRunner":
        if self._built:
            return self
        self._pt.build()

        device = self._pt.device or "cuda"
        logger.info(
            "Pi05PtTrtCompareRunner: loading TRT policy device=%s engines=%s",
            device,
            self._engine_dir,
        )
        trt_session = build_openpi_session(
            self._task,
            pytorch_device=resolve_pytorch_load_device(self._task),
        )
        prepare_openpi_policy_for_trt(trt_session.policy, str(device))
        trt_engines = load_trt_stage_engines(
            self._task,
            engine_dir=self._engine_dir,
            engines=self._engines,
            device=str(device),
            enable_cuda_graph=self._trt_cuda_graph,
        )
        pipeline = Pi05TrtPipeline(trt_engines, num_steps=self._num_steps)
        attach_trt_to_policy(trt_session.policy, pipeline, release_weights=False)
        self._policy_trt = trt_session.policy
        self._built = True
        return self

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
        return self._pt.infer(observation, noise=noise)

    def infer_dual(
        self,
        observation: dict[str, Any],
        *,
        sample_index: int = 0,
        noise: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._built:
            self.build()
        assert self._policy_trt is not None
        flow_noise = noise if noise is not None else self.noise_for_sample(sample_index)
        pred_pt = self._pt.infer(observation, noise=flow_noise)
        out_trt = self._policy_trt.infer(dict(observation), noise=flow_noise)
        pred_trt = np.asarray(out_trt["actions"], dtype=np.float32)
        if pred_trt.ndim == 3 and pred_trt.shape[0] == 1:
            pred_trt = pred_trt[0]
        if not np.isfinite(pred_trt).all():
            n_bad = int(np.size(pred_trt) - np.isfinite(pred_trt).sum())
            logger.warning(
                "Pi05PtTrtCompareRunner: TRT actions contain %d non-finite values "
                "(sample_index=%d shape=%s)",
                n_bad,
                sample_index,
                pred_trt.shape,
            )
        return pred_pt, pred_trt

    @property
    def action_horizon(self) -> int:
        return self._pt.action_horizon

    @property
    def action_dim(self) -> int:
        return self._pt.action_dim

    @property
    def metadata(self) -> dict[str, Any]:
        meta = dict(self._pt.metadata)
        meta.update(
            {
                "backend": "pt_trt_compare",
                "compare_mode": True,
                "engine_dir": str(self._engine_dir),
                "trt_engines": {
                    "vit": self._engines.vit,
                    "llm": self._engines.llm,
                    "expert": self._engines.expert,
                    "denoise": self._engines.denoise,
                },
                "noise": self._noise_mode,
                "noise_seed": self._noise_seed,
            }
        )
        return meta


register_policy_runner("pt_trt_compare", Pi05PtTrtCompareRunner, override=True)
