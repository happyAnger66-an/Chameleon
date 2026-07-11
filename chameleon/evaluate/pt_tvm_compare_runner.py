"""PyTorch vs TVM(mlc-vla M1) 双路对比策略运行器。

PT 路走 openpi 浮点；TVM 路复用 vit TRT + mlc-vla prefill/denoise（3.12 worker）。
两路共用同一 flow-matching 噪声，便于 WebUI 展示 PT−TVM 差异。

PT 设备可配（默认 ``cpu``，避免与 TVM 抢显存）：
  - ``model_overrides.pt_device``（优先）
  - 否则 ``evaluate.pytorch_load_device``（schema 默认 ``cpu``）
显存紧张时请保持 ``pt_device: cpu``；显存充足可设 ``cuda``。
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import torch

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
from chameleon.evaluate.task_utils import resolve_eval_device, resolve_pytorch_load_device
from chameleon.evaluate.trt_eval_utils import resolve_trt_engine_names
from chameleon.runtime.pi05_trt.adapter import prepare_openpi_policy_for_trt
from chameleon.runtime.pi05_tvm.engine import TvmWorkerClient, load_vit_engine
from chameleon.runtime.pi05_tvm.pipeline import Pi05TvmPipeline, attach_tvm_to_policy

logger = logging.getLogger(__name__)


def _empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_pt_device(task: TaskConfig) -> str:
    """PT 路设备：``model_overrides.pt_device`` > ``evaluate.pytorch_load_device`` > ``cpu``。"""
    raw = task.model_overrides.get("pt_device")
    if raw is None or str(raw).strip() == "":
        raw = resolve_pytorch_load_device(task)
    return str(raw or "cpu").strip() or "cpu"


class Pi05PtTvmCompareRunner(PolicyRunner, SupportsDualInfer, SupportsFixedNoise):
    """evaluate 双路：PyTorch 浮点 vs Chameleon TVM(mlc-vla)。"""

    def __init__(self, task: TaskConfig) -> None:
        self._task = task
        self._pt: OpenPiPolicyRunner | None = None
        self._policy_tvm: Any | None = None
        self._tvm_client: TvmWorkerClient | None = None
        self._engine_dir = resolve_engine_dir(task)
        self._engines = resolve_trt_engine_names(task)
        self._noise_mode = task.evaluate.noise
        self._noise_seed = int(task.evaluate.noise_seed)
        self._num_steps = int(
            task.infer.num_steps
            or task.model_overrides.get("num_denoise_steps")
            or 10
        )
        self._tvm_dtype = str(task.model_overrides.get("tvm_dtype") or "float16")
        self._use_loop = bool(task.model_overrides.get("tvm_loop", True))
        self._cuda_graph = bool(task.model_overrides.get("tvm_cuda_graph", False))
        self._pt_device = _resolve_pt_device(task)
        self._infer_device = resolve_eval_device(task) or "cuda"
        self._built = False

    @classmethod
    def from_task(cls, task: TaskConfig) -> "Pi05PtTvmCompareRunner":
        if not task.evaluate.compare_mode:
            raise ValueError(
                "policy_runner=pt_tvm_compare 需要 evaluate.compare_mode=true。"
            )
        return cls(task)

    def build(self) -> "Pi05PtTvmCompareRunner":
        if self._built:
            return self

        if self._pt_device.startswith("cuda") and self._infer_device.startswith("cuda"):
            logger.warning(
                "pt_tvm_compare: PT(%s) 与 TVM/vit(%s) 同卡，显存可能不足；"
                "OOM 时请设 model_overrides.pt_device=cpu。",
                self._pt_device,
                self._infer_device,
            )
        logger.info(
            "Pi05PtTvmCompareRunner: PT on %s, TVM/vit on %s (dtype=%s loop=%s)",
            self._pt_device,
            self._infer_device,
            self._tvm_dtype,
            self._use_loop,
        )
        pt_session = build_openpi_session(self._task, pytorch_device=self._pt_device)
        self._pt = OpenPiPolicyRunner(session=pt_session)
        _empty_cuda()

        if self._cuda_graph:
            logger.warning(
                "compare_mode 下自动关闭 tvm_cuda_graph（双策略串行；"
                "首次编译更占显存，默认关）。"
            )
            self._cuda_graph = False

        # TVM 侧 openpi 壳仍先落 CPU，再只把 embed 挪到 infer_device
        tvm_session = build_openpi_session(self._task, pytorch_device="cpu")
        prepare_openpi_policy_for_trt(tvm_session.policy, self._infer_device)
        _empty_cuda()

        vit = load_vit_engine(
            self._task,
            engine_dir=self._engine_dir,
            vit_name=self._engines.vit,
            device=self._infer_device,
            enable_cuda_graph=False,
        )
        _empty_cuda()

        self._tvm_client = TvmWorkerClient(
            tvm_session.checkpoint_dir,
            dtype=self._tvm_dtype,
            target="cuda",
            cuda_graph=self._cuda_graph,
        )
        pipeline = Pi05TvmPipeline(
            self._tvm_client,
            vit,
            num_steps=self._num_steps,
            use_loop=self._use_loop,
        )
        attach_tvm_to_policy(tvm_session.policy, pipeline)
        self._policy_tvm = tvm_session.policy
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
        assert self._pt is not None
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
        assert self._pt is not None and self._policy_tvm is not None
        flow_noise = noise if noise is not None else self.noise_for_sample(sample_index)
        # PT 与 TVM 串行；pt_device=cpu 时不与 TVM 抢 GPU
        pred_pt = self._pt.infer(observation, noise=flow_noise)
        out_tvm = self._policy_tvm.infer(dict(observation), noise=flow_noise)
        pred_tvm = np.asarray(out_tvm["actions"], dtype=np.float32)
        if pred_tvm.ndim == 3 and pred_tvm.shape[0] == 1:
            pred_tvm = pred_tvm[0]
        if not np.isfinite(pred_tvm).all():
            n_bad = int(np.size(pred_tvm) - np.isfinite(pred_tvm).sum())
            logger.warning(
                "Pi05PtTvmCompareRunner: TVM actions contain %d non-finite values "
                "(sample_index=%d shape=%s)",
                n_bad,
                sample_index,
                pred_tvm.shape,
            )
        return pred_pt, pred_tvm

    @property
    def action_horizon(self) -> int:
        if not self._built:
            self.build()
        assert self._pt is not None
        return self._pt.action_horizon

    @property
    def action_dim(self) -> int:
        if not self._built:
            self.build()
        assert self._pt is not None
        return self._pt.action_dim

    @property
    def metadata(self) -> dict[str, Any]:
        if not self._built:
            self.build()
        assert self._pt is not None
        meta = dict(self._pt.metadata)
        meta.update(
            {
                "backend": "pt_tvm_compare",
                "compare_mode": True,
                "pt_device": self._pt_device,
                "tvm_device": self._infer_device,
                "denoise_backend": (
                    "mlc-vla M1 (prefill + denoise_loop_kv)"
                    if self._use_loop
                    else "mlc-vla M1 (prefill + denoise_step_kv, host Euler)"
                ),
                "vit_backend": "tensorrt",
                "engine_dir": str(self._engine_dir),
                "tvm_dtype": self._tvm_dtype,
                "tvm_loop": self._use_loop,
                "tvm_cuda_graph": self._cuda_graph,
                "noise": self._noise_mode,
                "noise_seed": self._noise_seed,
                "num_steps": self._num_steps,
            }
        )
        return meta


register_policy_runner("pt_tvm_compare", Pi05PtTvmCompareRunner, override=True)
