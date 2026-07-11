"""仅 TVM(mlc-vla M1) 去噪策略运行器 — vit 走 TRT，llm prefill + denoise 走 TVM。

vs ground-truth 单路评测；配合 ``pt_tvm_compare`` 做 PT/TVM 双路对比。
"""

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
from chameleon.evaluate.trt_eval_utils import resolve_trt_engine_names
from chameleon.runtime.pi05_trt.adapter import prepare_openpi_policy_for_trt
from chameleon.runtime.pi05_tvm.engine import TvmWorkerClient, load_vit_engine
from chameleon.runtime.pi05_tvm.pipeline import Pi05TvmPipeline, attach_tvm_to_policy

logger = logging.getLogger(__name__)


class Pi05TvmOnlyRunner(PolicyRunner, SupportsFixedNoise):
    """openpi Policy + Pi05TvmPipeline（``policy.infer`` 保留 I/O transform，去噪换 TVM）。"""

    def __init__(self, task: TaskConfig) -> None:
        self._task = task
        self._session: OpenPiSession | None = None
        self._tvm_client = None
        self._built = False
        self._device = resolve_eval_device(task) or "cuda"
        self._pytorch_load_device = resolve_pytorch_load_device(task)
        self._engine_dir = resolve_engine_dir(task)
        self._engines_names = resolve_trt_engine_names(task)
        self._noise_mode = task.evaluate.noise
        self._noise_seed = int(task.evaluate.noise_seed)
        self._num_steps = int(task.infer.num_steps or task.model_overrides.get("num_denoise_steps") or 10)
        self._tvm_dtype = str(task.model_overrides.get("tvm_dtype") or "bfloat16")
        # 图内整段 Euler 环（denoise_loop_kv）+ 整段 CUDA Graph（消除每步 host↔device / IPC 往返）
        self._use_loop = bool(task.model_overrides.get("tvm_loop", True))
        self._cuda_graph = bool(task.model_overrides.get("tvm_cuda_graph", False))
        self._param_bytes = 0

    @classmethod
    def from_task(cls, task: TaskConfig) -> "Pi05TvmOnlyRunner":
        return cls(task)

    def build(self) -> "Pi05TvmOnlyRunner":
        if self._built:
            return self
        self._session = build_openpi_session(self._task, pytorch_device=self._pytorch_load_device)
        # 释放 backbone/vision 大权重，把 embed_tokens 放到 infer_device（语言 embedding 仍走 PT）
        prepare_openpi_policy_for_trt(self._session.policy, self._device)

        vit = load_vit_engine(
            self._task, engine_dir=self._engine_dir, vit_name=self._engines_names.vit,
            device=self._device, enable_cuda_graph=bool(self._task.evaluate.trt_cuda_graph),
        )
        # TVM 去噪跑在独立 3.12 子进程（tvm_ffi 需 3.12，openpi venv 为 3.11）
        self._tvm_client = TvmWorkerClient(
            self._session.checkpoint_dir,
            dtype=self._tvm_dtype,
            target="cuda",
            cuda_graph=self._cuda_graph,
        )
        self._param_bytes = self._tvm_client.param_bytes
        pipeline = Pi05TvmPipeline(
            self._tvm_client,
            vit,
            num_steps=self._num_steps,
            use_loop=self._use_loop,
        )
        attach_tvm_to_policy(self._session.policy, pipeline)
        logger.info(
            "Pi05TvmOnlyRunner built: checkpoint=%s dtype=%s num_steps=%d "
            "loop=%s cuda_graph=%s denoise_param=%.1fMB",
            self._session.checkpoint_dir,
            self._tvm_dtype,
            self._num_steps,
            self._use_loop,
            self._cuda_graph,
            self._param_bytes / 1e6,
        )
        self._built = True
        return self

    def noise_for_sample(self, sample_index: int) -> np.ndarray | None:
        return flow_match_noise(
            action_horizon=self.action_horizon, action_dim=self.action_dim,
            sample_index=sample_index, noise_mode=self._noise_mode, noise_seed=self._noise_seed,
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
        denoise = (
            "mlc-vla M1 (prefill + denoise_loop_kv)"
            if self._use_loop
            else "mlc-vla M1 (prefill + denoise_step_kv, host Euler)"
        )
        return {
            "backend": "tvm_only",
            "compare_mode": False,
            "denoise_backend": denoise,
            "vit_backend": "tensorrt",
            "openpi_config": resolve_openpi_config(self._task),
            "checkpoint_dir": str(resolve_checkpoint_dir(self._task)),
            "tvm_dtype": self._tvm_dtype,
            "tvm_loop": self._use_loop,
            "tvm_cuda_graph": self._cuda_graph,
            "denoise_param_bytes": int(self._param_bytes),
            "noise": self._noise_mode,
            "noise_seed": self._noise_seed,
            "num_steps": self._num_steps,
        }


register_policy_runner("tvm_only", Pi05TvmOnlyRunner, override=True)
