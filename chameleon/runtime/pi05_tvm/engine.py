"""TVM denoise 引擎客户端：拉起 Python 3.12 worker（tvm+mlc-vla），通过管道收发 numpy。

openpi 评测在 3.11 venv，tvm 需 3.12，故 TVM 去噪跑在独立 3.12 子进程里。本模块封装子进程
生命周期与 4 字节长度前缀 + pickle 的收发协议。
"""

from __future__ import annotations

import logging
import os
import pickle
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 默认路径（可用环境变量覆盖）
_DEFAULT_TVM_HOME = "/home/zhangxa/codes/edgeLLM/tvm"
_DEFAULT_MLC_VLA = "/home/zhangxa/codes/edgeLLM/mlc-vla"


def _worker_env() -> dict[str, str]:
    """Build env for the 3.12 TVM worker.

    Newer TVM imports ``tvm_ffi`` as a top-level package (sibling of ``tvm`` under
    ``$TVM_HOME/python``, or from ``$TVM_HOME/3rdparty/tvm-ffi/python``). Thor setups
    that only put ``tvm`` on PYTHONPATH will hit ``ModuleNotFoundError: tvm_ffi``.
    """
    tvm_home = Path(os.environ.get("TVM_HOME", _DEFAULT_TVM_HOME)).expanduser()
    mlc_vla = Path(os.environ.get("MLC_VLA_HOME", _DEFAULT_MLC_VLA)).expanduser()
    env = dict(os.environ)
    py_paths: list[str] = [
        str(tvm_home / "python"),
        str(tvm_home / "3rdparty" / "tvm-ffi" / "python"),
        str(mlc_vla / "python"),
    ]
    # Keep any pre-existing PYTHONPATH entries (e.g. site-packages with apache-tvm-ffi).
    existing = env.get("PYTHONPATH", "")
    if existing:
        py_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(p for p in py_paths if p)
    env.setdefault("TVM_LIBRARY_PATH", str(tvm_home / "build" / "lib"))
    env.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")
    return env


class TvmWorkerClient:
    """管理 3.12 TVM worker 子进程，转发 prefill+denoise 请求。"""

    def __init__(self, checkpoint_dir: str | Path, *, dtype: str = "bfloat16",
                 target: str = "cuda", python_exe: str | None = None,
                 cuda_graph: bool = False):
        self.python_exe = python_exe or os.environ.get("MLC_VLA_PY", "python3")
        self.cuda_graph = cuda_graph
        worker_mod = "chameleon.runtime.pi05_tvm.worker"
        # worker 只依赖 mlc_vla+tvm，但用 -m 需能 import chameleon 包 → 直接跑文件路径更稳。
        worker_file = str(Path(__file__).with_name("worker.py"))
        cmd = [self.python_exe, "-u", worker_file,
               "--checkpoint-dir", str(checkpoint_dir), "--dtype", dtype, "--target", target]
        if cuda_graph:
            cmd.append("--cuda-graph")
        logger.info("Launching TVM worker: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, env=_worker_env(),
        )
        info = self._recv()
        if info is None or not info.get("ready"):
            err = (info or {}).get("error", "worker 未返回 ready")
            raise RuntimeError(f"TVM worker 启动失败:\n{err}")
        self.info = info
        self.param_bytes = int(info.get("param_bytes", 0))
        self.last_timings: dict[str, Any] = {}
        logger.info("TVM worker ready: prefix_len=%d denoise_param=%.1fMB",
                    info["prefix_len"], self.param_bytes / 1e6)

    # ---- 协议 ----
    def _send(self, obj) -> None:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.proc.stdin.write(struct.pack(">Q", len(data)))
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _recv(self):
        header = self.proc.stdout.read(8)
        if not header or len(header) < 8:
            rc = self.proc.poll()
            raise RuntimeError(f"TVM worker 管道关闭（returncode={rc}）")
        (n,) = struct.unpack(">Q", header)
        return pickle.loads(self.proc.stdout.read(n))

    def sample(self, prefix_embs: np.ndarray, prefix_pad: np.ndarray,
               noise: np.ndarray, num_steps: int, loop: bool = True,
               *, return_timings: bool = False):
        self._send({
            "op": "sample",
            "prefix_embs": np.ascontiguousarray(prefix_embs, dtype=np.float32),
            "prefix_pad": np.ascontiguousarray(prefix_pad, dtype=np.float32),
            "noise": np.ascontiguousarray(noise, dtype=np.float32),
            "num_steps": int(num_steps),
            "loop": bool(loop),
            "timed": bool(return_timings),
        })
        r = self._recv()
        if not r.get("ok"):
            raise RuntimeError(f"TVM worker sample 失败:\n{r.get('error')}")
        actions = r["actions"]
        timings = dict(r.get("timings") or {})
        self.last_timings = timings
        if return_timings:
            return actions, timings
        return actions

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self._send({"op": "shutdown"})
                self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def load_vit_engine(task, *, engine_dir, vit_name: str, device: str, enable_cuda_graph: bool = False):
    """只加载 vit TRT engine（TVM 路径复用 SigLIP vit，llm/denoise 走 TVM，无需加载）。"""
    from chameleon.core.artifact import Artifact
    from chameleon.core.context import RunContext
    from chameleon.core.platform import get_platform
    from chameleon.runtime.tensorrt.backend import TensorRTRuntime

    platform = get_platform(task.platform)
    ctx = RunContext(
        platform=platform,
        architecture=task.architecture,
        options={"torch_device": device, "enable_cuda_graph": enable_cuda_graph},
    )
    artifact = Artifact(kind="engine", stage="vit", platform=platform.name,
                        path=str(Path(engine_dir) / vit_name))
    engine = TensorRTRuntime().load(artifact, ctx)
    logger.info("Loaded TRT vit engine: %s", Path(engine_dir) / vit_name)
    return engine
