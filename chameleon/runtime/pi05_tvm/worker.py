"""TVM denoise 工作进程（Python 3.12 + tvm + mlc-vla）。

openpi 评测跑在 Python 3.11 venv，而本仓库的 tvm_ffi 需 Python 3.12，二者不能同进程。
本 worker 由 3.11 侧以 3.12 解释器启动，通过 stdin/stdout（4 字节长度前缀 + pickle）通信：
  - 启动后建好 mlc-vla M1 engine（prefill + denoise_step_kv）+ 载权重，回一帧 {"ready":True,...}
  - 收到 {"op":"sample", ...} → 回 {"ok":True, actions, timings?}

``timed=True`` 时拆分：
  - ``tvm_prefill_ms``：``_prepare`` / ``vm["prefill"]``
  - ``tvm_denoise_ms``：``denoise_loop_kv`` 或逐步 ``denoise_step_kv`` 总和
  - ``tvm_denoise_step_mean_ms``：仅逐步模式
  - ``tvm_worker_ms``：二者之和
"""

from __future__ import annotations

import argparse
import os
import pickle
import struct
import sys
import time


def _send(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">Q", len(data)))
    stream.write(data)
    stream.flush()


def _recv(stream):
    header = stream.read(8)
    if not header or len(header) < 8:
        return None
    (n,) = struct.unpack(">Q", header)
    buf = stream.read(n)
    return pickle.loads(buf)


def _device_sync() -> None:
    """Best-effort GPU sync so worker wall-clock covers kernel time."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        import tvm

        if hasattr(tvm, "cuda"):
            tvm.cuda(0).sync()
    except Exception:  # noqa: BLE001
        pass


def _sample_timed_loop(runner, params, prefix_embs, noise, prefix_pad) -> tuple:
    """图内环：分别计 prefill (_prepare) 与 denoise_loop_kv。"""
    import numpy as np
    from mlc_vla.sample import make_time_embs

    cfg = runner.config
    tvm = runner._tvm
    dt = cfg.dtype
    n = cfg.num_denoise_steps

    _device_sync()
    t0 = time.perf_counter()
    keys, values, prefix_mask, suffix_cos, suffix_sin, _ = runner._prepare(
        params, prefix_embs, prefix_pad
    )
    _device_sync()
    prefill_ms = (time.perf_counter() - t0) * 1e3

    noise_np = runner._noise(noise, 0)
    x0 = tvm.runtime.tensor(noise_np, runner.dev)
    time_embs = np.concatenate(make_time_embs(n, cfg.action_expert.width), axis=0).astype(dt)
    te_dev = tvm.runtime.tensor(time_embs, runner.dev)

    _device_sync()
    t1 = time.perf_counter()
    out = runner.vm["denoise_loop_kv"](
        keys, values, x0, te_dev, suffix_cos, suffix_sin, prefix_mask, params
    )
    _device_sync()
    denoise_ms = (time.perf_counter() - t1) * 1e3

    actions = out.numpy() if hasattr(out, "numpy") else out[0].numpy()
    timings = {
        "tvm_prefill_ms": prefill_ms,
        "tvm_denoise_ms": denoise_ms,
        "tvm_worker_ms": prefill_ms + denoise_ms,
        "loop": True,
        "num_steps": int(n),
    }
    return actions, timings


def _sample_timed_steps(runner, params, prefix_embs, noise, prefix_pad, num_steps: int) -> tuple:
    """宿主逐步环：分别计 prefill 与 denoise_step_kv 总和/均值。"""
    from mlc_vla.sample import euler_loop, make_time_embs

    cfg = runner.config
    tvm = runner._tvm
    dt = cfg.dtype

    _device_sync()
    t0 = time.perf_counter()
    keys, values, prefix_mask, suffix_cos, suffix_sin, _ = runner._prepare(
        params, prefix_embs, prefix_pad
    )
    _device_sync()
    prefill_ms = (time.perf_counter() - t0) * 1e3

    noise_np = runner._noise(noise, 0)
    time_embs = make_time_embs(num_steps, cfg.action_expert.width)
    te_dev = [tvm.runtime.tensor(te.astype(dt), runner.dev) for te in time_embs]
    step_ms: list[float] = []

    def step_fn(x_np, i):
        x_dev = tvm.runtime.tensor(x_np.astype(dt), runner.dev)
        _device_sync()
        ts = time.perf_counter()
        v = runner.vm["denoise_step_kv"](
            keys, values, x_dev, te_dev[i], suffix_cos, suffix_sin, prefix_mask, params
        )
        _device_sync()
        step_ms.append((time.perf_counter() - ts) * 1e3)
        return v.numpy() if hasattr(v, "numpy") else v[0].numpy()

    t1 = time.perf_counter()
    actions = euler_loop(step_fn, noise_np, num_steps)
    # denoise_ms = sum of per-step (already synced); also record wall of euler_loop host overhead
    denoise_ms = float(sum(step_ms)) if step_ms else (time.perf_counter() - t1) * 1e3
    mean_step = denoise_ms / float(len(step_ms)) if step_ms else 0.0

    timings = {
        "tvm_prefill_ms": prefill_ms,
        "tvm_denoise_ms": denoise_ms,
        "tvm_denoise_step_mean_ms": mean_step,
        "tvm_denoise_step_ms": list(step_ms),
        "tvm_worker_ms": prefill_ms + denoise_ms,
        "loop": False,
        "num_steps": int(num_steps),
    }
    return actions, timings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--target", default="cuda")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="开启整段去噪环的 CUDA Graph 捕获（配合图内环 denoise_loop_kv）")
    ap.add_argument("--cublas", action=argparse.BooleanOptionalAction, default=None,
                    help="cuBLAS+FuseTransposeMatmul（Phase B）；默认自动（CUDA 且扩展可用即开）")
    args = ap.parse_args()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    try:
        # Fail fast with actionable guidance when tvm_ffi is missing (common on Thor).
        try:
            import tvm_ffi  # noqa: F401
        except ImportError as exc:
            tvm_home = os.environ.get("TVM_HOME", "<unset>")
            raise ImportError(
                "Cannot import tvm_ffi (required by modern TVM). "
                f"TVM_HOME={tvm_home!r} PYTHONPATH={os.environ.get('PYTHONPATH', '')!r}. "
                "Fix on the worker Python (MLC_VLA_PY):\n"
                "  1) Ensure $TVM_HOME/python/tvm_ffi exists (from a prior TVM/ffi install), OR\n"
                "  2) pip install -e $TVM_HOME/3rdparty/tvm-ffi"
                "     (NOTE: install the tvm-ffi repo root, NOT .../python)\n"
                "  3) export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/3rdparty/tvm-ffi/python:"
                "$MLC_VLA_HOME/python:$PYTHONPATH\n"
                f"Original error: {exc}"
            ) from exc

        from mlc_vla.model.pi0 import Pi0Config, pi0_loader
        from mlc_vla.sample import PiZeroRunner
        from pathlib import Path

        config = Pi0Config.from_openpi_config(args.checkpoint_dir, dtype=args.dtype)
        runner = PiZeroRunner(config, args.target, cuda_graph=args.cuda_graph, cublas=args.cublas)
        ckpt_file = Path(args.checkpoint_dir) / "model.safetensors"
        raw = pi0_loader.load_safetensors(str(ckpt_file), dtype="float32")
        src = pi0_loader.load_params(config, raw, named_params=runner.named_params, dtype=args.dtype)
        del raw
        arrays = [src[name] for name, _ in runner.named_params]
        param_bytes = int(sum(a.nbytes for a in arrays))
        params = runner.to_params(arrays)
    except Exception as e:  # noqa: BLE001
        import traceback
        _send(stdout, {"ready": False, "error": f"{e}\n{traceback.format_exc()}"})
        return

    _send(stdout, {
        "ready": True,
        "prefix_len": int(config.prefix_len),
        "suffix_len": int(config.suffix_len),
        "action_horizon": int(config.action_horizon),
        "action_dim": int(config.action_dim),
        "num_denoise_steps": int(config.num_denoise_steps),
        "cuda_graph": bool(args.cuda_graph),
        "cublas": bool(getattr(runner, "cublas", False)),
        "param_bytes": param_bytes,
    })

    while True:
        req = _recv(stdin)
        if req is None or req.get("op") == "shutdown":
            break
        try:
            if req["op"] == "sample":
                num_steps = int(req.get("num_steps", config.num_denoise_steps))
                use_loop = bool(req.get("loop", True)) and num_steps == config.num_denoise_steps
                timed = bool(req.get("timed", False))
                prefix_embs = req["prefix_embs"]
                prefix_pad = req.get("prefix_pad")
                noise = req.get("noise")

                if timed:
                    if use_loop:
                        actions, timings = _sample_timed_loop(
                            runner, params, prefix_embs, noise, prefix_pad
                        )
                    else:
                        actions, timings = _sample_timed_steps(
                            runner, params, prefix_embs, noise, prefix_pad, num_steps
                        )
                elif use_loop:
                    actions = runner.sample_graph(
                        params, prefix_embs, noise=noise, prefix_pad=prefix_pad,
                    )
                    timings = {}
                else:
                    actions = runner.sample(
                        params, prefix_embs, noise=noise, num_steps=num_steps,
                        prefix_pad=prefix_pad,
                    )
                    timings = {}
                _send(stdout, {"ok": True, "actions": actions, "timings": timings})
            else:
                _send(stdout, {"ok": False, "error": f"unknown op {req.get('op')!r}"})
        except Exception as e:  # noqa: BLE001
            import traceback
            _send(stdout, {"ok": False, "error": f"{e}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    main()
