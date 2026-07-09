"""TVM denoise 工作进程（Python 3.12 + tvm + mlc-vla）。

openpi 评测跑在 Python 3.11 venv，而本仓库的 tvm_ffi 需 Python 3.12，二者不能同进程。
本 worker 由 3.11 侧以 3.12 解释器启动，通过 stdin/stdout（4 字节长度前缀 + pickle）通信：
  - 启动后建好 mlc-vla M1 engine（prefill + denoise_step_kv）+ 载权重，回一帧 {"ready":True,...}
  - 收到 {"op":"sample", prefix_embs, prefix_pad, noise, num_steps} → 回 {"ok":True, actions}

用法（由 client 自动拉起）：
    python3.12 -m chameleon.runtime.pi05_tvm.worker --checkpoint-dir DIR --dtype bfloat16 --target cuda
（worker 只依赖 mlc_vla + tvm，PYTHONPATH 无需含 chameleon；也可用文件路径直接跑。）
"""

from __future__ import annotations

import argparse
import pickle
import struct
import sys


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--target", default="cuda")
    args = ap.parse_args()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    try:
        from mlc_vla.model.pi0 import Pi0Config, pi0_loader
        from mlc_vla.sample import PiZeroRunner
        from pathlib import Path

        config = Pi0Config.from_openpi_config(args.checkpoint_dir, dtype=args.dtype)
        runner = PiZeroRunner(config, args.target)
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
        "param_bytes": param_bytes,
    })

    while True:
        req = _recv(stdin)
        if req is None or req.get("op") == "shutdown":
            break
        try:
            if req["op"] == "sample":
                actions = runner.sample(
                    params,
                    req["prefix_embs"],
                    noise=req.get("noise"),
                    num_steps=int(req.get("num_steps", 10)),
                    prefix_pad=req.get("prefix_pad"),
                )
                _send(stdout, {"ok": True, "actions": actions})
            else:
                _send(stdout, {"ok": False, "error": f"unknown op {req.get('op')!r}"})
        except Exception as e:  # noqa: BLE001
            import traceback
            _send(stdout, {"ok": False, "error": f"{e}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    main()
