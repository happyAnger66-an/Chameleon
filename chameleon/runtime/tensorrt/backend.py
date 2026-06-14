"""TensorRT runtime.

Loads a serialized TRT engine and executes it through a declarative
:class:`TensorRegistry` (cf. TensorRT-Edge-LLM's ``TensorRegistry`` /
``EngineExecutor``). Inputs are bound *positionally* -- the orchestrator passes a
dict whose value order matches the stage's forward signature, which in turn
matches the ONNX export input order -- so binding is robust to ONNX tensor
renaming. Persistent device buffers are cached per shape so the action-expert
denoise loop (same shapes every step) reuses allocations and is CUDA-graph
capturable.

Requires CUDA torch tensors (uses ``data_ptr()``); the NVIDIA platforms set
``torch_device = "cuda"``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.runtime.base import Engine, RuntimeBackend, register_runtime

logger = logging.getLogger(__name__)


def _trt_to_torch_dtype(trt_dtype) -> torch.dtype:
    import tensorrt as trt  # type: ignore

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    # Newer TRT versions expose additional dtypes; add them when present.
    for attr, torch_dt in (("INT64", torch.int64), ("BF16", torch.bfloat16), ("FP8", torch.float8_e4m3fn)):
        if hasattr(trt.DataType, attr):
            mapping[getattr(trt.DataType, attr)] = torch_dt
    return mapping.get(trt_dtype, torch.float32)


class TensorRegistry:
    """Declarative view of an engine's I/O tensors, with cached device buffers.

    Discovers input/output tensor names in definition order and lazily allocates
    persistent CUDA buffers keyed by (name, shape) for reuse across calls.
    """

    def __init__(self, engine, device: str = "cuda") -> None:
        import tensorrt as trt  # type: ignore

        self.engine = engine
        self.device = device
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        self._buffers: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}

    def buffer(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        key = (name, shape)
        buf = self._buffers.get(key)
        if buf is None or buf.dtype != dtype:
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._buffers[key] = buf
        return buf


class TensorRTEngine(Engine):
    """A loaded TRT engine exposing the unified ``run(inputs) -> outputs`` API."""

    def __init__(
        self,
        engine: Any,
        stage: str | None,
        device: str,
        *,
        enable_cuda_graph: bool = False,
        profile_index: int = 0,
    ) -> None:
        self._engine = engine
        self.stage = stage
        self.device = device
        self._context = engine.create_execution_context()
        # Select an optimization profile (e.g. context/prefill vs generation/decode)
        # when the engine was built with more than the implicit profile.
        if getattr(engine, "num_optimization_profiles", 1) > 1:
            self._context.set_optimization_profile_async(
                profile_index, torch.cuda.current_stream().cuda_stream
            )
            torch.cuda.current_stream().synchronize()
        self._registry = TensorRegistry(engine, device=device)
        self._enable_cuda_graph = enable_cuda_graph
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_inputs: list[torch.Tensor] = []
        self._graph_outputs: dict[str, torch.Tensor] = {}

    def _bind_inputs(self, inputs: dict[str, Any]) -> list[torch.Tensor]:
        # Positional binding: dict value order matches the stage signature.
        values = list(inputs.values())
        if len(values) != len(self._registry.input_names):
            raise ValueError(
                f"Stage {self.stage!r} expects {len(self._registry.input_names)} inputs, "
                f"got {len(values)}."
            )
        bound: list[torch.Tensor] = []
        for name, value in zip(self._registry.input_names, values, strict=True):
            trt_dtype = self._engine.get_tensor_dtype(name)
            torch_dtype = _trt_to_torch_dtype(trt_dtype)
            t = value
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            t = t.to(device=self.device, dtype=torch_dtype).contiguous()
            self._context.set_input_shape(name, tuple(t.shape))
            self._context.set_tensor_address(name, t.data_ptr())
            bound.append(t)
        return bound

    def _bind_outputs(self) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for name in self._registry.output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            torch_dtype = _trt_to_torch_dtype(self._engine.get_tensor_dtype(name))
            out = self._registry.buffer(name, shape, torch_dtype)
            self._context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out
        return outputs

    def _normalize_outputs(self, outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        # The orchestrator consumes ``["output"]``; expose a single output under
        # that canonical key while still returning the named dict for multi-output.
        if len(outputs) == 1:
            return {"output": next(iter(outputs.values()))}
        result: dict[str, Any] = dict(outputs)
        result.setdefault("output", next(iter(outputs.values())))
        return result

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        bound = self._bind_inputs(inputs)
        outputs = self._bind_outputs()

        if self._enable_cuda_graph and self.device.startswith("cuda"):
            return self._run_with_graph(bound, outputs)

        stream = torch.cuda.current_stream()
        ok = self._context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"TRT execute_async_v3 failed for stage {self.stage!r}.")
        stream.synchronize()
        return self._normalize_outputs({k: v.clone() for k, v in outputs.items()})

    def _run_with_graph(self, bound: list[torch.Tensor], outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        # Best-effort CUDA-graph capture. Buffers are persistent (same data_ptr),
        # so after the first capture we copy new inputs into the captured buffers
        # and replay. Falls back to plain execution on any capture error.
        try:
            if self._graph is None:
                # Warmup once so TRT finalizes kernel selection before capture.
                self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                torch.cuda.current_stream().synchronize()
                self._graph_inputs = bound
                self._graph_outputs = outputs
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            else:
                for captured, incoming in zip(self._graph_inputs, bound, strict=True):
                    captured.copy_(incoming)
                self._graph.replay()
                torch.cuda.current_stream().synchronize()
                return self._normalize_outputs({k: v.clone() for k, v in self._graph_outputs.items()})
            torch.cuda.current_stream().synchronize()
            return self._normalize_outputs({k: v.clone() for k, v in outputs.items()})
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA graph path failed (%s); falling back to plain execution.", exc)
            self._enable_cuda_graph = False
            self._graph = None
            stream = torch.cuda.current_stream()
            self._context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            return self._normalize_outputs({k: v.clone() for k, v in outputs.items()})


class TensorRTRuntime(RuntimeBackend):
    name = "tensorrt"

    def available(self) -> bool:
        try:
            import tensorrt  # noqa: F401

            return torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            return False

    def load(self, artifact: Artifact, ctx: RunContext) -> Engine:
        if not self.available():
            raise RuntimeError(
                "TensorRT runtime unavailable (needs tensorrt + CUDA torch). On "
                f"{ctx.platform.name} this would deserialize {artifact.path} and "
                "create an execution context."
            )
        if artifact.kind != "engine" or not artifact.path:
            raise ValueError("TensorRTRuntime expects an engine artifact with a path.")

        import tensorrt as trt  # type: ignore

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(artifact.path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine at {artifact.path}.")
        device = ctx.torch_device if ctx.torch_device.startswith("cuda") else "cuda"
        enable_graph = bool(ctx.options.get("enable_cuda_graph", False))
        profile_index = int(ctx.options.get("profile_index", 0))
        return TensorRTEngine(
            engine,
            artifact.stage,
            device,
            enable_cuda_graph=enable_graph,
            profile_index=profile_index,
        )


register_runtime(TensorRTRuntime(), override=True)
