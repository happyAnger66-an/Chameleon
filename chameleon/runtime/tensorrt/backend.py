"""TensorRT runtime (scaffolded).

Loads a serialized TRT engine and binds I/O tensors via a declarative tensor
registry (cf. TensorRT-Edge-LLM's ``TensorRegistry`` / ``EngineExecutor``), with
support for prefill/decode optimization profiles and CUDA-graph capture. The
deserialize + execute path runs when ``tensorrt`` is importable; otherwise
loading raises a clear error.
"""

from __future__ import annotations

from typing import Any

from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.runtime.base import Engine, RuntimeBackend, register_runtime


class TensorRTEngine(Engine):
    def __init__(self, engine: Any, stage: str | None) -> None:
        self._engine = engine
        self.stage = stage

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Full impl: bind named inputs to engine bindings, select the right
        # optimization profile, enqueueV3 (or replay a captured CUDA graph),
        # then read back named outputs.
        raise NotImplementedError(
            "TensorRTEngine.run is scaffolded; bind I/O via a TensorRegistry and "
            "enqueueV3 on the selected profile."
        )


class TensorRTRuntime(RuntimeBackend):
    name = "tensorrt"

    def available(self) -> bool:
        try:
            import tensorrt  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def load(self, artifact: Artifact, ctx: RunContext) -> Engine:
        if not self.available():
            raise RuntimeError(
                "TensorRT runtime unavailable in this environment. On "
                f"{ctx.platform.name} this would deserialize {artifact.path} and "
                "create an execution context."
            )
        if artifact.kind != "engine" or not artifact.path:
            raise ValueError("TensorRTRuntime expects an engine artifact with a path.")

        import tensorrt as trt  # type: ignore

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(artifact.path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        return TensorRTEngine(engine, artifact.stage)


register_runtime(TensorRTRuntime(), override=True)
