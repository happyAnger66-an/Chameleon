"""NVIDIA TensorRT compiler backend (first-class target, scaffolded).

Implements the full call structure for building a TensorRT engine from an ONNX
graph, including custom-plugin shared-library preloading (see
``model_optimizer``'s ``_load_trt_plugin_shared_libraries``) and dual
context/generation optimization profiles for prefill/decode separation. The
heavy ``trt.Builder`` calls run when ``tensorrt`` is importable; otherwise a
clear error is raised describing what an on-device build would do.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import Sequence

from chameleon.compile.base import CompilerBackend, register_compiler
from chameleon.core.artifact import Artifact
from chameleon.core.context import CompileContext
from chameleon.quantization.base import QuantMetadata

logger = logging.getLogger(__name__)


class TensorRTCompiler(CompilerBackend):
    name = "tensorrt"

    def available(self) -> bool:
        try:
            import tensorrt  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False

    def _preload_plugins(self, plugin_libs: Sequence[str]) -> None:
        # Plugins must be loaded RTLD_GLOBAL before the ONNX parser runs so the
        # parser can resolve custom op symbols (matches model_optimizer). Missing
        # libraries (plugins not yet built) are skipped with a warning so the
        # build of plugin-free graphs still proceeds.
        for lib in plugin_libs:
            if not Path(lib).exists():
                logger.warning("TRT plugin library not found, skipping preload: %s", lib)
                continue
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            logger.info("Loaded TRT plugin library: %s", lib)

    def compile(
        self,
        graph: Artifact,
        quant_meta: QuantMetadata | None,
        ctx: CompileContext,
        cfg: dict | None = None,
    ) -> Artifact:
        cfg = cfg or {}
        if graph.kind != "onnx" or not graph.path:
            raise ValueError("TensorRTCompiler expects an ONNX graph artifact with a path.")

        engine_path = Path(ctx.output_dir) / f"{graph.stage or 'model'}.engine"

        if not self.available():
            raise RuntimeError(
                "TensorRT is not available in this environment. On an NVIDIA "
                f"target ({ctx.platform.name}) this step would: (1) preload custom "
                "plugin .so files, (2) parse the ONNX graph, (3) apply quant flags "
                f"from quant_meta={quant_meta}, (4) set context+generation optimization "
                f"profiles, and (5) build {engine_path}."
            )

        import tensorrt as trt  # type: ignore

        self._preload_plugins(cfg.get("plugin_libs", []))

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)
        # parse_from_file (not parse(bytes)) so the parser can resolve any
        # external weight data files (``*.onnx.data``) next to the model.
        if not parser.parse_from_file(graph.path):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

        build_config = builder.create_builder_config()
        self._apply_precision_flags(builder, build_config, quant_meta, ctx)
        if "workspace_mb" in cfg:
            build_config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, int(cfg["workspace_mb"]) << 20
            )
        num_profiles = self._add_optimization_profiles(builder, build_config, cfg)

        serialized = builder.build_serialized_network(network, build_config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build returned None.")
        engine_path.write_bytes(serialized)

        return Artifact(
            kind="engine",
            stage=graph.stage,
            platform=ctx.platform.name,
            path=str(engine_path),
            metadata={
                "compiler": self.name,
                "quant": quant_meta.component_dtypes if quant_meta else {},
                "num_profiles": num_profiles,
            },
        )

    @staticmethod
    def _add_optimization_profiles(builder, build_config, cfg: dict) -> int:
        """Add optimization profiles for dynamic shapes.

        ``cfg["profiles"]`` is a list of profiles (e.g. a context/prefill profile
        and a generation/decode profile). Each profile maps an input tensor name
        to ``{"min": [...], "opt": [...], "max": [...]}``. Static-shape engines
        (the pi05 reference path) need no profiles, so this is a no-op then.
        """
        profiles = cfg.get("profiles") or []
        count = 0
        for profile_spec in profiles:
            profile = builder.create_optimization_profile()
            for tensor_name, dims in profile_spec.items():
                profile.set_shape(
                    tensor_name,
                    min=tuple(dims["min"]),
                    opt=tuple(dims["opt"]),
                    max=tuple(dims["max"]),
                )
            build_config.add_optimization_profile(profile)
            count += 1
        return count

    @staticmethod
    def _apply_precision_flags(builder, build_config, quant_meta, ctx) -> None:
        import tensorrt as trt  # type: ignore

        dtypes = quant_meta.component_dtypes if quant_meta else {}
        if "fp16" in ctx.platform.dtypes:
            build_config.set_flag(trt.BuilderFlag.FP16)
        if dtypes.get("weight") == "int8" or dtypes.get("activation") == "int8":
            build_config.set_flag(trt.BuilderFlag.INT8)
        if dtypes.get("weight") == "fp8" and hasattr(trt.BuilderFlag, "FP8"):
            build_config.set_flag(trt.BuilderFlag.FP8)


register_compiler(TensorRTCompiler(), override=True)
