"""非 NVIDIA 平台编译后端脚手架 — OpenVINO / TVM / 地平线 BPU。

作用：
    注册 OpenVINOCompiler、TVMCompiler、HorizonCompiler 三个 scaffold
    后端，声明各平台的集成方案并在 compile() 时抛出 NotImplementedError。
    周边流水线（frontend、quant 契约、runtime registry）已平台无关。

架构位置：
    优化/编译流水线 — compile/ 的占位实现，对应阶段三/四路线图
    （Intel OpenVINO、AMD/CPU TVM、地平线 BPU BYOC）。
"""

from __future__ import annotations

from chameleon.compile.base import CompilerBackend, register_compiler
from chameleon.core.artifact import Artifact
from chameleon.core.context import CompileContext
from chameleon.quantization.base import QuantMetadata


class _ScaffoldCompiler(CompilerBackend):
    #: Human description of how this backend would lower a graph.
    plan: str = ""

    def available(self) -> bool:
        return False

    def compile(
        self,
        graph: Artifact,
        quant_meta: QuantMetadata | None,
        ctx: CompileContext,
        cfg: dict | None = None,
    ) -> Artifact:
        raise NotImplementedError(
            f"[{self.name}] compiler backend is scaffolded, not yet implemented.\n"
            f"Integration plan: {self.plan}\n"
            f"Inputs ready: graph={graph.kind}@{graph.path}, "
            f"quant={quant_meta.method if quant_meta else None}, platform={ctx.platform.name}."
        )


class OpenVINOCompiler(_ScaffoldCompiler):
    name = "openvino"
    plan = (
        "Convert ONNX -> OpenVINO IR via openvino.convert_model; apply INT8 PTQ "
        "with NNCF using quant_meta; serialize .xml/.bin for Intel CPU/GPU."
    )


class TVMCompiler(_ScaffoldCompiler):
    name = "tvm"
    plan = (
        "Import ONNX -> Relax IRModule; run relax.get_pipeline with the platform "
        "Target (amd rocm / llvm cpu); DLight auto-schedule; relax.build -> .so. "
        "Use this path for AMD GPU and generic CPU."
    )


class HorizonCompiler(_ScaffoldCompiler):
    name = "horizon"
    plan = (
        "Either (a) TVM BYOC: pattern-match BPU-supported subgraphs and codegen "
        "via Horizon SDK, or (b) Horizon hb_mapper: ONNX -> .bin quantized model. "
        "INT8-only target (Journey BPU)."
    )


def _register_all() -> None:
    for backend in (OpenVINOCompiler(), TVMCompiler(), HorizonCompiler()):
        register_compiler(backend, override=True)


_register_all()
