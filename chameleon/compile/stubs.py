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
    #: π0.5 中已由 mlc-vla 落地（TVM Relax nn 前端 + 编译 pipeline）的 stage。
    _MLC_VLA_STAGES = ("denoise", "action_expert")

    def available(self) -> bool:
        from chameleon.compile import tvm_mlc_vla

        return tvm_mlc_vla.available()

    def compile(
        self,
        graph: Artifact,
        quant_meta: QuantMetadata | None,
        ctx: CompileContext,
        cfg: dict | None = None,
    ) -> Artifact:
        stage = graph.stage or (cfg or {}).get("stage")
        # 第一阶段：只有 denoise stage 走 mlc-vla；其余 stage 仍标注为后续。
        if stage not in self._MLC_VLA_STAGES:
            raise NotImplementedError(
                f"[tvm] stage {stage!r} 尚未落地（vit/llm/去噪环留后续）。"
                f"当前仅支持 {self._MLC_VLA_STAGES}（经 mlc-vla 编译）。\n"
                f"Integration plan: {self.plan}"
            )

        from chameleon.compile import tvm_mlc_vla

        cfg = cfg or {}
        checkpoint_dir = (
            cfg.get("checkpoint_dir")
            or ctx.options.get("checkpoint_dir")
            or graph.metadata.get("checkpoint_dir")
        )
        if not checkpoint_dir:
            raise ValueError(
                "[tvm] denoise 编译需要 checkpoint_dir（step.options / ctx.options / graph.metadata）。"
            )
        target = tvm_mlc_vla.resolve_target(ctx.platform.device, cfg)
        # mode: "M0"=每步联合 denoise_step；"M1"=prefix 固化 + suffix-only denoise_step_kv（默认）。
        mode = str(cfg.get("mode", "M1")).upper()
        if mode == "M1":
            meta = tvm_mlc_vla.compile_denoise_kv(
                checkpoint_dir=str(checkpoint_dir),
                output_dir=str(ctx.output_dir),
                target=target,
                dtype=cfg.get("tvm_dtype"),
                cuda_graph=bool(cfg.get("cuda_graph", True)),
                # quant: group 量化预设（如 "q4bf16_1"）；None=fp。M1+ 显存优化选项。
                quant=cfg.get("quant"),
                verify=bool(cfg.get("verify", False)),
                threshold=float(cfg.get("threshold", 0.99)),
            )
        else:
            meta = tvm_mlc_vla.compile_denoise(
                checkpoint_dir=str(checkpoint_dir),
                output_dir=str(ctx.output_dir),
                target=target,
                dtype=cfg.get("tvm_dtype"),
                verify=bool(cfg.get("verify", False)),
                threshold=float(cfg.get("threshold", 0.99)),
            )
        return Artifact(
            kind="engine",
            stage=stage,
            platform=ctx.platform.name,
            path=meta["engine"],
            metadata=meta,
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
