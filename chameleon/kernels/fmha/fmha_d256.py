"""fmha_d256 融合多头注意力 — head_dim=256 的自定义算子示例。

作用：
    实现 pi05 PaliGemma head_dim=256 专用 FMHA 的三段式：
    Stage 1: torch.ops.chameleon.fmha_d256 custom op（eager=SDPA 参考）
    Stage 2: ONNX symbolic → trt::FmhaD256AttentionPlugin 节点
    Stage 3: 按 PlatformSpec.kernel_tag（sm_87/sm_101）选择 CuTe DSL plugin .so

架构位置：
    算子层 — kernels/ 的参考实现，演示 OpSpec + 多平台 KernelImpl 模式。
    被 compile/tensorrt 在 build 前预加载 plugin，被 frontend 在导出时
    生成 plugin 节点。
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from chameleon.kernels.base import KernelImpl, OpSpec, register_kernel, register_op

logger = logging.getLogger(__name__)

FMHA_D256 = OpSpec(
    name="fmha_d256",
    description="Multi-head attention specialized for head_dim=256.",
    attributes={"head_dim": 256},
)
register_op(FMHA_D256, override=True)

_OP_QUALNAME = "chameleon::fmha_d256"
_custom_op_registered = False


def _sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q,k,v: [B, num_heads, S, head_dim]
    return F.scaled_dot_product_attention(q, k, v)


def _register_custom_op() -> bool:
    """Register the ``chameleon::fmha_d256`` torch custom op (idempotent)."""
    global _custom_op_registered
    if _custom_op_registered:
        return True
    try:

        @torch.library.custom_op(_OP_QUALNAME, mutates_args=())
        def fmha_d256(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # Eager path uses the SDPA reference; a fused kernel replaces this at
            # the graph/backend level on supported platforms.
            return _sdpa_reference(q, k, v)

        @fmha_d256.register_fake
        def _(q, k, v):  # shape/dtype propagation for export/compile
            return torch.empty_like(q)

        _custom_op_registered = True
    except Exception as exc:  # noqa: BLE001 - older torch / re-registration
        logger.debug("fmha_d256 custom op registration skipped: %s", exc)
        _custom_op_registered = False
    return _custom_op_registered


def fmha_d256(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Public entry point. Uses the registered custom op when available."""
    if _register_custom_op():
        return torch.ops.chameleon.fmha_d256(q, k, v)
    return _sdpa_reference(q, k, v)


class CpuFmhaD256(KernelImpl):
    op = "fmha_d256"
    platform_vendor = "cpu"

    def frontend_stub(self):
        _register_custom_op()
        return fmha_d256

    def reference(self, q, k, v):
        return _sdpa_reference(q, k, v)


class NvidiaFmhaD256(KernelImpl):
    op = "fmha_d256"
    platform_vendor = "nvidia"

    def frontend_stub(self):
        _register_custom_op()
        return fmha_d256

    def graph_node(self, g, q, k, v, *attrs):
        # Emitted into the ONNX graph; the TRT compiler resolves the plugin from
        # backend_artifact() and preloads its .so before parsing.
        return g.op("trt::FmhaD256AttentionPlugin", q, k, v, head_dim_i=256)

    def backend_artifact(self, kernel_tag: str | None = None):
        # Resolves to a CuTe DSL artifact selected by SM tag (e.g. sm_87 / sm_101).
        return {"plugin": "libfmha_d256.so", "kernel_tag": kernel_tag}

    def reference(self, q, k, v):
        return _sdpa_reference(q, k, v)


def _register_all() -> None:
    register_kernel(CpuFmhaD256(), override=True)
    register_kernel(NvidiaFmhaD256(), override=True)


_register_all()
