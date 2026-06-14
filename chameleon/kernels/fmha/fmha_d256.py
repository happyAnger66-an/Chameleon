"""Example custom op: fused multi-head attention with head_dim=256.

pi05's PaliGemma uses head_dim=256, which is past the fast path of many stock
attention kernels -- hence ``model_optimizer`` ships a dedicated ``fmha_d256``
CuTe DSL plugin. Here we show the cross-platform shape: one :class:`OpSpec` with
per-vendor :class:`KernelImpl`s. The CPU implementation is a real PyTorch
reference; the NVIDIA implementation points at a (to-be-built) TRT plugin.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from chameleon.kernels.base import KernelImpl, OpSpec, register_kernel, register_op

FMHA_D256 = OpSpec(
    name="fmha_d256",
    description="Multi-head attention specialized for head_dim=256.",
    attributes={"head_dim": 256},
)
register_op(FMHA_D256, override=True)


def _sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q,k,v: [B, num_heads, S, head_dim]
    return F.scaled_dot_product_attention(q, k, v)


class CpuFmhaD256(KernelImpl):
    op = "fmha_d256"
    platform_vendor = "cpu"

    def reference(self, q, k, v):
        return _sdpa_reference(q, k, v)


class NvidiaFmhaD256(KernelImpl):
    op = "fmha_d256"
    platform_vendor = "nvidia"

    def frontend_stub(self):
        # Pattern (see model_optimizer ops/fmha_d256_attention_plugin.py):
        #   @torch.library.custom_op("chameleon::fmha_d256", mutates_args=())
        #   def fmha_d256(q, kv_cache, ...): ...  # eager dummy
        # plus an ONNX symbolic emitting g.op("trt::FmhaD256AttentionPlugin", ...).
        return None

    def graph_node(self, g, q, kv_cache, *attrs):
        return g.op("trt::FmhaD256AttentionPlugin", q, kv_cache, outputs=2)

    def backend_artifact(self, kernel_tag: str | None = None):
        # Resolves to a CuTe DSL artifact selected by SM tag (e.g. sm_87 / sm_101).
        return {"plugin": "libfmha_d256.so", "kernel_tag": kernel_tag}

    def reference(self, q, k, v):
        return _sdpa_reference(q, k, v)


def _register_all() -> None:
    register_kernel(CpuFmhaD256(), override=True)
    register_kernel(NvidiaFmhaD256(), override=True)


_register_all()
