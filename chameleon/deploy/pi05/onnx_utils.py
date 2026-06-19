"""ONNX 导出辅助 — SigLIP eager 注意力、SDPA math 路径。"""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def sdp_math_backend_only():
    """Force SDPA math path so ONNX export avoids ComplexDouble issues."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    except Exception:
        pass
    torch_cuda = getattr(torch.backends, "cuda", None)
    sdp_kernel_fn = getattr(torch_cuda, "sdp_kernel", None) if torch_cuda is not None else None
    if sdp_kernel_fn is not None:
        with sdp_kernel_fn(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            yield
    else:
        yield


@contextlib.contextmanager
def force_vision_eager_attention(vision_tower: torch.nn.Module):
    """SigLIP: use eager attention during ONNX export."""
    cfg = getattr(vision_tower, "config", None)
    if cfg is None or not hasattr(cfg, "_attn_implementation"):
        yield
        return
    saved = getattr(cfg, "_attn_implementation", None)
    try:
        setattr(cfg, "_attn_implementation", "eager")
        yield
    finally:
        if saved is not None:
            setattr(cfg, "_attn_implementation", saved)
