"""cosmos3 ONNX 导出辅助 — GQA 注意力的导出期等价替换。

对标 pi05 的 ``deploy/pi05/onnx_utils.py``：pi05 用 ``_attn_implementation="eager"``
让 HF 走手动 ``repeat_kv`` 分支以绕开 ``F.scaled_dot_product_attention(enable_gqa=True)``
（TorchScript ONNX 导出器 opset14 对该 flag 直接 assert 失败）。

Cosmos3 的注意力是 diffusers 自定义 processor（``Cosmos3AttnProcessor``），不认 eager flag，
但它统一通过模块级 ``dispatch_attention_fn`` 落到 SDPA。故这里以同样的「进入替换 / 退出还原」
范式，临时把该分发函数换成 **手动扩 KV 头 + 普通 SDPA** 的等价实现（精确复刻 diffusers
``_native_attention`` 的 permute 语义），数值一致且可被 ONNX 转换。作用域仅限 cosmos3。
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """扩展 KV 头 ``[B, Hkv, S, D]`` → ``[B, Hkv*n_rep, S, D]``（HF ``repeat_kv`` 写法）。

    只用 ``expand`` + ``reshape``，不引入 index 张量 —— 避免 ``repeat_interleave`` 在
    tracing 期把 index 建到 CPU 而与 CUDA 数据设备不一致（``index_select`` 报错），也避免
    tensor→bool 的 ``TracerWarning``；expand 不复制显存，可正确导出为 ONNX。
    """
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _export_dispatch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    **kwargs,
) -> torch.Tensor:
    """``dispatch_attention_fn`` 的导出期替身。

    入参布局与 diffusers native 后端一致：``query/key/value`` 为 ``[B, S, H, D]``，内部
    permute 到 ``[B, H, S, D]`` 后调 SDPA，再 permute 回 ``[B, S, H, D]``。``enable_gqa``
    改为手动 ``repeat_interleave`` 扩展 KV 头（``n_rep = Hq // Hkv``），与 GQA 语义等价。
    忽略 ``backend`` / ``parallel_config`` 等 kwargs（导出走单卡 native 路径）。
    """
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if enable_gqa:
        hq, hkv = int(q.shape[1]), int(k.shape[1])
        if hkv > 0 and hq != hkv and hq % hkv == 0:
            n_rep = hq // hkv
            k = _repeat_kv(k, n_rep)
            v = _repeat_kv(v, n_rep)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )
    return out.transpose(1, 2)


@contextlib.contextmanager
def force_cosmos3_export_attention():
    """导出期把 cosmos3 transformer 的 ``dispatch_attention_fn`` 换成 ONNX 友好等价实现。"""
    try:
        from diffusers.models.transformers import transformer_cosmos3 as _mod
    except Exception:
        # diffusers 不含 cosmos3（reference / 旧版本）时无需替换。
        yield
        return

    saved = getattr(_mod, "dispatch_attention_fn", None)
    if saved is None:
        yield
        return
    try:
        _mod.dispatch_attention_fn = _export_dispatch_attention
        yield
    finally:
        _mod.dispatch_attention_fn = saved
