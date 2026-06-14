"""推理输出精度对比 — 计算 reference 与 candidate 动作张量的差异指标。

作用：
    提供 compare_actions()，返回 max_abs / mean_abs / cosine 三项指标。
    用于 compile→infer 闭环的数值校验（如 TRT FP16 vs PyTorch）。

架构位置：
    工具层 — evaluate/ 的具体实现，被测试脚本或手动验证调用。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ActionDiff:
    max_abs: float
    mean_abs: float
    cosine: float


def compare_actions(reference: torch.Tensor, candidate: torch.Tensor) -> ActionDiff:
    """Compare two action tensors of identical shape."""
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {candidate.shape}")
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    diff = (ref - cand).abs()
    cosine = torch.nn.functional.cosine_similarity(ref, cand, dim=0).item()
    return ActionDiff(
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        cosine=float(cosine),
    )
