"""Accuracy comparison between two inference outputs."""

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
