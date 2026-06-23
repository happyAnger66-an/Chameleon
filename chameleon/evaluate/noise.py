"""flow-matching 初值噪声 — PT/TRT 双路对比与固定噪声评测。"""

from __future__ import annotations

import numpy as np


def flow_match_noise(
    *,
    action_horizon: int,
    action_dim: int,
    sample_index: int,
    noise_mode: str,
    noise_seed: int,
) -> np.ndarray | None:
    """``noise=fixed`` 时返回可复现噪声；``random`` 时返回 ``None``（由模型采样）。"""
    if noise_mode != "fixed":
        return None
    ss = np.random.SeedSequence([int(noise_seed), int(sample_index)])
    rng = np.random.default_rng(ss)
    return rng.standard_normal((action_horizon, action_dim), dtype=np.float32)
