"""评测流式累计统计 — 供 WebUI 展示各 action dim 的 diff MAE/MSE 均值。"""

from __future__ import annotations

import numpy as np


class RunningDimDiffStats:
    """对已推送的每个 step（单行动作向量）累计 per-dim MAE/MSE 均值。"""

    def __init__(self, action_dim: int) -> None:
        d = max(1, int(action_dim))
        self._dim = d
        self._mae_sum = np.zeros(d, dtype=np.float64)
        self._mse_sum = np.zeros(d, dtype=np.float64)
        self._count = 0

    @property
    def step_count(self) -> int:
        return self._count

    def update(self, gt_row: np.ndarray, pred_row: np.ndarray) -> None:
        diff = np.asarray(pred_row, dtype=np.float64) - np.asarray(gt_row, dtype=np.float64)
        flat = np.ravel(diff)
        n = min(flat.size, self._dim)
        if n < self._dim:
            pad_mae = np.zeros(self._dim, dtype=np.float64)
            pad_mse = np.zeros(self._dim, dtype=np.float64)
            pad_mae[:n] = np.abs(flat[:n])
            pad_mse[:n] = flat[:n] * flat[:n]
            self._mae_sum += pad_mae
            self._mse_sum += pad_mse
        else:
            self._mae_sum += np.abs(flat[: self._dim])
            self._mse_sum += flat[: self._dim] * flat[: self._dim]
        self._count += 1

    def mae_dim_mean(self) -> list[float]:
        if self._count == 0:
            return [0.0] * self._dim
        return [float(x) for x in (self._mae_sum / self._count).tolist()]

    def mse_dim_mean(self) -> list[float]:
        if self._count == 0:
            return [0.0] * self._dim
        return [float(x) for x in (self._mse_sum / self._count).tolist()]

    def scalar_mae_mean(self) -> float:
        """所有 dim、所有已见 step 的 |diff| 全局均值。"""
        if self._count == 0:
            return 0.0
        return float(np.mean(self._mae_sum / self._count))

    def scalar_mse_mean(self) -> float:
        if self._count == 0:
            return 0.0
        return float(np.mean(self._mse_sum / self._count))

    def metrics_payload(self) -> dict[str, list[float] | float | int]:
        return {
            "mae_dim_mean": self.mae_dim_mean(),
            "mse_dim_mean": self.mse_dim_mean(),
            "mae_cum": self.scalar_mae_mean(),
            "mse_cum": self.scalar_mse_mean(),
            "diff_step_count": int(self._count),
        }
