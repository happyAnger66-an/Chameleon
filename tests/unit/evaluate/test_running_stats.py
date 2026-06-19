"""RunningDimDiffStats 单元测试。"""

from __future__ import annotations

import numpy as np

from chameleon.evaluate.running_stats import RunningDimDiffStats


def test_running_dim_diff_means() -> None:
    rs = RunningDimDiffStats(action_dim=3)
    rs.update(np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.2, 0.0]))
    rs.update(np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.0]))
    assert rs.step_count == 2
    assert rs.mae_dim_mean()[0] == 0.2  # (0.1 + 0.3) / 2
    assert rs.mae_dim_mean()[1] == 0.1  # (0.2 + 0) / 2
    assert rs.mse_dim_mean()[0] == 0.05  # (0.01 + 0.09) / 2
    payload = rs.metrics_payload()
    assert len(payload["mae_dim_mean"]) == 3
    assert payload["diff_step_count"] == 2
