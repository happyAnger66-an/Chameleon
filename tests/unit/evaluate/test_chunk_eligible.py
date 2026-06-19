"""chunk_eligible 单元测试 — WebUI global_index 单调性前提。"""

from __future__ import annotations

import numpy as np

from chameleon.evaluate.lerobot_eval import chunk_eligible
from tests.helpers.fakes import FakeDataSource


def test_aligned_chunks_eligible() -> None:
    ds = FakeDataSource(length=25, start_index=0, action_horizon=10, frame_count=100)
    assert chunk_eligible(ds, 0) == 0
    assert chunk_eligible(ds, 10) == 0


def test_misaligned_index_rejected() -> None:
    ds = FakeDataSource(length=25, action_horizon=10)
    assert chunk_eligible(ds, 1) is None
    assert chunk_eligible(ds, 5) is None


def test_episode_boundary_rejected() -> None:
    ds = FakeDataSource(length=25, action_horizon=10, frame_count=100)
    ep = ds.episode_ids_per_frame.copy()
    ep[5:15] = 1
    ds._episode_ids = ep  # noqa: SLF001 — 测试边界
    assert chunk_eligible(ds, 0) is None
