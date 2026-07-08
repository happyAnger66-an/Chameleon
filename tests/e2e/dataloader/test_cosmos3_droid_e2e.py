"""cosmos3 DROID 数据源 E2E — registry / loader / spec / 懒加载错误路径。"""

from __future__ import annotations

import pytest

from chameleon.dataloader import build_dataset, get_dataset_spec, list_datasets
from chameleon.dataloader.base import get_loader
from chameleon.dataloader.cosmos3_droid import (
    DROID_ACTION_HORIZON,
    DROID_RAW_ACTION_DIM,
    DroidRldsDataSource,
)


@pytest.mark.e2e
class TestCosmos3DroidRegistryE2E:
    def test_dataset_registered(self) -> None:
        assert "cosmos3_droid" in list_datasets()

    def test_spec_fields(self) -> None:
        spec = get_dataset_spec("cosmos3_droid")
        assert spec.loader == "droid_rlds"
        assert spec.action_horizon == DROID_ACTION_HORIZON
        assert spec.extra["builder"] == "droid_100"
        assert spec.extra["raw_action_dim"] == DROID_RAW_ACTION_DIM

    def test_loader_registered(self) -> None:
        assert get_loader("droid_rlds") is DroidRldsDataSource

    def test_build_dataset_returns_droid_source(self) -> None:
        src = build_dataset("cosmos3_droid", overrides={"dataset_root": "/tmp/does-not-matter"})
        assert isinstance(src, DroidRldsDataSource)
        # 懒加载：未 build 前即可读元信息
        assert src.action_horizon == DROID_ACTION_HORIZON
        assert src.action_dim == DROID_RAW_ACTION_DIM
        assert src.repo_id == "droid_100"

    def test_build_requires_dataset_root(self) -> None:
        src = build_dataset("cosmos3_droid")  # 无 dataset_root
        with pytest.raises(ValueError, match="dataset_root"):
            src.build()

    def test_overrides_apply(self) -> None:
        src = build_dataset(
            "cosmos3_droid",
            overrides={"dataset_root": "/data", "start_index": 5, "num_samples": 3},
        )
        assert src.start_index == 5
        assert src.spec.num_samples == 3
