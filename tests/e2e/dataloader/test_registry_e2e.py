"""dataloader 层 E2E — registry 与 DatasetSpec。"""

from __future__ import annotations

import pytest

from chameleon.dataloader import get_dataset_spec, list_datasets
from chameleon.dataloader.base import ChameleonSample


@pytest.mark.e2e
class TestDataloaderRegistryE2E:
    def test_pi05_libero_registered(self) -> None:
        names = list_datasets()
        assert "pi05_libero" in names

    def test_dataset_spec_fields(self) -> None:
        spec = get_dataset_spec("pi05_libero")
        assert spec.loader == "lerobot"
        assert spec.openpi_config == "pi05_libero"

    def test_chameleon_sample_contract(self) -> None:
        import numpy as np

        sample = ChameleonSample(
            observation={"image": np.zeros((3, 4, 4))},
            actions_gt=np.zeros((10, 7)),
            index=3,
        )
        assert sample.index == 3
