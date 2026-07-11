"""LibriSpeech test-clean ASR dataset spec."""

from __future__ import annotations

from chameleon.dataloader.base import DatasetSpec, register_dataset

register_dataset(
    DatasetSpec(
        name="librispeech_test_clean",
        loader="asr_hf",
        openpi_config="",
        extra={
            "hf_name": "librispeech_asr",
            "config": "clean",
            "split": "test.clean",
        },
    ),
    override=True,
)
