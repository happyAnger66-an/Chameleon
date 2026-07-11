"""Unit tests for Edge-LLM ASR quantize guards."""

from __future__ import annotations

import pytest

from chameleon.config.schema import DeployConfig, QuantizeStep, TaskConfig


def test_nvfp4_fp8_audio_guard_for_0_6b(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from chameleon.deploy import qwen3_asr_edgellm as mod

    monkeypatch.setattr(mod, "resolve_edgellm_home", lambda task: tmp_path)
    monkeypatch.setattr(mod, "resolve_checkpoint_dir", lambda task: tmp_path)

    task = TaskConfig(
        architecture="qwen3_asr",
        model="qwen3_asr_0.6b",
        output_dir=str(tmp_path / "out"),
        deploy=DeployConfig(backend="qwen3_asr", checkpoint_dir=str(tmp_path)),
        model_overrides={"quantization": "nvfp4", "audio_quantization": "fp8"},
        quantize=[QuantizeStep(stage="llm", method="nvfp4")],
    )
    from chameleon.core.artifact import Manifest

    with pytest.raises(ValueError, match="NVFP4 LLM \\+ FP8 audio"):
        mod.run_qwen3_asr_quantize(task, Manifest(output_dir=str(tmp_path / "out")))
