"""Unit tests for Qwen3-ASR parse / text_norm / streaming helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chameleon.evaluate.text_norm import (
    char_error_rate,
    normalize_asr_text,
    word_error_rate,
)
from chameleon.models.qwen3_asr.adapter import (
    build_force_language_suffix,
    parse_asr_output,
)
from chameleon.runtime.edgellm.streaming import (
    AsrStreamingState,
    feed_pcm,
    finish_stream,
)


def test_parse_asr_output_with_language_tag() -> None:
    lang, text = parse_asr_output("language English<asr_text>hello world")
    assert lang == "English"
    assert text == "hello world"


def test_parse_asr_output_forced_language() -> None:
    lang, text = parse_asr_output("hello", user_language="Chinese")
    assert lang == "Chinese"
    assert text == "hello"


def test_parse_asr_output_empty() -> None:
    assert parse_asr_output("") == ("", "")
    assert parse_asr_output(None) == ("", "")  # type: ignore[arg-type]


def test_force_language_suffix() -> None:
    assert build_force_language_suffix(None) == ""
    assert build_force_language_suffix("English") == "language English<asr_text>"


def test_normalize_and_wer() -> None:
    assert normalize_asr_text("Hello, World!") == "hello world"
    assert word_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello world", "hello there") == pytest.approx(0.5)
    assert char_error_rate("abc", "abc") == 0.0
    assert char_error_rate("你好", "你好", lang="Chinese") == 0.0


def test_streaming_chunk_refeed(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def infer_fn(audio_path: str, context: str, language: str | None) -> str:
        calls.append((audio_path, context))
        n = len(calls)
        return f"language English<asr_text>chunk{n}"

    state = AsrStreamingState(
        chunk_size_sec=1.0,
        sample_rate=16000,
        unfixed_chunk_num=1,
        unfixed_token_num=1,
    )
    # 2.5 seconds → 2 full chunks + flush
    pcm = np.zeros(int(16000 * 2.5), dtype=np.float32)
    events: list[dict] = []
    feed_pcm(state, pcm, infer_fn, tmp_dir=tmp_path, on_update=events.append)
    finish_stream(state, infer_fn, tmp_dir=tmp_path, on_update=events.append)
    assert len(calls) == 3
    assert state.chunk_id == 3
    assert "chunk" in state.text
    assert events[-1]["fixed_text"] is not None
