"""ASR 流式状态机 V1 — 整段重喂 + 前缀回退（对齐 qwen_asr 官方语义）。

每满 ``chunk_size_sec`` 秒：累积 PCM → 写临时 wav → Edge-LLM 全量推理；
``chunk_id >= unfixed_chunk_num`` 时用历史解码文本去掉末尾 ``unfixed_token_num``
token 作为 context 前缀（通过 system context 拼接，避免依赖 pybind prompt 注入）。
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from chameleon.models.qwen3_asr.adapter import parse_asr_output

logger = logging.getLogger(__name__)

InferFn = Callable[[str, str, str | None], str]
"""(audio_path, context, language) -> raw decode text."""


@dataclass
class AsrStreamingState:
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5
    chunk_size_sec: float = 2.0
    sample_rate: int = 16000
    force_language: str | None = None
    base_context: str = ""

    chunk_id: int = 0
    buffer: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    language: str = ""
    text: str = ""
    _raw_decoded: str = ""
    tokenizer: Any | None = None

    @property
    def chunk_size_samples(self) -> int:
        return max(1, int(round(self.chunk_size_sec * self.sample_rate)))


def _to_pcm16k(pcm: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm)
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0)
    return x.astype(np.float32, copy=False)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    try:
        import soundfile as sf

        sf.write(str(path), x, int(sr))
        return
    except ImportError:
        pass
    # Stdlib fallback (PCM16 WAV) for unit tests / minimal envs.
    import wave

    pcm = np.clip(x, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())


def _encode_tokens(tokenizer: Any | None, text: str) -> list[int]:
    if not text:
        return []
    if tokenizer is not None:
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    # Whitespace fallback when no HF tokenizer.
    return list(range(len(text.split())))


def _decode_tokens(tokenizer: Any | None, ids: list[int], *, text: str) -> str:
    if not ids:
        return ""
    if tokenizer is not None:
        return str(tokenizer.decode(ids))
    words = text.split()
    return " ".join(words[: len(ids)])


def _prefix_with_rollback(state: AsrStreamingState) -> str:
    if state.chunk_id < state.unfixed_chunk_num:
        return ""
    raw = state._raw_decoded or ""
    cur_ids = _encode_tokens(state.tokenizer, raw)
    k = int(state.unfixed_token_num)
    while True:
        end_idx = max(0, len(cur_ids) - k)
        prefix = _decode_tokens(state.tokenizer, cur_ids[:end_idx], text=raw) if end_idx > 0 else ""
        if "\ufffd" not in prefix:
            return prefix
        if end_idx == 0:
            return ""
        k += 1


def _split_fixed_pending(state: AsrStreamingState) -> tuple[str, str]:
    """Split latest text into fixed / pending regions for UI."""
    full = state.text or ""
    if not full:
        return "", ""
    k = int(state.unfixed_token_num)
    if state.tokenizer is not None:
        ids = _encode_tokens(state.tokenizer, full)
        if len(ids) <= k:
            return "", full
        fixed = _decode_tokens(state.tokenizer, ids[:-k], text=full)
        pending = _decode_tokens(state.tokenizer, ids[-k:], text=full)
        return fixed, pending
    words = full.split()
    if len(words) <= k:
        return "", full
    return " ".join(words[:-k]), " ".join(words[-k:])


def _build_context(state: AsrStreamingState, prefix: str) -> str:
    parts = [p for p in (state.base_context, prefix) if p]
    return "\n".join(parts)


def _run_chunk(
    state: AsrStreamingState,
    infer_fn: InferFn,
    *,
    tmp_dir: Path,
    flush: bool = False,
) -> dict[str, Any]:
    if state.audio_accum.size == 0:
        return {
            "chunk_id": state.chunk_id,
            "language": state.language,
            "text": state.text,
            "fixed_text": "",
            "pending_text": "",
            "raw_text": state._raw_decoded,
        }

    prefix = _prefix_with_rollback(state)
    context = _build_context(state, prefix)
    wav_path = tmp_dir / f"stream_accum_{state.chunk_id}.wav"
    _write_wav(wav_path, state.audio_accum, state.sample_rate)
    raw_gen = infer_fn(str(wav_path), context, state.force_language)
    # When prefix was injected via context, model may still emit full form;
    # accumulate like official: prefix + gen when prefix used as prompt suffix.
    # Our Edge-LLM path cannot append to generation prompt easily, so treat
    # returned text as full utterance decode and keep raw as returned.
    state._raw_decoded = str(raw_gen or "")
    lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
    state.language = lang
    state.text = txt
    state.chunk_id += 1
    fixed, pending = _split_fixed_pending(state)
    return {
        "chunk_id": state.chunk_id - 1,
        "language": state.language,
        "text": state.text,
        "fixed_text": fixed,
        "pending_text": pending,
        "raw_text": state._raw_decoded,
        "flushed": flush,
    }


def feed_pcm(
    state: AsrStreamingState,
    pcm16k: np.ndarray,
    infer_fn: InferFn,
    *,
    tmp_dir: Path,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> AsrStreamingState:
    """Buffer PCM; run infer whenever a full chunk is ready."""
    x = _to_pcm16k(pcm16k)
    if x.shape[0] > 0:
        state.buffer = np.concatenate([state.buffer, x], axis=0)

    while state.buffer.shape[0] >= state.chunk_size_samples:
        chunk = state.buffer[: state.chunk_size_samples]
        state.buffer = state.buffer[state.chunk_size_samples :]
        if state.audio_accum.shape[0] == 0:
            state.audio_accum = chunk.copy()
        else:
            state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)
        evt = _run_chunk(state, infer_fn, tmp_dir=tmp_dir, flush=False)
        if on_update is not None:
            on_update(evt)
    return state


def finish_stream(
    state: AsrStreamingState,
    infer_fn: InferFn,
    *,
    tmp_dir: Path,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> AsrStreamingState:
    """Flush remaining buffer (may be shorter than one chunk)."""
    if state.buffer is None or state.buffer.shape[0] == 0:
        return state
    tail = state.buffer
    state.buffer = np.zeros((0,), dtype=np.float32)
    if state.audio_accum.shape[0] == 0:
        state.audio_accum = tail.copy()
    else:
        state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)
    evt = _run_chunk(state, infer_fn, tmp_dir=tmp_dir, flush=True)
    if on_update is not None:
        on_update(evt)
    return state


def load_audio_pcm(path: str | Path, *, sample_rate: int = 16000) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=False)
    except Exception:
        # Fallback: leave path to engine for offline; for streaming we need PCM.
        raise
    x = np.asarray(data, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if int(sr) != int(sample_rate):
        try:
            import librosa

            x = librosa.resample(x, orig_sr=int(sr), target_sr=int(sample_rate))
        except ImportError as exc:
            raise ImportError(
                f"audio sr={sr} != {sample_rate}; install librosa to resample"
            ) from exc
    return x.astype(np.float32, copy=False)
