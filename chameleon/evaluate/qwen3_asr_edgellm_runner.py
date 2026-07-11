"""Qwen3-ASR Edge-LLM runner — wav → EdgeLLMAsrEngine → AsrResult。"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from chameleon.config.schema import TaskConfig
from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.core.platform import get_platform
from chameleon.deploy.paths import resolve_engine_dir
from chameleon.evaluate.asr_runner_base import AsrResult, AsrRunner, register_asr_runner
from chameleon.runtime.edgellm.backend import EdgeLLMRuntimeBackend

logger = logging.getLogger(__name__)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    try:
        import soundfile as sf

        sf.write(str(path), x, int(sr))
        return
    except ImportError:
        pass
    import wave

    pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())


class Qwen3AsrEdgellmRunner(AsrRunner):
    def __init__(self, task: TaskConfig) -> None:
        self._task = task
        self._engine = None
        self._built = False
        self._tmp_dir: tempfile.TemporaryDirectory | None = None

    @classmethod
    def from_task(cls, task: TaskConfig) -> "Qwen3AsrEdgellmRunner":
        return cls(task)

    def build(self) -> "Qwen3AsrEdgellmRunner":
        if self._built:
            return self
        engine_dir = resolve_engine_dir(self._task)
        platform = get_platform(self._task.platform)
        home = getattr(self._task.deploy, "edgellm_home", None)
        ctx = RunContext(
            platform=platform,
            architecture=self._task.architecture,
            options={
                "edgellm_home": home,
                "max_new_tokens": int(self._task.asr.max_new_tokens),
            },
        )
        art = Artifact(
            kind="engine",
            stage="asr",
            platform=platform.name,
            path=str(engine_dir),
        )
        self._engine = EdgeLLMRuntimeBackend().load(art, ctx)
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="chameleon_asr_pcm_")
        self._built = True
        logger.info("Qwen3AsrEdgellmRunner: engines=%s", engine_dir)
        return self

    def transcribe(
        self,
        audio: str | np.ndarray,
        *,
        context: str = "",
        language: str | None = None,
        sample_rate: int | None = None,
    ) -> AsrResult:
        if not self._built:
            self.build()
        assert self._engine is not None

        if isinstance(audio, np.ndarray):
            assert self._tmp_dir is not None
            sr = int(sample_rate or 16000)
            wav_path = Path(self._tmp_dir.name) / "utt.wav"
            _write_wav(wav_path, audio, sr)
            audio_path = str(wav_path)
        else:
            audio_path = str(Path(audio).expanduser().resolve())

        lang = language if language is not None else self._task.asr.language
        ctx = context if context else self._task.asr.context
        out = self._engine.run(
            {
                "audio": audio_path,
                "context": ctx,
                "language": lang,
                "max_new_tokens": int(self._task.asr.max_new_tokens),
            }
        )
        return AsrResult(
            language=str(out.get("language") or ""),
            text=str(out.get("text") or ""),
            raw_text=str(out.get("raw_text") or ""),
            metrics=dict(out.get("metrics") or {}),
        )


register_asr_runner("qwen3_asr_edgellm", Qwen3AsrEdgellmRunner, override=True)
register_asr_runner("qwen3_asr", Qwen3AsrEdgellmRunner, override=True)
