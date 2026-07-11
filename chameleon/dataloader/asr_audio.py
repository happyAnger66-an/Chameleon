"""ASR 音频数据源 — LibriSpeech / 本地 manifest。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chameleon.dataloader.base import DatasetSpec, register_loader

logger = logging.getLogger(__name__)


@dataclass
class AsrSample:
    audio: np.ndarray | str
    sr: int
    ref_text: str
    language: str | None
    duration_sec: float
    index: int = 0


class AsrAudioDataSource:
    """Loads ASR eval samples (PCM + reference text)."""

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self._built = False
        self._items: list[dict[str, Any]] = []

    def build(self) -> "AsrAudioDataSource":
        if self._built:
            return self
        loader = self.spec.loader
        if loader == "asr_manifest":
            self._load_manifest()
        elif loader == "asr_hf":
            self._load_hf()
        else:
            raise ValueError(f"Unknown ASR loader: {loader}")
        start = int(self.spec.start_index or 0)
        n = self.spec.num_samples
        self._items = self._items[start:]
        if n is not None:
            self._items = self._items[: int(n)]
        self._built = True
        logger.info("AsrAudioDataSource built: n=%d loader=%s", len(self._items), loader)
        return self

    def _load_manifest(self) -> None:
        root = Path(self.spec.dataset_root or ".").expanduser()
        manifest = self.spec.extra.get("manifest") or "manifest.jsonl"
        path = root / manifest if not Path(manifest).is_file() else Path(manifest)
        if not path.is_file():
            raise FileNotFoundError(f"ASR manifest not found: {path}")
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            wav = row.get("audio") or row.get("wav") or row.get("path")
            text = row.get("text") or row.get("ref") or row.get("transcript") or ""
            lang = row.get("language")
            items.append({"audio": str(Path(wav).expanduser()), "text": text, "language": lang})
        self._items = items

    def _load_hf(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError("asr_hf loader needs `datasets`") from exc
        name = self.spec.extra.get("hf_name") or "librispeech_asr"
        split = self.spec.extra.get("split") or "test.clean"
        config = self.spec.extra.get("config") or "clean"
        ds = load_dataset(name, config, split=split)
        items = []
        for i, row in enumerate(ds):
            audio = row.get("audio") or {}
            arr = np.asarray(audio.get("array"), dtype=np.float32)
            sr = int(audio.get("sampling_rate") or 16000)
            text = str(row.get("text") or row.get("transcription") or "")
            items.append({"audio": arr, "sr": sr, "text": text, "language": "English"})
            if self.spec.num_samples and len(items) >= int(self.spec.num_samples) + int(
                self.spec.start_index or 0
            ):
                break
        self._items = items

    def __len__(self) -> int:
        if not self._built:
            self.build()
        return len(self._items)

    def __getitem__(self, index: int) -> AsrSample:
        if not self._built:
            self.build()
        row = self._items[index]
        audio = row["audio"]
        sr = int(row.get("sr") or 16000)
        if isinstance(audio, str):
            # Keep path; runner accepts path. Duration unknown until load.
            dur = float(row.get("duration_sec") or 0.0)
            pcm: np.ndarray | str = audio
        else:
            pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
            if sr != 16000:
                try:
                    import librosa

                    pcm = librosa.resample(pcm, orig_sr=sr, target_sr=16000)
                    sr = 16000
                except ImportError:
                    pass
            dur = float(len(pcm)) / float(sr)
        return AsrSample(
            audio=pcm,
            sr=sr,
            ref_text=str(row.get("text") or ""),
            language=row.get("language"),
            duration_sec=dur,
            index=index,
        )


register_loader("asr_hf", AsrAudioDataSource, override=True)
register_loader("asr_manifest", AsrAudioDataSource, override=True)
