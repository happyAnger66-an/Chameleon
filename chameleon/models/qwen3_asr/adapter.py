"""qwen3_asr 模型适配器 — host 侧 prompt / 输出解析（不加载整网权重）。

Edge-LLM 负责 audio encoder + LLM engine；本适配器只提供：
- checkpoint / tokenizer 路径解析
- chat template prompt 构建（可选 language 强制）
- ``parse_asr_output``
- ``example_observation``（infer 冒烟用音频路径）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chameleon.models.base import ModelAdapter, register_model

logger = logging.getLogger(__name__)

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


@dataclass
class Qwen3AsrConfig:
    checkpoint: str = ""
    dtype: str = "fp16"
    max_new_tokens: int = 256
    sample_rate: int = 16000
    # Dummy action fields so ModelAdapter property accessors don't crash if called.
    action_dim: int = 0
    action_horizon: int = 0
    num_denoise_steps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def parse_asr_output(raw: str, user_language: str | None = None) -> tuple[str, str]:
    """Parse Qwen3-ASR raw decode into ``(language, text)``.

    Mirrors ``qwen_asr.inference.utils.parse_asr_output`` (without repetition fix).
    """
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    if user_language:
        return user_language, s

    if _ASR_TEXT_TAG not in s:
        return "", s

    meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    meta_lower = meta_part.lower()
    if "language none" in meta_lower:
        t = text_part.strip()
        return ("", t) if t else ("", "")

    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX) :].strip()
            if val:
                lang = val[:1].upper() + val[1:] if val else ""
            break
    return lang, text_part.strip()


def build_force_language_suffix(language: str | None) -> str:
    if not language:
        return ""
    return f"language {language}{_ASR_TEXT_TAG}"


class Qwen3AsrAdapter(ModelAdapter):
    """Lightweight host adapter for Edge-LLM ASR deploy/eval/infer."""

    architecture = "qwen3_asr"

    def __init__(self, config: Qwen3AsrConfig) -> None:
        super().__init__(config)
        self._checkpoint: Path | None = None
        self._tokenizer: Any = None
        self._device = "cpu"

    @classmethod
    def make_config(cls, overrides: dict[str, Any] | None = None) -> Qwen3AsrConfig:
        raw = dict(overrides or {})
        known = {f.name for f in Qwen3AsrConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        cfg = Qwen3AsrConfig(**kwargs)
        cfg.extra = extra
        return cfg

    def build(self, device: str = "cpu") -> "Qwen3AsrAdapter":
        self._device = device
        ckpt = str(self.config.checkpoint or "").strip()
        if not ckpt:
            raise ValueError("qwen3_asr requires model_overrides.checkpoint (HF model dir).")
        path = Path(ckpt).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Qwen3-ASR checkpoint dir not found: {path}")
        self._checkpoint = path
        logger.info("Qwen3AsrAdapter: checkpoint=%s device=%s", path, device)
        return self

    @property
    def checkpoint_dir(self) -> Path:
        if self._checkpoint is None:
            raise RuntimeError("Call build() first.")
        return self._checkpoint

    def stage_module(self, stage: str):
        raise NotImplementedError(
            "qwen3_asr stages are Edge-LLM engines; use deploy.backend=qwen3_asr "
            f"(requested stage={stage!r})."
        )

    def example_observation(self, batch_size: int = 1, device: str = "cpu") -> dict[str, Any]:
        audio = self.config.extra.get("audio") or ""
        return {
            "audio": audio,
            "context": "",
            "language": None,
            "batch_size": batch_size,
        }

    def _lazy_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("qwen3_asr prompt build needs transformers") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_dir), trust_remote_code=True
        )
        return self._tokenizer

    def build_prompt(self, *, context: str = "", language: str | None = None) -> str:
        """Build chat-template prompt with optional language force suffix.

        Audio is represented as a placeholder content item; Edge-LLM runtime
        replaces / binds the actual wav path via multimodal messages.
        """
        tok = self._lazy_tokenizer()
        messages = [
            {"role": "system", "content": context or ""},
            {
                "role": "user",
                "content": [{"type": "audio", "audio": ""}],
            },
        ]
        try:
            base = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:  # noqa: BLE001
            # Fallback when chat template rejects audio content type.
            base = (
                f"<|im_start|>system\n{context or ''}<|im_end|>\n"
                f"<|im_start|>user\n<|audio_pad|><|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        if not isinstance(base, str):
            base = str(base)
        return base + build_force_language_suffix(language)

    @property
    def orchestrator_key(self) -> str | None:
        return "qwen3_asr"


register_model("qwen3_asr", Qwen3AsrAdapter, override=True)
register_model("qwen3_asr_0.6b", Qwen3AsrAdapter, override=True)
register_model("qwen3_asr_1.7b", Qwen3AsrAdapter, override=True)
