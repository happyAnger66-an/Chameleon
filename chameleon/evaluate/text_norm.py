"""ASR 文本规范化 — WER/CER 前处理（纯函数，便于单测）。"""

from __future__ import annotations

import re
import unicodedata


def normalize_asr_text(text: str, *, lang: str | None = None) -> str:
    """Lowercase (latin), strip punctuation, collapse whitespace; CJK keeps chars."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip()
    s = s.lower()
    # Drop most punctuation but keep CJK / alnum / space
    s = re.sub(r"[^\w\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_for_wer(text: str, *, lang: str | None = None) -> list[str]:
    """English-like: whitespace words; CJK/Japanese/Korean: character tokens."""
    s = normalize_asr_text(text, lang=lang)
    if not s:
        return []
    lang_l = (lang or "").lower()
    if any(x in lang_l for x in ("chinese", "japanese", "korean", "cantonese", "zh", "ja", "ko")):
        return [c for c in s if not c.isspace()]
    # Heuristic: if mostly CJK chars, use char-level
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    if cjk >= max(1, len(s.replace(" ", "")) // 2):
        return [c for c in s if not c.isspace()]
    return s.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance on token sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def word_error_rate(ref: str, hyp: str, *, lang: str | None = None) -> float:
    """WER = edit_distance / len(ref_tokens); empty ref → 0 if hyp empty else 1."""
    r = tokenize_for_wer(ref, lang=lang)
    h = tokenize_for_wer(hyp, lang=lang)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / float(len(r))


def char_error_rate(ref: str, hyp: str, *, lang: str | None = None) -> float:
    r = list(normalize_asr_text(ref, lang=lang).replace(" ", ""))
    h = list(normalize_asr_text(hyp, lang=lang).replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / float(len(r))
