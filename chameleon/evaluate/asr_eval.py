"""ASR 离线评测循环 — WER/CER/RTF。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from chameleon.evaluate.asr_runner_base import AsrRunner
from chameleon.evaluate.text_norm import char_error_rate, word_error_rate

logger = logging.getLogger(__name__)


@dataclass
class AsrSampleResult:
    index: int
    ref_text: str
    hyp_text: str
    language: str
    wer: float
    cer: float
    rtf: float
    duration_sec: float


@dataclass
class AsrEvalSummary:
    num_samples: int
    mean_wer: float
    mean_cer: float
    mean_rtf: float
    samples: list[AsrSampleResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"ASR eval: n={self.num_samples} "
            f"mean_wer={self.mean_wer:.4f} mean_cer={self.mean_cer:.4f} "
            f"mean_rtf={self.mean_rtf:.3f}"
        )


def evaluate_asr(
    data_source: Any,
    asr_runner: AsrRunner,
    *,
    num_samples: int = 50,
    stride: int = 1,
    log_every: int = 10,
) -> AsrEvalSummary:
    asr_runner.build()
    n = min(int(num_samples), len(data_source))
    results: list[AsrSampleResult] = []
    wer_num = wer_den = 0.0
    cer_acc = 0.0
    rtf_acc = 0.0

    for k, i in enumerate(range(0, n * stride, stride)):
        if k >= n:
            break
        sample = data_source[i]
        audio = sample.audio
        ref = str(sample.ref_text or "")
        lang = getattr(sample, "language", None)
        dur = float(getattr(sample, "duration_sec", 0.0) or 0.0)

        t0 = time.perf_counter()
        if hasattr(audio, "shape"):
            out = asr_runner.transcribe(audio, language=lang, sample_rate=getattr(sample, "sr", 16000))
        else:
            out = asr_runner.transcribe(str(audio), language=lang)
        elapsed = time.perf_counter() - t0
        if dur <= 0 and hasattr(audio, "shape"):
            sr = int(getattr(sample, "sr", 16000) or 16000)
            dur = float(len(audio)) / float(sr)
        rtf = (elapsed / dur) if dur > 0 else 0.0

        wer = word_error_rate(ref, out.text, lang=lang or out.language)
        cer = char_error_rate(ref, out.text, lang=lang or out.language)
        # Weighted WER by ref token count approx via 1/wer inverse — use equal weight mean for MVP
        wer_num += wer
        wer_den += 1.0
        cer_acc += cer
        rtf_acc += rtf
        results.append(
            AsrSampleResult(
                index=int(getattr(sample, "index", i)),
                ref_text=ref,
                hyp_text=out.text,
                language=out.language,
                wer=wer,
                cer=cer,
                rtf=rtf,
                duration_sec=dur,
            )
        )
        if (k + 1) % max(1, log_every) == 0:
            logger.info(
                "[asr-eval] %d/%d wer=%.3f hyp=%r",
                k + 1,
                n,
                wer,
                out.text[:80],
            )

    m = max(1.0, wer_den)
    summary = AsrEvalSummary(
        num_samples=len(results),
        mean_wer=wer_num / m,
        mean_cer=cer_acc / m,
        mean_rtf=rtf_acc / m,
        samples=results,
    )
    logger.info(
        "[asr-eval] done n=%d mean_wer=%.4f mean_cer=%.4f mean_rtf=%.3f",
        summary.num_samples,
        summary.mean_wer,
        summary.mean_cer,
        summary.mean_rtf,
    )
    return summary
