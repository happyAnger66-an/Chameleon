"""Console 评测展示 — 轻量进度日志，不阻塞推理。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chameleon.evaluate.viewers.base import EvalEventSink, EvalStepEvent

if TYPE_CHECKING:
    from chameleon.evaluate.lerobot_eval import EvalSummary

logger = logging.getLogger(__name__)


class ConsoleViewer(EvalEventSink):
    def __init__(self, *, log_every: int = 10) -> None:
        self._log_every = max(1, int(log_every))
        self._step_count = 0
        self._chunk_count = 0

    def on_run_start(self, meta: dict) -> None:
        logger.info(
            "[eval] start run_id=%s repo=%s backend=%s horizon=%s dim=%s",
            meta.get("run_id"),
            meta.get("repo_id"),
            meta.get("backend"),
            meta.get("action_horizon"),
            meta.get("action_dim"),
        )

    def on_step(self, event: EvalStepEvent) -> None:
        if not event.is_chunk_start:
            return
        self._chunk_count += 1
        mae = event.metrics.get("mae")
        mse = event.metrics.get("mse")
        infer_ms = event.infer_ms
        timing = f" infer_ms={infer_ms:.1f}" if infer_ms is not None else ""
        if self._chunk_count % self._log_every == 0:
            logger.info(
                "[eval] chunk %d idx=%d mae=%.6f mse=%.6f%s",
                self._chunk_count,
                event.global_index,
                float(mae) if mae is not None else float("nan"),
                float(mse) if mse is not None else float("nan"),
                timing,
            )
        self._step_count += 1

    def on_run_done(self, summary: EvalSummary) -> None:  # noqa: F821
        logger.info("[eval] done %s", summary.describe())
