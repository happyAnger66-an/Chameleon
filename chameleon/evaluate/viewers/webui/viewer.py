"""WebUI EventSink — 推理线程仅非阻塞入队。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chameleon.evaluate.viewers.base import EvalEventSink, EvalStepEvent
from chameleon.evaluate.viewers.webui.bridge import AsyncOutboundBridge
from chameleon.evaluate.viewers.webui.protocol import event_to_json

if TYPE_CHECKING:
    from chameleon.evaluate.lerobot_eval import EvalSummary


class WebUIEventSink(EvalEventSink):
    def __init__(
        self,
        *,
        bridge: AsyncOutboundBridge,
        run_id: str,
        jpeg_quality: int,
        send_wrist: bool,
    ) -> None:
        self._bridge = bridge
        self._run_id = run_id
        self._jpeg_quality = int(jpeg_quality)
        self._send_wrist = bool(send_wrist)
        self._meta_sent = False

    def on_run_start(self, meta: dict) -> None:
        if self._meta_sent:
            return
        self._meta_sent = True
        self._bridge.sync_emit_text(event_to_json(meta))

    def on_step(self, event: EvalStepEvent) -> None:
        self._bridge.sync_emit_step(event)

    def on_run_done(self, summary: EvalSummary) -> None:  # noqa: F821
        payload = {
            "type": "done",
            "run_id": self._run_id,
            "summary": {
                "num_samples": summary.num_samples,
                "mean_max_abs": summary.mean_max_abs,
                "mean_mean_abs": summary.mean_mean_abs,
                "mean_cosine": summary.mean_cosine,
                "worst_max_abs": summary.worst_max_abs,
                "worst_index": summary.worst_index,
            },
        }
        self._bridge.sync_emit_text(event_to_json(payload), add_history=False)
