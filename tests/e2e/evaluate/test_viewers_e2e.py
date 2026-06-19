"""evaluate viewers 层 E2E — console / webui bridge。"""

from __future__ import annotations

import asyncio
import json

import pytest

from chameleon.config.schema import TaskConfig
from chameleon.evaluate.viewers.base import EvalStepEvent, build_eval_viewer, row_step_metrics
from chameleon.evaluate.viewers.console import ConsoleViewer
from chameleon.evaluate.viewers.webui.broadcaster import WebsocketBroadcaster
from chameleon.evaluate.viewers.webui.bridge import AsyncOutboundBridge, OUTBOUND_STOP
from chameleon.evaluate.viewers.webui.protocol import event_to_json
from chameleon.evaluate.viewers.webui.server import default_client_ws_url, write_webui_server_hint
from tests.helpers.fakes import CaptureEventSink


@pytest.mark.e2e
class TestConsoleViewerE2E:
    def test_console_viewer_accepts_steps(self) -> None:
        sink = CaptureEventSink()
        console = ConsoleViewer(log_every=1)
        meta = {"type": "meta", "run_id": "r", "repo_id": "x", "backend": "fake"}
        console.on_run_start(meta)
        ev = EvalStepEvent(
            run_id="r",
            episode_id=0,
            global_index=0,
            k_in_chunk=0,
            is_chunk_start=True,
            action_horizon=10,
            prompt="p",
            gt_action=[0.0],
            pred_action=[0.1],
            metrics=row_step_metrics([0.0], [0.1]),
            infer_ms=1.0,
        )
        console.on_step(ev)
        assert True  # 不抛错即通过


@pytest.mark.e2e
class TestWebUIViewerE2E:
    def test_build_eval_viewer_console_only(self) -> None:
        task = TaskConfig()
        sink = build_eval_viewer(
            task,
            run_id="r",
            repo_id="repo",
            action_horizon=10,
            action_dim=7,
            num_samples=5,
        )
        assert isinstance(sink, ConsoleViewer)

    def test_default_ws_url_and_hint(self, tmp_path, monkeypatch) -> None:
        task = TaskConfig()
        task.evaluate.webui_port = 9999
        url = default_client_ws_url(task.evaluate)
        assert url == "ws://127.0.0.1:9999/ws"
        import chameleon.evaluate.viewers.webui.server as srv

        monkeypatch.setattr(srv, "_client_dir", lambda: tmp_path)
        written = write_webui_server_hint(task.evaluate)
        assert written == url
        hint = json.loads((tmp_path / "server_hint.json").read_text())
        assert hint["default_ws_url"] == url

    @pytest.mark.asyncio
    async def test_bridge_queue_full_does_not_block(self) -> None:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        bc = WebsocketBroadcaster(history_size=0)
        bridge = AsyncOutboundBridge(loop, q, bc)
        ev = EvalStepEvent(
            run_id="r",
            episode_id=0,
            global_index=0,
            k_in_chunk=0,
            is_chunk_start=True,
            action_horizon=1,
            prompt=None,
            gt_action=[0.0],
            pred_action=[0.0],
            metrics={},
        )
        bridge.sync_emit_step(ev)
        bridge.sync_emit_step(ev)
        await asyncio.sleep(0.05)
        assert bridge.dropped >= 1
        bridge.sync_close()
        await bridge.drain(lambda e: event_to_json({"type": "step", "global_index": e.global_index}))
