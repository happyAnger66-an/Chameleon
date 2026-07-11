"""Lightweight WebSocket broadcaster for ASR stream fixed/pending text UI."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class AsrStreamWebUI:
    """Background asyncio WS server; sync ``emit`` from the inference thread."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8768, path: str = "/ws") -> None:
        self.host = host
        self.port = int(port)
        self.path = path if path.startswith("/") else f"/{path}"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._clients: set[Any] = set()
        self._queue: asyncio.Queue[str] | None = None
        self._ready = threading.Event()

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}{self.path}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="asr-stream-webui", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("ASR stream WebUI failed to start")
        logger.info("ASR stream WebUI: %s (open asr_stream.html?ws=%s)", self.ws_url, self.ws_url)

    def _run(self) -> None:
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ASR stream WebUI needs websockets") from exc

        async def handler(ws: Any) -> None:
            self._clients.add(ws)
            try:
                async for _ in ws:
                    pass
            finally:
                self._clients.discard(ws)

        async def pump() -> None:
            assert self._queue is not None
            while True:
                msg = await self._queue.get()
                dead = []
                for ws in list(self._clients):
                    try:
                        await ws.send(msg)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self._clients.discard(ws)

        async def main() -> None:
            self._queue = asyncio.Queue()
            async with websockets.serve(handler, self.host, self.port):
                self._ready.set()
                await pump()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(main())

    def emit(self, event: dict[str, Any]) -> None:
        if self._loop is None or self._queue is None:
            return
        text = json.dumps(event, ensure_ascii=False)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, text)

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
