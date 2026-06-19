"""推理线程 → asyncio 泵：有界队列 + 满则丢弃，保证不阻塞 infer。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chameleon.evaluate.viewers.base import EvalStepEvent

if TYPE_CHECKING:
    from chameleon.evaluate.viewers.webui.broadcaster import WebsocketBroadcaster

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutboundMessage:
    text: str
    add_history: bool = True


class OutboundStop:
    """队列结束标记。"""


OUTBOUND_STOP = OutboundStop()


class AsyncOutboundBridge:
    """线程侧 ``sync_emit_*`` / 事件循环侧 ``drain``。"""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[Any],
        broadcaster: WebsocketBroadcaster,
    ) -> None:
        self._loop = loop
        self._queue = queue
        self._broadcaster = broadcaster
        self.dropped = 0

    def _put_nowait(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped += 1

    def sync_emit_step(self, event: EvalStepEvent) -> None:
        self._loop.call_soon_threadsafe(self._put_nowait, event)

    def sync_emit_text(self, text: str, *, add_history: bool = True) -> None:
        self._loop.call_soon_threadsafe(
            self._put_nowait,
            OutboundMessage(text=text, add_history=add_history),
        )

    def sync_close(self) -> None:
        self._loop.call_soon_threadsafe(self._put_nowait, OUTBOUND_STOP)

    async def drain(self, encode_step) -> None:
        """``encode_step(EvalStepEvent) -> str`` 在消费侧做 JPEG 与 JSON。"""
        while True:
            item = await self._queue.get()
            if item is OUTBOUND_STOP:
                break
            if isinstance(item, EvalStepEvent):
                text = await asyncio.to_thread(encode_step, item)
                if text:
                    self._broadcaster.add_history(text)
                    await self._broadcaster.broadcast(text)
            elif isinstance(item, OutboundMessage):
                if item.add_history:
                    self._broadcaster.add_history(item.text)
                await self._broadcaster.broadcast(item.text)
        if self.dropped:
            logger.warning("[webui] 出站队列满，丢弃 %d 条 UI 事件（推理未阻塞）", self.dropped)
