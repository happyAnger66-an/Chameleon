"""评测事件模型与 EventSink 抽象。

推理循环只调用 ``on_step`` 等非阻塞接口；WebUI 侧在独立 asyncio 泵中编码
JPEG 并广播，避免拖慢 ``policy_runner.infer``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from chameleon.config.schema import TaskConfig

if TYPE_CHECKING:
    from chameleon.evaluate.lerobot_eval import EvalSummary


@dataclass
class EvalStepEvent:
    """单步评测事件（与 model_optimizer WebUI 协议 v1 对齐）。"""

    run_id: str
    episode_id: int
    global_index: int
    k_in_chunk: int
    is_chunk_start: bool
    action_horizon: int
    prompt: str | None
    gt_action: list[float]
    pred_action: list[float]
    metrics: dict[str, Any]
    pred_action_trt: list[float] | None = None
    """compare_mode 下 TensorRT 路单步动作。"""
    observation: dict[str, Any] | None = None
    """仅 ``k_in_chunk==0`` 且需要图像时携带；JPEG 编码在消费侧完成。"""
    infer_ms: float | None = None
    """仅 chunk 首步（``k_in_chunk==0``）携带单次 policy.infer 耗时。"""


class EvalEventSink(ABC):
    """评测事件接收端；``on_step`` 必须快速返回，不得阻塞推理。"""

    @abstractmethod
    def on_run_start(self, meta: dict[str, Any]) -> None: ...

    @abstractmethod
    def on_step(self, event: EvalStepEvent) -> None: ...

    @abstractmethod
    def on_run_done(self, summary: EvalSummary) -> None: ...


class NullEventSink(EvalEventSink):
    def on_run_start(self, meta: dict[str, Any]) -> None:
        return

    def on_step(self, event: EvalStepEvent) -> None:
        return

    def on_run_done(self, summary: EvalSummary) -> None:
        return


class CompositeEventSink(EvalEventSink):
    def __init__(self, sinks: list[EvalEventSink]) -> None:
        self._sinks = sinks

    def on_run_start(self, meta: dict[str, Any]) -> None:
        for s in self._sinks:
            s.on_run_start(meta)

    def on_step(self, event: EvalStepEvent) -> None:
        for s in self._sinks:
            s.on_step(event)

    def on_run_done(self, summary: EvalSummary) -> None:
        for s in self._sinks:
            s.on_run_done(summary)


def build_eval_viewer(
    task: TaskConfig,
    *,
    run_id: str,
    repo_id: str,
    action_horizon: int,
    action_dim: int,
    num_samples: int,
    start_index: int = 0,
) -> EvalEventSink:
    """按 ``task.evaluate.viewer`` 构建展示 sink（webui 模式需配合 ``run_eval_webui``）。"""
    from chameleon.evaluate.viewers.console import ConsoleViewer

    ev = task.evaluate
    mode = (ev.viewer or "console").strip().lower()
    sinks: list[EvalEventSink] = []

    if mode in ("console", "both"):
        sinks.append(ConsoleViewer(log_every=10))

    if mode in ("webui", "both"):
        from chameleon.evaluate.viewers.webui.viewer import WebUIEventSink
        from chameleon.evaluate.viewers.webui.server import get_active_webui_bridge

        bridge = get_active_webui_bridge()
        if bridge is None:
            raise RuntimeError(
                "viewer=webui 需先启动 WebUI 服务（run_eval 在 webui 模式下会自动启动）。"
            )
        sinks.append(
            WebUIEventSink(
                bridge=bridge,
                run_id=run_id,
                jpeg_quality=ev.webui_jpeg_quality,
                send_wrist=ev.webui_show_wrist,
            )
        )

    if not sinks:
        return NullEventSink()
    if len(sinks) == 1:
        return sinks[0]
    return CompositeEventSink(sinks)
