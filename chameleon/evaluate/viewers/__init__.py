"""评测结果展示 — console / WebUI，推理线程仅非阻塞投递事件。"""

from chameleon.evaluate.viewers.base import (
    EvalEventSink,
    EvalStepEvent,
    NullEventSink,
    build_eval_viewer,
)

__all__ = [
    "EvalEventSink",
    "EvalStepEvent",
    "NullEventSink",
    "build_eval_viewer",
    "ConsoleViewer",
]


def __getattr__(name: str):
    if name == "ConsoleViewer":
        from chameleon.evaluate.viewers.console import ConsoleViewer

        return ConsoleViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
