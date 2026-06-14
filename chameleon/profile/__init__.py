"""性能 profiling 工具包 — 导出延迟测量接口。

作用：
    re-export profile_infer / LatencyResult。

架构位置：
    工具层（入口/编排层旁路）— 被 cli profile 子命令调用，不参与主流水线。
"""

from chameleon.profile.latency import LatencyResult, profile_infer

__all__ = ["LatencyResult", "profile_infer"]
