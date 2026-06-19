"""性能 profiling 工具包 — 延迟与计算/访存统计。

作用：
    re-export profile_infer / stats_infer 等接口。

架构位置：
    工具层（入口/编排层旁路）— 被 cli profile / stats 子命令调用。
"""

from chameleon.profile.counters import StatsResult
from chameleon.profile.compute_stats import stats_infer
from chameleon.profile.latency import LatencyResult, profile_infer

__all__ = ["LatencyResult", "StatsResult", "profile_infer", "stats_infer"]
