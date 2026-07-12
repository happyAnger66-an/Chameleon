"""性能 profiling 工具包 — 延迟、stage bench 与计算/访存统计。

作用：
    re-export profile_infer / stats_infer / run_bench 等接口。

架构位置：
    工具层（入口/编排层旁路）— 被 cli profile / stats / bench 子命令调用。
"""

from chameleon.profile.bench import BenchReport, run_bench
from chameleon.profile.counters import StatsResult
from chameleon.profile.compute_stats import stats_infer
from chameleon.profile.latency import LatencyResult, profile_infer
from chameleon.profile.stage_timer import StageStats, StageTimer, format_comparison_table

__all__ = [
    "BenchReport",
    "LatencyResult",
    "StageStats",
    "StageTimer",
    "StatsResult",
    "format_comparison_table",
    "profile_infer",
    "run_bench",
    "stats_infer",
]
