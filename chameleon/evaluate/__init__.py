"""评估工具包 — 对比不同推理路径的精度差异。

作用：
    re-export compare_actions / ActionDiff。

架构位置：
    工具层（入口/编排层旁路）— 用于 TRT vs PyTorch 等路径的回归校验，
    不参与主推理流水线。
"""

from chameleon.evaluate.compare import ActionDiff, compare_actions

__all__ = ["ActionDiff", "compare_actions"]
