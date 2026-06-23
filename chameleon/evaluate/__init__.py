"""评估工具包 — 对比推理路径精度，以及真实数据集上的动作评测。

作用：
    re-export 张量级对比、PolicyRunner 抽象/registry、openpi / Chameleon 两种
    策略运行器，以及 LeRobot 离线评测入口。

架构位置：
    工具层 — api.run_eval / CLI ``eval`` 消费；import 时注册 policy_runner。
"""

from chameleon.evaluate.chameleon_runner import ChameleonOrchestratorRunner
from chameleon.evaluate.pt_trt_compare_runner import Pi05PtTrtCompareRunner
from chameleon.evaluate.trt_only_runner import Pi05TrtOnlyRunner
from chameleon.evaluate.compare import ActionDiff, compare_actions
from chameleon.evaluate.lerobot_eval import (
    EvalSampleResult,
    EvalSummary,
    evaluate_lerobot,
)
from chameleon.evaluate.policy import OpenPiPolicyRunner
from chameleon.evaluate.runner_base import (
    PolicyRunner,
    build_policy_runner,
    list_policy_runners,
    register_policy_runner,
)

__all__ = [
    "ActionDiff",
    "compare_actions",
    "PolicyRunner",
    "build_policy_runner",
    "list_policy_runners",
    "register_policy_runner",
    "OpenPiPolicyRunner",
    "ChameleonOrchestratorRunner",
    "Pi05PtTrtCompareRunner",
    "Pi05TrtOnlyRunner",
    "evaluate_lerobot",
    "EvalSummary",
    "EvalSampleResult",
]
