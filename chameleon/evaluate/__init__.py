"""评估工具包 — 对比推理路径精度，以及真实数据集上的动作/ASR 评测。

作用：
    re-export 张量级对比、PolicyRunner / AsrRunner 抽象/registry、
    openpi / Chameleon 策略运行器，以及 LeRobot / ASR 离线评测入口。

架构位置：
    工具层 — api.run_eval / CLI ``eval`` 消费；import 时注册 policy_runner / asr_runner。
"""

from chameleon.evaluate.chameleon_runner import ChameleonOrchestratorRunner
from chameleon.evaluate.cosmos3_runner import Cosmos3PolicyRunner
from chameleon.evaluate.cosmos3_pt_trt_compare_runner import Cosmos3PtTrtCompareRunner
from chameleon.evaluate.cosmos3_trt_runner import Cosmos3TrtPolicyRunner
from chameleon.evaluate.pt_trt_compare_runner import Pi05PtTrtCompareRunner
from chameleon.evaluate.pt_tvm_compare_runner import Pi05PtTvmCompareRunner
from chameleon.evaluate.trt_only_runner import Pi05TrtOnlyRunner
from chameleon.evaluate.tvm_only_runner import Pi05TvmOnlyRunner
from chameleon.evaluate.compare import ActionDiff, compare_actions
from chameleon.evaluate.lerobot_eval import (
    EvalSampleResult,
    EvalSummary,
    evaluate_lerobot,
)
from chameleon.evaluate.asr_eval import AsrEvalSummary, AsrSampleResult, evaluate_asr
from chameleon.evaluate.asr_runner_base import (
    ASR_RUNNER_REGISTRY,
    AsrResult,
    AsrRunner,
    build_asr_runner,
    is_asr_runner_name,
    list_asr_runners,
    register_asr_runner,
)
from chameleon.evaluate.qwen3_asr_edgellm_runner import Qwen3AsrEdgellmRunner
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
    "Cosmos3PolicyRunner",
    "Cosmos3PtTrtCompareRunner",
    "Cosmos3TrtPolicyRunner",
    "Pi05PtTrtCompareRunner",
    "Pi05PtTvmCompareRunner",
    "Pi05TrtOnlyRunner",
    "Pi05TvmOnlyRunner",
    "evaluate_lerobot",
    "EvalSummary",
    "EvalSampleResult",
    "AsrRunner",
    "AsrResult",
    "ASR_RUNNER_REGISTRY",
    "build_asr_runner",
    "register_asr_runner",
    "list_asr_runners",
    "is_asr_runner_name",
    "Qwen3AsrEdgellmRunner",
    "evaluate_asr",
    "AsrEvalSummary",
    "AsrSampleResult",
]
