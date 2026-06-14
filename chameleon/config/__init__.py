"""统一配置 schema 包 — 导出 TaskConfig 及相关模型。

作用：
    re-export TaskConfig、QuantizeStep、CompileStep、InferConfig。

架构位置：
    入口/编排层 — 被 CLI、api、workflows 加载，替代 model_optimizer 原先
    混杂的 .py / JSON / argparse 配置。
"""

from chameleon.config.schema import (
    CompileStep,
    InferConfig,
    QuantizeStep,
    TaskConfig,
)

__all__ = ["TaskConfig", "QuantizeStep", "CompileStep", "InferConfig"]
