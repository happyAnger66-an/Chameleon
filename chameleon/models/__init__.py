"""模型适配层包 — 导出 ModelAdapter 抽象与注册表。

作用：
    re-export ModelAdapter 接口，import 时加载 pi05 等内置适配器。

架构位置：
    模型/架构层 — 连接外部模型实现（openpi PI0Pytorch）与 Chameleon
    stage 接口，为 frontend（追踪导出）、quantization（按 stage 量化）、
    runtime（按 stage 推理）提供 nn.Module。
"""

from chameleon.models.base import (
    MODEL_REGISTRY,
    ModelAdapter,
    get_model_adapter,
    list_models,
    register_model,
)

# Import-time registration of built-in model adapters.
from . import pi05  # noqa: F401
from . import cosmos3  # noqa: F401
from . import qwen3_asr  # noqa: F401

__all__ = [
    "MODEL_REGISTRY",
    "ModelAdapter",
    "get_model_adapter",
    "list_models",
    "register_model",
]
