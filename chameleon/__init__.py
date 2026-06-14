"""Chameleon 包入口 — 跨平台端侧 VLA 量化 / 编译 / 推理工具包。

作用：
    按依赖顺序导入各子系统，触发 import-time 插件注册（平台、架构、模型、
    量化方法、图捕获、编译后端、自定义算子、运行时、编排器），使所有
    Registry 在首次使用前已就绪。

架构位置：
    最顶层包 — 整个框架的导入根。被 CLI（cli.py）、高层 API（api.py）、
    工作流（workflows/）间接依赖。导入链：
    core → architectures → frontend → quantization → kernels
    → compile → models → runtime。
"""

from __future__ import annotations

__version__ = "0.1.0"

# Order matters: low-level subsystems first, then those that depend on them.
from chameleon import core  # noqa: F401
from chameleon import architectures  # noqa: F401
from chameleon import frontend  # noqa: F401
from chameleon import quantization  # noqa: F401
from chameleon import kernels  # noqa: F401
from chameleon import compile  # noqa: F401
from chameleon import models  # noqa: F401
from chameleon import runtime  # noqa: F401

from chameleon.config.schema import TaskConfig  # noqa: E402

__all__ = ["TaskConfig", "__version__"]
