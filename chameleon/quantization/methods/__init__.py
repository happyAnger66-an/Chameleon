"""内置量化方法包 — import 时注册所有 PTQ 方法。

作用：
    加载 modelopt_ptq 模块，触发 int8/fp8/int4_awq 等方法注册。

架构位置：
    优化/编译流水线 — quantization/methods/ 的聚合入口。
"""

from chameleon.quantization.methods import modelopt_ptq  # noqa: F401

__all__ = ["modelopt_ptq"]
