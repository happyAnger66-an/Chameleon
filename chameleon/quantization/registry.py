"""量化方法注册表 — QuantMethod 插件的发现与查询。

作用：
    维护 QUANT_METHOD_REGISTRY，提供 register_quant_method /
    get_quant_method / list_quant_methods。

架构位置：
    优化/编译流水线 — 被 api.run_quantize 和 cli info 子命令查询。
"""

from __future__ import annotations

from chameleon.core.registry import Registry
from chameleon.quantization.base import QuantMethod

QUANT_METHOD_REGISTRY: Registry[str, QuantMethod] = Registry("quant_method")


def register_quant_method(method: QuantMethod, *, override: bool = False) -> QuantMethod:
    return QUANT_METHOD_REGISTRY.register(method.name, method, override=override)


def get_quant_method(name: str) -> QuantMethod:
    return QUANT_METHOD_REGISTRY.get(name)


def list_quant_methods() -> list[str]:
    return QUANT_METHOD_REGISTRY.keys()
