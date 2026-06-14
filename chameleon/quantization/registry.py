"""Quantization method registry."""

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
