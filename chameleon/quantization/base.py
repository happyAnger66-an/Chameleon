"""量化抽象 — QuantMethod / QuantMetadata 契约与配置。

作用：
    定义 QuantConfig（用户量化请求）、QuantMetadata（各组件数值格式契约，
    如 weight/activation/kv_cache dtype）和 QuantMethod ABC。编译与运行时
    据此 dispatch，避免字符串硬编码。

架构位置：
    优化/编译流水线 — 设计对标 vLLM get_quant_method 与 TensorRT-Edge-LLM
    metadata 契约。被 api.run_quantize 和 compile/base 消费。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chameleon.core.platform import PlatformSpec
from chameleon.quantization.calibrate.base import Calibrator


@dataclass
class QuantConfig:
    """User-facing quantization request."""

    method: str
    """Registered quant method name, e.g. ``fp8`` / ``int8`` / ``int4_awq``."""

    weight_dtype: str = "int8"
    activation_dtype: str | None = None
    kv_cache_dtype: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantMetadata:
    """Contract emitted by quantization, consumed by compile/runtime."""

    method: str
    component_dtypes: dict[str, str] = field(default_factory=dict)
    """e.g. ``{"weight": "int8", "activation": "fp16", "kv_cache": "fp8"}``."""

    scales_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class QuantMethod(ABC):
    """A quantization algorithm (PTQ recipe)."""

    name: str

    @abstractmethod
    def quantize(
        self,
        module: Any,
        calibrator: Calibrator,
        platform: PlatformSpec,
        config: QuantConfig,
    ) -> tuple[Any, QuantMetadata]:
        """Quantize ``module`` in-place / functionally and return ``(module, metadata)``."""

    def supports_platform(self, platform: PlatformSpec, config: QuantConfig) -> bool:
        """Whether this method can target ``platform`` for ``config``."""
        return platform.supports_dtype(config.weight_dtype)
