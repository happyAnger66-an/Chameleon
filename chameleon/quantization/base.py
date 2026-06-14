"""Quantization abstractions.

The quantization subsystem mirrors the registry pattern used by vLLM/sglang
(``get_quant_method`` per layer) and the metadata-contract idea from
TensorRT-Edge-LLM: a :class:`QuantMethod` produces both a quantized module and a
:class:`QuantMetadata` describing per-component numeric formats. Compiler
backends consume that contract to select kernels / build flags.
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
