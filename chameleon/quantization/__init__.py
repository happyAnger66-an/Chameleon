"""Quantization subsystem."""

from chameleon.quantization.base import QuantConfig, QuantMetadata, QuantMethod
from chameleon.quantization.calibrate import (
    Calibrator,
    TensorCalibrator,
    get_calibrator_factory,
    register_calibrator,
)
from chameleon.quantization.registry import (
    QUANT_METHOD_REGISTRY,
    get_quant_method,
    list_quant_methods,
    register_quant_method,
)

# Import-time registration of built-in methods.
from chameleon.quantization import methods  # noqa: F401,E402

__all__ = [
    "QuantConfig",
    "QuantMetadata",
    "QuantMethod",
    "Calibrator",
    "TensorCalibrator",
    "get_calibrator_factory",
    "register_calibrator",
    "QUANT_METHOD_REGISTRY",
    "get_quant_method",
    "list_quant_methods",
    "register_quant_method",
]
