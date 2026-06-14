"""量化子系统包 — 导出量化抽象、校准器与注册表。

作用：
    re-export QuantMethod / QuantConfig / QuantMetadata 等类型，
    import 时加载 modelopt_ptq 等内置量化方法。

架构位置：
    优化/编译流水线 — 位于 models 与 compile 之间，产出 QuantMetadata
    契约供编译后端选择 kernel / build flag。
"""

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
