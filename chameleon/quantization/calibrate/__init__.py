"""校准数据抽象包 — 导出 Calibrator 接口与注册表。

作用：
    re-export Calibrator / TensorCalibrator 及 CALIBRATOR_REGISTRY。

架构位置：
    优化/编译流水线 — quantization 子模块，为 PTQ 算法提供校准批次。
"""

from chameleon.quantization.calibrate.base import (
    CALIBRATOR_REGISTRY,
    Calibrator,
    TensorCalibrator,
    get_calibrator_factory,
    register_calibrator,
)

__all__ = [
    "CALIBRATOR_REGISTRY",
    "Calibrator",
    "TensorCalibrator",
    "get_calibrator_factory",
    "register_calibrator",
]
