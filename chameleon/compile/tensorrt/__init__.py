"""TensorRT 编译后端包 — 导出 TensorRTCompiler。

作用：
    import 时注册 NVIDIA TensorRT 编译实现。

架构位置：
    优化/编译流水线 — compile/tensorrt/ 的聚合入口，对应 nvidia_orin /
    nvidia_thor 平台的默认 compiler。
"""

from chameleon.compile.tensorrt.backend import TensorRTCompiler  # noqa: F401

__all__ = ["TensorRTCompiler"]
