"""TensorRT 运行时包 — 导出 TensorRTRuntime / TensorRTEngine。

作用：
    import 时注册 "tensorrt" 运行时后端。

架构位置：
    运行时层 — runtime/tensorrt/ 的聚合入口，nvidia_orin / nvidia_thor
    平台默认 runtime，消费 compile 产出的 .engine 文件。
"""

from chameleon.runtime.tensorrt.backend import TensorRTEngine, TensorRTRuntime  # noqa: F401

__all__ = ["TensorRTEngine", "TensorRTRuntime"]
