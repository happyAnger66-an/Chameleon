"""可插拔编译后端包 — 导出 CompilerBackend 抽象与注册表。

作用：
    re-export 编译接口，import 时注册 TensorRT 编译器与非 NVIDIA stub。

架构位置：
    优化/编译流水线 — 核心可插拔扩展点，平台差异（TRT / OpenVINO /
    TVM / 地平线 BPU）收敛于此层。
"""

from chameleon.compile.base import (
    COMPILER_REGISTRY,
    CompilerBackend,
    get_compiler,
    list_compilers,
    register_compiler,
)

# Import-time registration of built-in backends.
from chameleon.compile import tensorrt  # noqa: F401,E402
from chameleon.compile import stubs  # noqa: F401,E402

__all__ = [
    "COMPILER_REGISTRY",
    "CompilerBackend",
    "get_compiler",
    "list_compilers",
    "register_compiler",
]
