"""PyTorch 参考运行时包 — 导出 PyTorchRuntime / PyTorchEngine。

作用：
    import 时注册 "pytorch" 运行时后端。

架构位置：
    运行时层 — runtime/pytorch/ 的聚合入口，generic_cpu 等平台默认 runtime。
"""

from chameleon.runtime.pytorch.backend import PyTorchEngine, PyTorchRuntime  # noqa: F401

__all__ = ["PyTorchEngine", "PyTorchRuntime"]
