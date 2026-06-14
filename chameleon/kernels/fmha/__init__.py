"""FMHA 自定义算子包 — import 时注册 fmha_d256 算子。

作用：
    加载 fmha/fmha_d256 模块，触发算子三段式注册。

架构位置：
    算子层 — kernels/fmha/ 的聚合入口，服务 pi05 PaliGemma head_dim=256
    的 attention 热点。
"""

from chameleon.kernels.fmha import fmha_d256  # noqa: F401

__all__ = ["fmha_d256"]
