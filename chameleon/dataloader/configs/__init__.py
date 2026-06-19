"""数据集规格注册 — import 时触发各 benchmark 的注册副作用。

作用：
    汇总 dataloader/configs/ 下的数据集注册模块；被 dataloader/__init__.py
    import 以填充 DATASET_REGISTRY。

架构位置：
    数据层（配置）— 新增数据集只需在本包加文件并在此 import。
"""

from __future__ import annotations

from chameleon.dataloader.configs import pi05_libero  # noqa: F401

__all__ = ["pi05_libero"]
