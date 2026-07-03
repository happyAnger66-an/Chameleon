"""cosmos3 模型适配包 — 导出 Cosmos3Adapter 与 Cosmos3Config。

作用：
    re-export cosmos3 适配器与配置类型。

架构位置：
    模型/架构层 — cosmos3 具体实现入口，adapter.py 与 reference.py 的聚合导出。
"""

from chameleon.models.cosmos3.adapter import Cosmos3Adapter
from chameleon.models.cosmos3.reference import Cosmos3Config, Cosmos3ReferenceModel

__all__ = ["Cosmos3Adapter", "Cosmos3Config", "Cosmos3ReferenceModel"]
