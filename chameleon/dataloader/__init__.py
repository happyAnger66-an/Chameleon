"""dataloader 包 — 真实数据集加载的统一入口。

作用：
    对外导出统一样本契约（ChameleonSample）、数据集规格与注册表
    （DatasetSpec / register_dataset / get_dataset_spec / list_datasets）、
    LeRobot 数据源与构建工厂（LeRobotDataSource / build_dataset /
    build_dataset_from_config）。import 本包时触发 configs/ 下的数据集注册。

架构位置：
    数据层 — 上游：evaluate / CLI；下游：openpi Policy 或 Pi05RealOrchestrator。
    借鉴 starVLA 的「DataConfig + Registry + 统一 sample」分层，复用 openpi 的
    repack / norm 作为单一事实来源。
"""

from __future__ import annotations

from chameleon.dataloader.base import (
    DATASET_REGISTRY,
    ChameleonSample,
    DatasetSpec,
    get_dataset_spec,
    list_datasets,
    register_dataset,
)
from chameleon.dataloader.lerobot import (
    LeRobotDataSource,
    build_dataset,
    build_dataset_from_config,
)

# 触发数据集注册（import 副作用）。
from chameleon.dataloader import configs as _configs  # noqa: F401,E402

__all__ = [
    "ChameleonSample",
    "DatasetSpec",
    "DATASET_REGISTRY",
    "register_dataset",
    "get_dataset_spec",
    "list_datasets",
    "LeRobotDataSource",
    "build_dataset",
    "build_dataset_from_config",
]
