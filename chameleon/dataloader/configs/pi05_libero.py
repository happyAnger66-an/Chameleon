"""pi05 + LIBERO 数据集规格 — 注册到 DATASET_REGISTRY。

作用：
    声明 ``pi05_libero`` 数据集：复用 openpi 的 ``pi05_libero`` TrainConfig
    解析 repack / norm_stats / repo_id / action_horizon。新增其它 benchmark
    时，在本目录下再加一个注册文件即可，无需改核心 dataloader。

架构位置：
    数据层（配置）— import 时副作用注册，由 dataloader/__init__.py 统一触发。
"""

from __future__ import annotations

from chameleon.dataloader.base import DatasetSpec, register_dataset

register_dataset(
    DatasetSpec(
        name="pi05_libero",
        loader="lerobot",
        openpi_config="pi05_libero",
        # repo_id / dataset_root 留空：默认走 openpi DataConfig 与 HF 缓存；
        # 本地数据可在 TaskConfig.data 或 build_dataset overrides 中指定。
    ),
    override=True,
)
