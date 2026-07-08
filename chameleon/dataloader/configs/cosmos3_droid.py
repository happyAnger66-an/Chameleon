"""Cosmos3 + DROID(RLDS) 数据集规格 — 注册到 DATASET_REGISTRY。

作用：
    声明 ``cosmos3_droid`` 数据集：用 ``droid_rlds`` loader 读取 DROID RLDS
    （如 ``droid_100``），产出 cosmos3 的 10D 动作 GT 与条件图像观测。与 pi05 的
    openpi repack 通路解耦（动作表示不同，见 ``dataloader/cosmos3_droid.py``）。

    ``dataset_root``（tfds data_dir）在 eval yaml 的 ``data.dataset_root`` 或
    ``build_dataset`` overrides 中指定；``extra`` 里可覆盖 builder / version / camera。

架构位置：
    数据层（配置）— import 时副作用注册，由 dataloader/configs/__init__.py 触发。
"""

from __future__ import annotations

from chameleon.dataloader.base import DatasetSpec, register_dataset
from chameleon.dataloader.cosmos3_droid import (
    DROID_ACTION_HORIZON,
    DROID_RAW_ACTION_DIM,
)

register_dataset(
    DatasetSpec(
        name="cosmos3_droid",
        loader="droid_rlds",
        # openpi_config 对 droid_rlds loader 无意义（不走 openpi repack），保留占位。
        openpi_config="",
        action_horizon=DROID_ACTION_HORIZON,
        extra={
            "builder": "droid_100",
            "camera": "exterior_image_1_left",
            "wrist_camera": "wrist_image_left",
            "raw_action_dim": DROID_RAW_ACTION_DIM,
        },
    ),
    override=True,
)
