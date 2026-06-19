"""数据加载抽象 — 统一样本契约 + 数据集规格注册表。

作用：
    - ChameleonSample：dataloader 输出的统一样本结构（openpi-ready observation
      + ground-truth 动作 + 元信息），供 evaluate / 真实数据推理路径消费。
    - DatasetSpec：声明一个可加载数据集所需的最小信息（openpi 配置名、repo_id、
      数据根目录、action_horizon 等），与具体加载实现解耦。
    - DATASET_REGISTRY：按名称注册数据集规格（借鉴 starVLA 的 data_mix registry），
      新增 benchmark 只需在 dataloader/configs/ 下注册一个 spec，无需改核心代码。

架构位置：
    数据层 — 被 dataloader/lerobot.py（构建具体 DataSource）与上层 evaluate /
    CLI 使用。不直接依赖 openpi / lerobot，保证无依赖环境下也能 import。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from chameleon.core.registry import Registry


@dataclass
class ChameleonSample:
    """dataloader 产出的单帧样本（已 repack 成 openpi 推理输入格式）。

    Attributes:
        observation: openpi ``Policy.infer`` 可直接消费的 obs dict
            （含 ``image`` / ``state`` / ``prompt`` 等 repack 后的键），
            归一化 / tokenize 由下游 policy 内部完成。
        actions_gt: ground-truth 动作序列，形状 ``[action_horizon, action_dim]``。
        prompt: 文本指令（若数据集提供）。
        index: 该样本在数据集中的全局帧索引。
        episode_id: 所属 episode 的 id（若可解析，否则为 None）。
    """

    observation: dict[str, Any]
    actions_gt: Any
    prompt: str | None = None
    index: int = 0
    episode_id: int | None = None


@dataclass
class DatasetSpec:
    """一个可加载数据集的声明式规格。

    与加载实现解耦：``loader`` 指明用哪个 DataSource 实现（当前仅 ``lerobot``）。
    其余字段为该实现的参数；``build_dataset`` 时可被 overrides 覆盖。
    """

    name: str
    loader: str = "lerobot"

    # openpi TrainConfig 名（解析 repack / norm_stats / repo_id / action_horizon）。
    openpi_config: str = "pi05_libero"

    # 可选覆盖项：缺省时从 openpi DataConfig / model config 推导。
    repo_id: str | None = None
    dataset_root: str | None = None
    action_horizon: int | None = None

    # 取样区间（快速冒烟用）。
    start_index: int = 0
    num_samples: int | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def merged(self, overrides: dict[str, Any] | None = None) -> "DatasetSpec":
        """返回应用了 overrides 的副本（仅覆盖非 None 值）。"""
        if not overrides:
            return self
        valid = {f.name for f in self.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        clean = {k: v for k, v in overrides.items() if k in valid and v is not None}
        return replace(self, **clean)


DATASET_REGISTRY: Registry[str, DatasetSpec] = Registry("dataset")


def register_dataset(spec: DatasetSpec, *, override: bool = False) -> DatasetSpec:
    return DATASET_REGISTRY.register(spec.name, spec, override=override)


def get_dataset_spec(name: str) -> DatasetSpec:
    return DATASET_REGISTRY.get(name)


def list_datasets() -> list[str]:
    return DATASET_REGISTRY.keys()
