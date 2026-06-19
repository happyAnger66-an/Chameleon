"""LeRobot 数据源 — 复用 openpi DataConfig + repack 加载真实数据集。

作用：
    LeRobotDataSource 把 openpi 的数据管线包装成 Chameleon 的统一接口：
      1. ``openpi.training.config.get_config(name)`` 解析 TrainConfig；
      2. ``train_cfg.data.create(...)`` 得到 DataConfig（含 repo_id / repack /
         norm_stats / action_sequence_keys / prompt_from_task）；
      3. 构建带 ``delta_timestamps`` 的 ``LeRobotDataset``（取 action_horizon 帧动作）；
      4. ``repack_only`` 把原始 lerobot 列映射成 openpi 推理输入键
         （``image`` / ``state`` / ``actions`` / ``prompt``）；
      5. ``__getitem__`` 拆出 ground-truth 动作与 observation，打包为 ChameleonSample。
    归一化 / tokenize 仍交由下游 openpi ``Policy.infer`` 内部完成（单一事实来源）。

架构位置：
    数据层 — 由 build_dataset() 依据 DatasetSpec 构建。上游：evaluate / CLI；
    下游：openpi Policy 或 Pi05RealOrchestrator。openpi / lerobot 为延迟导入，
    无依赖环境下本模块仍可 import（仅 build() 时报错）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from chameleon.dataloader.base import (
    ChameleonSample,
    DatasetSpec,
    get_dataset_spec,
)

logger = logging.getLogger(__name__)


def _tree_to_numpy(obj: Any) -> Any:
    """LeRobot 常返回 torch.Tensor；openpi transforms 期望 numpy。"""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch 总是存在于推理环境
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _tree_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_to_numpy(x) for x in obj)
    return obj


def _episode_ids_per_frame(dataset: Any, n_frames: int) -> np.ndarray:
    """尽力解析每帧所属 episode id；失败时回退为全 0。"""
    base = dataset
    while hasattr(base, "_dataset"):
        base = base._dataset
    hf = getattr(base, "hf_dataset", None)
    if hf is not None:
        cols = getattr(hf, "column_names", []) or []
        for col in ("episode_index", "episode_idx", "ep_index"):
            if col in cols:
                ep = np.asarray(hf[col][:n_frames])
                if ep.shape[0] == n_frames:
                    return ep.astype(np.int64, copy=False)
    return np.zeros(n_frames, dtype=np.int64)


class LeRobotDataSource:
    """按 DatasetSpec 加载 LeRobot 数据集，产出 ChameleonSample。

    懒加载：构造仅记录 spec，真正的 openpi / lerobot 导入与数据集打开发生在
    build()（或首次 __getitem__）时。
    """

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self._built = False
        self._dataset: Any = None
        self._repack_fn: Any = None
        self._action_horizon: int = 0
        self._action_dim: int = 0
        self._repo_id: str | None = None
        self._start = max(0, int(spec.start_index))
        self._length = 0
        self._episode_ids: np.ndarray | None = None

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self) -> "LeRobotDataSource":
        if self._built:
            return self

        try:
            import openpi.transforms as _transforms
            from openpi.training import config as _config
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise ImportError(
                "加载 LeRobot 数据集需要可 import 的 openpi（openpi.training.config / "
                "openpi.transforms）。请在 openpi 环境下运行（见 README 的真实权重说明）。"
            ) from exc

        train_cfg = _config.get_config(self.spec.openpi_config)
        data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

        repo_id = self.spec.repo_id or getattr(data_config, "repo_id", None)
        if not repo_id:
            raise ValueError(
                f"数据集 {self.spec.name!r}（openpi_config={self.spec.openpi_config!r}）"
                "未解析出 repo_id；请在 DatasetSpec / data 配置中显式设置 repo_id。"
            )

        action_horizon = int(self.spec.action_horizon or train_cfg.model.action_horizon)
        action_dim = int(getattr(train_cfg.model, "action_dim", 0) or 0)
        action_keys = tuple(data_config.action_sequence_keys)

        dataset = self._make_lerobot_dataset(
            repo_id=repo_id,
            action_horizon=action_horizon,
            action_sequence_keys=action_keys,
            prompt_from_task=bool(getattr(data_config, "prompt_from_task", False)),
            dataset_root=self.spec.dataset_root,
            transforms_mod=_transforms,
        )
        repack_fn = _transforms.compose([*data_config.repack_transforms.inputs])

        n_total = len(dataset)
        start = min(self._start, n_total)
        if self.spec.num_samples is not None:
            length = min(int(self.spec.num_samples), n_total - start)
        else:
            length = n_total - start

        self._dataset = dataset
        self._repack_fn = repack_fn
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._repo_id = repo_id
        self._start = start
        self._length = max(0, length)
        self._episode_ids = _episode_ids_per_frame(dataset, n_total)
        self._n_total = n_total
        self._built = True

        logger.info(
            "LeRobot dataset ready: repo_id=%s frames=%d (window [%d, %d)) "
            "action_horizon=%d action_dim=%d",
            repo_id,
            n_total,
            start,
            start + self._length,
            action_horizon,
            action_dim,
        )
        return self

    @staticmethod
    def _make_lerobot_dataset(
        *,
        repo_id: str,
        action_horizon: int,
        action_sequence_keys: tuple[str, ...],
        prompt_from_task: bool,
        dataset_root: str | None,
        transforms_mod: Any,
    ) -> Any:
        try:
            import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        except ImportError:
            try:
                import lerobot.datasets.lerobot_dataset as lerobot_dataset  # type: ignore[no-redef]
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "需要安装 lerobot（openpi 使用 lerobot.common.datasets.lerobot_dataset；"
                    "新版包可能为 lerobot.datasets.lerobot_dataset）。"
                ) from exc

        from openpi.training.data_loader import TransformedDataset

        meta_kw: dict = {"repo_id": repo_id}
        if dataset_root is not None:
            meta_kw["root"] = str(dataset_root)
        try:
            meta = lerobot_dataset.LeRobotDatasetMetadata(**meta_kw)
        except TypeError:
            meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

        ds_kwargs: dict = {
            "repo_id": repo_id,
            "delta_timestamps": {
                key: [t / meta.fps for t in range(action_horizon)] for key in action_sequence_keys
            },
        }
        if dataset_root is not None:
            ds_kwargs["root"] = str(dataset_root)
        try:
            dataset = lerobot_dataset.LeRobotDataset(**ds_kwargs)
        except TypeError:
            ds_kwargs.pop("root", None)
            logger.warning("当前 lerobot 版本忽略 dataset_root，请用 HF 缓存或升级 lerobot。")
            dataset = lerobot_dataset.LeRobotDataset(**ds_kwargs)

        if prompt_from_task:
            dataset = TransformedDataset(dataset, [transforms_mod.PromptFromLeRobotTask(meta.tasks)])
        return dataset

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if not self._built:
            self.build()
        return self._length

    def __getitem__(self, index: int) -> ChameleonSample:
        if not self._built:
            self.build()
        if index < 0:
            index += self._length
        if not (0 <= index < self._length):
            raise IndexError(f"index {index} out of range [0, {self._length})")

        global_index = self._start + index
        raw = _tree_to_numpy(self._dataset[global_index])
        packed = self._repack_fn(dict(raw))
        if "actions" not in packed:
            raise KeyError(
                "repack 后缺少 'actions'，请检查数据集列名与 openpi DataConfig 是否一致。"
            )

        actions_gt = np.asarray(packed["actions"])
        observation = {k: v for k, v in packed.items() if k != "actions"}

        prompt: str | None = None
        if "prompt" in packed:
            try:
                prompt = str(packed["prompt"])
            except Exception:  # noqa: BLE001
                prompt = None

        episode_id = None
        if self._episode_ids is not None and global_index < self._episode_ids.shape[0]:
            episode_id = int(self._episode_ids[global_index])

        return ChameleonSample(
            observation=observation,
            actions_gt=actions_gt,
            prompt=prompt,
            index=global_index,
            episode_id=episode_id,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # --- metadata --------------------------------------------------------
    @property
    def repo_id(self) -> str | None:
        return self._repo_id

    @property
    def action_horizon(self) -> int:
        if not self._built:
            self.build()
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        if not self._built:
            self.build()
        return self._action_dim

    @property
    def start_index(self) -> int:
        if not self._built:
            self.build()
        return self._start

    @property
    def frame_count(self) -> int:
        """底层 LeRobot 数据集总帧数（全局 index 上界）。"""
        if not self._built:
            self.build()
        return self._n_total

    @property
    def eval_end_exclusive(self) -> int:
        """当前评测窗口在全局帧 index 上的右开区间上界。"""
        if not self._built:
            self.build()
        return self._start + self._length

    @property
    def episode_ids_per_frame(self) -> np.ndarray:
        if not self._built:
            self.build()
        return self._episode_ids


_LOADERS: dict[str, type[LeRobotDataSource]] = {
    "lerobot": LeRobotDataSource,
}


def build_dataset(name: str, *, overrides: dict[str, Any] | None = None) -> LeRobotDataSource:
    """按注册名构建数据源（应用可选 overrides），不立即打开数据集。

    Args:
        name: ``DATASET_REGISTRY`` 中的数据集名（见 ``list_datasets()``）。
        overrides: 覆盖 DatasetSpec 字段（如 ``repo_id`` / ``dataset_root`` /
            ``num_samples``），仅覆盖非 None 值。

    Returns:
        懒加载的 DataSource（调用 ``build()`` / ``__getitem__`` 时才真正打开）。
    """
    spec = get_dataset_spec(name).merged(overrides)
    loader_cls = _LOADERS.get(spec.loader)
    if loader_cls is None:
        raise KeyError(
            f"Unknown dataset loader {spec.loader!r} for dataset {name!r}. "
            f"Available: {sorted(_LOADERS)}"
        )
    return loader_cls(spec)


def build_dataset_from_config(data_cfg: Any) -> LeRobotDataSource:
    """从 TaskConfig.data（DataConfig）构建数据源。"""
    if not getattr(data_cfg, "dataset", None):
        raise ValueError("TaskConfig.data.dataset 未设置，无法构建 dataloader。")
    overrides = {
        "repo_id": getattr(data_cfg, "repo_id", None),
        "dataset_root": getattr(data_cfg, "dataset_root", None),
        "openpi_config": getattr(data_cfg, "openpi_config", None),
        "action_horizon": getattr(data_cfg, "action_horizon", None),
        "start_index": getattr(data_cfg, "start_index", None),
        "num_samples": getattr(data_cfg, "num_samples", None),
    }
    return build_dataset(data_cfg.dataset, overrides=overrides)


def _smoke_main(argv: list[str] | None = None) -> int:
    """冒烟入口：``python -m chameleon.dataloader.lerobot --dataset pi05_libero``。"""
    import argparse

    import chameleon.dataloader  # noqa: F401  触发数据集注册

    parser = argparse.ArgumentParser(description="Chameleon LeRobot dataloader smoke test")
    parser.add_argument("--dataset", default="pi05_libero")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--openpi-config", default=None)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    source = build_dataset(
        args.dataset,
        overrides={
            "repo_id": args.repo_id,
            "dataset_root": args.dataset_root,
            "openpi_config": args.openpi_config,
            "num_samples": args.num_samples,
            "start_index": args.start_index,
        },
    )
    source.build()
    print(f"dataset={args.dataset} repo_id={source.repo_id} len={len(source)}")
    print(f"action_horizon={source.action_horizon} action_dim={source.action_dim}")

    n = min(len(source), args.num_samples)
    for i in range(n):
        sample = source[i]
        obs_keys = sorted(sample.observation.keys())
        print(
            f"[{i}] global_index={sample.index} episode={sample.episode_id} "
            f"prompt={sample.prompt!r} actions_gt={np.asarray(sample.actions_gt).shape} "
            f"obs_keys={obs_keys}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_smoke_main())
