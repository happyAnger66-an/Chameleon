"""evaluate 任务配置解析辅助 — 避免 api ↔ evaluate 循环 import。"""

from __future__ import annotations

from pathlib import Path

import torch

from chameleon.config.schema import TaskConfig


def resolve_torch_device(requested: str | None) -> str | None:
    if requested is None:
        return None
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def resolve_checkpoint_dir(task: TaskConfig) -> str:
    eval_cfg = task.evaluate
    checkpoint_dir = eval_cfg.checkpoint_dir
    if checkpoint_dir is None:
        ckpt = task.model_overrides.get("checkpoint")
        if ckpt:
            checkpoint_dir = str(Path(ckpt).parent)
    if not checkpoint_dir:
        raise ValueError(
            "无法确定 checkpoint 目录：请设置 evaluate.checkpoint_dir 或 "
            "model_overrides.checkpoint。"
        )
    return checkpoint_dir


def resolve_openpi_config(task: TaskConfig) -> str:
    if task.data.openpi_config:
        return task.data.openpi_config
    if task.data.dataset:
        from chameleon.dataloader import get_dataset_spec

        return get_dataset_spec(task.data.dataset).openpi_config
    return "pi05_libero"


def resolve_eval_device(task: TaskConfig) -> str | None:
    device = task.evaluate.device or task.infer.torch_device
    return resolve_torch_device(device)


def sync_eval_num_samples(task: TaskConfig) -> int:
    """统一评测帧数：``evaluate.num_samples`` 为权威值，并扩展 ``data`` 窗口。

    ``data.num_samples`` 限制 dataloader 暴露长度；若小于 ``evaluate.num_samples``，
    评测循环会被 ``len(data_source)`` 截断（CLI ``--num-samples`` 只改 evaluate 时
    会出现此问题）。
    """
    n = int(task.evaluate.num_samples)
    data_n = task.data.num_samples
    if data_n is None or int(data_n) < n:
        task.data.num_samples = n
    return n
