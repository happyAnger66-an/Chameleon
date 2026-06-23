"""evaluate 任务配置解析辅助 — 避免 api ↔ evaluate 循环 import。"""

from __future__ import annotations

import torch

from chameleon.config.schema import TaskConfig


def resolve_torch_device(requested: str | None) -> str | None:
    if requested is None:
        return None
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


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


def resolve_pytorch_load_device(task: TaskConfig) -> str:
    """openpi 权重加载 device（TRT 评测路径在释放大模块前使用）。"""
    raw = task.evaluate.pytorch_load_device or "cpu"
    return resolve_torch_device(raw) or "cpu"


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
