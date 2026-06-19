"""sync_eval_num_samples 单元测试。"""

from __future__ import annotations

from chameleon.config.schema import DataConfig, EvaluateConfig, TaskConfig
from chameleon.evaluate.task_utils import sync_eval_num_samples


def test_sync_expands_data_window_when_smaller() -> None:
    task = TaskConfig(
        data=DataConfig(num_samples=20),
        evaluate=EvaluateConfig(num_samples=1000),
    )
    n = sync_eval_num_samples(task)
    assert n == 1000
    assert task.data.num_samples == 1000


def test_sync_keeps_larger_data_window() -> None:
    task = TaskConfig(
        data=DataConfig(num_samples=2000),
        evaluate=EvaluateConfig(num_samples=1000),
    )
    n = sync_eval_num_samples(task)
    assert n == 1000
    assert task.data.num_samples == 2000


def test_sync_sets_data_when_unset() -> None:
    task = TaskConfig(
        data=DataConfig(),
        evaluate=EvaluateConfig(num_samples=50),
    )
    sync_eval_num_samples(task)
    assert task.data.num_samples == 50
