"""工作流运行器 — 按 TaskConfig.actions 顺序执行各 action。

作用：
    WorkflowRunner 是纯分发层：遍历 task.actions，从 ACTION_REGISTRY 解析对应的
    WorkflowAction 并调用其 plan/run。各 action 的具体逻辑（quantize / export /
    compile / trt_profile / infer）与跨步骤共享状态封装在 workflows/actions.py，
    runner 本身不感知任何 action 名称，新增 action 无需改动此文件。

架构位置：
    入口/编排层 — 被 cli workflow / quantize / compile 子命令调用。
    设计对标 model_optimizer WorkflowRunner（复用 API 而非重复逻辑）。
"""

from __future__ import annotations

import logging

from chameleon.config.schema import TaskConfig
from chameleon.core.artifact import Manifest
from chameleon.workflows.actions import ACTION_REGISTRY, WorkflowContext

logger = logging.getLogger(__name__)


class WorkflowRunner:
    def __init__(self, task: TaskConfig) -> None:
        self.task = task
        self.manifest = Manifest(task.output_dir)

    def plan(self) -> list[str]:
        """Return a human-readable description of the steps without running them."""
        ctx = WorkflowContext(self.task, self.manifest)
        lines = [
            f"architecture={self.task.architecture} model={self.task.model} "
            f"platform={self.task.platform}",
            f"actions={self.task.actions}",
        ]
        for name in self.task.actions:
            action = ACTION_REGISTRY.get_or_none(name)
            if action is not None:
                lines.extend(action.plan(ctx))
        return lines

    def run(self, *, dry_run: bool = False) -> Manifest:
        if dry_run:
            for line in self.plan():
                logger.info(line)
            return self.manifest

        ctx = WorkflowContext(self.task, self.manifest)
        for name in self.task.actions:
            action = ACTION_REGISTRY.get_or_none(name)
            if action is None:
                raise ValueError(
                    f"Unknown action {name!r}. Available: {ACTION_REGISTRY.keys()}"
                )
            action.run(ctx)
        self.manifest.save()
        return self.manifest
