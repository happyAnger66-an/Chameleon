"""工作流 action 注册表 — 每个 action 自带 plan/run，由 WorkflowRunner 分发。

作用：
    定义 WorkflowContext（跨 action 共享 adapter / compiled_engines）与
    WorkflowAction 抽象，把 quantize / export / compile / trt_profile / infer 各自的
    计划与执行逻辑收拢为独立可注册的命令对象。新增 action 只需写一个带 @register
    的类，无需改动 WorkflowRunner 的分发循环。

架构位置：
    入口/编排层 — 被 workflows/runner.py 遍历分发；复用 api.py 的薄编排函数。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from chameleon.api import (
    build_adapter,
    run_compile,
    run_deploy_build,
    run_export,
    run_infer,
    run_quantize,
    run_trt_profile,
)
from chameleon.config.schema import TaskConfig
from chameleon.core.artifact import Artifact, Manifest
from chameleon.core.registry import Registry
from chameleon.deploy.backends import is_cosmos3_deploy_backend, is_pi05_deploy_backend
from chameleon.deploy.trt_profile import iter_profile_steps

logger = logging.getLogger(__name__)

ACTION_REGISTRY: Registry[str, "WorkflowAction"] = Registry("workflow_action")


@dataclass
class WorkflowContext:
    """Action 间共享的可变上下文。

    承载 quantize/compile/infer 共用的惰性 adapter，以及 compile 产出、
    infer 消费的 compiled_engines，使分发循环无需感知任何跨步骤状态。
    """

    task: TaskConfig
    manifest: Manifest
    compiled_engines: dict[str, Artifact] = field(default_factory=dict)
    _adapter: Any = None

    def adapter(self) -> Any:
        """惰性构建并缓存 adapter（首次调用时构建，后续步骤复用）。"""
        if self._adapter is None:
            self._adapter = build_adapter(self.task)
        return self._adapter

    def infer_engine_bindings(
        self,
    ) -> tuple[dict[str, Artifact] | None, dict[str, str] | None]:
        """决定 infer 在哪些 stage 上运行已编译 engine。

        仅当 ``infer.use_compiled_engines`` 开启且确有 engine 被构建时，这些 stage
        使用平台运行时（如 tensorrt），其余回退到任务配置的 stage_runtimes。
        """
        if not self.task.infer.use_compiled_engines or not self.compiled_engines:
            return None, None
        from chameleon.core.platform import get_platform

        runtime_name = get_platform(self.task.platform).runtime
        stage_runtimes = dict(self.task.stage_runtimes)
        for stage in self.compiled_engines:
            stage_runtimes[stage] = runtime_name
        return self.compiled_engines, stage_runtimes


class WorkflowAction(ABC):
    """单个工作流动作：自带 plan（dry-run 描述）与 run（执行）。"""

    name: str

    @abstractmethod
    def plan(self, ctx: WorkflowContext) -> list[str]:
        """返回该 action 在 dry-run 下的人类可读描述行。"""

    @abstractmethod
    def run(self, ctx: WorkflowContext) -> None:
        """执行该 action，按需读写 ``ctx`` 上的共享状态与 manifest。"""


def register_action(action: WorkflowAction, *, override: bool = False) -> WorkflowAction:
    return ACTION_REGISTRY.register(action.name, action, override=override)


class QuantizeAction(WorkflowAction):
    name = "quantize"

    def plan(self, ctx: WorkflowContext) -> list[str]:
        return [
            f"  quantize: stage={s.stage} method={s.method}" for s in ctx.task.quantize
        ]

    def run(self, ctx: WorkflowContext) -> None:
        run_quantize(ctx.task, ctx.adapter(), ctx.manifest)


class ExportAction(WorkflowAction):
    name = "export"

    def plan(self, ctx: WorkflowContext) -> list[str]:
        task = ctx.task
        backend = task.deploy.backend
        export_dir = task.deploy.export_dir or f"{task.output_dir}/onnx"
        lines = [
            f"  export:   stage={s.stage} backend={backend}" for s in task.export or []
        ]
        if not task.export:
            if is_cosmos3_deploy_backend(backend):
                default_stages = "vae_encode/text_embed/dit/vae_decode"
            else:
                default_stages = "vit/llm/expert/denoise"
            lines.append(f"  export:   default stages {default_stages} -> {export_dir}")
        return lines

    def run(self, ctx: WorkflowContext) -> None:
        run_export(ctx.task, ctx.manifest)


class CompileAction(WorkflowAction):
    name = "compile"

    def plan(self, ctx: WorkflowContext) -> list[str]:
        return [
            f"  compile:  stage={s.stage} options={s.options}" for s in ctx.task.compile
        ]

    def run(self, ctx: WorkflowContext) -> None:
        backend = ctx.task.deploy.backend
        if is_pi05_deploy_backend(backend) or is_cosmos3_deploy_backend(backend):
            ctx.compiled_engines = run_deploy_build(ctx.task, ctx.manifest)
        else:
            ctx.compiled_engines = run_compile(ctx.task, ctx.adapter(), ctx.manifest)


class TrtProfileAction(WorkflowAction):
    name = "trt_profile"

    def plan(self, ctx: WorkflowContext) -> list[str]:
        task = ctx.task
        profile_dir = task.profile.profile_dir or f"{task.output_dir}/profiles"
        lines = [
            f"  trt_profile: stage={step.stage} -> "
            f"{profile_dir}/{step.stage}.profile.json"
            for step in iter_profile_steps(task)
        ]
        lines.append(
            f"  trt_profile viewer={task.profile.viewer} "
            f"iterations={task.profile.iterations}"
        )
        return lines

    def run(self, ctx: WorkflowContext) -> None:
        run_trt_profile(ctx.task, ctx.manifest)


class InferAction(WorkflowAction):
    name = "infer"

    def plan(self, ctx: WorkflowContext) -> list[str]:
        task = ctx.task
        return [
            f"  infer:    batch={task.infer.batch_size} "
            f"stage_runtimes={task.stage_runtimes or 'platform-default'}"
        ]

    def run(self, ctx: WorkflowContext) -> None:
        # 与重构前保持一致：infer 前确保 adapter 已构建（含其构建期副作用）。
        ctx.adapter()
        stage_artifacts, stage_runtimes = ctx.infer_engine_bindings()
        actions_out = run_infer(
            ctx.task,
            stage_artifacts=stage_artifacts,
            stage_runtimes=stage_runtimes,
        )
        used = sorted(stage_artifacts) if stage_artifacts else []
        metadata: dict[str, Any] = {"engines_used": used}
        try:
            tensor_out = torch.as_tensor(actions_out)
            metadata["action_shape"] = list(tensor_out.shape)
            metadata["action_mean"] = float(tensor_out.float().mean())
        except (TypeError, ValueError, RuntimeError):
            # Non-tensor primary output (e.g. cosmos3 PIL / list video); record type only.
            metadata["output_type"] = type(actions_out).__name__
        ctx.manifest.add(
            Artifact(
                kind="inference",
                platform=ctx.task.platform,
                metadata=metadata,
            )
        )
        if used:
            logger.info("Inference consumed compiled engines for stages: %s", used)


for _action in (
    QuantizeAction(),
    ExportAction(),
    CompileAction(),
    TrtProfileAction(),
    InferAction(),
):
    register_action(_action)
