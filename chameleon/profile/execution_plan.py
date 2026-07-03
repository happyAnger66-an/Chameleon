"""从 TaskConfig 推断单次推理的计算统计执行计划。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chameleon.architectures.registry import get_architecture
from chameleon.config.schema import TaskConfig
from chameleon.deploy.backends import is_cosmos3_deploy_backend, is_pi05_deploy_backend

_DEFAULT_PI05_DEPLOY_STAGES = ("vit", "llm", "expert", "denoise")
_DEFAULT_COSMOS3_STAGES = ("vae_encode", "text_embed", "dit", "vae_decode")


class PlanMode(str, Enum):
    DEPLOY = "deploy"
    REFERENCE = "reference"
    REAL = "real"


@dataclass(frozen=True)
class StageRepeat:
    stage: str
    repeat: int


@dataclass(frozen=True)
class ExecutionPlan:
    mode: PlanMode
    stages: tuple[StageRepeat, ...]
    num_steps: int
    batch_size: int

    def describe(self) -> str:
        parts = [f"{sr.stage}×{sr.repeat}" for sr in self.stages]
        return f"{self.mode.value} ({', '.join(parts)})"


def _configured_deploy_stages(task: TaskConfig, *, default_stages: tuple[str, ...]) -> tuple[str, ...]:
    if task.export:
        return tuple(s.stage for s in task.export)
    if task.compile:
        return tuple(s.stage for s in task.compile)
    return default_stages


def _pi05_deploy_stage_repeats(configured: tuple[str, ...], num_steps: int) -> tuple[StageRepeat, ...]:
    repeats: list[StageRepeat] = []
    if "vit" in configured:
        repeats.append(StageRepeat("vit", 1))
    if "llm" in configured:
        repeats.append(StageRepeat("llm", 1))
    # denoise ONNX 已内嵌 expert；与 TRT 运行时一致，避免重复计数。
    if "denoise" in configured:
        repeats.append(StageRepeat("denoise", num_steps))
    elif "expert" in configured:
        repeats.append(StageRepeat("expert", num_steps))
    return tuple(repeats)


def _cosmos3_stage_repeats(configured: tuple[str, ...], num_steps: int) -> tuple[StageRepeat, ...]:
    repeats: list[StageRepeat] = []
    if "vae_encode" in configured:
        repeats.append(StageRepeat("vae_encode", 1))
    if "text_embed" in configured:
        repeats.append(StageRepeat("text_embed", 1))
    if "dit" in configured:
        repeats.append(StageRepeat("dit", num_steps))
    if "vae_decode" in configured:
        repeats.append(StageRepeat("vae_decode", 1))
    return tuple(repeats)


def build_execution_plan(task: TaskConfig) -> ExecutionPlan:
    """根据 TaskConfig 生成 stage × repeat 计划。"""
    arch = get_architecture(task.architecture)
    num_steps = task.infer.num_steps
    if num_steps is None:
        gen_steps = getattr(task.generate, "num_inference_steps", None)
        if gen_steps is not None:
            num_steps = int(gen_steps)
        else:
            num_steps = int(
                task.model_overrides.get("num_denoise_steps", arch.metadata.get("num_denoise_steps", 10))
            )
    batch_size = task.infer.batch_size

    use_reference = bool(task.model_overrides.get("use_reference", True))
    arch_name = task.architecture

    if is_cosmos3_deploy_backend(task.deploy.backend):
        configured = _configured_deploy_stages(task, default_stages=_DEFAULT_COSMOS3_STAGES)
        stages = _cosmos3_stage_repeats(configured, num_steps)
        mode = PlanMode.DEPLOY
    elif arch_name == "cosmos3" and not use_reference:
        configured = _configured_deploy_stages(task, default_stages=_DEFAULT_COSMOS3_STAGES)
        stages = _cosmos3_stage_repeats(configured, num_steps)
        mode = PlanMode.REAL
    elif arch_name == "cosmos3":
        stages = _cosmos3_stage_repeats(_DEFAULT_COSMOS3_STAGES, num_steps)
        mode = PlanMode.REFERENCE
    elif is_pi05_deploy_backend(task.deploy.backend):
        configured = _configured_deploy_stages(task, default_stages=_DEFAULT_PI05_DEPLOY_STAGES)
        stages = _pi05_deploy_stage_repeats(configured, num_steps)
        mode = PlanMode.DEPLOY
    elif not use_reference and arch_name == "pi05":
        configured = _configured_deploy_stages(task, default_stages=_DEFAULT_PI05_DEPLOY_STAGES)
        stages = _pi05_deploy_stage_repeats(configured, num_steps)
        mode = PlanMode.REAL
    else:
        stages = (
            StageRepeat("vit", 1),
            StageRepeat("llm_prefix", 1),
            StageRepeat("action_expert", num_steps),
        )
        mode = PlanMode.REFERENCE

    if not stages:
        raise ValueError("Execution plan has no stages; check export/compile configuration.")

    return ExecutionPlan(mode=mode, stages=stages, num_steps=num_steps, batch_size=batch_size)
