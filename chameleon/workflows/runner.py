"""Workflow runner: executes a TaskConfig's actions in order.

Thin orchestration over the high-level API (mirrors ``model_optimizer``'s
``WorkflowRunner`` which translates a manifest into existing commands rather than
re-implementing logic). Records every artifact in a :class:`Manifest`.
"""

from __future__ import annotations

import logging

import torch

from chameleon.api import build_adapter, run_compile, run_infer, run_quantize
from chameleon.config.schema import TaskConfig
from chameleon.core.artifact import Artifact, Manifest

logger = logging.getLogger(__name__)


class WorkflowRunner:
    def __init__(self, task: TaskConfig) -> None:
        self.task = task
        self.manifest = Manifest(task.output_dir)

    def plan(self) -> list[str]:
        """Return a human-readable description of the steps without running them."""
        lines = [
            f"architecture={self.task.architecture} model={self.task.model} "
            f"platform={self.task.platform}",
            f"actions={self.task.actions}",
        ]
        for action in self.task.actions:
            if action == "quantize":
                for s in self.task.quantize:
                    lines.append(f"  quantize: stage={s.stage} method={s.method}")
            elif action == "compile":
                for s in self.task.compile:
                    lines.append(f"  compile:  stage={s.stage} options={s.options}")
            elif action == "infer":
                lines.append(
                    f"  infer:    batch={self.task.infer.batch_size} "
                    f"stage_runtimes={self.task.stage_runtimes or 'platform-default'}"
                )
        return lines

    def run(self, *, dry_run: bool = False) -> Manifest:
        if dry_run:
            for line in self.plan():
                logger.info(line)
            return self.manifest

        adapter = build_adapter(self.task)
        compiled_engines: dict[str, Artifact] = {}
        for action in self.task.actions:
            if action == "quantize":
                run_quantize(self.task, adapter, self.manifest)
            elif action == "compile":
                compiled_engines = run_compile(self.task, adapter, self.manifest)
            elif action == "infer":
                stage_artifacts, stage_runtimes = self._infer_engine_bindings(compiled_engines)
                actions_out = run_infer(
                    self.task,
                    stage_artifacts=stage_artifacts,
                    stage_runtimes=stage_runtimes,
                )
                used = sorted(stage_artifacts) if stage_artifacts else []
                self.manifest.add(
                    Artifact(
                        kind="inference",
                        platform=self.task.platform,
                        metadata={
                            "action_shape": list(actions_out.shape),
                            "action_mean": float(torch.as_tensor(actions_out).float().mean()),
                            "engines_used": used,
                        },
                    )
                )
                if used:
                    logger.info("Inference consumed compiled engines for stages: %s", used)
            else:
                raise ValueError(f"Unknown action {action!r}.")
        self.manifest.save()
        return self.manifest

    def _infer_engine_bindings(
        self, compiled_engines: dict[str, Artifact]
    ) -> tuple[dict[str, Artifact] | None, dict[str, str] | None]:
        """Decide which stages infer should run on compiled engines.

        Only when ``infer.use_compiled_engines`` is set and an engine was built
        for a stage; those stages use the platform runtime (e.g. tensorrt), the
        rest fall back to the task's configured stage runtimes.
        """
        if not self.task.infer.use_compiled_engines or not compiled_engines:
            return None, None
        from chameleon.core.platform import get_platform

        runtime_name = get_platform(self.task.platform).runtime
        stage_runtimes = dict(self.task.stage_runtimes)
        for stage in compiled_engines:
            stage_runtimes[stage] = runtime_name
        return compiled_engines, stage_runtimes
