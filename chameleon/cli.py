"""Chameleon command-line interface.

Subcommands:
    platforms        list registered deployment platforms
    architectures    list architectures and their stages
    info             list registered backends (compilers/runtimes/quant/kernels)
    infer            run a single inference from a task config
    quantize         run only the quantize action of a task config
    compile          run only the compile action of a task config
    workflow         run a full task config (quantize -> compile -> infer)
    profile          measure inference latency
"""

from __future__ import annotations

import argparse
import logging

import chameleon  # noqa: F401  (triggers import-time registration)
from chameleon.architectures.registry import ARCHITECTURE_REGISTRY, get_architecture
from chameleon.compile.base import COMPILER_REGISTRY, get_compiler
from chameleon.config.schema import TaskConfig
from chameleon.core.platform import list_platforms
from chameleon.kernels.base import KERNEL_REGISTRY
from chameleon.quantization.registry import QUANT_METHOD_REGISTRY
from chameleon.runtime.base import RUNTIME_REGISTRY


def _load_task(args) -> TaskConfig:
    task = TaskConfig.load(args.config)
    if getattr(args, "platform", None):
        task.platform = args.platform
    if getattr(args, "runtime", None):
        # Apply one runtime to all stages (overrides stage_runtimes).
        task.stage_runtimes = {s: args.runtime for s in get_architecture(task.architecture).stage_names}
    return task


def cmd_platforms(args) -> None:
    print(f"{'NAME':<16}{'VENDOR':<10}{'DEVICE':<8}{'COMPILER':<12}{'RUNTIME':<10}DTYPES")
    for spec in list_platforms():
        print(
            f"{spec.name:<16}{spec.vendor:<10}{spec.device:<8}"
            f"{spec.compiler:<12}{spec.runtime:<10}{','.join(spec.dtypes)}"
        )


def cmd_architectures(args) -> None:
    for name in ARCHITECTURE_REGISTRY.keys():
        spec = get_architecture(name)
        print(f"{name}: orchestrator={spec.orchestrator}")
        for stage in spec.stages:
            platforms = ",".join(stage.supported_platforms) or "all"
            print(f"  - {stage.name:<14} quantizable={stage.quantizable} platforms=[{platforms}]")


def cmd_info(args) -> None:
    print("compilers:   ", ", ".join(COMPILER_REGISTRY.keys()))
    print("runtimes:    ", ", ".join(RUNTIME_REGISTRY.keys()))
    print("quant methods:", ", ".join(QUANT_METHOD_REGISTRY.keys()))
    print("kernels:     ", ", ".join(f"{op}@{vendor}" for op, vendor in KERNEL_REGISTRY.keys()))


def cmd_infer(args) -> None:
    from chameleon.api import run_infer

    task = _load_task(args)
    actions = run_infer(task)
    print(f"action shape: {tuple(actions.shape)}")
    print(f"action mean:  {actions.float().mean().item():.6f}")
    print(f"action std:   {actions.float().std().item():.6f}")


def cmd_quantize(args) -> None:
    task = _load_task(args)
    task.actions = ["quantize"]
    from chameleon.workflows.runner import WorkflowRunner

    WorkflowRunner(task).run(dry_run=args.dry_run)


def cmd_compile(args) -> None:
    task = _load_task(args)
    task.actions = ["compile"]
    from chameleon.workflows.runner import WorkflowRunner

    WorkflowRunner(task).run(dry_run=args.dry_run)


def cmd_workflow(args) -> None:
    from chameleon.workflows.runner import WorkflowRunner

    task = _load_task(args)
    manifest = WorkflowRunner(task).run(dry_run=args.dry_run)
    if not args.dry_run:
        print(f"manifest: {manifest.path} ({len(manifest.artifacts)} artifacts)")


def cmd_profile(args) -> None:
    from chameleon.profile.latency import profile_infer

    task = _load_task(args)
    result = profile_infer(task, runs=args.runs)
    print(f"runs={result.runs} mean={result.mean_ms:.2f}ms p50={result.p50_ms:.2f}ms p90={result.p90_ms:.2f}ms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chameleon", description="Cross-platform edge VLA toolkit.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("platforms", help="List deployment platforms.").set_defaults(func=cmd_platforms)
    sub.add_parser("architectures", help="List architectures and stages.").set_defaults(func=cmd_architectures)
    sub.add_parser("info", help="List registered backends.").set_defaults(func=cmd_info)

    def _add_config_args(p, *, with_dry_run: bool = False):
        p.add_argument("--config", required=True, help="Path to a task YAML.")
        p.add_argument("--platform", help="Override the task platform.")
        p.add_argument("--runtime", help="Override the runtime for all stages.")
        if with_dry_run:
            p.add_argument("--dry-run", action="store_true", help="Print the plan without running.")

    p_infer = sub.add_parser("infer", help="Run a single inference.")
    _add_config_args(p_infer)
    p_infer.set_defaults(func=cmd_infer)

    p_quant = sub.add_parser("quantize", help="Run the quantize action only.")
    _add_config_args(p_quant, with_dry_run=True)
    p_quant.set_defaults(func=cmd_quantize)

    p_compile = sub.add_parser("compile", help="Run the compile action only.")
    _add_config_args(p_compile, with_dry_run=True)
    p_compile.set_defaults(func=cmd_compile)

    p_wf = sub.add_parser("workflow", help="Run a full task workflow.")
    _add_config_args(p_wf, with_dry_run=True)
    p_wf.set_defaults(func=cmd_workflow)

    p_profile = sub.add_parser("profile", help="Profile inference latency.")
    _add_config_args(p_profile)
    p_profile.add_argument("--runs", type=int, default=20)
    p_profile.set_defaults(func=cmd_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
