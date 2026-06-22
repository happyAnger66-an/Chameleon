"""trt-profile 子命令 — 对已构建 engine 运行 trtexec layer profiling。"""

from __future__ import annotations

import argparse

from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task
from chameleon.workflows.runner import WorkflowRunner


def trt_profile_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chameleon trt-profile",
        description="Profile compiled TRT engines with trtexec (--dumpProfile / --exportProfile).",
    )
    add_global_arguments(parser)
    add_config_arguments(parser, with_dry_run=True)
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    task.actions = ["trt_profile"]
    WorkflowRunner(task).run(dry_run=args.dry_run)
    return 0
