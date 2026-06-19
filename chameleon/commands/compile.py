"""compile 子命令 — 仅执行编译 workflow 步骤。"""

from __future__ import annotations

import argparse

from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task
from chameleon.workflows.runner import WorkflowRunner


def compile_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chameleon compile", description="Run the compile action only.")
    add_global_arguments(parser)
    add_config_arguments(parser, with_dry_run=True)
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    task.actions = ["compile"]
    WorkflowRunner(task).run(dry_run=args.dry_run)
    return 0
