"""export 子命令 — 仅执行 ONNX 导出（pi05_openpi）。"""

from __future__ import annotations

import argparse

from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task
from chameleon.workflows.runner import WorkflowRunner


def export_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chameleon export",
        description="Export ONNX for deployment (Chameleon pi05 exporters).",
    )
    add_global_arguments(parser)
    add_config_arguments(parser, with_dry_run=True)
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    task.actions = ["export"]
    WorkflowRunner(task).run(dry_run=args.dry_run)
    return 0
