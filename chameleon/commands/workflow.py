"""workflow 子命令 — 完整 quantize / compile / infer 编排。"""

from __future__ import annotations

import argparse

from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task
from chameleon.workflows.runner import WorkflowRunner


def workflow_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chameleon workflow", description="Run a full task workflow.")
    add_global_arguments(parser)
    add_config_arguments(parser, with_dry_run=True)
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    manifest = WorkflowRunner(task).run(dry_run=args.dry_run)
    if not args.dry_run:
        print(f"manifest: {manifest.path} ({len(manifest.artifacts)} artifacts)")
    return 0
