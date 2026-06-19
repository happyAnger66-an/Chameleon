"""infer 子命令 — 单次推理冒烟。"""

from __future__ import annotations

import argparse

from chameleon.api import run_infer
from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task


def infer_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chameleon infer", description="Run a single inference.")
    add_global_arguments(parser)
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    actions = run_infer(task)
    print(f"action shape: {tuple(actions.shape)}")
    print(f"action mean:  {actions.float().mean().item():.6f}")
    print(f"action std:   {actions.float().std().item():.6f}")
    return 0
