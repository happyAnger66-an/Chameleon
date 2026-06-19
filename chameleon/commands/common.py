"""CLI 共用工具 — 日志、TaskConfig 加载与通用参数。"""

from __future__ import annotations

import argparse
import logging

from chameleon.architectures.registry import get_architecture
from chameleon.config.schema import TaskConfig


def configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def add_global_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logging.")


def add_config_arguments(parser: argparse.ArgumentParser, *, with_dry_run: bool = False) -> None:
    parser.add_argument("--config", required=True, help="Path to a task YAML.")
    parser.add_argument("--platform", help="Override the task platform.")
    parser.add_argument("--runtime", help="Override the runtime for all stages.")
    if with_dry_run:
        parser.add_argument("--dry-run", action="store_true", help="Print the plan without running.")


def load_task(args: argparse.Namespace) -> TaskConfig:
    task = TaskConfig.load(args.config)
    if getattr(args, "platform", None):
        task.platform = args.platform
    if getattr(args, "runtime", None):
        task.stage_runtimes = {
            stage: args.runtime for stage in get_architecture(task.architecture).stage_names
        }
    return task
