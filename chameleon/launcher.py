"""Chameleon CLI 启动器 — 按子命令分发到各模块 cli handler。

作用：
    参考 model_optimizer.launcher：顶层只做命令路由，各子命令在
    chameleon/commands/ 下维护自己的 ArgumentParser 与业务逻辑。

架构位置：
    入口/编排层 — 被 chameleon.cli.main 调用；不含具体任务实现。
"""

from __future__ import annotations

import sys
from collections.abc import Callable

USAGE = (
    "-" * 68
    + "\n"
    + "| Usage:                                                           |\n"
    + "|   chameleon platforms          List deployment platforms         |\n"
    + "|   chameleon architectures      List architectures and stages     |\n"
    + "|   chameleon info               List registered backends          |\n"
    + "|   chameleon infer   --config <yaml>                            |\n"
    + "|   chameleon eval    --config <yaml>                            |\n"
    + "|   chameleon export  --config <yaml>                            |\n"
    + "|   chameleon quantize --config <yaml>                           |\n"
    + "|   chameleon compile  --config <yaml>                           |\n"
    + "|   chameleon workflow --config <yaml>                           |\n"
    + "|   chameleon profile  --config <yaml>                           |\n"
    + "|   chameleon stats    --config <yaml>                           |\n"
    + "|   chameleon help               Show this message                 |\n"
    + "-" * 68
)


def _dispatch_table() -> dict[str, Callable[[list[str] | None], int]]:
    from chameleon.commands.compile import compile_cli
    from chameleon.commands.eval import eval_cli
    from chameleon.commands.export import export_cli
    from chameleon.commands.info import architectures_cli, info_cli, platforms_cli
    from chameleon.commands.infer import infer_cli
    from chameleon.commands.profile import profile_cli
    from chameleon.commands.quantize import quantize_cli
    from chameleon.commands.stats import stats_cli
    from chameleon.commands.workflow import workflow_cli

    return {
        "platforms": platforms_cli,
        "architectures": architectures_cli,
        "info": info_cli,
        "infer": infer_cli,
        "eval": eval_cli,
        "export": export_cli,
        "quantize": quantize_cli,
        "compile": compile_cli,
        "workflow": workflow_cli,
        "profile": profile_cli,
        "stats": stats_cli,
    }


def launch(argv: list[str] | None = None) -> int:
    """Dispatch ``argv[0]`` to the matching subcommand handler."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"help", "-h", "--help"}:
        print(USAGE)
        return 0

    command, *rest = args
    handler = _dispatch_table().get(command)
    if handler is None:
        print(f"Unknown command: {command!r}\n")
        print(USAGE)
        return 1

    return handler(rest)


def main(argv: list[str] | None = None) -> int:
    return launch(argv)
