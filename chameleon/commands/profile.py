"""profile 子命令 — 推理延迟 profiling。"""

from __future__ import annotations

import argparse

from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task
from chameleon.profile.latency import profile_infer


def profile_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chameleon profile", description="Profile inference latency.")
    add_global_arguments(parser)
    add_config_arguments(parser)
    parser.add_argument("--runs", type=int, default=20, help="Number of timed inference runs.")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    result = profile_infer(task, runs=args.runs)
    print(
        f"runs={result.runs} mean={result.mean_ms:.2f}ms "
        f"p50={result.p50_ms:.2f}ms p90={result.p90_ms:.2f}ms"
    )
    return 0
