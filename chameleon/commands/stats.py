"""stats 子命令 — 模型计算量与访存量统计。"""

from __future__ import annotations

import argparse
import sys

from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task
from chameleon.profile.compute_stats import format_stats_table, stats_infer, stats_result_to_dict, write_stats_json


def stats_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chameleon stats",
        description="Estimate compute (MACs/FLOPs) and memory traffic for a full inference.",
    )
    add_global_arguments(parser)
    add_config_arguments(parser, with_dry_run=True)
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--output",
        help="Write JSON output to this path (implies --format json when set).",
    )
    parser.add_argument(
        "--measured",
        action="store_true",
        help="Also run torch.profiler on CUDA for validation (requires GPU).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-stage progress on stderr.",
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    if args.dry_run:
        from chameleon.profile.execution_plan import build_execution_plan
        from chameleon.profile.shape_resolver import resolve_precision

        plan = build_execution_plan(task)
        precision = resolve_precision(task)
        print(f"dry-run: plan={plan.describe()} batch={plan.batch_size} precision={precision}")
        return 0

    try:
        result = stats_infer(task, measured=args.measured, progress=not args.quiet)
    except RuntimeError as exc:
        print(f"stats failed: {exc}", file=sys.stderr)
        return 1

    out_format = "json" if args.output else args.format
    if out_format == "json":
        payload = stats_result_to_dict(result)
        text = __import__("json").dumps(payload, indent=2)
        if args.output:
            write_stats_json(result, args.output)
            print(f"Wrote {args.output}")
        else:
            print(text)
    else:
        print(format_stats_table(result))

    for w in result.warnings:
        print(f"WARN: {w}", file=sys.stderr)

    return 0
