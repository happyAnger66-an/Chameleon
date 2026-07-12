"""bench 子命令 — pi05 TRT/TVM stage 延迟对比。"""

from __future__ import annotations

import argparse

from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task
from chameleon.profile.bench import run_bench


def bench_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chameleon bench",
        description="Stage-level latency bench (pi05 TRT vs TVM).",
    )
    add_global_arguments(parser)
    add_config_arguments(parser)
    parser.add_argument("--runs", type=int, default=None, help="Override bench.runs.")
    parser.add_argument("--warmup", type=int, default=None, help="Override bench.warmup.")
    parser.add_argument(
        "--backends",
        default=None,
        help="Comma-separated backends (trt,tvm). Overrides bench.backends.",
    )
    parser.add_argument("--output", default=None, help="Override bench.output JSON path.")
    parser.add_argument(
        "--tvm-loop",
        choices=["true", "false"],
        default=None,
        help="Override bench.tvm_loop (false = stepwise denoise for fair TRT compare).",
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    if args.runs is not None:
        task.bench.runs = int(args.runs)
    if args.warmup is not None:
        task.bench.warmup = int(args.warmup)
    if args.backends:
        task.bench.backends = [x.strip() for x in args.backends.split(",") if x.strip()]
    if args.output:
        task.bench.output = args.output
    if args.tvm_loop is not None:
        task.bench.tvm_loop = args.tvm_loop == "true"

    report = run_bench(task)
    print(report.format_table(stages=task.bench.stages))
    print()
    if report.delta:
        print("delta (tvm_p50 - trt_p50) ms:")
        for k, v in report.delta.items():
            print(f"  {k}: {v:+.2f}")
    out = report.meta.get("output")
    if out:
        print(f"\nJSON: {out}")
    return 0
