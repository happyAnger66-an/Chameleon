"""eval 子命令 — LeRobot 离线动作评测。"""

from __future__ import annotations

import argparse

from chameleon.api import run_eval
from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task


def eval_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chameleon eval",
        description="Evaluate predicted vs ground-truth actions on a LeRobot dataset.",
    )
    add_global_arguments(parser)
    add_config_arguments(parser)
    parser.add_argument("--num-samples", type=int, default=None, help="Override evaluated frame count.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing model.safetensors (overrides config).",
    )
    parser.add_argument(
        "--policy-runner",
        choices=["openpi", "chameleon"],
        default=None,
        help="Policy backend: openpi (Policy.infer) or chameleon (InferenceSession).",
    )
    parser.add_argument(
        "--viewer",
        choices=["console", "webui", "both"],
        default=None,
        help="Eval result viewer: console log, webui (WebSocket), or both.",
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    if args.num_samples is not None:
        task.evaluate.num_samples = args.num_samples
    if args.checkpoint_dir:
        task.evaluate.checkpoint_dir = args.checkpoint_dir
    if args.policy_runner:
        task.evaluate.policy_runner = args.policy_runner
    if args.viewer:
        task.evaluate.viewer = args.viewer

    from chameleon.evaluate.task_utils import sync_eval_num_samples

    sync_eval_num_samples(task)

    summary = run_eval(task)
    print(summary.describe())
    return 0
