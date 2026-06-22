"""draw 子命令 — TRT layer profile 浏览器可视化。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from chameleon.commands.common import add_global_arguments, configure_logging
from chameleon.config.schema import TaskConfig
from chameleon.deploy.paths import resolve_profile_dir
from chameleon.draw.trt_profile_viewer import (
    build_multi_stage_dashboard,
    build_profile_html,
    build_stage_profile_html,
    load_profile_bundles_from_dir,
    load_trtexec_profile_rows,
    pick_free_port,
    serve_profile_html,
)


def draw_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chameleon draw",
        description="Visual tools (trtexec layer profile in the browser).",
    )
    sub = parser.add_subparsers(dest="sub", required=True)

    p_prof = sub.add_parser(
        "profile",
        help="Open trtexec --exportProfile JSON in a local web table (sort / filter).",
    )
    p_prof.add_argument(
        "json_path",
        nargs="?",
        type=str,
        help="Path to a single trtexec profile JSON, or omit when using --config.",
    )
    add_global_arguments(p_prof)
    p_prof.add_argument(
        "--config",
        help="Task YAML — open multi-stage dashboard or a single stage with --stage.",
    )
    p_prof.add_argument(
        "--stage",
        help="When used with --config, open one stage only (e.g. llm, vit).",
    )
    p_prof.add_argument(
        "--use-cache",
        action="store_true",
        help="With --config, serve existing profiles/index.html instead of rebuilding.",
    )
    p_prof.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    p_prof.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port (0 = pick a free port; config uses profile.webui_port when set).",
    )
    p_prof.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab automatically.",
    )

    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    if args.sub != "profile":
        parser.print_help()
        return 2

    if args.config:
        return _draw_from_config(args)
    if not args.json_path:
        parser.error("json_path or --config is required.")
    return _draw_single_file(args)


def _draw_single_file(args: argparse.Namespace) -> int:
    path = os.path.abspath(os.path.expanduser(args.json_path))
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return 1

    rows, iteration_count = load_trtexec_profile_rows(path)
    title = os.path.basename(path)
    html = build_profile_html(rows, title, iteration_count)
    port = pick_free_port(args.host, args.port)
    serve_profile_html(html, host=args.host, port=port, open_browser=not args.no_browser)
    return 0


def _draw_from_config(args: argparse.Namespace) -> int:
    task = TaskConfig.load(args.config)
    profile_dir = resolve_profile_dir(task)
    index_path = profile_dir / "index.html"

    if args.use_cache and index_path.is_file() and not args.stage:
        html = index_path.read_text(encoding="utf-8")
    else:
        try:
            bundles = load_profile_bundles_from_dir(profile_dir)
        except FileNotFoundError as exc:
            print(
                f"{exc}. Run `chameleon trt-profile --config {args.config}` first."
            )
            return 1

        if args.stage:
            stage = args.stage.strip()
            bundle = bundles.get(stage)
            if bundle is None:
                print(f"Unknown stage {stage!r}; available: {', '.join(bundles)}")
                return 1
            html = build_stage_profile_html(bundle)
        else:
            html = build_multi_stage_dashboard(bundles)
            index_path.write_text(html, encoding="utf-8")

    preferred = args.port or task.profile.webui_port
    host = args.host or task.profile.webui_host
    port = pick_free_port(host, preferred)
    serve_profile_html(html, host=host, port=port, open_browser=not args.no_browser)
    return 0
