"""webui 子命令 — 启动浏览器端配置编排前端（M1）。"""

from __future__ import annotations

import argparse

from chameleon.commands.common import add_global_arguments, configure_logging


def webui_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chameleon webui", description="Launch the browser config WebUI."
    )
    add_global_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8800, help="Bind port (default: 8800).")
    parser.add_argument(
        "--configs-dir", default="configs", help="Directory of task YAMLs (default: configs)."
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    from chameleon.webui.server import run_webui

    run_webui(host=args.host, port=args.port, configs_dir=args.configs_dir)
    return 0
