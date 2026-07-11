"""stream 子命令 — Qwen3-ASR 流式转写 demo（整段重喂 + 前缀回退）。"""

from __future__ import annotations

import argparse
import json
import logging

from chameleon.api import run_stream
from chameleon.commands.common import add_config_arguments, add_global_arguments, configure_logging, load_task

logger = logging.getLogger(__name__)


def stream_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chameleon stream",
        description="Streaming ASR demo (chunk re-feed + prefix rollback).",
    )
    add_global_arguments(parser)
    add_config_arguments(parser)
    parser.add_argument(
        "--audio",
        default=None,
        help="Override asr.audio path (wav/mp3/flac).",
    )
    parser.add_argument(
        "--json-events",
        action="store_true",
        help="Emit one JSON object per chunk (fixed/pending text regions).",
    )
    parser.add_argument(
        "--viewer",
        choices=["console", "webui", "both"],
        default=None,
        help="Override evaluate.viewer for stream text UI.",
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    task = load_task(args)
    if args.audio:
        task.asr.audio = args.audio
    if args.viewer:
        task.evaluate.viewer = args.viewer

    viewer = (task.evaluate.viewer or "console").strip().lower()
    webui = None
    if viewer in ("webui", "both"):
        from chameleon.runtime.edgellm.webui_text import AsrStreamWebUI

        webui = AsrStreamWebUI(
            host=task.evaluate.webui_host or "127.0.0.1",
            port=int(task.evaluate.webui_port or 8768),
            path=task.evaluate.webui_path or "/ws",
        )
        webui.start()
        print(
            f"[stream] WebUI WS: {webui.ws_url}\n"
            f"[stream] Open: chameleon/evaluate/webui_client/asr_stream.html?ws={webui.ws_url}",
            flush=True,
        )

    def on_update(evt: dict) -> None:
        if webui is not None:
            webui.emit(evt)
        fixed = evt.get("fixed_text") or ""
        pending = evt.get("pending_text") or ""
        if args.json_events:
            print(json.dumps(evt, ensure_ascii=False), flush=True)
            return
        if viewer == "webui":
            return
        lang = evt.get("language") or ""
        cid = evt.get("chunk_id", "?")
        print(f"[chunk {cid}] lang={lang}", flush=True)
        print(f"  fixed:   {fixed}", flush=True)
        print(f"  pending: {pending}", flush=True)
        print(f"  full:    {fixed}{pending}", flush=True)

    final = run_stream(task, on_update=on_update)
    if webui is not None:
        webui.emit({"event": "final", **final})

    if args.json_events:
        print(json.dumps({"event": "final", **final}, ensure_ascii=False), flush=True)
    elif viewer != "webui":
        print("---", flush=True)
        print(f"final language: {final.get('language') or ''}", flush=True)
        print(f"final text:     {final.get('text') or ''}", flush=True)
    return 0
