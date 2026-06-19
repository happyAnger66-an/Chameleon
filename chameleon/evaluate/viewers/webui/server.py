"""WebUI WebSocket 服务与 eval 编排。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chameleon.config.schema import EvaluateConfig, TaskConfig
from chameleon.evaluate.lerobot_eval import EvalSummary
from chameleon.evaluate.viewers.base import EvalStepEvent, build_eval_viewer
from chameleon.evaluate.viewers.webui.bridge import AsyncOutboundBridge
from chameleon.evaluate.viewers.webui.broadcaster import WebsocketBroadcaster
from chameleon.evaluate.viewers.webui.media import encode_observation_images
from chameleon.evaluate.viewers.webui.protocol import StepEvent, step_event_to_json

logger = logging.getLogger(__name__)

_ACTIVE_BRIDGE: AsyncOutboundBridge | None = None
_BRIDGE_LOCK = threading.Lock()


def get_active_webui_bridge() -> AsyncOutboundBridge | None:
    with _BRIDGE_LOCK:
        return _ACTIVE_BRIDGE


def _set_active_webui_bridge(bridge: AsyncOutboundBridge | None) -> None:
    global _ACTIVE_BRIDGE
    with _BRIDGE_LOCK:
        _ACTIVE_BRIDGE = bridge


def _client_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "webui_client"


def default_client_ws_url(cfg: EvaluateConfig) -> str:
    host = cfg.webui_host.strip()
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    path = cfg.webui_path if cfg.webui_path.startswith("/") else f"/{cfg.webui_path}"
    return f"ws://{host}:{cfg.webui_port}{path}"


def write_webui_server_hint(cfg: EvaluateConfig) -> str:
    url = default_client_ws_url(cfg)
    client_dir = _client_dir()
    client_dir.mkdir(parents=True, exist_ok=True)
    hint_path = client_dir / "server_hint.json"
    hint_path.write_text(
        json.dumps({"default_ws_url": url}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return url


def _handshake_path(ws: Any) -> str | None:
    p = getattr(ws, "path", None)
    if isinstance(p, str):
        return p
    req = getattr(ws, "request", None)
    if req is not None:
        rp = getattr(req, "path", None)
        if isinstance(rp, str):
            return rp
    return None


def _paths_equivalent(a: str, b: str) -> bool:
    aa = a.rstrip("/") or "/"
    bb = b.rstrip("/") or "/"
    return aa == bb


@dataclass
class WebUIServerRuntime:
    cfg: EvaluateConfig
    run_id: str
    broadcaster: WebsocketBroadcaster
    bridge: AsyncOutboundBridge
    meta_ready: dict[str, Any]
    jpeg_quality: int
    send_wrist: bool


def _build_runtime(loop: asyncio.AbstractEventLoop, cfg: EvaluateConfig) -> WebUIServerRuntime:
    run_id = uuid.uuid4().hex[:12]
    broadcaster = WebsocketBroadcaster(history_size=cfg.webui_history_size)
    qmax = max(0, int(cfg.webui_queue_maxsize))
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=qmax if qmax > 0 else 0)
    bridge = AsyncOutboundBridge(loop, queue, broadcaster)
    return WebUIServerRuntime(
        cfg=cfg,
        run_id=run_id,
        broadcaster=broadcaster,
        bridge=bridge,
        meta_ready={"msg": None},
        jpeg_quality=int(cfg.webui_jpeg_quality),
        send_wrist=bool(cfg.webui_show_wrist),
    )


def _encode_step_event(rt: WebUIServerRuntime, event: EvalStepEvent) -> str:
    images = None
    if event.is_chunk_start and event.observation is not None:
        images = encode_observation_images(
            event.observation,
            send_wrist=rt.send_wrist,
            jpeg_quality=rt.jpeg_quality,
        )
    timing = None
    if event.infer_ms is not None and event.k_in_chunk == 0:
        timing = {"infer_ms": float(event.infer_ms)}
    step = StepEvent(
        type="step",
        run_id=event.run_id,
        episode_id=int(event.episode_id),
        global_index=int(event.global_index),
        k_in_chunk=int(event.k_in_chunk),
        is_chunk_start=bool(event.is_chunk_start),
        action_horizon=int(event.action_horizon),
        prompt=event.prompt,
        gt_action=list(event.gt_action),
        pred_action=list(event.pred_action),
        metrics=dict(event.metrics),
        images=images,
        server_timing=timing,
    )
    return step_event_to_json(step)


def _infer_worker(task: TaskConfig, rt: WebUIServerRuntime, result: dict[str, Any]) -> None:
    from chameleon.dataloader import build_dataset_from_config
    from chameleon.evaluate import evaluate_lerobot
    from chameleon.evaluate.runner_base import build_policy_runner

    _set_active_webui_bridge(rt.bridge)
    try:
        data_cfg = task.data
        data_source = build_dataset_from_config(data_cfg)
        data_source.build()
        runner = build_policy_runner(task)

        repo_id = data_cfg.repo_id or getattr(data_source, "repo_id", "") or ""
        action_horizon = int(getattr(data_source, "action_horizon", 10) or 10)
        action_dim = int(getattr(data_source, "action_dim", 7) or 7)
        start_index = int(getattr(data_cfg, "start_index", 0) or 0)
        num_samples = int(task.evaluate.num_samples)

        meta = {
            "type": "meta",
            "run_id": rt.run_id,
            "repo_id": repo_id,
            "backend": task.evaluate.policy_runner,
            "compare_mode": False,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "start_index": start_index,
            "end_index_exclusive": start_index + num_samples,
            "send_wrist": rt.send_wrist,
            "jpeg_quality": rt.jpeg_quality,
        }
        rt.meta_ready["msg"] = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))

        sink = build_eval_viewer(
            task,
            run_id=rt.run_id,
            repo_id=repo_id,
            action_horizon=action_horizon,
            action_dim=action_dim,
            num_samples=num_samples,
            start_index=start_index,
        )
        summary = evaluate_lerobot(
            data_source,
            runner,
            num_samples=num_samples,
            stride=task.evaluate.stride,
            compare_horizon=task.evaluate.compare_horizon,
            event_sink=sink,
            run_meta=meta,
            run_id=rt.run_id,
            log_every=0 if task.evaluate.viewer in ("webui", "both") else 10,
        )
        result["summary"] = summary
    except Exception as exc:
        result["error"] = exc
        logger.exception("[webui] eval worker failed")
    finally:
        rt.bridge.sync_close()
        _set_active_webui_bridge(None)


async def _run_server(task: TaskConfig) -> EvalSummary:
    try:
        import websockets.asyncio.server as ws_server
        import websockets.exceptions as wsex
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "WebUI 需要 websockets，请安装: pip install 'chameleon-vla[eval-ui]'"
        ) from exc

    cfg = task.evaluate
    loop = asyncio.get_running_loop()
    rt = _build_runtime(loop, cfg)
    hint_url = write_webui_server_hint(cfg)
    client_dir = _client_dir()

    result: dict[str, Any] = {}
    infer_thread = threading.Thread(
        target=_infer_worker,
        args=(task, rt, result),
        daemon=True,
        name="chameleon_eval",
    )

    loading_msg = json.dumps(
        {
            "type": "meta",
            "phase": "loading",
            "message": "Chameleon eval 正在加载模型与数据集…",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    async def handler(ws: Any) -> None:
        req_path = _handshake_path(ws)
        expected = cfg.webui_path if cfg.webui_path.startswith("/") else f"/{cfg.webui_path}"
        if req_path is None or not _paths_equivalent(req_path, expected):
            await ws.close(code=1008, reason="invalid path")
            return
        await rt.broadcaster.register(ws)
        try:
            if rt.meta_ready["msg"] is None:
                await rt.broadcaster.send_to(ws, loading_msg)
            else:
                await rt.broadcaster.send_to(ws, rt.meta_ready["msg"])
                if cfg.webui_history_size > 0:
                    await rt.broadcaster.send_history(ws)
            async for _raw in ws:
                pass
        except (wsex.ConnectionClosedOK, wsex.ConnectionClosedError):
            pass
        finally:
            await rt.broadcaster.unregister(ws)

    encode_fn = lambda ev: _encode_step_event(rt, ev)
    pump_task = asyncio.create_task(rt.bridge.drain(encode_fn), name="webui_pump")

    logger.info(
        "[webui] ws=%s  static=%s  (python -m http.server 8080 -d %s)",
        hint_url,
        client_dir,
        client_dir,
    )
    print(f"[webui] WebSocket: {hint_url}")
    print(f"[webui] 浏览器打开: cd {client_dir} && python -m http.server 8080")

    infer_thread.start()
    async with ws_server.serve(
        handler,
        cfg.webui_host,
        cfg.webui_port,
        compression=None,
        max_size=None,
    ):
        await pump_task

    infer_thread.join(timeout=5.0)
    if "error" in result:
        raise result["error"]
    if "summary" not in result:
        raise RuntimeError("评测线程未返回 summary")
    return result["summary"]


def run_eval_webui(task: TaskConfig) -> EvalSummary:
    return asyncio.run(_run_server(task))
