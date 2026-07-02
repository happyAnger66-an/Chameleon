"""WebUI 后端 — FastAPI 应用与本地服务启动（M1 配置 IO + M2 运行/日志）。

作用：
    提供配置列举 / 读取 / 校验的 REST 接口并托管前端静态资源（M1）；以子进程
    方式运行 ``chameleon <command> --config`` 并通过 WebSocket 流式回传日志与
    状态、支持取消（M2）。进度/产物汇总留待 M3。

架构位置：
    入口/编排层 — 由 ``chameleon.commands.webui`` 启动；依赖 fastapi/uvicorn
    (可选 extra ``[webui]``) 与 ``chameleon.webui.config_store`` / ``runner``。

注意：
    本模块**不使用** ``from __future__ import annotations``。FastAPI 依赖端点函数
    签名上的真实类型对象做依赖注入；若注解被 PEP 563 变成字符串，会用模块全局
    命名空间去解析 ``create_app`` 内的局部类型（如 ``WebSocket`` / 请求模型），从而
    解析失败（表现为请求参数错位、WebSocket 被 1008 拒绝）。
"""

import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_CLIENT_DIR = Path(__file__).resolve().parent / "client"


class ValidateRequest(BaseModel):
    text: str


class RunRequest(BaseModel):
    command: str = "workflow"
    text: str
    interpreter: str | None = None
    config_name: str | None = None


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise RuntimeError(
            "WebUI 需要 fastapi 与 uvicorn，请安装: pip install 'chameleon-vla[webui]'"
        ) from exc


def create_app(configs_dir: str | Path = "configs"):
    """构造 FastAPI 应用（M1：配置 IO + 校验 + 静态前端）。"""
    _require_fastapi()

    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    from chameleon.webui.config_store import ConfigStore, ConfigStoreError
    from chameleon.webui.runner import ALLOWED_COMMANDS, RunError, RunManager

    store = ConfigStore(configs_dir)
    runs = RunManager(project_root=store.root.parent)
    app = FastAPI(title="Chameleon WebUI", version="0.1.0")

    @app.get("/api/configs")
    def list_configs() -> dict:
        return {
            "configs": store.list_configs(),
            "configs_dir": str(store.root),
            "commands": list(ALLOWED_COMMANDS),
        }

    @app.get("/api/configs/{name}")
    def get_config(name: str) -> dict:
        try:
            text = store.read_text(name)
        except ConfigStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        result = store.validate_text(text)
        return {"name": name, "text": text, "actions": result.actions}

    @app.post("/api/validate")
    def validate(req: ValidateRequest) -> dict:
        result = store.validate_text(req.text)
        return {"ok": result.ok, "error": result.error, "actions": result.actions}

    @app.post("/api/run")
    async def start_run(req: RunRequest) -> dict:
        try:
            record = await runs.start(
                command=req.command,
                config_text=req.text,
                interpreter=req.interpreter,
                config_name=req.config_name,
            )
        except RunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return record.snapshot()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"run 不存在: {run_id}")
        snap = record.snapshot()
        snap["lines"] = list(record.lines)
        return snap

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict:
        ok = await runs.cancel(run_id)
        return {"cancelled": ok}

    @app.websocket("/ws/run/{run_id}")
    async def run_ws(ws: WebSocket, run_id: str) -> None:
        await ws.accept()
        record = runs.get(run_id)
        if record is None:
            await ws.send_json({"type": "error", "message": f"run 不存在: {run_id}"})
            await ws.close()
            return
        queue, history, status = runs.subscribe(record)
        try:
            await ws.send_json({"type": "status", **status})
            for line in history:
                await ws.send_json({"type": "log", "line": line})
            # subscribe() 与 _finish() 广播互斥：订阅时若已是终态则不会再有队列消息，
            # 直接结束，避免永久阻塞在 queue.get()。
            if status.get("status") == "running":
                while True:
                    msg = await queue.get()
                    await ws.send_json(msg)
                    if msg.get("type") == "status" and msg.get("status") != "running":
                        break
        except WebSocketDisconnect:
            pass
        finally:
            runs.unsubscribe(record, queue)
            try:
                await ws.close()
            except RuntimeError:
                pass

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_CLIENT_DIR / "index.html")

    if _CLIENT_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_CLIENT_DIR)), name="static")

    return app


def run_webui(
    *,
    host: str = "127.0.0.1",
    port: int = 8800,
    configs_dir: str | Path = "configs",
) -> None:
    """阻塞式启动本地 WebUI 服务。"""
    _require_fastapi()
    import uvicorn

    app = create_app(configs_dir)
    print(f"[webui] Chameleon WebUI: http://{host}:{port}  (configs: {Path(configs_dir).resolve()})")
    uvicorn.run(app, host=host, port=port, log_level="info")
