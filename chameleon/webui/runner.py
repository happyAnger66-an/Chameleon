"""运行管理器 — 以子进程方式执行 ``chameleon <command> --config`` 并流式回传日志（M2）。

作用：
    把编辑后的 YAML 落盘为一次运行的配置，spawn 一个独立 Python 进程运行 CLI，
    逐行采集 stdout/stderr 并广播给（多个）WebSocket 订阅者；维护运行状态与
    历史日志，支持取消。子进程隔离了 torch/tensorrt 等重依赖，并允许切换到不同
    的解释器（如 openpi venv）。

架构位置：
    编排层支撑模块 — 被 ``chameleon.webui.server`` 使用；仅依赖标准库 asyncio。
    真正的任务逻辑仍在被调起的 ``chameleon.cli`` 进程里执行。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# 允许通过 WebUI 触发的 CLI 子命令白名单（防止 command 字段注入任意子命令）。
ALLOWED_COMMANDS = (
    "workflow",
    "infer",
    "eval",
    "export",
    "quantize",
    "compile",
    "trt-profile",
    "profile",
    "stats",
)

# 运行状态取值。
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


class RunError(Exception):
    """运行请求非法（命令不在白名单、已有运行在跑等）。"""


@dataclass
class RunRecord:
    run_id: str
    command: str
    config_path: Path
    interpreter: str
    status: str = STATUS_RUNNING
    returncode: int | None = None
    created_at: float = field(default_factory=time.time)
    lines: list[str] = field(default_factory=list)
    proc: asyncio.subprocess.Process | None = None
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "command": self.command,
            "config_path": str(self.config_path),
            "interpreter": self.interpreter,
            "status": self.status,
            "returncode": self.returncode,
            "created_at": self.created_at,
        }


class RunManager:
    """管理 WebUI 触发的运行（MVP：同一时刻仅允许一个活跃运行）。"""

    def __init__(self, project_root: str | Path, run_base: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.run_base = (
            Path(run_base).resolve()
            if run_base is not None
            else self.project_root / "output" / "webui_runs"
        )
        self._runs: dict[str, RunRecord] = {}
        self._active: str | None = None

    # ---- 查询 ---------------------------------------------------------------
    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def active_run_id(self) -> str | None:
        rid = self._active
        if rid is None:
            return None
        rec = self._runs.get(rid)
        if rec is None or rec.status != STATUS_RUNNING:
            return None
        return rid

    # ---- 订阅（同步、无 await，保证与 _pump 广播原子）------------------------
    def subscribe(self, record: RunRecord) -> tuple[asyncio.Queue, list[str], dict]:
        queue: asyncio.Queue = asyncio.Queue()
        history = list(record.lines)
        status = record.snapshot()
        record.subscribers.add(queue)
        return queue, history, status

    def unsubscribe(self, record: RunRecord, queue: asyncio.Queue) -> None:
        record.subscribers.discard(queue)

    # ---- 启动 ---------------------------------------------------------------
    async def start(
        self,
        *,
        command: str,
        config_text: str,
        interpreter: str | None = None,
        config_name: str | None = None,
        verbose: bool = True,
    ) -> RunRecord:
        command = (command or "").strip()
        if command not in ALLOWED_COMMANDS:
            raise RunError(
                f"不支持的命令 {command!r}；可选: {', '.join(ALLOWED_COMMANDS)}"
            )
        if self.active_run_id() is not None:
            raise RunError("已有运行进行中，请先取消或等待其结束。")

        interp = (interpreter or "").strip() or sys.executable
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.run_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / (config_name or "config.yaml")
        config_path.write_text(config_text, encoding="utf-8")

        record = RunRecord(
            run_id=run_id, command=command, config_path=config_path, interpreter=interp
        )
        self._runs[run_id] = record
        self._active = run_id

        argv = [interp, "-m", "chameleon.cli", command, "--config", str(config_path)]
        if verbose:
            argv.append("-v")

        env = os.environ.copy()
        root = str(self.project_root)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=root,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            self._emit(record, f"[webui] 启动进程失败: {exc}")
            self._finish(record, STATUS_FAILED, returncode=None)
            return record

        record.proc = proc
        self._emit(record, f"[webui] $ {' '.join(argv)}")
        asyncio.create_task(self._pump(record))
        return record

    # ---- 取消 ---------------------------------------------------------------
    async def cancel(self, run_id: str) -> bool:
        record = self._runs.get(run_id)
        if record is None or record.status != STATUS_RUNNING or record.proc is None:
            return False
        proc = record.proc
        record.status = STATUS_CANCELLED  # 标记，避免 _pump 覆盖为 done/failed
        try:
            proc.terminate()
        except ProcessLookupError:
            return True
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        return True

    # ---- 内部 ---------------------------------------------------------------
    async def _pump(self, record: RunRecord) -> None:
        proc = record.proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                text = raw.decode("utf-8", errors="replace").rstrip("\n")
                self._emit(record, text)
            returncode = await proc.wait()
        except Exception as exc:  # noqa: BLE001 — 保证 pump 不静默挂死
            self._emit(record, f"[webui] 读取输出异常: {exc}")
            returncode = record.returncode

        if record.status == STATUS_CANCELLED:
            self._emit(record, "[webui] 运行已取消。")
            self._finish(record, STATUS_CANCELLED, returncode=returncode)
        else:
            status = STATUS_DONE if returncode == 0 else STATUS_FAILED
            self._emit(record, f"[webui] 进程结束，returncode={returncode}")
            self._finish(record, status, returncode=returncode)

    def _emit(self, record: RunRecord, line: str) -> None:
        """追加一行并广播（同步、无 await：与 subscribe 快照互斥）。"""
        record.lines.append(line)
        msg = {"type": "log", "line": line}
        for q in list(record.subscribers):
            q.put_nowait(msg)

    def _finish(self, record: RunRecord, status: str, *, returncode: int | None) -> None:
        record.status = status
        record.returncode = returncode
        if self._active == record.run_id:
            self._active = None
        msg = {"type": "status", **record.snapshot()}
        for q in list(record.subscribers):
            q.put_nowait(msg)
        record.done.set()
