"""WebUI 运行层 E2E — 子进程运行 + WebSocket 流式日志 + 取消 / 并发保护。

这些用例会真实 spawn ``python -m chameleon.cli`` 子进程（infer 会加载参考模型），
因此标记为 ``e2e_slow``。
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from chameleon.webui.server import create_app  # noqa: E402


@pytest.fixture
def client(configs_dir):
    app = create_app(configs_dir=str(configs_dir))
    with TestClient(app) as c:
        yield c


def _cpu_config_text(configs_dir) -> str:
    return (configs_dir / "pi05_cpu.yaml").read_text(encoding="utf-8")


def _drain_ws(client, run_id, *, max_messages: int = 5000):
    """连上 run 的 WS，收完所有消息直到服务端关闭，返回 (日志行, 终态状态)。"""
    logs: list[str] = []
    final: dict | None = None
    with client.websocket_connect(f"/ws/run/{run_id}") as wsc:
        for _ in range(max_messages):
            try:
                msg = wsc.receive_json()
            except WebSocketDisconnect:
                break
            if msg.get("type") == "log":
                logs.append(msg["line"])
            elif msg.get("type") == "status":
                final = msg
    return logs, final


@pytest.mark.e2e
@pytest.mark.e2e_slow
class TestWebUIRunE2E:
    def test_infer_run_streams_and_completes(self, client, configs_dir) -> None:
        snap = client.post(
            "/api/run",
            json={
                "command": "infer",
                "text": _cpu_config_text(configs_dir),
                "config_name": "pi05_cpu.yaml",
            },
        ).json()
        run_id = snap["run_id"]
        assert snap["status"] == "running"

        logs, final = _drain_ws(client, run_id)
        assert final is not None, "should receive a terminal status"
        assert final["status"] == "done", f"logs=\n" + "\n".join(logs)
        assert final["returncode"] == 0
        # 第一行是 webui 打印的启动命令行。
        assert any("chameleon.cli" in ln for ln in logs)

        # 结束后 REST 快照仍可查，且携带完整日志历史。
        got = client.get(f"/api/runs/{run_id}").json()
        assert got["status"] == "done"
        assert len(got["lines"]) == len(logs)

    def test_concurrent_run_rejected_then_cancel(self, client, configs_dir) -> None:
        first = client.post(
            "/api/run",
            json={"command": "infer", "text": _cpu_config_text(configs_dir)},
        ).json()
        run_id = first["run_id"]
        assert first["status"] == "running"

        # 单活跃运行保护：第二次立即请求应被拒。
        second = client.post(
            "/api/run",
            json={"command": "infer", "text": _cpu_config_text(configs_dir)},
        )
        assert second.status_code == 409

        # 取消首个运行并等待其进入终态。
        assert client.post(f"/api/runs/{run_id}/cancel").json()["cancelled"] is True

        deadline = time.time() + 60
        status = "running"
        while time.time() < deadline:
            status = client.get(f"/api/runs/{run_id}").json()["status"]
            if status != "running":
                break
            time.sleep(0.2)
        assert status == "cancelled"

        # 取消后应能再次发起新的运行（活跃锁已释放）。
        third = client.post(
            "/api/run",
            json={"command": "infer", "text": _cpu_config_text(configs_dir)},
        )
        assert third.status_code == 200
        client.post(f"/api/runs/{third.json()['run_id']}/cancel")
