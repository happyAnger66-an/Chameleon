"""WebUI API 层 E2E — 配置 IO / 校验 / 静态前端（快，走 FastAPI TestClient）。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # starlette TestClient 依赖

from fastapi.testclient import TestClient  # noqa: E402

from chameleon.webui.server import create_app  # noqa: E402


@pytest.fixture
def client(configs_dir):
    app = create_app(configs_dir=str(configs_dir))
    with TestClient(app) as c:
        yield c


@pytest.mark.e2e
class TestWebUIApiE2E:
    def test_index_served(self, client) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Chameleon" in resp.text

    def test_static_assets(self, client) -> None:
        for name in ("app.js", "style.css"):
            resp = client.get(f"/static/{name}")
            assert resp.status_code == 200

    def test_list_configs(self, client) -> None:
        data = client.get("/api/configs").json()
        assert "pi05_cpu.yaml" in data["configs"]
        assert "workflow" in data["commands"]
        assert "infer" in data["commands"]

    def test_get_config(self, client) -> None:
        data = client.get("/api/configs/pi05_cpu.yaml").json()
        assert data["name"] == "pi05_cpu.yaml"
        assert "architecture: pi05" in data["text"]
        assert data["actions"] == ["infer"]

    def test_path_traversal_blocked(self, client) -> None:
        resp = client.get("/api/configs/..%2f..%2fpyproject.toml")
        assert resp.status_code == 404

    def test_missing_config(self, client) -> None:
        resp = client.get("/api/configs/does_not_exist.yaml")
        assert resp.status_code == 404

    def test_validate_ok(self, client) -> None:
        text = (
            "architecture: pi05\nmodel: pi05\nplatform: generic_cpu\n"
            "output_dir: output/x\nactions: [infer]\n"
            "infer: {batch_size: 1, num_steps: 2}\n"
        )
        data = client.post("/api/validate", json={"text": text}).json()
        assert data["ok"] is True
        assert data["error"] is None
        assert data["actions"] == ["infer"]

    def test_validate_bad_yaml(self, client) -> None:
        data = client.post("/api/validate", json={"text": "a: [b: c"}).json()
        assert data["ok"] is False
        assert "YAML" in (data["error"] or "")

    def test_run_bad_command_rejected(self, client) -> None:
        resp = client.post(
            "/api/run",
            json={"command": "rm-rf", "text": "actions: [infer]"},
        )
        assert resp.status_code == 409

    def test_get_missing_run(self, client) -> None:
        assert client.get("/api/runs/nope").status_code == 404
