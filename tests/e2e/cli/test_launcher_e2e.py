"""CLI launcher E2E。"""

from __future__ import annotations

import pytest


@pytest.mark.e2e
class TestCliLauncher:
    def test_help(self, cli_cmd) -> None:
        result = cli_cmd("help")
        assert result.returncode == 0
        assert "chameleon eval" in result.stdout
        assert "chameleon export" in result.stdout

    def test_info(self, cli_cmd) -> None:
        result = cli_cmd("info")
        assert result.returncode == 0
        assert "tensorrt" in result.stdout.lower() or "pytorch" in result.stdout.lower()

    def test_platforms(self, cli_cmd) -> None:
        result = cli_cmd("platforms")
        assert result.returncode == 0
        assert "generic_cpu" in result.stdout

    def test_architectures(self, cli_cmd) -> None:
        result = cli_cmd("architectures")
        assert result.returncode == 0
        assert "pi05" in result.stdout

    def test_eval_help_has_viewer(self, cli_cmd) -> None:
        result = cli_cmd("eval", "--help")
        assert result.returncode == 0
        assert "--viewer" in result.stdout
        assert "webui" in result.stdout

    def test_unknown_command(self, cli_cmd) -> None:
        result = cli_cmd("not-a-command", check=False)
        assert result.returncode == 1
        assert "Unknown command" in result.stdout
