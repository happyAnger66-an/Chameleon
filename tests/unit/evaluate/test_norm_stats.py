"""norm_stats 路径解析单元测试。"""

from __future__ import annotations

from pathlib import Path

from chameleon.evaluate.norm_stats import resolve_norm_stats_assets_dir


def test_resolve_norm_stats_expands_tilde(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    assets = home / ".cache" / "openpi" / "assets"
    assets.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    resolved = resolve_norm_stats_assets_dir(
        checkpoint_dir=tmp_path / "ckpt",
        norm_stats_dir="~/.cache/openpi/assets",
    )
    assert resolved == assets.resolve()
