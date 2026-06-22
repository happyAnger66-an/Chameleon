"""TRT profile viewer 单元测试。"""

from __future__ import annotations

from pathlib import Path

from chameleon.draw.trt_profile_viewer import (
    ProfileBundle,
    build_multi_stage_dashboard,
    build_profile_html,
    build_stage_profile_html,
    load_trtexec_profile_rows,
)


def test_load_trtexec_profile_rows(fixtures_dir: Path) -> None:
    path = fixtures_dir / "sample_trt_profile.json"
    rows, count = load_trtexec_profile_rows(str(path))
    assert count == 20
    assert len(rows) == 3
    assert rows[0]["name"] == "MatMul_0"
    assert rows[0]["percentage"] == 45.2


def test_build_profile_html_contains_table() -> None:
    rows = [{"name": "A", "timeMs": 1.0, "averageMs": 0.1, "medianMs": 0.1, "percentage": 100.0}]
    html = build_profile_html(rows, "test.json", 10)
    assert "Layer timing" in html
    assert 'id="q"' in html
    assert "DATA" in html


def test_build_stage_profile_html_uses_filter_controls() -> None:
    bundle = ProfileBundle(
        stage="llm",
        rows=[{"name": "MatMul_0", "timeMs": 1.0, "averageMs": 0.1, "medianMs": 0.1, "percentage": 100.0}],
        iteration_count=10,
        profile_path="llm.profile.json",
    )
    html = build_stage_profile_html(bundle)
    assert 'id="q"' in html
    assert "Filter name" in html
    assert "nameMatch" in html


def test_build_multi_stage_dashboard_lists_stages() -> None:
    bundle = ProfileBundle(
        stage="vit",
        rows=[{"name": "L", "timeMs": 1.0, "averageMs": 0.1, "medianMs": 0.1, "percentage": 100.0}],
        iteration_count=5,
        profile_path="vit.profile.json",
    )
    html = build_multi_stage_dashboard(
        {
            "vit": bundle,
            "llm": ProfileBundle(stage="llm", rows=bundle.rows, iteration_count=5),
            "expert": ProfileBundle(stage="expert", rows=bundle.rows, iteration_count=5),
            "denoise": ProfileBundle(stage="denoise", rows=bundle.rows, iteration_count=5),
        }
    )
    for stage in ("vit", "llm", "expert", "denoise"):
        assert stage in html
    assert "Filter name" in html
    assert "setActiveStage" in html
    assert "nameMatch" in html
