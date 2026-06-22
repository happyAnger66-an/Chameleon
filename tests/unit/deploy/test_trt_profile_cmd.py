"""trtexec profile 命令构建单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chameleon.config.schema import CompileStep, TaskConfig, TrtProfileStep
from chameleon.deploy.build_cfg import load_build_cfg
from chameleon.deploy.paths import resolve_deploy_paths, stage_engine_path, stage_profile_path
from chameleon.deploy.trt_profile import (
    _hint_from_trtexec_log,
    build_trtexec_profile_command,
    format_shapes_for_trtexec,
    iter_profile_steps,
    profile_engine_stage,
    resolve_plugin_lib_paths,
)


def test_format_shapes_for_trtexec() -> None:
    assert format_shapes_for_trtexec({"pixel_values": (1, 3, 224, 224)}) == "pixel_values:1x3x224x224"
    assert format_shapes_for_trtexec({}) is None


def test_build_trtexec_profile_command_uses_build_cfg_shapes(
    task_deploy_yaml: Path,
    build_configs_dir: Path,
) -> None:
    task = TaskConfig.load(task_deploy_yaml)
    task.profile.iterations = 50
    task.profile.warmup = 200
    build_cfg = load_build_cfg(build_configs_dir / "vit_build_cfg.py")
    paths = resolve_deploy_paths(task)
    engine = stage_engine_path(paths, "vit")
    profile = stage_profile_path(Path(task.profile.profile_dir or f"{task.output_dir}/profiles"), "vit")

    cmd = build_trtexec_profile_command(
        engine_path=engine,
        profile_path=profile,
        build_cfg=build_cfg,
        iterations=50,
        warmup=200,
        separate_profile_run=True,
        profiling_verbosity="detailed",
        export_layer_info=False,
        layer_info_path=None,
        export_times=False,
        times_path=None,
        plugin_lib_paths=["/tmp/libplugin.so"],
        trtexec_path="/usr/bin/trtexec",
    )

    assert cmd[0] == "/usr/bin/trtexec"
    assert f"--loadEngine={engine}" in cmd
    assert "--dumpProfile" in cmd
    assert "--separateProfileRun" in cmd
    assert f"--exportProfile={profile}" in cmd
    assert "--minShapes=pixel_values:1x3x224x224" in cmd
    assert "--optShapes=pixel_values:3x3x224x224" in cmd
    assert "--maxShapes=pixel_values:3x3x224x224" in cmd
    assert "--plugins=/tmp/libplugin.so" in cmd
    assert "--iterations=50" in cmd


def test_resolve_plugin_lib_paths_merges_task_and_build_cfg() -> None:
    task = TaskConfig()
    task.profile.plugin_lib_paths = ["/global/plugin.so"]
    build_cfg = {"plugin_lib_paths": ["/cfg/plugin.so", "/global/plugin.so"]}
    paths = resolve_plugin_lib_paths(task, build_cfg)
    assert paths == ["/global/plugin.so", "/cfg/plugin.so"]


def test_iter_profile_steps_defaults_to_compile() -> None:
    task = TaskConfig(
        compile=[CompileStep(stage="vit"), CompileStep(stage="llm")],
    )
    stages = [s.stage for s in iter_profile_steps(task)]
    assert stages == ["vit", "llm"]


def test_iter_profile_steps_defaults_without_compile_or_trt_profile() -> None:
    task = TaskConfig()
    assert [s.stage for s in iter_profile_steps(task)] == ["vit", "llm", "expert", "denoise"]


def test_iter_profile_steps_uses_explicit_list() -> None:
    task = TaskConfig(trt_profile=[TrtProfileStep(stage="vit")])
    assert [s.stage for s in iter_profile_steps(task)] == ["vit"]


def test_profile_engine_stage_missing_engine_raises(tmp_path: Path) -> None:
    task = TaskConfig(
        output_dir=str(tmp_path / "out"),
        deploy={"backend": "pi05", "engine_dir": str(tmp_path / "engines")},
        model_overrides={"checkpoint": str(tmp_path / "ckpt" / "model.safetensors")},
    )
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "model.safetensors").write_bytes(b"x")
    step = TrtProfileStep(stage="vit")
    with pytest.raises(FileNotFoundError, match="Engine not found"):
        profile_engine_stage(task, step)


@patch("chameleon.deploy.trt_profile.subprocess.run")
def test_profile_engine_stage_invokes_trtexec(
    mock_run,
    tmp_path: Path,
    build_configs_dir: Path,
) -> None:
    mock_run.return_value.returncode = 0
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    engine = engine_dir / "vit.engine"
    engine.write_bytes(b"x" * 2048)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"x")

    task = TaskConfig(
        output_dir=str(tmp_path / "out"),
        deploy={
            "backend": "pi05",
            "engine_dir": str(engine_dir),
            "checkpoint_dir": str(ckpt),
            "build_cfg_dir": str(build_configs_dir),
        },
        compile=[CompileStep(stage="vit", options={"build_cfg": "vit_build_cfg.py"})],
        profile={"profile_dir": str(tmp_path / "profiles"), "iterations": 10},
    )
    paths = resolve_deploy_paths(task)

    result = profile_engine_stage(task, TrtProfileStep(stage="vit"), paths=paths)
    assert result.returncode == 0
    assert mock_run.called
    cmd = mock_run.call_args.args[0]
    assert f"--loadEngine={engine}" in cmd[1]
    assert "--exportProfile=" in " ".join(cmd)


def test_validate_engine_rejects_tiny_file(tmp_path: Path, build_configs_dir: Path) -> None:
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    engine = engine_dir / "vit.engine"
    engine.write_bytes(b"fake-engine")

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"x")

    task = TaskConfig(
        output_dir=str(tmp_path / "out"),
        deploy={
            "backend": "pi05",
            "engine_dir": str(engine_dir),
            "checkpoint_dir": str(ckpt),
            "build_cfg_dir": str(build_configs_dir),
        },
        compile=[CompileStep(stage="vit", options={"build_cfg": "vit_build_cfg.py"})],
    )
    with pytest.raises(ValueError, match="looks invalid or truncated"):
        profile_engine_stage(task, TrtProfileStep(stage="vit"))


def test_hint_from_trtexec_log_version_mismatch(tmp_path: Path) -> None:
    log = tmp_path / "llm.trtexec.log"
    log.write_text(
        "Current Version: 240, Serialized Engine Version: 243",
        encoding="utf-8",
    )
    hint = _hint_from_trtexec_log(log)
    assert hint is not None
    assert "serialization version mismatch" in hint
