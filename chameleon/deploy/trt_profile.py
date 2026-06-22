"""对已构建 TensorRT engine 运行 trtexec layer profiling。"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from chameleon.config.schema import TaskConfig, TrtProfileStep
from chameleon.core.artifact import Artifact, Manifest
from chameleon.deploy.build_cfg import load_build_cfg
from chameleon.deploy.paths import (
    DeployPaths,
    resolve_build_cfg_path,
    resolve_deploy_paths,
    resolve_profile_dir,
    stage_engine_path,
    stage_layer_info_path,
    stage_profile_log_path,
    stage_profile_path,
    stage_times_path,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE_STAGES = ("vit", "llm", "expert", "denoise")
_MIN_ENGINE_BYTES = 1024
"""Real serialized TRT engines are much larger; tiny files are almost always corrupt placeholders."""


@dataclass(frozen=True)
class TrtProfileResult:
    stage: str
    engine_path: Path
    profile_path: Path
    log_path: Path
    returncode: int
    layer_info_path: Path | None = None
    times_path: Path | None = None


def format_shapes_for_trtexec(shapes: dict[str, Sequence[int]] | None) -> str | None:
    """Convert ``{name: (d1, d2, ...)}`` to trtexec ``name:d1xd2,...`` string."""
    if not shapes:
        return None
    parts: list[str] = []
    for name, dims in shapes.items():
        if dims is None:
            continue
        parts.append(f"{name}:{'x'.join(str(d) for d in dims)}")
    return ",".join(parts) if parts else None


def resolve_plugin_lib_paths(
    task: TaskConfig,
    build_cfg: dict[str, Any],
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for raw in list(task.profile.plugin_lib_paths) + list(build_cfg.get("plugin_lib_paths") or []):
        p = str(raw).strip()
        if not p or p in seen:
            continue
        seen.add(p)
        paths.append(p)
    return paths


def build_trtexec_profile_command(
    *,
    engine_path: Path,
    profile_path: Path,
    build_cfg: dict[str, Any],
    iterations: int,
    warmup: int,
    separate_profile_run: bool,
    profiling_verbosity: str,
    export_layer_info: bool,
    layer_info_path: Path | None,
    export_times: bool,
    times_path: Path | None,
    plugin_lib_paths: Sequence[str],
    trtexec_extra_args: Sequence[str] | None = None,
    trtexec_path: str | None = None,
) -> list[str]:
    """Assemble ``trtexec --loadEngine=... --dumpProfile --exportProfile=...`` argv."""
    exe = trtexec_path or shutil.which("trtexec") or "trtexec"
    cmd: list[str] = [exe, f"--loadEngine={engine_path}"]

    if separate_profile_run:
        cmd.append("--separateProfileRun")
    cmd.extend(
        [
            "--dumpProfile",
            f"--exportProfile={profile_path}",
            f"--profilingVerbosity={profiling_verbosity}",
            f"--iterations={iterations}",
            f"--warmUp={warmup}",
        ]
    )

    for flag, key in (
        ("minShapes", "min_shapes"),
        ("optShapes", "opt_shapes"),
        ("maxShapes", "max_shapes"),
    ):
        shape_str = format_shapes_for_trtexec(build_cfg.get(key))
        if shape_str:
            cmd.append(f"--{flag}={shape_str}")

    if export_layer_info and layer_info_path is not None:
        cmd.append(f"--exportLayerInfo={layer_info_path}")
    if export_times and times_path is not None:
        cmd.append(f"--exportTimes={times_path}")

    for plugin in plugin_lib_paths:
        cmd.append(f"--plugins={plugin}")

    if trtexec_extra_args:
        cmd.extend(trtexec_extra_args)

    return cmd


def iter_profile_steps(task: TaskConfig) -> list[TrtProfileStep]:
    if task.trt_profile:
        return list(task.trt_profile)
    if task.compile:
        return [TrtProfileStep(stage=step.stage, options=dict(step.options)) for step in task.compile]
    return [TrtProfileStep(stage=s) for s in _DEFAULT_PROFILE_STAGES]


def _validate_engine_file(engine_path: Path) -> None:
    size = engine_path.stat().st_size
    if size < _MIN_ENGINE_BYTES:
        raise ValueError(
            f"Engine file {engine_path} is only {size} bytes and looks invalid or truncated. "
            "Re-run compile for this stage before trt_profile."
        )


def _hint_from_trtexec_log(log_path: Path) -> str | None:
    """Return an actionable hint when trtexec log indicates a known failure mode."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "Serialized Engine Version" in text and "Current Version" in text:
        return (
            "TensorRT serialization version mismatch: engine was built with a different "
            "TensorRT than the trtexec/runtime loading it. Use profile.trtexec_path pointing "
            "to trtexec from the same TensorRT install as compile (openpi venv uses "
            "tensorrt 11.x; system /usr/bin/trtexec is often 10.x), or recompile all engines "
            "with the same TensorRT as trtexec."
        )
    if "Failed to read header from the stream" in text:
        return (
            "Engine file looks corrupt or truncated. Re-run compile for this stage "
            "(do not use unit-test placeholder files)."
        )
    return None


def _resolve_trtexec_path(task: TaskConfig, step: TrtProfileStep) -> str | None:
    raw = step.options.get("trtexec_path") or task.profile.trtexec_path
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"trtexec not found: {path}")
        return str(path.resolve())
    return None


def profile_engine_stage(
    task: TaskConfig,
    step: TrtProfileStep,
    *,
    paths: DeployPaths | None = None,
) -> TrtProfileResult:
    """Run trtexec profiling for one compiled engine stage."""
    if step.options.get("skip"):
        raise ValueError(f"Stage {step.stage!r} marked skip=true.")

    paths = paths or resolve_deploy_paths(task)
    profile_dir = resolve_profile_dir(task)
    profile_dir.mkdir(parents=True, exist_ok=True)

    engine_path = stage_engine_path(paths, step.stage)
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"Engine not found for stage {step.stage!r}: {engine_path}. Run compile first."
        )
    _validate_engine_file(engine_path)

    build_cfg_path = step.options.get("build_cfg")
    if build_cfg_path:
        cfg_path = Path(str(build_cfg_path)).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (paths.build_cfg_dir / cfg_path).resolve()
    else:
        cfg_path = resolve_build_cfg_path(task, step.stage, paths)
    build_cfg = load_build_cfg(cfg_path)

    profile_path = stage_profile_path(profile_dir, step.stage)
    log_path = stage_profile_log_path(profile_dir, step.stage)
    layer_info_path = stage_layer_info_path(profile_dir, step.stage)
    times_path = stage_times_path(profile_dir, step.stage)

    iterations = int(step.options.get("iterations", task.profile.iterations))
    warmup = int(step.options.get("warmup", task.profile.warmup))
    separate = bool(step.options.get("separate_profile_run", task.profile.separate_profile_run))
    verbosity = str(step.options.get("profiling_verbosity", task.profile.profiling_verbosity))
    export_layer_info = bool(step.options.get("export_layer_info", task.profile.export_layer_info))
    export_times = bool(step.options.get("export_times", task.profile.export_times))
    extra_raw = step.options.get("trtexec_extra_args")
    extra_args: list[str] | None = None
    if extra_raw:
        extra_args = extra_raw if isinstance(extra_raw, list) else str(extra_raw).split()

    plugin_paths = resolve_plugin_lib_paths(task, build_cfg)
    cmd = build_trtexec_profile_command(
        engine_path=engine_path,
        profile_path=profile_path,
        build_cfg=build_cfg,
        iterations=iterations,
        warmup=warmup,
        separate_profile_run=separate,
        profiling_verbosity=verbosity,
        export_layer_info=export_layer_info,
        layer_info_path=layer_info_path if export_layer_info else None,
        export_times=export_times,
        times_path=times_path if export_times else None,
        plugin_lib_paths=plugin_paths,
        trtexec_extra_args=extra_args,
        trtexec_path=_resolve_trtexec_path(task, step),
    )

    logger.info("Profiling stage %s: %s", step.stage, " ".join(cmd))
    timeout = int(step.options.get("timeout_sec", task.profile.timeout_sec))
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"# command: {' '.join(cmd)}\n\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout if timeout > 0 else None,
            check=False,
        )

    if proc.returncode != 0:
        hint = _hint_from_trtexec_log(log_path)
        if hint:
            logger.error("trtexec failed for stage %s: %s", step.stage, hint)

    return TrtProfileResult(
        stage=step.stage,
        engine_path=engine_path,
        profile_path=profile_path,
        log_path=log_path,
        returncode=proc.returncode,
        layer_info_path=layer_info_path if export_layer_info else None,
        times_path=times_path if export_times else None,
    )


def write_profile_viewer_artifacts(
    task: TaskConfig,
    results: dict[str, TrtProfileResult],
) -> tuple[Path, Path]:
    """Write ``profiles/index.html`` and ``profiles/manifest.json``."""
    from chameleon.draw.trt_profile_viewer import (
        ProfileBundle,
        build_multi_stage_dashboard,
        build_stage_profile_html,
        load_trtexec_profile_rows,
    )

    profile_dir = resolve_profile_dir(task)
    profile_dir.mkdir(parents=True, exist_ok=True)

    bundles: dict[str, ProfileBundle] = {}
    manifest: dict[str, Any] = {"stages": {}, "profile_dir": str(profile_dir)}

    for stage, result in results.items():
        if result.returncode != 0 or not result.profile_path.is_file():
            continue
        rows, iteration_count = load_trtexec_profile_rows(str(result.profile_path))
        rel_profile = result.profile_path.relative_to(profile_dir)
        bundles[stage] = ProfileBundle(
            stage=stage,
            rows=rows,
            iteration_count=iteration_count,
            profile_path=str(rel_profile),
        )
        manifest["stages"][stage] = {
            "profile_json": str(rel_profile),
            "engine": str(result.engine_path),
            "trtexec_log": str(result.log_path.relative_to(profile_dir)),
            "returncode": result.returncode,
        }

    index_path = profile_dir / "index.html"
    index_path.write_text(build_multi_stage_dashboard(bundles), encoding="utf-8")

    for stage, bundle in bundles.items():
        stage_html = profile_dir / f"{stage}.html"
        stage_html.write_text(build_stage_profile_html(bundle), encoding="utf-8")
        manifest["stages"][stage]["viewer_html"] = str(stage_html.relative_to(profile_dir))

    manifest_path = profile_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return index_path, manifest_path


def maybe_serve_profile_viewer(task: TaskConfig, index_path: Path) -> None:
    viewer = (task.profile.viewer or "static").strip().lower()
    if viewer not in ("webui", "both"):
        return

    from chameleon.draw.trt_profile_viewer import pick_free_port, serve_profile_html

    host = task.profile.webui_host
    port = pick_free_port(host, task.profile.webui_port)
    html = index_path.read_text(encoding="utf-8")
    serve_profile_html(html, host=host, port=port, open_browser=task.profile.open_browser)


def run_trt_profile(task: TaskConfig, manifest: Manifest) -> dict[str, Artifact]:
    """Profile all configured stages and optionally write/serve the WebUI."""
    paths = resolve_deploy_paths(task)
    profile_dir = resolve_profile_dir(task)
    profile_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, TrtProfileResult] = {}
    artifacts: dict[str, Artifact] = {}
    fail_fast = task.profile.fail_fast

    for step in iter_profile_steps(task):
        try:
            result = profile_engine_stage(task, step, paths=paths)
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("TRT profile skipped for stage %s: %s", step.stage, exc)
            manifest.add(
                Artifact(
                    kind="trt_profile_skipped",
                    stage=step.stage,
                    platform=task.platform,
                    metadata={"reason": str(exc)[:300]},
                )
            )
            if fail_fast or step.options.get("fail_fast"):
                raise
            continue

        results[step.stage] = result
        metadata: dict[str, Any] = {
            "engine": str(result.engine_path),
            "iterations": int(step.options.get("iterations", task.profile.iterations)),
            "trtexec_log": str(result.log_path.relative_to(profile_dir)),
            "returncode": result.returncode,
        }
        if result.layer_info_path and result.layer_info_path.is_file():
            metadata["layer_info"] = str(result.layer_info_path.relative_to(profile_dir))
        if result.times_path and result.times_path.is_file():
            metadata["times"] = str(result.times_path.relative_to(profile_dir))

        artifact = Artifact(
            kind="trt_profile",
            stage=step.stage,
            platform=task.platform,
            path=str(result.profile_path),
            metadata=metadata,
        )
        manifest.add(artifact)
        artifacts[step.stage] = artifact

        if result.returncode != 0:
            logger.warning(
                "trtexec exited %s for stage %s (see %s)",
                result.returncode,
                step.stage,
                result.log_path,
            )
            if fail_fast or step.options.get("fail_fast"):
                raise RuntimeError(
                    f"trtexec failed for stage {step.stage!r} (exit {result.returncode})."
                )

    if results:
        index_path, _ = write_profile_viewer_artifacts(task, results)
        manifest.add(
            Artifact(
                kind="trt_profile_viewer",
                platform=task.platform,
                path=str(index_path),
                metadata={"viewer": task.profile.viewer, "profile_dir": str(profile_dir)},
            )
        )
        maybe_serve_profile_viewer(task, index_path)

    return artifacts
