"""qwen3_asr deploy — 封装 TensorRT-Edge-LLM export / audio_build / llm_build。

产物目录对齐 Edge-LLM runtime 加载约定::

    {engine_dir}/
      llm/     # llm_build 输出
      audio/   # audio_build 输出
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from chameleon.config.schema import TaskConfig
from chameleon.core.artifact import Artifact, Manifest
from chameleon.deploy.paths import resolve_checkpoint_dir

logger = logging.getLogger(__name__)


def resolve_edgellm_home(task: TaskConfig) -> Path:
    raw = getattr(task.deploy, "edgellm_home", None) or os.environ.get("TENSORRT_EDGELLM_HOME")
    if not raw:
        raise ValueError(
            "deploy.edgellm_home or TENSORRT_EDGELLM_HOME required for qwen3_asr deploy."
        )
    home = Path(str(raw)).expanduser().resolve()
    if not home.is_dir():
        raise FileNotFoundError(f"Edge-LLM home not found: {home}")
    return home


def _export_dir(task: TaskConfig) -> Path:
    d = task.deploy.export_dir or f"{task.output_dir}/onnx"
    path = Path(d).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _engine_dir(task: TaskConfig) -> Path:
    d = task.deploy.engine_dir or f"{task.output_dir}/engines"
    path = Path(d).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _which_or_path(home: Path, names: list[str]) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for name in names:
        for candidate in (
            home / "build" / "examples" / "llm" / name,
            home / "build" / "examples" / "multimodal" / name,
            home / name,
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _load_build_cfg(task: TaskConfig, stage: str) -> dict[str, Any]:
    """Load optional build_cfg .py (returns ``build_cfg`` dict)."""
    from chameleon.deploy.paths import resolve_build_cfg_dir, resolve_deploy_paths

    root = Path(__file__).resolve().parents[2]
    defaults = {
        "audio_encoder": "qwen3_asr_audio_build_cfg.py",
        "llm": "qwen3_asr_llm_build_cfg.py",
    }
    rel = (task.deploy.build_cfgs or {}).get(stage) or defaults.get(stage)
    candidates: list[Path] = []
    if rel:
        p = Path(rel)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(resolve_build_cfg_dir(task) / p)
            candidates.append(root / "configs" / "build_configs" / p.name)
    for path in candidates:
        if path.is_file():
            ns: dict[str, Any] = {}
            exec(path.read_text(), ns)  # noqa: S102 — trusted local build_cfg
            return dict(ns.get("build_cfg") or {})
    return {}


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def run_qwen3_asr_export(task: TaskConfig, manifest: Manifest) -> dict[str, Artifact]:
    home = resolve_edgellm_home(task)
    ckpt = resolve_checkpoint_dir(task)
    out = _export_dir(task)

    export_cli = _which_or_path(home, ["tensorrt-edgellm-export"])
    if export_cli is None:
        # Module form
        export_cmd = [
            os.environ.get("EDGELLM_PYTHON", "python3"),
            "-m",
            "tensorrt_edgellm.scripts.export",
            str(ckpt),
            str(out),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(home), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        logger.info("Running: %s", " ".join(export_cmd))
        subprocess.run(export_cmd, check=True, env=env, cwd=str(home))
    else:
        _run([export_cli, str(ckpt), str(out)], cwd=home)

    arts: dict[str, Artifact] = {}
    for stage, sub in (("audio_encoder", "audio"), ("llm", "llm")):
        stage_dir = out / sub
        arts[stage] = Artifact(
            kind="onnx",
            stage=stage,
            platform=task.platform,
            path=str(stage_dir),
        )
        manifest.add(arts[stage])
    return arts


def run_qwen3_asr_build(task: TaskConfig, manifest: Manifest) -> dict[str, Artifact]:
    home = resolve_edgellm_home(task)
    export = _export_dir(task)
    engine = _engine_dir(task)

    audio_build = _which_or_path(home, ["audio_build"])
    llm_build = _which_or_path(home, ["llm_build"])
    if audio_build is None or llm_build is None:
        raise FileNotFoundError(
            f"audio_build/llm_build not found under {home} or PATH. "
            "Build Edge-LLM examples first."
        )

    audio_cfg = _load_build_cfg(task, "audio_encoder")
    llm_cfg = _load_build_cfg(task, "llm")

    audio_out = engine / "audio"
    llm_out = engine / "llm"
    audio_out.mkdir(parents=True, exist_ok=True)
    llm_out.mkdir(parents=True, exist_ok=True)

    min_ts = int(audio_cfg.get("min_time_steps", audio_cfg.get("minTimeSteps", 1000)))
    max_ts = int(audio_cfg.get("max_time_steps", audio_cfg.get("maxTimeSteps", 3000)))
    _run(
        [
            audio_build,
            "--onnxDir",
            str(export / "audio"),
            "--engineDir",
            str(audio_out),
            "--minTimeSteps",
            str(min_ts),
            "--maxTimeSteps",
            str(max_ts),
        ],
        cwd=home,
    )

    max_bs = int(llm_cfg.get("max_batch_size", llm_cfg.get("maxBatchSize", 1)))
    max_in = int(llm_cfg.get("max_input_len", llm_cfg.get("maxInputLen", 1024)))
    max_kv = int(llm_cfg.get("max_kv_cache_capacity", llm_cfg.get("maxKVCacheCapacity", 4096)))
    _run(
        [
            llm_build,
            "--onnxDir",
            str(export / "llm"),
            "--engineDir",
            str(llm_out),
            "--maxBatchSize",
            str(max_bs),
            "--maxInputLen",
            str(max_in),
            "--maxKVCacheCapacity",
            str(max_kv),
        ],
        cwd=home,
    )

    arts: dict[str, Artifact] = {
        "audio_encoder": Artifact(
            kind="engine", stage="audio_encoder", platform=task.platform, path=str(audio_out)
        ),
        "llm": Artifact(kind="engine", stage="llm", platform=task.platform, path=str(llm_out)),
    }
    for a in arts.values():
        manifest.add(a)
    return arts


def run_qwen3_asr_quantize(task: TaskConfig, manifest: Manifest) -> dict[str, Artifact]:
    """Optional Edge-LLM quantize step (LLM ± audio).

    Reads ``task.quantize`` steps / ``model_overrides``:
      - quantization: fp8 | nvfp4
      - audio_quantization: fp8 | None
      - lm_head_quantization: optional
    """
    home = resolve_edgellm_home(task)
    ckpt = resolve_checkpoint_dir(task)
    quant = (task.model_overrides.get("quantization") or "").strip().lower()
    if not quant and task.quantize:
        # Derive from first llm-ish step method
        quant = str(task.quantize[0].method).strip().lower()
    if not quant:
        raise ValueError("qwen3_asr quantize needs model_overrides.quantization=fp8|nvfp4")

    audio_q = task.model_overrides.get("audio_quantization")
    if audio_q is None:
        for step in task.quantize:
            if step.stage in ("audio_encoder", "audio") and step.method:
                audio_q = step.method
                break
    lm_head_q = task.model_overrides.get("lm_head_quantization")

    # Guard: 0.6B NVFP4 LLM + FP8 audio is known-bad
    model_name = str(task.model or "")
    if "0.6" in model_name and quant == "nvfp4" and str(audio_q).lower() == "fp8":
        raise ValueError(
            "Qwen3-ASR-0.6B does not support NVFP4 LLM + FP8 audio "
            "(empty output). Use NVFP4+FP16 audio or FP8+FP8."
        )

    out = Path(task.output_dir).expanduser().resolve() / f"quant_{quant}"
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        os.environ.get("EDGELLM_PYTHON", "python3"),
        "-m",
        "tensorrt_edgellm.scripts.quantize",
        "llm",
        "--model_dir",
        str(ckpt),
        "--output_dir",
        str(out),
        "--quantization",
        quant,
    ]
    if audio_q:
        cmd.extend(["--audio_quantization", str(audio_q)])
    if lm_head_q:
        cmd.extend(["--lm_head_quantization", str(lm_head_q)])

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(home), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(home))

    art = Artifact(kind="quantized", stage="llm", platform=task.platform, path=str(out))
    manifest.add(art)
    # Point subsequent export at quantized dir
    task.deploy.checkpoint_dir = str(out)
    task.model_overrides["checkpoint"] = str(out)
    return {"llm": art}
