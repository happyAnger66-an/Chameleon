"""Edge-LLM ASR runtime — 复合 engine（audio encoder + LLM prefill/decode）。

优先 pybind ``_edgellm_runtime.LLMRuntime``；不可用时回退到 ``llm_inference`` CLI。
输入契约（wav 直喂，C++ mel）::

    inputs = {
        "audio": "/path/to.wav",   # 或 audio_path
        "context": "",
        "language": None,          # 强制语种时填 canonical 名
        "max_new_tokens": 256,
        "prompt": None,            # 可选；None 则由 runtime 走 chat template
    }
    outputs = {"text": str, "language": str, "raw_text": str, "metrics": dict}
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from chameleon.core.artifact import Artifact
from chameleon.core.context import RunContext
from chameleon.models.qwen3_asr.adapter import parse_asr_output
from chameleon.runtime.base import Engine, RuntimeBackend, register_runtime

logger = logging.getLogger(__name__)


def _try_import_pybind():
    try:
        import _edgellm_runtime as rt  # type: ignore

        return rt
    except ImportError:
        return None


class EdgeLLMAsrEngine(Engine):
    stage = "asr"

    def __init__(
        self,
        *,
        llm_engine_dir: Path,
        multimodal_engine_dir: Path,
        edgellm_home: Path | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.llm_engine_dir = Path(llm_engine_dir)
        self.multimodal_engine_dir = Path(multimodal_engine_dir)
        self.edgellm_home = Path(edgellm_home) if edgellm_home else None
        self.max_new_tokens = int(max_new_tokens)
        self._py = _try_import_pybind()
        self._rt = None
        if self._py is not None:
            try:
                self._rt = self._py.LLMRuntime(
                    str(self.llm_engine_dir),
                    str(self.multimodal_engine_dir),
                    {},
                )
                logger.info(
                    "EdgeLLMAsrEngine: pybind LLMRuntime ready llm=%s multimodal=%s",
                    self.llm_engine_dir,
                    self.multimodal_engine_dir,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("pybind LLMRuntime init failed (%s); CLI fallback.", exc)
                self._rt = None

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        audio = inputs.get("audio") or inputs.get("audio_path")
        if not audio:
            raise ValueError("EdgeLLMAsrEngine.run requires inputs['audio'] path")
        audio_path = str(Path(audio).expanduser().resolve())
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        context = str(inputs.get("context") or "")
        language = inputs.get("language")
        max_new = int(inputs.get("max_new_tokens") or self.max_new_tokens)
        stream_channel = inputs.get("stream_channel")

        if self._rt is not None:
            raw = self._run_pybind(
                audio_path=audio_path,
                context=context,
                language=language,
                max_new_tokens=max_new,
                stream_channel=stream_channel,
            )
        else:
            raw = self._run_cli(
                audio_path=audio_path,
                context=context,
                max_new_tokens=max_new,
            )

        lang, text = parse_asr_output(raw, user_language=language)
        return {
            "text": text,
            "language": lang,
            "raw_text": raw,
            "metrics": {},
        }

    def _run_pybind(
        self,
        *,
        audio_path: str,
        context: str,
        language: str | None,
        max_new_tokens: int,
        stream_channel: Any | None,
    ) -> str:
        assert self._py is not None and self._rt is not None
        py = self._py
        # Build request matching Edge-LLM asr.md JSON shape.
        content = py.MessageContent() if hasattr(py, "MessageContent") else None
        # Prefer high-level constructors if exposed; else dict-like Request API.
        req = py.LLMGenerationRequest()
        req.temperature = 1.0
        req.top_p = 1.0
        req.top_k = 50
        req.max_generate_length = max_new_tokens
        req.apply_chat_template = True
        req.add_generation_prompt = True

        inner = py.Request()
        # messages: system + user(audio)
        # pybind Message API varies; use formatted path when available.
        try:
            from collections import namedtuple

            # Fallback: use FormattedRequest if Message binding is awkward
            msg_system = {"role": "system", "content": context}
            msg_user = {"role": "user", "content": [{"type": "audio", "audio": audio_path}]}
            # Many pybind builds expect C++ Message objects — try attribute setters.
            if hasattr(py, "Message") and hasattr(py, "MessageContent"):
                m0 = py.Message()
                m0.role = "system"
                c0 = py.MessageContent()
                c0.type = "text"
                c0.content = context
                m0.contents = [c0]
                m1 = py.Message()
                m1.role = "user"
                c1 = py.MessageContent()
                c1.type = "audio"
                c1.content = audio_path
                m1.contents = [c1]
                inner.messages = [m0, m1]
            else:
                # Last resort: write JSON and use CLI
                return self._run_cli(
                    audio_path=audio_path, context=context, max_new_tokens=max_new_tokens
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pybind request assemble failed (%s); CLI fallback.", exc)
            return self._run_cli(
                audio_path=audio_path, context=context, max_new_tokens=max_new_tokens
            )

        if language:
            # Force language via generation prompt suffix by disabling template
            # and using a pre-rendered prompt is complex; rely on system context
            # note + model free-form, or leave to adapter-level prompt in CLI path.
            pass

        req.requests = [inner]
        if stream_channel is not None and hasattr(req, "stream_channels"):
            req.stream_channels = [stream_channel]

        resp = self._rt.handle_request(req)
        # Response shape: list of strings or object with .outputs
        if hasattr(resp, "outputs"):
            outs = resp.outputs
            if outs:
                first = outs[0]
                return str(getattr(first, "text", first) or "")
        if isinstance(resp, (list, tuple)) and resp:
            return str(resp[0])
        return str(resp or "")

    def _run_cli(self, *, audio_path: str, context: str, max_new_tokens: int) -> str:
        home = self.edgellm_home
        llm_inf = None
        if home is not None:
            cand = home / "build" / "examples" / "llm" / "llm_inference"
            if cand.is_file():
                llm_inf = str(cand)
        if llm_inf is None:
            llm_inf = shutil.which("llm_inference")
        if llm_inf is None:
            raise RuntimeError(
                "Neither _edgellm_runtime pybind nor llm_inference CLI is available."
            )

        payload = {
            "batch_size": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "max_generate_length": max_new_tokens,
            "requests": [
                {
                    "messages": [
                        {"role": "system", "content": context},
                        {
                            "role": "user",
                            "content": [{"type": "audio", "audio": audio_path}],
                        },
                    ]
                }
            ],
        }
        with tempfile.TemporaryDirectory(prefix="chameleon_asr_") as td:
            inp = Path(td) / "input.json"
            out = Path(td) / "output.json"
            inp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cmd = [
                llm_inf,
                "--engineDir",
                str(self.llm_engine_dir),
                "--multimodalEngineDir",
                str(self.multimodal_engine_dir),
                "--inputFile",
                str(inp),
                "--outputFile",
                str(out),
            ]
            logger.info("EdgeLLMAsrEngine CLI: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
            data = json.loads(out.read_text(encoding="utf-8"))
        # Common shapes: {"responses":[{"text":...}]} or list
        if isinstance(data, dict):
            if "responses" in data and data["responses"]:
                r0 = data["responses"][0]
                return str(r0.get("text") or r0.get("output") or r0)
            if "outputs" in data and data["outputs"]:
                return str(data["outputs"][0])
        if isinstance(data, list) and data:
            return str(data[0])
        return str(data)


class EdgeLLMRuntimeBackend(RuntimeBackend):
    name = "edgellm"

    def available(self) -> bool:
        if _try_import_pybind() is not None:
            return True
        return shutil.which("llm_inference") is not None or bool(
            os.environ.get("TENSORRT_EDGELLM_HOME")
        )

    def load(self, artifact: Artifact, ctx: RunContext) -> Engine:
        """Load ASR composite engine.

        ``artifact.path`` should be the parent engines dir containing ``llm/`` and ``audio/``,
        or the llm engine dir (then multimodal is sibling ``../audio`` or options).
        """
        path = Path(artifact.path)
        opts = dict(ctx.options or {})
        if (path / "llm").is_dir() and (path / "audio").is_dir():
            llm_dir, audio_dir = path / "llm", path / "audio"
        elif path.name == "llm":
            llm_dir = path
            audio_dir = Path(opts.get("multimodal_engine_dir") or (path.parent / "audio"))
        else:
            llm_dir = path / "llm" if (path / "llm").is_dir() else path
            audio_dir = Path(opts.get("multimodal_engine_dir") or (path.parent / "audio"))

        home = opts.get("edgellm_home") or os.environ.get("TENSORRT_EDGELLM_HOME")
        return EdgeLLMAsrEngine(
            llm_engine_dir=llm_dir,
            multimodal_engine_dir=audio_dir,
            edgellm_home=Path(home) if home else None,
            max_new_tokens=int(opts.get("max_new_tokens") or 256),
        )


register_runtime(EdgeLLMRuntimeBackend(), override=True)
