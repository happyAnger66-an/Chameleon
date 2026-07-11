"""qwen3_asr 架构定义 — 音频编码器 + LLM 双 stage（Edge-LLM 双 engine）。

作用：
    注册 qwen3_asr ArchitectureSpec。stage 主要用于产物路径、trt_profile
    （audio_encoder 可 trtexec）与 metadata；实际 export/build 委托
    deploy backend ``qwen3_asr``（Edge-LLM CLI）。

架构位置：
    模型/架构层 — 与 models/qwen3_asr/、deploy/qwen3_asr_edgellm 配对。
"""

from __future__ import annotations

from chameleon.architectures.base import ArchitectureSpec, StageSpec
from chameleon.architectures.registry import register_architecture

ARCHITECTURE_NAME = "qwen3_asr"

_NVIDIA = ("nvidia_orin", "nvidia_thor", "nvidia_ada")

QWEN3_ASR_SPEC = ArchitectureSpec(
    name=ARCHITECTURE_NAME,
    description="Qwen3-ASR: Whisper-style audio encoder + Qwen3 LLM (via TensorRT-Edge-LLM).",
    orchestrator="qwen3_asr",
    stages=(
        StageSpec(
            name="audio_encoder",
            description="Whisper-style audio encoder (mel → embeddings); one-shot, no KV.",
            quantizable=True,
            supported_platforms=_NVIDIA,
        ),
        StageSpec(
            name="llm",
            description="Qwen3 causal LM with multimodal embedding inject + KV decode.",
            quantizable=True,
            supported_platforms=_NVIDIA,
        ),
    ),
    metadata={
        "sample_rate": 16000,
        "mel_bins": 128,
        "max_new_tokens": 256,
        "audio_tokens_per_sec": 13,
    },
)

register_architecture(QWEN3_ASR_SPEC, override=True)
