"""pi05 架构定义 — MVP 模型的三阶段拆分规格。

作用：
    注册 pi05 ArchitectureSpec：vit（SigLIP 视觉编码）→ llm_prefix
    （PaliGemma 前缀/KV 缓存，每步算一次）→ action_expert（Gemma 动作
    专家，flow-matching 去噪热点环）。声明各 stage 支持的平台列表。

架构位置：
    模型/架构层 — 具体架构实例，import 时注册到 ARCHITECTURE_REGISTRY。
    与 models/pi05/ 配对：本文件定义"拆什么"，adapter 定义"怎么拿模块"。
"""

from __future__ import annotations

from chameleon.architectures.base import ArchitectureSpec, StageSpec
from chameleon.architectures.registry import register_architecture

ARCHITECTURE_NAME = "pi05"

_ALL_NVIDIA = ("nvidia_orin", "nvidia_thor")
_ALL = ("nvidia_orin", "nvidia_thor", "intel_cpu", "amd_gpu", "horizon_bpu", "generic_cpu")

PI05_SPEC = ArchitectureSpec(
    name=ARCHITECTURE_NAME,
    description="pi0.5 vision-language-action flow-matching policy.",
    orchestrator="pi05",
    stages=(
        StageSpec(
            name="vit",
            description="SigLIP image encoder.",
            quantizable=True,
            supported_platforms=_ALL,
        ),
        StageSpec(
            name="llm_prefix",
            description="PaliGemma prefix; builds the KV cache (run once per inference).",
            quantizable=True,
            supported_platforms=_ALL,
        ),
        StageSpec(
            name="action_expert",
            description="Gemma action expert; denoise hot loop (run num_steps times).",
            quantizable=True,
            supported_platforms=_ALL,
        ),
    ),
    metadata={
        "action_dim": 32,
        "action_horizon": 50,
        "num_denoise_steps": 10,
    },
)

register_architecture(PI05_SPEC, override=True)
