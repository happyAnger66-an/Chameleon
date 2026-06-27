"""Cosmos3 架构定义 — Omni 世界基础模型的四阶段拆分规格。

作用：
    注册 cosmos3 ArchitectureSpec：vae_encode（Wan VAE 条件编码）→
    text_embed（文本 token embedding，每次推理算一次）→ dit（MoT 联合
    transformer，flow-matching 去噪热点环，每步整模型 forward）→ vae_decode
    （隐变量解码为视频）。对齐 pi05 的 vit / llm_prefix / action_expert 三段
    范式，但去噪核心是单个 MoT 联合 transformer 而非轻量 cross-attn expert。

架构位置：
    模型/架构层 — 具体架构实例，import 时注册到 ARCHITECTURE_REGISTRY。
    与 models/cosmos3/ 配对：本文件定义"拆什么 stage"，adapter 定义"怎么拿
    模块"。两种生成模式（action 策略 / video 世界生成）共用同一组 stage。
"""

from __future__ import annotations

from chameleon.architectures.base import ArchitectureSpec, StageSpec
from chameleon.architectures.registry import register_architecture

ARCHITECTURE_NAME = "cosmos3"

_ALL = ("nvidia_orin", "nvidia_thor", "nvidia_ada", "intel_cpu", "amd_gpu", "horizon_bpu", "generic_cpu")

COSMOS3_SPEC = ArchitectureSpec(
    name=ARCHITECTURE_NAME,
    description="Cosmos3 omni world-foundation model (MoT generator) — action + video generation.",
    orchestrator="cosmos3",
    stages=(
        StageSpec(
            name="vae_encode",
            description="Wan VAE encoder; conditioning image/video -> latents (run once).",
            quantizable=False,
            supported_platforms=_ALL,
        ),
        StageSpec(
            name="text_embed",
            description="Text token embedding / understanding prefix (run once per inference).",
            quantizable=True,
            supported_platforms=_ALL,
        ),
        StageSpec(
            name="dit",
            description="Cosmos3 MoT joint transformer; denoise hot loop (run num_steps x CFG).",
            quantizable=True,
            supported_platforms=_ALL,
        ),
        StageSpec(
            name="vae_decode",
            description="Wan VAE decoder; latents -> video frames (run once).",
            quantizable=False,
            supported_platforms=_ALL,
        ),
    ),
    metadata={
        # action 模式输出维度（对齐 diffusers CosmosActionCondition）。
        "action_dim": 32,
        "action_horizon": 16,
        "num_denoise_steps": 35,
        # 生成默认（video 模式）。
        "num_frames": 189,
        "height": 720,
        "width": 1280,
        "fps": 24.0,
        "guidance_scale": 6.0,
    },
)

register_architecture(COSMOS3_SPEC, override=True)
