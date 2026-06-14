"""pi05 (pi0.5) architecture definition.

pi05 is a vision-language-action (VLA) flow-matching policy. Following the
reference implementation in openpi's ``pi0_pytorch.py`` it decomposes into
three stages:

* ``vit``          - SigLIP image encoder producing image tokens.
* ``llm_prefix``   - PaliGemma (Gemma) processing image + language tokens,
                     producing the prefix KV cache (computed once per step).
* ``action_expert``- Gemma action expert run inside the flow-matching denoise
                     loop, consuming the prefix KV cache (the latency hot path).
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
