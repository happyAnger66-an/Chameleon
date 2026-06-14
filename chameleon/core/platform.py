"""Platform abstraction.

A :class:`PlatformSpec` is the cross-cutting description of a deployment target
(vendor, device, supported dtypes, default toolchain, kernel artifact tag). The
quantization, compile and runtime subsystems all query the platform to decide
how to lower a model. This is the central abstraction that the previous
``model_optimizer`` implementation lacked (it was hard-wired to NVIDIA).

The design mirrors TVM's ``Target`` and MLC-LLM's ``auto_target`` presets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chameleon.core.registry import Registry


@dataclass(frozen=True)
class PlatformSpec:
    """Static capability description of a deployment target."""

    name: str
    """Unique key, e.g. ``"nvidia_orin"``."""

    vendor: str
    """One of ``nvidia | amd | intel | horizon | cpu``."""

    device: str
    """Runtime device kind, e.g. ``cuda | rocm | xpu | cpu | bpu``."""

    dtypes: tuple[str, ...]
    """Numeric formats the platform can execute, e.g. ``("fp16", "int8", "fp8")``."""

    compiler: str
    """Default :class:`~chameleon.compile.base.CompilerBackend` key for this platform."""

    runtime: str
    """Default :class:`~chameleon.runtime.base.RuntimeBackend` key for this platform."""

    kernel_tag: str | None = None
    """Tag used to select a custom-kernel artifact, e.g. ``"sm_87"`` / ``"bpu_xj3"``."""

    torch_device: str = "cpu"
    """The ``torch.device`` string used by the PyTorch reference runtime."""

    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def supports_dtype(self, dtype: str) -> bool:
        return dtype in self.dtypes


PLATFORM_REGISTRY: Registry[str, PlatformSpec] = Registry("platform")


def register_platform(spec: PlatformSpec, *, override: bool = False) -> PlatformSpec:
    PLATFORM_REGISTRY.register(spec.name, spec, override=override)
    for alias in spec.aliases:
        PLATFORM_REGISTRY.register(alias, spec, override=override)
    return spec


def get_platform(name: str) -> PlatformSpec:
    return PLATFORM_REGISTRY.get(name)


def list_platforms() -> list[PlatformSpec]:
    # De-duplicate (aliases point at the same spec object).
    seen: dict[int, PlatformSpec] = {}
    for spec in PLATFORM_REGISTRY.values():
        seen[id(spec)] = spec
    return sorted(seen.values(), key=lambda s: s.name)


def _register_builtin_platforms() -> None:
    builtins = [
        PlatformSpec(
            name="nvidia_orin",
            vendor="nvidia",
            device="cuda",
            dtypes=("fp32", "fp16", "int8", "int4"),
            compiler="tensorrt",
            runtime="tensorrt",
            kernel_tag="sm_87",
            torch_device="cuda",
            description="NVIDIA Jetson Orin (Ampere, JetPack).",
            aliases=("orin",),
        ),
        PlatformSpec(
            name="nvidia_thor",
            vendor="nvidia",
            device="cuda",
            dtypes=("fp32", "fp16", "int8", "fp8", "int4", "nvfp4"),
            compiler="tensorrt",
            runtime="tensorrt",
            kernel_tag="sm_101",
            torch_device="cuda",
            description="NVIDIA Jetson/DRIVE Thor (Blackwell).",
            aliases=("thor",),
        ),
        PlatformSpec(
            name="intel_cpu",
            vendor="intel",
            device="cpu",
            dtypes=("fp32", "fp16", "int8"),
            compiler="openvino",
            runtime="openvino",
            torch_device="cpu",
            description="Intel CPU via OpenVINO.",
        ),
        PlatformSpec(
            name="intel_gpu",
            vendor="intel",
            device="xpu",
            dtypes=("fp32", "fp16", "int8"),
            compiler="openvino",
            runtime="openvino",
            torch_device="cpu",
            description="Intel iGPU/Arc via OpenVINO.",
        ),
        PlatformSpec(
            name="amd_gpu",
            vendor="amd",
            device="rocm",
            dtypes=("fp32", "fp16", "int8"),
            compiler="tvm",
            runtime="tvm",
            torch_device="cpu",
            description="AMD GPU via TVM (Relax) codegen.",
        ),
        PlatformSpec(
            name="horizon_bpu",
            vendor="horizon",
            device="bpu",
            dtypes=("int8",),
            compiler="horizon",
            runtime="horizon",
            kernel_tag="bpu_j5",
            torch_device="cpu",
            description="Horizon Robotics BPU (Journey) via vendor SDK / TVM BYOC.",
            aliases=("horizon", "bpu"),
        ),
        PlatformSpec(
            name="generic_cpu",
            vendor="cpu",
            device="cpu",
            dtypes=("fp32", "fp16", "int8"),
            compiler="tvm",
            runtime="pytorch",
            torch_device="cpu",
            description="Generic CPU reference target (PyTorch / TVM).",
            aliases=("cpu",),
        ),
    ]
    for spec in builtins:
        register_platform(spec, override=True)


_register_builtin_platforms()
