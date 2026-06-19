"""平台抽象 — 部署目标的统一能力描述。

作用：
    定义 PlatformSpec（vendor / device / dtypes / compiler / runtime /
    kernel_tag），并注册 7 个内置平台（Orin、Thor、Intel CPU/GPU、
    AMD GPU、地平线 BPU、通用 CPU）。量化、编译、运行时均查询平台
    决定如何 lower 模型。

架构位置：
    平台抽象层（最底层横切关注点）— 贯穿 quantization / compile / runtime
    三层。设计对标 TVM Target 与 MLC-LLM auto_target preset；补全了
    model_optimizer 缺失的平台维度。
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
            name="nvidia_ada",
            vendor="nvidia",
            device="cuda",
            dtypes=("fp32", "fp16", "int8", "fp8"),
            compiler="tensorrt",
            runtime="tensorrt",
            kernel_tag="sm_89",
            torch_device="cuda",
            description="NVIDIA desktop Ada Lovelace (RTX 40 series, e.g. RTX 4070).",
            aliases=("rtx4070", "rtx_4070", "ada", "sm_89"),
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
