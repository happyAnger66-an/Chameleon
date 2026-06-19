"""info 类子命令 — platforms / architectures / info。"""

from __future__ import annotations

import argparse

from chameleon.architectures.registry import ARCHITECTURE_REGISTRY, get_architecture
from chameleon.commands.common import add_global_arguments, configure_logging
from chameleon.compile.base import COMPILER_REGISTRY
from chameleon.core.platform import list_platforms
from chameleon.kernels.base import KERNEL_REGISTRY
from chameleon.quantization.registry import QUANT_METHOD_REGISTRY
from chameleon.runtime.base import RUNTIME_REGISTRY


def platforms_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chameleon platforms", description="List deployment platforms.")
    add_global_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    print(f"{'NAME':<16}{'VENDOR':<10}{'DEVICE':<8}{'COMPILER':<12}{'RUNTIME':<10}DTYPES")
    for spec in list_platforms():
        print(
            f"{spec.name:<16}{spec.vendor:<10}{spec.device:<8}"
            f"{spec.compiler:<12}{spec.runtime:<10}{','.join(spec.dtypes)}"
        )
    return 0


def architectures_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chameleon architectures", description="List architectures and stages.")
    add_global_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    for name in ARCHITECTURE_REGISTRY.keys():
        spec = get_architecture(name)
        print(f"{name}: orchestrator={spec.orchestrator}")
        for stage in spec.stages:
            platforms = ",".join(stage.supported_platforms) or "all"
            print(f"  - {stage.name:<14} quantizable={stage.quantizable} platforms=[{platforms}]")
    return 0


def info_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chameleon info", description="List registered backends.")
    add_global_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    print("compilers:   ", ", ".join(COMPILER_REGISTRY.keys()))
    print("runtimes:    ", ", ".join(RUNTIME_REGISTRY.keys()))
    print("quant methods:", ", ".join(QUANT_METHOD_REGISTRY.keys()))
    print("kernels:     ", ", ".join(f"{op}@{vendor}" for op, vendor in KERNEL_REGISTRY.keys()))
    return 0
