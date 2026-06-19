"""TensorRT API 版本兼容（8.x / 9.x / 10.x）。"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def network_creation_flags(*, strongly_typed: bool = False) -> int | None:
    """Build ``create_network`` flags.

    TensorRT 10+ removed ``EXPLICIT_BATCH`` (networks are always explicit-batch).
    Returns ``None`` when no flags are needed (TRT 10 default path).
    """
    import tensorrt as trt

    flags = 0
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit_batch is not None:
        flags |= 1 << int(explicit_batch)

    if strongly_typed:
        st_flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
        if st_flag is not None:
            flags |= 1 << int(st_flag)

    return flags if flags else None


def create_onnx_network(builder, *, strongly_typed: bool = False):
    """Create a network for ONNX parsing across TensorRT versions."""
    flags = network_creation_flags(strongly_typed=strongly_typed)
    if flags is None:
        logger.debug("create_network() with default flags (TRT 10+ explicit batch)")
        return builder.create_network()
    return builder.create_network(flags)


def describe_network_flags(*, strongly_typed: bool = False) -> str:
    """Human-readable flag summary for logs."""
    import tensorrt as trt

    parts: list[str] = []
    if getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None) is not None:
        parts.append("EXPLICIT_BATCH")
    else:
        parts.append("explicit-batch(default)")
    if strongly_typed:
        if getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None) is not None:
            parts.append("STRONGLY_TYPED")
        else:
            parts.append("STRONGLY_TYPED(unavailable)")
    return " | ".join(parts)


_BUILDER_FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    "FP16": ("FP16",),
    "BF16": ("BF16",),
    "FP8": ("FP8",),
    "INT4": ("INT4",),
    "CUDA_GRAPH": ("CUDA_GRAPH",),
    "OBEY_PRECISION_CONSTRAINTS": ("OBEY_PRECISION_CONSTRAINTS",),
    "PREFER_PRECISION_CONSTRAINTS": ("PREFER_PRECISION_CONSTRAINTS",),
}


def set_builder_flag_if_present(config, flag_name: str, *, log: logging.Logger | None = None) -> bool:
    """Set a ``BuilderFlag`` when the current TensorRT Python bindings expose it."""
    import tensorrt as trt

    log = log or logger
    candidates = _BUILDER_FLAG_ALIASES.get(flag_name, (flag_name,))
    for name in candidates:
        flag = getattr(trt.BuilderFlag, name, None)
        if flag is not None:
            config.set_flag(flag)
            return True
    log.warning("BuilderFlag.%s not available in this TensorRT build; skipping", flag_name)
    return False


def apply_precision_constraints_policy(
    config,
    policy: str | None,
    *,
    log: logging.Logger | None = None,
) -> None:
    """Apply ``precision_constraints`` when supported (TRT 8/9; optional on TRT 10+)."""
    log = log or logger
    pol = str(policy or "").strip().lower()
    if not pol:
        return
    if pol == "obey":
        if set_builder_flag_if_present(config, "OBEY_PRECISION_CONSTRAINTS", log=log):
            log.info("Enabled OBEY_PRECISION_CONSTRAINTS")
        else:
            log.info("precision_constraints='obey' ignored (flag unavailable on this TensorRT)")
        return
    if pol == "prefer":
        if set_builder_flag_if_present(config, "PREFER_PRECISION_CONSTRAINTS", log=log):
            log.info("Enabled PREFER_PRECISION_CONSTRAINTS")
        else:
            log.info("precision_constraints='prefer' ignored (flag unavailable on this TensorRT)")
        return
    raise ValueError(
        f"Unknown precision_constraints policy: {policy!r} (expected 'obey' or 'prefer')"
    )
