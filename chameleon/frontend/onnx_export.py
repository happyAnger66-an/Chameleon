"""ONNX 图捕获 — 基于 torch.onnx.export 的前端实现。

作用：
    实现 OnnxExport GraphCapture：支持 dynamo 导出失败时回退 legacy
    TorchScript 导出器；对 modelopt 量化模块自动进入 export_torch_mode
    以生成 QDQ 节点。

架构位置：
    优化/编译流水线 — frontend 层的默认实现，注册为 "onnx" 键。
    上游：models 的 stage nn.Module；下游：compile/tensorrt 等编译后端。
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Sequence

import torch

from chameleon.core.artifact import Artifact
from chameleon.frontend.base import GraphCapture, register_graph_capture

logger = logging.getLogger(__name__)


def _modelopt_export_ctx(module: Any):
    """Enter modelopt's export mode when the module has been quantized by modelopt.

    modelopt fake-quant ops only translate to ONNX QDQ nodes inside this context;
    otherwise the (dynamo) exporter fails on the custom autograd functions. No-op
    when modelopt is absent or the module is not quantized.
    """
    try:
        from modelopt.torch.quantization.nn import TensorQuantizer
        from modelopt.torch.quantization.utils import export_torch_mode

        has_quantizer = any(isinstance(m, TensorQuantizer) for m in module.modules())
        if has_quantizer:
            return export_torch_mode()
    except Exception as exc:  # noqa: BLE001
        logger.debug("modelopt export mode unavailable: %s", exc)
    return contextlib.nullcontext()


class OnnxExport(GraphCapture):
    name = "onnx"

    def __init__(self, opset: int = 17) -> None:
        self.opset = opset

    def capture(
        self,
        module: Any,
        example_inputs: Sequence[Any],
        *,
        stage: str,
        output_path: str | None = None,
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        dynamic_axes: dict[str, Any] | None = None,
    ) -> Artifact:
        if output_path is None:
            raise ValueError("OnnxExport requires an output_path.")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        was_training = getattr(module, "training", False)
        if hasattr(module, "eval"):
            module.eval()
        kwargs = dict(
            input_names=list(input_names) if input_names else None,
            output_names=list(output_names) if output_names else None,
            dynamic_axes=dynamic_axes,
            opset_version=self.opset,
        )
        try:
            with torch.no_grad(), _modelopt_export_ctx(module):
                try:
                    torch.onnx.export(module, tuple(example_inputs), str(path), **kwargs)
                except Exception:  # noqa: BLE001
                    # The dynamo exporter's optimizer can choke on some ops
                    # (e.g. nn.MultiheadAttention) and on modelopt fake-quant.
                    # Retry with the legacy TorchScript exporter.
                    torch.onnx.export(
                        module, tuple(example_inputs), str(path), dynamo=False, **kwargs
                    )
        finally:
            if was_training and hasattr(module, "train"):
                module.train()

        return Artifact(
            kind="onnx",
            stage=stage,
            path=str(path),
            metadata={"opset": self.opset, "input_names": list(input_names or [])},
        )


register_graph_capture(OnnxExport(), override=True)
