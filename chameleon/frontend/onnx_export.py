"""ONNX graph capture using ``torch.onnx.export``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from chameleon.core.artifact import Artifact
from chameleon.frontend.base import GraphCapture, register_graph_capture


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
            with torch.no_grad():
                try:
                    torch.onnx.export(module, tuple(example_inputs), str(path), **kwargs)
                except Exception:  # noqa: BLE001
                    # The dynamo exporter's optimizer can choke on some ops
                    # (e.g. nn.MultiheadAttention). Retry with the legacy path.
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
