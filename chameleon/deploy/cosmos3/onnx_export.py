"""cosmos3 通用 ONNX 导出 helper — 从 adapter stage 模块导出单个子图。"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_stage_module(
    adapter,
    stage: str,
    export_dir: str | Path,
    onnx_name: str,
    *,
    device: str = "cpu",
    opset_version: int = 19,
    dynamo: bool = False,
    do_constant_folding: bool = True,
) -> Path:
    """torch.onnx.export a single cosmos3 stage using the adapter's hooks.

    Inputs / names / dynamic batch axis are derived from
    ``adapter.stage_example_inputs`` and ``adapter.stage_io_names`` so the export
    stays in lockstep with the runtime stage signatures.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / onnx_name

    module = adapter.stage_module(stage)
    if hasattr(module, "to"):
        module = module.to(device)
    if hasattr(module, "eval"):
        module.eval()

    obs = adapter.example_observation(1, device=device)
    example_inputs = adapter.stage_example_inputs(stage, obs)
    input_names, output_names = adapter.stage_io_names(stage)

    dynamic_axes = {}
    for name in (input_names or []):
        dynamic_axes[name] = {0: "batch_size"}
    for name in (output_names or []):
        dynamic_axes[name] = {0: "batch_size"}

    start = time.time()
    logger.info("Exporting cosmos3 %s -> %s", stage, out_path)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            tuple(example_inputs),
            str(out_path),
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=dynamo,
            do_constant_folding=do_constant_folding,
            dynamic_axes=dynamic_axes or None,
        )
    logger.info("cosmos3 %s export done in %.1fs", stage, time.time() - start)
    if not out_path.is_file():
        raise FileNotFoundError(f"ONNX export finished but file missing: {out_path}")
    return out_path
