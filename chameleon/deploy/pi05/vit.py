"""Pi05 SigLIP ViT ONNX 导出。"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from chameleon.deploy.pi05.onnx_utils import force_vision_eager_attention, sdp_math_backend_only

logger = logging.getLogger(__name__)


class Pi05VitExport(nn.Module):
    def __init__(self, config, vision_tower, multi_modal_projector):
        super().__init__()
        self.config = config
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

    def forward(self, pixel_values):
        image_outputs = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(image_outputs.last_hidden_state)
        hidden = self.config.text_config.hidden_size
        return image_features / (hidden**0.5)

    @classmethod
    def from_pi05_model(cls, pi05_model) -> "Pi05VitExport":
        pwe = pi05_model.paligemma_with_expert.paligemma
        return cls(pwe.config, pwe.model.vision_tower, pwe.model.multi_modal_projector)


def export_vit(pi05_model, export_dir: str | Path, *, dynamo: bool = True) -> Path:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / "vit.onnx"

    model = Pi05VitExport.from_pi05_model(pi05_model).eval().cuda()
    pixel_values = torch.randn((1, 3, 224, 224), dtype=torch.float32, device="cuda")

    start = time.time()
    logger.info("Exporting vit.onnx -> %s", out_path)
    with torch.inference_mode():
        with force_vision_eager_attention(model.vision_tower):
            with sdp_math_backend_only():
                torch.onnx.export(
                    model,
                    (pixel_values,),
                    str(out_path),
                    input_names=["pixel_values"],
                    output_names=["image_features"],
                    opset_version=19,
                    dynamo=dynamo,
                    do_constant_folding=True,
                    dynamic_axes={
                        "pixel_values": {0: "batch_size"},
                        "image_features": {0: "batch_size"},
                    },
                )
    logger.info("vit.onnx export done in %.1fs", time.time() - start)
    return out_path
