"""cosmos3 部署侧模型加载 — 复用 Cosmos3Adapter 构建 stage 模块。

导出器直接从 adapter 取 stage_module / stage_example_inputs / stage_io_names，
因此 reference 模型可在 CPU 上离线导出 ONNX（无需 diffusers 权重）；真实权重时
设置 model_overrides.use_reference=false，adapter 会加载 diffusers transformer/vae。
"""

from __future__ import annotations

import logging

from chameleon.config.schema import TaskConfig
from chameleon.models.cosmos3.adapter import Cosmos3Adapter

logger = logging.getLogger(__name__)


def load_cosmos3_adapter(task: TaskConfig, device: str = "cpu") -> Cosmos3Adapter:
    """Build a Cosmos3Adapter (reference or diffusers) on ``device`` for export/build."""
    config = Cosmos3Adapter.make_config(task.model_overrides)
    adapter = Cosmos3Adapter(config).build(device)
    logger.info(
        "Loaded cosmos3 adapter for deploy (reference=%s, device=%s).",
        config.use_reference,
        device,
    )
    return adapter
