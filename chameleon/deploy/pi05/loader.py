"""加载 openpi pi05 PyTorch 模型（供 ONNX 导出使用）。"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_pi05_model(
    checkpoint_dir: str,
    train_config: str | None = None,
    *,
    device: str = "cpu",
):
    """从 checkpoint 目录加载 ``PI0Pytorch``（含权重与 norm 配置）。

    默认在 CPU 上加载并做 selective bf16，避免 12GB 级 GPU 在导出多 stage 时 OOM。
    各 stage exporter 会临时把所需子模块搬到 CUDA，导出后由
    :func:`chameleon.deploy.pi05.memory.release_export_cuda_memory` 回收。
    """
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    config_name = train_config or "pi05_libero"
    ckpt = Path(checkpoint_dir).expanduser().resolve()
    logger.info(
        "Loading pi05 checkpoint=%s train_config=%s device=%s",
        ckpt,
        config_name,
        device,
    )
    cfg = openpi_config.get_config(config_name)
    policy = policy_config.create_trained_policy(cfg, str(ckpt), pytorch_device=device)
    model = policy._model
    model.eval()
    return model
