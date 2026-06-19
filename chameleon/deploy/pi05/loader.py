"""加载 openpi pi05 PyTorch 模型（供 ONNX 导出使用）。"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_pi05_model(checkpoint_dir: str, train_config: str | None = None):
    """从 checkpoint 目录加载 ``PI0Pytorch``（含权重与 norm 配置）。"""
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    config_name = train_config or "pi05_libero"
    logger.info("Loading pi05 checkpoint=%s train_config=%s", checkpoint_dir, config_name)
    cfg = openpi_config.get_config(config_name)
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir)
    return policy._model
