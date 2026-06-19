"""导出阶段 GPU 显存回收。"""

from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def release_export_cuda_memory(pi05_model=None) -> None:
    """Move the full pi05 model back to CPU and drop cached CUDA allocations."""
    if pi05_model is not None:
        try:
            pi05_model.cpu()
        except Exception as exc:  # noqa: BLE001
            logger.debug("pi05_model.cpu() failed: %s", exc)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
