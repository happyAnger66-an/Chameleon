"""ModelOpt PTQ 量化方法 — 封装 NVIDIA TensorRT Model Optimizer。

作用：
    注册 int8 / int8_sq / fp8 / int4_awq / w4a8_awq / nvfp4 六种方法，
    映射到 modelopt 配置。modelopt 不可用时降级为 metadata-only（权重
    保持全精度但产出 QuantMetadata），保证 CPU 开发机可跑通流水线。

架构位置：
    优化/编译流水线 — quantization/methods/ 的具体实现，被
    api.run_quantize 按 TaskConfig.quantize 步骤调用。
"""

from __future__ import annotations

import logging
from typing import Any

from chameleon.core.platform import PlatformSpec
from chameleon.quantization.base import QuantConfig, QuantMetadata, QuantMethod
from chameleon.quantization.calibrate.base import Calibrator
from chameleon.quantization.registry import register_quant_method

logger = logging.getLogger(__name__)

# Chameleon method name -> (modelopt config attribute, weight dtype, activation dtype).
_RECIPES = {
    "int8": ("INT8_DEFAULT_CFG", "int8", "int8"),
    "int8_sq": ("INT8_SMOOTHQUANT_CFG", "int8", "int8"),
    "fp8": ("FP8_DEFAULT_CFG", "fp8", "fp8"),
    "int4_awq": ("INT4_AWQ_CFG", "int4", "fp16"),
    "w4a8_awq": ("W4A8_AWQ_BETA_CFG", "int4", "fp8"),
    "nvfp4": ("NVFP4_DEFAULT_CFG", "nvfp4", "nvfp4"),
}


class ModelOptQuantMethod(QuantMethod):
    def __init__(self, name: str, modelopt_cfg_attr: str, weight_dtype: str, act_dtype: str) -> None:
        self.name = name
        self._cfg_attr = modelopt_cfg_attr
        self._weight_dtype = weight_dtype
        self._act_dtype = act_dtype

    def quantize(
        self,
        module: Any,
        calibrator: Calibrator,
        platform: PlatformSpec,
        config: QuantConfig,
    ) -> tuple[Any, QuantMetadata]:
        metadata = QuantMetadata(
            method=self.name,
            component_dtypes={
                "weight": self._weight_dtype,
                "activation": self._act_dtype,
                **({"kv_cache": config.kv_cache_dtype} if config.kv_cache_dtype else {}),
            },
            extra={"modelopt_cfg": self._cfg_attr, "platform": platform.name},
        )
        try:
            import modelopt.torch.quantization as mtq  # type: ignore

            quant_cfg = getattr(mtq, self._cfg_attr)
            module = mtq.quantize(module, quant_cfg, forward_loop=calibrator.forward_loop)
            metadata.extra["applied"] = True
        except Exception as exc:  # noqa: BLE001 - graceful CPU-dev fallback
            logger.warning(
                "modelopt unavailable or failed for method %r (%s); emitting "
                "metadata-only quantization.",
                self.name,
                exc,
            )
            # Still drive the calibrator so the forward path is exercised.
            try:
                calibrator.forward_loop(module)
            except Exception:  # noqa: BLE001
                pass
            metadata.extra["applied"] = False
        return module, metadata


def _register_all() -> None:
    for name, (cfg_attr, w_dtype, a_dtype) in _RECIPES.items():
        register_quant_method(
            ModelOptQuantMethod(name, cfg_attr, w_dtype, a_dtype), override=True
        )


_register_all()
