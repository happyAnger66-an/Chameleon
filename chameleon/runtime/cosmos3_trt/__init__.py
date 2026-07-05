"""cosmos3 TRT 运行时包 — TRT 推理管线与编排器。

作用：
    导出 Cosmos3TrtPipeline（vae_encode → text_embed → dit 去噪环 ×CFG →
    vae_decode 的 TRT 内核）与 Cosmos3TrtOrchestrator（注册 key ``cosmos3_trt``）。

架构位置：
    运行时层 — 对照 runtime/pi05_trt/。import 时触发 orchestrator 注册。
"""

from chameleon.runtime.cosmos3_trt.pipeline import Cosmos3PolicyTrtPipeline, Cosmos3TrtPipeline

__all__ = ["Cosmos3PolicyTrtPipeline", "Cosmos3TrtPipeline"]
