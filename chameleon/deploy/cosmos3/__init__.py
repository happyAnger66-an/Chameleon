"""cosmos3 deploy 包 — 分阶段 ONNX 导出与 TRT engine 构建支撑。

作用：
    提供 vae_encode / text_embed / dit / vae_decode 四个 stage 的 ONNX 导出器
    （loader/vae/dit/sound/export）与固定 profile 形状常量（shapes），供
    deploy/cosmos3_diffusers.py 的 run_cosmos3_export / run_cosmos3_build 使用。

架构位置：
    部署层 — 对照 deploy/pi05/。导出从 Cosmos3Adapter 的 stage_module 取模块，
    用 stage_example_inputs / stage_io_names 生成 torch.onnx.export 输入，因此
    reference 模型可在 CPU 上离线导出（无需 diffusers 权重）。
"""

from chameleon.deploy.cosmos3.export import COSMOS3_STAGES, export_cosmos3_stages, export_stage

__all__ = ["COSMOS3_STAGES", "export_cosmos3_stages", "export_stage"]
