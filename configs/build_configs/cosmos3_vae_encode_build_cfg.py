# Cosmos3 vae_encode (vae_encode.onnx) TensorRT build 配置。
# 须与导出 ONNX 的 input_names / 张量秩一致。默认对应 reference 模型尺寸：
#   cond_pixels (B, 3, image_size, image_size) -> cond_latent (B, num_video_tokens, token_dim)
# 真实 Wan VAE 权重部署时同步修改下列形状与 precision。

_BATCH = 1
_CHANNELS = 3
_IMG = 64

build_cfg = {
    "precision": "fp32",
    "strongly_typed_network": False,
    "workspace_mb": 4096,
    "min_shapes": {"cond_pixels": (_BATCH, _CHANNELS, _IMG, _IMG)},
    "opt_shapes": {"cond_pixels": (_BATCH, _CHANNELS, _IMG, _IMG)},
    "max_shapes": {"cond_pixels": (_BATCH, _CHANNELS, _IMG, _IMG)},
}
