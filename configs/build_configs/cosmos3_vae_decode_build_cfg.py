# Cosmos3 vae_decode (vae_decode.onnx) TensorRT build 配置。
#   latent (B, num_video_tokens, token_dim) -> video (B, T, C, H, W)
# 真实 Wan VAE 权重部署时同步修改形状与 precision。

from chameleon.deploy.cosmos3.shapes import COSMOS3_VIDEO_TOKENS

_BATCH = 1
_TOKENS = COSMOS3_VIDEO_TOKENS
_TOKEN_DIM = 32

build_cfg = {
    "precision": "fp32",
    "strongly_typed_network": False,
    "workspace_mb": 4096,
    "min_shapes": {"latent": (_BATCH, _TOKENS, _TOKEN_DIM)},
    "opt_shapes": {"latent": (_BATCH, _TOKENS, _TOKEN_DIM)},
    "max_shapes": {"latent": (_BATCH, _TOKENS, _TOKEN_DIM)},
}
