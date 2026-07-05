# Cosmos3 Policy-DROID vae_decode (vae_decode.onnx) TensorRT build 配置。
# 固定 profile（见 chameleon/deploy/cosmos3/shapes.py: POLICY_DROID）。
# 输入 latent [1, latent_channels, latent_t, latent_h, latent_w]（归一化后 latent）。

from chameleon.deploy.cosmos3.shapes import POLICY_DROID as _P

_latent = (1, _P.latent_channels, _P.latent_t, _P.latent_h, _P.latent_w)

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": False,
    "workspace_mb": 8192,
    "min_shapes": {"latent": _latent},
    "opt_shapes": {"latent": _latent},
    "max_shapes": {"latent": _latent},
}
