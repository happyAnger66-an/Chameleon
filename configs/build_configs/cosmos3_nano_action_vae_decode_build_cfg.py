# Cosmos3-Nano action vae_decode build 配置（NANO_ACTION 固定 profile）。
from chameleon.deploy.cosmos3.shapes import NANO_ACTION as _P

_latent = (1, _P.latent_channels, _P.latent_t, _P.latent_h, _P.latent_w)

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": False,
    "workspace_mb": 8192,
    "min_shapes": {"latent": _latent},
    "opt_shapes": {"latent": _latent},
    "max_shapes": {"latent": _latent},
}
