# Cosmos3-Nano action dit_step build 配置（NANO_ACTION 固定 profile）— 复用同一 export 代码。
# v1: guidance=1 单路 dit。CFG(guidance>1) 扩展见 docs/models/cosmos3_trt_deploy.md（v2）。
from chameleon.deploy.cosmos3.shapes import NANO_ACTION as _P

_num_noisy_vision = (_P.latent_t - 1) * _P.patch_h * _P.patch_w
_num_noisy_action = _P.chunk_size

_shapes = {
    "vision_tokens": (1, _P.latent_channels, _P.latent_t, _P.latent_h, _P.latent_w),
    "vision_timesteps": (_num_noisy_vision,),
    "action_tokens": (_P.chunk_size, _P.action_dim),
    "action_timesteps": (_num_noisy_action,),
}

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": False,
    "workspace_mb": 16384,
    "min_shapes": dict(_shapes),
    "opt_shapes": dict(_shapes),
    "max_shapes": dict(_shapes),
}
