# Cosmos3 Policy-DROID dit_step (dit.onnx) TensorRT build 配置 — 去噪环热点、最大 engine。
# 固定 profile（见 chameleon/deploy/cosmos3/shapes.py: POLICY_DROID）；所有 shape 锁死，
# 须与 Cosmos3DitStepExport 导出的静态 pack + 动态样例完全一致。
#
# 动态输入（每 step）：
#   vision_tokens    [1, C, latent_t, latent_h, latent_w]
#   vision_timesteps [num_noisy_vision_tokens]
#   action_tokens    [chunk_size, action_dim]
#   action_timesteps [num_noisy_action_tokens]
# 输出 v_vision / v_action 为 flow velocity（host 做 mask / scheduler.step）。

from chameleon.deploy.cosmos3.shapes import POLICY_DROID as _P

_num_noisy_vision = (_P.latent_t - 1) * _P.patch_h * _P.patch_w  # policy: latent frame 0 clean
_num_noisy_action = _P.chunk_size  # policy: all action tokens noisy

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
