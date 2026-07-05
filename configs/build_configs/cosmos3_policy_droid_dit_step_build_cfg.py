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

from chameleon.deploy.cosmos3.shapes import POLICY_DROID as _P, dit_trt_dynamic_shapes

_shapes = dit_trt_dynamic_shapes(_P)

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": False,
    "workspace_mb": 16384,
    "min_shapes": dict(_shapes),
    "opt_shapes": dict(_shapes),
    "max_shapes": dict(_shapes),
}
