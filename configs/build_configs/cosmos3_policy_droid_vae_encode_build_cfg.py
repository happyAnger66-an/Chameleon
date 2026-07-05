# Cosmos3 Policy-DROID vae_encode (vae_encode.onnx) TensorRT build 配置。
# 固定 profile（见 chameleon/deploy/cosmos3/shapes.py: POLICY_DROID）。改 profile 须
# 重新 export + build 全部 stage。输入 video [1,3,num_frames,canvas_h,canvas_w]。

from chameleon.deploy.cosmos3.shapes import POLICY_DROID as _P

_video = (1, 3, _P.num_frames, _P.canvas_h, _P.canvas_w)

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": False,
    "workspace_mb": 8192,
    "min_shapes": {"video": _video},
    "opt_shapes": {"video": _video},
    "max_shapes": {"video": _video},
}
