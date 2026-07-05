# Cosmos3-Nano action vae_encode build 配置（NANO_ACTION 固定 profile）。
from chameleon.deploy.cosmos3.shapes import NANO_ACTION as _P

_video = (1, 3, _P.num_frames, _P.canvas_h, _P.canvas_w)

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": False,
    "workspace_mb": 8192,
    "min_shapes": {"video": _video},
    "opt_shapes": {"video": _video},
    "max_shapes": {"video": _video},
}
