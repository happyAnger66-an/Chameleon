# Cosmos3-Nano action dit_step build 配置（NANO_ACTION 固定 profile）— 复用同一 export 代码。
# v1: guidance=1 单路 dit。CFG(guidance>1) 扩展见 docs/models/cosmos3_trt_deploy.md（v2）。
from chameleon.deploy.cosmos3.shapes import NANO_ACTION as _P, dit_trt_dynamic_shapes

_shapes = dit_trt_dynamic_shapes(_P)

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": False,
    "workspace_mb": 16384,
    "min_shapes": dict(_shapes),
    "opt_shapes": dict(_shapes),
    "max_shapes": dict(_shapes),
}
