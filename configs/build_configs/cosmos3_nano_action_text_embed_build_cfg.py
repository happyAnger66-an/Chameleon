# Cosmos3-Nano action text_embed build 配置（NANO_ACTION 固定 profile）。
from chameleon.deploy.cosmos3.shapes import NANO_ACTION as _P

_ids = (_P.text_prefix_len,)

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": False,
    "workspace_mb": 2048,
    "min_shapes": {"input_ids": _ids},
    "opt_shapes": {"input_ids": _ids},
    "max_shapes": {"input_ids": _ids},
}
