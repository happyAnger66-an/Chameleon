# Cosmos3 Policy-DROID text_embed (text_embed.onnx) TensorRT build 配置。
# transformer.embed_tokens 的 embedding lookup。输入 input_ids [text_prefix_len]（int64）。
# JSON caption / tokenize / _prepare_text_segment 仍在 host；TRT 只做 embedding。

from chameleon.deploy.cosmos3.shapes import POLICY_DROID as _P

_ids = (_P.text_prefix_len,)

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": False,
    "workspace_mb": 2048,
    "min_shapes": {"input_ids": _ids},
    "opt_shapes": {"input_ids": _ids},
    "max_shapes": {"input_ids": _ids},
}
