# Cosmos3 text_embed (text_embed.onnx) TensorRT build 配置。
#   lang_tokens (B, max_lang_len) int64 -> text_mem (B, max_lang_len, hidden_size)
# 固定文本前缀长度（COSMOS3_TEXT_PREFIX_LEN）；真实权重部署时按 tokenizer 上限调整。

from chameleon.deploy.cosmos3.shapes import COSMOS3_TEXT_PREFIX_LEN

_BATCH = 1
_LEN = COSMOS3_TEXT_PREFIX_LEN

build_cfg = {
    "precision": "fp32",
    "strongly_typed_network": False,
    "workspace_mb": 4096,
    "min_shapes": {"lang_tokens": (_BATCH, _LEN)},
    "opt_shapes": {"lang_tokens": (_BATCH, _LEN)},
    "max_shapes": {"lang_tokens": (_BATCH, _LEN)},
}
