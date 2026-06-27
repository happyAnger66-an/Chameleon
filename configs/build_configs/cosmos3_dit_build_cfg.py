# Cosmos3 dit (dit.onnx) — MoT 联合 transformer 单步 forward，去噪环热点。
# 输入（与 Cosmos3Dit.forward 一致）：
#   text_mem    (B, text_len, hidden)    文本理解前缀（und 路径）
#   cond_latent (B, num_video_tokens, token_dim)  条件 VAE 隐变量
#   x_t         (B, gen_tokens, token_dim)         当前噪声隐变量（gen 路径）
#   time_emb    (B, hidden)                        flow 时间 embedding
# 输出 v_t (B, gen_tokens, token_dim) 为 flow velocity。
#
# 联合序列长度随分辨率/帧数变化 → TRT 须固定一个 profile。默认按 video 模式
# （gen_tokens=num_video_tokens）。action 模式 gen_tokens=action_horizon，需另建 profile。

from chameleon.deploy.cosmos3.shapes import (
    COSMOS3_TEXT_PREFIX_LEN,
    COSMOS3_VIDEO_TOKENS,
)

_BATCH = 1
_TEXT_LEN = COSMOS3_TEXT_PREFIX_LEN
_HIDDEN = 128
_TOKEN_DIM = 32
_COND_TOKENS = COSMOS3_VIDEO_TOKENS
_GEN_TOKENS = COSMOS3_VIDEO_TOKENS

_shapes = {
    "text_mem": (_BATCH, _TEXT_LEN, _HIDDEN),
    "cond_latent": (_BATCH, _COND_TOKENS, _TOKEN_DIM),
    "x_t": (_BATCH, _GEN_TOKENS, _TOKEN_DIM),
    "time_emb": (_BATCH, _HIDDEN),
}

build_cfg = {
    "precision": "fp32",
    "strongly_typed_network": False,
    "workspace_mb": 8192,
    "min_shapes": dict(_shapes),
    "opt_shapes": dict(_shapes),
    "max_shapes": dict(_shapes),
}
