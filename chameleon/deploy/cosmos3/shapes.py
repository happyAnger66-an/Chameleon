"""cosmos3 deploy / TRT build 共享形状常量 — export 与 compile 须保持一致。

Cosmos3 的 MoT 联合序列长度随 prompt 长度 / 分辨率 / 帧数变化；TRT 部署必须固定
一个 profile。下列常量给出参考路径（小尺寸 MoT）的默认值，真实权重部署时按
checkpoint 的 config 同步修改并对齐 build_cfg。
"""

from __future__ import annotations

# 文本理解前缀（und 路径）token 长度 — 对应 max_lang_len，固定 profile 用。
COSMOS3_TEXT_PREFIX_LEN = 16

# 视频隐变量 token 数（gen 路径）= latent_t * latent_h * latent_w。
COSMOS3_VIDEO_TOKENS = 32

# action chunk 长度（action 模式 gen token 数）。
COSMOS3_ACTION_HORIZON = 16
