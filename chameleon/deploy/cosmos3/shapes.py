"""cosmos3 deploy / TRT build 共享形状常量 — export 与 compile 须保持一致。

Cosmos3 的 MoT 联合序列长度随 prompt 长度 / 分辨率 / 帧数变化；TRT 部署必须固定
一个 profile。本文件提供两类常量：

1. **reference 小尺寸 MoT** 默认（``COSMOS3_TEXT_PREFIX_LEN`` 等）——供 reference
   路径 build_cfg / 冒烟使用，保持向后兼容。
2. **真实权重固定 profile**（:class:`Cosmos3Profile`）——``policy_droid`` 与
   ``nano_action`` 两套。真实权重部署时以 checkpoint 的 config 为准（VAE ``z_dim`` /
   ``scale_factor_*``、transformer ``latent_patch_size`` 等），下列数值为文档化默认，
   ``build_policy_pack`` 在加载 pipeline 后会用真实 config 复核并覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass

# --- reference 小尺寸 MoT 默认（向后兼容，勿删）---------------------------------
# 文本理解前缀（und 路径）token 长度 — 对应 max_lang_len，固定 profile 用。
COSMOS3_TEXT_PREFIX_LEN = 16
# 视频隐变量 token 数（gen 路径）= latent_t * latent_h * latent_w。
COSMOS3_VIDEO_TOKENS = 32
# action chunk 长度（action 模式 gen token 数）。
COSMOS3_ACTION_HORIZON = 16


@dataclass(frozen=True)
class Cosmos3Profile:
    """固定 TRT profile — 锁死 export / build / runtime 三处的所有 shape。

    改任一字段都需 **重新 export + 重新 build 全部 stage**（与 pi05 ``PI05_LIBERO_PREFIX_LEN``
    同理）。``text_prefix_len`` 为 JSON caption 分词后的固定长度（不足右侧 padding，超过
    截断）；``latent_*`` 为 Wan VAE 输出（不做 padding 移除，full-canvas，保证 encode 输出
    与 dit 输入 shape 恒定）。
    """

    name: str
    model_id: str
    mode: str  # "action"
    action_mode: str  # "policy"
    domain_name: str
    chunk_size: int
    num_frames: int  # pixel canvas 时间长度 = chunk_size + 1
    resolution_tier: int
    canvas_h: int  # tier canvas 像素高
    canvas_w: int  # tier canvas 像素宽
    fps: float
    guidance_scale: float
    num_inference_steps: int
    # transformer / VAE 派生量（Cosmos3-Nano 族：latent_channel=48, action_dim=64）。
    # 加载 pipeline 后 ``build_policy_pack`` 会用 checkpoint config 复核；若与下列值
    # 不一致会打 warning 并以 checkpoint 为准 export，此时 build_cfg 须同步更新。
    latent_channels: int  # transformer latent_channel / VAE z_dim（MoT 输入 C）
    latent_patch_size: int  # transformer latent_patch_size（空间 patch 化）
    scale_factor_temporal: int
    scale_factor_spatial: int
    action_dim: int  # 模型 action 头宽度（raw_action_dim pad 到此）
    raw_action_dim: int  # embodiment 原始动作维度（输出裁剪到此）
    text_prefix_len: int  # JSON caption 固定分词长度

    @property
    def latent_t(self) -> int:
        return (self.num_frames - 1) // self.scale_factor_temporal + 1

    @property
    def latent_h(self) -> int:
        return self.canvas_h // self.scale_factor_spatial

    @property
    def latent_w(self) -> int:
        return self.canvas_w // self.scale_factor_spatial

    @property
    def patch_h(self) -> int:
        return -(-self.latent_h // self.latent_patch_size)  # ceil

    @property
    def patch_w(self) -> int:
        return -(-self.latent_w // self.latent_patch_size)

    @property
    def num_vision_tokens(self) -> int:
        return self.latent_t * self.patch_h * self.patch_w

    @property
    def sequence_length(self) -> int:
        return self.text_prefix_len + self.num_vision_tokens + self.chunk_size


# DROID concat 观测常见宽高比 ~1.73 → tier 480 canvas 480×832（H×W）。
POLICY_DROID = Cosmos3Profile(
    name="policy_droid",
    model_id="nvidia/Cosmos3-Nano-Policy-DROID",
    mode="action",
    action_mode="policy",
    domain_name="droid_lerobot",
    chunk_size=16,
    num_frames=17,
    resolution_tier=480,
    canvas_h=480,
    canvas_w=832,
    fps=24.0,
    guidance_scale=1.0,
    num_inference_steps=30,
    latent_channels=48,
    latent_patch_size=2,
    scale_factor_temporal=4,
    scale_factor_spatial=16,
    action_dim=64,
    raw_action_dim=10,
    text_prefix_len=256,
)

# Nano action（bridge domain）复用同一 export 代码，profile 尺寸不同。
NANO_ACTION = Cosmos3Profile(
    name="nano_action",
    model_id="nvidia/Cosmos3-Nano",
    mode="action",
    action_mode="policy",
    domain_name="bridge_orig_lerobot",
    chunk_size=16,
    num_frames=17,
    resolution_tier=480,
    canvas_h=480,
    canvas_w=832,
    fps=24.0,
    guidance_scale=1.0,
    num_inference_steps=35,
    latent_channels=48,
    latent_patch_size=2,
    scale_factor_temporal=4,
    scale_factor_spatial=16,
    action_dim=64,
    raw_action_dim=10,
    text_prefix_len=256,
)

_PROFILES: dict[str, Cosmos3Profile] = {
    POLICY_DROID.name: POLICY_DROID,
    NANO_ACTION.name: NANO_ACTION,
}


def get_profile(name: str) -> Cosmos3Profile:
    if name not in _PROFILES:
        raise KeyError(f"Unknown cosmos3 profile {name!r}; expected one of {sorted(_PROFILES)}.")
    return _PROFILES[name]


def dit_trt_dynamic_shapes(profile: Cosmos3Profile) -> dict[str, tuple[int, ...]]:
    """TRT build 用 dit 动态输入 shape（须与 export 的 ONNX 静态维一致）。"""
    num_noisy_vision = (profile.latent_t - 1) * profile.patch_h * profile.patch_w
    return {
        "vision_tokens": (1, profile.latent_channels, profile.latent_t, profile.latent_h, profile.latent_w),
        "vision_timesteps": (num_noisy_vision,),
        "action_tokens": (profile.chunk_size, profile.action_dim),
        "action_timesteps": (profile.chunk_size,),
    }


def infer_onnx_static_input_shapes(onnx_path: str) -> dict[str, tuple[int, ...]]:
    """读取 ONNX 输入的静态维度（全为常量的轴）。"""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    shapes: dict[str, tuple[int, ...]] = {}
    for inp in model.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value <= 0:
                break
            dims.append(int(d.dim_value))
        else:
            shapes[inp.name] = tuple(dims)
    return shapes


def validate_build_cfg_matches_onnx(onnx_path: str, build_cfg: dict, *, stage: str) -> None:
    """build 前校验 build_cfg profile 与 ONNX 静态输入维一致，避免 TRT kMIN 报错。"""
    onnx_shapes = infer_onnx_static_input_shapes(onnx_path)
    expected = build_cfg.get("opt_shapes") or build_cfg.get("min_shapes") or {}
    mismatches: list[str] = []
    for name, shape in expected.items():
        if name not in onnx_shapes:
            continue
        exp = tuple(int(x) for x in shape)
        got = onnx_shapes[name]
        if exp != got:
            mismatches.append(f"{name}: build_cfg={exp} onnx={got}")
    if mismatches:
        raise ValueError(
            f"cosmos3 stage {stage!r} build_cfg shapes do not match exported ONNX {onnx_path}: "
            + "; ".join(mismatches)
            + ". Re-export with the correct profile or update build_cfg / shapes.py."
        )
