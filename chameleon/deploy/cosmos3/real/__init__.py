"""cosmos3 真实权重 Export 子包 — 从 diffusers ``Cosmos3OmniPipeline`` 导出可 TRT 化子图。

作用：
    与 reference 路径（``deploy/cosmos3/*.py``）并列，提供真实权重的 ONNX 导出封装：

    - :mod:`vae`      — Wan VAE encode/decode + 数据集级 latent 归一化 / 反归一化
    - :mod:`pack`     — host 侧联合序列静态打包（policy：text + vision + action）
    - :mod:`dit_step` — MoT 单步去噪 Export wrapper（静态字段 buffer 化，动态仅 latent/timestep）

架构位置：
    部署层 — 当 ``model_overrides.use_reference=false`` 时由 export.py 选用；输入尺寸由
    :mod:`chameleon.deploy.cosmos3.shapes` 的固定 profile 决定，须与 build_cfg 对齐。
"""

from chameleon.deploy.cosmos3.real.dit_step import Cosmos3DitStepExport
from chameleon.deploy.cosmos3.real.pack import Cosmos3PolicyPack, build_policy_pack
from chameleon.deploy.cosmos3.real.vae import WanVaeDecodeExport, WanVaeEncodeExport

__all__ = [
    "Cosmos3DitStepExport",
    "Cosmos3PolicyPack",
    "build_policy_pack",
    "WanVaeDecodeExport",
    "WanVaeEncodeExport",
]
