"""Cosmos3 模型适配器 — 将 diffusers / 参考模型映射为 Chameleon stage 接口。

作用：
    实现 Cosmos3Adapter：支持 reference 路径（无需权重，CPU 冒烟）和真实
    diffusers 路径（``Cosmos3OmniPipeline.from_pretrained``）。通过
    _STAGE_ATTR / _DIFFUSERS_STAGE_ATTR 将 vae_encode / text_embed / dit /
    vae_decode 映射到具体 nn.Module 子模块，供 frontend 导出 / quantization /
    orchestrator 消费。两种生成模式（action / video）共用同一适配器。

架构位置：
    模型/架构层 — 被 api.build_adapter 实例化。reference 时使用三/四段式
    默认编排器（key ``cosmos3``）；真实模型时返回 ``cosmos3_real`` 走 diffusers
    pipeline 端到端。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from chameleon.architectures.cosmos3 import ARCHITECTURE_NAME
from chameleon.models.base import ModelAdapter, register_model
from chameleon.models.cosmos3.reference import Cosmos3Config, Cosmos3ReferenceModel

logger = logging.getLogger(__name__)

# Reference model: attributes match the stage names directly.
_STAGE_ATTR = {
    "vae_encode": "vae_encode",
    "text_embed": "text_embed",
    "dit": "dit",
    "vae_decode": "vae_decode",
}

# Real diffusers Cosmos3OmniPipeline: map each Chameleon stage to a real submodule.
# ``vae_encode`` / ``vae_decode`` both resolve to the Wan VAE (encode/decode picked
# at call time); ``dit`` is the MoT joint transformer; text embedding lives inside
# the transformer (``transformer.embed_tokens``).
_DIFFUSERS_STAGE_ATTR = {
    "vae_encode": ("vae",),
    "text_embed": ("transformer", "embed_tokens"),
    "dit": ("transformer",),
    "vae_decode": ("vae",),
}


def _resolve_attr_path(root, path: tuple[str, ...]):
    obj = root
    for attr in path:
        obj = getattr(obj, attr)
    return obj


def _ensure_huggingface_hub_compat() -> None:
    """Shim removed ``huggingface_hub.cached_download`` so older diffusers imports.

    Newer ``huggingface_hub`` (>=0.26) dropped ``cached_download`` which some
    diffusers builds still import at module load time (``cannot import name
    'cached_download' from 'huggingface_hub'``). Alias it to ``hf_hub_download``
    so ``import diffusers`` succeeds. Only patches the name when it is missing.
    """
    try:
        import huggingface_hub as hf
    except Exception:  # noqa: BLE001 - hf_hub always present in inference env
        return
    if not hasattr(hf, "cached_download"):
        fallback = getattr(hf, "hf_hub_download", None)
        if fallback is not None:
            hf.cached_download = fallback  # type: ignore[attr-defined]


class Cosmos3Adapter(ModelAdapter):
    architecture = ARCHITECTURE_NAME

    def __init__(self, config: Cosmos3Config | None = None) -> None:
        super().__init__(config or Cosmos3Config())
        self.model: Cosmos3ReferenceModel | Any | None = None
        self.pipeline: Any | None = None
        self._device = "cpu"
        self._is_real_diffusers = False
        self._diffusers_error: str | None = None

    @classmethod
    def make_config(cls, overrides: dict[str, Any] | None = None) -> Cosmos3Config:
        valid = {f for f in Cosmos3Config.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in (overrides or {}).items() if k in valid}
        return Cosmos3Config(**filtered)

    def build(self, device: str = "cpu") -> "Cosmos3Adapter":
        self._device = device
        if self.config.use_reference:
            self.model = Cosmos3ReferenceModel(self.config).to(device).eval()
        else:
            self.model = self._load_diffusers(device)
        return self

    def _load_diffusers(self, device: str):
        """Load the real diffusers ``Cosmos3OmniPipeline`` (+ keep a transformer/vae handle).

        失败时回退到参考模型并告警，保证流水线不硬失败（对照 pi05 adapter）。
        """
        try:
            _ensure_huggingface_hub_compat()
            from diffusers import Cosmos3OmniPipeline

            dtype = torch.bfloat16 if self.config.precision == "bfloat16" else torch.float32
            source = self.config.checkpoint or self.config.model_id
            pipe = Cosmos3OmniPipeline.from_pretrained(
                source,
                torch_dtype=dtype,
                enable_safety_checker=self.config.enable_safety_checker,
            )
            pipe = pipe.to(device)
            self.pipeline = pipe
            self._is_real_diffusers = True
            return pipe.transformer
        except Exception as exc:  # noqa: BLE001 - graceful MVP fallback
            logger.warning(
                "Falling back to cosmos3 reference model (could not load diffusers pipeline: %s)",
                exc,
            )
            self.config.use_reference = True
            self._is_real_diffusers = False
            self.pipeline = None
            self._diffusers_error = f"{type(exc).__name__}: {exc}"
            return Cosmos3ReferenceModel(self.config).to(device).eval()

    def stage_module(self, stage: str):
        if self.model is None:
            raise RuntimeError("Call build() before stage_module().")
        if self._is_real_diffusers:
            path = _DIFFUSERS_STAGE_ATTR.get(stage)
            if path is None:
                raise KeyError(f"Unknown cosmos3 stage {stage!r}.")
            return _resolve_attr_path(self.pipeline, path)
        attr = _STAGE_ATTR.get(stage)
        if attr is None:
            raise KeyError(f"Unknown cosmos3 stage {stage!r}.")
        return getattr(self.model, attr)

    @property
    def time_embed_dim(self) -> int:
        """Dimension of the sinusoidal time embedding fed to the dit denoise step."""
        return int(self.config.hidden_size)

    @property
    def mode(self) -> str:
        return str(getattr(self.config, "mode", "video"))

    @property
    def orchestrator_key(self) -> str | None:
        # 真实 diffusers 模型走 Cosmos3OmniPipeline 端到端；参考模型用四段式默认编排器。
        return "cosmos3_real" if self._is_real_diffusers else None

    def example_observation(self, batch_size: int = 1, device: str = "cpu") -> dict[str, Any]:
        cfg = self.config
        return {
            "mode": self.mode,
            "cond_pixels": torch.randn(
                batch_size, cfg.image_channels, cfg.image_size, cfg.image_size, device=device
            ),
            "lang_tokens": torch.randint(
                0, cfg.vocab_size, (batch_size, cfg.max_lang_len), device=device
            ),
            "neg_lang_tokens": torch.zeros(
                batch_size, cfg.max_lang_len, dtype=torch.long, device=device
            ),
            "has_condition": False,
        }

    # --- compile-path helpers (reference backend ONNX capture) ----------------
    _STAGE_IO_NAMES = {
        "vae_encode": (["cond_pixels"], ["cond_latent"]),
        "text_embed": (["lang_tokens"], ["text_mem"]),
        "dit": (["text_mem", "cond_latent", "x_t", "time_emb"], ["v_t"]),
        "vae_decode": (["latent"], ["video"]),
    }

    def stage_io_names(self, stage: str) -> tuple[list[str] | None, list[str] | None]:
        names = self._STAGE_IO_NAMES.get(stage)
        return names if names is not None else (None, None)

    def stage_example_inputs(self, stage: str, obs: dict[str, Any]):
        cfg = self.config
        cond_pixels = obs["cond_pixels"]
        lang_tokens = obs["lang_tokens"]
        b = cond_pixels.shape[0]
        device = cond_pixels.device
        if stage == "vae_encode":
            return (cond_pixels,)
        if stage == "text_embed":
            return (lang_tokens,)
        if stage == "dit":
            cond_latent = self.stage_module("vae_encode")(cond_pixels)
            text_mem = self.stage_module("text_embed")(lang_tokens)
            n = cfg.action_horizon if self.mode == "action" else cfg.num_video_tokens
            x_t = torch.randn(b, n, cfg.token_dim, device=device)
            time_emb = torch.randn(b, self.time_embed_dim, device=device)
            return (text_mem, cond_latent, x_t, time_emb)
        if stage == "vae_decode":
            latent = torch.randn(b, cfg.num_video_tokens, cfg.token_dim, device=device)
            return (latent,)
        raise KeyError(f"No example inputs defined for cosmos3 stage {stage!r}.")


register_model("cosmos3", Cosmos3Adapter, override=True)
register_model("cosmos3_nano", Cosmos3Adapter, override=True)
register_model("cosmos3_super", Cosmos3Adapter, override=True)
