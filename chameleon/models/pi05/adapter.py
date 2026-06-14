"""pi05 model adapter.

Exposes the pi05 stages through :class:`~chameleon.models.base.ModelAdapter`.
Two backing implementations share the same stage interface:

* ``use_reference=True`` (default): the lightweight :mod:`reference` model, so the
  pipeline runs without external weights.
* ``use_reference=False`` with a ``checkpoint``: best-effort wrapping of openpi's
  ``PI0Pytorch`` (requires the openpi runtime + transformers_replace install).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from chameleon.architectures.pi05 import ARCHITECTURE_NAME
from chameleon.models.base import ModelAdapter, register_model
from chameleon.models.pi05.reference import Pi05Config, Pi05ReferenceModel

logger = logging.getLogger(__name__)

_STAGE_ATTR = {
    "vit": "vit",
    "llm_prefix": "llm_prefix",
    "action_expert": "action_expert",
}


class Pi05Adapter(ModelAdapter):
    architecture = ARCHITECTURE_NAME

    def __init__(self, config: Pi05Config | None = None) -> None:
        super().__init__(config or Pi05Config())
        self.model: Pi05ReferenceModel | Any | None = None
        self._device = "cpu"

    @classmethod
    def make_config(cls, overrides: dict[str, Any] | None = None) -> Pi05Config:
        valid = {f for f in Pi05Config.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in (overrides or {}).items() if k in valid}
        return Pi05Config(**filtered)

    def build(self, device: str = "cpu") -> "Pi05Adapter":
        self._device = device
        if self.config.use_reference:
            self.model = Pi05ReferenceModel(self.config).to(device).eval()
        else:
            self.model = self._load_openpi(device)
        return self

    def _load_openpi(self, device: str):
        """Best-effort load of the real openpi PI0Pytorch model.

        Kept intentionally thin: real checkpoint loading + transformers_replace
        wiring is part of the NVIDIA bring-up phase. Falls back to the reference
        model with a warning so the pipeline never hard-fails for the MVP.
        """
        try:
            from openpi.models.pi0_config import Pi0Config
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

            pi0_cfg = Pi0Config(
                action_dim=self.config.action_dim,
                action_horizon=self.config.action_horizon,
                pi05=True,
                paligemma_variant=self.config.paligemma_variant,
                action_expert_variant=self.config.action_expert_variant,
                pytorch_compile_mode=None,
            )
            model = PI0Pytorch(pi0_cfg)
            if self.config.checkpoint:
                state = torch.load(self.config.checkpoint, map_location=device)
                model.load_state_dict(state, strict=False)
            return model.to(device).eval()
        except Exception as exc:  # noqa: BLE001 - graceful MVP fallback
            logger.warning(
                "Falling back to pi05 reference model (could not load openpi: %s)", exc
            )
            self.config.use_reference = True
            return Pi05ReferenceModel(self.config).to(device).eval()

    def stage_module(self, stage: str):
        if self.model is None:
            raise RuntimeError("Call build() before stage_module().")
        attr = _STAGE_ATTR.get(stage)
        if attr is None:
            raise KeyError(f"Unknown pi05 stage {stage!r}.")
        return getattr(self.model, attr)

    @property
    def time_embed_dim(self) -> int:
        """Dimension of the sinusoidal time embedding fed to the action expert."""
        return int(self.config.expert_width)

    def example_observation(self, batch_size: int = 1, device: str = "cpu") -> dict[str, Any]:
        cfg = self.config
        return {
            "images": torch.randn(
                batch_size, cfg.image_channels, cfg.image_size, cfg.image_size, device=device
            ),
            "lang_tokens": torch.randint(
                0, cfg.vocab_size, (batch_size, cfg.max_lang_len), device=device
            ),
            "state": torch.randn(batch_size, cfg.action_dim, device=device),
        }


register_model("pi05", Pi05Adapter, override=True)
register_model("pi05_libero", Pi05Adapter, override=True)
