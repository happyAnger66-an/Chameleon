"""pi05 模型适配器 — 将 openpi / 参考模型映射为 Chameleon stage 接口。

作用：
    实现 Pi05Adapter：支持 reference 路径（无需权重）和真实 openpi 路径
    （checkpoint 加载 .pt/.pth/.safetensors）。通过 _STAGE_ATTR /
    _OPENPI_STAGE_ATTR 将 vit / llm_prefix / action_expert 映射到具体
    nn.Module 子模块。

架构位置：
    模型/架构层 — 被 api.build_adapter 实例化。下游：frontend 按 stage
    导出 ONNX，quantization 按 stage 量化，orchestrator 按 stage 推理。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from chameleon.architectures.pi05 import ARCHITECTURE_NAME
from chameleon.models.base import ModelAdapter, register_model
from chameleon.models.pi05.reference import Pi05Config, Pi05ReferenceModel

logger = logging.getLogger(__name__)

# Reference model: attributes match the stage names directly.
_STAGE_ATTR = {
    "vit": "vit",
    "llm_prefix": "llm_prefix",
    "action_expert": "action_expert",
}

# Real openpi PI0Pytorch: map each Chameleon stage to a real submodule. These
# submodules are independently quantizable / exportable. NOTE: driving the full
# real model through the simplified Pi05Orchestrator additionally requires
# aligning the KV-cache plumbing with openpi's sample_actions (tracked as a
# bring-up follow-up); per-submodule quantize/compile works today.
_OPENPI_STAGE_ATTR = {
    "vit": ("paligemma_with_expert", "paligemma", "vision_tower"),
    "llm_prefix": ("paligemma_with_expert", "paligemma", "language_model"),
    "action_expert": ("paligemma_with_expert", "gemma_expert"),
}


def _resolve_attr_path(root, path: tuple[str, ...]):
    obj = root
    for attr in path:
        obj = getattr(obj, attr)
    return obj


class Pi05Adapter(ModelAdapter):
    architecture = ARCHITECTURE_NAME

    def __init__(self, config: Pi05Config | None = None) -> None:
        super().__init__(config or Pi05Config())
        self.model: Pi05ReferenceModel | Any | None = None
        self._device = "cpu"
        self._is_real_openpi = False

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
        """Load the real openpi PI0Pytorch model and (optionally) a checkpoint.

        镜像 openpi 自身的 PyTorch 加载路径（见 ``model.py:load_pytorch`` +
        ``policy_config``）：构建 ``PI0Pytorch`` → ``safetensors.torch.load_model``
        加载（正确处理 tied weights）→ ``to_bfloat16_for_selected_params`` 做
        选择性 bf16（vision patch embedding / layernorm / final norm 保持 fp32，
        其余 bf16），这样 ~3.6B 参数的模型才能放进 12GB 显存。

        支持 ``.safetensors``（save_model 导出）与 ``.pt``/``.pth``。加载失败时
        回退到参考模型并告警，保证流水线不硬失败。
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
            # 先在 CPU 上构建并加载，做完精度转换后再搬到目标设备，避免 fp32 整模型占满显存。
            model = PI0Pytorch(pi0_cfg)
            if self.config.checkpoint:
                self._load_checkpoint(model, self.config.checkpoint)
            self._apply_precision(model)
            self._is_real_openpi = True
            return model.to(device).eval()
        except Exception as exc:  # noqa: BLE001 - graceful MVP fallback
            logger.warning(
                "Falling back to pi05 reference model (could not load openpi: %s)", exc
            )
            self.config.use_reference = True
            self._is_real_openpi = False
            return Pi05ReferenceModel(self.config).to(device).eval()

    @staticmethod
    def _load_checkpoint(model, checkpoint: str) -> None:
        """在 CPU 上把权重加载进 model（就地）。"""
        if checkpoint.endswith(".safetensors"):
            # 优先用 safetensors.torch.load_model（与 openpi save_model 配对，处理 tied weights）。
            try:
                from safetensors.torch import load_model

                missing, unexpected = load_model(model, checkpoint, strict=False)
            except Exception:  # noqa: BLE001 - 退回到裸 state_dict 加载
                from safetensors.torch import load_file

                state = load_file(checkpoint)
                missing, unexpected = model.load_state_dict(state, strict=False)
        else:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            "Loaded checkpoint %s (missing=%d, unexpected=%d keys)",
            checkpoint,
            len(missing),
            len(unexpected),
        )

    def _apply_precision(self, model) -> None:
        """对齐 openpi：对 paligemma_with_expert 做选择性 bf16 / fp32。"""
        target = getattr(model, "paligemma_with_expert", None)
        cast = getattr(target, "to_bfloat16_for_selected_params", None)
        if cast is not None:
            cast(self.config.precision)
        elif self.config.precision == "bfloat16":
            model.to(dtype=torch.bfloat16)

    def stage_module(self, stage: str):
        if self.model is None:
            raise RuntimeError("Call build() before stage_module().")
        if self._is_real_openpi:
            path = _OPENPI_STAGE_ATTR.get(stage)
            if path is None:
                raise KeyError(f"Unknown pi05 stage {stage!r}.")
            return _resolve_attr_path(self.model, path)
        attr = _STAGE_ATTR.get(stage)
        if attr is None:
            raise KeyError(f"Unknown pi05 stage {stage!r}.")
        return getattr(self.model, attr)

    @property
    def time_embed_dim(self) -> int:
        """Dimension of the sinusoidal time embedding fed to the action expert."""
        return int(self.config.expert_width)

    @property
    def orchestrator_key(self) -> str | None:
        # 真实 openpi 模型走 sample_actions 端到端路径；参考模型用三段式默认编排器。
        return "pi05_real" if self._is_real_openpi else None

    # pi05 真实模型预处理需要的相机视图键（见 openpi preprocessing_pytorch.IMAGE_KEYS）。
    _OPENPI_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    def to_openpi_observation(self, observation: dict[str, Any], device: str):
        """把简单 observation dict 转为真实 openpi ``Observation``（冒烟测试用）。

        简单 obs 仅含单张 image + lang_tokens + state。这里把该图复制到 pi05
        预处理要求的三个相机视图，并合成全有效掩码与 prompt mask。
        （pi05 的 action 后缀不使用 state，仅其 batch 维参与计算。）
        """
        from openpi.models.model import Observation  # 延迟导入，避免无 openpi 环境报错

        images = observation["images"].to(device)
        lang_tokens = observation["lang_tokens"].to(device)
        state = observation["state"].to(device)
        bsize = state.shape[0]

        image_dict = {k: images for k in self._OPENPI_IMAGE_KEYS}
        mask = torch.ones(bsize, dtype=torch.bool, device=device)
        mask_dict = {k: mask for k in self._OPENPI_IMAGE_KEYS}

        data = {
            "image": image_dict,
            "image_mask": mask_dict,
            "state": state,
            "tokenized_prompt": lang_tokens,
            "tokenized_prompt_mask": torch.ones_like(lang_tokens, dtype=torch.bool),
        }
        return Observation.from_dict(data)

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
