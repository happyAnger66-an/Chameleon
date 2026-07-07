"""pi05 四段式 TRT 推理管线 — vit → llm → denoise 环（非框架 Orchestrator）。"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from chameleon.deploy.pi05.shapes import PI05_LIBERO_PREFIX_LEN
from chameleon.models.pi05.attention import make_att_2d_masks
from chameleon.runtime.base import Engine

logger = logging.getLogger(__name__)

# TRT softmax 对极小 mask 值（openpi 用的 finfo.min）易产生 NaN；与 model_optimizer 一致裁剪到 -1e4。
_TRT_ATTN_MASK_NEG_CAP = -1e4


def _sanitize_attention_mask_for_trt(attention_mask: torch.Tensor) -> torch.Tensor:
    """将 additive attention mask 裁剪到 TRT 安全范围。"""
    return attention_mask.clamp(min=_TRT_ATTN_MASK_NEG_CAP)


def _restore_vit_scale_for_openpi(img_emb: torch.Tensor, model: Any) -> torch.Tensor:
    """vit.onnx 导出含 ``/sqrt(hidden)``，运行时需乘回以匹配 openpi ``embed_image``。"""
    try:
        hidden = int(model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
    except Exception:  # noqa: BLE001
        hidden = 2048
    scale = math.sqrt(float(hidden))
    return (img_emb.to(torch.float32) * scale).to(img_emb.dtype)


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    n_bad = int(tensor.numel() - torch.isfinite(tensor).sum().item())
    logger.error(
        "Pi05TrtPipeline: non-finite values in %s (%d/%d); TRT stage may need rebuild "
        "or attention_mask sanitization.",
        name,
        n_bad,
        tensor.numel(),
    )


def _pad_prefix_to_static_len(
    prefix_embs: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 prefix 右侧 pad 到 TRT 静态 engine 的 seq_len（无效位 pad_mask=False）。"""
    seq_len = int(prefix_embs.shape[1])
    if seq_len == target_len:
        return prefix_embs, prefix_pad_masks, prefix_att_masks
    if seq_len > target_len:
        raise ValueError(
            f"prefix length {seq_len} exceeds TRT static target {target_len}; "
            "rebuild llm/denoise engines with a larger seq_len or shorten prompt/images."
        )
    pad_len = target_len - seq_len
    batch = prefix_embs.shape[0]
    hidden = prefix_embs.shape[2]
    device = prefix_embs.device
    logger.debug(
        "Padding prefix sequence %d -> %d for static TRT llm/denoise engines.",
        seq_len,
        target_len,
    )
    prefix_embs = torch.cat(
        [
            prefix_embs,
            torch.zeros(batch, pad_len, hidden, device=device, dtype=prefix_embs.dtype),
        ],
        dim=1,
    )
    pad_false = torch.zeros(batch, pad_len, device=device, dtype=torch.bool)
    prefix_pad_masks = torch.cat([prefix_pad_masks, pad_false], dim=1)
    prefix_att_masks = torch.cat([prefix_att_masks, pad_false], dim=1)
    return prefix_embs, prefix_pad_masks, prefix_att_masks


def _embed_prefix_trt(
    model: Any,
    vit_engine: Engine,
    images: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SigLIP 走 TRT vit.engine；语言 embedding 仍用 PyTorch embed_tokens。"""
    embs: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []
    att_masks: list[int] = []

    for img, img_mask in zip(images, img_masks, strict=True):
        img_emb = vit_engine.run({"pixel_values": img})["output"]
        img_emb = _restore_vit_scale_for_openpi(img_emb, model)
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        att_masks.extend([0] * num_img_embs)

    pwe = model.paligemma_with_expert
    lang_emb = pwe.embed_language_tokens(lang_tokens)
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb = lang_emb * math.sqrt(lang_emb_dim)

    embs.append(lang_emb)
    pad_masks.append(lang_masks)
    att_masks.extend([0] * lang_emb.shape[1])

    embs_cat = torch.cat(embs, dim=1)
    pad_cat = torch.cat(pad_masks, dim=1)
    bsize = pad_cat.shape[0]
    att_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_cat.device)
    att_t = att_t[None, :].expand(bsize, att_t.shape[0])
    return embs_cat, pad_cat, att_t


class Pi05TrtPipeline:
    """对齐 openpi ``PI0Pytorch.sample_actions`` 的 TRT 推理内核。"""

    def __init__(
        self,
        engines: dict[str, Engine],
        *,
        num_steps: int = 10,
        static_prefix_len: int = PI05_LIBERO_PREFIX_LEN,
    ) -> None:
        self._engines = engines
        self._num_steps = num_steps
        self._static_prefix_len = int(static_prefix_len)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def infer(
        self,
        model: Any,
        device: str | torch.device,
        observation: Any,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Run TRT vit → llm → denoise loop and return action chunk ``[B, H, D]``."""
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        steps = int(num_steps if num_steps is not None else self._num_steps)
        bsize = observation.state.shape[0]
        if noise is None:
            noise = model.sample_noise(
                (bsize, model.config.action_horizon, model.config.action_dim),
                dev,
            )

        images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(  # noqa: SLF001
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = _embed_prefix_trt(
            model,
            self._engines["vit"],
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = _pad_prefix_to_static_len(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            self._static_prefix_len,
        )

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = model._prepare_attention_masks_4d(prefix_att_2d)  # noqa: SLF001
        prefix_att_4d = _sanitize_attention_mask_for_trt(prefix_att_4d)

        llm_out = self._engines["llm"].run(
            {
                "inputs_embeds": prefix_embs,
                "attention_mask": prefix_att_4d,
                "position_ids": prefix_position_ids,
            }
        )
        past_keys = llm_out.get("past_keys", llm_out["output"])
        past_values = llm_out["past_values"]
        _check_finite("llm.past_keys", past_keys)
        _check_finite("llm.past_values", past_values)

        dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=dev)
        x_t = noise.to(dtype=torch.float32, device=dev) if isinstance(noise, torch.Tensor) else noise
        time = torch.tensor(1.0, dtype=torch.float32, device=dev)
        denoise = self._engines["denoise"]

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            step_out = denoise.run(
                {
                    "prefix_pad_masks": prefix_pad_masks,
                    "past_keys": past_keys,
                    "past_values": past_values,
                    "x_t": x_t,
                    "timestep": expanded_time,
                }
            )
            v_t = step_out.get("v_t", step_out["output"])
            _check_finite("denoise.v_t", v_t)
            x_t = x_t + dt * v_t
            time = time + dt

        _check_finite("actions", x_t)
        return x_t
