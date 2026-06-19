"""Pi05 denoise step ONNX 导出（单次 flow 步）。"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)

_ATTN_MASK_FILL_VALUE = -2.3819763e38


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must be shape (batch_size,)")

    dev = device if device is not None else time.device
    dtype = torch.float64 if dev.type == "cpu" else torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=dev)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class Pi05DenoiseExport(nn.Module):
    """单次 denoise 步：对齐 openpi ``denoise_step`` + ``embed_suffix``（pi05）。"""

    def __init__(
        self,
        gemma_expert: nn.Module,
        expert_config,
        action_in_proj: nn.Linear,
        time_mlp_in: nn.Linear,
        time_mlp_out: nn.Linear,
        action_out_proj: nn.Linear,
        *,
        action_horizon: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.gemma_expert = gemma_expert
        self.expert_config = expert_config
        self.action_in_proj = action_in_proj
        self.time_mlp_in = time_mlp_in
        self.time_mlp_out = time_mlp_out
        self.action_out_proj = action_out_proj
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.gemma_expert.config._attn_implementation = "eager"

        suffix_ar = [1] + [0] * (action_horizon - 1)
        self.register_buffer(
            "_suffix_ar_mask",
            torch.tensor(suffix_ar, dtype=torch.int32),
            persistent=False,
        )

    @classmethod
    def from_pi05_model(cls, pi05_model) -> "Pi05DenoiseExport":
        if not getattr(pi05_model.config, "pi05", False):
            raise ValueError("Pi05DenoiseExport requires config.pi05=True")
        pwe = pi05_model.paligemma_with_expert
        return cls(
            gemma_expert=pwe.gemma_expert.model,
            expert_config=pwe.gemma_expert.config,
            action_in_proj=pi05_model.action_in_proj,
            time_mlp_in=pi05_model.time_mlp_in,
            time_mlp_out=pi05_model.time_mlp_out,
            action_out_proj=pi05_model.action_out_proj,
            action_horizon=pi05_model.config.action_horizon,
            action_dim=pi05_model.config.action_dim,
        )

    def _wrap_past_key_values(self, past_keys: torch.Tensor, past_values: torch.Tensor) -> DynamicCache:
        cache = DynamicCache()
        num_layers = past_keys.shape[0]
        for i in range(num_layers):
            cache.update(past_keys[i : i + 1], past_values[i : i + 1], i)
        return cache

    @staticmethod
    def _prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, _ATTN_MASK_FILL_VALUE)

    def _compute_adarms_cond(self, timestep: torch.Tensor) -> torch.Tensor:
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.to(dtype=self.time_mlp_in.weight.dtype)
        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        return F.silu(x)

    def _embed_suffix_pi05(self, noisy_actions, timestep):
        action_emb = self.action_in_proj(noisy_actions.to(dtype=self.action_in_proj.weight.dtype))
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device
        pad_masks = torch.ones(bsize, action_emb.shape[1], dtype=torch.bool, device=device)
        att_base = self._suffix_ar_mask.to(device=device).expand(bsize, -1)
        adarms_cond = self._compute_adarms_cond(timestep)
        return action_emb, pad_masks, att_base, adarms_cond

    def forward(
        self,
        prefix_pad_masks: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix_pi05(
            x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1, dtype=torch.int64)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1, dtype=torch.int64) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        expert_dtype = self.gemma_expert.layers[0].self_attn.q_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=expert_dtype)

        past = self._wrap_past_key_values(past_keys, past_values)
        outputs = self.gemma_expert(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past,
            inputs_embeds=suffix_embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        suffix_out = outputs.last_hidden_state[:, -self.action_horizon :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


def export_denoise(
    pi05_model,
    export_dir: str | Path,
    *,
    dynamo: bool = False,
    export_dtype: torch.dtype = torch.bfloat16,
    prefix_len: int = 968,
) -> Path:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / "denoise.onnx"

    model = Pi05DenoiseExport.from_pi05_model(pi05_model).eval().cuda()
    num_layers = int(model.expert_config.num_hidden_layers)

    prefix_pad_masks = torch.ones((1, prefix_len), dtype=torch.bool, device="cuda")
    past_keys = torch.cat(
        [
            torch.randn((1, 1, prefix_len, 256), dtype=export_dtype, device="cuda")
            for _ in range(num_layers)
        ],
        dim=0,
    )
    past_values = torch.cat(
        [
            torch.randn((1, 1, prefix_len, 256), dtype=export_dtype, device="cuda")
            for _ in range(num_layers)
        ],
        dim=0,
    )
    x_t = torch.randn(
        (1, model.action_horizon, model.action_dim), dtype=torch.float32, device="cuda"
    )
    timestep = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    start = time.time()
    logger.info("Exporting denoise.onnx -> %s", out_path)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (prefix_pad_masks, past_keys, past_values, x_t, timestep),
            str(out_path),
            export_params=True,
            input_names=["prefix_pad_masks", "past_keys", "past_values", "x_t", "timestep"],
            output_names=["v_t"],
            opset_version=19,
            dynamo=dynamo,
            do_constant_folding=True,
            dynamic_axes={
                "prefix_pad_masks": {0: "batch_size", 1: "prefix_seq_len"},
                "past_keys": {1: "batch_size", 2: "prefix_seq_len"},
                "past_values": {1: "batch_size", 2: "prefix_seq_len"},
                "x_t": {0: "batch_size"},
                "timestep": {0: "batch_size"},
                "v_t": {0: "batch_size"},
            },
        )
    logger.info("denoise.onnx export done in %.1fs", time.time() - start)
    return out_path
