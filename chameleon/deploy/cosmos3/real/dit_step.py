"""MoT 单步去噪 Export wrapper — 对标 pi05 ``Pi05DenoiseExport``。

``Cosmos3OmniTransformer.forward`` 有 ~28 个参数（含 list / python int），无法直接进
去噪环做 ONNX 导出。本 wrapper 把 **静态** 联合序列字段（``input_ids`` / ``position_ids`` /
``*_indexes`` / ``token_shapes`` / ``noisy_frame_indexes`` / ``action_domain_ids`` …）在构造时
固化（张量 register_buffer，python 标量/嵌套 list 存属性），forward 仅接收逐 step 变化的
4 个动态张量，输出 ``(v_vision, v_action)``（原始 velocity，掩码/CFG 放 host）。

Policy-DROID v1 简化：``guidance_scale=1`` → 单路 dit，无 sound。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Cosmos3DitStepExport(nn.Module):
    """Single MoT denoise step: dynamic (vision/action latents + timesteps) → velocity."""

    def __init__(self, transformer: nn.Module, static: dict[str, Any]) -> None:
        super().__init__()
        self.transformer = transformer

        # Tensor static fields → buffers (baked as ONNX constants; not saved to state_dict).
        self.register_buffer("input_ids", static["input_ids"], persistent=False)
        self.register_buffer("text_indexes", static["text_indexes"], persistent=False)
        self.register_buffer("position_ids", static["position_ids"], persistent=False)
        self.register_buffer("vision_sequence_indexes", static["vision_sequence_indexes"], persistent=False)
        self.register_buffer("vision_mse_loss_indexes", static["vision_mse_loss_indexes"], persistent=False)
        self.register_buffer("action_sequence_indexes", static["action_sequence_indexes"], persistent=False)
        self.register_buffer("action_mse_loss_indexes", static["action_mse_loss_indexes"], persistent=False)
        self.register_buffer("action_domain_ids", static["action_domain_ids"], persistent=False)
        # list[Tensor] fields: store the single inner tensor, re-wrap in a list at call time.
        self.register_buffer("_vision_noisy_frame_indexes", static["vision_noisy_frame_indexes"][0], persistent=False)
        self.register_buffer("_action_noisy_frame_indexes", static["action_noisy_frame_indexes"][0], persistent=False)

        # Python-native static fields (kept as attributes; not traced as tensors).
        self.und_len = int(static["und_len"])
        self.sequence_length = int(static["sequence_length"])
        self.vision_token_shapes = static["vision_token_shapes"]
        self.action_token_shapes = static["action_token_shapes"]

    def forward(
        self,
        vision_tokens: torch.Tensor,
        vision_timesteps: torch.Tensor,
        action_tokens: torch.Tensor,
        action_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds_vision, _preds_sound, preds_action = self.transformer(
            input_ids=self.input_ids,
            text_indexes=self.text_indexes,
            position_ids=self.position_ids,
            und_len=self.und_len,
            sequence_length=self.sequence_length,
            vision_tokens=[vision_tokens],
            vision_token_shapes=self.vision_token_shapes,
            vision_sequence_indexes=self.vision_sequence_indexes,
            vision_mse_loss_indexes=self.vision_mse_loss_indexes,
            vision_timesteps=vision_timesteps,
            vision_noisy_frame_indexes=[self._vision_noisy_frame_indexes],
            action_tokens=[action_tokens],
            action_token_shapes=self.action_token_shapes,
            action_sequence_indexes=self.action_sequence_indexes,
            action_mse_loss_indexes=self.action_mse_loss_indexes,
            action_timesteps=action_timesteps,
            action_noisy_frame_indexes=[self._action_noisy_frame_indexes],
            action_domain_ids=[self.action_domain_ids],
        )
        return preds_vision[0], preds_action[0]
