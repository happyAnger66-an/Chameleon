"""Pi05 action expert ONNX 导出。"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)


class Pi05ExpertExport(nn.Module):
    def __init__(self, config, gemma_expert):
        super().__init__()
        self.config = config
        self.gemma_expert = gemma_expert
        self.gemma_expert.config._attn_implementation = "eager"

    def _wrap_past_key_values(self, input_keys, input_values):
        cache = DynamicCache()
        num_layers = input_keys.shape[0]
        for i in range(num_layers):
            cache.update(input_keys[i : i + 1], input_values[i : i + 1], i)
        return cache

    def forward(
        self,
        attention_mask,
        position_ids,
        inputs_embeds,
        adarms_cond,
        past_keys,
        past_values,
    ):
        k_v_cache = self._wrap_past_key_values(past_keys, past_values)
        output = self.gemma_expert(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            adarms_cond=adarms_cond,
            past_key_values=k_v_cache,
        )
        return output.last_hidden_state

    @classmethod
    def from_pi05_model(cls, pi05_model) -> "Pi05ExpertExport":
        expert = pi05_model.paligemma_with_expert.gemma_expert
        return cls(expert.config, expert.model)


def export_expert(
    pi05_model,
    export_dir: str | Path,
    *,
    dynamo: bool = True,
    export_dtype: torch.dtype = torch.bfloat16,
    prefix_len: int = 968,
    action_seq_len: int = 10,
) -> Path:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / "expert.onnx"

    model = Pi05ExpertExport.from_pi05_model(pi05_model).eval().cuda()
    num_layers = int(model.config.num_hidden_layers)

    adarms_cond = torch.zeros(1, 1024, dtype=torch.float32, device="cuda")
    attention_mask = torch.randn(
        (1, 1, action_seq_len, prefix_len + action_seq_len),
        dtype=torch.float32,
        device="cuda",
    )
    position_ids = torch.randint(
        1, model.config.vocab_size, (1, action_seq_len), dtype=torch.int64, device="cuda"
    )
    inputs_embeds = torch.randn((1, action_seq_len, 1024), dtype=torch.float32, device="cuda")

    past_keys = [
        torch.randn((1, 1, prefix_len, 256), dtype=export_dtype, device="cuda")
        for _ in range(num_layers)
    ]
    past_values = [
        torch.randn((1, 1, prefix_len, 256), dtype=export_dtype, device="cuda")
        for _ in range(num_layers)
    ]
    past_keys_tensor = torch.cat(past_keys, dim=0)
    past_values_tensor = torch.cat(past_values, dim=0)

    start = time.time()
    logger.info("Exporting expert.onnx -> %s", out_path)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (
                attention_mask,
                position_ids,
                inputs_embeds,
                adarms_cond,
                past_keys_tensor,
                past_values_tensor,
            ),
            str(out_path),
            input_names=[
                "attention_mask",
                "position_ids",
                "inputs_embeds",
                "adarms_cond",
                "past_keys",
                "past_values",
            ],
            output_names=["last_hidden_state"],
            opset_version=19,
            dynamo=dynamo,
            do_constant_folding=True,
            dynamic_axes={
                "attention_mask": {0: "batch_size"},
                "position_ids": {0: "batch_size"},
                "inputs_embeds": {0: "batch_size"},
                "adarms_cond": {0: "batch_size"},
                "past_keys": {2: "llm_seq_len"},
                "past_values": {2: "llm_seq_len"},
            },
        )
    logger.info("expert.onnx export done in %.1fs", time.time() - start)
    return out_path
