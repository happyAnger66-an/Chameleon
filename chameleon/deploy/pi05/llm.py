"""Pi05 LLM prefix ONNX 导出。"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)


class Pi05LlmExport(nn.Module):
    def __init__(self, llm_decoder):
        super().__init__()
        self.model = llm_decoder
        self.model.config._attn_implementation = "eager"

    def forward(self, inputs_embeds, attention_mask, position_ids):
        mod_dtype = getattr(self.model, "dtype", None)
        if mod_dtype is None:
            try:
                mod_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                mod_dtype = None
        if mod_dtype is not None and inputs_embeds.dtype != mod_dtype:
            inputs_embeds = inputs_embeds.to(mod_dtype)

        prefix_output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_caches = prefix_output.past_key_values
        if isinstance(past_key_caches, DynamicCache):
            past_keys_tensor = torch.cat(list(past_key_caches.key_cache), dim=0)
            past_values_tensor = torch.cat(list(past_key_caches.value_cache), dim=0)
        else:
            past_keys_tensor = torch.cat([kv[0] for kv in past_key_caches], dim=0)
            past_values_tensor = torch.cat([kv[1] for kv in past_key_caches], dim=0)
        return past_keys_tensor, past_values_tensor, prefix_output.last_hidden_state

    @classmethod
    def from_pi05_model(cls, pi05_model) -> "Pi05LlmExport":
        paligemma = pi05_model.paligemma_with_expert.paligemma
        return cls(paligemma.get_decoder())


def export_llm(
    pi05_model,
    export_dir: str | Path,
    *,
    dynamo: bool = False,
    seq_len: int = 968,
) -> Path:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / "llm.onnx"

    model = Pi05LlmExport.from_pi05_model(pi05_model).eval().cuda()
    inputs_embeds = torch.randn((1, seq_len, 2048), dtype=torch.bfloat16, device="cuda")
    attention_mask = torch.randn((1, 1, seq_len, seq_len), dtype=torch.float32, device="cuda")
    position_ids = torch.randint(1, 1000, (1, seq_len), dtype=torch.int64, device="cuda")

    start = time.time()
    logger.info("Exporting llm.onnx (seq_len=%d) -> %s", seq_len, out_path)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (inputs_embeds, attention_mask, position_ids),
            str(out_path),
            export_params=True,
            input_names=["inputs_embeds", "attention_mask", "position_ids"],
            output_names=["past_keys", "past_values", "last_hidden_state"],
            opset_version=19,
            dynamo=dynamo,
            do_constant_folding=True,
            dynamic_axes={
                "inputs_embeds": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 2: "seq_len", 3: "seq_len"},
                "position_ids": {0: "batch_size", 1: "seq_len"},
                "past_keys": {0: "past_keys_dim0", 2: "seq_len"},
                "past_values": {0: "past_values_dim0", 2: "seq_len"},
                "last_hidden_state": {0: "batch_size", 1: "seq_len"},
            },
        )
    logger.info("llm.onnx export done in %.1fs", time.time() - start)
    return out_path
