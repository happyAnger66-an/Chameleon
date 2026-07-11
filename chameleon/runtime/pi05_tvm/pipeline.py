"""pi05 TVM 去噪管线 — vit(TRT) 前缀嵌入 → mlc-vla TVM prefill + 图内/宿主 Euler denoise。

与 ``Pi05TrtPipeline`` 相比：
- 前缀嵌入（SigLIP vit + 语言 embedding + pad/att mask）完全复用 TRT 路径的 ``_embed_prefix_trt``，
  保证与 openpi 一致；
- LLM prefill（TRT ``llm.engine``）与 TRT ``denoise.engine`` 换成 mlc-vla M1：
  expert-0 ``prefill(prefix_embs, prefix_mask)`` 固化逐层 prefix K/V；
  默认 ``use_loop=True`` 走图内 ``denoise_loop_kv``（整段 Euler，可 CUDA Graph）；
  否则宿主逐步调 ``denoise_step_kv``（suffix RoPE offset = 有效 prefix 长度）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from chameleon.deploy.pi05.shapes import PI05_LIBERO_PREFIX_LEN
from chameleon.runtime.base import Engine
from chameleon.runtime.pi05_trt.pipeline import _embed_prefix_trt, _pad_prefix_to_static_len

logger = logging.getLogger(__name__)


class Pi05TvmPipeline:
    """openpi Policy 的 ``_sample_actions`` 后端：TVM M1 去噪。"""

    def __init__(
        self,
        tvm_client: Any,           # TvmWorkerClient（3.12 子进程）
        vit_engine: Engine,
        *,
        num_steps: int = 10,
        static_prefix_len: int = PI05_LIBERO_PREFIX_LEN,
        use_loop: bool = True,
    ) -> None:
        self._tvm = tvm_client
        self._vit = vit_engine
        self._num_steps = num_steps
        self._use_loop = bool(use_loop)
        self._static_prefix_len = int(static_prefix_len)
        cfg_pl = int(tvm_client.info["prefix_len"])
        if cfg_pl != self._static_prefix_len:
            logger.warning(
                "TVM prefix_len(%d) != static_prefix_len(%d)；以 TVM 配置为准。",
                cfg_pl, self._static_prefix_len,
            )
            self._static_prefix_len = cfg_pl

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
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        steps = int(num_steps if num_steps is not None else self._num_steps)
        bsize = observation.state.shape[0]
        if bsize != 1:
            raise NotImplementedError(f"Pi05TvmPipeline 目前仅支持 batch=1（收到 {bsize}）")
        if noise is None:
            noise = model.sample_noise(
                (bsize, model.config.action_horizon, model.config.action_dim), dev
            )

        images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(  # noqa: SLF001
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = _embed_prefix_trt(
            model, self._vit, images, img_masks, lang_tokens, lang_masks,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = _pad_prefix_to_static_len(
            prefix_embs, prefix_pad_masks, prefix_att_masks, self._static_prefix_len,
        )

        prefix_np = prefix_embs.detach().to(torch.float32).cpu().numpy()  # [1,P,W]
        pad_np = prefix_pad_masks.detach().cpu().numpy().reshape(-1).astype("float32")  # [P]
        noise_np = noise.detach().to(torch.float32).cpu().numpy()  # [1,H,D]

        actions = self._tvm.sample(prefix_np, pad_np, noise_np, steps, loop=self._use_loop)  # [1,H,D] fp32
        return torch.from_numpy(np.asarray(actions)).to(dev)


def attach_tvm_to_policy(policy: Any, pipeline: Pi05TvmPipeline) -> None:
    """将 openpi Policy 的 ``_sample_actions`` 路由到 TVM 管线。"""
    model = policy._model
    num_steps = pipeline.num_steps

    def _sample_actions_tvm(device, observation, noise=None, **kwargs):
        ns = int(kwargs.get("num_steps", num_steps))
        return pipeline.infer(model, device, observation, noise=noise, num_steps=ns)

    policy._sample_actions = _sample_actions_tvm
