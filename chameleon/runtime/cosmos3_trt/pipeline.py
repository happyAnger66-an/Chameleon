"""cosmos3 TRT 推理管线 — vae_encode → text_embed → dit 去噪环 → vae_decode。

对齐 Cosmos3OmniPipeline 的去噪控制流，但每个 stage 由 TRT engine 执行：条件 VAE
编码与文本 embedding 各算一次，dit 去噪步在 flow-matching 环内重复（每步 CFG
cond/uncond 两遍合成 velocity）。与参考编排器共享相同 engine-dict 接口（``run`` 返回
dict），因此可用 PyTorch / TensorRT 任一后端的 Engine 驱动。
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from chameleon.models.cosmos3.reference import create_sinusoidal_pos_embedding
from chameleon.runtime.base import Engine

logger = logging.getLogger(__name__)


def _take(out: dict[str, Any], *names: str):
    """Pull a tensor from an engine output dict by known name, else first value."""
    for name in names:
        if name in out:
            return out[name]
    if "output" in out:
        return out["output"]
    return next(iter(out.values()))


class Cosmos3TrtPipeline:
    """Cosmos3 generator denoise kernel over a stage engine dict (TRT or PyTorch)."""

    def __init__(
        self,
        engines: dict[str, Engine],
        *,
        num_steps: int = 35,
        guidance_scale: float = 6.0,
    ) -> None:
        self._engines = engines
        self._num_steps = num_steps
        self._guidance = guidance_scale

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def infer(
        self,
        observation: dict[str, Any],
        device: str | torch.device,
        *,
        mode: str = "video",
        token_dim: int,
        num_video_tokens: int,
        action_horizon: int,
        time_dim: int,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        steps = int(num_steps if num_steps is not None else self._num_steps)
        guidance = float(guidance_scale if guidance_scale is not None else self._guidance)

        cond_pixels = observation["cond_pixels"].to(dev)
        lang_tokens = observation["lang_tokens"].to(dev)
        neg_lang_tokens = observation.get("neg_lang_tokens")
        if neg_lang_tokens is None:
            neg_lang_tokens = torch.zeros_like(lang_tokens)
        neg_lang_tokens = neg_lang_tokens.to(dev)
        bsize = cond_pixels.shape[0]
        num_tokens = action_horizon if mode == "action" else num_video_tokens

        cond_latent = _take(
            self._engines["vae_encode"].run({"cond_pixels": cond_pixels}), "cond_latent"
        )
        text_mem = _take(
            self._engines["text_embed"].run({"lang_tokens": lang_tokens}), "text_mem"
        )
        neg_text_mem = _take(
            self._engines["text_embed"].run({"lang_tokens": neg_lang_tokens}), "text_mem"
        )

        dit = self._engines["dit"]
        do_cfg = guidance != 1.0
        x_t = torch.randn(bsize, num_tokens, token_dim, device=dev)
        dt = -1.0 / steps
        time = 1.0
        while time >= -dt / 2:
            t = torch.full((bsize,), time, dtype=torch.float32, device=dev)
            time_emb = create_sinusoidal_pos_embedding(t, time_dim)
            v_cond = _take(
                dit.run(
                    {
                        "text_mem": text_mem,
                        "cond_latent": cond_latent,
                        "x_t": x_t,
                        "time_emb": time_emb,
                    }
                ),
                "v_t",
            )
            if do_cfg:
                v_uncond = _take(
                    dit.run(
                        {
                            "text_mem": neg_text_mem,
                            "cond_latent": cond_latent,
                            "x_t": x_t,
                            "time_emb": time_emb,
                        }
                    ),
                    "v_t",
                )
                v_t = v_uncond + guidance * (v_cond - v_uncond)
            else:
                v_t = v_cond
            x_t = x_t + dt * v_t
            time += dt

        if mode == "action":
            return x_t
        return _take(self._engines["vae_decode"].run({"latent": x_t}), "video")


class Cosmos3PolicyTrtPipeline:
    """Cosmos3 Policy (action) TRT 去噪内核 — 真实权重、固定 profile、guidance=1。

    与 :class:`Cosmos3TrtPipeline`（reference surrogate）不同，本类对齐真实
    ``Cosmos3OmniPipeline.__call__`` 的 policy 通路：

    - **host（diffusers pipe）**：tokenize / mRoPE 打包（``build_policy_pack``）、
      velocity 掩码（``pipe._mask_velocity_predictions``）、UniPC ``scheduler.step``；
    - **TRT engine**：``vae_encode``（观测→z0）、``dit``（每步 velocity）、
      可选 ``vae_decode``（rollout 可视化）。

    v1 简化：``guidance_scale=1`` 单路 dit、``enable_sound=False``、固定 profile 全画布
    latent（不做 padding 移除，保证 engine shape 恒定）。
    """

    def __init__(
        self,
        engines: dict[str, Engine],
        pipe: Any,
        profile: Any,
        *,
        num_steps: int | None = None,
    ) -> None:
        self._engines = engines
        self._pipe = pipe
        self._profile = profile
        self._num_steps = int(num_steps or profile.num_inference_steps)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def _init_latents(self, pack, video: torch.Tensor, dev: torch.device, dtype: torch.dtype):
        """Host prepare_latents 等价逻辑（固定 profile；vision frame 0 锁 clean，action 全噪声）。"""
        z0 = _take(self._engines["vae_encode"].run({"video": video}), "vision_latent")
        z0 = z0.to(device=dev, dtype=dtype)
        vcm = pack.vision_condition_mask.to(device=dev, dtype=dtype)
        latents = vcm * z0 + (1.0 - vcm) * torch.randn_like(z0)

        action_dim = int(pack.meta["action_shape"][1])
        chunk = int(self._profile.chunk_size)
        action_latents = torch.randn(chunk, action_dim, device=dev, dtype=dtype)
        action_latents[:, pack.raw_action_dim :] = 0
        return latents, action_latents

    def infer_policy(
        self,
        observation: dict[str, Any],
        device: str | torch.device,
        *,
        num_steps: int | None = None,
        return_video: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the policy denoise loop; return action chunk ``[chunk, raw_action_dim]``.

        ``observation`` must carry ``video`` = conditioning canvas tensor
        ``[1, 3, num_frames, canvas_h, canvas_w]`` (already preprocessed to the tier canvas).
        """
        import copy

        from chameleon.deploy.cosmos3.real.pack import build_policy_pack

        dev = torch.device(device) if not isinstance(device, torch.device) else device
        steps = int(num_steps if num_steps is not None else self._num_steps)
        pipe = self._pipe
        dtype = pipe.transformer.dtype

        pack = build_policy_pack(pipe, self._profile, device=dev, dtype=dtype)
        meta = pack.meta

        video = observation["video"].to(device=dev, dtype=dtype)
        latents, action_latents = self._init_latents(pack, video, dev, dtype)

        vcm = pack.vision_condition_mask.to(device=dev, dtype=dtype)
        acm = pack.action_condition_mask.to(device=dev, dtype=dtype)
        nvt = int(meta["num_noisy_vision_tokens"])
        nat = int(meta["num_noisy_action_tokens"])
        raw_dim = int(pack.raw_action_dim)

        pipe.scheduler.set_timesteps(steps, device=dev)
        timesteps = pipe.scheduler.timesteps
        action_scheduler = copy.deepcopy(pipe.scheduler)

        dit = self._engines["dit"]
        for t in timesteps:
            ts = float(t.item())
            out = dit.run(
                {
                    "vision_tokens": latents.to(dtype),
                    "vision_timesteps": torch.full((nvt,), ts, device=dev),
                    "action_tokens": action_latents.to(dtype),
                    "action_timesteps": torch.full((nat,), ts, device=dev),
                }
            )
            v_vision = _take(out, "v_vision")
            v_action = _take(out, "v_action")
            cond_v_vision, _cv_sound, cond_v_action = pipe._mask_velocity_predictions(
                [v_vision],
                None,
                vision_condition_mask=[vcm],
                preds_action=[v_action],
                action_condition_mask=[acm],
                raw_action_dim=raw_dim,
            )
            latents = pipe.scheduler.step(
                cond_v_vision.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
            if cond_v_action is not None:
                action_latents = action_scheduler.step(
                    cond_v_action.unsqueeze(0), t, action_latents.unsqueeze(0), return_dict=False
                )[0].squeeze(0)
                action_latents[:, raw_dim:] = 0

        action_out = action_latents[:, :raw_dim]
        if not return_video:
            return action_out
        video_out = _take(self._engines["vae_decode"].run({"latent": latents.to(dtype)}), "video")
        return {"action": action_out, "video": video_out}
