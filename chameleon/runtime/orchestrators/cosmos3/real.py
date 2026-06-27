"""cosmos3 真实 diffusers 编排 — Cosmos3OmniPipeline 端到端生成路径。

按 ``task.generate.mode`` 组织 ``Cosmos3OmniPipeline.__call__``：video 模式走
text/image/video-to-video 扩散，action 模式走 CosmosActionCondition 策略生成。
不拆 stage / 不加载 per-stage engine（requires_stage_engines=False）。返回主模态
张量（action chunk 或视频帧 pt 张量），完整输出（video/sound/action）挂在
``self.last_output`` 上供上层取用。
"""

from __future__ import annotations

from typing import Any

import torch

from chameleon.core.context import RunContext
from chameleon.models.base import ModelAdapter
from chameleon.runtime.base import Engine
from chameleon.runtime.orchestrator.base import Orchestrator, register_orchestrator


def _load_media(path: str | None, *, is_video: bool):
    if not path:
        return None
    if is_video:
        from diffusers.utils import load_video

        return load_video(path)
    from diffusers.utils import load_image

    return load_image(path)


class Cosmos3RealOrchestrator(Orchestrator):
    """真实 diffusers Cosmos3OmniPipeline 端到端编排（不拆 stage）。"""

    architecture = "cosmos3"
    requires_stage_engines = False

    def __init__(self, adapter: ModelAdapter, engines: dict[str, Engine], ctx: RunContext) -> None:
        super().__init__(adapter, engines, ctx)
        self.last_output: Any = None

    def _generate_cfg(self):
        gen = self.ctx.options.get("generate")
        if gen is not None:
            return gen
        # Fallback: synthesize a minimal config from the architecture metadata.
        from chameleon.config.schema import GenerateConfig

        return GenerateConfig(mode=getattr(self.adapter, "mode", "video"))

    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        pipe = getattr(self.adapter, "pipeline", None)
        if pipe is None or not getattr(self.adapter, "_is_real_diffusers", False):
            raise RuntimeError(
                "Cosmos3RealOrchestrator requires a built real diffusers pipeline; "
                "use the cosmos3 reference orchestrator for Cosmos3ReferenceModel."
            )

        gen = self._generate_cfg()
        num_steps = int(self.ctx.options.get("num_steps", gen.num_inference_steps))
        prompt = gen.prompt or observation.get("prompt") or ""

        common = dict(
            prompt=prompt,
            negative_prompt=gen.negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=gen.guidance_scale,
            fps=gen.fps,
            output_type=gen.output_type,
        )

        if gen.mode == "action":
            from diffusers import CosmosActionCondition

            a = gen.action
            condition = CosmosActionCondition(
                mode=a.mode,
                chunk_size=a.chunk_size,
                domain_name=a.domain_name,
                resolution_tier=a.resolution_tier,
                view_point=a.view_point,
                image=_load_media(a.image, is_video=False),
                video=_load_media(a.video, is_video=True),
            )
            result = pipe(action=condition, use_system_prompt=False, **common)
        else:
            image = _load_media(gen.image, is_video=False)
            video = _load_media(gen.video, is_video=True)
            result = pipe(
                image=image,
                video=video,
                num_frames=gen.num_frames,
                height=gen.height,
                width=gen.width,
                enable_sound=gen.enable_sound,
                **common,
            )

        self.last_output = result

        # Return the primary modality tensor so the existing tensor-based infer
        # plumbing (api.run_infer / InferAction) keeps working.
        action = getattr(result, "action", None)
        if gen.mode == "action" and action:
            first = action[0]
            return first if isinstance(first, torch.Tensor) else torch.as_tensor(first)

        video_out = getattr(result, "video", None)
        if isinstance(video_out, torch.Tensor):
            return video_out
        # latent / non-tensor outputs: surface a small tensor proxy for logging.
        return torch.as_tensor(video_out) if video_out is not None else torch.zeros(1)


register_orchestrator("cosmos3_real", Cosmos3RealOrchestrator, override=True)
