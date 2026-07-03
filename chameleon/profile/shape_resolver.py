"""合并 infer 参数与 build_cfg opt_shapes，生成 stage 输入形状。"""

from __future__ import annotations

from typing import Any

from chameleon.config.schema import TaskConfig
from chameleon.deploy.build_cfg import load_build_cfg
from chameleon.deploy.cosmos3.shapes import (
    COSMOS3_ACTION_HORIZON,
    COSMOS3_TEXT_PREFIX_LEN,
    COSMOS3_VIDEO_TOKENS,
)
from chameleon.deploy.paths import resolve_build_cfg_path, resolve_deploy_paths
from chameleon.deploy.pi05.shapes import PI05_LIBERO_PREFIX_LEN
from chameleon.profile.execution_plan import ExecutionPlan, PlanMode


def precision_to_dtype_bytes(precision: str) -> int:
    key = precision.lower().replace("_", "")
    if key in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 2
    if key in {"fp8", "e4m3", "e5m2"}:
        return 1
    return 4


def resolve_precision(task: TaskConfig) -> str:
    override = task.model_overrides.get("precision")
    if override:
        return str(override)
    if task.compile:
        for step in task.compile:
            build_cfg = step.options.get("build_cfg")
            if build_cfg:
                try:
                    cfg = load_build_cfg(build_cfg)
                    if cfg.get("precision"):
                        return str(cfg["precision"])
                except (FileNotFoundError, ValueError, ImportError):
                    pass
    return "bfloat16"


def _reference_shapes(task: TaskConfig, stage: str, plan: ExecutionPlan) -> dict[str, tuple[int, ...]]:
    cfg = task.model_overrides
    batch = plan.batch_size
    action_dim = int(cfg.get("action_dim", 32))
    action_horizon = int(cfg.get("action_horizon", 50))
    num_image_tokens = 64
    max_lang_len = 48
    image_size = 224

    if stage == "vit":
        return {"images": (batch, 3, image_size, image_size)}
    if stage == "llm_prefix":
        return {
            "img_tokens": (batch, num_image_tokens, 256),
            "lang_tokens": (batch, max_lang_len),
        }
    if stage == "action_expert":
        return {
            "state": (batch, action_dim),
            "prefix_memory": (batch, num_image_tokens + max_lang_len, 256),
            "x_t": (batch, action_horizon, action_dim),
            "time_emb": (batch, 256),
        }
    raise KeyError(f"Unknown reference stage {stage!r}.")


def _cosmos3_cfg_values(task: TaskConfig) -> dict[str, int | str]:
    cfg = task.model_overrides
    mode = str(cfg.get("mode") or getattr(task.generate, "mode", None) or "video")
    return {
        "mode": mode,
        "batch": task.infer.batch_size,
        "image_channels": int(cfg.get("image_channels", 3)),
        "image_size": int(cfg.get("image_size", 64)),
        "max_lang_len": int(cfg.get("max_lang_len", COSMOS3_TEXT_PREFIX_LEN)),
        "vocab_size": int(cfg.get("vocab_size", 1024)),
        "num_video_tokens": int(cfg.get("num_video_tokens", COSMOS3_VIDEO_TOKENS)),
        "action_horizon": int(cfg.get("action_horizon", COSMOS3_ACTION_HORIZON)),
        "action_dim": int(cfg.get("action_dim", 32)),
        "hidden_size": int(cfg.get("hidden_size", 128)),
        "token_dim": int(cfg.get("action_dim", 32)),
    }


def _cosmos3_reference_shapes(task: TaskConfig, stage: str, plan: ExecutionPlan) -> dict[str, tuple[int, ...]]:
    c = _cosmos3_cfg_values(task)
    batch = plan.batch_size
    if stage == "vae_encode":
        return {"cond_pixels": (batch, c["image_channels"], c["image_size"], c["image_size"])}
    if stage == "text_embed":
        return {"lang_tokens": (batch, c["max_lang_len"])}
    if stage == "dit":
        gen_tokens = c["action_horizon"] if c["mode"] == "action" else c["num_video_tokens"]
        return {
            "text_mem": (batch, c["max_lang_len"], c["hidden_size"]),
            "cond_latent": (batch, c["num_video_tokens"], c["token_dim"]),
            "x_t": (batch, gen_tokens, c["token_dim"]),
            "time_emb": (batch, c["hidden_size"]),
        }
    if stage == "vae_decode":
        return {"latent": (batch, c["num_video_tokens"], c["token_dim"])}
    raise KeyError(f"Unknown cosmos3 reference stage {stage!r}.")


def _cosmos3_default_deploy_shapes(task: TaskConfig, stage: str, plan: ExecutionPlan) -> dict[str, tuple[int, ...]]:
    # Deploy build_cfg defaults mirror reference small MoT profile.
    return _cosmos3_reference_shapes(task, stage, plan)


def _default_deploy_shapes(task: TaskConfig, stage: str, plan: ExecutionPlan) -> dict[str, tuple[int, ...]]:
    batch = plan.batch_size
    action_dim = int(task.model_overrides.get("action_dim", 32))
    action_horizon = int(task.model_overrides.get("action_horizon", 10))
    prefix_len = PI05_LIBERO_PREFIX_LEN
    num_layers = 18
    head_dim = 256

    if stage == "vit":
        return {"pixel_values": (batch, 3, 224, 224)}
    if stage == "llm":
        return {
            "inputs_embeds": (batch, prefix_len, 2048),
            "attention_mask": (batch, 1, prefix_len, prefix_len),
            "position_ids": (batch, prefix_len),
        }
    if stage == "expert":
        seq_len = action_horizon
        return {
            "attention_mask": (batch, 1, seq_len, prefix_len + seq_len),
            "position_ids": (batch, seq_len),
            "inputs_embeds": (batch, seq_len, 1024),
            "adarms_cond": (batch, 1024),
            "past_keys": (num_layers, batch, prefix_len, head_dim),
            "past_values": (num_layers, batch, prefix_len, head_dim),
        }
    if stage == "denoise":
        return {
            "prefix_pad_masks": (batch, prefix_len),
            "past_keys": (num_layers, batch, prefix_len, head_dim),
            "past_values": (num_layers, batch, prefix_len, head_dim),
            "x_t": (batch, action_horizon, action_dim),
            "timestep": (batch,),
        }
    raise KeyError(f"Unknown deploy stage {stage!r}.")


def resolve_stage_shapes(task: TaskConfig, stage: str, plan: ExecutionPlan) -> dict[str, tuple[int, ...]]:
    if task.architecture == "cosmos3":
        if plan.mode == PlanMode.REFERENCE:
            return _cosmos3_reference_shapes(task, stage, plan)
        try:
            from chameleon.deploy.cosmos3.paths import resolve_build_cfg_path as resolve_cosmos3_build_cfg
            from chameleon.deploy.cosmos3.paths import resolve_cosmos3_paths

            paths = resolve_cosmos3_paths(task)
            cfg_path = resolve_cosmos3_build_cfg(task, stage, paths)
            cfg = load_build_cfg(cfg_path)
            opt = cfg.get("opt_shapes")
            if isinstance(opt, dict) and opt:
                return {k: tuple(v) for k, v in opt.items()}
        except (FileNotFoundError, KeyError, ValueError, ImportError):
            pass
        return _cosmos3_default_deploy_shapes(task, stage, plan)

    if plan.mode == PlanMode.REFERENCE:
        return _reference_shapes(task, stage, plan)

    try:
        paths = resolve_deploy_paths(task)
        cfg_path = resolve_build_cfg_path(task, stage, paths)
        cfg = load_build_cfg(cfg_path)
        opt = cfg.get("opt_shapes")
        if isinstance(opt, dict) and opt:
            return {k: tuple(v) for k, v in opt.items()}
    except (FileNotFoundError, KeyError, ValueError):
        pass

    return _default_deploy_shapes(task, stage, plan)


def shapes_summary(shapes: dict[str, tuple[int, ...]]) -> dict[str, list[int]]:
    return {k: list(v) for k, v in shapes.items()}
