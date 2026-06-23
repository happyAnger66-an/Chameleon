"""openpi 评测 transform 辅助 — 在不加载完整 Policy 的情况下构建 I/O 变换链。

作用：
    build_openpi_eval_transforms() 复刻 create_trained_policy 中的 input/output
    transform 组合（跳过 repack，因 dataloader 已完成 repack），供
    ChameleonOrchestratorRunner 与 OpenPiPolicyRunner 共用，保证两条 evaluate
    路径的归一化 / tokenize 一致。

架构位置：
    工具层（evaluate）— 被 OpenPiPolicyRunner / ChameleonOrchestratorRunner 调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from chameleon.evaluate.norm_stats import load_norm_stats_for_eval, resolve_norm_stats_assets_dir

logger = logging.getLogger(__name__)


@dataclass
class OpenPiEvalTransforms:
    """openpi 评测用 input/output transform + 模型维度元信息。"""

    input_transform: Callable[[dict], dict]
    output_transform: Callable[[dict], dict]
    action_horizon: int
    action_dim: int
    openpi_config: str
    asset_id: str | None = None


def build_openpi_eval_transforms(
    *,
    openpi_config: str,
    checkpoint_dir: str | Path,
    norm_stats_dir: str | Path | None = None,
    asset_id: str | None = None,
    default_prompt: str | None = None,
) -> OpenPiEvalTransforms:
    """构建与 create_trained_policy 对齐的 I/O transform（不含 repack）。"""
    try:
        import openpi.transforms as transforms
        from openpi.training import checkpoints as _checkpoints
        from openpi.training import config as _config
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "build_openpi_eval_transforms 需要 openpi（openpi.transforms / openpi.training.*）。"
        ) from exc

    train_cfg = _config.get_config(openpi_config)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    resolved_asset_id = asset_id or getattr(data_config, "asset_id", None)

    norm_stats = load_norm_stats_for_eval(
        checkpoint_dir=Path(checkpoint_dir).expanduser(),
        norm_stats_dir=norm_stats_dir,
        asset_id=resolved_asset_id,
        data_config=data_config,
        checkpoints_mod=_checkpoints,
    )
    if norm_stats is None:
        if resolved_asset_id is None:
            raise ValueError(
                f"openpi_config={openpi_config!r} 未解析 asset_id，且无法从 checkpoint 加载 norm_stats。"
            )
        assets_dir = resolve_norm_stats_assets_dir(
            checkpoint_dir=checkpoint_dir,
            norm_stats_dir=norm_stats_dir,
        )
        if assets_dir is None:
            raise FileNotFoundError(
                f"无法找到 norm_stats（checkpoint_dir={checkpoint_dir!r} norm_stats_dir={norm_stats_dir!r}）。"
            )
        norm_stats = _checkpoints.load_norm_stats(str(assets_dir), resolved_asset_id)

    input_transform = transforms.compose(
        [
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )

    return OpenPiEvalTransforms(
        input_transform=input_transform,
        output_transform=output_transform,
        action_horizon=int(train_cfg.model.action_horizon),
        action_dim=int(getattr(train_cfg.model, "action_dim", 0) or 0),
        openpi_config=openpi_config,
        asset_id=resolved_asset_id,
    )


def apply_output_transform(output_transform: Callable[[dict], dict], actions_t: Any, inputs: dict) -> np.ndarray:
    """对模型输出 tensor 应用 output_transform，返回 numpy 物理动作。"""
    import numpy as np

    if hasattr(actions_t, "detach"):
        actions_np = actions_t.detach().cpu().numpy()
    else:
        actions_np = np.asarray(actions_t)
    if actions_np.ndim == 3:
        actions_np = actions_np[0]

    state = inputs.get("state")
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    outputs = output_transform({"actions": actions_np, "state": state})
    return np.asarray(outputs["actions"])
