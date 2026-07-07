"""LeRobot 离线评测 — 逐帧对比 pi05 预测动作与 ground-truth。

作用：
    evaluate_lerobot() 把 dataloader（LeRobotDataSource）产出的样本逐帧喂给
    OpenPiPolicyRunner 推理，用 evaluate.compare_actions 计算每帧动作误差
    （max_abs / mean_abs / cosine），并汇总成 EvalSummary。这是 Chameleon 在
    真实数据上验证推理路径正确性的入口（区别于 synthetic-smoke 的 infer）。

架构位置：
    工具层（evaluate）— 上游：api.run_eval / CLI ``eval``；下游：dataloader 与
    OpenPiPolicyRunner。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from chameleon.dataloader.lerobot import LeRobotDataSource
from chameleon.evaluate.compare import ActionDiff, compare_actions
from chameleon.evaluate.metrics import attach_pt_trt_pair_metrics, metrics_pt_vs_gt, step_metrics
from chameleon.evaluate.runner_base import PolicyRunner, SupportsDualInfer, SupportsFixedNoise
from chameleon.evaluate.running_stats import RunningDimDiffStats
from chameleon.evaluate.viewers.base import (
    EvalEventSink,
    EvalStepEvent,
    NullEventSink,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalSampleResult:
    index: int
    episode_id: int | None
    diff: ActionDiff
    prompt: str | None = None


@dataclass
class EvalSummary:
    num_samples: int
    mean_max_abs: float
    mean_mean_abs: float
    mean_cosine: float
    worst_max_abs: float
    worst_index: int
    samples: list[EvalSampleResult] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"samples={self.num_samples} "
            f"mean_max_abs={self.mean_max_abs:.6f} "
            f"mean_mean_abs={self.mean_mean_abs:.6f} "
            f"mean_cosine={self.mean_cosine:.6f} "
            f"worst_max_abs={self.worst_max_abs:.6f}@idx{self.worst_index}"
        )


def _align_horizon(
    pred: np.ndarray,
    gt: np.ndarray,
    compare_horizon: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """裁剪到共同的 [horizon, action_dim]，使 pred 与 gt 形状一致。"""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if gt.ndim == 3 and gt.shape[0] == 1:
        gt = gt[0]
    if pred.ndim == 1:
        pred = pred[None, :]
    if gt.ndim == 1:
        gt = gt[None, :]

    horizon = min(pred.shape[0], gt.shape[0])
    if compare_horizon is not None:
        horizon = min(horizon, int(compare_horizon))
    dim = min(pred.shape[-1], gt.shape[-1])
    return pred[:horizon, :dim], gt[:horizon, :dim]


def chunk_eligible(
    data_source: LeRobotDataSource,
    global_index: int,
) -> int | None:
    """与 model_optimizer ``chunk_eligibility`` 一致：非对齐 chunk 不推 WebUI step。

    仅在 ``(idx - start) % action_horizon == 0`` 且整段落在同一 episode 内时
    返回 episode_id，否则 ``None``。避免相邻帧 chunk 的 ``global_index`` 重叠导致
    前端 Plotly 折线折返。
    """
    ah = data_source.action_horizon
    start = data_source.start_index
    if (global_index - start) % ah != 0:
        return None
    end = data_source.eval_end_exclusive
    n_total = data_source.frame_count
    if global_index + ah > n_total or global_index + ah > end:
        return None
    ep_ids = data_source.episode_ids_per_frame
    ep0 = int(ep_ids[global_index])
    ep_last = int(ep_ids[global_index + ah - 1])
    if ep0 != ep_last:
        return None
    return ep0


def _emit_step_events(
    sink: EvalEventSink,
    *,
    run_id: str,
    sample_index: int,
    episode_id: int | None,
    prompt: str | None,
    pred_h: np.ndarray,
    gt_h: np.ndarray,
    observation: dict[str, Any],
    infer_ms: float,
    running: RunningDimDiffStats | None = None,
    pred_trt_h: np.ndarray | None = None,
) -> None:
    ah = int(pred_h.shape[0])
    ep = int(episode_id) if episode_id is not None else 0
    for k in range(ah):
        gt_row = gt_h[k]
        pred_row = pred_h[k]
        pred_trt_row = pred_trt_h[k] if pred_trt_h is not None and k < pred_trt_h.shape[0] else None
        if pred_trt_row is not None:
            metrics = metrics_pt_vs_gt(pred_row - gt_row)
            metrics = attach_pt_trt_pair_metrics(
                metrics,
                pred_pt_row=pred_row,
                pred_trt_row=pred_trt_row,
                gt_row=gt_row,
            )
        else:
            metrics = step_metrics(gt_row, pred_row)
        if running is not None:
            running.update(gt_row, pred_row)
            metrics = {**metrics, **running.metrics_payload()}
        pred_trt_list = None
        if pred_trt_row is not None:
            row = pred_trt_row.astype(np.float64)
            pred_trt_list = [float(x) for x in row.tolist()]
            if k == 0 and sample_index == 0 and not np.isfinite(row).all():
                logger.warning(
                    "[eval] pred_action_trt has non-finite values at sample_index=0 "
                    "(WebUI 会将 NaN/Inf 序列化为 null，曲线可能不可见)"
                )
        sink.on_step(
            EvalStepEvent(
                run_id=run_id,
                episode_id=ep,
                global_index=int(sample_index + k),
                k_in_chunk=int(k),
                is_chunk_start=bool(k == 0),
                action_horizon=ah,
                prompt=prompt if k == 0 else None,
                gt_action=[float(x) for x in gt_row.astype(np.float64).tolist()],
                pred_action=[float(x) for x in pred_row.astype(np.float64).tolist()],
                pred_action_trt=pred_trt_list,
                metrics=metrics,
                observation=observation if k == 0 else None,
                infer_ms=float(infer_ms) if k == 0 else None,
            )
        )


def evaluate_lerobot(
    data_source: LeRobotDataSource,
    policy_runner: PolicyRunner,
    *,
    num_samples: int = 50,
    stride: int = 1,
    compare_horizon: int | None = None,
    keep_samples: bool = True,
    log_every: int = 10,
    event_sink: EvalEventSink | None = None,
    run_meta: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> EvalSummary:
    """在 LeRobot 数据集上评测策略，返回汇总误差。

    Args:
        data_source: 已构建（或可懒构建）的 LeRobot 数据源。
        policy_runner: 真实 pi05 策略运行器。
        num_samples: 评测帧数上限。
        stride: 帧采样步长（>1 跳帧）。
        compare_horizon: 仅对比前 N 个动作步；None=全 horizon。
        keep_samples: 是否在 summary 中保留逐帧结果。
        log_every: 每多少帧打一条进度日志（0=关闭）。
        event_sink: 评测事件接收端；``on_step`` 必须非阻塞。
        run_meta: 传给 ``on_run_start`` 的 meta 字典。
        run_id: WebUI run_id；缺省时自动生成。
    """
    sink = event_sink or NullEventSink()
    rid = run_id or uuid.uuid4().hex[:12]

    data_source.build()
    policy_runner.build()

    if run_meta is not None:
        sink.on_run_start(run_meta)
    else:
        sink.on_run_start({"type": "meta", "run_id": rid})

    total = len(data_source)
    indices = list(range(0, total, max(1, int(stride))))[:num_samples]
    if not indices:
        raise ValueError("数据集为空或 num_samples/stride 配置导致无可评测帧。")

    sum_max_abs = 0.0
    sum_mean_abs = 0.0
    sum_cosine = 0.0
    worst_max_abs = -1.0
    worst_index = -1
    results: list[EvalSampleResult] = []
    n_done = 0
    ui_running: RunningDimDiffStats | None = None
    if not isinstance(sink, NullEventSink):
        ui_running = RunningDimDiffStats(action_dim=max(1, int(policy_runner.action_dim)))

    for i in indices:
        sample = data_source[i]
        t0 = time.perf_counter()
        infer_dual = policy_runner.infer_dual if isinstance(policy_runner, SupportsDualInfer) else None
        pred_trt: np.ndarray | None = None
        noise_fn = policy_runner.noise_for_sample if isinstance(policy_runner, SupportsFixedNoise) else None
        flow_noise = noise_fn(sample.index) if noise_fn is not None else None
        if infer_dual is not None:
            pred, pred_trt = infer_dual(
                sample.observation,
                sample_index=sample.index,
                noise=flow_noise,
            )
        else:
            pred = policy_runner.infer(sample.observation, noise=flow_noise)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        pred_a, gt_a = _align_horizon(pred, sample.actions_gt, compare_horizon)
        diff = compare_actions(torch.from_numpy(gt_a), torch.from_numpy(pred_a))
        pred_trt_a = None
        if pred_trt is not None:
            pred_trt_a, _ = _align_horizon(pred_trt, sample.actions_gt, compare_horizon)

        ep_for_ui = chunk_eligible(data_source, sample.index)
        if ep_for_ui is not None:
            _emit_step_events(
                sink,
                run_id=rid,
                sample_index=sample.index,
                episode_id=ep_for_ui,
                prompt=sample.prompt,
                pred_h=pred_a,
                gt_h=gt_a,
                observation=sample.observation,
                infer_ms=infer_ms,
                running=ui_running,
                pred_trt_h=pred_trt_a,
            )

        sum_max_abs += diff.max_abs
        sum_mean_abs += diff.mean_abs
        sum_cosine += diff.cosine
        if diff.max_abs > worst_max_abs:
            worst_max_abs = diff.max_abs
            worst_index = sample.index
        if keep_samples:
            results.append(
                EvalSampleResult(
                    index=sample.index,
                    episode_id=sample.episode_id,
                    diff=diff,
                    prompt=sample.prompt,
                )
            )

        n_done += 1
        if log_every and n_done % log_every == 0:
            logger.info(
                "[eval] %d/%d idx=%d max_abs=%.6f mean_abs=%.6f cosine=%.6f infer_ms=%.1f",
                n_done,
                len(indices),
                sample.index,
                diff.max_abs,
                diff.mean_abs,
                diff.cosine,
                infer_ms,
            )

    summary = EvalSummary(
        num_samples=n_done,
        mean_max_abs=sum_max_abs / n_done,
        mean_mean_abs=sum_mean_abs / n_done,
        mean_cosine=sum_cosine / n_done,
        worst_max_abs=worst_max_abs,
        worst_index=worst_index,
        samples=results,
    )
    sink.on_run_done(summary)
    return summary
