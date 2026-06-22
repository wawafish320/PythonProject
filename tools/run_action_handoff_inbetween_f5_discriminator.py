#!/usr/bin/env python3
from __future__ import annotations

"""F5 discriminator clean rerun: masked_cmd vs masked_cmd_smooth vs capacity-matched AR.

Arms:
  A) masked_cmd
  B) masked_cmd_smooth
  C) ar_cmd_capacity_matched
"""

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_action_handoff_inbetween_commanded_yaw_conditioned_probe import (  # noqa: E402
    CommandedYawMaskedMiddlePredictor,
)
from tools.run_action_handoff_inbetween_masked_smoke import (  # noqa: E402
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    _draw_batch,
    _dump_json,
    _dump_md,
    _fmt,
    _pivot_channel_mse_weighted,
    _seam_c1_loss_weighted,
)
from train.action_handoff_inbetween_model import (  # noqa: E402
    GateThresholds,
    MinimalGoalAR,
    ModelConfig,
    StateNormalizer,
    evaluate_rollout_state_space,
)
from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    POSE_SLICE,
    SEAM_LEN_K,
    WALK_F,
    YAW_RATE_SLICE,
    InbetweenSampler,
    SamplerConfig,
    load_clip_states,
)

DEFAULT_OUT_PREFIX = "debug_output/_tmp_action_handoff_f5_discriminator_clean_"
CANONICAL_OUT_DATE = "20260531"
FOCUS_CLIP_L2R = "Walk_L_To_R"
FOCUS_CLIP_R2L = "Walk_R_To_L"
ARM_MASKED_CMD = "masked_cmd"
ARM_MASKED_CMD_SMOOTH = "masked_cmd_smooth"
ARM_AR_CMD = "ar_cmd_capacity_matched"
CANONICAL_ARMS = (ARM_MASKED_CMD, ARM_MASKED_CMD_SMOOTH, ARM_AR_CMD)
CANONICAL_CELLS = ("fullsup", "mirror_r2l")
CANONICAL_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class CellSpec:
    name: str
    holdout_policy: str
    holdout_clip: Optional[str]
    focus_clip: str
    monitor_clips: Tuple[str, ...]


CELL_SPECS: Dict[str, CellSpec] = {
    "fullsup": CellSpec(
        name="fullsup",
        holdout_policy="none",
        holdout_clip=None,
        focus_clip=FOCUS_CLIP_L2R,
        monitor_clips=(),
    ),
    "mirror_r2l": CellSpec(
        name="mirror_r2l",
        holdout_policy="mirror_l_r",
        holdout_clip=FOCUS_CLIP_R2L,
        focus_clip=FOCUS_CLIP_R2L,
        monitor_clips=(FOCUS_CLIP_L2R,),
    ),
}


@dataclass
class PlateauStats:
    loss_first: float
    loss_last: float
    loss_min: float
    loss_max: float
    loss_mean: float
    best_window_mean: float
    tail_window_mean: float
    tail_over_best_ratio: float
    loss_decreased: bool
    plateau_ok: bool
    actual_steps_used: int
    max_steps: int
    window_size: int


def smoothness_delta_mse_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pose_w: float,
    ego_w: float,
    contact_w: float,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Delta-MSE smoothness on learnable channels only (yaw excluded)."""
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(f"pred/target must be [B,H,D], got {tuple(pred.shape)} and {tuple(target.shape)}")
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.shape[1] < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    dp = pred[:, 1:, :] - pred[:, :-1, :]
    dt = target[:, 1:, :] - target[:, :-1, :]
    per = (
        float(pose_w) * torch.mean((dp[..., POSE_SLICE] - dt[..., POSE_SLICE]) ** 2, dim=(1, 2))
        + float(ego_w) * torch.mean((dp[..., EGO_VEL_SLICE] - dt[..., EGO_VEL_SLICE]) ** 2, dim=(1, 2))
        + float(contact_w) * torch.mean((dp[..., CONTACT_SLICE] - dt[..., CONTACT_SLICE]) ** 2, dim=(1, 2))
    ) / max(float(pose_w + ego_w + contact_w), 1e-8)
    if sample_weights is None:
        return torch.mean(per)
    sw = sample_weights.to(device=per.device, dtype=per.dtype)
    sw = sw / torch.clamp(torch.mean(sw), min=1e-8)
    return torch.mean(per * sw)


def evaluate_shared_rollout_state(
    rollout_raw: np.ndarray,
    goal_seam_raw: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
) -> Dict[str, float | bool | int]:
    """Shared F5 state-space evaluator for all arms."""
    roll = np.asarray(rollout_raw, dtype=np.float64)
    goal = np.asarray(goal_seam_raw, dtype=np.float64)
    std64 = np.asarray(std, dtype=np.float64)
    if roll.ndim != 2 or int(roll.shape[1]) != 281:
        raise ValueError(f"rollout_raw must be [H,281], got {tuple(roll.shape)}")
    if goal.ndim != 2 or int(goal.shape[1]) != 281:
        raise ValueError(f"goal_seam_raw must be [K,281], got {tuple(goal.shape)}")
    if std64.ndim != 1 or int(std64.shape[0]) != 281:
        raise ValueError(f"std must be [281], got {tuple(std64.shape)}")

    state = evaluate_rollout_state_space(roll, goal, std64, thr)
    ri = int(state["resume_rollout_frame"])
    tj = int(state["resume_target_frame"])
    ego_diff = (roll[ri, EGO_VEL_SLICE] - goal[tj, EGO_VEL_SLICE]) / std64[EGO_VEL_SLICE]
    con_diff = (roll[ri, CONTACT_SLICE] - goal[tj, CONTACT_SLICE]) / std64[CONTACT_SLICE]
    ego_pop = float(np.mean(np.abs(ego_diff)))
    contact_pop = float(np.mean(np.abs(con_diff)))
    return {
        "pop_safe": bool(state["pop_safe"]),
        "pop": float(state["pop"]),
        "ego_pop": ego_pop,
        "contact_pop": contact_pop,
        "best_pose_d": float(state["best_pose_d"]),
        "resume_rollout_frame": ri,
        "resume_target_frame": tj,
    }


def _param_count(model: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()))


def _parse_int_csv(s: str) -> List[int]:
    vals = [x.strip() for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return [int(v) for v in vals]


def _parse_hidden_candidates(s: str) -> List[int]:
    vals = sorted({int(v.strip()) for v in str(s).split(",") if v.strip()})
    if not vals:
        raise ValueError("ar-hidden-candidates cannot be empty")
    if any(v <= 0 for v in vals):
        raise ValueError("ar-hidden-candidates must be > 0")
    return vals


def _cmd_tensor_from_middle(middle_raw: torch.Tensor, middle_n: torch.Tensor, *, cmd_scale: str) -> torch.Tensor:
    if cmd_scale == "normalized":
        return middle_n[..., YAW_RATE_SLICE]
    if cmd_scale == "raw":
        return middle_raw[..., YAW_RATE_SLICE]
    raise ValueError(f"unsupported cmd_scale: {cmd_scale}")


def _compute_plateau_stats(
    losses: Sequence[float],
    *,
    max_steps: int,
    window_size: int,
    plateau_tail_over_best_max: float,
) -> PlateauStats:
    arr = np.asarray(list(losses), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("loss history is empty")
    w = max(1, int(min(window_size, arr.size)))
    window_means = np.convolve(arr, np.ones((w,), dtype=np.float64) / float(w), mode="valid")
    best_window = float(np.min(window_means))
    tail_window = float(np.mean(arr[-w:]))
    ratio = float(tail_window / max(best_window, 1e-12))
    loss_first = float(arr[0])
    loss_last = float(arr[-1])
    plateau_ok = bool((loss_last < loss_first) and (ratio <= float(plateau_tail_over_best_max)))
    return PlateauStats(
        loss_first=loss_first,
        loss_last=loss_last,
        loss_min=float(np.min(arr)),
        loss_max=float(np.max(arr)),
        loss_mean=float(np.mean(arr)),
        best_window_mean=best_window,
        tail_window_mean=tail_window,
        tail_over_best_ratio=ratio,
        loss_decreased=bool(loss_last < loss_first),
        plateau_ok=plateau_ok,
        actual_steps_used=int(arr.size),
        max_steps=int(max_steps),
        window_size=w,
    )


def _train_masked_arm(
    *,
    model: CommandedYawMaskedMiddlePredictor,
    sampler: InbetweenSampler,
    normalizer: StateNormalizer,
    max_steps: int,
    batch: int,
    seed: int,
    holdout_policy: str,
    holdout_clip: Optional[str],
    max_sample_retries: int,
    f5_pose_w: float,
    f5_ego_w: float,
    f5_contact_w: float,
    smooth_w: float,
    cmd_scale: str,
    lr: float,
    with_smooth: bool,
    plateau_window: int,
    plateau_tail_over_best_max: float,
    early_stop_on_plateau: bool,
) -> Tuple[PlateauStats, Dict[str, int]]:
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    model.train()
    rng = np.random.default_rng(int(seed))
    losses: List[float] = []
    sample_filter_totals: Dict[str, int] = {
        "attempted": 0,
        "accepted": 0,
        "removed_grounded_cross_manifold": 0,
        "removed_full_holdout": 0,
    }
    for _step in range(int(max_steps)):
        ctx, middle, seam, _metas, batch_audit = _draw_batch(
            sampler,
            int(batch),
            rng,
            holdout_clip=holdout_clip,
            holdout_policy=holdout_policy,
            max_sample_retries=int(max_sample_retries),
        )
        for k, v in batch_audit.items():
            sample_filter_totals[k] = int(sample_filter_totals.get(k, 0) + int(v))
        ctx_n = normalizer.normalize(ctx)
        middle_n = normalizer.normalize(middle)
        seam_n = normalizer.normalize(seam)
        cmd = _cmd_tensor_from_middle(middle, middle_n, cmd_scale=cmd_scale)
        pred_n = model(ctx_n, seam_n, cmd)
        sample_w = torch.ones((int(batch),), dtype=pred_n.dtype, device=pred_n.device)
        loss_mid = _pivot_channel_mse_weighted(
            pred_n,
            middle_n,
            pose_w=float(f5_pose_w),
            ego_w=float(f5_ego_w),
            contact_w=float(f5_contact_w),
            sample_weights=sample_w,
        )
        loss = loss_mid
        if with_smooth:
            loss_smooth = smoothness_delta_mse_weighted(
                pred_n,
                middle_n,
                pose_w=float(f5_pose_w),
                ego_w=float(f5_ego_w),
                contact_w=float(f5_contact_w),
                sample_weights=sample_w,
            )
            loss = loss + float(smooth_w) * loss_smooth
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if int(len(losses)) >= int(plateau_window):
            stats_now = _compute_plateau_stats(
                losses,
                max_steps=max_steps,
                window_size=plateau_window,
                plateau_tail_over_best_max=plateau_tail_over_best_max,
            )
            if bool(early_stop_on_plateau) and bool(stats_now.plateau_ok):
                break

    stats = _compute_plateau_stats(
        losses,
        max_steps=max_steps,
        window_size=plateau_window,
        plateau_tail_over_best_max=plateau_tail_over_best_max,
    )
    return stats, sample_filter_totals


def _train_ar_arm(
    *,
    model: MinimalGoalAR,
    sampler: InbetweenSampler,
    normalizer: StateNormalizer,
    max_steps: int,
    batch: int,
    seed: int,
    holdout_policy: str,
    holdout_clip: Optional[str],
    max_sample_retries: int,
    f5_pose_w: float,
    f5_ego_w: float,
    f5_contact_w: float,
    seam_c1_weight: float,
    seam_c1_ego_weight: float,
    seam_c1_contact_weight: float,
    lr: float,
    plateau_window: int,
    plateau_tail_over_best_max: float,
    early_stop_on_plateau: bool,
) -> Tuple[PlateauStats, Dict[str, int]]:
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    model.train()
    rng = np.random.default_rng(int(seed))
    losses: List[float] = []
    sample_filter_totals: Dict[str, int] = {
        "attempted": 0,
        "accepted": 0,
        "removed_grounded_cross_manifold": 0,
        "removed_full_holdout": 0,
    }
    for _step in range(int(max_steps)):
        ctx, middle, seam, _metas, batch_audit = _draw_batch(
            sampler,
            int(batch),
            rng,
            holdout_clip=holdout_clip,
            holdout_policy=holdout_policy,
            max_sample_retries=int(max_sample_retries),
        )
        for k, v in batch_audit.items():
            sample_filter_totals[k] = int(sample_filter_totals.get(k, 0) + int(v))

        ctx_n = normalizer.normalize(ctx)
        middle_n = normalizer.normalize(middle)
        seam_n = normalizer.normalize(seam)
        mid_pred, _seam_pred = model.rollout_teacher_forced(ctx_n, seam_n, middle_n)
        sample_w = torch.ones((int(batch),), dtype=mid_pred.dtype, device=mid_pred.device)
        loss_mid = _pivot_channel_mse_weighted(
            mid_pred,
            middle_n,
            pose_w=float(f5_pose_w),
            ego_w=float(f5_ego_w),
            contact_w=float(f5_contact_w),
            sample_weights=sample_w,
        )
        loss = loss_mid
        if float(seam_c1_weight) > 0.0:
            loss_c1 = _seam_c1_loss_weighted(
                mid_pred,
                seam_n,
                ego_w=float(seam_c1_ego_weight),
                contact_w=float(seam_c1_contact_weight),
                sample_weights=sample_w,
            )
            loss = loss + float(seam_c1_weight) * loss_c1
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if int(len(losses)) >= int(plateau_window):
            stats_now = _compute_plateau_stats(
                losses,
                max_steps=max_steps,
                window_size=plateau_window,
                plateau_tail_over_best_max=plateau_tail_over_best_max,
            )
            if bool(early_stop_on_plateau) and bool(stats_now.plateau_ok):
                break

    stats = _compute_plateau_stats(
        losses,
        max_steps=max_steps,
        window_size=plateau_window,
        plateau_tail_over_best_max=plateau_tail_over_best_max,
    )
    return stats, sample_filter_totals


def _per_step_error_curve(
    pred_raw: np.ndarray,
    target_middle_raw: np.ndarray,
    std: np.ndarray,
) -> Dict[str, List[float]]:
    """Compute per-step-vs-t errors on yaw-excluded channels.

    Args:
      pred_raw: [H,281] float64 CPU.
      target_middle_raw: [H,281] float64 CPU.
      std: [281] float64 CPU.
    """
    pred = np.asarray(pred_raw, dtype=np.float64)
    target = np.asarray(target_middle_raw, dtype=np.float64)
    std64 = np.asarray(std, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 2 or int(pred.shape[1]) != 281:
        raise ValueError(f"pred_raw must be [H,281], got {tuple(pred.shape)}")
    if std64.ndim != 1 or int(std64.shape[0]) != 281:
        raise ValueError(f"std must be [281], got {tuple(std64.shape)}")

    diff = (pred - target) / np.maximum(std64, 1e-8)
    pose = np.mean(np.abs(diff[:, POSE_SLICE]), axis=1)
    ego = np.mean(np.abs(diff[:, EGO_VEL_SLICE]), axis=1)
    contact = np.mean(np.abs(diff[:, CONTACT_SLICE]), axis=1)
    combined = (pose + ego + contact) / 3.0
    return {
        "pose_error": pose.astype(np.float64).tolist(),
        "ego_vel_error": ego.astype(np.float64).tolist(),
        "contact_error": contact.astype(np.float64).tolist(),
        "combined_f5_error": combined.astype(np.float64).tolist(),
    }


def _mean_curve(curves: Sequence[Mapping[str, Sequence[float]]], key: str) -> List[float]:
    if not curves:
        return []
    arr = np.asarray([np.asarray(c[key], dtype=np.float64) for c in curves], dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"curve list for {key} must be rank-2, got {arr.shape}")
    return np.mean(arr, axis=0).astype(np.float64).tolist()


def classify_ar_drift_from_curve(
    combined_curve: Sequence[float],
    *,
    n_rollouts: int,
    min_rollouts: int,
    ratio_threshold: float,
    slope_threshold: float,
    min_horizon: int,
) -> Dict[str, Any]:
    curve = np.asarray(list(combined_curve), dtype=np.float64)
    if int(n_rollouts) < int(min_rollouts) or int(curve.size) < int(min_horizon):
        return {
            "label": "DRIFT_EVIDENCE_INSUFFICIENT",
            "evidence_sufficient": False,
            "n_rollouts": int(n_rollouts),
            "min_rollouts": int(min_rollouts),
            "horizon": int(curve.size),
            "min_horizon": int(min_horizon),
            "first_third_mean": float("nan"),
            "last_third_mean": float("nan"),
            "last_over_first_ratio": float("nan"),
            "slope_vs_t": float("nan"),
            "ratio_threshold": float(ratio_threshold),
            "slope_threshold": float(slope_threshold),
        }

    third = max(1, int(curve.size // 3))
    first = float(np.mean(curve[:third]))
    last = float(np.mean(curve[-third:]))
    ratio = float(last / max(first, 1e-12))
    x = np.arange(int(curve.size), dtype=np.float64)
    var_x = float(np.var(x))
    slope = 0.0 if var_x <= 0.0 else float(np.cov(x, curve, bias=True)[0, 1] / max(var_x, 1e-12))
    increasing = bool((ratio > float(ratio_threshold)) and (slope > float(slope_threshold)))
    return {
        "label": "AR_DRIFT_PRESENT" if increasing else "AR_NO_DRIFT_EVIDENCE_STRONG",
        "evidence_sufficient": True,
        "n_rollouts": int(n_rollouts),
        "min_rollouts": int(min_rollouts),
        "horizon": int(curve.size),
        "min_horizon": int(min_horizon),
        "first_third_mean": first,
        "last_third_mean": last,
        "last_over_first_ratio": ratio,
        "slope_vs_t": slope,
        "ratio_threshold": float(ratio_threshold),
        "slope_threshold": float(slope_threshold),
    }


def _eval_arm_clip(
    *,
    arm_name: str,
    model: nn.Module,
    clip_name: str,
    states: Mapping[str, np.ndarray],
    sampler: InbetweenSampler,
    normalizer: StateNormalizer,
    starts: Sequence[int],
    goal_horizon: int,
    cmd_scale: str,
    drift_ratio_threshold: float,
    drift_slope_threshold: float,
    drift_min_rollouts: int,
    drift_min_horizon: int,
) -> Dict[str, Any]:
    thr = GateThresholds()
    cfg = sampler.config
    hub = states[WALK_F]
    t_f = int(hub.shape[0])
    horizon = int(cfg.gap_min)
    target = states[clip_name]
    g0 = int(min(goal_horizon, target.shape[0] - SEAM_LEN_K))
    target_middle = target[g0 - horizon : g0]
    goal_seam_raw = target[g0 : g0 + SEAM_LEN_K]
    goal_seam_n = normalizer.normalize(torch.as_tensor(goal_seam_raw, dtype=torch.float32)).unsqueeze(0)
    target_middle_t = torch.as_tensor(target_middle, dtype=torch.float32)
    target_middle_n = normalizer.normalize(target_middle_t)
    cmd_eval_n = target_middle_n.unsqueeze(0)[..., YAW_RATE_SLICE]
    cmd_eval_raw = target_middle_t.unsqueeze(0)[..., YAW_RATE_SLICE]

    rows: List[Dict[str, float | bool | int]] = []
    curves: List[Dict[str, List[float]]] = []
    contract: Optional[Dict[str, Any]] = None
    model.eval()
    with torch.no_grad():
        for phase in starts:
            idx = (np.arange(phase - cfg.context_len, phase) % t_f).astype(np.int64)
            ctx_raw = hub[idx]
            ctx_n = normalizer.normalize(torch.as_tensor(ctx_raw, dtype=torch.float32)).unsqueeze(0)
            if arm_name in (ARM_MASKED_CMD, ARM_MASKED_CMD_SMOOTH):
                cmd_in = cmd_eval_n if cmd_scale == "normalized" else cmd_eval_raw
                pred_n = model(ctx_n, goal_seam_n, cmd_in)[0]
                uses_rollout_free_cmd_yaw = False
            elif arm_name == ARM_AR_CMD:
                pred_n = model.rollout_free_commanded_yaw(ctx_n, goal_seam_n, cmd_eval_n)[0]
                cmd_in = cmd_eval_n
                uses_rollout_free_cmd_yaw = True
            else:
                raise ValueError(f"unsupported arm: {arm_name}")

            pred_raw_np = normalizer.denormalize(pred_n).detach().cpu().numpy().astype(np.float64)
            if contract is None:
                contract = {
                    "ctx": {
                        "shape": list(ctx_n.shape),
                        "dtype": str(ctx_n.dtype),
                        "device": str(ctx_n.device),
                    },
                    "goal_seam": {
                        "shape": list(goal_seam_n.shape),
                        "dtype": str(goal_seam_n.dtype),
                        "device": str(goal_seam_n.device),
                    },
                    "cmd_yaw": {
                        "shape": list(cmd_in.shape),
                        "dtype": str(cmd_in.dtype),
                        "device": str(cmd_in.device),
                    },
                    "rollout_raw": {
                        "shape": list(pred_raw_np.shape),
                        "dtype": str(pred_raw_np.dtype),
                        "device": "cpu",
                    },
                }
            metric = evaluate_shared_rollout_state(
                pred_raw_np,
                np.asarray(goal_seam_raw, dtype=np.float64),
                np.asarray(normalizer.std, dtype=np.float64),
                thr,
            )
            rows.append(metric)
            curves.append(
                _per_step_error_curve(
                    pred_raw_np,
                    np.asarray(target_middle, dtype=np.float64),
                    np.asarray(normalizer.std, dtype=np.float64),
                )
            )

    pop_safe = np.asarray([1.0 if bool(r["pop_safe"]) else 0.0 for r in rows], dtype=np.float64)
    pop = np.asarray([float(r["pop"]) for r in rows], dtype=np.float64)
    ego = np.asarray([float(r["ego_pop"]) for r in rows], dtype=np.float64)
    contact = np.asarray([float(r["contact_pop"]) for r in rows], dtype=np.float64)
    pose = np.asarray([float(r["best_pose_d"]) for r in rows], dtype=np.float64)
    mean_curves = {
        "pose_error": _mean_curve(curves, "pose_error"),
        "ego_vel_error": _mean_curve(curves, "ego_vel_error"),
        "contact_error": _mean_curve(curves, "contact_error"),
        "combined_f5_error": _mean_curve(curves, "combined_f5_error"),
    }

    out: Dict[str, Any] = {
        "n": int(len(rows)),
        "pop_safe_rate": float(np.mean(pop_safe)),
        "pop_mean": float(np.mean(pop)),
        "ego_pop_mean": float(np.mean(ego)),
        "contact_pop_mean": float(np.mean(contact)),
        "best_pose_d_mean": float(np.mean(pose)),
        "rows": rows,
        "mean_per_step_curves": mean_curves,
        "yaw_path": {
            "uses_rollout_free_commanded_yaw": bool(uses_rollout_free_cmd_yaw),
            "posthoc_yaw_replacement": False,
        },
        "tensor_contract": contract,
    }
    if arm_name == ARM_AR_CMD:
        out["drift_fingerprint"] = classify_ar_drift_from_curve(
            mean_curves["combined_f5_error"],
            n_rollouts=int(len(rows)),
            min_rollouts=int(drift_min_rollouts),
            ratio_threshold=float(drift_ratio_threshold),
            slope_threshold=float(drift_slope_threshold),
            min_horizon=int(drift_min_horizon),
        )
    return out


def _yaw_body_sensitivity(
    *,
    arm_name: str,
    model: nn.Module,
    clip_name: str,
    states: Mapping[str, np.ndarray],
    sampler: InbetweenSampler,
    normalizer: StateNormalizer,
    starts: Sequence[int],
    goal_horizon: int,
    cmd_scale: str,
    command_effect_eps: float,
) -> Dict[str, Any]:
    cfg = sampler.config
    hub = states[WALK_F]
    t_f = int(hub.shape[0])
    horizon = int(cfg.gap_min)
    target = states[clip_name]
    g0 = int(min(goal_horizon, target.shape[0] - SEAM_LEN_K))
    target_middle = target[g0 - horizon : g0]
    goal_seam_raw = target[g0 : g0 + SEAM_LEN_K]
    goal_seam_n = normalizer.normalize(torch.as_tensor(goal_seam_raw, dtype=torch.float32)).unsqueeze(0)
    tm_raw = torch.as_tensor(target_middle, dtype=torch.float32).unsqueeze(0)
    tm_n = normalizer.normalize(tm_raw)
    cmd_base_n = tm_n[..., YAW_RATE_SLICE]
    cmd_base_raw = tm_raw[..., YAW_RATE_SLICE]
    cmd_zero_n = torch.zeros_like(cmd_base_n)
    cmd_zero_raw = torch.zeros_like(cmd_base_raw)
    cmd_flip_n = -cmd_base_n
    cmd_flip_raw = -cmd_base_raw

    by_variant = {
        "zero_cmd": {"pose": [], "ego": [], "contact": []},
        "sign_flipped_cmd": {"pose": [], "ego": [], "contact": []},
    }
    model.eval()
    with torch.no_grad():
        for phase in starts:
            idx = (np.arange(phase - cfg.context_len, phase) % t_f).astype(np.int64)
            ctx_n = normalizer.normalize(torch.as_tensor(hub[idx], dtype=torch.float32)).unsqueeze(0)
            if arm_name in (ARM_MASKED_CMD, ARM_MASKED_CMD_SMOOTH):
                if cmd_scale == "normalized":
                    cmd_b, cmd_z, cmd_f = cmd_base_n, cmd_zero_n, cmd_flip_n
                else:
                    cmd_b, cmd_z, cmd_f = cmd_base_raw, cmd_zero_raw, cmd_flip_raw
                pred_b = model(ctx_n, goal_seam_n, cmd_b)[0]
                pred_z = model(ctx_n, goal_seam_n, cmd_z)[0]
                pred_f = model(ctx_n, goal_seam_n, cmd_f)[0]
            elif arm_name == ARM_AR_CMD:
                pred_b = model.rollout_free_commanded_yaw(ctx_n, goal_seam_n, cmd_base_n)[0]
                pred_z = model.rollout_free_commanded_yaw(ctx_n, goal_seam_n, cmd_zero_n)[0]
                pred_f = model.rollout_free_commanded_yaw(ctx_n, goal_seam_n, cmd_flip_n)[0]
            else:
                raise ValueError(f"unsupported arm: {arm_name}")

            raw_b = normalizer.denormalize(pred_b)
            raw_z = normalizer.denormalize(pred_z)
            raw_f = normalizer.denormalize(pred_f)
            dz = torch.abs(raw_z - raw_b)
            df = torch.abs(raw_f - raw_b)
            by_variant["zero_cmd"]["pose"].append(float(torch.mean(dz[..., POSE_SLICE]).item()))
            by_variant["zero_cmd"]["ego"].append(float(torch.mean(dz[..., EGO_VEL_SLICE]).item()))
            by_variant["zero_cmd"]["contact"].append(float(torch.mean(dz[..., CONTACT_SLICE]).item()))
            by_variant["sign_flipped_cmd"]["pose"].append(float(torch.mean(df[..., POSE_SLICE]).item()))
            by_variant["sign_flipped_cmd"]["ego"].append(float(torch.mean(df[..., EGO_VEL_SLICE]).item()))
            by_variant["sign_flipped_cmd"]["contact"].append(float(torch.mean(df[..., CONTACT_SLICE]).item()))

    def _mean(vals: Sequence[float]) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    zero_pose = _mean(by_variant["zero_cmd"]["pose"])
    zero_ego = _mean(by_variant["zero_cmd"]["ego"])
    zero_contact = _mean(by_variant["zero_cmd"]["contact"])
    flip_pose = _mean(by_variant["sign_flipped_cmd"]["pose"])
    flip_ego = _mean(by_variant["sign_flipped_cmd"]["ego"])
    flip_contact = _mean(by_variant["sign_flipped_cmd"]["contact"])
    body_delta_pose_mean = float(np.mean([zero_pose, flip_pose]))
    body_delta_ego_mean = float(np.mean([zero_ego, flip_ego]))
    body_delta_contact_mean = float(np.mean([zero_contact, flip_contact]))
    command_ignored = bool(
        body_delta_pose_mean < float(command_effect_eps)
        and body_delta_ego_mean < float(command_effect_eps)
        and body_delta_contact_mean < float(command_effect_eps)
    )
    return {
        "body_delta_pose_mean": body_delta_pose_mean,
        "body_delta_ego_mean": body_delta_ego_mean,
        "body_delta_contact_mean": body_delta_contact_mean,
        "command_effect_eps": float(command_effect_eps),
        "command_ignored": command_ignored,
        "commanded_yaw_read": bool(not command_ignored),
        "variants": {
            "zero_cmd": {
                "body_delta_pose_mean": zero_pose,
                "body_delta_ego_mean": zero_ego,
                "body_delta_contact_mean": zero_contact,
            },
            "sign_flipped_cmd": {
                "body_delta_pose_mean": flip_pose,
                "body_delta_ego_mean": flip_ego,
                "body_delta_contact_mean": flip_contact,
            },
        },
    }


def _resolve_ar_capacity(
    *,
    state_dim: int,
    context_len: int,
    seam_len: int,
    horizon: int,
    masked_hidden: int,
    ar_hidden_candidates: Sequence[int],
    max_param_ratio: float,
    preferred_param_ratio: float,
) -> Dict[str, Any]:
    masked_probe = CommandedYawMaskedMiddlePredictor(
        state_dim=state_dim,
        context_len=context_len,
        seam_len=seam_len,
        horizon=horizon,
        hidden=int(masked_hidden),
    )
    masked_params = _param_count(masked_probe)
    del masked_probe

    candidates: List[Dict[str, Any]] = []
    for hidden in ar_hidden_candidates:
        probe = MinimalGoalAR(ModelConfig(state_dim=state_dim, seam_len=seam_len, hidden=int(hidden)))
        ar_params = _param_count(probe)
        del probe
        ratio = float(max(masked_params, ar_params) / max(min(masked_params, ar_params), 1))
        candidates.append(
            {
                "ar_hidden": int(hidden),
                "param_count_ar": int(ar_params),
                "param_count_masked": int(masked_params),
                "capacity_ratio": ratio,
                "within_max_ratio": bool(ratio <= float(max_param_ratio)),
                "within_preferred_ratio": bool(ratio <= float(preferred_param_ratio)),
            }
        )

    def _sort_key(rec: Mapping[str, Any]) -> Tuple[int, float, int]:
        pref_rank = 0 if bool(rec["within_preferred_ratio"]) else 1
        return (pref_rank, abs(float(rec["capacity_ratio"]) - 1.0), int(rec["ar_hidden"]))

    valid = [c for c in candidates if bool(c["within_max_ratio"])]
    selected = min(valid, key=_sort_key) if valid else None
    selected_hidden = int(selected["ar_hidden"]) if isinstance(selected, dict) else None
    selected_ratio = float(selected["capacity_ratio"]) if isinstance(selected, dict) else float("nan")
    selected_ar_params = int(selected["param_count_ar"]) if isinstance(selected, dict) else -1
    return {
        "selection_rule": "metric_blind_param_count_only",
        "max_param_ratio": float(max_param_ratio),
        "preferred_param_ratio": float(preferred_param_ratio),
        "param_count_masked": int(masked_params),
        "candidate_list": candidates,
        "selected_ar_hidden": selected_hidden,
        "selected_param_count_ar": selected_ar_params,
        "selected_capacity_ratio": selected_ratio,
        "capacity_match_found": bool(selected_hidden is not None),
    }


def _improved_focus(
    base: Mapping[str, float],
    cand: Mapping[str, float],
    *,
    pop_safe_eps: float,
    pop_eps: float,
) -> bool:
    pop_safe_up = float(cand["pop_safe_rate"]) - float(base["pop_safe_rate"])
    pop_safe_drop = float(base["pop_safe_rate"]) - float(cand["pop_safe_rate"])
    pop_down = float(base["pop_mean"]) - float(cand["pop_mean"])
    contact_down = float(base["contact_pop_mean"]) - float(cand["contact_pop_mean"])
    return bool((pop_safe_up >= float(pop_safe_eps)) or ((pop_down >= float(pop_eps)) and (contact_down >= float(pop_eps)) and (pop_safe_drop <= 0.0)))


def _monitor_regressed(
    base: Mapping[str, float],
    cand: Mapping[str, float],
    *,
    pop_safe_eps: float,
    pop_eps: float,
) -> bool:
    pop_safe_drop = float(base["pop_safe_rate"]) - float(cand["pop_safe_rate"])
    pop_worse = float(cand["pop_mean"]) - float(base["pop_mean"])
    return bool((pop_safe_drop >= float(pop_safe_eps)) or (pop_worse >= float(pop_eps)))


def _seed_metric_stats(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}


def _collect_focus_metric_stats(cells_out: Mapping[str, Any], seeds: Sequence[int]) -> Dict[str, Any]:
    metrics = ("pop_safe_rate", "pop_mean", "ego_pop_mean", "contact_pop_mean", "best_pose_d_mean")
    out: Dict[str, Any] = {}
    for cell_name, cell in cells_out.items():
        out[cell_name] = {"focus_clip": cell["cell"]["focus_clip"], "arms": {}}
        focus = cell["cell"]["focus_clip"]
        for arm in CANONICAL_ARMS:
            arm_stats: Dict[str, Any] = {}
            for m in metrics:
                vals = [
                    float(cell["seeds"][str(seed)]["arms"][arm]["per_clip"][focus][m])
                    for seed in seeds
                    if str(seed) in cell["seeds"]
                ]
                arm_stats[m] = _seed_metric_stats(vals)
            sens_pose = [
                float(cell["seeds"][str(seed)]["arms"][arm]["yaw_body_sensitivity"]["body_delta_pose_mean"])
                for seed in seeds
                if str(seed) in cell["seeds"]
            ]
            sens_ego = [
                float(cell["seeds"][str(seed)]["arms"][arm]["yaw_body_sensitivity"]["body_delta_ego_mean"])
                for seed in seeds
                if str(seed) in cell["seeds"]
            ]
            sens_contact = [
                float(cell["seeds"][str(seed)]["arms"][arm]["yaw_body_sensitivity"]["body_delta_contact_mean"])
                for seed in seeds
                if str(seed) in cell["seeds"]
            ]
            arm_stats["yaw_body_sensitivity"] = {
                "body_delta_pose_mean": _seed_metric_stats(sens_pose),
                "body_delta_ego_mean": _seed_metric_stats(sens_ego),
                "body_delta_contact_mean": _seed_metric_stats(sens_contact),
            }
            out[cell_name]["arms"][arm] = arm_stats
    return out


def _collect_plateau_table(cells_out: Mapping[str, Any], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell_name, cell in cells_out.items():
        for seed in seeds:
            seed_key = str(seed)
            if seed_key not in cell["seeds"]:
                continue
            arms = cell["seeds"][seed_key]["arms"]
            for arm in CANONICAL_ARMS:
                train = arms[arm]["train_plateau"]
                rows.append(
                    {
                        "cell": cell_name,
                        "seed": int(seed),
                        "arm": arm,
                        "loss_history_summary": {
                            "loss_first": float(train["loss_first"]),
                            "loss_last": float(train["loss_last"]),
                            "loss_min": float(train["loss_min"]),
                            "loss_max": float(train["loss_max"]),
                            "loss_mean": float(train["loss_mean"]),
                        },
                        "best_window_mean": float(train["best_window_mean"]),
                        "tail_window_mean": float(train["tail_window_mean"]),
                        "tail_over_best_ratio": float(train["tail_over_best_ratio"]),
                        "plateau_ok": bool(train["plateau_ok"]),
                        "actual_steps_used": int(train["actual_steps_used"]),
                    }
                )
    return rows


def _collect_ar_drift_table(cells_out: Mapping[str, Any], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell_name, cell in cells_out.items():
        focus = cell["cell"]["focus_clip"]
        for seed in seeds:
            seed_key = str(seed)
            if seed_key not in cell["seeds"]:
                continue
            drift = cell["seeds"][seed_key]["arms"][ARM_AR_CMD]["per_clip"][focus].get("drift_fingerprint", {})
            rows.append(
                {
                    "cell": cell_name,
                    "seed": int(seed),
                    "focus_clip": focus,
                    "drift_label": str(drift.get("label", "DRIFT_EVIDENCE_INSUFFICIENT")),
                    "first_third_mean": float(drift.get("first_third_mean", float("nan"))),
                    "last_third_mean": float(drift.get("last_third_mean", float("nan"))),
                    "last_over_first_ratio": float(drift.get("last_over_first_ratio", float("nan"))),
                    "slope_vs_t": float(drift.get("slope_vs_t", float("nan"))),
                    "evidence_sufficient": bool(drift.get("evidence_sufficient", False)),
                    "n_rollouts": int(drift.get("n_rollouts", 0)),
                    "horizon": int(drift.get("horizon", 0)),
                }
            )
    return rows


def _derive_signals(
    *,
    cells_out: Mapping[str, Any],
    seeds: Sequence[int],
    capacity_info: Mapping[str, Any],
    pop_safe_gate_threshold: float,
    improve_pop_safe_eps: float,
    improve_pop_eps: float,
) -> Dict[str, Any]:
    yaw_path_valid = True
    any_command_ignored = False
    plateau_ok_all = True
    all_arms_fail_gate = True
    continuity_stable = True
    continuity_monitor_regression = False
    ar_stable_better = True

    drift_seed_cell: Dict[str, Dict[str, str]] = {}

    for cell_name, cell in cells_out.items():
        focus = cell["cell"]["focus_clip"]
        drift_seed_cell[cell_name] = {}
        for seed in seeds:
            seed_key = str(seed)
            if seed_key not in cell["seeds"]:
                yaw_path_valid = False
                plateau_ok_all = False
                continuity_stable = False
                ar_stable_better = False
                all_arms_fail_gate = False
                drift_seed_cell[cell_name][seed_key] = "DRIFT_EVIDENCE_INSUFFICIENT"
                continue
            arms = cell["seeds"][seed_key]["arms"]

            for arm in CANONICAL_ARMS:
                sens = arms[arm]["yaw_body_sensitivity"]
                any_command_ignored = any_command_ignored or bool(sens["command_ignored"])
                yaw_path = arms[arm]["per_clip"][focus]["yaw_path"]
                if bool(yaw_path.get("posthoc_yaw_replacement", True)):
                    yaw_path_valid = False
                if arm == ARM_AR_CMD and not bool(yaw_path.get("uses_rollout_free_commanded_yaw", False)):
                    yaw_path_valid = False
                if arm in (ARM_MASKED_CMD, ARM_MASKED_CMD_SMOOTH) and bool(yaw_path.get("uses_rollout_free_commanded_yaw", False)):
                    yaw_path_valid = False

                train = arms[arm]["train_plateau"]
                plateau_ok_all = plateau_ok_all and bool(train["plateau_ok"])

                focus_metrics = arms[arm]["per_clip"][focus]
                if float(focus_metrics["pop_safe_rate"]) >= float(pop_safe_gate_threshold):
                    all_arms_fail_gate = False

            base_focus = arms[ARM_MASKED_CMD]["per_clip"][focus]
            smooth_focus = arms[ARM_MASKED_CMD_SMOOTH]["per_clip"][focus]
            ar_focus = arms[ARM_AR_CMD]["per_clip"][focus]
            if not _improved_focus(
                base_focus,
                smooth_focus,
                pop_safe_eps=improve_pop_safe_eps,
                pop_eps=improve_pop_eps,
            ):
                continuity_stable = False
            if not _improved_focus(
                base_focus,
                ar_focus,
                pop_safe_eps=improve_pop_safe_eps,
                pop_eps=improve_pop_eps,
            ):
                ar_stable_better = False

            for mclip in cell["cell"]["monitor_clips"]:
                base_m = arms[ARM_MASKED_CMD]["per_clip"][mclip]
                smooth_m = arms[ARM_MASKED_CMD_SMOOTH]["per_clip"][mclip]
                if _monitor_regressed(base_m, smooth_m, pop_safe_eps=improve_pop_safe_eps, pop_eps=improve_pop_eps):
                    continuity_monitor_regression = True

            drift_label = str(ar_focus.get("drift_fingerprint", {}).get("label", "DRIFT_EVIDENCE_INSUFFICIENT"))
            drift_seed_cell[cell_name][seed_key] = drift_label

    yaw_path_valid = bool(yaw_path_valid and (not any_command_ignored))

    total_seed_cells = 0
    drift_present_count = 0
    drift_insufficient_count = 0
    drift_cells_with_present = 0
    drift_present_in_two_thirds = False

    for cell_name in drift_seed_cell:
        labels = list(drift_seed_cell[cell_name].values())
        total_seed_cells += int(len(labels))
        present = int(sum(1 for x in labels if x == "AR_DRIFT_PRESENT"))
        insufficient = int(sum(1 for x in labels if x == "DRIFT_EVIDENCE_INSUFFICIENT"))
        drift_present_count += present
        drift_insufficient_count += insufficient
        if present > 0:
            drift_cells_with_present += 1
        if len(labels) > 0 and present >= int(np.ceil((2.0 / 3.0) * len(labels))):
            drift_present_in_two_thirds = True

    drift_evidence_sufficient = bool(total_seed_cells > 0 and drift_insufficient_count == 0)
    drift_present_reproduced = bool(drift_present_in_two_thirds or drift_cells_with_present >= 2)
    ar_drift_present = bool(drift_present_count > 0 and drift_present_reproduced)
    ar_no_drift_evidence_strong = bool(drift_evidence_sufficient and (not ar_drift_present))

    capacity_match_found = bool(capacity_info.get("capacity_match_found", False))

    continuity_signal = bool(continuity_stable and (not continuity_monitor_regression))
    ar_arch_signal = bool(capacity_match_found and plateau_ok_all and ar_stable_better and ar_no_drift_evidence_strong)

    license_grant_possible = bool(
        yaw_path_valid
        and capacity_match_found
        and plateau_ok_all
        and ar_no_drift_evidence_strong
        and (not continuity_signal)
        and (not ar_arch_signal)
        and all_arms_fail_gate
    )

    return {
        "yaw_path_valid": yaw_path_valid,
        "any_command_ignored": any_command_ignored,
        "capacity_match_found": capacity_match_found,
        "plateau_ok_all": plateau_ok_all,
        "drift_evidence_sufficient": drift_evidence_sufficient,
        "ar_drift_present": ar_drift_present,
        "ar_no_drift_evidence_strong": ar_no_drift_evidence_strong,
        "continuity_prior_arch_signal": continuity_signal,
        "ar_arch_signal": ar_arch_signal,
        "continuity_monitor_regression": continuity_monitor_regression,
        "all_arms_fail_gate": all_arms_fail_gate,
        "license_grant_possible": license_grant_possible,
        "drift_seed_cell_labels": drift_seed_cell,
        "drift_present_count": int(drift_present_count),
        "drift_insufficient_count": int(drift_insufficient_count),
        "drift_total_seed_cells": int(total_seed_cells),
    }


def resolve_precommitted_decision(signals: Mapping[str, Any]) -> Dict[str, Any]:
    labels: List[str] = []
    if not bool(signals.get("yaw_path_valid", False)):
        labels.append("INSTRUMENT_INVALID_YAW_PATH")
    if not bool(signals.get("capacity_match_found", False)):
        labels.append("INSTRUMENT_INVALID_CAPACITY")
    if not bool(signals.get("plateau_ok_all", False)):
        labels.append("INSTRUMENT_INVALID_PLATEAU")
    if not bool(signals.get("drift_evidence_sufficient", False)):
        labels.append("DRIFT_EVIDENCE_INSUFFICIENT")
    if bool(signals.get("ar_drift_present", False)):
        labels.append("AR_DRIFT_PRESENT")
        labels.append("AR_DRIFT_CONFOUNDED")
    if bool(signals.get("continuity_prior_arch_signal", False)):
        labels.append("CONTINUITY_PRIOR_ARCH_SIGNAL")
    if bool(signals.get("ar_arch_signal", False)):
        labels.append("AR_ARCH_SIGNAL")
    if bool(signals.get("license_grant_possible", False)):
        labels.append("LICENSE_DATA_OR_FORMULATION_BOTTLENECK")

    if not bool(signals.get("yaw_path_valid", False)):
        primary = "INSTRUMENT_INVALID_YAW_PATH"
    elif not bool(signals.get("capacity_match_found", False)):
        primary = "INSTRUMENT_INVALID_CAPACITY"
    elif not bool(signals.get("plateau_ok_all", False)):
        primary = "INSTRUMENT_INVALID_PLATEAU"
    elif not bool(signals.get("drift_evidence_sufficient", False)):
        primary = "INCONCLUSIVE"
    elif bool(signals.get("ar_drift_present", False)):
        primary = "AR_DRIFT_CONFOUNDED"
    elif bool(signals.get("continuity_prior_arch_signal", False)):
        primary = "CONTINUITY_PRIOR_ARCH_SIGNAL"
    elif bool(signals.get("ar_arch_signal", False)):
        primary = "AR_ARCH_SIGNAL"
    elif bool(signals.get("license_grant_possible", False)):
        primary = "LICENSE_DATA_OR_FORMULATION_BOTTLENECK"
    else:
        primary = "INCONCLUSIVE"

    if primary == "INCONCLUSIVE":
        labels.append("INCONCLUSIVE")
    labels = sorted(set(labels))
    return {
        "primary_decision": primary,
        "decision_labels": labels,
        "data_or_formulation_license_granted": bool(primary == "LICENSE_DATA_OR_FORMULATION_BOTTLENECK"),
    }


def _precommitted_reading_rules() -> List[str]:
    return [
        "若 yaw path invalid：主结论 INSTRUMENT_INVALID_YAW_PATH",
        "若 capacity mismatch：主结论 INSTRUMENT_INVALID_CAPACITY",
        "若 plateau mismatch：主结论 INSTRUMENT_INVALID_PLATEAU",
        "若 drift evidence insufficient：主结论最多 INCONCLUSIVE",
        "若 AR drift present：不能给数据瓶颈，主结论 AR_DRIFT_CONFOUNDED",
        "若 masked+smooth 在两个 focus cells、3 seeds 上稳定改善 pop_safe 或同时改善 pop_mean/contact_pop 且无 pop_safe 退化：CONTINUITY_PRIOR_ARCH_SIGNAL",
        "若 capacity+plateau matched AR 稳定优于 masked，且 per-step drift 无增长：AR_ARCH_SIGNAL",
        "若三臂在 blocking cells 仍全不过 gate，且 yaw path/capacity/plateau/drift 条件都满足，且 smoothness/AR 都无稳定帮助：LICENSE_DATA_OR_FORMULATION_BOTTLENECK",
        "clean discriminator 后不再追加再-clean 仪器；若拿到 license，下一步只能数据/formulation 决策或 PARK，且 residual head 继续 blocked",
    ]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="F5 discriminator clean rerun (capacity/plateau/per-step-drift).")
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--ar-hidden-candidates", type=str, default="192,224,256,288,320,352,384,448,512,640")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--goal-horizon", type=int, default=12)
    p.add_argument("--max-sample-retries", type=int, default=256)
    p.add_argument("--seed-list", type=str, default="0,1,2")
    p.add_argument("--cells", type=str, default="fullsup,mirror_r2l")
    p.add_argument("--allow-noncanonical", action="store_true")
    p.add_argument("--cmd-yaw-scale", type=str, default="normalized", choices=("normalized", "raw"))
    p.add_argument("--f5-loss-pose-weight", type=float, default=1.0)
    p.add_argument("--f5-loss-ego-weight", type=float, default=1.0)
    p.add_argument("--f5-loss-contact-weight", type=float, default=1.0)
    p.add_argument("--ar-seam-c1-weight", type=float, default=0.0)
    p.add_argument("--ar-seam-c1-ego-weight", type=float, default=1.0)
    p.add_argument("--ar-seam-c1-contact-weight", type=float, default=1.0)
    p.add_argument("--smooth-w", type=float, default=0.25)
    p.add_argument("--command-effect-eps", type=float, default=1e-4)
    p.add_argument("--plateau-window", type=int, default=40)
    p.add_argument("--plateau-tail-over-best-max", type=float, default=1.10)
    p.add_argument("--plateau-early-stop", action="store_true")
    p.add_argument("--max-param-ratio", type=float, default=1.15)
    p.add_argument("--preferred-param-ratio", type=float, default=1.10)
    p.add_argument("--improve-pop-safe-eps", type=float, default=0.02)
    p.add_argument("--improve-pop-eps", type=float, default=0.01)
    p.add_argument("--drift-ratio-threshold", type=float, default=1.10)
    p.add_argument("--drift-slope-threshold", type=float, default=1e-4)
    p.add_argument("--drift-min-rollouts", type=int, default=6)
    p.add_argument("--drift-min-horizon", type=int, default=6)
    p.add_argument("--pop-safe-gate-threshold", type=float, default=1.0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not args.z_features.exists():
        raise FileNotFoundError(f"z-features not found: {args.z_features}")
    if float(args.smooth_w) < 0.0:
        raise ValueError("smooth_w must be >= 0")
    if float(args.ar_seam_c1_weight) < 0.0:
        raise ValueError("ar_seam_c1_weight must be >= 0")

    seeds = _parse_int_csv(args.seed_list)
    cell_names = [c.strip() for c in str(args.cells).split(",") if c.strip()]
    bad_cells = [c for c in cell_names if c not in CELL_SPECS]
    if bad_cells:
        raise ValueError(f"unsupported cells: {bad_cells}; supported={sorted(CELL_SPECS.keys())}")
    if not cell_names:
        raise ValueError("at least one cell is required")
    if (not bool(args.allow_noncanonical)) and (tuple(seeds) != CANONICAL_SEEDS):
        raise ValueError(f"canonical clean run requires seeds={CANONICAL_SEEDS}, got {tuple(seeds)}")
    if (not bool(args.allow_noncanonical)) and (tuple(cell_names) != CANONICAL_CELLS):
        raise ValueError(f"canonical clean run requires cells={CANONICAL_CELLS}, got {tuple(cell_names)}")

    out_dir = args.out_dir
    if out_dir is None:
        date_tag = datetime.now().strftime("%Y%m%d")
        out_dir = Path(f"{DEFAULT_OUT_PREFIX}{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    states = load_clip_states(args.z_features, args.npz_root)
    sampler = InbetweenSampler(states, SamplerConfig())
    cfg = sampler.config
    normalizer = StateNormalizer(states)
    horizon = int(cfg.gap_min)
    if int(args.goal_horizon) < horizon:
        raise ValueError(f"goal_horizon must be >= horizon ({horizon}), got {args.goal_horizon}")

    hub = states[WALK_F]
    t_f = int(hub.shape[0])
    starts = [int(round(x)) % max(t_f, 1) for x in np.linspace(0, t_f - 1, int(args.n_starts))]
    state_dim = int(hub.shape[1])

    ar_hidden_candidates = _parse_hidden_candidates(args.ar_hidden_candidates)
    capacity_info = _resolve_ar_capacity(
        state_dim=state_dim,
        context_len=int(cfg.context_len),
        seam_len=int(cfg.seam_len),
        horizon=horizon,
        masked_hidden=int(args.hidden),
        ar_hidden_candidates=ar_hidden_candidates,
        max_param_ratio=float(args.max_param_ratio),
        preferred_param_ratio=float(args.preferred_param_ratio),
    )

    selected_ar_hidden = capacity_info["selected_ar_hidden"]
    if selected_ar_hidden is None:
        selected_ar_hidden = int(ar_hidden_candidates[0])

    param_counts = {
        "param_count_masked_cmd": int(capacity_info["param_count_masked"]),
        "param_count_masked_cmd_smooth": int(capacity_info["param_count_masked"]),
        "param_count_ar_cmd_capacity_matched": int(capacity_info["selected_param_count_ar"]),
        "selected_capacity_ratio": float(capacity_info["selected_capacity_ratio"]),
    }

    cells_out: Dict[str, Any] = {}
    for cell_name in cell_names:
        spec = CELL_SPECS[cell_name]
        per_seed: Dict[str, Any] = {}
        for seed in seeds:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

            model_a = CommandedYawMaskedMiddlePredictor(
                state_dim=state_dim,
                context_len=int(cfg.context_len),
                seam_len=int(cfg.seam_len),
                horizon=horizon,
                hidden=int(args.hidden),
            )
            model_b = CommandedYawMaskedMiddlePredictor(
                state_dim=state_dim,
                context_len=int(cfg.context_len),
                seam_len=int(cfg.seam_len),
                horizon=horizon,
                hidden=int(args.hidden),
            )
            model_c = MinimalGoalAR(
                ModelConfig(state_dim=state_dim, seam_len=int(cfg.seam_len), hidden=int(selected_ar_hidden))
            )

            train_a, audit_a = _train_masked_arm(
                model=model_a,
                sampler=sampler,
                normalizer=normalizer,
                max_steps=int(args.steps),
                batch=int(args.batch),
                seed=int(seed),
                holdout_policy=spec.holdout_policy,
                holdout_clip=spec.holdout_clip,
                max_sample_retries=int(args.max_sample_retries),
                f5_pose_w=float(args.f5_loss_pose_weight),
                f5_ego_w=float(args.f5_loss_ego_weight),
                f5_contact_w=float(args.f5_loss_contact_weight),
                smooth_w=float(args.smooth_w),
                cmd_scale=str(args.cmd_yaw_scale),
                lr=float(args.lr),
                with_smooth=False,
                plateau_window=int(args.plateau_window),
                plateau_tail_over_best_max=float(args.plateau_tail_over_best_max),
                early_stop_on_plateau=bool(args.plateau_early_stop),
            )
            train_b, audit_b = _train_masked_arm(
                model=model_b,
                sampler=sampler,
                normalizer=normalizer,
                max_steps=int(args.steps),
                batch=int(args.batch),
                seed=int(seed),
                holdout_policy=spec.holdout_policy,
                holdout_clip=spec.holdout_clip,
                max_sample_retries=int(args.max_sample_retries),
                f5_pose_w=float(args.f5_loss_pose_weight),
                f5_ego_w=float(args.f5_loss_ego_weight),
                f5_contact_w=float(args.f5_loss_contact_weight),
                smooth_w=float(args.smooth_w),
                cmd_scale=str(args.cmd_yaw_scale),
                lr=float(args.lr),
                with_smooth=True,
                plateau_window=int(args.plateau_window),
                plateau_tail_over_best_max=float(args.plateau_tail_over_best_max),
                early_stop_on_plateau=bool(args.plateau_early_stop),
            )
            train_c, audit_c = _train_ar_arm(
                model=model_c,
                sampler=sampler,
                normalizer=normalizer,
                max_steps=int(args.steps),
                batch=int(args.batch),
                seed=int(seed),
                holdout_policy=spec.holdout_policy,
                holdout_clip=spec.holdout_clip,
                max_sample_retries=int(args.max_sample_retries),
                f5_pose_w=float(args.f5_loss_pose_weight),
                f5_ego_w=float(args.f5_loss_ego_weight),
                f5_contact_w=float(args.f5_loss_contact_weight),
                seam_c1_weight=float(args.ar_seam_c1_weight),
                seam_c1_ego_weight=float(args.ar_seam_c1_ego_weight),
                seam_c1_contact_weight=float(args.ar_seam_c1_contact_weight),
                lr=float(args.lr),
                plateau_window=int(args.plateau_window),
                plateau_tail_over_best_max=float(args.plateau_tail_over_best_max),
                early_stop_on_plateau=bool(args.plateau_early_stop),
            )

            clip_set = [spec.focus_clip, *spec.monitor_clips]
            arms_eval: Dict[str, Dict[str, Any]] = {
                ARM_MASKED_CMD: {"per_clip": {}, "train_plateau": asdict(train_a), "sample_filter_totals": audit_a},
                ARM_MASKED_CMD_SMOOTH: {
                    "per_clip": {},
                    "train_plateau": asdict(train_b),
                    "sample_filter_totals": audit_b,
                },
                ARM_AR_CMD: {"per_clip": {}, "train_plateau": asdict(train_c), "sample_filter_totals": audit_c},
            }
            for clip in clip_set:
                arms_eval[ARM_MASKED_CMD]["per_clip"][clip] = _eval_arm_clip(
                    arm_name=ARM_MASKED_CMD,
                    model=model_a,
                    clip_name=clip,
                    states=states,
                    sampler=sampler,
                    normalizer=normalizer,
                    starts=starts,
                    goal_horizon=int(args.goal_horizon),
                    cmd_scale=str(args.cmd_yaw_scale),
                    drift_ratio_threshold=float(args.drift_ratio_threshold),
                    drift_slope_threshold=float(args.drift_slope_threshold),
                    drift_min_rollouts=int(args.drift_min_rollouts),
                    drift_min_horizon=int(args.drift_min_horizon),
                )
                arms_eval[ARM_MASKED_CMD_SMOOTH]["per_clip"][clip] = _eval_arm_clip(
                    arm_name=ARM_MASKED_CMD_SMOOTH,
                    model=model_b,
                    clip_name=clip,
                    states=states,
                    sampler=sampler,
                    normalizer=normalizer,
                    starts=starts,
                    goal_horizon=int(args.goal_horizon),
                    cmd_scale=str(args.cmd_yaw_scale),
                    drift_ratio_threshold=float(args.drift_ratio_threshold),
                    drift_slope_threshold=float(args.drift_slope_threshold),
                    drift_min_rollouts=int(args.drift_min_rollouts),
                    drift_min_horizon=int(args.drift_min_horizon),
                )
                arms_eval[ARM_AR_CMD]["per_clip"][clip] = _eval_arm_clip(
                    arm_name=ARM_AR_CMD,
                    model=model_c,
                    clip_name=clip,
                    states=states,
                    sampler=sampler,
                    normalizer=normalizer,
                    starts=starts,
                    goal_horizon=int(args.goal_horizon),
                    cmd_scale=str(args.cmd_yaw_scale),
                    drift_ratio_threshold=float(args.drift_ratio_threshold),
                    drift_slope_threshold=float(args.drift_slope_threshold),
                    drift_min_rollouts=int(args.drift_min_rollouts),
                    drift_min_horizon=int(args.drift_min_horizon),
                )

            for arm_name, model_ref in (
                (ARM_MASKED_CMD, model_a),
                (ARM_MASKED_CMD_SMOOTH, model_b),
                (ARM_AR_CMD, model_c),
            ):
                arms_eval[arm_name]["yaw_body_sensitivity"] = _yaw_body_sensitivity(
                    arm_name=arm_name,
                    model=model_ref,
                    clip_name=spec.focus_clip,
                    states=states,
                    sampler=sampler,
                    normalizer=normalizer,
                    starts=starts,
                    goal_horizon=int(args.goal_horizon),
                    cmd_scale=str(args.cmd_yaw_scale),
                    command_effect_eps=float(args.command_effect_eps),
                )

            per_seed[str(seed)] = {"arms": arms_eval}

        cells_out[cell_name] = {"cell": asdict(spec), "seeds": per_seed}

    focus_metric_stats = _collect_focus_metric_stats(cells_out, seeds)
    plateau_table = _collect_plateau_table(cells_out, seeds)
    drift_table = _collect_ar_drift_table(cells_out, seeds)

    signals = _derive_signals(
        cells_out=cells_out,
        seeds=seeds,
        capacity_info=capacity_info,
        pop_safe_gate_threshold=float(args.pop_safe_gate_threshold),
        improve_pop_safe_eps=float(args.improve_pop_safe_eps),
        improve_pop_eps=float(args.improve_pop_eps),
    )
    decision = resolve_precommitted_decision(signals)

    validity_gates = {
        "yaw_path_valid": bool(signals["yaw_path_valid"]),
        "capacity_matched": bool(signals["capacity_match_found"]),
        "plateau_matched": bool(signals["plateau_ok_all"]),
        "drift_evidence_sufficient": bool(signals["drift_evidence_sufficient"]),
        "ar_no_drift_evidence_strong": bool(signals["ar_no_drift_evidence_strong"]),
    }

    summary = {
        "task": "F5 discriminator clean rerun (capacity-match + plateau-match + per-step-vs-t drift)",
        "canonical_run_contract": {
            "focus_cells": list(CANONICAL_CELLS),
            "arms": list(CANONICAL_ARMS),
            "seeds": list(CANONICAL_SEEDS),
            "canonical_out_date": CANONICAL_OUT_DATE,
        },
        "cells": cells_out,
        "seeds": [int(s) for s in seeds],
        "capacity": {
            "resolver": capacity_info,
            "param_counts": param_counts,
        },
        "plateau_table": plateau_table,
        "per_step_drift_table": drift_table,
        "focus_cell_metrics": focus_metric_stats,
        "yaw_sensitivity": {
            "command_effect_eps": float(args.command_effect_eps),
            "any_command_ignored": bool(signals["any_command_ignored"]),
        },
        "validity_gates": validity_gates,
        "decision_signals": signals,
        "primary_decision": decision["primary_decision"],
        "decision_labels": decision["decision_labels"],
        "data_or_formulation_license_granted": decision["data_or_formulation_license_granted"],
        "precommitted_reading": _precommitted_reading_rules(),
        "config": {
            "steps": int(args.steps),
            "batch": int(args.batch),
            "hidden": int(args.hidden),
            "selected_ar_hidden": int(selected_ar_hidden),
            "ar_hidden_candidates": list(ar_hidden_candidates),
            "lr": float(args.lr),
            "n_starts": int(args.n_starts),
            "goal_horizon": int(args.goal_horizon),
            "cmd_yaw_scale": str(args.cmd_yaw_scale),
            "smooth_w": float(args.smooth_w),
            "plateau_window": int(args.plateau_window),
            "plateau_tail_over_best_max": float(args.plateau_tail_over_best_max),
            "plateau_early_stop": bool(args.plateau_early_stop),
            "max_param_ratio": float(args.max_param_ratio),
            "preferred_param_ratio": float(args.preferred_param_ratio),
            "improve_pop_safe_eps": float(args.improve_pop_safe_eps),
            "improve_pop_eps": float(args.improve_pop_eps),
            "drift_ratio_threshold": float(args.drift_ratio_threshold),
            "drift_slope_threshold": float(args.drift_slope_threshold),
            "drift_min_rollouts": int(args.drift_min_rollouts),
            "drift_min_horizon": int(args.drift_min_horizon),
            "pop_safe_gate_threshold": float(args.pop_safe_gate_threshold),
            "allow_noncanonical": bool(args.allow_noncanonical),
        },
    }

    json_path = out_dir / "f5_discriminator_clean_summary.json"
    md_path = out_dir / "f5_discriminator_clean_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# F5 Discriminator Clean Summary")
    lines.append("")
    lines.append(f"- primary_decision: {summary['primary_decision']}")
    lines.append(f"- decision_labels: {summary['decision_labels']}")
    lines.append(
        "- validity_gates: "
        f"yaw_path_valid={validity_gates['yaw_path_valid']}, "
        f"capacity_matched={validity_gates['capacity_matched']}, "
        f"plateau_matched={validity_gates['plateau_matched']}, "
        f"drift_evidence_sufficient={validity_gates['drift_evidence_sufficient']}, "
        f"ar_no_drift_evidence_strong={validity_gates['ar_no_drift_evidence_strong']}"
    )
    lines.append(
        f"- param_counts: masked={param_counts['param_count_masked_cmd']}, "
        f"masked_smooth={param_counts['param_count_masked_cmd_smooth']}, "
        f"ar={param_counts['param_count_ar_cmd_capacity_matched']}, "
        f"ratio={_fmt(param_counts['selected_capacity_ratio'], 4)}, "
        f"selected_ar_hidden={selected_ar_hidden}"
    )
    lines.append("")

    lines.append("## Plateau Table")
    lines.append("| cell | seed | arm | best_window_mean | tail_window_mean | tail/best | plateau_ok | steps_used |")
    lines.append("|---|---:|---|---:|---:|---:|---|---:|")
    for row in plateau_table:
        lines.append(
            f"| {row['cell']} | {row['seed']} | {row['arm']} | {_fmt(row['best_window_mean'],6)} | "
            f"{_fmt(row['tail_window_mean'],6)} | {_fmt(row['tail_over_best_ratio'],4)} | {row['plateau_ok']} | {row['actual_steps_used']} |"
        )
    lines.append("")

    lines.append("## Per-Step Drift Table (AR Focus)")
    lines.append("| cell | seed | label | first_third | last_third | ratio | slope_vs_t | sufficient | n_rollouts | H |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---|---:|---:|")
    for row in drift_table:
        lines.append(
            f"| {row['cell']} | {row['seed']} | {row['drift_label']} | {_fmt(row['first_third_mean'],6)} | "
            f"{_fmt(row['last_third_mean'],6)} | {_fmt(row['last_over_first_ratio'],4)} | {_fmt(row['slope_vs_t'],6)} | "
            f"{row['evidence_sufficient']} | {row['n_rollouts']} | {row['horizon']} |"
        )
    lines.append("")

    for cell_name, cell_stats in focus_metric_stats.items():
        lines.append(f"## Focus Cell Metrics: {cell_name} ({cell_stats['focus_clip']})")
        lines.append("| arm | pop_safe_rate(mean/min/max) | pop_mean(mean/min/max) | ego_vel_pop_mean(mean/min/max) | contact_pop_mean(mean/min/max) | best_pose_d_mean(mean/min/max) |")
        lines.append("|---|---|---|---|---|---|")
        for arm in CANONICAL_ARMS:
            a = cell_stats["arms"][arm]
            lines.append(
                f"| {arm} | {_fmt(a['pop_safe_rate']['mean'],3)}/{_fmt(a['pop_safe_rate']['min'],3)}/{_fmt(a['pop_safe_rate']['max'],3)} | "
                f"{_fmt(a['pop_mean']['mean'],4)}/{_fmt(a['pop_mean']['min'],4)}/{_fmt(a['pop_mean']['max'],4)} | "
                f"{_fmt(a['ego_pop_mean']['mean'],4)}/{_fmt(a['ego_pop_mean']['min'],4)}/{_fmt(a['ego_pop_mean']['max'],4)} | "
                f"{_fmt(a['contact_pop_mean']['mean'],4)}/{_fmt(a['contact_pop_mean']['min'],4)}/{_fmt(a['contact_pop_mean']['max'],4)} | "
                f"{_fmt(a['best_pose_d_mean']['mean'],4)}/{_fmt(a['best_pose_d_mean']['min'],4)}/{_fmt(a['best_pose_d_mean']['max'],4)} |"
            )
        lines.append("")

    lines.append("## Yaw Sensitivity")
    lines.append(f"- any_command_ignored: {signals['any_command_ignored']}")
    lines.append(f"- yaw_path_valid: {signals['yaw_path_valid']}")
    lines.append("")
    lines.append(
        "- data/formulation license granted: "
        f"{summary['data_or_formulation_license_granted']}"
    )

    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[decision] primary={summary['primary_decision']} labels={summary['decision_labels']}")


if __name__ == "__main__":
    main()
