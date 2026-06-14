#!/usr/bin/env python3
from __future__ import annotations

"""Minimal F4/F5 confound probe: post-hoc yaw vs commanded-yaw-conditioned masked model.

This tool keeps the non-AR masked setup and adds a commanded-yaw-conditioned arm:
  baseline arm:        ctx + seam -> middle, then post-hoc yaw replacement
  conditioned arm:     ctx + seam + cmd_yaw_middle -> middle, then canonical yaw replacement

It runs only focused cells (no sweep):
  - fullsup_l2r: holdout=none, focus clip Walk_L_To_R
  - mirror_r2l: holdout=mirror_l_r on Walk_R_To_L, while monitoring Walk_L_To_R regression
"""

import argparse
import json
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

from tools.run_action_handoff_inbetween_masked_smoke import (
    DEFAULT_NPZ_ROOT,
    DEFAULT_STEP0_COVERAGE_SUMMARY,
    DEFAULT_W1B_SUMMARY,
    DEFAULT_Z_FEATURES,
    F5_PIVOT_SLICES,
    MaskedMiddlePredictor,
    TrainStats,
    _draw_batch,
    _dump_json,
    _dump_md,
    _f5_only_gate_decision,
    _fmt,
    _groundability_from_summary,
    _load_json,
    _pivot_channel_mse_weighted,
    _sample_train_weight,
    _seam_c1_loss_weighted,
    _yaw_metrics,
)
from train.action_handoff_inbetween_commanded_yaw import replace_yaw_rate_slice
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer, evaluate_rollout_state_space
from train.action_handoff_inbetween_reach import cos_dist, summarize_absolute_self_reach
from train.data.action_handoff_inbetween import (
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    POSE_SLICE,
    SEAM_LEN_K,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    InbetweenSampler,
    SamplerConfig,
    load_clip_states,
)

DEFAULT_OUT_PREFIX = "debug_output/_tmp_action_handoff_f4f5_commanded_yaw_conditioned_"


@dataclass(frozen=True)
class CellSpec:
    name: str
    holdout_policy: str
    holdout_clip: Optional[str]
    focus_clip: str
    monitor_clips: Tuple[str, ...]


CELL_SPECS: Dict[str, CellSpec] = {
    "fullsup_l2r": CellSpec(
        name="fullsup_l2r",
        holdout_policy="none",
        holdout_clip=None,
        focus_clip="Walk_L_To_R",
        monitor_clips=(),
    ),
    "mirror_r2l": CellSpec(
        name="mirror_r2l",
        holdout_policy="mirror_l_r",
        holdout_clip="Walk_R_To_L",
        focus_clip="Walk_R_To_L",
        monitor_clips=("Walk_L_To_R",),
    ),
}


class CommandedYawMaskedMiddlePredictor(nn.Module):
    """Tiny masked predictor with explicit commanded-yaw middle trajectory input."""

    def __init__(self, state_dim: int, context_len: int, seam_len: int, horizon: int, hidden: int = 256) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_len = int(context_len)
        self.seam_len = int(seam_len)
        self.horizon = int(horizon)
        self.cmd_dim = int(horizon)
        self.input_dim = self.context_len * self.state_dim + self.seam_len * self.state_dim + self.cmd_dim
        self.output_dim = self.horizon * self.state_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.output_dim),
        )

    def forward(self, ctx: torch.Tensor, seam: torch.Tensor, cmd_yaw_middle: torch.Tensor) -> torch.Tensor:
        b = int(ctx.shape[0])
        if cmd_yaw_middle.ndim != 3 or int(cmd_yaw_middle.shape[0]) != b or int(cmd_yaw_middle.shape[2]) != 1:
            raise ValueError(
                "cmd_yaw_middle must be [B,H,1], got "
                f"{tuple(cmd_yaw_middle.shape)} with B={b}"
            )
        if int(cmd_yaw_middle.shape[1]) != int(self.horizon):
            raise ValueError(
                f"cmd_yaw_middle horizon mismatch: expected H={self.horizon}, got {int(cmd_yaw_middle.shape[1])}"
            )
        x = torch.cat(
            [
                ctx.reshape(b, -1),
                seam.reshape(b, -1),
                cmd_yaw_middle.reshape(b, -1),
            ],
            dim=-1,
        )
        y = self.net(x)
        return y.reshape(b, self.horizon, self.state_dim)


class CommandedYawLandingConditionedMaskedMiddlePredictor(nn.Module):
    """Commanded-yaw model with explicit seam-landing ego/contact conditioning."""

    def __init__(self, state_dim: int, context_len: int, seam_len: int, horizon: int, hidden: int = 256) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_len = int(context_len)
        self.seam_len = int(seam_len)
        self.horizon = int(horizon)
        self.cmd_dim = int(horizon)
        self.landing_cond_dim = int((EGO_VEL_SLICE.stop - EGO_VEL_SLICE.start) + (CONTACT_SLICE.stop - CONTACT_SLICE.start))
        self.input_dim = (
            self.context_len * self.state_dim
            + self.seam_len * self.state_dim
            + self.cmd_dim
            + self.landing_cond_dim
        )
        self.output_dim = self.horizon * self.state_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.output_dim),
        )

    def forward(
        self,
        ctx: torch.Tensor,
        seam: torch.Tensor,
        cmd_yaw_middle: torch.Tensor,
        landing_ego_contact: torch.Tensor,
    ) -> torch.Tensor:
        b = int(ctx.shape[0])
        if cmd_yaw_middle.ndim != 3 or int(cmd_yaw_middle.shape[0]) != b or int(cmd_yaw_middle.shape[2]) != 1:
            raise ValueError(
                "cmd_yaw_middle must be [B,H,1], got "
                f"{tuple(cmd_yaw_middle.shape)} with B={b}"
            )
        if int(cmd_yaw_middle.shape[1]) != int(self.horizon):
            raise ValueError(
                f"cmd_yaw_middle horizon mismatch: expected H={self.horizon}, got {int(cmd_yaw_middle.shape[1])}"
            )
        if landing_ego_contact.ndim != 2 or int(landing_ego_contact.shape[0]) != b:
            raise ValueError(
                "landing_ego_contact must be [B,4], got "
                f"{tuple(landing_ego_contact.shape)} with B={b}"
            )
        if int(landing_ego_contact.shape[1]) != int(self.landing_cond_dim):
            raise ValueError(
                f"landing_ego_contact dim mismatch: expected {self.landing_cond_dim}, got {int(landing_ego_contact.shape[1])}"
            )
        x = torch.cat(
            [
                ctx.reshape(b, -1),
                seam.reshape(b, -1),
                cmd_yaw_middle.reshape(b, -1),
                landing_ego_contact,
            ],
            dim=-1,
        )
        y = self.net(x)
        return y.reshape(b, self.horizon, self.state_dim)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="F4/F5 commanded-yaw-conditioned masked minimal confound probe.")
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--w1b-summary", type=Path, default=Path(DEFAULT_W1B_SUMMARY))
    p.add_argument("--step0-coverage-summary", type=Path, default=Path(DEFAULT_STEP0_COVERAGE_SUMMARY))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--goal-horizon", type=int, default=12)
    p.add_argument("--end-window-k", type=int, default=12)
    p.add_argument("--max-sample-retries", type=int, default=256)
    p.add_argument("--f5-loss-pose-weight", type=float, default=1.0)
    p.add_argument("--f5-loss-ego-weight", type=float, default=1.0)
    p.add_argument("--f5-loss-contact-weight", type=float, default=1.0)
    p.add_argument("--f5-seam-c1-weight", type=float, default=0.0)
    p.add_argument("--f5-seam-c1-ego-weight", type=float, default=1.0)
    p.add_argument("--f5-seam-c1-contact-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cells", type=str, default="fullsup_l2r,mirror_r2l")
    p.add_argument(
        "--conditioned-variant",
        type=str,
        default="cmd_yaw",
        choices=("cmd_yaw", "cmd_yaw_plus_landing"),
    )
    p.add_argument("--cmd-yaw-scale", type=str, default="normalized", choices=("normalized", "raw"))
    p.add_argument("--pop-safe-improve-eps", type=float, default=0.05)
    p.add_argument("--pop-mean-improve-eps", type=float, default=0.02)
    p.add_argument("--command-effect-eps", type=float, default=1e-4)
    return p


def _cmd_yaw_middle_tensor(
    middle_raw: torch.Tensor,
    middle_n: torch.Tensor,
    *,
    cmd_scale: str,
) -> torch.Tensor:
    if cmd_scale == "normalized":
        return middle_n[..., YAW_RATE_SLICE]
    if cmd_scale == "raw":
        return middle_raw[..., YAW_RATE_SLICE]
    raise ValueError(f"unsupported cmd_scale: {cmd_scale}")


def _landing_ego_contact_condition(seam_n: torch.Tensor) -> torch.Tensor:
    if seam_n.ndim != 3:
        raise ValueError(f"seam_n must be [B,K,D], got {tuple(seam_n.shape)}")
    first = seam_n[:, 0]
    ego = first[..., EGO_VEL_SLICE]
    contact = first[..., CONTACT_SLICE]
    return torch.cat([ego, contact], dim=-1)


def _forward_conditioned(
    model: nn.Module,
    ctx_n: torch.Tensor,
    seam_n: torch.Tensor,
    cmd_yaw_middle: torch.Tensor,
    *,
    conditioned_variant: str,
) -> torch.Tensor:
    if conditioned_variant == "cmd_yaw":
        return model(ctx_n, seam_n, cmd_yaw_middle)
    if conditioned_variant == "cmd_yaw_plus_landing":
        landing_cond = _landing_ego_contact_condition(seam_n)
        return model(ctx_n, seam_n, cmd_yaw_middle, landing_cond)
    raise ValueError(f"unsupported conditioned_variant: {conditioned_variant}")


def _train_arm(
    *,
    arm_name: str,
    model: nn.Module,
    sampler: InbetweenSampler,
    normalizer: StateNormalizer,
    clip_lengths: Mapping[str, int],
    steps: int,
    batch: int,
    seed: int,
    holdout_policy: str,
    holdout_clip: Optional[str],
    max_sample_retries: int,
    f5_loss_pose_weight: float,
    f5_loss_ego_weight: float,
    f5_loss_contact_weight: float,
    f5_seam_c1_weight: float,
    f5_seam_c1_ego_weight: float,
    f5_seam_c1_contact_weight: float,
    lr: float,
    cmd_scale: str,
    conditioned_variant: str,
) -> Tuple[TrainStats, Dict[str, int]]:
    del clip_lengths
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
    for _ in range(int(steps)):
        ctx, middle, seam, metas, batch_audit = _draw_batch(
            sampler,
            int(batch),
            rng,
            holdout_clip=holdout_clip,
            holdout_policy=str(holdout_policy),
            max_sample_retries=int(max_sample_retries),
        )
        del metas
        for k, v in batch_audit.items():
            sample_filter_totals[k] = int(sample_filter_totals.get(k, 0) + int(v))
        ctx_n = normalizer.normalize(ctx)
        middle_n = normalizer.normalize(middle)
        seam_n = normalizer.normalize(seam)
        if arm_name == "baseline_posthoc":
            pred_n = model(ctx_n, seam_n)
        elif arm_name == "commanded_yaw_conditioned":
            cmd_yaw_middle = _cmd_yaw_middle_tensor(middle, middle_n, cmd_scale=cmd_scale)
            pred_n = _forward_conditioned(
                model,
                ctx_n,
                seam_n,
                cmd_yaw_middle,
                conditioned_variant=conditioned_variant,
            )
        else:
            raise ValueError(f"unsupported arm_name: {arm_name}")
        sample_w = torch.ones((int(batch),), dtype=pred_n.dtype)
        loss_mid = _pivot_channel_mse_weighted(
            pred_n,
            middle_n,
            pose_w=float(f5_loss_pose_weight),
            ego_w=float(f5_loss_ego_weight),
            contact_w=float(f5_loss_contact_weight),
            sample_weights=sample_w,
        )
        loss_c1 = _seam_c1_loss_weighted(
            pred_n,
            seam_n,
            ego_w=float(f5_seam_c1_ego_weight),
            contact_w=float(f5_seam_c1_contact_weight),
            sample_weights=sample_w,
        )
        loss = loss_mid + float(f5_seam_c1_weight) * loss_c1
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("no training losses recorded")
    train_stats = TrainStats(
        loss_first=float(losses[0]),
        loss_last=float(losses[-1]),
        loss_min=float(np.min(losses)),
        loss_decreased=bool(losses[-1] < losses[0]),
    )
    return train_stats, sample_filter_totals


def _evaluate_arm(
    *,
    arm_name: str,
    model: nn.Module,
    states: Mapping[str, np.ndarray],
    sampler: InbetweenSampler,
    normalizer: StateNormalizer,
    starts: Sequence[int],
    goal_horizon: int,
    end_window_k: int,
    cmd_scale: str,
    conditioned_variant: str,
) -> Dict[str, Dict[str, Any]]:
    thr = GateThresholds()
    cfg = sampler.config
    hub = states[WALK_F]
    t_f = int(hub.shape[0])
    horizon = int(cfg.gap_min)
    model.eval()
    out: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        for clip in TURN_CLIPS:
            target = states[clip]
            g0 = int(min(goal_horizon, target.shape[0] - SEAM_LEN_K))
            target_middle = target[g0 - horizon : g0]
            goal_seam_raw = target[g0 : g0 + SEAM_LEN_K]
            target_middle_t = torch.as_tensor(target_middle, dtype=torch.float32)
            target_middle_n = normalizer.normalize(target_middle_t)
            cmd_eval = _cmd_yaw_middle_tensor(
                target_middle_t.unsqueeze(0),
                target_middle_n.unsqueeze(0),
                cmd_scale=cmd_scale,
            )

            k = int(min(end_window_k, target.shape[0]))
            centroid = target[-k:].mean(axis=0)
            self_floor = float(np.min(cos_dist(target, centroid).reshape(-1)))

            abs_cos_vals: List[float] = []
            pop_safe_vals: List[float] = []
            pose_vals: List[float] = []
            pop_vals: List[float] = []
            ego_pop_vals: List[float] = []
            contact_pop_vals: List[float] = []
            yaw_corr_vals: List[float] = []
            heading_mae_vals: List[float] = []
            goal_seam_n = normalizer.normalize(torch.as_tensor(goal_seam_raw, dtype=torch.float32)).unsqueeze(0)
            for phase in starts:
                idx = (np.arange(phase - cfg.context_len, phase) % t_f).astype(np.int64)
                ctx_raw = hub[idx]
                ctx_n = normalizer.normalize(torch.as_tensor(ctx_raw, dtype=torch.float32)).unsqueeze(0)
                if arm_name == "baseline_posthoc":
                    pred_n = model(ctx_n, goal_seam_n)[0]
                elif arm_name == "commanded_yaw_conditioned":
                    pred_n = _forward_conditioned(
                        model,
                        ctx_n,
                        goal_seam_n,
                        cmd_eval,
                        conditioned_variant=conditioned_variant,
                    )[0]
                else:
                    raise ValueError(f"unsupported arm_name: {arm_name}")
                pred_raw = normalizer.denormalize(pred_n).cpu().numpy()
                pred_cmd = replace_yaw_rate_slice(pred_raw, target_middle[:, YAW_RATE_SLICE])

                abs_cos_vals.append(float(np.min(cos_dist(pred_cmd, centroid).reshape(-1))))
                state = evaluate_rollout_state_space(pred_cmd, goal_seam_raw, normalizer.std, thr)
                pop_safe_vals.append(1.0 if bool(state["pop_safe"]) else 0.0)
                pose_vals.append(float(state["best_pose_d"]))
                pop_vals.append(float(state["pop"]))
                ri = int(state["resume_rollout_frame"])
                tj = int(state["resume_target_frame"])
                ego_diff = (pred_cmd[ri, EGO_VEL_SLICE] - goal_seam_raw[tj, EGO_VEL_SLICE]) / normalizer.std[EGO_VEL_SLICE]
                con_diff = (pred_cmd[ri, CONTACT_SLICE] - goal_seam_raw[tj, CONTACT_SLICE]) / normalizer.std[CONTACT_SLICE]
                ego_pop_vals.append(float(np.mean(np.abs(ego_diff))))
                contact_pop_vals.append(float(np.mean(np.abs(con_diff))))
                yaw = _yaw_metrics(
                    pred_cmd[:, YAW_RATE_SLICE].reshape(-1),
                    target_middle[:, YAW_RATE_SLICE].reshape(-1),
                )
                yaw_corr_vals.append(float(yaw["corr"]))
                heading_mae_vals.append(float(yaw["heading_mae_rad"]))

            self_gate = summarize_absolute_self_reach(
                abs_cos_vals,
                self_reach_abs_cos=self_floor,
                k_values=(2.0, 3.0, 5.0),
            )
            out[clip] = {
                "n": int(len(starts)),
                "reach_available": True,
                "self_reach_gate": self_gate,
                "self_reach_rate_k3": float(self_gate["rate_by_k"].get("k=3", float("nan"))),
                "pop_safe_rate": float(np.mean(pop_safe_vals)),
                "pop_mean": float(np.mean(pop_vals)),
                "ego_vel_pop_mean": float(np.mean(ego_pop_vals)),
                "contact_pop_mean": float(np.mean(contact_pop_vals)),
                "best_pose_d_mean": float(np.mean(pose_vals)),
                "best_pose_d_min": float(np.min(pose_vals)),
                "best_pose_d_max": float(np.max(pose_vals)),
                "yaw_corr": float(np.nanmean(np.asarray(yaw_corr_vals, dtype=np.float64))),
                "heading_mae_rad": float(np.nanmean(np.asarray(heading_mae_vals, dtype=np.float64))),
            }
    return out


def _yaw_sensitivity(
    *,
    model: nn.Module,
    states: Mapping[str, np.ndarray],
    sampler: InbetweenSampler,
    normalizer: StateNormalizer,
    starts: Sequence[int],
    goal_horizon: int,
    cmd_scale: str,
    command_effect_eps: float,
    conditioned_variant: str,
) -> Dict[str, Any]:
    cfg = sampler.config
    hub = states[WALK_F]
    t_f = int(hub.shape[0])
    horizon = int(cfg.gap_min)

    by_variant = {
        "zero": {"pose": [], "ego": [], "contact": []},
        "sign_flipped": {"pose": [], "ego": [], "contact": []},
    }
    model.eval()
    with torch.no_grad():
        for clip in TURN_CLIPS:
            target = states[clip]
            g0 = int(min(goal_horizon, target.shape[0] - SEAM_LEN_K))
            target_middle = target[g0 - horizon : g0]
            goal_seam_raw = target[g0 : g0 + SEAM_LEN_K]
            target_middle_t = torch.as_tensor(target_middle, dtype=torch.float32)
            target_middle_n = normalizer.normalize(target_middle_t)
            cmd_base = _cmd_yaw_middle_tensor(
                target_middle_t.unsqueeze(0),
                target_middle_n.unsqueeze(0),
                cmd_scale=cmd_scale,
            )
            cmd_zero = torch.zeros_like(cmd_base)
            cmd_flip = -cmd_base
            goal_seam_n = normalizer.normalize(torch.as_tensor(goal_seam_raw, dtype=torch.float32)).unsqueeze(0)
            for phase in starts:
                idx = (np.arange(phase - cfg.context_len, phase) % t_f).astype(np.int64)
                ctx_raw = hub[idx]
                ctx_n = normalizer.normalize(torch.as_tensor(ctx_raw, dtype=torch.float32)).unsqueeze(0)
                pred_base_n = _forward_conditioned(
                    model,
                    ctx_n,
                    goal_seam_n,
                    cmd_base,
                    conditioned_variant=conditioned_variant,
                )[0]
                pred_zero_n = _forward_conditioned(
                    model,
                    ctx_n,
                    goal_seam_n,
                    cmd_zero,
                    conditioned_variant=conditioned_variant,
                )[0]
                pred_flip_n = _forward_conditioned(
                    model,
                    ctx_n,
                    goal_seam_n,
                    cmd_flip,
                    conditioned_variant=conditioned_variant,
                )[0]

                dz = torch.abs(pred_zero_n - pred_base_n)
                df = torch.abs(pred_flip_n - pred_base_n)
                by_variant["zero"]["pose"].append(float(torch.mean(dz[..., POSE_SLICE]).item()))
                by_variant["zero"]["ego"].append(float(torch.mean(dz[..., EGO_VEL_SLICE]).item()))
                by_variant["zero"]["contact"].append(float(torch.mean(dz[..., CONTACT_SLICE]).item()))
                by_variant["sign_flipped"]["pose"].append(float(torch.mean(df[..., POSE_SLICE]).item()))
                by_variant["sign_flipped"]["ego"].append(float(torch.mean(df[..., EGO_VEL_SLICE]).item()))
                by_variant["sign_flipped"]["contact"].append(float(torch.mean(df[..., CONTACT_SLICE]).item()))

    def _m(vals: Sequence[float]) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.mean(arr)) if arr.size > 0 else float("nan")

    zero_pose = _m(by_variant["zero"]["pose"])
    zero_ego = _m(by_variant["zero"]["ego"])
    zero_contact = _m(by_variant["zero"]["contact"])
    flip_pose = _m(by_variant["sign_flipped"]["pose"])
    flip_ego = _m(by_variant["sign_flipped"]["ego"])
    flip_contact = _m(by_variant["sign_flipped"]["contact"])
    body_delta_pose_mean = float(np.mean([zero_pose, flip_pose]))
    body_delta_ego_mean = float(np.mean([zero_ego, flip_ego]))
    body_delta_contact_mean = float(np.mean([zero_contact, flip_contact]))
    command_ignored = bool(
        body_delta_pose_mean < float(command_effect_eps)
        and body_delta_ego_mean < float(command_effect_eps)
        and body_delta_contact_mean < float(command_effect_eps)
    )
    return {
        "cmd_yaw_scale": cmd_scale,
        "space": "normalized_prediction_space",
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
        "body_delta_pose_mean": body_delta_pose_mean,
        "body_delta_ego_mean": body_delta_ego_mean,
        "body_delta_contact_mean": body_delta_contact_mean,
        "command_effect_eps": float(command_effect_eps),
        "command_ignored": command_ignored,
    }


def _clip_delta(base: Mapping[str, Any], cond: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "pop_safe_rate_delta": float(cond["pop_safe_rate"] - base["pop_safe_rate"]),
        "pop_mean_delta": float(cond["pop_mean"] - base["pop_mean"]),
        "ego_vel_pop_mean_delta": float(cond["ego_vel_pop_mean"] - base["ego_vel_pop_mean"]),
        "contact_pop_mean_delta": float(cond["contact_pop_mean"] - base["contact_pop_mean"]),
        "best_pose_d_mean_delta": float(cond["best_pose_d_mean"] - base["best_pose_d_mean"]),
    }


def _is_significant_f5_improvement(
    delta: Mapping[str, float],
    *,
    pop_safe_improve_eps: float,
    pop_mean_improve_eps: float,
) -> bool:
    pop_safe_delta = float(delta["pop_safe_rate_delta"])
    pop_delta = float(delta["pop_mean_delta"])
    contact_delta = float(delta["contact_pop_mean_delta"])
    # F5-first rule:
    # 1) pop_safe lift is always a significant improvement.
    # 2) without pop_safe lift, accept only if pop/contact are clearly better and pop_safe does not regress.
    if pop_safe_delta >= float(pop_safe_improve_eps):
        return True
    if pop_safe_delta < 0.0:
        return False
    return bool(
        pop_delta <= -float(pop_mean_improve_eps)
        and contact_delta <= -float(pop_mean_improve_eps)
    )


def _yaw_positive_control_ok(per_clip: Mapping[str, Mapping[str, Any]]) -> bool:
    for clip in TURN_CLIPS:
        row = per_clip.get(clip, {})
        corr = float(row.get("yaw_corr", float("nan")))
        mae = float(row.get("heading_mae_rad", float("nan")))
        if not np.isfinite(corr) or not np.isfinite(mae):
            return False
        if corr < 0.99 or mae > 1e-4:
            return False
    return True


def _run_cell(
    *,
    spec: CellSpec,
    args: argparse.Namespace,
    states: Mapping[str, np.ndarray],
    sampler: InbetweenSampler,
    normalizer: StateNormalizer,
    clip_lengths: Mapping[str, int],
    starts: Sequence[int],
    baseline_free: Mapping[str, Mapping[str, Any]],
    baseline_pinned: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    horizon = int(sampler.config.gap_min)
    state_dim = int(states[WALK_F].shape[1])
    baseline_model = MaskedMiddlePredictor(
        state_dim=state_dim,
        context_len=int(sampler.config.context_len),
        seam_len=int(sampler.config.seam_len),
        horizon=horizon,
        hidden=int(args.hidden),
    )
    if str(args.conditioned_variant) == "cmd_yaw":
        conditioned_model = CommandedYawMaskedMiddlePredictor(
            state_dim=state_dim,
            context_len=int(sampler.config.context_len),
            seam_len=int(sampler.config.seam_len),
            horizon=horizon,
            hidden=int(args.hidden),
        )
    elif str(args.conditioned_variant) == "cmd_yaw_plus_landing":
        conditioned_model = CommandedYawLandingConditionedMaskedMiddlePredictor(
            state_dim=state_dim,
            context_len=int(sampler.config.context_len),
            seam_len=int(sampler.config.seam_len),
            horizon=horizon,
            hidden=int(args.hidden),
        )
    else:
        raise ValueError(f"unsupported conditioned_variant: {args.conditioned_variant}")

    baseline_train, baseline_audit = _train_arm(
        arm_name="baseline_posthoc",
        model=baseline_model,
        sampler=sampler,
        normalizer=normalizer,
        clip_lengths=clip_lengths,
        steps=int(args.steps),
        batch=int(args.batch),
        seed=int(args.seed),
        holdout_policy=spec.holdout_policy,
        holdout_clip=spec.holdout_clip,
        max_sample_retries=int(args.max_sample_retries),
        f5_loss_pose_weight=float(args.f5_loss_pose_weight),
        f5_loss_ego_weight=float(args.f5_loss_ego_weight),
        f5_loss_contact_weight=float(args.f5_loss_contact_weight),
        f5_seam_c1_weight=float(args.f5_seam_c1_weight),
        f5_seam_c1_ego_weight=float(args.f5_seam_c1_ego_weight),
        f5_seam_c1_contact_weight=float(args.f5_seam_c1_contact_weight),
        lr=float(args.lr),
        cmd_scale=str(args.cmd_yaw_scale),
        conditioned_variant=str(args.conditioned_variant),
    )
    conditioned_train, conditioned_audit = _train_arm(
        arm_name="commanded_yaw_conditioned",
        model=conditioned_model,
        sampler=sampler,
        normalizer=normalizer,
        clip_lengths=clip_lengths,
        steps=int(args.steps),
        batch=int(args.batch),
        seed=int(args.seed),
        holdout_policy=spec.holdout_policy,
        holdout_clip=spec.holdout_clip,
        max_sample_retries=int(args.max_sample_retries),
        f5_loss_pose_weight=float(args.f5_loss_pose_weight),
        f5_loss_ego_weight=float(args.f5_loss_ego_weight),
        f5_loss_contact_weight=float(args.f5_loss_contact_weight),
        f5_seam_c1_weight=float(args.f5_seam_c1_weight),
        f5_seam_c1_ego_weight=float(args.f5_seam_c1_ego_weight),
        f5_seam_c1_contact_weight=float(args.f5_seam_c1_contact_weight),
        lr=float(args.lr),
        cmd_scale=str(args.cmd_yaw_scale),
        conditioned_variant=str(args.conditioned_variant),
    )

    baseline_per_clip = _evaluate_arm(
        arm_name="baseline_posthoc",
        model=baseline_model,
        states=states,
        sampler=sampler,
        normalizer=normalizer,
        starts=starts,
        goal_horizon=int(args.goal_horizon),
        end_window_k=int(args.end_window_k),
        cmd_scale=str(args.cmd_yaw_scale),
        conditioned_variant=str(args.conditioned_variant),
    )
    conditioned_per_clip = _evaluate_arm(
        arm_name="commanded_yaw_conditioned",
        model=conditioned_model,
        states=states,
        sampler=sampler,
        normalizer=normalizer,
        starts=starts,
        goal_horizon=int(args.goal_horizon),
        end_window_k=int(args.end_window_k),
        cmd_scale=str(args.cmd_yaw_scale),
        conditioned_variant=str(args.conditioned_variant),
    )
    baseline_gate = _f5_only_gate_decision(
        baseline_per_clip,
        baseline_free=baseline_free,
        baseline_pinned=baseline_pinned,
        pose_degradation_tol=0.05,
        require_reach_lift=False,
    )
    conditioned_gate = _f5_only_gate_decision(
        conditioned_per_clip,
        baseline_free=baseline_free,
        baseline_pinned=baseline_pinned,
        pose_degradation_tol=0.05,
        require_reach_lift=False,
    )
    sensitivity = _yaw_sensitivity(
        model=conditioned_model,
        states=states,
        sampler=sampler,
        normalizer=normalizer,
        starts=starts,
        goal_horizon=int(args.goal_horizon),
        cmd_scale=str(args.cmd_yaw_scale),
        command_effect_eps=float(args.command_effect_eps),
        conditioned_variant=str(args.conditioned_variant),
    )

    compare_clips: List[str] = [spec.focus_clip, *spec.monitor_clips]
    per_clip_delta: Dict[str, Any] = {}
    key_improved: List[bool] = []
    key_pop_fail: List[bool] = []
    for clip in compare_clips:
        b = baseline_per_clip[clip]
        c = conditioned_per_clip[clip]
        d = _clip_delta(b, c)
        improved = _is_significant_f5_improvement(
            d,
            pop_safe_improve_eps=float(args.pop_safe_improve_eps),
            pop_mean_improve_eps=float(args.pop_mean_improve_eps),
        )
        rec = {
            "baseline": {k: float(b[k]) for k in ("pop_safe_rate", "pop_mean", "ego_vel_pop_mean", "contact_pop_mean", "best_pose_d_mean")},
            "conditioned": {k: float(c[k]) for k in ("pop_safe_rate", "pop_mean", "ego_vel_pop_mean", "contact_pop_mean", "best_pose_d_mean")},
            "delta": d,
            "significant_f5_improvement": bool(improved),
        }
        per_clip_delta[clip] = rec
        if clip == spec.focus_clip:
            key_improved.append(bool(improved))
            key_pop_fail.append(float(c["pop_safe_rate"]) <= 0.0)

    comparison = {
        "focus_clip": spec.focus_clip,
        "monitor_clips": list(spec.monitor_clips),
        "per_clip_delta": per_clip_delta,
        "key_improved_any": bool(any(key_improved)),
        "key_pop_safe_still_fail_all": bool(all(key_pop_fail)) if key_pop_fail else False,
        "pop_safe_improve_eps": float(args.pop_safe_improve_eps),
        "pop_mean_improve_eps": float(args.pop_mean_improve_eps),
    }

    return {
        "cell": asdict(spec),
        "train_config": {
            "steps": int(args.steps),
            "batch": int(args.batch),
            "hidden": int(args.hidden),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "cmd_yaw_scale": str(args.cmd_yaw_scale),
            "conditioned_variant": str(args.conditioned_variant),
        },
        "arms": {
            "baseline_posthoc": {
                "train_stats": asdict(baseline_train),
                "sample_filter_totals": baseline_audit,
                "per_clip": baseline_per_clip,
                "f5_gate": baseline_gate,
                "yaw_positive_control_ok": _yaw_positive_control_ok(baseline_per_clip),
            },
            "commanded_yaw_conditioned": {
                "train_stats": asdict(conditioned_train),
                "sample_filter_totals": conditioned_audit,
                "per_clip": conditioned_per_clip,
                "f5_gate": conditioned_gate,
                "yaw_positive_control_ok": _yaw_positive_control_ok(conditioned_per_clip),
                "yaw_sensitivity": sensitivity,
            },
        },
        "comparison": comparison,
    }


def _final_decision(cells: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    missing = []
    command_ignored_cells = []
    improved_cells = []
    all_key_pop_fail = True
    positive_control_ok = True
    for cell_name, rec in cells.items():
        comp = rec.get("comparison", {})
        arms = rec.get("arms", {})
        cond = arms.get("commanded_yaw_conditioned", {})
        sens = cond.get("yaw_sensitivity", {})
        if bool(sens.get("command_ignored", False)):
            command_ignored_cells.append(cell_name)
        if bool(comp.get("key_improved_any", False)):
            improved_cells.append(cell_name)
        all_key_pop_fail = all_key_pop_fail and bool(comp.get("key_pop_safe_still_fail_all", False))
        positive_control_ok = positive_control_ok and bool(arms.get("baseline_posthoc", {}).get("yaw_positive_control_ok", False))
        positive_control_ok = positive_control_ok and bool(cond.get("yaw_positive_control_ok", False))
        if "comparison" not in rec or "arms" not in rec:
            missing.append(cell_name)

    if missing:
        return {
            "decision": "INCONCLUSIVE",
            "reason": f"missing required fields in cells={missing}",
            "command_ignored_cells": command_ignored_cells,
            "improved_cells": improved_cells,
        }
    if not positive_control_ok:
        return {
            "decision": "INCONCLUSIVE",
            "reason": "yaw positive-control failed in at least one arm/cell",
            "command_ignored_cells": command_ignored_cells,
            "improved_cells": improved_cells,
        }
    if command_ignored_cells:
        return {
            "decision": "COMMAND_IGNORED",
            "reason": "yaw sensitivity shows near-zero body response to commanded yaw in at least one cell",
            "command_ignored_cells": command_ignored_cells,
            "improved_cells": improved_cells,
        }
    if improved_cells:
        return {
            "decision": "CONDITIONING_CONFIRMED_IMPROVES_F5",
            "reason": "commanded-yaw conditioning materially improves at least one focus clip F5 metric",
            "command_ignored_cells": command_ignored_cells,
            "improved_cells": improved_cells,
        }
    if all_key_pop_fail:
        return {
            "decision": "CONFIRMED_F5_BOTTLENECK",
            "reason": "conditioning is read by model but key focus clips still fail pop-safe without significant F5 improvement",
            "command_ignored_cells": command_ignored_cells,
            "improved_cells": improved_cells,
        }
    return {
        "decision": "INCONCLUSIVE",
        "reason": "mixed signals across cells; cannot assign bottleneck/improvement cleanly",
        "command_ignored_cells": command_ignored_cells,
        "improved_cells": improved_cells,
    }


def main() -> None:
    args = _build_parser().parse_args()
    for p in (args.z_features, args.w1b_summary, args.step0_coverage_summary):
        if not p.exists():
            raise FileNotFoundError(f"required input not found: {p}")
    if float(args.f5_seam_c1_weight) < 0.0:
        raise ValueError("f5_seam_c1_weight must be >= 0.0")
    if float(args.pop_safe_improve_eps) < 0.0 or float(args.pop_mean_improve_eps) < 0.0:
        raise ValueError("improvement eps values must be >= 0.0")
    if float(args.command_effect_eps) < 0.0:
        raise ValueError("command_effect_eps must be >= 0.0")

    cell_names = [c.strip() for c in str(args.cells).split(",") if c.strip()]
    if not cell_names:
        raise ValueError("at least one cell must be selected")
    bad_cells = [c for c in cell_names if c not in CELL_SPECS]
    if bad_cells:
        raise ValueError(f"unsupported cells: {bad_cells}; supported={sorted(CELL_SPECS.keys())}")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"{DEFAULT_OUT_PREFIX}{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    states = load_clip_states(args.z_features, args.npz_root)
    sampler = InbetweenSampler(states, SamplerConfig())
    cfg = sampler.config
    normalizer = StateNormalizer(states)
    clip_lengths = {k: int(v.shape[0]) for k, v in states.items()}
    horizon = int(cfg.gap_min)
    if int(args.goal_horizon) < horizon:
        raise ValueError(f"goal_horizon must be >= horizon ({horizon}), got {args.goal_horizon}")

    w1b = _load_json(args.w1b_summary)
    step0_cov = _load_json(args.step0_coverage_summary)
    groundable_by_clip = _groundability_from_summary(step0_cov, sampler)
    phase2 = w1b["evaluated_objects"]["phase2_trained_goal"]["per_clip"]
    baseline_free = {clip: phase2[clip]["free_no_goal"] for clip in TURN_CLIPS}
    baseline_pinned = {clip: phase2[clip]["pinned_goal"] for clip in TURN_CLIPS}

    hub = states[WALK_F]
    t_f = int(hub.shape[0])
    starts = [int(round(x)) % max(t_f, 1) for x in np.linspace(0, t_f - 1, int(args.n_starts))]

    cells_out: Dict[str, Any] = {}
    for cell_name in cell_names:
        spec = CELL_SPECS[cell_name]
        cells_out[cell_name] = _run_cell(
            spec=spec,
            args=args,
            states=states,
            sampler=sampler,
            normalizer=normalizer,
            clip_lengths=clip_lengths,
            starts=starts,
            baseline_free=baseline_free,
            baseline_pinned=baseline_pinned,
        )

    decision = _final_decision(cells_out)

    summary = {
        "task": "F4/F5 commanded-yaw-conditioned masked minimal confound probe",
        "cmd_yaw_input": {
            "shape": "[B,H,1]",
            "scale": str(args.cmd_yaw_scale),
            "note": "cmd_yaw_middle is from target middle yaw trajectory",
        },
        "conditioned_variant": str(args.conditioned_variant),
        "focus_cells": cell_names,
        "groundability": {clip: bool(groundable_by_clip.get(clip, False)) for clip in TURN_CLIPS},
        "cells": cells_out,
        "decision": decision,
        "f5_pivot_slices": [(int(s.start), int(s.stop)) for s in F5_PIVOT_SLICES],
        "baseline_ref": str(args.w1b_summary.resolve()),
    }

    json_path = out_dir / "commanded_yaw_conditioned_summary.json"
    md_path = out_dir / "commanded_yaw_conditioned_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# F4/F5 Commanded-Yaw-Conditioned Minimal Confound Probe")
    lines.append("")
    lines.append(f"- decision: {decision['decision']}")
    lines.append(f"- reason: {decision['reason']}")
    lines.append(f"- cmd_yaw_input_scale: {args.cmd_yaw_scale}")
    lines.append(f"- cells: {cell_names}")
    lines.append("")
    for cell_name in cell_names:
        cell = cells_out[cell_name]
        cmp_row = cell["comparison"]
        sens = cell["arms"]["commanded_yaw_conditioned"]["yaw_sensitivity"]
        lines.append(f"## cell: {cell_name}")
        lines.append(
            f"- holdout_policy={cell['cell']['holdout_policy']}, holdout_clip={cell['cell']['holdout_clip']}, "
            f"focus_clip={cell['cell']['focus_clip']}, monitor={cell['cell']['monitor_clips']}"
        )
        lines.append(
            f"- yaw_sensitivity(body_delta_pose/ego/contact): "
            f"{_fmt(sens['body_delta_pose_mean'],6)} / {_fmt(sens['body_delta_ego_mean'],6)} / {_fmt(sens['body_delta_contact_mean'],6)} "
            f"(ignored={sens['command_ignored']}, eps={_fmt(sens['command_effect_eps'],6)})"
        )
        lines.append(
            f"- key_improved_any={cmp_row['key_improved_any']}, "
            f"key_pop_safe_still_fail_all={cmp_row['key_pop_safe_still_fail_all']}"
        )
        lines.append("")
        lines.append("| clip | arm | pop_safe | pop_mean | ego_pop | contact_pop | best_pose_d |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for clip in [cell["cell"]["focus_clip"], *cell["cell"]["monitor_clips"]]:
            b = cmp_row["per_clip_delta"][clip]["baseline"]
            c = cmp_row["per_clip_delta"][clip]["conditioned"]
            lines.append(
                f"| {clip} | baseline_posthoc | {_fmt(b['pop_safe_rate'],3)} | {_fmt(b['pop_mean'],4)} | "
                f"{_fmt(b['ego_vel_pop_mean'],4)} | {_fmt(b['contact_pop_mean'],4)} | {_fmt(b['best_pose_d_mean'],4)} |"
            )
            lines.append(
                f"| {clip} | commanded_conditioned | {_fmt(c['pop_safe_rate'],3)} | {_fmt(c['pop_mean'],4)} | "
                f"{_fmt(c['ego_vel_pop_mean'],4)} | {_fmt(c['contact_pop_mean'],4)} | {_fmt(c['best_pose_d_mean'],4)} |"
            )
            d = cmp_row["per_clip_delta"][clip]["delta"]
            lines.append(
                f"| {clip} | delta(cond-base) | {_fmt(d['pop_safe_rate_delta'],3)} | {_fmt(d['pop_mean_delta'],4)} | "
                f"{_fmt(d['ego_vel_pop_mean_delta'],4)} | {_fmt(d['contact_pop_mean_delta'],4)} | {_fmt(d['best_pose_d_mean_delta'],4)} |"
            )
        lines.append("")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[decision] {decision['decision']} :: {decision['reason']}")


if __name__ == "__main__":
    main()
