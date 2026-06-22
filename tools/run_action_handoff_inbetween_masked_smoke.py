#!/usr/bin/env python3
"""W1c STEP-2(B) minimal masked in-betweening smoke (no B4/seam runtime).

Fallback path after STEP-0 says A is blocked. This tool trains a tiny non-AR masked
middle predictor on the existing 5-clip states only (no new data), then evaluates with
F5 acceptance fields (`pop_safe_rate`, `best_pose_d`; optional reach), while keeping yaw
as commanded positive-control diagnostics only. It also persists a reproducible state artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.action_handoff_inbetween_goal_injection import (
    DEFAULT_POSE_DEGRADATION_TOL,
    DEFAULT_SELF_REACH_RATE_LIFT,
    DEFAULT_YAW_MAE_TAU_RAD,
    joint_action_binding_gate_decision,
)
from train.action_handoff_inbetween_commanded_yaw import replace_yaw_rate_slice
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer, evaluate_rollout_state_space
from train.action_handoff_inbetween_reach import cos_dist, summarize_absolute_self_reach
from train.data.action_handoff_inbetween import (
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    FPS,
    POSE_SLICE,
    SAMPLE_TYPE_AUGMENTED,
    SAMPLE_TYPE_GROUNDED,
    SAMPLE_TYPE_WITHIN,
    SEAM_LEN_K,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    InbetweenSampler,
    SamplerConfig,
    load_clip_states,
)

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
DEFAULT_W1B_SUMMARY = "debug_output/_tmp_action_handoff_w1c_gate_migration_20260530/gate_migration_eval_summary.json"
DEFAULT_STEP0_COVERAGE_SUMMARY = "debug_output/_tmp_action_handoff_w1c_step0_20260530/inbetween_sampler_coverage_summary.json"
F5_PIVOT_SLICES = (POSE_SLICE, EGO_VEL_SLICE, CONTACT_SLICE)


class MaskedMiddlePredictor(nn.Module):
    """Tiny masked middle predictor: (ctx, future seam) -> middle."""

    def __init__(self, state_dim: int, context_len: int, seam_len: int, horizon: int, hidden: int = 256) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_len = int(context_len)
        self.seam_len = int(seam_len)
        self.horizon = int(horizon)
        inp = self.context_len * self.state_dim + self.seam_len * self.state_dim
        out = self.horizon * self.state_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, ctx: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ctx.reshape(ctx.shape[0], -1), seam.reshape(seam.shape[0], -1)], dim=-1)
        y = self.net(x)
        return y.reshape(ctx.shape[0], self.horizon, self.state_dim)


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "null"
    return f"{float(v):.{digits}f}"


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _yaw_from_rate(yaw_rate: np.ndarray, fps: float) -> np.ndarray:
    yr = np.asarray(yaw_rate, dtype=np.float64).reshape(-1)
    return np.cumsum(yr) / max(float(fps), 1e-8)


def _yaw_metrics(pred_yaw_rate: np.ndarray, tgt_yaw_rate: np.ndarray) -> Dict[str, float]:
    pred = _yaw_from_rate(pred_yaw_rate, FPS)
    tgt = _yaw_from_rate(tgt_yaw_rate, FPS)
    n = int(min(pred.shape[0], tgt.shape[0]))
    if n < 2:
        return {"corr": float("nan"), "heading_mae_rad": float("nan")}
    a = pred[:n]
    b = tgt[:n]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    mae = float(np.mean(np.abs(a - b)))
    return {"corr": corr, "heading_mae_rad": mae}


def _pivot_channel_mse_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pose_w: float,
    ego_w: float,
    contact_w: float,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted F5 loss over pivot channels (pose/ego/contact), optional per-sample weights."""
    per = (
        float(pose_w) * torch.mean((pred[..., POSE_SLICE] - target[..., POSE_SLICE]) ** 2, dim=(1, 2))
        + float(ego_w) * torch.mean((pred[..., EGO_VEL_SLICE] - target[..., EGO_VEL_SLICE]) ** 2, dim=(1, 2))
        + float(contact_w) * torch.mean((pred[..., CONTACT_SLICE] - target[..., CONTACT_SLICE]) ** 2, dim=(1, 2))
    ) / max(float(pose_w + ego_w + contact_w), 1e-8)
    if sample_weights is None:
        return torch.mean(per)
    sw = sample_weights.to(device=per.device, dtype=per.dtype)
    sw = sw / torch.clamp(torch.mean(sw), min=1e-8)
    return torch.mean(per * sw)


def _seam_c1_loss_weighted(
    pred_middle: torch.Tensor,
    seam_target: torch.Tensor,
    *,
    ego_w: float,
    contact_w: float,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted seam-C1 continuity on ego_vel/contact between pred last and seam first."""
    last = pred_middle[:, -1]
    first = seam_target[:, 0]
    per = (
        float(ego_w) * torch.mean((last[..., EGO_VEL_SLICE] - first[..., EGO_VEL_SLICE]) ** 2, dim=1)
        + float(contact_w) * torch.mean((last[..., CONTACT_SLICE] - first[..., CONTACT_SLICE]) ** 2, dim=1)
    ) / max(float(ego_w + contact_w), 1e-8)
    if sample_weights is None:
        return torch.mean(per)
    sw = sample_weights.to(device=per.device, dtype=per.dtype)
    sw = sw / torch.clamp(torch.mean(sw), min=1e-8)
    return torch.mean(per * sw)


def _meta_target_clip(meta: Mapping[str, Any]) -> Optional[str]:
    turn = meta.get("turn_clip")
    if turn:
        return str(turn)
    clip = meta.get("clip")
    if clip:
        return str(clip)
    return None


def _meta_seam_pos(meta: Mapping[str, Any]) -> Optional[int]:
    if "seam_start" in meta:
        try:
            return int(meta["seam_start"])
        except Exception:
            return None
    if "onset_used" in meta and "horizon" in meta:
        try:
            return int(meta["onset_used"]) + int(meta["horizon"])
        except Exception:
            return None
    if "middle_start" in meta and "gap" in meta:
        try:
            return int(meta["middle_start"]) + int(meta["gap"])
        except Exception:
            return None
    return None


def _seam_phase_bucket(meta: Mapping[str, Any], clip_len: int, *, n_buckets: int = 4) -> Optional[int]:
    pos = _meta_seam_pos(meta)
    if pos is None or clip_len <= 1:
        return None
    idx = int(np.clip(pos, 0, clip_len - 1))
    return int(min(n_buckets - 1, (idx * n_buckets) // clip_len))


def _sample_train_weight(
    meta: Mapping[str, Any],
    clip_lengths: Mapping[str, int],
    *,
    focus_clip: Optional[str],
    focus_phase_bucket: Optional[int],
    focus_weight: float,
) -> float:
    if not focus_clip:
        return 1.0
    clip = _meta_target_clip(meta)
    if clip != str(focus_clip):
        return 1.0
    w = float(max(focus_weight, 1.0))
    if focus_phase_bucket is None:
        return w
    t = int(clip_lengths.get(clip, 0))
    b = _seam_phase_bucket(meta, t, n_buckets=4)
    if b is None:
        return 1.0
    return w if b == int(focus_phase_bucket) else 1.0


def _sample_is_removed_by_policy(
    meta: Mapping[str, Any],
    *,
    holdout_clip: Optional[str],
    holdout_policy: str,
) -> Tuple[bool, str]:
    if holdout_clip is None or holdout_policy == "none":
        return False, "accepted"

    sample_type = str(meta.get("sample_type", ""))
    clip = str(meta.get("clip", ""))
    turn_clip = str(meta.get("turn_clip", ""))
    base_type = str(meta.get("base_type", ""))
    fallback = meta.get("fallback")

    if holdout_policy == "mirror_l_r":
        # Remove only grounded cross-manifold supervision for held-out g
        # (Walk_F -> g onset), while keeping g within-clip samples.
        direct_grounded = (
            sample_type == SAMPLE_TYPE_GROUNDED
            and turn_clip == holdout_clip
            and fallback != "within_clip"
        )
        augmented_from_grounded = (
            sample_type == SAMPLE_TYPE_AUGMENTED
            and base_type == SAMPLE_TYPE_GROUNDED
            and turn_clip == holdout_clip
            and fallback != "within_clip"
        )
        if direct_grounded or augmented_from_grounded:
            return True, "removed_grounded_cross_manifold"
        return False, "accepted"

    if holdout_policy == "full_holdout":
        if clip == holdout_clip or turn_clip == holdout_clip:
            return True, "removed_full_holdout"
        return False, "accepted"

    raise ValueError(f"unsupported holdout_policy: {holdout_policy}")


def _draw_batch(
    sampler: InbetweenSampler,
    batch: int,
    rng: np.random.Generator,
    *,
    holdout_clip: Optional[str],
    holdout_policy: str,
    max_sample_retries: int,
):
    ctx, middle, seam = [], [], []
    metas: List[Dict[str, Any]] = []
    audit = {
        "attempted": 0,
        "accepted": 0,
        "removed_grounded_cross_manifold": 0,
        "removed_full_holdout": 0,
    }
    for _ in range(int(batch)):
        accepted = False
        for _try in range(int(max_sample_retries)):
            s = sampler.sample(rng, progress=0.0)
            audit["attempted"] += 1
            removed, reason = _sample_is_removed_by_policy(
                s.meta,
                holdout_clip=holdout_clip,
                holdout_policy=holdout_policy,
            )
            if removed:
                if reason not in audit:
                    audit[reason] = 0
                audit[reason] += 1
                continue
            ctx.append(s.ctx)
            middle.append(s.gt_middle)
            seam.append(s.seam_target)
            metas.append(dict(s.meta))
            audit["accepted"] += 1
            accepted = True
            break
        if not accepted:
            raise RuntimeError(
                "failed to draw a batch sample under holdout constraints "
                f"(policy={holdout_policy}, holdout_clip={holdout_clip}, max_retries={max_sample_retries})"
            )
    return (
        torch.as_tensor(np.stack(ctx), dtype=torch.float32),
        torch.as_tensor(np.stack(middle), dtype=torch.float32),
        torch.as_tensor(np.stack(seam), dtype=torch.float32),
        metas,
        audit,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="W1c STEP-2(B) masked in-betweening smoke.")
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
    p.add_argument("--holdout-policy", type=str, default="none", choices=("none", "mirror_l_r", "full_holdout"))
    p.add_argument("--holdout-clip", type=str, default=None, choices=TURN_CLIPS)
    p.add_argument("--max-sample-retries", type=int, default=256)
    p.add_argument("--f5-loss-pose-weight", type=float, default=1.0)
    p.add_argument("--f5-loss-ego-weight", type=float, default=1.0)
    p.add_argument("--f5-loss-contact-weight", type=float, default=1.0)
    p.add_argument("--f5-seam-c1-weight", type=float, default=0.0)
    p.add_argument("--f5-seam-c1-ego-weight", type=float, default=1.0)
    p.add_argument("--f5-seam-c1-contact-weight", type=float, default=1.0)
    p.add_argument("--focus-clip", type=str, default=None, choices=TURN_CLIPS)
    p.add_argument("--focus-phase-bucket", type=int, default=None, choices=(0, 1, 2, 3))
    p.add_argument("--focus-sample-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return p


@dataclass
class TrainStats:
    loss_first: float
    loss_last: float
    loss_min: float
    loss_decreased: bool


def _groundability_from_summary(
    coverage_summary: Mapping[str, Any],
    sampler: InbetweenSampler,
) -> Dict[str, bool]:
    grounded = {}
    fb = coverage_summary.get("grounded_fallback_rate", {})
    if isinstance(fb, Mapping):
        for clip in TURN_CLIPS:
            row = fb.get(clip, {})
            ok_rate = float(row.get("grounded_ok_rate", float("nan"))) if isinstance(row, Mapping) else float("nan")
            if np.isfinite(ok_rate):
                grounded[clip] = bool(ok_rate > 0.0)
    for clip in TURN_CLIPS:
        if clip not in grounded:
            grounded[clip] = bool(sampler.grounded_alignment(clip, onset=0).groundable)
    return grounded


def _legacy_action_only_gate_decision(
    per_clip: Mapping[str, Mapping[str, Any]],
    *,
    baseline_free: Mapping[str, Mapping[str, Any]],
    baseline_pinned: Mapping[str, Mapping[str, Any]],
    tau_yaw_rad: float,
    pose_degradation_tol: float,
) -> Dict[str, Any]:
    checks_by_clip: Dict[str, Dict[str, bool]] = {}
    pass_by_clip: Dict[str, bool] = {}
    for clip in TURN_CLIPS:
        row = per_clip[clip]
        yaw_corr = float(row.get("yaw_corr", float("nan")))
        heading_mae = float(row.get("heading_mae_rad", float("nan")))
        pop_safe = float(row.get("pop_safe_rate", float("nan")))
        best_pose = float(row.get("best_pose_d_mean", float("nan")))
        base_pose = min(
            float(baseline_free[clip].get("best_pose_d_mean", float("inf"))),
            float(baseline_pinned[clip].get("best_pose_d_mean", float("inf"))),
        )
        checks = {
            "yaw_corr_positive": bool(np.isfinite(yaw_corr) and yaw_corr > 0.0),
            "heading_mae_under_tau": bool(np.isfinite(heading_mae) and heading_mae < float(tau_yaw_rad)),
            "pop_safe_positive": bool(np.isfinite(pop_safe) and pop_safe > 0.0),
            "best_pose_not_degraded": bool(np.isfinite(best_pose) and best_pose <= base_pose + float(pose_degradation_tol)),
        }
        checks_by_clip[clip] = checks
        pass_by_clip[clip] = bool(all(checks.values()))
    return {
        "per_clip_checks": checks_by_clip,
        "per_clip_pass": pass_by_clip,
        "all_pass": bool(all(pass_by_clip.values())),
        "l_to_r_pass": bool(pass_by_clip.get("Walk_L_To_R", False)),
        "thresholds": {
            "tau_yaw_rad": float(tau_yaw_rad),
            "pose_degradation_tol": float(pose_degradation_tol),
        },
    }


def _f5_only_gate_decision(
    per_clip: Mapping[str, Mapping[str, Any]],
    *,
    baseline_free: Mapping[str, Mapping[str, Any]],
    baseline_pinned: Mapping[str, Mapping[str, Any]],
    pose_degradation_tol: float,
    require_reach_lift: bool = False,
    min_reach_lift: float = DEFAULT_SELF_REACH_RATE_LIFT,
) -> Dict[str, Any]:
    """F5 acceptance gate under commanded-yaw policy.

    Yaw is treated as control-positive-check only and is excluded from F5 pass/fail.
    """
    checks_by_clip: Dict[str, Dict[str, bool]] = {}
    pass_by_clip: Dict[str, bool] = {}
    for clip in TURN_CLIPS:
        row = per_clip[clip]
        pop_safe = float(row.get("pop_safe_rate", float("nan")))
        best_pose = float(row.get("best_pose_d_mean", float("nan")))
        reach_rate = float(row.get("self_reach_rate_k3", float("nan")))
        base_reach = float(baseline_free[clip].get("self_reach_gate", {}).get("rate_by_k", {}).get("k=3", float("nan")))
        base_pose = min(
            float(baseline_free[clip].get("best_pose_d_mean", float("inf"))),
            float(baseline_pinned[clip].get("best_pose_d_mean", float("inf"))),
        )
        checks = {
            "pop_safe_positive": bool(np.isfinite(pop_safe) and pop_safe > 0.0),
            "best_pose_not_degraded": bool(
                np.isfinite(best_pose) and best_pose <= base_pose + float(pose_degradation_tol)
            ),
        }
        if require_reach_lift:
            checks["reach_lift_ge_min"] = bool(
                np.isfinite(reach_rate)
                and np.isfinite(base_reach)
                and (reach_rate - base_reach) >= float(min_reach_lift)
            )
        checks_by_clip[clip] = checks
        pass_by_clip[clip] = bool(all(checks.values()))
    return {
        "policy": "F5_only_commanded_yaw",
        "per_clip_checks": checks_by_clip,
        "per_clip_pass": pass_by_clip,
        "all_pass": bool(all(pass_by_clip.values())),
        "l_to_r_pass": bool(pass_by_clip.get("Walk_L_To_R", False)),
        "thresholds": {
            "pose_degradation_tol": float(pose_degradation_tol),
            "require_reach_lift": bool(require_reach_lift),
            "min_reach_lift": float(min_reach_lift),
        },
    }


def main() -> None:
    args = _build_parser().parse_args()
    if not args.z_features.exists():
        raise FileNotFoundError(f"z-features not found: {args.z_features}")
    if not args.w1b_summary.exists():
        raise FileNotFoundError(f"W1b summary not found: {args.w1b_summary}")
    if not args.step0_coverage_summary.exists():
        raise FileNotFoundError(f"STEP0 coverage summary not found: {args.step0_coverage_summary}")
    if args.holdout_policy != "none" and not args.holdout_clip:
        raise ValueError("holdout_clip is required when holdout_policy is not 'none'")
    if args.holdout_policy == "none" and args.holdout_clip is not None:
        raise ValueError("holdout_clip must be omitted when holdout_policy is 'none'")
    if float(args.focus_sample_weight) < 1.0:
        raise ValueError("focus_sample_weight must be >= 1.0")
    if float(args.f5_seam_c1_weight) < 0.0:
        raise ValueError("f5_seam_c1_weight must be >= 0.0")

    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_inbetween_masked_smoke_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    states = load_clip_states(args.z_features, args.npz_root)
    sampler = InbetweenSampler(states, SamplerConfig())
    cfg = sampler.config
    normalizer = StateNormalizer(states)
    std = normalizer.std
    clip_lengths = {k: int(v.shape[0]) for k, v in states.items()}

    horizon = int(cfg.gap_min)
    if int(args.goal_horizon) < horizon:
        raise ValueError(f"goal_horizon must be >= horizon ({horizon}), got {args.goal_horizon}")
    model = MaskedMiddlePredictor(
        state_dim=int(states[WALK_F].shape[1]),
        context_len=int(cfg.context_len),
        seam_len=int(cfg.seam_len),
        horizon=horizon,
        hidden=int(args.hidden),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    model.train()
    losses: List[float] = []
    sample_filter_totals: Dict[str, int] = {
        "attempted": 0,
        "accepted": 0,
        "removed_grounded_cross_manifold": 0,
        "removed_full_holdout": 0,
    }
    for _ in range(int(args.steps)):
        ctx, middle, seam, metas, batch_audit = _draw_batch(
            sampler,
            int(args.batch),
            rng,
            holdout_clip=args.holdout_clip,
            holdout_policy=str(args.holdout_policy),
            max_sample_retries=int(args.max_sample_retries),
        )
        for k, v in batch_audit.items():
            sample_filter_totals[k] = int(sample_filter_totals.get(k, 0) + int(v))
        ctx_n = normalizer.normalize(ctx)
        middle_n = normalizer.normalize(middle)
        seam_n = normalizer.normalize(seam)
        pred_n = model(ctx_n, seam_n)
        sample_w = torch.as_tensor(
            [
                _sample_train_weight(
                    m,
                    clip_lengths,
                    focus_clip=args.focus_clip,
                    focus_phase_bucket=args.focus_phase_bucket,
                    focus_weight=float(args.focus_sample_weight),
                )
                for m in metas
            ],
            dtype=pred_n.dtype,
        )
        loss_mid = _pivot_channel_mse_weighted(
            pred_n,
            middle_n,
            pose_w=float(args.f5_loss_pose_weight),
            ego_w=float(args.f5_loss_ego_weight),
            contact_w=float(args.f5_loss_contact_weight),
            sample_weights=sample_w,
        )
        loss_c1 = _seam_c1_loss_weighted(
            pred_n,
            seam_n,
            ego_w=float(args.f5_seam_c1_ego_weight),
            contact_w=float(args.f5_seam_c1_contact_weight),
            sample_weights=sample_w,
        )
        loss = loss_mid + float(args.f5_seam_c1_weight) * loss_c1
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    train_stats = TrainStats(
        loss_first=float(losses[0]),
        loss_last=float(losses[-1]),
        loss_min=float(np.min(losses)),
        loss_decreased=bool(losses[-1] < losses[0]),
    )

    w1b = _load_json(args.w1b_summary)
    step0_cov = _load_json(args.step0_coverage_summary)
    groundable_by_clip = _groundability_from_summary(step0_cov, sampler)
    grounded_clips = [clip for clip in TURN_CLIPS if groundable_by_clip.get(clip, False)]
    ungrounded_clips = [clip for clip in TURN_CLIPS if not groundable_by_clip.get(clip, False)]
    phase2 = w1b["evaluated_objects"]["phase2_trained_goal"]["per_clip"]
    baseline_free = {clip: phase2[clip]["free_no_goal"] for clip in TURN_CLIPS}
    baseline_pinned = {clip: phase2[clip]["pinned_goal"] for clip in TURN_CLIPS}
    recorded_rows = w1b["evaluated_objects"]["phase2_trained_goal"]["recorded_turn_positive_control"]["per_clip"]

    thr = GateThresholds()
    hub = states[WALK_F]
    t_f = int(hub.shape[0])
    starts = [int(round(x)) % max(t_f, 1) for x in np.linspace(0, t_f - 1, int(args.n_starts))]

    model.eval()
    per_clip: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        for clip in TURN_CLIPS:
            target = states[clip]
            g0 = int(min(args.goal_horizon, target.shape[0] - SEAM_LEN_K))
            target_middle = target[g0 - horizon : g0]
            goal_seam_raw = target[g0 : g0 + SEAM_LEN_K]

            k = int(min(args.end_window_k, target.shape[0]))
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
                pred_n = model(ctx_n, goal_seam_n)[0]
                pred_raw = normalizer.denormalize(pred_n).cpu().numpy()
                pred_cmd = replace_yaw_rate_slice(
                    pred_raw,
                    target_middle[:, YAW_RATE_SLICE],
                )
                abs_cos_vals.append(float(np.min(cos_dist(pred_cmd, centroid).reshape(-1))))

                state = evaluate_rollout_state_space(pred_cmd, goal_seam_raw, std, thr)
                pop_safe_vals.append(1.0 if bool(state["pop_safe"]) else 0.0)
                pose_vals.append(float(state["best_pose_d"]))
                pop_vals.append(float(state["pop"]))
                ri = int(state["resume_rollout_frame"])
                tj = int(state["resume_target_frame"])
                ego_diff = (pred_cmd[ri, EGO_VEL_SLICE] - goal_seam_raw[tj, EGO_VEL_SLICE]) / std[EGO_VEL_SLICE]
                con_diff = (pred_cmd[ri, CONTACT_SLICE] - goal_seam_raw[tj, CONTACT_SLICE]) / std[CONTACT_SLICE]
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
            per_clip[clip] = {
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
                "baseline_self_reach_rate": float(baseline_free[clip]["self_reach_gate"]["rate_by_k"].get("k=3", 0.0)),
                "baseline_best_pose_d": float(baseline_free[clip]["best_pose_d_mean"]),
            }

    legacy_joint = joint_action_binding_gate_decision(
        per_clip,
        baseline_metrics=(baseline_free, baseline_pinned),
        k_label="k=3",
        min_reach_lift=DEFAULT_SELF_REACH_RATE_LIFT,
        tau_yaw_rad=DEFAULT_YAW_MAE_TAU_RAD,
        pose_degradation_tol=DEFAULT_POSE_DEGRADATION_TOL,
    )
    legacy_recorded_joint = joint_action_binding_gate_decision(
        recorded_rows,
        baseline_metrics=(),
        k_label="k=3",
        min_reach_lift=DEFAULT_SELF_REACH_RATE_LIFT,
        tau_yaw_rad=DEFAULT_YAW_MAE_TAU_RAD,
        pose_degradation_tol=DEFAULT_POSE_DEGRADATION_TOL,
    )

    legacy_action_only = _legacy_action_only_gate_decision(
        per_clip,
        baseline_free=baseline_free,
        baseline_pinned=baseline_pinned,
        tau_yaw_rad=DEFAULT_YAW_MAE_TAU_RAD,
        pose_degradation_tol=DEFAULT_POSE_DEGRADATION_TOL,
    )
    f5_only = _f5_only_gate_decision(
        per_clip,
        baseline_free=baseline_free,
        baseline_pinned=baseline_pinned,
        pose_degradation_tol=DEFAULT_POSE_DEGRADATION_TOL,
        require_reach_lift=False,
        min_reach_lift=DEFAULT_SELF_REACH_RATE_LIFT,
    )

    negative_clips = [
        clip for clip in TURN_CLIPS
        if float(baseline_free[clip].get("pop_safe_rate", 0.0)) == 0.0
    ]
    negative_ungrounded = [clip for clip in negative_clips if clip in ungrounded_clips]
    lifted_clips = [
        clip for clip in negative_clips
        if bool(f5_only["per_clip_pass"].get(clip, False))
    ]
    lifted_ungrounded = [clip for clip in lifted_clips if clip in ungrounded_clips]
    lifted_grounded = [clip for clip in lifted_clips if clip in grounded_clips]
    f5_acceptance = {
        "recorded_identity_pass": bool(legacy_recorded_joint.all_pass),
        "grounded_clips": grounded_clips,
        "ungrounded_clips": ungrounded_clips,
        "ar_free_negative_clips_pop_only": negative_clips,
        "ar_free_negative_clips_ungrounded": negative_ungrounded,
        "lifted_clips_f5_only": lifted_clips,
        "lifted_grounded_clips": lifted_grounded,
        "lifted_ungrounded_clips": lifted_ungrounded,
        "lift_exists_any": bool(len(lifted_clips) > 0),
        "lift_exists_ungrounded": bool(len(lifted_ungrounded) > 0),
        "memorization_suspected": bool(len(lifted_grounded) > 0 and len(lifted_ungrounded) == 0),
        "step2_pass_any_lift_nonbinding": bool(legacy_recorded_joint.all_pass and len(lifted_clips) > 0),
        "step2_pass_generalization_binding": bool(legacy_recorded_joint.all_pass and len(lifted_ungrounded) > 0),
        "f5_gate_all_pass": bool(f5_only["all_pass"]),
        "f5_gate_l_to_r_pass": bool(f5_only["l_to_r_pass"]),
    }

    state_path = out_dir / "masked_smoke_state.pt"
    torch.save(
        {
            "kind": "action_handoff_inbetween_masked_smoke_state",
            "seed": int(args.seed),
            "args": vars(args),
            "context_len": int(cfg.context_len),
            "seam_len": int(cfg.seam_len),
            "horizon": int(horizon),
            "model_state_dict": model.state_dict(),
            "normalizer_mean": normalizer.mean.tolist(),
            "normalizer_std": normalizer.std.tolist(),
        },
        state_path,
    )

    summary = {
        "task": "W1c STEP-2(B) masked in-betweening minimal smoke",
        "binding_metric": "F5-only criterion fields (pop+pose; optional reach)",
        "yaw_control_mode": "commanded_from_target_cond_dir_window (not learned)",
        "f5_learning_target": "pivot/continuity over pose[276]+ego_vel[2]+contact[2]",
        "no_new_data": True,
        "state_saved": str(state_path.resolve()),
        "train_stats": asdict(train_stats),
        "thresholds": {
            "min_reach_lift": DEFAULT_SELF_REACH_RATE_LIFT,
            "tau_yaw_rad": DEFAULT_YAW_MAE_TAU_RAD,
            "pose_degradation_tol": DEFAULT_POSE_DEGRADATION_TOL,
        },
        "f5_train_weighting": {
            "pivot_loss_weights": {
                "pose": float(args.f5_loss_pose_weight),
                "ego_vel": float(args.f5_loss_ego_weight),
                "contact": float(args.f5_loss_contact_weight),
            },
            "seam_c1_weight": float(args.f5_seam_c1_weight),
            "seam_c1_channel_weights": {
                "ego_vel": float(args.f5_seam_c1_ego_weight),
                "contact": float(args.f5_seam_c1_contact_weight),
            },
            "focus_clip": args.focus_clip,
            "focus_phase_bucket_q4": args.focus_phase_bucket,
            "focus_sample_weight": float(args.focus_sample_weight),
        },
        "training_holdout_policy": {
            "policy": str(args.holdout_policy),
            "holdout_clip": str(args.holdout_clip) if args.holdout_clip else None,
            "max_sample_retries": int(args.max_sample_retries),
            "sample_filter_totals": {k: int(v) for k, v in sample_filter_totals.items()},
        },
        "per_clip": per_clip,
        "f5_only_gate_decision": f5_only,
        "legacy_w1b_alignment": {
            "joint_gate_decision": asdict(legacy_joint),
            "action_only_gate_decision": legacy_action_only,
            "note": "legacy yaw-containing gate retained for historical comparability only; not F5 acceptance",
        },
        "space_compatibility": {
            "candidate_space": "281d_state",
            "w1b_reach_space": "hidden_pre_512",
            "reach_component_comparable": False,
            "reach_proxy_space": "281d_all_channels",
            "note": "masked candidate has no hidden_pre; reach_proxy remains a non-binding 281-d summary, not an F5 gate.",
        },
        "recorded_turn_positive_control": {
            "per_clip": recorded_rows,
            "joint_gate_decision": asdict(legacy_recorded_joint),
        },
        "acceptance": f5_acceptance,
        "baseline_ref": str(args.w1b_summary.resolve()),
    }
    if args.holdout_clip:
        held_out = str(args.holdout_clip)
        summary["held_out_clip"] = {
            "clip": held_out,
            "metrics": per_clip[held_out],
            "f5_only_pass": bool(f5_only["per_clip_pass"].get(held_out, False)),
            "legacy_action_only_pass": bool(legacy_action_only["per_clip_pass"].get(held_out, False)),
            "legacy_joint_pass": bool(legacy_joint.per_clip_pass.get(held_out, False)),
        }
    json_path = out_dir / "masked_smoke_summary.json"
    md_path = out_dir / "masked_smoke_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# W1c STEP-2(B) Masked In-Betweening Smoke")
    lines.append("")
    lines.append(f"- train loss: {_fmt(train_stats.loss_first)} -> {_fmt(train_stats.loss_last)} (min={_fmt(train_stats.loss_min)})")
    lines.append(
        "- training holdout policy: "
        f"{args.holdout_policy} "
        f"(holdout_clip={args.holdout_clip}, "
        f"removed_grounded={sample_filter_totals.get('removed_grounded_cross_manifold', 0)}, "
        f"removed_full_holdout={sample_filter_totals.get('removed_full_holdout', 0)})"
    )
    lines.append(f"- recorded identity pass: {f5_acceptance['recorded_identity_pass']}")
    lines.append(f"- grounded clips: {f5_acceptance['grounded_clips']}")
    lines.append(f"- ungrounded clips: {f5_acceptance['ungrounded_clips']}")
    lines.append(f"- AR-free negative clips (pop_only): {f5_acceptance['ar_free_negative_clips_pop_only']}")
    lines.append(f"- lifted clips (F5-only gate): {f5_acceptance['lifted_clips_f5_only']}")
    lines.append(f"- lifted ungrounded clips: {f5_acceptance['lifted_ungrounded_clips']}")
    lines.append(f"- memorization suspected: {f5_acceptance['memorization_suspected']}")
    lines.append(f"- F5 gate all-pass: {f5_acceptance['f5_gate_all_pass']}")
    lines.append(f"- F5 gate L_to_R pass: {f5_acceptance['f5_gate_l_to_r_pass']}")
    lines.append(f"- STEP2 pass (any-lift, nonbinding): {f5_acceptance['step2_pass_any_lift_nonbinding']}")
    lines.append(f"- STEP2 pass (generalization, binding): {f5_acceptance['step2_pass_generalization_binding']}")
    lines.append("- note: yaw metrics are retained as commanded positive-control diagnostics; not used for F5 acceptance.")
    lines.append("")
    lines.append("| clip | self_k3 | yaw_corr | heading_MAE_deg | pop_safe | pop_mean | ego_pop | contact_pop | best_pose_d | F5_pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for clip in TURN_CLIPS:
        row = per_clip[clip]
        lines.append(
            f"| {clip} | {_fmt(row['self_reach_rate_k3'],2)} | {_fmt(row['yaw_corr'],2)} | "
            f"{_fmt(np.degrees(row['heading_mae_rad']),2)} | {_fmt(row['pop_safe_rate'],2)} | "
            f"{_fmt(row['pop_mean'],3)} | {_fmt(row['ego_vel_pop_mean'],3)} | {_fmt(row['contact_pop_mean'],3)} | "
            f"{_fmt(row['best_pose_d_mean'],4)} | {f5_only['per_clip_pass'].get(clip)} |"
        )
    if args.holdout_clip:
        held_row = per_clip[str(args.holdout_clip)]
        held_pass = bool(f5_only["per_clip_pass"].get(str(args.holdout_clip), False))
        lines.append("")
        lines.append(
            f"- held-out clip ({args.holdout_policy}): {args.holdout_clip} | "
            f"yaw_corr={_fmt(held_row['yaw_corr'],4)}, "
            f"heading_MAE_rad={_fmt(held_row['heading_mae_rad'],4)}, "
            f"pop_safe={_fmt(held_row['pop_safe_rate'],4)}, "
            f"pop={_fmt(held_row['pop_mean'],4)}, "
            f"ego_pop={_fmt(held_row['ego_vel_pop_mean'],4)}, "
            f"contact_pop={_fmt(held_row['contact_pop_mean'],4)}, "
            f"best_pose_d_mean={_fmt(held_row['best_pose_d_mean'],4)}, "
            f"f5_only_pass={held_pass}"
        )
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[ok] wrote: {state_path}")
    print(
        "[W1c-B] "
        f"identity_pass={f5_acceptance['recorded_identity_pass']} "
        f"lifted_f5={f5_acceptance['lifted_clips_f5_only']} "
        f"lifted_ungrounded={f5_acceptance['lifted_ungrounded_clips']} "
        f"step2_pass_generalization={f5_acceptance['step2_pass_generalization_binding']}"
    )


if __name__ == "__main__":
    main()
