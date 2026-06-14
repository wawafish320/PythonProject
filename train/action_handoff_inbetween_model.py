from __future__ import annotations

"""Action-handoff goal-conditioned in-betweening — minimal B1 probe model (§7.3, 3a).

Stage 3a "wiring smoke" per the plan
(`docs/aperiodic_transition/2026-05-30_action_handoff_inbetween_73_b1_probe_plan.md`):
a deliberately tiny, **from-scratch** (no checkpoint) AR-with-goal-conditioning model plus
the minimal three losses and a free-rollout + state-space B1 metric harness.

This module is the §7.3 owner surface; it does NOT modify the frozen §7.2 data pipeline.
It consumes the §7.2 sampler tensors (ctx[C,281] / gt_middle[H,281] / seam_target[K,281]).

Scope (3a only):
  - NO checkpoint / base-model dependency (that is 3b).
  - NO z-reach (z rides on the base model → 3b); 3a reports STATE-SPACE metrics only.
  - reach numbers here are NON-BINDING and cannot trigger the spec §6 STOP.

Default model form is AR-with-goal-conditioning (spec §3 default). Everything operates in
pooled-std group-normalized space so the 276-d pose term does not drown ego_vel / yaw_rate
/ contact (spec §1.1 posture). All thresholds are PROVISIONAL.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from train.data.action_handoff_inbetween import (
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    POSE_DIM,
    POSE_SLICE,
    STATE_DIM,
    YAW_RATE_SLICE,
    contact_distance,
    pose_distance,
)

EPS = 1e-8

# F5 policy: yaw_rate is commanded from cond_dir (control input), not learned.
# In-between learns only pivot/continuity channels: pose + ego_vel + contact.
LEARN_SLICES: Tuple[slice, ...] = (POSE_SLICE, EGO_VEL_SLICE, CONTACT_SLICE)

# Channel groups for weighted losses / group-norm over learnable channels only.
# Default weights are equal in std units — PROVISIONAL, tune after fair (3b) probe.
GROUP_SLICES: Dict[str, slice] = {
    "pose": POSE_SLICE,
    "ego_vel": EGO_VEL_SLICE,
    "contact": CONTACT_SLICE,
}
DEFAULT_GROUP_WEIGHTS: Dict[str, float] = {
    "pose": 1.0,
    "ego_vel": 1.0,
    "contact": 1.0,
}
# Channels whose continuity the seam C1 term watches under commanded yaw policy.
SEAM_C1_SLICES: Tuple[slice, ...] = (EGO_VEL_SLICE, CONTACT_SLICE)


# =============================================================== pooled group normalizer
class StateNormalizer:
    """Per-channel pooled mean/std over all clip frames (group-norm posture, spec §1.1)."""

    def __init__(self, clips: Dict[str, np.ndarray]) -> None:
        allf = np.concatenate([np.asarray(v, dtype=np.float64) for v in clips.values()], axis=0)
        self.mean = allf.mean(axis=0).astype(np.float32)
        std = allf.std(axis=0)
        self.std = np.where(std > 1e-6, std, 1.0).astype(np.float32)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        s = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        return (x - m) / s

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        s = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        return x * s + m


# ===================================================================== minimal AR+goal model
@dataclass
class ModelConfig:
    state_dim: int = STATE_DIM
    seam_len: int = 6
    hidden: int = 128


class MinimalGoalAR(nn.Module):
    """Tiny residual AR next-frame predictor conditioned on context + goal seam window.

    All I/O is in normalized space. `s_{t+1} = s_t + Δ(s_t, ctx_emb, goal_emb)`.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.cfg = cfg
        h = cfg.hidden
        self.ctx_gru = nn.GRU(cfg.state_dim, h, batch_first=True)
        self.goal_mlp = nn.Sequential(
            nn.Linear(cfg.seam_len * cfg.state_dim, h), nn.ReLU(), nn.Linear(h, h)
        )
        self.step_mlp = nn.Sequential(
            nn.Linear(cfg.state_dim + 2 * h, h), nn.ReLU(), nn.Linear(h, cfg.state_dim)
        )

    def encode(self, ctx: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, hN = self.ctx_gru(ctx)  # ctx [B,C,D]
        ctx_emb = hN[-1]  # [B,h]
        goal_emb = self.goal_mlp(goal.reshape(goal.shape[0], -1))  # goal [B,K,D]
        return ctx_emb, goal_emb

    def step(self, s_t: torch.Tensor, ctx_emb: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([s_t, ctx_emb, goal_emb], dim=-1)
        return s_t + self.step_mlp(inp)

    def rollout_teacher_forced(
        self, ctx: torch.Tensor, goal: torch.Tensor, gt_middle: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced middle [B,H,D] + free seam continuation [B,K,D] (normalized)."""
        ctx_emb, goal_emb = self.encode(ctx, goal)
        prev = ctx[:, -1]
        mids: List[torch.Tensor] = []
        for t in range(gt_middle.shape[1]):
            mids.append(self.step(prev, ctx_emb, goal_emb))
            prev = gt_middle[:, t]  # teacher forcing
        mid_pred = torch.stack(mids, dim=1)
        # seam: continue free from the last (true) middle frame, K steps.
        prev = gt_middle[:, -1]
        seams: List[torch.Tensor] = []
        for _ in range(self.cfg.seam_len):
            cur = self.step(prev, ctx_emb, goal_emb)
            seams.append(cur)
            prev = cur
        seam_pred = torch.stack(seams, dim=1)
        return mid_pred, seam_pred

    def rollout_free(self, ctx: torch.Tensor, goal: torch.Tensor, horizon: int) -> torch.Tensor:
        """Free AR rollout (NO teacher forcing) for `horizon` steps [B,horizon,D]."""
        ctx_emb, goal_emb = self.encode(ctx, goal)
        prev = ctx[:, -1]
        out: List[torch.Tensor] = []
        for _ in range(int(horizon)):
            cur = self.step(prev, ctx_emb, goal_emb)
            out.append(cur)
            prev = cur
        return torch.stack(out, dim=1)

    def rollout_free_commanded_yaw(
        self,
        ctx: torch.Tensor,
        goal: torch.Tensor,
        cmd_yaw: torch.Tensor,
    ) -> torch.Tensor:
        """Free AR rollout with per-step commanded yaw overwrite.

        Args:
            ctx: [B,C,D] normalized state context.
            goal: [B,K,D] normalized seam goal context.
            cmd_yaw: [B,H,1] normalized commanded yaw-rate trajectory.

        Returns:
            [B,H,D] normalized rollout where each step's yaw slice is overwritten
            by cmd_yaw before feedback to the next AR step.
        """
        _validate_rollout_free_commanded_yaw_inputs(ctx=ctx, goal=goal, cmd_yaw=cmd_yaw)
        ctx_emb, goal_emb = self.encode(ctx, goal)
        prev = ctx[:, -1]
        out: List[torch.Tensor] = []
        horizon = int(cmd_yaw.shape[1])
        for t in range(horizon):
            cur = self.step(prev, ctx_emb, goal_emb)
            cur = torch.cat(
                [
                    cur[..., : YAW_RATE_SLICE.start],
                    cmd_yaw[:, t : t + 1, :].reshape(cur.shape[0], 1),
                    cur[..., YAW_RATE_SLICE.stop :],
                ],
                dim=-1,
            )
            out.append(cur)
            prev = cur
        return torch.stack(out, dim=1)


def _validate_rollout_free_commanded_yaw_inputs(
    *,
    ctx: torch.Tensor,
    goal: torch.Tensor,
    cmd_yaw: torch.Tensor,
) -> None:
    if ctx.ndim != 3:
        raise ValueError(f"ctx must be rank-3 [B,C,D], got shape={tuple(ctx.shape)}")
    if goal.ndim != 3:
        raise ValueError(f"goal must be rank-3 [B,K,D], got shape={tuple(goal.shape)}")
    if cmd_yaw.ndim != 3:
        raise ValueError(f"cmd_yaw must be rank-3 [B,H,1], got shape={tuple(cmd_yaw.shape)}")

    b = int(ctx.shape[0])
    d = int(ctx.shape[2])
    if int(goal.shape[0]) != b:
        raise ValueError(f"ctx/goal batch mismatch: B_ctx={b}, B_goal={int(goal.shape[0])}")
    if int(cmd_yaw.shape[0]) != b:
        raise ValueError(f"ctx/cmd_yaw batch mismatch: B_ctx={b}, B_cmd={int(cmd_yaw.shape[0])}")
    if int(goal.shape[2]) != d:
        raise ValueError(f"ctx/goal state dim mismatch: D_ctx={d}, D_goal={int(goal.shape[2])}")
    if d != int(STATE_DIM):
        raise ValueError(f"state dim mismatch: expected D={int(STATE_DIM)}, got D={d}")
    if int(cmd_yaw.shape[2]) != 1:
        raise ValueError(f"cmd_yaw last dim must be 1, got {int(cmd_yaw.shape[2])}")
    if cmd_yaw.shape[1] < 1:
        raise ValueError("cmd_yaw horizon must be >= 1")
    if ctx.device != goal.device or ctx.device != cmd_yaw.device:
        raise ValueError(
            f"device mismatch: ctx={ctx.device}, goal={goal.device}, cmd_yaw={cmd_yaw.device}"
        )


def rollout_free_commanded_yaw(
    model: MinimalGoalAR,
    ctx: torch.Tensor,
    goal: torch.Tensor,
    cmd_yaw: torch.Tensor,
) -> torch.Tensor:
    """Functional helper wrapping MinimalGoalAR.rollout_free_commanded_yaw."""
    return model.rollout_free_commanded_yaw(ctx=ctx, goal=goal, cmd_yaw=cmd_yaw)


# ============================================================================ minimal losses
def _grouped_weighted_mse(
    pred: torch.Tensor, target: torch.Tensor, weights: Dict[str, float]
) -> torch.Tensor:
    """Mean over groups of (per-group mean-MSE × weight); pred/target [...,D] normalized."""
    total = pred.new_zeros(())
    wsum = 0.0
    for name, sl in GROUP_SLICES.items():
        w = float(weights.get(name, 1.0))
        total = total + w * torch.mean((pred[..., sl] - target[..., sl]) ** 2)
        wsum += w
    return total / max(wsum, EPS)


def _learnable_channel_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean MSE over learnable pivot channels only (pose/ego_vel/contact)."""
    total = pred.new_zeros(())
    for sl in LEARN_SLICES:
        total = total + torch.mean((pred[..., sl] - target[..., sl]) ** 2)
    return total / len(LEARN_SLICES)


def loss_middle(mid_pred: torch.Tensor, gt_middle_norm: torch.Tensor) -> torch.Tensor:
    """AR reconstruction over learnable channels only (yaw is commanded)."""
    return _learnable_channel_mse(mid_pred, gt_middle_norm)


def loss_reach(
    seam_pred: torch.Tensor, seam_target_norm: torch.Tensor, weights: Dict[str, float]
) -> torch.Tensor:
    """Weighted L over the horizon-end K-window vs the goal seam (spec §4 L_reach, w=1.0)."""
    return _grouped_weighted_mse(seam_pred, seam_target_norm, weights)


def loss_seam_c1(mid_pred: torch.Tensor, seam_target_norm: torch.Tensor) -> torch.Tensor:
    """Light C1: discontinuity between the last generated frame and the seam's first frame,
    on ego_vel / contact under commanded-yaw policy. Normalized space."""
    last = mid_pred[:, -1]
    first = seam_target_norm[:, 0]
    total = last.new_zeros(())
    for sl in SEAM_C1_SLICES:
        total = total + torch.mean((last[..., sl] - first[..., sl]) ** 2)
    return total / len(SEAM_C1_SLICES)


@dataclass
class LossWeights:
    middle: float = 1.0
    reach: float = 1.0
    seam_c1: float = 0.5
    group: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GROUP_WEIGHTS))


def compute_losses(
    model: MinimalGoalAR,
    ctx: torch.Tensor,
    goal: torch.Tensor,
    gt_middle: torch.Tensor,
    seam_target: torch.Tensor,
    weights: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """All inputs normalized [B,*,D]. Returns total loss + scalar component dict."""
    mid_pred, seam_pred = model.rollout_teacher_forced(ctx, goal, gt_middle)
    l_mid = loss_middle(mid_pred, gt_middle)
    l_reach = loss_reach(seam_pred, seam_target, weights.group)
    l_c1 = loss_seam_c1(mid_pred, seam_target)
    total = weights.middle * l_mid + weights.reach * l_reach + weights.seam_c1 * l_c1
    return total, {
        "total": float(total.detach()),
        "L_middle": float(l_mid.detach()),
        "L_reach": float(l_reach.detach()),
        "L_seam_C1": float(l_c1.detach()),
    }


# ============================================================== state-space B1 metrics (3a)
@dataclass
class GateThresholds:
    tau_pose: float = 0.15  # [PROVISIONAL] clip-resumable pose_d gate
    tau_pop: float = 0.30  # [PROVISIONAL] seam pop (ego_vel/contact) gate
    reach_proxy_thr: float = 1.0  # [PROVISIONAL] state-space reach proxy (group-std units)


def _group_std_distance(a: np.ndarray, b_centroid: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-frame mean |a-centroid|/std over all channels (group-std units)."""
    d = np.abs((a - b_centroid[None, :]) / std[None, :])
    return d.mean(axis=1)


def evaluate_rollout_state_space(
    roll_raw: np.ndarray,
    goal_seam_raw: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
) -> Dict[str, object]:
    """One (start, target) free-rollout outcome in STATE SPACE (3a; NOT z-reach).

    "clip-resumable" means converging to the target's RESUMABLE seam region, NOT its
    onset — turn clips begin on the walk manifold (authored loops), so matching against
    the whole clip's onset is trivially satisfied. We therefore match against the goal
    seam window only.

    - clip_resumable: min pose_d(rollout frame, goal-seam frame) ≤ τ_pose.
    - pop_safe: at the resumable frame, ego_vel/contact discontinuity vs the
      matched seam frame ≤ τ_pop (group-std units).
    - reach_proxy: min over rollout of group-std distance to the goal seam centroid
      (explicitly a STATE-SPACE proxy, NOT the spec z-reach).
    """
    # pose_d matrix rollout × goal-seam (resumable region only).
    pose_roll = roll_raw[:, POSE_SLICE]
    pose_tgt = goal_seam_raw[:, POSE_SLICE]
    scale = np.sqrt(POSE_DIM)
    dmat = np.linalg.norm(pose_roll[:, None, :] - pose_tgt[None, :, :], axis=2) / scale
    ri, tj = np.unravel_index(int(np.argmin(dmat)), dmat.shape)
    best_pose_d = float(dmat[ri, tj])
    clip_resumable = bool(best_pose_d <= thr.tau_pose)

    # pop at the matched frame: ego_vel/contact diff in group-std units.
    pop = 0.0
    for sl in SEAM_C1_SLICES:
        diff = (roll_raw[ri, sl] - goal_seam_raw[tj, sl]) / std[sl]
        pop += float(np.mean(np.abs(diff)))
    pop /= len(SEAM_C1_SLICES)
    pop_safe = bool(pop <= thr.tau_pop)

    goal_centroid = goal_seam_raw.mean(axis=0)
    reach_proxy = float(np.min(_group_std_distance(roll_raw, goal_centroid, std)))
    reached_proxy = bool(reach_proxy <= thr.reach_proxy_thr)

    return {
        "best_pose_d": best_pose_d,
        "clip_resumable": clip_resumable,
        "pop": pop,
        "pop_safe": pop_safe,
        "reach_proxy": reach_proxy,
        "reached_proxy": reached_proxy,
        "fallback": bool(not clip_resumable),
        "resume_rollout_frame": int(ri),
        "resume_target_frame": int(tj),
    }
