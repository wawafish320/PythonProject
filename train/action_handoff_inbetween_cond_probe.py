from __future__ import annotations

"""Action-handoff in-betweening — §7.3 3b Slice 2 cond-driven baseline probe (pure parts).

**NON-BINDING floor diagnostic.** This module owns the model-free, unit-testable pieces of
the cond-driven baseline probe (per
`docs/aperiodic_transition/2026-05-30_action_handoff_inbetween_73b_path_ab_plan.md` §2.2):
arbitrary-Walk_F-phase seeding (wrap), the cond override construction, and per-clip
aggregation of the reach (hidden_pre) + state-space outcomes. The base-model rollout,
hidden_pre capture, and reach metric live in
`tools/run_action_handoff_inbetween_b1_cond_baseline_probe.py` /
`train/action_handoff_inbetween_reach.py`.

Two findings from the locked data drive the cond-override design (both surfaced in the
probe output):

1. ``act_oh`` is identical (``[0,1,0,0]``) across ALL five locked clips — overriding the
   action one-hot is a no-op in this dataset; the only turn-distinguishing cond channel is
   ``cond_dir`` (world heading).
2. The base model normalizes ``cond_in`` with PER-WINDOW robust mean/std
   (``MotionEventDataset.__getitem__``). A *constant* cond_dir override therefore collapses
   to ~0 after normalization (indistinguishable from Walk_F) — the turn signal lives in the
   ``cond_dir`` TRAJECTORY (heading ramp). So "condition on the target turn" is realized by
   injecting the turn's recorded cond TRAJECTORY and normalizing it with that turn's own
   per-window robust stats (exactly what the model saw when running that turn).

All thresholds PROVISIONAL.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from train.data.action_handoff_inbetween import (
    CONTACT_DIM,
    POSE_DIM,
    STATE_DIM,
    TURN_CLIPS,
    WALK_F,
    build_egocentric_state,
)

COND_DIM = 7  # act_oh(4) + cond_dir(2) + cond_speed(1)
COND_NORM_CLIP = 6.0  # [PROVISIONAL] matches dataset cond clip (MotionEventDataset)
WALK_L_TO_R = "Walk_L_To_R"  # zero grounded supervision (§2b) → always its own row


def _robust_mean_std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel robust (IQR-trimmed) mean/std — mirrors ``MotionEventDataset._robust_mean_std``.

    Replicated (not imported) to keep this module model-free and unit-testable; kept byte-for-byte
    equivalent so the injected turn cond is normalized exactly as the dataset would normalize it.
    """
    a = np.asarray(arr, dtype=np.float64)
    q1 = np.percentile(a, 25, axis=0)
    q3 = np.percentile(a, 75, axis=0)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    mask = (a >= lo) & (a <= hi)
    safe = np.where(mask, a, np.nan)
    mu = np.nanmean(safe, axis=0)
    std = np.nanstd(safe, axis=0)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    std = np.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6).astype(np.float32, copy=False)
    std = np.clip(std, 1e-6, None)
    return mu.reshape(1, -1), std.reshape(1, -1)


def phase_seed_indices(phase: int, horizon: int, clip_len: int) -> np.ndarray:
    """Wrapped frame indices for an arbitrary-phase seed on the periodic Walk_F clip.

    Walk_F is a single periodic locomotion cycle, so any phase start with wrap-around is a
    valid context (plan + spec §2a). Returns int indices ``(phase + [0..horizon)) % clip_len``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if clip_len < 1:
        raise ValueError(f"clip_len must be >= 1, got {clip_len}")
    return (int(phase) + np.arange(int(horizon), dtype=np.int64)) % int(clip_len)


def select_start_phases(clip_len: int, n_starts: int) -> List[int]:
    """N≥20 arbitrary Walk_F start phases, evenly spread over the cycle (matches 3a probe)."""
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    return [int(round(x)) % int(clip_len) for x in np.linspace(0, clip_len - 1, int(n_starts))]


@dataclass
class CondOverride:
    """The target turn's conditioning, extended to the rollout horizon (raw + normalized).

    ``raw`` [H,7] is the turn's cond trajectory held at its last frame past ``turn_len``.
    ``mu``/``std`` [7] are the turn trajectory's per-window robust stats (NOT the extended
    window's) — i.e. exactly the normalization the dataset applies when running that turn.
    ``norm`` [H,7] = clip((raw - mu)/std, ±clip).
    """

    raw: np.ndarray
    norm: np.ndarray
    mu: np.ndarray
    std: np.ndarray
    turn_len: int


def build_cond_override(
    turn_cond_raw: np.ndarray, horizon: int, *, clip: float = COND_NORM_CLIP
) -> CondOverride:
    """Build the cond override for a target turn (finding #2: inject the trajectory)."""
    raw = np.asarray(turn_cond_raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != COND_DIM:
        raise ValueError(f"turn_cond_raw must be [T,{COND_DIM}], got {raw.shape}")
    turn_len = int(raw.shape[0])
    if turn_len < 1:
        raise ValueError("turn_cond_raw is empty")
    H = int(horizon)
    if H < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    # extend by holding the last turn frame (the model's exogenous cond past the turn end)
    if H <= turn_len:
        raw_ext = raw[:H].copy()
    else:
        tail = np.repeat(raw[-1:], H - turn_len, axis=0)
        raw_ext = np.concatenate([raw, tail], axis=0)
    # per-window stats over the TURN trajectory only (canonical for that turn)
    mu, std = _robust_mean_std(raw)
    norm = (raw_ext - mu) / std
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip(norm, -float(clip), float(clip)).astype(np.float32)
    return CondOverride(
        raw=raw_ext.astype(np.float32),
        norm=norm,
        mu=mu.reshape(-1).astype(np.float32),
        std=std.reshape(-1).astype(np.float32),
        turn_len=turn_len,
    )


def rollout_to_egocentric(
    rot6d: np.ndarray,
    root_vel: np.ndarray,
    cond_dir: np.ndarray,
    contact: np.ndarray,
    *,
    fps: float = 60.0,
) -> np.ndarray:
    """Generated rollout → egocentric state [T,281] (spec §1.1), reusing ``build_egocentric_state``.

    ``rot6d``/``root_vel`` are the GENERATED (denormalized) per-step state. ``cond_dir`` is the
    injected (commanded) turn heading — the pose representation is heading-invariant (root
    rot6d yaw is ~constant across clips), so heading for the ego_vel/yaw_rate channels must come
    from cond, matching how the recorded-clip egocentric state is built. ``contact`` is the
    exogenous (seed) contact stream actually fed to the model. All signals truncated to a common
    length.
    """
    r6 = np.asarray(rot6d, dtype=np.float64)
    rv = np.asarray(root_vel, dtype=np.float64)
    cd = np.asarray(cond_dir, dtype=np.float64)
    cn = np.asarray(contact, dtype=np.float64)
    t = int(min(r6.shape[0], rv.shape[0], cd.shape[0], cn.shape[0]))
    if t < 1:
        raise ValueError("empty rollout for egocentric conversion")
    bone = r6[:t].reshape(t, -1)
    if bone.shape[1] != POSE_DIM:
        raise ValueError(f"rot6d flat dim {bone.shape[1]} != {POSE_DIM}")
    return build_egocentric_state(bone.reshape(t, POSE_DIM // 6, 6), rv[:t], cd[:t], cn[:t], fps=fps)


def summarize_reach(min_norms: Sequence[float], conv_norm_thr: float) -> Dict[str, float]:
    """Aggregate per-start reach_min_norm (continuous) + floor rate (min_norm ≤ conv_norm_thr)."""
    arr = np.asarray([m for m in min_norms if np.isfinite(m)], dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "reach_min_norm_mean": float("nan"),
            "reach_min_norm_median": float("nan"),
            "reach_min_norm_p90": float("nan"),
            "reach_min_norm_min": float("nan"),
            "reach_floor_rate": float("nan"),
        }
    return {
        "n": int(arr.size),
        "reach_min_norm_mean": float(np.mean(arr)),
        "reach_min_norm_median": float(np.median(arr)),
        "reach_min_norm_p90": float(np.percentile(arr, 90)),
        "reach_min_norm_min": float(np.min(arr)),
        "reach_floor_rate": float(np.mean(arr <= float(conv_norm_thr))),
    }


def aggregate_clip_record(
    reach_min_norms: Sequence[float],
    state_outcomes: Sequence[Dict[str, object]],
    conv_norm_thr: float,
) -> Dict[str, object]:
    """Per-clip record: reach (hidden_pre) summary + the four state-space rates.

    Five headline metrics (per the task): reach_min_norm (dist) + reach_floor_rate,
    clip_resumable_rate, pop_safe_rate, fallback_rate.
    """
    reach = summarize_reach(reach_min_norms, conv_norm_thr)
    n = len(state_outcomes)
    rec: Dict[str, object] = {
        "n_starts": int(max(n, reach["n"])),
        **reach,
    }
    if n > 0:
        rec["clip_resumable_rate"] = float(np.mean([bool(o["clip_resumable"]) for o in state_outcomes]))
        rec["pop_safe_rate"] = float(np.mean([bool(o["pop_safe"]) for o in state_outcomes]))
        rec["fallback_rate"] = float(np.mean([bool(o["fallback"]) for o in state_outcomes]))
        rec["mean_best_pose_d"] = float(np.mean([float(o["best_pose_d"]) for o in state_outcomes]))
        rec["mean_pop"] = float(np.mean([float(o["pop"]) for o in state_outcomes]))
    else:
        rec["clip_resumable_rate"] = float("nan")
        rec["pop_safe_rate"] = float("nan")
        rec["fallback_rate"] = float("nan")
        rec["mean_best_pose_d"] = float("nan")
        rec["mean_pop"] = float("nan")
    return rec


def turn_clip_order(turn_clips: Sequence[str] = TURN_CLIPS) -> List[str]:
    """Turn clips with Walk_L_To_R guaranteed present as its own row (never averaged away)."""
    order = list(turn_clips)
    if WALK_L_TO_R not in order:
        raise ValueError(f"{WALK_L_TO_R} must be among turn clips (zero-grounded row): {order}")
    return order
