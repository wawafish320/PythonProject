#!/usr/bin/env python3
"""Soft-endpoint reframe — zero-training falsifiability probe.

Asks one question with existing artifacts only, NO training / NO base unfreeze / NO latent
injection: does the `action-handoff in-betweening` reframe — switch-signal decoupled from a
SOFT, emergent endpoint (re-anchor to an arbitrary in-regime landing) plus a phase-difference
BRIDGE — change the W1d PARK conclusion?

Two complementary angles (both zero-training):

  A. RE-ANCHOR vs PRECISE FRAME (recorded data, mechanism side-evidence). On recorded clips,
     splice an arbitrary-phase Walk_F frame to (i) the fixed seam frame vs (ii) the best
     in-regime re-anchor. Shows soft re-anchor lowers pop, AND the residual pop is non-zero
     (arbitrary phase has no perfectly-aligned frame) → a generated bridge is still required
     (MM-cut/interpolation insufficient).

  B. RE-SCORE THE PARKED GENERATED BRIDGES (decisive). Reload the W1d LOGO parked masked
     bridges and re-judge them under the SOFT caliper (arbitrary in-regime re-anchor) vs the
     PRECISE caliper (fixed seam frame). Do the held-out clips that FAIL precise get revived
     under soft WITH motion consistency intact (realized-yaw correct, pop genuinely safe,
     region read from motion, nothing injected)?

A pre-committed decision rule (written before the numbers) maps the result to UNPARK vs
KEEP-PARK. Red lines enforced in `train/action_handoff_inbetween_soft_endpoint.py`: soft is
NOT threshold relaxation (identical thresholds; only the resume candidate set widens), the
region is the turn regime (not the post-turn walk return), realized-yaw is caliper-invariant,
and a positive/negative control guards against an always-yes gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_action_handoff_inbetween_masked_smoke import MaskedMiddlePredictor, _yaw_metrics
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer
from train.action_handoff_inbetween_soft_endpoint import (
    DEFAULT_REGIME_FRAC,
    PRECISE,
    SOFT,
    CaliperScore,
    region_entry_min_dist,
    resume_region,
    score_rollout,
    soft_endpoint_decision,
    splice_pop,
)
from train.data.action_handoff_inbetween import (
    SEAM_LEN_K,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    load_clip_states,
)

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
DEFAULT_W1B_SUMMARY = "debug_output/_tmp_action_handoff_w1c_gate_migration_20260530/gate_migration_eval_summary.json"

# W1d LOGO parked-bridge artifacts (full-supervision table + per-clip MIRROR-L_R held-outs).
DEFAULT_FULLSUP_STATE = "debug_output/_tmp_action_handoff_w1d_logo_20260531_fullsup/masked_smoke_state.pt"
DEFAULT_MIRROR_STATE_FMT = "debug_output/_tmp_action_handoff_w1d_logo_20260531_mirror_{clip}/masked_smoke_state.pt"
# W1d LOGO held the three grounded clips (Walk_L_To_R is ungrounded → never a LOGO holdout).
LOGO_HELD_OUT_CLIPS = ("Walk_L_To_L", "Walk_R_To_L", "Walk_R_To_R")

# Inherited PROVISIONAL W1d action-only thresholds (NOT changed by this probe).
TAU_YAW_RAD = 0.25
POSE_DEGRADATION_TOL = 0.01


# --------------------------------------------------------------------------- io helpers
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


# ----------------------------------------------------------------- parked-bridge reload
def _load_parked_model(state_path: Path, state_dim: int) -> Tuple[MaskedMiddlePredictor, Dict[str, Any]]:
    blob = torch.load(state_path, map_location="cpu", weights_only=False)
    model = MaskedMiddlePredictor(
        state_dim=state_dim,
        context_len=int(blob["context_len"]),
        seam_len=int(blob["seam_len"]),
        horizon=int(blob["horizon"]),
        hidden=int(blob["args"]["hidden"]),
    )
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, blob


def _walk_f_starts(t_f: int, n_starts: int) -> List[int]:
    return [int(round(x)) % max(t_f, 1) for x in np.linspace(0, t_f - 1, int(n_starts))]


def _generate_bridges(
    model: MaskedMiddlePredictor,
    normalizer: StateNormalizer,
    hub: np.ndarray,
    target: np.ndarray,
    starts: List[int],
    *,
    context_len: int,
    goal_horizon: int,
    horizon: int,
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """Regenerate the parked masked bridges for one turn clip across Walk_F start phases.

    Mirrors the W1d masked-smoke eval EXACTLY (same g0, same seam conditioning) so the bridge
    is the parked artifact, not a new generation. Only the downstream scoring caliper changes.
    """
    t_f = int(hub.shape[0])
    g0 = int(min(goal_horizon, target.shape[0] - SEAM_LEN_K))
    target_middle = target[g0 - horizon : g0]
    goal_seam_raw = target[g0 : g0 + SEAM_LEN_K]
    goal_seam_n = normalizer.normalize(torch.as_tensor(goal_seam_raw, dtype=torch.float32)).unsqueeze(0)
    rolls: List[np.ndarray] = []
    with torch.no_grad():
        for phase in starts:
            idx = (np.arange(phase - context_len, phase) % t_f).astype(np.int64)
            ctx_n = normalizer.normalize(torch.as_tensor(hub[idx], dtype=torch.float32)).unsqueeze(0)
            pred_n = model(ctx_n, goal_seam_n)[0]
            pred_raw = normalizer.denormalize(pred_n).cpu().numpy()
            rolls.append(np.asarray(pred_raw, dtype=np.float64))
    return rolls, np.asarray(target_middle, dtype=np.float64), g0


# ----------------------------------------------------------------- per-caliper scoring
def _score_clip_caliper(
    rolls: List[np.ndarray],
    target: np.ndarray,
    target_middle: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
    caliper: str,
    *,
    goal_horizon: int,
    regime_frac: float,
) -> Dict[str, Any]:
    """Aggregate one clip's bridges under a caliper into action-only-gate fields."""
    pop_safe_vals: List[float] = []
    pose_vals: List[float] = []
    entry_vals: List[float] = []
    yaw_corr_vals: List[float] = []
    heading_mae_vals: List[float] = []
    n_cand = 0
    for roll in rolls:
        sc: CaliperScore = score_rollout(
            roll,
            target,
            std,
            thr,
            caliper,
            goal_horizon=goal_horizon,
            seam_len=SEAM_LEN_K,
            regime_frac=regime_frac,
        )
        n_cand = sc.n_candidates
        pop_safe_vals.append(1.0 if sc.pop_safe else 0.0)
        pose_vals.append(sc.best_pose_d)
        entry_vals.append(sc.region_entry_dist)
        # Realized-yaw is caliper-INVARIANT (verification of the heading ramp, not the endpoint).
        yaw = _yaw_metrics(roll[:, YAW_RATE_SLICE].reshape(-1), target_middle[:, YAW_RATE_SLICE].reshape(-1))
        yaw_corr_vals.append(float(yaw["corr"]))
        heading_mae_vals.append(float(yaw["heading_mae_rad"]))
    return {
        "caliper": caliper,
        "n": int(len(rolls)),
        "n_resume_candidates": int(n_cand),
        "pop_safe_rate": float(np.mean(pop_safe_vals)),
        "best_pose_d_mean": float(np.mean(pose_vals)),
        "region_entry_dist_mean": float(np.mean(entry_vals)),
        "yaw_corr": float(np.nanmean(np.asarray(yaw_corr_vals, dtype=np.float64))),
        "heading_mae_rad": float(np.nanmean(np.asarray(heading_mae_vals, dtype=np.float64))),
    }


def _action_only_pass(row: Mapping[str, Any], *, baseline_pose: float) -> Tuple[bool, Dict[str, bool]]:
    yaw_corr = float(row.get("yaw_corr", float("nan")))
    heading_mae = float(row.get("heading_mae_rad", float("nan")))
    pop_safe = float(row.get("pop_safe_rate", float("nan")))
    best_pose = float(row.get("best_pose_d_mean", float("nan")))
    checks = {
        "yaw_corr_positive": bool(np.isfinite(yaw_corr) and yaw_corr > 0.0),
        "heading_mae_under_tau": bool(np.isfinite(heading_mae) and heading_mae < TAU_YAW_RAD),
        "pop_safe_positive": bool(np.isfinite(pop_safe) and pop_safe > 0.0),
        "best_pose_not_degraded": bool(
            np.isfinite(best_pose)
            and (not np.isfinite(baseline_pose) or best_pose <= baseline_pose + POSE_DEGRADATION_TOL)
        ),
    }
    return bool(all(checks.values())), checks


def _motion_consistency_ok(row: Mapping[str, Any]) -> bool:
    """Red-line check: a revival must keep a REAL heading ramp + contact/vel continuity.

    Realized-yaw correct (corr>0, heading_mae<tau) AND pop genuinely safe (>0). This is the
    bridge's verification — it is independent of widening the resume region, so a wrong-way
    heading ramp (yaw_corr<0) can never be revived by re-anchoring.
    """
    yaw_corr = float(row.get("yaw_corr", float("nan")))
    heading_mae = float(row.get("heading_mae_rad", float("nan")))
    pop_safe = float(row.get("pop_safe_rate", float("nan")))
    return bool(
        np.isfinite(yaw_corr)
        and yaw_corr > 0.0
        and np.isfinite(heading_mae)
        and heading_mae < TAU_YAW_RAD
        and np.isfinite(pop_safe)
        and pop_safe > 0.0
    )


# ----------------------------------------------------------------------------- controls
def _soft_positive_control(
    states: Mapping[str, np.ndarray],
    std: np.ndarray,
    thr: GateThresholds,
    *,
    goal_horizon: int,
    regime_frac: float,
) -> Dict[str, Any]:
    """Recorded turn motion re-anchored under soft must stay pop-safe (gate not always-no)."""
    per_clip: Dict[str, Any] = {}
    all_pass = True
    for clip in TURN_CLIPS:
        target = states[clip]
        region = resume_region(target, SOFT, goal_horizon=goal_horizon, seam_len=SEAM_LEN_K, regime_frac=regime_frac)
        # The recorded in-regime span IS genuine, resumable motion → must re-anchor pop-safe.
        roll = target[region.indices]
        sc = score_rollout(roll, target, std, thr, SOFT, goal_horizon=goal_horizon, seam_len=SEAM_LEN_K, regime_frac=regime_frac)
        ok = bool(sc.pop_safe and sc.clip_resumable)
        per_clip[clip] = {"pop_safe": bool(sc.pop_safe), "best_pose_d": sc.best_pose_d, "pop": sc.pop, "pass": ok}
        all_pass = all_pass and ok
    return {"per_clip": per_clip, "all_pass": bool(all_pass)}


def _soft_negative_control(
    states: Mapping[str, np.ndarray],
    std: np.ndarray,
    thr: GateThresholds,
    *,
    goal_horizon: int,
    horizon: int,
    regime_frac: float,
    n_phases: int = 6,
) -> Dict[str, Any]:
    """Anti-always-yes: straight Walk_F (no turn) scored vs each turn region under soft MUST fail.

    A non-turn rollout has flat yaw_rate → no heading ramp → realized-yaw check fails, so even
    the soft re-anchor cannot pass it. Proves soft re-anchor still discriminates."""
    hub = states[WALK_F]
    t_f = int(hub.shape[0])
    phases = _walk_f_starts(t_f, n_phases)
    per_clip: Dict[str, Any] = {}
    holds = True
    for clip in TURN_CLIPS:
        target = states[clip]
        g0 = int(min(goal_horizon, target.shape[0] - SEAM_LEN_K))
        target_middle = target[g0 - horizon : g0]
        any_pass = False
        rows: List[Dict[str, Any]] = []
        for ph in phases:
            idx = (np.arange(ph, ph + horizon) % t_f).astype(np.int64)
            roll = hub[idx].astype(np.float64)
            sc = score_rollout(roll, target, std, thr, SOFT, goal_horizon=goal_horizon, seam_len=SEAM_LEN_K, regime_frac=regime_frac)
            yaw = _yaw_metrics(roll[:, YAW_RATE_SLICE].reshape(-1), target_middle[:, YAW_RATE_SLICE].reshape(-1))
            row = {
                "pop_safe_rate": 1.0 if sc.pop_safe else 0.0,
                "best_pose_d_mean": sc.best_pose_d,
                "yaw_corr": float(yaw["corr"]),
                "heading_mae_rad": float(yaw["heading_mae_rad"]),
            }
            passed, _ = _action_only_pass(row, baseline_pose=float("inf"))
            any_pass = any_pass or passed
            rows.append({**row, "action_only_pass": bool(passed)})
        per_clip[clip] = {"n_phases": len(phases), "any_phase_passed": bool(any_pass), "rows": rows}
        holds = holds and (not any_pass)
    return {"per_clip": per_clip, "negative_control_holds": bool(holds)}


# --------------------------------------------------------------- Angle A (mechanism)
def _angle_a_splice(
    states: Mapping[str, np.ndarray],
    std: np.ndarray,
    thr: GateThresholds,
    *,
    goal_horizon: int,
    regime_frac: float,
    n_phases: int,
) -> Dict[str, Any]:
    """Arbitrary-phase Walk_F → turn MM-cut: precise fixed-frame vs soft re-anchor pop."""
    hub = states[WALK_F]
    t_f = int(hub.shape[0])
    phases = _walk_f_starts(t_f, n_phases)
    per_clip: Dict[str, Any] = {}
    for clip in TURN_CLIPS:
        target = states[clip]
        pop_precise: List[float] = []
        pop_soft: List[float] = []
        safe_precise: List[float] = []
        safe_soft: List[float] = []
        align_gap: List[float] = []  # min group-std region-entry distance of the cut frame
        soft_region = resume_region(target, SOFT, goal_horizon=goal_horizon, seam_len=SEAM_LEN_K, regime_frac=regime_frac)
        for ph in phases:
            cut = hub[ph].astype(np.float64)
            sp = splice_pop(cut, target, std, thr, PRECISE, goal_horizon=goal_horizon, seam_len=SEAM_LEN_K, regime_frac=regime_frac)
            ss = splice_pop(cut, target, std, thr, SOFT, goal_horizon=goal_horizon, seam_len=SEAM_LEN_K, regime_frac=regime_frac)
            pop_precise.append(sp.pop)
            pop_soft.append(ss.pop)
            safe_precise.append(1.0 if sp.pop_safe else 0.0)
            safe_soft.append(1.0 if ss.pop_safe else 0.0)
            align_gap.append(region_entry_min_dist(cut[None, :], soft_region.frames, std))
        per_clip[clip] = {
            "n_phases": len(phases),
            "pop_precise_mean": float(np.mean(pop_precise)),
            "pop_soft_mean": float(np.mean(pop_soft)),
            "pop_reduction_mean": float(np.mean(pop_precise) - np.mean(pop_soft)),
            "pop_safe_rate_precise": float(np.mean(safe_precise)),
            "pop_safe_rate_soft": float(np.mean(safe_soft)),
            "soft_residual_pop_safe_lt1": bool(np.mean(safe_soft) < 1.0),
            "cut_to_region_gap_mean": float(np.mean(align_gap)),
            "n_soft_candidates": int(soft_region.n_candidates),
        }
    soft_lower = all(per_clip[c]["pop_soft_mean"] <= per_clip[c]["pop_precise_mean"] for c in TURN_CLIPS)
    residual = all(per_clip[c]["soft_residual_pop_safe_lt1"] for c in TURN_CLIPS)
    n_lower = int(sum(per_clip[c]["pop_soft_mean"] <= per_clip[c]["pop_precise_mean"] for c in TURN_CLIPS))
    reading = (
        "arbitrary-phase MM-cut (no bridge) leaves substantial pop under BOTH calipers; "
        f"soft re-anchor reduces it on {n_lower}/{len(TURN_CLIPS)} clips (it can INCREASE pop where "
        "the genuine turn regime sits further from a straight-walk cut frame). Either way the soft "
        "splice still leaves residual unsafe pop and the cut→region gap is large → re-anchoring a raw "
        "cut does NOT substitute for a generated bridge; MM-cut/interpolation is insufficient."
    )
    return {
        "per_clip": per_clip,
        "soft_reanchor_lowers_pop_all_clips": bool(soft_lower),
        "soft_reanchor_lowers_pop_clip_count": n_lower,
        "soft_splice_still_leaves_residual_pop_all_clips": bool(residual),
        "reading": reading,
    }


# ------------------------------------------------------------------------ main pipeline
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Soft-endpoint reframe zero-training falsifiability probe.")
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--w1b-summary", type=Path, default=Path(DEFAULT_W1B_SUMMARY))
    p.add_argument("--fullsup-state", type=Path, default=Path(DEFAULT_FULLSUP_STATE))
    p.add_argument("--mirror-state-fmt", type=str, default=DEFAULT_MIRROR_STATE_FMT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--goal-horizon", type=int, default=12)
    p.add_argument("--regime-frac", type=float, default=DEFAULT_REGIME_FRAC)
    p.add_argument("--angle-a-phases", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    for path in (args.z_features, args.w1b_summary, args.fullsup_state):
        if not Path(path).exists():
            raise FileNotFoundError(f"required artifact not found: {path}")

    torch.manual_seed(int(args.seed))
    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_inbetween_soft_endpoint_probe_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    states = load_clip_states(args.z_features, args.npz_root)
    normalizer = StateNormalizer(states)
    std = normalizer.std.astype(np.float64)
    thr = GateThresholds()
    state_dim = int(states[WALK_F].shape[1])
    hub = states[WALK_F]
    starts = _walk_f_starts(int(hub.shape[0]), int(args.n_starts))

    fullsup_model, fullsup_blob = _load_parked_model(Path(args.fullsup_state), state_dim)
    context_len = int(fullsup_blob["context_len"])
    horizon = int(fullsup_blob["horizon"])

    w1b = _load_json(args.w1b_summary)
    phase2 = w1b["evaluated_objects"]["phase2_trained_goal"]["per_clip"]
    baseline_pose = {
        clip: min(
            float(phase2[clip]["free_no_goal"]["best_pose_d_mean"]),
            float(phase2[clip]["pinned_goal"]["best_pose_d_mean"]),
        )
        for clip in TURN_CLIPS
    }
    recorded_identity_pass = bool(
        w1b["evaluated_objects"]["phase2_trained_goal"]["recorded_turn_positive_control"]
        .get("joint_gate_decision", {})
        .get("all_pass", False)
    )

    # ---- Angle B: re-score the full-supervision parked bridges, precise vs soft, per clip.
    per_clip: Dict[str, Any] = {}
    for clip in TURN_CLIPS:
        rolls, target_middle, g0 = _generate_bridges(
            fullsup_model, normalizer, hub, states[clip], starts,
            context_len=context_len, goal_horizon=int(args.goal_horizon), horizon=horizon,
        )
        rows: Dict[str, Any] = {}
        for caliper in (PRECISE, SOFT):
            row = _score_clip_caliper(
                rolls, states[clip], target_middle, std, thr, caliper,
                goal_horizon=int(args.goal_horizon), regime_frac=float(args.regime_frac),
            )
            passed, checks = _action_only_pass(row, baseline_pose=baseline_pose[clip])
            row["action_only_pass"] = bool(passed)
            row["action_only_checks"] = checks
            rows[caliper] = row
        per_clip[clip] = {
            "seam_start_g0": int(g0),
            "baseline_best_pose_d": float(baseline_pose[clip]),
            PRECISE: rows[PRECISE],
            SOFT: rows[SOFT],
        }

    # ---- Angle B: held-out single rows from the per-clip MIRROR-L_R parked bridges.
    held_out_rows: Dict[str, Any] = {}
    precise_pass: Dict[str, bool] = {}
    soft_pass: Dict[str, bool] = {}
    motion_ok: Dict[str, bool] = {}
    for clip in LOGO_HELD_OUT_CLIPS:
        mirror_path = Path(args.mirror_state_fmt.format(clip=clip))
        if not mirror_path.exists():
            raise FileNotFoundError(f"MIRROR-L_R parked state not found for {clip}: {mirror_path}")
        model, blob = _load_parked_model(mirror_path, state_dim)
        rolls, target_middle, g0 = _generate_bridges(
            model, normalizer, hub, states[clip], starts,
            context_len=int(blob["context_len"]), goal_horizon=int(args.goal_horizon), horizon=int(blob["horizon"]),
        )
        entry = {"seam_start_g0": int(g0), "baseline_best_pose_d": float(baseline_pose[clip])}
        for caliper in (PRECISE, SOFT):
            row = _score_clip_caliper(
                rolls, states[clip], target_middle, std, thr, caliper,
                goal_horizon=int(args.goal_horizon), regime_frac=float(args.regime_frac),
            )
            passed, checks = _action_only_pass(row, baseline_pose=baseline_pose[clip])
            row["action_only_pass"] = bool(passed)
            row["action_only_checks"] = checks
            entry[caliper] = row
        precise_pass[clip] = bool(entry[PRECISE]["action_only_pass"])
        soft_pass[clip] = bool(entry[SOFT]["action_only_pass"])
        motion_ok[clip] = bool(_motion_consistency_ok(entry[SOFT]))
        entry["motion_consistency_ok_soft"] = motion_ok[clip]
        held_out_rows[clip] = entry

    # ---- controls (guard against an always-yes soft caliper).
    pos_ctrl = _soft_positive_control(states, std, thr, goal_horizon=int(args.goal_horizon), regime_frac=float(args.regime_frac))
    neg_ctrl = _soft_negative_control(
        states, std, thr, goal_horizon=int(args.goal_horizon), horizon=horizon, regime_frac=float(args.regime_frac)
    )
    positive_control_pass = bool(recorded_identity_pass and pos_ctrl["all_pass"])
    negative_control_holds = bool(neg_ctrl["negative_control_holds"])

    # ---- Angle A (mechanism side-evidence).
    angle_a = _angle_a_splice(
        states, std, thr, goal_horizon=int(args.goal_horizon), regime_frac=float(args.regime_frac),
        n_phases=int(args.angle_a_phases),
    )

    # ---- pre-committed decision.
    verdict = soft_endpoint_decision(
        held_out_clips=LOGO_HELD_OUT_CLIPS,
        precise_pass=precise_pass,
        soft_pass=soft_pass,
        motion_consistency_ok=motion_ok,
        positive_control_pass=positive_control_pass,
        negative_control_holds=negative_control_holds,
    )

    summary = {
        "task": "Soft-endpoint reframe — zero-training falsifiability probe",
        "no_training": True,
        "no_base_unfreeze": True,
        "no_latent_injection": True,
        "latent_read": "281d_state_space_no_latent_to_inject (region read downstream from generated motion)",
        "calipers": {
            "precise": "resume candidate set = fixed K-frame seam window target[g0:g0+K] (W1d semantics)",
            "soft": "resume candidate set = turn-regime span at/after g0 (|yaw_rate| elevated); re-anchor to best in-regime phase",
            "soft_is_not_threshold_relaxation": True,
            "thresholds_identical_between_calipers": True,
            "realized_yaw_caliper_invariant": True,
        },
        "thresholds": {
            "tau_yaw_rad": TAU_YAW_RAD,
            "pose_degradation_tol": POSE_DEGRADATION_TOL,
            "regime_frac": float(args.regime_frac),
            "tau_pose": thr.tau_pose,
            "tau_pop": thr.tau_pop,
        },
        "pre_committed_decision_rule": {
            "UNPARK_if": "gate valid AND some held-out clip fails precise but passes soft with motion consistency intact → update spec, un-park (bridge training still data-gated; soft endpoint lowers the data need)",
            "KEEP_PARK_if": "gate valid AND no held-out clip revived under soft with motion consistency → formalize soft endpoint into spec, keep PARK until data arrives",
            "GATE_INVALID_if": "soft caliper fails its positive/negative discrimination guard → fix the gate first, hold PARK",
            "angle_a_role": "mechanism side-evidence only (non-binding)",
        },
        "recorded_identity_pass": recorded_identity_pass,
        "soft_positive_control": pos_ctrl,
        "soft_negative_control": neg_ctrl,
        "positive_control_pass": positive_control_pass,
        "negative_control_holds": negative_control_holds,
        "angle_b_full_supervision_per_clip": per_clip,
        "angle_b_held_out_logo": held_out_rows,
        "angle_a_mechanism": angle_a,
        "verdict": {
            "held_out_clips": verdict.held_out_clips,
            "precise_pass": verdict.precise_pass,
            "soft_pass": verdict.soft_pass,
            "motion_consistency_ok": verdict.motion_consistency_ok,
            "revived_clips": verdict.revived_clips,
            "gate_valid": verdict.gate_valid,
            "decision": verdict.decision,
            "reason": verdict.reason,
        },
        "artifacts": {
            "fullsup_state": str(Path(args.fullsup_state).resolve()),
            "w1b_summary": str(Path(args.w1b_summary).resolve()),
        },
    }

    json_path = out_dir / "soft_endpoint_probe_summary.json"
    md_path = out_dir / "soft_endpoint_probe_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# Soft-Endpoint Reframe — Zero-Training Falsifiability Probe")
    lines.append("")
    lines.append("Zero training / no base unfreeze / no latent injection. Soft = region + re-anchor "
                 "(NOT threshold relaxation); thresholds identical between calipers; realized-yaw caliper-invariant.")
    lines.append("")
    lines.append(f"- recorded identity positive control (W1b): {recorded_identity_pass}")
    lines.append(f"- soft positive control (recorded turn re-anchored, pop-safe): {pos_ctrl['all_pass']}")
    lines.append(f"- soft negative control holds (straight Walk_F still fails soft): {negative_control_holds}")
    lines.append(f"- gate valid: {verdict.gate_valid}")
    lines.append("")
    lines.append(f"## DECISION: **{verdict.decision}**")
    lines.append("")
    lines.append(f"- {verdict.reason}")
    lines.append(f"- revived held-out clips: {verdict.revived_clips}")
    lines.append("")
    lines.append("## Angle B — parked bridge re-score (full supervision), soft vs precise")
    lines.append("| clip | caliper | pop_safe | best_pose_d | yaw_corr | heading_MAE_deg | region_entry | n_cand | action_only |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for clip in TURN_CLIPS:
        for caliper in (PRECISE, SOFT):
            r = per_clip[clip][caliper]
            lines.append(
                f"| {clip} | {caliper} | {_fmt(r['pop_safe_rate'],2)} | {_fmt(r['best_pose_d_mean'],4)} | "
                f"{_fmt(r['yaw_corr'],2)} | {_fmt(np.degrees(r['heading_mae_rad']),2)} | "
                f"{_fmt(r['region_entry_dist_mean'],3)} | {r['n_resume_candidates']} | {r['action_only_pass']} |"
            )
    lines.append("")
    lines.append("## Angle B — held-out (MIRROR-L_R) single rows, soft vs precise")
    lines.append("| held-out clip | precise pass | soft pass | motion-consistent (soft) | revived |")
    lines.append("|---|---|---|---|---|")
    for clip in LOGO_HELD_OUT_CLIPS:
        revived = clip in verdict.revived_clips
        lines.append(
            f"| {clip} | {precise_pass[clip]} | {soft_pass[clip]} | {motion_ok[clip]} | {revived} |"
        )
    lines.append("")
    for clip in LOGO_HELD_OUT_CLIPS:
        e = held_out_rows[clip]
        lines.append(
            f"- {clip}: precise(pop_safe={_fmt(e[PRECISE]['pop_safe_rate'],2)}, "
            f"pose={_fmt(e[PRECISE]['best_pose_d_mean'],4)}) "
            f"soft(pop_safe={_fmt(e[SOFT]['pop_safe_rate'],2)}, pose={_fmt(e[SOFT]['best_pose_d_mean'],4)}) "
            f"yaw_corr={_fmt(e[SOFT]['yaw_corr'],2)} heading_MAE_deg={_fmt(np.degrees(e[SOFT]['heading_mae_rad']),2)}"
        )
    lines.append("")
    lines.append("## Angle A — MM-cut pop: precise fixed-frame vs soft re-anchor (mechanism side-evidence)")
    lines.append("| clip | pop precise | pop soft | reduction | pop_safe precise | pop_safe soft | cut→region gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for clip in TURN_CLIPS:
        a = angle_a["per_clip"][clip]
        lines.append(
            f"| {clip} | {_fmt(a['pop_precise_mean'],3)} | {_fmt(a['pop_soft_mean'],3)} | "
            f"{_fmt(a['pop_reduction_mean'],3)} | {_fmt(a['pop_safe_rate_precise'],2)} | "
            f"{_fmt(a['pop_safe_rate_soft'],2)} | {_fmt(a['cut_to_region_gap_mean'],3)} |"
        )
    lines.append("")
    lines.append(f"- {angle_a['reading']}")
    lines.append(f"- soft re-anchor lowers pop (all clips): {angle_a['soft_reanchor_lowers_pop_all_clips']}; "
                 f"soft splice still leaves residual pop (all clips): {angle_a['soft_splice_still_leaves_residual_pop_all_clips']}")
    lines.append("")
    lines.append("Phase-difference characterization: see "
                 "`tools/run_action_handoff_reentry_resolver_diag.py` (arbitrary Walk_F phase has no "
                 "perfectly-aligned turn frame → MM-cut must fail → bridge necessary).")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(
        f"[soft-endpoint] decision={verdict.decision} revived={verdict.revived_clips} "
        f"gate_valid={verdict.gate_valid} pos_ctrl={positive_control_pass} neg_ctrl={negative_control_holds}"
    )


if __name__ == "__main__":
    main()
