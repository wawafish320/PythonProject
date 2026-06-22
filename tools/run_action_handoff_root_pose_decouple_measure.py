#!/usr/bin/env python3
"""Debug-only / read-only measurement: is action-handoff `between` a low-dim root problem?

This probe DOES NOT train, DOES NOT touch any production trainer/runtime/gate/checkpoint,
DOES NOT change any loss/model. It only reads the existing `directlocal` (root-excluded
local) representation and the existing reconstruction + acceptance scoring path, and reports
two things on the 188 matched turn windows (Walk_L_To_L / Walk_R_To_L / Walk_R_To_R):

  (a) root-vs-local residual: in the root-excluded local representation
      (rot6d local pose + root-local bone_angvel), how much of the Walk_F -> turn
      difference energy is carried by the *root* channels (heading/yaw/ego-vel + root bone)
      vs the *local* residual (non-root rot6d + non-root bone_angvel).

  (b) decouple usability: drive the Walk_F *local* cycle (phase-aligned) along a *real turn
      root path*, reconstruct world state281 through the EXACT path that gives GT
      acceptance==1.0, and score foot_slip / support_side_correctness / pose_continuity
      against the existing per-clip contract bands.

Path identity: items, reconstruction (`_seq_from_prediction`), GT reconstruction
(`_reconstructed_gt_seq`), band calibration, and `_evaluate_seq_common` are imported
verbatim from `run_action_handoff_oracle_schedule_trajectory_decoder_smoke`, with the same
flags (oracle_contact_passthrough=True, command_align_root_vel=False,
reconstructed_baseline_quantile=100.0) under which reconstructed GT acceptance == 1.0.

Representation note (verified, see summary.md):
  - state281 = pose_rot6d[0:276] + ego_vel[276:278] + yaw_rate[278:279] + contact[279:281].
  - rot6d is a heading-canonical (root-excluded) LOCAL pose: the root bone (dims 0:6)
    temporal std is ~0.1 while the world path curves; world heading is reapplied at
    reconstruction time through cond_dir (see `_world_root_vel_from_ego`).
  - bone_angvel[138] = 46 bones x 3, per-bone LOCAL angular velocity (bone0 rms is
    comparable to limbs, not a dominant shared offset -> not world-frame).
  - "root-excluded local" channels  = rot6d non-root bones[6:276] + bone_angvel non-root[3:138].
  - "root" channels                  = rot6d root bone[0:6] + bone_angvel bone0[0:3]
                                       + ego_vel + yaw_rate + cond_dir heading.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    CONTEXT_LEN_C,
    EGO_VEL_SLICE,
    POSE_SLICE,
    POSE_TOPK,
    STATE_DIM,
    WALK_F,
    YAW_RATE_SLICE,
    contact_distance,
    full_state_align,
    pose_distance,
)
from tools.run_action_handoff_middle_acceptance_replay_probe import (  # noqa: E402
    ANGVEL_DIM,
    DEFAULT_NPZ_ROOT,
    DEFAULT_Z_FEATURES,
    _load_clips,
    _load_skeleton_meta,
)
from tools.run_action_handoff_oracle_schedule_trajectory_decoder_smoke import (  # noqa: E402
    _build_items,
    _calibrate_reconstructed_baseline_bands,
    _calibrate_reconstructed_support_side_bands,
    _evaluate_seq_common,
    _reconstructed_gt_seq,
    _seq_from_prediction,
)
from tools.run_action_handoff_support_schedule_oracle_feasibility_probe import (  # noqa: E402
    DEFAULT_HORIZON,
)

# The 188 "matched" windows are the groundable turn clips; Walk_L_To_R is ungroundable.
GROUNDABLE_TURNS = ("Walk_L_To_L", "Walk_R_To_L", "Walk_R_To_R")

# Path-identity flags (same as the GT guard that reconstructs to acceptance==1.0).
OCP = True   # oracle_contact_passthrough
CARV = False  # command_align_root_vel
RECON_QUANTILE = 100.0

# rot6d / bone_angvel root-vs-local channel splits.
ROT_ROOT = slice(0, 6)        # root bone (heading-canonical body orientation)
ROT_LOCAL = slice(6, 276)     # 45 non-root bones (local articulation)
AV_ROOT = slice(0, 3)         # bone0 angular velocity (root)
AV_LOCAL = slice(3, ANGVEL_DIM)  # 45 non-root bone angular velocities

DEFAULT_OUT_DIR = Path("debug_output/_tmp_action_handoff_root_pose_decouple_measure_20260605")


def _walkf_window(walkf, phi: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk_F local cycle window starting at phase phi, wrapping the loop."""
    n = int(walkf.rot6d.shape[0])
    idx = [(int(phi) + k) % n for k in range(int(horizon))]
    rot6d = np.asarray(walkf.rot6d, dtype=np.float32)[idx]
    angvel = np.asarray(walkf.bone_angvel, dtype=np.float32)[idx]
    state = np.asarray(walkf.state281, dtype=np.float32)[idx]
    return rot6d, angvel, state


def _heading_rad(cond_dir: np.ndarray) -> np.ndarray:
    cd = np.asarray(cond_dir, dtype=np.float64).reshape(-1, 2)
    return np.arctan2(cd[:, 1], cd[:, 0])


def _per_frame_match(turn_state_frame: np.ndarray, hub_state: np.ndarray, topk: int = POSE_TOPK) -> int:
    """Best Walk_F frame for a single turn frame: pose top-k then contact refine
    (the same localization rule as full_state_align). Removes phase drift as a nuisance
    so the residual isolates *irreducible* local articulation/dynamics difference."""
    q = np.asarray(turn_state_frame, dtype=np.float64).reshape(-1)
    pose_d = pose_distance(hub_state[:, POSE_SLICE], q[POSE_SLICE])
    contact_d = contact_distance(hub_state[:, CONTACT_SLICE], q[CONTACT_SLICE])
    k = int(min(max(topk, 1), hub_state.shape[0]))
    topk_frames = np.argsort(pose_d)[:k]
    return int(topk_frames[int(np.argmin(contact_d[topk_frames]))])


def _channel_scales(clips) -> Dict[str, np.ndarray]:
    """Per-channel natural scale = std pooled across ALL clip frames (avoids div-by-zero,
    makes cross-channel energy comparable)."""
    rot = np.concatenate([np.asarray(c.rot6d, dtype=np.float64) for c in clips.values()], axis=0)
    av = np.concatenate([np.asarray(c.bone_angvel, dtype=np.float64) for c in clips.values()], axis=0)
    ego = np.concatenate([np.asarray(c.state281, dtype=np.float64)[:, EGO_VEL_SLICE] for c in clips.values()], axis=0)
    yaw = np.concatenate([np.asarray(c.state281, dtype=np.float64)[:, YAW_RATE_SLICE] for c in clips.values()], axis=0)
    floor = 1e-6
    return {
        "rot6d": np.maximum(rot.std(axis=0), floor),
        "angvel": np.maximum(av.std(axis=0), floor),
        "ego": np.maximum(ego.std(axis=0), floor),
        "yaw": np.maximum(yaw.std(axis=0), floor),
        "heading": np.asarray([1.0]),  # heading compared in radians directly
    }


def _norm_energy(delta: np.ndarray, scale: np.ndarray) -> float:
    d = np.asarray(delta, dtype=np.float64).reshape(-1, scale.shape[-1]) / scale.reshape(1, -1)
    return float(np.sum(d * d))


def walkf_self_floor(walkf, exclude_radius: int = 3) -> Dict[str, float]:
    """Within-Walk_F manifold thickness / noise floor: match each Walk_F frame to its
    nearest OTHER Walk_F frame (excluding +-exclude_radius temporal neighbors) and report
    the residual. This is the yardstick for 'near-invariant': a turn residual at this level
    means the turn pose/dynamics is indistinguishable from Walk_F's own internal spread."""
    hub_state = np.asarray(walkf.state281, dtype=np.float64)
    hub_rot = np.asarray(walkf.rot6d, dtype=np.float64)
    hub_av = np.asarray(walkf.bone_angvel, dtype=np.float64)
    n = hub_state.shape[0]
    rot_local, rot_root, av_total, av_local, av_root_e, pose_ds = [], [], [], [], [], []
    for i in range(n):
        pose_d = pose_distance(hub_state[:, POSE_SLICE], hub_state[i, POSE_SLICE])
        contact_d = contact_distance(hub_state[:, CONTACT_SLICE], hub_state[i, CONTACT_SLICE])
        mask = np.ones(n, dtype=bool)
        lo, hi = max(0, i - exclude_radius), min(n, i + exclude_radius + 1)
        mask[lo:hi] = False
        cand = np.where(mask)[0]
        k = int(min(POSE_TOPK, cand.shape[0]))
        order = cand[np.argsort(pose_d[cand])[:k]]
        j = int(order[int(np.argmin(contact_d[order]))])
        d_rot = hub_rot[i] - hub_rot[j]
        d_av = hub_av[i] - hub_av[j]
        rot_local.append(float(np.sqrt(np.mean(d_rot[ROT_LOCAL] ** 2))))
        rot_root.append(float(np.sqrt(np.mean(d_rot[ROT_ROOT] ** 2))))
        av_total.append(float(np.sqrt(np.mean(d_av ** 2))))
        av_local.append(float(np.sqrt(np.mean(d_av[AV_LOCAL] ** 2))))
        te = float(np.sum(d_av ** 2)); le = float(np.sum(d_av[AV_LOCAL] ** 2))
        av_root_e.append(1.0 - le / max(te, 1e-12))
        pose_ds.append(float(pose_distance(hub_rot[j:j + 1], hub_rot[i])[0]))
    return {
        "n": float(n),
        "pose_d_mean": float(np.mean(pose_ds)),
        "rot6d_local_residual_rms_mean": float(np.mean(rot_local)),
        "rot6d_root_residual_rms_mean": float(np.mean(rot_root)),
        "bone_angvel_total_delta_rms_mean": float(np.mean(av_total)),
        "bone_angvel_local_delta_rms_mean": float(np.mean(av_local)),
        "bone_angvel_root_explained_frac_mean": float(np.mean(av_root_e)),
    }


def measure_a(items, walkf, scales) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """(a) root vs local residual in the root-excluded local representation."""
    rows: List[Dict[str, Any]] = []
    horizon = int(items[0].seq["state281"].shape[0])
    hub_state = np.asarray(walkf.state281, dtype=np.float64)
    hub_rot = np.asarray(walkf.rot6d, dtype=np.float64)
    hub_av = np.asarray(walkf.bone_angvel, dtype=np.float64)
    for it in items:
        query = np.asarray(it.seq["state281"], dtype=np.float64)[0]
        al = full_state_align(hub_state, query)
        phi = int(al.full_state_phi)
        w_rot, w_av, w_state = _walkf_window(walkf, phi, horizon)

        t_rot = np.asarray(it.seq["rot6d"], dtype=np.float64)
        t_av = np.asarray(it.seq["bone_angvel"], dtype=np.float64)
        t_state = np.asarray(it.seq["state281"], dtype=np.float64)

        # --- per-frame phase-optimal match (removes phase drift): each turn frame -> best
        #     Walk_F phase by pose. Residual = irreducible local difference Walk_F cannot
        #     represent at ANY phase.
        match_idx = [_per_frame_match(t_state[k], hub_state) for k in range(t_state.shape[0])]
        m_rot = hub_rot[match_idx]
        m_av = hub_av[match_idx]
        dm_rot = t_rot - m_rot
        dm_av = t_av - m_av

        def _rms(a):
            a = np.asarray(a, dtype=np.float64)
            return float(np.sqrt(np.mean(a * a))) if a.size else 0.0

        pm_rot_local_rms = _rms(dm_rot[:, ROT_LOCAL])
        pm_rot_root_rms = _rms(dm_rot[:, ROT_ROOT])
        pm_av_total_rms = _rms(dm_av)
        pm_av_local_rms = _rms(dm_av[:, AV_LOCAL])
        pm_av_total_e = float(np.sum(dm_av * dm_av))
        pm_av_local_e = float(np.sum(dm_av[:, AV_LOCAL] ** 2))
        pm_av_root_explained = 1.0 - (pm_av_local_e / max(pm_av_total_e, 1e-12))
        pm_pose_d = float(np.mean([
            float(pose_distance(m_rot[k:k + 1], t_rot[k])[0]) for k in range(t_rot.shape[0])
        ]))

        d_rot = t_rot - w_rot
        d_av = t_av - w_av
        d_ego = t_state[:, EGO_VEL_SLICE] - w_state[:, EGO_VEL_SLICE]
        d_yaw = t_state[:, YAW_RATE_SLICE] - w_state[:, YAW_RATE_SLICE]
        # heading: the turn's world heading relative to its own onset (Walk_F heading is
        # ~constant); this is the root "where am I pointing" curve that drives the turn.
        head_turn = np.unwrap(_heading_rad(it.seq["cond_dir"]))
        d_head = head_turn - head_turn[0]

        # raw rms (interpretability)
        def rms(a):
            a = np.asarray(a, dtype=np.float64)
            return float(np.sqrt(np.mean(a * a))) if a.size else 0.0

        rot_local_rms = rms(d_rot[:, ROT_LOCAL])
        rot_root_rms = rms(d_rot[:, ROT_ROOT])
        av_total_rms = rms(d_av)
        av_local_rms = rms(d_av[:, AV_LOCAL])
        av_root_rms = rms(d_av[:, AV_ROOT])
        # fraction of bone_angvel delta variance surviving root removal
        av_total_e = float(np.sum(d_av * d_av))
        av_local_e = float(np.sum(d_av[:, AV_LOCAL] ** 2))
        av_root_explained = 1.0 - (av_local_e / max(av_total_e, 1e-12))

        # normalized energy decomposition (root vs local)
        e_rot_local = _norm_energy(d_rot[:, ROT_LOCAL], scales["rot6d"][ROT_LOCAL])
        e_rot_root = _norm_energy(d_rot[:, ROT_ROOT], scales["rot6d"][ROT_ROOT])
        e_av_local = _norm_energy(d_av[:, AV_LOCAL], scales["angvel"][AV_LOCAL])
        e_av_root = _norm_energy(d_av[:, AV_ROOT], scales["angvel"][AV_ROOT])
        e_ego = _norm_energy(d_ego, scales["ego"])
        e_yaw = _norm_energy(d_yaw, scales["yaw"])
        e_head = _norm_energy(d_head.reshape(-1, 1), scales["heading"])

        e_local = e_rot_local + e_av_local
        e_root = e_rot_root + e_av_root + e_ego + e_yaw + e_head
        root_explained = e_root / max(e_root + e_local, 1e-12)

        rows.append({
            "clip": it.clip,
            "start": int(it.start),
            "end": int(it.end),
            "walkf_phi": phi,
            "align_pose_d": round(float(al.full_state_pose_d), 6),
            "align_contact_d": round(float(al.full_state_contact_d), 6),
            "groundable_align": bool(al.groundable),
            "pm_pose_d_mean": round(pm_pose_d, 6),
            "pm_rot6d_local_residual_rms": round(pm_rot_local_rms, 6),
            "pm_rot6d_root_residual_rms": round(pm_rot_root_rms, 6),
            "pm_bone_angvel_total_delta_rms": round(pm_av_total_rms, 6),
            "pm_bone_angvel_local_delta_rms": round(pm_av_local_rms, 6),
            "pm_bone_angvel_root_explained_frac": round(pm_av_root_explained, 6),
            "rot6d_local_residual_rms": round(rot_local_rms, 6),
            "rot6d_root_residual_rms": round(rot_root_rms, 6),
            "bone_angvel_total_delta_rms": round(av_total_rms, 6),
            "bone_angvel_local_delta_rms": round(av_local_rms, 6),
            "bone_angvel_root_delta_rms": round(av_root_rms, 6),
            "bone_angvel_root_explained_frac": round(av_root_explained, 6),
            "heading_total_turn_rad": round(float(d_head[-1]), 6),
            "norm_e_local": round(e_local, 6),
            "norm_e_root": round(e_root, 6),
            "root_explained_frac": round(root_explained, 6),
        })

    # per-clip aggregates
    per_clip: Dict[str, Dict[str, float]] = {}
    agg_keys = [
        "pm_pose_d_mean",
        "pm_rot6d_local_residual_rms", "pm_rot6d_root_residual_rms",
        "pm_bone_angvel_total_delta_rms", "pm_bone_angvel_local_delta_rms",
        "pm_bone_angvel_root_explained_frac",
        "rot6d_local_residual_rms", "rot6d_root_residual_rms",
        "bone_angvel_total_delta_rms", "bone_angvel_local_delta_rms",
        "bone_angvel_root_delta_rms", "bone_angvel_root_explained_frac",
        "heading_total_turn_rad", "root_explained_frac",
    ]
    for clip in GROUNDABLE_TURNS:
        crows = [r for r in rows if r["clip"] == clip]
        if not crows:
            continue
        per_clip[clip] = {"n": float(len(crows))}
        for k in agg_keys:
            vals = np.asarray([r[k] for r in crows], dtype=np.float64)
            per_clip[clip][f"{k}_mean"] = float(np.mean(vals))
            per_clip[clip][f"{k}_p95"] = float(np.percentile(vals, 95))
    # overall
    allrows = rows
    per_clip["ALL"] = {"n": float(len(allrows))}
    for k in agg_keys:
        vals = np.asarray([r[k] for r in allrows], dtype=np.float64)
        per_clip["ALL"][f"{k}_mean"] = float(np.mean(vals))
        per_clip["ALL"][f"{k}_p95"] = float(np.percentile(vals, 95))
    return rows, per_clip


def measure_b(items, walkf, skel, baseline_bands, support_bands) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(b) decouple usability: Walk_F local pose + real turn root path -> contract."""
    rows: List[Dict[str, Any]] = []
    horizon = int(items[0].seq["state281"].shape[0])
    for it in items:
        query = np.asarray(it.seq["state281"], dtype=np.float64)[0]
        al = full_state_align(np.asarray(walkf.state281, dtype=np.float64), query)
        phi = int(al.full_state_phi)
        w_rot, w_av, _w_state = _walkf_window(walkf, phi, horizon)

        # Decoupled state281: Walk_F local pose (rot6d) + turn root channels
        # (ego_vel, yaw_rate, contact). Contact is oracle-passthrough overwritten with the
        # turn schedule anyway. cond_dir + root_pos[0] (the real turn root path) come from
        # item.seq inside _seq_from_prediction.
        dec_state = np.asarray(it.seq["state281"], dtype=np.float32).copy()
        dec_state[:, POSE_SLICE] = w_rot.astype(np.float32)
        dec_aux = w_av.astype(np.float32)  # Walk_F regime witness travels with the pose

        seq = _seq_from_prediction(
            it, dec_state, dec_aux,
            oracle_contact_passthrough=OCP, command_align_root_vel=CARV,
        )
        r = _evaluate_seq_common(
            variant="decouple_walkf_pose_turn_root", split="all", split_kind="decouple",
            partition="all", item=it, seq=seq,
            baseline_bands=baseline_bands, support_bands=support_bands, skeleton=skel,
            min_run_frames=2, endpoint_note="decouple", oracle_contact_passthrough=OCP,
            command_align_root_vel=CARV, calibration_domain="reconstructed_state281",
        )
        metrics = r.get("metrics", {}) or {}
        rows.append({
            "clip": it.clip,
            "start": int(it.start),
            "end": int(it.end),
            "walkf_phi": phi,
            "pass": bool(r.get("pass", False)),
            "foot_slip_p95_mps": round(float(metrics.get("foot_slip_p95_mps", 0.0) or 0.0), 6),
            "foot_slip_to_band_ratio": round(float(r.get("foot_slip_p95_to_band_ratio", 0.0) or 0.0), 6),
            "support_honesty": bool(r.get("support_honesty", False)),
            "support_side_correctness": bool(r.get("support_side_correctness", False)),
            "pose_continuity": bool(r.get("pose_continuity", False)),
            "regime_reached": bool(r.get("regime_reached", False)),
            "rate_budget": bool(r.get("rate_budget", False)),
            "command_response": bool(r.get("command_response", False)),
            "endpoint_bridgeability": bool(r.get("endpoint_bridgeability", False)),
            "failed_family": str(r.get("failed_family", "")),
        })

    def rate(rs, key):
        return float(np.mean([bool(x[key]) for x in rs])) if rs else 0.0

    summary: Dict[str, Any] = {}
    for clip in list(GROUNDABLE_TURNS) + ["ALL"]:
        rs = rows if clip == "ALL" else [r for r in rows if r["clip"] == clip]
        if not rs:
            continue
        summary[clip] = {
            "n": len(rs),
            "acceptance_pass_rate": rate(rs, "pass"),
            "support_honesty_foot_slip_pass_rate": rate(rs, "support_honesty"),
            "support_side_correctness_pass_rate": rate(rs, "support_side_correctness"),
            "pose_continuity_pass_rate": rate(rs, "pose_continuity"),
            "regime_reached_pass_rate": rate(rs, "regime_reached"),
            "rate_budget_pass_rate": rate(rs, "rate_budget"),
            "command_response_pass_rate": rate(rs, "command_response"),
            "foot_slip_p95_mps_mean": float(np.mean([r["foot_slip_p95_mps"] for r in rs])),
            "foot_slip_to_band_ratio_mean": float(np.mean([r["foot_slip_to_band_ratio"] for r in rs])),
        }
    return rows, summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = _load_clips(DEFAULT_NPZ_ROOT, DEFAULT_Z_FEATURES)
    skel = _load_skeleton_meta(DEFAULT_NPZ_ROOT)
    walkf = clips[WALK_F]

    all_items = _build_items(
        clips, horizon=DEFAULT_HORIZON, context_len=CONTEXT_LEN_C, min_run_frames=2, stride=1
    )
    items = [it for it in all_items if it.clip in GROUNDABLE_TURNS]
    assert len(items) == 188, f"expected 188 matched windows, got {len(items)}"

    scales = _channel_scales(clips)

    # ---- (a)
    a_rows, a_per_clip = measure_a(items, walkf, scales)
    a_per_clip["WALKF_SELF_FLOOR"] = walkf_self_floor(walkf)
    _write_csv(out_dir / "per_window_a.csv", a_rows)
    # per-clip aggregate (root explained ratio + local residual), incl. Walk_F self-floor null
    pc_rows = []
    for clip, rec in a_per_clip.items():
        row = {"group": clip}
        row.update({k: round(float(v), 6) for k, v in rec.items()})
        pc_rows.append(row)
    _write_csv(out_dir / "per_clip.csv", pc_rows)

    # ---- bands (GT reconstructed, same path as acceptance==1.0)
    baseline_bands = _calibrate_reconstructed_baseline_bands(
        items, skel, quantile=RECON_QUANTILE, oracle_contact_passthrough=OCP,
        command_align_root_vel=CARV,
    )
    support_bands = _calibrate_reconstructed_support_side_bands(
        items, skel, horizon=DEFAULT_HORIZON, min_run_frames=2,
        oracle_contact_passthrough=OCP, command_align_root_vel=CARV,
    )

    # ---- GT reference (must be 1.0 -> path identity)
    gt_pass = []
    for it in items:
        seq = _reconstructed_gt_seq(it, oracle_contact_passthrough=OCP, command_align_root_vel=CARV)
        r = _evaluate_seq_common(
            variant="gt", split="all", split_kind="gt", partition="all", item=it, seq=seq,
            baseline_bands=baseline_bands, support_bands=support_bands, skeleton=skel,
            min_run_frames=2, endpoint_note="gt", oracle_contact_passthrough=OCP,
            command_align_root_vel=CARV, calibration_domain="reconstructed_state281",
        )
        gt_pass.append(bool(r["pass"]))
    gt_accept = float(np.mean(gt_pass))

    # ---- (b)
    b_rows, b_summary = measure_b(items, walkf, skel, baseline_bands, support_bands)
    _write_csv(out_dir / "decouple_usability.csv", b_rows)

    result = {
        "n_matched_windows": len(items),
        "horizon": int(DEFAULT_HORIZON),
        "flags": {"oracle_contact_passthrough": OCP, "command_align_root_vel": CARV,
                   "reconstructed_baseline_quantile": RECON_QUANTILE},
        "gt_reconstructed_acceptance": gt_accept,
        "a_root_vs_local": a_per_clip,
        "b_decouple_usability": b_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
