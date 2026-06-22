#!/usr/bin/env python3
"""Action-handoff grounded cross-manifold alignment check (spec §7.1 → §7.2).

Productizes the one-off §7.1 data check from
`docs/aperiodic_transition/2026-05-29_goal_conditioned_inbetweening_spec.md`. For
each turn clip it aligns the recorded onset frame (turn[0]) to the Walk_F hub and
reports:

  - pose-only φ vs FULL-STATE φ (pose localizes the cycle-phase neighborhood,
    contact refines within it), with each pick's pose_d and contact_d;
  - the yaw_rate onset ramp (turn clips begin in the walk manifold and ramp
    yaw_rate from ~0), plus Walk_F yaw_rate / ego-lateral flatness as a sanity
    check on the egocentric transform;
  - the per-clip groundability gate verdict (spec §2b).

Known results it MUST reproduce (locked acceptance, spec §2b/§7.1):
  - Walk_R_To_L: groundable — full-state φ = f2, pose_d≈0.011, contact_d≈0.162;
    pose-only φ = f0 had contact_d≈0.96 (→ full-state φ required).
  - Walk_L_To_R: FAILS the gate — pose-top10 min contact_d≈0.70 (onset
    foot-state off the walk cycle) → fallback path; possible authoring
    inconsistency.
  - Walk_L_To_L, Walk_R_To_R: groundable (onset contact gap ≈0.11 / ≈0.03).

Frozen artifacts only; no retrain; no checkpoint dependency (spec §0 lock). The
egocentric state and full-state alignment are imported from
`train.data.action_handoff_inbetween` so this diagnostic and the sampler share a
single source of truth. All thresholds are PROVISIONAL (smoke-only).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    GROUNDED_GROUNDED_FALLBACK_MAX_ONSET,
    POSE_SLICE,
    POSE_TOPK,
    SEAM_LEN_K,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    contact_distance,
    full_state_align,
    load_clip_states,
    pose_distance,
)

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
ONSET_RAMP_FRAMES = 8  # how many onset frames of the turn clip to report for the ramp
LATER_ONSET_SCAN_FRAMES = 16  # later-onset scan depth for failed-gate clips (artifact-backs the fallback)


def _fmt(v: float | None, digits: int = 6) -> str:
    if v is None or not np.isfinite(v):
        return "null"
    return f"{float(v):.{digits}f}"


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grounded cross-manifold alignment check (spec §7.1); frozen artifacts, no retrain."
    )
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    z_path = Path(args.z_features)
    npz_root = Path(args.npz_root)
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(
        f"debug_output/_tmp_action_handoff_grounded_alignment_check_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    states = load_clip_states(z_path, npz_root)
    hub = states[WALK_F]
    n_f = int(hub.shape[0])

    # Egocentric-transform sanity on the Walk_F hub (spec §1.1 rationale):
    # lateral velocity ≈ 0 and yaw_rate ≈ 0 (phase-flat straight walk).
    hub_yaw = hub[:, YAW_RATE_SLICE].reshape(-1)
    hub_ego_lat = hub[:, EGO_VEL_SLICE][:, 1]
    walk_f_sanity = {
        "yaw_rate_min": float(np.min(hub_yaw)),
        "yaw_rate_med": float(np.median(hub_yaw)),
        "yaw_rate_max": float(np.max(hub_yaw)),
        "ego_lat_abs_max": float(np.max(np.abs(hub_ego_lat))),
        "note": "Walk_F is straight walk: yaw_rate≈0 and ego lateral≈0 (turn signal lives in heading rotation).",
    }

    per_clip: dict[str, Any] = {}
    for clip in TURN_CLIPS:
        turn = states[clip]
        onset_frame = turn[0]
        align = full_state_align(
            hub,
            onset_frame,
            topk=int(args.pose_topk),
            contact_thr=float(args.ground_contact_thr),
            pose_thr=float(args.ground_pose_thr),
        )
        ramp_n = int(min(ONSET_RAMP_FRAMES, turn.shape[0]))
        yaw_ramp = turn[:ramp_n, YAW_RATE_SLICE].reshape(-1)
        per_clip[clip] = {
            "turn_frames": int(turn.shape[0]),
            "pose_only_phi": align.pose_only_phi,
            "pose_only_phi_cyc": float(align.pose_only_phi / max(n_f, 1)),
            "pose_only_pose_d": align.pose_only_pose_d,
            "pose_only_contact_d": align.pose_only_contact_d,
            "full_state_phi": align.full_state_phi,
            "full_state_phi_cyc": float(align.full_state_phi / max(n_f, 1)),
            "full_state_pose_d": align.full_state_pose_d,
            "full_state_contact_d": align.full_state_contact_d,
            "pose_topk_frames": align.pose_topk_frames,
            "yaw_rate_onset_ramp": [float(v) for v in yaw_ramp.tolist()],
            "yaw_rate_onset_abs_peak": float(np.max(np.abs(yaw_ramp))),
            "groundable": align.groundable,
            "full_state_differs_from_pose_only": bool(
                align.full_state_phi != align.pose_only_phi
            ),
        }

    groundable_clips = [c for c in TURN_CLIPS if per_clip[c]["groundable"]]
    failed_clips = [c for c in TURN_CLIPS if not per_clip[c]["groundable"]]

    # --- Standardized 281-d L2 comparator (artifact-backs the §3.2 design claim) ---
    # A genuine group-normed 281-d nearest-frame is pose-dominated and collapses to the
    # pose-only pick (leaving the contact gap), which is WHY full-state φ is implemented
    # as pose-localize + contact-refine instead. We record it per clip so the claim is
    # reproducible, not asserted.
    all_states = np.concatenate([states[c] for c in states], axis=0)
    pooled_std = np.std(all_states, axis=0)
    pooled_std = np.where(pooled_std > 1e-8, pooled_std, 1.0)
    hub_norm = hub / pooled_std
    standardized_compare: dict[str, Any] = {}
    for clip in TURN_CLIPS:
        q = states[clip][0]
        d = np.linalg.norm(hub_norm - (q / pooled_std)[None, :], axis=1)
        sphi = int(np.argmin(d))
        standardized_compare[clip] = {
            "standardized_281d_phi": sphi,
            "standardized_281d_contact_d": float(
                contact_distance(hub[:, CONTACT_SLICE], q[CONTACT_SLICE])[sphi]
            ),
            "standardized_281d_pose_d": float(
                pose_distance(hub[:, POSE_SLICE], q[POSE_SLICE])[sphi]
            ),
            "equals_pose_only_phi": bool(sphi == per_clip[clip]["pose_only_phi"]),
            "full_state_phi": per_clip[clip]["full_state_phi"],
        }

    # --- Later-onset scan for failed-gate clips (artifact-backs the fallback path) ---
    # Shows why a later onset does NOT rescue the clip within the sampler's configured
    # scan window, and that scanning past it only trades the contact gate for the pose
    # gate (onset drifts off the Walk_F loop pose).
    fallback_window = int(GROUNDED_GROUNDED_FALLBACK_MAX_ONSET)
    later_onset_scan: dict[str, Any] = {}
    for clip in failed_clips:
        turn = states[clip]
        scan_max = int(min(LATER_ONSET_SCAN_FRAMES, turn.shape[0] - SEAM_LEN_K - 1))
        rows = []
        for o in range(scan_max + 1):
            a = full_state_align(
                hub, turn[o], topk=int(args.pose_topk),
                contact_thr=float(args.ground_contact_thr), pose_thr=float(args.ground_pose_thr),
            )
            rows.append(
                {
                    "onset": o,
                    "phi": a.full_state_phi,
                    "contact_d": a.full_state_contact_d,
                    "pose_d": a.full_state_pose_d,
                    "groundable": a.groundable,
                    "within_fallback_window": bool(1 <= o <= fallback_window),
                }
            )
        in_window = [r for r in rows if r["within_fallback_window"]]
        best_in_window = min(in_window, key=lambda r: r["contact_d"]) if in_window else None
        later_onset_scan[clip] = {
            "fallback_window": fallback_window,
            "scan_rows": rows,
            "best_contact_d_in_window": (best_in_window["contact_d"] if best_in_window else None),
            "best_onset_in_window": (best_in_window["onset"] if best_in_window else None),
            "any_groundable_in_scan": any(r["groundable"] for r in rows),
        }

    summary = {
        "task": "Action handoff grounded cross-manifold alignment check (spec §7.1)",
        "scope": "frozen-artifact data check; no retrain; no checkpoint dependency (spec §0)",
        "z_features_path": str(z_path.resolve()),
        "npz_root": str(npz_root.resolve()),
        "walk_f_cycle_frames": n_f,
        "alignment_method": (
            "full-state φ = pose top-k neighborhood (cycle-phase localization) refined by "
            "min contact distance; ego_vel phase-flat + yaw_rate≈0 at onset are not used to "
            "pick the frame (re-entry resolver design D3)."
        ),
        "provisional_thresholds": {
            "pose_topk": int(args.pose_topk),
            "ground_contact_thr": float(args.ground_contact_thr),
            "ground_pose_thr": float(args.ground_pose_thr),
        },
        "walk_f_egocentric_sanity": walk_f_sanity,
        "per_clip": per_clip,
        "standardized_281d_comparator": {
            "method": "group-normed (pooled std over all locked clips) 281-d nearest Walk_F frame",
            "purpose": "shows the genuine full-dim L2 is pose-dominated → collapses to pose-only; "
            "this is WHY full-state φ uses pose-localize + contact-refine (review §3.2).",
            "per_clip": standardized_compare,
        },
        "later_onset_scan_failed_clips": later_onset_scan,
        "groundable_clips": groundable_clips,
        "failed_gate_clips": failed_clips,
        "fallback_note": (
            "Failed-gate clips (e.g. Walk_L_To_R) drop from pure-grounded sampling: align to "
            "a later onset frame or fall back to within-clip + augmentation (spec §2b); flag as "
            "possible onset authoring inconsistency."
        ),
    }
    json_path = out_dir / "grounded_alignment_check_summary.json"
    _dump_json(json_path, summary)

    lines: list[str] = []
    lines.append("# Action Handoff Grounded Cross-Manifold Alignment Check (spec §7.1)")
    lines.append("")
    lines.append(f"- z-features: {z_path.resolve()}")
    lines.append(f"- raw npz root: {npz_root.resolve()}")
    lines.append(f"- Walk_F cycle frames: {n_f}")
    lines.append(
        f"- thresholds [PROVISIONAL]: pose_topk={args.pose_topk}, "
        f"ground_contact_thr={_fmt(args.ground_contact_thr, 3)}, "
        f"ground_pose_thr={_fmt(args.ground_pose_thr, 3)}"
    )
    lines.append("")
    lines.append("## Walk_F egocentric transform sanity")
    lines.append(
        f"- yaw_rate min/med/max = {_fmt(walk_f_sanity['yaw_rate_min'], 3)} / "
        f"{_fmt(walk_f_sanity['yaw_rate_med'], 3)} / {_fmt(walk_f_sanity['yaw_rate_max'], 3)} rad/s; "
        f"ego lateral |max| = {_fmt(walk_f_sanity['ego_lat_abs_max'], 4)}."
    )
    lines.append("")
    lines.append("## Per-clip onset alignment to Walk_F")
    lines.append(
        "| clip | pose-only φ (contact_d) | full-state φ (cyc) | pose_d | contact_d | yaw onset peak | groundable |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for clip in TURN_CLIPS:
        m = per_clip[clip]
        lines.append(
            f"| {clip} | f{m['pose_only_phi']} ({_fmt(m['pose_only_contact_d'], 3)}) | "
            f"**f{m['full_state_phi']}** ({_fmt(m['full_state_phi_cyc'], 2)}) | "
            f"{_fmt(m['full_state_pose_d'], 3)} | {_fmt(m['full_state_contact_d'], 3)} | "
            f"{_fmt(m['yaw_rate_onset_abs_peak'], 2)} | {m['groundable']} |"
        )
    lines.append("")
    lines.append("## yaw_rate onset ramp (first frames, rad/s)")
    for clip in TURN_CLIPS:
        ramp = ", ".join(_fmt(v, 3) for v in per_clip[clip]["yaw_rate_onset_ramp"])
        lines.append(f"- {clip}: [{ramp}]")
    lines.append("")
    lines.append("## Standardized 281-d comparator (why full-state φ ≠ genuine 281-d L2)")
    lines.append("| clip | std-281d φ (contact_d) | == pose-only φ? | full-state φ |")
    lines.append("|---|---|---|---|")
    for clip in TURN_CLIPS:
        s = standardized_compare[clip]
        lines.append(
            f"| {clip} | f{s['standardized_281d_phi']} ({_fmt(s['standardized_281d_contact_d'], 3)}) | "
            f"{s['equals_pose_only_phi']} | f{s['full_state_phi']} |"
        )
    lines.append(
        "- group-normed 281-d L2 is pose-dominated → collapses to pose-only (e.g. R_L picks f0 "
        "with contact_d≈0.96), leaving the contact gap. Full-state φ uses pose-localize + "
        "contact-refine instead (review §3.2)."
    )
    if later_onset_scan:
        lines.append("")
        lines.append("## Later-onset scan for failed-gate clips")
        for clip, sc in later_onset_scan.items():
            bw = sc["best_contact_d_in_window"]
            lines.append(
                f"- **{clip}** (fallback window = onsets 1..{sc['fallback_window']}): "
                f"best contact_d in window = {_fmt(bw, 3)} at onset {sc['best_onset_in_window']} "
                f"(> {_fmt(args.ground_contact_thr, 2)} ⇒ no rescue); any groundable in scan = "
                f"{sc['any_groundable_in_scan']}."
            )
            lines.append(f"  | onset | φ | contact_d | pose_d | groundable | in-window |")
            lines.append("  |---|---|---|---|---|---|")
            for r in sc["scan_rows"]:
                lines.append(
                    f"  | {r['onset']} | f{r['phi']} | {_fmt(r['contact_d'], 3)} | "
                    f"{_fmt(r['pose_d'], 3)} | {r['groundable']} | {r['within_fallback_window']} |"
                )
        lines.append(
            "- Note: scanning past the window, contact_d eventually clears the contact gate but "
            "pose_d crosses the pose gate (onset drifts off the Walk_F loop pose) — failure reason "
            "shifts, clip stays non-groundable → within-clip + augmentation fallback."
        )
    lines.append("")
    lines.append("## Groundability gate verdict")
    lines.append(f"- groundable clips: {groundable_clips}")
    lines.append(f"- failed-gate clips: {failed_clips}")
    lines.append(f"- fallback: {summary['fallback_note']}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")

    md_path = out_dir / "grounded_alignment_check_summary.md"
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[ok] groundable={groundable_clips} failed={failed_clips}")


if __name__ == "__main__":
    main()
