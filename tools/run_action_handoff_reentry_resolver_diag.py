#!/usr/bin/env python3
"""Action-handoff re-entry resolver diagnostic (pose-first).

Follow-up to the attractor-support audit A3, which found that `z` is NOT
cross-clip phase-comparable for picking the Walk_F re-entry frame (0/4 agree
with contact). This tool determines which signal DOES give a reliable re-entry
phase, by comparing three candidate signals per turn clip end-frame against the
Walk_F cycle:

  - pose   : full-body rot6d L2 (design pose_l2 metric, /sqrt(dim))
  - contact: soft-contact L2 (future_desc contact channels)
  - egovel : egocentric (heading-invariant) planar velocity, from raw world
             root_vel rotated by -heading (cond_dir = cond_in[:, 4:6])

Findings this tool reproduces (frozen artifacts, no retrain):
  - egocentric velocity is ~phase-flat for forward-walk clips (lateral ~0,
    forward only mildly oscillating) -> it cannot disambiguate gait phase, so a
    velocity "tiebreak" is not a usable re-entry signal.
  - turn-clip end pose is sharply nearest the Walk_F cycle start (these clips are
    authored to return to the loop pose) -> pose localizes the re-entry phase
    neighborhood unambiguously, including for Walk_R_To_L which contact aliases.

Resolver (the contract's component [2]):
  1. pose top-k nearest Walk_F frames -> phase neighborhood (must be sharp).
  2. within that neighborhood, pick min contact distance -> re-entry frame.
  3. velocity is not used; world-frame turn residual is the bridge's (root-yaw).

Caveat: re-entry is well-defined BECAUSE these 5 clips terminate near the Walk_F
loop pose. Clips that do not return to the loop pose must be re-checked.

No retrain; no in-basin pipeline change (design §0 lock).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

LOCKED_CLIPS = (
    "Walk_F",
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)
TURN_CLIPS = (
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
)
WALK_F = "Walk_F"

CONTACT_SLICE = (278, 280)
COND_DIR_SLICE = (4, 6)  # cond_in = act_oh(4) + cond_dir(2) + cond_speed(1)

EPS = 1e-8
DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"

# Provisional gates (smoke-only; calibrate before production).
POSE_TOPK = 5
POSE_SHARP_MAX = 0.15  # top-3 pose-NN circular cycle spread; above -> phase ambiguous


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


def _circular_index_gap(i: int, j: int, period: int) -> int:
    delta = abs(int(i) - int(j))
    return int(min(delta, period - delta))


def _circular_spread(indices: list[int], period: int) -> float:
    if not indices:
        return 0.0
    ref = int(indices[0])
    return float(
        np.mean([_circular_index_gap(int(j), ref, period) for j in indices]) / max(period, 1)
    )


def _l2_to_query(track: np.ndarray, query: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    return np.linalg.norm(track - query[None, :], axis=1) / scale


def _load_signals(
    z_features: Path, npz_root: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Return per-clip aligned contact, pose, egovel + frame counts.

    Frames align from index 0 (the z probe truncated all features to t_feat from
    the front), so z-features frame i == raw npz frame i.
    """
    z = np.load(z_features, allow_pickle=True)
    contact_by: dict[str, np.ndarray] = {}
    pose_by: dict[str, np.ndarray] = {}
    egovel_by: dict[str, np.ndarray] = {}
    nframes: dict[str, int] = {}

    for clip in LOCKED_CLIPS:
        fd_key = f"{clip}__future_desc"
        if fd_key not in z.files:
            raise RuntimeError(f"missing npz key: {fd_key}")
        contact_full = np.asarray(z[fd_key], dtype=np.float64)[:, CONTACT_SLICE[0] : CONTACT_SLICE[1]]

        raw_path = npz_root / f"{clip}.npz"
        if not raw_path.exists():
            raise FileNotFoundError(f"raw processed npz not found: {raw_path}")
        with np.load(raw_path, allow_pickle=True) as d:
            bone_rot6d = np.asarray(d["bone_rot6d"], dtype=np.float64)
            root_vel = np.asarray(d["root_vel"], dtype=np.float64)  # world-frame planar
            cond_in = np.asarray(d["cond_in"], dtype=np.float64)

        t = int(min(contact_full.shape[0], bone_rot6d.shape[0], root_vel.shape[0], cond_in.shape[0]))
        if t < 2:
            raise RuntimeError(f"{clip}: aligned frame count too small ({t})")

        pose = bone_rot6d[:t].reshape(t, -1)
        contact = contact_full[:t]

        rv = root_vel[:t]
        cd = cond_in[:t, COND_DIR_SLICE[0] : COND_DIR_SLICE[1]]
        fdir = cd / np.maximum(np.linalg.norm(cd, axis=1, keepdims=True), EPS)
        ego_fwd = rv[:, 0] * fdir[:, 0] + rv[:, 1] * fdir[:, 1]
        ego_lat = -rv[:, 0] * fdir[:, 1] + rv[:, 1] * fdir[:, 0]
        egovel = np.stack([ego_fwd, ego_lat], axis=1)

        contact_by[clip] = contact
        pose_by[clip] = pose
        egovel_by[clip] = egovel
        nframes[clip] = t

    return contact_by, pose_by, egovel_by, nframes


def _resolve_clip(
    clip: str,
    *,
    pose_f: np.ndarray,
    contact_f: np.ndarray,
    egovel_f: np.ndarray,
    pose_c: np.ndarray,
    contact_c: np.ndarray,
    egovel_c: np.ndarray,
    n_f: int,
    pose_scale: float,
) -> dict[str, Any]:
    e = pose_c.shape[0] - 1  # turn-end re-entry candidate

    pose_d = _l2_to_query(pose_f, pose_c[e], scale=pose_scale)
    contact_d = _l2_to_query(contact_f, contact_c[e])
    egovel_d = _l2_to_query(egovel_f, egovel_c[e])

    pose_nn = int(np.argmin(pose_d))
    contact_nn = int(np.argmin(contact_d))
    egovel_nn = int(np.argmin(egovel_d))

    pose_top = np.argsort(pose_d)[:POSE_TOPK]
    pose_sharpness = _circular_spread([int(j) for j in np.argsort(pose_d)[:3].tolist()], n_f)

    # Resolver: pose localizes the phase neighborhood; contact picks within it.
    refined = int(pose_top[int(np.argmin(contact_d[pose_top]))])

    return {
        "turn_end_frame": int(e),
        "pose_nn_frame": pose_nn,
        "pose_nn_cycle_frac": float(pose_nn / max(n_f, 1)),
        "pose_sharpness_circular": pose_sharpness,
        "contact_nn_frame": contact_nn,
        "contact_nn_cycle_frac": float(contact_nn / max(n_f, 1)),
        "egovel_nn_frame": egovel_nn,
        "egovel_nn_cycle_frac": float(egovel_nn / max(n_f, 1)),
        "pose_topk_frames": [int(j) for j in pose_top.tolist()],
        "pose_topk_cycle_fracs": [float(j / max(n_f, 1)) for j in pose_top.tolist()],
        "resolver_reentry_frame": refined,
        "resolver_reentry_cycle_frac": float(refined / max(n_f, 1)),
        "resolver_contact_d_at_pick": float(contact_d[refined]),
        "resolver_pose_d_at_pick": float(pose_d[refined]),
        "pose_vs_contact_disagree": bool(
            _circular_index_gap(pose_nn, contact_nn, n_f) / max(n_f, 1) >= 0.15
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-entry resolver diagnostic (pose-first); frozen artifacts, no retrain."
    )
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    z_path = Path(args.z_features)
    npz_root = Path(args.npz_root)
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(
        f"debug_output/_tmp_action_handoff_reentry_resolver_diag_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    contact_by, pose_by, egovel_by, nframes = _load_signals(z_path, npz_root)

    pose_f = pose_by[WALK_F]
    contact_f = contact_by[WALK_F]
    egovel_f = egovel_by[WALK_F]
    n_f = nframes[WALK_F]
    pose_scale = float(np.sqrt(pose_f.shape[1]))

    # Egocentric-velocity phase-flatness evidence on the Walk_F cycle.
    egovel_flatness = {
        "forward_min": float(np.min(egovel_f[:, 0])),
        "forward_max": float(np.max(egovel_f[:, 0])),
        "forward_range": float(np.ptp(egovel_f[:, 0])),
        "lateral_abs_max": float(np.max(np.abs(egovel_f[:, 1]))),
        "note": "lateral ~0 and forward only mildly oscillating -> egovel carries little gait-phase info",
    }

    per_clip: dict[str, Any] = {}
    pose_sharp_vals: list[float] = []
    pose_reentry_fracs: list[float] = []
    disagree_count = 0
    for clip in TURN_CLIPS:
        res = _resolve_clip(
            clip,
            pose_f=pose_f,
            contact_f=contact_f,
            egovel_f=egovel_f,
            pose_c=pose_by[clip],
            contact_c=contact_by[clip],
            egovel_c=egovel_by[clip],
            n_f=n_f,
            pose_scale=pose_scale,
        )
        per_clip[clip] = res
        pose_sharp_vals.append(res["pose_sharpness_circular"])
        pose_reentry_fracs.append(res["resolver_reentry_cycle_frac"])
        if res["pose_vs_contact_disagree"]:
            disagree_count += 1

    mean_pose_sharp = float(np.mean(pose_sharp_vals))
    all_pose_sharp = all(s < POSE_SHARP_MAX for s in pose_sharp_vals)
    if all_pose_sharp:
        verdict = "pose_first_resolver_well_supported"
    else:
        verdict = "pose_phase_ambiguous_fallback_required"

    summary = {
        "task": "Action handoff re-entry resolver diagnostic (pose-first)",
        "scope": "frozen-artifact follow-up to audit A3; no retrain; in-basin pipeline locked",
        "z_features_path": str(z_path.resolve()),
        "npz_root": str(npz_root.resolve()),
        "walk_f_cycle_frames": n_f,
        "signals": {
            "pose": "bone_rot6d L2 / sqrt(dim) (design pose_l2)",
            "contact": "future_desc contact channels L2",
            "egovel": "egocentric planar velocity (world root_vel rotated by -cond_dir heading)",
        },
        "egovel_phase_flatness_walk_f": egovel_flatness,
        "provisional_gates": {"pose_topk": POSE_TOPK, "pose_sharp_max": POSE_SHARP_MAX},
        "per_clip": per_clip,
        "aggregate": {
            "mean_pose_sharpness": mean_pose_sharp,
            "all_clips_pose_sharp": bool(all_pose_sharp),
            "pose_vs_contact_disagree_count": disagree_count,
            "n_turn_clips": len(TURN_CLIPS),
            "resolver_reentry_cycle_fracs": {c: per_clip[c]["resolver_reentry_cycle_frac"] for c in TURN_CLIPS},
        },
        "caveat": (
            "Re-entry is well-defined because these 5 clips terminate near the Walk_F loop "
            "pose; clips that do not return to the loop pose must be re-checked. Single Walk_F "
            "cycle: multi-cycle aliasing untestable (closeout §13)."
        ),
        "verdict": verdict,
    }
    json_path = out_dir / "reentry_resolver_diag_summary.json"
    _dump_json(json_path, summary)

    lines: list[str] = []
    lines.append("# Action Handoff Re-entry Resolver Diagnostic (pose-first)")
    lines.append("")
    lines.append(f"- z-features: {z_path.resolve()}")
    lines.append(f"- raw npz root: {npz_root.resolve()}")
    lines.append(f"- Walk_F cycle frames: {n_f}")
    lines.append("")
    lines.append("## Egocentric velocity is phase-flat (why velocity is not a tiebreak)")
    lines.append(
        f"- Walk_F egovel forward range={_fmt(egovel_flatness['forward_range'])} "
        f"(min={_fmt(egovel_flatness['forward_min'])}, max={_fmt(egovel_flatness['forward_max'])}), "
        f"lateral_abs_max={_fmt(egovel_flatness['lateral_abs_max'])}."
    )
    lines.append("")
    lines.append("## Per-clip re-entry (cycle fraction)")
    lines.append("| clip | pose-NN | contact-NN | egovel-NN | resolver | pose sharpness |")
    lines.append("|---|---|---|---|---|---|")
    for clip in TURN_CLIPS:
        m = per_clip[clip]
        lines.append(
            f"| {clip} | {_fmt(m['pose_nn_cycle_frac'],2)} | {_fmt(m['contact_nn_cycle_frac'],2)} | "
            f"{_fmt(m['egovel_nn_cycle_frac'],2)} | **{_fmt(m['resolver_reentry_cycle_frac'],2)}** "
            f"(f{m['resolver_reentry_frame']}) | {_fmt(m['pose_sharpness_circular'],3)} |"
        )
    lines.append("")
    lines.append(
        f"- aggregate: mean pose sharpness={_fmt(mean_pose_sharp,3)}, all clips pose-sharp={all_pose_sharp}, "
        f"pose-vs-contact disagree={disagree_count}/{len(TURN_CLIPS)}."
    )
    lines.append(f"- caveat: {summary['caveat']}")
    lines.append(f"- **verdict: {verdict}**")
    lines.append("")
    lines.append("## Resolver contract")
    lines.append("1. pose top-k nearest Walk_F frames -> phase neighborhood (require pose sharpness < gate).")
    lines.append("2. within that neighborhood pick min contact distance -> re-entry frame.")
    lines.append("3. egocentric velocity is NOT used (phase-flat); world-frame turn residual -> bridge root-yaw.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")

    md_path = out_dir / "reentry_resolver_diag_summary.md"
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[ok] verdict={verdict} mean_pose_sharpness={mean_pose_sharp:.3f} disagree={disagree_count}/{len(TURN_CLIPS)}")


if __name__ == "__main__":
    main()
