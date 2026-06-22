#!/usr/bin/env python3
"""F5 failure decomposition (commanded yaw, zero-training, no base changes).

Replays saved masked-smoke states and decomposes failures by:
  1) Walk_F start phase bucket;
  2) seam pop contributions (ego_vel/contact) at the matched resume frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_action_handoff_inbetween_masked_smoke import MaskedMiddlePredictor
from train.action_handoff_inbetween_commanded_yaw import replace_yaw_rate_slice
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer, evaluate_rollout_state_space
from train.data.action_handoff_inbetween import (
    EGO_VEL_SLICE,
    SEAM_LEN_K,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    CONTACT_SLICE,
    load_clip_states,
)

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
DEFAULT_RUNS = (
    "fullsup=debug_output/_tmp_action_handoff_w1d_logo_f5_20260531_fullsup/masked_smoke_state.pt",
    "mirror_Walk_L_To_L=debug_output/_tmp_action_handoff_w1d_logo_f5_20260531_mirror_Walk_L_To_L/masked_smoke_state.pt",
    "mirror_Walk_R_To_L=debug_output/_tmp_action_handoff_w1d_logo_f5_20260531_mirror_Walk_R_To_L/masked_smoke_state.pt",
    "mirror_Walk_R_To_R=debug_output/_tmp_action_handoff_w1d_logo_f5_20260531_mirror_Walk_R_To_R/masked_smoke_state.pt",
)
DEFAULT_FOCUS_CLIPS = ("Walk_L_To_R", "Walk_R_To_L")


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


def _walk_f_starts(t_f: int, n_starts: int) -> List[int]:
    return [int(round(x)) % max(t_f, 1) for x in np.linspace(0, t_f - 1, int(n_starts))]


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


def _generate_rollouts(
    model: MaskedMiddlePredictor,
    normalizer: StateNormalizer,
    hub: np.ndarray,
    target: np.ndarray,
    starts: Sequence[int],
    *,
    context_len: int,
    goal_horizon: int,
    horizon: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
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
            pred_cmd = replace_yaw_rate_slice(pred_raw, target_middle[:, YAW_RATE_SLICE])
            rolls.append(np.asarray(pred_cmd, dtype=np.float64))
    return rolls, np.asarray(goal_seam_raw, dtype=np.float64)


def _channel_pop_components(
    roll: np.ndarray,
    goal_seam: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
) -> Dict[str, float]:
    state = evaluate_rollout_state_space(roll, goal_seam, std, thr)
    ri = int(state["resume_rollout_frame"])
    tj = int(state["resume_target_frame"])
    comps: Dict[str, float] = {}
    for name, sl in (("ego_vel", EGO_VEL_SLICE), ("contact", CONTACT_SLICE)):
        diff = (roll[ri, sl] - goal_seam[tj, sl]) / std[sl]
        comps[name] = float(np.mean(np.abs(diff)))
    comps["pop"] = float(state["pop"])
    comps["pop_safe"] = float(1.0 if bool(state["pop_safe"]) else 0.0)
    comps["best_pose_d"] = float(state["best_pose_d"])
    comps["resume_rollout_frame"] = float(ri)
    comps["resume_target_frame"] = float(tj)
    return comps


def _phase_bucket(ix: int, n: int, n_buckets: int = 4) -> int:
    if n <= 1:
        return 0
    return int(min(n_buckets - 1, (ix * n_buckets) // n))


def _value_bucket(v: int, total: int, n_buckets: int = 4) -> int:
    if total <= 1:
        return 0
    idx = int(np.clip(v, 0, total - 1))
    return int(min(n_buckets - 1, (idx * n_buckets) // total))


def _decompose_clip(
    *,
    rolls: Sequence[np.ndarray],
    goal_seam: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
    start_phases: Sequence[int],
    walk_f_len: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    for i, roll in enumerate(rolls):
        comp = _channel_pop_components(roll, goal_seam, std, thr)
        comp["start_index"] = float(i)
        comp["phase_bucket"] = float(_phase_bucket(i, len(rolls), n_buckets=4))
        ph = int(start_phases[i]) if i < len(start_phases) else 0
        comp["start_phase"] = float(ph)
        comp["start_phase_bucket"] = float(_value_bucket(ph, walk_f_len, n_buckets=4))
        comp["seam_target_bucket"] = float(
            _value_bucket(int(comp["resume_target_frame"]), goal_seam.shape[0], n_buckets=4)
        )
        rows.append(comp)

    pop = np.asarray([r["pop"] for r in rows], dtype=np.float64)
    safe = np.asarray([r["pop_safe"] for r in rows], dtype=np.float64)
    pose = np.asarray([r["best_pose_d"] for r in rows], dtype=np.float64)
    ego = np.asarray([r["ego_vel"] for r in rows], dtype=np.float64)
    con = np.asarray([r["contact"] for r in rows], dtype=np.float64)

    by_bucket: Dict[str, Any] = {}
    for b in range(4):
        idx = np.asarray([int(r["start_phase_bucket"]) == b for r in rows], dtype=bool)
        if not np.any(idx):
            continue
        by_bucket[str(b)] = {
            "n": int(np.sum(idx)),
            "pop_safe_rate": float(np.mean(safe[idx])),
            "pop_mean": float(np.mean(pop[idx])),
            "best_pose_d_mean": float(np.mean(pose[idx])),
            "ego_vel_pop_mean": float(np.mean(ego[idx])),
            "contact_pop_mean": float(np.mean(con[idx])),
        }

    worst = sorted(rows, key=lambda r: r["pop"], reverse=True)[:5]
    for w in worst:
        w["phase_bucket"] = int(w["phase_bucket"])
        w["start_index"] = int(w["start_index"])
        w["start_phase"] = int(w["start_phase"])
        w["start_phase_bucket"] = int(w["start_phase_bucket"])
        w["seam_target_bucket"] = int(w["seam_target_bucket"])

    return {
        "n_starts": int(len(rows)),
        "pop_safe_rate": float(np.mean(safe)),
        "pop_mean": float(np.mean(pop)),
        "best_pose_d_mean": float(np.mean(pose)),
        "ego_vel_pop_mean": float(np.mean(ego)),
        "contact_pop_mean": float(np.mean(con)),
        "start_phase_buckets_q4": by_bucket,
        "worst_pop_starts_top5": worst,
    }


def _parse_run_specs(specs: Sequence[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"run spec must be name=path, got: {s}")
        name, p = s.split("=", 1)
        out.append((name.strip(), Path(p.strip())))
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="F5 failure decomposition (commanded yaw).")
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--run", action="append", default=list(DEFAULT_RUNS), help="name=masked_smoke_state.pt")
    p.add_argument("--focus-clip", action="append", default=list(DEFAULT_FOCUS_CLIPS), choices=TURN_CLIPS)
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not args.z_features.exists():
        raise FileNotFoundError(f"z-features not found: {args.z_features}")

    run_specs = _parse_run_specs(args.run)
    for _, p in run_specs:
        if not p.exists():
            raise FileNotFoundError(f"masked_smoke state not found: {p}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_f5_failure_decomp_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    states = load_clip_states(args.z_features, args.npz_root)
    normalizer = StateNormalizer(states)
    std = normalizer.std.astype(np.float64)
    thr = GateThresholds()
    hub = states[WALK_F]
    state_dim = int(hub.shape[1])
    focus = [str(c) for c in args.focus_clip]

    summary: Dict[str, Any] = {
        "task": "F5 failure decomposition under commanded yaw",
        "no_training": True,
        "no_base_unfreeze": True,
        "focus_clips": focus,
        "thresholds": {"tau_pose": float(thr.tau_pose), "tau_pop": float(thr.tau_pop)},
        "runs": {},
    }

    for run_name, state_path in run_specs:
        model, blob = _load_parked_model(state_path, state_dim)
        n_starts = int((blob.get("args") or {}).get("n_starts", 20))
        starts = _walk_f_starts(int(hub.shape[0]), n_starts)
        goal_horizon = int((blob.get("args") or {}).get("goal_horizon", 12))
        context_len = int(blob["context_len"])
        horizon = int(blob["horizon"])

        run_out: Dict[str, Any] = {
            "state_path": str(state_path.resolve()),
            "n_starts": n_starts,
            "goal_horizon": goal_horizon,
            "clips": {},
        }
        for clip in focus:
            rolls, goal_seam = _generate_rollouts(
                model,
                normalizer,
                hub,
                states[clip],
                starts,
                context_len=context_len,
                goal_horizon=goal_horizon,
                horizon=horizon,
            )
            run_out["clips"][clip] = _decompose_clip(
                rolls=rolls,
                goal_seam=goal_seam,
                std=std,
                thr=thr,
                start_phases=starts,
                walk_f_len=int(hub.shape[0]),
            )
        summary["runs"][run_name] = run_out

    json_path = out_dir / "f5_failure_decomp_summary.json"
    md_path = out_dir / "f5_failure_decomp_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# F5 Failure Decomposition (Commanded Yaw)")
    lines.append("")
    lines.append("Zero-training replay over saved masked-smoke states.")
    lines.append("Focus: phase buckets and pop contributions (ego_vel/contact).")
    lines.append("")
    for run_name, run in summary["runs"].items():
        lines.append(f"## {run_name}")
        lines.append(f"- state: {run['state_path']}")
        lines.append("| clip | pop_safe_rate | pop_mean | ego_vel_pop | contact_pop | best_pose_d |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for clip in focus:
            c = run["clips"][clip]
            lines.append(
                f"| {clip} | {_fmt(c['pop_safe_rate'],3)} | {_fmt(c['pop_mean'],3)} | "
                f"{_fmt(c['ego_vel_pop_mean'],3)} | {_fmt(c['contact_pop_mean'],3)} | {_fmt(c['best_pose_d_mean'],4)} |"
            )
        for clip in focus:
            c = run["clips"][clip]
            lines.append(f"- {clip} start_phase buckets (q4):")
            for b in ("0", "1", "2", "3"):
                if b not in c["start_phase_buckets_q4"]:
                    continue
                row = c["start_phase_buckets_q4"][b]
                lines.append(
                    f"  - q{b}: n={row['n']}, pop_safe={_fmt(row['pop_safe_rate'],2)}, pop={_fmt(row['pop_mean'],3)}, "
                    f"ego={_fmt(row['ego_vel_pop_mean'],3)}, contact={_fmt(row['contact_pop_mean'],3)}"
                )
        lines.append("")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")


if __name__ == "__main__":
    main()
