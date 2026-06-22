#!/usr/bin/env python3
"""F4 commanded-yaw formalization probe (zero training / zero base changes).

Question: if we keep generated pose/contact untouched and replace ONLY generated yaw_rate
with canonical commanded yaw (from cond_dir), do realized-yaw metrics recover while pop
still fails? This isolates F4 (direction control) from F5 (pose/contact continuity).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.run_action_handoff_inbetween_masked_smoke import MaskedMiddlePredictor, _yaw_metrics
from train.action_handoff_inbetween_commanded_yaw import (
    F4CommandedYawDecision,
    classify_f4_commanded_yaw,
    replace_yaw_rate_slice,
)
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer
from train.action_handoff_inbetween_soft_endpoint import PRECISE, SOFT, CaliperScore, score_rollout
from train.data.action_handoff_inbetween import (
    FPS,
    RAW_COND_DIR_SLICE,
    SEAM_LEN_K,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    load_clip_states,
    yaw_rate_from_cond_dir,
)

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
DEFAULT_W1B_SUMMARY = "debug_output/_tmp_action_handoff_w1c_gate_migration_20260530/gate_migration_eval_summary.json"
DEFAULT_MIRROR_STATE_FMT = "debug_output/_tmp_action_handoff_w1d_logo_20260531_mirror_{clip}/masked_smoke_state.pt"
HELD_OUT_CLIPS = ("Walk_L_To_L", "Walk_R_To_L", "Walk_R_To_R")
TAU_YAW_RAD = 0.25
POSE_DEGRADATION_TOL = 0.01


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
    starts: Sequence[int],
    *,
    context_len: int,
    goal_horizon: int,
    horizon: int,
) -> Tuple[List[np.ndarray], np.ndarray, int]:
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


def _load_commanded_yaw_full(npz_root: Path, clip: str, t_aligned: int) -> np.ndarray:
    npz_path = npz_root / f"{clip}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"raw processed npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as d:
        cond_in = np.asarray(d["cond_in"], dtype=np.float64)
    if cond_in.shape[0] < int(t_aligned):
        raise ValueError(
            f"{clip}: cond_in too short for aligned state length ({cond_in.shape[0]} < {t_aligned})"
        )
    cond_dir = cond_in[:t_aligned, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]]
    return np.asarray(yaw_rate_from_cond_dir(cond_dir, fps=FPS), dtype=np.float64)


def _score_rolls(
    rolls: Sequence[np.ndarray],
    target: np.ndarray,
    target_middle: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
    caliper: str,
    *,
    goal_horizon: int,
    baseline_pose: float,
) -> Dict[str, Any]:
    pop_safe_vals: List[float] = []
    pose_vals: List[float] = []
    pop_vals: List[float] = []
    yaw_corr_vals: List[float] = []
    heading_mae_vals: List[float] = []
    n_cand = 0

    tgt_yaw = target_middle[:, YAW_RATE_SLICE].reshape(-1)
    for roll in rolls:
        sc: CaliperScore = score_rollout(
            roll,
            target,
            std,
            thr,
            caliper,
            goal_horizon=goal_horizon,
            seam_len=SEAM_LEN_K,
        )
        n_cand = int(sc.n_candidates)
        pop_safe_vals.append(1.0 if sc.pop_safe else 0.0)
        pose_vals.append(float(sc.best_pose_d))
        pop_vals.append(float(sc.pop))
        yaw = _yaw_metrics(roll[:, YAW_RATE_SLICE].reshape(-1), tgt_yaw)
        yaw_corr_vals.append(float(yaw["corr"]))
        heading_mae_vals.append(float(yaw["heading_mae_rad"]))

    yaw_corr = float(np.nanmean(np.asarray(yaw_corr_vals, dtype=np.float64)))
    heading_mae_rad = float(np.nanmean(np.asarray(heading_mae_vals, dtype=np.float64)))
    pop_safe_rate = float(np.mean(pop_safe_vals))
    best_pose_d_mean = float(np.mean(pose_vals))
    checks = {
        "yaw_corr_positive": bool(np.isfinite(yaw_corr) and yaw_corr > 0.0),
        "heading_mae_under_tau": bool(np.isfinite(heading_mae_rad) and heading_mae_rad < TAU_YAW_RAD),
        "pop_safe_positive": bool(np.isfinite(pop_safe_rate) and pop_safe_rate > 0.0),
        "best_pose_not_degraded": bool(
            np.isfinite(best_pose_d_mean)
            and (not np.isfinite(baseline_pose) or best_pose_d_mean <= baseline_pose + POSE_DEGRADATION_TOL)
        ),
    }
    return {
        "n": int(len(rolls)),
        "n_resume_candidates": int(n_cand),
        "yaw_corr": yaw_corr,
        "heading_mae_rad": heading_mae_rad,
        "heading_mae_deg": float(np.degrees(heading_mae_rad)),
        "pop_safe_rate": pop_safe_rate,
        "best_pose_d_mean": best_pose_d_mean,
        "pop_mean": float(np.mean(pop_vals)),
        "action_only_checks": checks,
        "action_only_pass": bool(all(checks.values())),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="F4 commanded-yaw zero-training probe.")
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--w1b-summary", type=Path, default=Path(DEFAULT_W1B_SUMMARY))
    p.add_argument("--mirror-state-fmt", type=str, default=DEFAULT_MIRROR_STATE_FMT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--goal-horizon", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    for path in (args.z_features, args.w1b_summary):
        if not Path(path).exists():
            raise FileNotFoundError(f"required artifact not found: {path}")
    for clip in HELD_OUT_CLIPS:
        mirror_path = Path(args.mirror_state_fmt.format(clip=clip))
        if not mirror_path.exists():
            raise FileNotFoundError(f"MIRROR-L_R parked state not found for {clip}: {mirror_path}")

    torch.manual_seed(int(args.seed))
    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_f4_commanded_yaw_probe_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    states = load_clip_states(args.z_features, args.npz_root)
    normalizer = StateNormalizer(states)
    std = normalizer.std.astype(np.float64)
    thr = GateThresholds()
    state_dim = int(states[WALK_F].shape[1])
    hub = states[WALK_F]
    starts = _walk_f_starts(int(hub.shape[0]), int(args.n_starts))

    w1b = _load_json(args.w1b_summary)
    phase2 = w1b["evaluated_objects"]["phase2_trained_goal"]["per_clip"]
    baseline_pose = {
        clip: min(
            float(phase2[clip]["free_no_goal"]["best_pose_d_mean"]),
            float(phase2[clip]["pinned_goal"]["best_pose_d_mean"]),
        )
        for clip in TURN_CLIPS
    }

    per_clip: Dict[str, Any] = {}
    for clip in HELD_OUT_CLIPS:
        model, blob = _load_parked_model(Path(args.mirror_state_fmt.format(clip=clip)), state_dim)
        rolls_raw, target_middle, g0 = _generate_bridges(
            model,
            normalizer,
            hub,
            states[clip],
            starts,
            context_len=int(blob["context_len"]),
            goal_horizon=int(args.goal_horizon),
            horizon=int(blob["horizon"]),
        )

        commanded_yaw_full = _load_commanded_yaw_full(Path(args.npz_root), clip, states[clip].shape[0])
        h = int(target_middle.shape[0])
        commanded_yaw_window = commanded_yaw_full[g0 - h : g0]
        if commanded_yaw_window.shape[0] != h:
            raise ValueError(
                f"{clip}: commanded yaw window length mismatch "
                f"({commanded_yaw_window.shape[0]} vs horizon {h})"
            )
        target_yaw = target_middle[:, YAW_RATE_SLICE].reshape(-1)
        gate_target_self = _yaw_metrics(target_yaw, target_yaw)
        gate_cmd_vs_target = _yaw_metrics(commanded_yaw_window.reshape(-1), target_yaw)
        gate_clip_ok = bool(
            np.isfinite(float(gate_target_self["corr"]))
            and float(gate_target_self["corr"]) > 0.999
            and np.isfinite(float(gate_target_self["heading_mae_rad"]))
            and float(gate_target_self["heading_mae_rad"]) < 1e-8
            and np.isfinite(float(gate_cmd_vs_target["corr"]))
            and float(gate_cmd_vs_target["corr"]) > 0.999
            and np.isfinite(float(gate_cmd_vs_target["heading_mae_rad"]))
            and float(gate_cmd_vs_target["heading_mae_rad"]) < 1e-4
        )

        rolls_cmd = [replace_yaw_rate_slice(roll, commanded_yaw_window) for roll in rolls_raw]
        arms = {
            "baseline_generated": rolls_raw,
            "commanded_yaw_only": rolls_cmd,
        }
        clip_rows: Dict[str, Any] = {
            "seam_start_g0": int(g0),
            "horizon_h": int(h),
            "gate_positive_control": {
                "target_vs_self": gate_target_self,
                "commanded_vs_target_middle": gate_cmd_vs_target,
                "pass": gate_clip_ok,
            },
        }
        for caliper in (PRECISE, SOFT):
            cal_rows: Dict[str, Any] = {}
            for arm_name, arm_rolls in arms.items():
                cal_rows[arm_name] = _score_rolls(
                    arm_rolls,
                    states[clip],
                    target_middle,
                    std,
                    thr,
                    caliper,
                    goal_horizon=int(args.goal_horizon),
                    baseline_pose=float(baseline_pose[clip]),
                )
            clip_rows[caliper] = cal_rows
        per_clip[clip] = clip_rows

    gate_valid = bool(all(bool(per_clip[c]["gate_positive_control"]["pass"]) for c in HELD_OUT_CLIPS))
    verdict_by_caliper: Dict[str, F4CommandedYawDecision] = {}
    for caliper in (PRECISE, SOFT):
        base_rows = {c: per_clip[c][caliper]["baseline_generated"] for c in HELD_OUT_CLIPS}
        cmd_rows = {c: per_clip[c][caliper]["commanded_yaw_only"] for c in HELD_OUT_CLIPS}
        verdict_by_caliper[caliper] = classify_f4_commanded_yaw(
            held_out_clips=HELD_OUT_CLIPS,
            baseline_rows=base_rows,
            commanded_rows=cmd_rows,
            tau_yaw_rad=TAU_YAW_RAD,
            gate_valid=gate_valid,
        )

    summary = {
        "task": "F4 commanded-yaw formalization probe (zero-training rescore)",
        "no_training": True,
        "no_base_unfreeze": True,
        "no_upstream_latent_injection": True,
        "held_out_clips": list(HELD_OUT_CLIPS),
        "thresholds": {
            "tau_yaw_rad": TAU_YAW_RAD,
            "pose_degradation_tol": POSE_DEGRADATION_TOL,
            "tau_pose": thr.tau_pose,
            "tau_pop": thr.tau_pop,
        },
        "replacement_policy": "replace only YAW_RATE_SLICE with commanded yaw; keep pose/contact/ego_vel unchanged",
        "per_clip": per_clip,
        "gate_valid": gate_valid,
        "verdict_by_caliper": {k: asdict(v) for k, v in verdict_by_caliper.items()},
        "artifacts": {
            "w1b_summary": str(Path(args.w1b_summary).resolve()),
            "mirror_state_fmt": str(args.mirror_state_fmt),
        },
    }

    json_path = out_dir / "f4_commanded_yaw_probe_summary.json"
    md_path = out_dir / "f4_commanded_yaw_probe_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# F4 Commanded-Yaw Probe (Zero-Training)")
    lines.append("")
    lines.append("Only `YAW_RATE_SLICE` is replaced by commanded yaw from `cond_dir`; pose/contact/ego_vel are untouched.")
    lines.append("")
    lines.append(f"- gate_valid: {gate_valid}")
    lines.append(f"- verdict_precise: {verdict_by_caliper[PRECISE].decision}")
    lines.append(f"- verdict_soft: {verdict_by_caliper[SOFT].decision}")
    lines.append("")
    lines.append("## Held-out metrics")
    lines.append("| clip | caliper | arm | yaw_corr | heading_MAE_deg | pop_safe_rate | pop_mean | best_pose_d | action_only |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for clip in HELD_OUT_CLIPS:
        for caliper in (PRECISE, SOFT):
            for arm in ("baseline_generated", "commanded_yaw_only"):
                r = per_clip[clip][caliper][arm]
                lines.append(
                    f"| {clip} | {caliper} | {arm} | {_fmt(r['yaw_corr'],2)} | {_fmt(r['heading_mae_deg'],2)} | "
                    f"{_fmt(r['pop_safe_rate'],2)} | {_fmt(r['pop_mean'],3)} | {_fmt(r['best_pose_d_mean'],4)} | "
                    f"{r['action_only_pass']} |"
                )
    lines.append("")
    lines.append("## Yaw positive controls")
    lines.append("| clip | target_vs_self corr | target_vs_self MAE_deg | commanded_vs_target corr | commanded_vs_target MAE_deg | pass |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for clip in HELD_OUT_CLIPS:
        g = per_clip[clip]["gate_positive_control"]
        lines.append(
            f"| {clip} | {_fmt(g['target_vs_self']['corr'],4)} | {_fmt(np.degrees(g['target_vs_self']['heading_mae_rad']),6)} | "
            f"{_fmt(g['commanded_vs_target_middle']['corr'],4)} | {_fmt(np.degrees(g['commanded_vs_target_middle']['heading_mae_rad']),6)} | "
            f"{g['pass']} |"
        )
    lines.append("")
    lines.append(f"- precise reason: {verdict_by_caliper[PRECISE].reason}")
    lines.append(f"- soft reason: {verdict_by_caliper[SOFT].reason}")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(
        f"[f4-commanded-yaw] gate_valid={gate_valid} "
        f"precise={verdict_by_caliper[PRECISE].decision} soft={verdict_by_caliper[SOFT].decision}"
    )


if __name__ == "__main__":
    main()
