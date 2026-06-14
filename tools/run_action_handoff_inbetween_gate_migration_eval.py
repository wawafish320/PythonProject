#!/usr/bin/env python3
"""W1b read-only migration eval for action-handoff in-betweening binding gate.

No training is run here. The tool loads the frozen base checkpoint and the saved W1a
PHASE2 state, rebuilds hidden_pre anchors from each evaluated model's own capture, and
compares the legacy radius diagnostic with the migrated joint action gate.
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

import train.validate.run_freerun_cycles as freerun  # noqa: E402
from train.action_handoff_inbetween_goal_injection import (  # noqa: E402
    DEFAULT_POSE_DEGRADATION_TOL,
    DEFAULT_SELF_REACH_RATE_LIFT,
    DEFAULT_YAW_MAE_TAU_RAD,
    joint_action_binding_gate_decision,
    summarize_reach_rate,
)
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer, evaluate_rollout_state_space  # noqa: E402
from train.action_handoff_inbetween_reach import (  # noqa: E402
    DEFAULT_CONV_NORM_THR,
    DEFAULT_END_WINDOW_K,
    DEFAULT_SELF_REACH_K,
    DEFAULT_SELF_REACH_K_VALUES,
    LOCKED_CLIPS,
    TURN_CLIPS,
    build_hidden_pre_anchors,
    build_same_source_hidden_pre_anchors,
    k_label,
    load_hidden_pre,
    summarize_absolute_self_reach,
)
from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    POSE_SLICE,
    RAW_COND_DIR_SLICE,
    SEAM_LEN_K,
    TURN_CLIPS as DATA_TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    load_clip_states,
)
from tools.run_action_handoff_inbetween_reach_aware_rewire_probe import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CKPT,
    DEFAULT_ENCODER_BUNDLE,
    DEFAULT_NPZ_ROOT,
    DEFAULT_PRETRAIN_TEMPLATE,
    DEFAULT_Z_FEATURES,
    FPS,
    _capture_context_teacher_hidden,
    _capture_fullseq_hidden,
    _fmt,
    _goal_flat,
    _run_context_ar_rollout,
    _runner_args,
)
from tools.run_action_handoff_inbetween_reach_honesty_probe import (  # noqa: E402
    DEFAULT_PHASE2_SUMMARY,
    ROOT_VEL_OUT_SLICE,
    _apply_phase2_model_state,
    _build_goal_head_from_phase2_state,
    _goal_config_from_phase2_state,
    _load_phase2_state,
    _mean_finite,
    _rollout_state_and_yaw,
    _run_target_cond_self_contact_rollout,
    _target_cond_dir_from_sample,
    _yaw_alignment,
    calibration_relerr,
)

DEFAULT_PHASE2_STATE = (
    "debug_output/_tmp_action_handoff_inbetween_phase2_guarded_finetune_20260530_w1a_state/"
    "phase2_trained_state.pt"
)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json_if_present(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(tok.strip()) for tok in str(raw or "").replace(";", ",").split(",") if tok.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([bool(r.get(key, False)) for r in rows]))


def _metric_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _mean_finite([float(r.get(key, float("nan"))) for r in rows])


def _metric_min(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.min(vals)) if vals.size else float("nan")


def _metric_max(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals)) if vals.size else float("nan")


def _load_raw_turn_motion(npz_root: Path, clip: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_root / f"{clip}.npz", allow_pickle=True) as d:
        root_vel = np.asarray(d["root_vel"], dtype=np.float64)
        cond_in = np.asarray(d["cond_in"], dtype=np.float64)
    cond_dir = cond_in[:, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]]
    return root_vel, cond_dir


def _build_samples(
    runner: Any,
    *,
    npz_root: Path,
    target_states: Mapping[str, np.ndarray],
    goal_horizon: int,
) -> Tuple[Any, Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    walk_ds = runner._build_dataset(npz_root / f"{WALK_F}.npz", seq_len=64)
    runner._ensure_model_ready(walk_ds)
    walk_clip = walk_ds.clips[0]
    walk_sample = freerun._build_full_cycle_sample(walk_ds, walk_clip, seq_len=int(walk_clip.X.shape[0]))

    target_samples: Dict[str, Dict[str, torch.Tensor]] = {}
    for clip in TURN_CLIPS:
        target_T = int(target_states[clip].shape[0])
        ds = runner._build_dataset(npz_root / f"{clip}.npz", seq_len=target_T)
        runner._ensure_model_ready(ds)
        target_samples[clip] = freerun._build_full_cycle_sample(ds, ds.clips[0], seq_len=target_T)

    goal_flat: Dict[str, torch.Tensor] = {}
    goal_seam: Dict[str, np.ndarray] = {}
    for clip in TURN_CLIPS:
        target = target_states[clip]
        g0 = int(min(int(goal_horizon), target.shape[0] - SEAM_LEN_K))
        goal_seam[clip] = target[g0 : g0 + SEAM_LEN_K]
        goal_flat[clip] = _goal_flat(goal_seam[clip]).to(runner.device)
    return walk_sample, target_samples, goal_flat, goal_seam


def _capture_hidden_by_clip(
    runner: Any,
    *,
    npz_root: Path,
    target_states: Mapping[str, np.ndarray],
    context_len: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if runner.model is None:
        raise RuntimeError("runner model is not initialized")
    fullseq: Dict[str, np.ndarray] = {}
    context: Dict[str, np.ndarray] = {}
    runner.model.eval()
    for clip in TURN_CLIPS:
        T = int(target_states[clip].shape[0])
        ds = runner._build_dataset(npz_root / f"{clip}.npz", seq_len=T)
        runner._ensure_model_ready(ds)
        sample = freerun._build_full_cycle_sample(ds, ds.clips[0], seq_len=T)
        fullseq[clip] = _capture_fullseq_hidden(runner.model, sample)
        context[clip] = _capture_context_teacher_hidden(runner.model, sample, context_len=int(context_len))
    return fullseq, context


def _diag_dict(diag: Mapping[str, Any]) -> Dict[str, Any]:
    return {clip: asdict(row) for clip, row in diag.items()}


def _anchor_metadata(anchors: Mapping[str, Any], diag: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for clip, anchor in anchors.items():
        d = diag.get(clip)
        out[clip] = {
            "centroid_shape": [int(anchor.centroid.shape[0])],
            "radius": float(anchor.radius),
            "clip_spread": float(anchor.clip_spread),
            "diffuseness": float(anchor.diffuseness),
            "well_defined": bool(anchor.well_defined),
            "radius_degenerate": bool(anchor.radius_degenerate),
            "same_source_diagnostics": asdict(d) if d is not None else None,
        }
    return out


def _summarize_rollouts_for_anchor(
    rollouts: Sequence[Any],
    *,
    anchor: Any,
    self_reach_abs_cos: float,
    reach_available: bool,
    target_cond_dir: np.ndarray,
    goal_seam: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
    conv_norm_thr: float,
    k_values: Sequence[float],
) -> Dict[str, Any]:
    min_norms: List[float] = []
    abs_cos: List[float] = []
    states: List[Mapping[str, Any]] = []
    yaws: List[Mapping[str, Any]] = []
    for roll in rollouts:
        hidden_np = roll.hidden_pre.detach().cpu().numpy()
        min_norms.append(float(anchor.min_norm(hidden_np)))
        abs_cos.append(float(anchor.min_abs_cos(hidden_np)))
        state, yaw = _rollout_state_and_yaw(
            roll,
            target_cond_dir=target_cond_dir,
            goal_seam=goal_seam,
            std=std,
            thr=thr,
        )
        states.append(state)
        yaws.append(yaw)
    radius = summarize_reach_rate(min_norms, conv_norm_thr)
    self_gate = summarize_absolute_self_reach(
        abs_cos,
        self_reach_abs_cos=float(self_reach_abs_cos),
        k_values=k_values,
    )
    out: Dict[str, Any] = {
        "n": int(len(rollouts)),
        "reach_available": bool(reach_available),
        "reach_rate_radius": radius["reach_rate"],
        "reach_min_norm_mean": radius["reach_min_norm_mean"],
        "reach_min_norm_median": radius["reach_min_norm_median"],
        "reach_min_norm_min": radius["reach_min_norm_min"],
        "reach_abs_cos_mean": float(np.mean(abs_cos)) if abs_cos else float("nan"),
        "reach_abs_cos_min": float(np.min(abs_cos)) if abs_cos else float("nan"),
        "self_reach_gate": self_gate,
        "self_reach_rate_k3": self_gate["rate_by_k"].get("k=3", float("nan")),
        "pop_safe_rate": _mean_bool(states, "pop_safe"),
        "best_pose_d_mean": _metric_mean(states, "best_pose_d"),
        "best_pose_d_min": _metric_min(states, "best_pose_d"),
        "best_pose_d_max": _metric_max(states, "best_pose_d"),
        "mean_pop": _metric_mean(states, "pop"),
    }
    for key in (
        "realized_final_rad",
        "target_final_rad",
        "command_final_rad",
        "final_abs_err_rad",
        "heading_mae_rad",
        "yaw_rate_mae_rad_s",
        "corr",
        "valid_speed_fraction",
    ):
        out[f"yaw_{key}_mean"] = _metric_mean(yaws, key)
    out["yaw_corr"] = out["yaw_corr_mean"]
    out["heading_mae_rad"] = out["yaw_heading_mae_rad_mean"]
    return out


def _recorded_positive_row(
    *,
    clip: str,
    anchor: Any,
    self_reach_abs_cos: float,
    fullseq_hidden: np.ndarray,
    target_states: Mapping[str, np.ndarray],
    goal_seam: Mapping[str, np.ndarray],
    npz_root: Path,
    std: np.ndarray,
    thr: GateThresholds,
    k_values: Sequence[float],
) -> Dict[str, Any]:
    state = np.asarray(target_states[clip], dtype=np.float64)
    outcome = evaluate_rollout_state_space(state, goal_seam[clip], std, thr)
    root_vel, cond_dir = _load_raw_turn_motion(npz_root, clip)
    m = int(min(len(root_vel), len(cond_dir), state.shape[0]))
    yaw = _yaw_alignment(root_vel[:m], cond_dir[:m], cond_dir[:m], fps=FPS)
    abs_cos = float(anchor.min_abs_cos(fullseq_hidden))
    self_gate = summarize_absolute_self_reach([abs_cos], self_reach_abs_cos=self_reach_abs_cos, k_values=k_values)
    row: Dict[str, Any] = {
        "n": 1,
        "reach_available": bool(anchor.well_defined),
        "reach_abs_cos_min": abs_cos,
        "self_reach_gate": self_gate,
        "self_reach_rate_k3": self_gate["rate_by_k"].get("k=3", float("nan")),
        "pop_safe_rate": 1.0 if bool(outcome["pop_safe"]) else 0.0,
        "best_pose_d_mean": float(outcome["best_pose_d"]),
        "best_pose_d_min": float(outcome["best_pose_d"]),
        "best_pose_d_max": float(outcome["best_pose_d"]),
        "mean_pop": float(outcome["pop"]),
        "yaw_corr": float(yaw["corr"]),
        "heading_mae_rad": float(yaw["heading_mae_rad"]),
        "yaw_corr_mean": float(yaw["corr"]),
        "yaw_heading_mae_rad_mean": float(yaw["heading_mae_rad"]),
        "yaw_realized_final_rad_mean": float(yaw["realized_final_rad"]),
        "yaw_target_final_rad_mean": float(yaw["target_final_rad"]),
        "baseline_self_reach_rate": 0.0,
    }
    return row


def _run_rollout_sets(
    *,
    runner: Any,
    walk_sample: Mapping[str, torch.Tensor],
    target_samples: Mapping[str, Mapping[str, torch.Tensor]],
    goal_head: Optional[Any],
    goal_flat: Mapping[str, torch.Tensor],
    goal_config: Mapping[str, Any],
    start_phases: Sequence[int],
    horizon: int,
    context_len: int,
) -> Dict[str, Dict[str, List[Any]]]:
    out: Dict[str, Dict[str, List[Any]]] = {
        clip: {"pinned_goal": [], "free_goal": [], "free_no_goal": [], "pinned_no_goal": []}
        for clip in TURN_CLIPS
    }
    with torch.no_grad():
        for clip in TURN_CLIPS:
            delta = goal_head(goal_flat[clip]).detach() if goal_head is not None else None
            for phase in start_phases:
                out[clip]["pinned_no_goal"].append(
                    _run_context_ar_rollout(
                        runner,
                        walk_sample,
                        phase=int(phase),
                        horizon=int(horizon),
                        context_len=int(context_len),
                        delta=None,
                        capture_grad=False,
                    )
                )
                out[clip]["free_no_goal"].append(
                    _run_target_cond_self_contact_rollout(
                        runner,
                        walk_sample,
                        target_samples[clip],
                        phase=int(phase),
                        horizon=int(horizon),
                        context_len=int(context_len),
                        delta=None,
                    )[0]
                )
                if delta is not None:
                    out[clip]["pinned_goal"].append(
                        _run_context_ar_rollout(
                            runner,
                            walk_sample,
                            phase=int(phase),
                            horizon=int(horizon),
                            context_len=int(context_len),
                            delta=delta,
                            injection_targets=str(goal_config["injection_targets"]),
                            injection_mode=str(goal_config["mode"]),
                            capture_grad=False,
                        )
                    )
                    out[clip]["free_goal"].append(
                        _run_target_cond_self_contact_rollout(
                            runner,
                            walk_sample,
                            target_samples[clip],
                            phase=int(phase),
                            horizon=int(horizon),
                            context_len=int(context_len),
                            delta=delta,
                            injection_targets=str(goal_config["injection_targets"]),
                            injection_mode=str(goal_config["mode"]),
                        )[0]
                    )
    return out


def _summarize_object(
    *,
    object_name: str,
    anchor_name: str,
    anchors: Mapping[str, Any],
    anchor_diag: Mapping[str, Any],
    rollouts: Mapping[str, Mapping[str, Sequence[Any]]],
    target_samples: Mapping[str, Mapping[str, torch.Tensor]],
    goal_seam: Mapping[str, np.ndarray],
    target_states: Mapping[str, np.ndarray],
    npz_root: Path,
    std: np.ndarray,
    thr: GateThresholds,
    conv_norm_thr: float,
    k_values: Sequence[float],
    candidate_key: str,
    baseline_keys: Sequence[str],
    recorded_hidden: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    per_clip: Dict[str, Any] = {}
    candidate_rows: Dict[str, Mapping[str, object]] = {}
    baseline_tables: List[Dict[str, Mapping[str, object]]] = [dict() for _ in baseline_keys]
    recorded_rows: Dict[str, Mapping[str, object]] = {}

    for clip in TURN_CLIPS:
        diag = anchor_diag[clip]
        any_rollout = next((rows[0] for rows in rollouts[clip].values() if rows), None)
        target_horizon = int(any_rollout.out_raw.shape[0]) if any_rollout is not None else 1
        target_cond_dir = _target_cond_dir_from_sample(target_samples[clip], target_horizon)
        row_by_kind: Dict[str, Any] = {}
        for kind, rows in rollouts[clip].items():
            if not rows:
                continue
            row_by_kind[kind] = _summarize_rollouts_for_anchor(
                rows,
                anchor=anchors[clip],
                self_reach_abs_cos=float(diag.self_reach_abs_cos),
                reach_available=bool(diag.reach_available),
                target_cond_dir=target_cond_dir,
                goal_seam=goal_seam[clip],
                std=std,
                thr=thr,
                conv_norm_thr=conv_norm_thr,
                k_values=k_values,
            )
        recorded = _recorded_positive_row(
            clip=clip,
            anchor=anchors[clip],
            self_reach_abs_cos=float(diag.self_reach_abs_cos),
            fullseq_hidden=recorded_hidden[clip],
            target_states=target_states,
            goal_seam=goal_seam,
            npz_root=npz_root,
            std=std,
            thr=thr,
            k_values=k_values,
        )
        row_by_kind["recorded_turn_positive_control"] = recorded
        per_clip[clip] = row_by_kind
        candidate_rows[clip] = row_by_kind[candidate_key]
        for i, key in enumerate(baseline_keys):
            if key in row_by_kind:
                baseline_tables[i][clip] = row_by_kind[key]
        recorded_rows[clip] = recorded

    joint = joint_action_binding_gate_decision(
        candidate_rows,
        baseline_metrics=baseline_tables,
        k_label="k=3",
        min_reach_lift=DEFAULT_SELF_REACH_RATE_LIFT,
        tau_yaw_rad=DEFAULT_YAW_MAE_TAU_RAD,
        pose_degradation_tol=DEFAULT_POSE_DEGRADATION_TOL,
    )
    recorded_joint = joint_action_binding_gate_decision(
        recorded_rows,
        baseline_metrics=(),
        k_label="k=3",
        min_reach_lift=DEFAULT_SELF_REACH_RATE_LIFT,
        tau_yaw_rad=DEFAULT_YAW_MAE_TAU_RAD,
        pose_degradation_tol=DEFAULT_POSE_DEGRADATION_TOL,
    )
    return {
        "object": object_name,
        "anchor_default": anchor_name,
        "candidate_key": candidate_key,
        "baseline_keys": list(baseline_keys),
        "per_clip": per_clip,
        "joint_gate_decision": asdict(joint),
        "recorded_turn_positive_control": {
            "per_clip": recorded_rows,
            "joint_gate_decision": asdict(recorded_joint),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="W1b gate migration eval (read-only).")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--phase2-state", type=Path, default=Path(DEFAULT_PHASE2_STATE))
    p.add_argument("--phase2-summary", type=Path, default=Path(DEFAULT_PHASE2_SUMMARY))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--goal-horizon", type=int, default=12)
    p.add_argument("--conv-norm-thr", type=float, default=DEFAULT_CONV_NORM_THR)
    p.add_argument("--end-window-k", type=int, default=DEFAULT_END_WINDOW_K)
    p.add_argument("--self-reach-k", type=float, default=DEFAULT_SELF_REACH_K)
    p.add_argument("--report-k", type=str, default=",".join(str(v) for v in DEFAULT_SELF_REACH_K_VALUES))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    npz_root = Path(args.npz_root)
    z_path = Path(args.z_features)
    phase2_state_path = Path(args.phase2_state)
    if not Path(args.checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")
    if not phase2_state_path.exists():
        raise FileNotFoundError(f"phase2 state not found: {phase2_state_path}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_inbetween_gate_migration_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = _parse_float_list(str(args.report_k))
    target_states = load_clip_states(z_path, npz_root, fps=FPS)
    std = StateNormalizer(dict(target_states)).std
    thr = GateThresholds()
    legacy_hidden = load_hidden_pre(z_path, LOCKED_CLIPS)
    legacy_anchors = build_hidden_pre_anchors(legacy_hidden, TURN_CLIPS, int(args.end_window_k))
    phase2_state = _load_phase2_state(phase2_state_path)
    phase2_summary = _load_json_if_present(Path(args.phase2_summary))

    runner = freerun.FreeRunCycleRunner(_runner_args(args))
    walk_sample, target_samples, goal_flat, goal_seam = _build_samples(
        runner,
        npz_root=npz_root,
        target_states=target_states,
        goal_horizon=int(args.goal_horizon),
    )
    walk_T = int(walk_sample["motion"].shape[0])
    start_phases = [int(v) for v in phase2_state.get("start_phases", [])] if isinstance(phase2_state.get("start_phases"), Sequence) else []
    if not start_phases:
        from train.action_handoff_inbetween_cond_probe import select_start_phases

        start_phases = select_start_phases(walk_T, int(args.n_starts))

    base_fullseq, base_context = _capture_hidden_by_clip(
        runner,
        npz_root=npz_root,
        target_states=target_states,
        context_len=int(args.context_len),
    )
    base_anchors, base_diag = build_same_source_hidden_pre_anchors(
        base_fullseq,
        self_check_hidden_by_clip=base_context,
        turn_clips=TURN_CLIPS,
        end_window_k=int(args.end_window_k),
        self_reach_k=float(args.self_reach_k),
    )
    base_rollouts = _run_rollout_sets(
        runner=runner,
        walk_sample=walk_sample,
        target_samples=target_samples,
        goal_head=None,
        goal_flat=goal_flat,
        goal_config={"injection_targets": "shared_encoder.1", "mode": "additive"},
        start_phases=start_phases,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
    )
    base_eval = _summarize_object(
        object_name="frozen_base_no_goal",
        anchor_name="frozen_base_same_source",
        anchors=base_anchors,
        anchor_diag=base_diag,
        rollouts=base_rollouts,
        target_samples=target_samples,
        goal_seam=goal_seam,
        target_states=target_states,
        npz_root=npz_root,
        std=std,
        thr=thr,
        conv_norm_thr=float(args.conv_norm_thr),
        k_values=k_values,
        candidate_key="free_no_goal",
        baseline_keys=("pinned_no_goal",),
        recorded_hidden=base_fullseq,
    )

    phase2_load = _apply_phase2_model_state(runner, phase2_state)
    goal_head = _build_goal_head_from_phase2_state(
        phase2_state,
        device=runner.device,
        fallback_goal_flat_dim=int(SEAM_LEN_K * goal_seam[TURN_CLIPS[0]].shape[1]),
    )
    goal_config = _goal_config_from_phase2_state(phase2_state, args)
    trained_fullseq, trained_context = _capture_hidden_by_clip(
        runner,
        npz_root=npz_root,
        target_states=target_states,
        context_len=int(args.context_len),
    )
    trained_anchors, trained_diag = build_same_source_hidden_pre_anchors(
        trained_fullseq,
        self_check_hidden_by_clip=trained_context,
        turn_clips=TURN_CLIPS,
        end_window_k=int(args.end_window_k),
        self_reach_k=float(args.self_reach_k),
    )
    trained_rollouts = _run_rollout_sets(
        runner=runner,
        walk_sample=walk_sample,
        target_samples=target_samples,
        goal_head=goal_head,
        goal_flat=goal_flat,
        goal_config=goal_config,
        start_phases=start_phases,
        horizon=int(args.horizon),
        context_len=int(args.context_len),
    )
    trained_eval = _summarize_object(
        object_name="phase2_trained_goal",
        anchor_name="phase2_trained_same_source",
        anchors=trained_anchors,
        anchor_diag=trained_diag,
        rollouts=trained_rollouts,
        target_samples=target_samples,
        goal_seam=goal_seam,
        target_states=target_states,
        npz_root=npz_root,
        std=std,
        thr=thr,
        conv_norm_thr=float(args.conv_norm_thr),
        k_values=k_values,
        candidate_key="free_goal",
        baseline_keys=("free_no_goal", "pinned_goal"),
        recorded_hidden=trained_fullseq,
    )

    # G-C: score the same trained rollouts under the stale legacy anchor口径.
    legacy_diag_map = {}
    for clip in TURN_CLIPS:
        floor = float(legacy_anchors[clip].min_abs_cos(legacy_hidden[clip]))
        check_abs = float(legacy_anchors[clip].min_abs_cos(trained_fullseq[clip]))
        threshold = float(args.self_reach_k) * max(floor, 1e-12)
        legacy_diag_map[clip] = type(
            "_Diag",
            (),
            {
                "self_reach_abs_cos": floor,
                "reach_available": bool(legacy_anchors[clip].well_defined and check_abs <= threshold),
            },
        )()
    trained_on_legacy_anchor = _summarize_object(
        object_name="phase2_trained_goal",
        anchor_name="legacy_saved_z_features_anchor_diagnostic_only",
        anchors=legacy_anchors,
        anchor_diag=legacy_diag_map,
        rollouts=trained_rollouts,
        target_samples=target_samples,
        goal_seam=goal_seam,
        target_states=target_states,
        npz_root=npz_root,
        std=std,
        thr=thr,
        conv_norm_thr=float(args.conv_norm_thr),
        k_values=k_values,
        candidate_key="free_goal",
        baseline_keys=("free_no_goal", "pinned_goal"),
        recorded_hidden=trained_fullseq,
    )

    same_source_consistency: Dict[str, Any] = {}
    for clip in TURN_CLIPS:
        same = trained_eval["per_clip"][clip]["free_goal"]
        legacy = trained_on_legacy_anchor["per_clip"][clip]["free_goal"]
        same_source_consistency[clip] = {
            "trained_fullseq_relerr_vs_legacy_saved": calibration_relerr(trained_fullseq[clip], legacy_hidden[clip]),
            "trained_fullseq_relerr_vs_frozen_base_capture": calibration_relerr(trained_fullseq[clip], base_fullseq[clip]),
            "legacy_anchor_free_goal_k3": legacy["self_reach_gate"]["rate_by_k"].get("k=3"),
            "trained_same_source_free_goal_k3": same["self_reach_gate"]["rate_by_k"].get("k=3"),
            "legacy_anchor_free_goal_abs_mean": legacy["reach_abs_cos_mean"],
            "trained_same_source_free_goal_abs_mean": same["reach_abs_cos_mean"],
            "default_after_migration": "phase2_trained_same_source",
        }

    ltr = trained_eval["per_clip"]["Walk_L_To_R"]["free_goal"]
    recorded_ltr = trained_eval["recorded_turn_positive_control"]["per_clip"]["Walk_L_To_R"]
    g_a = {
        "passed": bool(
            not trained_eval["joint_gate_decision"]["per_clip_pass"].get("Walk_L_To_R", True)
            and float(ltr["self_reach_gate"]["rate_by_k"].get("k=3", float("nan"))) == 0.0
            and float(ltr["yaw_corr"]) < 0.0
            and float(ltr["pop_safe_rate"]) == 0.0
        ),
        "walk_l_to_r": {
            "self_reach_k3": ltr["self_reach_gate"]["rate_by_k"].get("k=3"),
            "yaw_corr": ltr["yaw_corr"],
            "heading_mae_rad": ltr["heading_mae_rad"],
            "pop_safe_rate": ltr["pop_safe_rate"],
            "joint_pass": trained_eval["joint_gate_decision"]["per_clip_pass"].get("Walk_L_To_R"),
        },
    }
    g_b = {
        "passed": bool(trained_eval["recorded_turn_positive_control"]["joint_gate_decision"]["all_pass"]),
        "walk_l_to_r": {
            "self_reach_k3": recorded_ltr["self_reach_gate"]["rate_by_k"].get("k=3"),
            "yaw_corr": recorded_ltr["yaw_corr"],
            "heading_mae_rad": recorded_ltr["heading_mae_rad"],
            "pop_safe_rate": recorded_ltr["pop_safe_rate"],
            "best_pose_d": recorded_ltr["best_pose_d_mean"],
            "joint_pass": trained_eval["recorded_turn_positive_control"]["joint_gate_decision"]["per_clip_pass"].get("Walk_L_To_R"),
        },
    }
    g_c = {
        "passed": True,
        "per_clip": same_source_consistency,
        "note": "default migrated gate uses phase2_trained_same_source; legacy saved/frozen anchors are diagnostic only.",
    }
    legacy_ltr_pinned = trained_on_legacy_anchor["per_clip"]["Walk_L_To_R"]["pinned_goal"]

    summary: Dict[str, Any] = {
        "task": "W1b action-handoff in-betweening gate migration eval",
        "no_training": True,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "phase2_state": str(phase2_state_path.resolve()),
        "phase2_summary_loaded": phase2_summary is not None,
        "phase2_load": phase2_load,
        "z_features_path": str(z_path.resolve()),
        "npz_root": str(npz_root.resolve()),
        "config": {
            "context_len": int(args.context_len),
            "horizon": int(args.horizon),
            "n_starts": int(len(start_phases)),
            "start_phases": [int(v) for v in start_phases],
            "conv_norm_thr_radius_diagnostic": float(args.conv_norm_thr),
            "self_reach_k_default": float(args.self_reach_k),
            "reported_k_values": [float(v) for v in k_values],
            "joint_gate_thresholds": {
                "min_reach_lift": DEFAULT_SELF_REACH_RATE_LIFT,
                "tau_yaw_rad": DEFAULT_YAW_MAE_TAU_RAD,
                "tau_yaw_deg": float(np.degrees(DEFAULT_YAW_MAE_TAU_RAD)),
                "pose_degradation_tol": DEFAULT_POSE_DEGRADATION_TOL,
                "tau_yaw_basis": "PROVISIONAL: recorded turns are ~0 deg MAE while W1a PHASE2 L_R free is ~39.6 deg.",
            },
        },
        "anchors": {
            "legacy_saved_z_features_anchor_diagnostic_only": {
                clip: {
                    "radius": float(legacy_anchors[clip].radius),
                    "well_defined": bool(legacy_anchors[clip].well_defined),
                    "self_reach_abs_cos_saved": float(legacy_anchors[clip].min_abs_cos(legacy_hidden[clip])),
                }
                for clip in TURN_CLIPS
            },
            "frozen_base_same_source": _anchor_metadata(base_anchors, base_diag),
            "phase2_trained_same_source": _anchor_metadata(trained_anchors, trained_diag),
        },
        "evaluated_objects": {
            "frozen_base_no_goal": base_eval,
            "phase2_trained_goal": trained_eval,
            "phase2_trained_goal_on_legacy_anchor_diagnostic_only": trained_on_legacy_anchor,
        },
        "acceptance": {
            "G_A_negative_reject_phase2_l_r_artifact": g_a,
            "G_B_positive_recorded_turn_accepts": g_b,
            "G_C_same_source_consistency": g_c,
            "G_D_scope": {
                "no_training": True,
                "training_logic_modified": False,
                "entered_b4_or_seam": False,
                "old_radius_fields_retained_as_diagnostics": True,
            },
        },
        "walk_l_to_r_side_by_side": {
            "legacy_old_radius_pinned_artifact": {
                "reach_rate_radius": legacy_ltr_pinned["reach_rate_radius"],
                "self_reach_k3": legacy_ltr_pinned["self_reach_gate"]["rate_by_k"].get("k=3"),
                "yaw_corr": legacy_ltr_pinned["yaw_corr"],
                "heading_mae_rad": legacy_ltr_pinned["heading_mae_rad"],
                "pop_safe_rate": legacy_ltr_pinned["pop_safe_rate"],
                "joint_pass": "diagnostic_only_old_gate",
            },
            "migrated_same_source_free_candidate": g_a["walk_l_to_r"],
            "recorded_turn_positive_control": g_b["walk_l_to_r"],
        },
    }

    json_path = out_dir / "gate_migration_eval_summary.json"
    md_path = out_dir / "gate_migration_eval_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# W1b Gate Migration Eval")
    lines.append("")
    lines.append("> Read-only evaluator. No training was run.")
    lines.append("")
    lines.append("## Acceptance")
    lines.append(
        f"- G-A PHASE2 L_R rejected: **{g_a['passed']}** "
        f"(k3={_fmt(g_a['walk_l_to_r']['self_reach_k3'],2)}, "
        f"yaw_corr={_fmt(g_a['walk_l_to_r']['yaw_corr'],2)}, "
        f"heading_MAE={_fmt(np.degrees(g_a['walk_l_to_r']['heading_mae_rad']),1)} deg, "
        f"pop_safe={_fmt(g_a['walk_l_to_r']['pop_safe_rate'],2)})"
    )
    lines.append(
        f"- G-B recorded turn positive control: **{g_b['passed']}** "
        f"(L_R yaw_corr={_fmt(g_b['walk_l_to_r']['yaw_corr'],2)}, "
        f"heading_MAE={_fmt(np.degrees(g_b['walk_l_to_r']['heading_mae_rad']),6)} deg, "
        f"pop_safe={_fmt(g_b['walk_l_to_r']['pop_safe_rate'],2)}, "
        f"best_pose_d={_fmt(g_b['walk_l_to_r']['best_pose_d'],6)})"
    )
    lines.append("- G-C same-source default: **phase2_trained_same_source**; legacy/frozen anchors are diagnostic only.")
    lines.append("")
    lines.append("## PHASE2 L_R Side-by-Side")
    lines.append("|口径|old radius reach|self k3|yaw corr|heading MAE deg|pop_safe|joint pass|")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    phase2_ltr = trained_eval["per_clip"]["Walk_L_To_R"]
    legacy_phase2_ltr = trained_on_legacy_anchor["per_clip"]["Walk_L_To_R"]
    for label, source, key in (
        ("legacy pinned radius artifact", legacy_phase2_ltr, "pinned_goal"),
        ("same-source pinned diagnostic", phase2_ltr, "pinned_goal"),
        ("free goal candidate", phase2_ltr, "free_goal"),
        ("free no-goal baseline", phase2_ltr, "free_no_goal"),
        ("recorded positive", phase2_ltr, "recorded_turn_positive_control"),
    ):
        r = source[key]
        joint_pass = (
            trained_eval["joint_gate_decision"]["per_clip_pass"].get("Walk_L_To_R")
            if key == "free_goal"
            else trained_eval["recorded_turn_positive_control"]["joint_gate_decision"]["per_clip_pass"].get("Walk_L_To_R")
            if key == "recorded_turn_positive_control"
            else "diagnostic"
        )
        lines.append(
            f"| {label} | {_fmt(r.get('reach_rate_radius', float('nan')),2)} | "
            f"{_fmt(r['self_reach_gate']['rate_by_k'].get('k=3'),2)} | "
            f"{_fmt(r.get('yaw_corr', r.get('yaw_corr_mean', float('nan'))),2)} | "
            f"{_fmt(np.degrees(r.get('heading_mae_rad', r.get('yaw_heading_mae_rad_mean', float('nan')))),1)} | "
            f"{_fmt(r['pop_safe_rate'],2)} | {joint_pass} |"
        )
    lines.append("")
    lines.append("## G-C Anchor口径")
    lines.append("|target|relerr trained vs saved|legacy k3|same-source k3|legacy abs mean|same-source abs mean|")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for clip in TURN_CLIPS:
        r = same_source_consistency[clip]
        lines.append(
            f"| {clip} | {_fmt(r['trained_fullseq_relerr_vs_legacy_saved'],3)} | "
            f"{_fmt(r['legacy_anchor_free_goal_k3'],2)} | {_fmt(r['trained_same_source_free_goal_k3'],2)} | "
            f"{_fmt(r['legacy_anchor_free_goal_abs_mean'],6)} | {_fmt(r['trained_same_source_free_goal_abs_mean'],6)} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- JSON: `{json_path.resolve()}`")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(
        "[W1b] "
        f"G-A={g_a['passed']} "
        f"G-B={g_b['passed']} "
        f"L_R_free_k3={_fmt(g_a['walk_l_to_r']['self_reach_k3'],2)} "
        f"L_R_yaw_corr={_fmt(g_a['walk_l_to_r']['yaw_corr'],2)}"
    )


if __name__ == "__main__":
    main()
