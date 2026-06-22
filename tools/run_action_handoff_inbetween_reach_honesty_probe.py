#!/usr/bin/env python3
"""Reach honesty probe for action-handoff goal-conditioned in-betweening W0.

This tool is read-only: it does not train, mutate checkpoints, or change the existing
PHASE2 training path. It audits three measurement questions:

A. radius-normalized reach vs absolute self-reach-relative reach;
B. Walk_F-pinned rollout reach vs a target-cond, self-carried-contact rollout;
C. motion-space yaw/heading realization vs the target turn.

When ``--phase2-state`` is provided, B/C are run on the exact saved PHASE2 fine-tuned base
plus goal_head state. Without it, the older no-goal base-checkpoint B/C path remains
available and is labelled as a limitation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train.validate.run_freerun_cycles as freerun  # noqa: E402
from train import rollout_kernel as _rollout_kernel  # noqa: E402
from train.action_handoff_inbetween_cond_probe import rollout_to_egocentric, select_start_phases  # noqa: E402
from train.action_handoff_inbetween_goal_injection import (  # noqa: E402
    GoalHead,
    calibration_records_all_pass,
    calibration_relerr,
    context_window_indices,
    register_goal_injection_pre_temporal,
)
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer, evaluate_rollout_state_space  # noqa: E402
from train.action_handoff_inbetween_reach import (  # noqa: E402
    DEFAULT_CONV_NORM_THR,
    DEFAULT_END_WINDOW_K,
    LOCKED_CLIPS,
    TURN_CLIPS,
    build_hidden_pre_anchors,
    load_hidden_pre,
)
from train.data.action_handoff_inbetween import (  # noqa: E402
    RAW_COND_DIR_SLICE,
    SEAM_LEN_K,
    WALK_F,
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
    RolloutCapture,
    _append_window,
    _capture_context_teacher_hidden,
    _capture_fullseq_hidden,
    _fmt,
    _goal_flat,
    _next_pose_hist_norm,
    _phase_take,
    _run_context_ar_rollout,
    _runner_args,
)

DEFAULT_PHASE2_SUMMARY = (
    "debug_output/_tmp_action_handoff_inbetween_phase2_guarded_finetune_20260530_tail500/"
    "phase2_guarded_finetune_summary.json"
)
DEFAULT_REACH_ANCHOR_SUMMARY = (
    "debug_output/_tmp_action_handoff_inbetween_reach_anchor_check_20260530/"
    "reach_anchor_check_summary.json"
)
DEFAULT_SELF_REACH_K = (2.0, 3.0, 5.0)
ROOT_VEL_OUT_SLICE = slice(276, 278)


@dataclass
class FreeRolloutMeta:
    """Small trace for the target-cond self-contact rollout."""

    future_contact_source: str
    seed_contact_source: str
    cond_source: str
    generated_contact_fraction: float
    fallback_contact_steps: int


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
    vals: List[float] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        s = tok.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("empty float list")
    return vals


def _target_step_index(step: int, target_len: int) -> int:
    if int(target_len) <= 0:
        raise ValueError("target_len must be positive")
    return int(min(max(0, int(step)), int(target_len) - 1))


def _last_contact_from_ret(ret: Mapping[str, Any], fallback: torch.Tensor) -> Tuple[torch.Tensor, str]:
    for key in ("contacts_plan", "contacts_meas"):
        val = ret.get(key, None)
        if not torch.is_tensor(val):
            continue
        if val.dim() == 3:
            step = val[0, -1]
        elif val.dim() == 2:
            step = val[0]
        elif val.dim() == 1:
            step = val
        else:
            continue
        return step.detach().clamp(0.0, 1.0), key
    return fallback.detach(), "fallback_previous_contact"


def _run_target_cond_self_contact_rollout(
    runner: Any,
    walk_sample: Dict[str, torch.Tensor],
    target_sample: Dict[str, torch.Tensor],
    *,
    phase: int,
    horizon: int,
    context_len: int,
    delta: Optional[torch.Tensor] = None,
    injection_targets: str = "shared_encoder.1",
    injection_mode: str = "additive",
) -> Tuple[RolloutCapture, FreeRolloutMeta]:
    """AR rollout with Walk_F pose seed, target turn cond, and generated contact carry.

    Seed context motion/pose_history are from Walk_F at the arbitrary phase. Future cond comes
    from the target turn trajectory (held at the target's last frame past its length). Future
    contacts are carried from the model's own ``contacts_plan`` when available. This is a
    diagnostic copy of ``_run_context_ar_rollout``; the original pinned function is untouched.
    """

    model = runner.model
    trainer = runner.trainer
    device = runner.device
    walk_T = int(walk_sample["motion"].shape[0])
    target_T = int(target_sample["motion"].shape[0])
    C = int(context_len)

    walk_idx0 = context_window_indices(int(phase), C, walk_T, mode="wrap")
    target_idx0 = context_window_indices(0, C, target_T, mode="edge")
    motion_hist = _phase_take(walk_sample, "motion", walk_idx0, device)
    cond_hist = _phase_take(target_sample, "cond_in", target_idx0, device)
    cond_raw_hist = _phase_take(target_sample, "cond_tgt_raw", target_idx0, device)
    contacts_hist = _phase_take(walk_sample, "contacts", walk_idx0, device)
    pose_hist = _phase_take(walk_sample, "pose_hist", walk_idx0, device)

    free_carry_cfg = _rollout_kernel.resolve_free_carry_runtime_config(trainer)
    if bool(getattr(trainer, "use_freerun_state_sync", False)) and isinstance(free_carry_cfg.angvel_x_slice, slice):
        angvel_hist = motion_hist[:, free_carry_cfg.angvel_x_slice]
    else:
        angvel_hist = _phase_take(walk_sample, "angvel", walk_idx0, device)

    y_raw_prev = trainer._denorm(walk_sample["gt_motion"][int(phase) % walk_T].to(device=device).unsqueeze(0))
    motion_raw_last = trainer.normalizer.denorm_x(motion_hist[-1:].detach())

    hidden_steps: List[torch.Tensor] = []
    raw_steps: List[torch.Tensor] = []
    cond_dir_steps: List[torch.Tensor] = []
    contact_steps: List[torch.Tensor] = []
    contact_source_counts: Dict[str, int] = {}

    hook = (
        register_goal_injection_pre_temporal(
            model,
            delta,
            targets=str(injection_targets),
            mode=str(injection_mode),
        )
        if delta is not None
        else None
    )
    try:
        for step in range(int(horizon)):
            cap: Dict[str, torch.Tensor] = {}

            def _pre(_m: Any, inp: Tuple[Any, ...]) -> None:
                cap["h"] = inp[0][:, -1]

            hcap = model._pasa_lnq.register_forward_pre_hook(_pre)
            try:
                ret = model(
                    motion_hist.unsqueeze(0),
                    cond_hist.unsqueeze(0),
                    contacts=contacts_hist.unsqueeze(0),
                    angvel=angvel_hist.unsqueeze(0),
                    pose_history=pose_hist.unsqueeze(0),
                )
            finally:
                hcap.remove()

            out = ret["out"]
            delta_norm = out[:, -1] if out.dim() == 3 else out
            try:
                y_used_raw = trainer._compose_delta_to_raw(y_raw_prev, delta_norm)
            except Exception:
                y_used_raw = trainer._denorm(delta_norm)

            hidden_steps.append(cap["h"][0] if cap["h"].dim() == 2 else cap["h"].reshape(-1, cap["h"].shape[-1])[-1])
            raw_steps.append(y_used_raw[0])
            target_idx = _target_step_index(step, target_T)
            cond_dir_steps.append(
                target_sample["cond_tgt_raw"][target_idx, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]].to(device)
            )
            contact_step, contact_source = _last_contact_from_ret(ret, contacts_hist[-1])
            contact_source_counts[contact_source] = int(contact_source_counts.get(contact_source, 0)) + 1
            contact_steps.append(contact_step)

            next_idx = _target_step_index(step + 1, target_T)
            cond_next_raw = target_sample["cond_tgt_raw"][next_idx].to(device=device).unsqueeze(0)
            motion_raw_next = _rollout_kernel.apply_free_carry_raw(
                x_prev=motion_raw_last.detach(),
                y_next_raw=y_used_raw.detach(),
                cond_next_raw=cond_next_raw,
                rot6d_x_slice=free_carry_cfg.rot6d_x_slice,
                rot6d_y_slice=free_carry_cfg.rot6d_y_slice,
                angvel_x_slice=free_carry_cfg.angvel_x_slice,
                rootvel_x_slice=free_carry_cfg.rootvel_x_slice,
                rootpos_x_slice=free_carry_cfg.rootpos_x_slice,
                bone_hz=free_carry_cfg.bone_hz,
                columns=free_carry_cfg.columns,
            ).detach()
            motion_next = trainer._diag_norm_x(motion_raw_next)[0].detach()
            pose_hist_next = _next_pose_hist_norm(
                trainer,
                pose_hist[-1],
                y_used_raw,
                rot_slice=free_carry_cfg.rot6d_y_slice
                if isinstance(free_carry_cfg.rot6d_y_slice, slice)
                else slice(0, y_used_raw.shape[-1]),
            )

            motion_hist = _append_window(motion_hist, motion_next)
            cond_hist = _append_window(cond_hist, target_sample["cond_in"][next_idx].to(device))
            cond_raw_hist = _append_window(cond_raw_hist, target_sample["cond_tgt_raw"][next_idx].to(device))
            contacts_hist = _append_window(contacts_hist, contact_step)
            pose_hist = _append_window(pose_hist, pose_hist_next)
            if bool(getattr(trainer, "use_freerun_state_sync", False)) and isinstance(free_carry_cfg.angvel_x_slice, slice):
                angvel_hist = motion_hist[:, free_carry_cfg.angvel_x_slice]
            else:
                angvel_hist = _append_window(angvel_hist, target_sample["angvel"][next_idx].to(device))
            y_raw_prev = y_used_raw.detach()
            motion_raw_last = motion_raw_next
    finally:
        if hook is not None:
            hook.remove()

    hidden = torch.stack(hidden_steps, dim=0).detach()
    raw = torch.stack(raw_steps, dim=0).detach()
    cond_dir = torch.stack(cond_dir_steps, dim=0)
    contacts = torch.stack(contact_steps, dim=0)
    generated = int(contact_source_counts.get("contacts_plan", 0))
    fallback = int(contact_source_counts.get("fallback_previous_contact", 0))
    meta = FreeRolloutMeta(
        future_contact_source="contacts_plan_selfcarry_when_available",
        seed_contact_source="Walk_F_context_only",
        cond_source="target_turn_cond_trajectory_edge_held",
        generated_contact_fraction=float(generated / max(1, int(horizon))),
        fallback_contact_steps=fallback,
    )
    _ = cond_raw_hist  # kept for symmetry with the pinned rollout and future debugging.
    return RolloutCapture(hidden_pre=hidden, out_raw=raw, cond_dir_raw=cond_dir, contacts=contacts), meta


def _vector_heading_unwrapped(vec: np.ndarray, *, speed_eps: float = 1e-5) -> Tuple[np.ndarray, float]:
    arr = np.nan_to_num(np.asarray(vec, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected [T,2] vectors, got {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64), 0.0
    speed = np.linalg.norm(arr, axis=1)
    raw = np.arctan2(arr[:, 1], arr[:, 0])
    valid = speed > float(speed_eps)
    if not bool(np.any(valid)):
        return np.zeros((arr.shape[0],), dtype=np.float64), 0.0
    filled = raw.copy()
    first = int(np.argmax(valid))
    filled[:first] = raw[first]
    last = raw[first]
    for i in range(first, arr.shape[0]):
        if valid[i]:
            last = raw[i]
        else:
            filled[i] = last
    return np.unwrap(filled), float(np.mean(valid))


def _cond_heading_unwrapped(cond_dir: np.ndarray) -> np.ndarray:
    cd = np.nan_to_num(np.asarray(cond_dir, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if cd.ndim != 2 or cd.shape[1] != 2:
        raise ValueError(f"expected cond_dir [T,2], got {cd.shape}")
    if cd.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.unwrap(np.arctan2(cd[:, 1], cd[:, 0]))


def _yaw_rate_from_heading(heading: np.ndarray, fps: float) -> np.ndarray:
    h = np.asarray(heading, dtype=np.float64).reshape(-1)
    if h.size <= 1:
        return np.zeros_like(h)
    out = np.empty_like(h)
    out[1:] = np.diff(h) * float(fps)
    out[0] = out[1]
    return out


def _yaw_alignment(
    root_vel: np.ndarray,
    rollout_cond_dir: np.ndarray,
    target_cond_dir: np.ndarray,
    *,
    fps: float,
) -> Dict[str, Any]:
    m = int(min(len(root_vel), len(rollout_cond_dir), len(target_cond_dir)))
    if m < 1:
        return {
            "n": 0,
            "realized_final_rad": float("nan"),
            "target_final_rad": float("nan"),
            "command_final_rad": float("nan"),
            "final_abs_err_rad": float("nan"),
            "heading_mae_rad": float("nan"),
            "yaw_rate_mae_rad_s": float("nan"),
            "corr": float("nan"),
            "valid_speed_fraction": 0.0,
        }
    rv_heading, valid_frac = _vector_heading_unwrapped(root_vel[:m])
    cmd_heading = _cond_heading_unwrapped(rollout_cond_dir[:m])
    tgt_heading = _cond_heading_unwrapped(target_cond_dir[:m])
    rv_cum = rv_heading - rv_heading[0]
    cmd_cum = cmd_heading - cmd_heading[0]
    tgt_cum = tgt_heading - tgt_heading[0]
    diff = rv_cum - tgt_cum
    rv_rate = _yaw_rate_from_heading(rv_heading, fps)
    tgt_rate = _yaw_rate_from_heading(tgt_heading, fps)
    corr = float("nan")
    if m >= 2 and float(np.std(rv_cum)) > 1e-8 and float(np.std(tgt_cum)) > 1e-8:
        corr = float(np.corrcoef(rv_cum, tgt_cum)[0, 1])
    return {
        "n": int(m),
        "realized_final_rad": float(rv_cum[-1]),
        "target_final_rad": float(tgt_cum[-1]),
        "command_final_rad": float(cmd_cum[-1]),
        "final_abs_err_rad": float(abs(diff[-1])),
        "heading_mae_rad": float(np.mean(np.abs(diff))),
        "yaw_rate_mae_rad_s": float(np.mean(np.abs(rv_rate - tgt_rate))),
        "corr": corr,
        "valid_speed_fraction": float(valid_frac),
    }


def _mean_finite(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _k_label(k: float) -> str:
    return f"k={_fmt(k, 0) if float(k).is_integer() else _fmt(k, 2)}"


def _self_reach_rates(
    abs_cos: Sequence[float],
    *,
    self_abs_floor: float,
    k_values: Sequence[float],
) -> Dict[str, Any]:
    vals = np.asarray([float(v) for v in abs_cos], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    out: Dict[str, Any] = {
        "self_abs_floor": float(self_abs_floor),
        "n": int(vals.size),
        "finite_n": int(finite.size),
        "rate_by_k": {},
        "count_by_k": {},
        "threshold_abs_by_k": {},
        "margin_mean_by_k": {},
        "margin_min_by_k": {},
    }
    denom_floor = max(float(self_abs_floor), 1e-12)
    for k in k_values:
        label = _k_label(float(k))
        threshold = float(k) * denom_floor
        if finite.size:
            passed = finite <= threshold
            margins = finite / max(threshold, 1e-12)
            out["rate_by_k"][label] = float(np.mean(passed))
            out["count_by_k"][label] = int(np.sum(passed))
            out["margin_mean_by_k"][label] = float(np.mean(margins))
            out["margin_min_by_k"][label] = float(np.min(margins))
        else:
            out["rate_by_k"][label] = float("nan")
            out["count_by_k"][label] = 0
            out["margin_mean_by_k"][label] = float("nan")
            out["margin_min_by_k"][label] = float("nan")
        out["threshold_abs_by_k"][label] = threshold
    return out


def _rollout_state_and_yaw(
    roll: RolloutCapture,
    *,
    target_cond_dir: np.ndarray,
    goal_seam: np.ndarray,
    std: np.ndarray,
    thr: GateThresholds,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    out_np = roll.out_raw.detach().cpu().numpy()
    gen_rot6d = out_np[:, :276].reshape(out_np.shape[0], 46, 6)
    gen_rv = out_np[:, ROOT_VEL_OUT_SLICE]
    cond_dir_np = roll.cond_dir_raw.detach().cpu().numpy()
    contact_np = roll.contacts.detach().cpu().numpy()
    m = int(min(len(gen_rot6d), len(cond_dir_np), len(contact_np), len(target_cond_dir)))
    roll_raw = rollout_to_egocentric(gen_rot6d[:m], gen_rv[:m], cond_dir_np[:m], contact_np[:m], fps=FPS)
    return (
        evaluate_rollout_state_space(roll_raw, goal_seam, std, thr),
        _yaw_alignment(gen_rv[:m], cond_dir_np[:m], target_cond_dir[:m], fps=FPS),
    )


def _summarize_runtime_rows(
    *,
    min_norms: Sequence[float],
    abs_cos: Sequence[float],
    state_outcomes: Sequence[Mapping[str, Any]],
    yaw_rows: Sequence[Mapping[str, Any]],
    conv_norm_thr: float,
    self_abs_floor: Optional[float] = None,
    k_values: Sequence[float] = (),
) -> Dict[str, Any]:
    mn = np.asarray([float(v) for v in min_norms], dtype=np.float64)
    ac = np.asarray([float(v) for v in abs_cos], dtype=np.float64)
    n = int(mn.size)
    out: Dict[str, Any] = {
        "n": n,
        "reach_rate_radius": float(np.mean(mn <= float(conv_norm_thr))) if n else float("nan"),
        "reach_min_norm_mean": float(np.mean(mn)) if n else float("nan"),
        "reach_min_norm_median": float(np.median(mn)) if n else float("nan"),
        "reach_min_norm_min": float(np.min(mn)) if n else float("nan"),
        "reach_abs_cos_mean": float(np.mean(ac)) if ac.size else float("nan"),
        "reach_abs_cos_min": float(np.min(ac)) if ac.size else float("nan"),
    }
    if self_abs_floor is not None:
        out["self_reach_gate"] = _self_reach_rates(ac, self_abs_floor=float(self_abs_floor), k_values=k_values)
    if state_outcomes:
        best_pose = np.asarray([float(o.get("best_pose_d", float("nan"))) for o in state_outcomes], dtype=np.float64)
        out.update(
            {
                "pop_safe_rate": float(np.mean([bool(o.get("pop_safe", False)) for o in state_outcomes])),
                "mean_best_pose_d": float(np.mean(best_pose)),
                "best_pose_d_mean": float(np.mean(best_pose)),
                "best_pose_d_min": float(np.min(best_pose)),
                "best_pose_d_max": float(np.max(best_pose)),
                "mean_pop": float(np.mean([float(o.get("pop", float("nan"))) for o in state_outcomes])),
            }
        )
    else:
        out.update(
            {
                "pop_safe_rate": float("nan"),
                "mean_best_pose_d": float("nan"),
                "best_pose_d_mean": float("nan"),
                "best_pose_d_min": float("nan"),
                "best_pose_d_max": float("nan"),
                "mean_pop": float("nan"),
            }
        )
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
        out[f"yaw_{key}_mean"] = _mean_finite([float(r.get(key, float("nan"))) for r in yaw_rows])
    return out


def _target_cond_dir_from_sample(sample: Dict[str, torch.Tensor], horizon: int) -> np.ndarray:
    raw = sample["cond_tgt_raw"].detach().cpu().numpy()
    idx = [_target_step_index(i, raw.shape[0]) for i in range(int(horizon))]
    arr = raw[idx, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]]
    return np.asarray(arr, dtype=np.float64)


def _load_phase2_state(path: Path) -> Dict[str, Any]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"phase2 state must be a dict: {path}")
    if payload.get("kind") != "action_handoff_inbetween_phase2_trained_state":
        raise ValueError(f"unexpected phase2 state kind in {path}: {payload.get('kind')!r}")
    if "model_state_dict" not in payload or "goal_head_state_dict" not in payload:
        raise ValueError(f"phase2 state missing model_state_dict or goal_head_state_dict: {path}")
    return payload


def _build_goal_head_from_phase2_state(payload: Mapping[str, Any], *, device: torch.device, fallback_goal_flat_dim: int):
    cfg = payload.get("goal_head_config", {})
    if not isinstance(cfg, Mapping):
        raise ValueError("phase2 state goal_head_config must be a mapping")
    goal_flat_dim = int(cfg.get("goal_flat_dim", fallback_goal_flat_dim))
    hidden = int(cfg.get("hidden", 512))
    depth = int(cfg.get("depth", 2))
    mode = str(cfg.get("mode", "additive"))
    goal_head = GoalHead.build(
        goal_flat_dim=goal_flat_dim,
        hidden=hidden,
        depth=depth,
        init_scale=float(cfg.get("init_scale", 0.0)),
        mode=mode,
    ).to(device)
    goal_head.load_state_dict(payload["goal_head_state_dict"], strict=True)
    goal_head.eval()
    return goal_head


def _apply_phase2_model_state(runner: Any, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if runner.model is None:
        raise RuntimeError("runner model must be built before loading phase2 state")
    incompatible = runner.model.load_state_dict(payload["model_state_dict"], strict=True)
    runner.model.eval()
    return {
        "model_strict_load": True,
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }


def _goal_config_from_phase2_state(payload: Optional[Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = payload.get("goal_head_config", {}) if isinstance(payload, Mapping) else {}
    return {
        "injection_targets": str(cfg.get("injection_targets", getattr(args, "goal_injection_targets", "shared_encoder.1"))),
        "mode": str(cfg.get("mode", getattr(args, "goal_head_mode", "additive"))),
    }


def _run_trained_self_reach_calibration(
    runner: Any,
    *,
    npz_root: Path,
    saved_hidden: Mapping[str, np.ndarray],
    anchors: Mapping[str, Any],
    context_len: int,
    conv_norm_thr: float,
) -> Dict[str, Any]:
    if runner.model is None:
        raise RuntimeError("runner model must be ready before calibration")
    model = runner.model
    model.eval()
    records: Dict[str, Dict[str, Any]] = {}
    for clip in TURN_CLIPS:
        T = int(saved_hidden[clip].shape[0])
        ds = runner._build_dataset(npz_root / f"{clip}.npz", seq_len=T)
        runner._ensure_model_ready(ds)
        sample = freerun._build_full_cycle_sample(ds, ds.clips[0], seq_len=T)
        fullseq = _capture_fullseq_hidden(model, sample)
        context = _capture_context_teacher_hidden(model, sample, context_len=int(context_len))
        records[clip] = {
            "fullseq_shape": list(fullseq.shape),
            "fullseq_dtype": str(fullseq.dtype),
            "fullseq_device": "cpu",
            "fullseq_relerr_vs_saved": calibration_relerr(fullseq, saved_hidden[clip]),
            "fullseq_self_min_norm": float(anchors[clip].min_norm(fullseq)),
            "fullseq_self_abs_cos": float(anchors[clip].min_norm(fullseq) * anchors[clip].radius),
            "fullseq_self_reached": bool(anchors[clip].reached(fullseq, float(conv_norm_thr))),
            "context_window_shape": list(context.shape),
            "context_window_dtype": str(context.dtype),
            "context_window_device": "cpu",
            "context_relerr_vs_saved_fullseq": calibration_relerr(context, saved_hidden[clip]),
            "context_self_min_norm": float(anchors[clip].min_norm(context)),
            "context_self_abs_cos": float(anchors[clip].min_norm(context) * anchors[clip].radius),
            "context_self_reached": bool(anchors[clip].reached(context, float(conv_norm_thr))),
            "anchor_radius_cos": float(anchors[clip].radius),
        }
    return {
        "all_pass": bool(calibration_records_all_pass(records, conv_norm_thr=float(conv_norm_thr))),
        "per_clip": records,
        "note": "G1 uses trained base model; hidden_pre captures are np.float32 CPU arrays with shape [T,512].",
    }


def _self_abs_floor_for_clip(calibration: Optional[Mapping[str, Any]], clip: str, anchors: Mapping[str, Any], hidden: Mapping[str, np.ndarray]) -> float:
    if isinstance(calibration, Mapping):
        per_clip = calibration.get("per_clip", {})
        rec = per_clip.get(clip, {}) if isinstance(per_clip, Mapping) else {}
        if isinstance(rec, Mapping) and "context_self_abs_cos" in rec:
            return float(rec["context_self_abs_cos"])
    return float(anchors[clip].min_norm(hidden[clip]) * anchors[clip].radius)


def _run_phase2_pinned_gate_replay(
    args: argparse.Namespace,
    *,
    runner: Any,
    walk_sample: Dict[str, torch.Tensor],
    anchors: Mapping[str, Any],
    goal_head: Any,
    goal_flat: Mapping[str, torch.Tensor],
    goal_seam: Mapping[str, np.ndarray],
    start_phases: Sequence[int],
    target_states: Mapping[str, np.ndarray],
    phase2_summary: Optional[Mapping[str, Any]],
    goal_config: Mapping[str, Any],
    g0_abs_tol: float,
) -> Dict[str, Any]:
    std = StateNormalizer(dict(target_states)).std
    thr = GateThresholds()
    gate_horizon = int(
        phase2_summary.get("config", {}).get("gate_horizon", int(args.horizon))
        if isinstance(phase2_summary, Mapping) and isinstance(phase2_summary.get("config", {}), Mapping)
        else int(args.horizon)
    )
    per_clip: Dict[str, Any] = {}
    with torch.no_grad():
        for clip in TURN_CLIPS:
            delta = goal_head(goal_flat[clip]).detach()
            min_norms: List[float] = []
            outcomes: List[Mapping[str, Any]] = []
            # State-space metrics use the same goal seam as PHASE2; yaw is not part of G0.
            for phase in start_phases:
                roll = _run_context_ar_rollout(
                    runner,
                    walk_sample,
                    phase=int(phase),
                    horizon=gate_horizon,
                    context_len=int(args.context_len),
                    delta=delta,
                    injection_targets=str(goal_config["injection_targets"]),
                    injection_mode=str(goal_config["mode"]),
                    capture_grad=False,
                )
                hidden_np = roll.hidden_pre.detach().cpu().numpy()
                min_norms.append(float(anchors[clip].min_norm(hidden_np)))
                out_np = roll.out_raw.detach().cpu().numpy()
                gen_rot6d = out_np[:, :276].reshape(out_np.shape[0], 46, 6)
                gen_rv = out_np[:, ROOT_VEL_OUT_SLICE]
                cond_dir_np = roll.cond_dir_raw.detach().cpu().numpy()
                contact_np = roll.contacts.detach().cpu().numpy()
                m = min(gen_rot6d.shape[0], cond_dir_np.shape[0], contact_np.shape[0])
                roll_raw = rollout_to_egocentric(gen_rot6d[:m], gen_rv[:m], cond_dir_np[:m], contact_np[:m], fps=FPS)
                outcomes.append(evaluate_rollout_state_space(roll_raw, goal_seam[clip], std, thr))
            mn = np.asarray(min_norms, dtype=np.float64)
            per_clip[clip] = {
                "n": int(mn.size),
                "reach_rate": float(np.mean(mn <= float(args.conv_norm_thr))) if mn.size else float("nan"),
                "reach_min_norm_mean": float(np.mean(mn)) if mn.size else float("nan"),
                "reach_min_norm_min": float(np.min(mn)) if mn.size else float("nan"),
                "pop_safe_rate": float(np.mean([bool(o.get("pop_safe", False)) for o in outcomes])),
                "mean_best_pose_d": float(np.mean([float(o.get("best_pose_d", float("nan"))) for o in outcomes])),
            }

    comparisons: Dict[str, Any] = {}
    max_abs_delta = 0.0
    passed = True
    expected_root = (
        phase2_summary.get("section6_ar_gate", {}).get("per_clip", {})
        if isinstance(phase2_summary, Mapping) and isinstance(phase2_summary.get("section6_ar_gate", {}), Mapping)
        else {}
    )
    for clip in TURN_CLIPS:
        actual = per_clip[clip]
        expected = expected_root.get(clip, {}) if isinstance(expected_root, Mapping) else {}
        rows: Dict[str, Any] = {}
        for key in ("reach_rate", "reach_min_norm_mean", "reach_min_norm_min", "pop_safe_rate", "mean_best_pose_d"):
            if not isinstance(expected, Mapping) or key not in expected:
                rows[key] = {"expected": None, "actual": actual.get(key), "abs_delta": None, "passed": False}
                passed = False
                continue
            e = float(expected[key])
            a = float(actual[key])
            delta = abs(a - e)
            max_abs_delta = max(max_abs_delta, float(delta))
            ok = bool(delta <= float(g0_abs_tol))
            rows[key] = {"expected": e, "actual": a, "abs_delta": float(delta), "passed": ok}
            passed = bool(passed and ok)
        comparisons[clip] = rows
    return {
        "passed": bool(passed),
        "tolerance_abs": float(g0_abs_tol),
        "max_abs_delta": float(max_abs_delta),
        "start_phases": [int(v) for v in start_phases],
        "per_clip_recomputed": per_clip,
        "comparisons": comparisons,
    }


def _phase2_self_abs_rows(
    *,
    phase2: Optional[Mapping[str, Any]],
    anchors: Mapping[str, Any],
    hidden: Mapping[str, np.ndarray],
    k_values: Sequence[float],
    conv_norm_thr: float,
    reach_gate: float,
) -> Dict[str, Any]:
    if phase2 is None:
        raise FileNotFoundError("PHASE2 summary is required for exact A generated PHASE2 rows")
    per_clip = (
        phase2.get("section6_ar_gate", {}).get("per_clip", {})
        if isinstance(phase2.get("section6_ar_gate", {}), Mapping)
        else {}
    )
    lever = (
        phase2.get("lever1_per_step_calibration", {}).get("per_clip", {})
        if isinstance(phase2.get("lever1_per_step_calibration", {}), Mapping)
        else {}
    )
    rows: Dict[str, Any] = {}
    for clip in TURN_CLIPS:
        if clip not in per_clip:
            raise RuntimeError(f"PHASE2 summary missing section6_ar_gate.per_clip.{clip}")
        a = anchors[clip]
        rec = per_clip[clip]
        self_min_norm = None
        if isinstance(lever.get(clip), Mapping) and "fullseq_self_min_norm" in lever[clip]:
            self_min_norm = float(lever[clip]["fullseq_self_min_norm"])
        if self_min_norm is None:
            self_min_norm = float(a.min_norm(hidden[clip]))
        radius = float(a.radius)
        self_abs = float(self_min_norm * radius)
        gen_min_norm_min = float(rec["reach_min_norm_min"])
        gen_min_norm_mean = float(rec["reach_min_norm_mean"])
        gen_abs_min = float(gen_min_norm_min * radius)
        gen_abs_mean = float(gen_min_norm_mean * radius)
        k_pass = {
            f"k={_fmt(k, 0) if float(k).is_integer() else _fmt(k, 2)}": bool(gen_abs_min <= float(k) * self_abs)
            for k in k_values
        }
        rows[clip] = {
            "anchor_radius_cos": radius,
            "self_reach_min_norm_fullseq": self_min_norm,
            "self_reach_abs_cos": self_abs,
            "phase2_generated_min_norm_min": gen_min_norm_min,
            "phase2_generated_min_norm_mean": gen_min_norm_mean,
            "phase2_generated_abs_cos_min": gen_abs_min,
            "phase2_generated_abs_cos_mean": gen_abs_mean,
            "old_radius_best_start_pass_min_norm_le_conv": bool(gen_min_norm_min <= float(conv_norm_thr)),
            "old_radius_reach_rate": float(rec["reach_rate"]),
            "old_radius_reach_rate_gate_pass": bool(float(rec["reach_rate"]) >= float(reach_gate)),
            "new_self_abs_best_start_pass": k_pass,
            "new_self_abs_margin_vs_k": {
                key: float(gen_abs_min / max(float(k) * self_abs, 1e-12))
                for key, k in zip(k_pass.keys(), k_values)
            },
            "pop_safe_rate": float(rec.get("pop_safe_rate", float("nan"))),
            "mean_best_pose_d": float(rec.get("mean_best_pose_d", float("nan"))),
            "note": "new self gate uses best-start abs cos because PHASE2 did not save per-start hidden_pre/min_norm arrays.",
        }
    ltr = rows.get("Walk_L_To_R", {})
    rows["_decision"] = {
        "l_to_r_old_rate_gate_pass": bool(ltr.get("old_radius_reach_rate_gate_pass", False)),
        "l_to_r_new_k2_k3_k5_all_fail": bool(
            ltr and not any(bool(v) for v in ltr.get("new_self_abs_best_start_pass", {}).values())
        ),
        "l_to_r_conclusion_flips_under_self_abs_gate": bool(
            ltr.get("old_radius_reach_rate_gate_pass", False)
            and not any(bool(v) for v in ltr.get("new_self_abs_best_start_pass", {}).values())
        ),
    }
    return rows


def _run_runtime_bc(
    args: argparse.Namespace,
    *,
    anchors: Mapping[str, Any],
    hidden: Mapping[str, np.ndarray],
    target_states: Mapping[str, np.ndarray],
    phase2_state: Optional[Mapping[str, Any]],
    phase2_summary: Optional[Mapping[str, Any]],
    k_values: Sequence[float],
) -> Dict[str, Any]:
    npz_root = Path(args.npz_root)
    runner = freerun.FreeRunCycleRunner(_runner_args(args))
    walk_ds = runner._build_dataset(npz_root / f"{WALK_F}.npz", seq_len=64)
    runner._ensure_model_ready(walk_ds)
    runner.model.eval()
    walk_clip = walk_ds.clips[0]
    walk_T = int(walk_clip.X.shape[0])
    walk_sample = freerun._build_full_cycle_sample(walk_ds, walk_clip, seq_len=walk_T)

    phase2_load: Optional[Dict[str, Any]] = None
    goal_head = None
    goal_config = _goal_config_from_phase2_state(phase2_state, args)
    if phase2_state is not None:
        if phase2_summary is None:
            raise RuntimeError("--phase2-state requires a PHASE2 summary for G0 round-trip verification")
        phase2_load = _apply_phase2_model_state(runner, phase2_state)

    target_samples: Dict[str, Dict[str, torch.Tensor]] = {}
    for clip in TURN_CLIPS:
        target_T = int(target_states[clip].shape[0])
        ds = runner._build_dataset(npz_root / f"{clip}.npz", seq_len=target_T)
        runner._ensure_model_ready(ds)
        target_samples[clip] = freerun._build_full_cycle_sample(ds, ds.clips[0], seq_len=target_T)

    runner._ensure_model_ready(walk_ds)
    runner.model.eval()
    std = StateNormalizer(dict(target_states)).std
    thr = GateThresholds()
    if phase2_state is not None and isinstance(phase2_state.get("start_phases"), Sequence):
        start_phases = [int(v) for v in phase2_state.get("start_phases", [])]
    else:
        start_phases = select_start_phases(walk_T, int(args.n_starts))
    if not start_phases:
        start_phases = select_start_phases(walk_T, int(args.n_starts))

    K = SEAM_LEN_K
    goal_seam: Dict[str, np.ndarray] = {}
    goal_flat: Dict[str, torch.Tensor] = {}
    for clip in TURN_CLIPS:
        target = target_states[clip]
        g0 = int(min(args.goal_horizon, target.shape[0] - K))
        goal_seam[clip] = target[g0 : g0 + K]
        goal_flat[clip] = _goal_flat(goal_seam[clip]).to(runner.device)

    calibration: Optional[Dict[str, Any]] = None
    g0_roundtrip: Optional[Dict[str, Any]] = None
    if phase2_state is not None:
        fallback_goal_flat_dim = int(K * goal_seam[TURN_CLIPS[0]].shape[1])
        goal_head = _build_goal_head_from_phase2_state(
            phase2_state,
            device=runner.device,
            fallback_goal_flat_dim=fallback_goal_flat_dim,
        )
        calibration = _run_trained_self_reach_calibration(
            runner,
            npz_root=npz_root,
            saved_hidden=hidden,
            anchors=anchors,
            context_len=int(args.context_len),
            conv_norm_thr=float(args.conv_norm_thr),
        )
        runner._ensure_model_ready(walk_ds)
        runner.model.eval()
        g0_roundtrip = _run_phase2_pinned_gate_replay(
            args,
            runner=runner,
            walk_sample=walk_sample,
            anchors=anchors,
            goal_head=goal_head,
            goal_flat=goal_flat,
            goal_seam=goal_seam,
            start_phases=start_phases,
            target_states=target_states,
            phase2_summary=phase2_summary,
            goal_config=goal_config,
            g0_abs_tol=float(args.g0_abs_tol),
        )
        if not bool(g0_roundtrip["passed"]):
            raise RuntimeError(
                "G0 round-trip failed: saved PHASE2 state does not reproduce phase2_summary "
                f"within abs_tol={float(args.g0_abs_tol)}; max_abs_delta={g0_roundtrip['max_abs_delta']}"
            )

    out: Dict[str, Any] = {
        "status": "exact_phase2_trained_state" if phase2_state is not None else "base_checkpoint_no_goal_head_readonly",
        "limitation": None
        if phase2_state is not None
        else (
            "Exact PHASE2 goal-head/free rollout was not run because no --phase2-state was provided; "
            "B/C use the base checkpoint with no goal head."
        ),
        "phase2_state_loaded": bool(phase2_state is not None),
        "phase2_load": phase2_load,
        "G0_roundtrip": g0_roundtrip,
        "G1_trained_self_reach_calibration": calibration,
        "config": {
            "context_len": int(args.context_len),
            "n_starts": len(start_phases),
            "horizon": int(args.horizon),
            "walk_f_len": walk_T,
            "goal_injection_targets": str(goal_config["injection_targets"]),
            "goal_injection_mode": str(goal_config["mode"]),
            "contact_free_definition": "seed context contacts from Walk_F, future contacts from model contacts_plan self-carry",
            "cond_free_definition": "target turn cond trajectory, edge-held after target end",
        },
        "per_clip": {},
    }
    H = int(args.horizon)
    pinned_key = "pinned_walk_f_cond_contact_trained_goal" if phase2_state is not None else "pinned_walk_f_cond_contact_no_goal"
    free_key = (
        "free_target_cond_self_contact_trained_goal"
        if phase2_state is not None
        else "free_target_cond_self_contact_no_goal"
    )
    with torch.no_grad():
        for clip in TURN_CLIPS:
            anchor = anchors[clip]
            target_sample = target_samples[clip]
            target_cond_dir = _target_cond_dir_from_sample(target_sample, H)
            clip_goal_seam = goal_seam[clip]
            clip_delta = goal_head(goal_flat[clip]).detach() if goal_head is not None else None
            self_abs_floor = _self_abs_floor_for_clip(calibration, clip, anchors, hidden)

            pinned_min_norms: List[float] = []
            pinned_abs: List[float] = []
            pinned_states: List[Mapping[str, Any]] = []
            pinned_yaw: List[Mapping[str, Any]] = []
            free_min_norms: List[float] = []
            free_abs: List[float] = []
            free_states: List[Mapping[str, Any]] = []
            free_yaw: List[Mapping[str, Any]] = []
            free_generated_contact_fraction: List[float] = []
            free_fallback_contact_steps: List[int] = []

            for phase in start_phases:
                pinned = _run_context_ar_rollout(
                    runner,
                    walk_sample,
                    phase=int(phase),
                    horizon=H,
                    context_len=int(args.context_len),
                    delta=clip_delta,
                    injection_targets=str(goal_config["injection_targets"]),
                    injection_mode=str(goal_config["mode"]),
                    capture_grad=False,
                )
                pinned_hidden = pinned.hidden_pre.detach().cpu().numpy()
                pmn = float(anchor.min_norm(pinned_hidden))
                pinned_min_norms.append(pmn)
                pinned_abs.append(float(pmn * anchor.radius))
                pinned_state, pinned_yaw_row = _rollout_state_and_yaw(
                    pinned,
                    target_cond_dir=target_cond_dir,
                    goal_seam=clip_goal_seam,
                    std=std,
                    thr=thr,
                )
                pinned_states.append(pinned_state)
                pinned_yaw.append(pinned_yaw_row)

                free, meta = _run_target_cond_self_contact_rollout(
                    runner,
                    walk_sample,
                    target_sample,
                    phase=int(phase),
                    horizon=H,
                    context_len=int(args.context_len),
                    delta=clip_delta,
                    injection_targets=str(goal_config["injection_targets"]),
                    injection_mode=str(goal_config["mode"]),
                )
                free_hidden = free.hidden_pre.detach().cpu().numpy()
                fmn = float(anchor.min_norm(free_hidden))
                free_min_norms.append(fmn)
                free_abs.append(float(fmn * anchor.radius))
                free_state, free_yaw_row = _rollout_state_and_yaw(
                    free,
                    target_cond_dir=target_cond_dir,
                    goal_seam=clip_goal_seam,
                    std=std,
                    thr=thr,
                )
                free_states.append(free_state)
                free_yaw.append(free_yaw_row)
                free_generated_contact_fraction.append(float(meta.generated_contact_fraction))
                free_fallback_contact_steps.append(int(meta.fallback_contact_steps))

            pinned_summary = _summarize_runtime_rows(
                min_norms=pinned_min_norms,
                abs_cos=pinned_abs,
                state_outcomes=pinned_states,
                yaw_rows=pinned_yaw,
                conv_norm_thr=float(args.conv_norm_thr),
                self_abs_floor=self_abs_floor,
                k_values=k_values,
            )
            free_summary = _summarize_runtime_rows(
                min_norms=free_min_norms,
                abs_cos=free_abs,
                state_outcomes=free_states,
                yaw_rows=free_yaw,
                conv_norm_thr=float(args.conv_norm_thr),
                self_abs_floor=self_abs_floor,
                k_values=k_values,
            )
            free_summary["generated_contact_fraction_mean"] = _mean_finite(free_generated_contact_fraction)
            free_summary["fallback_contact_steps_total"] = int(sum(free_fallback_contact_steps))
            out["per_clip"][clip] = {
                pinned_key: pinned_summary,
                free_key: free_summary,
                "free_vs_pinned_min_norm_ratio": float(
                    free_summary["reach_min_norm_mean"] / max(float(pinned_summary["reach_min_norm_mean"]), 1e-12)
                ),
                "free_vs_pinned_abs_cos_ratio": float(
                    free_summary["reach_abs_cos_mean"] / max(float(pinned_summary["reach_abs_cos_mean"]), 1e-12)
                ),
                "self_abs_floor_source": "trained_context_window" if calibration is not None else "saved_fullseq_hidden",
                "self_abs_floor": float(self_abs_floor),
            }
    out["column_keys"] = {"pinned": pinned_key, "free": free_key}
    if "Walk_L_To_R" in out["per_clip"]:
        out["walk_l_to_r"] = out["per_clip"]["Walk_L_To_R"]
    return out


def _g3_decision(bc_rows: Optional[Mapping[str, Any]], *, k_label: str = "k=3") -> Dict[str, Any]:
    if not isinstance(bc_rows, Mapping) or not isinstance(bc_rows.get("walk_l_to_r"), Mapping):
        return {
            "B4_seam_status": "blocked",
            "reason": "trained free-rollout rows unavailable",
            "hidden_pre_self_reach_is_necessary_not_sufficient": True,
        }
    cols = bc_rows.get("column_keys", {})
    pinned_key = cols.get("pinned") if isinstance(cols, Mapping) else None
    free_key = cols.get("free") if isinstance(cols, Mapping) else None
    ltr = bc_rows["walk_l_to_r"]
    pinned = ltr.get(pinned_key, {}) if pinned_key else {}
    free = ltr.get(free_key, {}) if free_key else {}
    pinned_self = pinned.get("self_reach_gate", {}) if isinstance(pinned, Mapping) else {}
    free_self = free.get("self_reach_gate", {}) if isinstance(free, Mapping) else {}
    pinned_rate = float(pinned_self.get("rate_by_k", {}).get(k_label, float("nan"))) if isinstance(pinned_self, Mapping) else float("nan")
    free_rate = float(free_self.get("rate_by_k", {}).get(k_label, float("nan"))) if isinstance(free_self, Mapping) else float("nan")
    pinned_mae = float(pinned.get("yaw_heading_mae_rad_mean", float("nan"))) if isinstance(pinned, Mapping) else float("nan")
    free_mae = float(free.get("yaw_heading_mae_rad_mean", float("nan"))) if isinstance(free, Mapping) else float("nan")
    free_corr = float(free.get("yaw_corr_mean", float("nan"))) if isinstance(free, Mapping) else float("nan")
    free_pop = float(free.get("pop_safe_rate", float("nan"))) if isinstance(free, Mapping) else float("nan")
    mae_ratio = float(free_mae / max(pinned_mae, 1e-12)) if np.isfinite(free_mae) and np.isfinite(pinned_mae) else float("nan")
    checks = {
        f"self_reach_{k_label}_rate_lifted_free_vs_pinned": bool(np.isfinite(free_rate) and np.isfinite(pinned_rate) and free_rate > pinned_rate),
        "realized_yaw_corr_positive": bool(np.isfinite(free_corr) and free_corr > 0.0),
        "heading_mae_significantly_down_vs_pinned": bool(np.isfinite(mae_ratio) and mae_ratio <= 0.90),
        "pop_safe_positive": bool(np.isfinite(free_pop) and free_pop > 0.0),
    }
    unblocked = bool(all(checks.values()))
    return {
        "B4_seam_status": "candidate_unblocked" if unblocked else "blocked",
        "hidden_pre_self_reach_is_necessary_not_sufficient": True,
        "injection_still_writes_hidden_pre_directly": True,
        "required_simultaneous_checks": checks,
        "pinned_self_reach_rate_k3": pinned_rate,
        "free_self_reach_rate_k3": free_rate,
        "free_minus_pinned_self_reach_rate_k3": float(free_rate - pinned_rate)
        if np.isfinite(free_rate) and np.isfinite(pinned_rate)
        else float("nan"),
        "pinned_heading_mae_rad": pinned_mae,
        "free_heading_mae_rad": free_mae,
        "free_over_pinned_heading_mae_ratio": mae_ratio,
        "free_yaw_corr": free_corr,
        "free_pop_safe_rate": free_pop,
        "reason": (
            "all required trained free-rollout checks passed"
            if unblocked
            else "one or more trained free-rollout checks failed; B4/seam remains blocked"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="W0 reach honesty probe for action-handoff in-betweening.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=str, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=str, default=DEFAULT_Z_FEATURES)
    p.add_argument("--phase2-summary", type=Path, default=Path(DEFAULT_PHASE2_SUMMARY))
    p.add_argument("--phase2-state", type=Path, default=None)
    p.add_argument("--reach-anchor-summary", type=Path, default=Path(DEFAULT_REACH_ANCHOR_SUMMARY))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--horizon", type=int, default=72)
    p.add_argument("--goal-horizon", type=int, default=12)
    p.add_argument("--conv-norm-thr", type=float, default=DEFAULT_CONV_NORM_THR)
    p.add_argument("--end-window-k", type=int, default=DEFAULT_END_WINDOW_K)
    p.add_argument("--reach-gate", type=float, default=0.7)
    p.add_argument("--self-reach-k", type=str, default="2,3,5")
    p.add_argument("--g0-abs-tol", type=float, default=1e-5)
    p.add_argument("--skip-runtime-rollouts", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    z_path = Path(args.z_features)
    npz_root = Path(args.npz_root)
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")
    if not npz_root.exists():
        raise FileNotFoundError(f"npz root not found: {npz_root}")
    if not Path(args.checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_inbetween_reach_honesty_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = _parse_float_list(str(args.self_reach_k))
    hidden = load_hidden_pre(z_path, LOCKED_CLIPS)
    anchors = build_hidden_pre_anchors(hidden, TURN_CLIPS, int(args.end_window_k))
    target_states = load_clip_states(z_path, npz_root, fps=FPS)
    phase2_state: Optional[Dict[str, Any]] = None
    phase2_summary_path = Path(args.phase2_summary)
    if args.phase2_state is not None:
        phase2_state_path = Path(args.phase2_state).expanduser()
        phase2_state = _load_phase2_state(phase2_state_path)
        default_summary = Path(DEFAULT_PHASE2_SUMMARY)
        sibling_summary = phase2_state_path.parent / "phase2_guarded_finetune_summary.json"
        if phase2_summary_path == default_summary and sibling_summary.exists():
            phase2_summary_path = sibling_summary
    phase2 = _load_json_if_present(phase2_summary_path)
    anchor_summary = _load_json_if_present(Path(args.reach_anchor_summary))

    a_rows = _phase2_self_abs_rows(
        phase2=phase2,
        anchors=anchors,
        hidden=hidden,
        k_values=k_values,
        conv_norm_thr=float(args.conv_norm_thr),
        reach_gate=float(args.reach_gate),
    )
    bc_rows: Optional[Dict[str, Any]] = None
    if not bool(args.skip_runtime_rollouts):
        bc_rows = _run_runtime_bc(
            args,
            anchors=anchors,
            hidden=hidden,
            target_states=target_states,
            phase2_state=phase2_state,
            phase2_summary=phase2,
            k_values=k_values,
        )
    g3 = _g3_decision(bc_rows)

    summary: Dict[str, Any] = {
        "task": "Action-handoff in-betweening W1a exact PHASE2 trained reach honesty probe",
        "status": "EXACT_PHASE2_TRAINED" if phase2_state is not None else "PROVISIONAL_NO_PHASE2_STATE",
        "no_training": True,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "z_features_path": str(z_path.resolve()),
        "npz_root": str(npz_root.resolve()),
        "phase2_summary_path": str(phase2_summary_path.resolve()),
        "phase2_state_path": str(Path(args.phase2_state).expanduser().resolve()) if args.phase2_state is not None else None,
        "reach_anchor_summary_path": str(Path(args.reach_anchor_summary).resolve()),
        "phase2_summary_loaded": phase2 is not None,
        "phase2_state_loaded": phase2_state is not None,
        "reach_anchor_summary_loaded": anchor_summary is not None,
        "limitations": {
            "phase2_goal_head_state_saved": bool(phase2_state is not None),
            "exact_phase2_free_rollout_status": (
                "DONE: B/C use the saved PHASE2 fine-tuned base model.state_dict plus goal_head.state_dict."
                if phase2_state is not None
                else "TODO: provide --phase2-state to compute exact PHASE2 trained free rollout."
            ),
            "self_gate_rate_status": (
                "A remains the saved PHASE2 aggregate-only flip check. Exact B/C rows report per-start "
                "self-reach reach_rate(k=2/3/5) from newly replayed hidden_pre sequences."
            ),
        },
        "thresholds_provisional": {
            "old_conv_norm_thr_radius": float(args.conv_norm_thr),
            "old_reach_rate_gate": float(args.reach_gate),
            "self_reach_k_values": [float(k) for k in k_values],
            "k_rationale": {
                "2": "strict: generated hidden_pre must be within 2x the clip's own fullseq self-reach floor",
                "3": "middle diagnostic band",
                "5": "lenient: still rejects a target that only passes because its anchor radius is loose",
            },
        },
        "A_absolute_self_reach_gate": a_rows,
        "B_pinned_vs_free_reach": bc_rows,
        "C_realized_yaw": bc_rows,
        "G0_roundtrip": bc_rows.get("G0_roundtrip") if isinstance(bc_rows, Mapping) else None,
        "G1_trained_self_reach_calibration": bc_rows.get("G1_trained_self_reach_calibration")
        if isinstance(bc_rows, Mapping)
        else None,
        "G2_trained_free_rollout": bc_rows,
        "G3_decision": g3,
        "walk_l_to_r_row": "Walk_L_To_R",
    }

    json_path = out_dir / "reach_honesty_probe_summary.json"
    md_path = out_dir / "reach_honesty_probe_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# Reach Honesty Probe - W1a Exact PHASE2 Trained Replay")
    lines.append("")
    lines.append("> Read-only diagnostic. No training was run by this probe.")
    if phase2_state is not None:
        lines.append(f"> Loaded PHASE2 state: `{Path(args.phase2_state).expanduser().resolve()}`")
    else:
        lines.append("> No `--phase2-state` was provided; B/C are the old no-goal base fallback.")
    lines.append("")
    lines.append("## A. Radius Gate vs Absolute Self-Reach Gate")
    lines.append(
        "| target | old radius reach_rate | old rate pass | old best min_norm | gen abs min | self abs floor | k=2 | k=3 | k=5 | pop_safe |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---|---|---|---:|")
    for clip in TURN_CLIPS:
        r = a_rows[clip]
        kp = r["new_self_abs_best_start_pass"]
        lines.append(
            f"| {clip} | {_fmt(r['old_radius_reach_rate'], 2)} | {r['old_radius_reach_rate_gate_pass']} | "
            f"{_fmt(r['phase2_generated_min_norm_min'], 3)} | {_fmt(r['phase2_generated_abs_cos_min'], 5)} | "
            f"{_fmt(r['self_reach_abs_cos'], 5)} | {kp.get('k=2', False)} | {kp.get('k=3', False)} | "
            f"{kp.get('k=5', False)} | {_fmt(r['pop_safe_rate'], 2)} |"
        )
    dec = a_rows["_decision"]
    lines.append("")
    lines.append(
        f"- L_R old radius gate pass: {dec['l_to_r_old_rate_gate_pass']}; "
        f"self-abs k=2/3/5 all fail: {dec['l_to_r_new_k2_k3_k5_all_fail']}; "
        f"conclusion flips: **{dec['l_to_r_conclusion_flips_under_self_abs_gate']}**."
    )
    lines.append("")
    lines.append("## G0. PHASE2 State Round-Trip")
    if bc_rows is None or bc_rows.get("G0_roundtrip") is None:
        lines.append("- Not run.")
    else:
        g0 = bc_rows["G0_roundtrip"]
        lines.append(
            f"- passed: **{g0['passed']}**, abs_tol={_fmt(g0['tolerance_abs'], 8)}, "
            f"max_abs_delta={_fmt(g0['max_abs_delta'], 8)}"
        )
        ltr_cmp = g0["comparisons"]["Walk_L_To_R"]
        lines.append(
            f"- L_R reach_min_norm_min: {_fmt(ltr_cmp['reach_min_norm_min']['actual'], 6)} "
            f"(expected {_fmt(ltr_cmp['reach_min_norm_min']['expected'], 6)}); "
            f"reach_rate: {_fmt(ltr_cmp['reach_rate']['actual'], 2)} "
            f"(expected {_fmt(ltr_cmp['reach_rate']['expected'], 2)})"
        )
    lines.append("")
    lines.append("## G1. Trained Self-Reach Calibration")
    if bc_rows is None or bc_rows.get("G1_trained_self_reach_calibration") is None:
        lines.append("- Not run.")
    else:
        g1 = bc_rows["G1_trained_self_reach_calibration"]
        lines.append(f"- all_pass: **{g1['all_pass']}**")
        lines.append("| target | fullseq min_norm | context min_norm | context self abs | context reached |")
        lines.append("|---|---:|---:|---:|---|")
        for clip in TURN_CLIPS:
            r = g1["per_clip"][clip]
            lines.append(
                f"| {clip} | {_fmt(r['fullseq_self_min_norm'], 3)} | "
                f"{_fmt(r['context_self_min_norm'], 3)} | {_fmt(r['context_self_abs_cos'], 6)} | "
                f"{r['context_self_reached']} |"
            )
    lines.append("")
    lines.append("## B. Pinned vs Free Reach")
    if bc_rows is None:
        lines.append("- Runtime rollout skipped.")
    else:
        if phase2_state is None:
            lines.append("> These B rows are base-checkpoint/no-goal-head read-only rollouts.")
        else:
            lines.append("> These B rows use the saved PHASE2 trained base + goal_head.")
        cols = bc_rows.get("column_keys", {})
        pinned_key = cols.get("pinned", "pinned_walk_f_cond_contact_no_goal")
        free_key = cols.get("free", "free_target_cond_self_contact_no_goal")
        lines.append(
            "| target | pinned min_norm mean/min | free min_norm mean/min | pinned k3 | free k3 | "
            "free/pinned mean | pinned pop_safe | free pop_safe |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for clip in TURN_CLIPS:
            r = bc_rows["per_clip"][clip]
            p0 = r[pinned_key]
            fr = r[free_key]
            p_k3 = p0.get("self_reach_gate", {}).get("rate_by_k", {}).get("k=3", float("nan"))
            f_k3 = fr.get("self_reach_gate", {}).get("rate_by_k", {}).get("k=3", float("nan"))
            lines.append(
                f"| {clip} | {_fmt(p0['reach_min_norm_mean'], 2)}/{_fmt(p0['reach_min_norm_min'], 2)} | "
                f"{_fmt(fr['reach_min_norm_mean'], 2)}/{_fmt(fr['reach_min_norm_min'], 2)} | "
                f"{_fmt(p_k3, 2)} | {_fmt(f_k3, 2)} | {_fmt(r['free_vs_pinned_min_norm_ratio'], 2)} | "
                f"{_fmt(p0['pop_safe_rate'], 2)} | {_fmt(fr['pop_safe_rate'], 2)} |"
            )
    lines.append("")
    lines.append("## C. Realized Yaw")
    if bc_rows is None:
        lines.append("- Runtime rollout skipped.")
    else:
        lines.append(
            "| target | pinned realized/target final deg | free realized/target final deg | free heading MAE deg | free yaw-rate MAE rad/s | free corr |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        cols = bc_rows.get("column_keys", {})
        pinned_key = cols.get("pinned", "pinned_walk_f_cond_contact_no_goal")
        free_key = cols.get("free", "free_target_cond_self_contact_no_goal")
        for clip in TURN_CLIPS:
            r = bc_rows["per_clip"][clip]
            p0 = r[pinned_key]
            fr = r[free_key]
            lines.append(
                f"| {clip} | {_fmt(np.degrees(p0['yaw_realized_final_rad_mean']), 1)}/"
                f"{_fmt(np.degrees(p0['yaw_target_final_rad_mean']), 1)} | "
                f"{_fmt(np.degrees(fr['yaw_realized_final_rad_mean']), 1)}/"
                f"{_fmt(np.degrees(fr['yaw_target_final_rad_mean']), 1)} | "
                f"{_fmt(np.degrees(fr['yaw_heading_mae_rad_mean']), 1)} | "
                f"{_fmt(fr['yaw_yaw_rate_mae_rad_s_mean'], 2)} | {_fmt(fr['yaw_corr_mean'], 2)} |"
            )
    lines.append("")
    lines.append("## G3. Decision")
    lines.append("- hidden_pre self-reach is necessary, not sufficient; the injection still writes into hidden_pre directly.")
    lines.append(f"- B4/seam status: **{g3['B4_seam_status']}**")
    lines.append(f"- reason: {g3['reason']}")
    if isinstance(g3.get("required_simultaneous_checks"), Mapping):
        for key, value in g3["required_simultaneous_checks"].items():
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Limitations")
    lines.append(f"- {summary['limitations']['exact_phase2_free_rollout_status']}")
    lines.append(f"- {summary['limitations']['self_gate_rate_status']}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(
        "[W1a] L_R flip="
        f"{a_rows['_decision']['l_to_r_conclusion_flips_under_self_abs_gate']} "
        f"runtime_rollouts={'skipped' if bc_rows is None else 'done'} "
        f"phase2_state={'loaded' if phase2_state is not None else 'not_loaded'} "
        f"B4={g3['B4_seam_status']}"
    )


if __name__ == "__main__":
    main()
