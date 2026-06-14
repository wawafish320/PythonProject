#!/usr/bin/env python3
"""Read-only bone-angular-velocity bridge probe for action handoff.

This tool does not train, mutate checkpoints, or modify production gates.  It
tests whether a k-frame transition in the base-model normalized
BoneAngularVelocities slice is sufficient to collapse the matched-seam trunk
jump, then checks a local realized-motion metric that includes FK foot slip and
bone-angular-velocity continuity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train.validate.run_freerun_cycles as freerun  # noqa: E402
from train.action_handoff_inbetween_cond_probe import rollout_to_egocentric  # noqa: E402
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer, evaluate_rollout_state_space  # noqa: E402
from train.data.action_handoff_inbetween import (  # noqa: E402
    CONTACT_SLICE,
    EGO_VEL_SLICE,
    FPS,
    GROUND_CONTACT_THR,
    GROUND_POSE_THR,
    POSE_SLICE,
    POSE_TOPK,
    RAW_COND_DIR_SLICE,
    SEAM_LEN_K,
    STATE_DIM,
    TURN_CLIPS,
    WALK_F,
    YAW_RATE_SLICE,
    load_clip_states,
)
from tools.run_action_handoff_inbetween_b1_cond_baseline_probe import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CKPT,
    DEFAULT_ENCODER_BUNDLE,
    DEFAULT_NPZ_ROOT,
    DEFAULT_PRETRAIN_TEMPLATE,
    DEFAULT_Z_FEATURES,
    _make_runner_args,
)
from tools.run_action_handoff_matched_seam_neuron_audit import (  # noqa: E402
    ANGVEL_KEY,
    ROOT_POS_KEY,
    ROOT_VEL_KEY,
    ROT6D_KEY,
    _candidate_onset,
    _layout_slice,
    _load_clip,
    _load_npz_raw,
    _make_sample,
)
from tools.run_action_handoff_regime_bridge_probe import (  # noqa: E402
    ROOT_VEL_OUT_SLICE,
    ForwardProbeResult,
    _build_variant_samples,
    _collapse_fraction,
    _dump_json,
    _dump_md,
    _fmt,
    _foot_slip_summary,
    _forward_probe,
    _load_hidden_pre,
    _load_raw_angvel,
    _make_mapping_pairs,
    _mean_finite,
    _parse_clips,
    _r2_score,
    _split_indices,
    _standardize_train_test,
)


COMBO_COND_DIR_SLICE = slice(RAW_COND_DIR_SLICE[0], RAW_COND_DIR_SLICE[1])
SIGNAL_KEYS = {
    "shared_encoder_0": "shared_encoder_0",
    "hidden_pre": "hidden_pre_pasa_lnq_input",
    "h_final": "out__h_final",
    "out": "out__out",
    "contacts_plan": "out__contacts_plan",
}


@dataclass
class RidgePredictor:
    alpha: float
    x_mu: np.ndarray
    x_std: np.ndarray
    coef: np.ndarray
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xr = np.asarray(X, dtype=np.float64)
        Xs = (Xr - self.x_mu) / self.x_std
        Xi = np.concatenate([Xs, np.ones((Xs.shape[0], 1), dtype=np.float64)], axis=1)
        return (Xi @ self.coef).astype(np.float32)


def _parse_int_list(raw: str, *, name: str) -> List[int]:
    vals: List[int] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError(f"{name} must include at least one integer")
    return vals


def _slice_bounds(sl: slice) -> List[int]:
    return [int(sl.start or 0), int(sl.stop or 0)]


def _clone_sample(sample: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {str(k): v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}


def _ramp_post_slice(
    matched: Mapping[str, torch.Tensor],
    walk: Mapping[str, torch.Tensor],
    *,
    field: str,
    sl: Optional[slice],
    cut: int,
    frames: int,
    target_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    src = matched[field]
    out = src.clone()
    k = max(1, min(int(frames), int(src.shape[1]) - int(cut)))
    if k <= 0:
        return out
    if sl is None:
        walk_seg = walk[field][:, int(cut) : int(cut) + k]
        target_seg = src[:, int(cut) : int(cut) + k] if target_values is None else target_values[:, :k]
        alpha = torch.linspace(
            1.0 / float(k + 1),
            float(k) / float(k + 1),
            steps=k,
            device=src.device,
            dtype=src.dtype,
        ).view(1, k, 1)
        out[:, int(cut) : int(cut) + k] = (1.0 - alpha) * walk_seg + alpha * target_seg.to(src)
        return out
    walk_seg = walk[field][:, int(cut) : int(cut) + k, sl]
    target_seg = src[:, int(cut) : int(cut) + k, sl] if target_values is None else target_values[:, :k]
    alpha = torch.linspace(
        1.0 / float(k + 1),
        float(k) / float(k + 1),
        steps=k,
        device=src.device,
        dtype=src.dtype,
    ).view(1, k, 1)
    out[:, int(cut) : int(cut) + k, sl] = (1.0 - alpha) * walk_seg + alpha * target_seg.to(src)
    return out


def _build_k_ramp_samples(
    matched: Mapping[str, torch.Tensor],
    walk: Mapping[str, torch.Tensor],
    *,
    cut: int,
    k_values: Sequence[int],
    rootvel_sl: slice,
    state_angvel_sl: slice,
) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for k in k_values:
        kk = int(k)
        s = _clone_sample(matched)
        s["state"] = _ramp_post_slice(
            matched,
            walk,
            field="state",
            sl=state_angvel_sl,
            cut=int(cut),
            frames=kk,
        )
        out[f"bone_angvel_ramp_k{kk}"] = s

        c = _clone_sample(matched)
        c["state"] = _ramp_post_slice(
            matched,
            walk,
            field="state",
            sl=state_angvel_sl,
            cut=int(cut),
            frames=kk,
        )
        c["state"] = _ramp_post_slice(
            {**matched, "state": c["state"]},
            walk,
            field="state",
            sl=rootvel_sl,
            cut=int(cut),
            frames=kk,
        )
        if c["cond"].shape[-1] >= COMBO_COND_DIR_SLICE.stop:
            c["cond"] = _ramp_post_slice(
                c,
                walk,
                field="cond",
                sl=COMBO_COND_DIR_SLICE,
                cut=int(cut),
                frames=kk,
            )
        out[f"bone_angvel_rootvel_cmdyaw_ramp_k{kk}"] = c
    return out


def _signal_ratio(probe: ForwardProbeResult, logical_name: str) -> Optional[float]:
    key = SIGNAL_KEYS[logical_name]
    return probe.summary_by_signal.get(key, {}).get("cut_over_pre4")


def _summarize_probe_row(
    *,
    target: str,
    variant: str,
    probe: ForwardProbeResult,
    base_probe: ForwardProbeResult,
    walk_probe: ForwardProbeResult,
) -> Dict[str, Any]:
    base_hidden = _signal_ratio(base_probe, "hidden_pre")
    walk_hidden = _signal_ratio(walk_probe, "hidden_pre")
    base_shared = _signal_ratio(base_probe, "shared_encoder_0")
    walk_shared = _signal_ratio(walk_probe, "shared_encoder_0")
    row: Dict[str, Any] = {"target": target, "variant": variant}
    for logical in SIGNAL_KEYS:
        row[f"{logical}_cut_over_pre4"] = _signal_ratio(probe, logical)
    row["hidden_pre_collapse_fraction_to_walk"] = _collapse_fraction(
        base_hidden,
        row.get("hidden_pre_cut_over_pre4"),
        walk_hidden,
    )
    row["shared_encoder0_collapse_fraction_to_walk"] = _collapse_fraction(
        base_shared,
        row.get("shared_encoder_0_cut_over_pre4"),
        walk_shared,
    )
    row["trunk_input_dim"] = int(probe.trunk_input_dim)
    row["plan_feat_slice_in_trunk_x"] = list(probe.plan_feat_slice) if probe.plan_feat_slice is not None else None
    return row


def _aggregate_by_variant(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for variant in sorted({str(r["variant"]) for r in rows}):
        xs = [r for r in rows if str(r["variant"]) == variant]
        out[variant] = {
            "n": int(len(xs)),
            "shared_encoder_0_cut_over_pre4_mean": _mean_finite([r.get("shared_encoder_0_cut_over_pre4") for r in xs]),
            "hidden_pre_cut_over_pre4_mean": _mean_finite([r.get("hidden_pre_cut_over_pre4") for r in xs]),
            "h_final_cut_over_pre4_mean": _mean_finite([r.get("h_final_cut_over_pre4") for r in xs]),
            "out_cut_over_pre4_mean": _mean_finite([r.get("out_cut_over_pre4") for r in xs]),
            "contacts_plan_cut_over_pre4_mean": _mean_finite([r.get("contacts_plan_cut_over_pre4") for r in xs]),
            "hidden_pre_collapse_fraction_to_walk_mean": _mean_finite(
                [r.get("hidden_pre_collapse_fraction_to_walk") for r in xs]
            ),
            "shared_encoder0_collapse_fraction_to_walk_mean": _mean_finite(
                [r.get("shared_encoder0_collapse_fraction_to_walk") for r in xs]
            ),
        }
    return out


def _denorm_state(sample: Mapping[str, torch.Tensor], mu_x: np.ndarray, std_x: np.ndarray) -> np.ndarray:
    x = sample["state"].detach().cpu().float().numpy()[0].astype(np.float32)
    return (x * std_x.reshape(1, -1) + mu_x.reshape(1, -1)).astype(np.float32)


def _readout_raw_from_probe(
    trainer: Any,
    sample: Mapping[str, torch.Tensor],
    probe: ForwardProbeResult,
    *,
    cut: int,
    mu_x: np.ndarray,
    std_x: np.ndarray,
    rot_x_sl: slice,
    rootvel_sl: slice,
    out_dim: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
    out_arr = probe.arrays.get("out__out")
    if out_arr is None:
        return None, None, None, "missing out__out"
    state_raw = _denorm_state(sample, mu_x, std_x)
    if int(cut) <= 0 or int(cut) >= int(state_raw.shape[0]):
        return None, None, None, "cut outside state sequence"

    device = next(trainer.model.parameters()).device if hasattr(trainer, "model") else torch.device("cpu")
    dtype = torch.float32
    y_prev_np = np.zeros((int(out_dim),), dtype=np.float32)
    rot_len = min(int(rot_x_sl.stop - rot_x_sl.start), int(out_dim))
    y_prev_np[:rot_len] = state_raw[int(cut) - 1, rot_x_sl][:rot_len]
    rv_len = min(int(rootvel_sl.stop - rootvel_sl.start), int(ROOT_VEL_OUT_SLICE.stop - ROOT_VEL_OUT_SLICE.start))
    if int(out_dim) >= int(ROOT_VEL_OUT_SLICE.stop) and rv_len > 0:
        y_prev_np[ROOT_VEL_OUT_SLICE.start : ROOT_VEL_OUT_SLICE.start + rv_len] = state_raw[
            int(cut) - 1,
            rootvel_sl,
        ][:rv_len]
    y_prev = torch.as_tensor(y_prev_np, dtype=dtype, device=device).unsqueeze(0)

    raw_steps: List[torch.Tensor] = []
    out_norm = torch.as_tensor(out_arr[int(cut) :], dtype=dtype, device=device)
    with torch.no_grad():
        for i in range(int(out_norm.shape[0])):
            delta_norm = out_norm[i].unsqueeze(0)
            try:
                y_raw = trainer._compose_delta_to_raw(y_prev, delta_norm)
            except Exception:
                y_raw = trainer._denorm(delta_norm)
            raw_steps.append(y_raw[0].detach().cpu())
            y_prev = y_raw.detach()
    if not raw_steps:
        return None, None, None, "empty post-cut readout"
    y_raw = torch.stack(raw_steps, dim=0).float()

    contacts_arr = probe.arrays.get("out__contacts_plan")
    if contacts_arr is None or int(contacts_arr.shape[0]) < int(cut) + 1:
        contacts_arr = sample["contacts"].detach().cpu().float().numpy()[0]
    contacts = torch.as_tensor(contacts_arr[int(cut) : int(cut) + int(y_raw.shape[0])], dtype=torch.float32).clamp(0.0, 1.0)

    cond_arr = sample["cond"].detach().cpu().float().numpy()[0]
    if cond_arr.shape[1] >= COMBO_COND_DIR_SLICE.stop:
        cond_dir = cond_arr[int(cut) : int(cut) + int(y_raw.shape[0]), COMBO_COND_DIR_SLICE]
    else:
        cond_dir = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (int(y_raw.shape[0]), 1))
    cond_dir_t = torch.as_tensor(cond_dir, dtype=torch.float32)
    return y_raw, contacts, cond_dir_t, None


def _yaw_rate_from_dir(arr: np.ndarray, *, fps: float) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return np.zeros((0,), dtype=np.float64)
    heading = np.arctan2(x[:, 1], x[:, 0])
    d = np.diff(heading)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    return d * float(fps)


def _realized_yaw_rate_from_rootvel(rootvel: np.ndarray, *, fps: float) -> np.ndarray:
    rv = np.asarray(rootvel, dtype=np.float64)
    if rv.ndim != 2 or rv.shape[0] < 2 or rv.shape[1] < 2:
        return np.zeros((0,), dtype=np.float64)
    speed = np.linalg.norm(rv[:, :2], axis=1)
    valid = speed > 1e-5
    heading = np.arctan2(rv[:, 1], rv[:, 0])
    d = np.diff(heading)
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    rate = d * float(fps)
    keep = valid[1:] & valid[:-1]
    return rate[keep]


def _seam_delta_stats(arr: np.ndarray, *, cut: int, per_joint_width: int = 3) -> Dict[str, Any]:
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2 or not (0 < int(cut) < int(x.shape[0])):
        return {
            "delta_rms": None,
            "delta_p95": None,
            "top5": [],
        }
    d = x[int(cut)] - x[int(cut) - 1]
    rms = float(np.linalg.norm(d.reshape(-1)) / math.sqrt(max(1, d.size)))
    per = np.linalg.norm(d.reshape(-1, int(per_joint_width)), axis=1)
    order = np.argsort(-per)[:5]
    return {
        "delta_rms": rms,
        "delta_p95": float(np.percentile(per, 95)) if per.size else None,
        "top5": [{"joint_index": int(i), "delta_rad_s": float(per[int(i)])} for i in order.tolist()],
    }


def _motion_safe_v2(
    *,
    trainer: Any,
    sample: Mapping[str, torch.Tensor],
    probe: ForwardProbeResult,
    goal_seam: np.ndarray,
    normalizer: StateNormalizer,
    gate_thresholds: GateThresholds,
    mu_x: np.ndarray,
    std_x: np.ndarray,
    rot_x_sl: slice,
    rootvel_sl: slice,
    state_angvel_sl: slice,
    cut: int,
    out_dim: int,
    skeleton_meta: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    state_raw = _denorm_state(sample, mu_x, std_x)
    ang = state_raw[:, state_angvel_sl]
    bone = _seam_delta_stats(ang, cut=int(cut), per_joint_width=3)
    pose = _seam_delta_stats(state_raw[:, rot_x_sl], cut=int(cut), per_joint_width=6)
    ego = state_raw[:, rootvel_sl]
    contact_in = sample["contacts"].detach().cpu().float().numpy()[0]
    ego_delta = None
    contact_delta = None
    if 0 < int(cut) < ego.shape[0]:
        ego_delta = float(np.linalg.norm(ego[int(cut)] - ego[int(cut) - 1]))
    if 0 < int(cut) < contact_in.shape[0]:
        contact_delta = float(np.linalg.norm(contact_in[int(cut)] - contact_in[int(cut) - 1]))

    y_raw, contacts, cond_dir, readout_skip = _readout_raw_from_probe(
        trainer,
        sample,
        probe,
        cut=int(cut),
        mu_x=mu_x,
        std_x=std_x,
        rot_x_sl=rot_x_sl,
        rootvel_sl=rootvel_sl,
        out_dim=int(out_dim),
    )
    if readout_skip is not None or y_raw is None or contacts is None or cond_dir is None:
        return {
            "status": "unavailable",
            "reason": readout_skip,
            "bone_angvel_delta_rms_rad_s": bone["delta_rms"],
            "bone_angvel_delta_p95_rad_s": bone["delta_p95"],
            "top5_angvel_delta_joints": bone["top5"],
            "ego_velocity_delta_l2": ego_delta,
            "contact_delta_l2": contact_delta,
        }

    foot = _foot_slip_summary(
        trainer,
        y_raw,
        contacts,
        cond_dir,
        fps=FPS,
        skeleton_meta=skeleton_meta,
    )
    foot_p95 = _mean_finite(
        [
            foot.get("right", {}).get("p95_mps") if isinstance(foot.get("right"), Mapping) else None,
            foot.get("left", {}).get("p95_mps") if isinstance(foot.get("left"), Mapping) else None,
        ]
    )

    cond_np = cond_dir.detach().cpu().float().numpy()
    cmd_yaw = _yaw_rate_from_dir(cond_np, fps=FPS)
    rv_np = y_raw[:, ROOT_VEL_OUT_SLICE].detach().cpu().float().numpy()
    real_yaw = _realized_yaw_rate_from_rootvel(rv_np, fps=FPS)
    n_yaw = min(int(cmd_yaw.shape[0]), int(real_yaw.shape[0]))
    yaw_abs = np.abs(real_yaw[:n_yaw] - cmd_yaw[:n_yaw]) if n_yaw > 0 else np.zeros((0,), dtype=np.float64)

    try:
        roll_state = rollout_to_egocentric(
            y_raw[:, :276].detach().cpu().numpy(),
            y_raw[:, ROOT_VEL_OUT_SLICE].detach().cpu().numpy(),
            cond_np,
            contacts.detach().cpu().numpy(),
            fps=FPS,
        )
        old_gate = evaluate_rollout_state_space(
            roll_state,
            np.asarray(goal_seam, dtype=np.float32),
            normalizer.std,
            gate_thresholds,
        )
    except Exception as exc:
        old_gate = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}", "pop_safe": False}

    return {
        "status": "ok",
        "motion_safe_v2_pass": None,
        "pose_delta_rms_rot6d": pose["delta_rms"],
        "pose_delta_p95_rot6d_joint": pose["delta_p95"],
        "bone_angvel_delta_rms_rad_s": bone["delta_rms"],
        "bone_angvel_delta_p95_rad_s": bone["delta_p95"],
        "top5_angvel_delta_joints": bone["top5"],
        "fk_foot_slip_status": foot.get("status"),
        "fk_foot_slip_mean_mps": foot.get("mean_mps_over_sides"),
        "fk_foot_slip_p95_mps": foot_p95,
        "fk_foot_slip_max_mps": foot.get("max_mps_over_sides"),
        "realized_yaw_rate_deviation_mean_deg_s": float(np.mean(yaw_abs) * 180.0 / math.pi) if yaw_abs.size else None,
        "realized_yaw_rate_deviation_p95_deg_s": float(np.percentile(yaw_abs, 95) * 180.0 / math.pi) if yaw_abs.size else None,
        "ego_velocity_delta_l2": ego_delta,
        "contact_delta_l2": contact_delta,
        "old_pop_safe": bool(old_gate.get("pop_safe", False)),
        "old_pop": old_gate.get("pop"),
        "old_best_pose_d": old_gate.get("best_pose_d"),
        "readout_contract": {
            "y_raw": {"shape": [int(x) for x in y_raw.shape], "dtype": "float32", "device": "cpu"},
            "contacts": {"shape": [int(x) for x in contacts.shape], "dtype": "float32", "device": "cpu"},
            "cond_dir": {"shape": [int(x) for x in cond_dir.shape], "dtype": "float32", "device": "cpu"},
        },
    }


def _calibrate_motion_safe(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    multiplier: float,
) -> Tuple[Dict[str, Optional[float]], List[Dict[str, Any]]]:
    bases = [
        r for r in rows
        if str(r.get("target")) == str(target) and str(r.get("variant")) in {"walk_continuous", "target_continuous"}
    ]
    keys = [
        "pose_delta_rms_rot6d",
        "bone_angvel_delta_rms_rad_s",
        "fk_foot_slip_mean_mps",
        "realized_yaw_rate_deviation_mean_deg_s",
        "ego_velocity_delta_l2",
        "contact_delta_l2",
    ]
    thresholds: Dict[str, Optional[float]] = {}
    for key in keys:
        base = _mean_finite([r.get(key) for r in bases])
        thresholds[key] = float(base) * float(multiplier) if base is not None else None

    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("target")) != str(target):
            continue
        r = dict(row)
        checks: Dict[str, Optional[bool]] = {}
        for key, thr in thresholds.items():
            val = r.get(key)
            if thr is None or val is None:
                checks[key] = None
            else:
                checks[key] = bool(float(val) <= float(thr))
        valid = [v for v in checks.values() if v is not None]
        r["motion_safe_v2_thresholds"] = thresholds
        r["motion_safe_v2_checks"] = checks
        r["motion_safe_v2_pass"] = bool(valid) and all(valid)
        out.append(r)
    return thresholds, out


def _fit_ridge_predictor(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: Sequence[float],
) -> Tuple[Dict[str, Any], Optional[RidgePredictor]]:
    if int(train_idx.size) < 2 or int(test_idx.size) < 1:
        return {"status": "insufficient_pairs", "n_train": int(train_idx.size), "n_test": int(test_idx.size)}, None
    Xtr, Xte, mu, std = _standardize_train_test(X, train_idx, test_idx)
    Ytr = np.asarray(Y[train_idx], dtype=np.float64)
    Yte = np.asarray(Y[test_idx], dtype=np.float64)
    Xtr_i = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1), dtype=np.float64)], axis=1)
    Xte_i = np.concatenate([Xte, np.ones((Xte.shape[0], 1), dtype=np.float64)], axis=1)
    eye = np.eye(int(Xtr_i.shape[1]), dtype=np.float64)
    eye[-1, -1] = 0.0
    best: Optional[Tuple[Dict[str, Any], np.ndarray, np.ndarray]] = None
    for alpha in alphas:
        a = float(alpha)
        try:
            coef = np.linalg.solve(Xtr_i.T @ Xtr_i + a * eye, Xtr_i.T @ Ytr)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(Xtr_i.T @ Xtr_i + a * eye) @ Xtr_i.T @ Ytr
        pred_te = Xte_i @ coef
        pred_tr = Xtr_i @ coef
        rec = {
            "alpha": a,
            "r2": _r2_score(Yte, pred_te),
            "rmse": float(np.sqrt(np.mean((Yte - pred_te) ** 2))),
            "mae": float(np.mean(np.abs(Yte - pred_te))),
            "train_r2": _r2_score(Ytr, pred_tr),
            "train_rmse": float(np.sqrt(np.mean((Ytr - pred_tr) ** 2))),
        }
        if best is None or (math.isfinite(float(rec["r2"])) and float(rec["r2"]) > float(best[0]["r2"])):
            best = (rec, coef, pred_te)
    assert best is not None
    rec, coef, pred_te = best
    predictor = RidgePredictor(
        alpha=float(rec["alpha"]),
        x_mu=mu.astype(np.float64),
        x_std=std.astype(np.float64),
        coef=coef.astype(np.float64),
        train_r2=float(rec["train_r2"]),
        test_r2=float(rec["r2"]),
        train_rmse=float(rec["train_rmse"]),
        test_rmse=float(rec["rmse"]),
    )
    return {
        "status": "ok",
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "input_dim": int(X.shape[1]),
        "output_dim": int(Y.shape[1]),
        "best": rec,
    }, predictor


def _one_hot(index: int, n: int) -> np.ndarray:
    out = np.zeros((int(n),), dtype=np.float32)
    out[int(index)] = 1.0
    return out


def _build_mapping_feature(
    *,
    group: str,
    walk_state281: np.ndarray,
    walk_hidden: np.ndarray,
    walk_base_angvel: np.ndarray,
    target_oh: np.ndarray,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    if group in {"state281", "state281_angvel", "combo"}:
        parts.append(np.asarray(walk_state281, dtype=np.float32).reshape(-1))
    if group in {"hidden", "combo"}:
        parts.append(np.asarray(walk_hidden, dtype=np.float32).reshape(-1))
    if group in {"state281_angvel", "combo"}:
        parts.append(np.asarray(walk_base_angvel, dtype=np.float32).reshape(-1))
    parts.append(np.asarray(target_oh, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _build_angvel_mapping_matrices(
    *,
    map_rows: Sequence[Mapping[str, Any]],
    states_281: Mapping[str, np.ndarray],
    hidden: Mapping[str, np.ndarray],
    base_angvel_by_clip: Mapping[str, np.ndarray],
    target_clips: Sequence[str],
    group: str,
) -> Tuple[np.ndarray, np.ndarray]:
    clip_to_idx = {clip: i for i, clip in enumerate(target_clips)}
    X: List[np.ndarray] = []
    Y: List[np.ndarray] = []
    for row in map_rows:
        clip = str(row["clip"])
        phi = int(row["walk_phi"])
        tgt_t = int(row["target_frame"])
        oh = _one_hot(clip_to_idx[clip], len(target_clips))
        X.append(
            _build_mapping_feature(
                group=group,
                walk_state281=states_281[WALK_F][phi],
                walk_hidden=hidden[WALK_F][phi],
                walk_base_angvel=base_angvel_by_clip[WALK_F][phi],
                target_oh=oh,
            )
        )
        Y.append(base_angvel_by_clip[clip][tgt_t])
    if not X:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    return np.stack(X, axis=0).astype(np.float32), np.stack(Y, axis=0).astype(np.float32)


def _motion_rows_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "n": int(len(rows)),
        "pass_rate": _mean_finite([1.0 if bool(r.get("motion_safe_v2_pass")) else 0.0 for r in rows]),
        "old_pop_safe_rate": _mean_finite([1.0 if bool(r.get("old_pop_safe")) else 0.0 for r in rows]),
        "bone_angvel_delta_rms_rad_s_mean": _mean_finite([r.get("bone_angvel_delta_rms_rad_s") for r in rows]),
        "bone_angvel_delta_p95_rad_s_mean": _mean_finite([r.get("bone_angvel_delta_p95_rad_s") for r in rows]),
        "fk_foot_slip_mean_mps_mean": _mean_finite([r.get("fk_foot_slip_mean_mps") for r in rows]),
        "fk_foot_slip_p95_mps_mean": _mean_finite([r.get("fk_foot_slip_p95_mps") for r in rows]),
        "fk_foot_slip_max_mps_mean": _mean_finite([r.get("fk_foot_slip_max_mps") for r in rows]),
        "realized_yaw_rate_deviation_mean_deg_s_mean": _mean_finite(
            [r.get("realized_yaw_rate_deviation_mean_deg_s") for r in rows]
        ),
        "ego_velocity_delta_l2_mean": _mean_finite([r.get("ego_velocity_delta_l2") for r in rows]),
        "contact_delta_l2_mean": _mean_finite([r.get("contact_delta_l2") for r in rows]),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only bone-angular-velocity bridge probe.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--pre-frames", type=int, default=16)
    p.add_argument("--post-frames", type=int, default=24)
    p.add_argument("--onset-scan", type=int, default=8)
    p.add_argument("--target-clips", type=str, default=",".join(TURN_CLIPS))
    p.add_argument("--pose-topk", type=int, default=POSE_TOPK)
    p.add_argument("--ground-contact-thr", type=float, default=GROUND_CONTACT_THR)
    p.add_argument("--ground-pose-thr", type=float, default=GROUND_POSE_THR)
    p.add_argument("--k-frames", type=str, default="1,2,3,4,5,6")
    p.add_argument("--mapping-k-frames", type=str, default="3,4")
    p.add_argument("--mapping-pose-thr", type=float, default=0.08)
    p.add_argument("--mapping-contact-thr", type=float, default=0.30)
    p.add_argument("--mapping-test-frac", type=float, default=0.25)
    p.add_argument("--motion-safe-threshold-mult", type=float, default=2.0)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    npz_root = Path(args.npz_root)
    z_features = Path(args.z_features)
    target_clips = _parse_clips(args.target_clips)
    k_values = _parse_int_list(args.k_frames, name="--k-frames")
    mapping_k_values = _parse_int_list(args.mapping_k_frames, name="--mapping-k-frames")
    if not z_features.exists():
        raise FileNotFoundError(f"z-features not found: {z_features}")
    if not npz_root.exists():
        raise FileNotFoundError(f"npz root not found: {npz_root}")
    if not Path(args.checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(f"debug_output/_tmp_action_handoff_bone_angvel_bridge_probe_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = freerun.FreeRunCycleRunner(_make_runner_args(args))
    seq_len = max(64, int(args.pre_frames) + int(args.post_frames))
    walk_clip = _load_clip(runner, npz_root, WALK_F, seq_len=seq_len)
    runner.model.eval()
    model = runner.model

    norm_spec = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    mu_x = np.asarray(norm_spec["MuX"], dtype=np.float32)
    std_x = np.asarray(norm_spec["StdX"], dtype=np.float32)
    std_x = np.where(np.abs(std_x) > 1e-8, std_x, 1.0).astype(np.float32)

    walk_raw = _load_npz_raw(npz_root, WALK_F)
    rootpos_sl = _layout_slice(walk_clip.state_layout_norm, ROOT_POS_KEY, fallback=slice(0, 3))
    rootvel_sl = _layout_slice(walk_clip.state_layout_norm, ROOT_VEL_KEY, fallback=slice(3, 5))
    rot_x_sl = _layout_slice(walk_clip.state_layout_norm, ROT6D_KEY, fallback=slice(5, 281))
    state_angvel_sl = _layout_slice(walk_clip.state_layout_norm, ANGVEL_KEY, fallback=slice(281, 419))
    rot_y_sl = _layout_slice(walk_clip.output_layout_norm, ROT6D_KEY, fallback=slice(0, 276))
    dims = {
        "state": int(getattr(model, "in_state_dim", walk_clip.X.shape[1])),
        "cond": int(getattr(model, "cond_dim", walk_clip.C.shape[1])),
        "contact": int(getattr(model, "contact_dim", 0) or 0),
        "angvel": int(getattr(model, "angvel_dim", 0) or 0),
        "pose_hist": int(getattr(model, "pose_hist_dim", 0) or 0),
        "out": int(getattr(model, "out_motion_dim", walk_clip.Y.shape[1])),
    }
    slices = {"rootpos_x": rootpos_sl, "rootvel_x": rootvel_sl, "rot_x": rot_x_sl, "rot_y": rot_y_sl}

    states_281 = load_clip_states(z_features, npz_root)
    hidden = _load_hidden_pre(z_features, [WALK_F, *target_clips])
    raw_angvel = _load_raw_angvel(npz_root, [WALK_F, *target_clips])
    base_angvel_by_clip: Dict[str, np.ndarray] = {}
    for clip in [WALK_F, *target_clips]:
        raw = _load_npz_raw(npz_root, clip)
        x_raw = np.asarray(raw["x_raw"], dtype=np.float32)
        base_angvel_by_clip[clip] = (
            (x_raw[:, state_angvel_sl] - mu_x[state_angvel_sl].reshape(1, -1))
            / std_x[state_angvel_sl].reshape(1, -1)
        ).astype(np.float32)

    try:
        from tools.run_action_handoff_regime_bridge_probe import _load_skeleton_meta  # noqa: WPS433

        skeleton_meta = _load_skeleton_meta(npz_root, WALK_F)
    except Exception:
        skeleton_meta = None

    normalizer = StateNormalizer(states_281)
    gate_thresholds = GateThresholds()

    payload: Dict[str, Any] = {
        "task": "action_handoff_bone_angvel_bridge_probe",
        "scope": "read-only k-frame bone-angvel ramp + motion_safe_v2 + ridge mapping; no base training",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "z_features": str(z_features.resolve()),
        "npz_root": str(npz_root.resolve()),
        "config": {
            "pre_frames": int(args.pre_frames),
            "post_frames": int(args.post_frames),
            "cut_step": int(args.pre_frames),
            "target_clips": list(target_clips),
            "device": str(args.device),
            "k_frames": [int(k) for k in k_values],
            "mapping_k_frames": [int(k) for k in mapping_k_values],
            "mapping_pose_thr": float(args.mapping_pose_thr),
            "mapping_contact_thr": float(args.mapping_contact_thr),
            "motion_safe_threshold_mult": float(args.motion_safe_threshold_mult),
        },
        "code_refs": {
            "canonical_281_schema": "train/data/action_handoff_inbetween.py:9",
            "canonical_281_dim": "train/data/action_handoff_inbetween.py:45",
            "base_trunk_concat": "train/models.py:5020",
            "base_angvel_side_input": "train/models.py:5001",
            "direct_readout_concat": "train/models.py:5260",
            "matched_sample_contract": "tools/run_action_handoff_matched_seam_neuron_audit.py:455",
        },
        "input_contract": {
            "base_model_forward_tensors": {
                "state": {
                    "shape": [1, int(args.pre_frames) + int(args.post_frames), dims["state"]],
                    "dtype": "float32",
                    "device_before_forward": "cpu",
                },
                "cond": {
                    "shape": [1, int(args.pre_frames) + int(args.post_frames), dims["cond"]],
                    "dtype": "float32",
                    "device_before_forward": "cpu",
                },
                "contacts": {
                    "shape": [1, int(args.pre_frames) + int(args.post_frames), dims["contact"]],
                    "dtype": "float32",
                    "device_before_forward": "cpu",
                },
                "angvel": {
                    "shape": [1, int(args.pre_frames) + int(args.post_frames), dims["angvel"]],
                    "dtype": "float32",
                    "device_before_forward": "cpu",
                },
                "pose_history": {
                    "shape": [1, int(args.pre_frames) + int(args.post_frames), dims["pose_hist"]],
                    "dtype": "float32",
                    "device_before_forward": "cpu",
                },
            },
            "base_normalized_state_layout": {
                "rootpos_x": _slice_bounds(rootpos_sl),
                "rootvel_x": _slice_bounds(rootvel_sl),
                "rot6d_x": _slice_bounds(rot_x_sl),
                "bone_angvel_x": _slice_bounds(state_angvel_sl),
            },
            "canonical_action_handoff_state_281": {
                "shape": ["T", STATE_DIM],
                "dtype": "float32",
                "device": "cpu",
                "layout": {
                    "pose_rot6d": _slice_bounds(POSE_SLICE),
                    "ego_vel": _slice_bounds(EGO_VEL_SLICE),
                    "yaw_rate": _slice_bounds(YAW_RATE_SLICE),
                    "contact": _slice_bounds(CONTACT_SLICE),
                },
                "contains_bone_angvel": False,
            },
        },
        "model_flags": {
            "contact_plan_enable": bool(getattr(model, "contact_plan_enable", False)),
            "contact_plan_inject": str(getattr(model, "contact_plan_inject", "")),
            "direct_pose_enable": bool(getattr(model, "direct_pose_enable", False)),
            "direct_pose_feat_source": str(getattr(model, "direct_pose_feat_source", "")),
            "lambda_fusion_enable": bool(getattr(model, "lambda_fusion_enable", False)),
            "lambda_fusion_mode": str(getattr(model, "lambda_fusion_mode", "")),
        },
        "A_k_frame_ramp_causality": {"per_target": {}, "aggregate": {}},
        "B_motion_safe_v2": {"per_target": {}, "aggregate": {}},
        "C_mapping_probe": {"learnability": {}, "readout_motion_safe_v2": {}},
        "conclusion": {},
    }

    selected_meta: Dict[str, Dict[str, Any]] = {}
    ramp_rows_all: List[Dict[str, Any]] = []
    motion_rows_all_raw: List[Dict[str, Any]] = []
    stored_samples: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
    stored_probes: Dict[Tuple[str, str], ForwardProbeResult] = {}

    for target in target_clips:
        cand = _candidate_onset(
            states_281[WALK_F],
            states_281[target],
            onset_scan=int(args.onset_scan),
            pose_topk=int(args.pose_topk),
            contact_thr=float(args.ground_contact_thr),
            pose_thr=float(args.ground_pose_thr),
        )
        selected = cand.get("selected") if isinstance(cand, dict) else None
        target_payload: Dict[str, Any] = {"alignment": cand, "variants": []}
        motion_target_rows: List[Dict[str, Any]] = []
        if not selected:
            target_payload["skip_reason"] = "no groundable matched onset in scan window"
            payload["A_k_frame_ramp_causality"]["per_target"][target] = target_payload
            payload["B_motion_safe_v2"]["per_target"][target] = {"skip_reason": target_payload["skip_reason"]}
            continue

        phi = int(selected["phi"])
        onset = int(selected["onset"])
        selected_meta[target] = {"phi": phi, "onset": onset, "alignment": selected}
        target_clip = _load_clip(runner, npz_root, target, seq_len=seq_len)
        target_raw = _load_npz_raw(npz_root, target)

        matched_sample, matched_meta = _make_sample(
            case="matched_positive_xhist",
            walk_clip=walk_clip,
            target_clip=target_clip,
            walk_raw=walk_raw,
            target_raw=target_raw,
            phi=phi,
            onset=onset,
            pre=int(args.pre_frames),
            post=int(args.post_frames),
            dims=dims,
            slices=slices,
            mu_x=mu_x,
            std_x=std_x,
            norm_spec=norm_spec,
            cond_ramp_frames=8,
        )
        walk_sample, _ = _make_sample(
            case="walk_continuous",
            walk_clip=walk_clip,
            target_clip=target_clip,
            walk_raw=walk_raw,
            target_raw=target_raw,
            phi=phi,
            onset=onset,
            pre=int(args.pre_frames),
            post=int(args.post_frames),
            dims=dims,
            slices=slices,
            mu_x=mu_x,
            std_x=std_x,
            norm_spec=norm_spec,
            cond_ramp_frames=8,
        )
        target_sample_tf, _ = _make_sample(
            case="target_continuous",
            walk_clip=walk_clip,
            target_clip=target_clip,
            walk_raw=walk_raw,
            target_raw=target_raw,
            phi=phi,
            onset=onset,
            pre=int(args.pre_frames),
            post=int(args.post_frames),
            dims=dims,
            slices=slices,
            mu_x=mu_x,
            std_x=std_x,
            norm_spec=norm_spec,
            cond_ramp_frames=8,
        )

        base_variants = _build_variant_samples(
            matched_sample,
            walk_sample,
            cut=int(args.pre_frames),
            rootpos_sl=rootpos_sl,
            rootvel_sl=rootvel_sl,
            rot_x_sl=rot_x_sl,
            state_angvel_sl=state_angvel_sl,
        )
        ramp_variants = _build_k_ramp_samples(
            matched_sample,
            walk_sample,
            cut=int(args.pre_frames),
            k_values=k_values,
            rootvel_sl=rootvel_sl,
            state_angvel_sl=state_angvel_sl,
        )
        samples: Dict[str, Dict[str, torch.Tensor]] = {
            "walk_continuous": walk_sample,
            "target_continuous": target_sample_tf,
            "matched_base": base_variants["matched_base"],
            "state_all_to_walk": base_variants["state_all_to_walk"],
            "x_state_angvel_to_walk": base_variants["x_state_angvel_to_walk"],
            **ramp_variants,
        }
        probes: Dict[str, ForwardProbeResult] = {}
        for name, sample in samples.items():
            probes[name] = _forward_probe(
                model,
                sample,
                device=runner.device,
                cut_step=int(args.pre_frames),
                topk_dims=8,
                state_dim=dims["state"],
                cond_dim=dims["cond"],
            )
            stored_samples[(target, name)] = sample
            stored_probes[(target, name)] = probes[name]

        for name, probe in probes.items():
            row = _summarize_probe_row(
                target=target,
                variant=name,
                probe=probe,
                base_probe=probes["matched_base"],
                walk_probe=probes["walk_continuous"],
            )
            target_payload["variants"].append(row)
            ramp_rows_all.append(row)

        goal_seam = states_281[target][onset : onset + SEAM_LEN_K]
        for name, sample in samples.items():
            ms = _motion_safe_v2(
                trainer=runner.trainer,
                sample=sample,
                probe=probes[name],
                goal_seam=goal_seam,
                normalizer=normalizer,
                gate_thresholds=gate_thresholds,
                mu_x=mu_x,
                std_x=std_x,
                rot_x_sl=rot_x_sl,
                rootvel_sl=rootvel_sl,
                state_angvel_sl=state_angvel_sl,
                cut=int(args.pre_frames),
                out_dim=dims["out"],
                skeleton_meta=skeleton_meta,
            )
            motion_target_rows.append({"target": target, "variant": name, **ms})

        _, calibrated_rows = _calibrate_motion_safe(
            motion_target_rows,
            target=target,
            multiplier=float(args.motion_safe_threshold_mult),
        )
        motion_rows_all_raw.extend(calibrated_rows)

        target_payload["matched_meta"] = {
            "phi": phi,
            "onset": onset,
            "pose_d": float(selected["pose_d"]),
            "contact_d": float(selected["contact_d"]),
            "rootvel_norm_step_l2_at_cut": matched_meta.get("x_rootvel_norm_step_l2_at_cut"),
            "history_rot6d_step_l2_at_cut": matched_meta.get("history_rot6d_step_l2_at_cut"),
            "contact_step_l2_at_cut": matched_meta.get("contact_step_l2_at_cut"),
            "cond_step_l2_at_cut": matched_meta.get("cond_step_l2_at_cut"),
        }
        payload["A_k_frame_ramp_causality"]["per_target"][target] = target_payload
        payload["B_motion_safe_v2"]["per_target"][target] = {
            "phi": phi,
            "onset": onset,
            "rows": calibrated_rows,
            "summary_by_variant": {
                variant: _motion_rows_summary([r for r in calibrated_rows if str(r.get("variant")) == variant])
                for variant in sorted({str(r.get("variant")) for r in calibrated_rows})
            },
        }

    payload["A_k_frame_ramp_causality"]["aggregate"] = {
        "mean_by_variant": _aggregate_by_variant(ramp_rows_all),
    }
    payload["B_motion_safe_v2"]["aggregate"] = {
        "summary_by_variant": {
            variant: _motion_rows_summary([r for r in motion_rows_all_raw if str(r.get("variant")) == variant])
            for variant in sorted({str(r.get("variant")) for r in motion_rows_all_raw})
        },
        "definition": (
            "motion_safe_v2 is local to this read-only probe: pose continuity, de-normalized "
            "bone_angvel continuity, FK foot-slip under predicted contacts, realized root-velocity "
            "yaw-rate deviation from the commanded cond-dir path, ego/root velocity continuity, "
            "and contact continuity. old_pop_safe is reported only as a comparator."
        ),
    }

    # --------------------------------------------------------------- ridge mapping
    Xh, Xs, _Yh, map_rows = _make_mapping_pairs(
        states=states_281,
        hidden=hidden,
        target_clips=target_clips,
        pose_thr=float(args.mapping_pose_thr),
        contact_thr=float(args.mapping_contact_thr),
        pose_topk=int(args.pose_topk),
    )
    del Xh, Xs
    train_idx, test_idx = _split_indices(map_rows, seed=int(args.seed), test_frac=float(args.mapping_test_frac))
    alphas = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    mapping_payload: Dict[str, Any] = {
        "pair_definition": (
            "for each target frame, pick Walk_F full_state_align frame; keep pairs with "
            f"pose_d<={float(args.mapping_pose_thr)} and contact_d<={float(args.mapping_contact_thr)}"
        ),
        "n_pairs": int(len(map_rows)),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "groups": {},
        "selected_group_for_readout": None,
        "honesty_note": (
            "R2 is only low-complexity learnability evidence. A mapping is not counted as a "
            "bridge success unless predicted angvel also passes motion_safe_v2."
        ),
    }
    predictors: Dict[str, RidgePredictor] = {}
    for group in ("state281", "hidden", "state281_angvel", "combo"):
        Xg, Yg = _build_angvel_mapping_matrices(
            map_rows=map_rows,
            states_281=states_281,
            hidden=hidden,
            base_angvel_by_clip=base_angvel_by_clip,
            target_clips=target_clips,
            group=group,
        )
        result, predictor = _fit_ridge_predictor(
            Xg,
            Yg,
            train_idx=train_idx,
            test_idx=test_idx,
            alphas=alphas,
        )
        mapping_payload["groups"][group] = result
        if predictor is not None:
            predictors[group] = predictor
    usable = [
        (g, p.test_r2)
        for g, p in predictors.items()
        if math.isfinite(float(p.test_r2))
    ]
    usable.sort(key=lambda x: x[1], reverse=True)
    selected_group = usable[0][0] if usable else None
    mapping_payload["selected_group_for_readout"] = selected_group
    payload["C_mapping_probe"]["learnability"] = mapping_payload

    mapping_motion_rows: List[Dict[str, Any]] = []
    if selected_group is not None:
        predictor = predictors[selected_group]
        for target, meta in selected_meta.items():
            target_idx = target_clips.index(target)
            oh = _one_hot(target_idx, len(target_clips))
            phi = int(meta["phi"])
            onset = int(meta["onset"])
            goal_seam = states_281[target][onset : onset + SEAM_LEN_K]
            matched_sample = stored_samples[(target, "matched_base")]
            walk_sample = stored_samples[(target, "walk_continuous")]
            walk_len = int(base_angvel_by_clip[WALK_F].shape[0])
            max_k = max(mapping_k_values)
            feats: List[np.ndarray] = []
            for j in range(max_k):
                widx = (phi + j) % walk_len
                feats.append(
                    _build_mapping_feature(
                        group=selected_group,
                        walk_state281=states_281[WALK_F][widx],
                        walk_hidden=hidden[WALK_F][min(widx, hidden[WALK_F].shape[0] - 1)],
                        walk_base_angvel=base_angvel_by_clip[WALK_F][widx],
                        target_oh=oh,
                    )
                )
            pred = predictor.predict(np.stack(feats, axis=0))
            for k in mapping_k_values:
                kk = int(k)
                sample = _clone_sample(matched_sample)
                target_values = torch.as_tensor(pred[:kk], dtype=sample["state"].dtype).view(1, kk, -1)
                sample["state"] = _ramp_post_slice(
                    matched_sample,
                    walk_sample,
                    field="state",
                    sl=state_angvel_sl,
                    cut=int(args.pre_frames),
                    frames=kk,
                    target_values=target_values,
                )
                variant = f"mapping_{selected_group}_bone_angvel_ramp_k{kk}"
                probe = _forward_probe(
                    model,
                    sample,
                    device=runner.device,
                    cut_step=int(args.pre_frames),
                    topk_dims=8,
                    state_dim=dims["state"],
                    cond_dim=dims["cond"],
                )
                row = _summarize_probe_row(
                    target=target,
                    variant=variant,
                    probe=probe,
                    base_probe=stored_probes[(target, "matched_base")],
                    walk_probe=stored_probes[(target, "walk_continuous")],
                )
                ms = _motion_safe_v2(
                    trainer=runner.trainer,
                    sample=sample,
                    probe=probe,
                    goal_seam=goal_seam,
                    normalizer=normalizer,
                    gate_thresholds=gate_thresholds,
                    mu_x=mu_x,
                    std_x=std_x,
                    rot_x_sl=rot_x_sl,
                    rootvel_sl=rootvel_sl,
                    state_angvel_sl=state_angvel_sl,
                    cut=int(args.pre_frames),
                    out_dim=dims["out"],
                    skeleton_meta=skeleton_meta,
                )
                mapping_motion_rows.append({"target": target, **row, **ms})
        calibrated_mapping: List[Dict[str, Any]] = []
        for target in selected_meta:
            _, target_rows = _calibrate_motion_safe(
                [r for r in mapping_motion_rows if str(r.get("target")) == target]
                + [r for r in motion_rows_all_raw if str(r.get("target")) == target and str(r.get("variant")) in {"walk_continuous", "target_continuous"}],
                target=target,
                multiplier=float(args.motion_safe_threshold_mult),
            )
            calibrated_mapping.extend([r for r in target_rows if str(r.get("variant", "")).startswith("mapping_")])
        mapping_motion_rows = calibrated_mapping

    payload["C_mapping_probe"]["readout_motion_safe_v2"] = {
        "rows": mapping_motion_rows,
        "summary_by_variant": {
            variant: _motion_rows_summary([r for r in mapping_motion_rows if str(r.get("variant")) == variant])
            for variant in sorted({str(r.get("variant")) for r in mapping_motion_rows})
        },
    }

    agg = payload["A_k_frame_ramp_causality"]["aggregate"]["mean_by_variant"]
    bone_k34 = [
        row.get("hidden_pre_collapse_fraction_to_walk_mean")
        for name, row in agg.items()
        if name in {"bone_angvel_ramp_k3", "bone_angvel_ramp_k4"}
    ]
    combo_k34 = [
        row.get("hidden_pre_collapse_fraction_to_walk_mean")
        for name, row in agg.items()
        if name in {"bone_angvel_rootvel_cmdyaw_ramp_k3", "bone_angvel_rootvel_cmdyaw_ramp_k4"}
    ]
    motion_summary = payload["B_motion_safe_v2"]["aggregate"]["summary_by_variant"]
    mapping_summary = payload["C_mapping_probe"]["readout_motion_safe_v2"]["summary_by_variant"]
    payload["conclusion"] = {
        "supports_bone_angvel_as_primary_bridge_variable": (
            _mean_finite(bone_k34) is not None and float(_mean_finite(bone_k34)) >= 0.7
        ),
        "bone_angvel_k3k4_hidden_pre_collapse_mean": _mean_finite(bone_k34),
        "combo_k3k4_hidden_pre_collapse_mean": _mean_finite(combo_k34),
        "motion_safe_v2_pass_rate_best_ramp": max(
            [
                float(v.get("pass_rate"))
                for k, v in motion_summary.items()
                if str(k).startswith("bone_angvel") and v.get("pass_rate") is not None
            ]
            or [0.0]
        ),
        "mapping_r2_is_motion_success": False,
        "mapping_motion_safe_v2_pass_rate_best": max(
            [float(v.get("pass_rate")) for v in mapping_summary.values() if v.get("pass_rate") is not None] or [0.0]
        ),
        "caveat": "No bridge was trained; this does not prove final free-run success.",
    }

    json_path = out_dir / "bone_angvel_bridge_probe_summary.json"
    md_path = out_dir / "bone_angvel_bridge_probe_summary.md"
    _dump_json(json_path, payload)

    lines: List[str] = []
    lines.append("# Bone Angular Velocity Bridge Probe")
    lines.append("")
    lines.append("Read-only probe. No base-model training, checkpoint mutation, production gate change, commit, push, or stash.")
    lines.append("")
    lines.append("## 1. Input Tensor Contract")
    lines.append("")
    lines.append("| tensor | shape | dtype | device before forward |")
    lines.append("|---|---:|---|---|")
    for key, spec in payload["input_contract"]["base_model_forward_tensors"].items():
        lines.append(f"| {key} | `{spec['shape']}` | `{spec['dtype']}` | `{spec['device_before_forward']}` |")
    lines.append("")
    lines.append(
        f"- base normalized `bone_angvel_x` slice: `{_slice_bounds(state_angvel_sl)}`; "
        f"canonical 281-d action-handoff state contains bone_angvel: `False`."
    )
    lines.append("")
    lines.append("## 2. Per-Target K-Frame Ramp")
    for target, target_payload in payload["A_k_frame_ramp_causality"]["per_target"].items():
        lines.append("")
        lines.append(f"### {target}")
        if target_payload.get("skip_reason"):
            lines.append(f"- skipped: `{target_payload['skip_reason']}`")
            continue
        lines.append("| variant | hidden_pre cut/pre | hidden collapse to Walk | shared0 cut/pre | h_final cut/pre | out cut/pre | contacts_plan cut/pre |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in target_payload["variants"]:
            name = str(row["variant"])
            if name not in {"matched_base", "walk_continuous", "target_continuous", "state_all_to_walk", "x_state_angvel_to_walk"} and not name.startswith("bone_angvel"):
                continue
            lines.append(
                f"| {name} | {_fmt(row.get('hidden_pre_cut_over_pre4'))} | "
                f"{_fmt(row.get('hidden_pre_collapse_fraction_to_walk'))} | "
                f"{_fmt(row.get('shared_encoder_0_cut_over_pre4'))} | "
                f"{_fmt(row.get('h_final_cut_over_pre4'))} | "
                f"{_fmt(row.get('out_cut_over_pre4'))} | "
                f"{_fmt(row.get('contacts_plan_cut_over_pre4'))} |"
            )
    lines.append("")
    lines.append("## 3. Aggregate Collapse")
    lines.append("")
    lines.append("| variant | n | hidden_pre cut/pre mean | hidden collapse mean | shared0 cut/pre mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant, row in payload["A_k_frame_ramp_causality"]["aggregate"]["mean_by_variant"].items():
        lines.append(
            f"| {variant} | {row['n']} | {_fmt(row.get('hidden_pre_cut_over_pre4_mean'))} | "
            f"{_fmt(row.get('hidden_pre_collapse_fraction_to_walk_mean'))} | "
            f"{_fmt(row.get('shared_encoder_0_cut_over_pre4_mean'))} |"
        )
    lines.append("")
    lines.append("## 4. motion_safe_v2")
    lines.append("")
    lines.append("| variant | n | pass_rate | old_pop_safe | angvel rms | angvel p95 | foot mean | foot p95 | yaw dev deg/s | ego Δ | contact Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant, row in payload["B_motion_safe_v2"]["aggregate"]["summary_by_variant"].items():
        lines.append(
            f"| {variant} | {row['n']} | {_fmt(row.get('pass_rate'))} | {_fmt(row.get('old_pop_safe_rate'))} | "
            f"{_fmt(row.get('bone_angvel_delta_rms_rad_s_mean'))} | "
            f"{_fmt(row.get('bone_angvel_delta_p95_rad_s_mean'))} | "
            f"{_fmt(row.get('fk_foot_slip_mean_mps_mean'))} | "
            f"{_fmt(row.get('fk_foot_slip_p95_mps_mean'))} | "
            f"{_fmt(row.get('realized_yaw_rate_deviation_mean_deg_s_mean'))} | "
            f"{_fmt(row.get('ego_velocity_delta_l2_mean'))} | "
            f"{_fmt(row.get('contact_delta_l2_mean'))} |"
        )
    lines.append("")
    lines.append("> `old_pop_safe` is reported only as a comparator; it does not include bone angular velocity.")
    lines.append("")
    lines.append("## 5. Mapping Probe")
    lines.append("")
    lines.append("| group | n_train | n_test | input_dim | output_dim | ridge R2 | RMSE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for group, row in payload["C_mapping_probe"]["learnability"].get("groups", {}).items():
        best = row.get("best", {}) if isinstance(row, Mapping) else {}
        lines.append(
            f"| {group} | {row.get('n_train')} | {row.get('n_test')} | {row.get('input_dim')} | "
            f"{row.get('output_dim')} | {_fmt(best.get('r2'))} | {_fmt(best.get('rmse'))} |"
        )
    lines.append("")
    lines.append(f"- selected readout group: `{payload['C_mapping_probe']['learnability'].get('selected_group_for_readout')}`")
    lines.append("")
    lines.append("| mapping variant | n | pass_rate | old_pop_safe | hidden collapse | angvel rms | foot mean | yaw dev deg/s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for variant, row in payload["C_mapping_probe"]["readout_motion_safe_v2"]["summary_by_variant"].items():
        collapse = _mean_finite(
            [
                r.get("hidden_pre_collapse_fraction_to_walk")
                for r in payload["C_mapping_probe"]["readout_motion_safe_v2"]["rows"]
                if str(r.get("variant")) == variant
            ]
        )
        lines.append(
            f"| {variant} | {row['n']} | {_fmt(row.get('pass_rate'))} | {_fmt(row.get('old_pop_safe_rate'))} | "
            f"{_fmt(collapse)} | {_fmt(row.get('bone_angvel_delta_rms_rad_s_mean'))} | "
            f"{_fmt(row.get('fk_foot_slip_mean_mps_mean'))} | "
            f"{_fmt(row.get('realized_yaw_rate_deviation_mean_deg_s_mean'))} |"
        )
    lines.append("")
    lines.append("## 6. Conclusion")
    lines.append("")
    c = payload["conclusion"]
    lines.append(
        f"- supports `bone_angvel` transition as primary bridge variable: `{c['supports_bone_angvel_as_primary_bridge_variable']}` "
        f"(k=3/4 hidden collapse mean `{_fmt(c['bone_angvel_k3k4_hidden_pre_collapse_mean'])}`)."
    )
    lines.append(
        f"- combo rootvel/commanded-yaw observation k=3/4 hidden collapse mean: "
        f"`{_fmt(c['combo_k3k4_hidden_pre_collapse_mean'])}`."
    )
    lines.append(
        f"- best ramp motion_safe_v2 pass rate: `{_fmt(c['motion_safe_v2_pass_rate_best_ramp'])}`; "
        f"best mapping motion_safe_v2 pass rate: `{_fmt(c['mapping_motion_safe_v2_pass_rate_best'])}`."
    )
    lines.append("- caveat: no bridge was trained, and final free-run success is not proven.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `{json_path.resolve()}`")
    lines.append(f"- `{md_path.resolve()}`")
    _dump_md(md_path, lines)

    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {md_path}")
    print(
        "[A] bone k3/k4 collapse mean="
        f"{_fmt(payload['conclusion']['bone_angvel_k3k4_hidden_pre_collapse_mean'])} "
        "combo k3/k4="
        f"{_fmt(payload['conclusion']['combo_k3k4_hidden_pre_collapse_mean'])}"
    )
    print(
        "[B] best ramp motion_safe_v2 pass="
        f"{_fmt(payload['conclusion']['motion_safe_v2_pass_rate_best_ramp'])}"
    )
    print(
        "[C] mapping selected="
        f"{payload['C_mapping_probe']['learnability'].get('selected_group_for_readout')} "
        "best motion_safe_v2 pass="
        f"{_fmt(payload['conclusion']['mapping_motion_safe_v2_pass_rate_best'])}"
    )


if __name__ == "__main__":
    main()
