#!/usr/bin/env python3
"""Read-only MM-like head audit for action-handoff bridge diagnostics.

This probe does not train, mutate checkpoints, or run the in-betweening MLP.  It loads the
base/posttrain EventMotionModel and teacher-forces short sequences through the model to inspect
how the contact-plan GRU state, hidden features, motion head, direct head, and fusion heads react
around a raw motion-matching style cut.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train.validate.run_freerun_cycles as freerun  # noqa: E402
from train.data.action_handoff_inbetween import TURN_CLIPS, WALK_F  # noqa: E402
from tools.run_action_handoff_inbetween_b1_cond_baseline_probe import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CKPT,
    DEFAULT_ENCODER_BUNDLE,
    DEFAULT_NPZ_ROOT,
    DEFAULT_PRETRAIN_TEMPLATE,
    _make_runner_args,
)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _finite_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fmt(v: Any, digits: int = 4) -> str:
    x = _finite_float(v)
    if x is None:
        return "null"
    return f"{x:.{digits}f}"


def _parse_clips(raw: str) -> List[str]:
    clips = [tok.strip() for tok in str(raw or "").replace(";", ",").split(",") if tok.strip()]
    if not clips:
        raise ValueError("--target-clips must contain at least one clip")
    valid = set(TURN_CLIPS)
    bad = [c for c in clips if c not in valid]
    if bad:
        raise ValueError(f"unsupported target clip(s): {bad}; expected subset of {sorted(valid)}")
    return clips


def _rolled_indices(start: int, n: int, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("clip length must be positive")
    start_i = int(start) % int(length)
    return (np.arange(int(n), dtype=np.int64) + start_i) % int(length)


def _gather(arr: Optional[np.ndarray], idx: np.ndarray, width: int) -> torch.Tensor:
    if arr is None:
        return torch.zeros((int(idx.shape[0]), int(width)), dtype=torch.float32)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    if int(a.shape[1]) != int(width):
        if int(a.shape[1]) > int(width):
            a = a[:, : int(width)]
        else:
            pad = np.zeros((a.shape[0], int(width) - int(a.shape[1])), dtype=np.float32)
            a = np.concatenate([a, pad], axis=1)
    return torch.from_numpy(np.ascontiguousarray(a[idx])).float()


def _clip_len(clip: Any) -> int:
    return int(np.asarray(clip.X).shape[0])


def _load_clip(runner: Any, npz_root: Path, clip_name: str, seq_len: int) -> Any:
    ds = runner._build_dataset(npz_root / f"{clip_name}.npz", seq_len=int(seq_len))
    runner._ensure_model_ready(ds)
    return ds.clips[0]


def _build_case_sequence(
    *,
    case: str,
    walk_clip: Any,
    turn_clip: Any,
    pre_frames: int,
    post_frames: int,
    walk_start: int,
    target_start: int,
    dims: Mapping[str, int],
) -> Tuple[Dict[str, torch.Tensor], List[str], Dict[str, Any]]:
    """Build one teacher-forced sequence [1,T,*] for an MM-like discontinuity audit."""
    pre = int(pre_frames)
    post = int(post_frames)
    total = pre + post
    walk_pre_idx = _rolled_indices(int(walk_start), pre, _clip_len(walk_clip))
    walk_post_idx = _rolled_indices(int(walk_start) + pre, post, _clip_len(walk_clip))
    walk_all_idx = np.concatenate([walk_pre_idx, walk_post_idx], axis=0)

    turn_pre_start = int(target_start) - pre
    turn_pre_idx = _rolled_indices(turn_pre_start, pre, _clip_len(turn_clip))
    turn_post_idx = _rolled_indices(int(target_start), post, _clip_len(turn_clip))
    turn_all_idx = np.concatenate([turn_pre_idx, turn_post_idx], axis=0)

    if case == "walk_control":
        x_idx = walk_all_idx
        c_idx = walk_all_idx
        x_clip = c_clip = walk_clip
        src = ["walk"] * total
        description = "continuous Walk_F teacher-forced control"
    elif case == "turn_ref":
        x_idx = turn_all_idx
        c_idx = turn_all_idx
        x_clip = c_clip = turn_clip
        src = ["turn"] * total
        description = "continuous target-turn teacher-forced reference"
    elif case == "mm_cut":
        x_idx = np.concatenate([walk_pre_idx, turn_post_idx], axis=0)
        c_idx = x_idx
        x_pre_clip, x_post_clip = walk_clip, turn_clip
        c_pre_clip, c_post_clip = walk_clip, turn_clip
        src = ["walk"] * pre + ["turn"] * post
        description = "raw MM-style cut: state/contact/cond all jump from Walk_F to target turn"
        sample = _concat_two_source_sample(
            x_pre_clip=x_pre_clip,
            x_post_clip=x_post_clip,
            c_pre_clip=c_pre_clip,
            c_post_clip=c_post_clip,
            pre_idx=walk_pre_idx,
            post_idx=turn_post_idx,
            dims=dims,
        )
        return sample, src, {"description": description, "x_indices": x_idx.tolist(), "cond_indices": c_idx.tolist()}
    elif case == "walk_state_turn_cond":
        x_pre_clip = x_post_clip = walk_clip
        c_pre_clip, c_post_clip = walk_clip, turn_clip
        src = ["walk"] * pre + ["walk_state+turn_cond"] * post
        description = "Walk_F state/contact stream, but cond jumps to target turn after cut"
        sample = _concat_two_source_sample(
            x_pre_clip=x_pre_clip,
            x_post_clip=x_post_clip,
            c_pre_clip=c_pre_clip,
            c_post_clip=c_post_clip,
            pre_idx=walk_pre_idx,
            post_idx=turn_post_idx,
            dims=dims,
        )
        return sample, src, {
            "description": description,
            "x_indices": np.concatenate([walk_pre_idx, walk_post_idx], axis=0).tolist(),
            "cond_indices": np.concatenate([walk_pre_idx, turn_post_idx], axis=0).tolist(),
        }
    else:
        raise ValueError(f"unsupported case={case!r}")

    sample = {
        "state": _gather(x_clip.X, x_idx, dims["state"]).unsqueeze(0),
        "cond": _gather(c_clip.C, c_idx, dims["cond"]).unsqueeze(0),
        "contacts": _gather(getattr(x_clip, "contacts", None), x_idx, dims["contact"]).unsqueeze(0),
        "angvel": _gather(getattr(x_clip, "angvel_norm", None), x_idx, dims["angvel"]).unsqueeze(0),
        "pose_history": _gather(getattr(x_clip, "pose_hist_norm", None), x_idx, dims["pose_hist"]).unsqueeze(0),
    }
    return sample, src, {"description": description, "x_indices": x_idx.tolist(), "cond_indices": c_idx.tolist()}


def _concat_two_source_sample(
    *,
    x_pre_clip: Any,
    x_post_clip: Any,
    c_pre_clip: Any,
    c_post_clip: Any,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    dims: Mapping[str, int],
) -> Dict[str, torch.Tensor]:
    def cat(field: str, pre_clip: Any, post_clip: Any, width: int) -> torch.Tensor:
        pre = _gather(getattr(pre_clip, field, None), pre_idx, width)
        post = _gather(getattr(post_clip, field, None), post_idx, width)
        return torch.cat([pre, post], dim=0).unsqueeze(0)

    state = torch.cat(
        [
            _gather(x_pre_clip.X, pre_idx, dims["state"]),
            _gather(x_post_clip.X, post_idx, dims["state"]),
        ],
        dim=0,
    ).unsqueeze(0)
    cond = torch.cat(
        [
            _gather(c_pre_clip.C, pre_idx, dims["cond"]),
            _gather(c_post_clip.C, post_idx, dims["cond"]),
        ],
        dim=0,
    ).unsqueeze(0)
    return {
        "state": state,
        "cond": cond,
        "contacts": cat("contacts", x_pre_clip, x_post_clip, dims["contact"]),
        "angvel": cat("angvel_norm", x_pre_clip, x_post_clip, dims["angvel"]),
        "pose_history": cat("pose_hist_norm", x_pre_clip, x_post_clip, dims["pose_hist"]),
    }


def _mean_step_array(t: torch.Tensor, *, query_steps: int) -> Optional[np.ndarray]:
    if not torch.is_tensor(t):
        return None
    v = t.detach().float().cpu()
    if v.ndim == 0:
        return np.full((int(query_steps), 1), float(v.item()), dtype=np.float32)
    if v.ndim == 1:
        if int(v.shape[0]) == int(query_steps):
            return v.reshape(int(query_steps), 1).numpy()
        return v.reshape(1, -1).expand(int(query_steps), -1).numpy()
    if v.ndim >= 2:
        if int(v.shape[0]) == 1:
            v = v[0]
        if v.ndim >= 2 and int(v.shape[0]) == int(query_steps):
            return v.reshape(int(query_steps), -1).numpy()
        if v.ndim >= 3 and int(v.shape[1]) == int(query_steps):
            return v.mean(dim=0).reshape(int(query_steps), -1).numpy()
    return None


def _tensor_key_series(result: Mapping[str, Any], key: str, *, query_steps: int) -> Optional[np.ndarray]:
    value = result.get(key)
    if not torch.is_tensor(value):
        return None
    return _mean_step_array(value, query_steps=query_steps)


def _norm_or_none(row: Optional[np.ndarray], t: int) -> Optional[float]:
    if row is None or int(t) >= int(row.shape[0]) or int(row.shape[1]) == 0:
        return None
    return float(np.linalg.norm(row[int(t)].astype(np.float64)))


def _mean_or_none(row: Optional[np.ndarray], t: int) -> Optional[float]:
    if row is None or int(t) >= int(row.shape[0]) or int(row.shape[1]) == 0:
        return None
    return float(np.mean(row[int(t)].astype(np.float64)))


def _std_or_none(row: Optional[np.ndarray], t: int) -> Optional[float]:
    if row is None or int(t) >= int(row.shape[0]) or int(row.shape[1]) == 0:
        return None
    return float(np.std(row[int(t)].astype(np.float64)))


def _vec_or_none(row: Optional[np.ndarray], t: int, limit: Optional[int] = None) -> Optional[List[float]]:
    if row is None or int(t) >= int(row.shape[0]) or int(row.shape[1]) == 0:
        return None
    v = row[int(t)]
    if limit is not None:
        v = v[: int(limit)]
    return [float(x) for x in v.astype(np.float64).tolist()]


def _cos_prev(row: Optional[np.ndarray], t: int) -> Optional[float]:
    if row is None or int(t) <= 0 or int(t) >= int(row.shape[0]) or int(row.shape[1]) == 0:
        return None
    a = row[int(t) - 1].astype(np.float64)
    b = row[int(t)].astype(np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return None
    return float(np.dot(a, b) / den)


def _delta_norm(row: Optional[np.ndarray], t: int) -> Optional[float]:
    if row is None or int(t) <= 0 or int(t) >= int(row.shape[0]) or int(row.shape[1]) == 0:
        return None
    d = row[int(t)].astype(np.float64) - row[int(t) - 1].astype(np.float64)
    return float(np.linalg.norm(d))


def _rms_slice(row: Optional[np.ndarray], t: int, sl: Optional[slice]) -> Optional[float]:
    if row is None or sl is None or int(t) >= int(row.shape[0]) or int(row.shape[1]) == 0:
        return None
    start = int(sl.start or 0)
    stop = int(sl.stop or row.shape[1])
    stop = min(stop, int(row.shape[1]))
    if stop <= start:
        return None
    v = row[int(t), start:stop].astype(np.float64)
    return float(np.sqrt(np.mean(v * v)))


def _omega_mean_deg(row: Optional[np.ndarray], t: int) -> Optional[float]:
    if row is None or int(t) >= int(row.shape[0]) or int(row.shape[1]) == 0:
        return None
    flat = row[int(t)].astype(np.float64)
    if flat.size % 3 != 0:
        return float(np.linalg.norm(flat) * 180.0 / math.pi)
    omega = flat.reshape(-1, 3)
    return float(np.mean(np.linalg.norm(omega, axis=-1)) * 180.0 / math.pi)


def _generic_head_per_step(
    row: Optional[np.ndarray], *, cut_step: int, source_labels: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """Per-frame summary for an arbitrary output head reshaped to [T, D]."""
    if row is None:
        return None
    T = int(row.shape[0])
    per_step: List[Dict[str, Any]] = []
    for t in range(T):
        per_step.append(
            {
                "step": int(t),
                "rel_to_cut": int(t - int(cut_step)),
                "source": str(source_labels[t]) if t < len(source_labels) else "unknown",
                "norm": _norm_or_none(row, t),
                "delta_norm": _delta_norm(row, t),
                "cos_prev": _cos_prev(row, t),
                "mean": _mean_or_none(row, t),
                "std": _std_or_none(row, t),
            }
        )
    return {
        "width": int(row.shape[1]),
        "cut_delta_norm": _delta_norm(row, int(cut_step)),
        "cut_cos_prev": _cos_prev(row, int(cut_step)),
        "per_step": per_step,
    }


def _case_forward_audit(
    model: Any,
    sample: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    cut_step: int,
    source_labels: Sequence[str],
    y_rot_slice: Optional[slice],
    y_root_slice: Optional[slice],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    state = sample["state"].to(device=device)
    cond = sample["cond"].to(device=device)
    contacts = sample["contacts"].to(device=device)
    angvel = sample["angvel"].to(device=device)
    pose_history = sample["pose_history"].to(device=device)
    _, T, _ = state.shape

    plan_gru_raw: List[torch.Tensor] = []
    plan_head_inputs: List[torch.Tensor] = []
    hidden_pre_inputs: List[torch.Tensor] = []

    handles = []
    cell = getattr(model, "contact_plan_cell", None)
    head = getattr(model, "contact_plan_head", None)
    pasa_lnq = getattr(model, "_pasa_lnq", None)
    if cell is not None:
        handles.append(cell.register_forward_hook(lambda _m, _inp, out: plan_gru_raw.append(out.detach().float().cpu())))
    if head is not None:
        def _head_pre(_m: Any, inputs: Tuple[Any, ...]) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                plan_head_inputs.append(inputs[0].detach().float().cpu())
        handles.append(head.register_forward_pre_hook(_head_pre))
    if pasa_lnq is not None:
        def _hidden_pre(_m: Any, inputs: Tuple[Any, ...]) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                hidden_pre_inputs.append(inputs[0].detach().float().cpu())
        handles.append(pasa_lnq.register_forward_pre_hook(_hidden_pre))

    try:
        if hasattr(model, "set_eval_runtime_controls"):
            model.set_eval_runtime_controls(debug_contact_plan_logits_decomp=True)
        with torch.no_grad():
            result = model(
                state,
                cond,
                contacts=contacts,
                angvel=angvel,
                pose_history=pose_history,
                time_index=torch.arange(T, device=device, dtype=state.dtype).view(1, T),
                rollout_step=torch.arange(T, device=device, dtype=state.dtype).view(1, T),
            )
    finally:
        for h in handles:
            h.remove()
        if hasattr(model, "_reset_eval_runtime_controls"):
            model._reset_eval_runtime_controls()

    if not isinstance(result, dict):
        raise RuntimeError("EventMotionModel.forward did not return a dict")

    output_contract: Dict[str, Any] = {}
    all_series: Dict[str, np.ndarray] = {}
    for k, v in sorted(result.items()):
        if torch.is_tensor(v):
            output_contract[k] = {
                "shape": [int(x) for x in v.shape],
                "dtype": str(v.dtype).replace("torch.", ""),
                "device": str(v.device),
            }
            row = _mean_step_array(v, query_steps=int(T))
            if row is not None:
                all_series[k] = row.astype(np.float32)

    plan_raw = np.stack([x.reshape(-1, x.shape[-1]).mean(dim=0).numpy() for x in plan_gru_raw], axis=0) if plan_gru_raw else None
    plan_head = np.stack([x.reshape(-1, x.shape[-1]).mean(dim=0).numpy() for x in plan_head_inputs], axis=0) if plan_head_inputs else None
    plan_corr = None
    plan_head_call_pattern = "none"
    if plan_head is not None:
        if int(plan_head.shape[0]) == int(T) * 2:
            plan_corr = plan_head[1::2]
            plan_head_call_pattern = "raw_then_corrected_per_step"
        elif int(plan_head.shape[0]) == int(T):
            plan_corr = plan_head
            plan_head_call_pattern = "single_plan_head_call_per_step"
        else:
            plan_head_call_pattern = f"unexpected_{int(plan_head.shape[0])}_calls_for_T_{int(T)}"
    hidden_pre = None
    if hidden_pre_inputs:
        # _pasa_lnq is called once with [B,T,H] in this teacher-forced probe.
        hidden_pre = _mean_step_array(hidden_pre_inputs[-1], query_steps=T)

    series = {
        "h_final": _tensor_key_series(result, "h_final", query_steps=T),
        "out": _tensor_key_series(result, "out", query_steps=T),
        "out_direct": _tensor_key_series(result, "out_direct", query_steps=T),
        "contacts_plan": _tensor_key_series(result, "contacts_plan", query_steps=T),
        "contacts_plan_logits": _tensor_key_series(result, "contacts_plan_logits", query_steps=T),
        "contacts_plan_logits_base": _tensor_key_series(result, "contacts_plan_logits_base", query_steps=T),
        "contacts_plan_logits_raw": _tensor_key_series(result, "contacts_plan_logits_raw", query_steps=T),
        "contacts_plan_logits_time": _tensor_key_series(result, "contacts_plan_logits_time", query_steps=T),
        "contacts_meas": _tensor_key_series(result, "contacts_meas", query_steps=T),
        "contacts_err": _tensor_key_series(result, "contacts_err", query_steps=T),
        "event_clock_lambda_corr": _tensor_key_series(result, "event_clock_lambda_corr", query_steps=T),
        "event_clock_lambda_logit": _tensor_key_series(result, "event_clock_lambda_logit", query_steps=T),
        "event_clock_dynamic_prior": _tensor_key_series(result, "event_clock_dynamic_prior", query_steps=T),
        "event_clock_delta_z": _tensor_key_series(result, "event_clock_delta_z", query_steps=T),
        "lambda_fusion": _tensor_key_series(result, "lambda_fusion", query_steps=T),
        "lambda_fusion_logits": _tensor_key_series(result, "lambda_fusion_logits", query_steps=T),
        "omega_hat": _tensor_key_series(result, "omega_hat", query_steps=T),
        "direct_leg_omega": _tensor_key_series(result, "direct_leg_omega", query_steps=T),
        "direct_leg_gate": _tensor_key_series(result, "direct_leg_gate", query_steps=T),
        "direct_leg_scale": _tensor_key_series(result, "direct_leg_scale", query_steps=T),
        "period_pred": _tensor_key_series(result, "period_pred", query_steps=T),
        "hidden_pre": hidden_pre,
        "plan_z_raw_gru": plan_raw,
        "plan_z_corr_head_input": plan_corr,
    }

    per_step: List[Dict[str, Any]] = []
    for t in range(int(T)):
        per_step.append(
            {
                "step": int(t),
                "rel_to_cut": int(t - int(cut_step)),
                "source": str(source_labels[t]) if t < len(source_labels) else "unknown",
                "plan_z_raw_norm": _norm_or_none(series["plan_z_raw_gru"], t),
                "plan_z_corr_norm": _norm_or_none(series["plan_z_corr_head_input"], t),
                "plan_z_corr_delta_norm": _delta_norm(series["plan_z_corr_head_input"], t),
                "plan_z_corr_cos_prev": _cos_prev(series["plan_z_corr_head_input"], t),
                "contacts_plan": _vec_or_none(series["contacts_plan"], t),
                "contacts_plan_logits": _vec_or_none(series["contacts_plan_logits"], t),
                "contacts_plan_logits_base": _vec_or_none(series["contacts_plan_logits_base"], t),
                "contacts_plan_logits_raw": _vec_or_none(series["contacts_plan_logits_raw"], t),
                "contacts_plan_logits_time": _vec_or_none(series["contacts_plan_logits_time"], t),
                "contacts_meas": _vec_or_none(series["contacts_meas"], t),
                "contacts_err": _vec_or_none(series["contacts_err"], t),
                "hidden_pre_norm": _norm_or_none(series["hidden_pre"], t),
                "hidden_pre_delta_norm": _delta_norm(series["hidden_pre"], t),
                "h_final_norm": _norm_or_none(series["h_final"], t),
                "h_final_delta_norm": _delta_norm(series["h_final"], t),
                "h_final_cos_prev": _cos_prev(series["h_final"], t),
                "motion_out_norm": _norm_or_none(series["out"], t),
                "motion_out_pose_rms": _rms_slice(series["out"], t, y_rot_slice),
                "motion_out_root_rms": _rms_slice(series["out"], t, y_root_slice),
                "direct_out_norm": _norm_or_none(series["out_direct"], t),
                "direct_out_pose_rms": _rms_slice(series["out_direct"], t, y_rot_slice),
                "direct_out_root_rms": _rms_slice(series["out_direct"], t, y_root_slice),
                "lambda_mean": _mean_or_none(series["lambda_fusion"], t),
                "lambda_std": _std_or_none(series["lambda_fusion"], t),
                "event_clock_lambda_corr": _mean_or_none(series["event_clock_lambda_corr"], t),
                "event_clock_dynamic_prior": _mean_or_none(series["event_clock_dynamic_prior"], t),
                "event_clock_delta_z_norm": _norm_or_none(series["event_clock_delta_z"], t),
                "omega_hat_mean_deg": _omega_mean_deg(series["omega_hat"], t),
                "direct_leg_omega_mean_deg": _omega_mean_deg(series["direct_leg_omega"], t),
                "period_pred_norm": _norm_or_none(series["period_pred"], t),
            }
        )

    jump_keys = [
        "plan_z_corr_delta_norm",
        "h_final_delta_norm",
        "hidden_pre_delta_norm",
        "motion_out_pose_rms",
        "motion_out_root_rms",
        "direct_out_pose_rms",
        "direct_out_root_rms",
        "event_clock_delta_z_norm",
        "omega_hat_mean_deg",
        "direct_leg_omega_mean_deg",
    ]
    cut_row = per_step[int(cut_step)] if 0 <= int(cut_step) < len(per_step) else {}
    pre_rows = per_step[max(0, int(cut_step) - 4): int(cut_step)]
    post_rows = per_step[int(cut_step): min(len(per_step), int(cut_step) + 4)]

    def _mean_key(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        vals = [_finite_float(r.get(key)) for r in rows]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "T": int(T),
        "cut_step": int(cut_step),
        "available_keys": sorted([k for k, v in result.items() if torch.is_tensor(v)]),
        "plan_head_call_pattern": plan_head_call_pattern,
        "plan_gru_steps_captured": int(0 if plan_raw is None else plan_raw.shape[0]),
        "cut_row": {k: cut_row.get(k) for k in ["step", "source", *jump_keys]},
        "pre4_mean": {k: _mean_key(pre_rows, k) for k in jump_keys},
        "post4_mean": {k: _mean_key(post_rows, k) for k in jump_keys},
        "contacts_plan_cut": cut_row.get("contacts_plan"),
        "contacts_meas_cut": cut_row.get("contacts_meas"),
        "lambda_cut_mean": cut_row.get("lambda_mean"),
        "omega_cut_mean_deg": cut_row.get("omega_hat_mean_deg"),
    }
    all_heads = {
        k: _generic_head_per_step(row, cut_step=int(cut_step), source_labels=source_labels)
        for k, row in sorted(all_series.items())
    }

    audit = {
        "input_contract": {
            "state_shape": [int(x) for x in state.shape],
            "state_dtype": str(state.dtype).replace("torch.", ""),
            "state_device": str(state.device),
            "cond_shape": [int(x) for x in cond.shape],
            "contacts_shape": [int(x) for x in contacts.shape],
            "angvel_shape": [int(x) for x in angvel.shape],
            "pose_history_shape": [int(x) for x in pose_history.shape],
        },
        "output_contract": output_contract,
        "summary": summary,
        "per_step": per_step,
        "all_heads": all_heads,
        "all_heads_keys": sorted(all_series.keys()),
    }
    return audit, all_series


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only MM-like EventMotionModel head audit.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=str, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--pre-frames", type=int, default=16)
    p.add_argument("--post-frames", type=int, default=24)
    p.add_argument("--walk-start", type=int, default=21)
    p.add_argument("--target-start", type=int, default=0)
    p.add_argument("--target-clips", type=str, default="Walk_L_To_R")
    p.add_argument(
        "--cases",
        type=str,
        default="walk_control,turn_ref,mm_cut,walk_state_turn_cond",
        help="Comma-separated subset of walk_control,turn_ref,mm_cut,walk_state_turn_cond.",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    npz_root = Path(args.npz_root)
    ckpt = Path(args.checkpoint).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not npz_root.exists():
        raise FileNotFoundError(f"npz root not found: {npz_root}")
    target_clips = _parse_clips(args.target_clips)
    valid_cases = {"walk_control", "turn_ref", "mm_cut", "walk_state_turn_cond"}
    cases = [c.strip() for c in str(args.cases).replace(";", ",").split(",") if c.strip()]
    bad_cases = [c for c in cases if c not in valid_cases]
    if bad_cases:
        raise ValueError(f"unsupported case(s): {bad_cases}; expected subset of {sorted(valid_cases)}")
    if not cases:
        raise ValueError("--cases must contain at least one case")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"debug_output/_tmp_action_handoff_mm_like_head_audit_{date_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = freerun.FreeRunCycleRunner(_make_runner_args(args))
    seq_len = max(64, int(args.pre_frames) + int(args.post_frames))
    walk_clip = _load_clip(runner, npz_root, WALK_F, seq_len=seq_len)
    runner.model.eval()
    model = runner.model

    dims = {
        "state": int(getattr(model, "in_state_dim", walk_clip.X.shape[1])),
        "cond": int(getattr(model, "cond_dim", walk_clip.C.shape[1])),
        "contact": int(getattr(model, "contact_dim", 0) or 0),
        "angvel": int(getattr(model, "angvel_dim", 0) or 0),
        "pose_hist": int(getattr(model, "pose_hist_dim", 0) or 0),
    }

    payload: Dict[str, Any] = {
        "task": "action_handoff_mm_like_head_audit",
        "scope": "read-only EventMotionModel forward probe; no training, no checkpoint mutation",
        "checkpoint": str(ckpt.resolve()),
        "npz_root": str(npz_root.resolve()),
        "config": {
            "pre_frames": int(args.pre_frames),
            "post_frames": int(args.post_frames),
            "cut_step": int(args.pre_frames),
            "walk_start": int(args.walk_start),
            "target_start": int(args.target_start),
            "cases": cases,
            "target_clips": target_clips,
            "device": str(args.device),
        },
        "model_flags": {
            "contact_plan_enable": bool(getattr(model, "contact_plan_enable", False)),
            "contact_plan_hidden": int(getattr(model, "contact_plan_hidden", 0) or 0),
            "contact_plan_inject": str(getattr(model, "contact_plan_inject", "")),
            "contact_plan_init_mode": str(getattr(model, "contact_plan_init_mode", "")),
            "use_event_clock": bool(getattr(model, "use_event_clock", False)),
            "direct_pose_enable": bool(getattr(model, "direct_pose_enable", False)),
            "direct_pose_feat_source": str(getattr(model, "direct_pose_feat_source", "")),
            "direct_pose_use_phase_z": bool(getattr(model, "direct_pose_use_phase_z", False)),
            "direct_pose_meas_mode": str(getattr(model, "direct_pose_meas_mode", "")),
            "lambda_fusion_enable": bool(getattr(model, "lambda_fusion_enable", False)),
            "lambda_fusion_mode": str(getattr(model, "lambda_fusion_mode", "")),
            "so3_delta_corrector": bool(getattr(model, "so3_delta_corrector", None) is not None),
        },
        "dims": dims,
        "targets": {},
        "interpretation": {
            "mm_cut": "raw motion-matching style discontinuity: Walk_F pre-cut then target-turn post-cut inputs.",
            "walk_state_turn_cond": (
                "same Walk_F state/contact stream, but cond switches to target turn after the cut; "
                "isolates contact-plan GRU response to cond from state/contact discontinuity."
            ),
            "head_metric_units": {
                "plan_z_*": "L2 norms / cosine in contact-plan GRU hidden space",
                "contacts_*": "prob/logit per contact channel",
                "motion/direct_*_rms": "normalized Y-space RMS over output slices",
                "omega_*_deg": "axis-angle norm converted to degrees",
            },
        },
    }

    y_rot_slice = getattr(runner.trainer, "rot6d_y_slice", None)
    y_root_slice = getattr(runner.trainer, "rootvel_slice", None)

    npz_arrays: Dict[str, np.ndarray] = {}
    for target in target_clips:
        turn_clip = _load_clip(runner, npz_root, target, seq_len=seq_len)
        target_payload: Dict[str, Any] = {
            "clip_lengths": {WALK_F: _clip_len(walk_clip), target: _clip_len(turn_clip)},
            "cases": {},
        }
        for case in cases:
            sample, labels, meta = _build_case_sequence(
                case=case,
                walk_clip=walk_clip,
                turn_clip=turn_clip,
                pre_frames=int(args.pre_frames),
                post_frames=int(args.post_frames),
                walk_start=int(args.walk_start),
                target_start=int(args.target_start),
                dims=dims,
            )
            audit, all_series = _case_forward_audit(
                model,
                sample,
                device=runner.device,
                cut_step=int(args.pre_frames),
                source_labels=labels,
                y_rot_slice=y_rot_slice,
                y_root_slice=y_root_slice,
            )
            target_payload["cases"][case] = {
                "sequence_meta": meta,
                **audit,
            }
            for head_key, row in all_series.items():
                npz_arrays[f"{target}__{case}__{head_key}"] = row
        payload["targets"][target] = target_payload

    json_path = out_dir / "mm_like_head_audit_summary.json"
    _dump_json(json_path, payload)

    npz_path = out_dir / "mm_like_head_audit_arrays.npz"
    np.savez_compressed(npz_path, **npz_arrays)
    payload["all_heads_arrays_npz"] = str(npz_path.resolve())
    payload["all_heads_arrays_layout"] = "key='{target}__{case}__{head}', value=float32 [T, D] per-frame head trace"
    _dump_json(json_path, payload)

    lines: List[str] = []
    lines.append("# MM-like Head Audit")
    lines.append("")
    lines.append("Read-only forward probe over base `EventMotionModel`; no training and no checkpoint mutation.")
    lines.append("")
    lines.append(f"- checkpoint: `{ckpt.name}`")
    lines.append(f"- cut: Walk_F start `{int(args.walk_start)}`, target start `{int(args.target_start)}`, pre/post `{int(args.pre_frames)}/{int(args.post_frames)}`")
    lines.append(f"- cases: `{', '.join(cases)}`")
    lines.append("")
    lines.append("## Model Flags")
    for k, v in payload["model_flags"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    for target in target_clips:
        lines.append(f"## Target `{target}`")
        lines.append("| case | plan jump@cut | hidden jump@cut | motion pose RMS@cut | direct pose RMS@cut | contact plan@cut | lambda@cut | omega_hat deg@cut |")
        lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
        for case in cases:
            s = payload["targets"][target]["cases"][case]["summary"]
            cut = s["cut_row"]
            lines.append(
                f"| {case} | "
                f"{_fmt(cut.get('plan_z_corr_delta_norm'))} | "
                f"{_fmt(cut.get('h_final_delta_norm'))} | "
                f"{_fmt(cut.get('motion_out_pose_rms'))} | "
                f"{_fmt(cut.get('direct_out_pose_rms'))} | "
                f"`{s.get('contacts_plan_cut')}` | "
                f"{_fmt(s.get('lambda_cut_mean'))} | "
                f"{_fmt(s.get('omega_cut_mean_deg'))} |"
            )
        lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `{json_path.resolve()}`")
    lines.append(f"- `{npz_path.resolve()}` (per-frame [T,D] trace for every output head)")
    md_path = out_dir / "mm_like_head_audit_summary.md"
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {npz_path}")
    print(f"[ok] wrote: {md_path}")
    for target in target_clips:
        row_bits = []
        for case in cases:
            cut = payload["targets"][target]["cases"][case]["summary"]["cut_row"]
            row_bits.append(
                f"{case}: plan_jump={_fmt(cut.get('plan_z_corr_delta_norm'))}, "
                f"h_jump={_fmt(cut.get('h_final_delta_norm'))}"
            )
        print(f"[target {target}] " + " | ".join(row_bits))


if __name__ == "__main__":
    main()
