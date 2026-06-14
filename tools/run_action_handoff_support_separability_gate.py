#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import rollout_kernel as _rollout_kernel  # noqa: E402
from train.data.action_handoff_inbetween import CONTACT_SLICE, WALK_F, load_clip_states  # noqa: E402
from train.history import advance_pose_hist_state_with_tail  # noqa: E402
from train.validate.contact_meas_whitebox import compute_contact_meas_whitebox  # noqa: E402
from train.validate.run_freerun_cycles import (  # noqa: E402
    FreeRunCycleRunner,
    _init_eval_pose_hist_state,
    _resolve_eval_pose_hist_input,
)


DATE_TAG = "20260607"
DEFAULT_STEM = "support_separability_gate"
CALIBRATION_PATH = ROOT / "debug_output" / "20260606_gap_selection_goal_contract_calibration.py"
HELPER_PATH = ROOT / "tools" / "run_action_handoff_freerun_contact_stability_probe.py"
CLASSES = ("right", "left", "dual", "flight")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class GateCase:
    case_id: str
    clip_name: str
    start_frame: int
    group_key: str
    group_index: int
    phase_bin8: int
    phase_label: str
    start_label: str
    region_key: str
    cycle_phase: float
    cycle_phase_sin: float
    cycle_phase_cos: float


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            sk = str(key)
            if sk not in seen:
                seen.add(sk)
                keys.append(sk)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _slice_width(value: Any, *, name: str) -> int:
    if not isinstance(value, slice) or value.start is None or value.stop is None:
        raise RuntimeError(f"{name} must be a concrete slice")
    width = int(value.stop) - int(value.start)
    if width <= 0:
        raise RuntimeError(f"{name} width must be positive, got {value}")
    return width


def _as_batch(sample: Mapping[str, torch.Tensor], key: str, device: torch.device) -> Optional[torch.Tensor]:
    value = sample.get(key)
    if not torch.is_tensor(value):
        return None
    return value.unsqueeze(0).to(device=device, dtype=torch.float32)


def _squeeze_step_tensor(value: Any) -> Optional[torch.Tensor]:
    if not torch.is_tensor(value):
        return None
    out = value.detach()
    if out.dim() == 3 and out.shape[1] == 1:
        out = out[:, 0]
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return out


def _time_inputs(motion: torch.Tensor, step: int, total_steps: int) -> tuple[int, torch.Tensor]:
    denom = max(1, int(total_steps) - 1)
    rollout_step = torch.full((motion.shape[0], 1, 1), float(step) / float(denom), device=motion.device, dtype=motion.dtype)
    return int(step), rollout_step


def _state_angvel(motion: torch.Tensor, cfg: Any) -> Optional[torch.Tensor]:
    sl = getattr(cfg, "angvel_x_slice", None)
    if isinstance(sl, slice):
        return motion[..., sl].detach()
    return None


def _compose_raw(
    trainer: Any,
    ret: Mapping[str, Any],
    y_prev_raw: torch.Tensor,
    step_idx: int,
    total_steps: int,
    *,
    apply_lambda: bool,
) -> torch.Tensor:
    y_inc_raw = trainer._compose_delta_to_raw(
        y_prev_raw,
        ret["out"],
        omega_hat=ret.get("omega_hat", None) if bool(getattr(trainer, "so3_corr_apply", False)) else None,
        so3_gate=getattr(trainer, "so3_corr_gate_force", None),
        so3_max_deg=getattr(trainer, "so3_corr_max_deg", None),
    )
    if bool(apply_lambda) and torch.is_tensor(ret.get("out_direct", None)) and torch.is_tensor(ret.get("lambda_fusion", None)):
        lam_eff, _ = trainer._lambda_fusion_apply_reliability(
            ret["lambda_fusion"],
            step_idx=int(step_idx),
            total_steps=int(total_steps),
            ret=dict(ret),
        )
        return trainer._apply_lambda_fusion_to_raw(y_inc_raw, direct_norm=ret["out_direct"], lambda_fusion=lam_eff)
    return y_inc_raw


def _label_from_scores_rl(scores: Sequence[float], *, thr: float) -> str:
    vals = list(scores)
    right_on = bool(len(vals) > 0 and _finite_float(vals[0], 0.0) > float(thr))
    left_on = bool(len(vals) > 1 and _finite_float(vals[1], 0.0) > float(thr))
    if right_on and left_on:
        return "dual"
    if right_on:
        return "right"
    if left_on:
        return "left"
    return "flight"


def _short_run_normalize(labels: Sequence[str], min_len: int = 2) -> tuple[list[str], int]:
    vals = [str(x) for x in labels]
    if not vals:
        return [], 0
    out = list(vals)
    changed = 0
    i = 0
    n = len(out)
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if j - i < int(min_len):
            repl = None
            if i > 0 and j < n and out[i - 1] == out[j]:
                repl = out[i - 1]
            elif i > 0:
                repl = out[i - 1]
            elif j < n:
                repl = out[j]
            if repl is not None and repl != out[i]:
                for k in range(i, j):
                    out[k] = repl
                changed += 1
        i = j
    return out, changed


def _side_from_foot_name(name: str) -> Optional[str]:
    text = str(name).lower()
    if text.endswith("_r") or text.endswith(".r") or text.endswith(" r") or text in ("right", "r") or "_right" in text:
        return "right"
    if text.endswith("_l") or text.endswith(".l") or text.endswith(" l") or text in ("left", "l") or "_left" in text:
        return "left"
    return None


def _contact_scores_to_right_left(scores: np.ndarray, foot_names: Sequence[str]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.full((arr.shape[0], 2), np.nan, dtype=np.float64)
    if arr.shape[1] >= 2 and not foot_names:
        out[:, 0] = arr[:, 0]
        out[:, 1] = arr[:, 1]
        return out
    for col in range(arr.shape[1]):
        side = _side_from_foot_name(foot_names[col] if col < len(foot_names) else "")
        if side == "right":
            out[:, 0] = arr[:, col]
        elif side == "left":
            out[:, 1] = arr[:, col]
    if np.isnan(out).any() and arr.shape[1] >= 2:
        # Fallback is marked in metadata. It is only used if skeleton names cannot
        # resolve side semantics.
        if np.isnan(out[:, 0]).all():
            out[:, 0] = arr[:, 0]
        if np.isnan(out[:, 1]).all():
            out[:, 1] = arr[:, 1]
    return out


def _reset_fk_whitebox_state(trainer: Any) -> None:
    for name in (
        "_contact_meas_prev_root_pos",
        "_contact_meas_ground_z",
        "_contact_meas_ground_z_hist",
        "_contact_meas_whitebox_debug",
    ):
        if hasattr(trainer, name):
            try:
                delattr(trainer, name)
            except Exception:
                setattr(trainer, name, None)


def _fk_support_from_state_sequence(
    trainer: Any,
    state_raw_seq: np.ndarray,
    *,
    thr: float,
) -> dict[str, Any]:
    raw = np.asarray(state_raw_seq, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] <= 0:
        return {
            "valid": False,
            "scores_rl": np.zeros((0, 2), dtype=np.float64),
            "labels_raw": [],
            "labels_norm": [],
            "arrival_label_raw": "flight",
            "arrival_label_norm": "flight",
            "foot_names": [],
            "foot_indices": [],
            "short_run_rewrites": 0,
            "side_fallback_used": True,
        }
    old_log = bool(getattr(trainer, "log_contacts_whitebox", False))
    _reset_fk_whitebox_state(trainer)
    prev_foot_pos = None
    rows: list[list[float]] = []
    foot_names: list[str] = []
    foot_indices: list[int] = []
    try:
        setattr(trainer, "log_contacts_whitebox", False)
        with torch.no_grad():
            for frame in raw:
                x_raw = torch.from_numpy(frame.reshape(1, -1)).to(device=trainer.device, dtype=torch.float32)
                meas, prev_foot_pos = compute_contact_meas_whitebox(trainer, x_raw, prev_foot_pos=prev_foot_pos)
                if torch.is_tensor(meas):
                    vals = meas.detach().cpu().reshape(-1).to(torch.float64).numpy().astype(np.float64, copy=False)
                    rows.append([float(v) for v in vals])
                else:
                    rows.append([float("nan"), float("nan")])
                idxs = getattr(trainer, "_contact_meas_foot_idxs", None)
                if isinstance(idxs, (list, tuple)) and not foot_indices:
                    foot_indices = [int(x) for x in idxs]
                    names_src = getattr(getattr(trainer, "loss_fn", None), "bone_names", None)
                    if names_src is None or (hasattr(names_src, "__len__") and len(names_src) == 0):
                        names_src = getattr(trainer, "_bone_names", None)
                    if names_src is None or (hasattr(names_src, "__len__") and len(names_src) == 0):
                        meta = getattr(getattr(trainer, "loss_fn", None), "meta", None)
                        if isinstance(meta, dict):
                            names_src = meta.get("bone_names") or meta.get("skeleton", {}).get("bone_names")
                    if names_src is None or (hasattr(names_src, "__len__") and len(names_src) == 0):
                        meta = getattr(trainer, "_bundle_meta", None)
                        if isinstance(meta, dict):
                            names_src = meta.get("bone_names") or meta.get("skeleton", {}).get("bone_names")
                    if isinstance(names_src, np.ndarray):
                        names = [str(x) for x in names_src.reshape(-1).tolist()]
                    else:
                        names = [str(x) for x in names_src] if isinstance(names_src, (list, tuple)) else []
                    foot_names = [names[i] if 0 <= int(i) < len(names) else str(i) for i in foot_indices]
    finally:
        setattr(trainer, "log_contacts_whitebox", old_log)
        _reset_fk_whitebox_state(trainer)
    scores = np.asarray(rows, dtype=np.float64)
    scores_rl = _contact_scores_to_right_left(scores, foot_names)
    labels_raw = [_label_from_scores_rl(row, thr=float(thr)) for row in scores_rl]
    labels_norm, rewrites = _short_run_normalize(labels_raw, min_len=2)
    fallback = not any(_side_from_foot_name(name) in ("right", "left") for name in foot_names)
    return {
        "valid": bool(np.isfinite(scores_rl).any()),
        "scores_rl": scores_rl,
        "labels_raw": labels_raw,
        "labels_norm": labels_norm,
        "arrival_label_raw": labels_raw[-1] if labels_raw else "flight",
        "arrival_label_norm": labels_norm[-1] if labels_norm else "flight",
        "foot_names": foot_names,
        "foot_indices": foot_indices,
        "short_run_rewrites": int(rewrites),
        "side_fallback_used": bool(fallback),
    }


def _capture_temporal_hook(model: torch.nn.Module) -> tuple[list[np.ndarray], Any]:
    captures: list[np.ndarray] = []
    module = getattr(model, "_pasa_lnq", None)
    if module is None:
        raise RuntimeError("model has no _pasa_lnq module for h_temporal tap")

    def _pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            captures.append(inputs[0].detach().cpu().numpy().astype(np.float32, copy=False))

    handle = module.register_forward_pre_hook(_pre_hook)
    return captures, handle


def _last_step_hidden(value: Any) -> np.ndarray:
    if not torch.is_tensor(value):
        return np.zeros((0,), dtype=np.float32)
    out = value.detach()
    if out.dim() == 3:
        out = out[:, -1, :]
    elif out.dim() == 1:
        out = out.unsqueeze(0)
    return out.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def _last_hook_hidden(captured: np.ndarray) -> np.ndarray:
    arr = np.asarray(captured, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, -1, :]
    elif arr.ndim == 2:
        pass
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.reshape(-1).astype(np.float32, copy=False)


def _run_sequence_for_gate(
    *,
    trainer: Any,
    model: torch.nn.Module,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    mode: str,
    apply_lambda: bool,
    early_prefix: int,
) -> dict[str, Any]:
    state_seq = _as_batch(sample, "motion", device)
    gt_seq = _as_batch(sample, "gt_motion", device)
    cond_seq = _as_batch(sample, "cond_in", device)
    cond_raw_seq = _as_batch(sample, "cond_tgt_raw", device)
    pose_hist_seq = _as_batch(sample, "pose_hist", device)
    if state_seq is None or gt_seq is None or cond_seq is None or cond_raw_seq is None:
        raise RuntimeError("sample missing required motion/gt/cond tensors")
    total = int(state_seq.shape[1])
    cfg = _rollout_kernel.resolve_free_carry_runtime_config(trainer)
    _slice_width(cfg.rot6d_x_slice, name="rot6d_x_slice")
    _slice_width(cfg.rot6d_y_slice, name="rot6d_y_slice")
    if not isinstance(cfg.rootpos_x_slice, slice):
        raise RuntimeError("rootpos_x_slice missing; FK target needs root position")

    motion = state_seq[:, 0].detach()
    motion_raw = trainer.normalizer.denorm_x(motion)
    initial_state_raw = motion_raw.detach().cpu().numpy()[0].astype(np.float64, copy=False)
    y_prev_raw = trainer._denorm(gt_seq[:, 0])
    pose_hist_state = _init_eval_pose_hist_state(
        trainer,
        ref_tensor=state_seq,
        pose_hist_seq=pose_hist_seq,
        step=0,
        device=device,
        dtype=state_seq.dtype,
    )
    plan_z = None
    h_temporal_steps: list[np.ndarray] = []
    h_final_steps: list[np.ndarray] = []
    state_raw_out: list[np.ndarray] = []
    y_raw_out: list[np.ndarray] = []
    plan_probs: list[np.ndarray] = []
    model.eval()
    captures, handle = _capture_temporal_hook(model)
    try:
        with torch.no_grad():
            for t in range(total - 1):
                if mode == "teacher":
                    motion = state_seq[:, t].detach()
                    motion_raw = trainer.normalizer.denorm_x(motion)
                    y_prev_raw = trainer._denorm(gt_seq[:, t])
                    pose_hist_t = _resolve_eval_pose_hist_input(
                        state=pose_hist_state,
                        pose_hist_seq=pose_hist_seq,
                        idx=t,
                        source="seq",
                        batch_size=int(motion.shape[0]),
                        device=device,
                        dtype=motion.dtype,
                    )
                else:
                    pose_hist_t = _resolve_eval_pose_hist_input(
                        state=pose_hist_state,
                        pose_hist_seq=pose_hist_seq,
                        idx=t,
                        source="buffer",
                        batch_size=int(motion.shape[0]),
                        device=device,
                        dtype=motion.dtype,
                    )
                before = len(captures)
                ti, rs = _time_inputs(motion, t, total - 1)
                ret = model(
                    motion,
                    cond_seq[:, t],
                    contacts=None,
                    angvel=_state_angvel(motion, cfg),
                    pose_history=pose_hist_t,
                    plan_z=plan_z,
                    phase_z=None,
                    phase_event_age=None,
                    meas_logits_prev=None,
                    time_index=ti,
                    rollout_step=rs,
                )
                if not isinstance(ret, dict) or not torch.is_tensor(ret.get("out", None)):
                    raise RuntimeError("model forward did not return ret['out']")
                if len(captures) <= before:
                    raise RuntimeError("h_temporal hook did not fire")
                h_temporal_steps.append(_last_hook_hidden(captures[-1]))
                h_final_steps.append(_last_step_hidden(ret.get("h_final", None)))
                y_used_raw = _compose_raw(trainer, ret, y_prev_raw, t, total - 1, apply_lambda=apply_lambda)
                cond_next_raw = cond_raw_seq[:, min(t, cond_raw_seq.shape[1] - 1)]
                next_raw = _rollout_kernel.apply_free_carry_raw(
                    x_prev=motion_raw,
                    y_next_raw=y_used_raw,
                    cond_next_raw=cond_next_raw,
                    rot6d_x_slice=cfg.rot6d_x_slice,
                    rot6d_y_slice=cfg.rot6d_y_slice,
                    angvel_x_slice=cfg.angvel_x_slice,
                    rootvel_x_slice=cfg.rootvel_x_slice,
                    rootpos_x_slice=cfg.rootpos_x_slice,
                    bone_hz=cfg.bone_hz,
                    columns=cfg.columns,
                )
                cp = _squeeze_step_tensor(ret.get("contacts_plan", None))
                if torch.is_tensor(cp):
                    plan_probs.append(cp.detach().cpu().numpy()[0].astype(np.float64, copy=False))
                else:
                    plan_probs.append(np.full((2,), np.nan, dtype=np.float64))
                y_raw_out.append(y_used_raw.detach().cpu().numpy()[0].astype(np.float64, copy=False))
                state_raw_out.append(next_raw.detach().cpu().numpy()[0].astype(np.float64, copy=False))
                plan_z_next = ret.get("plan_z_next", None)
                plan_z = plan_z_next.detach() if torch.is_tensor(plan_z_next) else None
                if mode == "free":
                    y_prev_raw = y_used_raw.detach()
                    motion_raw = next_raw.detach()
                    motion = trainer._diag_norm_x(motion_raw).detach()
                    pose_hist_state = advance_pose_hist_state_with_tail(
                        pose_hist_state,
                        rot_tail_raw=y_used_raw[..., cfg.rot6d_y_slice],
                    )
    finally:
        handle.remove()

    prefix_n = max(1, min(int(early_prefix), len(h_temporal_steps)))
    h_temporal_arr = np.asarray(h_temporal_steps, dtype=np.float32)
    h_final_arr = np.asarray(h_final_steps, dtype=np.float32)
    return {
        "initial_state_raw": initial_state_raw,
        "state_raw": np.asarray(state_raw_out, dtype=np.float64),
        "y_raw": np.asarray(y_raw_out, dtype=np.float64),
        "plan_probs": np.asarray(plan_probs, dtype=np.float64),
        "h_temporal_entry": h_temporal_arr[0] if h_temporal_arr.size else np.zeros((0,), dtype=np.float32),
        "h_temporal_prefix": h_temporal_arr[:prefix_n].mean(axis=0) if h_temporal_arr.size else np.zeros((0,), dtype=np.float32),
        "h_final_entry": h_final_arr[0] if h_final_arr.size else np.zeros((0,), dtype=np.float32),
        "h_temporal_shape": [int(x) for x in h_temporal_arr.shape],
        "h_final_shape": [int(x) for x in h_final_arr.shape],
        "state_shape": [int(x) for x in state_seq.shape],
        "state_dtype": str(state_seq.dtype).replace("torch.", ""),
        "state_device": str(state_seq.device),
        "cond_shape": [int(x) for x in cond_seq.shape],
        "cond_dtype": str(cond_seq.dtype).replace("torch.", ""),
        "cond_device": str(cond_seq.device),
        "cond_finite": bool(torch.isfinite(cond_seq).all().item()),
    }


def _make_group_specs(clip_names: Sequence[str], *, max_groups: int) -> list[tuple[str, float]]:
    ordered = [str(x) for x in clip_names]
    if WALK_F in ordered:
        ordered.remove(WALK_F)
        ordered.insert(0, WALK_F)
    specs: list[tuple[str, float]] = [(name, 0.0) for name in ordered]
    # Start-group replicas are deliberately named as replicas, not extra clips.
    # They are used only if the corpus has fewer than the requested 5-7 groups.
    extra_offsets = (0.5, 0.25, 0.75)
    idx = 0
    while len(specs) < int(max_groups) and ordered:
        specs.append((ordered[idx % len(ordered)], extra_offsets[(idx // max(1, len(ordered))) % len(extra_offsets)]))
        idx += 1
    return specs[: max(1, int(max_groups))]


def _build_cases(
    *,
    calib: Any,
    labels_by_clip: Mapping[str, Sequence[str]],
    clip_names: Sequence[str],
    max_groups: int,
    phase_bins: int,
) -> list[GateCase]:
    cases: list[GateCase] = []
    group_specs = _make_group_specs(clip_names, max_groups=max_groups)
    for group_idx, (clip_name, offset) in enumerate(group_specs):
        labels = list(labels_by_clip[clip_name])
        n = max(1, int(len(labels)))
        for b in range(max(1, int(phase_bins))):
            phase_float = (float(b) + 0.5 + float(offset)) / float(max(1, int(phase_bins)))
            start = int(math.floor((phase_float % 1.0) * float(n))) % n
            ph = calib._phase_at(labels, start, isolation_frames=34)
            cycle_phase = float(start) / float(n)
            cases.append(
                GateCase(
                    case_id=f"{clip_name}:g{group_idx}:p{b}:s{start}",
                    clip_name=str(clip_name),
                    start_frame=int(start),
                    group_key=f"{clip_name}:start_group:{group_idx}:offset{offset:.2f}",
                    group_index=int(group_idx),
                    phase_bin8=int(b),
                    phase_label=str(ph["phase_bin"]),
                    start_label=str(ph["label"]),
                    region_key=f"{clip_name}:{ph['region_key_suffix']}",
                    cycle_phase=float(cycle_phase),
                    cycle_phase_sin=float(math.sin(2.0 * math.pi * cycle_phase)),
                    cycle_phase_cos=float(math.cos(2.0 * math.pi * cycle_phase)),
                )
            )
    return cases


def _cond_features(cond_raw: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(cond_raw, dtype=np.float64)
    if arr.ndim == 2:
        row = arr[0]
    else:
        row = arr.reshape(-1)
    if row.size >= 3:
        dir_x = float(row[-3])
        dir_y = float(row[-2])
        speed = float(row[-1])
    elif row.size >= 2:
        dir_x = float(row[0])
        dir_y = float(row[1])
        speed = float(np.linalg.norm(row[:2]))
    elif row.size == 1:
        dir_x = float(row[0])
        dir_y = 0.0
        speed = abs(float(row[0]))
    else:
        dir_x = 0.0
        dir_y = 0.0
        speed = 0.0
    angle = math.atan2(dir_y, dir_x) if math.isfinite(dir_x) and math.isfinite(dir_y) else 0.0
    quadrant = int(math.floor(((angle + math.pi) / (2.0 * math.pi)) * 4.0)) % 4
    speed_bin = int(np.clip(math.floor(max(0.0, speed) * 4.0), 0, 7)) if math.isfinite(speed) else 0
    return {
        "cond_raw_entry": row.astype(np.float32, copy=False),
        "cond_dir_x": dir_x,
        "cond_dir_y": dir_y,
        "cond_speed": speed,
        "cond_dir_quadrant": int(quadrant),
        "cond_speed_bin": int(speed_bin),
        "cond_bin": f"q{quadrant}:s{speed_bin}",
    }


def _entry_pose(trainer: Any, sample: Mapping[str, torch.Tensor], device: torch.device, cfg: Any) -> np.ndarray:
    state_seq = _as_batch(sample, "motion", device)
    if state_seq is None:
        return np.zeros((0,), dtype=np.float32)
    with torch.no_grad():
        raw0 = trainer.normalizer.denorm_x(state_seq[:, 0].detach())
    return raw0[..., cfg.rot6d_x_slice].detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def _collect_rows(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, int(args.torch_threads)))
    calib = _load_module(CALIBRATION_PATH, "support_separability_calibration")
    helper = _load_module(HELPER_PATH, "support_separability_helper")
    args.checkpoint = Path(args.checkpoint or calib.CKPT)
    args.bundle = Path(args.bundle or calib.BUNDLE)
    args.pretrain_template = Path(args.pretrain_template or calib.PRETRAIN_TEMPLATE)
    args.encoder_bundle = Path(args.encoder_bundle or calib.ENCODER_BUNDLE)
    args.npz_root = Path(args.npz_root or calib.NPZ_ROOT)
    args.z_features = Path(args.z_features or calib.Z_FEATURES)

    state281 = load_clip_states(args.z_features, args.npz_root)
    labels_by_clip = {name: calib._labels_from_contacts(arr[:, CONTACT_SLICE]) for name, arr in state281.items()}
    clip_names = [str(x) for x in sorted(labels_by_clip.keys())]
    if WALK_F in clip_names:
        clip_names.remove(WALK_F)
        clip_names.insert(0, WALK_F)
    runner = FreeRunCycleRunner(calib._runner_args(args))
    ds_by_clip: dict[str, Any] = {}
    clip_by_name: dict[str, Any] = {}
    max_gap = int(args.max_gap)
    for clip_name in clip_names:
        ds = runner._build_dataset(args.npz_root / f"{clip_name}.npz", seq_len=max(2, max_gap + 1))
        runner._ensure_model_ready(ds)
        ds_by_clip[clip_name] = ds
        clip_by_name[clip_name] = ds.clips[0]
    if runner.trainer is None or runner.model is None:
        raise RuntimeError("runner did not initialize trainer/model")
    trainer = runner.trainer
    model = runner.model
    device = runner.device
    cfg = _rollout_kernel.resolve_free_carry_runtime_config(trainer)
    _slice_width(cfg.rot6d_x_slice, name="rot6d_x_slice")
    _slice_width(cfg.rot6d_y_slice, name="rot6d_y_slice")
    cases = _build_cases(
        calib=calib,
        labels_by_clip=labels_by_clip,
        clip_names=clip_names,
        max_groups=int(args.max_groups),
        phase_bins=int(args.phase_bins),
    )
    gaps = list(range(int(args.min_gap), int(args.max_gap) + 1, max(1, int(args.gap_step))))
    arms = ["free"]
    if bool(args.include_teacher_audit):
        arms.append("teacher")
    rows: list[dict[str, Any]] = []
    total_jobs = len(cases) * len(gaps) * len(arms)
    job_idx = 0
    t0 = time.time()
    for case in cases:
        ds = ds_by_clip[case.clip_name]
        clip = clip_by_name[case.clip_name]
        for gap in gaps:
            sample = helper._build_wrapped_window_sample(ds, clip, int(case.start_frame), int(gap) + 1)
            entry_pose = _entry_pose(trainer, sample, device, cfg)
            cond_np = np.asarray(sample["cond_tgt_raw"], dtype=np.float32)
            cond = _cond_features(cond_np)
            for arm in arms:
                job_idx += 1
                seq = _run_sequence_for_gate(
                    trainer=trainer,
                    model=model,
                    sample=sample,
                    device=device,
                    mode="free" if arm == "free" else "teacher",
                    apply_lambda=True,
                    early_prefix=int(args.early_prefix),
                )
                full_state_raw = np.concatenate(
                    [
                        np.asarray(seq["initial_state_raw"], dtype=np.float64).reshape(1, -1),
                        np.asarray(seq["state_raw"], dtype=np.float64),
                    ],
                    axis=0,
                )
                fk = _fk_support_from_state_sequence(trainer, full_state_raw, thr=float(args.fk_thr))
                valid = bool(fk["valid"]) and str(fk["arrival_label_norm"]) in CLASS_TO_ID
                row = {
                    "row_id": len(rows),
                    "valid": int(valid),
                    "invalid_reason": "" if valid else "fk_support_unavailable_or_unknown_label",
                    "arm": arm,
                    "case_id": case.case_id,
                    "clip_name": case.clip_name,
                    "start_frame": int(case.start_frame),
                    "gap": int(gap),
                    "gap_bin": f"g{int(gap):03d}",
                    "delivery_band": int(int(args.delivery_min_gap) <= int(gap) <= int(args.delivery_max_gap)),
                    "stress_audit": int(not (int(args.delivery_min_gap) <= int(gap) <= int(args.delivery_max_gap))),
                    "group_key": case.group_key,
                    "group_index": int(case.group_index),
                    "phase_bin8": int(case.phase_bin8),
                    "phase_label": case.phase_label,
                    "start_label": case.start_label,
                    "region_key": case.region_key,
                    "cycle_phase": float(case.cycle_phase),
                    "cycle_phase_sin": float(case.cycle_phase_sin),
                    "cycle_phase_cos": float(case.cycle_phase_cos),
                    "cond_dir_x": float(cond["cond_dir_x"]),
                    "cond_dir_y": float(cond["cond_dir_y"]),
                    "cond_speed": float(cond["cond_speed"]),
                    "cond_dir_quadrant": int(cond["cond_dir_quadrant"]),
                    "cond_speed_bin": int(cond["cond_speed_bin"]),
                    "cond_bin": str(cond["cond_bin"]),
                    "fk_arrival_label": str(fk["arrival_label_norm"]),
                    "fk_arrival_label_raw": str(fk["arrival_label_raw"]),
                    "fk_arrival_class_id": int(CLASS_TO_ID.get(str(fk["arrival_label_norm"]), -1)),
                    "fk_support_scores_rl_json": _json_dumps_compact(fk["scores_rl"]),
                    "fk_support_label_seq_json": _json_dumps_compact(fk["labels_norm"]),
                    "fk_support_label_raw_seq_json": _json_dumps_compact(fk["labels_raw"]),
                    "fk_short_run_rewrites": int(fk["short_run_rewrites"]),
                    "fk_foot_indices_json": _json_dumps_compact(fk["foot_indices"]),
                    "fk_foot_names_json": _json_dumps_compact(fk["foot_names"]),
                    "fk_side_fallback_used": int(bool(fk["side_fallback_used"])),
                    "entry_pose_json": _json_dumps_compact(entry_pose),
                    "entry_pose_dim": int(entry_pose.shape[0]),
                    "h_temporal_entry_json": _json_dumps_compact(seq["h_temporal_entry"]),
                    "h_temporal_prefix_json": _json_dumps_compact(seq["h_temporal_prefix"]),
                    "h_final_entry_json": _json_dumps_compact(seq["h_final_entry"]),
                    "h_temporal_dim": int(np.asarray(seq["h_temporal_entry"]).reshape(-1).shape[0]),
                    "h_final_dim": int(np.asarray(seq["h_final_entry"]).reshape(-1).shape[0]),
                    "h_temporal_shape_json": _json_dumps_compact(seq["h_temporal_shape"]),
                    "h_final_shape_json": _json_dumps_compact(seq["h_final_shape"]),
                    "cond_raw_entry_json": _json_dumps_compact(cond["cond_raw_entry"]),
                    "state_shape_json": _json_dumps_compact(seq["state_shape"]),
                    "state_dtype": seq["state_dtype"],
                    "state_device": seq["state_device"],
                    "cond_shape_json": _json_dumps_compact(seq["cond_shape"]),
                    "cond_dtype": seq["cond_dtype"],
                    "cond_device": seq["cond_device"],
                    "cond_finite": int(bool(seq["cond_finite"])),
                    "read_only_forward": 1,
                    "no_generation_model_training": 1,
                }
                rows.append(row)
                if job_idx % max(1, int(args.progress_every)) == 0:
                    elapsed = time.time() - t0
                    print(f"[collect] {job_idx}/{total_jobs} rows={len(rows)} elapsed={elapsed:.1f}s", flush=True)
    meta = {
        "task": "action_handoff_support_separability_gate",
        "date": DATE_TAG,
        "checkpoint": str(args.checkpoint),
        "bundle": str(args.bundle),
        "pretrain_template": str(args.pretrain_template),
        "encoder_bundle": str(args.encoder_bundle),
        "npz_root": str(args.npz_root),
        "z_features": str(args.z_features),
        "out_dir": str(out_dir),
        "read_only_forward": True,
        "no_generation_model_training": True,
        "no_weight_write": True,
        "production_modules_modified": False,
        "target": "FK-support-arrival label from realized rollout state_raw via existing compute_contact_meas_whitebox/fk_positions_from_rot6d; prior-laden Layer-2 common-mode physical target",
        "target_excludes": ["contacts_plan", "CONTACT_SLICE"],
        "CONTACT_SLICE_scope": "used only to build entry-phase sampling inventory through existing calibration helper, not as FK target",
        "latent_main": "h_temporal_entry captured by _pasa_lnq forward pre-hook before PASA attention output and command-conditioned FiLM",
        "latent_aux": "h_final_entry is auxiliary only because it includes PASA window context and FiLM(cond)",
        "rows": int(len(rows)),
        "cases": int(len(cases)),
        "gaps": [int(x) for x in gaps],
        "arms": arms,
        "groups": sorted({str(c.group_key) for c in cases}),
        "tensor_contract": {
            "state_seq": "[1,H,Dx] float32 device",
            "cond_seq": "[1,H,Dc] float32 device",
            "entry_pose": "[J*6] float32 cpu serialized JSON from denormalized X BoneRotations6D",
            "h_temporal_entry": "[hidden_dim] float32 cpu serialized JSON captured at rollout step 0",
            "h_temporal_prefix": "[hidden_dim] float32 cpu serialized JSON mean over first early_prefix steps",
            "h_final_entry": "[hidden_dim] float32 cpu serialized JSON auxiliary",
            "fk_support_scores_rl": "[H+1,2] float64 cpu serialized JSON in right,left order",
        },
    }
    return rows, meta


def _parse_vector(row: Mapping[str, Any], key: str) -> np.ndarray:
    value = _json_loads(row.get(key), [])
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _matrix_from_json(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    vecs = [_parse_vector(row, key) for row in rows]
    width = max([int(v.shape[0]) for v in vecs], default=0)
    out = np.zeros((len(vecs), width), dtype=np.float64)
    for i, vec in enumerate(vecs):
        n = min(width, int(vec.shape[0]))
        if n > 0:
            out[i, :n] = vec[:n]
    return out


def _one_hot(values: Sequence[Any], levels: Sequence[Any]) -> np.ndarray:
    level_list = [str(x) for x in levels]
    idx = {v: i for i, v in enumerate(level_list)}
    out = np.zeros((len(values), len(level_list)), dtype=np.float64)
    for r, value in enumerate(values):
        j = idx.get(str(value))
        if j is not None:
            out[r, j] = 1.0
    return out


def _base_features(rows: Sequence[Mapping[str, Any]], *, levels: Mapping[str, Sequence[Any]], layer: str) -> np.ndarray:
    n = len(rows)
    phase_vals = [row.get("phase_bin8", "") for row in rows]
    gap_vals = [row.get("gap_bin", "") for row in rows]
    phase = _one_hot(phase_vals, levels["phase"])
    gap = _one_hot(gap_vals, levels["gap"])
    cond = np.asarray(
        [
            [
                _finite_float(row.get("cond_dir_x"), 0.0),
                _finite_float(row.get("cond_dir_y"), 0.0),
                _finite_float(row.get("cond_speed"), 0.0),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    phase_gap = _one_hot([f"{p}|{g}" for p, g in zip(phase_vals, gap_vals)], levels["phase_gap"])
    phase_cond = np.einsum("np,nc->npc", phase, cond).reshape(n, -1)
    gap_cond = np.einsum("ng,nc->ngc", gap, cond).reshape(n, -1)
    cols = [phase, gap, cond, phase_gap, phase_cond, gap_cond]
    if layer in ("B2", "B3", "aug"):
        cols.append(
            np.asarray(
                [
                    [
                        _finite_float(row.get("cycle_phase_sin"), 0.0),
                        _finite_float(row.get("cycle_phase_cos"), 1.0),
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )
        )
    return np.concatenate(cols, axis=1) if cols else np.zeros((n, 0), dtype=np.float64)


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mu = np.nanmean(x_train, axis=0) if x_train.size else np.zeros((x_train.shape[1],), dtype=np.float64)
    std = np.nanstd(x_train, axis=0) if x_train.size else np.ones((x_train.shape[1],), dtype=np.float64)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        np.nan_to_num((x_train - mu) / std, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num((x_test - mu) / std, nan=0.0, posinf=0.0, neginf=0.0),
        {"standardized_dim": int(x_train.shape[1]), "std_floor_n": int((std <= 1e-8).sum())},
    )


def _pca_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if x_train.size == 0 or int(n_components) <= 0:
        return (
            np.zeros((x_train.shape[0], 0), dtype=np.float64),
            np.zeros((x_test.shape[0], 0), dtype=np.float64),
            {"pc_dim": 0, "explained_var": []},
        )
    mu = np.nanmean(x_train, axis=0)
    train = np.nan_to_num(x_train - mu, nan=0.0, posinf=0.0, neginf=0.0)
    test = np.nan_to_num(x_test - mu, nan=0.0, posinf=0.0, neginf=0.0)
    max_rank = max(0, min(train.shape[0] - 1, train.shape[1], int(n_components)))
    if max_rank <= 0:
        return (
            np.zeros((x_train.shape[0], 0), dtype=np.float64),
            np.zeros((x_test.shape[0], 0), dtype=np.float64),
            {"pc_dim": 0, "explained_var": []},
        )
    _, s, vt = np.linalg.svd(train, full_matrices=False)
    comp = vt[:max_rank]
    denom = float(np.sum(s * s))
    ev = ((s[:max_rank] * s[:max_rank]) / denom).tolist() if denom > 1e-12 else [0.0] * max_rank
    return train @ comp.T, test @ comp.T, {"pc_dim": int(max_rank), "explained_var": [float(x) for x in ev]}


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _fit_softmax_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    num_classes: int,
    max_iter: int,
) -> dict[str, Any]:
    x = torch.as_tensor(np.asarray(x_train, dtype=np.float32), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y_train, dtype=np.int64), dtype=torch.long)
    n = int(x.shape[0])
    d = int(x.shape[1])
    k = int(num_classes)
    w = torch.zeros((d, k), dtype=torch.float32, requires_grad=True)
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=k).astype(np.float64) + 1.0
    pri = counts / counts.sum()
    b = torch.tensor(np.log(pri / np.exp(np.mean(np.log(pri)))), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=int(max_iter), line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        logits = x @ w + b
        loss = torch.nn.functional.cross_entropy(logits, y)
        if d > 0 and float(alpha) > 0.0:
            loss = loss + 0.5 * float(alpha) * torch.sum(w * w) / float(max(1, n))
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        w_np = w.detach().cpu().numpy().astype(np.float64, copy=False)
        b_np = b.detach().cpu().numpy().astype(np.float64, copy=False)
    return {
        "w": w_np,
        "b": b_np,
        "alpha": float(alpha),
        "coef_nonzero_abs_gt_1e-6": int((np.abs(w_np) > 1e-6).sum()),
        "param_count": int(w_np.size + b_np.size),
        "weight_l2": float(np.sqrt(np.sum(w_np * w_np))),
    }


def _predict_proba(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    w = np.asarray(model["w"], dtype=np.float64)
    b = np.asarray(model["b"], dtype=np.float64)
    return _softmax_np(np.asarray(x, dtype=np.float64) @ w + b.reshape(1, -1))


def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(prob, dtype=np.float64)
    if y.size == 0:
        return {"ce": float("nan"), "brier": float("nan"), "balanced_acc": float("nan"), "acc": float("nan")}
    eps = 1e-12
    ce = float(-np.mean(np.log(np.clip(p[np.arange(y.size), y], eps, 1.0))))
    oh = np.zeros_like(p)
    oh[np.arange(y.size), y] = 1.0
    brier = float(np.mean(np.sum((p - oh) * (p - oh), axis=1)))
    pred = np.argmax(p, axis=1)
    acc = float(np.mean(pred == y))
    recalls: list[float] = []
    for c in sorted(set(int(v) for v in y.tolist())):
        mask = y == c
        if int(mask.sum()) > 0:
            recalls.append(float(np.mean(pred[mask] == y[mask])))
    bacc = float(np.mean(recalls)) if recalls else float("nan")
    return {"ce": ce, "brier": brier, "balanced_acc": bacc, "acc": acc}


def _majority_model(y_train: np.ndarray, *, num_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=int(num_classes)).astype(np.float64) + 1e-6
    return counts / counts.sum()


def _class_prior_metrics(y: np.ndarray, prior: np.ndarray) -> dict[str, Any]:
    p = np.repeat(np.asarray(prior, dtype=np.float64).reshape(1, -1), int(len(y)), axis=0)
    return _metrics(y, p)


def _evaluate_design(
    *,
    rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    groups: np.ndarray,
    levels: Mapping[str, Sequence[Any]],
    pose_matrix: np.ndarray,
    latent_matrix: Optional[np.ndarray],
    layer: str,
    alpha: float,
    pose_pcs: int,
    max_iter: int,
    latent_pcs: int = 0,
) -> dict[str, Any]:
    unique_groups = [str(g) for g in sorted(set(str(x) for x in groups.tolist()))]
    folds: list[dict[str, Any]] = []
    train_metrics_accum: list[dict[str, float]] = []
    test_metrics_accum: list[dict[str, float]] = []
    coef_nonzero: list[int] = []
    pc_dims: list[int] = []
    latent_pc_dims: list[int] = []
    for group in unique_groups:
        test_mask = groups == group
        train_mask = ~test_mask
        y_train = y[train_mask]
        y_test = y[test_mask]
        if len(set(int(v) for v in y_train.tolist())) < 2:
            folds.append({"group": group, "skipped": True, "reason": "train_has_lt_2_classes"})
            continue
        base_train = _base_features([rows[i] for i in np.where(train_mask)[0]], levels=levels, layer=layer)
        base_test = _base_features([rows[i] for i in np.where(test_mask)[0]], levels=levels, layer=layer)
        parts_train = [base_train]
        parts_test = [base_test]
        pc_meta = {"pc_dim": 0, "explained_var": []}
        if layer in ("B3", "aug"):
            pose_train, pose_test, pc_meta = _pca_train_test(
                pose_matrix[train_mask],
                pose_matrix[test_mask],
                n_components=int(pose_pcs),
            )
            parts_train.append(pose_train)
            parts_test.append(pose_test)
        if latent_matrix is not None:
            latent_pc_meta = {"pc_dim": 0, "explained_var": []}
            if int(latent_pcs) > 0:
                latent_train, latent_test, latent_pc_meta = _pca_train_test(
                    latent_matrix[train_mask],
                    latent_matrix[test_mask],
                    n_components=int(latent_pcs),
                )
            else:
                latent_train = latent_matrix[train_mask]
                latent_test = latent_matrix[test_mask]
                latent_pc_meta = {
                    "pc_dim": int(latent_train.shape[1]),
                    "explained_var": [],
                    "raw_unprojected": True,
                }
            parts_train.append(latent_train)
            parts_test.append(latent_test)
            latent_pc_dims.append(int(latent_pc_meta["pc_dim"]))
        x_train = np.concatenate(parts_train, axis=1) if parts_train else np.zeros((int(train_mask.sum()), 0))
        x_test = np.concatenate(parts_test, axis=1) if parts_test else np.zeros((int(test_mask.sum()), 0))
        x_train, x_test, std_meta = _standardize_train_test(x_train, x_test)
        model = _fit_softmax_ridge(
            x_train,
            y_train,
            alpha=float(alpha),
            num_classes=len(CLASSES),
            max_iter=int(max_iter),
        )
        p_train = _predict_proba(model, x_train)
        p_test = _predict_proba(model, x_test)
        train_m = _metrics(y_train, p_train)
        test_m = _metrics(y_test, p_test)
        train_metrics_accum.append(train_m)
        test_metrics_accum.append(test_m)
        coef_nonzero.append(int(model["coef_nonzero_abs_gt_1e-6"]))
        pc_dims.append(int(pc_meta["pc_dim"]))
        folds.append(
            {
                "group": group,
                "skipped": False,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "train_class_counts": {CLASSES[i]: int(np.sum(y_train == i)) for i in range(len(CLASSES))},
                "test_class_counts": {CLASSES[i]: int(np.sum(y_test == i)) for i in range(len(CLASSES))},
                "train": train_m,
                "heldout": test_m,
                "alpha": float(alpha),
                "coef_nonzero_abs_gt_1e-6": int(model["coef_nonzero_abs_gt_1e-6"]),
                "param_count": int(model["param_count"]),
                "pc_meta": pc_meta,
                "latent_pc_meta": latent_pc_meta if latent_matrix is not None else None,
                "std_meta": std_meta,
            }
        )
    if not test_metrics_accum:
        return {"folds": folds, "mean_train": {}, "mean_heldout": {}, "train_minus_heldout": {}, "valid_folds": 0}

    def mean_metric(metric_rows: Sequence[Mapping[str, float]], key: str) -> float:
        vals = [_finite_float(row.get(key), float("nan")) for row in metric_rows]
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    mean_train = {key: mean_metric(train_metrics_accum, key) for key in ("ce", "brier", "balanced_acc", "acc")}
    mean_test = {key: mean_metric(test_metrics_accum, key) for key in ("ce", "brier", "balanced_acc", "acc")}
    gap = {
        "ce_heldout_minus_train": float(mean_test["ce"] - mean_train["ce"]),
        "brier_heldout_minus_train": float(mean_test["brier"] - mean_train["brier"]),
        "balanced_acc_train_minus_heldout": float(mean_train["balanced_acc"] - mean_test["balanced_acc"]),
    }
    return {
        "folds": folds,
        "mean_train": mean_train,
        "mean_heldout": mean_test,
        "train_heldout_gap": gap,
        "valid_folds": int(len(test_metrics_accum)),
        "alpha": float(alpha),
        "coef_nonzero_abs_gt_1e-6_mean": float(np.mean(coef_nonzero)) if coef_nonzero else 0.0,
        "pose_pc_dim_mean": float(np.mean(pc_dims)) if pc_dims else 0.0,
        "latent_pc_dim_mean": float(np.mean(latent_pc_dims)) if latent_pc_dims else 0.0,
        "latent_pcs_requested": int(latent_pcs),
        "feature_layer": layer,
        "latent_dim": (
            int(np.mean(latent_pc_dims))
            if latent_matrix is not None and latent_pc_dims
            else int(latent_matrix.shape[1])
            if latent_matrix is not None
            else 0
        ),
        "latent_raw_dim": int(latent_matrix.shape[1]) if latent_matrix is not None else 0,
    }


def _evaluate_b0(rows: Sequence[Mapping[str, Any]], y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    unique_groups = [str(g) for g in sorted(set(str(x) for x in groups.tolist()))]
    folds: list[dict[str, Any]] = []
    train_accum: list[dict[str, float]] = []
    test_accum: list[dict[str, float]] = []
    for group in unique_groups:
        test_mask = groups == group
        train_mask = ~test_mask
        y_train = y[train_mask]
        y_test = y[test_mask]
        prior = _majority_model(y_train, num_classes=len(CLASSES))
        train_m = _class_prior_metrics(y_train, prior)
        test_m = _class_prior_metrics(y_test, prior)
        train_accum.append(train_m)
        test_accum.append(test_m)
        folds.append(
            {
                "group": group,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "train_class_counts": {CLASSES[i]: int(np.sum(y_train == i)) for i in range(len(CLASSES))},
                "test_class_counts": {CLASSES[i]: int(np.sum(y_test == i)) for i in range(len(CLASSES))},
                "prior": {CLASSES[i]: float(prior[i]) for i in range(len(CLASSES))},
                "train": train_m,
                "heldout": test_m,
            }
        )

    def mean_metric(metric_rows: Sequence[Mapping[str, float]], key: str) -> float:
        vals = [_finite_float(row.get(key), float("nan")) for row in metric_rows]
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    mean_train = {key: mean_metric(train_accum, key) for key in ("ce", "brier", "balanced_acc", "acc")}
    mean_test = {key: mean_metric(test_accum, key) for key in ("ce", "brier", "balanced_acc", "acc")}
    return {
        "folds": folds,
        "mean_train": mean_train,
        "mean_heldout": mean_test,
        "train_heldout_gap": {
            "ce_heldout_minus_train": float(mean_test["ce"] - mean_train["ce"]),
            "brier_heldout_minus_train": float(mean_test["brier"] - mean_train["brier"]),
            "balanced_acc_train_minus_heldout": float(mean_train["balanced_acc"] - mean_test["balanced_acc"]),
        },
        "valid_folds": int(len(test_accum)),
        "alpha": None,
        "coef_nonzero_abs_gt_1e-6_mean": 0.0,
        "feature_layer": "B0",
        "latent_dim": 0,
    }


def _delta_vs_base(aug: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    aug_folds = {str(f.get("group")): f for f in aug.get("folds", []) if not bool(f.get("skipped", False))}
    base_folds = {str(f.get("group")): f for f in base.get("folds", []) if not bool(f.get("skipped", False))}
    fold_rows: list[dict[str, Any]] = []
    for group in sorted(set(aug_folds.keys()) & set(base_folds.keys())):
        a = aug_folds[group]["heldout"]
        b = base_folds[group]["heldout"]
        d_ce = _finite_float(a.get("ce")) - _finite_float(b.get("ce"))
        d_brier = _finite_float(a.get("brier")) - _finite_float(b.get("brier"))
        d_bacc = _finite_float(a.get("balanced_acc")) - _finite_float(b.get("balanced_acc"))
        fold_rows.append(
            {
                "group": group,
                "delta_ce_aug_minus_b3": float(d_ce),
                "delta_brier_aug_minus_b3": float(d_brier),
                "balanced_acc_lift": float(d_bacc),
                "ce_sign": "positive_lift" if d_ce < 0.0 else "no_lift",
                "brier_sign": "positive_lift" if d_brier < 0.0 else "no_lift",
                "balanced_acc_sign": "positive_lift" if d_bacc > 0.0 else "no_lift",
            }
        )
    if not fold_rows:
        return {"folds": [], "mean_delta_ce": float("nan"), "mean_delta_brier": float("nan"), "mean_balanced_acc_lift": float("nan")}
    return {
        "folds": fold_rows,
        "mean_delta_ce": float(np.mean([r["delta_ce_aug_minus_b3"] for r in fold_rows])),
        "mean_delta_brier": float(np.mean([r["delta_brier_aug_minus_b3"] for r in fold_rows])),
        "mean_balanced_acc_lift": float(np.mean([r["balanced_acc_lift"] for r in fold_rows])),
        "ce_positive_lift_groups": int(sum(1 for r in fold_rows if r["delta_ce_aug_minus_b3"] < 0.0)),
        "brier_positive_lift_groups": int(sum(1 for r in fold_rows if r["delta_brier_aug_minus_b3"] < 0.0)),
        "balanced_acc_positive_lift_groups": int(sum(1 for r in fold_rows if r["balanced_acc_lift"] > 0.0)),
        "group_n": int(len(fold_rows)),
    }


def _shuffle_latent_condition_preserving(
    rows: Sequence[Mapping[str, Any]],
    latent: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    keys = [
        (
            str(row.get("phase_bin8")),
            str(row.get("gap_bin")),
            str(row.get("cond_bin")),
        )
        for row in rows
    ]
    out = np.asarray(latent, dtype=np.float64).copy()
    by_key: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        by_key[key].append(idx)
    for idxs in by_key.values():
        if len(idxs) >= 2:
            perm = np.asarray(idxs, dtype=np.int64).copy()
            rng.shuffle(perm)
            out[np.asarray(idxs, dtype=np.int64)] = latent[perm]
    return out


def _levels_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    phase = sorted({str(row.get("phase_bin8")) for row in rows})
    gap = sorted({str(row.get("gap_bin")) for row in rows})
    phase_gap = sorted({f"{row.get('phase_bin8')}|{row.get('gap_bin')}" for row in rows})
    return {"phase": phase, "gap": gap, "phase_gap": phase_gap}


def _gate_subset(rows: Sequence[Mapping[str, Any]], *, band: str) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for row in rows:
        if str(row.get("arm")) != "free":
            continue
        if _finite_int(row.get("valid"), 0) != 1:
            continue
        if band == "delivery" and _finite_int(row.get("delivery_band"), 0) != 1:
            continue
        out.append(row)
    return out


def _teacher_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if _finite_int(row.get("valid"), 0) != 1:
            continue
        by_key[(str(row.get("case_id")), _finite_int(row.get("gap"), 0))][str(row.get("arm"))] = row
    pairs: list[dict[str, Any]] = []
    for (case_id, gap), arms in sorted(by_key.items()):
        if "free" not in arms or "teacher" not in arms:
            continue
        free = arms["free"]
        teacher = arms["teacher"]
        pairs.append(
            {
                "case_id": case_id,
                "gap": int(gap),
                "group_key": str(free.get("group_key")),
                "free_label": str(free.get("fk_arrival_label")),
                "teacher_label": str(teacher.get("fk_arrival_label")),
                "same": bool(str(free.get("fk_arrival_label")) == str(teacher.get("fk_arrival_label"))),
            }
        )
    delivery = [p for p in pairs if 12 <= int(p["gap"]) <= 30]
    return {
        "pair_n": int(len(pairs)),
        "delivery_pair_n": int(len(delivery)),
        "label_agreement_rate_all": float(np.mean([p["same"] for p in pairs])) if pairs else None,
        "label_agreement_rate_delivery": float(np.mean([p["same"] for p in delivery])) if delivery else None,
        "disagreement_rate_delivery": float(1.0 - np.mean([p["same"] for p in delivery])) if delivery else None,
        "pairs_preview": pairs[:20],
    }


def _recalc_from_rows(
    rows_csv: Path,
    *,
    alpha: float,
    pose_pcs: int,
    latent_pcs: int,
    max_iter: int,
    null_repeats: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_rows = _read_csv(rows_csv)
    rows = _gate_subset(all_rows, band="delivery")
    y = np.asarray([_finite_int(row.get("fk_arrival_class_id"), -1) for row in rows], dtype=np.int64)
    keep = (y >= 0) & (y < len(CLASSES))
    rows = [row for row, ok in zip(rows, keep.tolist()) if ok]
    y = y[keep]
    groups = np.asarray([str(row.get("group_key")) for row in rows], dtype=object)
    class_counts = {CLASSES[i]: int(np.sum(y == i)) for i in range(len(CLASSES))}
    group_counts = dict(Counter(groups.tolist()))
    levels = _levels_from_rows(rows)
    pose = _matrix_from_json(rows, "entry_pose_json")
    h_temporal_entry = _matrix_from_json(rows, "h_temporal_entry_json")
    h_temporal_prefix = _matrix_from_json(rows, "h_temporal_prefix_json")
    h_final_entry = _matrix_from_json(rows, "h_final_entry_json")

    b0 = _evaluate_b0(rows, y, groups)
    b1 = _evaluate_design(
        rows=rows,
        y=y,
        groups=groups,
        levels=levels,
        pose_matrix=pose,
        latent_matrix=None,
        layer="B1",
        alpha=float(alpha),
        pose_pcs=int(pose_pcs),
        max_iter=int(max_iter),
    )
    b2 = _evaluate_design(
        rows=rows,
        y=y,
        groups=groups,
        levels=levels,
        pose_matrix=pose,
        latent_matrix=None,
        layer="B2",
        alpha=float(alpha),
        pose_pcs=int(pose_pcs),
        max_iter=int(max_iter),
    )
    b3 = _evaluate_design(
        rows=rows,
        y=y,
        groups=groups,
        levels=levels,
        pose_matrix=pose,
        latent_matrix=None,
        layer="B3",
        alpha=float(alpha),
        pose_pcs=int(pose_pcs),
        max_iter=int(max_iter),
    )
    ladder_summary = {
        "scope": "delivery_gap_12_30_free_run_only",
        "n": int(len(rows)),
        "effective_group_n": int(len(group_counts)),
        "group_counts": group_counts,
        "class_counts": class_counts,
        "classes": list(CLASSES),
        "levels": {key: list(value) for key, value in levels.items()},
        "model_family": "multinomial softmax ridge logistic implemented with torch LBFGS on diagnostic features only",
        "alpha": float(alpha),
        "pose_pcs": int(pose_pcs),
        "latent_pcs": int(latent_pcs),
        "B0": b0,
        "B1": b1,
        "B2": b2,
        "B3": b3,
    }

    augmented_specs = {
        "h_temporal_entry": h_temporal_entry,
        "h_temporal_prefix": h_temporal_prefix,
        "h_final_entry_aux": h_final_entry,
    }
    augmented: dict[str, Any] = {}
    lift: dict[str, Any] = {}
    for name, latent in augmented_specs.items():
        aug = _evaluate_design(
            rows=rows,
            y=y,
            groups=groups,
            levels=levels,
            pose_matrix=pose,
            latent_matrix=latent,
            layer="aug",
            alpha=float(alpha),
            pose_pcs=int(pose_pcs),
            max_iter=int(max_iter),
            latent_pcs=int(latent_pcs),
        )
        augmented[name] = aug
        lift[name] = _delta_vs_base(aug, b3)

    null_runs: dict[str, list[dict[str, Any]]] = {key: [] for key in ("h_temporal_entry", "h_temporal_prefix")}
    for rep in range(max(0, int(null_repeats))):
        for name, latent in (("h_temporal_entry", h_temporal_entry), ("h_temporal_prefix", h_temporal_prefix)):
            shuf = _shuffle_latent_condition_preserving(rows, latent, seed=int(seed) + 1009 * rep + (0 if name.endswith("entry") else 17))
            aug_null = _evaluate_design(
                rows=rows,
                y=y,
                groups=groups,
                levels=levels,
                pose_matrix=pose,
                latent_matrix=shuf,
                layer="aug",
                alpha=float(alpha),
                pose_pcs=int(pose_pcs),
                max_iter=int(max_iter),
                latent_pcs=int(latent_pcs),
            )
            d = _delta_vs_base(aug_null, b3)
            d["rep"] = int(rep)
            null_runs[name].append(d)

    def _null_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"repeats": 0}
        return {
            "repeats": int(len(items)),
            "mean_delta_ce_mean": float(np.mean([_finite_float(x.get("mean_delta_ce")) for x in items])),
            "mean_delta_brier_mean": float(np.mean([_finite_float(x.get("mean_delta_brier")) for x in items])),
            "mean_balanced_acc_lift_mean": float(np.mean([_finite_float(x.get("mean_balanced_acc_lift")) for x in items])),
            "ce_positive_lift_groups_mean": float(np.mean([_finite_float(x.get("ce_positive_lift_groups"), 0.0) for x in items])),
            "raw": list(items),
        }

    lift_summary = {
        "scope": "delivery_gap_12_30_free_run_only",
        "delta_definition": "augmented minus B3 on grouped held-out folds; ΔCE<0 and ΔBrier<0 are positive lift, balanced_acc_lift>0 is positive",
        "main_latent": "h_temporal_entry",
        "auxiliary_latents": ["h_temporal_prefix", "h_final_entry_aux"],
        "latent_pcs": int(latent_pcs),
        "augmented": augmented,
        "lift_vs_B3": lift,
        "condition_preserving_null": {
            "strata": "phase_bin8 × gap_bin × cond_bin",
            "repeats": int(null_repeats),
            "h_temporal_entry": _null_summary(null_runs["h_temporal_entry"]),
            "h_temporal_prefix": _null_summary(null_runs["h_temporal_prefix"]),
        },
    }

    invalid_reasons: list[str] = []
    if len(rows) <= 0:
        invalid_reasons.append("no_valid_delivery_free_rows")
    if len(group_counts) < 5:
        invalid_reasons.append(f"effective_group_n_lt_5:{len(group_counts)}")
    if sum(1 for v in class_counts.values() if v > 0) < 2:
        invalid_reasons.append(f"target_class_degenerate:{class_counts}")
    if b3.get("valid_folds", 0) < max(2, min(5, len(group_counts))):
        invalid_reasons.append(f"B3_valid_folds_insufficient:{b3.get('valid_folds', 0)}")
    chance = 1.0 / float(max(1, sum(1 for v in class_counts.values() if v > 0)))
    b3_bacc = _finite_float(b3.get("mean_heldout", {}).get("balanced_acc"), float("nan"))
    b0_bacc = _finite_float(b0.get("mean_heldout", {}).get("balanced_acc"), float("nan"))
    if math.isfinite(b3_bacc) and b3_bacc <= chance + 0.02 and b3_bacc <= b0_bacc + 0.02:
        invalid_reasons.append(f"B3_heldout_near_chance:b3_bacc={b3_bacc:.6f}:chance={chance:.6f}")
    b3_gap_ce = _finite_float(b3.get("train_heldout_gap", {}).get("ce_heldout_minus_train"), 0.0)
    if b3_gap_ce > 1.25:
        invalid_reasons.append(f"B3_train_heldout_ce_gap_large:{b3_gap_ce:.6f}")

    main_lift = lift.get("h_temporal_entry", {})
    group_n = int(main_lift.get("group_n", 0) or 0)
    ce_pos = int(main_lift.get("ce_positive_lift_groups", 0) or 0)
    brier_pos = int(main_lift.get("brier_positive_lift_groups", 0) or 0)
    bacc_pos = int(main_lift.get("balanced_acc_positive_lift_groups", 0) or 0)
    stable_threshold = max(1, int(math.floor(group_n / 2.0) + 1))
    if group_n >= 6:
        stable_threshold = 4 if group_n == 6 else 5
    elif group_n == 5:
        stable_threshold = 4
    mean_dce = _finite_float(main_lift.get("mean_delta_ce"), float("nan"))
    mean_dbrier = _finite_float(main_lift.get("mean_delta_brier"), float("nan"))
    mean_bacc_lift = _finite_float(main_lift.get("mean_balanced_acc_lift"), float("nan"))
    null_entry = lift_summary["condition_preserving_null"]["h_temporal_entry"]
    null_dce = _finite_float(null_entry.get("mean_delta_ce_mean"), float("nan"))
    null_disappears = (not math.isfinite(null_dce)) or null_dce >= -1e-4
    pass_conditions = [
        math.isfinite(mean_dce) and mean_dce < 0.0,
        math.isfinite(mean_dbrier) and mean_dbrier < 0.0,
        math.isfinite(mean_bacc_lift) and mean_bacc_lift > 0.0,
        ce_pos >= stable_threshold,
        brier_pos >= stable_threshold,
        bacc_pos >= stable_threshold,
        bool(null_disappears),
    ]
    if invalid_reasons:
        verdict = "INVALID"
        downstream = "重设计采样/平衡 class 后重跑闸；不下 PASS/FAIL 结论。"
    elif all(pass_conditions):
        verdict = "PASS"
        downstream = "进入下一阶段 interventional 验证（GoalHead / latent perturbation 训练）；当前 PASS 仍只是 observational 必要非充分条件。"
    else:
        verdict = "FAIL"
        downstream = "当前证据下不值得继续 support-control 训练，回落交付 (a) kinematics-only；不是原理不可能。"

    stress_rows = _gate_subset(all_rows, band="all")
    stress_free = [r for r in stress_rows if _finite_int(r.get("stress_audit"), 0) == 1]
    redteam = {
        "verdict": verdict,
        "downstream": downstream,
        "invalid_reasons": invalid_reasons,
        "observational_semantics": (
            "PASS only means h_temporal_entry contains conditionally decodable FK-support information after "
            "phase/gap/command/continuous phase/entry-pose PCs; it is necessary not sufficient for control."
        ),
        "negative_scope": "FAIL means current evidence does not justify support-control training, not that support control is impossible in principle.",
        "decision_inputs": {
            "main_latent": "h_temporal_entry",
            "mean_delta_ce": mean_dce,
            "mean_delta_brier": mean_dbrier,
            "mean_balanced_acc_lift": mean_bacc_lift,
            "ce_positive_lift_groups": ce_pos,
            "brier_positive_lift_groups": brier_pos,
            "balanced_acc_positive_lift_groups": bacc_pos,
            "group_n": group_n,
            "stable_group_threshold": stable_threshold,
            "null_mean_delta_ce": null_dce,
            "null_disappears": bool(null_disappears),
            "pass_conditions": [bool(x) for x in pass_conditions],
        },
        "rows_recalc": {
            "rows_csv": str(rows_csv),
            "delivery_free_n": int(len(rows)),
            "stress_free_n": int(len(stress_free)),
            "effective_group_n": int(len(group_counts)),
            "class_counts": class_counts,
            "teacher_audit": _teacher_audit(all_rows),
        },
        "method_redlines": {
            "target": "FK-support-arrival from realized rollout pose; prior-laden Layer-2 common-mode physical target",
            "target_not_used": ["contacts_plan", "CONTACT_SLICE"],
            "latent_main": "h_temporal_entry/pre-PASA/pre-FiLM hook",
            "h_final_scope": "auxiliary only due cond-FiLM and PASA window leakage",
            "endpoint_latent_scope": "not collected and not used for PASS",
            "group_cv": "leave-one-group-out with group_key=clip/region/start-group",
        },
    }
    return ladder_summary, lift_summary, redteam


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only grouped separability gate for action-handoff support-control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--pretrain-template", type=Path, default=None)
    parser.add_argument("--encoder-bundle", type=Path, default=None)
    parser.add_argument("--npz-root", type=Path, default=None)
    parser.add_argument("--z-features", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--min-gap", type=int, default=12)
    parser.add_argument("--max-gap", type=int, default=84)
    parser.add_argument("--gap-step", type=int, default=6)
    parser.add_argument("--delivery-min-gap", type=int, default=12)
    parser.add_argument("--delivery-max-gap", type=int, default=30)
    parser.add_argument("--phase-bins", type=int, default=8)
    parser.add_argument("--max-groups", type=int, default=6)
    parser.add_argument("--early-prefix", type=int, default=6)
    parser.add_argument("--fk-thr", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--pose-pcs", type=int, default=12)
    parser.add_argument(
        "--latent-pcs",
        type=int,
        default=0,
        help="If >0, project each latent block with train-fold PCA before augmentation; 0 keeps the raw latent.",
    )
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--null-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--include-teacher-audit", action="store_true", default=True)
    parser.add_argument("--no-teacher-audit", dest="include_teacher_audit", action="store_false")
    parser.add_argument("--recalc-only", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path("debug_output") / f"_tmp_support_separability_gate_{DATE_TAG}_{stamp}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / "rows.csv"
    if args.recalc_only is not None:
        rows_csv = Path(args.recalc_only)
    else:
        rows, meta = _collect_rows(args, out_dir)
        _write_csv(rows_csv, rows)
        _write_json(out_dir / "collection_meta.json", meta)
    ladder, lift, redteam = _recalc_from_rows(
        rows_csv,
        alpha=float(args.alpha),
        pose_pcs=int(args.pose_pcs),
        latent_pcs=int(args.latent_pcs),
        max_iter=int(args.max_iter),
        null_repeats=int(args.null_repeats),
        seed=int(args.seed),
    )
    _write_json(out_dir / "ladder_summary.json", ladder)
    _write_json(out_dir / "lift_summary.json", lift)
    _write_json(out_dir / "redteam_recalc.json", redteam)
    print(f"wrote {rows_csv}")
    print(f"wrote {out_dir / 'ladder_summary.json'}")
    print(f"wrote {out_dir / 'lift_summary.json'}")
    print(f"wrote {out_dir / 'redteam_recalc.json'}")
    print(json.dumps(_jsonable(redteam["decision_inputs"]), ensure_ascii=False, indent=2, allow_nan=False))
    print(f"VERDICT={redteam['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
