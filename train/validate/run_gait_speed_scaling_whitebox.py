#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.geometry import geodesic_R, reproject_rot6d, root_relative_matrices, rot6d_to_matrix
from train.validate.run_freerun_cycles import (
    FreeRunCycleRunner,
    _build_full_cycle_sample,
    _load_json,
    _resolve_npz_path,
)


DEFAULT_BUNDLE = "raw_data/processed_data/norm_template.json"
DEFAULT_PRETRAIN_TEMPLATE = "models/pretrain_template.json"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
DEFAULT_SCALES = "0.8,0.9,1.0,1.1,1.2"
DEFAULT_ENCODER_CANDIDATES = (
    "models/motion_encoder_equiv.pt",
    "models/motion_encoder_equiv.pt.best.pt",
    "models/motion_encoder_equiv_stageA.pt",
)


@dataclass
class RolloutTrace:
    scale: float
    pred_y_raw: torch.Tensor
    gt_y_raw: torch.Tensor
    motion_raw: torch.Tensor
    root_speed: np.ndarray
    root_pos: np.ndarray
    contacts_plan: Optional[np.ndarray]
    contacts_meas: Optional[np.ndarray]
    contacts_teacher: Optional[np.ndarray]
    contact_source_used: str
    touchdown_channel: int
    touchdown_indices: List[int]
    cycle_start_indices: List[int]
    cycle_stop_indices: List[int]
    td_unstable: bool


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run a first-pass white-box gait speed scaling evaluator on Walk-style clips."
    )
    ap.add_argument("--clip", required=True, help="Clip name, e.g. Walk_F")
    ap.add_argument("--model", required=True, help="Checkpoint path")
    ap.add_argument("--teacher", default=None, help="Teacher batch JSON; defaults to validate/teacher_batches/<clip>_teacher.json")
    ap.add_argument("--config", default=None, help="Optional config path; recorded into the output metadata only")
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--pretrain-template", default=DEFAULT_PRETRAIN_TEMPLATE)
    ap.add_argument("--encoder-bundle", default=None)
    ap.add_argument("--npz-root", default=DEFAULT_NPZ_ROOT)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--scales", default=DEFAULT_SCALES)
    ap.add_argument("--rounds", type=int, default=5, help="How many cycles to tile for the freerun rollout")
    ap.add_argument("--contact-source", default="auto", choices=("auto", "meas", "plan", "teacher"))
    ap.add_argument("--touchdown-threshold", type=float, default=0.5)
    ap.add_argument("--cycle-len", type=int, default=87, help="Target normalized cycle length")
    ap.add_argument("--export-series", action="store_true")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def _parse_scales(spec: str) -> List[float]:
    vals: List[float] = []
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("No valid scales were provided.")
    return vals


def _default_teacher_path(clip: str) -> Path:
    return Path("validate") / "teacher_batches" / f"{clip}_teacher.json"


def _default_out_path(clip: str, model_path: Path) -> Path:
    stem = model_path.stem
    return Path("debug_output") / "gait_speed_scaling_whitebox" / f"{clip}_{stem}_whitebox.json"


def _resolve_encoder_bundle_path(spec: Optional[str]) -> Optional[Path]:
    if spec:
        path = Path(spec).expanduser().resolve()
        return path if path.is_file() else None
    for candidate in DEFAULT_ENCODER_CANDIDATES:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
    return None


def _build_runner_args(args: argparse.Namespace) -> argparse.Namespace:
    encoder_bundle = _resolve_encoder_bundle_path(args.encoder_bundle)
    return SimpleNamespace(
        model=str(Path(args.model).expanduser().resolve()),
        bundle=str(Path(args.bundle).expanduser().resolve()),
        pretrain_template=str(Path(args.pretrain_template).expanduser().resolve()),
        encoder_bundle=str(encoder_bundle) if encoder_bundle is not None else None,
        device=str(args.device),
        num_heads=4,
        dropout=0.1,
        context_len=16,
        depth=2,
    )


def _clone_sample(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _scale_sample(
    sample: Dict[str, torch.Tensor],
    trainer: Any,
    *,
    scale: float,
) -> Dict[str, torch.Tensor]:
    scaled = _clone_sample(sample)
    cond_raw = scaled.get("cond_tgt_raw")
    if cond_raw is None:
        raise ValueError("Sample does not contain cond_tgt_raw; cannot apply speed scaling.")
    cond_raw = cond_raw.clone()
    cond_raw[..., -1] = cond_raw[..., -1] * float(scale)
    scaled["cond_tgt_raw"] = cond_raw

    cond_mu = scaled.get("cond_norm_mu")
    cond_std = scaled.get("cond_norm_std")
    cond_norm = trainer._normalize_cond_from_raw(cond_raw, cond_mu, cond_std)
    scaled["cond_in"] = cond_norm if cond_norm is not None else cond_raw.clone()
    scaled["cond_tgt"] = scaled["cond_in"].clone()

    rootvel_sl = getattr(trainer, "rootvel_x_slice", None)
    if isinstance(rootvel_sl, slice):
        motion = scaled.get("motion")
        if motion is not None:
            motion = motion.clone()
            motion[..., rootvel_sl] = motion[..., rootvel_sl] * float(scale)
            scaled["motion"] = motion
    return scaled


def _tile_sequence(t: Optional[torch.Tensor], rounds: int, device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    x = t.unsqueeze(0).to(device)
    if x.dim() == 3:
        return x.repeat(1, int(rounds), 1)
    return x


def _to_cpu_np(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if t is None:
        return None
    return t.detach().cpu().numpy()


def _infer_root_speed_and_pos(
    trainer: Any,
    *,
    state_seq: torch.Tensor,
    pred_y_raw: torch.Tensor,
    cond_raw_seq: Optional[torch.Tensor],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    motion = state_seq[:, 0]
    motion_raw = trainer.normalizer.denorm_x(motion)
    motion_raw_series: List[torch.Tensor] = [motion_raw[0].detach().cpu()]
    pred_steps = int(pred_y_raw.shape[0])
    for step_idx in range(pred_steps):
        cond_raw_step = None
        if cond_raw_seq is not None:
            idx = min(int(cond_raw_seq.shape[1]) - 1, int(step_idx) + 1)
            cond_raw_step = cond_raw_seq[:, idx]
        motion_raw = trainer._apply_free_carry(
            motion_raw,
            pred_y_raw[step_idx : step_idx + 1],
            cond_next_raw=cond_raw_step,
        ).detach()
        motion_raw_series.append(motion_raw[0].detach().cpu())

    motion_raw_stack = torch.stack(motion_raw_series, dim=0)
    rootvel_sl = getattr(trainer, "rootvel_x_slice", None)
    rootpos_sl = getattr(trainer, "rootpos_x_slice", None)

    if isinstance(rootvel_sl, slice):
        root_vel = motion_raw_stack[:, rootvel_sl].numpy()
    else:
        root_vel = np.zeros((motion_raw_stack.shape[0], 2), dtype=np.float32)

    if isinstance(rootpos_sl, slice):
        root_pos = motion_raw_stack[:, rootpos_sl].numpy()
    else:
        dt = 1.0 / max(float(getattr(trainer, "bone_hz", 60.0) or 60.0), 1e-6)
        planar_vel = root_vel[:, : min(2, root_vel.shape[1])]
        root_pos = np.concatenate(
            [
                np.zeros((1, planar_vel.shape[1]), dtype=np.float32),
                np.cumsum(planar_vel[1:] * dt, axis=0, dtype=np.float32),
            ],
            axis=0,
        )

    root_speed = np.linalg.norm(root_vel[:, : min(2, root_vel.shape[1])], axis=-1)
    return motion_raw_stack, root_speed, root_pos


def _choose_contact_series(
    *,
    requested: str,
    contacts_meas: Optional[np.ndarray],
    contacts_plan: Optional[np.ndarray],
    contacts_teacher: Optional[np.ndarray],
    threshold: float,
) -> tuple[str, Optional[np.ndarray]]:
    candidates = {
        "meas": contacts_meas,
        "plan": contacts_plan,
        "teacher": contacts_teacher,
    }
    if requested != "auto":
        return requested, candidates.get(requested)

    best_key = "none"
    best_arr: Optional[np.ndarray] = None
    best_score = -1
    for key in ("meas", "plan", "teacher"):
        arr = candidates.get(key)
        if arr is None:
            continue
        if arr.size <= 0:
            continue
        if not np.isfinite(arr).any():
            continue
        if float(np.abs(arr).max()) <= 1e-6:
            continue
        score = 0
        if arr.ndim == 2:
            for ch in range(int(arr.shape[1])):
                score = max(score, int(_rising_edges(arr[:, ch], threshold=threshold).size))
        if score > best_score:
            best_key = key
            best_arr = arr
            best_score = score
    return best_key, best_arr


def _rising_edges(signal: np.ndarray, threshold: float) -> np.ndarray:
    if signal.ndim != 1 or signal.size <= 0:
        return np.zeros((0,), dtype=np.int64)
    active = signal > float(threshold)
    rises = np.nonzero((~active[:-1]) & active[1:])[0] + 1
    return rises.astype(np.int64, copy=False)


def _detect_touchdowns(
    contact_signal: Optional[np.ndarray],
    *,
    threshold: float,
    rounds: int,
) -> tuple[int, List[int], bool]:
    if contact_signal is None or contact_signal.ndim != 2 or contact_signal.shape[1] <= 0:
        return 0, [], True

    best_channel = 0
    best_events: List[int] = []
    channel_events: List[List[int]] = []
    for ch in range(int(contact_signal.shape[1])):
        events = _rising_edges(contact_signal[:, ch], threshold=threshold).tolist()
        channel_events.append(events)
        if len(events) > len(best_events):
            best_channel = ch
            best_events = events

    min_events = max(2, int(rounds) - 1)
    td_unstable = len(best_events) < min_events
    if len(channel_events) >= 2:
        counts = [len(v) for v in channel_events]
        if max(counts) - min(counts) > 1:
            td_unstable = True
    return best_channel, best_events, td_unstable


def _fallback_cycle_boundaries(total_steps: int, rounds: int) -> tuple[List[int], List[int]]:
    if rounds <= 0 or total_steps <= 1:
        return [], []
    cycle = max(1, total_steps // rounds)
    starts = [int(i * cycle) for i in range(rounds)]
    stops = [int(min(total_steps, (i + 1) * cycle)) for i in range(rounds)]
    keep = [(s, e) for s, e in zip(starts, stops) if e - s >= 2]
    return [s for s, _ in keep], [e for _, e in keep]


def _build_cycle_boundaries(
    touchdown_indices: Sequence[int],
    *,
    total_steps: int,
    rounds: int,
) -> tuple[List[int], List[int]]:
    events = [int(x) for x in touchdown_indices if 0 <= int(x) < total_steps]
    if len(events) >= 2:
        starts = events[:-1]
        stops = events[1:]
        keep = [(s, e) for s, e in zip(starts, stops) if e - s >= 2]
        if keep:
            return [s for s, _ in keep], [e for _, e in keep]
    return _fallback_cycle_boundaries(total_steps=total_steps, rounds=rounds)


def _resample_flat_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    if seq.ndim < 2:
        raise ValueError(f"Expected seq with shape [T, ...], got {seq.shape}")
    src_len = int(seq.shape[0])
    if src_len == target_len:
        return seq.copy()
    if src_len <= 1:
        return np.repeat(seq[:1], target_len, axis=0)
    src_x = np.linspace(0.0, 1.0, src_len, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    flat = seq.reshape(src_len, -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(dst_x, src_x, flat[:, col])
    return out.reshape((target_len,) + seq.shape[1:])


def _build_cycle_template(
    pred_y_raw: torch.Tensor,
    cycle_starts: Sequence[int],
    cycle_stops: Sequence[int],
    cycle_len: int,
) -> tuple[Optional[np.ndarray], int]:
    seq = pred_y_raw.detach().cpu().numpy().astype(np.float32, copy=False)
    cycles: List[np.ndarray] = []
    for start, stop in zip(cycle_starts, cycle_stops):
        s = int(start)
        e = int(stop)
        if e - s < 2:
            continue
        cycles.append(_resample_flat_sequence(seq[s:e], cycle_len))
    if not cycles:
        return None, 0
    stacked = np.stack(cycles, axis=0)
    return stacked.mean(axis=0), int(stacked.shape[0])


def _rot_local_group_metrics(
    seq_a_raw: torch.Tensor,
    seq_b_raw: torch.Tensor,
    *,
    trainer: Any,
) -> Dict[str, float]:
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        raise RuntimeError("Trainer is missing rot6d_y_slice / rot6d_slice.")

    if seq_a_raw.dim() != 2 or seq_b_raw.dim() != 2:
        raise ValueError("Expected raw pose sequences with shape [T, Dy].")
    steps = min(int(seq_a_raw.shape[0]), int(seq_b_raw.shape[0]))
    seq_a_raw = seq_a_raw[:steps]
    seq_b_raw = seq_b_raw[:steps]

    rot_dim = int(rot_slice.stop - rot_slice.start)
    joint_count = rot_dim // 6
    if rot_dim <= 0 or rot_dim % 6 != 0 or joint_count <= 0:
        raise RuntimeError(f"Invalid rot6d slice width: {rot_dim}")

    pred_6d = reproject_rot6d(seq_a_raw[:, rot_slice].reshape(steps, joint_count, 6))
    ref_6d = reproject_rot6d(seq_b_raw[:, rot_slice].reshape(steps, joint_count, 6))
    pred_m = rot6d_to_matrix(pred_6d)
    ref_m = rot6d_to_matrix(ref_6d)

    root_idx = int(getattr(trainer.loss_fn, "root_idx", getattr(trainer, "root_idx", 0)) or 0)
    pred_rel = root_relative_matrices(pred_m, root_idx=root_idx)
    ref_rel = root_relative_matrices(ref_m, root_idx=root_idx)
    geo_deg = geodesic_R(pred_rel, ref_rel) * (180.0 / math.pi)

    masks = trainer.loss_fn._resolve_direct_group_masks(joint_count=joint_count, device=geo_deg.device)
    if masks is None:
        all_mean = float(geo_deg.mean().item())
        return {
            "all_deg": all_mean,
            "leg_deg": all_mean,
            "nonleg_deg": all_mean,
        }

    def _masked(mask_name: str) -> float:
        mask = masks.get(mask_name)
        if mask is None or int(mask.numel()) != joint_count or not bool(mask.any().item()):
            return float("nan")
        return float(geo_deg[..., mask].mean().item())

    all_deg = _masked("all_ex_root")
    if not math.isfinite(all_deg):
        all_deg = float(geo_deg.mean().item())
    leg_deg = _masked("leg")
    nonleg_deg = _masked("nonleg")
    if not math.isfinite(leg_deg):
        leg_deg = all_deg
    if not math.isfinite(nonleg_deg):
        nonleg_deg = all_deg
    return {
        "all_deg": all_deg,
        "leg_deg": leg_deg,
        "nonleg_deg": nonleg_deg,
    }


def _run_scaled_trace(
    runner: FreeRunCycleRunner,
    *,
    sample: Dict[str, torch.Tensor],
    rounds: int,
    scale: float,
    contact_source: str,
    touchdown_threshold: float,
) -> RolloutTrace:
    trainer = runner.trainer
    assert trainer is not None

    device = runner.device
    state_seq = _tile_sequence(sample.get("motion"), rounds=rounds, device=device)
    gt_seq = _tile_sequence(sample.get("gt_motion"), rounds=rounds, device=device)
    cond_seq = _tile_sequence(sample.get("cond_in"), rounds=rounds, device=device)
    cond_raw_seq = _tile_sequence(sample.get("cond_tgt_raw"), rounds=rounds, device=device)
    contacts_seq = _tile_sequence(sample.get("contacts"), rounds=rounds, device=device)
    angvel_seq = _tile_sequence(sample.get("angvel"), rounds=rounds, device=device)
    pose_hist_seq = _tile_sequence(sample.get("pose_hist"), rounds=rounds, device=device)
    cond_norm_mu = sample.get("cond_norm_mu")
    cond_norm_std = sample.get("cond_norm_std")
    if cond_norm_mu is not None:
        cond_norm_mu = cond_norm_mu.to(device)
    if cond_norm_std is not None:
        cond_norm_std = cond_norm_std.to(device)
    time_base = sample.get("start", None)
    if torch.is_tensor(time_base):
        time_base = time_base.to(device)

    with torch.no_grad():
        preds, _ = trainer._rollout_sequence(
            state_seq,
            cond_seq,
            cond_raw_seq,
            contacts_seq=contacts_seq,
            angvel_seq=angvel_seq,
            pose_hist_seq=pose_hist_seq,
            gt_seq=gt_seq,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
            mode="mixed",
            tf_ratio=0.0,
            time_base=time_base,
        )

    pred_y = preds.get("out")
    if not torch.is_tensor(pred_y):
        raise RuntimeError("Rollout did not return preds['out'].")
    pred_y = pred_y[0].detach()
    pred_y_raw = trainer._denorm(pred_y).detach()

    if gt_seq is None:
        raise RuntimeError("Expected tiled gt_seq for evaluator diagnostics.")
    gt_steps = min(int(gt_seq.shape[1]), int(pred_y.shape[0]))
    gt_y_raw = trainer._denorm(gt_seq[:, :gt_steps]).detach()[0]
    pred_y_raw = pred_y_raw[:gt_steps]

    motion_raw_stack, root_speed, root_pos = _infer_root_speed_and_pos(
        trainer,
        state_seq=state_seq,
        pred_y_raw=pred_y_raw,
        cond_raw_seq=cond_raw_seq,
    )

    contacts_plan = _to_cpu_np(preds.get("contacts_plan"))
    contacts_meas = _to_cpu_np(preds.get("contacts_meas"))
    contacts_teacher = _to_cpu_np(contacts_seq)
    if contacts_plan is not None:
        contacts_plan = contacts_plan[0, :gt_steps]
    if contacts_meas is not None:
        contacts_meas = contacts_meas[0, :gt_steps]
    if contacts_teacher is not None:
        contacts_teacher = contacts_teacher[0, :gt_steps]

    contact_source_used, contact_signal = _choose_contact_series(
        requested=contact_source,
        contacts_meas=contacts_meas,
        contacts_plan=contacts_plan,
        contacts_teacher=contacts_teacher,
        threshold=touchdown_threshold,
    )
    touchdown_channel, touchdown_indices, td_unstable = _detect_touchdowns(
        contact_signal,
        threshold=touchdown_threshold,
        rounds=rounds,
    )
    cycle_starts, cycle_stops = _build_cycle_boundaries(
        touchdown_indices,
        total_steps=int(pred_y_raw.shape[0]),
        rounds=rounds,
    )
    if not cycle_starts:
        td_unstable = True

    return RolloutTrace(
        scale=float(scale),
        pred_y_raw=pred_y_raw.detach().cpu(),
        gt_y_raw=gt_y_raw.detach().cpu(),
        motion_raw=motion_raw_stack.detach().cpu(),
        root_speed=root_speed.astype(np.float32, copy=False),
        root_pos=root_pos.astype(np.float32, copy=False),
        contacts_plan=contacts_plan,
        contacts_meas=contacts_meas,
        contacts_teacher=contacts_teacher,
        contact_source_used=contact_source_used,
        touchdown_channel=int(touchdown_channel),
        touchdown_indices=[int(x) for x in touchdown_indices],
        cycle_start_indices=[int(x) for x in cycle_starts],
        cycle_stop_indices=[int(x) for x in cycle_stops],
        td_unstable=bool(td_unstable),
    )


def _compute_cycle_speed_metrics(
    trace: RolloutTrace,
    *,
    fps: float,
) -> Dict[str, Any]:
    eps = 1e-8
    root_speed = np.asarray(trace.root_speed, dtype=np.float64)
    root_pos = np.asarray(trace.root_pos, dtype=np.float64)
    v_pred = float(np.mean(root_speed[1:])) if root_speed.size > 1 else float(np.mean(root_speed))

    periods: List[float] = []
    strides: List[float] = []
    for start, stop in zip(trace.cycle_start_indices, trace.cycle_stop_indices):
        s = int(start)
        e = int(stop)
        if e <= s or (e + 1) >= root_pos.shape[0]:
            continue
        period = float(e - s) / max(float(fps), eps)
        stride = float(np.linalg.norm(root_pos[e + 1, : min(2, root_pos.shape[1])] - root_pos[s + 1, : min(2, root_pos.shape[1])]))
        if period <= 0.0:
            continue
        periods.append(period)
        strides.append(stride)

    if not periods or not strides:
        return {
            "v_pred": v_pred,
            "cycle_count": 0,
            "period_s": float("nan"),
            "period_std_s": float("nan"),
            "freq_hz": float("nan"),
            "freq_std_hz": float("nan"),
            "stride_length": float("nan"),
            "stride_std_length": float("nan"),
            "E_cycle_speed_consistency": float("nan"),
            "cycle_speed_outlier_ratio": float("nan"),
        }

    periods_arr = np.asarray(periods, dtype=np.float64)
    strides_arr = np.asarray(strides, dtype=np.float64)
    v_cycle = strides_arr / np.maximum(periods_arr, eps)
    rel = np.abs(v_cycle - v_pred) / max(v_pred, eps)
    outlier_ratio = float(np.mean(rel > 0.10))

    freq_arr = 1.0 / np.maximum(periods_arr, eps)
    return {
        "v_pred": v_pred,
        "cycle_count": int(len(periods)),
        "period_s": float(periods_arr.mean()),
        "period_std_s": float(periods_arr.std(ddof=0)),
        "freq_hz": float(freq_arr.mean()),
        "freq_std_hz": float(freq_arr.std(ddof=0)),
        "stride_length": float(strides_arr.mean()),
        "stride_std_length": float(strides_arr.std(ddof=0)),
        "E_cycle_speed_consistency": float(rel.mean()),
        "cycle_speed_outlier_ratio": outlier_ratio,
    }


def _heuristic_status(
    entry: Dict[str, Any],
) -> str:
    if bool(entry.get("td_unstable", False)):
        return "fail"
    e_cycle = entry.get("E_cycle_speed_consistency")
    r_nonleg = entry.get("R_nonleg")
    e_cycle_leg = entry.get("E_cycle_leg")
    if any(
        isinstance(v, (int, float)) and math.isfinite(float(v))
        for v in (e_cycle, r_nonleg, e_cycle_leg)
    ):
        if (
            (isinstance(e_cycle, (int, float)) and math.isfinite(float(e_cycle)) and float(e_cycle) > 0.30)
            or (isinstance(r_nonleg, (int, float)) and math.isfinite(float(r_nonleg)) and float(r_nonleg) > 1.35)
            or (isinstance(e_cycle_leg, (int, float)) and math.isfinite(float(e_cycle_leg)) and float(e_cycle_leg) > 18.0)
        ):
            return "fail"
        if (
            (isinstance(e_cycle, (int, float)) and math.isfinite(float(e_cycle)) and float(e_cycle) > 0.15)
            or (isinstance(r_nonleg, (int, float)) and math.isfinite(float(r_nonleg)) and float(r_nonleg) > 1.15)
            or (isinstance(e_cycle_leg, (int, float)) and math.isfinite(float(e_cycle_leg)) and float(e_cycle_leg) > 10.0)
        ):
            return "warn"
    return "pass"


def _attach_monotonic_flags(per_scale: Dict[str, Dict[str, Any]]) -> None:
    ordered = sorted((float(k), v) for k, v in per_scale.items())
    for _, entry in ordered:
        entry["freq_monotonic_ok"] = True
        entry["stride_monotonic_ok"] = True

    for idx in range(len(ordered) - 1):
        _, prev_entry = ordered[idx]
        _, next_entry = ordered[idx + 1]

        prev_f = prev_entry.get("freq_hz")
        prev_f_std = prev_entry.get("freq_std_hz")
        next_f = next_entry.get("freq_hz")
        next_f_std = next_entry.get("freq_std_hz")
        if all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in (prev_f, prev_f_std, next_f, next_f_std)):
            if float(next_f) + float(next_f_std) < float(prev_f) - float(prev_f_std):
                prev_entry["freq_monotonic_ok"] = False
                next_entry["freq_monotonic_ok"] = False

        prev_l = prev_entry.get("stride_length")
        prev_l_std = prev_entry.get("stride_std_length")
        next_l = next_entry.get("stride_length")
        next_l_std = next_entry.get("stride_std_length")
        if all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in (prev_l, prev_l_std, next_l, next_l_std)):
            if float(next_l) + float(next_l_std) < float(prev_l) - float(prev_l_std):
                prev_entry["stride_monotonic_ok"] = False
                next_entry["stride_monotonic_ok"] = False

    for _, entry in ordered:
        if not bool(entry.get("freq_monotonic_ok", True)) or not bool(entry.get("stride_monotonic_ok", True)):
            if entry.get("status") == "pass":
                entry["status"] = "warn"


def _series_payload(trace: RolloutTrace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contact_source_used": trace.contact_source_used,
        "touchdown_channel": int(trace.touchdown_channel),
        "touchdown_indices": [int(x) for x in trace.touchdown_indices],
        "cycle_start_indices": [int(x) for x in trace.cycle_start_indices],
        "cycle_stop_indices": [int(x) for x in trace.cycle_stop_indices],
        "root_speed": [float(x) for x in np.asarray(trace.root_speed).tolist()],
    }
    if trace.contacts_meas is not None:
        payload["contacts_meas"] = np.asarray(trace.contacts_meas).tolist()
    if trace.contacts_plan is not None:
        payload["contacts_plan"] = np.asarray(trace.contacts_plan).tolist()
    if trace.contacts_teacher is not None:
        payload["contacts_teacher"] = np.asarray(trace.contacts_teacher).tolist()
    return payload


def main() -> None:
    args = _parse_args()
    scales = _parse_scales(args.scales)
    if 1.0 not in scales:
        scales.append(1.0)
        scales = sorted(scales)

    teacher_path = Path(args.teacher).expanduser() if args.teacher else _default_teacher_path(args.clip)
    teacher_path = teacher_path.resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Teacher batch not found: {teacher_path}")

    model_path = Path(args.model).expanduser().resolve()
    out_path = Path(args.out).expanduser() if args.out else _default_out_path(args.clip, model_path)
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runner = FreeRunCycleRunner(_build_runner_args(args))

    teacher_payload = _load_json(teacher_path)
    clip_name = str(teacher_payload.get("clip") or args.clip)
    teacher_block = teacher_payload.get("teacher")
    if not isinstance(teacher_block, dict):
        raise ValueError(f"{teacher_path}: missing teacher block.")
    state_arr = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
    if state_arr.ndim != 2:
        raise ValueError(f"{teacher_path}: invalid teacher state_norm shape.")
    cycle_len_teacher = int(state_arr.shape[0])

    npz_path = _resolve_npz_path(clip_name, teacher_payload.get("source_json"), Path(args.npz_root).expanduser().resolve())
    ds = runner._build_dataset(npz_path, seq_len=cycle_len_teacher)
    runner._ensure_model_ready(ds)
    trainer = runner.trainer
    if trainer is None:
        raise RuntimeError("Failed to initialize trainer for white-box evaluator.")

    clip = ds.clips[0]
    base_sample = _build_full_cycle_sample(ds, clip, seq_len=cycle_len_teacher)

    traces: Dict[float, RolloutTrace] = {}
    for scale in scales:
        scaled_sample = _scale_sample(base_sample, trainer, scale=float(scale))
        traces[float(scale)] = _run_scaled_trace(
            runner,
            sample=scaled_sample,
            rounds=int(args.rounds),
            scale=float(scale),
            contact_source=str(args.contact_source),
            touchdown_threshold=float(args.touchdown_threshold),
        )

    ref_trace = traces[1.0]
    fps = float(teacher_payload.get("fps", getattr(ds, "fps", 60.0)) or 60.0)
    v_ref = float(np.mean(ref_trace.root_speed[1:])) if ref_trace.root_speed.size > 1 else float(np.mean(ref_trace.root_speed))

    ref_aligned = _rot_local_group_metrics(ref_trace.pred_y_raw, ref_trace.gt_y_raw, trainer=trainer)
    ref_template_np, ref_cycle_count = _build_cycle_template(
        ref_trace.pred_y_raw,
        ref_trace.cycle_start_indices,
        ref_trace.cycle_stop_indices,
        int(args.cycle_len),
    )
    if ref_template_np is None:
        raise RuntimeError("Failed to build a 1.0x reference cycle template.")
    ref_template = torch.from_numpy(ref_template_np).to(torch.float32)

    per_scale: Dict[str, Dict[str, Any]] = {}
    optional_series: Dict[str, Any] = {}
    for scale in sorted(traces.keys()):
        trace = traces[scale]
        scale_key = f"{scale:.3f}".rstrip("0").rstrip(".")

        cycle_stats = _compute_cycle_speed_metrics(trace, fps=fps)
        v_pred = float(cycle_stats["v_pred"])
        v_tgt = float(scale) * v_ref
        e_speed = abs(v_pred - v_tgt) / max(v_tgt, 1e-8)

        aligned = _rot_local_group_metrics(trace.pred_y_raw, trace.gt_y_raw, trainer=trainer)
        template_np, template_cycles = _build_cycle_template(
            trace.pred_y_raw,
            trace.cycle_start_indices,
            trace.cycle_stop_indices,
            int(args.cycle_len),
        )
        if template_np is not None:
            template = torch.from_numpy(template_np).to(torch.float32)
            cycle_consistency = _rot_local_group_metrics(template, ref_template, trainer=trainer)
        else:
            cycle_consistency = {"all_deg": float("nan"), "leg_deg": float("nan"), "nonleg_deg": float("nan")}

        entry: Dict[str, Any] = {
            "scale": float(scale),
            "v_pred": float(v_pred),
            "v_tgt": float(v_tgt),
            "E_speed": float(e_speed),
            "touchdown_source": trace.contact_source_used,
            "touchdown_channel": int(trace.touchdown_channel),
            "touchdown_count": int(len(trace.touchdown_indices)),
            "cycle_count": int(cycle_stats["cycle_count"]),
            "template_cycle_count": int(template_cycles),
            "td_unstable": bool(trace.td_unstable),
            "E_cycle_speed_consistency": cycle_stats["E_cycle_speed_consistency"],
            "cycle_speed_outlier_ratio": cycle_stats["cycle_speed_outlier_ratio"],
            "period_s": cycle_stats["period_s"],
            "period_std_s": cycle_stats["period_std_s"],
            "freq_hz": cycle_stats["freq_hz"],
            "freq_std_hz": cycle_stats["freq_std_hz"],
            "stride_length": cycle_stats["stride_length"],
            "stride_std_length": cycle_stats["stride_std_length"],
            "leg_metric_deg": aligned["leg_deg"],
            "nonleg_metric_deg": aligned["nonleg_deg"],
            "R_leg": float(aligned["leg_deg"] / max(ref_aligned["leg_deg"], 1e-8)),
            "R_nonleg": float(aligned["nonleg_deg"] / max(ref_aligned["nonleg_deg"], 1e-8)),
            "E_cycle_all": cycle_consistency["all_deg"],
            "E_cycle_leg": cycle_consistency["leg_deg"],
            "E_cycle_nonleg": cycle_consistency["nonleg_deg"],
        }
        entry["status"] = _heuristic_status(entry)
        per_scale[scale_key] = entry
        if bool(args.export_series):
            optional_series[scale_key] = _series_payload(trace)

    _attach_monotonic_flags(per_scale)

    payload: Dict[str, Any] = {
        "summary": {
            "clip": clip_name,
            "mode": "d_only",
            "model": str(model_path),
            "teacher": str(teacher_path),
            "config": str(Path(args.config).expanduser().resolve()) if args.config else None,
            "bundle": str(Path(args.bundle).expanduser().resolve()),
            "pretrain_template": str(Path(args.pretrain_template).expanduser().resolve()),
            "encoder_bundle": getattr(runner.args, "encoder_bundle", None),
            "npz_path": str(npz_path),
            "fps": fps,
            "teacher_cycle_len": int(cycle_len_teacher),
            "normalized_cycle_len": int(args.cycle_len),
            "rounds": int(args.rounds),
            "scales": [float(x) for x in sorted(traces.keys())],
            "contact_source": str(args.contact_source),
            "touchdown_threshold": float(args.touchdown_threshold),
            "reference_scale": 1.0,
            "reference_speed": float(v_ref),
            "reference_leg_metric_deg": float(ref_aligned["leg_deg"]),
            "reference_nonleg_metric_deg": float(ref_aligned["nonleg_deg"]),
            "reference_template_cycle_count": int(ref_cycle_count),
            "status_policy": "heuristic_v0",
            "notes": [
                "E_speed uses carried root velocity after applying scaled cond_raw speed.",
                "R_leg/R_nonleg are aligned-to-original-GT degradation proxies, not scaled-GT errors.",
                "Cycle consistency compares normalized predicted cycles against the 1.0x predicted template.",
                "Touchdown events are detected from runtime contacts_meas/contacts_plan when available, then fall back to teacher contacts.",
            ],
        },
        "per_scale": per_scale,
        "optional_series": optional_series,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Whitebox] wrote {out_path}")


if __name__ == "__main__":
    main()
