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
from train.validate.run_freerun_cycles import FreeRunCycleRunner  # noqa: E402


DATE_TAG = "20260607"
DEFAULT_STEM = "phase_conditioned_gap_support_steering"
CALIBRATION_PATH = ROOT / "debug_output" / "20260606_gap_selection_goal_contract_calibration.py"
ROLLOUT_HELPER_PATH = ROOT / "debug_output" / "20260606_cond_contactplan_support_steering_probe.py"
WINDOW_HELPER_PATH = ROOT / "tools" / "run_action_handoff_freerun_contact_stability_probe.py"

PLANTED_LABELS = ("right", "left")
ARRIVAL_LABELS = ("right", "left", "true_flight")


@dataclass(frozen=True)
class EntryCase:
    entry_id: str
    independent_unit: str
    clip_name: str
    region_key: str
    phase_bin: str
    start_label: str
    cycle_bin8: int
    run_start: int
    run_end: int
    run_len: int
    base_frame: int
    seed_frames: tuple[int, ...]


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


def _safe_mean(values: Sequence[float]) -> float:
    vals = [_finite_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_min(values: Sequence[float]) -> float:
    vals = [_finite_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.min(vals)) if vals else float("nan")


def _safe_max(values: Sequence[float]) -> float:
    vals = [_finite_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.max(vals)) if vals else float("nan")


def _side_from_foot_name(name: str) -> Optional[str]:
    text = str(name).lower()
    if text.endswith("_r") or text.endswith(".r") or text.endswith(" r") or text in ("right", "r") or "_right" in text:
        return "right"
    if text.endswith("_l") or text.endswith(".l") or text.endswith(" l") or text in ("left", "l") or "_left" in text:
        return "left"
    return None


def _to_right_left(values: Any, foot_names: Sequence[str]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
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
        if np.isnan(out[:, 0]).all():
            out[:, 0] = arr[:, 0]
        if np.isnan(out[:, 1]).all():
            out[:, 1] = arr[:, 1]
    return out


def _foot_names_from_trainer(trainer: Any) -> tuple[list[str], list[int], bool]:
    idxs = getattr(trainer, "_contact_meas_foot_idxs", None)
    foot_indices = [int(x) for x in idxs] if isinstance(idxs, (list, tuple)) else []
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
    fallback = not any(_side_from_foot_name(name) in ("right", "left") for name in foot_names)
    return foot_names, foot_indices, bool(fallback)


def _pair_or_nan(arr: np.ndarray, idx: int) -> tuple[float, float]:
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 2:
        return float("nan"), float("nan")
    i = max(0, min(int(idx), arr.shape[0] - 1))
    return float(arr[i, 0]), float(arr[i, 1])


def _arrival_from_sequence(
    seq: Mapping[str, Any],
    *,
    foot_names: Sequence[str],
    margin_min: float,
    true_flight_dist_thr: float,
    skate_vxy_thr: float,
) -> dict[str, Any]:
    contacts = _to_right_left(seq.get("contacts", []), foot_names)
    comps = seq.get("contact_score_components", {}) if isinstance(seq.get("contact_score_components", {}), Mapping) else {}
    dist = _to_right_left(comps.get("dist_score", []), foot_names)
    vz = _to_right_left(comps.get("vz_score", []), foot_names)
    vxy = _to_right_left(comps.get("vxy_score", []), foot_names)
    n = int(max(contacts.shape[0], dist.shape[0], vz.shape[0], vxy.shape[0]))
    idx = max(0, n - 1)
    cr, cl = _pair_or_nan(contacts, idx)
    dr, dl = _pair_or_nan(dist, idx)
    vr, vl = _pair_or_nan(vz, idx)
    xr, xl = _pair_or_nan(vxy, idx)
    contact_pair = np.asarray([cr, cl], dtype=np.float64)
    dist_pair = np.asarray([dr, dl], dtype=np.float64)
    vxy_pair = np.asarray([xr, xl], dtype=np.float64)
    finite_contact = np.isfinite(contact_pair)
    finite_dist = np.isfinite(dist_pair)
    finite_vxy = np.isfinite(vxy_pair)
    contact_arg = int(np.nanargmax(np.where(finite_contact, contact_pair, -np.inf))) if finite_contact.any() else 0
    dist_arg = int(np.nanargmax(np.where(finite_dist, dist_pair, -np.inf))) if finite_dist.any() else contact_arg
    winner = PLANTED_LABELS[contact_arg]
    dist_winner = PLANTED_LABELS[dist_arg]
    contact_margin = abs(cr - cl) if np.isfinite(contact_pair).all() else float("nan")
    dist_margin = abs(dr - dl) if np.isfinite(dist_pair).all() else float("nan")
    max_dist = float(np.nanmax(dist_pair)) if finite_dist.any() else float("nan")
    max_contact = float(np.nanmax(contact_pair)) if finite_contact.any() else float("nan")
    max_vxy = float(np.nanmax(vxy_pair)) if finite_vxy.any() else float("nan")
    true_flight = bool(math.isfinite(max_dist) and max_dist < float(true_flight_dist_thr))
    skate_like = bool((not true_flight) and math.isfinite(max_dist) and math.isfinite(max_vxy) and max_vxy < float(skate_vxy_thr))
    boundary = bool((not true_flight) and ((not math.isfinite(contact_margin)) or contact_margin < float(margin_min)))
    support_label = "true_flight" if true_flight else winner
    internal = bool((not true_flight) and (not boundary) and support_label in PLANTED_LABELS)

    def _frac_true(values: np.ndarray, predicate: Any) -> float:
        if values.ndim != 2 or values.shape[0] <= 0:
            return float("nan")
        flags = [bool(predicate(row)) for row in values]
        return float(np.mean(flags)) if flags else float("nan")

    true_flight_frac = _frac_true(
        dist,
        lambda row: np.isfinite(row).any() and float(np.nanmax(row)) < float(true_flight_dist_thr),
    )
    skate_frac = float("nan")
    if dist.ndim == 2 and vxy.ndim == 2 and dist.shape[0] > 0 and vxy.shape[0] > 0:
        m = min(dist.shape[0], vxy.shape[0])
        flags = []
        for i in range(m):
            drow = dist[i]
            xrow = vxy[i]
            if not np.isfinite(drow).any() or not np.isfinite(xrow).any():
                continue
            dmax = float(np.nanmax(drow))
            xmax = float(np.nanmax(xrow))
            flags.append(bool(dmax >= float(true_flight_dist_thr) and xmax < float(skate_vxy_thr)))
        skate_frac = float(np.mean(flags)) if flags else float("nan")

    return {
        "arrival_label": support_label,
        "arrival_winner_contact": winner,
        "arrival_winner_dist": dist_winner,
        "arrival_internal": int(internal),
        "arrival_boundary": int(boundary),
        "arrival_true_flight": int(true_flight),
        "arrival_skate_like": int(skate_like),
        "arrival_contact_right": cr,
        "arrival_contact_left": cl,
        "arrival_dist_right": dr,
        "arrival_dist_left": dl,
        "arrival_vz_right": vr,
        "arrival_vz_left": vl,
        "arrival_vxy_right": xr,
        "arrival_vxy_left": xl,
        "arrival_contact_margin_abs": float(contact_margin),
        "arrival_dist_margin_abs": float(dist_margin),
        "arrival_contact_max": float(max_contact),
        "arrival_dist_max": float(max_dist),
        "arrival_vxy_max": float(max_vxy),
        "seq_true_flight_frame_frac": true_flight_frac,
        "seq_skate_frame_frac": skate_frac,
        "seq_contact_margin_min": _safe_min([abs(float(r[0] - r[1])) for r in contacts if np.isfinite(r).all()]),
        "seq_contact_margin_mean": _safe_mean([abs(float(r[0] - r[1])) for r in contacts if np.isfinite(r).all()]),
        "seq_dist_max_min": _safe_min([float(np.nanmax(r)) for r in dist if np.isfinite(r).any()]),
        "seq_vxy_max_mean": _safe_mean([float(np.nanmax(r)) for r in vxy if np.isfinite(r).any()]),
    }


def _select_seed_frames(frames: Sequence[int], *, seeds_per_entry: int) -> tuple[int, ...]:
    vals = sorted({int(x) for x in frames})
    if len(vals) <= int(seeds_per_entry):
        return tuple(vals)
    mid_i = len(vals) // 2
    offsets = [0, -2, 2, -4, 4, -6, 6, -8, 8, -10, 10]
    picked: list[int] = []
    for off in offsets:
        idx = max(0, min(len(vals) - 1, mid_i + int(off)))
        frame = vals[idx]
        if frame not in picked:
            picked.append(frame)
        if len(picked) >= int(seeds_per_entry):
            break
    if len(picked) < int(seeds_per_entry):
        for idx in np.linspace(0, len(vals) - 1, int(seeds_per_entry), dtype=int).tolist():
            frame = vals[int(idx)]
            if frame not in picked:
                picked.append(frame)
            if len(picked) >= int(seeds_per_entry):
                break
    return tuple(sorted(picked[: int(seeds_per_entry)]))


def _build_entries(
    *,
    calib: Any,
    labels_by_clip: Mapping[str, Sequence[str]],
    entry_count: int,
    seeds_per_entry: int,
    delivery_max_gap: int,
    context_len: int,
    isolation_frames: int,
    include_start_labels: Sequence[str],
) -> list[EntryCase]:
    label_allow = {str(x) for x in include_start_labels if str(x)}
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    meta_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for clip_name in sorted(labels_by_clip.keys()):
        labels = list(labels_by_clip[clip_name])
        if not labels:
            continue
        last = max(int(context_len), int(len(labels)) - int(delivery_max_gap) - 1)
        for frame in range(int(context_len), max(int(context_len), last) + 1):
            ph = calib._phase_at(labels, frame, isolation_frames=int(isolation_frames))
            start_label = str(ph["label"])
            if label_allow and start_label not in label_allow:
                continue
            region_key = f"{clip_name}:{ph['region_key_suffix']}"
            phase_bin = str(ph["phase_bin"])
            key = (str(clip_name), region_key, phase_bin)
            grouped[key].append(int(frame))
            meta_by_key[key] = {
                "start_label": start_label,
                "cycle_bin8": int(ph["cycle_bin8"]),
                "run_start": int(ph["run_start"]),
                "run_end": int(ph["run_end"]),
                "run_len": int(ph["run_len"]),
            }

    candidates: list[EntryCase] = []
    for key, frames in grouped.items():
        if len(set(frames)) < int(seeds_per_entry):
            continue
        clip_name, region_key, phase_bin = key
        seeds = _select_seed_frames(frames, seeds_per_entry=int(seeds_per_entry))
        if len(seeds) < int(seeds_per_entry):
            continue
        base = int(seeds[len(seeds) // 2])
        meta = meta_by_key[key]
        independent_unit = f"{clip_name}:{region_key}:{phase_bin}"
        entry_id = f"{independent_unit}:base{base}"
        candidates.append(
            EntryCase(
                entry_id=entry_id,
                independent_unit=independent_unit,
                clip_name=clip_name,
                region_key=region_key,
                phase_bin=phase_bin,
                start_label=str(meta["start_label"]),
                cycle_bin8=int(meta["cycle_bin8"]),
                run_start=int(meta["run_start"]),
                run_end=int(meta["run_end"]),
                run_len=int(meta["run_len"]),
                base_frame=base,
                seed_frames=tuple(int(x) for x in seeds),
            )
        )

    by_clip: dict[str, list[EntryCase]] = defaultdict(list)
    for c in sorted(candidates, key=lambda c: (c.clip_name != WALK_F, c.clip_name, c.region_key, c.phase_bin, c.base_frame)):
        by_clip[c.clip_name].append(c)
    clip_order = sorted(by_clip.keys(), key=lambda x: (x != WALK_F, x))
    selected: list[EntryCase] = []
    round_idx = 0
    while len(selected) < int(entry_count):
        added = False
        for clip_name in clip_order:
            items = by_clip.get(clip_name, [])
            if round_idx < len(items):
                selected.append(items[round_idx])
                added = True
                if len(selected) >= int(entry_count):
                    break
        if not added:
            break
        round_idx += 1
    return selected


def _collect_rows(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, int(args.torch_threads)))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    calib = _load_module(CALIBRATION_PATH, "phase_conditioned_gap_calibration")
    rollout_helper = _load_module(ROLLOUT_HELPER_PATH, "phase_conditioned_gap_rollout_helper")
    window_helper = _load_module(WINDOW_HELPER_PATH, "phase_conditioned_gap_window_helper")

    args.checkpoint = Path(args.checkpoint or calib.CKPT)
    args.bundle = Path(args.bundle or calib.BUNDLE)
    args.pretrain_template = Path(args.pretrain_template or calib.PRETRAIN_TEMPLATE)
    args.encoder_bundle = Path(args.encoder_bundle or calib.ENCODER_BUNDLE)
    args.npz_root = Path(args.npz_root or calib.NPZ_ROOT)
    args.z_features = Path(args.z_features or calib.Z_FEATURES)

    state281 = load_clip_states(args.z_features, args.npz_root)
    labels_by_clip = {name: calib._labels_from_contacts(arr[:, CONTACT_SLICE]) for name, arr in state281.items()}
    entries = _build_entries(
        calib=calib,
        labels_by_clip=labels_by_clip,
        entry_count=int(args.entries),
        seeds_per_entry=int(args.seeds_per_entry),
        delivery_max_gap=int(args.delivery_max_gap),
        context_len=int(args.context_len),
        isolation_frames=int(args.isolation_frames),
        include_start_labels=[x.strip() for x in str(args.include_start_labels).split(",")],
    )
    gaps = list(range(int(args.min_gap), int(args.max_gap) + 1, max(1, int(args.gap_step))))
    runner = FreeRunCycleRunner(calib._runner_args(args))
    clip_names = sorted(labels_by_clip.keys(), key=lambda x: (x != WALK_F, x))
    ds_by_clip: dict[str, Any] = {}
    clip_by_name: dict[str, Any] = {}
    for clip_name in clip_names:
        ds = runner._build_dataset(args.npz_root / f"{clip_name}.npz", seq_len=max(2, int(args.max_gap) + 1))
        runner._ensure_model_ready(ds)
        ds_by_clip[clip_name] = ds
        clip_by_name[clip_name] = ds.clips[0]
    if runner.trainer is None or runner.model is None:
        raise RuntimeError("runner did not initialize trainer/model")
    trainer = runner.trainer
    model = runner.model
    device = runner.device
    setattr(trainer, "contact_meas_vxy_mode", str(args.contact_vxy_mode))
    cfg = _rollout_kernel.resolve_free_carry_runtime_config(trainer)
    arms = ["free", "teacher"]
    rows: list[dict[str, Any]] = []
    total_jobs = len(entries) * max(1, int(args.seeds_per_entry)) * len(gaps) * len(arms)
    job_idx = 0
    t0 = time.time()
    for entry in entries:
        ds = ds_by_clip[entry.clip_name]
        clip = clip_by_name[entry.clip_name]
        clip_len = int(clip.X.shape[0])
        for seed_idx, seed_frame in enumerate(entry.seed_frames):
            for gap in gaps:
                total = int(gap) + 1
                sample = window_helper._build_wrapped_window_sample(ds, clip, int(seed_frame), total)
                wrapped_window = int(int(seed_frame) + total > clip_len)
                for arm in arms:
                    job_idx += 1
                    seq = rollout_helper._run_sequence(
                        trainer=trainer,
                        model=model,
                        sample=sample,
                        device=device,
                        mode=arm,
                        apply_lambda=True,
                    )
                    foot_names, foot_indices, side_fallback = _foot_names_from_trainer(trainer)
                    arrival = _arrival_from_sequence(
                        seq,
                        foot_names=foot_names,
                        margin_min=float(args.margin_min),
                        true_flight_dist_thr=float(args.true_flight_dist_thr),
                        skate_vxy_thr=float(args.skate_vxy_thr),
                    )
                    cond_meta = seq.get("tensor_meta", {}).get("cond", {}) if isinstance(seq.get("tensor_meta", {}), Mapping) else {}
                    rows.append(
                        {
                            "row_id": int(len(rows)),
                            "valid": 1,
                            "arm": arm,
                            "entry_id": entry.entry_id,
                            "independent_unit": entry.independent_unit,
                            "clip_name": entry.clip_name,
                            "region_key": entry.region_key,
                            "phase_bin": entry.phase_bin,
                            "cycle_bin8": int(entry.cycle_bin8),
                            "start_label": entry.start_label,
                            "run_start": int(entry.run_start),
                            "run_end": int(entry.run_end),
                            "run_len": int(entry.run_len),
                            "base_frame": int(entry.base_frame),
                            "seed_idx": int(seed_idx),
                            "seed_frame": int(seed_frame),
                            "gap": int(gap),
                            "gap_bin": f"g{int(gap):03d}",
                            "delivery_band": int(int(args.delivery_min_gap) <= int(gap) <= int(args.delivery_max_gap)),
                            "stress_audit": int(not (int(args.delivery_min_gap) <= int(gap) <= int(args.delivery_max_gap))),
                            "wrapped_window": int(wrapped_window),
                            "contact_vxy_mode": str(args.contact_vxy_mode),
                            "foot_names_json": _json_dumps_compact(foot_names),
                            "foot_indices_json": _json_dumps_compact(foot_indices),
                            "side_fallback_used": int(bool(side_fallback)),
                            **arrival,
                            "cond_shape_json": _json_dumps_compact(cond_meta.get("shape", [])),
                            "cond_dtype": str(cond_meta.get("dtype", "")),
                            "cond_device": str(cond_meta.get("device", "")),
                            "cond_finite": int(bool(cond_meta.get("finite", False))),
                            "state_raw_shape_json": _json_dumps_compact(list(np.asarray(seq.get("state_raw", [])).shape)),
                            "state_raw_dtype": "float64",
                            "state_raw_device": "cpu",
                            "read_only_forward": 1,
                            "no_generation_model_training": 1,
                            "no_weight_write": 1,
                            "target_excludes_contacts_plan": 1,
                            "target_excludes_CONTACT_SLICE": 1,
                            "rot6d_columns_json": _json_dumps_compact(list(getattr(cfg, "columns", ("X", "Z")))),
                        }
                    )
                    if job_idx % max(1, int(args.progress_every)) == 0:
                        elapsed = time.time() - t0
                        print(f"[collect] {job_idx}/{total_jobs} rows={len(rows)} elapsed={elapsed:.1f}s", flush=True)

    meta = {
        "task": "action_handoff_phase_conditioned_gap_support_steering",
        "date": DATE_TAG,
        "out_dir": str(out_dir),
        "checkpoint": str(args.checkpoint),
        "bundle": str(args.bundle),
        "pretrain_template": str(args.pretrain_template),
        "encoder_bundle": str(args.encoder_bundle),
        "npz_root": str(args.npz_root),
        "z_features": str(args.z_features),
        "read_only_forward": True,
        "no_generation_model_training": True,
        "no_weight_write": True,
        "production_modules_modified": False,
        "target": "Layer-2 FK soft support from realized rollout pose via existing compute_contact_meas_whitebox; no contacts_plan/CONTACT_SLICE target",
        "CONTACT_SLICE_scope": "used only to build entry-phase sampling inventory, not as target",
        "entries_requested": int(args.entries),
        "entries_collected": int(len(entries)),
        "seeds_per_entry_requested": int(args.seeds_per_entry),
        "gaps": [int(g) for g in gaps],
        "delivery_gap_range": [int(args.delivery_min_gap), int(args.delivery_max_gap)],
        "stress_gap_range": [int(args.min_gap), int(args.max_gap), int(args.gap_step)],
        "arms": arms,
        "entry_cases": [_entry_to_json(c) for c in entries],
        "thresholds": {
            "margin_min": float(args.margin_min),
            "true_flight_dist_thr": float(args.true_flight_dist_thr),
            "skate_vxy_thr": float(args.skate_vxy_thr),
            "min_plateau_len": int(args.min_plateau_len),
            "seed_agree_frac": float(args.seed_agree_frac),
        },
        "tensor_contract": {
            "cond": "[1,H,7] float32 rollout-device; serialized per row as shape/dtype/device/finite",
            "state_raw": "[H-1,Dx] float64 cpu materialized from read-only rollout; Dx follows checkpoint layout",
            "soft_scores": "[H-1,2] float64 cpu right,left after foot-name reordering",
        },
    }
    return rows, meta


def _entry_to_json(entry: EntryCase) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "independent_unit": entry.independent_unit,
        "clip_name": entry.clip_name,
        "region_key": entry.region_key,
        "phase_bin": entry.phase_bin,
        "start_label": entry.start_label,
        "cycle_bin8": int(entry.cycle_bin8),
        "base_frame": int(entry.base_frame),
        "seed_frames": [int(x) for x in entry.seed_frames],
        "run_start": int(entry.run_start),
        "run_end": int(entry.run_end),
        "run_len": int(entry.run_len),
    }


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("entry_id")), _finite_int(row.get("seed_idx")), _finite_int(row.get("gap")))


def _stable_label(rows: Sequence[Mapping[str, Any]], *, seed_agree_frac: float) -> tuple[str, float]:
    labels = [
        str(r.get("free_label", r.get("arrival_label")))
        for r in rows
        if int(_finite_int(r.get("reliable"), 0)) == 1
        if str(r.get("free_label", r.get("arrival_label", ""))) in PLANTED_LABELS
    ]
    if not labels:
        return "", 0.0
    counts = Counter(labels)
    label, count = counts.most_common(1)[0]
    frac = float(count) / float(max(1, len(rows)))
    if frac >= float(seed_agree_frac):
        return str(label), frac
    return "", frac


def _contiguous_ranges(gaps: Sequence[int]) -> list[dict[str, int]]:
    vals = sorted({int(g) for g in gaps})
    if not vals:
        return []
    ranges: list[dict[str, int]] = []
    start = prev = vals[0]
    for val in vals[1:]:
        if int(val) == int(prev) + 1:
            prev = int(val)
            continue
        ranges.append({"start": int(start), "end": int(prev), "length": int(prev - start + 1)})
        start = prev = int(val)
    ranges.append({"start": int(start), "end": int(prev), "length": int(prev - start + 1)})
    return ranges


def _entry_lookup_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    delivery_min_gap: int,
    delivery_max_gap: int,
    min_plateau_len: int,
    seed_agree_frac: float,
) -> dict[str, Any]:
    by_pair: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if _finite_int(row.get("valid"), 0) != 1:
            continue
        by_pair[_row_key(row)][str(row.get("arm"))] = row

    by_entry_gap: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (entry_id, seed_idx, gap), arms in sorted(by_pair.items()):
        free = arms.get("free")
        teacher = arms.get("teacher")
        if free is None or teacher is None:
            continue
        free_label = str(free.get("arrival_label"))
        teacher_label = str(teacher.get("arrival_label"))
        free_internal = int(_finite_int(free.get("arrival_internal"), 0)) == 1
        teacher_internal = int(_finite_int(teacher.get("arrival_internal"), 0)) == 1
        same = bool(free_label == teacher_label)
        reliable = bool(same and free_internal and teacher_internal and free_label in PLANTED_LABELS)
        by_entry_gap[(entry_id, int(gap))].append(
            {
                "entry_id": entry_id,
                "seed_idx": int(seed_idx),
                "seed_frame": _finite_int(free.get("seed_frame")),
                "gap": int(gap),
                "free_label": free_label,
                "teacher_label": teacher_label,
                "same": int(same),
                "free_internal": int(free_internal),
                "teacher_internal": int(teacher_internal),
                "reliable": int(reliable),
                "free_margin": _finite_float(free.get("arrival_contact_margin_abs")),
                "teacher_margin": _finite_float(teacher.get("arrival_contact_margin_abs")),
                "free_true_flight": _finite_int(free.get("arrival_true_flight"), 0),
                "teacher_true_flight": _finite_int(teacher.get("arrival_true_flight"), 0),
                "free_skate_like": _finite_int(free.get("arrival_skate_like"), 0),
                "teacher_skate_like": _finite_int(teacher.get("arrival_skate_like"), 0),
                "wrapped_window": _finite_int(free.get("wrapped_window"), 0),
            }
        )

    entry_meta: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        eid = str(row.get("entry_id"))
        if eid and eid not in entry_meta:
            entry_meta[eid] = row

    entries: dict[str, Any] = {}
    for (entry_id, gap), seed_rows in sorted(by_entry_gap.items()):
        meta = entry_meta.get(entry_id, {})
        expected_seeds = len({int(_finite_int(r.get("seed_idx"))) for r in rows if str(r.get("entry_id")) == entry_id})
        reliable_rows = [r for r in seed_rows if int(r["reliable"]) == 1]
        label, agree_frac = _stable_label(seed_rows, seed_agree_frac=float(seed_agree_frac))
        seed_complete = int(len(seed_rows) >= max(1, expected_seeds))
        stable = bool(seed_complete and label)
        delivery = int(int(delivery_min_gap) <= int(gap) <= int(delivery_max_gap))
        gap_rec = {
            "gap": int(gap),
            "delivery_band": delivery,
            "stress_audit": int(not delivery),
            "seed_complete": int(seed_complete),
            "seed_n": int(len(seed_rows)),
            "reliable_seed_n": int(len(reliable_rows)),
            "seed_agree_frac": float(agree_frac),
            "stable_internal": int(stable),
            "stable_label": label,
            "free_labels": [str(r["free_label"]) for r in seed_rows],
            "teacher_labels": [str(r["teacher_label"]) for r in seed_rows],
            "fr_teacher_agree_rate": float(np.mean([bool(r["same"]) for r in seed_rows])) if seed_rows else float("nan"),
            "free_margin_min": _safe_min([_finite_float(r["free_margin"]) for r in seed_rows]),
            "teacher_margin_min": _safe_min([_finite_float(r["teacher_margin"]) for r in seed_rows]),
            "true_flight_seed_n": int(sum(int(r["free_true_flight"]) for r in seed_rows)),
            "skate_like_seed_n": int(sum(int(r["free_skate_like"]) for r in seed_rows)),
            "wrapped_seed_n": int(sum(int(r["wrapped_window"]) for r in seed_rows)),
            "seed_rows": seed_rows,
        }
        entry = entries.setdefault(
            entry_id,
            {
                "entry_id": entry_id,
                "independent_unit": str(meta.get("independent_unit")),
                "clip_name": str(meta.get("clip_name")),
                "region_key": str(meta.get("region_key")),
                "phase_bin": str(meta.get("phase_bin")),
                "start_label": str(meta.get("start_label")),
                "base_frame": _finite_int(meta.get("base_frame")),
                "seed_frames": sorted({_finite_int(r.get("seed_frame")) for r in rows if str(r.get("entry_id")) == entry_id}),
                "gaps": [],
            },
        )
        entry["gaps"].append(gap_rec)

    for entry in entries.values():
        stable_delivery_by_label: dict[str, list[int]] = defaultdict(list)
        stable_stress_by_label: dict[str, list[int]] = defaultdict(list)
        for rec in entry["gaps"]:
            if int(rec["stable_internal"]) != 1:
                continue
            label = str(rec["stable_label"])
            if label not in PLANTED_LABELS:
                continue
            if int(rec["delivery_band"]) == 1:
                stable_delivery_by_label[label].append(int(rec["gap"]))
            else:
                stable_stress_by_label[label].append(int(rec["gap"]))
        plateaus: dict[str, Any] = {}
        for label in PLANTED_LABELS:
            delivery_ranges = [r for r in _contiguous_ranges(stable_delivery_by_label[label]) if int(r["length"]) >= int(min_plateau_len)]
            stress_ranges = [r for r in _contiguous_ranges(stable_stress_by_label[label]) if int(r["length"]) >= int(min_plateau_len)]
            plateaus[label] = {
                "delivery": delivery_ranges,
                "stress": stress_ranges,
                "delivery_gap_n": int(len(stable_delivery_by_label[label])),
                "stress_gap_n": int(len(stable_stress_by_label[label])),
            }
        start_label = str(entry.get("start_label"))
        reachable_targets = [
            label
            for label in PLANTED_LABELS
            if label != start_label and any(p["delivery"] for p in [plateaus[label]])
        ]
        stable_delivery_labels = sorted(
            {
                str(rec["stable_label"])
                for rec in entry["gaps"]
                if int(rec["delivery_band"]) == 1 and int(rec["stable_internal"]) == 1 and str(rec["stable_label"]) in PLANTED_LABELS
            }
        )
        stable_all_labels = sorted(
            {
                str(rec["stable_label"])
                for rec in entry["gaps"]
                if int(rec["stable_internal"]) == 1 and str(rec["stable_label"]) in PLANTED_LABELS
            }
        )
        invalid_reasons: list[str] = []
        if len(entry.get("seed_frames", [])) < 2:
            invalid_reasons.append("seed_count_lt_2")
        if not stable_delivery_labels:
            invalid_reasons.append("no_stable_delivery_planted_plateau")
        if len(stable_all_labels) < 2:
            invalid_reasons.append("lookup_degenerate_lt_2_stable_planted_labels_all_gaps")
        entry["plateaus"] = plateaus
        entry["stable_delivery_labels"] = stable_delivery_labels
        entry["stable_all_labels"] = stable_all_labels
        entry["lookup_nondegenerate_all_gaps"] = int(len(stable_all_labels) >= 2)
        entry["delivery_reachable_targets_outside_start"] = reachable_targets
        entry["entry_pass"] = int(bool(reachable_targets))
        entry["entry_invalid"] = int(bool(invalid_reasons))
        entry["invalid_reasons"] = invalid_reasons
    return {"entries": entries}


def _recalc_from_rows(
    rows_csv: Path,
    *,
    delivery_min_gap: int,
    delivery_max_gap: int,
    min_plateau_len: int,
    seed_agree_frac: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = _read_csv(rows_csv)
    lookup = _entry_lookup_from_rows(
        rows,
        delivery_min_gap=int(delivery_min_gap),
        delivery_max_gap=int(delivery_max_gap),
        min_plateau_len=int(min_plateau_len),
        seed_agree_frac=float(seed_agree_frac),
    )
    entries = lookup["entries"]
    entry_values = list(entries.values())
    effective_entry_n = len({str(e.get("independent_unit")) for e in entry_values})
    pass_entries = [e for e in entry_values if int(e.get("entry_pass", 0)) == 1]
    invalid_entries = [e for e in entry_values if int(e.get("entry_invalid", 0)) == 1]
    valid_for_fail = [e for e in entry_values if int(e.get("entry_invalid", 0)) == 0]
    if effective_entry_n >= 7:
        pass_threshold = 5
    elif effective_entry_n >= 6:
        pass_threshold = 4
    elif effective_entry_n >= 5:
        pass_threshold = 4
    else:
        pass_threshold = max(1, int(math.ceil(0.67 * max(1, effective_entry_n))))

    invalid_reasons: list[str] = []
    if effective_entry_n < 5:
        invalid_reasons.append(f"effective_entry_n_lt_5:{effective_entry_n}")
    if len(valid_for_fail) < max(1, min(4, effective_entry_n)):
        invalid_reasons.append(f"valid_entry_n_insufficient:{len(valid_for_fail)}")
    delivery_rows = [
        r
        for r in rows
        if int(delivery_min_gap) <= _finite_int(r.get("gap"), -1) <= int(delivery_max_gap)
    ]
    if not delivery_rows:
        invalid_reasons.append("no_delivery_rows")
    side_fallback_rate = _safe_mean([_finite_float(r.get("side_fallback_used"), 0.0) for r in rows])
    if math.isfinite(side_fallback_rate) and side_fallback_rate > 0.0:
        invalid_reasons.append(f"foot_side_fallback_used_rate:{side_fallback_rate:.6f}")

    if invalid_reasons:
        verdict = "INVALID"
        downstream = "采样或足侧语义不足；重设 entry/seed 后重跑，不下 PASS/FAIL。"
    elif len(pass_entries) >= pass_threshold:
        verdict = "PASS"
        downstream = "组合 support steering 存在，但只限 phase-conditioned per-entry lookup 和盆内部 target；不推出任意命令 support。"
    else:
        verdict = "FAIL"
        downstream = "phase-conditioned + soft-margin + seed/teacher 过滤后仍不能稳定把 support 推出 entry 起点盆；当前 scope 下干净负结论成立。"

    reachability_rows: list[dict[str, Any]] = []
    for entry in entry_values:
        for label in PLANTED_LABELS:
            delivery_ranges = entry.get("plateaus", {}).get(label, {}).get("delivery", [])
            stress_ranges = entry.get("plateaus", {}).get(label, {}).get("stress", [])
            reachability_rows.append(
                {
                    "entry_id": entry.get("entry_id"),
                    "independent_unit": entry.get("independent_unit"),
                    "clip_name": entry.get("clip_name"),
                    "phase_bin": entry.get("phase_bin"),
                    "start_label": entry.get("start_label"),
                    "target_label": label,
                    "target_outside_start": int(label != str(entry.get("start_label"))),
                    "delivery_plateau_n": int(len(delivery_ranges)),
                    "stress_plateau_n": int(len(stress_ranges)),
                    "delivery_ranges": delivery_ranges,
                    "stress_ranges": stress_ranges,
                    "delivery_reachable": int(bool(delivery_ranges)),
                    "delivery_reachable_outside_start": int(bool(delivery_ranges) and label != str(entry.get("start_label"))),
                }
            )

    summary = {
        "task": "action_handoff_phase_conditioned_gap_support_steering",
        "scope": "delivery gaps 12..30 for PASS/FAIL; 12..84 rows retained as stress audit",
        "rows_csv": str(rows_csv),
        "row_n": int(len(rows)),
        "delivery_row_n": int(len(delivery_rows)),
        "effective_entry_n": int(effective_entry_n),
        "entry_n": int(len(entry_values)),
        "pass_entry_n": int(len(pass_entries)),
        "invalid_entry_n": int(len(invalid_entries)),
        "pass_threshold": int(pass_threshold),
        "entries_passed": [str(e.get("entry_id")) for e in pass_entries],
        "entries_invalid": {str(e.get("entry_id")): list(e.get("invalid_reasons", [])) for e in invalid_entries},
        "label_counts_free_delivery": dict(Counter(str(r.get("arrival_label")) for r in delivery_rows if str(r.get("arm")) == "free")),
        "label_counts_teacher_delivery": dict(Counter(str(r.get("arrival_label")) for r in delivery_rows if str(r.get("arm")) == "teacher")),
        "free_true_flight_row_rate_delivery": _safe_mean(
            [_finite_float(r.get("arrival_true_flight"), 0.0) for r in delivery_rows if str(r.get("arm")) == "free"]
        ),
        "free_skate_like_row_rate_delivery": _safe_mean(
            [_finite_float(r.get("arrival_skate_like"), 0.0) for r in delivery_rows if str(r.get("arm")) == "free"]
        ),
        "teacher_agreement_rate_delivery_seed_pairs": _teacher_agreement_rate(entries, delivery_only=True),
        "seed_stable_gap_rate_delivery": _seed_stable_gap_rate(entries, delivery_only=True),
        "reachability_rows": reachability_rows,
    }
    redteam = {
        "verdict": verdict,
        "downstream": downstream,
        "invalid_reasons": invalid_reasons,
        "decision_inputs": {
            "effective_entry_n": int(effective_entry_n),
            "entry_n": int(len(entry_values)),
            "pass_entry_n": int(len(pass_entries)),
            "pass_threshold": int(pass_threshold),
            "valid_entry_n": int(len(valid_for_fail)),
            "invalid_entry_n": int(len(invalid_entries)),
            "teacher_agreement_rate_delivery_seed_pairs": summary["teacher_agreement_rate_delivery_seed_pairs"],
            "seed_stable_gap_rate_delivery": summary["seed_stable_gap_rate_delivery"],
            "free_true_flight_row_rate_delivery": summary["free_true_flight_row_rate_delivery"],
            "free_skate_like_row_rate_delivery": summary["free_skate_like_row_rate_delivery"],
        },
        "negative_scope": "FAIL is limited to delivered checkpoint, read-only rollout, sampled entries, FK-soft Layer-2 metric, and delivery gaps 12..30.",
        "pass_scope": "PASS only supports phase-conditioned per-entry basin-internal gap steering among naturally reachable planted support basins.",
        "method_redlines": {
            "no_training": True,
            "no_weight_write": True,
            "production_trainer_gate_model_untouched": True,
            "target_not_used": ["contacts_plan", "CONTACT_SLICE"],
            "CONTACT_SLICE_scope": "entry inventory only",
            "FK_source": "train.validate.contact_meas_whitebox.compute_contact_meas_whitebox via existing rollout helper",
            "support_metric": "right/left soft contact margin plus dist/vxy true-flight vs skate decomposition; no 0.5 contact threshold or min-run rewrite",
        },
        "closeout_update_suggestion": _closeout_suggestion(verdict),
    }
    return lookup, summary, redteam


def _teacher_agreement_rate(entries: Mapping[str, Any], *, delivery_only: bool) -> float:
    vals: list[float] = []
    for entry in entries.values():
        for rec in entry.get("gaps", []):
            if delivery_only and int(rec.get("delivery_band", 0)) != 1:
                continue
            vals.append(_finite_float(rec.get("fr_teacher_agree_rate")))
    return _safe_mean(vals)


def _seed_stable_gap_rate(entries: Mapping[str, Any], *, delivery_only: bool) -> float:
    vals: list[float] = []
    for entry in entries.values():
        for rec in entry.get("gaps", []):
            if delivery_only and int(rec.get("delivery_band", 0)) != 1:
                continue
            vals.append(float(int(rec.get("stable_internal", 0)) == 1))
    return _safe_mean(vals)


def _closeout_suggestion(verdict: str) -> str:
    if verdict == "PASS":
        return (
            "把 closeout 从 blanket uncontrollable 拆成两句：cond/latent independent support handle 未发现；"
            "phase-conditioned per-entry gap lookup 在盆内部成立，但不是任意 support command。"
        )
    if verdict == "FAIL":
        return (
            "保留 read-only support uncontrollable，但把旧 gap-selection/flight 伪影证据降级为污染先验；"
            "用本 rows.csv 的 soft-margin + teacher/seed 复算作为干净负证据。"
        )
    return "不更新 PASS/FAIL closeout；先修采样或足侧/FK-soft margin 分辨率。"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only phase-conditioned gap->soft-support lookup probe.",
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
    parser.add_argument("--gap-step", type=int, default=1)
    parser.add_argument("--delivery-min-gap", type=int, default=12)
    parser.add_argument("--delivery-max-gap", type=int, default=30)
    parser.add_argument("--entries", type=int, default=6)
    parser.add_argument("--seeds-per-entry", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--isolation-frames", type=int, default=34)
    parser.add_argument("--include-start-labels", default="right,left")
    parser.add_argument("--contact-vxy-mode", default="abs")
    parser.add_argument("--margin-min", type=float, default=0.05)
    parser.add_argument("--true-flight-dist-thr", type=float, default=0.20)
    parser.add_argument("--skate-vxy-thr", type=float, default=0.20)
    parser.add_argument("--min-plateau-len", type=int, default=2)
    parser.add_argument("--seed-agree-frac", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--recalc-only", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path("debug_output") / f"_tmp_{DEFAULT_STEM}_{DATE_TAG}_{stamp}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / "rows.csv"
    if args.recalc_only is not None:
        rows_csv = Path(args.recalc_only)
    else:
        rows, meta = _collect_rows(args, out_dir)
        _write_csv(rows_csv, rows)
        _write_json(out_dir / "collection_meta.json", meta)
    lookup, summary, redteam = _recalc_from_rows(
        rows_csv,
        delivery_min_gap=int(args.delivery_min_gap),
        delivery_max_gap=int(args.delivery_max_gap),
        min_plateau_len=int(args.min_plateau_len),
        seed_agree_frac=float(args.seed_agree_frac),
    )
    _write_json(out_dir / "per_entry_lookup.json", lookup)
    _write_json(out_dir / "reachability_summary.json", summary)
    _write_json(out_dir / "redteam_recalc.json", redteam)
    print(f"wrote {rows_csv}")
    print(f"wrote {out_dir / 'per_entry_lookup.json'}")
    print(f"wrote {out_dir / 'reachability_summary.json'}")
    print(f"wrote {out_dir / 'redteam_recalc.json'}")
    print(json.dumps(_jsonable(redteam["decision_inputs"]), ensure_ascii=False, indent=2, allow_nan=False))
    print(f"VERDICT={redteam['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
