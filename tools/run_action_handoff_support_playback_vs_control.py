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
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_action_handoff_phase_conditioned_gap_support_steering as base  # noqa: E402
from train import rollout_kernel as _rollout_kernel  # noqa: E402
from train.data.action_handoff_inbetween import CONTACT_SLICE, WALK_F, load_clip_states  # noqa: E402
from train.validate.run_freerun_cycles import FreeRunCycleRunner  # noqa: E402


DATE_TAG = "20260607"
DEFAULT_STEM = "support_playback_vs_control"
CALIBRATION_PATH = ROOT / "debug_output" / "20260606_gap_selection_goal_contract_calibration.py"
ROLLOUT_HELPER_PATH = ROOT / "debug_output" / "20260606_cond_contactplan_support_steering_probe.py"
WINDOW_HELPER_PATH = ROOT / "tools" / "run_action_handoff_freerun_contact_stability_probe.py"
PHASE_SLOTS = ("q0", "q1", "q2", "q3")
PLANTED_LABELS = base.PLANTED_LABELS


@dataclass(frozen=True)
class EntryCase:
    entry_id: str
    independent_unit: str
    cell_key: str
    clip_name: str
    region_key: str
    phase_bin: str
    phase_slot: str
    start_label: str
    cycle_bin8: int
    run_start: int
    run_end: int
    run_len: int
    base_frame: int
    entry_phase_frac: float
    seed_frames: tuple[int, ...]


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_json(path, payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base._write_csv(path, rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    return base._finite_float(value, default)


def _finite_int(value: Any, default: int = 0) -> int:
    return base._finite_int(value, default)


def _safe_mean(values: Sequence[float]) -> float:
    vals = [_finite_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_min(values: Sequence[float]) -> float:
    vals = [_finite_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.min(vals)) if vals else float("nan")


def _json_dumps_compact(value: Any) -> str:
    return base._json_dumps_compact(value)


def _phase_slot(phase_bin: Any) -> str:
    text = str(phase_bin)
    if ":" not in text:
        return "unknown"
    return text.split(":", 1)[1]


def _phase_frac(ph: Mapping[str, Any], frame: int) -> float:
    run_len = max(1, _finite_int(ph.get("run_len"), 1))
    run_start = _finite_int(ph.get("run_start"), 0)
    frac = float(int(frame) - run_start) / float(run_len)
    return float(max(0.0, min(1.0, frac)))


def _rel_label(label: Any, start_label: Any) -> str:
    lab = str(label)
    start = str(start_label)
    if lab in PLANTED_LABELS and start in PLANTED_LABELS:
        return "same" if lab == start else "opposite"
    return lab


def _label_at(labels: Sequence[str], idx: int, *, wrap: bool) -> str:
    if not labels:
        return "unknown"
    if wrap:
        return str(labels[int(idx) % len(labels)])
    return str(labels[max(0, min(int(idx), len(labels) - 1))])


def _build_phase_matched_entries(
    *,
    calib: Any,
    labels_by_clip: Mapping[str, Sequence[str]],
    entries_per_cell: int,
    seeds_per_entry: int,
    entry_min_frame: int,
    isolation_frames: int,
    include_start_labels: Sequence[str],
) -> tuple[list[EntryCase], dict[str, Any]]:
    label_allow = {str(x) for x in include_start_labels if str(x)}
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    meta_by_unit: dict[tuple[str, str, str], dict[str, Any]] = {}
    for clip_name in sorted(labels_by_clip.keys()):
        labels = list(labels_by_clip[clip_name])
        for frame in range(max(0, int(entry_min_frame)), len(labels)):
            ph = calib._phase_at(labels, frame, isolation_frames=int(isolation_frames))
            start_label = str(ph["label"])
            phase_slot = _phase_slot(ph["phase_bin"])
            if start_label not in PLANTED_LABELS or phase_slot not in PHASE_SLOTS:
                continue
            if label_allow and start_label not in label_allow:
                continue
            region_key = f"{clip_name}:{ph['region_key_suffix']}"
            phase_bin = f"{start_label}:{phase_slot}"
            independent_unit = f"{region_key}:{start_label}:{phase_slot}"
            key = (start_label, phase_slot, independent_unit)
            grouped[key].append(int(frame))
            meta_by_unit[key] = {
                "clip_name": str(clip_name),
                "region_key": str(region_key),
                "phase_bin": str(phase_bin),
                "phase_slot": str(phase_slot),
                "start_label": str(start_label),
            }

    candidates_by_cell: dict[tuple[str, str], list[EntryCase]] = defaultdict(list)
    inventory: dict[str, Any] = {}
    for key, frames in grouped.items():
        start_label, phase_slot, independent_unit = key
        unique_frames = sorted({int(x) for x in frames})
        cell_key = f"{start_label}:{phase_slot}"
        if len(unique_frames) < int(seeds_per_entry):
            inventory.setdefault(cell_key, {"short_units": []})["short_units"].append(
                {
                    "independent_unit": independent_unit,
                    "frame_n": int(len(unique_frames)),
                    "frames": unique_frames,
                }
            )
            continue
        seeds = base._select_seed_frames(unique_frames, seeds_per_entry=int(seeds_per_entry))
        if len(seeds) < int(seeds_per_entry):
            continue
        base_frame = int(seeds[len(seeds) // 2])
        meta = meta_by_unit[key]
        labels = labels_by_clip[str(meta["clip_name"])]
        ph = calib._phase_at(labels, base_frame, isolation_frames=int(isolation_frames))
        entry = EntryCase(
            entry_id=f"{independent_unit}:base{base_frame}",
            independent_unit=independent_unit,
            cell_key=cell_key,
            clip_name=str(meta["clip_name"]),
            region_key=str(meta["region_key"]),
            phase_bin=str(meta["phase_bin"]),
            phase_slot=str(meta["phase_slot"]),
            start_label=str(meta["start_label"]),
            cycle_bin8=int(ph["cycle_bin8"]),
            run_start=int(ph["run_start"]),
            run_end=int(ph["run_end"]),
            run_len=int(ph["run_len"]),
            base_frame=base_frame,
            entry_phase_frac=_phase_frac(ph, base_frame),
            seed_frames=tuple(int(x) for x in seeds),
        )
        candidates_by_cell[(start_label, phase_slot)].append(entry)

    selected: list[EntryCase] = []
    for start_label in PLANTED_LABELS:
        for phase_slot in PHASE_SLOTS:
            cell = (start_label, phase_slot)
            cell_key = f"{start_label}:{phase_slot}"
            items = sorted(
                candidates_by_cell.get(cell, []),
                key=lambda e: (e.clip_name != WALK_F, -int(e.run_len), e.clip_name, e.region_key, e.base_frame),
            )
            picked = items[: max(0, int(entries_per_cell))]
            selected.extend(picked)
            rec = inventory.setdefault(cell_key, {})
            rec.update(
                {
                    "available_entry_n": int(len(items)),
                    "selected_entry_n": int(len(picked)),
                    "selected_independent_units": [e.independent_unit for e in picked],
                    "selected_entry_ids": [e.entry_id for e in picked],
                }
            )
    return selected, inventory


def _entry_to_json(entry: EntryCase) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "independent_unit": entry.independent_unit,
        "cell_key": entry.cell_key,
        "clip_name": entry.clip_name,
        "region_key": entry.region_key,
        "phase_bin": entry.phase_bin,
        "phase_slot": entry.phase_slot,
        "start_label": entry.start_label,
        "cycle_bin8": int(entry.cycle_bin8),
        "base_frame": int(entry.base_frame),
        "entry_phase_frac": float(entry.entry_phase_frac),
        "seed_frames": [int(x) for x in entry.seed_frames],
        "run_start": int(entry.run_start),
        "run_end": int(entry.run_end),
        "run_len": int(entry.run_len),
    }


def _collect_rows(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, int(args.torch_threads)))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    calib = _load_module(CALIBRATION_PATH, "support_playback_calibration")
    rollout_helper = _load_module(ROLLOUT_HELPER_PATH, "support_playback_rollout_helper")
    window_helper = _load_module(WINDOW_HELPER_PATH, "support_playback_window_helper")

    args.checkpoint = Path(args.checkpoint or calib.CKPT)
    args.bundle = Path(args.bundle or calib.BUNDLE)
    args.pretrain_template = Path(args.pretrain_template or calib.PRETRAIN_TEMPLATE)
    args.encoder_bundle = Path(args.encoder_bundle or calib.ENCODER_BUNDLE)
    args.npz_root = Path(args.npz_root or calib.NPZ_ROOT)
    args.z_features = Path(args.z_features or calib.Z_FEATURES)

    state281 = load_clip_states(args.z_features, args.npz_root)
    labels_by_clip = {name: calib._labels_from_contacts(arr[:, CONTACT_SLICE]) for name, arr in state281.items()}
    entries, inventory = _build_phase_matched_entries(
        calib=calib,
        labels_by_clip=labels_by_clip,
        entries_per_cell=int(args.entries_per_cell),
        seeds_per_entry=int(args.seeds_per_entry),
        entry_min_frame=int(args.entry_min_frame),
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
        labels = labels_by_clip[entry.clip_name]
        label_len = int(len(labels))
        for seed_idx, seed_frame in enumerate(entry.seed_frames):
            seed_ph = calib._phase_at(labels, int(seed_frame) % max(1, label_len), isolation_frames=int(args.isolation_frames))
            seed_phase_frac = _phase_frac(seed_ph, int(seed_frame) % max(1, label_len))
            for gap in gaps:
                total = int(gap) + 1
                sample = window_helper._build_wrapped_window_sample(ds, clip, int(seed_frame), total)
                wrapped_window = int(int(seed_frame) + total > clip_len)
                gt_idx = (int(seed_frame) + int(gap)) % max(1, label_len)
                gt_label = _label_at(labels, gt_idx, wrap=False)
                gt_ph = calib._phase_at(labels, gt_idx, isolation_frames=int(args.isolation_frames))
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
                    foot_names, foot_indices, side_fallback = base._foot_names_from_trainer(trainer)
                    arrival = base._arrival_from_sequence(
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
                            "cell_key": entry.cell_key,
                            "clip_name": entry.clip_name,
                            "region_key": entry.region_key,
                            "phase_bin": entry.phase_bin,
                            "phase_slot": entry.phase_slot,
                            "entry_phase_frac": float(entry.entry_phase_frac),
                            "seed_phase_frac": float(seed_phase_frac),
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
                            "gt_clock_idx": int(gt_idx),
                            "gt_clock_label": str(gt_label),
                            "gt_clock_phase_bin": str(gt_ph["phase_bin"]),
                            "gt_clock_phase_slot": _phase_slot(gt_ph["phase_bin"]),
                            "gt_clock_phase_frac": _phase_frac(gt_ph, gt_idx),
                            "gt_clock_planted": int(str(gt_label) in PLANTED_LABELS),
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
                            "CONTACT_SLICE_scope": "entry_inventory_and_gt_clip_clock_only",
                            "rot6d_columns_json": _json_dumps_compact(list(getattr(cfg, "columns", ("X", "Z")))),
                        }
                    )
                    if job_idx % max(1, int(args.progress_every)) == 0:
                        elapsed = time.time() - t0
                        print(f"[collect] {job_idx}/{total_jobs} rows={len(rows)} elapsed={elapsed:.1f}s", flush=True)

    meta = {
        "task": "action_handoff_support_playback_vs_control",
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
        "target": "Layer-2 FK soft support from realized rollout pose via existing compute_contact_meas_whitebox; no contacts_plan/CONTACT_SLICE control target",
        "clock_refs": {
            "teacher_primary": "teacher-forced model rollout S_teacher(entry,gap)",
            "gtclip_secondary": "GT source clip support label at (entry_frame+gap) modulo clip length",
        },
        "CONTACT_SLICE_scope": "entry phase inventory and secondary GT clip clock only; not used as control target",
        "entries_per_cell_requested": int(args.entries_per_cell),
        "entries_collected": int(len(entries)),
        "phase_cells": [f"{side}:{q}" for side in PLANTED_LABELS for q in PHASE_SLOTS],
        "phase_cell_inventory": inventory,
        "entry_min_frame": int(args.entry_min_frame),
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
            "seed_agree_frac": float(args.seed_agree_frac),
            "playback_agree_min": float(args.playback_agree_min),
            "symmetry_match_min": float(args.symmetry_match_min),
            "symmetry_delta_max": float(args.symmetry_delta_max),
            "determinism_min": float(args.determinism_min),
        },
        "tensor_contract": {
            "cond": "[1,H,7] float32 rollout-device; serialized per row as shape/dtype/device/finite",
            "state_raw": "[H-1,Dx] float64 cpu materialized from read-only rollout; Dx follows checkpoint layout",
            "soft_scores": "[H-1,2] float64 cpu right,left after foot-name reordering",
        },
    }
    return rows, meta


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("entry_id")), _finite_int(row.get("seed_idx")), _finite_int(row.get("gap")))


def _pair_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if _finite_int(row.get("valid"), 0) != 1:
            continue
        by_pair[_row_key(row)][str(row.get("arm"))] = row
    pairs: list[dict[str, Any]] = []
    for (entry_id, seed_idx, gap), arms in sorted(by_pair.items()):
        free = arms.get("free")
        teacher = arms.get("teacher")
        if free is None or teacher is None:
            continue
        start_label = str(free.get("start_label"))
        free_label = str(free.get("arrival_label"))
        teacher_label = str(teacher.get("arrival_label"))
        gt_label = str(free.get("gt_clock_label"))
        free_internal = int(_finite_int(free.get("arrival_internal"), 0)) == 1 and free_label in PLANTED_LABELS
        teacher_internal = int(_finite_int(teacher.get("arrival_internal"), 0)) == 1 and teacher_label in PLANTED_LABELS
        gt_planted = gt_label in PLANTED_LABELS
        pairs.append(
            {
                "entry_id": entry_id,
                "seed_idx": int(seed_idx),
                "seed_frame": _finite_int(free.get("seed_frame")),
                "gap": int(gap),
                "delivery_band": _finite_int(free.get("delivery_band"), 0),
                "wrapped_window": _finite_int(free.get("wrapped_window"), 0),
                "independent_unit": str(free.get("independent_unit")),
                "cell_key": str(free.get("cell_key")),
                "clip_name": str(free.get("clip_name")),
                "region_key": str(free.get("region_key")),
                "phase_bin": str(free.get("phase_bin")),
                "phase_slot": str(free.get("phase_slot")),
                "start_label": start_label,
                "entry_phase_frac": _finite_float(free.get("entry_phase_frac")),
                "seed_phase_frac": _finite_float(free.get("seed_phase_frac")),
                "free_label": free_label,
                "teacher_label": teacher_label,
                "gt_clock_label": gt_label,
                "gt_clock_phase_bin": str(free.get("gt_clock_phase_bin")),
                "free_rel_label": _rel_label(free_label, start_label),
                "teacher_rel_label": _rel_label(teacher_label, start_label),
                "gt_rel_label": _rel_label(gt_label, start_label),
                "free_teacher_same": int(free_label == teacher_label),
                "free_gt_same": int(gt_planted and free_label == gt_label),
                "teacher_gt_same": int(gt_planted and teacher_label == gt_label),
                "free_internal": int(free_internal),
                "teacher_internal": int(teacher_internal),
                "both_internal": int(free_internal and teacher_internal),
                "gt_planted": int(gt_planted),
                "free_boundary": _finite_int(free.get("arrival_boundary"), 0),
                "teacher_boundary": _finite_int(teacher.get("arrival_boundary"), 0),
                "free_true_flight": _finite_int(free.get("arrival_true_flight"), 0),
                "teacher_true_flight": _finite_int(teacher.get("arrival_true_flight"), 0),
                "free_skate_like": _finite_int(free.get("arrival_skate_like"), 0),
                "teacher_skate_like": _finite_int(teacher.get("arrival_skate_like"), 0),
                "free_margin": _finite_float(free.get("arrival_contact_margin_abs")),
                "teacher_margin": _finite_float(teacher.get("arrival_contact_margin_abs")),
                "side_fallback_used": _finite_int(free.get("side_fallback_used"), 0),
            }
        )
    return pairs


def _rate_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gt_pairs = [p for p in pairs if int(p.get("gt_planted", 0)) == 1]
    both_internal = [p for p in pairs if int(p.get("both_internal", 0)) == 1]
    return {
        "pair_n": int(len(pairs)),
        "independent_entry_n": int(len({str(p.get("independent_unit")) for p in pairs})),
        "free_teacher_agree_rate": _safe_mean([_finite_float(p.get("free_teacher_same"), 0.0) for p in pairs]),
        "free_teacher_agree_rate_both_internal": _safe_mean(
            [_finite_float(p.get("free_teacher_same"), 0.0) for p in both_internal]
        ),
        "drift_floor_disagree_rate": 1.0 - _safe_mean([_finite_float(p.get("free_teacher_same"), 0.0) for p in pairs])
        if pairs
        else float("nan"),
        "free_gt_agree_rate_planted_gt": _safe_mean([_finite_float(p.get("free_gt_same"), 0.0) for p in gt_pairs]),
        "teacher_gt_agree_rate_planted_gt": _safe_mean([_finite_float(p.get("teacher_gt_same"), 0.0) for p in gt_pairs]),
        "free_internal_rate": _safe_mean([_finite_float(p.get("free_internal"), 0.0) for p in pairs]),
        "teacher_internal_rate": _safe_mean([_finite_float(p.get("teacher_internal"), 0.0) for p in pairs]),
        "free_boundary_rate": _safe_mean([_finite_float(p.get("free_boundary"), 0.0) for p in pairs]),
        "teacher_boundary_rate": _safe_mean([_finite_float(p.get("teacher_boundary"), 0.0) for p in pairs]),
        "free_true_flight_rate": _safe_mean([_finite_float(p.get("free_true_flight"), 0.0) for p in pairs]),
        "free_skate_like_rate": _safe_mean([_finite_float(p.get("free_skate_like"), 0.0) for p in pairs]),
        "free_margin_min": _safe_min([_finite_float(p.get("free_margin")) for p in pairs]),
        "teacher_margin_min": _safe_min([_finite_float(p.get("teacher_margin")) for p in pairs]),
        "wrapped_pair_rate": _safe_mean([_finite_float(p.get("wrapped_window"), 0.0) for p in pairs]),
        "free_label_counts": dict(Counter(str(p.get("free_label")) for p in pairs)),
        "teacher_label_counts": dict(Counter(str(p.get("teacher_label")) for p in pairs)),
        "gt_clock_label_counts": dict(Counter(str(p.get("gt_clock_label")) for p in pairs)),
    }


def _stable_majority(labels: Sequence[str], *, expected_n: int, seed_agree_frac: float) -> tuple[str, float, int]:
    vals = [str(x) for x in labels if str(x) in PLANTED_LABELS]
    if not vals:
        return "", 0.0, 0
    label, count = Counter(vals).most_common(1)[0]
    denom = max(1, int(expected_n))
    frac = float(count) / float(denom)
    if frac >= float(seed_agree_frac):
        return str(label), frac, int(count)
    return "", frac, int(count)


def _entry_gap_records(pairs: Sequence[Mapping[str, Any]], *, seed_agree_frac: float) -> list[dict[str, Any]]:
    seeds_by_entry = {
        str(entry_id): len({int(_finite_int(p.get("seed_idx"))) for p in group})
        for entry_id, group in _group_list(pairs, lambda p: str(p.get("entry_id"))).items()
    }
    out: list[dict[str, Any]] = []
    for (entry_id, gap), group in _group_list(pairs, lambda p: (str(p.get("entry_id")), _finite_int(p.get("gap")))).items():
        expected_n = max(1, seeds_by_entry.get(str(entry_id), 1))
        first = group[0]
        free_label, free_frac, free_count = _stable_majority(
            [str(p.get("free_label")) for p in group if int(p.get("free_internal", 0)) == 1],
            expected_n=expected_n,
            seed_agree_frac=float(seed_agree_frac),
        )
        teacher_label, teacher_frac, teacher_count = _stable_majority(
            [str(p.get("teacher_label")) for p in group if int(p.get("teacher_internal", 0)) == 1],
            expected_n=expected_n,
            seed_agree_frac=float(seed_agree_frac),
        )
        gt_label, gt_frac, gt_count = _stable_majority(
            [str(p.get("gt_clock_label")) for p in group if int(p.get("gt_planted", 0)) == 1],
            expected_n=expected_n,
            seed_agree_frac=float(seed_agree_frac),
        )
        out.append(
            {
                "entry_id": str(entry_id),
                "gap": int(gap),
                "seed_pair_n": int(len(group)),
                "expected_seed_n": int(expected_n),
                "delivery_band": _finite_int(first.get("delivery_band"), 0),
                "independent_unit": str(first.get("independent_unit")),
                "cell_key": str(first.get("cell_key")),
                "clip_name": str(first.get("clip_name")),
                "region_key": str(first.get("region_key")),
                "phase_bin": str(first.get("phase_bin")),
                "phase_slot": str(first.get("phase_slot")),
                "start_label": str(first.get("start_label")),
                "entry_phase_frac": _finite_float(first.get("entry_phase_frac")),
                "free_stable_label": free_label,
                "teacher_stable_label": teacher_label,
                "gt_stable_label": gt_label,
                "free_rel_label": _rel_label(free_label, first.get("start_label")) if free_label else "",
                "teacher_rel_label": _rel_label(teacher_label, first.get("start_label")) if teacher_label else "",
                "gt_rel_label": _rel_label(gt_label, first.get("start_label")) if gt_label else "",
                "free_seed_agree_frac": float(free_frac),
                "teacher_seed_agree_frac": float(teacher_frac),
                "gt_seed_agree_frac": float(gt_frac),
                "free_seed_count": int(free_count),
                "teacher_seed_count": int(teacher_count),
                "gt_seed_count": int(gt_count),
                "free_teacher_stable_same": int(bool(free_label and teacher_label and free_label == teacher_label)),
                "free_gt_stable_same": int(bool(free_label and gt_label and free_label == gt_label)),
                "teacher_gt_stable_same": int(bool(teacher_label and gt_label and teacher_label == gt_label)),
            }
        )
    return out


def _group_list(values: Sequence[Any], key_fn: Any) -> dict[Any, list[Any]]:
    out: dict[Any, list[Any]] = defaultdict(list)
    for value in values:
        out[key_fn(value)].append(value)
    return dict(out)


def _majority_fraction(values: Sequence[str]) -> tuple[str, float, int]:
    vals = [str(v) for v in values if str(v)]
    if not vals:
        return "", float("nan"), 0
    label, count = Counter(vals).most_common(1)[0]
    return str(label), float(count) / float(len(vals)), int(len(vals))


def _determinism(records: Sequence[Mapping[str, Any]], label_key: str) -> dict[str, Any]:
    groups = _group_list(
        [r for r in records if str(r.get(label_key, ""))],
        lambda r: (str(r.get("start_label")), str(r.get("phase_slot")), _finite_int(r.get("gap"))),
    )
    rows: list[dict[str, Any]] = []
    for (side, q, gap), group in sorted(groups.items()):
        label, frac, n = _majority_fraction([str(r.get(label_key)) for r in group])
        rows.append({"start_label": side, "phase_slot": q, "gap": int(gap), "majority_label": label, "majority_frac": frac, "entry_n": n})
    return {
        "group_n": int(len(rows)),
        "mean_majority_frac": _safe_mean([_finite_float(r.get("majority_frac")) for r in rows]),
        "min_majority_frac": _safe_min([_finite_float(r.get("majority_frac")) for r in rows]),
        "groups": rows,
    }


def _side_symmetry(records: Sequence[Mapping[str, Any]], label_key: str) -> dict[str, Any]:
    groups = _group_list(
        [r for r in records if str(r.get(label_key, ""))],
        lambda r: (str(r.get("start_label")), str(r.get("phase_slot")), _finite_int(r.get("gap"))),
    )
    majority: dict[tuple[str, str, int], tuple[str, float, int]] = {}
    for key, group in groups.items():
        majority[key] = _majority_fraction([str(r.get(label_key)) for r in group])

    rows: list[dict[str, Any]] = []
    by_phase: dict[str, list[int]] = defaultdict(list)
    for q in PHASE_SLOTS:
        gaps = sorted(
            {
                int(k[2])
                for k in majority.keys()
                if str(k[1]) == q and str(k[0]) in PLANTED_LABELS
            }
        )
        for gap in gaps:
            left = majority.get(("left", q, int(gap)))
            right = majority.get(("right", q, int(gap)))
            if left is None or right is None:
                continue
            match = int(str(left[0]) == str(right[0]))
            by_phase[q].append(match)
            rows.append(
                {
                    "phase_slot": q,
                    "gap": int(gap),
                    "left_rel_majority": str(left[0]),
                    "right_rel_majority": str(right[0]),
                    "left_entry_n": int(left[2]),
                    "right_entry_n": int(right[2]),
                    "left_majority_frac": float(left[1]),
                    "right_majority_frac": float(right[1]),
                    "side_normalized_match": match,
                }
            )
    return {
        "paired_group_n": int(len(rows)),
        "side_normalized_match_rate": _safe_mean([_finite_float(r.get("side_normalized_match"), 0.0) for r in rows]),
        "by_phase_match_rate": {q: _safe_mean([float(x) for x in vals]) for q, vals in sorted(by_phase.items())},
        "pairs": rows,
    }


def _teacher_relative_side_symmetry(free_sym: Mapping[str, Any], teacher_sym: Mapping[str, Any]) -> dict[str, Any]:
    free_pairs = {
        (str(r.get("phase_slot")), _finite_int(r.get("gap"))): r
        for r in free_sym.get("pairs", [])
        if isinstance(r, Mapping)
    }
    teacher_pairs = {
        (str(r.get("phase_slot")), _finite_int(r.get("gap"))): r
        for r in teacher_sym.get("pairs", [])
        if isinstance(r, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(free_pairs.keys()) & set(teacher_pairs.keys())):
        fr = free_pairs[key]
        tr = teacher_pairs[key]
        same_pattern = int(
            str(fr.get("left_rel_majority")) == str(tr.get("left_rel_majority"))
            and str(fr.get("right_rel_majority")) == str(tr.get("right_rel_majority"))
        )
        same_symmetry_flag = int(_finite_int(fr.get("side_normalized_match"), -1) == _finite_int(tr.get("side_normalized_match"), -2))
        rows.append(
            {
                "phase_slot": key[0],
                "gap": int(key[1]),
                "free_left_rel_majority": str(fr.get("left_rel_majority")),
                "free_right_rel_majority": str(fr.get("right_rel_majority")),
                "teacher_left_rel_majority": str(tr.get("left_rel_majority")),
                "teacher_right_rel_majority": str(tr.get("right_rel_majority")),
                "same_left_right_pattern": same_pattern,
                "same_symmetry_flag": same_symmetry_flag,
            }
        )
    free_rate = _finite_float(free_sym.get("side_normalized_match_rate"))
    teacher_rate = _finite_float(teacher_sym.get("side_normalized_match_rate"))
    return {
        "paired_group_n": int(len(rows)),
        "free_teacher_left_right_pattern_match_rate": _safe_mean([_finite_float(r.get("same_left_right_pattern"), 0.0) for r in rows]),
        "free_teacher_symmetry_flag_match_rate": _safe_mean([_finite_float(r.get("same_symmetry_flag"), 0.0) for r in rows]),
        "free_minus_teacher_abs_symmetry_delta": abs(free_rate - teacher_rate)
        if math.isfinite(free_rate) and math.isfinite(teacher_rate)
        else float("nan"),
        "pairs": rows,
    }


def _clock_signature(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_extra = 0
    total_missed = 0
    total_adj = 0
    for entry_id, group in sorted(_group_list(records, lambda r: str(r.get("entry_id"))).items()):
        vals = sorted(group, key=lambda r: _finite_int(r.get("gap")))
        usable = [r for r in vals if str(r.get("free_stable_label")) and str(r.get("teacher_stable_label"))]
        extra = missed = adj = match = 0
        for prev, cur in zip(usable, usable[1:]):
            if _finite_int(cur.get("gap")) != _finite_int(prev.get("gap")) + 1:
                continue
            adj += 1
            free_changed = str(prev.get("free_stable_label")) != str(cur.get("free_stable_label"))
            teacher_changed = str(prev.get("teacher_stable_label")) != str(cur.get("teacher_stable_label"))
            if str(cur.get("free_stable_label")) == str(cur.get("teacher_stable_label")):
                match += 1
            if free_changed and not teacher_changed:
                extra += 1
            if teacher_changed and not free_changed:
                missed += 1
        total_extra += extra
        total_missed += missed
        total_adj += adj
        rows.append(
            {
                "entry_id": str(entry_id),
                "independent_unit": str(vals[0].get("independent_unit")) if vals else "",
                "cell_key": str(vals[0].get("cell_key")) if vals else "",
                "usable_gap_n": int(len(usable)),
                "adjacent_gap_n": int(adj),
                "adjacent_match_rate": float(match) / float(adj) if adj else float("nan"),
                "extra_free_transition_n": int(extra),
                "missed_teacher_transition_n": int(missed),
                "free_sequence": "".join(str(r.get("free_stable_label", ""))[:1].upper() or "." for r in usable),
                "teacher_sequence": "".join(str(r.get("teacher_stable_label", ""))[:1].upper() or "." for r in usable),
            }
        )
    return {
        "entry_n": int(len(rows)),
        "adjacent_gap_n": int(total_adj),
        "extra_free_transition_n": int(total_extra),
        "missed_teacher_transition_n": int(total_missed),
        "extra_free_transition_rate": float(total_extra) / float(total_adj) if total_adj else float("nan"),
        "missed_teacher_transition_rate": float(total_missed) / float(total_adj) if total_adj else float("nan"),
        "entries": rows,
    }


def _phase_cell_counts(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for side in PLANTED_LABELS:
        for q in PHASE_SLOTS:
            cell_key = f"{side}:{q}"
            cell_pairs = [p for p in pairs if str(p.get("start_label")) == side and str(p.get("phase_slot")) == q]
            out[cell_key] = {
                "independent_entry_n": int(len({str(p.get("independent_unit")) for p in cell_pairs})),
                "pair_n": int(len(cell_pairs)),
                "delivery_pair_n": int(sum(int(p.get("delivery_band", 0)) for p in cell_pairs)),
                "independent_units": sorted({str(p.get("independent_unit")) for p in cell_pairs}),
            }
    return out


def _by_cell_clock(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cell_key, group in sorted(_group_list(pairs, lambda p: str(p.get("cell_key"))).items()):
        out[cell_key] = _rate_summary(group)
    return out


def _recalc_from_rows(
    rows_csv: Path,
    *,
    delivery_min_gap: int,
    delivery_max_gap: int,
    seed_agree_frac: float,
    entries_per_cell: int,
    playback_agree_min: float,
    symmetry_match_min: float,
    symmetry_delta_max: float,
    determinism_min: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = _read_csv(rows_csv)
    pairs = _pair_rows(rows)
    delivery_pairs = [p for p in pairs if int(delivery_min_gap) <= _finite_int(p.get("gap"), -1) <= int(delivery_max_gap)]
    stress_pairs = list(pairs)
    records = _entry_gap_records(pairs, seed_agree_frac=float(seed_agree_frac))
    delivery_records = [r for r in records if int(delivery_min_gap) <= _finite_int(r.get("gap"), -1) <= int(delivery_max_gap)]

    clock_delivery = _rate_summary(delivery_pairs)
    clock_stress = _rate_summary(stress_pairs)
    drift_floor = _finite_float(clock_delivery.get("drift_floor_disagree_rate"))
    teacher_agree = _finite_float(clock_delivery.get("free_teacher_agree_rate"))
    gt_free = _finite_float(clock_delivery.get("free_gt_agree_rate_planted_gt"))
    gt_teacher = _finite_float(clock_delivery.get("teacher_gt_agree_rate_planted_gt"))
    gt_excess_mismatch = max(0.0, (1.0 - gt_free) - (1.0 - gt_teacher)) if math.isfinite(gt_free) and math.isfinite(gt_teacher) else float("nan")
    clock_consistency = {
        "task": "action_handoff_support_playback_vs_control",
        "rows_csv": str(rows_csv),
        "primary_clock": "teacher_forced_model_rollout",
        "secondary_clock": "gt_source_clip_support_at_entry_plus_gap_mod_clip",
        "delivery_gap_range": [int(delivery_min_gap), int(delivery_max_gap)],
        "row_n": int(len(rows)),
        "pair_n": int(len(pairs)),
        "delivery": clock_delivery,
        "stress_12_84": clock_stress,
        "by_cell_delivery": _by_cell_clock(delivery_pairs),
        "gt_secondary_adjusted": {
            "free_gt_mismatch_minus_teacher_gt_mismatch": float(gt_excess_mismatch),
            "interpretation": "positive values mean free diverges from GT clip clock beyond teacher-vs-GT model-clock mismatch; teacher remains the primary H0 clock",
        },
        "mismatch_examples_delivery": [
            {
                "entry_id": p.get("entry_id"),
                "seed_idx": p.get("seed_idx"),
                "gap": p.get("gap"),
                "cell_key": p.get("cell_key"),
                "free_label": p.get("free_label"),
                "teacher_label": p.get("teacher_label"),
                "gt_clock_label": p.get("gt_clock_label"),
                "free_margin": p.get("free_margin"),
                "teacher_margin": p.get("teacher_margin"),
            }
            for p in delivery_pairs
            if int(p.get("free_teacher_same", 0)) == 0
        ][:40],
    }

    cell_counts = _phase_cell_counts(delivery_pairs)
    det_free = _determinism(delivery_records, "free_rel_label")
    det_teacher = _determinism(delivery_records, "teacher_rel_label")
    sym_free = _side_symmetry(delivery_records, "free_rel_label")
    sym_teacher = _side_symmetry(delivery_records, "teacher_rel_label")
    sym_relative = _teacher_relative_side_symmetry(sym_free, sym_teacher)
    signature_all = _clock_signature(records)
    signature_delivery = _clock_signature(delivery_records)
    phase_symmetry = {
        "task": "action_handoff_support_playback_vs_control",
        "rows_csv": str(rows_csv),
        "phase_cell_counts_delivery": cell_counts,
        "phase_determinism_delivery": {
            "free": det_free,
            "teacher": det_teacher,
        },
        "side_symmetry_delivery": {
            "free": sym_free,
            "teacher": sym_teacher,
            "teacher_relative": sym_relative,
            "label_space": "side-normalized: same/opposite relative to entry support side",
        },
        "clock_signature": {
            "delivery": signature_delivery,
            "stress_12_84": signature_all,
            "meaning": "extra free transitions are free support changes at adjacent gap steps where teacher clock did not change; missed transitions are teacher changes not followed by free.",
        },
        "absolute_label_counts_delivery": {
            "free": dict(Counter(str(p.get("free_label")) for p in delivery_pairs)),
            "teacher": dict(Counter(str(p.get("teacher_label")) for p in delivery_pairs)),
            "gt_clock": dict(Counter(str(p.get("gt_clock_label")) for p in delivery_pairs)),
        },
    }

    invalid_reasons: list[str] = []
    short_cells = [
        cell
        for cell, rec in cell_counts.items()
        if int(rec.get("independent_entry_n", 0)) < int(entries_per_cell)
    ]
    if short_cells:
        invalid_reasons.append("phase_cell_independent_entry_n_lt_requested:" + ",".join(short_cells))
    if not delivery_pairs:
        invalid_reasons.append("no_delivery_pairs")
    side_fallback_rate = _safe_mean([_finite_float(p.get("side_fallback_used"), 0.0) for p in pairs])
    if math.isfinite(side_fallback_rate) and side_fallback_rate > 0.0:
        invalid_reasons.append(f"foot_side_fallback_used_rate:{side_fallback_rate:.6f}")
    free_internal_rate = _finite_float(clock_delivery.get("free_internal_rate"))
    teacher_internal_rate = _finite_float(clock_delivery.get("teacher_internal_rate"))
    if math.isfinite(free_internal_rate) and free_internal_rate < 0.5:
        invalid_reasons.append(f"free_internal_rate_lt_0.5:{free_internal_rate:.6f}")
    if math.isfinite(teacher_internal_rate) and teacher_internal_rate < 0.5:
        invalid_reasons.append(f"teacher_internal_rate_lt_0.5:{teacher_internal_rate:.6f}")

    free_sym = _finite_float(sym_free.get("side_normalized_match_rate"))
    teacher_sym = _finite_float(sym_teacher.get("side_normalized_match_rate"))
    symmetry_pattern_match = _finite_float(sym_relative.get("free_teacher_left_right_pattern_match_rate"))
    symmetry_flag_match = _finite_float(sym_relative.get("free_teacher_symmetry_flag_match_rate"))
    symmetry_delta = _finite_float(sym_relative.get("free_minus_teacher_abs_symmetry_delta"))
    free_det = _finite_float(det_free.get("mean_majority_frac"))
    teacher_det = _finite_float(det_teacher.get("mean_majority_frac"))
    extra_transition_rate = _finite_float(signature_all.get("extra_free_transition_rate"))
    transition_budget = drift_floor + 0.05 if math.isfinite(drift_floor) else float("nan")

    structural_deviation = bool(
        math.isfinite(extra_transition_rate)
        and math.isfinite(transition_budget)
        and extra_transition_rate > transition_budget
        and math.isfinite(teacher_agree)
        and teacher_agree < float(playback_agree_min)
    )
    playback_confirmed = bool(
        not invalid_reasons
        and math.isfinite(teacher_agree)
        and teacher_agree >= float(playback_agree_min)
        and (
            (math.isfinite(symmetry_pattern_match) and symmetry_pattern_match >= float(symmetry_match_min))
            or (math.isfinite(symmetry_delta) and symmetry_delta <= float(symmetry_delta_max))
        )
        and math.isfinite(free_det)
        and free_det >= float(determinism_min)
        and (not math.isfinite(extra_transition_rate) or not math.isfinite(transition_budget) or extra_transition_rate <= transition_budget)
    )
    if invalid_reasons:
        verdict = "INVALID"
        closeout = "采样或 FK-soft support 分辨率不足；不更新 closeout。"
    elif playback_confirmed:
        verdict = "PLAYBACK-CONFIRMED"
        closeout = (
            "关档定稿：support arrival 是确定性 gait clock 的 readout；gap 是读钟调度，不是独立 goal-steering DOF。"
        )
    elif structural_deviation:
        verdict = "NOT-PURE-PLAYBACK"
        closeout = "不下 playback 定论；记录结构性超 drift 偏离，后续解释 free-run 为何偏离 teacher clock。"
    else:
        verdict = "INVALID"
        invalid_reasons.append("metrics_inconclusive_for_playback_or_structural_deviation")
        closeout = "指标未能同时满足 playback 和 structural-deviation 判据；重设采样或阈值后复算。"

    redteam = {
        "verdict": verdict,
        "closeout_update_suggestion": closeout,
        "invalid_reasons": invalid_reasons,
        "decision_inputs": {
            "row_n": int(len(rows)),
            "pair_n": int(len(pairs)),
            "delivery_pair_n": int(len(delivery_pairs)),
            "effective_entry_n": int(len({str(p.get("independent_unit")) for p in pairs})),
            "phase_cells_requested": int(len(PLANTED_LABELS) * len(PHASE_SLOTS)),
            "entries_per_cell_requested": int(entries_per_cell),
            "free_teacher_agree_rate_delivery": teacher_agree,
            "drift_floor_disagree_rate_delivery": drift_floor,
            "free_gt_agree_rate_delivery_planted_gt": gt_free,
            "teacher_gt_agree_rate_delivery_planted_gt": gt_teacher,
            "gt_secondary_excess_mismatch": gt_excess_mismatch,
            "free_phase_determinism_delivery": free_det,
            "teacher_phase_determinism_delivery": teacher_det,
            "free_side_symmetry_delivery": free_sym,
            "teacher_side_symmetry_delivery": teacher_sym,
            "free_teacher_side_pattern_match_delivery": symmetry_pattern_match,
            "free_teacher_symmetry_flag_match_delivery": symmetry_flag_match,
            "free_teacher_abs_symmetry_delta_delivery": symmetry_delta,
            "extra_free_transition_rate_stress": extra_transition_rate,
            "transition_budget_drift_plus_0.05": transition_budget,
            "free_internal_rate_delivery": free_internal_rate,
            "teacher_internal_rate_delivery": teacher_internal_rate,
            "free_true_flight_rate_delivery": clock_delivery.get("free_true_flight_rate"),
            "free_skate_like_rate_delivery": clock_delivery.get("free_skate_like_rate"),
            "wrapped_pair_rate_delivery": clock_delivery.get("wrapped_pair_rate"),
        },
        "method_redlines": {
            "no_training": True,
            "no_weight_write": True,
            "production_trainer_gate_model_untouched": True,
            "target_not_used": ["contacts_plan", "CONTACT_SLICE"],
            "CONTACT_SLICE_scope": "entry inventory and secondary GT clip clock only",
            "FK_source": "train.validate.contact_meas_whitebox.compute_contact_meas_whitebox via existing rollout helper",
            "support_metric": "right/left soft contact margin plus dist/vxy true-flight vs skate decomposition; no 0.5 support threshold or min-run rewrite",
            "effective_n": "unique independent entry = clip/region/support-side/phase-bin, not frame/gap/seed row count",
        },
        "scope": "delivery gaps 12..30 decide; gaps 12..84 retained as stress clock-signature audit",
    }
    return clock_consistency, phase_symmetry, redteam


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only playback-vs-control falsification probe for action-handoff support arrival.",
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
    parser.add_argument("--entry-min-frame", type=int, default=0)
    parser.add_argument("--min-gap", type=int, default=12)
    parser.add_argument("--max-gap", type=int, default=84)
    parser.add_argument("--gap-step", type=int, default=1)
    parser.add_argument("--delivery-min-gap", type=int, default=12)
    parser.add_argument("--delivery-max-gap", type=int, default=30)
    parser.add_argument("--entries-per-cell", type=int, default=2)
    parser.add_argument("--seeds-per-entry", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2007)
    parser.add_argument("--isolation-frames", type=int, default=34)
    parser.add_argument("--include-start-labels", default="right,left")
    parser.add_argument("--contact-vxy-mode", default="abs")
    parser.add_argument("--margin-min", type=float, default=0.05)
    parser.add_argument("--true-flight-dist-thr", type=float, default=0.20)
    parser.add_argument("--skate-vxy-thr", type=float, default=0.20)
    parser.add_argument("--seed-agree-frac", type=float, default=1.0)
    parser.add_argument("--playback-agree-min", type=float, default=0.80)
    parser.add_argument("--symmetry-match-min", type=float, default=0.70)
    parser.add_argument("--symmetry-delta-max", type=float, default=0.05)
    parser.add_argument("--determinism-min", type=float, default=0.70)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=200)
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

    clock, symmetry, redteam = _recalc_from_rows(
        rows_csv,
        delivery_min_gap=int(args.delivery_min_gap),
        delivery_max_gap=int(args.delivery_max_gap),
        seed_agree_frac=float(args.seed_agree_frac),
        entries_per_cell=int(args.entries_per_cell),
        playback_agree_min=float(args.playback_agree_min),
        symmetry_match_min=float(args.symmetry_match_min),
        symmetry_delta_max=float(args.symmetry_delta_max),
        determinism_min=float(args.determinism_min),
    )
    _write_json(out_dir / "clock_consistency.json", clock)
    _write_json(out_dir / "phase_matched_symmetry.json", symmetry)
    _write_json(out_dir / "redteam_recalc.json", redteam)
    print(f"wrote {rows_csv}")
    print(f"wrote {out_dir / 'clock_consistency.json'}")
    print(f"wrote {out_dir / 'phase_matched_symmetry.json'}")
    print(f"wrote {out_dir / 'redteam_recalc.json'}")
    print(json.dumps(base._jsonable(redteam["decision_inputs"]), ensure_ascii=False, indent=2, allow_nan=False))
    print(f"VERDICT={redteam['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
