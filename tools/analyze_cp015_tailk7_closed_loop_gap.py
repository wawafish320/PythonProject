#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_tailk7_vs_baseline_leg_linear_probe import _make_runner_args, _resolve_device
from train.geometry import geodesic_R, reproject_rot6d, root_relative_matrices, rot6d_to_matrix
from train.models import DEFAULT_DIRECT_POSE_LEG_BONES, STAGE6_3WAY_ARMCHAIN_BONES
from train.training_MPL import RolloutSequenceInputs
from train.validate.run_freerun_cycles import (
    FreeRunCycleRunner,
    _apply_direct_arm_residual_correction_norm,
    _apply_direct_leg_so3_correction_norm,
    _build_full_cycle_sample,
    _load_json,
    _run_freerun_cycles,
)


DEFAULT_TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
DEFAULT_TAIL_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404"
    / "e3x60_adapter_factorized"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth"
)
DEFAULT_TAIL_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404"
    / "eval_model_source"
    / "e3x60_adapter_factorized"
    / "Walk_F_freerun_cycles.json"
)
DEFAULT_BASELINE_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70b_replace_lowdrift"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth"
)
DEFAULT_BASELINE_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "new70b_replace_lowdrift"
    / "Walk_F_freerun_cycles.json"
)
DEFAULT_OUT = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_closed_loop_stability_analysis_20260404"
    / "gap_analysis.json"
)


GROUP_KEYS: tuple[str, ...] = ("arm", "all_ex_root", "leg")
ABS_DEPTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("d0", 0, 0),
    ("d1_4", 1, 4),
    ("d5_9", 5, 9),
    ("d10_19", 10, 19),
    ("d20_43", 20, 43),
    ("d44_86", 44, 86),
    ("d87_173", 87, 173),
    ("d174_346", 174, 346),
    ("d347_433", 347, 433),
)
SIC_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("sic0_10", 0, 10),
    ("sic11_21", 11, 21),
    ("sic22_43", 22, 43),
    ("sic44_64", 44, 64),
    ("sic65_86", 65, 86),
)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _finite(vals: Iterable[Any]) -> np.ndarray:
    out: List[float] = []
    for v in vals:
        fv = _safe_float(v)
        if math.isfinite(fv):
            out.append(fv)
    return np.asarray(out, dtype=np.float64)


def _summary(vals: np.ndarray) -> Dict[str, float]:
    if vals.size <= 0:
        return {
            "samples": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "samples": int(vals.size),
        "mean": float(np.mean(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
    }


def _infer_bone_names(trainer: Any, joint_count: int) -> List[str]:
    names = None
    try:
        names = getattr(getattr(trainer, "loss_fn", None), "bone_names", None)
        if not names:
            names = getattr(trainer, "_bone_names", None)
        if not names:
            meta = getattr(getattr(trainer, "loss_fn", None), "meta", None)
            if isinstance(meta, dict):
                names = meta.get("bone_names") or ((meta.get("skeleton") or {}).get("bone_names"))
    except Exception:
        names = None
    out = [str(x) for x in names] if isinstance(names, (list, tuple)) else []
    if len(out) < int(joint_count):
        out = out + [f"joint_{i}" for i in range(len(out), int(joint_count))]
    return out[: int(joint_count)]


def _group_indices(names: Sequence[str], root_idx: int) -> Dict[str, List[int]]:
    leg_set = {str(x) for x in DEFAULT_DIRECT_POSE_LEG_BONES}
    arm_set = {str(x) for x in STAGE6_3WAY_ARMCHAIN_BONES}
    idx_all = [i for i in range(len(names)) if int(i) != int(root_idx)]
    idx_leg = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in leg_set]
    idx_arm = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in arm_set]
    return {
        "arm": idx_arm,
        "leg": idx_leg,
        "all_ex_root": idx_all,
    }


def _direct_local_geo_deg(
    *,
    pred_raw: torch.Tensor,
    gt_raw: torch.Tensor,
    rot_slice: slice,
    root_idx: int,
    columns: Sequence[str],
) -> torch.Tensor:
    if pred_raw.ndim != 2 or gt_raw.ndim != 2 or pred_raw.shape != gt_raw.shape:
        raise ValueError(f"pred_raw/gt_raw shape mismatch: {tuple(pred_raw.shape)} vs {tuple(gt_raw.shape)}")
    rot_w = int(rot_slice.stop - rot_slice.start)
    if rot_w <= 0 or (rot_w % 6) != 0:
        raise ValueError(f"Invalid rot slice width: {rot_w}")
    joint_count = int(rot_w // 6)
    pred6 = reproject_rot6d(pred_raw[:, rot_slice]).reshape(int(pred_raw.shape[0]), joint_count, 6)
    gt6 = reproject_rot6d(gt_raw[:, rot_slice]).reshape(int(gt_raw.shape[0]), joint_count, 6)
    pred_r = rot6d_to_matrix(pred6, columns=tuple(columns))
    gt_r = rot6d_to_matrix(gt6, columns=tuple(columns))
    pred_rel = root_relative_matrices(pred_r, int(root_idx))
    gt_rel = root_relative_matrices(gt_r, int(root_idx))
    geo = geodesic_R(pred_rel, gt_rel, reduce=None) * (180.0 / math.pi)
    if 0 <= int(root_idx) < int(geo.shape[1]):
        geo = geo.clone()
        geo[:, int(root_idx)] = 0.0
    return geo


def _select_rows(mat: np.ndarray, rows: Sequence[int], cols: Sequence[int]) -> np.ndarray:
    if len(rows) <= 0 or len(cols) <= 0:
        return np.asarray([], dtype=np.float64)
    sub = mat[np.asarray(rows, dtype=np.int64)][:, np.asarray(cols, dtype=np.int64)]
    return sub.reshape(-1).astype(np.float64, copy=False)


def _summarize_matrix_groups(
    *,
    mat: np.ndarray,
    groups: Mapping[str, Sequence[int]],
    rows: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    return {
        key: _summary(_select_rows(mat, rows, cols))
        for key, cols in groups.items()
        if key in GROUP_KEYS
    }


def _depth_bucket_rows(total_steps: int, lo: int, hi: int) -> List[int]:
    lo_i = max(0, int(lo))
    hi_i = min(int(total_steps) - 1, int(hi))
    if hi_i < lo_i:
        return []
    return list(range(lo_i, hi_i + 1))


def _sic_bucket_rows(per_step: Sequence[Mapping[str, Any]], lo: int, hi: int) -> List[int]:
    rows: List[int] = []
    for i, rec in enumerate(per_step):
        try:
            sic = int(rec.get("step_in_cycle", -1) or -1)
        except Exception:
            sic = -1
        if int(lo) <= sic <= int(hi):
            rows.append(int(i))
    return rows


def _batched_sample(sample: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if not torch.is_tensor(value):
            continue
        out[key] = value.unsqueeze(0).to(device)
    return out


def _step_inputs_for_rollout(
    trainer: Any,
    rollout: Any,
    rollout_inputs: RolloutSequenceInputs,
    step_idx: int,
) -> Any:
    return trainer._resolve_rollout_step_inputs(
        SimpleNamespace(
            step_idx=int(step_idx),
            total_steps=rollout.total_steps,
            motion=rollout.motion,
            motion_raw_local=rollout.motion_raw_local,
            y_raw_local=rollout.y_raw_local,
            state_seq=rollout_inputs.state_seq,
            gt_seq=rollout_inputs.gt_seq,
            cond_seq=rollout_inputs.cond_seq,
            cond_raw_seq=rollout_inputs.cond_raw_seq,
            contacts_seq=rollout_inputs.contacts_seq,
            angvel_seq=rollout_inputs.angvel_seq,
            pose_hist_seq=rollout_inputs.pose_hist_seq,
            cond_norm_mu=rollout.cond_norm_mu,
            cond_norm_std=rollout.cond_norm_std,
            has_time_dim=rollout.has_time_dim,
            pose_hist_state=rollout.pose_hist_state,
            plan_enable=rollout.plan_enable,
            mode=rollout.mode,
            enable_reprojection=rollout.enable_reprojection,
            time_base_local=rollout.time_base_local,
            prev_foot_pos_meas=rollout.prev_foot_pos_meas,
        )
    )


def _load_case(
    *,
    case_name: str,
    ckpt_path: Path,
    eval_json_path: Path,
    teacher_path: Path,
    device_pref: str,
) -> Dict[str, Any]:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"{case_name}: unexpected checkpoint format: {ckpt_path}")
    post_cfg = dict(payload.get("posttrain_cfg") or {})
    eval_obj = _load_json(eval_json_path)
    teacher_obj = _load_json(teacher_path)
    clip_name = str(teacher_obj.get("clip") or teacher_path.stem.replace("_teacher", ""))
    teacher_block = teacher_obj.get("teacher")
    if not isinstance(teacher_block, Mapping):
        raise RuntimeError(f"{case_name}: invalid teacher payload: {teacher_path}")
    try:
        seq_len = int(np.asarray(teacher_block.get("state_norm"), dtype=np.float32).shape[0])
    except Exception as exc:
        raise RuntimeError(f"{case_name}: failed to infer seq_len from teacher batch") from exc
    if seq_len <= 0:
        raise RuntimeError(f"{case_name}: invalid seq_len inferred from teacher batch: {seq_len}")

    bundle_json = Path(str(post_cfg.get("bundle_json") or eval_obj.get("bundle") or "")).expanduser()
    pretrain_template = Path(
        str(post_cfg.get("pretrain_template") or eval_obj.get("pretrain_template") or "")
    ).expanduser()
    encoder_bundle = Path(str(eval_obj.get("encoder_bundle") or post_cfg.get("encoder_bundle") or "")).expanduser()
    if not bundle_json.is_file():
        raise FileNotFoundError(f"{case_name}: bundle_json missing: {bundle_json}")
    if not pretrain_template.is_file():
        raise FileNotFoundError(f"{case_name}: pretrain_template missing: {pretrain_template}")
    if not encoder_bundle.is_file():
        raise FileNotFoundError(f"{case_name}: encoder_bundle missing: {encoder_bundle}")

    runner_args = _make_runner_args(
        ckpt_path=ckpt_path,
        posttrain_cfg=post_cfg,
        bundle_json=bundle_json,
        encoder_bundle=encoder_bundle,
        pretrain_template=pretrain_template,
        device=_resolve_device(device_pref),
    )
    for key, value in (
        ("time_index_mode", eval_obj.get("time_index_mode", "cycle")),
        ("direct_pose_meas_source", eval_obj.get("direct_pose_meas_source", "model")),
        ("direct_pose_plan_source", eval_obj.get("direct_pose_plan_source", "model")),
        ("contacts_meas_source", eval_obj.get("contacts_meas_source", "model")),
        ("pose_hist_source", eval_obj.get("pose_hist_source", "buffer")),
        ("pose_hist_update_source", eval_obj.get("pose_hist_update_source", "pred")),
        ("lambda_fusion_apply", bool(eval_obj.get("lambda_fusion_apply", True))),
        ("so3_corr_apply", bool(eval_obj.get("so3_corr_apply", False))),
    ):
        setattr(runner_args, key, value)

    runner = FreeRunCycleRunner(runner_args)
    npz_root = ROOT / "raw_data" / "processed_data"
    ds = runner._build_dataset(npz_root / f"{clip_name}.npz", seq_len)
    runner._ensure_model_ready(ds)
    if runner.trainer is None or runner.model is None:
        raise RuntimeError(f"{case_name}: failed to reconstruct trainer/model")

    sample = _build_full_cycle_sample(ds, ds.clips[0], seq_len=seq_len)
    batched = _batched_sample(sample, runner.device)
    trainer = runner.trainer
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        raise RuntimeError(f"{case_name}: failed to infer rot slice")
    joint_count = int((int(rot_slice.stop) - int(rot_slice.start)) // 6)
    root_idx = int(getattr(trainer, "eval_root_idx", getattr(trainer, "root_idx", 0)) or 0)
    bone_names = _infer_bone_names(trainer, joint_count)
    groups = _group_indices(bone_names, root_idx=root_idx)
    columns = getattr(getattr(trainer, "loss_fn", None), "_rot6d_columns", ("X", "Z"))
    if not (isinstance(columns, (list, tuple)) and len(columns) >= 2):
        columns = ("X", "Z")

    return {
        "case_name": case_name,
        "ckpt_path": str(ckpt_path),
        "eval_json_path": str(eval_json_path),
        "teacher_path": str(teacher_path),
        "runner": runner,
        "trainer": trainer,
        "sample": sample,
        "batched": batched,
        "rot_slice": rot_slice,
        "root_idx": int(root_idx),
        "bone_names": bone_names,
        "groups": groups,
        "columns": tuple(str(x) for x in columns[:2]),
        "runtime_overrides": {
            "encoder_bundle": str(encoder_bundle),
            "time_index_mode": getattr(runner_args, "time_index_mode"),
            "direct_pose_meas_source": getattr(runner_args, "direct_pose_meas_source"),
            "direct_pose_plan_source": getattr(runner_args, "direct_pose_plan_source"),
            "contacts_meas_source": getattr(runner_args, "contacts_meas_source"),
            "pose_hist_source": getattr(runner_args, "pose_hist_source"),
            "pose_hist_update_source": getattr(runner_args, "pose_hist_update_source"),
            "lambda_fusion_apply": bool(getattr(runner_args, "lambda_fusion_apply")),
            "so3_corr_apply": bool(getattr(runner_args, "so3_corr_apply")),
        },
    }


def _teacher_forced_payload(case: Mapping[str, Any]) -> Dict[str, Any]:
    runner = case["runner"]
    metrics_per_round, per_step, extra = _run_freerun_cycles(
        trainer=case["trainer"],
        sample=case["sample"],
        rounds=1,
        device=runner.device,
        time_index_mode=str(case["runtime_overrides"]["time_index_mode"]),
        lambda_fusion_apply=bool(case["runtime_overrides"]["lambda_fusion_apply"]),
        export_joint_direct_geolocal_series=True,
        pose_hist_source="seq",
        pose_hist_update_source="gt",
        freerun_x_gt=True,
    )
    per = (extra or {}).get("per_step_direct_geolocal_deg")
    if not isinstance(per, Mapping):
        raise RuntimeError(f"{case['case_name']}: teacher-forced run missing per_step_direct_geolocal_deg export")
    mat = np.asarray(per.get("DirectGeoLocalDeg"), dtype=np.float64)
    if mat.ndim != 2:
        raise RuntimeError(f"{case['case_name']}: invalid teacher direct geolocal matrix shape: {mat.shape}")
    return {
        "metrics_per_round": metrics_per_round,
        "per_step": per_step,
        "extra": extra,
        "direct_geolocal": mat,
        "definition": {
            "driver": "_run_freerun_cycles",
            "rounds": 1,
            "freerun_x_gt": True,
            "pose_hist_source": "seq",
            "pose_hist_update_source": "gt",
            "metric": "DirectGeoLocalDeg",
        },
    }


def _freerun_payload(case: Mapping[str, Any], rounds: int) -> Dict[str, Any]:
    runner = case["runner"]
    metrics_per_round, per_step, extra = _run_freerun_cycles(
        trainer=case["trainer"],
        sample=case["sample"],
        rounds=int(rounds),
        device=runner.device,
        time_index_mode=str(case["runtime_overrides"]["time_index_mode"]),
        lambda_fusion_apply=bool(case["runtime_overrides"]["lambda_fusion_apply"]),
        export_joint_direct_geolocal_series=True,
        pose_hist_source=str(case["runtime_overrides"]["pose_hist_source"]),
        pose_hist_update_source=str(case["runtime_overrides"]["pose_hist_update_source"]),
        debug_rot_gain=False,
    )
    per = (extra or {}).get("per_step_direct_geolocal_deg")
    if not isinstance(per, Mapping):
        raise RuntimeError(f"{case['case_name']}: freerun missing per_step_direct_geolocal_deg export")
    mat = np.asarray(per.get("DirectGeoLocalDeg"), dtype=np.float64)
    if mat.ndim != 2:
        raise RuntimeError(f"{case['case_name']}: invalid freerun direct geolocal matrix shape: {mat.shape}")
    return {
        "metrics_per_round": metrics_per_round,
        "per_step": per_step,
        "extra": extra,
        "direct_geolocal": mat,
    }


def _case_report(case: Mapping[str, Any], rounds: int) -> Dict[str, Any]:
    teacher_forced = _teacher_forced_payload(case)
    teacher_np = teacher_forced["direct_geolocal"]
    freerun = _freerun_payload(case, rounds=rounds)
    free_np = freerun["direct_geolocal"]
    per_step = freerun["per_step"]
    groups = case["groups"]

    teacher_rows = list(range(int(teacher_np.shape[0])))
    depth_rows_one = [0] if int(free_np.shape[0]) > 0 else []

    report: Dict[str, Any] = {
        "case_name": case["case_name"],
        "ckpt_path": case["ckpt_path"],
        "teacher_path": case["teacher_path"],
        "eval_json_path": case["eval_json_path"],
        "runtime_overrides": case["runtime_overrides"],
        "teacher_forced_definition": teacher_forced["definition"],
        "freerun_definition": {
            "driver": "_run_freerun_cycles",
            "rounds": int(rounds),
            "pose_hist_source": str(case["runtime_overrides"]["pose_hist_source"]),
            "pose_hist_update_source": str(case["runtime_overrides"]["pose_hist_update_source"]),
            "metric": "DirectGeoLocalDeg",
        },
        "group_names": {
            key: [case["bone_names"][i] for i in case["groups"][key]]
            for key in GROUP_KEYS
        },
        "teacher_forced_one_step": _summarize_matrix_groups(
            mat=teacher_np,
            groups=groups,
            rows=teacher_rows,
        ),
        "freerun_one_step": _summarize_matrix_groups(
            mat=free_np,
            groups=groups,
            rows=depth_rows_one,
        ),
        "freerun_depth_buckets": {},
        "freerun_step_in_cycle_buckets": {},
    }

    for label, lo, hi in ABS_DEPTH_BUCKETS:
        rows = _depth_bucket_rows(int(free_np.shape[0]), lo, hi)
        report["freerun_depth_buckets"][label] = {
            "row_range": [int(lo), int(hi)],
            "metrics": _summarize_matrix_groups(mat=free_np, groups=groups, rows=rows),
        }
    for label, lo, hi in SIC_BUCKETS:
        rows = _sic_bucket_rows(per_step, lo, hi)
        report["freerun_step_in_cycle_buckets"][label] = {
            "sic_range": [int(lo), int(hi)],
            "metrics": _summarize_matrix_groups(mat=free_np, groups=groups, rows=rows),
        }

    return report


def _build_comparison(case_a: Mapping[str, Any], case_b: Mapping[str, Any]) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    for section in ("teacher_forced_one_step", "freerun_one_step"):
        comparison[section] = {}
        for group in GROUP_KEYS:
            row_a = case_a.get(section, {}).get(group, {})
            row_b = case_b.get(section, {}).get(group, {})
            comparison[section][group] = {
                "tailk7_factorized_mean": _safe_float(row_a.get("mean")),
                "baseline_replace_mean": _safe_float(row_b.get("mean")),
                "delta_tail_minus_base_mean": _safe_float(row_a.get("mean")) - _safe_float(row_b.get("mean")),
                "tailk7_factorized_p95": _safe_float(row_a.get("p95")),
                "baseline_replace_p95": _safe_float(row_b.get("p95")),
                "delta_tail_minus_base_p95": _safe_float(row_a.get("p95")) - _safe_float(row_b.get("p95")),
            }
    for section in ("freerun_depth_buckets", "freerun_step_in_cycle_buckets"):
        comparison[section] = {}
        keys = sorted(set(case_a.get(section, {}).keys()) & set(case_b.get(section, {}).keys()))
        for bucket in keys:
            comparison[section][bucket] = {}
            for group in GROUP_KEYS:
                row_a = ((case_a.get(section, {}) or {}).get(bucket, {}) or {}).get("metrics", {}).get(group, {})
                row_b = ((case_b.get(section, {}) or {}).get(bucket, {}) or {}).get("metrics", {}).get(group, {})
                comparison[section][bucket][group] = {
                    "tailk7_factorized_mean": _safe_float(row_a.get("mean")),
                    "baseline_replace_mean": _safe_float(row_b.get("mean")),
                    "delta_tail_minus_base_mean": _safe_float(row_a.get("mean")) - _safe_float(row_b.get("mean")),
                    "tailk7_factorized_p95": _safe_float(row_a.get("p95")),
                    "baseline_replace_p95": _safe_float(row_b.get("p95")),
                    "delta_tail_minus_base_p95": _safe_float(row_a.get("p95")) - _safe_float(row_b.get("p95")),
                }
    return comparison


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal closed-loop gap decomposition for cp015 tailk7 replace.")
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--tail-ckpt", type=Path, default=DEFAULT_TAIL_CKPT)
    ap.add_argument("--tail-eval", type=Path, default=DEFAULT_TAIL_EVAL)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    teacher = args.teacher.expanduser().resolve()
    tail_ckpt = args.tail_ckpt.expanduser().resolve()
    tail_eval = args.tail_eval.expanduser().resolve()
    base_ckpt = args.baseline_ckpt.expanduser().resolve()
    base_eval = args.baseline_eval.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    for path in (teacher, tail_ckpt, tail_eval, base_ckpt, base_eval):
        if not path.is_file():
            raise SystemExit(f"[FATAL] missing input: {path}")

    tail_case = _load_case(
        case_name="tailk7_factorized",
        ckpt_path=tail_ckpt,
        eval_json_path=tail_eval,
        teacher_path=teacher,
        device_pref=str(args.device),
    )
    base_case = _load_case(
        case_name="baseline_replace",
        ckpt_path=base_ckpt,
        eval_json_path=base_eval,
        teacher_path=teacher,
        device_pref=str(args.device),
    )

    tail_report = _case_report(tail_case, rounds=int(args.rounds))
    base_report = _case_report(base_case, rounds=int(args.rounds))

    payload = {
        "analysis": "teacher_forced_vs_freerun_gap",
        "teacher_batch": str(teacher),
        "definitions": {
            "teacher_forced_one_step": "Per-step direct-head local geodesic error (out_direct vs GT), aggregated over a teacher-conditioned runtime pass driven by _run_freerun_cycles(rounds=1, freerun_x_gt=True, pose_hist_source='seq', pose_hist_update_source='gt').",
            "freerun_one_step": "Direct-head local geodesic error at rollout depth 1 (global step 0) from multi-cycle freerun on the same Walk_F clip.",
            "freerun_depth_buckets": "Direct-head local geodesic error aggregated over absolute rollout-depth buckets from a 5-cycle freerun.",
            "freerun_step_in_cycle_buckets": "Direct-head local geodesic error aggregated by step_in_cycle buckets across the full 5-cycle freerun.",
        },
        "cases": {
            "tailk7_factorized": tail_report,
            "baseline_replace": base_report,
        },
        "comparison": _build_comparison(tail_report, base_report),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
