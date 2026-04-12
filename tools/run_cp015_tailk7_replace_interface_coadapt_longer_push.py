#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
        load_json,
        make_generated_config,
        run_cmd,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        create_replace_zerophase_warmstart,
        load_json,
        make_generated_config,
        run_cmd,
        write_json,
    )

from tools.analyze_cp015_tailk7_rot_row_group_pose_swaps import (  # noqa: E402
    PRIMARY_METRICS,
    _case_summary,
    _relative_improvement,
    _safe_float,
)


RUN_DATE = "20260406"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
STATUS_JSON = OUT_ROOT / "status.json"

CPU_EXEC = ROOT / "debug_output" / "_tmp_phasecd_min_ablation_20260330" / "cpu_nomps_exec.py"
ROOT_NAME = "pelvis"
ROW_ALL_ROT = ((0, 276),)
REQUIRED_BUCKETS: Tuple[str, ...] = ("d0_9", "d10_20", "d21_43", "sic0_10", "sic11_21", "sic22_43")
SECONDARY_METRICS: Tuple[str, ...] = ("GeoLocalDeg",)
READOUT_LR = 5e-5
BASE_EPOCHS = 1
BEST_INTERFACE_LR_SCALE = 0.04
INTERFACE_MODE = "tail"
INTERFACE_PREFIXES: Tuple[str, ...] = (
    "shared_encoder.8",
    "residual_proj",
    "_pasa_lnq",
    "_pasa_q",
    "_pasa_k",
    "_pasa_v",
    "_pasa_o",
    "_pasa_film",
    "coupling_norm",
)

CURRENT_70A_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402"
    / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth"
)
CURRENT_CONTROL_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404"
    / "e3x60_adapter_factorized"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth"
)
CURRENT_CONTROL_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404"
    / "eval_model_source"
    / "e3x60_adapter_factorized"
    / "Walk_F_freerun_cycles.json"
)
BASELINE_REPLACE_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "70b_replace_lowdrift"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth"
)
BASELINE_REPLACE_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_posttrain_pipeline_from_bestfree_20260317"
    / "eval_model_source"
    / "new70b_replace_lowdrift"
    / "Walk_F_freerun_cycles.json"
)
LOWLR_WINNER_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_ep014center_replace_lowlr_sweep_20260328"
    / "configs"
    / "posttrain_70b_replace_lowdrift_lr5e5_from_ep014center_20260328.json"
)

SWEEP_OUT_ROOT = ROOT / "debug_output" / "_tmp_cp015_tailk7_replace_interface_coadapt_saturation_sweep_20260406"
SWEEP_MODEL_ROOT = ROOT / "models" / "__tmp_cp015_tailk7_replace_interface_coadapt_saturation_sweep_20260406"

ANCHOR_CASE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "coadapt_allrot_interface_lrscale_0p04",
        "row_ranges": ROW_ALL_ROT,
        "interface_mode": INTERFACE_MODE,
        "interface_lr_scale": BEST_INTERFACE_LR_SCALE,
        "epochs": BASE_EPOCHS,
        "steps_per_epoch": 60,
        "reuse_artifacts": {
            "config": SWEEP_OUT_ROOT / "configs" / "posttrain_70b_replace_lowdrift_coadapt_allrot_interface_lrscale_0p04_20260406.json",
            "ckpt": SWEEP_MODEL_ROOT
            / "coadapt_allrot_interface_lrscale_0p04"
            / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_lrscale_0p04_20260406.pth",
            "eval": SWEEP_OUT_ROOT
            / "eval_model_source"
            / "coadapt_allrot_interface_lrscale_0p04"
            / "Walk_F_freerun_cycles.json",
            "note": "reuse prior 2026-04-06 0.04 @ 60 sweep artifacts",
        },
    },
    {
        "name": "coadapt_allrot_interface_bestlr_longer_1p5x",
        "row_ranges": ROW_ALL_ROT,
        "interface_mode": INTERFACE_MODE,
        "interface_lr_scale": BEST_INTERFACE_LR_SCALE,
        "epochs": BASE_EPOCHS,
        "steps_per_epoch": 90,
        "reuse_artifacts": {
            "config": SWEEP_OUT_ROOT
            / "configs"
            / "posttrain_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_1p5x_20260406.json",
            "ckpt": SWEEP_MODEL_ROOT
            / "coadapt_allrot_interface_bestlr_longer_1p5x"
            / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_1p5x_20260406.pth",
            "eval": SWEEP_OUT_ROOT
            / "eval_model_source"
            / "coadapt_allrot_interface_bestlr_longer_1p5x"
            / "Walk_F_freerun_cycles.json",
            "note": "reuse prior 2026-04-06 0.04 @ 90 sweep artifacts",
        },
    },
    {
        "name": "coadapt_allrot_interface_bestlr_longer_2x",
        "row_ranges": ROW_ALL_ROT,
        "interface_mode": INTERFACE_MODE,
        "interface_lr_scale": BEST_INTERFACE_LR_SCALE,
        "epochs": BASE_EPOCHS,
        "steps_per_epoch": 120,
        "reuse_artifacts": {
            "config": SWEEP_OUT_ROOT
            / "configs"
            / "posttrain_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_2x_20260406.json",
            "ckpt": SWEEP_MODEL_ROOT
            / "coadapt_allrot_interface_bestlr_longer_2x"
            / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_2x_20260406.pth",
            "eval": SWEEP_OUT_ROOT
            / "eval_model_source"
            / "coadapt_allrot_interface_bestlr_longer_2x"
            / "Walk_F_freerun_cycles.json",
            "note": "reuse prior 2026-04-06 0.04 @ 120 sweep artifacts",
        },
    },
)

CASE_3X: Dict[str, Any] = {
    "name": "coadapt_allrot_interface_bestlr_longer_3x",
    "row_ranges": ROW_ALL_ROT,
    "interface_mode": INTERFACE_MODE,
    "interface_lr_scale": BEST_INTERFACE_LR_SCALE,
    "epochs": BASE_EPOCHS,
    "steps_per_epoch": 180,
}
CASE_4X: Dict[str, Any] = {
    "name": "coadapt_allrot_interface_bestlr_longer_4x",
    "row_ranges": ROW_ALL_ROT,
    "interface_mode": INTERFACE_MODE,
    "interface_lr_scale": BEST_INTERFACE_LR_SCALE,
    "epochs": BASE_EPOCHS,
    "steps_per_epoch": 240,
}

WARMSTART_CKPT = MODEL_ROOT / "warmstart" / f"ckpt_last_cp015_tailk7_70a_replace_zerophase_{RUN_DATE}.pth"
WARMSTART_REPORT = OUT_ROOT / "warmstart" / "replace_zerophase_report.json"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifacts:\n" + "\n".join(missing))


def _fmt(x: Any) -> str:
    val = _safe_float(x)
    return "nan" if not math.isfinite(val) else f"{val:.6f}"


def _fmt_pct(x: Any) -> str:
    val = _safe_float(x)
    return "nan" if not math.isfinite(val) else f"{100.0 * val:+.2f}%"


def _load_eval_summary(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    return _case_summary(
        {
            "metrics_per_round": list(payload.get("metrics_per_round", []) or []),
            "metrics_per_step": list(payload.get("metrics_per_step", []) or []),
        },
        root_name=ROOT_NAME,
    )


def _primary_relative_mean(case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any]) -> float:
    vals: List[float] = []
    for metric in PRIMARY_METRICS:
        cur = (((ref_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        var = (((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        rel = _relative_improvement(cur, var)
        if math.isfinite(rel):
            vals.append(float(rel))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _primary_bucket_relative_mean(case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any], bucket: str) -> float:
    vals: List[float] = []
    for metric in PRIMARY_METRICS:
        cur = (
            (((ref_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}) or {}
        ).get("mean")
        var = (
            (((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("buckets", {}) or {}).get(bucket, {}) or {}
        ).get("mean")
        rel = _relative_improvement(cur, var)
        if math.isfinite(rel):
            vals.append(float(rel))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _overall_primary_row(label: str, case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"variant": label}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        row[metric] = (
            (((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        )
    row["primary_relative_improvement_vs_current"] = _primary_relative_mean(case_summary, ref_summary)
    return row


def _bucket_row(label: str, case_summary: Mapping[str, Any], ref_summary: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"variant": label}
    for bucket in REQUIRED_BUCKETS:
        row[bucket] = _primary_bucket_relative_mean(case_summary, ref_summary, bucket)
    return row


def _primary_delta_vs_anchor_row(
    label: str,
    case_summary: Mapping[str, Any],
    anchor_summary: Mapping[str, Any],
    ref_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "variant": label,
        "overall_primary_rel_delta_vs_anchor": (
            _primary_relative_mean(case_summary, ref_summary) - _primary_relative_mean(anchor_summary, ref_summary)
        ),
    }
    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        cur = (((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        anc = (((anchor_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean")
        row[metric] = _safe_float(cur) - _safe_float(anc)
    return row


def _delta_vs_anchor_bucket_row(
    label: str,
    case_summary: Mapping[str, Any],
    anchor_summary: Mapping[str, Any],
    ref_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "variant": label,
        "overall_primary_rel_delta_vs_anchor": (
            _primary_relative_mean(case_summary, ref_summary) - _primary_relative_mean(anchor_summary, ref_summary)
        ),
    }
    for bucket in REQUIRED_BUCKETS:
        row[bucket] = _primary_bucket_relative_mean(case_summary, ref_summary, bucket) - _primary_bucket_relative_mean(
            anchor_summary,
            ref_summary,
            bucket,
        )
    return row


def _primary_metric_mean(case_summary: Mapping[str, Any], metric: str) -> float:
    return _safe_float((((case_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean"))


def _primary_metric_win_count(case_summary: Mapping[str, Any], other_summary: Mapping[str, Any]) -> int:
    wins = 0
    for metric in PRIMARY_METRICS:
        cur = _primary_metric_mean(case_summary, metric)
        oth = _primary_metric_mean(other_summary, metric)
        if math.isfinite(cur) and math.isfinite(oth) and cur < oth:
            wins += 1
    return int(wins)


def _longer_still_improving(
    longer_summary: Mapping[str, Any],
    base_summary: Mapping[str, Any],
    current_summary: Mapping[str, Any],
) -> bool:
    if _primary_relative_mean(longer_summary, current_summary) <= _primary_relative_mean(base_summary, current_summary):
        return False
    return _primary_metric_win_count(longer_summary, base_summary) >= 3


def _case_total_steps(case: Mapping[str, Any]) -> int:
    return int(case["epochs"]) * int(case["steps_per_epoch"])


def _make_case_config(case: Mapping[str, Any]) -> Tuple[Path, Path, str]:
    case_name = str(case["name"])
    out_dir = MODEL_ROOT / case_name
    run_name = f"WalkF_stage7_70b_replace_lowdrift_{case_name}_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_70b_replace_lowdrift_{case_name}_{RUN_DATE}.json"
    make_generated_config(
        LOWLR_WINNER_CONFIG,
        cfg_json,
        {
            "ckpt_in": str(WARMSTART_CKPT),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "device": "cpu",
            "epochs": int(case["epochs"]),
            "steps_per_epoch": int(case["steps_per_epoch"]),
            "lr": READOUT_LR,
            "weight_decay": 0.0,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
            "optimizer_param_group_overrides": None,
            "train_direct_pose": False,
            "train_incremental_replace": True,
            "train_lambda_head": False,
            "train_arm_residual": False,
            "train_arm_leg_residual": False,
            "incremental_motion_head_row_ranges": [[int(st), int(ed)] for st, ed in case["row_ranges"]],
            "incremental_interface_mode": str(case["interface_mode"]),
            "incremental_interface_lr_scale": float(case["interface_lr_scale"]),
        },
    )
    return cfg_json, out_dir, run_name


def _run_posttrain_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    case_name = str(case["name"])
    reuse_artifacts = case.get("reuse_artifacts")
    if isinstance(reuse_artifacts, Mapping):
        config_path = Path(str(reuse_artifacts["config"]))
        ckpt = Path(str(reuse_artifacts["ckpt"]))
        eval_json = Path(str(reuse_artifacts["eval"]))
        if config_path.is_file() and ckpt.is_file() and eval_json.is_file():
            case_summary = _load_eval_summary(eval_json)
            return {
                "name": case_name,
                "config": str(config_path),
                "ckpt": str(ckpt),
                "eval": str(eval_json),
                "summary": case_summary,
                "row_ranges": [[int(st), int(ed)] for st, ed in case["row_ranges"]],
                "interface_mode": str(case["interface_mode"]),
                "interface_lr_scale": float(case["interface_lr_scale"]),
                "epochs": int(case["epochs"]),
                "steps_per_epoch": int(case["steps_per_epoch"]),
                "total_steps": int(_case_total_steps(case)),
                "reuse_existing_artifacts": True,
                "reuse_note": str(reuse_artifacts.get("note", "")),
            }

    cfg_json, out_dir, run_name = _make_case_config(case)
    ckpt = out_dir / f"ckpt_last_{run_name}.pth"
    eval_dir = OUT_ROOT / "eval_model_source" / case_name
    eval_json = eval_dir / "Walk_F_freerun_cycles.json"
    if not ckpt.is_file():
        run_cmd(
            [
                sys.executable,
                str(CPU_EXEC),
                "-m",
                "train.posttrain",
                "--config",
                str(cfg_json),
                "--ckpt_in",
                str(WARMSTART_CKPT),
                "--out_dir",
                str(out_dir),
                "--run_name",
                run_name,
                "--posttrain_contacts_source",
                "pretrain_contact",
                "--posttrain_contacts_pretrain_clamp",
                str(PRETRAIN_CLAMP),
                "--encoder_bundle",
                str(ENCODER_BUNDLE),
                "--posttrain_contacts_pretrain_affine_stats",
                str(AFFINE_STATS),
            ],
            log_file=LOG_FILE,
        )
    if not eval_json.is_file():
        run_cmd(
            [
                sys.executable,
                str(CPU_EXEC),
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                str(ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"),
                "--model",
                str(ckpt),
                "--rounds",
                "5",
                "--depth",
                "3",
                "--time-index-mode",
                "cycle",
                "--event_clock",
                "auto",
                "--phase_reset_source",
                "none",
                "--contacts_meas_source",
                "model",
                "--lambda_fusion_apply",
                "--log_contacts",
                "--export_direct_arm_probe",
                "--export_joint_direct_geolocal_series",
                "--out",
                str(eval_dir),
                "--force",
            ],
            log_file=LOG_FILE,
        )
    case_summary = _load_eval_summary(eval_json)
    return {
        "name": case_name,
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "eval": str(eval_json),
        "summary": case_summary,
        "row_ranges": [[int(st), int(ed)] for st, ed in case["row_ranges"]],
        "interface_mode": str(case["interface_mode"]),
        "interface_lr_scale": float(case["interface_lr_scale"]),
        "epochs": int(case["epochs"]),
        "steps_per_epoch": int(case["steps_per_epoch"]),
        "total_steps": int(_case_total_steps(case)),
        "reuse_existing_artifacts": False,
    }


def _load_model_state(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"checkpoint missing model state: {ckpt_path}")
    return {str(k): v.detach().cpu().float() for k, v in state.items() if torch.is_tensor(v)}


def _keys_for_prefix(base_state: Mapping[str, torch.Tensor], target_state: Mapping[str, torch.Tensor], prefix: str) -> List[str]:
    keys = [str(k) for k in base_state.keys() if str(k).startswith(prefix) and str(k) in target_state]
    if not keys:
        raise RuntimeError(f"no matching tensors for prefix {prefix!r}")
    return sorted(keys)


def _metrics_for_keys(
    *,
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    keys: Sequence[str],
) -> Dict[str, Any]:
    diff_chunks: List[torch.Tensor] = []
    base_chunks: List[torch.Tensor] = []
    param_count = 0
    for key in keys:
        base = base_state[str(key)]
        target = target_state[str(key)]
        if tuple(base.shape) != tuple(target.shape):
            raise RuntimeError(f"shape mismatch for {key}: base={tuple(base.shape)} target={tuple(target.shape)}")
        diff = (target - base).reshape(-1)
        diff_chunks.append(diff)
        base_chunks.append(base.reshape(-1))
        param_count += int(base.numel())
    diff_vec = torch.cat(diff_chunks) if diff_chunks else torch.zeros(0, dtype=torch.float32)
    base_vec = torch.cat(base_chunks) if base_chunks else torch.zeros(0, dtype=torch.float32)
    if diff_vec.numel() <= 0:
        max_abs_diff = float("nan")
        mean_abs_diff = float("nan")
        rms_diff = float("nan")
    else:
        abs_diff = diff_vec.abs()
        max_abs_diff = float(abs_diff.max().item())
        mean_abs_diff = float(abs_diff.mean().item())
        rms_diff = float(diff_vec.pow(2).mean().sqrt().item())
    if base_vec.numel() <= 0:
        rel_rms_vs_base = float("nan")
    else:
        base_rms = float(base_vec.pow(2).mean().sqrt().item())
        rel_rms_vs_base = float(rms_diff / base_rms) if base_rms > 0.0 else float("nan")
    return {
        "tensor_count": int(len(keys)),
        "param_count": int(param_count),
        "sample_keys": list(keys[:6]),
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(mean_abs_diff),
        "rms_diff": float(rms_diff),
        "rel_rms_vs_base": float(rel_rms_vs_base),
    }


def _compute_integrity_report(target_ckpt: Path) -> Dict[str, Any]:
    base_state = _load_model_state(CURRENT_70A_CKPT)
    target_state = _load_model_state(target_ckpt)
    per_prefix: Dict[str, Any] = {}
    overall_keys: List[str] = []
    for prefix in INTERFACE_PREFIXES:
        keys = _keys_for_prefix(base_state, target_state, prefix)
        per_prefix[str(prefix)] = _metrics_for_keys(base_state=base_state, target_state=target_state, keys=keys)
        overall_keys.extend(keys)
    overall_keys = sorted(dict.fromkeys(overall_keys))
    return {
        "reference_ckpt": str(CURRENT_70A_CKPT),
        "target_ckpt": str(target_ckpt),
        "prefixes": list(INTERFACE_PREFIXES),
        "overall": _metrics_for_keys(base_state=base_state, target_state=target_state, keys=overall_keys),
        "per_prefix": per_prefix,
    }


def _trajectory_slope_per_step(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_path: Tuple[str, ...],
) -> Optional[Dict[str, Any]]:
    if len(rows) < 3:
        return None

    def _get(row: Mapping[str, Any]) -> float:
        cur: Any = row
        for key in key_path:
            if not isinstance(cur, Mapping):
                return float("nan")
            cur = cur.get(key)
        return _safe_float(cur)

    prev_row = rows[-2]
    prev_prev_row = rows[-3]
    cur_row = rows[-1]
    prev_delta_steps = int(prev_row["total_steps"]) - int(prev_prev_row["total_steps"])
    cur_delta_steps = int(cur_row["total_steps"]) - int(prev_row["total_steps"])
    prev_val = _get(prev_prev_row)
    mid_val = _get(prev_row)
    cur_val = _get(cur_row)
    if (
        prev_delta_steps <= 0
        or cur_delta_steps <= 0
        or not math.isfinite(prev_val)
        or not math.isfinite(mid_val)
        or not math.isfinite(cur_val)
    ):
        return None
    prev_slope = (mid_val - prev_val) / float(prev_delta_steps)
    cur_slope = (cur_val - mid_val) / float(cur_delta_steps)
    ratio = float("inf") if abs(prev_slope) <= 1e-12 else float(cur_slope / prev_slope)
    return {
        "prev_segment": f"{int(prev_prev_row['total_steps'])}->{int(prev_row['total_steps'])}",
        "current_segment": f"{int(prev_row['total_steps'])}->{int(cur_row['total_steps'])}",
        "prev_slope_per_step": float(prev_slope),
        "current_slope_per_step": float(cur_slope),
        "slope_ratio": float(ratio),
    }


def _integrity_is_smooth_and_safe(trajectory_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not trajectory_rows:
        return {
            "safe": False,
            "reason": "missing integrity trajectory",
        }
    latest = trajectory_rows[-1]
    overall_rel = _safe_float(latest.get("rel_rms_vs_base"))
    overall_max = _safe_float(latest.get("max_abs_diff"))
    shared_max = _safe_float(latest.get("shared_encoder_8_max_abs_diff"))
    overall_slope = _trajectory_slope_per_step(trajectory_rows, key_path=("rel_rms_vs_base",))
    shared_slope = _trajectory_slope_per_step(trajectory_rows, key_path=("shared_encoder_8_max_abs_diff",))
    smooth = True
    reasons: List[str] = []
    if not math.isfinite(overall_rel) or overall_rel > 0.005:
        smooth = False
        reasons.append("overall rel_rms_vs_base crossed 0.5%")
    if not math.isfinite(overall_max) or overall_max > 0.001:
        smooth = False
        reasons.append("overall max_abs_diff crossed 1e-3")
    if not math.isfinite(shared_max) or shared_max > 5e-4:
        smooth = False
        reasons.append("shared_encoder.8 max_abs_diff crossed 5e-4")
    if overall_slope is not None and math.isfinite(float(overall_slope["slope_ratio"])) and float(overall_slope["slope_ratio"]) > 1.75:
        smooth = False
        reasons.append("overall rel_rms_vs_base slope accelerated >1.75x")
    if shared_slope is not None and math.isfinite(float(shared_slope["slope_ratio"])) and float(shared_slope["slope_ratio"]) > 1.75:
        smooth = False
        reasons.append("shared_encoder.8 max_abs_diff slope accelerated >1.75x")
    return {
        "safe": bool(smooth),
        "reason": "smooth small-growth trajectory" if smooth else "; ".join(reasons),
        "latest_overall_rel_rms_vs_base": float(overall_rel),
        "latest_overall_max_abs_diff": float(overall_max),
        "latest_shared_encoder_8_max_abs_diff": float(shared_max),
        "overall_rel_rms_slope": overall_slope,
        "shared_encoder_8_max_abs_slope": shared_slope,
    }


def _write_status(case_payloads: Mapping[str, Any], executed_order: Sequence[str], notes: Mapping[str, Any]) -> None:
    write_json(
        STATUS_JSON,
        {
            "done_cases": list(case_payloads.keys()),
            "executed_order": list(executed_order),
            "notes": dict(notes),
        },
    )


def _build_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# cp015 tailk7 replace interface co-adaptation longer push",
        "",
        "## Code Facts",
        "",
    ]
    for fact in summary["code_facts"]["current_train_incremental_replace"]:
        lines.append(f"- {fact}")
    lines.append("")
    for fact in summary["code_facts"]["longer_push_and_integrity_runner"]:
        lines.append(f"- {fact}")
    lines.extend(
        [
            "",
            "## Experiment Matrix",
            "",
            "| variant | row_ranges | interface_mode | interface_lr_scale | epochs | steps_per_epoch | total_steps | reuse |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["experiment_matrix"]:
        lines.append(
            f"| {row['name']} | {row['row_ranges']} | {row['interface_mode']} | {float(row['interface_lr_scale']):.4f} | "
            f"{int(row['epochs'])} | {int(row['steps_per_epoch'])} | {int(row['total_steps'])} | {row['reuse_existing_artifacts']} |"
        )
    lines.extend(
        [
            "",
            "## Primary Pose Table",
            "",
            "| variant | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean | GeoLocalDeg | primary rel vs current |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["tables"]["primary_pose"]:
        lines.append(
            f"| {row['variant']} | {_fmt(row['Rot6dLocalL2'])} | {_fmt(row['Rot6dLocalL2Weighted'])} | {_fmt(row['GeoDeg'])} | "
            f"{_fmt(row['KeyBoneGeoDegMean'])} | {_fmt(row['KeyBoneGeoLocalDegMean'])} | {_fmt(row['GeoLocalDeg'])} | "
            f"{_fmt_pct(row['primary_relative_improvement_vs_current'])} |"
        )
    lines.extend(
        [
            "",
            "## Primary Delta Vs 2x",
            "",
            "| variant | overall primary delta vs 2x | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean | GeoLocalDeg |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["tables"]["primary_delta_vs_2x"]:
        lines.append(
            f"| {row['variant']} | {_fmt_pct(row['overall_primary_rel_delta_vs_anchor'])} | {_fmt(row['Rot6dLocalL2'])} | "
            f"{_fmt(row['Rot6dLocalL2Weighted'])} | {_fmt(row['GeoDeg'])} | {_fmt(row['KeyBoneGeoDegMean'])} | "
            f"{_fmt(row['KeyBoneGeoLocalDegMean'])} | {_fmt(row['GeoLocalDeg'])} |"
        )
    lines.extend(
        [
            "",
            "## Bucket-Wise Table",
            "",
            "| variant | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["tables"]["bucket_primary_relative_vs_current"]:
        lines.append(
            f"| {row['variant']} | {_fmt_pct(row['d0_9'])} | {_fmt_pct(row['d10_20'])} | {_fmt_pct(row['d21_43'])} | "
            f"{_fmt_pct(row['sic0_10'])} | {_fmt_pct(row['sic11_21'])} | {_fmt_pct(row['sic22_43'])} |"
        )
    lines.extend(
        [
            "",
            "## Bucket Delta Vs 2x",
            "",
            "| variant | overall primary delta vs 2x | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["tables"]["bucket_delta_vs_2x"]:
        lines.append(
            f"| {row['variant']} | {_fmt_pct(row['overall_primary_rel_delta_vs_anchor'])} | {_fmt_pct(row['d0_9'])} | "
            f"{_fmt_pct(row['d10_20'])} | {_fmt_pct(row['d21_43'])} | {_fmt_pct(row['sic0_10'])} | "
            f"{_fmt_pct(row['sic11_21'])} | {_fmt_pct(row['sic22_43'])} |"
        )
    lines.extend(
        [
            "",
            "## Donor Integrity Trajectory",
            "",
            "| variant | total_steps | max_abs_diff | mean_abs_diff | rms_diff | rel_rms_vs_base | shared_encoder.8 max_abs_diff | shared_encoder.8 rel_rms_vs_base |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["tables"]["donor_integrity_overall"]:
        lines.append(
            f"| {row['variant']} | {int(row['total_steps'])} | {_fmt(row['max_abs_diff'])} | {_fmt(row['mean_abs_diff'])} | "
            f"{_fmt(row['rms_diff'])} | {_fmt_pct(row['rel_rms_vs_base'])} | {_fmt(row['shared_encoder_8_max_abs_diff'])} | "
            f"{_fmt_pct(row['shared_encoder_8_rel_rms_vs_base'])} |"
        )
    lines.extend(
        [
            "",
            "## Donor Integrity Per Prefix",
            "",
            "| variant | prefix | max_abs_diff | mean_abs_diff | rms_diff | rel_rms_vs_base |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["tables"]["donor_integrity_per_prefix"]:
        lines.append(
            f"| {row['variant']} | {row['prefix']} | {_fmt(row['max_abs_diff'])} | {_fmt(row['mean_abs_diff'])} | "
            f"{_fmt(row['rms_diff'])} | {_fmt_pct(row['rel_rms_vs_base'])} |"
        )
    lines.extend(
        [
            "",
            "## Judgement",
            "",
            f"- q1_3x_pose_primary: {summary['judgements']['q1_3x_pose_primary']}",
            f"- q2_3x_gain_location: {summary['judgements']['q2_3x_gain_location']}",
            f"- q3_donor_integrity_trajectory: {summary['judgements']['q3_donor_integrity_trajectory']}",
            f"- q4_shared_encoder_8_drift: {summary['judgements']['q4_shared_encoder_8_drift']}",
            f"- q5_worth_running_4x: {summary['judgements']['q5_worth_running_4x']}",
            f"- q6_next_priority: {summary['judgements']['q6_next_priority']}",
            f"- q7_proximity_or_ewc_priority: {summary['judgements']['q7_proximity_or_ewc_priority']}",
            f"- q8_adapter_priority: {summary['judgements']['q8_adapter_priority']}",
            f"- final_recommendation: {summary['judgements']['final_recommendation']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    assert_exists(
        [
            CURRENT_70A_CKPT,
            CURRENT_CONTROL_CKPT,
            CURRENT_CONTROL_EVAL,
            BASELINE_REPLACE_CKPT,
            BASELINE_REPLACE_EVAL,
            LOWLR_WINNER_CONFIG,
            CPU_EXEC,
            ENCODER_BUNDLE,
            AFFINE_STATS,
            Path(str(ANCHOR_CASE_SPECS[0]["reuse_artifacts"]["config"])),
            Path(str(ANCHOR_CASE_SPECS[0]["reuse_artifacts"]["ckpt"])),
            Path(str(ANCHOR_CASE_SPECS[0]["reuse_artifacts"]["eval"])),
            Path(str(ANCHOR_CASE_SPECS[1]["reuse_artifacts"]["config"])),
            Path(str(ANCHOR_CASE_SPECS[1]["reuse_artifacts"]["ckpt"])),
            Path(str(ANCHOR_CASE_SPECS[1]["reuse_artifacts"]["eval"])),
            Path(str(ANCHOR_CASE_SPECS[2]["reuse_artifacts"]["config"])),
            Path(str(ANCHOR_CASE_SPECS[2]["reuse_artifacts"]["ckpt"])),
            Path(str(ANCHOR_CASE_SPECS[2]["reuse_artifacts"]["eval"])),
        ]
    )
    create_replace_zerophase_warmstart(CURRENT_70A_CKPT, WARMSTART_CKPT, WARMSTART_REPORT)

    current_summary = _load_eval_summary(CURRENT_CONTROL_EVAL)
    baseline_summary = _load_eval_summary(BASELINE_REPLACE_EVAL)

    case_payloads: Dict[str, Any] = {}
    executed_order: List[str] = []
    notes: Dict[str, Any] = {}

    for case in ANCHOR_CASE_SPECS:
        case_name = str(case["name"])
        log(f"loading anchor {case_name}")
        case_payloads[case_name] = _run_posttrain_case(case)
        executed_order.append(case_name)
        _write_status(case_payloads, executed_order, notes)

    log(f"running {CASE_3X['name']}")
    case_payloads[str(CASE_3X["name"])] = _run_posttrain_case(CASE_3X)
    executed_order.append(str(CASE_3X["name"]))
    _write_status(case_payloads, executed_order, notes)

    summary_2x = case_payloads["coadapt_allrot_interface_bestlr_longer_2x"]["summary"]
    summary_3x = case_payloads["coadapt_allrot_interface_bestlr_longer_3x"]["summary"]
    three_x_improved = _longer_still_improving(summary_3x, summary_2x, current_summary)
    notes["three_x_improved"] = bool(three_x_improved)

    integrity_reports: Dict[str, Any] = {}
    trajectory_case_names = [
        "coadapt_allrot_interface_lrscale_0p04",
        "coadapt_allrot_interface_bestlr_longer_1p5x",
        "coadapt_allrot_interface_bestlr_longer_2x",
        "coadapt_allrot_interface_bestlr_longer_3x",
    ]
    donor_integrity_overall_rows: List[Dict[str, Any]] = []
    for case_name in trajectory_case_names:
        report = _compute_integrity_report(Path(str(case_payloads[case_name]["ckpt"])))
        integrity_reports[case_name] = report
        donor_integrity_overall_rows.append(
            {
                "variant": case_name,
                "total_steps": int(case_payloads[case_name]["total_steps"]),
                "max_abs_diff": float(report["overall"]["max_abs_diff"]),
                "mean_abs_diff": float(report["overall"]["mean_abs_diff"]),
                "rms_diff": float(report["overall"]["rms_diff"]),
                "rel_rms_vs_base": float(report["overall"]["rel_rms_vs_base"]),
                "shared_encoder_8_max_abs_diff": float(report["per_prefix"]["shared_encoder.8"]["max_abs_diff"]),
                "shared_encoder_8_rel_rms_vs_base": float(report["per_prefix"]["shared_encoder.8"]["rel_rms_vs_base"]),
            }
        )
    integrity_guard = _integrity_is_smooth_and_safe(donor_integrity_overall_rows)
    notes["integrity_guard_after_3x"] = integrity_guard

    run_4x = bool(three_x_improved and integrity_guard["safe"])
    notes["run_4x"] = bool(run_4x)
    if run_4x:
        log(f"running {CASE_4X['name']}")
        case_payloads[str(CASE_4X["name"])] = _run_posttrain_case(CASE_4X)
        executed_order.append(str(CASE_4X["name"]))
        _write_status(case_payloads, executed_order, notes)
        summary_4x = case_payloads["coadapt_allrot_interface_bestlr_longer_4x"]["summary"]
        notes["four_x_improved_vs_3x"] = bool(_longer_still_improving(summary_4x, summary_3x, current_summary))
        report_4x = _compute_integrity_report(Path(str(case_payloads["coadapt_allrot_interface_bestlr_longer_4x"]["ckpt"])))
        integrity_reports["coadapt_allrot_interface_bestlr_longer_4x"] = report_4x
        donor_integrity_overall_rows.append(
            {
                "variant": "coadapt_allrot_interface_bestlr_longer_4x",
                "total_steps": int(case_payloads["coadapt_allrot_interface_bestlr_longer_4x"]["total_steps"]),
                "max_abs_diff": float(report_4x["overall"]["max_abs_diff"]),
                "mean_abs_diff": float(report_4x["overall"]["mean_abs_diff"]),
                "rms_diff": float(report_4x["overall"]["rms_diff"]),
                "rel_rms_vs_base": float(report_4x["overall"]["rel_rms_vs_base"]),
                "shared_encoder_8_max_abs_diff": float(report_4x["per_prefix"]["shared_encoder.8"]["max_abs_diff"]),
                "shared_encoder_8_rel_rms_vs_base": float(report_4x["per_prefix"]["shared_encoder.8"]["rel_rms_vs_base"]),
            }
        )

    ordered_case_names = [
        "coadapt_allrot_interface_lrscale_0p04",
        "coadapt_allrot_interface_bestlr_longer_1p5x",
        "coadapt_allrot_interface_bestlr_longer_2x",
        "coadapt_allrot_interface_bestlr_longer_3x",
    ]
    if run_4x:
        ordered_case_names.append("coadapt_allrot_interface_bestlr_longer_4x")

    primary_pose_rows = [
        _overall_primary_row("current_frozen_trunk_replace_control", current_summary, current_summary),
        _overall_primary_row("baseline_replace", baseline_summary, current_summary),
    ]
    primary_pose_rows.extend(
        _overall_primary_row(name, case_payloads[name]["summary"], current_summary) for name in ordered_case_names
    )

    bucket_rows = [
        _bucket_row("current_frozen_trunk_replace_control", current_summary, current_summary),
        _bucket_row("baseline_replace", baseline_summary, current_summary),
    ]
    bucket_rows.extend(_bucket_row(name, case_payloads[name]["summary"], current_summary) for name in ordered_case_names)

    primary_delta_vs_2x_rows = [
        _primary_delta_vs_anchor_row(
            name,
            case_payloads[name]["summary"],
            summary_2x,
            current_summary,
        )
        for name in ordered_case_names
        if name in ("coadapt_allrot_interface_bestlr_longer_3x", "coadapt_allrot_interface_bestlr_longer_4x")
    ]
    bucket_delta_vs_2x_rows = [
        _delta_vs_anchor_bucket_row(
            name,
            case_payloads[name]["summary"],
            summary_2x,
            current_summary,
        )
        for name in ordered_case_names
        if name in ("coadapt_allrot_interface_bestlr_longer_3x", "coadapt_allrot_interface_bestlr_longer_4x")
    ]

    donor_integrity_per_prefix_rows: List[Dict[str, Any]] = []
    for variant in ("coadapt_allrot_interface_bestlr_longer_3x", "coadapt_allrot_interface_bestlr_longer_4x"):
        report = integrity_reports.get(variant)
        if not isinstance(report, Mapping):
            continue
        for prefix in INTERFACE_PREFIXES:
            payload = report["per_prefix"][prefix]
            donor_integrity_per_prefix_rows.append(
                {
                    "variant": variant,
                    "prefix": prefix,
                    "max_abs_diff": float(payload["max_abs_diff"]),
                    "mean_abs_diff": float(payload["mean_abs_diff"]),
                    "rms_diff": float(payload["rms_diff"]),
                    "rel_rms_vs_base": float(payload["rel_rms_vs_base"]),
                }
            )

    experiment_matrix = [
        {
            "name": name,
            "row_ranges": case_payloads[name]["row_ranges"],
            "interface_mode": case_payloads[name]["interface_mode"],
            "interface_lr_scale": float(case_payloads[name]["interface_lr_scale"]),
            "epochs": int(case_payloads[name]["epochs"]),
            "steps_per_epoch": int(case_payloads[name]["steps_per_epoch"]),
            "total_steps": int(case_payloads[name]["total_steps"]),
            "reuse_existing_artifacts": bool(case_payloads[name].get("reuse_existing_artifacts", False)),
        }
        for name in ordered_case_names
    ]

    three_x_bucket_delta = next(
        row for row in bucket_delta_vs_2x_rows if row["variant"] == "coadapt_allrot_interface_bestlr_longer_3x"
    )
    three_x_sic_gain = float(
        max(
            _safe_float(three_x_bucket_delta["sic11_21"]),
            _safe_float(three_x_bucket_delta["sic22_43"]),
        )
    )
    three_x_depth_gain = float(
        max(
            _safe_float(three_x_bucket_delta["d10_20"]),
            _safe_float(three_x_bucket_delta["d21_43"]),
        )
    )
    q2_3x_gain_location = (
        "still concentrated in sic11_21 / sic22_43; depth buckets are positive but clearly smaller"
        if three_x_sic_gain > three_x_depth_gain
        else "gains have spread materially into depth buckets"
    )

    shared_3x = integrity_reports["coadapt_allrot_interface_bestlr_longer_3x"]["per_prefix"]["shared_encoder.8"]
    q4_shared_encoder_8_drift = (
        "still very small; no warning-level drift"
        if _safe_float(shared_3x["max_abs_diff"]) < 5e-4
        else "approaching a warning-level drift"
    )

    if len(donor_integrity_overall_rows) >= 3:
        rel_slope = _trajectory_slope_per_step(donor_integrity_overall_rows, key_path=("rel_rms_vs_base",))
        q3_donor_integrity_trajectory = (
            "smooth small growth"
            if integrity_guard["safe"]
            else "starting to accelerate"
        )
    else:
        rel_slope = None
        q3_donor_integrity_trajectory = "insufficient trajectory"

    if run_4x:
        four_x_summary = case_payloads["coadapt_allrot_interface_bestlr_longer_4x"]["summary"]
        four_x_improved = _longer_still_improving(four_x_summary, summary_3x, current_summary)
        q5_worth_running_4x = "yes; 4x was run because 3x still improved and integrity stayed smooth"
    else:
        four_x_improved = False
        q5_worth_running_4x = (
            "yes; 3x still improved and integrity stayed smooth, but 4x was not run"
            if three_x_improved and integrity_guard["safe"]
            else "no; gate failed because 3x plateaued or integrity stopped looking smooth/safe"
        )

    q1_3x_pose_primary = (
        "continues to improve on pose-side primary metrics"
        if three_x_improved
        else "starting to plateau on pose-side primary metrics"
    )
    q6_next_priority = (
        "still prioritize replace-stage longer training over going back to basetrain / 70a"
        if three_x_improved and integrity_guard["safe"]
        else "do not prioritize more longer training before revisiting the next constraint"
    )
    q7_proximity_or_ewc_priority = (
        "not yet; keep it below longer training until saturation or integrity degradation appears"
        if three_x_improved and integrity_guard["safe"]
        else "yes; it should enter the next priority band now"
    )
    q8_adapter_priority = "adapter is still not the next first priority"
    final_recommendation = (
        "continue full [0:276] final rot rows + broad interface tail with interface_lr_scale=0.04 and longer training; only bring in proximity/EWC-style constraints or go back to basetrain / 70a if longer training plateaus or donor integrity clearly worsens. not adapter-first."
        if three_x_improved and integrity_guard["safe"]
        else "hold on further longer-push escalation; only now consider proximity/EWC-style constraints or an upstream robustness return if the plateau/integrity warning is confirmed. still not adapter-first."
    )

    summary = {
        "run_date": RUN_DATE,
        "artifacts": {
            "summary_json": str(SUMMARY_JSON),
            "summary_md": str(SUMMARY_MD),
            "warmstart_ckpt": str(WARMSTART_CKPT),
            "warmstart_report": str(WARMSTART_REPORT),
            "log_file": str(LOG_FILE),
        },
        "references": {
            "current_70a_ckpt": str(CURRENT_70A_CKPT),
            "current_frozen_trunk_replace_control_ckpt": str(CURRENT_CONTROL_CKPT),
            "current_frozen_trunk_replace_control_eval": str(CURRENT_CONTROL_EVAL),
            "baseline_replace_ckpt": str(BASELINE_REPLACE_CKPT),
            "baseline_replace_eval": str(BASELINE_REPLACE_EVAL),
            "lowlr_winner_config": str(LOWLR_WINNER_CONFIG),
            "saturation_sweep_summary": str(SWEEP_OUT_ROOT / "summary.json"),
        },
        "code_facts": {
            "current_train_incremental_replace": [
                "train_incremental_replace still trains the deployed incremental path with objective='inc'; this round does not change runtime contract or loss family.",
                "incremental_motion_head_row_ranges still masks grads only on the final motion_head Linear weight/bias rows; unselected rows are zeroed by hooks.",
                "the locked topology is still full [0:276] final rot rows plus broad interface tail: shared_encoder.8 / residual_proj / _pasa_lnq / _pasa_q / _pasa_k / _pasa_v / _pasa_o / _pasa_film / coupling_norm.",
                "readout parameters stay on the default AdamW group at cfg.lr=5e-5, while interface params get an auto-created incremental_interface group at cfg.lr * 0.04.",
            ],
            "longer_push_and_integrity_runner": [
                "this round adds tools/run_cp015_tailk7_replace_interface_coadapt_longer_push.py; it does not reopen subset probe, basetrain / 70a, donor hidden drift, or adapter-first work.",
                "the runner reuses the 2026-04-06 broad-tail 0.04 anchors at 60 / 90 / 120 total steps, then runs the mandatory 3x = 180-step case from the same zerophase warmstart.",
                "4x = 240 steps is only allowed after the 180-step case if pose-side primary metrics still improve and donor integrity remains smooth/safe.",
                "donor integrity is monitored only on the broad-tail interface params against the original 70a donor, with overall and per-prefix max_abs_diff / mean_abs_diff / rms_diff / rel_rms_vs_base plus a 60->90->120->180->240 trajectory.",
            ],
        },
        "experiment_matrix": experiment_matrix,
        "cases": case_payloads,
        "integrity_reports": integrity_reports,
        "tables": {
            "primary_pose": primary_pose_rows,
            "primary_delta_vs_2x": primary_delta_vs_2x_rows,
            "bucket_primary_relative_vs_current": bucket_rows,
            "bucket_delta_vs_2x": bucket_delta_vs_2x_rows,
            "donor_integrity_overall": donor_integrity_overall_rows,
            "donor_integrity_per_prefix": donor_integrity_per_prefix_rows,
        },
        "judgements": {
            "main_question": "the main question is not whether co-adapt works; it is whether best-LR replace-stage longer training still improves and whether donor integrity is still safe.",
            "primary_metric_family": "pose-side metrics only: Rot6dLocalL2 / Rot6dLocalL2Weighted / GeoDeg / KeyBoneGeoDegMean / KeyBoneGeoLocalDegMean; GeoLocalDeg is secondary only.",
            "three_x_improved": bool(three_x_improved),
            "four_x_run": bool(run_4x),
            "four_x_improved": bool(four_x_improved),
            "integrity_guard_after_3x": integrity_guard,
            "q1_3x_pose_primary": q1_3x_pose_primary,
            "q2_3x_gain_location": q2_3x_gain_location,
            "q3_donor_integrity_trajectory": q3_donor_integrity_trajectory,
            "q4_shared_encoder_8_drift": q4_shared_encoder_8_drift,
            "q5_worth_running_4x": q5_worth_running_4x,
            "q6_next_priority": q6_next_priority,
            "q7_proximity_or_ewc_priority": q7_proximity_or_ewc_priority,
            "q8_adapter_priority": q8_adapter_priority,
            "final_recommendation": final_recommendation,
            "rel_rms_slope": rel_slope,
        },
    }
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(_build_markdown(summary), encoding="utf-8")
    write_json(
        STATUS_JSON,
        {
            "done_cases": list(case_payloads.keys()),
            "executed_order": executed_order,
            "completed": True,
            "judgements": summary["judgements"],
        },
    )
    log(f"wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
