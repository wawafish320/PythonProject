#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
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
        load_json,
        make_generated_config,
        run_cmd,
        run_eval,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        load_json,
        make_generated_config,
        run_cmd,
        run_eval,
        write_json,
    )

from tools.analyze_cp015_tailk7_rot_row_group_pose_swaps import PRIMARY_METRICS  # noqa: E402
from tools.compare_cp015_tailk7_replace_freerun_table import (  # noqa: E402
    _metrics_for_eval_json,
    _pose_case_summary,
    _safe_float,
)


RUN_DATE = "20260406"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_replace_direct_recovery_bridge_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_tailk7_replace_direct_recovery_bridge_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
STATUS_JSON = OUT_ROOT / "status.json"

CPU_EXEC = ROOT / "debug_output" / "_tmp_phasecd_min_ablation_20260330" / "cpu_nomps_exec.py"
DIRECT_GATE_MARGIN = 0.01
POSE_PRESERVE_REL_TOL = 0.01
SHORT_STEPS = 60

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
BASELINE_WARMSTART_CKPT = (
    ROOT
    / "models"
    / "__tmp_posttrain_pipeline_from_bestfree_20260317"
    / "warmstart"
    / "ckpt_last_70a_replace_zerophase_20260317.pth"
)
COADAPT_60_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_interface_coadapt_saturation_sweep_20260406"
    / "coadapt_allrot_interface_lrscale_0p04"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_lrscale_0p04_20260406.pth"
)
COADAPT_120_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_interface_coadapt_saturation_sweep_20260406"
    / "coadapt_allrot_interface_bestlr_longer_2x"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_2x_20260406.pth"
)
COADAPT_180_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "coadapt_allrot_interface_bestlr_longer_3x"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_3x_20260406.pth"
)
COADAPT_240_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "coadapt_allrot_interface_bestlr_longer_4x"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth"
)
COADAPT_240_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "eval_model_source"
    / "coadapt_allrot_interface_bestlr_longer_4x"
    / "Walk_F_freerun_cycles.json"
)
COADAPT_WARMSTART_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "warmstart"
    / "ckpt_last_cp015_tailk7_70a_replace_zerophase_20260406.pth"
)
COADAPT_240_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406"
    / "configs"
    / "posttrain_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.json"
)

ANCHORS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "current_frozen_trunk_replace_control",
        "kind": "anchor",
        "ckpt": CURRENT_CONTROL_CKPT,
        "eval": CURRENT_CONTROL_EVAL,
        "direct_source": "current_frozen_trunk_replace_control",
        "pose_source": "current_frozen_trunk_replace_control",
    },
    {
        "name": "baseline_replace",
        "kind": "anchor",
        "ckpt": BASELINE_REPLACE_CKPT,
        "eval": BASELINE_REPLACE_EVAL,
        "direct_source": "baseline_replace",
        "pose_source": "baseline_replace",
    },
    {
        "name": "coadapt_allrot_interface_bestlr_longer_4x",
        "kind": "anchor",
        "ckpt": COADAPT_240_CKPT,
        "eval": COADAPT_240_EVAL,
        "direct_source": "coadapt_allrot_interface_bestlr_longer_4x",
        "pose_source": "coadapt_allrot_interface_bestlr_longer_4x",
    },
)

SWAP_CASES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "coadapt_4x_plus_baseline_directpose_swap",
        "pose_source": "coadapt_allrot_interface_bestlr_longer_4x",
        "direct_source": "baseline_replace",
        "dst_ckpt": COADAPT_240_CKPT,
        "src_ckpt": BASELINE_REPLACE_CKPT,
        "swap_mode": "exact",
    },
    {
        "name": "coadapt_4x_plus_control_directpose_swap",
        "pose_source": "coadapt_allrot_interface_bestlr_longer_4x",
        "direct_source": "current_frozen_trunk_replace_control",
        "dst_ckpt": COADAPT_240_CKPT,
        "src_ckpt": CURRENT_CONTROL_CKPT,
        "swap_mode": "compatible_intersection",
    },
    {
        "name": "baseline_plus_coadapt4x_directpose_swap",
        "pose_source": "baseline_replace",
        "direct_source": "coadapt_allrot_interface_bestlr_longer_4x",
        "dst_ckpt": BASELINE_REPLACE_CKPT,
        "src_ckpt": COADAPT_240_CKPT,
        "swap_mode": "exact",
    },
)

DIRECT_ONLY_CASES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "coadapt_4x_directonly_calibration_short",
        "steps_per_epoch": SHORT_STEPS,
        "lr": 5e-5,
        "weight_decay": 0.0,
    },
    {
        "name": "coadapt_4x_directonly_calibration_120",
        "steps_per_epoch": 120,
        "lr": 5e-5,
        "weight_decay": 0.0,
    },
    {
        "name": "coadapt_4x_directonly_calibration_180",
        "steps_per_epoch": 180,
        "lr": 5e-5,
        "weight_decay": 0.0,
    },
    {
        "name": "coadapt_4x_directonly_calibration_240",
        "steps_per_epoch": 240,
        "lr": 5e-5,
        "weight_decay": 0.0,
        "trajectory_order": 240,
    },
    {
        "name": "coadapt_4x_directonly_calibration_360",
        "steps_per_epoch": 360,
        "lr": 5e-5,
        "weight_decay": 0.0,
        "trajectory_order": 360,
    },
    {
        "name": "coadapt_4x_directonly_calibration_240plus120_lowlr",
        "steps_per_epoch": 120,
        "lr": 3e-5,
        "weight_decay": 0.0,
        "warmstart_case": "coadapt_4x_directonly_calibration_240",
        "trajectory_order": 361,
    },
)


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


def _load_ckpt(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError(f"unexpected checkpoint format: {path}")
    return obj


def _direct_pose_keys(state_dict: Mapping[str, torch.Tensor]) -> List[str]:
    return sorted(str(key) for key in state_dict.keys() if str(key).startswith("direct_pose_"))


def _direct_pose_cmp(a_ckpt: Path, b_ckpt: Path) -> Dict[str, Any]:
    a_obj = _load_ckpt(a_ckpt)
    b_obj = _load_ckpt(b_ckpt)
    a_model = a_obj["model"]
    b_model = b_obj["model"]
    keys_a = _direct_pose_keys(a_model)
    keys_b = _direct_pose_keys(b_model)
    common = sorted(set(keys_a).intersection(keys_b))
    diff_keys: List[Dict[str, Any]] = []
    max_abs = 0.0
    for key in common:
        ta = a_model[key]
        tb = b_model[key]
        if tuple(ta.shape) != tuple(tb.shape):
            diff_keys.append(
                {
                    "key": key,
                    "reason": "shape_mismatch",
                    "a_shape": list(ta.shape),
                    "b_shape": list(tb.shape),
                }
            )
            continue
        delta = float((ta - tb).abs().max().item()) if int(ta.numel()) > 0 else 0.0
        if delta > 0.0:
            diff_keys.append({"key": key, "reason": "value_diff", "max_abs": delta})
            max_abs = max(max_abs, delta)
    return {
        "a_ckpt": str(a_ckpt),
        "b_ckpt": str(b_ckpt),
        "a_key_count": int(len(keys_a)),
        "b_key_count": int(len(keys_b)),
        "common_key_count": int(len(common)),
        "a_only_keys": [key for key in keys_a if key not in set(common)],
        "b_only_keys": [key for key in keys_b if key not in set(common)],
        "diff_key_count": int(len(diff_keys)),
        "max_abs_diff": float(max_abs),
        "diff_key_samples": diff_keys[:12],
        "all_common_tensors_identical": bool(len(diff_keys) == 0),
    }


def _materialize_anchor_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "name": str(case["name"]),
        "kind": str(case["kind"]),
        "ckpt": str(case["ckpt"]),
        "eval": str(case["eval"]),
        "direct_source": str(case["direct_source"]),
        "pose_source": str(case["pose_source"]),
    }


def _copy_direct_pose_tensors(
    *,
    name: str,
    dst_ckpt: Path,
    src_ckpt: Path,
    pose_source: str,
    direct_source: str,
    swap_mode: str,
) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / f"{name}.pth"
    report_json = OUT_ROOT / "reports" / f"{name}_swap_report.json"
    if out_ckpt.is_file() and report_json.is_file():
        report = load_json(report_json)
        return {
            "name": str(name),
            "kind": "directpose_swap",
            "ckpt": str(out_ckpt),
            "eval": str((OUT_ROOT / "eval_model_source" / name / "Walk_F_freerun_cycles.json")),
            "direct_source": str(direct_source),
            "pose_source": str(pose_source),
            "swap_mode": str(swap_mode),
            "swap_report": report,
        }
    dst_obj = _load_ckpt(dst_ckpt)
    src_obj = _load_ckpt(src_ckpt)
    dst_model = dict(dst_obj["model"])
    src_model = src_obj["model"]

    dst_keys = _direct_pose_keys(dst_model)
    src_keys = _direct_pose_keys(src_model)
    dst_key_set = set(dst_keys)
    src_key_set = set(src_keys)
    common = sorted(dst_key_set.intersection(src_key_set))
    copied: List[str] = []
    shape_mismatch: List[Dict[str, Any]] = []
    for key in common:
        dst_tensor = dst_model[key]
        src_tensor = src_model[key]
        if tuple(dst_tensor.shape) != tuple(src_tensor.shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "dst_shape": list(dst_tensor.shape),
                    "src_shape": list(src_tensor.shape),
                }
            )
            continue
        dst_model[key] = src_tensor.detach().cpu().clone()
        copied.append(key)

    if str(swap_mode) == "exact":
        expected = (set(dst_keys) == set(src_keys)) and (len(shape_mismatch) == 0)
        if not expected:
            raise RuntimeError(
                f"{name}: exact direct_pose swap requested but key sets are incompatible; "
                f"dst_only={sorted(dst_key_set - src_key_set)} src_only={sorted(src_key_set - dst_key_set)} "
                f"shape_mismatch={shape_mismatch}"
            )

    out_obj = dict(dst_obj)
    out_obj["model"] = dst_model
    out_obj["direct_recovery_bridge_meta"] = {
        "case_name": str(name),
        "pose_source": str(pose_source),
        "direct_source": str(direct_source),
        "swap_mode": str(swap_mode),
    }
    torch.save(out_obj, out_ckpt)

    report = {
        "case_name": str(name),
        "pose_source": str(pose_source),
        "direct_source": str(direct_source),
        "dst_ckpt": str(dst_ckpt),
        "src_ckpt": str(src_ckpt),
        "output_ckpt": str(out_ckpt),
        "swap_mode": str(swap_mode),
        "dst_direct_pose_key_count": int(len(dst_keys)),
        "src_direct_pose_key_count": int(len(src_keys)),
        "common_direct_pose_key_count": int(len(common)),
        "copied_direct_pose_key_count": int(len(copied)),
        "copied_direct_pose_keys": copied,
        "dst_only_keys": [key for key in dst_keys if key not in src_key_set],
        "src_only_keys": [key for key in src_keys if key not in dst_key_set],
        "shape_mismatch": shape_mismatch,
        "full_direct_pose_transplant": bool(
            len(copied) == len(dst_keys) == len(src_keys)
            and not shape_mismatch
            and not [key for key in dst_keys if key not in src_key_set]
            and not [key for key in src_keys if key not in dst_key_set]
        ),
    }
    write_json(report_json, report)
    return {
        "name": str(name),
        "kind": "directpose_swap",
        "ckpt": str(out_ckpt),
        "eval": str((OUT_ROOT / "eval_model_source" / name / "Walk_F_freerun_cycles.json")),
        "direct_source": str(direct_source),
        "pose_source": str(pose_source),
        "swap_mode": str(swap_mode),
        "swap_report": report,
    }


def _ensure_eval(case: Mapping[str, Any]) -> Path:
    eval_json = Path(str(case["eval"]))
    if eval_json.is_file():
        return eval_json
    model_ckpt = Path(str(case["ckpt"]))
    out_dir = eval_json.parent
    run_eval(
        model_ckpt=model_ckpt,
        out_dir=out_dir,
        contacts_source="model",
        log_file=LOG_FILE,
    )
    return eval_json


def _combined_case_metrics(eval_json: Path) -> Dict[str, Any]:
    runtime = _metrics_for_eval_json(eval_json, cycle_gte=1)
    pose_summary = _pose_case_summary(eval_json)
    out: Dict[str, Any] = {
        "runtime": runtime,
        "pose": {},
    }
    for metric in PRIMARY_METRICS:
        out["pose"][metric] = _safe_float(
            ((((pose_summary.get("metrics", {}) or {}).get(metric, {}) or {}).get("steps", {}) or {}).get("mean"))
        )
    out["pose"]["GeoLocalDeg"] = _safe_float(
        ((((pose_summary.get("metrics", {}) or {}).get("GeoLocalDeg", {}) or {}).get("steps", {}) or {}).get("mean"))
    )
    return out


def _pose_preservation(candidate_pose: Mapping[str, Any], ref_pose: Mapping[str, Any]) -> Dict[str, Any]:
    per_metric: Dict[str, Any] = {}
    max_rel_abs = 0.0
    within = True
    for metric in PRIMARY_METRICS:
        cur = _safe_float(candidate_pose.get(metric))
        ref = _safe_float(ref_pose.get(metric))
        abs_delta = float(cur - ref) if math.isfinite(cur) and math.isfinite(ref) else float("nan")
        denom = abs(ref) if math.isfinite(ref) and abs(ref) > 1e-12 else 1.0
        rel_abs = abs(abs_delta) / denom if math.isfinite(abs_delta) else float("nan")
        max_rel_abs = max(max_rel_abs, rel_abs if math.isfinite(rel_abs) else 0.0)
        if not math.isfinite(rel_abs) or rel_abs > POSE_PRESERVE_REL_TOL:
            within = False
        per_metric[metric] = {
            "candidate": cur,
            "reference": ref,
            "abs_delta": abs_delta,
            "rel_abs_delta": rel_abs,
        }
    return {
        "within_rel_tol": bool(within),
        "rel_tol": float(POSE_PRESERVE_REL_TOL),
        "max_rel_abs_delta": float(max_rel_abs),
        "per_metric": per_metric,
    }


def _direct_gate(candidate_runtime: Mapping[str, Any], baseline_runtime: Mapping[str, Any]) -> Dict[str, Any]:
    candidate = _safe_float(candidate_runtime.get("direct_geolocaldeg"))
    baseline = _safe_float(baseline_runtime.get("direct_geolocaldeg"))
    delta = float(candidate - baseline) if math.isfinite(candidate) and math.isfinite(baseline) else float("nan")
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta_vs_baseline": delta,
        "margin": float(DIRECT_GATE_MARGIN),
        "non_regression": bool(math.isfinite(delta) and delta <= float(DIRECT_GATE_MARGIN)),
    }


def _direct_only_run_name(case_name: str) -> str:
    return f"WalkF_stage7_70b_replace_lowdrift_{case_name}_{RUN_DATE}"


def _direct_only_ckpt_path(case_name: str) -> Path:
    return MODEL_ROOT / case_name / f"ckpt_last_{_direct_only_run_name(case_name)}.pth"


def _parse_direct_only_trainable_log(case_name: str) -> Dict[str, Any]:
    if not LOG_FILE.is_file():
        return {"found": False, "case_name": str(case_name), "log_file": str(LOG_FILE)}
    run_name = _direct_only_run_name(case_name)
    lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_block = False
    train_mode: Optional[str] = None
    trainable_count: Optional[int] = None
    sample_names: List[str] = []
    for line in lines:
        if line.startswith("$ "):
            in_block = run_name in line and "train.posttrain" in line
            continue
        if not in_block:
            continue
        if "[posttrain] mode=" in line and train_mode is None:
            train_mode = line.split("=", 1)[-1].strip()
            continue
        if "[posttrain] trainable=" not in line:
            continue
        match = re.search(r"trainable=(\d+)\s+params:\s*(.+)$", line)
        if match is not None:
            trainable_count = int(match.group(1))
            sample_names = [part.strip() for part in match.group(2).split(",") if part.strip()]
            break
    return {
        "found": trainable_count is not None,
        "case_name": str(case_name),
        "run_name": run_name,
        "log_file": str(LOG_FILE),
        "train_mode": train_mode,
        "trainable_param_count": trainable_count,
        "sample_names": sample_names,
        "all_sample_names_are_direct_pose": bool(sample_names) and all(
            name.startswith("direct_pose_") for name in sample_names
        ),
    }


def _make_direct_only_config(
    case_name: str,
    *,
    warmstart_ckpt: Path,
    steps_per_epoch: int,
    lr: float,
    weight_decay: float,
) -> Tuple[Path, Path, str]:
    out_dir = MODEL_ROOT / case_name
    run_name = _direct_only_run_name(case_name)
    cfg_json = CONFIG_ROOT / f"{case_name}_{RUN_DATE}.json"
    make_generated_config(
        COADAPT_240_CONFIG,
        cfg_json,
        {
            "ckpt_in": str(warmstart_ckpt),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "device": "cpu",
            "epochs": 1,
            "steps_per_epoch": int(steps_per_epoch),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
            "optimizer_param_group_overrides": None,
            "train_direct_pose": True,
            "train_incremental_replace": False,
            "train_lambda_head": False,
            "train_arm_residual": False,
            "train_arm_leg_residual": False,
            "incremental_motion_head_row_ranges": None,
            "incremental_interface_mode": "off",
            "incremental_interface_lr_scale": 0.0,
            "direct_pose_leg_train_only": False,
            "direct_pose_leg_gate_train_only": False,
            "direct_pose_nonleg_train_only": False,
        },
    )
    return cfg_json, out_dir, run_name


def _run_direct_only_calibration(spec: Mapping[str, Any]) -> Dict[str, Any]:
    case_name = str(spec["name"])
    steps_per_epoch = int(spec["steps_per_epoch"])
    lr = float(spec.get("lr", 5e-5))
    weight_decay = float(spec.get("weight_decay", 0.0))
    warmstart_case = spec.get("warmstart_case")
    warmstart_ckpt = (
        _direct_only_ckpt_path(str(warmstart_case))
        if warmstart_case is not None
        else COADAPT_240_CKPT
    )
    cfg_json, out_dir, run_name = _make_direct_only_config(
        case_name,
        warmstart_ckpt=warmstart_ckpt,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        weight_decay=weight_decay,
    )
    ckpt = out_dir / f"ckpt_last_{run_name}.pth"
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
                str(warmstart_ckpt),
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
    return {
        "name": str(case_name),
        "kind": "directonly_calibration",
        "config": str(cfg_json),
        "ckpt": str(ckpt),
        "eval": str((OUT_ROOT / "eval_model_source" / case_name / "Walk_F_freerun_cycles.json")),
        "direct_source": str(case_name),
        "pose_source": str(case_name),
        "steps_per_epoch": int(steps_per_epoch),
        "lr": lr,
        "weight_decay": weight_decay,
        "warmstart_ckpt": str(warmstart_ckpt),
        "warmstart_case": None if warmstart_case is None else str(warmstart_case),
        "trajectory_order": int(spec.get("trajectory_order", steps_per_epoch)),
        "trainable_log_report": _parse_direct_only_trainable_log(case_name),
    }


def _enrich_case_payload(
    case: Mapping[str, Any],
    *,
    baseline_runtime: Mapping[str, Any],
    coadapt_pose: Mapping[str, Any],
) -> Dict[str, Any]:
    eval_json = _ensure_eval(case)
    metrics = _combined_case_metrics(eval_json)
    direct_gate = _direct_gate(metrics["runtime"], baseline_runtime)
    pose_preservation = _pose_preservation(metrics["pose"], coadapt_pose)
    out = dict(case)
    out["metrics"] = metrics
    out["direct_gate_vs_baseline"] = direct_gate
    out["pose_preservation_vs_coadapt4x"] = pose_preservation
    return out


def _build_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# cp015 tailk7 replace direct recovery bridge",
        "",
        "## Code Facts",
        "",
        f"- direct_pose_feat_source in current coadapt 4x config: `{summary['code_facts']['direct_head_runtime_source']['coadapt4x_config_value']}`",
        f"- train_incremental_replace trainable set: `{summary['code_facts']['train_incremental_replace_trainable_set']['summary']}`",
        f"- direct-only trainable set: `{summary['code_facts']['direct_only_trainable_set']['summary']}`",
        f"- coadapt direct_pose identical across 60/120/180/240 and warmstart: `{summary['code_facts']['ckpt_direct_pose_identity']['coadapt_all_identical_to_warmstart']}`",
        "",
        "## Direct Recovery Cases",
        "",
        "| case | direct source | pose source | swap mode | DirectGeoLocalDeg | delta vs baseline | pose preserved vs coadapt4x? |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for case in summary["cases"]:
        gate = case.get("direct_gate_vs_baseline", {})
        pose_keep = case.get("pose_preservation_vs_coadapt4x", {})
        lines.append(
            f"| {case['name']} | {case['direct_source']} | {case['pose_source']} | {case.get('swap_mode', case['kind'])} | "
            f"{_fmt(gate.get('candidate'))} | {_fmt(gate.get('delta_vs_baseline'))} | "
            f"{str(bool(pose_keep.get('within_rel_tol', False))).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Judgement",
            "",
            f"- baseline direct swap repairs direct gate: {summary['judgements']['baseline_direct_swap_repairs_direct_gate']}",
            f"- baseline direct swap preserves coadapt pose-side primary metrics: {summary['judgements']['baseline_direct_swap_preserves_pose_primary']}",
            f"- reverse swap degrades baseline direct path: {summary['judgements']['reverse_swap_degrades_baseline_direct']}",
            f"- proceed to direct-only calibration: {summary['judgements']['proceed_to_direct_only_calibration']}",
            f"- all direct-only calibration cases executed: {summary['judgements'].get('all_direct_only_calibration_cases_executed', False)}",
            f"- any direct-only calibration clears baseline gate: {summary['judgements'].get('any_direct_only_case_clears_gate', False)}",
            f"- best direct-only candidate: {summary['judgements'].get('best_direct_only_case_name')}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    assert_exists(
        [
            CURRENT_CONTROL_CKPT,
            CURRENT_CONTROL_EVAL,
            BASELINE_REPLACE_CKPT,
            BASELINE_REPLACE_EVAL,
            BASELINE_WARMSTART_CKPT,
            COADAPT_60_CKPT,
            COADAPT_120_CKPT,
            COADAPT_180_CKPT,
            COADAPT_240_CKPT,
            COADAPT_240_EVAL,
            COADAPT_WARMSTART_CKPT,
            COADAPT_240_CONFIG,
            CPU_EXEC,
            ENCODER_BUNDLE,
            AFFINE_STATS,
        ]
    )
    anchor_payloads = [_materialize_anchor_case(case) for case in ANCHORS]
    baseline_metrics = _combined_case_metrics(BASELINE_REPLACE_EVAL)
    coadapt_metrics = _combined_case_metrics(COADAPT_240_EVAL)

    code_facts = {
        "direct_head_runtime_source": {
            "source_refs": ["train/models.py:1570", "train/models.py:3735", "train/models.py:3748"],
            "coadapt4x_config_value": str(load_json(COADAPT_240_CONFIG).get("direct_pose_feat_source")),
            "runtime_default": "cond",
            "summary": "direct head chooses cond unless direct_pose_feat_source switches to hidden/cond+hidden variants",
        },
        "train_incremental_replace_trainable_set": {
            "source_refs": ["train/posttrain.py:4965", "train/posttrain.py:5108", "train/posttrain.py:5204"],
            "summary": "freeze all -> unfreeze motion_head final Linear + optional interface tail prefixes only; direct_pose_* stays frozen",
            "coadapt4x_row_ranges": load_json(COADAPT_240_CONFIG).get("incremental_motion_head_row_ranges"),
            "coadapt4x_interface_mode": load_json(COADAPT_240_CONFIG).get("incremental_interface_mode"),
            "coadapt4x_interface_lr_scale": load_json(COADAPT_240_CONFIG).get("incremental_interface_lr_scale"),
        },
        "ckpt_direct_pose_identity": {
            "coadapt60_vs_120": _direct_pose_cmp(COADAPT_60_CKPT, COADAPT_120_CKPT),
            "coadapt60_vs_180": _direct_pose_cmp(COADAPT_60_CKPT, COADAPT_180_CKPT),
            "coadapt60_vs_240": _direct_pose_cmp(COADAPT_60_CKPT, COADAPT_240_CKPT),
            "coadapt60_vs_warmstart": _direct_pose_cmp(COADAPT_60_CKPT, COADAPT_WARMSTART_CKPT),
            "baseline_vs_warmstart": _direct_pose_cmp(BASELINE_REPLACE_CKPT, BASELINE_WARMSTART_CKPT),
            "control_vs_coadapt240": _direct_pose_cmp(CURRENT_CONTROL_CKPT, COADAPT_240_CKPT),
            "coadapt_all_identical_to_warmstart": bool(
                _direct_pose_cmp(COADAPT_60_CKPT, COADAPT_120_CKPT)["all_common_tensors_identical"]
                and _direct_pose_cmp(COADAPT_60_CKPT, COADAPT_180_CKPT)["all_common_tensors_identical"]
                and _direct_pose_cmp(COADAPT_60_CKPT, COADAPT_240_CKPT)["all_common_tensors_identical"]
                and _direct_pose_cmp(COADAPT_60_CKPT, COADAPT_WARMSTART_CKPT)["all_common_tensors_identical"]
            ),
        },
    }

    swap_cases: List[Dict[str, Any]] = []
    for spec in SWAP_CASES:
        log(f"materializing {spec['name']}")
        case = _copy_direct_pose_tensors(
            name=str(spec["name"]),
            dst_ckpt=Path(spec["dst_ckpt"]),
            src_ckpt=Path(spec["src_ckpt"]),
            pose_source=str(spec["pose_source"]),
            direct_source=str(spec["direct_source"]),
            swap_mode=str(spec["swap_mode"]),
        )
        swap_cases.append(
            _enrich_case_payload(
                case,
                baseline_runtime=baseline_metrics["runtime"],
                coadapt_pose=coadapt_metrics["pose"],
            )
        )
        write_json(
            STATUS_JSON,
            {
                "done_cases": [payload["name"] for payload in swap_cases],
                "total_cases": int(len(SWAP_CASES)),
                "phase": "swap_eval",
            },
        )

    swap_by_name = {case["name"]: case for case in swap_cases}
    baseline_swap = swap_by_name["coadapt_4x_plus_baseline_directpose_swap"]
    reverse_swap = swap_by_name["baseline_plus_coadapt4x_directpose_swap"]
    proceed_direct_only = bool(
        baseline_swap["direct_gate_vs_baseline"]["non_regression"]
        and baseline_swap["pose_preservation_vs_coadapt4x"]["within_rel_tol"]
    )

    calibration_cases: List[Dict[str, Any]] = []
    if proceed_direct_only:
        for spec in DIRECT_ONLY_CASES:
            direct_case = _run_direct_only_calibration(spec)
            direct_case = _enrich_case_payload(
                direct_case,
                baseline_runtime=baseline_metrics["runtime"],
                coadapt_pose=coadapt_metrics["pose"],
            )
            calibration_cases.append(direct_case)

    direct_only_log_reports = {
        case["name"]: case.get("trainable_log_report", _parse_direct_only_trainable_log(str(case["name"])))
        for case in calibration_cases
    }
    representative_direct_only_report = next(
        (
            report
            for report in direct_only_log_reports.values()
            if bool(report.get("found"))
        ),
        None,
    )
    code_facts["direct_only_trainable_set"] = {
        "source_refs": ["train/posttrain.py:4507", "train/posttrain.py:5083", "train/posttrain.py:5135", "train/posttrain.py:7210"],
        "summary": (
            "train_direct_pose lane only unfreezes direct_pose_* modules; "
            f"observed trainable count={representative_direct_only_report.get('trainable_param_count')}"
            if representative_direct_only_report is not None
            else "train_direct_pose lane only unfreezes direct_pose_* modules"
        ),
        "config_flags": {
            "train_direct_pose": True,
            "train_incremental_replace": False,
            "train_lambda_head": False,
            "train_arm_residual": False,
            "train_arm_leg_residual": False,
            "incremental_interface_mode": "off",
            "incremental_interface_lr_scale": 0.0,
        },
        "representative_log_report": representative_direct_only_report,
        "per_case_log_reports": direct_only_log_reports,
        "only_direct_pose_modules_observed": bool(
            representative_direct_only_report is not None
            and representative_direct_only_report.get("all_sample_names_are_direct_pose")
        ),
    }

    cases = [
        _enrich_case_payload(
            case,
            baseline_runtime=baseline_metrics["runtime"],
            coadapt_pose=coadapt_metrics["pose"],
        )
        for case in anchor_payloads
    ]
    cases.extend(swap_cases)
    cases.extend(calibration_cases)

    best_direct_only_case = min(
        calibration_cases,
        key=lambda case: _safe_float(case["metrics"]["runtime"]["direct_geolocaldeg"]),
    ) if calibration_cases else None

    judgements = {
        "baseline_direct_swap_repairs_direct_gate": bool(baseline_swap["direct_gate_vs_baseline"]["non_regression"]),
        "baseline_direct_swap_preserves_pose_primary": bool(
            baseline_swap["pose_preservation_vs_coadapt4x"]["within_rel_tol"]
        ),
        "reverse_swap_degrades_baseline_direct": bool(
            _safe_float(reverse_swap["direct_gate_vs_baseline"]["candidate"])
            > _safe_float(baseline_metrics["runtime"]["direct_geolocaldeg"]) + float(DIRECT_GATE_MARGIN)
        ),
        "proceed_to_direct_only_calibration": bool(proceed_direct_only),
        "all_direct_only_calibration_cases_executed": bool(
            proceed_direct_only and len(calibration_cases) == len(DIRECT_ONLY_CASES)
        ),
        "any_direct_only_case_clears_gate": bool(
            any(case["direct_gate_vs_baseline"]["non_regression"] for case in calibration_cases)
        ),
        "best_direct_only_case_name": None if best_direct_only_case is None else str(best_direct_only_case["name"]),
        "best_direct_only_case_direct_geolocaldeg": (
            float("nan")
            if best_direct_only_case is None
            else _safe_float(best_direct_only_case["metrics"]["runtime"]["direct_geolocaldeg"])
        ),
    }

    summary = {
        "run_date": RUN_DATE,
        "artifacts": {
            "summary_json": str(SUMMARY_JSON),
            "summary_md": str(SUMMARY_MD),
            "log_file": str(LOG_FILE),
        },
        "references": {
            "current_frozen_trunk_replace_control_ckpt": str(CURRENT_CONTROL_CKPT),
            "current_frozen_trunk_replace_control_eval": str(CURRENT_CONTROL_EVAL),
            "baseline_replace_ckpt": str(BASELINE_REPLACE_CKPT),
            "baseline_replace_eval": str(BASELINE_REPLACE_EVAL),
            "coadapt_allrot_interface_bestlr_longer_4x_ckpt": str(COADAPT_240_CKPT),
            "coadapt_allrot_interface_bestlr_longer_4x_eval": str(COADAPT_240_EVAL),
            "coadapt_allrot_interface_bestlr_longer_4x_config": str(COADAPT_240_CONFIG),
            "coadapt_warmstart_ckpt": str(COADAPT_WARMSTART_CKPT),
            "baseline_warmstart_ckpt": str(BASELINE_WARMSTART_CKPT),
        },
        "code_facts": code_facts,
        "experiment_matrix": {
            "anchors": list(anchor_payloads),
            "swap_cases": [
                {
                    "name": str(case["name"]),
                    "pose_source": str(case["pose_source"]),
                    "direct_source": str(case["direct_source"]),
                    "dst_ckpt": str(case["dst_ckpt"]),
                    "src_ckpt": str(case["src_ckpt"]),
                    "swap_mode": str(case["swap_mode"]),
                }
                for case in SWAP_CASES
            ],
            "direct_only_cases": [
                {
                    "name": str(spec["name"]),
                    "steps_per_epoch": int(spec["steps_per_epoch"]),
                    "lr": float(spec.get("lr", 5e-5)),
                    "weight_decay": float(spec.get("weight_decay", 0.0)),
                    "warmstart_ckpt": str(
                        _direct_only_ckpt_path(str(spec["warmstart_case"]))
                        if spec.get("warmstart_case") is not None
                        else COADAPT_240_CKPT
                    ),
                    "warmstart_case": spec.get("warmstart_case"),
                    "trajectory_order": int(spec.get("trajectory_order", spec["steps_per_epoch"])),
                }
                for spec in DIRECT_ONLY_CASES
            ],
            "direct_gate_margin": float(DIRECT_GATE_MARGIN),
            "pose_preserve_rel_tol": float(POSE_PRESERVE_REL_TOL),
        },
        "cases": cases,
        "judgements": judgements,
    }
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(_build_markdown(summary), encoding="utf-8")
    write_json(
        STATUS_JSON,
        {
            "completed": True,
            "done_cases": [case["name"] for case in cases],
            "total_cases": int(len(cases)),
        },
    )
    log(f"wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
