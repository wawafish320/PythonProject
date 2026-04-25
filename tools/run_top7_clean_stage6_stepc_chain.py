#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import torch

try:
    from run_cp015_oldplan_downstream_chain import (
        fmt,
        group_metrics,
        load_json,
        masked_metric_means,
        safe_float,
        window_group_stats,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        fmt,
        group_metrics,
        load_json,
        masked_metric_means,
        safe_float,
        window_group_stats,
        write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "20260412"
PYTHON = Path(sys.executable)

OUT_ROOT = ROOT / "debug_output" / f"_tmp_top7_clean_stage6_stepc_chain_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_top7_clean_stage6_stepc_chain_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DECISION_MD = OUT_ROOT / "decision.md"

TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
PRETRAIN_CLAMP = "1.0"

TOP7_BASETRAIN_CONFIG = (
    ROOT
    / "config"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json"
)
TOP7_BASETRAIN_CKPT = (
    ROOT
    / "models"
    / "cp015_phasecd_tailk_probe_20260331"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401"
    / "ckpt_epoch_014.pth"
)
TOP7_BASETRAIN_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_cp015_tailk7_rankmix_tw020_20260401" / "final_report.json"
TOP7_BASETRAIN_GROUP_SUMMARY = (
    ROOT
    / "models"
    / "cp015_phasecd_tailk_probe_20260331"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401"
    / "basetrain_keybone_group_summary.json"
)
TOP7_BASETRAIN_DECISION_MD = ROOT / "debug_output" / "_tmp_cp015_tailk7_rankmix_tw020_20260401" / "final_report.md"
TOP7_BASETRAIN_SELECTOR_JSON = (
    ROOT / "debug_output" / "_tmp_cp015_tailk7_rankmix_tw020_20260401" / "stage6_exact" / "selector_summary.json"
)

STAGE6_OLDCUT_BASE_CONFIG = (
    ROOT
    / "config"
    / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json"
)

REFERENCE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_top7_posttrain_oldcut_vs_stepc_20260412" / "summary.json"
REFERENCE_SUMMARY_MD = ROOT / "debug_output" / "_tmp_top7_posttrain_oldcut_vs_stepc_20260412" / "summary.md"
REFERENCE_DECISION_MD = ROOT / "debug_output" / "_tmp_top7_posttrain_oldcut_vs_stepc_20260412" / "decision.md"
REFERENCE_PHASE1_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "stepc_chain_verify_summary.json"
REFERENCE_PHASE1_SUMMARY_MD = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "stepc_chain_verify_summary.md"
REFERENCE_PHASE1_CHAIN_MD = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "chain_summary.md"

INCUMBENT_GROUP = (
    Path("/tmp/PythonProject_stage6_baseline_audit_20260411")
    / "debug_output"
    / "_tmp_stage6_A5_probe_raw"
    / "current_bad"
    / "teacher_x_gt"
    / "Walk_F_group_summary.json"
)
STEPB_DECISION = (
    Path("/tmp/PythonProject_stage6_baseline_audit_20260411")
    / "debug_output"
    / "_tmp_stage6_direct_pose_stabilization_v1"
    / "decision.md"
)

STEPA_THRESHOLD_ALL_EX_ROOT_MEAN = 0.3605764035602339
STEPA_THRESHOLD_LEG_P95 = 1.9471660614013672
HARD_REJECT_THRESHOLD_NONLEG_P95 = 1.069789964020848

RUN_NAME_STAGE6_CLEAN = f"lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stepc_clean_{RUN_DATE}"
RUN_NAME_70A_CLEAN = f"WalkF_stage7_70a_from_cp015_tailk7_stage6stepc_clean_{RUN_DATE}"
RUN_NAME_REPLACE_CLEAN = f"WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_cleanstepc_{RUN_DATE}"
RUN_NAME_70R_CLEAN = f"WalkF_stage7_70R_from_cp015_tailk7_replace_cleanstepc_s180_{RUN_DATE}"
RUN_NAME_71_CLEAN = f"WalkF_stage7_71_from_top7_70R_cleanstepc_lr3e4_{RUN_DATE}"
RUN_NAME_72_CLEAN = f"WalkF_stage7_72_from_top7_71_cleanstepc_lr1e4_{RUN_DATE}"
RUN_NAME_LAMBDA_CLEAN = f"WalkF_stage7_lambda_from_top7_72_cleanstepc_{RUN_DATE}"

CFG_STAGE6_CLEAN = CONFIG_ROOT / f"posttrain_stage6_tailfix_top7_clean_stepc_{RUN_DATE}.json"
CFG_70A_CLEAN = CONFIG_ROOT / f"posttrain_70a_from_top7_stage6_clean_stepc_{RUN_DATE}.json"
CFG_REPLACE_CLEAN = CONFIG_ROOT / f"posttrain_replace_from_top7_70a_clean_stepc_{RUN_DATE}.json"
CFG_70R_CLEAN = CONFIG_ROOT / f"posttrain_70R_from_top7_replace_clean_stepc_{RUN_DATE}.json"
CFG_71_CLEAN = CONFIG_ROOT / f"posttrain_71_from_top7_70R_clean_stepc_{RUN_DATE}.json"
CFG_72_CLEAN = CONFIG_ROOT / f"posttrain_72_from_top7_71_clean_stepc_{RUN_DATE}.json"
CFG_LAMBDA_CLEAN = CONFIG_ROOT / f"posttrain_lambda_from_top7_72_clean_stepc_{RUN_DATE}.json"

CKPT_STAGE6_CLEAN = MODEL_ROOT / "stage6_stepc_handoff" / f"ckpt_last_{RUN_NAME_STAGE6_CLEAN}.pth"
CKPT_70A_CLEAN = MODEL_ROOT / "70a_clean" / f"ckpt_last_{RUN_NAME_70A_CLEAN}.pth"
CKPT_REPLACE_WARMSTART_CLEAN = MODEL_ROOT / "warmstart_clean" / f"ckpt_last_cp015_tailk7_70a_replace_zerophase_cleanstepc_{RUN_DATE}.pth"
CKPT_REPLACE_CLEAN = MODEL_ROOT / "replace_clean" / f"ckpt_last_{RUN_NAME_REPLACE_CLEAN}.pth"
CKPT_70R_CLEAN = MODEL_ROOT / "70R_clean" / f"ckpt_last_{RUN_NAME_70R_CLEAN}.pth"
CKPT_71_CLEAN = MODEL_ROOT / "71_clean" / f"ckpt_last_{RUN_NAME_71_CLEAN}.pth"
CKPT_72_CLEAN = MODEL_ROOT / "72_clean" / f"ckpt_last_{RUN_NAME_72_CLEAN}.pth"
CKPT_LAMBDA_CLEAN = MODEL_ROOT / "lambda_clean" / f"ckpt_last_{RUN_NAME_LAMBDA_CLEAN}.pth"

REPORT_WARMSTART_CLEAN = OUT_ROOT / "warmstart_clean" / "replace_zerophase_report.json"
EVAL_STAGE6_CLEAN = OUT_ROOT / "stage6_stepc_handoff" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_STAGE6_CLEAN = OUT_ROOT / "stage6_stepc_handoff" / "eval_model_source_group_summary.json"
EVAL_70A_CLEAN = OUT_ROOT / "70a_clean" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_70A_CLEAN = OUT_ROOT / "70a_clean" / "eval_model_source_group_summary.json"
EVAL_REPLACE_CLEAN = OUT_ROOT / "replace_clean" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_REPLACE_CLEAN = OUT_ROOT / "replace_clean" / "eval_model_source_group_summary.json"
EVAL_70R_CLEAN = OUT_ROOT / "70R_clean" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_70R_CLEAN = OUT_ROOT / "70R_clean" / "eval_model_source_group_summary.json"
EVAL_71_CLEAN = OUT_ROOT / "71_clean" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_71_CLEAN = OUT_ROOT / "71_clean" / "eval_model_source_group_summary.json"
EVAL_72_CLEAN = OUT_ROOT / "72_clean" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_72_CLEAN = OUT_ROOT / "72_clean" / "eval_model_source_group_summary.json"
EVAL_LAMBDA_CLEAN = OUT_ROOT / "lambda_clean" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_LAMBDA_CLEAN = OUT_ROOT / "lambda_clean" / "eval_model_source_group_summary.json"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required file(s):\n" + "\n".join(missing))


def run_cmd(cmd: Sequence[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(str(x) for x in cmd) + "\n")
        fh.flush()
        log("RUN " + " ".join(str(x) for x in cmd))
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        code = int(proc.wait())
        fh.write(f"[exit_code] {code}\n")
        fh.flush()
    if code != 0:
        raise SystemExit(code)


def make_generated_config(base_config: Path, out_json: Path, overrides: Mapping[str, Any]) -> Path:
    payload = load_json(base_config)
    payload.setdefault("load_context", "chain_hop")
    payload.update(dict(overrides))
    write_json(out_json, payload)
    return out_json


def state_and_cfg(ckpt: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    obj = torch.load(ckpt, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"unsupported ckpt payload: {ckpt}")
    state = obj.get("model", obj)
    cfg = obj.get("posttrain_cfg", {})
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported state payload: {ckpt}")
    if not isinstance(cfg, dict):
        cfg = {}
    return dict(state), dict(cfg)


def ckpt_layout(ckpt: Path) -> Dict[str, bool]:
    state, _cfg = state_and_cfg(ckpt)
    return {
        "has_direct_pose_leg_terminal": any(str(key).startswith("direct_pose_leg_terminal.") for key in state),
    }


def assert_stepc_layout(ckpt: Path) -> Dict[str, bool]:
    layout = ckpt_layout(ckpt)
    if not layout["has_direct_pose_leg_terminal"]:
        raise RuntimeError(f"checkpoint is missing canonical split leg terminal: {ckpt} layout={layout}")
    return layout


def assert_oldcut_layout(ckpt: Path) -> Dict[str, bool]:
    raise RuntimeError(f"legacy split leg linear checkpoints are retired: {ckpt}")


def run_posttrain(config_json: Path, ckpt_in: Path, out_dir: Path, run_name: str) -> Path:
    ckpt_out = out_dir / f"ckpt_last_{run_name}.pth"
    if ckpt_out.is_file():
        return ckpt_out
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            str(PYTHON),
            "-m",
            "train.posttrain",
            "--config",
            str(config_json),
            "--ckpt_in",
            str(ckpt_in),
            "--load_context",
            "chain_hop",
            "--out_dir",
            str(out_dir),
            "--run_name",
            run_name,
            "--posttrain_contacts_source",
            "pretrain_contact",
            "--posttrain_contacts_pretrain_clamp",
            PRETRAIN_CLAMP,
            "--encoder_bundle",
            str(ENCODER_BUNDLE),
            "--posttrain_contacts_pretrain_affine_stats",
            str(AFFINE_STATS),
        ]
    )
    return ckpt_out


def run_70r(config_json: Path, out_dir: Path, run_name: str) -> Path:
    ckpt_out = out_dir / f"ckpt_last_{run_name}.pth"
    if ckpt_out.is_file():
        return ckpt_out
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            str(PYTHON),
            str(ROOT / "tools" / "run_posttrain_nonleg_trunk_ablation.py"),
            "--config",
            str(config_json),
            "--trunk-mode",
            "full",
            "--out-dir",
            str(out_dir),
            "--run-name",
            run_name,
            "--epochs",
            "1",
            "--steps-per-epoch",
            "180",
            "--save-step-ckpts",
            "0,1,5,20,60,180",
        ]
    )
    return ckpt_out


def run_eval(model_ckpt: Path, out_dir: Path) -> Path:
    eval_json = out_dir / "Walk_F_freerun_cycles.json"
    if eval_json.is_file():
        return eval_json
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            str(PYTHON),
            "-m",
            "train.validate.run_freerun_cycles",
            "--teacher",
            str(TEACHER),
            "--model",
            str(model_ckpt),
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
            str(out_dir),
            "--force",
        ]
    )
    return eval_json


def ensure_group_summary(eval_json: Path, out_json: Path) -> None:
    if out_json.is_file():
        return
    run_cmd(
        [
            str(PYTHON),
            str(ROOT / "tools" / "phasea_group_summary.py"),
            str(eval_json),
            "--cycle_gte",
            "1",
            "--drop_wrap",
            "--out",
            str(out_json),
        ]
    )


def group_metric(path: Path, group: str, key: str) -> float:
    return safe_float(load_json(path).get("groups", {}).get(group, {}).get(key))


def load_metrics(group_json: Path) -> Dict[str, float]:
    return {
        "all_ex_root_mean": group_metric(group_json, "all_ex_root", "mean"),
        "all_ex_root_p95": group_metric(group_json, "all_ex_root", "p95"),
        "leg_mean": group_metric(group_json, "leg", "mean"),
        "leg_p95": group_metric(group_json, "leg", "p95"),
        "nonleg_mean": group_metric(group_json, "nonleg", "mean"),
        "nonleg_p95": group_metric(group_json, "nonleg", "p95"),
        "arm_mean": group_metric(group_json, "arm", "mean"),
        "arm_p95": group_metric(group_json, "arm", "p95"),
    }


def selected_metrics(eval_json: Path, group_json: Path) -> Dict[str, float]:
    masked = masked_metric_means(eval_json)
    groups = load_metrics(group_json)
    window = window_group_stats(eval_json)
    return {
        "DirectGeoLocalDeg": safe_float(masked.get("DirectGeoLocalDeg")),
        **groups,
        "foot_l_ball_l_SIC12_15": safe_float(window.get("hotspots", {}).get("foot_l_ball_l_SIC12_15")),
        "calf_r_SIC2_4": safe_float(window.get("hotspots", {}).get("calf_r_SIC2_4")),
    }


def collect_eval(eval_json: Path, group_json: Path) -> Dict[str, Any]:
    return {
        "masked_means": masked_metric_means(eval_json),
        "direct_group_summary": group_metrics(group_json),
        "window_summary": window_group_stats(eval_json),
        "selected_metrics": selected_metrics(eval_json, group_json),
        "paths": {
            "eval_json": str(eval_json),
            "group_summary": str(group_json),
        },
    }


def gate_flags(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    all_ex_root_mean = safe_float(metrics.get("all_ex_root_mean"))
    leg_p95 = safe_float(metrics.get("leg_p95"))
    nonleg_p95 = safe_float(metrics.get("nonleg_p95"))
    pass_step_a = bool(
        all_ex_root_mean <= STEPA_THRESHOLD_ALL_EX_ROOT_MEAN
        and leg_p95 <= STEPA_THRESHOLD_LEG_P95
        and nonleg_p95 < HARD_REJECT_THRESHOLD_NONLEG_P95
    )
    hard_reject = bool(nonleg_p95 >= HARD_REJECT_THRESHOLD_NONLEG_P95)
    if hard_reject:
        conclusion = "hard reject vs fixed incumbent current_bad.teacher_x_gt"
    elif pass_step_a:
        conclusion = "passes Step A gate vs fixed incumbent current_bad.teacher_x_gt"
    else:
        conclusion = "fails Step A gate vs fixed incumbent current_bad.teacher_x_gt"
    return {
        "step_a_gate": pass_step_a,
        "hard_reject": hard_reject,
        "relative_fixed_incumbent": conclusion,
    }


def metric_delta(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, float]:
    keys = set(cur.keys()) | set(ref.keys())
    return {key: safe_float(cur.get(key)) - safe_float(ref.get(key)) for key in sorted(keys)}


def compare_stepb(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, Any]:
    d_mean = safe_float(cur.get("all_ex_root_mean")) - safe_float(ref.get("all_ex_root_mean"))
    d_p95 = safe_float(cur.get("all_ex_root_p95")) - safe_float(ref.get("all_ex_root_p95"))
    d_leg = safe_float(cur.get("leg_mean")) - safe_float(ref.get("leg_mean"))
    hard_reject = bool(safe_float(cur.get("nonleg_p95")) >= HARD_REJECT_THRESHOLD_NONLEG_P95)
    primary_triggered = False
    tie_break1_triggered = False
    tie_break2_triggered = False
    if hard_reject:
        verdict = "lose_hard_reject"
        rationale = "hard_reject=fixed_incumbent_nonleg_p95"
        trigger = "hard_reject"
    elif abs(d_mean) >= 0.002:
        primary_triggered = True
        verdict = "win" if d_mean < 0.0 else "lose"
        rationale = "primary=all_ex_root_mean"
        trigger = "primary"
    elif abs(d_p95) >= 0.01:
        tie_break1_triggered = True
        verdict = "win" if d_p95 < 0.0 else "lose"
        rationale = "tie_break1=all_ex_root_p95"
        trigger = "tie_break1"
    else:
        tie_break2_triggered = True
        verdict = "win" if d_leg < 0.0 else ("tie" if abs(d_leg) < 1e-12 else "lose")
        rationale = "tie_break2=leg_mean"
        trigger = "tie_break2"
    return {
        "verdict": verdict,
        "rationale": rationale,
        "trigger": trigger,
        "primary_triggered": primary_triggered,
        "tie_break1_triggered": tie_break1_triggered,
        "tie_break2_triggered": tie_break2_triggered,
        "hard_reject_triggered": hard_reject,
        "delta": {
            "all_ex_root_mean": d_mean,
            "all_ex_root_p95": d_p95,
            "leg_mean": d_leg,
            "leg_p95": safe_float(cur.get("leg_p95")) - safe_float(ref.get("leg_p95")),
            "nonleg_p95": safe_float(cur.get("nonleg_p95")) - safe_float(ref.get("nonleg_p95")),
        },
    }


def stage_record(
    *,
    stage: str,
    lane: str,
    config: Path,
    ckpt: Path,
    eval_json: Path,
    group_json: Path,
    input_artifact: Path | None = None,
    launch_command: str | None = None,
    base_config: Path | None = None,
    lr_override: float | None = None,
) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "lane": lane,
        "config": str(config),
        "output_ckpt": str(ckpt),
        "eval_artifact": str(eval_json),
        "group_summary": str(group_json),
        "metrics": load_metrics(group_json),
        "eval": collect_eval(eval_json, group_json),
        "layout": ckpt_layout(ckpt),
    }
    if input_artifact is not None:
        row["input_artifact"] = str(input_artifact)
    if launch_command is not None:
        row["launch_command"] = launch_command
    if base_config is not None:
        row["base_config"] = str(base_config)
    if lr_override is not None:
        row["locked_lr_override"] = float(lr_override)
    row["gate"] = gate_flags(row["metrics"])
    return row


def reference_stage(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(row))
    required = [Path(str(out["config"])), Path(str(out["output_ckpt"])), Path(str(out["eval_artifact"])), Path(str(out["group_summary"]))]
    if out.get("input_artifact"):
        required.append(Path(str(out["input_artifact"])))
    assert_exists(required)
    lane = str(out.get("lane", ""))
    ckpt = Path(str(out["output_ckpt"]))
    if lane == "old-cut":
        out["layout"] = assert_oldcut_layout(ckpt)
    elif "StepC" in lane or "clean" in lane.lower():
        out["layout"] = assert_stepc_layout(ckpt)
    return out


def load_reference_artifacts() -> Dict[str, Any]:
    summary = load_json(REFERENCE_SUMMARY_JSON)
    phase1_stages = summary.get("phase1", {}).get("stages", {})
    phase2_payload = summary.get("phase2", {})
    phase2_stages = phase2_payload.get("stages", {}) if isinstance(phase2_payload, dict) else {}
    refs = {
        "summary_json": str(REFERENCE_SUMMARY_JSON),
        "summary_md": str(REFERENCE_SUMMARY_MD),
        "decision_md": str(REFERENCE_DECISION_MD),
        "phase1_summary_json": str(REFERENCE_PHASE1_SUMMARY_JSON),
        "phase1_summary_md": str(REFERENCE_PHASE1_SUMMARY_MD),
        "phase1_chain_md": str(REFERENCE_PHASE1_CHAIN_MD),
        "top7_summary": summary,
        "phase1": {
            "O_70a": reference_stage(phase1_stages["70a_oldcut"]),
            "P_70a": reference_stage(phase1_stages["70a_stepc"]),
            "O_replace": reference_stage(phase1_stages["replace_oldcut"]),
            "P_replace": reference_stage(phase1_stages["replace_stepc"]),
            "O_70R": reference_stage(phase1_stages["70R_oldcut"]),
            "P_70R": reference_stage(phase1_stages["70R_stepc"]),
        },
        "phase2_available": bool(phase2_payload.get("executed")),
        "phase2": {},
    }
    if refs["phase2_available"]:
        refs["phase2"] = {
            "O_71": reference_stage(phase2_stages["71_oldcut"]),
            "P_71": reference_stage(phase2_stages["71_stepc"]),
            "O_72": reference_stage(phase2_stages["72_oldcut"]),
            "P_72": reference_stage(phase2_stages["72_stepc"]),
            "O_lambda": reference_stage(phase2_stages["lambda_oldcut"]),
            "P_lambda": reference_stage(phase2_stages["lambda_stepc"]),
        }
    return refs


def phase_compare(candidate_row: Mapping[str, Any], ref_row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "delta": metric_delta(candidate_row["metrics"], ref_row["metrics"]),
        "stepb_verdict": compare_stepb(candidate_row["metrics"], ref_row["metrics"]),
    }


def copy_replace_zerophase_warmstart(src_ckpt: Path, dst_ckpt: Path, report_json: Path) -> Dict[str, Any]:
    if dst_ckpt.is_file() and report_json.is_file():
        return load_json(report_json)
    obj = torch.load(src_ckpt, map_location="cpu")
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError(f"unexpected checkpoint format: {src_ckpt}")
    out_obj = dict(obj)
    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, dst_ckpt)
    report = {
        "source_ckpt": str(src_ckpt),
        "output_ckpt": str(dst_ckpt),
        "copied_without_phase_z_direct_adaptation": True,
    }
    write_json(report_json, report)
    return report


def build_launch_posttrain(config_json: Path, ckpt_in: Path, out_dir: Path, run_name: str) -> str:
    return (
        f"PYTHONPATH=. {PYTHON} -m train.posttrain "
        f"--config {config_json} "
        f"--ckpt_in {ckpt_in} "
        f"--out_dir {out_dir} "
        f"--run_name {run_name} "
        "--posttrain_contacts_source pretrain_contact "
        f"--posttrain_contacts_pretrain_clamp {PRETRAIN_CLAMP} "
        f"--encoder_bundle {ENCODER_BUNDLE} "
        f"--posttrain_contacts_pretrain_affine_stats {AFFINE_STATS}"
    )


def build_launch_70r(config_json: Path, out_dir: Path, run_name: str) -> str:
    return (
        f"PYTHONPATH=. {PYTHON} {ROOT / 'tools' / 'run_posttrain_nonleg_trunk_ablation.py'} "
        f"--config {config_json} "
        "--trunk-mode full "
        f"--out-dir {out_dir} "
        f"--run-name {run_name} "
        "--epochs 1 "
        "--steps-per-epoch 180 "
        "--save-step-ckpts 0,1,5,20,60,180"
    )


def run_locked_posttrain_stage(
    *,
    stage: str,
    lane: str,
    base_config: Path,
    out_config: Path,
    ckpt_in: Path,
    out_dir: Path,
    run_name: str,
    stepc_enabled: bool,
    eval_json: Path,
    group_json: Path,
    lr_override: float | None = None,
    extra_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "ckpt_in": str(ckpt_in),
        "out_dir": str(out_dir),
        "run_name": run_name,
        "encoder_bundle": str(ENCODER_BUNDLE),
        "posttrain_contacts_source": "pretrain_contact",
        "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
        "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
    }
    if lr_override is not None:
        overrides["lr"] = float(lr_override)
    if extra_overrides:
        overrides.update(dict(extra_overrides))
    cfg_json = make_generated_config(base_config, out_config, overrides)
    ckpt_out = run_posttrain(cfg_json, ckpt_in, out_dir, run_name)
    layout = assert_stepc_layout(ckpt_out) if stepc_enabled else assert_oldcut_layout(ckpt_out)
    eval_out = run_eval(ckpt_out, eval_json.parent)
    ensure_group_summary(eval_out, group_json)
    row = stage_record(
        stage=stage,
        lane=lane,
        config=cfg_json,
        ckpt=ckpt_out,
        eval_json=eval_out,
        group_json=group_json,
        input_artifact=ckpt_in,
        launch_command=build_launch_posttrain(cfg_json, ckpt_in, out_dir, run_name),
        base_config=base_config,
        lr_override=lr_override,
    )
    row["layout"] = layout
    return row


def run_locked_70r_stage(
    *,
    stage: str,
    lane: str,
    base_config: Path,
    out_config: Path,
    ckpt_in: Path,
    out_dir: Path,
    run_name: str,
    stepc_enabled: bool,
    eval_json: Path,
    group_json: Path,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "ckpt_in": str(ckpt_in),
        "out_dir": str(out_dir),
        "run_name": run_name,
        "encoder_bundle": str(ENCODER_BUNDLE),
        "posttrain_contacts_source": "pretrain_contact",
        "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
        "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
    }
    cfg_json = make_generated_config(base_config, out_config, overrides)
    ckpt_out = run_70r(cfg_json, out_dir, run_name)
    layout = assert_stepc_layout(ckpt_out) if stepc_enabled else assert_oldcut_layout(ckpt_out)
    eval_out = run_eval(ckpt_out, eval_json.parent)
    ensure_group_summary(eval_out, group_json)
    row = stage_record(
        stage=stage,
        lane=lane,
        config=cfg_json,
        ckpt=ckpt_out,
        eval_json=eval_out,
        group_json=group_json,
        input_artifact=ckpt_in,
        launch_command=build_launch_70r(cfg_json, out_dir, run_name),
        base_config=base_config,
        lr_override=None,
    )
    row["layout"] = layout
    return row


def make_phase0(reference_summary: Mapping[str, Any]) -> Dict[str, Any]:
    row = run_locked_posttrain_stage(
        stage="stage6",
        lane="clean-StepC handoff",
        base_config=STAGE6_OLDCUT_BASE_CONFIG,
        out_config=CFG_STAGE6_CLEAN,
        ckpt_in=TOP7_BASETRAIN_CKPT,
        out_dir=MODEL_ROOT / "stage6_stepc_handoff",
        run_name=RUN_NAME_STAGE6_CLEAN,
        stepc_enabled=True,
        eval_json=EVAL_STAGE6_CLEAN,
        group_json=GROUP_STAGE6_CLEAN,
        lr_override=None,
    )
    row["provenance"] = {
        "source_donor_ckpt": str(TOP7_BASETRAIN_CKPT),
        "source_donor_config": str(TOP7_BASETRAIN_CONFIG),
        "source_donor_summary_json": str(TOP7_BASETRAIN_SUMMARY_JSON),
        "source_donor_group_summary": str(TOP7_BASETRAIN_GROUP_SUMMARY),
        "source_donor_decision_md": str(TOP7_BASETRAIN_DECISION_MD),
        "source_donor_selector_json": str(TOP7_BASETRAIN_SELECTOR_JSON),
        "current_old_stage6_handoff_ckpt": str(reference_summary["top7_canonical_donor"]["current_downstream_stage6_ckpt"]),
        "current_old_stage6_handoff_group_summary": str(reference_summary["top7_canonical_donor"]["current_downstream_stage6_group_summary"]),
        "reference_old_vs_pseudo_summary_json": str(REFERENCE_SUMMARY_JSON),
        "reference_old_vs_pseudo_decision_md": str(REFERENCE_DECISION_MD),
        "stage6_recipe_base_config": str(STAGE6_OLDCUT_BASE_CONFIG),
        "same_canonical_top7_donor": True,
    }
    return {
        "stage6_clean_stepc_handoff": row,
        "layout_check": dict(row["layout"]),
    }


def make_phase1(phase0: Mapping[str, Any], refs: Mapping[str, Any]) -> Dict[str, Any]:
    ref1 = refs["phase1"]
    stage6_ckpt = Path(str(phase0["stage6_clean_stepc_handoff"]["output_ckpt"]))
    stage70a = run_locked_posttrain_stage(
        stage="70a",
        lane="clean-StepC",
        base_config=Path(str(ref1["O_70a"]["config"])),
        out_config=CFG_70A_CLEAN,
        ckpt_in=stage6_ckpt,
        out_dir=MODEL_ROOT / "70a_clean",
        run_name=RUN_NAME_70A_CLEAN,
        stepc_enabled=True,
        eval_json=EVAL_70A_CLEAN,
        group_json=GROUP_70A_CLEAN,
    )
    warmstart_report = copy_replace_zerophase_warmstart(Path(str(stage70a["output_ckpt"])), CKPT_REPLACE_WARMSTART_CLEAN, REPORT_WARMSTART_CLEAN)
    stage70a["warmstart_output_ckpt"] = str(CKPT_REPLACE_WARMSTART_CLEAN)
    stage70a["warmstart_report"] = str(REPORT_WARMSTART_CLEAN)

    stage_replace = run_locked_posttrain_stage(
        stage="replace",
        lane="clean-StepC",
        base_config=Path(str(ref1["O_replace"]["config"])),
        out_config=CFG_REPLACE_CLEAN,
        ckpt_in=CKPT_REPLACE_WARMSTART_CLEAN,
        out_dir=MODEL_ROOT / "replace_clean",
        run_name=RUN_NAME_REPLACE_CLEAN,
        stepc_enabled=True,
        eval_json=EVAL_REPLACE_CLEAN,
        group_json=GROUP_REPLACE_CLEAN,
    )
    stage_replace["warmstart_report"] = str(REPORT_WARMSTART_CLEAN)
    stage_replace["warmstart_report_payload"] = warmstart_report

    stage70r = run_locked_70r_stage(
        stage="70R",
        lane="clean-StepC",
        base_config=Path(str(ref1["O_70R"]["config"])),
        out_config=CFG_70R_CLEAN,
        ckpt_in=Path(str(stage_replace["output_ckpt"])),
        out_dir=MODEL_ROOT / "70R_clean",
        run_name=RUN_NAME_70R_CLEAN,
        stepc_enabled=True,
        eval_json=EVAL_70R_CLEAN,
        group_json=GROUP_70R_CLEAN,
    )

    stages = {
        "O_70a": ref1["O_70a"],
        "P_70a": ref1["P_70a"],
        "C_70a": stage70a,
        "O_replace": ref1["O_replace"],
        "P_replace": ref1["P_replace"],
        "C_replace": stage_replace,
        "O_70R": ref1["O_70R"],
        "P_70R": ref1["P_70R"],
        "C_70R": stage70r,
    }
    comparisons = {
        "C_vs_O_70a": phase_compare(stages["C_70a"], stages["O_70a"]),
        "C_vs_P_70a": phase_compare(stages["C_70a"], stages["P_70a"]),
        "C_vs_O_replace": phase_compare(stages["C_replace"], stages["O_replace"]),
        "C_vs_P_replace": phase_compare(stages["C_replace"], stages["P_replace"]),
        "C_vs_O_70R": phase_compare(stages["C_70R"], stages["O_70R"]),
        "C_vs_P_70R": phase_compare(stages["C_70R"], stages["P_70R"]),
    }
    verdict_o_70r = comparisons["C_vs_O_70R"]["stepb_verdict"]
    verdict_p_70r = comparisons["C_vs_P_70R"]["stepb_verdict"]
    early_improve_vs_p = any(
        str(comparisons[key]["stepb_verdict"]["verdict"]) == "win"
        for key in ("C_vs_P_70a", "C_vs_P_replace")
    )
    proceed_to_phase2 = (
        str(verdict_o_70r["verdict"]) == "win"
        and safe_float(verdict_o_70r["delta"]["all_ex_root_mean"]) < -0.01
        and str(verdict_p_70r["verdict"]) in {"win", "tie"}
    )
    proceed_reason = (
        "C-70R clearly beats O-70R and is not weaker than P-70R; continue locked Phase 2."
        if proceed_to_phase2
        else "Phase 1 does not yet justify locked Phase 2; stop at 70a/replace/70R."
    )
    return {
        "stages": stages,
        "comparisons": comparisons,
        "proceed_to_phase2": proceed_to_phase2,
        "proceed_reason": proceed_reason,
        "early_improve_vs_pseudo_stepc": early_improve_vs_p,
    }


def _phase2_base_and_lr(refs: Mapping[str, Any], key: str, fallback: Path, fallback_lr: float | None) -> tuple[Path, float | None]:
    row = refs.get(key, {})
    base = Path(str(row.get("base_config", fallback)))
    lr = row.get("locked_lr_override", fallback_lr)
    return base, (None if lr is None else float(lr))


def make_phase2(phase1: Mapping[str, Any], refs: Mapping[str, Any]) -> Dict[str, Any]:
    if not bool(refs.get("phase2_available")):
        raise RuntimeError("reference Lane O/P Phase 2 artifacts are missing")
    ref2 = refs["phase2"]
    base71, lr71 = _phase2_base_and_lr(
        ref2,
        "P_71",
        ROOT / "config" / "posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json",
        3e-4,
    )
    base72, lr72 = _phase2_base_and_lr(
        ref2,
        "P_72",
        ROOT / "config" / "posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json",
        1e-4,
    )
    base_lambda, _ = _phase2_base_and_lr(
        ref2,
        "P_lambda",
        ROOT / "config" / "posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json",
        None,
    )
    stage71 = run_locked_posttrain_stage(
        stage="71",
        lane="clean-StepC",
        base_config=base71,
        out_config=CFG_71_CLEAN,
        ckpt_in=Path(str(phase1["stages"]["C_70R"]["output_ckpt"])),
        out_dir=MODEL_ROOT / "71_clean",
        run_name=RUN_NAME_71_CLEAN,
        stepc_enabled=True,
        eval_json=EVAL_71_CLEAN,
        group_json=GROUP_71_CLEAN,
        lr_override=lr71,
    )
    stage72 = run_locked_posttrain_stage(
        stage="72",
        lane="clean-StepC",
        base_config=base72,
        out_config=CFG_72_CLEAN,
        ckpt_in=Path(str(stage71["output_ckpt"])),
        out_dir=MODEL_ROOT / "72_clean",
        run_name=RUN_NAME_72_CLEAN,
        stepc_enabled=True,
        eval_json=EVAL_72_CLEAN,
        group_json=GROUP_72_CLEAN,
        lr_override=lr72,
    )
    stage_lambda = run_locked_posttrain_stage(
        stage="lambda",
        lane="clean-StepC",
        base_config=base_lambda,
        out_config=CFG_LAMBDA_CLEAN,
        ckpt_in=Path(str(stage72["output_ckpt"])),
        out_dir=MODEL_ROOT / "lambda_clean",
        run_name=RUN_NAME_LAMBDA_CLEAN,
        stepc_enabled=True,
        eval_json=EVAL_LAMBDA_CLEAN,
        group_json=GROUP_LAMBDA_CLEAN,
        lr_override=None,
    )
    stages = {
        "O_71": ref2["O_71"],
        "P_71": ref2["P_71"],
        "C_71": stage71,
        "O_72": ref2["O_72"],
        "P_72": ref2["P_72"],
        "C_72": stage72,
        "O_lambda": ref2["O_lambda"],
        "P_lambda": ref2["P_lambda"],
        "C_lambda": stage_lambda,
    }
    comparisons = {
        "C_vs_O_71": phase_compare(stages["C_71"], stages["O_71"]),
        "C_vs_P_71": phase_compare(stages["C_71"], stages["P_71"]),
        "C_vs_O_72": phase_compare(stages["C_72"], stages["O_72"]),
        "C_vs_P_72": phase_compare(stages["C_72"], stages["P_72"]),
        "C_vs_O_lambda": phase_compare(stages["C_lambda"], stages["O_lambda"]),
        "C_vs_P_lambda": phase_compare(stages["C_lambda"], stages["P_lambda"]),
    }
    return {
        "executed": True,
        "stages": stages,
        "comparisons": comparisons,
    }


def classify_vs_pseudo_early(phase1: Mapping[str, Any]) -> str:
    verdict_70a = phase1["comparisons"]["C_vs_P_70a"]["stepb_verdict"]
    verdict_replace = phase1["comparisons"]["C_vs_P_replace"]["stepb_verdict"]
    clear_hits = 0
    partial_hits = 0
    for verdict in (verdict_70a, verdict_replace):
        d_mean = safe_float(verdict["delta"]["all_ex_root_mean"])
        if str(verdict["verdict"]) == "win" and d_mean <= -0.01:
            clear_hits += 1
        elif str(verdict["verdict"]) == "win" or d_mean <= -0.002:
            partial_hits += 1
    if clear_hits >= 1 and (clear_hits + partial_hits) >= 2:
        return "clear"
    if clear_hits >= 1 or partial_hits >= 1:
        return "partial"
    return "almost_none"


def build_answers(phase1: Mapping[str, Any], phase2: Mapping[str, Any] | None) -> Dict[str, Any]:
    early_vs_p = classify_vs_pseudo_early(phase1)
    verdict_c_vs_o_70r = phase1["comparisons"]["C_vs_O_70R"]["stepb_verdict"]
    verdict_c_vs_p_70r = phase1["comparisons"]["C_vs_P_70R"]["stepb_verdict"]
    p2_executed = bool(phase2 and phase2.get("executed"))

    if early_vs_p == "clear" and str(verdict_c_vs_p_70r["verdict"]) in {"win", "tie"}:
        preferred = "old-stage6-handoff/downstream-boundary-induced"
        rescue_degree = "明显 rescue"
        top7_phrase = (
            "top7 不是简单的“太 aggressive”；更准确地说，它超出了 legacy stage6 handoff / old boundary contract 在 early downstream 能干净吸收的范围。"
            "当 stage6 本身改成 clean StepC handoff 后，70a/replace 的早期拖累就明显缩小。"
        )
        top3_phrase = (
            "top3 更像是“旧 stage6 handoff + old boundary 仍能 handle 的 donor 范围”，而不是天然最优 tail scope。"
        )
        worth_more = not p2_executed
        next_step = (
            "如果还没跑 full chain，只补一个 locked `71->72->lambda` clean-StepC continuation；不要扩成 sweep。"
            if worth_more
            else "不需要再扩；clean stage6-StepC 的主因果链已经足够。"
        )
    elif str(verdict_c_vs_o_70r["verdict"]) == "win" and str(verdict_c_vs_p_70r["verdict"]) in {"win", "tie"}:
        preferred = "two-layer interaction"
        rescue_degree = "部分 rescue" if early_vs_p != "almost_none" else "几乎没有"
        top7_phrase = (
            "top7 不是一句“太 aggressive”就能概括；更准确地说，它既带有 donor 自身的 early downstream 负担，"
            "又被旧 stage6 handoff / boundary fragmentation 进一步放大。clean StepC handoff 能减轻拖累，但不能在 70a/replace 彻底抹掉它。"
        )
        top3_phrase = (
            "top3 更像是一个“双重更安全”的范围：一方面 donor 本身负担更轻，另一方面旧 boundary 也还勉强吸得住。"
        )
        worth_more = not p2_executed and str(verdict_c_vs_p_70r["verdict"]) in {"win", "tie"}
        next_step = (
            "如果需要更完整确认，只补一个 locked clean-StepC `71->72->lambda` continuation；不要再做 sweep。"
            if worth_more
            else "先停在这里；当前最稳妥的表述仍是 two-layer interaction。"
        )
    else:
        preferred = "basetrain-compromise-dominant"
        rescue_degree = "几乎没有"
        top7_phrase = (
            "top7 更准确的说法不是“太 aggressive”，而是 donor 本身已经带着会穿过 downstream cleanup 的 compromise；"
            "旧 handoff 可能有放大作用，但不是主因。"
        )
        top3_phrase = (
            "top3 更像是真正更轻的 donor regime，而不只是旧 boundary 恰好还能 handle 的范围。"
        )
        worth_more = False
        next_step = (
            "不要扩成大链或 sweep；如果还要做一个最小 follow-up，只做单个 locked clean-StepC `71` continuation。"
        )

    answer1 = {
        "clear": "yes_clear",
        "partial": "yes_partial",
        "almost_none": "no_material",
    }[early_vs_p]
    if rescue_degree == "明显 rescue":
        early_statement = "clean stage6-StepC 相比 pseudo-StepC，已经在 70a/replace 提供了明显额外 rescue。"
    elif rescue_degree == "部分 rescue":
        early_statement = "clean stage6-StepC 相比 pseudo-StepC，只提供了有限额外 rescue；主改善仍更靠近 70R。"
    else:
        early_statement = "clean stage6-StepC 相比 pseudo-StepC，几乎没有额外 rescue；70a/replace 仍基本不干净。"

    return {
        "clean_stage6_vs_pseudo_stepc_70a_replace": answer1,
        "clean_stage6_vs_pseudo_stepc_70a_replace_statement": early_statement,
        "rescue_degree": rescue_degree,
        "preferred_explanation": preferred,
        "top7_precise_statement": top7_phrase,
        "top3_precise_statement": top3_phrase,
        "worth_more_full_chain_validation": worth_more,
        "minimal_next_step": next_step,
    }


def build_summary(phase0: Mapping[str, Any], phase1: Mapping[str, Any], phase2: Mapping[str, Any] | None, refs: Mapping[str, Any]) -> Dict[str, Any]:
    fixed_incumbent_metrics = load_metrics(INCUMBENT_GROUP)
    return {
        "run_date": RUN_DATE,
        "caveat": "N=5 / limited-N",
        "script": str(ROOT / "tools" / "run_top7_clean_stage6_stepc_chain.py"),
        "canonical_top7_donor": {
            "basetrain_config": str(TOP7_BASETRAIN_CONFIG),
            "basetrain_ckpt": str(TOP7_BASETRAIN_CKPT),
            "basetrain_summary_json": str(TOP7_BASETRAIN_SUMMARY_JSON),
            "basetrain_group_summary": str(TOP7_BASETRAIN_GROUP_SUMMARY),
            "basetrain_decision_md": str(TOP7_BASETRAIN_DECISION_MD),
            "basetrain_selector_json": str(TOP7_BASETRAIN_SELECTOR_JSON),
        },
        "reference_lanes": {
            "lane_o": "old stage6 handoff -> old-cut downstream",
            "lane_p": "old stage6 handoff -> downstream StepC compatibility",
            "summary_json": refs["summary_json"],
            "summary_md": refs["summary_md"],
            "decision_md": refs["decision_md"],
            "phase1_summary_json": refs["phase1_summary_json"],
            "phase1_summary_md": refs["phase1_summary_md"],
            "phase1_chain_md": refs["phase1_chain_md"],
        },
        "locked_policy": {
            "step_a_gate": "necessary-but-not-sufficient",
            "step_b_prime": {
                "primary": "all_ex_root_mean",
                "tie_break1": "all_ex_root_p95 if abs(delta_all_ex_root_mean) < 0.002°",
                "tie_break2": "leg_mean if abs(delta_all_ex_root_p95) < 0.01°",
                "hard_reject": "nonleg_p95 regression >= fixed incumbent threshold",
            },
            "incumbent": "current_bad.teacher_x_gt",
            "reference_decision": str(STEPB_DECISION),
        },
        "fixed_incumbent": {
            "group_summary": str(INCUMBENT_GROUP),
            "metrics": fixed_incumbent_metrics,
            "thresholds": {
                "step_a_all_ex_root_mean": STEPA_THRESHOLD_ALL_EX_ROOT_MEAN,
                "step_a_leg_p95": STEPA_THRESHOLD_LEG_P95,
                "hard_reject_nonleg_p95": HARD_REJECT_THRESHOLD_NONLEG_P95,
            },
        },
        "phase0": phase0,
        "phase1": phase1,
        "phase2": phase2 if phase2 is not None else {"executed": False},
        "answers": build_answers(phase1, phase2),
    }


def _append_stage_table(lines: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    lines.extend(
        [
            "| stage | lane | input artifact | ckpt | config | eval json | group summary |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['lane']} | `{row.get('input_artifact', '')}` | `{row['output_ckpt']}` | `{row['config']}` | `{row['eval_artifact']}` | `{row['group_summary']}` |"
        )


def _append_compare_block(lines: list[str], title: str, ref_row: Mapping[str, Any], cand_row: Mapping[str, Any], verdict: Mapping[str, Any]) -> None:
    delta = verdict["delta"]
    lines.extend(
        [
            f"### {title}",
            "",
            f"- verdict: `{verdict['verdict']}`",
            f"- trigger: `{verdict['trigger']}`",
            f"- primary_triggered: `{str(bool(verdict['primary_triggered'])).lower()}`",
            f"- tie_break1_triggered: `{str(bool(verdict['tie_break1_triggered'])).lower()}`",
            f"- tie_break2_triggered: `{str(bool(verdict['tie_break2_triggered'])).lower()}`",
            f"- hard_reject_triggered: `{str(bool(verdict['hard_reject_triggered'])).lower()}`",
            "",
            "| lane | all_ex_root_mean | all_ex_root_p95 | leg_mean | leg_p95 | nonleg_p95 | arm_mean | arm_p95 | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 | Step A | hard reject |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in (ref_row, cand_row):
        metrics = row["metrics"]
        sel = row["eval"]["selected_metrics"]
        gate = row["gate"]
        lines.append(
            f"| {row['lane']} | {fmt(metrics['all_ex_root_mean'])} | {fmt(metrics['all_ex_root_p95'])} | {fmt(metrics['leg_mean'])} | {fmt(metrics['leg_p95'])} | {fmt(metrics['nonleg_p95'])} | {fmt(metrics['arm_mean'])} | {fmt(metrics['arm_p95'])} | {fmt(sel['foot_l_ball_l_SIC12_15'])} | {fmt(sel['calf_r_SIC2_4'])} | {'pass' if gate['step_a_gate'] else 'fail'} | {'yes' if gate['hard_reject'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"- delta(all_ex_root_mean/all_ex_root_p95/leg_mean/leg_p95/nonleg_p95): `{fmt(delta['all_ex_root_mean'])}, {fmt(delta['all_ex_root_p95'])}, {fmt(delta['leg_mean'])}, {fmt(delta['leg_p95'])}, {fmt(delta['nonleg_p95'])}`",
            "",
        ]
    )


def _append_verdict_table(lines: list[str], title: str, items: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
    lines.extend(
        [
            f"### {title}",
            "",
            "| compare | verdict | trigger | primary | tie-break1 | tie-break2 | hard reject |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for label, verdict in items:
        lines.append(
            f"| {label} | `{verdict['verdict']}` | `{verdict['trigger']}` | "
            f"`{str(bool(verdict['primary_triggered'])).lower()}` | "
            f"`{str(bool(verdict['tie_break1_triggered'])).lower()}` | "
            f"`{str(bool(verdict['tie_break2_triggered'])).lower()}` | "
            f"`{str(bool(verdict['hard_reject_triggered'])).lower()}` |"
        )
    lines.append("")


def build_summary_md(summary: Mapping[str, Any]) -> str:
    phase1 = summary["phase1"]
    phase2 = summary["phase2"]
    lines = [
        "# Top7 clean stage6-StepC causality summary",
        "",
        f"- run_date: `{summary['run_date']}`",
        f"- caveat: `{summary['caveat']}`",
        f"- script: `{summary['script']}`",
        f"- donor ckpt: `{summary['canonical_top7_donor']['basetrain_ckpt']}`",
        f"- clean stage6-StepC handoff ckpt: `{summary['phase0']['stage6_clean_stepc_handoff']['output_ckpt']}`",
        "",
        "## Phase 1",
        "",
        "| compare | delta all_ex_root_mean | delta all_ex_root_p95 | delta leg_mean | delta leg_p95 | delta nonleg_p95 | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, key in (
        ("C-70a vs O-70a", "C_vs_O_70a"),
        ("C-70a vs P-70a", "C_vs_P_70a"),
        ("C-replace vs O-replace", "C_vs_O_replace"),
        ("C-replace vs P-replace", "C_vs_P_replace"),
        ("C-70R vs O-70R", "C_vs_O_70R"),
        ("C-70R vs P-70R", "C_vs_P_70R"),
    ):
        verdict = phase1["comparisons"][key]["stepb_verdict"]
        delta = verdict["delta"]
        lines.append(
            f"| {label} | {fmt(delta['all_ex_root_mean'])} | {fmt(delta['all_ex_root_p95'])} | {fmt(delta['leg_mean'])} | {fmt(delta['leg_p95'])} | {fmt(delta['nonleg_p95'])} | {verdict['verdict']} |"
        )
    lines.extend(
        [
            "",
            f"- proceed_to_phase2: `{str(bool(phase1['proceed_to_phase2'])).lower()}`",
            f"- reason: `{phase1['proceed_reason']}`",
            "",
        ]
    )
    if bool(phase2.get("executed")):
        lines.extend(
            [
                "## Phase 2",
                "",
                "| compare | delta all_ex_root_mean | delta all_ex_root_p95 | delta leg_mean | delta leg_p95 | delta nonleg_p95 | verdict |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for label, key in (
            ("C-71 vs O-71", "C_vs_O_71"),
            ("C-71 vs P-71", "C_vs_P_71"),
            ("C-72 vs O-72", "C_vs_O_72"),
            ("C-72 vs P-72", "C_vs_P_72"),
            ("C-lambda vs O-lambda", "C_vs_O_lambda"),
            ("C-lambda vs P-lambda", "C_vs_P_lambda"),
        ):
            verdict = phase2["comparisons"][key]["stepb_verdict"]
            delta = verdict["delta"]
            lines.append(
                f"| {label} | {fmt(delta['all_ex_root_mean'])} | {fmt(delta['all_ex_root_p95'])} | {fmt(delta['leg_mean'])} | {fmt(delta['leg_p95'])} | {fmt(delta['nonleg_p95'])} | {verdict['verdict']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Answers",
            "",
            f"- clean stage6-StepC vs pseudo-StepC at 70a/replace: `{summary['answers']['clean_stage6_vs_pseudo_stepc_70a_replace']}`",
            f"- rescue degree: `{summary['answers']['rescue_degree']}`",
            f"- preferred explanation: `{summary['answers']['preferred_explanation']}`",
            f"- worth more full-chain validation: `{str(bool(summary['answers']['worth_more_full_chain_validation'])).lower()}`",
            f"- minimal next step: `{summary['answers']['minimal_next_step']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_decision_md(summary: Mapping[str, Any]) -> str:
    phase0 = summary["phase0"]
    phase1 = summary["phase1"]
    phase2 = summary["phase2"]
    answers = summary["answers"]
    p0 = phase0["stage6_clean_stepc_handoff"]
    p1 = phase1["stages"]
    lines: list[str] = [
        "# Top7 clean stage6-StepC causality decision",
        "",
        f"- run_date: `{summary['run_date']}`",
        f"- caveat: `{summary['caveat']}`",
        f"- script: `{summary['script']}`",
        "",
        "## Canonical donor",
        "",
        f"- basetrain ckpt: `{summary['canonical_top7_donor']['basetrain_ckpt']}`",
        f"- basetrain config: `{summary['canonical_top7_donor']['basetrain_config']}`",
        f"- basetrain summary json: `{summary['canonical_top7_donor']['basetrain_summary_json']}`",
        f"- basetrain group summary: `{summary['canonical_top7_donor']['basetrain_group_summary']}`",
        f"- basetrain decision md: `{summary['canonical_top7_donor']['basetrain_decision_md']}`",
        f"- basetrain selector json: `{summary['canonical_top7_donor']['basetrain_selector_json']}`",
        "",
        "## Reference lanes",
        "",
        "- Lane O: `old stage6 handoff -> old-cut downstream`",
        "- Lane P: `old stage6 handoff -> downstream StepC compatibility`",
        f"- reference summary json: `{summary['reference_lanes']['summary_json']}`",
        f"- reference summary md: `{summary['reference_lanes']['summary_md']}`",
        f"- reference decision md: `{summary['reference_lanes']['decision_md']}`",
        "",
        "## A. 是否成功接入",
        "",
        "### Phase 0 — clean stage6-StepC handoff",
        "",
        f"- base config: `{p0['base_config']}`",
        f"- input donor ckpt: `{p0['input_artifact']}`",
        f"- clean stage6-StepC ckpt: `{p0['output_ckpt']}`",
        f"- clean stage6-StepC config: `{p0['config']}`",
        f"- clean stage6-StepC eval json: `{p0['eval_artifact']}`",
        f"- clean stage6-StepC group summary: `{p0['group_summary']}`",
        f"- provenance: `{p0['provenance']}`",
        f"- layout check: `{phase0['layout_check']}`",
        "",
        "### Phase 1 artifacts",
        "",
    ]
    _append_stage_table(
        lines,
        (
            p1["O_70a"],
            p1["P_70a"],
            p1["C_70a"],
            p1["O_replace"],
            p1["P_replace"],
            p1["C_replace"],
            p1["O_70R"],
            p1["P_70R"],
            p1["C_70R"],
        ),
    )
    if bool(phase2.get("executed")):
        p2 = phase2["stages"]
        lines.extend(["", "### Phase 2 artifacts", ""])
        _append_stage_table(
            lines,
            (
                p2["O_71"],
                p2["P_71"],
                p2["C_71"],
                p2["O_72"],
                p2["P_72"],
                p2["C_72"],
                p2["O_lambda"],
                p2["P_lambda"],
                p2["C_lambda"],
            ),
        )
    lines.extend(["", "## B. 对比表", ""])
    _append_compare_block(lines, "Phase 1 — C-70a vs O-70a", p1["O_70a"], p1["C_70a"], phase1["comparisons"]["C_vs_O_70a"]["stepb_verdict"])
    _append_compare_block(lines, "Phase 1 — C-replace vs O-replace", p1["O_replace"], p1["C_replace"], phase1["comparisons"]["C_vs_O_replace"]["stepb_verdict"])
    _append_compare_block(lines, "Phase 1 — C-70R vs O-70R", p1["O_70R"], p1["C_70R"], phase1["comparisons"]["C_vs_O_70R"]["stepb_verdict"])
    _append_compare_block(lines, "Phase 1 — C-70a vs P-70a", p1["P_70a"], p1["C_70a"], phase1["comparisons"]["C_vs_P_70a"]["stepb_verdict"])
    _append_compare_block(lines, "Phase 1 — C-replace vs P-replace", p1["P_replace"], p1["C_replace"], phase1["comparisons"]["C_vs_P_replace"]["stepb_verdict"])
    _append_compare_block(lines, "Phase 1 — C-70R vs P-70R", p1["P_70R"], p1["C_70R"], phase1["comparisons"]["C_vs_P_70R"]["stepb_verdict"])
    if bool(phase2.get("executed")):
        p2 = phase2["stages"]
        _append_compare_block(lines, "Phase 2 — C-71 vs O-71", p2["O_71"], p2["C_71"], phase2["comparisons"]["C_vs_O_71"]["stepb_verdict"])
        _append_compare_block(lines, "Phase 2 — C-72 vs O-72", p2["O_72"], p2["C_72"], phase2["comparisons"]["C_vs_O_72"]["stepb_verdict"])
        _append_compare_block(lines, "Phase 2 — C-lambda vs O-lambda", p2["O_lambda"], p2["C_lambda"], phase2["comparisons"]["C_vs_O_lambda"]["stepb_verdict"])
        _append_compare_block(lines, "Phase 2 — C-71 vs P-71", p2["P_71"], p2["C_71"], phase2["comparisons"]["C_vs_P_71"]["stepb_verdict"])
        _append_compare_block(lines, "Phase 2 — C-72 vs P-72", p2["P_72"], p2["C_72"], phase2["comparisons"]["C_vs_P_72"]["stepb_verdict"])
        _append_compare_block(lines, "Phase 2 — C-lambda vs P-lambda", p2["P_lambda"], p2["C_lambda"], phase2["comparisons"]["C_vs_P_lambda"]["stepb_verdict"])
    lines.extend(["## C. Step B' verdict", ""])
    _append_verdict_table(
        lines,
        "Phase 1",
        (
            ("C-70a vs O-70a", phase1["comparisons"]["C_vs_O_70a"]["stepb_verdict"]),
            ("C-replace vs O-replace", phase1["comparisons"]["C_vs_O_replace"]["stepb_verdict"]),
            ("C-70R vs O-70R", phase1["comparisons"]["C_vs_O_70R"]["stepb_verdict"]),
            ("C-70a vs P-70a", phase1["comparisons"]["C_vs_P_70a"]["stepb_verdict"]),
            ("C-replace vs P-replace", phase1["comparisons"]["C_vs_P_replace"]["stepb_verdict"]),
            ("C-70R vs P-70R", phase1["comparisons"]["C_vs_P_70R"]["stepb_verdict"]),
        ),
    )
    if bool(phase2.get("executed")):
        _append_verdict_table(
            lines,
            "Phase 2",
            (
                ("C-71 vs O-71", phase2["comparisons"]["C_vs_O_71"]["stepb_verdict"]),
                ("C-72 vs O-72", phase2["comparisons"]["C_vs_O_72"]["stepb_verdict"]),
                ("C-lambda vs O-lambda", phase2["comparisons"]["C_vs_O_lambda"]["stepb_verdict"]),
                ("C-71 vs P-71", phase2["comparisons"]["C_vs_P_71"]["stepb_verdict"]),
                ("C-72 vs P-72", phase2["comparisons"]["C_vs_P_72"]["stepb_verdict"]),
                ("C-lambda vs P-lambda", phase2["comparisons"]["C_vs_P_lambda"]["stepb_verdict"]),
            ),
        )
    lines.extend(
        [
            "## D. 因果解释结论",
            "",
            f"1. clean `stage6-StepC` 是否比当前 pseudo-StepC lane 更进一步 rescue `70a/replace`？ `{answers['clean_stage6_vs_pseudo_stepc_70a_replace']}`",
            f"2. 如果有 rescue，程度是：`{answers['rescue_degree']}`",
            f"3. 相比上一轮的 `two-layer interaction`，这轮更支持：`{answers['preferred_explanation']}`",
            f"4. 对 `top7` 的更精确表述：`{answers['top7_precise_statement']}`",
            f"5. 对 `top3` 的更精确表述：`{answers['top3_precise_statement']}`",
            f"6. 是否值得继续做更完整验证：`{str(bool(answers['worth_more_full_chain_validation'])).lower()}`；最小下一步：`{answers['minimal_next_step']}`",
            "",
            f"- 补充判断：`{answers['clean_stage6_vs_pseudo_stepc_70a_replace_statement']}`",
            "",
            "## E. 改动清单",
            "",
            "- `tools/run_top7_clean_stage6_stepc_chain.py`: 最小 focused runner；只负责生成/定位 clean `stage6-StepC` handoff、复用 Lane O/P、运行 Lane C、做 model-source eval、生成 comparison / decision。",
            f"- `{SUMMARY_JSON}`: 本轮 machine-readable 汇总。",
            f"- `{SUMMARY_MD}`: 本轮简表。",
            f"- `{DECISION_MD}`: 本轮因果判断正文。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    assert_exists(
        [
            TEACHER,
            ENCODER_BUNDLE,
            AFFINE_STATS,
            TOP7_BASETRAIN_CONFIG,
            TOP7_BASETRAIN_CKPT,
            TOP7_BASETRAIN_SUMMARY_JSON,
            TOP7_BASETRAIN_GROUP_SUMMARY,
            TOP7_BASETRAIN_DECISION_MD,
            TOP7_BASETRAIN_SELECTOR_JSON,
            STAGE6_OLDCUT_BASE_CONFIG,
            REFERENCE_SUMMARY_JSON,
            REFERENCE_SUMMARY_MD,
            REFERENCE_DECISION_MD,
            REFERENCE_PHASE1_SUMMARY_JSON,
            REFERENCE_PHASE1_SUMMARY_MD,
            REFERENCE_PHASE1_CHAIN_MD,
            INCUMBENT_GROUP,
            STEPB_DECISION,
        ]
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    log("loading existing Lane O/P references")
    refs = load_reference_artifacts()
    log("building clean stage6-StepC handoff")
    phase0 = make_phase0(refs["top7_summary"])
    log("running clean lane through 70a/replace/70R")
    phase1 = make_phase1(phase0, refs)
    phase2: Dict[str, Any] | None = None
    if bool(phase1["proceed_to_phase2"]):
        log("Phase 1 is strong enough; continuing locked 71/72/lambda")
        phase2 = make_phase2(phase1, refs)
    else:
        log("Phase 1 is not clean enough; stopping before full continuation")

    summary = build_summary(phase0, phase1, phase2, refs)
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(build_summary_md(summary), encoding="utf-8")
    DECISION_MD.write_text(build_decision_md(summary), encoding="utf-8")
    log(f"wrote summary_json={SUMMARY_JSON}")
    log(f"wrote summary_md={SUMMARY_MD}")
    log(f"wrote decision_md={DECISION_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
