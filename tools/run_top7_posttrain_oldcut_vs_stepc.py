#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

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

OUT_ROOT = ROOT / "debug_output" / f"_tmp_top7_posttrain_oldcut_vs_stepc_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_top7_posttrain_oldcut_vs_stepc_{RUN_DATE}"
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
TOP7_BASETRAIN_SELECTOR_JSON = (
    ROOT / "debug_output" / "_tmp_cp015_tailk7_rankmix_tw020_20260401" / "stage6_exact" / "selector_summary.json"
)
TOP7_BASETRAIN_GROUP_SUMMARY = (
    ROOT
    / "models"
    / "cp015_phasecd_tailk_probe_20260331"
    / "exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401"
    / "basetrain_keybone_group_summary.json"
)
TOP7_BASETRAIN_DECISION_MD = ROOT / "debug_output" / "_tmp_cp015_tailk7_rankmix_tw020_20260401" / "final_report.md"

TOP7_STAGE6_SELECTOR_JSON = ROOT / "debug_output" / "_tmp_cp015_tailk7_stage6_tailfix_20260401" / "selector_summary.json"
TOP7_STAGE6_OLDCUT_CKPT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_stage6_tailfix_20260401"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.pth"
)
TOP7_STAGE6_OLDCUT_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_stage6_tailfix_20260401"
    / "lr3e4_e8x60_wd1e4_reinit1"
    / "stage6_group_summary.json"
)

PHASE1_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "stepc_chain_verify_summary.json"
PHASE1_SUMMARY_MD = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "stepc_chain_verify_summary.md"
PHASE1_CHAIN_MD = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "chain_summary.md"

CFG_70A_OLDCUT = ROOT / "debug_output" / "_tmp_ep014center_70a_lowlr_sweep_20260328" / "configs" / "posttrain_70a_lr3e4_from_ep014center_20260328.json"
CKPT_70A_OLDCUT = ROOT / "models" / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402" / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth"
EVAL_70A_OLDCUT = ROOT / "debug_output" / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_70A_OLDCUT = ROOT / "debug_output" / "_tmp_cp015_tailk7_stage70a_from_tailfix_20260402" / "eval_model_source_group_summary.json"

CFG_70A_STEPC = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "configs" / "posttrain_70a_lr3e4_from_cp015_tailk7_stage6tailfix_stepc_20260412.json"
CKPT_70A_STEPC = ROOT / "models" / "__tmp_stage6_stepc_chain_verify_20260412" / "70a_stepc" / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_stepc_20260412.pth"
EVAL_70A_STEPC = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "70a_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_70A_STEPC = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "70a_stepc" / "eval_model_source_group_summary.json"

CFG_REPLACE_OLDCUT = (
    ROOT
    / "debug_output"
    / "_tmp_cp015_tailk7_replace_schedule_ablation_20260402"
    / "configs"
    / "posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.json"
)
CKPT_REPLACE_OLDCUT = (
    ROOT
    / "models"
    / "__tmp_cp015_tailk7_replace_schedule_ablation_20260402"
    / "e3x60"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_20260402.pth"
)
EVAL_REPLACE_OLDCUT = ROOT / "debug_output" / "_tmp_cp015_tailk7_replace_schedule_ablation_20260402" / "eval_model_source" / "e3x60" / "Walk_F_freerun_cycles.json"
GROUP_REPLACE_OLDCUT = ROOT / "debug_output" / "_tmp_cp015_tailk7_replace_schedule_ablation_20260402" / "eval_model_source" / "e3x60_group_summary.json"

CFG_REPLACE_STEPC = (
    ROOT
    / "debug_output"
    / "_tmp_stage6_stepc_chain_verify_20260412"
    / "configs"
    / "posttrain_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_stepc_20260412.json"
)
CKPT_REPLACE_STEPC = (
    ROOT
    / "models"
    / "__tmp_stage6_stepc_chain_verify_20260412"
    / "replace_stepc"
    / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_cp015_tailk7_70a_stepc_20260412.pth"
)
EVAL_REPLACE_STEPC = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "replace_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_REPLACE_STEPC = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "replace_stepc" / "eval_model_source_group_summary.json"

CFG_70R_OLDCUT = (
    ROOT
    / "debug_output"
    / "_tmp_stage6_stepc_chain_verify_20260412"
    / "configs"
    / "posttrain_70R_from_cp015_tailk7_replace_e3x60_control_20260412.json"
)
CKPT_70R_OLDCUT = (
    ROOT
    / "models"
    / "__tmp_stage6_stepc_chain_verify_20260412"
    / "70R_control"
    / "ckpt_last_WalkF_stage7_70R_from_cp015_tailk7_replace_e3x60_control_s180_20260412.pth"
)
EVAL_70R_OLDCUT = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "70R_control" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_70R_OLDCUT = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "70R_control" / "eval_model_source_group_summary.json"

CFG_70R_STEPC = (
    ROOT
    / "debug_output"
    / "_tmp_stage6_stepc_chain_verify_20260412"
    / "configs"
    / "posttrain_70R_from_cp015_tailk7_replace_e3x60_stepc_20260412.json"
)
CKPT_70R_STEPC = (
    ROOT
    / "models"
    / "__tmp_stage6_stepc_chain_verify_20260412"
    / "70R_stepc"
    / "ckpt_last_WalkF_stage7_70R_from_cp015_tailk7_replace_e3x60_stepc_s180_20260412.pth"
)
EVAL_70R_STEPC = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "70R_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_70R_STEPC = ROOT / "debug_output" / "_tmp_stage6_stepc_chain_verify_20260412" / "70R_stepc" / "eval_model_source_group_summary.json"

BASE_CONFIG_71 = ROOT / "config" / "posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json"
BASE_CONFIG_72 = ROOT / "config" / "posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json"
BASE_CONFIG_LAMBDA = ROOT / "config" / "posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json"

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

RUN_NAME_71_OLDCUT = f"WalkF_stage7_71_from_top7_70R_oldcut_lr3e4_{RUN_DATE}"
RUN_NAME_71_STEPC = f"WalkF_stage7_71_from_top7_70R_stepc_lr3e4_{RUN_DATE}"
RUN_NAME_72_OLDCUT = f"WalkF_stage7_72_from_top7_71_oldcut_lr1e4_{RUN_DATE}"
RUN_NAME_72_STEPC = f"WalkF_stage7_72_from_top7_71_stepc_lr1e4_{RUN_DATE}"
RUN_NAME_LAMBDA_OLDCUT = f"WalkF_stage7_lambda_from_top7_72_oldcut_{RUN_DATE}"
RUN_NAME_LAMBDA_STEPC = f"WalkF_stage7_lambda_from_top7_72_stepc_{RUN_DATE}"

CFG_71_OLDCUT = CONFIG_ROOT / f"posttrain_71_from_top7_70R_oldcut_lr3e4_{RUN_DATE}.json"
CFG_71_STEPC = CONFIG_ROOT / f"posttrain_71_from_top7_70R_stepc_lr3e4_{RUN_DATE}.json"
CFG_72_OLDCUT = CONFIG_ROOT / f"posttrain_72_from_top7_71_oldcut_lr1e4_{RUN_DATE}.json"
CFG_72_STEPC = CONFIG_ROOT / f"posttrain_72_from_top7_71_stepc_lr1e4_{RUN_DATE}.json"
CFG_LAMBDA_OLDCUT = CONFIG_ROOT / f"posttrain_lambda_from_top7_72_oldcut_{RUN_DATE}.json"
CFG_LAMBDA_STEPC = CONFIG_ROOT / f"posttrain_lambda_from_top7_72_stepc_{RUN_DATE}.json"

CKPT_71_OLDCUT = MODEL_ROOT / "71_oldcut" / f"ckpt_last_{RUN_NAME_71_OLDCUT}.pth"
CKPT_71_STEPC = MODEL_ROOT / "71_stepc" / f"ckpt_last_{RUN_NAME_71_STEPC}.pth"
CKPT_72_OLDCUT = MODEL_ROOT / "72_oldcut" / f"ckpt_last_{RUN_NAME_72_OLDCUT}.pth"
CKPT_72_STEPC = MODEL_ROOT / "72_stepc" / f"ckpt_last_{RUN_NAME_72_STEPC}.pth"
CKPT_LAMBDA_OLDCUT = MODEL_ROOT / "lambda_oldcut" / f"ckpt_last_{RUN_NAME_LAMBDA_OLDCUT}.pth"
CKPT_LAMBDA_STEPC = MODEL_ROOT / "lambda_stepc" / f"ckpt_last_{RUN_NAME_LAMBDA_STEPC}.pth"

EVAL_71_OLDCUT = OUT_ROOT / "71_oldcut" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_71_OLDCUT = OUT_ROOT / "71_oldcut" / "eval_model_source_group_summary.json"
EVAL_71_STEPC = OUT_ROOT / "71_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_71_STEPC = OUT_ROOT / "71_stepc" / "eval_model_source_group_summary.json"
EVAL_72_OLDCUT = OUT_ROOT / "72_oldcut" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_72_OLDCUT = OUT_ROOT / "72_oldcut" / "eval_model_source_group_summary.json"
EVAL_72_STEPC = OUT_ROOT / "72_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_72_STEPC = OUT_ROOT / "72_stepc" / "eval_model_source_group_summary.json"
EVAL_LAMBDA_OLDCUT = OUT_ROOT / "lambda_oldcut" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_LAMBDA_OLDCUT = OUT_ROOT / "lambda_oldcut" / "eval_model_source_group_summary.json"
EVAL_LAMBDA_STEPC = OUT_ROOT / "lambda_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_LAMBDA_STEPC = OUT_ROOT / "lambda_stepc" / "eval_model_source_group_summary.json"


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
    record = {
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
        record["input_artifact"] = str(input_artifact)
    if launch_command is not None:
        record["launch_command"] = launch_command
    if base_config is not None:
        record["base_config"] = str(base_config)
    if lr_override is not None:
        record["locked_lr_override"] = float(lr_override)
    record["gate"] = gate_flags(record["metrics"])
    return record


def reference_stage(*, stage: str, lane: str, config: Path, ckpt: Path, eval_json: Path, group_json: Path, input_artifact: Path | None = None) -> Dict[str, Any]:
    return stage_record(
        stage=stage,
        lane=lane,
        config=config,
        ckpt=ckpt,
        eval_json=eval_json,
        group_json=group_json,
        input_artifact=input_artifact,
    )


def run_locked_stage(
    *,
    stage: str,
    lane: str,
    base_config: Path,
    out_config: Path,
    ckpt_in: Path,
    out_dir: Path,
    run_name: str,
    lr_override: float | None,
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
    if lr_override is not None:
        overrides["lr"] = float(lr_override)
    cfg_json = make_generated_config(base_config, out_config, overrides)
    ckpt_out = run_posttrain(cfg_json, ckpt_in, out_dir, run_name)
    layout = assert_stepc_layout(ckpt_out) if stepc_enabled else assert_oldcut_layout(ckpt_out)
    eval_out = run_eval(ckpt_out, eval_json.parent)
    ensure_group_summary(eval_out, group_json)
    launch_cmd = (
        f"PYTHONPATH=. {PYTHON} -m train.posttrain "
        f"--config {cfg_json} "
        f"--ckpt_in {ckpt_in} "
        f"--out_dir {out_dir} "
        f"--run_name {run_name} "
        "--posttrain_contacts_source pretrain_contact "
        f"--posttrain_contacts_pretrain_clamp {PRETRAIN_CLAMP} "
        f"--encoder_bundle {ENCODER_BUNDLE} "
        f"--posttrain_contacts_pretrain_affine_stats {AFFINE_STATS}"
    )
    row = stage_record(
        stage=stage,
        lane=lane,
        config=cfg_json,
        ckpt=ckpt_out,
        eval_json=eval_out,
        group_json=group_json,
        input_artifact=ckpt_in,
        launch_command=launch_cmd,
        base_config=base_config,
        lr_override=lr_override,
    )
    row["layout"] = layout
    return row


def phase_compare(stepc_row: Mapping[str, Any], oldcut_row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "delta": metric_delta(stepc_row["metrics"], oldcut_row["metrics"]),
        "stepb_verdict": compare_stepb(stepc_row["metrics"], oldcut_row["metrics"]),
    }


def build_phase1() -> Dict[str, Any]:
    stages = {
        "70a_oldcut": reference_stage(
            stage="70a",
            lane="old-cut",
            config=CFG_70A_OLDCUT,
            ckpt=CKPT_70A_OLDCUT,
            eval_json=EVAL_70A_OLDCUT,
            group_json=GROUP_70A_OLDCUT,
            input_artifact=TOP7_STAGE6_OLDCUT_CKPT,
        ),
        "70a_stepc": reference_stage(
            stage="70a",
            lane="StepC",
            config=CFG_70A_STEPC,
            ckpt=CKPT_70A_STEPC,
            eval_json=EVAL_70A_STEPC,
            group_json=GROUP_70A_STEPC,
            input_artifact=TOP7_STAGE6_OLDCUT_CKPT,
        ),
        "replace_oldcut": reference_stage(
            stage="replace",
            lane="old-cut",
            config=CFG_REPLACE_OLDCUT,
            ckpt=CKPT_REPLACE_OLDCUT,
            eval_json=EVAL_REPLACE_OLDCUT,
            group_json=GROUP_REPLACE_OLDCUT,
            input_artifact=CKPT_70A_OLDCUT,
        ),
        "replace_stepc": reference_stage(
            stage="replace",
            lane="StepC",
            config=CFG_REPLACE_STEPC,
            ckpt=CKPT_REPLACE_STEPC,
            eval_json=EVAL_REPLACE_STEPC,
            group_json=GROUP_REPLACE_STEPC,
            input_artifact=CKPT_70A_STEPC,
        ),
        "70R_oldcut": reference_stage(
            stage="70R",
            lane="old-cut",
            config=CFG_70R_OLDCUT,
            ckpt=CKPT_70R_OLDCUT,
            eval_json=EVAL_70R_OLDCUT,
            group_json=GROUP_70R_OLDCUT,
            input_artifact=CKPT_REPLACE_OLDCUT,
        ),
        "70R_stepc": reference_stage(
            stage="70R",
            lane="StepC",
            config=CFG_70R_STEPC,
            ckpt=CKPT_70R_STEPC,
            eval_json=EVAL_70R_STEPC,
            group_json=GROUP_70R_STEPC,
            input_artifact=CKPT_REPLACE_STEPC,
        ),
    }
    comparisons = {
        "70a_stepc_vs_oldcut_70a": phase_compare(stages["70a_stepc"], stages["70a_oldcut"]),
        "replace_stepc_vs_oldcut_replace": phase_compare(stages["replace_stepc"], stages["replace_oldcut"]),
        "70R_stepc_vs_oldcut_70R": phase_compare(stages["70R_stepc"], stages["70R_oldcut"]),
    }
    verdict_70r = comparisons["70R_stepc_vs_oldcut_70R"]["stepb_verdict"]
    proceed_to_phase2 = str(verdict_70r["verdict"]) == "win" and safe_float(verdict_70r["delta"]["all_ex_root_mean"]) < -0.01
    proceed_reason = (
        "70R-StepC clearly beats top7 old-cut 70R on primary and core tails; continue locked Phase 2."
        if proceed_to_phase2
        else "Phase 1 does not show a clear 70R-StepC rescue; stop before full downstream continuation."
    )
    return {
        "source_summary_json": str(PHASE1_SUMMARY_JSON),
        "source_summary_md": str(PHASE1_SUMMARY_MD),
        "source_chain_md": str(PHASE1_CHAIN_MD),
        "stages": stages,
        "comparisons": comparisons,
        "proceed_to_phase2": proceed_to_phase2,
        "proceed_reason": proceed_reason,
    }


def classify_rescue(phase1: Mapping[str, Any], phase2: Mapping[str, Any] | None) -> str:
    p1_70r_win = str(phase1["comparisons"]["70R_stepc_vs_oldcut_70R"]["stepb_verdict"]["verdict"]) == "win"
    p1_replace_win = str(phase1["comparisons"]["replace_stepc_vs_oldcut_replace"]["stepb_verdict"]["verdict"]) == "win"
    p1_70a_win = str(phase1["comparisons"]["70a_stepc_vs_oldcut_70a"]["stepb_verdict"]["verdict"]) == "win"
    phase2_wins = []
    if phase2 and bool(phase2.get("executed")):
        for key in ("71_stepc_vs_oldcut_71", "72_stepc_vs_oldcut_72", "lambda_stepc_vs_oldcut_lambda"):
            phase2_wins.append(str(phase2["comparisons"][key]["stepb_verdict"]["verdict"]) == "win")
    if p1_70r_win and p1_replace_win and p1_70a_win and (not phase2_wins or all(phase2_wins)):
        return "complete_rescue"
    if p1_70r_win and (not phase2_wins or any(phase2_wins) or all(phase2_wins)):
        return "partial_rescue"
    return "almost_no_rescue"


def build_answers(phase1: Mapping[str, Any], phase2: Mapping[str, Any] | None) -> Dict[str, Any]:
    rescue_state = classify_rescue(phase1, phase2)
    if rescue_state == "complete_rescue":
        explanation = "downstream-boundary-induced"
        top3_read = (
            "yes_strongly: top3 looks more like the old boundary's handleable range than an inherently optimal tail scope."
        )
        next_step = (
            "No new sweep. This round already serves as the minimal top7 + StepC full-chain validation."
        )
    elif rescue_state == "partial_rescue":
        explanation = "two-layer interaction"
        top3_read = (
            "yes_partially: the evidence now fits 'top3 is what the old boundary could still absorb', not 'top3 is naturally optimal', but early-stage recipe/contract mismatch still exists."
        )
        next_step = (
            "Worth only one minimal follow-up: a single locked 70a-StepC retune from the same top7 stage6 handoff to test whether the early 70a drag is recipe-boundary mismatch rather than donor fatality."
        )
    else:
        explanation = "basetrain-compromise-dominant"
        top3_read = (
            "not_supported_yet: current evidence still leaves top3 looking safer for reasons beyond the old downstream boundary."
        )
        next_step = (
            "Do not expand to a sweep. If you want one more check, do a single locked 70a-StepC retry; otherwise keep the causal claim conservative."
        )

    phase2_completed = bool(phase2 and phase2.get("executed"))
    return {
        "stepc_rescues_regression": rescue_state != "almost_no_rescue",
        "rescue_extent": rescue_state,
        "preferred_explanation": explanation,
        "top3_interpretation": top3_read,
        "worth_additional_full_chain_validation": not phase2_completed,
        "minimal_next_step": next_step,
    }


def build_summary(phase1: Mapping[str, Any], phase2: Mapping[str, Any] | None) -> Dict[str, Any]:
    fixed_incumbent_metrics = load_metrics(INCUMBENT_GROUP)
    answers = build_answers(phase1, phase2)
    return {
        "run_date": RUN_DATE,
        "caveat": "N=5 / limited-N",
        "script": str(ROOT / "tools" / "run_top7_posttrain_oldcut_vs_stepc.py"),
        "top7_canonical_donor": {
            "basetrain_config": str(TOP7_BASETRAIN_CONFIG),
            "basetrain_ckpt": str(TOP7_BASETRAIN_CKPT),
            "basetrain_summary_json": str(TOP7_BASETRAIN_SUMMARY_JSON),
            "basetrain_selector_json": str(TOP7_BASETRAIN_SELECTOR_JSON),
            "basetrain_group_summary": str(TOP7_BASETRAIN_GROUP_SUMMARY),
            "basetrain_decision_md": str(TOP7_BASETRAIN_DECISION_MD),
            "stage6_selector_json": str(TOP7_STAGE6_SELECTOR_JSON),
            "current_downstream_stage6_ckpt": str(TOP7_STAGE6_OLDCUT_CKPT),
            "current_downstream_stage6_group_summary": str(TOP7_STAGE6_OLDCUT_GROUP),
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
        "phase1": phase1,
        "phase2": phase2 if phase2 is not None else {"executed": False},
        "answers": answers,
    }


def build_summary_md(summary: Mapping[str, Any]) -> str:
    phase1 = summary["phase1"]
    phase2 = summary["phase2"]
    lines = [
        "# Top7 old-cut vs StepC downstream causality test",
        "",
        f"- run_date: `{summary['run_date']}`",
        f"- caveat: `{summary['caveat']}`",
        f"- script: `{summary['script']}`",
        f"- top7 donor ckpt: `{summary['top7_canonical_donor']['basetrain_ckpt']}`",
        "",
        "## Phase 1",
        "",
        "| compare | delta all_ex_root_mean | delta all_ex_root_p95 | delta leg_mean | delta leg_p95 | delta nonleg_p95 | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, key in (
        ("top7 StepC 70a vs old-cut 70a", "70a_stepc_vs_oldcut_70a"),
        ("top7 StepC replace vs old-cut replace", "replace_stepc_vs_oldcut_replace"),
        ("top7 StepC 70R vs old-cut 70R", "70R_stepc_vs_oldcut_70R"),
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
            ("top7 StepC 71 vs old-cut 71", "71_stepc_vs_oldcut_71"),
            ("top7 StepC 72 vs old-cut 72", "72_stepc_vs_oldcut_72"),
            ("top7 StepC lambda vs old-cut lambda", "lambda_stepc_vs_oldcut_lambda"),
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
            f"- StepC rescues top7 downstream regression: `{str(bool(summary['answers']['stepc_rescues_regression'])).lower()}`",
            f"- rescue extent: `{summary['answers']['rescue_extent']}`",
            f"- preferred explanation: `{summary['answers']['preferred_explanation']}`",
            f"- top3 interpretation: `{summary['answers']['top3_interpretation']}`",
            f"- minimal next step: `{summary['answers']['minimal_next_step']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _append_artifact_rows(lines: list[str], items: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
    lines.extend(
        [
            "| stage | lane | ckpt | config | eval json | group summary |",
            "|---|---|---|---|---|---|",
        ]
    )
    for label, row in items:
        lines.append(
            f"| {label} | {row['lane']} | `{row['output_ckpt']}` | `{row['config']}` | `{row['eval_artifact']}` | `{row['group_summary']}` |"
        )


def _append_compare_block(lines: list[str], title: str, verdict: Mapping[str, Any], oldcut_row: Mapping[str, Any], stepc_row: Mapping[str, Any]) -> None:
    delta = verdict["delta"]
    lines.extend(
        [
            f"### {title}",
            "",
            f"- verdict: `{verdict['verdict']}`",
            f"- primary_triggered: `{str(verdict['primary_triggered']).lower()}`",
            f"- tie_break1_triggered: `{str(verdict['tie_break1_triggered']).lower()}`",
            f"- tie_break2_triggered: `{str(verdict['tie_break2_triggered']).lower()}`",
            f"- hard_reject_triggered: `{str(verdict['hard_reject_triggered']).lower()}`",
            "",
            "| lane | all_ex_root_mean | all_ex_root_p95 | leg_mean | leg_p95 | nonleg_p95 | arm_mean | arm_p95 | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 | Step A | hard reject |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in (oldcut_row, stepc_row):
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


def build_decision_md(summary: Mapping[str, Any]) -> str:
    phase1 = summary["phase1"]
    phase2 = summary["phase2"]
    answers = summary["answers"]
    p1 = phase1["stages"]
    lines: list[str] = [
        "# Top7 downstream old-cut vs StepC decision",
        "",
        f"- run_date: `{summary['run_date']}`",
        f"- caveat: `{summary['caveat']}`",
        f"- script: `{summary['script']}`",
        "",
        "## Canonical donor",
        "",
        f"- basetrain ckpt: `{summary['top7_canonical_donor']['basetrain_ckpt']}`",
        f"- basetrain config: `{summary['top7_canonical_donor']['basetrain_config']}`",
        f"- basetrain summary json: `{summary['top7_canonical_donor']['basetrain_summary_json']}`",
        f"- basetrain group summary: `{summary['top7_canonical_donor']['basetrain_group_summary']}`",
        f"- basetrain decision md: `{summary['top7_canonical_donor']['basetrain_decision_md']}`",
        f"- current downstream stage6 handoff ckpt: `{summary['top7_canonical_donor']['current_downstream_stage6_ckpt']}`",
        "",
        "## Locked policy",
        "",
        "- Step A gate remains necessary-but-not-sufficient.",
        "- promotion / ranking remains bound to Step B'.",
        "- incumbent remains fixed at `current_bad.teacher_x_gt`.",
        "- hard reject remains the fixed incumbent `nonleg_p95` threshold.",
        "",
        "## A. 接入清单",
        "",
        "### Phase 1",
        "",
    ]
    _append_artifact_rows(
        lines,
        (
            ("70a", p1["70a_oldcut"]),
            ("70a", p1["70a_stepc"]),
            ("replace", p1["replace_oldcut"]),
            ("replace", p1["replace_stepc"]),
            ("70R", p1["70R_oldcut"]),
            ("70R", p1["70R_stepc"]),
        ),
    )
    if bool(phase2.get("executed")):
        p2 = phase2["stages"]
        lines.extend(
            [
                "",
                "### Phase 2",
                "",
            ]
        )
        _append_artifact_rows(
            lines,
            (
                ("71", p2["71_oldcut"]),
                ("71", p2["71_stepc"]),
                ("72", p2["72_oldcut"]),
                ("72", p2["72_stepc"]),
                ("lambda", p2["lambda_oldcut"]),
                ("lambda", p2["lambda_stepc"]),
            ),
        )
    lines.extend(
        [
            "",
            "## B. 对比表",
            "",
        ]
    )
    _append_compare_block(
        lines,
        "Phase 1 — top7 old-cut 70a vs top7 StepC 70a",
        phase1["comparisons"]["70a_stepc_vs_oldcut_70a"]["stepb_verdict"],
        p1["70a_oldcut"],
        p1["70a_stepc"],
    )
    _append_compare_block(
        lines,
        "Phase 1 — top7 old-cut replace vs top7 StepC replace",
        phase1["comparisons"]["replace_stepc_vs_oldcut_replace"]["stepb_verdict"],
        p1["replace_oldcut"],
        p1["replace_stepc"],
    )
    _append_compare_block(
        lines,
        "Phase 1 — top7 old-cut 70R vs top7 StepC 70R",
        phase1["comparisons"]["70R_stepc_vs_oldcut_70R"]["stepb_verdict"],
        p1["70R_oldcut"],
        p1["70R_stepc"],
    )
    if bool(phase2.get("executed")):
        p2 = phase2["stages"]
        _append_compare_block(
            lines,
            "Phase 2 — top7 old-cut 71 vs top7 StepC 71",
            phase2["comparisons"]["71_stepc_vs_oldcut_71"]["stepb_verdict"],
            p2["71_oldcut"],
            p2["71_stepc"],
        )
        _append_compare_block(
            lines,
            "Phase 2 — top7 old-cut 72 vs top7 StepC 72",
            phase2["comparisons"]["72_stepc_vs_oldcut_72"]["stepb_verdict"],
            p2["72_oldcut"],
            p2["72_stepc"],
        )
        _append_compare_block(
            lines,
            "Phase 2 — top7 old-cut lambda vs top7 StepC lambda",
            phase2["comparisons"]["lambda_stepc_vs_oldcut_lambda"]["stepb_verdict"],
            p2["lambda_oldcut"],
            p2["lambda_stepc"],
        )
    lines.extend(["## C. Step B' verdict", ""])
    _append_verdict_table(
        lines,
        "Phase 1",
        (
            ("top7 StepC 70a vs old-cut 70a", phase1["comparisons"]["70a_stepc_vs_oldcut_70a"]["stepb_verdict"]),
            ("top7 StepC replace vs old-cut replace", phase1["comparisons"]["replace_stepc_vs_oldcut_replace"]["stepb_verdict"]),
            ("top7 StepC 70R vs old-cut 70R", phase1["comparisons"]["70R_stepc_vs_oldcut_70R"]["stepb_verdict"]),
        ),
    )
    if bool(phase2.get("executed")):
        _append_verdict_table(
            lines,
            "Phase 2",
            (
                ("top7 StepC 71 vs old-cut 71", phase2["comparisons"]["71_stepc_vs_oldcut_71"]["stepb_verdict"]),
                ("top7 StepC 72 vs old-cut 72", phase2["comparisons"]["72_stepc_vs_oldcut_72"]["stepb_verdict"]),
                ("top7 StepC lambda vs old-cut lambda", phase2["comparisons"]["lambda_stepc_vs_oldcut_lambda"]["stepb_verdict"]),
            ),
        )
    lines.extend(
        [
            "## D. 因果解释结论",
            "",
            f"1. `top7` 的 regression 是否能被 StepC downstream 明显 rescue？ `{str(bool(answers['stepc_rescues_regression'])).lower()}`",
            f"2. rescue 程度：`{answers['rescue_extent']}`",
            f"3. 更支持的解释：`{answers['preferred_explanation']}`",
            f"4. 对 `top3` 的更精确写法：`{answers['top3_interpretation']}`",
            f"5. 后续最小下一步：`{answers['minimal_next_step']}`",
            "",
            "建议表述（N=5 / limited-N caveat）:",
            "",
        ]
    )
    if answers["preferred_explanation"] == "downstream-boundary-induced":
        lines.extend(
            [
                "- 不要写成 `top7 本身错误`。",
                "- 更准确写法：`top7 donor 在 old posttrain boundary 下被放大成 downstream regression；StepC unified leg terminal 基本消除了该放大。`",
                "",
            ]
        )
    elif answers["preferred_explanation"] == "two-layer interaction":
        lines.extend(
            [
                "- 不要只写 `top7 太 aggressive`。",
                "- 更准确写法：`top7 donor 的 downstream regression 不是单层原因；old boundary / fragmented handoff 明显放大了问题，但 upstream donor / early-stage recipe 仍保留一部分负担，因此更像 two-layer interaction。`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- 不要直接推到新架构或 sweep。",
                "- 更准确写法：`StepC 并没有明显救回 top7 downstream regression，因此当前证据更偏 basetrain-compromise-dominant；old boundary 可能有放大作用，但不是主因。`",
                "",
            ]
        )
    lines.extend(
        [
            "## E. 改动清单",
            "",
            "- `tools/run_top7_posttrain_oldcut_vs_stepc.py`: 最小 focused runner；只负责复用 top7 Phase 1 artifact、补跑 locked `71/72/lambda` old-cut / StepC lane、做 model-source eval、生成 summary / decision。",
            f"- `{SUMMARY_JSON}`: 本轮 machine-readable 汇总。",
            f"- `{SUMMARY_MD}`: 本轮简表。",
            f"- `{DECISION_MD}`: 本轮因果判断正文。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_phase2() -> Dict[str, Any]:
    stages = {
        "71_oldcut": run_locked_stage(
            stage="71",
            lane="old-cut",
            base_config=BASE_CONFIG_71,
            out_config=CFG_71_OLDCUT,
            ckpt_in=CKPT_70R_OLDCUT,
            out_dir=MODEL_ROOT / "71_oldcut",
            run_name=RUN_NAME_71_OLDCUT,
            lr_override=3e-4,
            stepc_enabled=False,
            eval_json=EVAL_71_OLDCUT,
            group_json=GROUP_71_OLDCUT,
        ),
        "71_stepc": run_locked_stage(
            stage="71",
            lane="StepC",
            base_config=BASE_CONFIG_71,
            out_config=CFG_71_STEPC,
            ckpt_in=CKPT_70R_STEPC,
            out_dir=MODEL_ROOT / "71_stepc",
            run_name=RUN_NAME_71_STEPC,
            lr_override=3e-4,
            stepc_enabled=True,
            eval_json=EVAL_71_STEPC,
            group_json=GROUP_71_STEPC,
        ),
    }
    stages["72_oldcut"] = run_locked_stage(
        stage="72",
        lane="old-cut",
        base_config=BASE_CONFIG_72,
        out_config=CFG_72_OLDCUT,
        ckpt_in=Path(stages["71_oldcut"]["output_ckpt"]),
        out_dir=MODEL_ROOT / "72_oldcut",
        run_name=RUN_NAME_72_OLDCUT,
        lr_override=1e-4,
        stepc_enabled=False,
        eval_json=EVAL_72_OLDCUT,
        group_json=GROUP_72_OLDCUT,
    )
    stages["72_stepc"] = run_locked_stage(
        stage="72",
        lane="StepC",
        base_config=BASE_CONFIG_72,
        out_config=CFG_72_STEPC,
        ckpt_in=Path(stages["71_stepc"]["output_ckpt"]),
        out_dir=MODEL_ROOT / "72_stepc",
        run_name=RUN_NAME_72_STEPC,
        lr_override=1e-4,
        stepc_enabled=True,
        eval_json=EVAL_72_STEPC,
        group_json=GROUP_72_STEPC,
    )
    stages["lambda_oldcut"] = run_locked_stage(
        stage="lambda",
        lane="old-cut",
        base_config=BASE_CONFIG_LAMBDA,
        out_config=CFG_LAMBDA_OLDCUT,
        ckpt_in=Path(stages["72_oldcut"]["output_ckpt"]),
        out_dir=MODEL_ROOT / "lambda_oldcut",
        run_name=RUN_NAME_LAMBDA_OLDCUT,
        lr_override=None,
        stepc_enabled=False,
        eval_json=EVAL_LAMBDA_OLDCUT,
        group_json=GROUP_LAMBDA_OLDCUT,
    )
    stages["lambda_stepc"] = run_locked_stage(
        stage="lambda",
        lane="StepC",
        base_config=BASE_CONFIG_LAMBDA,
        out_config=CFG_LAMBDA_STEPC,
        ckpt_in=Path(stages["72_stepc"]["output_ckpt"]),
        out_dir=MODEL_ROOT / "lambda_stepc",
        run_name=RUN_NAME_LAMBDA_STEPC,
        lr_override=None,
        stepc_enabled=True,
        eval_json=EVAL_LAMBDA_STEPC,
        group_json=GROUP_LAMBDA_STEPC,
    )
    comparisons = {
        "71_stepc_vs_oldcut_71": phase_compare(stages["71_stepc"], stages["71_oldcut"]),
        "72_stepc_vs_oldcut_72": phase_compare(stages["72_stepc"], stages["72_oldcut"]),
        "lambda_stepc_vs_oldcut_lambda": phase_compare(stages["lambda_stepc"], stages["lambda_oldcut"]),
        "lambda_stepc_vs_70R_stepc": {
            "delta": metric_delta(stages["lambda_stepc"]["metrics"], load_metrics(GROUP_70R_STEPC)),
        },
    }
    return {
        "executed": True,
        "stages": stages,
        "comparisons": comparisons,
    }


def main() -> int:
    assert_exists(
        [
            TEACHER,
            ENCODER_BUNDLE,
            AFFINE_STATS,
            TOP7_BASETRAIN_CONFIG,
            TOP7_BASETRAIN_CKPT,
            TOP7_BASETRAIN_SUMMARY_JSON,
            TOP7_BASETRAIN_SELECTOR_JSON,
            TOP7_BASETRAIN_GROUP_SUMMARY,
            TOP7_BASETRAIN_DECISION_MD,
            TOP7_STAGE6_SELECTOR_JSON,
            TOP7_STAGE6_OLDCUT_CKPT,
            TOP7_STAGE6_OLDCUT_GROUP,
            PHASE1_SUMMARY_JSON,
            PHASE1_SUMMARY_MD,
            PHASE1_CHAIN_MD,
            CFG_70A_OLDCUT,
            CKPT_70A_OLDCUT,
            EVAL_70A_OLDCUT,
            GROUP_70A_OLDCUT,
            CFG_70A_STEPC,
            CKPT_70A_STEPC,
            EVAL_70A_STEPC,
            GROUP_70A_STEPC,
            CFG_REPLACE_OLDCUT,
            CKPT_REPLACE_OLDCUT,
            EVAL_REPLACE_OLDCUT,
            GROUP_REPLACE_OLDCUT,
            CFG_REPLACE_STEPC,
            CKPT_REPLACE_STEPC,
            EVAL_REPLACE_STEPC,
            GROUP_REPLACE_STEPC,
            CFG_70R_OLDCUT,
            CKPT_70R_OLDCUT,
            EVAL_70R_OLDCUT,
            GROUP_70R_OLDCUT,
            CFG_70R_STEPC,
            CKPT_70R_STEPC,
            EVAL_70R_STEPC,
            GROUP_70R_STEPC,
            BASE_CONFIG_71,
            BASE_CONFIG_72,
            BASE_CONFIG_LAMBDA,
            INCUMBENT_GROUP,
            STEPB_DECISION,
        ]
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    log("loading existing top7 Phase 1 artifacts")
    phase1 = build_phase1()
    phase2: Dict[str, Any] | None = None
    if bool(phase1["proceed_to_phase2"]):
        log("Phase 1 shows clear 70R rescue; continuing locked Phase 2")
        phase2 = run_phase2()
    else:
        log("Phase 1 does not justify locked Phase 2; stopping at 70a/replace/70R")

    summary = build_summary(phase1, phase2)
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(build_summary_md(summary), encoding="utf-8")
    DECISION_MD.write_text(build_decision_md(summary), encoding="utf-8")
    log(f"wrote summary_json={SUMMARY_JSON}")
    log(f"wrote summary_md={SUMMARY_MD}")
    log(f"wrote decision_md={DECISION_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
