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

OUT_ROOT = ROOT / "debug_output" / f"_tmp_stage6_stepc_70r_to_lambda_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_stage6_stepc_70r_to_lambda_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DECISION_MD = OUT_ROOT / "decision.md"

TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv_20260317.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
PRETRAIN_CLAMP = "1.0"

START_CKPT = (
    ROOT
    / "models"
    / "__tmp_stage6_stepc_canonical_chain_20260412"
    / "70R_stepc"
    / "ckpt_last_WalkF_stage7_70R_fromfresh_stepc_s180_20260412.pth"
)
START_CONFIG = (
    ROOT
    / "debug_output"
    / "_tmp_stage6_stepc_canonical_chain_20260412"
    / "configs"
    / "posttrain_70R_fromfresh_stepc_20260412.json"
)
START_EVAL = (
    ROOT
    / "debug_output"
    / "_tmp_stage6_stepc_canonical_chain_20260412"
    / "70R_stepc"
    / "eval_model_source"
    / "Walk_F_freerun_cycles.json"
)
START_GROUP = (
    ROOT
    / "debug_output"
    / "_tmp_stage6_stepc_canonical_chain_20260412"
    / "70R_stepc"
    / "eval_model_source_group_summary.json"
)
STEP_C_SOURCE_SUMMARY = ROOT / "debug_output" / "_tmp_stage6_stepc_canonical_chain_20260412" / "summary.json"
STEP_C_SOURCE_DECISION = ROOT / "debug_output" / "_tmp_stage6_stepc_canonical_chain_20260412" / "decision.md"

BASE_CONFIG_71 = ROOT / "config" / "posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json"
BASE_CONFIG_72 = ROOT / "config" / "posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json"
BASE_CONFIG_LAMBDA = (
    ROOT / "config" / "posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json"
)

OLD_71_CONFIG = ROOT / "debug_output" / "_tmp_71_lowlr_sweep_20260314" / "configs" / "posttrain_71_lr3e4_20260314.json"
OLD_71_CKPT = (
    ROOT
    / "models"
    / "__tmp_71_lowlr_sweep_20260314"
    / "lr3e4"
    / "ckpt_last_WalkF_stage7_71_lr3e4_from_candidate70R_20260314.pth"
)
OLD_71_EVAL = ROOT / "debug_output" / "_tmp_71_lowlr_sweep_20260314" / "eval_model" / "lr3e4" / "s180" / "Walk_F_freerun_cycles.json"
OLD_71_GROUP = ROOT / "debug_output" / "_tmp_71_lowlr_sweep_20260314" / "eval_model" / "lr3e4" / "s180_group_summary.json"

OLD_72_CONFIG = ROOT / "debug_output" / "_tmp_72_lowlr_sweep_20260314" / "configs" / "posttrain_72_lr1e4_20260314.json"
OLD_72_CKPT = (
    ROOT
    / "models"
    / "__tmp_72_lowlr_sweep_20260314"
    / "lr1e4"
    / "ckpt_last_WalkF_stage7_72_lr1e4_from_lowlr71_20260314.pth"
)
OLD_72_EVAL = ROOT / "debug_output" / "_tmp_72_lowlr_sweep_20260314" / "eval_model" / "lr1e4" / "s180" / "Walk_F_freerun_cycles.json"
OLD_72_GROUP = ROOT / "debug_output" / "_tmp_72_lowlr_sweep_20260314" / "eval_model" / "lr1e4" / "s180_group_summary.json"

OLD_LAMBDA_CONFIG = BASE_CONFIG_LAMBDA
OLD_LAMBDA_CKPT = (
    ROOT
    / "models"
    / "__tmp_72_lowlr_to_lambda_20260315"
    / "lambda"
    / "ckpt_last_WalkF_stage7_lambda_from_lowlr72lr1e4_20260315.pth"
)
OLD_LAMBDA_EVAL = ROOT / "debug_output" / "_tmp_72_lowlr_to_lambda_20260315" / "eval_lambda_model" / "Walk_F_freerun_cycles.json"
OLD_LAMBDA_GROUP = ROOT / "debug_output" / "_tmp_72_lowlr_to_lambda_20260315" / "eval_lambda_model_group_summary.json"

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

RUN_NAME_71 = f"WalkF_stage7_71_from_70R_stepc_lr3e4_{RUN_DATE}"
RUN_NAME_72 = f"WalkF_stage7_72_from_71_stepc_lr1e4_{RUN_DATE}"
RUN_NAME_LAMBDA = f"WalkF_stage7_lambda_from_72_stepc_{RUN_DATE}"

CFG_71 = CONFIG_ROOT / f"posttrain_71_from_70R_stepc_lr3e4_{RUN_DATE}.json"
CFG_72 = CONFIG_ROOT / f"posttrain_72_from_71_stepc_lr1e4_{RUN_DATE}.json"
CFG_LAMBDA = CONFIG_ROOT / f"posttrain_lambda_from_72_stepc_{RUN_DATE}.json"

CKPT_71 = MODEL_ROOT / "71_stepc" / f"ckpt_last_{RUN_NAME_71}.pth"
CKPT_72 = MODEL_ROOT / "72_stepc" / f"ckpt_last_{RUN_NAME_72}.pth"
CKPT_LAMBDA = MODEL_ROOT / "lambda_stepc" / f"ckpt_last_{RUN_NAME_LAMBDA}.pth"

EVAL_71 = OUT_ROOT / "71_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_71 = OUT_ROOT / "71_stepc" / "eval_model_source_group_summary.json"
EVAL_72 = OUT_ROOT / "72_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_72 = OUT_ROOT / "72_stepc" / "eval_model_source_group_summary.json"
EVAL_LAMBDA = OUT_ROOT / "lambda_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_LAMBDA = OUT_ROOT / "lambda_stepc" / "eval_model_source_group_summary.json"


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
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def reference_stage(
    *,
    stage: str,
    config: Path,
    ckpt: Path,
    eval_json: Path,
    group_json: Path,
) -> Dict[str, Any]:
    return stage_record(stage=stage, config=config, ckpt=ckpt, eval_json=eval_json, group_json=group_json)


def run_locked_stage(
    *,
    stage: str,
    base_config: Path,
    out_config: Path,
    ckpt_in: Path,
    out_dir: Path,
    run_name: str,
    lr_override: float | None,
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
    layout = assert_stepc_layout(ckpt_out)
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


def classify_handoff(
    benefit_70r: Mapping[str, Any],
    benefit_lambda: Mapping[str, Any],
    lambda_vs_old_verdict: Mapping[str, Any],
) -> str:
    if str(lambda_vs_old_verdict.get("verdict")) != "win":
        return "washed_out_or_reversed"
    b70 = safe_float(benefit_70r.get("all_ex_root_mean"))
    bl = safe_float(benefit_lambda.get("all_ex_root_mean"))
    if bl > b70 + 0.002:
        return "amplified"
    if bl >= max(0.0, b70 - 0.002):
        return "retained"
    return "retained_but_attenuated"


def build_summary(chain: Mapping[str, Any], baselines: Mapping[str, Any]) -> Dict[str, Any]:
    fixed_incumbent_metrics = load_metrics(INCUMBENT_GROUP)
    comparisons = {
        "71_stepc_vs_oldcut_71_lr3e4": {
            "delta": metric_delta(chain["71_stepc"]["metrics"], baselines["71_oldcut"]["metrics"]),
            "stepb_verdict": compare_stepb(chain["71_stepc"]["metrics"], baselines["71_oldcut"]["metrics"]),
        },
        "72_stepc_vs_oldcut_72_lr1e4": {
            "delta": metric_delta(chain["72_stepc"]["metrics"], baselines["72_oldcut"]["metrics"]),
            "stepb_verdict": compare_stepb(chain["72_stepc"]["metrics"], baselines["72_oldcut"]["metrics"]),
        },
        "lambda_stepc_vs_oldcut_lambda": {
            "delta": metric_delta(chain["lambda_stepc"]["metrics"], baselines["lambda_oldcut"]["metrics"]),
            "stepb_verdict": compare_stepb(chain["lambda_stepc"]["metrics"], baselines["lambda_oldcut"]["metrics"]),
        },
        "lambda_stepc_vs_70R_stepc": {
            "delta": metric_delta(chain["lambda_stepc"]["metrics"], chain["70R_stepc"]["metrics"]),
        },
    }
    stepc_benefit = {
        "70R_stepc_vs_oldcut_70R": {
            "delta": metric_delta(chain["70R_stepc"]["metrics"], baselines["70R_oldcut"]["metrics"]),
            "improvement": metric_delta(baselines["70R_oldcut"]["metrics"], chain["70R_stepc"]["metrics"]),
            "stepb_verdict": compare_stepb(chain["70R_stepc"]["metrics"], baselines["70R_oldcut"]["metrics"]),
        },
        "lambda_stepc_vs_oldcut_lambda": {
            "delta": metric_delta(chain["lambda_stepc"]["metrics"], baselines["lambda_oldcut"]["metrics"]),
            "improvement": metric_delta(baselines["lambda_oldcut"]["metrics"], chain["lambda_stepc"]["metrics"]),
            "stepb_verdict": comparisons["lambda_stepc_vs_oldcut_lambda"]["stepb_verdict"],
        },
    }
    answers = {
        "chain_completed": all(Path(str(chain[key]["output_ckpt"])).is_file() for key in ("71_stepc", "72_stepc", "lambda_stepc")),
        "unified_leg_terminal_crosses_full_chain": str(comparisons["lambda_stepc_vs_oldcut_lambda"]["stepb_verdict"]["verdict"]) in {"win", "tie"},
        "lambda_vs_70r_stepc_handoff": classify_handoff(
            stepc_benefit["70R_stepc_vs_oldcut_70R"]["improvement"],
            stepc_benefit["lambda_stepc_vs_oldcut_lambda"]["improvement"],
            comparisons["lambda_stepc_vs_oldcut_lambda"]["stepb_verdict"],
        ),
        "need_canonical_handoff_explanation_change": (
            "yes_stepc_survives_full_chain"
            if str(comparisons["lambda_stepc_vs_oldcut_lambda"]["stepb_verdict"]["verdict"]) in {"win", "tie"}
            else "no_but_mark_as_downstream_continuation_regression"
        ),
        "need_70a_audit_followup": (
            "not_in_this_task; only if downstream regression persists"
            if str(comparisons["lambda_stepc_vs_oldcut_lambda"]["stepb_verdict"]["verdict"]) in {"win", "tie"}
            else "yes_minimal_followup_only; do not expand to new 70a sweep here"
        ),
    }
    return {
        "run_date": RUN_DATE,
        "caveat": "N=5 / limited-N",
        "script": str(ROOT / "tools" / "run_stage6_stepc_70r_to_lambda.py"),
        "source_checkpoint": str(START_CKPT),
        "source_stepc_summary": str(STEP_C_SOURCE_SUMMARY),
        "source_stepc_decision": str(STEP_C_SOURCE_DECISION),
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
        "chain": chain,
        "oldcut_references": baselines,
        "comparisons": comparisons,
        "stepc_benefit_tracking": stepc_benefit,
        "answers": answers,
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    chain = summary["chain"]
    baselines = summary["oldcut_references"]
    comparisons = summary["comparisons"]
    lines = [
        "# StepC 70R -> 71/72/lambda",
        "",
        f"- run_date: `{summary['run_date']}`",
        f"- caveat: `{summary['caveat']}`",
        f"- start ckpt: `{summary['source_checkpoint']}`",
        "- eval contract: `model-source`",
        "",
        "## Chain metrics",
        "",
        "| stage | all_ex_root_mean | all_ex_root_p95 | leg_mean | leg_p95 | nonleg_p95 | arm_mean | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("70R_stepc", "71_stepc", "72_stepc", "lambda_stepc"):
        metrics = chain[key]["metrics"]
        sel = chain[key]["eval"]["selected_metrics"]
        lines.append(
            f"| {key} | {fmt(metrics['all_ex_root_mean'])} | {fmt(metrics['all_ex_root_p95'])} | {fmt(metrics['leg_mean'])} | {fmt(metrics['leg_p95'])} | {fmt(metrics['nonleg_p95'])} | {fmt(metrics['arm_mean'])} | {fmt(sel['foot_l_ball_l_SIC12_15'])} | {fmt(sel['calf_r_SIC2_4'])} |"
        )
    lines.extend(
        [
            "",
            "## Step B' vs old-cut",
            "",
            "| stage | d_all_ex_root_mean | d_all_ex_root_p95 | d_leg_mean | d_leg_p95 | d_nonleg_p95 | verdict | trigger |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for label, comp_key in (
        ("71_stepc", "71_stepc_vs_oldcut_71_lr3e4"),
        ("72_stepc", "72_stepc_vs_oldcut_72_lr1e4"),
        ("lambda_stepc", "lambda_stepc_vs_oldcut_lambda"),
    ):
        delta = comparisons[comp_key]["stepb_verdict"]["delta"]
        verdict = comparisons[comp_key]["stepb_verdict"]
        lines.append(
            f"| {label} | {fmt(delta['all_ex_root_mean'])} | {fmt(delta['all_ex_root_p95'])} | {fmt(delta['leg_mean'])} | {fmt(delta['leg_p95'])} | {fmt(delta['nonleg_p95'])} | {verdict['verdict']} | {verdict['trigger']} |"
        )
    lines.extend(
        [
            "",
            "## Old-cut references",
            "",
            f"- 71 old-cut: `{baselines['71_oldcut']['group_summary']}`",
            f"- 72 old-cut: `{baselines['72_oldcut']['group_summary']}`",
            f"- lambda old-cut: `{baselines['lambda_oldcut']['group_summary']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_decision_markdown(summary: Mapping[str, Any]) -> str:
    chain = summary["chain"]
    baselines = summary["oldcut_references"]
    comparisons = summary["comparisons"]
    answers = summary["answers"]
    lines = [
        "# StepC downstream continuation decision",
        "",
        f"- run_date: `{summary['run_date']}`",
        f"- caveat: `{summary['caveat']}`",
        f"- script: `{summary['script']}`",
        f"- source checkpoint: `{summary['source_checkpoint']}`",
        "",
        "## Outputs",
        "",
    ]
    for key in ("71_stepc", "72_stepc", "lambda_stepc"):
        stage = chain[key]
        lines.extend(
            [
                f"### {key}",
                "",
                f"- config: `{stage['config']}`",
                f"- ckpt: `{stage['output_ckpt']}`",
                f"- eval: `{stage['eval_artifact']}`",
                f"- group summary: `{stage['group_summary']}`",
                f"- layout: `{stage['layout']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Metrics",
            "",
            "| stage | lane | all_ex_root_mean | all_ex_root_p95 | leg_mean | leg_p95 | nonleg_p95 | Step A gate | hard reject |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    table_rows = [
        ("70R", "StepC start", chain["70R_stepc"]),
        ("71", "old-cut", baselines["71_oldcut"]),
        ("71", "StepC", chain["71_stepc"]),
        ("72", "old-cut", baselines["72_oldcut"]),
        ("72", "StepC", chain["72_stepc"]),
        ("lambda", "old-cut", baselines["lambda_oldcut"]),
        ("lambda", "StepC", chain["lambda_stepc"]),
    ]
    for stage_name, lane_name, payload in table_rows:
        metrics = payload["metrics"]
        gate = payload["gate"]
        lines.append(
            f"| {stage_name} | {lane_name} | {fmt(metrics['all_ex_root_mean'])} | {fmt(metrics['all_ex_root_p95'])} | {fmt(metrics['leg_mean'])} | {fmt(metrics['leg_p95'])} | {fmt(metrics['nonleg_p95'])} | {'pass' if gate['step_a_gate'] else 'fail'} | {'yes' if gate['hard_reject'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Step B'",
            "",
        ]
    )
    for label, comp_key in (
        ("71-StepC vs old-cut 71(lr=3e-4)", "71_stepc_vs_oldcut_71_lr3e4"),
        ("72-StepC vs old-cut 72(lr=1e-4)", "72_stepc_vs_oldcut_72_lr1e4"),
        ("lambda-StepC vs old-cut lambda", "lambda_stepc_vs_oldcut_lambda"),
    ):
        verdict = comparisons[comp_key]["stepb_verdict"]
        delta = verdict["delta"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- verdict: `{verdict['verdict']}`",
                f"- trigger: `{verdict['trigger']}`",
                f"- primary_triggered: `{str(verdict['primary_triggered']).lower()}`",
                f"- tie_break1_triggered: `{str(verdict['tie_break1_triggered']).lower()}`",
                f"- tie_break2_triggered: `{str(verdict['tie_break2_triggered']).lower()}`",
                f"- hard_reject_triggered: `{str(verdict['hard_reject_triggered']).lower()}`",
                f"- delta(all_ex_root_mean/all_ex_root_p95/leg_mean/leg_p95/nonleg_p95): `{fmt(delta['all_ex_root_mean'])}, {fmt(delta['all_ex_root_p95'])}, {fmt(delta['leg_mean'])}, {fmt(delta['leg_p95'])}, {fmt(delta['nonleg_p95'])}`",
                "",
            ]
        )
    lambda_vs_70r = comparisons["lambda_stepc_vs_70R_stepc"]["delta"]
    lines.extend(
        [
            "## Answers",
            "",
            f"- full chain completed: `{str(bool(answers['chain_completed'])).lower()}`",
            f"- unified leg terminal crosses full downstream chain: `{str(bool(answers['unified_leg_terminal_crosses_full_chain'])).lower()}`",
            f"- lambda-StepC vs 70R-StepC: all_ex_root_mean=`{fmt(lambda_vs_70r['all_ex_root_mean'])}`, all_ex_root_p95=`{fmt(lambda_vs_70r['all_ex_root_p95'])}`, leg_mean=`{fmt(lambda_vs_70r['leg_mean'])}`, nonleg_p95=`{fmt(lambda_vs_70r['nonleg_p95'])}`",
            f"- StepC benefit state by lambda handoff: `{answers['lambda_vs_70r_stepc_handoff']}`",
            f"- canonical downstream handoff explanation change: `{answers['need_canonical_handoff_explanation_change']}`",
            f"- 70a follow-up status: `{answers['need_70a_audit_followup']}`",
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
            START_CKPT,
            START_CONFIG,
            START_EVAL,
            START_GROUP,
            STEP_C_SOURCE_SUMMARY,
            STEP_C_SOURCE_DECISION,
            BASE_CONFIG_71,
            BASE_CONFIG_72,
            BASE_CONFIG_LAMBDA,
            OLD_71_CONFIG,
            OLD_71_CKPT,
            OLD_71_EVAL,
            OLD_71_GROUP,
            OLD_72_CONFIG,
            OLD_72_CKPT,
            OLD_72_EVAL,
            OLD_72_GROUP,
            OLD_LAMBDA_CKPT,
            OLD_LAMBDA_EVAL,
            OLD_LAMBDA_GROUP,
            INCUMBENT_GROUP,
            STEPB_DECISION,
        ]
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    start_layout = assert_stepc_layout(START_CKPT)
    log(f"start checkpoint verified layout={start_layout}")

    chain = {
        "70R_stepc": stage_record(
            stage="70R_stepc",
            config=START_CONFIG,
            ckpt=START_CKPT,
            eval_json=START_EVAL,
            group_json=START_GROUP,
        )
    }
    baselines = {
        "70R_oldcut": reference_stage(
            stage="70R_oldcut",
            config=(
                ROOT
                / "debug_output"
                / "_tmp_posttrain_pipeline_from_bestfree_20260317"
                / "configs"
                / "posttrain_70R_fromfresh_20260317.json"
            ),
            ckpt=(
                ROOT
                / "models"
                / "__tmp_posttrain_pipeline_from_bestfree_20260317"
                / "70R"
                / "ckpt_last_WalkF_stage7_70R_fromfresh_s180_20260317.pth"
            ),
            eval_json=(
                ROOT
                / "debug_output"
                / "_tmp_posttrain_pipeline_from_bestfree_20260317"
                / "eval_model_source"
                / "70R"
                / "Walk_F_freerun_cycles.json"
            ),
            group_json=(
                ROOT
                / "debug_output"
                / "_tmp_posttrain_pipeline_from_bestfree_20260317"
                / "eval_model_source"
                / "70R_group_summary.json"
            ),
        ),
        "71_oldcut": reference_stage(
            stage="71_oldcut",
            config=OLD_71_CONFIG,
            ckpt=OLD_71_CKPT,
            eval_json=OLD_71_EVAL,
            group_json=OLD_71_GROUP,
        ),
        "72_oldcut": reference_stage(
            stage="72_oldcut",
            config=OLD_72_CONFIG,
            ckpt=OLD_72_CKPT,
            eval_json=OLD_72_EVAL,
            group_json=OLD_72_GROUP,
        ),
        "lambda_oldcut": reference_stage(
            stage="lambda_oldcut",
            config=OLD_LAMBDA_CONFIG,
            ckpt=OLD_LAMBDA_CKPT,
            eval_json=OLD_LAMBDA_EVAL,
            group_json=OLD_LAMBDA_GROUP,
        ),
    }

    log("=== stage 71 StepC ===")
    chain["71_stepc"] = run_locked_stage(
        stage="71_stepc",
        base_config=BASE_CONFIG_71,
        out_config=CFG_71,
        ckpt_in=START_CKPT,
        out_dir=MODEL_ROOT / "71_stepc",
        run_name=RUN_NAME_71,
        lr_override=3e-4,
        eval_json=EVAL_71,
        group_json=GROUP_71,
    )

    log("=== stage 72 StepC ===")
    chain["72_stepc"] = run_locked_stage(
        stage="72_stepc",
        base_config=BASE_CONFIG_72,
        out_config=CFG_72,
        ckpt_in=CKPT_71,
        out_dir=MODEL_ROOT / "72_stepc",
        run_name=RUN_NAME_72,
        lr_override=1e-4,
        eval_json=EVAL_72,
        group_json=GROUP_72,
    )

    log("=== stage lambda StepC ===")
    chain["lambda_stepc"] = run_locked_stage(
        stage="lambda_stepc",
        base_config=BASE_CONFIG_LAMBDA,
        out_config=CFG_LAMBDA,
        ckpt_in=CKPT_72,
        out_dir=MODEL_ROOT / "lambda_stepc",
        run_name=RUN_NAME_LAMBDA,
        lr_override=None,
        eval_json=EVAL_LAMBDA,
        group_json=GROUP_LAMBDA,
    )

    summary = build_summary(chain, baselines)
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(build_markdown(summary), encoding="utf-8")
    DECISION_MD.write_text(build_decision_markdown(summary), encoding="utf-8")
    log(f"DONE summary={SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
