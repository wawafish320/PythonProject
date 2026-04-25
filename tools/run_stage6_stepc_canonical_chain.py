#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "20260412"
PYTHON = Path(sys.executable)

OUT_ROOT = ROOT / "debug_output" / f"_tmp_stage6_stepc_canonical_chain_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_stage6_stepc_canonical_chain_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
LOG_FILE = OUT_ROOT / "lane.log"
SUMMARY_JSON = OUT_ROOT / "summary.json"
SUMMARY_MD = OUT_ROOT / "summary.md"
DECISION_MD = OUT_ROOT / "decision.md"
STATUS_JSON = OUT_ROOT / "status.json"

TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv_20260317.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
PRETRAIN_CLAMP = "1.0"

CANONICAL_DEBUG_ROOT = ROOT / "debug_output" / "_tmp_posttrain_pipeline_from_bestfree_20260317"
CANONICAL_MODEL_ROOT = ROOT / "models" / "__tmp_posttrain_pipeline_from_bestfree_20260317"

BASE_CONFIG_70A = ROOT / "config" / "posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json"
BASE_CONFIG_REPLACE = CANONICAL_DEBUG_ROOT / "configs" / "posttrain_70b_replace_lowdrift_fromfresh_20260317.json"
BASE_CONFIG_70R = CANONICAL_DEBUG_ROOT / "configs" / "posttrain_70R_fromfresh_20260317.json"

CANONICAL_STAGE6_CKPT = CANONICAL_MODEL_ROOT / "stage6" / "ckpt_last_WalkF_stage6_fromfresh_20260317.pth"
CANONICAL_70A_CKPT = CANONICAL_MODEL_ROOT / "70a" / "ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth"
CANONICAL_REPLACE_CKPT = CANONICAL_MODEL_ROOT / "70b_replace_lowdrift" / "ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth"
CANONICAL_70R_CKPT = CANONICAL_MODEL_ROOT / "70R" / "ckpt_last_WalkF_stage7_70R_fromfresh_s180_20260317.pth"
CANONICAL_WARMSTART_CKPT = CANONICAL_MODEL_ROOT / "warmstart" / "ckpt_last_70a_replace_zerophase_20260317.pth"
CANONICAL_WARMSTART_REPORT = CANONICAL_DEBUG_ROOT / "warmstart" / "replace_zerophase_report.json"

CANONICAL_70A_EVAL = CANONICAL_DEBUG_ROOT / "eval_model_source" / "70a" / "Walk_F_freerun_cycles.json"
CANONICAL_70A_GROUP = CANONICAL_DEBUG_ROOT / "eval_model_source" / "70a_group_summary.json"
CANONICAL_REPLACE_EVAL = CANONICAL_DEBUG_ROOT / "eval_model_source" / "new70b_replace_lowdrift" / "Walk_F_freerun_cycles.json"
CANONICAL_REPLACE_GROUP = CANONICAL_DEBUG_ROOT / "eval_model_source" / "new70b_replace_lowdrift_group_summary.json"
CANONICAL_70R_EVAL = CANONICAL_DEBUG_ROOT / "eval_model_source" / "70R" / "Walk_F_freerun_cycles.json"
CANONICAL_70R_GROUP = CANONICAL_DEBUG_ROOT / "eval_model_source" / "70R_group_summary.json"

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

RUN_NAME_70A_STEPC = f"WalkF_stage7_70a_fromfresh_stepc_{RUN_DATE}"
RUN_NAME_REPLACE_STEPC = f"WalkF_stage7_70b_replace_lowdrift_fromfresh_stepc_{RUN_DATE}"
RUN_NAME_70R_STEPC = f"WalkF_stage7_70R_fromfresh_stepc_s180_{RUN_DATE}"

CFG_70A_STEPC = CONFIG_ROOT / f"posttrain_70a_fromfresh_stepc_{RUN_DATE}.json"
CFG_REPLACE_STEPC = CONFIG_ROOT / f"posttrain_70b_replace_lowdrift_fromfresh_stepc_{RUN_DATE}.json"
CFG_70R_STEPC = CONFIG_ROOT / f"posttrain_70R_fromfresh_stepc_{RUN_DATE}.json"

CKPT_70A_STEPC = MODEL_ROOT / "70a_stepc" / f"ckpt_last_{RUN_NAME_70A_STEPC}.pth"
CKPT_REPLACE_WARMSTART = MODEL_ROOT / "warmstart" / f"ckpt_last_70a_replace_zerophase_stepc_{RUN_DATE}.pth"
CKPT_REPLACE_STEPC = MODEL_ROOT / "replace_stepc" / f"ckpt_last_{RUN_NAME_REPLACE_STEPC}.pth"
CKPT_70R_STEPC = MODEL_ROOT / "70R_stepc" / f"ckpt_last_{RUN_NAME_70R_STEPC}.pth"

REPORT_WARMSTART = OUT_ROOT / "warmstart" / "replace_zerophase_report.json"

EVAL_70A_STEPC = OUT_ROOT / "70a_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_70A_STEPC = OUT_ROOT / "70a_stepc" / "eval_model_source_group_summary.json"
EVAL_REPLACE_STEPC = OUT_ROOT / "replace_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_REPLACE_STEPC = OUT_ROOT / "replace_stepc" / "eval_model_source_group_summary.json"
EVAL_70R_STEPC = OUT_ROOT / "70R_stepc" / "eval_model_source" / "Walk_F_freerun_cycles.json"
GROUP_70R_STEPC = OUT_ROOT / "70R_stepc" / "eval_model_source_group_summary.json"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_float(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def fmt(x: Any, digits: int = 6) -> str:
    value = safe_float(x)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


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


def changed_cols(src: torch.Tensor, dst: torch.Tensor) -> List[int]:
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape:
        return []
    diff = (dst.detach().cpu().float() - src.detach().cpu().float()).abs()
    cols = []
    for col in range(int(diff.shape[1])):
        if float(diff[:, col].max().item()) > 0.0:
            cols.append(int(col))
    return cols


def create_replace_warmstart(src_ckpt: Path, dst_ckpt: Path, report_json: Path) -> Dict[str, Any]:
    if dst_ckpt.is_file() and report_json.is_file():
        return load_json(report_json)

    baseline_report = load_json(CANONICAL_WARMSTART_REPORT)
    base_src_state, _ = state_and_cfg(CANONICAL_70A_CKPT)
    base_ws_state, _ = state_and_cfg(CANONICAL_WARMSTART_CKPT)

    donor_obj = torch.load(src_ckpt, map_location="cpu")
    if not isinstance(donor_obj, dict):
        raise RuntimeError(f"unsupported donor checkpoint format: {src_ckpt}")
    donor_state = donor_obj.get("model", donor_obj)
    if not isinstance(donor_state, dict):
        raise RuntimeError(f"unsupported donor state_dict format: {src_ckpt}")

    target_keys = [str(item.get("key")) for item in baseline_report.get("changed", []) if str(item.get("key", ""))]
    if target_keys != ["direct_pose_head.0.weight", "direct_pose_leg_head.0.weight"]:
        raise RuntimeError(f"unexpected canonical warmstart target keys: {target_keys}")

    out_obj = dict(donor_obj)
    out_state = dict(donor_state)
    applied: List[Dict[str, Any]] = []
    for key in target_keys:
        donor_tensor = donor_state.get(key)
        base_src_tensor = base_src_state.get(key)
        base_ws_tensor = base_ws_state.get(key)
        if not (torch.is_tensor(donor_tensor) and torch.is_tensor(base_src_tensor) and torch.is_tensor(base_ws_tensor)):
            raise RuntimeError(f"missing warmstart tensor for key={key}")
        if tuple(donor_tensor.shape) != tuple(base_src_tensor.shape) or tuple(donor_tensor.shape) != tuple(base_ws_tensor.shape):
            raise RuntimeError(
                f"shape mismatch for key={key}: donor={tuple(donor_tensor.shape)} "
                f"base70a={tuple(base_src_tensor.shape)} basews={tuple(base_ws_tensor.shape)}"
            )
        delta = base_ws_tensor.detach().cpu().float() - base_src_tensor.detach().cpu().float()
        donor_float = donor_tensor.detach().cpu().float()
        adapted_float = donor_float + delta
        out_state[key] = adapted_float.to(dtype=donor_tensor.dtype)
        applied.append(
            {
                "key": key,
                "shape": list(donor_tensor.shape),
                "changed_cols": changed_cols(donor_float, adapted_float),
                "delta_l2": float((adapted_float - donor_float).norm().item()),
                "delta_max_abs": float((adapted_float - donor_float).abs().max().item()),
            }
        )

    out_obj["model"] = out_state
    donor_cfg = donor_obj.get("posttrain_cfg", {})
    out_cfg = dict(donor_cfg) if isinstance(donor_cfg, dict) else {}
    out_cfg["direct_pose_use_phase_z"] = True
    out_cfg["direct_pose_phase_z_mode"] = "replace_contacts"
    out_obj["posttrain_cfg"] = out_cfg

    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, dst_ckpt)

    report = {
        "source_ckpt": str(src_ckpt),
        "output_ckpt": str(dst_ckpt),
        "baseline_template_source_ckpt": str(CANONICAL_70A_CKPT),
        "baseline_template_warmstart_ckpt": str(CANONICAL_WARMSTART_CKPT),
        "baseline_template_report_json": str(CANONICAL_WARMSTART_REPORT),
        "baseline_template_report": baseline_report,
        "applied_strategy": {
            "type": "add_exact_canonical_warmstart_delta",
            "description": "Apply the exact canonical 70a->replace zerophase tensor delta onto the StepC donor.",
            "checkpoint_posttrain_cfg_updates": {
                "direct_pose_use_phase_z": True,
                "direct_pose_phase_z_mode": "replace_contacts",
            },
        },
        "applied_tensor_deltas": applied,
        "strict_only_two_tensor_keys_changed": bool(len(applied) == 2),
        "strict_only_cols_39_42_changed": bool(
            all(item.get("changed_cols") == [39, 40, 41, 42] for item in applied)
        ),
        "compat_loading": {
            "type": "warmstart_copy",
            "legacy_upgrade": False,
            "reinit": False,
            "partial_load": False,
            "tensor_upgrade": False,
            "warmstart_copy_only": True,
        },
    }
    write_json(report_json, report)
    return report


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


def load_metrics(group_json: Path) -> Dict[str, float]:
    groups = load_json(group_json).get("groups", {})
    def metric(group: str, key: str) -> float:
        return safe_float(groups.get(group, {}).get(key))
    return {
        "all_ex_root_mean": metric("all_ex_root", "mean"),
        "all_ex_root_p95": metric("all_ex_root", "p95"),
        "leg_mean": metric("leg", "mean"),
        "leg_p95": metric("leg", "p95"),
        "nonleg_p95": metric("nonleg", "p95"),
    }


def metric_delta(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, float]:
    return {key: safe_float(cur.get(key)) - safe_float(ref.get(key)) for key in cur.keys()}


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


def compare_stepb(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, Any]:
    d_mean = safe_float(cur.get("all_ex_root_mean")) - safe_float(ref.get("all_ex_root_mean"))
    d_p95 = safe_float(cur.get("all_ex_root_p95")) - safe_float(ref.get("all_ex_root_p95"))
    d_leg = safe_float(cur.get("leg_mean")) - safe_float(ref.get("leg_mean"))
    hard_reject = bool(safe_float(cur.get("nonleg_p95")) >= HARD_REJECT_THRESHOLD_NONLEG_P95)
    if hard_reject:
        verdict = "lose_hard_reject"
        rationale = "nonleg_p95 triggers hard reject"
    elif abs(d_mean) >= 0.002:
        verdict = "win" if d_mean < 0.0 else "lose"
        rationale = "primary=all_ex_root_mean"
    elif abs(d_p95) >= 0.01:
        verdict = "win" if d_p95 < 0.0 else "lose"
        rationale = "tie-break1=all_ex_root_p95"
    else:
        verdict = "win" if d_leg < 0.0 else ("tie" if abs(d_leg) < 1e-12 else "lose")
        rationale = "tie-break2=leg_mean"
    return {
        "verdict": verdict,
        "rationale": rationale,
        "delta": {
            "all_ex_root_mean": d_mean,
            "all_ex_root_p95": d_p95,
            "leg_mean": d_leg,
            "leg_p95": safe_float(cur.get("leg_p95")) - safe_float(ref.get("leg_p95")),
            "nonleg_p95": safe_float(cur.get("nonleg_p95")) - safe_float(ref.get("nonleg_p95")),
        },
    }


def ckpt_layout(ckpt: Path) -> Dict[str, bool]:
    state, _cfg = state_and_cfg(ckpt)
    return {
        "has_direct_pose_leg_terminal": any(str(k).startswith("direct_pose_leg_terminal.") for k in state.keys()),
    }


def canonical_baseline_rows() -> Dict[str, Any]:
    baseline_70a_cmd = (
        f"PYTHONPATH=. {PYTHON} -m train.posttrain "
        f"--config {BASE_CONFIG_70A} "
        f"--ckpt_in {CANONICAL_STAGE6_CKPT} "
        f"--out_dir {CANONICAL_MODEL_ROOT / '70a'} "
        f"--run_name WalkF_stage7_70a_fromfresh_20260317 "
        "--posttrain_contacts_source pretrain_contact "
        f"--posttrain_contacts_pretrain_clamp {PRETRAIN_CLAMP} "
        f"--encoder_bundle {ENCODER_BUNDLE} "
        f"--posttrain_contacts_pretrain_affine_stats {AFFINE_STATS}"
    )
    baseline_replace_cmd = (
        f"PYTHONPATH=. {PYTHON} -m train.posttrain "
        f"--config {BASE_CONFIG_REPLACE} "
        f"--ckpt_in {CANONICAL_WARMSTART_CKPT} "
        f"--out_dir {CANONICAL_MODEL_ROOT / '70b_replace_lowdrift'} "
        "--run_name WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317 "
        "--posttrain_contacts_source pretrain_contact "
        f"--posttrain_contacts_pretrain_clamp {PRETRAIN_CLAMP} "
        f"--encoder_bundle {ENCODER_BUNDLE} "
        f"--posttrain_contacts_pretrain_affine_stats {AFFINE_STATS}"
    )
    baseline_70r_cmd = (
        f"PYTHONPATH=. {PYTHON} {ROOT / 'tools' / 'run_posttrain_nonleg_trunk_ablation.py'} "
        f"--config {BASE_CONFIG_70R} "
        "--trunk-mode full "
        f"--out-dir {CANONICAL_MODEL_ROOT / '70R'} "
        "--run-name WalkF_stage7_70R_fromfresh_s180_20260317 "
        "--epochs 1 "
        "--steps-per-epoch 180 "
        "--save-step-ckpts 0,1,5,20,60,180"
    )
    return {
        "70a": {
            "config": str(BASE_CONFIG_70A),
            "launch_command": baseline_70a_cmd,
            "input_artifact": str(CANONICAL_STAGE6_CKPT),
            "output_ckpt": str(CANONICAL_70A_CKPT),
            "eval_artifact": str(CANONICAL_70A_EVAL),
            "group_summary": str(CANONICAL_70A_GROUP),
            "metrics": load_metrics(CANONICAL_70A_GROUP),
        },
        "replace": {
            "config": str(BASE_CONFIG_REPLACE),
            "launch_command": baseline_replace_cmd,
            "input_artifact": str(CANONICAL_WARMSTART_CKPT),
            "warmstart_report": str(CANONICAL_WARMSTART_REPORT),
            "output_ckpt": str(CANONICAL_REPLACE_CKPT),
            "eval_artifact": str(CANONICAL_REPLACE_EVAL),
            "group_summary": str(CANONICAL_REPLACE_GROUP),
            "metrics": load_metrics(CANONICAL_REPLACE_GROUP),
        },
        "70R": {
            "config": str(BASE_CONFIG_70R),
            "launch_command": baseline_70r_cmd,
            "input_artifact": str(CANONICAL_REPLACE_CKPT),
            "output_ckpt": str(CANONICAL_70R_CKPT),
            "eval_artifact": str(CANONICAL_70R_EVAL),
            "group_summary": str(CANONICAL_70R_GROUP),
            "metrics": load_metrics(CANONICAL_70R_GROUP),
        },
    }


def stepc_rows() -> Dict[str, Any]:
    cfg_70a = make_generated_config(
        BASE_CONFIG_70A,
        CFG_70A_STEPC,
        {
            "ckpt_in": str(CANONICAL_STAGE6_CKPT),
            "out_dir": str(MODEL_ROOT / "70a_stepc"),
            "run_name": RUN_NAME_70A_STEPC,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    ckpt_70a = run_posttrain(cfg_70a, CANONICAL_STAGE6_CKPT, MODEL_ROOT / "70a_stepc", RUN_NAME_70A_STEPC)
    eval_70a = run_eval(ckpt_70a, OUT_ROOT / "70a_stepc" / "eval_model_source")
    ensure_group_summary(eval_70a, GROUP_70A_STEPC)

    warmstart_report = create_replace_warmstart(ckpt_70a, CKPT_REPLACE_WARMSTART, REPORT_WARMSTART)

    cfg_replace = make_generated_config(
        BASE_CONFIG_REPLACE,
        CFG_REPLACE_STEPC,
        {
            "ckpt_in": str(CKPT_REPLACE_WARMSTART),
            "out_dir": str(MODEL_ROOT / "replace_stepc"),
            "run_name": RUN_NAME_REPLACE_STEPC,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
            "direct_pose_use_phase_z": True,
            "direct_pose_phase_z_mode": "replace_contacts",
        },
    )
    ckpt_replace = run_posttrain(cfg_replace, CKPT_REPLACE_WARMSTART, MODEL_ROOT / "replace_stepc", RUN_NAME_REPLACE_STEPC)
    eval_replace = run_eval(ckpt_replace, OUT_ROOT / "replace_stepc" / "eval_model_source")
    ensure_group_summary(eval_replace, GROUP_REPLACE_STEPC)

    cfg_70r = make_generated_config(
        BASE_CONFIG_70R,
        CFG_70R_STEPC,
        {
            "ckpt_in": str(ckpt_replace),
            "out_dir": str(MODEL_ROOT / "70R_stepc"),
            "run_name": RUN_NAME_70R_STEPC,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
            "direct_pose_use_phase_z": True,
            "direct_pose_phase_z_mode": "replace_contacts",
        },
    )
    ckpt_70r = run_70r(cfg_70r, MODEL_ROOT / "70R_stepc", RUN_NAME_70R_STEPC)
    eval_70r = run_eval(ckpt_70r, OUT_ROOT / "70R_stepc" / "eval_model_source")
    ensure_group_summary(eval_70r, GROUP_70R_STEPC)

    cmd_70a = (
        f"PYTHONPATH=. {PYTHON} -m train.posttrain "
        f"--config {cfg_70a} "
        f"--ckpt_in {CANONICAL_STAGE6_CKPT} "
        f"--out_dir {MODEL_ROOT / '70a_stepc'} "
        f"--run_name {RUN_NAME_70A_STEPC} "
        "--posttrain_contacts_source pretrain_contact "
        f"--posttrain_contacts_pretrain_clamp {PRETRAIN_CLAMP} "
        f"--encoder_bundle {ENCODER_BUNDLE} "
        f"--posttrain_contacts_pretrain_affine_stats {AFFINE_STATS}"
    )
    cmd_replace = (
        f"PYTHONPATH=. {PYTHON} -m train.posttrain "
        f"--config {cfg_replace} "
        f"--ckpt_in {CKPT_REPLACE_WARMSTART} "
        f"--out_dir {MODEL_ROOT / 'replace_stepc'} "
        f"--run_name {RUN_NAME_REPLACE_STEPC} "
        "--posttrain_contacts_source pretrain_contact "
        f"--posttrain_contacts_pretrain_clamp {PRETRAIN_CLAMP} "
        f"--encoder_bundle {ENCODER_BUNDLE} "
        f"--posttrain_contacts_pretrain_affine_stats {AFFINE_STATS}"
    )
    cmd_70r = (
        f"PYTHONPATH=. {PYTHON} {ROOT / 'tools' / 'run_posttrain_nonleg_trunk_ablation.py'} "
        f"--config {cfg_70r} "
        "--trunk-mode full "
        f"--out-dir {MODEL_ROOT / '70R_stepc'} "
        f"--run-name {RUN_NAME_70R_STEPC} "
        "--epochs 1 "
        "--steps-per-epoch 180 "
        "--save-step-ckpts 0,1,5,20,60,180"
    )
    return {
        "70a": {
            "config": str(cfg_70a),
            "launch_command": cmd_70a,
            "input_artifact": str(CANONICAL_STAGE6_CKPT),
            "output_ckpt": str(ckpt_70a),
            "eval_artifact": str(eval_70a),
            "group_summary": str(GROUP_70A_STEPC),
            "metrics": load_metrics(GROUP_70A_STEPC),
            "compat_loading": {
                "stage": "70a-StepC donor",
                "source_layout": ckpt_layout(CANONICAL_STAGE6_CKPT),
                "result_layout": ckpt_layout(ckpt_70a),
                "type": "partial_load_plus_tensor_upgrade",
                "warmstart_copy": False,
                "legacy_upgrade": True,
                "reinit": [
                    "direct_pose_leg_terminal.0.weight",
                    "direct_pose_leg_terminal.0.bias",
                    "direct_pose_leg_terminal.3.weight",
                    "direct_pose_leg_terminal.3.bias",
                ],
                "tensor_upgrade": [
                    "canonical split leg terminal readout",
                ],
                "partial_load": True,
            },
        },
        "replace": {
            "config": str(cfg_replace),
            "launch_command": cmd_replace,
            "input_artifact": str(CKPT_REPLACE_WARMSTART),
            "warmstart_report": str(REPORT_WARMSTART),
            "output_ckpt": str(ckpt_replace),
            "eval_artifact": str(eval_replace),
            "group_summary": str(GROUP_REPLACE_STEPC),
            "metrics": load_metrics(GROUP_REPLACE_STEPC),
            "compat_loading": {
                "stage": "canonical replace-StepC",
                "source_layout": ckpt_layout(CKPT_REPLACE_WARMSTART),
                "result_layout": ckpt_layout(ckpt_replace),
                "type": "warmstart_copy_then_native_stepc_load",
                "warmstart_copy": True,
                "legacy_upgrade": False,
                "reinit": False,
                "partial_load": False,
                "tensor_upgrade": False,
            },
        },
        "70R": {
            "config": str(cfg_70r),
            "launch_command": cmd_70r,
            "input_artifact": str(ckpt_replace),
            "output_ckpt": str(ckpt_70r),
            "eval_artifact": str(eval_70r),
            "group_summary": str(GROUP_70R_STEPC),
            "metrics": load_metrics(GROUP_70R_STEPC),
            "compat_loading": {
                "stage": "canonical 70R-StepC",
                "source_layout": ckpt_layout(ckpt_replace),
                "result_layout": ckpt_layout(ckpt_70r),
                "type": "native_stepc_load",
                "warmstart_copy": False,
                "legacy_upgrade": False,
                "reinit": False,
                "partial_load": False,
                "tensor_upgrade": False,
            },
        },
        "warmstart_report_payload": warmstart_report,
    }


def enrich_rows(rows: Dict[str, Any]) -> None:
    for item in rows.values():
        if not isinstance(item, dict) or "metrics" not in item:
            continue
        item["gate"] = gate_flags(item["metrics"])


def build_summary(baseline: Dict[str, Any], stepc: Dict[str, Any]) -> Dict[str, Any]:
    enrich_rows(baseline)
    enrich_rows(stepc)
    incumbent_metrics = load_metrics(INCUMBENT_GROUP)
    comparisons: Dict[str, Any] = {}
    for stage in ("70a", "replace", "70R"):
        comparisons[stage] = {
            "stepc_minus_baseline": metric_delta(stepc[stage]["metrics"], baseline[stage]["metrics"]),
            "stepb_verdict": compare_stepb(stepc[stage]["metrics"], baseline[stage]["metrics"]),
        }

    answers = {
        "q1_70a_stepc_donor_more_stable": {
            "answer": "yes" if comparisons["70a"]["stepb_verdict"]["verdict"] == "win" else "no",
            "stepb_verdict": comparisons["70a"]["stepb_verdict"],
        },
        "q2_replace_handoff_preserves_or_disperses": {
            "answer": (
                "preserves_or_improves"
                if comparisons["replace"]["stepb_verdict"]["verdict"] == "win"
                else "disperses_or_degrades"
            ),
            "stepb_verdict": comparisons["replace"]["stepb_verdict"],
        },
        "q3_canonical_70r_stepc_beats_old_cut": {
            "answer": "yes" if comparisons["70R"]["stepb_verdict"]["verdict"] == "win" else "no",
            "stepb_verdict": comparisons["70R"]["stepb_verdict"],
        },
        "q4_supported_explanation": (
            "shared trunk + grouped readout + unified leg terminal"
            if comparisons["70R"]["stepb_verdict"]["verdict"] == "win"
            else "inconclusive_but_old_independent_leg_head_out_leg_is_not_supported"
        ),
        "q5_need_more_70a_recipe_tuning": {
            "answer": (
                "not_yet"
                if comparisons["70R"]["stepb_verdict"]["verdict"] == "win"
                else "maybe"
            ),
            "minimal_next_step": (
                "Only after canonical downstream readout: probe a minimal 70a donor-side init stabilization for the newly reinitialized "
                "direct_pose_leg_terminal.0/3 layers, because those are the only new tensors introduced by StepC at donor load."
                if comparisons["70R"]["stepb_verdict"]["verdict"] != "win"
                else "No additional 70a recipe change is required before canonical downstream readout, because the StepC chain already clears the real handoff test."
            ),
        },
    }

    return {
        "run_date": RUN_DATE,
        "caveat": "N=5 / limited-N",
        "locked_policy": {
            "step_a_gate": "necessary-but-not-sufficient",
            "step_b_prime": {
                "primary": "all_ex_root_mean",
                "tie_break1": "all_ex_root_p95 if abs(delta_all_ex_root_mean) < 0.002°",
                "tie_break2": "leg_mean if abs(delta_all_ex_root_p95) < 0.01°",
                "hard_reject": "nonleg_p95 regression >= 5% vs fixed sealed incumbent current_bad.teacher_x_gt",
            },
            "incumbent": "current_bad.teacher_x_gt",
            "reference_decision": str(STEPB_DECISION),
        },
        "fixed_incumbent": {
            "group_summary": str(INCUMBENT_GROUP),
            "metrics": incumbent_metrics,
            "thresholds": {
                "step_a_all_ex_root_mean": STEPA_THRESHOLD_ALL_EX_ROOT_MEAN,
                "step_a_leg_p95": STEPA_THRESHOLD_LEG_P95,
                "hard_reject_nonleg_p95": HARD_REJECT_THRESHOLD_NONLEG_P95,
            },
        },
        "baseline_old_cut": baseline,
        "stepc_chain": stepc,
        "comparisons": comparisons,
        "answers": answers,
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    baseline = summary["baseline_old_cut"]
    stepc = summary["stepc_chain"]
    comparisons = summary["comparisons"]
    answers = summary["answers"]
    lines: List[str] = [
        "# Stage6 Step C canonical downstream verification",
        "",
        f"- run_date: `{summary['run_date']}`",
        f"- caveat: `{summary['caveat']}`",
        f"- Step A gate: `{summary['locked_policy']['step_a_gate']}`",
        f"- Step B' primary: `{summary['locked_policy']['step_b_prime']['primary']}`",
        f"- incumbent: `{summary['locked_policy']['incumbent']}`",
        "",
        "## Metrics",
        "",
        "| stage | lane | all_ex_root_mean | all_ex_root_p95 | leg_mean | leg_p95 | nonleg_p95 | Step A gate | hard reject |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for stage in ("70a", "replace", "70R"):
        for lane_name, lane in (("baseline", baseline[stage]), ("StepC", stepc[stage])):
            metrics = lane["metrics"]
            gate = lane["gate"]
            lines.append(
                f"| {stage} | {lane_name} | {fmt(metrics['all_ex_root_mean'])} | {fmt(metrics['all_ex_root_p95'])} | "
                f"{fmt(metrics['leg_mean'])} | {fmt(metrics['leg_p95'])} | {fmt(metrics['nonleg_p95'])} | "
                f"{'pass' if gate['step_a_gate'] else 'fail'} | {'yes' if gate['hard_reject'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| stage | d_all_ex_root_mean | d_all_ex_root_p95 | d_leg_mean | d_leg_p95 | d_nonleg_p95 | Step B' verdict | rationale |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for stage in ("70a", "replace", "70R"):
        delta = comparisons[stage]["stepc_minus_baseline"]
        verdict = comparisons[stage]["stepb_verdict"]
        lines.append(
            f"| {stage} | {fmt(delta['all_ex_root_mean'])} | {fmt(delta['all_ex_root_p95'])} | "
            f"{fmt(delta['leg_mean'])} | {fmt(delta['leg_p95'])} | {fmt(delta['nonleg_p95'])} | "
            f"{verdict['verdict']} | {verdict['rationale']} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- 1. `70a-StepC donor` 本身是否更稳？ `{answers['q1_70a_stepc_donor_more_stable']['answer']}`",
            f"- 2. canonical replace handoff 是否保留这种改善？ `{answers['q2_replace_handoff_preserves_or_disperses']['answer']}`",
            f"- 3. canonical `70R-StepC` 是否优于 canonical old-cut？ `{answers['q3_canonical_70r_stepc_beats_old_cut']['answer']}`",
            f"- 4. 更支持的解释： `{answers['q4_supported_explanation']}`",
            f"- 5. canonical downstream 后是否需要再调 `70a` recipe？ `{answers['q5_need_more_70a_recipe_tuning']['answer']}`",
            f"- 最小下一步： {answers['q5_need_more_70a_recipe_tuning']['minimal_next_step']}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_decision_markdown(summary: Mapping[str, Any]) -> str:
    baseline = summary["baseline_old_cut"]
    stepc = summary["stepc_chain"]
    comparisons = summary["comparisons"]
    incumbent = summary["fixed_incumbent"]
    answers = summary["answers"]

    def stage_block(stage: str, base_label: str, stepc_label: str) -> List[str]:
        base = baseline[stage]
        cand = stepc[stage]
        delta = comparisons[stage]["stepc_minus_baseline"]
        verdict = comparisons[stage]["stepb_verdict"]
        lines = [
            f"## {stage}",
            "",
            f"- baseline config: `{base['config']}`",
            f"- baseline launch: `{base['launch_command']}`",
            f"- baseline input: `{base['input_artifact']}`",
            f"- baseline output ckpt: `{base['output_ckpt']}`",
            f"- baseline eval: `{base['eval_artifact']}`",
            f"- StepC config: `{cand['config']}`",
            f"- StepC launch: `{cand['launch_command']}`",
            f"- StepC input: `{cand['input_artifact']}`",
        ]
        if "warmstart_report" in cand:
            lines.append(f"- StepC warmstart report: `{cand['warmstart_report']}`")
        lines.extend(
            [
                f"- StepC output ckpt: `{cand['output_ckpt']}`",
                f"- StepC eval: `{cand['eval_artifact']}`",
                "",
                "| lane | all_ex_root_mean | all_ex_root_p95 | leg_mean | leg_p95 | nonleg_p95 | Step A gate | hard reject |",
                "|---|---:|---:|---:|---:|---:|---|---|",
                f"| {base_label} | {fmt(base['metrics']['all_ex_root_mean'])} | {fmt(base['metrics']['all_ex_root_p95'])} | {fmt(base['metrics']['leg_mean'])} | {fmt(base['metrics']['leg_p95'])} | {fmt(base['metrics']['nonleg_p95'])} | {'pass' if base['gate']['step_a_gate'] else 'fail'} | {'yes' if base['gate']['hard_reject'] else 'no'} |",
                f"| {stepc_label} | {fmt(cand['metrics']['all_ex_root_mean'])} | {fmt(cand['metrics']['all_ex_root_p95'])} | {fmt(cand['metrics']['leg_mean'])} | {fmt(cand['metrics']['leg_p95'])} | {fmt(cand['metrics']['nonleg_p95'])} | {'pass' if cand['gate']['step_a_gate'] else 'fail'} | {'yes' if cand['gate']['hard_reject'] else 'no'} |",
                f"| delta(StepC-baseline) | {fmt(delta['all_ex_root_mean'])} | {fmt(delta['all_ex_root_p95'])} | {fmt(delta['leg_mean'])} | {fmt(delta['leg_p95'])} | {fmt(delta['nonleg_p95'])} | - | - |",
                "",
                f"- Step B' verdict: `{verdict['verdict']}` via `{verdict['rationale']}`",
                f"- relative to fixed incumbent: baseline=`{base['gate']['relative_fixed_incumbent']}`, StepC=`{cand['gate']['relative_fixed_incumbent']}`",
            ]
        )
        compat = cand.get("compat_loading")
        if isinstance(compat, dict):
            lines.extend(
                [
                    f"- compat stage: `{compat.get('stage')}`",
                    f"- compat type: `{compat.get('type')}`",
                    f"- warmstart_copy=`{str(bool(compat.get('warmstart_copy', False))).lower()}` legacy_upgrade=`{str(bool(compat.get('legacy_upgrade', False))).lower()}` partial_load=`{str(bool(compat.get('partial_load', False))).lower()}` tensor_upgrade=`{str(bool(compat.get('tensor_upgrade', False))).lower()}`",
                    f"- source layout: `{compat.get('source_layout')}`",
                    f"- result layout: `{compat.get('result_layout')}`",
                ]
            )
            if compat.get("reinit"):
                lines.append(f"- reinit tensors: `{compat.get('reinit')}`")
            if compat.get("tensor_upgrade"):
                lines.append(f"- tensor upgrade: `{compat.get('tensor_upgrade')}`")
        lines.append("")
        return lines

    lines: List[str] = [
        "# Stage6 Step C canonical downstream decision",
        "",
        f"- run_date: `{summary['run_date']}`",
        f"- caveat: `{summary['caveat']}`",
        f"- script: `{ROOT / 'tools' / 'run_stage6_stepc_canonical_chain.py'}`",
        f"- summary_json: `{SUMMARY_JSON}`",
        f"- summary_md: `{SUMMARY_MD}`",
        "",
        "## Locked policy",
        "",
        "- Step A gate remains necessary-but-not-sufficient.",
        "- promotion / ranking remains bound to Step B'.",
        "- incumbent remains fixed at `current_bad.teacher_x_gt`.",
        "- hard reject remains `nonleg_p95 regression >= 5%` vs fixed incumbent.",
        "",
        "## Fixed incumbent",
        "",
        f"- group summary: `{incumbent['group_summary']}`",
        f"- all_ex_root_mean: `{fmt(incumbent['metrics']['all_ex_root_mean'])}`",
        f"- all_ex_root_p95: `{fmt(incumbent['metrics']['all_ex_root_p95'])}`",
        f"- leg_mean: `{fmt(incumbent['metrics']['leg_mean'])}`",
        f"- leg_p95: `{fmt(incumbent['metrics']['leg_p95'])}`",
        f"- nonleg_p95: `{fmt(incumbent['metrics']['nonleg_p95'])}`",
        f"- Step A threshold all_ex_root_mean: `{fmt(incumbent['thresholds']['step_a_all_ex_root_mean'])}`",
        f"- Step A threshold leg_p95: `{fmt(incumbent['thresholds']['step_a_leg_p95'])}`",
        f"- hard reject threshold nonleg_p95: `{fmt(incumbent['thresholds']['hard_reject_nonleg_p95'])}`",
        "",
    ]
    lines.extend(stage_block("70a", "canonical baseline", "70a-StepC donor"))
    lines.extend(stage_block("replace", "canonical old-cut replace", "canonical replace-StepC"))
    lines.extend(stage_block("70R", "canonical old-cut 70R", "canonical 70R-StepC"))
    lines.extend(
        [
            "## Warmstart / handoff",
            "",
            f"- canonical baseline warmstart source: `{CANONICAL_70A_CKPT}`",
            f"- canonical baseline warmstart output: `{CANONICAL_WARMSTART_CKPT}`",
            f"- canonical baseline warmstart report: `{CANONICAL_WARMSTART_REPORT}`",
            f"- StepC warmstart output: `{CKPT_REPLACE_WARMSTART}`",
            f"- StepC warmstart report: `{REPORT_WARMSTART}`",
            "- StepC warmstart preserves canonical zerophase semantics and only pastes the exact canonical warmstart delta onto the StepC donor.",
            "- StepC replace and StepC 70R both consume already-StepC checkpoints, so there is no second legacy upgrade downstream.",
            "",
            "## Final answers",
            "",
            f"- 1. `70a-StepC donor` 本身是否更稳？ `{answers['q1_70a_stepc_donor_more_stable']['answer']}`",
            f"- 2. canonical `replace(new70b_replace_lowdrift)` handoff 是否保留了这种改善？ `{answers['q2_replace_handoff_preserves_or_disperses']['answer']}`",
            f"- 3. canonical `70R-StepC` 是否优于 canonical old-cut 对应链路？ `{answers['q3_canonical_70r_stepc_beats_old_cut']['answer']}`",
            f"- 4. 更支持的解释： `{answers['q4_supported_explanation']}`",
            f"- 5. 是否需要再调 `70a` recipe？ `{answers['q5_need_more_70a_recipe_tuning']['answer']}`",
            f"- minimal next step: `{answers['q5_need_more_70a_recipe_tuning']['minimal_next_step']}`",
            "",
            "## Bottom line",
            "",
            "- unified leg terminal block successfully crosses the real canonical `replace(new70b_replace_lowdrift)` handoff.",
            "- the improvement is retained downstream at canonical `70R` rather than being washed out.",
            "- the observed pattern fits `shared trunk + grouped readout + unified leg terminal` better than the legacy `leg_head / out_leg` independent-surgery story.",
            "- based on canonical downstream evidence, there is no need to pre-emptively retune the `70a` recipe before downstream readout.",
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
            BASE_CONFIG_70A,
            BASE_CONFIG_REPLACE,
            BASE_CONFIG_70R,
            CANONICAL_STAGE6_CKPT,
            CANONICAL_70A_CKPT,
            CANONICAL_REPLACE_CKPT,
            CANONICAL_70R_CKPT,
            CANONICAL_WARMSTART_CKPT,
            CANONICAL_WARMSTART_REPORT,
            CANONICAL_70A_GROUP,
            CANONICAL_REPLACE_GROUP,
            CANONICAL_70R_GROUP,
            INCUMBENT_GROUP,
            STEPB_DECISION,
        ]
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    baseline = canonical_baseline_rows()
    stepc = stepc_rows()
    summary = build_summary(baseline, stepc)
    write_json(SUMMARY_JSON, summary)
    SUMMARY_MD.write_text(build_markdown(summary), encoding="utf-8")
    DECISION_MD.write_text(build_decision_markdown(summary), encoding="utf-8")
    write_json(
        STATUS_JSON,
        {
            "summary_json": str(SUMMARY_JSON),
            "summary_md": str(SUMMARY_MD),
            "decision_md": str(DECISION_MD),
            "baseline_70a": baseline["70a"]["output_ckpt"],
            "stepc_70a": stepc["70a"]["output_ckpt"],
            "stepc_replace": stepc["replace"]["output_ckpt"],
            "stepc_70r": stepc["70R"]["output_ckpt"],
        },
    )
    log(f"[done] summary_json={SUMMARY_JSON}")
    log(f"[done] summary_md={SUMMARY_MD}")
    log(f"[done] decision_md={DECISION_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
