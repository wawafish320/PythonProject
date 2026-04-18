#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent
STAMP = RUN_ROOT.name.removeprefix("_tmp_tail_top7_fresh_chain_")
RUN_TAG = f"tail_top7_fresh_chain_{STAMP}"
MODEL_ROOT = ROOT / "models" / f"__tmp_{RUN_TAG}"

CPU_WRAPPER = ROOT / "debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py"
BASE_CONFIG = ROOT / "config/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401.json"
ENCODER_BUNDLE = ROOT / "models/motion_encoder_equiv.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json"
CONTACT_SOURCE = "pretrain_contact"
CONTACT_CLAMP = "1.0"

FROZEN_CFG_ROOT = ROOT / "debug_output/_tmp_top7_clean_stage6_stepc_chain_20260412/configs"
PT_CFG_STAGE6_BASE = FROZEN_CFG_ROOT / "posttrain_stage6_tailfix_top7_clean_stepc_20260412.json"
PT_CFG_70A_BASE = FROZEN_CFG_ROOT / "posttrain_70a_from_top7_stage6_clean_stepc_20260412.json"
PT_CFG_REPLACE_BASE = FROZEN_CFG_ROOT / "posttrain_replace_from_top7_70a_clean_stepc_20260412.json"
PT_CFG_70R_BASE = FROZEN_CFG_ROOT / "posttrain_70R_from_top7_replace_clean_stepc_20260412.json"
PT_CFG_71_BASE = FROZEN_CFG_ROOT / "posttrain_71_from_top7_70R_clean_stepc_20260412.json"
PT_CFG_72_BASE = FROZEN_CFG_ROOT / "posttrain_72_from_top7_71_clean_stepc_20260412.json"
PT_CFG_LAMBDA_BASE = FROZEN_CFG_ROOT / "posttrain_lambda_from_top7_72_clean_stepc_20260412.json"

BASELINE_ROOT = ROOT / "debug_output/_tmp_tail_top7_fresh_chain_20260413_195656"
BASELINE_SUMMARY = BASELINE_ROOT / "lambda_clean/eval_model_source_group_summary.json"
PREV_MIGRATED_RUNNER = ROOT / "debug_output/_tmp_tail_top7_fresh_chain_20260416_212417/run_posttrain_nonleg_trunk_ablation_migrated.py"

CONFIG_DIR = RUN_ROOT / "configs"
LOG_DIR = RUN_ROOT / "logs"
LAMBDA_EVAL_DIR = RUN_ROOT / "lambda_clean/eval_model_source"
LAMBDA_EVAL_JSON = LAMBDA_EVAL_DIR / "Walk_F_freerun_cycles.json"
LAMBDA_GROUP = RUN_ROOT / "lambda_clean/eval_model_source_group_summary.json"
REGRESSION_JSON = RUN_ROOT / "lambda_clean/regression_vs_20260413.json"
CONFIG_COMPARE_JSON = RUN_ROOT / "config_compare_vs_20260413.json"
RUN_CONTEXT_JSON = RUN_ROOT / "run_context.json"

BASETRAIN_CFG = CONFIG_DIR / "basetrain_runtime.json"
BASETRAIN_RUN_NAME = f"fresh_tail_top7_basetrain_{STAMP}"
BASETRAIN_OUT_ROOT = MODEL_ROOT / "basetrain"
BASETRAIN_OUT_DIR = BASETRAIN_OUT_ROOT / BASETRAIN_RUN_NAME
BASETRAIN_CKPT = BASETRAIN_OUT_DIR / f"ckpt_last_{BASETRAIN_RUN_NAME}.pth"
BASETRAIN_FREERUN_DIAG = RUN_ROOT / "basetrain" / f"freerun_diag_{BASETRAIN_RUN_NAME}.pt"

STAGE6_RUN_NAME = f"stage6_tailfix_top7_stepc_clean_fromfresh_{STAMP}"
STAGE6_OUT_DIR = MODEL_ROOT / "stage6_stepc_handoff"
STAGE6_CFG = CONFIG_DIR / f"stage6_{STAMP}.json"
STAGE6_CKPT = STAGE6_OUT_DIR / f"ckpt_last_{STAGE6_RUN_NAME}.pth"

S70A_RUN_NAME = f"WalkF_stage7_70a_from_fresh_tailk7_stage6stepc_clean_{STAMP}"
S70A_OUT_DIR = MODEL_ROOT / "70a_clean"
S70A_CFG = CONFIG_DIR / f"70a_{STAMP}.json"
S70A_CKPT = S70A_OUT_DIR / f"ckpt_last_{S70A_RUN_NAME}.pth"

WARMSTART_OUT_DIR = MODEL_ROOT / "warmstart_clean"
WARMSTART_CKPT = WARMSTART_OUT_DIR / f"ckpt_last_fresh_tail_top7_70a_replace_zerophase_cleanstepc_{STAMP}.pth"

REPLACE_RUN_NAME = f"WalkF_stage7_70b_replace_lowdrift_e3x60_lr5e5_from_fresh_tailk7_70a_cleanstepc_{STAMP}"
REPLACE_OUT_DIR = MODEL_ROOT / "replace_clean"
REPLACE_CFG = CONFIG_DIR / f"replace_{STAMP}.json"
REPLACE_CKPT = REPLACE_OUT_DIR / f"ckpt_last_{REPLACE_RUN_NAME}.pth"

S70R_RUN_NAME = f"WalkF_stage7_70R_from_fresh_tailk7_replace_cleanstepc_s180_{STAMP}"
S70R_OUT_DIR = MODEL_ROOT / "70R_clean"
S70R_CFG = CONFIG_DIR / f"70R_{STAMP}.json"
S70R_CKPT = S70R_OUT_DIR / f"ckpt_last_{S70R_RUN_NAME}.pth"

S71_RUN_NAME = f"WalkF_stage7_71_from_fresh_70R_cleanstepc_lr3e4_{STAMP}"
S71_OUT_DIR = MODEL_ROOT / "71_clean"
S71_CFG = CONFIG_DIR / f"71_{STAMP}.json"
S71_CKPT = S71_OUT_DIR / f"ckpt_last_{S71_RUN_NAME}.pth"

S72_RUN_NAME = f"WalkF_stage7_72_from_fresh_71_cleanstepc_lr1e4_{STAMP}"
S72_OUT_DIR = MODEL_ROOT / "72_clean"
S72_CFG = CONFIG_DIR / f"72_{STAMP}.json"
S72_CKPT = S72_OUT_DIR / f"ckpt_last_{S72_RUN_NAME}.pth"

LAMBDA_RUN_NAME = f"WalkF_stage7_lambda_from_fresh_72_cleanstepc_{STAMP}"
LAMBDA_OUT_DIR = MODEL_ROOT / "lambda_clean"
LAMBDA_CFG = CONFIG_DIR / f"lambda_{STAMP}.json"
LAMBDA_CKPT = LAMBDA_OUT_DIR / f"ckpt_last_{LAMBDA_RUN_NAME}.pth"

STAGE_SPECS: dict[str, dict[str, Any]] = {}
LAUNCH_COMMANDS: dict[str, list[str]] = {}


def ensure_inputs() -> None:
    required = [
        CPU_WRAPPER,
        BASE_CONFIG,
        ENCODER_BUNDLE,
        AFFINE_STATS,
        PT_CFG_STAGE6_BASE,
        PT_CFG_70A_BASE,
        PT_CFG_REPLACE_BASE,
        PT_CFG_70R_BASE,
        PT_CFG_71_BASE,
        PT_CFG_72_BASE,
        PT_CFG_LAMBDA_BASE,
        BASELINE_SUMMARY,
        PREV_MIGRATED_RUNNER,
        ROOT / "validate/teacher_batches/Walk_F_teacher.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def env() -> dict[str, str]:
    out = os.environ.copy()
    out["PYTHONPATH"] = str(ROOT)
    return out


def command_text(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def run_logged(stage: str, cmd: list[str], log_path: Path) -> None:
    LAUNCH_COMMANDS[stage] = [str(part) for part in cmd]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN:{stage}] {command_text(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=ROOT,
            env=env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
        log.write(f"\n[exit_code] {rc}\n")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def git_status(path: Path) -> None:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    path.write_text(proc.stdout, encoding="utf-8")


def prepare_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "lambda_clean").mkdir(parents=True, exist_ok=True)


def generate_basetrain_config() -> None:
    payload = read_json(BASE_CONFIG)
    for key in [
        "save_fit_ckpt_epochs",
        "seed",
        "rot_local_tail_rank_mix",
        "rot_local_tail_reduce",
        "rot_local_tail_uniform_mix",
        "trainbase_contacts_source",
        "adaptive_bone_weights",
    ]:
        payload.pop(key, None)
    payload["out"] = str(BASETRAIN_OUT_ROOT)
    payload["run_name"] = BASETRAIN_RUN_NAME
    payload["freerun_debug_path"] = str(BASETRAIN_FREERUN_DIAG)
    payload["amp"] = False
    payload["config_json"] = str(BASE_CONFIG)
    write_json(BASETRAIN_CFG, payload)


def generate_posttrain_configs() -> None:
    common = {
        "encoder_bundle": str(ENCODER_BUNDLE),
        "device": "cpu",
        "posttrain_contacts_source": CONTACT_SOURCE,
        "posttrain_contacts_pretrain_clamp": CONTACT_CLAMP,
        "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        "direct_pose_stepc_unified_leg_terminal": True,
    }
    specs = [
        (PT_CFG_STAGE6_BASE, STAGE6_CFG, {"ckpt_in": str(BASETRAIN_CKPT), "out_dir": str(STAGE6_OUT_DIR), "run_name": STAGE6_RUN_NAME}),
        (PT_CFG_70A_BASE, S70A_CFG, {"ckpt_in": str(STAGE6_CKPT), "out_dir": str(S70A_OUT_DIR), "run_name": S70A_RUN_NAME, "lr": 3e-4}),
        (PT_CFG_REPLACE_BASE, REPLACE_CFG, {"ckpt_in": str(WARMSTART_CKPT), "out_dir": str(REPLACE_OUT_DIR), "run_name": REPLACE_RUN_NAME, "epochs": 3, "steps_per_epoch": 60, "lr": 5e-5}),
        (PT_CFG_70R_BASE, S70R_CFG, {"ckpt_in": str(REPLACE_CKPT), "out_dir": str(S70R_OUT_DIR), "run_name": S70R_RUN_NAME, "epochs": 1, "lr": 3e-4}),
        (PT_CFG_71_BASE, S71_CFG, {"ckpt_in": str(S70R_CKPT), "out_dir": str(S71_OUT_DIR), "run_name": S71_RUN_NAME, "lr": 3e-4}),
        (PT_CFG_72_BASE, S72_CFG, {"ckpt_in": str(S71_CKPT), "out_dir": str(S72_OUT_DIR), "run_name": S72_RUN_NAME, "lr": 1e-4}),
        (PT_CFG_LAMBDA_BASE, LAMBDA_CFG, {"ckpt_in": str(S72_CKPT), "out_dir": str(LAMBDA_OUT_DIR), "run_name": LAMBDA_RUN_NAME}),
    ]
    for base_path, out_path, overrides in specs:
        payload = read_json(base_path)
        payload.update(common)
        payload.update(overrides)
        write_json(out_path, payload)

    s70r = read_json(S70R_CFG)
    expected = {
        "direct_pose_nonleg_train_only": True,
        "train_direct_pose": True,
        "direct_pose_nonleg_proj_dim": 256,
    }
    for key, value in expected.items():
        if s70r.get(key) != value:
            raise AssertionError(f"70R recipe drift: {key}={s70r.get(key)!r}, expected {value!r}")


def prepare_migrated_runner() -> Path:
    out = RUN_ROOT / "run_posttrain_nonleg_trunk_ablation_migrated.py"
    shutil.copy2(PREV_MIGRATED_RUNNER, out)
    out.chmod(0o755)
    return out


def require_file(stage: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{stage} did not produce expected file: {path}")


def run_chain(migrated_runner: Path) -> None:
    run_logged(
        "basetrain",
        [str(CPU_WRAPPER), "-m", "train.training_MPL", "--config_json", str(BASETRAIN_CFG)],
        LOG_DIR / "basetrain.log",
    )
    require_file("basetrain", BASETRAIN_CKPT)

    posttrain_stages = [
        ("stage6", STAGE6_CFG, STAGE6_CKPT, LOG_DIR / "stage6.log"),
        ("70a", S70A_CFG, S70A_CKPT, LOG_DIR / "70a.log"),
    ]
    for name, cfg, ckpt, log_path in posttrain_stages:
        run_logged(name, [str(CPU_WRAPPER), "-m", "train.posttrain", "--config", str(cfg)], log_path)
        require_file(name, ckpt)

    WARMSTART_OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(S70A_CKPT, WARMSTART_CKPT)
    require_file("warmstart_clean", WARMSTART_CKPT)
    LAUNCH_COMMANDS["warmstart_clean"] = ["cp", str(S70A_CKPT), str(WARMSTART_CKPT)]

    run_logged("replace", [str(CPU_WRAPPER), "-m", "train.posttrain", "--config", str(REPLACE_CFG)], LOG_DIR / "replace.log")
    require_file("replace", REPLACE_CKPT)

    run_logged(
        "70R",
        [
            str(CPU_WRAPPER),
            str(migrated_runner),
            "--config",
            str(S70R_CFG),
            "--trunk-mode",
            "full",
            "--out-dir",
            str(S70R_OUT_DIR),
            "--run-name",
            S70R_RUN_NAME,
            "--epochs",
            "1",
            "--steps-per-epoch",
            "180",
            "--save-step-ckpts",
            "0,1,5,20,60,180",
        ],
        LOG_DIR / "70R.log",
    )
    require_file("70R", S70R_CKPT)

    for name, cfg, ckpt in [
        ("71", S71_CFG, S71_CKPT),
        ("72", S72_CFG, S72_CKPT),
        ("lambda", LAMBDA_CFG, LAMBDA_CKPT),
    ]:
        run_logged(name, [str(CPU_WRAPPER), "-m", "train.posttrain", "--config", str(cfg)], LOG_DIR / f"{name}.log")
        require_file(name, ckpt)


def run_lambda_eval() -> None:
    run_logged(
        "lambda_eval",
        [
            str(CPU_WRAPPER),
            "-m",
            "train.validate.run_freerun_cycles",
            "--teacher",
            str(ROOT / "validate/teacher_batches/Walk_F_teacher.json"),
            "--model",
            str(LAMBDA_CKPT),
            "--encoder-bundle",
            str(ENCODER_BUNDLE),
            "--out",
            str(LAMBDA_EVAL_DIR),
            "--device",
            "cpu",
            "--contacts_meas_source",
            "model",
            "--lambda_fusion_apply",
            "--log_contacts",
            "--export_direct_arm_probe",
            "--export_joint_direct_geolocal_series",
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
            "--force",
        ],
        LOG_DIR / "lambda_eval.log",
    )
    require_file("lambda_eval", LAMBDA_EVAL_JSON)

    run_logged(
        "lambda_group_summary",
        [
            sys.executable,
            str(ROOT / "tools/phasea_group_summary.py"),
            str(LAMBDA_EVAL_JSON),
            "--cycle_gte",
            "1",
            "--drop_wrap",
            "--out",
            str(LAMBDA_GROUP),
        ],
        LOG_DIR / "lambda_group_summary.log",
    )
    require_file("lambda_group_summary", LAMBDA_GROUP)


def groups_from_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"group summary missing groups: {path}")
    return groups


def write_regression_json() -> dict[str, Any]:
    baseline = groups_from_summary(BASELINE_SUMMARY)
    current = groups_from_summary(LAMBDA_GROUP)
    groups: dict[str, Any] = {}
    any_worse = False
    primary_worse = False
    for group in ["all_ex_root", "leg", "nonleg", "arm", "else"]:
        row: dict[str, Any] = {}
        for metric in ["mean", "p50", "p90", "p95"]:
            b = baseline[group][metric]
            c = current[group][metric]
            delta = c - b
            if delta > 0:
                any_worse = True
                if group == "all_ex_root":
                    primary_worse = True
            row[metric] = {
                "baseline": b,
                "current": c,
                "delta": delta,
                "relative_percent": None if b == 0 else (delta / b * 100.0),
            }
        row["samples"] = {
            "baseline": baseline[group].get("samples"),
            "current": current[group].get("samples"),
        }
        groups[group] = row
    out = {
        "baseline": str(BASELINE_SUMMARY.relative_to(ROOT)),
        "current": str(LAMBDA_GROUP.relative_to(ROOT)),
        "higher_is_worse": True,
        "strict_any_metric_worse": any_worse,
        "strict_all_ex_root_worse": primary_worse,
        "groups": groups,
    }
    write_json(REGRESSION_JSON, out)
    return out


def normalize_for_config_compare(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: normalize_for_config_compare(value) for key, value in sorted(obj.items())}
    if isinstance(obj, list):
        return [normalize_for_config_compare(value) for value in obj]
    if isinstance(obj, str):
        text = obj.replace(str(ROOT) + "/", "")
        text = re.sub(r"202604\d{2}_\d{6}", "{STAMP}", text)
        return text
    return obj


def diff_values(a: Any, b: Any, path: str = "") -> list[dict[str, Any]]:
    if type(a) is not type(b):
        return [{"path": path, "baseline": a, "current": b, "kind": "type_or_value"}]
    if isinstance(a, dict):
        out: list[dict[str, Any]] = []
        keys = sorted(set(a) | set(b))
        for key in keys:
            child = f"{path}.{key}" if path else key
            if key not in a:
                out.append({"path": child, "baseline": "<MISSING>", "current": b[key], "kind": "missing_baseline"})
            elif key not in b:
                out.append({"path": child, "baseline": a[key], "current": "<MISSING>", "kind": "missing_current"})
            else:
                out.extend(diff_values(a[key], b[key], child))
        return out
    if isinstance(a, list):
        out = []
        if len(a) != len(b):
            out.append({"path": f"{path}.__len__", "baseline": len(a), "current": len(b), "kind": "length"})
        for idx, (av, bv) in enumerate(zip(a, b)):
            out.extend(diff_values(av, bv, f"{path}[{idx}]"))
        return out
    if a != b:
        return [{"path": path, "baseline": a, "current": b, "kind": "value"}]
    return []


def stage_config_path(config_root: Path, stage: str) -> Path:
    matches = sorted(config_root.glob(f"{stage}_*.json"))
    if not matches:
        raise FileNotFoundError(f"missing {stage} config under {config_root}")
    return matches[0]


def write_config_compare_json() -> dict[str, Any]:
    baseline_cfg_root = BASELINE_ROOT / "configs"
    posttrain: dict[str, Any] = {}
    posttrain_ok = True
    for stage in ["stage6", "70a", "replace", "70R", "71", "72", "lambda"]:
        baseline_cfg = stage_config_path(baseline_cfg_root, stage)
        current_cfg = stage_config_path(CONFIG_DIR, stage)
        baseline_norm = normalize_for_config_compare(read_json(baseline_cfg))
        current_norm = normalize_for_config_compare(read_json(current_cfg))
        diffs = diff_values(baseline_norm, current_norm)
        allowed = []
        unexpected = diffs
        if stage == "70R":
            allowed = [d for d in diffs if d["path"] == "device" and d["baseline"] == "auto" and d["current"] == "cpu"]
            unexpected = [d for d in diffs if d not in allowed]
        if unexpected:
            posttrain_ok = False
        posttrain[stage] = {
            "baseline_config": str(baseline_cfg.relative_to(ROOT)),
            "current_config": str(current_cfg.relative_to(ROOT)),
            "diff_count": len(diffs),
            "allowed_semantic_diff_count": len(allowed),
            "unexpected_diff_count": len(unexpected),
            "allowed_semantic_diffs": allowed,
            "unexpected_diffs": unexpected[:100],
        }

    baseline_base = baseline_cfg_root / "basetrain_sanitized.json"
    current_base = BASETRAIN_CFG
    base_norm = normalize_for_config_compare(read_json(baseline_base))
    curr_norm = normalize_for_config_compare(read_json(current_base))
    base_diffs = diff_values(base_norm, curr_norm)
    only_baseline = sorted(set(base_norm) - set(curr_norm)) if isinstance(base_norm, dict) and isinstance(curr_norm, dict) else []
    only_current = sorted(set(curr_norm) - set(base_norm)) if isinstance(base_norm, dict) and isinstance(curr_norm, dict) else []
    common_value_diffs = [
        diff for diff in base_diffs
        if diff["kind"] == "value" and "." not in diff["path"] and "[" not in diff["path"]
    ]

    out = {
        "baseline_run": str(BASELINE_ROOT.relative_to(ROOT)),
        "current_run": str(RUN_ROOT.relative_to(ROOT)),
        "normalization": [
            "absolute repo root stripped from paths",
            "202604DD_HHMMSS timestamps replaced with {STAMP}",
        ],
        "posttrain_configs_match_except_paths_timestamps": posttrain_ok,
        "posttrain": posttrain,
        "70R_semantic_device_only": (
            posttrain["70R"]["diff_count"] == posttrain["70R"]["allowed_semantic_diff_count"]
            and posttrain["70R"]["allowed_semantic_diff_count"] == 1
        ),
        "basetrain_runtime_vs_20260413_sanitized": {
            "baseline_config": str(baseline_base.relative_to(ROOT)),
            "current_config": str(current_base.relative_to(ROOT)),
            "match_after_normalization": len(base_diffs) == 0,
            "diff_count": len(base_diffs),
            "only_baseline_keys": only_baseline,
            "only_current_keys": only_current,
            "common_top_level_value_diffs": common_value_diffs[:100],
        },
    }
    write_json(CONFIG_COMPARE_JSON, out)
    return out


def metrics_from_group_summary(path: Path) -> dict[str, Any]:
    groups = groups_from_summary(path)
    return {
        name: {metric: groups[name].get(metric) for metric in ["mean", "p50", "p90", "p95", "samples"]}
        for name in ["all_ex_root", "leg", "nonleg", "arm", "else"]
    }


def write_run_context(regression: dict[str, Any], config_compare: dict[str, Any], migrated_runner: Path) -> None:
    def stage_row(
        name: str,
        config: Path | None,
        input_ckpt: Path | None,
        output_ckpt: Path | None,
        log: Path | None,
        group_summary: Path | None = None,
        eval_artifact: Path | None = None,
    ) -> dict[str, Any]:
        return {
            "success": bool(output_ckpt and output_ckpt.is_file()) if output_ckpt else True,
            "launch_command": command_text(LAUNCH_COMMANDS.get(name, [])),
            "input_ckpt": str(input_ckpt) if input_ckpt else None,
            "output_ckpt": str(output_ckpt) if output_ckpt else None,
            "config": str(config) if config else None,
            "log": str(log) if log else None,
            "eval_artifact": str(eval_artifact) if eval_artifact else None,
            "group_summary": str(group_summary) if group_summary else None,
        }

    context = {
        "run_root": str(RUN_ROOT),
        "model_root": str(MODEL_ROOT),
        "source_runbook": "docs/basetrain_to_posttrain_top7_fresh_chain_runbook.md",
        "baseline_group_summary": str(BASELINE_SUMMARY.relative_to(ROOT)),
        "current_lambda_eval": str(LAMBDA_EVAL_JSON),
        "current_lambda_group_summary": str(LAMBDA_GROUP),
        "regression_compare": str(REGRESSION_JSON),
        "config_compare": str(CONFIG_COMPARE_JSON),
        "repo_status_initial": str(RUN_ROOT / "git_status_initial.txt"),
        "repo_status_final": str(RUN_ROOT / "git_status_final.txt"),
        "runtime_contract_notes": [
            "posttrain stages launched config-only: no --posttrain_contacts_source CLI compatibility was added or used.",
            "PYTHONPATH is set to repo root for every cpu_nomps_exec.py invocation.",
            "70R used a run-root migrated runner to preserve nonleg_train_only=true, trunk-mode full, epochs=1, steps_per_epoch=180, save_step_ckpts=0,1,5,20,60,180 without patching tools/ or train/.",
            "basetrain randomness remains a known legacy issue and was not fixed in this run.",
            "basetrain runtime copy additionally strips adaptive_bone_weights because current train.training_MPL rejects it; this is recorded as current BASE_CONFIG/runtime drift, not a posttrain recipe change.",
        ],
        "stages": {
            "basetrain": stage_row("basetrain", BASETRAIN_CFG, None, BASETRAIN_CKPT, LOG_DIR / "basetrain.log"),
            "stage6": stage_row("stage6", STAGE6_CFG, BASETRAIN_CKPT, STAGE6_CKPT, LOG_DIR / "stage6.log"),
            "70a": stage_row("70a", S70A_CFG, STAGE6_CKPT, S70A_CKPT, LOG_DIR / "70a.log"),
            "warmstart_clean": stage_row("warmstart_clean", None, S70A_CKPT, WARMSTART_CKPT, None),
            "replace": stage_row("replace", REPLACE_CFG, WARMSTART_CKPT, REPLACE_CKPT, LOG_DIR / "replace.log"),
            "70R": stage_row("70R", S70R_CFG, REPLACE_CKPT, S70R_CKPT, LOG_DIR / "70R.log"),
            "71": stage_row("71", S71_CFG, S70R_CKPT, S71_CKPT, LOG_DIR / "71.log"),
            "72": stage_row("72", S72_CFG, S71_CKPT, S72_CKPT, LOG_DIR / "72.log"),
            "lambda": stage_row("lambda", LAMBDA_CFG, S72_CKPT, LAMBDA_CKPT, LOG_DIR / "lambda.log", LAMBDA_GROUP, LAMBDA_EVAL_JSON),
            "lambda_eval": stage_row("lambda_eval", None, LAMBDA_CKPT, LAMBDA_EVAL_JSON, LOG_DIR / "lambda_eval.log", LAMBDA_GROUP, LAMBDA_EVAL_JSON),
        },
        "lambda_metrics": metrics_from_group_summary(LAMBDA_GROUP),
        "regression_flags": {
            "strict_any_metric_worse": regression["strict_any_metric_worse"],
            "strict_all_ex_root_worse": regression["strict_all_ex_root_worse"],
        },
        "config_compare_summary": {
            "posttrain_configs_match_except_paths_timestamps": config_compare["posttrain_configs_match_except_paths_timestamps"],
            "70R_semantic_device_only": config_compare["70R_semantic_device_only"],
            "basetrain_runtime_match_20260413_sanitized": config_compare["basetrain_runtime_vs_20260413_sanitized"]["match_after_normalization"],
            "basetrain_only_baseline_key_count": len(config_compare["basetrain_runtime_vs_20260413_sanitized"]["only_baseline_keys"]),
        },
        "70R_migrated_runner": str(migrated_runner.relative_to(ROOT)),
    }
    write_json(RUN_CONTEXT_JSON, context)


def main() -> None:
    print(f"[init] root={ROOT}")
    print(f"[init] run_root={RUN_ROOT}")
    print(f"[init] model_root={MODEL_ROOT}")
    prepare_dirs()
    ensure_inputs()
    git_status(RUN_ROOT / "git_status_initial.txt")
    generate_basetrain_config()
    generate_posttrain_configs()
    migrated_runner = prepare_migrated_runner()
    run_chain(migrated_runner)
    run_lambda_eval()
    regression = write_regression_json()
    config_compare = write_config_compare_json()
    git_status(RUN_ROOT / "git_status_final.txt")
    write_run_context(regression, config_compare, migrated_runner)
    print(f"[OK] run_context={RUN_CONTEXT_JSON}")
    print(f"[OK] lambda_group={LAMBDA_GROUP}")
    print(f"[OK] regression={REGRESSION_JSON}")


if __name__ == "__main__":
    main()
