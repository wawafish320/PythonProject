#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_strict_stageb_resolvedcfg_rerun as strict_ref  # noqa: E402

CPU_WRAPPER = ROOT / "debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py"
SUMMARY_TOOL = ROOT / "tools/phasea_group_summary.py"
TEACHER_JSON = ROOT / "validate/teacher_batches/Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models/motion_encoder_equiv.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json"
DEFAULT_CONFIG_ROOT = ROOT / "debug_output/_tmp_strict_stageB_finalstate_20260427_080658/stageB_strict/configs"
DEFAULT_SOURCE_STAGE6 = (
    ROOT
    / "debug_output/_tmp_legacy_ckpt_stage6final_rerun0425_20260429_001234/migrated_ckpts"
    / "stage6_final_strict_from0425.pth"
)
DEFAULT_DIRECT_POSE_DONOR_STEP0 = (
    ROOT
    / "debug_output/_tmp_strict_contract_fullchain_preflight_20260426_173158/70R_lr_probe/lr1e4_step20/checkpoints"
    / "ckpt_step_000000_WalkF_stage7_70R_lr1e4_step20_20260426_173158.pth"
)
REF_REPLACE_GROUP = ROOT / "debug_output/_tmp_tail_top7_fresh_chain_20260413_195656/replace_clean/eval_model_source_group_summary.json"
REF_LAMBDA_GROUP = ROOT / "debug_output/_tmp_tail_top7_fresh_chain_20260418_074813/lambda_clean/eval_model_source_group_summary.json"

RUN_NAMES = {
    "70a": "WalkF_stage7_70a_from_stage6final_strict",
    "replace": "WalkF_stage7_replace_from_stage6final_70a_concat",
    "70R": "WalkF_stage7_70R_from_stage6final_replace_concat",
    "71": "WalkF_stage7_71_from_stage6final_70R_concat",
    "72": "WalkF_stage7_72_from_stage6final_71_concat",
    "lambda": "WalkF_stage7_lambda_from_stage6final_72_concat",
}

COMMON_OVERRIDES = {
    "encoder_bundle": str(ENCODER_BUNDLE),
    "device": "cpu",
    "posttrain_contacts_source": "pretrain_contact",
    "posttrain_contacts_pretrain_clamp": "1.0",
    "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
    "direct_pose_stepc_unified_leg_terminal": True,
    "strict_current_model_build": True,
    "load_context": "chain_hop",
    "event_clock": "on",
    "width": 512,
}

STAGE_OVERRIDES = {
    "70a": {"lr": 3e-4},
    "replace": {"epochs": 3, "steps_per_epoch": 60, "lr": 5e-5},
    "70R": {
        "epochs": 1,
        "steps_per_epoch": 180,
        "lr": 1e-4,
        "save_step_ckpts": "0,1,5,20,60,180",
        "direct_pose_nonleg_train_only": True,
        "direct_pose_nonleg_trunk_mode": "full",
    },
    "71": {"lr": 3e-4},
    "72": {"lr": 1e-4},
    "lambda": {},
}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_root = ROOT / "debug_output" / f"_tmp_stage6final_canonical_downstream_{stamp}"
    ap = argparse.ArgumentParser(
        description="Run canonical downstream posttrain stages from a strict/current stage6-final checkpoint."
    )
    ap.add_argument("--source-stage6", type=Path, default=DEFAULT_SOURCE_STAGE6)
    ap.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    ap.add_argument("--direct-pose-donor-step0", type=Path, default=DEFAULT_DIRECT_POSE_DONOR_STEP0)
    ap.add_argument("--run-root", type=Path, default=default_run_root)
    ap.add_argument("--lr-70r", type=float, default=None)
    ap.add_argument("--stop-after", choices=("70a", "replace", "70R", "71", "72", "lambda"), default="lambda")
    ap.add_argument("--skip-eval", action="store_true")
    return ap.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def env(run_root: Path) -> dict[str, str]:
    out = dict(os.environ)
    out["PYTHONPATH"] = str(ROOT)
    out["PYTHONUNBUFFERED"] = "1"
    out["MPLBACKEND"] = "Agg"
    out["RUNTIME_SEMANTICS_TRACE_EVENTS"] = str(run_root / "runtime_semantics_events.jsonl")
    return out


def python_cmd() -> list[object]:
    if CPU_WRAPPER.is_file():
        return [CPU_WRAPPER]
    return [sys.executable]


def command_text(cmd: list[object]) -> str:
    return shlex.join([str(part) for part in cmd])


def run_logged(*, label: str, cmd: list[object], log_path: Path, run_root: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_list = [str(part) for part in cmd]
    started = time.time()
    print(f"\n[RUN:{label}] {command_text(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd_list,
            cwd=str(ROOT),
            env=env(run_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = int(proc.wait())
        log.write(f"\n[exit_code] {code}\n[elapsed_sec] {time.time() - started:.3f}\n")
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd_list)


def require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing required file: {path}")


def canonicalize_direct_pose_config_fields(payload: dict[str, Any]) -> None:
    hidden_override = payload.pop("direct_pose_hidden_override", None)
    if hidden_override is not None and "direct_pose_hidden" not in payload:
        payload["direct_pose_hidden"] = int(hidden_override)
    meas_mode_override = payload.pop("direct_pose_meas_mode_override", None)
    if meas_mode_override is not None and str(meas_mode_override).strip():
        payload["direct_pose_meas_mode"] = str(meas_mode_override).strip()
    elif "direct_pose_meas_mode" not in payload:
        payload["direct_pose_meas_mode"] = "concat"


def build_stage_config(
    *,
    name: str,
    ckpt_in: Path,
    config_root: Path,
    run_root: Path,
    args: argparse.Namespace,
) -> Path:
    payload = read_json(config_root / f"{name}.json")
    if not isinstance(payload, dict):
        raise RuntimeError(f"base config must be object: {config_root / f'{name}.json'}")
    payload = dict(payload)
    canonicalize_direct_pose_config_fields(payload)
    for key in ("legacy_checkpoint_compat", "contact_plan_init_hidden", "contact_plan_init_dropout"):
        payload.pop(key, None)
    if name != "replace":
        for key in ("direct_pose_use_phase_z", "direct_pose_phase_z_mode"):
            payload.pop(key, None)
    else:
        payload["direct_pose_use_phase_z"] = True
        payload["direct_pose_phase_z_mode"] = "concat"
    if name != "lambda":
        for key in tuple(payload.keys()):
            if str(key).startswith("lambda_fusion_"):
                payload.pop(str(key), None)
        payload["train_lambda_head"] = False
        payload.pop("lambda_fusion_enable", None)
    else:
        payload["train_lambda_head"] = True
        payload["lambda_fusion_enable"] = True
    payload.update(COMMON_OVERRIDES)
    payload.update(STAGE_OVERRIDES[name])
    if name == "70R" and args.lr_70r is not None:
        payload["lr"] = float(args.lr_70r)
    payload.update(
        {
            "ckpt_in": str(ckpt_in),
            "out_dir": str(run_root / name / "checkpoints"),
            "run_name": RUN_NAMES[name],
            "config_json": str(config_root / f"{name}.json"),
        }
    )
    config_path = run_root / "configs" / f"{name}.json"
    write_json(config_path, payload)
    return config_path


def expected_last(stage: str, run_root: Path) -> Path:
    return run_root / stage / "checkpoints" / f"ckpt_last_{RUN_NAMES[stage]}.pth"


def expected_step(stage: str, step: int, run_root: Path) -> Path:
    return run_root / stage / "checkpoints" / f"ckpt_step_{step:06d}_{RUN_NAMES[stage]}.pth"


def contractize(
    *,
    label: str,
    source_ckpt: Path,
    target_config: Path,
    out_ckpt: Path,
    run_root: Path,
    allow_missing_prefixes: tuple[str, ...] = (),
    tensor_donor: Path | None = None,
    transplant_prefixes: tuple[str, ...] = (),
) -> Path:
    cmd: list[object] = [
        sys.executable,
        ROOT / "tools/contractize_strict_posttrain_handoff.py",
        "--source",
        source_ckpt,
        "--target-config",
        target_config,
        "--out",
        out_ckpt,
        "--report",
        run_root / "handoffs" / f"{label}_contractize.json",
        "--label",
        label,
    ]
    for prefix in allow_missing_prefixes:
        cmd.extend(["--allow-missing-prefix", prefix])
    if tensor_donor is not None:
        cmd.extend(["--tensor-donor", tensor_donor])
        for prefix in transplant_prefixes:
            cmd.extend(["--transplant-prefix", prefix])
    run_logged(
        label=f"contractize:{label}",
        cmd=cmd,
        log_path=run_root / "logs" / f"contractize_{label}.log",
        run_root=run_root,
    )
    require_file(out_ckpt, label=f"{label}_handoff")
    require_file(run_root / "handoffs" / f"{label}_contractize.json", label=f"{label}_contractize_report")
    return out_ckpt


def strip_checkpoint_prefixes(*, source_ckpt: Path, output_ckpt: Path, prefixes: tuple[str, ...]) -> Path:
    payload = torch.load(source_ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint is not dict: {source_ckpt}")
    payload = dict(payload)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint model state missing: {source_ckpt}")
    payload["model"] = {k: v for k, v in state.items() if not any(str(k).startswith(prefix) for prefix in prefixes)}
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_ckpt)
    return output_ckpt


def eval_checkpoint(*, label: str, ckpt: Path, out_dir: Path, run_root: Path, lambda_apply: bool = False) -> Path:
    cmd: list[object] = [
        *python_cmd(),
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        TEACHER_JSON,
        "--model",
        ckpt,
        "--encoder-bundle",
        ENCODER_BUNDLE,
        "--out",
        out_dir,
        "--device",
        "cpu",
        "--contacts_meas_source",
        "model",
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
        "on",
        "--contact_plan_init_mode",
        "learnable",
        "--phase_reset_source",
        "none",
        "--force",
    ]
    if lambda_apply:
        cmd.append("--lambda_fusion_apply")
    run_logged(label=f"eval:{label}", cmd=cmd, log_path=run_root / "logs" / f"eval_{label}.log", run_root=run_root)
    eval_json = out_dir / "Walk_F_freerun_cycles.json"
    require_file(eval_json, label=f"eval_json_{label}")
    group_json = out_dir.parent / "group_summary.json"
    group_json.parent.mkdir(parents=True, exist_ok=True)
    run_logged(
        label=f"group:{label}",
        cmd=[sys.executable, SUMMARY_TOOL, eval_json, "--cycle_gte", "1", "--drop_wrap", "--out", group_json],
        log_path=run_root / "logs" / f"group_{label}.log",
        run_root=run_root,
    )
    require_file(group_json, label=f"group_json_{label}")
    return group_json


def ensure_inputs(args: argparse.Namespace) -> None:
    required = [
        args.source_stage6,
        args.config_root / "70a.json",
        args.config_root / "replace.json",
        args.config_root / "70R.json",
        args.config_root / "71.json",
        args.config_root / "72.json",
        args.config_root / "lambda.json",
        args.direct_pose_donor_step0,
        TEACHER_JSON,
        ENCODER_BUNDLE,
        AFFINE_STATS,
        SUMMARY_TOOL,
        REF_REPLACE_GROUP,
        REF_LAMBDA_GROUP,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))


def prepare_dirs(run_root: Path) -> None:
    if run_root.exists():
        raise FileExistsError(f"run_root already exists: {run_root}")
    for rel in ("", "configs", "logs", "handoffs", "evals"):
        (run_root / rel).mkdir(parents=True, exist_ok=True)


def run_chain(args: argparse.Namespace) -> dict[str, Any]:
    ensure_inputs(args)
    run_root = args.run_root.resolve()
    prepare_dirs(run_root)

    config_70a = build_stage_config(name="70a", ckpt_in=args.source_stage6, config_root=args.config_root, run_root=run_root, args=args)
    stage6_to_70a_report = strict_ref.build_direct_handoff_report(
        label="stage6final_to_70a",
        source_ckpt=args.source_stage6,
        target_config=config_70a,
        out_path=run_root / "handoffs" / "stage6final_to_70a_direct_report.json",
    )

    config_replace = build_stage_config(
        name="replace",
        ckpt_in=run_root / "handoffs" / "70a_to_replace_strict_contract.pth",
        config_root=args.config_root,
        run_root=run_root,
        args=args,
    )
    config_70r = build_stage_config(
        name="70R",
        ckpt_in=run_root / "handoffs" / "replace_to_70R_warmstart_bridge_strict_contract.pth",
        config_root=args.config_root,
        run_root=run_root,
        args=args,
    )
    config_71 = build_stage_config(
        name="71",
        ckpt_in=run_root / "handoffs" / "70R_to_71_strict_contract.pth",
        config_root=args.config_root,
        run_root=run_root,
        args=args,
    )
    config_72 = build_stage_config(name="72", ckpt_in=expected_last("71", run_root), config_root=args.config_root, run_root=run_root, args=args)
    config_lambda = build_stage_config(
        name="lambda",
        ckpt_in=run_root / "handoffs" / "72_to_lambda_strict_contract.pth",
        config_root=args.config_root,
        run_root=run_root,
        args=args,
    )

    config_manifest = {
        "70a": read_json(config_70a),
        "replace": read_json(config_replace),
        "70R": read_json(config_70r),
        "71": read_json(config_71),
        "72": read_json(config_72),
        "lambda": read_json(config_lambda),
    }
    write_json(run_root / "config_manifest.json", config_manifest)

    result: dict[str, Any] = {
        "run_root": str(run_root),
        "source_stage6": str(args.source_stage6),
        "direct_pose_donor_step0": str(args.direct_pose_donor_step0),
        "lr_70r": None if args.lr_70r is None else float(args.lr_70r),
        "stop_after": str(args.stop_after),
        "stage6final_to_70a_report": str(run_root / "handoffs" / "stage6final_to_70a_direct_report.json"),
    }

    run_logged(
        label="70a",
        cmd=[*python_cmd(), "-m", "train.posttrain", "--config", config_70a],
        log_path=run_root / "logs" / "70a.log",
        run_root=run_root,
    )
    require_file(expected_last("70a", run_root), label="70a_last")
    if args.stop_after == "70a":
        result["stage_outputs"] = {"70a": str(expected_last("70a", run_root))}
        write_json(run_root / "run_result.json", result)
        write_json(run_root / "handoff_summary.json", {"stage6final_to_70a": stage6_to_70a_report})
        return result

    replace_handoff = contractize(
        label="70a_to_replace",
        source_ckpt=expected_last("70a", run_root),
        target_config=config_replace,
        out_ckpt=run_root / "handoffs" / "70a_to_replace_strict_contract.pth",
        run_root=run_root,
    )
    replace_cfg = read_json(config_replace)
    replace_cfg["ckpt_in"] = str(replace_handoff)
    write_json(config_replace, replace_cfg)
    run_logged(
        label="replace",
        cmd=[*python_cmd(), "-m", "train.posttrain", "--config", config_replace],
        log_path=run_root / "logs" / "replace.log",
        run_root=run_root,
    )
    require_file(expected_last("replace", run_root), label="replace_last")
    if args.stop_after == "replace":
        result["stage_outputs"] = {
            "70a": str(expected_last("70a", run_root)),
            "replace": str(expected_last("replace", run_root)),
        }
        if not args.skip_eval:
            replace_group = eval_checkpoint(
                label="replace",
                ckpt=expected_last("replace", run_root),
                out_dir=run_root / "evals" / "replace" / "eval_model_source",
                run_root=run_root,
            )
            replace_compare = strict_ref.compare_group_summary(
                replace_group,
                REF_REPLACE_GROUP,
                baseline_label=str(REF_REPLACE_GROUP),
                current_label=str(replace_group),
            )
            write_json(run_root / "replace_vs_ref_compare.json", replace_compare)
            result.update(
                {
                    "replace_group_summary": str(replace_group),
                    "replace_vs_ref_compare": str(run_root / "replace_vs_ref_compare.json"),
                }
            )
        write_json(run_root / "run_result.json", result)
        write_json(run_root / "handoff_summary.json", {"stage6final_to_70a": stage6_to_70a_report})
        return result

    replace_without_direct_pose = strip_checkpoint_prefixes(
        source_ckpt=expected_last("replace", run_root),
        output_ckpt=run_root / "handoffs" / "replace_without_direct_pose_source.pth",
        prefixes=("direct_pose_",),
    )
    handoff_70r = contractize(
        label="replace_to_70R_warmstart_bridge",
        source_ckpt=replace_without_direct_pose,
        target_config=config_70r,
        out_ckpt=run_root / "handoffs" / "replace_to_70R_warmstart_bridge_strict_contract.pth",
        run_root=run_root,
        allow_missing_prefixes=("direct_pose_",),
        tensor_donor=args.direct_pose_donor_step0,
        transplant_prefixes=("direct_pose_",),
    )
    cfg_70r = read_json(config_70r)
    cfg_70r["ckpt_in"] = str(handoff_70r)
    write_json(config_70r, cfg_70r)
    run_logged(
        label="70R",
        cmd=[
            *python_cmd(),
            ROOT / "tools/run_posttrain_nonleg_trunk_ablation.py",
            "--config",
            config_70r,
            "--trunk-mode",
            "full",
            "--out-dir",
            run_root / "70R" / "checkpoints",
            "--run-name",
            RUN_NAMES["70R"],
            "--epochs",
            "1",
            "--steps-per-epoch",
            "180",
            "--save-step-ckpts",
            "0,1,5,20,60,180",
        ],
        log_path=run_root / "logs" / "70R.log",
        run_root=run_root,
    )
    require_file(expected_step("70R", 180, run_root), label="70R_step180")
    if not expected_last("70R", run_root).is_file():
        shutil.copy2(expected_step("70R", 180, run_root), expected_last("70R", run_root))
    if args.stop_after == "70R":
        result["stage_outputs"] = {
            "70a": str(expected_last("70a", run_root)),
            "replace": str(expected_last("replace", run_root)),
            "70R": str(expected_last("70R", run_root)),
            "70R_step20": str(expected_step("70R", 20, run_root)),
            "70R_step180": str(expected_step("70R", 180, run_root)),
        }
        if not args.skip_eval:
            replace_group = eval_checkpoint(
                label="replace",
                ckpt=expected_last("replace", run_root),
                out_dir=run_root / "evals" / "replace" / "eval_model_source",
                run_root=run_root,
            )
            group_70r_step20 = eval_checkpoint(
                label="70R_step20",
                ckpt=expected_step("70R", 20, run_root),
                out_dir=run_root / "evals" / "70R" / "step_000020" / "eval_model_source",
                run_root=run_root,
            )
            group_70r_step180 = eval_checkpoint(
                label="70R_step180",
                ckpt=expected_step("70R", 180, run_root),
                out_dir=run_root / "evals" / "70R" / "step_000180" / "eval_model_source",
                run_root=run_root,
            )
            replace_compare = strict_ref.compare_group_summary(
                replace_group,
                REF_REPLACE_GROUP,
                baseline_label=str(REF_REPLACE_GROUP),
                current_label=str(replace_group),
            )
            write_json(run_root / "replace_vs_ref_compare.json", replace_compare)
            result.update(
                {
                    "replace_group_summary": str(replace_group),
                    "replace_vs_ref_compare": str(run_root / "replace_vs_ref_compare.json"),
                    "70R_step20_group_summary": str(group_70r_step20),
                    "70R_step180_group_summary": str(group_70r_step180),
                }
            )
        write_json(run_root / "run_result.json", result)
        write_json(run_root / "handoff_summary.json", {"stage6final_to_70a": stage6_to_70a_report})
        return result

    handoff_71 = contractize(
        label="70R_to_71",
        source_ckpt=expected_step("70R", 180, run_root),
        target_config=config_71,
        out_ckpt=run_root / "handoffs" / "70R_to_71_strict_contract.pth",
        run_root=run_root,
    )
    cfg_71 = read_json(config_71)
    cfg_71["ckpt_in"] = str(handoff_71)
    write_json(config_71, cfg_71)
    run_logged(
        label="71",
        cmd=[*python_cmd(), "-m", "train.posttrain", "--config", config_71],
        log_path=run_root / "logs" / "71.log",
        run_root=run_root,
    )
    require_file(expected_last("71", run_root), label="71_last")
    if args.stop_after == "71":
        result["stage_outputs"] = {
            "70a": str(expected_last("70a", run_root)),
            "replace": str(expected_last("replace", run_root)),
            "70R": str(expected_last("70R", run_root)),
            "71": str(expected_last("71", run_root)),
        }
        write_json(run_root / "run_result.json", result)
        write_json(run_root / "handoff_summary.json", {"stage6final_to_70a": stage6_to_70a_report})
        return result

    cfg_72 = read_json(config_72)
    cfg_72["ckpt_in"] = str(expected_last("71", run_root))
    write_json(config_72, cfg_72)
    run_logged(
        label="72",
        cmd=[*python_cmd(), "-m", "train.posttrain", "--config", config_72],
        log_path=run_root / "logs" / "72.log",
        run_root=run_root,
    )
    require_file(expected_last("72", run_root), label="72_last")
    if args.stop_after == "72":
        result["stage_outputs"] = {
            "70a": str(expected_last("70a", run_root)),
            "replace": str(expected_last("replace", run_root)),
            "70R": str(expected_last("70R", run_root)),
            "71": str(expected_last("71", run_root)),
            "72": str(expected_last("72", run_root)),
        }
        write_json(run_root / "run_result.json", result)
        write_json(run_root / "handoff_summary.json", {"stage6final_to_70a": stage6_to_70a_report})
        return result

    handoff_lambda = contractize(
        label="72_to_lambda",
        source_ckpt=expected_last("72", run_root),
        target_config=config_lambda,
        out_ckpt=run_root / "handoffs" / "72_to_lambda_strict_contract.pth",
        run_root=run_root,
        allow_missing_prefixes=("lambda_fusion_head.",),
    )
    cfg_lambda = read_json(config_lambda)
    cfg_lambda["ckpt_in"] = str(handoff_lambda)
    write_json(config_lambda, cfg_lambda)
    run_logged(
        label="lambda",
        cmd=[*python_cmd(), "-m", "train.posttrain", "--config", config_lambda],
        log_path=run_root / "logs" / "lambda.log",
        run_root=run_root,
    )
    require_file(expected_last("lambda", run_root), label="lambda_last")
    result["stage_outputs"] = {
        "70a": str(expected_last("70a", run_root)),
        "replace": str(expected_last("replace", run_root)),
        "70R": str(expected_last("70R", run_root)),
        "71": str(expected_last("71", run_root)),
        "72": str(expected_last("72", run_root)),
        "lambda": str(expected_last("lambda", run_root)),
    }
    if not args.skip_eval:
        replace_group = eval_checkpoint(
            label="replace",
            ckpt=expected_last("replace", run_root),
            out_dir=run_root / "evals" / "replace" / "eval_model_source",
            run_root=run_root,
        )
        lambda_group = eval_checkpoint(
            label="lambda",
            ckpt=expected_last("lambda", run_root),
            out_dir=run_root / "evals" / "lambda" / "eval_model_source",
            run_root=run_root,
            lambda_apply=True,
        )
        replace_compare = strict_ref.compare_group_summary(
            replace_group,
            REF_REPLACE_GROUP,
            baseline_label=str(REF_REPLACE_GROUP),
            current_label=str(replace_group),
        )
        lambda_compare = strict_ref.compare_group_summary(
            lambda_group,
            REF_LAMBDA_GROUP,
            baseline_label=str(REF_LAMBDA_GROUP),
            current_label=str(lambda_group),
        )
        write_json(run_root / "replace_vs_ref_compare.json", replace_compare)
        write_json(run_root / "lambda_vs_ref_compare.json", lambda_compare)
        result.update(
            {
                "replace_group_summary": str(replace_group),
                "lambda_group_summary": str(lambda_group),
                "replace_vs_ref_compare": str(run_root / "replace_vs_ref_compare.json"),
                "lambda_vs_ref_compare": str(run_root / "lambda_vs_ref_compare.json"),
            }
        )
    write_json(run_root / "run_result.json", result)
    write_json(run_root / "handoff_summary.json", {"stage6final_to_70a": stage6_to_70a_report})
    return result


def main() -> int:
    args = parse_args()
    result = run_chain(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
