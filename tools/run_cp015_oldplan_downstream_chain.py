#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

try:
    from train.models import DEFAULT_DIRECT_POSE_LEG_BONES, STAGE6_3WAY_ARMCHAIN_BONES
except Exception:
    DEFAULT_DIRECT_POSE_LEG_BONES = (
        "thigh_r",
        "calf_r",
        "foot_r",
        "ball_r",
        "thigh_l",
        "calf_l",
        "foot_l",
        "ball_l",
    )
    STAGE6_3WAY_ARMCHAIN_BONES = (
        "clavicle_l",
        "upperarm_l",
        "RUpArmTwist_l_01",
        "RUpArmTwist_l_02",
        "lowerarm_l",
        "L_ForeTwist_01",
        "L_ForeTwist_02",
        "hand_l",
        "index_01_l",
        "middle_01_l",
        "ring_01_l",
        "pinky_01_l",
        "thumb_01_l",
        "clavicle_r",
        "upperarm_r",
        "RUpArmTwist_r_01",
        "RUpArmTwist_r_02",
        "lowerarm_r",
        "R_ForeTwist_01",
        "R_ForeTwist_02",
        "hand_r",
        "index_01_r",
        "middle_01_r",
        "ring_01_r",
        "pinky_01_r",
        "thumb_01_r",
    )


ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "20260314"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_oldplan_downstream_chain_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_cp015_oldplan_downstream_chain_{RUN_DATE}"
STAGE6_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_stage6_plantransplant_20260314" / "summary.json"
TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
PRETRAIN_CLAMP = "1.0"

CONFIG_70A = ROOT / "config" / "posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json"
CONFIG_70B = ROOT / "config" / "posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json"
CONFIG_70R = ROOT / "config" / "posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json"
CONFIG_71 = ROOT / "config" / "posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json"
CONFIG_72 = ROOT / "config" / "posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json"
CONFIG_LAMBDA = ROOT / "config" / "posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json"

ACCEPTED_DIRECT_SUMMARY = ROOT / "debug_output" / "_tmp_chain_s180promote_20260308" / "compare_vs_accepted_r5_direct" / "global_signal_summary.txt"
ACCEPTED_BLEND_SUMMARY = ROOT / "debug_output" / "_tmp_chain_s180promote_20260308" / "compare_vs_accepted_r5_blend" / "summary_metrics.txt"
ACCEPTED_CHAIN_VERDICT = ROOT / "debug_output" / "_tmp_chain_s180promote_20260308" / "chain_verdict.md"
EVALON_DIRECT_SUMMARY = ROOT / "debug_output" / "_tmp_chain_s180promote_20260308" / "compare_vs_evalon_20260307_direct" / "global_signal_summary.txt"
EVALON_BLEND_SUMMARY = ROOT / "debug_output" / "_tmp_chain_s180promote_20260308" / "compare_vs_evalon_20260307_blend" / "summary_metrics.txt"


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
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def fmt(x: Any, digits: int = 6) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def mean(values: Iterable[Any]) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def diff(cur: Any, ref: Any) -> float:
    a = safe_float(cur)
    b = safe_float(ref)
    if not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    return float(a - b)


def improvement(base: Any, cur: Any) -> float:
    a = safe_float(base)
    b = safe_float(cur)
    if not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    return float(a - b)


def run_cmd(cmd: Sequence[str], *, log_file: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
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


def _git_show_text(path: Path) -> Optional[str]:
    rel = path.relative_to(ROOT).as_posix()
    proc = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def read_text_or_git(path: Path) -> str:
    if path.is_file() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8")
    txt = _git_show_text(path)
    if txt:
        return txt
    raise RuntimeError(f"missing text artifact in worktree/git: {path}")


def load_stage6_case(name: str) -> Dict[str, Any]:
    summary = load_json(STAGE6_SUMMARY_JSON)
    cases = summary.get("cases", [])
    if not isinstance(cases, list):
        raise RuntimeError(f"invalid cases in {STAGE6_SUMMARY_JSON}")
    for case in cases:
        if isinstance(case, dict) and str(case.get("name")) == name:
            return case
    raise RuntimeError(f"missing case {name} in {STAGE6_SUMMARY_JSON}")


def make_generated_config(base_config: Path, out_json: Path, overrides: Mapping[str, Any]) -> Path:
    payload = load_json(base_config)
    payload.update(dict(overrides))
    write_json(out_json, payload)
    return out_json


def create_replace_zerophase_warmstart(src_ckpt: Path, dst_ckpt: Path, report_json: Path) -> None:
    if dst_ckpt.is_file() and report_json.is_file():
        return
    obj = torch.load(src_ckpt, map_location="cpu")
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError(f"unexpected checkpoint format: {src_ckpt}")
    out_obj = dict(obj)

    dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, dst_ckpt)
    write_json(
        report_json,
        {
            "source_ckpt": str(src_ckpt),
            "output_ckpt": str(dst_ckpt),
            "copied_without_phase_z_direct_adaptation": True,
        },
    )


def run_posttrain_stage(
    *,
    config: Path,
    ckpt_in: Path,
    out_dir: Path,
    run_name: str,
    log_file: Path,
) -> Path:
    ckpt_out = out_dir / f"ckpt_last_{run_name}.pth"
    if ckpt_out.is_file():
        return ckpt_out
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            sys.executable,
            "-m",
            "train.posttrain",
            "--config",
            str(config),
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
        ],
        log_file=log_file,
    )
    return ckpt_out


def run_70r_promote(
    *,
    config_json: Path,
    out_dir: Path,
    run_name: str,
    log_file: Path,
) -> Path:
    ckpt_out = out_dir / f"ckpt_last_{run_name}.pth"
    if ckpt_out.is_file():
        return ckpt_out
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            sys.executable,
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
        ],
        log_file=log_file,
    )
    return ckpt_out


def run_eval(
    *,
    model_ckpt: Path,
    out_dir: Path,
    contacts_source: str,
    log_file: Path,
) -> Path:
    eval_json = out_dir / "Walk_F_freerun_cycles.json"
    if eval_json.is_file():
        return eval_json
    cmd = [
        sys.executable,
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
        contacts_source,
        "--lambda_fusion_apply",
        "--log_contacts",
        "--export_direct_arm_probe",
        "--export_joint_direct_geolocal_series",
        "--out",
        str(out_dir),
        "--force",
    ]
    if contacts_source == "pretrain_contact":
        cmd.extend(
            [
                "--contacts_meas_pretrain_clamp",
                PRETRAIN_CLAMP,
                "--contacts_meas_pretrain_affine_stats",
                str(AFFINE_STATS),
                "--encoder-bundle",
                str(ENCODER_BUNDLE),
            ]
        )
    run_cmd(cmd, log_file=log_file)
    return eval_json


def ensure_group_summary(eval_json: Path, out_json: Path, *, log_file: Path) -> None:
    if out_json.is_file():
        return
    run_cmd(
        [
            sys.executable,
            str(ROOT / "tools" / "phasea_group_summary.py"),
            str(eval_json),
            "--cycle_gte",
            "1",
            "--drop_wrap",
            "--out",
            str(out_json),
        ],
        log_file=log_file,
    )


def masked_metric_means(eval_json: Path) -> Dict[str, float]:
    obj = load_json(eval_json)
    rows = obj.get("metrics_per_step", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"missing metrics_per_step in {eval_json}")
    masked = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if int(row.get("cycle", -1)) < 1:
            continue
        if bool(row.get("wrap_boundary_step", False)):
            continue
        masked.append(row)
    keys = (
        "BlendGeoLocalDeg",
        "BlendGeoLocalDegWeighted",
        "GeoLocalDeg",
        "GeoLocalDegWeighted",
        "DirectGeoLocalDeg",
        "DirectGeoLocalDegWeighted",
        "LambdaMean",
        "LambdaEffMean",
        "LambdaRelMean",
    )
    out = {"masked_steps": float(len(masked))}
    for key in keys:
        out[key] = mean(row.get(key) for row in masked)
    return out


def group_metrics(path: Path) -> Dict[str, float]:
    groups = load_json(path).get("groups", {})
    return {
        "all_ex_root": safe_float(groups.get("all_ex_root", {}).get("mean")),
        "leg": safe_float(groups.get("leg", {}).get("mean")),
        "nonleg": safe_float(groups.get("nonleg", {}).get("mean")),
        "arm": safe_float(groups.get("arm", {}).get("mean")),
        "else": safe_float(groups.get("else", {}).get("mean")),
    }


def _build_group_indices(names: Sequence[str], root_idx: int) -> Dict[str, List[int]]:
    leg_set = {str(x) for x in DEFAULT_DIRECT_POSE_LEG_BONES}
    arm_set = {str(x) for x in STAGE6_3WAY_ARMCHAIN_BONES}
    idx_leg = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in leg_set]
    idx_arm = [i for i, name in enumerate(names) if int(i) != int(root_idx) and str(name) in arm_set]
    idx_arm_left = [i for i in idx_arm if str(names[i]).endswith("_l")]
    idx_arm_right = [i for i in idx_arm if str(names[i]).endswith("_r")]
    idx_nonleg = [i for i in range(len(names)) if int(i) != int(root_idx) and i not in set(idx_leg)]
    idx_all = [i for i in range(len(names)) if int(i) != int(root_idx)]
    return {
        "leg": idx_leg,
        "arm": idx_arm,
        "arm_left": idx_arm_left,
        "arm_right": idx_arm_right,
        "nonleg": idx_nonleg,
        "all_ex_root": idx_all,
    }


def _mean_joint_values(
    mat: Sequence[Any],
    steps: Sequence[Mapping[str, Any]],
    joint_indices: Sequence[int],
    *,
    cycle_gte: int = 1,
    drop_wrap: bool = True,
    sic_lo: Optional[int] = None,
    sic_hi: Optional[int] = None,
) -> float:
    values: List[float] = []
    for step_i, step in enumerate(steps):
        if step_i >= len(mat):
            break
        if int(step.get("cycle", 0) or 0) < int(cycle_gte):
            continue
        if drop_wrap and bool(step.get("wrap_boundary_step", False)):
            continue
        sic = int(step.get("step_in_cycle", -1) or -1)
        if sic_lo is not None and sic < int(sic_lo):
            continue
        if sic_hi is not None and sic > int(sic_hi):
            continue
        row = mat[step_i]
        if not isinstance(row, list):
            continue
        for joint_i in joint_indices:
            if int(joint_i) >= len(row):
                continue
            v = safe_float(row[joint_i])
            if math.isfinite(v):
                values.append(v)
    return mean(values)


def window_group_stats(eval_json: Path) -> Dict[str, Any]:
    obj = load_json(eval_json)
    steps = obj.get("metrics_per_step", [])
    per = obj.get("per_step_direct_geolocal_deg", {})
    if not isinstance(steps, list) or not isinstance(per, Mapping):
        raise RuntimeError(f"missing direct geolocal payload in {eval_json}")
    names = [str(x) for x in per.get("bone_names", [])]
    mat = per.get("DirectGeoLocalDeg", [])
    if not isinstance(mat, list) or not names:
        raise RuntimeError(f"invalid per_step_direct_geolocal_deg in {eval_json}")
    root_idx = int(per.get("root_idx", 0) or 0)
    idx = _build_group_indices(names, root_idx)
    name_to_idx = {str(name): int(i) for i, name in enumerate(names)}

    def _section(sic_lo: Optional[int], sic_hi: Optional[int]) -> Dict[str, float]:
        return {
            "legs_main": _mean_joint_values(mat, steps, idx["leg"], sic_lo=sic_lo, sic_hi=sic_hi),
            "arms_main": _mean_joint_values(mat, steps, idx["arm"], sic_lo=sic_lo, sic_hi=sic_hi),
            "left_arm_main": _mean_joint_values(mat, steps, idx["arm_left"], sic_lo=sic_lo, sic_hi=sic_hi),
            "right_arm_main": _mean_joint_values(mat, steps, idx["arm_right"], sic_lo=sic_lo, sic_hi=sic_hi),
        }

    foot_ball_left = [name_to_idx[x] for x in ("foot_l", "ball_l") if x in name_to_idx]
    calf_r = [name_to_idx["calf_r"]] if "calf_r" in name_to_idx else []

    return {
        "overall": _section(None, None),
        "A_52_59": _section(52, 59),
        "B_76_80": _section(76, 80),
        "hotspots": {
            "foot_l_ball_l_SIC12_15": _mean_joint_values(mat, steps, foot_ball_left, sic_lo=12, sic_hi=15),
            "calf_r_SIC2_4": _mean_joint_values(mat, steps, calf_r, sic_lo=2, sic_hi=4),
        },
    }


def parse_global_signal_summary(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "metrics": {},
        "paths": {},
    }
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s and s.split("=", 1)[0] in {"old_json", "new_json", "mask"}:
            k, v = s.split("=", 1)
            out["paths"][k] = v
            continue
        m = re.match(r"^(mean_old|mean_new|mean_delta|leg8_mean_old|leg8_mean_new|leg8_mean_delta|non_leg_mean_old|non_leg_mean_new|non_leg_mean_delta|improved_ratio|worse_ratio|median_delta|bones_excl_root|bones_regress_by_mean|bones_improve_by_mean)=([-+0-9.eE]+)$", s)
        if m:
            out["metrics"][m.group(1)] = safe_float(m.group(2))
    out["direct_group_summary"] = {
        "all_ex_root_old": safe_float(out["metrics"].get("mean_old")),
        "all_ex_root_new": safe_float(out["metrics"].get("mean_new")),
        "leg_old": safe_float(out["metrics"].get("leg8_mean_old")),
        "leg_new": safe_float(out["metrics"].get("leg8_mean_new")),
        "nonleg_old": safe_float(out["metrics"].get("non_leg_mean_old")),
        "nonleg_new": safe_float(out["metrics"].get("non_leg_mean_new")),
    }
    return out


def parse_blend_summary(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "metrics": {},
        "paths": {},
    }
    metric_re = re.compile(r"^([A-Za-z0-9]+): .*?old_mean=([-+0-9.eE]+) new_mean=([-+0-9.eE]+) delta=([-+0-9.eE]+)")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s and s.split("=", 1)[0] in {
            "old_json",
            "new_json",
            "mask",
            "old_phase_reset",
            "new_phase_reset",
            "old_contacts_meas_source",
            "new_contacts_meas_source",
            "old_lambda_fusion_apply",
            "new_lambda_fusion_apply",
        }:
            k, v = s.split("=", 1)
            out["paths"][k] = v
            continue
        m = metric_re.match(s)
        if m:
            out["metrics"][m.group(1)] = {
                "old_mean": safe_float(m.group(2)),
                "new_mean": safe_float(m.group(3)),
                "delta": safe_float(m.group(4)),
            }
    return out


def parse_chain_verdict(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    section = None
    pat_overall = re.compile(
        r"overall: `legs_main=([-+0-9.]+)`, `arms_main=([-+0-9.]+)`, `left_arm_main=([-+0-9.]+)`, `right_arm_main=([-+0-9.]+)`"
    )
    pat_window = re.compile(r"(A window \(52-59\)|B window \(76-80\)): `legs_main=([-+0-9.]+)`, `arms_main=([-+0-9.]+)`")
    pat_blend = re.compile(r"- `([A-Za-z0-9]+)`: `([-+0-9.]+)`")
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## Stage71"):
            section = "stage71"
            out[section] = {}
            continue
        if s.startswith("## Stage72"):
            section = "stage72"
            out[section] = {}
            continue
        if s.startswith("## LambdaFinal direct-path"):
            section = "lambda_direct_path"
            out[section] = {}
            continue
        if s.startswith("## LambdaFinal blend-aware"):
            section = "lambda_blend_summary"
            out[section] = {}
            continue
        m = pat_overall.search(s)
        if m and section and section != "lambda_blend_summary":
            out[section]["overall"] = {
                "legs_main": safe_float(m.group(1)),
                "arms_main": safe_float(m.group(2)),
                "left_arm_main": safe_float(m.group(3)),
                "right_arm_main": safe_float(m.group(4)),
            }
            continue
        m = pat_window.search(s)
        if m and section and section != "lambda_blend_summary":
            key = "A_52_59" if "52-59" in m.group(1) else "B_76_80"
            out[section][key] = {
                "legs_main": safe_float(m.group(2)),
                "arms_main": safe_float(m.group(3)),
            }
            continue
        m = pat_blend.match(s)
        if m and section == "lambda_blend_summary":
            out[section][m.group(1)] = safe_float(m.group(2))
    return out


def parse_reference_payload() -> Dict[str, Any]:
    accepted_direct = parse_global_signal_summary(read_text_or_git(ACCEPTED_DIRECT_SUMMARY))
    accepted_blend = parse_blend_summary(read_text_or_git(ACCEPTED_BLEND_SUMMARY))
    evalon_direct = parse_global_signal_summary(read_text_or_git(EVALON_DIRECT_SUMMARY))
    evalon_blend = parse_blend_summary(read_text_or_git(EVALON_BLEND_SUMMARY))
    chain_verdict = parse_chain_verdict(read_text_or_git(ACCEPTED_CHAIN_VERDICT))
    return {
        "accepted_old_baseline_r5": {
            "source_direct_summary": str(ACCEPTED_DIRECT_SUMMARY),
            "source_blend_summary": str(ACCEPTED_BLEND_SUMMARY),
            "masked_means": {k: v["old_mean"] for k, v in accepted_blend["metrics"].items()},
            "direct_group_summary": {
                "all_ex_root": accepted_direct["direct_group_summary"]["all_ex_root_old"],
                "leg": accepted_direct["direct_group_summary"]["leg_old"],
                "nonleg": accepted_direct["direct_group_summary"]["nonleg_old"],
            },
        },
        "accepted_final_model_source": {
            "source_direct_summary": str(ACCEPTED_DIRECT_SUMMARY),
            "source_blend_summary": str(ACCEPTED_BLEND_SUMMARY),
            "masked_means": {k: v["new_mean"] for k, v in accepted_blend["metrics"].items()},
            "direct_group_summary": {
                "all_ex_root": accepted_direct["direct_group_summary"]["all_ex_root_new"],
                "leg": accepted_direct["direct_group_summary"]["leg_new"],
                "nonleg": accepted_direct["direct_group_summary"]["nonleg_new"],
            },
            "artifact_note": "Current accepted final compare artifact is model-source, not strict pretrain_contact.",
        },
        "evalon_20260307_baseline": {
            "source_direct_summary": str(EVALON_DIRECT_SUMMARY),
            "source_blend_summary": str(EVALON_BLEND_SUMMARY),
            "masked_means": {k: v["old_mean"] for k, v in evalon_blend["metrics"].items()},
            "direct_group_summary": {
                "all_ex_root": evalon_direct["direct_group_summary"]["all_ex_root_old"],
                "leg": evalon_direct["direct_group_summary"]["leg_old"],
                "nonleg": evalon_direct["direct_group_summary"]["nonleg_old"],
            },
        },
        "accepted_chain_verdict": {
            "source": str(ACCEPTED_CHAIN_VERDICT),
            "parsed": chain_verdict,
        },
        "strict_reference_status": {
            "available": False,
            "reason": (
                "Repo docs explicitly note the accepted chain compare artifacts are model-source; "
                "a strict pretrain_contact+affine_mix08 accepted-final eval snapshot is not archived locally."
            ),
        },
    }


def build_stage_paths() -> Dict[str, Path]:
    return {
        "lane_log": OUT_ROOT / "lane.log",
        "status_json": OUT_ROOT / "status.json",
        "summary_json": OUT_ROOT / "summary.json",
        "summary_md": OUT_ROOT / "summary.md",
        "cfg_70b_replace": OUT_ROOT / "configs" / "posttrain_70b_replacecontacts_from_cp015oldplan_20260314.json",
        "cfg_70r": OUT_ROOT / "configs" / "posttrain_70R_from_cp015oldplan_replace_lr3e4_e1_s60_20260314.json",
        "warmstart_ckpt": MODEL_ROOT / "warmstart" / "ckpt_last_cp015_oldplan_70a_replacecontacts_zerophase_20260314.pth",
        "warmstart_report": OUT_ROOT / "warmstart" / "replace_zerophase_report.json",
        "ckpt_70a": MODEL_ROOT / "70a" / "ckpt_last_WalkF_stage7_70a_from_cp015_oldplan_20260314.pth",
        "ckpt_70b": MODEL_ROOT / "70b" / "ckpt_last_WalkF_stage7_70b_concat_from_cp015_oldplan_20260314.pth",
        "ckpt_70b_replace": MODEL_ROOT / "70b_replace" / "ckpt_last_WalkF_stage7_70b_replacecontacts_from_cp015_oldplan_20260314.pth",
        "ckpt_70r": MODEL_ROOT / "70R" / "ckpt_last_WalkF_stage7_70R_from_cp015_oldplan_trunkfull_s180_20260314.pth",
        "ckpt_71": MODEL_ROOT / "71" / "ckpt_last_WalkF_stage7_71_from_cp015_oldplan_20260314.pth",
        "ckpt_72": MODEL_ROOT / "72" / "ckpt_last_WalkF_stage7_72_from_cp015_oldplan_20260314.pth",
        "ckpt_lambda": MODEL_ROOT / "lambda" / "ckpt_last_WalkF_stage7_lambda_from_cp015_oldplan_20260314.pth",
        "eval_strict_dir": OUT_ROOT / "eval_pretrain_contact",
        "eval_strict_json": OUT_ROOT / "eval_pretrain_contact" / "Walk_F_freerun_cycles.json",
        "eval_strict_group": OUT_ROOT / "eval_pretrain_contact_group_summary.json",
        "eval_model_dir": OUT_ROOT / "eval_model_source",
        "eval_model_json": OUT_ROOT / "eval_model_source" / "Walk_F_freerun_cycles.json",
        "eval_model_group": OUT_ROOT / "eval_model_source_group_summary.json",
    }


def build_summary(
    stage6_case: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    refs = parse_reference_payload()
    strict_masked = masked_metric_means(paths["eval_strict_json"])
    strict_group = group_metrics(paths["eval_strict_group"])
    strict_windows = window_group_stats(paths["eval_strict_json"])
    model_masked = masked_metric_means(paths["eval_model_json"])
    model_group = group_metrics(paths["eval_model_group"])
    model_windows = window_group_stats(paths["eval_model_json"])

    ref_old = refs["accepted_old_baseline_r5"]
    ref_final = refs["accepted_final_model_source"]
    ref_evalon = refs["evalon_20260307_baseline"]

    summary = {
        "run_date": RUN_DATE,
        "stage6_source_summary": str(STAGE6_SUMMARY_JSON),
        "stage6_case": stage6_case,
        "policy": {
            "teacher": str(TEACHER),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "final_eval_strict": {
                "rounds": 5,
                "depth": 3,
                "time_index_mode": "cycle",
                "event_clock": "auto",
                "phase_reset_source": "none",
                "contacts_meas_source": "pretrain_contact",
                "contacts_meas_pretrain_clamp": PRETRAIN_CLAMP,
                "contacts_meas_pretrain_affine_stats": str(AFFINE_STATS),
                "encoder_bundle": str(ENCODER_BUNDLE),
            },
            "final_eval_model_source": {
                "rounds": 5,
                "depth": 3,
                "time_index_mode": "cycle",
                "event_clock": "auto",
                "phase_reset_source": "none",
                "contacts_meas_source": "model",
            },
        },
        "stages": {
            "transplant_stage6": {
                "config": str(CONFIG_70A.parent / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json"),
                "ckpt": str(stage6_case["paths"]["stage6_ckpt"]),
                "group_summary": str(stage6_case["paths"]["stage6_group_summary"]),
            },
            "70a": {"config": str(CONFIG_70A), "ckpt": str(paths["ckpt_70a"])},
            "70b": {"config": str(CONFIG_70B), "ckpt": str(paths["ckpt_70b"])},
            "70a_replace_warmstart": {
                "ckpt": str(paths["warmstart_ckpt"]),
                "report": str(paths["warmstart_report"]),
            },
            "new70b_replace": {"config": str(paths["cfg_70b_replace"]), "ckpt": str(paths["ckpt_70b_replace"])},
            "70R_promoted_s180": {"config": str(paths["cfg_70r"]), "ckpt": str(paths["ckpt_70r"])},
            "71": {"config": str(CONFIG_71), "ckpt": str(paths["ckpt_71"])},
            "72": {"config": str(CONFIG_72), "ckpt": str(paths["ckpt_72"])},
            "lambda_final": {"config": str(CONFIG_LAMBDA), "ckpt": str(paths["ckpt_lambda"])},
        },
        "accepted_references": refs,
        "final_evals": {
            "strict_pretrain_contact": {
                "masked_means": strict_masked,
                "direct_group_summary": strict_group,
                "window_summary": strict_windows,
                "paths": {
                    "eval_json": str(paths["eval_strict_json"]),
                    "group_summary": str(paths["eval_strict_group"]),
                },
            },
            "model_source": {
                "masked_means": model_masked,
                "direct_group_summary": model_group,
                "window_summary": model_windows,
                "paths": {
                    "eval_json": str(paths["eval_model_json"]),
                    "group_summary": str(paths["eval_model_group"]),
                },
            },
        },
        "comparisons": {
            "model_source_vs_accepted_final_model_source": {
                "masked_means_delta": {
                    key: diff(model_masked.get(key), ref_final["masked_means"].get(key))
                    for key in (
                        "DirectGeoLocalDeg",
                        "DirectGeoLocalDegWeighted",
                        "BlendGeoLocalDeg",
                        "BlendGeoLocalDegWeighted",
                        "GeoLocalDeg",
                        "GeoLocalDegWeighted",
                    )
                },
                "direct_group_delta": {
                    key: diff(model_group.get(key), ref_final["direct_group_summary"].get(key))
                    for key in ("all_ex_root", "leg", "nonleg")
                },
            },
            "model_source_vs_accepted_old_baseline_r5": {
                "masked_means_delta": {
                    key: diff(model_masked.get(key), ref_old["masked_means"].get(key))
                    for key in (
                        "DirectGeoLocalDeg",
                        "DirectGeoLocalDegWeighted",
                        "BlendGeoLocalDeg",
                        "BlendGeoLocalDegWeighted",
                        "GeoLocalDeg",
                        "GeoLocalDegWeighted",
                    )
                },
                "direct_group_delta": {
                    key: diff(model_group.get(key), ref_old["direct_group_summary"].get(key))
                    for key in ("all_ex_root", "leg", "nonleg")
                },
            },
            "strict_pretrain_contact_vs_accepted_old_baseline_r5": {
                "masked_means_delta": {
                    key: diff(strict_masked.get(key), ref_old["masked_means"].get(key))
                    for key in (
                        "DirectGeoLocalDeg",
                        "DirectGeoLocalDegWeighted",
                        "BlendGeoLocalDeg",
                        "BlendGeoLocalDegWeighted",
                        "GeoLocalDeg",
                        "GeoLocalDegWeighted",
                    )
                },
                "direct_group_delta": {
                    key: diff(strict_group.get(key), ref_old["direct_group_summary"].get(key))
                    for key in ("all_ex_root", "leg", "nonleg")
                },
            },
            "model_source_vs_evalon_20260307_baseline": {
                "masked_means_delta": {
                    key: diff(model_masked.get(key), ref_evalon["masked_means"].get(key))
                    for key in (
                        "DirectGeoLocalDeg",
                        "DirectGeoLocalDegWeighted",
                        "BlendGeoLocalDeg",
                        "BlendGeoLocalDegWeighted",
                        "GeoLocalDeg",
                        "GeoLocalDegWeighted",
                    )
                },
                "direct_group_delta": {
                    key: diff(model_group.get(key), ref_evalon["direct_group_summary"].get(key))
                    for key in ("all_ex_root", "leg", "nonleg")
                },
            },
        },
        "answers": {},
    }

    model_vs_final = summary["comparisons"]["model_source_vs_accepted_final_model_source"]
    model_vs_old = summary["comparisons"]["model_source_vs_accepted_old_baseline_r5"]
    strict_vs_old = summary["comparisons"]["strict_pretrain_contact_vs_accepted_old_baseline_r5"]

    beats_current_final = (
        safe_float(model_vs_final["masked_means_delta"]["DirectGeoLocalDeg"]) < 0.0
        and safe_float(model_vs_final["masked_means_delta"]["BlendGeoLocalDeg"]) < 0.0
        and safe_float(model_vs_final["direct_group_delta"]["all_ex_root"]) < 0.0
    )
    carries_to_final = (
        safe_float(model_vs_old["masked_means_delta"]["DirectGeoLocalDeg"]) < 0.0
        and safe_float(model_vs_old["direct_group_delta"]["all_ex_root"]) < 0.0
    )
    strict_still_supports = (
        safe_float(strict_vs_old["masked_means_delta"]["DirectGeoLocalDeg"]) < 0.0
        and safe_float(strict_vs_old["direct_group_delta"]["all_ex_root"]) < 0.0
    )

    summary["answers"] = {
        "q1_advantage_penetrates_to_lambda_final": {
            "value": bool(carries_to_final),
            "interpretation": (
                "Measured against the accepted old baseline r5 anchor, not against the current accepted final."
            ),
        },
        "q2_final_beats_current_accepted_mainline": {"value": bool(beats_current_final)},
        "q3_strict_eval_still_supports_advantage": {
            "value": bool(strict_still_supports),
            "note": refs["strict_reference_status"]["reason"],
        },
        "q4_should_switch_baseline_now": {"value": bool(beats_current_final and strict_still_supports)},
        "q5_should_only_do_future_simplification_on_this_chain": {
            "value": bool(beats_current_final and strict_still_supports),
            "note": "Only promote after it clears the current accepted final, not merely the old baseline.",
        },
    }
    return summary


def build_markdown(summary: Mapping[str, Any]) -> str:
    stage6_case = summary["stage6_case"]
    strict_eval = summary["final_evals"]["strict_pretrain_contact"]
    model_eval = summary["final_evals"]["model_source"]
    refs = summary["accepted_references"]
    answers = summary["answers"]
    cmp_model_final = summary["comparisons"]["model_source_vs_accepted_final_model_source"]
    cmp_model_old = summary["comparisons"]["model_source_vs_accepted_old_baseline_r5"]
    cmp_strict_old = summary["comparisons"]["strict_pretrain_contact_vs_accepted_old_baseline_r5"]

    lines: List[str] = []
    lines.append("# cp015 old-plan downstream chain")
    lines.append("")
    lines.append(f"- run_date: {summary['run_date']}")
    lines.append(f"- stage6_case: `{stage6_case['name']}`")
    lines.append(f"- transplant_stage6_ckpt: `{stage6_case['paths']['stage6_ckpt']}`")
    lines.append(
        f"- stage6_exit: all_ex_root={fmt(stage6_case['stage6_exit']['all_ex_root'])}, "
        f"leg={fmt(stage6_case['stage6_exit']['leg'])}, nonleg={fmt(stage6_case['stage6_exit']['nonleg'])}"
    )
    lines.append("")
    lines.append("## Stage ckpts")
    lines.append("")
    lines.append("| stage | ckpt |")
    lines.append("|---|---|")
    for key in (
        "transplant_stage6",
        "70a",
        "70b",
        "70a_replace_warmstart",
        "new70b_replace",
        "70R_promoted_s180",
        "71",
        "72",
        "lambda_final",
    ):
        info = summary["stages"][key]
        lines.append(f"| {key} | `{info['ckpt']}` |")
    lines.append("")
    lines.append("## Accepted references")
    lines.append("")
    lines.append("| ref | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ref_key in ("accepted_old_baseline_r5", "accepted_final_model_source", "evalon_20260307_baseline"):
        ref = refs[ref_key]
        mm = ref["masked_means"]
        gg = ref["direct_group_summary"]
        lines.append(
            f"| {ref_key} | {fmt(mm.get('DirectGeoLocalDeg'))} | {fmt(mm.get('BlendGeoLocalDeg'))} | "
            f"{fmt(mm.get('GeoLocalDeg'))} | {fmt(gg.get('all_ex_root'))} | {fmt(gg.get('leg'))} | {fmt(gg.get('nonleg'))} |"
        )
    lines.append("")
    lines.append("## Final evals")
    lines.append("")
    lines.append("| lane | DirectGeoLocalDeg | DirectGeoLocalDegWeighted | BlendGeoLocalDeg | BlendGeoLocalDegWeighted | GeoLocalDeg | GeoLocalDegWeighted | all_ex_root | leg | nonleg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lane_name, lane in (
        ("strict_pretrain_contact", strict_eval),
        ("model_source", model_eval),
    ):
        mm = lane["masked_means"]
        gg = lane["direct_group_summary"]
        lines.append(
            f"| {lane_name} | {fmt(mm.get('DirectGeoLocalDeg'))} | {fmt(mm.get('DirectGeoLocalDegWeighted'))} | "
            f"{fmt(mm.get('BlendGeoLocalDeg'))} | {fmt(mm.get('BlendGeoLocalDegWeighted'))} | "
            f"{fmt(mm.get('GeoLocalDeg'))} | {fmt(mm.get('GeoLocalDegWeighted'))} | "
            f"{fmt(gg.get('all_ex_root'))} | {fmt(gg.get('leg'))} | {fmt(gg.get('nonleg'))} |"
        )
    lines.append("")
    lines.append("## Window summary")
    lines.append("")
    lines.append("| lane | section | legs_main | arms_main | left_arm_main | right_arm_main |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for lane_name, lane in (
        ("strict_pretrain_contact", strict_eval),
        ("model_source", model_eval),
    ):
        for sec in ("overall", "A_52_59", "B_76_80"):
            row = lane["window_summary"][sec]
            lines.append(
                f"| {lane_name} | {sec} | {fmt(row.get('legs_main'))} | {fmt(row.get('arms_main'))} | "
                f"{fmt(row.get('left_arm_main'))} | {fmt(row.get('right_arm_main'))} |"
            )
    lines.append("")
    lines.append("| lane | foot_l_ball_l_SIC12_15 | calf_r_SIC2_4 |")
    lines.append("|---|---:|---:|")
    for lane_name, lane in (
        ("strict_pretrain_contact", strict_eval),
        ("model_source", model_eval),
    ):
        hot = lane["window_summary"]["hotspots"]
        lines.append(
            f"| {lane_name} | {fmt(hot.get('foot_l_ball_l_SIC12_15'))} | {fmt(hot.get('calf_r_SIC2_4'))} |"
        )
    lines.append("")
    lines.append("## Deltas")
    lines.append("")
    lines.append("| compare | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| model - accepted_final_model_source | {fmt(cmp_model_final['masked_means_delta']['DirectGeoLocalDeg'])} | "
        f"{fmt(cmp_model_final['masked_means_delta']['BlendGeoLocalDeg'])} | {fmt(cmp_model_final['masked_means_delta']['GeoLocalDeg'])} | "
        f"{fmt(cmp_model_final['direct_group_delta']['all_ex_root'])} | {fmt(cmp_model_final['direct_group_delta']['leg'])} | "
        f"{fmt(cmp_model_final['direct_group_delta']['nonleg'])} |"
    )
    lines.append(
        f"| model - accepted_old_baseline_r5 | {fmt(cmp_model_old['masked_means_delta']['DirectGeoLocalDeg'])} | "
        f"{fmt(cmp_model_old['masked_means_delta']['BlendGeoLocalDeg'])} | {fmt(cmp_model_old['masked_means_delta']['GeoLocalDeg'])} | "
        f"{fmt(cmp_model_old['direct_group_delta']['all_ex_root'])} | {fmt(cmp_model_old['direct_group_delta']['leg'])} | "
        f"{fmt(cmp_model_old['direct_group_delta']['nonleg'])} |"
    )
    lines.append(
        f"| strict - accepted_old_baseline_r5 | {fmt(cmp_strict_old['masked_means_delta']['DirectGeoLocalDeg'])} | "
        f"{fmt(cmp_strict_old['masked_means_delta']['BlendGeoLocalDeg'])} | {fmt(cmp_strict_old['masked_means_delta']['GeoLocalDeg'])} | "
        f"{fmt(cmp_strict_old['direct_group_delta']['all_ex_root'])} | {fmt(cmp_strict_old['direct_group_delta']['leg'])} | "
        f"{fmt(cmp_strict_old['direct_group_delta']['nonleg'])} |"
    )
    lines.append("")
    verdict = refs["accepted_chain_verdict"]["parsed"]
    lam_direct = verdict.get("lambda_direct_path", {})
    lam_blend = verdict.get("lambda_blend_summary", {})
    lines.append("## chain_verdict reference")
    lines.append("")
    if lam_direct:
        lines.append(
            f"- accepted lambda direct-path delta vs previous new chain: overall legs_main={fmt(lam_direct.get('overall', {}).get('legs_main'))}, "
            f"arms_main={fmt(lam_direct.get('overall', {}).get('arms_main'))}, "
            f"A arms={fmt(lam_direct.get('A_52_59', {}).get('arms_main'))}, "
            f"B arms={fmt(lam_direct.get('B_76_80', {}).get('arms_main'))}"
        )
    if lam_blend:
        lines.append(
            f"- accepted lambda blend delta vs previous new chain: BlendGeoLocalDeg={fmt(lam_blend.get('BlendGeoLocalDeg'))}, "
            f"GeoLocalDeg={fmt(lam_blend.get('GeoLocalDeg'))}, DirectGeoLocalDeg={fmt(lam_blend.get('DirectGeoLocalDeg'))}"
        )
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    lines.append(
        f"1. Advantage penetrates to lambda final: `{str(bool(answers['q1_advantage_penetrates_to_lambda_final']['value'])).lower()}` "
        f"(candidate still beats the accepted old baseline r5 anchor at lambda final)."
    )
    lines.append(
        f"2. Final beats current accepted mainline: `{str(bool(answers['q2_final_beats_current_accepted_mainline']['value'])).lower()}` "
        f"(current accepted compare artifact is model-source)."
    )
    lines.append(
        f"3. Strict eval still supports the carry claim: `{str(bool(answers['q3_strict_eval_still_supports_advantage']['value'])).lower()}` "
        f"(note: {answers['q3_strict_eval_still_supports_advantage']['note']})"
    )
    lines.append(
        f"4. Switch baseline now: `{str(bool(answers['q4_should_switch_baseline_now']['value'])).lower()}`"
    )
    lines.append(
        f"5. Future simplification only on this chain: `{str(bool(answers['q5_should_only_do_future_simplification_on_this_chain']['value'])).lower()}`"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage6-case", type=str, default="cp015_with_old_planstack")
    args = ap.parse_args()

    required = [
        STAGE6_SUMMARY_JSON,
        TEACHER,
        ENCODER_BUNDLE,
        AFFINE_STATS,
        CONFIG_70A,
        CONFIG_70B,
        CONFIG_70R,
        CONFIG_71,
        CONFIG_72,
        CONFIG_LAMBDA,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    stage6_case = load_stage6_case(args.stage6_case)
    stage6_ckpt = Path(str(stage6_case["paths"]["stage6_ckpt"]))
    if not stage6_ckpt.is_file():
        raise SystemExit(f"missing stage6 ckpt: {stage6_ckpt}")

    paths = build_stage_paths()
    run_name_70a = "WalkF_stage7_70a_from_cp015_oldplan_20260314"
    run_name_70b = "WalkF_stage7_70b_concat_from_cp015_oldplan_20260314"
    run_name_70b_replace = "WalkF_stage7_70b_replacecontacts_from_cp015_oldplan_20260314"
    run_name_70r = "WalkF_stage7_70R_from_cp015_oldplan_trunkfull_s180_20260314"
    run_name_71 = "WalkF_stage7_71_from_cp015_oldplan_20260314"
    run_name_72 = "WalkF_stage7_72_from_cp015_oldplan_20260314"
    run_name_lambda = "WalkF_stage7_lambda_from_cp015_oldplan_20260314"

    log("=== stage 70a ===")
    ckpt_70a = run_posttrain_stage(
        config=CONFIG_70A,
        ckpt_in=stage6_ckpt,
        out_dir=MODEL_ROOT / "70a",
        run_name=run_name_70a,
        log_file=paths["lane_log"],
    )

    log("=== stage 70b concat ===")
    ckpt_70b = run_posttrain_stage(
        config=CONFIG_70B,
        ckpt_in=ckpt_70a,
        out_dir=MODEL_ROOT / "70b",
        run_name=run_name_70b,
        log_file=paths["lane_log"],
    )

    log("=== 70a replace zerophase warmstart ===")
    create_replace_zerophase_warmstart(
        src_ckpt=ckpt_70a,
        dst_ckpt=paths["warmstart_ckpt"],
        report_json=paths["warmstart_report"],
    )

    log("=== new70b replace ===")
    cfg_70b_replace = make_generated_config(
        CONFIG_70B,
        paths["cfg_70b_replace"],
        {
            "ckpt_in": str(paths["warmstart_ckpt"]),
            "out_dir": str(MODEL_ROOT / "70b_replace"),
            "run_name": run_name_70b_replace,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    ckpt_70b_replace = run_posttrain_stage(
        config=cfg_70b_replace,
        ckpt_in=paths["warmstart_ckpt"],
        out_dir=MODEL_ROOT / "70b_replace",
        run_name=run_name_70b_replace,
        log_file=paths["lane_log"],
    )

    log("=== promoted 70R s180 ===")
    cfg_70r = make_generated_config(
        CONFIG_70R,
        paths["cfg_70r"],
        {
            "ckpt_in": str(ckpt_70b_replace),
            "out_dir": str(MODEL_ROOT / "70R"),
            "run_name": run_name_70r,
            "lr": 3e-4,
            "epochs": 1,
            "steps_per_epoch": 60,
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    ckpt_70r = run_70r_promote(
        config_json=cfg_70r,
        out_dir=MODEL_ROOT / "70R",
        run_name=run_name_70r,
        log_file=paths["lane_log"],
    )

    log("=== stage 71 ===")
    ckpt_71 = run_posttrain_stage(
        config=CONFIG_71,
        ckpt_in=ckpt_70r,
        out_dir=MODEL_ROOT / "71",
        run_name=run_name_71,
        log_file=paths["lane_log"],
    )

    log("=== stage 72 ===")
    ckpt_72 = run_posttrain_stage(
        config=CONFIG_72,
        ckpt_in=ckpt_71,
        out_dir=MODEL_ROOT / "72",
        run_name=run_name_72,
        log_file=paths["lane_log"],
    )

    log("=== lambda final ===")
    ckpt_lambda = run_posttrain_stage(
        config=CONFIG_LAMBDA,
        ckpt_in=ckpt_72,
        out_dir=MODEL_ROOT / "lambda",
        run_name=run_name_lambda,
        log_file=paths["lane_log"],
    )

    log("=== strict eval ===")
    eval_strict_json = run_eval(
        model_ckpt=ckpt_lambda,
        out_dir=paths["eval_strict_dir"],
        contacts_source="pretrain_contact",
        log_file=paths["lane_log"],
    )
    ensure_group_summary(eval_strict_json, paths["eval_strict_group"], log_file=paths["lane_log"])

    log("=== model-source eval ===")
    eval_model_json = run_eval(
        model_ckpt=ckpt_lambda,
        out_dir=paths["eval_model_dir"],
        contacts_source="model",
        log_file=paths["lane_log"],
    )
    ensure_group_summary(eval_model_json, paths["eval_model_group"], log_file=paths["lane_log"])

    status_payload = {
        "stage6_case": stage6_case["name"],
        "stage6_ckpt": str(stage6_ckpt),
        "stage_ckpts": {
            "70a": str(ckpt_70a),
            "70b": str(ckpt_70b),
            "warmstart": str(paths["warmstart_ckpt"]),
            "70b_replace": str(ckpt_70b_replace),
            "70R": str(ckpt_70r),
            "71": str(ckpt_71),
            "72": str(ckpt_72),
            "lambda": str(ckpt_lambda),
        },
        "evals": {
            "strict_pretrain_contact": str(eval_strict_json),
            "model_source": str(eval_model_json),
        },
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(paths["status_json"], status_payload)

    summary = build_summary(stage6_case, paths)
    write_json(paths["summary_json"], summary)
    paths["summary_md"].write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={paths['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
