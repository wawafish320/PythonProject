#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

RUN_DATE = "20260313"
ROOT = Path(__file__).resolve().parents[2]
DEBUG_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "models" / f"__tmp_stage6_basetrain_compare_{RUN_DATE}"
CONFIG = ROOT / "config" / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json"
TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
PRETRAIN_CLAMP = "1.0"

LANES: List[Dict[str, str]] = [
    {
        "name": "old_bestfree",
        "family": "main_old",
        "selector": "best_free",
        "ckpt": str(ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1" / "ckpt_best_free_exp_phase_DirectBranch_v1_d1.pth"),
    },
    {
        "name": "cp015_bestfree",
        "family": "main_cp015_tailk3",
        "selector": "best_free",
        "ckpt": str(ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3" / "ckpt_best_free_exp_phase_DirectBranch_v1_d1_cp015_tailk3.pth"),
    },
    {
        "name": "geofix_bestteacher",
        "family": "geofix_r3",
        "selector": "best_teacher",
        "ckpt": "/tmp/MLPL2_DirectBranch_v1_geofix_r3/exp_phase_DirectBranch_v1_d1_geofix_r3/ckpt_best_teacher_exp_phase_DirectBranch_v1_d1_geofix_r3.pth",
    },
    {
        "name": "geofix_bestfree",
        "family": "geofix_r3",
        "selector": "best_free",
        "ckpt": "/tmp/MLPL2_DirectBranch_v1_geofix_r3/exp_phase_DirectBranch_v1_d1_geofix_r3/ckpt_best_free_exp_phase_DirectBranch_v1_d1_geofix_r3.pth",
    },
    {
        "name": "geofix_last",
        "family": "geofix_r3",
        "selector": "last",
        "ckpt": "/tmp/MLPL2_DirectBranch_v1_geofix_r3/exp_phase_DirectBranch_v1_d1_geofix_r3/ckpt_last_exp_phase_DirectBranch_v1_d1_geofix_r3.pth",
    },
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def is_finite(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def mean(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values if is_finite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def run_cmd(cmd: List[str], *, log_file: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"RUN {' '.join(cmd)}")
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
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
        code = proc.wait()
        fh.write(f"[exit_code] {code}\n")
        fh.flush()
    if code != 0:
        raise SystemExit(code)


def extract_stage6_init(log_json: Path, out_json: Path) -> Dict[str, Any]:
    obj = json.loads(log_json.read_text())
    rows = obj.get("log", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"missing log rows in {log_json}")

    def row_payload(row: Dict[str, Any]) -> Dict[str, float]:
        g_arm = safe_float(row.get("direct_grad_norm_out_arm"))
        g_else = safe_float(row.get("direct_grad_norm_out_else"))
        grad_arm_over_else = float("nan")
        if math.isfinite(g_arm) and math.isfinite(g_else) and g_else > 0.0:
            grad_arm_over_else = float(g_arm / g_else)
        return {
            "dir_leg_base": safe_float(row.get("dir_leg_base")),
            "dir_nonleg_base": safe_float(row.get("dir_nonleg_base")),
            "leg_over_nonleg": safe_float(row.get("leg_over_nonleg")),
            "arm_over_else": safe_float(row.get("arm_over_else")),
            "direct_grad_norm_out_arm": g_arm,
            "direct_grad_norm_out_else": g_else,
            "grad_arm_over_else": grad_arm_over_else,
            "step": safe_float(row.get("step")),
        }

    step1 = row_payload(rows[0])
    head = [row_payload(row) for row in rows[: min(20, len(rows))]]
    head20 = {
        "dir_leg_base": mean(r["dir_leg_base"] for r in head),
        "dir_nonleg_base": mean(r["dir_nonleg_base"] for r in head),
        "leg_over_nonleg": mean(r["leg_over_nonleg"] for r in head),
        "arm_over_else": mean(r["arm_over_else"] for r in head),
        "direct_grad_norm_out_arm": mean(r["direct_grad_norm_out_arm"] for r in head),
        "direct_grad_norm_out_else": mean(r["direct_grad_norm_out_else"] for r in head),
        "grad_arm_over_else": mean(r["grad_arm_over_else"] for r in head),
    }
    payload = {
        "source": str(log_json),
        "rows": int(len(rows)),
        "head_count": int(len(head)),
        "step1": step1,
        "head20_mean": head20,
    }
    write_json(out_json, payload)
    return payload


def load_group_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def group_mean(path: Path, group: str) -> float:
    obj = load_group_summary(path)
    return safe_float(obj.get("groups", {}).get(group, {}).get("mean"))


def lane_paths(name: str) -> Dict[str, Path]:
    lane_root = DEBUG_ROOT / name
    model_root = MODELS_ROOT / name
    run_name = f"{name}_stage6_cmp_{RUN_DATE}"
    return {
        "lane_root": lane_root,
        "model_root": model_root,
        "run_name": Path(run_name),
        "lane_log": lane_root / "lane.log",
        "basetrain_eval_dir": lane_root / "basetrain_freerun",
        "basetrain_eval_json": lane_root / "basetrain_freerun" / "Walk_F_freerun_cycles.json",
        "basetrain_group_json": lane_root / "basetrain_group_summary.json",
        "stage6_log_json": model_root / f"posttrain_log_{run_name}.json",
        "stage6_ckpt": model_root / f"ckpt_last_{run_name}.pth",
        "stage6_init_json": lane_root / "posttrain_stage6_init_stats.json",
        "stage6_eval_dir": lane_root / "stage6_freerun",
        "stage6_eval_json": lane_root / "stage6_freerun" / "Walk_F_freerun_cycles.json",
        "stage6_group_json": lane_root / "stage6_group_summary.json",
    }


def ensure_basetrain_eval(lane: Dict[str, str], paths: Dict[str, Path]) -> None:
    if not paths["basetrain_group_json"].is_file():
        run_cmd(
            [
                sys.executable,
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                str(TEACHER),
                "--model",
                lane["ckpt"],
                "--rounds",
                "5",
                "--depth",
                "3",
                "--time-index-mode",
                "cycle",
                "--phase_reset_source",
                "none",
                "--contacts_meas_source",
                "pretrain_contact",
                "--contacts_meas_pretrain_clamp",
                PRETRAIN_CLAMP,
                "--contacts_meas_pretrain_affine_stats",
                str(AFFINE_STATS),
                "--encoder-bundle",
                str(ENCODER_BUNDLE),
                "--export_joint_direct_geolocal_series",
                "--out",
                str(paths["basetrain_eval_dir"]),
                "--force",
            ],
            log_file=paths["lane_log"],
        )
        run_cmd(
            [
                sys.executable,
                str(ROOT / "tools" / "phasea_group_summary.py"),
                str(paths["basetrain_eval_json"]),
                "--cycle_gte",
                "1",
                "--drop_wrap",
                "--out",
                str(paths["basetrain_group_json"]),
            ],
            log_file=paths["lane_log"],
        )


def ensure_stage6(lane: Dict[str, str], paths: Dict[str, Path]) -> None:
    if not paths["stage6_ckpt"].is_file() or not paths["stage6_log_json"].is_file():
        run_cmd(
            [
                sys.executable,
                "-m",
                "train.posttrain",
                "--config",
                str(CONFIG),
                "--ckpt_in",
                lane["ckpt"],
                "--out_dir",
                str(paths["model_root"]),
                "--run_name",
                str(paths["run_name"]),
                "--posttrain_contacts_source",
                "pretrain_contact",
                "--posttrain_contacts_pretrain_clamp",
                PRETRAIN_CLAMP,
                "--encoder_bundle",
                str(ENCODER_BUNDLE),
                "--posttrain_contacts_pretrain_affine_stats",
                str(AFFINE_STATS),
            ],
            log_file=paths["lane_log"],
        )
        extract_stage6_init(paths["stage6_log_json"], paths["stage6_init_json"])
    elif not paths["stage6_init_json"].is_file() and paths["stage6_log_json"].is_file():
        extract_stage6_init(paths["stage6_log_json"], paths["stage6_init_json"])


def ensure_stage6_eval(paths: Dict[str, Path]) -> None:
    if not paths["stage6_group_json"].is_file():
        run_cmd(
            [
                sys.executable,
                "-m",
                "train.validate.run_freerun_cycles",
                "--teacher",
                str(TEACHER),
                "--model",
                str(paths["stage6_ckpt"]),
                "--rounds",
                "5",
                "--depth",
                "3",
                "--time-index-mode",
                "cycle",
                "--phase_reset_source",
                "none",
                "--contacts_meas_source",
                "pretrain_contact",
                "--contacts_meas_pretrain_clamp",
                PRETRAIN_CLAMP,
                "--contacts_meas_pretrain_affine_stats",
                str(AFFINE_STATS),
                "--encoder-bundle",
                str(ENCODER_BUNDLE),
                "--export_joint_direct_geolocal_series",
                "--out",
                str(paths["stage6_eval_dir"]),
                "--force",
            ],
            log_file=paths["lane_log"],
        )
        run_cmd(
            [
                sys.executable,
                str(ROOT / "tools" / "phasea_group_summary.py"),
                str(paths["stage6_eval_json"]),
                "--cycle_gte",
                "1",
                "--drop_wrap",
                "--out",
                str(paths["stage6_group_json"]),
            ],
            log_file=paths["lane_log"],
        )


def build_summary() -> Dict[str, Any]:
    baseline_name = "old_bestfree"
    summary_rows: List[Dict[str, Any]] = []
    for lane in LANES:
        paths = lane_paths(lane["name"])
        basetrain_group = load_group_summary(paths["basetrain_group_json"])
        stage6_group = load_group_summary(paths["stage6_group_json"])
        stage6_init = json.loads(paths["stage6_init_json"].read_text())
        row = {
            "name": lane["name"],
            "family": lane["family"],
            "selector": lane["selector"],
            "ckpt": lane["ckpt"],
            "basetrain": {
                "all_ex_root_mean": safe_float(basetrain_group["groups"]["all_ex_root"]["mean"]),
                "leg_mean": safe_float(basetrain_group["groups"]["leg"]["mean"]),
                "nonleg_mean": safe_float(basetrain_group["groups"]["nonleg"]["mean"]),
                "arm_mean": safe_float(basetrain_group["groups"]["arm"]["mean"]),
                "else_mean": safe_float(basetrain_group["groups"]["else"]["mean"]),
            },
            "stage6_init": stage6_init,
            "stage6_exit": {
                "all_ex_root_mean": safe_float(stage6_group["groups"]["all_ex_root"]["mean"]),
                "leg_mean": safe_float(stage6_group["groups"]["leg"]["mean"]),
                "nonleg_mean": safe_float(stage6_group["groups"]["nonleg"]["mean"]),
                "arm_mean": safe_float(stage6_group["groups"]["arm"]["mean"]),
                "else_mean": safe_float(stage6_group["groups"]["else"]["mean"]),
            },
            "paths": {
                "basetrain_group_summary": str(paths["basetrain_group_json"]),
                "stage6_init_stats": str(paths["stage6_init_json"]),
                "stage6_group_summary": str(paths["stage6_group_json"]),
                "stage6_ckpt": str(paths["stage6_ckpt"]),
            },
        }
        summary_rows.append(row)

    baseline = next(row for row in summary_rows if row["name"] == baseline_name)
    for row in summary_rows:
        row["delta_vs_old_bestfree"] = {
            "basetrain_all_ex_root_mean": row["basetrain"]["all_ex_root_mean"] - baseline["basetrain"]["all_ex_root_mean"],
            "basetrain_leg_mean": row["basetrain"]["leg_mean"] - baseline["basetrain"]["leg_mean"],
            "stage6_step1_leg_over_nonleg": safe_float(row["stage6_init"]["step1"]["leg_over_nonleg"]) - safe_float(baseline["stage6_init"]["step1"]["leg_over_nonleg"]),
            "stage6_head20_leg_over_nonleg": safe_float(row["stage6_init"]["head20_mean"]["leg_over_nonleg"]) - safe_float(baseline["stage6_init"]["head20_mean"]["leg_over_nonleg"]),
            "stage6_exit_all_ex_root_mean": row["stage6_exit"]["all_ex_root_mean"] - baseline["stage6_exit"]["all_ex_root_mean"],
            "stage6_exit_leg_mean": row["stage6_exit"]["leg_mean"] - baseline["stage6_exit"]["leg_mean"],
            "stage6_exit_nonleg_mean": row["stage6_exit"]["nonleg_mean"] - baseline["stage6_exit"]["nonleg_mean"],
        }
    return {
        "run_date": RUN_DATE,
        "policy": {
            "stage6_config": str(CONFIG),
            "teacher": str(TEACHER),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contract": "docs/posttrain_pipeline.md Stage6 current mainline: pretrain_contact + affine_mix08 + shared encoder_bundle",
        },
        "baseline": baseline_name,
        "lanes": summary_rows,
    }


def build_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Stage6 basetrain compare")
    lines.append("")
    lines.append(f"- run_date: {summary['run_date']}")
    lines.append(f"- baseline: {summary['baseline']}")
    lines.append(f"- stage6_config: `{summary['policy']['stage6_config']}`")
    lines.append(f"- encoder_bundle: `{summary['policy']['encoder_bundle']}`")
    lines.append("")
    lines.append("## Basetrain endpoint")
    lines.append("")
    lines.append("| lane | selector | all_ex_root | leg | nonleg | arm | else |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary["lanes"]:
        b = row["basetrain"]
        lines.append(f"| {row['name']} | {row['selector']} | {b['all_ex_root_mean']:.6f} | {b['leg_mean']:.6f} | {b['nonleg_mean']:.6f} | {b['arm_mean']:.6f} | {b['else_mean']:.6f} |")
    lines.append("")
    lines.append("## Stage6 init")
    lines.append("")
    lines.append("| lane | step1 leg/nonleg | head20 leg/nonleg | head20 grad arm/else | head20 arm/else |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary["lanes"]:
        s = row["stage6_init"]
        lines.append(
            f"| {row['name']} | {safe_float(s['step1']['leg_over_nonleg']):.6f} | {safe_float(s['head20_mean']['leg_over_nonleg']):.6f} | {safe_float(s['head20_mean']['grad_arm_over_else']):.6f} | {safe_float(s['head20_mean']['arm_over_else']):.6f} |"
        )
    lines.append("")
    lines.append("## Stage6 exit")
    lines.append("")
    lines.append("| lane | all_ex_root | leg | nonleg | arm | else | delta all_ex_root vs old | delta leg vs old |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["lanes"]:
        s = row["stage6_exit"]
        d = row["delta_vs_old_bestfree"]
        lines.append(
            f"| {row['name']} | {s['all_ex_root_mean']:.6f} | {s['leg_mean']:.6f} | {s['nonleg_mean']:.6f} | {s['arm_mean']:.6f} | {s['else_mean']:.6f} | {d['stage6_exit_all_ex_root_mean']:+.6f} | {d['stage6_exit_leg_mean']:+.6f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    missing = [str(p) for p in [CONFIG, TEACHER, ENCODER_BUNDLE, AFFINE_STATS] if not p.is_file()]
    missing.extend(lane["ckpt"] for lane in LANES if not Path(lane["ckpt"]).is_file())
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_date": RUN_DATE,
        "root": str(ROOT),
        "config": str(CONFIG),
        "teacher": str(TEACHER),
        "encoder_bundle": str(ENCODER_BUNDLE),
        "affine_stats": str(AFFINE_STATS),
        "lanes": LANES,
    }
    write_json(DEBUG_ROOT / "manifest.json", manifest)

    for idx, lane in enumerate(LANES, start=1):
        paths = lane_paths(lane["name"])
        log(f"=== [{idx}/{len(LANES)}] lane={lane['name']} selector={lane['selector']} ===")
        paths["lane_root"].mkdir(parents=True, exist_ok=True)
        paths["model_root"].mkdir(parents=True, exist_ok=True)
        ensure_basetrain_eval(lane, paths)
        ensure_stage6(lane, paths)
        ensure_stage6_eval(paths)
        status = {
            "lane": lane,
            "basetrain_group_summary": str(paths["basetrain_group_json"]),
            "stage6_init_stats": str(paths["stage6_init_json"]),
            "stage6_group_summary": str(paths["stage6_group_json"]),
            "stage6_ckpt": str(paths["stage6_ckpt"]),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(paths["lane_root"] / "status.json", status)

    summary = build_summary()
    write_json(DEBUG_ROOT / "compare_summary.json", summary)
    (DEBUG_ROOT / "compare_summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={DEBUG_ROOT / 'compare_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
