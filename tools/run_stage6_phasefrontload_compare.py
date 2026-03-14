#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "20260314"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_stage6_phasefrontload_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
MODEL_ROOT = ROOT / "models" / f"__tmp_stage6_phasefrontload_{RUN_DATE}"
BASELINE_COMPARE_JSON = ROOT / "debug_output" / "_tmp_stage6_basetrain_compare_20260313" / "compare_summary.json"
PLANTRANSPLANT_JSON = ROOT / "debug_output" / "_tmp_stage6_plantransplant_20260314" / "summary.json"
BASE_STAGE6_CONFIG = ROOT / "config" / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json"
TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
PRETRAIN_CLAMP = "1.0"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    baseline_case: str
    run_name: str
    semantic_overrides: Dict[str, Any]
    note: str


CASES: Sequence[CaseSpec] = (
    CaseSpec(
        name="cp015_stage6_phasezin_frontload",
        baseline_case="cp015_bestfree",
        run_name="cp015_stage6_phasezin_frontload_20260314",
        semantic_overrides={
            "direct_pose_use_phase_z": True,
            "direct_pose_phase_z_mode": "concat",
        },
        note="Stage6 frontloads historical 70b concat semantics while keeping Stage6 reinit/train budget unchanged.",
    ),
    CaseSpec(
        name="cp015_stage6_replacecontacts_frontload",
        baseline_case="cp015_bestfree",
        run_name="cp015_stage6_replacecontacts_frontload_20260314",
        semantic_overrides={
            "direct_pose_use_phase_z": True,
            "direct_pose_phase_z_mode": "replace_contacts",
        },
        note="Stage6 frontloads historical 70c replace_contacts semantics while keeping Stage6 reinit/train budget unchanged.",
    ),
)


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


def is_finite(x: Any) -> bool:
    return math.isfinite(safe_float(x))


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


def improvement(baseline: Any, cur: Any) -> float:
    base = safe_float(baseline)
    now = safe_float(cur)
    if not math.isfinite(base) or not math.isfinite(now):
        return float("nan")
    return float(base - now)


def fmt(x: Any, digits: int = 6) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def enrich_init_section(section: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(section)
    dir_leg = safe_float(payload.get("dir_leg_base"))
    dir_nonleg = safe_float(payload.get("dir_nonleg_base"))
    grad_arm = safe_float(payload.get("direct_grad_norm_out_arm"))
    grad_else = safe_float(payload.get("direct_grad_norm_out_else"))
    if not is_finite(payload.get("leg_over_nonleg")):
        if math.isfinite(dir_leg) and math.isfinite(dir_nonleg) and dir_nonleg != 0.0:
            payload["leg_over_nonleg"] = float(dir_leg / dir_nonleg)
        else:
            payload["leg_over_nonleg"] = float("nan")
    if not is_finite(payload.get("grad_arm_over_else")):
        if math.isfinite(grad_arm) and math.isfinite(grad_else) and grad_else != 0.0:
            payload["grad_arm_over_else"] = float(grad_arm / grad_else)
        else:
            payload["grad_arm_over_else"] = float("nan")
    return payload


def extract_stage6_init(log_json: Path, out_json: Path) -> Dict[str, Any]:
    obj = load_json(log_json)
    rows = obj.get("log", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"missing log rows in {log_json}")

    def row_payload(row: Mapping[str, Any]) -> Dict[str, float]:
        return {
            "step": safe_float(row.get("step")),
            "dir_leg_base": safe_float(row.get("dir_leg_base")),
            "dir_nonleg_base": safe_float(row.get("dir_nonleg_base")),
            "leg_over_nonleg": safe_float(row.get("leg_over_nonleg")),
            "arm_over_else": safe_float(row.get("arm_over_else")),
            "direct_grad_norm_out_arm": safe_float(row.get("direct_grad_norm_out_arm")),
            "direct_grad_norm_out_else": safe_float(row.get("direct_grad_norm_out_else")),
            "grad_arm_over_else": safe_float(row.get("grad_arm_over_else")),
        }

    step1 = enrich_init_section(row_payload(rows[0]))
    head = [enrich_init_section(row_payload(row)) for row in rows[: min(20, len(rows))]]
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
        "head20_mean": enrich_init_section(head20),
    }
    write_json(out_json, payload)
    return payload


def group_metrics(path: Path) -> Dict[str, float]:
    groups = load_json(path).get("groups", {})
    return {
        "all_ex_root": safe_float(groups.get("all_ex_root", {}).get("mean")),
        "leg": safe_float(groups.get("leg", {}).get("mean")),
        "nonleg": safe_float(groups.get("nonleg", {}).get("mean")),
        "arm": safe_float(groups.get("arm", {}).get("mean")),
        "else": safe_float(groups.get("else", {}).get("mean")),
    }


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


def lane_paths(spec: CaseSpec) -> Dict[str, Path]:
    lane_root = OUT_ROOT / spec.name
    model_dir = MODEL_ROOT / spec.name
    return {
        "lane_root": lane_root,
        "lane_log": lane_root / "lane.log",
        "config_json": CONFIG_ROOT / f"{spec.name}.json",
        "model_dir": model_dir,
        "stage6_log_json": model_dir / f"posttrain_log_{spec.run_name}.json",
        "stage6_ckpt": model_dir / f"ckpt_last_{spec.run_name}.pth",
        "stage6_init_json": lane_root / "posttrain_stage6_init_stats.json",
        "stage6_eval_dir": lane_root / "stage6_freerun",
        "stage6_eval_json": lane_root / "stage6_freerun" / "Walk_F_freerun_cycles.json",
        "stage6_group_json": lane_root / "stage6_group_summary.json",
        "status_json": lane_root / "status.json",
    }


def load_reference_rows() -> Dict[str, Dict[str, Any]]:
    baseline = load_json(BASELINE_COMPARE_JSON)
    lanes = baseline.get("lanes", [])
    if not isinstance(lanes, list):
        raise RuntimeError(f"invalid lanes in {BASELINE_COMPARE_JSON}")
    rows = {str(row["name"]): row for row in lanes if isinstance(row, dict) and "name" in row}

    transplant = load_json(PLANTRANSPLANT_JSON)
    cases = transplant.get("cases", [])
    for case in cases:
        if isinstance(case, dict) and "name" in case:
            name = str(case["name"])
            if name not in rows:
                rows[name] = case
    return rows


def reference_stage6_exit(row: Mapping[str, Any]) -> Dict[str, float]:
    if "stage6_exit" in row:
        block = row.get("stage6_exit", {})
        return {
            "all_ex_root": safe_float(block.get("all_ex_root", block.get("all_ex_root_mean"))),
            "leg": safe_float(block.get("leg", block.get("leg_mean"))),
            "nonleg": safe_float(block.get("nonleg", block.get("nonleg_mean"))),
            "arm": safe_float(block.get("arm", block.get("arm_mean"))),
            "else": safe_float(block.get("else", block.get("else_mean"))),
        }
    raise RuntimeError(f"row missing stage6_exit: {row}")


def build_case_config(spec: CaseSpec, *, base_cfg: Mapping[str, Any], baseline_ckpt: str, paths: Mapping[str, Path]) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    cfg["run_name"] = spec.run_name
    cfg["out_dir"] = str(paths["model_dir"])
    cfg["ckpt_in"] = str(baseline_ckpt)
    for key, value in spec.semantic_overrides.items():
        cfg[key] = value
    return cfg


def semantic_diff(base_cfg: Mapping[str, Any], case_cfg: Mapping[str, Any], spec: CaseSpec) -> Dict[str, Dict[str, Any]]:
    diff_block: Dict[str, Dict[str, Any]] = {}
    for key in sorted(spec.semantic_overrides.keys()):
        diff_block[key] = {
            "base": base_cfg.get(key),
            "case": case_cfg.get(key),
        }
    return diff_block


def ensure_case(spec: CaseSpec, *, base_cfg: Mapping[str, Any], reference_rows: Mapping[str, Dict[str, Any]]) -> None:
    paths = lane_paths(spec)
    baseline_row = reference_rows[spec.baseline_case]
    baseline_ckpt = str(baseline_row["ckpt"])
    case_cfg = build_case_config(spec, base_cfg=base_cfg, baseline_ckpt=baseline_ckpt, paths=paths)

    if not paths["config_json"].is_file():
        write_json(paths["config_json"], case_cfg)

    if not paths["stage6_ckpt"].is_file() or not paths["stage6_log_json"].is_file():
        run_cmd(
            [
                sys.executable,
                "-m",
                "train.posttrain",
                "--config",
                str(paths["config_json"]),
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
    elif not paths["stage6_init_json"].is_file():
        extract_stage6_init(paths["stage6_log_json"], paths["stage6_init_json"])

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

    write_json(
        paths["status_json"],
        {
            "case": spec.__dict__,
            "config_json": str(paths["config_json"]),
            "stage6_ckpt": str(paths["stage6_ckpt"]),
            "stage6_group_summary": str(paths["stage6_group_json"]),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def build_summary(base_cfg: Mapping[str, Any], reference_rows: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    refs = {
        "cp015_bestfree": reference_stage6_exit(reference_rows["cp015_bestfree"]),
        "old_bestfree": reference_stage6_exit(reference_rows["old_bestfree"]),
        "cp015_with_old_planstack": reference_stage6_exit(reference_rows["cp015_with_old_planstack"]),
    }

    cases: List[Dict[str, Any]] = []
    best_case_name = None
    best_case_value = float("inf")

    for spec in CASES:
        paths = lane_paths(spec)
        case_cfg = load_json(paths["config_json"])
        stage6_init = load_json(paths["stage6_init_json"])
        stage6_exit = group_metrics(paths["stage6_group_json"])

        if stage6_exit["all_ex_root"] < best_case_value:
            best_case_value = stage6_exit["all_ex_root"]
            best_case_name = spec.name

        cases.append(
            {
                "name": spec.name,
                "baseline_case": spec.baseline_case,
                "note": spec.note,
                "config_json": str(paths["config_json"]),
                "config_diff_vs_base": semantic_diff(base_cfg, case_cfg, spec),
                "runtime_contract": {
                    "config_base": str(BASE_STAGE6_CONFIG),
                    "posttrain_contacts_source": "pretrain_contact",
                    "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
                    "encoder_bundle": str(ENCODER_BUNDLE),
                    "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
                    "validate_teacher": str(TEACHER),
                    "validate_rounds": 5,
                    "validate_depth": 3,
                    "validate_time_index_mode": "cycle",
                    "validate_phase_reset_source": "none",
                    "validate_contacts_meas_source": "pretrain_contact",
                    "validate_contacts_meas_pretrain_clamp": PRETRAIN_CLAMP,
                    "validate_contacts_meas_pretrain_affine_stats": str(AFFINE_STATS),
                    "group_summary_args": ["--cycle_gte", "1", "--drop_wrap"],
                },
                "stage6_init": stage6_init,
                "stage6_exit": stage6_exit,
                "improvement_vs_cp015_bestfree": {
                    key: improvement(refs["cp015_bestfree"].get(key), stage6_exit.get(key))
                    for key in ("all_ex_root", "leg", "nonleg", "arm", "else")
                },
                "delta_vs_old_bestfree": {
                    key: diff(stage6_exit.get(key), refs["old_bestfree"].get(key))
                    for key in ("all_ex_root", "leg", "nonleg", "arm", "else")
                },
                "delta_vs_cp015_with_old_planstack": {
                    key: diff(stage6_exit.get(key), refs["cp015_with_old_planstack"].get(key))
                    for key in ("all_ex_root", "leg", "nonleg", "arm", "else")
                },
                "paths": {
                    "lane_root": str(paths["lane_root"]),
                    "lane_log": str(paths["lane_log"]),
                    "stage6_ckpt": str(paths["stage6_ckpt"]),
                    "stage6_init_stats": str(paths["stage6_init_json"]),
                    "stage6_eval_json": str(paths["stage6_eval_json"]),
                    "stage6_group_summary": str(paths["stage6_group_json"]),
                },
            }
        )

    case_map = {case["name"]: case for case in cases}
    phasezin = case_map["cp015_stage6_phasezin_frontload"]
    replace = case_map["cp015_stage6_replacecontacts_frontload"]

    def closer_to_old(case: Mapping[str, Any]) -> float:
        return abs(safe_float(case["delta_vs_old_bestfree"]["all_ex_root"]))

    closest_to_old = min(cases, key=closer_to_old)["name"]

    answers = {
        "q1_phasezin_frontload_improvement_vs_cp015_bestfree": phasezin["improvement_vs_cp015_bestfree"],
        "q2_replacecontacts_frontload_improvement_vs_cp015_bestfree": replace["improvement_vs_cp015_bestfree"],
        "q3_closest_to_old_bestfree": {
            "case": closest_to_old,
            "all_ex_root_gap": case_map[closest_to_old]["delta_vs_old_bestfree"]["all_ex_root"],
            "old_bestfree_all_ex_root": refs["old_bestfree"]["all_ex_root"],
        },
        "q4_replacecontacts_vs_cp015_with_old_planstack": {
            "replacecontacts_all_ex_root": replace["stage6_exit"]["all_ex_root"],
            "cp015_with_old_planstack_all_ex_root": refs["cp015_with_old_planstack"]["all_ex_root"],
            "gap_all_ex_root": replace["delta_vs_cp015_with_old_planstack"]["all_ex_root"],
            "leg_gap": replace["delta_vs_cp015_with_old_planstack"]["leg"],
            "nonleg_gap": replace["delta_vs_cp015_with_old_planstack"]["nonleg"],
        },
        "best_frontload_case": best_case_name,
    }

    return {
        "run_date": RUN_DATE,
        "policy": {
            "base_stage6_config": str(BASE_STAGE6_CONFIG),
            "baseline_compare_summary": str(BASELINE_COMPARE_JSON),
            "plantransplant_summary": str(PLANTRANSPLANT_JSON),
            "teacher": str(TEACHER),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "validate_contacts_meas_source": "pretrain_contact",
            "validate_contacts_meas_pretrain_clamp": PRETRAIN_CLAMP,
            "validate_phase_reset_source": "none",
        },
        "reference_stage6_exit": refs,
        "cases": cases,
        "answers": answers,
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    refs = summary["reference_stage6_exit"]
    cases = summary["cases"]
    answers = summary["answers"]
    lines: List[str] = []
    lines.append("# Stage6 phase-hint frontload")
    lines.append("")
    lines.append(f"- run_date: {summary['run_date']}")
    lines.append(f"- base_stage6_config: `{summary['policy']['base_stage6_config']}`")
    lines.append(f"- baseline_compare_summary: `{summary['policy']['baseline_compare_summary']}`")
    lines.append(f"- plantransplant_summary: `{summary['policy']['plantransplant_summary']}`")
    lines.append("")
    lines.append("## Reference Stage6 exits")
    lines.append("")
    lines.append("| reference | all_ex_root | leg | nonleg |")
    lines.append("|---|---:|---:|---:|")
    for name in ("cp015_bestfree", "old_bestfree", "cp015_with_old_planstack"):
        row = refs[name]
        lines.append(f"| {name} | {fmt(row['all_ex_root'])} | {fmt(row['leg'])} | {fmt(row['nonleg'])} |")
    lines.append("")
    lines.append("## Case config diffs")
    lines.append("")
    lines.append("| case | direct_pose_use_phase_z | direct_pose_phase_z_mode | direct_pose_reinit(base) |")
    lines.append("|---|---|---|---|")
    base_cfg = load_json(Path(summary["policy"]["base_stage6_config"]))
    for case in cases:
        diff_block = case["config_diff_vs_base"]
        lines.append(
            f"| {case['name']} | {diff_block['direct_pose_use_phase_z']['case']} | "
            f"{diff_block['direct_pose_phase_z_mode']['case']} | {base_cfg.get('direct_pose_reinit')} |"
        )
    lines.append("")
    lines.append("## Stage6 exit")
    lines.append("")
    lines.append("| case | all_ex_root | leg | nonleg | delta_vs_old | delta_vs_cp015_old_plan | improve_vs_cp015 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for case in cases:
        s = case["stage6_exit"]
        d_old = case["delta_vs_old_bestfree"]
        d_plan = case["delta_vs_cp015_with_old_planstack"]
        imp = case["improvement_vs_cp015_bestfree"]
        lines.append(
            f"| {case['name']} | {fmt(s['all_ex_root'])} | {fmt(s['leg'])} | {fmt(s['nonleg'])} | "
            f"{fmt(d_old['all_ex_root'])} | {fmt(d_plan['all_ex_root'])} | {fmt(imp['all_ex_root'])} |"
        )
    lines.append("")
    lines.append("## Stage6 init")
    lines.append("")
    lines.append("| case | step1 leg/nonleg | head20 leg/nonleg | head20 grad arm/else |")
    lines.append("|---|---:|---:|---:|")
    for case in cases:
        init = case["stage6_init"]
        lines.append(
            f"| {case['name']} | {fmt(init['step1'].get('leg_over_nonleg'))} | "
            f"{fmt(init['head20_mean'].get('leg_over_nonleg'))} | {fmt(init['head20_mean'].get('grad_arm_over_else'))} |"
        )
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    q1 = answers["q1_phasezin_frontload_improvement_vs_cp015_bestfree"]
    q2 = answers["q2_replacecontacts_frontload_improvement_vs_cp015_bestfree"]
    q3 = answers["q3_closest_to_old_bestfree"]
    q4 = answers["q4_replacecontacts_vs_cp015_with_old_planstack"]
    lines.append(
        f"1. `phasezin_frontload` vs `cp015_bestfree` (delta = baseline - case): "
        f"all_ex_root {fmt(q1['all_ex_root'])}, leg {fmt(q1['leg'])}, nonleg {fmt(q1['nonleg'])}."
    )
    lines.append(
        f"2. `replacecontacts_frontload` vs `cp015_bestfree` (delta = baseline - case): "
        f"all_ex_root {fmt(q2['all_ex_root'])}, leg {fmt(q2['leg'])}, nonleg {fmt(q2['nonleg'])}."
    )
    lines.append(
        f"3. Closest to `old_bestfree` is `{q3['case']}` with all_ex_root gap {fmt(q3['all_ex_root_gap'])} "
        f"(old_bestfree={fmt(q3['old_bestfree_all_ex_root'])})."
    )
    lines.append(
        f"4. `replacecontacts_frontload` vs `cp015_with_old_planstack`: all_ex_root gap {fmt(q4['gap_all_ex_root'])}, "
        f"leg gap {fmt(q4['leg_gap'])}, nonleg gap {fmt(q4['nonleg_gap'])}."
    )
    lines.append(f"5. Recommended Stage2 seed: `{answers['best_frontload_case']}`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [
        BASELINE_COMPARE_JSON,
        PLANTRANSPLANT_JSON,
        BASE_STAGE6_CONFIG,
        TEACHER,
        ENCODER_BUNDLE,
        AFFINE_STATS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    reference_rows = load_reference_rows()
    needed_rows = {"cp015_bestfree", "old_bestfree", "cp015_with_old_planstack"}
    missing_rows = sorted(needed_rows - set(reference_rows.keys()))
    if missing_rows:
        raise SystemExit("missing reference rows:\n" + "\n".join(missing_rows))

    base_cfg = load_json(BASE_STAGE6_CONFIG)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    write_json(
        OUT_ROOT / "manifest.json",
        {
            "run_date": RUN_DATE,
            "root": str(ROOT),
            "base_stage6_config": str(BASE_STAGE6_CONFIG),
            "baseline_compare_summary": str(BASELINE_COMPARE_JSON),
            "plantransplant_summary": str(PLANTRANSPLANT_JSON),
            "teacher": str(TEACHER),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "cases": [spec.__dict__ for spec in CASES],
        },
    )

    for idx, spec in enumerate(CASES, start=1):
        log(f"=== [{idx}/{len(CASES)}] {spec.name} ===")
        ensure_case(spec, base_cfg=base_cfg, reference_rows=reference_rows)

    summary = build_summary(base_cfg, reference_rows)
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
