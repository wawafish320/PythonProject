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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "20260314"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_phasefrontload_chain_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_phasefrontload_chain_{RUN_DATE}"
STAGE6_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_stage6_phasefrontload_20260314" / "summary.json"
TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
PRETRAIN_CLAMP = "1.0"

CONFIG_70R = ROOT / "config" / "posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json"
CONFIG_71 = ROOT / "config" / "posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json"
CONFIG_72 = ROOT / "config" / "posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json"
CONFIG_LAMBDA = ROOT / "config" / "posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json"

HIST_BLEND_PATH = "debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_blend/summary_metrics.txt"
HIST_DIRECT_PATH = "debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_direct/global_signal_summary.txt"


@dataclass(frozen=True)
class LaneSpec:
    name: str
    use_70r: bool
    note: str


LANES: Sequence[LaneSpec] = (
    LaneSpec(
        name="minimal_skip70R",
        use_70r=False,
        note="Primary simplification test: best frontloaded Stage6 -> 71 -> 72 -> lambda final.",
    ),
    LaneSpec(
        name="control_with70R",
        use_70r=True,
        note="Conservative control: keep a 70R recovery stage between best frontloaded Stage6 and 71.",
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


def fmt(x: Any, digits: int = 6) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


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


def run_git_show(path_in_head: str) -> Optional[str]:
    proc = subprocess.run(
        ["git", "show", f"HEAD:{path_in_head}"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def parse_historical_reference() -> Dict[str, Any]:
    blend_txt = run_git_show(HIST_BLEND_PATH)
    direct_txt = run_git_show(HIST_DIRECT_PATH)

    out: Dict[str, Any] = {
        "source_blend_summary": HIST_BLEND_PATH,
        "source_direct_summary": HIST_DIRECT_PATH,
        "blend_masked_means": {},
        "direct_group_summary": {},
    }
    if blend_txt:
        metric_re = re.compile(r"^([A-Za-z0-9]+): .*?new_mean=([-+0-9.eE]+)")
        for line in blend_txt.splitlines():
            m = metric_re.match(line.strip())
            if m:
                out["blend_masked_means"][m.group(1)] = safe_float(m.group(2))
    if direct_txt:
        for line in direct_txt.splitlines():
            s = line.strip()
            if s.startswith("mean_new="):
                out["direct_group_summary"]["all_ex_root"] = safe_float(s.split("=", 1)[1])
            elif s.startswith("leg8_mean_new="):
                out["direct_group_summary"]["leg"] = safe_float(s.split("=", 1)[1])
            elif s.startswith("non_leg_mean_new="):
                out["direct_group_summary"]["nonleg"] = safe_float(s.split("=", 1)[1])
    return out


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

    metrics = (
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
    for key in metrics:
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


def lane_paths(frontload_case: str, lane: LaneSpec) -> Dict[str, Path]:
    lane_root = OUT_ROOT / f"{frontload_case}__{lane.name}"
    model_dir = MODEL_ROOT / f"{frontload_case}__{lane.name}"
    return {
        "lane_root": lane_root,
        "lane_log": lane_root / "lane.log",
        "model_dir": model_dir,
        "ckpt_70r": model_dir / f"ckpt_last_{frontload_case}_{lane.name}_70R_20260314.pth",
        "ckpt_71": model_dir / f"ckpt_last_{frontload_case}_{lane.name}_71_20260314.pth",
        "ckpt_72": model_dir / f"ckpt_last_{frontload_case}_{lane.name}_72_20260314.pth",
        "ckpt_lambda": model_dir / f"ckpt_last_{frontload_case}_{lane.name}_lambda_20260314.pth",
        "eval_pretrain_dir": lane_root / "eval_pretrain_contact",
        "eval_pretrain_json": lane_root / "eval_pretrain_contact" / "Walk_F_freerun_cycles.json",
        "eval_pretrain_group_json": lane_root / "eval_pretrain_contact_group_summary.json",
        "eval_model_dir": lane_root / "eval_model_source",
        "eval_model_json": lane_root / "eval_model_source" / "Walk_F_freerun_cycles.json",
        "eval_model_group_json": lane_root / "eval_model_source_group_summary.json",
        "status_json": lane_root / "status.json",
    }


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


def ensure_lane(
    *,
    frontload_case: str,
    frontload_stage6_ckpt: Path,
    lane: LaneSpec,
) -> Dict[str, Any]:
    paths = lane_paths(frontload_case, lane)
    model_dir = paths["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    stage_runs: List[Dict[str, Any]] = []
    ckpt_prev = frontload_stage6_ckpt

    if lane.use_70r:
        run_name = f"{frontload_case}_{lane.name}_70R_20260314"
        ckpt_prev = run_posttrain_stage(
            config=CONFIG_70R,
            ckpt_in=ckpt_prev,
            out_dir=model_dir,
            run_name=run_name,
            log_file=paths["lane_log"],
        )
        stage_runs.append({"stage": "70R", "config": str(CONFIG_70R), "run_name": run_name, "ckpt": str(ckpt_prev)})

    run_name_71 = f"{frontload_case}_{lane.name}_71_20260314"
    ckpt_71 = run_posttrain_stage(
        config=CONFIG_71,
        ckpt_in=ckpt_prev,
        out_dir=model_dir,
        run_name=run_name_71,
        log_file=paths["lane_log"],
    )
    stage_runs.append({"stage": "71", "config": str(CONFIG_71), "run_name": run_name_71, "ckpt": str(ckpt_71)})

    run_name_72 = f"{frontload_case}_{lane.name}_72_20260314"
    ckpt_72 = run_posttrain_stage(
        config=CONFIG_72,
        ckpt_in=ckpt_71,
        out_dir=model_dir,
        run_name=run_name_72,
        log_file=paths["lane_log"],
    )
    stage_runs.append({"stage": "72", "config": str(CONFIG_72), "run_name": run_name_72, "ckpt": str(ckpt_72)})

    run_name_lambda = f"{frontload_case}_{lane.name}_lambda_20260314"
    ckpt_lambda = run_posttrain_stage(
        config=CONFIG_LAMBDA,
        ckpt_in=ckpt_72,
        out_dir=model_dir,
        run_name=run_name_lambda,
        log_file=paths["lane_log"],
    )
    stage_runs.append({"stage": "lambda_final", "config": str(CONFIG_LAMBDA), "run_name": run_name_lambda, "ckpt": str(ckpt_lambda)})

    eval_pretrain_json = run_eval(
        model_ckpt=ckpt_lambda,
        out_dir=paths["eval_pretrain_dir"],
        contacts_source="pretrain_contact",
        log_file=paths["lane_log"],
    )
    ensure_group_summary(eval_pretrain_json, paths["eval_pretrain_group_json"], log_file=paths["lane_log"])

    eval_model_json = run_eval(
        model_ckpt=ckpt_lambda,
        out_dir=paths["eval_model_dir"],
        contacts_source="model",
        log_file=paths["lane_log"],
    )
    ensure_group_summary(eval_model_json, paths["eval_model_group_json"], log_file=paths["lane_log"])

    payload = {
        "lane": lane.__dict__,
        "frontload_case": frontload_case,
        "stage_runs": stage_runs,
        "eval_pretrain_contact": {
            "eval_json": str(eval_pretrain_json),
            "group_summary": str(paths["eval_pretrain_group_json"]),
        },
        "eval_model_source": {
            "eval_json": str(eval_model_json),
            "group_summary": str(paths["eval_model_group_json"]),
        },
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(paths["status_json"], payload)
    return payload


def build_lane_entry(frontload_case: str, lane: LaneSpec) -> Dict[str, Any]:
    paths = lane_paths(frontload_case, lane)
    pretrain_masked = masked_metric_means(paths["eval_pretrain_json"])
    pretrain_group = group_metrics(paths["eval_pretrain_group_json"])
    model_masked = masked_metric_means(paths["eval_model_json"])
    model_group = group_metrics(paths["eval_model_group_json"])
    status = load_json(paths["status_json"])
    return {
        "name": lane.name,
        "use_70r": lane.use_70r,
        "note": lane.note,
        "stage_runs": status.get("stage_runs", []),
        "pretrain_contact_eval": {
            "masked_means": pretrain_masked,
            "direct_group_summary": pretrain_group,
            "paths": {
                "eval_json": str(paths["eval_pretrain_json"]),
                "group_summary": str(paths["eval_pretrain_group_json"]),
            },
        },
        "model_source_eval": {
            "masked_means": model_masked,
            "direct_group_summary": model_group,
            "paths": {
                "eval_json": str(paths["eval_model_json"]),
                "group_summary": str(paths["eval_model_group_json"]),
            },
        },
        "paths": {
            "lane_root": str(paths["lane_root"]),
            "lane_log": str(paths["lane_log"]),
            "status_json": str(paths["status_json"]),
        },
    }


def pick_frontload_case(summary: Mapping[str, Any], preferred_case: Optional[str]) -> Dict[str, Any]:
    cases = summary.get("cases", [])
    if not isinstance(cases, list):
        raise RuntimeError(f"invalid cases in {STAGE6_SUMMARY_JSON}")
    case_map = {str(case["name"]): case for case in cases if isinstance(case, dict) and "name" in case}
    chosen = preferred_case or str(summary.get("answers", {}).get("best_frontload_case", "") or "")
    if chosen and chosen in case_map:
        return case_map[chosen]
    if not cases:
        raise RuntimeError(f"no cases in {STAGE6_SUMMARY_JSON}")
    best = min(cases, key=lambda case: safe_float(case.get("stage6_exit", {}).get("all_ex_root")))
    return best


def build_summary(frontload_case: Mapping[str, Any], lane_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hist = parse_historical_reference()
    lane_map = {str(entry["name"]): dict(entry) for entry in lane_entries}

    answers: Dict[str, Any] = {}
    minimal = lane_map.get("minimal_skip70R")
    control = lane_map.get("control_with70R")
    hist_blend = hist.get("blend_masked_means", {})
    hist_direct = hist.get("direct_group_summary", {})

    if minimal:
        answers["minimal_vs_historical_accepted_model_source"] = {
            "DirectGeoLocalDeg_delta": diff(
                minimal["model_source_eval"]["masked_means"].get("DirectGeoLocalDeg"),
                hist_blend.get("DirectGeoLocalDeg"),
            ),
            "BlendGeoLocalDeg_delta": diff(
                minimal["model_source_eval"]["masked_means"].get("BlendGeoLocalDeg"),
                hist_blend.get("BlendGeoLocalDeg"),
            ),
            "all_ex_root_delta": diff(
                minimal["model_source_eval"]["direct_group_summary"].get("all_ex_root"),
                hist_direct.get("all_ex_root"),
            ),
            "leg_delta": diff(
                minimal["model_source_eval"]["direct_group_summary"].get("leg"),
                hist_direct.get("leg"),
            ),
            "nonleg_delta": diff(
                minimal["model_source_eval"]["direct_group_summary"].get("nonleg"),
                hist_direct.get("nonleg"),
            ),
        }

    if minimal and control:
        answers["minimal_vs_with70R_pretrain_contact"] = {
            "DirectGeoLocalDeg_improvement_without_70R": improvement(
                control["pretrain_contact_eval"]["masked_means"].get("DirectGeoLocalDeg"),
                minimal["pretrain_contact_eval"]["masked_means"].get("DirectGeoLocalDeg"),
            ),
            "BlendGeoLocalDeg_improvement_without_70R": improvement(
                control["pretrain_contact_eval"]["masked_means"].get("BlendGeoLocalDeg"),
                minimal["pretrain_contact_eval"]["masked_means"].get("BlendGeoLocalDeg"),
            ),
            "all_ex_root_improvement_without_70R": improvement(
                control["pretrain_contact_eval"]["direct_group_summary"].get("all_ex_root"),
                minimal["pretrain_contact_eval"]["direct_group_summary"].get("all_ex_root"),
            ),
            "leg_improvement_without_70R": improvement(
                control["pretrain_contact_eval"]["direct_group_summary"].get("leg"),
                minimal["pretrain_contact_eval"]["direct_group_summary"].get("leg"),
            ),
            "nonleg_improvement_without_70R": improvement(
                control["pretrain_contact_eval"]["direct_group_summary"].get("nonleg"),
                minimal["pretrain_contact_eval"]["direct_group_summary"].get("nonleg"),
            ),
        }

    return {
        "run_date": RUN_DATE,
        "frontload_source_summary": str(STAGE6_SUMMARY_JSON),
        "frontload_case": frontload_case,
        "policy": {
            "teacher": str(TEACHER),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "final_eval_pretrain_contact": {
                "rounds": 5,
                "depth": 3,
                "time_index_mode": "cycle",
                "event_clock": "auto",
                "phase_reset_source": "none",
                "contacts_meas_source": "pretrain_contact",
                "contacts_meas_pretrain_clamp": PRETRAIN_CLAMP,
            },
            "final_eval_model_source": {
                "rounds": 5,
                "depth": 3,
                "time_index_mode": "cycle",
                "event_clock": "auto",
                "phase_reset_source": "none",
                "contacts_meas_source": "model",
            },
            "chain_configs": {
                "70R": str(CONFIG_70R),
                "71": str(CONFIG_71),
                "72": str(CONFIG_72),
                "lambda_final": str(CONFIG_LAMBDA),
            },
        },
        "historical_accepted_model_source_reference": hist,
        "lanes": list(lane_entries),
        "answers": answers,
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    frontload_case = summary["frontload_case"]
    hist = summary["historical_accepted_model_source_reference"]
    lanes = summary["lanes"]
    answers = summary["answers"]
    lines: List[str] = []
    lines.append("# Phasefrontload downstream chain")
    lines.append("")
    lines.append(f"- run_date: {summary['run_date']}")
    lines.append(f"- frontload_case: `{frontload_case['name']}`")
    lines.append(f"- frontload_stage6_ckpt: `{frontload_case['paths']['stage6_ckpt']}`")
    lines.append(f"- frontload_stage6_all_ex_root: {fmt(frontload_case['stage6_exit']['all_ex_root'])}")
    lines.append("")
    lines.append("## Historical accepted final (model-source reference)")
    lines.append("")
    lines.append("| metric | accepted_final |")
    lines.append("|---|---:|")
    for key in ("DirectGeoLocalDeg", "BlendGeoLocalDeg", "GeoLocalDeg"):
        lines.append(f"| {key} | {fmt(hist.get('blend_masked_means', {}).get(key))} |")
    for key in ("all_ex_root", "leg", "nonleg"):
        lines.append(f"| direct_{key} | {fmt(hist.get('direct_group_summary', {}).get(key))} |")
    lines.append("")
    lines.append("## Current lanes (pretrain_contact eval)")
    lines.append("")
    lines.append("| lane | use_70r | DirectGeoLocalDeg | BlendGeoLocalDeg | all_ex_root | leg | nonleg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for lane in lanes:
        masked = lane["pretrain_contact_eval"]["masked_means"]
        group = lane["pretrain_contact_eval"]["direct_group_summary"]
        lines.append(
            f"| {lane['name']} | {str(bool(lane['use_70r'])).lower()} | {fmt(masked.get('DirectGeoLocalDeg'))} | "
            f"{fmt(masked.get('BlendGeoLocalDeg'))} | {fmt(group.get('all_ex_root'))} | {fmt(group.get('leg'))} | {fmt(group.get('nonleg'))} |"
        )
    lines.append("")
    lines.append("## Current lanes (model-source eval)")
    lines.append("")
    lines.append("| lane | DirectGeoLocalDeg | BlendGeoLocalDeg | all_ex_root | leg | nonleg |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for lane in lanes:
        masked = lane["model_source_eval"]["masked_means"]
        group = lane["model_source_eval"]["direct_group_summary"]
        lines.append(
            f"| {lane['name']} | {fmt(masked.get('DirectGeoLocalDeg'))} | {fmt(masked.get('BlendGeoLocalDeg'))} | "
            f"{fmt(group.get('all_ex_root'))} | {fmt(group.get('leg'))} | {fmt(group.get('nonleg'))} |"
        )
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    if "minimal_vs_with70R_pretrain_contact" in answers:
        q = answers["minimal_vs_with70R_pretrain_contact"]
        lines.append(
            f"1. `minimal_skip70R` vs `control_with70R` (pretrain_contact): "
            f"DirectGeoLocalDeg improvement_without_70R={fmt(q['DirectGeoLocalDeg_improvement_without_70R'])}, "
            f"BlendGeoLocalDeg improvement_without_70R={fmt(q['BlendGeoLocalDeg_improvement_without_70R'])}, "
            f"all_ex_root improvement_without_70R={fmt(q['all_ex_root_improvement_without_70R'])}."
        )
    if "minimal_vs_historical_accepted_model_source" in answers:
        q = answers["minimal_vs_historical_accepted_model_source"]
        lines.append(
            f"2. `minimal_skip70R` vs historical accepted final (model-source): "
            f"DirectGeoLocalDeg delta={fmt(q['DirectGeoLocalDeg_delta'])}, "
            f"BlendGeoLocalDeg delta={fmt(q['BlendGeoLocalDeg_delta'])}, "
            f"all_ex_root delta={fmt(q['all_ex_root_delta'])}."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontload-case", type=str, default=None)
    args = ap.parse_args()

    required = [
        STAGE6_SUMMARY_JSON,
        TEACHER,
        ENCODER_BUNDLE,
        AFFINE_STATS,
        CONFIG_71,
        CONFIG_72,
        CONFIG_LAMBDA,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    stage6_summary = load_json(STAGE6_SUMMARY_JSON)
    frontload_case = pick_frontload_case(stage6_summary, args.frontload_case)
    frontload_stage6_ckpt = Path(str(frontload_case["paths"]["stage6_ckpt"]))
    if not frontload_stage6_ckpt.is_file():
        raise SystemExit(f"missing frontload Stage6 ckpt: {frontload_stage6_ckpt}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    lane_entries: List[Dict[str, Any]] = []
    for idx, lane in enumerate(LANES, start=1):
        log(f"=== [{idx}/{len(LANES)}] lane={lane.name} frontload={frontload_case['name']} ===")
        ensure_lane(frontload_case=frontload_case["name"], frontload_stage6_ckpt=frontload_stage6_ckpt, lane=lane)
        lane_entries.append(build_lane_entry(frontload_case["name"], lane))

    summary = build_summary(frontload_case, lane_entries)
    write_json(OUT_ROOT / "summary.json", summary)
    (OUT_ROOT / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
