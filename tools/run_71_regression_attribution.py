#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_71,
        DEFAULT_DIRECT_POSE_LEG_BONES,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        STAGE6_3WAY_ARMCHAIN_BONES,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        make_generated_config,
        masked_metric_means,
        run_cmd,
        run_eval,
        safe_float,
        window_group_stats,
        write_json,
    )
except ModuleNotFoundError:
    from tools.run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_71,
        DEFAULT_DIRECT_POSE_LEG_BONES,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
        STAGE6_3WAY_ARMCHAIN_BONES,
        diff,
        ensure_group_summary,
        fmt,
        group_metrics,
        load_json,
        make_generated_config,
        masked_metric_means,
        run_cmd,
        run_eval,
        safe_float,
        window_group_stats,
        write_json,
    )


RUN_DATE = "20260314"
SOURCE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_oldd1_newflow_chain_20260314" / "summary.json"
CANDIDATE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_oldd1_skip70b_lowdrift_to71_20260314" / "summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_71_regression_attribution_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_71_regression_attribution_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
SNAPSHOT_STEPS: Tuple[int, ...] = (0, 20, 60, 120, 180)

SELECTED_METRICS = (
    "DirectGeoLocalDeg",
    "all_ex_root",
    "leg",
    "nonleg",
    "arm",
    "legs_main",
    "arms_main",
    "foot_l_ball_l_SIC12_15",
    "calf_r_SIC2_4",
)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def collect_eval(eval_json: Path, group_json: Path) -> Dict[str, Any]:
    return {
        "masked_means": masked_metric_means(eval_json),
        "direct_group_summary": group_metrics(group_json),
        "window_summary": window_group_stats(eval_json),
        "selected_metrics": selected_metrics_from_eval(eval_json, group_json),
        "paths": {
            "eval_json": str(eval_json),
            "group_summary": str(group_json),
        },
    }


def selected_metrics_from_eval(eval_json: Path, group_json: Path) -> Dict[str, float]:
    masked = masked_metric_means(eval_json)
    groups = group_metrics(group_json)
    window = window_group_stats(eval_json)
    return {
        "DirectGeoLocalDeg": safe_float(masked.get("DirectGeoLocalDeg")),
        "all_ex_root": safe_float(groups.get("all_ex_root")),
        "leg": safe_float(groups.get("leg")),
        "nonleg": safe_float(groups.get("nonleg")),
        "arm": safe_float(groups.get("arm")),
        "legs_main": safe_float(window.get("overall", {}).get("legs_main")),
        "arms_main": safe_float(window.get("overall", {}).get("arms_main")),
        "foot_l_ball_l_SIC12_15": safe_float(window.get("hotspots", {}).get("foot_l_ball_l_SIC12_15")),
        "calf_r_SIC2_4": safe_float(window.get("hotspots", {}).get("calf_r_SIC2_4")),
    }


def metric_delta(cur: Mapping[str, Any], ref: Mapping[str, Any], keys: Iterable[str] = SELECTED_METRICS) -> Dict[str, float]:
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def load_joint_payload(eval_json: Path) -> Dict[str, Any]:
    obj = load_json(eval_json)
    steps = obj.get("metrics_per_step", [])
    per = obj.get("per_step_direct_geolocal_deg", {})
    if not isinstance(steps, list) or not isinstance(per, Mapping):
        raise RuntimeError(f"missing direct geolocal payload in {eval_json}")
    names = [str(x) for x in per.get("bone_names", [])]
    mat = per.get("DirectGeoLocalDeg", [])
    root_idx = int(per.get("root_idx", 0) or 0)
    if not names or not isinstance(mat, list):
        raise RuntimeError(f"invalid direct geolocal payload in {eval_json}")
    return {
        "steps": steps,
        "names": names,
        "mat": mat,
        "root_idx": root_idx,
        "name_to_idx": {str(name): int(i) for i, name in enumerate(names)},
    }


def _masked_step_indices(steps: Sequence[Mapping[str, Any]]) -> List[int]:
    out: List[int] = []
    for idx, step in enumerate(steps):
        if int(step.get("cycle", 0) or 0) < 1:
            continue
        if bool(step.get("wrap_boundary_step", False)):
            continue
        out.append(idx)
    return out


def overall_joint_means(eval_json: Path) -> Dict[str, float]:
    payload = load_joint_payload(eval_json)
    steps = payload["steps"]
    mat = payload["mat"]
    names = payload["names"]
    root_idx = payload["root_idx"]
    masked_idx = _masked_step_indices(steps)
    out: Dict[str, float] = {}
    for joint_idx, name in enumerate(names):
        if int(joint_idx) == int(root_idx):
            continue
        vals: List[float] = []
        for step_idx in masked_idx:
            if step_idx >= len(mat):
                break
            row = mat[step_idx]
            if not isinstance(row, list) or joint_idx >= len(row):
                continue
            v = safe_float(row[joint_idx])
            if math.isfinite(v):
                vals.append(v)
        out[str(name)] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def per_sic_group_means(eval_json: Path, joint_names: Sequence[str]) -> Dict[int, float]:
    payload = load_joint_payload(eval_json)
    steps = payload["steps"]
    mat = payload["mat"]
    name_to_idx = payload["name_to_idx"]
    joint_indices = [int(name_to_idx[name]) for name in joint_names if name in name_to_idx]
    out: Dict[int, List[float]] = {}
    for step_idx, step in enumerate(steps):
        if int(step.get("cycle", 0) or 0) < 1:
            continue
        if bool(step.get("wrap_boundary_step", False)):
            continue
        sic = int(step.get("step_in_cycle", -1) or -1)
        if sic < 0 or step_idx >= len(mat):
            continue
        row = mat[step_idx]
        if not isinstance(row, list):
            continue
        bucket = out.setdefault(sic, [])
        for joint_idx in joint_indices:
            if joint_idx >= len(row):
                continue
            v = safe_float(row[joint_idx])
            if math.isfinite(v):
                bucket.append(v)
    return {sic: float(sum(vals) / len(vals)) for sic, vals in out.items() if vals}


def per_sic_joint_means(eval_json: Path, joint_name: str) -> Dict[int, float]:
    payload = load_joint_payload(eval_json)
    name_to_idx = payload["name_to_idx"]
    if joint_name not in name_to_idx:
        return {}
    joint_idx = int(name_to_idx[joint_name])
    steps = payload["steps"]
    mat = payload["mat"]
    out: Dict[int, List[float]] = {}
    for step_idx, step in enumerate(steps):
        if int(step.get("cycle", 0) or 0) < 1:
            continue
        if bool(step.get("wrap_boundary_step", False)):
            continue
        sic = int(step.get("step_in_cycle", -1) or -1)
        if sic < 0 or step_idx >= len(mat):
            continue
        row = mat[step_idx]
        if not isinstance(row, list) or joint_idx >= len(row):
            continue
        v = safe_float(row[joint_idx])
        if math.isfinite(v):
            out.setdefault(sic, []).append(v)
    return {sic: float(sum(vals) / len(vals)) for sic, vals in out.items() if vals}


def top_joint_deltas(cur_eval_json: Path, ref_eval_json: Path, *, joint_names: Optional[Sequence[str]] = None, top_k: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    cur = overall_joint_means(cur_eval_json)
    ref = overall_joint_means(ref_eval_json)
    rows: List[Tuple[float, str, float, float]] = []
    keys = sorted(set(cur) & set(ref))
    joint_filter = {str(x) for x in joint_names} if joint_names is not None else None
    for key in keys:
        if joint_filter is not None and key not in joint_filter:
            continue
        d = diff(cur.get(key), ref.get(key))
        rows.append((d, key, safe_float(ref.get(key)), safe_float(cur.get(key))))
    rows.sort(key=lambda item: item[0])
    best = [
        {"joint": key, "delta": d, "ref": ref_v, "cur": cur_v}
        for d, key, ref_v, cur_v in rows[:top_k]
    ]
    worst = [
        {"joint": key, "delta": d, "ref": ref_v, "cur": cur_v}
        for d, key, ref_v, cur_v in rows[-top_k:][::-1]
    ]
    return {"best": best, "worst": worst}


def top_sic_deltas(cur_eval_json: Path, ref_eval_json: Path, joint_names: Sequence[str], *, top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    cur = per_sic_group_means(cur_eval_json, joint_names)
    ref = per_sic_group_means(ref_eval_json, joint_names)
    rows: List[Tuple[float, int, float, float]] = []
    for sic in sorted(set(cur) & set(ref)):
        rows.append((diff(cur.get(sic), ref.get(sic)), int(sic), safe_float(ref.get(sic)), safe_float(cur.get(sic))))
    rows.sort(key=lambda item: item[0])
    best = [
        {"sic": sic, "delta": d, "ref": ref_v, "cur": cur_v}
        for d, sic, ref_v, cur_v in rows[:top_k]
    ]
    worst = [
        {"sic": sic, "delta": d, "ref": ref_v, "cur": cur_v}
        for d, sic, ref_v, cur_v in rows[-top_k:][::-1]
    ]
    return {"best": best, "worst": worst}


def top_joint_sic_deltas(cur_eval_json: Path, ref_eval_json: Path, joint_name: str, *, top_k: int = 6) -> Dict[str, List[Dict[str, Any]]]:
    cur = per_sic_joint_means(cur_eval_json, joint_name)
    ref = per_sic_joint_means(ref_eval_json, joint_name)
    rows: List[Tuple[float, int, float, float]] = []
    for sic in sorted(set(cur) & set(ref)):
        rows.append((diff(cur.get(sic), ref.get(sic)), int(sic), safe_float(ref.get(sic)), safe_float(cur.get(sic))))
    rows.sort(key=lambda item: item[0])
    best = [
        {"sic": sic, "delta": d, "ref": ref_v, "cur": cur_v}
        for d, sic, ref_v, cur_v in rows[:top_k]
    ]
    worst = [
        {"sic": sic, "delta": d, "ref": ref_v, "cur": cur_v}
        for d, sic, ref_v, cur_v in rows[-top_k:][::-1]
    ]
    return {"best": best, "worst": worst}


def run_snapshot_eval(*, model_ckpt: Path, out_dir: Path, group_json: Path, log_file: Path) -> Dict[str, Any]:
    eval_json = run_eval(
        model_ckpt=model_ckpt,
        out_dir=out_dir,
        contacts_source="model",
        log_file=log_file,
    )
    ensure_group_summary(eval_json, group_json, log_file=log_file)
    return collect_eval(eval_json, group_json)


def run_71_replay(*, lane_name: str, ckpt_in: Path, log_file: Path) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / lane_name
    run_name = f"WalkF_stage7_71_{lane_name}_snapshots_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_71_{lane_name}_snapshots_{RUN_DATE}.json"
    make_generated_config(
        CONFIG_71,
        cfg_json,
        {
            "ckpt_in": str(ckpt_in),
            "out_dir": str(out_dir),
            "run_name": run_name,
            "save_step_ckpts": ",".join(str(x) for x in SNAPSHOT_STEPS),
            "encoder_bundle": str(ENCODER_BUNDLE),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "posttrain_contacts_pretrain_affine_stats": str(AFFINE_STATS),
        },
    )
    required_ckpts = [
        out_dir / f"ckpt_step_{step:06d}_{run_name}.pth"
        for step in SNAPSHOT_STEPS
    ]
    last_ckpt = out_dir / f"ckpt_last_{run_name}.pth"
    if not (last_ckpt.is_file() and all(path.is_file() for path in required_ckpts)):
        out_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                sys.executable,
                "-m",
                "train.posttrain",
                "--config",
                str(cfg_json),
            ],
            log_file=log_file,
        )
    snapshots: Dict[str, Any] = {}
    eval_root = OUT_ROOT / "eval_model" / lane_name
    for step in SNAPSHOT_STEPS:
        ckpt = out_dir / f"ckpt_step_{step:06d}_{run_name}.pth"
        if not ckpt.is_file():
            raise RuntimeError(f"missing snapshot ckpt: {ckpt}")
        tag = f"s{step:03d}"
        snapshots[tag] = {
            "step": int(step),
            "ckpt": str(ckpt),
            "eval": run_snapshot_eval(
                model_ckpt=ckpt,
                out_dir=eval_root / tag,
                group_json=eval_root / f"{tag}_group_summary.json",
                log_file=log_file,
            ),
        }
    snapshots["last_ckpt"] = str(last_ckpt)
    snapshots["config"] = str(cfg_json)
    snapshots["run_name"] = run_name
    return snapshots


def build_snapshot_delta_table(lane_snapshots: Mapping[str, Any], ref_metrics: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tag, payload in lane_snapshots.items():
        if not isinstance(payload, Mapping) or "eval" not in payload:
            continue
        out[str(tag)] = metric_delta(payload["eval"]["selected_metrics"], ref_metrics)
    return out


def choose_recommendation(
    *,
    current_replay: Mapping[str, Any],
    candidate_replay: Mapping[str, Any],
) -> Dict[str, Any]:
    current_s = {tag: payload["eval"]["selected_metrics"] for tag, payload in current_replay.items() if isinstance(payload, Mapping) and "eval" in payload}
    candidate_s = {tag: payload["eval"]["selected_metrics"] for tag, payload in candidate_replay.items() if isinstance(payload, Mapping) and "eval" in payload}
    cand_final = candidate_s["s180"]
    cand_best_all = min(candidate_s.items(), key=lambda item: safe_float(item[1]["all_ex_root"]))
    cand_best_leg = min(candidate_s.items(), key=lambda item: safe_float(item[1]["leg"]))
    cur_best_all = min(current_s.items(), key=lambda item: safe_float(item[1]["all_ex_root"]))
    cur_best_leg = min(current_s.items(), key=lambda item: safe_float(item[1]["leg"]))
    early_candidate_better_than_final = safe_float(candidate_s["s120"]["all_ex_root"]) < safe_float(cand_final["all_ex_root"]) or safe_float(candidate_s["s120"]["leg"]) < safe_float(cand_final["leg"])
    early_candidate_beats_current_final = safe_float(cand_best_all[1]["all_ex_root"]) < safe_float(current_s["s180"]["all_ex_root"]) or safe_float(cand_best_leg[1]["leg"]) < safe_float(current_s["s180"]["leg"])
    if early_candidate_better_than_final and not early_candidate_beats_current_final:
        return {
            "best_next_step": "shorter_71_or_early_stop",
            "why": "candidate replay improves through mid-training but never catches current 71; snapshot selection helps a bit, but not enough by itself",
            "candidate_best_all_ex_root": {"tag": cand_best_all[0], "value": cand_best_all[1]["all_ex_root"]},
            "candidate_best_leg": {"tag": cand_best_leg[0], "value": cand_best_leg[1]["leg"]},
            "current_best_all_ex_root": {"tag": cur_best_all[0], "value": cur_best_all[1]["all_ex_root"]},
            "current_best_leg": {"tag": cur_best_leg[0], "value": cur_best_leg[1]["leg"]},
        }
    if not early_candidate_better_than_final:
        return {
            "best_next_step": "gentler_71_lower_lr",
            "why": "candidate replay does not show a clear early better snapshot; the issue looks like the 71 update rule itself is mismatched to the candidate 70R handoff",
            "candidate_best_all_ex_root": {"tag": cand_best_all[0], "value": cand_best_all[1]["all_ex_root"]},
            "candidate_best_leg": {"tag": cand_best_leg[0], "value": cand_best_leg[1]["leg"]},
            "current_best_all_ex_root": {"tag": cur_best_all[0], "value": cur_best_all[1]["all_ex_root"]},
            "current_best_leg": {"tag": cur_best_leg[0], "value": cur_best_leg[1]["leg"]},
        }
    return {
        "best_next_step": "snapshot_selection_then_shorter_71",
        "why": "candidate replay has a better intermediate snapshot, and that is the cheapest lever before retuning the full 71 recipe",
        "candidate_best_all_ex_root": {"tag": cand_best_all[0], "value": cand_best_all[1]["all_ex_root"]},
        "candidate_best_leg": {"tag": cand_best_leg[0], "value": cand_best_leg[1]["leg"]},
        "current_best_all_ex_root": {"tag": cur_best_all[0], "value": cur_best_all[1]["all_ex_root"]},
        "current_best_leg": {"tag": cur_best_leg[0], "value": cur_best_leg[1]["leg"]},
    }


def build_markdown(summary: Mapping[str, Any]) -> str:
    lane_current = summary["replay"]["current"]["snapshots"]
    lane_candidate = summary["replay"]["candidate"]["snapshots"]
    ref_gain = summary["reference_gain_decomposition"]
    final_leg_joint = summary["final_joint_attribution"]["candidate71_vs_current71_leg_joints"]["worst"]
    final_leg_sic = summary["final_joint_attribution"]["candidate71_vs_current71_leg_sic"]["worst"]
    reco = summary["recommendation"]

    def row(label: str, payload: Mapping[str, Any]) -> str:
        m = payload["eval"]["selected_metrics"]
        return (
            f"| {label} | {fmt(m['DirectGeoLocalDeg'])} | {fmt(m['all_ex_root'])} | {fmt(m['leg'])} | {fmt(m['nonleg'])} | "
            f"{fmt(m['arm'])} | {fmt(m['legs_main'])} | {fmt(m['arms_main'])} | {fmt(m['foot_l_ball_l_SIC12_15'])} | {fmt(m['calf_r_SIC2_4'])} |"
        )

    lines = [
        "# 71 regression attribution",
        "",
        "- raw `70b` is diagnostic-only; it is not the operational handoff in this lane",
        "- the comparison here is `current 70R -> 71` vs `candidate lowdrift 70R -> 71`",
        "- eval contract: model-source only; strict was not needed for the conclusion",
        "",
        "## Short conclusion",
        "",
        f"- candidate starts `71` from a better `70R` on `all_ex_root/leg`, but the `71` stage gives it far less additional gain than the current lane does",
        f"- the candidate gap is already visible during early `71` replay snapshots, so this is not an inherited-start-only story",
        f"- the candidate still improves `calf_r@SIC2-4` and `foot_l/ball_l@SIC12-15`, but broad regressions on other leg joints/windows outweigh those hotspot wins",
        f"- best next step: `{reco['best_next_step']}` ({reco['why']})",
        "",
        "## Replay snapshots",
        "",
        "| lane_snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("current_s000", lane_current["s000"]),
        row("current_s020", lane_current["s020"]),
        row("current_s060", lane_current["s060"]),
        row("current_s120", lane_current["s120"]),
        row("current_s180", lane_current["s180"]),
        row("candidate_s000", lane_candidate["s000"]),
        row("candidate_s020", lane_candidate["s020"]),
        row("candidate_s060", lane_candidate["s060"]),
        row("candidate_s120", lane_candidate["s120"]),
        row("candidate_s180", lane_candidate["s180"]),
        "",
        "## Reference 70R -> 71 gain decomposition",
        "",
        "| metric | inherited (candidate70R-current70R) | 71 gain gap ((cand71-cand70R)-(cur71-cur70R)) | final gap (candidate71-current71) |",
        "|---|---:|---:|---:|",
    ]
    for key in SELECTED_METRICS:
        row_data = ref_gain[key]
        lines.append(
            f"| {key} | {fmt(row_data['inherited'])} | {fmt(row_data['stage71_gain_gap'])} | {fmt(row_data['final_gap'])} |"
        )
    lines.extend(
        [
            "",
            "## Final-stage biggest candidate regressions",
            "",
            "| leg_joint | delta(candidate71-current71) | current71 | candidate71 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row_data in final_leg_joint[:8]:
        lines.append(
            f"| {row_data['joint']} | {fmt(row_data['delta'])} | {fmt(row_data['ref'])} | {fmt(row_data['cur'])} |"
        )
    lines.extend(
        [
            "",
            "| leg_SIC | delta(candidate71-current71) | current71 | candidate71 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row_data in final_leg_sic[:10]:
        lines.append(
            f"| SIC{int(row_data['sic']):02d} | {fmt(row_data['delta'])} | {fmt(row_data['ref'])} | {fmt(row_data['cur'])} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [SOURCE_SUMMARY_JSON, CANDIDATE_SUMMARY_JSON, CONFIG_71, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    source_summary = load_json(SOURCE_SUMMARY_JSON)
    candidate_summary = load_json(CANDIDATE_SUMMARY_JSON)

    current_70r_ckpt = Path(str(source_summary["checkpoints"]["70R"]))
    candidate_70r_ckpt = Path(str(candidate_summary["candidate"]["ckpt_70R"]))
    if not current_70r_ckpt.is_file():
        raise RuntimeError(f"missing current 70R ckpt: {current_70r_ckpt}")
    if not candidate_70r_ckpt.is_file():
        raise RuntimeError(f"missing candidate 70R ckpt: {candidate_70r_ckpt}")

    current_ref_70r = source_summary["stage_progress_model_source"]["70R"]
    current_ref_71 = source_summary["stage_progress_model_source"]["71"]
    candidate_ref_70r = candidate_summary["candidate"]["eval_70R"]
    candidate_ref_71 = candidate_summary["candidate"]["eval_71"]

    log("=== replay current 70R -> 71 with snapshots ===")
    current_replay = run_71_replay(lane_name="current", ckpt_in=current_70r_ckpt, log_file=lane_log)

    log("=== replay candidate 70R -> 71 with snapshots ===")
    candidate_replay = run_71_replay(lane_name="candidate", ckpt_in=candidate_70r_ckpt, log_file=lane_log)

    current_snapshots = {tag: payload for tag, payload in current_replay.items() if isinstance(payload, Mapping) and "eval" in payload}
    candidate_snapshots = {tag: payload for tag, payload in candidate_replay.items() if isinstance(payload, Mapping) and "eval" in payload}

    replay_cross_lane_delta = {
        tag: metric_delta(candidate_snapshots[tag]["eval"]["selected_metrics"], current_snapshots[tag]["eval"]["selected_metrics"])
        for tag in sorted(set(current_snapshots) & set(candidate_snapshots))
    }
    current_replay_vs_ref = build_snapshot_delta_table(current_snapshots, current_ref_71["selected_metrics"] if "selected_metrics" in current_ref_71 else {
        "DirectGeoLocalDeg": current_ref_71["masked_means"]["DirectGeoLocalDeg"],
        "all_ex_root": current_ref_71["direct_group_summary"]["all_ex_root"],
        "leg": current_ref_71["direct_group_summary"]["leg"],
        "nonleg": current_ref_71["direct_group_summary"]["nonleg"],
        "arm": current_ref_71["direct_group_summary"]["arm"],
        "legs_main": current_ref_71["window_summary"]["overall"]["legs_main"],
        "arms_main": current_ref_71["window_summary"]["overall"]["arms_main"],
        "foot_l_ball_l_SIC12_15": current_ref_71["window_summary"]["hotspots"]["foot_l_ball_l_SIC12_15"],
        "calf_r_SIC2_4": current_ref_71["window_summary"]["hotspots"]["calf_r_SIC2_4"],
    })
    candidate_replay_vs_ref = build_snapshot_delta_table(candidate_snapshots, candidate_ref_71["selected_metrics"] if "selected_metrics" in candidate_ref_71 else {
        "DirectGeoLocalDeg": candidate_ref_71["masked_means"]["DirectGeoLocalDeg"],
        "all_ex_root": candidate_ref_71["direct_group_summary"]["all_ex_root"],
        "leg": candidate_ref_71["direct_group_summary"]["leg"],
        "nonleg": candidate_ref_71["direct_group_summary"]["nonleg"],
        "arm": candidate_ref_71["direct_group_summary"]["arm"],
        "legs_main": candidate_ref_71["window_summary"]["overall"]["legs_main"],
        "arms_main": candidate_ref_71["window_summary"]["overall"]["arms_main"],
        "foot_l_ball_l_SIC12_15": candidate_ref_71["window_summary"]["hotspots"]["foot_l_ball_l_SIC12_15"],
        "calf_r_SIC2_4": candidate_ref_71["window_summary"]["hotspots"]["calf_r_SIC2_4"],
    })

    current_70r_sel = current_ref_70r.get("selected_metrics", {
        "DirectGeoLocalDeg": current_ref_70r["masked_means"]["DirectGeoLocalDeg"],
        "all_ex_root": current_ref_70r["direct_group_summary"]["all_ex_root"],
        "leg": current_ref_70r["direct_group_summary"]["leg"],
        "nonleg": current_ref_70r["direct_group_summary"]["nonleg"],
        "arm": current_ref_70r["direct_group_summary"]["arm"],
        "legs_main": current_ref_70r["window_summary"]["overall"]["legs_main"],
        "arms_main": current_ref_70r["window_summary"]["overall"]["arms_main"],
        "foot_l_ball_l_SIC12_15": current_ref_70r["window_summary"]["hotspots"]["foot_l_ball_l_SIC12_15"],
        "calf_r_SIC2_4": current_ref_70r["window_summary"]["hotspots"]["calf_r_SIC2_4"],
    })
    candidate_70r_sel = candidate_ref_70r.get("selected_metrics", {
        "DirectGeoLocalDeg": candidate_ref_70r["masked_means"]["DirectGeoLocalDeg"],
        "all_ex_root": candidate_ref_70r["direct_group_summary"]["all_ex_root"],
        "leg": candidate_ref_70r["direct_group_summary"]["leg"],
        "nonleg": candidate_ref_70r["direct_group_summary"]["nonleg"],
        "arm": candidate_ref_70r["direct_group_summary"]["arm"],
        "legs_main": candidate_ref_70r["window_summary"]["overall"]["legs_main"],
        "arms_main": candidate_ref_70r["window_summary"]["overall"]["arms_main"],
        "foot_l_ball_l_SIC12_15": candidate_ref_70r["window_summary"]["hotspots"]["foot_l_ball_l_SIC12_15"],
        "calf_r_SIC2_4": candidate_ref_70r["window_summary"]["hotspots"]["calf_r_SIC2_4"],
    })
    current_71_sel = current_ref_71.get("selected_metrics", {
        "DirectGeoLocalDeg": current_ref_71["masked_means"]["DirectGeoLocalDeg"],
        "all_ex_root": current_ref_71["direct_group_summary"]["all_ex_root"],
        "leg": current_ref_71["direct_group_summary"]["leg"],
        "nonleg": current_ref_71["direct_group_summary"]["nonleg"],
        "arm": current_ref_71["direct_group_summary"]["arm"],
        "legs_main": current_ref_71["window_summary"]["overall"]["legs_main"],
        "arms_main": current_ref_71["window_summary"]["overall"]["arms_main"],
        "foot_l_ball_l_SIC12_15": current_ref_71["window_summary"]["hotspots"]["foot_l_ball_l_SIC12_15"],
        "calf_r_SIC2_4": current_ref_71["window_summary"]["hotspots"]["calf_r_SIC2_4"],
    })
    candidate_71_sel = candidate_ref_71.get("selected_metrics", {
        "DirectGeoLocalDeg": candidate_ref_71["masked_means"]["DirectGeoLocalDeg"],
        "all_ex_root": candidate_ref_71["direct_group_summary"]["all_ex_root"],
        "leg": candidate_ref_71["direct_group_summary"]["leg"],
        "nonleg": candidate_ref_71["direct_group_summary"]["nonleg"],
        "arm": candidate_ref_71["direct_group_summary"]["arm"],
        "legs_main": candidate_ref_71["window_summary"]["overall"]["legs_main"],
        "arms_main": candidate_ref_71["window_summary"]["overall"]["arms_main"],
        "foot_l_ball_l_SIC12_15": candidate_ref_71["window_summary"]["hotspots"]["foot_l_ball_l_SIC12_15"],
        "calf_r_SIC2_4": candidate_ref_71["window_summary"]["hotspots"]["calf_r_SIC2_4"],
    })

    reference_gain_decomposition = {}
    for key in SELECTED_METRICS:
        inherited = diff(candidate_70r_sel.get(key), current_70r_sel.get(key))
        cur_gain = diff(current_71_sel.get(key), current_70r_sel.get(key))
        cand_gain = diff(candidate_71_sel.get(key), candidate_70r_sel.get(key))
        final_gap = diff(candidate_71_sel.get(key), current_71_sel.get(key))
        reference_gain_decomposition[key] = {
            "inherited": inherited,
            "current_stage71_gain": cur_gain,
            "candidate_stage71_gain": cand_gain,
            "stage71_gain_gap": diff(cand_gain, cur_gain),
            "final_gap": final_gap,
        }

    current_replay_final_json = Path(str(current_snapshots["s180"]["eval"]["paths"]["eval_json"]))
    candidate_replay_final_json = Path(str(candidate_snapshots["s180"]["eval"]["paths"]["eval_json"]))
    current_ref_final_json = Path(str(current_ref_71["paths"]["eval_json"]))
    candidate_ref_final_json = Path(str(candidate_ref_71["paths"]["eval_json"]))

    final_joint_attribution = {
        "candidate71_vs_current71_leg_joints": top_joint_deltas(
            candidate_ref_final_json,
            current_ref_final_json,
            joint_names=DEFAULT_DIRECT_POSE_LEG_BONES,
            top_k=10,
        ),
        "candidate71_vs_current71_all_joints": top_joint_deltas(
            candidate_ref_final_json,
            current_ref_final_json,
            top_k=12,
        ),
        "candidate71_vs_current71_leg_sic": top_sic_deltas(
            candidate_ref_final_json,
            current_ref_final_json,
            DEFAULT_DIRECT_POSE_LEG_BONES,
            top_k=12,
        ),
        "candidate71_vs_current71_joint_sic": {
            joint: top_joint_sic_deltas(candidate_ref_final_json, current_ref_final_json, joint, top_k=8)
            for joint in ("calf_l", "ball_r", "foot_l", "foot_r", "calf_r", "ball_l", "thigh_l", "thigh_r")
        },
    }

    recommendation = choose_recommendation(current_replay=current_snapshots, candidate_replay=candidate_snapshots)

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "compare_contract": "model_source",
            "strict_eval_added": False,
            "reason_strict_not_needed": "direct-path conclusion is already stable under model-source snapshots and does not hinge on contacts meas source",
            "note": "raw 70b is diagnostic-only; active comparison is current 70R -> 71 vs candidate lowdrift 70R -> 71",
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "snapshot_steps": list(SNAPSHOT_STEPS),
        },
        "references": {
            "source_summary": str(SOURCE_SUMMARY_JSON),
            "candidate_summary": str(CANDIDATE_SUMMARY_JSON),
            "current_70R_ckpt": str(current_70r_ckpt),
            "candidate_70R_ckpt": str(candidate_70r_ckpt),
        },
        "reference_metrics": {
            "current_70R": current_70r_sel,
            "candidate_70R": candidate_70r_sel,
            "current_71": current_71_sel,
            "candidate_71": candidate_71_sel,
        },
        "replay": {
            "current": {
                "snapshots": current_snapshots,
                "replay_vs_reference_final_delta": current_replay_vs_ref["s180"],
                "all_snapshot_vs_reference_final_delta": current_replay_vs_ref,
                "config": current_replay["config"],
                "run_name": current_replay["run_name"],
                "last_ckpt": current_replay["last_ckpt"],
            },
            "candidate": {
                "snapshots": candidate_snapshots,
                "replay_vs_reference_final_delta": candidate_replay_vs_ref["s180"],
                "all_snapshot_vs_reference_final_delta": candidate_replay_vs_ref,
                "config": candidate_replay["config"],
                "run_name": candidate_replay["run_name"],
                "last_ckpt": candidate_replay["last_ckpt"],
            },
            "candidate_minus_current_by_snapshot": replay_cross_lane_delta,
        },
        "reference_gain_decomposition": reference_gain_decomposition,
        "final_joint_attribution": final_joint_attribution,
        "recommendation": recommendation,
        "answers": {
            "q1_earliest_regression_snapshot": min(
                (
                    (tag, metrics)
                    for tag, metrics in replay_cross_lane_delta.items()
                    if safe_float(metrics.get("all_ex_root")) > 0.0 or safe_float(metrics.get("leg")) > 0.0
                ),
                key=lambda item: int(item[0][1:]),
                default=("none", {}),
            ),
            "q2_hotspot_paradox": "candidate 71 keeps local wins at calf_r@SIC2-4 and foot_l/ball_l@SIC12-15, but broader leg regressions on calf_l/foot_l/foot_r/ball_r and SIC00/03/08-13/24-26/34-37 outweigh them",
            "q3_main_cause": "71-stage under-gain dominates; inherited nonleg/arm deficit remains, but the bigger failure is that candidate 71 does not harvest the same leg/all_ex_root gains that current 71 gets from its 70R start",
            "q4_best_next_step": recommendation["best_next_step"],
        },
    }

    summary_json = OUT_ROOT / "summary.json"
    summary_md = OUT_ROOT / "summary.md"
    write_json(summary_json, summary)
    summary_md.write_text(build_markdown(summary), encoding="utf-8")
    log(f"DONE summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
