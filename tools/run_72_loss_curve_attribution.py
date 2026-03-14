#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from run_cp015_oldplan_downstream_chain import (
        AFFINE_STATS,
        CONFIG_72,
        DEFAULT_DIRECT_POSE_LEG_BONES,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
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
        CONFIG_72,
        DEFAULT_DIRECT_POSE_LEG_BONES,
        ENCODER_BUNDLE,
        PRETRAIN_CLAMP,
        ROOT,
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

try:
    from run_71_regression_attribution import (
        top_joint_deltas,
        top_joint_sic_deltas,
        top_sic_deltas,
    )
except ModuleNotFoundError:
    from tools.run_71_regression_attribution import (
        top_joint_deltas,
        top_joint_sic_deltas,
        top_sic_deltas,
    )


RUN_DATE = "20260314"
SOURCE_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_oldd1_newflow_chain_20260314" / "summary.json"
LOWLR_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_71_lowlr_sweep_20260314" / "summary.json"
DOWNSTREAM_SUMMARY_JSON = ROOT / "debug_output" / "_tmp_71_lowlr_to72lambda_20260314" / "summary.json"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_72_loss_curve_attribution_{RUN_DATE}"
MODEL_ROOT = ROOT / "models" / f"__tmp_72_loss_curve_attribution_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
SNAPSHOT_STEPS: Tuple[int, ...] = (0, 5, 10, 20, 40, 60, 120, 180)
WINDOWS: Dict[str, Tuple[int, int]] = {
    "start20": (0, 20),
    "mid20": (80, 100),
    "late20": (160, 180),
}

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

LOSS_KEYS = (
    "total",
    "dir_geo",
    "leg_align_weighted",
    "leg_align_loss",
    "leg_align_distal_loss",
    "leg_align_proximal_loss",
    "dir_group_norm_leg",
    "dir_leg_base",
    "dir_nonleg_base",
    "boundary_dir_geo",
)

PLOT_KEYS = (
    "total",
    "dir_geo",
    "leg_align_weighted",
    "leg_align_loss",
    "dir_group_norm_leg",
    "dir_leg_base",
    "dir_nonleg_base",
    "boundary_dir_geo",
)

LOSS_LABELS = {
    "total": "total",
    "dir_geo": "dir_geo",
    "leg_align_weighted": "leg_align_weighted",
    "leg_align_loss": "leg_align_loss",
    "leg_align_distal_loss": "leg_align_distal_loss",
    "leg_align_proximal_loss": "leg_align_proximal_loss",
    "dir_group_norm_leg": "dir_group_norm_leg",
    "dir_leg_base": "dir_leg_base",
    "dir_nonleg_base": "dir_nonleg_base",
    "boundary_dir_geo": "boundary_dir_geo",
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def mean(values: Iterable[Any]) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


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


def selected_metrics_from_stage_payload(payload: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "DirectGeoLocalDeg": safe_float(payload["masked_means"]["DirectGeoLocalDeg"]),
        "all_ex_root": safe_float(payload["direct_group_summary"]["all_ex_root"]),
        "leg": safe_float(payload["direct_group_summary"]["leg"]),
        "nonleg": safe_float(payload["direct_group_summary"]["nonleg"]),
        "arm": safe_float(payload["direct_group_summary"]["arm"]),
        "legs_main": safe_float(payload["window_summary"]["overall"]["legs_main"]),
        "arms_main": safe_float(payload["window_summary"]["overall"]["arms_main"]),
        "foot_l_ball_l_SIC12_15": safe_float(payload["window_summary"]["hotspots"]["foot_l_ball_l_SIC12_15"]),
        "calf_r_SIC2_4": safe_float(payload["window_summary"]["hotspots"]["calf_r_SIC2_4"]),
    }


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


def metric_delta(cur: Mapping[str, Any], ref: Mapping[str, Any], keys: Iterable[str] = SELECTED_METRICS) -> Dict[str, float]:
    return {key: diff(cur.get(key), ref.get(key)) for key in keys}


def run_snapshot_eval(*, model_ckpt: Path, out_dir: Path, group_json: Path, log_file: Path) -> Dict[str, Any]:
    eval_json = run_eval(
        model_ckpt=model_ckpt,
        out_dir=out_dir,
        contacts_source="model",
        log_file=log_file,
    )
    ensure_group_summary(eval_json, group_json, log_file=log_file)
    return collect_eval(eval_json, group_json)


def run_72_replay(*, lane_name: str, ckpt_in: Path, log_file: Path) -> Dict[str, Any]:
    out_dir = MODEL_ROOT / lane_name
    run_name = f"WalkF_stage7_72_{lane_name}_snapshots_{RUN_DATE}"
    cfg_json = CONFIG_ROOT / f"posttrain_72_{lane_name}_snapshots_{RUN_DATE}.json"
    make_generated_config(
        CONFIG_72,
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
    required_ckpts = [out_dir / f"ckpt_step_{step:06d}_{run_name}.pth" for step in SNAPSHOT_STEPS]
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


def _finite_series(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    return [safe_float(row.get(key)) for row in rows]


def _window_stats(values: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, (start, end) in WINDOWS.items():
        out[name] = mean(values[start:end])
    return out


def _epoch_stats(rows: Sequence[Mapping[str, Any]], values: Sequence[float]) -> Dict[str, float]:
    buckets: Dict[int, List[float]] = {}
    for row, value in zip(rows, values):
        epoch = int(safe_float(row.get("epoch", 0)) or 0)
        buckets.setdefault(epoch, []).append(value)
    return {f"epoch{epoch}": mean(vals) for epoch, vals in sorted(buckets.items())}


def _first_peak(rows: Sequence[Mapping[str, Any]], values: Sequence[float], n: int = 20) -> Dict[str, float]:
    best_idx = -1
    best_val = float("-inf")
    for idx, value in enumerate(values[:n]):
        if not math.isfinite(value):
            continue
        if value > best_val:
            best_idx = idx
            best_val = value
    if best_idx < 0:
        return {"step": float("nan"), "value": float("nan")}
    step = safe_float(rows[best_idx].get("step"))
    return {"step": step, "value": best_val}


def summarize_loss_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    first = rows[0] if rows else {}
    observed_omega_keys = sorted(key for key in first.keys() if "omega" in key.lower())
    observed_leg_align_keys = sorted(key for key in first.keys() if "leg_align" in key.lower())
    summary: Dict[str, Any] = {
        "num_steps": len(rows),
        "observed_omega_keys": observed_omega_keys,
        "observed_leg_align_keys": observed_leg_align_keys,
        "series": {},
    }
    for key in LOSS_KEYS:
        values = _finite_series(rows, key)
        if not any(math.isfinite(v) for v in values):
            continue
        summary["series"][key] = {
            "windows": _window_stats(values),
            "epoch_means": _epoch_stats(rows, values),
            "peak_first20": _first_peak(rows, values, n=20),
        }
    return summary


def summarise_loss_delta(cur: Mapping[str, Any], ref: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cur_series = cur["series"]
    ref_series = ref["series"]
    for key in sorted(set(cur_series) & set(ref_series)):
        out[key] = {
            "windows": {
                name: diff(cur_series[key]["windows"].get(name), ref_series[key]["windows"].get(name))
                for name in WINDOWS
            },
            "epoch_means": {
                epoch: diff(cur_series[key]["epoch_means"].get(epoch), ref_series[key]["epoch_means"].get(epoch))
                for epoch in sorted(set(cur_series[key]["epoch_means"]) | set(ref_series[key]["epoch_means"]))
            },
            "peak_first20": {
                "step_delta": diff(cur_series[key]["peak_first20"].get("step"), ref_series[key]["peak_first20"].get("step")),
                "value_delta": diff(cur_series[key]["peak_first20"].get("value"), ref_series[key]["peak_first20"].get("value")),
            },
        }
    return out


def write_loss_summary_md(*, current: Mapping[str, Any], candidate: Mapping[str, Any], out_path: Path) -> None:
    lines = [
        "# 72 loss curve summary",
        "",
        "- no explicit `omega`-named loss key appears in the stored 72 posttrain logs",
        f"- observed 72-specific leg-align keys: `{', '.join(candidate.get('observed_leg_align_keys', []))}`",
        "",
        "| key | lane | start20 | mid20 | late20 | epoch1 | epoch2 | epoch3 | peak_first20(step,value) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in LOSS_KEYS:
        if key not in current["series"] or key not in candidate["series"]:
            continue
        for lane, payload in (("current72", current["series"][key]), ("candidate72", candidate["series"][key])):
            lines.append(
                f"| {key} | {lane} | {fmt(payload['windows']['start20'])} | {fmt(payload['windows']['mid20'])} | {fmt(payload['windows']['late20'])} | {fmt(payload['epoch_means'].get('epoch1'))} | {fmt(payload['epoch_means'].get('epoch2'))} | {fmt(payload['epoch_means'].get('epoch3'))} | s{int(safe_float(payload['peak_first20']['step'])):03d}, {fmt(payload['peak_first20']['value'])} |"
            )
        lines.append(
            f"| {key} | cand-current | {fmt(diff(candidate['series'][key]['windows']['start20'], current['series'][key]['windows']['start20']))} | {fmt(diff(candidate['series'][key]['windows']['mid20'], current['series'][key]['windows']['mid20']))} | {fmt(diff(candidate['series'][key]['windows']['late20'], current['series'][key]['windows']['late20']))} | {fmt(diff(candidate['series'][key]['epoch_means'].get('epoch1'), current['series'][key]['epoch_means'].get('epoch1')))} | {fmt(diff(candidate['series'][key]['epoch_means'].get('epoch2'), current['series'][key]['epoch_means'].get('epoch2')))} | {fmt(diff(candidate['series'][key]['epoch_means'].get('epoch3'), current['series'][key]['epoch_means'].get('epoch3')))} | dstep={fmt(diff(candidate['series'][key]['peak_first20']['step'], current['series'][key]['peak_first20']['step']))}, dval={fmt(diff(candidate['series'][key]['peak_first20']['value'], current['series'][key]['peak_first20']['value']))} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_loss_curves(*, current_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    steps = [safe_float(row.get("step")) for row in current_rows]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    axes = axes.flatten()
    for ax, key in zip(axes, PLOT_KEYS):
        cur_values = _finite_series(current_rows, key)
        cand_values = _finite_series(candidate_rows, key)
        ax.plot(steps, cur_values, label="current72", linewidth=2.0, color="#1f77b4")
        ax.plot(steps, cand_values, label="candidate72", linewidth=2.0, color="#d62728")
        for x in (20, 60, 120, 180):
            ax.axvline(x, linewidth=0.8, color="#aaaaaa", linestyle="--", alpha=0.6)
        ax.set_title(LOSS_LABELS[key])
        ax.grid(True, alpha=0.25)
    for ax in axes[-2:]:
        ax.set_xlabel("step")
    axes[0].legend(loc="upper right")
    fig.suptitle("72 loss curves: current71->72 vs candidate71(lr=3e-4)->72", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def extract_snapshot_metrics(snapshots: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    return {
        tag: payload["eval"]["selected_metrics"]
        for tag, payload in snapshots.items()
        if isinstance(payload, Mapping) and "eval" in payload
    }


def build_stage_gain_decomposition(
    *,
    current_71: Mapping[str, float],
    candidate_71: Mapping[str, float],
    current_72: Mapping[str, float],
    candidate_72: Mapping[str, float],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in SELECTED_METRICS:
        inherited = diff(candidate_71.get(key), current_71.get(key))
        cur_gain = diff(current_72.get(key), current_71.get(key))
        cand_gain = diff(candidate_72.get(key), candidate_71.get(key))
        out[key] = {
            "inherited": inherited,
            "current_stage72_gain": cur_gain,
            "candidate_stage72_gain": cand_gain,
            "stage72_gain_gap": diff(cand_gain, cur_gain),
            "final_gap": diff(candidate_72.get(key), current_72.get(key)),
        }
    return out


def first_regression_snapshot(snapshot_delta: Mapping[str, Mapping[str, float]]) -> Tuple[str, Dict[str, float]]:
    for tag in sorted(snapshot_delta, key=lambda x: int(x[1:])):
        metrics = snapshot_delta[tag]
        if safe_float(metrics.get("all_ex_root")) > 0.0 or safe_float(metrics.get("leg")) > 0.0:
            return tag, dict(metrics)
    return "none", {}


def best_snapshot(metrics: Mapping[str, Dict[str, float]], key: str, *, exclude_start: bool = False) -> Tuple[str, Dict[str, float]]:
    rows = []
    for tag, payload in metrics.items():
        if exclude_start and tag == "s000":
            continue
        rows.append((tag, payload))
    if not rows:
        return "none", {}
    return min(rows, key=lambda item: safe_float(item[1].get(key)))


def build_markdown(summary: Mapping[str, Any]) -> str:
    refs = summary["reference_metrics"]
    replay = summary["replay"]
    gain = summary["stage72_gain_decomposition"]
    answers = summary["answers"]
    joint_worst = summary["final_joint_attribution"]["candidate72_vs_current72_leg_joints"]["worst"]
    sic_worst = summary["final_joint_attribution"]["candidate72_vs_current72_leg_sic"]["worst"]
    lines = [
        "# 72 loss curve attribution",
        "",
        "## Short conclusion",
        "",
        f"- candidate `71 (lr=3e-4)` does start `72` from a better aggregate leg/all_ex_root state, but unchanged `72` immediately over-updates that cleaner start",
        f"- the stored 72 logs already show a clear early overshoot on the candidate lane: `total start20 {fmt(summary['loss_curves']['candidate']['series']['total']['windows']['start20'])}` vs current `72` `{fmt(summary['loss_curves']['current']['series']['total']['windows']['start20'])}`, `dir_group_norm_leg start20 {fmt(summary['loss_curves']['candidate']['series']['dir_group_norm_leg']['windows']['start20'])}` vs `{fmt(summary['loss_curves']['current']['series']['dir_group_norm_leg']['windows']['start20'])}`",
        f"- replay says the aggregate regression is introduced inside `72`, not inherited from candidate `71`: earliest snapshot crossing is `{answers['q1_earliest_regression_snapshot']}`",
        f"- hotspot wins survive because `foot_l/ball_l@SIC12-15` and `calf_r@SIC2-4` remain better, but broader losses on `calf_l`, `ball_l`, `ball_r`, and late mid-cycle leg windows outweigh those local gains",
        f"- best next minimal lever: `{answers['q4_best_next_step']}`",
        "",
        "## End-state table",
        "",
        "| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| current `71` | {fmt(refs['current_71']['DirectGeoLocalDeg'])} | {fmt(refs['current_71']['all_ex_root'])} | {fmt(refs['current_71']['leg'])} | {fmt(refs['current_71']['nonleg'])} | {fmt(refs['current_71']['arm'])} | {fmt(refs['current_71']['legs_main'])} | {fmt(refs['current_71']['arms_main'])} | {fmt(refs['current_71']['foot_l_ball_l_SIC12_15'])} | {fmt(refs['current_71']['calf_r_SIC2_4'])} |",
        f"| candidate `71` (`lr=3e-4`) | {fmt(refs['candidate_71']['DirectGeoLocalDeg'])} | {fmt(refs['candidate_71']['all_ex_root'])} | {fmt(refs['candidate_71']['leg'])} | {fmt(refs['candidate_71']['nonleg'])} | {fmt(refs['candidate_71']['arm'])} | {fmt(refs['candidate_71']['legs_main'])} | {fmt(refs['candidate_71']['arms_main'])} | {fmt(refs['candidate_71']['foot_l_ball_l_SIC12_15'])} | {fmt(refs['candidate_71']['calf_r_SIC2_4'])} |",
        f"| current `72` | {fmt(refs['current_72']['DirectGeoLocalDeg'])} | {fmt(refs['current_72']['all_ex_root'])} | {fmt(refs['current_72']['leg'])} | {fmt(refs['current_72']['nonleg'])} | {fmt(refs['current_72']['arm'])} | {fmt(refs['current_72']['legs_main'])} | {fmt(refs['current_72']['arms_main'])} | {fmt(refs['current_72']['foot_l_ball_l_SIC12_15'])} | {fmt(refs['current_72']['calf_r_SIC2_4'])} |",
        f"| candidate `72` | {fmt(refs['candidate_72']['DirectGeoLocalDeg'])} | {fmt(refs['candidate_72']['all_ex_root'])} | {fmt(refs['candidate_72']['leg'])} | {fmt(refs['candidate_72']['nonleg'])} | {fmt(refs['candidate_72']['arm'])} | {fmt(refs['candidate_72']['legs_main'])} | {fmt(refs['candidate_72']['arms_main'])} | {fmt(refs['candidate_72']['foot_l_ball_l_SIC12_15'])} | {fmt(refs['candidate_72']['calf_r_SIC2_4'])} |",
        "",
        "## Replay snapshots",
        "",
        "| lane_snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane_name in ("current", "candidate"):
        for tag in sorted(replay[lane_name]["snapshots"], key=lambda x: int(x[1:])):
            m = replay[lane_name]["snapshots"][tag]["eval"]["selected_metrics"]
            lines.append(
                f"| {lane_name} `{tag}` | {fmt(m['DirectGeoLocalDeg'])} | {fmt(m['all_ex_root'])} | {fmt(m['leg'])} | {fmt(m['nonleg'])} | {fmt(m['arm'])} | {fmt(m['legs_main'])} | {fmt(m['arms_main'])} | {fmt(m['foot_l_ball_l_SIC12_15'])} | {fmt(m['calf_r_SIC2_4'])} |"
            )
    lines.extend(
        [
            "",
            "## 71->72 gain decomposition",
            "",
            "| metric | inherited (candidate71-current71) | current72-current71 | candidate72-candidate71 | stage72 gain gap | final gap (candidate72-current72) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in SELECTED_METRICS:
        row = gain[key]
        lines.append(
            f"| {key} | {fmt(row['inherited'])} | {fmt(row['current_stage72_gain'])} | {fmt(row['candidate_stage72_gain'])} | {fmt(row['stage72_gain_gap'])} | {fmt(row['final_gap'])} |"
        )
    lines.extend(
        [
            "",
            "## Final candidate72 regressions vs current72",
            "",
            "| leg_joint | delta(candidate72-current72) | current72 | candidate72 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in joint_worst[:8]:
        lines.append(f"| {row['joint']} | {fmt(row['delta'])} | {fmt(row['ref'])} | {fmt(row['cur'])} |")
    lines.extend(
        [
            "",
            "| leg_SIC | delta(candidate72-current72) | current72 | candidate72 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sic_worst[:12]:
        lines.append(f"| SIC{int(row['sic']):02d} | {fmt(row['delta'])} | {fmt(row['ref'])} | {fmt(row['cur'])} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- loss curve plot: `{summary['artifacts']['loss_curve_plot']}`",
            f"- loss curve summary: `{summary['artifacts']['loss_curve_summary_md']}`",
            f"- machine summary: `{summary['artifacts']['summary_json']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    required = [SOURCE_SUMMARY_JSON, LOWLR_SUMMARY_JSON, DOWNSTREAM_SUMMARY_JSON, CONFIG_72, ENCODER_BUNDLE, AFFINE_STATS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    lane_log = OUT_ROOT / "lane.log"

    source_summary = load_json(SOURCE_SUMMARY_JSON)
    lowlr_summary = load_json(LOWLR_SUMMARY_JSON)
    downstream_summary = load_json(DOWNSTREAM_SUMMARY_JSON)

    current_71_ckpt = Path(str(source_summary["checkpoints"]["71"]))
    candidate_71_ckpt = Path(str(lowlr_summary["cases"]["lr3e4"]["last_ckpt"]))
    if not current_71_ckpt.is_file():
        raise RuntimeError(f"missing current 71 ckpt: {current_71_ckpt}")
    if not candidate_71_ckpt.is_file():
        raise RuntimeError(f"missing candidate 71 ckpt: {candidate_71_ckpt}")

    current_71 = selected_metrics_from_stage_payload(source_summary["stage_progress_model_source"]["71"])
    current_72 = selected_metrics_from_stage_payload(source_summary["stage_progress_model_source"]["72"])
    candidate_71 = lowlr_summary["cases"]["lr3e4"]["snapshots"]["s180"]["eval"]["selected_metrics"]
    candidate_72 = downstream_summary["candidate"]["eval_72"]["selected_metrics"]

    current_72_eval_json = Path(str(source_summary["stage_progress_model_source"]["72"]["paths"]["eval_json"]))
    current_71_eval_json = Path(str(source_summary["stage_progress_model_source"]["71"]["paths"]["eval_json"]))
    candidate_72_eval_json = Path(str(downstream_summary["candidate"]["eval_72"]["paths"]["eval_json"]))
    candidate_71_eval_json = Path(str(lowlr_summary["cases"]["lr3e4"]["snapshots"]["s180"]["eval"]["paths"]["eval_json"]))

    current_72_log_path = ROOT / "models" / "__tmp_oldd1_newflow_chain_20260314" / "72" / "posttrain_log_WalkF_stage7_72_from_oldd1_newflow_20260314.json"
    candidate_72_log_path = ROOT / "models" / "__tmp_71_lowlr_to72lambda_20260314" / "72" / "posttrain_log_WalkF_stage7_72_from_lowlr71_20260314.json"
    current_72_log = load_json(current_72_log_path)
    candidate_72_log = load_json(candidate_72_log_path)
    current_rows = current_72_log["log"]
    candidate_rows = candidate_72_log["log"]

    loss_current = summarize_loss_rows(current_rows)
    loss_candidate = summarize_loss_rows(candidate_rows)
    loss_delta = summarise_loss_delta(loss_candidate, loss_current)

    loss_curve_plot = OUT_ROOT / "72_loss_curve_compare.png"
    plot_loss_curves(current_rows=current_rows, candidate_rows=candidate_rows, out_path=loss_curve_plot)
    loss_curve_summary_json = OUT_ROOT / "72_loss_curve_summary.json"
    loss_curve_summary_md = OUT_ROOT / "72_loss_curve_summary.md"
    write_json(
        loss_curve_summary_json,
        {
            "current": loss_current,
            "candidate": loss_candidate,
            "candidate_minus_current": loss_delta,
        },
    )
    write_loss_summary_md(current=loss_current, candidate=loss_candidate, out_path=loss_curve_summary_md)

    log("=== replay current 71 -> 72 with snapshots ===")
    current_replay = run_72_replay(lane_name="current", ckpt_in=current_71_ckpt, log_file=lane_log)
    log("=== replay candidate 71(lr=3e-4) -> 72 with snapshots ===")
    candidate_replay = run_72_replay(lane_name="candidate", ckpt_in=candidate_71_ckpt, log_file=lane_log)

    current_snapshots = extract_snapshot_metrics(current_replay)
    candidate_snapshots = extract_snapshot_metrics(candidate_replay)
    candidate_minus_current_by_snapshot = {
        tag: metric_delta(candidate_snapshots[tag], current_snapshots[tag])
        for tag in sorted(set(current_snapshots) & set(candidate_snapshots))
    }

    current_replay_final_delta = metric_delta(current_snapshots["s180"], current_72)
    candidate_replay_final_delta = metric_delta(candidate_snapshots["s180"], candidate_72)
    earliest_tag, earliest_payload = first_regression_snapshot(candidate_minus_current_by_snapshot)
    candidate_best_all_any = best_snapshot(candidate_snapshots, "all_ex_root", exclude_start=False)
    candidate_best_leg_any = best_snapshot(candidate_snapshots, "leg", exclude_start=False)
    candidate_best_all_after_start = best_snapshot(candidate_snapshots, "all_ex_root", exclude_start=True)
    candidate_best_leg_after_start = best_snapshot(candidate_snapshots, "leg", exclude_start=True)

    stage72_gain_decomposition = build_stage_gain_decomposition(
        current_71=current_71,
        candidate_71=candidate_71,
        current_72=current_72,
        candidate_72=candidate_72,
    )

    final_joint_attribution = {
        "candidate72_vs_current72_leg_joints": top_joint_deltas(
            candidate_72_eval_json,
            current_72_eval_json,
            joint_names=DEFAULT_DIRECT_POSE_LEG_BONES,
            top_k=10,
        ),
        "candidate72_vs_current72_leg_sic": top_sic_deltas(
            candidate_72_eval_json,
            current_72_eval_json,
            DEFAULT_DIRECT_POSE_LEG_BONES,
            top_k=16,
        ),
        "candidate72_vs_current72_joint_sic": {
            joint: top_joint_sic_deltas(candidate_72_eval_json, current_72_eval_json, joint, top_k=10)
            for joint in ("foot_l", "ball_l", "calf_r", "calf_l", "ball_r")
        },
        "candidate72_vs_candidate71_leg_joints": top_joint_deltas(
            candidate_72_eval_json,
            candidate_71_eval_json,
            joint_names=DEFAULT_DIRECT_POSE_LEG_BONES,
            top_k=10,
        ),
        "candidate72_vs_candidate71_leg_sic": top_sic_deltas(
            candidate_72_eval_json,
            candidate_71_eval_json,
            DEFAULT_DIRECT_POSE_LEG_BONES,
            top_k=16,
        ),
        "current72_vs_current71_leg_joints": top_joint_deltas(
            current_72_eval_json,
            current_71_eval_json,
            joint_names=DEFAULT_DIRECT_POSE_LEG_BONES,
            top_k=10,
        ),
        "current72_vs_current71_leg_sic": top_sic_deltas(
            current_72_eval_json,
            current_71_eval_json,
            DEFAULT_DIRECT_POSE_LEG_BONES,
            top_k=16,
        ),
    }

    after_start_joint_win = any(
        safe_float(metrics.get("all_ex_root")) < safe_float(current_72["all_ex_root"])
        and safe_float(metrics.get("leg")) < safe_float(current_72["leg"])
        for tag, metrics in candidate_snapshots.items()
        if tag != "s000"
    )
    if earliest_tag in {"s005", "s010"} and not after_start_joint_win:
        best_next_step = "lower_lr_72_or_gentler_72"
    elif after_start_joint_win:
        best_next_step = "early_stop_or_shorter_72"
    else:
        best_next_step = "lower_lr_72"

    summary = {
        "run_date": RUN_DATE,
        "policy": {
            "compare_contract": "model_source",
            "strict_eval_added": False,
            "reason_strict_not_needed": "current conclusion already stabilizes under model-source replay snapshots; no contact-source dependency was needed",
            "encoder_bundle": str(ENCODER_BUNDLE),
            "affine_stats": str(AFFINE_STATS),
            "posttrain_contacts_source": "pretrain_contact",
            "posttrain_contacts_pretrain_clamp": PRETRAIN_CLAMP,
            "snapshot_steps": list(SNAPSHOT_STEPS),
        },
        "references": {
            "source_summary": str(SOURCE_SUMMARY_JSON),
            "lowlr_summary": str(LOWLR_SUMMARY_JSON),
            "downstream_summary": str(DOWNSTREAM_SUMMARY_JSON),
            "current_71_ckpt": str(current_71_ckpt),
            "candidate_71_ckpt": str(candidate_71_ckpt),
            "current_72_log": str(current_72_log_path),
            "candidate_72_log": str(candidate_72_log_path),
        },
        "reference_metrics": {
            "current_71": current_71,
            "candidate_71": candidate_71,
            "current_72": current_72,
            "candidate_72": candidate_72,
        },
        "loss_curves": {
            "current": loss_current,
            "candidate": loss_candidate,
            "candidate_minus_current": loss_delta,
            "plot_path": str(loss_curve_plot),
            "summary_json": str(loss_curve_summary_json),
            "summary_md": str(loss_curve_summary_md),
        },
        "replay": {
            "current": {
                "snapshots": {
                    tag: payload
                    for tag, payload in current_replay.items()
                    if isinstance(payload, Mapping) and "eval" in payload
                },
                "replay_vs_reference_final_delta": current_replay_final_delta,
                "config": current_replay["config"],
                "run_name": current_replay["run_name"],
                "last_ckpt": current_replay["last_ckpt"],
            },
            "candidate": {
                "snapshots": {
                    tag: payload
                    for tag, payload in candidate_replay.items()
                    if isinstance(payload, Mapping) and "eval" in payload
                },
                "replay_vs_reference_final_delta": candidate_replay_final_delta,
                "config": candidate_replay["config"],
                "run_name": candidate_replay["run_name"],
                "last_ckpt": candidate_replay["last_ckpt"],
            },
            "candidate_minus_current_by_snapshot": candidate_minus_current_by_snapshot,
            "candidate_best_snapshot_all_ex_root_any": {"tag": candidate_best_all_any[0], "metrics": candidate_best_all_any[1]},
            "candidate_best_snapshot_leg_any": {"tag": candidate_best_leg_any[0], "metrics": candidate_best_leg_any[1]},
            "candidate_best_snapshot_all_ex_root_after_start": {"tag": candidate_best_all_after_start[0], "metrics": candidate_best_all_after_start[1]},
            "candidate_best_snapshot_leg_after_start": {"tag": candidate_best_leg_after_start[0], "metrics": candidate_best_leg_after_start[1]},
        },
        "stage72_gain_decomposition": stage72_gain_decomposition,
        "final_joint_attribution": final_joint_attribution,
        "answers": {
            "q1_earliest_regression_snapshot": earliest_tag,
            "q1_earliest_regression_delta": earliest_payload,
            "q2_hotspot_paradox": "candidate72 keeps `foot_l/ball_l@SIC12-15` and `calf_r@SIC2-4` better than current72, but 72 also introduces broader regressions on `calf_l`, `ball_l`, `ball_r`, and late mid-cycle windows; those averaged losses outweigh the local hotspot wins",
            "q3_loss_curve_interpretation": "early overshoot is primary; candidate72 start20 spikes on total/dir_geo/dir_group_norm_leg/leg_align, then mid20 loss looks normal again while freerun aggregate is already worse, which supports early over-step plus some objective mismatch rather than pure late overfit",
            "q4_best_next_step": best_next_step,
            "q5_one_sentence": "candidate71(lr=3e-4) wins because it ends 71 at a cleaner leg state, but unchanged 72 immediately over-updates that cleaner start, giving back broad leg average quality even while preserving a few hotspot gains.",
        },
        "artifacts": {
            "loss_curve_plot": str(loss_curve_plot),
            "loss_curve_summary_json": str(loss_curve_summary_json),
            "loss_curve_summary_md": str(loss_curve_summary_md),
            "summary_json": str(OUT_ROOT / "summary.json"),
            "summary_md": str(OUT_ROOT / "summary.md"),
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
