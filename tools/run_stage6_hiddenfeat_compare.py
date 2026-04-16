#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUN_DATE = "20260313"
OUT_ROOT = ROOT / "debug_output" / f"_tmp_stage6_hiddenfeat_compare_{RUN_DATE}"
CONFIG_ROOT = OUT_ROOT / "configs"
RESULTS_ROOT = OUT_ROOT / "results"
MODEL_ROOT = ROOT / "models" / f"__tmp_stage6_hiddenfeat_compare_{RUN_DATE}"
STAGE6_MODEL_ROOT = ROOT / "models" / f"__tmp_stage6_hiddenfeat_stage6_{RUN_DATE}"
RUN_LOG = OUT_ROOT / "runner.log"
GRADPROBE_JSON = ROOT / "debug_output" / f"_tmp_stage6_hiddenfeat_gradprobe_{RUN_DATE}" / "hiddenfeat_backbone_grad_probe.json"
BASELINE_JSON = ROOT / "debug_output" / "_tmp_stage6_basetrain_compare_20260313" / "compare_summary.json"
DPOFF_JSON = ROOT / "debug_output" / "_tmp_stage6_directposeoff_compare_20260313" / "directposeoff_vs_baseline_summary.json"
STAGE6_CONFIG = ROOT / "config" / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json"


@dataclass(frozen=True)
class LaneSpec:
    lane_name: str
    family: str
    variant: str
    source_config: Path
    run_name: str
    detach_feat: bool


LANES: Sequence[LaneSpec] = (
    LaneSpec(
        lane_name="old_hidden_gradon",
        family="old",
        variant="hidden+gradon",
        source_config=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1" / "config_resolved.json",
        run_name="exp_phase_DirectBranch_v1_d1_hidden_gradon_20260313",
        detach_feat=False,
    ),
    LaneSpec(
        lane_name="old_hidden_gradoff",
        family="old",
        variant="hidden+gradoff",
        source_config=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1" / "config_resolved.json",
        run_name="exp_phase_DirectBranch_v1_d1_hidden_gradoff_20260313",
        detach_feat=True,
    ),
    LaneSpec(
        lane_name="cp015_hidden_gradon",
        family="cp015",
        variant="hidden+gradon",
        source_config=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3" / "config_resolved.json",
        run_name="exp_phase_DirectBranch_v1_d1_cp015_tailk3_hidden_gradon_20260313",
        detach_feat=False,
    ),
    LaneSpec(
        lane_name="cp015_hidden_gradoff",
        family="cp015",
        variant="hidden+gradoff",
        source_config=ROOT / "models" / "MLPL2_DirectBranch_v1" / "exp_phase_DirectBranch_v1_d1_cp015_tailk3" / "config_resolved.json",
        run_name="exp_phase_DirectBranch_v1_d1_cp015_tailk3_hidden_gradoff_20260313",
        detach_feat=True,
    ),
)


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _fmt(v: Any, nd: int = 6) -> str:
    x = _safe_float(v)
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _abs_path(path_like: Any) -> Path:
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _mean_key(block: Dict[str, Any], prefix: str) -> float:
    return _safe_float(block.get(f"{prefix}_mean"))


def _run_cmd(cmd: Sequence[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(str(x) for x in cmd) + "\n")
        f.flush()
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
            f.write(line)
        code = int(proc.wait())
        f.write(f"[exit_code] {code}\n")
    if code != 0:
        raise SystemExit(code)


def _build_config(spec: LaneSpec) -> Path:
    cfg = _load_json(spec.source_config)
    cfg.pop("_trainbase_contacts_source_resolved", None)
    cfg["out"] = str(MODEL_ROOT)
    cfg["run_name"] = spec.run_name
    cfg["resume"] = None
    cfg["direct_pose_enable"] = True
    cfg["direct_pose_detach_plan"] = True
    cfg["direct_pose_feat_source"] = "hidden"
    cfg["direct_pose_detach_feat"] = bool(spec.detach_feat)
    path = CONFIG_ROOT / f"{spec.run_name}.json"
    _write_json(path, cfg)
    return path


def _expected_last_ckpt(spec: LaneSpec) -> Path:
    return MODEL_ROOT / spec.run_name / f"ckpt_last_{spec.run_name}.pth"


def _expected_stage6_ckpt(spec: LaneSpec) -> str | None:
    stage6_dir = RESULTS_ROOT / spec.lane_name
    init_stats_path = stage6_dir / "posttrain_stage6_init_stats.json"
    if init_stats_path.is_file():
        try:
            init_stats = _load_json(init_stats_path)
            src = str(init_stats.get("source", "") or "")
            if src:
                stem = Path(src).stem
                if stem.startswith("posttrain_log_"):
                    run_name = stem[len("posttrain_log_") :]
                    ckpts = sorted((STAGE6_MODEL_ROOT / spec.lane_name).glob(f"ckpt_last_{run_name}.pth"))
                    if ckpts:
                        return str(ckpts[0])
        except Exception:
            pass
    ckpts = sorted((STAGE6_MODEL_ROOT / spec.lane_name).glob("ckpt_last_*.pth"))
    if ckpts:
        return str(ckpts[0])
    return None


def _ensure_gradient_probe_ready() -> Dict[str, Any]:
    if not GRADPROBE_JSON.is_file():
        raise SystemExit(f"[FATAL] missing gradient probe result: {GRADPROBE_JSON}")
    obj = _load_json(GRADPROBE_JSON)
    if not bool(obj.get("overall_pass", False)):
        raise SystemExit(f"[FATAL] gradient probe did not pass: {GRADPROBE_JSON}")
    return obj


def _build_entry_from_lane(row: Dict[str, Any], *, family: str, variant: str, source: str) -> Dict[str, Any]:
    return {
        "name": str(row["name"]),
        "family": family,
        "variant": variant,
        "source": source,
        "ckpt": row.get("ckpt"),
        "basetrain": {
            "all_ex_root": _mean_key(row.get("basetrain", {}), "all_ex_root"),
            "leg": _mean_key(row.get("basetrain", {}), "leg"),
            "nonleg": _mean_key(row.get("basetrain", {}), "nonleg"),
        },
        "stage6_exit": {
            "all_ex_root": _mean_key(row.get("stage6_exit", {}), "all_ex_root"),
            "leg": _mean_key(row.get("stage6_exit", {}), "leg"),
            "nonleg": _mean_key(row.get("stage6_exit", {}), "nonleg"),
        },
        "paths": row.get("paths", {}),
    }


def _selector_row_to_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    name = str(row["name"])
    family = "cp015" if name.startswith("cp015_") else "old"
    variant = "hidden+gradoff" if name.endswith("gradoff") else "hidden+gradon"
    return _build_entry_from_lane(row, family=family, variant=variant, source="hiddenfeat")


def _baseline_row_to_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    name = str(row["name"])
    family = "cp015" if name.startswith("cp015_") else "old"
    return _build_entry_from_lane(row, family=family, variant="baseline(cond)", source="baseline")


def _dpoff_row_to_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    name = str(row["name"])
    family = "cp015" if name.startswith("cp015_") else "old"
    return _build_entry_from_lane(row, family=family, variant="direct_pose=false", source="directposeoff")


def _group_summary_to_metrics(path: Path) -> Dict[str, float]:
    obj = _load_json(path)
    groups = obj.get("groups", {})
    return {
        "all_ex_root": _safe_float(groups.get("all_ex_root", {}).get("mean")),
        "leg": _safe_float(groups.get("leg", {}).get("mean")),
        "nonleg": _safe_float(groups.get("nonleg", {}).get("mean")),
    }


def _latest_metric_epoch(metrics_dir: Path, tag: str) -> int | None:
    prefix = f"{tag}_ep"
    epochs: List[int] = []
    for path in metrics_dir.glob(f"{prefix}*.json"):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        try:
            epochs.append(int(stem[len(prefix) :]))
        except Exception:
            continue
    return max(epochs) if epochs else None


def _resolve_ckpt_train_epoch(run_root: Path, ckpt_path: Path) -> tuple[str, int | None]:
    selector = "unknown"
    epoch: int | None = None
    summary_path = run_root / "basetrain_keybone_group_summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    ckpt_name = ckpt_path.name
    if ckpt_name.startswith("ckpt_last_"):
        selector = "last"
    metrics_dir = run_root / "metrics"
    if (epoch is None) or epoch <= 0:
        epoch = _latest_metric_epoch(metrics_dir, "train")
    if epoch is not None and epoch <= 0:
        epoch = None
    return selector, epoch


def _load_ckpt_contact_plan_stats(ckpt_like: Any) -> Dict[str, Any]:
    payload = {
        "selector": "unknown",
        "epoch": None,
        "run_root": None,
        "train_metrics_path": None,
        "contact_plan_bce": float("nan"),
        "contact_plan_mse": float("nan"),
        "contact_plan_weighted": float("nan"),
    }
    if not ckpt_like:
        return payload
    ckpt_path = _abs_path(ckpt_like)
    payload["run_root"] = str(ckpt_path.parent)
    if not ckpt_path.is_file():
        return payload
    selector, epoch = _resolve_ckpt_train_epoch(ckpt_path.parent, ckpt_path)
    payload["selector"] = selector
    payload["epoch"] = epoch
    if epoch is None:
        return payload
    train_metrics_path = ckpt_path.parent / "metrics" / f"train_ep{int(epoch):03d}.json"
    payload["train_metrics_path"] = str(train_metrics_path)
    if not train_metrics_path.is_file():
        return payload
    metrics = _load_json(train_metrics_path).get("metrics", {})
    if not isinstance(metrics, dict):
        return payload
    payload["contact_plan_bce"] = _safe_float(metrics.get("contact_plan_bce"))
    payload["contact_plan_mse"] = _safe_float(metrics.get("contact_plan_mse"))
    payload["contact_plan_weighted"] = _safe_float(metrics.get("contact_plan_weighted"))
    return payload


def _stage6_delta_from_init(basetrain: Dict[str, float], stage6_exit: Dict[str, float]) -> Dict[str, float]:
    return {
        key: _delta(stage6_exit.get(key, float("nan")), basetrain.get(key, float("nan")))
        for key in ("all_ex_root", "leg", "nonleg")
    }


def _load_hidden_lane_entry(spec: LaneSpec) -> Dict[str, Any]:
    lane_root = RESULTS_ROOT / spec.lane_name
    basetrain_path = lane_root / "basetrain_group_summary.json"
    stage6_path = lane_root / "stage6_group_summary.json"
    init_stats_path = lane_root / "posttrain_stage6_init_stats.json"
    lane_log_path = lane_root / "lane.log"
    required = [basetrain_path, stage6_path, init_stats_path, lane_log_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "[FATAL] missing per-lane Stage6 artifacts for hidden-feature summary: "
            + ", ".join(missing)
        )
    paths: Dict[str, Any] = {
        "lane_log": str(lane_log_path),
        "basetrain_group_summary": str(basetrain_path),
        "stage6_init_stats": str(init_stats_path),
        "stage6_group_summary": str(stage6_path),
    }
    stage6_ckpt = _expected_stage6_ckpt(spec)
    if stage6_ckpt:
        paths["stage6_ckpt"] = stage6_ckpt
    return {
        "name": spec.lane_name,
        "family": spec.family,
        "variant": spec.variant,
        "source": "hiddenfeat",
        "ckpt": str(_expected_last_ckpt(spec)),
        "basetrain": _group_summary_to_metrics(basetrain_path),
        "stage6_exit": _group_summary_to_metrics(stage6_path),
        "paths": paths,
    }


def _write_hidden_lane_selector_artifacts(rows: Sequence[Dict[str, Any]]) -> Path:
    selector_json = RESULTS_ROOT / "selector_summary.json"
    ranking = sorted(
        [
            {
                "name": row["name"],
                "family": row["family"],
                "variant": row["variant"],
                "stage6_exit": row["stage6_exit"],
                "basetrain": row["basetrain"],
            }
            for row in rows
        ],
        key=lambda row: _sort_key(row, "stage6_exit"),
    )
    recommended = ranking[0]["name"] if ranking else None
    payload = {
        "run_tag": f"{RUN_DATE}_hiddenfeat_per_lane",
        "policy": {
            "source": "per_lane_stage6_outputs",
            "stage6_config": str(STAGE6_CONFIG),
            "reinit": True,
            "note": "Aggregated directly from finished per-lane outputs; no combined selector rerun.",
        },
        "recommended": recommended,
        "ranking": ranking,
        "rows": rows,
    }
    _write_json(selector_json, payload)
    md_lines = [
        "# Hidden-feature per-lane Stage6 aggregation",
        "",
        "- source: finished per-lane outputs only",
        f"- stage6 config: `{STAGE6_CONFIG}`",
        f"- recommended by Stage6 all_ex_root: `{recommended}`" if recommended else "- recommended by Stage6 all_ex_root: `n/a`",
        "",
        "| lane | family | variant | stage6 all_ex_root | leg | nonleg |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in ranking:
        tgt = row["stage6_exit"]
        md_lines.append(
            f"| {row['name']} | {row['family']} | {row['variant']} | "
            f"{_fmt(tgt['all_ex_root'])} | {_fmt(tgt['leg'])} | {_fmt(tgt['nonleg'])} |"
        )
    md_lines.append("")
    (RESULTS_ROOT / "selector_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    return selector_json


def _sort_key(block: Dict[str, Any], metric: str) -> tuple[float, float, float, str]:
    target = block[metric]
    return (
        _safe_float(target["all_ex_root"]),
        _safe_float(target["leg"]),
        _safe_float(target["nonleg"]),
        str(block["name"]),
    )


def _delta(cur: float, ref: float) -> float:
    if not math.isfinite(cur) or not math.isfinite(ref):
        return float("nan")
    return float(cur - ref)


def _verdict_from_deltas(deltas: Iterable[float]) -> str:
    vals = [float(v) for v in deltas if math.isfinite(v)]
    if not vals:
        return "unknown"
    if all(v < 0.0 for v in vals):
        return "better"
    if all(v > 0.0 for v in vals):
        return "worse"
    return "mixed"


def _render_table(entries: Sequence[Dict[str, Any]], metric: str) -> List[str]:
    lines = [
        f"## {'Basetrain last endpoint' if metric == 'basetrain' else 'Stage6 exit'}",
        "",
        "| family | lane | variant | all_ex_root | leg | nonleg | delta_vs_baseline all_ex_root |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in entries:
        tgt = row[metric]
        baseline_ref = row["delta_vs_baseline"][metric]
        lines.append(
            f"| {row['family']} | {row['name']} | {row['variant']} | "
            f"{_fmt(tgt['all_ex_root'])} | {_fmt(tgt['leg'])} | {_fmt(tgt['nonleg'])} | "
            f"{_fmt(baseline_ref['all_ex_root'])} |"
        )
    lines.append("")
    return lines


def _render_contact_plan_table(entries: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Basetrain checkpoint contact-plan stats",
        "",
        "| family | lane | variant | ckpt selector | train epoch | contact_plan_bce | contact_plan_mse |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in entries:
        stats = row.get("contact_plan_at_ckpt", {})
        epoch = stats.get("epoch")
        epoch_str = str(int(epoch)) if isinstance(epoch, int) or (isinstance(epoch, float) and math.isfinite(epoch)) else "n/a"
        lines.append(
            f"| {row['family']} | {row['name']} | {row['variant']} | "
            f"{stats.get('selector', 'unknown')} | {epoch_str} | "
            f"{_fmt(stats.get('contact_plan_bce'))} | {_fmt(stats.get('contact_plan_mse'))} |"
        )
    lines.append("")
    return lines


def _render_stage6_delta_table(entries: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "## Stage6 delta from init",
        "",
        "- definition: `stage6_exit - basetrain`, negative is better",
        "",
        "| family | lane | variant | delta all_ex_root | delta leg | delta nonleg |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in entries:
        delta = row.get("stage6_delta_from_init", {})
        lines.append(
            f"| {row['family']} | {row['name']} | {row['variant']} | "
            f"{_fmt(delta.get('all_ex_root'))} | {_fmt(delta.get('leg'))} | {_fmt(delta.get('nonleg'))} |"
        )
    lines.append("")
    return lines


def _variant_rank(row: Dict[str, Any]) -> int:
    order = {
        "baseline(cond)": 0,
        "direct_pose=false": 1,
        "hidden+gradon": 2,
        "hidden+gradoff": 3,
    }
    return int(order.get(str(row.get("variant")), 99))


def _render_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Hidden-feature direct branch vs baseline")
    lines.append("")
    lines.append(f"- grad probe: `passed` (`{payload['paths']['grad_probe_json']}`)")
    lines.append(f"- no-op check: `{payload['answers']['no_op']}`")
    lines.append(f"- stage6 config: `{payload['paths']['stage6_config']}`")
    lines.append("- stage6 reinit: `true`")
    lines.append("- hidden path fallback used: `no` (`direct_pose_feat_source=\"hidden\"` for all four new lanes)")
    lines.append(f"- hidden-lane aggregation json: `{payload['paths']['selector_summary_json']}`")
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    lines.append(f"1. hidden-feature direct branch连到backbone后，basetrain verdict=`{payload['answers']['q1_basetrain']}`，Stage6 verdict=`{payload['answers']['q1_stage6']}`。")
    lines.append(f"2. `direct_pose_detach_feat=true` 对 Stage6 handoff 的结论=`{payload['answers']['q2_stage6_detach_feat']}`。")
    for family in ("old", "cp015"):
        fam = payload["family_verdicts"][family]
        lines.append(
            f"3. `{family}`: Stage6 最好的是 `{fam['best_stage6']['name']}` ({fam['best_stage6']['variant']}); "
            f"basetrain 最好的是 `{fam['best_basetrain']['name']}` ({fam['best_basetrain']['variant']})."
        )
    lines.append("")
    lines.append("## Gradient probe")
    lines.append("")
    lines.append("| lane | detach_feat | direct_pose_head grad | shared_encoder grad | contact_plan grad |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in payload["gradient_probe"]["results"]:
        lines.append(
            f"| {row['name']} | {str(bool(row['detach_feat'])).lower()} | "
            f"{_fmt(row['direct_pose_head'])} | {_fmt(row['shared_encoder'])} | {_fmt(row['contact_plan'])} |"
        )
    lines.append("")
    lines.extend(_render_table(payload["table_rows"], "basetrain"))
    lines.extend(_render_contact_plan_table(payload["table_rows"]))
    lines.extend(_render_table(payload["table_rows"], "stage6_exit"))
    lines.extend(_render_stage6_delta_table(payload["table_rows"]))
    lines.append("## Family deltas")
    lines.append("")
    for family in ("old", "cp015"):
        fam = payload["family_verdicts"][family]
        lines.append(
            f"- `{family}` hidden+gradon vs baseline: basetrain Δall={_fmt(fam['gradon_vs_baseline']['basetrain']['all_ex_root'])}, "
            f"Stage6 Δall={_fmt(fam['gradon_vs_baseline']['stage6_exit']['all_ex_root'])}"
        )
        lines.append(
            f"- `{family}` hidden+gradoff vs baseline: basetrain Δall={_fmt(fam['gradoff_vs_baseline']['basetrain']['all_ex_root'])}, "
            f"Stage6 Δall={_fmt(fam['gradoff_vs_baseline']['stage6_exit']['all_ex_root'])}"
        )
        lines.append(
            f"- `{family}` hidden+gradoff vs hidden+gradon: basetrain Δall={_fmt(fam['gradoff_vs_gradon']['basetrain']['all_ex_root'])}, "
            f"Stage6 Δall={_fmt(fam['gradoff_vs_gradon']['stage6_exit']['all_ex_root'])}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    STAGE6_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    gradprobe = _ensure_gradient_probe_ready()

    config_rows = []
    for spec in LANES:
        cfg_path = _build_config(spec)
        ckpt_path = _expected_last_ckpt(spec)
        config_rows.append(
            {
                "lane_name": spec.lane_name,
                "family": spec.family,
                "variant": spec.variant,
                "source_config": str(spec.source_config),
                "config": str(cfg_path),
                "run_name": spec.run_name,
                "expected_last_ckpt": str(ckpt_path),
            }
        )
        if ckpt_path.is_file():
            print(f"[skip] basetrain exists: {ckpt_path}")
            continue
        _run_cmd([sys.executable, "-m", "train.training_MPL", "--config_json", str(cfg_path)])
        if not ckpt_path.is_file():
            raise SystemExit(f"[FATAL] missing last ckpt after training: {ckpt_path}")

    manifest = {
        "run_date": RUN_DATE,
        "out_root": str(OUT_ROOT),
        "model_root": str(MODEL_ROOT),
        "stage6_model_root": str(STAGE6_MODEL_ROOT),
        "gradprobe_json": str(GRADPROBE_JSON),
        "lanes": config_rows,
    }
    _write_json(OUT_ROOT / "manifest.json", manifest)

    hidden_rows = [_load_hidden_lane_entry(spec) for spec in LANES]
    selector_json = _write_hidden_lane_selector_artifacts(hidden_rows)

    baseline_obj = _load_json(BASELINE_JSON)
    dpoff_obj = _load_json(DPOFF_JSON)

    baseline_rows = [_baseline_row_to_entry(row) for row in baseline_obj.get("lanes", []) if row.get("name") in ("old_bestfree", "cp015_bestfree")]
    dpoff_rows = [_dpoff_row_to_entry(row) for row in dpoff_obj.get("lanes", []) if row.get("name") in ("old_dpoff_bestfree", "cp015_dpoff_bestfree")]
    all_rows = baseline_rows + dpoff_rows + hidden_rows
    for row in all_rows:
        row["contact_plan_at_ckpt"] = _load_ckpt_contact_plan_stats(row.get("ckpt"))
        row["stage6_delta_from_init"] = _stage6_delta_from_init(row["basetrain"], row["stage6_exit"])

    by_name = {row["name"]: row for row in all_rows}
    table_rows: List[Dict[str, Any]] = []
    family_verdicts: Dict[str, Any] = {}
    q1_basetrain_deltas: List[float] = []
    q1_stage6_deltas: List[float] = []
    q2_stage6_deltas: List[float] = []

    for family in ("old", "cp015"):
        if family == "old":
            names = ["old_bestfree", "old_dpoff_bestfree", "old_hidden_gradon", "old_hidden_gradoff"]
        else:
            names = ["cp015_bestfree", "cp015_dpoff_bestfree", "cp015_hidden_gradon", "cp015_hidden_gradoff"]
        family_rows = [by_name[name] for name in names]
        baseline = family_rows[0]
        gradon = family_rows[2]
        gradoff = family_rows[3]
        for row in family_rows:
            row = dict(row)
            row["delta_vs_baseline"] = {
                "basetrain": {
                    k: _delta(row["basetrain"][k], baseline["basetrain"][k])
                    for k in ("all_ex_root", "leg", "nonleg")
                },
                "stage6_exit": {
                    k: _delta(row["stage6_exit"][k], baseline["stage6_exit"][k])
                    for k in ("all_ex_root", "leg", "nonleg")
                },
            }
            table_rows.append(row)
        best_stage6 = min(family_rows, key=lambda row: _sort_key(row, "stage6_exit"))
        best_basetrain = min(family_rows, key=lambda row: _sort_key(row, "basetrain"))
        family_verdicts[family] = {
            "best_stage6": {"name": best_stage6["name"], "variant": best_stage6["variant"]},
            "best_basetrain": {"name": best_basetrain["name"], "variant": best_basetrain["variant"]},
            "gradon_vs_baseline": {
                "basetrain": {k: _delta(gradon["basetrain"][k], baseline["basetrain"][k]) for k in ("all_ex_root", "leg", "nonleg")},
                "stage6_exit": {k: _delta(gradon["stage6_exit"][k], baseline["stage6_exit"][k]) for k in ("all_ex_root", "leg", "nonleg")},
            },
            "gradoff_vs_baseline": {
                "basetrain": {k: _delta(gradoff["basetrain"][k], baseline["basetrain"][k]) for k in ("all_ex_root", "leg", "nonleg")},
                "stage6_exit": {k: _delta(gradoff["stage6_exit"][k], baseline["stage6_exit"][k]) for k in ("all_ex_root", "leg", "nonleg")},
            },
            "gradoff_vs_gradon": {
                "basetrain": {k: _delta(gradoff["basetrain"][k], gradon["basetrain"][k]) for k in ("all_ex_root", "leg", "nonleg")},
                "stage6_exit": {k: _delta(gradoff["stage6_exit"][k], gradon["stage6_exit"][k]) for k in ("all_ex_root", "leg", "nonleg")},
            },
        }
        q1_basetrain_deltas.append(family_verdicts[family]["gradon_vs_baseline"]["basetrain"]["all_ex_root"])
        q1_stage6_deltas.append(family_verdicts[family]["gradon_vs_baseline"]["stage6_exit"]["all_ex_root"])
        q2_stage6_deltas.append(family_verdicts[family]["gradoff_vs_gradon"]["stage6_exit"]["all_ex_root"])

    table_rows.sort(key=lambda row: (row["family"], _variant_rank(row), str(row["name"])))
    payload = {
        "run_date": RUN_DATE,
        "paths": {
            "grad_probe_json": str(GRADPROBE_JSON),
            "baseline_json": str(BASELINE_JSON),
            "directposeoff_json": str(DPOFF_JSON),
            "selector_summary_json": str(selector_json),
            "stage6_config": str(STAGE6_CONFIG),
        },
        "gradient_probe": {
            "overall_pass": bool(gradprobe.get("overall_pass", False)),
            "path": str(GRADPROBE_JSON),
            "results": [
                {
                    "name": str(row.get("name")),
                    "family": str(row.get("family")),
                    "detach_feat": bool(row.get("detach_feat", False)),
                    "direct_pose_head": _safe_float(row.get("grad_norms", {}).get("direct_pose_head")),
                    "shared_encoder": _safe_float(row.get("grad_norms", {}).get("shared_encoder")),
                    "contact_plan": _safe_float(row.get("grad_norms", {}).get("contact_plan")),
                    "pass": bool(row.get("pass", False)),
                }
                for row in gradprobe.get("results", [])
            ],
        },
        "answers": {
            "no_op": "no" if bool(gradprobe.get("overall_pass", False)) else "unknown",
            "q1_basetrain": _verdict_from_deltas(q1_basetrain_deltas),
            "q1_stage6": _verdict_from_deltas(q1_stage6_deltas),
            "q2_stage6_detach_feat": _verdict_from_deltas(q2_stage6_deltas),
        },
        "family_verdicts": family_verdicts,
        "table_rows": table_rows,
    }
    _write_json(OUT_ROOT / "hiddenfeat_vs_baseline_summary.json", payload)
    (OUT_ROOT / "hiddenfeat_vs_baseline_summary.md").write_text(_render_md(payload), encoding="utf-8")
    print(f"[done] {OUT_ROOT / 'hiddenfeat_vs_baseline_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
