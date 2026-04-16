#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
DEFAULT_ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
DEFAULT_AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
GROUP_SUMMARY_TOOL = ROOT / "tools" / "phasea_group_summary.py"


@dataclass
class Candidate:
    name: str
    ckpt: Path
    family: str = ""
    selector: str = ""


@dataclass
class CandidatePaths:
    lane_root: Path
    lane_log: Path
    eval_dir: Path
    eval_json: Path
    group_json: Path


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: Iterable[Any]) -> float:
    finite = [_safe_float(x) for x in values]
    finite = [x for x in finite if math.isfinite(x)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _fmt(value: float, nd: int = 6) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{nd}f}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_candidate(spec: str) -> Candidate:
    parts = [x.strip() for x in str(spec).split("|")]
    if not parts or "=" not in parts[0]:
        raise SystemExit(f"[FATAL] invalid --candidate spec: {spec!r}; expected name=ckpt or name=ckpt|family|selector")
    name, ckpt = [x.strip() for x in parts[0].split("=", 1)]
    family = parts[1] if len(parts) >= 2 else ""
    selector = parts[2] if len(parts) >= 3 else ""
    return Candidate(name=name, ckpt=_resolve(ckpt), family=family, selector=selector)


def _build_paths(*, out_root: Path, name: str) -> CandidatePaths:
    lane_root = out_root / name
    return CandidatePaths(
        lane_root=lane_root,
        lane_log=lane_root / "entry_eval.log",
        eval_dir=lane_root / "entry_freerun",
        eval_json=lane_root / "entry_freerun" / "Walk_F_freerun_cycles.json",
        group_json=lane_root / "basetrain_entry_group_summary.json",
    )


def _run_cmd(cmd: Sequence[str], *, log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(str(x) for x in cmd) + "\n")
        handle.flush()
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
            handle.write(line)
        code = int(proc.wait())
        handle.write(f"[exit_code] {code}\n")
    if code != 0:
        raise SystemExit(code)


def _run_entry_eval(args: argparse.Namespace, cand: Candidate, paths: CandidatePaths) -> None:
    if paths.group_json.is_file() and not bool(args.force_rerun):
        return
    _run_cmd(
        [
            sys.executable,
            "-m",
            "train.validate.run_freerun_cycles",
            "--teacher",
            str(args.teacher),
            "--model",
            str(cand.ckpt),
            "--rounds",
            str(int(args.rounds)),
            "--depth",
            str(int(args.depth)),
            "--time-index-mode",
            str(args.time_index_mode),
            "--phase_reset_source",
            str(args.phase_reset_source),
            "--contacts_meas_source",
            "pretrain_contact",
            "--contacts_meas_pretrain_clamp",
            str(args.pretrain_clamp),
            "--contacts_meas_pretrain_affine_stats",
            str(args.affine_stats),
            "--encoder-bundle",
            str(args.encoder_bundle),
            "--export_joint_direct_geolocal_series",
            "--out",
            str(paths.eval_dir),
            "--force",
        ],
        log_path=paths.lane_log,
    )
    _run_cmd(
        [
            sys.executable,
            str(GROUP_SUMMARY_TOOL),
            str(paths.eval_json),
            "--cycle_gte",
            str(int(args.group_cycle_gte)),
            "--drop_wrap",
            "--out",
            str(paths.group_json),
        ],
        log_path=paths.lane_log,
    )


def _extract_keybone_group_mean(metrics: Mapping[str, Any], key: str) -> float:
    summary = metrics.get("KeyBoneSummary", {})
    if isinstance(summary, Mapping):
        group_mean = summary.get("group_mean", {})
        if isinstance(group_mean, Mapping):
            return _safe_float(group_mean.get(key))
    return float("nan")


def _compute_geo_deg_slope(metrics: Mapping[str, Any]) -> float:
    curve = metrics.get("GeoDegCurve")
    if not isinstance(curve, list) or not curve:
        start = _safe_float(metrics.get("GeoDegStart", metrics.get("GeoDeg")))
        end = _safe_float(metrics.get("GeoDegEnd", start))
        horizon = int(metrics.get("eval_horizon", 0) or 0)
        return (end - start) / max(1, horizon - 1)
    if isinstance(curve[0], (list, tuple)) and curve[0]:
        horizon = len(curve[0])
        mean_curve: List[float] = []
        for step_idx in range(horizon):
            vals = []
            for batch_curve in curve:
                if isinstance(batch_curve, (list, tuple)) and step_idx < len(batch_curve):
                    value = _safe_float(batch_curve[step_idx])
                    if math.isfinite(value):
                        vals.append(value)
            if vals:
                mean_curve.append(float(sum(vals) / len(vals)))
    else:
        mean_curve = [_safe_float(v) for v in curve if math.isfinite(_safe_float(v))]
    if len(mean_curve) < 2:
        return float("inf")
    return float((mean_curve[-1] - mean_curve[0]) / max(1, len(mean_curve) - 1))


def _entry_score(groups: Mapping[str, Any]) -> float:
    all_ex_root = _safe_float(groups.get("all_ex_root", {}).get("mean"))
    leg = _safe_float(groups.get("leg", {}).get("mean"))
    nonleg = _safe_float(groups.get("nonleg", {}).get("mean"))
    return float(all_ex_root + 1.5 * leg + 0.5 * nonleg)


def _build_row(cand: Candidate, paths: CandidatePaths) -> Dict[str, Any]:
    eval_payload = _load_json(paths.eval_json)
    group_payload = _load_json(paths.group_json)
    metrics = eval_payload.get("metrics", {})
    groups = group_payload.get("groups", {})
    row = {
        "name": cand.name,
        "family": cand.family,
        "selector": cand.selector,
        "ckpt": str(cand.ckpt),
        "entry": {
            "all_ex_root_mean": _safe_float(groups.get("all_ex_root", {}).get("mean")),
            "leg_mean": _safe_float(groups.get("leg", {}).get("mean")),
            "nonleg_mean": _safe_float(groups.get("nonleg", {}).get("mean")),
            "arm_mean": _safe_float(groups.get("arm", {}).get("mean")),
            "else_mean": _safe_float(groups.get("else", {}).get("mean")),
            "geo_deg": _safe_float(metrics.get("GeoDeg")),
            "geo_local_deg": _safe_float(metrics.get("GeoLocalDeg")),
            "geo_deg_slope": _compute_geo_deg_slope(metrics),
            "root_vel_mae": _safe_float(metrics.get("RootVelMAE")),
            "ang_vel_mae": _safe_float(metrics.get("AngVelMAE")),
            "keybone_arm_mean": _extract_keybone_group_mean(metrics, "arm"),
            "keybone_trunk_mean": _extract_keybone_group_mean(metrics, "trunk"),
            "keybone_leg_mean": _extract_keybone_group_mean(metrics, "leg"),
        },
        "paths": {
            "eval_json": str(paths.eval_json),
            "group_json": str(paths.group_json),
        },
    }
    row["score"] = {
        "formula": "all_ex_root + 1.5*leg + 0.5*nonleg",
        "value": _entry_score(groups),
    }
    return row


def _render_md(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Basetrain Entry Summary")
    lines.append("")
    lines.append(f"- run_tag: {summary['run_tag']}")
    lines.append(f"- teacher: `{summary['policy']['teacher']}`")
    lines.append(f"- encoder_bundle: `{summary['policy']['encoder_bundle']}`")
    lines.append(f"- contract: `contacts_meas_source=pretrain_contact, clamp={summary['policy']['pretrain_clamp']}, phase_reset_source={summary['policy']['phase_reset_source']}`")
    lines.append(f"- recommended: `{summary['recommended']['name']}`")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| rank | lane | selector | entry score | all_ex_root | leg | nonleg | arm | else | GeoDeg | GeoDegSlope | RootVelMAE |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in summary["ranking"]:
        row = item["row"]
        entry = row["entry"]
        lines.append(
            f"| {item['rank']} | {row['name']} | {row['selector'] or '-'} | {_fmt(_safe_float(row['score']['value']))} | "
            f"{_fmt(_safe_float(entry['all_ex_root_mean']))} | {_fmt(_safe_float(entry['leg_mean']))} | "
            f"{_fmt(_safe_float(entry['nonleg_mean']))} | {_fmt(_safe_float(entry['arm_mean']))} | "
            f"{_fmt(_safe_float(entry['else_mean']))} | {_fmt(_safe_float(entry['geo_deg']))} | "
            f"{_fmt(_safe_float(entry['geo_deg_slope']))} | {_fmt(_safe_float(entry['root_vel_mae']))} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run Stage6-contract basetrain entry eval and summarize direct group metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--candidate", action="append", required=True, help="Repeatable spec: name=ckpt|family|selector")
    ap.add_argument("--run-tag", default="manual")
    ap.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    ap.add_argument("--encoder-bundle", default=str(DEFAULT_ENCODER_BUNDLE))
    ap.add_argument("--affine-stats", default=str(DEFAULT_AFFINE_STATS))
    ap.add_argument("--pretrain-clamp", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--time-index-mode", default="cycle")
    ap.add_argument("--phase-reset-source", default="none")
    ap.add_argument("--group-cycle-gte", type=int, default=1)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()

    args.teacher = _resolve(args.teacher)
    args.encoder_bundle = _resolve(args.encoder_bundle)
    args.affine_stats = _resolve(args.affine_stats)
    out_root = _resolve(args.out_root) if str(args.out_root).strip() else (ROOT / "debug_output" / f"_tmp_basetrain_entry_summary_{args.run_tag}")

    required = [args.teacher, args.encoder_bundle, args.affine_stats, GROUP_SUMMARY_TOOL]
    missing = [str(path) for path in required if not Path(path).is_file()]
    candidates = [_parse_candidate(spec) for spec in args.candidate]
    missing.extend(str(c.ckpt) for c in candidates if not c.ckpt.is_file())
    if missing:
        raise SystemExit("[FATAL] missing required files:\n" + "\n".join(missing))

    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates, start=1):
        print(f"[candidate {idx}/{len(candidates)}] {cand.name} -> {cand.ckpt}", flush=True)
        paths = _build_paths(out_root=out_root, name=cand.name)
        paths.lane_root.mkdir(parents=True, exist_ok=True)
        _run_entry_eval(args, cand, paths)
        rows.append(_build_row(cand, paths))

    rows.sort(
        key=lambda row: (
            _safe_float(row["score"]["value"]),
            _safe_float(row["entry"]["all_ex_root_mean"]),
            _safe_float(row["entry"]["leg_mean"]),
            row["name"],
        )
    )
    ranking = [{"rank": idx + 1, "row": row} for idx, row in enumerate(rows)]
    summary = {
        "run_tag": str(args.run_tag),
        "policy": {
            "teacher": str(args.teacher),
            "encoder_bundle": str(args.encoder_bundle),
            "affine_stats": str(args.affine_stats),
            "pretrain_clamp": float(args.pretrain_clamp),
            "phase_reset_source": str(args.phase_reset_source),
            "time_index_mode": str(args.time_index_mode),
            "score_formula": "all_ex_root + 1.5*leg + 0.5*nonleg",
        },
        "recommended": {"name": ranking[0]["row"]["name"], "score": ranking[0]["row"]["score"]["value"]},
        "ranking": ranking,
        "rows": rows,
    }
    _write_json(out_root / "entry_summary.json", summary)
    (out_root / "entry_summary.md").write_text(_render_md(summary), encoding="utf-8")
    print(f"[done] {out_root / 'entry_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
