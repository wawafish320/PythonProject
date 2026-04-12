#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _fmt(value: Any, nd: int = 6) -> str:
    value = _safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{nd}f}"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metric_path(metrics_dir: Path, tag: str, epoch: int) -> Path:
    return metrics_dir / f"{tag}_ep{int(epoch):03d}.json"


def _load_metric(metrics_dir: Path, tag: str, epoch: int) -> Dict[str, Any] | None:
    path = _metric_path(metrics_dir, tag, epoch)
    if not path.is_file():
        return None
    payload = _load_json(path)
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, Mapping) else None


def _extract_group_mean(metrics: Mapping[str, Any], name: str) -> float:
    summary = metrics.get("KeyBoneSummary", {})
    if isinstance(summary, Mapping):
        group_mean = summary.get("group_mean", {})
        if isinstance(group_mean, Mapping):
            return _safe_float(group_mean.get(name))
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


def _discover_saved_ckpts(exp_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pattern, key in (
        ("ckpt_best_free_*.pth", "best_free"),
        ("ckpt_best_teacher_*.pth", "best_teacher"),
        ("ckpt_last_*.pth", "last"),
    ):
        matches = sorted(exp_dir.glob(pattern))
        if matches:
            out[key] = str(matches[0])
    epoch_ckpts = sorted(exp_dir.glob("ckpt_epoch_*.pth"))
    if epoch_ckpts:
        out["epoch_ckpt_count"] = str(len(epoch_ckpts))
    return out


def _discover_epoch_ckpts(exp_dir: Path, epoch_start: int, epoch_end: int) -> List[str]:
    out: List[str] = []
    for path in sorted(exp_dir.glob("ckpt_epoch_*.pth")):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        try:
            epoch = int(parts[2])
        except Exception:
            continue
        if epoch_start <= epoch <= epoch_end:
            out.append(str(path))
    return out


def _build_epoch_rows(exp_dir: Path, epoch_start: int, epoch_end: int) -> List[Dict[str, Any]]:
    metrics_dir = exp_dir / "metrics"
    rows: List[Dict[str, Any]] = []
    for epoch in range(int(epoch_start), int(epoch_end) + 1):
        free_metrics = _load_metric(metrics_dir, "valfree", epoch)
        teacher_metrics = _load_metric(metrics_dir, "teacher", epoch)
        if free_metrics is None or teacher_metrics is None:
            continue
        row = {
            "epoch": int(epoch),
            "freerun": {
                "geo_deg": _safe_float(free_metrics.get("GeoDeg")),
                "geo_local_deg": _safe_float(free_metrics.get("GeoLocalDeg")),
                "geo_deg_slope": _compute_geo_deg_slope(free_metrics),
                "geo_local_proxy": _safe_float(free_metrics.get("GeoDriftSlopeProxy")),
                "root_vel_mae": _safe_float(free_metrics.get("RootVelMAE")),
                "arm_mean": _extract_group_mean(free_metrics, "arm"),
                "trunk_mean": _extract_group_mean(free_metrics, "trunk"),
                "leg_mean": _extract_group_mean(free_metrics, "leg"),
            },
            "teacher": {
                "geo_local_deg": _safe_float(teacher_metrics.get("GeoLocalDeg")),
                "arm_mean": _extract_group_mean(teacher_metrics, "arm"),
                "trunk_mean": _extract_group_mean(teacher_metrics, "trunk"),
                "leg_mean": _extract_group_mean(teacher_metrics, "leg"),
            },
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            _safe_float(row["freerun"]["geo_deg_slope"]),
            _safe_float(row["freerun"]["arm_mean"]),
            _safe_float(row["freerun"]["trunk_mean"]),
            _safe_float(row["freerun"]["leg_mean"]),
            _safe_float(row["teacher"]["arm_mean"]),
            _safe_float(row["teacher"]["trunk_mean"]),
            _safe_float(row["teacher"]["geo_local_deg"]),
            int(row["epoch"]),
        )
    )
    return rows


def _render_md(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Basetrain Handoff Proxy Scan")
    lines.append("")
    lines.append(f"- exp_dir: `{summary['exp_dir']}`")
    lines.append(f"- epoch_range: `{summary['epoch_range']['start']}-{summary['epoch_range']['end']}`")
    lines.append(f"- recommended_epoch: `{summary['recommended_epoch']}`")
    if summary["notes"]:
        for note in summary["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    lines.append("## Proxy Ranking")
    lines.append("")
    lines.append("| rank | epoch | GeoDegSlope | arm_free | trunk_free | leg_free | GeoLocalProxy | GeoDeg | RootVelMAE | arm_teacher | trunk_teacher | teacher_GeoLocal |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in summary["ranking"]:
        row = item["row"]
        free = row["freerun"]
        teacher = row["teacher"]
        lines.append(
            f"| {item['rank']} | {row['epoch']} | {_fmt(free['geo_deg_slope'])} | {_fmt(free['arm_mean'])} | "
            f"{_fmt(free['trunk_mean'])} | {_fmt(free['leg_mean'])} | {_fmt(free['geo_local_proxy'])} | "
            f"{_fmt(free['geo_deg'])} | {_fmt(free['root_vel_mae'])} | {_fmt(teacher['arm_mean'])} | "
            f"{_fmt(teacher['trunk_mean'])} | {_fmt(teacher['geo_local_deg'])} |"
        )
    lines.append("")
    lines.append("## Saved Checkpoints")
    lines.append("")
    saved = summary["saved_ckpts"]
    if not saved:
        lines.append("- none")
    else:
        for key, value in saved.items():
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rank basetrain epochs with a Stage6-oriented proxy when exact per-epoch ckpts are unavailable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--exp-dir", required=True, help="Basetrain experiment directory")
    ap.add_argument("--epoch-start", type=int, default=6)
    ap.add_argument("--epoch-end", type=int, default=18)
    ap.add_argument("--out", default="", help="Output directory")
    args = ap.parse_args()

    exp_dir = _resolve(args.exp_dir)
    if not exp_dir.is_dir():
        raise SystemExit(f"[FATAL] missing exp dir: {exp_dir}")

    rows = _build_epoch_rows(exp_dir, int(args.epoch_start), int(args.epoch_end))
    if not rows:
        raise SystemExit("[FATAL] no teacher/valfree metric pairs found in the requested epoch range")

    ranking = [{"rank": idx + 1, "row": row} for idx, row in enumerate(rows)]
    saved_ckpts = _discover_saved_ckpts(exp_dir)
    epoch_ckpts = _discover_epoch_ckpts(exp_dir, int(args.epoch_start), int(args.epoch_end))
    notes: List[str] = []
    if not epoch_ckpts:
        notes.append("No ckpt_epoch_XXX checkpoints were found in this exp dir, so the epoch table is a metrics-only Stage6 proxy, not an exact downstream replay.")
        notes.append("Future exact scans can use train/training_MPL.py --save_epoch_ckpts --save_epoch_ckpts_start 6.")
    if "best_free" in saved_ckpts:
        notes.append("The saved best_free checkpoint tracks GeoDegSlope from valfree GeoDegCurve, not GeoDriftSlopeProxy.")

    summary = {
        "exp_dir": str(exp_dir),
        "epoch_range": {"start": int(args.epoch_start), "end": int(args.epoch_end)},
        "recommended_epoch": int(ranking[0]["row"]["epoch"]),
        "ranking": ranking,
        "saved_ckpts": saved_ckpts,
        "epoch_ckpts": epoch_ckpts,
        "notes": notes,
    }

    out_dir = _resolve(args.out) if str(args.out).strip() else (ROOT / "debug_output" / f"_tmp_basetrain_handoff_proxy_scan_{exp_dir.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(_render_md(summary), encoding="utf-8")
    print(out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
