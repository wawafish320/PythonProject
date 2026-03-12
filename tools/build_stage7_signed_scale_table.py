#!/usr/bin/env python3
"""
Build Stage7 artifacts from direct_leg_omega alpha-sweep output.

One script, two outputs:
1) Stage7.4 signed-scale table (`alpha_by_sic_bone`)
2) Stage7.3 SIC hotspot list derived from that table

Hotspot definition:
  hotspot_sic = {sic | exists bone s.t. abs(scale[sic][bone]) > threshold}

Input must be a `run_freerun_cycles` JSON containing `direct_leg_omega_alpha_sweep`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _parse_on_off_auto(spec: str) -> str:
    s = str(spec or "").strip().lower()
    if s in ("auto", "on", "off"):
        return s
    return "auto"


def _parse_bool(spec: str, *, default: bool) -> bool:
    s = str(spec or "").strip().lower()
    if not s:
        return bool(default)
    if s in ("1", "true", "on", "yes", "y", "enable", "enabled"):
        return True
    if s in ("0", "false", "off", "no", "n", "disable", "disabled"):
        return False
    return bool(default)


def _round_if_needed(v: float, ndigits: int) -> float:
    if ndigits < 0:
        return float(v)
    return float(round(float(v), int(ndigits)))


def _choose_mode(
    values: List[float],
    *,
    tie_break: str,
    neutral: float,
) -> Tuple[float, int]:
    c = Counter(values)
    max_cnt = max(c.values())
    cands = [float(k) for k, v in c.items() if int(v) == int(max_cnt)]
    if len(cands) == 1:
        return float(cands[0]), int(max_cnt)

    tie = str(tie_break or "").strip().lower()
    med = float(median(values))
    if tie == "maxabs":
        # Legacy behavior: prefer strongest absolute alpha when all counts tie.
        chosen = max(cands, key=lambda v: (abs(float(v)), float(v)))
    elif tie == "neutral":
        # Prefer alpha closest to neutral scale.
        chosen = min(cands, key=lambda v: (abs(float(v) - neutral), abs(float(v)), float(v)))
    else:
        # Default robust fallback for ties: closest to sample median.
        chosen = min(
            cands,
            key=lambda v: (abs(float(v) - med), abs(float(v) - neutral), abs(float(v)), float(v)),
        )
    return float(chosen), int(max_cnt)


def _aggregate(
    values: List[float],
    method: str,
    *,
    mode_tie_break: str,
    neutral: float,
) -> Tuple[float, Dict[str, Any]]:
    xs = [float(v) for v in values if _as_float(v) is not None]
    if not xs:
        raise ValueError("empty values")
    method_l = str(method or "").strip().lower()
    c = Counter(xs)
    mode_val, mode_cnt = _choose_mode(xs, tie_break=mode_tie_break, neutral=float(neutral))

    if method_l == "mode":
        agg = float(mode_val)
    elif method_l == "median":
        agg = float(median(xs))
    elif method_l == "mean":
        agg = float(sum(xs) / len(xs))
    elif method_l == "maxabs":
        agg = float(max(xs, key=lambda v: (abs(float(v)), float(v))))
    else:
        raise ValueError(f"unsupported aggregate method: {method}")

    meta = {
        "n_samples": int(len(xs)),
        "mode": float(mode_val),
        "mode_count": int(mode_cnt),
        "sample_min": float(min(xs)),
        "sample_max": float(max(xs)),
        "sample_abs_max": float(max(abs(float(min(xs))), abs(float(max(xs))))),
        "sample_mean": float(sum(xs) / len(xs)),
        "sample_median": float(median(xs)),
        "sample_hist": {str(k): int(v) for k, v in sorted(c.items(), key=lambda kv: (abs(float(kv[0])), float(kv[0])))},
    }
    return agg, meta


def _read_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"[FATAL] JSON root must be dict: {path}")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build Stage7 signed-scale table and Stage7.3 SIC hotspots from direct_leg_omega_alpha_sweep."
    )
    ap.add_argument("--freerun-json", type=str, required=True, help="Path to Walk_F_freerun_cycles.json.")
    ap.add_argument("--out-table", type=str, required=True, help="Output table JSON (for 7.4).")
    ap.add_argument("--out-hotspots", type=str, required=True, help="Output hotspot JSON (for 7.3 SIC list).")
    ap.add_argument(
        "--out-sic-csv",
        type=str,
        default="",
        help="Optional output text file containing comma-separated hotspot SIC list (diagnostic).",
    )
    ap.add_argument(
        "--aggregate",
        type=str,
        default="mode",
        choices=("mode", "median", "mean", "maxabs"),
        help="How to aggregate per-cycle best_alpha into one (sic,bone) scale.",
    )
    ap.add_argument(
        "--mode-tie-break",
        type=str,
        default="median",
        choices=("median", "neutral", "maxabs"),
        help=(
            "Tie-break strategy when --aggregate=mode and multiple values share max frequency: "
            "median=closest to sample median (recommended), "
            "neutral=closest to neutral-scale, "
            "maxabs=legacy strongest-|alpha|."
        ),
    )
    ap.add_argument(
        "--cycle-gte",
        type=int,
        default=-1,
        help="Filter lower bound for cycle. -1 => use alpha-sweep mask/default.",
    )
    ap.add_argument(
        "--drop-wrap",
        type=str,
        default="auto",
        choices=("auto", "on", "off"),
        help="Whether to drop wrap boundary steps. auto => use alpha-sweep mask/default.",
    )
    ap.add_argument(
        "--min-samples-per-pair",
        type=int,
        default=1,
        help="Ignore (sic,bone) pairs with fewer than this many samples after filtering.",
    )
    ap.add_argument(
        "--neutral-scale",
        type=float,
        default=1.0,
        help="Neutral scale. Sparse table keeps entries with |scale-neutral| > keep threshold.",
    )
    ap.add_argument(
        "--table-keep-abs-delta-from-neutral",
        type=float,
        default=0.0,
        help="Sparse table keep condition: |scale-neutral| > threshold.",
    )
    ap.add_argument(
        "--hotspot-abs-scale-threshold",
        type=float,
        default=2.0,
        help="Hotspot condition threshold applied to --hotspot-value-source.",
    )
    ap.add_argument(
        "--hotspot-value-source",
        type=str,
        default="scale",
        choices=("scale", "sample_abs_max"),
        help=(
            "Metric used for hotspot detection: "
            "scale=|aggregated scale|, "
            "sample_abs_max=max absolute raw best_alpha observed across cycles."
        ),
    )
    ap.add_argument(
        "--negative-scale-policy",
        type=str,
        default="keep",
        choices=("keep", "drop", "abs"),
        help=(
            "How to handle negative aggregated scales: "
            "keep=use signed value, drop=ignore pair (fallback to neutral), abs=use abs(scale)."
        ),
    )
    ap.add_argument(
        "--max-abs-scale",
        type=float,
        default=-1.0,
        help="Optional clamp on |scale| after aggregation/policy. <=0 disables clamp.",
    )
    ap.add_argument(
        "--round-decimals",
        type=int,
        default=6,
        help="Round output scale values to N decimals. Use -1 to disable rounding.",
    )
    ap.add_argument("--table-name", type=str, default="", help="Optional table name. Default: auto-generated.")
    ap.add_argument(
        "--include-all-pairs-in-hotspots-json",
        type=str,
        default="off",
        choices=("on", "off"),
        help="Include per-(sic,bone) aggregate stats for all pairs in hotspot JSON.",
    )
    ap.add_argument(
        "--stage73-config",
        type=str,
        default="",
        help="Optional Stage7.3 config JSON to update `direct_pose_loss_sics` from table-derived SICs.",
    )
    ap.add_argument(
        "--stage73-config-out",
        type=str,
        default="",
        help="Optional output path for updated Stage7.3 config. Default: overwrite --stage73-config.",
    )
    ap.add_argument(
        "--stage74-config",
        type=str,
        default="",
        help=(
            "Optional Stage7.4 config JSON to update "
            "`direct_pose_leg_scale_sup_alpha_table_json` and table-derived `direct_pose_loss_sics`."
        ),
    )
    ap.add_argument(
        "--stage74-config-out",
        type=str,
        default="",
        help="Optional output path for updated Stage7.4 config. Default: overwrite --stage74-config.",
    )
    args = ap.parse_args()

    p_in = Path(args.freerun_json).expanduser().resolve()
    p_table = Path(args.out_table).expanduser().resolve()
    p_hot = Path(args.out_hotspots).expanduser().resolve()
    p_csv = Path(args.out_sic_csv).expanduser().resolve() if str(args.out_sic_csv or "").strip() else None

    if not p_in.is_file():
        raise SystemExit(f"[FATAL] --freerun-json not found: {p_in}")

    src = _read_json(p_in)
    sw = src.get("direct_leg_omega_alpha_sweep", None)
    if not isinstance(sw, dict):
        raise SystemExit(
            "[FATAL] Missing 'direct_leg_omega_alpha_sweep' in input JSON. "
            "Re-run with --export_direct_leg_omega_alpha_sweep."
        )
    steps = sw.get("steps", None)
    if not isinstance(steps, list) or not steps:
        raise SystemExit("[FATAL] direct_leg_omega_alpha_sweep.steps is missing/empty.")

    mask = sw.get("mask", None)
    mask_cycle = 1
    mask_drop_wrap = True
    if isinstance(mask, dict):
        cyc = _as_int(mask.get("cycle_gte", 1))
        if cyc is not None:
            mask_cycle = int(cyc)
        mask_drop_wrap = bool(mask.get("drop_wrap", True))

    cycle_gte = int(args.cycle_gte) if int(args.cycle_gte) >= 0 else int(mask_cycle)
    drop_wrap_mode = _parse_on_off_auto(args.drop_wrap)
    if drop_wrap_mode == "on":
        drop_wrap = True
    elif drop_wrap_mode == "off":
        drop_wrap = False
    else:
        drop_wrap = bool(mask_drop_wrap)

    min_samples = max(1, int(args.min_samples_per_pair))
    neutral = float(args.neutral_scale)
    keep_delta = max(0.0, float(args.table_keep_abs_delta_from_neutral))
    hotspot_thr = max(0.0, float(args.hotspot_abs_scale_threshold))
    hotspot_src = str(args.hotspot_value_source).strip().lower()
    neg_policy = str(args.negative_scale_policy).strip().lower()
    max_abs_scale = float(args.max_abs_scale)
    round_dec = int(args.round_decimals)
    include_all_pairs = _parse_bool(args.include_all_pairs_in_hotspots_json, default=False)
    mode_tie_break = str(args.mode_tie_break).strip().lower()

    buckets: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    total_steps = 0
    kept_steps = 0

    for st in steps:
        if not isinstance(st, dict):
            continue
        total_steps += 1
        cyc = _as_int(st.get("cycle", 0))
        sic = _as_int(st.get("step_in_cycle", None))
        if cyc is None or sic is None:
            continue
        if int(cyc) < int(cycle_gte):
            continue
        if drop_wrap and bool(st.get("wrap_boundary_step", False)):
            continue
        per_bone = st.get("per_bone", None)
        if not isinstance(per_bone, dict):
            per_bone = st.get("bones", None)
        if not isinstance(per_bone, dict):
            continue
        kept_steps += 1
        for bone, dat in per_bone.items():
            if not isinstance(dat, dict):
                continue
            alpha = _as_float(dat.get("best_alpha", None))
            if alpha is None:
                continue
            buckets[(int(sic), str(bone))].append(float(alpha))

    if not buckets:
        raise SystemExit(
            f"[FATAL] No usable best_alpha samples after filter: cycle>={cycle_gte}, drop_wrap={drop_wrap}. "
            "Check input JSON / mask settings."
        )

    pair_rows: List[Dict[str, Any]] = []
    alpha_by_sic_bone: Dict[str, Dict[str, float]] = {}

    for (sic, bone), vals in sorted(buckets.items(), key=lambda kv: (int(kv[0][0]), str(kv[0][1]))):
        if len(vals) < min_samples:
            continue
        agg, meta = _aggregate(
            vals,
            str(args.aggregate),
            mode_tie_break=mode_tie_break,
            neutral=float(neutral),
        )
        raw_scale = _round_if_needed(agg, round_dec)
        scale = float(raw_scale)
        negative_dropped = False
        if scale < 0.0:
            if neg_policy == "drop":
                negative_dropped = True
            elif neg_policy == "abs":
                scale = abs(float(scale))
        if (not negative_dropped) and max_abs_scale > 0.0 and abs(scale) > max_abs_scale:
            scale = math.copysign(float(max_abs_scale), float(scale))
            scale = _round_if_needed(scale, round_dec)
        in_table = (not negative_dropped) and (abs(float(scale) - neutral) > keep_delta)
        row = {
            "sic": int(sic),
            "bone": str(bone),
            "raw_scale": float(raw_scale),
            "scale": float(scale),
            "abs_scale": float(abs(scale)),
            "in_table": bool(in_table),
            "negative_dropped": bool(negative_dropped),
            "aggregate": str(args.aggregate),
            **meta,
        }
        pair_rows.append(row)
        if in_table:
            k = str(int(sic))
            alpha_by_sic_bone.setdefault(k, {})
            alpha_by_sic_bone[k][str(bone)] = float(scale)

    if not pair_rows:
        raise SystemExit("[FATAL] No (sic,bone) pairs survived aggregation and min-samples filtering.")

    # Stable key ordering in JSON output.
    alpha_by_sic_bone = {
        str(k): {bk: float(bv) for bk, bv in sorted(v.items(), key=lambda x: str(x[0]))}
        for k, v in sorted(alpha_by_sic_bone.items(), key=lambda x: int(x[0]))
    }

    table_name = str(args.table_name or "").strip()
    if not table_name:
        table_name = f"{p_in.stem}_signed_scale_auto_{str(args.aggregate).lower()}"

    table_obj: Dict[str, Any] = {
        "name": table_name,
        "coord": "sic",
        "mask": {"cycle_gte": int(cycle_gte), "drop_wrap": bool(drop_wrap)},
        "alpha_by_sic_bone": alpha_by_sic_bone,
        "note": (
            "Auto-generated from direct_leg_omega_alpha_sweep.best_alpha. "
            "Defaults to scale=+1 for missing (sic,bone)."
        ),
        "meta": {
            "source_freerun_json": str(p_in),
            "aggregate": str(args.aggregate),
            "min_samples_per_pair": int(min_samples),
            "neutral_scale": float(neutral),
            "table_keep_abs_delta_from_neutral": float(keep_delta),
            "negative_scale_policy": str(neg_policy),
            "max_abs_scale": float(max_abs_scale),
            "round_decimals": int(round_dec),
            "input_steps_total": int(total_steps),
            "input_steps_after_mask": int(kept_steps),
            "pairs_total_after_agg": int(len(pair_rows)),
            "pairs_kept_in_table": int(sum(1 for r in pair_rows if bool(r.get("in_table", False)))),
            "pairs_negative_dropped": int(sum(1 for r in pair_rows if bool(r.get("negative_dropped", False)))),
        },
    }

    table_pairs = [r for r in pair_rows if bool(r.get("in_table", False))]
    hotspot_pairs = [
        {
            "sic": int(r["sic"]),
            "bone": str(r["bone"]),
            "scale": float(r["scale"]),
            "abs_scale": float(r["abs_scale"]),
            "n_samples": int(r["n_samples"]),
            "mode": float(r["mode"]),
            "mode_count": int(r["mode_count"]),
            "sample_mean": float(r["sample_mean"]),
            "sample_median": float(r["sample_median"]),
            "sample_min": float(r["sample_min"]),
            "sample_max": float(r["sample_max"]),
            "sample_abs_max": float(r["sample_abs_max"]),
            "sample_hist": dict(r["sample_hist"]),
        }
        for r in table_pairs
        if float(
            abs(r["scale"]) if hotspot_src == "scale" else float(r.get("sample_abs_max", abs(r["scale"])))
        )
        > hotspot_thr
    ]
    for r in hotspot_pairs:
        r["hotspot_value_source"] = str(hotspot_src)
        r["hotspot_score"] = float(abs(r["scale"]) if hotspot_src == "scale" else float(r["sample_abs_max"]))
    hotspot_pairs.sort(key=lambda r: (-float(r["hotspot_score"]), int(r["sic"]), str(r["bone"])))
    hotspot_sics = sorted({int(r["sic"]) for r in hotspot_pairs})
    hotspot_csv = ",".join(str(x) for x in hotspot_sics)

    # Config loss_sics should be consistent with the learned 7.4 supervision table.
    # Use SICs that have at least one non-zero scale entry in the sparse table.
    cfg_loss_sics = sorted(
        {
            int(r["sic"])
            for r in table_pairs
            if abs(float(r.get("scale", 0.0))) > 0.0
        }
    )
    cfg_loss_sics_csv = ",".join(str(x) for x in cfg_loss_sics)

    hot_obj: Dict[str, Any] = {
        "source_freerun_json": str(p_in),
        "source_table_json": str(p_table),
        "derivation": {
            "aggregate": str(args.aggregate),
            "cycle_gte": int(cycle_gte),
            "drop_wrap": bool(drop_wrap),
            "min_samples_per_pair": int(min_samples),
            "neutral_scale": float(neutral),
            "table_keep_abs_delta_from_neutral": float(keep_delta),
            "negative_scale_policy": str(neg_policy),
            "max_abs_scale": float(max_abs_scale),
            "hotspot_abs_scale_threshold": float(hotspot_thr),
            "hotspot_value_source": str(hotspot_src),
            "round_decimals": int(round_dec),
        },
        "counts": {
            "pairs_total_after_agg": int(len(pair_rows)),
            "pairs_kept_in_table": int(len(table_pairs)),
            "pairs_negative_dropped": int(sum(1 for r in pair_rows if bool(r.get("negative_dropped", False)))),
            "hotspot_pairs": int(len(hotspot_pairs)),
            "hotspot_sics": int(len(hotspot_sics)),
            "config_loss_sics": int(len(cfg_loss_sics)),
        },
        "hotspot_pairs": hotspot_pairs,
        "hotspot_sics": hotspot_sics,
        "hotspot_sics_csv": hotspot_csv,
        "config_loss_sics_source": "table_nonzero_scale",
        "config_loss_sics": cfg_loss_sics,
        "config_loss_sics_csv": cfg_loss_sics_csv,
    }
    if include_all_pairs:
        hot_obj["all_pairs_after_agg"] = pair_rows

    p_table.parent.mkdir(parents=True, exist_ok=True)
    p_hot.parent.mkdir(parents=True, exist_ok=True)
    p_table.write_text(json.dumps(table_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    p_hot.write_text(json.dumps(hot_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if p_csv is not None:
        p_csv.parent.mkdir(parents=True, exist_ok=True)
        p_csv.write_text(hotspot_csv + "\n", encoding="utf-8")

    print(f"[OK] wrote table: {p_table}")
    print(
        f"[OK] table entries={len(table_pairs)} sics={len(alpha_by_sic_bone)} "
        f"(pairs_after_agg={len(pair_rows)} filter=|scale-{neutral:g}|>{keep_delta:g})"
    )
    print(f"[OK] wrote hotspots: {p_hot}")
    hotspot_cond = (
        f"|scale|>{hotspot_thr:g}"
        if hotspot_src == "scale"
        else f"{hotspot_src}>{hotspot_thr:g}"
    )
    print(
        f"[OK] hotspot pairs={len(hotspot_pairs)} sics={len(hotspot_sics)} "
        f"(condition={hotspot_cond})"
    )
    print(f"[Diag] hotspot_sics_csv = {hotspot_csv if hotspot_csv else '<empty>'}")
    print(
        "[Stage7.x] direct_pose_loss_sics(table_nonzero_scale) = "
        f"{cfg_loss_sics_csv if cfg_loss_sics_csv else '<empty>'}"
    )
    if p_csv is not None:
        print(f"[OK] wrote SIC csv: {p_csv}")

    # Optional config updates (full integration path).
    stage73_cfg = str(args.stage73_config or "").strip()
    if stage73_cfg:
        p73_in = Path(stage73_cfg).expanduser().resolve()
        if not p73_in.is_file():
            raise SystemExit(f"[FATAL] --stage73-config not found: {p73_in}")
        p73_out = (
            Path(str(args.stage73_config_out).strip()).expanduser().resolve()
            if str(args.stage73_config_out or "").strip()
            else p73_in
        )
        obj73 = _read_json(p73_in)
        obj73["direct_pose_loss_sics"] = str(cfg_loss_sics_csv)
        p73_out.parent.mkdir(parents=True, exist_ok=True)
        p73_out.write_text(json.dumps(obj73, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] updated Stage7.3 config: {p73_out} (direct_pose_loss_sics={cfg_loss_sics_csv})")

    stage74_cfg = str(args.stage74_config or "").strip()
    if stage74_cfg:
        p74_in = Path(stage74_cfg).expanduser().resolve()
        if not p74_in.is_file():
            raise SystemExit(f"[FATAL] --stage74-config not found: {p74_in}")
        p74_out = (
            Path(str(args.stage74_config_out).strip()).expanduser().resolve()
            if str(args.stage74_config_out or "").strip()
            else p74_in
        )
        obj74 = _read_json(p74_in)
        # Keep config path style as user input (usually workspace-relative), not resolved absolute.
        obj74["direct_pose_leg_scale_sup_alpha_table_json"] = str(args.out_table)
        if "direct_pose_loss_sics" in obj74:
            obj74["direct_pose_loss_sics"] = str(cfg_loss_sics_csv)
        p74_out.parent.mkdir(parents=True, exist_ok=True)
        p74_out.write_text(json.dumps(obj74, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            "[OK] updated Stage7.4 config: "
            f"{p74_out} (direct_pose_leg_scale_sup_alpha_table_json={args.out_table}, "
            f"direct_pose_loss_sics={cfg_loss_sics_csv})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
