#!/usr/bin/env python3
"""Audit c4/c5(cond dir) semantics on reconstructed diag batches.

Inputs:
- rho-delta JSON produced by tools/diagnose_cond_rho_delta.py

Outputs:
- Per-batch and aggregate statistics on raw cond channels (c4,c5,c6)
- Batch-level association between direction stats and Delta rho (c4/c5)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.dataset import MotionEventDataset
from train.posttrain_common import _merge_norm_spec


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 3:
        return float("nan")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    sa = float(a.std())
    sb = float(b.std())
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_dataset(data_root: Path, bundle: Path, pretrain_template: Path, seq_len: int) -> MotionEventDataset:
    norm_spec = _merge_norm_spec(bundle.resolve(), pretrain_template.resolve())
    ds = MotionEventDataset(
        data_dir=str(data_root.resolve()),
        seq_len=int(seq_len),
        paths=None,
        pose_hist_len=int(norm_spec.get("pose_hist_len", 0) or 0),
        norm_spec=norm_spec,
        index_mode="sliding",
    )
    ds.is_train = False
    return ds


def _rebuild_cond_raw(ds: MotionEventDataset, diag_pt: Path) -> np.ndarray:
    payload = torch.load(str(diag_pt), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid diag pt payload: {diag_pt}")
    clip_id = payload.get("clip_id")
    start = payload.get("start")
    if not (torch.is_tensor(clip_id) and torch.is_tensor(start)):
        raise RuntimeError(f"diag pt missing clip_id/start: {diag_pt}")

    clip_ids = [int(x) for x in clip_id.view(-1).tolist()]
    starts = [int(x) for x in start.view(-1).tolist()]
    if len(clip_ids) != len(starts) or not clip_ids:
        raise RuntimeError(f"invalid clip/start in {diag_pt}")

    pair_to_idx = {(int(cid), int(st)): int(i) for i, (cid, st) in enumerate(ds.index)}
    cond_list: List[np.ndarray] = []
    for cid, st in zip(clip_ids, starts):
        idx = pair_to_idx.get((cid, st))
        if idx is None:
            raise RuntimeError(f"missing sample for (clip_id,start)=({cid},{st}) in {diag_pt}")
        sample = ds[idx]
        if "cond_tgt_raw" not in sample:
            raise RuntimeError("dataset sample missing cond_tgt_raw")
        cond_list.append(np.asarray(sample["cond_tgt_raw"], dtype=np.float32))
    return np.stack(cond_list, axis=0)  # (B,T,C)


def _batch_stats(cond_raw: np.ndarray, delta4: float, delta5: float, name: str) -> Dict[str, Any]:
    flat = cond_raw.reshape(-1, cond_raw.shape[-1])
    c4 = flat[:, 4]
    c5 = flat[:, 5]
    c6 = flat[:, 6]
    dir_norm = np.linalg.norm(flat[:, 4:6], axis=-1)
    theta_deg = np.degrees(np.arctan2(c5, c4))

    return {
        "diag_name": name,
        "n_rows": int(flat.shape[0]),
        "delta_rho_c4": _safe_float(delta4),
        "delta_rho_c5": _safe_float(delta5),
        "action_channel_means": [float(flat[:, i].mean()) for i in range(4)],
        "action_channel_std": [float(flat[:, i].std()) for i in range(4)],
        "c4_mean": float(c4.mean()),
        "c4_std": float(c4.std()),
        "c4_abs_mean": float(np.abs(c4).mean()),
        "c5_mean": float(c5.mean()),
        "c5_std": float(c5.std()),
        "c5_abs_mean": float(np.abs(c5).mean()),
        "c5_pos_rate": float((c5 > 0).mean()),
        "c5_neg_rate": float((c5 < 0).mean()),
        "c5_zero_rate": float((np.abs(c5) <= 1e-6).mean()),
        "speed_mean": float(c6.mean()),
        "speed_std": float(c6.std()),
        "speed_min": float(c6.min()),
        "speed_max": float(c6.max()),
        "dir_norm_mean": float(dir_norm.mean()),
        "dir_norm_std": float(dir_norm.std()),
        "dir_norm_min": float(dir_norm.min()),
        "dir_norm_max": float(dir_norm.max()),
        "theta_deg_p10": float(np.percentile(theta_deg, 10)),
        "theta_deg_p50": float(np.percentile(theta_deg, 50)),
        "theta_deg_p90": float(np.percentile(theta_deg, 90)),
        "corr_c4_speed": _corr(c4, c6),
        "corr_c5_speed": _corr(c5, c6),
    }


def _aggregate(batch_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def arr(key: str) -> np.ndarray:
        return np.asarray([_safe_float(r.get(key, float("nan"))) for r in batch_rows], dtype=np.float64)

    c4_abs = arr("c4_abs_mean")
    c5_abs = arr("c5_abs_mean")
    c5_pos = arr("c5_pos_rate")
    c5_neg = arr("c5_neg_rate")
    d4 = arr("delta_rho_c4")
    d5 = arr("delta_rho_c5")

    # pooled stats via mean over batch-level summaries
    out = {
        "n_batches": int(len(batch_rows)),
        "c4_abs_mean_of_batches": float(np.nanmean(c4_abs)),
        "c5_abs_mean_of_batches": float(np.nanmean(c5_abs)),
        "c5_pos_rate_mean_of_batches": float(np.nanmean(c5_pos)),
        "c5_neg_rate_mean_of_batches": float(np.nanmean(c5_neg)),
        "delta4_mean": float(np.nanmean(d4)),
        "delta5_mean": float(np.nanmean(d5)),
        "corr_batch_c5_pos_rate_vs_delta5": _corr(c5_pos, d5),
        "corr_batch_c5_abs_mean_vs_delta5": _corr(c5_abs, d5),
        "corr_batch_c4_abs_mean_vs_delta4": _corr(c4_abs, d4),
    }
    return out


def _to_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Cond Dir Semantics Audit")
    lines.append("")
    lines.append(f"- source rho json: `{payload.get('source_rho_json', '')}`")
    lines.append(f"- diag batches: `{int(payload.get('n_batches', 0) or 0)}`")
    lines.append("- channel semantics (from data pipeline): `c4=dir_x`, `c5=dir_y`, `c6=speed_multiplier`")
    lines.append("")

    agg = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
    if agg:
        lines.append("## Aggregate")
        lines.append("")
        lines.append(
            f"- abs magnitude: `|c4|={_safe_float(agg.get('c4_abs_mean_of_batches', float('nan'))):.4f}`, "
            f"`|c5|={_safe_float(agg.get('c5_abs_mean_of_batches', float('nan'))):.4f}`"
        )
        lines.append(
            f"- c5 sign mix: `pos={_safe_float(agg.get('c5_pos_rate_mean_of_batches', float('nan')))*100:.1f}%`, "
            f"`neg={_safe_float(agg.get('c5_neg_rate_mean_of_batches', float('nan')))*100:.1f}%`"
        )
        lines.append(
            f"- delta means: `Δρ4={_safe_float(agg.get('delta4_mean', float('nan'))):+.4f}`, "
            f"`Δρ5={_safe_float(agg.get('delta5_mean', float('nan'))):+.4f}`"
        )
        lines.append(
            f"- batch-level association: corr(`c5_pos_rate`, `Δρ5`)={_safe_float(agg.get('corr_batch_c5_pos_rate_vs_delta5', float('nan'))):+.3f}, "
            f"corr(`|c5|`, `Δρ5`)={_safe_float(agg.get('corr_batch_c5_abs_mean_vs_delta5', float('nan'))):+.3f}"
        )
        lines.append("")

    rows = payload.get("batch_rows", []) if isinstance(payload.get("batch_rows"), list) else []
    if rows:
        lines.append("## Per-batch snapshot")
        lines.append("")
        lines.append("|diag|Δρ4|Δρ5|`|c4|`|`|c5|`|c5_pos|corr(c5,speed)|")
        lines.append("|:--|--:|--:|--:|--:|--:|--:|")
        for r in rows:
            lines.append(
                f"|`{r.get('diag_name', '')}`|"
                f"{_safe_float(r.get('delta_rho_c4', float('nan'))):+.4f}|"
                f"{_safe_float(r.get('delta_rho_c5', float('nan'))):+.4f}|"
                f"{_safe_float(r.get('c4_abs_mean', float('nan'))):.4f}|"
                f"{_safe_float(r.get('c5_abs_mean', float('nan'))):.4f}|"
                f"{_safe_float(r.get('c5_pos_rate', float('nan')))*100:.1f}%|"
                f"{_safe_float(r.get('corr_c5_speed', float('nan'))):+.3f}|"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit c4/c5/c6 semantics from diag batches and rho-delta output.")
    ap.add_argument("--rho-json", type=str, required=True)
    ap.add_argument("--data-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")
    args = ap.parse_args()

    rho_json = Path(args.rho_json).expanduser().resolve()
    payload = json.loads(rho_json.read_text(encoding="utf-8"))
    per_batch = payload.get("per_batch", []) if isinstance(payload.get("per_batch"), list) else []
    if not per_batch:
        raise SystemExit(f"[FATAL] no per_batch in {rho_json}")

    ds = _load_dataset(
        data_root=Path(args.data_root).expanduser().resolve(),
        bundle=Path(args.bundle).expanduser().resolve(),
        pretrain_template=Path(args.pretrain_template).expanduser().resolve(),
        seq_len=int(args.seq_len),
    )

    rows: List[Dict[str, Any]] = []
    for pb in per_batch:
        diag_pt = Path(str(pb.get("diag_pt", ""))).expanduser().resolve()
        if not diag_pt.is_file():
            raise SystemExit(f"[FATAL] diag pt missing: {diag_pt}")
        delta4 = _safe_float(pb.get("delta", {}).get("4", {}).get("delta_rho", float("nan")))
        delta5 = _safe_float(pb.get("delta", {}).get("5", {}).get("delta_rho", float("nan")))
        cond_raw = _rebuild_cond_raw(ds, diag_pt)
        rows.append(_batch_stats(cond_raw, delta4, delta5, diag_pt.name))

    out = {
        "source_rho_json": str(rho_json),
        "n_batches": int(len(rows)),
        "channel_semantics": {
            "c4": "dir_x (command direction x in world plane)",
            "c5": "dir_y (command direction y in world plane)",
            "c6": "speed_multiplier",
        },
        "aggregate": _aggregate(rows),
        "batch_rows": rows,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if str(args.out_json).strip():
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Saved] {out_json}")

    if str(args.out_md).strip():
        out_md = Path(args.out_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_to_markdown(out), encoding="utf-8")
        print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
