#!/usr/bin/env python3
"""Action-handoff in-betweening — B1 hidden_pre reach-anchor check (§7.3, 3b Slice 1).

Path A+B foundation: validates the binding-gate REACH METRIC (the measurement half) in
hidden_pre(512) space on FROZEN artifacts, before any generator exists.

  - Builds per-turn-clip hidden_pre anchors (end-window centroid + radius).
  - Reports anchor well-definedness (diffuseness vs the A2 0.80 bar) and a provisional
    CONV_DIST (= conv_norm_thr × radius).
  - Reports OFFLINE source separation: how far recorded Walk_F (and other clips') frames
    sit from each anchor — expected off-support offline (audit A2 in z-space), confirming
    reach must be measured on GENERATED rollouts, not recorded frames.

This is NOT the binding gate yet: it has no generator and runs no base model. The binding
gate needs Slice 2 (extend run_freerun_cycles: base EventMotionModel init-from-ckpt + goal
head + free-run + hidden_pre capture → reach via this metric). No training, no z-head, no
checkpoint dependency here (frozen hidden_pre only). All thresholds PROVISIONAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.action_handoff_inbetween_reach import (  # noqa: E402
    ANCHOR_DIFFUSE_THR,
    DEFAULT_CONV_NORM_THR,
    DEFAULT_END_WINDOW_K,
    LOCKED_CLIPS,
    TURN_CLIPS,
    build_hidden_pre_anchors,
    cos_dist,
    load_hidden_pre,
)

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "null"
    return f"{float(v):.{digits}f}"


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="B1 hidden_pre reach-anchor check (§7.3 3b Slice 1); frozen artifacts, no model."
    )
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--end-window-k", type=int, default=DEFAULT_END_WINDOW_K)
    p.add_argument("--conv-norm-thr", type=float, default=DEFAULT_CONV_NORM_THR)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    z_path = Path(args.z_features)
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(
        f"debug_output/_tmp_action_handoff_inbetween_reach_anchor_check_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden = load_hidden_pre(z_path, LOCKED_CLIPS)
    anchors = build_hidden_pre_anchors(hidden, TURN_CLIPS, int(args.end_window_k))

    per_anchor: dict[str, Any] = {}
    for tgt, a in anchors.items():
        conv_dist = float(args.conv_norm_thr) * a.radius
        sources = {}
        for src in LOCKED_CLIPS:
            if src == tgt:
                continue
            dmin = float(np.min(cos_dist(hidden[src], a.centroid)))
            sources[src] = {
                "offline_min_cos_dist": dmin,
                "offline_min_norm_radius": float(dmin / max(a.radius, 1e-8)),
                "offline_reaches": bool(dmin <= conv_dist),
            }
        per_anchor[tgt] = {
            "end_window_k": a.end_window_k,
            "anchor_radius_cos": a.radius,
            "clip_spread_cos": a.clip_spread,
            "diffuseness": a.diffuseness,
            "well_defined": a.well_defined,
            "radius_degenerate": a.radius_degenerate,
            "provisional_conv_dist": conv_dist,
            "offline_sources": sources,
        }

    all_well_defined = all(a.well_defined for a in anchors.values())
    summary = {
        "task": "Action handoff in-betweening B1 hidden_pre reach-anchor check (§7.3 3b Slice 1)",
        "stage": "3b-slice1-reach-metric-foundation",
        "binding": False,
        "no_model": True,
        "no_checkpoint": True,
        "reach_space": "hidden_pre(512) (Path A+B: hidden_pre carries z's regime info per A1; "
        "z-head was never persisted)",
        "z_features_path": str(z_path.resolve()),
        "provisional_thresholds": {
            "end_window_k": int(args.end_window_k),
            "conv_norm_thr": float(args.conv_norm_thr),
            "anchor_diffuse_thr": ANCHOR_DIFFUSE_THR,
        },
        "all_anchors_well_defined": all_well_defined,
        "per_anchor": per_anchor,
        "remaining_for_binding_gate": (
            "Slice 2: extend run_freerun_cycles — base EventMotionModel init-from-ckpt + goal "
            "head, free-run from arbitrary Walk_F phase conditioned on a turn anchor, capture "
            "hidden_pre during rollout, then apply this reach metric per-clip (Walk_L_To_R "
            "separate). Only that run is binding / can trigger the spec §6 STOP."
        ),
    }
    json_path = out_dir / "reach_anchor_check_summary.json"
    _dump_json(json_path, summary)

    lines: list[str] = []
    lines.append("# B1 hidden_pre Reach-Anchor Check (§7.3, 3b Slice 1)")
    lines.append("")
    lines.append("> Foundation only — the REACH METRIC in hidden_pre space, on frozen artifacts.")
    lines.append("> No generator, no base model, NON-BINDING. The binding gate is Slice 2.")
    lines.append("")
    lines.append(f"- hidden_pre source: {z_path.resolve()}")
    lines.append(
        f"- reach space: hidden_pre(512) [Path A+B]; end_window_k={args.end_window_k}, "
        f"conv_norm_thr={_fmt(args.conv_norm_thr, 2)} [PROVISIONAL]"
    )
    lines.append(f"- all anchors well-defined (diffuseness < {ANCHOR_DIFFUSE_THR}): {all_well_defined}")
    lines.append("")
    lines.append("## Per-target anchor + offline separation")
    lines.append("| target | radius | diffuseness | degenerate | well-def | CONV_DIST | Walk_F offline d_min (×radius) |")
    lines.append("|---|---|---|---|---|---|---|")
    for tgt, m in per_anchor.items():
        wf = m["offline_sources"].get("Walk_F", {})
        lines.append(
            f"| {tgt} | {_fmt(m['anchor_radius_cos'])} | {_fmt(m['diffuseness'], 3)} | "
            f"{m['radius_degenerate']} | {m['well_defined']} | {_fmt(m['provisional_conv_dist'])} | "
            f"{_fmt(wf.get('offline_min_cos_dist'))} ({_fmt(wf.get('offline_min_norm_radius'), 2)}x) |"
        )
    lines.append("")
    lines.append(
        "- Walk_F recorded frames sit several × the anchor radius away (off-support OFFLINE, "
        "mirroring audit A2 in z-space) → reach is only meaningful on GENERATED rollouts, not "
        "recorded frames."
    )
    lines.append("")
    lines.append("## Remaining for the binding gate")
    lines.append(f"- {summary['remaining_for_binding_gate']}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")

    md_path = out_dir / "reach_anchor_check_summary.md"
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[ok 3b-slice1 NON-BINDING] all_anchors_well_defined={all_well_defined}")


if __name__ == "__main__":
    main()
