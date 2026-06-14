#!/usr/bin/env python3
"""Action-handoff in-betweening sampler coverage diagnostic (spec §7.2).

Validates the §7.2 sampler on the REAL 5 locked clips (the unit tests use
synthetic data). Zero training, no model, no checkpoint — it only draws samples
and reports the realized distribution, so the §7.3 model build starts from known
data-coverage facts rather than assumptions. It surfaces, in particular, the B1
data-coverage risk concentrated on Walk_L_To_R (the gate-failing turn clip).

Reports, across a small set of curriculum `progress` points:
  - realized sample-type mix vs the configured ratios (spec §2);
  - within-clip gap distribution per progress (curriculum 12→30);
  - within-clip biased-sampling lift: mean interest over the chosen masked
    middle vs the clip's mean interest (>1 ⇒ oversampling turn-onset / contact-
    transition / clip-edge regions, spec §2a);
  - per-turn grounded resolution: full-state φ, pose-only φ, onset contact_d,
    gate verdict, and the realized fallback-trigger rate per turn clip (spec §2b
    — Walk_L_To_R is expected to fall back every time).

Frozen artifacts only; no retrain; no checkpoint dependency (spec §0 lock).
Imports the sampler/state from `train.data.action_handoff_inbetween` (single
source of truth). All thresholds are PROVISIONAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data.action_handoff_inbetween import (  # noqa: E402
    DEFAULT_RATIOS,
    SAMPLE_TYPE_GROUNDED,
    SAMPLE_TYPE_WITHIN,
    TURN_CLIPS,
    InbetweenSampler,
    SamplerConfig,
    load_clip_states,
)

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
PROGRESS_POINTS = (0.0, 0.5, 1.0)


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


def _stat(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"min": None, "median": None, "max": None, "mean": None}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="In-betweening sampler coverage on real clips (spec §7.2); frozen artifacts."
    )
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-mix", type=int, default=6000, help="samples per progress for the type mix")
    p.add_argument("--n-grounded", type=int, default=4000, help="grounded-only draws for fallback rate")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    z_path = Path(args.z_features)
    npz_root = Path(args.npz_root)
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(
        f"debug_output/_tmp_action_handoff_inbetween_sampler_coverage_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    states = load_clip_states(z_path, npz_root)
    sampler = InbetweenSampler(states)
    cfg = sampler.config
    rng = np.random.default_rng(int(args.seed))

    clip_frames = {name: int(st.shape[0]) for name, st in states.items()}

    # --- (1) realized type mix + gap distribution + biased lift, per progress ---
    per_progress: dict[str, Any] = {}
    for progress in PROGRESS_POINTS:
        type_counts: Counter[str] = Counter()
        within_gaps: list[float] = []
        within_clip_counts: Counter[str] = Counter()
        within_interest_lift: list[float] = []
        for _ in range(int(args.n_mix)):
            s = sampler.sample(rng, progress=progress)
            type_counts[s.meta["sample_type"]] += 1
            # within-clip provenance (also when grounded fell back to within).
            if s.meta.get("clip") is not None and s.meta.get("middle_start") is not None:
                clip = s.meta["clip"]
                gap = int(s.meta["gap"])
                i = int(s.meta["middle_start"])
                within_gaps.append(float(gap))
                within_clip_counts[clip] += 1
                interest = sampler._interest[clip]
                mid_mean = float(interest[i : i + gap].mean())
                clip_mean = float(interest.mean())
                within_interest_lift.append(mid_mean / max(clip_mean, 1e-8))
        n = int(args.n_mix)
        per_progress[f"{progress:.2f}"] = {
            "type_mix": {k: type_counts[k] / n for k in DEFAULT_RATIOS},
            "configured_ratios": dict(DEFAULT_RATIOS),
            "within_gap_stats": _stat(within_gaps),
            "within_clip_selection": {k: within_clip_counts[k] for k in clip_frames},
            "within_biased_interest_lift": _stat(within_interest_lift),
        }

    # --- (2) per-turn grounded resolution (static) + realized fallback rate ---
    grounded_static: dict[str, Any] = {}
    for clip in TURN_CLIPS:
        a = sampler.grounded_alignment(clip, onset=0)
        grounded_static[clip] = {
            "pose_only_phi": a.pose_only_phi,
            "pose_only_contact_d": a.pose_only_contact_d,
            "full_state_phi": a.full_state_phi,
            "full_state_contact_d": a.full_state_contact_d,
            "full_state_pose_d": a.full_state_pose_d,
            "groundable": a.groundable,
        }

    fallback_counts: dict[str, Counter[str]] = {c: Counter() for c in TURN_CLIPS}
    grounded_draw_counts: Counter[str] = Counter()
    for _ in range(int(args.n_grounded)):
        s = sampler.sample_grounded(rng, progress=0.5)
        clip = s.meta["turn_clip"]
        grounded_draw_counts[clip] += 1
        kind = s.meta.get("fallback") or "grounded_ok"
        fallback_counts[clip][kind] += 1

    grounded_fallback = {}
    for clip in TURN_CLIPS:
        total = max(grounded_draw_counts[clip], 1)
        grounded_fallback[clip] = {
            "draws": grounded_draw_counts[clip],
            "grounded_ok_rate": fallback_counts[clip]["grounded_ok"] / total,
            "later_onset_rate": fallback_counts[clip]["later_onset"] / total,
            "within_clip_fallback_rate": fallback_counts[clip]["within_clip"] / total,
        }

    summary = {
        "task": "Action handoff in-betweening sampler coverage (spec §7.2, real clips)",
        "scope": "frozen-artifact sampler diagnostic; no training; no checkpoint dependency",
        "z_features_path": str(z_path.resolve()),
        "npz_root": str(npz_root.resolve()),
        "clip_frames": clip_frames,
        "config": {
            "context_len": cfg.context_len,
            "seam_len": cfg.seam_len,
            "gap_min": cfg.gap_min,
            "gap_max": cfg.gap_max,
            "pose_topk": cfg.pose_topk,
            "ground_contact_thr": cfg.ground_contact_thr,
            "ratios": dict(cfg.ratios),
        },
        "n_mix_per_progress": int(args.n_mix),
        "n_grounded_draws": int(args.n_grounded),
        "per_progress": per_progress,
        "grounded_static_resolution": grounded_static,
        "grounded_fallback_rate": grounded_fallback,
        "b1_coverage_note": (
            "Grounded supervision cleanly covers only the groundable turn clips; the "
            "gate-failing clip (Walk_L_To_R) is served entirely by within-clip + augmentation, "
            "so B1 generalization risk is concentrated there. The §6 gate must report per-clip, "
            "not collapsed."
        ),
    }
    json_path = out_dir / "inbetween_sampler_coverage_summary.json"
    _dump_json(json_path, summary)

    lines: list[str] = []
    lines.append("# Action Handoff In-Betweening Sampler Coverage (spec §7.2, real clips)")
    lines.append("")
    lines.append(f"- z-features: {z_path.resolve()}")
    lines.append(f"- raw npz root: {npz_root.resolve()}")
    lines.append(f"- clip frames: {clip_frames}")
    lines.append(
        f"- config: C={cfg.context_len}, K={cfg.seam_len}, gap {cfg.gap_min}→{cfg.gap_max}, "
        f"pose_topk={cfg.pose_topk}, ground_contact_thr={_fmt(cfg.ground_contact_thr, 2)} [PROVISIONAL]"
    )
    lines.append(f"- draws: n_mix/progress={args.n_mix}, n_grounded={args.n_grounded}, seed={args.seed}")
    lines.append("")
    lines.append("## Realized sample-type mix vs configured ratios")
    lines.append("| progress | within | grounded | augmented | gap min/med/max | biased interest lift (med) |")
    lines.append("|---|---|---|---|---|---|")
    for progress in PROGRESS_POINTS:
        d = per_progress[f"{progress:.2f}"]
        tm = d["type_mix"]
        g = d["within_gap_stats"]
        lift = d["within_biased_interest_lift"]["median"]
        lines.append(
            f"| {progress:.2f} | {_fmt(tm[SAMPLE_TYPE_WITHIN], 3)} | {_fmt(tm[SAMPLE_TYPE_GROUNDED], 3)} | "
            f"{_fmt(tm['start_state_augmented'], 3)} | "
            f"{_fmt(g['min'], 1)}/{_fmt(g['median'], 1)}/{_fmt(g['max'], 1)} | {_fmt(lift, 2)} |"
        )
    lines.append(
        f"- configured ratios: within={DEFAULT_RATIOS[SAMPLE_TYPE_WITHIN]}, "
        f"grounded={DEFAULT_RATIOS[SAMPLE_TYPE_GROUNDED]}, augmented={DEFAULT_RATIOS['start_state_augmented']}."
    )
    lines.append("- biased interest lift >1 ⇒ masked middles oversample turn-onset / contact-transition / edge regions (spec §2a).")
    lines.append("")
    lines.append("## Per-turn grounded resolution + realized fallback rate")
    lines.append("| turn clip | full-state φ (contact_d) | pose-only φ (contact_d) | groundable | grounded_ok | later_onset | within fallback |")
    lines.append("|---|---|---|---|---|---|---|")
    for clip in TURN_CLIPS:
        st = grounded_static[clip]
        fb = grounded_fallback[clip]
        lines.append(
            f"| {clip} | f{st['full_state_phi']} ({_fmt(st['full_state_contact_d'], 3)}) | "
            f"f{st['pose_only_phi']} ({_fmt(st['pose_only_contact_d'], 3)}) | {st['groundable']} | "
            f"{_fmt(fb['grounded_ok_rate'], 2)} | {_fmt(fb['later_onset_rate'], 2)} | "
            f"{_fmt(fb['within_clip_fallback_rate'], 2)} |"
        )
    lines.append("")
    lines.append(f"- B1 coverage: {summary['b1_coverage_note']}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")

    md_path = out_dir / "inbetween_sampler_coverage_summary.md"
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    gr = {c: round(grounded_fallback[c]["within_clip_fallback_rate"], 2) for c in TURN_CLIPS}
    print(f"[ok] within-clip fallback rate per turn: {gr}")


if __name__ == "__main__":
    main()
