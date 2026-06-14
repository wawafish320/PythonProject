#!/usr/bin/env python3
"""Action-handoff in-betweening B1 probe — stage 3a wiring smoke (§7.3).

Per the plan
(`docs/aperiodic_transition/2026-05-30_action_handoff_inbetween_73_b1_probe_plan.md`):
this is the **3a wiring smoke**, NOT the binding B1 gate.

  - from-scratch tiny model, **NO checkpoint / no base-model dependency**;
  - **NO z-reach** (z rides on the base model → 3b); state-space metrics only;
  - reach numbers here are **NON-BINDING** and MUST NOT trigger the spec §6 STOP.

It validates the harness end-to-end: a short teacher-forced smoke train (loss should
decrease), then a FREE AR rollout from arbitrary Walk_F phases conditioned on each turn
target, reporting per-clip state-space rates (`clip_resumable_rate`, `pop_safe_rate`,
`fallback_rate`, plus a clearly-labelled state-space `reach_proxy_rate`). Reporting is
per-clip with Walk_L_To_R on its own row (review record: L_R has zero grounded
supervision). The binding gate with the real z-region `reach_rate` and checkpoint init is
stage 3b. All thresholds PROVISIONAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.action_handoff_inbetween_model import (  # noqa: E402
    GateThresholds,
    LossWeights,
    MinimalGoalAR,
    ModelConfig,
    StateNormalizer,
    compute_losses,
    evaluate_rollout_state_space,
)
from train.data.action_handoff_inbetween import (  # noqa: E402
    SEAM_LEN_K,
    TURN_CLIPS,
    WALK_F,
    InbetweenSampler,
    SamplerConfig,
    load_clip_states,
)

DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"


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


def _draw_batch(sampler: InbetweenSampler, n: int, rng: np.random.Generator):
    """Fixed-horizon batch (progress=0 → gap=gap_min=H) so shapes batch cleanly."""
    ctx, mid, seam = [], [], []
    for _ in range(n):
        s = sampler.sample(rng, progress=0.0)
        ctx.append(s.ctx)
        mid.append(s.gt_middle)
        seam.append(s.seam_target)
    return (
        torch.as_tensor(np.stack(ctx), dtype=torch.float32),
        torch.as_tensor(np.stack(mid), dtype=torch.float32),
        torch.as_tensor(np.stack(seam), dtype=torch.float32),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="B1 probe stage 3a wiring smoke (§7.3); from-scratch, no checkpoint, NON-BINDING."
    )
    p.add_argument("--z-features", type=Path, default=Path(DEFAULT_Z_FEATURES))
    p.add_argument("--npz-root", type=Path, default=Path(DEFAULT_NPZ_ROOT))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--steps", type=int, default=300, help="smoke train steps (minute-scale)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-starts", type=int, default=24, help="arbitrary Walk_F start phases (spec N≥20)")
    p.add_argument("--eval-horizon", type=int, default=24)
    p.add_argument("--goal-horizon", type=int, default=12, help="goal seam offset into the turn clip")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    z_path = Path(args.z_features)
    npz_root = Path(args.npz_root)
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or Path(
        f"debug_output/_tmp_action_handoff_inbetween_b1_probe_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    states = load_clip_states(z_path, npz_root)
    sampler = InbetweenSampler(states, SamplerConfig())
    cfg = sampler.config
    normalizer = StateNormalizer(states)
    std = normalizer.std  # [281] group-std for state-space pop/reach proxy

    model = MinimalGoalAR(ModelConfig(seam_len=cfg.seam_len, hidden=int(args.hidden)))
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    weights = LossWeights()

    # --- (1) teacher-forced smoke train (loss should decrease) ---
    model.train()
    loss_first: dict[str, float] = {}
    loss_last: dict[str, float] = {}
    loss_curve: list[float] = []
    for step in range(int(args.steps)):
        ctx, mid, seam = _draw_batch(sampler, int(args.batch), rng)
        ctx_n, mid_n, seam_n = normalizer.normalize(ctx), normalizer.normalize(mid), normalizer.normalize(seam)
        goal_n = seam_n  # goal = the seam window (encode_goal flattens it)
        total, comps = compute_losses(model, ctx_n, goal_n, mid_n, seam_n, weights)
        opt.zero_grad()
        total.backward()
        opt.step()
        loss_curve.append(comps["total"])
        if step == 0:
            loss_first = comps
        loss_last = comps
    loss_decreased = bool(loss_last["total"] < loss_first["total"])

    # --- (2) free-rollout eval, per turn target, from arbitrary Walk_F phases ---
    model.eval()
    thr = GateThresholds()
    hub = states[WALK_F]
    t_f = hub.shape[0]
    C, K = cfg.context_len, cfg.seam_len
    start_phases = [int(round(x)) % t_f for x in np.linspace(0, t_f - 1, int(args.n_starts))]

    per_clip: dict[str, Any] = {}
    with torch.no_grad():
        for clip in TURN_CLIPS:
            target = states[clip]
            g0 = int(min(args.goal_horizon, target.shape[0] - SEAM_LEN_K))
            goal_seam_raw = target[g0 : g0 + K]
            goal_n = normalizer.normalize(torch.as_tensor(goal_seam_raw, dtype=torch.float32)).unsqueeze(0)
            outcomes = []
            for i in start_phases:
                idx = (np.arange(i - C, i)) % t_f
                ctx_raw = hub[idx]
                ctx_n = normalizer.normalize(torch.as_tensor(ctx_raw, dtype=torch.float32)).unsqueeze(0)
                roll_n = model.rollout_free(ctx_n, goal_n, int(args.eval_horizon))
                roll_raw = normalizer.denormalize(roll_n)[0].cpu().numpy()
                outcomes.append(
                    evaluate_rollout_state_space(roll_raw, goal_seam_raw, std, thr)
                )
            n = len(outcomes)
            per_clip[clip] = {
                "n_starts": n,
                "clip_resumable_rate": float(np.mean([o["clip_resumable"] for o in outcomes])),
                "pop_safe_rate": float(np.mean([o["pop_safe"] for o in outcomes])),
                "fallback_rate": float(np.mean([o["fallback"] for o in outcomes])),
                "reach_proxy_rate": float(np.mean([o["reached_proxy"] for o in outcomes])),
                "mean_best_pose_d": float(np.mean([o["best_pose_d"] for o in outcomes])),
                "mean_pop": float(np.mean([o["pop"] for o in outcomes])),
                "mean_reach_proxy": float(np.mean([o["reach_proxy"] for o in outcomes])),
            }

    def _agg(key: str) -> float:
        return float(np.mean([per_clip[c][key] for c in TURN_CLIPS]))

    aggregate = {
        "clip_resumable_rate": _agg("clip_resumable_rate"),
        "pop_safe_rate": _agg("pop_safe_rate"),
        "fallback_rate": _agg("fallback_rate"),
        "reach_proxy_rate": _agg("reach_proxy_rate"),
        "note": "aggregate is informational only; per-clip (esp. Walk_L_To_R) is the signal.",
    }

    summary = {
        "task": "Action handoff in-betweening B1 probe — stage 3a wiring smoke (§7.3)",
        "stage": "3a",
        "binding": False,
        "no_checkpoint": True,
        "no_z_reach": True,
        "disclaimer": (
            "NON-BINDING wiring smoke: from-scratch tiny model, no checkpoint, no z-reach. "
            "reach/resumable numbers here CANNOT trigger the spec §6 STOP — a fair B1 gate "
            "requires checkpoint init (3b) and the z-region reach_rate."
        ),
        "z_features_path": str(z_path.resolve()),
        "npz_root": str(npz_root.resolve()),
        "config": {
            "context_len": C,
            "seam_len": K,
            "train_steps": int(args.steps),
            "batch": int(args.batch),
            "hidden": int(args.hidden),
            "lr": float(args.lr),
            "n_starts": len(start_phases),
            "eval_horizon": int(args.eval_horizon),
            "goal_horizon": int(args.goal_horizon),
            "thresholds_provisional": {
                "tau_pose": thr.tau_pose,
                "tau_pop": thr.tau_pop,
                "reach_proxy_thr": thr.reach_proxy_thr,
            },
            "loss_weights": {"middle": weights.middle, "reach": weights.reach, "seam_c1": weights.seam_c1},
        },
        "training": {
            "loss_first": loss_first,
            "loss_last": loss_last,
            "loss_decreased": loss_decreased,
        },
        "per_clip_state_space": per_clip,
        "aggregate_state_space": aggregate,
    }
    json_path = out_dir / "b1_probe_3a_summary.json"
    _dump_json(json_path, summary)

    lines: list[str] = []
    lines.append("# Action Handoff In-Betweening B1 Probe — Stage 3a Wiring Smoke (§7.3)")
    lines.append("")
    lines.append("> **NON-BINDING.** From-scratch tiny model, no checkpoint, no z-reach. These")
    lines.append("> numbers cannot trigger the spec §6 STOP — the binding B1 gate is stage 3b")
    lines.append("> (checkpoint init + z-region reach_rate).")
    lines.append("")
    lines.append(f"- z-features: {z_path.resolve()}")
    lines.append(
        f"- config: C={C}, K={K}, steps={args.steps}, batch={args.batch}, hidden={args.hidden}, "
        f"n_starts={len(start_phases)}, eval_horizon={args.eval_horizon} [PROVISIONAL]"
    )
    lines.append("")
    lines.append("## Smoke training (loss should decrease)")
    lines.append(
        f"- total: {_fmt(loss_first['total'])} → {_fmt(loss_last['total'])} "
        f"(decreased={loss_decreased})"
    )
    lines.append(
        f"- components last: L_middle={_fmt(loss_last['L_middle'])}, "
        f"L_reach={_fmt(loss_last['L_reach'])}, L_seam_C1={_fmt(loss_last['L_seam_C1'])}"
    )
    lines.append("")
    lines.append("## Per-clip free-rollout state-space metrics (Walk_L_To_R on its own row)")
    lines.append("| turn target | n | clip_resumable | pop_safe | fallback | reach_proxy | mean pose_d |")
    lines.append("|---|---|---|---|---|---|---|")
    for clip in TURN_CLIPS:
        m = per_clip[clip]
        lines.append(
            f"| {clip} | {m['n_starts']} | {_fmt(m['clip_resumable_rate'], 2)} | "
            f"{_fmt(m['pop_safe_rate'], 2)} | {_fmt(m['fallback_rate'], 2)} | "
            f"{_fmt(m['reach_proxy_rate'], 2)} | {_fmt(m['mean_best_pose_d'], 3)} |"
        )
    a = aggregate
    lines.append(
        f"| **aggregate** | — | {_fmt(a['clip_resumable_rate'], 2)} | {_fmt(a['pop_safe_rate'], 2)} | "
        f"{_fmt(a['fallback_rate'], 2)} | {_fmt(a['reach_proxy_rate'], 2)} | — |"
    )
    lines.append("")
    lines.append(
        "- `reach_proxy` is a STATE-SPACE proxy (distance into the goal seam region), NOT the "
        "spec z-region reach_rate (3b). Aggregate is informational; per-clip is the signal."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")

    md_path = out_dir / "b1_probe_3a_summary.md"
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(
        f"[ok 3a NON-BINDING] loss {loss_first['total']:.3f}→{loss_last['total']:.3f} "
        f"(decreased={loss_decreased}); per-clip clip_resumable: "
        + ", ".join(f"{c}={per_clip[c]['clip_resumable_rate']:.2f}" for c in TURN_CLIPS)
    )


if __name__ == "__main__":
    main()
