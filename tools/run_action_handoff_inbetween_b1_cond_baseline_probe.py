#!/usr/bin/env python3
"""Action-handoff in-betweening — §7.3 3b Slice 2: cond-driven baseline probe.

**NON-BINDING floor diagnostic** (plan
`docs/aperiodic_transition/2026-05-30_action_handoff_inbetween_73b_path_ab_plan.md` §2.2).

What it does: from N≥20 arbitrary Walk_F phases, free-run the BASE EventMotionModel
(init-from-ckpt, no grad, no goal head, no training) with the conditioning OVERRIDDEN toward
each target turn, capture ``hidden_pre`` at ``model._pasa_lnq`` during the rollout, and apply
the Slice-1 reach metric (``train/action_handoff_inbetween_reach``) + the state-space metric
(``train/action_handoff_inbetween_model.evaluate_rollout_state_space``) against each target's
RESUMABLE goal seam. Reuses the ``run_freerun_cycles`` base-AR machinery (no new rollout, no
change to its existing paths).

What it does NOT do / what it CANNOT conclude (read before trusting any number):
  - It is NOT the binding B1 gate. It has NO goal head / goal injection — the binding gate is
    plan §2.4 (base-space free-run AFTER a goal head exists). These numbers CANNOT trigger the
    spec §6 STOP. A low floor is NOT evidence against B1; a high floor does NOT "pass" B1.
  - It only tells us whether the EXISTING base ``cond`` conditioning already carries signal
    toward the turn anchor from an arbitrary Walk_F start.

Two locked-data findings shape the cond override (see
``train/action_handoff_inbetween_cond_probe``):
  1. ``act_oh`` is identical (``[0,1,0,0]``) across all five clips → action one-hot override is
     a no-op here; only ``cond_dir`` distinguishes a turn.
  2. The base model per-window-normalizes ``cond_in`` → a CONSTANT cond_dir override collapses
     to ~0 (looks like Walk_F). The turn signal is the cond_dir TRAJECTORY, so we inject the
     target turn's recorded cond trajectory normalized with that turn's own per-window stats.
     ``cond_reprojection`` is disabled so the model receives exactly this cond.

All thresholds (conv_norm_thr, tau_pose, tau_pop, N, horizon, goal_horizon) are PROVISIONAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train.validate.run_freerun_cycles as freerun  # noqa: E402
from train.action_handoff_inbetween_cond_probe import (  # noqa: E402
    aggregate_clip_record,
    build_cond_override,
    phase_seed_indices,
    rollout_to_egocentric,
    select_start_phases,
    turn_clip_order,
)
from train.action_handoff_inbetween_model import (  # noqa: E402
    GateThresholds,
    StateNormalizer,
    evaluate_rollout_state_space,
)
from train.action_handoff_inbetween_reach import (  # noqa: E402
    DEFAULT_CONV_NORM_THR,
    DEFAULT_END_WINDOW_K,
    LOCKED_CLIPS,
    build_hidden_pre_anchors,
    load_hidden_pre,
)
from train.data.action_handoff_inbetween import (  # noqa: E402
    RAW_COND_DIR_SLICE,
    SEAM_LEN_K,
    WALK_F,
    load_clip_states,
)

DEFAULT_CKPT = (
    "debug_output/_tmp_71_lr1e4_lowlr_downstream_20260504/lambda/checkpoints/"
    "ckpt_last_WalkF_stage7_lambda_from_lowlr72_lr1e4_20260504.pth"
)
DEFAULT_BUNDLE = "raw_data/processed_data/norm_template.json"
DEFAULT_PRETRAIN_TEMPLATE = "models/pretrain_template.json"
DEFAULT_ENCODER_BUNDLE = "models/motion_encoder_equiv_stageA.pt"
DEFAULT_NPZ_ROOT = "raw_data/processed_data"
DEFAULT_Z_FEATURES = "debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz"

FPS = 60.0


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "null"
    return f"{float(v):.{digits}f}"


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _make_runner_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=str(Path(args.checkpoint).expanduser()),
        bundle=str(Path(args.bundle).expanduser()),
        pretrain_template=str(Path(args.pretrain_template).expanduser()),
        encoder_bundle=str(Path(args.encoder_bundle).expanduser()),
        device=str(args.device),
        num_heads=4,
        dropout=0.1,
        context_len=int(args.context_len),
        depth=2,
        lambda_fusion_apply=True,
        allow_lambda_apply_off_ablation=False,
    )


def _build_probe_sample(
    clip: Any,
    *,
    idx: np.ndarray,
    cond_norm: np.ndarray,
    cond_raw: np.ndarray,
    cond_mu: np.ndarray,
    cond_std: np.ndarray,
    contact_dim: int,
    angvel_dim: int,
    pose_hist_dim: int,
) -> Dict[str, torch.Tensor]:
    """Walk_F seed rolled to an arbitrary phase (wrap), with cond overridden to the turn.

    Only ``cond`` is overridden (task scope); motion / contacts / angvel / pose_history are the
    Walk_F exogenous stream at the phase-rolled indices. ``run_freerun_cycles`` autoregresses
    the pose state and consumes cond / contacts / angvel as per-step exogenous inputs.
    """
    H = int(idx.shape[0])

    def _gather(arr, dim, dtype=np.float32):
        if arr is None:
            return torch.zeros((H, int(dim)), dtype=torch.float32)
        a = np.asarray(arr, dtype=dtype)[idx]
        return torch.from_numpy(np.ascontiguousarray(a)).float()

    sample: Dict[str, torch.Tensor] = {
        "motion": _gather(clip.X, clip.X.shape[1]),
        "gt_motion": _gather(clip.Y, clip.Y.shape[1]),
        "clip_id": torch.tensor(0, dtype=torch.int64),
        "start": torch.tensor(0, dtype=torch.int64),
        "cond_in": torch.from_numpy(np.ascontiguousarray(cond_norm.astype(np.float32))).float(),
        "cond_tgt": torch.from_numpy(np.ascontiguousarray(cond_norm.astype(np.float32))).float(),
        "cond_tgt_raw": torch.from_numpy(np.ascontiguousarray(cond_raw.astype(np.float32))).float(),
        "cond_norm_mu": torch.from_numpy(np.ascontiguousarray(cond_mu.astype(np.float32))).float(),
        "cond_norm_std": torch.from_numpy(np.ascontiguousarray(cond_std.astype(np.float32))).float(),
        "contacts": _gather(getattr(clip, "contacts", None), contact_dim),
        "angvel": _gather(getattr(clip, "angvel_norm", None), angvel_dim),
        "pose_hist": _gather(getattr(clip, "pose_hist_norm", None), pose_hist_dim),
    }
    if getattr(clip, "angvel_raw", None) is not None:
        sample["angvel_raw"] = _gather(clip.angvel_raw, clip.angvel_raw.shape[1])
    return sample


def _run_one_rollout(
    runner: Any,
    sample: Dict[str, torch.Tensor],
    *,
    rot6d_x_slice: slice,
    rootvel_x_slice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One free-run rollout → (hidden_pre [S,512], gen_rot6d [S,46,6], gen_root_vel [S,2]).

    Captures via hooks only (no change to ``run_freerun_cycles``): hidden_pre at
    ``model._pasa_lnq`` and the carried (normalized) state at ``model``; the carried state is
    denormalized to recover the GENERATED rot6d / root_vel.
    """
    model = runner.model
    hid: List[np.ndarray] = []
    carried: List[torch.Tensor] = []

    def _hid_hook(_m, inputs):
        x = inputs[0]
        if not torch.is_tensor(x):
            raise RuntimeError("hidden_pre capture: _pasa_lnq input is not a tensor")
        hid.append(x.detach().reshape(-1, x.shape[-1])[-1].float().cpu().numpy())

    def _state_hook(_m, inputs):
        carried.append(inputs[0].detach().float().cpu())

    h1 = model._pasa_lnq.register_forward_pre_hook(_hid_hook)  # type: ignore[attr-defined]
    h2 = model.register_forward_pre_hook(_state_hook)
    try:
        with torch.no_grad():
            freerun._run_freerun_cycles(
                trainer=runner.trainer,
                sample=sample,
                rounds=1,
                device=runner.device,
                cond_reprojection="off",  # model must see exactly the injected cond
            )
    finally:
        h1.remove()
        h2.remove()

    if not hid or not carried:
        raise RuntimeError("rollout produced no captured steps")
    hidden = np.stack(hid, axis=0)  # [S,512]
    states_norm = torch.cat(carried, dim=0)  # [S,419]
    states_raw = runner.trainer.normalizer.denorm_x(states_norm).cpu().numpy()
    gen_rot6d = states_raw[:, rot6d_x_slice].reshape(states_raw.shape[0], -1, 6)
    gen_root_vel = states_raw[:, rootvel_x_slice]
    return hidden, gen_rot6d, gen_root_vel


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="§7.3 3b Slice 2 cond-driven baseline probe (NON-BINDING floor; base ckpt)."
    )
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=str, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=str, default=DEFAULT_Z_FEATURES)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--n-starts", type=int, default=24, help="arbitrary Walk_F start phases (spec N≥20)")
    p.add_argument("--horizon", type=int, default=120, help="free-run steps (≈ K + cycles) [PROVISIONAL]")
    p.add_argument("--goal-horizon", type=int, default=12, help="resumable seam offset into the turn clip")
    p.add_argument("--conv-norm-thr", type=float, default=DEFAULT_CONV_NORM_THR)
    p.add_argument("--end-window-k", type=int, default=DEFAULT_END_WINDOW_K)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    npz_root = Path(args.npz_root)
    z_path = Path(args.z_features)
    if not Path(args.checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"debug_output/_tmp_action_handoff_inbetween_b1_cond_baseline_probe_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- reach anchors (frozen hidden_pre) + state-space goal targets ---
    anchors = build_hidden_pre_anchors(
        load_hidden_pre(z_path, LOCKED_CLIPS), end_window_k=int(args.end_window_k)
    )
    target_states = load_clip_states(z_path, npz_root, fps=FPS)
    std = StateNormalizer(target_states).std
    thr = GateThresholds()

    # --- base model + Walk_F seed clip ---
    runner = freerun.FreeRunCycleRunner(_make_runner_args(args))
    walk_npz = npz_root / f"{WALK_F}.npz"
    ds = runner._build_dataset(walk_npz, seq_len=64)
    runner._ensure_model_ready(ds)
    runner.model.eval()
    walk_clip = ds.clips[0]
    walk_len = int(walk_clip.X.shape[0])
    rot6d_x_slice = runner.trainer.rot6d_x_slice
    rootvel_x_slice = runner.trainer.rootvel_x_slice
    contact_dim = int(getattr(runner, "contact_dim", 0) or 0)
    angvel_dim = int(getattr(runner, "angvel_dim", 0) or 0)
    pose_hist_dim = int(getattr(runner, "pose_hist_dim", 0) or 0)

    H = int(args.horizon)
    K = SEAM_LEN_K
    start_phases = select_start_phases(walk_len, int(args.n_starts))
    turn_clips = turn_clip_order()

    per_clip: Dict[str, Any] = {}
    for clip in turn_clips:
        # cond override: inject the target turn's recorded cond trajectory (finding #2)
        with np.load(npz_root / f"{clip}.npz", allow_pickle=True) as d:
            turn_cond_raw = np.asarray(d["cond_in"], dtype=np.float32)
        cond = build_cond_override(turn_cond_raw, H)
        cond_dir_seq = cond.raw[:, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]]  # [H,2]

        # resumable goal seam (NOT the onset — turn onsets sit on the walk manifold)
        target = target_states[clip]
        g0 = int(min(args.goal_horizon, target.shape[0] - K))
        goal_seam_raw = target[g0 : g0 + K]

        reach_min_norms: List[float] = []
        state_outcomes: List[Dict[str, object]] = []
        for phase in start_phases:
            idx = phase_seed_indices(phase, H, walk_len)
            sample = _build_probe_sample(
                walk_clip,
                idx=idx,
                cond_norm=cond.norm,
                cond_raw=cond.raw,
                cond_mu=cond.mu,
                cond_std=cond.std,
                contact_dim=contact_dim,
                angvel_dim=angvel_dim,
                pose_hist_dim=pose_hist_dim,
            )
            hidden, gen_rot6d, gen_root_vel = _run_one_rollout(
                runner, sample, rot6d_x_slice=rot6d_x_slice, rootvel_x_slice=rootvel_x_slice
            )
            # reach in hidden_pre space (Slice-1 metric)
            reach_min_norms.append(float(anchors[clip].min_norm(hidden)))
            # state-space outcome vs the resumable goal seam
            s = min(gen_rot6d.shape[0], cond_dir_seq.shape[0])
            seed_contact = sample["contacts"].cpu().numpy()[:s]
            roll_raw = rollout_to_egocentric(
                gen_rot6d[:s], gen_root_vel[:s], cond_dir_seq[:s], seed_contact, fps=FPS
            )
            state_outcomes.append(evaluate_rollout_state_space(roll_raw, goal_seam_raw, std, thr))

        rec = aggregate_clip_record(reach_min_norms, state_outcomes, float(args.conv_norm_thr))
        rec["turn_len"] = cond.turn_len
        rec["goal_seam_offset"] = g0
        rec["anchor_well_defined"] = bool(anchors[clip].well_defined)
        per_clip[clip] = rec

    disclaimer = (
        "NON-BINDING floor diagnostic (plan §2.2). No goal head / goal injection / training. "
        "Reflects only whether the EXISTING base cond conditioning already carries signal toward "
        "the turn anchor from an arbitrary Walk_F start. It does NOT define a gate and CANNOT "
        "trigger the spec §6 STOP: a low floor is NOT evidence against B1, a high floor does NOT "
        "pass B1. The binding gate is plan §2.4 (base-space free-run after a goal head)."
    )
    summary = {
        "task": "Action-handoff in-betweening — §7.3 3b Slice 2 cond-driven baseline probe",
        "stage": "3b-slice2-cond-baseline-probe",
        "binding": False,
        "diagnostic_kind": "NON-BINDING floor",
        "disclaimer": disclaimer,
        "reach_space": "hidden_pre(512) [Path A+B]",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "z_features_path": str(z_path.resolve()),
        "npz_root": str(npz_root.resolve()),
        "data_findings": {
            "act_oh_uniform_across_clips": True,
            "act_oh_value": [0, 1, 0, 0],
            "cond_per_window_normalized": True,
            "cond_override": (
                "injected the target turn's recorded cond TRAJECTORY (act_oh no-op; cond_dir "
                "ramp is the turn signal) normalized with that turn's own per-window robust "
                "stats; cond_reprojection disabled. A constant cond_dir override would be a "
                "no-op under per-window normalization."
            ),
            "rollout_heading_source": (
                "injected cond_dir (pose rot6d is heading-invariant — root yaw ~constant across "
                "clips); ego_vel/yaw_rate use commanded heading. contact = Walk_F seed stream "
                "(only cond is overridden)."
            ),
            "clip_resumable_degenerate": (
                "pose_rot6d is heading-invariant, so ALL locomotion clips share a near-identical "
                "pose distribution (verified: a Walk_F rollout sits pose_d≈0.05 << tau_pose=0.15 "
                "from every turn seam, at ANY seam offset incl. the clip end). clip_resumable is "
                "therefore TRIVIALLY ~1.0 and fallback (=not clip_resumable) TRIVIALLY ~0.0 for "
                "this base-model floor — NOT a success signal. The informative floor signals are "
                "reach (hidden_pre regime distance to the turn anchor) and pop_safe (the "
                "heading-dependent ego_vel/yaw_rate/contact seam discontinuity)."
            ),
        },
        "config": {
            "context_len": int(args.context_len),
            "n_starts": len(start_phases),
            "horizon": H,
            "goal_horizon": int(args.goal_horizon),
            "seam_len_k": K,
            "walk_f_len": walk_len,
            "thresholds_provisional": {
                "conv_norm_thr": float(args.conv_norm_thr),
                "end_window_k": int(args.end_window_k),
                "tau_pose": thr.tau_pose,
                "tau_pop": thr.tau_pop,
            },
        },
        "per_clip": per_clip,
        "walk_l_to_r_row": "Walk_L_To_R",
        "binding_gate_remaining": (
            "Plan §2.4: extend run_freerun_cycles with goal injection + arbitrary phase + "
            "hidden_pre capture → reach; only that run is binding / can trigger spec §6 STOP."
        ),
    }
    json_path = out_dir / "cond_baseline_probe_summary.json"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# B1 Cond-Driven Baseline Probe — §7.3 3b Slice 2")
    lines.append("")
    lines.append("> **NON-BINDING floor diagnostic.** Base model + cond override only — NO goal")
    lines.append("> head, NO goal injection, NO training. Reflects only whether the EXISTING base")
    lines.append("> `cond` conditioning already carries signal toward the turn anchor from an")
    lines.append("> arbitrary Walk_F start. It does **not** define a gate and **cannot** trigger")
    lines.append("> the spec §6 STOP: a low floor is NOT evidence against B1, a high floor does")
    lines.append("> NOT pass B1. The binding gate is plan §2.4 (free-run after a goal head).")
    lines.append("")
    lines.append("## Data findings that shaped the cond override")
    lines.append("- `act_oh` is identical `[0,1,0,0]` across all five clips → action one-hot override is a no-op.")
    lines.append("- `cond_in` is per-window normalized → a CONSTANT cond_dir override collapses to ~0 (looks like Walk_F).")
    lines.append("- ⇒ we inject the target turn's recorded cond **trajectory** (cond_dir ramp), normalized with that turn's own per-window stats; `cond_reprojection` disabled.")
    lines.append("- Rollout heading = injected cond_dir (pose rot6d is heading-invariant); ego_vel/yaw_rate use commanded heading, contact = Walk_F seed stream.")
    lines.append("")
    lines.append("> **Read `clip_resumable`/`fallback` with care.** `pose_rot6d` is heading-invariant, so")
    lines.append("> every locomotion clip shares a near-identical pose distribution — a Walk_F rollout")
    lines.append("> sits `pose_d≈0.05 << tau_pose=0.15` from every turn seam at ANY offset (incl. clip")
    lines.append("> end). So `clip_resumable` is TRIVIALLY ~1.0 and `fallback` TRIVIALLY ~0.0 here —")
    lines.append("> **not** a success signal. The informative floor signals are `reach_*` (regime")
    lines.append("> distance to the turn anchor) and `pop_safe` (heading-dependent seam discontinuity).")
    lines.append("")
    lines.append(f"- checkpoint: {Path(args.checkpoint).name}")
    lines.append(
        f"- config: N={len(start_phases)}, horizon={H}, goal_horizon={args.goal_horizon}, K={K}, "
        f"conv_norm_thr={_fmt(args.conv_norm_thr, 2)} [all PROVISIONAL]"
    )
    lines.append("")
    lines.append("## Per-clip floor metrics (Walk_L_To_R on its own row — zero grounded supervision)")
    lines.append(
        "| turn target | N | reach_min_norm (mean/med/p90/min) | reach_floor_rate | "
        "clip_resumable_rate | pop_safe_rate | fallback_rate |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for clip in turn_clips:
        m = per_clip[clip]
        lines.append(
            f"| {clip} | {m['n_starts']} | "
            f"{_fmt(m['reach_min_norm_mean'], 2)}/{_fmt(m['reach_min_norm_median'], 2)}/"
            f"{_fmt(m['reach_min_norm_p90'], 2)}/{_fmt(m['reach_min_norm_min'], 2)} | "
            f"{_fmt(m['reach_floor_rate'], 2)} | {_fmt(m['clip_resumable_rate'], 2)} | "
            f"{_fmt(m['pop_safe_rate'], 2)} | {_fmt(m['fallback_rate'], 2)} |"
        )
    lines.append("")
    lines.append(
        "- `reach_min_norm` = min cos_dist(hidden_pre, anchor) / anchor_radius over the rollout; "
        "`reach_floor_rate` = fraction with min_norm ≤ conv_norm_thr. Both NON-BINDING."
    )
    lines.append("")
    lines.append("## Binding gate remaining")
    lines.append(f"- {summary['binding_gate_remaining']}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")
    md_path = out_dir / "cond_baseline_probe_summary.md"
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(
        "[ok 3b-slice2 NON-BINDING floor] reach_floor_rate: "
        + ", ".join(f"{c}={_fmt(per_clip[c]['reach_floor_rate'], 2)}" for c in turn_clips)
    )


if __name__ == "__main__":
    main()
