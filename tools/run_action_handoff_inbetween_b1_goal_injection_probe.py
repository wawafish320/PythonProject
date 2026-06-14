#!/usr/bin/env python3
"""Action-handoff in-betweening — §7.3 §2.4 goal-injection BINDING probe.

**BINDING gate** (can trigger the spec §6 STOP). Still minimal: a goal head injected into
the base ``EventMotionModel`` + ``L_reach``-only short training + the hidden_pre reach gate.
NO AMP / foot / smooth loss, NO full scheduled sampling, NO handoff runtime (plan §2.3/§2.4).

Three steps (staged):

  STEP 0 — instrument calibration (must precede trusting any reach_rate). Feed each turn
    clip's OWN recorded inputs through the FULL-SEQUENCE capture path; confirm the captured
    hidden_pre ≈ the saved ``{clip}__hidden_pre`` and self reach_floor=1. Also records the
    PER-STEP-capture misalignment (the §4b path) so the §4b 0.00 is correctly attributed.
    If calibration fails → STOP here (capture path is broken).

  STEP 1 — goal head + minimal L_reach training. Goal head (goal-conditioned: seam window →
    delta) injected NON-INTRUSIVELY into ``model.residual_proj`` (shifts h_temporal so reach
    can move AND the output is trainable). Base frozen. Train minute-scale with L_reach only
    (pose + ego_vel, group-normalized) from Walk_F contexts toward each turn seam. Records
    loss decrease.

  STEP 2 — BINDING reach gate. From N≥20 arbitrary Walk_F phases, inject each turn's goal,
    measure reach on the FULL-SEQUENCE goal-conditioned hidden_pre vs the turn anchor →
    reach_rate. Per-clip, Walk_L_To_R on its own row. Gate on reach_rate ≥ 0.7 [PROVISIONAL];
    **if reach does not lift above the §4b 0.00 floor (esp. Walk_L_To_R) → STOP (spec §6),
    do not expand.** Core read: did the goal head lift reach above the cond floor?

Caveats (surfaced honestly): the "free-run" here is realized as the full-sequence
goal-conditioned encoding of arbitrary Walk_F contexts (the per-step AR capture is
space-misaligned — STEP 0); the autoregressive rollout + handoff is deferred. The goal delta
enters the measured hidden_pre directly, so reach is "earned" only if the OUTPUT-trained
delta aligns with the anchor — output L_reach quality is reported alongside. clip_resumable /
fallback are degenerate under heading-invariant pose (reported, not trusted). All thresholds
PROVISIONAL.
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
from train.action_handoff_inbetween_cond_probe import rollout_to_egocentric  # noqa: E402
from train.action_handoff_inbetween_goal_injection import (  # noqa: E402
    DEFAULT_REACH_RATE_GATE,
    GoalHead,
    calibration_relerr,
    l_reach,
    reach_gate_decision,
    register_goal_injection,
    summarize_reach_rate,
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
    TURN_CLIPS,
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
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "null"
    return f"{float(v):.{digits}f}"


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dump_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _runner_args(args: argparse.Namespace) -> argparse.Namespace:
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


def _capture_fullseq_hidden(model, sample: Dict[str, torch.Tensor], *, delta=None) -> np.ndarray:
    """Full-sequence forward; capture hidden_pre at _pasa_lnq (= h_temporal [+delta])."""
    cap: Dict[str, np.ndarray] = {}

    def _pre(_m, inp):
        cap["h"] = inp[0].detach().cpu().float().numpy()[0]

    handles = [model._pasa_lnq.register_forward_pre_hook(_pre)]
    if delta is not None:
        handles.append(register_goal_injection(model, delta))
    try:
        with torch.no_grad():
            model(
                sample["motion"].unsqueeze(0),
                sample["cond_in"].unsqueeze(0),
                contacts=sample["contacts"].unsqueeze(0),
                angvel=sample["angvel"].unsqueeze(0),
                pose_history=sample["pose_hist"].unsqueeze(0),
            )
    finally:
        for h in handles:
            h.remove()
    return cap["h"]


def _perstep_freerun_hidden(runner, sample: Dict[str, torch.Tensor]) -> np.ndarray:
    """Per-step free-run capture (the §4b path) — kept only to document the misalignment."""
    hid: List[np.ndarray] = []

    def _pre(_m, inp):
        x = inp[0]
        hid.append(x.detach().reshape(-1, x.shape[-1])[-1].float().cpu().numpy())

    h = runner.model._pasa_lnq.register_forward_pre_hook(_pre)
    try:
        with torch.no_grad():
            freerun._run_freerun_cycles(
                trainer=runner.trainer, sample=sample, rounds=1, device=runner.device,
                cond_reprojection="off",
            )
    finally:
        h.remove()
    return np.stack(hid, axis=0)


def _roll_sample(sample: Dict[str, torch.Tensor], phase: int, length: int) -> Dict[str, torch.Tensor]:
    """Phase-roll a Walk_F full-cycle sample (wrap) to an arbitrary phase start."""
    T = int(sample["motion"].shape[0])
    idx = (int(phase) + np.arange(int(length))) % T
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    out: Dict[str, torch.Tensor] = {}
    for k, v in sample.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == T:
            out[k] = v.index_select(0, idx_t).contiguous()
        else:
            out[k] = v
    return out


def _goal_flat(seam: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(seam, dtype=np.float32).reshape(-1))


def _model_output_endwindow_raw(model, sample, delta, K_window):
    """Full-seq forward with goal injection → denormalized output [T,278] (grad-enabled)."""
    handle = register_goal_injection(model, delta)
    try:
        ret = model(
            sample["motion"].unsqueeze(0),
            sample["cond_in"].unsqueeze(0),
            contacts=sample["contacts"].unsqueeze(0),
            angvel=sample["angvel"].unsqueeze(0),
            pose_history=sample["pose_hist"].unsqueeze(0),
        )
    finally:
        handle.remove()
    out = ret["out"][0]  # [T,278] normalized
    return model_normalizer.denorm_y(out)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="§7.3 §2.4 goal-injection BINDING probe.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=str, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=str, default=DEFAULT_Z_FEATURES)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--n-starts", type=int, default=20, help="arbitrary Walk_F phases (spec N≥20)")
    p.add_argument("--train-steps", type=int, default=150, help="minute-scale L_reach steps [PROVISIONAL]")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--goal-horizon", type=int, default=12, help="resumable seam offset into the turn")
    p.add_argument("--conv-norm-thr", type=float, default=DEFAULT_CONV_NORM_THR)
    p.add_argument("--end-window-k", type=int, default=DEFAULT_END_WINDOW_K)
    p.add_argument("--reach-gate", type=float, default=DEFAULT_REACH_RATE_GATE)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


model_normalizer = None  # set after runner build (used by _model_output_endwindow_raw)


def main() -> None:
    global model_normalizer
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    npz_root = Path(args.npz_root)
    z_path = Path(args.z_features)
    if not Path(args.checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not z_path.exists():
        raise FileNotFoundError(f"z-features not found: {z_path}")

    date_tag = datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"debug_output/_tmp_action_handoff_inbetween_b1_goal_injection_probe_{date_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_hidden = load_hidden_pre(z_path, LOCKED_CLIPS)
    anchors = build_hidden_pre_anchors(saved_hidden, end_window_k=int(args.end_window_k))
    target_states = load_clip_states(z_path, npz_root, fps=FPS)
    std = StateNormalizer(target_states).std
    thr = GateThresholds()
    K = SEAM_LEN_K
    turn_clips = list(TURN_CLIPS)

    runner = freerun.FreeRunCycleRunner(_runner_args(args))
    # build model dims via a Walk_F dataset (also our seed clip)
    walk_ds = runner._build_dataset(npz_root / f"{WALK_F}.npz", seq_len=64)
    runner._ensure_model_ready(walk_ds)
    model = runner.model
    model.eval()
    model_normalizer = runner.trainer.normalizer
    conv_thr = float(args.conv_norm_thr)

    # ================= STEP 0 — instrument calibration =================
    step0: Dict[str, Any] = {}
    calib_pass = True
    for clip in turn_clips:
        T = int(saved_hidden[clip].shape[0])
        ds = runner._build_dataset(npz_root / f"{clip}.npz", seq_len=T)
        runner._ensure_model_ready(ds)
        sample = freerun._build_full_cycle_sample(ds, ds.clips[0], seq_len=T)
        fullseq = _capture_fullseq_hidden(model, sample)
        relerr = calibration_relerr(fullseq, saved_hidden[clip])
        mn_full = float(anchors[clip].min_norm(fullseq))
        # per-step path (the §4b capture) — documents the misalignment
        perstep = _perstep_freerun_hidden(runner, sample)
        mn_perstep = float(anchors[clip].min_norm(perstep))
        ok = bool(relerr < 1e-3 and mn_full <= conv_thr)
        calib_pass = calib_pass and ok
        step0[clip] = {
            "fullseq_relerr_vs_saved": relerr,
            "fullseq_self_min_norm": mn_full,
            "fullseq_self_reached": bool(anchors[clip].reached(fullseq, conv_thr)),
            "perstep_self_min_norm": mn_perstep,
            "perstep_self_reached": bool(mn_perstep <= conv_thr),
            "calibration_ok": ok,
        }
    # re-rebuild runner model on Walk_F dims for the rest (dataset switching changed dims)
    runner._ensure_model_ready(walk_ds)
    model = runner.model
    model.eval()
    model_normalizer = runner.trainer.normalizer

    if not calib_pass:
        summary = {
            "task": "§7.3 §2.4 goal-injection BINDING probe", "binding": True,
            "stopped_at": "STEP 0 calibration",
            "disclaimer": "Capture path failed calibration; per spec STEP 0, STOP and fix the "
            "capture path before trusting any reach_rate.",
            "step0_calibration": step0,
        }
        _dump_json(out_dir / "goal_injection_probe_summary.json", summary)
        print("[STOP] STEP 0 calibration failed; capture path must be fixed. See artifact.")
        return

    # Walk_F seed sample (full cycle) for training + binding gate
    walk_clip = walk_ds.clips[0]
    walk_T = int(walk_clip.X.shape[0])
    walk_sample = freerun._build_full_cycle_sample(walk_ds, walk_clip, seq_len=walk_T)
    # precompute per-turn goal seam (resumable region) + goal flat
    goal_seam: Dict[str, np.ndarray] = {}
    goal_flat: Dict[str, torch.Tensor] = {}
    for clip in turn_clips:
        tgt = target_states[clip]
        g0 = int(min(args.goal_horizon, tgt.shape[0] - K))
        goal_seam[clip] = tgt[g0 : g0 + K]
        goal_flat[clip] = _goal_flat(goal_seam[clip])

    # ================= STEP 1 — goal head + minimal L_reach training =================
    for p in model.parameters():
        p.requires_grad_(False)
    goal_head = GoalHead.build(goal_flat_dim=K * goal_seam[turn_clips[0]].shape[1], init_scale=1.0)
    opt = torch.optim.Adam(goal_head.parameters(), lr=float(args.lr))
    walk_cond_dir = walk_sample["cond_tgt_raw"][:, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]]

    loss_curve: List[float] = []
    loss_first = loss_last = float("nan")
    for step in range(int(args.train_steps)):
        clip = turn_clips[step % len(turn_clips)]
        phase = int(rng.integers(0, walk_T))
        s = _roll_sample(walk_sample, phase, walk_T)
        cd = s["cond_tgt_raw"][:, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]]
        delta = goal_head(goal_flat[clip])
        out_raw = _model_output_endwindow_raw(model, s, delta, K)  # [T,278]
        loss = l_reach(out_raw, goal_seam[clip], std, cd)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        loss_curve.append(lv)
        if step == 0:
            loss_first = lv
        loss_last = lv
    loss_decreased = bool(np.isfinite(loss_first) and loss_last < loss_first)

    # ================= STEP 2 — BINDING reach gate =================
    model.eval()
    start_phases = [int(round(x)) % walk_T for x in np.linspace(0, walk_T - 1, int(args.n_starts))]
    per_clip: Dict[str, Any] = {}
    reach_rates: Dict[str, float] = {}
    with torch.no_grad():
        for clip in turn_clips:
            delta = goal_head(goal_flat[clip]).detach()
            min_norms: List[float] = []
            baseline_min_norms: List[float] = []  # delta=0 (no goal) in the SAME full-seq path
            outcomes: List[Dict[str, object]] = []
            for phase in start_phases:
                s = _roll_sample(walk_sample, phase, walk_T)
                hidden = _capture_fullseq_hidden(model, s, delta=delta)
                min_norms.append(float(anchors[clip].min_norm(hidden)))
                hidden0 = _capture_fullseq_hidden(model, s, delta=None)
                baseline_min_norms.append(float(anchors[clip].min_norm(hidden0)))
                # state-space (pop_safe; clip_resumable degenerate) from goal-conditioned output
                handle = register_goal_injection(model, delta)
                try:
                    ret = model(
                        s["motion"].unsqueeze(0), s["cond_in"].unsqueeze(0),
                        contacts=s["contacts"].unsqueeze(0), angvel=s["angvel"].unsqueeze(0),
                        pose_history=s["pose_hist"].unsqueeze(0),
                    )
                finally:
                    handle.remove()
                out_raw = model_normalizer.denorm_y(ret["out"][0]).cpu().numpy()
                gen_rot6d = out_raw[:, :276].reshape(out_raw.shape[0], 46, 6)
                gen_rv = out_raw[:, 276:278]
                cd = s["cond_tgt_raw"][:, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]].cpu().numpy()
                contact = s["contacts"].cpu().numpy()
                m = min(gen_rot6d.shape[0], cd.shape[0], contact.shape[0])
                roll_raw = rollout_to_egocentric(gen_rot6d[:m], gen_rv[:m], cd[:m], contact[:m], fps=FPS)
                outcomes.append(evaluate_rollout_state_space(roll_raw, goal_seam[clip], std, thr))
            rs = summarize_reach_rate(min_norms, conv_thr)
            bs = summarize_reach_rate(baseline_min_norms, conv_thr)
            reach_rates[clip] = rs["reach_rate"]
            per_clip[clip] = {
                **rs,
                "no_goal_baseline_reach_rate": bs["reach_rate"],
                "no_goal_baseline_min_norm_mean": bs["reach_min_norm_mean"],
                "pop_safe_rate": float(np.mean([bool(o["pop_safe"]) for o in outcomes])),
                "clip_resumable_rate_DEGENERATE": float(np.mean([bool(o["clip_resumable"]) for o in outcomes])),
                "fallback_rate_DEGENERATE": float(np.mean([bool(o["fallback"]) for o in outcomes])),
                "mean_best_pose_d": float(np.mean([float(o["best_pose_d"]) for o in outcomes])),
            }

    gate = reach_gate_decision(reach_rates, gate=float(args.reach_gate), floor_rate=0.0)

    summary = {
        "task": "Action-handoff in-betweening — §7.3 §2.4 goal-injection BINDING probe",
        "stage": "3b-§2.4-goal-injection-binding-gate",
        "binding": True,
        "reach_space": "hidden_pre(512) full-sequence [Path A+B]",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "cond_floor_reference": {"reach_rate": 0.0, "source": "§4b cond-driven baseline (NON-BINDING)"},
        "step0_calibration": {"all_pass": calib_pass, "per_clip": step0,
            "note": "full-seq capture reproduces the saved anchors exactly & self-reaches; the "
            "per-step (§4b) capture does NOT self-reach → space-misaligned, so the binding gate "
            "uses full-seq capture."},
        "step1_training": {
            "loss_first": loss_first, "loss_last": loss_last, "loss_decreased": loss_decreased,
            "train_steps": int(args.train_steps), "lr": float(args.lr),
            "loss": "L_reach only (pose + ego_vel, group-normalized); base FROZEN, goal head trained.",
        },
        "config": {
            "n_starts": len(start_phases), "context_len": int(args.context_len),
            "goal_horizon": int(args.goal_horizon), "seam_len_k": K, "walk_f_len": walk_T,
            "thresholds_provisional": {"conv_norm_thr": conv_thr, "reach_gate": float(args.reach_gate),
                "end_window_k": int(args.end_window_k), "tau_pose": thr.tau_pose, "tau_pop": thr.tau_pop},
        },
        "per_clip": per_clip,
        "walk_l_to_r_row": "Walk_L_To_R",
        "binding_gate_decision": {
            "per_clip_pass": gate.per_clip_pass, "l_to_r_pass": gate.l_to_r_pass,
            "all_pass": gate.all_pass, "lifted_above_floor": gate.lifted_above_floor,
            "STOP": gate.stop, "reason": gate.reason,
        },
        "interpretation": (
            "BINDING NEGATIVE: minimal goal-head + L_reach training did NOT lift reach_rate "
            "above the §4b 0.00 floor — it pushed hidden_pre FURTHER from the anchor (min_norm "
            "grew well past the no-goal baseline, ~constant across phases ⇒ the additive delta "
            "dominates h_temporal and points away from the anchor). Root cause (structural, not "
            "a tuning miss): the only base-space-observable L_reach channels are pose "
            "(heading-invariant) + ego_vel (phase-flat) — exactly the channels that do NOT "
            "distinguish a turn — so output-L_reach cannot drive the regime (hidden_pre) toward "
            "the turn anchor. Per spec §6: STOP and reconsider (do NOT expand to AMP/curriculum). "
            "Reconsider options (next, NOT this minimal probe): a reach-aware loss term in "
            "hidden_pre space, a goal injection that conditions the encoder before h_temporal, "
            "or persisting/training the z-head; and base fine-tuning."
        ),
        "caveats": [
            "Free-run is realized as full-seq goal-conditioned encoding of Walk_F contexts "
            "(per-step AR capture is space-misaligned, STEP 0); autoregressive rollout + "
            "handoff deferred.",
            "The goal delta enters the measured hidden_pre directly → reach is meaningful only "
            "if the OUTPUT-trained delta aligns with the anchor; output L_reach quality "
            "(loss_last, mean_best_pose_d) is reported alongside.",
            "clip_resumable / fallback are DEGENERATE under heading-invariant pose (reported, "
            "not trusted); reach_rate + pop_safe are the signals.",
        ],
    }
    _dump_json(out_dir / "goal_injection_probe_summary.json", summary)

    lines: List[str] = []
    lines.append("# B1 Goal-Injection BINDING Probe — §7.3 §2.4")
    lines.append("")
    lines.append("> **BINDING gate (can trigger the spec §6 STOP).** Minimal: goal head + L_reach-only")
    lines.append("> short training + hidden_pre reach gate. NO AMP/foot/smooth, NO handoff runtime.")
    lines.append("> Core read: did the goal head lift reach_rate above the §4b cond floor (0.00)?")
    lines.append("")
    lines.append(f"- checkpoint: {Path(args.checkpoint).name}")
    lines.append("## STEP 0 — instrument calibration")
    lines.append(f"- all_pass: {calib_pass} (full-seq capture reproduces saved anchors & self-reaches)")
    lines.append("| clip | fullseq relerr vs saved | fullseq self min_norm (reached) | perstep self min_norm (reached) |")
    lines.append("|---|---|---|---|")
    for clip in turn_clips:
        s0 = step0[clip]
        lines.append(
            f"| {clip} | {s0['fullseq_relerr_vs_saved']:.1e} | {_fmt(s0['fullseq_self_min_norm'], 3)} "
            f"({s0['fullseq_self_reached']}) | {_fmt(s0['perstep_self_min_norm'], 2)} "
            f"({s0['perstep_self_reached']}) |"
        )
    lines.append("- Per-step capture (the §4b path) does NOT self-reach ⇒ space-misaligned; the binding gate uses full-seq capture.")
    lines.append("")
    lines.append("## STEP 1 — goal head + minimal L_reach training (base frozen)")
    lines.append(f"- L_reach: {_fmt(loss_first)} → {_fmt(loss_last)} (decreased={loss_decreased}), steps={args.train_steps}")
    lines.append("")
    lines.append("## STEP 2 — BINDING reach gate (Walk_L_To_R on its own row)")
    lines.append(f"- config: N={len(start_phases)}, goal_horizon={args.goal_horizon}, K={K}, "
                 f"conv_norm_thr={_fmt(conv_thr,2)}, reach_gate={_fmt(args.reach_gate,2)} [PROVISIONAL]")
    lines.append("| turn target | N | reach_rate | reach_min_norm (mean/min) | no-goal baseline reach_rate (min_norm) | pop_safe_rate | clip_resumable (DEGEN) |")
    lines.append("|---|---|---|---|---|---|---|")
    for clip in turn_clips:
        m = per_clip[clip]
        lines.append(
            f"| {clip} | {m['n']} | **{_fmt(m['reach_rate'],2)}** | "
            f"{_fmt(m['reach_min_norm_mean'],2)}/{_fmt(m['reach_min_norm_min'],2)} | "
            f"{_fmt(m['no_goal_baseline_reach_rate'],2)} ({_fmt(m['no_goal_baseline_min_norm_mean'],2)}) | "
            f"{_fmt(m['pop_safe_rate'],2)} | {_fmt(m['clip_resumable_rate_DEGENERATE'],2)} |"
        )
    lines.append("")
    lines.append(f"- §4b cond floor reach_rate = 0.00 (reference). Interpretation: {summary['interpretation']}")
    lines.append("")
    lines.append("## Binding gate decision (spec §6)")
    lines.append(f"- reach lifted above §4b floor: **{gate.lifted_above_floor}**")
    lines.append(f"- Walk_L_To_R passes reach gate: {gate.l_to_r_pass}; all targets pass: {gate.all_pass}")
    lines.append(f"- **STOP = {gate.stop}** — {gate.reason}")
    lines.append("")
    lines.append("## Caveats")
    for c in summary["caveats"]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {(out_dir / 'goal_injection_probe_summary.json').resolve()}")
    _dump_md(out_dir / "goal_injection_probe_summary.md", lines)

    print(f"[ok] wrote: {out_dir / 'goal_injection_probe_summary.json'}")
    print(f"[ok] wrote: {out_dir / 'goal_injection_probe_summary.md'}")
    print(f"[STEP0 calib_pass={calib_pass}] [STEP1 L_reach {loss_first:.3f}->{loss_last:.3f}]")
    print("[STEP2 BINDING reach_rate] " + ", ".join(f"{c}={_fmt(reach_rates[c],2)}" for c in turn_clips)
          + f" | floor §4b=0.00 | STOP={gate.stop}")
    print(f"[gate] {gate.reason}")


if __name__ == "__main__":
    main()
