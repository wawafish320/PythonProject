#!/usr/bin/env python3
"""Action-handoff in-betweening §7.3 — reach-aware rewire BINDING probe.

Scope: frozen base + three ordered levers only:
  1. per-step AR hidden_pre calibration via context_len teacher windows,
  2. hidden_pre-space reach loss for the goal head,
  3. pre-temporal non-intrusive goal injection.

If lever 1 fails, the probe stops before training. If all three levers run and reach still
sticks to the §4c 0.00 floor, the STOP conclusion upgrades to: the frozen current
basetrain latent cannot be driven from Walk_F into the turn regime; base fine-tune /
goal-conditioned retraining is the next step.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train.validate.run_freerun_cycles as freerun  # noqa: E402
from train import rollout_kernel as _rollout_kernel  # noqa: E402
from train.action_handoff_inbetween_cond_probe import rollout_to_egocentric, select_start_phases  # noqa: E402
from train.action_handoff_inbetween_goal_injection import (  # noqa: E402
    DEFAULT_REACH_RATE_GATE,
    GoalHead,
    calibration_records_all_pass,
    calibration_relerr,
    context_window_indices,
    hidden_pre_anchor_loss,
    l_reach,
    loss_plateau_status,
    reach_gate_decision,
    register_goal_injection_pre_temporal,
    summarize_reach_rate,
)
from train.action_handoff_inbetween_model import GateThresholds, StateNormalizer, evaluate_rollout_state_space  # noqa: E402
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
from train.history import pose_hist_inverse_vec, pose_hist_transform_vec  # noqa: E402

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


@dataclass
class RolloutCapture:
    hidden_pre: torch.Tensor  # [H,512], graph-preserving when requested
    out_raw: torch.Tensor  # [H,278], graph-preserving when requested
    cond_dir_raw: torch.Tensor  # [H,2], Walk_F commanded heading used for state metrics
    contacts: torch.Tensor  # [H,2]


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or not np.isfinite(float(v)):
        return "null"
    return f"{float(v):.{digits}f}"


def _downsample(values: List[float], max_points: int) -> List[float]:
    """Evenly subsample a curve to <= ``max_points`` points (keep endpoints)."""
    n = len(values)
    if n <= int(max_points) or n == 0:
        return [float(v) for v in values]
    idx = np.linspace(0, n - 1, int(max_points)).round().astype(int)
    return [float(values[i]) for i in idx]


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


def _capture_fullseq_hidden(model, sample: Dict[str, torch.Tensor]) -> np.ndarray:
    cap: Dict[str, np.ndarray] = {}

    def _pre(_m, inp):
        cap["h"] = inp[0].detach().cpu().float().numpy()[0]

    h = model._pasa_lnq.register_forward_pre_hook(_pre)
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
        h.remove()
    return cap["h"]


def _capture_context_teacher_hidden(
    model,
    sample: Dict[str, torch.Tensor],
    *,
    context_len: int,
) -> np.ndarray:
    """Per-step capture with a real ``context_len`` teacher window; keeps the last step."""
    T = int(sample["motion"].shape[0])
    out: List[np.ndarray] = []
    for t in range(T):
        idx = torch.as_tensor(context_window_indices(t, context_len, T, mode="edge"), dtype=torch.long)
        cap: Dict[str, np.ndarray] = {}

        def _pre(_m, inp):
            cap["h"] = inp[0][:, -1].detach().cpu().float().numpy()[0]

        h = model._pasa_lnq.register_forward_pre_hook(_pre)
        try:
            with torch.no_grad():
                model(
                    sample["motion"].index_select(0, idx).unsqueeze(0),
                    sample["cond_in"].index_select(0, idx).unsqueeze(0),
                    contacts=sample["contacts"].index_select(0, idx).unsqueeze(0),
                    angvel=sample["angvel"].index_select(0, idx).unsqueeze(0),
                    pose_history=sample["pose_hist"].index_select(0, idx).unsqueeze(0),
                )
        finally:
            h.remove()
        out.append(cap["h"])
    return np.stack(out, axis=0)


def _phase_take(sample: Dict[str, torch.Tensor], key: str, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    src = sample[key]
    idx = torch.as_tensor(indices, dtype=torch.long)
    return src.index_select(0, idx).to(device=device).contiguous()


def _append_window(win: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 1:
        value = value.unsqueeze(0)
    return torch.cat([win[1:], value.detach()], dim=0)


def _next_pose_hist_norm(
    trainer,
    pose_hist_last: torch.Tensor,
    y_used_raw: torch.Tensor,
    *,
    rot_slice: slice,
) -> torch.Tensor:
    pose_hist_dim = int(getattr(trainer, "pose_hist_dim", 0) or 0)
    pose_hist_len = int(getattr(trainer, "pose_hist_len", 0) or 0)
    if pose_hist_dim <= 0 or pose_hist_len <= 0 or pose_hist_last.numel() == 0:
        return pose_hist_last.detach()
    stride = int(pose_hist_dim // max(1, pose_hist_len))
    if stride <= 0:
        return pose_hist_last.detach()
    scales = getattr(trainer, "pose_hist_scales", None)
    mu = getattr(trainer, "pose_hist_mu", None)
    std = getattr(trainer, "pose_hist_std", None)
    try:
        raw = pose_hist_inverse_vec(pose_hist_last.unsqueeze(0), scales, mu, std)
        raw_next = torch.roll(raw, shifts=-stride, dims=-1)
        raw_next[..., -stride:] = y_used_raw.detach()[..., rot_slice]
        return pose_hist_transform_vec(raw_next, scales, mu, std)[0].detach()
    except Exception:
        return pose_hist_last.detach()


def _run_context_ar_rollout(
    runner,
    sample: Dict[str, torch.Tensor],
    *,
    phase: int,
    horizon: int,
    context_len: int,
    delta: Optional[torch.Tensor],
    injection_targets: str = "shared_encoder.1",
    injection_mode: str = "additive",
    capture_grad: bool,
) -> RolloutCapture:
    """Windowed AR rollout: feed a context_len rolling window, carry only the last output."""
    model = runner.model
    trainer = runner.trainer
    device = runner.device
    T = int(sample["motion"].shape[0])
    C = int(context_len)
    idx0 = context_window_indices(int(phase), C, T, mode="wrap")
    motion_hist = _phase_take(sample, "motion", idx0, device)
    cond_hist = _phase_take(sample, "cond_in", idx0, device)
    cond_raw_hist = _phase_take(sample, "cond_tgt_raw", idx0, device)
    contacts_hist = _phase_take(sample, "contacts", idx0, device)
    pose_hist = _phase_take(sample, "pose_hist", idx0, device)

    free_carry_cfg = _rollout_kernel.resolve_free_carry_runtime_config(trainer)
    if bool(getattr(trainer, "use_freerun_state_sync", False)) and isinstance(free_carry_cfg.angvel_x_slice, slice):
        angvel_hist = motion_hist[:, free_carry_cfg.angvel_x_slice]
    else:
        angvel_hist = _phase_take(sample, "angvel", idx0, device)

    y_raw_prev = trainer._denorm(sample["gt_motion"][int(phase) % T].to(device=device).unsqueeze(0))
    motion_raw_last = trainer.normalizer.denorm_x(motion_hist[-1:].detach())

    hidden_steps: List[torch.Tensor] = []
    raw_steps: List[torch.Tensor] = []
    cond_dir_steps: List[torch.Tensor] = []
    contact_steps: List[torch.Tensor] = []

    hook = (
        register_goal_injection_pre_temporal(
            model,
            delta,
            targets=str(injection_targets),
            mode=str(injection_mode),
        )
        if delta is not None
        else None
    )
    try:
        for step in range(int(horizon)):
            cap: Dict[str, torch.Tensor] = {}

            def _pre(_m, inp):
                cap["h"] = inp[0][:, -1]

            hcap = model._pasa_lnq.register_forward_pre_hook(_pre)
            try:
                ret = model(
                    motion_hist.unsqueeze(0),
                    cond_hist.unsqueeze(0),
                    contacts=contacts_hist.unsqueeze(0),
                    angvel=angvel_hist.unsqueeze(0),
                    pose_history=pose_hist.unsqueeze(0),
                )
            finally:
                hcap.remove()
            out = ret["out"]
            if out.dim() == 3:
                delta_norm = out[:, -1]
            else:
                delta_norm = out
            try:
                y_used_raw = trainer._compose_delta_to_raw(y_raw_prev, delta_norm)
            except Exception:
                y_used_raw = trainer._denorm(delta_norm)
            hidden_steps.append(cap["h"][0] if cap["h"].dim() == 2 else cap["h"].reshape(-1, cap["h"].shape[-1])[-1])
            raw_steps.append(y_used_raw[0])
            cur_idx = (int(phase) + int(step)) % T
            cond_dir_steps.append(sample["cond_tgt_raw"][cur_idx, RAW_COND_DIR_SLICE[0] : RAW_COND_DIR_SLICE[1]].to(device))
            contact_steps.append(sample["contacts"][cur_idx].to(device))

            next_idx = (int(phase) + int(step) + 1) % T
            cond_next_raw = sample["cond_tgt_raw"][next_idx].to(device=device).unsqueeze(0)
            motion_raw_next = _rollout_kernel.apply_free_carry_raw(
                x_prev=motion_raw_last.detach(),
                y_next_raw=y_used_raw.detach(),
                cond_next_raw=cond_next_raw,
                rot6d_x_slice=free_carry_cfg.rot6d_x_slice,
                rot6d_y_slice=free_carry_cfg.rot6d_y_slice,
                angvel_x_slice=free_carry_cfg.angvel_x_slice,
                rootvel_x_slice=free_carry_cfg.rootvel_x_slice,
                rootpos_x_slice=free_carry_cfg.rootpos_x_slice,
                bone_hz=free_carry_cfg.bone_hz,
                columns=free_carry_cfg.columns,
            ).detach()
            motion_next = trainer._diag_norm_x(motion_raw_next)[0].detach()
            pose_hist_next = _next_pose_hist_norm(
                trainer,
                pose_hist[-1],
                y_used_raw,
                rot_slice=free_carry_cfg.rot6d_y_slice if isinstance(free_carry_cfg.rot6d_y_slice, slice) else slice(0, y_used_raw.shape[-1]),
            )

            motion_hist = _append_window(motion_hist, motion_next)
            cond_hist = _append_window(cond_hist, sample["cond_in"][next_idx].to(device))
            cond_raw_hist = _append_window(cond_raw_hist, sample["cond_tgt_raw"][next_idx].to(device))
            contacts_hist = _append_window(contacts_hist, sample["contacts"][next_idx].to(device))
            pose_hist = _append_window(pose_hist, pose_hist_next)
            if bool(getattr(trainer, "use_freerun_state_sync", False)) and isinstance(free_carry_cfg.angvel_x_slice, slice):
                angvel_hist = motion_hist[:, free_carry_cfg.angvel_x_slice]
            else:
                angvel_hist = _append_window(angvel_hist, sample["angvel"][next_idx].to(device))
            y_raw_prev = y_used_raw.detach()
            motion_raw_last = motion_raw_next
    finally:
        if hook is not None:
            hook.remove()

    hidden = torch.stack(hidden_steps, dim=0)
    raw = torch.stack(raw_steps, dim=0)
    cond_dir = torch.stack(cond_dir_steps, dim=0)
    contacts = torch.stack(contact_steps, dim=0)
    if not capture_grad:
        hidden = hidden.detach()
        raw = raw.detach()
    return RolloutCapture(hidden_pre=hidden, out_raw=raw, cond_dir_raw=cond_dir, contacts=contacts)


def _goal_flat(seam: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(seam, dtype=np.float32).reshape(-1))


def _eval_hidden_reach_snapshot(
    runner,
    walk_sample: Dict[str, torch.Tensor],
    anchors,
    goal_head,
    goal_flat: Dict[str, torch.Tensor],
    clips: List[str],
    *,
    phase: int,
    horizon: int,
    context_len: int,
    end_window_k: int,
    injection_targets: str = "shared_encoder.1",
    injection_mode: str = "additive",
) -> Dict[str, Any]:
    by_clip: Dict[str, Dict[str, float]] = {}
    losses: List[float] = []
    min_norms: List[float] = []
    with torch.no_grad():
        for clip in clips:
            delta = goal_head(goal_flat[clip]).detach()
            roll = _run_context_ar_rollout(
                runner,
                walk_sample,
                phase=int(phase),
                horizon=int(horizon),
                context_len=int(context_len),
                delta=delta,
                injection_targets=str(injection_targets),
                injection_mode=str(injection_mode),
                capture_grad=False,
            )
            centroid = torch.as_tensor(anchors[clip].centroid, dtype=roll.hidden_pre.dtype, device=roll.hidden_pre.device)
            loss = float(hidden_pre_anchor_loss(roll.hidden_pre, centroid, end_window_k=int(end_window_k)).detach().cpu())
            mn = float(anchors[clip].min_norm(roll.hidden_pre.detach().cpu().numpy()))
            by_clip[clip] = {"hidden_loss": loss, "min_norm": mn}
            losses.append(loss)
            min_norms.append(mn)
    return {
        "hidden_loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "min_norm_mean": float(np.mean(min_norms)) if min_norms else float("nan"),
        "by_clip": by_clip,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="§7.3 reach-aware rewire BINDING probe.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=str, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=str, default=DEFAULT_Z_FEATURES)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--train-steps", type=int, default=80)
    p.add_argument("--train-horizon", type=int, default=24)
    p.add_argument("--gate-horizon", type=int, default=72)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["none", "cosine", "step"])
    p.add_argument("--lr-step-size", type=int, default=200)
    p.add_argument("--lr-step-gamma", type=float, default=0.5)
    p.add_argument("--lr-floor", type=float, default=0.0, help="minimum LR after scheduler step (<=0 disables)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm for the goal head (<=0 disables)")
    p.add_argument("--goal-head-hidden", type=int, default=256)
    p.add_argument("--goal-head-depth", type=int, default=1)
    p.add_argument("--goal-head-mode", type=str, default="additive", choices=["additive", "film"])
    p.add_argument(
        "--goal-injection-targets",
        type=str,
        default="shared_encoder.1",
        help="comma-separated hook targets, e.g. shared_encoder.0 or shared_encoder.0,shared_encoder.1",
    )
    p.add_argument("--hidden-loss-weight", type=float, default=1.0)
    p.add_argument("--output-loss-weight", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=0, help="periodic eval-snapshot stride in train steps (0 = before/after only)")
    p.add_argument("--plateau-window", type=int, default=40)
    p.add_argument("--plateau-rel-delta", type=float, default=0.02)
    p.add_argument("--min-upgrade-train-steps", type=int, default=300)
    p.add_argument("--eval-plateau-window", type=int, default=3, help="plateau window in EVAL snapshots (eval min_norm curve)")
    p.add_argument("--eval-plateau-min-samples", type=int, default=6, help="min eval snapshots before an eval min_norm plateau can read True")
    p.add_argument("--goal-horizon", type=int, default=12)
    p.add_argument("--conv-norm-thr", type=float, default=DEFAULT_CONV_NORM_THR)
    p.add_argument("--end-window-k", type=int, default=DEFAULT_END_WINDOW_K)
    p.add_argument("--reach-gate", type=float, default=DEFAULT_REACH_RATE_GATE)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
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
        f"debug_output/_tmp_action_handoff_inbetween_reach_aware_rewire_probe_{date_tag}"
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
    walk_ds = runner._build_dataset(npz_root / f"{WALK_F}.npz", seq_len=64)
    runner._ensure_model_ready(walk_ds)
    model = runner.model
    model.eval()

    # ---------------- LEVER 1: context-window per-step calibration ----------------
    step1_records: Dict[str, Dict[str, Any]] = {}
    for clip in turn_clips:
        T = int(saved_hidden[clip].shape[0])
        ds = runner._build_dataset(npz_root / f"{clip}.npz", seq_len=T)
        runner._ensure_model_ready(ds)
        sample = freerun._build_full_cycle_sample(ds, ds.clips[0], seq_len=T)
        fullseq = _capture_fullseq_hidden(model, sample)
        context = _capture_context_teacher_hidden(model, sample, context_len=int(args.context_len))
        rec = {
            "fullseq_relerr_vs_saved": calibration_relerr(fullseq, saved_hidden[clip]),
            "fullseq_self_min_norm": float(anchors[clip].min_norm(fullseq)),
            "fullseq_self_reached": bool(anchors[clip].reached(fullseq, float(args.conv_norm_thr))),
            "context_window_shape": list(context.shape),
            "context_window_dtype": str(context.dtype),
            "context_window_device": "cpu",
            "context_relerr_vs_saved_fullseq": calibration_relerr(context, saved_hidden[clip]),
            "context_self_min_norm": float(anchors[clip].min_norm(context)),
            "context_self_reached": bool(anchors[clip].reached(context, float(args.conv_norm_thr))),
        }
        step1_records[clip] = rec
    lever1_pass = calibration_records_all_pass(step1_records, conv_norm_thr=float(args.conv_norm_thr))

    # Restore model dimensions to Walk_F after iterating turn datasets.
    runner._ensure_model_ready(walk_ds)
    model = runner.model
    model.eval()

    if not lever1_pass:
        summary = {
            "task": "Action-handoff in-betweening §7.3 reach-aware rewire probe",
            "binding": True,
            "stopped_at": "LEVER 1 per-step AR hidden_pre calibration",
            "step1_per_step_calibration": {"all_pass": False, "per_clip": step1_records},
            "decision": {
                "STOP": True,
                "reason": "context-window per-step capture did not self-reach the target anchors; hidden_pre reach gate remains uncalibrated.",
            },
        }
        _dump_json(out_dir / "reach_aware_rewire_probe_summary.json", summary)
        _dump_md(out_dir / "reach_aware_rewire_probe_summary.md", [
            "# Reach-Aware Rewire Probe",
            "",
            "- stopped_at: LEVER 1 per-step calibration",
            "- STOP: true",
            f"- JSON: {(out_dir / 'reach_aware_rewire_probe_summary.json').resolve()}",
        ])
        print("[STOP] LEVER 1 calibration failed; see artifact.")
        return

    # Walk_F seed sample for train/gate.
    walk_clip = walk_ds.clips[0]
    walk_T = int(walk_clip.X.shape[0])
    walk_sample = freerun._build_full_cycle_sample(walk_ds, walk_clip, seq_len=walk_T)
    start_phases = select_start_phases(walk_T, int(args.n_starts))

    goal_seam: Dict[str, np.ndarray] = {}
    goal_flat: Dict[str, torch.Tensor] = {}
    for clip in turn_clips:
        tgt = target_states[clip]
        g0 = int(min(args.goal_horizon, tgt.shape[0] - K))
        goal_seam[clip] = tgt[g0 : g0 + K]
        goal_flat[clip] = _goal_flat(goal_seam[clip]).to(runner.device)

    # ---------------- LEVER 2/3: hidden_pre loss + pre-temporal injection ----------------
    for p in model.parameters():
        p.requires_grad_(False)
    goal_head = GoalHead.build(
        goal_flat_dim=K * goal_seam[turn_clips[0]].shape[1],
        hidden=int(args.goal_head_hidden),
        depth=int(args.goal_head_depth),
        init_scale=0.0,
        mode=str(args.goal_head_mode),
    ).to(runner.device)
    opt = torch.optim.Adam(goal_head.parameters(), lr=float(args.lr))
    sched_kind = str(args.lr_schedule)
    if sched_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(args.train_steps)))
    elif sched_kind == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, int(args.lr_step_size)), gamma=float(args.lr_step_gamma))
    else:
        scheduler = None

    # --- convergence trajectory: eval the gated quantity (eval min_norm) periodically ---
    eval_steps: List[int] = []
    eval_hidden_curve: List[float] = []
    eval_min_norm_curve: List[float] = []
    eval_lr_curve: List[float] = []

    def _record_eval(step_idx: int) -> Dict[str, Any]:
        snap = _eval_hidden_reach_snapshot(
            runner,
            walk_sample,
            anchors,
            goal_head,
            goal_flat,
            turn_clips,
            phase=int(start_phases[0]),
            horizon=int(args.train_horizon),
            context_len=int(args.context_len),
            end_window_k=int(args.end_window_k),
            injection_targets=str(args.goal_injection_targets),
            injection_mode=str(args.goal_head_mode),
        )
        eval_steps.append(int(step_idx))
        eval_hidden_curve.append(float(snap["hidden_loss_mean"]))
        eval_min_norm_curve.append(float(snap["min_norm_mean"]))
        eval_lr_curve.append(float(opt.param_groups[0]["lr"]))
        return snap

    eval_before = _record_eval(0)

    loss_curve: List[float] = []
    hidden_curve: List[float] = []
    output_curve: List[float] = []
    eval_every = int(args.eval_every)
    for step in range(int(args.train_steps)):
        clip = turn_clips[step % len(turn_clips)]
        phase = int(rng.choice(start_phases))
        delta = goal_head(goal_flat[clip])
        roll = _run_context_ar_rollout(
            runner,
            walk_sample,
            phase=phase,
            horizon=int(args.train_horizon),
            context_len=int(args.context_len),
            delta=delta,
            injection_targets=str(args.goal_injection_targets),
            injection_mode=str(args.goal_head_mode),
            capture_grad=True,
        )
        centroid = torch.as_tensor(anchors[clip].centroid, dtype=roll.hidden_pre.dtype, device=roll.hidden_pre.device)
        loss_h = hidden_pre_anchor_loss(roll.hidden_pre, centroid, end_window_k=int(args.end_window_k))
        if int(roll.out_raw.shape[0]) >= K:
            loss_out = l_reach(
                roll.out_raw,
                goal_seam[clip],
                std,
                roll.cond_dir_raw,
            )
        else:
            loss_out = roll.hidden_pre.new_zeros(())
        loss = float(args.hidden_loss_weight) * loss_h + float(args.output_loss_weight) * loss_out
        opt.zero_grad()
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(goal_head.parameters(), float(args.grad_clip))
        opt.step()
        if scheduler is not None:
            scheduler.step()
        lr_floor = float(args.lr_floor)
        if lr_floor > 0:
            for pg in opt.param_groups:
                pg["lr"] = max(float(pg["lr"]), lr_floor)
        loss_curve.append(float(loss.detach().cpu()))
        hidden_curve.append(float(loss_h.detach().cpu()))
        output_curve.append(float(loss_out.detach().cpu()))
        if eval_every > 0 and (step + 1) % eval_every == 0 and (step + 1) < int(args.train_steps):
            _record_eval(step + 1)

    eval_after = _record_eval(int(args.train_steps))

    # ---------------- GATE: binding per-step AR free-run reach ----------------
    per_clip: Dict[str, Any] = {}
    reach_rates: Dict[str, float] = {}
    with torch.no_grad():
        for clip in turn_clips:
            delta = goal_head(goal_flat[clip]).detach()
            min_norms: List[float] = []
            baseline_min_norms: List[float] = []
            outcomes: List[Dict[str, object]] = []
            for phase in start_phases:
                roll = _run_context_ar_rollout(
                    runner,
                    walk_sample,
                    phase=int(phase),
                    horizon=int(args.gate_horizon),
                    context_len=int(args.context_len),
                    delta=delta,
                    injection_targets=str(args.goal_injection_targets),
                    injection_mode=str(args.goal_head_mode),
                    capture_grad=False,
                )
                base = _run_context_ar_rollout(
                    runner,
                    walk_sample,
                    phase=int(phase),
                    horizon=int(args.gate_horizon),
                    context_len=int(args.context_len),
                    delta=None,
                    capture_grad=False,
                )
                hidden_np = roll.hidden_pre.detach().cpu().numpy()
                hidden0_np = base.hidden_pre.detach().cpu().numpy()
                min_norms.append(float(anchors[clip].min_norm(hidden_np)))
                baseline_min_norms.append(float(anchors[clip].min_norm(hidden0_np)))

                out_np = roll.out_raw.detach().cpu().numpy()
                gen_rot6d = out_np[:, :276].reshape(out_np.shape[0], 46, 6)
                gen_rv = out_np[:, 276:278]
                cond_dir_np = roll.cond_dir_raw.detach().cpu().numpy()
                contact_np = roll.contacts.detach().cpu().numpy()
                m = min(gen_rot6d.shape[0], cond_dir_np.shape[0], contact_np.shape[0])
                roll_raw = rollout_to_egocentric(gen_rot6d[:m], gen_rv[:m], cond_dir_np[:m], contact_np[:m], fps=FPS)
                outcomes.append(evaluate_rollout_state_space(roll_raw, goal_seam[clip], std, thr))

            rs = summarize_reach_rate(min_norms, float(args.conv_norm_thr))
            bs = summarize_reach_rate(baseline_min_norms, float(args.conv_norm_thr))
            reach_rates[clip] = rs["reach_rate"]
            per_clip[clip] = {
                **rs,
                "no_goal_baseline_reach_rate": bs["reach_rate"],
                "no_goal_baseline_min_norm_mean": bs["reach_min_norm_mean"],
                "no_goal_baseline_min_norm_min": bs["reach_min_norm_min"],
                "pop_safe_rate": float(np.mean([bool(o["pop_safe"]) for o in outcomes])),
                "clip_resumable_rate_DEGENERATE": float(np.mean([bool(o["clip_resumable"]) for o in outcomes])),
                "fallback_rate_DEGENERATE": float(np.mean([bool(o["fallback"]) for o in outcomes])),
                "mean_best_pose_d": float(np.mean([float(o["best_pose_d"]) for o in outcomes])),
            }

    gate = reach_gate_decision(reach_rates, gate=float(args.reach_gate), floor_rate=0.0)
    # Plateau is judged on BOTH the train-batch hidden loss AND the EVAL min_norm trajectory
    # (the eval min_norm is the quantity the binding gate reads off the gate-rollout, so the
    # train-batch hidden loss can decouple from it). A NEGATIVE upgrade requires both to be
    # flat ("持平"): still-descending OR worsening on EITHER curve keeps it INCONCLUSIVE.
    plateau_train = loss_plateau_status(
        hidden_curve,
        min_steps=int(args.min_upgrade_train_steps),
        window=int(args.plateau_window),
        rel_delta=float(args.plateau_rel_delta),
    )
    plateau_eval_min_norm = loss_plateau_status(
        eval_min_norm_curve,
        min_steps=int(args.eval_plateau_min_samples),
        window=int(args.eval_plateau_window),
        rel_delta=float(args.plateau_rel_delta),
    )
    converged = bool(plateau_train["plateau"] and plateau_eval_min_norm["plateau"])
    plateau = {
        "plateau": converged,
        "train_hidden_loss": plateau_train,
        "eval_min_norm": plateau_eval_min_norm,
        "reason": (
            "converged: train hidden loss AND eval min_norm both plateaued"
            if converged
            else (
                "not converged ("
                f"train hidden loss: {plateau_train['reason']}; "
                f"eval min_norm: {plateau_eval_min_norm['reason']})"
            )
        ),
    }
    can_upgrade_negative = bool(gate.stop and converged)
    if bool(gate.stop) and not converged:
        gate_reason = (
            "reach_rate is still below the gate, but the frozen-base goal-head run is not "
            f"converged enough for an upgrade STOP: {plateau['reason']}. Treat this as "
            "BINDING/INCONCLUSIVE for base-finetune escalation; continue the cheap frozen-base "
            "goal-head training until the eval min_norm (and train hidden loss) plateau before deciding."
        )
    elif can_upgrade_negative and not gate.lifted_above_floor:
        gate_reason = (
            "reach_rate did not lift above the §4c/§4b 0.00 floor for any target after all "
            "three required levers were active (context-window per-step calibration, hidden_pre "
            "reach loss, pre-temporal injection). Upgrade conclusion: frozen current basetrain "
            "latent could not be driven from Walk_F to the turn regime; next step is base "
            "fine-tune / goal-conditioned retraining, not AMP/curriculum stacking."
        )
    elif can_upgrade_negative:
        gate_reason = (
            gate.reason
            + " All three required levers were active (context-window per-step calibration, "
            "hidden_pre reach loss, pre-temporal injection). Upgrade conclusion: frozen current "
            "basetrain latent could not be driven from Walk_F to the turn regime; next step is "
            "base fine-tune / goal-conditioned retraining, not AMP/curriculum stacking."
        )
    else:
        gate_reason = (
            gate.reason
            + " Reach lifted under the frozen-base rewire probe; current basetrain has a usable "
            "latent signal for the next controlled iteration."
        )

    summary = {
        "task": "Action-handoff in-betweening §7.3 reach-aware rewire BINDING probe",
        "stage": "frozen-base-three-lever-rewire",
        "binding": True,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "reach_space": "hidden_pre(512), fullseq anchors, per-step context-window capture",
        "baseline_reference": {
            "section_4c_floor_reach_rate": 0.0,
            "no_goal_baseline": "reported per clip in the same context-window AR path",
        },
        "lever1_per_step_calibration": {
            "all_pass": lever1_pass,
            "decision": (
                "aligned input pipeline: each step feeds a context_len teacher window and keeps the last hidden_pre; "
                "no per-step anchor redefinition was needed."
            ),
            "per_clip": step1_records,
        },
        "lever2_hidden_pre_loss": {
            "train_loss_first_observed": loss_curve[0] if loss_curve else float("nan"),
            "train_loss_last_observed": loss_curve[-1] if loss_curve else float("nan"),
            "train_hidden_loss_first_observed": hidden_curve[0] if hidden_curve else float("nan"),
            "train_hidden_loss_last_observed": hidden_curve[-1] if hidden_curve else float("nan"),
            "train_output_loss_first_observed": output_curve[0] if output_curve else float("nan"),
            "train_output_loss_last_observed": output_curve[-1] if output_curve else float("nan"),
            "eval_hidden_loss_before": eval_before["hidden_loss_mean"],
            "eval_hidden_loss_after": eval_after["hidden_loss_mean"],
            "eval_min_norm_before": eval_before["min_norm_mean"],
            "eval_min_norm_after": eval_after["min_norm_mean"],
            "eval_by_clip_before": eval_before["by_clip"],
            "eval_by_clip_after": eval_after["by_clip"],
            "hidden_loss_decreased": bool(eval_after["hidden_loss_mean"] < eval_before["hidden_loss_mean"]),
            "min_norm_decreased": bool(eval_after["min_norm_mean"] < eval_before["min_norm_mean"]),
            "plateau_status": plateau,
            "convergence_trajectory": {
                "eval_steps": eval_steps,
                "eval_hidden_loss": [float(v) for v in eval_hidden_curve],
                "eval_min_norm": [float(v) for v in eval_min_norm_curve],
                "lr_at_eval": [float(v) for v in eval_lr_curve],
                "train_hidden_loss_downsampled": _downsample(hidden_curve, 60),
                "train_total_loss_downsampled": _downsample(loss_curve, 60),
            },
            "note": "hidden_pre tensors used for loss are graph-preserving torch.float32 tensors on the selected device; metric captures are detached only for diagnostics.",
        },
        "lever3_pre_temporal_injection": {
            "hook": "register_goal_injection_pre_temporal(model, delta)",
            "target": str(args.goal_injection_targets),
            "mode": str(args.goal_head_mode),
            "target_default_reference": "model.shared_encoder[1] forward_hook, before shared_encoder[2:] and before fw.h_temporal / _pasa_lnq",
            "models_py_modified": False,
            "legacy_residual_proj_entry_preserved": True,
        },
        "config": {
            "context_len": int(args.context_len),
            "n_starts": len(start_phases),
            "train_steps": int(args.train_steps),
            "train_horizon": int(args.train_horizon),
            "gate_horizon": int(args.gate_horizon),
            "lr": float(args.lr),
            "lr_schedule": str(args.lr_schedule),
            "lr_floor": float(args.lr_floor),
            "grad_clip": float(args.grad_clip),
            "goal_head_hidden": int(args.goal_head_hidden),
            "goal_head_depth": int(args.goal_head_depth),
            "goal_head_mode": str(args.goal_head_mode),
            "goal_injection_targets": str(args.goal_injection_targets),
            "eval_every": int(args.eval_every),
            "plateau_window": int(args.plateau_window),
            "plateau_rel_delta": float(args.plateau_rel_delta),
            "min_upgrade_train_steps": int(args.min_upgrade_train_steps),
            "eval_plateau_window": int(args.eval_plateau_window),
            "eval_plateau_min_samples": int(args.eval_plateau_min_samples),
            "goal_horizon": int(args.goal_horizon),
            "seam_len_k": K,
            "walk_f_len": walk_T,
            "thresholds_provisional": {
                "conv_norm_thr": float(args.conv_norm_thr),
                "reach_gate": float(args.reach_gate),
                "end_window_k": int(args.end_window_k),
                "tau_pose": thr.tau_pose,
                "tau_pop": thr.tau_pop,
            },
        },
        "per_clip": per_clip,
        "walk_l_to_r_row": "Walk_L_To_R",
        "binding_gate_decision": {
            "per_clip_pass": gate.per_clip_pass,
            "l_to_r_pass": gate.l_to_r_pass,
            "all_pass": gate.all_pass,
            "lifted_above_floor": gate.lifted_above_floor,
            "raw_reach_stop": bool(gate.stop),
            "upgrade_negative": bool(can_upgrade_negative),
            "STOP": bool(gate.stop),
            "reason": gate_reason,
        },
        "stop_semantics": {
            "scope": "frozen-base + LEVER1 context-window calibration + LEVER2 hidden_pre reach loss + LEVER3 pre-temporal injection",
            "if_failed_after_plateau_next_step": "base fine-tune / goal-conditioned retraining",
            "if_not_plateau_next_step": "continue cheap frozen-base goal-head training",
            "do_not_expand_to": ["AMP", "feature matching", "L_foot", "L_smooth", "curriculum stacking"],
        },
    }

    json_path = out_dir / "reach_aware_rewire_probe_summary.json"
    md_path = out_dir / "reach_aware_rewire_probe_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# Reach-Aware Rewire BINDING Probe — §7.3")
    lines.append("")
    lines.append("> Scope: frozen base + per-step context calibration + hidden_pre reach loss + pre-temporal injection.")
    lines.append("")
    lines.append("## LEVER 1 — per-step AR hidden_pre calibration")
    lines.append(f"- all_pass: **{lever1_pass}**; choice: align input pipeline, no anchor redefinition.")
    lines.append("| clip | fullseq relerr | fullseq self min_norm | context relerr | context self min_norm (reached) |")
    lines.append("|---|---|---|---|---|")
    for clip in turn_clips:
        r = step1_records[clip]
        lines.append(
            f"| {clip} | {r['fullseq_relerr_vs_saved']:.1e} | {_fmt(r['fullseq_self_min_norm'],3)} | "
            f"{_fmt(r['context_relerr_vs_saved_fullseq'],4)} | {_fmt(r['context_self_min_norm'],3)} "
            f"({r['context_self_reached']}) |"
        )
    lines.append("")
    lines.append("## LEVER 2/3 — training")
    lines.append(
        f"- eval hidden_pre loss: {_fmt(summary['lever2_hidden_pre_loss']['eval_hidden_loss_before'])} -> "
        f"{_fmt(summary['lever2_hidden_pre_loss']['eval_hidden_loss_after'])} "
        f"(decreased={summary['lever2_hidden_pre_loss']['hidden_loss_decreased']})"
    )
    lines.append(
        f"- eval min_norm mean: {_fmt(summary['lever2_hidden_pre_loss']['eval_min_norm_before'],2)} -> "
        f"{_fmt(summary['lever2_hidden_pre_loss']['eval_min_norm_after'],2)} "
        f"(decreased={summary['lever2_hidden_pre_loss']['min_norm_decreased']})"
    )
    lines.append(
        f"- LR={_fmt(args.lr,4)} schedule={args.lr_schedule} grad_clip={_fmt(args.grad_clip,2)} "
        f"train_steps={args.train_steps} lr_floor={_fmt(args.lr_floor,6)}"
    )
    lines.append(
        f"- head: hidden={args.goal_head_hidden}, depth={args.goal_head_depth}, "
        f"mode={args.goal_head_mode}; injection_targets={args.goal_injection_targets}"
    )
    lines.append(f"- **converged (train hidden loss AND eval min_norm plateau): {plateau['plateau']}**")
    lines.append(f"  - train hidden loss: {plateau_train['plateau']} — {plateau_train['reason']}")
    lines.append(f"  - eval min_norm: {plateau_eval_min_norm['plateau']} — {plateau_eval_min_norm['reason']}")
    if eval_steps:
        traj = ", ".join(
            f"{s}:{_fmt(mn,2)}" for s, mn in zip(eval_steps, eval_min_norm_curve)
        )
        lines.append(f"- eval min_norm trajectory (step:min_norm): {traj}")
    lines.append(
        f"- injection: pre-temporal hook target(s) `{args.goal_injection_targets}`; "
        "`models.py` unchanged; legacy residual hook preserved."
    )
    lines.append("")
    lines.append("## GATE — context-window AR free-run")
    lines.append(f"- config: N={len(start_phases)}, horizon={args.gate_horizon}, conv_norm_thr={_fmt(args.conv_norm_thr,2)}, reach_gate={_fmt(args.reach_gate,2)}")
    lines.append("| turn target | N | reach_rate | reach_min_norm mean/min | no-goal baseline reach_rate (mean/min) | pop_safe | clip_resumable DEGEN |")
    lines.append("|---|---|---|---|---|---|---|")
    for clip in turn_clips:
        m = per_clip[clip]
        lines.append(
            f"| {clip} | {m['n']} | **{_fmt(m['reach_rate'],2)}** | "
            f"{_fmt(m['reach_min_norm_mean'],2)}/{_fmt(m['reach_min_norm_min'],2)} | "
            f"{_fmt(m['no_goal_baseline_reach_rate'],2)} "
            f"({_fmt(m['no_goal_baseline_min_norm_mean'],2)}/{_fmt(m['no_goal_baseline_min_norm_min'],2)}) | "
            f"{_fmt(m['pop_safe_rate'],2)} | {_fmt(m['clip_resumable_rate_DEGENERATE'],2)} |"
        )
    lines.append("")
    lines.append("## Binding decision")
    lines.append(f"- reach lifted above §4c 0.00 floor: **{gate.lifted_above_floor}**")
    lines.append(f"- Walk_L_To_R passes: {gate.l_to_r_pass}; all targets pass: {gate.all_pass}")
    lines.append(f"- raw reach STOP = {gate.stop}; upgrade_negative = {can_upgrade_negative}")
    lines.append(f"- **STOP = {gate.stop}** — {gate_reason}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(
        f"[LEVER1 all_pass={lever1_pass}] "
        f"[eval_hidden_loss {eval_before['hidden_loss_mean']:.4f}->{eval_after['hidden_loss_mean']:.4f}]"
    )
    print(
        "[GATE reach_rate] "
        + ", ".join(f"{c}={_fmt(reach_rates[c],2)}" for c in turn_clips)
        + f" | converged={converged} (train={plateau_train['plateau']},eval_min_norm={plateau_eval_min_norm['plateau']})"
        + f" | raw_STOP={gate.stop} | upgrade_negative={can_upgrade_negative}"
    )
    print(f"[gate] {gate_reason}")


if __name__ == "__main__":
    main()
