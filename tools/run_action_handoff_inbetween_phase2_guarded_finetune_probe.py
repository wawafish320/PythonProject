#!/usr/bin/env python3
"""PHASE 2 guarded base fine-tune for action-handoff in-betweening §7.3.

Scope: first guarded base-weight update after PHASE 1 eval-min_norm ceiling. The probe keeps
goal head + hidden_pre reach loss, adds a small base LR with L2-to-init, and reports both the
real §6 arbitrary-phase AR reach gate and a Walk_F no-goal freerun drift guard.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train.validate.run_freerun_cycles as freerun  # noqa: E402
from train.action_handoff_inbetween_cond_probe import rollout_to_egocentric, select_start_phases  # noqa: E402
from train.action_handoff_inbetween_goal_injection import (  # noqa: E402
    DEFAULT_REACH_RATE_GATE,
    GoalHead,
    calibration_records_all_pass,
    calibration_relerr,
    hidden_pre_anchor_loss,
    l_reach,
    loss_plateau_status,
    reach_gate_decision,
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
from tools.run_action_handoff_inbetween_reach_aware_rewire_probe import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_CKPT,
    DEFAULT_ENCODER_BUNDLE,
    DEFAULT_NPZ_ROOT,
    DEFAULT_PRETRAIN_TEMPLATE,
    DEFAULT_Z_FEATURES,
    FPS,
    _capture_context_teacher_hidden,
    _capture_fullseq_hidden,
    _dump_json,
    _dump_md,
    _downsample,
    _eval_hidden_reach_snapshot,
    _fmt,
    _goal_flat,
    _run_context_ar_rollout,
    _runner_args,
)


def _select_trainable_base_params(model, policy: str) -> List[Tuple[str, torch.nn.Parameter]]:
    policy_l = str(policy or "tail").lower().strip()
    selected: List[Tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        ok = False
        if policy_l == "tail":
            ok = (
                name.startswith("shared_encoder.2")
                or name.startswith("shared_encoder.3")
                or name.startswith("shared_encoder.4")
                or name.startswith("shared_encoder.5")
                or name.startswith("shared_encoder.6")
                or name.startswith("shared_encoder.7")
                or name.startswith("shared_encoder.8")
                or name.startswith("residual_proj.")
            )
        elif policy_l == "residual_only":
            ok = name.startswith("residual_proj.")
        elif policy_l == "pasa_tail":
            ok = (
                name.startswith("shared_encoder.2")
                or name.startswith("shared_encoder.3")
                or name.startswith("shared_encoder.4")
                or name.startswith("shared_encoder.5")
                or name.startswith("shared_encoder.6")
                or name.startswith("shared_encoder.7")
                or name.startswith("shared_encoder.8")
                or name.startswith("residual_proj.")
                or name.startswith("_pasa_")
                or name.startswith("coupling_norm.")
            )
        else:
            raise ValueError(f"unknown unfreeze policy: {policy!r}")
        if ok:
            p.requires_grad_(True)
            selected.append((name, p))
        else:
            p.requires_grad_(False)
    if not selected:
        raise RuntimeError(f"unfreeze policy selected no params: {policy!r}")
    return selected


def _snapshot_params(params: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in params}


def _l2_to_init(params: List[Tuple[str, torch.nn.Parameter]], init: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    count = 0
    for name, p in params:
        diff = p - init[name].to(device=p.device, dtype=p.dtype)
        val = diff.pow(2).mean()
        total = val if total is None else total + val
        count += 1
    if total is None:
        raise RuntimeError("cannot compute L2-to-init for empty param list")
    return total / max(1, count)


def _l2_to_init_value(params: List[Tuple[str, torch.nn.Parameter]], init: Dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        return float(_l2_to_init(params, init).detach().cpu())


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _module_state_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _selected_state_cpu(params: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in params}


def _anchor_metadata(anchors: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for clip, anchor in anchors.items():
        out[str(clip)] = {
            "centroid_shape": list(np.asarray(anchor.centroid).shape),
            "radius": float(anchor.radius),
            "clip_spread": float(anchor.clip_spread),
            "diffuseness": float(anchor.diffuseness),
            "end_window_k": int(anchor.end_window_k),
            "well_defined": bool(anchor.well_defined),
            "radius_degenerate": bool(anchor.radius_degenerate),
        }
    return out


def _git_metadata() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> Tuple[int, str, str]:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return int(proc.returncode), proc.stdout.strip(), proc.stderr.strip()

    rc_commit, commit, err_commit = _run(["git", "rev-parse", "HEAD"])
    rc_branch, branch, err_branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    rc_status, status, err_status = _run(["git", "status", "--short"])
    dirty = bool(status.strip()) if rc_status == 0 else None
    return {
        "commit": commit if rc_commit == 0 else None,
        "commit_error": err_commit if rc_commit != 0 else None,
        "branch": branch if rc_branch == 0 else None,
        "branch_error": err_branch if rc_branch != 0 else None,
        "dirty": dirty,
        "status_short": status.splitlines() if rc_status == 0 else [],
        "status_error": err_status if rc_status != 0 else None,
        "worktree_note": "dirty worktree" if dirty else ("clean worktree" if dirty is False else "git status unavailable"),
    }


def _save_phase2_trained_state(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
    goal_head: torch.nn.Module,
    base_params: List[Tuple[str, torch.nn.Parameter]],
    anchors: Dict[str, Any],
    start_phases: List[int],
    goal_flat_dim: int,
) -> Dict[str, Any]:
    state_path = out_dir / "phase2_trained_state.pt"
    manifest_path = out_dir / "phase2_trained_state_manifest.json"
    trainable_names = [name for name, _ in base_params]
    payload = {
        "kind": "action_handoff_inbetween_phase2_trained_state",
        "format_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": _jsonable_args(args),
        "seed": int(args.seed),
        "base_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "goal_head_config": {
            "goal_flat_dim": int(goal_flat_dim),
            "hidden": int(args.goal_head_hidden),
            "depth": int(args.goal_head_depth),
            "mode": str(args.goal_head_mode),
            "injection_targets": str(args.goal_injection_targets),
            "init_scale": 0.0,
        },
        "trainable_base": {
            "policy": str(args.unfreeze_policy),
            "names": trainable_names,
            "num_tensors": int(len(trainable_names)),
            "num_params": int(sum(p.numel() for _, p in base_params)),
            "state_dict_subset": _selected_state_cpu(base_params),
        },
        "model_state_dict": _module_state_cpu(model),
        "goal_head_state_dict": _module_state_cpu(goal_head),
        "anchor_metadata": _anchor_metadata(anchors),
        "start_phases": [int(v) for v in start_phases],
        "git": _git_metadata(),
        "note": (
            "PHASE2 instrumentation-only artifact. Training logic and gate contract are unchanged; "
            "this file captures the trained base model and goal_head for exact replay probes."
        ),
    }
    torch.save(payload, state_path)
    manifest = {
        "kind": payload["kind"],
        "format_version": payload["format_version"],
        "created_at": payload["created_at"],
        "state_path": str(state_path.resolve()),
        "base_checkpoint": payload["base_checkpoint"],
        "seed": payload["seed"],
        "goal_head_config": payload["goal_head_config"],
        "trainable_base": {k: v for k, v in payload["trainable_base"].items() if k != "state_dict_subset"},
        "anchor_metadata": payload["anchor_metadata"],
        "start_phases": payload["start_phases"],
        "git": payload["git"],
        "note": payload["note"],
    }
    _dump_json(manifest_path, manifest)
    return {
        "state_path": str(state_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "kind": payload["kind"],
        "format_version": payload["format_version"],
        "model_state_tensors": int(len(payload["model_state_dict"])),
        "goal_head_state_tensors": int(len(payload["goal_head_state_dict"])),
        "trainable_base_num_tensors": int(len(trainable_names)),
        "git": payload["git"],
    }


def _walk_f_drift_eval(
    runner,
    walk_sample: Dict[str, torch.Tensor],
    target_states: Dict[str, np.ndarray],
    std: np.ndarray,
    *,
    phases: List[int],
    horizon: int,
    context_len: int,
) -> Dict[str, float]:
    walk_state = target_states[WALK_F]
    thr = GateThresholds()
    pose_ds: List[float] = []
    pop_vals: List[float] = []
    root_speed_vals: List[float] = []
    for phase in phases:
        roll = _run_context_ar_rollout(
            runner,
            walk_sample,
            phase=int(phase),
            horizon=int(horizon),
            context_len=int(context_len),
            delta=None,
            capture_grad=False,
        )
        out_np = roll.out_raw.detach().cpu().numpy()
        gen_rot6d = out_np[:, :276].reshape(out_np.shape[0], 46, 6)
        gen_rv = out_np[:, 276:278]
        cond_dir_np = roll.cond_dir_raw.detach().cpu().numpy()
        contact_np = roll.contacts.detach().cpu().numpy()
        m = min(gen_rot6d.shape[0], cond_dir_np.shape[0], contact_np.shape[0])
        roll_raw = rollout_to_egocentric(gen_rot6d[:m], gen_rv[:m], cond_dir_np[:m], contact_np[:m], fps=FPS)
        idx = (int(phase) + np.arange(m)) % int(walk_state.shape[0])
        ref = walk_state[idx]
        out = evaluate_rollout_state_space(roll_raw, ref[-SEAM_LEN_K:], std, thr)
        pose_ds.append(float(out["best_pose_d"]))
        pop_vals.append(float(out["pop"]))
        root_speed_vals.append(float(np.linalg.norm(gen_rv[:m], axis=1).mean()))
    return {
        "n": int(len(phases)),
        "best_pose_d_mean": float(np.mean(pose_ds)),
        "best_pose_d_max": float(np.max(pose_ds)),
        "pop_mean": float(np.mean(pop_vals)),
        "pop_max": float(np.max(pop_vals)),
        "root_speed_mean": float(np.mean(root_speed_vals)),
    }


def _drift_guard(before: Dict[str, float], after: Dict[str, float], *, rel_tol: float, abs_tol: float) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, float | bool]] = {}
    ok = True
    for key in ("best_pose_d_mean", "pop_mean", "root_speed_mean"):
        b = float(before[key])
        a = float(after[key])
        abs_delta = float(a - b)
        rel_delta = float(abs_delta / max(abs(b), 1e-8))
        passed = bool(abs_delta <= float(abs_tol) or rel_delta <= float(rel_tol))
        checks[key] = {"before": b, "after": a, "abs_delta": abs_delta, "rel_delta": rel_delta, "passed": passed}
        ok = bool(ok and passed)
    return {"passed": ok, "rel_tol": float(rel_tol), "abs_tol": float(abs_tol), "checks": checks}


def _phase2_decision(gate: Any, *, drift_passed: bool) -> str:
    if bool(gate.lifted_above_floor) and not bool(drift_passed):
        return "failure_reach_lifted_but_walk_f_drifted"
    if bool(gate.all_pass) and bool(drift_passed):
        return "success_all_targets_pass_walk_f_stable"
    if bool(gate.l_to_r_pass) and bool(drift_passed):
        return "partial_success_l_r_passes_walk_f_stable_all_targets_not_yet"
    if bool(gate.lifted_above_floor) and bool(drift_passed):
        return "partial_reach_lifted_walk_f_stable_l_r_not_yet"
    return "reconsider_reach_not_lifted"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PHASE 2 guarded base fine-tune probe.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    p.add_argument("--bundle", type=str, default=DEFAULT_BUNDLE)
    p.add_argument("--pretrain-template", type=str, default=DEFAULT_PRETRAIN_TEMPLATE)
    p.add_argument("--encoder-bundle", type=str, default=DEFAULT_ENCODER_BUNDLE)
    p.add_argument("--npz-root", type=str, default=DEFAULT_NPZ_ROOT)
    p.add_argument("--z-features", type=str, default=DEFAULT_Z_FEATURES)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--n-starts", type=int, default=20)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--train-horizon", type=int, default=24)
    p.add_argument("--gate-horizon", type=int, default=72)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--lr-base", type=float, default=2e-5)
    p.add_argument("--lr-schedule", type=str, default="step", choices=["none", "step"])
    p.add_argument("--lr-step-size", type=int, default=250)
    p.add_argument("--lr-step-gamma", type=float, default=0.5)
    p.add_argument("--lr-floor-head", type=float, default=1e-4)
    p.add_argument("--lr-floor-base", type=float, default=5e-6)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--base-l2-weight", type=float, default=10.0)
    p.add_argument("--unfreeze-policy", type=str, default="tail", choices=["tail", "residual_only", "pasa_tail"])
    p.add_argument("--goal-head-hidden", type=int, default=512)
    p.add_argument("--goal-head-depth", type=int, default=2)
    p.add_argument("--goal-head-mode", type=str, default="additive", choices=["additive", "film"])
    p.add_argument("--goal-injection-targets", type=str, default="shared_encoder.0,shared_encoder.1")
    p.add_argument("--hidden-loss-weight", type=float, default=1.0)
    p.add_argument("--output-loss-weight", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--plateau-window", type=int, default=80)
    p.add_argument("--plateau-rel-delta", type=float, default=0.02)
    p.add_argument("--eval-plateau-window", type=int, default=3)
    p.add_argument("--eval-plateau-min-samples", type=int, default=5)
    p.add_argument("--goal-horizon", type=int, default=12)
    p.add_argument("--conv-norm-thr", type=float, default=DEFAULT_CONV_NORM_THR)
    p.add_argument("--end-window-k", type=int, default=DEFAULT_END_WINDOW_K)
    p.add_argument("--reach-gate", type=float, default=DEFAULT_REACH_RATE_GATE)
    p.add_argument("--drift-rel-tol", type=float, default=0.20)
    p.add_argument("--drift-abs-tol", type=float, default=0.03)
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
        f"debug_output/_tmp_action_handoff_inbetween_phase2_guarded_finetune_{date_tag}"
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

    # Calibration remains the prerequisite for hidden_pre reach.
    step1_records: Dict[str, Dict[str, Any]] = {}
    for clip in turn_clips:
        T = int(saved_hidden[clip].shape[0])
        ds = runner._build_dataset(npz_root / f"{clip}.npz", seq_len=T)
        runner._ensure_model_ready(ds)
        sample = freerun._build_full_cycle_sample(ds, ds.clips[0], seq_len=T)
        fullseq = _capture_fullseq_hidden(model, sample)
        context = _capture_context_teacher_hidden(model, sample, context_len=int(args.context_len))
        step1_records[clip] = {
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
    lever1_pass = calibration_records_all_pass(step1_records, conv_norm_thr=float(args.conv_norm_thr))
    runner._ensure_model_ready(walk_ds)
    model = runner.model
    model.eval()
    if not lever1_pass:
        raise RuntimeError("context-window per-step hidden_pre calibration failed; cannot run PHASE 2.")

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

    drift_before = _walk_f_drift_eval(
        runner,
        walk_sample,
        target_states,
        std,
        phases=start_phases,
        horizon=int(args.gate_horizon),
        context_len=int(args.context_len),
    )

    goal_head = GoalHead.build(
        goal_flat_dim=K * goal_seam[turn_clips[0]].shape[1],
        hidden=int(args.goal_head_hidden),
        depth=int(args.goal_head_depth),
        init_scale=0.0,
        mode=str(args.goal_head_mode),
    ).to(runner.device)
    for p in goal_head.parameters():
        p.requires_grad_(True)
    base_params = _select_trainable_base_params(model, str(args.unfreeze_policy))
    base_init = _snapshot_params(base_params)
    opt = torch.optim.Adam(
        [
            {"params": list(goal_head.parameters()), "lr": float(args.lr_head)},
            {"params": [p for _, p in base_params], "lr": float(args.lr_base)},
        ]
    )
    scheduler = None
    if str(args.lr_schedule) == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, int(args.lr_step_size)), gamma=float(args.lr_step_gamma))

    eval_steps: List[int] = []
    eval_hidden_curve: List[float] = []
    eval_min_norm_curve: List[float] = []
    eval_lr_head_curve: List[float] = []
    eval_lr_base_curve: List[float] = []

    def _record_eval(step_idx: int) -> Dict[str, Any]:
        was_training = model.training
        model.eval()
        goal_head.eval()
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
        if was_training:
            model.train()
            goal_head.train()
        eval_steps.append(int(step_idx))
        eval_hidden_curve.append(float(snap["hidden_loss_mean"]))
        eval_min_norm_curve.append(float(snap["min_norm_mean"]))
        eval_lr_head_curve.append(float(opt.param_groups[0]["lr"]))
        eval_lr_base_curve.append(float(opt.param_groups[1]["lr"]))
        return snap

    eval_before = _record_eval(0)

    model.train()
    goal_head.train()
    loss_curve: List[float] = []
    hidden_curve: List[float] = []
    output_curve: List[float] = []
    l2_curve: List[float] = []
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
            loss_out = l_reach(roll.out_raw, goal_seam[clip], std, roll.cond_dir_raw)
        else:
            loss_out = roll.hidden_pre.new_zeros(())
        loss_l2 = _l2_to_init(base_params, base_init)
        loss = (
            float(args.hidden_loss_weight) * loss_h
            + float(args.output_loss_weight) * loss_out
            + float(args.base_l2_weight) * loss_l2
        )
        opt.zero_grad()
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(list(goal_head.parameters()) + [p for _, p in base_params], float(args.grad_clip))
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if float(args.lr_floor_head) > 0:
            opt.param_groups[0]["lr"] = max(float(opt.param_groups[0]["lr"]), float(args.lr_floor_head))
        if float(args.lr_floor_base) > 0:
            opt.param_groups[1]["lr"] = max(float(opt.param_groups[1]["lr"]), float(args.lr_floor_base))
        loss_curve.append(float(loss.detach().cpu()))
        hidden_curve.append(float(loss_h.detach().cpu()))
        output_curve.append(float(loss_out.detach().cpu()))
        l2_curve.append(float(loss_l2.detach().cpu()))
        if eval_every > 0 and (step + 1) % eval_every == 0 and (step + 1) < int(args.train_steps):
            _record_eval(step + 1)

    eval_after = _record_eval(int(args.train_steps))
    model.eval()
    goal_head.eval()
    state_artifact = _save_phase2_trained_state(
        out_dir=out_dir,
        args=args,
        model=model,
        goal_head=goal_head,
        base_params=base_params,
        anchors=anchors,
        start_phases=start_phases,
        goal_flat_dim=K * goal_seam[turn_clips[0]].shape[1],
    )

    drift_after = _walk_f_drift_eval(
        runner,
        walk_sample,
        target_states,
        std,
        phases=start_phases,
        horizon=int(args.gate_horizon),
        context_len=int(args.context_len),
    )
    drift = _drift_guard(drift_before, drift_after, rel_tol=float(args.drift_rel_tol), abs_tol=float(args.drift_abs_tol))

    # Real §6 AR gate: arbitrary Walk_F phases, per-step context-window AR, per-clip reach.
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
    plateau_eval_min_norm = loss_plateau_status(
        eval_min_norm_curve,
        min_steps=int(args.eval_plateau_min_samples),
        window=int(args.eval_plateau_window),
        rel_delta=float(args.plateau_rel_delta),
    )
    decision = _phase2_decision(gate, drift_passed=bool(drift["passed"]))

    summary = {
        "task": "Action-handoff in-betweening §7.3 PHASE 2 guarded base fine-tune",
        "stage": "phase2-guarded-base-finetune",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "reach_space": "hidden_pre(512), fullseq anchors, arbitrary-phase context-window AR capture",
        "guarded": True,
        "lever1_per_step_calibration": {"all_pass": lever1_pass, "per_clip": step1_records},
        "trainable_base": {
            "policy": str(args.unfreeze_policy),
            "num_tensors": len(base_params),
            "num_params": int(sum(p.numel() for _, p in base_params)),
            "names": [n for n, _ in base_params],
            "l2_to_init_final": _l2_to_init_value(base_params, base_init),
        },
        "config": {
            "context_len": int(args.context_len),
            "n_starts": len(start_phases),
            "train_steps": int(args.train_steps),
            "train_horizon": int(args.train_horizon),
            "gate_horizon": int(args.gate_horizon),
            "lr_head": float(args.lr_head),
            "lr_base": float(args.lr_base),
            "lr_schedule": str(args.lr_schedule),
            "lr_floor_head": float(args.lr_floor_head),
            "lr_floor_base": float(args.lr_floor_base),
            "grad_clip": float(args.grad_clip),
            "base_l2_weight": float(args.base_l2_weight),
            "goal_head_hidden": int(args.goal_head_hidden),
            "goal_head_depth": int(args.goal_head_depth),
            "goal_head_mode": str(args.goal_head_mode),
            "goal_injection_targets": str(args.goal_injection_targets),
            "thresholds_provisional": {
                "conv_norm_thr": float(args.conv_norm_thr),
                "reach_gate": float(args.reach_gate),
                "end_window_k": int(args.end_window_k),
                "tau_pose": thr.tau_pose,
                "tau_pop": thr.tau_pop,
            },
        },
        "training": {
            "eval_hidden_loss_before": eval_before["hidden_loss_mean"],
            "eval_hidden_loss_after": eval_after["hidden_loss_mean"],
            "eval_min_norm_before": eval_before["min_norm_mean"],
            "eval_min_norm_after": eval_after["min_norm_mean"],
            "eval_by_clip_before": eval_before["by_clip"],
            "eval_by_clip_after": eval_after["by_clip"],
            "eval_min_norm_plateau_status": plateau_eval_min_norm,
            "convergence_trajectory": {
                "eval_steps": eval_steps,
                "eval_hidden_loss": [float(v) for v in eval_hidden_curve],
                "eval_min_norm": [float(v) for v in eval_min_norm_curve],
                "lr_head_at_eval": [float(v) for v in eval_lr_head_curve],
                "lr_base_at_eval": [float(v) for v in eval_lr_base_curve],
                "train_hidden_loss_downsampled": _downsample(hidden_curve, 60),
                "train_total_loss_downsampled": _downsample(loss_curve, 60),
                "base_l2_downsampled": _downsample(l2_curve, 60),
            },
            "note": "hidden_pre loss tensors are graph-preserving torch.float32 on selected device; metrics are detached for diagnostics.",
        },
        "walk_f_drift_guard": {"before": drift_before, "after": drift_after, "decision": drift},
        "state_artifact": state_artifact,
        "section6_ar_gate": {
            "description": "arbitrary Walk_F phase context-window AR free rollout with per-step calibrated hidden_pre reach",
            "per_clip": per_clip,
            "walk_l_to_r_row": "Walk_L_To_R",
            "binding_gate_decision": {
                "per_clip_pass": gate.per_clip_pass,
                "l_to_r_pass": gate.l_to_r_pass,
                "all_pass": gate.all_pass,
                "lifted_above_floor": gate.lifted_above_floor,
                "raw_reach_stop": bool(gate.stop),
                "STOP": bool(gate.stop),
                "reason": gate.reason,
            },
        },
        "phase2_outcome": {
            "decision": decision,
            "walk_f_stable": bool(drift["passed"]),
            "l_to_r_pass": bool(gate.l_to_r_pass),
            "all_targets_pass": bool(gate.all_pass),
            "reach_lifted_above_floor": bool(gate.lifted_above_floor),
            "raw_reach_stop": bool(gate.stop),
            "note": "L_R partial pass is not all-target success; all-target success requires all_pass=True.",
        },
        "phase2_decision": decision,
    }
    json_path = out_dir / "phase2_guarded_finetune_summary.json"
    md_path = out_dir / "phase2_guarded_finetune_summary.md"
    _dump_json(json_path, summary)

    lines: List[str] = []
    lines.append("# PHASE 2 Guarded Base Fine-Tune — §7.3")
    lines.append("")
    lines.append(f"- decision: **{decision}**")
    lines.append("- semantics: L_R partial pass is reported separately from all-target success")
    lines.append(f"- trainable policy: `{args.unfreeze_policy}`, tensors={len(base_params)}, params={summary['trainable_base']['num_params']}")
    lines.append(f"- L2-to-init final: {_fmt(summary['trainable_base']['l2_to_init_final'], 8)}")
    lines.append("")
    lines.append("## Training")
    lines.append(f"- eval hidden loss: {_fmt(eval_before['hidden_loss_mean'])} -> {_fmt(eval_after['hidden_loss_mean'])}")
    lines.append(f"- eval min_norm mean: {_fmt(eval_before['min_norm_mean'],2)} -> {_fmt(eval_after['min_norm_mean'],2)}")
    lines.append(f"- eval min_norm plateau: {plateau_eval_min_norm['plateau']} ({plateau_eval_min_norm['reason']})")
    if eval_steps:
        lines.append("- eval min_norm trajectory: " + ", ".join(f"{s}:{_fmt(v,2)}" for s, v in zip(eval_steps, eval_min_norm_curve)))
    lines.append("")
    lines.append("## Walk_F Drift Guard")
    lines.append(f"- passed: **{drift['passed']}**")
    for key, row in drift["checks"].items():
        lines.append(
            f"- {key}: {_fmt(row['before'],4)} -> {_fmt(row['after'],4)} "
            f"(abs_delta={_fmt(row['abs_delta'],4)}, rel_delta={_fmt(row['rel_delta'],4)}, passed={row['passed']})"
        )
    lines.append("")
    lines.append("## §6 AR Gate")
    lines.append("| turn target | N | reach_rate | reach_min_norm mean/min | no-goal baseline mean/min | pop_safe |")
    lines.append("|---|---|---|---|---|---|")
    for clip in turn_clips:
        m = per_clip[clip]
        lines.append(
            f"| {clip} | {m['n']} | **{_fmt(m['reach_rate'],2)}** | "
            f"{_fmt(m['reach_min_norm_mean'],2)}/{_fmt(m['reach_min_norm_min'],2)} | "
            f"{_fmt(m['no_goal_baseline_min_norm_mean'],2)}/{_fmt(m['no_goal_baseline_min_norm_min'],2)} | "
            f"{_fmt(m['pop_safe_rate'],2)} |"
        )
    lines.append("")
    lines.append(f"- reach lifted above floor: **{gate.lifted_above_floor}**")
    lines.append(f"- Walk_L_To_R passes: {gate.l_to_r_pass}; all targets pass: {gate.all_pass}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- {json_path.resolve()}")
    _dump_md(md_path, lines)

    print(f"[ok] wrote: {json_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[PHASE2] {decision} | reach_lifted={gate.lifted_above_floor} | drift_pass={drift['passed']}")


if __name__ == "__main__":
    main()
