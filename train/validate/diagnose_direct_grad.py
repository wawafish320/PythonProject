#!/usr/bin/env python3
"""
Diagnose whether the direct pose head uses (plan/meas) by measuring input sensitivity.

We hook the first Linear layer of `EventMotionModel.direct_pose_head` and compute:
  - |dL/dx| mean per feature (x = direct_flat = [cond, plan, meas] concatenation)
  - (|dL/dx| * std(x)) as a scale-aware importance proxy

Typical use:
    python -m train.validate.diagnose_direct_grad \\
        --teacher validate/teacher_batches/Walk_F_teacher.json \\
        --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_meas_only.pth \\
        --steps 0,1,2,10,20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from train.geometry import geodesic_R, rot6d_to_matrix
from train.validate.run_freerun_cycles import (
    FreeRunCycleRunner,
    _build_full_cycle_sample,
    _load_json,
    _run_freerun_cycles,
    _resolve_npz_path,
)


def _parse_steps(spec: str, T: int) -> List[int]:
    spec = str(spec or "").strip().lower()
    if spec in ("", "all", "*"):
        return list(range(T))
    if spec in ("first10", "first_10"):
        return list(range(min(10, T)))
    out: List[int] = []
    for chunk in spec.split(","):
        s = chunk.strip()
        if not s:
            continue
        if "-" in s:
            a, b = s.split("-", 1)
            try:
                lo = int(a.strip())
                hi = int(b.strip())
            except Exception:
                continue
            if hi < lo:
                lo, hi = hi, lo
            out.extend(list(range(lo, hi + 1)))
        else:
            try:
                out.append(int(s))
            except Exception:
                continue
    # Keep order, dedup, clamp to range.
    seen = set()
    clamped: List[int] = []
    for t in out:
        if not (0 <= int(t) < int(T)):
            continue
        if int(t) in seen:
            continue
        seen.add(int(t))
        clamped.append(int(t))
    return clamped


def _infer_rot_slice(runner: FreeRunCycleRunner, Dy: int) -> slice:
    sl = getattr(getattr(runner, "trainer", None), "rot6d_y_slice", None) or getattr(getattr(runner, "trainer", None), "rot6d_slice", None)
    if isinstance(sl, slice):
        return sl
    return slice(0, int(Dy))


def _group_slices(model) -> Tuple[int, int, int, slice, slice, Optional[slice]]:
    Dc = int(getattr(model, "cond_dim", 0) or 0)
    C = int(getattr(model, "contact_dim", 0) or 0)
    mode = str(getattr(model, "direct_pose_meas_mode", "concat") or "concat").lower().strip()
    s_cond = slice(0, Dc)
    s_plan = slice(Dc, Dc + C) if C > 0 else slice(Dc, Dc)
    s_meas = None
    if mode == "concat" and C > 0:
        s_meas = slice(Dc + C, Dc + 2 * C)
    return Dc, C, int(Dc + C + (C if s_meas is not None else 0)), s_cond, s_plan, s_meas


def _summarize_groups(vec: torch.Tensor, s_cond: slice, s_plan: slice, s_meas: Optional[slice]) -> Dict[str, float]:
    def _mean(sl: slice) -> float:
        if sl.start is None or sl.stop is None or sl.stop <= sl.start:
            return float("nan")
        return float(vec[sl].mean().item())

    out = {
        "cond": _mean(s_cond),
        "plan": _mean(s_plan),
    }
    if s_meas is not None:
        out["meas"] = _mean(s_meas)
    return out


def _slice_dim(sl: Optional[slice]) -> int:
    if sl is None:
        return 0
    a = 0 if sl.start is None else int(sl.start)
    b = 0 if sl.stop is None else int(sl.stop)
    return max(0, int(b - a))


def _mean_or_nan(x: Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.mean().item())


def _estimate_group_jacobian_norms(
    *,
    y: Tensor,  # (N, Dy)
    x: Tensor,  # (N, Dx) leaf with requires_grad=True
    s_cond: slice,
    s_plan: slice,
    s_meas: Optional[slice],
    num_proj: int,
    seed: int,
) -> Dict[str, Dict[str, Tensor]]:
    """
    Estimate per-row Jacobian Frobenius norms via random projections.

    For each row i (independent sample), we estimate ||∂y_i/∂x_group||_F by:
        r ~ N(0, I), g = ∂<y, r>/∂x,  E[||g_group||^2] = ||J_group||_F^2

    Returns:
        {group: {"l2": (N,), "rms": (N,)}}
        where rms = l2 / sqrt(dim_group) (dimension-normalized).
    """
    if num_proj <= 0:
        raise ValueError(f"num_proj must be > 0, got {num_proj}.")
    if seed is not None:
        torch.manual_seed(int(seed))

    N = int(y.shape[0])
    device = y.device
    dtype = y.dtype

    sq: Dict[str, Tensor] = {}
    dim: Dict[str, int] = {}
    if _slice_dim(s_cond) > 0:
        sq["cond"] = torch.zeros((N,), device=device, dtype=dtype)
        dim["cond"] = _slice_dim(s_cond)
    if _slice_dim(s_plan) > 0:
        sq["plan"] = torch.zeros((N,), device=device, dtype=dtype)
        dim["plan"] = _slice_dim(s_plan)
    if s_meas is not None and _slice_dim(s_meas) > 0:
        sq["meas"] = torch.zeros((N,), device=device, dtype=dtype)
        dim["meas"] = _slice_dim(s_meas)

    if not sq:
        return {}

    for k in range(int(num_proj)):
        r = torch.randn_like(y)
        s = (y * r).sum()
        g_full = torch.autograd.grad(
            s,
            x,
            retain_graph=(k != int(num_proj) - 1),
            create_graph=False,
            allow_unused=False,
        )[0]
        if "cond" in sq:
            g = g_full[:, s_cond]
            sq["cond"] += (g * g).sum(dim=-1)
        if "plan" in sq:
            g = g_full[:, s_plan]
            sq["plan"] += (g * g).sum(dim=-1)
        if "meas" in sq and s_meas is not None:
            g = g_full[:, s_meas]
            sq["meas"] += (g * g).sum(dim=-1)

    out: Dict[str, Dict[str, Tensor]] = {}
    for name, v in sq.items():
        l2 = (v / float(num_proj)).clamp_min(0.0).sqrt()
        d = max(1, int(dim[name]))
        rms = l2 / float(np.sqrt(d))
        out[name] = {"l2": l2, "rms": rms}
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Gradient-based sensitivity check for direct head inputs (cond/plan/meas).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--teacher", type=str, required=True, help="Teacher JSON (validate/teacher_batches/*.json).")
    p.add_argument("--model", type=str, required=True, help="Checkpoint path (.pth).")
    p.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    p.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    p.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv.pt")
    p.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--context-len", type=int, default=16)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument(
        "--rollout",
        type=str,
        default="teacher",
        choices=("teacher", "freerun"),
        help="Which state distribution to probe (teacher inputs vs multi-cycle free-run).",
    )
    p.add_argument("--rounds", type=int, default=5, help="Only used when --rollout=freerun.")
    p.add_argument(
        "--time-index-mode",
        type=str,
        default="auto",
        choices=("auto", "none", "cycle", "global"),
        help="Only used when --rollout=freerun.",
    )
    p.add_argument("--so3_corr_apply", action="store_true", help="Only used when --rollout=freerun.")
    p.add_argument("--so3_corr_max_deg", type=float, default=20.0, help="Only used when --rollout=freerun.")
    p.add_argument("--lambda_fusion_apply", action="store_true", help="Only used when --rollout=freerun.")
    p.add_argument(
        "--steps",
        type=str,
        default="0,1,2,10,20",
        help="Comma list or ranges (e.g. '0,1,10-20') or 'all'/'first10'. Index within one cycle.",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="mse_rot6d",
        choices=("mse_rot6d", "mse_all"),
        help="Loss used to generate gradients.",
    )
    p.add_argument(
        "--jacobian-proj",
        type=int,
        default=8,
        help="Number of random projections used to estimate ||∂direct_out/∂x_group||_F (higher = lower variance).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used for Jacobian projections and plan shuffle perturbation.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path. Empty disables writing.",
    )
    args = p.parse_args()

    teacher_path = Path(args.teacher).expanduser().resolve()
    teacher_data = _load_json(teacher_path)
    clip_name = str(teacher_data.get("clip") or teacher_path.stem.replace("_teacher", ""))
    teacher_block = teacher_data.get("teacher")
    if not isinstance(teacher_block, dict):
        raise SystemExit(f"[FATAL] {teacher_path} missing 'teacher' payload.")
    state_arr = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
    cond_arr = np.asarray(teacher_block.get("cond"), dtype=np.float32)
    if state_arr.ndim != 2 or cond_arr.ndim != 2:
        raise SystemExit(f"[FATAL] {teacher_path} invalid teacher state/cond shapes.")
    T_base = int(state_arr.shape[0])

    npz_root = Path(args.npz_root).expanduser().resolve()
    npz_path = _resolve_npz_path(clip_name, teacher_data.get("source_json"), npz_root)

    runner_args = argparse.Namespace(
        model=str(Path(args.model).expanduser().resolve()),
        device=str(args.device),
        bundle=str(Path(args.bundle).expanduser()),
        pretrain_template=str(Path(args.pretrain_template).expanduser()),
        encoder_bundle=str(Path(args.encoder_bundle).expanduser()),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        context_len=int(args.context_len),
        depth=int(args.depth),
        so3_corr_apply=bool(args.so3_corr_apply),
        so3_corr_max_deg=float(args.so3_corr_max_deg),
        lambda_fusion_apply=bool(args.lambda_fusion_apply),
    )
    runner = FreeRunCycleRunner(runner_args)

    ds = runner._build_dataset(npz_path, seq_len=T_base)
    runner._ensure_model_ready(ds)
    if runner.model is None:
        raise SystemExit("[FATAL] runner.model is None after _ensure_model_ready.")

    clip = ds.clips[0]
    sample = _build_full_cycle_sample(ds, clip, seq_len=T_base)

    device = runner.device
    model = runner.model.to(device)
    model.eval()

    if model.direct_pose_head is None:
        raise SystemExit("[FATAL] model.direct_pose_head is None (checkpoint has no direct head).")

    if str(args.rollout).strip().lower() == "teacher":
        # Prepare inputs (B=1, T=T_base)
        state = sample["motion"].unsqueeze(0).to(device)  # (1,T,Dx)
        gt = sample["gt_motion"].unsqueeze(0).to(device)  # (1,T,Dy)
        cond = sample.get("cond_in")
        cond = cond.unsqueeze(0).to(device) if torch.is_tensor(cond) else None
        angvel = sample.get("angvel")
        angvel = angvel.unsqueeze(0).to(device) if torch.is_tensor(angvel) else None
        pose_hist = sample.get("pose_hist")
        pose_hist = pose_hist.unsqueeze(0).to(device) if torch.is_tensor(pose_hist) else None

        saved: Dict[str, torch.Tensor] = {}

        def _hook(m: torch.nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            x = inp[0]
            if x.requires_grad:
                x.retain_grad()
            saved["x"] = x

        h = model.direct_pose_head[0].register_forward_hook(_hook)
        try:
            with torch.enable_grad():
                # IMPORTANT: do NOT pass GT contacts into `contacts=` when the model has a learned
                # contact_meas_head. Passing contacts overrides contacts_meas (and removes logits),
                # making x.requires_grad=False and breaking dL/dx diagnostics.
                contacts = None
                try:
                    use_learned_meas = bool(getattr(model, "contact_meas_enable", False)) and getattr(model, "contact_meas_head", None) is not None
                except Exception:
                    use_learned_meas = False
                if not use_learned_meas:
                    c = sample.get("contacts")
                    contacts = c.unsqueeze(0).to(device) if torch.is_tensor(c) else None
                ret = model(
                    state, cond=cond, contacts=contacts, angvel=angvel, pose_history=pose_hist, plan_z=None, time_index=None
                )
                if not isinstance(ret, dict) or "out_direct" not in ret:
                    raise SystemExit("[FATAL] model forward did not return 'out_direct'.")
                out_direct = ret["out_direct"]
                if out_direct.ndim != 3:
                    raise SystemExit(f"[FATAL] out_direct has shape {tuple(out_direct.shape)} (expected (B,T,Dy)).")
                x = saved.get("x", None)
                if x is None:
                    raise SystemExit("[FATAL] Failed to capture direct head input via hook.")

                Dc, C, in_dim_exp, s_cond, s_plan, s_meas = _group_slices(model)
                in_dim = int(x.shape[-1])
                if in_dim != in_dim_exp:
                    print(f"[WARN] inferred in_dim={in_dim} differs from expected={in_dim_exp} (mode={model.direct_pose_meas_mode}).")

                rot_slice = _infer_rot_slice(runner, int(out_direct.shape[-1]))
                if args.loss == "mse_rot6d":
                    pred_all = out_direct[..., rot_slice]
                    gt_all = gt[..., rot_slice]
                else:
                    pred_all = out_direct
                    gt_all = gt

                steps = _parse_steps(args.steps, int(out_direct.shape[1]))

                # Scale proxy: std of x across time (flattened rows).
                x_std = x.detach().std(dim=0, unbiased=False).clamp_min(1e-8)
                B = int(out_direct.shape[0])
                T = int(out_direct.shape[1])
                N = int(B * T)

                def _compute(loss: torch.Tensor, *, retain_graph: bool, step_idx: Optional[int] = None) -> Dict[str, Any]:
                    model.zero_grad(set_to_none=True)
                    if x.grad is not None:
                        x.grad.zero_()
                    loss.backward(retain_graph=retain_graph)
                    if step_idx is None:
                        g = x.grad.detach().abs().mean(dim=0)  # (in_dim,)
                    else:
                        g_full = x.grad.detach().view(B, T, -1).abs()
                        g = g_full[:, int(step_idx)].mean(dim=0)  # (in_dim,)
                    imp = g * x_std
                    return {
                        "loss": float(loss.detach().item()),
                        "grad_abs_mean": _summarize_groups(g, s_cond, s_plan, s_meas),
                        "grad_abs_mean_vec": [float(v) for v in g.detach().cpu().tolist()],
                        "std_vec": [float(v) for v in x_std.detach().cpu().tolist()],
                        "grad_abs_mean_x_std": _summarize_groups(imp, s_cond, s_plan, s_meas),
                    }

                results: Dict[str, Any] = {
                    "rollout": "teacher",
                    "clip": clip_name,
                    "teacher": str(teacher_path),
                    "model": str(Path(args.model).expanduser().resolve()),
                    "device": str(device),
                    "T": int(out_direct.shape[1]),
                    "direct_pose_meas_mode": str(getattr(model, "direct_pose_meas_mode", "concat")),
                    "cond_dim": Dc,
                    "contact_dim": C,
                    "direct_in_dim": int(in_dim),
                    "loss_type": str(args.loss),
                    "rot_slice": {"start": int(rot_slice.start or 0), "stop": int(rot_slice.stop or pred_all.shape[-1])},
                    "steps": steps,
                    "jacobian_proj": int(args.jacobian_proj),
                    "seed": int(args.seed),
                }

                # Global (all steps) gradient
                loss_all = F.mse_loss(pred_all, gt_all)
                results["all_steps"] = _compute(loss_all, retain_graph=bool(steps))

                # Per-step gradients
                per_step: Dict[int, Any] = {}
                for i, t in enumerate(steps):
                    loss_t = F.mse_loss(pred_all[:, t], gt_all[:, t])
                    per_step[int(t)] = _compute(loss_t, retain_graph=(i != len(steps) - 1), step_idx=int(t))
                results["per_step"] = {str(k): v for k, v in per_step.items()}

                # ---- Model-intrinsic plan usage: Jacobian norms + perturbations ----
                try:
                    # Re-run direct head as a pure function y = f(x) with x as a leaf to get d(y)/d(x_group).
                    head = model.direct_pose_head
                    if head is None:
                        raise RuntimeError("model.direct_pose_head is None.")
                    x_leaf = x.detach().requires_grad_(True)  # (N, Dx)
                    y_full = head(x_leaf)
                    if y_full.ndim != 2 or int(y_full.shape[0]) != int(x_leaf.shape[0]):
                        raise RuntimeError(f"direct_pose_head(x) returned {tuple(y_full.shape)} for x={tuple(x_leaf.shape)}.")
                    # Use same target slice as loss for comparability.
                    y_tgt = y_full[:, rot_slice] if args.loss == "mse_rot6d" else y_full

                    jac = _estimate_group_jacobian_norms(
                        y=y_tgt,
                        x=x_leaf,
                        s_cond=s_cond,
                        s_plan=s_plan,
                        s_meas=s_meas,
                        num_proj=int(args.jacobian_proj),
                        seed=int(args.seed),
                    )

                    def _summarize_norms(vec: Tensor) -> Dict[str, float]:
                        per = vec.reshape(B, T)
                        out: Dict[str, float] = {"all_steps_mean": _mean_or_nan(vec)}
                        for t in steps:
                            out[f"step{int(t):03d}"] = _mean_or_nan(per[:, int(t)])
                        return out

                    jac_summary: Dict[str, Any] = {
                        "target": "rot6d" if args.loss == "mse_rot6d" else "all",
                        "num_proj": int(args.jacobian_proj),
                        "group_norm_l2": {},
                        "group_norm_rms": {},
                    }
                    for name, rec in jac.items():
                        jac_summary["group_norm_l2"][name] = _summarize_norms(rec["l2"])
                        jac_summary["group_norm_rms"][name] = _summarize_norms(rec["rms"])
                    results["direct_out_jacobian"] = jac_summary

                    # Simple plan perturbations at the direct head input: zero and time-shuffle.
                    y_base_full = y_full.detach()
                    y_base_tgt = y_tgt.detach()
                    pert: Dict[str, Any] = {"target": jac_summary["target"]}

                    def _try_geo_deg(y_other_full: Tensor) -> Optional[Tensor]:
                        try:
                            if rot_slice.stop is None or rot_slice.start is None:
                                return None
                            rot_len = int(rot_slice.stop - rot_slice.start)
                            if rot_len <= 0 or (rot_len % 6) != 0:
                                return None
                            J = int(rot_len // 6)
                            y1 = y_base_full[:, rot_slice].reshape(B, T, J, 6)
                            y2 = y_other_full[:, rot_slice].reshape(B, T, J, 6)
                            R1 = rot6d_to_matrix(y1)
                            R2 = rot6d_to_matrix(y2)
                            ang = geodesic_R(R1, R2)  # (B,T,J)
                            deg = ang * (180.0 / float(np.pi))
                            return deg.mean(dim=-1)  # (B,T) mean over joints
                        except Exception:
                            return None

                    # plan = 0
                    x_zero = x.detach().clone()
                    if _slice_dim(s_plan) > 0:
                        x_zero[:, s_plan] = 0.0
                    y_zero = head(x_zero)
                    y_zero_tgt = y_zero[:, rot_slice] if args.loss == "mse_rot6d" else y_zero
                    diff0 = (y_zero_tgt - y_base_tgt)
                    diff0_l2 = (diff0 * diff0).sum(dim=-1).clamp_min(0.0).sqrt()  # (N,)
                    pert["plan_zero_l2"] = _summarize_norms(diff0_l2)
                    deg0 = _try_geo_deg(y_zero.detach())
                    if deg0 is not None:
                        pert["plan_zero_geo_deg"] = _summarize_norms(deg0.reshape(N))

                    # plan = shuffled along time (within each batch element)
                    torch.manual_seed(int(args.seed))
                    x_shuf = x.detach().reshape(B, T, -1).clone()
                    if _slice_dim(s_plan) > 0:
                        for b in range(B):
                            perm = torch.randperm(T, device=x_shuf.device)
                            x_shuf[b, :, s_plan] = x_shuf[b, perm, s_plan]
                    x_shuf = x_shuf.reshape(N, -1)
                    y_shuf = head(x_shuf)
                    y_shuf_tgt = y_shuf[:, rot_slice] if args.loss == "mse_rot6d" else y_shuf
                    diffs = (y_shuf_tgt - y_base_tgt)
                    diffs_l2 = (diffs * diffs).sum(dim=-1).clamp_min(0.0).sqrt()
                    pert["plan_shuffle_l2"] = _summarize_norms(diffs_l2)
                    degs = _try_geo_deg(y_shuf.detach())
                    if degs is not None:
                        pert["plan_shuffle_geo_deg"] = _summarize_norms(degs.reshape(N))

                    results["direct_out_plan_perturb"] = pert
                except Exception as e:
                    results["direct_out_jacobian_error"] = str(e)
        finally:
            try:
                h.remove()
            except Exception:
                pass

        # Print a compact summary
        print(
            f"[DirectGrad] rollout=teacher clip={results['clip']} mode={results['direct_pose_meas_mode']} in_dim={results['direct_in_dim']} loss={results['loss_type']}"
        )
        g_all = results["all_steps"]["grad_abs_mean"]
        i_all = results["all_steps"]["grad_abs_mean_x_std"]
        print(f"  all_steps grad_abs_mean: {g_all}")
        print(f"  all_steps grad_abs_mean*std: {i_all}")
        for t in results["steps"]:
            rec = results["per_step"][str(t)]
            print(f"  step{int(t):03d} grad_abs_mean: {rec['grad_abs_mean']}  grad_abs_mean*std: {rec['grad_abs_mean_x_std']}")
        if "direct_out_jacobian" in results:
            j = results["direct_out_jacobian"]
            print(f"[DirectJacobian] target={j.get('target')} proj={j.get('num_proj')} seed={results.get('seed')}")
            for name in ("cond", "plan", "meas"):
                if name in j.get("group_norm_rms", {}):
                    print(f"  {name} rms(all)={j['group_norm_rms'][name].get('all_steps_mean'):.6g}")
        if "direct_out_plan_perturb" in results:
            p = results["direct_out_plan_perturb"]
            print(f"[DirectPlanPerturb] target={p.get('target')}")
            if "plan_zero_l2" in p:
                print(f"  plan_zero  l2(all)={p['plan_zero_l2'].get('all_steps_mean'):.6g}")
            if "plan_zero_geo_deg" in p:
                print(f"  plan_zero  geo_deg(all)={p['plan_zero_geo_deg'].get('all_steps_mean'):.6g}")
            if "plan_shuffle_l2" in p:
                print(f"  plan_shuffle l2(all)={p['plan_shuffle_l2'].get('all_steps_mean'):.6g}")
            if "plan_shuffle_geo_deg" in p:
                print(f"  plan_shuffle geo_deg(all)={p['plan_shuffle_geo_deg'].get('all_steps_mean'):.6g}")
    else:
        # ------------------------------------------------------------------
        #   Free-run: run multi-cycle rollout, capture direct_head input x(t)
        # ------------------------------------------------------------------
        if runner.trainer is None:
            raise SystemExit("[FATAL] runner.trainer is None after _ensure_model_ready.")
        trainer = runner.trainer
        trainer.model.eval()

        rounds = max(1, int(args.rounds))
        steps_in_cycle = _parse_steps(args.steps, int(T_base))
        cycle_len = int(T_base)
        total_steps = int(cycle_len * rounds - 1)

        x_steps_cpu: List[Tensor] = []
        capture_enabled = True

        def _hook_capture(m: torch.nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            nonlocal capture_enabled
            if not capture_enabled:
                return
            x = inp[0].detach()
            if x.ndim == 1:
                x = x.view(1, -1)
            if x.ndim != 2:
                return
            # Expect B==1 for this diagnostic.
            if int(x.shape[0]) != 1:
                raise RuntimeError(f"Expected direct head batch=1, got x.shape={tuple(x.shape)}")
            x_steps_cpu.append(x.cpu())

        h = model.direct_pose_head[0].register_forward_hook(_hook_capture)
        try:
            with torch.no_grad():
                _run_freerun_cycles(
                    trainer,
                    sample,
                    rounds=rounds,
                    device=device,
                    time_index_mode=str(args.time_index_mode),
                    lambda_fusion_apply=bool(args.lambda_fusion_apply),
                )
        finally:
            capture_enabled = False
            try:
                h.remove()
            except Exception:
                pass

        if not x_steps_cpu:
            raise SystemExit("[FATAL] No direct head inputs captured during free-run (is direct head disabled?).")
        x_all = torch.cat(x_steps_cpu, dim=0)  # (N_steps, Dx)
        N_steps = int(x_all.shape[0])
        if N_steps != total_steps:
            print(f"[WARN] captured N_steps={N_steps} but expected total_steps={total_steps} (rounds={rounds}, cycle_len={cycle_len}).")

        Dc, C, in_dim_exp, s_cond, s_plan, s_meas = _group_slices(model)
        in_dim = int(x_all.shape[-1])
        if in_dim != in_dim_exp:
            print(f"[WARN] inferred in_dim={in_dim} differs from expected={in_dim_exp} (mode={model.direct_pose_meas_mode}).")

        head = model.direct_pose_head
        if head is None:
            raise SystemExit("[FATAL] model.direct_pose_head is None unexpectedly.")
        rot_slice = _infer_rot_slice(runner, int(getattr(model, "out_motion_dim", 0) or 0) or int(sample["gt_motion"].shape[-1]))

        # Build global indices to probe: step-in-cycle across all rounds.
        points: List[Dict[str, int]] = []
        indices: List[int] = []
        for r in range(rounds):
            for s in steps_in_cycle:
                idx = int(r * cycle_len + int(s))
                if idx < 0 or idx >= N_steps:
                    continue
                points.append({"round": int(r), "step": int(s), "global_step": int(idx)})
                indices.append(int(idx))
        if not indices:
            raise SystemExit("[FATAL] No valid probe points selected (check --steps / --rounds / cycle length).")
        # Sort by global step for readability.
        order = np.argsort(np.asarray(indices, dtype=np.int64)).tolist()
        indices = [indices[i] for i in order]
        points = [points[i] for i in order]

        x_std_all = x_all.std(dim=0, unbiased=False).clamp_min(1e-8)
        x_points = x_all[indices]

        # Base outputs at points.
        with torch.no_grad():
            y_points_full = head(x_points.to(device)).detach().cpu()
        y_points_tgt = y_points_full[:, rot_slice] if args.loss == "mse_rot6d" else y_points_full

        # ---- Jacobian norms (direct_out wrt x_group) at points ----
        jac_summary: Dict[str, Any] = {"target": "rot6d" if args.loss == "mse_rot6d" else "all", "num_proj": int(args.jacobian_proj)}
        capture_enabled = False
        with torch.enable_grad():
            x_leaf = x_points.to(device).detach().requires_grad_(True)
            y_full = head(x_leaf)
            y_tgt = y_full[:, rot_slice] if args.loss == "mse_rot6d" else y_full
            jac = _estimate_group_jacobian_norms(
                y=y_tgt,
                x=x_leaf,
                s_cond=s_cond,
                s_plan=s_plan,
                s_meas=s_meas,
                num_proj=int(args.jacobian_proj),
                seed=int(args.seed),
            )
        jac_summary["group_norm_rms_mean"] = {k: float(v["rms"].mean().item()) for k, v in jac.items()}
        jac_summary["group_norm_l2_mean"] = {k: float(v["l2"].mean().item()) for k, v in jac.items()}

        # ---- Gradient of loss w.r.t x (head-only), averaged over probe points ----
        grad_summary: Optional[Dict[str, Any]] = None
        try:
            gt_full = sample["gt_motion"].unsqueeze(0).repeat(1, rounds, 1)  # (1, T_total, Dy)
            gt_points = gt_full[0, indices].to(device)
            gt_points_tgt = gt_points[:, rot_slice] if args.loss == "mse_rot6d" else gt_points
            with torch.enable_grad():
                x_leaf2 = x_points.to(device).detach().requires_grad_(True)
                y2 = head(x_leaf2)
                y2_tgt = y2[:, rot_slice] if args.loss == "mse_rot6d" else y2
                loss = F.mse_loss(y2_tgt, gt_points_tgt)
                loss.backward()
                g = x_leaf2.grad.detach().abs().mean(dim=0)  # (Dx,)
                grad_summary = {
                    "loss": float(loss.detach().item()),
                    "grad_abs_mean": _summarize_groups(g, s_cond, s_plan, s_meas),
                    "grad_abs_mean_x_std": _summarize_groups(g * x_std_all.to(device), s_cond, s_plan, s_meas),
                }
        except Exception as e:
            grad_summary = {"error": str(e)}

        # ---- Plan perturb at points: zero + time-shuffle (within each round) ----
        def _geo_deg_mean_over_joints(y_a: Tensor, y_b: Tensor) -> Optional[Tensor]:
            try:
                if rot_slice.stop is None or rot_slice.start is None:
                    return None
                rot_len = int(rot_slice.stop - rot_slice.start)
                if rot_len <= 0 or (rot_len % 6) != 0:
                    return None
                J = int(rot_len // 6)
                a6 = y_a[:, rot_slice].reshape(-1, J, 6).to(device)
                b6 = y_b[:, rot_slice].reshape(-1, J, 6).to(device)
                R1 = rot6d_to_matrix(a6)
                R2 = rot6d_to_matrix(b6)
                ang = geodesic_R(R1, R2)  # (N,J)
                deg = ang * (180.0 / float(np.pi))
                return deg.mean(dim=-1).detach().cpu()  # (N,)
            except Exception:
                return None

        # plan=0
        x_zero = x_points.clone()
        if _slice_dim(s_plan) > 0:
            x_zero[:, s_plan] = 0.0
        with torch.no_grad():
            y_zero_full = head(x_zero.to(device)).detach().cpu()
        y_zero_tgt = y_zero_full[:, rot_slice] if args.loss == "mse_rot6d" else y_zero_full
        diff0 = (y_zero_tgt - y_points_tgt)
        diff0_l2 = (diff0 * diff0).sum(dim=-1).clamp_min(0.0).sqrt()
        diff0_geo = _geo_deg_mean_over_joints(y_points_full, y_zero_full)

        # plan=time-shuffle within each round (use full x_all so shuffle has enough support)
        torch.manual_seed(int(args.seed))
        x_shuf_all = x_all.clone()
        if _slice_dim(s_plan) > 0:
            x_src = x_all.clone()
            for r in range(rounds):
                s0 = int(r * cycle_len)
                s1 = min(int((r + 1) * cycle_len), int(x_all.shape[0]))
                if s1 <= s0 + 1:
                    continue
                perm = torch.randperm(s1 - s0)
                x_shuf_all[s0:s1, s_plan] = x_src[s0:s1, s_plan][perm]
        x_shuf = x_shuf_all[indices]
        with torch.no_grad():
            y_shuf_full = head(x_shuf.to(device)).detach().cpu()
        y_shuf_tgt = y_shuf_full[:, rot_slice] if args.loss == "mse_rot6d" else y_shuf_full
        diffs = (y_shuf_tgt - y_points_tgt)
        diffs_l2 = (diffs * diffs).sum(dim=-1).clamp_min(0.0).sqrt()
        diffs_geo = _geo_deg_mean_over_joints(y_points_full, y_shuf_full)

        results = {
            "rollout": "freerun",
            "clip": clip_name,
            "teacher": str(teacher_path),
            "model": str(Path(args.model).expanduser().resolve()),
            "device": str(device),
            "cycle_len": int(cycle_len),
            "rounds": int(rounds),
            "free_steps": int(N_steps),
            "time_index_mode": str(args.time_index_mode),
            "so3_corr_apply": bool(args.so3_corr_apply),
            "so3_corr_max_deg": float(args.so3_corr_max_deg),
            "lambda_fusion_apply": bool(args.lambda_fusion_apply),
            "direct_pose_meas_mode": str(getattr(model, "direct_pose_meas_mode", "concat")),
            "cond_dim": Dc,
            "contact_dim": C,
            "direct_in_dim": int(in_dim),
            "direct_in_std_mean": _summarize_groups(x_std_all, s_cond, s_plan, s_meas),
            "loss_type": str(args.loss),
            "rot_slice": {"start": int(rot_slice.start or 0), "stop": int(rot_slice.stop or 0)},
            "steps": steps_in_cycle,
            "points": points,
            "jacobian_proj": int(args.jacobian_proj),
            "seed": int(args.seed),
            "direct_out_jacobian": jac_summary,
            "direct_out_loss_grad": grad_summary,
            "direct_out_plan_perturb": {
                "target": jac_summary.get("target"),
                "plan_zero_l2_mean": float(diff0_l2.mean().item()),
                "plan_zero_geo_deg_mean": float(diff0_geo.mean().item()) if diff0_geo is not None else None,
                "plan_shuffle_l2_mean": float(diffs_l2.mean().item()),
                "plan_shuffle_geo_deg_mean": float(diffs_geo.mean().item()) if diffs_geo is not None else None,
            },
        }

        # Print compact summary (freerun)
        print(
            f"[DirectGrad] rollout=freerun clip={results['clip']} rounds={results['rounds']} "
            f"mode={results['direct_pose_meas_mode']} in_dim={results['direct_in_dim']} loss={results['loss_type']}"
        )
        print(f"[DirectInStd] mean={results.get('direct_in_std_mean')}")
        print(f"[DirectJacobian] mean_rms={results['direct_out_jacobian'].get('group_norm_rms_mean')}")
        print(f"[DirectPlanPerturb] mean plan_zero_geo_deg={results['direct_out_plan_perturb'].get('plan_zero_geo_deg_mean')}")
        print(f"[DirectPlanPerturb] mean plan_shuffle_geo_deg={results['direct_out_plan_perturb'].get('plan_shuffle_geo_deg_mean')}")
        if isinstance(results.get("direct_out_loss_grad"), dict) and "grad_abs_mean_x_std" in results["direct_out_loss_grad"]:
            print(f"[DirectLossGrad] grad_abs_mean*std={results['direct_out_loss_grad']['grad_abs_mean_x_std']}")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
