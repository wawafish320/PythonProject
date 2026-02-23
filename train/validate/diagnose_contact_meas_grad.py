#!/usr/bin/env python3
"""
Diagnose gradient "hijacking" of contact_meas_head by comparing per-loss gradients.

We measure gradient norms on `EventMotionModel.contact_meas_head.*` coming from:
  1) contact_meas_bce: BCEWithLogits(contacts_meas_logits, gt_contacts)
  2) direct_pose_geo:  geodesic loss between out_direct and gt_motion (rot6d only)

Typical use (Walk_F, left support window):
    python -m train.validate.diagnose_contact_meas_grad \\
        --teacher validate/teacher_batches/Walk_F_teacher.json \\
        --model models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1/ckpt_best_teacher_exp_phase_DirectBranch_v1_d1.pth \\
        --encoder-bundle models/motion_encoder_equiv_stageA.pt \\
        --depth 3 --steps 50-75 \\
        --w_contact_meas 0.05 --w_direct_pose 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train.models import MotionJointLoss
from train.validate.run_freerun_cycles import FreeRunCycleRunner, _build_full_cycle_sample, _load_json, _resolve_npz_path


def _parse_steps(spec: str, T: int) -> List[int]:
    spec = str(spec or "").strip().lower()
    if spec in ("", "all", "*"):
        return list(range(T))
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


def _index_time(x: torch.Tensor, steps: List[int]) -> torch.Tensor:
    if not steps:
        raise ValueError("steps is empty")
    T = int(x.shape[1])
    if len(steps) == T and steps[0] == 0 and steps[-1] == T - 1:
        return x
    idx = torch.as_tensor(steps, device=x.device, dtype=torch.long)
    return x.index_select(1, idx)


def _grad_l2(grads: List[Optional[torch.Tensor]]) -> torch.Tensor:
    acc = None
    for g in grads:
        if g is None:
            continue
        v = (g * g).sum()
        acc = v if acc is None else acc + v
    if acc is None:
        return torch.tensor(0.0)
    return acc.clamp_min(0.0).sqrt()


def _grad_dot(
    grads_a: List[Optional[torch.Tensor]],
    grads_b: List[Optional[torch.Tensor]],
) -> torch.Tensor:
    acc = None
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None:
            continue
        v = (ga * gb).sum()
        acc = v if acc is None else acc + v
    if acc is None:
        return torch.tensor(0.0)
    return acc


def _per_param_norms(
    names: List[str],
    grads: List[Optional[torch.Tensor]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for n, g in zip(names, grads):
        if g is None:
            out[n] = 0.0
        else:
            out[n] = float((g * g).sum().clamp_min(0.0).sqrt().detach().cpu().item())
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Gradient attribution for contact_meas_head (contact_meas_bce vs direct_pose_geo).",
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
    p.add_argument("--steps", type=str, default="all", help="Comma list or ranges (e.g. '50-75' or '0,1,2').")
    p.add_argument("--train-mode", action="store_true", help="Set model.train() for the forward/grad computation.")
    p.add_argument("--w_contact_meas", type=float, default=1.0, help="Scale applied to contact_meas_bce for grads.")
    p.add_argument("--w_direct_pose", type=float, default=1.0, help="Scale applied to direct_pose_geo for grads.")
    p.add_argument("--out", type=str, default="", help="Optional output JSON path. Empty disables writing.")
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
        so3_corr_apply=False,
        so3_corr_max_deg=20.0,
        lambda_fusion_apply=False,
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
    if bool(args.train_mode):
        model.train()
    else:
        model.eval()

    # Pull contact_meas_head params (by name for readable reporting).
    names: List[str] = []
    params: List[torch.nn.Parameter] = []
    for n, p0 in model.named_parameters():
        if n.startswith("contact_meas_head.") and p0.requires_grad:
            names.append(str(n))
            params.append(p0)
    if not params:
        raise SystemExit("[FATAL] model has no trainable contact_meas_head parameters.")

    # Build a MotionJointLoss helper for direct_pose_geo (needs MuY/StdY denorm for rot6d).
    loss_fn: MotionJointLoss = runner.loss_fn if runner.loss_fn is not None else MotionJointLoss(
        output_layout=runner.bundle.output_layout,
        fps=runner.bundle.fps,
        rot6d_spec=runner.bundle.rot6d_spec,
        meta=runner.bundle.meta,
    )
    loss_fn = loss_fn.to(device)
    loss_fn.mu_y = runner.bundle.mu_y
    loss_fn.std_y = runner.bundle.std_y

    # Prepare inputs (B=1, T=T_base); do NOT pass contacts into the model (avoid overriding learned meas).
    state = sample["motion"].unsqueeze(0).to(device)  # (1,T,Dx)
    gt_motion = sample["gt_motion"].unsqueeze(0).to(device)  # (1,T,Dy)
    gt_contacts = sample.get("contacts")
    gt_contacts = gt_contacts.unsqueeze(0).to(device) if torch.is_tensor(gt_contacts) else None
    cond = sample.get("cond_in")
    cond = cond.unsqueeze(0).to(device) if torch.is_tensor(cond) else None
    angvel = sample.get("angvel")
    angvel = angvel.unsqueeze(0).to(device) if torch.is_tensor(angvel) else None
    pose_hist = sample.get("pose_hist")
    pose_hist = pose_hist.unsqueeze(0).to(device) if torch.is_tensor(pose_hist) else None

    # Forward once; compute per-loss grads by autograd.grad on the same graph.
    with torch.enable_grad():
        ret = model(state, cond=cond, contacts=None, angvel=angvel, pose_history=pose_hist, plan_z=None, time_index=None)
        if not isinstance(ret, dict):
            raise SystemExit("[FATAL] model forward did not return a dict.")

        meas_logits = ret.get("contacts_meas_logits", None)
        direct_out = ret.get("out_direct", None)

        T = int(state.shape[1])
        steps = _parse_steps(args.steps, T)
        if not steps:
            raise SystemExit(f"[FATAL] empty step selection for steps={args.steps!r} with T={T}.")

        # Losses
        l_contact = None
        l_direct = None

        if torch.is_tensor(meas_logits) and torch.is_tensor(gt_contacts):
            logits = meas_logits
            if logits.ndim == 2:
                logits = logits.unsqueeze(1)
            gt = gt_contacts
            if gt.ndim == 2:
                gt = gt.unsqueeze(1)
            logits = _index_time(logits, steps)
            gt = _index_time(gt, steps).clamp(0.0, 1.0)
            if logits.shape != gt.shape:
                raise SystemExit(f"[FATAL] contacts_meas_logits {tuple(logits.shape)} != gt_contacts {tuple(gt.shape)}.")
            l_contact = F.binary_cross_entropy_with_logits(logits, gt)

        if torch.is_tensor(direct_out):
            d = direct_out
            if d.ndim == 2:
                d = d.unsqueeze(1)
            g = gt_motion
            if g.ndim == 2:
                g = g.unsqueeze(1)
            d = _index_time(d, steps)
            g = _index_time(g, steps)
            l_direct = loss_fn.compute_rot6d_geo_loss(d, g)

        if l_contact is None and l_direct is None:
            raise SystemExit("[FATAL] Neither contacts_meas_logits nor out_direct available for gradient attribution.")

        # Weighted variants (to mirror training scales).
        w_contact = float(args.w_contact_meas)
        w_direct = float(args.w_direct_pose)
        l_contact_w = (l_contact * w_contact) if l_contact is not None else None
        l_direct_w = (l_direct * w_direct) if l_direct is not None else None

        # Compute grads.
        zeros = [None for _ in params]

        grads_contact = (
            torch.autograd.grad(l_contact_w, params, retain_graph=True, allow_unused=True)
            if l_contact_w is not None
            else tuple(zeros)
        )
        grads_direct = (
            torch.autograd.grad(l_direct_w, params, retain_graph=False, allow_unused=True)
            if l_direct_w is not None
            else tuple(zeros)
        )
        grads_contact_l: List[Optional[torch.Tensor]] = list(grads_contact)
        grads_direct_l: List[Optional[torch.Tensor]] = list(grads_direct)

        # Combine vectors (grad is linear): g_total = g_contact + g_direct
        grads_total: List[Optional[torch.Tensor]] = []
        for gc, gd in zip(grads_contact_l, grads_direct_l):
            if gc is None and gd is None:
                grads_total.append(None)
            elif gc is None:
                grads_total.append(gd)
            elif gd is None:
                grads_total.append(gc)
            else:
                grads_total.append(gc + gd)

        n_contact = _grad_l2(grads_contact_l)
        n_direct = _grad_l2(grads_direct_l)
        n_total = _grad_l2(grads_total)
        dot_cd = _grad_dot(grads_contact_l, grads_direct_l)
        denom = (n_contact * n_direct).clamp_min(1e-12)
        cos_cd = dot_cd / denom if (float(n_contact.detach().cpu()) > 0.0 and float(n_direct.detach().cpu()) > 0.0) else torch.tensor(float("nan"))

        payload: Dict[str, Any] = {
            "clip": clip_name,
            "teacher": str(teacher_path),
            "model": str(Path(args.model).expanduser().resolve()),
            "bundle": str(Path(args.bundle).expanduser().resolve()),
            "T": int(T),
            "steps": steps,
            "w_contact_meas": float(w_contact),
            "w_direct_pose": float(w_direct),
            "loss_contact_meas_bce": float(l_contact.detach().cpu().item()) if l_contact is not None else None,
            "loss_direct_pose_geo_rad": float(l_direct.detach().cpu().item()) if l_direct is not None else None,
            "grad_l2_contact": float(n_contact.detach().cpu().item()),
            "grad_l2_direct": float(n_direct.detach().cpu().item()),
            "grad_l2_total": float(n_total.detach().cpu().item()),
            "grad_cos_contact_vs_direct": float(cos_cd.detach().cpu().item()) if torch.is_tensor(cos_cd) else float("nan"),
            "per_param_l2_contact": _per_param_norms(names, grads_contact_l),
            "per_param_l2_direct": _per_param_norms(names, grads_direct_l),
            "per_param_l2_total": _per_param_norms(names, grads_total),
        }
        # Convenience ratios (scalar norms).
        try:
            payload["grad_ratio_direct_over_contact"] = float((n_direct / n_contact.clamp_min(1e-12)).detach().cpu().item())
        except Exception:
            payload["grad_ratio_direct_over_contact"] = float("nan")

    # Console summary (compact).
    step_span = f"{steps[0]}-{steps[-1]}" if (len(steps) > 1 and steps[-1] - steps[0] + 1 == len(steps)) else f"{len(steps)} steps"
    print(f"[OK] clip={clip_name} T={T} sel={step_span} | w_contact={w_contact:g} w_direct={w_direct:g}")
    if payload["loss_contact_meas_bce"] is not None:
        print(f"  loss_contact_meas_bce={payload['loss_contact_meas_bce']:.6g}")
    if payload["loss_direct_pose_geo_rad"] is not None:
        print(f"  loss_direct_pose_geo_rad={payload['loss_direct_pose_geo_rad']:.6g}")
    print(
        "  grad_l2(contact_meas_head): "
        f"contact={payload['grad_l2_contact']:.6g} "
        f"direct={payload['grad_l2_direct']:.6g} "
        f"ratio(direct/contact)={payload.get('grad_ratio_direct_over_contact', float('nan')):.6g} "
        f"cos={payload.get('grad_cos_contact_vs_direct', float('nan')):.6g}"
    )

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

