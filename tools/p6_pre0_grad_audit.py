#!/usr/bin/env python3
"""
One-batch gradient-path audit for Stage7 pre0-collapse Phase1 (O1/O2/O3 triage).

This script:
1) Loads a checkpoint with the same runtime stack as run_freerun_cycles.
2) Runs single-step forward(s) on teacher samples.
3) Splits DirectGeoLocalDeg into leg vs non-leg components.
4) Compares autograd.grad(loss_leg) vs autograd.grad(loss_main) on:
   - leg head first layer weight
   - direct output tensor (pre-compose)
   - leg head pre0 input tensor (via forward hook)
   - cond input tensor (forced requires_grad for path tracing)
5) Dumps trainable-parameter selection snapshots for direct_pose modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from train.geometry import geodesic_R, reproject_rot6d, rot6d_to_matrix
from train.runtime.freeze import _freeze_all, _select_trainable_params, _unfreeze_direct_pose
from train.validate import run_freerun_cycles as rf


def _parse_steps(spec: str) -> List[int]:
    out: set[int] = set()
    for tok in str(spec or "").replace(";", ",").split(","):
        s = tok.strip()
        if not s:
            continue
        if "-" in s:
            a, b = [x.strip() for x in s.split("-", 1)]
            if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                lo = int(a)
                hi = int(b)
                if lo > hi:
                    lo, hi = hi, lo
                for v in range(lo, hi + 1):
                    out.add(int(v))
            continue
        if s.lstrip("-").isdigit():
            out.add(int(s))
    return sorted(out)


def _grad_norm(x: Optional[torch.Tensor]) -> Optional[float]:
    if x is None:
        return None
    g = x.detach()
    if not torch.isfinite(g).all():
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return float(g.float().pow(2).sum().sqrt().cpu().item())


def _autograd_norm(loss: torch.Tensor, target: Optional[torch.Tensor]) -> Optional[float]:
    if target is None or not torch.is_tensor(target) or (not bool(target.requires_grad)):
        return None
    grad = torch.autograd.grad(loss, target, retain_graph=True, allow_unused=True)[0]
    return _grad_norm(grad)


def _mask_to_index(mask: torch.Tensor) -> List[int]:
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
    return [int(v) for v in idx]


def _summarize_trainable(model: torch.nn.Module) -> Dict[str, Any]:
    modes = [
        ("direct_default", dict(hinge_only=False, gate_only=False, leg_only=False, leg_gate_only=False)),
        ("leg_train_only", dict(hinge_only=False, gate_only=False, leg_only=True, leg_gate_only=False)),
        ("leg_gate_only", dict(hinge_only=False, gate_only=False, leg_only=False, leg_gate_only=True)),
    ]
    out: Dict[str, Any] = {}
    for mode_name, kwargs in modes:
        _freeze_all(model)
        _unfreeze_direct_pose(model, **kwargs)
        _, names = _select_trainable_params(model)
        out[mode_name] = {
            "count": int(len(names)),
            "has_leg_head": any(str(n).startswith("direct_pose_leg_head.") for n in names),
            "has_leg_head_shared": any(str(n).startswith("direct_pose_leg_head_shared.") for n in names),
            "has_leg_gate": any(
                str(n).startswith("direct_pose_leg_gate_head.")
                or str(n).startswith("direct_pose_leg_gate_head_shared.")
                for n in names
            ),
            "has_direct_pose_head": any(str(n).startswith("direct_pose_head.") for n in names),
            "preview": [str(n) for n in names[:16]],
        }
    return out


def _leg_head_first_linear(model: torch.nn.Module) -> tuple[str, Optional[torch.nn.Parameter], Optional[torch.nn.Module]]:
    def _first_linear_with_name(prefix: str, mod: Any) -> tuple[str, Optional[torch.nn.Parameter], Optional[torch.nn.Module]]:
        if not isinstance(mod, torch.nn.Module):
            return "missing", None, None
        for name, sub in mod.named_modules():
            if isinstance(sub, torch.nn.Linear):
                key = f"{prefix}.{name}.weight" if name else f"{prefix}.weight"
                return key, sub.weight, sub
        return "missing", None, None

    side_routing = bool(getattr(model, "direct_pose_leg_side_routing", False))
    if side_routing and getattr(model, "direct_pose_leg_head_shared", None) is not None:
        k, w, m = _first_linear_with_name("direct_pose_leg_head_shared", getattr(model, "direct_pose_leg_head_shared", None))
        if m is not None:
            return k, w, m
    k, w, m = _first_linear_with_name("direct_pose_leg_head", getattr(model, "direct_pose_leg_head", None))
    if m is not None:
        return k, w, m
    return "missing", None, None


def _build_runner(model_path: Path, teacher_path: Path, bundle: Optional[str], pretrain_template: Optional[str], encoder_bundle: Optional[str]) -> rf.FreeRunCycleRunner:
    saved = list(sys.argv)
    argv = [
        "run_freerun_cycles.py",
        "--teacher",
        str(teacher_path),
        "--model",
        str(model_path),
    ]
    if bundle:
        argv.extend(["--bundle", str(bundle)])
    if pretrain_template:
        argv.extend(["--pretrain-template", str(pretrain_template)])
    if encoder_bundle:
        argv.extend(["--encoder-bundle", str(encoder_bundle)])
    try:
        sys.argv = argv
        args = rf.parse_args()
    finally:
        sys.argv = saved
    return rf.FreeRunCycleRunner(args)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase1 pre0 gradient-path audit (single-step).")
    ap.add_argument("--model", type=str, required=True, help="Checkpoint (.pth)")
    ap.add_argument("--teacher", type=str, required=True, help="Teacher JSON")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--steps", type=str, default="0", help="Comma/range list, e.g. 0 or 9-14,39-42")
    ap.add_argument("--direct-pose-leg-detach-feat", type=str, default=None, help="Override true/false")
    ap.add_argument("--direct-pose-leg-stopgrad-main", type=str, default=None, help="Override true/false")
    ap.add_argument("--bundle", type=str, default=None)
    ap.add_argument("--pretrain-template", type=str, default=None)
    ap.add_argument("--encoder-bundle", type=str, default=None)
    ap.add_argument("--out-json", type=str, required=True)
    args = ap.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    teacher_path = Path(args.teacher).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    steps = _parse_steps(args.steps)
    if not steps:
        raise SystemExit("[FATAL] --steps resolved to empty.")

    runner = _build_runner(
        model_path=model_path,
        teacher_path=teacher_path,
        bundle=args.bundle,
        pretrain_template=args.pretrain_template,
        encoder_bundle=args.encoder_bundle,
    )

    teacher_payload = json.loads(teacher_path.read_text())
    clip_name = str(teacher_payload.get("clip") or teacher_path.stem.replace("_teacher", ""))
    npz_path = rf._resolve_npz_path(clip_name, teacher_payload.get("source_json"), Path(args.npz_root).expanduser().resolve())
    ds = runner._build_dataset(npz_path, seq_len=int(teacher_payload.get("num_pairs", 0) or 0))
    runner._ensure_model_ready(ds)
    clip = ds.clips[0]
    sample = rf._build_full_cycle_sample(ds, clip, seq_len=int(teacher_payload.get("num_pairs", 0) or 0))

    model = runner.model
    trainer = runner.trainer
    assert model is not None
    assert trainer is not None
    model.eval()

    # Runtime flag overrides (for R0/R1/R2/R3 style gradient checks).
    if args.direct_pose_leg_detach_feat is not None:
        setattr(
            model,
            "direct_pose_leg_detach_feat",
            str(args.direct_pose_leg_detach_feat).strip().lower() in ("1", "true", "yes", "on"),
        )
    if args.direct_pose_leg_stopgrad_main is not None:
        setattr(
            model,
            "direct_pose_leg_stopgrad_main",
            str(args.direct_pose_leg_stopgrad_main).strip().lower() in ("1", "true", "yes", "on"),
        )

    leg_w_name, leg_w, leg_first_linear = _leg_head_first_linear(model)
    if leg_w is None or leg_first_linear is None:
        raise SystemExit("[FATAL] Cannot locate leg head first linear layer.")

    # Columns for rot6d<->matrix conversion.
    columns = ("X", "Z")
    rot6d_spec = getattr(ds, "rot6d_spec", None)
    if isinstance(rot6d_spec, dict):
        cols = rot6d_spec.get("columns")
        if isinstance(cols, (list, tuple)) and len(cols) >= 2:
            a = str(cols[0]).strip().upper()
            b = str(cols[1]).strip().upper()
            if a in ("X", "Y", "Z") and b in ("X", "Y", "Z") and a != b:
                columns = (a, b)

    motion_seq = sample["motion"].to(runner.device)
    gt_seq = sample["gt_motion"].to(runner.device)
    cond_seq = sample["cond_in"].to(runner.device)
    contacts_seq = sample.get("contacts", None)
    angvel_seq = sample.get("angvel", None)
    pose_hist_seq = sample.get("pose_hist", None)
    if torch.is_tensor(contacts_seq):
        contacts_seq = contacts_seq.to(runner.device)
    if torch.is_tensor(angvel_seq):
        angvel_seq = angvel_seq.to(runner.device)
    if torch.is_tensor(pose_hist_seq):
        pose_hist_seq = pose_hist_seq.to(runner.device)

    T = int(motion_seq.shape[0])
    step_rows: List[Dict[str, Any]] = []

    for t in steps:
        if t < 0 or t >= T:
            continue
        model.zero_grad(set_to_none=True)

        motion_t = motion_seq[t : t + 1].clone().detach().requires_grad_(True)
        cond_t = cond_seq[t : t + 1].clone().detach().requires_grad_(True)
        gt_t = gt_seq[t : t + 1]

        contacts_t = contacts_seq[t : t + 1] if torch.is_tensor(contacts_seq) else None
        angvel_t = angvel_seq[t : t + 1] if torch.is_tensor(angvel_seq) else None
        pose_hist_t = pose_hist_seq[t : t + 1] if torch.is_tensor(pose_hist_seq) else None

        pre0_holder: Dict[str, Any] = {"tensor": None}

        def _hook(_module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            x = inputs[0] if inputs else None
            pre0_holder["tensor"] = x
            if torch.is_tensor(x) and bool(x.requires_grad):
                x.retain_grad()

        hk = leg_first_linear.register_forward_hook(_hook)
        try:
            ret = model(
                motion_t.unsqueeze(1),
                cond_t.unsqueeze(1),
                contacts=contacts_t.unsqueeze(1) if torch.is_tensor(contacts_t) else None,
                angvel=angvel_t.unsqueeze(1) if torch.is_tensor(angvel_t) else None,
                pose_history=pose_hist_t.unsqueeze(1) if torch.is_tensor(pose_hist_t) else None,
                time_index=torch.tensor([int(t)], device=motion_t.device, dtype=torch.long),
            )
        finally:
            hk.remove()

        direct_norm = ret.get("out_direct", None)
        if not torch.is_tensor(direct_norm):
            continue
        if direct_norm.dim() == 3 and direct_norm.size(1) == 1:
            direct_norm = direct_norm[:, 0]
        direct_norm.retain_grad()

        omega_leg = ret.get("direct_leg_omega", None)
        if torch.is_tensor(omega_leg) and omega_leg.dim() == 4 and omega_leg.size(1) == 1:
            omega_leg = omega_leg[:, 0]

        # Compose SO(3) leg correction in normalized Y space (same helper as freerun).
        direct_norm_eff = rf._apply_direct_leg_so3_correction_norm(
            trainer=trainer,
            model=model,
            direct_norm=direct_norm,
            omega_leg=omega_leg,
            columns=columns,
            omega_scale=1.0,
            omega_sign=1.0,
            apply_side="left",
        )

        # DirectGeoLocalDeg matrix (B,J) in degrees.
        rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
        if not isinstance(rot_slice, slice):
            continue
        rot_len = int(rot_slice.stop - rot_slice.start)
        if rot_len <= 0 or (rot_len % 6) != 0:
            continue
        J = int(rot_len // 6)

        pred_raw = trainer._denorm(direct_norm_eff)
        gt_raw = trainer._denorm(gt_t)
        pred6 = reproject_rot6d(pred_raw[..., rot_slice]).view(pred_raw.shape[0], J, 6)
        gt6 = reproject_rot6d(gt_raw[..., rot_slice]).view(gt_raw.shape[0], J, 6)
        R_pred = rot6d_to_matrix(pred6, columns=columns)
        R_gt = rot6d_to_matrix(gt6, columns=columns)
        dloc_deg = geodesic_R(R_pred, R_gt) * (180.0 / float(np.pi))

        root_idx = int(getattr(getattr(trainer, "loss_fn", None), "root_idx", 0) or 0)
        root_idx = max(0, min(J - 1, root_idx))
        non_root = torch.ones(J, device=dloc_deg.device, dtype=torch.bool)
        non_root[root_idx] = False

        leg_idx = getattr(model, "direct_pose_leg_joint_idx_tensor", None)
        leg_mask = torch.zeros(J, device=dloc_deg.device, dtype=torch.bool)
        if torch.is_tensor(leg_idx) and int(leg_idx.numel()) > 0:
            idx_use = leg_idx.to(device=dloc_deg.device, dtype=torch.long)
            idx_use = idx_use[(idx_use >= 0) & (idx_use < J)]
            if int(idx_use.numel()) > 0:
                leg_mask[idx_use] = True
        leg_mask = leg_mask & non_root
        main_mask = non_root & (~leg_mask)

        if bool(leg_mask.any().detach().cpu().item()):
            loss_leg = dloc_deg[:, leg_mask].mean()
        else:
            loss_leg = dloc_deg.new_zeros(())
        if bool(main_mask.any().detach().cpu().item()):
            loss_main = dloc_deg[:, main_mask].mean()
        else:
            loss_main = dloc_deg.new_zeros(())

        pre0_t = pre0_holder.get("tensor", None)

        # Component-wise autograd.grad comparison.
        comp_leg = {
            "leg_head_w": _autograd_norm(loss_leg, leg_w),
            "direct_out": _autograd_norm(loss_leg, direct_norm),
            "pre0": _autograd_norm(loss_leg, pre0_t if torch.is_tensor(pre0_t) else None),
            "cond_in": _autograd_norm(loss_leg, cond_t),
        }
        comp_main = {
            "leg_head_w": _autograd_norm(loss_main, leg_w),
            "direct_out": _autograd_norm(loss_main, direct_norm),
            "pre0": _autograd_norm(loss_main, pre0_t if torch.is_tensor(pre0_t) else None),
            "cond_in": _autograd_norm(loss_main, cond_t),
        }

        # Sanity backward on combined loss.
        model.zero_grad(set_to_none=True)
        total = loss_leg + loss_main
        total.backward()
        backward = {
            "leg_head_w": _grad_norm(leg_w.grad if leg_w is not None else None),
            "direct_out": _grad_norm(direct_norm.grad),
            "pre0": _grad_norm(pre0_t.grad if torch.is_tensor(pre0_t) else None),
            "cond_in": _grad_norm(cond_t.grad),
        }

        step_rows.append(
            {
                "step": int(t),
                "loss_leg_deg": float(loss_leg.detach().cpu().item()),
                "loss_main_deg": float(loss_main.detach().cpu().item()),
                "joint_mask": {
                    "root_idx": int(root_idx),
                    "leg_idx": _mask_to_index(leg_mask),
                    "main_idx": _mask_to_index(main_mask),
                },
                "pre0_requires_grad": bool(torch.is_tensor(pre0_t) and pre0_t.requires_grad),
                "flags": {
                    "direct_pose_leg_detach_feat": bool(getattr(model, "direct_pose_leg_detach_feat", False)),
                    "direct_pose_leg_stopgrad_main": bool(getattr(model, "direct_pose_leg_stopgrad_main", False)),
                },
                "autograd_grad_norm": {
                    "loss_leg": comp_leg,
                    "loss_main": comp_main,
                },
                "backward_grad_norm": backward,
            }
        )

    # Aggregate quick statistics across audited steps.
    def _median(vals: Iterable[Optional[float]]) -> Optional[float]:
        arr = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        if not arr:
            return None
        return float(np.median(np.asarray(arr, dtype=np.float64)))

    agg = {
        "steps_used": int(len(step_rows)),
        "median_autograd": {
            "loss_leg_leg_head_w": _median(r["autograd_grad_norm"]["loss_leg"]["leg_head_w"] for r in step_rows),
            "loss_main_leg_head_w": _median(r["autograd_grad_norm"]["loss_main"]["leg_head_w"] for r in step_rows),
            "loss_leg_direct_out": _median(r["autograd_grad_norm"]["loss_leg"]["direct_out"] for r in step_rows),
            "loss_main_direct_out": _median(r["autograd_grad_norm"]["loss_main"]["direct_out"] for r in step_rows),
            "loss_leg_pre0": _median(r["autograd_grad_norm"]["loss_leg"]["pre0"] for r in step_rows),
            "loss_main_pre0": _median(r["autograd_grad_norm"]["loss_main"]["pre0"] for r in step_rows),
            "loss_leg_cond_in": _median(r["autograd_grad_norm"]["loss_leg"]["cond_in"] for r in step_rows),
            "loss_main_cond_in": _median(r["autograd_grad_norm"]["loss_main"]["cond_in"] for r in step_rows),
        },
        "median_backward": {
            "leg_head_w": _median(r["backward_grad_norm"]["leg_head_w"] for r in step_rows),
            "direct_out": _median(r["backward_grad_norm"]["direct_out"] for r in step_rows),
            "pre0": _median(r["backward_grad_norm"]["pre0"] for r in step_rows),
            "cond_in": _median(r["backward_grad_norm"]["cond_in"] for r in step_rows),
        },
    }

    trainable_audit = _summarize_trainable(model)

    out = {
        "model": str(model_path),
        "teacher": str(teacher_path),
        "npz": str(npz_path),
        "clip": str(clip_name),
        "settings": {
            "steps": steps,
            "direct_pose_leg_detach_feat": bool(getattr(model, "direct_pose_leg_detach_feat", False)),
            "direct_pose_leg_stopgrad_main": bool(getattr(model, "direct_pose_leg_stopgrad_main", False)),
            "leg_head_weight_name": str(leg_w_name),
        },
        "aggregate": agg,
        "step_results": step_rows,
        "trainable_param_audit": trainable_audit,
    }

    out_json.write_text(json.dumps(out, indent=2))
    print(f"[OK] wrote {out_json}")
    if step_rows:
        print(
            "[Quick] median grads:",
            json.dumps(
                {
                    "loss_leg->leg_head_w": agg["median_autograd"]["loss_leg_leg_head_w"],
                    "loss_main->leg_head_w": agg["median_autograd"]["loss_main_leg_head_w"],
                    "loss_leg->direct_out": agg["median_autograd"]["loss_leg_direct_out"],
                    "loss_main->direct_out": agg["median_autograd"]["loss_main_direct_out"],
                },
                ensure_ascii=False,
            ),
        )
    else:
        print("[WARN] no valid step rows collected.")


if __name__ == "__main__":
    main()
