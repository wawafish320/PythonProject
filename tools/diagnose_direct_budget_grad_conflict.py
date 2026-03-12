#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from train.geometry import geodesic_R, reproject_rot6d, rot6d_to_matrix
from train.models import MotionJointLoss
from train.validate.run_freerun_cycles import FreeRunCycleRunner, _build_full_cycle_sample, _load_json, _resolve_npz_path


def _as_float(v: Any, default: float) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _as_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _safe_div(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
    return float(a / b)


def _parse_columns(rot6d_spec: Dict[str, Any]) -> Tuple[str, str]:
    cols = None
    if isinstance(rot6d_spec, dict):
        cols = rot6d_spec.get("columns")
    if isinstance(cols, (list, tuple)) and len(cols) >= 2:
        a = str(cols[0]).strip().upper()
        b = str(cols[1]).strip().upper()
        if a in ("X", "Y", "Z") and b in ("X", "Y", "Z") and a != b:
            return (a, b)
    return ("X", "Z")


def _grad_pair_stats(
    grads_a: Sequence[Optional[torch.Tensor]],
    grads_b: Sequence[Optional[torch.Tensor]],
) -> Tuple[float, float, float]:
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    for ga, gb in zip(grads_a, grads_b):
        if ga is not None:
            na2 += float((ga.detach().float().pow(2).sum()).item())
        if gb is not None:
            nb2 += float((gb.detach().float().pow(2).sum()).item())
        if ga is not None and gb is not None:
            dot += float((ga.detach().float() * gb.detach().float()).sum().item())
    na = math.sqrt(max(0.0, na2))
    nb = math.sqrt(max(0.0, nb2))
    if na > 1e-12 and nb > 1e-12:
        cos = float(dot / (na * nb))
    else:
        cos = float("nan")
    ratio = float(nb / na) if na > 1e-12 else float("nan")
    return na, nb, ratio if math.isfinite(ratio) else float("nan"), cos


def _summary_stats(vals: Iterable[float]) -> Dict[str, float]:
    arr = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not arr:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    arr_sorted = sorted(arr)
    p90_idx = min(len(arr_sorted) - 1, max(0, int(round(0.9 * (len(arr_sorted) - 1)))))
    return {
        "count": len(arr_sorted),
        "mean": float(mean(arr_sorted)),
        "median": float(median(arr_sorted)),
        "p90": float(arr_sorted[p90_idx]),
        "min": float(arr_sorted[0]),
        "max": float(arr_sorted[-1]),
    }


def _ratio_frac(vals: Iterable[float], thr: float) -> float:
    arr = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not arr:
        return float("nan")
    cnt = sum(1 for x in arr if x > float(thr))
    return float(cnt / max(1, len(arr)))


def _neg_frac(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not arr:
        return float("nan")
    cnt = sum(1 for x in arr if x < 0.0)
    return float(cnt / max(1, len(arr)))


def _finite_or_none(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _safe_joint_names(names: Optional[Sequence[Any]], joint_count: int) -> List[str]:
    out: List[str] = []
    if isinstance(names, (list, tuple)):
        out = [str(x) for x in names[: max(0, int(joint_count))]]
    if len(out) < int(joint_count):
        out.extend([f"joint_{i}" for i in range(len(out), int(joint_count))])
    return out[: int(joint_count)]


def _resolve_joint_indices_with_names(spec: str, joint_count: int, names: Sequence[str]) -> List[int]:
    idxs: List[int] = []
    seen: set[int] = set()
    name_exact = {str(n): int(i) for i, n in enumerate(names[: max(0, int(joint_count))])}
    name_lower = {str(n).lower(): int(i) for i, n in enumerate(names[: max(0, int(joint_count))])}
    for tok in str(spec or "").split(","):
        t = str(tok or "").strip()
        if not t:
            continue
        idx: Optional[int] = None
        try:
            iv = int(t)
            if 0 <= iv < int(joint_count):
                idx = int(iv)
        except Exception:
            idx = None
        if idx is None:
            if t in name_exact:
                idx = int(name_exact[t])
            else:
                idx = name_lower.get(t.lower(), None)
        if idx is None or idx in seen:
            continue
        seen.add(int(idx))
        idxs.append(int(idx))
    return idxs


def _joint_names_from_indices(names: Sequence[str], idxs: Sequence[int]) -> List[str]:
    out: List[str] = []
    for idx in idxs:
        i = int(idx)
        if 0 <= i < len(names):
            out.append(str(names[i]))
        else:
            out.append(f"joint_{i}")
    return out


@dataclass
class ArmSpec:
    name: str
    runtime_config: Path
    ckpt: Path


def _parse_arm(text: str) -> ArmSpec:
    parts = [p.strip() for p in str(text or "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid --arm {text!r}; expected 'name,runtime_json,ckpt'.")
    return ArmSpec(name=parts[0], runtime_config=Path(parts[1]).expanduser().resolve(), ckpt=Path(parts[2]).expanduser().resolve())


def _build_loss_fn(cfg: Dict[str, Any], runner: FreeRunCycleRunner) -> MotionJointLoss:
    loss_fn = MotionJointLoss(
        output_layout=runner.bundle.output_layout,
        fps=runner.bundle.fps,
        rot6d_spec=runner.bundle.rot6d_spec,
        meta=runner.bundle.meta,
        w_direct_pose_trigger_twist=_as_float(cfg.get("w_direct_pose_trigger_twist", 0.0), 0.0),
        w_direct_pose_trigger_swing_x=_as_float(cfg.get("w_direct_pose_trigger_swing_x", 0.0), 0.0),
        w_direct_pose_trigger_swing_y=_as_float(cfg.get("w_direct_pose_trigger_swing_y", 0.0), 0.0),
        direct_pose_trigger_joint=str(cfg.get("direct_pose_trigger_joint", "foot_r")),
        direct_pose_trigger_contact_r_thr=_as_float(cfg.get("direct_pose_trigger_contact_r_thr", 0.3), 0.3),
        direct_pose_trigger_left_stance_thr=_as_float(cfg.get("direct_pose_trigger_left_stance_thr", 0.55), 0.55),
        direct_pose_trigger_phase_margin=_as_float(cfg.get("direct_pose_trigger_phase_margin", 0.05), 0.05),
        direct_pose_trigger_gate_mode=str(cfg.get("direct_pose_trigger_gate_mode", "hard")),
        direct_pose_trigger_tau_phase=_as_float(cfg.get("direct_pose_trigger_tau_phase", 0.05), 0.05),
        direct_pose_trigger_tau_contact=_as_float(cfg.get("direct_pose_trigger_tau_contact", 0.05), 0.05),
        direct_pose_trigger_tau_twist_deg=_as_float(cfg.get("direct_pose_trigger_tau_twist_deg", 5.0), 5.0),
        direct_pose_trigger_twist_ref=str(cfg.get("direct_pose_trigger_twist_ref", "gt")),
        direct_pose_trigger_sign_source=str(cfg.get("direct_pose_trigger_sign_source", "gt")),
        direct_pose_trigger_loss=str(cfg.get("direct_pose_trigger_loss", "smooth_l1")),
        direct_pose_trigger_beta_deg=_as_float(cfg.get("direct_pose_trigger_beta_deg", 5.0), 5.0),
        w_direct_pose_trigger_total=_as_float(cfg.get("w_direct_pose_trigger_total", 1.0), 1.0),
        direct_pose_trigger_under_weight=_as_float(cfg.get("direct_pose_trigger_under_weight", 1.0), 1.0),
        direct_pose_trigger_under_mode=str(cfg.get("direct_pose_trigger_under_mode", "off")),
        direct_pose_budget_mode=str(cfg.get("direct_pose_budget_mode", "off")),
        direct_pose_budget_ema_beta=_as_float(cfg.get("direct_pose_budget_ema_beta", 0.95), 0.95),
        direct_pose_budget_eps=_as_float(cfg.get("direct_pose_budget_eps", 1e-4), 1e-4),
        direct_pose_budget_lambda_trigger=_as_float(cfg.get("direct_pose_budget_lambda_trigger", 0.0), 0.0),
        direct_pose_budget_lambda_chain=_as_float(cfg.get("direct_pose_budget_lambda_chain", 0.0), 0.0),
        direct_pose_budget_lambda_guard=_as_float(cfg.get("direct_pose_budget_lambda_guard", 0.0), 0.0),
        direct_pose_budget_chain_joints=str(cfg.get("direct_pose_budget_chain_joints", "calf_r,ball_r")),
        direct_pose_budget_chain_frame_mode=str(cfg.get("direct_pose_budget_chain_frame_mode", "trigger")),
        direct_pose_budget_guard_frame_mode=str(cfg.get("direct_pose_budget_guard_frame_mode", "non_trigger")),
        direct_pose_budget_guard_exclude_joints=str(cfg.get("direct_pose_budget_guard_exclude_joints", "")),
        direct_pose_budget_grad_scope=str(cfg.get("direct_pose_budget_grad_scope", "all")),
    )
    loss_fn = loss_fn.to(runner.device)
    try:
        loss_fn.mu_y = runner.bundle.mu_y
        loss_fn.std_y = runner.bundle.std_y
    except Exception:
        pass
    return loss_fn


def _slice_step(x: Optional[torch.Tensor], idx: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if not torch.is_tensor(x):
        return None
    if x.dim() < 2:
        return None
    t = int(x.shape[0])
    i = int(max(0, min(t - 1, idx)))
    return x[i : i + 1].unsqueeze(0).to(device=device, dtype=dtype)


def _arm_run(
    *,
    arm: ArmSpec,
    teacher_path: Path,
    npz_root: Path,
    bundle: Path,
    pretrain_template: Path,
    encoder_bundle: Path,
    device: str,
    depth: int,
    num_heads: int,
    dropout: float,
    context_len: int,
    steps: int,
    seed: int,
    budget_activation_thr: float,
) -> Dict[str, Any]:
    cfg = json.loads(arm.runtime_config.read_text())
    teacher_data = _load_json(teacher_path)
    clip_name = str(teacher_data.get("clip") or teacher_path.stem.replace("_teacher", ""))
    seq_len = int(teacher_data.get("num_pairs", 0) or 0)
    npz_path = _resolve_npz_path(clip_name, teacher_data.get("source_json"), npz_root)

    runner_args = argparse.Namespace(
        model=str(arm.ckpt),
        teacher=str(teacher_path),
        bundle=str(bundle),
        pretrain_template=str(pretrain_template),
        encoder_bundle=str(encoder_bundle),
        device=str(device),
        num_heads=int(num_heads),
        dropout=float(dropout),
        context_len=int(context_len),
        depth=int(depth),
        so3_corr_apply=True,
        so3_corr_max_deg=20.0,
        so3_corr_gate_force=None,
        so3_corr_gate_from_contacts_err=False,
        so3_corr_gate_from_contacts_err_mode="global",
        so3_corr_gate_err_k=0.0,
        so3_corr_gate_err_bias=0.0,
        so3_corr_gate_err_max=1.0,
        so3_corr_gate_err_ref_steps=16,
        so3_corr_gate_err_margin=0.0,
        so3_corr_gate_err_use_ref=False,
        so3_corr_gate_scale_max=1.0,
        lambda_fusion_apply=True,
        phase_reset_source="none",
        phase_reset_source_strict="on",
        contact_plan_measure_kind="heuristic",
        contact_plan_measure_force=False,
        contact_plan_measure_force_value=0.0,
        contact_plan_inject_scale=1.0,
        contact_plan_time_bias_scale=1.0,
        direct_pose_meas_force_zero=False,
        direct_pose_leg_cross_leg_ablate="none",
        direct_pose_leg_side_plan_other_ablate="none",
        log_contact_plan_logits_decomp=False,
    )
    runner = FreeRunCycleRunner(runner_args)
    ds = runner._build_dataset(npz_path, seq_len=seq_len)
    runner._ensure_model_ready(ds)
    if runner.model is None:
        raise RuntimeError(f"[{arm.name}] runner.model is None")
    model = runner.model.to(runner.device)
    trainer = runner.trainer
    if trainer is None:
        raise RuntimeError(f"[{arm.name}] runner.trainer is None")

    clip = ds.clips[0]
    sample = _build_full_cycle_sample(ds, clip, seq_len=seq_len)
    motion_seq = sample["motion"]
    gt_seq = sample["gt_motion"]
    cond_seq = sample.get("cond_in")
    contacts_seq = sample.get("contacts")
    angvel_seq = sample.get("angvel")
    pose_hist_seq = sample.get("pose_hist")

    columns = _parse_columns(getattr(ds, "rot6d_spec", {}) or {})
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        rot_slice = slice(0, int(gt_seq.shape[-1]))
    rot_len = int(rot_slice.stop - rot_slice.start)
    if rot_len <= 0 or rot_len % 6 != 0:
        raise RuntimeError(f"[{arm.name}] invalid rot slice len={rot_len}")
    num_joints = rot_len // 6

    loss_fn = _build_loss_fn(cfg, runner)
    if getattr(ds, "bone_names", None):
        try:
            loss_fn.set_bone_names(ds.bone_names)
        except Exception:
            pass
    runtime_bone_names = _safe_joint_names(getattr(loss_fn, "bone_names", None), num_joints)
    dataset_bone_names = _safe_joint_names(getattr(ds, "bone_names", None), num_joints)
    root_idx = int(getattr(loss_fn, "root_idx", 0) or 0)
    lr = _as_float(cfg.get("lr", 1e-3), 1e-3)
    weight_decay = _as_float(cfg.get("weight_decay", 0.0), 0.0)

    chain_spec = str(getattr(loss_fn, "direct_pose_budget_chain_joints", "") or "")
    chain_idxs_runtime = list(loss_fn._resolve_joint_indices_spec(chain_spec, num_joints))
    chain_idxs_dataset = _resolve_joint_indices_with_names(chain_spec, num_joints, dataset_bone_names)
    guard_excl_spec = str(getattr(loss_fn, "direct_pose_budget_guard_exclude_joints", "") or "").strip()
    guard_excl_idxs_runtime = list(loss_fn._resolve_joint_indices_spec(guard_excl_spec, num_joints)) if guard_excl_spec else []
    guard_excl_idxs_dataset = (
        _resolve_joint_indices_with_names(guard_excl_spec, num_joints, dataset_bone_names) if guard_excl_spec else []
    )

    for p in model.parameters():
        p.requires_grad_(False)
    direct_param_names: List[str] = []
    direct_params: List[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if str(n).startswith("direct_pose_head."):
            p.requires_grad_(True)
            direct_param_names.append(str(n))
            direct_params.append(p)
    if not direct_params:
        raise RuntimeError(f"[{arm.name}] no trainable direct_pose_head.* params.")

    model.train()
    optimizer = torch.optim.AdamW(direct_params, lr=lr, weight_decay=weight_decay)
    t_total = int(motion_seq.shape[0])
    if t_total <= 1:
        raise RuntimeError(f"[{arm.name}] sample length too short: T={t_total}")

    rows: List[Dict[str, Any]] = []
    per_param_bucket: Dict[str, Dict[str, List[float]]] = {
        n: {"main_norm": [], "budget_norm": [], "ratio": [], "cos": []} for n in direct_param_names
    }
    overall_cos: List[float] = []
    overall_ratio: List[float] = []
    budget_active_steps = 0
    budget_joint_count = {
        "trigger": [0 for _ in range(num_joints)],
        "chain": [0 for _ in range(num_joints)],
        "guard": [0 for _ in range(num_joints)],
        "any": [0 for _ in range(num_joints)],
    }
    budget_joint_weight = {
        "trigger": [0.0 for _ in range(num_joints)],
        "chain": [0.0 for _ in range(num_joints)],
        "guard": [0.0 for _ in range(num_joints)],
        "any": [0.0 for _ in range(num_joints)],
    }

    for step_idx in range(int(steps)):
        idx = int(step_idx % (t_total - 1))
        motion_t = _slice_step(motion_seq, idx, runner.device, next(model.parameters()).dtype)
        cond_t = _slice_step(cond_seq, idx, runner.device, next(model.parameters()).dtype)
        contacts_t = _slice_step(contacts_seq, idx, runner.device, next(model.parameters()).dtype)
        angvel_t = _slice_step(angvel_seq, idx, runner.device, next(model.parameters()).dtype)
        pose_hist_t = _slice_step(pose_hist_seq, idx, runner.device, next(model.parameters()).dtype)
        gt_t_norm = _slice_step(gt_seq, idx, runner.device, next(model.parameters()).dtype)
        if motion_t is None or gt_t_norm is None:
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            ret = model(
                motion_t,
                cond=cond_t,
                contacts=contacts_t,
                angvel=angvel_t,
                pose_history=pose_hist_t,
                plan_z=None,
                time_index=int(idx),
                rollout_step=None,
            )
            if not isinstance(ret, dict):
                raise RuntimeError(f"[{arm.name}] model forward did not return dict.")
            direct_raw = ret.get("out_direct", None)
            if not torch.is_tensor(direct_raw):
                raise RuntimeError(f"[{arm.name}] out_direct missing.")
            if direct_raw.dim() == 3:
                direct_raw = direct_raw[:, -1]
            if direct_raw.dim() != 2:
                raise RuntimeError(f"[{arm.name}] out_direct shape invalid: {tuple(direct_raw.shape)}")

            gt_raw = trainer._denorm(gt_t_norm[:, 0])

            dir6 = reproject_rot6d(direct_raw[..., rot_slice]).view(1, num_joints, 6)
            gt6 = reproject_rot6d(gt_raw[..., rot_slice]).view(1, num_joints, 6)
            r_dir = rot6d_to_matrix(dir6, columns=columns)
            r_gt = rot6d_to_matrix(gt6, columns=columns)
            theta = geodesic_R(r_dir, r_gt, reduce=None)  # (1, J)
            if num_joints > 1 and 0 <= root_idx < num_joints:
                mask = torch.ones((num_joints,), dtype=torch.bool, device=theta.device)
                mask[root_idx] = False
                main_loss = theta[:, mask].mean()
            else:
                main_loss = theta.mean()

            direct_pred_step = trainer.normalizer.norm_y(direct_raw.unsqueeze(1))
            direct_gt_step = trainer.normalizer.norm_y(gt_raw.unsqueeze(1))
            tb_batch = None
            if contacts_t is not None and contacts_t.dim() == 3 and int(contacts_t.shape[-1]) >= 2:
                tb_batch = {"contacts": contacts_t}
            trig_loss, trig_stats, trig_aux = loss_fn._compute_direct_pose_trigger_twist_loss(
                direct_pred=direct_pred_step,
                direct_gt=direct_gt_step,
                theta_per_joint=theta.unsqueeze(1),
                batch=tb_batch,
            )
            budget_loss, budget_stats, _ = loss_fn._compute_direct_pose_budget_loss(
                theta_per_joint=theta.unsqueeze(1),
                trig_loss=trig_loss if torch.is_tensor(trig_loss) else None,
                trig_aux=trig_aux if isinstance(trig_aux, dict) else None,
            )
            if not torch.is_tensor(budget_loss):
                budget_loss = main_loss.new_tensor(0.0)

            grads_main = list(torch.autograd.grad(main_loss, direct_params, retain_graph=True, allow_unused=True))
            if bool(getattr(budget_loss, "requires_grad", False)):
                grads_budget = list(torch.autograd.grad(budget_loss, direct_params, retain_graph=True, allow_unused=True))
            else:
                grads_budget = [None for _ in direct_params]
            n_main, n_budget, ratio, cos = _grad_pair_stats(grads_main, grads_budget)

            per_param = {}
            for n, gm, gb in zip(direct_param_names, grads_main, grads_budget):
                gm_l = float(math.sqrt(float((gm.detach().float().pow(2).sum()).item()))) if gm is not None else 0.0
                gb_l = float(math.sqrt(float((gb.detach().float().pow(2).sum()).item()))) if gb is not None else 0.0
                if gm is not None and gb is not None and gm_l > 1e-12 and gb_l > 1e-12:
                    cos_p = float((gm.detach().float() * gb.detach().float()).sum().item() / (gm_l * gb_l))
                else:
                    cos_p = float("nan")
                ratio_p = float(gb_l / gm_l) if gm_l > 1e-12 else float("nan")
                per_param[n] = {"main_norm": gm_l, "budget_norm": gb_l, "ratio": ratio_p, "cos": cos_p}
                per_param_bucket[n]["main_norm"].append(gm_l)
                per_param_bucket[n]["budget_norm"].append(gb_l)
                if math.isfinite(ratio_p):
                    per_param_bucket[n]["ratio"].append(ratio_p)
                if math.isfinite(cos_p):
                    per_param_bucket[n]["cos"].append(cos_p)

            total_loss = main_loss + budget_loss
            total_loss.backward()
            optimizer.step()

        # Mirror budget branch masking logic in MotionJointLoss to recover per-joint activation structure.
        hard_mask = None
        trig_joint_idx = None
        if isinstance(trig_aux, dict):
            m = trig_aux.get("hard_mask", None)
            if torch.is_tensor(m) and tuple(m.shape) == (1, 1):
                hard_mask = m.to(device=theta.device, dtype=torch.bool)
            try:
                jv = int(trig_aux.get("joint_idx", -1))
                if 0 <= jv < num_joints:
                    trig_joint_idx = int(jv)
            except Exception:
                trig_joint_idx = None
        if hard_mask is None:
            hard_mask = torch.zeros((1, 1), dtype=torch.bool, device=theta.device)
        hard_any = bool(hard_mask.any().detach().cpu().item())

        chain_joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=theta.device)
        for ii in chain_idxs_runtime:
            if 0 <= int(ii) < num_joints:
                chain_joint_mask[int(ii)] = True

        chain_frame_mode = str(getattr(loss_fn, "direct_pose_budget_chain_frame_mode", "trigger") or "trigger").strip().lower()
        if chain_frame_mode == "trigger":
            chain_frame_mask = hard_mask if hard_any else torch.ones_like(hard_mask, dtype=torch.bool)
        else:
            chain_frame_mask = torch.ones_like(hard_mask, dtype=torch.bool)
        chain_frame_active = bool(chain_frame_mask[0, 0].detach().cpu().item())

        guard_frame_mode = str(getattr(loss_fn, "direct_pose_budget_guard_frame_mode", "non_trigger") or "non_trigger").strip().lower()
        if guard_frame_mode == "non_trigger":
            guard_frame_mask = (~hard_mask) if hard_any else torch.ones_like(hard_mask, dtype=torch.bool)
        else:
            guard_frame_mask = torch.ones_like(hard_mask, dtype=torch.bool)
        guard_frame_active = bool(guard_frame_mask[0, 0].detach().cpu().item())

        guard_joint_mask = torch.ones((num_joints,), dtype=torch.bool, device=theta.device)
        guard_excl_step = list(guard_excl_idxs_runtime)
        if not guard_excl_step:
            guard_excl_step = list(chain_idxs_runtime)
            if trig_joint_idx is not None:
                guard_excl_step.append(int(trig_joint_idx))
        for ii in guard_excl_step:
            if 0 <= int(ii) < num_joints:
                guard_joint_mask[int(ii)] = False
        if not bool(guard_joint_mask.any().detach().cpu().item()):
            guard_joint_mask = ~chain_joint_mask

        trigger_weighted = _finite_or_none((budget_stats or {}).get("direct_pose_budget_trigger_weighted"))
        chain_weighted = _finite_or_none((budget_stats or {}).get("direct_pose_budget_chain_weighted"))
        guard_weighted = _finite_or_none((budget_stats or {}).get("direct_pose_budget_guard_weighted"))
        budget_total_val = _finite_or_none((budget_stats or {}).get("direct_pose_budget_total_weighted"))
        budget_active = bool(
            budget_total_val is not None and abs(float(budget_total_val)) > float(max(0.0, budget_activation_thr))
        )
        if budget_active:
            budget_active_steps += 1

        branch_to_joint_idxs: Dict[str, List[int]] = {"trigger": [], "chain": [], "guard": []}
        if trigger_weighted is not None and abs(float(trigger_weighted)) > 1e-12 and trig_joint_idx is not None and hard_any:
            branch_to_joint_idxs["trigger"] = [int(trig_joint_idx)]
        if chain_weighted is not None and abs(float(chain_weighted)) > 1e-12 and chain_frame_active:
            branch_to_joint_idxs["chain"] = [int(i) for i in torch.nonzero(chain_joint_mask, as_tuple=False).flatten().tolist()]
        if guard_weighted is not None and abs(float(guard_weighted)) > 1e-12 and guard_frame_active:
            branch_to_joint_idxs["guard"] = [int(i) for i in torch.nonzero(guard_joint_mask, as_tuple=False).flatten().tolist()]

        for branch_name, weighted_val in (
            ("trigger", trigger_weighted),
            ("chain", chain_weighted),
            ("guard", guard_weighted),
        ):
            js = branch_to_joint_idxs.get(branch_name, [])
            if (not budget_active) or (weighted_val is None) or (abs(float(weighted_val)) <= 1e-12) or (not js):
                continue
            per_joint_structural = abs(float(weighted_val)) / float(len(js))
            for j in js:
                if not (0 <= int(j) < num_joints):
                    continue
                jj = int(j)
                budget_joint_count[branch_name][jj] += 1
                budget_joint_count["any"][jj] += 1
                budget_joint_weight[branch_name][jj] += float(per_joint_structural)
                budget_joint_weight["any"][jj] += float(per_joint_structural)

        step_row = {
            "step": int(step_idx),
            "sample_idx": int(idx),
            "main_loss": float(main_loss.detach().cpu().item()),
            "budget_loss": float(budget_loss.detach().cpu().item()),
            "trigger_loss": float(trig_loss.detach().cpu().item()) if torch.is_tensor(trig_loss) else None,
            "grad_main_norm": float(n_main),
            "grad_budget_norm": float(n_budget),
            "grad_ratio_budget_over_main": float(ratio),
            "grad_cos_budget_vs_main": float(cos),
            "trigger_n": _finite_or_none((trig_stats or {}).get("direct_pose_trigger_n")),
            "trigger_frac": _finite_or_none((trig_stats or {}).get("direct_pose_trigger_frac")),
            "budget_total_weighted": _finite_or_none((budget_stats or {}).get("direct_pose_budget_total_weighted")),
            "budget_share_trigger": _finite_or_none((budget_stats or {}).get("direct_pose_budget_share_trigger")),
            "budget_share_chain": _finite_or_none((budget_stats or {}).get("direct_pose_budget_share_chain")),
            "budget_share_guard": _finite_or_none((budget_stats or {}).get("direct_pose_budget_share_guard")),
            "budget_trigger_weighted": trigger_weighted,
            "budget_chain_weighted": chain_weighted,
            "budget_guard_weighted": guard_weighted,
            "budget_activation_thr": float(max(0.0, budget_activation_thr)),
            "budget_active": bool(budget_active),
            "budget_trigger_active_joint_count": int(len(branch_to_joint_idxs["trigger"])),
            "budget_chain_active_joint_count": int(len(branch_to_joint_idxs["chain"])),
            "budget_guard_active_joint_count": int(len(branch_to_joint_idxs["guard"])),
            "budget_runtime_chain_joint_count": int(len(chain_idxs_runtime)),
            "budget_runtime_guard_joint_count": int(guard_joint_mask.sum().detach().cpu().item()),
            "per_param": per_param,
        }
        rows.append(step_row)
        if math.isfinite(cos):
            overall_cos.append(cos)
        if math.isfinite(ratio):
            overall_ratio.append(ratio)

    per_param_summary = {}
    for n in direct_param_names:
        c = per_param_bucket[n]
        per_param_summary[n] = {
            "main_norm": _summary_stats(c["main_norm"]),
            "budget_norm": _summary_stats(c["budget_norm"]),
            "ratio": _summary_stats(c["ratio"]),
            "cos": _summary_stats(c["cos"]),
            "cos_neg_frac": _neg_frac(c["cos"]),
            "ratio_gt3_frac": _ratio_frac(c["ratio"], 3.0),
            "ratio_gt10_frac": _ratio_frac(c["ratio"], 10.0),
        }

    aggregate = {
        "steps": int(len(rows)),
        "grad_ratio_budget_over_main": _summary_stats(overall_ratio),
        "grad_cos_budget_vs_main": _summary_stats(overall_cos),
        "cos_neg_frac": _neg_frac(overall_cos),
        "ratio_gt3_frac": _ratio_frac(overall_ratio, 3.0),
        "ratio_gt10_frac": _ratio_frac(overall_ratio, 10.0),
        "per_param": per_param_summary,
    }

    joint_rows: List[Dict[str, Any]] = []
    total_structural = float(sum(float(x) for x in budget_joint_weight["any"]))
    denom_active = max(1, int(budget_active_steps))
    for j in range(num_joints):
        joint_rows.append(
            {
                "joint_idx": int(j),
                "joint_runtime": str(runtime_bone_names[j]),
                "joint_dataset": str(dataset_bone_names[j]),
                "budget_active_count": int(budget_joint_count["any"][j]),
                "budget_active_frac": float(budget_joint_count["any"][j] / float(denom_active)),
                "budget_structural_weight_sum": float(budget_joint_weight["any"][j]),
                "budget_structural_weight_share": (
                    float(budget_joint_weight["any"][j] / total_structural) if total_structural > 1e-12 else 0.0
                ),
                "branch_active_frac_trigger": float(budget_joint_count["trigger"][j] / float(denom_active)),
                "branch_active_frac_chain": float(budget_joint_count["chain"][j] / float(denom_active)),
                "branch_active_frac_guard": float(budget_joint_count["guard"][j] / float(denom_active)),
            }
        )
    joint_rows.sort(key=lambda r: float(r.get("budget_structural_weight_sum", 0.0)), reverse=True)

    chain_resolution = {
        "chain_spec": chain_spec,
        "guard_exclude_spec": guard_excl_spec,
        "runtime_bone_name_count": int(len(getattr(loss_fn, "bone_names", []) or [])),
        "dataset_bone_name_count": int(len(getattr(ds, "bone_names", []) or [])),
        "chain_idxs_runtime": [int(i) for i in chain_idxs_runtime],
        "chain_names_runtime": _joint_names_from_indices(runtime_bone_names, chain_idxs_runtime),
        "chain_idxs_if_dataset_names": [int(i) for i in chain_idxs_dataset],
        "chain_names_if_dataset_names": _joint_names_from_indices(dataset_bone_names, chain_idxs_dataset),
        "guard_excl_idxs_runtime": [int(i) for i in guard_excl_idxs_runtime],
        "guard_excl_idxs_if_dataset_names": [int(i) for i in guard_excl_idxs_dataset],
        "runtime_chain_missing_while_dataset_resolves": bool(len(chain_idxs_runtime) == 0 and len(chain_idxs_dataset) > 0),
    }
    aggregate["budget_joint_activation"] = {
        "budget_activation_thr": float(max(0.0, budget_activation_thr)),
        "budget_active_steps": int(budget_active_steps),
        "budget_active_frac": float(budget_active_steps / max(1, len(rows))),
        "joint_count": int(num_joints),
        "top_joints_by_structural_weight": joint_rows[: min(10, len(joint_rows))],
        "joint_rows": joint_rows,
    }
    aggregate["budget_chain_resolution"] = chain_resolution

    return {
        "name": arm.name,
        "runtime_config": str(arm.runtime_config),
        "ckpt": str(arm.ckpt),
        "lr": lr,
        "weight_decay": weight_decay,
        "direct_param_names": direct_param_names,
        "rows": rows,
        "aggregate": aggregate,
    }


def _render_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# direct_pose budget/main gradient diagnostics")
    lines.append("")
    lines.append(f"- generated_at: `{payload['generated_at']}`")
    lines.append(f"- teacher: `{payload['teacher']}`")
    lines.append(f"- steps_per_arm: `{payload['steps']}`")
    lines.append("")
    for arm in payload.get("arms", []):
        a = arm["aggregate"]
        lines.append(f"## Arm: `{arm['name']}`")
        lines.append("")
        lines.append(f"- runtime_config: `{arm['runtime_config']}`")
        lines.append(f"- ckpt: `{arm['ckpt']}`")
        lines.append(f"- optimizer: AdamW(lr={arm['lr']}, wd={arm['weight_decay']})")
        lines.append(
            f"- overall cosine<0 frac: `{a['cos_neg_frac']:.4f}` | ratio>3 frac: `{a['ratio_gt3_frac']:.4f}` | ratio>10 frac: `{a['ratio_gt10_frac']:.4f}`"
        )
        lines.append(
            "- overall ratio stats: "
            f"mean={a['grad_ratio_budget_over_main']['mean']:.4f}, "
            f"median={a['grad_ratio_budget_over_main']['median']:.4f}, "
            f"p90={a['grad_ratio_budget_over_main']['p90']:.4f}"
        )
        lines.append(
            "- overall cosine stats: "
            f"mean={a['grad_cos_budget_vs_main']['mean']:.4f}, "
            f"median={a['grad_cos_budget_vs_main']['median']:.4f}, "
            f"p90={a['grad_cos_budget_vs_main']['p90']:.4f}"
        )
        bj = a.get("budget_joint_activation", {})
        if isinstance(bj, dict):
            lines.append(
                f"- budget_active_frac(>|budget_total|>{bj.get('budget_activation_thr', float('nan')):.2e}): "
                f"`{bj.get('budget_active_frac', float('nan')):.4f}` "
                f"({int(bj.get('budget_active_steps', 0))}/{a.get('steps', 0)} steps)"
            )
        bcr = a.get("budget_chain_resolution", {})
        if isinstance(bcr, dict):
            lines.append(
                f"- chain_joint_resolution: runtime={bcr.get('chain_idxs_runtime', [])} "
                f"| if_dataset_names={bcr.get('chain_idxs_if_dataset_names', [])} "
                f"| runtime_bone_names={bcr.get('runtime_bone_name_count', 0)} "
                f"| dataset_bone_names={bcr.get('dataset_bone_name_count', 0)} "
                f"| runtime_chain_missing_while_dataset_resolves={bool(bcr.get('runtime_chain_missing_while_dataset_resolves', False))}"
            )
        lines.append("")
        lines.append("| tensor | cos_neg_frac | ratio>3 frac | ratio>10 frac | ratio mean | ratio p90 | cos mean | cos median |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for n, s in a["per_param"].items():
            lines.append(
                f"| {n} | {s['cos_neg_frac']:.4f} | {s['ratio_gt3_frac']:.4f} | {s['ratio_gt10_frac']:.4f} | "
                f"{s['ratio']['mean']:.4f} | {s['ratio']['p90']:.4f} | {s['cos']['mean']:.4f} | {s['cos']['median']:.4f} |"
            )
        lines.append("")
        topj = None
        if isinstance(bj, dict):
            topj = bj.get("top_joints_by_structural_weight", None)
        if isinstance(topj, list) and topj:
            lines.append("| joint(dataset) | idx | active_frac | structural_share | trig_frac | chain_frac | guard_frac |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in topj:
                lines.append(
                    f"| {row.get('joint_dataset', row.get('joint_runtime', 'NA'))} | {int(row.get('joint_idx', -1))} | "
                    f"{float(row.get('budget_active_frac', float('nan'))):.4f} | "
                    f"{float(row.get('budget_structural_weight_share', float('nan'))):.4f} | "
                    f"{float(row.get('branch_active_frac_trigger', float('nan'))):.4f} | "
                    f"{float(row.get('branch_active_frac_chain', float('nan'))):.4f} | "
                    f"{float(row.get('branch_active_frac_guard', float('nan'))):.4f} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnose direct_pose_head gradient conflict between main pose loss and budget loss."
    )
    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-activation-thr", type=float, default=1e-8, help="Budget step active threshold on |budget_total_weighted|.")
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        help="Format: name,runtime_json,ckpt (repeatable).",
    )
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--out-md", type=str, default="")
    args = ap.parse_args()

    if not args.arm:
        raise SystemExit("[FATAL] At least one --arm is required.")

    teacher = Path(args.teacher).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()
    bundle = Path(args.bundle).expanduser().resolve()
    pretrain_template = Path(args.pretrain_template).expanduser().resolve()
    encoder_bundle = Path(args.encoder_bundle).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve() if str(args.out_md or "").strip() else out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    arms = [_parse_arm(x) for x in args.arm]
    results = []
    for arm in arms:
        if not arm.runtime_config.is_file():
            raise SystemExit(f"[FATAL] runtime config not found: {arm.runtime_config}")
        if not arm.ckpt.is_file():
            raise SystemExit(f"[FATAL] ckpt not found: {arm.ckpt}")
        print(f"[RUN] arm={arm.name}")
        res = _arm_run(
            arm=arm,
            teacher_path=teacher,
            npz_root=npz_root,
            bundle=bundle,
            pretrain_template=pretrain_template,
            encoder_bundle=encoder_bundle,
            device=str(args.device),
            depth=int(args.depth),
            num_heads=int(args.num_heads),
            dropout=float(args.dropout),
            context_len=int(args.context_len),
            steps=int(args.steps),
            seed=int(args.seed),
            budget_activation_thr=float(args.budget_activation_thr),
        )
        results.append(res)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "teacher": str(teacher),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "arms": results,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    out_md.write_text(_render_md(payload))
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
