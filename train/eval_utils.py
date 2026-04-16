from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch

from .diagnostics import (
    aggregate_metric_samples,
    attach_delta_energy_metrics,
    collect_freerun_step_debug_record,
    diagnose_free_run,
    save_freerun_debug_payload,
)


@dataclass
class FreeRunSettings:
    """Configuration for free-run evaluation."""

    warmup_steps: int = 0
    horizon: Optional[int] = None
    max_batches: int = 8


def evaluate_teacher(
    trainer,
    loader: Iterable[Dict[str, torch.Tensor]],
    *,
    mode: str = "mixed",
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Teacher forcing评估：输出均值loss并复用自由评估的诊断统计。"""
    _ = mode
    device = trainer.device
    total_loss = 0.0
    count = 0
    stats_accum: Dict[str, list[Any]] = {}

    def _split_steps(t: Optional[torch.Tensor]) -> list[torch.Tensor]:
        if t is None or not isinstance(t, torch.Tensor) or t.dim() < 2:
            return []
        return [t[:, i] for i in range(t.shape[1])]

    trainer._diag_scope = "single_step"
    try:
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            x_cand = trainer._pick_first(batch, ("motion", "X", "x_in_features"))
            y_cand = trainer._pick_first(batch, ("gt_motion", "Y", "y_out_features", "y_out_seq"))
            if x_cand is None or y_cand is None:
                continue

            state_seq = x_cand.to(device).float()
            gt_seq = y_cand.to(device).float()
            cond_seq = batch.get("cond_in")
            cond_seq = cond_seq.to(device).float() if cond_seq is not None else None
            cond_raw_seq = batch.get("cond_tgt_raw")
            cond_raw_seq = cond_raw_seq.to(device).float() if cond_raw_seq is not None else None
            contacts_seq = batch.get("contacts")
            contacts_seq = contacts_seq.to(device).float() if contacts_seq is not None else None
            angvel_seq = batch.get("angvel")
            angvel_seq = angvel_seq.to(device).float() if angvel_seq is not None else None
            angvel_raw_seq = batch.get("angvel_raw")
            angvel_raw_seq = angvel_raw_seq.to(device).float() if angvel_raw_seq is not None else None
            pose_hist_seq = batch.get("pose_hist")
            pose_hist_seq = pose_hist_seq.to(device).float() if pose_hist_seq is not None else None
            cond_norm_mu = batch.get("cond_norm_mu")
            cond_norm_mu = cond_norm_mu.to(device).float() if cond_norm_mu is not None else None
            cond_norm_std = batch.get("cond_norm_std")
            cond_norm_std = cond_norm_std.to(device).float() if cond_norm_std is not None else None

            preds_dict, last_attn = trainer._rollout_sequence(
                state_seq,
                cond_seq,
                cond_raw_seq,
                contacts_seq=contacts_seq,
                angvel_seq=angvel_seq,
                pose_hist_seq=pose_hist_seq,
                gt_seq=gt_seq,
                cond_norm_mu=cond_norm_mu,
                cond_norm_std=cond_norm_std,
                mode="mixed",
                tf_ratio=1.0,
            )
            out = trainer.loss_fn(preds_dict, gt_seq, attn_weights=last_attn, batch=batch)
            loss = out[0] if isinstance(out, tuple) else out
            loss_val = float(loss.detach().cpu())
            total_loss += loss_val
            count += 1

            predY = preds_dict.get("out") if isinstance(preds_dict, dict) else None
            if predY is None:
                stats_accum.setdefault("TeacherLoss", []).append(loss_val)
                continue

            steps = predY.shape[1]
            diag = diagnose_free_run(
                trainer,
                batch=batch,
                predY=predY,
                gtY=gt_seq[:, :steps],
                predsX=[],
                period_seq_pred=_split_steps(preds_dict.get("period_pred")),
                motion_seq=state_seq[:, :steps],
                y_seq=gt_seq,
                contacts_seq=contacts_seq,
                angvel_seq=angvel_seq,
                pose_hist_seq=pose_hist_seq,
                angvel_raw_seq=angvel_raw_seq,
            )
            if isinstance(diag, dict):
                for key, value in diag.items():
                    stats_accum.setdefault(key, []).append(value)
            stats_accum.setdefault("TeacherLoss", []).append(loss_val)
    finally:
        if hasattr(trainer, "_diag_scope"):
            delattr(trainer, "_diag_scope")

    mean_loss = total_loss / max(1, count)
    summary = aggregate_metric_samples(stats_accum)
    summary.setdefault("TeacherLoss", mean_loss)
    summary["loss"] = mean_loss
    summary["batches"] = count
    summary["phase"] = "teacher"
    summary["tf_ratio"] = float(getattr(trainer, "_last_tf_ratio", 1.0))
    return summary


def evaluate_freerun(
    trainer,
    loader: Iterable[Dict[str, torch.Tensor]],
    settings: FreeRunSettings,
) -> Dict[str, Any]:
    """
    Run autoregressive (free-run) evaluation with optional warmup and finite horizon.

    Returns:
        Aggregated diagnostics averaged over processed batches.
    """
    device = trainer.device
    model = trainer.model
    stats_accum: Dict[str, list[Any]] = {}
    base_debug_path = getattr(trainer, "freerun_debug_path", None)
    batches_processed = 0

    trainer._diag_scope = "free_run"
    try:
        iterator = iter(loader)
        while batches_processed < settings.max_batches:
            try:
                batch = next(iterator)
            except StopIteration:
                break

            state_seq = trainer._pick_first(batch, ("motion", "X", "x_in_features"))
            y_seq = trainer._pick_first(batch, ("gt_motion", "Y", "y_out_features", "y_out_seq"))
            if state_seq is None or y_seq is None:
                continue

            state_seq = state_seq.to(device).float()
            gt_seq = y_seq.to(device).float()
            cond_seq = batch.get("cond_in")
            cond_seq = cond_seq.to(device).float() if cond_seq is not None else None
            cond_seq_raw = batch.get("cond_tgt_raw")
            cond_seq_raw = cond_seq_raw.to(device).float() if cond_seq_raw is not None else None
            contacts_seq = batch.get("contacts")
            contacts_seq = contacts_seq.to(device).float() if contacts_seq is not None else None
            angvel_seq = batch.get("angvel")
            angvel_seq = angvel_seq.to(device).float() if angvel_seq is not None else None
            pose_hist_seq = batch.get("pose_hist")
            pose_hist_seq = pose_hist_seq.to(device).float() if pose_hist_seq is not None else None
            cond_norm_mu = batch.get("cond_norm_mu")
            cond_norm_mu = cond_norm_mu.to(device).float() if cond_norm_mu is not None else None
            cond_norm_std = batch.get("cond_norm_std")
            cond_norm_std = cond_norm_std.to(device).float() if cond_norm_std is not None else None
            cond_norm_mu = trainer._prepare_cond_stat(cond_norm_mu, state_seq) if cond_norm_mu is not None else None
            cond_norm_std = trainer._prepare_cond_stat(cond_norm_std, state_seq) if cond_norm_std is not None else None

            _, total_steps, _ = state_seq.shape
            if total_steps < 2:
                continue

            warmup = max(0, min(int(settings.warmup_steps), total_steps - 1))
            horizon = total_steps - 1 if settings.horizon is None else max(0, min(int(settings.horizon), total_steps - 1))
            start_t = warmup
            end_t = min(total_steps - 1, warmup + horizon)
            if end_t <= start_t:
                continue

            predsY = []
            predsX = []
            period_seq_pred = []
            diag_records: list[dict[str, Any]] = []
            tf_ratio = float(getattr(trainer, "_last_tf_ratio", 1.0))
            enable_reprojection = bool(getattr(trainer, "enable_cond_reprojection", True))

            time_base = None
            try:
                base = batch.get("start", None) if isinstance(batch, dict) else None
                if base is not None and torch.is_tensor(base):
                    base = base.to(device=device)
                time_base = base
            except Exception:
                time_base = None

            motion = state_seq[:, start_t]
            motion_raw = None
            if getattr(trainer, "normalizer", None) is not None:
                try:
                    motion_raw = trainer.normalizer.denorm_x(motion)
                except Exception:
                    motion_raw = None

            try:
                y_raw_prev = trainer._denorm(gt_seq[:, start_t])
            except Exception:
                y_raw_prev = None
            if y_raw_prev is None and motion_raw is not None:
                rot6d_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
                if isinstance(rot6d_slice, slice):
                    slice_len = rot6d_slice.stop - rot6d_slice.start
                    if slice_len == gt_seq.shape[-1]:
                        try:
                            y_raw_prev = motion_raw[:, rot6d_slice].clone()
                        except Exception:
                            y_raw_prev = None

            gt_motion_raw = motion_raw.clone() if motion_raw is not None else None
            if warmup > 0:
                motion_raw_tmp = motion_raw
                try:
                    if motion_raw_tmp is None and getattr(trainer, "normalizer", None) is not None:
                        motion_raw_tmp = trainer.normalizer.denorm_x(state_seq[:, 0])
                    for t in range(1, warmup + 1):
                        src = state_seq[:, t]
                        if getattr(trainer, "normalizer", None) is not None:
                            try:
                                motion_raw_tmp = trainer.normalizer.denorm_x(src, prev_raw=motion_raw_tmp)
                            except Exception:
                                motion_raw_tmp = None
                        motion = src
                    if motion_raw_tmp is not None:
                        motion_raw = motion_raw_tmp
                except Exception:
                    pass

            plan_z = None
            for t in range(start_t, end_t):
                cond_input = cond_seq[:, t] if (cond_seq is not None and cond_seq.dim() == 3) else cond_seq
                contacts_t = contacts_seq[:, t] if (contacts_seq is not None and contacts_seq.dim() == 3) else contacts_seq
                if getattr(trainer, "use_freerun_state_sync", False) and isinstance(getattr(trainer, "angvel_x_slice", None), slice):
                    angvel_t = motion[..., trainer.angvel_x_slice].detach()
                else:
                    angvel_t = angvel_seq[:, t] if (angvel_seq is not None and angvel_seq.dim() == 3) else angvel_seq
                pose_hist_t = pose_hist_seq[:, t] if (pose_hist_seq is not None and pose_hist_seq.dim() == 3) else pose_hist_seq
                gt_motion_next = state_seq[:, t + 1]
                if gt_motion_raw is not None:
                    try:
                        gt_motion_raw = trainer.normalizer.denorm_x(gt_motion_next, prev_raw=gt_motion_raw)
                    except Exception:
                        gt_motion_raw = None

                cond_raw_step = None
                if cond_seq_raw is not None:
                    cond_raw_step = cond_seq_raw[:, min(cond_seq_raw.shape[1] - 1, t + 1)] if cond_seq_raw.dim() == 3 else cond_seq_raw

                cond_raw_for_model = cond_raw_step
                if enable_reprojection and t > 0 and cond_raw_step is not None:
                    gt_yaw = None
                    try:
                        gt_idx = min(gt_seq.shape[1] - 1, t)
                        gt_raw_frame = trainer._denorm(gt_seq[:, gt_idx])
                        gt_yaw = trainer._infer_root_yaw_from_rot6d(gt_raw_frame)
                    except Exception:
                        gt_yaw = None
                    if gt_yaw is None and state_seq is not None:
                        try:
                            state_raw = trainer.normalizer.denorm_x(state_seq[:, t], prev_raw=motion_raw)
                            gt_yaw = trainer._infer_root_yaw_from_rot6d(state_raw)
                        except Exception:
                            gt_yaw = None
                    pred_yaw = None
                    if y_raw_prev is not None:
                        try:
                            pred_yaw = trainer._infer_root_yaw_from_rot6d(y_raw_prev)
                        except Exception:
                            pred_yaw = None
                    if gt_yaw is not None and pred_yaw is not None:
                        cond_proj = trainer._reproject_cond_to_local_frame(cond_raw_step, gt_yaw, pred_yaw)
                        if cond_proj is not None:
                            cond_raw_for_model = cond_proj

                if cond_raw_for_model is not None:
                    cond_override = trainer._normalize_cond_from_raw(cond_raw_for_model, cond_norm_mu, cond_norm_std)
                    if cond_override is not None:
                        cond_input = cond_override

                device_type = getattr(device, "type", "cpu")
                if device_type == "mps":
                    amp_ctx = torch.autocast(device_type="mps", dtype=torch.float16, enabled=getattr(trainer, "use_amp", False))
                elif device_type == "cuda":
                    amp_ctx = torch.amp.autocast("cuda", enabled=getattr(trainer, "use_amp", False))
                else:
                    amp_ctx = nullcontext()

                time_index_t = int(t)
                if time_base is not None:
                    try:
                        time_index_t = time_base + int(t)
                    except Exception:
                        time_index_t = int(t)

                rollout_step_t = None
                try:
                    denom = int(horizon - warmup - 1)
                    step_norm = float(int(t - warmup)) / float(denom) if denom > 0 else 0.0
                    rollout_step_t = torch.full((motion.shape[0], 1, 1), step_norm, device=device, dtype=motion.dtype)
                except Exception:
                    rollout_step_t = None

                contacts_in_t = contacts_t
                if bool(getattr(model, "contact_plan_enable", False)):
                    try:
                        contacts_in_t = trainer._predict_pretrain_contacts_from_frozen(
                            motion_step_t=motion_raw,
                            pose_hist_step_t=pose_hist_t,
                        )
                    except Exception:
                        contacts_in_t = None

                with amp_ctx:
                    ret = model(
                        motion,
                        cond_input,
                        contacts=contacts_in_t,
                        angvel=angvel_t,
                        pose_history=pose_hist_t,
                        plan_z=plan_z,
                        time_index=time_index_t,
                        rollout_step=rollout_step_t,
                    )

                if not isinstance(ret, dict):
                    raise RuntimeError("Model forward must return a dict with at least 'out'.")
                out = ret.get("out")
                period_pred = ret.get("period_pred")
                if bool(getattr(model, "contact_plan_enable", False)):
                    try:
                        z_next = ret.get("plan_z_next", None)
                        if z_next is not None:
                            plan_z = z_next.detach()
                    except Exception:
                        pass
                if out is None:
                    break

                delta_norm = out
                if y_raw_prev is not None:
                    try:
                        y_inc_raw = trainer._compose_delta_to_raw(y_raw_prev, delta_norm)
                        y_raw = y_inc_raw
                        if bool(getattr(trainer, "lambda_fusion_apply", False)):
                            lam_eff = ret.get("lambda_fusion", None)
                            if lam_eff is not None:
                                try:
                                    lam_eff, _ = trainer._lambda_fusion_apply_reliability(
                                        lam_eff,
                                        step_idx=int(t - warmup),
                                        total_steps=int(max(1, int(horizon - warmup - 1))),
                                        rollout_step=rollout_step_t,
                                        ret=ret,
                                    )
                                except Exception:
                                    lam_eff = ret.get("lambda_fusion", None)
                            y_raw = trainer._apply_lambda_fusion_to_raw(
                                y_inc_raw,
                                direct_norm=ret.get("out_direct", None),
                                lambda_fusion=lam_eff,
                            )
                    except Exception:
                        y_raw = trainer._denorm(delta_norm)
                else:
                    y_raw = trainer._denorm(delta_norm)
                y_raw_prev = y_raw.detach()

                try:
                    y_norm = trainer._norm_y(y_raw)
                except Exception:
                    y_norm = delta_norm

                predsY.append(y_norm)
                if period_pred is not None:
                    period_seq_pred.append(period_pred)

                if motion_raw is not None:
                    motion_raw = trainer._apply_free_carry(motion_raw, y_raw, cond_next_raw=cond_raw_step).detach()
                    motion = trainer._diag_norm_x(motion_raw)
                else:
                    motion = trainer._apply_free_carry(motion, y_raw, cond_next_raw=None).detach()
                predsX.append(motion)

                rec = collect_freerun_step_debug_record(
                    trainer,
                    step_idx=int(t - start_t),
                    motion_raw=motion_raw,
                    gt_motion_raw=gt_motion_raw,
                    cond_raw_step=cond_raw_step,
                    delta_norm=delta_norm,
                )
                if rec:
                    diag_records.append(rec)

            if not predsY:
                continue

            predY = torch.stack(predsY, dim=1)
            free_steps = predY.shape[1]
            gt_start = start_t
            gt_end = gt_start + free_steps
            gtY = gt_seq[:, gt_start:gt_end]
            motion_ref = state_seq[:, gt_start:gt_end + 1]

            batch_stats = diagnose_free_run(
                trainer,
                batch=batch,
                predY=predY,
                gtY=gtY,
                predsX=predsX,
                period_seq_pred=period_seq_pred,
                motion_seq=motion_ref,
                y_seq=gt_seq,
                contacts_seq=contacts_seq,
                angvel_seq=angvel_seq,
                pose_hist_seq=pose_hist_seq,
            )
            if batch_stats is None:
                continue

            attach_delta_energy_metrics(batch_stats, diag_records)
            save_freerun_debug_payload(
                trainer,
                batch=batch,
                batch_stats=batch_stats,
                diag_records=diag_records,
                batches_processed=batches_processed,
                warmup=warmup,
                horizon=int(end_t - start_t),
                tf_ratio=tf_ratio,
                base_debug_path=base_debug_path,
            )

            for key, value in batch_stats.items():
                stats_accum.setdefault(key, []).append(value)
            batches_processed += 1
    finally:
        if hasattr(trainer, "_diag_scope"):
            delattr(trainer, "_diag_scope")

    defaults: Dict[str, Any] = {
        "MSEnormY": float("nan"),
        "GeoDeg": float("nan"),
        "GeoLocalDeg": float("nan"),
        "RootVelMAE": float("nan"),
        "AngVelMAE": float("nan"),
        "AngVelMagRel": float("nan"),
        "KeyBoneSummary": {},
        "KeyBoneDetails": {},
    }
    summary = aggregate_metric_samples(stats_accum, defaults=defaults)
    summary["phase"] = "free_run"
    summary["tf_ratio"] = float(getattr(trainer, "_last_tf_ratio", 1.0))
    summary["batches"] = batches_processed
    return summary
