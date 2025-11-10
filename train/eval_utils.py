from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import torch


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
    mode: str = "teacher",
) -> Dict[str, Any]:
    """Teacher forcing评估：输出均值loss并复用自由评估的诊断统计。"""
    device = trainer.device
    total_loss = 0.0
    count = 0
    stats_accum: Dict[str, list[Any]] = {}

    def _split_steps(t: Optional[torch.Tensor]) -> list[torch.Tensor]:
        if t is None:
            return []
        if not isinstance(t, torch.Tensor):
            return []
        if t.dim() < 2:
            return []
        return [t[:, i] for i in range(t.shape[1])]

    trainer._diag_scope = 'single_step'
    try:
        for batch in loader:
            x_cand = trainer._pick_first(batch, ("motion", "X", "x_in_features"))
            y_cand = trainer._pick_first(batch, ("gt_motion", "Y", "y_out_features", "y_out_seq"))
            if x_cand is None or y_cand is None:
                continue

            state_seq = x_cand.to(device).float()
            gt_seq = y_cand.to(device).float()

            cond_seq = batch.get("cond_in")
            if cond_seq is not None:
                cond_seq = cond_seq.to(device).float()
            cond_raw_seq = batch.get("cond_tgt_raw")
            if cond_raw_seq is not None:
                cond_raw_seq = cond_raw_seq.to(device).float()
            contacts_seq = batch.get("contacts")
            if contacts_seq is not None:
                contacts_seq = contacts_seq.to(device).float()
            angvel_seq = batch.get("angvel")
            if angvel_seq is not None:
                angvel_seq = angvel_seq.to(device).float()
            angvel_raw_seq = batch.get("angvel_raw")
            if angvel_raw_seq is not None:
                angvel_raw_seq = angvel_raw_seq.to(device).float()
            pose_hist_seq = batch.get("pose_hist")
            if pose_hist_seq is not None:
                pose_hist_seq = pose_hist_seq.to(device).float()

            preds_dict, last_attn = trainer._rollout_sequence(
                state_seq,
                cond_seq,
                cond_raw_seq,
                contacts_seq=contacts_seq,
                angvel_seq=angvel_seq,
                pose_hist_seq=pose_hist_seq,
                gt_seq=gt_seq,
                mode=mode,
                tf_ratio=1.0,
            )
            out = trainer.loss_fn(preds_dict, gt_seq, attn_weights=last_attn, batch=batch)
            loss = out[0] if isinstance(out, tuple) else out
            loss_val = float(loss.detach().cpu())
            total_loss += loss_val
            count += 1

            predY = preds_dict.get("out") if isinstance(preds_dict, dict) else None
            if predY is None:
                continue

            steps = predY.shape[1]
            mse_norm = torch.mean((predY - gt_seq[:, :steps]) ** 2).item()

            diag = trainer._diagnose_free_run(
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

            diag.setdefault("MSEnormY", mse_norm)
            for key, value in diag.items():
                stats_accum.setdefault(key, []).append(value)
            stats_accum.setdefault("TeacherLoss", []).append(loss_val)
    finally:
        if hasattr(trainer, '_diag_scope'):
            delattr(trainer, '_diag_scope')

    def _avg_simple_dict(dict_list):
        totals = {}
        counts = {}
        for item in dict_list:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if isinstance(v, (int, float)):
                    totals[k] = totals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: (totals[k] / max(1, counts[k])) for k in totals}

    def _avg_nested_dict(dict_list):
        result = {}
        counts = {}
        for item in dict_list:
            if not isinstance(item, dict):
                continue
            for bone, metrics in item.items():
                if not isinstance(metrics, dict):
                    continue
                sub_res = result.setdefault(bone, {})
                sub_cnt = counts.setdefault(bone, {})
                for mk, mv in metrics.items():
                    if isinstance(mv, (int, float)):
                        sub_res[mk] = sub_res.get(mk, 0.0) + float(mv)
                        sub_cnt[mk] = sub_cnt.get(mk, 0) + 1
        for bone, metrics in result.items():
            cnt = counts.get(bone, {})
            for mk in list(metrics.keys()):
                metrics[mk] = metrics[mk] / max(1, cnt.get(mk, 1))
        return result

    summary: Dict[str, Any] = {}
    for key, values in stats_accum.items():
        if not values:
            continue
        sample = values[0]
        if isinstance(sample, dict):
            if key == "KeyBoneDetails":
                summary[key] = _avg_nested_dict(values)
            else:
                summary[key] = _avg_simple_dict(values)
            continue
        try:
            summary[key] = float(sum(values) / max(1, len(values)))
        except Exception:
            summary[key] = values

    mean_loss = total_loss / max(1, count)
    summary.setdefault("TeacherLoss", mean_loss)
    summary["loss"] = mean_loss
    summary["batches"] = count
    summary["phase"] = "teacher"
    summary["tf_ratio"] = float(getattr(trainer, "_last_tf_ratio", 1.0))

    if hasattr(trainer, '_diag_scope'):
        delattr(trainer, '_diag_scope')
    return summary


def evaluate_freerun(
    trainer,
    loader: Iterable[Dict[str, torch.Tensor]],
    settings: FreeRunSettings,
) -> Dict[str, Any]:
    """
    Run autoregressive (free-run) evaluation with optional warmup and finite horizon.

    Args:
        trainer: Active ``Trainer`` instance (provides model, normalizer, diagnostics).
        loader: Iterable loader yielding batches compatible with ``_rollout_sequence``.
        settings: ``FreeRunSettings`` controlling warmup frames, eval horizon, and the number of batches.

    Returns:
        Dict[str, Any]: Aggregated diagnostics averaged over processed batches. Keys expected by the
        caller (``MSEnormY``, ``GeoDeg`` etc.) are always populated; missing quantities fallback to ``nan``.
    """
    device = trainer.device
    model = trainer.model

    stats_accum = {}
    base_debug_path = getattr(trainer, "freerun_debug_path", None)
    last_debug_path = None
    debug_path = base_debug_path  # used as a simple flag before we resolve epoch-specific suffix
    debug_saved = False
    batches_processed = 0

    trainer._diag_scope = 'free_run'
    it = iter(loader)
    while batches_processed < settings.max_batches:
        if hasattr(trainer, "_carry_debug_buffer"):
            trainer._carry_debug_buffer = []
        try:
            batch = next(it)
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

        B, T, Dx = state_seq.shape
        if T < 2:
            continue

        warmup = max(0, min(int(settings.warmup_steps), T - 1))
        horizon = settings.horizon
        if horizon is None:
            horizon = T - 1
        horizon = max(0, min(int(horizon), T - 1))

        predsY = []
        predsX = []
        period_seq_pred = []
        diag_records: list[dict[str, Any]] = []

        tf_ratio = getattr(trainer, "_last_tf_ratio", 1.0)

        start_t = warmup
        end_t = min(T - 1, warmup + horizon)
        if end_t <= start_t:
            continue

        motion = state_seq[:, start_t]
        motion_raw = None
        if hasattr(trainer, "normalizer") and trainer.normalizer is not None:
            try:
                motion_raw = trainer.normalizer.denorm_x(motion)
            except Exception:
                motion_raw = None

        y_raw_prev = None
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

        # Teacher warmup to align raw hidden state if requested
        if warmup > 0:
            # Reconstruct raw state progressively to avoid discontinuities
            motion_raw_tmp = motion_raw
            try:
                if motion_raw_tmp is None and hasattr(trainer, "normalizer") and trainer.normalizer is not None:
                    motion_raw_tmp = trainer.normalizer.denorm_x(state_seq[:, 0])
                for t in range(1, warmup + 1):
                    src = state_seq[:, t]
                    if hasattr(trainer, "normalizer") and trainer.normalizer is not None:
                        try:
                            motion_raw_tmp = trainer.normalizer.denorm_x(src, prev_raw=motion_raw_tmp)
                        except Exception:
                            motion_raw_tmp = None
                    motion = src
                if motion_raw_tmp is not None:
                    motion_raw = motion_raw_tmp
            except Exception:
                pass

        for t in range(start_t, end_t):
            cond_t = cond_seq[:, t] if (cond_seq is not None and cond_seq.dim() == 3) else cond_seq
            contacts_t = contacts_seq[:, t] if (contacts_seq is not None and contacts_seq.dim() == 3) else contacts_seq
            if getattr(trainer, 'use_freerun_state_sync', False) and isinstance(getattr(trainer, 'angvel_x_slice', None), slice):
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

            _devt = getattr(device, "type", "cpu")
            if _devt == "mps":
                amp_ctx = torch.autocast(device_type="mps", dtype=torch.float16, enabled=getattr(trainer, "use_amp", False))
            elif _devt == "cuda":
                amp_ctx = torch.amp.autocast("cuda", enabled=getattr(trainer, "use_amp", False))
            else:
                from contextlib import nullcontext
                amp_ctx = nullcontext()

            with amp_ctx:
                ret = model(
                    motion,
                    cond_t,
                    contacts=contacts_t,
                    angvel=angvel_t,
                    pose_history=pose_hist_t,
                )

            if not isinstance(ret, dict):
                raise RuntimeError("Model forward must return a dict with at least 'out'.")
            out = ret.get("out")
            period_pred = ret.get("period_pred")

            if out is None:
                break

            delta_norm = out
            if y_raw_prev is not None:
                try:
                    y_raw = trainer._compose_delta_to_raw(y_raw_prev, delta_norm)
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

            cond_next_raw = None
            if cond_seq_raw is not None:
                if cond_seq_raw.dim() == 3:
                    idx = min(cond_seq_raw.shape[1] - 1, t + 1)
                    cond_next_raw = cond_seq_raw[:, idx]
                else:
                    cond_next_raw = cond_seq_raw
            if motion_raw is not None:
                motion_raw = trainer._apply_free_carry(motion_raw, y_raw, cond_next_raw=cond_next_raw).detach()
                motion = trainer._diag_norm_x(motion_raw)
            else:
                motion = trainer._apply_free_carry(motion, y_raw, cond_next_raw=None).detach()

            predsX.append(motion)

            if debug_path and (not debug_saved):
                yaw_sl = getattr(trainer, 'yaw_x_slice', None)
                rootvel_sl = getattr(trainer, 'rootvel_x_slice', None)
                rot6d_sl = getattr(trainer, 'rot6d_y_slice', None) or getattr(trainer, 'rot6d_slice', None)
                rec: dict[str, Any] = {"step": int(t - start_t)}
                yaw_pred_scalar: Optional[torch.Tensor] = None
                yaw_gt_scalar: Optional[torch.Tensor] = None
                if isinstance(yaw_sl, slice) and motion_raw is not None and gt_motion_raw is not None:
                    dyaw = torch.atan2(
                        torch.sin(motion_raw[..., yaw_sl] - gt_motion_raw[..., yaw_sl]),
                        torch.cos(motion_raw[..., yaw_sl] - gt_motion_raw[..., yaw_sl])
                    )
                    rec["yaw_abs_deg"] = float(dyaw.abs().mean().item() * (180.0 / torch.pi))
                    if motion_raw.shape[0] > 0:
                        yaw_pred_scalar = motion_raw[0, yaw_sl].mean()
                        rec["yaw_pred_s0"] = float(yaw_pred_scalar.item())
                    if gt_motion_raw.shape[0] > 0:
                        yaw_gt_scalar = gt_motion_raw[0, yaw_sl].mean()
                        rec["yaw_gt_s0"] = float(yaw_gt_scalar.item())
                if isinstance(rootvel_sl, slice) and motion_raw is not None and gt_motion_raw is not None:
                    rv_err = (motion_raw[..., rootvel_sl] - gt_motion_raw[..., rootvel_sl]).abs().mean().item()
                    rec["root_vel_mae"] = float(rv_err)
                    if motion_raw.shape[0] > 0:
                        rec["root_vel_pred_s0"] = motion_raw[0, rootvel_sl].detach().cpu().tolist()
                    if gt_motion_raw.shape[0] > 0:
                        rec["root_vel_gt_s0"] = gt_motion_raw[0, rootvel_sl].detach().cpu().tolist()
                yaw_cmd_scalar: Optional[torch.Tensor] = None
                if cond_next_raw is not None and torch.is_tensor(cond_next_raw) and cond_next_raw.shape[0] > 0:
                    cond0 = cond_next_raw[0].detach()
                    rec["cond_next_raw_s0"] = cond0.cpu().tolist()
                    cond_dim = cond0.shape[0]
                    dir_vec = None
                    speed_val: Optional[torch.Tensor] = None
                    if cond_dim >= 3:
                        action_dim = max(0, cond_dim - 3)
                        dir_vec = cond0[action_dim:action_dim + 2]
                        speed_val = cond0[action_dim + 2]
                    elif cond_dim >= 2:
                        dir_vec = cond0[-2:]
                    if dir_vec is not None and dir_vec.numel() == 2:
                        dir_norm = dir_vec.norm().clamp_min(1e-6)
                        dir_unit = dir_vec / dir_norm
                        yaw_cmd_scalar = torch.atan2(dir_unit[1], dir_unit[0])
                        rec["yaw_cmd_s0"] = float(yaw_cmd_scalar.item())
                        rec["yaw_cmd_deg"] = float(yaw_cmd_scalar.item() * (180.0 / torch.pi))
                    if speed_val is not None:
                        rec["yaw_cmd_speed"] = float(speed_val.item())
                    elif dir_vec is not None:
                        rec["yaw_cmd_speed"] = float(dir_vec.norm().item())
                if yaw_cmd_scalar is not None and yaw_pred_scalar is not None:
                    yaw_diff = torch.atan2(
                        torch.sin(yaw_pred_scalar - yaw_cmd_scalar),
                        torch.cos(yaw_pred_scalar - yaw_cmd_scalar),
                    )
                    rec["yaw_cmd_vs_pred_deg"] = float(yaw_diff.abs().item() * (180.0 / torch.pi))
                if yaw_cmd_scalar is not None and yaw_gt_scalar is not None:
                    yaw_diff_gt = torch.atan2(
                        torch.sin(yaw_gt_scalar - yaw_cmd_scalar),
                        torch.cos(yaw_gt_scalar - yaw_cmd_scalar),
                    )
                    rec["yaw_cmd_vs_gt_deg"] = float(yaw_diff_gt.abs().item() * (180.0 / torch.pi))
                rec["delta_norm_abs"] = float(delta_norm.abs().mean().item())
                if rec:
                    diag_records.append(rec)

        carry_debug = getattr(trainer, "_carry_debug_buffer", None)
        if carry_debug:
            for rec, extra in zip(diag_records, carry_debug):
                for k, v in extra.items():
                    rec[f"carry/{k}"] = v

        if not predsY:
            continue

        predY = torch.stack(predsY, dim=1)
        free_steps = predY.shape[1]
        gt_start = start_t
        gt_end = gt_start + free_steps
        gtY = gt_seq[:, gt_start:gt_end]
        mse_normY = torch.mean((predY - gtY) ** 2).item()

        motion_ref = state_seq[:, gt_start:gt_end + 1]

        batch_stats = trainer._diagnose_free_run(
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
        batch_stats["MSEnormY"] = mse_normY

        for key, value in batch_stats.items():
            stats_accum.setdefault(key, []).append(value)

        debug_steps = int(getattr(trainer, "freerun_debug_steps", 0) or 0)
        if debug_steps > 0 and diag_records:
            limit = min(debug_steps, len(diag_records))
            print(f"[FreeRunDiag][preview] batch#{batches_processed} showing {limit} steps:")
            for rec in diag_records[:limit]:
                step = rec.get("step", -1)
                yaw_pred = float(rec.get("yaw_cmd_vs_pred_deg", float("nan")))
                yaw_gt = float(rec.get("yaw_cmd_vs_gt_deg", float("nan")))
                yaw_abs = float(rec.get("yaw_abs_deg", float("nan")))
                root_mae = float(rec.get("root_vel_mae", float("nan")))
                delta_abs = float(rec.get("delta_norm_abs", float("nan")))
                print(
                    f"  step={int(step):02d} yaw_cmd_vs_pred={yaw_pred:.2f}° "
                    f"yaw_cmd_vs_gt={yaw_gt:.2f}° yaw_abs={yaw_abs:.2f}° "
                    f"root_vel_mae={root_mae:.4f} delta_norm_abs={delta_abs:.4f}"
                )

        if diag_records and base_debug_path:
            epoch = int(getattr(trainer, "cur_epoch", 0) or 0)
            run_name = getattr(trainer, "_current_run_name", None)
            suffix = f"ep{epoch:03d}" if epoch > 0 else "ep"
            if run_name:
                suffix = f"{run_name}_{suffix}"
            candidate = Path(base_debug_path)
            if candidate.is_dir() or str(base_debug_path).endswith("/"):
                candidate = candidate / f"freerun_diag_{suffix}.pt"
            else:
                candidate = candidate.with_name(candidate.stem + f"_{suffix}" + candidate.suffix)
            debug_path = str(candidate)
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "clip_id": batch.get("clip_id"),
                    "start": batch.get("start"),
                    "records": diag_records,
                    "metrics": batch_stats,
                    "keybone_summary": batch_stats.get("KeyBoneSummary"),
                    "keybone_details": batch_stats.get("KeyBoneDetails"),
                }
                torch.save(payload, debug_path)
                last_debug_path = debug_path
                debug_saved = True
                print(f"[FreeRunDiag] saved diagnostics to {debug_path}")
            except Exception as exc:
                print(f"[FreeRunDiag][WARN] failed to save diagnostics: {exc}")

        batches_processed += 1

    def _avg_simple_dict(dict_list):
        totals = {}
        counts = {}
        for item in dict_list:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if isinstance(v, (int, float)):
                    totals[k] = totals.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: (totals[k] / max(1, counts[k])) for k in totals}

    def _avg_nested_dict(dict_list):
        result = {}
        counts = {}
        for item in dict_list:
            if not isinstance(item, dict):
                continue
            for bone, metrics in item.items():
                if not isinstance(metrics, dict):
                    continue
                sub_res = result.setdefault(bone, {})
                sub_cnt = counts.setdefault(bone, {})
                for mk, mv in metrics.items():
                    if isinstance(mv, (int, float)):
                        sub_res[mk] = sub_res.get(mk, 0.0) + float(mv)
                        sub_cnt[mk] = sub_cnt.get(mk, 0) + 1
        for bone, metrics in result.items():
            cnt = counts.get(bone, {})
            for mk in list(metrics.keys()):
                metrics[mk] = metrics[mk] / max(1, cnt.get(mk, 1))
        return result

    summary: Dict[str, Any] = {}
    for key, values in stats_accum.items():
        if not values:
            continue
        sample = values[0]
        if isinstance(sample, dict):
            if key == 'KeyBoneDetails':
                summary[key] = _avg_nested_dict(values)
            else:
                summary[key] = _avg_simple_dict(values)
            continue
        try:
            summary[key] = float(sum(values) / max(1, len(values)))
        except Exception:
            summary[key] = values

    defaults: Dict[str, Any] = {
        "MSEnormY": float("nan"),
        "GeoDeg": float("nan"),
        "YawAbsDeg": float("nan"),
        "RootVelMAE": float("nan"),
        "AngVelMAE": float("nan"),
        "AngVelMagRel": float("nan"),
        "KeyBoneSummary": {},
        "KeyBoneDetails": {},
    }
    for key, val in defaults.items():
        summary.setdefault(key, val if not isinstance(val, dict) else dict(val))

    if hasattr(trainer, '_diag_scope'):
        delattr(trainer, '_diag_scope')
    return summary
