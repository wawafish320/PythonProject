#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.geometry import geodesic_R, reproject_rot6d, rot6d_to_matrix
from train.validate.run_freerun_cycles import FreeRunCycleRunner


DEFAULT_TAILK7_CKPT = ROOT / "models" / "__tmp_cp015_tailk7_stage70a_from_tailfix_20260402" / "ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth"
DEFAULT_BASELINE_CKPT = ROOT / "models" / "__tmp_posttrain_pipeline_from_bestfree_20260317" / "70a" / "ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth"
DEFAULT_NPZ = ROOT / "raw_data" / "processed_data" / "Walk_F.npz"
DEFAULT_BUNDLE = ROOT / "raw_data" / "processed_data" / "norm_template.json"
DEFAULT_PRETRAIN_TEMPLATE = ROOT / "models" / "pretrain_template.json"
DEFAULT_MODEL_OUT = ROOT / "models" / "__tmp_tailk7_vs_baseline_leg_linear_probe_20260403"
DEFAULT_DEBUG_OUT = ROOT / "debug_output" / "_tmp_tailk7_vs_baseline_leg_linear_probe_20260403"


@dataclass
class ProbeMetrics:
    loss: float
    geo_mean_deg: float
    geo_p90_deg: float
    geo_p95_deg: float
    sample_count: int
    joint_count: int


@dataclass
class ProbeCheckpointSummary:
    epoch: int
    step: int
    metrics: ProbeMetrics
    ckpt_path: str


@dataclass
class WeightAudit:
    donor_trainable_param_names: List[str]
    donor_changed_param_count: int
    donor_max_abs_delta: float
    probe_changed_param_count: int
    probe_max_abs_delta: float
    probe_module_type: str
    probe_param_names: List[str]


@dataclass
class DonorSummary:
    donor_name: str
    ckpt_path: str
    encoder_bundle: str
    bundle_json: str
    data_path: str
    tap_name: str
    tap_dim: int
    tap_shape: List[int]
    leg_joint_names: List[str]
    leg_joint_indices: List[int]
    holdout_eval_timestep_indices: List[int]
    train_timestep_indices: List[int]
    train_final: ProbeMetrics
    eval_final: ProbeMetrics
    best_checkpoint: ProbeCheckpointSummary
    last_checkpoint: ProbeCheckpointSummary
    weight_audit: WeightAudit


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(pref: str) -> str:
    pref = str(pref or "cpu").strip().lower()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        return "mps" if has_mps else "cpu"
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"
    return "cpu"


def _make_runner_args(
    *,
    ckpt_path: Path,
    posttrain_cfg: Dict[str, Any],
    bundle_json: Path,
    encoder_bundle: Path,
    pretrain_template: Path,
    device: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=str(ckpt_path),
        device=str(device),
        bundle=str(bundle_json),
        pretrain_template=str(pretrain_template),
        encoder_bundle=str(encoder_bundle),
        event_clock="auto",
        contact_plan_init_mode=None,
        contact_plan_init_hidden=None,
        contact_plan_init_dropout=None,
        so3_corr_apply=False,
        so3_corr_max_deg=20.0,
        so3_corr_gate_force=None,
        so3_corr_gate_from_contacts_err=False,
        so3_corr_gate_from_contacts_err_mode="scale",
        so3_corr_gate_err_k=1.0,
        so3_corr_gate_err_bias=0.0,
        so3_corr_gate_err_max=1.0,
        so3_corr_gate_err_ref_steps=8,
        so3_corr_gate_err_margin=0.0,
        so3_corr_gate_err_use_ref=False,
        so3_corr_gate_scale_max=2.0,
        log_contact_plan_logits_decomp=False,
        export_contact_meas_head_swap=False,
        log_contacts=False,
        contact_plan_inject_scale=1.0,
        contact_plan_time_bias_scale=1.0,
        contact_plan_init_hidden_override=None,
        contact_plan_init_dropout_override=None,
        direct_pose_meas_source="model",
        direct_pose_meas_warmup_steps=0,
        direct_pose_plan_source="model",
        direct_pose_softgt_stats=None,
        direct_pose_leg_cross_leg_ablate="none",
        direct_pose_leg_side_plan_other_ablate="none",
        contacts_meas_source="model",
        contacts_meas_pretrain_clamp=1.0,
        contacts_meas_pretrain_affine_stats=None,
        contacts_meas_pretrain_anchor_ckpt=None,
        contacts_meas_model_logit_scale=1.0,
        contacts_meas_model_onehot=False,
        contacts_meas_model_onehot_conditional=False,
        contacts_meas_model_onehot_ds_thr=0.5,
        contacts_meas_gt_override_sics="",
        contacts_meas_gt_override_cycle_gte=1,
        contacts_meas_gt_override_drop_wrap="on",
        phase_event_thr=0.5,
        phase_event_hyst=0.0,
        phase_event_min_interval=0,
        phase_reset_source="contacts_meas",
        phase_reset_source_strict="off",
        ttc_event_kind="touchdown",
        ttc_max=None,
        ttc_gt_event_shift="",
        lambda_fusion_apply=False,
        lambda_reliability_mode="none",
        lambda_reliability_warmup_steps=0,
        lambda_reliability_contact_err_max=1.0,
        lambda_reliability_warmup_joint_scales=None,
        num_heads=int(posttrain_cfg.get("num_heads", 4) or 4),
        dropout=float(posttrain_cfg.get("dropout", 0.1) or 0.1),
        context_len=int(posttrain_cfg.get("context_len", 16) or 16),
        depth=int(posttrain_cfg.get("depth", 3) or 3),
    )


def _extract_rot6d_columns(trainer: Any) -> tuple[str, str]:
    cols = ("X", "Z")
    try:
        cand = getattr(getattr(trainer, "loss_fn", None), "_rot6d_columns", None)
        if isinstance(cand, (tuple, list)) and len(cand) >= 2:
            a = str(cand[0]).upper().strip()
            b = str(cand[1]).upper().strip()
            if a in ("X", "Y", "Z") and b in ("X", "Y", "Z") and a != b:
                cols = (a, b)
    except Exception:
        cols = ("X", "Z")
    return cols


def _select_even_holdout_indices(total_steps: int, holdout_count: int) -> np.ndarray:
    if total_steps <= 1:
        raise ValueError(f"total_steps must be >1, got {total_steps}")
    holdout_count = max(1, min(int(holdout_count), total_steps - 1))
    idx = np.linspace(0, total_steps - 1, num=holdout_count, dtype=np.int64)
    idx = np.unique(idx)
    if idx.size < holdout_count:
        full = list(idx.tolist())
        for cand in range(total_steps):
            if cand not in full:
                full.append(int(cand))
            if len(full) >= holdout_count:
                break
        idx = np.asarray(sorted(full[:holdout_count]), dtype=np.int64)
    return idx


def _metrics_from_prediction(
    *,
    pred_raw_leg: torch.Tensor,
    gt_raw_leg: torch.Tensor,
    columns: Sequence[str],
) -> ProbeMetrics:
    pred = pred_raw_leg.detach().float().cpu()
    gt = gt_raw_leg.detach().float().cpu()
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape or pred.shape[0] <= 0:
        raise ValueError(f"Invalid prediction/target shapes: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")
    joint_count = int(pred.shape[1] // 6)
    pred6 = reproject_rot6d(pred).view(-1, joint_count, 6)
    gt6 = reproject_rot6d(gt).view(-1, joint_count, 6)
    loss = torch.mean((pred - gt) ** 2)
    pred_R = rot6d_to_matrix(pred6, columns=tuple(columns))
    gt_R = rot6d_to_matrix(gt6, columns=tuple(columns))
    geo_deg = geodesic_R(pred_R, gt_R, reduce=None).reshape(-1) * (180.0 / math.pi)
    geo_np = geo_deg.detach().cpu().numpy().astype(np.float64, copy=False)
    return ProbeMetrics(
        loss=float(loss.item()),
        geo_mean_deg=float(np.nanmean(geo_np)),
        geo_p90_deg=float(np.nanpercentile(geo_np, 90)),
        geo_p95_deg=float(np.nanpercentile(geo_np, 95)),
        sample_count=int(pred.shape[0]),
        joint_count=int(joint_count),
    )


def _state_max_abs_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> tuple[int, float]:
    changed = 0
    max_abs = 0.0
    for name, before_tensor in before.items():
        after_tensor = after.get(name)
        if after_tensor is None:
            raise KeyError(f"Missing parameter after training: {name}")
        delta = (after_tensor.detach().cpu() - before_tensor.detach().cpu()).abs()
        cur_max = float(delta.max().item()) if delta.numel() > 0 else 0.0
        if cur_max > 0.0:
            changed += 1
            max_abs = max(max_abs, cur_max)
    return changed, max_abs


def _save_probe_ckpt(
    *,
    ckpt_path: Path,
    donor_name: str,
    probe: nn.Linear,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    epoch: int,
    step: int,
    metrics: ProbeMetrics,
    meta: Dict[str, Any],
) -> None:
    payload = {
        "donor_name": donor_name,
        "probe_type": "strict_linear",
        "probe_state_dict": probe.state_dict(),
        "feature_norm": {
            "mean": feat_mean.detach().cpu(),
            "std": feat_std.detach().cpu(),
        },
        "epoch": int(epoch),
        "step": int(step),
        "metrics": asdict(metrics),
        "meta": meta,
    }
    torch.save(payload, ckpt_path)


def _feature_and_target_bundle(
    *,
    ckpt_path: Path,
    donor_name: str,
    npz_path: Path,
    device: str,
    seq_len: int,
    bundle_json_override: Path | None,
    pretrain_template: Path,
) -> Dict[str, Any]:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")
    posttrain_cfg = dict(payload.get("posttrain_cfg") or {})
    bundle_json = bundle_json_override if bundle_json_override is not None else Path(str(posttrain_cfg.get("bundle_json") or DEFAULT_BUNDLE)).expanduser()
    encoder_bundle = Path(str(posttrain_cfg.get("encoder_bundle") or "")).expanduser()
    if not bundle_json.is_file():
        raise FileNotFoundError(f"bundle_json not found for {donor_name}: {bundle_json}")
    if not encoder_bundle.is_file():
        raise FileNotFoundError(f"encoder_bundle not found for {donor_name}: {encoder_bundle}")

    runner_args = _make_runner_args(
        ckpt_path=ckpt_path,
        posttrain_cfg=posttrain_cfg,
        bundle_json=bundle_json,
        encoder_bundle=encoder_bundle,
        pretrain_template=pretrain_template,
        device=device,
    )
    runner = FreeRunCycleRunner(runner_args)
    ds = runner._build_dataset(npz_path, seq_len)
    runner._ensure_model_ready(ds)
    model = runner.model
    trainer = runner.trainer
    if model is None or trainer is None:
        raise RuntimeError(f"Failed to reconstruct donor runtime for {donor_name}")

    leg_joint_idx = getattr(model, "direct_pose_leg_joint_idx", None)
    leg_joint_names = getattr(model, "direct_pose_leg_joint_names", None)
    if not isinstance(leg_joint_idx, list) or not leg_joint_idx:
        raise RuntimeError(f"{donor_name}: missing direct_pose_leg_joint_idx")
    if not isinstance(leg_joint_names, list) or len(leg_joint_names) != len(leg_joint_idx):
        raise RuntimeError(f"{donor_name}: missing/mismatched direct_pose_leg_joint_names")
    if getattr(model, "direct_pose_head", None) is None:
        raise RuntimeError(f"{donor_name}: donor has no direct_pose_head shared trunk")

    sample = ds[0]
    donor_state_before = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    donor_trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    activations: Dict[str, torch.Tensor] = {}

    def _hook_trunk(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        activations["trunk_hidden"] = output.detach().cpu()

    hook = model.direct_pose_head.register_forward_hook(_hook_trunk)
    with torch.no_grad():
        ret = model(
            sample["motion"].unsqueeze(0),
            sample["cond_in"].unsqueeze(0),
            contacts=sample["contacts"].unsqueeze(0),
            angvel=sample["angvel"].unsqueeze(0),
            pose_history=sample["pose_hist"].unsqueeze(0),
        )
    hook.remove()

    trunk_hidden = activations.get("trunk_hidden")
    if trunk_hidden is None:
        raise RuntimeError(f"{donor_name}: failed to capture trunk_hidden")
    if trunk_hidden.ndim != 2:
        raise RuntimeError(f"{donor_name}: unexpected trunk_hidden shape {tuple(trunk_hidden.shape)}")

    gt_motion = sample["gt_motion"].unsqueeze(0)
    gt_raw = trainer._denorm(gt_motion)
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice) or rot_slice.start is None or rot_slice.stop is None:
        raise RuntimeError(f"{donor_name}: missing rot6d slice on trainer")
    rot_raw = gt_raw[:, :, rot_slice].detach().cpu().view(1, gt_raw.shape[1], -1)
    joint_count = int(rot_raw.shape[-1] // 6)
    leg_idx_tensor = torch.as_tensor(leg_joint_idx, dtype=torch.long)
    if int(leg_idx_tensor.max().item()) >= joint_count:
        raise RuntimeError(
            f"{donor_name}: leg joint index out of bounds (max={int(leg_idx_tensor.max().item())}, joints={joint_count})"
        )
    gt_leg_raw = rot_raw.view(1, gt_raw.shape[1], joint_count, 6)[:, :, leg_idx_tensor, :].reshape(gt_raw.shape[1], -1)
    columns = _extract_rot6d_columns(trainer)
    return {
        "donor_name": donor_name,
        "ckpt_path": str(ckpt_path),
        "bundle_json": str(bundle_json),
        "encoder_bundle": str(encoder_bundle),
        "posttrain_cfg": posttrain_cfg,
        "runner": runner,
        "model": model,
        "trainer": trainer,
        "trunk_hidden": trunk_hidden.float(),
        "gt_leg_raw": gt_leg_raw.float(),
        "columns": columns,
        "leg_joint_idx": [int(x) for x in leg_joint_idx],
        "leg_joint_names": [str(x) for x in leg_joint_names],
        "tap_name": "direct_pose_head output (shared trunk_hidden before all direct pose readouts)",
        "tap_dim": int(trunk_hidden.shape[-1]),
        "tap_shape": [int(x) for x in trunk_hidden.shape],
        "donor_state_before": donor_state_before,
        "donor_trainable": donor_trainable,
        "out_direct_shape": list(ret["out_direct"].shape) if isinstance(ret, dict) and "out_direct" in ret else None,
    }


def _run_single_donor_probe(
    *,
    donor_payload: Dict[str, Any],
    holdout_count: int,
    epochs: int,
    steps_per_epoch: int,
    lr: float,
    seed: int,
    out_dir: Path,
) -> DonorSummary:
    _set_seed(int(seed))
    donor_name = str(donor_payload["donor_name"])
    features = donor_payload["trunk_hidden"].clone()
    targets = donor_payload["gt_leg_raw"].clone()
    total_steps = int(features.shape[0])
    eval_idx = _select_even_holdout_indices(total_steps, holdout_count)
    train_mask = np.ones(total_steps, dtype=bool)
    train_mask[eval_idx] = False
    train_idx = np.nonzero(train_mask)[0].astype(np.int64)
    if train_idx.size <= 0 or eval_idx.size <= 0:
        raise RuntimeError(f"{donor_name}: invalid train/eval split")

    x_train = features[train_idx]
    y_train = targets[train_idx]
    x_eval = features[eval_idx]
    y_eval = targets[eval_idx]

    feat_mean = x_train.mean(dim=0, keepdim=True)
    feat_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_train_norm = (x_train - feat_mean) / feat_std
    x_eval_norm = (x_eval - feat_mean) / feat_std

    probe = nn.Linear(int(features.shape[-1]), int(targets.shape[-1]))
    probe_initial = {name: tensor.detach().cpu().clone() for name, tensor in probe.state_dict().items()}
    optimizer = torch.optim.AdamW(probe.parameters(), lr=float(lr), weight_decay=0.0)
    loss_fn = nn.MSELoss(reduction="mean")
    columns = donor_payload["columns"]

    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best_eval_mean = float("inf")
    best_state = copy.deepcopy(probe.state_dict())
    last_state = copy.deepcopy(probe.state_dict())
    best_ckpt_path = out_dir / f"ckpt_best_{donor_name}_strict_linear_probe.pth"
    last_ckpt_path = out_dir / f"ckpt_last_{donor_name}_strict_linear_probe.pth"
    best_summary: ProbeCheckpointSummary | None = None
    last_train_metrics: ProbeMetrics | None = None
    last_eval_metrics: ProbeMetrics | None = None

    global_step = 0
    for epoch in range(1, int(epochs) + 1):
        probe.train()
        train_loss_value = float("nan")
        for _ in range(int(steps_per_epoch)):
            optimizer.zero_grad(set_to_none=True)
            pred_train = probe(x_train_norm)
            loss = loss_fn(pred_train, y_train)
            loss.backward()
            optimizer.step()
            train_loss_value = float(loss.detach().cpu().item())
            global_step += 1

        probe.eval()
        with torch.no_grad():
            pred_train_final = probe(x_train_norm)
            pred_eval_final = probe(x_eval_norm)
        train_metrics = _metrics_from_prediction(
            pred_raw_leg=pred_train_final,
            gt_raw_leg=y_train,
            columns=columns,
        )
        eval_metrics = _metrics_from_prediction(
            pred_raw_leg=pred_eval_final,
            gt_raw_leg=y_eval,
            columns=columns,
        )
        history.append(
            {
                "epoch": int(epoch),
                "step": int(global_step),
                "train": asdict(train_metrics),
                "eval": asdict(eval_metrics),
                "train_loss_step_last": float(train_loss_value),
            }
        )
        last_state = copy.deepcopy(probe.state_dict())
        last_train_metrics = train_metrics
        last_eval_metrics = eval_metrics
        if float(eval_metrics.geo_mean_deg) < float(best_eval_mean):
            best_eval_mean = float(eval_metrics.geo_mean_deg)
            best_state = copy.deepcopy(probe.state_dict())
            best_metrics = ProbeCheckpointSummary(
                epoch=int(epoch),
                step=int(global_step),
                metrics=eval_metrics,
                ckpt_path=str(best_ckpt_path),
            )
            best_summary = best_metrics
            _save_probe_ckpt(
                ckpt_path=best_ckpt_path,
                donor_name=donor_name,
                probe=probe,
                feat_mean=feat_mean,
                feat_std=feat_std,
                epoch=int(epoch),
                step=int(global_step),
                metrics=eval_metrics,
                meta={
                    "split": {
                        "train_timestep_indices": train_idx.tolist(),
                        "eval_timestep_indices": eval_idx.tolist(),
                    },
                    "columns": list(columns),
                    "leg_joint_names": donor_payload["leg_joint_names"],
                    "leg_joint_indices": donor_payload["leg_joint_idx"],
                },
            )

    if last_train_metrics is None or last_eval_metrics is None:
        raise RuntimeError(f"{donor_name}: probe did not record last-epoch metrics")
    probe.load_state_dict(last_state)
    probe.eval()
    with torch.no_grad():
        pred_train_last = probe(x_train_norm)
        pred_eval_last = probe(x_eval_norm)
    train_final = _metrics_from_prediction(pred_raw_leg=pred_train_last, gt_raw_leg=y_train, columns=columns)
    eval_final = _metrics_from_prediction(pred_raw_leg=pred_eval_last, gt_raw_leg=y_eval, columns=columns)
    _save_probe_ckpt(
        ckpt_path=last_ckpt_path,
        donor_name=donor_name,
        probe=probe,
        feat_mean=feat_mean,
        feat_std=feat_std,
        epoch=int(epochs),
        step=int(global_step),
        metrics=eval_final,
        meta={
            "split": {
                "train_timestep_indices": train_idx.tolist(),
                "eval_timestep_indices": eval_idx.tolist(),
            },
            "columns": list(columns),
            "leg_joint_names": donor_payload["leg_joint_names"],
            "leg_joint_indices": donor_payload["leg_joint_idx"],
        },
    )

    donor_state_after = {name: param.detach().cpu().clone() for name, param in donor_payload["model"].named_parameters()}
    donor_changed_count, donor_max_abs_delta = _state_max_abs_delta(
        donor_payload["donor_state_before"],
        donor_state_after,
    )
    probe_changed_count, probe_max_abs_delta = _state_max_abs_delta(
        probe_initial,
        {name: tensor.detach().cpu().clone() for name, tensor in probe.state_dict().items()},
    )
    if best_summary is None:
        raise RuntimeError(f"{donor_name}: probe failed to record a best checkpoint")

    weight_audit = WeightAudit(
        donor_trainable_param_names=list(donor_payload["donor_trainable"]),
        donor_changed_param_count=int(donor_changed_count),
        donor_max_abs_delta=float(donor_max_abs_delta),
        probe_changed_param_count=int(probe_changed_count),
        probe_max_abs_delta=float(probe_max_abs_delta),
        probe_module_type=probe.__class__.__name__,
        probe_param_names=[name for name, _ in probe.named_parameters()],
    )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    summary = DonorSummary(
        donor_name=donor_name,
        ckpt_path=str(donor_payload["ckpt_path"]),
        encoder_bundle=str(donor_payload["encoder_bundle"]),
        bundle_json=str(donor_payload["bundle_json"]),
        data_path=str(DEFAULT_NPZ),
        tap_name=str(donor_payload["tap_name"]),
        tap_dim=int(donor_payload["tap_dim"]),
        tap_shape=list(donor_payload["tap_shape"]),
        leg_joint_names=list(donor_payload["leg_joint_names"]),
        leg_joint_indices=list(donor_payload["leg_joint_idx"]),
        holdout_eval_timestep_indices=eval_idx.tolist(),
        train_timestep_indices=train_idx.tolist(),
        train_final=train_final,
        eval_final=eval_final,
        best_checkpoint=best_summary,
        last_checkpoint=ProbeCheckpointSummary(
            epoch=int(epochs),
            step=int(global_step),
            metrics=eval_final,
            ckpt_path=str(last_ckpt_path),
        ),
        weight_audit=weight_audit,
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def _table(summary_tailk7: DonorSummary, summary_baseline: DonorSummary) -> List[Dict[str, Any]]:
    rows = []
    for metric_name in ("geo_mean_deg", "geo_p90_deg", "geo_p95_deg", "loss"):
        tail_val = _safe_float(getattr(summary_tailk7.eval_final, metric_name if metric_name != "loss" else "loss"))
        base_val = _safe_float(getattr(summary_baseline.eval_final, metric_name if metric_name != "loss" else "loss"))
        rows.append(
            {
                "metric": metric_name,
                "tailk7": tail_val,
                "baseline": base_val,
                "abs_delta_tailk7_minus_baseline": tail_val - base_val,
                "rel_delta_vs_baseline": (tail_val - base_val) / base_val if math.isfinite(base_val) and abs(base_val) > 1e-12 else float("nan"),
            }
        )
    return rows


def _write_debug_summary(
    *,
    out_dir: Path,
    tailk7: DonorSummary,
    baseline: DonorSummary,
    config: Dict[str, Any],
) -> None:
    comparison_rows = _table(tailk7, baseline)
    payload = {
        "config": config,
        "donors": {
            "tailk7": asdict(tailk7),
            "baseline": asdict(baseline),
        },
        "comparison": comparison_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Tailk7 Vs Baseline Leg Linear Probe",
        "",
        "## Protocol",
        "",
        f"- tap: `{tailk7.tap_name}`",
        f"- batch: `{config['batch']}`",
        f"- seq_len: `{config['seq_len']}`",
        f"- epochs: `{config['epochs']}`",
        f"- steps_per_epoch: `{config['steps_per_epoch']}`",
        f"- lr: `{config['lr']}`",
        f"- seed: `{config['seed']}`",
        f"- strict linear: `{tailk7.weight_audit.probe_module_type}`",
        "",
        "## Eval Comparison",
        "",
        "| metric | tailk7 | baseline | tailk7-baseline | rel_delta_vs_baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        md_lines.append(
            f"| {row['metric']} | {row['tailk7']:.6f} | {row['baseline']:.6f} | "
            f"{row['abs_delta_tailk7_minus_baseline']:.6f} | {row['rel_delta_vs_baseline']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            "## Weight Audit",
            "",
            f"- tailk7 donor changed params: `{tailk7.weight_audit.donor_changed_param_count}` max_abs_delta=`{tailk7.weight_audit.donor_max_abs_delta:.6e}`",
            f"- baseline donor changed params: `{baseline.weight_audit.donor_changed_param_count}` max_abs_delta=`{baseline.weight_audit.donor_max_abs_delta:.6e}`",
            f"- tailk7 probe changed params: `{tailk7.weight_audit.probe_changed_param_count}` max_abs_delta=`{tailk7.weight_audit.probe_max_abs_delta:.6e}`",
            f"- baseline probe changed params: `{baseline.weight_audit.probe_changed_param_count}` max_abs_delta=`{baseline.weight_audit.probe_max_abs_delta:.6e}`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Strict linear leg probe on frozen tailk7 vs baseline donor trunks.")
    ap.add_argument("--tailk7-ckpt", type=Path, default=DEFAULT_TAILK7_CKPT)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--bundle-json", type=Path, default=None, help="Optional override for both donors.")
    ap.add_argument("--pretrain-template", type=Path, default=DEFAULT_PRETRAIN_TEMPLATE)
    ap.add_argument("--model-out-dir", type=Path, default=DEFAULT_MODEL_OUT)
    ap.add_argument("--debug-out-dir", type=Path, default=DEFAULT_DEBUG_OUT)
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda", "mps", "auto"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=87)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps-per-epoch", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-holdout-count", type=int, default=27)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    device = _resolve_device(args.device)
    if int(args.batch) != 1:
        raise SystemExit("This minimal probe runner only supports batch=1.")

    donor_specs = [
        ("tailk7", Path(args.tailk7_ckpt).expanduser().resolve()),
        ("baseline", Path(args.baseline_ckpt).expanduser().resolve()),
    ]
    donor_payloads: Dict[str, Dict[str, Any]] = {}
    for donor_name, ckpt_path in donor_specs:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        donor_payloads[donor_name] = _feature_and_target_bundle(
            ckpt_path=ckpt_path,
            donor_name=donor_name,
            npz_path=Path(args.npz).expanduser().resolve(),
            device=device,
            seq_len=int(args.seq_len),
            bundle_json_override=Path(args.bundle_json).expanduser().resolve() if args.bundle_json else None,
            pretrain_template=Path(args.pretrain_template).expanduser().resolve(),
        )

    tail_names = donor_payloads["tailk7"]["leg_joint_names"]
    base_names = donor_payloads["baseline"]["leg_joint_names"]
    tail_idx = donor_payloads["tailk7"]["leg_joint_idx"]
    base_idx = donor_payloads["baseline"]["leg_joint_idx"]
    if tail_names != base_names or tail_idx != base_idx:
        raise RuntimeError(
            "Leg target set mismatch between donors; refusing to continue with unfair target alignment. "
            f"tailk7 names={tail_names}, baseline names={base_names}, tailk7 idx={tail_idx}, baseline idx={base_idx}"
        )
    if donor_payloads["tailk7"]["tap_dim"] != donor_payloads["baseline"]["tap_dim"]:
        raise RuntimeError(
            "Tap feature dim mismatch between donors; refusing to continue with unfair tap alignment. "
            f"tailk7 dim={donor_payloads['tailk7']['tap_dim']} baseline dim={donor_payloads['baseline']['tap_dim']}"
        )

    model_out_dir = Path(args.model_out_dir).expanduser().resolve()
    debug_out_dir = Path(args.debug_out_dir).expanduser().resolve()
    tail_summary = _run_single_donor_probe(
        donor_payload=donor_payloads["tailk7"],
        holdout_count=int(args.eval_holdout_count),
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch),
        lr=float(args.lr),
        seed=int(args.seed),
        out_dir=model_out_dir / "tailk7",
    )
    baseline_summary = _run_single_donor_probe(
        donor_payload=donor_payloads["baseline"],
        holdout_count=int(args.eval_holdout_count),
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch),
        lr=float(args.lr),
        seed=int(args.seed),
        out_dir=model_out_dir / "baseline",
    )

    _write_debug_summary(
        out_dir=debug_out_dir,
        tailk7=tail_summary,
        baseline=baseline_summary,
        config={
            "batch": int(args.batch),
            "seq_len": int(args.seq_len),
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "device": str(device),
            "eval_holdout_count": int(args.eval_holdout_count),
        },
    )

    print(f"[leg-linear-probe] tailk7 eval mean/p90/p95 deg = {tail_summary.eval_final.geo_mean_deg:.6f} / {tail_summary.eval_final.geo_p90_deg:.6f} / {tail_summary.eval_final.geo_p95_deg:.6f}")
    print(f"[leg-linear-probe] baseline eval mean/p90/p95 deg = {baseline_summary.eval_final.geo_mean_deg:.6f} / {baseline_summary.eval_final.geo_p90_deg:.6f} / {baseline_summary.eval_final.geo_p95_deg:.6f}")
    print(f"[leg-linear-probe] summary_json = {debug_out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
