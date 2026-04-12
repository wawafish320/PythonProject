#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_tailk7_vs_baseline_leg_linear_probe import (
    DEFAULT_BASELINE_CKPT,
    DEFAULT_NPZ,
    DEFAULT_PRETRAIN_TEMPLATE,
    DEFAULT_TAILK7_CKPT,
    ProbeCheckpointSummary,
    ProbeMetrics,
    WeightAudit,
    _extract_rot6d_columns,
    _make_runner_args,
    _metrics_from_prediction,
    _resolve_device,
    _safe_float,
    _select_even_holdout_indices,
    _set_seed,
    _state_max_abs_delta,
)
from train.models import STAGE6_3WAY_ARMCHAIN_BONES, _resolve_joint_spec_indices
from train.validate.run_freerun_cycles import FreeRunCycleRunner


DEFAULT_MODEL_OUT = ROOT / "models" / "__tmp_cp015_tailk7_cond_arm_probe_20260404"
DEFAULT_DEBUG_OUT = ROOT / "debug_output" / "_tmp_cp015_tailk7_cond_arm_probe_20260404"

OLD_PROBE_TAP = "direct_pose_head output (shared trunk_hidden before all direct pose readouts)"
OLD_PROBE_TARGET = "teacher-forced local pose target restricted to leg joints via model.direct_pose_leg_joint_idx"
COND_TAP = "frozen cond_in (direct_pose_feat_source=cond feature before downstream direct-pose trunk/readout)"
TRUNK_HIDDEN_TAP = "direct_pose_head output (shared trunk_hidden before all direct pose readouts)"
NEW_PROBE_TARGET = "teacher-forced local pose target restricted to arm joints via model.direct_pose_arm_out_idx"


@dataclass
class ProbeRunSummary:
    probe_name: str
    probe_arch: str
    probe_param_count: int
    train_final: ProbeMetrics
    eval_final: ProbeMetrics
    best_checkpoint: ProbeCheckpointSummary
    last_checkpoint: ProbeCheckpointSummary
    weight_audit: WeightAudit


@dataclass
class DonorSummary:
    donor_name: str
    ckpt_path: str
    encoder_bundle: str
    bundle_json: str
    data_path: str
    direct_pose_feat_source: str
    tap_name: str
    tap_dim: int
    tap_shape: List[int]
    target_name: str
    target_dim: int
    target_shape: List[int]
    arm_joint_names: List[str]
    arm_joint_indices: List[int]
    holdout_eval_timestep_indices: List[int]
    train_timestep_indices: List[int]
    probes: Dict[str, ProbeRunSummary]


def _save_probe_ckpt(
    *,
    ckpt_path: Path,
    donor_name: str,
    probe_name: str,
    probe_arch: str,
    probe: nn.Module,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    epoch: int,
    step: int,
    metrics: ProbeMetrics,
    meta: Dict[str, Any],
) -> None:
    payload = {
        "donor_name": donor_name,
        "probe_name": probe_name,
        "probe_arch": probe_arch,
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


def _build_probe(probe_name: str, in_dim: int, out_dim: int, tiny_mlp_hidden: int) -> tuple[nn.Module, str]:
    if probe_name == "strict_linear":
        return nn.Linear(int(in_dim), int(out_dim)), f"Linear({int(in_dim)}->{int(out_dim)})"
    if probe_name == "tiny_mlp":
        hidden = int(tiny_mlp_hidden)
        if hidden <= 0:
            raise ValueError(f"tiny_mlp_hidden must be positive, got {hidden}")
        return (
            nn.Sequential(
                nn.Linear(int(in_dim), hidden),
                nn.ReLU(),
                nn.Linear(hidden, int(out_dim)),
            ),
            f"TinyMLP({int(in_dim)}->{hidden}->{int(out_dim)}, ReLU)",
        )
    raise ValueError(f"Unknown probe_name={probe_name}")


def _infer_bone_names(trainer: Any) -> List[str]:
    bone_names = None
    try:
        bone_names = getattr(getattr(trainer, "loss_fn", None), "bone_names", None)
        if bone_names is None:
            bone_names = getattr(trainer, "_bone_names", None)
        if bone_names is None:
            bundle_meta = getattr(trainer, "bundle_meta", None)
            if isinstance(bundle_meta, dict):
                bone_names = bundle_meta.get("bone_names") or bundle_meta.get("skeleton", {}).get("bone_names")
    except Exception:
        bone_names = None
    return [str(x) for x in bone_names] if isinstance(bone_names, (list, tuple)) else []


def _normalize_arm_name_spec(spec: Any) -> List[str]:
    if isinstance(spec, str):
        items = [t.strip() for t in spec.split(",") if t.strip()]
        return items if items else list(STAGE6_3WAY_ARMCHAIN_BONES)
    if isinstance(spec, (list, tuple)):
        items = [str(t).strip() for t in spec if str(t).strip()]
        return items if items else list(STAGE6_3WAY_ARMCHAIN_BONES)
    return list(STAGE6_3WAY_ARMCHAIN_BONES)


def _capture_trunk_hidden(
    model: Any,
    sample: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Any]:
    if getattr(model, "direct_pose_head", None) is None:
        raise RuntimeError("model has no direct_pose_head shared trunk")
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
        raise RuntimeError("failed to capture trunk_hidden")
    if trunk_hidden.ndim != 2:
        raise RuntimeError(f"unexpected trunk_hidden shape {tuple(trunk_hidden.shape)}")
    return trunk_hidden.float(), ret


def _feature_and_target_bundle(
    *,
    ckpt_path: Path,
    donor_name: str,
    npz_path: Path,
    device: str,
    seq_len: int,
    pretrain_template: Path,
    tap_source: str,
) -> Dict[str, Any]:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")
    posttrain_cfg = dict(payload.get("posttrain_cfg") or {})
    bundle_json = Path(str(posttrain_cfg.get("bundle_json") or "")).expanduser()
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

    feat_source = str(getattr(model, "direct_pose_feat_source", posttrain_cfg.get("direct_pose_feat_source", "")) or "").lower().strip()
    if feat_source != "cond":
        raise RuntimeError(f"{donor_name}: expected direct_pose_feat_source=cond, got {feat_source!r}")
    if not bool(getattr(model, "direct_pose_split_enable", False)):
        raise RuntimeError(f"{donor_name}: expected direct_pose_split_enable=true")
    if not bool(getattr(model, "direct_pose_arm_split_enable", False)):
        raise RuntimeError(f"{donor_name}: expected direct_pose_arm_split_enable=true")

    sample = ds[0]
    cond = sample.get("cond_in")
    if not torch.is_tensor(cond) or cond.ndim != 2:
        raise RuntimeError(f"{donor_name}: sample['cond_in'] shape invalid: {tuple(getattr(cond, 'shape', ())) }")

    donor_state_before = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    donor_trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    tap_source = str(tap_source or "").strip().lower()
    if tap_source == "cond":
        features = cond.detach().cpu().float()
        tap_name = COND_TAP
        with torch.no_grad():
            ret = model(
                sample["motion"].unsqueeze(0),
                sample["cond_in"].unsqueeze(0),
                contacts=sample["contacts"].unsqueeze(0),
                angvel=sample["angvel"].unsqueeze(0),
                pose_history=sample["pose_hist"].unsqueeze(0),
            )
    elif tap_source == "trunk_hidden":
        features, ret = _capture_trunk_hidden(model, sample)
        tap_name = TRUNK_HIDDEN_TAP
    else:
        raise RuntimeError(f"{donor_name}: unsupported tap_source={tap_source!r}")

    gt_motion = sample["gt_motion"].unsqueeze(0)
    gt_raw = trainer._denorm(gt_motion).detach().cpu()
    if gt_raw.ndim != 3 or gt_raw.shape[0] != 1:
        raise RuntimeError(f"{donor_name}: unexpected gt_raw shape {tuple(gt_raw.shape)}")

    arm_out_idx = getattr(model, "direct_pose_arm_out_idx", None)
    if not torch.is_tensor(arm_out_idx) or int(arm_out_idx.numel()) <= 0:
        raise RuntimeError(f"{donor_name}: missing direct_pose_arm_out_idx")
    arm_out_idx = arm_out_idx.detach().cpu().to(dtype=torch.long)
    if int(arm_out_idx.numel()) % 6 != 0:
        raise RuntimeError(f"{donor_name}: arm target dim is not a multiple of 6: {int(arm_out_idx.numel())}")
    gt_arm_raw = gt_raw[:, :, arm_out_idx].reshape(int(gt_raw.shape[1]), -1)

    bone_names = _infer_bone_names(trainer)
    arm_name_spec = _normalize_arm_name_spec(getattr(model, "direct_pose_arm_bones", None))
    joint_count = 0
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if isinstance(rot_slice, slice) and rot_slice.start is not None and rot_slice.stop is not None:
        joint_count = int(max(0, (int(rot_slice.stop) - int(rot_slice.start)) // 6))
    elif bone_names:
        joint_count = int(len(bone_names))
    arm_joint_idx, arm_joint_names = _resolve_joint_spec_indices(
        getattr(model, "direct_pose_arm_bones", None),
        default_items=STAGE6_3WAY_ARMCHAIN_BONES,
        bone_names=bone_names if bone_names else None,
        joint_count=max(0, int(joint_count)),
        collect_names=True,
    )
    if isinstance(rot_slice, slice) and rot_slice.start is not None:
        rot_start = int(rot_slice.start)
    else:
        rot_start = 0
    arm_joint_idx_from_dims = sorted(
        {
            int((int(dim) - rot_start) // 6)
            for dim in arm_out_idx.tolist()
            if int(dim) >= rot_start
        }
    )
    if len(arm_joint_idx) <= 0:
        arm_joint_idx = list(arm_joint_idx_from_dims)
    if len(arm_joint_names) <= 0:
        if len(arm_name_spec) == len(arm_joint_idx):
            arm_joint_names = list(arm_name_spec)
        else:
            arm_joint_names = [f"joint_{int(idx)}" for idx in arm_joint_idx]
    if len(arm_joint_idx) <= 0:
        raise RuntimeError(f"{donor_name}: failed to resolve arm joint indices")
    if int(len(arm_joint_idx) * 6) != int(arm_out_idx.numel()):
        raise RuntimeError(
            f"{donor_name}: resolved arm joints imply {int(len(arm_joint_idx) * 6)} dims but direct_pose_arm_out_idx has {int(arm_out_idx.numel())}"
        )

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
        "features": features,
        "gt_arm_raw": gt_arm_raw.detach().cpu().float(),
        "columns": columns,
        "arm_joint_idx": [int(x) for x in arm_joint_idx],
        "arm_joint_names": [str(x) for x in arm_joint_names],
        "tap_name": tap_name,
        "tap_dim": int(features.shape[-1]),
        "tap_shape": [int(x) for x in features.shape],
        "target_name": NEW_PROBE_TARGET,
        "target_dim": int(gt_arm_raw.shape[-1]),
        "target_shape": [int(x) for x in gt_arm_raw.shape],
        "donor_state_before": donor_state_before,
        "donor_trainable": donor_trainable,
        "direct_pose_feat_source": feat_source,
        "out_direct_shape": list(ret["out_direct"].shape) if isinstance(ret, dict) and "out_direct" in ret else None,
    }


def _run_single_probe(
    *,
    donor_payload: Dict[str, Any],
    probe_name: str,
    train_idx: Sequence[int],
    eval_idx: Sequence[int],
    epochs: int,
    steps_per_epoch: int,
    lr: float,
    seed: int,
    tiny_mlp_hidden: int,
    out_dir: Path,
) -> ProbeRunSummary:
    _set_seed(int(seed))
    donor_name = str(donor_payload["donor_name"])
    features = donor_payload["features"].clone()
    targets = donor_payload["gt_arm_raw"].clone()

    x_train = features[list(train_idx)]
    y_train = targets[list(train_idx)]
    x_eval = features[list(eval_idx)]
    y_eval = targets[list(eval_idx)]

    feat_mean = x_train.mean(dim=0, keepdim=True)
    feat_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_train_norm = (x_train - feat_mean) / feat_std
    x_eval_norm = (x_eval - feat_mean) / feat_std

    probe, probe_arch = _build_probe(
        probe_name=probe_name,
        in_dim=int(features.shape[-1]),
        out_dim=int(targets.shape[-1]),
        tiny_mlp_hidden=int(tiny_mlp_hidden),
    )
    probe_initial = {name: tensor.detach().cpu().clone() for name, tensor in probe.state_dict().items()}
    optimizer = torch.optim.AdamW(probe.parameters(), lr=float(lr), weight_decay=0.0)
    loss_fn = nn.MSELoss(reduction="mean")
    columns = donor_payload["columns"]

    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best_eval_mean = float("inf")
    last_state = copy.deepcopy(probe.state_dict())
    best_ckpt_path = out_dir / f"ckpt_best_{donor_name}_{probe_name}.pth"
    last_ckpt_path = out_dir / f"ckpt_last_{donor_name}_{probe_name}.pth"
    best_summary: ProbeCheckpointSummary | None = None

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
        if float(eval_metrics.geo_mean_deg) < float(best_eval_mean):
            best_eval_mean = float(eval_metrics.geo_mean_deg)
            best_summary = ProbeCheckpointSummary(
                epoch=int(epoch),
                step=int(global_step),
                metrics=eval_metrics,
                ckpt_path=str(best_ckpt_path),
            )
            _save_probe_ckpt(
                ckpt_path=best_ckpt_path,
                donor_name=donor_name,
                probe_name=probe_name,
                probe_arch=probe_arch,
                probe=probe,
                feat_mean=feat_mean,
                feat_std=feat_std,
                epoch=int(epoch),
                step=int(global_step),
                metrics=eval_metrics,
                meta={
                    "tap_name": donor_payload["tap_name"],
                    "target_name": donor_payload["target_name"],
                    "split": {
                        "train_timestep_indices": list(train_idx),
                        "eval_timestep_indices": list(eval_idx),
                    },
                    "columns": list(columns),
                    "arm_joint_names": donor_payload["arm_joint_names"],
                    "arm_joint_indices": donor_payload["arm_joint_idx"],
                },
            )

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
        probe_name=probe_name,
        probe_arch=probe_arch,
        probe=probe,
        feat_mean=feat_mean,
        feat_std=feat_std,
        epoch=int(epochs),
        step=int(global_step),
        metrics=eval_final,
        meta={
            "tap_name": donor_payload["tap_name"],
            "target_name": donor_payload["target_name"],
            "split": {
                "train_timestep_indices": list(train_idx),
                "eval_timestep_indices": list(eval_idx),
            },
            "columns": list(columns),
            "arm_joint_names": donor_payload["arm_joint_names"],
            "arm_joint_indices": donor_payload["arm_joint_idx"],
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
        raise RuntimeError(f"{donor_name}: {probe_name} failed to record a best checkpoint")

    weight_audit = WeightAudit(
        donor_trainable_param_names=list(donor_payload["donor_trainable"]),
        donor_changed_param_count=int(donor_changed_count),
        donor_max_abs_delta=float(donor_max_abs_delta),
        probe_changed_param_count=int(probe_changed_count),
        probe_max_abs_delta=float(probe_max_abs_delta),
        probe_module_type=str(probe_arch),
        probe_param_names=[name for name, _ in probe.named_parameters()],
    )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return ProbeRunSummary(
        probe_name=probe_name,
        probe_arch=probe_arch,
        probe_param_count=int(sum(p.numel() for p in probe.parameters())),
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


def _comparison_rows(tail: ProbeRunSummary, base: ProbeRunSummary) -> List[Dict[str, Any]]:
    rows = []
    for metric_name in ("geo_mean_deg", "geo_p90_deg", "geo_p95_deg", "loss"):
        tail_val = _safe_float(getattr(tail.eval_final, metric_name if metric_name != "loss" else "loss"))
        base_val = _safe_float(getattr(base.eval_final, metric_name if metric_name != "loss" else "loss"))
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
    comparisons = {
        probe_name: _comparison_rows(tailk7.probes[probe_name], baseline.probes[probe_name])
        for probe_name in ("strict_linear", "tiny_mlp")
    }
    payload = {
        "config": config,
        "old_probe_reference": {
            "tap_name": OLD_PROBE_TAP,
            "target_name": OLD_PROBE_TARGET,
            "script": str(ROOT / "tools" / "run_tailk7_vs_baseline_leg_linear_probe.py"),
            "summary_json": str(ROOT / "debug_output" / "_tmp_tailk7_vs_baseline_leg_linear_probe_20260403" / "summary.json"),
        },
        "new_probe_protocol": {
            "tap_name": tailk7.tap_name,
            "target_name": tailk7.target_name,
        },
        "donors": {
            "tailk7": asdict(tailk7),
            "baseline": asdict(baseline),
        },
        "comparisons": comparisons,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Tailk7 Vs Baseline Arm Probe",
        "",
        "## Probe Facts",
        "",
        f"- old probe tap: `{OLD_PROBE_TAP}`",
        f"- old probe target: `{OLD_PROBE_TARGET}`",
        f"- new probe tap: `{tailk7.tap_name}`",
        f"- new probe target: `{tailk7.target_name}`",
        f"- tap shape: `{tuple(tailk7.tap_shape)}`",
        f"- arm target shape: `{tuple(tailk7.target_shape)}`",
        f"- arm joints: `{', '.join(tailk7.arm_joint_names)}`",
        "",
        "## Protocol",
        "",
        f"- batch: `{config['batch']}`",
        f"- seq_len: `{config['seq_len']}`",
        f"- epochs: `{config['epochs']}`",
        f"- steps_per_epoch: `{config['steps_per_epoch']}`",
        f"- lr: `{config['lr']}`",
        f"- seed: `{config['seed']}`",
        f"- eval_holdout_count: `{config['eval_holdout_count']}`",
        f"- strict linear params: `{tailk7.probes['strict_linear'].probe_param_count}`",
        f"- tiny MLP params: `{tailk7.probes['tiny_mlp'].probe_param_count}`",
        "",
    ]
    for probe_name, title in (("strict_linear", "Strict Linear Eval Comparison"), ("tiny_mlp", "Tiny MLP Eval Comparison")):
        md_lines.extend(
            [
                f"## {title}",
                "",
                "| metric | tailk7 | baseline | tailk7-baseline | rel_delta_vs_baseline |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in comparisons[probe_name]:
            md_lines.append(
                f"| {row['metric']} | {row['tailk7']:.6f} | {row['baseline']:.6f} | "
                f"{row['abs_delta_tailk7_minus_baseline']:.6f} | {row['rel_delta_vs_baseline']:.6f} |"
            )
        md_lines.append("")
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal arm probe falsifier for tailk7 vs baseline.")
    ap.add_argument("--tailk7-ckpt", type=Path, default=DEFAULT_TAILK7_CKPT)
    ap.add_argument("--baseline-ckpt", type=Path, default=DEFAULT_BASELINE_CKPT)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--pretrain-template", type=Path, default=DEFAULT_PRETRAIN_TEMPLATE)
    ap.add_argument("--model-out-dir", type=Path, default=DEFAULT_MODEL_OUT)
    ap.add_argument("--debug-out-dir", type=Path, default=DEFAULT_DEBUG_OUT)
    ap.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda", "mps", "auto"))
    ap.add_argument("--tap-source", type=str, default="trunk_hidden", choices=("trunk_hidden", "cond"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=87)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps-per-epoch", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-holdout-count", type=int, default=27)
    ap.add_argument("--tiny-mlp-hidden", type=int, default=32)
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
            pretrain_template=Path(args.pretrain_template).expanduser().resolve(),
            tap_source=str(args.tap_source),
        )

    if donor_payloads["tailk7"]["arm_joint_names"] != donor_payloads["baseline"]["arm_joint_names"]:
        raise RuntimeError(
            "Arm target set mismatch between donors; refusing to continue with unfair target alignment. "
            f"tailk7 names={donor_payloads['tailk7']['arm_joint_names']}, baseline names={donor_payloads['baseline']['arm_joint_names']}"
        )
    if donor_payloads["tailk7"]["arm_joint_idx"] != donor_payloads["baseline"]["arm_joint_idx"]:
        raise RuntimeError(
            "Arm target indices mismatch between donors; refusing to continue with unfair target alignment. "
            f"tailk7 idx={donor_payloads['tailk7']['arm_joint_idx']}, baseline idx={donor_payloads['baseline']['arm_joint_idx']}"
        )
    if donor_payloads["tailk7"]["tap_shape"] != donor_payloads["baseline"]["tap_shape"]:
        raise RuntimeError(
            "Tap shape mismatch between donors; refusing to continue with unfair tap alignment. "
            f"tailk7 shape={donor_payloads['tailk7']['tap_shape']} baseline shape={donor_payloads['baseline']['tap_shape']}"
        )
    if donor_payloads["tailk7"]["target_shape"] != donor_payloads["baseline"]["target_shape"]:
        raise RuntimeError(
            "Target shape mismatch between donors; refusing to continue with unfair target alignment. "
            f"tailk7 target_shape={donor_payloads['tailk7']['target_shape']} baseline target_shape={donor_payloads['baseline']['target_shape']}"
        )

    total_steps = int(donor_payloads["tailk7"]["tap_shape"][0])
    eval_idx = _select_even_holdout_indices(total_steps, int(args.eval_holdout_count))
    train_mask = torch.ones(total_steps, dtype=torch.bool)
    train_mask[torch.as_tensor(eval_idx, dtype=torch.long)] = False
    train_idx = torch.nonzero(train_mask, as_tuple=False).flatten().tolist()
    eval_idx_list = [int(x) for x in eval_idx.tolist()]

    model_out_dir = Path(args.model_out_dir).expanduser().resolve()
    debug_out_dir = Path(args.debug_out_dir).expanduser().resolve()

    summaries: Dict[str, DonorSummary] = {}
    for donor_name in ("tailk7", "baseline"):
        payload = donor_payloads[donor_name]
        probes = {
            "strict_linear": _run_single_probe(
                donor_payload=payload,
                probe_name="strict_linear",
                train_idx=train_idx,
                eval_idx=eval_idx_list,
                epochs=int(args.epochs),
                steps_per_epoch=int(args.steps_per_epoch),
                lr=float(args.lr),
                seed=int(args.seed),
                tiny_mlp_hidden=int(args.tiny_mlp_hidden),
                out_dir=model_out_dir / donor_name / "strict_linear",
            ),
            "tiny_mlp": _run_single_probe(
                donor_payload=payload,
                probe_name="tiny_mlp",
                train_idx=train_idx,
                eval_idx=eval_idx_list,
                epochs=int(args.epochs),
                steps_per_epoch=int(args.steps_per_epoch),
                lr=float(args.lr),
                seed=int(args.seed),
                tiny_mlp_hidden=int(args.tiny_mlp_hidden),
                out_dir=model_out_dir / donor_name / "tiny_mlp",
            ),
        }
        summary = DonorSummary(
            donor_name=donor_name,
            ckpt_path=str(payload["ckpt_path"]),
            encoder_bundle=str(payload["encoder_bundle"]),
            bundle_json=str(payload["bundle_json"]),
            data_path=str(Path(args.npz).expanduser().resolve()),
            direct_pose_feat_source=str(payload["direct_pose_feat_source"]),
            tap_name=str(payload["tap_name"]),
            tap_dim=int(payload["tap_dim"]),
            tap_shape=list(payload["tap_shape"]),
            target_name=str(payload["target_name"]),
            target_dim=int(payload["target_dim"]),
            target_shape=list(payload["target_shape"]),
            arm_joint_names=list(payload["arm_joint_names"]),
            arm_joint_indices=list(payload["arm_joint_idx"]),
            holdout_eval_timestep_indices=list(eval_idx_list),
            train_timestep_indices=list(train_idx),
            probes=probes,
        )
        donor_dir = model_out_dir / donor_name
        donor_dir.mkdir(parents=True, exist_ok=True)
        (donor_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
        summaries[donor_name] = summary

    _write_debug_summary(
        out_dir=debug_out_dir,
        tailk7=summaries["tailk7"],
        baseline=summaries["baseline"],
        config={
            "batch": int(args.batch),
            "seq_len": int(args.seq_len),
            "epochs": int(args.epochs),
            "steps_per_epoch": int(args.steps_per_epoch),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "device": str(device),
            "tap_source": str(args.tap_source),
            "eval_holdout_count": int(args.eval_holdout_count),
            "tiny_mlp_hidden": int(args.tiny_mlp_hidden),
        },
    )

    for probe_name in ("strict_linear", "tiny_mlp"):
        tail_eval = summaries["tailk7"].probes[probe_name].eval_final
        base_eval = summaries["baseline"].probes[probe_name].eval_final
        print(
            f"[arm-probe][{probe_name}] tailk7 eval mean/p90/p95 deg = "
            f"{tail_eval.geo_mean_deg:.6f} / {tail_eval.geo_p90_deg:.6f} / {tail_eval.geo_p95_deg:.6f}"
        )
        print(
            f"[arm-probe][{probe_name}] baseline eval mean/p90/p95 deg = "
            f"{base_eval.geo_mean_deg:.6f} / {base_eval.geo_p90_deg:.6f} / {base_eval.geo_p95_deg:.6f}"
        )
    print(f"[arm-probe] summary_json = {debug_out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
