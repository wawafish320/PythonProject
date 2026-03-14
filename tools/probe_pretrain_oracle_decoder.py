#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.report_pretrain_sic_bone import (  # noqa: E402
    ClipEval,
    _aggregate,
    _build_clip_inputs,
    _build_models,
    _build_pose_hist_norm,
    _build_pose_target_norm,
    _device_from_arg,
    _evaluate_clip,
    _read_json,
    _write_count_csv,
    _write_matrix_csv,
    _write_rows_csv,
)
from train.geometry import geodesic_R, reproject_rot6d, rot6d_to_matrix  # noqa: E402
from train.models import DEFAULT_DIRECT_POSE_LEG_BONES  # noqa: E402
from train.normalizers import _make_angnorm_from_spec  # noqa: E402
from train.pretrain_mpl_min import InputProjectors, InputSlices  # noqa: E402
from train.utils import build_mlp  # noqa: E402


WINDOW_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "calf_r_56_62",
        "label": "calf_r @ sic 56-62",
        "bones": ("calf_r",),
        "sic_min": 56,
        "sic_max": 62,
    },
    {
        "id": "calf_l_78_85",
        "label": "calf_l @ sic 78-85",
        "bones": ("calf_l",),
        "sic_min": 78,
        "sic_max": 85,
    },
    {
        "id": "foot_l_ball_l_12_15",
        "label": "foot_l + ball_l @ sic 12-15",
        "bones": ("foot_l", "ball_l"),
        "sic_min": 12,
        "sic_max": 15,
    },
    {
        "id": "legs_all",
        "label": "legs @ all sic",
        "bones": tuple(DEFAULT_DIRECT_POSE_LEG_BONES),
        "sic_min": None,
        "sic_max": None,
    },
    {
        "id": "legs_ge_40",
        "label": "legs @ sic >= 40",
        "bones": tuple(DEFAULT_DIRECT_POSE_LEG_BONES),
        "sic_min": 40,
        "sic_max": None,
    },
)

SELECTION_WINDOW_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("calf_r_56_62", 0.4),
    ("calf_l_78_85", 0.4),
    ("legs_all", 0.1),
    ("legs_ge_40", 0.1),
)


@dataclass
class ProbeSpec:
    probe_id: str
    label: str
    latent_kind: str
    output_mode: str


@dataclass
class SeedRun:
    seed: int
    best_epoch: int
    stop_metric_deg: float
    train_loss: float
    train_geodesic_rad: float
    train_aux_mse: float
    window_metrics: Dict[str, Dict[str, Any]]
    coverage_note: str


@dataclass
class ProbeSummary:
    probe_id: str
    label: str
    bundle: str
    seeds: List[int]
    output_mode: str
    latent_kind: str
    overall_mean_deg_ex_root: float
    window_metrics: Dict[str, Dict[str, Any]]
    seed_runs: List[Dict[str, Any]]
    coverage_note: str


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(mean(vals)), float(pstdev(vals))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _bundle_name(path: Path) -> str:
    name = path.name
    if "stageA" in name:
        return "stageA"
    if "best" in name:
        return "pt_best"
    return path.stem.replace(".", "_")


def _default_out_dir() -> str:
    return f"debug_output/_tmp_pretrain_oracle_decoder_{date.today().strftime('%Y%m%d')}"


def _load_bundle_context(bundle_path: Path, norm_path: Path, npz_path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(str(bundle_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"bundle must be a dict: {bundle_path}")

    norm_spec = _read_json(norm_path)
    joint_count = int(norm_spec.get("J", 0) or 0)
    if joint_count <= 0:
        raise RuntimeError(f"invalid J in norm spec: {norm_path}")

    ang_norm = _make_angnorm_from_spec(norm_spec, J_times_3=joint_count * 3, require_zscore=False)
    pose_hist_len = int(norm_spec.get("pose_hist_len", 0) or 0)
    pose_hist_norm = _build_pose_hist_norm(norm_spec, pose_hist_len, joint_count)
    clip_data = _build_clip_inputs(
        npz_path,
        ang_norm=ang_norm,
        pose_hist_norm=pose_hist_norm,
        pose_hist_len=pose_hist_len,
    )

    encoder, period_head, _contact_head, decoder_pose = _build_models(payload, device=device)

    meta = dict(payload.get("meta", {}))
    input_dim = int(meta.get("input_dim", 0) or 0)
    ang_dim = int(meta.get("ang_dim", joint_count * 3))
    pose_dim = int(meta.get("pose_dim", joint_count * 6))
    pose_hist_dim = int(max(0, input_dim - 2 - ang_dim))
    expected_pose_hist_dim = int(pose_hist_len * joint_count * 6)
    if pose_hist_dim != expected_pose_hist_dim:
        raise RuntimeError(
            f"pose_hist_dim mismatch for {bundle_path.name}: bundle={pose_hist_dim} norm={expected_pose_hist_dim}"
        )
    if int(clip_data["inputs"].shape[1]) != int(input_dim):
        raise RuntimeError(
            f"input dim mismatch for {bundle_path.name}: clip={clip_data['inputs'].shape[1]} bundle={input_dim}"
        )

    layout = InputSlices(
        contact=slice(0, 2),
        ang=slice(2, 2 + ang_dim),
        pose_hist=(slice(2 + ang_dim, input_dim) if pose_hist_dim > 0 else None),
    )
    projectors = InputProjectors(
        layout,
        period_include_ang_sign=False,
        period_use_ang_features=True,
        angnorm=ang_norm,
        amp_linear=True,
    )
    pose_target_norm = _build_pose_target_norm(norm_spec, pose_dim)

    x = torch.from_numpy(clip_data["inputs"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        period_inputs = projectors.period(x)
        _, h_period = encoder(period_inputs, return_summary=True)
        soft_period = torch.tanh(period_head(h_period))

    return {
        "bundle_path": bundle_path,
        "bundle_name": _bundle_name(bundle_path),
        "payload": payload,
        "meta": meta,
        "norm_spec": norm_spec,
        "clip_data": clip_data,
        "encoder": encoder,
        "period_head": period_head,
        "decoder_pose": decoder_pose,
        "pose_target_norm": pose_target_norm,
        "projectors": projectors,
        "device": device,
        "soft_period": soft_period[0].detach().cpu(),
        "h_period": h_period[0].detach().cpu(),
    }


def _subset_pose_raw(pose_raw: np.ndarray, joint_count: int, bone_indices: Sequence[int]) -> np.ndarray:
    arr = np.asarray(pose_raw, dtype=np.float32)
    return arr.reshape(arr.shape[0], int(joint_count), 6)[:, list(bone_indices)].reshape(arr.shape[0], -1)


def _make_clip_eval(
    clip_data: Dict[str, Any],
    pred_pose_raw: np.ndarray,
    *,
    bone_indices: Optional[Sequence[int]] = None,
) -> ClipEval:
    columns = tuple(clip_data["columns"])
    full_bones = list(clip_data["bone_names"])
    full_joint_count = int(len(full_bones))
    target_raw = np.asarray(clip_data["pose_target_raw"], dtype=np.float32)
    pred_raw = np.asarray(pred_pose_raw, dtype=np.float32)

    if bone_indices is None:
        bone_names = full_bones
        root_idx = int(clip_data["root_idx"])
        tgt_sel = target_raw
        pred_sel = pred_raw
    else:
        idx = [int(i) for i in bone_indices]
        bone_names = [str(full_bones[i]) for i in idx]
        root_idx = -1
        tgt_sel = _subset_pose_raw(target_raw, full_joint_count, idx)
        pred_sel = pred_raw

    joint_count = int(len(bone_names))
    pred6 = reproject_rot6d(torch.from_numpy(pred_sel).view(-1, joint_count * 6)).view(-1, joint_count, 6)
    tgt6 = reproject_rot6d(torch.from_numpy(tgt_sel).view(-1, joint_count * 6)).view(-1, joint_count, 6)
    rp = rot6d_to_matrix(pred6, columns=columns)
    rg = rot6d_to_matrix(tgt6, columns=columns)
    geo_deg = (geodesic_R(rp, rg) * (180.0 / math.pi)).cpu().numpy().astype(np.float64, copy=False)

    return ClipEval(
        clip=str(clip_data["clip"]),
        npz_path=Path(clip_data["npz_path"]).resolve(),
        source_json=str(clip_data["source_json"]),
        columns=columns,
        bone_names=bone_names,
        root_idx=int(root_idx),
        sic=np.arange(geo_deg.shape[0], dtype=np.int64),
        geo_deg=geo_deg,
    )


def _window_metric(eval_obj: ClipEval, spec: Dict[str, Any]) -> Dict[str, Any]:
    sic_min = spec.get("sic_min")
    sic_max = spec.get("sic_max")
    requested_bones = [str(b) for b in spec["bones"]]
    bone_indices = [i for i, bone in enumerate(eval_obj.bone_names) if bone in requested_bones]
    available_bones = [str(eval_obj.bone_names[i]) for i in bone_indices]

    sic_mask = np.ones(eval_obj.sic.shape[0], dtype=bool)
    if sic_min is not None:
        sic_mask &= eval_obj.sic >= int(sic_min)
    if sic_max is not None:
        sic_mask &= eval_obj.sic <= int(sic_max)

    if not bone_indices or not bool(np.any(sic_mask)):
        return {
            "window_id": str(spec["id"]),
            "label": str(spec["label"]),
            "mean_deg": float("nan"),
            "p90_deg": float("nan"),
            "count": 0,
            "requested_bones": requested_bones,
            "available_bones": available_bones,
            "coverage": f"{len(available_bones)}/{len(requested_bones)}",
            "sic_min": sic_min,
            "sic_max": sic_max,
        }

    vals = eval_obj.geo_deg[sic_mask][:, bone_indices].reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        mean_deg = float("nan")
        p90_deg = float("nan")
        count = 0
    else:
        mean_deg = float(np.nanmean(vals))
        p90_deg = float(np.nanpercentile(vals, 90))
        count = int(vals.size)
    return {
        "window_id": str(spec["id"]),
        "label": str(spec["label"]),
        "mean_deg": mean_deg,
        "p90_deg": p90_deg,
        "count": count,
        "requested_bones": requested_bones,
        "available_bones": available_bones,
        "coverage": f"{len(available_bones)}/{len(requested_bones)}",
        "sic_min": sic_min,
        "sic_max": sic_max,
    }


def _compute_window_metrics(eval_obj: ClipEval) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for spec in WINDOW_SPECS:
        row = _window_metric(eval_obj, spec)
        out[str(spec["id"])] = row
    return out


def _selection_metric(window_metrics: Dict[str, Dict[str, Any]]) -> float:
    numer = 0.0
    denom = 0.0
    for key, weight in SELECTION_WINDOW_WEIGHTS:
        metric = window_metrics.get(str(key), {})
        value = float(metric.get("mean_deg", float("nan")))
        if math.isfinite(value):
            numer += float(weight) * value
            denom += float(weight)
    return numer / max(denom, 1e-8) if denom > 0 else float("inf")


def _coverage_note(window_metrics: Dict[str, Dict[str, Any]]) -> str:
    legs = window_metrics.get("legs_all", {})
    foot = window_metrics.get("foot_l_ball_l_12_15", {})
    return (
        f"legs coverage {legs.get('coverage', 'NA')}; "
        f"foot_l+ball_l coverage {foot.get('coverage', 'NA')}"
    )


def _make_decoder(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    return build_mlp(
        int(input_dim),
        int(hidden_dim),
        num_layers=int(num_layers),
        activation=nn.GELU,
        final_dim=int(output_dim),
    )


def _train_decoder_probe(
    *,
    features: torch.Tensor,
    target_raw: torch.Tensor,
    columns: Sequence[str],
    bone_names: Sequence[str],
    seed: int,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    hidden_dim: int,
    num_layers: int,
    lr: float,
    weight_decay: float,
    aux_mse_weight: float,
    min_delta_deg: float,
) -> tuple[nn.Sequential, SeedRun]:
    _set_seed(seed)
    x = features.detach().clone().float()
    feat_mean = x.mean(dim=0, keepdim=True)
    feat_std = x.std(dim=0, keepdim=True).clamp_min(1e-4)
    x = (x - feat_mean) / feat_std
    y = target_raw.detach().clone().float()

    joint_count = int(len(bone_names))
    target6 = reproject_rot6d(y).view(-1, joint_count, 6)
    target_R = rot6d_to_matrix(target6, columns=tuple(columns))

    decoder = _make_decoder(
        input_dim=int(x.shape[-1]),
        output_dim=int(y.shape[-1]),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
    )
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_windows: Optional[Dict[str, Dict[str, Any]]] = None
    best_metric = float("inf")
    best_epoch = -1
    best_loss = float("inf")
    best_geo = float("inf")
    best_mse = float("inf")
    stale = 0

    for epoch in range(1, int(max_epochs) + 1):
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        pred_raw = decoder(x)
        pred6 = reproject_rot6d(pred_raw).view(-1, joint_count, 6)
        pred_R = rot6d_to_matrix(pred6, columns=tuple(columns))
        loss_geo = geodesic_R(pred_R, target_R, reduce="mean")
        loss_mse = F.mse_loss(pred6.view(-1, joint_count * 6), target6.view(-1, joint_count * 6))
        loss = loss_geo + float(aux_mse_weight) * loss_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()

        decoder.eval()
        with torch.no_grad():
            pred_eval = decoder(x).detach().cpu().numpy().astype(np.float32, copy=False)
        clip_eval = _make_clip_eval(
            {
                "clip": "Walk_F",
                "npz_path": "raw_data/processed_data/Walk_F.npz",
                "source_json": "raw_data/Walk_F.json",
                "columns": tuple(columns),
                "bone_names": list(bone_names),
                "root_idx": -1,
                "pose_target_raw": y.cpu().numpy(),
            },
            pred_eval,
        )
        window_metrics = _compute_window_metrics(clip_eval)
        stop_metric = _selection_metric(window_metrics)

        if stop_metric < (best_metric - float(min_delta_deg)):
            best_state = copy.deepcopy(decoder.state_dict())
            best_windows = copy.deepcopy(window_metrics)
            best_metric = float(stop_metric)
            best_epoch = int(epoch)
            best_loss = float(loss.detach().cpu().item())
            best_geo = float(loss_geo.detach().cpu().item())
            best_mse = float(loss_mse.detach().cpu().item())
            stale = 0
        else:
            stale += 1

        if epoch >= int(min_epochs) and stale >= int(patience):
            break

    if best_state is None or best_windows is None:
        raise RuntimeError("decoder probe failed to record a best checkpoint")

    decoder.load_state_dict(best_state)
    run = SeedRun(
        seed=int(seed),
        best_epoch=int(best_epoch),
        stop_metric_deg=float(best_metric),
        train_loss=float(best_loss),
        train_geodesic_rad=float(best_geo),
        train_aux_mse=float(best_mse),
        window_metrics=best_windows,
        coverage_note=_coverage_note(best_windows),
    )
    return decoder, run


def _aggregate_window_metrics(seed_runs: Sequence[SeedRun]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for spec in WINDOW_SPECS:
        key = str(spec["id"])
        seed_rows = [run.window_metrics.get(key, {}) for run in seed_runs]
        means = [float(row.get("mean_deg", float("nan"))) for row in seed_rows]
        p90s = [float(row.get("p90_deg", float("nan"))) for row in seed_rows]
        mean_deg, std_deg = _mean_std(means)
        p90_deg, p90_std_deg = _mean_std(p90s)
        template = seed_rows[0] if seed_rows else {}
        out[key] = {
            "window_id": key,
            "label": str(spec["label"]),
            "mean_deg": mean_deg,
            "std_deg": std_deg,
            "p90_deg": p90_deg,
            "p90_std_deg": p90_std_deg,
            "requested_bones": list(template.get("requested_bones", list(spec["bones"]))),
            "available_bones": list(template.get("available_bones", [])),
            "coverage": str(template.get("coverage", "NA")),
            "sic_min": template.get("sic_min"),
            "sic_max": template.get("sic_max"),
            "seed_values_deg": means,
            "seed_p90_deg": p90s,
            "count": int(template.get("count", 0)),
        }
    return out


def _write_probe_outputs(
    probe_dir: Path,
    *,
    bundle_ctx: Dict[str, Any],
    probe_spec: ProbeSpec,
    clip_evals: Sequence[ClipEval],
    seed_runs: Sequence[SeedRun],
) -> ProbeSummary:
    probe_dir.mkdir(parents=True, exist_ok=True)
    agg = _aggregate(clip_evals)

    _write_matrix_csv(
        probe_dir / "per_sic_bone_mean_deg.csv",
        sics=agg["sics"],
        bones=agg["bone_names"],
        mat=agg["mean_mat"],
    )
    _write_matrix_csv(
        probe_dir / "per_sic_bone_p90_deg.csv",
        sics=agg["sics"],
        bones=agg["bone_names"],
        mat=agg["p90_mat"],
    )
    _write_count_csv(
        probe_dir / "per_sic_bone_count.csv",
        sics=agg["sics"],
        bones=agg["bone_names"],
        mat=agg["cnt_mat"],
    )
    _write_rows_csv(
        probe_dir / "top_pairs.csv",
        rows=agg["top_pairs"],
        fieldnames=["sic", "bone", "mean_deg", "p50_deg", "p90_deg", "count"],
    )

    seed_run_rows = [asdict(run) for run in seed_runs]
    coverage_note = seed_runs[0].coverage_note if seed_runs else "NA"
    window_metrics = _aggregate_window_metrics(seed_runs) if seed_runs else {}

    summary = ProbeSummary(
        probe_id=str(probe_spec.probe_id),
        label=str(probe_spec.label),
        bundle=str(bundle_ctx["bundle_name"]),
        seeds=[int(run.seed) for run in seed_runs],
        output_mode=str(probe_spec.output_mode),
        latent_kind=str(probe_spec.latent_kind),
        overall_mean_deg_ex_root=float(agg["overall_mean_deg_ex_root"]),
        window_metrics=window_metrics,
        seed_runs=seed_run_rows,
        coverage_note=coverage_note,
    )

    (probe_dir / "summary.json").write_text(
        json.dumps(
            {
                "bundle_path": str(bundle_ctx["bundle_path"]),
                "probe": asdict(summary),
                "bundle_meta": bundle_ctx["meta"],
                "clip": {
                    "clip": str(bundle_ctx["clip_data"]["clip"]),
                    "npz_path": str(bundle_ctx["clip_data"]["npz_path"]),
                    "source_json": str(bundle_ctx["clip_data"]["source_json"]),
                    "columns": list(bundle_ctx["clip_data"]["columns"]),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    md_lines = [
        f"# {probe_spec.label}",
        "",
        f"- bundle: `{bundle_ctx['bundle_path']}`",
        f"- probe_id: `{probe_spec.probe_id}`",
        f"- latent_kind: `{probe_spec.latent_kind}`",
        f"- output_mode: `{probe_spec.output_mode}`",
        f"- seeds: {', '.join(str(run.seed) for run in seed_runs) if seed_runs else 'none'}",
        f"- overall_mean_deg_ex_root: {_fmt_float(agg['overall_mean_deg_ex_root'], 6)}",
        f"- coverage: {coverage_note}",
        "",
        "## Key windows",
        "",
        "| window | mean_deg | std_deg | p90_deg | coverage |",
        "|---|---:|---:|---:|---|",
    ]
    for spec in WINDOW_SPECS:
        row = window_metrics.get(str(spec["id"]), {})
        md_lines.append(
            f"| {spec['label']} | {_fmt_float(row.get('mean_deg'))} | {_fmt_float(row.get('std_deg'))} | "
            f"{_fmt_float(row.get('p90_deg'))} | {row.get('coverage', 'NA')} |"
        )
    md_lines.extend(
        [
            "",
            "## Seed runs",
            "",
            "| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in seed_runs:
        md_lines.append(
            f"| {run.seed} | {run.best_epoch} | {_fmt_float(run.stop_metric_deg)} | {_fmt_float(run.train_loss, 6)} | "
            f"{_fmt_float(run.train_geodesic_rad, 6)} | {_fmt_float(run.train_aux_mse, 6)} |"
        )
    (probe_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


def _run_baseline(bundle_ctx: Dict[str, Any], probe_dir: Path) -> ProbeSummary:
    clip_eval = _evaluate_clip(
        bundle_ctx["clip_data"],
        encoder=bundle_ctx["encoder"],
        period_head=bundle_ctx["period_head"],
        decoder_pose=bundle_ctx["decoder_pose"],
        pose_target_norm=bundle_ctx["pose_target_norm"],
        projectors=bundle_ctx["projectors"],
        device=bundle_ctx["device"],
    )
    seed_run = SeedRun(
        seed=0,
        best_epoch=0,
        stop_metric_deg=float(_selection_metric(_compute_window_metrics(clip_eval))),
        train_loss=float("nan"),
        train_geodesic_rad=float("nan"),
        train_aux_mse=float("nan"),
        window_metrics=_compute_window_metrics(clip_eval),
        coverage_note=_coverage_note(_compute_window_metrics(clip_eval)),
    )
    return _write_probe_outputs(
        probe_dir,
        bundle_ctx=bundle_ctx,
        probe_spec=ProbeSpec(
            probe_id="baseline_frozen",
            label="baseline frozen decoder",
            latent_kind="soft_period",
            output_mode="baseline",
        ),
        clip_evals=[clip_eval],
        seed_runs=[seed_run],
    )


def _run_oracle_probe(
    bundle_ctx: Dict[str, Any],
    probe_spec: ProbeSpec,
    probe_dir: Path,
    *,
    seeds: Sequence[int],
    max_epochs: int,
    min_epochs: int,
    patience: int,
    hidden_dim: int,
    num_layers: int,
    lr: float,
    weight_decay: float,
    aux_mse_weight: float,
    min_delta_deg: float,
) -> ProbeSummary:
    clip_data = bundle_ctx["clip_data"]
    full_bone_names = list(clip_data["bone_names"])
    columns = tuple(clip_data["columns"])
    pose_target_raw = torch.from_numpy(np.asarray(clip_data["pose_target_raw"], dtype=np.float32))

    if probe_spec.latent_kind == "soft_period":
        features = bundle_ctx["soft_period"]
    elif probe_spec.latent_kind == "h_period":
        features = bundle_ctx["h_period"]
    else:
        raise RuntimeError(f"unsupported latent_kind: {probe_spec.latent_kind}")

    bone_indices: Optional[List[int]] = None
    target = pose_target_raw
    bone_names = full_bone_names
    if probe_spec.output_mode == "calf_only":
        bone_indices = [int(full_bone_names.index("calf_l")), int(full_bone_names.index("calf_r"))]
        target = torch.from_numpy(
            _subset_pose_raw(np.asarray(clip_data["pose_target_raw"], dtype=np.float32), len(full_bone_names), bone_indices)
        )
        bone_names = [full_bone_names[i] for i in bone_indices]
    elif probe_spec.output_mode != "full":
        raise RuntimeError(f"unsupported output_mode: {probe_spec.output_mode}")

    clip_evals: List[ClipEval] = []
    seed_runs: List[SeedRun] = []
    for seed in [int(s) for s in seeds]:
        decoder, run = _train_decoder_probe(
            features=features,
            target_raw=target,
            columns=columns,
            bone_names=bone_names,
            seed=seed,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            patience=patience,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            aux_mse_weight=aux_mse_weight,
            min_delta_deg=min_delta_deg,
        )
        with torch.no_grad():
            x = features.detach().clone().float()
            x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1e-4)
            pred = decoder(x).detach().cpu().numpy().astype(np.float32, copy=False)
        clip_evals.append(_make_clip_eval(clip_data, pred, bone_indices=bone_indices))
        seed_runs.append(run)

    return _write_probe_outputs(
        probe_dir,
        bundle_ctx=bundle_ctx,
        probe_spec=probe_spec,
        clip_evals=clip_evals,
        seed_runs=seed_runs,
    )


def _window_value(summary: ProbeSummary, key: str) -> float:
    row = summary.window_metrics.get(str(key), {})
    return float(row.get("mean_deg", float("nan")))


def _calf_hotspot_mean(summary: ProbeSummary) -> float:
    vals = [
        _window_value(summary, "calf_r_56_62"),
        _window_value(summary, "calf_l_78_85"),
    ]
    finite = [v for v in vals if math.isfinite(v)]
    return float(mean(finite)) if finite else float("nan")


def _bundle_hypothesis_verdict(bundle: str, probe_map: Dict[str, ProbeSummary]) -> Dict[str, Any]:
    baseline = probe_map["baseline_frozen"]
    soft_full = probe_map["oracle_soft_period_full"]
    soft_calf = probe_map["oracle_soft_period_calf_only"]
    h_full = probe_map["oracle_h_period_full"]

    baseline_score = _calf_hotspot_mean(baseline)
    soft_full_score = _calf_hotspot_mean(soft_full)
    soft_calf_score = _calf_hotspot_mean(soft_calf)
    h_full_score = _calf_hotspot_mean(h_full)

    supports_h1 = (
        math.isfinite(soft_full_score)
        and math.isfinite(baseline_score)
        and soft_full_score < 15.0
        and soft_full_score <= (baseline_score - 5.0)
    )
    supports_h3 = (
        math.isfinite(soft_calf_score)
        and math.isfinite(soft_full_score)
        and soft_calf_score <= (soft_full_score - 3.0)
    )
    supports_h2 = (
        math.isfinite(h_full_score)
        and math.isfinite(soft_full_score)
        and math.isfinite(soft_calf_score)
        and soft_full_score > 15.0
        and soft_calf_score > 15.0
        and h_full_score <= min(soft_full_score, soft_calf_score) - 5.0
    )

    if supports_h1:
        verdict = "H1"
    elif supports_h3:
        verdict = "H3"
    elif supports_h2:
        verdict = "H2"
    else:
        verdict = "mixed_or_earlier"

    explanation_parts = [
        f"baseline hotspot mean={_fmt_float(baseline_score)}deg",
        f"soft_period->full={_fmt_float(soft_full_score)}deg",
        f"soft_period->calf_only={_fmt_float(soft_calf_score)}deg",
        f"h_period->full={_fmt_float(h_full_score)}deg",
    ]
    if verdict == "H1":
        explanation_parts.append("soft_period full decoder already removes the calf hotspot, so latent information is present")
        if math.isfinite(h_full_score) and math.isfinite(soft_full_score) and h_full_score < soft_full_score - 0.5:
            explanation_parts.append("h_period still leaves some headroom, but it is secondary because soft_period is already sufficient")
    elif verdict == "H3":
        explanation_parts.append("calf-only readout is materially better than shared full-pose readout")
    elif verdict == "H2":
        explanation_parts.append("soft_period probes stay weak while h_period full is much stronger")
    else:
        explanation_parts.append("even oracle probes do not cleanly separate the hypotheses")

    return {
        "bundle": bundle,
        "verdict": verdict,
        "baseline_hotspot_mean_deg": baseline_score,
        "soft_period_full_hotspot_mean_deg": soft_full_score,
        "soft_period_calf_only_hotspot_mean_deg": soft_calf_score,
        "h_period_full_hotspot_mean_deg": h_full_score,
        "explanation": "; ".join(explanation_parts),
    }


def _write_compare_windows_csv(path: Path, bundle_probe_rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "bundle",
        "probe",
        "seeds",
        "coverage_note",
    ]
    for spec in WINDOW_SPECS:
        key = str(spec["id"])
        fieldnames.extend(
            [
                f"{key}_mean_deg",
                f"{key}_std_deg",
                f"{key}_coverage",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in bundle_probe_rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Oracle decoder probe for frozen pretrain latent information audits.")
    ap.add_argument(
        "--bundle",
        action="append",
        default=None,
        help="Repeatable bundle path. Defaults to stageA and pt.best bundles.",
    )
    ap.add_argument("--norm-spec", type=str, default="models/pretrain_template.json")
    ap.add_argument("--npz", type=str, default="raw_data/processed_data/Walk_F.npz")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-dir", type=str, default=_default_out_dir())
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--max-epochs", type=int, default=800)
    ap.add_argument("--min-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--decoder-hidden-dim", type=int, default=256)
    ap.add_argument("--decoder-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--aux-mse-weight", type=float, default=0.02)
    ap.add_argument("--min-delta-deg", type=float, default=1e-4)
    args = ap.parse_args()

    bundle_specs = args.bundle or [
        "models/motion_encoder_equiv_stageA.pt",
        "models/motion_encoder_equiv.pt.best.pt",
    ]
    seeds = [int(tok.strip()) for tok in str(args.seeds).split(",") if tok.strip()]
    if not seeds:
        raise SystemExit("[FATAL] --seeds resolved to an empty list")

    norm_path = Path(args.norm_spec).expanduser().resolve()
    npz_path = Path(args.npz).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)

    probe_specs = [
        ProbeSpec(
            probe_id="oracle_soft_period_full",
            label="oracle soft_period full decoder",
            latent_kind="soft_period",
            output_mode="full",
        ),
        ProbeSpec(
            probe_id="oracle_soft_period_calf_only",
            label="oracle soft_period calf-only decoder",
            latent_kind="soft_period",
            output_mode="calf_only",
        ),
        ProbeSpec(
            probe_id="oracle_h_period_full",
            label="oracle h_period full decoder",
            latent_kind="h_period",
            output_mode="full",
        ),
    ]

    all_results: Dict[str, Dict[str, ProbeSummary]] = {}
    compare_rows: List[Dict[str, Any]] = []

    for bundle_spec in bundle_specs:
        bundle_path = Path(bundle_spec).expanduser().resolve()
        bundle_ctx = _load_bundle_context(bundle_path, norm_path, npz_path, device)
        bundle_name = str(bundle_ctx["bundle_name"])
        bundle_dir = out_dir / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Probe] bundle={bundle_name} path={bundle_path}")

        bundle_results: Dict[str, ProbeSummary] = {}
        baseline_summary = _run_baseline(bundle_ctx, bundle_dir / "baseline_frozen")
        bundle_results["baseline_frozen"] = baseline_summary
        print(
            "[Probe] baseline"
            f" calf_r56-62={_fmt_float(_window_value(baseline_summary, 'calf_r_56_62'))}"
            f" calf_l78-85={_fmt_float(_window_value(baseline_summary, 'calf_l_78_85'))}"
        )

        for spec in probe_specs:
            print(f"[Probe] train {bundle_name} / {spec.probe_id} / seeds={seeds}")
            summary = _run_oracle_probe(
                bundle_ctx,
                spec,
                bundle_dir / spec.probe_id,
                seeds=seeds,
                max_epochs=int(args.max_epochs),
                min_epochs=int(args.min_epochs),
                patience=int(args.patience),
                hidden_dim=int(args.decoder_hidden_dim),
                num_layers=int(args.decoder_layers),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                aux_mse_weight=float(args.aux_mse_weight),
                min_delta_deg=float(args.min_delta_deg),
            )
            bundle_results[str(spec.probe_id)] = summary
            print(
                "[Probe] done"
                f" {spec.probe_id}"
                f" calf_r56-62={_fmt_float(_window_value(summary, 'calf_r_56_62'))}"
                f" calf_l78-85={_fmt_float(_window_value(summary, 'calf_l_78_85'))}"
            )

        all_results[bundle_name] = bundle_results

        for probe_id, summary in bundle_results.items():
            row = {
                "bundle": bundle_name,
                "probe": probe_id,
                "seeds": ",".join(str(s) for s in summary.seeds),
                "coverage_note": summary.coverage_note,
            }
            for spec in WINDOW_SPECS:
                key = str(spec["id"])
                metric = summary.window_metrics.get(key, {})
                row[f"{key}_mean_deg"] = _fmt_float(metric.get("mean_deg"))
                row[f"{key}_std_deg"] = _fmt_float(metric.get("std_deg"))
                row[f"{key}_coverage"] = metric.get("coverage", "NA")
            compare_rows.append(row)

    verdicts = [_bundle_hypothesis_verdict(bundle, probe_map) for bundle, probe_map in sorted(all_results.items())]

    overall_verdict = "mixed"
    verdict_labels = [str(v["verdict"]) for v in verdicts]
    if verdict_labels and all(v == "H1" for v in verdict_labels):
        overall_verdict = "H1"
    elif "H1" in verdict_labels and not any(v in ("H2", "H3") for v in verdict_labels):
        overall_verdict = "H1_leaning"
    elif verdict_labels and all(v == "H3" for v in verdict_labels):
        overall_verdict = "H3"
    elif verdict_labels and all(v == "H2" for v in verdict_labels):
        overall_verdict = "H2"

    summary_payload = {
        "config": {
            "bundles": [str(Path(p).expanduser().resolve()) for p in bundle_specs],
            "norm_spec": str(norm_path),
            "npz": str(npz_path),
            "device": str(device),
            "out_dir": str(out_dir),
            "seeds": seeds,
            "max_epochs": int(args.max_epochs),
            "min_epochs": int(args.min_epochs),
            "patience": int(args.patience),
            "decoder_hidden_dim": int(args.decoder_hidden_dim),
            "decoder_layers": int(args.decoder_layers),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "aux_mse_weight": float(args.aux_mse_weight),
            "min_delta_deg": float(args.min_delta_deg),
        },
        "overall_verdict": overall_verdict,
        "bundle_verdicts": verdicts,
        "results": {
            bundle: {probe_id: asdict(summary) for probe_id, summary in probe_map.items()}
            for bundle, probe_map in sorted(all_results.items())
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_compare_windows_csv(out_dir / "comparison_windows.csv", compare_rows)

    md_lines = [
        "# pretrain latent information audit / oracle decoder probe",
        "",
        "## Setup",
        f"- clip: `{npz_path}`",
        f"- norm_spec: `{norm_path}`",
        f"- device: `{device}`",
        f"- seeds: {', '.join(str(s) for s in seeds)}",
        f"- oracle decoder: hidden_dim={int(args.decoder_hidden_dim)}, layers={int(args.decoder_layers)}, lr={float(args.lr):.6g}",
        f"- overall verdict: `{overall_verdict}`",
        "",
        "## Window compare",
        "",
        "| bundle | probe | calf_r @ 56-62 | calf_l @ 78-85 | foot_l+ball_l @ 12-15 | legs @ all sic | legs @ sic>=40 | coverage |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for bundle in sorted(all_results.keys()):
        for probe_id in ("baseline_frozen", "oracle_soft_period_full", "oracle_soft_period_calf_only", "oracle_h_period_full"):
            summary = all_results[bundle][probe_id]
            md_lines.append(
                f"| {bundle} | {probe_id} | {_fmt_float(_window_value(summary, 'calf_r_56_62'))} +/- {_fmt_float(summary.window_metrics['calf_r_56_62'].get('std_deg'))} | "
                f"{_fmt_float(_window_value(summary, 'calf_l_78_85'))} +/- {_fmt_float(summary.window_metrics['calf_l_78_85'].get('std_deg'))} | "
                f"{_fmt_float(_window_value(summary, 'foot_l_ball_l_12_15'))} +/- {_fmt_float(summary.window_metrics['foot_l_ball_l_12_15'].get('std_deg'))} | "
                f"{_fmt_float(_window_value(summary, 'legs_all'))} +/- {_fmt_float(summary.window_metrics['legs_all'].get('std_deg'))} | "
                f"{_fmt_float(_window_value(summary, 'legs_ge_40'))} +/- {_fmt_float(summary.window_metrics['legs_ge_40'].get('std_deg'))} | "
                f"{summary.coverage_note} |"
            )
    md_lines.extend(["", "## Hypothesis interpretation", ""])
    for verdict in verdicts:
        md_lines.append(
            f"- `{verdict['bundle']}` -> `{verdict['verdict']}`: {verdict['explanation']}."
        )
    md_lines.extend(
        [
            "",
            "## Decision rules used",
            "",
            "- If `soft_period -> full decoder` already pulls the calf hotspot well below 15deg, treat that as strong H1 support.",
            "- If `soft_period -> full decoder` is limited but `soft_period -> calf-only decoder` is much better, treat that as H3 support.",
            "- If both `soft_period` probes stay weak while `h_period -> full decoder` is much better, treat that as H2 support.",
            "- If even `h_period -> full decoder` stays weak, treat the issue as earlier than the period bottleneck or structurally ambiguous on the single clip.",
            "",
            "## Notes",
            "",
            "- This is a single-clip oracle extraction probe on `Walk_F`, so it measures recoverability from frozen latent, not cross-clip generalization.",
            "- `oracle_soft_period_calf_only` only predicts `calf_l` and `calf_r`; its leg metrics therefore cover the calf overlap only, and `foot_l + ball_l` is NA by design.",
            "- All oracle losses use `reproject_rot6d -> rot6d_to_matrix -> geodesic_R` as the main term, with a small auxiliary 6D MSE.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {out_dir / 'summary.json'}")
    print(f"[OK] wrote {out_dir / 'summary.md'}")
    print(f"[OK] wrote {out_dir / 'comparison_windows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
