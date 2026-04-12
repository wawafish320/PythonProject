#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.stage6_anchor_boundary import (  # noqa: E402
    apply_locked_boundary,
    canonical_affine_stats,
    canonical_encoder_bundle,
    canonical_teacher,
    candidate_pool_policy,
    payload_contract_ok,
)


DEFAULT_TEACHER = canonical_teacher(ROOT)
DEFAULT_ENCODER_BUNDLE = canonical_encoder_bundle(ROOT)
DEFAULT_AFFINE_STATS = canonical_affine_stats(ROOT)
LEG8_BONES = ["thigh_r", "calf_r", "foot_r", "ball_r", "thigh_l", "calf_l", "foot_l", "ball_l"]


@dataclass
class Candidate:
    name: str
    ckpt: Path
    selector: str
    family: str
    basetrain_epoch: Optional[int] = None
    discovery_tags: Tuple[str, ...] = ()


@dataclass
class CandidatePaths:
    lane_root: Path
    lane_log: Path
    eval_dir: Path
    eval_json: Path


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _emit_best_handoff_ckpt(*, selected_row: Optional[Mapping[str, Any]], out_dir: Path, run_name: str) -> Optional[str]:
    if not isinstance(selected_row, Mapping):
        return None
    src = Path(str(selected_row.get("ckpt", ""))).expanduser()
    if not src.is_file():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"ckpt_best_handoff_{str(run_name)}.pth"
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    except Exception:
        pass
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        shutil.copy2(src, dst)
    return str(dst)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _fmt(value: Any, nd: int = 6) -> str:
    value = _safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{nd}f}"


def _nanmean(values: Iterable[Any]) -> float:
    finite = [_safe_float(v) for v in values]
    finite = [v for v in finite if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _build_paths(*, out_root: Path, name: str) -> CandidatePaths:
    lane_root = out_root / name
    return CandidatePaths(
        lane_root=lane_root,
        lane_log=lane_root / "handoff_eval.log",
        eval_dir=lane_root / "handoff_eval",
        eval_json=lane_root / "handoff_eval" / "Walk_F_freerun_cycles.json",
    )


def _run_cmd(cmd: Sequence[str], *, log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(str(x) for x in cmd) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = int(proc.wait())
        handle.write(f"[exit_code] {code}\n")
    if code != 0:
        raise SystemExit(code)


def _maybe_load_metric(exp_dir: Path, tag: str, epoch: Optional[int]) -> Dict[str, Any] | None:
    if epoch is None:
        return None
    path = exp_dir / "metrics" / f"{tag}_ep{int(epoch):03d}.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    metrics = payload.get("metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else None


def _find_max_epoch(exp_dir: Path) -> Optional[int]:
    metrics_dir = exp_dir / "metrics"
    epochs: List[int] = []
    for path in metrics_dir.glob("valfree_ep*.json"):
        stem = path.stem
        try:
            epochs.append(int(stem.rsplit("ep", 1)[1]))
        except Exception:
            continue
    return max(epochs) if epochs else None


def _load_basetrain_summary(exp_dir: Path) -> Dict[str, Any]:
    path = exp_dir / "basetrain_keybone_group_summary.json"
    return _load_json(path) if path.is_file() else {}


def _extract_summary_epoch(summary: Mapping[str, Any], key: str) -> Optional[int]:
    payload = summary.get(key)
    if not isinstance(payload, Mapping):
        return None
    try:
        return int(payload.get("epoch"))
    except Exception:
        return None


def _discover_epoch_ckpts(exp_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for path in sorted(exp_dir.glob("ckpt_epoch_*.pth")):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 4:
            continue
        try:
            epoch = int(parts[2])
        except Exception:
            continue
        out[int(epoch)] = path
    return out


def _discover_candidates(exp_dir: Path, *, smoke_only: bool) -> List[Candidate]:
    run_name = exp_dir.name
    summary = _load_basetrain_summary(exp_dir)
    last_epoch = _find_max_epoch(exp_dir)
    rows: List[Candidate] = []
    specs = [
        ("best_free", exp_dir / f"ckpt_best_free_{run_name}.pth", _extract_summary_epoch(summary, "best_free_by_GeoDriftSlopeProxy")),
        ("last", exp_dir / f"ckpt_last_{run_name}.pth", last_epoch),
        ("best_teacher", exp_dir / f"ckpt_best_teacher_{run_name}.pth", _extract_summary_epoch(summary, "best_teacher_by_GeoLocalDeg")),
    ]
    for selector, ckpt, epoch in specs:
        if smoke_only and selector != "best_free":
            continue
        if ckpt.is_file():
            rows.append(
                Candidate(
                    name=selector,
                    ckpt=ckpt,
                    selector=selector,
                    family=run_name,
                    basetrain_epoch=epoch,
                    discovery_tags=("saved_selector",),
                )
            )
    if not rows:
        raise SystemExit(f"[FATAL] no saved selector checkpoints found under: {exp_dir}")
    return rows


def _extract_group_mean(metrics: Mapping[str, Any], group_name: str) -> float:
    summary = metrics.get("KeyBoneSummary", {})
    if isinstance(summary, Mapping):
        group_mean = summary.get("group_mean", {})
        if isinstance(group_mean, Mapping):
            return _safe_float(group_mean.get(group_name))
    return float("nan")


def _compute_geo_deg_slope(metrics: Mapping[str, Any]) -> float:
    curve = metrics.get("GeoDegCurve")
    if not isinstance(curve, list) or not curve:
        start = _safe_float(metrics.get("GeoDegStart", metrics.get("GeoDeg")))
        end = _safe_float(metrics.get("GeoDegEnd", start))
        horizon = int(metrics.get("eval_horizon", 0) or 0)
        return (end - start) / max(1, horizon - 1)
    if isinstance(curve[0], (list, tuple)) and curve[0]:
        horizon = len(curve[0])
        mean_curve: List[float] = []
        for step_idx in range(horizon):
            vals = []
            for batch_curve in curve:
                if isinstance(batch_curve, (list, tuple)) and step_idx < len(batch_curve):
                    value = _safe_float(batch_curve[step_idx])
                    if math.isfinite(value):
                        vals.append(value)
            if vals:
                mean_curve.append(float(sum(vals) / len(vals)))
    else:
        mean_curve = [_safe_float(v) for v in curve if math.isfinite(_safe_float(v))]
    if len(mean_curve) < 2:
        return float("inf")
    return float((mean_curve[-1] - mean_curve[0]) / max(1, len(mean_curve) - 1))


def _build_proxy_epoch_rows(exp_dir: Path, *, epoch_start: int, epoch_end: int, topk: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for epoch in range(int(epoch_start), int(epoch_end) + 1):
        free_metrics = _maybe_load_metric(exp_dir, "valfree", epoch)
        teacher_metrics = _maybe_load_metric(exp_dir, "teacher", epoch)
        if free_metrics is None or teacher_metrics is None:
            continue
        row = {
            "epoch": int(epoch),
            "freerun": {
                "geo_deg": _safe_float(free_metrics.get("GeoDeg")),
                "geo_local_deg": _safe_float(free_metrics.get("GeoLocalDeg")),
                "geo_deg_slope": _compute_geo_deg_slope(free_metrics),
                "geo_local_proxy": _safe_float(free_metrics.get("GeoDriftSlopeProxy")),
                "root_vel_mae": _safe_float(free_metrics.get("RootVelMAE")),
                "arm_mean": _extract_group_mean(free_metrics, "arm"),
                "trunk_mean": _extract_group_mean(free_metrics, "trunk"),
                "leg_mean": _extract_group_mean(free_metrics, "leg"),
            },
            "teacher": {
                "geo_local_deg": _safe_float(teacher_metrics.get("GeoLocalDeg")),
                "arm_mean": _extract_group_mean(teacher_metrics, "arm"),
                "trunk_mean": _extract_group_mean(teacher_metrics, "trunk"),
                "leg_mean": _extract_group_mean(teacher_metrics, "leg"),
            },
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            _safe_float(row["freerun"]["geo_deg_slope"]),
            _safe_float(row["freerun"]["arm_mean"]),
            _safe_float(row["freerun"]["trunk_mean"]),
            _safe_float(row["freerun"]["leg_mean"]),
            _safe_float(row["teacher"]["geo_local_deg"]),
            int(row["epoch"]),
        )
    )
    return [{"rank": idx + 1, "row": row} for idx, row in enumerate(rows[: int(topk)])]


def _discover_exact_epoch_candidates(
    exp_dir: Path,
    *,
    base_candidates: Sequence[Candidate],
    proxy_epoch_rows: Sequence[Dict[str, Any]],
    exact_topk: int,
    exact_window: int,
) -> List[Candidate]:
    epoch_ckpts = _discover_epoch_ckpts(exp_dir)
    if not epoch_ckpts:
        return []

    by_epoch: Dict[int, List[str]] = {}
    for cand in base_candidates:
        if cand.basetrain_epoch is None:
            continue
        by_epoch.setdefault(int(cand.basetrain_epoch), []).extend(list(cand.discovery_tags) or [cand.selector])

    summary = _load_basetrain_summary(exp_dir)
    best_free_epoch = _extract_summary_epoch(summary, "best_free_by_GeoDriftSlopeProxy")
    wanted: Dict[int, List[str]] = {}

    if best_free_epoch is not None:
        for epoch in range(int(best_free_epoch) - int(exact_window), int(best_free_epoch) + int(exact_window) + 1):
            if epoch in epoch_ckpts:
                wanted.setdefault(int(epoch), []).append("best_free_window")

    for item in list(proxy_epoch_rows)[: int(exact_topk)]:
        row = item.get("row", {})
        try:
            epoch = int(row.get("epoch"))
        except Exception:
            continue
        if epoch in epoch_ckpts:
            wanted.setdefault(int(epoch), []).append("proxy_topk")

    rows: List[Candidate] = []
    for epoch, tags in sorted(wanted.items()):
        if epoch in by_epoch:
            continue
        rows.append(
            Candidate(
                name=f"epoch{epoch:03d}",
                ckpt=epoch_ckpts[epoch],
                selector="epoch_exact",
                family=exp_dir.name,
                basetrain_epoch=int(epoch),
                discovery_tags=tuple(sorted(set(tags))),
            )
        )
    return rows


def _run_candidate_eval(args: argparse.Namespace, cand: Candidate, paths: CandidatePaths) -> None:
    if paths.eval_json.is_file() and not bool(args.force_rerun):
        return
    _run_cmd(
        [
            sys.executable,
            "-m",
            "train.validate.run_freerun_cycles",
            "--teacher",
            str(args.teacher),
            "--model",
            str(cand.ckpt),
            "--rounds",
            str(int(args.rounds)),
            "--depth",
            str(int(args.depth)),
            "--time-index-mode",
            str(args.time_index_mode),
            "--event_clock",
            str(args.event_clock),
            "--phase_reset_source",
            str(args.phase_reset_source),
            "--contacts_meas_source",
            "pretrain_contact",
            "--contacts_meas_pretrain_clamp",
            str(args.pretrain_clamp),
            "--contacts_meas_pretrain_affine_stats",
            str(args.affine_stats),
            "--encoder-bundle",
            str(args.encoder_bundle),
            "--log_contacts",
            "--export_joint_direct_geolocal_series",
            "--analyze_phase_shift",
            "--out",
            str(paths.eval_dir),
            "--force",
        ],
        log_path=paths.lane_log,
    )


def _build_step_mask(steps: Sequence[Mapping[str, Any]], *, cycle_gte: int, drop_wrap: bool) -> List[bool]:
    mask: List[bool] = []
    for step in steps:
        try:
            cycle = int(step.get("cycle", 0) or 0)
        except Exception:
            cycle = 0
        if cycle < int(cycle_gte):
            mask.append(False)
            continue
        if drop_wrap and bool(step.get("wrap_boundary_step", False)):
            mask.append(False)
            continue
        mask.append(True)
    return mask


def _build_round_mask(rounds: Sequence[Mapping[str, Any]], *, cycle_gte: int) -> List[bool]:
    mask: List[bool] = []
    for item in rounds:
        try:
            round_idx = int(item.get("round", 0) or 0)
        except Exception:
            round_idx = 0
        mask.append(round_idx >= int(cycle_gte))
    return mask


def _mean_from_mask(values: Sequence[Any], mask: Sequence[bool]) -> float:
    picked: List[float] = []
    for keep, value in zip(mask, values):
        if not keep:
            continue
        value = _safe_float(value)
        if math.isfinite(value):
            picked.append(value)
    return _nanmean(picked)


def _mean_direct_deg(
    per_step: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    step_mask: Sequence[bool],
    *,
    bones: Sequence[str],
    sic_lo: Optional[int],
    sic_hi: Optional[int],
) -> float:
    names = per_step.get("bone_names")
    mat = per_step.get("DirectGeoLocalDeg")
    if not isinstance(names, list) or not isinstance(mat, list):
        return float("nan")
    name_to_idx = {str(name): idx for idx, name in enumerate(names)}
    indices: List[int] = []
    for bone in bones:
        idx = name_to_idx.get(str(bone))
        if idx is not None:
            indices.append(int(idx))
    if not indices:
        return float("nan")
    vals: List[float] = []
    for keep, step, row in zip(step_mask, steps, mat):
        if not keep or not isinstance(row, list):
            continue
        sic = step.get("step_in_cycle", step.get("sic"))
        try:
            sic_i = int(sic)
        except Exception:
            sic_i = None
        if sic_lo is not None and sic_hi is not None:
            if sic_i is None or sic_i < int(sic_lo) or sic_i > int(sic_hi):
                continue
        for idx in indices:
            if idx >= len(row):
                continue
            value = _safe_float(row[idx])
            if math.isfinite(value):
                vals.append(value)
    return _nanmean(vals)


def _extract_phase_shift_metrics(payload: Mapping[str, Any], *, cycle_gte: int) -> Dict[str, Any]:
    phase_shift = payload.get("phase_shift")
    if not isinstance(phase_shift, Mapping):
        return {
            "phase_shift_contact_plan_abs_mean": float("nan"),
            "phase_shift_contact_plan_mse_mean": float("nan"),
            "phase_shift_contact_plan_mse0_mean": float("nan"),
            "phase_shift_direct_geo_at_plan_shift_mean": float("nan"),
            "phase_shift_direct_geo_zero_mean": float("nan"),
            "phase_shift_direct_geo_gain": float("nan"),
            "phase_shift_cycles_kept": 0,
        }

    kept_cycles: List[Mapping[str, Any]] = []
    for item in phase_shift.get("cycles", []):
        if not isinstance(item, Mapping):
            continue
        try:
            cycle = int(item.get("cycle", 0) or 0)
        except Exception:
            cycle = 0
        if cycle >= int(cycle_gte):
            kept_cycles.append(item)

    shifts: List[float] = []
    plan_mse: List[float] = []
    plan_mse0: List[float] = []
    direct_at_plan: List[float] = []
    direct_zero: List[float] = []

    for item in kept_cycles:
        contact_plan = item.get("contact_plan", {})
        if isinstance(contact_plan, Mapping):
            shift = _safe_float(contact_plan.get("shift"))
            if math.isfinite(shift):
                shifts.append(abs(shift))
            mse = _safe_float(contact_plan.get("mse"))
            if math.isfinite(mse):
                plan_mse.append(mse)
            mse0 = _safe_float(contact_plan.get("mse0"))
            if math.isfinite(mse0):
                plan_mse0.append(mse0)
        at_plan = _safe_float(item.get("direct_geo_local_deg_mean_at_plan_shift"))
        if math.isfinite(at_plan):
            direct_at_plan.append(at_plan)
        direct_pose = item.get("direct_pose", {})
        if isinstance(direct_pose, Mapping):
            zero = _safe_float(direct_pose.get("geo_local_deg_mean0"))
            if math.isfinite(zero):
                direct_zero.append(zero)

    direct_at_plan_mean = _nanmean(direct_at_plan)
    direct_zero_mean = _nanmean(direct_zero)
    return {
        "phase_shift_contact_plan_abs_mean": _nanmean(shifts),
        "phase_shift_contact_plan_mse_mean": _nanmean(plan_mse),
        "phase_shift_contact_plan_mse0_mean": _nanmean(plan_mse0),
        "phase_shift_direct_geo_at_plan_shift_mean": direct_at_plan_mean,
        "phase_shift_direct_geo_zero_mean": direct_zero_mean,
        "phase_shift_direct_geo_gain": direct_at_plan_mean - direct_zero_mean
        if math.isfinite(direct_at_plan_mean) and math.isfinite(direct_zero_mean)
        else float("nan"),
        "phase_shift_cycles_kept": int(len(kept_cycles)),
    }


def _build_candidate_row(
    *,
    cand: Candidate,
    paths: CandidatePaths,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    payload = _load_json(paths.eval_json)
    steps = payload.get("metrics_per_step", [])
    rounds = payload.get("metrics_per_round", [])
    per_step_direct = payload.get("per_step_direct_geolocal_deg", {})
    if not isinstance(steps, list) or not isinstance(rounds, list):
        raise SystemExit(f"[FATAL] invalid eval json: missing metrics_per_step/metrics_per_round: {paths.eval_json}")

    step_mask = _build_step_mask(steps, cycle_gte=int(args.group_cycle_gte), drop_wrap=True)
    round_mask = _build_round_mask(rounds, cycle_gte=int(args.group_cycle_gte))

    schema = {
        "has_metrics_per_step": bool(steps),
        "has_metrics_per_round": bool(rounds),
        "has_per_step_direct_geolocal_deg": isinstance(per_step_direct, Mapping) and bool(per_step_direct),
        "has_contact_err_abs_mean": math.isfinite(_mean_from_mask([st.get("ContactErrAbsMean") for st in steps], step_mask)),
        "has_contact_plan_gt_abs_mean": math.isfinite(_mean_from_mask([st.get("ContactPlanGtAbsMean") for st in steps], step_mask)),
        "has_phase_shift": isinstance(payload.get("phase_shift"), Mapping),
    }

    phase_shift_metrics = _extract_phase_shift_metrics(payload, cycle_gte=int(args.group_cycle_gte))
    metrics = {
        "contact_err_abs_mean": _mean_from_mask([st.get("ContactErrAbsMean") for st in steps], step_mask),
        "contact_plan_gt_abs_mean": _mean_from_mask([st.get("ContactPlanGtAbsMean") for st in steps], step_mask),
        "contact_meas_gt_abs_mean": _mean_from_mask([st.get("ContactMeasGtAbsMean") for st in steps], step_mask),
        "geo_local_deg_weighted": _mean_from_mask([rd.get("GeoLocalDegWeighted") for rd in rounds], round_mask),
        "geo_local_deg": _mean_from_mask([rd.get("GeoLocalDeg") for rd in rounds], round_mask),
        "keybone_geo_local_deg_mean": _mean_from_mask([rd.get("KeyBoneGeoLocalDegMean") for rd in rounds], round_mask),
        "leg8_mean": _mean_direct_deg(per_step_direct, steps, step_mask, bones=LEG8_BONES, sic_lo=None, sic_hi=None),
        "sic12_15_foot_l_ball_l_mean": _mean_direct_deg(
            per_step_direct, steps, step_mask, bones=["foot_l", "ball_l"], sic_lo=12, sic_hi=15
        ),
        "calf_r_global_mean": _mean_direct_deg(per_step_direct, steps, step_mask, bones=["calf_r"], sic_lo=None, sic_hi=None),
        "calf_r_sic2_4_mean": _mean_direct_deg(per_step_direct, steps, step_mask, bones=["calf_r"], sic_lo=2, sic_hi=4),
    }
    metrics.update(phase_shift_metrics)

    exp_dir = _resolve(args.exp_dir) if str(args.exp_dir).strip() else None
    valfree_metrics = _maybe_load_metric(exp_dir, "valfree", cand.basetrain_epoch) if exp_dir else None
    teacher_metrics = _maybe_load_metric(exp_dir, "teacher", cand.basetrain_epoch) if exp_dir else None
    basetrain_metrics = {
        "epoch": cand.basetrain_epoch,
        "valfree_geo_local_deg": _safe_float(valfree_metrics.get("GeoLocalDeg")) if valfree_metrics else float("nan"),
        "valfree_geo_drift_slope_proxy": _safe_float(valfree_metrics.get("GeoDriftSlopeProxy")) if valfree_metrics else float("nan"),
        "valfree_root_vel_mae": _safe_float(valfree_metrics.get("RootVelMAE")) if valfree_metrics else float("nan"),
        "teacher_geo_local_deg": _safe_float(teacher_metrics.get("GeoLocalDeg")) if teacher_metrics else float("nan"),
    }

    contract = {
        "rounds": payload.get("rounds"),
        "requested_depth": int(args.depth),
        "time_index_mode": payload.get("time_index_mode"),
        "phase_reset_source": payload.get("phase_reset_source"),
        "event_clock": payload.get("event_clock"),
        "contacts_meas_source": payload.get("contacts_meas_source"),
        "contacts_meas_pretrain_clamp": payload.get("contacts_meas_pretrain_clamp"),
        "contacts_meas_pretrain_affine_stats_spec": payload.get("contacts_meas_pretrain_affine_stats_spec"),
        "analyze_phase_shift": payload.get("analyze_phase_shift", False),
        "mask": {
            "cycle_gte": int(args.group_cycle_gte),
            "drop_wrap": True,
        },
    }

    return {
        "name": cand.name,
        "selector": cand.selector,
        "family": cand.family,
        "ckpt": str(cand.ckpt),
        "eval_json": str(paths.eval_json),
        "basetrain_epoch": cand.basetrain_epoch,
        "discovery_tags": list(cand.discovery_tags),
        "schema": schema,
        "contract": contract,
        "metrics": metrics,
        "basetrain_metrics": basetrain_metrics,
    }


def _assign_rank(records: Sequence[Dict[str, Any]], metric_key: str) -> Dict[str, int]:
    pairs: List[Tuple[str, float]] = []
    for row in records:
        value = _safe_float(row["metrics"].get(metric_key))
        if math.isfinite(value):
            pairs.append((str(row["name"]), value))
    pairs.sort(key=lambda item: (item[1], item[0]))
    return {name: idx + 1 for idx, (name, _) in enumerate(pairs)}


def _evaluate_rows(rows: List[Dict[str, Any]], *, args: argparse.Namespace) -> Dict[str, Any]:
    baseline = None
    for row in rows:
        if row["selector"] == "best_free":
            baseline = row
            break
    if baseline is None and rows:
        baseline = rows[0]

    baseline_geo = _safe_float(baseline["metrics"].get("geo_local_deg_weighted")) if baseline else float("nan")
    baseline_drift = _safe_float(baseline["basetrain_metrics"].get("valfree_geo_drift_slope_proxy")) if baseline else float("nan")

    pass_rows: List[Dict[str, Any]] = []
    for row in rows:
        schema = row["schema"]
        contract = row["contract"]
        metrics = row["metrics"]
        base_metrics = row["basetrain_metrics"]

        payload_view = {
            "contacts_meas_source": contract.get("contacts_meas_source"),
            "rounds": contract.get("rounds"),
            "time_index_mode": contract.get("time_index_mode"),
            "phase_reset_source": contract.get("phase_reset_source"),
            "event_clock": contract.get("event_clock"),
            "contacts_meas_pretrain_clamp": contract.get("contacts_meas_pretrain_clamp"),
            "contacts_meas_pretrain_affine_stats_spec": contract.get("contacts_meas_pretrain_affine_stats_spec"),
        }
        contract_checks = payload_contract_ok(payload_view, root=ROOT)
        contract_checks["analyze_phase_shift_ok"] = bool(contract.get("analyze_phase_shift")) is True
        mask = contract.get("mask", {})
        contract_checks["mask_cycle_gte_ok"] = int(mask.get("cycle_gte", -1) or -1) == int(args.group_cycle_gte)
        contract_checks["mask_drop_wrap_ok"] = bool(mask.get("drop_wrap")) is True
        schema_ok = (
            bool(schema.get("has_metrics_per_step"))
            and bool(schema.get("has_metrics_per_round"))
            and bool(schema.get("has_per_step_direct_geolocal_deg"))
            and bool(schema.get("has_contact_err_abs_mean"))
            and bool(schema.get("has_contact_plan_gt_abs_mean"))
            and bool(schema.get("has_phase_shift"))
        )
        contract_ok = all(bool(v) for v in contract_checks.values())

        geo_local = _safe_float(metrics.get("geo_local_deg_weighted"))
        geo_rel_delta_pct = (
            100.0 * (geo_local - baseline_geo) / baseline_geo
            if math.isfinite(geo_local) and math.isfinite(baseline_geo) and baseline_geo != 0.0
            else float("nan")
        )
        geo_guardrail_ok = (not math.isfinite(geo_rel_delta_pct)) or (geo_rel_delta_pct <= float(args.geo_guardrail_pct))

        drift_proxy = _safe_float(base_metrics.get("valfree_geo_drift_slope_proxy"))
        drift_rel_delta_pct = (
            100.0 * (drift_proxy - baseline_drift) / baseline_drift
            if math.isfinite(drift_proxy) and math.isfinite(baseline_drift) and baseline_drift != 0.0
            else float("nan")
        )
        drift_guardrail_ok = (not math.isfinite(drift_rel_delta_pct)) or (drift_rel_delta_pct <= float(args.drift_guardrail_pct))

        guardrail = {
            "schema_ok": bool(schema_ok),
            "contract_ok": bool(contract_ok),
            "geo_local_rel_delta_pct": geo_rel_delta_pct,
            "geo_local_guardrail_ok": bool(geo_guardrail_ok),
            "basetrain_drift_rel_delta_pct": drift_rel_delta_pct,
            "basetrain_drift_guardrail_ok": bool(drift_guardrail_ok),
        }
        guardrail["pass"] = bool(schema_ok and contract_ok and geo_guardrail_ok and drift_guardrail_ok)
        reasons: List[str] = []
        if not schema_ok:
            reasons.append("schema_incomplete")
        if not contract_ok:
            reasons.append("contract_mismatch")
        if not geo_guardrail_ok:
            reasons.append("geo_local_guardrail")
        if not drift_guardrail_ok:
            reasons.append("basetrain_drift_guardrail")
        guardrail["fail_reasons"] = reasons
        row["guardrail"] = guardrail
        if guardrail["pass"]:
            pass_rows.append(row)

    if not pass_rows:
        pass_rows = list(rows)

    score_weights = {
        "contact_err_abs_mean": 0.25,
        "contact_plan_gt_abs_mean": 0.20,
        "phase_shift_contact_plan_abs_mean": 0.20,
        "leg8_mean": 0.15,
        "sic12_15_foot_l_ball_l_mean": 0.10,
        "calf_r_global_mean": 0.10,
    }
    rank_tables = {key: _assign_rank(pass_rows, key) for key in score_weights}
    for row in rows:
        rank_terms: Dict[str, Any] = {}
        score = 0.0
        score_valid = True
        for key, weight in score_weights.items():
            rank = rank_tables.get(key, {}).get(str(row["name"]))
            rank_terms[key] = {"rank": rank, "weight": weight, "value": row["metrics"].get(key)}
            if rank is None:
                score_valid = False
            else:
                score += float(weight) * float(rank)
        row["score"] = {
            "version": "v1",
            "formula": "0.25*rank(contact_err_abs_mean)+0.20*rank(contact_plan_gt_abs_mean)+0.20*rank(phase_shift_contact_plan_abs_mean)+0.15*rank(leg8_mean)+0.10*rank(sic12_15_foot_l_ball_l_mean)+0.10*rank(calf_r_global_mean)",
            "rank_terms": rank_terms,
            "value": float(score) if score_valid else float("nan"),
            "score_valid": bool(score_valid),
            "note": "v1 formally includes phase_shift_contact_plan_abs_mean as a weighted rank term.",
        }

    selected_pool = [row for row in rows if row["guardrail"]["pass"] and row["score"]["score_valid"]]
    if not selected_pool:
        selected_pool = [row for row in rows if row["score"]["score_valid"]]
    if not selected_pool:
        selected_pool = list(rows)

    selected_pool.sort(
        key=lambda row: (
            0 if row["guardrail"]["pass"] else 1,
            _safe_float(row["score"].get("value")),
            _safe_float(row["metrics"].get("contact_err_abs_mean")),
            _safe_float(row["metrics"].get("leg8_mean")),
            str(row["name"]),
        )
    )
    selected = selected_pool[0] if selected_pool else None
    return {
        "baseline_name": baseline["name"] if baseline else None,
        "baseline_geo_local_deg_weighted": baseline_geo,
        "selected_name": selected["name"] if selected else None,
        "selected_row": selected,
    }


def _render_md(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Basetrain Handoff Selector")
    lines.append("")
    lines.append(f"- run_tag: `{summary['run_tag']}`")
    lines.append(f"- exp_dir: `{summary.get('exp_dir', '')}`")
    boundary = summary.get("locked_boundary", {})
    eval_contract = boundary.get("eval_contract", {})
    mask = eval_contract.get("mask", {})
    if eval_contract:
        lines.append(
            "- locked_boundary: "
            f"`contacts_meas_source={eval_contract.get('contacts_meas_source')}`, "
            f"`affine=affine_mix08 canonical stats`, "
            f"`event_clock={eval_contract.get('event_clock')}`, "
            f"`rounds={eval_contract.get('rounds')}`, "
            f"`depth={eval_contract.get('depth')}`, "
            f"`time_index_mode={eval_contract.get('time_index_mode')}`, "
            f"`phase_reset_source={eval_contract.get('phase_reset_source')}`, "
            f"`cycle>={mask.get('cycle_gte')}`, "
            f"`drop_wrap={mask.get('drop_wrap')}`"
        )
    pool = summary.get("candidate_pool_policy", {})
    required_selectors = pool.get("required_saved_selectors_per_run", [])
    if required_selectors:
        lines.append(
            f"- candidate_pool_policy: `{','.join(required_selectors)} + {pool.get('include_ckpt_epoch_glob', 'ckpt_epoch_*.pth')}`"
        )
    lines.append(f"- selected_candidate: `{summary['selected_candidate']['name']}`")
    lines.append(f"- baseline_candidate: `{summary['baseline']['name']}`")
    if summary.get("best_handoff_ckpt"):
        lines.append(f"- best_handoff_ckpt: `{summary['best_handoff_ckpt']}`")
    if summary.get("exact_epoch_candidates"):
        lines.append(f"- exact_epoch_candidates: `{', '.join(summary['exact_epoch_candidates'])}`")
    lines.append("")
    lines.append("## Candidates")
    lines.append("")
    lines.append("| name | selector | epoch | tags | pass | handoff_score_v1 | ContactErrAbsMean | ContactPlanGtAbsMean | phase_shift_abs_mean | leg8_mean | SIC12-15 foot_l/ball_l | calf_r_global | GeoLocalDegWeighted |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["candidates"]:
        metrics = row["metrics"]
        guardrail = row["guardrail"]
        score = row["score"]
        tags = ",".join(row.get("discovery_tags", [])) or "-"
        lines.append(
            f"| {row['name']} | {row['selector']} | {row.get('basetrain_epoch') or '-'} | "
            f"{tags} | {'Y' if guardrail.get('pass') else 'N'} | {_fmt(score.get('value'))} | "
            f"{_fmt(metrics.get('contact_err_abs_mean'))} | {_fmt(metrics.get('contact_plan_gt_abs_mean'))} | "
            f"{_fmt(metrics.get('phase_shift_contact_plan_abs_mean'))} | {_fmt(metrics.get('leg8_mean'))} | "
            f"{_fmt(metrics.get('sic12_15_foot_l_ball_l_mean'))} | {_fmt(metrics.get('calf_r_global_mean'))} | "
            f"{_fmt(metrics.get('geo_local_deg_weighted'))} |"
        )
    proxy_rows = summary.get("proxy_epoch_candidates", [])
    if proxy_rows:
        lines.append("")
        lines.append("## Proxy Epoch Scan")
        lines.append("")
        lines.append("| rank | epoch | GeoDegSlope | GeoDriftSlopeProxy | freerun_leg | teacher_GeoLocal |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for item in proxy_rows:
            row = item["row"]
            lines.append(
                f"| {item['rank']} | {row['epoch']} | {_fmt(row['freerun']['geo_deg_slope'])} | "
                f"{_fmt(row['freerun']['geo_local_proxy'])} | {_fmt(row['freerun']['leg_mean'])} | "
                f"{_fmt(row['teacher']['geo_local_deg'])} |"
            )
    notes = summary.get("notes", [])
    if notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _parse_candidate(spec: str) -> Candidate:
    parts = [x.strip() for x in str(spec).split("|")]
    if not parts or "=" not in parts[0]:
        raise SystemExit(f"[FATAL] invalid --candidate spec: {spec!r}; expected name=ckpt or name=ckpt|family|selector|epoch")
    name, ckpt = [x.strip() for x in parts[0].split("=", 1)]
    family = parts[1] if len(parts) >= 2 else ""
    selector = parts[2] if len(parts) >= 3 else name
    epoch = None
    if len(parts) >= 4 and parts[3] != "":
        try:
            epoch = int(parts[3])
        except Exception:
            epoch = None
    return Candidate(
        name=name,
        ckpt=_resolve(ckpt),
        family=family,
        selector=selector,
        basetrain_epoch=epoch,
        discovery_tags=("manual",),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run basetrain handoff selector smoke/evaluator/sweep for saved selector checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--exp-dir", default="", help="Basetrain experiment directory; auto-discovers best_free/last/best_teacher when --candidate is omitted.")
    ap.add_argument("--candidate", action="append", default=[], help="Repeatable spec: name=ckpt|family|selector|epoch")
    ap.add_argument("--run-tag", default="manual")
    ap.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    ap.add_argument("--encoder-bundle", default=str(DEFAULT_ENCODER_BUNDLE))
    ap.add_argument("--affine-stats", default=str(DEFAULT_AFFINE_STATS))
    ap.add_argument("--pretrain-clamp", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--time-index-mode", default="cycle")
    ap.add_argument("--event-clock", default="auto", choices=("auto", "on", "off"))
    ap.add_argument("--phase-reset-source", default="none")
    ap.add_argument("--group-cycle-gte", type=int, default=1)
    ap.add_argument("--geo-guardrail-pct", type=float, default=5.0)
    ap.add_argument("--drift-guardrail-pct", type=float, default=25.0)
    ap.add_argument("--proxy-epoch-start", type=int, default=6)
    ap.add_argument("--proxy-epoch-end", type=int, default=18)
    ap.add_argument("--proxy-topk", type=int, default=5)
    ap.add_argument("--exact-epoch-topk", type=int, default=5, help="Use top-k proxy epochs as exact candidates when ckpt_epoch_*.pth exists.")
    ap.add_argument("--exact-epoch-window", type=int, default=2, help="Also include best_free_epoch +/- window as exact candidates when ckpt_epoch_*.pth exists.")
    ap.add_argument("--emit-best-handoff-ckpt", action="store_true", help="Emit ckpt_best_handoff_<run_name>.pth for the selected candidate.")
    ap.add_argument("--best-handoff-link-dir", default="", help="Directory where ckpt_best_handoff_<run_name>.pth is emitted.")
    ap.add_argument("--best-handoff-run-name", default="", help="Run name used in ckpt_best_handoff_<run_name>.pth.")
    ap.add_argument("--smoke-only", action="store_true", help="Only evaluate best_free when using --exp-dir auto-discovery.")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--allow-noncanonical-boundary", action="store_true", help="Allow off-contract diagnostics instead of enforcing the locked Stage6 anchor boundary.")
    args = ap.parse_args()

    args.teacher = _resolve(args.teacher)
    args.encoder_bundle = _resolve(args.encoder_bundle)
    args.affine_stats = _resolve(args.affine_stats)
    boundary = apply_locked_boundary(args, root=ROOT)

    required = [args.teacher, args.encoder_bundle, args.affine_stats]
    missing = [str(path) for path in required if not Path(path).is_file()]

    exp_dir = _resolve(args.exp_dir) if str(args.exp_dir).strip() else None
    candidates = [_parse_candidate(spec) for spec in args.candidate]
    proxy_scan_topk = max(int(args.proxy_topk), int(args.exact_epoch_topk))
    proxy_epoch_candidates = (
        _build_proxy_epoch_rows(exp_dir, epoch_start=int(args.proxy_epoch_start), epoch_end=int(args.proxy_epoch_end), topk=proxy_scan_topk)
        if exp_dir is not None
        else []
    )
    if not candidates:
        if exp_dir is None:
            raise SystemExit("[FATAL] provide either --exp-dir or at least one --candidate")
        candidates = _discover_candidates(exp_dir, smoke_only=bool(args.smoke_only))
        if not bool(args.smoke_only):
            candidates.extend(
                _discover_exact_epoch_candidates(
                    exp_dir,
                    base_candidates=candidates,
                    proxy_epoch_rows=proxy_epoch_candidates,
                    exact_topk=int(args.exact_epoch_topk),
                    exact_window=int(args.exact_epoch_window),
                )
            )
    missing.extend(str(c.ckpt) for c in candidates if not c.ckpt.is_file())
    if exp_dir is not None and not exp_dir.is_dir():
        missing.append(str(exp_dir))
    if missing:
        raise SystemExit("[FATAL] missing required files:\n" + "\n".join(missing))

    out_root = _resolve(args.out_root) if str(args.out_root).strip() else (ROOT / "debug_output" / f"_tmp_basetrain_handoff_selector_{args.run_tag}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates, start=1):
        print(f"[candidate {idx}/{len(candidates)}] {cand.name} -> {cand.ckpt}", flush=True)
        paths = _build_paths(out_root=out_root, name=cand.name)
        paths.lane_root.mkdir(parents=True, exist_ok=True)
        _run_candidate_eval(args, cand, paths)
        rows.append(_build_candidate_row(cand=cand, paths=paths, args=args))

    eval_summary = _evaluate_rows(rows, args=args)
    notes: List[str] = []
    if exp_dir is not None:
        has_epoch_ckpts = any(exp_dir.glob("ckpt_epoch_*.pth"))
        if not has_epoch_ckpts:
            notes.append("No ckpt_epoch_*.pth checkpoints were found, so Step 2 epoch sweep is currently proxy-only; exact epoch replay still needs per-epoch ckpt saving.")
        else:
            notes.append("ckpt_epoch_*.pth checkpoints were found, so proxy-ranked epochs are auto-promoted into exact epoch candidates when not already covered by best_free/last/best_teacher.")
    notes.append("Step 0 smoke is represented by schema/contract checks on the best_free eval JSON.")
    notes.append("Evaluator v1 formally includes phase_shift_contact_plan_abs_mean in the weighted handoff score.")

    selected = eval_summary["selected_row"]
    best_handoff_ckpt = None
    if bool(args.emit_best_handoff_ckpt):
        best_handoff_ckpt = _emit_best_handoff_ckpt(
            selected_row=selected,
            out_dir=_resolve(args.best_handoff_link_dir) if str(args.best_handoff_link_dir).strip() else out_root,
            run_name=str(args.best_handoff_run_name or (exp_dir.name if exp_dir is not None else args.run_tag)),
        )
    summary = {
        "run_tag": str(args.run_tag),
        "exp_dir": str(exp_dir) if exp_dir is not None else "",
        "contract": {
            "teacher": str(args.teacher),
            "encoder_bundle": str(args.encoder_bundle),
            "affine_stats": str(args.affine_stats),
            "pretrain_clamp": float(args.pretrain_clamp),
            "rounds": int(args.rounds),
            "depth": int(args.depth),
            "time_index_mode": str(args.time_index_mode),
            "event_clock": str(args.event_clock),
            "phase_reset_source": str(args.phase_reset_source),
            "group_cycle_gte": int(args.group_cycle_gte),
        },
        "locked_boundary": boundary,
        "candidate_pool_policy": candidate_pool_policy(),
        "baseline": {
            "name": eval_summary["baseline_name"],
            "geo_local_deg_weighted": eval_summary["baseline_geo_local_deg_weighted"],
        },
        "selected_candidate": {
            "name": eval_summary["selected_name"],
            "score": selected["score"]["value"] if selected else None,
            "guardrail_pass": bool(selected["guardrail"]["pass"]) if selected else False,
            "ckpt": selected.get("ckpt") if selected else None,
        },
        "best_handoff_ckpt": best_handoff_ckpt,
        "candidates": rows,
        "exact_epoch_candidates": [row["name"] for row in rows if row.get("selector") == "epoch_exact"],
        "proxy_epoch_candidates": proxy_epoch_candidates[: int(args.proxy_topk)],
        "notes": notes,
    }

    _write_json(out_root / "handoff_selector_candidates.json", {"candidates": rows})
    _write_json(out_root / "handoff_selector_summary.json", summary)
    (out_root / "handoff_selector_summary.md").write_text(_render_md(summary), encoding="utf-8")
    print(f"[done] {out_root / 'handoff_selector_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
