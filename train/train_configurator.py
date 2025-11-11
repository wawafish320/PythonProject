#!/usr/bin/env python3
"""
Unified training configuration orchestrator.

Features:
- Dataset profiling + schedule synthesis (was tools/strategy_generator.py).
- Stage metric feedback + auto nudging (was tools/stage_auto_tuner.py).
- Optional Bayesian optimizer that consumes past runs and proposes new hyper-parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# ===== shared templates =====

STAGE_TEMPLATE: List[Dict[str, Any]] = [
    {
        "name": "stage1_teacher",
        "ratio": 0.3,
        "motion": {
            "lookahead_weight_scale": 0.2,
            "freerun_weight_scale": 0.0,
            "freerun_horizon_scale": 0.6,
            "latent_scale": 0.1,
        },
        "posture": {"scale": 0.2},
        "tf_max": 1.0,
        "yaw_ratio": (0.75, 1.25),
        "root_ratio": (0.02, 0.04),
        "rot_ratio": (0.9, 1.4),
    },
    {
        "name": "stage2_mixed",
        "ratio": 0.4,
        "motion": {
            "lookahead_weight_scale": 0.6,
            "freerun_weight_scale": 0.35,
            "freerun_horizon_scale": 0.9,
            "latent_scale": 0.4,
        },
        "posture": {"scale": 0.65},
        "tf_max": 0.75,
        "yaw_ratio": (0.65, 1.1),
        "root_ratio": (0.018, 0.035),
        "rot_ratio": (0.75, 1.1),
    },
    {
        "name": "stage3_freerun",
        "ratio": 0.3,
        "motion": {
            "lookahead_weight_scale": 0.9,
            "freerun_weight_scale": 0.65,
            "freerun_horizon_scale": 1.2,
            "latent_scale": 0.8,
        },
        "posture": {"scale": 1.0},
        "tf_max": 0.5,
        "yaw_ratio": (0.55, 0.95),
        "root_ratio": (0.015, 0.03),
        "rot_ratio": (0.65, 1.0),
    },
]

DEFAULT_TARGETS = {
    "yaw": {"lo_ratio": 0.75, "hi_ratio": 1.1, "ref": 18.0},
    "root": {"lo_ratio": 0.02, "hi_ratio": 0.035, "ref": 0.75},
    "rot": {"lo_ratio": 0.7, "hi_ratio": 1.1, "ref": 2.0},
}

DEFAULT_METRIC_WEIGHTS = {
    "YawAbsDeg": 0.4,
    "RootVelMAE": 0.35,
    "InputRotGeoDeg": 0.25,
}


# ===== IO helpers =====

def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ===== dataset profiling =====

def _rot6d_to_angle(vec: Sequence[float]) -> float:
    a = vec[:3]
    b = vec[3:6]

    def _norm(v: Sequence[float]) -> List[float]:
        n = math.sqrt(sum(x * x for x in v))
        if n < 1e-8:
            return list(v)
        return [x / n for x in v]

    X = _norm(a)
    proj = sum(X[i] * b[i] for i in range(3))
    b_ortho = [b[i] - proj * X[i] for i in range(3)]
    Z = _norm(b_ortho)
    Y = _norm(
        [
            Z[1] * X[2] - Z[2] * X[1],
            Z[2] * X[0] - Z[0] * X[2],
            Z[0] * X[1] - Z[1] * X[0],
        ]
    )
    trace = X[0] + Y[1] + Z[2]
    val = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return math.degrees(math.acos(val))


class DatasetProfiler:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def profile(self) -> Dict[str, Any]:
        files = sorted(self.raw_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON clips under {self.raw_dir}")
        seq_lengths: List[int] = []
        yaw_vals: List[float] = []
        speed_vals: List[float] = []
        bone_angles: List[float] = []

        for path in files:
            data = json.loads(path.read_text())
            frames = data.get("Frames") or []
            seq_lengths.append(len(frames))
            for fr in frames:
                yaw_vals.append(abs(fr.get("RootYaw", 0.0)) * (180.0 / math.pi))
                rv = fr.get("RootVelocityXY") or [0.0, 0.0]
                speed_vals.append(math.hypot(rv[0], rv[1]))
                rotations = fr.get("BoneRotations")
                if rotations:
                    for rot in rotations:
                        try:
                            bone_angles.append(_rot6d_to_angle(rot))
                        except Exception:
                            continue

        total_frames = sum(seq_lengths)
        yaw_mean = stats.mean(yaw_vals)
        yaw_std = stats.pstdev(yaw_vals) if len(yaw_vals) > 1 else 0.0
        speed_mean = stats.mean(speed_vals)
        speed_std = stats.pstdev(speed_vals) if len(speed_vals) > 1 else 0.0
        bone_mean = stats.mean(bone_angles) if bone_angles else 45.0
        bone_std = stats.pstdev(bone_angles) if len(bone_angles) > 1 else 0.0
        complexity = min(2.0, 0.5 * (yaw_std / 30.0 + speed_std / 0.3))

        return {
            "n_clips": len(files),
            "total_frames": total_frames,
            "avg_seq_len": stats.mean(seq_lengths) if seq_lengths else 60.0,
            "median_seq_len": stats.median(seq_lengths) if seq_lengths else 60.0,
            "yaw_mean_deg": yaw_mean,
            "yaw_std_deg": yaw_std,
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "bone_angle_mean_deg": bone_mean,
            "bone_angle_std_deg": bone_std,
            "complexity": complexity,
        }


# ===== schedule synthesis =====

def compute_total_epochs(total_frames: int) -> int:
    if total_frames < 2000:
        return 60
    if total_frames < 10000:
        return 45
    if total_frames < 50000:
        return 30
    return 20


def compute_batch_size(avg_seq_len: float) -> int:
    if avg_seq_len < 50:
        return 16
    if avg_seq_len < 90:
        return 8
    if avg_seq_len < 160:
        return 6
    return 4


def compute_base_lr(total_frames: int, complexity: float, batch_size: int) -> float:
    scale = max(1.5, math.log10(max(total_frames, 10))) / 5.0
    comp = 1.0 / (1.0 + complexity)
    lr = 1e-3 * scale * comp * math.sqrt(batch_size / 8.0)
    return max(1e-5, min(6e-4, lr))


class TrainingConfigBuilder:
    def __init__(self, base_cfg: Optional[Mapping[str, Any]] = None):
        self.base_cfg = dict(base_cfg or {})

    def build(self, profile: Mapping[str, Any]) -> Dict[str, Any]:
        total_epochs = self.base_cfg.get("epochs") or compute_total_epochs(int(profile["total_frames"]))
        batch_size = self.base_cfg.get("batch") or compute_batch_size(float(profile["avg_seq_len"]))
        lr = self.base_cfg.get("lr") or compute_base_lr(int(profile["total_frames"]), float(profile["complexity"]), batch_size)

        stages, refs = self._build_stage_schedule(profile, total_epochs)
        cfg = dict(self.base_cfg)
        cfg["dataset_profile"] = dict(profile)
        cfg["epochs"] = int(total_epochs)
        cfg["batch"] = int(batch_size)
        cfg["lr"] = float(round(lr, 6))
        cfg["lookahead_stage_schedule"] = stages
        cfg["freerun_weight"] = stages[0]["trainer"]["freerun_weight"]
        cfg["w_latent_consistency"] = stages[0]["trainer"]["w_latent_consistency"]
        cfg["w_fk_pos"] = stages[0]["loss"]["w_fk_pos"]
        cfg["w_rot_local"] = stages[0]["loss"]["w_rot_local"]
        cfg.setdefault("tf_mode", "epoch_linear")
        cfg["tf_start_epoch"] = 1
        cfg["tf_end_epoch"] = max(2, int(total_epochs * 0.65))
        cfg["tf_max"] = stages[0]["tf"]["max"]
        cfg["tf_min"] = 0.0
        cfg.setdefault("seq_len", int(profile["avg_seq_len"]))
        cfg.setdefault("freerun_horizon", stages[0]["trainer"]["freerun_horizon"])
        cfg.setdefault("freerun_horizon_ramp_epochs", max(4, int(total_epochs * 0.15)))
        cfg.setdefault("strategy_meta", {})["reference_targets"] = refs
        return cfg

    def _build_stage_schedule(self, profile: Mapping[str, Any], total_epochs: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        avg_seq_len = float(profile["avg_seq_len"])
        base_lookahead = max(2, int(round(avg_seq_len * 0.12)))
        base_horizon = max(6, int(round(avg_seq_len * 0.2)))
        posture_ref = max(1.2, float(profile["bone_angle_mean_deg"]) * 0.04)
        dataset_refs = {
            "yaw": float(profile["yaw_mean_deg"]),
            "root": float(profile["speed_mean"]),
            "rot": posture_ref,
        }

        stages: List[Dict[str, Any]] = []
        cursor = 1
        for template in STAGE_TEMPLATE:
            length = max(1, round(total_epochs * template["ratio"]))
            start = cursor
            end = min(total_epochs, cursor + length - 1)
            cursor = end + 1
            motion = template["motion"]
            posture = template["posture"]

            lookahead_steps = max(1, int(round(base_lookahead * motion["lookahead_weight_scale"] * 1.2)))
            freerun_horizon = max(4, int(round(base_horizon * motion["freerun_horizon_scale"])))
            lookahead_weight = min(0.8, max(0.0, motion["lookahead_weight_scale"] * 0.35))
            freerun_weight = min(0.8, max(0.0, motion["freerun_weight_scale"] * 0.5))
            latent = min(0.6, max(0.0, motion["latent_scale"] * 0.3))
            posture_weight = min(0.8, max(0.01, posture["scale"] * 0.35))

            yaw_lo_ratio, yaw_hi_ratio = template["yaw_ratio"]
            root_lo_ratio, root_hi_ratio = template["root_ratio"]
            rot_lo_ratio, rot_hi_ratio = template["rot_ratio"]

            stages.append(
                {
                    "range": [start, end],
                    "label": template["name"],
                    "trainer": {
                        "lookahead_steps": lookahead_steps,
                        "lookahead_weight": round(lookahead_weight, 4),
                        "freerun_weight": round(freerun_weight, 4),
                        "freerun_horizon": freerun_horizon,
                        "w_latent_consistency": round(latent, 4),
                    },
                    "loss": {
                        "w_fk_pos": round(posture_weight, 4),
                        "w_rot_local": round(posture_weight, 4),
                    },
                    "tf": {"max": template["tf_max"]},
                    "targets": {
                        "yaw": {"ref": dataset_refs["yaw"], "lo_ratio": yaw_lo_ratio, "hi_ratio": yaw_hi_ratio},
                        "root": {"ref": dataset_refs["root"], "lo_ratio": root_lo_ratio, "hi_ratio": root_hi_ratio},
                        "rot": {"ref": dataset_refs["rot"], "lo_ratio": rot_lo_ratio, "hi_ratio": rot_hi_ratio},
                    },
                }
            )

        stages[-1]["range"][1] = total_epochs
        return stages, dataset_refs


# ===== metrics ingestion / stage tuning =====

Metric = Dict[str, Optional[float]]


def load_val_metrics(metrics_dir: Path) -> Dict[int, Metric]:
    epochs: Dict[int, Metric] = {}
    if not metrics_dir or not metrics_dir.exists():
        raise FileNotFoundError(f"metrics dir not found: {metrics_dir}")
    for path in sorted(metrics_dir.glob("valfree_ep*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")
            continue
        epoch = int(data.get("epoch") or 0)
        payload = data.get("metrics") or {}
        epochs[epoch] = {
            "YawAbsDeg": _maybe_float(payload.get("YawAbsDeg")),
            "RootVelMAE": _maybe_float(payload.get("RootVelMAE")),
            "InputRotGeoDeg": _maybe_float(payload.get("InputRotGeoDeg")),
        }
    return epochs


def _maybe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_stage_metrics(schedule: Sequence[Mapping[str, Any]], val_metrics: Mapping[int, Metric]) -> Dict[str, Metric]:
    summary: Dict[str, Metric] = {}
    for stage in schedule:
        rng = stage.get("range") or [stage.get("start"), stage.get("end")]
        if not rng:
            continue
        start = int(rng[0])
        end = int(rng[-1] if len(rng) > 1 else rng[0])
        samples = [val_metrics[e] for e in range(start, end + 1) if e in val_metrics]
        if not samples:
            continue
        agg: Dict[str, float] = {}
        for key in ("YawAbsDeg", "RootVelMAE", "InputRotGeoDeg"):
            values = [m[key] for m in samples if m.get(key) is not None]
            if values:
                agg[key] = sum(values) / len(values)
        summary[stage.get("label", f"{start}-{end}")] = agg
    return summary


def _resolve_bounds(stage: Mapping[str, Any], key: str, ref_values: Mapping[str, float]) -> Tuple[float, float]:
    stage_targets = stage.get("targets") or {}
    spec = stage_targets.get(key) or {}
    defaults = DEFAULT_TARGETS[key]
    ref = spec.get("ref") or ref_values.get(key) or defaults["ref"]

    lo_ratio = spec.get("lo_ratio", defaults["lo_ratio"])
    hi_ratio = spec.get("hi_ratio", defaults["hi_ratio"])
    lo = spec.get("lo")
    hi = spec.get("hi")
    if lo is None and lo_ratio is not None:
        lo = ref * float(lo_ratio)
    if hi is None and hi_ratio is not None:
        hi = ref * float(hi_ratio)
    if lo is None:
        lo = defaults["ref"] * defaults["lo_ratio"]
    if hi is None:
        hi = defaults["ref"] * defaults["hi_ratio"]
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _scale(value: float, factor: float, lo: float, hi: float) -> float:
    return _clamp(value * factor, lo, hi)


class StageMetricAdjuster:
    def __init__(self, config: MutableMapping[str, Any], ref_values: Optional[Mapping[str, float]] = None):
        self.config = config
        self.ref_values = ref_values or (config.get("strategy_meta") or {}).get("reference_targets") or {}

    def apply(self, val_metrics: Mapping[int, Metric]) -> Dict[str, float]:
        schedule = self.config.get("lookahead_stage_schedule")
        if not schedule:
            raise ValueError("config missing lookahead_stage_schedule")
        summary = summarize_stage_metrics(schedule, val_metrics)
        changed: Dict[str, float] = {}
        for stage in schedule:
            label = stage.get("label")
            metrics = summary.get(label)
            if not metrics:
                continue
            updates = self._adjust_stage(stage, metrics)
            changed.update({f"{label}.{k}": v for k, v in updates.items()})
        return changed

    def _adjust_stage(self, stage: MutableMapping[str, Any], metrics: Mapping[str, float]) -> Dict[str, float]:
        trainer = stage.setdefault("trainer", {})
        loss_cfg = stage.setdefault("loss", {})
        changed: Dict[str, float] = {}

        yaw = metrics.get("YawAbsDeg")
        root_mae = metrics.get("RootVelMAE")
        rot_geo = metrics.get("InputRotGeoDeg")

        yaw_lo, yaw_hi = _resolve_bounds(stage, "yaw", self.ref_values)
        root_lo, root_hi = _resolve_bounds(stage, "root", self.ref_values)
        rot_lo, rot_hi = _resolve_bounds(stage, "rot", self.ref_values)

        lookahead_w = float(trainer.get("lookahead_weight", 0.0) or 0.0)
        freerun_w = float(trainer.get("freerun_weight", 0.0) or 0.0)
        freerun_h = int(trainer.get("freerun_horizon", 0) or 0)
        latent_w = float(trainer.get("w_latent_consistency", 0.0) or 0.0)
        w_fk = float(loss_cfg.get("w_fk_pos", 0.0) or 0.0)
        w_rot_local = float(loss_cfg.get("w_rot_local", 0.0) or 0.0)

        if yaw is not None:
            if yaw > yaw_hi:
                new = _scale(lookahead_w, 1.15, 0.0, 0.8)
                if new != lookahead_w:
                    trainer["lookahead_weight"] = round(new, 4)
                    changed["lookahead_weight"] = trainer["lookahead_weight"]
                    lookahead_w = new
                new_h = _clamp(int(round(freerun_h * 1.1 or 1)), 4, 64)
                if new_h != freerun_h:
                    trainer["freerun_horizon"] = new_h
                    changed["freerun_horizon"] = trainer["freerun_horizon"]
                    freerun_h = new_h
            elif yaw < yaw_lo:
                new = _scale(lookahead_w, 0.9, 0.0, 0.8)
                if new != lookahead_w:
                    trainer["lookahead_weight"] = round(new, 4)
                    changed["lookahead_weight"] = trainer["lookahead_weight"]
                    lookahead_w = new
                new_h = _clamp(int(round(freerun_h * 0.9 or 1)), 4, 64)
                if new_h != freerun_h:
                    trainer["freerun_horizon"] = new_h
                    changed["freerun_horizon"] = trainer["freerun_horizon"]
                    freerun_h = new_h

        if root_mae is not None:
            if root_mae > root_hi:
                new = _scale(freerun_w, 1.2, 0.0, 0.8)
                if new != freerun_w:
                    trainer["freerun_weight"] = round(new, 4)
                    changed["freerun_weight"] = trainer["freerun_weight"]
                    freerun_w = new
            elif root_mae < root_lo:
                new = _scale(freerun_w, 0.9, 0.0, 0.8)
                if new != freerun_w:
                    trainer["freerun_weight"] = round(new, 4)
                    changed["freerun_weight"] = trainer["freerun_weight"]
                    freerun_w = new

        if rot_geo is not None:
            if rot_geo > rot_hi:
                new = _scale(latent_w, 1.2, 0.0, 0.6)
                if new != latent_w:
                    trainer["w_latent_consistency"] = round(new, 4)
                    changed["w_latent_consistency"] = trainer["w_latent_consistency"]
                    latent_w = new
                new_fk = _scale(w_fk, 1.15, 0.01, 0.8)
                if new_fk != w_fk:
                    loss_cfg["w_fk_pos"] = round(new_fk, 4)
                    changed["loss.w_fk_pos"] = loss_cfg["w_fk_pos"]
                    w_fk = new_fk
                new_rot_local = _scale(w_rot_local, 1.15, 0.01, 0.8)
                if new_rot_local != w_rot_local:
                    loss_cfg["w_rot_local"] = round(new_rot_local, 4)
                    changed["loss.w_rot_local"] = loss_cfg["w_rot_local"]
                    w_rot_local = new_rot_local
            elif rot_geo < rot_lo:
                new = _scale(latent_w, 0.9, 0.0, 0.6)
                if new != latent_w:
                    trainer["w_latent_consistency"] = round(new, 4)
                    changed["w_latent_consistency"] = trainer["w_latent_consistency"]
                    latent_w = new
                new_fk = _scale(w_fk, 0.9, 0.01, 0.8)
                if new_fk != w_fk:
                    loss_cfg["w_fk_pos"] = round(new_fk, 4)
                    changed["loss.w_fk_pos"] = loss_cfg["w_fk_pos"]
                    w_fk = new_fk
                new_rot_local = _scale(w_rot_local, 0.9, 0.01, 0.8)
                if new_rot_local != w_rot_local:
                    loss_cfg["w_rot_local"] = round(new_rot_local, 4)
                    changed["loss.w_rot_local"] = loss_cfg["w_rot_local"]
                    w_rot_local = new_rot_local

        return changed


def aggregate_metrics(metrics_dir: Path) -> Dict[str, float]:
    epochs = load_val_metrics(metrics_dir)
    agg: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for metric in epochs.values():
        for key, value in metric.items():
            if value is None:
                continue
            agg[key] = agg.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {k: agg[k] / counts[k] for k in agg}


# ===== Bayesian optimization scaffold =====

@dataclass
class ParamSpec:
    key: str
    kind: str  # "float" or "int"
    bounds: Tuple[float, float]
    scale: str = "linear"  # or "log"

    def to_unit(self, value: float) -> float:
        lo, hi = self.bounds
        if self.scale == "log":
            lo, hi = math.log(lo, 10), math.log(hi, 10)
            value = math.log(value, 10)
        return (value - lo) / (hi - lo)

    def from_unit(self, value: float) -> float:
        lo, hi = self.bounds
        if self.scale == "log":
            lo, hi = math.log(lo, 10), math.log(hi, 10)
            val = lo + value * (hi - lo)
            return 10 ** val
        val = lo + value * (hi - lo)
        if self.kind == "int":
            return int(round(val))
        return val


DEFAULT_BAYES_SPECS: List[ParamSpec] = [
    ParamSpec("lr", "float", (1e-5, 6e-4), "log"),
    ParamSpec("stage2.trainer.lookahead_weight", "float", (0.05, 0.65)),
    ParamSpec("stage2.trainer.freerun_weight", "float", (0.0, 0.8)),
    ParamSpec("stage2.trainer.w_latent_consistency", "float", (0.01, 0.4)),
    ParamSpec("stage2.loss.w_fk_pos", "float", (0.05, 0.6)),
    ParamSpec("stage2.loss.w_rot_local", "float", (0.05, 0.6)),
    ParamSpec("stage3.trainer.lookahead_weight", "float", (0.05, 0.8)),
    ParamSpec("stage3.trainer.freerun_weight", "float", (0.05, 0.8)),
    ParamSpec("stage3.trainer.w_latent_consistency", "float", (0.05, 0.45)),
    ParamSpec("stage3.loss.w_fk_pos", "float", (0.1, 0.8)),
    ParamSpec("stage3.loss.w_rot_local", "float", (0.1, 0.8)),
]


class BayesHistory:
    def __init__(self, path: Path):
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.entries = data

    def add(self, params: Mapping[str, float], metrics: Mapping[str, float], score: float) -> None:
        self.entries.append({"params": dict(params), "metrics": dict(metrics), "score": float(score)})

    def save(self) -> None:
        dump_json(self.path, self.entries)


class BayesianOptimizer:
    """
    Lightweight TPE-style optimizer implemented without external deps.
    """

    def __init__(self, specs: Sequence[ParamSpec], seed: int = 0, quantile: float = 0.3):
        self.specs = list(specs)
        self.rng = random.Random(seed)
        self.quantile = min(max(quantile, 0.1), 0.5)

    def suggest(self, history: Sequence[Mapping[str, Any]], n_candidates: int = 64) -> Dict[str, float]:
        usable = [entry for entry in history if self._has_all_params(entry)]
        if len(usable) < 4:
            return self._random_candidate()
        usable = sorted(usable, key=lambda e: e.get("score", float("inf")))
        split = max(1, int(math.ceil(self.quantile * len(usable))))
        good = usable[:split]
        bad = usable[split:]
        good_stats = self._fit_gaussians(good)
        bad_stats = self._fit_gaussians(bad)

        best_cand = None
        best_ratio = -float("inf")
        for _ in range(n_candidates):
            cand = self._sample_from_good(good_stats)
            ratio = self._log_density_ratio(cand, good_stats, bad_stats)
            if ratio > best_ratio:
                best_ratio = ratio
                best_cand = cand
        return best_cand or self._random_candidate()

    def _has_all_params(self, entry: Mapping[str, Any]) -> bool:
        params = entry.get("params") or {}
        return all(spec.key in params for spec in self.specs)

    def _random_candidate(self) -> Dict[str, float]:
        cand: Dict[str, float] = {}
        for spec in self.specs:
            cand[spec.key] = self._sample_uniform(spec)
        return cand

    def _sample_uniform(self, spec: ParamSpec) -> float:
        lo, hi = spec.bounds
        if spec.scale == "log":
            lo_log, hi_log = math.log(lo, 10), math.log(hi, 10)
            val = 10 ** self.rng.uniform(lo_log, hi_log)
        else:
            val = self.rng.uniform(lo, hi)
        if spec.kind == "int":
            val = round(val)
        return float(val)

    def _fit_gaussians(self, entries: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[float, float]]:
        stats_map: Dict[str, Tuple[float, float]] = {}
        if not entries:
            return stats_map
        for spec in self.specs:
            values: List[float] = []
            for entry in entries:
                params = entry.get("params") or {}
                if spec.key in params:
                    values.append(params[spec.key])
            if not values:
                continue
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(max(variance, 1e-8))
            else:
                std = (spec.bounds[1] - spec.bounds[0]) * 0.1
            std = max(std, (spec.bounds[1] - spec.bounds[0]) * 0.05)
            stats_map[spec.key] = (mean, std)
        return stats_map

    def _sample_from_good(self, good_stats: Mapping[str, Tuple[float, float]]) -> Dict[str, float]:
        cand: Dict[str, float] = {}
        for spec in self.specs:
            mu, sigma = good_stats.get(spec.key, (None, None))
            if mu is None:
                cand[spec.key] = self._sample_uniform(spec)
                continue
            val = None
            for _ in range(8):
                draw = self.rng.gauss(mu, sigma)
                if spec.kind == "int":
                    draw = round(draw)
                if spec.bounds[0] <= draw <= spec.bounds[1]:
                    val = float(draw)
                    break
            if val is None:
                val = self._sample_uniform(spec)
            cand[spec.key] = val
        return cand

    def _log_density_ratio(
        self,
        cand: Mapping[str, float],
        good_stats: Mapping[str, Tuple[float, float]],
        bad_stats: Mapping[str, Tuple[float, float]],
    ) -> float:
        total = 0.0
        for spec in self.specs:
            value = cand.get(spec.key)
            if value is None:
                continue
            g = good_stats.get(spec.key)
            b = bad_stats.get(spec.key)
            if not g:
                continue
            log_g = self._log_gaussian(value, g[0], g[1])
            log_b = self._log_gaussian(value, b[0], b[1]) if b else 0.0
            total += log_g - log_b
        return total

    @staticmethod
    def _log_gaussian(x: float, mu: float, sigma: float) -> float:
        sigma = max(sigma, 1e-6)
        return -0.5 * math.log(2 * math.pi * sigma * sigma) - 0.5 * ((x - mu) ** 2) / (sigma * sigma)


def compute_history_score(metrics: Mapping[str, float], weights: Optional[Mapping[str, float]] = None) -> float:
    weights = weights or DEFAULT_METRIC_WEIGHTS
    num = 0.0
    denom = 0.0
    for key, w in weights.items():
        value = metrics.get(key)
        if value is None:
            continue
        num += w * value
        denom += w
    return num / max(denom, 1e-6)


def extract_param_vector(cfg: Mapping[str, Any], specs: Sequence[ParamSpec]) -> Dict[str, float]:
    schedule = {stage.get("label"): stage for stage in (cfg.get("lookahead_stage_schedule") or [])}
    values: Dict[str, float] = {}
    for spec in specs:
        if spec.key == "lr":
            if cfg.get("lr") is not None:
                values["lr"] = float(cfg["lr"])
            continue
        tokens = spec.key.split(".", 1)
        if len(tokens) != 2:
            continue
        stage_label, rest = tokens
        stage = schedule.get(stage_label)
        if not stage:
            continue
        value = _get_nested(stage, rest.split("."))
        if value is not None:
            values[spec.key] = float(value)
    return values


def apply_param_updates(cfg: MutableMapping[str, Any], updates: Mapping[str, float]) -> None:
    if not updates:
        return
    schedule = cfg.get("lookahead_stage_schedule") or []
    stage_map = {stage.get("label"): stage for stage in schedule}
    for key, value in updates.items():
        if key == "lr":
            cfg["lr"] = float(value)
            continue
        tokens = key.split(".", 1)
        if len(tokens) != 2:
            continue
        stage_label, rest = tokens
        stage = stage_map.get(stage_label)
        if not stage:
            continue
        _set_nested(stage, rest.split("."), float(value))


def _get_nested(obj: Mapping[str, Any], path: Sequence[str]) -> Optional[float]:
    cursor: Any = obj
    for key in path:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(key)
    if cursor is None:
        return None
    return float(cursor)


def _set_nested(obj: MutableMapping[str, Any], path: Sequence[str], value: float) -> None:
    cursor: MutableMapping[str, Any] = obj
    for key in path[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[path[-1]] = float(value)


# ===== CLI =====

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unified training config manager")
    ap.add_argument("--data-root", default="raw_data", help="Directory containing raw JSON clips")
    ap.add_argument("--metrics-root", default=None, help="Directory containing valfree metrics JSON files")
    ap.add_argument("--base-config", default="config/exp_phase_mpl.json", help="Existing config to load as baseline")
    ap.add_argument("--output", default="config/exp_phase_mpl.json", help="Where to write the updated config")
    ap.add_argument("--history", default="train/bayes_history.json", help="Bayesian optimization history file")
    ap.add_argument("--profile", action="store_true", help="Recompute dataset profile before building config")
    ap.add_argument("--tune", action="store_true", help="Apply metric-driven stage adjustments")
    ap.add_argument("--bayes-opt", action="store_true", help="Enable Bayesian optimizer suggestions")
    ap.add_argument("--bayes-update-history", action="store_true", help="Append current run (config+metrics) to history")
    ap.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for candidate sampling")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    base_cfg = load_json(Path(args.base_config))

    profile = base_cfg.get("dataset_profile")
    if args.profile or not profile:
        profiler = DatasetProfiler(Path(args.data_root))
        profile = profiler.profile()
        print(f"[profile] samples={profile['n_clips']} frames={profile['total_frames']} avg_seq={profile['avg_seq_len']:.1f}")

    builder = TrainingConfigBuilder(base_cfg)
    config = builder.build(profile)
    print(f"[build] epochs={config['epochs']} batch={config['batch']} lr={config['lr']}")

    if args.tune and args.metrics_root:
        val_metrics = load_val_metrics(Path(args.metrics_root))
        adjuster = StageMetricAdjuster(config)
        changed = adjuster.apply(val_metrics)
        if changed:
            print("[tune] updated:", ", ".join(f"{k}={v}" for k, v in changed.items()))
        else:
            print("[tune] no changes applied")

    history = BayesHistory(Path(args.history))
    metrics_summary: Optional[Dict[str, float]] = None
    if args.metrics_root:
        try:
            metrics_summary = aggregate_metrics(Path(args.metrics_root))
        except FileNotFoundError:
            metrics_summary = None

    if args.bayes_update_history and metrics_summary:
        params = extract_param_vector(config, DEFAULT_BAYES_SPECS)
        score = compute_history_score(metrics_summary)
        history.add(params, metrics_summary, score)
        history.save()
        print(f"[history] appended entry score={score:.4f} ({len(history.entries)} total)")

    if args.bayes_opt:
        optimizer = BayesianOptimizer(DEFAULT_BAYES_SPECS, seed=args.seed)
        choice = optimizer.suggest(history.entries)
        if choice:
            apply_param_updates(config, choice)
            print("[bayes] suggestion:", ", ".join(f"{k}={choice[k]:.4g}" for k in sorted(choice)))
        else:
            print("[bayes] insufficient history, skipped")

    if args.dry_run:
        print("[dry-run] updated config not written")
    else:
        dump_json(Path(args.output), config)
        print(f"[write] saved config to {args.output}")


if __name__ == "__main__":
    main()
