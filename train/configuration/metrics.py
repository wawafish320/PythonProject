from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

DEFAULT_TARGETS = {
    "yaw": {"lo_ratio": 0.75, "hi_ratio": 1.1, "ref": 18.0},
    "root": {"lo_ratio": 0.02, "hi_ratio": 0.035, "ref": 0.75},
    "rot": {"lo_ratio": 0.7, "hi_ratio": 1.1, "ref": 2.0},
}


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


class StageMetricAdjuster:
    def __init__(self, config: MutableMapping[str, Any], ref_values: Optional[Mapping[str, float]] = None):
        self.config = config
        self.ref_values = ref_values or (config.get("strategy_meta") or {}).get("reference_targets") or {}
        self.std_values = (config.get("dataset_profile") or {}) if isinstance(config, Mapping) else {}
        meta = config.get("strategy_meta") or {}
        self.smoothing_beta = float(meta.get("metric_ema_beta", 0.4))
        self.adjust_patience = int(meta.get("adjust_patience", 2))
        self.min_delta_ratio = float(meta.get("min_delta_ratio", 0.02))
        # 每个阶段/指标的连续超界计数和EMA
        self._hits: Dict[str, Dict[str, int]] = {}
        self._ema: Dict[str, Dict[str, float]] = {}

    def apply(self, val_metrics: Mapping[int, Metric]) -> Dict[str, float]:
        schedule = self.config.get("freerun_stage_schedule")
        if not schedule:
            raise ValueError("config missing freerun_stage_schedule")
        summary = summarize_stage_metrics(schedule, val_metrics)
        changed: Dict[str, float] = {}
        for stage in schedule:
            label = stage.get("label")
            metrics = summary.get(label)
            if not metrics:
                continue
            smooth = self._smooth_metrics(label, metrics)
            updates = self._adjust_stage(stage, smooth)
            changed.update({f"{label}.{k}": v for k, v in updates.items()})
        return changed

    def _adjust_stage(self, stage: MutableMapping[str, Any], metrics: Mapping[str, float]) -> Dict[str, float]:
        trainer = stage.setdefault("trainer", {})
        loss_cfg = stage.setdefault("loss", {})
        changed: Dict[str, float] = {}

        yaw = metrics.get("YawAbsDeg")
        root_mae = metrics.get("RootVelMAE")
        rot_geo = metrics.get("InputRotGeoDeg")

        yaw_lo, yaw_hi, yaw_ref = _resolve_bounds(stage, "yaw", self.ref_values, self.std_values)
        root_lo, root_hi, root_ref = _resolve_bounds(stage, "root", self.ref_values, self.std_values)
        rot_lo, rot_hi, rot_ref = _resolve_bounds(stage, "rot", self.ref_values, self.std_values)

        freerun_w = float(trainer.get("freerun_weight", 0.0) or 0.0)
        freerun_h = int(trainer.get("freerun_horizon", 0) or 0)
        w_fk = float(loss_cfg.get("w_fk_pos", 0.0) or 0.0)
        w_rot_local = float(loss_cfg.get("w_rot_local", 0.0) or 0.0)

        if yaw is not None and self._should_adjust("yaw", stage, yaw, yaw_lo, yaw_hi, yaw_ref):
            if yaw > yaw_hi:
                new_h = _clamp(int(round(freerun_h * 1.1 or 1)), 4, 64)
            else:
                new_h = _clamp(int(round(freerun_h * 0.9 or 1)), 4, 64)
            if new_h != freerun_h:
                trainer["freerun_horizon"] = new_h
                changed["freerun_horizon"] = trainer["freerun_horizon"]
                freerun_h = new_h

        if root_mae is not None and self._should_adjust("root", stage, root_mae, root_lo, root_hi, root_ref):
            if root_mae > root_hi:
                new = _scale(freerun_w, 1.2, 0.0, 0.8)
            else:
                new = _scale(freerun_w, 0.9, 0.0, 0.8)
            if new != freerun_w:
                trainer["freerun_weight"] = round(new, 4)
                changed["freerun_weight"] = trainer["freerun_weight"]
                freerun_w = new

        if rot_geo is not None and self._should_adjust("rot", stage, rot_geo, rot_lo, rot_hi, rot_ref):
            if rot_geo > rot_hi:
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
            else:
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

    def _smooth_metrics(self, label: str, metrics: Mapping[str, float]) -> Dict[str, float]:
        ema = self._ema.setdefault(label, {})
        smoothed: Dict[str, float] = {}
        for k, v in metrics.items():
            if v is None:
                continue
            prev = ema.get(k, v)
            cur = self.smoothing_beta * v + (1.0 - self.smoothing_beta) * prev
            ema[k] = cur
            smoothed[k] = cur
        return smoothed

    def _should_adjust(self, key: str, stage: Mapping[str, Any], value: float, lo: float, hi: float, ref: float) -> bool:
        if ref != 0:
            if abs(value - ref) < abs(ref) * self.min_delta_ratio:
                return False
        label = stage.get("label", "stage")
        hits = self._hits.setdefault(label, {})
        direction = "high" if value > hi else "low"
        if lo <= value <= hi:
            hits.pop(key, None)
            return False
        count = hits.get(key, 0) + 1
        hits[key] = count
        return count >= self.adjust_patience


def _maybe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_bounds(
    stage: Mapping[str, Any], key: str, ref_values: Mapping[str, float], std_values: Mapping[str, Any]
) -> tuple[float, float, float]:
    stage_targets = stage.get("targets") or {}
    spec = stage_targets.get(key) or {}
    defaults = DEFAULT_TARGETS[key]
    ref = spec.get("ref") or ref_values.get(key) or defaults["ref"]
    std_k = float(spec.get("std_k", 0.75))
    std_lookup = {
        "yaw": std_values.get("yaw_std_deg"),
        "root": std_values.get("speed_std"),
        "rot": std_values.get("bone_angle_std_deg"),
    }
    std_val = _maybe_float(std_lookup.get(key))

    lo_ratio = spec.get("lo_ratio", defaults["lo_ratio"])
    hi_ratio = spec.get("hi_ratio", defaults["hi_ratio"])
    lo = spec.get("lo")
    hi = spec.get("hi")
    if lo is None and std_val is not None:
        lo = ref - std_k * std_val
    if hi is None and std_val is not None:
        hi = ref + std_k * std_val
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
    return float(lo), float(hi), float(ref)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _scale(value: float, factor: float, lo: float, hi: float) -> float:
    return _clamp(value * factor, lo, hi)
