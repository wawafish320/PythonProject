from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .io import dump_json

DEFAULT_METRIC_WEIGHTS = {
    "YawAbsDeg": 0.4,
    "RootVelMAE": 0.35,
    "InputRotGeoDeg": 0.25,
}


@dataclass
class ParamSpec:
    key: str
    kind: str  # "float" or "int"
    bounds: tuple[float, float]
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
    """Lightweight TPE-style optimizer implemented without heavy deps."""

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

    def _fit_gaussians(self, entries: Sequence[Mapping[str, Any]]) -> Dict[str, tuple[float, float]]:
        stats_map: Dict[str, tuple[float, float]] = {}
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

    def _sample_from_good(self, good_stats: Mapping[str, tuple[float, float]]) -> Dict[str, float]:
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
        good_stats: Mapping[str, tuple[float, float]],
        bad_stats: Mapping[str, tuple[float, float]],
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


def apply_param_updates(cfg: Mapping[str, Any], updates: Mapping[str, float]) -> None:
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


def _set_nested(obj: Mapping[str, Any], path: Sequence[str], value: float) -> None:
    cursor: Any = obj
    for key in path[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[path[-1]] = float(value)
