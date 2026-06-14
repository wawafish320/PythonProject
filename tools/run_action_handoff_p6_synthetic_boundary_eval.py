#!/usr/bin/env python3
"""Standalone P6 synthetic-boundary eval scaffold (dry-run only).

This tool is intentionally scaffold/planning stage:
- default is dry-run only
- no evaluator wiring
- no rollout execution
- no train entry changes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from train.validate.injection_windows import WindowSpec, compute_window_bounds, summarize_window_metrics


DEFAULT_Z_FEATURES = Path("debug_output/_tmp_action_handoff_z_probe_v1_20260524/z_features_per_clip.npz")
DEFAULT_P4_ALT_SWEEP_SUMMARY = Path(
    "debug_output/_tmp_action_handoff_z_probe_v1_p4_alt_sweep_20260524/p4_alt_sweep_summary.json"
)
DEFAULT_WALK_L_TO_R_ANALYSIS = Path(
    "debug_output/_tmp_action_handoff_z_probe_v1_walk_l_to_r_failure_analysis_20260524/walk_l_to_r_failure_analysis.json"
)
DEFAULT_SUBSTRATE_SWEEP_CONFIG = Path("debug_output/_tmp_turn_a_to_b_entry_probe_20260515/sweep_config.json")
DEFAULT_SUBSTRATE_ROOT = DEFAULT_SUBSTRATE_SWEEP_CONFIG.parent
ALLOWED_EXECUTION_MODE_DRY = "dry_run_only"
ALLOWED_EXECUTION_MODE_REPLAY = "artifact_replay"
ALLOWED_EXECUTION_MODE_RUNNER = "runner_invoke"
ALLOWED_ARTIFACT_REPLAY_SCOPE_SMOKE = "smoke"
ALLOWED_ARTIFACT_REPLAY_SCOPE_FULL = "full_matrix"
REQUIRED_SAFETY_METRICS = (
    "ContactMismatchRate",
    "FootSlipBallL",
    "FootSlipBallR",
    "RootStepDispErr",
    "GeoLocalDeg",
)


STRONG_NORMAL_PAIRS = [
    ("Walk_F", "Walk_R_To_L"),
    ("Walk_F", "Walk_R_To_R"),
]
WEAK_STRESS_PAIRS = [
    ("Walk_L_To_R", "Walk_R_To_L"),
    ("Walk_L_To_R", "Walk_R_To_R"),
]
ARTIFACT_REPLAY_SMOKE_ROWS = [
    {
        "source_clip": "Walk_F",
        "target_clip": "Walk_R_To_L",
        "horizon_N": 12,
        "case_type": "normal",
        "execution_binding": {
            "mode": "artifact_replay",
            "substrate_trial_id": "trial_003_Walk_R_To_L_M0_N40",
            "metric_window": "entry_window",
        },
    },
    {
        "source_clip": "Walk_L_To_R",
        "target_clip": "Walk_R_To_R",
        "horizon_N": 24,
        "case_type": "weak_stress",
        "execution_binding": {
            "mode": "artifact_replay",
            "substrate_trial_id": "trial_002_Walk_L_To_R_M0_N80",
            "metric_window": "entry_window",
        },
    },
]
FULL_MATRIX_INJECTED_SMOKE_ROWS = [
    {"source_clip": "Walk_F", "target_clip": "Walk_R_To_L", "horizon_N": 12, "case_type": "normal"},
    {"source_clip": "Walk_F", "target_clip": "Walk_R_To_L", "horizon_N": 24, "case_type": "normal"},
    {"source_clip": "Walk_F", "target_clip": "Walk_R_To_R", "horizon_N": 12, "case_type": "normal"},
    {"source_clip": "Walk_F", "target_clip": "Walk_R_To_R", "horizon_N": 24, "case_type": "normal"},
    {"source_clip": "Walk_L_To_R", "target_clip": "Walk_R_To_L", "horizon_N": 12, "case_type": "weak_stress"},
    {"source_clip": "Walk_L_To_R", "target_clip": "Walk_R_To_L", "horizon_N": 24, "case_type": "weak_stress"},
    {"source_clip": "Walk_L_To_R", "target_clip": "Walk_R_To_R", "horizon_N": 12, "case_type": "weak_stress"},
    {"source_clip": "Walk_L_To_R", "target_clip": "Walk_R_To_R", "horizon_N": 24, "case_type": "weak_stress"},
]


def _fatal(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _today_tag_local() -> str:
    return datetime.now().strftime("%Y%m%d")


def _parse_horizons(text: str) -> List[int]:
    chunks = [c.strip() for c in text.split(",") if c.strip()]
    if not chunks:
        _fatal("--horizons must provide at least one integer, e.g. 12,24")
    out: List[int] = []
    seen = set()
    for c in chunks:
        try:
            n = int(c)
        except ValueError as exc:
            _fatal(f"Invalid horizon value: {c!r}")  # pragma: no cover - fatal exit
            raise exc
        if n <= 0:
            _fatal(f"Horizon must be positive, got {n}")
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Standalone P6 synthetic-boundary eval scaffold (dry-run -> minimal-run placeholder)."
    )
    ap.add_argument("--z-features", type=Path, default=DEFAULT_Z_FEATURES)
    ap.add_argument("--p4-alt-sweep-summary", type=Path, default=DEFAULT_P4_ALT_SWEEP_SUMMARY)
    ap.add_argument("--walk-l-to-r-analysis", type=Path, default=DEFAULT_WALK_L_TO_R_ANALYSIS)
    ap.add_argument("--substrate-sweep-config", type=Path, default=DEFAULT_SUBSTRATE_SWEEP_CONFIG)
    ap.add_argument("--trial-matrix", type=Path, default=None)
    ap.add_argument("--horizons", type=str, default="12,24")
    ap.add_argument("--include-weak-stress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-execute", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--execution-mode",
        type=str,
        default=ALLOWED_EXECUTION_MODE_DRY,
        choices=(ALLOWED_EXECUTION_MODE_DRY, ALLOWED_EXECUTION_MODE_REPLAY, ALLOWED_EXECUTION_MODE_RUNNER),
    )
    ap.add_argument(
        "--artifact-replay-scope",
        type=str,
        default=ALLOWED_ARTIFACT_REPLAY_SCOPE_SMOKE,
        choices=(ALLOWED_ARTIFACT_REPLAY_SCOPE_SMOKE, ALLOWED_ARTIFACT_REPLAY_SCOPE_FULL),
    )
    ap.add_argument("--print-commands-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--runner-timeout-s", type=int, default=900)
    ap.add_argument("--allow-missing-substrate", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = Path(f"debug_output/_tmp_action_handoff_p6_synthetic_boundary_eval_{_today_tag_local()}")
    args.horizon_values = _parse_horizons(args.horizons)
    return args


def _validate_execution_gate(args: argparse.Namespace) -> str:
    if not bool(args.allow_execute):
        if str(args.execution_mode) != ALLOWED_EXECUTION_MODE_DRY:
            _fatal(
                f"When --allow-execute=false, --execution-mode must be {ALLOWED_EXECUTION_MODE_DRY!r}, "
                f"got {args.execution_mode!r}."
            )
        if not bool(args.dry_run):
            _fatal("When --allow-execute=false, --dry-run must be true.")
        return ALLOWED_EXECUTION_MODE_DRY

    mode = str(args.execution_mode)
    if mode not in (ALLOWED_EXECUTION_MODE_REPLAY, ALLOWED_EXECUTION_MODE_RUNNER):
        _fatal(
            f"Unsupported execution mode: {args.execution_mode!r}. "
            f"Only {ALLOWED_EXECUTION_MODE_REPLAY!r}/{ALLOWED_EXECUTION_MODE_RUNNER!r} are allowed when --allow-execute=true."
        )
    if bool(args.dry_run):
        _fatal("When --allow-execute=true, --dry-run must be false.")
    if bool(args.allow_missing_substrate):
        _fatal("--allow-missing-substrate is forbidden in execution mode.")
    if int(args.runner_timeout_s) <= 0:
        _fatal("--runner-timeout-s must be > 0.")
    return mode


def _ensure_file(path: Path, *, optional: bool = False) -> Path:
    rp = path.resolve()
    if not rp.exists():
        if optional:
            return rp
        _fatal(f"Required artifact missing: {rp}")
    if not rp.is_file():
        _fatal(f"Artifact path is not a file: {rp}")
    return rp


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - fail-fast path
        _fatal(f"Failed to parse JSON {path}: {exc}")
        raise exc


def _validate_dict(obj: Any, *, name: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        _fatal(f"{name} must be a JSON object, got {type(obj).__name__}")
    return obj


def _finite_or_none(name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        _fatal(f"{name} must be numeric or null, got {value!r}")
    if not math.isfinite(fval):
        _fatal(f"{name} is non-finite: {fval}")
    return fval


def _extract_z_feature_contract(npz_path: Path) -> Dict[str, Any]:
    try:
        pack = np.load(npz_path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - fail-fast path
        _fatal(f"Failed to load z-features npz: {npz_path} ({exc})")
        raise exc

    z_keys = [k for k in pack.files if k.endswith("__z")]
    if not z_keys:
        _fatal(f"No '*__z' keys found in z-features artifact: {npz_path}")

    clip_order_raw = pack.get("clip_order")
    clip_order: List[str] = []
    if clip_order_raw is not None:
        clip_order = [str(x) for x in np.asarray(clip_order_raw).tolist()]

    repr_dims = set()
    dtypes = set()
    per_clip: Dict[str, Dict[str, Any]] = {}
    for key in sorted(z_keys):
        arr = np.asarray(pack[key])
        clip = key.rsplit("__", 1)[0]
        if arr.ndim < 2:
            _fatal(f"{key} must be rank-2+, got shape={arr.shape}")
        repr_dims.add(int(arr.shape[-1]))
        dtypes.add(str(arr.dtype))
        per_clip[clip] = {
            "z_shape": [int(x) for x in arr.shape],
            "dtype": str(arr.dtype),
            "device": "cpu",
        }
    if len(repr_dims) != 1:
        _fatal(f"Inconsistent z repr_dim across clips: {sorted(repr_dims)}")
    if len(dtypes) != 1:
        _fatal(f"Inconsistent z dtype across clips: {sorted(dtypes)}")
    return {
        "repr_dim": int(next(iter(repr_dims))),
        "dtype": str(next(iter(dtypes))),
        "device": "cpu",
        "clip_order": clip_order,
        "per_clip": per_clip,
    }


def _validate_p4_summary(sweep: Dict[str, Any], path: Path) -> List[Dict[str, Any]]:
    configs = sweep.get("configs")
    if not isinstance(configs, list) or not configs:
        _fatal(f"{path} missing non-empty 'configs' list")
    for idx, cfg in enumerate(configs):
        if not isinstance(cfg, dict):
            _fatal(f"{path} configs[{idx}] must be object")
        for key in ("config_id", "future_horizon_n", "summary_json"):
            if key not in cfg:
                _fatal(f"{path} configs[{idx}] missing key '{key}'")
        _ = int(cfg["future_horizon_n"])
    return configs


def _select_config_for_horizon(configs: List[Dict[str, Any]], horizon_n: int) -> Dict[str, Any]:
    exact_id = f"n{horizon_n}_q0p10_topk5"
    for cfg in configs:
        if str(cfg.get("config_id")) == exact_id:
            return cfg
    candidates = [c for c in configs if int(c.get("future_horizon_n")) == horizon_n]
    if not candidates:
        _fatal(f"No P4-alt config found for horizon N={horizon_n}")

    def _score(item: Dict[str, Any]) -> float:
        q = float(item.get("oracle_top_q", 0.1))
        top_k = int(item.get("top_k", 5))
        return abs(q - 0.1) + abs(top_k - 5) * 10.0

    return sorted(candidates, key=_score)[0]


def _load_per_config_summary(cfg: Dict[str, Any]) -> Dict[str, Any]:
    summary_path = _ensure_file(Path(str(cfg["summary_json"])))
    summary = _validate_dict(_load_json(summary_path), name=f"per-config summary {summary_path}")
    per_pair = summary.get("per_pair")
    per_source = summary.get("per_source_clip")
    if not isinstance(per_pair, dict):
        _fatal(f"{summary_path} missing dict 'per_pair'")
    if not isinstance(per_source, dict):
        _fatal(f"{summary_path} missing dict 'per_source_clip'")
    return summary


def _trial_case_type(source_clip: str, target_clip: str, explicit: Optional[str]) -> str:
    if explicit is not None:
        if explicit not in ("normal", "weak_stress"):
            _fatal(f"Invalid case_type/pair_bucket: {explicit!r}")
        return explicit
    if (source_clip, target_clip) in WEAK_STRESS_PAIRS:
        return "weak_stress"
    return "normal"


def _default_trial_matrix(horizons: List[int], include_weak_stress: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in horizons:
        for src, tgt in STRONG_NORMAL_PAIRS:
            rows.append(
                {
                    "source_clip": src,
                    "target_clip": tgt,
                    "horizon_N": int(n),
                    "case_type": "normal",
                }
            )
        if include_weak_stress:
            for src, tgt in WEAK_STRESS_PAIRS:
                rows.append(
                    {
                        "source_clip": src,
                        "target_clip": tgt,
                        "horizon_N": int(n),
                        "case_type": "weak_stress",
                    }
                )
    return rows


def _expand_trial_matrix(trial_matrix_obj: Any, horizons: List[int], include_weak_stress: bool) -> List[Dict[str, Any]]:
    if trial_matrix_obj is None:
        return _default_trial_matrix(horizons, include_weak_stress)

    if isinstance(trial_matrix_obj, dict):
        raw_pairs = trial_matrix_obj.get("pairs")
    elif isinstance(trial_matrix_obj, list):
        raw_pairs = trial_matrix_obj
    else:
        _fatal("--trial-matrix JSON must be object(with 'pairs') or list")
        return []  # unreachable

    if not isinstance(raw_pairs, list) or not raw_pairs:
        _fatal("--trial-matrix has empty/non-list 'pairs'")

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_pairs):
        if not isinstance(item, dict):
            _fatal(f"--trial-matrix pairs[{idx}] must be object")
        src = str(item.get("source_clip", "")).strip()
        tgt = str(item.get("target_clip", "")).strip()
        if not src or not tgt:
            _fatal(f"--trial-matrix pairs[{idx}] missing source_clip/target_clip")
        explicit_case = item.get("case_type")
        if explicit_case is None:
            explicit_case = item.get("pair_bucket")
            if explicit_case == "strong":
                explicit_case = "normal"
        case_type = _trial_case_type(src, tgt, str(explicit_case) if explicit_case is not None else None)
        h_item = item.get("horizon_N")
        if h_item is None:
            for n in horizons:
                out.append({"source_clip": src, "target_clip": tgt, "horizon_N": int(n), "case_type": case_type})
        else:
            n = int(h_item)
            if n <= 0:
                _fatal(f"--trial-matrix pairs[{idx}] has non-positive horizon_N={n}")
            out.append({"source_clip": src, "target_clip": tgt, "horizon_N": n, "case_type": case_type})
    return out


def _build_execution_trial_matrix(
    *,
    scope: str,
    trial_matrix_obj: Any,
    horizons: List[int],
    include_weak_stress: bool,
    execution_mode: str,
    print_commands_only: bool,
) -> List[Dict[str, Any]]:
    if scope == ALLOWED_ARTIFACT_REPLAY_SCOPE_SMOKE:
        if trial_matrix_obj is None:
            rows = [dict(x) for x in ARTIFACT_REPLAY_SMOKE_ROWS]
        else:
            rows = _expand_trial_matrix(trial_matrix_obj, horizons, include_weak_stress)
        if execution_mode == ALLOWED_EXECUTION_MODE_RUNNER and not print_commands_only:
            if trial_matrix_obj is None:
                _fatal("runner_invoke real smoke requires explicit --trial-matrix with exactly 2 rows.")
            if len(rows) != 2:
                _fatal("runner_invoke real smoke allows exactly 2 rows (1 normal + 1 weak_stress).")
            case_types = [str(r.get("case_type")) for r in rows]
            if case_types.count("normal") != 1 or case_types.count("weak_stress") != 1:
                _fatal("runner_invoke real smoke requires case_type composition: exactly 1 normal and 1 weak_stress.")
            for row in rows:
                if int(row.get("horizon_N")) not in (12, 24):
                    _fatal("runner_invoke real smoke requires horizon_N in {12,24}.")
            return rows
        if len(rows) != 2:
            _fatal("artifact_replay smoke scope requires exactly 2 rows (1 normal + 1 weak_stress).")
        case_types = [str(r.get("case_type")) for r in rows]
        if case_types.count("normal") != 1 or case_types.count("weak_stress") != 1:
            _fatal("artifact_replay smoke scope requires case_type composition: exactly 1 normal and 1 weak_stress.")
        return rows

    if scope == ALLOWED_ARTIFACT_REPLAY_SCOPE_FULL:
        rows = [dict(x) for x in FULL_MATRIX_INJECTED_SMOKE_ROWS]
        if trial_matrix_obj is not None:
            custom_rows = _expand_trial_matrix(trial_matrix_obj, horizons, include_weak_stress)
            got = {
                (str(r.get("source_clip")), str(r.get("target_clip")), int(r.get("horizon_N")), str(r.get("case_type")))
                for r in custom_rows
            }
            expected = {
                (str(r["source_clip"]), str(r["target_clip"]), int(r["horizon_N"]), str(r["case_type"])) for r in rows
            }
            if got != expected or len(custom_rows) != len(rows):
                _fatal(
                    "full_matrix injected smoke requires fixed 8-row matrix: "
                    "Walk_F->{Walk_R_To_L,Walk_R_To_R} and Walk_L_To_R->{Walk_R_To_L,Walk_R_To_R} over N={12,24}."
                )
        return rows

    _fatal(f"Unsupported artifact replay scope: {scope!r}")
    return []


def _default_replay_binding_for_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_clip = str(row.get("source_clip"))
    target_clip = str(row.get("target_clip"))
    horizon_n = int(row.get("horizon_N"))
    case_type = str(row.get("case_type"))
    if case_type == "normal" and source_clip == "Walk_F" and target_clip in ("Walk_R_To_L", "Walk_R_To_R"):
        trial_id = "trial_003_Walk_R_To_L_M0_N40" if horizon_n == 12 else "trial_004_Walk_R_To_L_M0_N80" if horizon_n == 24 else None
        if trial_id is None:
            return None
        return {"mode": ALLOWED_EXECUTION_MODE_REPLAY, "substrate_trial_id": trial_id, "metric_window": "entry_window"}
    if case_type == "weak_stress" and source_clip == "Walk_L_To_R" and target_clip in ("Walk_R_To_L", "Walk_R_To_R"):
        trial_id = "trial_001_Walk_L_To_R_M0_N40" if horizon_n == 12 else "trial_002_Walk_L_To_R_M0_N80" if horizon_n == 24 else None
        if trial_id is None:
            return None
        return {"mode": ALLOWED_EXECUTION_MODE_REPLAY, "substrate_trial_id": trial_id, "metric_window": "entry_window"}
    return None


def _resolve_row_execution_binding(row: Dict[str, Any], substrate_root: Path) -> Dict[str, Any]:
    custom = row.get("execution_binding")
    if custom is not None:
        if not isinstance(custom, dict):
            _fatal("execution_binding must be an object when provided.")
        mode = str(custom.get("mode", ""))
        if mode != ALLOWED_EXECUTION_MODE_REPLAY:
            _fatal(f"execution_binding.mode must be {ALLOWED_EXECUTION_MODE_REPLAY!r}, got {mode!r}")
        trial_id = str(custom.get("substrate_trial_id", "")).strip()
        metric_window = str(custom.get("metric_window", "entry_window") or "entry_window")
    else:
        default_binding = _default_replay_binding_for_row(row)
        if default_binding is None:
            return {
                "mode": ALLOWED_EXECUTION_MODE_REPLAY,
                "binding_status": "artifact_binding_missing",
                "binding_reason": "artifact_binding_missing_for_row",
            }
        trial_id = str(default_binding["substrate_trial_id"])
        metric_window = str(default_binding["metric_window"])
    if not trial_id:
        return {
            "mode": ALLOWED_EXECUTION_MODE_REPLAY,
            "binding_status": "artifact_binding_missing",
            "binding_reason": "empty_substrate_trial_id",
        }
    if metric_window not in ("entry_window", "post_inject_recovery"):
        _fatal("execution_binding.metric_window must be 'entry_window' or 'post_inject_recovery'.")

    trial_dir = substrate_root / "trials" / trial_id
    paired_delta = (trial_dir / "paired_delta.json").resolve()
    freerun_json = (trial_dir / "Walk_F_freerun_cycles.json").resolve()
    missing_paths: List[str] = []
    if not trial_dir.is_dir():
        missing_paths.append(str(trial_dir))
    if not paired_delta.is_file():
        missing_paths.append(str(paired_delta))
    if not freerun_json.is_file():
        missing_paths.append(str(freerun_json))
    if missing_paths:
        return {
            "mode": ALLOWED_EXECUTION_MODE_REPLAY,
            "binding_status": "artifact_binding_missing",
            "binding_reason": "artifact_path_missing",
            "missing_paths": missing_paths,
            "substrate_trial_id": trial_id,
            "metric_window": metric_window,
            "trial_dir": str(trial_dir.resolve()),
        }
    return {
        "mode": ALLOWED_EXECUTION_MODE_REPLAY,
        "binding_status": "bound",
        "substrate_trial_id": trial_id,
        "trial_dir": str(trial_dir.resolve()),
        "paired_delta_json": str(paired_delta),
        "freerun_json": str(freerun_json),
        "metric_window": metric_window,
    }


def _extract_safety_metrics_from_replay(binding: Dict[str, Any]) -> Dict[str, Any]:
    paired = _validate_dict(_load_json(Path(binding["paired_delta_json"])), name="paired_delta replay payload")
    window_name = str(binding["metric_window"])
    window_obj = paired.get(window_name)
    if not isinstance(window_obj, dict):
        _fatal(f"paired_delta missing window object: {window_name}")
    metric_summary = window_obj.get("metric_summary")
    if not isinstance(metric_summary, dict):
        _fatal(f"paired_delta {window_name} missing metric_summary")

    out: Dict[str, Any] = {"status": "executed_artifact_replay_v1"}
    source_used: Dict[str, Optional[str]] = {}
    missing_metrics: List[str] = []
    for metric_name in REQUIRED_SAFETY_METRICS:
        rec = metric_summary.get(metric_name)
        if not isinstance(rec, dict):
            _fatal(f"paired_delta {window_name} missing metric_summary[{metric_name}]")
        n = int(rec.get("n", 0))
        mean_val = rec.get("mean")
        if n <= 0:
            if metric_name in ("FootSlipBallL", "FootSlipBallR"):
                out[metric_name] = None
                source_used[metric_name] = None
                missing_metrics.append(metric_name)
                continue
            _fatal(f"paired_delta {window_name} metric {metric_name} has n<=0 and cannot be null.")
        out[metric_name] = _finite_or_none(f"{window_name}.{metric_name}.mean", mean_val)
        if out[metric_name] is None:
            _fatal(f"paired_delta {window_name} metric {metric_name} mean cannot be null when n>0.")
        source_used[metric_name] = metric_name
    out["execution_mode"] = ALLOWED_EXECUTION_MODE_REPLAY
    out["source_metric_window"] = window_name
    out["missing_metrics"] = missing_metrics
    out["metric_source_used"] = source_used
    out["proxy_metric_used"] = False
    out["proxy_metric_fields"] = []
    out["canonical_metric_missing"] = list(missing_metrics)
    out["canonical_metric_complete"] = len(missing_metrics) == 0
    out["injection_apply_records_n"] = 0
    out["injection_field_apply"] = {
        "rootvel": {"requested": True, "applied": None, "reason": "artifact_replay_no_injection_record"},
        "rot6d": {"requested": True, "applied": None, "reason": "artifact_replay_no_injection_record"},
        "angvel": {"requested": True, "applied": None, "reason": "artifact_replay_no_injection_record"},
    }
    out["injection_warnings"] = ["artifact_replay_no_runtime_injection_records"]
    return out


def _horizon_to_inject_at_step(horizon_n: int) -> int:
    if int(horizon_n) == 12:
        return 40
    if int(horizon_n) == 24:
        return 80
    raise ValueError(f"runner_invoke supports only horizon_N in {{12,24}}, got {horizon_n}")


def _build_runner_invoke_binding(
    *,
    row: Dict[str, Any],
    substrate_cfg: Dict[str, Any],
    out_dir: Path,
    timeout_s: int,
) -> Dict[str, Any]:
    model_path = str(substrate_cfg.get("model", "")).strip()
    teacher_path = str(substrate_cfg.get("teacher", "")).strip()
    if not model_path or not teacher_path:
        raise ValueError("substrate sweep_config must contain model/teacher for runner_invoke.")
    source_clip = str(row["source_clip"])
    source_teacher = (Path("validate/teacher_batches") / f"{source_clip}_teacher.json").resolve()
    if source_teacher.is_file():
        teacher_path = str(source_teacher)
    else:
        raise FileNotFoundError(
            f"source_clip teacher missing for runner_invoke row binding audit: {source_teacher}"
        )

    target_clip = str(row["target_clip"])
    horizon_n = int(row["horizon_N"])
    inject_at_step = _horizon_to_inject_at_step(horizon_n)
    turn_npz = (Path("raw_data/processed_data") / f"{target_clip}.npz").resolve()
    if not turn_npz.is_file():
        raise FileNotFoundError(f"target_clip npz missing for runner_invoke: {turn_npz}")

    rounds = int(substrate_cfg.get("rounds", 4) or 4)
    inject_fields = "rootvel,rot6d,angvel"
    inject_from_step = 0
    device = str(substrate_cfg.get("device", "auto") or "auto")

    trial_slug = str(row.get("trial_id", f"{row['source_clip']}_{row['target_clip']}_{horizon_n}")).replace(":", "_").replace("/", "_")
    trial_out_dir = (out_dir / "runner_invoke_trials" / trial_slug).resolve()
    inject_label = f"{row['source_clip']}_to_{row['target_clip']}_N{horizon_n}"

    cmd: List[str] = [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--model",
        model_path,
        "--teacher",
        teacher_path,
        "--bundle",
        "raw_data/processed_data/norm_template.json",
        "--npz-root",
        "raw_data/processed_data",
        "--out",
        str(trial_out_dir),
        "--rounds",
        str(rounds),
        "--device",
        device,
        "--force",
        "--inject-turn-npz",
        str(turn_npz),
        "--inject-at-step",
        str(inject_at_step),
        "--inject-from-step",
        str(inject_from_step),
        "--inject-fields",
        inject_fields,
        "--inject-label",
        inject_label,
        "--log_contacts_whitebox",
    ]
    if bool(substrate_cfg.get("lambda_fusion_apply", False)):
        cmd.append("--lambda_fusion_apply")
    extra_cli = str(substrate_cfg.get("extra_cli", "") or "").strip()
    if extra_cli:
        cmd.extend(shlex.split(extra_cli))

    return {
        "mode": ALLOWED_EXECUTION_MODE_RUNNER,
        "binding_status": "planned_print_only",
        "runner_entry": "python -m train.validate.run_freerun_cycles",
        "timeout_s": int(timeout_s),
        "trial_out_dir": str(trial_out_dir),
        "command": cmd,
        "command_shell": " ".join(shlex.quote(tok) for tok in cmd),
        "row_mapping": {
            "source_clip": str(row["source_clip"]),
            "target_clip": str(row["target_clip"]),
            "teacher": str(teacher_path),
            "horizon_N": int(horizon_n),
            "inject_at_step": int(inject_at_step),
            "turn_npz": str(turn_npz),
            "inject_from_step": int(inject_from_step),
            "inject_fields": inject_fields,
            "inject_label": inject_label,
            "inject_args_supported_by_runner_cli": True,
            "runner_smoke_proxy_note": None,
        },
    }


def _execute_runner_invoke_binding(binding: Dict[str, Any]) -> Dict[str, Any]:
    cmd = binding.get("command")
    if not isinstance(cmd, list) or not cmd:
        raise ValueError("runner binding missing command list")
    timeout_s = int(binding.get("timeout_s", 900))
    trial_out_dir = Path(str(binding.get("trial_out_dir", ""))).resolve()
    trial_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as ex:
        return {
            "ok": False,
            "error_type": "TimeoutExpired",
            "error_message": str(ex),
            "returncode": None,
            "stdout": (ex.stdout or "")[-4000:],
            "stderr": (ex.stderr or "")[-4000:],
            "freerun_json": str((trial_out_dir / "Walk_F_freerun_cycles.json").resolve()),
        }

    freerun_json = (trial_out_dir / "Walk_F_freerun_cycles.json").resolve()
    if not freerun_json.is_file():
        cands = sorted(trial_out_dir.glob("*_freerun_cycles.json"))
        if len(cands) == 1:
            freerun_json = cands[0].resolve()
    if proc.returncode != 0:
        return {
            "ok": False,
            "error_type": "RunnerReturnNonZero",
            "error_message": f"runner returned non-zero rc={proc.returncode}",
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout)[-4000:],
            "stderr": str(proc.stderr)[-4000:],
            "freerun_json": str(freerun_json),
        }
    if not freerun_json.is_file():
        return {
            "ok": False,
            "error_type": "MissingRunnerArtifact",
            "error_message": f"expected artifact not found: {freerun_json}",
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout)[-4000:],
            "stderr": str(proc.stderr)[-4000:],
            "freerun_json": str(freerun_json),
        }
    return {
        "ok": True,
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout)[-4000:],
        "stderr": str(proc.stderr)[-4000:],
        "freerun_json": str(freerun_json),
    }


def _extract_safety_metrics_from_runner_output(freerun_json_path: Path) -> Dict[str, Any]:
    payload = _validate_dict(_load_json(freerun_json_path), name="runner freerun payload")
    per_step = payload.get("metrics_per_step")
    if not isinstance(per_step, list) or not per_step:
        raise ValueError("runner freerun payload missing non-empty metrics_per_step")

    def _mean_metric(metric_name: str) -> Optional[float]:
        vals: List[float] = []
        for rec in per_step:
            if not isinstance(rec, dict):
                continue
            v = rec.get(metric_name)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                vals.append(fv)
        if not vals:
            return None
        return float(sum(vals) / float(len(vals)))

    def _mean_metric_nonzero(metric_name: str) -> Optional[float]:
        vals: List[float] = []
        for rec in per_step:
            if not isinstance(rec, dict):
                continue
            v = rec.get(metric_name)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            if abs(fv) <= 1e-12:
                continue
            vals.append(fv)
        if not vals:
            return None
        return float(sum(vals) / float(len(vals)))

    metric_candidates: Dict[str, List[str]] = {
        "ContactMismatchRate": ["ContactMismatchRate", "ContactErrAbsMean"],
        "FootSlipBallL": ["FootSlipBallL"],
        "FootSlipBallR": ["FootSlipBallR"],
        "RootStepDispErr": ["RootStepDispErr"],
        "GeoLocalDeg": ["GeoLocalDeg"],
    }
    source_used: Dict[str, Optional[str]] = {}
    missing_metrics: List[str] = []
    canonical_missing: List[str] = []
    proxy_metric_fields: List[str] = []
    out: Dict[str, Any] = {"status": "executed_runner_invoke_v1", "execution_mode": ALLOWED_EXECUTION_MODE_RUNNER}
    for metric_name in REQUIRED_SAFETY_METRICS:
        mv = None
        src_name = None
        if metric_name in ("FootSlipBallL", "FootSlipBallR"):
            # Keep canonical skip contract: rows with no dual-frame GT contact remain null.
            # Use non-zero mean to avoid degenerate 0.0 from non-applicable frames.
            mv = _mean_metric_nonzero(metric_name)
            if mv is not None:
                src_name = metric_name
        else:
            for cand in metric_candidates.get(metric_name, [metric_name]):
                mv = _mean_metric(cand)
                if mv is not None:
                    src_name = cand
                    break
        if mv is None:
            if metric_name in ("FootSlipBallL", "FootSlipBallR"):
                out[metric_name] = None
                source_used[metric_name] = None
                missing_metrics.append(metric_name)
                canonical_missing.append(metric_name)
                continue
            raise ValueError(f"required runner metric missing/non-finite: {metric_name}")
        out[metric_name] = mv
        source_used[metric_name] = src_name
        if metric_name == "ContactMismatchRate" and src_name == "ContactErrAbsMean":
            proxy_metric_fields.append("ContactMismatchRate")
            canonical_missing.append("ContactMismatchRate")
    inject_records = payload.get("injection_apply_records")
    if not isinstance(inject_records, list):
        inject_records = []
    per_field_report: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    if inject_records:
        first = inject_records[0]
        fields = first.get("fields_applied") if isinstance(first, dict) else None
        if isinstance(fields, list):
            for field_name in ("rootvel", "rot6d", "angvel"):
                rec = next(
                    (
                        x
                        for x in fields
                        if isinstance(x, dict) and str(x.get("field", "")).strip().lower() == field_name
                    ),
                    None,
                )
                if isinstance(rec, dict):
                    per_field_report[field_name] = {
                        "requested": bool(rec.get("requested", False)),
                        "applied": bool(rec.get("applied", False)),
                        "reason": str(rec.get("reason", "")),
                        "target_slice": rec.get("target_slice"),
                        "payload_shape": rec.get("payload_shape"),
                    }
                else:
                    per_field_report[field_name] = {
                        "requested": False,
                        "applied": False,
                        "reason": "field_record_missing",
                        "target_slice": None,
                        "payload_shape": None,
                    }
        else:
            warnings.append("injection_apply_records[0].fields_applied missing_or_invalid")
    else:
        warnings.append("injection_apply_records_missing_or_empty")
        for field_name in ("rootvel", "rot6d", "angvel"):
            per_field_report[field_name] = {
                "requested": False,
                "applied": False,
                "reason": "record_missing",
                "target_slice": None,
                "payload_shape": None,
            }
    angvel_report = per_field_report.get("angvel", {})
    if (
        bool(angvel_report.get("requested"))
        and not bool(angvel_report.get("applied"))
        and str(angvel_report.get("reason")) == "target_slice_missing"
    ):
        warnings.append("angvel_target_slice_missing_expected_under_current_runner_layout")
    out["injection_apply_records_n"] = int(len(inject_records))
    out["injection_field_apply"] = per_field_report
    out["injection_warnings"] = warnings
    out["missing_metrics"] = missing_metrics
    out["freerun_json"] = str(freerun_json_path.resolve())
    out["window_policy"] = "full_sequence_mean_v1"
    out["metric_source_used"] = source_used
    out["proxy_metric_used"] = len(proxy_metric_fields) > 0
    out["proxy_metric_fields"] = proxy_metric_fields
    out["canonical_metric_missing"] = canonical_missing
    out["canonical_metric_complete"] = len(canonical_missing) == 0
    try:
        per_step_payload = json.dumps(per_step, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        out["metrics_per_step_sha256"] = hashlib.sha256(per_step_payload.encode("utf-8")).hexdigest()
        out["metrics_per_step_n"] = int(len(per_step))
    except Exception as ex:
        warnings.append(f"metrics_per_step_sha256_failed:{type(ex).__name__}:{ex}")
        out["metrics_per_step_sha256"] = None
        out["metrics_per_step_n"] = int(len(per_step))
    inject_at_step = payload.get("inject_at_step")
    inject_turn_npz = payload.get("inject_turn_npz")
    out["inject_at_step"] = int(inject_at_step) if isinstance(inject_at_step, int) else None
    out["inject_turn_npz"] = str(inject_turn_npz) if inject_turn_npz is not None else None

    try:
        if isinstance(inject_at_step, int):
            bounds = compute_window_bounds(
                inject_at_step=int(inject_at_step),
                total_steps=len(per_step),
                spec=WindowSpec(entry_window_pre_k=8, entry_window_post_k=8, recovery_window_k=16),
            )
            windows = summarize_window_metrics(
                per_step_metrics=per_step,
                bounds=bounds,
                required_metrics=list(REQUIRED_SAFETY_METRICS),
            )
            out["window_policy"] = "post_inject_fixed_window_v1"
            out["windowed_metric_summary"] = windows
    except Exception as ex:
        warnings.append(f"windowed_metric_summary_failed:{type(ex).__name__}:{ex}")

    # Row binding audit anchors.
    row_binding_audit = {}
    row_binding_audit["injection_apply_record_step"] = (
        int(inject_records[0].get("step")) if inject_records and isinstance(inject_records[0], dict) else None
    )
    row_binding_audit["injection_apply_record_matches_inject_at_step"] = (
        isinstance(inject_at_step, int)
        and row_binding_audit["injection_apply_record_step"] is not None
        and int(row_binding_audit["injection_apply_record_step"]) == int(inject_at_step)
    )
    out["row_binding_audit"] = row_binding_audit
    return out


def _known_weak_pair(source_clip: str, target_clip: str) -> bool:
    return (source_clip, target_clip) in WEAK_STRESS_PAIRS


def _augment_metric_completeness(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metrics)
    missing_metrics = out.get("missing_metrics")
    if not isinstance(missing_metrics, list):
        missing_metrics = []
    source_used = out.get("metric_source_used")
    if not isinstance(source_used, dict):
        source_used = {}
    proxy_fields = out.get("proxy_metric_fields")
    if not isinstance(proxy_fields, list):
        proxy_fields = []
    canonical_missing = out.get("canonical_metric_missing")
    if not isinstance(canonical_missing, list):
        canonical_missing = []

    if source_used.get("ContactMismatchRate") == "ContactErrAbsMean":
        if "ContactMismatchRate" not in proxy_fields:
            proxy_fields.append("ContactMismatchRate")
        if "ContactMismatchRate" not in canonical_missing:
            canonical_missing.append("ContactMismatchRate")
    for metric_name in REQUIRED_SAFETY_METRICS:
        if out.get(metric_name) is None and metric_name not in canonical_missing:
            canonical_missing.append(metric_name)
        if out.get(metric_name) is None and metric_name not in missing_metrics:
            missing_metrics.append(metric_name)

    out["metric_source_used"] = source_used
    out["missing_metrics"] = missing_metrics
    out["proxy_metric_fields"] = proxy_fields
    out["proxy_metric_used"] = bool(proxy_fields)
    out["canonical_metric_missing"] = canonical_missing
    out["canonical_metric_complete"] = len(canonical_missing) == 0
    out.setdefault("injection_apply_records_n", 0)
    out.setdefault(
        "injection_field_apply",
        {
            "rootvel": {
                "requested": True,
                "applied": None,
                "reason": "not_reported",
                "target_slice": None,
                "payload_shape": None,
            },
            "rot6d": {
                "requested": True,
                "applied": None,
                "reason": "not_reported",
                "target_slice": None,
                "payload_shape": None,
            },
            "angvel": {
                "requested": True,
                "applied": None,
                "reason": "not_reported",
                "target_slice": None,
                "payload_shape": None,
            },
        },
    )
    iw = out.get("injection_warnings")
    if not isinstance(iw, list):
        iw = []
    out["injection_warnings"] = iw
    out.setdefault("windowed_metric_summary", None)
    out.setdefault("row_binding_audit", {})
    return out


def _compute_fallback(
    *,
    case_type: str,
    source_clip: str,
    target_clip: str,
    horizon_n: int,
    z_distance: Optional[float],
    future_equiv_score: Optional[float],
    spearman: Optional[float],
    source_pass_like: Optional[bool],
    weak_flags: Dict[str, Any],
) -> Dict[str, Any]:
    known_weak = bool(case_type == "weak_stress" or _known_weak_pair(source_clip, target_clip))
    long_warn = bool(known_weak and horizon_n >= 24)
    reasons: List[str] = []
    status = "selected"

    if z_distance is None or future_equiv_score is None or spearman is None:
        reasons.append("artifact_metric_missing")
    elif known_weak:
        if z_distance >= 0.95:
            reasons.append("z_distance_too_large")
        if future_equiv_score <= 0.0:
            reasons.append("future_equiv_below_floor")
        if spearman <= 0.20:
            reasons.append("insufficient_future_equiv_signal")
        if bool(weak_flags.get("target_specific_weakness", False)):
            reasons.append("stress_pair_policy")
        if long_warn and bool(weak_flags.get("long_horizon_degradation", False)):
            reasons.append("long_horizon_degradation_risk")
    else:
        # Strong rows stay selected by dry-run default unless artifact-level pass-like says otherwise.
        if source_pass_like is False:
            reasons.append("artifact_source_not_pass_like")

    no_good = False
    if known_weak and long_warn and z_distance is not None and future_equiv_score is not None:
        if z_distance >= 1.0 and future_equiv_score <= 0.0:
            no_good = True

    if no_good:
        status = "no_good_candidate"
        if "stress_pair_no_good_candidate" not in reasons:
            reasons.append("stress_pair_no_good_candidate")
    elif reasons:
        status = "fallback"

    if status == "selected":
        fallback_triggered = False
        fallback_reason = None
    else:
        fallback_triggered = True
        fallback_reason = reasons[0] if reasons else "policy_fallback"

    return {
        "retrieval_status": status,
        "fallback_triggered": bool(fallback_triggered),
        "fallback_reason": fallback_reason,
        "no_good_candidate": bool(status == "no_good_candidate"),
        "long_horizon_warning": bool(long_warn),
        "known_weak_source_risk": bool(known_weak),
        "fallback_reasons_all": reasons,
    }


def _validate_row_schema(row: Dict[str, Any]) -> None:
    for key in ("trial_id", "source_clip", "target_clip", "horizon_N", "case_type"):
        if key not in row:
            _fatal(f"Row missing key: {key}")
    if row["case_type"] not in ("normal", "weak_stress"):
        _fatal(f"Invalid case_type: {row['case_type']!r}")

    meta = row.get("p6_retrieval_metadata")
    fb = row.get("p6_fallback")
    sm = row.get("p6_safety_metrics")
    decision = row.get("decision")
    if not isinstance(meta, dict) or not isinstance(fb, dict) or not isinstance(sm, dict) or not isinstance(decision, dict):
        _fatal(f"Row {row['trial_id']} has invalid nested schema")

    zc = meta.get("z_feature_contract")
    if not isinstance(zc, dict):
        _fatal(f"Row {row['trial_id']} missing p6_retrieval_metadata.z_feature_contract")
    for key in ("repr_dim", "dtype", "device"):
        if key not in zc:
            _fatal(f"Row {row['trial_id']} missing z_feature_contract.{key}")

    status = str(fb.get("retrieval_status"))
    if status not in ("selected", "fallback", "no_good_candidate", "not_executed"):
        _fatal(f"Row {row['trial_id']} has invalid retrieval_status={status!r}")
    if bool(fb.get("long_horizon_warning")) and int(row["horizon_N"]) < 24:
        _fatal(f"Row {row['trial_id']} has long_horizon_warning=true but horizon_N<24")
    if row["case_type"] == "weak_stress" and int(row["horizon_N"]) >= 24 and not bool(fb.get("long_horizon_warning")):
        _fatal(f"Row {row['trial_id']} weak_stress long horizon must set long_horizon_warning=true")
    safety_status = str(sm.get("status"))
    if safety_status not in (
        "not_executed_dry_run",
        "executed_artifact_replay_v1",
        "executed_runner_invoke_v1",
        "artifact_binding_missing",
        "runner_invoke_print_only",
        "runner_invoke_failed",
    ):
        _fatal(
            f"Row {row['trial_id']} p6_safety_metrics.status must be one of "
            "'not_executed_dry_run'/'executed_artifact_replay_v1'/'executed_runner_invoke_v1'/'artifact_binding_missing'/'runner_invoke_print_only'/'runner_invoke_failed'"
        )
    decision_status = str(decision.get("status"))
    if decision_status not in (
        "not_evaluated_dry_run",
        "executed_smoke_not_pass_gate",
        "executed_runner_smoke_not_pass_gate",
        "full_matrix_smoke_not_pass_gate",
        "artifact_binding_missing",
        "runner_invoke_print_only",
        "runner_invoke_failed",
    ):
        _fatal(
            f"Row {row['trial_id']} decision.status must be "
            "'not_evaluated_dry_run'/'executed_smoke_not_pass_gate'/'executed_runner_smoke_not_pass_gate'/'full_matrix_smoke_not_pass_gate'/'artifact_binding_missing'/'runner_invoke_print_only'/'runner_invoke_failed'"
        )


def _build_rows(
    *,
    trial_matrix: List[Dict[str, Any]],
    cfg_by_horizon: Dict[int, Dict[str, Any]],
    per_cfg_summary: Dict[str, Dict[str, Any]],
    z_contract: Dict[str, Any],
    weak_analysis: Dict[str, Any],
    execution_mode: str,
    substrate_root: Path,
    substrate_cfg: Optional[Dict[str, Any]],
    out_dir: Path,
    print_commands_only: bool,
    runner_timeout_s: int,
) -> List[Dict[str, Any]]:
    weak_flags = weak_analysis.get("flags", {})
    if not isinstance(weak_flags, dict):
        weak_flags = {}

    rows: List[Dict[str, Any]] = []
    for idx, trial in enumerate(trial_matrix):
        source_clip = str(trial["source_clip"])
        target_clip = str(trial["target_clip"])
        horizon_n = int(trial["horizon_N"])
        case_type = str(trial["case_type"])
        pair_key = f"{source_clip}->{target_clip}"

        cfg = cfg_by_horizon.get(horizon_n)
        if cfg is None:
            _fatal(f"No selected config for horizon_N={horizon_n}")
        cfg_id = str(cfg["config_id"])
        summary = per_cfg_summary[cfg_id]
        per_pair = summary["per_pair"]
        per_source = summary["per_source_clip"]

        pair_stats = per_pair.get(pair_key)
        if not isinstance(pair_stats, dict):
            _fatal(f"Pair not found in selected config summary: {pair_key} (config={cfg_id})")
        src_stats = per_source.get(source_clip)
        if not isinstance(src_stats, dict):
            _fatal(f"source_clip not found in selected config summary per_source_clip: {source_clip} (config={cfg_id})")

        pair_ratio = _finite_or_none(
            f"{cfg_id}:{pair_key}:top1_future_distance_vs_random_ratio",
            pair_stats.get("top1_future_distance_vs_random_ratio"),
        )
        pair_spearman = _finite_or_none(
            f"{cfg_id}:{pair_key}:mean_spearman_zdist_vs_futuredist",
            pair_stats.get("mean_spearman_zdist_vs_futuredist"),
        )
        pair_hit_lift = _finite_or_none(
            f"{cfg_id}:{pair_key}:top1_equiv_hit_rate_vs_random_top1",
            pair_stats.get("top1_equiv_hit_rate_vs_random_top1"),
        )
        top1_hit = _finite_or_none(f"{cfg_id}:{pair_key}:top1_equiv_hit_rate", pair_stats.get("top1_equiv_hit_rate"))
        random_top1 = _finite_or_none(
            f"{cfg_id}:{pair_key}:random_top1_expectation", pair_stats.get("random_top1_expectation")
        )
        z_margin = None if (top1_hit is None or random_top1 is None) else float(random_top1 - top1_hit)

        src_ratio = _finite_or_none(
            f"{cfg_id}:{source_clip}:source_ratio", src_stats.get("top1_future_distance_vs_random_ratio")
        )
        src_spearman = _finite_or_none(
            f"{cfg_id}:{source_clip}:source_spearman", src_stats.get("mean_spearman_zdist_vs_futuredist")
        )

        source_pass_like_obj = cfg.get("per_source_pass_like", {})
        source_pass_like = None
        if isinstance(source_pass_like_obj, dict) and source_clip in source_pass_like_obj:
            source_pass_like = bool(source_pass_like_obj[source_clip])

        fallback = _compute_fallback(
            case_type=case_type,
            source_clip=source_clip,
            target_clip=target_clip,
            horizon_n=horizon_n,
            z_distance=pair_ratio,
            future_equiv_score=pair_hit_lift,
            spearman=pair_spearman,
            source_pass_like=source_pass_like,
            weak_flags=weak_flags,
        )

        trial_id = str(trial.get("trial_id") or f"{cfg_id}:{pair_key}:N{horizon_n}:{idx:03d}")
        execution_binding = None
        if execution_mode == ALLOWED_EXECUTION_MODE_REPLAY:
            execution_binding = _resolve_row_execution_binding(trial, substrate_root)
            if str(execution_binding.get("binding_status")) == "bound":
                safety_metrics = _augment_metric_completeness(_extract_safety_metrics_from_replay(execution_binding))
                decision_obj = {
                    "status": "full_matrix_smoke_not_pass_gate",
                    "note": "Execution mode=artifact_replay (existing artifact read-only), injected smoke only; not pass gate.",
                }
            else:
                safety_metrics = _augment_metric_completeness(
                    {
                    "status": "artifact_binding_missing",
                    "ContactMismatchRate": None,
                    "FootSlipBallL": None,
                    "FootSlipBallR": None,
                    "RootStepDispErr": None,
                    "GeoLocalDeg": None,
                    "execution_mode": ALLOWED_EXECUTION_MODE_REPLAY,
                    "binding_reason": str(execution_binding.get("binding_reason", "artifact_binding_missing")),
                    }
                )
                decision_obj = {
                    "status": "artifact_binding_missing",
                    "note": "artifact_replay binding missing for this row; metrics not backfilled.",
                }
        elif execution_mode == ALLOWED_EXECUTION_MODE_RUNNER:
            try:
                if substrate_cfg is None:
                    raise ValueError("runner_invoke requires loaded substrate sweep config.")
                execution_binding = _build_runner_invoke_binding(
                    row={**trial, "trial_id": trial_id},
                    substrate_cfg=substrate_cfg,
                    out_dir=out_dir,
                    timeout_s=runner_timeout_s,
                )
                if print_commands_only:
                    safety_metrics = _augment_metric_completeness(
                        {
                        "status": "runner_invoke_print_only",
                        "ContactMismatchRate": None,
                        "FootSlipBallL": None,
                        "FootSlipBallR": None,
                        "RootStepDispErr": None,
                        "GeoLocalDeg": None,
                        "execution_mode": ALLOWED_EXECUTION_MODE_RUNNER,
                        "print_commands_only": True,
                        }
                    )
                    decision_obj = {
                        "status": "runner_invoke_print_only",
                        "note": "runner_invoke command constructed only; subprocess execution is intentionally disabled in this phase.",
                    }
                else:
                    exec_result = _execute_runner_invoke_binding(execution_binding)
                    execution_binding["run_result"] = exec_result
                    if bool(exec_result.get("ok")):
                        execution_binding["binding_status"] = "executed"
                        safety_metrics = _augment_metric_completeness(
                            _extract_safety_metrics_from_runner_output(Path(str(exec_result["freerun_json"])))
                        )
                        decision_obj = {
                            "status": "full_matrix_smoke_not_pass_gate",
                            "note": "runner_invoke injected smoke executed for this row; execution coverage only, not pass gate.",
                        }
                    else:
                        execution_binding["binding_status"] = "execute_failed"
                        safety_metrics = _augment_metric_completeness(
                            {
                            "status": "runner_invoke_failed",
                            "ContactMismatchRate": None,
                            "FootSlipBallL": None,
                            "FootSlipBallR": None,
                            "RootStepDispErr": None,
                            "GeoLocalDeg": None,
                            "execution_mode": ALLOWED_EXECUTION_MODE_RUNNER,
                            "error_type": str(exec_result.get("error_type", "RunnerInvokeFailed")),
                            "error_message": str(exec_result.get("error_message", "runner invoke failed")),
                            }
                        )
                        decision_obj = {
                            "status": "runner_invoke_failed",
                            "note": (
                                "runner_invoke row execution failed: "
                                f"{exec_result.get('error_type')}: {exec_result.get('error_message')}"
                            ),
                        }
            except Exception as ex:
                safety_metrics = _augment_metric_completeness(
                    {
                    "status": "runner_invoke_failed",
                    "ContactMismatchRate": None,
                    "FootSlipBallL": None,
                    "FootSlipBallR": None,
                    "RootStepDispErr": None,
                    "GeoLocalDeg": None,
                    "execution_mode": ALLOWED_EXECUTION_MODE_RUNNER,
                    "error_type": type(ex).__name__,
                    "error_message": str(ex),
                    }
                )
                decision_obj = {
                    "status": "runner_invoke_failed",
                    "note": f"runner_invoke command construction failed: {type(ex).__name__}: {ex}",
                }
                execution_binding = {
                    "mode": ALLOWED_EXECUTION_MODE_RUNNER,
                    "binding_status": "command_build_failed",
                    "error_type": type(ex).__name__,
                    "error_message": str(ex),
                }
        else:
            safety_metrics = _augment_metric_completeness(
                {
                "status": "not_executed_dry_run",
                "ContactMismatchRate": None,
                "FootSlipBallL": None,
                "FootSlipBallR": None,
                "RootStepDispErr": None,
                "GeoLocalDeg": None,
                "execution_mode": ALLOWED_EXECUTION_MODE_DRY,
                }
            )
            decision_obj = {
                "status": "not_evaluated_dry_run",
                "note": "Dry-run scaffold row only; evaluator execution is intentionally not wired.",
            }
        row = {
            "trial_id": trial_id,
            "source_clip": source_clip,
            "target_clip": target_clip,
            "horizon_N": horizon_n,
            "case_type": case_type,
            "p6_retrieval_metadata": {
                "source_clip": source_clip,
                "target_clip": target_clip,
                "source_frame": None,
                "selected_target_frame": None,
                "horizon_N": horizon_n,
                "z_distance": pair_ratio,
                "z_rank": 1 if pair_ratio is not None else None,
                "z_margin": z_margin,
                "future_equiv_score": pair_hit_lift,
                "p4_alt_context": {
                    "source_mean_ratio": src_ratio,
                    "source_mean_spearman": src_spearman,
                    "pair_mean_ratio": pair_ratio,
                    "pair_mean_spearman": pair_spearman,
                },
                "z_feature_contract": {
                    "repr_dim": z_contract["repr_dim"],
                    "dtype": z_contract["dtype"],
                    "device": z_contract["device"],
                },
            },
            "p6_fallback": fallback,
            "p6_safety_metrics": safety_metrics,
            "decision": decision_obj,
            "provenance": {
                "selected_config_id": cfg_id,
                "selected_config_summary_json": str(cfg.get("summary_json")),
            },
        }
        if execution_binding is not None:
            row["execution_binding"] = execution_binding
        _validate_row_schema(row)
        rows.append(row)
    return rows


def _build_summary_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    weak_rows = [r for r in rows if r["case_type"] == "weak_stress"]
    normal_rows = [r for r in rows if r["case_type"] == "normal"]
    fallback_rows = [r for r in rows if bool(r["p6_fallback"]["fallback_triggered"])]
    no_good_rows = [r for r in rows if bool(r["p6_fallback"]["no_good_candidate"])]
    long_warn_rows = [r for r in rows if bool(r["p6_fallback"]["long_horizon_warning"])]
    selected_rows = [r for r in rows if r["p6_fallback"]["retrieval_status"] == "selected"]
    not_executed_rows = [r for r in rows if r["p6_fallback"]["retrieval_status"] == "not_executed"]
    executed_replay_rows = [r for r in rows if str(r["p6_safety_metrics"].get("status")) == "executed_artifact_replay_v1"]
    binding_missing_rows = [r for r in rows if str(r["p6_safety_metrics"].get("status")) == "artifact_binding_missing"]
    runner_print_only_rows = [r for r in rows if str(r["p6_safety_metrics"].get("status")) == "runner_invoke_print_only"]
    runner_failed_rows = [r for r in rows if str(r["p6_safety_metrics"].get("status")) == "runner_invoke_failed"]
    runner_executed_rows = [r for r in rows if str(r["p6_safety_metrics"].get("status")) == "executed_runner_invoke_v1"]
    canonical_complete_rows = [r for r in rows if bool(r["p6_safety_metrics"].get("canonical_metric_complete"))]
    canonical_missing_rows = [r for r in rows if not bool(r["p6_safety_metrics"].get("canonical_metric_complete"))]
    proxy_metric_rows = [r for r in rows if bool(r["p6_safety_metrics"].get("proxy_metric_used"))]
    weak_fallback_rows = [r for r in weak_rows if bool(r["p6_fallback"]["fallback_triggered"])]
    weak_long_warn_rows = [r for r in weak_rows if bool(r["p6_fallback"]["long_horizon_warning"])]
    safety_metric_completeness_status = (
        "complete" if len(canonical_missing_rows) == 0 else "incomplete_blocks_p6_gate"
    )
    binding_teacher_mismatch_rows = 0
    binding_inject_npz_mismatch_rows = 0
    binding_inject_step_mismatch_rows = 0
    binding_injection_record_step_mismatch_rows = 0
    binding_runtime_inject_npz_mismatch_rows = 0
    binding_runtime_inject_step_mismatch_rows = 0
    for r in rows:
        bind = r.get("execution_binding") or {}
        mapping = bind.get("row_mapping") or {}
        source_clip = str(r.get("source_clip", ""))
        target_clip = str(r.get("target_clip", ""))
        horizon_n = int(r.get("horizon_N", 0))
        expected_teacher = (Path("validate/teacher_batches") / f"{source_clip}_teacher.json").resolve()
        expected_npz = (Path("raw_data/processed_data") / f"{target_clip}.npz").resolve()
        expected_inject_step = _horizon_to_inject_at_step(horizon_n)
        got_teacher = str(mapping.get("teacher", ""))
        got_npz = str(mapping.get("turn_npz", ""))
        got_step = mapping.get("inject_at_step")
        if not got_teacher or Path(got_teacher).resolve() != expected_teacher:
            binding_teacher_mismatch_rows += 1
        if not got_npz or Path(got_npz).resolve() != expected_npz:
            binding_inject_npz_mismatch_rows += 1
        if int(got_step) != int(expected_inject_step):
            binding_inject_step_mismatch_rows += 1
        sm = r.get("p6_safety_metrics") or {}
        rb = sm.get("row_binding_audit") if isinstance(sm, dict) else {}
        if isinstance(rb, dict):
            matched = rb.get("injection_apply_record_matches_inject_at_step")
            if matched is False:
                binding_injection_record_step_mismatch_rows += 1
        runtime_inject_npz = sm.get("inject_turn_npz") if isinstance(sm, dict) else None
        runtime_inject_step = sm.get("inject_at_step") if isinstance(sm, dict) else None
        if runtime_inject_npz is not None and got_npz:
            if Path(str(runtime_inject_npz)).resolve() != Path(str(got_npz)).resolve():
                binding_runtime_inject_npz_mismatch_rows += 1
        if runtime_inject_step is not None and got_step is not None:
            if int(runtime_inject_step) != int(got_step):
                binding_runtime_inject_step_mismatch_rows += 1
    return {
        "total_rows": total,
        "normal_rows": len(normal_rows),
        "weak_stress_rows": len(weak_rows),
        "selected_rows": len(selected_rows),
        "fallback_rows": len(fallback_rows),
        "no_good_candidate_rows": len(no_good_rows),
        "not_executed_rows": len(not_executed_rows),
        "executed_artifact_replay_rows": len(executed_replay_rows),
        "artifact_binding_missing_rows": len(binding_missing_rows),
        "runner_invoke_print_only_rows": len(runner_print_only_rows),
        "runner_invoke_failed_rows": len(runner_failed_rows),
        "runner_invoke_executed_rows": len(runner_executed_rows),
        "rows_with_complete_canonical_metrics": len(canonical_complete_rows),
        "rows_with_missing_canonical_metrics": len(canonical_missing_rows),
        "rows_with_proxy_metrics": len(proxy_metric_rows),
        "safety_metric_completeness_status": safety_metric_completeness_status,
        "binding_teacher_mismatch_rows": int(binding_teacher_mismatch_rows),
        "binding_inject_npz_mismatch_rows": int(binding_inject_npz_mismatch_rows),
        "binding_inject_step_mismatch_rows": int(binding_inject_step_mismatch_rows),
        "binding_injection_record_step_mismatch_rows": int(binding_injection_record_step_mismatch_rows),
        "binding_runtime_inject_npz_mismatch_rows": int(binding_runtime_inject_npz_mismatch_rows),
        "binding_runtime_inject_step_mismatch_rows": int(binding_runtime_inject_step_mismatch_rows),
        "long_horizon_warning_rows": len(long_warn_rows),
        "weak_stress_fallback_rows": len(weak_fallback_rows),
        "weak_stress_long_horizon_warning_rows": len(weak_long_warn_rows),
    }


def _build_trial_matrix_expanded(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "trial_id": r["trial_id"],
                "source_clip": r["source_clip"],
                "target_clip": r["target_clip"],
                "horizon_N": r["horizon_N"],
                "case_type": r["case_type"],
                "retrieval_status": r["p6_fallback"]["retrieval_status"],
                "fallback_triggered": r["p6_fallback"]["fallback_triggered"],
                "long_horizon_warning": r["p6_fallback"]["long_horizon_warning"],
                "known_weak_source_risk": r["p6_fallback"]["known_weak_source_risk"],
                "execution_mode": str(r["p6_safety_metrics"].get("execution_mode", ALLOWED_EXECUTION_MODE_DRY)),
                "binding_status": str(
                    (r.get("execution_binding") or {}).get("binding_status", "not_applicable")
                ),
                "teacher": str((r.get("execution_binding") or {}).get("row_mapping", {}).get("teacher", "")),
                "turn_npz": str((r.get("execution_binding") or {}).get("row_mapping", {}).get("turn_npz", "")),
                "inject_at_step": (r.get("execution_binding") or {}).get("row_mapping", {}).get("inject_at_step"),
            }
        )
    return out


def _aggregate_metric_means(rows: List[Dict[str, Any]], metric_names: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for metric_name in metric_names:
        vals: List[float] = []
        for row in rows:
            sm = row.get("p6_safety_metrics", {})
            if not isinstance(sm, dict):
                continue
            value = sm.get(metric_name)
            if value is None:
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                vals.append(fv)
        out[metric_name] = (float(sum(vals) / float(len(vals)))) if vals else None
    return out


def _build_normal_vs_weak_comparison(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = list(REQUIRED_SAFETY_METRICS)
    normal_rows = [r for r in rows if str(r.get("case_type")) == "normal"]
    weak_rows = [r for r in rows if str(r.get("case_type")) == "weak_stress"]
    normal_means = _aggregate_metric_means(normal_rows, metric_names)
    weak_means = _aggregate_metric_means(weak_rows, metric_names)
    deltas: Dict[str, Optional[float]] = {}
    for metric_name in metric_names:
        n_val = normal_means.get(metric_name)
        w_val = weak_means.get(metric_name)
        if n_val is None or w_val is None:
            deltas[metric_name] = None
        else:
            deltas[metric_name] = float(n_val - w_val)

    incomplete_rows = [
        r
        for r in rows
        if not bool((r.get("p6_safety_metrics") or {}).get("canonical_metric_complete", False))
        or bool((r.get("p6_safety_metrics") or {}).get("proxy_metric_used", False))
    ]
    comparison_status = (
        "complete_comparison_available" if len(incomplete_rows) == 0 else "diagnostic_only_metric_incomplete"
    )
    caveat = None
    if comparison_status != "complete_comparison_available":
        caveat = (
            "One or more rows use proxy metrics or have missing canonical safety metrics; "
            "normal-vs-weak comparison is diagnostic only and cannot be used as pass gate evidence."
        )
    return {
        "metric_names": metric_names,
        "normal_aggregate_metric_means": normal_means,
        "weak_stress_aggregate_metric_means": weak_means,
        "normal_minus_weak": deltas,
        "comparison_status": comparison_status,
        "caveat": caveat,
    }


def _build_stress_differentiability_audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("target_clip", "")), int(r.get("horizon_N", 0)))
        bucket = by_key.setdefault(key, {})
        bucket[str(r.get("source_clip", ""))] = r

    pair_records: List[Dict[str, Any]] = []
    checked_pairs = 0
    identical_digest_pairs = 0
    for (target_clip, horizon_n), bucket in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        normal_row = bucket.get("Walk_F")
        weak_row = bucket.get("Walk_L_To_R")
        if not isinstance(normal_row, dict) or not isinstance(weak_row, dict):
            continue
        normal_sm = normal_row.get("p6_safety_metrics") if isinstance(normal_row, dict) else {}
        weak_sm = weak_row.get("p6_safety_metrics") if isinstance(weak_row, dict) else {}
        if not isinstance(normal_sm, dict) or not isinstance(weak_sm, dict):
            continue
        if str(normal_sm.get("status")) != "executed_runner_invoke_v1":
            continue
        if str(weak_sm.get("status")) != "executed_runner_invoke_v1":
            continue
        checked_pairs += 1
        normal_digest = str(normal_sm.get("metrics_per_step_sha256") or "")
        weak_digest = str(weak_sm.get("metrics_per_step_sha256") or "")
        digest_equal = bool(normal_digest and weak_digest and normal_digest == weak_digest)
        if digest_equal:
            identical_digest_pairs += 1
        pair_records.append(
            {
                "target_clip": target_clip,
                "horizon_N": int(horizon_n),
                "normal_trial_id": str(normal_row.get("trial_id")),
                "weak_trial_id": str(weak_row.get("trial_id")),
                "normal_metrics_per_step_sha256": normal_digest or None,
                "weak_metrics_per_step_sha256": weak_digest or None,
                "digest_equal": digest_equal,
            }
        )

    status = "not_available"
    note = "No comparable normal-vs-weak runner rows with per-step trace hash available."
    if checked_pairs > 0:
        if identical_digest_pairs > 0:
            status = "blocked_identical_trajectory_hash"
            note = (
                "At least one normal-vs-weak source swap pair has identical metrics_per_step hash. "
                "Stress differentiability remains blocked pending row-binding/windowed-trace audit."
            )
        else:
            status = "differentiable_trace_observed"
            note = "No identical metrics_per_step hash found in comparable normal-vs-weak source swap pairs."
    return {
        "checked_pairs": int(checked_pairs),
        "identical_digest_pairs": int(identical_digest_pairs),
        "status": status,
        "note": note,
        "pairs": pair_records,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_summary_markdown(
    *,
    summary_json_path: Path,
    matrix_path: Path,
    rows_path: Path,
    summary: Dict[str, Any],
) -> str:
    counts = summary["summary_counts"]
    comparison = summary.get("normal_vs_weak_comparison", {})
    exec_mode = str(summary.get("execution_mode", ALLOWED_EXECUTION_MODE_DRY))
    execution_status = str(summary.get("execution_status", "dry_run_only"))
    replay_scope = str(summary.get("inputs", {}).get("artifact_replay_scope", ALLOWED_ARTIFACT_REPLAY_SCOPE_SMOKE))
    if exec_mode == ALLOWED_EXECUTION_MODE_REPLAY:
        status_line = (
            "Status: artifact-replay full-matrix injected smoke (read-only existing artifacts, not pass gate)"
            if replay_scope == ALLOWED_ARTIFACT_REPLAY_SCOPE_FULL
            else "Status: artifact-replay smoke execution (read-only existing artifacts, not pass gate)"
        )
        mode_lines = [
            "- Execution mode is artifact_replay.",
            "- No new evaluator was called; metrics are replayed from existing substrate artifacts.",
            "- This is injected smoke / not pass gate.",
        ]
    elif exec_mode == ALLOWED_EXECUTION_MODE_RUNNER:
        if counts.get("runner_invoke_executed_rows", 0) > 0:
            status_line = "Status: runner-invoke full-matrix injected smoke executed (not pass gate)"
            mode_lines = [
                "- Execution mode is runner_invoke.",
                "- Subprocess execution is enabled for fixed full-matrix rows.",
                "- This is execution coverage + metric completeness check only, not pass gate.",
            ]
        else:
            status_line = "Status: runner-invoke print-only scaffold (no subprocess execution)"
            mode_lines = [
                "- Execution mode is runner_invoke.",
                "- Commands are constructed and printed in artifacts only (--print-commands-only).",
                "- No runner subprocess was executed in this phase.",
            ]
    else:
        status_line = "Status: scaffold/dry-run only"
        mode_lines = [
            "- This is scaffold/dry-run only.",
            "- No evaluator was called.",
            "- This is not a pass gate result.",
        ]
    lines = [
        "# P6 Synthetic Boundary Eval Scaffold Summary",
        "",
        status_line,
        "",
        *mode_lines,
        "- Walk_L_To_R weak-source risk is included as stress case.",
        "",
        "## Artifacts",
        f"- Summary JSON: {summary_json_path}",
        f"- Trial matrix expanded: {matrix_path}",
        f"- Dry-run rows: {rows_path}",
        "",
        "## Counts",
        f"- execution_status: {execution_status}",
        f"- total_rows: {counts['total_rows']}",
        f"- normal_rows: {counts['normal_rows']}",
        f"- weak_stress_rows: {counts['weak_stress_rows']}",
        f"- fallback_rows: {counts['fallback_rows']}",
        f"- executed_artifact_replay_rows: {counts['executed_artifact_replay_rows']}",
        f"- artifact_binding_missing_rows: {counts['artifact_binding_missing_rows']}",
        f"- runner_invoke_print_only_rows: {counts['runner_invoke_print_only_rows']}",
        f"- runner_invoke_failed_rows: {counts['runner_invoke_failed_rows']}",
        f"- runner_invoke_executed_rows: {counts['runner_invoke_executed_rows']}",
        f"- rows_with_complete_canonical_metrics: {counts['rows_with_complete_canonical_metrics']}",
        f"- rows_with_missing_canonical_metrics: {counts['rows_with_missing_canonical_metrics']}",
        f"- rows_with_proxy_metrics: {counts['rows_with_proxy_metrics']}",
        f"- safety_metric_completeness_status: {counts['safety_metric_completeness_status']}",
        f"- long_horizon_warning_rows: {counts['long_horizon_warning_rows']}",
        f"- weak_stress_fallback_rows: {counts['weak_stress_fallback_rows']}",
        f"- weak_stress_long_horizon_warning_rows: {counts['weak_stress_long_horizon_warning_rows']}",
        f"- binding_teacher_mismatch_rows: {counts['binding_teacher_mismatch_rows']}",
        f"- binding_inject_npz_mismatch_rows: {counts['binding_inject_npz_mismatch_rows']}",
        f"- binding_inject_step_mismatch_rows: {counts['binding_inject_step_mismatch_rows']}",
        f"- binding_injection_record_step_mismatch_rows: {counts['binding_injection_record_step_mismatch_rows']}",
        f"- binding_runtime_inject_npz_mismatch_rows: {counts['binding_runtime_inject_npz_mismatch_rows']}",
        f"- binding_runtime_inject_step_mismatch_rows: {counts['binding_runtime_inject_step_mismatch_rows']}",
        "",
        "## Normal vs Weak Comparison",
        f"- comparison_status: {comparison.get('comparison_status')}",
        f"- caveat: {comparison.get('caveat')}",
        f"- normal_means: {json.dumps(comparison.get('normal_aggregate_metric_means', {}), ensure_ascii=False)}",
        f"- weak_means: {json.dumps(comparison.get('weak_stress_aggregate_metric_means', {}), ensure_ascii=False)}",
        f"- normal_minus_weak: {json.dumps(comparison.get('normal_minus_weak', {}), ensure_ascii=False)}",
    ]
    stress = summary.get("stress_differentiability_audit", {})
    lines.extend(
        [
            "",
            "## Stress Differentiability Audit",
            f"- status: {stress.get('status')}",
            f"- checked_pairs: {stress.get('checked_pairs')}",
            f"- identical_digest_pairs: {stress.get('identical_digest_pairs')}",
            f"- note: {stress.get('note')}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    execution_mode = _validate_execution_gate(args)
    execute_artifact_replay = execution_mode == ALLOWED_EXECUTION_MODE_REPLAY
    execute_runner_invoke = execution_mode == ALLOWED_EXECUTION_MODE_RUNNER
    out_dir = args.out_dir.resolve()

    z_features_path = _ensure_file(args.z_features)
    p4_summary_path = _ensure_file(args.p4_alt_sweep_summary)
    walk_analysis_path = _ensure_file(args.walk_l_to_r_analysis)
    substrate_path = _ensure_file(args.substrate_sweep_config, optional=bool(args.allow_missing_substrate))
    substrate_root = substrate_path.resolve().parent

    trial_matrix_input_obj: Any = None
    trial_matrix_input_path: Optional[Path] = None
    if args.trial_matrix is not None:
        trial_matrix_input_path = _ensure_file(args.trial_matrix)
        trial_matrix_input_obj = _load_json(trial_matrix_input_path)

    p4_summary = _validate_dict(_load_json(p4_summary_path), name="p4-alt sweep summary")
    walk_analysis = _validate_dict(_load_json(walk_analysis_path), name="walk-l-to-r analysis")
    substrate = None
    if substrate_path.exists():
        substrate = _validate_dict(_load_json(substrate_path), name="substrate sweep config")
    elif not args.allow_missing_substrate:
        _fatal("Substrate config missing and --allow-missing-substrate not provided")

    if execute_artifact_replay or execute_runner_invoke:
        expected_substrate = DEFAULT_SUBSTRATE_SWEEP_CONFIG.resolve()
        if substrate_path.resolve() != expected_substrate:
            _fatal(
                "execution mode only allows the canonical substrate sweep config path: "
                f"{expected_substrate}"
            )
        if substrate is None:
            _fatal("execution mode requires existing substrate config.")
        _ensure_file(substrate_root / "contract_check_report.json")
        _ensure_file(substrate_root / "p2_entry_probe_check_report.json")

    p4_configs = _validate_p4_summary(p4_summary, p4_summary_path)
    z_contract = _extract_z_feature_contract(z_features_path)
    if execute_artifact_replay or execute_runner_invoke:
        trial_matrix = _build_execution_trial_matrix(
            scope=str(args.artifact_replay_scope),
            trial_matrix_obj=trial_matrix_input_obj,
            horizons=args.horizon_values,
            include_weak_stress=bool(args.include_weak_stress),
            execution_mode=execution_mode,
            print_commands_only=bool(args.print_commands_only),
        )
    else:
        trial_matrix = _expand_trial_matrix(trial_matrix_input_obj, args.horizon_values, bool(args.include_weak_stress))
    if not trial_matrix:
        _fatal("Expanded trial matrix is empty")

    cfg_by_horizon: Dict[int, Dict[str, Any]] = {}
    per_cfg_summary: Dict[str, Dict[str, Any]] = {}
    for h in sorted({int(t["horizon_N"]) for t in trial_matrix}):
        cfg = _select_config_for_horizon(p4_configs, h)
        cfg_by_horizon[h] = cfg
        cfg_id = str(cfg["config_id"])
        if cfg_id not in per_cfg_summary:
            per_cfg_summary[cfg_id] = _load_per_config_summary(cfg)

    rows = _build_rows(
        trial_matrix=trial_matrix,
        cfg_by_horizon=cfg_by_horizon,
        per_cfg_summary=per_cfg_summary,
        z_contract=z_contract,
        weak_analysis=walk_analysis,
        execution_mode=execution_mode,
        substrate_root=substrate_root,
        substrate_cfg=substrate,
        out_dir=out_dir,
        print_commands_only=bool(args.print_commands_only),
        runner_timeout_s=int(args.runner_timeout_s),
    )
    trial_matrix_expanded = _build_trial_matrix_expanded(rows)
    counts = _build_summary_counts(rows)
    normal_vs_weak_comparison = _build_normal_vs_weak_comparison(rows)
    stress_audit = _build_stress_differentiability_audit(rows)

    replay_scope = str(args.artifact_replay_scope)
    replay_exec_status = (
        "executed_artifact_replay_injected_full_matrix_smoke_not_pass_gate"
        if replay_scope == ALLOWED_ARTIFACT_REPLAY_SCOPE_FULL
        else "executed_artifact_replay_smoke_not_pass_gate"
    )
    runner_exec_status = "runner_invoke_print_only_scaffold"
    if execute_runner_invoke and not bool(args.print_commands_only):
        runner_exec_status = "executed_runner_injected_full_matrix_smoke_not_pass_gate"
    runner_real_exec = bool(execute_runner_invoke and not bool(args.print_commands_only))
    runner_failed_n = int(counts.get("runner_invoke_failed_rows", 0))
    summary = {
        "scope": {
            "tool": "run_action_handoff_p6_synthetic_boundary_eval",
            "status": (
                "planning_scaffold_dry_run_only"
                if execution_mode == ALLOWED_EXECUTION_MODE_DRY
                else "artifact_replay_injected_smoke_execution"
                if execute_artifact_replay
                else "runner_invoke_injected_smoke_execution"
                if runner_real_exec
                else "runner_invoke_print_only_scaffold"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scaffold_phase": "dry_run_to_minimal_run_placeholder"
            if execution_mode == ALLOWED_EXECUTION_MODE_DRY
            else "artifact_replay_injected_smoke_execution"
            if execute_artifact_replay
            else "runner_invoke_injected_smoke_execution"
            if runner_real_exec
            else "runner_invoke_print_only",
        },
        "inputs": {
            "z_features": str(z_features_path),
            "p4_alt_sweep_summary": str(p4_summary_path),
            "walk_l_to_r_analysis": str(walk_analysis_path),
            "substrate_sweep_config": str(substrate_path),
            "trial_matrix_input": str(trial_matrix_input_path) if trial_matrix_input_path is not None else None,
            "horizons": list(args.horizon_values),
            "include_weak_stress": bool(args.include_weak_stress),
            "dry_run": bool(args.dry_run),
            "allow_execute": bool(args.allow_execute),
            "execution_mode": str(args.execution_mode),
            "artifact_replay_scope": str(args.artifact_replay_scope),
            "print_commands_only": bool(args.print_commands_only),
            "runner_timeout_s": int(args.runner_timeout_s),
            "allow_missing_substrate": bool(args.allow_missing_substrate),
        },
        "substrate": {
            "loaded": bool(substrate is not None),
            "path": str(substrate_path),
            "keys": sorted(substrate.keys()) if isinstance(substrate, dict) else [],
        },
        "z_feature_contract": {
            "repr_dim": z_contract["repr_dim"],
            "dtype": z_contract["dtype"],
            "device": z_contract["device"],
            "clip_count": len(z_contract["per_clip"]),
            "clip_order": z_contract["clip_order"],
        },
        "trial_matrix": trial_matrix_expanded,
        "rows": rows,
        "summary_counts": counts,
        "known_risks": [
            "P6 full-matrix injected smoke only; not pass gate.",
            "Walk_L_To_R is treated as known weak-source stress risk.",
        ]
        + (["semantic_proxy_binding_used_in_smoke_v1"] if execute_artifact_replay else [])
        + (["runner_invoke_not_executed_print_only_scaffold"] if execute_runner_invoke and not runner_real_exec else []),
        "decision_boundary": {
            "status": (
                "not_evaluated_dry_run"
                if execution_mode == ALLOWED_EXECUTION_MODE_DRY
                else "full_matrix_smoke_not_pass_gate"
                if execute_artifact_replay
                else "runner_invoke_print_only"
                if not runner_real_exec
                else "full_matrix_smoke_not_pass_gate"
            ),
            "note": (
                "Decision boundary is scaffolded only. Concentration of failures on weak stress pairs "
                "must be reviewed after minimal evaluator wiring."
            )
            if execution_mode == ALLOWED_EXECUTION_MODE_DRY
            else (
                "artifact_replay full-matrix injected smoke only: validates schema/report plumbing with existing substrate artifacts; "
                "does not constitute pass gate."
            )
            if execute_artifact_replay
            else (
                "runner_invoke print-only scaffold: command construction validated; no subprocess execution performed."
                if not runner_real_exec
                else "runner_invoke full-matrix injected smoke executed with injection args; this is execution coverage only and not pass gate."
            ),
        },
        "execution_mode": execution_mode,
        "execution_status": (
            "dry_run_only"
            if execution_mode == ALLOWED_EXECUTION_MODE_DRY
            else replay_exec_status
            if execute_artifact_replay
            else runner_exec_status
        ),
    }
    summary["normal_vs_weak_comparison"] = normal_vs_weak_comparison
    summary["stress_differentiability_audit"] = stress_audit
    if str(counts.get("safety_metric_completeness_status")) == "incomplete_blocks_p6_gate":
        summary["known_risks"].append("safety_metric_completeness_failed_p6_gate_blocked")
        summary["decision_boundary"]["note"] = (
            str(summary["decision_boundary"]["note"])
            + " Safety metric completeness failed; P6 gate blocked."
        )
    binding_block_reasons: List[str] = []
    if int(counts.get("binding_teacher_mismatch_rows", 0)) > 0:
        binding_block_reasons.append("teacher_mismatch")
    if int(counts.get("binding_inject_npz_mismatch_rows", 0)) > 0:
        binding_block_reasons.append("inject_npz_mismatch")
    if int(counts.get("binding_inject_step_mismatch_rows", 0)) > 0:
        binding_block_reasons.append("inject_step_mismatch")
    if int(counts.get("binding_injection_record_step_mismatch_rows", 0)) > 0:
        binding_block_reasons.append("injection_record_step_mismatch")
    if int(counts.get("binding_runtime_inject_npz_mismatch_rows", 0)) > 0:
        binding_block_reasons.append("runtime_inject_npz_mismatch")
    if int(counts.get("binding_runtime_inject_step_mismatch_rows", 0)) > 0:
        binding_block_reasons.append("runtime_inject_step_mismatch")
    if binding_block_reasons:
        summary["known_risks"].append("execution_binding_audit_blocked")
        summary["decision_boundary"]["note"] = (
            str(summary["decision_boundary"]["note"])
            + " Execution binding audit blocked: "
            + ",".join(binding_block_reasons)
            + "."
        )
    if str(stress_audit.get("status")) == "blocked_identical_trajectory_hash":
        summary["known_risks"].append("stress_differentiability_blocked_identical_trace_hash")
        summary["decision_boundary"]["note"] = (
            str(summary["decision_boundary"]["note"])
            + " Stress differentiability blocked by identical normal-vs-weak per-step trace hash."
        )
    if execute_runner_invoke:
        runner_cmd_rows = []
        for r in rows:
            bind = r.get("execution_binding") or {}
            if str(bind.get("mode")) != ALLOWED_EXECUTION_MODE_RUNNER:
                continue
            runner_cmd_rows.append(
                {
                    "trial_id": r["trial_id"],
                    "binding_status": bind.get("binding_status"),
                    "command_shell": bind.get("command_shell"),
                    "trial_out_dir": bind.get("trial_out_dir"),
                    "timeout_s": bind.get("timeout_s"),
                    "error_type": bind.get("error_type"),
                    "error_message": bind.get("error_message"),
                }
            )
        summary["runner_invoke_plan"] = {
            "print_commands_only": bool(args.print_commands_only),
            "planned_rows": int(len(runner_cmd_rows)),
            "rows": runner_cmd_rows,
        }
    if int(counts.get("artifact_binding_missing_rows", 0)) > 0:
        summary["known_risks"].append("artifact_binding_missing_present_in_replay_matrix")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = out_dir / "p6_synthetic_boundary_eval_summary.json"
    summary_md_path = out_dir / "p6_synthetic_boundary_eval_summary.md"
    matrix_path = out_dir / "p6_trial_matrix_expanded.json"
    rows_path = out_dir / "p6_dryrun_rows.json"

    _write_json(summary_json_path, summary)
    _write_json(matrix_path, trial_matrix_expanded)
    _write_json(rows_path, rows)
    summary_md_path.write_text(
        _build_summary_markdown(
            summary_json_path=summary_json_path,
            matrix_path=matrix_path,
            rows_path=rows_path,
            summary=summary,
        ),
        encoding="utf-8",
    )

    print(f"[p6-scaffold] out_dir={out_dir}")
    print(f"[p6-scaffold] rows={len(rows)}")
    print(f"[p6-scaffold] execution_status={summary['execution_status']}")
    print(f"[p6-scaffold] summary_json={summary_json_path}")


if __name__ == "__main__":
    main()
