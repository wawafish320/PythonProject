#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_cp015_tailk7_direct_dependency_asymmetry import (  # noqa: E402
    CANDIDATE_SPECS,
    _load_base_eval_meta,
)
from train.validate.run_freerun_cycles import FreeRunCycleRunner, _load_json, _resolve_npz_path  # noqa: E402


RUN_DATE = "20260407"
DEFAULT_OUT_ROOT = ROOT / "debug_output" / f"_tmp_cp015_tailk7_nonplan_direct_whitebox_{RUN_DATE}"
DEFAULT_SUMMARY_JSON = DEFAULT_OUT_ROOT / "summary.json"
DEFAULT_SUMMARY_MD = DEFAULT_OUT_ROOT / "summary.md"
DEFAULT_AUDIT_LOG = DEFAULT_OUT_ROOT / "audit.log"
DEFAULT_PYTHON = sys.executable or "python3"
DEFAULT_MODES: Tuple[str, ...] = ("teacher_x_gt", "freerun")
DEFAULT_OVERRIDES: Tuple[Tuple[str, str], ...] = (
    ("model", "model"),
    ("zero", "zero"),
)


@dataclass(frozen=True)
class ArtifactSpec:
    candidate: str
    eval_mode_key: str
    eval_mode: str
    plan_source: str
    meas_source: str
    path: Path
    origin: str


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: Iterable[Any]) -> float:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _pctl(values: Iterable[Any], q: float) -> float:
    arr = np.asarray([_safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float("nan")
    return float(np.percentile(arr, float(q)))


def _fmt(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def _ratio(value: Any, ref: Any) -> float:
    v = _safe_float(value)
    r = _safe_float(ref)
    if (not math.isfinite(v)) or (not math.isfinite(r)) or abs(r) <= 1e-12:
        return float("nan")
    return float(v / r)


def _parse_candidate_names(raw: str) -> List[str]:
    if not str(raw or "").strip():
        return [
            "baseline_replace",
            "coadapt_4x_directonly_calibration_240",
            "coadapt_4x_direct_plus_plan_ownership_240_noeventclock",
        ]
    items = [part.strip() for part in str(raw).split(",") if part.strip()]
    unknown = [name for name in items if name not in CANDIDATE_SPECS]
    if unknown:
        raise SystemExit(f"[FATAL] unknown candidates: {unknown}")
    return items


def _parse_modes(raw: str) -> List[str]:
    if not str(raw or "").strip():
        return list(DEFAULT_MODES)
    items = [part.strip() for part in str(raw).split(",") if part.strip()]
    unknown = [name for name in items if name not in DEFAULT_MODES]
    if unknown:
        raise SystemExit(f"[FATAL] unknown modes: {unknown}")
    return items


def _parse_overrides(raw: str) -> List[Tuple[str, str]]:
    if not str(raw or "").strip():
        return list(DEFAULT_OVERRIDES)
    out: List[Tuple[str, str]] = []
    for chunk in str(raw).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "/" not in token:
            raise SystemExit(f"[FATAL] invalid override token {token!r}; expected plan/meas")
        plan_source, meas_source = [part.strip() for part in token.split("/", 1)]
        out.append((plan_source, meas_source))
    return out


def _label_mode(mode_key: str) -> str:
    return "teacher-conditioned" if str(mode_key) == "teacher_x_gt" else "freerun"


def _slug(plan_source: str, meas_source: str) -> str:
    return f"plan_{plan_source}__meas_{meas_source}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact_has_arm_probe(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    probe = payload.get("direct_arm_probe", None)
    return bool(isinstance(probe, dict) and probe.get("enabled") and probe.get("steps"))


def _build_eval_command(
    *,
    python_exe: str,
    meta: Mapping[str, Any],
    out_dir: Path,
    plan_source: str,
    meas_source: str,
    mode: str,
) -> List[str]:
    cmd = [
        python_exe,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        str(meta["teacher_json"]),
        "--model",
        str(meta["model"]),
        "--rounds",
        str(int(meta["rounds"])),
        "--depth",
        str(int(meta["depth"])),
        "--time-index-mode",
        str(meta["time_index_mode"]),
        "--event_clock",
        str(meta["event_clock"]),
        "--phase_reset_source",
        str(meta["phase_reset_source"]),
        "--contacts_meas_source",
        str(meta["contacts_meas_source"]),
        "--direct_pose_plan_source",
        str(plan_source),
        "--direct_pose_meas_source",
        str(meas_source),
        "--lambda_fusion_apply",
        "--log_contacts",
        "--export_joint_direct_geolocal_series",
        "--export_direct_arm_probe",
        "--out",
        str(out_dir),
        "--force",
    ]
    if meta.get("bundle"):
        cmd.extend(["--bundle", str(meta["bundle"])])
    if meta.get("pretrain_template"):
        cmd.extend(["--pretrain-template", str(meta["pretrain_template"])])
    if meta.get("encoder_bundle"):
        cmd.extend(["--encoder-bundle", str(meta["encoder_bundle"])])
    if str(mode) == "teacher_x_gt":
        cmd.append("--freerun_x_gt")
    return cmd


def _run_command(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(str(part) for part in cmd) + "\n")
        log_file.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        log_file.write(f"[exit={proc.returncode}]\n\n")
        log_file.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit={proc.returncode}: {' '.join(str(part) for part in cmd)}")


def _make_runner_args(meta: Mapping[str, Any], *, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        model=str(meta["model"]),
        device=str(device),
        bundle=str(meta["bundle"]),
        pretrain_template=str(meta["pretrain_template"]),
        encoder_bundle=str(meta["encoder_bundle"]) if meta.get("encoder_bundle") is not None else "",
        num_heads=4,
        dropout=0.1,
        context_len=16,
        depth=int(meta["depth"]),
        so3_corr_apply=False,
        so3_corr_max_deg=20.0,
        lambda_fusion_apply=False,
    )


def _infer_direct_layout(meta: Mapping[str, Any], *, device: str) -> Dict[str, Any]:
    teacher_payload = _load_json(Path(str(meta["teacher_json"])))
    teacher_block = teacher_payload.get("teacher")
    if not isinstance(teacher_block, dict):
        raise RuntimeError(f"missing teacher payload in {meta['teacher_json']}")
    state_arr = np.asarray(teacher_block.get("state_norm"), dtype=np.float32)
    if state_arr.ndim != 2:
        raise RuntimeError(f"invalid teacher state shape in {meta['teacher_json']}")
    clip_name = str(teacher_payload.get("clip") or Path(str(meta["teacher_json"])).stem.replace("_teacher", ""))
    npz_root = ROOT / "raw_data" / "processed_data"
    npz_path = _resolve_npz_path(clip_name, teacher_payload.get("source_json"), npz_root)
    runner = FreeRunCycleRunner(_make_runner_args(meta, device=device))
    ds = runner._build_dataset(npz_path, seq_len=int(state_arr.shape[0]))
    runner._ensure_model_ready(ds)
    model = runner.model
    if model is None:
        raise RuntimeError("failed to build runtime model for layout inference")
    feat_source = str(getattr(model, "direct_pose_feat_source", "cond") or "cond").lower().strip()
    cond_dim = int(getattr(model, "cond_dim", 0) or 0)
    hidden_dim = int(getattr(model, "hidden_dim", 0) or 0)
    contact_dim = int(getattr(model, "contact_dim", 0) or 0)
    time_dim = int(getattr(model, "direct_pose_time_pe_dim", 0) or 0)
    meas_mode = str(getattr(model, "direct_pose_meas_mode", "concat") or "concat").lower().strip()
    if feat_source in ("hidden", "hidden_pre"):
        base_dim = hidden_dim
    elif feat_source in ("cond+hidden", "cond+hidden_pre"):
        base_dim = cond_dim + hidden_dim
    else:
        base_dim = cond_dim
    direct_feat_dim = int(base_dim + time_dim)
    plan_dim = int(contact_dim)
    meas_dim = int(contact_dim if meas_mode == "concat" else 0)
    return {
        "direct_pose_feat_source": feat_source,
        "direct_pose_meas_mode": meas_mode,
        "direct_pose_split_enable": bool(getattr(model, "direct_pose_split_enable", False)),
        "direct_pose_arm_split_enable": bool(getattr(model, "direct_pose_arm_split_enable", False)),
        "direct_pose_factorized_readout_enable": bool(getattr(model, "direct_pose_factorized_readout_enable", False)),
        "direct_pose_input_adapter_enable": bool(getattr(model, "direct_pose_input_adapter_enable", False)),
        "direct_pose_factorized_input_adapter_enable": bool(
            getattr(model, "direct_pose_factorized_input_adapter_enable", False)
        ),
        "cond_dim": int(cond_dim),
        "hidden_dim": int(hidden_dim),
        "contact_dim": int(contact_dim),
        "direct_pose_time_pe_dim": int(time_dim),
        "direct_feat_dim": int(direct_feat_dim),
        "plan_dim": int(plan_dim),
        "meas_dim": int(meas_dim),
    }


def _vec_rms(vec: np.ndarray) -> float:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return float("nan")
    return float(np.linalg.norm(arr) / math.sqrt(float(arr.size)))


def _vec_std(vec: np.ndarray) -> float:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return float("nan")
    return float(np.std(arr))


def _extract_direct_error_map(payload: Mapping[str, Any]) -> Dict[int, float]:
    per = payload.get("per_step_direct_geolocal_deg", None)
    if not isinstance(per, dict):
        return {}
    rows = per.get("DirectGeoLocalDeg", None)
    if not isinstance(rows, list):
        return {}
    root_idx = int(per.get("root_idx", 0) or 0)
    out: Dict[int, float] = {}
    for step_i, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        vals = []
        for joint_idx, value in enumerate(row):
            if int(joint_idx) == int(root_idx):
                continue
            fv = _safe_float(value)
            if math.isfinite(fv):
                vals.append(fv)
        out[int(step_i)] = _mean(vals)
    return out


def _probe_vector_map(path: Path, *, layout: Mapping[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    probe = payload.get("direct_arm_probe", None)
    if not isinstance(probe, dict) or not probe.get("enabled"):
        raise RuntimeError(f"{path}: missing direct_arm_probe export")
    steps = probe.get("steps", None)
    if not isinstance(steps, list) or not steps:
        raise RuntimeError(f"{path}: empty direct_arm_probe.steps")
    direct_error_map = _extract_direct_error_map(payload)
    direct_feat_dim = int(layout["direct_feat_dim"])
    plan_dim = int(layout["plan_dim"])
    meas_dim = int(layout["meas_dim"])
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for ent in steps:
        if not isinstance(ent, dict):
            continue
        features = ent.get("features", None)
        if not isinstance(features, dict):
            continue
        x_raw = features.get("direct_in", None)
        h_raw = features.get("trunk_hidden", None)
        if not isinstance(x_raw, list) or not isinstance(h_raw, list):
            continue
        x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
        h = np.asarray(h_raw, dtype=np.float64).reshape(-1)
        feat_dim_eff = int(direct_feat_dim)
        if int(x.size) != int(direct_feat_dim + plan_dim + meas_dim):
            feat_dim_eff = max(0, int(x.size) - int(plan_dim) - int(meas_dim))
        feat = x[:feat_dim_eff]
        key = (int(ent.get("cycle", -1) or -1), int(ent.get("step_in_cycle", -1) or -1))
        out[key] = {
            "direct_feat_vec": feat,
            "direct_in_vec": x,
            "trunk_hidden_vec": h,
            "direct_error_mean": _safe_float(direct_error_map.get(int(ent.get("step", -1) or -1))),
        }
    return out


def _analyze_artifact(path: Path, *, layout: Mapping[str, Any]) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    probe = payload.get("direct_arm_probe", None)
    if not isinstance(probe, dict) or not probe.get("enabled"):
        raise RuntimeError(f"{path}: missing direct_arm_probe export")
    steps = probe.get("steps", None)
    if not isinstance(steps, list) or not steps:
        raise RuntimeError(f"{path}: empty direct_arm_probe.steps")

    direct_error_map = _extract_direct_error_map(payload)
    direct_feat_dim = int(layout["direct_feat_dim"])
    plan_dim = int(layout["plan_dim"])
    meas_dim = int(layout["meas_dim"])
    per_step_rows: List[Dict[str, Any]] = []
    prev_feat: Optional[np.ndarray] = None
    prev_in: Optional[np.ndarray] = None
    prev_hidden: Optional[np.ndarray] = None

    for ent in steps:
        if not isinstance(ent, dict):
            continue
        features = ent.get("features", None)
        if not isinstance(features, dict):
            continue
        x_raw = features.get("direct_in", None)
        h_raw = features.get("trunk_hidden", None)
        if not isinstance(x_raw, list) or not isinstance(h_raw, list):
            continue
        x = np.asarray(x_raw, dtype=np.float64).reshape(-1)
        h = np.asarray(h_raw, dtype=np.float64).reshape(-1)
        feat_dim_eff = int(direct_feat_dim)
        if int(x.size) != int(direct_feat_dim + plan_dim + meas_dim):
            feat_dim_eff = max(0, int(x.size) - int(plan_dim) - int(meas_dim))
        feat = x[:feat_dim_eff]
        plan = x[feat_dim_eff : feat_dim_eff + plan_dim]
        meas = x[feat_dim_eff + plan_dim : feat_dim_eff + plan_dim + meas_dim]
        row = {
            "step": int(ent.get("step", -1) or -1),
            "cycle": int(ent.get("cycle", -1) or -1),
            "step_in_cycle": int(ent.get("step_in_cycle", -1) or -1),
            "direct_error_mean": _safe_float(direct_error_map.get(int(ent.get("step", -1) or -1))),
            "direct_feat_dim": int(feat.size),
            "plan_dim": int(plan.size),
            "meas_dim": int(meas.size),
            "trunk_hidden_dim": int(h.size),
            "direct_feat_rms": _vec_rms(feat),
            "direct_feat_std": _vec_std(feat),
            "plan_rms": _vec_rms(plan),
            "plan_std": _vec_std(plan),
            "meas_rms": _vec_rms(meas),
            "meas_std": _vec_std(meas),
            "direct_in_rms": _vec_rms(x),
            "direct_in_std": _vec_std(x),
            "trunk_hidden_rms": _vec_rms(h),
            "trunk_hidden_std": _vec_std(h),
            "direct_feat_step_delta": _vec_rms(feat - prev_feat) if prev_feat is not None and prev_feat.shape == feat.shape else float("nan"),
            "direct_in_step_delta": _vec_rms(x - prev_in) if prev_in is not None and prev_in.shape == x.shape else float("nan"),
            "trunk_hidden_step_delta": _vec_rms(h - prev_hidden) if prev_hidden is not None and prev_hidden.shape == h.shape else float("nan"),
        }
        per_step_rows.append(row)
        prev_feat = feat.copy()
        prev_in = x.copy()
        prev_hidden = h.copy()

    summary = {
        "steps": int(len(per_step_rows)),
        "direct_error_mean": _mean(row["direct_error_mean"] for row in per_step_rows),
        "direct_error_p95": _pctl((row["direct_error_mean"] for row in per_step_rows), 95),
        "direct_feat_rms_mean": _mean(row["direct_feat_rms"] for row in per_step_rows),
        "direct_feat_rms_p95": _pctl((row["direct_feat_rms"] for row in per_step_rows), 95),
        "direct_feat_std_mean": _mean(row["direct_feat_std"] for row in per_step_rows),
        "direct_feat_std_p95": _pctl((row["direct_feat_std"] for row in per_step_rows), 95),
        "plan_rms_mean": _mean(row["plan_rms"] for row in per_step_rows),
        "meas_rms_mean": _mean(row["meas_rms"] for row in per_step_rows),
        "direct_in_rms_mean": _mean(row["direct_in_rms"] for row in per_step_rows),
        "direct_in_std_mean": _mean(row["direct_in_std"] for row in per_step_rows),
        "trunk_hidden_rms_mean": _mean(row["trunk_hidden_rms"] for row in per_step_rows),
        "trunk_hidden_rms_p95": _pctl((row["trunk_hidden_rms"] for row in per_step_rows), 95),
        "trunk_hidden_std_mean": _mean(row["trunk_hidden_std"] for row in per_step_rows),
        "trunk_hidden_std_p95": _pctl((row["trunk_hidden_std"] for row in per_step_rows), 95),
        "direct_feat_step_delta_mean": _mean(row["direct_feat_step_delta"] for row in per_step_rows),
        "direct_in_step_delta_mean": _mean(row["direct_in_step_delta"] for row in per_step_rows),
        "trunk_hidden_step_delta_mean": _mean(row["trunk_hidden_step_delta"] for row in per_step_rows),
        "trunk_hidden_step_delta_p95": _pctl((row["trunk_hidden_step_delta"] for row in per_step_rows), 95),
        "cycle_mean_direct_error": {
            str(cycle): _mean(row["direct_error_mean"] for row in per_step_rows if int(row["cycle"]) == int(cycle))
            for cycle in sorted({int(row["cycle"]) for row in per_step_rows})
        },
        "sic_bucket_direct_error": {
            "sic0_10": _mean(row["direct_error_mean"] for row in per_step_rows if 0 <= int(row["step_in_cycle"]) <= 10),
            "sic11_21": _mean(row["direct_error_mean"] for row in per_step_rows if 11 <= int(row["step_in_cycle"]) <= 21),
            "sic22_43": _mean(row["direct_error_mean"] for row in per_step_rows if 22 <= int(row["step_in_cycle"]) <= 43),
        },
    }
    return {
        "artifact_path": str(path.resolve()),
        "summary": summary,
        "per_step": per_step_rows,
    }


def _paired_drift(
    teacher_path: Path,
    freerun_path: Path,
    *,
    layout: Mapping[str, Any],
) -> Dict[str, Any]:
    teacher_map = _probe_vector_map(teacher_path, layout=layout)
    freerun_map = _probe_vector_map(freerun_path, layout=layout)
    feat_drift: List[float] = []
    din_drift: List[float] = []
    hidden_drift: List[float] = []
    err_delta: List[float] = []
    matched = 0
    for key, row in freerun_map.items():
        ref = teacher_map.get(key, None)
        if ref is None:
            continue
        matched += 1
        for lhs_key, rhs_key, acc in (
            ("direct_feat_vec", "direct_feat_vec", feat_drift),
            ("direct_in_vec", "direct_in_vec", din_drift),
            ("trunk_hidden_vec", "trunk_hidden_vec", hidden_drift),
        ):
            lhs = row[lhs_key]
            rhs = ref[rhs_key]
            if isinstance(lhs, np.ndarray) and isinstance(rhs, np.ndarray) and lhs.shape == rhs.shape:
                acc.append(_vec_rms(lhs - rhs))
        delta_err = _safe_float(row["direct_error_mean"]) - _safe_float(ref["direct_error_mean"])
        if math.isfinite(delta_err):
            err_delta.append(delta_err)
    return {
        "matched_steps": int(matched),
        "direct_feat_rms_abs_drift_mean": _mean(feat_drift),
        "direct_in_rms_abs_drift_mean": _mean(din_drift),
        "trunk_hidden_rms_abs_drift_mean": _mean(hidden_drift),
        "direct_error_delta_mean": _mean(err_delta),
    }


def _render_artifact_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate / run | self-contained? | event_clock enabled? | eval mode | override mode | eval artifact path |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {'yes' if row['self_contained'] else 'no'} | "
            f"{'yes' if row['event_clock_enabled'] else 'no'} | {row['eval_mode']} | "
            f"{row['plan_source']}/{row['meas_source']} | {row['artifact_path']} |"
        )
    return lines


def _render_whitebox_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | eval mode | override | DirectGeoLocalDeg | direct_feat rms/std | trunk_hidden rms/std | step Δ trunk | dyn-range shrink vs baseline | step instability vs baseline |",
        "|---|---|---|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['eval_mode']} | {row['plan_source']}/{row['meas_source']} | "
            f"{_fmt(row['direct_error_mean'])} | "
            f"{_fmt(row['direct_feat_rms_mean'], 4)} / {_fmt(row['direct_feat_std_mean'], 4)} | "
            f"{_fmt(row['trunk_hidden_rms_mean'], 4)} / {_fmt(row['trunk_hidden_std_mean'], 4)} | "
            f"{_fmt(row['trunk_hidden_step_delta_mean'], 4)} | "
            f"{row['dynamic_range_flag']} | {row['step_instability_flag']} |"
        )
    return lines


def _render_drift_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "| candidate | override | direct_feat freerun drift | trunk_hidden freerun drift | freerun-teacher direct error Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['plan_source']}/{row['meas_source']} | "
            f"{_fmt(row['direct_feat_rms_abs_drift_mean'], 4)} | "
            f"{_fmt(row['trunk_hidden_rms_abs_drift_mean'], 4)} | "
            f"{_fmt(row['direct_error_delta_mean'], 4)} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal white-box audit for cp015 non-plan direct path.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG)
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--candidates",
        type=str,
        default="baseline_replace,coadapt_4x_directonly_calibration_240,coadapt_4x_direct_plus_plan_ownership_240_noeventclock",
    )
    parser.add_argument("--modes", type=str, default=",".join(DEFAULT_MODES))
    parser.add_argument("--overrides", type=str, default=",".join(f"{a}/{b}" for a, b in DEFAULT_OVERRIDES))
    parser.add_argument("--inspect-device", type=str, default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    candidate_names = _parse_candidate_names(args.candidates)
    modes = _parse_modes(args.modes)
    overrides = _parse_overrides(args.overrides)

    meta_cache: Dict[str, Dict[str, Any]] = {}
    layout_cache: Dict[str, Dict[str, Any]] = {}
    artifacts: List[ArtifactSpec] = []

    for candidate_name in candidate_names:
        spec = CANDIDATE_SPECS[candidate_name]
        meta = _load_base_eval_meta(spec.base_eval)
        meta_cache[candidate_name] = meta
        layout_cache[candidate_name] = _infer_direct_layout(meta, device=str(args.inspect_device))

        for mode_key in modes:
            for plan_source, meas_source in overrides:
                eval_mode = _label_mode(mode_key)
                out_dir = args.out_root / candidate_name / mode_key / _slug(plan_source, meas_source)
                path = out_dir / "Walk_F_freerun_cycles.json"
                origin = "probe"
                if args.force or (not _artifact_has_arm_probe(path)):
                    out_dir = path.parent
                    cmd = _build_eval_command(
                        python_exe=str(args.python),
                        meta=meta,
                        out_dir=out_dir,
                        plan_source=plan_source,
                        meas_source=meas_source,
                        mode=mode_key,
                    )
                    _run_command(cmd, cwd=ROOT, log_path=args.audit_log)
                if not _artifact_has_arm_probe(path):
                    raise RuntimeError(f"{candidate_name} {mode_key} {plan_source}/{meas_source}: missing arm probe in {path}")
                artifacts.append(
                    ArtifactSpec(
                        candidate=candidate_name,
                        eval_mode_key=mode_key,
                        eval_mode=eval_mode,
                        plan_source=plan_source,
                        meas_source=meas_source,
                        path=path.resolve(),
                        origin=origin,
                    )
                )

    artifact_rows: List[Dict[str, Any]] = []
    for art in artifacts:
        spec = CANDIDATE_SPECS[art.candidate]
        analyzed = _analyze_artifact(art.path, layout=layout_cache[art.candidate])
        summary = analyzed["summary"]
        artifact_rows.append(
            {
                "candidate": art.candidate,
                "self_contained": bool(spec.self_contained),
                "event_clock_enabled": bool(spec.event_clock_enabled),
                "eval_mode_key": art.eval_mode_key,
                "eval_mode": art.eval_mode,
                "plan_source": art.plan_source,
                "meas_source": art.meas_source,
                "artifact_path": str(art.path),
                "origin": art.origin,
                "summary": summary,
                "per_step": analyzed["per_step"],
            }
        )

    baseline_ref: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for row in artifact_rows:
        if row["candidate"] == "baseline_replace":
            baseline_ref[(row["eval_mode_key"], row["plan_source"], row["meas_source"])] = row

    whitebox_rows: List[Dict[str, Any]] = []
    for row in artifact_rows:
        summary = row["summary"]
        base = baseline_ref.get((row["eval_mode_key"], row["plan_source"], row["meas_source"]), None)
        shrink = "ref"
        unstable = "ref"
        if base is not None and row["candidate"] != "baseline_replace":
            base_summary = base["summary"]
            trunk_rms_ratio = _ratio(summary["trunk_hidden_rms_mean"], base_summary["trunk_hidden_rms_mean"])
            trunk_std_ratio = _ratio(summary["trunk_hidden_std_mean"], base_summary["trunk_hidden_std_mean"])
            step_ratio = _ratio(summary["trunk_hidden_step_delta_mean"], base_summary["trunk_hidden_step_delta_mean"])
            shrink = "yes" if (
                (math.isfinite(trunk_rms_ratio) and trunk_rms_ratio < 0.90)
                or (math.isfinite(trunk_std_ratio) and trunk_std_ratio < 0.90)
            ) else "no"
            unstable = "yes" if (math.isfinite(step_ratio) and step_ratio > 1.10) else "no"
        whitebox_rows.append(
            {
                "candidate": row["candidate"],
                "eval_mode_key": row["eval_mode_key"],
                "eval_mode": row["eval_mode"],
                "plan_source": row["plan_source"],
                "meas_source": row["meas_source"],
                "artifact_path": row["artifact_path"],
                "direct_error_mean": summary["direct_error_mean"],
                "direct_feat_rms_mean": summary["direct_feat_rms_mean"],
                "direct_feat_std_mean": summary["direct_feat_std_mean"],
                "plan_rms_mean": summary["plan_rms_mean"],
                "meas_rms_mean": summary["meas_rms_mean"],
                "trunk_hidden_rms_mean": summary["trunk_hidden_rms_mean"],
                "trunk_hidden_std_mean": summary["trunk_hidden_std_mean"],
                "trunk_hidden_step_delta_mean": summary["trunk_hidden_step_delta_mean"],
                "dynamic_range_flag": shrink,
                "step_instability_flag": unstable,
            }
        )

    drift_rows: List[Dict[str, Any]] = []
    for candidate_name in candidate_names:
        by_key = {
            (row["eval_mode_key"], row["plan_source"], row["meas_source"]): row
            for row in artifact_rows
            if row["candidate"] == candidate_name
        }
        for plan_source, meas_source in overrides:
            teacher = by_key.get(("teacher_x_gt", plan_source, meas_source), None)
            freerun = by_key.get(("freerun", plan_source, meas_source), None)
            if teacher is None or freerun is None:
                continue
            drift = _paired_drift(
                Path(str(teacher["artifact_path"])),
                Path(str(freerun["artifact_path"])),
                layout=layout_cache[candidate_name],
            )
            drift_rows.append(
                {
                    "candidate": candidate_name,
                    "plan_source": plan_source,
                    "meas_source": meas_source,
                    **drift,
                }
            )

    artifact_table = _render_artifact_table(artifact_rows)
    whitebox_table = _render_whitebox_table(whitebox_rows)
    drift_table = _render_drift_table(drift_rows)

    summary = {
        "out_root": str(args.out_root.resolve()),
        "audit_log": str(args.audit_log.resolve()),
        "candidates": {
            name: {
                "base_eval": str(CANDIDATE_SPECS[name].base_eval.resolve()),
                "layout": layout_cache[name],
            }
            for name in candidate_names
        },
        "artifacts": artifact_rows,
        "whitebox_rows": whitebox_rows,
        "drift_rows": drift_rows,
        "tables": {
            "artifact_table": artifact_table,
            "whitebox_table": whitebox_table,
            "drift_table": drift_table,
        },
    }
    _write_json(args.summary_json, summary)

    md_lines: List[str] = []
    md_lines.append("# Non-plan direct white-box audit")
    md_lines.append("")
    md_lines.append("## Artifact table")
    md_lines.extend(artifact_table)
    md_lines.append("")
    md_lines.append("## White-box table")
    md_lines.extend(whitebox_table)
    md_lines.append("")
    md_lines.append("## Teacher-vs-freerun drift table")
    md_lines.extend(drift_table)
    md_lines.append("")
    _write_text(args.summary_md, "\n".join(md_lines).rstrip() + "\n")

    print(f"[OK] wrote summary json: {args.summary_json}")
    print(f"[OK] wrote summary md: {args.summary_md}")


if __name__ == "__main__":
    main()
