#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from phasea_group_summary import _pick_group_indices


DEFAULT_MANIFEST = ROOT / "config" / "stage6_entry_contract_matrix_20260330_manifest.json"
GROUP_SUMMARY_TOOL = ROOT / "tools" / "phasea_group_summary.py"


@dataclass
class Resources:
    manifest_path: Path
    base_config: Path
    stage6_config: Path
    wrapper: Path
    teacher: Path
    encoder_bundle: Path
    affine_stats: Path
    old_stage6_exit_baseline: Path
    basetrain_out_root: Path
    stage6_out_root: Path
    debug_root: Path
    materialized_config_root: Path
    eval_rounds: int
    eval_depth: int
    eval_time_index_mode: str
    eval_phase_reset_source: str
    eval_pretrain_clamp: float
    group_cycle_gte: int
    group_drop_wrap: bool
    name: str


@dataclass
class FamilySpec:
    family: str
    notes: str
    payload: Dict[str, Any]


@dataclass
class CandidateSpec:
    family: str
    candidate: str
    source_ckpt: Path
    basetrain_epoch: Optional[int]
    selectors: List[str] = field(default_factory=list)
    selector_paths: List[str] = field(default_factory=list)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: Iterable[Any]) -> float:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _fmt(value: Any, digits: int = 6) -> str:
    value = _safe_float(value)
    return f"{value:.{digits}f}" if math.isfinite(value) else "missing"


def _rel(path: Path) -> str:
    return os.path.relpath(path, ROOT)


def _ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def _load_resources(manifest_path: Path) -> tuple[Resources, List[FamilySpec]]:
    payload = _load_json(manifest_path)
    freerun_eval = payload.get("freerun_eval", {}) if isinstance(payload.get("freerun_eval", {}), dict) else {}
    group_summary = payload.get("group_summary", {}) if isinstance(payload.get("group_summary", {}), dict) else {}
    resources = Resources(
        manifest_path=manifest_path,
        base_config=_resolve(payload["base_config"]),
        stage6_config=_resolve(payload["stage6_config"]),
        wrapper=_resolve(payload["wrapper"]),
        teacher=_resolve(payload["teacher"]),
        encoder_bundle=_resolve(payload["encoder_bundle"]),
        affine_stats=_resolve(payload["affine_stats"]),
        old_stage6_exit_baseline=_resolve(payload["old_stage6_exit_baseline"]),
        basetrain_out_root=_resolve(payload["basetrain_out_root"]),
        stage6_out_root=_resolve(payload["stage6_out_root"]),
        debug_root=_resolve(payload["debug_root"]),
        materialized_config_root=_resolve(payload["materialized_config_root"]),
        eval_rounds=int(freerun_eval.get("rounds", 5)),
        eval_depth=int(freerun_eval.get("depth", 3)),
        eval_time_index_mode=str(freerun_eval.get("time_index_mode", "cycle")),
        eval_phase_reset_source=str(freerun_eval.get("phase_reset_source", "none")),
        eval_pretrain_clamp=float(freerun_eval.get("pretrain_clamp", 1.0)),
        group_cycle_gte=int(group_summary.get("cycle_gte", 1)),
        group_drop_wrap=bool(group_summary.get("drop_wrap", True)),
        name=str(payload.get("name", manifest_path.stem)),
    )
    families_payload = payload.get("families", [])
    families: List[FamilySpec] = []
    if not isinstance(families_payload, list):
        raise SystemExit(f"[FATAL] manifest families must be a list: {manifest_path}")
    for item in families_payload:
        if not isinstance(item, dict):
            raise SystemExit(f"[FATAL] invalid family entry in manifest: {item!r}")
        family = str(item.get("family", "")).strip()
        if not family:
            raise SystemExit(f"[FATAL] family entry missing 'family': {item!r}")
        families.append(FamilySpec(family=family, notes=str(item.get("notes", "")).strip(), payload=dict(item)))
    return resources, families


def _require_paths(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("[FATAL] missing required inputs:\n- " + "\n- ".join(missing))


def _seed_suffix(base_run_name: str, fallback_tag: str) -> tuple[str, str]:
    match = re.match(r"^(?P<prefix>.+?)_seed(?P<seed>\d+)_(?P<date>\d{8})$", base_run_name)
    if match:
        return str(match.group("prefix")), f"seed{match.group('seed')}_{fallback_tag}"
    return base_run_name, fallback_tag


def _parse_epoch_from_name(path: Path) -> Optional[int]:
    match = re.search(r"ckpt_epoch_(\d+)\.pth$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


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


def _find_max_epoch(exp_dir: Path) -> Optional[int]:
    metrics_dir = exp_dir / "metrics"
    epochs: List[int] = []
    if metrics_dir.is_dir():
        for path in metrics_dir.glob("valfree_ep*.json"):
            try:
                epochs.append(int(path.stem.rsplit("ep", 1)[1]))
            except Exception:
                continue
    if epochs:
        return max(epochs)
    epoch_ckpts = sorted(exp_dir.glob("ckpt_epoch_*.pth"))
    parsed = [epoch for epoch in (_parse_epoch_from_name(path) for path in epoch_ckpts) if epoch is not None]
    return max(parsed) if parsed else None


def _expected_run_name(base_cfg: Mapping[str, Any], family: str, run_tag: str) -> str:
    base_run_name = str(base_cfg.get("run_name", "basetrain"))
    prefix, suffix = _seed_suffix(base_run_name, run_tag)
    return f"{prefix}_{family}_{suffix}"


def _materialize_family_config(resources: Resources, base_cfg: Mapping[str, Any], family: FamilySpec, run_tag: str) -> tuple[Path, Path, str]:
    cfg = copy.deepcopy(dict(base_cfg))
    run_name = _expected_run_name(base_cfg, family.family, run_tag)
    exp_dir = resources.basetrain_out_root / run_name
    config_path = resources.materialized_config_root / f"{run_name}.json"
    cfg["out"] = f"./{_rel(resources.basetrain_out_root)}"
    cfg["run_name"] = run_name
    cfg["save_fit_ckpt_epochs"] = str(family.payload.get("save_fit_ckpt_epochs", "10-15"))
    cfg["freerun_debug_path"] = f"./{_rel(resources.debug_root / 'basetrain_freerun_diag' / f'{run_name}.pt')}"
    if "freerun_stage_schedule" in family.payload:
        cfg["freerun_stage_schedule"] = copy.deepcopy(family.payload["freerun_stage_schedule"])
    if "epochs" in family.payload:
        cfg["epochs"] = int(family.payload["epochs"])
    meta = dict(cfg.get("strategy_meta") or {})
    meta.update(
        {
            "manifest_source": _rel(resources.manifest_path),
            "family": family.family,
            "notes": family.notes,
            "source_base_config": _rel(resources.base_config),
        }
    )
    cfg["strategy_meta"] = meta
    cfg["config_json"] = f"./{_rel(config_path)}"
    _write_json(config_path, cfg)
    return config_path, exp_dir, run_name


def _run_cmd(cmd: Sequence[str], *, log_path: Path, dry_run: bool) -> int:
    cmd = [str(part) for part in cmd]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        if dry_run:
            fh.write("[dry_run] skipped\n")
            fh.flush()
            return 0
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        proc = subprocess.Popen(
            cmd,
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
            fh.write(line)
        code = proc.wait()
        fh.write(f"[exit_code] {code}\n")
        fh.flush()
        return int(code)


def _saved_ckpt_filenames(exp_dir: Path, run_name: str) -> List[str]:
    names: List[str] = []
    names.extend(path.name for path in sorted(exp_dir.glob("ckpt_epoch_*.pth")))
    for selector in ("best_free", "best_teacher", "last"):
        path = exp_dir / f"ckpt_{selector}_{run_name}.pth"
        if path.is_file():
            names.append(path.name)
    return names


def _discover_candidates(exp_dir: Path, run_name: str) -> tuple[List[CandidateSpec], List[str]]:
    summary = _load_basetrain_summary(exp_dir)
    candidates_by_real: Dict[str, CandidateSpec] = {}
    missing: List[str] = []

    def add_candidate(*, candidate_name: str, selector: str, path: Path, basetrain_epoch: Optional[int]) -> None:
        if not path.is_file():
            missing.append(path.name)
            return
        real = path.resolve()
        key = str(real)
        existing = candidates_by_real.get(key)
        if existing is None:
            existing = CandidateSpec(
                family=exp_dir.name,
                candidate=candidate_name,
                source_ckpt=real,
                basetrain_epoch=basetrain_epoch,
                selectors=[],
                selector_paths=[],
            )
            candidates_by_real[key] = existing
        if basetrain_epoch is not None and existing.basetrain_epoch is None:
            existing.basetrain_epoch = basetrain_epoch
        if selector not in existing.selectors:
            existing.selectors.append(selector)
        selector_path = str(path.resolve())
        if selector_path not in existing.selector_paths:
            existing.selector_paths.append(selector_path)

    for path in sorted(exp_dir.glob("ckpt_epoch_*.pth")):
        epoch = _parse_epoch_from_name(path)
        if epoch is None:
            continue
        add_candidate(candidate_name=f"epoch{epoch:03d}", selector="ckpt_epoch", path=path, basetrain_epoch=epoch)

    alias_specs = [
        ("best_free", exp_dir / f"ckpt_best_free_{run_name}.pth", _extract_summary_epoch(summary, "best_free_by_GeoDriftSlopeProxy")),
        ("best_teacher", exp_dir / f"ckpt_best_teacher_{run_name}.pth", _extract_summary_epoch(summary, "best_teacher_by_GeoLocalDeg")),
        ("last", exp_dir / f"ckpt_last_{run_name}.pth", _find_max_epoch(exp_dir)),
    ]
    for selector, path, epoch in alias_specs:
        candidate_name = f"{selector}_ep{epoch:03d}" if epoch is not None else selector
        add_candidate(candidate_name=candidate_name, selector=selector, path=path, basetrain_epoch=epoch)

    rows = list(candidates_by_real.values())
    rows.sort(
        key=lambda row: (
            row.basetrain_epoch if row.basetrain_epoch is not None else 10**9,
            row.candidate,
            str(row.source_ckpt),
        )
    )
    return rows, missing


def _group_means_from_summary(summary_json: Path) -> Dict[str, float]:
    payload = _load_json(summary_json)
    groups = payload.get("groups", {})
    return {
        "all_ex_root": _safe_float(groups.get("all_ex_root", {}).get("mean")),
        "leg": _safe_float(groups.get("leg", {}).get("mean")),
        "nonleg": _safe_float(groups.get("nonleg", {}).get("mean")),
        "arm": _safe_float(groups.get("arm", {}).get("mean")),
        "else": _safe_float(groups.get("else", {}).get("mean")),
    }


def _build_step_mask(
    steps: Sequence[Mapping[str, Any]],
    *,
    cycle_gte: int,
    drop_wrap: bool,
    exclude_sic01: bool,
) -> List[bool]:
    mask: List[bool] = []
    for step in steps:
        try:
            cycle = int(step.get("cycle", 0) or 0)
        except Exception:
            cycle = 0
        keep = cycle >= int(cycle_gte)
        if keep and drop_wrap and bool(step.get("wrap_boundary_step", False)):
            keep = False
        if keep and exclude_sic01:
            try:
                sic = int(step.get("step_in_cycle", step.get("sic", -1)))
            except Exception:
                sic = -1
            if sic in (0, 1):
                keep = False
        mask.append(bool(keep))
    return mask


def _collect_values(
    *,
    mat: Sequence[Sequence[Any]],
    steps: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
    indices: Sequence[int],
    sic_lo: Optional[int] = None,
    sic_hi: Optional[int] = None,
) -> List[float]:
    vals: List[float] = []
    for keep, step, row in zip(mask, steps, mat):
        if not keep or not isinstance(row, list):
            continue
        try:
            sic = int(step.get("step_in_cycle", step.get("sic")))
        except Exception:
            sic = None
        if sic_lo is not None and sic_hi is not None:
            if sic is None or sic < int(sic_lo) or sic > int(sic_hi):
                continue
        for idx in indices:
            if idx >= len(row):
                continue
            value = _safe_float(row[idx])
            if math.isfinite(value):
                vals.append(value)
    return vals


def _window_mean(
    *,
    mat: Sequence[Sequence[Any]],
    steps: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
    indices: Sequence[int],
    sic_lo: int,
    sic_hi: int,
) -> float:
    return _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=indices, sic_lo=sic_lo, sic_hi=sic_hi))


def _group_curve_by_sic(
    *,
    mat: Sequence[Sequence[Any]],
    steps: Sequence[Mapping[str, Any]],
    mask: Sequence[bool],
    indices: Sequence[int],
) -> Dict[int, float]:
    by_sic: Dict[int, List[float]] = {}
    for keep, step, row in zip(mask, steps, mat):
        if not keep or not isinstance(row, list):
            continue
        try:
            sic = int(step.get("step_in_cycle", step.get("sic")))
        except Exception:
            continue
        bucket = by_sic.setdefault(sic, [])
        for idx in indices:
            if idx >= len(row):
                continue
            value = _safe_float(row[idx])
            if math.isfinite(value):
                bucket.append(value)
    return {sic: _mean(values) for sic, values in sorted(by_sic.items())}


def _curve_l1(curve_a: Mapping[int, float], curve_b: Mapping[int, float]) -> float:
    common = sorted(set(curve_a) & set(curve_b))
    if not common:
        return float("nan")
    return _mean(abs(_safe_float(curve_a[sic]) - _safe_float(curve_b[sic])) for sic in common)


def _metrics_for_eval_json(eval_json: Path, *, cycle_gte: int = 1, exclude_sic01: bool = False) -> Dict[str, Any]:
    payload = _load_json(eval_json)
    per = payload["per_step_direct_geolocal_deg"]
    names = [str(x) for x in per["bone_names"]]
    root_idx = int(per.get("root_idx", 0) or 0)
    mat = per["DirectGeoLocalDeg"]
    steps = payload["metrics_per_step"]
    groups = _pick_group_indices(names, root_idx)
    name_to_idx = {name: i for i, name in enumerate(names)}
    mask = _build_step_mask(steps, cycle_gte=cycle_gte, drop_wrap=True, exclude_sic01=exclude_sic01)

    leg_idx = groups["leg"]
    foot_idx = [name_to_idx["foot_l"], name_to_idx["ball_l"]]
    calf_r_idx = [name_to_idx["calf_r"]]
    all_ex_root = _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["all_ex_root"]))
    leg = _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=leg_idx))
    nonleg = _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["nonleg"]))
    arm = _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["arm"]))
    else_mean = _mean(_collect_values(mat=mat, steps=steps, mask=mask, indices=groups["else"]))

    foot_l_ball_l_sic12_15 = _window_mean(mat=mat, steps=steps, mask=mask, indices=foot_idx, sic_lo=12, sic_hi=15)
    calf_r_sic2_4 = _window_mean(mat=mat, steps=steps, mask=mask, indices=calf_r_idx, sic_lo=2, sic_hi=4)
    leg_sic12_24 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=12, sic_hi=24)
    leg_sic20_24 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=20, sic_hi=24)
    leg_sic49_52 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=49, sic_hi=52)
    leg_sic57_70 = _window_mean(mat=mat, steps=steps, mask=mask, indices=leg_idx, sic_lo=57, sic_hi=70)
    return {
        "source": str(eval_json),
        "mask": {
            "cycle_gte": int(cycle_gte),
            "drop_wrap": True,
            "exclude_sic01": bool(exclude_sic01),
            "kept_steps": int(sum(mask)),
            "total_steps": int(len(mask)),
        },
        "all_ex_root": all_ex_root,
        "leg": leg,
        "nonleg": nonleg,
        "arm": arm,
        "else": else_mean,
        "foot_l_ball_l_sic12_15": foot_l_ball_l_sic12_15,
        "calf_r_sic2_4": calf_r_sic2_4,
        "calf_r_over_leg": _ratio(calf_r_sic2_4, leg),
        "leg_12_24_over_57_70": _ratio(leg_sic12_24, leg_sic57_70),
        "leg_20_24_plus_49_52_over_57_70": _ratio(leg_sic20_24 + leg_sic49_52, leg_sic57_70),
        "leg_curve": _group_curve_by_sic(mat=mat, steps=steps, mask=mask, indices=leg_idx),
        "all_ex_root_curve": _group_curve_by_sic(mat=mat, steps=steps, mask=mask, indices=groups["all_ex_root"]),
    }


def _blended_distance(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    leg_dist = _curve_l1(metrics["leg_curve"], baseline["leg_curve"])
    all_dist = _curve_l1(metrics["all_ex_root_curve"], baseline["all_ex_root_curve"])
    if not math.isfinite(_safe_float(leg_dist)) or not math.isfinite(_safe_float(all_dist)):
        return float("nan")
    return 0.7 * _safe_float(leg_dist) + 0.3 * _safe_float(all_dist)


def _family_order_map(families: Sequence[FamilySpec]) -> Dict[str, int]:
    return {family.family: idx for idx, family in enumerate(families)}


def _lane_paths(resources: Resources, family: str, candidate: str) -> Dict[str, Path]:
    lane_root = resources.debug_root / family / candidate
    model_root = resources.stage6_out_root / family / candidate
    run_tag = resources.name.rsplit("_", 1)[-1] if "_" in resources.name else resources.name
    run_name = f"{family}_{candidate}_stage6_exact_{run_tag}"
    return {
        "lane_root": lane_root,
        "model_root": model_root,
        "run_name": Path(run_name),
        "lane_log": lane_root / "lane.log",
        "stage6_ckpt": model_root / f"ckpt_last_{run_name}.pth",
        "stage6_log_json": model_root / f"posttrain_log_{run_name}.json",
        "stage6_eval_dir": lane_root / "stage6_freerun",
        "stage6_eval_json": lane_root / "stage6_freerun" / "Walk_F_freerun_cycles.json",
        "stage6_group_json": lane_root / "stage6_group_summary.json",
        "status_json": lane_root / "status.json",
    }


def _run_stage6_candidate(
    resources: Resources,
    candidate: CandidateSpec,
    baseline_metrics: Mapping[str, Any],
    *,
    force_stage6: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    paths = _lane_paths(resources, candidate.family, candidate.candidate)
    paths["lane_root"].mkdir(parents=True, exist_ok=True)
    paths["model_root"].mkdir(parents=True, exist_ok=True)
    log_file = paths["lane_log"]
    missing: List[str] = []
    step_status: Dict[str, Any] = {}

    need_train = force_stage6 or (not paths["stage6_ckpt"].is_file()) or (not paths["stage6_log_json"].is_file())
    if need_train:
        step_status["stage6_train"] = _run_cmd(
            [
                str(resources.wrapper),
                "-m",
                "train.posttrain",
                "--config",
                str(resources.stage6_config),
                "--ckpt_in",
                str(candidate.source_ckpt),
                "--out_dir",
                str(paths["model_root"]),
                "--run_name",
                str(paths["run_name"]),
                "--posttrain_contacts_source",
                "pretrain_contact",
                "--posttrain_contacts_pretrain_clamp",
                str(resources.eval_pretrain_clamp),
                "--encoder_bundle",
                str(resources.encoder_bundle),
                "--posttrain_contacts_pretrain_affine_stats",
                str(resources.affine_stats),
            ],
            log_path=log_file,
            dry_run=dry_run,
        )
    else:
        step_status["stage6_train"] = "skipped_existing"

    if paths["stage6_ckpt"].is_file():
        need_eval = force_stage6 or (not paths["stage6_eval_json"].is_file())
        if need_eval:
            step_status["stage6_eval"] = _run_cmd(
                [
                    str(resources.wrapper),
                    "-m",
                    "train.validate.run_freerun_cycles",
                    "--teacher",
                    str(resources.teacher),
                    "--model",
                    str(paths["stage6_ckpt"]),
                    "--rounds",
                    str(resources.eval_rounds),
                    "--depth",
                    str(resources.eval_depth),
                    "--time-index-mode",
                    str(resources.eval_time_index_mode),
                    "--phase_reset_source",
                    str(resources.eval_phase_reset_source),
                    "--contacts_meas_source",
                    "pretrain_contact",
                    "--contacts_meas_pretrain_clamp",
                    str(resources.eval_pretrain_clamp),
                    "--contacts_meas_pretrain_affine_stats",
                    str(resources.affine_stats),
                    "--encoder-bundle",
                    str(resources.encoder_bundle),
                    "--export_joint_direct_geolocal_series",
                    "--out",
                    str(paths["stage6_eval_dir"]),
                    "--force",
                ],
                log_path=log_file,
                dry_run=dry_run,
            )
        else:
            step_status["stage6_eval"] = "skipped_existing"
    else:
        step_status["stage6_eval"] = "missing_stage6_ckpt"

    if paths["stage6_eval_json"].is_file():
        need_summary = force_stage6 or (not paths["stage6_group_json"].is_file())
        if need_summary:
            cmd = [
                str(resources.wrapper),
                str(GROUP_SUMMARY_TOOL),
                str(paths["stage6_eval_json"]),
                "--cycle_gte",
                str(resources.group_cycle_gte),
            ]
            if resources.group_drop_wrap:
                cmd.append("--drop_wrap")
            cmd.extend(["--out", str(paths["stage6_group_json"])])
            step_status["stage6_summary"] = _run_cmd(cmd, log_path=log_file, dry_run=dry_run)
        else:
            step_status["stage6_summary"] = "skipped_existing"
    else:
        step_status["stage6_summary"] = "missing_stage6_eval_json"

    row: Dict[str, Any] = {
        "family": candidate.family,
        "candidate": candidate.candidate,
        "source_basetrain_ckpt": str(candidate.source_ckpt),
        "basetrain_epoch": candidate.basetrain_epoch,
        "selector": ",".join(candidate.selectors),
        "selector_paths": list(candidate.selector_paths),
        "lane_log": str(log_file.resolve()),
        "stage6_ckpt": str(paths["stage6_ckpt"].resolve()) if paths["stage6_ckpt"].is_file() else "missing",
        "stage6_group_summary": str(paths["stage6_group_json"].resolve()) if paths["stage6_group_json"].is_file() else "missing",
        "stage6_eval_json": str(paths["stage6_eval_json"].resolve()) if paths["stage6_eval_json"].is_file() else "missing",
        "step_status": step_status,
        "all_ex_root": float("nan"),
        "leg": float("nan"),
        "nonleg": float("nan"),
        "arm": float("nan"),
        "else": float("nan"),
        "calf_r_over_leg": float("nan"),
        "ratio12_24_over_57_70": float("nan"),
        "ratio20_24_plus_49_52_over_57_70": float("nan"),
        "foot_l_ball_l_sic12_15": float("nan"),
        "blended_distance_to_old_exit": float("nan"),
        "off_basin": True,
    }

    if not paths["stage6_ckpt"].is_file():
        missing.append("stage6_ckpt")
    if not paths["stage6_eval_json"].is_file():
        missing.append("stage6_freerun_eval")
    if not paths["stage6_group_json"].is_file():
        missing.append("stage6_group_summary")

    if paths["stage6_group_json"].is_file():
        row.update(_group_means_from_summary(paths["stage6_group_json"]))
    if paths["stage6_eval_json"].is_file():
        try:
            final_metrics = _metrics_for_eval_json(paths["stage6_eval_json"], cycle_gte=1, exclude_sic01=False)
            row["calf_r_over_leg"] = _safe_float(final_metrics["calf_r_over_leg"])
            row["ratio12_24_over_57_70"] = _safe_float(final_metrics["leg_12_24_over_57_70"])
            row["ratio20_24_plus_49_52_over_57_70"] = _safe_float(final_metrics["leg_20_24_plus_49_52_over_57_70"])
            row["foot_l_ball_l_sic12_15"] = _safe_float(final_metrics["foot_l_ball_l_sic12_15"])
            row["blended_distance_to_old_exit"] = _blended_distance(final_metrics, baseline_metrics)
            row["off_basin"] = bool(
                _safe_float(row["foot_l_ball_l_sic12_15"]) > 1.25 * _safe_float(baseline_metrics["foot_l_ball_l_sic12_15"])
                or _safe_float(row["ratio20_24_plus_49_52_over_57_70"])
                > 1.25 * _safe_float(baseline_metrics["leg_20_24_plus_49_52_over_57_70"])
            )
        except Exception as exc:
            missing.append(f"shape_metrics({exc.__class__.__name__})")
            row["off_basin"] = True
    else:
        missing.append("shape_metrics")

    row["missing"] = "missing: " + ", ".join(missing) if missing else "-"
    _write_json(paths["status_json"], row)
    return row


def _render_markdown(
    resources: Resources,
    families: Sequence[FamilySpec],
    family_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, Any],
) -> str:
    family_order = _family_order_map(families)
    sorted_family_rows = sorted(family_rows, key=lambda row: family_order.get(str(row.get("family")), 10**9))
    sorted_candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            family_order.get(str(row.get("family")), 10**9),
            row.get("basetrain_epoch") if row.get("basetrain_epoch") is not None else 10**9,
            str(row.get("candidate")),
        ),
    )
    complete_stage6 = sum(1 for row in sorted_candidate_rows if str(row.get("missing", "-")) == "-")
    lines: List[str] = []
    lines.append("# Stage6 Entry Contract Matrix Report")
    lines.append("")
    lines.append("## A. run status")
    lines.append("")
    lines.append(f"- manifest: `{resources.manifest_path}`")
    lines.append(f"- basetrain base config: `{resources.base_config}`")
    lines.append(f"- Stage6 config: `{resources.stage6_config}`")
    lines.append(f"- wrapper: `{resources.wrapper}`")
    lines.append(f"- old Stage6 exit baseline: `{resources.old_stage6_exit_baseline}`")
    lines.append(f"- basetrain out root: `{resources.basetrain_out_root}`")
    lines.append(f"- Stage6 out root: `{resources.stage6_out_root}`")
    lines.append(f"- debug root: `{resources.debug_root}`")
    lines.append(f"- completed Stage6 candidates without missing outputs: `{complete_stage6}/{len(sorted_candidate_rows)}`")
    lines.append("")
    lines.append("| family | basetrain status | unique ckpt candidates | Stage6 complete | missing |")
    lines.append("|---|---|---:|---:|---|")
    for row in sorted_family_rows:
        lines.append(
            f"| {row['family']} | {row['basetrain_status']} | {row['unique_stage6_candidates']} | "
            f"{row['stage6_complete']} | {row['missing']} |"
        )
    lines.append("")
    lines.append("## B. basetrain family table")
    lines.append("")
    lines.append("| family | config/manifest source | exp_dir | saved ckpt list | missing |")
    lines.append("|---|---|---|---|---|")
    for row in sorted_family_rows:
        config_source = f"manifest=`{row['manifest_source']}`; config=`{row['materialized_config']}`"
        lines.append(
            f"| {row['family']} | {config_source} | `{row['exp_dir']}` | "
            f"`{row['saved_ckpt_list']}` | {row['missing']} |"
        )
    lines.append("")
    lines.append("## C. all-ckpt Stage6 final exit")
    lines.append("")
    lines.append("| family | candidate | source basetrain ckpt | basetrain epoch / selector | stage6_ckpt | stage6_group_summary | all_ex_root | leg | nonleg | arm | else | calf_r/leg | ratio12_24/57_70 | ratio20_24+49_52/57_70 | foot_l/ball_l@SIC12-15 | blended distance to old Stage6 exit basin | missing |")
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted_candidate_rows:
        epoch_selector = f"{row.get('basetrain_epoch', 'missing')} / {row.get('selector', 'missing') or 'missing'}"
        lines.append(
            f"| {row['family']} | {row['candidate']} | `{row['source_basetrain_ckpt']}` | {epoch_selector} | "
            f"`{row['stage6_ckpt']}` | `{row['stage6_group_summary']}` | {_fmt(row['all_ex_root'])} | {_fmt(row['leg'])} | "
            f"{_fmt(row['nonleg'])} | {_fmt(row['arm'])} | {_fmt(row['else'])} | {_fmt(row['calf_r_over_leg'])} | "
            f"{_fmt(row['ratio12_24_over_57_70'])} | {_fmt(row['ratio20_24_plus_49_52_over_57_70'])} | "
            f"{_fmt(row['foot_l_ball_l_sic12_15'])} | {_fmt(row['blended_distance_to_old_exit'])} | {row['missing']} |"
        )
    lines.append("")
    lines.append("## D. conclusion")
    lines.append("")
    lines.append(
        f"- old Stage6 exit reference `{resources.old_stage6_exit_baseline}`: all_ex_root={_fmt(baseline_metrics['all_ex_root'])}, "
        f"leg={_fmt(baseline_metrics['leg'])}, nonleg={_fmt(baseline_metrics['nonleg'])}, arm={_fmt(baseline_metrics['arm'])}, "
        f"else={_fmt(baseline_metrics['else'])}, calf_r/leg={_fmt(baseline_metrics['calf_r_over_leg'])}, "
        f"ratio12_24/57_70={_fmt(baseline_metrics['leg_12_24_over_57_70'])}, "
        f"ratio20_24+49_52/57_70={_fmt(baseline_metrics['leg_20_24_plus_49_52_over_57_70'])}, "
        f"foot_l/ball_l@SIC12-15={_fmt(baseline_metrics['foot_l_ball_l_sic12_15'])}."
    )
    finite_rows = [row for row in sorted_candidate_rows if math.isfinite(_safe_float(row.get("blended_distance_to_old_exit")))]
    if finite_rows:
        best_row = min(finite_rows, key=lambda row: _safe_float(row["blended_distance_to_old_exit"]))
        lines.append(
            f"- closest overall to the old Stage6 exit basin: `{best_row['family']} / {best_row['candidate']}` from "
            f"`{best_row['source_basetrain_ckpt']}` with blended distance `{_fmt(best_row['blended_distance_to_old_exit'])}`, "
            f"foot_l/ball_l@SIC12-15 `{_fmt(best_row['foot_l_ball_l_sic12_15'])}`, "
            f"ratio20_24+49_52/57_70 `{_fmt(best_row['ratio20_24_plus_49_52_over_57_70'])}`, "
            f"group summary `{best_row['stage6_group_summary']}`."
        )
    else:
        lines.append("- closest overall to the old Stage6 exit basin: missing.")

    family_best_rows: List[Mapping[str, Any]] = []
    for family in families:
        family_candidates = [row for row in finite_rows if str(row.get("family")) == family.family]
        if family_candidates:
            family_best_rows.append(min(family_candidates, key=lambda row: _safe_float(row["blended_distance_to_old_exit"])))
    off_basin_families = [row for row in family_best_rows if bool(row.get("off_basin", True))]
    if off_basin_families:
        lines.append("- families whose best Stage6 exit remains clearly off-basin:")
        for row in off_basin_families:
            lines.append(
                f"  - `{row['family']}` best is `{row['candidate']}` from `{row['source_basetrain_ckpt']}` with "
                f"foot_l/ball_l@SIC12-15 `{_fmt(row['foot_l_ball_l_sic12_15'])}` vs old `{_fmt(baseline_metrics['foot_l_ball_l_sic12_15'])}`, "
                f"ratio20_24+49_52/57_70 `{_fmt(row['ratio20_24_plus_49_52_over_57_70'])}` vs old `{_fmt(baseline_metrics['leg_20_24_plus_49_52_over_57_70'])}`, "
                f"blended distance `{_fmt(row['blended_distance_to_old_exit'])}`."
            )
    else:
        lines.append("- families whose best Stage6 exit remains clearly off-basin: none.")

    if finite_rows:
        lines.append(
            f"- interpretation: because the canonical Stage6 recipe stayed fixed at `{resources.stage6_config}` while only the basetrain entry family/ckpt changed, "
            "any material shift in final blended distance or hotspot metrics should be attributed primarily to the basetrain entry contract, not to a new Stage6 recipe."
        )
    else:
        lines.append("- interpretation: missing, because no completed Stage6 exits were available.")

    if finite_rows:
        on_basin = [row for row in family_best_rows if not bool(row.get("off_basin", True))]
        if on_basin:
            lines.append("- next round: worth continuing only as a minimal entry-contract follow-up around the best family above, not as a broad Stage6 loss sweep.")
        else:
            lines.append("- next round: if the best bridge family is still off-basin, the smallest justified follow-up is another entry-shape tweak around the phase_c->phase_d boundary, not a broader Stage6 loss retune.")
    else:
        lines.append("- next round: missing.")

    return "\n".join(lines) + "\n"


def _persist_report(
    *,
    resources: Resources,
    families: Sequence[FamilySpec],
    family_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, Any],
    report_json: Path,
    report_md: Path,
    run_status_json: Path,
) -> None:
    report = {
        "manifest": str(resources.manifest_path),
        "resources": {
            "base_config": str(resources.base_config),
            "stage6_config": str(resources.stage6_config),
            "wrapper": str(resources.wrapper),
            "teacher": str(resources.teacher),
            "encoder_bundle": str(resources.encoder_bundle),
            "affine_stats": str(resources.affine_stats),
            "old_stage6_exit_baseline": str(resources.old_stage6_exit_baseline),
            "basetrain_out_root": str(resources.basetrain_out_root),
            "stage6_out_root": str(resources.stage6_out_root),
            "debug_root": str(resources.debug_root),
        },
        "old_stage6_exit_metrics": baseline_metrics,
        "family_rows": list(family_rows),
        "candidate_rows": list(candidate_rows),
    }
    _write_json(report_json, report)
    report_md.write_text(_render_markdown(resources, families, family_rows, candidate_rows, baseline_metrics), encoding="utf-8")
    _write_json(
        run_status_json,
        {
            "report_json": str(report_json),
            "report_md": str(report_md),
            "families": len(family_rows),
            "candidates": len(candidate_rows),
            "stage6_complete": sum(1 for row in candidate_rows if str(row.get("missing", "-")) == "-"),
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--families", default="", help="Comma-separated family filter; empty means all manifest families.")
    ap.add_argument("--skip-basetrain", action="store_true")
    ap.add_argument("--skip-stage6", action="store_true")
    ap.add_argument("--force-basetrain", action="store_true")
    ap.add_argument("--force-stage6", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    resources, families = _load_resources(_resolve(args.manifest))
    requested = {item.strip() for item in str(args.families).split(",") if item.strip()}
    if requested:
        families = [family for family in families if family.family in requested]
    if not families:
        raise SystemExit("[FATAL] no families selected")

    _require_paths(
        [
            resources.base_config,
            resources.stage6_config,
            resources.wrapper,
            resources.teacher,
            resources.encoder_bundle,
            resources.affine_stats,
            resources.old_stage6_exit_baseline,
            GROUP_SUMMARY_TOOL,
        ]
    )
    resources.basetrain_out_root.mkdir(parents=True, exist_ok=True)
    resources.stage6_out_root.mkdir(parents=True, exist_ok=True)
    resources.debug_root.mkdir(parents=True, exist_ok=True)
    resources.materialized_config_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_json(resources.base_config)
    baseline_metrics = _metrics_for_eval_json(resources.old_stage6_exit_baseline, cycle_gte=1, exclude_sic01=False)
    report_json = resources.debug_root / "stage6_entry_contract_matrix_report.json"
    report_md = resources.debug_root / "stage6_entry_contract_matrix_report.md"
    run_status_json = resources.debug_root / "run_status.json"
    run_tag = resources.name.rsplit("_", 1)[-1] if "_" in resources.name else resources.name

    family_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    for family in families:
        log(f"[family] {family.family}")
        config_path, exp_dir, run_name = _materialize_family_config(resources, base_cfg, family, run_tag)
        family_log = resources.debug_root / "basetrain" / family.family / "train.log"

        expected_dense = [exp_dir / f"ckpt_epoch_{epoch:03d}.pth" for epoch in range(10, 16)]
        basetrain_complete = (
            (exp_dir / f"ckpt_last_{run_name}.pth").is_file()
            and all(path.is_file() for path in expected_dense)
        )
        basetrain_status: Any = "skipped_existing" if basetrain_complete and (not args.force_basetrain) else "pending"
        if not args.skip_basetrain and ((not basetrain_complete) or args.force_basetrain):
            basetrain_status = _run_cmd(
                [
                    str(resources.wrapper),
                    "-m",
                    "train.training_MPL",
                    "--config_json",
                    str(config_path),
                ],
                log_path=family_log,
                dry_run=bool(args.dry_run),
            )
        elif args.skip_basetrain:
            basetrain_status = "skipped_by_flag"

        saved_ckpt_list = ", ".join(_saved_ckpt_filenames(exp_dir, run_name)) if exp_dir.is_dir() else ""
        candidates, family_missing = _discover_candidates(exp_dir, run_name) if exp_dir.is_dir() else ([], ["exp_dir_missing"])
        stage6_complete = 0

        if not args.skip_stage6:
            for candidate in candidates:
                candidate.family = family.family
                log(f"[stage6] {family.family} / {candidate.candidate} -> {candidate.source_ckpt}")
                row = _run_stage6_candidate(
                    resources,
                    candidate,
                    baseline_metrics,
                    force_stage6=bool(args.force_stage6),
                    dry_run=bool(args.dry_run),
                )
                if str(row.get("missing", "-")) == "-":
                    stage6_complete += 1
                candidate_rows.append(row)
                _persist_report(
                    resources=resources,
                    families=families,
                    family_rows=family_rows,
                    candidate_rows=candidate_rows,
                    baseline_metrics=baseline_metrics,
                    report_json=report_json,
                    report_md=report_md,
                    run_status_json=run_status_json,
                )
        else:
            for candidate in candidates:
                candidate_rows.append(
                    {
                        "family": family.family,
                        "candidate": candidate.candidate,
                        "source_basetrain_ckpt": str(candidate.source_ckpt),
                        "basetrain_epoch": candidate.basetrain_epoch,
                        "selector": ",".join(candidate.selectors),
                        "selector_paths": list(candidate.selector_paths),
                        "lane_log": "missing",
                        "stage6_ckpt": "missing",
                        "stage6_group_summary": "missing",
                        "stage6_eval_json": "missing",
                        "step_status": {"stage6": "skipped_by_flag"},
                        "all_ex_root": float("nan"),
                        "leg": float("nan"),
                        "nonleg": float("nan"),
                        "arm": float("nan"),
                        "else": float("nan"),
                        "calf_r_over_leg": float("nan"),
                        "ratio12_24_over_57_70": float("nan"),
                        "ratio20_24_plus_49_52_over_57_70": float("nan"),
                        "foot_l_ball_l_sic12_15": float("nan"),
                        "blended_distance_to_old_exit": float("nan"),
                        "off_basin": True,
                        "missing": "missing: stage6_skipped",
                    }
                )
                _persist_report(
                    resources=resources,
                    families=families,
                    family_rows=family_rows,
                    candidate_rows=candidate_rows,
                    baseline_metrics=baseline_metrics,
                    report_json=report_json,
                    report_md=report_md,
                    run_status_json=run_status_json,
                )

        family_row = {
            "family": family.family,
            "notes": family.notes,
            "manifest_source": str(resources.manifest_path),
            "materialized_config": str(config_path),
            "exp_dir": str(exp_dir),
            "saved_ckpt_list": saved_ckpt_list if saved_ckpt_list else "missing",
            "basetrain_status": basetrain_status,
            "unique_stage6_candidates": len(candidates),
            "stage6_complete": stage6_complete,
            "missing": "missing: " + ", ".join(family_missing) if family_missing else "-",
        }
        family_rows.append(family_row)
        _persist_report(
            resources=resources,
            families=families,
            family_rows=family_rows,
            candidate_rows=candidate_rows,
            baseline_metrics=baseline_metrics,
            report_json=report_json,
            report_md=report_md,
            run_status_json=run_status_json,
        )
    log(f"report json: {report_json}")
    log(f"report md: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
