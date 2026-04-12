#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.stage6_anchor_boundary import (  # noqa: E402
    apply_locked_boundary,
    canonical_affine_stats,
    canonical_encoder_bundle,
    canonical_stage6_config,
    canonical_teacher,
    candidate_pool_policy,
    payload_contract_ok,
)


DEFAULT_STAGE6_CONFIG = canonical_stage6_config(ROOT)
DEFAULT_TEACHER = canonical_teacher(ROOT)
DEFAULT_ENCODER_BUNDLE = canonical_encoder_bundle(ROOT)
DEFAULT_AFFINE_STATS = canonical_affine_stats(ROOT)
GROUP_SUMMARY_TOOL = ROOT / "tools" / "phasea_group_summary.py"


@dataclass(frozen=True)
class Candidate:
    lane_name: str
    display_name: str
    ckpt: Path
    run_label: str
    selector: str
    family: str
    basetrain_epoch: Optional[int]
    source_summary: Optional[Path]
    discovery_tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidatePaths:
    lane_root: Path
    model_root: Path
    lane_log: Path
    stage6_log_json: Path
    stage6_ckpt: Path
    stage6_init_json: Path
    stage6_eval_dir: Path
    stage6_eval_json: Path
    stage6_group_json: Path
    run_name: str


PRIMARY_EXIT_METRICS: Tuple[str, ...] = (
    "all_ex_root_mean",
    "leg_mean",
    "nonleg_mean",
    "foot_l_ball_l_sic12_15_mean",
    "calf_r_sic2_4_mean",
)
RED_FLAG_METRICS: Tuple[str, ...] = (
    "leg_mean",
    "nonleg_mean",
    "foot_l_ball_l_sic12_15_mean",
    "calf_r_sic2_4_mean",
)


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _nanmean(values: Iterable[Any]) -> float:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _fmt(value: Any, nd: int = 6) -> str:
    value = _safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{nd}f}"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sanitize_token(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    collapsed = "".join(out).strip("_")
    return collapsed or "lane"


def _build_paths(*, out_root: Path, model_out_root: Path, lane_name: str, run_tag: str) -> CandidatePaths:
    lane_root = out_root / lane_name
    model_root = model_out_root / lane_name
    run_name = f"{lane_name}_stage6_anchor_{run_tag}"
    return CandidatePaths(
        lane_root=lane_root,
        model_root=model_root,
        lane_log=lane_root / "lane.log",
        stage6_log_json=model_root / f"posttrain_log_{run_name}.json",
        stage6_ckpt=model_root / f"ckpt_last_{run_name}.pth",
        stage6_init_json=lane_root / "stage6_init_stats.json",
        stage6_eval_dir=lane_root / "stage6_freerun",
        stage6_eval_json=lane_root / "stage6_freerun" / "Walk_F_freerun_cycles.json",
        stage6_group_json=lane_root / "stage6_group_summary.json",
        run_name=run_name,
    )


def _run_cmd(cmd: Sequence[str], *, log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
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


def _parse_manual_candidate(spec: str) -> Candidate:
    parts = [x.strip() for x in str(spec).split("|")]
    if not parts or "=" not in parts[0]:
        raise SystemExit(
            f"[FATAL] invalid --candidate spec: {spec!r}; expected lane=ckpt|run_label|selector|family|epoch"
        )
    lane_name, ckpt = [x.strip() for x in parts[0].split("=", 1)]
    run_label = parts[1] if len(parts) >= 2 and parts[1] else lane_name
    selector = parts[2] if len(parts) >= 3 and parts[2] else lane_name
    family = parts[3] if len(parts) >= 4 and parts[3] else run_label
    basetrain_epoch = None
    if len(parts) >= 5 and parts[4]:
        try:
            basetrain_epoch = int(parts[4])
        except Exception:
            basetrain_epoch = None
    return Candidate(
        lane_name=_sanitize_token(lane_name),
        display_name=lane_name,
        ckpt=_resolve(ckpt),
        run_label=run_label,
        selector=selector,
        family=family,
        basetrain_epoch=basetrain_epoch,
        source_summary=None,
        discovery_tags=("manual",),
    )


def _discover_candidates_from_summary(summary_path: Path) -> List[Candidate]:
    payload = _load_json(summary_path)
    run_label = summary_path.parent.name
    out: List[Candidate] = []
    for row in payload.get("candidates", []):
        if not isinstance(row, Mapping):
            continue
        display_name = str(row.get("name") or "")
        ckpt_raw = str(row.get("ckpt") or "")
        if not display_name or not ckpt_raw:
            continue
        selector = str(row.get("selector") or display_name)
        family = str(row.get("family") or run_label)
        epoch = row.get("basetrain_epoch")
        try:
            basetrain_epoch = int(epoch) if epoch is not None else None
        except Exception:
            basetrain_epoch = None
        lane_name = _sanitize_token(f"{run_label}__{display_name}")
        tags = tuple(str(tag) for tag in row.get("discovery_tags", []) if str(tag))
        out.append(
            Candidate(
                lane_name=lane_name,
                display_name=display_name,
                ckpt=_resolve(ckpt_raw),
                run_label=run_label,
                selector=selector,
                family=family,
                basetrain_epoch=basetrain_epoch,
                source_summary=summary_path,
                discovery_tags=tags,
            )
        )
    return out


def _collect_candidates(summary_paths: Sequence[Path], manual_specs: Sequence[str]) -> List[Candidate]:
    rows: List[Candidate] = []
    seen: set[Tuple[str, str]] = set()
    for summary_path in summary_paths:
        for cand in _discover_candidates_from_summary(summary_path):
            key = (cand.lane_name, str(cand.ckpt))
            if key in seen:
                continue
            seen.add(key)
            rows.append(cand)
    for spec in manual_specs:
        cand = _parse_manual_candidate(spec)
        key = (cand.lane_name, str(cand.ckpt))
        if key in seen:
            continue
        seen.add(key)
        rows.append(cand)
    return rows


def _state_dict_digest(path: Path) -> str:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state_dict, Mapping):
        return f"raw-file::{path.stat().st_size}"
    md5 = hashlib.md5()
    for key in sorted(state_dict):
        value = state_dict[key]
        md5.update(str(key).encode("utf-8"))
        if torch.is_tensor(value):
            value_cpu = value.detach().cpu().contiguous()
            md5.update(str(value_cpu.dtype).encode("utf-8"))
            md5.update(str(tuple(value_cpu.shape)).encode("utf-8"))
            md5.update(value_cpu.numpy().tobytes())
        else:
            md5.update(repr(value).encode("utf-8"))
    return md5.hexdigest()


def _group_equivalent_candidates(candidates: Sequence[Candidate]) -> Tuple[List[Candidate], Dict[str, str], Dict[str, str]]:
    reps: List[Candidate] = []
    lane_to_rep: Dict[str, str] = {}
    lane_to_digest: Dict[str, str] = {}
    digest_to_rep: Dict[str, str] = {}
    rep_by_lane: Dict[str, Candidate] = {}
    for cand in candidates:
        digest = _state_dict_digest(cand.ckpt)
        lane_to_digest[cand.lane_name] = digest
        rep_lane = digest_to_rep.get(digest)
        if rep_lane is None:
            digest_to_rep[digest] = cand.lane_name
            lane_to_rep[cand.lane_name] = cand.lane_name
            reps.append(cand)
            rep_by_lane[cand.lane_name] = cand
        else:
            lane_to_rep[cand.lane_name] = rep_lane
    reps.sort(key=lambda cand: cand.lane_name)
    return reps, lane_to_rep, lane_to_digest


def _extract_stage6_init(log_json: Path, out_json: Path) -> Dict[str, Any]:
    obj = _load_json(log_json)
    rows = obj.get("log", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"missing log rows in {log_json}")

    def build(row: Mapping[str, Any]) -> Dict[str, float]:
        dir_leg = _safe_float(row.get("dir_leg_base"))
        dir_nonleg = _safe_float(row.get("dir_nonleg_base"))
        leg_ratio = dir_leg / dir_nonleg if math.isfinite(dir_leg) and math.isfinite(dir_nonleg) and abs(dir_nonleg) > 1e-12 else float("nan")
        return {
            "dir_leg_base": dir_leg,
            "dir_nonleg_base": dir_nonleg,
            "leg_over_nonleg": leg_ratio,
            "step": _safe_float(row.get("step")),
        }

    head = [build(row) for row in rows[: min(len(rows), 20)]]
    payload = {
        "source": str(log_json),
        "rows": int(len(rows)),
        "head_count": int(len(head)),
        "step1": head[0],
        "head20_mean": {
            "dir_leg_base": _nanmean(r["dir_leg_base"] for r in head),
            "dir_nonleg_base": _nanmean(r["dir_nonleg_base"] for r in head),
            "leg_over_nonleg": _nanmean(r["leg_over_nonleg"] for r in head),
        },
    }
    _write_json(out_json, payload)
    return payload


def _stage6_init_needs_refresh(path: Path) -> bool:
    if not path.is_file():
        return True
    try:
        payload = _load_json(path)
    except Exception:
        return True
    step1 = payload.get("step1", {}) if isinstance(payload.get("step1"), Mapping) else {}
    head20 = payload.get("head20_mean", {}) if isinstance(payload.get("head20_mean"), Mapping) else {}
    return (not math.isfinite(_safe_float(step1.get("leg_over_nonleg")))) or (not math.isfinite(_safe_float(head20.get("leg_over_nonleg"))))


def _group_mean(path: Path, group: str) -> float:
    payload = _load_json(path)
    return _safe_float(payload.get("groups", {}).get(group, {}).get("mean"))


def _build_step_mask(steps: Sequence[Mapping[str, Any]], *, cycle_gte: int, drop_wrap: bool) -> List[bool]:
    mask: List[bool] = []
    for step in steps:
        try:
            cycle = int(step.get("cycle", 0) or 0)
        except Exception:
            cycle = 0
        keep = cycle >= int(cycle_gte)
        if keep and drop_wrap and bool(step.get("wrap_boundary_step", False)):
            keep = False
        mask.append(bool(keep))
    return mask


def _mean_direct_deg(
    payload: Mapping[str, Any],
    *,
    bones: Sequence[str],
    sic_lo: Optional[int],
    sic_hi: Optional[int],
    cycle_gte: int,
    drop_wrap: bool,
) -> float:
    steps = payload.get("metrics_per_step", [])
    per_step_direct = payload.get("per_step_direct_geolocal_deg", {})
    if not isinstance(steps, list) or not isinstance(per_step_direct, Mapping):
        return float("nan")
    names = per_step_direct.get("bone_names")
    mat = per_step_direct.get("DirectGeoLocalDeg")
    if not isinstance(names, list) or not isinstance(mat, list):
        return float("nan")
    name_to_idx = {str(name): idx for idx, name in enumerate(names)}
    indices = [int(name_to_idx[name]) for name in bones if name in name_to_idx]
    if not indices:
        return float("nan")
    mask = _build_step_mask(steps, cycle_gte=int(cycle_gte), drop_wrap=bool(drop_wrap))
    vals: List[float] = []
    for keep, step, row in zip(mask, steps, mat):
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


def _stage6_contract_checks(eval_payload: Mapping[str, Any], *, cycle_gte: int) -> Dict[str, bool]:
    checks = payload_contract_ok(eval_payload, root=ROOT)
    checks["mask_cycle_gte_ok"] = int(cycle_gte) == 1
    return checks


def _run_stage6(args: argparse.Namespace, cand: Candidate, paths: CandidatePaths) -> None:
    need_train = bool(args.force_rerun) or (not paths.stage6_ckpt.is_file()) or (not paths.stage6_log_json.is_file())
    if need_train:
        _run_cmd(
            [
                sys.executable,
                "-m",
                "train.posttrain",
                "--config",
                str(args.stage6_config),
                "--ckpt_in",
                str(cand.ckpt),
                "--out_dir",
                str(paths.model_root),
                "--run_name",
                str(paths.run_name),
                "--posttrain_contacts_source",
                "pretrain_contact",
                "--posttrain_contacts_pretrain_clamp",
                str(args.pretrain_clamp),
                "--encoder_bundle",
                str(args.encoder_bundle),
                "--posttrain_contacts_pretrain_affine_stats",
                str(args.affine_stats),
            ],
            log_path=paths.lane_log,
        )
    if bool(args.force_rerun) or _stage6_init_needs_refresh(paths.stage6_init_json):
        _extract_stage6_init(paths.stage6_log_json, paths.stage6_init_json)


def _run_stage6_eval(args: argparse.Namespace, paths: CandidatePaths) -> None:
    if paths.stage6_group_json.is_file() and not bool(args.force_rerun):
        return
    _run_cmd(
        [
            sys.executable,
            "-m",
            "train.validate.run_freerun_cycles",
            "--teacher",
            str(args.teacher),
            "--model",
            str(paths.stage6_ckpt),
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
            "--export_joint_direct_geolocal_series",
            "--out",
            str(paths.stage6_eval_dir),
            "--force",
        ],
        log_path=paths.lane_log,
    )
    _run_cmd(
        [
            sys.executable,
            str(GROUP_SUMMARY_TOOL),
            str(paths.stage6_eval_json),
            "--cycle_gte",
            str(int(args.group_cycle_gte)),
            "--drop_wrap",
            "--out",
            str(paths.stage6_group_json),
        ],
        log_path=paths.lane_log,
    )


def _build_row(args: argparse.Namespace, cand: Candidate, paths: CandidatePaths) -> Dict[str, Any]:
    stage6_init = _load_json(paths.stage6_init_json)
    stage6_eval = _load_json(paths.stage6_eval_json)
    stage6_contract_checks = _stage6_contract_checks(stage6_eval, cycle_gte=int(args.group_cycle_gte))
    exit_metrics = {
        "all_ex_root_mean": _group_mean(paths.stage6_group_json, "all_ex_root"),
        "leg_mean": _group_mean(paths.stage6_group_json, "leg"),
        "nonleg_mean": _group_mean(paths.stage6_group_json, "nonleg"),
        "arm_mean": _group_mean(paths.stage6_group_json, "arm"),
        "else_mean": _group_mean(paths.stage6_group_json, "else"),
        "foot_l_ball_l_sic12_15_mean": _mean_direct_deg(
            stage6_eval,
            bones=("foot_l", "ball_l"),
            sic_lo=12,
            sic_hi=15,
            cycle_gte=int(args.group_cycle_gte),
            drop_wrap=True,
        ),
        "calf_r_sic2_4_mean": _mean_direct_deg(
            stage6_eval,
            bones=("calf_r",),
            sic_lo=2,
            sic_hi=4,
            cycle_gte=int(args.group_cycle_gte),
            drop_wrap=True,
        ),
    }
    return {
        "lane_name": cand.lane_name,
        "display_name": cand.display_name,
        "run_label": cand.run_label,
        "selector": cand.selector,
        "family": cand.family,
        "ckpt": str(cand.ckpt),
        "basetrain_epoch": cand.basetrain_epoch,
        "source_summary": str(cand.source_summary) if cand.source_summary is not None else None,
        "discovery_tags": list(cand.discovery_tags),
        "stage6_init": stage6_init,
        "stage6_exit": exit_metrics,
        "contract_checks": stage6_contract_checks,
        "contract_ok": all(bool(v) for v in stage6_contract_checks.values()),
        "paths": {
            "lane_log": str(paths.lane_log),
            "stage6_ckpt": str(paths.stage6_ckpt),
            "stage6_init_json": str(paths.stage6_init_json),
            "stage6_eval_json": str(paths.stage6_eval_json),
            "stage6_group_json": str(paths.stage6_group_json),
        },
    }


def _assign_metric_ranks(rows: Sequence[Dict[str, Any]], metric_keys: Sequence[str]) -> Dict[str, Dict[str, int]]:
    rank_tables: Dict[str, Dict[str, int]] = {}
    for metric_key in metric_keys:
        pairs: List[Tuple[str, float]] = []
        for row in rows:
            value = _safe_float(row["stage6_exit"].get(metric_key))
            if math.isfinite(value):
                pairs.append((str(row["lane_name"]), value))
        pairs.sort(key=lambda item: (item[1], item[0]))
        rank_tables[metric_key] = {name: idx + 1 for idx, (name, _) in enumerate(pairs)}
    return rank_tables


def _sort_key(row: Mapping[str, Any]) -> Tuple[float, ...]:
    values = []
    for metric_key in PRIMARY_EXIT_METRICS:
        value = _safe_float(row["stage6_exit"].get(metric_key))
        values.append(value if math.isfinite(value) else float("inf"))
    values.append(str(row.get("lane_name")))
    return tuple(values)


def _annotate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(rows)
    rows.sort(key=_sort_key)
    rank_tables = _assign_metric_ranks(rows, PRIMARY_EXIT_METRICS)
    n = len(rows)
    red_flag_count = max(1, int(math.ceil(max(1, n) / 4.0)))
    worst_rank_start = max(1, n - red_flag_count + 1)
    for idx, row in enumerate(rows, start=1):
        row["overall_rank"] = int(idx)
        row["metric_ranks"] = {metric_key: rank_tables.get(metric_key, {}).get(str(row["lane_name"])) for metric_key in PRIMARY_EXIT_METRICS}
        red_flag_metrics: List[str] = []
        for metric_key in RED_FLAG_METRICS:
            rank = row["metric_ranks"].get(metric_key)
            if rank is not None and int(rank) >= int(worst_rank_start):
                red_flag_metrics.append(metric_key)
        row["red_flag_metrics"] = red_flag_metrics
        row["has_red_flag"] = bool(red_flag_metrics)
    return rows


def build_stage6_good_bad(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ranked = list(rows)
    n = len(ranked)
    if n == 0:
        return {
            "cut_rule": {
                "mode": "empty",
                "top_k": 0,
                "red_flag_rule": "none",
            },
            "good_lanes": [],
            "bad_lanes": [],
        }

    if n < 8:
        top_k = min(3, n)
        mode = "top3_remaining"
    else:
        top_k = max(1, int(math.ceil(n / 4.0)))
        mode = "top_quartile"

    provisional = ranked[:top_k]
    good_rows = [row for row in provisional if not bool(row.get("has_red_flag"))]
    fallback_used = False
    if not good_rows:
        good_rows = ranked[:1]
        fallback_used = True
    good_names = {str(row["lane_name"]) for row in good_rows}
    bad_rows = [row for row in ranked if str(row["lane_name"]) not in good_names]
    return {
        "cut_rule": {
            "mode": mode,
            "top_k": int(top_k),
            "red_flag_rule": "exclude provisional top-k candidates if any of leg/nonleg/foot_l_ball_l@SIC12-15/calf_r@SIC2-4 ranks in the worst quartile",
            "fallback_anchor_only_if_good_empty": bool(fallback_used),
        },
        "good_rows": good_rows,
        "bad_rows": bad_rows,
    }


def _render_md(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Stage6 Anchor Table")
    lines.append("")
    lines.append(f"- generated_at: `{summary['generated_at']}`")
    lines.append(f"- run_tag: `{summary['run_tag']}`")
    boundary = summary.get("locked_boundary", {})
    eval_contract = boundary.get("eval_contract", {})
    mask = eval_contract.get("mask", {})
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
    lines.append(
        f"- candidate_pool: `{summary['candidate_pool']['total_candidates']} candidates`, "
        f"`{summary['candidate_pool'].get('unique_stage6_replays', summary['candidate_pool']['total_candidates'])} unique Stage6 replays`, "
        f"`{','.join(summary['candidate_pool_policy']['required_saved_selectors_per_run'])} + {summary['candidate_pool_policy']['include_ckpt_epoch_glob']}`"
    )
    anchor = summary.get("anchor", {})
    lines.append(
        f"- anchor: `{anchor.get('lane_name')}` (`run={anchor.get('run_label')}`, `selector={anchor.get('selector')}`)"
    )
    lines.append(
        f"- Stage6-good set: `{', '.join(summary.get('stage6_good_set', [])) or '-'}; "
        f"Stage6-bad set: {len(summary.get('stage6_bad_set', []))}`"
    )
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append(
        "| rank | lane | run | selector | epoch | good | red_flag | "
        "all_ex_root | leg | nonleg | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 | "
        "step1 leg/nonleg | head20 leg/nonleg |"
    )
    lines.append("|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    good_set = set(summary.get("stage6_good_set", []))
    for row in summary.get("ranking", []):
        init = row.get("stage6_init", {})
        step1 = init.get("step1", {}) if isinstance(init.get("step1"), Mapping) else {}
        head20 = init.get("head20_mean", {}) if isinstance(init.get("head20_mean"), Mapping) else {}
        exit_metrics = row.get("stage6_exit", {})
        red_flag = ",".join(row.get("red_flag_metrics", [])) or "-"
        if row.get("shared_stage6_alias_of"):
            red_flag = f"{red_flag} (alias:{row.get('shared_stage6_alias_of')})"
        lines.append(
            f"| {row.get('overall_rank')} | {row.get('lane_name')} | {row.get('run_label')} | {row.get('selector')} | "
            f"{row.get('basetrain_epoch') if row.get('basetrain_epoch') is not None else '-'} | "
            f"{'Y' if row.get('lane_name') in good_set else 'N'} | "
            f"{red_flag} | "
            f"{_fmt(exit_metrics.get('all_ex_root_mean'))} | "
            f"{_fmt(exit_metrics.get('leg_mean'))} | "
            f"{_fmt(exit_metrics.get('nonleg_mean'))} | "
            f"{_fmt(exit_metrics.get('foot_l_ball_l_sic12_15_mean'))} | "
            f"{_fmt(exit_metrics.get('calf_r_sic2_4_mean'))} | "
            f"{_fmt(step1.get('leg_over_nonleg'))} | "
            f"{_fmt(head20.get('leg_over_nonleg'))} |"
        )
    lines.append("")
    lines.append("## Cut Rule")
    lines.append("")
    cut_rule = summary.get("good_bad_cut_rule", {})
    lines.append(f"- mode: `{cut_rule.get('mode')}`")
    lines.append(f"- top_k: `{cut_rule.get('top_k')}`")
    lines.append(f"- red_flag_rule: {cut_rule.get('red_flag_rule')}")
    if cut_rule.get("fallback_anchor_only_if_good_empty"):
        lines.append("- fallback: anchor-only fallback was used because provisional top-k was empty after red-flag exclusion.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run Step-1 Stage6 anchor discovery under the fixed basetrain->Stage6 contract.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--selector-summary", action="append", default=[], help="Repeatable path to handoff_selector_summary.json")
    ap.add_argument("--selector-summary-root", default="", help="Directory that contains */handoff_selector_summary.json")
    ap.add_argument("--candidate", action="append", default=[], help="Repeatable spec: lane=ckpt|run_label|selector|family|epoch")
    ap.add_argument("--run-tag", default=time.strftime("%Y%m%d"))
    ap.add_argument("--stage6-config", default=str(DEFAULT_STAGE6_CONFIG))
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
    ap.add_argument("--out-root", default="")
    ap.add_argument("--model-out-root", default="")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--allow-noncanonical-boundary", action="store_true", help="Allow off-contract diagnostics instead of enforcing the locked Stage6 anchor boundary.")
    args = ap.parse_args()

    args.stage6_config = _resolve(args.stage6_config)
    args.teacher = _resolve(args.teacher)
    args.encoder_bundle = _resolve(args.encoder_bundle)
    args.affine_stats = _resolve(args.affine_stats)
    boundary = apply_locked_boundary(args, root=ROOT, require_stage6_config=True)

    summary_paths: List[Path] = []
    if str(args.selector_summary_root).strip():
        summary_root = _resolve(args.selector_summary_root)
        summary_paths.extend(sorted(summary_root.glob("*/handoff_selector_summary.json")))
    summary_paths.extend(_resolve(path) for path in args.selector_summary)
    summary_paths = sorted({path for path in summary_paths})

    candidates = _collect_candidates(summary_paths, args.candidate)
    if not candidates:
        raise SystemExit("[FATAL] no candidates discovered; provide --selector-summary-root, --selector-summary, or --candidate")
    rep_candidates, lane_to_rep, lane_to_digest = _group_equivalent_candidates(candidates)

    required = [args.stage6_config, args.teacher, args.encoder_bundle, args.affine_stats, GROUP_SUMMARY_TOOL]
    missing = [str(path) for path in required if not Path(path).is_file()]
    missing.extend(str(path) for path in summary_paths if not path.is_file())
    missing.extend(str(cand.ckpt) for cand in candidates if not cand.ckpt.is_file())
    if missing:
        raise SystemExit("[FATAL] missing required files:\n" + "\n".join(missing))

    out_root = _resolve(args.out_root) if str(args.out_root).strip() else (ROOT / "debug_output" / f"_tmp_stage6_anchor_{args.run_tag}")
    model_out_root = _resolve(args.model_out_root) if str(args.model_out_root).strip() else (ROOT / "models" / f"__tmp_stage6_anchor_{args.run_tag}")
    out_root.mkdir(parents=True, exist_ok=True)
    model_out_root.mkdir(parents=True, exist_ok=True)

    rep_rows: Dict[str, Dict[str, Any]] = {}
    for idx, cand in enumerate(rep_candidates, start=1):
        print(f"[lane {idx}/{len(rep_candidates)}] {cand.lane_name} -> {cand.ckpt}", flush=True)
        paths = _build_paths(out_root=out_root, model_out_root=model_out_root, lane_name=cand.lane_name, run_tag=str(args.run_tag))
        paths.lane_root.mkdir(parents=True, exist_ok=True)
        paths.model_root.mkdir(parents=True, exist_ok=True)
        _run_stage6(args, cand, paths)
        _run_stage6_eval(args, paths)
        rep_rows[cand.lane_name] = _build_row(args, cand, paths)

    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        rep_lane = lane_to_rep[cand.lane_name]
        rep_row = rep_rows[rep_lane]
        row = json.loads(json.dumps(rep_row))
        row["lane_name"] = cand.lane_name
        row["display_name"] = cand.display_name
        row["run_label"] = cand.run_label
        row["selector"] = cand.selector
        row["family"] = cand.family
        row["ckpt"] = str(cand.ckpt)
        row["basetrain_epoch"] = cand.basetrain_epoch
        row["source_summary"] = str(cand.source_summary) if cand.source_summary is not None else None
        row["discovery_tags"] = list(cand.discovery_tags)
        row["state_dict_digest"] = lane_to_digest[cand.lane_name]
        row["shared_stage6_alias_of"] = None if rep_lane == cand.lane_name else rep_lane
        rows.append(row)

    ranked = _annotate_rows(rows)
    sets = build_stage6_good_bad(ranked)
    good_rows = sets["good_rows"]
    bad_rows = sets["bad_rows"]
    anchor = ranked[0]

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_tag": str(args.run_tag),
        "locked_boundary": boundary,
        "candidate_pool_policy": candidate_pool_policy(),
        "candidate_pool": {
            "selector_summary_paths": [str(path) for path in summary_paths],
            "total_candidates": int(len(candidates)),
            "unique_stage6_replays": int(len(rep_candidates)),
            "discovery_rule": "collect all candidates from supplied handoff_selector_summary.json files plus any manual --candidate entries",
        },
        "ranking_policy": {
            "primary_metric_order": list(PRIMARY_EXIT_METRICS),
            "sort_mode": "lexicographic_ascending",
            "tie_break": "lane_name",
            "note": "Exit metrics dominate Step 1. Init/readiness metrics are diagnostic only.",
        },
        "good_bad_cut_rule": sets["cut_rule"],
        "anchor": {
            "lane_name": anchor["lane_name"],
            "display_name": anchor["display_name"],
            "run_label": anchor["run_label"],
            "selector": anchor["selector"],
            "family": anchor["family"],
            "ckpt": anchor["ckpt"],
            "basetrain_epoch": anchor["basetrain_epoch"],
            "overall_rank": anchor["overall_rank"],
        },
        "stage6_good_set": [row["lane_name"] for row in good_rows],
        "stage6_bad_set": [row["lane_name"] for row in bad_rows],
        "ranking": ranked,
    }

    _write_json(out_root / "stage6_anchor_table.json", summary)
    (out_root / "stage6_anchor_table.md").write_text(_render_md(summary), encoding="utf-8")
    print(f"[done] {out_root / 'stage6_anchor_table.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
