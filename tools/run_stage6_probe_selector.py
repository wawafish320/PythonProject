#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE6_CONFIG = ROOT / "config" / "posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json"
DEFAULT_TEACHER = ROOT / "validate" / "teacher_batches" / "Walk_F_teacher.json"
DEFAULT_ENCODER_BUNDLE = ROOT / "models" / "motion_encoder_equiv.pt.best.pt"
DEFAULT_AFFINE_STATS = ROOT / "debug_output" / "_tmp_phaseb_affine_20260304" / "affine_fit_mix08" / "affine_stats.json"
GROUP_SUMMARY_TOOL = ROOT / "tools" / "phasea_group_summary.py"


@dataclass
class Candidate:
    name: str
    ckpt: Path
    family: str = ""
    selector: str = ""


@dataclass
class Paths:
    lane_root: Path
    model_root: Path
    lane_log: Path
    basetrain_eval_dir: Path
    basetrain_eval_json: Path
    basetrain_group_json: Path
    stage6_log_json: Path
    stage6_ckpt: Path
    stage6_init_json: Path
    stage6_eval_dir: Path
    stage6_eval_json: Path
    stage6_group_json: Path
    run_name: str


def _resolve(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _mean(values: Iterable[Any]) -> float:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float, nd: int = 6) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def _parse_candidate(spec: str) -> Candidate:
    parts = [x.strip() for x in str(spec).split("|")]
    if len(parts) == 1:
        if "=" not in parts[0]:
            raise SystemExit(f"[FATAL] invalid --candidate spec: {spec!r}; expected name=ckpt or name=ckpt|family|selector")
        name, ckpt = [x.strip() for x in parts[0].split("=", 1)]
        return Candidate(name=name, ckpt=_resolve(ckpt))
    if "=" not in parts[0]:
        raise SystemExit(f"[FATAL] invalid --candidate spec: {spec!r}; expected name=ckpt|family|selector")
    name, ckpt = [x.strip() for x in parts[0].split("=", 1)]
    family = parts[1] if len(parts) >= 2 else ""
    selector = parts[2] if len(parts) >= 3 else ""
    return Candidate(name=name, ckpt=_resolve(ckpt), family=family, selector=selector)


def _build_paths(*, out_root: Path, model_out_root: Path, name: str, run_tag: str, run_name_stem: str) -> Paths:
    lane_root = out_root / name
    model_root = model_out_root / name
    run_name = f"{name}_{run_name_stem}_{run_tag}"
    return Paths(
        lane_root=lane_root,
        model_root=model_root,
        lane_log=lane_root / "lane.log",
        basetrain_eval_dir=lane_root / "basetrain_freerun",
        basetrain_eval_json=lane_root / "basetrain_freerun" / "Walk_F_freerun_cycles.json",
        basetrain_group_json=lane_root / "basetrain_group_summary.json",
        stage6_log_json=model_root / f"posttrain_log_{run_name}.json",
        stage6_ckpt=model_root / f"ckpt_last_{run_name}.pth",
        stage6_init_json=lane_root / "posttrain_stage6_init_stats.json",
        stage6_eval_dir=lane_root / "stage6_freerun",
        stage6_eval_json=lane_root / "stage6_freerun" / "Walk_F_freerun_cycles.json",
        stage6_group_json=lane_root / "stage6_group_summary.json",
        run_name=run_name,
    )


def _run_cmd(cmd: Sequence[str], *, log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[cmd] " + " ".join(str(x) for x in cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(str(x) for x in cmd) + "\n")
        f.flush()
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
            f.write(line)
        code = int(proc.wait())
        f.write(f"[exit_code] {code}\n")
    if code != 0:
        raise SystemExit(code)


def _extract_stage6_init(log_json: Path, out_json: Path) -> Dict[str, Any]:
    obj = _load_json(log_json)
    rows = obj.get("log", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"missing log rows in {log_json}")

    def build(row: Dict[str, Any]) -> Dict[str, float]:
        dir_leg = _safe_float(row.get("dir_leg_base"))
        dir_nonleg = _safe_float(row.get("dir_nonleg_base"))
        arm = _safe_float(row.get("arm_over_else"))
        g_arm = _safe_float(row.get("direct_grad_norm_out_arm"))
        g_else = _safe_float(row.get("direct_grad_norm_out_else"))
        leg_ratio = dir_leg / dir_nonleg if math.isfinite(dir_leg) and math.isfinite(dir_nonleg) and abs(dir_nonleg) > 1e-12 else float("nan")
        grad_ratio = g_arm / g_else if math.isfinite(g_arm) and math.isfinite(g_else) and abs(g_else) > 1e-12 else float("nan")
        return {
            "dir_leg_base": dir_leg,
            "dir_nonleg_base": dir_nonleg,
            "leg_over_nonleg": leg_ratio,
            "arm_over_else": arm,
            "direct_grad_norm_out_arm": g_arm,
            "direct_grad_norm_out_else": g_else,
            "grad_arm_over_else": grad_ratio,
            "step": _safe_float(row.get("step")),
        }

    head = [build(row) for row in rows[: min(len(rows), 20)]]
    payload = {
        "source": str(log_json),
        "rows": int(len(rows)),
        "head_count": int(len(head)),
        "step1": head[0],
        "head20_mean": {
            "dir_leg_base": _mean(r["dir_leg_base"] for r in head),
            "dir_nonleg_base": _mean(r["dir_nonleg_base"] for r in head),
            "leg_over_nonleg": _mean(r["leg_over_nonleg"] for r in head),
            "arm_over_else": _mean(r["arm_over_else"] for r in head),
            "direct_grad_norm_out_arm": _mean(r["direct_grad_norm_out_arm"] for r in head),
            "direct_grad_norm_out_else": _mean(r["direct_grad_norm_out_else"] for r in head),
            "grad_arm_over_else": _mean(r["grad_arm_over_else"] for r in head),
        },
    }
    _write_json(out_json, payload)
    return payload


def _stage6_init_needs_refresh(path: Path) -> bool:
    if not path.is_file():
        return True
    try:
        obj = _load_json(path)
    except Exception:
        return True
    step1 = obj.get("step1", {}) if isinstance(obj.get("step1", {}), dict) else {}
    head20 = obj.get("head20_mean", {}) if isinstance(obj.get("head20_mean", {}), dict) else {}
    return (not math.isfinite(_safe_float(step1.get("leg_over_nonleg")))) or (not math.isfinite(_safe_float(head20.get("leg_over_nonleg"))))


def _maybe_run_basetrain_eval(args: argparse.Namespace, cand: Candidate, paths: Paths) -> None:
    if bool(args.skip_basetrain_eval):
        return
    if paths.basetrain_group_json.is_file() and not bool(args.force_rerun):
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
            str(paths.basetrain_eval_dir),
            "--force",
        ],
        log_path=paths.lane_log,
    )
    _run_cmd(
        [
            sys.executable,
            str(GROUP_SUMMARY_TOOL),
            str(paths.basetrain_eval_json),
            "--cycle_gte",
            str(int(args.group_cycle_gte)),
            "--drop_wrap",
            "--out",
            str(paths.basetrain_group_json),
        ],
        log_path=paths.lane_log,
    )


def _maybe_run_stage6(args: argparse.Namespace, cand: Candidate, paths: Paths) -> None:
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


def _maybe_run_stage6_eval(args: argparse.Namespace, paths: Paths) -> None:
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


def _group_mean(path: Path | None, group: str) -> float:
    if path is None or (not path.is_file()):
        return float("nan")
    obj = _load_json(path)
    return _safe_float(obj.get("groups", {}).get(group, {}).get("mean"))


def _build_row(args: argparse.Namespace, cand: Candidate, paths: Paths) -> Dict[str, Any]:
    stage6_init = _load_json(paths.stage6_init_json)
    base_path = None if bool(args.skip_basetrain_eval) else paths.basetrain_group_json
    row = {
        "name": cand.name,
        "family": cand.family,
        "selector": cand.selector,
        "ckpt": str(cand.ckpt),
        "score": {
            "weights": {
                "all_ex_root": float(args.score_w_all),
                "leg": float(args.score_w_leg),
                "nonleg": float(args.score_w_nonleg),
            },
            "stage6_exit": {
                "all_ex_root_mean": _group_mean(paths.stage6_group_json, "all_ex_root"),
                "leg_mean": _group_mean(paths.stage6_group_json, "leg"),
                "nonleg_mean": _group_mean(paths.stage6_group_json, "nonleg"),
            },
        },
        "basetrain": None if base_path is None else {
            "all_ex_root_mean": _group_mean(base_path, "all_ex_root"),
            "leg_mean": _group_mean(base_path, "leg"),
            "nonleg_mean": _group_mean(base_path, "nonleg"),
            "arm_mean": _group_mean(base_path, "arm"),
            "else_mean": _group_mean(base_path, "else"),
        },
        "stage6_init": stage6_init,
        "stage6_exit": {
            "all_ex_root_mean": _group_mean(paths.stage6_group_json, "all_ex_root"),
            "leg_mean": _group_mean(paths.stage6_group_json, "leg"),
            "nonleg_mean": _group_mean(paths.stage6_group_json, "nonleg"),
            "arm_mean": _group_mean(paths.stage6_group_json, "arm"),
            "else_mean": _group_mean(paths.stage6_group_json, "else"),
        },
        "paths": {
            "lane_log": str(paths.lane_log),
            "basetrain_group_summary": None if base_path is None else str(base_path),
            "stage6_init_stats": str(paths.stage6_init_json),
            "stage6_group_summary": str(paths.stage6_group_json),
            "stage6_ckpt": str(paths.stage6_ckpt),
        },
    }
    exit_block = row["stage6_exit"]
    row["score"]["value"] = (
        float(args.score_w_all) * _safe_float(exit_block["all_ex_root_mean"])
        + float(args.score_w_leg) * _safe_float(exit_block["leg_mean"])
        + float(args.score_w_nonleg) * _safe_float(exit_block["nonleg_mean"])
    )
    return row


def _render_md(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Stage6 probe selector")
    lines.append("")
    lines.append(f"- run_tag: {summary['run_tag']}")
    lines.append(f"- stage6_config: `{summary['policy']['stage6_config']}`")
    lines.append(f"- score: `{summary['policy']['score_formula']}`")
    lines.append(f"- recommended: `{summary['recommended']['name']}`")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| rank | lane | selector | score | stage6 all_ex_root | stage6 leg | stage6 nonleg | step1 leg/nonleg | head20 leg/nonleg |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for item in summary["ranking"]:
        row = item["row"]
        s1 = row["stage6_init"]["step1"]
        h20 = row["stage6_init"]["head20_mean"]
        exit_block = row["stage6_exit"]
        lines.append(
            f"| {item['rank']} | {row['name']} | {row['selector'] or '-'} | {_fmt(_safe_float(row['score']['value']))} | {_fmt(_safe_float(exit_block['all_ex_root_mean']))} | {_fmt(_safe_float(exit_block['leg_mean']))} | {_fmt(_safe_float(exit_block['nonleg_mean']))} | {_fmt(_safe_float(s1['leg_over_nonleg']))} | {_fmt(_safe_float(h20['leg_over_nonleg']))} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run or reuse Stage6 probe outputs and build a downstream-aware handoff ranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--candidate", action="append", required=True, help="Repeatable spec: name=ckpt|family|selector")
    ap.add_argument("--run-tag", default=time.strftime("%Y%m%d"))
    ap.add_argument("--stage6-config", default=str(DEFAULT_STAGE6_CONFIG))
    ap.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    ap.add_argument("--encoder-bundle", default=str(DEFAULT_ENCODER_BUNDLE))
    ap.add_argument("--affine-stats", default=str(DEFAULT_AFFINE_STATS))
    ap.add_argument("--pretrain-clamp", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--time-index-mode", default="cycle")
    ap.add_argument("--phase-reset-source", default="none")
    ap.add_argument("--group-cycle-gte", type=int, default=1)
    ap.add_argument("--run-name-stem", default="stage6_probe")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--model-out-root", default="")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--skip-basetrain-eval", action="store_true")
    ap.add_argument("--score-w-all", type=float, default=1.0)
    ap.add_argument("--score-w-leg", type=float, default=1.5)
    ap.add_argument("--score-w-nonleg", type=float, default=0.5)
    args = ap.parse_args()

    args.stage6_config = _resolve(args.stage6_config)
    args.teacher = _resolve(args.teacher)
    args.encoder_bundle = _resolve(args.encoder_bundle)
    args.affine_stats = _resolve(args.affine_stats)
    out_root = _resolve(args.out_root) if str(args.out_root).strip() else (ROOT / "debug_output" / f"_tmp_stage6_probe_selector_{args.run_tag}")
    model_out_root = _resolve(args.model_out_root) if str(args.model_out_root).strip() else (ROOT / "models" / f"__tmp_stage6_probe_selector_{args.run_tag}")

    required = [args.stage6_config, args.teacher, args.encoder_bundle, args.affine_stats, GROUP_SUMMARY_TOOL]
    missing = [str(p) for p in required if not Path(p).is_file()]
    candidates = [_parse_candidate(spec) for spec in args.candidate]
    missing.extend(str(c.ckpt) for c in candidates if not c.ckpt.is_file())
    if missing:
        raise SystemExit("[FATAL] missing required files:\n" + "\n".join(missing))

    out_root.mkdir(parents=True, exist_ok=True)
    model_out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates, start=1):
        print(f"[lane {idx}/{len(candidates)}] {cand.name} -> {cand.ckpt}", flush=True)
        paths = _build_paths(
            out_root=out_root,
            model_out_root=model_out_root,
            name=cand.name,
            run_tag=str(args.run_tag),
            run_name_stem=str(args.run_name_stem),
        )
        paths.lane_root.mkdir(parents=True, exist_ok=True)
        paths.model_root.mkdir(parents=True, exist_ok=True)
        _maybe_run_basetrain_eval(args, cand, paths)
        _maybe_run_stage6(args, cand, paths)
        _maybe_run_stage6_eval(args, paths)
        rows.append(_build_row(args, cand, paths))

    rows.sort(key=lambda row: (_safe_float(row["score"]["value"]), _safe_float(row["stage6_exit"]["leg_mean"]), _safe_float(row["stage6_exit"]["all_ex_root_mean"]), row["name"]))
    ranking = [{"rank": i + 1, "row": row} for i, row in enumerate(rows)]
    summary = {
        "run_tag": str(args.run_tag),
        "policy": {
            "stage6_config": str(args.stage6_config),
            "teacher": str(args.teacher),
            "encoder_bundle": str(args.encoder_bundle),
            "affine_stats": str(args.affine_stats),
            "score_formula": f"{float(args.score_w_all):.3f}*all_ex_root + {float(args.score_w_leg):.3f}*leg + {float(args.score_w_nonleg):.3f}*nonleg",
        },
        "recommended": {"name": ranking[0]["row"]["name"], "score": ranking[0]["row"]["score"]["value"]},
        "ranking": ranking,
        "rows": rows,
    }
    _write_json(out_root / "selector_summary.json", summary)
    (out_root / "selector_summary.md").write_text(_render_md(summary), encoding="utf-8")
    print(f"[done] {out_root / 'selector_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
