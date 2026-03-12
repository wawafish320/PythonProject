#!/usr/bin/env python3
"""
Run a short seq_len A/B probe and compare Stage7 gradient-side asymmetry.

Pipeline per seq_len:
  1) (optional) short training via train.training_MPL
  2) run tools/diagnose_stage7_sampling_grad_closure.py
  3) collect global + rot_geo grad_ratio_r_over_l and write summary
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float_tag(x: float) -> str:
    s = f"{float(x):.6g}"
    s = s.replace("-", "m").replace(".", "p")
    s = re.sub(r"p0$", "", s)
    return s


def _run_and_tee(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd]\n")
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
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
        rc = int(proc.wait())
    if rc != 0:
        raise SystemExit(f"[FATAL] command failed (exit={rc}): {' '.join(cmd)} (log: {log_path})")


def _parse_seq_lens(spec: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for tok in str(spec or "").split(","):
        t = tok.strip()
        if not t:
            continue
        v = int(t)
        if v <= 0:
            raise ValueError(f"seq_len must be >0, got {v}")
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    if not out:
        raise ValueError("empty --seq-lens")
    return out


def _resolve_path_from_root(p: str) -> Path:
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else (_ROOT / q)


def _parse_ckpt_map(items: Optional[List[str]]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for raw in (items or []):
        s = str(raw).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"invalid --ckpt-for-seq-len entry: {s!r} (expected seq_len=/path/to/ckpt)")
        k, v = s.split("=", 1)
        seq_len = int(k.strip())
        ckpt = _resolve_path_from_root(v.strip())
        if not ckpt.is_file():
            raise FileNotFoundError(f"ckpt not found for seq_len={seq_len}: {ckpt}")
        out[int(seq_len)] = ckpt.resolve()
    return out


def _pick_ckpt(train_out: Path, run_name: str) -> Path:
    run_dir = train_out / run_name
    cands = [
        run_dir / f"ckpt_best_free_{run_name}.pth",
        run_dir / f"ckpt_last_{run_name}.pth",
        run_dir / f"ckpt_best_teacher_{run_name}.pth",
    ]
    for p in cands:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"missing checkpoint for run={run_name}; tried: " + ", ".join(str(x) for x in cands)
    )


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


@dataclass
class ProbeRow:
    seq_len: int
    run_name: str
    ckpt: Path
    diagnose_json: Path
    global_ratio: float
    rot_geo_ratio: float
    target_windows: int
    total_windows: int
    active_clips: List[str]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run short seq_len A/B probe: train(optional) + sampling_grad_closure + summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config-json", type=str, default="config/exp_phase_DirectBranch_v1_d1_noreset.json")
    ap.add_argument("--out", type=str, default=None, help="Training --out override.")
    ap.add_argument("--base-run-name", type=str, default=None, help="Base run name prefix.")
    ap.add_argument("--seq-lens", type=str, default="60,50")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--w-rot-vel", type=float, default=3.0)
    ap.add_argument("--aug-lr-swap-prob", type=float, default=0.5)
    ap.add_argument(
        "--train-config-override",
        action="append",
        default=None,
        help="Extra KEY=VALUE forwarded to train.training_MPL (repeatable).",
    )
    ap.add_argument("--skip-train", action="store_true", help="Skip training and use existing ckpts.")
    ap.add_argument(
        "--ckpt-for-seq-len",
        action="append",
        default=None,
        help="Map seq_len to ckpt path, format: 50=models/.../ckpt.pth (repeatable).",
    )

    ap.add_argument("--diagnose-script", type=str, default="tools/diagnose_stage7_sampling_grad_closure.py")
    ap.add_argument("--target-clip", type=str, default="Walk_F")
    ap.add_argument("--loss-branch", type=str, default="out", choices=("out", "out_direct"))
    ap.add_argument("--component-losses", type=str, default="rot_geo,rot_vel,direct_pose,direct_delta")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda", "mps"))

    ap.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Output root under debug_output. Default: debug_output/seq_len_grad_probe_YYYYMMDD_HHMMSS",
    )
    args = ap.parse_args()

    cfg_path = _resolve_path_from_root(args.config_json)
    if not cfg_path.is_file():
        raise SystemExit(f"[FATAL] config-json not found: {cfg_path}")
    cfg = _load_json(cfg_path)

    train_out = _resolve_path_from_root(args.out or cfg.get("out") or "models")
    base_run = str(args.base_run_name or cfg.get("run_name") or "run")

    if args.sweep_dir:
        sweep_dir = _resolve_path_from_root(args.sweep_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = _ROOT / "debug_output" / f"seq_len_grad_probe_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = _parse_seq_lens(args.seq_lens)
    ckpt_map = _parse_ckpt_map(args.ckpt_for_seq_len)

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_ROOT))
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

    rows: List[ProbeRow] = []

    for seq_len in seq_lens:
        run_name = (
            f"{base_run}__seq{int(seq_len)}"
            f"_w{_fmt_float_tag(float(args.w_rot_vel))}"
            f"_lrswap_p{_fmt_float_tag(float(args.aug_lr_swap_prob))}"
            f"_e{int(args.epochs)}"
        )
        run_dir = sweep_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_map.get(int(seq_len), None)
        if ckpt_path is None:
            if not args.skip_train:
                train_cmd = [
                    sys.executable,
                    "-m",
                    "train.training_MPL",
                    "--config_json",
                    str(cfg_path),
                    "--out",
                    str(train_out),
                    "--run_name",
                    run_name,
                    "--config_override",
                    f"seq_len={int(seq_len)}",
                    "--config_override",
                    f"epochs={int(args.epochs)}",
                    "--config_override",
                    f"w_rot_vel={float(args.w_rot_vel)}",
                    "--config_override",
                    f"aug_lr_swap_prob={float(args.aug_lr_swap_prob)}",
                ]
                for ov in (args.train_config_override or []):
                    s = str(ov).strip()
                    if s:
                        train_cmd += ["--config_override", s]
                _run_and_tee(train_cmd, cwd=_ROOT, env=env, log_path=run_dir / "train.log")
            ckpt_path = _pick_ckpt(train_out, run_name)

        if not ckpt_path.is_file():
            raise SystemExit(f"[FATAL] ckpt not found for seq_len={seq_len}: {ckpt_path}")

        diag_out = run_dir / "sampling_grad_closure"
        diag_cmd = [
            sys.executable,
            str(_resolve_path_from_root(args.diagnose_script)),
            "--config-json",
            str(cfg_path),
            "--ckpt",
            str(ckpt_path),
            "--target-clip",
            str(args.target_clip),
            "--seq-len",
            str(int(seq_len)),
            "--depth",
            str(int(args.depth)),
            "--bundle",
            str(_resolve_path_from_root(args.bundle)),
            "--pretrain-template",
            str(_resolve_path_from_root(args.pretrain_template)),
            "--encoder-bundle",
            str(_resolve_path_from_root(args.encoder_bundle)),
            "--loss-branch",
            str(args.loss_branch),
            "--component-losses",
            str(args.component_losses),
            "--device",
            str(args.device),
            "--out-dir",
            str(diag_out),
        ]
        if int(args.max_windows) > 0:
            diag_cmd += ["--max-windows", str(int(args.max_windows))]
        _run_and_tee(diag_cmd, cwd=_ROOT, env=env, log_path=run_dir / "diagnose.log")

        diag_json = diag_out / "sampling_grad_closure.json"
        if not diag_json.is_file():
            raise SystemExit(f"[FATAL] diagnose output missing: {diag_json}")

        payload = _load_json(diag_json)
        g = payload.get("gradient", {}).get("global", {})
        comp_g = payload.get("component_gradient", {}).get("global", {}).get("rot_geo", {})
        ds = payload.get("dataset", {})
        clip_windows = ds.get("clip_window_counts", {}) if isinstance(ds, dict) else {}
        active_clips = [
            str(k)
            for k, v in clip_windows.items()
            if isinstance(v, (int, float)) and int(v) > 0
        ]

        rows.append(
            ProbeRow(
                seq_len=int(seq_len),
                run_name=run_name,
                ckpt=ckpt_path.resolve(),
                diagnose_json=diag_json.resolve(),
                global_ratio=_safe_float(g.get("grad_ratio_r_over_l", float("nan"))),
                rot_geo_ratio=_safe_float(comp_g.get("grad_ratio_r_over_l", float("nan"))),
                target_windows=int(ds.get("target_windows", 0) or 0),
                total_windows=int(ds.get("total_windows", 0) or 0),
                active_clips=active_clips,
            )
        )

    summary_rows: List[Dict[str, Any]] = []
    baseline_global = rows[0].global_ratio if rows else float("nan")
    baseline_rot_geo = rows[0].rot_geo_ratio if rows else float("nan")
    for r in rows:
        summary_rows.append(
            {
                "seq_len": int(r.seq_len),
                "run_name": r.run_name,
                "ckpt": str(r.ckpt),
                "diagnose_json": str(r.diagnose_json),
                "global_ratio_r_over_l": r.global_ratio,
                "rot_geo_ratio_r_over_l": r.rot_geo_ratio,
                "delta_global_vs_first": (_safe_float(r.global_ratio) - _safe_float(baseline_global)),
                "delta_rot_geo_vs_first": (_safe_float(r.rot_geo_ratio) - _safe_float(baseline_rot_geo)),
                "target_windows": int(r.target_windows),
                "total_windows": int(r.total_windows),
                "active_clips": list(r.active_clips),
            }
        )

    summary = {
        "config_json": str(cfg_path),
        "train_out": str(train_out),
        "base_run_name": base_run,
        "epochs": int(args.epochs),
        "w_rot_vel": float(args.w_rot_vel),
        "aug_lr_swap_prob": float(args.aug_lr_swap_prob),
        "loss_branch": str(args.loss_branch),
        "seq_lens": [int(x) for x in seq_lens],
        "rows": summary_rows,
    }

    out_json = sweep_dir / "summary.json"
    out_md = sweep_dir / "summary.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# seq_len gradient-side probe")
    lines.append("")
    lines.append(f"- config_json: `{cfg_path}`")
    lines.append(f"- train_out: `{train_out}`")
    lines.append(
        f"- epochs={int(args.epochs)}, w_rot_vel={float(args.w_rot_vel)}, aug_lr_swap_prob={float(args.aug_lr_swap_prob)}"
    )
    lines.append(f"- loss_branch: `{args.loss_branch}`")
    lines.append("")
    lines.append("|seq_len|global R/L|rot_geo R/L|delta global vs first|delta rot_geo vs first|target_windows|total_windows|active_clips|")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|:--|")
    for r in summary_rows:
        lines.append(
            "|{seq_len}|{g:.6f}|{rg:.6f}|{dg:+.6f}|{drg:+.6f}|{tw}|{allw}|{clips}|".format(
                seq_len=int(r["seq_len"]),
                g=_safe_float(r["global_ratio_r_over_l"]),
                rg=_safe_float(r["rot_geo_ratio_r_over_l"]),
                dg=_safe_float(r["delta_global_vs_first"]),
                drg=_safe_float(r["delta_rot_geo_vs_first"]),
                tw=int(r["target_windows"]),
                allw=int(r["total_windows"]),
                clips=", ".join(r.get("active_clips", [])),
            )
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for r in summary_rows:
        lines.append(f"- seq_len={int(r['seq_len'])}: ckpt=`{r['ckpt']}`")
        lines.append(f"  - diagnose_json=`{r['diagnose_json']}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_md}")


if __name__ == "__main__":
    main()
