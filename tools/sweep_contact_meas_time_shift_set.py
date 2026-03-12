#!/usr/bin/env python3
"""
Sweep `--pose_hist_time_shift` for contact_meas_head diagnostics over a teacher set.

Motivation:
  When contact_meas_head collapses under teacher-forced rollouts, one common hypothesis is
  that `pose_hist` (or its effective time alignment) is mismatched vs contact supervision.

This script:
  1) Runs `train/validate/run_teacher_rollout.py` for each time shift value.
  2) Computes:
     - contact_meas_head regime stats (left_support/right_support) via
       `tools/analyze_contact_meas_head.py`
     - lag / hysteresis metrics via `tools/analyze_contact_meas_lag.py`
  3) Aggregates weighted P(pred_L>pred_R | Lsup) across clips.

Example:
  python tools/sweep_contact_meas_time_shift_set.py \
    --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_teacher_lambda_cycles2_after_direct_pose_gatesup.pth \
    --teacher validate/teacher_batches/*.json \
    --encoder-bundle models/motion_encoder_equiv_stageA.pt --depth 3 \
    --pose-hist-source seq \
    --shifts -40..0 \
    --out-root debug_output/_tmp_teacher_debug/_pose_hist_time_shift_sweep \
    --force --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tools.analyze_contact_meas_head import analyze_teacher_pred_json  # type: ignore
    from tools.analyze_contact_meas_lag import analyze as analyze_contact_lag  # type: ignore
except Exception:  # pragma: no cover
    from analyze_contact_meas_head import analyze_teacher_pred_json  # type: ignore
    from analyze_contact_meas_lag import analyze as analyze_contact_lag  # type: ignore


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt_float(x: Any, *, digits: int = 3) -> str:
    v = _as_float(x)
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _expand_teacher_specs(specs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for spec in specs:
        if not spec:
            continue
        s = os.path.expanduser(str(spec))
        p = Path(s)
        matches: List[Path] = []
        if any(ch in s for ch in "*?[]"):
            matches = sorted(Path(".").glob(s))
        elif p.is_dir():
            matches = sorted(p.glob("*.json"))
        elif p.is_file():
            matches = [p]
        for m in matches:
            try:
                r = m.resolve()
            except Exception:
                r = m
            if r.is_file() and r not in seen:
                seen.add(r)
                out.append(r)
    return sorted(out)


def _parse_shifts(spec: str) -> List[int]:
    """
    Parse shift spec:
      - "a..b" (inclusive, step=1; supports descending ranges)
      - "a..b:step"
      - "a,b,c"
    """
    s = (spec or "").strip()
    if not s:
        return [0]
    if ".." in s:
        # range form
        if ":" in s:
            range_part, step_part = s.split(":", 1)
            step = int(step_part.strip())
        else:
            range_part, step = s, 1
        a_s, b_s = range_part.split("..", 1)
        a = int(a_s.strip())
        b = int(b_s.strip())
        step = int(step) if int(step) != 0 else 1
        if (b - a) * step < 0:
            step = -step
        vals: List[int] = []
        cur = a
        if step > 0:
            while cur <= b:
                vals.append(int(cur))
                cur += step
        else:
            while cur >= b:
                vals.append(int(cur))
                cur += step
        return vals
    # CSV form
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out if out else [0]


def _shift_dirname(shift: int) -> str:
    shift = int(shift)
    if shift == 0:
        return "shift0"
    if shift < 0:
        return f"shiftm{abs(shift)}"
    return f"shiftp{shift}"


def _load_ckpt_posttrain_cfg(ckpt_path: Path) -> Dict[str, Any]:
    if not ckpt_path.is_file() or ckpt_path.suffix.lower() != ".pth":
        return {}
    try:
        import torch  # noqa: WPS433
    except Exception:
        return {}
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    cfg = obj.get("posttrain_cfg", None)
    return cfg if isinstance(cfg, dict) else {}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_teacher_rollout(
    *,
    project_root: Path,
    model: Path,
    teacher_files: List[Path],
    out_dir: Path,
    bundle: Path,
    pretrain_template: Path,
    encoder_bundle: Path,
    npz_root: Path,
    device: str,
    depth: int,
    num_heads: int,
    dropout: float,
    context_len: int,
    pose_hist_source: str,
    pose_hist_time_shift: int,
    angvel_source: str,
    force: bool,
    quiet: bool,
) -> None:
    cmd: List[str] = [
        sys.executable,
        str(project_root / "train/validate/run_teacher_rollout.py"),
        "--model",
        str(model),
        "--teacher",
        *[str(p) for p in teacher_files],
        "--bundle",
        str(bundle),
        "--pretrain-template",
        str(pretrain_template),
        "--encoder-bundle",
        str(encoder_bundle),
        "--npz-root",
        str(npz_root),
        "--out",
        str(out_dir),
        "--device",
        str(device),
        "--depth",
        str(int(depth)),
        "--num-heads",
        str(int(num_heads)),
        "--dropout",
        str(float(dropout)),
        "--context-len",
        str(int(context_len)),
        "--pose_hist_source",
        str(pose_hist_source),
        "--pose_hist_time_shift",
        str(int(pose_hist_time_shift)),
        "--angvel_source",
        str(angvel_source),
        "--pose_hist_ablation",
        "none",
        "--angvel_ablation",
        "none",
    ]
    if force:
        cmd.append("--force")
    if quiet:
        cmd.append("--quiet")

    _ensure_dir(out_dir)
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"run_teacher_rollout failed (rc={proc.returncode}).\nCMD: {' '.join(cmd)}\n\n{proc.stdout}")


def _pred_files(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("*_teacher_pred.json"))


def _weighted_mean(pairs: List[Tuple[int, float]]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for w, v in pairs:
        if w <= 0:
            continue
        if v is None:
            continue
        num += float(w) * float(v)
        den += float(w)
    if den <= 0.0:
        return None
    return num / den


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep pose_hist_time_shift over a teacher set.")
    ap.add_argument("--model", type=str, required=True, help="Checkpoint (.pth) path.")
    ap.add_argument("--teacher", nargs="+", required=True, help="Teacher json files/dirs/globs (e.g. validate/teacher_batches/*.json).")
    ap.add_argument("--shifts", type=str, default="-40..0", help="Shift sweep spec: 'a..b[:step]' or 'a,b,c'.")
    ap.add_argument("--out-root", type=str, default="debug_output/_tmp_teacher_debug/_pose_hist_time_shift_sweep", help="Output root dir.")
    ap.add_argument("--tag", type=str, default=None, help="Optional subdir under --out-root.")

    ap.add_argument("--pose-hist-source", type=str, default="seq", choices=("seq", "buffer"), help="Pose hist source. Time shift is meaningful only for 'seq'.")
    ap.add_argument("--angvel-source", type=str, default="state", choices=("state", "seq"), help="Angvel source.")

    ap.add_argument("--max-lag", type=int, default=40, help="Lag search range in [-max_lag,+max_lag].")
    ap.add_argument("--on-th", type=float, default=0.8, help="Support ON threshold.")
    ap.add_argument("--off-th", type=float, default=0.1, help="Support OFF threshold.")

    ap.add_argument("--device", type=str, default=None, help="Override device (auto/cpu/cuda/mps).")
    ap.add_argument("--bundle", type=str, default=None, help="Override bundle_json (norm_template.json).")
    ap.add_argument("--pretrain-template", type=str, default=None, help="Override pretrain_template.")
    ap.add_argument("--encoder-bundle", type=str, default=None, help="Override encoder_bundle.")
    ap.add_argument("--npz-root", type=str, default=None, help="Override npz_root.")

    ap.add_argument("--depth", type=int, default=None, help="Override encoder depth (must match checkpoint).")
    ap.add_argument("--num-heads", type=int, default=None, help="Override num-heads (must match checkpoint).")
    ap.add_argument("--dropout", type=float, default=None, help="Override dropout (must match checkpoint).")
    ap.add_argument("--context-len", type=int, default=None, help="Override context-len (for record; must match checkpoint).")

    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"[FATAL] model not found: {model}")

    teacher_files = _expand_teacher_specs(args.teacher)
    if not teacher_files:
        raise SystemExit("[FATAL] --teacher expanded to empty file list.")

    ckpt_cfg = _load_ckpt_posttrain_cfg(model)
    depth = int(args.depth if args.depth is not None else ckpt_cfg.get("depth", 2))
    num_heads = int(args.num_heads if args.num_heads is not None else ckpt_cfg.get("num_heads", 4))
    dropout = float(args.dropout if args.dropout is not None else ckpt_cfg.get("dropout", 0.1))
    context_len = int(args.context_len if args.context_len is not None else ckpt_cfg.get("context_len", 16))

    device = str(args.device if args.device is not None else "auto")
    bundle = Path(args.bundle if args.bundle is not None else "raw_data/processed_data/norm_template.json").expanduser().resolve()
    pretrain_template = Path(args.pretrain_template if args.pretrain_template is not None else "models/pretrain_template.json").expanduser().resolve()
    encoder_bundle = Path(args.encoder_bundle if args.encoder_bundle is not None else "models/motion_encoder_equiv_stageA.pt").expanduser().resolve()
    npz_root = Path(args.npz_root if args.npz_root is not None else "raw_data/processed_data").expanduser().resolve()

    shifts = _parse_shifts(str(args.shifts))
    pose_hist_source = str(args.pose_hist_source or "seq").lower().strip()
    angvel_source = str(args.angvel_source or "state").lower().strip()

    tag = args.tag or f"depth{depth}_pose_{pose_hist_source}_ang_{angvel_source}"
    out_root = Path(args.out_root).expanduser().resolve() / tag
    _ensure_dir(out_root)

    # Avoid matplotlib cache pollution if downstream tools import it.
    os.environ.setdefault("MPLCONFIGDIR", str(out_root / "_mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_root / "_xdg_cache"))
    _ensure_dir(Path(os.environ["XDG_CACHE_HOME"]))

    meta = {
        "model": str(model),
        "teacher_files": [str(p) for p in teacher_files],
        "bundle": str(bundle),
        "pretrain_template": str(pretrain_template),
        "encoder_bundle": str(encoder_bundle),
        "npz_root": str(npz_root),
        "device": device,
        "depth": depth,
        "num_heads": num_heads,
        "dropout": dropout,
        "context_len": context_len,
        "pose_hist_source": pose_hist_source,
        "angvel_source": angvel_source,
        "shifts": shifts,
        "max_lag": int(args.max_lag),
        "thresholds": {"on": float(args.on_th), "off": float(args.off_th)},
    }
    (out_root / "sweep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    for shift in shifts:
        out_dir = out_root / _shift_dirname(int(shift))
        pred_list = _pred_files(out_dir)
        if not pred_list or bool(args.force):
            _run_teacher_rollout(
                project_root=project_root,
                model=model,
                teacher_files=teacher_files,
                out_dir=out_dir,
                bundle=bundle,
                pretrain_template=pretrain_template,
                encoder_bundle=encoder_bundle,
                npz_root=npz_root,
                device=device,
                depth=depth,
                num_heads=num_heads,
                dropout=dropout,
                context_len=context_len,
                pose_hist_source=pose_hist_source,
                pose_hist_time_shift=int(shift),
                angvel_source=angvel_source,
                force=bool(args.force),
                quiet=bool(args.quiet),
            )
            pred_list = _pred_files(out_dir)

        clip_rows: List[Dict[str, Any]] = []
        for pred_path in pred_list:
            meas = analyze_teacher_pred_json(pred_path, on_th=float(args.on_th), off_th=float(args.off_th), top_k=0)
            lag = analyze_contact_lag(pred_path, max_lag=int(args.max_lag), on_th=float(args.on_th), off_th=float(args.off_th))

            # Extract regime metrics
            regimes = {r.get("name"): r for r in meas.get("regimes", []) if isinstance(r, dict)}
            left = regimes.get("left_support", {}) if isinstance(regimes.get("left_support"), dict) else {}
            right = regimes.get("right_support", {}) if isinstance(regimes.get("right_support"), dict) else {}

            # Extract lag metrics (R fall tail)
            event = lag.get("event", {}) if isinstance(lag.get("event"), dict) else {}
            ev_r = event.get("R", {}) if isinstance(event.get("R"), dict) else {}
            fall_time = ev_r.get("falling_time", {}) if isinstance(ev_r.get("falling_time"), dict) else {}

            clip_rows.append(
                {
                    "clip": meas.get("clip"),
                    "T": meas.get("T"),
                    "json": str(pred_path),
                    "left_support_n": left.get("n"),
                    "left_support_p": left.get("p_pred_L_gt_R"),
                    "right_support_n": right.get("n"),
                    "right_support_p": right.get("p_pred_R_gt_L"),
                    "R_best_lag": ((lag.get("corr") or {}).get("R") or {}).get("best_lag"),
                    "R_best_corr": ((lag.get("corr") or {}).get("R") or {}).get("best_corr"),
                    "R_fall_pred_at_med": ((fall_time.get("pred_at_gt") or {}).get("median") if isinstance(fall_time.get("pred_at_gt"), dict) else None),
                    "R_fall_dt_mid_med": ((fall_time.get("time_to_le_mid") or {}).get("median") if isinstance(fall_time.get("time_to_le_mid"), dict) else None),
                    "R_fall_dt_on_med": ((fall_time.get("time_to_le_on") or {}).get("median") if isinstance(fall_time.get("time_to_le_on"), dict) else None),
                }
            )

        # Aggregate weighted by regime frame counts
        w_left = [(int(r.get("left_support_n") or 0), float(r.get("left_support_p") or 0.0)) for r in clip_rows]
        w_right = [(int(r.get("right_support_n") or 0), float(r.get("right_support_p") or 0.0)) for r in clip_rows]
        agg_left = _weighted_mean(w_left)
        agg_right = _weighted_mean(w_right)
        total_left = sum(int(r.get("left_support_n") or 0) for r in clip_rows)
        total_right = sum(int(r.get("right_support_n") or 0) for r in clip_rows)

        rows.append(
            {
                "shift": int(shift),
                "out_dir": str(out_dir),
                "total_left_support_n": int(total_left),
                "total_right_support_n": int(total_right),
                "weighted_p_L_gt_R": agg_left,
                "weighted_p_R_gt_L": agg_right,
                "clips": clip_rows,
            }
        )

    summary = {"meta": meta, "rows": rows}
    (out_root / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print compact table
    print(f"[SweepTimeShift] out_root={out_root}")
    print("| shift | N(Lsup) | P(L>R|Lsup) | N(Rsup) | P(R>L|Rsup) |")
    print("|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(r.get("shift")),
                    str(r.get("total_left_support_n")),
                    _fmt_float(r.get("weighted_p_L_gt_R"), digits=3),
                    str(r.get("total_right_support_n")),
                    _fmt_float(r.get("weighted_p_R_gt_L"), digits=3),
                ]
            )
            + " |"
        )
    print(f"[OK] wrote {out_root / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

