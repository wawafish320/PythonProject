#!/usr/bin/env python3
"""
Sweep pose_hist history-block ablations and quantify contact_meas_head lag.

Goal:
  - Reproduce/quantify temporal lag (phase delay / hysteresis proxy) induced by pose_hist
    in teacher-rollout, without changing weights or model structure.
  - Sweep keep_last K in {1..pose_hist_len} and compare pose_hist_source {buffer,seq}.

This script runs `train/validate/run_teacher_rollout.py` multiple times (one clip) and then
summarizes:
  - best_lag / best_corr for contacts_meas vs GT contacts (L/R + delta) via
    `tools/analyze_contact_meas_lag.py`
  - left_support regime stats (P(pred_L>pred_R|Lsup), etc) via
    `tools/analyze_contact_meas_head.py`

Example (Walk_F, buffer vs seq, K=1..3):
  python tools/sweep_contact_meas_hist_lag.py \\
    --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_teacher_lambda_cycles2_after_direct_pose_gatesup.pth \\
    --teacher validate/teacher_batches/Walk_F_teacher.json \\
    --out-root debug_output/_tmp_teacher_debug/_hist_ablation_sweep \\
    --force
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
    # When executed as `python tools/xxx.py`, sys.path[0] == "tools/", so import sibling directly.
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


def _fmt_int(x: Any, *, width: int = 0) -> str:
    try:
        v = int(x)
    except Exception:
        return "-".rjust(width) if width > 0 else "-"
    s = str(v)
    return s.rjust(width) if width > 0 else s


def _fmt_float(x: Any, *, digits: int = 3) -> str:
    v = _as_float(x)
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _parse_csv_ints(spec: str) -> List[int]:
    out: List[int] = []
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_csv_strs(spec: str) -> List[str]:
    out: List[str] = []
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(token)
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ckpt_posttrain_cfg(ckpt_path: Path) -> Dict[str, Any]:
    if not ckpt_path.is_file():
        return {}
    if ckpt_path.suffix.lower() != ".pth":
        return {}
    try:
        import torch
    except Exception:
        return {}
    try:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    cfg = obj.get("posttrain_cfg", None)
    return cfg if isinstance(cfg, dict) else {}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run_teacher_rollout(
    *,
    project_root: Path,
    model: Path,
    teacher: Path,
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
    pose_hist_ablation: str,
    pose_hist_keep_last: int,
    angvel_source: str,
    angvel_ablation: str,
    force: bool,
    quiet: bool,
) -> Path:
    cmd: List[str] = [
        sys.executable,
        str(project_root / "train/validate/run_teacher_rollout.py"),
        "--model",
        str(model),
        "--teacher",
        str(teacher),
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
        "--pose_hist_ablation",
        str(pose_hist_ablation),
        "--pose_hist_keep_last",
        str(int(pose_hist_keep_last)),
        "--angvel_source",
        str(angvel_source),
        "--angvel_ablation",
        str(angvel_ablation),
    ]
    if force:
        cmd.append("--force")
    if quiet:
        cmd.append("--quiet")

    _ensure_dir(out_dir)
    proc = subprocess.run(cmd, cwd=str(project_root), check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"run_teacher_rollout failed (rc={proc.returncode}).\nCMD: {' '.join(cmd)}\n\n{proc.stdout}")

    clip = _load_json(teacher).get("clip", None)
    if not isinstance(clip, str) or not clip:
        # Fall back: scan out_dir for any *_teacher_pred.json
        matches = sorted(out_dir.glob("*_teacher_pred.json"))
        if not matches:
            raise RuntimeError(f"run_teacher_rollout succeeded but no '*_teacher_pred.json' found in {out_dir}")
        return matches[0]

    pred_path = out_dir / f"{clip}_teacher_pred.json"
    if not pred_path.is_file():
        matches = sorted(out_dir.glob(f"{clip}*_teacher_pred.json"))
        if matches:
            return matches[0]
        raise RuntimeError(f"Expected output {pred_path} missing after run_teacher_rollout.")
    return pred_path


def _regime_by_name(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    regimes = summary.get("regimes", None)
    if not isinstance(regimes, list):
        return out
    for r in regimes:
        if not isinstance(r, dict):
            continue
        name = r.get("name", None)
        if isinstance(name, str):
            out[name] = r
    return out


def _compact_lag(lag_summary: Dict[str, Any]) -> Dict[str, Any]:
    corr = lag_summary.get("corr", {})
    out: Dict[str, Any] = {"T": lag_summary.get("T"), "C": lag_summary.get("C"), "max_lag": lag_summary.get("max_lag")}
    for key in ("L", "R", "L_delta", "R_delta"):
        block = corr.get(key) if isinstance(corr, dict) else None
        if not isinstance(block, dict):
            continue
        out[key] = {"best_lag": block.get("best_lag"), "best_corr": block.get("best_corr")}
    return out


def _compact_meas(meas_summary: Dict[str, Any]) -> Dict[str, Any]:
    overall = meas_summary.get("overall", {}) if isinstance(meas_summary.get("overall"), dict) else {}
    reg = _regime_by_name(meas_summary)
    left = reg.get("left_support", {})
    right = reg.get("right_support", {})
    return {
        "overall_pred_mean": overall.get("pred_mean"),
        "left_support_n": left.get("n"),
        "left_support_p_pred_L_gt_R": left.get("p_pred_L_gt_R"),
        "right_support_n": right.get("n"),
        "right_support_p_pred_R_gt_L": right.get("p_pred_R_gt_L"),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _maybe_import_matplotlib() -> Optional[Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return None
    return plt


def _plot_best_lag_vs_k(path: Path, rows: List[Dict[str, Any]], *, title: str = "R best_lag vs K") -> bool:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return False

    # Group by pose_hist_source
    by_src: Dict[str, List[Tuple[int, Optional[int]]]] = {}
    for r in rows:
        src = str(r.get("pose_hist_source", ""))
        k = _as_int(r.get("pose_hist_keep_last"), 0)
        lag = r.get("lag", {}).get("R", {}).get("best_lag") if isinstance(r.get("lag"), dict) else None
        lag_i = int(lag) if lag is not None else None
        by_src.setdefault(src, []).append((k, lag_i))

    plt.figure(figsize=(7.2, 4.2))
    for src, pts in sorted(by_src.items(), key=lambda kv: kv[0]):
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] if p[1] is not None else float("nan") for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=str(src))

    plt.axhline(0.0, color="k", linewidth=1.0, alpha=0.2)
    plt.grid(True, alpha=0.25)
    plt.xlabel("pose_hist_keep_last K")
    plt.ylabel("best_lag_R (frames)")
    plt.title(title)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep pose_hist keep_last and quantify contact_meas lag.")
    ap.add_argument("--model", type=str, required=True, help="Checkpoint (.pth) path.")
    ap.add_argument("--teacher", type=str, required=True, help="Teacher batch JSON (single clip).")

    ap.add_argument("--out-root", type=str, default="debug_output/_tmp_teacher_debug/_hist_ablation_sweep", help="Output root dir.")
    ap.add_argument("--tag", type=str, default=None, help="Optional subdir name under --out-root.")

    ap.add_argument(
        "--pose-hist-sources",
        type=str,
        default="buffer,seq",
        help="Comma-separated list of pose_hist_source values to compare.",
    )
    ap.add_argument(
        "--Ks",
        type=str,
        default="1,2,3",
        help="Comma-separated list of K values for --pose_hist_keep_last.",
    )

    ap.add_argument("--max-lag", type=int, default=30, help="Lag search range in [-max_lag,+max_lag].")
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
    ap.add_argument("--context-len", type=int, default=None, help="Override context-len (must match checkpoint).")

    ap.add_argument("--force", action="store_true", help="Overwrite existing rollout JSON files.")
    ap.add_argument("--quiet", action="store_true", help="Silence rollout script output.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model = Path(args.model).expanduser().resolve()
    teacher = Path(args.teacher).expanduser().resolve()
    if not teacher.is_file():
        raise SystemExit(f"[FATAL] teacher not found: {teacher}")
    if not model.is_file():
        raise SystemExit(f"[FATAL] model not found: {model}")

    teacher_meta = _load_json(teacher)
    clip = teacher_meta.get("clip", None)
    if not isinstance(clip, str) or not clip:
        clip = "unknown_clip"

    ckpt_cfg = _load_ckpt_posttrain_cfg(model)
    device = str(args.device or ckpt_cfg.get("device") or "auto")
    bundle = Path(args.bundle or ckpt_cfg.get("bundle_json") or "raw_data/processed_data/norm_template.json").expanduser().resolve()
    pretrain_template = Path(args.pretrain_template or ckpt_cfg.get("pretrain_template") or "models/pretrain_template.json").expanduser().resolve()
    encoder_bundle = Path(args.encoder_bundle or ckpt_cfg.get("encoder_bundle") or "models/motion_encoder_equiv.pt").expanduser().resolve()
    npz_root = Path(args.npz_root or ckpt_cfg.get("data") or "raw_data/processed_data").expanduser().resolve()
    depth = int(args.depth if args.depth is not None else ckpt_cfg.get("depth", 2))
    num_heads = int(args.num_heads if args.num_heads is not None else ckpt_cfg.get("num_heads", 4))
    dropout = float(args.dropout if args.dropout is not None else ckpt_cfg.get("dropout", 0.1))
    context_len = int(args.context_len if args.context_len is not None else ckpt_cfg.get("context_len", 16))

    pose_sources = _parse_csv_strs(str(args.pose_hist_sources))
    if not pose_sources:
        pose_sources = ["buffer", "seq"]
    Ks = _parse_csv_ints(str(args.Ks))
    if not Ks:
        Ks = [1, 2, 3]

    tag = args.tag or f"{clip}_depth{depth}"
    out_root = Path(args.out_root).expanduser().resolve() / tag
    _ensure_dir(out_root)

    # Matplotlib cache dir: avoid writing to ~/.matplotlib in sandboxed envs.
    os.environ.setdefault("MPLCONFIGDIR", str(out_root / "_mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_root / "_xdg_cache"))
    _ensure_dir(Path(os.environ["XDG_CACHE_HOME"]))

    meta = {
        "clip": clip,
        "teacher": str(teacher),
        "model": str(model),
        "bundle": str(bundle),
        "pretrain_template": str(pretrain_template),
        "encoder_bundle": str(encoder_bundle),
        "npz_root": str(npz_root),
        "device": device,
        "depth": depth,
        "num_heads": num_heads,
        "dropout": dropout,
        "context_len": context_len,
        "pose_hist_sources": pose_sources,
        "Ks": Ks,
        "max_lag": int(args.max_lag),
        "thresholds": {"on": float(args.on_th), "off": float(args.off_th)},
    }
    _write_json(out_root / "sweep_meta.json", meta)

    rows: List[Dict[str, Any]] = []
    for src in pose_sources:
        for k in Ks:
            out_dir = out_root / f"pose_{src}" / f"keep_last{k}"
            pred_path = out_dir / f"{clip}_teacher_pred.json"
            if not pred_path.is_file() or bool(args.force):
                pred_path = _run_teacher_rollout(
                    project_root=project_root,
                    model=model,
                    teacher=teacher,
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
                    pose_hist_source=src,
                    pose_hist_ablation="keep_last",
                    pose_hist_keep_last=int(k),
                    angvel_source="state",
                    angvel_ablation="none",
                    force=bool(args.force),
                    quiet=bool(args.quiet),
                )

            lag = analyze_contact_lag(pred_path, max_lag=int(args.max_lag))
            meas = analyze_teacher_pred_json(pred_path, on_th=float(args.on_th), off_th=float(args.off_th), top_k=0)
            row = {
                "pose_hist_source": src,
                "pose_hist_ablation": "keep_last",
                "pose_hist_keep_last": int(k),
                "pred_json": str(pred_path),
                "lag": _compact_lag(lag),
                "meas": _compact_meas(meas),
            }
            rows.append(row)

    # Write summary
    summary = {"meta": meta, "rows": rows}
    _write_json(out_root / "sweep_summary.json", summary)

    # Print compact table
    print(f"[Sweep] out_root={out_root} clip={clip} depth={depth} sources={pose_sources} Ks={Ks}")
    for src in pose_sources:
        for k in Ks:
            r = next((x for x in rows if x.get("pose_hist_source") == src and int(x.get("pose_hist_keep_last", -1)) == int(k)), None)
            if not r:
                continue
            lag_r = r.get("lag", {}).get("R", {}).get("best_lag") if isinstance(r.get("lag"), dict) else None
            lag_l = r.get("lag", {}).get("L", {}).get("best_lag") if isinstance(r.get("lag"), dict) else None
            p_lr = r.get("meas", {}).get("left_support_p_pred_L_gt_R") if isinstance(r.get("meas"), dict) else None
            pm = r.get("meas", {}).get("overall_pred_mean") if isinstance(r.get("meas"), dict) else None
            pm0 = pm[0] if isinstance(pm, list) and len(pm) >= 2 else None
            pm1 = pm[1] if isinstance(pm, list) and len(pm) >= 2 else None
            print(
                f"  src={src:6s} K={k}  best_lag(L,R)=({_fmt_int(lag_l, width=3)},{_fmt_int(lag_r, width=3)})"
                f"  pred_mean(L,R)=({_fmt_float(pm0)},{_fmt_float(pm1)})  P(L>R|Lsup)={_fmt_float(p_lr)}"
            )

    # Plot
    plot_ok = _plot_best_lag_vs_k(out_root / "best_lag_R_vs_K.png", rows, title=f"{clip}: R best_lag vs K")
    if plot_ok:
        print(f"[OK] wrote {out_root / 'best_lag_R_vs_K.png'}")
    else:
        print("[WARN] matplotlib unavailable; skip plot.")
    print(f"[OK] wrote {out_root / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
