#!/usr/bin/env python3
"""
Diagnose direct-pose generalization gap with per-SIC per-bone teacher-forcing metrics.

Two usage modes:

1) Analyze existing run_freerun_cycles JSONs:
   python tools/diagnose_direct_pose_train_eval_sic_gap.py \
     --train-json debug_output/train_xgt/*_freerun_cycles.json \
     --eval-json  debug_output/eval_xgt/*_freerun_cycles.json \
     --out debug_output/__tmp_directpose_diag

2) Auto-run x_gt rollouts first (then analyze):
   python tools/diagnose_direct_pose_train_eval_sic_gap.py \
     --model models/.../ckpt_last_*.pth \
     --train-teacher validate/teacher_batches/*.json \
     --eval-teacher validate/teacher_batches/*.json \
     --out debug_output/__tmp_directpose_diag

Notes:
- The script reads per_step_direct_geolocal_deg.DirectGeoLocalDeg (deg).
- If step_in_cycle is missing, it falls back to step % cycle_len, then step.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LEG_BONES_DEFAULT = [
    "thigh_r",
    "calf_r",
    "foot_r",
    "ball_r",
    "thigh_l",
    "calf_l",
    "foot_l",
    "ball_l",
]


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _split_specs(raw_specs: Sequence[str]) -> List[str]:
    toks: List[str] = []
    for raw in raw_specs:
        s = str(raw or "").strip()
        if not s:
            continue
        for part in s.replace(";", ",").split(","):
            t = part.strip()
            if t:
                toks.append(t)
    return toks


def _expand_specs(raw_specs: Sequence[str], *, file_suffix: str = "*.json") -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for tok in _split_specs(raw_specs):
        p = Path(tok).expanduser()
        matches: List[Path] = []
        if any(ch in tok for ch in "*?[]"):
            matches = [Path(x).expanduser() for x in glob.glob(tok)]
        elif p.is_dir():
            matches = sorted(p.glob(file_suffix))
        elif p.is_file():
            matches = [p]
        if not matches and p.parent.exists() and any(ch in p.name for ch in "*?[]"):
            matches = sorted(p.parent.glob(p.name))
        for m in matches:
            try:
                r = m.resolve()
            except Exception:
                r = m
            if r.is_file() and r not in seen:
                seen.add(r)
                out.append(r)
    return sorted(out)


def _parse_int_set(spec: str) -> Optional[set[int]]:
    s = str(spec or "").strip()
    if not s:
        return None
    out: set[int] = set()
    for tok in s.replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        if "-" in t:
            a, b = [x.strip() for x in t.split("-", 1)]
            if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            continue
        if t.lstrip("-").isdigit():
            out.add(int(t))
    return out if out else None


def _resolve_bones(spec: str, *, bone_names: List[str], root_idx: int) -> List[str]:
    s = str(spec or "").strip().lower()
    if not s:
        s = "leg"
    if s in ("leg", "legs"):
        keep = [b for b in LEG_BONES_DEFAULT if b in set(bone_names)]
        return [b for b in keep if bone_names.index(b) != int(root_idx)]
    if s in ("all", "*"):
        return [b for i, b in enumerate(bone_names) if int(i) != int(root_idx)]
    want = [x.strip() for x in str(spec).split(",") if x.strip()]
    out: List[str] = []
    seen: set[str] = set()
    for b in want:
        if b in bone_names and b not in seen and bone_names.index(b) != int(root_idx):
            out.append(b)
            seen.add(b)
    return out


def _npz_scalar_to_str(v: Any) -> str:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    return v if isinstance(v, str) else str(v)


def _npz_scalar_to_json_dict(v: Any) -> Dict[str, Any]:
    text = _npz_scalar_to_str(v)
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _build_teacher_from_npz(npz_paths: Sequence[Path], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=True) as d:
            x = d.get("X_norm")
            if x is None:
                x = d.get("x_in_features")
            c = d.get("cond_in")
            if x is None or c is None:
                raise RuntimeError(f"{npz_path}: missing X_norm/x_in_features or cond_in")
            x = np.asarray(x, dtype=np.float32)
            c = np.asarray(c, dtype=np.float32)
            if x.ndim != 2 or c.ndim != 2 or x.shape[0] != c.shape[0]:
                raise RuntimeError(f"{npz_path}: invalid X/cond shape {x.shape} / {c.shape}")

            source_json = ""
            if "source_json" in d:
                source_json = _npz_scalar_to_str(d["source_json"])

            state_layout = {}
            output_layout = {}
            if "state_layout_json" in d:
                state_layout = _npz_scalar_to_json_dict(d["state_layout_json"])
            if "output_layout_json" in d:
                output_layout = _npz_scalar_to_json_dict(d["output_layout_json"])
            fps = 60.0
            if "FPS" in d:
                fps = float(np.asarray(d["FPS"]).reshape(()))

        clip = npz_path.stem
        payload = {
            "clip": clip,
            "source_json": source_json,
            "fps": float(fps),
            "num_pairs": int(x.shape[0]),
            "layouts": {"state_layout": state_layout, "output_layout": output_layout},
            "teacher": {"state_norm": x.tolist(), "cond": c.tolist()},
        }
        out_path = out_dir / f"{clip}_teacher_from_npz.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        out.append(out_path)
    return sorted(out)


def _infer_train_npz_from_model(model_path: Path) -> List[Path]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for --infer-train-from-model") from exc
    ck = torch.load(model_path, map_location="cpu")
    if not isinstance(ck, dict):
        return []
    cfg = ck.get("posttrain_cfg")
    if not isinstance(cfg, dict):
        return []

    specs: List[str] = []
    train_files = cfg.get("train_files")
    if isinstance(train_files, str) and train_files.strip():
        specs.extend(_split_specs([train_files]))
    elif isinstance(train_files, (list, tuple)):
        for x in train_files:
            specs.extend(_split_specs([str(x)]))
    if not specs:
        data_dir = str(cfg.get("data") or "").strip()
        if data_dir:
            specs.append(str(Path(data_dir).expanduser() / "*.npz"))
    if not specs:
        return []
    return _expand_specs(specs, file_suffix="*.npz")


def _run_xgt_rollout(
    *,
    split_name: str,
    model: Path,
    teacher_files: Sequence[Path],
    out_dir: Path,
    bundle: Path,
    pretrain_template: Path,
    encoder_bundle: Path,
    npz_root: Path,
    rounds: int,
    depth: int,
    phase_reset_source: str,
    device: str,
    extra_args: Sequence[str],
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
    ]
    cmd.extend([str(p) for p in teacher_files])
    cmd.extend(
        [
            "--model",
            str(model),
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
            "--rounds",
            str(int(rounds)),
            "--depth",
            str(int(depth)),
            "--time-index-mode",
            "auto",
            "--phase_reset_source",
            str(phase_reset_source),
            "--phase_reset_source_strict",
            "on",
            "--freerun_x_gt",
            "--so3_corr_apply",
            "--lambda_fusion_apply",
            "--export_joint_direct_geolocal_series",
            "--force",
        ]
    )
    if device and device != "auto":
        cmd.extend(["--device", str(device)])
    for raw in extra_args:
        cmd.extend(shlex.split(str(raw)))

    print(f"[{split_name}] run x_gt rollout: {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-60:])
        raise RuntimeError(f"[{split_name}] rollout failed (exit={proc.returncode})\n{tail}")

    jsons = sorted(out_dir.glob("*_freerun_cycles.json"))
    if not jsons:
        raise RuntimeError(f"[{split_name}] no *_freerun_cycles.json found under {out_dir}")
    print(f"[{split_name}] rollout done: {len(jsons)} files")
    return jsons


@dataclass
class SplitStats:
    split: str
    files: List[Path]
    bone_names: List[str]
    sics: List[int]
    mean: np.ndarray  # (S, B)
    count: np.ndarray  # (S, B)
    pair_mean: Dict[Tuple[int, str], float]
    pair_count: Dict[Tuple[int, str], int]
    overall_mean: float
    overall_count: int


def _aggregate_split(
    *,
    split: str,
    files: Sequence[Path],
    bones_spec: str,
    sics_filter: Optional[set[int]],
    cycle_gte: int,
    drop_wrap: bool,
) -> SplitStats:
    if not files:
        raise RuntimeError(f"[{split}] empty file list")

    canonical_bone_names: Optional[List[str]] = None
    canonical_root: Optional[int] = None
    selected_bones: Optional[List[str]] = None
    selected_idx: Optional[List[int]] = None

    sums: Dict[Tuple[int, int], float] = defaultdict(float)
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    total_sum = 0.0
    total_cnt = 0
    sics_seen: set[int] = set()

    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        per = obj.get("per_step_direct_geolocal_deg")
        if not isinstance(per, dict):
            raise RuntimeError(f"{path}: missing per_step_direct_geolocal_deg (rerun with --export_joint_direct_geolocal_series)")
        names = list(per.get("bone_names") or [])
        if not names:
            raise RuntimeError(f"{path}: missing bone_names")
        root = _safe_int(per.get("root_idx"), 0) or 0
        mat = np.asarray(per.get("DirectGeoLocalDeg"), dtype=np.float64)
        if mat.ndim != 2 or mat.shape[1] != len(names):
            raise RuntimeError(f"{path}: invalid DirectGeoLocalDeg shape={mat.shape}, expected (*,{len(names)})")
        mps = obj.get("metrics_per_step")
        if not isinstance(mps, list) or not mps:
            raise RuntimeError(f"{path}: missing metrics_per_step")
        n = min(int(mat.shape[0]), len(mps))
        if n <= 0:
            continue

        if canonical_bone_names is None:
            canonical_bone_names = names
            canonical_root = int(root)
            selected_bones = _resolve_bones(bones_spec, bone_names=names, root_idx=root)
            if not selected_bones:
                raise RuntimeError(f"[{split}] no bones selected from spec='{bones_spec}'")
            selected_idx = [names.index(b) for b in selected_bones]
        else:
            if names != canonical_bone_names or int(root) != int(canonical_root):
                raise RuntimeError(f"{path}: bone_names/root_idx mismatch across files; provide homogeneous inputs per split")

        cycle_len = _safe_int(obj.get("cycle_len"), 0) or 0
        assert selected_idx is not None
        for i in range(n):
            st = mps[i] if isinstance(mps[i], dict) else {}
            cyc = _safe_int(st.get("cycle"))
            if cyc is not None and int(cyc) < int(cycle_gte):
                continue
            if drop_wrap and bool(st.get("wrap_boundary_step", False)):
                continue
            step = _safe_int(st.get("step"), i) or i
            sic = _safe_int(st.get("step_in_cycle"))
            if sic is None:
                if cycle_len > 0:
                    sic = int(step % cycle_len)
                else:
                    sic = int(step)
            sic = int(sic)
            if sics_filter is not None and sic not in sics_filter:
                continue

            row = mat[i]
            got_any = False
            for bpos, j in enumerate(selected_idx):
                if j < 0 or j >= row.shape[0]:
                    continue
                v = float(row[j])
                if not math.isfinite(v):
                    continue
                sums[(sic, bpos)] += v
                counts[(sic, bpos)] += 1
                total_sum += v
                total_cnt += 1
                got_any = True
            if got_any:
                sics_seen.add(sic)

    if canonical_bone_names is None or selected_bones is None:
        raise RuntimeError(f"[{split}] no valid data rows")
    if total_cnt <= 0 or not sics_seen:
        raise RuntimeError(f"[{split}] mask removed all samples (cycle_gte={cycle_gte}, drop_wrap={drop_wrap})")

    sics = sorted(sics_seen)
    B = len(selected_bones)
    mean = np.full((len(sics), B), np.nan, dtype=np.float64)
    cnt = np.zeros((len(sics), B), dtype=np.int64)
    pair_mean: Dict[Tuple[int, str], float] = {}
    pair_count: Dict[Tuple[int, str], int] = {}

    for si, sic in enumerate(sics):
        for bi, bone in enumerate(selected_bones):
            c = int(counts.get((sic, bi), 0))
            cnt[si, bi] = c
            pair_count[(int(sic), str(bone))] = c
            if c > 0:
                m = float(sums[(sic, bi)] / c)
                mean[si, bi] = m
                pair_mean[(int(sic), str(bone))] = m

    return SplitStats(
        split=str(split),
        files=list(files),
        bone_names=list(selected_bones),
        sics=sics,
        mean=mean,
        count=cnt,
        pair_mean=pair_mean,
        pair_count=pair_count,
        overall_mean=float(total_sum / max(1, total_cnt)),
        overall_count=int(total_cnt),
    )


def _write_matrix_csv(path: Path, *, sics: Sequence[int], bones: Sequence[str], mat: np.ndarray, digits: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sic", *bones])
        for i, sic in enumerate(sics):
            row: List[Any] = [int(sic)]
            for j in range(len(bones)):
                v = float(mat[i, j])
                row.append("" if not math.isfinite(v) else f"{v:.{digits}f}")
            w.writerow(row)


def _write_count_csv(path: Path, *, sics: Sequence[int], bones: Sequence[str], mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sic", *bones])
        for i, sic in enumerate(sics):
            row = [int(sic), *[int(mat[i, j]) for j in range(len(bones))]]
            w.writerow(row)


def _row_weighted_mean(vals: Sequence[float], weights: Sequence[int]) -> Tuple[float, int]:
    acc = 0.0
    cnt = 0
    for v, w in zip(vals, weights):
        if not math.isfinite(float(v)) or int(w) <= 0:
            continue
        acc += float(v) * int(w)
        cnt += int(w)
    if cnt <= 0:
        return float("nan"), 0
    return float(acc / cnt), cnt


def _maybe_plot_heatmap(path: Path, *, title: str, mat: np.ndarray, sics: Sequence[int], bones: Sequence[str], center_zero: bool) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except Exception as exc:
        return f"matplotlib unavailable ({exc})"

    if mat.size <= 0:
        return "empty matrix"
    arr = np.asarray(mat, dtype=np.float64)
    m = np.ma.masked_invalid(arr)
    if m.count() <= 0:
        return "all-NaN matrix"

    width = max(8.0, 0.8 * max(1, len(bones)))
    height = max(4.5, 0.22 * max(1, len(sics)))
    fig, ax = plt.subplots(figsize=(width, height))
    if center_zero:
        vmax = float(np.nanmax(np.abs(arr)))
        vmax = max(vmax, 1e-6)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="coolwarm", norm=norm, interpolation="nearest")
    else:
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("deg")

    ax.set_title(title)
    ax.set_xlabel("bone")
    ax.set_ylabel("step_in_cycle (sic)")

    xt = np.arange(len(bones), dtype=np.int64)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(b) for b in bones], rotation=70, ha="right")

    if len(sics) <= 24:
        yt = np.arange(len(sics), dtype=np.int64)
    else:
        stride = int(math.ceil(len(sics) / 24.0))
        yt = np.arange(0, len(sics), stride, dtype=np.int64)
    ax.set_yticks(yt)
    ax.set_yticklabels([str(int(sics[i])) for i in yt])

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return None


def _align_gap(train: SplitStats, eval_: SplitStats) -> Tuple[List[int], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bones = [b for b in train.bone_names if b in set(eval_.bone_names)]
    if not bones:
        raise RuntimeError("train/eval have no common selected bones")
    sics = sorted(set(train.sics) | set(eval_.sics))

    tr = np.full((len(sics), len(bones)), np.nan, dtype=np.float64)
    ev = np.full((len(sics), len(bones)), np.nan, dtype=np.float64)
    gp = np.full((len(sics), len(bones)), np.nan, dtype=np.float64)
    tr_n = np.zeros((len(sics), len(bones)), dtype=np.int64)
    ev_n = np.zeros((len(sics), len(bones)), dtype=np.int64)

    for si, sic in enumerate(sics):
        for bi, bone in enumerate(bones):
            kt = (int(sic), str(bone))
            if kt in train.pair_mean:
                tr[si, bi] = float(train.pair_mean[kt])
                tr_n[si, bi] = int(train.pair_count.get(kt, 0))
            if kt in eval_.pair_mean:
                ev[si, bi] = float(eval_.pair_mean[kt])
                ev_n[si, bi] = int(eval_.pair_count.get(kt, 0))
            if math.isfinite(float(tr[si, bi])) and math.isfinite(float(ev[si, bi])):
                gp[si, bi] = float(ev[si, bi] - tr[si, bi])
    return sics, bones, tr, ev, gp, tr_n, ev_n


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-SIC per-bone train-vs-eval teacher-forcing diagnostics for direct pose head.")
    ap.add_argument("--train-json", nargs="*", default=[], help="Existing train split *_freerun_cycles.json / dirs / globs.")
    ap.add_argument("--eval-json", nargs="*", default=[], help="Existing eval split *_freerun_cycles.json / dirs / globs.")

    ap.add_argument("--model", type=str, default="", help="Checkpoint path (required only when auto-running rollouts).")
    ap.add_argument("--train-teacher", nargs="*", default=[], help="Teacher JSON specs for train split.")
    ap.add_argument("--eval-teacher", nargs="*", default=[], help="Teacher JSON specs for eval split.")
    ap.add_argument("--train-npz", nargs="*", default=[], help="NPZ specs to auto-build teacher JSONs for train split.")
    ap.add_argument("--eval-npz", nargs="*", default=[], help="NPZ specs to auto-build teacher JSONs for eval split.")
    ap.add_argument("--infer-train-from-model", action="store_true", help="Infer train NPZ set from checkpoint posttrain_cfg.")
    ap.add_argument("--eval-default-teacher", type=str, default="validate/teacher_batches/*.json", help="Fallback eval teacher specs when --eval-* not provided.")

    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--rounds", type=int, default=2, help="x_gt rollout rounds (>=2 keeps step_in_cycle in metrics_per_step).")
    ap.add_argument("--depth", type=int, default=3, help="run_freerun_cycles depth argument.")
    ap.add_argument("--phase-reset-source", type=str, default="none")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--rollout-extra-arg", action="append", default=[], help="Extra arg string passed to run_freerun_cycles (repeatable).")

    ap.add_argument("--bones", type=str, default="leg", help="Bones to aggregate: leg|all|csv.")
    ap.add_argument("--sics", type=str, default="", help="Optional SIC filter: e.g. 12-15,49-55.")
    ap.add_argument("--focus-bones", type=str, default="foot_l,foot_r,ball_l,ball_r", help="Focus bones for summary table.")
    ap.add_argument("--focus-sics", type=str, default="12-15", help="Focus SICs for summary table.")
    ap.add_argument("--cycle-gte", type=int, default=1, help="Keep metrics_per_step.cycle >= N when available.")
    ap.add_argument("--drop-wrap", type=str, default="on", choices=["on", "off"], help="Drop wrap_boundary_step.")
    ap.add_argument("--topk", type=int, default=20, help="Top-K rows in markdown summaries.")
    ap.add_argument("--no-heatmap", action="store_true")
    ap.add_argument("--out", type=str, default="debug_output/__tmp_direct_pose_train_eval_diag")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_json = _expand_specs(args.train_json, file_suffix="*_freerun_cycles.json")
    eval_json = _expand_specs(args.eval_json, file_suffix="*_freerun_cycles.json")
    need_train_rollout = len(train_json) == 0
    need_eval_rollout = len(eval_json) == 0

    model_path = Path(args.model).expanduser().resolve() if args.model else None
    if (need_train_rollout or need_eval_rollout) and (model_path is None or not model_path.is_file()):
        raise SystemExit("[FATAL] missing --model for rollout mode (or provide --train-json/--eval-json directly).")

    if need_train_rollout:
        train_teacher = _expand_specs(args.train_teacher, file_suffix="*.json")
        if not train_teacher:
            train_npz = _expand_specs(args.train_npz, file_suffix="*.npz")
            if not train_npz and bool(args.infer_train_from_model):
                assert model_path is not None
                train_npz = _infer_train_npz_from_model(model_path)
                if train_npz:
                    print(f"[train] inferred npz from checkpoint: {len(train_npz)} files")
            if train_npz:
                tmp_teacher_dir = out_dir / "_tmp_teacher_train_from_npz"
                train_teacher = _build_teacher_from_npz(train_npz, tmp_teacher_dir)
                print(f"[train] built teacher jsons from npz: {len(train_teacher)} files")
        if not train_teacher:
            raise SystemExit("[FATAL] train split unresolved: provide --train-json, --train-teacher, --train-npz, or --infer-train-from-model.")
        train_json = _run_xgt_rollout(
            split_name="train",
            model=model_path,
            teacher_files=train_teacher,
            out_dir=out_dir / "rollout_train_xgt",
            bundle=Path(args.bundle).expanduser().resolve(),
            pretrain_template=Path(args.pretrain_template).expanduser().resolve(),
            encoder_bundle=Path(args.encoder_bundle).expanduser().resolve(),
            npz_root=Path(args.npz_root).expanduser().resolve(),
            rounds=int(args.rounds),
            depth=int(args.depth),
            phase_reset_source=str(args.phase_reset_source),
            device=str(args.device),
            extra_args=args.rollout_extra_arg,
        )

    if need_eval_rollout:
        eval_teacher = _expand_specs(args.eval_teacher, file_suffix="*.json")
        if not eval_teacher:
            eval_npz = _expand_specs(args.eval_npz, file_suffix="*.npz")
            if eval_npz:
                tmp_teacher_dir = out_dir / "_tmp_teacher_eval_from_npz"
                eval_teacher = _build_teacher_from_npz(eval_npz, tmp_teacher_dir)
                print(f"[eval] built teacher jsons from npz: {len(eval_teacher)} files")
        if not eval_teacher and str(args.eval_default_teacher or "").strip():
            eval_teacher = _expand_specs([args.eval_default_teacher], file_suffix="*.json")
            if eval_teacher:
                print(f"[eval] using default teacher specs: {len(eval_teacher)} files")
        if not eval_teacher:
            raise SystemExit("[FATAL] eval split unresolved: provide --eval-json/--eval-teacher/--eval-npz (or keep --eval-default-teacher).")
        eval_json = _run_xgt_rollout(
            split_name="eval",
            model=model_path,
            teacher_files=eval_teacher,
            out_dir=out_dir / "rollout_eval_xgt",
            bundle=Path(args.bundle).expanduser().resolve(),
            pretrain_template=Path(args.pretrain_template).expanduser().resolve(),
            encoder_bundle=Path(args.encoder_bundle).expanduser().resolve(),
            npz_root=Path(args.npz_root).expanduser().resolve(),
            rounds=int(args.rounds),
            depth=int(args.depth),
            phase_reset_source=str(args.phase_reset_source),
            device=str(args.device),
            extra_args=args.rollout_extra_arg,
        )

    sics_filter = _parse_int_set(args.sics)
    focus_sics = _parse_int_set(args.focus_sics)
    drop_wrap = str(args.drop_wrap).lower() == "on"

    train_stats = _aggregate_split(
        split="train",
        files=train_json,
        bones_spec=str(args.bones),
        sics_filter=sics_filter,
        cycle_gte=int(args.cycle_gte),
        drop_wrap=drop_wrap,
    )
    eval_stats = _aggregate_split(
        split="eval",
        files=eval_json,
        bones_spec=str(args.bones),
        sics_filter=sics_filter,
        cycle_gte=int(args.cycle_gte),
        drop_wrap=drop_wrap,
    )

    sics, bones, tr, ev, gp, tr_n, ev_n = _align_gap(train_stats, eval_stats)

    _write_matrix_csv(out_dir / "train_per_sic_bone_mean.csv", sics=sics, bones=bones, mat=tr)
    _write_matrix_csv(out_dir / "eval_per_sic_bone_mean.csv", sics=sics, bones=bones, mat=ev)
    _write_matrix_csv(out_dir / "gap_eval_minus_train_per_sic_bone_mean.csv", sics=sics, bones=bones, mat=gp)
    _write_count_csv(out_dir / "train_per_sic_bone_count.csv", sics=sics, bones=bones, mat=tr_n)
    _write_count_csv(out_dir / "eval_per_sic_bone_count.csv", sics=sics, bones=bones, mat=ev_n)

    focus_bones = _resolve_bones(str(args.focus_bones), bone_names=bones, root_idx=-1)
    if not focus_bones:
        focus_bones = list(bones)
    focus_sic_list = sorted(focus_sics) if focus_sics else list(sics)
    bi_map = {b: i for i, b in enumerate(bones)}
    si_map = {int(s): i for i, s in enumerate(sics)}

    focus_rows: List[Dict[str, Any]] = []
    for sic in focus_sic_list:
        if sic not in si_map:
            continue
        si = si_map[sic]
        for bone in focus_bones:
            if bone not in bi_map:
                continue
            bi = bi_map[bone]
            tr_v = float(tr[si, bi])
            ev_v = float(ev[si, bi])
            gp_v = float(gp[si, bi]) if (math.isfinite(tr_v) and math.isfinite(ev_v)) else float("nan")
            focus_rows.append(
                {
                    "sic": int(sic),
                    "bone": str(bone),
                    "train_mean_deg": tr_v,
                    "eval_mean_deg": ev_v,
                    "gap_eval_minus_train_deg": gp_v,
                    "train_count": int(tr_n[si, bi]),
                    "eval_count": int(ev_n[si, bi]),
                }
            )

    focus_rows.sort(
        key=lambda r: (
            -_safe_float(r.get("eval_mean_deg"), -1e9),
            -_safe_float(r.get("gap_eval_minus_train_deg"), -1e9),
            int(r.get("sic", 0)),
            str(r.get("bone", "")),
        )
    )

    focus_csv = out_dir / "focus_rows.csv"
    with focus_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sic", "bone", "train_mean_deg", "eval_mean_deg", "gap_eval_minus_train_deg", "train_count", "eval_count"])
        for r in focus_rows:
            def _fmt(v: float) -> str:
                return "" if not math.isfinite(float(v)) else f"{float(v):.6f}"
            w.writerow(
                [
                    int(r["sic"]),
                    str(r["bone"]),
                    _fmt(float(r["train_mean_deg"])),
                    _fmt(float(r["eval_mean_deg"])),
                    _fmt(float(r["gap_eval_minus_train_deg"])),
                    int(r["train_count"]),
                    int(r["eval_count"]),
                ]
            )

    per_sic_focus: List[Dict[str, Any]] = []
    for sic in focus_sic_list:
        if sic not in si_map:
            continue
        si = si_map[sic]
        vals_t: List[float] = []
        vals_e: List[float] = []
        w_t: List[int] = []
        w_e: List[int] = []
        for b in focus_bones:
            if b not in bi_map:
                continue
            bi = bi_map[b]
            vals_t.append(float(tr[si, bi]))
            vals_e.append(float(ev[si, bi]))
            w_t.append(int(tr_n[si, bi]))
            w_e.append(int(ev_n[si, bi]))
        tr_m, tr_cnt = _row_weighted_mean(vals_t, w_t)
        ev_m, ev_cnt = _row_weighted_mean(vals_e, w_e)
        gap_m = float(ev_m - tr_m) if (math.isfinite(tr_m) and math.isfinite(ev_m)) else float("nan")
        per_sic_focus.append(
            {
                "sic": int(sic),
                "train_mean_deg": tr_m,
                "eval_mean_deg": ev_m,
                "gap_eval_minus_train_deg": gap_m,
                "train_count": int(tr_cnt),
                "eval_count": int(ev_cnt),
            }
        )

    if not args.no_heatmap:
        note_a = _maybe_plot_heatmap(
            out_dir / "heatmap_train_mean_deg.png",
            title="Train x_gt mean DirectGeoLocalDeg (deg)",
            mat=tr,
            sics=sics,
            bones=bones,
            center_zero=False,
        )
        note_b = _maybe_plot_heatmap(
            out_dir / "heatmap_eval_mean_deg.png",
            title="Eval x_gt mean DirectGeoLocalDeg (deg)",
            mat=ev,
            sics=sics,
            bones=bones,
            center_zero=False,
        )
        note_c = _maybe_plot_heatmap(
            out_dir / "heatmap_gap_eval_minus_train_deg.png",
            title="Gap (Eval - Train) mean DirectGeoLocalDeg (deg)",
            mat=gp,
            sics=sics,
            bones=bones,
            center_zero=True,
        )
        for note in (note_a, note_b, note_c):
            if note:
                print(f"[WARN] heatmap skipped: {note}")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "train_json": [str(p) for p in train_json],
            "eval_json": [str(p) for p in eval_json],
            "bones": str(args.bones),
            "sics_filter": sorted(sics_filter) if sics_filter else None,
            "focus_bones": focus_bones,
            "focus_sics": focus_sic_list,
            "cycle_gte": int(args.cycle_gte),
            "drop_wrap": bool(drop_wrap),
        },
        "global": {
            "train_overall_mean_deg": float(train_stats.overall_mean),
            "train_overall_count": int(train_stats.overall_count),
            "eval_overall_mean_deg": float(eval_stats.overall_mean),
            "eval_overall_count": int(eval_stats.overall_count),
            "gap_eval_minus_train_deg": float(eval_stats.overall_mean - train_stats.overall_mean),
        },
        "per_sic_focus": per_sic_focus,
        "focus_rows_top_eval": focus_rows[: max(1, int(args.topk))],
        "focus_rows_top_gap": sorted(
            focus_rows,
            key=lambda r: _safe_float(r.get("gap_eval_minus_train_deg"), -1e9),
            reverse=True,
        )[: max(1, int(args.topk))],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines: List[str] = []
    md_lines.append("# Direct Pose Train-vs-Eval SIC Diagnostic")
    md_lines.append("")
    md_lines.append(f"- generated_at: {payload['generated_at']}")
    md_lines.append(f"- train_files: {len(train_json)}")
    md_lines.append(f"- eval_files: {len(eval_json)}")
    md_lines.append(f"- bones: `{args.bones}` -> {', '.join(bones)}")
    md_lines.append(f"- focus_sics: {focus_sic_list}")
    md_lines.append(f"- focus_bones: {focus_bones}")
    md_lines.append("")
    md_lines.append("## Global")
    md_lines.append("")
    md_lines.append("| split | mean_deg | samples |")
    md_lines.append("|---|---:|---:|")
    md_lines.append(f"| train | {train_stats.overall_mean:.6f} | {train_stats.overall_count} |")
    md_lines.append(f"| eval | {eval_stats.overall_mean:.6f} | {eval_stats.overall_count} |")
    md_lines.append(f"| gap (eval-train) | {eval_stats.overall_mean - train_stats.overall_mean:.6f} | - |")
    md_lines.append("")
    md_lines.append("## Focus SIC aggregate")
    md_lines.append("")
    md_lines.append("| sic | train_mean | eval_mean | gap(eval-train) | train_n | eval_n |")
    md_lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in per_sic_focus:
        def _fmt(x: float) -> str:
            return "nan" if not math.isfinite(float(x)) else f"{float(x):.6f}"
        md_lines.append(
            f"| {int(r['sic'])} | {_fmt(float(r['train_mean_deg']))} | {_fmt(float(r['eval_mean_deg']))} | "
            f"{_fmt(float(r['gap_eval_minus_train_deg']))} | {int(r['train_count'])} | {int(r['eval_count'])} |"
        )
    md_lines.append("")
    md_lines.append(f"## Top {max(1, int(args.topk))} focus rows by eval_mean")
    md_lines.append("")
    md_lines.append("| sic | bone | train_mean | eval_mean | gap(eval-train) | train_n | eval_n |")
    md_lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in focus_rows[: max(1, int(args.topk))]:
        tr_v = float(r["train_mean_deg"])
        ev_v = float(r["eval_mean_deg"])
        gp_v = float(r["gap_eval_minus_train_deg"])
        md_lines.append(
            f"| {int(r['sic'])} | {r['bone']} | "
            f"{'nan' if not math.isfinite(tr_v) else f'{tr_v:.6f}'} | "
            f"{'nan' if not math.isfinite(ev_v) else f'{ev_v:.6f}'} | "
            f"{'nan' if not math.isfinite(gp_v) else f'{gp_v:.6f}'} | "
            f"{int(r['train_count'])} | {int(r['eval_count'])} |"
        )
    md_lines.append("")
    md_lines.append(f"## Top {max(1, int(args.topk))} focus rows by gap(eval-train)")
    md_lines.append("")
    md_lines.append("| sic | bone | train_mean | eval_mean | gap(eval-train) | train_n | eval_n |")
    md_lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in sorted(
        focus_rows,
        key=lambda z: _safe_float(z.get("gap_eval_minus_train_deg"), -1e9),
        reverse=True,
    )[: max(1, int(args.topk))]:
        tr_v = float(r["train_mean_deg"])
        ev_v = float(r["eval_mean_deg"])
        gp_v = float(r["gap_eval_minus_train_deg"])
        md_lines.append(
            f"| {int(r['sic'])} | {r['bone']} | "
            f"{'nan' if not math.isfinite(tr_v) else f'{tr_v:.6f}'} | "
            f"{'nan' if not math.isfinite(ev_v) else f'{ev_v:.6f}'} | "
            f"{'nan' if not math.isfinite(gp_v) else f'{gp_v:.6f}'} | "
            f"{int(r['train_count'])} | {int(r['eval_count'])} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_dir / 'summary.md'}")
    print(f"[OK] wrote: {out_dir / 'summary.json'}")
    print(f"[OK] wrote: {out_dir / 'train_per_sic_bone_mean.csv'}")
    print(f"[OK] wrote: {out_dir / 'eval_per_sic_bone_mean.csv'}")
    print(f"[OK] wrote: {out_dir / 'gap_eval_minus_train_per_sic_bone_mean.csv'}")
    print(f"[OK] wrote: {focus_csv}")
    print(f"[RESULT] global train={train_stats.overall_mean:.6f} eval={eval_stats.overall_mean:.6f} gap={eval_stats.overall_mean - train_stats.overall_mean:+.6f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
