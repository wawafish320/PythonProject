#!/usr/bin/env python3
"""
Audit raw/processed/teacher-batch alignment for contacts + (de)normalization.

This is meant to catch "silent padding/truncation" and off-by-one length issues that can
poison supervision (e.g., contact_meas_head) across clips.

What it checks (per clip):
  - raw Frames length (from source_json)
  - processed lengths: X_flat / x_in_features / X_norm / y_out_features / Y_norm
  - contact length fit policy (truncate/pad) used by MotionEventDataset
  - denorm(Y_norm) == y_out_features (max abs diff)
  - denorm(X_norm) == x_in_features (max abs diff; includes inverse-tanh for RootVelocity/AngVel)
  - teacher_batches (optional): state_norm/target_norm dims and exact match vs npz
  - teacher_rollout outputs (optional): aux_inputs.contacts match vs raw (fit-to-pairs)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.io import load_soft_contacts_from_json, npz_scalar_to_str  # noqa: E402


def _expand_specs(specs: Sequence[str], *, want_suffix: str) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for spec in specs:
        if not spec:
            continue
        s = os.path.expanduser(str(spec))
        matches: List[Path] = []
        if any(ch in s for ch in "*?[]"):
            matches = [Path(p) for p in glob.glob(s)]
        else:
            p = Path(s)
            if p.is_dir():
                matches = sorted(p.glob(f"*{want_suffix}"))
            elif p.is_file():
                matches = [p]
        for m in matches:
            try:
                r = m.resolve()
            except Exception:
                r = m
            if r.is_file() and r.suffix == want_suffix and r not in seen:
                seen.add(r)
                out.append(r)
    return sorted(out)


def _atanh_safe(z: np.ndarray) -> np.ndarray:
    eps = 1.0 - 1e-6
    z = np.clip(z, -eps, eps)
    return 0.5 * (np.log1p(z) - np.log1p(-z))


def _fit_len(arr: np.ndarray, T: int) -> Tuple[np.ndarray, str]:
    """Pad (repeat last) / truncate to length T, like MotionEventDataset."""
    a = np.asarray(arr, dtype=np.float32)
    if a.shape[0] == T:
        return a, "ok"
    if a.shape[0] <= 0:
        return np.zeros((T, a.shape[1] if a.ndim == 2 else 0), dtype=np.float32), "empty"
    if a.shape[0] < T:
        pad = np.repeat(a[-1:], T - a.shape[0], axis=0)
        return np.concatenate([a, pad], axis=0), f"pad(+{T - a.shape[0]})"
    return a[:T], f"trunc(-{a.shape[0] - T})"


def _load_bundle(bundle_path: Path) -> Dict[str, Any]:
    with bundle_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{bundle_path}: expected dict json, got {type(obj)}")
    return obj


def _bundle_slices(bundle: Dict[str, Any]) -> Tuple[slice, slice, np.ndarray, np.ndarray]:
    meta = bundle.get("meta", {}) if isinstance(bundle.get("meta", {}), dict) else {}
    st_layout = meta.get("state_layout", {}) if isinstance(meta.get("state_layout", {}), dict) else {}

    def _sl(key: str) -> slice:
        ent = st_layout.get(key)
        if not isinstance(ent, dict):
            raise KeyError(f"bundle.meta.state_layout missing '{key}'")
        st = int(ent.get("start", 0))
        sz = int(ent.get("size", 0))
        if sz <= 0:
            raise ValueError(f"bundle.meta.state_layout['{key}'].size invalid: {sz}")
        return slice(st, st + sz)

    sl_root = _sl("RootVelocity")
    sl_ang = _sl("BoneAngularVelocities")
    scale_root = np.asarray(bundle.get("tanh_scales_rootvel", []), dtype=np.float32)
    scale_ang = np.asarray(bundle.get("tanh_scales_angvel", []), dtype=np.float32)
    if scale_root.size != (sl_root.stop - sl_root.start):
        raise ValueError(f"tanh_scales_rootvel len {scale_root.size} != RootVelocity size {sl_root.stop - sl_root.start}")
    if scale_ang.size != (sl_ang.stop - sl_ang.start):
        raise ValueError(f"tanh_scales_angvel len {scale_ang.size} != BoneAngularVelocities size {sl_ang.stop - sl_ang.start}")
    return sl_root, sl_ang, scale_root, scale_ang


def _denorm_y(Y_norm: np.ndarray, mu_y: np.ndarray, std_y: np.ndarray) -> np.ndarray:
    return Y_norm.astype(np.float32) * std_y.reshape(1, -1) + mu_y.reshape(1, -1)


def _denorm_x(
    X_norm: np.ndarray,
    mu_x: np.ndarray,
    std_x: np.ndarray,
    *,
    sl_root: slice,
    sl_ang: slice,
    scale_root: np.ndarray,
    scale_ang: np.ndarray,
) -> np.ndarray:
    X = X_norm.astype(np.float32) * std_x.reshape(1, -1) + mu_x.reshape(1, -1)
    X[:, sl_root] = _atanh_safe(X[:, sl_root]) * scale_root.reshape(1, -1)
    X[:, sl_ang] = _atanh_safe(X[:, sl_ang]) * scale_ang.reshape(1, -1)
    return X


def _load_teacher_batches(specs: Optional[Sequence[str]]) -> Dict[str, Dict[str, Any]]:
    if not specs:
        return {}
    files = _expand_specs(specs, want_suffix=".json")
    out: Dict[str, Dict[str, Any]] = {}
    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        clip = obj.get("clip") or p.stem.replace("_teacher", "")
        if isinstance(clip, str):
            out[clip] = obj
    return out


def _load_teacher_preds(specs: Optional[Sequence[str]]) -> Dict[str, Dict[str, Any]]:
    if not specs:
        return {}
    files = _expand_specs(specs, want_suffix=".json")
    out: Dict[str, Dict[str, Any]] = {}
    for p in files:
        if not p.name.endswith("_teacher_pred.json"):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        clip = obj.get("clip") or p.stem.replace("_teacher_pred", "")
        if isinstance(clip, str):
            out[clip] = obj
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit raw/processed/teacher alignment for contacts + (de)normalization.")
    ap.add_argument(
        "--npz",
        nargs="+",
        default=["raw_data/processed_data/*.npz"],
        help="NPZ paths/dirs/globs (default: raw_data/processed_data/*.npz).",
    )
    ap.add_argument(
        "--bundle",
        type=str,
        default="raw_data/processed_data/norm_template.json",
        help="Normalization bundle (norm_template.json).",
    )
    ap.add_argument(
        "--teacher",
        nargs="+",
        default=None,
        help="Optional teacher batch JSON specs to cross-check vs npz (e.g. validate/teacher_batches/*.json).",
    )
    ap.add_argument(
        "--teacher-pred",
        nargs="+",
        default=None,
        help="Optional teacher rollout output specs (`*_teacher_pred.json`) to verify aux_inputs.contacts vs raw.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output directory (writes audit_alignment.{json,csv,md}).",
    )
    args = ap.parse_args()

    npz_files = _expand_specs(args.npz, want_suffix=".npz")
    npz_files = [p for p in npz_files if p.name != "norm_template.npz"]
    if not npz_files:
        raise SystemExit("[FATAL] --npz expanded to empty list.")

    bundle_path = Path(args.bundle).expanduser()
    bundle = _load_bundle(bundle_path)
    mu_x = np.asarray(bundle.get("MuX", []), dtype=np.float32)
    std_x = np.asarray(bundle.get("StdX", []), dtype=np.float32)
    mu_y = np.asarray(bundle.get("MuY", []), dtype=np.float32)
    std_y = np.asarray(bundle.get("StdY", []), dtype=np.float32)
    sl_root, sl_ang, scale_root, scale_ang = _bundle_slices(bundle)

    teacher_batches = _load_teacher_batches(args.teacher)
    teacher_preds = _load_teacher_preds(args.teacher_pred)

    rows: List[Dict[str, Any]] = []
    for p in npz_files:
        z = np.load(p, allow_pickle=True)
        clip = p.stem

        X_flat = z.get("X_flat", None)
        x_in = z.get("x_in_features", None)
        Xn = z.get("X_norm", None)
        Y_raw = z.get("y_out_features", None)
        Yn = z.get("Y_norm", None)

        # Prefer pair length from Y_norm / X_norm when available.
        T_pairs = None
        try:
            if isinstance(Yn, np.ndarray) and Yn.ndim == 2:
                T_pairs = int(Yn.shape[0])
            elif isinstance(Xn, np.ndarray) and Xn.ndim == 2:
                T_pairs = int(Xn.shape[0])
        except Exception:
            T_pairs = None

        src_json = None
        try:
            if "source_json" in z:
                src_json = npz_scalar_to_str(z["source_json"])
        except Exception:
            src_json = None

        raw_T = None
        contact_fit_mode = None
        contact_max_diff_vs_teacher_pred = None
        if src_json and Path(src_json).is_file():
            try:
                sc_raw = load_soft_contacts_from_json(src_json)
                raw_T = int(sc_raw.shape[0])
                if T_pairs is not None:
                    sc_fit, contact_fit_mode = _fit_len(sc_raw, int(T_pairs))
            except Exception:
                raw_T = None
        else:
            src_json = None

        # Denorm checks
        denormY_max = None
        denormX_max = None
        try:
            if isinstance(Yn, np.ndarray) and isinstance(Y_raw, np.ndarray) and Yn.shape == Y_raw.shape:
                if mu_y.size == Yn.shape[1] and std_y.size == Yn.shape[1]:
                    Yd = _denorm_y(Yn, mu_y, std_y)
                    denormY_max = float(np.max(np.abs(Yd.astype(np.float64) - Y_raw.astype(np.float64))))
        except Exception:
            denormY_max = None

        try:
            if isinstance(Xn, np.ndarray) and isinstance(x_in, np.ndarray) and Xn.shape == x_in.shape:
                if mu_x.size == Xn.shape[1] and std_x.size == Xn.shape[1]:
                    Xd = _denorm_x(
                        Xn,
                        mu_x,
                        std_x,
                        sl_root=sl_root,
                        sl_ang=sl_ang,
                        scale_root=scale_root,
                        scale_ang=scale_ang,
                    )
                    denormX_max = float(np.max(np.abs(Xd.astype(np.float64) - x_in.astype(np.float64))))
        except Exception:
            denormX_max = None

        # Teacher batch exact match checks
        teacher_state_max = None
        teacher_target_max = None
        teacher_cond_max = None
        teacher_src_match = None
        tb = teacher_batches.get(clip)
        if isinstance(tb, dict):
            try:
                t_src = tb.get("source_json", None)
                if isinstance(t_src, str) and src_json:
                    teacher_src_match = bool(Path(t_src).resolve() == Path(src_json).resolve())
            except Exception:
                teacher_src_match = None
            try:
                t = tb.get("teacher", {})
                if isinstance(t, dict):
                    if isinstance(Xn, np.ndarray) and "state_norm" in t:
                        xs = np.asarray(t["state_norm"], dtype=np.float32)
                        if xs.shape == Xn.shape:
                            teacher_state_max = float(np.max(np.abs(xs.astype(np.float64) - Xn.astype(np.float64))))
                    if isinstance(Yn, np.ndarray) and "target_norm" in t:
                        yt = np.asarray(t["target_norm"], dtype=np.float32)
                        if yt.shape == Yn.shape:
                            teacher_target_max = float(np.max(np.abs(yt.astype(np.float64) - Yn.astype(np.float64))))
                    if "cond" in t and "cond_in" in z:
                        ct = np.asarray(t["cond"], dtype=np.float32)
                        ci = np.asarray(z["cond_in"], dtype=np.float32)
                        if ct.shape == ci.shape:
                            teacher_cond_max = float(np.max(np.abs(ct.astype(np.float64) - ci.astype(np.float64))))
            except Exception:
                pass

        # Teacher rollout aux_inputs.contacts vs raw contacts (fit to pair length)
        tp = teacher_preds.get(clip)
        if isinstance(tp, dict) and src_json and T_pairs is not None:
            try:
                sc_raw = load_soft_contacts_from_json(src_json)
                sc_fit, _ = _fit_len(sc_raw, int(T_pairs))
                aux = tp.get("aux_inputs", {})
                if isinstance(aux, dict):
                    c = aux.get("contacts", None)
                    c = np.asarray(c, dtype=np.float32)
                    if c.shape == sc_fit.shape:
                        contact_max_diff_vs_teacher_pred = float(np.max(np.abs(c.astype(np.float64) - sc_fit.astype(np.float64))))
            except Exception:
                contact_max_diff_vs_teacher_pred = None

        row = {
            "clip": clip,
            "npz": str(p),
            "raw_frames_T": raw_T,
            "X_flat_T": int(X_flat.shape[0]) if isinstance(X_flat, np.ndarray) and X_flat.ndim == 2 else None,
            "x_in_T": int(x_in.shape[0]) if isinstance(x_in, np.ndarray) and x_in.ndim == 2 else None,
            "X_norm_T": int(Xn.shape[0]) if isinstance(Xn, np.ndarray) and Xn.ndim == 2 else None,
            "y_out_T": int(Y_raw.shape[0]) if isinstance(Y_raw, np.ndarray) and Y_raw.ndim == 2 else None,
            "Y_norm_T": int(Yn.shape[0]) if isinstance(Yn, np.ndarray) and Yn.ndim == 2 else None,
            "pair_T": int(T_pairs) if T_pairs is not None else None,
            "contact_fit": contact_fit_mode,
            "denormY_max_abs": denormY_max,
            "denormX_max_abs": denormX_max,
            "teacher_src_match": teacher_src_match,
            "teacher_state_max_abs": teacher_state_max,
            "teacher_target_max_abs": teacher_target_max,
            "teacher_cond_max_abs": teacher_cond_max,
            "teacher_pred_contact_max_abs": contact_max_diff_vs_teacher_pred,
        }
        rows.append(row)

    # Sort by clip name for stable output
    rows.sort(key=lambda r: str(r.get("clip", "")))

    headers = [
        "clip",
        "raw_frames_T",
        "pair_T",
        "contact_fit",
        "denormY_max_abs",
        "denormX_max_abs",
        "teacher_src_match",
        "teacher_state_max_abs",
        "teacher_target_max_abs",
        "teacher_cond_max_abs",
        "teacher_pred_contact_max_abs",
    ]

    print(f"[AuditAlignment] clips={len(rows)} bundle={bundle_path}")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(r.get("clip", "")),
                    str(r.get("raw_frames_T", "")),
                    str(r.get("pair_T", "")),
                    str(r.get("contact_fit", "")),
                    f"{r.get('denormY_max_abs', '')}",
                    f"{r.get('denormX_max_abs', '')}",
                    str(r.get("teacher_src_match", "")),
                    f"{r.get('teacher_state_max_abs', '')}",
                    f"{r.get('teacher_target_max_abs', '')}",
                    f"{r.get('teacher_cond_max_abs', '')}",
                    f"{r.get('teacher_pred_contact_max_abs', '')}",
                ]
            )
            + " |"
        )

    if args.out:
        out_dir = Path(args.out).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "audit_alignment.json"
        out_csv = out_dir / "audit_alignment.csv"
        out_md = out_dir / "audit_alignment.md"

        out_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, None) for k in headers})
        out_md.write_text(
            "| " + " | ".join(headers) + " |\n"
            + "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|\n"
            + "\n".join(
                "| "
                + " | ".join(str(r.get(h, "")) for h in headers)
                + " |"
                for r in rows
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[Wrote] {out_json}")
        print(f"[Wrote] {out_csv}")
        print(f"[Wrote] {out_md}")


if __name__ == "__main__":
    main()

