#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from train.geometry import so3_exp_map, so3_log_map


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(cmd: List[str], *, log_path: Path) -> None:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[cmd]\n")
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
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
            f.write(line)
        rc = int(proc.wait())
    if rc != 0:
        raise RuntimeError(f"Command failed (exit={rc}): {' '.join(cmd)}")


def _swap_name(name: str) -> str:
    n = str(name)
    low = n.lower()
    if low.endswith("_l"):
        return n[:-2] + "_r"
    if low.endswith("_r"):
        return n[:-2] + "_l"
    if low.endswith("left"):
        return n[:-4] + "right"
    if low.endswith("right"):
        return n[:-5] + "left"
    return n


def _lr_pairs(names: List[str]) -> List[Tuple[int, int]]:
    idx = {str(n).lower(): i for i, n in enumerate(names)}
    out: List[Tuple[int, int]] = []
    seen = set()
    for i, n in enumerate(names):
        mate = _swap_name(str(n))
        if mate == n:
            continue
        j = idx.get(mate.lower())
        if j is None or j == i:
            continue
        a, b = min(i, j), max(i, j)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


def _swap_joint_block(arr: np.ndarray, start: int, size: int, per_joint_dim: int, pairs: List[Tuple[int, int]]) -> None:
    if size <= 0 or per_joint_dim <= 0:
        return
    j_count = size // per_joint_dim
    blk = arr[:, start : start + size].reshape(arr.shape[0], j_count, per_joint_dim).copy()
    for a, b in pairs:
        if a >= j_count or b >= j_count:
            continue
        tmp = blk[:, a, :].copy()
        blk[:, a, :] = blk[:, b, :]
        blk[:, b, :] = tmp
    arr[:, start : start + size] = blk.reshape(arr.shape[0], size)


def _make_swapped_teacher(
    *,
    teacher_path: Path,
    npz_path: Path,
    out_path: Path,
    swap_cond: bool = True,
) -> Dict[str, Any]:
    obj = _load_json(teacher_path)
    tea = obj.get("teacher", {})
    if not isinstance(tea, dict):
        raise RuntimeError(f"{teacher_path} missing teacher payload")

    state = np.asarray(tea.get("state_norm"), dtype=np.float64)
    cond = np.asarray(tea.get("cond"), dtype=np.float64)
    target = np.asarray(tea.get("target_norm"), dtype=np.float64)
    if state.ndim != 2 or cond.ndim != 2 or target.ndim != 2:
        raise RuntimeError("teacher shapes must be 2D")

    layout_state = obj.get("layouts", {}).get("state", {})
    layout_out = obj.get("layouts", {}).get("output", {})
    s_rot = layout_state.get("BoneRotations6D", {})
    s_ang = layout_state.get("BoneAngularVelocities", {})
    y_rot = layout_out.get("BoneRotations6D", {})
    if not s_rot or not y_rot:
        raise RuntimeError("teacher layouts missing BoneRotations6D")

    npz = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in np.asarray(npz["bone_names"]).tolist()]
    pairs = _lr_pairs(names)

    state_sw = state.copy()
    target_sw = target.copy()
    cond_sw = cond.copy()

    _swap_joint_block(
        state_sw,
        int(s_rot.get("start", 0)),
        int(s_rot.get("size", 0)),
        per_joint_dim=6,
        pairs=pairs,
    )
    _swap_joint_block(
        state_sw,
        int(s_ang.get("start", 0)),
        int(s_ang.get("size", 0)),
        per_joint_dim=3,
        pairs=pairs,
    )
    _swap_joint_block(
        target_sw,
        int(y_rot.get("start", 0)),
        int(y_rot.get("size", 0)),
        per_joint_dim=6,
        pairs=pairs,
    )

    # Optional cond heuristic: channel-0/1 is treated as an L/R pair.
    if bool(swap_cond) and cond_sw.shape[1] >= 2:
        c0 = cond_sw[:, 0].copy()
        cond_sw[:, 0] = cond_sw[:, 1]
        cond_sw[:, 1] = c0

    obj2 = dict(obj)
    tea2 = dict(tea)
    tea2["state_norm"] = state_sw.astype(np.float32).tolist()
    tea2["cond"] = cond_sw.astype(np.float32).tolist()
    tea2["target_norm"] = target_sw.astype(np.float32).tolist()
    obj2["teacher"] = tea2
    obj2["lr_swap_meta"] = {
        "source_teacher": str(teacher_path),
        "npz": str(npz_path),
        "pairs": [[int(a), int(b)] for a, b in pairs],
        "bone_names": names,
        "note": (
            "Deterministic LR swap on teacher payload: "
            "rot6d/angvel/target rot6d"
            + (" + cond ch0<->ch1." if bool(swap_cond) else ".")
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"pairs": pairs, "bone_names": names}


def _swap_joint_block_copy(
    arr: np.ndarray,
    *,
    start: int,
    size: int,
    per_joint_dim: int,
    pairs: List[Tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(arr).copy()
    if out.ndim != 2:
        return out
    d = int(size)
    s = int(start)
    pj = int(per_joint_dim)
    if d <= 0 or pj <= 0 or d % pj != 0:
        return out
    if s < 0 or (s + d) > int(out.shape[1]):
        return out
    j_count = d // pj
    blk = out[:, s : s + d].reshape(out.shape[0], j_count, pj).copy()
    for a, b in pairs:
        if a >= j_count or b >= j_count:
            continue
        tmp = blk[:, a, :].copy()
        blk[:, a, :] = blk[:, b, :]
        blk[:, b, :] = tmp
    out[:, s : s + d] = blk.reshape(out.shape[0], d)
    return out


def _swap_joint_arr_copy(
    arr: np.ndarray,
    *,
    per_joint_dim: int,
    pairs: List[Tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(arr).copy()
    if out.ndim != 3 or int(out.shape[-1]) != int(per_joint_dim):
        return out
    j_count = int(out.shape[1])
    for a, b in pairs:
        if a >= j_count or b >= j_count:
            continue
        tmp = out[:, a, :].copy()
        out[:, a, :] = out[:, b, :]
        out[:, b, :] = tmp
    return out


def _parse_layout_json_field(npz: Any, key: str) -> Dict[str, Any]:
    if key not in npz.files:
        return {}
    raw = np.asarray(npz[key]).item()
    if isinstance(raw, (dict, list)):
        return raw if isinstance(raw, dict) else {}
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _make_swapped_npz(
    *,
    npz_path: Path,
    out_path: Path,
    pairs: List[Tuple[int, int]],
    swap_cond: bool = False,
) -> Dict[str, Any]:
    src = np.load(npz_path, allow_pickle=True)
    state_layout = _parse_layout_json_field(src, "state_layout_json")
    out_layout = _parse_layout_json_field(src, "output_layout_json")
    s_rot = (state_layout.get("BoneRotations6D", {}) if isinstance(state_layout, dict) else {}) or {}
    s_ang = (state_layout.get("BoneAngularVelocities", {}) if isinstance(state_layout, dict) else {}) or {}
    y_rot = (out_layout.get("BoneRotations6D", {}) if isinstance(out_layout, dict) else {}) or {}

    out_payload: Dict[str, Any] = {}
    for k in src.files:
        out_payload[k] = np.asarray(src[k]).copy()

    for k in ("x_in_features", "X_norm", "X_flat"):
        if k not in out_payload:
            continue
        a = np.asarray(out_payload[k])
        a = _swap_joint_block_copy(
            a,
            start=int(s_rot.get("start", 0)),
            size=int(s_rot.get("size", 0)),
            per_joint_dim=6,
            pairs=pairs,
        )
        a = _swap_joint_block_copy(
            a,
            start=int(s_ang.get("start", 0)),
            size=int(s_ang.get("size", 0)),
            per_joint_dim=3,
            pairs=pairs,
        )
        out_payload[k] = a.astype(out_payload[k].dtype, copy=False)

    for k in ("y_out_features", "Y_norm"):
        if k not in out_payload:
            continue
        a = np.asarray(out_payload[k])
        a = _swap_joint_block_copy(
            a,
            start=int(y_rot.get("start", 0)),
            size=int(y_rot.get("size", 0)),
            per_joint_dim=6,
            pairs=pairs,
        )
        out_payload[k] = a.astype(out_payload[k].dtype, copy=False)

    if "bone_rot6d" in out_payload:
        out_payload["bone_rot6d"] = _swap_joint_arr_copy(
            np.asarray(out_payload["bone_rot6d"]),
            per_joint_dim=6,
            pairs=pairs,
        ).astype(out_payload["bone_rot6d"].dtype, copy=False)
    if "bone_ang_vel" in out_payload:
        out_payload["bone_ang_vel"] = _swap_joint_arr_copy(
            np.asarray(out_payload["bone_ang_vel"]),
            per_joint_dim=3,
            pairs=pairs,
        ).astype(out_payload["bone_ang_vel"].dtype, copy=False)

    if bool(swap_cond) and "cond_in" in out_payload:
        cond = np.asarray(out_payload["cond_in"]).copy()
        if cond.ndim == 2 and int(cond.shape[1]) >= 2:
            c0 = cond[:, 0].copy()
            cond[:, 0] = cond[:, 1]
            cond[:, 1] = c0
            out_payload["cond_in"] = cond.astype(out_payload["cond_in"].dtype, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_payload)
    return {
        "npz_source": str(npz_path),
        "npz_swapped": str(out_path),
        "pairs": [[int(a), int(b)] for a, b in pairs],
        "swap_cond": bool(swap_cond),
    }


def _rotvec_deg_dict_to_tensor(series: Dict[str, List[List[float]]], names: List[str], T: int) -> np.ndarray:
    out = np.zeros((T, len(names), 3), dtype=np.float64)
    for j, n in enumerate(names):
        a = np.asarray(series[str(n)], dtype=np.float64)
        out[:, j, :] = a[:T]
    return out


def _tensor_to_matrix(rotvec_deg: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rotvec_deg.astype(np.float32)) * (math.pi / 180.0)
    return so3_exp_map(x)


def _matrix_to_rotvec_deg(R: torch.Tensor) -> np.ndarray:
    w = so3_log_map(R)
    return (w * (180.0 / math.pi)).detach().cpu().numpy().astype(np.float64)


def _project_so3_mean(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    M = 0.5 * (Ra + Rb)
    U, _S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    det = torch.det(R)
    bad = det < 0
    if bool(torch.any(bad)):
        U_fix = U.clone()
        U_fix[bad, :, -1] *= -1.0
        R = U_fix @ Vh
    return R


def _stats(v: np.ndarray) -> Dict[str, float]:
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return {"mean": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan"), "N": 0}
    return {
        "mean": float(np.mean(vv)),
        "p90": float(np.percentile(vv, 90)),
        "p99": float(np.percentile(vv, 99)),
        "max": float(np.max(vv)),
        "N": int(vv.size),
    }


def _layer_gap(base_steps: List[Dict[str, Any]], flip_steps: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    b = {int(x["step"]): np.asarray(x.get(key, []), dtype=np.float64) for x in base_steps if key in x}
    f = {int(x["step"]): np.asarray(x.get(key, []), dtype=np.float64) for x in flip_steps if key in x}
    shared = sorted(set(b.keys()) & set(f.keys()))
    if not shared:
        return {"N": 0, "mean": float("nan"), "p90": float("nan"), "p99": float("nan")}
    d = []
    for t in shared:
        if b[t].shape != f[t].shape:
            continue
        d.append(float(np.linalg.norm(b[t] - f[t])))
    arr = np.asarray(d, dtype=np.float64)
    return {"N": int(arr.size), "mean": float(np.mean(arr)), "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99))}


def _layer_gap_transformed(
    base_steps: List[Dict[str, Any]],
    flip_steps: List[Dict[str, Any]],
    key: str,
    transform_base,
) -> Dict[str, Any]:
    b = {int(x["step"]): np.asarray(x.get(key, []), dtype=np.float64) for x in base_steps if key in x}
    f = {int(x["step"]): np.asarray(x.get(key, []), dtype=np.float64) for x in flip_steps if key in x}
    shared = sorted(set(b.keys()) & set(f.keys()))
    if not shared:
        return {"N": 0, "mean": float("nan"), "p90": float("nan"), "p99": float("nan")}
    d = []
    for t in shared:
        bv = transform_base(b[t])
        fv = f[t]
        if bv.shape != fv.shape:
            continue
        d.append(float(np.linalg.norm(fv - bv)))
    arr = np.asarray(d, dtype=np.float64)
    return {"N": int(arr.size), "mean": float(np.mean(arr)), "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99))}


def _swap_direct_in_vec(v: np.ndarray, contact_dim: int = 2) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).copy()
    c = int(max(0, contact_dim))
    if c <= 0:
        return x
    if x.size >= 2:
        x[0], x[1] = x[1], x[0]
    cond_dim = int(x.size - 2 * c)
    if cond_dim >= 0 and (cond_dim + 2 * c) <= x.size and c >= 2:
        p0, p1 = cond_dim, cond_dim + 1
        m0, m1 = cond_dim + c, cond_dim + c + 1
        if m1 < x.size:
            x[p0], x[p1] = x[p1], x[p0]
            x[m0], x[m1] = x[m1], x[m0]
    return x


def _swap_out_direct_vec(v: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).copy()
    if x.size < 8:
        return x
    if (x.size - 2) % 6 != 0:
        return x
    j = (x.size - 2) // 6
    blk = x[: j * 6].reshape(j, 6).copy()
    for a, b in pairs:
        if a >= j or b >= j:
            continue
        tmp = blk[a, :].copy()
        blk[a, :] = blk[b, :]
        blk[b, :] = tmp
    x[: j * 6] = blk.reshape(-1)
    return x


def _compute_dt_summary(
    *,
    err_deg_tjc: np.ndarray,
    bone_names: List[str],
    steps_meta: List[Dict[str, Any]],
    npz_path: Path,
    fps: float,
    joints: List[str],
    axis: int = 2,
    mu_thr: float = 0.5,
    omega_thr: float = 30.0,
) -> Dict[str, Any]:
    npz = np.load(npz_path, allow_pickle=True)
    omega_deg_s = np.asarray(npz["bone_ang_vel"], dtype=np.float64) * (180.0 / math.pi)
    cycle_len = int(max(s.get("step_in_cycle", 0) for s in steps_meta) + 1)
    omega_deg_s = omega_deg_s[:cycle_len]
    name_to_j_err = {str(n): i for i, n in enumerate(bone_names)}
    npz_bones = [str(x) for x in np.asarray(npz["bone_names"]).tolist()]
    name_to_j_npz = {str(n): i for i, n in enumerate(npz_bones)}

    by_joint: Dict[str, Any] = {}
    meds: Dict[str, float] = {}
    for name in joints:
        j_err = name_to_j_err.get(name)
        j_npz = name_to_j_npz.get(name)
        if j_err is None or j_npz is None:
            continue
        mu = np.full((cycle_len, 3), np.nan, dtype=np.float64)
        for sic in range(cycle_len):
            idx = [
                i
                for i, s in enumerate(steps_meta)
                if int(s.get("step_in_cycle", -1)) == sic
                and int(s.get("cycle", 0)) >= 1
                and not bool(s.get("wrap_boundary_step", False))
            ]
            if idx:
                mu[sic] = np.nanmean(err_deg_tjc[np.asarray(idx, dtype=np.int64), j_err, :], axis=0)
        mu_axis = mu[:, axis]
        mu_norm = np.linalg.norm(mu, axis=1)
        w_axis = omega_deg_s[:cycle_len, j_npz, axis]
        mask_mu = np.isfinite(mu_axis) & np.isfinite(mu_norm) & (mu_norm >= float(mu_thr))
        align = np.nan
        if np.any(mask_mu):
            align = float(np.mean((mu_axis[mask_mu] * w_axis[mask_mu]) > 0.0))
        ok = mask_mu & np.isfinite(w_axis) & (np.abs(w_axis) >= float(omega_thr))
        dt = np.full((cycle_len,), np.nan, dtype=np.float64)
        dt[ok] = (mu_axis[ok] / w_axis[ok]) * float(fps)
        vals = dt[np.isfinite(dt)]
        med = float(np.median(vals)) if vals.size else float("nan")
        meds[name] = med
        by_joint[name] = {
            "align_frac": align,
            "N_mu": int(np.sum(mask_mu)),
            "N_dt": int(vals.size),
            "dt_median": med,
            "dt_iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))] if vals.size else [float("nan"), float("nan")],
        }

    if len(meds) >= 2:
        j0, j1 = joints[0], joints[1]
        m0, m1 = meds.get(j0, float("nan")), meds.get(j1, float("nan"))
        common = 0.5 * (m0 + m1) if np.isfinite(m0) and np.isfinite(m1) else float("nan")
        asym = 0.5 * abs(m0 - m1) if np.isfinite(m0) and np.isfinite(m1) else float("nan")
        same = bool((m0 >= 0 and m1 >= 0) or (m0 <= 0 and m1 <= 0)) if np.isfinite(m0) and np.isfinite(m1) else False
    else:
        common = asym = float("nan")
        same = False

    return {"joints": by_joint, "common_dt": common, "asym_dt": asym, "same_sign": same}


def _format_md(summary: Dict[str, Any]) -> str:
    l = []
    l.append("# Stage7 strict LR-flip funnel (no-training)")
    l.append("")
    l.append("## Layered E_flip")
    l.append("")
    l.append("| layer | N | mean | p90 | p99 |")
    l.append("|---|---:|---:|---:|---:|")
    for k, v in summary["layer_e_flip"].items():
        l.append(f"| {k} | {int(v.get('N', 0))} | {v.get('mean', float('nan')):.4f} | {v.get('p90', float('nan')):.4f} | {v.get('p99', float('nan')):.4f} |")
    l.append("")
    l.append("## Strict Ensemble (direct local error)")
    l.append("")
    b = summary["direct_local_deg"]["base"]
    e = summary["direct_local_deg"]["ensemble"]
    l.append(f"- base: mean={b['mean']:.4f}, p99={b['p99']:.4f}, max={b['max']:.4f}")
    l.append(f"- ensemble: mean={e['mean']:.4f}, p99={e['p99']:.4f}, max={e['max']:.4f}")
    l.append("")
    l.append("## dt summary (axis=z)")
    l.append("")
    for tag in ("base", "ensemble"):
        s = summary["dt_summary"][tag]
        jl = s["joints"].get("calf_l", {})
        jr = s["joints"].get("calf_r", {})
        l.append(
            f"- {tag}: calf_l={jl.get('dt_median', float('nan')):.3f}, "
            f"calf_r={jr.get('dt_median', float('nan')):.3f}, "
            f"asym_dt={s.get('asym_dt', float('nan')):.3f}, same_sign={bool(s.get('same_sign', False))}"
        )
    l.append("")
    return "\n".join(l) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run strict LR-flip no-training funnel (ensemble + layered E_flip).")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--npz", type=str, default="raw_data/processed_data/Walk_F.npz")
    ap.add_argument(
        "--flip-source",
        type=str,
        default="npz_swap",
        choices=("npz_swap", "teacher_swap"),
        help=(
            "How to construct flipped rollout input. "
            "npz_swap swaps feature tensors in the clip npz (stricter); "
            "teacher_swap keeps legacy behavior by swapping teacher payload only."
        ),
    )
    ap.add_argument(
        "--swap-cond",
        action="store_true",
        help="Also swap cond channel-0/1 when preparing flipped input (off by default for npz_swap).",
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out-root", type=str, default="debug_output/_lrflip_strict_funnel_20260214")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    model = str(Path(args.model).expanduser().resolve())
    teacher = (ROOT / str(args.teacher)).expanduser().resolve()
    npz_path = (ROOT / str(args.npz)).expanduser().resolve()
    out_root = (ROOT / str(args.out_root)).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    npz_root_base = npz_path.parent.resolve()

    npz_obj = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in np.asarray(npz_obj["bone_names"]).tolist()]
    pairs = _lr_pairs(names)
    if not pairs:
        raise RuntimeError(f"No L/R joint pairs found from {npz_path}.")

    teacher_base = teacher
    teacher_flip = teacher
    npz_root_flip = npz_root_base
    swap_teacher: Path | None = None
    swap_npz: Path | None = None
    swap_meta: Dict[str, Any] = {"pairs": [[int(a), int(b)] for a, b in pairs], "bone_names": names}

    swap_cond = bool(args.swap_cond)
    if str(args.flip_source).strip().lower() == "teacher_swap":
        # Keep legacy path for reproducibility.
        # NOTE: run_freerun_cycles builds rollout sample from npz, so teacher_swap may be weak/ineffective.
        swap_teacher = out_root / "Walk_F_teacher_lr_swap_strict.json"
        swap_meta = _make_swapped_teacher(
            teacher_path=teacher,
            npz_path=npz_path,
            out_path=swap_teacher,
            swap_cond=True if not bool(args.swap_cond) else bool(args.swap_cond),
        )
        teacher_flip = swap_teacher
    else:
        # Stricter flip: swap the npz tensors that actually feed rollout inputs.
        swap_npz = out_root / "npz_swap" / npz_path.name
        swap_meta_npz = _make_swapped_npz(
            npz_path=npz_path,
            out_path=swap_npz,
            pairs=pairs,
            swap_cond=swap_cond,
        )
        npz_root_flip = swap_npz.parent.resolve()
        swap_meta.update(swap_meta_npz)

    common_args = [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--model",
        model,
        "--rounds",
        str(int(args.rounds)),
        "--export_joint_so3_error_series",
        "--joint_so3_error_series_branches",
        "direct",
        "--joint_so3_error_series_space",
        "body",
        "--export_keybone_state_series",
        "--keybone_state_series_bones",
        "all",
        "--keybone_state_series_branches",
        "direct",
        "--export_direct_head_io",
        "--direct_leg_omega_alpha_sweep_sic_range",
        "0-86",
    ]
    if bool(args.force):
        common_args.append("--force")

    base_dir = out_root / "base"
    flip_dir = out_root / "flip_tx"
    _run(
        common_args + ["--teacher", str(teacher_base), "--npz-root", str(npz_root_base), "--out", str(base_dir)],
        log_path=out_root / "run_base.log",
    )
    _run(
        common_args + ["--teacher", str(teacher_flip), "--npz-root", str(npz_root_flip), "--out", str(flip_dir)],
        log_path=out_root / "run_flip.log",
    )

    base = _load_json(base_dir / "Walk_F_freerun_cycles.json")
    flip = _load_json(flip_dir / "Walk_F_freerun_cycles.json")

    # Layered E_flip from direct head IO.
    base_io_steps = list((base.get("direct_head_io") or {}).get("steps", []))
    flip_io_steps = list((flip.get("direct_head_io") or {}).get("steps", []))
    layer_e_flip = {
        "h_final": _layer_gap(base_io_steps, flip_io_steps, "h_final"),
        "direct_in_raw": _layer_gap(base_io_steps, flip_io_steps, "direct_in"),
        "direct_in": _layer_gap_transformed(
            base_io_steps,
            flip_io_steps,
            "direct_in",
            transform_base=lambda v: _swap_direct_in_vec(v, contact_dim=2),
        ),
        "pre0": _layer_gap(base_io_steps, flip_io_steps, "pre0"),
        "out_direct_raw": _layer_gap(base_io_steps, flip_io_steps, "out_direct"),
        "out_direct": _layer_gap_transformed(
            base_io_steps,
            flip_io_steps,
            "out_direct",
            transform_base=lambda v: _swap_out_direct_vec(v, pairs),
        ),
    }

    # Final-output E_flip + strict ensemble on direct branch local rotations.
    ks_b = base["keybone_state"]["series"]["branches"]["direct"]["pred_rotvec_deg_xyz"]
    ks_f = flip["keybone_state"]["series"]["branches"]["direct"]["pred_rotvec_deg_xyz"]
    names_b = list(base["keybone_state"]["series"]["bones"])
    names_f = list(flip["keybone_state"]["series"]["bones"])

    # Build unswapped dict for flipped run.
    f_unswap: Dict[str, np.ndarray] = {}
    for n in names_f:
        arr = np.asarray(ks_f[str(n)], dtype=np.float64)
        f_unswap[_swap_name(str(n))] = arr

    common_names = [n for n in names_b if n in f_unswap]
    T = min(len(np.asarray(ks_b[common_names[0]])), len(f_unswap[common_names[0]]))
    rb = _rotvec_deg_dict_to_tensor({n: ks_b[n] for n in common_names}, common_names, T)
    rf = np.stack([f_unswap[n][:T] for n in common_names], axis=1)
    eflip_final = np.linalg.norm(rb - rf, axis=-1)
    layer_e_flip["final_rotvec"] = _stats(eflip_final)

    # Reconstruct GT from base prediction + base error series.
    err_obj = np.asarray(
        base["per_step_joint_so3_error"]["branches"]["direct"]["body"]["rotvec_deg_xyz"],
        dtype=np.float64,
    )
    err_bones = list(base["per_step_joint_so3_error"]["bone_names"])
    e_dict = {name: np.asarray(err_obj[:, i, :], dtype=np.float64) for i, name in enumerate(err_bones)}
    e_arr = np.stack([e_dict[n][:T] for n in common_names], axis=1)

    Rb = _tensor_to_matrix(rb.reshape(-1, 3)).reshape(T, len(common_names), 3, 3)
    Rf = _tensor_to_matrix(rf.reshape(-1, 3)).reshape(T, len(common_names), 3, 3)
    E_base = _tensor_to_matrix(e_arr.reshape(-1, 3)).reshape(T, len(common_names), 3, 3)
    Rg = torch.matmul(Rb, E_base)
    Rens = _project_so3_mean(Rb.reshape(-1, 3, 3), Rf.reshape(-1, 3, 3)).reshape(T, len(common_names), 3, 3)
    E_ens = torch.matmul(Rens.transpose(-1, -2), Rg)
    e_ens_deg = _matrix_to_rotvec_deg(E_ens.reshape(-1, 3, 3)).reshape(T, len(common_names), 3)
    ang_base = np.linalg.norm(e_arr, axis=-1)
    ang_ens = np.linalg.norm(e_ens_deg, axis=-1)

    # dt summary on calf_l/calf_r.
    steps_meta = list(base.get("metrics_per_step", []))[:T]
    dt_base = _compute_dt_summary(
        err_deg_tjc=e_arr,
        bone_names=common_names,
        steps_meta=steps_meta,
        npz_path=npz_path,
        fps=float(base.get("fps", 60.0)),
        joints=["calf_l", "calf_r"],
    )
    dt_ens = _compute_dt_summary(
        err_deg_tjc=e_ens_deg,
        bone_names=common_names,
        steps_meta=steps_meta,
        npz_path=npz_path,
        fps=float(base.get("fps", 60.0)),
        joints=["calf_l", "calf_r"],
    )

    summary = {
        "config": {
            "model": model,
            "flip_source": str(args.flip_source),
            "teacher_base": str(teacher_base),
            "teacher_flip": str(teacher_flip),
            "npz": str(npz_path),
            "npz_root_base": str(npz_root_base),
            "npz_root_flip": str(npz_root_flip),
            "npz_swapped": (str(swap_npz) if swap_npz is not None else None),
            "rounds": int(args.rounds),
            "pairs": [[int(a), int(b)] for a, b in pairs],
            "swap_cond": bool(args.swap_cond),
            "swap_meta": swap_meta,
        },
        "layer_e_flip": layer_e_flip,
        "direct_local_deg": {"base": _stats(ang_base), "ensemble": _stats(ang_ens)},
        "dt_summary": {"base": dt_base, "ensemble": dt_ens},
    }

    out_json = out_root / "strict_flip_ensemble_summary.json"
    out_md = out_root / "strict_flip_ensemble_summary.md"
    _write_json(out_json, summary)
    out_md.write_text(_format_md(summary), encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
