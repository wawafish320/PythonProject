#!/usr/bin/env python3
"""
Run H1 10.3.A epoch-0 gate (teacher vs freerun_x_gt proxy) requested in:
  docs/Problems/active/2026-02-11_WalkF_stage7_phase_lag_velocity_loss.md

Outputs:
  - <out_dir>/h1_10p3a_gate_summary.json
  - <out_dir>/h1_10p3a_gate_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from train.geometry import geodesic_R, rot6d_to_matrix, so3_log_map
from train.validate.run_freerun_cycles import (
    FreeRunCycleRunner,
    _build_full_cycle_sample,
    _load_json,
    _resolve_npz_path,
)


def _parse_index_spec(spec: str, *, upper: Optional[int] = None) -> List[int]:
    text = str(spec or "").strip().lower()
    if text in ("", "none", "null"):
        return []
    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            try:
                lo = int(a.strip())
                hi = int(b.strip())
            except Exception:
                continue
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(token))
            except Exception:
                continue
    out = sorted(set(out))
    if upper is not None:
        out = [x for x in out if 0 <= int(x) < int(upper)]
    return out


def _parse_bool_text(x: Any, default: bool) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", "none", "null", ""):
        return False
    return bool(default)


def _coerce_int(value: Any, default: int) -> Tuple[int, bool]:
    """Return (parsed_value, used_default)."""
    d = int(default)
    if value is None:
        return d, True
    try:
        if isinstance(value, bool):
            return int(value), False
        if isinstance(value, (np.integer, int)):
            return int(value), False
        if isinstance(value, (np.floating, float)):
            fv = float(value)
            if not math.isfinite(fv):
                return d, True
            return int(fv), False
        s = str(value).strip().lower()
        if s in ("", "none", "null", "nan"):
            return d, True
        return int(float(s)), False
    except Exception:
        return d, True


def _to_device(x: Any, device: torch.device) -> Optional[torch.Tensor]:
    if not torch.is_tensor(x):
        return None
    return x.to(device)


def _infer_rot_slice(runner: FreeRunCycleRunner, out_dim: int) -> slice:
    trainer = getattr(runner, "trainer", None)
    sl = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if isinstance(sl, slice):
        return sl
    return slice(0, int(out_dim))


def _build_runner(args: argparse.Namespace) -> FreeRunCycleRunner:
    runner_args = argparse.Namespace(
        model=str(Path(args.model).expanduser().resolve()),
        device=str(args.device),
        bundle=str(Path(args.bundle).expanduser()),
        pretrain_template=str(Path(args.pretrain_template).expanduser()),
        encoder_bundle=str(Path(args.encoder_bundle).expanduser()),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        context_len=int(args.context_len),
        depth=int(args.depth),
        so3_corr_apply=False,
        so3_corr_max_deg=20.0,
        lambda_fusion_apply=False,
    )
    return FreeRunCycleRunner(runner_args)


def _load_teacher_sample(
    runner: FreeRunCycleRunner,
    *,
    teacher_path: Path,
    npz_root: Path,
) -> Dict[str, Any]:
    payload = _load_json(teacher_path)
    clip_name = str(payload.get("clip") or teacher_path.stem.replace("_teacher", ""))
    teacher_blk = payload.get("teacher")
    if not isinstance(teacher_blk, dict):
        raise RuntimeError(f"{teacher_path}: missing teacher block.")
    state_arr = np.asarray(teacher_blk.get("state_norm"), dtype=np.float32)
    if state_arr.ndim != 2 or state_arr.shape[0] < 2:
        raise RuntimeError(f"{teacher_path}: invalid state_norm shape={state_arr.shape}.")
    seq_len = int(state_arr.shape[0])
    npz_path = _resolve_npz_path(clip_name, payload.get("source_json"), npz_root)
    ds = runner._build_dataset(npz_path, seq_len=seq_len)
    runner._ensure_model_ready(ds)
    clip = ds.clips[0]
    sample = _build_full_cycle_sample(ds, clip, seq_len=seq_len)
    bone_names = list(getattr(ds, "bone_names", []) or [])
    return {
        "clip": clip_name,
        "teacher_path": str(teacher_path),
        "npz_path": str(npz_path),
        "seq_len": seq_len,
        "sample": sample,
        "bone_names": bone_names,
    }


def _forward_teacher_rotvec_deg(
    runner: FreeRunCycleRunner,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    trainer = getattr(runner, "trainer", None)
    if trainer is None or runner.model is None:
        raise RuntimeError("Runner model/trainer is not initialized.")
    model = runner.model.to(runner.device)
    model.eval()

    state = _to_device(sample.get("motion"), runner.device)
    gt = _to_device(sample.get("gt_motion"), runner.device)
    cond = _to_device(sample.get("cond_in"), runner.device)
    angvel = _to_device(sample.get("angvel"), runner.device)
    pose_hist = _to_device(sample.get("pose_hist"), runner.device)
    contacts = _to_device(sample.get("contacts"), runner.device)

    if state is None or gt is None:
        raise RuntimeError("sample missing motion/gt_motion.")
    state = state.unsqueeze(0)
    gt = gt.unsqueeze(0)
    cond = cond.unsqueeze(0) if cond is not None else None
    angvel = angvel.unsqueeze(0) if angvel is not None else None
    pose_hist = pose_hist.unsqueeze(0) if pose_hist is not None else None
    contacts = contacts.unsqueeze(0) if contacts is not None else None

    use_learned_meas = bool(getattr(model, "contact_meas_enable", False)) and getattr(model, "contact_meas_head", None) is not None
    if use_learned_meas:
        contacts = None

    with torch.no_grad():
        ret = model(
            state,
            cond=cond,
            contacts=contacts,
            angvel=angvel,
            pose_history=pose_hist,
            plan_z=None,
            time_index=None,
        )
        if not isinstance(ret, dict) or "out_direct" not in ret:
            raise RuntimeError("model forward missing out_direct.")
        out_direct = ret["out_direct"]  # (1,T,Dy), normalized
        if out_direct.ndim != 3:
            raise RuntimeError(f"out_direct shape {tuple(out_direct.shape)} != (B,T,Dy).")
        out_direct_raw = trainer._denorm(out_direct)
        gt_raw = trainer._denorm(gt)

        rot_slice = _infer_rot_slice(runner, int(out_direct_raw.shape[-1]))
        r0 = int(rot_slice.start or 0)
        r1 = int(rot_slice.stop or out_direct_raw.shape[-1])
        rot_len = int(max(0, r1 - r0))
        if rot_len <= 0 or (rot_len % 6) != 0:
            raise RuntimeError(f"invalid rot slice [{r0}:{r1}] over Dy={out_direct_raw.shape[-1]}.")
        j_count = rot_len // 6

        pred6 = out_direct_raw[..., r0:r1].reshape(1, int(out_direct_raw.shape[1]), j_count, 6)
        gt6 = gt_raw[..., r0:r1].reshape(1, int(gt_raw.shape[1]), j_count, 6)

        rp = rot6d_to_matrix(pred6)
        rg = rot6d_to_matrix(gt6)
        rerr = torch.matmul(rp.transpose(-1, -2), rg)
        rotvec_deg = so3_log_map(rerr) * (180.0 / math.pi)
        ang_deg = geodesic_R(rp, rg) * (180.0 / math.pi)

    return {
        "rotvec_deg": rotvec_deg.detach().cpu().numpy()[0],  # (T,J,3)
        "ang_deg": ang_deg.detach().cpu().numpy()[0],  # (T,J)
        "time_steps": int(out_direct.shape[1]),
        "joint_count": int(j_count),
    }


def _joint_index(bone_names: Sequence[str], name: str) -> int:
    target = str(name).strip()
    for i, b in enumerate(bone_names):
        if str(b) == target:
            return int(i)
    raise KeyError(f"Joint {name} not found in bone_names.")


def _extract_proxy_arrays(
    freerun_payload: Dict[str, Any],
    *,
    branch: str,
    space: str,
) -> Dict[str, Any]:
    pjs = freerun_payload.get("per_step_joint_so3_error")
    if not isinstance(pjs, dict):
        raise RuntimeError(
            "freerun JSON missing per_step_joint_so3_error; re-run with "
            "--export_per_step_joint_so3_error --export_per_step_joint_so3_branches direct --per_step_joint_so3_space body"
        )
    branches = pjs.get("branches")
    if not isinstance(branches, dict):
        raise RuntimeError("per_step_joint_so3_error.branches missing.")
    b = branches.get(str(branch))
    if not isinstance(b, dict):
        raise RuntimeError(f"per_step_joint_so3_error.branches missing branch={branch!r}.")
    s = b.get(str(space))
    if not isinstance(s, dict):
        raise RuntimeError(f"per_step_joint_so3_error.branches[{branch!r}] missing space={space!r}.")

    rotvec = np.asarray(s.get("rotvec_deg_xyz"), dtype=np.float32)
    if rotvec.ndim != 3 or rotvec.shape[-1] != 3:
        raise RuntimeError(
            f"Unexpected rotvec_deg_xyz shape={rotvec.shape}; expected (steps,joints,3)."
        )

    mps = freerun_payload.get("metrics_per_step")
    if not isinstance(mps, list) or (not mps):
        raise RuntimeError("freerun JSON missing metrics_per_step for cycle/sic masking.")
    n = min(int(rotvec.shape[0]), int(len(mps)))
    rotvec = rotvec[:n]
    mps = mps[:n]

    step_vals: List[int] = []
    cycle_vals: List[int] = []
    step_defaulted = 0
    cycle_defaulted = 0
    for r in mps:
        sic_i, sic_defaulted = _coerce_int(r.get("step_in_cycle", -1), default=-1)
        cyc_i, cyc_defaulted = _coerce_int(r.get("cycle", 0), default=0)
        step_vals.append(int(sic_i))
        cycle_vals.append(int(cyc_i))
        step_defaulted += int(sic_defaulted)
        cycle_defaulted += int(cyc_defaulted)

    step_in_cycle = np.asarray(step_vals, dtype=np.int64)
    cycle = np.asarray(cycle_vals, dtype=np.int64)
    wrap = np.asarray([bool(r.get("wrap_boundary_step", False)) for r in mps], dtype=bool)

    bone_names = [str(x) for x in list(pjs.get("bone_names") or [])]
    return {
        "rotvec_deg": rotvec,
        "step_in_cycle": step_in_cycle,
        "cycle": cycle,
        "wrap_boundary_step": wrap,
        "bone_names": bone_names,
        "steps": int(n),
        "step_in_cycle_defaulted_count": int(step_defaulted),
        "cycle_defaulted_count": int(cycle_defaulted),
    }


def _per_sic_stats(
    *,
    rotvec_deg: np.ndarray,
    step_in_cycle: np.ndarray,
    cycle: np.ndarray,
    wrap_boundary_step: np.ndarray,
    joint_idx: int,
    sics: Sequence[int],
    cycle_gte: int,
    drop_wrap: bool,
) -> Dict[str, Any]:
    n = int(min(rotvec_deg.shape[0], step_in_cycle.shape[0], cycle.shape[0], wrap_boundary_step.shape[0]))
    rotvec_deg = rotvec_deg[:n]
    step_in_cycle = step_in_cycle[:n]
    cycle = cycle[:n]
    wrap_boundary_step = wrap_boundary_step[:n]

    base_mask = np.ones((n,), dtype=bool)
    base_mask &= np.isfinite(step_in_cycle)
    base_mask &= np.isfinite(cycle)
    if int(cycle_gte) > -999999:
        base_mask &= (cycle >= int(cycle_gte))
    if bool(drop_wrap):
        base_mask &= (~wrap_boundary_step)

    rows: List[Dict[str, Any]] = []
    for sic in sics:
        sic_i = int(sic)
        m = base_mask & (step_in_cycle == sic_i)
        cnt = int(m.sum())
        if cnt <= 0:
            rows.append(
                {
                    "sic": sic_i,
                    "count": 0,
                    "mu_xyz_deg": [float("nan"), float("nan"), float("nan")],
                    "mu_norm_deg": float("nan"),
                    "mu_z_deg": float("nan"),
                }
            )
            continue
        eps = rotvec_deg[m, int(joint_idx), :]  # (N,3)
        mu = np.mean(eps, axis=0)
        rows.append(
            {
                "sic": sic_i,
                "count": cnt,
                "mu_xyz_deg": [float(mu[0]), float(mu[1]), float(mu[2])],
                "mu_norm_deg": float(np.linalg.norm(mu)),
                "mu_z_deg": float(mu[2]),
            }
        )

    return {
        "cycle_gte": int(cycle_gte),
        "drop_wrap": bool(drop_wrap),
        "rows": rows,
    }


def _sign_bucket(x: float, eps: float) -> Optional[int]:
    if not math.isfinite(x):
        return None
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _compare_teacher_proxy(
    teacher_rows: Sequence[Dict[str, Any]],
    proxy_rows: Sequence[Dict[str, Any]],
    *,
    sign_eps_deg: float,
    min_abs_mu_z_for_sign_deg: float,
    sign_match_min: float,
    median_diff_max_deg: float,
) -> Dict[str, Any]:
    t_by_sic = {int(r.get("sic")): r for r in teacher_rows}
    p_by_sic = {int(r.get("sic")): r for r in proxy_rows}
    keys = sorted(set(t_by_sic.keys()) & set(p_by_sic.keys()))

    rows: List[Dict[str, Any]] = []
    abs_diffs: List[float] = []
    sign_flags: List[bool] = []
    min_abs_sign = float(max(0.0, float(min_abs_mu_z_for_sign_deg)))
    low_abs_excluded = 0
    for sic in keys:
        tr = t_by_sic[sic]
        pr = p_by_sic[sic]
        tz = float(tr.get("mu_z_deg", float("nan")))
        pz = float(pr.get("mu_z_deg", float("nan")))
        tnorm = float(tr.get("mu_norm_deg", float("nan")))
        pnorm = float(pr.get("mu_norm_deg", float("nan")))
        abs_diff = abs(tz - pz) if (math.isfinite(tz) and math.isfinite(pz)) else float("nan")
        if math.isfinite(abs_diff):
            abs_diffs.append(abs_diff)
        st = _sign_bucket(tz, float(sign_eps_deg))
        sp = _sign_bucket(pz, float(sign_eps_deg))
        sign_eligible = True
        if min_abs_sign > 0.0:
            sign_eligible = bool(
                math.isfinite(tz)
                and math.isfinite(pz)
                and (abs(tz) >= min_abs_sign)
                and (abs(pz) >= min_abs_sign)
            )
            if not sign_eligible:
                low_abs_excluded += 1
        sign_match = bool(st is not None and sp is not None and st == sp)
        sign_counted = bool(sign_eligible and st is not None and sp is not None)
        if sign_counted:
            sign_flags.append(sign_match)

        rows.append(
            {
                "sic": int(sic),
                "teacher_mu_z_deg": tz,
                "proxy_mu_z_deg": pz,
                "teacher_mu_norm_deg": tnorm,
                "proxy_mu_norm_deg": pnorm,
                "abs_mu_z_diff_deg": abs_diff,
                "teacher_sign": st,
                "proxy_sign": sp,
                "sign_match": sign_match,
                "sign_counted": sign_counted,
                "sign_low_abs_excluded": bool(not sign_eligible),
                "teacher_count": int(tr.get("count", 0)),
                "proxy_count": int(pr.get("count", 0)),
            }
        )

    sign_rate = float(np.mean(sign_flags)) if sign_flags else float("nan")
    med_abs = float(np.median(abs_diffs)) if abs_diffs else float("nan")
    passed = bool(
        len(sign_flags) > 0
        and len(abs_diffs) > 0
        and math.isfinite(sign_rate)
        and math.isfinite(med_abs)
        and (sign_rate >= float(sign_match_min))
        and (med_abs <= float(median_diff_max_deg))
    )

    return {
        "rows": rows,
        "sign_match_rate": sign_rate,
        "median_abs_mu_z_diff_deg": med_abs,
        "num_valid_sign": int(len(sign_flags)),
        "num_valid_abs_diff": int(len(abs_diffs)),
        "num_low_abs_sign_excluded": int(low_abs_excluded),
        "criteria": {
            "sign_eps_deg": float(sign_eps_deg),
            "min_abs_mu_z_for_sign_deg": float(min_abs_sign),
            "sign_match_min": float(sign_match_min),
            "median_abs_mu_z_diff_max_deg": float(median_diff_max_deg),
        },
        "gate_pass": bool(passed),
    }


def _build_strict_diag_rows(
    teacher_rows: Sequence[Dict[str, Any]],
    proxy_rows: Sequence[Dict[str, Any]],
    *,
    sign_eps_deg: float,
    min_abs_mu_z_for_sign_deg: float,
) -> Dict[str, Any]:
    t_by_sic = {int(r.get("sic")): r for r in teacher_rows}
    p_by_sic = {int(r.get("sic")): r for r in proxy_rows}
    keys = sorted(set(t_by_sic.keys()) & set(p_by_sic.keys()))
    min_abs_sign = float(max(0.0, float(min_abs_mu_z_for_sign_deg)))
    rows: List[Dict[str, Any]] = []
    n_low_abs = 0
    n_counted = 0
    for sic in keys:
        tr = t_by_sic[sic]
        pr = p_by_sic[sic]
        txyz = np.asarray(tr.get("mu_xyz_deg", [float("nan"), float("nan"), float("nan")]), dtype=np.float64).reshape(-1)
        pxyz = np.asarray(pr.get("mu_xyz_deg", [float("nan"), float("nan"), float("nan")]), dtype=np.float64).reshape(-1)
        if txyz.size < 3:
            txyz = np.asarray([float("nan"), float("nan"), float("nan")], dtype=np.float64)
        if pxyz.size < 3:
            pxyz = np.asarray([float("nan"), float("nan"), float("nan")], dtype=np.float64)
        tz = float(tr.get("mu_z_deg", float("nan")))
        pz = float(pr.get("mu_z_deg", float("nan")))
        st = _sign_bucket(tz, float(sign_eps_deg))
        sp = _sign_bucket(pz, float(sign_eps_deg))
        t_dead = bool(st == 0)
        p_dead = bool(sp == 0)

        low_abs = False
        if min_abs_sign > 0.0:
            low_abs = bool(
                (not math.isfinite(tz))
                or (not math.isfinite(pz))
                or (abs(tz) < min_abs_sign)
                or (abs(pz) < min_abs_sign)
            )
            n_low_abs += int(low_abs)

        sign_counted = bool((not low_abs) and st is not None and sp is not None)
        n_counted += int(sign_counted)
        sign_match = bool(st is not None and sp is not None and st == sp)

        rows.append(
            {
                "sic": int(sic),
                "teacher_count": int(tr.get("count", 0)),
                "proxy_count": int(pr.get("count", 0)),
                "teacher_mu_xyz_deg": [float(txyz[0]), float(txyz[1]), float(txyz[2])],
                "proxy_mu_xyz_deg": [float(pxyz[0]), float(pxyz[1]), float(pxyz[2])],
                "teacher_mu_norm_deg": float(tr.get("mu_norm_deg", float("nan"))),
                "proxy_mu_norm_deg": float(pr.get("mu_norm_deg", float("nan"))),
                "teacher_mu_z_deg": float(tz),
                "proxy_mu_z_deg": float(pz),
                "teacher_sign_bucket": st,
                "proxy_sign_bucket": sp,
                "teacher_dead_zone": t_dead,
                "proxy_dead_zone": p_dead,
                "low_abs_excluded": bool(low_abs),
                "sign_counted": bool(sign_counted),
                "sign_match": bool(sign_match),
                "abs_mu_z_diff_deg": abs(tz - pz) if (math.isfinite(tz) and math.isfinite(pz)) else float("nan"),
            }
        )

    return {
        "rows": rows,
        "sign_eps_deg": float(sign_eps_deg),
        "min_abs_mu_z_for_sign_deg": float(min_abs_sign),
        "num_rows": int(len(rows)),
        "num_low_abs_excluded": int(n_low_abs),
        "num_sign_counted": int(n_counted),
    }


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _fmt_float(v: Any) -> str:
    try:
        f = float(v)
    except Exception:
        return "nan"
    if not math.isfinite(f):
        return "nan"
    return f"{f:.3f}"


def _write_md(
    out_path: Path,
    *,
    out: Dict[str, Any],
) -> None:
    cfg = out.get("config", {})
    cmp = out.get("compare", {})
    t = out.get("teacher", {})
    p = out.get("proxy", {})

    lines: List[str] = []
    lines.append("# H1 10.3.A Epoch0 Gate (Teacher vs Proxy)")
    lines.append("")
    lines.append(f"- model: `{cfg.get('model')}`")
    lines.append(f"- teacher: `{cfg.get('teacher')}`")
    lines.append(f"- proxy freerun: `{cfg.get('proxy_freerun')}`")
    lines.append(f"- joint: `{cfg.get('joint')}`")
    lines.append(f"- hotspot_sics: `{cfg.get('hotspot_sics')}`")
    lines.append("")
    lines.append("## Gate Verdict")
    lines.append("")
    lines.append(f"- sign_match_rate: {cmp.get('sign_match_rate')}")
    lines.append(f"- median_abs_mu_z_diff_deg: {cmp.get('median_abs_mu_z_diff_deg')}")
    lines.append(f"- num_valid_sign: {cmp.get('num_valid_sign')}")
    lines.append(f"- num_low_abs_sign_excluded: {cmp.get('num_low_abs_sign_excluded')}")
    lines.append(f"- gate_pass: {cmp.get('gate_pass')}")
    lines.append("")
    crit = cmp.get("criteria", {})
    lines.append("## Sign Criteria")
    lines.append("")
    lines.append(f"- sign_eps_deg: {crit.get('sign_eps_deg')}")
    lines.append(f"- min_abs_mu_z_for_sign_deg: {crit.get('min_abs_mu_z_for_sign_deg')}")
    lines.append("")
    lines.append("## Teacher / Proxy Filters")
    lines.append("")
    lines.append(
        f"- teacher filter: cycle>= {t.get('cycle_gte')}, drop_wrap={t.get('drop_wrap')}"
    )
    lines.append(
        f"- proxy filter: cycle>= {p.get('cycle_gte')}, drop_wrap={p.get('drop_wrap')}"
    )
    lines.append("")
    lines.append("## Hotspot Per-SIC")
    lines.append("")
    lines.append("| sic | teacher_mu_z | proxy_mu_z | abs_diff | teacher_||mu|| | proxy_||mu|| | sign_match | sign_counted | low_abs_excluded | n_t | n_p |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in cmp.get("rows", []):
        lines.append(
            f"| {int(r.get('sic'))} | "
            f"{_fmt_float(r.get('teacher_mu_z_deg'))} | "
            f"{_fmt_float(r.get('proxy_mu_z_deg'))} | "
            f"{_fmt_float(r.get('abs_mu_z_diff_deg'))} | "
            f"{_fmt_float(r.get('teacher_mu_norm_deg'))} | "
            f"{_fmt_float(r.get('proxy_mu_norm_deg'))} | "
            f"{str(bool(r.get('sign_match')))} | "
            f"{str(bool(r.get('sign_counted')))} | "
            f"{str(bool(r.get('sign_low_abs_excluded')))} | "
            f"{int(r.get('teacher_count'))} | "
            f"{int(r.get('proxy_count'))} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_strict_diag_md(
    out_path: Path,
    *,
    out: Dict[str, Any],
) -> None:
    cfg = out.get("config", {})
    diag = out.get("strict_diag", {})
    rows = list(diag.get("rows") or [])
    lines: List[str] = []
    lines.append("# H1 10.3.A Strict Aperture Diagnostic")
    lines.append("")
    lines.append(f"- model: `{cfg.get('model')}`")
    lines.append(f"- teacher: `{cfg.get('teacher')}`")
    lines.append(f"- proxy freerun: `{cfg.get('proxy_freerun')}`")
    lines.append(f"- joint: `{cfg.get('joint')}`")
    lines.append(f"- hotspot_sics: `{cfg.get('hotspot_sics')}`")
    lines.append("")
    lines.append("## Shared Filter")
    lines.append("")
    lines.append(f"- cycle>= {diag.get('cycle_gte')}")
    lines.append(f"- drop_wrap={diag.get('drop_wrap')}")
    lines.append(f"- sign_eps_deg={diag.get('sign_eps_deg')}")
    lines.append(f"- min_abs_mu_z_for_sign_deg={diag.get('min_abs_mu_z_for_sign_deg')}")
    lines.append("")
    lines.append("## Per-SIC")
    lines.append("")
    lines.append("| sic | t_mu_xyz_deg | p_mu_xyz_deg | t_|mu| | p_|mu| | t_mu_z | p_mu_z | t_sign | p_sign | t_dead | p_dead | low_abs | sign_counted | sign_match | n_t | n_p |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        txyz = r.get("teacher_mu_xyz_deg", [float("nan"), float("nan"), float("nan")])
        pxyz = r.get("proxy_mu_xyz_deg", [float("nan"), float("nan"), float("nan")])
        txyz_text = f"[{_fmt_float(txyz[0])}, {_fmt_float(txyz[1])}, {_fmt_float(txyz[2])}]"
        pxyz_text = f"[{_fmt_float(pxyz[0])}, {_fmt_float(pxyz[1])}, {_fmt_float(pxyz[2])}]"
        lines.append(
            f"| {int(r.get('sic'))} | "
            f"{txyz_text} | "
            f"{pxyz_text} | "
            f"{_fmt_float(r.get('teacher_mu_norm_deg'))} | "
            f"{_fmt_float(r.get('proxy_mu_norm_deg'))} | "
            f"{_fmt_float(r.get('teacher_mu_z_deg'))} | "
            f"{_fmt_float(r.get('proxy_mu_z_deg'))} | "
            f"{r.get('teacher_sign_bucket')} | "
            f"{r.get('proxy_sign_bucket')} | "
            f"{str(bool(r.get('teacher_dead_zone')))} | "
            f"{str(bool(r.get('proxy_dead_zone')))} | "
            f"{str(bool(r.get('low_abs_excluded')))} | "
            f"{str(bool(r.get('sign_counted')))} | "
            f"{str(bool(r.get('sign_match')))} | "
            f"{int(r.get('teacher_count', 0))} | "
            f"{int(r.get('proxy_count', 0))} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run H1 10.3.A epoch-0 gate: true teacher vs freerun_x_gt proxy per-sic alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", type=str, required=True, help="Checkpoint path used for teacher one-step forward.")
    ap.add_argument("--teacher", type=str, required=True, help="Teacher batch JSON (e.g., validate/teacher_batches/Walk_F_teacher.json).")
    ap.add_argument("--proxy-freerun", type=str, required=True, help="freerun_x_gt JSON with per_step_joint_so3_error export.")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--joint", type=str, default="calf_r")
    ap.add_argument("--hotspot-sics", type=str, default="9-14,39-42")
    ap.add_argument("--teacher-cycle-gte", type=int, default=0, help="Teacher-side cycle filter (0 keeps all teacher one-step rows).")
    ap.add_argument("--proxy-cycle-gte", type=int, default=1, help="Proxy-side cycle filter (doc default: cycle>=1).")
    ap.add_argument("--teacher-drop-wrap", type=str, default="false", help="true|false")
    ap.add_argument("--proxy-drop-wrap", type=str, default="true", help="true|false")
    ap.add_argument("--proxy-branch", type=str, default="direct", help="Branch in per_step_joint_so3_error.branches")
    ap.add_argument("--proxy-space", type=str, default="body", choices=("body", "world"))
    ap.add_argument("--sign-eps-deg", type=float, default=0.25, help="Sign dead-zone (deg).")
    ap.add_argument(
        "--min-abs-mu-z-for-sign",
        type=float,
        default=0.0,
        help="If >0, SIC is excluded from sign denominator when |teacher_mu_z| or |proxy_mu_z| is below this threshold (deg).",
    )
    ap.add_argument("--sign-match-min", type=float, default=0.8, help="Gate threshold for sign consistency.")
    ap.add_argument("--median-diff-max-deg", type=float, default=1.0, help="Gate threshold for median |teacher-proxy| on mu_z.")
    ap.add_argument(
        "--strict-diag-cycle-gte",
        type=int,
        default=0,
        help="Shared cycle filter for strict aperture diagnostic export (applied to both teacher/proxy).",
    )
    ap.add_argument(
        "--strict-diag-drop-wrap",
        type=str,
        default="false",
        help="Shared drop_wrap for strict aperture diagnostic export (true|false).",
    )
    ap.add_argument("--out-dir", type=str, default="debug_output/h1_10p3a_gate")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hotspot_sics = _parse_index_spec(args.hotspot_sics)
    if not hotspot_sics:
        raise SystemExit("[FATAL] --hotspot-sics resolved to empty.")

    runner = _build_runner(args)
    teacher_path = Path(args.teacher).expanduser().resolve()
    proxy_path = Path(args.proxy_freerun).expanduser().resolve()

    clip_data = _load_teacher_sample(
        runner,
        teacher_path=teacher_path,
        npz_root=Path(args.npz_root).expanduser().resolve(),
    )
    teacher = _forward_teacher_rotvec_deg(runner, clip_data["sample"])
    teacher_bones = list(clip_data.get("bone_names") or [])
    if not teacher_bones:
        raise SystemExit("[FATAL] teacher sample has empty bone_names.")
    j_teacher = _joint_index(teacher_bones, str(args.joint))

    t_steps = int(teacher["rotvec_deg"].shape[0])
    teacher_step_in_cycle = np.arange(t_steps, dtype=np.int64)
    teacher_cycle = np.zeros((t_steps,), dtype=np.int64)
    teacher_wrap = np.zeros((t_steps,), dtype=bool)
    t_stats = _per_sic_stats(
        rotvec_deg=teacher["rotvec_deg"],
        step_in_cycle=teacher_step_in_cycle,
        cycle=teacher_cycle,
        wrap_boundary_step=teacher_wrap,
        joint_idx=j_teacher,
        sics=hotspot_sics,
        cycle_gte=int(args.teacher_cycle_gte),
        drop_wrap=_parse_bool_text(args.teacher_drop_wrap, False),
    )

    proxy_payload = _load_json(proxy_path)
    proxy = _extract_proxy_arrays(
        proxy_payload,
        branch=str(args.proxy_branch),
        space=str(args.proxy_space),
    )
    proxy_bones = list(proxy.get("bone_names") or [])
    if proxy_bones:
        j_proxy = _joint_index(proxy_bones, str(args.joint))
    else:
        # Fallback: use teacher bone ordering when exporter omitted names.
        j_proxy = int(j_teacher)

    p_stats = _per_sic_stats(
        rotvec_deg=proxy["rotvec_deg"],
        step_in_cycle=proxy["step_in_cycle"],
        cycle=proxy["cycle"],
        wrap_boundary_step=proxy["wrap_boundary_step"],
        joint_idx=j_proxy,
        sics=hotspot_sics,
        cycle_gte=int(args.proxy_cycle_gte),
        drop_wrap=_parse_bool_text(args.proxy_drop_wrap, True),
    )

    cmp = _compare_teacher_proxy(
        t_stats["rows"],
        p_stats["rows"],
        sign_eps_deg=float(args.sign_eps_deg),
        min_abs_mu_z_for_sign_deg=float(args.min_abs_mu_z_for_sign),
        sign_match_min=float(args.sign_match_min),
        median_diff_max_deg=float(args.median_diff_max_deg),
    )

    strict_cycle_gte = int(args.strict_diag_cycle_gte)
    strict_drop_wrap = bool(_parse_bool_text(args.strict_diag_drop_wrap, False))
    t_stats_strict = _per_sic_stats(
        rotvec_deg=teacher["rotvec_deg"],
        step_in_cycle=teacher_step_in_cycle,
        cycle=teacher_cycle,
        wrap_boundary_step=teacher_wrap,
        joint_idx=j_teacher,
        sics=hotspot_sics,
        cycle_gte=strict_cycle_gte,
        drop_wrap=strict_drop_wrap,
    )
    p_stats_strict = _per_sic_stats(
        rotvec_deg=proxy["rotvec_deg"],
        step_in_cycle=proxy["step_in_cycle"],
        cycle=proxy["cycle"],
        wrap_boundary_step=proxy["wrap_boundary_step"],
        joint_idx=j_proxy,
        sics=hotspot_sics,
        cycle_gte=strict_cycle_gte,
        drop_wrap=strict_drop_wrap,
    )
    strict_diag = _build_strict_diag_rows(
        t_stats_strict["rows"],
        p_stats_strict["rows"],
        sign_eps_deg=float(args.sign_eps_deg),
        min_abs_mu_z_for_sign_deg=float(args.min_abs_mu_z_for_sign),
    )

    out = {
        "config": {
            "model": str(Path(args.model).expanduser().resolve()),
            "teacher": str(teacher_path),
            "proxy_freerun": str(proxy_path),
            "joint": str(args.joint),
            "hotspot_sics": hotspot_sics,
            "proxy_branch": str(args.proxy_branch),
            "proxy_space": str(args.proxy_space),
            "teacher_cycle_gte": int(args.teacher_cycle_gte),
            "proxy_cycle_gte": int(args.proxy_cycle_gte),
            "teacher_drop_wrap": bool(_parse_bool_text(args.teacher_drop_wrap, False)),
            "proxy_drop_wrap": bool(_parse_bool_text(args.proxy_drop_wrap, True)),
            "sign_eps_deg": float(args.sign_eps_deg),
            "min_abs_mu_z_for_sign": float(args.min_abs_mu_z_for_sign),
            "sign_match_min": float(args.sign_match_min),
            "median_diff_max_deg": float(args.median_diff_max_deg),
            "strict_diag_cycle_gte": int(args.strict_diag_cycle_gte),
            "strict_diag_drop_wrap": bool(_parse_bool_text(args.strict_diag_drop_wrap, False)),
        },
        "teacher": {
            "clip": clip_data.get("clip"),
            "joint": str(args.joint),
            "joint_idx": int(j_teacher),
            "num_steps": int(teacher.get("time_steps", 0)),
            **t_stats,
        },
        "proxy": {
            "joint": str(args.joint),
            "joint_idx": int(j_proxy),
            "num_steps": int(proxy.get("steps", 0)),
            "step_in_cycle_defaulted_count": int(proxy.get("step_in_cycle_defaulted_count", 0)),
            "cycle_defaulted_count": int(proxy.get("cycle_defaulted_count", 0)),
            **p_stats,
        },
        "compare": cmp,
        "strict_diag": {
            "cycle_gte": int(strict_cycle_gte),
            "drop_wrap": bool(strict_drop_wrap),
            **strict_diag,
        },
    }

    out_json_path = out_dir / "h1_10p3a_gate_summary.json"
    out_md_path = out_dir / "h1_10p3a_gate_summary.md"
    out_strict_json_path = out_dir / "h1_10p3a_gate_strict_diag.json"
    out_strict_md_path = out_dir / "h1_10p3a_gate_strict_diag.md"
    out_json_path.write_text(json.dumps(_to_jsonable(out), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(out_md_path, out=out)
    out_strict_json_path.write_text(
        json.dumps(_to_jsonable({"config": out["config"], "strict_diag": out["strict_diag"]}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_strict_diag_md(out_strict_md_path, out=out)

    print(f"[OK] wrote {out_json_path}")
    print(f"[OK] wrote {out_md_path}")
    print(f"[OK] wrote {out_strict_json_path}")
    print(f"[OK] wrote {out_strict_md_path}")
    print(
        f"[gate] pass={bool(cmp.get('gate_pass'))} "
        f"sign={cmp.get('sign_match_rate')} "
        f"med_abs={cmp.get('median_abs_mu_z_diff_deg')}"
    )


if __name__ == "__main__":
    main()
