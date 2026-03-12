#!/usr/bin/env python3
"""Diagnose jac_RL_all conditioned on direction-sign subsets.

Given a rho-json (from diagnose_cond_rho_delta.py), rebuild each diag batch,
measure direct-head Jacobian R/L ratio on all direct-head inputs:

    jac_RL_all = ||d y_R / d x|| / ||d y_L / d x||

Then report subset aggregates conditioned by sign source:
- raw_c5       : sign of world c5
- dir_body_y   : sign of direction projected into pelvis/body frame

This answers whether asymmetry is direction-consistent (one side >1, other <1).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import diagnose_cond_rho_c5_signsplit as signsplit
import diagnose_cond_rho_delta as base
from train.dataset import MotionEventDataset


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    if not (math.isfinite(num) and math.isfinite(den)):
        return float("nan")
    if abs(den) <= eps:
        return float("nan")
    return float(num / den)


def _sign_test_two_sided(pos_n: int, neg_n: int) -> float:
    n = int(pos_n) + int(neg_n)
    if n <= 0:
        return float("nan")
    k = int(pos_n)
    cdf_lo = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    sf_hi = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * min(cdf_lo, sf_hi)))


def _parse_int_csv(spec: str) -> Set[int]:
    out: Set[int] = set()
    for tok in str(spec or "").split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            out.add(int(s))
        except Exception:
            raise SystemExit(f"[FATAL] invalid integer token in --clip-ids: {s}")
    return out


def _parse_lower_csv(spec: str) -> List[str]:
    out: List[str] = []
    for tok in str(spec or "").split(","):
        s = tok.strip().lower()
        if s:
            out.append(s)
    return out


def _diag_clip_ids(diag_pt: Path) -> List[int]:
    payload = torch.load(str(diag_pt), map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"[FATAL] invalid diag pt payload: {diag_pt}")
    clip_id = payload.get("clip_id")
    if not torch.is_tensor(clip_id):
        raise SystemExit(f"[FATAL] diag pt missing clip_id: {diag_pt}")
    return [int(x) for x in clip_id.view(-1).tolist()]


def _slice_batch_rows(batch: Dict[str, torch.Tensor], keep_rows: List[int]) -> Dict[str, torch.Tensor]:
    if not batch:
        return batch
    any_tensor = next((v for v in batch.values() if torch.is_tensor(v) and v.dim() >= 1), None)
    if any_tensor is None:
        return batch
    bsz = int(any_tensor.shape[0])
    if not keep_rows:
        raise SystemExit("[FATAL] empty keep_rows for batch slicing")
    idx = torch.as_tensor(keep_rows, dtype=torch.long, device=any_tensor.device)
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() >= 1 and int(v.shape[0]) == bsz:
            out[k] = v.index_select(0, idx)
        else:
            out[k] = v
    return out


def _build_clip_path_by_id(
    *,
    seq_len: int,
    data_root: Path,
    bundle: Path,
    pretrain_template: Path,
) -> Dict[int, str]:
    norm_spec = base._merge_norm_spec(bundle.resolve(), pretrain_template.resolve())
    ds = MotionEventDataset(
        data_dir=str(data_root.resolve()),
        seq_len=int(seq_len),
        paths=None,
        pose_hist_len=int(norm_spec.get("pose_hist_len", 0) or 0),
        norm_spec=norm_spec,
        index_mode="sliding",
    )
    paths = list(getattr(ds, "paths", []) or [])
    return {int(i): str(p) for i, p in enumerate(paths)}


def _measure_batch(
    *,
    model: torch.nn.Module,
    trainer: Any,
    batch: Dict[str, torch.Tensor],
    steps_spec: str,
    left_slices: List[slice],
    right_slices: List[slice],
    sign_source: str,
    pelvis_orientation_source: str,
    label_time: str,
    sign_eps: float,
) -> Dict[str, Any]:
    first_linear = signsplit._find_first_linear(getattr(model, "direct_pose_head", None))
    state = batch.get("motion")
    cond = batch.get("cond_in")
    contacts = batch.get("contacts")
    angvel = batch.get("angvel")
    pose_hist = batch.get("pose_hist")
    if not (torch.is_tensor(state) and torch.is_tensor(cond)):
        raise SystemExit("[FATAL] missing motion/cond_in")

    capture: Dict[str, torch.Tensor] = {}

    def _pre_hook(mod: torch.nn.Module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if not torch.is_tensor(x):
            return inputs
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        capture["x"] = x
        if len(inputs) == 1:
            return (x,)
        return (x, *inputs[1:])

    model.eval()
    model.zero_grad(set_to_none=True)
    hook = first_linear.register_forward_pre_hook(_pre_hook)

    rows: List[Dict[str, Any]] = []
    subset_keys: List[str] = []
    sign_info: Dict[str, Any] = {}
    try:
        with torch.enable_grad():
            use_learned_meas = bool(getattr(model, "contact_meas_enable", False)) and getattr(model, "contact_meas_head", None) is not None
            contacts_in = None if use_learned_meas else contacts
            ret = model(
                state,
                cond=cond,
                contacts=contacts_in,
                angvel=angvel,
                pose_history=pose_hist,
                plan_z=None,
                time_index=None,
            )
            out_direct = ret.get("out_direct") if isinstance(ret, dict) else None
            if not (torch.is_tensor(out_direct) and out_direct.dim() == 3):
                raise SystemExit("[FATAL] invalid out_direct")

            x = capture.get("x")
            if not torch.is_tensor(x):
                raise SystemExit("[FATAL] failed to capture direct head input")

            sign_feature, sign_key, sign_info = signsplit._resolve_sign_feature(
                trainer=trainer,
                batch=batch,
                ret=ret if isinstance(ret, dict) else {},
                sign_source=sign_source,
                pelvis_orientation_source=pelvis_orientation_source,
                label_time=label_time,
            )
            if int(sign_feature.shape[0]) != int(x.shape[0]):
                raise SystemExit(
                    f"[FATAL] sign feature rows ({sign_feature.shape[0]}) != x rows ({int(x.shape[0])})"
                )
            masks_np = signsplit._subset_masks(sign_feature, sign_key=sign_key, sign_eps=float(sign_eps))
            masks = {k: torch.from_numpy(v.astype(np.bool_)).to(x.device) for k, v in masks_np.items()}
            subset_keys = list(masks_np.keys())

            steps = base._parse_steps(steps_spec, int(out_direct.shape[1]))
            if not steps:
                raise SystemExit("[FATAL] no valid steps")

            for i, t in enumerate(steps):
                y_t = out_direct[:, int(t), :]
                loss_left = sum(y_t[:, sl].sum() for sl in left_slices)
                loss_right = sum(y_t[:, sl].sum() for sl in right_slices)
                g_left = torch.autograd.grad(loss_left, x, retain_graph=True, create_graph=False, allow_unused=False)[0]
                g_right = torch.autograd.grad(
                    loss_right,
                    x,
                    retain_graph=(i < len(steps) - 1),
                    create_graph=False,
                    allow_unused=False,
                )[0]

                rec: Dict[str, Any] = {"step": int(t), "subsets": {}}
                for name, m in masks.items():
                    n_eff = int(m.sum().detach().cpu())
                    if n_eff <= 0:
                        rec["subsets"][name] = {"n": 0, "jac_rl_all": float("nan")}
                        continue
                    l_all = float(g_left[m].norm().detach().cpu())
                    r_all = float(g_right[m].norm().detach().cpu())
                    rec["subsets"][name] = {
                        "n": n_eff,
                        "jac_rl_all": _safe_ratio(r_all, l_all),
                        "left_norm_all": l_all,
                        "right_norm_all": r_all,
                    }
                rows.append(rec)
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)

    agg: Dict[str, Any] = {}
    for name in subset_keys:
        vals = np.asarray([
            _safe_float(r.get("subsets", {}).get(name, {}).get("jac_rl_all", float("nan")))
            for r in rows
        ], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        n_rows = np.asarray([
            _safe_float(r.get("subsets", {}).get(name, {}).get("n", 0))
            for r in rows
        ], dtype=np.float64)
        n_rows = n_rows[np.isfinite(n_rows)]
        agg[name] = {
            "jac_rl_all_mean": float(vals.mean()) if vals.size > 0 else float("nan"),
            "jac_rl_all_std": float(vals.std()) if vals.size > 0 else float("nan"),
            "n_steps_effective": int(vals.size),
            "n_rows_mean": float(n_rows.mean()) if n_rows.size > 0 else float("nan"),
        }

    return {
        "sign_info": sign_info,
        "subset_keys": subset_keys,
        "steps": [int(r.get("step", -1)) for r in rows],
        "per_step": rows,
        "aggregate": agg,
        "direct_head_input_shape": list(capture.get("x").shape) if torch.is_tensor(capture.get("x")) else [],
    }


def _to_markdown(out: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Jacobian RL by Direction Sign")
    lines.append("")
    lines.append(f"- source rho json: `{out.get('source_rho_json', '')}`")
    lines.append(f"- model: `{out.get('model', '')}`")
    lines.append(f"- diag batches: `{len(out.get('diag_pts', []))}`")
    lines.append(f"- steps: `{out.get('steps', [])}`")
    lines.append(f"- sign source: `{out.get('sign_source', '')}`")
    lines.append(f"- pelvis orientation source: `{out.get('pelvis_orientation_source', '')}`")
    lines.append(f"- label time: `{out.get('label_time', '')}`")
    clip_filter = out.get("clip_filter", {}) if isinstance(out.get("clip_filter"), dict) else {}
    lines.append(f"- clip filter enabled: `{bool(clip_filter.get('enabled', False))}`")
    if bool(clip_filter.get("enabled", False)):
        lines.append(f"- clip ids: `{clip_filter.get('clip_ids', [])}`")
        lines.append(f"- clip name contains: `{clip_filter.get('clip_name_contains', [])}`")
    lines.append("")

    agg = out.get("aggregate", {}) if isinstance(out.get("aggregate"), dict) else {}
    subset_keys = out.get("subset_keys", []) if isinstance(out.get("subset_keys"), list) else []
    if agg and subset_keys:
        lines.append("## Aggregate")
        lines.append("")
        lines.append("|subset|jac_RL_all mean|std|`>1` n|`<1` n|p(two-sided)|")
        lines.append("|:--|--:|--:|--:|--:|--:|")
        for k in subset_keys:
            obj = agg.get(k, {}) if isinstance(agg.get(k), dict) else {}
            lines.append(
                f"|`{k}`|{_safe_float(obj.get('jac_rl_all_mean', float('nan'))):.6f}|"
                f"{_safe_float(obj.get('jac_rl_all_std', float('nan'))):.6f}|"
                f"{int(obj.get('gt1_n', 0) or 0)}|{int(obj.get('lt1_n', 0) or 0)}|"
                f"{_safe_float(obj.get('sign_test_p_two_sided', float('nan'))):.4f}|"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose jac_RL_all split by direction sign.")
    ap.add_argument("--rho-json", type=str, required=True)
    ap.add_argument("--teacher", type=str, default="validate/teacher_batches/Walk_F_teacher.json")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--npz-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--data-root", type=str, default="raw_data/processed_data")
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=str, default="", help="Override steps; default uses rho-json steps")
    ap.add_argument("--left-bones", type=str, default="thigh_l,calf_l,foot_l,ball_l")
    ap.add_argument("--right-bones", type=str, default="thigh_r,calf_r,foot_r,ball_r")
    ap.add_argument("--sign-source", type=str, default="dir_body_y", choices=("raw_c5", "dir_body_y"))
    ap.add_argument("--pelvis-orientation-source", type=str, default="gt", choices=("gt", "pred"))
    ap.add_argument("--label-time", type=str, default="t", choices=("t", "t1"))
    ap.add_argument("--sign-eps", type=float, default=1e-6)
    ap.add_argument(
        "--clip-ids",
        type=str,
        default="",
        help="Optional clip-id filter (comma-separated), applied inside each diag batch.",
    )
    ap.add_argument(
        "--clip-name-contains",
        type=str,
        default="",
        help="Optional lowercase-substring filter on clip npz path (comma-separated).",
    )
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")
    args = ap.parse_args()

    rho_path = Path(args.rho_json).expanduser().resolve()
    rho_payload = json.loads(rho_path.read_text(encoding="utf-8"))
    model_path = str(rho_payload.get("model", "")).strip()
    diag_pts = [str(x) for x in (rho_payload.get("diag_pts", []) or [])]
    if not model_path:
        raise SystemExit("[FATAL] model missing in rho-json")
    if not diag_pts:
        raise SystemExit("[FATAL] diag_pts missing in rho-json")

    runner_args = argparse.Namespace(**vars(args))
    runner_args.model = model_path
    runner, ds_one, _ = base._load_runner(runner_args)
    model = runner.model
    trainer = runner.trainer
    if model is None or trainer is None:
        raise SystemExit("[FATAL] failed to load model/trainer")

    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        raise SystemExit("[FATAL] trainer rot6d slice missing")
    dy = int(getattr(ds_one, "Y", np.zeros((1, 1), dtype=np.float32)).shape[-1])
    st = int(rot_slice.start or 0)
    ed = int(rot_slice.stop or dy)
    if ed <= st or ((ed - st) % 6) != 0:
        raise SystemExit(f"[FATAL] invalid rot6d slice [{st}:{ed}]")
    joint_count = (ed - st) // 6
    bone_names = list(getattr(ds_one, "bone_names", []) or [])[: int(joint_count)]
    name_to_idx = {str(n): i for i, n in enumerate(bone_names)}

    left_bones = base._parse_csv(args.left_bones)
    right_bones = base._parse_csv(args.right_bones)
    if len(left_bones) != len(right_bones):
        raise SystemExit("[FATAL] left/right bones length mismatch")
    miss = [b for b in (left_bones + right_bones) if b not in name_to_idx]
    if miss:
        raise SystemExit(f"[FATAL] unresolved bones: {miss}")
    left_slices = [base._joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in left_bones]
    right_slices = [base._joint_rot6d_slice(rot_slice, name_to_idx[b]) for b in right_bones]

    steps_spec = str(args.steps or "").strip()
    if not steps_spec:
        js_steps = rho_payload.get("steps", [])
        if isinstance(js_steps, list) and js_steps:
            steps_spec = ",".join(str(int(x)) for x in js_steps)
        else:
            steps_spec = "0,1"

    clip_ids_filter = _parse_int_csv(str(args.clip_ids or ""))
    clip_name_filter = _parse_lower_csv(str(args.clip_name_contains or ""))
    apply_clip_filter = bool(clip_ids_filter or clip_name_filter)
    clip_path_by_id: Dict[int, str] = {}
    if apply_clip_filter:
        clip_path_by_id = _build_clip_path_by_id(
            seq_len=int(args.seq_len),
            data_root=Path(args.data_root).expanduser().resolve(),
            bundle=Path(args.bundle).expanduser().resolve(),
            pretrain_template=Path(args.pretrain_template).expanduser().resolve(),
        )

    def _keep_clip(cid: int) -> bool:
        if not apply_clip_filter:
            return True
        if clip_ids_filter and cid not in clip_ids_filter:
            return False
        if clip_name_filter:
            p = str(clip_path_by_id.get(int(cid), "")).lower()
            if not p:
                return False
            if not any(tok in p for tok in clip_name_filter):
                return False
        return True

    per_batch: List[Dict[str, Any]] = []
    expected_subset_keys: Optional[List[str]] = None
    skipped_batches: List[Dict[str, Any]] = []
    for dp in diag_pts:
        diag_pt = Path(dp).expanduser().resolve()
        if not diag_pt.is_file():
            raise SystemExit(f"[FATAL] diag pt missing: {diag_pt}")
        clip_ids = _diag_clip_ids(diag_pt)
        keep_rows = [i for i, cid in enumerate(clip_ids) if _keep_clip(int(cid))]
        if apply_clip_filter and not keep_rows:
            skipped_batches.append(
                {
                    "diag_pt": str(diag_pt),
                    "reason": "no rows matched clip filter",
                    "rows_total": int(len(clip_ids)),
                }
            )
            continue
        batch = base._rebuild_batch_from_diag(
            diag_pt=diag_pt,
            seq_len=int(args.seq_len),
            data_root=Path(args.data_root).expanduser().resolve(),
            bundle=Path(args.bundle).expanduser(),
            pretrain_template=Path(args.pretrain_template).expanduser(),
            device=runner.device,
        )
        rows_total = int(len(clip_ids))
        rows_kept = int(len(keep_rows)) if apply_clip_filter else rows_total
        if apply_clip_filter and rows_kept < rows_total:
            batch = _slice_batch_rows(batch, keep_rows)
        ret = _measure_batch(
            model=model,
            trainer=trainer,
            batch=batch,
            steps_spec=steps_spec,
            left_slices=left_slices,
            right_slices=right_slices,
            sign_source=str(args.sign_source),
            pelvis_orientation_source=str(args.pelvis_orientation_source),
            label_time=str(args.label_time),
            sign_eps=float(args.sign_eps),
        )
        subset_keys = list(ret.get("subset_keys", []) or [])
        if not subset_keys:
            raise SystemExit("[FATAL] empty subset keys")
        if expected_subset_keys is None:
            expected_subset_keys = list(subset_keys)
        elif subset_keys != expected_subset_keys:
            raise SystemExit(
                f"[FATAL] inconsistent subset keys across batches: {subset_keys} vs {expected_subset_keys}"
            )
        per_batch.append(
            {
                "diag_pt": str(diag_pt),
                "rows_total": rows_total,
                "rows_kept": rows_kept,
                **ret,
            }
        )

    if not per_batch:
        raise SystemExit("[FATAL] no diag batch left after clip filtering")

    subset_keys = list(expected_subset_keys or [])
    agg: Dict[str, Any] = {}
    for k in subset_keys:
        arr = np.asarray([
            _safe_float(r.get("aggregate", {}).get(k, {}).get("jac_rl_all_mean", float("nan")))
            for r in per_batch
        ], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        gt1_n = int(np.sum(arr > 1.0))
        lt1_n = int(np.sum(arr < 1.0))
        agg[k] = {
            "jac_rl_all_mean": float(arr.mean()) if arr.size > 0 else float("nan"),
            "jac_rl_all_std": float(arr.std()) if arr.size > 0 else float("nan"),
            "n_batches_effective": int(arr.size),
            "gt1_n": gt1_n,
            "lt1_n": lt1_n,
            "sign_test_p_two_sided": _sign_test_two_sided(gt1_n, lt1_n),
        }

    out = {
        "source_rho_json": str(rho_path),
        "model": model_path,
        "diag_pts": [str(Path(x).expanduser().resolve()) for x in diag_pts],
        "steps": [int(x) for x in base._parse_steps(steps_spec, 1000)],
        "sign_source": str(args.sign_source),
        "pelvis_orientation_source": str(args.pelvis_orientation_source),
        "label_time": str(args.label_time),
        "sign_eps": float(args.sign_eps),
        "clip_filter": {
            "enabled": bool(apply_clip_filter),
            "clip_ids": sorted(int(x) for x in clip_ids_filter),
            "clip_name_contains": list(clip_name_filter),
            "skipped_batches": skipped_batches,
        },
        "subset_keys": subset_keys,
        "aggregate": agg,
        "per_batch": per_batch,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    if str(args.out_json).strip():
        p = Path(args.out_json).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Saved] {p}")

    if str(args.out_md).strip():
        p = Path(args.out_md).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_to_markdown(out), encoding="utf-8")
        print(f"[Saved] {p}")


if __name__ == "__main__":
    main()
