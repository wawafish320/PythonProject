#!/usr/bin/env python3
"""
Stage7 "distribution + gradient" closure diagnostic.

What it checks (for a target clip, default Walk_F):
1) Sampling distribution:
   - Frame/SIC coverage induced by sliding windows (seq_len from training config).
   - |omega_axis| histograms (raw-frame vs sampled-window weighted) for target joints.
2) Hidden-gradient distribution:
   - Per-window teacher-forced gradients dL_side/d(h_final) on the selected branch
     (default: out_direct), aggregated by omega bins and SIC.

This is intended to answer:
"Is high-|omega| under-sampled, and is hidden gradient side-asymmetric by phase?"
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train.dataset import MotionEventDataset
from train.validate.run_freerun_cycles import FreeRunCycleRunner, _merge_norm_spec


@dataclass
class JointSpec:
    name: str
    idx: int


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bins(spec: str) -> List[float]:
    vals: List[float] = []
    for tok in str(spec or "").split(","):
        s = tok.strip().lower()
        if not s:
            continue
        if s in ("inf", "+inf", "infinity"):
            vals.append(float("inf"))
        else:
            vals.append(float(s))
    if len(vals) < 2:
        raise ValueError(f"Need >=2 bin edges, got: {spec}")
    out = sorted(vals)
    if not math.isfinite(out[0]):
        raise ValueError("First bin edge must be finite.")
    if not math.isinf(out[-1]):
        out.append(float("inf"))
    return out


def _bin_labels(edges: Sequence[float]) -> List[str]:
    out: List[str] = []
    for i in range(len(edges) - 1):
        a, b = float(edges[i]), float(edges[i + 1])
        if math.isinf(b):
            out.append(f"[{a:.0f},inf)")
        else:
            out.append(f"[{a:.0f},{b:.0f})")
    return out


def _find_clip_index(ds: MotionEventDataset, target_clip: str) -> int:
    target = str(target_clip).strip().lower()
    if not target:
        raise ValueError("target clip is empty")
    for i, clip in enumerate(ds.clips):
        stem = Path(str(clip.npz_path)).stem.lower()
        if stem == target or target in stem:
            return int(i)
    stems = [Path(str(c.npz_path)).stem for c in ds.clips]
    raise ValueError(f"clip '{target_clip}' not found in dataset: {stems}")


def _coverage_counts(T: int, L: int) -> np.ndarray:
    cov = np.zeros((int(T),), dtype=np.float64)
    if T <= 0 or L <= 0 or T < L:
        return cov
    for s in range(0, int(T - L + 1)):
        cov[s : s + L] += 1.0
    return cov


def _hist_weighted(values: np.ndarray, bins: Sequence[float], weights: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    w = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    h, _ = np.histogram(arr, bins=np.asarray(bins, dtype=np.float64), weights=w)
    return h.astype(np.float64)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    n = _safe_float(num)
    d = _safe_float(den)
    if not (math.isfinite(n) and math.isfinite(d)):
        return float("nan")
    if abs(d) <= float(eps):
        return float("nan")
    return float(n / d)


def _rel_l2(a: Optional[torch.Tensor], b: Optional[torch.Tensor], eps: float = 1e-12) -> float:
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        return float("nan")
    if a.shape != b.shape:
        return float("nan")
    try:
        da = a.detach().float().reshape(-1)
        db = b.detach().float().reshape(-1)
        denom = float(torch.norm(da).item())
        if denom <= float(eps):
            return float("nan")
        return float(torch.norm(db - da).item() / denom)
    except Exception:
        return float("nan")


def _probe_direct_head_symmetry(
    *,
    model: Any,
    rot_slice: slice,
    joint_specs: Sequence[JointSpec],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": "unavailable",
        "layer": "",
        "joint_pair": [str(joint_specs[0].name), str(joint_specs[1].name)] if len(joint_specs) >= 2 else [],
        "row_dim": 0,
        "col_dim": 0,
        "weight_norm_l": float("nan"),
        "weight_norm_r": float("nan"),
        "weight_norm_ratio_r_over_l": float("nan"),
        "weight_rel_l2_raw": float("nan"),
        "weight_rel_l2_best_sign": float("nan"),
        "weight_row_cos_mean_raw": float("nan"),
        "weight_row_cos_mean_best_sign": float("nan"),
        "bias_norm_l": float("nan"),
        "bias_norm_r": float("nan"),
        "bias_norm_ratio_r_over_l": float("nan"),
        "bias_rel_l2_raw": float("nan"),
        "bias_rel_l2_best_sign": float("nan"),
    }
    if len(joint_specs) < 2:
        out["status"] = "joint_pair_invalid"
        return out
    if not hasattr(model, "direct_pose_head"):
        out["status"] = "direct_pose_head_missing"
        return out
    head = getattr(model, "direct_pose_head", None)
    if head is None:
        out["status"] = "direct_pose_head_none"
        return out

    last_linear = None
    last_name = ""
    if isinstance(head, torch.nn.Sequential):
        for idx in reversed(range(len(head))):
            mod = head[idx]
            if isinstance(mod, torch.nn.Linear):
                last_linear = mod
                last_name = f"direct_pose_head.{idx}"
                break
    if last_linear is None:
        out["status"] = "direct_pose_head_last_linear_not_found"
        return out
    if not torch.is_tensor(getattr(last_linear, "weight", None)):
        out["status"] = "direct_pose_head_weight_missing"
        return out

    w = last_linear.weight.detach()
    b = last_linear.bias.detach() if torch.is_tensor(getattr(last_linear, "bias", None)) else None
    sl_l = _slice_for_joint(rot_slice, int(joint_specs[0].idx))
    sl_r = _slice_for_joint(rot_slice, int(joint_specs[1].idx))
    st_l = int(sl_l.start or 0)
    ed_l = int(sl_l.stop or 0)
    st_r = int(sl_r.start or 0)
    ed_r = int(sl_r.stop or 0)
    out_dim = int(w.shape[0])
    if not (0 <= st_l < ed_l <= out_dim and 0 <= st_r < ed_r <= out_dim):
        out["status"] = "joint_slice_out_of_range"
        out["layer"] = str(last_name)
        return out

    wl = w[st_l:ed_l, :].detach().float().cpu()
    wr = w[st_r:ed_r, :].detach().float().cpu()
    out["layer"] = str(last_name)
    out["row_dim"] = int(wl.shape[0])
    out["col_dim"] = int(wl.shape[1]) if wl.dim() == 2 else 0
    out["weight_norm_l"] = float(torch.norm(wl).item())
    out["weight_norm_r"] = float(torch.norm(wr).item())
    out["weight_norm_ratio_r_over_l"] = _safe_ratio(out["weight_norm_r"], out["weight_norm_l"], eps=eps)
    out["weight_rel_l2_raw"] = _rel_l2(wl, wr, eps=eps)

    sign = torch.sign((wl * wr).sum(dim=1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    wr_aligned = wr * sign
    out["weight_rel_l2_best_sign"] = _rel_l2(wl, wr_aligned, eps=eps)
    try:
        cos_raw = torch.nn.functional.cosine_similarity(wl, wr, dim=1)
        cos_aligned = torch.nn.functional.cosine_similarity(wl, wr_aligned, dim=1)
        out["weight_row_cos_mean_raw"] = float(cos_raw.mean().item())
        out["weight_row_cos_mean_best_sign"] = float(cos_aligned.mean().item())
    except Exception:
        pass

    if torch.is_tensor(b) and b.ndim == 1:
        bl = b[st_l:ed_l].detach().float().cpu()
        br = b[st_r:ed_r].detach().float().cpu()
        out["bias_norm_l"] = float(torch.norm(bl).item())
        out["bias_norm_r"] = float(torch.norm(br).item())
        out["bias_norm_ratio_r_over_l"] = _safe_ratio(out["bias_norm_r"], out["bias_norm_l"], eps=eps)
        out["bias_rel_l2_raw"] = _rel_l2(bl, br, eps=eps)
        out["bias_rel_l2_best_sign"] = _rel_l2(bl, br * sign.view(-1), eps=eps)

    out["status"] = "ok"
    return out


def _infer_cond_feature_labels(cond_dim: int) -> Tuple[List[str], str]:
    d = int(cond_dim)
    if d <= 0:
        return [], "none"
    # Current converter uses: action one-hot + cmd_dir(x,y) + speed multiplier.
    if d >= 3:
        act_dim = int(max(0, d - 3))
        labels: List[str] = [f"action_oh_{i}" for i in range(act_dim)]
        labels += ["cmd_dir_x", "cmd_dir_y", "speed_multiplier"]
        if len(labels) < d:
            labels += [f"cond_{i}" for i in range(len(labels), d)]
        return labels[:d], "heuristic_action_oh_plus_dir_speed"
    return [f"cond_{i}" for i in range(d)], "generic_indexed"


def _summarize_cond_input_side_distribution(
    *,
    cond_rows: Sequence[np.ndarray],
    contact_rows: Sequence[np.ndarray],
    side_margin: float,
    contact_ch_l: int,
    contact_ch_r: int,
    topk: int,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": "unavailable",
        "num_rows": 0,
        "feature_dim": 0,
        "contact_dim": 0,
        "contact_ch_l": int(contact_ch_l),
        "contact_ch_r": int(contact_ch_r),
        "side_margin": float(side_margin),
        "left_rows": 0,
        "right_rows": 0,
        "neutral_rows": 0,
        "left_contact_mean": float("nan"),
        "right_contact_mean": float("nan"),
        "left_minus_right_contact_mean": float("nan"),
        "mean_abs_z": float("nan"),
        "max_abs_z": float("nan"),
        "top_abs_z_rows": [],
        "per_dim_rows": [],
        "feature_labels": [],
        "feature_label_mode": "none",
    }

    if not cond_rows:
        out["status"] = "cond_rows_empty"
        return out

    cond = np.asarray(cond_rows, dtype=np.float64)
    if cond.ndim != 2 or cond.shape[0] <= 0 or cond.shape[1] <= 0:
        out["status"] = "cond_shape_invalid"
        return out

    n, d = int(cond.shape[0]), int(cond.shape[1])
    out["num_rows"] = n
    out["feature_dim"] = d
    labels, label_mode = _infer_cond_feature_labels(d)
    out["feature_labels"] = labels
    out["feature_label_mode"] = str(label_mode)

    mean_all = np.mean(cond, axis=0)
    std_all = np.std(cond, axis=0)

    # Provide at least global cond moments, even when contacts are unavailable.
    global_rows: List[Dict[str, Any]] = []
    for i in range(d):
        global_rows.append(
            {
                "dim": int(i),
                "label": str(labels[i]) if i < len(labels) else f"cond_{i}",
                "mean_all": float(mean_all[i]),
                "std_all": float(std_all[i]),
                "mean_left": float("nan"),
                "mean_right": float("nan"),
                "std_left": float("nan"),
                "std_right": float("nan"),
                "mean_diff_right_minus_left": float("nan"),
                "abs_z": float("nan"),
            }
        )

    if not contact_rows:
        out["status"] = "contacts_missing"
        out["per_dim_rows"] = global_rows
        return out

    contacts = np.asarray(contact_rows, dtype=np.float64)
    if contacts.ndim != 2 or contacts.shape[0] != n or contacts.shape[1] <= 0:
        out["status"] = "contacts_shape_invalid"
        out["contact_dim"] = int(contacts.shape[1]) if contacts.ndim == 2 else 0
        out["per_dim_rows"] = global_rows
        return out

    cdim = int(contacts.shape[1])
    out["contact_dim"] = cdim
    max_idx = max(int(contact_ch_l), int(contact_ch_r))
    if max_idx >= cdim:
        out["status"] = "contact_channel_out_of_range"
        out["per_dim_rows"] = global_rows
        return out

    c_l = contacts[:, int(contact_ch_l)]
    c_r = contacts[:, int(contact_ch_r)]
    c_diff = c_l - c_r
    thr = max(0.0, float(side_margin))
    mask_l = c_diff > thr
    mask_r = c_diff < -thr
    mask_n = ~(mask_l | mask_r)

    n_l = int(np.sum(mask_l))
    n_r = int(np.sum(mask_r))
    n_n = int(np.sum(mask_n))
    out["left_rows"] = n_l
    out["right_rows"] = n_r
    out["neutral_rows"] = n_n
    out["left_contact_mean"] = float(np.mean(c_l)) if c_l.size > 0 else float("nan")
    out["right_contact_mean"] = float(np.mean(c_r)) if c_r.size > 0 else float("nan")
    out["left_minus_right_contact_mean"] = float(np.mean(c_diff)) if c_diff.size > 0 else float("nan")

    if n_l <= 0 or n_r <= 0:
        out["status"] = "insufficient_side_rows"
        out["per_dim_rows"] = global_rows
        return out

    mean_l = np.mean(cond[mask_l], axis=0)
    mean_r = np.mean(cond[mask_r], axis=0)
    std_l = np.std(cond[mask_l], axis=0)
    std_r = np.std(cond[mask_r], axis=0)
    z_abs = np.abs(mean_r - mean_l) / np.maximum(std_all, float(eps))

    per_dim_rows: List[Dict[str, Any]] = []
    for i in range(d):
        per_dim_rows.append(
            {
                "dim": int(i),
                "label": str(labels[i]) if i < len(labels) else f"cond_{i}",
                "mean_all": float(mean_all[i]),
                "std_all": float(std_all[i]),
                "mean_left": float(mean_l[i]),
                "mean_right": float(mean_r[i]),
                "std_left": float(std_l[i]),
                "std_right": float(std_r[i]),
                "mean_diff_right_minus_left": float(mean_r[i] - mean_l[i]),
                "abs_z": float(z_abs[i]) if math.isfinite(float(z_abs[i])) else float("nan"),
            }
        )

    valid_z = np.asarray(
        [float(r.get("abs_z", float("nan"))) for r in per_dim_rows if math.isfinite(_safe_float(r.get("abs_z")))],
        dtype=np.float64,
    )
    out["mean_abs_z"] = float(np.mean(valid_z)) if valid_z.size > 0 else float("nan")
    out["max_abs_z"] = float(np.max(valid_z)) if valid_z.size > 0 else float("nan")

    ranked = sorted(
        [r for r in per_dim_rows if math.isfinite(_safe_float(r.get("abs_z", float("nan"))))],
        key=lambda x: float(x.get("abs_z", 0.0)),
        reverse=True,
    )
    out["top_abs_z_rows"] = ranked[: max(1, int(topk))]
    out["per_dim_rows"] = per_dim_rows
    out["status"] = "ok"
    return out


def _mean_or_nan(vals: Sequence[float]) -> float:
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _p90_or_nan(vals: Sequence[float]) -> float:
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 90))


def _runner_args_from_cli(args: argparse.Namespace) -> argparse.Namespace:
    # Keep this minimal; FreeRunCycleRunner uses getattr(..., default) for most fields.
    return argparse.Namespace(
        model=str(Path(args.ckpt).expanduser().resolve()),
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


def _resolve_joints(ds: MotionEventDataset, joint_names: Sequence[str]) -> List[JointSpec]:
    names = [str(x) for x in getattr(ds, "bone_names", [])]
    lut = {n: i for i, n in enumerate(names)}
    out: List[JointSpec] = []
    for jn in joint_names:
        key = str(jn).strip()
        if key not in lut:
            raise ValueError(f"joint '{key}' not in bone_names")
        out.append(JointSpec(name=key, idx=int(lut[key])))
    return out


def _slice_for_joint(rot_slice: slice, joint_idx: int) -> slice:
    st = int(rot_slice.start or 0) + int(joint_idx) * 6
    ed = st + 6
    return slice(st, ed)


def _as_device_float(x: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if not torch.is_tensor(x):
        return None
    t = x.to(device)
    return t if t.dtype == torch.float32 else t.float()


def _get_clip_window_indices(ds: MotionEventDataset, clip_id: int) -> List[int]:
    out: List[int] = []
    for i, pair in enumerate(getattr(ds, "index", [])):
        try:
            cid, _ = pair
            if int(cid) == int(clip_id):
                out.append(int(i))
        except Exception:
            continue
    return out


def _to_bt(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not torch.is_tensor(x):
        return None
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x


def _to_btf(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Canonicalize tensor to (B, T, F) for time-wise gradient norm stats."""
    if not torch.is_tensor(x):
        return None
    if x.dim() == 0:
        return x.view(1, 1, 1)
    if x.dim() == 1:
        return x.view(1, 1, -1)
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x.reshape(x.shape[0], x.shape[1], -1)


def _grad_time_l2_series(g: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    bt = _to_btf(g)
    if not torch.is_tensor(bt):
        return None
    per_bt = bt.norm(dim=-1)
    if per_bt.dim() == 2:
        per_t = per_bt.mean(dim=0)
    elif per_bt.dim() == 1:
        per_t = per_bt
    else:
        return None
    return per_t.detach().cpu().numpy().astype(np.float64)


def _grad_time_l2_series_with_time_hint(g: Optional[torch.Tensor], *, time_hint: int) -> Optional[np.ndarray]:
    if not torch.is_tensor(g):
        return None
    t = g
    if t.dim() == 2 and int(time_hint) > 1:
        n0 = int(t.shape[0])
        if n0 % int(time_hint) == 0:
            bsz = int(n0 // int(time_hint))
            try:
                t = t.reshape(bsz, int(time_hint), int(t.shape[-1]))
            except Exception:
                pass
    return _grad_time_l2_series(t)


def _parse_dynamics_probe_list(spec: str) -> List[str]:
    alias = {
        "shared": "shared_encoder_pre_act",
        "shared_pre": "shared_encoder_pre_act",
        "shared_encoder": "shared_encoder_pre_act",
        "temporal": "temporal_pre_pasa",
        "temporal_pre": "temporal_pre_pasa",
        "pasa_in": "temporal_pre_pasa",
        "direct_pre": "direct_head_pre_out",
        "direct_head_pre": "direct_head_pre_out",
        "direct_pre_out": "direct_head_pre_out",
    }
    allowed = {"shared_encoder_pre_act", "temporal_pre_pasa", "direct_head_pre_out"}
    out: List[str] = []
    seen = set()
    for tok in str(spec or "").split(","):
        raw = str(tok).strip().lower()
        if not raw:
            continue
        name = alias.get(raw, raw)
        if name not in allowed or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _find_last_linear(module: Any, *, prefix: str) -> Tuple[Optional[torch.nn.Linear], str]:
    if isinstance(module, torch.nn.Linear):
        return module, str(prefix)
    if isinstance(module, torch.nn.Sequential):
        for idx in reversed(range(len(module))):
            mod = module[idx]
            if isinstance(mod, torch.nn.Linear):
                return mod, f"{prefix}.{idx}"
    return None, ""


def _resolve_dynamics_probe_specs(model: Any, requested: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    specs: List[Dict[str, Any]] = []
    meta: Dict[str, Dict[str, Any]] = {}

    req = list(requested)
    if "shared_encoder_pre_act" in req:
        shared = getattr(model, "shared_encoder", None)
        mod = None
        path = "shared_encoder.0"
        status = "missing"
        if isinstance(shared, torch.nn.Sequential) and len(shared) > 0 and isinstance(shared[0], torch.nn.Module):
            mod = shared[0]
            status = "ok"
        meta["shared_encoder_pre_act"] = {"status": status, "module": path, "capture": "output"}
        if mod is not None:
            specs.append({"name": "shared_encoder_pre_act", "module": mod, "capture": "output", "module_path": path})

    if "temporal_pre_pasa" in req:
        mod = getattr(model, "_pasa_lnq", None)
        path = "_pasa_lnq"
        status = "ok" if isinstance(mod, torch.nn.Module) else "missing"
        meta["temporal_pre_pasa"] = {"status": status, "module": path, "capture": "input"}
        if isinstance(mod, torch.nn.Module):
            specs.append({"name": "temporal_pre_pasa", "module": mod, "capture": "input", "module_path": path})

    if "direct_head_pre_out" in req:
        head = getattr(model, "direct_pose_head", None)
        mod, path = _find_last_linear(head, prefix="direct_pose_head")
        status = "ok" if mod is not None else "missing"
        meta["direct_head_pre_out"] = {"status": status, "module": path or "direct_pose_head", "capture": "input"}
        if mod is not None:
            specs.append({"name": "direct_head_pre_out", "module": mod, "capture": "input", "module_path": path})

    return specs, meta


def _extract_first_tensor(x: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            if torch.is_tensor(v):
                return v
    return None


def _canonicalize_dynamic_probe_tensor(x: Any, *, time_hint: int) -> Optional[torch.Tensor]:
    t = _extract_first_tensor(x)
    if not torch.is_tensor(t):
        return None
    return t


class _ActivationProbeRecorder:
    def __init__(self, specs: Sequence[Dict[str, Any]]) -> None:
        self._handles: List[Any] = []
        self._cache: Dict[str, torch.Tensor] = {}
        self._specs: List[Dict[str, Any]] = [dict(s) for s in specs]
        self._install()

    def _install(self) -> None:
        for spec in self._specs:
            name = str(spec.get("name", ""))
            mod = spec.get("module", None)
            capture = str(spec.get("capture", "output"))
            if not name or not isinstance(mod, torch.nn.Module):
                continue
            if capture == "input":
                def _pre_hook(_m: torch.nn.Module, inp: Tuple[Any, ...], _name: str = name) -> None:
                    t = _extract_first_tensor(inp)
                    if torch.is_tensor(t):
                        self._cache[_name] = t
                h = mod.register_forward_pre_hook(_pre_hook)
            else:
                def _hook(_m: torch.nn.Module, _inp: Tuple[Any, ...], out: Any, _name: str = name) -> None:
                    t = _extract_first_tensor(out)
                    if torch.is_tensor(t):
                        self._cache[_name] = t
                h = mod.register_forward_hook(_hook)
            self._handles.append(h)

    def clear(self) -> None:
        self._cache.clear()

    def get(self, name: str) -> Optional[torch.Tensor]:
        t = self._cache.get(str(name), None)
        return t if torch.is_tensor(t) else None

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


def _build_probe_tensor_map(
    *,
    out: Mapping[str, Any],
    loss_branch: str,
    h_final: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    probes: Dict[str, torch.Tensor] = {}
    if torch.is_tensor(h_final):
        probes["h_final"] = h_final

    branch_key = f"branch:{str(loss_branch)}"
    branch = _to_bt(out.get(str(loss_branch), None))
    if torch.is_tensor(branch):
        probes[branch_key] = branch

    out_main = _to_bt(out.get("out", None))
    if torch.is_tensor(out_main):
        probes["out"] = out_main

    out_direct = _to_bt(out.get("out_direct", None))
    if torch.is_tensor(out_direct):
        probes["out_direct"] = out_direct

    dd = out.get("direct_delta", None)
    dd_bt = _to_btf(dd) if torch.is_tensor(dd) else None
    if torch.is_tensor(dd_bt):
        probes["direct_delta"] = dd_bt
    return probes


def _probe_priority(component: str, loss_branch: str) -> List[str]:
    branch_key = f"branch:{str(loss_branch)}"
    comp = str(component)
    if comp == "__legacy__":
        cand = ["h_final", branch_key, "out", "out_direct"]
    elif comp in ("rot_geo", "rot_vel"):
        cand = ["h_final", "out", branch_key]
    elif comp == "direct_pose":
        cand = ["h_final", "out_direct", branch_key, "out"]
    elif comp == "direct_delta":
        cand = ["h_final", "direct_delta", "out_direct", branch_key, "out"]
    else:
        cand = ["h_final", branch_key, "out", "out_direct", "direct_delta"]
    out: List[str] = []
    seen = set()
    for k in cand:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _parse_component_list(spec: str) -> List[str]:
    allowed = ("rot_geo", "rot_vel", "direct_pose", "direct_delta")
    toks = [str(t).strip().lower() for t in str(spec or "").split(",") if str(t).strip()]
    out: List[str] = []
    seen = set()
    for t in toks:
        if t not in allowed or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out if out else list(allowed)


def _sync_loss_normalization_from_trainer(loss_fn: Any, trainer: Any) -> Dict[str, bool]:
    status = {"mu_y": False, "std_y": False}
    if loss_fn is None or trainer is None:
        return status
    normalizer = getattr(trainer, "normalizer", None)
    for key in ("mu_y", "std_y"):
        cur = getattr(loss_fn, key, None)
        if cur is not None:
            status[key] = True
            continue
        src = getattr(trainer, key, None)
        if src is None and normalizer is not None:
            src = getattr(normalizer, key, None)
        if src is None:
            status[key] = False
            continue
        try:
            setattr(loss_fn, key, src)
            status[key] = True
        except Exception:
            status[key] = False
    return status


def _apply_loss_config_overrides(loss_fn: Any, cfg: Mapping[str, Any]) -> Dict[str, float]:
    if loss_fn is None:
        return {}
    float_keys = (
        "w_attn_reg",
        "w_rot_ortho",
        "w_rot_local",
        "w_rot_vel",
        "rot_vel_log_scale",
        "rot_vel_omega_min_deg_s",
        "w_root_vel",
        "w_root_speed",
        "w_contact_plan",
        "w_contact_meas",
        "w_contact_td_hazard_bce",
        "w_contact_td_hazard_mass",
        "w_contact_td_hazard_unimodal",
        "w_direct_pose",
        "w_direct_delta",
        "direct_delta_omega_thr_deg_s",
        "direct_delta_omega_beta",
        "direct_delta_omega_wmax",
        "w_direct_delta_sym",
        "w_omega_l2",
        "event_clock_lambda_entropy_weight",
        "event_clock_lambda_prior_weight",
        "event_clock_delta_z_l2_weight",
    )
    str_keys = (
        "rot_vel_loss",
    )
    applied: Dict[str, float] = {}
    for key in float_keys:
        if key not in cfg or not hasattr(loss_fn, key):
            continue
        try:
            val = float(cfg.get(key))
        except Exception:
            continue
        try:
            setattr(loss_fn, key, val)
            applied[key] = val
        except Exception:
            continue
    for key in str_keys:
        if key not in cfg or not hasattr(loss_fn, key):
            continue
        try:
            setattr(loss_fn, key, str(cfg.get(key)))
        except Exception:
            continue
    return applied


def _resolve_rot_geo_weight(args: argparse.Namespace, cfg: Mapping[str, Any]) -> Tuple[float, str]:
    raw = getattr(args, "rot_geo_weight", None)
    if raw is not None:
        try:
            return float(raw), "cli"
        except Exception:
            pass
    if "w_rot_geo" in cfg:
        try:
            return float(cfg.get("w_rot_geo")), "config_json:w_rot_geo"
        except Exception:
            pass
    return 0.0, "default_zero"


def _per_joint_geo_losses(
    *,
    loss_fn: Any,
    pred_bt: Optional[torch.Tensor],
    gt_bt: Optional[torch.Tensor],
    joint_specs: Sequence[JointSpec],
    hinge_delta: Optional[torch.Tensor] = None,
    hinge_joint_idx: Optional[Sequence[int]] = None,
    hinge_axis: str = "Z",
    hinge_max_rad: Optional[float] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {js.name: None for js in joint_specs}
    if not (torch.is_tensor(pred_bt) and torch.is_tensor(gt_bt)):
        return out
    if pred_bt.dim() != 3 or gt_bt.dim() != 3:
        return out
    t_use = min(int(pred_bt.shape[1]), int(gt_bt.shape[1]))
    if t_use <= 0:
        return out
    pred_use = pred_bt[:, :t_use]
    gt_use = gt_bt[:, :t_use]
    hinge_use = None
    if torch.is_tensor(hinge_delta):
        try:
            if hinge_delta.dim() == 2 and pred_use.dim() == 3:
                hinge_delta = hinge_delta.unsqueeze(1)
            if hinge_delta.dim() >= 3:
                hinge_use = hinge_delta[:, :t_use]
        except Exception:
            hinge_use = None
    try:
        payload = loss_fn.compute_rot6d_geo_loss(
            pred_use,
            gt_use,
            return_per_joint=True,
            hinge_delta=hinge_use,
            hinge_joint_idx=hinge_joint_idx,
            hinge_axis=hinge_axis,
            hinge_max_rad=hinge_max_rad,
        )
    except Exception:
        return out
    if not (isinstance(payload, tuple) and len(payload) >= 2 and torch.is_tensor(payload[1])):
        return out
    theta = payload[1]
    weights = payload[2] if len(payload) >= 3 and torch.is_tensor(payload[2]) else None
    j_max = int(theta.shape[-1]) if theta.dim() > 0 else 0
    for js in joint_specs:
        j = int(js.idx)
        if j < 0 or j >= j_max:
            continue
        term = theta[..., j]
        if torch.is_tensor(weights) and weights.numel() > j:
            term = term * weights[j]
        out[js.name] = term.mean()
    return out


def _per_joint_rot_vel_losses(
    *,
    loss_fn: Any,
    pred_bt: Optional[torch.Tensor],
    gt_bt: Optional[torch.Tensor],
    joint_specs: Sequence[JointSpec],
) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {js.name: None for js in joint_specs}
    if not (torch.is_tensor(pred_bt) and torch.is_tensor(gt_bt)):
        return out
    if pred_bt.dim() != 3 or gt_bt.dim() != 3:
        return out
    t_use = min(int(pred_bt.shape[1]), int(gt_bt.shape[1]))
    if t_use <= 0:
        return out
    pred_use = pred_bt[:, :t_use]
    gt_use = gt_bt[:, :t_use]
    try:
        rp = loss_fn._rot6d_matrices(pred_use)
        rg = loss_fn._rot6d_matrices(gt_use)
    except Exception:
        rp, rg = None, None
    if not (torch.is_tensor(rp) and torch.is_tensor(rg)):
        return out
    if rp.shape != rg.shape or rp.dim() < 5:
        return out
    j_use = min(int(rp.shape[-3]), int(rg.shape[-3]))
    if j_use <= 0:
        return out
    for js in joint_specs:
        j = int(js.idx)
        if j < 0 or j >= j_use:
            continue
        try:
            out[js.name] = loss_fn._rot_vel_loss_from_mats(
                rp[..., j : j + 1, :, :],
                rg[..., j : j + 1, :, :],
                return_stats=False,
            )
        except Exception:
            out[js.name] = None
    return out


def _build_component_side_losses(
    *,
    out: Dict[str, Any],
    gt: torch.Tensor,
    joint_specs: Sequence[JointSpec],
    rot_slice: slice,
    loss_fn: Any,
    components: Sequence[str],
    component_weights: Mapping[str, float],
) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
    comp_set = set(components)
    out_main = _to_bt(out.get("out", None))
    out_direct = _to_bt(out.get("out_direct", None))
    gt_bt = _to_bt(gt)
    result: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
    if not torch.is_tensor(gt_bt):
        return result

    for comp in components:
        result[comp] = {js.name: None for js in joint_specs}

    # rot_geo / rot_vel on main branch (out), using full-rot segment then per-joint extract.
    if torch.is_tensor(out_main) and out_main.dim() == 3 and gt_bt.dim() == 3:
        if "rot_geo" in comp_set:
            w_geo = float(component_weights.get("rot_geo", 0.0) or 0.0)
            per_geo = _per_joint_geo_losses(
                loss_fn=loss_fn,
                pred_bt=out_main,
                gt_bt=gt_bt,
                joint_specs=joint_specs,
            )
            for js in joint_specs:
                l_geo = per_geo.get(js.name, None)
                if torch.is_tensor(l_geo):
                    result["rot_geo"][js.name] = l_geo * w_geo
                else:
                    result["rot_geo"][js.name] = None
        if "rot_vel" in comp_set:
            w_vel = float(component_weights.get("rot_vel", 0.0) or 0.0)
            per_vel = _per_joint_rot_vel_losses(
                loss_fn=loss_fn,
                pred_bt=out_main,
                gt_bt=gt_bt,
                joint_specs=joint_specs,
            )
            for js in joint_specs:
                l_vel = per_vel.get(js.name, None)
                if torch.is_tensor(l_vel):
                    result["rot_vel"][js.name] = l_vel * w_vel
                else:
                    result["rot_vel"][js.name] = None

    # direct_pose on out_direct branch.
    if "direct_pose" in comp_set and torch.is_tensor(out_direct) and out_direct.dim() == 3 and gt_bt.dim() == 3:
        w_dp = float(component_weights.get("direct_pose", 0.0) or 0.0)
        hinge_delta = out.get("direct_hinge_delta", None)
        hinge_idx = getattr(loss_fn, "direct_pose_hinge_joint_idx", None)
        hinge_axis = str(getattr(loss_fn, "direct_pose_hinge_axis", "Z") or "Z")
        hinge_max = getattr(loss_fn, "direct_pose_hinge_max_rad", None)
        per_direct = _per_joint_geo_losses(
            loss_fn=loss_fn,
            pred_bt=out_direct,
            gt_bt=gt_bt,
            joint_specs=joint_specs,
            hinge_delta=hinge_delta if torch.is_tensor(hinge_delta) else None,
            hinge_joint_idx=hinge_idx,
            hinge_axis=hinge_axis,
            hinge_max_rad=hinge_max,
        )
        for js in joint_specs:
            l_direct = per_direct.get(js.name, None)
            if torch.is_tensor(l_direct):
                result["direct_pose"][js.name] = l_direct * w_dp
            else:
                result["direct_pose"][js.name] = None

    # direct_delta side losses (weighted exactly like training term).
    if "direct_delta" in comp_set:
        try:
            dd = out.get("direct_delta", None)
            dd = _to_bt(dd)
            if torch.is_tensor(dd):
                if dd.dim() == 3 and dd.shape[-1] == 3:
                    dd = dd.unsqueeze(2)
                elif dd.dim() == 3 and (dd.shape[-1] % 3) == 0:
                    dd = dd.view(dd.shape[0], dd.shape[1], -1, 3)
                if dd.dim() == 4 and dd.shape[-1] == 3 and gt_bt.dim() == 3:
                    rg = loss_fn._rot6d_matrices(gt_bt)
                    delta_gt = loss_fn._rotation_delta_axis_angle(rg)
                    if torch.is_tensor(delta_gt) and delta_gt.dim() == 4:
                        t_use = min(int(dd.shape[1]), int(delta_gt.shape[1]))
                        j_use = min(int(dd.shape[2]), int(delta_gt.shape[2]))
                        if t_use > 0 and j_use > 0:
                            pred_use = dd[:, :t_use, :j_use, :]
                            gt_use = delta_gt[:, :t_use, :j_use, :]
                            fps_use = float(getattr(loss_fn, "fps", 60.0) or 60.0)
                            thr = float(getattr(loss_fn, "direct_delta_omega_thr_deg_s", 0.0) or 0.0)
                            beta = float(getattr(loss_fn, "direct_delta_omega_beta", 0.0) or 0.0)
                            wmax = float(getattr(loss_fn, "direct_delta_omega_wmax", 4.0) or 4.0)
                            if (not math.isfinite(wmax)) or wmax <= 0.0:
                                wmax = 4.0
                            w_dd = float(component_weights.get("direct_delta", 0.0) or 0.0)
                            for js in joint_specs:
                                j = int(js.idx)
                                if j < 0 or j >= j_use:
                                    result["direct_delta"][js.name] = None
                                    continue
                                p = pred_use[:, :, j, :]
                                g = gt_use[:, :, j, :]
                                l_per = F.smooth_l1_loss(p, g, reduction="none").mean(dim=-1)  # (B,T)
                                omega = g.norm(dim=-1) * (180.0 / math.pi) * fps_use
                                if thr > 0.0 and beta > 0.0:
                                    rel = (omega / max(thr, 1e-6)).clamp(min=0.0)
                                    w = 1.0 + beta * rel.clamp(max=wmax)
                                    l = (l_per * w).mean()
                                elif thr > 0.0:
                                    mask = omega >= thr
                                    if bool(mask.any()):
                                        l = l_per[mask].mean()
                                    else:
                                        l = l_per.mean()
                                else:
                                    l = l_per.mean()
                                result["direct_delta"][js.name] = l * w_dd
        except Exception:
            pass
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage7 sampling+gradient closure diagnostic for a target clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config-json", type=str, default="config/exp_phase_DirectBranch_v1_d1_noreset.json")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--target-clip", type=str, default="Walk_F")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--encoder-bundle", type=str, default="models/motion_encoder_equiv_stageA.pt")
    ap.add_argument("--seq-len", type=int, default=None, help="Override seq_len from config.")
    ap.add_argument("--depth", type=int, default=3, help="Must match freerun eval setting for this ckpt.")
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--context-len", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--cycle-len", type=int, default=0, help="<=0 means infer from clip metadata.")
    ap.add_argument("--joints", type=str, default="calf_l,calf_r")
    ap.add_argument("--omega-axis", type=str, default="z", choices=("x", "y", "z"))
    ap.add_argument("--omega-bins", type=str, default="0,50,100,150,200,250,300,inf")
    ap.add_argument("--loss-branch", type=str, default="out_direct", choices=("out_direct", "out"))
    ap.add_argument(
        "--component-losses",
        type=str,
        default="rot_geo,rot_vel,direct_pose,direct_delta",
        help="Comma-separated loss components for per-component dL/dh_final diagnostics.",
    )
    ap.add_argument(
        "--disable-component-grad",
        action="store_true",
        default=False,
        help="Disable per-component gradient split (faster, keeps legacy output only).",
    )
    ap.add_argument(
        "--dynamics-probes",
        type=str,
        default="shared_encoder_pre_act,temporal_pre_pasa,direct_head_pre_out",
        help="Comma-separated layer probes appended to dual-probe split (input/output-path dynamics).",
    )
    ap.add_argument(
        "--disable-dynamics-probe",
        action="store_true",
        default=False,
        help="Disable layer-wise dynamics probes (keep only out_direct/cond_in dual probes).",
    )
    ap.add_argument(
        "--skip-loss-config-override",
        action="store_true",
        default=False,
        help="Do not apply loss weight overrides from --config-json onto runtime loss_fn.",
    )
    ap.add_argument(
        "--rot-geo-weight",
        type=float,
        default=None,
        help="Diagnostic weight for rot_geo component. Default: cfg[w_rot_geo] if present else 0.0.",
    )
    ap.add_argument("--max-windows", type=int, default=0, help="0 means use all windows of target clip.")
    ap.add_argument(
        "--cond-side-margin",
        type=float,
        default=0.10,
        help="Contact-side margin used for cond_in side split: left if (c_l-c_r)>thr, right if < -thr.",
    )
    ap.add_argument(
        "--cond-side-topk",
        type=int,
        default=8,
        help="Top-K cond_in features to report by |z| side gap.",
    )
    ap.add_argument("--out-dir", type=str, default="", help="Default: debug_output/_dist_grad_closure_YYYYMMDD")
    args = ap.parse_args()

    cfg = _load_json(Path(args.config_json).expanduser())
    bundle_path = Path(args.bundle).expanduser()
    pretrain_path = Path(args.pretrain_template).expanduser()
    norm_spec = _merge_norm_spec(bundle_path, pretrain_path if pretrain_path.is_file() else None)

    seq_len = int(args.seq_len) if args.seq_len is not None else int(cfg.get("seq_len", 60))
    data_dir = str(cfg.get("data") or "raw_data/processed_data")
    ds = MotionEventDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        paths=None,
        pose_hist_len=int(norm_spec.get("pose_hist_len", 0) or 0),
        norm_spec=norm_spec,
        index_mode=str(cfg.get("dataset_index_mode") or cfg.get("index_mode") or "sliding"),
    )
    ds.is_train = True
    ds.normalize_c = bool(cfg.get("normalize_c", False))

    clip_id = _find_clip_index(ds, args.target_clip)
    clip = ds.clips[clip_id]
    T = int(clip.X.shape[0])
    L = int(seq_len)
    index_mode = str(getattr(ds, "index_mode", "sliding") or "sliding").strip().lower()
    target_indices_native = _get_clip_window_indices(ds, clip_id)
    window_count_native = len(target_indices_native)
    if window_count_native <= 0:
        raise SystemExit(f"[FATAL] clip '{args.target_clip}' has no windows under seq_len={L}.")

    # Dynamic modes keep one sentinel index and randomize start in __getitem__.
    if index_mode in ("clip_random", "sic_balanced"):
        default_windows = max(1, int(T - L + 1))
        window_count_effective = int(args.max_windows) if int(args.max_windows) > 0 else int(default_windows)
        target_indices = [int(target_indices_native[0])] * int(window_count_effective)
    else:
        target_indices = list(int(i) for i in target_indices_native)
        if int(args.max_windows) > 0:
            target_indices = target_indices[: int(args.max_windows)]
        window_count_effective = len(target_indices)
    if not target_indices:
        raise SystemExit("[FATAL] no target windows selected.")

    if index_mode == "sliding":
        starts = [int(ds.index[i][1]) for i in target_indices]
        coverage = np.zeros((int(T),), dtype=np.float64)
        for s in starts:
            s0 = int(max(0, min(int(s), max(0, int(T - L)))))
            coverage[s0 : s0 + int(L)] += 1.0
    elif index_mode == "start0":
        coverage = np.zeros((int(T),), dtype=np.float64)
        if T > 0 and L > 0:
            e = min(int(T), int(L))
            coverage[:e] += float(window_count_effective)
    elif index_mode == "clip_random":
        base = _coverage_counts(T=T, L=L)
        denom = max(1, int(T - L + 1))
        coverage = base * (float(window_count_effective) / float(denom))
    elif index_mode == "sic_balanced":
        if T > 0 and L > 0:
            coverage = np.full((int(T),), float(window_count_effective) * float(L) / float(T), dtype=np.float64)
        else:
            coverage = np.zeros((int(T),), dtype=np.float64)
    else:
        coverage = _coverage_counts(T=T, L=L)
    if int(args.cycle_len) > 0:
        cycle_len = int(args.cycle_len)
    else:
        if getattr(clip, "bone_rot6d", None) is not None and int(clip.bone_rot6d.shape[0]) > 0:
            cycle_len = int(clip.bone_rot6d.shape[0])
        else:
            cycle_len = int(T)

    joints = [s.strip() for s in str(args.joints).split(",") if s.strip()]
    if len(joints) != 2:
        raise SystemExit(f"[FATAL] expected exactly 2 joints, got: {joints}")
    joint_specs = _resolve_joints(ds, joints)

    axis_map = {"x": 0, "y": 1, "z": 2}
    axis = int(axis_map[str(args.omega_axis).lower()])
    bins = _parse_bins(args.omega_bins)
    bin_labels = _bin_labels(bins)
    component_list = _parse_component_list(args.component_losses)
    enable_component_grad = not bool(args.disable_component_grad)
    requested_dynamics_probes = (
        []
        if bool(args.disable_dynamics_probe)
        else _parse_dynamics_probe_list(args.dynamics_probes)
    )

    if getattr(clip, "angvel_raw", None) is None:
        raise SystemExit("[FATAL] clip.angvel_raw is missing; cannot compute omega diagnostics.")
    ang_raw_clip = np.asarray(clip.angvel_raw, dtype=np.float32).reshape(T, len(ds.bone_names), 3)

    # ---------- Distribution: raw-frame vs sampled-window ----------
    dist_rows: List[Dict[str, Any]] = []
    sic_raw = np.arange(T, dtype=np.int64) % int(cycle_len)
    sic_raw_counts = np.bincount(sic_raw, minlength=int(cycle_len)).astype(np.float64)
    sic_samp_counts = np.bincount(sic_raw, weights=coverage, minlength=int(cycle_len)).astype(np.float64)
    for js in joint_specs:
        omega = np.abs(np.asarray(ang_raw_clip[:, js.idx, axis], dtype=np.float64)) * (180.0 / math.pi)
        h_raw = _hist_weighted(omega, bins=bins, weights=None)
        h_smp = _hist_weighted(omega, bins=bins, weights=coverage)
        raw_total = float(np.sum(h_raw))
        smp_total = float(np.sum(h_smp))
        for i, lab in enumerate(bin_labels):
            dist_rows.append(
                {
                    "joint": js.name,
                    "bin": lab,
                    "raw_count": float(h_raw[i]),
                    "sampled_count": float(h_smp[i]),
                    "raw_frac": float(h_raw[i] / raw_total) if raw_total > 0 else float("nan"),
                    "sampled_frac": float(h_smp[i] / smp_total) if smp_total > 0 else float("nan"),
                }
            )

    # ---------- Gradient diagnostics ----------
    runner = FreeRunCycleRunner(_runner_args_from_cli(args))
    runner._ensure_model_ready(ds)
    if runner.model is None or runner.trainer is None:
        raise SystemExit("[FATAL] failed to initialize model/trainer from checkpoint.")
    model = runner.model.to(runner.device)
    model.eval()
    trainer = runner.trainer
    loss_fn = getattr(trainer, "loss_fn", None)
    norm_sync_status = _sync_loss_normalization_from_trainer(loss_fn, trainer)
    loss_cfg_overrides: Dict[str, float] = {}
    if not bool(args.skip_loss_config_override):
        loss_cfg_overrides = _apply_loss_config_overrides(loss_fn, cfg)

    rot_geo_w, rot_geo_w_source = _resolve_rot_geo_weight(args, cfg)
    component_weights: Dict[str, float] = {}
    if enable_component_grad:
        component_weights = {
            "rot_geo": float(rot_geo_w),
            "rot_vel": float(getattr(loss_fn, "w_rot_vel", 0.0) or 0.0),
            "direct_pose": float(getattr(loss_fn, "w_direct_pose", 0.0) or 0.0),
            "direct_delta": float(getattr(loss_fn, "w_direct_delta", 0.0) or 0.0),
        }

    rot_slice = getattr(trainer, "rot6d_y_slice", None)
    if not isinstance(rot_slice, slice):
        raise SystemExit("[FATAL] trainer.rot6d_y_slice is unavailable.")

    dynamics_specs, dynamics_probe_meta = _resolve_dynamics_probe_specs(model, requested_dynamics_probes)
    dynamics_probe_names = [str(s.get("name")) for s in dynamics_specs if str(s.get("name", "")).strip()]
    dynamics_probe_shapes: Dict[str, List[int]] = {}
    probe_recorder = _ActivationProbeRecorder(dynamics_specs)

    grad_rows: List[Dict[str, Any]] = []
    grad_unused_counts = {joint_specs[0].name: 0, joint_specs[1].name: 0}
    grad_fallback_counts = {joint_specs[0].name: 0, joint_specs[1].name: 0}
    grad_probe_source_counts: Dict[str, Dict[str, int]] = {
        joint_specs[0].name: {},
        joint_specs[1].name: {},
    }
    component_grad_rows: List[Dict[str, Any]] = []
    component_side_loss_rows: List[Dict[str, Any]] = []
    dual_probe_names = tuple(["out_direct", "cond_in", *dynamics_probe_names])
    dual_component_grad_rows: List[Dict[str, Any]] = []
    component_grad_unused: Dict[str, Dict[str, int]] = {}
    component_grad_missing: Dict[str, Dict[str, int]] = {}
    component_grad_fallback: Dict[str, Dict[str, int]] = {}
    component_grad_probe_sources: Dict[str, Dict[str, Dict[str, int]]] = {}
    dual_component_grad_unused: Dict[str, Dict[str, Dict[str, int]]] = {}
    dual_component_grad_missing: Dict[str, Dict[str, Dict[str, int]]] = {}
    cond_all_rows: List[np.ndarray] = []
    cond_side_cond_rows: List[np.ndarray] = []
    cond_side_contact_rows: List[np.ndarray] = []
    if enable_component_grad:
        for comp in component_list:
            component_grad_unused[comp] = {joint_specs[0].name: 0, joint_specs[1].name: 0}
            component_grad_missing[comp] = {joint_specs[0].name: 0, joint_specs[1].name: 0}
            component_grad_fallback[comp] = {joint_specs[0].name: 0, joint_specs[1].name: 0}
            component_grad_probe_sources[comp] = {
                joint_specs[0].name: {},
                joint_specs[1].name: {},
            }
            dual_component_grad_unused[comp] = {
                p: {joint_specs[0].name: 0, joint_specs[1].name: 0} for p in dual_probe_names
            }
            dual_component_grad_missing[comp] = {
                p: {joint_specs[0].name: 0, joint_specs[1].name: 0} for p in dual_probe_names
            }

    for ds_idx in target_indices:
        sample = ds[ds_idx]
        s0 = int(sample["start"].item()) if torch.is_tensor(sample.get("start")) else 0

        state = _as_device_float(sample.get("motion"), runner.device)
        gt = _as_device_float(sample.get("gt_motion"), runner.device)
        cond = _as_device_float(sample.get("cond_in"), runner.device)
        contacts = _as_device_float(sample.get("contacts"), runner.device)
        angvel = _as_device_float(sample.get("angvel"), runner.device)
        pose_hist = _as_device_float(sample.get("pose_hist"), runner.device)
        angvel_raw = _as_device_float(sample.get("angvel_raw"), runner.device)

        if state is None or gt is None or angvel_raw is None:
            continue

        # Input-layer probe: cond_in distribution (global + side-split by contacts_l/r).
        cond_np = None
        if torch.is_tensor(cond) and cond.dim() == 2 and cond.shape[0] > 0 and cond.shape[1] > 0:
            cond_np = cond.detach().cpu().numpy().astype(np.float64, copy=False)
            for t_row in range(int(cond_np.shape[0])):
                cond_all_rows.append(np.asarray(cond_np[t_row], dtype=np.float64))

        contacts_np = None
        if torch.is_tensor(contacts) and contacts.dim() == 2 and contacts.shape[0] > 0 and contacts.shape[1] > 0:
            contacts_np = contacts.detach().cpu().numpy().astype(np.float64, copy=False)
        if cond_np is not None and contacts_np is not None:
            t_use_probe = int(min(cond_np.shape[0], contacts_np.shape[0]))
            for t_row in range(t_use_probe):
                cond_side_cond_rows.append(np.asarray(cond_np[t_row], dtype=np.float64))
                cond_side_contact_rows.append(np.asarray(contacts_np[t_row], dtype=np.float64))

        state = state.unsqueeze(0)
        gt = gt.unsqueeze(0)
        cond = cond.unsqueeze(0) if cond is not None else None
        contacts = contacts.unsqueeze(0) if contacts is not None else None
        angvel = angvel.unsqueeze(0) if angvel is not None else None
        pose_hist = pose_hist.unsqueeze(0) if pose_hist is not None else None
        angvel_raw = angvel_raw.unsqueeze(0)
        cond_in = cond.clone().detach().requires_grad_(True) if torch.is_tensor(cond) else None

        probe_recorder.clear()
        with torch.enable_grad():
            out = model(
                state,
                cond=cond_in,
                contacts=contacts,
                angvel=angvel,
                pose_history=pose_hist,
                plan_z=None,
                time_index=None,
            )
            if not isinstance(out, dict) or args.loss_branch not in out:
                raise SystemExit(f"[FATAL] model output missing required keys for branch={args.loss_branch}.")

            pred = out[args.loss_branch]
            h_final = out.get("h_final", None)
            if not torch.is_tensor(pred):
                continue
            pred_bt = _to_bt(pred)
            t_hint = int(pred_bt.shape[1]) if torch.is_tensor(pred_bt) and pred_bt.dim() >= 2 else 1
            probe_tensors = _build_probe_tensor_map(out=out, loss_branch=str(args.loss_branch), h_final=h_final)
            if torch.is_tensor(cond_in):
                probe_tensors["cond_in"] = cond_in
            for pname in dynamics_probe_names:
                p_raw = probe_recorder.get(pname)
                p_t = _canonicalize_dynamic_probe_tensor(p_raw, time_hint=t_hint)
                if torch.is_tensor(p_t):
                    probe_tensors[str(pname)] = p_t
                    if pname not in dynamics_probe_shapes:
                        dynamics_probe_shapes[pname] = [int(x) for x in p_t.shape]
            if not probe_tensors:
                continue

            # Per-side losses on selected joints, then gradients to h_final.
            js_l, js_r = joint_specs[0], joint_specs[1]
            sl_l = _slice_for_joint(rot_slice, js_l.idx)
            sl_r = _slice_for_joint(rot_slice, js_r.idx)
            pred_l = pred[..., sl_l]
            pred_r = pred[..., sl_r]
            gt_l = gt[..., sl_l]
            gt_r = gt[..., sl_r]

            loss_l = F.mse_loss(pred_l, gt_l, reduction="mean")
            loss_r = F.mse_loss(pred_r, gt_r, reduction="mean")

            grad_reqs: List[Tuple[str, str, torch.Tensor]] = [
                ("__legacy__", js_l.name, loss_l),
                ("__legacy__", js_r.name, loss_r),
            ]

            if enable_component_grad:
                comp_losses = _build_component_side_losses(
                    out=out,
                    gt=gt,
                    joint_specs=joint_specs,
                    rot_slice=rot_slice,
                    loss_fn=loss_fn,
                    components=component_list,
                    component_weights=component_weights,
                )
                for comp in component_list:
                    per_side = comp_losses.get(comp, {})
                    for js in joint_specs:
                        l_comp = per_side.get(js.name, None)
                        if torch.is_tensor(l_comp):
                            grad_reqs.append((comp, js.name, l_comp))
                            try:
                                component_side_loss_rows.append(
                                    {
                                        "component": str(comp),
                                        "joint": str(js.name),
                                        "start": int(s0),
                                        "sic_start": int(int(s0) % int(cycle_len)),
                                        "loss": float(l_comp.detach().item()),
                                    }
                                )
                            except Exception:
                                pass
                        else:
                            component_grad_missing[comp][js.name] = int(component_grad_missing[comp][js.name]) + 1

            grad_map: Dict[Tuple[str, str], torch.Tensor] = {}
            grad_source_map: Dict[Tuple[str, str], str] = {}
            for i, (comp, jn, lval) in enumerate(grad_reqs):
                probe_keys = [
                    k for k in _probe_priority(comp, str(args.loss_branch)) if torch.is_tensor(probe_tensors.get(k))
                ]
                if not probe_keys:
                    probe_keys = [k for k, v in probe_tensors.items() if torch.is_tensor(v)]
                g = None
                used_key = "__none__"
                for k_idx, pkey in enumerate(probe_keys):
                    probe = probe_tensors.get(pkey, None)
                    if not torch.is_tensor(probe):
                        continue
                    g_try = torch.autograd.grad(lval, probe, retain_graph=True, allow_unused=True)[0]
                    if g_try is not None:
                        g = g_try
                        used_key = str(pkey)
                        break
                if g is None:
                    ref = None
                    for pkey in probe_keys:
                        pt = probe_tensors.get(pkey, None)
                        if torch.is_tensor(pt):
                            ref = pt
                            break
                    if ref is None and torch.is_tensor(h_final):
                        ref = h_final
                    if ref is None and torch.is_tensor(pred):
                        ref = pred
                    if ref is None:
                        continue
                    if comp == "__legacy__":
                        grad_unused_counts[jn] = int(grad_unused_counts[jn]) + 1
                    elif enable_component_grad:
                        component_grad_unused[comp][jn] = int(component_grad_unused[comp][jn]) + 1
                    g = torch.zeros_like(ref)
                else:
                    primary = probe_keys[0] if probe_keys else "__none__"
                    if used_key != str(primary):
                        if comp == "__legacy__":
                            grad_fallback_counts[jn] = int(grad_fallback_counts[jn]) + 1
                        elif enable_component_grad:
                            component_grad_fallback[comp][jn] = int(component_grad_fallback[comp][jn]) + 1
                    if comp == "__legacy__":
                        src = grad_probe_source_counts[jn]
                        src[used_key] = int(src.get(used_key, 0) + 1)
                    elif enable_component_grad:
                        src = component_grad_probe_sources[comp][jn]
                        src[used_key] = int(src.get(used_key, 0) + 1)
                grad_map[(comp, jn)] = g
                grad_source_map[(comp, jn)] = str(used_key)

                if enable_component_grad and comp != "__legacy__":
                    for pkey in dual_probe_names:
                        p = probe_tensors.get(str(pkey), None)
                        if not torch.is_tensor(p):
                            dual_component_grad_missing[comp][str(pkey)][jn] = (
                                int(dual_component_grad_missing[comp][str(pkey)][jn]) + 1
                            )
                            continue
                        g_dual = torch.autograd.grad(lval, p, retain_graph=True, allow_unused=True)[0]
                        if g_dual is None:
                            dual_component_grad_unused[comp][str(pkey)][jn] = (
                                int(dual_component_grad_unused[comp][str(pkey)][jn]) + 1
                            )
                            g_dual = torch.zeros_like(p)
                        grad_map[(f"__dual__:{comp}:{pkey}", jn)] = g_dual

            g_l = grad_map.get(("__legacy__", js_l.name), None)
            g_r = grad_map.get(("__legacy__", js_r.name), None)
            grad_t_l = _grad_time_l2_series(g_l)
            grad_t_r = _grad_time_l2_series(g_r)
            if grad_t_l is None or grad_t_r is None:
                continue

            # Legacy probe source used per window for left/right side.
            src_l = str(grad_source_map.get(("__legacy__", js_l.name), "__none__"))
            src_r = str(grad_source_map.get(("__legacy__", js_r.name), "__none__"))

            # Per-step reconstruction MSE (side-local) for context.
            mse_t_l = ((pred_l - gt_l) ** 2).mean(dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float64)
            mse_t_r = ((pred_r - gt_r) ** 2).mean(dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float64)

            w = angvel_raw.squeeze(0).reshape(angvel_raw.shape[1], len(ds.bone_names), 3)
            omega_t_l = np.abs(w[:, js_l.idx, axis].detach().cpu().numpy().astype(np.float64)) * (180.0 / math.pi)
            omega_t_r = np.abs(w[:, js_r.idx, axis].detach().cpu().numpy().astype(np.float64)) * (180.0 / math.pi)

            t_use_legacy = int(
                min(
                    grad_t_l.shape[0],
                    grad_t_r.shape[0],
                    mse_t_l.shape[0],
                    mse_t_r.shape[0],
                    omega_t_l.shape[0],
                    omega_t_r.shape[0],
                )
            )
            for t in range(t_use_legacy):
                sic = int((int(s0) + int(t)) % int(cycle_len))
                grad_rows.append(
                    {
                        "joint": js_l.name,
                        "start": int(s0),
                        "t": int(t),
                        "sic": int(sic),
                        "omega_deg_s": float(omega_t_l[t]),
                        "grad_h_l2": float(grad_t_l[t]),
                        "mse": float(mse_t_l[t]),
                        "probe_source": src_l,
                    }
                )
                grad_rows.append(
                    {
                        "joint": js_r.name,
                        "start": int(s0),
                        "t": int(t),
                        "sic": int(sic),
                        "omega_deg_s": float(omega_t_r[t]),
                        "grad_h_l2": float(grad_t_r[t]),
                        "mse": float(mse_t_r[t]),
                        "probe_source": src_r,
                    }
                )

            if enable_component_grad:
                for comp in component_list:
                    for js in joint_specs:
                        g_comp = grad_map.get((comp, js.name), None)
                        if g_comp is None:
                            continue
                        grad_t = _grad_time_l2_series(g_comp)
                        if grad_t is None:
                            continue
                        omega_t = omega_t_l if js.name == js_l.name else omega_t_r
                        t_use_comp = int(min(grad_t.shape[0], omega_t.shape[0]))
                        src_comp = str(grad_source_map.get((comp, js.name), "__none__"))
                        for t in range(t_use_comp):
                            sic = int((int(s0) + int(t)) % int(cycle_len))
                            component_grad_rows.append(
                                {
                                    "component": str(comp),
                                    "joint": js.name,
                                    "start": int(s0),
                                    "t": int(t),
                                    "sic": int(sic),
                                    "omega_deg_s": float(omega_t[t]),
                                    "grad_h_l2": float(grad_t[t]),
                                    "probe_source": src_comp,
                                }
                            )
                        for pkey in dual_probe_names:
                            g_dual = grad_map.get((f"__dual__:{comp}:{pkey}", js.name), None)
                            if g_dual is None:
                                continue
                            grad_t_dual = _grad_time_l2_series_with_time_hint(
                                g_dual,
                                time_hint=int(omega_t.shape[0]),
                            )
                            if grad_t_dual is None:
                                continue
                            t_use_dual = int(min(grad_t_dual.shape[0], omega_t.shape[0]))
                            for t in range(t_use_dual):
                                sic = int((int(s0) + int(t)) % int(cycle_len))
                                dual_component_grad_rows.append(
                                    {
                                        "component": str(comp),
                                        "probe": str(pkey),
                                        "joint": js.name,
                                        "start": int(s0),
                                        "t": int(t),
                                        "sic": int(sic),
                                        "omega_deg_s": float(omega_t[t]),
                                        "grad_h_l2": float(grad_t_dual[t]),
                                    }
                                )

    probe_recorder.close()

    # ---------- Aggregate gradient stats ----------
    def _group_grad_by_bin(joint_name: str) -> List[Dict[str, Any]]:
        rows = [r for r in grad_rows if r["joint"] == joint_name]
        out_rows: List[Dict[str, Any]] = []
        for i in range(len(bins) - 1):
            lo = float(bins[i])
            hi = float(bins[i + 1])
            sub = [r for r in rows if (r["omega_deg_s"] >= lo and (r["omega_deg_s"] < hi if math.isfinite(hi) else True))]
            gs = [float(r["grad_h_l2"]) for r in sub]
            ms = [float(r["mse"]) for r in sub]
            out_rows.append(
                {
                    "joint": joint_name,
                    "bin": bin_labels[i],
                    "n": int(len(sub)),
                    "grad_mean": _mean_or_nan(gs),
                    "grad_p90": _p90_or_nan(gs),
                    "mse_mean": _mean_or_nan(ms),
                }
            )
        return out_rows

    grad_bin_rows: List[Dict[str, Any]] = []
    for js in joint_specs:
        grad_bin_rows.extend(_group_grad_by_bin(js.name))

    # Per-SIC side asymmetry
    per_sic: List[Dict[str, Any]] = []
    for s in range(int(cycle_len)):
        rec: Dict[str, Any] = {"sic": int(s)}
        means: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for js in joint_specs:
            sub = [r for r in grad_rows if r["joint"] == js.name and int(r["sic"]) == int(s)]
            vals = [float(r["grad_h_l2"]) for r in sub]
            means[js.name] = _mean_or_nan(vals)
            counts[js.name] = int(len(vals))
            rec[f"{js.name}_n"] = int(len(vals))
            rec[f"{js.name}_grad_mean"] = means[js.name]
        l_name, r_name = joint_specs[0].name, joint_specs[1].name
        gl = means[l_name]
        gr = means[r_name]
        if math.isfinite(gl) and math.isfinite(gr) and gl > 1e-12:
            ratio = float(gr / gl)
            rec["grad_ratio_r_over_l"] = ratio
            rec["grad_log_ratio_r_over_l"] = float(math.log(ratio)) if ratio > 1e-12 else float("nan")
        else:
            rec["grad_ratio_r_over_l"] = float("nan")
            rec["grad_log_ratio_r_over_l"] = float("nan")
        per_sic.append(rec)

    # Global side stats
    def _joint_global(jn: str) -> Dict[str, Any]:
        sub = [r for r in grad_rows if r["joint"] == jn]
        gs = [float(r["grad_h_l2"]) for r in sub]
        ms = [float(r["mse"]) for r in sub]
        ws = [float(r["omega_deg_s"]) for r in sub]
        return {
            "n": int(len(sub)),
            "grad_mean": _mean_or_nan(gs),
            "grad_p90": _p90_or_nan(gs),
            "mse_mean": _mean_or_nan(ms),
            "omega_mean_deg_s": _mean_or_nan(ws),
        }

    g_l = _joint_global(joint_specs[0].name)
    g_r = _joint_global(joint_specs[1].name)
    if (
        math.isfinite(_safe_float(g_l.get("grad_mean")))
        and math.isfinite(_safe_float(g_r.get("grad_mean")))
        and float(g_l.get("grad_mean", 0.0)) > 1e-12
    ):
        grad_ratio_global = float(g_r["grad_mean"] / g_l["grad_mean"])
    else:
        grad_ratio_global = float("nan")

    component_global: Dict[str, Any] = {}
    component_per_sic: Dict[str, List[Dict[str, Any]]] = {}
    component_side_loss_global: Dict[str, Any] = {}
    component_side_loss_per_sic_start: Dict[str, List[Dict[str, Any]]] = {}
    dual_probe_global: Dict[str, Dict[str, Any]] = {}
    dual_probe_per_sic: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    if enable_component_grad:
        l_name, r_name = joint_specs[0].name, joint_specs[1].name
        for comp in component_list:
            component_global[comp] = {}
            for jn in (l_name, r_name):
                sub = [r for r in component_grad_rows if r["component"] == comp and r["joint"] == jn]
                gs = [float(r["grad_h_l2"]) for r in sub]
                ws = [float(r["omega_deg_s"]) for r in sub]
                component_global[comp][jn] = {
                    "n": int(len(sub)),
                    "grad_mean": _mean_or_nan(gs),
                    "grad_p90": _p90_or_nan(gs),
                    "omega_mean_deg_s": _mean_or_nan(ws),
                }
            gl = _safe_float(component_global[comp][l_name].get("grad_mean"))
            gr = _safe_float(component_global[comp][r_name].get("grad_mean"))
            if math.isfinite(gl) and math.isfinite(gr) and gl > 1e-12:
                ratio = float(gr / gl)
            else:
                ratio = float("nan")
            component_global[comp]["grad_ratio_r_over_l"] = ratio
            component_global[comp]["grad_unused_windows"] = dict(component_grad_unused.get(comp, {}))
            component_global[comp]["grad_missing_windows"] = dict(component_grad_missing.get(comp, {}))
            component_global[comp]["grad_fallback_windows"] = dict(component_grad_fallback.get(comp, {}))
            component_global[comp]["probe_source_windows"] = dict(component_grad_probe_sources.get(comp, {}))

            per_sic_rows: List[Dict[str, Any]] = []
            for s in range(int(cycle_len)):
                rec: Dict[str, Any] = {"sic": int(s)}
                means: Dict[str, float] = {}
                for jn in (l_name, r_name):
                    sub = [
                        r
                        for r in component_grad_rows
                        if r["component"] == comp and r["joint"] == jn and int(r["sic"]) == int(s)
                    ]
                    vals = [float(r["grad_h_l2"]) for r in sub]
                    means[jn] = _mean_or_nan(vals)
                    rec[f"{jn}_n"] = int(len(vals))
                    rec[f"{jn}_grad_mean"] = means[jn]
                gls = means[l_name]
                grs = means[r_name]
                if math.isfinite(gls) and math.isfinite(grs) and gls > 1e-12:
                    ratio = float(grs / gls)
                    rec["grad_ratio_r_over_l"] = ratio
                    rec["grad_log_ratio_r_over_l"] = float(math.log(ratio)) if ratio > 1e-12 else float("nan")
                else:
                    rec["grad_ratio_r_over_l"] = float("nan")
                    rec["grad_log_ratio_r_over_l"] = float("nan")
                per_sic_rows.append(rec)
            component_per_sic[comp] = per_sic_rows

            dual_probe_global[comp] = {}
            dual_probe_per_sic[comp] = {}
            for probe_name in dual_probe_names:
                dual_probe_global[comp][probe_name] = {}
                for jn in (l_name, r_name):
                    sub_dual = [
                        r
                        for r in dual_component_grad_rows
                        if r["component"] == comp and r["probe"] == probe_name and r["joint"] == jn
                    ]
                    gs_dual = [float(r["grad_h_l2"]) for r in sub_dual]
                    ws_dual = [float(r["omega_deg_s"]) for r in sub_dual]
                    dual_probe_global[comp][probe_name][jn] = {
                        "n": int(len(sub_dual)),
                        "grad_mean": _mean_or_nan(gs_dual),
                        "grad_p90": _p90_or_nan(gs_dual),
                        "omega_mean_deg_s": _mean_or_nan(ws_dual),
                    }
                gl_dual = _safe_float(dual_probe_global[comp][probe_name][l_name].get("grad_mean"))
                gr_dual = _safe_float(dual_probe_global[comp][probe_name][r_name].get("grad_mean"))
                if math.isfinite(gl_dual) and math.isfinite(gr_dual) and gl_dual > 1e-12:
                    ratio_dual = float(gr_dual / gl_dual)
                else:
                    ratio_dual = float("nan")
                dual_probe_global[comp][probe_name]["grad_ratio_r_over_l"] = ratio_dual
                dual_probe_global[comp][probe_name]["grad_unused_windows"] = dict(
                    dual_component_grad_unused.get(comp, {}).get(probe_name, {})
                )
                dual_probe_global[comp][probe_name]["grad_missing_windows"] = dict(
                    dual_component_grad_missing.get(comp, {}).get(probe_name, {})
                )

                per_sic_dual_rows: List[Dict[str, Any]] = []
                for s in range(int(cycle_len)):
                    rec_dual: Dict[str, Any] = {"sic": int(s)}
                    means_dual: Dict[str, float] = {}
                    for jn in (l_name, r_name):
                        sub_dual = [
                            r
                            for r in dual_component_grad_rows
                            if (
                                r["component"] == comp
                                and r["probe"] == probe_name
                                and r["joint"] == jn
                                and int(r["sic"]) == int(s)
                            )
                        ]
                        vals_dual = [float(r["grad_h_l2"]) for r in sub_dual]
                        means_dual[jn] = _mean_or_nan(vals_dual)
                        rec_dual[f"{jn}_n"] = int(len(vals_dual))
                        rec_dual[f"{jn}_grad_mean"] = means_dual[jn]
                    gls_dual = means_dual[l_name]
                    grs_dual = means_dual[r_name]
                    if math.isfinite(gls_dual) and math.isfinite(grs_dual) and gls_dual > 1e-12:
                        ratio_dual = float(grs_dual / gls_dual)
                        rec_dual["grad_ratio_r_over_l"] = ratio_dual
                        rec_dual["grad_log_ratio_r_over_l"] = float(math.log(ratio_dual)) if ratio_dual > 1e-12 else float("nan")
                    else:
                        rec_dual["grad_ratio_r_over_l"] = float("nan")
                        rec_dual["grad_log_ratio_r_over_l"] = float("nan")
                    per_sic_dual_rows.append(rec_dual)
                dual_probe_per_sic[comp][probe_name] = per_sic_dual_rows

            # Component side losses (configured-weighted scalar per window).
            component_side_loss_global[comp] = {}
            for jn in (l_name, r_name):
                sub_loss = [r for r in component_side_loss_rows if r["component"] == comp and r["joint"] == jn]
                lv = [float(r["loss"]) for r in sub_loss]
                component_side_loss_global[comp][jn] = {
                    "n": int(len(sub_loss)),
                    "loss_mean": _mean_or_nan(lv),
                    "loss_p90": _p90_or_nan(lv),
                }
            ll = _safe_float(component_side_loss_global[comp][l_name].get("loss_mean"))
            lr = _safe_float(component_side_loss_global[comp][r_name].get("loss_mean"))
            if math.isfinite(ll) and math.isfinite(lr) and ll > 1e-12:
                ratio_loss = float(lr / ll)
            else:
                ratio_loss = float("nan")
            if math.isfinite(ratio_loss) and ratio_loss > 1e-12:
                log_ratio_loss = float(math.log(ratio_loss))
            else:
                log_ratio_loss = float("nan")
            component_side_loss_global[comp]["loss_ratio_r_over_l"] = ratio_loss
            component_side_loss_global[comp]["loss_log_ratio_r_over_l"] = log_ratio_loss

            per_sic_loss_rows: List[Dict[str, Any]] = []
            for s in range(int(cycle_len)):
                rec_loss: Dict[str, Any] = {"sic_start": int(s)}
                means_loss: Dict[str, float] = {}
                for jn in (l_name, r_name):
                    sub_loss = [
                        r
                        for r in component_side_loss_rows
                        if r["component"] == comp and r["joint"] == jn and int(r["sic_start"]) == int(s)
                    ]
                    vals_loss = [float(r["loss"]) for r in sub_loss]
                    means_loss[jn] = _mean_or_nan(vals_loss)
                    rec_loss[f"{jn}_n"] = int(len(vals_loss))
                    rec_loss[f"{jn}_loss_mean"] = means_loss[jn]
                ll_s = means_loss[l_name]
                lr_s = means_loss[r_name]
                if math.isfinite(ll_s) and math.isfinite(lr_s) and ll_s > 1e-12:
                    ratio_s = float(lr_s / ll_s)
                    rec_loss["loss_ratio_r_over_l"] = ratio_s
                    rec_loss["loss_log_ratio_r_over_l"] = float(math.log(ratio_s)) if ratio_s > 1e-12 else float("nan")
                else:
                    rec_loss["loss_ratio_r_over_l"] = float("nan")
                    rec_loss["loss_log_ratio_r_over_l"] = float("nan")
                per_sic_loss_rows.append(rec_loss)
            component_side_loss_per_sic_start[comp] = per_sic_loss_rows

    cond_contact_ch_l = int(getattr(model, "direct_pose_leg_contact_ch_l", 0) or 0)
    cond_contact_ch_r = int(getattr(model, "direct_pose_leg_contact_ch_r", 1) or 1)
    cond_rows_for_side = cond_side_cond_rows if cond_side_cond_rows else cond_all_rows
    cond_contacts_for_side = cond_side_contact_rows if cond_side_cond_rows else []
    cond_in_side_stats = _summarize_cond_input_side_distribution(
        cond_rows=cond_rows_for_side,
        contact_rows=cond_contacts_for_side,
        side_margin=float(args.cond_side_margin),
        contact_ch_l=cond_contact_ch_l,
        contact_ch_r=cond_contact_ch_r,
        topk=int(args.cond_side_topk),
        eps=1e-8,
    )

    root_out_ratio = float("nan")
    root_cond_ratio = float("nan")
    root_direct_probe_ratio: Dict[str, float] = {}
    root_direct_probe_log_ratio: Dict[str, float] = {}
    root_direct_probe_unused: Dict[str, Dict[str, int]] = {}
    root_direct_probe_missing: Dict[str, Dict[str, int]] = {}
    if enable_component_grad:
        direct_dual = dual_probe_global.get("direct_pose", {}) if isinstance(dual_probe_global, dict) else {}
        if isinstance(direct_dual, Mapping):
            for pkey in dual_probe_names:
                p_payload = direct_dual.get(str(pkey), {})
                if isinstance(p_payload, Mapping):
                    ratio = _safe_float(p_payload.get("grad_ratio_r_over_l", float("nan")))
                else:
                    ratio = float("nan")
                root_direct_probe_ratio[str(pkey)] = ratio
                if math.isfinite(ratio) and ratio > 1e-12:
                    root_direct_probe_log_ratio[str(pkey)] = float(math.log(ratio))
                else:
                    root_direct_probe_log_ratio[str(pkey)] = float("nan")
                if isinstance(p_payload, Mapping):
                    root_direct_probe_unused[str(pkey)] = {
                        str(k): int(v)
                        for k, v in (p_payload.get("grad_unused_windows", {}) or {}).items()
                    }
                    root_direct_probe_missing[str(pkey)] = {
                        str(k): int(v)
                        for k, v in (p_payload.get("grad_missing_windows", {}) or {}).items()
                    }
            root_out_ratio = _safe_float(root_direct_probe_ratio.get("out_direct", float("nan")))
            root_cond_ratio = _safe_float(root_direct_probe_ratio.get("cond_in", float("nan")))
    root_ratio_gap = float("nan")
    if math.isfinite(root_out_ratio) and math.isfinite(root_cond_ratio):
        root_ratio_gap = float(root_out_ratio - root_cond_ratio)
    root_direct_loss_ratio = float("nan")
    root_direct_loss_log_ratio = float("nan")
    if enable_component_grad:
        dp_loss = component_side_loss_global.get("direct_pose", {}) if isinstance(component_side_loss_global, dict) else {}
        if isinstance(dp_loss, Mapping):
            root_direct_loss_ratio = _safe_float(dp_loss.get("loss_ratio_r_over_l", float("nan")))
            root_direct_loss_log_ratio = _safe_float(dp_loss.get("loss_log_ratio_r_over_l", float("nan")))

    def _log_delta(dst_key: str, src_key: str) -> float:
        dst = _safe_float(root_direct_probe_log_ratio.get(dst_key, float("nan")))
        src = _safe_float(root_direct_probe_log_ratio.get(src_key, float("nan")))
        if math.isfinite(dst) and math.isfinite(src):
            return float(dst - src)
        return float("nan")

    dynamics_step0 = {
        "requested_probes": [str(x) for x in requested_dynamics_probes],
        "active_probes": [str(x) for x in dynamics_probe_names],
        "probe_ratio_r_over_l": root_direct_probe_ratio,
        "probe_log_ratio_r_over_l": root_direct_probe_log_ratio,
        "probe_unused_windows": root_direct_probe_unused,
        "probe_missing_windows": root_direct_probe_missing,
        "amp_log_shared_to_temporal": _log_delta("temporal_pre_pasa", "shared_encoder_pre_act"),
        "amp_log_temporal_to_direct_head_pre_out": _log_delta("direct_head_pre_out", "temporal_pre_pasa"),
        "amp_log_direct_head_pre_out_to_out_direct": _log_delta("out_direct", "direct_head_pre_out"),
        "amp_log_shared_to_out_direct": _log_delta("out_direct", "shared_encoder_pre_act"),
        "module_status": dynamics_probe_meta,
        "sample_probe_shapes": dynamics_probe_shapes,
    }
    direct_head_sym = _probe_direct_head_symmetry(
        model=model,
        rot_slice=rot_slice,
        joint_specs=joint_specs,
        eps=1e-12,
    )

    # ---------- Build payload ----------
    clip_window_counts: Dict[str, int] = {}
    for cid, _s in ds.index:
        nm = Path(str(ds.clips[int(cid)].npz_path)).stem
        clip_window_counts[nm] = int(clip_window_counts.get(nm, 0) + 1)

    payload: Dict[str, Any] = {
        "config_json": str(Path(args.config_json).expanduser().resolve()),
        "ckpt": str(Path(args.ckpt).expanduser().resolve()),
        "target_clip": str(args.target_clip),
        "dataset": {
            "num_clips": int(len(ds.clips)),
            "seq_len": int(seq_len),
            "index_mode": str(getattr(ds, "index_mode", "unknown")),
            "total_windows": int(len(ds.index)),
            "clip_window_counts": clip_window_counts,
            "target_clip_id": int(clip_id),
            "target_clip_path": str(clip.npz_path),
            "target_T": int(T),
            "target_windows": int(window_count_effective),
            "target_windows_native": int(window_count_native),
            "target_windows_effective": int(window_count_effective),
        },
        "distribution": {
            "omega_axis": str(args.omega_axis),
            "omega_bins_deg_s": [float(x) if math.isfinite(float(x)) else "inf" for x in bins],
            "coverage_frame_min": float(np.min(coverage)) if coverage.size else float("nan"),
            "coverage_frame_max": float(np.max(coverage)) if coverage.size else float("nan"),
            "coverage_frame_mean": float(np.mean(coverage)) if coverage.size else float("nan"),
            "cycle_len": int(cycle_len),
            "sic_raw_counts": sic_raw_counts.tolist(),
            "sic_sampled_counts": sic_samp_counts.tolist(),
            "omega_hist_rows": dist_rows,
        },
        "gradient": {
            "loss_branch": str(args.loss_branch),
            "joint_pair": [joint_specs[0].name, joint_specs[1].name],
            "num_windows_used": int(len(target_indices)),
            "num_rows": int(len(grad_rows)),
            "global": {
                joint_specs[0].name: g_l,
                joint_specs[1].name: g_r,
                "grad_ratio_r_over_l": grad_ratio_global,
                "grad_unused_windows": grad_unused_counts,
                "grad_fallback_windows": grad_fallback_counts,
                "probe_source_windows": grad_probe_source_counts,
            },
            "omega_bin_rows": grad_bin_rows,
            "per_sic_rows": per_sic,
        },
        "component_gradient": {
            "enabled": bool(enable_component_grad),
            "components": list(component_list),
            "component_weights": component_weights,
            "probe_priority": {comp: _probe_priority(comp, str(args.loss_branch)) for comp in component_list},
            "num_rows": int(len(component_grad_rows)),
            "global": component_global,
            "per_sic_rows": component_per_sic,
        },
        "component_side_loss": {
            "enabled": bool(enable_component_grad),
            "components": list(component_list),
            "component_weights": component_weights,
            "num_rows": int(len(component_side_loss_rows)),
            "global": component_side_loss_global,
            "per_sic_start_rows": component_side_loss_per_sic_start,
        },
        "dual_probe_gradient": {
            "enabled": bool(enable_component_grad),
            "components": list(component_list),
            "probes": list(dual_probe_names),
            "num_rows": int(len(dual_component_grad_rows)),
            "global": dual_probe_global,
            "per_sic_rows": dual_probe_per_sic,
        },
        "root_cause_probe": {
            "joint_pair": [joint_specs[0].name, joint_specs[1].name],
            "out_direct_ratio_step0": root_out_ratio,
            "cond_in_ratio_step0": root_cond_ratio,
            "ratio_gap_step0": root_ratio_gap,
            "direct_pose_loss_ratio_step0": root_direct_loss_ratio,
            "direct_pose_loss_log_ratio_step0": root_direct_loss_log_ratio,
            "dynamics_probe_step0": dynamics_step0,
            "cond_in_side_stats": cond_in_side_stats,
            "direct_head_symmetry": direct_head_sym,
        },
        "loss_setup": {
            "loss_config_override_enabled": bool(not args.skip_loss_config_override),
            "loss_config_overrides_applied": loss_cfg_overrides,
            "norm_sync_status": norm_sync_status,
            "rot_geo_weight_source": str(rot_geo_w_source),
        },
        "notes": {
            "sampling_bias_definition": "sampled = expected frame frequency induced by dataset_index_mode (sliding/start0/clip_random/sic_balanced).",
            "gradient_definition": "grad_h_l2 = ||dL_joint / d(probe_t)||_2 with auto probe fallback (h_final -> branch tensor) to avoid false zeros when h_final is disconnected.",
            "component_gradient_definition": "component dL/dprobe_t uses configured component weights: rot_geo (*rot_geo_weight), rot_vel (*w_rot_vel), direct_pose (*w_direct_pose), direct_delta (*w_direct_delta).",
            "component_side_loss_definition": "component side loss stores configured-weighted scalar component losses per window and side (L/R), with ratio_r_over_l in global/per-sic_start summaries.",
            "dual_probe_definition": "Dual probe split reports component dL wrt out_direct and cond_in in parallel: grad_h_l2 = ||dL_component,joint / d(probe_t)||_2.",
            "dynamics_probe_definition": "Layer-wise dynamics probes append dL wrt shared_encoder_pre_act(shared_encoder.0 output), temporal_pre_pasa(input to _pasa_lnq ~= h_temporal), and direct_head_pre_out(input to direct_pose_head last linear).",
            "cond_side_definition": "cond_in side split uses contact channels (left/right) and margin: left if (c_l-c_r)>thr, right if < -thr.",
            "augmentation_caveat": "This diagnostic does not stochastically apply train-time lr_swap/time-warp/noise; it measures current model + dataset windowing.",
        },
    }

    out_dir = Path(args.out_dir).expanduser()
    if not str(out_dir):
        out_dir = Path("debug_output") / f"_dist_grad_closure_{date.today().strftime('%Y%m%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "sampling_grad_closure.json"
    out_md = out_dir / "sampling_grad_closure.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- Markdown ----------
    lines: List[str] = []
    lines.append("# Stage7 distribution + gradient closure")
    lines.append("")
    lines.append(f"- ckpt: `{Path(args.ckpt).expanduser().resolve()}`")
    lines.append(f"- target_clip: `{args.target_clip}`")
    lines.append(f"- seq_len={seq_len}, cycle_len={cycle_len}, branch={args.loss_branch}, omega_axis={args.omega_axis}")
    lines.append("")
    lines.append("## Dataset/window profile")
    lines.append("")
    lines.append(
        f"- clips={len(ds.clips)}, total_windows={len(ds.index)}, "
        f"target_windows={window_count_effective} (native={window_count_native})"
    )
    lines.append(f"- target clip path: `{clip.npz_path}`")
    lines.append(f"- target frame count T={T}")
    lines.append(
        "- frame coverage (sampled-window multiplicity): "
        f"min={payload['distribution']['coverage_frame_min']:.1f}, "
        f"max={payload['distribution']['coverage_frame_max']:.1f}, "
        f"mean={payload['distribution']['coverage_frame_mean']:.1f}"
    )
    lines.append(
        f"- loss setup: cfg_override={bool(payload['loss_setup']['loss_config_override_enabled'])}, "
        f"rot_geo_weight_source={payload['loss_setup']['rot_geo_weight_source']}"
    )
    lines.append("")
    lines.append("|clip|windows|")
    lines.append("|:--|--:|")
    for k, v in sorted(clip_window_counts.items(), key=lambda kv: kv[0]):
        lines.append(f"|{k}|{int(v)}|")

    lines.append("")
    lines.append("## Omega distribution (raw vs sampled)")
    lines.append("")
    lines.append("|joint|bin (deg/s)|raw_count|sampled_count|raw_frac|sampled_frac|")
    lines.append("|:--|:--|--:|--:|--:|--:|")
    for r in dist_rows:
        lines.append(
            f"|{r['joint']}|{r['bin']}|{r['raw_count']:.1f}|{r['sampled_count']:.1f}|"
            f"{r['raw_frac']:.4f}|{r['sampled_frac']:.4f}|"
        )

    lines.append("")
    lines.append("## Probe gradient by omega bins")
    lines.append("")
    lines.append("|joint|bin (deg/s)|N|grad_mean|grad_p90|mse_mean|")
    lines.append("|:--|:--|--:|--:|--:|--:|")
    for r in grad_bin_rows:
        gm = r["grad_mean"]
        gp = r["grad_p90"]
        mm = r["mse_mean"]
        lines.append(
            f"|{r['joint']}|{r['bin']}|{int(r['n'])}|"
            f"{gm:.6f}|{gp:.6f}|{mm:.6f}|"
        )

    lines.append("")
    lines.append("## Global side gradient stats")
    lines.append("")
    l_name, r_name = joint_specs[0].name, joint_specs[1].name
    lines.append("|joint|N|grad_mean|grad_p90|mse_mean|omega_mean_deg_s|")
    lines.append("|:--|--:|--:|--:|--:|--:|")
    for jn in (l_name, r_name):
        g = payload["gradient"]["global"][jn]
        lines.append(
            f"|{jn}|{int(g['n'])}|{_safe_float(g['grad_mean']):.6f}|{_safe_float(g['grad_p90']):.6f}|"
            f"{_safe_float(g['mse_mean']):.6f}|{_safe_float(g['omega_mean_deg_s']):.3f}|"
        )
    lines.append(
        f"- global grad ratio (right/left): {payload['gradient']['global']['grad_ratio_r_over_l']:.6f}"
    )
    lines.append(
        f"- legacy unused windows (L/R): "
        f"{int(payload['gradient']['global']['grad_unused_windows'].get(l_name, 0))}/"
        f"{int(payload['gradient']['global']['grad_unused_windows'].get(r_name, 0))}"
    )
    lines.append(
        f"- legacy fallback windows (L/R): "
        f"{int(payload['gradient']['global']['grad_fallback_windows'].get(l_name, 0))}/"
        f"{int(payload['gradient']['global']['grad_fallback_windows'].get(r_name, 0))}"
    )

    def _fmt_probe_sources(x: Any) -> str:
        if not isinstance(x, Mapping):
            return "-"
        items = [(str(k), int(v)) for k, v in x.items() if int(v) > 0]
        if not items:
            return "-"
        items = sorted(items, key=lambda kv: (-kv[1], kv[0]))
        return ",".join(f"{k}:{v}" for k, v in items[:4])

    legacy_src = payload["gradient"]["global"].get("probe_source_windows", {})
    lines.append(
        f"- legacy probe sources (L): {_fmt_probe_sources(legacy_src.get(l_name, {}))}; "
        f"(R): {_fmt_probe_sources(legacy_src.get(r_name, {}))}"
    )

    if enable_component_grad and component_list:
        lines.append("")
        lines.append("## Component gradient split (configured-weight dL/dprobe)")
        lines.append("")
        lines.append(
            "|component|weight|left_grad_mean|right_grad_mean|ratio_r_over_l|"
            "left_unused/fallback/missing|right_unused/fallback/missing|left_probe_src|right_probe_src|"
        )
        lines.append("|:--|--:|--:|--:|--:|--:|--:|:--|:--|")
        comp_global_payload = payload.get("component_gradient", {}).get("global", {})
        comp_weights_payload = payload.get("component_gradient", {}).get("component_weights", {})
        for comp in component_list:
            cg = comp_global_payload.get(comp, {})
            gl = cg.get(l_name, {})
            gr = cg.get(r_name, {})
            gu = cg.get("grad_unused_windows", {})
            gf = cg.get("grad_fallback_windows", {})
            gm = cg.get("grad_missing_windows", {})
            gs = cg.get("probe_source_windows", {})
            lines.append(
                f"|{comp}|{_safe_float(comp_weights_payload.get(comp)):.4f}|"
                f"{_safe_float(gl.get('grad_mean')):.6f}|{_safe_float(gr.get('grad_mean')):.6f}|"
                f"{_safe_float(cg.get('grad_ratio_r_over_l')):.6f}|"
                f"{int(gu.get(l_name, 0))}/{int(gf.get(l_name, 0))}/{int(gm.get(l_name, 0))}|"
                f"{int(gu.get(r_name, 0))}/{int(gf.get(r_name, 0))}/{int(gm.get(r_name, 0))}|"
                f"{_fmt_probe_sources(gs.get(l_name, {}))}|"
                f"{_fmt_probe_sources(gs.get(r_name, {}))}|"
            )

        for comp in component_list:
            rows_comp = component_per_sic.get(comp, [])
            sic_valid_comp = [r for r in rows_comp if math.isfinite(_safe_float(r.get("grad_log_ratio_r_over_l")))]
            sic_valid_comp = sorted(
                sic_valid_comp,
                key=lambda x: abs(float(x["grad_log_ratio_r_over_l"])),
                reverse=True,
            )
            lines.append("")
            lines.append(f"### Component SIC asymmetry: {comp}")
            lines.append("")
            lines.append("|sic|left_n|right_n|left_grad|right_grad|ratio_r_over_l|log_ratio|")
            lines.append("|--:|--:|--:|--:|--:|--:|--:|")
            for r in sic_valid_comp[:8]:
                lines.append(
                    f"|{int(r['sic'])}|{int(r[f'{l_name}_n'])}|{int(r[f'{r_name}_n'])}|"
                    f"{_safe_float(r[f'{l_name}_grad_mean']):.6f}|{_safe_float(r[f'{r_name}_grad_mean']):.6f}|"
                    f"{_safe_float(r['grad_ratio_r_over_l']):.6f}|{_safe_float(r['grad_log_ratio_r_over_l']):.6f}|"
                )

        lines.append("")
        lines.append("## Component side-loss split (configured-weight scalar)")
        lines.append("")
        lines.append("|component|weight|left_loss_mean|right_loss_mean|ratio_r_over_l|log_ratio|")
        lines.append("|:--|--:|--:|--:|--:|--:|")
        comp_loss_global_payload = payload.get("component_side_loss", {}).get("global", {})
        for comp in component_list:
            cg = comp_loss_global_payload.get(comp, {})
            ll = cg.get(l_name, {})
            rr = cg.get(r_name, {})
            lines.append(
                f"|{comp}|{_safe_float(comp_weights_payload.get(comp)):.4f}|"
                f"{_safe_float(ll.get('loss_mean')):.6f}|{_safe_float(rr.get('loss_mean')):.6f}|"
                f"{_safe_float(cg.get('loss_ratio_r_over_l')):.6f}|"
                f"{_safe_float(cg.get('loss_log_ratio_r_over_l')):.6f}|"
            )

    if enable_component_grad and component_list:
        lines.append("")
        lines.append("## Dual probe split (dL/dout_direct vs dL/dcond_in)")
        lines.append("")
        lines.append(
            "|component|probe|left_grad_mean|right_grad_mean|ratio_r_over_l|left_unused/missing|right_unused/missing|"
        )
        lines.append("|:--|:--|--:|--:|--:|--:|--:|")
        dual_global_payload = payload.get("dual_probe_gradient", {}).get("global", {})
        for comp in component_list:
            per_probe = dual_global_payload.get(comp, {})
            for probe_name in dual_probe_names:
                pg = per_probe.get(probe_name, {})
                pl = pg.get(l_name, {})
                pr = pg.get(r_name, {})
                gu = pg.get("grad_unused_windows", {})
                gm = pg.get("grad_missing_windows", {})
                lines.append(
                    f"|{comp}|{probe_name}|"
                    f"{_safe_float(pl.get('grad_mean')):.6f}|{_safe_float(pr.get('grad_mean')):.6f}|"
                    f"{_safe_float(pg.get('grad_ratio_r_over_l')):.6f}|"
                    f"{int(gu.get(l_name, 0))}/{int(gm.get(l_name, 0))}|"
                    f"{int(gu.get(r_name, 0))}/{int(gm.get(r_name, 0))}|"
                )

        dual_per_sic_payload = payload.get("dual_probe_gradient", {}).get("per_sic_rows", {})
        for comp in component_list:
            per_probe = dual_per_sic_payload.get(comp, {})
            for probe_name in dual_probe_names:
                rows_dual = per_probe.get(probe_name, [])
                valid_dual = [r for r in rows_dual if math.isfinite(_safe_float(r.get("grad_log_ratio_r_over_l")))]
                valid_dual = sorted(valid_dual, key=lambda x: abs(float(x["grad_log_ratio_r_over_l"])), reverse=True)
                lines.append("")
                lines.append(f"### Dual SIC asymmetry: {comp} / {probe_name}")
                lines.append("")
                lines.append("|sic|left_n|right_n|left_grad|right_grad|ratio_r_over_l|log_ratio|")
                lines.append("|--:|--:|--:|--:|--:|--:|--:|")
                for r in valid_dual[:6]:
                    lines.append(
                        f"|{int(r['sic'])}|{int(r[f'{l_name}_n'])}|{int(r[f'{r_name}_n'])}|"
                        f"{_safe_float(r[f'{l_name}_grad_mean']):.6f}|{_safe_float(r[f'{r_name}_grad_mean']):.6f}|"
                        f"{_safe_float(r['grad_ratio_r_over_l']):.6f}|{_safe_float(r['grad_log_ratio_r_over_l']):.6f}|"
                    )

    root_probe_payload = payload.get("root_cause_probe", {})
    if isinstance(root_probe_payload, Mapping):
        sym = root_probe_payload.get("direct_head_symmetry", {})
        lines.append("")
        lines.append("## Root-cause probe (step0 + direct-head symmetry)")
        lines.append("")
        lines.append(
            f"- step0 ratio (direct_pose@out_direct, r/l): "
            f"{_safe_float(root_probe_payload.get('out_direct_ratio_step0', float('nan'))):.6f}"
        )
        lines.append(
            f"- step0 ratio (direct_pose@cond_in, r/l): "
            f"{_safe_float(root_probe_payload.get('cond_in_ratio_step0', float('nan'))):.6f}"
        )
        lines.append(
            f"- step0 ratio gap (out_direct - cond_in): "
            f"{_safe_float(root_probe_payload.get('ratio_gap_step0', float('nan'))):.6f}"
        )
        lines.append(
            f"- step0 direct_pose loss ratio (R/L): "
            f"{_safe_float(root_probe_payload.get('direct_pose_loss_ratio_step0', float('nan'))):.6f} "
            f"(log={_safe_float(root_probe_payload.get('direct_pose_loss_log_ratio_step0', float('nan'))):+.6f})"
        )
        dyn = root_probe_payload.get("dynamics_probe_step0", {})
        if isinstance(dyn, Mapping):
            dyn_req = dyn.get("requested_probes", [])
            dyn_act = dyn.get("active_probes", [])
            lines.append(
                f"- dynamics probes (requested -> active): "
                f"`{','.join(str(x) for x in dyn_req)}` -> `{','.join(str(x) for x in dyn_act)}`"
            )
            ratio_map = dyn.get("probe_ratio_r_over_l", {})
            log_map = dyn.get("probe_log_ratio_r_over_l", {})
            unused_map = dyn.get("probe_unused_windows", {})
            missing_map = dyn.get("probe_missing_windows", {})
            if isinstance(ratio_map, Mapping) and ratio_map:
                lines.append("")
                lines.append("|probe|ratio_r_over_l|log_ratio|unused(L/R)|missing(L/R)|")
                lines.append("|:--|--:|--:|:--|:--|")
                for pkey in [str(p) for p in dual_probe_names]:
                    un = unused_map.get(pkey, {}) if isinstance(unused_map, Mapping) else {}
                    ms = missing_map.get(pkey, {}) if isinstance(missing_map, Mapping) else {}
                    lines.append(
                        f"|{pkey}|{_safe_float(ratio_map.get(pkey, float('nan'))):.6f}|"
                        f"{_safe_float(log_map.get(pkey, float('nan'))):.6f}|"
                        f"{int((un or {}).get(l_name, 0))}/{int((un or {}).get(r_name, 0))}|"
                        f"{int((ms or {}).get(l_name, 0))}/{int((ms or {}).get(r_name, 0))}|"
                    )
            lines.append(
                f"- dynamics amp log(shared->temporal): "
                f"{_safe_float(dyn.get('amp_log_shared_to_temporal', float('nan'))):+.6f}"
            )
            lines.append(
                f"- dynamics amp log(temporal->direct_head_pre_out): "
                f"{_safe_float(dyn.get('amp_log_temporal_to_direct_head_pre_out', float('nan'))):+.6f}"
            )
            lines.append(
                f"- dynamics amp log(direct_head_pre_out->out_direct): "
                f"{_safe_float(dyn.get('amp_log_direct_head_pre_out_to_out_direct', float('nan'))):+.6f}"
            )
            lines.append(
                f"- dynamics amp log(shared->out_direct): "
                f"{_safe_float(dyn.get('amp_log_shared_to_out_direct', float('nan'))):+.6f}"
            )
        cond_side = root_probe_payload.get("cond_in_side_stats", {})
        if isinstance(cond_side, Mapping):
            lines.append(
                f"- cond_in side stats: status=`{cond_side.get('status', 'unknown')}`, "
                f"rows={int(cond_side.get('num_rows', 0))}, "
                f"left/right/neutral={int(cond_side.get('left_rows', 0))}/"
                f"{int(cond_side.get('right_rows', 0))}/{int(cond_side.get('neutral_rows', 0))}, "
                f"margin={_safe_float(cond_side.get('side_margin', float('nan'))):.3f}"
            )
            lines.append(
                f"- cond_in side |z|: mean={_safe_float(cond_side.get('mean_abs_z', float('nan'))):.6f}, "
                f"max={_safe_float(cond_side.get('max_abs_z', float('nan'))):.6f}"
            )
            top_rows = cond_side.get("top_abs_z_rows", [])
            if isinstance(top_rows, Sequence) and len(top_rows) > 0:
                lines.append("")
                lines.append("|cond_dim|label|mean_left|mean_right|delta(R-L)|abs_z|")
                lines.append("|--:|:--|--:|--:|--:|--:|")
                for row in top_rows[: int(max(1, args.cond_side_topk))]:
                    if not isinstance(row, Mapping):
                        continue
                    lines.append(
                        f"|{int(row.get('dim', -1))}|{row.get('label', '-')}"
                        f"|{_safe_float(row.get('mean_left', float('nan'))):.6f}"
                        f"|{_safe_float(row.get('mean_right', float('nan'))):.6f}"
                        f"|{_safe_float(row.get('mean_diff_right_minus_left', float('nan'))):.6f}"
                        f"|{_safe_float(row.get('abs_z', float('nan'))):.6f}|"
                    )
        if isinstance(sym, Mapping):
            lines.append("")
            lines.append(
                f"- direct head symmetry status: `{sym.get('status', 'unknown')}` "
                f"(layer={sym.get('layer', '-')})"
            )
            lines.append("")
            lines.append("|metric|value|")
            lines.append("|:--|--:|")
            lines.append(f"|weight_norm_ratio_r_over_l|{_safe_float(sym.get('weight_norm_ratio_r_over_l')):.6f}|")
            lines.append(f"|weight_rel_l2_raw|{_safe_float(sym.get('weight_rel_l2_raw')):.6f}|")
            lines.append(f"|weight_rel_l2_best_sign|{_safe_float(sym.get('weight_rel_l2_best_sign')):.6f}|")
            lines.append(f"|weight_row_cos_mean_raw|{_safe_float(sym.get('weight_row_cos_mean_raw')):.6f}|")
            lines.append(f"|weight_row_cos_mean_best_sign|{_safe_float(sym.get('weight_row_cos_mean_best_sign')):.6f}|")
            lines.append(f"|bias_norm_ratio_r_over_l|{_safe_float(sym.get('bias_norm_ratio_r_over_l')):.6f}|")
            lines.append(f"|bias_rel_l2_raw|{_safe_float(sym.get('bias_rel_l2_raw')):.6f}|")
            lines.append(f"|bias_rel_l2_best_sign|{_safe_float(sym.get('bias_rel_l2_best_sign')):.6f}|")

    lines.append("")
    lines.append("## SIC asymmetry (top |log ratio|)")
    lines.append("")
    sic_valid = [r for r in per_sic if math.isfinite(_safe_float(r.get("grad_log_ratio_r_over_l")))]
    sic_valid = sorted(sic_valid, key=lambda x: abs(float(x["grad_log_ratio_r_over_l"])), reverse=True)
    lines.append("|sic|left_n|right_n|left_grad|right_grad|ratio_r_over_l|log_ratio|")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|")
    for r in sic_valid[:15]:
        lines.append(
            f"|{int(r['sic'])}|{int(r[f'{l_name}_n'])}|{int(r[f'{r_name}_n'])}|"
            f"{_safe_float(r[f'{l_name}_grad_mean']):.6f}|{_safe_float(r[f'{r_name}_grad_mean']):.6f}|"
            f"{_safe_float(r['grad_ratio_r_over_l']):.6f}|{_safe_float(r['grad_log_ratio_r_over_l']):.6f}|"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- {payload['notes']['sampling_bias_definition']}")
    lines.append(f"- {payload['notes']['gradient_definition']}")
    lines.append(f"- {payload['notes']['component_gradient_definition']}")
    lines.append(f"- {payload['notes']['component_side_loss_definition']}")
    lines.append(f"- {payload['notes']['dual_probe_definition']}")
    lines.append(f"- {payload['notes']['dynamics_probe_definition']}")
    lines.append(f"- {payload['notes']['cond_side_definition']}")
    lines.append(f"- {payload['notes']['augmentation_caveat']}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
