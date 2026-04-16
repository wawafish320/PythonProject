from __future__ import annotations

import math as _math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .geometry import (
    _root_relative_matrices,
    angvel_vec_from_R_seq,
    geodesic_R,
    reproject_rot6d,
    rot6d_to_matrix,
    wrap_to_pi_torch,
)
from .io import speed_from_X_layout as _speed_from_X_layout
from .models import DEFAULT_DIRECT_POSE_LEG_BONES, STAGE6_3WAY_ARMCHAIN_BONES
from .utils import grad_norm_of_module, safe_int_scalar, warn_once


_DIAG_WARN_ONCE_KEYS: set[str] = set()


def _diag_warn_once(key: str, message: str, exc: Optional[BaseException] = None) -> None:
    warn_once(
        _DIAG_WARN_ONCE_KEYS,
        category="Diagnostics",
        key=key,
        message=message,
        exc=exc,
    )


def _maybe_optimize_dataset_index(ds, args):
    """
    Rebuild ds.index using a stride strategy (optionally filtered by root speed).
    Safe to call for both train/val.
    """
    try:
        stride = max(1, int(getattr(args, 'index_stride', 1)))
        min_speed = float(getattr(args, 'min_speed', 0.0))
    except Exception:
        stride, min_speed = (1, 0.0)
    new_index = []
    for cid, clip in enumerate(ds.clips):
        X = clip.X
        T = int(X.shape[0])
        L = ds.seq_len
        starts = list(range(0, max(0, T - L + 1), stride))
        if min_speed > 0.0:
            spd = _speed_from_X_layout(X, clip.state_layout_norm)
            if spd is not None and spd.shape[0] >= T:
                keep = []
                for s in starts:
                    e = s + L
                    seg_spd = spd[s:e]
                    if seg_spd.mean() >= min_speed or seg_spd.max() >= min_speed * 0.8:
                        keep.append(s)
                starts = keep if keep else starts
        for s in starts:
            new_index.append((cid, int(s)))
    if new_index:
        ds.index = new_index
        print(f'[IndexOpt] stride={stride} min_speed={min_speed} -> windows={len(ds.index)}')
    else:
        print('[IndexOpt] No windows built; keep original.')
    return ds




# === [ARPG PATCH] One-shot normalization diagnostics (safe & self-contained) ===
def _norm_debug_once(trainer, train_loader, thr=8.0, topk=8, print_to_console=True, writer=None, tag_prefix="NormDiag"):
    import numpy as np
    try:
        import torch  # noqa: F401
    except Exception:
        torch = None

    def _to_np(x):
        if x is None:
            return None
        try:
            import torch as _t
            if isinstance(x, _t.Tensor):
                return x.detach().cpu().float().numpy()
        except Exception as _err:
            print(f"[Norm-ERR] failed to convert tensor to numpy during diagnostics: {_err}")
            pass
        try:
            return np.asarray(x)
        except Exception:
            return None

    try:
        batch = next(iter(train_loader))
    except Exception as e:
        if print_to_console:
            print("[NormDiag] cannot fetch batch:", e)
        return

    xz = yz = None
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 1: xz = _to_np(batch[0])
        if len(batch) >= 2: yz = _to_np(batch[1])
    elif isinstance(batch, dict):
        vals = [v for v in batch.values() if _to_np(v) is not None]
        if len(vals) >= 1: xz = _to_np(vals[0])
        if len(vals) >= 2: yz = _to_np(vals[1])
    else:
        xz = _to_np(batch)

    mu_x = _to_np(getattr(trainer, "mu_x", None))
    se_x = _to_np(getattr(trainer, "std_x", None))
    mu_y = _to_np(getattr(trainer, "mu_y", None))
    se_y = _to_np(getattr(trainer, "std_y", None))

    yaw_x     = getattr(trainer, "yaw_x_slice", None)
    rootvel_x = getattr(trainer, "rootvel_x_slice", None)
    angvel_x  = getattr(trainer, "angvel_x_slice", None)
    yaw_y     = getattr(trainer, "yaw_slice", None)
    rootvel_y = getattr(trainer, "rootvel_slice", None)
    angvel_y  = getattr(trainer, "angvel_slice", None)

    def _slice_from_layout(layout, key):
        v = None if layout is None else layout.get(key)
        if v is None: return None
        try:
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                s, l = int(v[0]), int(v[1])
                return slice(s, s + l)
        except Exception:
            return None
        return None

    rot6d_x = _slice_from_layout(getattr(trainer, "_x_layout", None), "BoneRotations6D")
    rot6d_y = _slice_from_layout(getattr(trainer, "_y_layout", None), "BoneRotations6D")

    def _sel(z, sl):
        if z is None or sl is None: return None
        try:
            return z[..., sl]
        except Exception:
            return None

    def _to_2d_last(z):
        if z is None: return None
        z = _to_np(z)
        if z is None: return None
        if z.ndim == 0: return None
        if z.ndim == 1: return z[None, :]
        if z.ndim >= 2: return z.reshape(-1, z.shape[-1])
        return None

    def _z_stats(z, name, thr):
        z2 = _to_2d_last(z)
        if z2 is None or z2.size == 0: return None
        absz = np.abs(z2)
        pct = float((absz > thr).mean() * 100.0)
        return dict(name=name,
                    absmax=float(np.nanmax(absz)),
                    p99=float(np.nanpercentile(absz, 99.0)),
                    p999=float(np.nanpercentile(absz, 99.9)),
                    pct_over_thr=pct,
                    mean=float(np.nanmean(z2)),
                    std=float(np.nanstd(z2)))

    def _raw_stats(z, mu, se, name):
        z2 = _to_2d_last(z)
        if z2 is None or mu is None or se is None: return None
        mu = _to_np(mu); se = _to_np(se)
        if mu is None or se is None: return None
        if z2.shape[-1] != mu.shape[-1] or mu.shape[-1] != se.shape[-1]:
            return None
        raw = z2 * se + mu
        return dict(name=name,
                    min=float(np.nanmin(raw)),
                    p1=float(np.nanpercentile(raw, 1.0)),
                    p50=float(np.nanpercentile(raw, 50.0)),
                    p99=float(np.nanpercentile(raw, 99.0)),
                    max=float(np.nanmax(raw)))

    def _roundtrip(z, mu, se):
        z2 = _to_2d_last(z); mu = _to_np(mu); se = _to_np(se)
        if z2 is None or mu is None or se is None: return None
        if z2.shape[-1] != mu.shape[-1] or mu.shape[-1] != se.shape[-1]: return None
        raw = z2 * se + mu
        z3 = (raw - mu) / se
        return float(np.nanmax(np.abs(z3 - z2)))

    if print_to_console:
        def _sh(x):
            try: return tuple(np.asarray(x).shape)
            except Exception: return None
        print("[NormDiag] shapes: Xz", _sh(xz), "Yz", _sh(yz),
              "muX", None if mu_x is None else len(mu_x),
              "seX", None if se_x is None else len(se_x),
              "muY", None if mu_y is None else len(mu_y),
              "seY", None if se_y is None else len(se_y))

    rows_z = [
        _z_stats(_sel(xz, yaw_x),     "X.yaw(z)", thr),
        _z_stats(_sel(xz, rootvel_x), "X.rootvel(z)", thr),
        _z_stats(_sel(xz, angvel_x),  "X.angvel(z)", thr),
        _z_stats(_sel(xz, rot6d_x),   "X.rot6d(z)", thr),
        _z_stats(_sel(yz, yaw_y),     "Y.yaw(z)", thr),
        _z_stats(_sel(yz, rootvel_y), "Y.rootvel(z)", thr),
        _z_stats(_sel(yz, angvel_y),  "Y.angvel(z)", thr),
        _z_stats(_sel(yz, rot6d_y),   "Y.rot6d(z)", thr),
    ]

    rows_raw = [
        _raw_stats(_sel(xz, yaw_x),     mu_x[yaw_x]     if (mu_x is not None and yaw_x     is not None) else None,
                                       se_x[yaw_x]     if (se_x is not None and yaw_x     is not None) else None, "X.yaw(raw)"),
        _raw_stats(_sel(xz, rootvel_x), mu_x[rootvel_x] if (mu_x is not None and rootvel_x is not None) else None,
                                       se_x[rootvel_x] if (se_x is not None and rootvel_x is not None) else None, "X.rootvel(raw)"),
        _raw_stats(_sel(xz, angvel_x),  mu_x[angvel_x]  if (mu_x is not None and angvel_x  is not None) else None,
                                       se_x[angvel_x]  if (se_x is not None and angvel_x  is not None) else None, "X.angvel(raw)"),
        _raw_stats(_sel(xz, rot6d_x),   mu_x[rot6d_x]   if (mu_x is not None and rot6d_x   is not None) else None,
                                       se_x[rot6d_x]   if (se_x is not None and rot6d_x   is not None) else None, "X.rot6d(raw)"),
        _raw_stats(_sel(yz, rot6d_y),   mu_y[rot6d_y]   if (mu_y is not None and rot6d_y   is not None) else None,
                                       se_y[rot6d_y]   if (se_y is not None and rot6d_y   is not None) else None, "Y.rot6d(raw)"),
    ]

    err_x = _roundtrip(xz, mu_x, se_x)
    err_y = _roundtrip(yz, mu_y, se_y)

    top_dims = None
    X2 = _to_2d_last(xz)
    if X2 is not None and X2.size:
        Z = np.abs(X2)
        p = np.nanpercentile(Z, 99.0, axis=0)
        order = np.argsort(-p)
        k = min(int(topk), p.shape[0])
        idx = order[:k]
        top_dims = [(int(i), float(p[i])) for i in idx]

    if print_to_console:
        print(f"[NormDiag] roundtrip_err: X={err_x}  Y={err_y}  thr={thr}")
        for r in rows_z:
            if r is not None:
                print("[NormDiag] {name}: absmax={absmax:.3f} p99={p99:.3f} p999={p999:.3f} "
                      "pct>|thr|={pct_over_thr:.2f}% mean={mean:.3f} std={std:.3f}".format(**r))
        for r in rows_raw:
            if r is not None:
                print("[NormDiag] {name}: min={min:.4f} p1={p1:.4f} p50={p50:.4f} p99={p99:.4f} max={max:.4f}".format(**r))
        if top_dims is not None:
            print(f"[NormDiag] X top-{topk} dims by p99(|z|):", top_dims)

    if writer is not None and hasattr(writer, "add_histogram"):
        try:
            import torch as _t
            def _h(arr, tag):
                if arr is None: return
                writer.add_histogram(f"{tag_prefix}/{tag}", _t.as_tensor(arr), 0)
            _h(_sel(xz, yaw_x), "X_yaw")
            _h(_sel(xz, rootvel_x), "X_rootvel")
            _h(_sel(xz, angvel_x), "X_angvel")
            _h(_sel(xz, rot6d_x), "X_rot6d")
            _h(_sel(yz, rot6d_y), "Y_rot6d")
        except Exception as e:
            if print_to_console:
                print("[NormDiag] histogram failed:", e)

def _parse_stage_schedule(spec: Optional[Any]):
    """Parse stage schedule definitions from CLI strings or structured JSON."""

    def _coerce_value(key: str, val: Any) -> Any:
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, bool) or val is None:
            return val
        if isinstance(val, str):
            txt = val.strip()
            if not txt:
                return txt
            lowered = txt.lower()
            if lowered in ('true', 'false'):
                return lowered == 'true'
            if lowered == 'none':
                return None
            try:
                if key.endswith(('steps', 'horizon', 'epoch', 'epochs')):
                    return int(float(txt))
                return float(txt)
            except ValueError:
                try:
                    return int(txt)
                except ValueError:
                    return txt
        return val

    def _normalize_goal(goal_entry: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(goal_entry, Mapping):
            return None
        metrics_cfg = goal_entry.get('metrics')
        if not isinstance(metrics_cfg, Mapping):
            return None
        normalized_metrics: Dict[str, Dict[str, Any]] = {}
        for name, cfg in metrics_cfg.items():
            if not isinstance(cfg, Mapping):
                continue
            metric = {
                'ref': float(cfg.get('ref', 0.0) or 0.0),
            }
            if 'hi' in cfg:
                metric['hi'] = float(cfg['hi'])
            if 'lo' in cfg:
                metric['lo'] = float(cfg['lo'])
            if 'hi_ratio' in cfg:
                metric['hi_ratio'] = float(cfg['hi_ratio'])
            if 'lo_ratio' in cfg:
                metric['lo_ratio'] = float(cfg['lo_ratio'])
            metric['mode'] = cfg.get('mode')
            normalized_metrics[str(name)] = metric
        if not normalized_metrics:
            return None
        tags = goal_entry.get('tags') or goal_entry.get('tag') or ['valfree']
        if isinstance(tags, str):
            tags = [tags]
        elif isinstance(tags, Sequence):
            tags = [str(t) for t in tags]
        else:
            tags = ['valfree']
        window = int(goal_entry.get('window', 3) or 3)
        min_epochs = int(goal_entry.get('min_epochs', 0) or 0)
        return {
            'metrics': normalized_metrics,
            'tags': tags,
            'window': max(1, window),
            'min_epochs': max(0, min_epochs),
        }

    def _append_stage(stages: list, start: int, end: int, params: Dict[str, Any], label: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        if start is None or end is None:
            return
        stage = {'start': int(start), 'end': int(end), 'params': dict(params)}
        if label:
            stage['label'] = str(label)
        if extra:
            for key, value in extra.items():
                if value is not None:
                    stage[key] = value
        stages.append(stage)

    def _parse_string(spec_str: str):
        out = []
        for entry in spec_str.split(';'):
            chunk = entry.strip()
            if not chunk or ':' not in chunk:
                continue
            range_part, params_part = chunk.split(':', 1)
            label = None
            if '@' in range_part:
                range_part, label = [seg.strip() for seg in range_part.split('@', 1)]
            range_part = range_part.strip()
            if '-' in range_part:
                start_s, end_s = range_part.split('-', 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
            else:
                start = end = int(range_part.strip())
            params = {}
            for token in params_part.split(','):
                token = token.strip()
                if not token or '=' not in token:
                    continue
                key, val = token.split('=', 1)
                key = key.strip()
                val = val.strip()
                params[key] = _coerce_value(key, val)
            _append_stage(out, start, end, params, label)
        return out

    def _normalize_range(entry: Mapping[str, Any]):
        start = entry.get('start')
        end = entry.get('end')
        if start is None and end is None:
            rng = entry.get('range') or entry.get('epochs')
            if isinstance(rng, str):
                part = rng.strip()
                if '-' in part:
                    s, e = part.split('-', 1)
                    return int(s.strip()), int(e.strip())
                return int(part), int(part)
            if isinstance(rng, Sequence) and rng:
                if len(rng) == 1:
                    val = int(rng[0])
                    return val, val
                return int(rng[0]), int(rng[-1])
        if start is None and end is not None:
            start = end
        if end is None and start is not None:
            end = start
        if start is None:
            return None, None
        return int(start), int(end)

    def _merge_params(entry: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        params: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        if not isinstance(entry, Mapping):
            return params, extras
        base = entry.get('params') if isinstance(entry.get('params'), Mapping) else {}
        for key, val in base.items():
            params[key] = val

        def _ingest(source: Optional[Mapping[str, Any]], prefix: Optional[str] = None):
            if not isinstance(source, Mapping):
                return
            for k, v in source.items():
                name = f"{prefix}.{k}" if prefix else k
                params[name] = v

        _ingest(entry.get('trainer'))
        _ingest(entry.get('loss'), prefix='loss')
        tf_cfg = entry.get('tf')
        if isinstance(tf_cfg, Mapping):
            if 'max' in tf_cfg:
                params['tf_max'] = tf_cfg['max']
            if 'min' in tf_cfg:
                params['tf_min'] = tf_cfg['min']

        loss_groups_cfg = entry.get('loss_groups')
        normalized_groups = {}
        if isinstance(loss_groups_cfg, Mapping):
            for group_name, group_vals in loss_groups_cfg.items():
                if not isinstance(group_vals, Mapping):
                    continue
                group_norm = {}
                for key, val in group_vals.items():
                    group_norm[key] = val
                    params[f'loss.{key}'] = val
                if group_norm:
                    normalized_groups[str(group_name)] = group_norm
        if normalized_groups:
            extras['loss_groups'] = normalized_groups

        reserved = {'start', 'end', 'range', 'epochs', 'params', 'trainer', 'loss', 'tf', 'label', 'name', 'updates', 'loss_groups', 'goal'}
        for key, val in entry.items():
            if key in reserved:
                continue
            params[key] = val

        updates = entry.get('updates')
        if isinstance(updates, Sequence) and not isinstance(updates, (str, bytes)):
            for item in updates:
                if isinstance(item, Mapping):
                    target = item.get('key') or item.get('name') or item.get('param')
                    value = item.get('value')
                    if target:
                        params[target] = value

        goal_norm = _normalize_goal(entry.get('goal'))
        if goal_norm:
            extras['goal'] = goal_norm

        coerced = {k: _coerce_value(k, v) for k, v in params.items()}
        return coerced, extras

    if not spec:
        return []
    if isinstance(spec, str):
        return _parse_string(spec)
    stages: list = []
    entries: Sequence[Any]
    if isinstance(spec, Mapping):
        entries = [spec]
    elif isinstance(spec, Sequence):
        entries = list(spec)
    else:
        return []
    for entry in entries:
        if isinstance(entry, str):
            stages.extend(_parse_string(entry))
            continue
        if not isinstance(entry, Mapping):
            continue
        start, end = _normalize_range(entry)
        label = entry.get('label') or entry.get('name')
        params, extra = _merge_params(entry)
        _append_stage(stages, start, end, params, label, extra)
    for idx, stage in enumerate(stages):
        stage['index'] = idx
    return stages


def _record_optional_diag_curve(
    result: Dict[str, Any],
    *,
    metric_name: str,
    curve: Any,
    curve_max: Optional[Any] = None,
    curve_bones: Optional[Mapping[str, Sequence[float]]] = None,
    scope_alias: Optional[str] = None,
) -> None:
    def _curve_payload(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    curve_key = f"{metric_name}Curve"
    result[curve_key] = _curve_payload(curve)
    curve_max_key = None
    if curve_max is not None:
        curve_max_key = f"{metric_name}CurveMax"
        result[curve_max_key] = _curve_payload(curve_max)
    curve_bones_key = None
    if curve_bones is not None:
        curve_bones_key = f"{metric_name}CurveBones"
        result[curve_bones_key] = curve_bones
    if scope_alias:
        result[f"{scope_alias}/{curve_key}"] = result[curve_key]
        if curve_max_key is not None:
            result[f"{scope_alias}/{curve_max_key}"] = result[curve_max_key]
        if curve_bones_key is not None:
            result[f"{scope_alias}/{curve_bones_key}"] = result[curve_bones_key]


@dataclass(frozen=True)
class FreeRunDiagSequences:
    predX_tensor: Optional[torch.Tensor]
    predX_raw: Optional[torch.Tensor]
    gtX_raw: Optional[torch.Tensor]
    gtX_raw_full: Optional[torch.Tensor]
    cond_raw_seq: Optional[torch.Tensor]
    model: Any


@dataclass
class FreeRunDiagKinematics:
    geo: Optional[torch.Tensor] = None
    geo_local: Optional[torch.Tensor] = None
    w_pred: Optional[torch.Tensor] = None
    w_gt: Optional[torch.Tensor] = None


def _record_diag_metric(
    result: Dict[str, Any],
    diag_scope: str,
    name: str,
    value: Any,
    *,
    extra_scope_aliases: Sequence[str] = (),
) -> None:
    result[name] = value
    scope_aliases: list[str] = []
    if diag_scope == "free_run":
        scope_aliases.append("FreeRun")
    elif diag_scope == "single_step":
        scope_aliases.append("SingleStep")
    for alias in extra_scope_aliases:
        if alias not in scope_aliases:
            scope_aliases.append(alias)
    for alias in scope_aliases:
        result[f"{alias}/{name}"] = value


def aggregate_metric_samples(
    stats_accum: Mapping[str, Sequence[Any]],
    *,
    defaults: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    def _avg_dict_recursive(dict_list: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        nested: Dict[str, list[Mapping[str, Any]]] = {}
        for item in dict_list:
            if not isinstance(item, Mapping):
                continue
            for key, value in item.items():
                if isinstance(value, Mapping):
                    nested.setdefault(str(key), []).append(value)
                elif isinstance(value, (int, float)):
                    totals[str(key)] = totals.get(str(key), 0.0) + float(value)
                    counts[str(key)] = counts.get(str(key), 0) + 1
        result = {key: (totals[key] / max(1, counts[key])) for key in totals}
        for key, values in nested.items():
            sub = _avg_dict_recursive(values)
            if sub:
                result[key] = sub
        return result

    summary: Dict[str, Any] = {}
    for key, values in stats_accum.items():
        if not values:
            continue
        sample = values[0]
        if isinstance(sample, Mapping):
            summary[str(key)] = _avg_dict_recursive(values)
            continue
        try:
            summary[str(key)] = float(sum(values) / max(1, len(values)))
        except Exception:
            summary[str(key)] = list(values)

    if defaults:
        for key, val in defaults.items():
            if isinstance(val, Mapping):
                summary.setdefault(str(key), dict(val))
            else:
                summary.setdefault(str(key), val)
    return summary


def collect_direct_pose_grad_stats(trainer) -> Dict[str, float]:
    model = getattr(trainer, "model", None)
    if model is None:
        return {}
    g_trunk = grad_norm_of_module(getattr(model, "direct_pose_head", None))
    g_leg = grad_norm_of_module(getattr(model, "direct_pose_out_leg", None))
    g_nonleg_head = grad_norm_of_module(getattr(model, "direct_pose_out_nonleg", None))
    g_arm = grad_norm_of_module(getattr(model, "direct_pose_out_arm", None))
    g_else = grad_norm_of_module(getattr(model, "direct_pose_out_else", None))

    def _merge_grad_norm(*vals: float) -> float:
        finite = [float(v) for v in vals if isinstance(v, (int, float)) and _math.isfinite(float(v))]
        if not finite:
            return float("nan")
        return float(_math.sqrt(sum(v * v for v in finite)))

    g_nonleg = _merge_grad_norm(g_nonleg_head, g_arm, g_else)
    ratio_nonleg_leg = float("nan")
    if _math.isfinite(g_leg) and _math.isfinite(g_nonleg):
        ratio_nonleg_leg = float(g_nonleg / max(1e-12, g_leg))
    ratio_arm_else = float("nan")
    if _math.isfinite(g_arm) and _math.isfinite(g_else):
        ratio_arm_else = float(g_arm / max(1e-12, g_else))
    stats = {
        "direct_grad_norm_trunk": float(g_trunk),
        "direct_grad_norm_out_leg": float(g_leg),
        "direct_grad_norm_out_nonleg": float(g_nonleg),
        "direct_grad_norm_out_arm": float(g_arm),
        "direct_grad_norm_out_else": float(g_else),
        "direct_grad_ratio_nonleg_over_leg": float(ratio_nonleg_leg),
        "direct_grad_ratio_arm_over_else": float(ratio_arm_else),
    }
    gate_thr = float(getattr(trainer, "direct_pose_grad_ratio_gate", 0.35) or 0.35)
    if _math.isfinite(ratio_nonleg_leg) and _math.isfinite(gate_thr) and gate_thr > 0.0:
        stats["direct_grad_ratio_gate"] = float(gate_thr)
        stats["direct_grad_ratio_alert"] = 1.0 if ratio_nonleg_leg < gate_thr else 0.0
    return stats


def _collect_diag_sequences(trainer, *, predsX, motion_seq, batch) -> FreeRunDiagSequences:
    predX_tensor = torch.stack(predsX, dim=1) if predsX else None
    model = getattr(trainer, "model", None)
    gtX_raw_full = None
    if motion_seq is not None:
        try:
            flat_motion = motion_seq.reshape(-1, motion_seq.shape[-1])
            gtX_raw_full = trainer.normalizer.denorm_x(flat_motion).view_as(motion_seq)
        except Exception as exc:
            trainer._raise_norm_error("normalizer.denorm_x 在诊断阶段还原 GT X 时失败", exc)

    if predX_tensor is not None:
        flat_pred = predX_tensor.reshape(-1, predX_tensor.shape[-1])
        try:
            predX_raw = trainer.normalizer.denorm_x(flat_pred).view_as(predX_tensor)
        except Exception as exc:
            trainer._raise_norm_error("normalizer.denorm_x 在诊断阶段还原预测 X 时失败", exc)
        if motion_seq is not None:
            if gtX_raw_full is None:
                trainer._raise_norm_error("诊断阶段缺少 GT RAW 序列。")
            gtX_raw = gtX_raw_full[:, :predX_tensor.shape[1]]
        else:
            gtX_raw = None
    else:
        predX_raw = None
        gtX_raw = None

    cond_raw_seq = None
    if isinstance(batch, dict):
        cond_raw_seq = batch.get("cond_tgt_raw")
        if cond_raw_seq is None:
            cond_raw_seq = batch.get("cond_in")

    return FreeRunDiagSequences(
        predX_tensor=predX_tensor,
        predX_raw=predX_raw,
        gtX_raw=gtX_raw,
        gtX_raw_full=gtX_raw_full,
        cond_raw_seq=cond_raw_seq,
        model=model,
    )


def _compute_input_drift_metrics(
    trainer,
    result: Dict[str, Any],
    cfg: SimpleNamespace,
    seqs: FreeRunDiagSequences,
    *,
    period_seq_pred,
) -> None:
    predX_raw = seqs.predX_raw
    gtX_raw = seqs.gtX_raw

    if isinstance(cfg.rv_x, slice) and predX_raw is not None and gtX_raw is not None:
        _record_diag_metric(
            result,
            cfg.diag_scope,
            "RootVelMAE",
            float((predX_raw[..., cfg.rv_x] - gtX_raw[..., cfg.rv_x]).abs().mean().item()),
        )
        if cfg.diag_input_stats:
            diff = (predX_raw[..., cfg.rv_x] - gtX_raw[..., cfg.rv_x]).abs()
            result["RootVelMAE_std"] = float(diff.std().item())
        if predX_raw.shape[1] > 0 and gtX_raw.shape[1] > 0:
            rv_end = (predX_raw[:, -1, cfg.rv_x] - gtX_raw[:, -1, cfg.rv_x]).abs().mean()
            _record_diag_metric(result, cfg.diag_scope, "RootVelEndMAE", float(rv_end.item()))

    cond_raw_seq = seqs.cond_raw_seq
    if torch.is_tensor(cond_raw_seq):
        cond_raw_seq = cond_raw_seq.float()
        if cond_raw_seq.dim() == 2:
            cond_raw_seq = cond_raw_seq.unsqueeze(0)
        if cond_raw_seq.dim() == 3 and predX_raw is not None:
            batch_size = predX_raw.shape[0]
            if cond_raw_seq.shape[0] == batch_size:
                start_idx = 1
                horizon = predX_raw.shape[1]
                if cond_raw_seq.shape[1] >= start_idx + horizon:
                    cond_slice = cond_raw_seq[:, start_idx:start_idx + horizon]
                else:
                    cond_slice = cond_raw_seq[:, -horizon:]
                cond_dim = cond_slice.shape[-1]
                if cond_dim >= 2:
                    if cond_dim >= 3:
                        dir_slice = cond_slice[..., cond_dim - 3:cond_dim - 1]
                        speed_slice = cond_slice[..., -1]
                    else:
                        dir_slice = cond_slice[..., -2:]
                        speed_slice = dir_slice.norm(dim=-1)
                    length = min(cond_slice.shape[1], predX_raw.shape[1])
                    if length > 0:
                        device = predX_raw.device
                        dir_slice = dir_slice[:, :length].to(device)
                        speed_slice = speed_slice[:, :length].to(device)
                        dir_norm = dir_slice.norm(dim=-1).clamp_min(1e-6)
                        dir_unit = dir_slice / dir_norm.unsqueeze(-1)
                        if isinstance(cfg.rv_x, slice):
                            cond_vel = dir_unit * speed_slice.unsqueeze(-1)
                            vel_pred = predX_raw[:, :length, cfg.rv_x]
                            _record_diag_metric(
                                result,
                                cfg.diag_scope,
                                "CondVelVsPredMAE",
                                float((vel_pred - cond_vel).abs().mean().item()),
                                extra_scope_aliases=("FreeRun",),
                            )
                            if gtX_raw is not None and gtX_raw.shape[1] >= start_idx + length:
                                vel_gt = gtX_raw[:, start_idx:start_idx + length, cfg.rv_x]
                                _record_diag_metric(
                                    result,
                                    cfg.diag_scope,
                                    "CondVelVsGTMAE",
                                    float((vel_gt - cond_vel).abs().mean().item()),
                                    extra_scope_aliases=("FreeRun",),
                                )

    if period_seq_pred:
        try:
            norm_period = []
            for period in period_seq_pred:
                if period.dim() == 3 and period.size(1) == 1:
                    norm_period.append(period.squeeze(1))
                elif period.dim() == 2:
                    norm_period.append(period)
                else:
                    norm_period.append(period.reshape(period.shape[0], -1))
            if norm_period:
                period_tensor = torch.stack(norm_period, dim=1)
                result["period_abs_mean"] = float(period_tensor.abs().mean().item())
                result["period_abs_std"] = float(period_tensor.abs().std().item())
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            _diag_warn_once("diag/period_abs_stats", "failed to compute period_abs_mean/std diagnostics", exc)


def _joint_group_masks(device: torch.device, joint_count: int, bone_names: Optional[Sequence[str]] = None):
    masks = {}
    if bone_names:
        torso_idx = []
        prox_idx = []
        dist_idx = []
        for idx, name in enumerate(bone_names):
            lname = str(name).lower()
            if any(key in lname for key in ("spine", "pelvis", "root", "torso", "chest", "neck")):
                torso_idx.append(idx)
            elif any(key in lname for key in ("upperarm", "thigh", "clavicle", "shoulder", "hip")):
                prox_idx.append(idx)
            else:
                dist_idx.append(idx)
    else:
        torso_count = min(5, joint_count)
        prox_count = min(5, max(0, joint_count - torso_count))
        torso_idx = list(range(torso_count))
        prox_idx = list(range(torso_count, torso_count + prox_count))
        dist_idx = list(range(torso_count + prox_count, joint_count))

    def _mask(indices):
        mask = torch.zeros(joint_count, dtype=torch.bool, device=device)
        if indices:
            valid = [idx for idx in indices if 0 <= idx < joint_count]
            if valid:
                mask[valid] = True
        return mask

    masks["torso"] = _mask(torso_idx)
    masks["proximal"] = _mask(prox_idx)
    masks["distal"] = _mask(dist_idx)
    return masks


def _summarize_angvel_dir(
    pred_w: Optional[torch.Tensor],
    gt_w: Optional[torch.Tensor],
    *,
    bone_names: Optional[Sequence[str]] = None,
    magnitude_threshold: float = 0.1,
    smooth_window: int = 3,
) -> dict[str, float]:
    if pred_w is None or gt_w is None or pred_w.numel() == 0 or gt_w.numel() == 0:
        return {}
    batch_size, steps, joint_count, _ = pred_w.shape
    eps = 1e-6
    dot = (pred_w * gt_w).sum(dim=-1)
    norm = pred_w.norm(dim=-1) * gt_w.norm(dim=-1)
    cos = torch.clamp(dot / (norm + eps), -1.0 + 1e-7, 1.0 - 1e-7)
    angle_deg = torch.acos(cos) * (180.0 / _math.pi)
    raw = float(angle_deg.mean().item())
    magnitude = gt_w.norm(dim=-1)
    weight = (magnitude > magnitude_threshold).float()
    weighted = float((angle_deg * weight).sum().item() / (weight.sum().item() + eps))
    smooth = weighted
    if smooth_window >= 3 and steps >= smooth_window:
        pad = smooth_window // 2
        pred_flat = pred_w.reshape(batch_size, steps, joint_count * 3).transpose(1, 2)
        gt_flat = gt_w.reshape(batch_size, steps, joint_count * 3).transpose(1, 2)
        pred_s = F.avg_pool1d(pred_flat, kernel_size=smooth_window, stride=1, padding=pad).transpose(1, 2).reshape(batch_size, steps, joint_count, 3)
        gt_s = F.avg_pool1d(gt_flat, kernel_size=smooth_window, stride=1, padding=pad).transpose(1, 2).reshape(batch_size, steps, joint_count, 3)
        dot_s = (pred_s * gt_s).sum(dim=-1)
        norm_s = pred_s.norm(dim=-1) * gt_s.norm(dim=-1)
        cos_s = torch.clamp(dot_s / (norm_s + eps), -1.0 + 1e-7, 1.0 - 1e-7)
        angle_s = torch.acos(cos_s) * (180.0 / _math.pi)
        smooth = float((angle_s * weight).sum().item() / (weight.sum().item() + eps))
    masks = _joint_group_masks(pred_w.device, joint_count, bone_names)
    group_vals = {}
    for key, mask in masks.items():
        if mask.any():
            mask_f = mask.view(1, 1, joint_count)
            grp_weight = weight * mask_f
            denom = grp_weight.sum().item()
            group_vals[key] = float((angle_deg * grp_weight).sum().item() / (denom + eps)) if denom > 0 else float("nan")
        else:
            group_vals[key] = float("nan")
    return {
        "raw": raw,
        "weighted": weighted,
        "smooth": smooth,
        "torso": group_vals.get("torso", float("nan")),
        "proximal": group_vals.get("proximal", float("nan")),
        "distal": group_vals.get("distal", float("nan")),
    }


def _compute_contact_and_angvel_metrics(
    trainer,
    result: Dict[str, Any],
    cfg: SimpleNamespace,
    seqs: FreeRunDiagSequences,
    *,
    predY,
    gtY,
    period_seq_pred,
    contacts_seq,
    angvel_seq,
    pose_hist_seq,
) -> FreeRunDiagKinematics:
    state = FreeRunDiagKinematics()
    if not isinstance(cfg.rot6d_y, slice):
        return state

    predY_raw = trainer._denorm(predY)
    gtY_raw = trainer._denorm(gtY)
    py = predY_raw[..., cfg.rot6d_y]
    gy = gtY_raw[..., cfg.rot6d_y]
    if py.shape[-1] % 6 != 0:
        return state

    joint_count = py.shape[-1] // 6
    py6 = reproject_rot6d(py).view(py.shape[0], py.shape[1], joint_count, 6)
    gy6 = reproject_rot6d(gy).view(gy.shape[0], gy.shape[1], joint_count, 6)
    pred_rot = rot6d_to_matrix(py6)
    gt_rot = rot6d_to_matrix(gy6)
    pred_rot_raw = pred_rot
    pred_rot_geo = pred_rot
    if cfg.eval_align_root and pred_rot.shape[1] > 0 and 0 <= cfg.root_idx < joint_count:
        pred_root0 = pred_rot[:, 0, cfg.root_idx]
        gt_root0 = gt_rot[:, 0, cfg.root_idx]
        align = gt_root0 @ pred_root0.transpose(-1, -2)
        pred_rot_geo = align.view(pred_rot.shape[0], 1, 1, 3, 3).expand_as(pred_rot) @ pred_rot

    state.geo = geodesic_R(pred_rot_geo, gt_rot)
    _record_diag_metric(
        result,
        cfg.diag_scope,
        "GeoDeg",
        float((state.geo.mean() * cfg.deg).item()),
        extra_scope_aliases=("SingleStep",),
    )
    try:
        geo_deg = state.geo * cfg.deg
        geo_curve = geo_deg.mean(dim=-1).mean(dim=0)
        geo_curve_max = geo_deg.max(dim=-1).values.max(dim=0).values
        _record_optional_diag_curve(result, metric_name="GeoDeg", curve=geo_curve, curve_max=geo_curve_max)
        _record_diag_metric(result, cfg.diag_scope, "GeoDegEnd", float(geo_curve[-1].item()))
        if cfg.bone_names:
            geo_per_bone = {}
            geo_mean_bone = geo_deg.mean(dim=0)
            for joint_idx, name in enumerate(cfg.bone_names[:geo_mean_bone.shape[1]]):
                geo_per_bone[name] = geo_mean_bone[:, joint_idx].detach().cpu().tolist()
            _record_optional_diag_curve(
                result,
                metric_name="GeoDeg",
                curve=result.get("GeoDegCurve", []),
                curve_max=result.get("GeoDegCurveMax"),
                curve_bones=geo_per_bone,
            )
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        _diag_warn_once("diag/geo_curve", "failed to compute GeoDeg curve diagnostics", exc)

    try:
        pred_root = _root_relative_matrices(pred_rot_raw, cfg.root_idx)
        gt_root = _root_relative_matrices(gt_rot, cfg.root_idx)
        _ = pred_root, gt_root
        state.geo_local = geodesic_R(pred_rot_raw, gt_rot) * cfg.deg
        joint_weights = trainer._joint_weights(pred_rot_geo, joint_count)
        if 0 <= cfg.root_idx < joint_weights.numel():
            joint_weights = joint_weights.clone()
            joint_weights[cfg.root_idx] = 0.0
        weights_sum = joint_weights.sum().clamp_min(1e-6)
        weight = joint_weights.view(1, 1, -1)
        geo_local_mean = (state.geo_local * weight).sum() / (
            weights_sum * state.geo_local.shape[0] * state.geo_local.shape[1]
        )
        _record_diag_metric(
            result,
            cfg.diag_scope,
            "GeoLocalDeg",
            float(geo_local_mean.item()),
            extra_scope_aliases=("SingleStep",),
        )
        step_vals = ((state.geo_local * weight).sum(dim=-1) / weights_sum).mean(dim=0)
        _record_optional_diag_curve(result, metric_name="GeoLocalDeg", curve=step_vals)
        drift_proxy = float((step_vals[-1] - step_vals[0]).detach().cpu() / max(1, int(step_vals.numel()) - 1)) if int(step_vals.numel()) >= 2 else float("nan")
        _record_diag_metric(result, cfg.diag_scope, "GeoDriftSlopeProxy", drift_proxy)
        geo_for_max = state.geo_local
        if 0 <= cfg.root_idx < geo_for_max.shape[-1]:
            geo_for_max = state.geo_local.clone()
            geo_for_max[..., cfg.root_idx] = -1e9
        max_vals = geo_for_max.max(dim=-1).values.max(dim=0).values
        _record_optional_diag_curve(
            result,
            metric_name="GeoLocalDeg",
            curve=result.get("GeoLocalDegCurve", []),
            curve_max=max_vals,
        )
        _record_diag_metric(result, cfg.diag_scope, "GeoLocalDegEnd", float(step_vals[-1].item()))
        if cfg.bone_names:
            geo_local_per_bone = {}
            geo_local_mean_bone = state.geo_local.mean(dim=0)
            if 0 <= cfg.root_idx < geo_local_mean_bone.shape[1]:
                geo_local_mean_bone[:, cfg.root_idx] = 0.0
            for joint_idx, name in enumerate(cfg.bone_names[:geo_local_mean_bone.shape[1]]):
                geo_local_per_bone[name] = geo_local_mean_bone[:, joint_idx].detach().cpu().tolist()
            _record_optional_diag_curve(
                result,
                metric_name="GeoLocalDeg",
                curve=result.get("GeoLocalDegCurve", []),
                curve_max=result.get("GeoLocalDegCurveMax"),
                curve_bones=geo_local_per_bone,
            )
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        _diag_warn_once("diag/geo_local", "failed to compute GeoLocalDeg diagnostics", exc)

    try:
        pred_parent = trainer._parent_relative_matrices(_root_relative_matrices(pred_rot_raw, cfg.root_idx))
        gt_parent = trainer._parent_relative_matrices(_root_relative_matrices(gt_rot, cfg.root_idx))
        state.w_pred = angvel_vec_from_R_seq(pred_parent, cfg.fps_eval)
        state.w_gt = angvel_vec_from_R_seq(gt_parent, cfg.fps_eval)
        _record_diag_metric(
            result,
            cfg.diag_scope,
            "AngVelMAE",
            float((state.w_pred - state.w_gt).abs().mean().item()),
        )
        mag_p = state.w_pred.norm(dim=-1)
        mag_g = state.w_gt.norm(dim=-1)
        mag_avg = 0.5 * (mag_p + mag_g)
        mask_mag = mag_avg > cfg.mag_rel_threshold
        mag_rel = (mag_p - mag_g).abs() / (mag_avg + cfg.mag_rel_beta)
        ang_mag_rel = (mag_rel * mask_mag).sum(dim=(0, 1)) / mask_mag.sum(dim=(0, 1)).clamp_min(1)
        _record_diag_metric(result, cfg.diag_scope, "AngVelMagRel", float(torch.nanmedian(ang_mag_rel).item()))

        ang_mae_full = (state.w_pred - state.w_gt).abs()
        ang_mae_curve = ang_mae_full.mean(dim=(0, 2))
        ang_mae_bone_curve = None
        if cfg.bone_names and ang_mae_full.shape[2] == len(cfg.bone_names):
            ang_mae_bones = ang_mae_full.mean(dim=0)
            ang_mae_bone_curve = {
                name: ang_mae_bones[:, joint_idx].norm(dim=-1).detach().cpu().tolist()
                for joint_idx, name in enumerate(cfg.bone_names)
            }
        _record_optional_diag_curve(
            result,
            metric_name="AngVelMAE",
            curve=ang_mae_curve,
            curve_bones=ang_mae_bone_curve,
            scope_alias="SingleStep" if cfg.diag_scope == "single_step" else None,
        )

        dot_full = (state.w_pred * state.w_gt).sum(dim=-1)
        ang_full = torch.zeros_like(dot_full)
        valid_full = (mag_p > cfg.angvel_dir_threshold) & (mag_g > cfg.angvel_dir_threshold)
        if valid_full.any():
            norm_full = (mag_p * mag_g).clamp_min(cfg.angvel_eps)
            cos = torch.clamp(dot_full / norm_full, -1.0 + 1e-6, 1.0 - 1e-6)
            ang_full[valid_full] = torch.acos(cos[valid_full])
        ang_full_deg = ang_full * cfg.deg
        ang_curve = None
        if valid_full.any():
            valid_f = valid_full.float()
            ang_sum = (ang_full_deg * valid_f).sum(dim=(0, 2))
            valid_cnt = valid_f.sum(dim=(0, 2)).clamp_min(1.0)
            ang_curve = ang_sum / valid_cnt
            ang_curve_max = ang_full_deg.max(dim=2).values.max(dim=0).values
            _record_optional_diag_curve(result, metric_name="AngVelDirDeg", curve=ang_curve, curve_max=ang_curve_max)
            result["AngVelDirDegValidRatio"] = float(valid_f.mean().item())
        else:
            _record_optional_diag_curve(result, metric_name="AngVelDirDeg", curve=[], curve_max=[])
            result["AngVelDirDegSkipped"] = True
        if ang_curve is not None and int(ang_curve.numel()) > 0:
            _record_diag_metric(result, cfg.diag_scope, "AngVelDirDegEnd", float(ang_curve[-1].item()))
        if cfg.diag_scope == "single_step":
            result["SingleStep/AngVelDirDegCurve"] = result["AngVelDirDegCurve"]
            result["SingleStep/AngVelDirDegCurveMax"] = result["AngVelDirDegCurveMax"]

        summary = _summarize_angvel_dir(state.w_pred, state.w_gt, bone_names=cfg.bone_names)
        if summary:
            _record_diag_metric(result, cfg.diag_scope, "AngVelDirDegRaw", summary.get("raw", float("nan")))
            _record_diag_metric(result, cfg.diag_scope, "AngVelDirDegWeighted", summary.get("weighted", float("nan")))
            _record_diag_metric(result, cfg.diag_scope, "AngVelDirDegSmooth", summary.get("smooth", float("nan")))
            _record_diag_metric(result, cfg.diag_scope, "AngVelDirDegTorso", summary.get("torso", float("nan")))
            _record_diag_metric(result, cfg.diag_scope, "AngVelDirDegProximal", summary.get("proximal", float("nan")))
            _record_diag_metric(result, cfg.diag_scope, "AngVelDirDegDistal", summary.get("distal", float("nan")))

        foot_names = ("foot_l", "foot_r")
        idx_map = {name: idx for idx, name in enumerate(cfg.bone_names)} if cfg.bone_names else {}

        def _masked_mean(val: torch.Tensor, mask: torch.Tensor):
            mask_f = mask.to(val.dtype)
            w_sum = mask_f.sum()
            if w_sum < 1e-6:
                return None
            return (val * mask_f).sum() / w_sum

        contacts_mask = None
        if torch.is_tensor(contacts_seq) and contacts_seq.dim() >= 3:
            contacts_mask = contacts_seq[:, : state.w_pred.shape[1]] > 0.5
        for foot_name in foot_names:
            joint_idx = idx_map.get(foot_name, None)
            if joint_idx is None or joint_idx >= state.w_pred.shape[2]:
                continue
            foot_idx = 0 if foot_name.endswith("_l") else 1
            w_p = state.w_pred[..., joint_idx, :]
            w_g = state.w_gt[..., joint_idx, :]
            mag_p = w_p.norm(dim=-1)
            mag_g = w_g.norm(dim=-1)
            stance_mask = swing_mask = None
            if contacts_mask is not None and foot_idx < contacts_mask.shape[-1]:
                stance_mask = contacts_mask[..., foot_idx]
                swing_mask = ~stance_mask
            if stance_mask is not None and stance_mask.any():
                mae_contact = _masked_mean((w_p - w_g).abs().norm(dim=-1), stance_mask)
                mag_contact = _masked_mean(mag_p, stance_mask)
                if mae_contact is not None:
                    _record_diag_metric(result, cfg.diag_scope, f"Foot/{foot_name}/ContactAngVelMAE", float(mae_contact.item()))
                if mag_contact is not None:
                    _record_diag_metric(result, cfg.diag_scope, f"Foot/{foot_name}/ContactAngVelMag", float(mag_contact.item()))
            dot = (w_p * w_g).sum(dim=-1)
            norm_prod = mag_p * mag_g
            ang = torch.zeros_like(dot)
            valid = norm_prod > 1e-6
            ang[valid] = torch.acos(torch.clamp(dot[valid] / norm_prod[valid], -1.0, 1.0)) * cfg.deg
            if stance_mask is not None and stance_mask.any():
                ang_stance = _masked_mean(ang, stance_mask)
                if ang_stance is not None:
                    _record_diag_metric(result, cfg.diag_scope, f"Foot/{foot_name}/AngVelDirDegStance", float(ang_stance.item()))
            if swing_mask is not None and swing_mask.any():
                ang_swing = _masked_mean(ang, swing_mask)
                if ang_swing is not None:
                    _record_diag_metric(result, cfg.diag_scope, f"Foot/{foot_name}/AngVelDirDegSwing", float(ang_swing.item()))
            if stance_mask is not None and stance_mask.any():
                mag_mae = _masked_mean((mag_p - mag_g).abs(), stance_mask)
                if mag_mae is not None:
                    _record_diag_metric(result, cfg.diag_scope, f"Foot/{foot_name}/AngVelMagMAEStance", float(mag_mae.item()))
            if swing_mask is not None and swing_mask.any():
                mag_mae_sw = _masked_mean((mag_p - mag_g).abs(), swing_mask)
                if mag_mae_sw is not None:
                    _record_diag_metric(result, cfg.diag_scope, f"Foot/{foot_name}/AngVelMagMAESwing", float(mag_mae_sw.item()))
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        _diag_warn_once("diag/angvel", "failed to compute angvel diagnostics", exc)

    period_pred = None
    if period_seq_pred:
        try:
            first_period = period_seq_pred[0]
            if isinstance(first_period, torch.Tensor):
                period_pred = torch.stack([p if p.dim() == 3 else p.unsqueeze(1) for p in period_seq_pred], dim=1)
                if period_pred.dim() == 4 and period_pred.size(2) == 1:
                    period_pred = period_pred.squeeze(2)
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            _diag_warn_once("diag/period_pred_pack", "failed to stack period predictions for diagnostics", exc)
            period_pred = None

    period_gt = None
    model = seqs.model
    if model is not None and getattr(model, "frozen_encoder", None) is not None and getattr(model, "frozen_period_head", None) is not None:
        try:
            enc_in_list = []
            for tensor in (contacts_seq, angvel_seq, pose_hist_seq):
                if torch.is_tensor(tensor):
                    enc_in_list.append(tensor)
            if enc_in_list:
                enc_input = torch.cat([tensor for tensor in enc_in_list if tensor is not None], dim=-1)
                enc_hidden = model.frozen_encoder(enc_input, return_summary=False)
                if isinstance(enc_hidden, tuple):
                    enc_hidden = enc_hidden[-1]
                period_gt = torch.tanh(model.frozen_period_head(enc_hidden))
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            _diag_warn_once("diag/period_gt_probe", "failed to compute period_gt embedding probe", exc)
            period_gt = None

    if period_pred is not None and period_gt is not None and period_pred.shape == period_gt.shape:
        try:
            diff = period_pred - period_gt
            embed_l2 = diff.norm(dim=-1).mean()
            _record_diag_metric(result, cfg.diag_scope, "Period/EmbedL2", float(embed_l2.item()))
            eps = 1e-6
            cos = ((period_pred * period_gt).sum(dim=-1)) / (
                period_pred.norm(dim=-1) * period_gt.norm(dim=-1) + eps
            )
            embed_cos = cos.clamp(-1.0, 1.0).mean()
            _record_diag_metric(result, cfg.diag_scope, "Period/EmbedCos", float(embed_cos.item()))
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            _diag_warn_once("diag/period_embed", "failed to compute period embedding diagnostics", exc)

    try:
        tgt = contacts_seq if torch.is_tensor(contacts_seq) else None
        if tgt is not None and tgt.shape[-1] >= 2:
            tgt = tgt[..., :2]
            ref = period_pred if period_pred is not None else (period_gt if period_gt is not None else tgt)
            tgt = tgt.to(ref.device).to(ref.dtype)
            tgt = tgt * 2.0 - 1.0
            if period_pred is not None and period_pred.shape[:2] == tgt.shape[:2] and period_pred.shape[-1] >= 2:
                pred_hint = period_pred[..., :2]
                _record_diag_metric(result, cfg.diag_scope, "Period/ContactHintMAE", float((pred_hint - tgt).abs().mean().item()))
            if period_gt is not None and period_gt.shape[:2] == tgt.shape[:2] and period_gt.shape[-1] >= 2:
                gt_hint = period_gt[..., :2]
                _record_diag_metric(result, cfg.diag_scope, "Period/ContactHintGTMAE", float((gt_hint - tgt).abs().mean().item()))
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        _diag_warn_once("diag/period_contact_hint", "failed to compute period/contact hint diagnostics", exc)

    result["Period/PhaseSkipped"] = True
    return state


def _compute_keybone_metrics(
    trainer,
    result: Dict[str, Any],
    cfg: SimpleNamespace,
    state: FreeRunDiagKinematics,
) -> None:
    if not cfg.bone_names:
        return

    if state.w_pred is not None and state.w_gt is not None:
        for joint_idx, name in enumerate(cfg.bone_names):
            if joint_idx >= state.w_pred.shape[2]:
                continue
            w_pred_b = state.w_pred[..., joint_idx, :]
            w_gt_b = state.w_gt[..., joint_idx, :]
            result[f"Bone/{name}/AngVelMagMAE"] = float((w_pred_b.norm(dim=-1) - w_gt_b.norm(dim=-1)).abs().mean().item())
            result[f"Bone/{name}/AngVelMAE"] = float((w_pred_b - w_gt_b).abs().mean().item())

    key_bone_names = getattr(trainer, "eval_key_bones", None) or [
        "pelvis",
        "upperarm_l", "lowerarm_l", "hand_l",
        "upperarm_r", "lowerarm_r", "hand_r",
        "thigh_l", "calf_l", "foot_l",
        "thigh_r", "calf_r", "foot_r",
    ]

    idx_map = {name: idx for idx, name in enumerate(cfg.bone_names)}
    key_indices = [idx_map[name] for name in key_bone_names if name in idx_map]
    key_geo_vals: list[float] = []
    key_geo_local_vals: list[float] = []
    key_ang_mae_vals: list[float] = []
    key_ang_mag_mae_vals: list[float] = []
    key_ang_mag_rel_vals: list[float] = []
    key_ang_dir_vals: list[float] = []
    keybone_details: Dict[str, Dict[str, float]] = {}

    geo_local_tensor = state.geo_local if torch.is_tensor(state.geo_local) else None
    if geo_local_tensor is None:
        raise RuntimeError("GeoLocalDeg metrics unavailable; ensure FK + geodesic computation succeeded before KeyBone diagnostics.")

    for name in key_bone_names:
        if name not in idx_map:
            continue
        joint_idx = idx_map[name]
        prefix = f"KeyBone/{name}"
        geo_val = float((state.geo[..., joint_idx].mean() * cfg.deg).item()) if state.geo is not None and state.geo.shape[-1] > joint_idx else float("nan")
        result[f"{prefix}/GeoDeg"] = geo_val
        if _math.isfinite(geo_val):
            key_geo_vals.append(geo_val)

        geo_local_val = float(geo_local_tensor[..., joint_idx].mean().item()) if geo_local_tensor.shape[-1] > joint_idx else float("nan")
        result[f"{prefix}/GeoLocalDeg"] = geo_local_val
        if _math.isfinite(geo_local_val):
            key_geo_local_vals.append(geo_local_val)

        if state.w_pred is not None and state.w_gt is not None and state.w_pred.shape[2] > joint_idx:
            w_pred_b = state.w_pred[..., joint_idx, :]
            w_gt_b = state.w_gt[..., joint_idx, :]
            ang_mae = float((w_pred_b - w_gt_b).abs().mean().item())
            mag_p = w_pred_b.norm(dim=-1)
            mag_g = w_gt_b.norm(dim=-1)
            mag_avg = 0.5 * (mag_p + mag_g)
            mag_rel = (mag_p - mag_g).abs() / (mag_avg + cfg.mag_rel_beta)
            mag_mae = float((mag_p - mag_g).abs().mean().item())
            valid_mag = mag_avg > cfg.mag_rel_threshold
            mag_rel_val = float(torch.median(mag_rel[valid_mag]).item()) if valid_mag.any() else float("nan")
            result[f"{prefix}/AngVelMAE"] = ang_mae
            result[f"{prefix}/AngVelMagMAE"] = mag_mae
            result[f"{prefix}/AngVelMagRel"] = mag_rel_val
            if _math.isfinite(ang_mae):
                key_ang_mae_vals.append(ang_mae)
            if _math.isfinite(mag_mae):
                key_ang_mag_mae_vals.append(mag_mae)
            if _math.isfinite(mag_rel_val):
                key_ang_mag_rel_vals.append(mag_rel_val)
        else:
            result[f"{prefix}/AngVelMAE"] = float("nan")
            result[f"{prefix}/AngVelMagMAE"] = float("nan")
            result[f"{prefix}/AngVelMagRel"] = float("nan")

        dir_val = geo_local_val
        if not _math.isfinite(dir_val):
            raise RuntimeError(f"GeoLocalDeg for key bone '{name}' is NaN; ensure FK skeleton matches outputs.")
        result[f"{prefix}/AngVelDirDeg"] = dir_val
        key_ang_dir_vals.append(dir_val)
        keybone_details[name] = {
            "GeoDeg": geo_val,
            "GeoLocalDeg": geo_local_val,
            "AngVelMAE": result[f"{prefix}/AngVelMAE"],
            "AngVelMagMAE": result[f"{prefix}/AngVelMagMAE"],
            "AngVelMagRel": result[f"{prefix}/AngVelMagRel"],
            "AngVelDirDeg": dir_val,
        }

    summary = {}
    try:
        geo_group_means = {}
        name_to_idx_full = {name: idx for idx, name in enumerate(cfg.bone_names[:geo_local_tensor.shape[-1]])}

        def _group_mean_from_names(names_group: Sequence[str]) -> Optional[float]:
            idxs = [name_to_idx_full[name] for name in names_group if name in name_to_idx_full]
            if not idxs:
                return None
            group_tensor = geo_local_tensor[..., idxs]
            return float(group_tensor.mean().item())

        leg_mean = _group_mean_from_names(DEFAULT_DIRECT_POSE_LEG_BONES)
        arm_mean = _group_mean_from_names(STAGE6_3WAY_ARMCHAIN_BONES)
        trunk_names = [
            name
            for name in cfg.bone_names[:geo_local_tensor.shape[-1]]
            if name not in set(DEFAULT_DIRECT_POSE_LEG_BONES)
            and name not in set(STAGE6_3WAY_ARMCHAIN_BONES)
            and name_to_idx_full.get(name, -1) != cfg.root_idx
        ]
        trunk_mean = _group_mean_from_names(trunk_names)
        if leg_mean is not None:
            geo_group_means["leg"] = leg_mean
        if arm_mean is not None:
            geo_group_means["arm"] = arm_mean
        if trunk_mean is not None:
            geo_group_means["trunk"] = trunk_mean
        if geo_group_means:
            summary["group_mean"] = geo_group_means
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        _diag_warn_once("diag/keybone_group_mean", "failed to build KeyBone group_mean summary", exc)

    if key_geo_vals:
        summary["GeoDegMean"] = float(sum(key_geo_vals) / len(key_geo_vals))
        _record_diag_metric(result, cfg.diag_scope, "KeyBone/GeoDegMean", summary["GeoDegMean"])
    if key_ang_mae_vals:
        summary["AngVelMAE"] = float(sum(key_ang_mae_vals) / len(key_ang_mae_vals))
        _record_diag_metric(result, cfg.diag_scope, "KeyBone/AngVelMAE", summary["AngVelMAE"])
    if key_ang_mag_mae_vals:
        summary["AngVelMagMAE"] = float(sum(key_ang_mag_mae_vals) / len(key_ang_mag_mae_vals))
        _record_diag_metric(result, cfg.diag_scope, "KeyBone/AngVelMagMAE", summary["AngVelMagMAE"])
    if key_ang_mag_rel_vals:
        summary["AngVelMagRel"] = float(sum(key_ang_mag_rel_vals) / len(key_ang_mag_rel_vals))
        _record_diag_metric(result, cfg.diag_scope, "KeyBone/AngVelMagRel", summary["AngVelMagRel"])
    if not key_geo_local_vals:
        raise RuntimeError("KeyBone GeoLocalDegMean is empty; diagnostics require valid limb geodesic values.")
    summary["GeoLocalDegMean"] = float(sum(key_geo_local_vals) / len(key_geo_local_vals))
    _record_diag_metric(result, cfg.diag_scope, "KeyBone/GeoLocalDegMean", summary["GeoLocalDegMean"])
    if key_ang_dir_vals:
        summary["AngVelDirDeg"] = float(sum(key_ang_dir_vals) / len(key_ang_dir_vals))
        _record_diag_metric(result, cfg.diag_scope, "KeyBone/AngVelDirDeg", summary["AngVelDirDeg"])
    if key_indices:
        kb_curve = geo_local_tensor[:, :, key_indices].mean(dim=(0, 2))
        result["KeyBone/AngVelDirDegCurve"] = kb_curve.detach().cpu().tolist()
    if keybone_details:
        _record_diag_metric(result, cfg.diag_scope, "KeyBoneDetails", keybone_details)
    if summary:
        _record_diag_metric(result, cfg.diag_scope, "KeyBoneSummary", summary)


@torch.no_grad()
def diagnose_free_run(
    trainer,
    *,
    batch,
    predY,
    gtY,
    predsX,
    period_seq_pred,
    motion_seq,
    y_seq,
    contacts_seq,
    angvel_seq,
    pose_hist_seq,
    angvel_raw_seq=None,
) -> Optional[Dict[str, Any]]:
    trainer._require_normalizer("diagnose_free_run")
    _ = y_seq, angvel_raw_seq

    bone_names_src = getattr(trainer, "_bone_names", None)
    if not bone_names_src:
        bundle_meta = getattr(trainer, "_bundle_meta", None)
        if isinstance(bundle_meta, dict):
            bone_names_src = bundle_meta.get("bone_names") or bundle_meta.get("skeleton", {}).get("bone_names")
    bone_names = [str(name) for name in bone_names_src] if isinstance(bone_names_src, (list, tuple)) else []

    result: Dict[str, Any] = {
        "MSEnormY": float(torch.mean((predY - gtY) ** 2).item()),
    }
    cfg = SimpleNamespace(
        diag_scope=str(getattr(trainer, "_diag_scope", "free_run")),
        rot6d_y=getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None),
        rv_x=getattr(trainer, "rootvel_x_slice", None),
        rot6d_x=getattr(trainer, "rot6d_x_slice", None) or getattr(trainer, "rot6d_slice", None),
        eval_align_root=bool(getattr(trainer, "eval_align_root0", True)),
        root_idx=int(getattr(trainer, "eval_root_idx", getattr(trainer, "root_idx", 0))),
        up_axis=int(getattr(trainer, "eval_up_axis", getattr(trainer, "_up_axis", 2))),
        fps_eval=float(getattr(trainer, "bone_hz", getattr(trainer, "fps", 60.0))),
        contact_threshold=float(getattr(trainer, "foot_contact_threshold", 1.5)),
        diag_input_stats=bool(getattr(trainer, "diag_input_stats", False)),
        yaw_forward_axis_offset=float(getattr(trainer, "yaw_forward_axis_offset", 0.0) or 0.0),
        mag_rel_beta=float(getattr(trainer, "eval_angvel_beta", 0.25) or 0.25),
        mag_rel_threshold=float(getattr(trainer, "eval_angvel_mag_threshold", 0.10) or 0.10),
        angvel_eps=float(getattr(trainer, "angvel_eps", 1e-6) or 1e-6),
        angvel_dir_threshold=float(getattr(trainer, "angvel_dir_threshold", 0.1) or 0.1),
        deg=180.0 / _math.pi,
        bone_names=bone_names,
        angvel_slice=getattr(trainer, "angvel_x_slice", None),
    )

    seqs = _collect_diag_sequences(trainer, predsX=predsX, motion_seq=motion_seq, batch=batch)
    _compute_input_drift_metrics(trainer, result, cfg, seqs, period_seq_pred=period_seq_pred)
    state = _compute_contact_and_angvel_metrics(
        trainer,
        result,
        cfg,
        seqs,
        predY=predY,
        gtY=gtY,
        period_seq_pred=period_seq_pred,
        contacts_seq=contacts_seq,
        angvel_seq=angvel_seq,
        pose_hist_seq=pose_hist_seq,
    )
    _compute_keybone_metrics(trainer, result, cfg, state)

    if state.w_gt is not None and seqs.gtX_raw_full is not None and isinstance(cfg.angvel_slice, slice):
        try:
            angvel_data = seqs.gtX_raw_full[:, :state.w_gt.shape[1] + 1, cfg.angvel_slice]
            joint_count = (cfg.angvel_slice.stop - cfg.angvel_slice.start) // 3
            if joint_count == state.w_gt.shape[2]:
                angvel_data = angvel_data[:, 1:state.w_gt.shape[1] + 1].reshape(
                    state.w_gt.shape[0], state.w_gt.shape[1], joint_count, 3
                )
                diff_gt = (state.w_gt - angvel_data).abs()
                result["AngVelGTReconMAE"] = float(diff_gt.mean().item())
                dot_gt = (state.w_gt * angvel_data).sum(dim=-1)
                norm_gt = state.w_gt.norm(dim=-1) * angvel_data.norm(dim=-1)
                mask_gt = norm_gt > 1e-6
                if mask_gt.any():
                    ang_dir = torch.acos(torch.clamp(dot_gt[mask_gt] / norm_gt[mask_gt], -1.0, 1.0)) * cfg.deg
                    result["AngVelGTReconDirDeg"] = float(ang_dir.mean().item())
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            _diag_warn_once("diag/angvel_gt_recon", "failed to compute AngVelGTRecon diagnostics", exc)

    if state.w_pred is not None:
        contact_pred = (state.w_pred.norm(dim=-1) < cfg.contact_threshold).float()
        result["FootContact"] = float(contact_pred.mean().item())
    if result is None:
        clip = batch.get("clip_id") if isinstance(batch, dict) else None
        start = batch.get("start") if isinstance(batch, dict) else None
        print(f"[FreeRunDiag][WARN] diagnose_free_run returned None (clip={clip}, start={start})")
    return result


def history_drift_debug(
    trainer,
    state_seq,
    gt_seq,
    cond_seq,
    cond_raw_seq,
    contacts_seq,
    angvel_seq,
    pose_hist_seq,
    *,
    epoch: int,
    batch_idx: int,
    cond_norm_mu=None,
    cond_norm_std=None,
) -> None:
    steps = min(int(getattr(trainer, "history_debug_steps", 0) or 0), state_seq.shape[1])
    if steps <= 1:
        return
    pred_out = _run_history_drift_rollout(
        trainer,
        steps=steps,
        state_seq=state_seq,
        gt_seq=gt_seq,
        cond_seq=cond_seq,
        cond_raw_seq=cond_raw_seq,
        contacts_seq=contacts_seq,
        angvel_seq=angvel_seq,
        pose_hist_seq=pose_hist_seq,
        cond_norm_mu=cond_norm_mu,
        cond_norm_std=cond_norm_std,
    )
    if pred_out is None:
        return
    try:
        gt_raw = trainer._denorm(gt_seq[:, :steps])
        pred_raw = trainer._denorm(pred_out)
    except Exception as exc:
        print(f"[HistDrift][warn] denorm failed: {exc}")
        return
    stats = _compute_history_drift_geo_local_stats(trainer, gt_raw=gt_raw, pred_raw=pred_raw)
    if not stats:
        return
    _emit_history_drift_debug_lines(trainer, stats, epoch=epoch, batch_idx=batch_idx, steps=steps)


def _run_history_drift_rollout(
    trainer,
    *,
    steps: int,
    state_seq,
    gt_seq,
    cond_seq,
    cond_raw_seq,
    contacts_seq,
    angvel_seq,
    pose_hist_seq,
    cond_norm_mu,
    cond_norm_std,
) -> Optional[torch.Tensor]:
    def _slice_time_seq(value):
        return value[:, :steps] if torch.is_tensor(value) and value.dim() == 3 else value

    with torch.no_grad():
        preds_free, _ = trainer._rollout_sequence(
            state_seq[:, :steps],
            _slice_time_seq(cond_seq),
            _slice_time_seq(cond_raw_seq),
            contacts_seq=_slice_time_seq(contacts_seq),
            angvel_seq=_slice_time_seq(angvel_seq),
            pose_hist_seq=_slice_time_seq(pose_hist_seq),
            gt_seq=gt_seq[:, :steps],
            mode="train_free",
            tf_ratio=0.0,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
        )
    return preds_free.get("out") if isinstance(preds_free, dict) else None


def _compute_history_drift_geo_local_stats(
    trainer,
    *,
    gt_raw: torch.Tensor,
    pred_raw: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    rot_slice = getattr(trainer, "rot6d_y_slice", None) or getattr(trainer, "rot6d_slice", None)
    if not isinstance(rot_slice, slice):
        return None
    rot_width = int(rot_slice.stop - rot_slice.start)
    if rot_width <= 0 or (rot_width % 6) != 0:
        return None
    batch_size, steps = gt_raw.shape[:2]
    joint_count = rot_width // 6
    gt_m = rot6d_to_matrix(reproject_rot6d(gt_raw[..., rot_slice].reshape(batch_size * steps, joint_count, 6))).view(batch_size, steps, joint_count, 3, 3)
    pred_m = rot6d_to_matrix(reproject_rot6d(pred_raw[..., rot_slice].reshape(batch_size * steps, joint_count, 6))).view(batch_size, steps, joint_count, 3, 3)
    root_idx = int(getattr(trainer, "eval_root_idx", getattr(trainer, "root_idx", 0)))
    pred_root = _root_relative_matrices(pred_m, root_idx)
    gt_root = _root_relative_matrices(gt_m, root_idx)
    joint_weights = trainer._joint_weights(pred_root, joint_count)
    weights_sum = joint_weights.sum().clamp_min(1e-6)
    geo_local_rad = geodesic_R(pred_root, gt_root).detach()
    geo_local_step_deg = ((geo_local_rad * (180.0 / _math.pi) * joint_weights.view(1, 1, -1)).sum(dim=-1) / weights_sum).mean(dim=0)
    return {
        "rot_local_mean_deg": float(geo_local_step_deg.mean().item()),
        "rot_local_step_deg": geo_local_step_deg.cpu().tolist(),
        "geo_local_rad": geo_local_rad,
    }


def _emit_history_drift_debug_lines(
    trainer,
    stats: Mapping[str, Any],
    *,
    epoch: int,
    batch_idx: int,
    steps: int,
) -> None:
    def _collect_limb_summary(geo_local_tensor_rad: Optional[torch.Tensor], *, error_label: str) -> dict[str, Any]:
        collect_fn = getattr(trainer.loss_fn, "_collect_limb_local_stats", None)
        if not (torch.is_tensor(geo_local_tensor_rad) and callable(collect_fn)):
            return {}
        try:
            summary = collect_fn(geo_local_tensor_rad)
        except Exception as exc:
            print(f"[HistDrift][ERR] {error_label}: {exc}")
            return {}
        return dict(summary) if isinstance(summary, Mapping) else {}

    def _summary_suffix(summary: Mapping[str, Any], pairs: Sequence[tuple[str, str]]) -> str:
        suffix = ""
        for key, template in pairs:
            value = summary.get(key, float("nan"))
            if isinstance(value, (float, int)) and _math.isfinite(float(value)):
                suffix += template.format(float(value))
        return suffix

    geo_local_tensor_rad = stats.get("geo_local_rad")
    local_val = stats.get("rot_local_mean_deg", float("nan"))
    print(
        "[HistDrift]"
        f"[ep {int(epoch):03d}]"
        f"[bi {int(batch_idx):04d}] "
        f"steps={steps}"
        f"{_summary_suffix(_collect_limb_summary(geo_local_tensor_rad, error_label='limb summary failed'), (('rot_local_limb_deg', ' limb={:.2f}°'), ('rot_local_limb_over_torso', ' limb/torso={:.2f}')))}"
        f"{f' local={float(local_val):.2f}°' if isinstance(local_val, (float, int)) and _math.isfinite(float(local_val)) else ''}"
    )
    local_curve = stats.get("rot_local_step_deg")
    if not isinstance(local_curve, list):
        return
    for idx, local_val_step in enumerate(local_curve, start=1):
        if not isinstance(local_val_step, (float, int)) or not _math.isfinite(float(local_val_step)):
            continue
        step_summary = {}
        if torch.is_tensor(geo_local_tensor_rad) and geo_local_tensor_rad.shape[1] >= idx:
            step_summary = _collect_limb_summary(
                geo_local_tensor_rad[:, idx - 1:idx],
                error_label=f"limb step summary failed (step={idx})",
            )
        print(
            "[HistDrift]"
            f"[ep {int(epoch):03d}]"
            f"[bi {int(batch_idx):04d}]"
            f"[step {idx:02d}] local={float(local_val_step):.2f}°"
            f"{_summary_suffix(step_summary, (('rot_local_limb_deg', ' limb={:.2f}°'), ('rot_local_torso_deg', ' torso={:.2f}°')))}"
        )


def test_gradient_connection(trainer, loader) -> None:
    if getattr(trainer, "_grad_connection_checked", False):
        return
    if not bool(getattr(trainer, "enable_grad_connection_test", True)):
        trainer._grad_connection_checked = True
        return
    sample_batch = None
    iterator = iter(loader)
    try:
        sample_batch = next(iterator)
    except StopIteration:
        print("[GradConn] skipped: empty loader.")
        trainer._grad_connection_checked = True
        return
    x_cand = trainer._pick_first(sample_batch, ("motion", "X", "x_in_features"))
    y_cand = trainer._pick_first(sample_batch, ("gt_motion", "Y", "y_out_features", "y_out_seq"))
    if x_cand is None or y_cand is None:
        print("[GradConn] skipped: batch missing motion/gt.")
        trainer._grad_connection_checked = True
        return
    state_seq = x_cand.to(trainer.device).float()
    gt_seq = y_cand.to(trainer.device).float()
    window = min(int(getattr(trainer, "grad_conn_window", 8) or 8), state_seq.shape[1])
    if window < 2:
        print("[GradConn] skipped: window < 2.")
        trainer._grad_connection_checked = True
        return
    state_seq = state_seq[:, :window]
    gt_seq = gt_seq[:, :window]

    def _slice_optional(key):
        val = sample_batch.get(key) if isinstance(sample_batch, dict) else None
        if val is None:
            return None
        tensor = val.to(trainer.device).float()
        if tensor.dim() == 3 and tensor.size(1) >= window:
            return tensor[:, :window]
        return tensor

    cond_seq = _slice_optional("cond_in")
    cond_raw_seq = _slice_optional("cond_tgt_raw")
    contacts_seq = _slice_optional("contacts")
    angvel_seq = _slice_optional("angvel")
    pose_hist_seq = _slice_optional("pose_hist")
    cond_norm_mu = sample_batch.get("cond_norm_mu") if isinstance(sample_batch, dict) else None
    cond_norm_std = sample_batch.get("cond_norm_std") if isinstance(sample_batch, dict) else None
    if cond_norm_mu is not None:
        cond_norm_mu = cond_norm_mu.to(trainer.device).float()
    if cond_norm_std is not None:
        cond_norm_std = cond_norm_std.to(trainer.device).float()
    time_base = None
    try:
        start_base = sample_batch.get("start") if isinstance(sample_batch, dict) else None
        if start_base is not None and torch.is_tensor(start_base):
            time_base = start_base.to(trainer.device).float()
    except Exception:
        time_base = None

    use_anomaly = bool(getattr(trainer, "grad_conn_detect_anomaly", True))
    import contextlib

    anomaly_ctx = torch.autograd.set_detect_anomaly if use_anomaly else contextlib.nullcontext
    with anomaly_ctx(True if use_anomaly else False):
        preds, attn = trainer._rollout_sequence(
            state_seq,
            cond_seq,
            cond_raw_seq,
            contacts_seq=contacts_seq,
            angvel_seq=angvel_seq,
            pose_hist_seq=pose_hist_seq,
            gt_seq=gt_seq,
            cond_norm_mu=cond_norm_mu,
            cond_norm_std=cond_norm_std,
            mode="train_free",
            tf_ratio=0.0,
            time_base=time_base,
        )
        with trainer._amp_context(trainer.use_amp):
            out = trainer.loss_fn(preds, gt_seq, attn_weights=attn, batch=sample_batch)
        loss = out[0] if isinstance(out, tuple) else out
        trainer.optimizer.zero_grad(set_to_none=True)
        try:
            loss.backward()
        except RuntimeError as exc:
            raise RuntimeError("[GradConn] backward failed; 检查 train_free 梯度链路。") from exc
    grad_hits = sum(1 for param in trainer.model.parameters() if param.grad is not None and torch.isfinite(param.grad).any())
    if grad_hits == 0:
        raise RuntimeError("[GradConn] backward produced no gradients; 可能仍有 detach().")
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._grad_connection_checked = True
    print(f"[GradConn] ok: window={window} grad_hits={grad_hits}.")


def dump_nan_grad_report(
    trainer,
    epoch,
    batch_idx,
    batch,
    state_seq,
    gt_seq,
    preds_dict,
    loss_value,
    stats,
) -> None:
    out_dir = getattr(trainer, "out_dir", None)
    if not out_dir:
        return
    limit = int(getattr(trainer, "nan_grad_report_limit", 0) or 0)
    if trainer.nan_grad_reports >= limit:
        return

    def _tensor_stats(tensor):
        if tensor is None:
            return None
        try:
            t = tensor.detach()
            if t.numel() == 0:
                return {"shape": list(t.shape), "numel": 0}
            t = t.to(dtype=torch.float32, device="cpu")
            return {
                "shape": list(t.shape),
                "numel": int(t.numel()),
                "min": float(t.min().item()),
                "max": float(t.max().item()),
                "mean": float(t.mean().item()),
                "std": float(t.std().item()),
            }
        except Exception as exc:
            return {"error": str(exc)}

    try:
        import json
        import os

        os.makedirs(os.path.join(out_dir, "nan_grad"), exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "tf_ratio": float(getattr(trainer, "_last_tf_ratio", 1.0)),
            "loss": float(loss_value),
            "loss_parts": dict(stats) if isinstance(stats, dict) else {},
            "state_stats": _tensor_stats(state_seq),
            "gt_stats": _tensor_stats(gt_seq),
            "pred_out_stats": _tensor_stats(preds_dict.get("out") if isinstance(preds_dict, dict) else None),
            "pred_delta_stats": _tensor_stats(preds_dict.get("delta") if isinstance(preds_dict, dict) else None),
            "batch_meta": {},
        }
        if isinstance(batch, dict):
            clip_id = batch.get("clip_id")
            start = batch.get("start")
            if clip_id is not None:
                clip_id_int = safe_int_scalar(clip_id)
                if clip_id_int is not None:
                    payload["batch_meta"]["clip_id"] = clip_id_int
            if start is not None:
                start_int = safe_int_scalar(start)
                if start_int is not None:
                    payload["batch_meta"]["start"] = start_int
        fname = os.path.join(out_dir, "nan_grad", f"ep{int(epoch):03d}_b{int(batch_idx):05d}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        trainer.nan_grad_reports += 1
        print(f"[GradNan] dumped diagnostic to {fname}")
    except Exception as exc:
        print(f"[GradNan][WARN] failed to dump diagnostic: {exc}")


def collect_freerun_step_debug_record(
    trainer,
    *,
    step_idx: int,
    motion_raw: Optional[torch.Tensor],
    gt_motion_raw: Optional[torch.Tensor],
    cond_raw_step: Optional[torch.Tensor],
    delta_norm: Optional[torch.Tensor],
) -> Optional[Dict[str, Any]]:
    rec: Dict[str, Any] = {"step": int(step_idx)}
    yaw_sl = getattr(trainer, "yaw_x_slice", None)
    rootvel_sl = getattr(trainer, "rootvel_x_slice", None)
    if isinstance(yaw_sl, slice) and motion_raw is not None and gt_motion_raw is not None:
        deg = 180.0 / torch.pi
        yaw_pred = motion_raw[..., yaw_sl]
        yaw_gt = gt_motion_raw[..., yaw_sl]
        if yaw_pred.shape[-1] == 1:
            yaw_pred = yaw_pred.squeeze(-1)
        if yaw_gt.shape[-1] == 1:
            yaw_gt = yaw_gt.squeeze(-1)
        dyaw_world = wrap_to_pi_torch(yaw_pred - yaw_gt)
        rec["yaw_world_abs_deg"] = float(dyaw_world.abs().mean().item() * deg)
        rec["yaw_abs_deg"] = rec["yaw_world_abs_deg"]
        if motion_raw.shape[0] > 0:
            rec["yaw_pred_s0"] = float(motion_raw[0, yaw_sl].mean().item())
        if gt_motion_raw.shape[0] > 0:
            rec["yaw_gt_s0"] = float(gt_motion_raw[0, yaw_sl].mean().item())
    if isinstance(rootvel_sl, slice) and motion_raw is not None and gt_motion_raw is not None:
        rv_err = (motion_raw[..., rootvel_sl] - gt_motion_raw[..., rootvel_sl]).abs().mean().item()
        rec["root_vel_mae"] = float(rv_err)
        if motion_raw.shape[0] > 0:
            rec["root_vel_pred_s0"] = motion_raw[0, rootvel_sl].detach().cpu().tolist()
        if gt_motion_raw.shape[0] > 0:
            rec["root_vel_gt_s0"] = gt_motion_raw[0, rootvel_sl].detach().cpu().tolist()
    if cond_raw_step is not None and torch.is_tensor(cond_raw_step) and cond_raw_step.shape[0] > 0:
        rec["cond_next_raw_s0"] = cond_raw_step[0].detach().cpu().tolist()
    if delta_norm is not None:
        rec["delta_norm_abs"] = float(delta_norm.abs().mean().item())
    return rec if rec else None


def attach_delta_energy_metrics(batch_stats: Dict[str, Any], diag_records: Sequence[Mapping[str, Any]]) -> None:
    if not diag_records:
        return
    delta_vals = [rec.get("delta_norm_abs") for rec in diag_records if isinstance(rec.get("delta_norm_abs"), (int, float))]
    if not delta_vals:
        return
    delta_mean = sum(delta_vals) / len(delta_vals)
    if len(delta_vals) > 1:
        delta_var = sum((value - delta_mean) ** 2 for value in delta_vals) / (len(delta_vals) - 1)
    else:
        delta_var = 0.0
    batch_stats["Diag/DeltaEnergyMean"] = float(delta_mean)
    batch_stats["Diag/DeltaEnergyVar"] = float(delta_var)


def save_freerun_debug_payload(
    trainer,
    *,
    batch,
    batch_stats: Mapping[str, Any],
    diag_records: Sequence[Mapping[str, Any]],
    batches_processed: int,
    warmup: int,
    horizon: int,
    tf_ratio: float,
    base_debug_path: Optional[str],
) -> Optional[str]:
    if not diag_records or not base_debug_path:
        return None

    def _summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
        focus_keys = ("yaw_abs_deg", "yaw_world_abs_deg", "root_vel_mae", "delta_norm_abs")
        summaries: Dict[str, Dict[str, float]] = {}
        for key in focus_keys:
            vals = [float(rec[key]) for rec in records if key in rec and isinstance(rec[key], (int, float))]
            if not vals:
                continue
            start_v, end_v = vals[0], vals[-1]
            num = len(vals)
            mean_v = float(sum(vals) / num)
            var_v = float(sum((value - mean_v) ** 2 for value in vals) / max(1, num - 1))
            summaries[key] = {
                "start": start_v,
                "end": end_v,
                "min": float(min(vals)),
                "max": float(max(vals)),
                "mean": mean_v,
                "std": float(_math.sqrt(var_v)),
                "trend": float(end_v - start_v),
                "per_step": float((end_v - start_v) / max(1, num - 1)),
            }
        return summaries

    epoch = int(getattr(trainer, "cur_epoch", 0) or 0)
    run_name = getattr(trainer, "_current_run_name", None)
    suffix = f"ep{epoch:03d}" if epoch > 0 else "ep"
    if run_name:
        suffix = f"{run_name}_{suffix}"
    candidate = Path(base_debug_path)
    if candidate.is_dir() or str(base_debug_path).endswith("/"):
        candidate = candidate / f"freerun_diag_{suffix}_b{batches_processed:02d}.pt"
    else:
        candidate = candidate.with_name(candidate.stem + f"_{suffix}_b{batches_processed:02d}" + candidate.suffix)
    try:
        slice_stats = _summarize_records(diag_records)
        meta = {
            "epoch": epoch,
            "run_name": run_name,
            "warmup": warmup,
            "horizon": int(horizon),
            "tf_ratio": float(tf_ratio),
            "freerun_weight": float(getattr(trainer, "freerun_weight", 0.0) or 0.0),
            "diag_steps": len(diag_records),
            "batch_index": batches_processed,
        }
        candidate.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clip_id": batch.get("clip_id") if isinstance(batch, dict) else None,
            "start": batch.get("start") if isinstance(batch, dict) else None,
            "records": list(diag_records),
            "metrics": dict(batch_stats),
            "slice_stats": slice_stats,
            "meta": meta,
            "keybone_summary": batch_stats.get("KeyBoneSummary"),
            "keybone_details": batch_stats.get("KeyBoneDetails"),
        }
        torch.save(payload, str(candidate))
        headline = []
        if "root_vel_mae" in slice_stats:
            headline.append(f"root_vel {slice_stats['root_vel_mae']['mean']:.3f}")
        if "delta_norm_abs" in slice_stats:
            headline.append(f"|Δ| {slice_stats['delta_norm_abs']['mean']:.3f}")
        summary_line = " | ".join(headline)
        print(f"[FreeRunDiag] saved diagnostics to {candidate}" + (f" :: {summary_line}" if summary_line else ""))
        return str(candidate)
    except Exception as exc:
        print(f"[FreeRunDiag][WARN] failed to save diagnostics: {exc}")
        return None


__all__ = [
    "_maybe_optimize_dataset_index",
    "_norm_debug_once",
    "_parse_stage_schedule",
    "aggregate_metric_samples",
    "attach_delta_energy_metrics",
    "collect_direct_pose_grad_stats",
    "collect_freerun_step_debug_record",
    "diagnose_free_run",
    "dump_nan_grad_report",
    "history_drift_debug",
    "save_freerun_debug_payload",
    "test_gradient_connection",
]
