from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from .io import speed_from_X_layout as _speed_from_X_layout


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



__all__ = ['_maybe_optimize_dataset_index','_norm_debug_once','_parse_stage_schedule']
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

