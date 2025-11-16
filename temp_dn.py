class DataNormalizer:
    """封装数据规格与(反)归一化逻辑。"""
    def __init__(self, *,
                 mu_x=None, std_x=None, mu_y=None, std_y=None,
                 y_to_x_map=None,
                 yaw_x_slice=None, yaw_y_slice=None,
                 rootvel_x_slice=None, rootvel_y_slice=None,
                 angvel_x_slice=None, angvel_y_slice=None,
                 tanh_scales_rootvel=None, tanh_scales_angvel=None,
                 traj_dir_slice=None,
                 angvel_mode=None, angvel_mu=None, angvel_std=None):
        import numpy as np
        self.mu_x = None if mu_x is None else np.asarray(mu_x, dtype=np.float32)
        self.std_x = None if std_x is None else np.asarray(std_x, dtype=np.float32)
        self.mu_y = None if mu_y is None else np.asarray(mu_y, dtype=np.float32)
        self.std_y = None if std_y is None else np.asarray(std_y, dtype=np.float32)
        self.y_to_x_map = y_to_x_map or []
        self.yaw_x_slice      = parse_layout_entry(yaw_x_slice,      'RootYaw')
        self.yaw_y_slice      = parse_layout_entry(yaw_y_slice,      'RootYaw')
        self.rootvel_x_slice  = parse_layout_entry(rootvel_x_slice,  'RootVelocity')
        self.rootvel_y_slice  = parse_layout_entry(rootvel_y_slice,  'RootVelocity')
        self.angvel_x_slice   = parse_layout_entry(angvel_x_slice,   'BoneAngularVelocities')
        self.angvel_y_slice   = parse_layout_entry(angvel_y_slice,   'BoneAngularVelocities')
        self.traj_dir_slice   = parse_layout_entry(traj_dir_slice,   'TrajectoryDir')

        self.tanh_scales_rootvel = None if tanh_scales_rootvel is None else np.asarray(tanh_scales_rootvel, dtype=np.float32)
        self.tanh_scales_angvel  = None if tanh_scales_angvel  is None else np.asarray(tanh_scales_angvel,  dtype=np.float32)
        self.angvel_mode = (angvel_mode or '').strip().lower() if isinstance(angvel_mode, str) else None
        self.angvel_mu = None if angvel_mu is None else np.asarray(angvel_mu, dtype=np.float32)
        self.angvel_std = None if angvel_std is None else np.asarray(angvel_std, dtype=np.float32)
        if self.angvel_std is not None:
            self.angvel_std = np.clip(self.angvel_std, 1e-6, None)

        # torch tensor cache: key = (name, device, dtype)
        self._tensor_cache: dict[tuple[str, str, str], "torch.Tensor"] = {}

        def _slice_width(sl):
            return int(sl.stop - sl.start) if isinstance(sl, slice) else 0

        root_widths = (_slice_width(self.rootvel_x_slice), _slice_width(self.rootvel_y_slice))
        max_root_width = max(root_widths)
        if max_root_width:
            if self.tanh_scales_rootvel is None:
                raise ValueError("DataNormalizer requires tanh_scales_rootvel when RootVelocity is present.")
            if self.tanh_scales_rootvel.size != max_root_width:
                raise ValueError(f"tanh_scales_rootvel length {self.tanh_scales_rootvel.size} "
                                 f"!= RootVelocity width {max_root_width}.")

        ang_widths = (_slice_width(self.angvel_x_slice), _slice_width(self.angvel_y_slice))
        max_ang_width = max(ang_widths)
        if max_ang_width:
            if self.angvel_mode == 'standardize':
                if self.angvel_mu is None or self.angvel_std is None:
                    raise ValueError("angvel_mode=standardize requires MuAngVel and StdAngVel.")
                if self.angvel_mu.size != max_ang_width or self.angvel_std.size != max_ang_width:
                    raise ValueError(f"MuAngVel/StdAngVel length mismatch ({self.angvel_mu.size}/{self.angvel_std.size}) "
                                     f"vs BoneAngularVelocities width {max_ang_width}.")
            else:
                self.angvel_mode = 'tanh'
                if self.tanh_scales_angvel is None:
                    raise ValueError("tanh_scales_angvel missing for BoneAngularVelocities slice.")
                if self.tanh_scales_angvel.size != max_ang_width:
                    raise ValueError(f"tanh_scales_angvel length {self.tanh_scales_angvel.size} "
                                     f"!= BoneAngularVelocities width {max_ang_width}.")
        else:
            if self.angvel_mode == 'standardize' and (self.angvel_mu is None or self.angvel_std is None):
                raise ValueError("angvel_mode=standardize declared but BoneAngularVelocities slice is absent.")
            if self.angvel_mode not in (None, 'standardize'):
                self.angvel_mode = 'tanh'
        if self.angvel_mode is None:
            self.angvel_mode = 'tanh'

    def _match_tensor(self, key: str, arr, ref_tensor):
        import torch
        if arr is None:
            return None
        cache_key = (key, str(ref_tensor.device), str(ref_tensor.dtype))
        tensor = self._tensor_cache.get(cache_key)
        if tensor is None:
            tensor = torch.as_tensor(arr, device=ref_tensor.device, dtype=ref_tensor.dtype)
            self._tensor_cache[cache_key] = tensor
        while tensor.dim() < ref_tensor.dim():
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _atanh_safe_t(x, torch):
        x = torch.clamp(x, -0.999999, 0.999999)
        return torch.atanh(x) if hasattr(torch, "atanh") else 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def norm(self, x_raw_t):
        import torch
        x_raw = x_raw_t

        x_proc = x_raw

        if isinstance(self.rootvel_x_slice, slice):
            if self.tanh_scales_rootvel is None:
                raise RuntimeError("tanh_scales_rootvel missing; cannot normalize RootVelocity.")
            x_proc = x_proc.clone()
            sc = self._match_tensor('tanh_rootvel', self.tanh_scales_rootvel, x_raw).clamp_min(1e-6)
            x_proc[..., self.rootvel_x_slice] = torch.tanh(
                x_raw[..., self.rootvel_x_slice] / sc
            )

        if isinstance(self.angvel_x_slice, slice):
            x_proc = x_proc.clone()
            width = self.angvel_x_slice.stop - self.angvel_x_slice.start
            if (
                self.angvel_mode == 'standardize'
                and self.angvel_mu is not None
                and self.angvel_std is not None
                and width == self.angvel_mu.size
            ):
                mu = self._match_tensor('angvel_mu_x', self.angvel_mu, x_raw)
                std = self._match_tensor('angvel_std_x', self.angvel_std, x_raw).clamp_min(1e-6)
                x_proc[..., self.angvel_x_slice] = (
                    x_raw[..., self.angvel_x_slice] - mu
                ) / std
            elif self.angvel_mode == 'tanh':
                if self.tanh_scales_angvel is None:
                    raise RuntimeError("tanh_scales_angvel missing; cannot normalize BoneAngularVelocities.")
                sc = self._match_tensor('tanh_angvel', self.tanh_scales_angvel, x_raw).clamp_min(1e-6)
                x_proc[..., self.angvel_x_slice] = torch.tanh(
                    x_raw[..., self.angvel_x_slice] / sc
                )
            else:
                raise RuntimeError(f"Unsupported angvel_mode during norm(): {self.angvel_mode}")

        if isinstance(self.yaw_x_slice, slice):
            x_proc = x_proc.clone()
            x_proc[..., self.yaw_x_slice] = x_raw[..., self.yaw_x_slice].abs()

        if (self.mu_x is not None) and (self.std_x is not None):
            mu = self._match_tensor('mu_x', self.mu_x, x_raw)
            sd = self._match_tensor('std_x', self.std_x, x_raw).clamp_min(1e-3)
            z = (x_proc - mu) / sd
        else:
            z = x_proc
        return z

    def denorm_x(self, xz_t, prev_raw=None):
        """把 X 的 Z 域还原为 RAW：做 μ/σ 逆 + 分组逆变换；Yaw 因 abs 丢符号，若给了 prev_raw 就延用上一帧符号。"""
        import torch

        xz = xz_t

        if (self.mu_x is not None) and (self.std_x is not None):
            mu = self._match_tensor('mu_x', self.mu_x, xz)
            sd = self._match_tensor('std_x', self.std_x, xz).clamp_min(1e-6)
            x_pre = xz * sd + mu
        else:
            x_pre = xz.clone()

        x_raw = x_pre

        # 分组逆变换（与 norm 对称）
        if isinstance(self.rootvel_x_slice, slice):
            if self.tanh_scales_rootvel is None:
                raise RuntimeError("tanh_scales_rootvel missing; cannot denormalize RootVelocity.")
            x_raw = x_raw.clone()
            sc = self._match_tensor('tanh_rootvel', self.tanh_scales_rootvel, xz)
            x_raw[..., self.rootvel_x_slice] = self._atanh_safe_t(
                x_pre[..., self.rootvel_x_slice], torch
            ) * sc

        if isinstance(self.angvel_x_slice, slice):
            x_raw = x_raw.clone()
            width = self.angvel_x_slice.stop - self.angvel_x_slice.start
            if (
                self.angvel_mode == 'standardize'
                and self.angvel_mu is not None
                and self.angvel_std is not None
                and width == self.angvel_mu.size
            ):
                mu = self._match_tensor('angvel_mu_x', self.angvel_mu, xz)
                std = self._match_tensor('angvel_std_x', self.angvel_std, xz).clamp_min(1e-6)
                x_raw[..., self.angvel_x_slice] = x_pre[..., self.angvel_x_slice] * std + mu
            elif self.angvel_mode == 'tanh':
                if self.tanh_scales_angvel is None:
                    raise RuntimeError("tanh_scales_angvel missing; cannot denormalize BoneAngularVelocities.")
                sc = self._match_tensor('tanh_angvel', self.tanh_scales_angvel, xz)
                x_raw[..., self.angvel_x_slice] = self._atanh_safe_t(
                    x_pre[..., self.angvel_x_slice], torch
                ) * sc
            else:
                raise RuntimeError(f"Unsupported angvel_mode during denorm_x(): {self.angvel_mode}")

        # yaw: 使用上一帧 RAW 的符号（若可用）
        if isinstance(self.yaw_x_slice, slice) and (prev_raw is not None):
            s = self.yaw_x_slice
            prev = torch.as_tensor(prev_raw, device=xz.device, dtype=xz.dtype)
            sign = torch.sign(prev[..., s]).clamp(min=-1.0, max=1.0)
            x_raw = x_raw.clone()
            x_raw[..., s] = torch.abs(x_raw[..., s]) * sign

        return x_raw



    def denorm(self, y_t):
        import torch, math
        y_pre = y_t
        if self.std_y is not None and self.mu_y is not None:
            std = self._match_tensor('std_y', self.std_y, y_t).clamp_min(1e-6)
            mu  = self._match_tensor('mu_y', self.mu_y, y_t)
            y_pre = y_pre * std + mu
        else:
            y_pre = y_t.clone()

        y = y_pre.clone()

        if isinstance(self.yaw_y_slice, slice):
            yaw_clamped = torch.clamp(y_pre[..., self.yaw_y_slice], -1.0, 1.0)
            y[..., self.yaw_y_slice] = yaw_clamped * math.pi

        if isinstance(self.rootvel_y_slice, slice):
            if self.tanh_scales_rootvel is None:
                raise RuntimeError("tanh_scales_rootvel missing; cannot denormalize RootVelocity.")
            sc = self._match_tensor('tanh_rootvel', self.tanh_scales_rootvel, y_pre)
            y[..., self.rootvel_y_slice] = self._atanh_safe_t(
                y_pre[..., self.rootvel_y_slice], torch
            ) * sc

        if isinstance(self.angvel_y_slice, slice):
            width = self.angvel_y_slice.stop - self.angvel_y_slice.start
            if (
                self.angvel_mode == 'standardize'
                and self.angvel_mu is not None
                and self.angvel_std is not None
                and width == self.angvel_mu.size
            ):
                mu = self._match_tensor('angvel_mu_y', self.angvel_mu, y_pre)
                std = self._match_tensor('angvel_std_y', self.angvel_std, y_pre).clamp_min(1e-6)
                y[..., self.angvel_y_slice] = y_pre[..., self.angvel_y_slice] * std + mu
            elif self.angvel_mode == 'tanh':
                if self.tanh_scales_angvel is None:
                    raise RuntimeError("tanh_scales_angvel missing; cannot denormalize BoneAngularVelocities.")
                sc = self._match_tensor('tanh_angvel', self.tanh_scales_angvel, y_pre)
                y[..., self.angvel_y_slice] = self._atanh_safe_t(
                    y_pre[..., self.angvel_y_slice], torch
                ) * sc
            else:
                raise RuntimeError(f"Unsupported angvel_mode during denorm(): {self.angvel_mode}")
        try:
            sl = self.traj_dir_slice
            if isinstance(sl, slice):
                a, b = sl.start, sl.stop - sl.start
                dim = 3 if b % 3 == 0 else 2 if b % 2 == 0 else 0
                if dim > 0:
                    blk = y[..., a:a+b].view(*y.shape[:-1], b // dim, dim)
                    y[..., a:a+b] = torch.nn.functional.normalize(blk, dim=-1).reshape_as(y[..., a:a+b])
        except Exception as _err:
            print(f"[Norm-ERR] trajectory direction normalization failed: {_err}")
            pass
        return y

    def norm_y(self, y_raw_t):
        import torch, math
        y_raw = y_raw_t

        y_pre = y_raw

        if isinstance(self.yaw_y_slice, slice):
            y_pre = y_pre.clone()
            y_pre[..., self.yaw_y_slice] = torch.clamp(
                y_raw[..., self.yaw_y_slice] / math.pi,
                -0.999999,
                0.999999,
            )

        if isinstance(self.rootvel_y_slice, slice):
            if self.tanh_scales_rootvel is None:
                raise RuntimeError("tanh_scales_rootvel missing; cannot normalize RootVelocity.")
            y_pre = y_pre.clone()
            sc = self._match_tensor('tanh_rootvel', self.tanh_scales_rootvel, y_raw).clamp_min(1e-6)
            y_pre[..., self.rootvel_y_slice] = torch.tanh(
                y_raw[..., self.rootvel_y_slice] / sc
            )

        if isinstance(self.angvel_y_slice, slice):
            width = self.angvel_y_slice.stop - self.angvel_y_slice.start
            y_pre = y_pre.clone()
            if (
                self.angvel_mode == 'standardize'
                and self.angvel_mu is not None
                and self.angvel_std is not None
                and width == self.angvel_mu.size
            ):
                mu = self._match_tensor('angvel_mu_y', self.angvel_mu, y_raw)
                std = self._match_tensor('angvel_std_y', self.angvel_std, y_raw).clamp_min(1e-6)
                y_pre[..., self.angvel_y_slice] = (
                    y_raw[..., self.angvel_y_slice] - mu
                ) / std
            elif self.angvel_mode == 'tanh':
                if self.tanh_scales_angvel is None:
                    raise RuntimeError("tanh_scales_angvel missing; cannot normalize BoneAngularVelocities.")
                sc = self._match_tensor('tanh_angvel', self.tanh_scales_angvel, y_raw).clamp_min(1e-6)
                y_pre[..., self.angvel_y_slice] = torch.tanh(
                    y_raw[..., self.angvel_y_slice] / sc
                )
            else:
                raise RuntimeError(f"Unsupported angvel_mode during norm_y(): {self.angvel_mode}")

        try:
            if isinstance(self.traj_dir_slice, slice):
                a = self.traj_dir_slice.start
                b = self.traj_dir_slice.stop - self.traj_dir_slice.start
                dim = 3 if b % 3 == 0 else 2 if b % 2 == 0 else 0
                if dim > 0:
                    blk = y_raw[..., a:a + b].view(*y_raw.shape[:-1], b // dim, dim)
                    blk = torch.nn.functional.normalize(blk, dim=-1, eps=1e-9)
                    y_pre = y_pre.clone()
                    y_pre[..., a:a + b] = blk.reshape_as(y_raw[..., a:a + b])
        except Exception as _err:
            print(f"[Norm-ERR] trajectory direction norm_y failed: {_err}")

        if self.mu_y is not None and self.std_y is not None:
            mu = self._match_tensor('mu_y', self.mu_y, y_raw)
            std = self._match_tensor('std_y', self.std_y, y_raw).clamp_min(1e-6)
            return (y_pre - mu) / std
        return y_pre

    @classmethod
    def from_bundle(cls, bundle: dict):
        get = bundle.get
        def key2slice(d, key):
            if d is None: return None
            meta = d.get(key) if isinstance(d, dict) else None
            if meta is None: return None
            if isinstance(meta, dict):
                st, sz = int(meta.get('start', 0)), int(meta.get('size', 0))
                return slice(st, st+sz) if sz > 0 else None
            if isinstance(meta, (list, tuple)) and len(meta) >= 2:
                st, sz = int(meta[0]), int(meta[1])
                return slice(st, st+sz)
            return None
        s_layout = get('state_layout') or {}
        o_layout = get('output_layout') or {}
        return cls(
            mu_x = get('MuX'), std_x = get('StdX'),
            mu_y = get('MuY'), std_y = get('StdY'),
            y_to_x_map = get('y_to_x_map', []),
            yaw_x_slice     = key2slice(s_layout, 'RootYaw') or key2slice(s_layout, 'Yaw'),
            yaw_y_slice     = key2slice(o_layout, 'RootYaw') or key2slice(o_layout, 'Yaw'),
            rootvel_x_slice = key2slice(s_layout, 'RootVelocity'),
            rootvel_y_slice = key2slice(o_layout, 'RootVelocity'),
            angvel_x_slice  = key2slice(s_layout, 'BoneAngularVelocities'),
            angvel_y_slice  = key2slice(o_layout, 'BoneAngularVelocities'),
            tanh_scales_rootvel = get('tanh_scales_rootvel'),
            tanh_scales_angvel  = get('tanh_scales_angvel'),
            traj_dir_slice = key2slice(o_layout, 'TrajectoryDir'),
        )


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


