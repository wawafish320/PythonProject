#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UE JSON → NPZ 转换工具（v4）
兼容“局部坐标增强 + 参数缩减”的新版 JSON（单位：米，local_6d，支持 RootVelocityXY、FootEvidence、可选缺失字段），
同时兼容旧版 JSON（RootVelocity、Contacts、BonePositions/BoneVelocities/BoneAngularVelocities/Phase 等）。

主要改动：
 - RootVelocity 可自动读取 RootVelocityXY，仅提取 XY 两个分量
 - Contacts 已移除；训练侧需从 JSON 的 FootEvidence 重建软/硬接触
 - TrajectoryPos 缺失时自动补零（与 TrajectoryDir 使用相同的 K）
 - Phase 缺失时默认不写入，可通过 --use-phase 强制写入 0
 - X 特征按实际存在的通道动态打包，state_layout_json 会与之完全对齐
 - **RootYaw 已废弃**：不再写入特征，也不在模板/layout 中保留对应切片
 - 保持旧版 CLI 兼容，同时新增若干控制开关

依赖：Python 3.8+，仅依赖标准库与 numpy
"""
import os, sys, json, argparse, math

# --- canonicalize state_layout keys for training-time alignment ---
def _canon_state_layout(d: dict) -> dict:
    mapping = {
        "root_pos": "RootPosition",
        "root_vel": "RootVelocity",
        "rot6d": "BoneRotations6D",
        "angvel": "BoneAngularVelocities",
        "RootPosition": "RootPosition",
        "RootVelocity": "RootVelocity",
        "BoneRotations6D": "BoneRotations6D",
        "BoneAngularVelocities": "BoneAngularVelocities",
    }
    out = {}
    for k, v in (d or {}).items():
        if v is None:
            continue
        try:
            st, ed = int(v[0]), int(v[1])
        except Exception:
            # Fallback for dict form {'start':..., 'size':...} if ever used
            try:
                st = int(v.get("start", v.get("offset", 0)))
                ln = int(v.get("size", v.get("length", 0)))
                ed = st + ln
            except Exception:
                continue
        out[mapping.get(k, k)] = [st, ed]
    return out
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# ===== Embedded: Adaptive Grouped Normalizer (no defaults, data-driven) =====
from dataclasses import dataclass, asdict, field
import numpy as _np
import re as _re

def _wrap_to_pi(x: _np.ndarray) -> _np.ndarray:
    y = (x + _np.pi) % (2.0*_np.pi) - _np.pi
    y[(y == -_np.pi)] = _np.pi
    return y

def _robust_mu_std(arr: _np.ndarray):
    q1 = _np.percentile(arr, 25, axis=0); q3 = _np.percentile(arr, 75, axis=0)
    iqr = q3 - q1
    std = iqr / 1.349
    mu  = _np.median(arr, axis=0)
    std = _np.clip(_np.nan_to_num(std, copy=False, nan=1e-6, posinf=1e6, neginf=1e6), 1e-6, None)
    mu  = _np.nan_to_num(mu,  copy=False, nan=0.0,  posinf=0.0, neginf=0.0)
    return mu.astype(_np.float32), std.astype(_np.float32)

def _atanh_safe(x: _np.ndarray) -> _np.ndarray:
    return _np.arctanh(_np.clip(x, -1.0+1e-6, 1.0-1e-6))

def _tanh_compress(x: _np.ndarray, scale: _np.ndarray) -> _np.ndarray:
    return _np.tanh(x / _np.clip(scale, 1e-6, None))

def _tanh_decompress(y: _np.ndarray, scale: _np.ndarray) -> _np.ndarray:
    return _atanh_safe(y) * _np.clip(scale, 1e-6, None)

def _regex_any(s: str, pats):
    s = s.lower()
    return any(_re.search(p, s) for p in pats)

@dataclass
class _Spans:
    root_pos: tuple[int,int] | None = None
    root_vel: tuple[int,int] | None = None
    rot6d:    tuple[int,int] | None = None
    angvel:   tuple[int,int] | None = None

@dataclass
class _ConfigGN:
    rot6d_floor: float | None = None
    other_floor: float | None = None
    angvel_floor: float | None = 0.02
    group_prior_alpha: float | None = None
    use_group_prior_median: bool = True
    tanh_percentile_low: float = 2.5
    tanh_percentile_high: float = 97.5
    alpha_solver: str = 'kneedle'      # or 'quantile'
    alpha_quantile: float = 0.10
    numeric_eps: float = 1e-6

@dataclass
class _TemplateGN:
    bundle_version: int = 2
    notes: str = "Grouped Normalizer (adaptive)"
    MuX: list[float] = field(default_factory=list)
    StdX: list[float] = field(default_factory=list)
    MuY: list[float] = field(default_factory=list)
    StdY: list[float] = field(default_factory=list)
    tanh_scales_rootvel: list[float] | None = None
    tanh_scales_angvel:  list[float] | None = None
    cond_norm_config: dict = field(default_factory=dict)
    group_priors_rot6d: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    s_eff_X: list[float] | None = None
    s_eff_Y: list[float] | None = None
    s_eff_meta: dict = field(default_factory=dict)

class GroupedNormalizer:
    def __init__(self, cfg: _ConfigGN = _ConfigGN()):
        self.cfg = cfg
        self.template = _TemplateGN()
        self._source_meta = None
        self._first_npz_meta = None
        self._first_npz = None
        self._fitted = False
        self._spans: _Spans | None = None
        self._bone_names: list[str] | None = None
        self._rot6d_groups: dict[str, list[int]] = {}
        self._rot6d_group_prior: _np.ndarray | None = None
        self._tanh_scales_rootvel: _np.ndarray | None = None
        self._tanh_scales_angvel: _np.ndarray | None = None
        self._resolved_alpha: float | None = None

    # --------- public API ---------
    def fit(self, npz_paths: list[str]) -> None:
        Xs, Ys, span0, bone_names = [], [], None, None
        out_layout = {}

        for p in npz_paths:
            with _np.load(p, allow_pickle=True) as d:
                if self._first_npz is None:
                    self._first_npz = p
                    try:
                        sj = d.get('source_json', None)
                        if sj is not None:
                            if hasattr(sj, 'item'):
                                sj = sj.item()
                            if isinstance(sj, (bytes, bytearray)):
                                sj = sj.decode('utf-8', errors='ignore')
                            src_json_path = str(sj)
                            try:
                                with open(src_json_path, 'r', encoding='utf-8') as _f:
                                    _raw = json.load(_f)
                                    _meta0 = (_raw.get('meta') or {}) if isinstance(_raw, dict) else {}
                                    keep_keys = [
                                        'state_layout', 'output_layout', 'alignment', 'y_to_x_map', 'bone_names',
                                        'adaptive',
                                        'units', 'handedness', 'axes', 'spaces', 'rot6d_spec', 'skeleton',
                                        'root_motion', 'trajectory', 'asset_to_ue', 'action_bits'
                                    ]
                                    self._source_meta = {k: _meta0.get(k) for k in keep_keys if k in _meta0}
                            except Exception:
                                self._source_meta = None
                    except Exception:
                        pass
                    meta_npz = d.get('meta_json')
                    if meta_npz is not None:
                        try:
                            if hasattr(meta_npz, 'item'):
                                meta_npz = meta_npz.item()
                            if isinstance(meta_npz, (bytes, bytearray)):
                                meta_npz = meta_npz.decode('utf-8', errors='ignore')
                            if isinstance(meta_npz, str):
                                self._first_npz_meta = json.loads(meta_npz)
                        except Exception:
                            self._first_npz_meta = None
                X = d['X_flat'].astype(_np.float32)
                Y = d['y_out_features'].astype(_np.float32)
                if not out_layout:
                    out_layout = self._load_output_layout(d)
                if span0 is None:
                    span0 = self._parse_spans(d)
                if bone_names is None and 'bone_names' in d:
                    try:
                        bn = d['bone_names']
                        bone_names = [str(x) for x in (bn.tolist() if hasattr(bn, 'tolist') else list(bn))]
                    except Exception:
                        bone_names = None
            Xs.append(X);
            Ys.append(Y)

        # 兜底：若 spans 缺失，则按导出端默认顺序合成，保证模板可写
        if span0 is None or all(
                getattr(span0, k) is None for k in ['root_pos', 'root_vel', 'rot6d', 'angvel']):
            Xdim0 = int((Xs[0].shape[1]) if Xs else 0)
            span0 = self._synthesize_spans(out_layout, Xdim0, include_yaw=False)

        self._spans = span0
        self._bone_names = bone_names or []

        Xcat = _np.concatenate(Xs, axis=0)
        Ycat = _np.concatenate(Ys, axis=0)

        # 在 pre-transform 域做鲁棒统计（其中 yaw 会 unit-π，Root/AngVel 可能 tanh 压缩）
        Xprep, _ = self._pre_transform_X(Xcat, span0)
        Yprep, _ = self._pre_transform_Y(Ycat)

        muX, stdX = _robust_mu_std(Xprep)
        muY, stdY = _robust_mu_std(Yprep)

        # Rot6D：分组 prior + 自适应 α floor（仍在 pre-transform 域；Rot6D本身未做unit-π/tanh）
        if span0.rot6d is not None:
            st, ed = span0.rot6d
            ln = ed - st
            B = ln // 6
            self._rot6d_groups = self._build_rot6d_groups(self._bone_names, B)
            prior = self._compute_group_prior(Xprep[:, st:st + ln], B)
            self._rot6d_group_prior = prior.astype(_np.float32)

            ratios = stdX[st:st + ln] / _np.maximum(prior, self.cfg.numeric_eps)
            alpha = (self.cfg.group_prior_alpha
                     if self.cfg.group_prior_alpha is not None
                     else self._solve_alpha_from_ratios(ratios))
            self._resolved_alpha = float(alpha)

            floor_vec = alpha * prior
            if self.cfg.rot6d_floor is not None:
                floor_vec = _np.maximum(floor_vec, float(self.cfg.rot6d_floor))
            stdX[st:st + ln] = _np.maximum(stdX[st:st + ln], floor_vec)

        # 其它通道 floor（仍在 pre-transform 域）
        stdX = self._apply_other_floors(stdX, span0)
        if span0.angvel is not None:
            st, ed = span0.angvel
            if ed > st:
                stdX[st:ed] = 1.0

        stdY = self._apply_other_floors(stdY, self._span_from_Y_like(Ycat.shape[1], span0))

        # 写模板
        self.template.MuX = muX.tolist();
        self.template.StdX = stdX.tolist()
        self.template.MuY = muY.tolist();
        self.template.StdY = stdY.tolist()
        self.template.tanh_scales_rootvel = (
            self._tanh_scales_rootvel.tolist() if self._tanh_scales_rootvel is not None else None)
        self.template.tanh_scales_angvel = (
            self._tanh_scales_angvel.tolist() if self._tanh_scales_angvel is not None else None)
        self.template.cond_norm_config = self._emit_floor_config(span0)
        self.template.group_priors_rot6d = self._emit_group_priors()
        self.template.meta = dict(
            state_layout=_canon_state_layout(self._spans.__dict__),
            output_layout=out_layout,
            alignment=dict(
                x_to_y=dict(mode='next', offset=1),
                cond_to_y=dict(mode='match_y', offset=1),
            ),
            y_to_x_map=self._build_y_to_x_map(self._spans.__dict__, out_layout),
            bone_names=self._bone_names or [],
            adaptive=dict(
                mode='adaptive' if self.cfg.group_prior_alpha is None else 'fixed',
                alpha_solver=self.cfg.alpha_solver,
                resolved_alpha=float(self._resolved_alpha if self._resolved_alpha is not None else -1.0),
                rot6d_floor=self.cfg.rot6d_floor,
                other_floor=self.cfg.other_floor,
                angvel_floor=self.cfg.angvel_floor,
            )
        )
        def _merge_meta(src: dict):
            if not isinstance(src, dict):
                return
            for _k, _v in src.items():
                if _k not in self.template.meta:
                    self.template.meta[_k] = _v
                    continue
                tv = self.template.meta.get(_k)
                if isinstance(tv, dict) and isinstance(_v, dict):
                    merged = dict(tv)
                    for _sk, _sv in _v.items():
                        if _sk not in merged:
                            merged[_sk] = _sv
                    self.template.meta[_k] = merged

        if self._first_npz_meta:
            _merge_meta(self._first_npz_meta)
        # 若存在源 JSON 的 meta，做一次“补齐式合并”：保留我们这里计算出的键，补齐缺失键。
        if self._source_meta:
            _merge_meta(self._source_meta)
        self._fitted = True

    def normalize_X(self, X: _np.ndarray, meta: dict) -> _np.ndarray:
        self._assert_fitted()
        spans = self._spans or self._parse_spans_from_meta(meta)
        Xprep, _ = self._pre_transform_X(X, spans)
        mu = _np.asarray(self.template.MuX, dtype=_np.float32)
        std= _np.asarray(self.template.StdX, dtype=_np.float32)
        return (Xprep - mu) / _np.clip(std, self.cfg.numeric_eps, None)

    def denormalize_X(self, Xn: _np.ndarray, meta: dict) -> _np.ndarray:
        self._assert_fitted()
        spans = self._spans or self._parse_spans_from_meta(meta)
        mu = _np.asarray(self.template.MuX, dtype=_np.float32)
        std= _np.asarray(self.template.StdX, dtype=_np.float32)
        Xprep = Xn * _np.clip(std, self.cfg.numeric_eps, None) + mu
        return self._post_inverse_X(Xprep, spans)

    def normalize_Y(self, Y: _np.ndarray) -> _np.ndarray:
        self._assert_fitted()
        Yprep, _ = self._pre_transform_Y(Y)
        mu = _np.asarray(self.template.MuY, dtype=_np.float32)
        std= _np.asarray(self.template.StdY, dtype=_np.float32)
        return (Yprep - mu) / _np.clip(std, self.cfg.numeric_eps, None)

    def denormalize_Y(self, Yn: _np.ndarray) -> _np.ndarray:
        self._assert_fitted()
        mu = _np.asarray(self.template.MuY, dtype=_np.float32)
        std= _np.asarray(self.template.StdY, dtype=_np.float32)
        Yprep = Yn * _np.clip(std, self.cfg.numeric_eps, None) + mu
        return self._post_inverse_Y(Yprep)

    
    
    def save_template(self, out_json: str) -> None:
        d = asdict(self.template)
        # Normalize meta layouts to explicit {"start","size"}
        meta = d.get('meta', {}) or {}
        out_layout = meta.get('output_layout', meta.get('out_layout', {})) or {}
        state_layout = meta.get('state_layout', {}) or {}
        def to_explicit(m):
            out = {}
            for k, v in (m or {}).items():
                try:
                    # v like [start,end]
                    s, e = int(v[0]), int(v[1])
                    out[k] = {'start': s, 'size': max(0, e - s)}
                except Exception:
                    if isinstance(v, dict) and 'start' in v and ('size' in v or 'end' in v):
                        s = int(v.get('start', 0))
                        if 'size' in v:
                            l = int(v['size']); out[k] = {'start': s, 'size': max(0, l)}
                        else:
                            e = int(v.get('end', s)); out[k] = {'start': s, 'size': max(0, e - s)}
            return out
        state_layout = to_explicit(state_layout)
        out_layout = to_explicit(out_layout)
        # Materialize missing BoneRotations6D in output_layout
        if 'BoneRotations6D' not in out_layout:
            Dy = int(len(d.get('MuY', []) or d.get('StdY', [])) or 276)
            out_layout['BoneRotations6D'] = {'start': 0, 'size': Dy}
            print(f"[TemplateGuard] output_layout missing BoneRotations6D -> filled with {{'start':0,'size':{Dy}}}")
        meta['state_layout'] = state_layout
        meta['output_layout'] = out_layout
        if 'out_layout' in meta:
            del meta['out_layout']
        d['meta'] = meta

        # ---- Precompute s_eff_X / s_eff_Y for training-time loss weighting ----
        try:
            Dx = len(d.get('StdX', []) or [])
            Dy = len(d.get('StdY', []) or [])
            s_eff_x = np.asarray(d.get('StdX', []), dtype=np.float32)
            s_eff_y = np.asarray(d.get('StdY', []), dtype=np.float32)
            meta2 = meta  # explicit {'start','size'}
            spans = meta2.get('state_layout', {})
            # yaw
            # root vel tanh scales
            rv_sc = d.get('tanh_scales_rootvel', None)
            rv_ok = False
            if rv_sc is not None and ('RootVelocity' in spans or 'root_vel' in spans):
                st = (spans.get('RootVelocity') or spans.get('root_vel'))['start']
                ln = (spans.get('RootVelocity') or spans.get('root_vel'))['size']
                if len(rv_sc) == ln:
                    s_eff_x[st:st+ln] = s_eff_x[st:st+ln] * np.asarray(rv_sc, dtype=np.float32)
                    rv_ok = True
            # ang vel tanh scales
            av_sc = d.get('tanh_scales_angvel', None)
            av_ok = False
            if av_sc is not None and ('BoneAngularVelocities' in spans or 'angvel' in spans):
                st = (spans.get('BoneAngularVelocities') or spans.get('angvel'))['start']
                ln = (spans.get('BoneAngularVelocities') or spans.get('angvel'))['size']
                if len(av_sc) == ln:
                    s_eff_x[st:st+ln] = s_eff_x[st:st+ln] * np.asarray(av_sc, dtype=np.float32)
                    av_ok = True
            d['s_eff_X'] = s_eff_x.astype(np.float32).tolist()
            d['s_eff_Y'] = s_eff_y.astype(np.float32).tolist()
            # brief diagnostics
            if Dx > 0 and Dy > 0:
                import numpy as _np
                def _mm(a): 
                    return float(a.min()), float(_np.median(a)), float(a.max())
                mn, md, mx = _mm(s_eff_y)
                print(f"[s_eff][Y] min={mn:.3g} med={md:.5g} max={mx:.3g}")
                mn, md, mx = _mm(s_eff_x)
                print(f"[s_eff][X] min={mn:.3g} med={md:.5g} max={mx:.3g}")
        except Exception as _e:
            print(f"[s_eff][WARN] failed to compute s_eff: {_e}")

        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

        def to_explicit(m):
            out = {}
            for k, v in (m or {}).items():
                try:
                    # v like [start,end]
                    s, e = int(v[0]), int(v[1])
                    out[k] = {'start': s, 'size': max(0, e - s)}
                except Exception:
                    if isinstance(v, dict) and 'start' in v and ('size' in v or 'end' in v):
                        s = int(v.get('start', 0))
                        if 'size' in v:
                            l = int(v['size']); out[k] = {'start': s, 'size': max(0, l)}
                        else:
                            e = int(v.get('end', s)); out[k] = {'start': s, 'size': max(0, e - s)}
            return out
        meta['state_layout'] = to_explicit(state_layout)
        meta['output_layout'] = to_explicit(out_layout)
        # remove legacy alias if present
        if 'out_layout' in meta:
            del meta['out_layout']
        d['meta'] = meta
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        # --------- internals ---------
    def _assert_fitted(self):
        if not self._fitted: raise RuntimeError("GroupedNormalizer 尚未 fit()")

    @staticmethod
    def _parse_spans(d) -> _Spans:
        s = d.get('state_layout_json', None)
        if s is None: return _Spans()


    def _synthesize_spans(self, out_layout: dict, Xdim: int, include_yaw: bool = True) -> _Spans:
        """当 state_layout_json 缺失时，按导出端的默认顺序合成 spans。
        默认顺序: RootPosition(3) → RootVelocity(2) → BoneRotations6D(B*6) → BoneAngularVelocities(B*3)
        其中 B 从 output_layout['BoneRotations6D'][1]//6 推断；若缺失则根据 Xdim 反解。
        """
        st = 0
        root_pos = (st, st+3); st += 3
        root_vel = (st, st+2); st += 2
        root_yaw = (st, st+1) if include_yaw else None
        if include_yaw: st += 1
        # rot6d size:
        rot6d_size = 0
        if out_layout and 'BoneRotations6D' in out_layout:
            rot6d_size = int(out_layout['BoneRotations6D'][1])
        else:
            # 尝试从 Xdim 反推：剩余应为 rot6d + angvel
            # 先假设 B 与 6D 整除（B*6），并 angvel 是 B*3
            # 令 rem = Xdim - st = B*(6+3) => B = rem / 9
            rem = max(0, int(Xdim) - st)
            B = max(0, rem // 9)
            rot6d_size = B * 6
        rot6d = (st, st+rot6d_size); st += rot6d_size
        # angvel size: 3 per bone
        rem = max(0, int(Xdim) - st)
        angvel_size = (rot6d_size//6) * 3 if rot6d_size>0 else (rem if rem%3==0 else 0)
        angvel = (st, st+angvel_size) if angvel_size>0 else None
        # 统一返回为 _Spans
        to_span = lambda t: (int(t[0]), int(t[1])) if (t is not None and (t[1]>t[0])) else None
        return _Spans(
            root_pos=to_span(root_pos),
            root_vel=to_span(root_vel),
                        rot6d=to_span(rot6d),
            angvel=to_span(angvel)
        )



    @staticmethod
    def _load_output_layout(d) -> dict:
        """Try to read 'output_layout_json' from a loaded npz object 'd'.
        Accepts either {"start","size"} dicts or [start, end]/[start, size] lists.
        Returns a normalized dict where values are [start, end).
        """
        s = d.get('output_layout_json', None)
        if s is None:
            return {}
        if hasattr(s, 'item'):
            s = s.item()
        if isinstance(s, (bytes, bytearray)):
            try:
                s = s.decode('utf-8')
            except Exception:
                s = str(s)
        try:
            j = json.loads(s)
            out = {}
            for k, v in (j or {}).items():
                if isinstance(v, dict) and 'start' in v and ('size' in v or 'end' in v):
                    s0 = int(v.get('start', 0))
                    if 'size' in v:
                        ln = int(v['size']);
                        out[k] = [s0, s0 + ln]
                    else:
                        e = int(v['end']);
                        out[k] = [s0, e]
                elif isinstance(v, (list, tuple)) and len(v) == 2:
                    a, b = int(v[0]), int(v[1])
                    out[k] = [a, b] if b > a else [a, a + b]
            return out
        except Exception:
            return {}

    @staticmethod
    def _build_y_to_x_map(x_layout: dict, y_layout: dict) -> list[dict]:
        """Produce a coarse name-based alignment map used for sanity checks/visualization."""
        names = ["RootVelocity","BoneAngularVelocities","BoneRotations6D","RootPosition"]
        y2x = []
        for n in names:
            if n in x_layout and n in y_layout:
                xs, xsz = int(x_layout[n][0]), int(x_layout[n][1])
                ys, ysz = int(y_layout[n][0]), int(y_layout[n][1])
                k = min(int(xsz), int(ysz))
                if k > 0:
                    y2x.append(dict(name=n, x_start=xs, x_size=k, y_start=ys, y_size=k))
        return y2x

        if hasattr(s, 'item'): s = s.item()
        if isinstance(s, bytes): s = s.decode('utf-8')
        j = json.loads(s)
        g = lambda k: tuple(j[k]) if (k in j and len(j[k])==2 and (j[k][1]>j[k][0])) else None
        return _Spans(
            root_pos=g('RootPosition'), root_vel=g('RootVelocity'),
            rot6d=g('BoneRotations6D'), angvel=g('BoneAngularVelocities')
        )

    @staticmethod
    def _parse_spans_from_meta(meta: dict) -> _Spans:
        j = meta.get('state_layout', {})
        g = lambda k: tuple(j[k]) if (k in j and len(j[k])==2 and (j[k][1]>j[k][0])) else None
        return _Spans(
            root_pos=g('RootPosition'), root_vel=g('RootVelocity'),
            rot6d=g('BoneRotations6D'), angvel=g('BoneAngularVelocities')
        )

    def _span_from_Y_like(self, Dy: int, s: _Spans) -> _Spans:
        rs = s.rot6d if s.rot6d is not None else (0, Dy)
        return _Spans(rot6d=rs)

    def _pre_transform_X(self, X: _np.ndarray, s: _Spans):
        X = X.astype(_np.float32, copy=True); meta = {}
        if s.root_vel is not None:
            st, ed = s.root_vel; rv = X[:, st:ed]
            if self._tanh_scales_rootvel is None:
                low = _np.percentile(rv, self.cfg.tanh_percentile_low, axis=0)
                high= _np.percentile(rv, self.cfg.tanh_percentile_high, axis=0)
                scale = _np.maximum((high - low)*0.5, 1e-3)
                self._tanh_scales_rootvel = scale.astype(_np.float32)
            X[:, st:ed] = _tanh_compress(rv, self._tanh_scales_rootvel)
        if s.angvel is not None:
            st, ed = s.angvel
            if ed > st:
                av = X[:, st:ed]
                if self._tanh_scales_angvel is None:
                    low = _np.percentile(av, self.cfg.tanh_percentile_low, axis=0)
                    high= _np.percentile(av, self.cfg.tanh_percentile_high, axis=0)
                    scale = _np.maximum((high - low)*0.5, 1e-3)
                    self._tanh_scales_angvel = scale.astype(_np.float32)
                X[:, st:ed] = _tanh_compress(av, self._tanh_scales_angvel)
        return X, meta

    def _pre_transform_Y(self, Y: _np.ndarray):
        return Y.astype(_np.float32, copy=True), {}

    def _post_inverse_X(self, Xprep: _np.ndarray, s: _Spans) -> _np.ndarray:
        X = Xprep.astype(_np.float32, copy=True)
        if s.angvel is not None and self._tanh_scales_angvel is not None:
            st, ed = s.angvel; X[:, st:ed] = _tanh_decompress(X[:, st:ed], self._tanh_scales_angvel)
        if s.root_vel is not None and self._tanh_scales_rootvel is not None:
            st, ed = s.root_vel; X[:, st:ed] = _tanh_decompress(X[:, st:ed], self._tanh_scales_rootvel)
        return X

    def _post_inverse_Y(self, Yprep: _np.ndarray) -> _np.ndarray:
        return Yprep.astype(_np.float32, copy=True)

    def _build_rot6d_groups(self, bone_names: list[str], B: int) -> dict[str, list[int]]:
        groups = {'pelvis':[], 'spine':[], 'upper_limb':[], 'lower_limb':[], 'hand':[], 'finger':[], 'twist':[], 'foot':[], 'other':[]}
        if not bone_names or len(bone_names) < B:
            for i in range(B): groups[f'bone_{i:02d}'] = [i]
            return groups
        def assign(i: int, name: str):
            n = name.lower()
            if _regex_any(n, [r'pelvis', r'hip(?!.*(left|right))', r'root']): groups['pelvis'].append(i)
            elif _regex_any(n, [r'spine', r'chest', r'neck', r'head']): groups['spine'].append(i)
            elif _regex_any(n, [r'clav', r'shoulder', r'upperarm', r'arm(?!.*fore)', r'humerus']): groups['upper_limb'].append(i)
            elif _regex_any(n, [r'forearm', r'lowerarm', r'ulna', r'radius']): groups['lower_limb'].append(i)
            elif _regex_any(n, [r'hand(?!.*(twist|roll))']): groups['hand'].append(i)
            elif _regex_any(n, [r'thumb', r'index', r'middle', r'ring', r'pinky', r'finger', r'metacarp']): groups['finger'].append(i)
            elif _regex_any(n, [r'twist', r'roll']): groups['twist'].append(i)
            elif _regex_any(n, [r'thigh', r'upleg', r'calf', r'lowerleg', r'shin', r'leg', r'foot', r'toe', r'ball']): groups['foot'].append(i)
            else: groups['other'].append(i)
        for i in range(B):
            nm = bone_names[i] if i < len(bone_names) else f'B{i}'
            assign(i, nm)
        return {k:v for k,v in groups.items() if len(v)>0}

    def _compute_group_prior(self, rot6d_X: _np.ndarray, B: int) -> _np.ndarray:
        per_bone_std = rot6d_X.reshape(-1, B, 6).std(axis=0)  # [B,6]
        prior = _np.zeros((B,6), dtype=_np.float32)
        for gname, idxs in self._rot6d_groups.items():
            s = per_bone_std[idxs, :]  # [n_g,6]
            if s.size == 0: continue
            gp = _np.median(s, axis=0) if self.cfg.use_group_prior_median else _np.mean(s, axis=0)
            gp = _np.maximum(gp, self.cfg.numeric_eps)
            for i in idxs: prior[i, :] = gp
        prior = _np.maximum(prior, self.cfg.numeric_eps)
        return prior.reshape(-1)

    def _solve_alpha_from_ratios(self, ratios: _np.ndarray) -> float:
        if self.cfg.group_prior_alpha is not None: return float(self.cfg.group_prior_alpha)
        r = _np.asarray(ratios, dtype=_np.float64); r = r[_np.isfinite(r)]
        if r.size == 0: return 0.0
        r = _np.sort(r)
        if self.cfg.alpha_solver == 'quantile':
            q = float(_np.clip(self.cfg.alpha_quantile, 0.0, 1.0)); return float(_np.quantile(r, q))
        y = (r - r[0]) / max(r[-1] - r[0], self.cfg.numeric_eps)
        n = y.size
        if n < 3: return float(_np.median(r))
        x = _np.linspace(0.0, 1.0, n); diff = x - y
        i = int(_np.argmax(diff)); return float(r[i])

    def _apply_other_floors(self, std: _np.ndarray, s: _Spans) -> _np.ndarray:
        std = std.copy()
        if s.angvel is not None and (self.cfg.angvel_floor is not None):
            st, ed = s.angvel; std[st:ed] = _np.maximum(std[st:ed], float(self.cfg.angvel_floor))
        if s.root_pos is not None and (self.cfg.other_floor is not None):
            st, ed = s.root_pos; std[st:ed] = _np.maximum(std[st:ed], float(self.cfg.other_floor))
        return std

    def _emit_floor_config(self, s: _Spans) -> dict:
        return dict(
            mode="Adaptive",
            start=0, size=None,
            floor_rot6d=dict(strategy="alpha*group_prior",
                             alpha=float(self._resolved_alpha if self._resolved_alpha is not None else -1.0),
                             dims=int((s.rot6d[1]-s.rot6d[0]) if s.rot6d else 0)),
            floor_other=dict(value=self.cfg.other_floor, dims=int((s.root_pos[1]-s.root_pos[0]) if s.root_pos else 0)),
                        floor_angvel=dict(value=self.cfg.angvel_floor, dims=int((s.angvel[1]-s.angvel[0]) if s.angvel else 0)),
        )

    def _emit_group_priors(self) -> dict:
        if self._rot6d_group_prior is None: return {}
        return {'groups': self._rot6d_groups, 'prior_per_dim': self._rot6d_group_prior.tolist()}

# --- helpers to export normalized merged dataset ---
def _merge_normalized_dataset(npz_paths: list[str], gn: GroupedNormalizer, out_path: str) -> dict:
    Xn_list, Yn_list, C_list, meta_list = [], [], [], []
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as d:
            X = d['X_flat'].astype(np.float32)
            Y = d['y_out_features'].astype(np.float32)
            C = d.get('cond_in', None)
            if C is None:
                C = np.zeros((X.shape[0]-1, 0), dtype=np.float32)
            # x[t] -> y[t+1] 对齐
            X = X[:-1]; Y = Y[1:]
            layout = d['state_layout_json'].item() if hasattr(d['state_layout_json'], 'item') else d['state_layout_json']
            layout = layout.decode('utf-8') if isinstance(layout, bytes) else layout
            Xn = gn.normalize_X(X, meta={'state_layout': json.loads(layout)})
            Yn = gn.normalize_Y(Y)
            Xn_list.append(Xn); Yn_list.append(Yn); C_list.append(C[:-1])
            meta_list.append(dict(
                FPS=int(d.get('FPS', 60)),
                source=str(d.get('source_json', '')),
                state_layout=str(layout),
                output_layout=str(d.get('output_layout_json', ''))
            ))
    Xn = np.concatenate(Xn_list, axis=0)
    Yn = np.concatenate(Yn_list, axis=0)
    C  = np.concatenate(C_list, axis=0)
    np.savez_compressed(out_path, X_norm=Xn.astype(np.float32), Y_norm=Yn.astype(np.float32), C=C.astype(np.float32),
                        meta=np.array(meta_list, dtype=object))
    return dict(N=len(npz_paths), T=int(Xn.shape[0]), Dx=int(Xn.shape[1]), Dy=int(Yn.shape[1]))
# ===== End Embedded Normalizer =====



def find_json_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path] if path.lower().endswith(".json") else []
    hits = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".json"):
                hits.append(os.path.join(root, f))
    hits.sort()
    return hits

def np_array(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype)

def gram_schmidt_renorm(rot6d: np.ndarray) -> np.ndarray:
    """
    rot6d: (..., 6). 列A = X, 列B = Z。做一次 GS，使两列单位正交。
    """
    orig_shape = rot6d.shape
    flat = rot6d.reshape(-1, 6)
    a = flat[:, 0:3]
    b = flat[:, 3:6]
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b - np.sum(a*b, axis=1, keepdims=True)*a
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    out = np.concatenate([a, b], axis=1).reshape(orig_shape)
    return out

def resample_series(arr: np.ndarray, new_T: int) -> np.ndarray:
    """对 (T, ...) 的连续序列做线性重采样。"""
    T = arr.shape[0]
    if new_T == T:
        return arr.copy()
    x_old = np.linspace(0.0, 1.0, T, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, new_T, dtype=np.float64)
    flat = arr.reshape(T, -1).astype(np.float64)
    out = np.empty((new_T, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(x_new, x_old, flat[:, c])
    return out.astype(np.float32).reshape((new_T,) + arr.shape[1:])

def resample_contacts(binary_arr: np.ndarray, new_T: int) -> np.ndarray:
    """对 (T, C) 的二值接触做重采样：线性插值后阈值化到 {0,1}。"""
    if new_T == binary_arr.shape[0]:
        return binary_arr.copy()
    cont = resample_series(binary_arr.astype(np.float32), new_T)
    return (cont >= 0.5).astype(np.uint8)

def sg_smooth_1d(x: np.ndarray) -> np.ndarray:
    """Savitzky-Golay (7点,2阶) 对 1D 做平滑；边界用 edge-clamp。"""
    coef = np.array([-2, 3, 6, 7, 6, 3, -2], dtype=np.float32) / 21.0
    T = x.shape[0]
    y = np.empty_like(x)
    for t in range(T):
        acc = 0.0
        for k in range(-3, 4):
            idx = min(max(t + k, 0), T - 1)
            acc += coef[k + 3] * x[idx]
        y[t] = acc
    return y

def sg_smooth_series(arr: np.ndarray) -> np.ndarray:
    """对 (T, C) 的序列逐列做 SG 平滑。"""
    T, C = arr.shape
    out = np.empty_like(arr)
    for c in range(C):
        out[:, c] = sg_smooth_1d(arr[:, c])
    return out

def sanitize_f32(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr

def _load_action_meta_and_clip_action(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ab = (data.get("meta", {}).get("action_bits") or {})
    enum = [str(s).strip() for s in ab.get("enum", [])]
    size = ab.get("size", None)
    if not enum or not size or int(size) < 2:
        enum = ["Idle","Walk","Run","Jump"]
        size = len(enum)
    else:
        size = int(size)
    clip_action = str(data.get("ClipAction") or data.get("clip_action") or "")
    return enum, size, clip_action

def _compute_planar_vel_dir_and_speed(clip: dict):
    # 优先用 root_vel；否则用 root_pos 做差分 * FPS
    if "root_vel" in clip and clip["root_vel"] is not None:
        v2 = np.asarray(clip["root_vel"], dtype=np.float32)[:, :2]
    else:
        fps = float(clip.get("FPS", 60.0))
        rp2 = np.asarray(clip["root_pos"], dtype=np.float32)[:, :2]
        v2 = np.zeros_like(rp2); v2[1:] = (rp2[1:] - rp2[:-1]) * fps
    speed = np.linalg.norm(v2, axis=1, keepdims=True).astype(np.float32)
    eps = 1e-8
    vel_dir = v2 / np.maximum(speed, eps)
    vel_dir[speed[:, 0] < 1e-3] = 0.0
    return vel_dir, speed


def _rotate_dir_deg(v2: np.ndarray, deg: float) -> np.ndarray:
    """Rotate 2D vectors by given degrees about +Z."""
    if v2 is None:
        return None
    rad = np.deg2rad(float(deg))
    c, s = (np.cos(rad), np.sin(rad))
    x, y = v2[..., 0], v2[..., 1]
    rot = np.stack([c * x - s * y, s * x + c * y], axis=-1)
    return rot

def _infer_up_axis_from_meta(meta: dict | None) -> int:
    axes = (meta or {}).get("axes", {}) or {}
    for axis_name, idx in (("x", 0), ("y", 1), ("z", 2)):
        if str(axes.get(axis_name, "")).lower() == "up":
            return idx
    return 2

def _rot6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    """
    将 (...,6) 的 6D 表达还原为 (...,3,3) 旋转矩阵（列向量形式）。
    """
    a = rot6d[..., 0:3]
    b = rot6d[..., 3:6]
    a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8, None)
    c = np.cross(a, b, axis=-1)
    return np.stack([a, b, c], axis=-1)

def _wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def _nan_forward_fill(arr: np.ndarray) -> np.ndarray:
    mask = np.isfinite(arr)
    if not mask.any():
        arr[...] = 0.0
        return arr
    last = float(arr[mask][0])
    for idx in range(arr.shape[0]):
        if mask[idx]:
            last = float(arr[idx])
        else:
            arr[idx] = last
    return arr

def _compute_traj_cmd_dir_and_yaw(clip: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    选取 TrajectoryDir 中央前瞻点（若缺失则退化为 root_vel）作为 cond 指令方向，
    返回单位化后的 2D 向量以及平面 yaw（弧度）。
    """
    traj_dir = clip.get("traj_dir")
    T = int(clip.get("T", 0))
    source = "root_vel"
    if traj_dir is not None and traj_dir.shape[1] >= 2:
        K = traj_dir.shape[1] // 2
        center = K // 2
        dir_raw = traj_dir[:, center * 2:center * 2 + 2].astype(np.float32)
        source = "traj_dir"
    else:
        dir_raw, _ = _compute_planar_vel_dir_and_speed(clip)
    norm = np.linalg.norm(dir_raw, axis=1, keepdims=True)
    dir_unit = np.zeros_like(dir_raw, dtype=np.float32)
    valid = norm[:, 0] > 1e-6
    if valid.any():
        dir_unit[valid] = dir_raw[valid] / norm[valid]
    yaw = np.full((T,), np.nan, dtype=np.float32)
    yaw[valid] = np.arctan2(dir_unit[valid, 1], dir_unit[valid, 0])
    return dir_unit, yaw, source

def _align_pelvis_yaw(clip: Dict[str, Any], cmd_yaw: np.ndarray | None) -> Tuple[np.ndarray | None, Dict[str, float]]:
    """
    从根骨 6D 旋转推回骨盆朝向，并与 cond yaw 对齐，返回 (T,) yaw 序列以及诊断信息。
    """
    rot6d = clip.get("bone_rot6d")
    if rot6d is None or rot6d.size == 0:
        return None, {}
    root_rot = rot6d[:, 0, :].reshape(-1, 6)
    mats = _rot6d_to_matrix_np(root_rot).reshape(rot6d.shape[0], 3, 3)
    up_axis = _infer_up_axis_from_meta(clip.get("meta"))
    planar_axes = [ax for ax in (0, 1, 2) if ax != up_axis]
    if len(planar_axes) != 2:
        planar_axes = [0, 1]
    targets = []
    if cmd_yaw is not None:
        targets.append(cmd_yaw)
    traj_raw = clip.get("traj_yaw_raw")
    if isinstance(traj_raw, np.ndarray) and traj_raw.shape[0] == mats.shape[0]:
        targets.append(traj_raw.reshape(-1))

    def _score_axis(axis_idx: int):
        vec = mats[:, :, axis_idx]
        planar = vec[:, planar_axes]
        planar_norm = np.linalg.norm(planar, axis=1)
        yaw_axis = np.full((planar.shape[0],), np.nan, dtype=np.float32)
        mask = planar_norm > 1e-6
        yaw_axis[mask] = np.arctan2(planar[mask, 1], planar[mask, 0])
        errs = []
        offsets = []
        for tgt in targets:
            tgt = np.asarray(tgt, dtype=np.float32)
            valid = np.isfinite(tgt) & mask
            if valid.sum() < max(8, int(0.1 * len(tgt))):
                continue
            diff = _wrap_to_pi_np(yaw_axis[valid] - tgt[valid])
            errs.append(float(np.median(np.abs(diff))))
            offsets.append(float(np.median(diff)))
        if errs:
            return yaw_axis, float(np.median(errs)), float(np.median(offsets))
        return yaw_axis, None, None

    best_axis = 0
    best_offset = 0.0
    best_err = None
    best_series = None
    for axis_idx in range(3):
        yaw_axis, err, offset = _score_axis(axis_idx)
        if err is None:
            if best_series is None:
                best_series = yaw_axis
            continue
        if best_err is None or err < best_err:
            best_err = err
            best_offset = 0.0 if offset is None else offset
            best_axis = axis_idx
            best_series = yaw_axis
    if best_series is None:
        return None, {}
    yaw_aligned = _wrap_to_pi_np(best_series - best_offset)
    yaw_aligned = _nan_forward_fill(yaw_aligned)
    diag = {
        "forward_axis": int(best_axis),
        "yaw_offset_rad": float(best_offset),
        "median_err_deg": None if best_err is None else float(np.degrees(best_err)),
    }
    return yaw_aligned, diag

def _frame_get(fr: dict, key: str, default=None):
    return fr[key] if key in fr else default

# -----------------------------
# 解析单个 JSON Clip（新旧兼容）
# -----------------------------
def load_clip(json_path: str,
              use_phase_if_missing: bool = False,
              root_yaw_as_feature: bool = False) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = dict(data.get("meta", {}) or {})

    units = meta.get("units", "")
    if units != "meters":
        raise ValueError(f"{json_path}: expect meters, got units='{units}'")

    skel = meta.get("skeleton", {})
    bone_names = skel.get("bone_names", [])
    parents = skel.get("parents", [])
    B = len(bone_names)

    FPS = int(data.get("FPS", 60))
    frames = data.get("Frames", [])
    T = len(frames)
    if T == 0:
        raise ValueError("Empty Frames")

    # ---------- 逐帧提取 ----------
    # Phase（可缺省）
    if "Phase" in frames[0]:
        phase = np_array([_frame_get(fr, "Phase") for fr in frames])  # (T,2)
        has_phase = True
    else:
        has_phase = False
        if use_phase_if_missing:
            phase = np.zeros((T, 2), dtype=np.float32)
        else:
            phase = None

    # Trajectory：Dir 一定有；Pos 可能没有 -> 自动置零 (T,K*2)
    traj_dir_list = [_frame_get(fr, "TrajectoryDir", []) for fr in frames]
    K = len(traj_dir_list[0]) if traj_dir_list and isinstance(traj_dir_list[0], list) else 0
    traj_dir = np_array([sum(td, []) for td in traj_dir_list]) if K > 0 else np.zeros((T, 0), np.float32)

    if "TrajectoryPos" in frames[0]:
        traj_pos = np_array([sum(_frame_get(fr, "TrajectoryPos"), []) for fr in frames])
    else:
        traj_pos = np.zeros((T, K*2), dtype=np.float32) if K > 0 else np.zeros((T, 0), np.float32)

    # RootPosition / RootVelocity
    root_pos = np_array([_frame_get(fr, "RootPosition", [0,0,0]) for fr in frames])
    if "RootVelocity" in frames[0]:
        root_vel = np_array([_frame_get(fr, "RootVelocity") for fr in frames])[:, :2]
    else:
        # 新字段 RootVelocityXY
        root_vel = np_array([_frame_get(fr, "RootVelocityXY", [0,0]) for fr in frames])[:, :2]


    # Contacts removed: NOT exported; FootEvidence to be re-read from JSON at training time.
    contacts = None
    contacts_soft2 = None

    # Bones
    # 必有 Rot6D；其他可缺失
    bone_rot6d = np_array([sum(_frame_get(fr, "BoneRotations"), []) for fr in frames]).reshape(T, B, 6)
    bone_pos = None
    bone_vel = None
    bone_ang_vel = None
    if "BonePositions" in frames[0]:
        bone_pos = np_array([sum(_frame_get(fr, "BonePositions"), []) for fr in frames]).reshape(T, B, 3)
    if "BoneVelocities" in frames[0]:
        bone_vel = np_array([sum(_frame_get(fr, "BoneVelocities"), []) for fr in frames]).reshape(T, B, 3)
    if "BoneAngularVelocities" in frames[0]:
        bone_ang_vel = np_array([sum(_frame_get(fr, "BoneAngularVelocities"), []) for fr in frames]).reshape(T, B, 3)

    out = dict(
        path=json_path,
        FPS=FPS, T=T, B=B,
        bone_names=bone_names,
        parents=np_array(parents).astype(np.int32),
        phase=phase, has_phase=has_phase,
        traj_pos=traj_pos, traj_dir=traj_dir,
        root_pos=root_pos, root_vel=root_vel,         contacts=contacts, contacts_soft2=contacts_soft2,
        bone_pos=bone_pos, bone_rot6d=bone_rot6d, bone_vel=bone_vel, bone_ang_vel=bone_ang_vel,
        meta=meta,
    )
    return out

# -----------------------------
# 打包为扁平特征（动态选择）
# -----------------------------

def pack_flat_features(clip: Dict[str, Any],
                       include_phase: bool = False,
                       include_root_yaw: bool = False,
                       include_bone_pos: Optional[bool] = None,
                       include_lin_vel: Optional[bool] = None,
                       include_ang_vel: Optional[bool] = True) -> Tuple[np.ndarray, Dict[str, Tuple[int,int]]]:
    """
    返回 X_flat 以及对应的 state_layout（用于和训练端对齐）。
    注意：Contacts 已移除，不会写入特征或 NPZ。
    """
    T, B = clip["T"], clip["B"]
    parts: List[np.ndarray] = []
    layout: Dict[str, Tuple[int,int]] = {}
    off = 0

    def _append(name: str, arr: Optional[np.ndarray]):
        nonlocal off
        if arr is None or getattr(arr, "size", 0) == 0:
            return
        arr = arr.astype(np.float32)
        parts.append(arr)
        F = arr.shape[1]
        layout[name] = (off, off + F)
        off += F

    if include_phase and clip.get("phase") is not None:
        _append("Phase", clip["phase"])

    # # 轨迹
    # _append("TrajectoryPos", clip.get("traj_pos"))
    # _append("TrajectoryDir", clip.get("traj_dir"))

    # 根
    _append("RootPosition", clip.get("root_pos"))
    _append("RootVelocity", clip.get("root_vel"))

    # Contacts 移除：不再拼接到 X

    # 骨骼
    br6 = clip["bone_rot6d"].reshape(T, B*6)
    _append("BoneRotations6D", br6)

    def _decide(flag: Optional[bool], arr: Optional[np.ndarray]) -> bool:
        if flag is None:
            return arr is not None
        return bool(flag)

    if _decide(include_bone_pos, clip.get("bone_pos")):
        _append("BonePositions", clip["bone_pos"].reshape(T, B*3))
    if _decide(include_lin_vel, clip.get("bone_vel")):
        _append("BoneVelocities", clip["bone_vel"].reshape(T, B*3))
    if _decide(include_ang_vel, clip.get("bone_ang_vel")):
        _append("BoneAngularVelocities", clip["bone_ang_vel"].reshape(T, B*3))

    X = np.concatenate(parts, axis=1) if parts else np.zeros((T,0), dtype=np.float32)
    return X, layout
# -----------------------------
# 轨迹压缩：把 K 个采样点裁剪为指定的 L 个“前瞻点”
# -----------------------------
def _reduce_traj_channels(clip: Dict[str, Any], keep_idx: List[int]) -> None:
    """就地 clip["traj_pos"]/["traj_dir"] 仅保留指定前瞻点（以 2 列为 block）。"""
    if "traj_pos" not in clip or "traj_dir" not in clip:
        return
    K = clip["traj_pos"].shape[1] // 2
    if K <= 0:
        return
    uniq = sorted({int(i) for i in keep_idx if isinstance(i, (int, np.integer)) or (isinstance(i, str) and str(i).isdigit())})
    uniq = [i for i in uniq if 0 <= i < K]
    if not uniq:
        return
    def _slice_blocks(arr, idxs):
        pieces = [arr[:, 2*i:2*i+2] for i in idxs]
        return np.concatenate(pieces, axis=1) if pieces else arr
    clip["traj_pos"] = _slice_blocks(clip["traj_pos"], uniq).astype(np.float32)
    clip["traj_dir"] = _slice_blocks(clip["traj_dir"], uniq).astype(np.float32)

# -----------------------------
# 主流程：单文件转换
# -----------------------------
def convert_one(json_path: str,
                out_dir: str,
                target_fps: Optional[int] = None,
                smooth_vel: bool = False,
                traj_keep_idx: Optional[List[int]] = None,
                use_phase_if_missing: bool = False,
                include_root_yaw: bool = False,
                include_bone_pos: Optional[bool] = None,
                include_lin_vel: Optional[bool] = None,
                include_ang_vel: Optional[bool] = True) -> Dict[str, Any]:
    clip = load_clip(json_path, use_phase_if_missing, include_root_yaw)
    if clip.get("root_yaw") is not None:
        clip["traj_yaw_raw"] = clip["root_yaw"].copy()
    else:
        clip["traj_yaw_raw"] = None

    # 重采样
    if target_fps is not None and target_fps > 0 and target_fps != clip["FPS"]:
        ratio = target_fps / float(clip["FPS"])
        new_T = max(1, int(round(clip["T"] * ratio)))
        for k in ["phase", "traj_pos", "traj_dir", "root_pos", "root_vel", "root_yaw", "traj_yaw_raw"]:
            if clip.get(k) is not None and isinstance(clip[k], np.ndarray) and clip[k].ndim >= 2:
                clip[k] = resample_series(clip[k], new_T)
# bones
        T_old = clip["T"]; B = clip["B"]
        if clip["bone_pos"] is not None:
            clip["bone_pos"] = resample_series(clip["bone_pos"].reshape(T_old, -1), new_T).reshape(new_T, B, 3)
        if clip["bone_rot6d"] is not None:
            clip["bone_rot6d"] = resample_series(clip["bone_rot6d"].reshape(T_old, -1), new_T).reshape(new_T, B, 6)
            clip["bone_rot6d"] = gram_schmidt_renorm(clip["bone_rot6d"])
        if clip["bone_vel"] is not None:
            clip["bone_vel"] = resample_series(clip["bone_vel"].reshape(T_old, -1), new_T).reshape(new_T, B, 3)
        if clip["bone_ang_vel"] is not None:
            clip["bone_ang_vel"] = resample_series(clip["bone_ang_vel"].reshape(T_old, -1), new_T).reshape(new_T, B, 3)
        clip["FPS"] = target_fps
        clip["T"] = new_T

    # 轨迹压缩
    if traj_keep_idx:
        _reduce_traj_channels(clip, traj_keep_idx)

    # 可选速度平滑（root_vel / 骨骼平滑）
    if smooth_vel:
        T, B = clip["T"], clip["B"]
        if clip["root_vel"] is not None and clip["root_vel"].size > 0:
            clip["root_vel"] = sg_smooth_series(clip["root_vel"])
        if clip["bone_vel"] is not None:
            bv = clip["bone_vel"].reshape(T, -1)
            clip["bone_vel"] = sg_smooth_series(bv).reshape(T, B, 3)
        if clip["bone_ang_vel"] is not None:
            bav = clip["bone_ang_vel"].reshape(T, -1)
            clip["bone_ang_vel"] = sg_smooth_series(bav).reshape(T, B, 3)

    # 统一 cond 指令方向 + 骨盆 yaw
    cmd_dir_unit, cmd_yaw, cmd_source = _compute_traj_cmd_dir_and_yaw(clip)
    pelvis_yaw, yaw_diag = _align_pelvis_yaw(clip, cmd_yaw)
    if pelvis_yaw is None:
        pelvis_yaw = np.zeros((clip["T"],), dtype=np.float32)
        yaw_diag = {}
    # 手性与整体方向校正：若骨盆朝向与根速度方向平均夹角反向，则整体加 π
    yaw_corr = _wrap_to_pi_np(pelvis_yaw)
    root_vel = clip.get("root_vel")
    if root_vel is not None and root_vel.size > 0:
        ref_dir = np.asarray(root_vel, dtype=np.float32)
        ref_norm = np.linalg.norm(ref_dir, axis=1, keepdims=True)
        ref_dir = np.where(ref_norm > 1e-6, ref_dir / np.clip(ref_norm, 1e-6, None), ref_dir)
        ref_yaw = np.arctan2(ref_dir[:, 1], ref_dir[:, 0])
        cos_mean = float(np.cos(yaw_corr[:-1] - ref_yaw[:-1]).mean())
        if cos_mean < 0.0:  # 反向则整体翻转
            yaw_corr = _wrap_to_pi_np(yaw_corr + np.pi)
            yaw_diag["yaw_flip_applied"] = True
            yaw_diag["yaw_flip_reason"] = "mean_cos_with_root_vel<0"
        else:
            yaw_diag["yaw_flip_applied"] = False
        clip["traj_cmd_yaw"] = cmd_yaw.reshape(clip["T"], 1) if cmd_yaw is not None else None
    traj_meta = clip.setdefault("meta", {}).setdefault("trajectory", {})
    traj_meta["cmd_source"] = cmd_source
    if yaw_diag.get("forward_axis") is not None:
        traj_meta["pelvis_forward_axis"] = yaw_diag["forward_axis"]
    if yaw_diag.get("yaw_offset_rad") is not None:
        traj_meta["pelvis_forward_offset_rad"] = yaw_diag["yaw_offset_rad"]
    if yaw_diag.get("median_err_deg") is not None:
        traj_meta["pelvis_forward_err_deg"] = yaw_diag["median_err_deg"]

    # 打平特征（动态）
    X_flat, layout = pack_flat_features(
        clip,
        include_phase=use_phase_if_missing or clip["has_phase"],
        include_root_yaw=include_root_yaw,
        include_bone_pos=include_bone_pos,
        include_lin_vel=include_lin_vel,
        include_ang_vel=include_ang_vel
    )

    # 自回归对齐：x[t] -> y[t+1]（这里的 Y 用 6D 旋转；如需扩展可调整）
    T = int(clip["T"]); B = int(clip["B"])
    Y_flat = clip["bone_rot6d"].reshape(T, B*6).astype(np.float32)
    x_in_features = sanitize_f32(X_flat[:-1], "x_in_features")
    y_out_features = sanitize_f32(Y_flat[1:], "y_out_features")

    # cond_in: onehot(action) + 方向(cos,sin) + 速度标量（7 维），其中方向优先使用 TrajectoryDir（保证与骨架对齐）
    enum, act_size, clip_action = _load_action_meta_and_clip_action(json_path)
    act_name_norm = clip_action.split("::")[-1].strip().lower()
    act_idx = 0
    for i, name in enumerate(enum):
        if name.lower() == act_name_norm:
            act_idx = i
            break
    act_oh = np.zeros((T - 1, act_size), dtype=np.float32)
    act_oh[:, act_idx] = 1.0

    vel_dir, speed = _compute_planar_vel_dir_and_speed(dict(FPS=clip["FPS"], root_vel=clip["root_vel"], root_pos=clip["root_pos"]))
    # --- align command/velocity direction to UE 世界坐标系，并与根速度对齐 ---
    rot_deg = 0.0
    asset_meta = (clip.get("meta") or {}).get("asset_to_ue", {}) or {}
    try:
        rot_list = asset_meta.get("rotator_deg_RPY") or []
        if len(rot_list) >= 3:
            rot_deg = float(rot_list[2])  # yaw 分量
    except Exception:
        rot_deg = 0.0

    forward_offset_deg = 0.0
    traj_meta = (clip.get("meta") or {}).get("trajectory", {}) or {}
    try:
        forward_offset_deg = float(np.degrees(traj_meta.get("pelvis_forward_offset_rad", 0.0)))
    except Exception:
        forward_offset_deg = 0.0

    hand_sign = -1.0 if str((clip.get("meta") or {}).get("handedness", "right")).lower().startswith("left") else 1.0
    cmd_world = _rotate_dir_deg(cmd_dir_unit, rot_deg + forward_offset_deg) if cmd_dir_unit is not None else None
    if cmd_world is not None:
        # 左手坐标资产在转换到 UE 右手时需要翻转平面 X，以保证与 rot6d/pose 同手性
        cmd_world[..., 0] *= hand_sign
    # root_vel 在 JSON 中已是 UE 世界系，不再旋转；仅用于余弦检查/回退
    vel_world = vel_dir

    cond_dir_world = cmd_world if cmd_world is not None else vel_world
    if cmd_world is not None and vel_world is not None:
        cos = np.sum(cmd_world * vel_world, axis=-1)
        cond_dir_world = cmd_world.copy()
        # 强制避免反向：若夹角>90°直接改用 root_vel；若 18°~90°则也回退
        mask_back = cos < 0.0
        mask_mis = (cos >= 0.0) & (cos < 0.95)
        cond_dir_world[mask_back | mask_mis] = vel_world[mask_back | mask_mis]

    cond_dir_world = np.asarray(cond_dir_world, dtype=np.float32)
    norm = np.linalg.norm(cond_dir_world, axis=1, keepdims=True)
    cond_dir_world[norm[:, 0] < 1e-6] = 0.0
    cond_dir_world[norm[:, 0] >= 1e-6] /= np.clip(norm[norm[:, 0] >= 1e-6], 1e-6, None)

    cond_dir = cond_dir_world[:-1]
    cond_speed = speed[:-1]
    cond_in = sanitize_f32(np.concatenate([act_oh, cond_dir, cond_speed], axis=-1), "cond_in")


    # 保存
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_npz = os.path.join(out_dir, f"{base}.npz")

    # 序列化 layout（严格与 X_flat 一致）
    import json as _json
    state_layout_json = _json.dumps({k: {'start': int(v[0]), 'size': int(v[1]) - int(v[0])} for k, v in layout.items()}, ensure_ascii=False)
    output_layout_json = _json.dumps({'BoneRotations6D': {'start': 0, 'size': B*6}}, ensure_ascii=False)

    meta_payload = dict(clip.get("meta") or {})
    if clip.get("bone_ang_vel") is not None and clip["bone_ang_vel"].size > 0:
        bav = clip["bone_ang_vel"].reshape(clip["T"], -1).astype(np.float32)
        lo = np.percentile(bav, 2.5, axis=0)
        hi = np.percentile(bav, 97.5, axis=0)
        scale = np.maximum((hi - lo) * 0.5, 1e-3)
        motion_stats = meta_payload.setdefault("motion_stats", {})
        motion_stats["bone_angular_velocity"] = {
            "mode": "tanh",
            "scale_hint": scale.astype(np.float32).tolist()
        }
    if not meta_payload.get("handedness"):
        traj_hand = (meta_payload.get("trajectory") or {}).get("handedness")
        meta_payload["handedness"] = traj_hand or "right"

    meta_json = _json.dumps(meta_payload, ensure_ascii=False)

    np.savez_compressed(
        out_npz,
        x_in_features=x_in_features,
        y_out_features=y_out_features,
        cond_in=cond_in,
        X_flat=X_flat,
        FPS=np.int32(clip["FPS"]),
        bone_names=np.array(clip["bone_names"], dtype=object),
        parents=clip["parents"],
        bone_rot6d=clip["bone_rot6d"],
        bone_pos=(clip["bone_pos"] if clip["bone_pos"] is not None else np.zeros((0,), dtype=np.float32)),
        bone_vel=(clip["bone_vel"] if clip["bone_vel"] is not None else np.zeros((0,), dtype=np.float32)),
        bone_ang_vel=(clip["bone_ang_vel"] if clip["bone_ang_vel"] is not None else np.zeros((0,), dtype=np.float32)),
        traj_pos=clip["traj_pos"],
        traj_dir=clip["traj_dir"],
        root_pos=clip["root_pos"],
        root_vel=clip["root_vel"],
                meta_json=np.array(meta_json, dtype=object),
        source_json=np.array(os.path.abspath(json_path)),
        state_layout_json=state_layout_json,
        output_layout_json=output_layout_json,
    )

    K = clip["traj_dir"].shape[1] // 2 if clip["traj_dir"].size else 0
    return dict(path=json_path, out=out_npz, T=clip["T"], B=clip["B"], FPS=clip["FPS"], F=X_flat.shape[1], K=K)

# -----------------------------
# 合并多个样本为一个数据集
# -----------------------------
def merge_dataset(npz_paths: List[str], out_path: str):
    Xs, metas = [], []
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as z:
            Xs.append(z["X_flat"])
            metas.append(dict(T=z["X_flat"].shape[0], F=z["X_flat"].shape[1], FPS=int(z["FPS"])))
    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0,0), dtype=np.float32)
    np.savez_compressed(out_path, X_flat=X)
    return dict(N=len(npz_paths), T_total=X.shape[0], F=X.shape[1])

def main():
    ap = argparse.ArgumentParser(description="Convert UE JSON (v4: meters/local_6d) to training .npz")
    ap.add_argument("src", type=str, help="JSON 文件或目录")
    ap.add_argument("--out", type=str, default="./converted_npz", help="输出目录")
    ap.add_argument("--fps", type=int, default=0, help="目标 FPS（0=不重采样）")
    ap.add_argument("--sg-vel", action="store_true", help="对导出线/角速度做 SG(7,2) 平滑")
    ap.add_argument("--merge", action="store_true", help="把所有样本合并成 dataset.npz")
    ap.add_argument("--traj-n", type=int, default=2, help="未指定 --traj-keep 时，默认保留前 N 个前瞻点")
    ap.add_argument("--traj-keep", type=str, default="", help="逗号分隔的前瞻点索引（0 基）。留空则取前 N 个（从 0 开始）")

    # 新增开关
    # 自适应 Normalizer 相关
    ap.add_argument("--export-norm", action="store_true", help="拟合分组 Normalizer 并导出 norm_template.json")
    ap.add_argument("--export-merged-norm", action="store_true", help="同时导出合并后的 normalized_dataset.npz")
    ap.add_argument("--alpha-solver", type=str, default="kneedle", choices=["kneedle","quantile"], help="自适应 α 求解器")
    ap.add_argument("--alpha-quantile", type=float, default=0.10, help="当 alpha-solver=quantile 时使用的分位")
    ap.add_argument("--alpha-fixed", type=float, default=None, help="固定 α（留空则自适应）")
    
    ap.add_argument("--use-phase", action="store_true", help="Phase 缺失时是否写入 0 通道（默认不包含 Phase）")
    ap.add_argument("--use-bone-pos", action="store_true", help="如有 BonePositions 则写入 X")
    ap.add_argument("--use-bone-linvel", action="store_true", help="如有 BoneVelocities 则写入 X")
    ap.add_argument("--no-bone-angvel", action="store_true", help="不写 BoneAngularVelocities 到 X")

    args = ap.parse_args()

    files = find_json_files(args.src)
    if not files:
        print("❌ 未找到 JSON 文件")
        sys.exit(1)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    made = []
    for j in files:
        # 跳过输出目录中的 JSON（例如之前生成的 norm_template.json）
        if os.path.commonpath([os.path.abspath(j), os.path.abspath(out_dir)]) == os.path.abspath(out_dir):
            continue
        base_name = os.path.basename(j).lower()
        if base_name.startswith('norm_template'):
            continue
        try:
            traj_keep_idx = None
            if args.traj_keep.strip():
                traj_keep_idx = [int(x) for x in args.traj_keep.split(",") if x.strip().isdigit()]
            elif args.traj_n and args.traj_n > 0:
                traj_keep_idx = list(range(args.traj_n))

            info = convert_one(
                j, out_dir,
                target_fps=(args.fps if args.fps and args.fps > 0 else None),
                smooth_vel=args.sg_vel,
                traj_keep_idx=traj_keep_idx,
                use_phase_if_missing=args.use_phase,
                include_root_yaw=False,
                include_bone_pos=(True if args.use_bone_pos else None),
                include_lin_vel=(True if args.use_bone_linvel else None),
                include_ang_vel=(False if args.no_bone_angvel else True),
            )
            made.append(info["out"])
            print(f"✔ {os.path.basename(j)} -> {os.path.basename(info['out'])} "
                  f"(T={info['T']}, B={info['B']}, FPS={info['FPS']}, K={info['K']}, F={info['F']})")
        except Exception as e:
            print(f"✖ {os.path.basename(j)}: {e}")

    # ====== 自适应 Normalizer 拟合与导出（任选） ======
    if (args.export_norm or args.export_merged_norm) and len(made) > 0:
        cfg = _ConfigGN(
            group_prior_alpha=(args.alpha_fixed if args.alpha_fixed is not None else None),
            # yaw removed
            # 其余 floor 参数保持 None（完全数据驱动）
            alpha_solver=args.alpha_solver,
            alpha_quantile=args.alpha_quantile
        )
        gn = GroupedNormalizer(cfg)
        gn.fit(made)
        tmpl_path = os.path.join(out_dir, "norm_template.json")
        gn.save_template(tmpl_path)
        print(f"✔ 写出模板：{os.path.basename(tmpl_path)} "
              f"(alpha={gn.template.meta.get('adaptive',{}).get('resolved_alpha',-1):.4f})")

        # 将单个片段内联写入归一化后的 X/Y，训练时可直接读取
        for npz_path in made:
            try:
                with np.load(npz_path, allow_pickle=True) as d:
                    X = d['x_in_features']
                    Y = d['y_out_features']
                    state_layout_json = d.get('state_layout_json', None)
                    if state_layout_json is not None and hasattr(state_layout_json, 'item'):
                        state_layout_json = state_layout_json.item()
                    if isinstance(state_layout_json, (bytes, bytearray)):
                        state_layout_json = state_layout_json.decode('utf-8', 'ignore')
                    try:
                        state_layout = json.loads(state_layout_json) if state_layout_json else {}
                    except Exception:
                        state_layout = {}

                    meta_json = d.get('meta_json', None)
                    if meta_json is not None and hasattr(meta_json, 'item'):
                        meta_json = meta_json.item()
                    if isinstance(meta_json, (bytes, bytearray)):
                        meta_json = meta_json.decode('utf-8', 'ignore')
                    try:
                        meta_full = json.loads(meta_json) if isinstance(meta_json, str) else {}
                    except Exception:
                        meta_full = {}
                    meta_full.setdefault('state_layout', state_layout)

                    Xn = gn.normalize_X(X, meta_full)
                    Yn = gn.normalize_Y(Y)

                    payload = {k: d[k] for k in d.files}
                payload['X_norm'] = Xn.astype(np.float32)
                payload['Y_norm'] = Yn.astype(np.float32)
                np.savez_compressed(npz_path, **payload)
            except Exception as _e:
                print(f"[WARN] 写入 {os.path.basename(npz_path)} 的归一化特征失败：{_e}")

        if args.export_merged_norm:
            merged_norm = os.path.join(out_dir, "normalized_dataset.npz")
            info2 = _merge_normalized_dataset(made, gn, merged_norm)
            print(f"📦 合并标准化数据：N={info2['N']}, T={info2['T']}, Dx={info2['Dx']}, Dy={info2['Dy']} → {os.path.basename(merged_norm)}")


    if args.merge and made:
        merged = os.path.join(out_dir, "dataset.npz")
        info = merge_dataset(made, merged)
        print(f"📦 合并完成：{info['N']} 个样本，总帧 {info['T_total']}，F={info['F']} → {os.path.basename(merged)}")

if __name__ == "__main__":
    main()
