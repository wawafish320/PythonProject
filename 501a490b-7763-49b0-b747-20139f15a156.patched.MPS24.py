from __future__ import annotations


# ===== Common Helpers (extracted) =====

# ========== [Unified Geometry Utilities] ==========
import torch


def rot6d_to_matrix(xJ6: torch.Tensor, *, columns=("X","Z")) -> torch.Tensor:
    """
    xJ6: (..., J, 6)
      columns=("X","Z") 表示提供的是 X 列和 Z 列（列即轴，columns_are_axes），派生列按左手系规则补齐：
        remaining = second_column × first_column
      例如 X、Z 提供，则 Y = Z × X
    返回: (..., J, 3, 3)  列即轴
    """
    assert xJ6.shape[-1] == 6
    # 取两列
    a = xJ6[..., 0:3]   # first column
    b = xJ6[..., 3:6]   # second column

    # 归一化 + 正交化
    def _norm(v):
        return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    ax1 = columns[0]  # "X" or "Y" or "Z"
    ax2 = columns[1]

    r = {}
    r[ax1] = _norm(a)
    # 让第二列与第一列正交
    b = b - (r[ax1] * b).sum(dim=-1, keepdim=True) * r[ax1]
    r[ax2] = _norm(b)

    # 派生列（左手）：remaining = second × first
    remaining = [ax for ax in ("X","Y","Z") if ax not in (ax1, ax2)][0]
    def _cross(u, v):  # torch.cross(u, v)
        return torch.cross(u, v, dim=-1)
    r[remaining] = _norm(_cross(r[ax2], r[ax1]))

    # 组装列为轴
    RX = r["X"].unsqueeze(-1)
    RY = r["Y"].unsqueeze(-1)
    RZ = r["Z"].unsqueeze(-1)
    R = torch.cat([RX, RY, RZ], dim=-1)  # (..., J, 3, 3)

    # 行列式修正：若 det<0，仅翻转“派生列”
    orig_shape = R.shape[:-2]  # (..., J)
    det = torch.linalg.det(R.reshape(-1, 3, 3)).reshape(orig_shape)
    neg = (det < 0.0).unsqueeze(-1)  # (..., J, 1)

    col_idx = {"X": 0, "Y": 1, "Z": 2}[remaining]
    R = R.clone()
    # R[..., :, col_idx] 选择的是“该列”的 3 个分量（行方向切片）
    R[..., :, col_idx] = torch.where(neg, -R[..., :, col_idx], R[..., :, col_idx])

    return R

def geodesic_R(R_pred: torch.Tensor, R_gt: torch.Tensor, *, reduce=None) -> torch.Tensor:
    """
    Geodesic angular distance (radians) between two rotation matrices (...,J,3,3).
    Returns tensor shape (...,J) unless reduced.
    """
    assert R_pred.shape[-2:] == (3,3) and R_gt.shape[-2:] == (3,3)
    Rt = torch.matmul(R_pred.transpose(-1, -2), R_gt)  # (...,J,3,3)
    # Clamp trace for numerical stability
    trace = Rt[..., 0,0] + Rt[..., 1,1] + Rt[..., 2,2]
    cos = (trace - 1.) * 0.5
    cos = cos.clamp(-1.0, 1.0)
    ang = torch.acos(cos)  # radians
    if reduce == "mean":
        return ang.mean()
    if reduce == "sum":
        return ang.sum()
    return ang
# ========== [End Utilities] ==========


# ========== [End Loader] ==========
import math as _math

# === Unified layout entry parser (single source of truth) ===
from typing import Any, Optional

def parse_layout_entry(
    entry_value: Any,
    entry_name: Optional[str] = None,
    total_dim: Optional[int] = None
) -> Optional[slice]:
    """
    将布局条目解析为 slice，采用“严格 SIZE 语义”：
      - dict: {'start': s, 'size': k} / {'start': s, 'dim': k} / {'start': s, 'end': e}
      - list/tuple: [s, k]  ->  slice(s, s+k)   # 永远把第二项当 size
    不做任何启发式，不根据名称或 total_dim 改变语义。
    total_dim 仅用于（可选）越界检查（若提供）。
    """
    if entry_value is None:
        return None
    if isinstance(entry_value, slice):
        return entry_value
    # dict 形式
    if isinstance(entry_value, dict):
        if 'start' not in entry_value:
            return None
        s = int(entry_value['start'])
        if 'end' in entry_value:
            e = int(entry_value['end'])
            if (total_dim is not None) and not (0 <= s <= e <= total_dim):
                # 仅检查，不更改行为
                pass
            return slice(s, e)
        # 支持 size/dim 同义
        k = entry_value.get('size', entry_value.get('dim', None))
        if k is None:
            return None
        k = int(k)
        if (total_dim is not None) and not (0 <= s <= s+k <= total_dim):
            # 仅检查，不更改行为
            pass
        return slice(s, s + max(0, k))
    # list/tuple 形式：严格把第二项当 size
    if isinstance(entry_value, (list, tuple)) and len(entry_value) >= 2:
        s = int(entry_value[0])
        k = int(entry_value[1])
        if (total_dim is not None) and not (0 <= s <= s+k <= total_dim):
            # 仅检查，不更改行为
            pass
        return slice(s, s + max(0, k))
    return None



def _normalize_layout(raw_layout: Dict[str, Any], D: int) -> Dict[str, Tuple[int, int]]:
    """Normalize diverse span specs (start/size/end, tuple/list) -> (start, size).
    Strict checks: 0 <= start < end <= D.
    """
    norm = {}
    for name, meta in (raw_layout or {}).items():
        st = ed = sz = None
        if isinstance(meta, dict):
            if 'start' in meta and 'end' in meta:
                st = int(meta['start']); ed = int(meta['end']); sz = ed - st
            elif 'start' in meta and 'size' in meta:
                st = int(meta['start']); sz = int(meta['size']); ed = st + sz
        elif isinstance(meta, (list, tuple)) and len(meta) >= 2:
            # Heuristic: treat as (start,size); if looks like (start,end) with end>start and D known, still valid
            st = int(meta[0]); second = int(meta[1])
            if D is not None and second > st and second <= D:
                # ambiguous; prefer (start,size) since project uses [start,size], but accept (start,end) too
                # If interpreted as size and exceeds bound, switch to end
                as_size_ok = (st + second) <= D
                if as_size_ok:
                    sz = second; ed = st + sz
                else:
                    ed = second; sz = ed - st
            else:
                sz = second; ed = st + sz
        if st is None or (sz is None and ed is None):
            raise AssertionError(f"[FATAL] layout[{name}] cannot be normalized: {meta}")
        if ed is None: ed = st + sz
        if sz is None: sz = ed - st
        if not (0 <= st < ed <= D):
            raise AssertionError(f"[FATAL] layout[{name}] must be [start,end) with 0<=start<end<=D, got ({st},{ed}) and D={D}")
        norm[name] = (int(st), int(sz))
    return norm


def _layout_span(layout: Dict[str, Any], key: str) -> Optional[Tuple[int,int]]:
    sl = parse_layout_entry(layout.get(key), key)
    if sl is None:
        return None
    return (int(sl.start), int(sl.stop))


def _apply_stats_to_trainer(trainer, layout_center) -> None:
    """Central place to push stats, layouts, slices into trainer."""
    # Delegate to existing method to avoid duplication
    if hasattr(layout_center, 'apply_to_trainer'):
        layout_center.apply_to_trainer(trainer)
        return
    # Fallback (shouldn't happen): set minimal attributes
    import numpy as np, torch
    trainer._x_layout = dict(getattr(layout_center, 'state_layout', {}))
    trainer._y_layout = dict(getattr(layout_center, 'output_layout', {}))
    for attr in ('mu_x','std_x','mu_y','std_y'):
        arr = getattr(layout_center, attr, None)
        if arr is not None:
            setattr(trainer, attr, torch.tensor(np.asarray(arr).reshape(1,-1), dtype=torch.float32))
    trainer.tanh_scales_rootvel = getattr(layout_center, 'tanh_scales_rootvel', None)
    trainer.tanh_scales_angvel  = getattr(layout_center, 'tanh_scales_angvel', None)

def _resolve_paths(spec) -> list:
    """Generalized path resolver:
    - Comma-separated string, list of patterns, or @file includes
    - Each item: directory -> *.npz, glob pattern, or concrete file
    """
    files = []
    if spec is None: 
        return files
    if isinstance(spec, (list, tuple)):
        parts = []
        for it in spec:
            if isinstance(it, str):
                parts.extend([s.strip() for s in it.split(',') if s.strip()])
            else:
                continue
    else:
        parts = [s.strip() for s in str(spec).split(',') if s.strip()]
    for pth in parts:
        if pth.startswith('@') and os.path.isfile(pth[1:]):
            with open(pth[1:], 'r', encoding='utf-8') as f:
                for line in f:
                    q = line.strip()
                    if not q: 
                        continue
                    if os.path.isdir(q):
                        files.extend(sorted(glob.glob(os.path.join(q, '*.npz'))))
                    elif any(ch in q for ch in '*?['):
                        files.extend(sorted(glob.glob(q)))
                    elif os.path.isfile(q):
                        files.append(q)
        else:
            q = pth
            if os.path.isdir(q):
                files.extend(sorted(glob.glob(os.path.join(q, '*.npz'))))
            elif any(ch in q for ch in '*?['):
                files.extend(sorted(glob.glob(q)))
            elif os.path.isfile(q):
                files.append(q)
    # unique while preserving order
    out = []
    seen = set()
    for f in files:
        if f not in seen:
            out.append(f); seen.add(f)
    return out

# Backward-compat shim
def _expand_paths(spec: str):
    return _resolve_paths(spec)
# ===== End Helpers =====
# =========================
# [STRICT] LayoutCenter (start,end) semantics only
# =========================
class LayoutCenter:
    """
    Single source of truth for:
      - meta.state_layout / meta.output_layout  (strictly [start,end))
      - y_to_x_map
      - MuX/StdX, MuY/StdY
      - optional: tanh_scales_* , fps, bone_names, rot6d_spec
    No fallback or guessing. Fail fast on missing or out-of-range.
    """
    def __init__(self, bundle_path: str):
        import json, numpy as np
        with open(bundle_path, "r", encoding="utf-8") as f:
            b = json.load(f)
        self.bundle = b
        self.np = np
        self.meta = b.get("meta", {}) or {}
        # required stats
        self.mu_x = np.asarray(b["MuX"], dtype=np.float32)
        self.std_x = np.asarray(b["StdX"], dtype=np.float32)
        self.mu_y = np.asarray(b["MuY"], dtype=np.float32)
        self.std_y = np.asarray(b["StdY"], dtype=np.float32)
        # raw layouts (strictly interpreted as [start,end))
        self.state_layout_raw  = dict(self.meta.get("state_layout", {}))
        self.output_layout_raw = dict(self.meta.get("output_layout", {}))
        # normalized to [start,size]
        self.state_layout  = None
        self.output_layout = None

        # optional
        self.y_to_x_map = list(self.meta.get("y_to_x_map", []))
        self.tanh_scales_rootvel = self.bundle.get("tanh_scales_rootvel", None)
        self.tanh_scales_angvel  = self.bundle.get("tanh_scales_angvel", None)
        self.fps = float(self.meta.get("fps", 60.0))
        self.bone_names = list(self.meta.get("bone_names", []))
        self.rot6d_spec = dict(self.meta.get("rot6d_spec", {}))
    def strict_validate(self, Dx: int, Dy: int):
        # 1) stats dimension
        assert self.mu_x.size == Dx and self.std_x.size == Dx, \
            f"[FATAL] MuX/StdX length ({self.mu_x.size}/{self.std_x.size}) != Dx({Dx})"
        assert self.mu_y.size == Dy and self.std_y.size == Dy, \
            f"[FATAL] MuY/StdY length ({self.mu_y.size}/{self.std_y.size}) != Dy({Dy})"
        # 2) normalize to [start,size] using strict [start,end) semantics
        self.state_layout  = _normalize_layout(self.state_layout_raw,  Dx)
        self.output_layout = _normalize_layout(self.output_layout_raw, Dy)
        # 3) required keys
        need_x = ("RootYaw","RootVelocity","BoneRotations6D","BoneAngularVelocities")
        need_y = ("BoneRotations6D",)
        for k in need_x:
            assert k in self.state_layout, f"[FATAL] meta.state_layout missing key: {k}"
        for k in need_y:
            assert k in self.output_layout, f"[FATAL] meta.output_layout missing key: {k}"
        # 4) ranges already validated above, but keep assertions
        for name, (st, sz) in self.state_layout.items():
            assert 0 <= st < Dx and 0 < sz <= Dx-st, f"[FATAL] state_layout[{name}] OOR: start={st} size={sz} Dx={Dx}"
        for name, (st, sz) in self.output_layout.items():
            assert 0 <= st < Dy and 0 < sz <= Dy-st, f"[FATAL] output_layout[{name}] OOR: start={st} size={sz} Dy={Dy}"

    def materialize_y_to_x_map(self):
        if self.y_to_x_map:
            return self.y_to_x_map
        y2x = []
        common = sorted(set(self.state_layout.keys()) & set(self.output_layout.keys()))
        for name in common:
            xs, xz = self.state_layout[name]
            ys, yz = self.output_layout[name]
            k = min(int(xz), int(yz))
            if k > 0:
                y2x.append({"name": name, "x_start": int(xs), "x_size": k, "y_start": int(ys), "y_size": k})
        return y2x

    def apply_to_dataset(self, ds):
        ds.state_layout  = dict(self.state_layout)
        ds.output_layout = dict(self.output_layout)
        if not hasattr(ds, "fps") or ds.fps is None:
            ds.fps = self.fps

    def apply_to_trainer(self, trainer):
        import torch, numpy as np
        trainer._x_layout = dict(self.state_layout)
        trainer._y_layout = dict(self.output_layout)
        trainer.y_to_x_map = self.materialize_y_to_x_map()
        # stats
        trainer.mu_x  = torch.tensor(np.asarray(self.mu_x).reshape(1, -1), dtype=torch.float32)
        trainer.std_x = torch.tensor(np.asarray(self.std_x).reshape(1, -1), dtype=torch.float32)
        trainer.mu_y  = torch.tensor(np.asarray(self.mu_y).reshape(1, -1), dtype=torch.float32)
        trainer.std_y = torch.tensor(np.asarray(self.std_y).reshape(1, -1), dtype=torch.float32)
        # tanh scales passthrough
        trainer.tanh_scales_rootvel = self.tanh_scales_rootvel
        trainer.tanh_scales_angvel  = self.tanh_scales_angvel
        # named slices
        trainer.yaw_slice       = parse_layout_entry(trainer._y_layout.get('RootYaw'), 'RootYaw')
        trainer.rootvel_slice   = parse_layout_entry(trainer._y_layout.get('RootVelocity'), 'RootVelocity')
        trainer.angvel_slice    = parse_layout_entry(trainer._y_layout.get('BoneAngularVelocities'), 'BoneAngularVelocities')
        trainer.yaw_x_slice     = parse_layout_entry(trainer._x_layout.get('RootYaw'), 'RootYaw')
        trainer.rootvel_x_slice = parse_layout_entry(trainer._x_layout.get('RootVelocity'), 'RootVelocity')
        trainer.angvel_x_slice  = parse_layout_entry(trainer._x_layout.get('BoneAngularVelocities'), 'BoneAngularVelocities')
        # 👉 新增：明确注入 Rot6D 的 X/Y 切片（供 Y→X 写回用）
        trainer.rot6d_x_slice = parse_layout_entry(trainer._x_layout.get('BoneRotations6D'), 'BoneRotations6D')
        trainer.rot6d_y_slice = parse_layout_entry(trainer._y_layout.get('BoneRotations6D'), 'BoneRotations6D')


def __apply_layout_center(ds_train, trainer):
    bundle_path = _arg('bundle_json', None) if '_arg' in globals() else None
    if not bundle_path:
        raise SystemExit("[FATAL] 需要 --bundle_json 指定集中化的归一化与布局模板（不再支持猜测/回退）")

    # 1) 加载并校验
    _center = LayoutCenter(bundle_path)
    _center.strict_validate(int(ds_train.Dx), int(ds_train.Dy))

    # 落到数据集（fps/布局等）
    _center.apply_to_dataset(ds_train)
    # 2) 先把布局/切片/统计等“落到 trainer 身上”
    _center.apply_to_trainer(trainer)

    # 显式设置 Rot6D 的 X/Y 切片，供 Y→X 写回使用（避免依赖隐式回退）
    trainer.rot6d_x_slice = parse_layout_entry(trainer._x_layout.get('BoneRotations6D'), 'BoneRotations6D')
    trainer.rot6d_y_slice = parse_layout_entry(trainer._y_layout.get('BoneRotations6D'), 'BoneRotations6D')

    # 保存 bundle meta（验证用；不影响训练）
    try:
        trainer._bundle_meta = dict(_center.meta)
    except Exception:
        trainer._bundle_meta = {}

    # 骨骼名（若数据集提供）
    if getattr(trainer, '_bone_names', None) is None and getattr(ds_train, 'bone_names', None):
        try:
            trainer._bone_names = list(ds_train.bone_names)
        except Exception:
            pass

    # 3) 注入 normalizer（关键：放在 apply_to_trainer 之后，保证切片已存在）
    trainer.normalizer = DataNormalizer(
        mu_x=_center.mu_x, std_x=_center.std_x,
        mu_y=_center.mu_y, std_y=_center.std_y,
        s_eff_x=getattr(trainer, 's_eff_x', None),
        s_eff_y=getattr(trainer, 's_eff_y', None),
        y_to_x_map=_center.materialize_y_to_x_map(),
        yaw_x_slice=trainer.yaw_x_slice,   yaw_y_slice=trainer.yaw_slice,
        rootvel_x_slice=trainer.rootvel_x_slice, rootvel_y_slice=trainer.rootvel_slice,
        angvel_x_slice=trainer.angvel_x_slice,   angvel_y_slice=trainer.angvel_slice,
        tanh_scales_rootvel=_center.tanh_scales_rootvel,
        tanh_scales_angvel=_center.tanh_scales_angvel,
    )

    # 4) 同步计算并注入 s_eff_y 与 s_eff_x（支持 282→276 的裁剪）
    try:
        # 4.1 取必要材料
        ylay = _center.output_layout   # dict: {'BoneRotations6D': {'start':..., 'size':...}, ...}
        xlay = _center.state_layout
        std_y = _center.std_y          # np.ndarray, shape=[Dy]
        std_x = _center.std_x          # np.ndarray, shape=[Dx]

        # Rot6D：alpha*prior floor（按你的模板）
        priors = np.asarray(_center.bundle["group_priors_rot6d"]["prior_per_dim"], dtype=np.float32)  # len=282
        alpha  = float(_center.bundle["cond_norm_config"]["floor_rot6d"]["alpha"])  # e.g. 1.1037

        sl_y = parse_layout_entry(ylay.get("BoneRotations6D"), "BoneRotations6D")
        sl_x = parse_layout_entry(xlay.get("BoneRotations6D"), "BoneRotations6D")
        assert isinstance(sl_y, slice) and isinstance(sl_x, slice), "[s_eff] Rot6D 切片缺失"

        Dy_r6 = sl_y.stop - sl_y.start  # 276
        Dx_r6 = sl_x.stop - sl_x.start  # 276

        def _align_priors(target_len: int) -> np.ndarray:
            if len(priors) == target_len:
                return priors
            if len(priors) == target_len + 6:
                return priors[6:6 + target_len]  # 丢掉 root 的 6 维
            print(f"[s_eff][WARN] prior_len={len(priors)} 与 target_len={target_len} 不匹配，使用常量 1.0 回退")
            return np.ones(target_len, dtype=np.float32)

        prior_y = _align_priors(Dy_r6)
        prior_x = _align_priors(Dx_r6)

        s_eff_y = std_y.copy()
        s_eff_x = std_x.copy()

        y_block = s_eff_y[sl_y.start:sl_y.stop]
        x_block = s_eff_x[sl_x.start:sl_x.stop]
        y_block = np.maximum(y_block, alpha * prior_y)
        x_block = np.maximum(x_block, alpha * prior_x)
        y_block = np.maximum(y_block, 1e-3)
        x_block = np.maximum(x_block, 1e-3)
        s_eff_y[sl_y.start:sl_y.stop] = y_block
        s_eff_x[sl_x.start:sl_x.stop] = x_block

        # 4.2 注入到 loss（_pick_s_eff 会 .to(device,dtype) 并按 D 适配）
        trainer.loss_fn.s_eff_y = torch.as_tensor(s_eff_y, dtype=torch.float32)
        # 你这版 loss 支持 X 的 z-safe，所以一并注入
        trainer.loss_fn.s_eff_x = torch.as_tensor(s_eff_x, dtype=torch.float32)

        # 4.3 一次性诊断
        if not getattr(trainer.loss_fn, "_s_eff_dbg_once", False):
            def _stats(arr, name):
                m, M = float(arr.min()), float(arr.max())
                med = float(np.median(arr))
                print(f"[s_eff][{name}] min={m:.4g} med={med:.4g} max={M:.4g}")

            _stats(s_eff_y, "Y"); _stats(s_eff_x, "X")
            print(f"[s_eff] alpha={alpha:.4f} | rot6d_x={Dx_r6} rot6d_y={Dy_r6}")
            print(f"[s_eff][CHK] yaw π={np.pi:.3f} | rv_scales={_center.bundle.get('tanh_scales_rootvel', None) is not None} | "
                  f"ang_scales={_center.bundle.get('tanh_scales_angvel', None) is not None}")
            trainer.loss_fn._s_eff_dbg_once = True

    except Exception as e:
        print(f"[s_eff][ERROR] 计算/注入失败：{e}，回退到 Std。")

    # 5) 继续集中化把 mu/std、切片等推给 trainer（如有该函数）
    if '_apply_stats_to_trainer' in globals():
        _apply_stats_to_trainer(trainer, _center)

    # 6) 诊断输出（保持你原来的格式）
    xlay = getattr(trainer, "_x_layout", getattr(ds_train, "state_layout", None))
    ylay = getattr(trainer, "_y_layout", getattr(ds_train, "output_layout", None))

    yaw    = parse_layout_entry(ylay.get('RootYaw'), 'RootYaw')
    rootv  = parse_layout_entry(ylay.get('RootVelocity'), 'RootVelocity')
    angv   = parse_layout_entry(ylay.get('BoneAngularVelocities'), 'BoneAngularVelocities')

    yaw_x   = parse_layout_entry(xlay.get('RootYaw'), 'RootYaw')
    rootv_x = parse_layout_entry(xlay.get('RootVelocity'), 'RootVelocity')
    angv_x  = parse_layout_entry(xlay.get('BoneAngularVelocities'), 'BoneAngularVelocities')

    rot6d_x_span = _layout_span(xlay, 'BoneRotations6D')
    rot6d_y_span = _layout_span(ylay, 'BoneRotations6D')

    mx = getattr(trainer, 'mu_x', None); sx = getattr(trainer, 'std_x', None)
    my = getattr(trainer, 'mu_y', None); sy = getattr(trainer, 'std_y', None)
    len_mx = int(mx.numel()) if hasattr(mx, 'numel') else (len(mx) if mx is not None else 0)
    len_sx = int(sx.numel()) if hasattr(sx, 'numel') else (len(sx) if sx is not None else 0)
    len_my = int(my.numel()) if hasattr(my, 'numel') else (len(my) if my is not None else 0)
    len_sy = int(sy.numel()) if hasattr(sy, 'numel') else (len(sy) if sy is not None else 0)

    print(f"[DiagSlices][final] yaw={yaw} rootvel={rootv} angvel={angv} rot6d_x={rot6d_x_span} rot6d_y={rot6d_y_span}")
    print(f"[DiagSlices][X] yaw_x={yaw_x} rootvel_x={rootv_x} angvel_x={angv_x} rot6d_x={rot6d_x_span}")
    print(f"[DiagSlices][Y] yaw={yaw} rootvel={rootv} angvel={angv} rot6d_y={rot6d_y_span}")
    print(f"[Bundle->Trainer post] MuX/StdX: {len_mx} {len_sx} | MuY/StdY: {len_my} {len_sy}")


from torch import nn

# ==== ARPG-PATCH: angular velocity helpers (eval-only) ====
import torch
import math as _math

def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) log map. R: [...,3,3] -> axis-angle vector phi [...,3] (radians * axis)
    """
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) * 0.5
    tr = torch.clamp(tr, -1.0, 1.0)
    theta = torch.acos(tr)
    W = R - R.transpose(-1, -2)
    vee = torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1) * 0.5
    sin_theta = torch.sin(theta)
    k = theta / torch.clamp(2.0 * sin_theta, min=1e-6)
    k = k.unsqueeze(-1)
    phi = k * vee
    small = (theta < 1e-5).unsqueeze(-1)
    phi = torch.where(small, 0.5 * vee, phi)
    return phi

def angvel_vec_from_R_seq(R_seq: torch.Tensor, fps: float) -> torch.Tensor:
    """
    R_seq: [B,T,J,3,3] GS'ed rotation matrices
    Returns omega: [B,T-1,J,3] in rad/s
    """
    dR = torch.matmul(R_seq[:, 1:], R_seq[:, :-1].transpose(-1, -2))
    phi = so3_log_map(dR)
    omega = phi * float(fps)
    return omega

import torch
import torch.nn.functional as F
import numpy as np


def validate_and_fix_model_(m: nn.Module, Dx: int | None = None, Dc: int | None = None, *, reinit_on_nonfinite: bool = True) -> None:
    """
    Single entrypoint for production sanity:
    - Optionally checks first Linear's in_features against Dx+Dc when both are provided.
    - Scans parameters; if any non-finite, re-initializes that leaf module once.
    - Asserts all params are finite at the end.
    Side effects: may modify `m` in-place.
    """
    # 1) optional in_features check on the first Linear we encounter
    if Dx is not None and Dc is not None:
        first_linear_in = None
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                first_linear_in = mod.in_features
                break
        if first_linear_in is not None and first_linear_in != (Dx + Dc):
            raise RuntimeError(f"First Linear in_features={first_linear_in} != Dx+Dc={Dx+Dc}")

    # 2) re-init leaf modules that contain non-finite params
    def _reinit_module_(mod: nn.Module) -> None:
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_uniform_(mod.weight, a=_math.sqrt(5))
            if getattr(mod, 'bias', None) is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(mod.weight)
                bound = 1.0 / (_math.sqrt(max(fan_in, 1)))
                nn.init.uniform_(mod.bias, -bound, bound)
        elif hasattr(mod, 'reset_parameters'):
            try:
                mod.reset_parameters()  # type: ignore[attr-defined]
            except Exception:
                pass


    with torch.no_grad():
        for name, mod in m.named_modules():
            has_bad = False
            for pname, p in mod.named_parameters(recurse=False):
                if not torch.isfinite(p).all():
                    has_bad = True
                    break
            if has_bad and reinit_on_nonfinite:
                _reinit_module_(mod)

        # final assert
        for name, mod in m.named_modules():
            for pname, p in mod.named_parameters(recurse=False):
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"param still non-finite after reinit: {name}.{pname}")


import sys
def _fix_firstdim_any(v, L: int):
    """
    Recursively pad/truncate so that the FIRST dimension equals L for tensors/ndarrays,
    and apply the same fix inside dict/list/tuple. Scalars are kept untouched.
    NOTE: numpy arrays with non-numeric dtype (e.g., strings '<U..' or object) are converted to lists
    to avoid PyTorch default_collate errors.
    """

    # PyTorch tensor
    if isinstance(v, torch.Tensor):
        if v.dim() >= 1:
            n = v.shape[0]
            if n == L:
                return v
            if n > L:
                return v[:L]
            pad = v[-1:].expand(L - n, *v.shape[1:])
            return torch.cat([v, pad], dim=0)
        return v
    # Numpy ndarray
    if isinstance(v, np.ndarray):
        # If dtype is non-numeric (strings, object), DO NOT collate as ndarray -> convert to list
        if v.dtype.kind in ('U', 'S', 'O'):
            return v.tolist()
        # numeric arrays: pad/truncate by first dimension
        if v.ndim >= 1:
            n = v.shape[0]
            if n == L:
                return v
            if n > L:
                return v[:L]
            pad = np.repeat(v[-1:, ...], L - n, axis=0)
            return np.concatenate([v, pad], axis=0)
        return v
    # Dict: recurse
    if isinstance(v, dict):
        return {k: _fix_firstdim_any(vv, L) for k, vv in v.items()}
    # Sequence: recurse
    if isinstance(v, (list, tuple)):
        seq = [_fix_firstdim_any(x, L) for x in v]
        return type(v)(seq)
    # Other scalars/strings: return as-is
    return v





def make_fixedlen_collate(seq_len: int):
    from torch.utils.data._utils.collate import default_collate as _default_collate


    DROP_KEYS = {'json_path', 'npz_path', 'foot_json_path', 'source_json', 'json_candidates'}

    def _collate(batch):
        fixed = []
        for sample in batch:
            s = _fix_firstdim_any(sample, seq_len)

            for k in list(s.keys()):
                if k in DROP_KEYS:
                    s.pop(k, None)

            for k, v in list(s.items()):
                if isinstance(v, np.ndarray) and v.dtype.kind in ('U', 'S', 'O'):
                    s[k] = v.tolist()

            fixed.append(s)
        return _default_collate(fixed)

    return _collate




def _arg(name, default=None):
    g = globals().get('GLOBAL_ARGS', None)
    try:
        return getattr(g, name)
    except Exception:
        return default



class _CondFiLM(nn.Module):

    def __init__(self, cond_dim: int, hidden_dim: int, film_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, film_dim * 2)

    def forward(self, cond: torch.Tensor):
        h = torch.nn.functional.gelu(self.fc1(cond))
        g, b = self.fc2(h).chunk(2, dim=-1)
        g = 1.0 + 0.5 * torch.tanh(g)
        b = 0.5 * torch.tanh(b)
        return (g, b)

'\nv_ARPG_Focused_UE_Final_patched.py\n\n合并并增强的训练脚本：\n- 动态 Teacher Forcing（计划采样）+ 残差 + 注意力耦合\n- 速度项：长窗速度 + 逐帧速度曲线（可门控）\n- 相位输出归一化 + 相位时序平滑\n- 数据增强（高斯噪声/时间缩放）、Mixup、SWA（可选）\n- 评估指标：SpeedCurve RMSE/Corr + Diversity\n- 导出：UE schema/stats + 简化双输出 ONNX\n\n与原版参数兼容；新增若干可选参数（见 --help）。\n'
import os, json, math, glob, time, random, argparse

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional
try:
    from tqdm import tqdm
except ImportError:
    print('Warning: tqdm not found. For a progress bar, run: pip install tqdm')

    def tqdm(iterable, *GLOBAL_ARGS, **kwargs):
        return iterable

def reproject_rot6d(flat_6d: torch.Tensor) -> torch.Tensor:
    orig = flat_6d.shape                    # (..., D)
    D = orig[-1]
    if D % 6 != 0:
        raise ValueError(f"[reproject_rot6d] last dim={D} not divisible by 6")
    J = D // 6

    x = flat_6d.view(*orig[:-1], J, 6)      # (..., J, 6)
    b1 = F.normalize(x[..., :3], dim=-1, eps=1e-8)
    b2 = x[..., 3:]
    b2 = b2 - (b1 * b2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)

    y = torch.cat([b1, b2], dim=-1)        # (..., J, 6)
    return y.view(*orig[:-1], 6 * J)        # 还原到 (..., D)



def harmonize_angvel_units_inplace(ds, prefer='rad/s', guess=True, verbose=True):
    """
    把 ds.clips 里 X 的 BoneAngularVelocities 统一为 rad/s。
    若 guess=True，会按幅度做一次启发式判断：
      - 若 95 分位 < ~π：很可能是 rad/步 → 乘以 fps
      - 否则当作已经是 rad/s
    """

    # --- 优先看 clip.meta 的明确声明（你的数据里就有） ---
    unit_hint = ''
    try:
        first_meta = (ds.clips[0].get('meta') or {}) if getattr(ds, 'clips', None) else {}
        spaces = first_meta.get('spaces') or {}
        unit_hint = str(spaces.get('bone_angular_velocities', '')).lower()
    except Exception:
        pass
    if 'per_sec' in unit_hint or 'rad/s' in unit_hint:
        if verbose:
            print('[AngVelUnits] meta says rad/s; skip harmonization.')
        return

    # --- 以下保持你原有逻辑 ---
    layout = getattr(ds, 'state_layout', {}) or getattr(ds, 'state_layout', {}) or {}
    if 'BoneAngularVelocities' not in layout:
        return
    s, e = (int(layout['BoneAngularVelocities'][0]), int(layout['BoneAngularVelocities'][1]))
    if e <= s:
        return
    fps = float(getattr(ds, 'fps', 60.0))

    vals = []
    for clip in getattr(ds, 'clips', []):
        X = clip.get('X', None)
        if X is None:
            continue
        w = X[:, s:e].astype(np.float32, copy=False)
        vals.append(np.abs(w).reshape(-1))
    if not vals:
        return
    vals = np.concatenate(vals, axis=0)
    q95 = float(np.quantile(vals, 0.95)) if vals.size else 0.0

    is_rad_per_step = False
    if guess and q95 < 3.2:  # 你的原阈值
        is_rad_per_step = True

    if verbose:
        unit = 'rad/步' if is_rad_per_step else 'rad/s'
        print(f'[AngVelUnits] detected X≈{unit} (q95={q95:.3f}); target=rad/s')

    if is_rad_per_step:
        for clip in getattr(ds, 'clips', []):
            X = clip.get('X', None)
            if X is None:
                continue
            X[:, s:e] *= fps


class MotionAugmentation:
    """
    训练时的数据增强：
      - 时间扭曲（等长重采样，反射边界，保持可微）
      - 加性高斯噪声

    """

    def __init__(self, noise_std: float=0.0, time_warp_prob: float=0.0):
        self.noise_std = float(noise_std)
        self.time_warp_prob = float(time_warp_prob)

    @torch.no_grad()
    def _time_warp(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        B, T, D = x.shape
        if T <= 1 or scale <= 0:
            return x
        device = x.device
        t = torch.arange(T, device=device, dtype=torch.float32)
        c = (T - 1) / 2.0
        src = (t - c) / scale + c
        period = 2.0 * (T - 1)
        s = torch.remainder(src, period)
        src_reflect = torch.where(s <= T - 1, s, period - s)
        i0 = torch.clamp(src_reflect.floor().to(torch.long), 0, T - 1)
        i1 = torch.clamp(i0 + 1, 0, T - 1)
        a = (src_reflect - i0.to(src_reflect.dtype)).view(1, T, 1)
        idx0 = i0.view(1, T, 1).repeat(B, 1, D)
        idx1 = i1.view(1, T, 1).repeat(B, 1, D)
        x0 = torch.gather(x, 1, idx0)
        x1 = torch.gather(x, 1, idx1)
        y = torch.lerp(x0, x1, a)
        return y


def _resolve_json_path(npz_file_path: str, json_val) -> str | None:
    """
    解析 FootEvidence JSON 的实际可用路径：
      - 支持 numpy 的 Unicode 0维标量（<U*，shape=()）与 Python 字符串
      - 如果是陈旧的绝对路径，回落为 basename，在多个“合理根”中浅层搜索
      - 支持通过环境变量 FOOT_JSON_ROOT 指定优先根
    搜索根的优先级： [FOOT_JSON_ROOT] + [npz_dir, npz_dir/.., data_root, data_root/.., data_root/../raw_data, npz_dir/../raw_data]
    """
    import os, numpy as np
    # 先把 numpy 标量 / bytes 统一成 Python str
    if isinstance(json_val, np.ndarray):
        try:
            if json_val.shape == ():
                json_val = json_val.item()
        except Exception as _hz_e:
            print(f"[HZ-ERR] hazard branch failed: {_hz_e}")
            pass
    if isinstance(json_val, (bytes, bytearray)):
        try:
            json_val = json_val.decode('utf-8')
        except Exception:
            try:
                json_val = json_val.decode('latin1')
            except Exception:
                json_val = None
    if json_val is None:
        return None
    if not isinstance(json_val, str) or len(json_val) == 0:
        return None

    # 1) 绝对路径可直接使用
    if os.path.isabs(json_val) and os.path.exists(json_val):
        return json_val

    base = os.path.basename(json_val)
    npz_dir = os.path.dirname(os.path.abspath(npz_file_path))

    # 候选根
    roots = []
    env_root = os.environ.get('FOOT_JSON_ROOT')
    if env_root:
        roots.append(env_root)
    roots.append(npz_dir)
    roots.append(os.path.dirname(npz_dir))

    # 尝试从调用方上下文推断 data_root（在本文件内，调用处会传入 self.data_dir 或者数据目录；若不可用则忽略）
    data_root = None
    try:
        # best-effort: 当前函数外层可见变量（可能没有）
        data_root = globals().get('DATA_ROOT', None)
    except Exception:
        data_root = None
    for dr in [data_root]:
        if dr:
            dr = os.path.abspath(dr)
            roots += [dr, os.path.dirname(dr), os.path.join(os.path.dirname(dr), 'raw_data')]
    roots += [os.path.join(os.path.dirname(npz_dir), 'raw_data')]

    # 浅层扫描（两层）
    def scan_one(root: str):
        if not root or not os.path.isdir(root):
            return None
        cand = os.path.join(root, base)
        if os.path.exists(cand):
            return cand
        try:
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if os.path.isfile(p) and os.path.basename(p) == base:
                    return p
                if os.path.isdir(p):
                    p2 = os.path.join(p, base)
                    if os.path.exists(p2):
                        return p2
        except Exception:
            return None
        return None

    for r in roots:
        hit = scan_one(r)
        if hit:
            return hit

    # 最后兜底：尝试相对 npz_dir 直接拼原始字符串
    tail = os.path.join(npz_dir, json_val)
    if os.path.exists(tail):
        return tail
    return None


def _load_foot_soft_from_json(json_path: str):
    """Return np.ndarray [T,2] of soft_contact_score from FootEvidence JSON."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
        frames = js.get('Frames') or js.get('frames')
        if not frames:
            return None
        T = len(frames)
        arr = np.zeros((T, 2), dtype=np.float32)
        for t, fr in enumerate(frames):
            fe = fr.get('FootEvidence') or fr.get('foot_evidence') or {}
            L = fe.get('L') or fe.get('l') or {}
            R = fe.get('R') or fe.get('r') or {}
            arr[t, 0] = float(L.get('soft_contact_score', L.get('softScore', 0.0)) or 0.0)
            arr[t, 1] = float(R.get('soft_contact_score', R.get('softScore', 0.0)) or 0.0)
        return arr
    except Exception:
        return None


class MotionEventDataset(Dataset):
    """
       Phase-free 数据集：
         - 不再切分 C=[phase|other]，全部 C 作为条件；内部不使用 phase_dim。
         - 提供健壮的 cond 归一化（可开关），并暴露 C_mu/C_std 以便 val 继承。
         - 保留 yaw 增强、Rot6D/轨迹分量旋转、contacts→evi_soft 生成。
       """

    def __init__(self, data_dir: str, seq_len: int, skeleton_file: None = None, paths: None = None,
                 align_cond_to_y: bool = False):
        self.norm_stats_inherited = None
        self.align_cond_to_y = bool(align_cond_to_y)
        self.paths = sorted(paths) if paths is not None else sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        if not self.paths:
            self.paths = sorted(glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True))
        # [PATCH-B2] drop legacy summary npz that isn't a clip
        self.paths = [p for p in self.paths if os.path.basename(p) != 'normalized_dataset.npz']
        if not self.paths:
            raise FileNotFoundError(f'No .npz files found under {data_dir} (tried *.npz and **/*.npz)')
        self.seq_len = int(seq_len)
        self.clips = []
        self.index = []
        _cnt_npz = _cnt_json = _cnt_missing = _cnt_hits = 0
        _first_json_path = None
        for p in self.paths:
            try:
                clip = dict(np.load(p, allow_pickle=True))
                # 优先使用合并后的“分组自适应归一化”大包；无则回退到原始分片
                X = clip.get('X_norm') or clip.get('x_in_features')
                Y = clip.get('Y_norm') or clip.get('y_out_features')
                C = clip.get('C') or clip.get('cond_in')
                clip['__is_normed__'] = bool(('X_norm' in clip) and ('Y_norm' in clip))

                contacts_arr = clip.get('contacts', None)
                foot_soft_arr = clip.get('foot_soft', None)
                foot_hits = None
                if foot_soft_arr is None:
                    foot_hits = clip.get('foot_hits', None)
                    if foot_hits is not None:
                        try:
                            fh = np.asarray(foot_hits)
                            if fh.ndim == 2 and fh.shape[1] >= 2:
                                foot_soft_arr = (fh[:, :2] > 0).astype(np.float32)
                        except Exception:
                            foot_soft_arr = None
                json_val = clip.get('source_json') or clip.get('json_path') or clip.get('foot_json_path')
                resolved_json = _resolve_json_path(p, json_val) if foot_soft_arr is None else None
                if X is None or Y is None or C is None:
                    print(f'[Dataset] Skip {p}: missing X/Y/C')
                    continue
                s_layout = clip.get('state_layout', None)
                o_layout = clip.get('output_layout', None)
                if s_layout is None and 'state_layout_json' in clip:
                    try:
                        s_layout = json.loads(str(clip['state_layout_json']))
                    except Exception:
                        pass
                if o_layout is None and 'output_layout_json' in clip:
                    try:
                        o_layout = json.loads(str(clip['output_layout_json']))
                    except Exception:
                        pass
                fps_val = clip.get('fps', None)
                bone_names_val = clip.get('bone_names', [])
                rot6d_spec_val, asset_to_ue_val, units_val, traj_meta_val = ({}, {}, None, {})
                meta_val = clip.get('meta', None)
                if meta_val is not None:
                    try:
                        if hasattr(meta_val, 'item'):
                            meta_val = meta_val.item()
                        if isinstance(meta_val, (bytes, bytearray)):
                            meta_val = meta_val.decode('utf-8')
                        if isinstance(meta_val, str):
                            meta_json = json.loads(meta_val)
                            fps_val = meta_json.get('fps', fps_val)
                            bone_names_val = meta_json.get('bone_names', bone_names_val)
                            rot6d_spec_val = meta_json.get('rot6d_spec', {})
                            asset_to_ue_val = meta_json.get('asset_to_ue', {})
                            units_val = meta_json.get('units', None)
                            traj_meta_val = meta_json.get('trajectory', {})
                    except Exception:
                        pass
                T = int(X.shape[0])
                meta = {'state_layout': s_layout or {}, 'output_layout': o_layout or {},
                        'fps': float(fps_val if fps_val is not None else 60.0),
                        'bone_names': list(bone_names_val) if bone_names_val is not None else [],
                        'rot6d_spec': rot6d_spec_val or {}, 'asset_to_ue': asset_to_ue_val or {},
                        'units': units_val or 'meters', 'trajectory': traj_meta_val or {}}
                if 'foot_soft_arr' in locals() and foot_soft_arr is None and (resolved_json is not None):
                    fs = _load_foot_soft_from_json(resolved_json)
                    if isinstance(fs, np.ndarray):
                        if fs.shape[0] != X.shape[0]:
                            fs = fs[:int(X.shape[0])]
                        foot_soft_arr = fs

                # --- source counting for FootEvidence soft ---
                try:
                    _src_code = 'missing'
                    if isinstance(foot_soft_arr, np.ndarray):
                        if resolved_json is not None and os.path.exists(str(resolved_json)):
                            _src_code = 'json'
                        elif foot_hits is not None:
                            _src_code = 'npz_hits'
                        else:
                            _src_code = 'npz'
                    if _src_code == 'npz':
                        _cnt_npz += 1
                    elif _src_code == 'json':
                        _cnt_json += 1
                        if _first_json_path is None:
                            _first_json_path = resolved_json
                    elif _src_code == 'npz_hits':
                        _cnt_hits += 1
                    else:
                        _cnt_missing += 1
                except Exception:
                    _cnt_missing += 1
                self.clips.append({'npz_path': p, 'X': X, 'Y': Y, 'C': C, 'meta': meta, 'contacts': contacts_arr,
                                   'foot_soft': foot_soft_arr, 'foot_json_path': resolved_json})
                cid = len(self.clips) - 1
                for s in range(0, T - self.seq_len):
                    self.index.append((cid, s))
            except Exception as e:
                print(f'[Dataset] Warning: skip {p}: {e}')
        if not self.clips:
            raise ValueError('No valid clips were loaded from the dataset.')
        # Print FootEvidence source summary
        try:
            print(
                f"[Dataset] FootEvidence(soft) source: npz={_cnt_npz} json={_cnt_json} missing={_cnt_missing} hits={_cnt_hits} / clips={len(self.clips)}")
            if _cnt_json > 0 and _first_json_path:
                print(f"[Dataset] Example JSON used: {_first_json_path}")
        except Exception as _hz_e:
            print(f"[HZ-ERR] hazard branch failed: {_hz_e}")
            pass

        # Extra diag: show mean of foot_soft (first clip) if available
        try:
            _fs0 = self.clips[0].get('foot_soft', None)
            if isinstance(_fs0, np.ndarray):
                print(f"[Dataset][Diag] foot_soft[0] mean={_fs0.mean():.4f} shape={_fs0.shape}")
        except Exception as _hz_e:
            print(f"[HZ-ERR] hazard branch failed: {_hz_e}")
            pass
        first = self.clips[0]
        self.Dx = int(first['X'].shape[1])
        self.Dy = int(first['Y'].shape[1])
        self.Dc = int(first['C'].shape[1])
        self.fps = float(first['meta']['fps'])
        self.bone_names = first['meta']['bone_names']

        def _get_layout(m, key):
            """Strict layout getter: only 'state_layout' and 'output_layout' are accepted."""
            if key not in ('state_layout', 'output_layout'):
                return {}
            li = (m or {}).get(key)
            if li is not None and hasattr(li, 'item'):
                li = li.item()
            return li if isinstance(li, dict) else {}

        self.state_layout = _get_layout(first['meta'], 'state_layout')
        self.output_layout = _get_layout(first['meta'], 'output_layout')
        if not self.output_layout and self.Dy == 629:
            self.output_layout = {'TrajectoryPos': [0, 39], 'TrajectoryDir': [39, 65], 'BonePositions': [65, 206],
                                  'BoneRotations6D': [206, 488], 'BoneVelocities': [488, 629]}
            self.traj_elem_dim = 3
            self.traj_plane_axes = [0, 1]
            print('[WARN] output_layout missing; use default 629 split.')
        else:
            traj_meta = first['meta'].get('trajectory', {}) or {}
            self.traj_elem_dim = int(traj_meta.get('elem_dim', 2))
            self.traj_plane_axes = list(traj_meta.get('plane_axes', [0, 1]))
        # 现有：
        self._up_axis = self._infer_up_axis_from_meta(first['meta'].get('asset_to_ue', {}))
        print(f'[Spec] SRC up-axis = {self._up_axis} (0:X,1:Y,2:Z)')

        # 新增（基于 JSON 模版）：
        _hand = None
        try:
            jpath = first.get('foot_json_path', None)
            if isinstance(jpath, str) and os.path.exists(jpath):
                with open(jpath, 'r', encoding='utf-8') as f:
                    _meta = (json.load(f) or {}).get('meta', {}) or {}
                _hand = _meta.get('handedness') or (_meta.get('frame_matrix', {}) or {}).get('handedness')
        except Exception:
            _hand = None

        _hand_l = (str(_hand).lower() if isinstance(_hand, str) else 'right')
        self._hand_sign = -1.0 if _hand_l.startswith('left') else 1.0
        print(f'[Spec] handedness(from JSON) = {_hand_l}  hand_sign={self._hand_sign:+.0f}')

        self.yaw_aug_deg = 0.0
        self.is_train = True
        # 统一默认：对 C 做窗口后归一化（post-transform, per-window）
        self.normalize_c = True
        self.c_norm_scope = 'window'  # 固定为 window，避免外部再传错配置

        Cs = [clip['C'] for clip in self.clips if clip['C'] is not None and clip['C'].shape[1] > 0]
        if Cs:
            Ccat = np.concatenate(Cs, axis=0).astype(np.float32, copy=False)
            self.C_mu, self.C_std = self._robust_mean_std(Ccat)
        else:
            self.C_mu, self.C_std = (None, None)

    @staticmethod
    def _robust_mean_std(arr):
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        mask = (arr >= lo) & (arr <= hi)
        safe = np.where(mask, arr, np.nan)
        mu = np.nanmean(safe, axis=0)
        std = np.nanstd(safe, axis=0)
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        std = np.nan_to_num(std, nan=1e-06, posinf=1e-06, neginf=1e-06).astype(np.float32, copy=False)
        std = np.clip(std, 1e-06, None)
        return (mu.reshape(1, -1), std.reshape(1, -1))

    @staticmethod
    def _infer_up_axis_from_meta(asset_to_ue: dict) -> int:
        s = (asset_to_ue or {}).get('source_axes_in_ue', '')
        if not isinstance(s, str):
            return 2
        mapping = {}
        for part in s.split(','):
            part = part.strip()
            if '->' not in part:
                continue
            src, ue = part.split('->', 1)
            src = src.strip().upper()
            ue = ue.strip().upper()
            if not src or not ue:
                continue
            mapping[src] = ue[-1]
        inv = {ue_axis: src for src, ue_axis in mapping.items()}
        src_up_letter = inv.get('Z', 'Z')
        return {'X': 0, 'Y': 1, 'Z': 2}.get(src_up_letter, 2)

    @staticmethod
    def _rot2_(xy, c, s):
        x, y = (xy[..., 0].copy(), xy[..., 1].copy())
        xy[..., 0] = c * x - s * y
        xy[..., 1] = s * x + c * y

    @staticmethod
    def _rot3_about_up_(arr_xyz, up_idx, c, s):
        if up_idx == 2:
            i1, i2 = (0, 1)
        elif up_idx == 0:
            i1, i2 = (1, 2)
        else:
            i1, i2 = (2, 0)
        x = arr_xyz[..., i1].copy()
        y = arr_xyz[..., i2].copy()
        arr_xyz[..., i1] = c * x - s * y
        arr_xyz[..., i2] = s * x + c * y

    def _apply_yaw_inplace(self, arr_TD, layout, deg):
        if abs(deg) < 1e-06:
            return
        t = np.deg2rad(deg)
        hs = float(getattr(self, '_hand_sign', 1.0))
        c, s = (np.cos(t), np.sin(t) * hs)

        T, D = arr_TD.shape
        if 'BonePositions' in layout:
            a, b = layout['BonePositions']
            v = arr_TD[:, a:b].reshape(T, -1, 3)
            self._rot3_about_up_(v, self._up_axis, c, s)
            arr_TD[:, a:b] = v.reshape(T, -1)
        if 'BoneVelocities' in layout:
            a, b = layout['BoneVelocities']
            v = arr_TD[:, a:b].reshape(T, -1, 3)
            self._rot3_about_up_(v, self._up_axis, c, s)
            arr_TD[:, a:b] = v.reshape(T, -1)
        if 'BoneRotations6D' in layout:
            a, b = layout['BoneRotations6D']
            vec = arr_TD[:, a:b].reshape(T, -1, 2, 3)
            self._rot3_about_up_(vec[..., 0, :], self._up_axis, c, s)
            self._rot3_about_up_(vec[..., 1, :], self._up_axis, c, s)
            arr_TD[:, a:b] = vec.reshape(T, -1)
        horiz = sorted(list({0, 1, 2} - {self._up_axis}))
        plane_axes = list(self.traj_plane_axes)
        if 'TrajectoryPos' in layout:
            a, b = layout['TrajectoryPos']
            dim = int(self.traj_elem_dim)
            if dim == 3:
                v = arr_TD[:, a:b].reshape(T, -1, 3)
                self._rot3_about_up_(v, self._up_axis, c, s)
                arr_TD[:, a:b] = v.reshape(T, -1)
            elif dim == 2 and sorted(plane_axes) == horiz:
                v = arr_TD[:, a:b].reshape(T, -1, 2)
                self._rot2_(v, c, s)
                arr_TD[:, a:b] = v.reshape(T, -1)
        if 'TrajectoryDir' in layout and int(self.traj_elem_dim) == 2 and (sorted(plane_axes) == horiz):
            a, b = layout['TrajectoryDir']
            v = arr_TD[:, a:b].reshape(T, -1, 2)
            self._rot2_(v, c, s)
            n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-08
            v /= n
            arr_TD[:, a:b] = v.reshape(T, -1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        clip_id, s = self.index[idx]
        s = int(s)
        e = s + int(self.seq_len)
        clip = self.clips[clip_id]

        # Y 已在转换阶段对齐到 “下一帧”，这里不要再 +1
        Xv = clip['X'][s:e]
        Yv = clip['Y'][s:e]
        C_in_win = clip['C'][s:e]
        C_tgt_win = clip['C'][s + 1:e + 1]
        # 双保险
        assert len(Xv) == len(Yv), f"X/Y length mismatch: {len(Xv)} vs {len(Yv)} at s={s}, e={e}"

        need_aug = self.is_train and getattr(self, 'yaw_aug_deg', 0.0) > 0.0
        X = Xv.copy() if need_aug else Xv
        Y = Yv.copy() if need_aug else Yv
        C_in = C_in_win.copy() if need_aug else C_in_win
        C_tgt = C_tgt_win.copy() if need_aug else C_tgt_win
        C_tgt_raw = C_tgt.copy()

        if self.is_train and self.yaw_aug_deg > 0.0:
            deg = float(np.random.uniform(-self.yaw_aug_deg, self.yaw_aug_deg))
            self._apply_yaw_inplace(X, self.state_layout, deg)
            self._apply_yaw_inplace(Y, self.output_layout, deg)
            if C_in.shape[1] >= 2:
                xy = C_in[:, -2:].copy()
                t = np.deg2rad(deg)
                hs = float(getattr(self, '_hand_sign', 1.0))
                c_t, s_t = (np.cos(t), np.sin(t) * hs)

                x, y = (xy[:, 0].copy(), xy[:, 1].copy())
                xy[:, 0] = c_t * x - s_t * y
                xy[:, 1] = s_t * x + c_t * y
                n = np.linalg.norm(xy, axis=-1, keepdims=True) + 1e-08
                C_in[:, -2:] = xy / n
            if C_tgt.shape[1] >= 2:
                xy2 = C_tgt[:, -2:].copy()
                t = np.deg2rad(deg)
                hs = float(getattr(self, '_hand_sign', 1.0))
                c_t, s_t = (np.cos(t), np.sin(t) * hs)
                x2, y2 = (xy2[:, 0].copy(), xy2[:, 1].copy())
                xy2[:, 0] = c_t * x2 - s_t * y2
                xy2[:, 1] = s_t * x2 + c_t * y2
                n2 = np.linalg.norm(xy2, axis=-1, keepdims=True) + 1e-08
                C_tgt[:, -2:] = xy2 / n2

        if self.normalize_c and (C_in.shape[1] > 0):
            # 窗口后归一：对“当前窗口、且已完成全部确定性变换后的 C”做鲁棒均/方
            mu, std = self._robust_mean_std(C_in)
            # 极端情况下做一次兜底（比如某些通道常数）：

            try:
                std = np.clip(np.nan_to_num(std, nan=1e-6, posinf=1e-6, neginf=1e-6), 1e-6, None)
                mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                # 兜底到全局统计量（仍然做 clip）
                mu = np.nan_to_num(self.C_mu, nan=0.0, posinf=0.0, neginf=0.0) if (self.C_mu is not None) else 0.0
                std = np.nan_to_num(self.C_std, nan=1e-6, posinf=1e-6, neginf=1e-6) if (
                            self.C_std is not None) else 1e-6
                std = np.clip(std, 1e-6, None)
            C_in = (C_in - mu) / std
            C_tgt = (C_tgt - mu) / std
            np.nan_to_num(C_in, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(C_tgt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(C_in, -6.0, 6.0, out=C_in)
            np.clip(C_tgt, -6.0, 6.0, out=C_tgt)

        sample = {'motion': torch.from_numpy(X).float(), 'gt_motion': torch.from_numpy(Y).float()}
        # 边界一致性所需的索引信息（在 collate 中保持为 [B] 标量）
        try:
            sample['clip_id'] = torch.tensor(int(clip_id), dtype=torch.int64)
            sample['start']   = torch.tensor(int(s), dtype=torch.int64)
        except Exception:
            sample['clip_id'] = torch.tensor(0, dtype=torch.int64)
            sample['start']   = torch.tensor(int(s), dtype=torch.int64)

        # expose paths for lazy JSON fallback
        if clip.get('foot_json_path'): sample['foot_json_path'] = clip['foot_json_path']
        if clip.get('npz_path'): sample['npz_path'] = clip['npz_path']
        if clip.get('json_candidates'): sample['json_path'] = clip['json_candidates']

        if C_in.shape[1] > 0:
            sample['cond_in'] = torch.from_numpy(C_in.astype(np.float32, copy=False)).float()
            sample['cond_tgt'] = torch.from_numpy(C_tgt.astype(np.float32, copy=False)).float()
            sample['cond_tgt_raw'] = torch.from_numpy(C_tgt_raw.astype(np.float32, copy=False)).float()
        try:
            foot_soft = clip.get('foot_soft', None)
            if foot_soft is not None:
                fs = foot_soft[s:e]
                if isinstance(fs, np.ndarray):
                    evi = torch.from_numpy(fs.astype(np.float32, copy=False)).float()
                else:
                    evi = torch.tensor(fs, dtype=torch.float32)
                if torch.is_tensor(evi):
                    try:
                        _sum = float(evi.sum().detach().cpu())
                    except Exception:
                        _sum = float(evi.sum().item()) if hasattr(evi, 'sum') else 0.0
                    if _sum == 0.0 and not hasattr(self, "_once_zero_evi_warn"):
                        print("[HZ-GUARD] evi_soft window sum==0 at s=%d e=%d" % (int(s), int(e)))
                        self._once_zero_evi_warn = True
                sample['evi_soft'] = evi
                sample['contact_state'] = (evi >= float(getattr(self, 'evt_hard_thresh', 0.5))).to(torch.float32)
                if evi.size(0) >= 2:
                    sw = (sample['contact_state'][1:] != sample['contact_state'][:-1]).to(torch.float32)
                    sample['hazard_target'] = sw
            else:
                contacts = clip.get('contacts', None)
                if contacts is not None:
                    cont = contacts[s + 1:e + 1]
                    c = torch.from_numpy(cont).to(torch.int64)
                    if c.size(-1) >= 4:
                        L = ((c[:, 0] > 0) | (c[:, 1] > 0)).to(torch.float32)
                        R = ((c[:, 2] > 0) | (c[:, 3] > 0)).to(torch.float32)
                    elif c.size(-1) == 2:
                        L, R = ((c[:, 0] > 0).to(torch.float32), (c[:, 1] > 0).to(torch.float32))
                    else:
                        L = torch.zeros(c.size(0), dtype=torch.float32)
                        R = torch.zeros_like(L)
                    evi = torch.stack([L, R], dim=-1)
                    sample['evi_soft'] = evi
                    sample['contact_state'] = (evi > 0.5).to(torch.float32)
                    if evi.size(0) >= 2:
                        sw = (sample['contact_state'][1:] != sample['contact_state'][:-1]).to(torch.float32)
                        sample['hazard_target'] = sw
        except Exception as _hz_e:
            print(f"[HZ-ERR] hazard branch failed: {_hz_e}")
            pass
        # --- strict fallback: always provide evi_soft (T,2) to keep training deterministic ---
        if 'evi_soft' not in sample or sample['evi_soft'] is None:
            T_win = int(e - s)
            import torch as _torch
            sample['evi_soft'] = _torch.zeros((T_win, 2), dtype=torch.float32)
        return sample

    def set_cond_norm_stats(self, mu, std, inherit=False):
        if isinstance(mu, torch.Tensor):
            mu = mu.detach().cpu().numpy()
        if isinstance(std, torch.Tensor):
            std = std.detach().cpu().numpy()
        self.C_mu = np.array(mu, dtype=np.float32).reshape(1, -1).copy()
        self.C_std = np.array(std, dtype=np.float32).reshape(1, -1).copy()
        self.norm_stats_inherited = bool(inherit)

class EventMotionModel(nn.Module):
    """
    无相位（phase-free）的动作预测模型。
    输入: state_t (X), cond_t
    输出: motion_{t+1} (Y 维)
    """

    def __init__(self, in_state_dim: int, out_motion_dim: int, cond_dim: int=0, hidden_dim: int=256, num_layers: int=3, num_heads: int=4, dropout: float=0.1, context_len: int=32, use_layer_norm: bool=True):
        super().__init__()
        self.in_state_dim = int(in_state_dim)
        self.out_motion_dim = int(out_motion_dim)
        self.cond_dim = int(cond_dim)
        self.context_len = int(context_len)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        input_dim = self.in_state_dim + self.cond_dim
        enc = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(), nn.ReLU(), nn.Dropout(dropout)]
        self.shared_encoder = nn.Sequential(*enc)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self._pasa_heads = num_heads
        self._pasa_dhead = hidden_dim // num_heads
        self._pasa_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._pasa_lnq = nn.LayerNorm(hidden_dim)
        self._pasa_film = _CondFiLM(cond_dim=self.cond_dim, hidden_dim=128, film_dim=hidden_dim)
        self.coupling_norm = nn.LayerNorm(hidden_dim)
        self.motion_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_motion_dim))
        self.event_channels = getattr(self, 'event_channels', 2)
        self.events_step_head = _StepEventsHead(hidden_dim, event_dim=self.event_channels)

    def forward(self, state: torch.Tensor, cond: Optional[torch.Tensor]=None, hidden: Optional[torch.Tensor]=None, kv_mem: Optional[torch.Tensor]=None):
        """
        支持单步 [B,D] 或序列 [B,T,D]。
        返回: out, hidden_new, attn_weights, kv_mem
        """
        is_single = state.ndim == 2
        if is_single:
            state = state.unsqueeze(1)
            if cond is not None:
                cond = cond.unsqueeze(1)
        if cond is None and self.cond_dim > 0:
            cond = torch.zeros(state.shape[:-1] + (self.cond_dim,), device=state.device, dtype=state.dtype)
        x = torch.cat([state] + ([cond] if cond is not None else []), dim=-1)
        if not torch.isfinite(x).all():
            bad_mask = (~torch.isfinite(x))
            _nz = bad_mask.nonzero(as_tuple=False)
            bad = _nz[0].tolist() if _nz.numel() > 0 else []
            s_ok = bool(torch.isfinite(state).all())
            c_ok = True
            if cond is not None and cond.is_floating_point():
                c_ok = bool(torch.isfinite(cond).all())
            raise RuntimeError(f'[Guard] x to shared_encoder has non-finite at {bad} | state_finite={s_ok} cond_finite={c_ok}')
        _lin0 = self.shared_encoder[0]
        _w_ok = bool(torch.isfinite(_lin0.weight).all())
        _b_ok = _lin0.bias is None or bool(torch.isfinite(_lin0.bias).all())
        if not _w_ok or not _b_ok:
            raise RuntimeError('[Guard] shared_encoder.0 has non-finite params; refusing to auto-reinit in forward')
        x = torch.nan_to_num(x, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
        x = x.clamp(-100.0, 100.0)
        with torch.no_grad():
            for _idx, _mod in enumerate(self.shared_encoder):
                if isinstance(_mod, torch.nn.Linear):
                    _ok_w = bool(torch.isfinite(_mod.weight).all())
                    _ok_b = _mod.bias is None or bool(torch.isfinite(_mod.bias).all())
                    if not _ok_w or not _ok_b:
                        raise RuntimeError(f"[Guard] shared_encoder.{_idx} has non-finite params; refusing to auto-reinit in forward")
        lin0 = self.shared_encoder[0]
        act1 = self.shared_encoder[1]
        z0 = lin0(x)
        y1 = act1(z0)
        h = self.shared_encoder[2:](y1)
        h_temporal, hidden_new = self.temporal(h, hidden)
        h_temporal = h_temporal + self.residual_proj(x)
        h_temporal = torch.nan_to_num(h_temporal, nan=0.0, posinf=1000000.0, neginf=-1000000.0).clamp(-100.0, 100.0)
        kv_mem = torch.cat([kv_mem, h_temporal.detach()], dim=1) if kv_mem is not None else h_temporal.detach()
        if kv_mem.size(1) > self.context_len:
            kv_mem = kv_mem[:, -self.context_len:]
        B, Tq, H = h_temporal.shape
        L = kv_mem.size(1)
        Dh = self._pasa_dhead
        scale = 1.0 / math.sqrt(max(1, Dh))
        cond_in = torch.zeros(B, self.cond_dim, device=h_temporal.device, dtype=h_temporal.dtype)
        g, b = self._pasa_film(cond_in)
        q_in = self._pasa_lnq(h_temporal)
        Q = self._pasa_q(q_in).view(B, Tq, self._pasa_heads, Dh).transpose(1, 2)
        K = self._pasa_k(kv_mem).view(B, L, self._pasa_heads, Dh).permute(0, 2, 1, 3)
        V = self._pasa_v(kv_mem).view(B, L, self._pasa_heads, Dh).permute(0, 2, 1, 3)
        attn = torch.softmax(Q * scale @ K.transpose(-1, -2), dim=-1)
        ctx = (attn @ V).transpose(1, 2).contiguous().view(B, Tq, -1)
        attn_out = self._pasa_o(ctx)
        h_final = self.coupling_norm((h_temporal + attn_out) * (1 + g).unsqueeze(1) + b.unsqueeze(1))
        out = self.motion_head(h_final)
        if is_single:
            out = out.squeeze(1)
        hz_seq = self.events_step_head(h_temporal)
        setattr(self, '_last_hidden_seq', h_temporal.detach())
        attn_weights = attn.mean(dim=1) if attn is not None else None
        return {'out': out.squeeze(1) if is_single else out, 'hidden': hidden_new, 'attn': attn_weights, 'kv_mem': kv_mem, 'hazard_logits': hz_seq.squeeze(1) if is_single else hz_seq}

class MotionJointLoss(nn.Module):
    def _pick_s_eff(self, D: int, device, dtype):
        """
        Return per-dim effective scale tensor of length D.
        Priority: match Y length -> match X length -> merge by layout -> fallback (median).
        Strictly uses 'state_layout' and 'output_layout' from self.meta if present.
        """
        import torch
        if getattr(self, 's_eff_y', None) is not None and self.s_eff_y.numel() == D:
            return self.s_eff_y.to(device=device, dtype=dtype).view(1, -1)
        if getattr(self, 's_eff_x', None) is not None and self.s_eff_x.numel() == D:
            return self.s_eff_x.to(device=device, dtype=dtype).view(1, -1)
        if getattr(self, 'meta', None):
            y_layout = (self.meta.get('output_layout') or {})
            x_layout = (self.meta.get('state_layout') or {})
            s = torch.empty(D, device=device, dtype=dtype)
            s[:] = float('nan')
            if getattr(self, 's_eff_y', None) is not None and isinstance(y_layout, dict):
                for k, v in y_layout.items():
                    st, sz = int(v[0]), int(v[1])
                    if st >= 0 and st + sz <= self.s_eff_y.numel() and st + sz <= D:
                        s[st:st+sz] = self.s_eff_y[st:st+sz].to(device=device, dtype=dtype)
            if getattr(self, 's_eff_x', None) is not None and isinstance(x_layout, dict):
                for k, v in x_layout.items():
                    st, sz = int(v[0]), int(v[1])
                    if st >= 0 and st + sz <= self.s_eff_x.numel() and st + sz <= D:
                        m = s[st:st+sz]
                        mask = torch.isnan(m)
                        if mask.any():
                            m[mask] = self.s_eff_x[st:st+sz].to(device=device, dtype=dtype)[mask]
                            s[st:st+sz] = m
            if torch.isnan(s).any():
                avail = s[~torch.isnan(s)]
                fill = (avail.median() if avail.numel() else torch.tensor(1e-3, device=device, dtype=dtype))
                s[torch.isnan(s)] = fill
            return s.view(1, -1)
        return torch.full((1, D), 1e-3, device=device, dtype=dtype)


    def __init__(self, w_attn_reg: float=0.01, w_speed: float=0.2, w_dir: float=0.0, w_speed_curve: float=0.2, output_layout: Dict[str, Any]=None, fps: float=60.0, traj_hz: float=60.0, use_huber: bool=True, huber_delta: float=1.0, rot6d_spec: Dict[str, Any]=None, w_rot_geo: float=0.0, w_rot_ortho: float=0.0, ignore_motion_groups: str='', w_rot_delta: float=1.0, w_rot_delta_root: float=0.0, w_rot_log: float=0.0):
        super().__init__()
        self.w_attn_reg = float(w_attn_reg)
        self.w_speed = float(w_speed)
        self.w_dir = float(w_dir)
        self.w_speed_curve = float(w_speed_curve)
        self.use_huber = bool(use_huber)
        self.huber_delta = float(huber_delta)
        self.w_rot_geo = float(w_rot_geo)
        self.w_rot_ortho = float(w_rot_ortho)
        self.w_rot_delta = float(w_rot_delta)
        self.w_rot_delta_root = float(w_rot_delta_root)
        self.w_rot_log = float(w_rot_log)
        self.fps = float(fps)
        self.traj_hz = float(traj_hz)
        self.output_layout = output_layout or {}
        self.rot6d_spec = rot6d_spec or {}
        layout = self.output_layout or {}
        inner = layout.get('slices') if isinstance(layout.get('slices'), dict) else layout
        total_dim_hint = next((int(inner[k]) for k in ('output_dim','D','dim','size','total_dim') if isinstance(inner.get(k), int)), None)
        self.group_slices = {name: sl for name, sl in ((n, parse_layout_entry(v, n, total_dim_hint)) for n, v in inner.items()) if isinstance(name, str) and isinstance(sl, slice)}
        self.ignore_groups = [g.strip() for g in (ignore_motion_groups or '').split(',') if g.strip()]
        self.attn_lambda_local = getattr(self, 'attn_lambda_local', 0.02)
        self.attn_lambda_entropy = getattr(self, 'attn_lambda_entropy', 0.0)
        self._warned_bad_rot6d = False

        # z-safe loss params (filled later from template)
        self.s_eff_y = None
        self.sigma_cap = 6.0
        self.s_eff_x = None
        self.meta = None
    def _forward_base_inner(self, pred_motion: torch.Tensor, gt_motion: torch.Tensor, attn_weights=None) -> tuple[torch.Tensor, dict[str, float]]:
        """
        参数:
            pred_motion: [B,T,D] or [T,D] or [B,D]
            gt_motion:   同形状
            attn_weights: None 或 [B,H,T,T]/[L,B,H,T,T] 或 list/tuple/dict 的任意嵌套
        返回:
            loss 标量, 分项 dict (float)
        """
        Z = lambda v: gt_motion.new_tensor(float(v))
        pm, gm = (pred_motion, gt_motion)
        assert pm.shape == gm.shape, f'pred/gt shape mismatch: {pm.shape} vs {gm.shape}'
        D = pm.shape[-1]
        # === z-safe motion loss (优先使用) ===
        s = self._pick_s_eff(D=pm.shape[-1], device=pm.device, dtype=pm.dtype)

        
        # --- z-safe on non-rot6d dims; yaw uses wrapped difference ---
        # slices
        yaw_sl = self.group_slices.get('RootYaw', None) or self.group_slices.get('Yaw', None)
        rot_sl = self.group_slices.get('BoneRotations6D', None)

        # residual in raw domain semantics for linear dims
        r = pm - gm
        # yaw: wrap to [-pi, pi]
        if isinstance(yaw_sl, slice):
            r[..., yaw_sl] = torch.remainder(r[..., yaw_sl] + math.pi, 2.0 * math.pi) - math.pi

        # s_eff per-dim
        s_use = s.view(*([1] * (r.ndim - 1)), -1).clamp_min(1e-6)

        # mask out rot6d dims from this term
        if isinstance(rot_sl, slice):
            mask = torch.ones(r.shape[-1], dtype=torch.float32, device=r.device)
            mask[rot_sl] = 0.0
        else:
            mask = torch.ones(r.shape[-1], dtype=torch.float32, device=r.device)

        r_scaled = r / s_use
        sc = float(getattr(self, 'sigma_cap', 6.0))
        r_scaled = r_scaled / (1.0 + r_scaled.abs() / sc)

        # reduce only over included dims
        w = mask.view(*([1] * (r.dim() - 1)), -1)
        denom = w.sum(dim=-1).clamp_min(1.0)  # per-timestep denom if needed
        # square error and mean over dims with mask
        err2 = (r_scaled * r_scaled) * w
        l_motion = (err2.sum(dim=-1) / denom).mean()

        # === 其他辅助项 ===
        if attn_weights is not None:
            l_attn = self.compute_attention_regularization(attn_weights, geomask=None)
        else:
            l_attn = gm.new_zeros(())
        l_speed = self.compute_speed_loss(pm, gm) if self.w_speed > 0 else Z(0.0)
        l_dir = self.compute_dir_loss(pm, gm) if self.w_dir > 0 else Z(0.0)
        l_speed_curve = self.compute_speed_curve_loss(pm, gm) if self.w_speed_curve > 0 else Z(0.0)
        l_geo = self.compute_rot6d_geo_loss(pm, gm) if self.w_rot_geo > 0 else Z(0.0)
        l_ortho = self.compute_rot6d_ortho_loss(pm) if self.w_rot_ortho > 0 else Z(0.0)
        loss = l_motion + self.w_attn_reg * l_attn + self.w_speed * l_speed + self.w_dir * l_dir + \
               self.w_speed_curve * l_speed_curve + self.w_rot_geo * l_geo + self.w_rot_ortho * l_ortho
        stats = {
            'motion': float(l_motion.detach().cpu()), 'attn': float(l_attn.detach().cpu()),
            'speed': float(l_speed.detach().cpu()), 'dir': float(l_dir.detach().cpu()),
            'speed_curve': float(l_speed_curve.detach().cpu()), 'rot_geo': float(l_geo.detach().cpu()),
            'rot_ortho': float((self.w_rot_ortho * l_ortho).detach().cpu()),
            'rot_ortho_raw': float(l_ortho.detach().cpu())
        }
        return loss, stats

    def _slice_if_exists(self, name: str, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        从 self.group_slices 中获取预先解析好的 slice，并应用于张量。
        """
        sl = self.group_slices.get(name)

        # 因为 self.group_slices 只包含 slice 对象，所以只需做一次类型检查即可。
        if isinstance(sl, slice):
            return X[..., sl]

        return None

    @staticmethod
    def _build_ignore_mask(D: int, group_slices: Dict[str, slice], ignore_groups: list, device) -> torch.Tensor:
        """
        返回一个布尔 mask，True=参与计算，False=忽略。
        """
        mask = torch.ones(D, dtype=torch.bool, device=device)
        for g in ignore_groups:
            sl = group_slices.get(g, None)
            if sl is not None:
                mask[sl] = False
        return mask

    def compute_attention_regularization(self, attn_weights, geomask=None):
        """
        返回一个标量 loss：
        - 支持 Tensor: [B,H,T,T] 或 [L,B,H,T,T]
        - 支持 list/tuple/dict: 递归展开后逐个累加
        - geomask: None 或可广播到 [..., T, T] 的掩码（1=允许区域，0=不鼓励区域）
        """
        if attn_weights is None:
            if geomask is not None and torch.is_tensor(geomask):
                return torch.zeros((), device=geomask.device, dtype=geomask.dtype)
            return torch.tensor(0.0)

        def _flatten_items(x):
            if torch.is_tensor(x):
                return [x]
            if isinstance(x, (list, tuple)):
                items = []
                for y in x:
                    items.extend(_flatten_items(y))
                return items
            if isinstance(x, dict):
                items = []
                for y in x.values():
                    items.extend(_flatten_items(y))
                return items
            return []
        items = _flatten_items(attn_weights)
        if len(items) == 0:
            return torch.tensor(0.0, device=geomask.device if geomask is not None and torch.is_tensor(geomask) else None)
        loss_total = None
        count = 0
        for A in items:
            if A.dim() == 5:
                L, B, H, T, _ = A.shape
                A = A.reshape(L * B * H, T, T)
            elif A.dim() == 4:
                B, H, T, _ = A.shape
                A = A.reshape(B * H, T, T)
            elif A.dim() == 3:
                T = A.shape[-1]
            else:
                continue
            A = A.float()
            A = A / A.sum(-1, keepdim=True).clamp_min(1e-06)
            device = A.device
            if geomask is not None:
                try:
                    M = 1.0 - geomask
                    loss_local = (A * M).mean()
                except Exception:
                    gm = geomask
                    if torch.is_tensor(gm):
                        if gm.dim() == 2:
                            gm = gm.view(1, T, T)
                        elif gm.dim() == 4:
                            gm = gm.mean(0)
                        M = 1.0 - gm
                        loss_local = (A * M).mean()
                    else:
                        idx = torch.arange(T, device=device)
                        Dmat = (idx[None, :] - idx[:, None]).abs().float()
                        Dmat = Dmat / Dmat.max().clamp_min(1.0)
                        loss_local = (A * Dmat).mean()
            else:
                idx = torch.arange(T, device=device)
                Dmat = (idx[None, :] - idx[:, None]).abs().float()
                Dmat = Dmat / Dmat.max().clamp_min(1.0)
                loss_local = (A * Dmat).mean()
            Aeps = A.clamp_min(1e-06)
            entropy = -(Aeps * Aeps.log()).sum(-1).mean()
            loss_attn = self.attn_lambda_local * loss_local + self.attn_lambda_entropy * -entropy
            loss_total = loss_attn if loss_total is None else loss_total + loss_attn
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=items[0].device)
        return loss_total / count

    def compute_speed_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        期望 layout 包含:
        - "RootSpeed" 或 "TrajSpeed"（1 维）或 "RootVel"（2/3 维，取范数）
        若无对应切片 -> 返回 0
        """
        Z = lambda v: gt.new_tensor(float(v))
        sp_p = self._slice_if_exists('RootSpeed', pred) or self._slice_if_exists('TrajSpeed', pred)
        sp_g = self._slice_if_exists('RootSpeed', gt) or self._slice_if_exists('TrajSpeed', gt)
        if sp_p is None or sp_g is None:
            vp = self._slice_if_exists('RootVel', pred)
            vg = self._slice_if_exists('RootVel', gt)
            if vp is None or vg is None:
                return Z(0.0)
            sp_p = vp.norm(dim=-1, keepdim=True)
            sp_g = vg.norm(dim=-1, keepdim=True)
        return F.l1_loss(sp_p, sp_g)

    def compute_dir_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        期望 layout 含 "TrajectoryDir"（2 或 3 维方向向量，已单位化）
        若无 -> 0
        """
        Z = lambda v: gt.new_tensor(float(v))
        dp = self._slice_if_exists('TrajectoryDir', pred)
        dg = self._slice_if_exists('TrajectoryDir', gt)
        if dp is None or dg is None:
            return Z(0.0)
        dp = F.normalize(dp, dim=-1)
        dg = F.normalize(dg, dim=-1)
        c = (dp * dg).sum(dim=-1)
        return (1.0 - c).mean()

    def compute_speed_curve_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: gt.new_tensor(float(v))

        # 1) 有 Root/TrajSpeed 就直接用
        sp_p = self._slice_if_exists('RootSpeed', pred) or self._slice_if_exists('TrajSpeed', pred)
        sp_g = self._slice_if_exists('RootSpeed', gt) or self._slice_if_exists('TrajSpeed', gt)
        if sp_p is not None and sp_g is not None and sp_p.dim() >= 2:
            dp = sp_p[..., 1:, :] - sp_p[..., :-1, :]
            dg = sp_g[..., 1:, :] - sp_g[..., :-1, :]
            return F.smooth_l1_loss(dp, dg, beta=0.5)

        # 2) 回退：基于 Rot6D 的关节角速度曲线
        pr = self._slice_if_exists('BoneRotations6D', pred)
        gr = self._slice_if_exists('BoneRotations6D', gt)
        if pr is None or gr is None or pr.dim() < 2:
            return Z(0.0)
        D = pr.shape[-1]
        if D % 6 != 0:
            return Z(0.0)
        J = D // 6

        # 正交化：你的 reproject_rot6d 已修好支持 (..., D)
        pr = reproject_rot6d(pr)
        gr = reproject_rot6d(gr)

        # 6D -> 3x3
        Rp = rot6d_to_matrix(pr.view(*pr.shape[:-1], J, 6))
        Rg = rot6d_to_matrix(gr.view(*gr.shape[:-1], J, 6))

        # 逐帧相对旋转（角位移）
        RtR_p = torch.matmul(Rp[..., 1:, :, :].transpose(-1, -2), Rp[..., :-1, :, :])
        RtR_g = torch.matmul(Rg[..., 1:, :, :].transpose(-1, -2), Rg[..., :-1, :, :])
        tr_p = RtR_p[..., 0, 0] + RtR_p[..., 1, 1] + RtR_p[..., 2, 2]
        tr_g = RtR_g[..., 0, 0] + RtR_g[..., 1, 1] + RtR_g[..., 2, 2]
        cos_p = ((tr_p - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        cos_g = ((tr_g - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta_p = torch.arccos(cos_p)  # [*, T-1, J]
        theta_g = torch.arccos(cos_g)

        # 全身“速度曲线” & 二阶差分对齐
        sp_p = theta_p.mean(dim=-1, keepdim=True)  # [*, T-1, 1]
        sp_g = theta_g.mean(dim=-1, keepdim=True)
        dp = sp_p[..., 1:, :] - sp_p[..., :-1, :]
        dg = sp_g[..., 1:, :] - sp_g[..., :-1, :]
        return F.smooth_l1_loss(dp, dg, beta=0.5)

    def _maybe_get_rot6d(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        若存在 "BoneRotations6D" 切片，则返回该切片；否则 None。
        """
        rot = self._slice_if_exists('BoneRotations6D', X)
        return rot

    def compute_rot6d_geo_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        Z = lambda v: gt.new_tensor(float(v))
        sl = self.group_slices.get('BoneRotations6D', None)

        # 1) 只取 rot6d 的扁平切片 (…, D)，不要先 reshape 到 (J,6)
        pr = self._maybe_get_rot6d(pred)  # (…, D) or None
        gr = self._maybe_get_rot6d(gt)  # (…, D) or None
        if pr is None or gr is None:
            return Z(0.0)

        D = pr.shape[-1]
        if D % 6 != 0:
            if not self._warned_bad_rot6d:
                self._warned_bad_rot6d = True
                print(
                    f"[Loss][WARN] BoneRotations6D slice dim={D} (not multiple of 6). slice={sl}, total_pred_D={pred.shape[-1]}. Skip rot6d_geo this run.")
            return Z(0.0)
        J = D // 6

        # 2) 训练端反归一化：在扁平 (…, D) 上做 raw = z*StdY + MuY
        try:
            sl_b = self.group_slices.get('BoneRotations6D', None)
            if isinstance(sl_b, slice) and getattr(self, "mu_y", None) is not None and getattr(self, "std_y",
                                                                                               None) is not None:
                st = int(sl_b.start);
                ln = int(sl_b.stop - sl_b.start)
                if ln == D:  # 只有当这段就是完整 rot6d 段时才生效
                    import torch
                    mu = torch.as_tensor(self.mu_y, device=pr.device, dtype=pr.dtype)[..., st:st + ln]
                    sd = torch.as_tensor(self.std_y, device=pr.device, dtype=pr.dtype)[..., st:st + ln].clamp(min=1e-6)
                    while mu.dim() < pr.dim():
                        mu = mu.unsqueeze(0);
                        sd = sd.unsqueeze(0)
                    pr = pr * sd + mu
                    gr = gr * sd + mu
                    if not hasattr(self, "_train_denorm_hit"):
                        print("[GeoLoss] TRAIN denorm(Y.rot6d) applied on flat D.")
                        self._train_denorm_hit = True

        except Exception:
            pass

        # 3) 先在扁平 (…, D) 上做 reproject，再 reshape 到 (…, J, 6)
        pr = reproject_rot6d(pr)  # (…, D)
        gr = reproject_rot6d(gr)  # (…, D)
        pr = pr.view(*pr.shape[:-1], J, 6)  # (…, J, 6)
        gr = gr.view(*gr.shape[:-1], J, 6)  # (…, J, 6)

        # 4) geodesic
        Rp = rot6d_to_matrix(pr)
        Rg = rot6d_to_matrix(gr)
        RtR = torch.matmul(Rp.transpose(-1, -2), Rg)
        tr = RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]
        cos = (tr - 1.0) * 0.5
        cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.arccos(cos)
        return theta.mean()


    def compute_rot6d_ortho_loss(self, pred: torch.Tensor) -> torch.Tensor:
        
        """Ortho penalty on **raw 6D** (pre-GS):
        encourage unit-length columns and mutual orthogonality.
        This must NOT use rot6d_to_matrix (which orthonormalizes and would yield ~0 loss).
        """
        Z = lambda v: pred.new_tensor(float(v))
        pr = self._maybe_get_rot6d(pred)  # (..., D) or None
        if pr is None:
            return Z(0.0)
        D = pr.shape[-1]
        if D % 6 != 0:
            if not getattr(self, '_warned_bad_rot6d_ortho', False):
                self._warned_bad_rot6d_ortho = True
                print(f"[Loss][WARN] BoneRotations6D slice dim={D} not multiple of 6. Skip rot6d_ortho.")
            return Z(0.0)
        J = D // 6
        a6 = pr.view(*pr.shape[:-1], J, 6)  # (..., J, 6) raw 6D (no GS)
        v1 = a6[..., 0:3]
        v2 = a6[..., 3:6]
        len_p = (v1.norm(dim=-1) - 1.0).pow(2) + (v2.norm(dim=-1) - 1.0).pow(2)
        ortho_p = (v1.mul(v2).sum(dim=-1)).pow(2)
        return (len_p + ortho_p).mean()


    def forward(self, pred_motion, gt_motion, attn_weights=None, batch=None):
        # 统一拿出模型输出（可能是 dict 或 tensor）
        pm = pred_motion.get('out') if isinstance(pred_motion, dict) else pred_motion

        # 这里的 _forward_base_inner 内部已经计算了 hazard（软接触）相关的 loss/stats
        base_out = self._forward_base_inner(pm, gt_motion, attn_weights=attn_weights)  # type: ignore
        if isinstance(base_out, tuple):
            loss, stats = base_out
        else:
            loss, stats = base_out, {}


        # === Hazard（软接触）可选项：根据 evi_soft / hazard_target 叠加 ===
        try:
            hz_dbg = int(getattr(self, 'hz_dbg', 0))
            if (float(getattr(self, 'w_hazard_bce', 0.0)) > 0.0) and (batch is not None):
                hz_logits = None
                if isinstance(pred_motion, dict):
                    hz_logits = pred_motion.get('hazard_logits', None)
                evi_soft = None
                if isinstance(batch, dict):
                    evi_soft = batch.get('hazard_target', None)
                    if evi_soft is None:
                        evi_soft = batch.get('evi_soft', None)
                    if evi_soft is None:
                        evi_soft = batch.get('cond_tgt_raw', None)

                if (hz_logits is None) and hasattr(self, '_last_hazard_logits_seq'):
                    hz_logits = getattr(self, '_last_hazard_logits_seq')

                if (hz_logits is not None) and (evi_soft is not None):
                    hz_loss, hz_stats = self._hazard_losses_soft_only(
                        hz_logits, evi_soft,
                        w_bce=float(getattr(self, 'w_hazard_bce', 1.0)),
                        w_smooth=float(getattr(self, 'w_hazard_smooth', 0.0)),
                        label_smoothing=float(getattr(self, 'evt_bce_w_base', 0.0)),
                        smooth_on_logits=bool(getattr(self, 'smooth_on_logits', False)),
                        smooth_edge_guard=int(getattr(self, 'smooth_edge_guard', 0)),
                        evt_hard_thresh=float(getattr(self, 'evt_hard_thresh', 0.5)),
                        pos_weight=getattr(self, 'hz_pos_weight', None),
                    )
                    # 累加到总损失并合并统计
                    loss = loss + hz_loss
                    if isinstance(stats, dict):
                        stats.update(hz_stats)
                    # 供 Trainer 控制台打印使用
                    try:
                        setattr(self, "_last_hz_stats", {k: (v.detach() if hasattr(v, "detach") else v) for k,v in (hz_stats or {}).items()})
                    except Exception:
                        pass
                    # 可选：在 forward 内部也打印一行（仅首个 batch）
                    if hz_dbg > 0 and int(getattr(self, "_hz_dbg_count", 0)) == 0:
                        print(f"[Loss][HZ] BCE={float(hz_stats.get('hz_bce', 0.0)):.4f} "
                              f"Smooth={float(hz_stats.get('hz_smooth', 0.0)):.4f} "
                              f"mask={float(hz_stats.get('hz_mask_ratio', 0.0)):.2f} "
                              f"tgt_mean={float(hz_stats.get('target_mean', 0.0)):.3f}")
                        self._hz_dbg_count = 1
                else:
                    if hz_dbg > 0 and int(getattr(self, "_hz_dbg_count", 0)) == 0:
                        print(f"[HZ-SKIP] logits={None if hz_logits is None else 'OK'} evi_soft={None if evi_soft is None else 'OK'}")
                        self._hz_dbg_count = 1
        except Exception as _hz_ex:
            print(f"[HZ-ERR][forward] {type(_hz_ex).__name__}: {_hz_ex}")
        # 仅保留 free-run 附加项（如果需要）
        try:
            if float(getattr(self, 'w_free', 0.0)) > 0.0 and int(getattr(self, 'free_k', 0)) > 0 and hasattr(self,
                                                                                                             'free_runner'):
                fk = int(getattr(self, 'free_k', 0))
                free_loss = self.free_runner(batch, fk)  # type: ignore
                if free_loss is not None:
                    loss = loss + float(getattr(self, 'w_free', 0.0)) * free_loss
                    if isinstance(stats, dict):
                        stats['free_loss'] = getattr(free_loss, 'detach', lambda: free_loss)()

        except Exception as _e:
            print(f"[FREE-ERR] free-run branch failed: {_e}")

        return loss, stats

    def _hazard_losses_soft_only(self, hz_logits_seq, evi_soft_seq, **kw):
        return hazard_losses_soft_only(hz_logits_seq, evi_soft_seq, **kw)


class _StepEventsHead(nn.Module):
    """Per-step event head: hidden H -> E logits"""

    def __init__(self, in_dim: int, event_dim: int=2, bias: bool=True):
        super().__init__()
        self.proj = nn.Linear(in_dim, event_dim, bias=bias)

    def forward(self, h):
        if hasattr(h, 'dim') and h.dim() == 3:
            B, T, H = h.shape
            return self.proj(h.reshape(B * T, H)).reshape(B, T, -1)
        return self.proj(h)

class _AuxEventsHead(nn.Module):

    def __init__(self, in_dim: int, out_events: int=2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim // 2 if in_dim >= 8 else max(4, in_dim)), nn.ReLU(), nn.Linear(in_dim // 2 if in_dim >= 8 else max(4, in_dim), out_events))

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        return self.net(h_seq)
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F

def hazard_losses_soft_only(
    hz_logits_seq: torch.Tensor,
    evi_soft_seq: torch.Tensor,
    *,
    w_bce: float = 1.0,
    w_smooth: float = 0.05,
    label_smoothing: float = 0.0,
    smooth_on_logits: bool = True,
    smooth_edge_guard: int = 1,
    evt_hard_thresh: float = 0.5,
    pos_weight=None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Soft-contact hazard loss with phase tolerance and confidence weighting.
    Returns: (loss, stats)
    - BCE uses hazard_bce_tolerant on the positive channel.
    - Smooth uses L2 difference on adjacent steps of pos-logit (or prob), with edge-guard around events.
    """
    assert evi_soft_seq is not None, "evi_soft_seq is required for soft-only hazard loss."

    # Coerce shapes
    if hz_logits_seq.dim() == 2:   # (B,T) -> interpret as pos-logit only
        hz_pos_logit = hz_logits_seq
    elif hz_logits_seq.dim() == 3 and hz_logits_seq.size(-1) >= 2:
        hz_pos_logit = hz_logits_seq[..., 1]
    else:
        raise ValueError(f"Unexpected hz_logits_seq shape: {hz_logits_seq.shape}")

    if evi_soft_seq.dim() == 3 and evi_soft_seq.size(-1) == 2:
        target_pos = evi_soft_seq[..., 1]
    elif evi_soft_seq.dim() == 3 and evi_soft_seq.size(-1) == 1:
        target_pos = evi_soft_seq.squeeze(-1)
    else:
        target_pos = evi_soft_seq  # assume (B,T)
    # --- ensure both on the same device ---
    dev = hz_pos_logit.device
    if target_pos.device != dev:
        target_pos = target_pos.to(dev)


    # Label smoothing (optional)
    if label_smoothing > 0:
        target_pos = target_pos * (1.0 - label_smoothing) + 0.5 * label_smoothing

    # BCE (phase-tolerant)
    # If original logits had 2 channels, rebuild a (B,T,2) logits tensor for tolerant BCE.
    if hz_logits_seq.dim() == 3 and hz_logits_seq.size(-1) >= 2:
        logits2 = hz_logits_seq
    else:
        logits2 = torch.stack([-hz_pos_logit, hz_pos_logit], dim=-1)
    hz_bce_val = hazard_bce_tolerant(logits2, target_pos, win=0, mask=None)
    hazard_bce = w_bce * hz_bce_val

    # Smooth term
    if w_smooth > 0:
        signal = hz_pos_logit if smooth_on_logits else torch.sigmoid(hz_pos_logit)
        diff = signal[:, 1:] - signal[:, :-1]   # (B,T-1)
        # Build edge guard mask around events (where target_pos > threshold)
        evt = (target_pos > evt_hard_thresh)
        if smooth_edge_guard > 0:
            # dilate event mask by r frames
            r = int(smooth_edge_guard)
            dil = evt.clone()
            for s in range(1, r+1):
                dil = dil | torch.roll(evt, shifts=s, dims=1) | torch.roll(evt, shifts=-s, dims=1)
            # for pair (t-1,t), guard if either endpoint is inside dil
            pair_guard = ~(dil[:, 1:] | dil[:, :-1])
        else:
            pair_guard = torch.ones_like(diff, dtype=torch.bool)
        denom = pair_guard.float().sum().clamp_min(1.0)
        smooth_raw = (diff * pair_guard.float()).pow(2).sum() / denom
        hazard_smooth = w_smooth * smooth_raw
    else:
        hazard_smooth = torch.tensor(0.0, device=hz_pos_logit.device, dtype=hz_pos_logit.dtype)

    # Stats
    stats: Dict[str, torch.Tensor] = {}
    stats['hz_bce'] = hazard_bce.detach()
    stats['hz_smooth'] = hazard_smooth.detach()
    stats['hz_mask_ratio'] = torch.tensor(1.0, device=hz_pos_logit.device)  # placeholder (no hard mask here)
    stats['target_mean'] = target_pos.mean().detach()
    stats['logits_abs_mean'] = (hz_pos_logit.abs().mean()).detach()
    stats['logits_abs_max'] = (hz_pos_logit.abs().amax()).detach()
    stats['all_zero'] = (hz_pos_logit.abs().amax() == 0).to(hz_pos_logit.dtype)

    # pos_weight stats passthrough (for logging compatibility)
    if pos_weight is not None:
        if isinstance(pos_weight, (tuple, list)):
            if len(pos_weight) == 2:
                stats['hz_pw0'] = torch.tensor(float(pos_weight[0]), device=hz_pos_logit.device)
                stats['hz_pw1'] = torch.tensor(float(pos_weight[1]), device=hz_pos_logit.device)
        elif torch.is_tensor(pos_weight):
            if pos_weight.numel() == 2:
                stats['hz_pw0'] = pos_weight.reshape(-1)[0].detach()
                stats['hz_pw1'] = pos_weight.reshape(-1)[1].detach()

    return hazard_bce + hazard_smooth, stats

def hazard_bce_tolerant(hz_logits: torch.Tensor,
                        target: torch.Tensor,
                        *,
                        win: int = 2,
                        mask: Optional[torch.Tensor] = None,
                        reduction: str = 'mean') -> torch.Tensor:
    """
    hz_logits: (B, T, 2)  -> use positive logit: logits[..., 1]
    target  : (B, T) in [0, 1], soft evidence
    mask    : (B, T) in {0,1} or float (1=valid)
    win     : +/- frames for small phase tolerance
    """
    p_logit = hz_logits[..., 1]                     # (B,T)
    # confidence: near 0/1 => high; near 0.5 => low
    conf = (target - 0.5).abs() * 2.0               # (B,T) in [0,1]
    if mask is not None:
        conf = conf * mask.float()

    # windowed min-BCE across small temporal shifts
    losses = []
    for s in range(-win, win + 1):
        tgt_s = target if s == 0 else torch.roll(target, shifts=s, dims=1)
        l = F.binary_cross_entropy_with_logits(p_logit, tgt_s, reduction='none')  # (B,T)
        losses.append(l)
    L = torch.stack(losses, dim=-1)                 # (B,T,2*win+1)
    Lmin, _ = torch.min(L, dim=-1)                  # (B,T)

    Lw = Lmin * conf
    if reduction == 'mean':
        denom = (conf > 0).float().sum().clamp_min(1.0)
        return Lw.sum() / denom
    elif reduction == 'sum':
        return Lw.sum()
    else:
        return Lw                                    # (B,T)


class DataNormalizer:
    """封装数据规格与(反)归一化逻辑。"""
    def __init__(self, *,
                 mu_x=None, std_x=None, mu_y=None, std_y=None,
                 s_eff_x=None, s_eff_y=None,
                 y_to_x_map=None,
                 yaw_x_slice=None, yaw_y_slice=None,
                 rootvel_x_slice=None, rootvel_y_slice=None,
                 angvel_x_slice=None, angvel_y_slice=None,
                 tanh_scales_rootvel=None, tanh_scales_angvel=None,
                 traj_dir_slice=None):
        import numpy as np
        self.mu_x = None if mu_x is None else np.asarray(mu_x, dtype=np.float32)
        self.std_x = None if std_x is None else np.asarray(std_x, dtype=np.float32)
        self.mu_y = None if mu_y is None else np.asarray(mu_y, dtype=np.float32)
        self.std_y = None if std_y is None else np.asarray(std_y, dtype=np.float32)
        self.s_eff_x = None if s_eff_x is None else np.asarray(s_eff_x, dtype=np.float32)
        self.s_eff_y = None if s_eff_y is None else np.asarray(s_eff_y, dtype=np.float32)
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

    @staticmethod
    def _atanh_safe_t(x, torch):
        x = torch.clamp(x, -0.999999, 0.999999)
        return torch.atanh(x) if hasattr(torch, "atanh") else 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def norm(self, x_raw_t):
        import torch
        x_raw = x_raw_t
        if (self.mu_x is not None) and (self.std_x is not None):
            mu = torch.as_tensor(self.mu_x,  device=x_raw.device, dtype=x_raw.dtype)
            sd = torch.as_tensor(self.std_x, device=x_raw.device, dtype=x_raw.dtype)
            z  = (x_raw - mu) / torch.clamp(sd, min=1e-3)
        else:
            z = x_raw
        if isinstance(self.rootvel_x_slice, slice):
            z[..., self.rootvel_x_slice] = self._atanh_safe_t(x_raw[..., self.rootvel_x_slice], torch)
        if isinstance(self.angvel_x_slice, slice):
            z[..., self.angvel_x_slice]  = self._atanh_safe_t(x_raw[..., self.angvel_x_slice], torch)
        if isinstance(self.yaw_x_slice, slice):
            z[..., self.yaw_x_slice] = x_raw[..., self.yaw_x_slice].abs()
        return z

    def denorm_x(self, xz_t, prev_raw=None):
            """把 X 的 Z 域还原为 RAW：做 μ/σ 逆 + 分组逆变换；Yaw 因 abs 丢符号，若给了 prev_raw 就延用上一帧符号。"""
            import torch
            xz = xz_t
            # 先做 μ/σ 的逆
            if (self.mu_x is not None) and (self.std_x is not None):
                mu = torch.as_tensor(self.mu_x,  device=xz.device, dtype=xz.dtype)
                sd = torch.as_tensor(self.std_x, device=xz.device, dtype=xz.dtype).clamp(min=1e-6)
                while mu.dim() < xz.dim():
                    mu = mu.unsqueeze(0); sd = sd.unsqueeze(0)
                x_raw = xz * sd + mu
            else:
                x_raw = xz.clone()

            # 分组逆变换（与 norm 对称）
            if isinstance(self.rootvel_x_slice, slice):
                x_raw = x_raw.clone()
                x_raw[..., self.rootvel_x_slice] = torch.tanh(xz[..., self.rootvel_x_slice])
            if isinstance(self.angvel_x_slice, slice):
                x_raw = x_raw.clone()
                x_raw[..., self.angvel_x_slice]  = torch.tanh(xz[..., self.angvel_x_slice])

            # yaw: 使用上一帧 RAW 的符号（若可用）
            if isinstance(self.yaw_x_slice, slice) and (prev_raw is not None):
                s = self.yaw_x_slice
                prev = torch.as_tensor(prev_raw, device=xz.device, dtype=xz.dtype)
                sign = torch.sign(prev[..., s]).clamp(min=-1., max=1.)
                x_raw = x_raw.clone()
                x_raw[..., s] = torch.abs(x_raw[..., s]) * sign

            return x_raw



    def denorm(self, y_t):
        import torch, math
        y = y_t
        if isinstance(self.yaw_y_slice, slice):
            x = torch.clamp(y[..., self.yaw_y_slice], -1.0, 1.0)
            yaw_raw = x * math.pi
            y = y.clone()
            y[..., self.yaw_y_slice] = yaw_raw
        if isinstance(self.rootvel_y_slice, slice) and self.tanh_scales_rootvel is not None:
            sc = torch.as_tensor(self.tanh_scales_rootvel, device=y.device, dtype=y.dtype)
            y = y.clone()
            y[..., self.rootvel_y_slice] = self._atanh_safe_t(y[..., self.rootvel_y_slice], torch) * sc
        if isinstance(self.angvel_y_slice, slice) and self.tanh_scales_angvel is not None:
            sc = torch.as_tensor(self.tanh_scales_angvel, device=y.device, dtype=y.dtype)
            y = y.clone()
            y[..., self.angvel_y_slice] = self._atanh_safe_t(y[..., self.angvel_y_slice], torch) * sc
        if self.std_y is not None and self.mu_y is not None:
            std = torch.as_tensor(self.std_y, device=y.device, dtype=y.dtype)
            mu  = torch.as_tensor(self.mu_y,  device=y.device, dtype=y.dtype)
            while std.dim() < y.dim():
                std = std.unsqueeze(0); mu = mu.unsqueeze(0)
            y = y * std + mu
        try:
            sl = self.traj_dir_slice
            if isinstance(sl, slice):
                a, b = sl.start, sl.stop - sl.start
                dim = 3 if b % 3 == 0 else 2 if b % 2 == 0 else 0
                if dim > 0:
                    blk = y[..., a:a+b].view(*y.shape[:-1], b // dim, dim)
                    y[..., a:a+b] = torch.nn.functional.normalize(blk, dim=-1).reshape_as(y[..., a:a+b])
        except Exception as _hz_e:
            print(f"[HZ-ERR] hazard branch failed: {_hz_e}")
            pass
        return y

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
            s_eff_x = get('s_eff_x'), s_eff_y = get('s_eff_y'),
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


class Trainer:
    def _rollout_sequence(self, state_seq, cond_seq=None, mode='mixed', tf_ratio=1.0):
        import torch
        assert state_seq.dim() == 3, "state_seq expects [B,T,Dx]"
        B, T, _ = state_seq.shape

        motion = state_seq[:, 0]                  # X in Z-domain at t=0
        hidden, kv_mem, last_attn = (None, None, None)
        outs, hz_seq = ([], [])

        # convenience: locate rot6d slice on X
        def _rot6d_slice_for_x():
            sx = getattr(self.train_loader, "rot6d_x_slice", None) if hasattr(self, "train_loader") else None
            if sx is None:
                sx = getattr(self, "rot6d_x_slice", None) or getattr(self, "rot6d_slice", None)
            return sx if isinstance(sx, slice) else slice(0, motion.size(-1))

        # keep a RAW copy for free/mixed rollouts to avoid domain-mixing
        motion_raw_local = None
        if hasattr(self, 'normalizer') and (self.normalizer is not None):
            try:
                motion_raw_local = self.normalizer.denorm_x(motion)
            except Exception:
                motion_raw_local = None

        for t in range(T):
            cond_t = cond_seq[:, t] if (cond_seq is not None and getattr(cond_seq, 'dim', lambda:0)() == 3) else cond_seq
            _devt = getattr(self.device, 'type', 'cpu')
            if _devt == 'mps':
                _amp_ctx = torch.autocast(device_type='mps', dtype=torch.float16, enabled=getattr(self, 'use_amp', False))
            elif _devt == 'cuda':
                _amp_ctx = torch.amp.autocast('cuda', enabled=getattr(self, 'use_amp', False))
            else:
                import contextlib as _ctx
                _amp_ctx = _ctx.nullcontext()
            with _amp_ctx:
                    ret = self.model(motion, cond_t, hidden, kv_mem)

            # unpack model return
            if isinstance(ret, dict):
                out    = ret.get('out', None)
                hidden = ret.get('hidden', hidden)
                last_attn = ret.get('attn', ret.get('last_attn', last_attn))
                kv_mem = ret.get('kv_mem', kv_mem)
                hz_t   = ret.get('hazard_logits', None)
            elif isinstance(ret, (tuple, list)):
                out    = ret[0] if len(ret) > 0 else None
                hidden = ret[1] if len(ret) > 1 else hidden
                last_attn = ret[2] if len(ret) > 2 else last_attn
                kv_mem = ret[3] if len(ret) > 3 else kv_mem
                hz_t   = None
            else:
                out, hz_t = ret, None

            if out is None:
                raise RuntimeError("Model forward must return 'out' tensor.")

            outs.append(out)
            if hz_t is not None:
                hz_seq.append(hz_t)

            if t < T - 1:
                if mode == 'teacher':
                    # Next input is GT (Z-domain); sync RAW for potential later use
                    motion = state_seq[:, t + 1]
                    if hasattr(self, 'normalizer') and (self.normalizer is not None):
                        try:
                            motion_raw_local = self.normalizer.denorm_x(motion, prev_raw=motion_raw_local)
                        except Exception:
                            motion_raw_local = None

                elif mode == 'free':
                    # Use model Y (RAW) to update X (RAW), then re-normalize to Z
                    y_raw = self._denorm(out)
                    if motion_raw_local is not None:
                        motion_raw_local = self._apply_free_carry(motion_raw_local, y_raw).detach()
                        motion = self._diag_norm_x(motion_raw_local)
                    else:
                        # fallback (no normalizer)
                        motion = self._apply_free_carry(motion, y_raw).detach()

                else:  # mixed (scheduled sampling on rot6d)
                    y_raw = self._denorm(out)
                    if motion_raw_local is not None:
                        free_raw  = self._apply_free_carry(motion_raw_local, y_raw).detach()
                        free_z    = self._diag_norm_x(free_raw)
                        gt_next   = state_seq[:, t + 1]
                        sx = _rot6d_slice_for_x()
                        sel = (torch.rand(B, device=self.device) < float(tf_ratio)).float().unsqueeze(-1)
                        motion_next = gt_next.clone()
                        motion_next[..., sx] = sel * gt_next[..., sx] + (1.0 - sel) * free_z[..., sx]
                        motion = motion_next
                        # resync RAW for next step
                        try:
                            motion_raw_local = self.normalizer.denorm_x(motion, prev_raw=motion_raw_local)
                        except Exception:
                            motion_raw_local = None
                    else:
                        # fallback (no normalizer)
                        gt_next   = state_seq[:, t + 1]
                        free_next = self._apply_free_carry(motion, y_raw).detach()
                        sx = _rot6d_slice_for_x()
                        motion_next = gt_next.clone()
                        sel = (torch.rand(B, device=self.device) < float(tf_ratio)).float().unsqueeze(-1)
                        motion_next[..., sx] = sel * gt_next[..., sx] + (1.0 - sel) * free_next[..., sx]
                        motion = motion_next

        y = torch.stack(outs, dim=1)
        hz = torch.stack(hz_seq, dim=1) if len(hz_seq) > 0 else None
        return {'out': y, 'hazard_logits': hz}, last_attn




    def __init__(self, model, loss_fn, lr=0.0001, use_dynamic_tf=True, grad_clip=0.0, weight_decay=0.01, tf_warmup_steps=0, tf_total_steps=0, mixup_alpha=0.0, augmentor=None, use_swa=False, swa_start_epoch=0, use_amp=False, accum_steps=1, scheduler=None):
        import torch
        self.model = model
        self.loss_fn = loss_fn
        # Make MuY/StdY available on Trainer for _denorm()
        self.mu_y = getattr(loss_fn, 'mu_y', None)
        self.std_y = getattr(loss_fn, 'std_y', None)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[LR-DBG:init] arg_lr={lr:.2e} opt_pg0={self.optimizer.param_groups[0]['lr']:.2e}")

        self.grad_clip = float(grad_clip)
        self.use_dynamic_tf = bool(use_dynamic_tf)
        self.tf_warmup_steps = int(tf_warmup_steps)
        self.tf_total_steps = int(tf_total_steps)
        self.mixup_alpha = float(mixup_alpha)
        self.augmentor = augmentor
        self.use_swa = bool(use_swa)
        self.swa_model = None
        self.swa_start_epoch = int(swa_start_epoch)
        self.use_amp = bool(use_amp)
        self.accum_steps = int(accum_steps)
        self.scheduler = scheduler
        self.device = next(model.parameters()).device
        if getattr(self.device, 'type', None) == 'mps' and getattr(self, 'use_amp', False):
            print('[AMP] MPS backend detected; using torch.autocast(mps, fp16).')
        self._x_layout = {}
        self._y_layout = {}
        self.fps = 60.0
        self.y_to_x_map = []
        self.MuY = None
        self.StdY = None

    def _diag_norm_x(self, x_raw, mu_x=None, std_x=None):
        # 仅使用 DataNormalizer，未注入时返回原值
        if hasattr(self, 'normalizer') and (self.normalizer is not None):
            return self.normalizer.norm(x_raw)
        return x_raw

    def _pick_first(self, batch, keys):
        if batch is None:
            return None
        if isinstance(batch, dict):
            for k in keys:
                if k in batch and batch[k] is not None:
                    return batch[k]
        return None

    @torch.no_grad()
    def eval_epoch(self, loader, mode='teacher'):
        self.model.eval()
        total_loss, count = 0.0, 0
        for batch in loader:
            x_cand = self._pick_first(batch, ('motion', 'X', 'x_in_features'))
            y_cand = self._pick_first(batch, ('gt_motion', 'Y', 'y_out_features', 'y_out_seq'))
            if x_cand is None or y_cand is None:
                continue

            state_seq = x_cand.to(self.device).float()
            gt_seq = y_cand.to(self.device).float()

            cond_seq = batch.get('cond_in', None) if isinstance(batch, dict) else None
            if cond_seq is not None:
                cond_seq = cond_seq.to(self.device).float()

            # ✅ 仅训练侧增强；无论 cond_seq 是否存在都调用（内部会处理 None）
            if mode == 'train':
                state_seq, gt_seq, cond_seq = self._train_augment_if_needed(state_seq, gt_seq, cond_seq)

            preds_dict, last_attn = self._rollout_sequence(state_seq, cond_seq, mode=mode, tf_ratio=1.0)
            out = self.loss_fn(preds_dict, gt_seq, attn_weights=last_attn, batch=batch)
            loss = out[0] if isinstance(out, tuple) else out
            total_loss += float(loss.detach().cpu())
            count += 1

        return total_loss / max(1, count)

    def fit(self, train_loader, epochs=10, log_every=50, out_dir=None, patience=10, run_name='run'):
        import torch, os
        self.model.train()
        self.train_loader = train_loader
        _devs = getattr(self.device, 'type', 'cpu')
        scaler = torch.amp.GradScaler('cuda' if _devs=='cuda' else 'cpu', enabled=(getattr(self, 'use_amp', False) and _devs=='cuda'))
        accum_steps = int(getattr(self, 'accum_steps', 1) or 1)
        best_val, best_ckpt = float('inf'), None
        best_valfree = float('inf')
        history = {'train':[], 'val':[]}
        tf_mode = getattr(self, 'tf_mode', 'epoch_linear')
        tf_start = int(getattr(self, 'tf_start_epoch', 0))
        tf_end   = int(getattr(self, 'tf_end_epoch', 0))
        tf_max   = float(getattr(self, 'tf_max', 1.0))
        tf_min   = float(getattr(self, 'tf_min', 0.0))

        for ep in range(1, int(epochs)+1):

            # record epoch for schedulers
            try:
                self.cur_epoch = int(ep)
                self.total_epochs = int(epochs)
            except Exception:
                pass
            epoch_sums = {}
            epoch_cnt = 0
            print(f"[LR-DBG:fit-epoch{ep:03d}-start] pg0={self.optimizer.param_groups[0]['lr']:.2e}")

            if tf_mode == 'epoch_linear' and tf_end > tf_start:
                if ep <= tf_start: tf_ratio = tf_max
                elif ep >= tf_end: tf_ratio = tf_min
                else:
                    r = (ep - tf_start) / max(1, (tf_end - tf_start))
                    tf_ratio = tf_max + (tf_min - tf_max) * r
            else:
                tf_ratio = tf_max
            self._last_tf_ratio = float(tf_ratio)
            # reset hz debug counter each epoch
            try:
                self.loss_fn._hz_dbg_count = 0
            except Exception:
                pass
            running, cnt = 0.0, 0
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)


            for bi, batch in enumerate(train_loader, start=1):
                x_cand = self._pick_first(batch, ('motion','X','x_in_features'))
                y_cand = self._pick_first(batch, ('gt_motion','Y','y_out_features','y_out_seq'))
                if x_cand is None or y_cand is None:
                    continue
                # 位置：Trainer.train(...) 里
                state_seq = x_cand.to(self.device).float()
                gt_seq = y_cand.to(self.device).float()
                cond_seq = batch.get('cond_in', None) if isinstance(batch, dict) else None
                if cond_seq is not None:
                    cond_seq = cond_seq.to(self.device).float()

                # === 插入开始：一次性打印训练端 X(z) 的 RMS，验证不是 0 ===
                if bi == 1 and ep == 1:  # 只在第1个 epoch 的第1个 batch 打一次，避免刷屏
                    import torch
                    Xz = state_seq  # [B,T,Dx]：此时就是送入模型前的 X（已归一化 z）

                    def _rms_or_na(sl):
                        if isinstance(sl, slice):
                            x = Xz[..., sl]
                            return float(torch.sqrt(torch.mean(x.float() ** 2)).item())
                        return float('nan')

                    yaw_sl = getattr(self, 'yaw_x_slice', None)
                    rootvel_sl = getattr(self, 'rootvel_x_slice', None)
                    angvel_sl = getattr(self, 'angvel_x_slice', None)

                    rms_yaw = _rms_or_na(yaw_sl)
                    rms_rootvel = _rms_or_na(rootvel_sl)
                    rms_angvel = _rms_or_na(angvel_sl)

                    # 与你推理端风格一致的三行打印
                    print(f"[X:RootYaw] rms|z|={(0.0 if torch.isnan(torch.tensor(rms_yaw)) else rms_yaw):.3f}")
                    print(
                        f"[X:RootVelocity] rms|z|={(0.0 if torch.isnan(torch.tensor(rms_rootvel)) else rms_rootvel):.3f}")
                    print(
                        f"[X:AngVel(ALL)] rms|z|={(0.0 if torch.isnan(torch.tensor(rms_angvel)) else rms_angvel):.3f}")
                # === 插入结束 ===

                preds_dict, last_attn = self._rollout_sequence(state_seq, cond_seq, mode='mixed', tf_ratio=tf_ratio)

                _devt2 = getattr(self.device, 'type', 'cpu')
                if _devt2 == 'mps':
                    _amp2 = torch.autocast(device_type='mps', dtype=torch.float16, enabled=getattr(self, 'use_amp', False))
                elif _devt2 == 'cuda':
                    _amp2 = torch.amp.autocast('cuda', enabled=getattr(self, 'use_amp', False))
                else:
                    import contextlib as _ctx
                    _amp2 = _ctx.nullcontext()
                with _amp2:
                    out = self.loss_fn(preds_dict, gt_seq, attn_weights=last_attn, batch=batch)
                    loss = out[0] if isinstance(out, tuple) else out
                    # --- Boundary Consistency Loss (Multi-Stride) ---
                    if float(getattr(self, 'w_boundary', 0.0) or 0.0) > 0.0:
                        try:
                            y_pred = preds_dict.get('out', None) if isinstance(preds_dict, dict) else None
                            bc_loss, bc_stats = self._boundary_consistency_from_batch(batch, y_pred)
                            loss = loss + float(getattr(self, 'w_boundary', 0.0)) * bc_loss
                        except Exception as _bc_e:
                            if int(getattr(self, 'hz_dbg', 0)) > 0:
                                print('[BC] error:', _bc_e)
                            bc_stats = {'boundary': 0.0, 'bc_pairs': 0.0, 'bc_stride_now': 0.0}
                    else:
                        bc_stats = None

                    # --- accumulate per-batch loss parts for epoch summary ---
                    try:
                        
                        # merge BC stats
                        if bc_stats and isinstance(bc_stats, dict):
                            try:
                                for _k, _v in bc_stats.items():
                                    val = float(_v)
                                    epoch_sums[_k] = epoch_sums.get(_k, 0.0) + val
                            except Exception:
                                pass
                        stats = out[1] if (
                                    isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], dict)) else None
                        if isinstance(stats, dict):
                            for _k, _v in stats.items():
                                try:
                                    val = float(_v.detach().cpu()) if hasattr(_v, 'detach') else float(_v)
                                    epoch_sums[_k] = epoch_sums.get(_k, 0.0) + val
                                except Exception:
                                    pass
                        epoch_cnt += 1
                    except Exception:
                        pass

                scaler.scale(loss / accum_steps).backward()
                if bi % accum_steps == 0:
                    scaler.unscale_(self.optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                        max_norm=float(getattr(self, 'grad_clip', 1.0)))

                    if (bi % int(log_every or 50) == 0):
                        lr0 = float(self.optimizer.param_groups[0].get('lr', 0.0))
                        if log_every:
                            print(f"[Grad] ep={ep:03d} bi={bi:04d} gn={float(gn):.3e} lr={lr0:.2e}")
                    scaler.step(self.optimizer)
                    if log_every:
                        print(f"[LR-DBG:after-opt-step] pg0={self.optimizer.param_groups[0]['lr']:.2e}")
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)



                    _sched = getattr(self, 'lr_scheduler', None)
                    if _sched is not None:
                        try:
                            _sched.step()
                            # print(f"[LR-DBG:after-sched-step] pg0={self.optimizer.param_groups[0]['lr']:.2e}")
                        except Exception:
                            pass

                running += float(loss.detach().cpu()); cnt += 1
                if log_every and (bi % int(log_every) == 0):
                    print("[Train][ep %03d][%04d/%d] loss=%.4f tf=%.3f" % (ep, bi, len(train_loader), running/max(1,cnt), float(tf_ratio)))
            avg_train = running / max(1, cnt)
            history['train'].append(avg_train)
            if getattr(self, "print_hz_to_console", False) and hasattr(self.loss_fn, "_last_hz_stats"):
                _hz = getattr(self.loss_fn, "_last_hz_stats", {}) or {}
                _hz_str = f" | hz_bce={_hz.get('hz_bce',0):.3f} hz_smooth={_hz.get('hz_smooth',0):.3f} hz_mask={_hz.get('hz_mask_ratio',0):.2f}"
            else:
                _hz_str = ""
            _bc_str = ""
            if epoch_cnt > 0 and isinstance(epoch_sums, dict) and "boundary" in epoch_sums:
                _bc = float(epoch_sums.get("boundary", 0.0)) / max(1, epoch_cnt)
                _bp = float(epoch_sums.get("bc_pairs", 0.0)) / max(1, epoch_cnt)
                _bs = float(epoch_sums.get("bc_stride_now", 0.0)) / max(1, epoch_cnt)
                _bc_str = f" | boundary={_bc:.3f} bc_pairs={_bp:.1f} bc_stride={_bs:.0f}"

            print("[Train][ep %03d] loss=%.4f%s%s" % (ep, avg_train, _hz_str, _bc_str))

            if epoch_cnt > 0:
                keys = ("motion", "rot_geo", "rot_ortho", "rot_ortho_raw", "speed_curve", "energy", "hz_bce", "hz_smooth", "hz_count", "free_loss")
                parts = [f"{k}={epoch_sums[k] / max(1, epoch_cnt):.4f}" for k in keys if k in epoch_sums]
                if parts:
                    print("[LossParts]", " | ".join(parts))

            # --- online (free-run) metrics each epoch ---
            _metrics = None
            try:
                if getattr(self, 'val_mode', 'none') == 'online' and not bool(getattr(self, 'no_monitor', False)):
                    vloader = self.train_loader
                    _mon_batches = int(getattr(self, 'events_monitor_batches', 8) or 8)
                    _metrics = self.validate_autoreg_online(vloader, max_batches=_mon_batches)
                    print(
                        f"[ValFree@ep {ep:03d}] "
                        f"MSEnormY={_metrics['MSEnormY']:.6f} | "
                        f"GeoDeg={_metrics['GeoDeg']:.3f}° | "
                        f"YawAbsDeg={_metrics['YawAbsDeg']:.3f} | "
                        f"RootVelMAE={_metrics['RootVelMAE']:.5f} | "
                        f"AngVelMAE={_metrics.get('AngVelMAE', float('nan')):.5f} rad/s | "
                        # f"AngDirDeg={_metrics.get('AngVelDirDeg', float('nan')):.3f}° | "
                        f"AngMagRel={_metrics.get('AngVelMagRel', float('nan')):.3f} | "
                        f"AngVecSRE={_metrics.get('AngVelSRE', float('nan')):.3f} | "
                        f"StaticExcess={_metrics.get('StaticExcess', float('nan')):.5f} rad/s"
                    )



            except Exception as _e:
                print(f"[ValFree@ep {ep:03d}] skipped due to error: {_e}")

            # ---- OT proxy gap print (teacher vs. online) ----
            try:
                _vloader = self.train_loader
                _mon_batches = int(getattr(self, 'events_monitor_batches', 8) or 8)

                # teacher：已有函数，返回标量 loss
                teach_loss = self.eval_epoch(_vloader, mode='teacher')

                # online：优先复用上面算过的 _metrics，避免重复跑
                online = _metrics if (_metrics is not None) else self.validate_autoreg_online(_vloader, max_batches=_mon_batches)

                print(
                    f"[Gap@ep {ep:03d}] "
                    f"teach_loss={teach_loss:.6f} | "
                    f"GeoDeg={online['GeoDeg']:.3f}° | "
                    f"AngVelMAE={online.get('AngVelMAE', float('nan')):.5f} | "
                    f"MSEnormY={online['MSEnormY']:.6f}"
                )
            except Exception as _e:
                print(f"[Gap@ep {ep:03d}] skipped: {_e}")

            # --- choose best by metric_for_best ---
            choose = 'val_free'
            if choose == 'val_free' and (_metrics is not None):
                current = float(_metrics.get('MSEnormY', float('inf')))
                if current < best_valfree - 1e-9:
                    best_valfree = current
                    if out_dir:
                        os.makedirs(out_dir, exist_ok=True)
                        best_ckpt = os.path.join(out_dir, 'ckpt_best_' + str(run_name) + '.pth')
                        torch.save({'model': self.model.state_dict()}, best_ckpt)
        if out_dir:

            import os, torch

            os.makedirs(out_dir, exist_ok=True)

            last_ckpt = os.path.join(out_dir, 'ckpt_last_' + str(run_name) + '.pth')

            torch.save({'model': self.model.state_dict()}, last_ckpt)

        return best_ckpt, history


    def _current_bc_strides(self):
        """Return a list of strides to use this epoch for Multi-Stride BC."""
        try:
            s0 = max(1, int(getattr(self, 'bc_stride', 1) or 1))
            smax_cfg = max(s0, int(getattr(self, 'bc_stride_max', s0) or s0))
            ep = int(getattr(self, 'cur_epoch', 0) or 0)
            tot = int(getattr(self, 'total_epochs', 0) or 0)
            rs = int(getattr(self, 'bc_ramp_start', 1) or 1)
            re = int(getattr(self, 'bc_ramp_end', 0) or 0)
            if smax_cfg <= s0:
                return [s0]
            # schedule
            if re and re > rs and tot > 0:
                # piecewise linear to bc_stride_max between [rs, re]
                if ep < rs:
                    smax = s0
                elif ep >= re:
                    smax = smax_cfg
                else:
                    frac = (ep - rs) / float(max(1, re - rs))
                    smax = int(round(s0 + frac * (smax_cfg - s0)))
            else:
                # no ramp: always smax_cfg
                smax = smax_cfg
            smax = max(s0, min(smax, smax_cfg))
            return list(range(s0, smax + 1))
        except Exception:
            return [1]

    def _boundary_consistency_from_batch(self, batch, y_pred):
        """
        Compute Boundary Consistency (BC) loss on overlapping regions of adjacent windows
        from the *same* clip. Works in normalized Z space.
        - batch 需要包含: 'clip_id': LongTensor[B], 'start': LongTensor[B]
        - y_pred: [B, T, Dy] (Z 域)
        返回: (bc_loss_scalar, stats_dict)
        stats_dict: {'boundary': float, 'bc_pairs': float, 'bc_stride_now': float}
        """
        import torch

        w = float(getattr(self, 'w_boundary', 0.0) or 0.0)
        if w <= 0.0 or y_pred is None:
            return y_pred.new_tensor(0.0), {'boundary': 0.0, 'bc_pairs': 0.0, 'bc_stride_now': 0.0}

        try:
            # ---- 取 batch 中的索引信息 ----
            if not isinstance(batch, dict):
                return y_pred.new_tensor(0.0), {'boundary': 0.0, 'bc_pairs': 0.0, 'bc_stride_now': 0.0}
            cid = batch.get('clip_id', None)
            st = batch.get('start', None)
            if cid is None or st is None:
                return y_pred.new_tensor(0.0), {'boundary': 0.0, 'bc_pairs': 0.0, 'bc_stride_now': 0.0}

            # tensor -> python ints
            if torch.is_tensor(cid): cid = cid.detach().cpu().tolist()
            if torch.is_tensor(st):  st = st.detach().cpu().tolist()
            cid = [int(x) for x in (cid if isinstance(cid, (list, tuple)) else [cid])]
            st = [int(x) for x in (st if isinstance(st, (list, tuple)) else [st])]

            # (clip_id, start) -> batch_idx
            index_map = {(c, s): i for i, (c, s) in enumerate(zip(cid, st))}
            strides = self._current_bc_strides()  # e.g., [1] 或 [1,2,3,4]（多尺度调度）
            if not strides:
                return y_pred.new_tensor(0.0), {'boundary': 0.0, 'bc_pairs': 0.0, 'bc_stride_now': 0.0}

            B, T, Dy = y_pred.shape

            # ---- 选择 rot6d 通道做 BC，并做 s_eff 夹紧，避免早期爆炸 ----
            # 1) 维度尺度（复用你已有的工具函数）
            try:
                s_eff = self.loss_fn._pick_s_eff(D=Dy, device=y_pred.device, dtype=y_pred.dtype)  # [1, Dy]
            except Exception:
                s_eff = torch.ones((1, Dy), dtype=y_pred.dtype, device=y_pred.device)

            # 2) 数值稳定：给 s_eff 下限，避免极小值把差分放大
            s_eff = s_eff.clamp(min=0.03)

            # 3) 只在 rot6d 上做 BC：优先从切片信息里拿范围，取不到时回退到 276（你的日志里 Y.rot6d=(0,276)）
            rot_end = Dy
            try:
                rot_slice = None
                # 优先看 loss_fn 上是否暴露 Y 侧切片
                if hasattr(self.loss_fn, 'slices_y') and isinstance(self.loss_fn.slices_y, dict):
                    rot_slice = self.loss_fn.slices_y.get('rot6d', None)
                # Trainer 里如果也存了切片
                if rot_slice is None and hasattr(self, 'slices_y') and isinstance(self.slices_y, dict):
                    rot_slice = self.slices_y.get('rot6d', None)
                if isinstance(rot_slice, slice):
                    rot_end = int(rot_slice.stop)
                elif isinstance(rot_slice, (tuple, list)) and len(rot_slice) == 2:
                    rot_end = int(rot_slice[1])
            except Exception:
                pass
            if rot_end <= 0 or rot_end > Dy:
                # 回退：常见 rot6d 维度
                rot_end = min(Dy, 276)

            s_eff_rot = s_eff[:, :rot_end]  # [1, rot_end]

            # ---- 逐 stride 配对相邻窗口并累计损失 ----
            total_losses = []
            total_pairs = 0
            used_max_stride = 0

            for stride in strides:
                stride = int(stride)
                O = T - stride  # overlap length
                if O <= 0:
                    continue

                pairs = []
                for c, s in zip(cid, st):
                    j = index_map.get((c, s + stride), None)
                    if j is not None:
                        i = index_map[(c, s)]
                        pairs.append((i, j))
                if not pairs:
                    continue

                used_max_stride = max(used_max_stride, stride)

                # 堆叠成张量计算，避免 Python 循环
                A = torch.stack([y_pred[i, stride:stride + O, :rot_end] for i, _ in pairs], dim=0)  # [P, O, rot_end]
                B2 = torch.stack([y_pred[j, :O, :rot_end] for _, j in pairs], dim=0)  # [P, O, rot_end]

                diff = (A - B2) / s_eff_rot  # 广播: [1, rot_end]
                total_losses.append((diff * diff).mean())
                total_pairs += len(pairs)

            if not total_losses:
                return y_pred.new_tensor(0.0), {'boundary': 0.0, 'bc_pairs': 0.0,
                                                'bc_stride_now': float(used_max_stride)}

            bc = torch.stack(total_losses, dim=0).mean()  # 多个 stride 的均值
            return bc, {
                'boundary': float(bc.detach().cpu()),
                'bc_pairs': float(total_pairs),
                'bc_stride_now': float(used_max_stride)
            }

        except Exception as e:
            if int(getattr(self, 'hz_dbg', 0)) > 0:
                print('[BC] skip due to:', e)
            return y_pred.new_tensor(0.0), {'boundary': 0.0, 'bc_pairs': 0.0, 'bc_stride_now': 0.0}

    # ===== from layout build slice (strict keys) =====
    def _sl_from_layout(self, layout, key):
        if not isinstance(layout, dict) or key not in layout:
            return None
        st, ln = int(layout[key][0]), int(layout[key][1])
        return slice(st, st+ln) if ln > 0 else None

    # ===== autoregressive online validation (UE-shaped) =====
    @torch.no_grad()
    def validate_autoreg_online(self, loader, max_batches=8):
        """
        自回归在线评估（与训练 teacher 链路同口径），去除调试探针与打印。
        依赖:
          - self.normalizer.denorm(...), self.normalizer.denorm_x(...)
          - self._denorm(y_z), self._apply_free_carry(x_raw, y_raw), self._diag_norm_x(x_raw)
          - self._sl_from_layout(layout, key)
          - rot6d_to_matrix, reproject_rot6d, geodesic_R, angvel_vec_from_R_seq
        可选:
          - self.eval_align_root0: bool = True   # 根骨 t=0 常量左乘配准（仅用于指标，默认 True）
          - self.eval_root_idx:    int  = 0      # 根骨索引
          - self.bone_hz:          float = 60.0
        """
        import math
        import torch

        self.model.eval()

        # --- 切片定位 ---
        rot6d_y = self._sl_from_layout(getattr(self, "_y_layout", {}), 'BoneRotations6D')
        yaw_x = self._sl_from_layout(getattr(self, "_x_layout", {}), 'RootYaw')
        rv_x = self._sl_from_layout(getattr(self, "_x_layout", {}), 'RootVelocity')

        # --- 评估设定 ---
        eval_align_root = bool(getattr(self, 'eval_align_root0', True))
        root_idx = int(getattr(self, 'eval_root_idx', 0))
        fps_eval = float(getattr(self, 'bone_hz', 60.0))

        # --- 指标累积 ---
        mse_normY = []
        geo_deg = []
        yaw_deg_aligned = []
        rv_mae = []
        angvel_mae_list, ang_dir_deg_list = [], []
        ang_mag_rel_list, ang_vec_sre_list = [], []
        static_excess_list = []

        def _first_not_none(batch, keys):
            for k in keys:
                if k in batch:
                    v = batch.get(k, None)
                    if v is not None:
                        return v
            return None

        it = iter(loader)
        with torch.no_grad():
            for _ in range(max_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break

                # 兼容 DataLoader 返回
                if isinstance(batch, dict):
                    motion_seq = _first_not_none(batch, ['motion', 'X', 'x_in_features'])
                    cond_seq = _first_not_none(batch, ['cond', 'C', 'conditions'])
                    y_seq = _first_not_none(batch, ['gt_motion', 'Y', 'y_out_features', 'y_out_seq'])
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    motion_seq, y_seq = batch[0], batch[1]
                    cond_seq = batch[2] if len(batch) > 2 else None
                else:
                    continue

                if motion_seq is None or y_seq is None:
                    continue

                device = next(self.model.parameters()).device
                motion_seq = motion_seq.to(device)
                y_seq = y_seq.to(device)
                cond_seq = cond_seq.to(device) if (cond_seq is not None and hasattr(cond_seq, "to")) else None

                B, T, Dx = motion_seq.shape
                if T < 2:
                    continue

                # ===== 自回归 free-run（保持 RAW↔Z 回路一致）=====
                motion = motion_seq[:, 0]  # X(z) @ t=0
                try:
                    motion_raw = self.normalizer.denorm_x(motion) if hasattr(self, 'normalizer') and (
                                self.normalizer is not None) else None
                except Exception:
                    motion_raw = None

                hidden, kv_mem = None, None
                predsY, predsX = [], []

                for t in range(T - 1):
                    cond_t = cond_seq[:, t] if (
                                cond_seq is not None and getattr(cond_seq, 'dim', lambda: 0)() == 3) else cond_seq
                    ret = self.model(motion, cond_t, hidden, kv_mem)
                    if isinstance(ret, dict):
                        out = ret.get('out', None)
                        hidden = ret.get('hidden', hidden)
                        kv_mem = ret.get('kv_mem', kv_mem)
                    else:
                        out = ret[0] if len(ret) > 0 else None
                        hidden = ret[1] if len(ret) > 1 else hidden
                        kv_mem = ret[3] if len(ret) > 3 else kv_mem
                    if out is None:
                        break

                    predsY.append(out)
                    y_raw = self._denorm(out)  # Y(z) -> Y(raw)

                    if motion_raw is None and hasattr(self, 'normalizer') and (self.normalizer is not None):
                        try:
                            motion_raw = self.normalizer.denorm_x(motion)
                        except Exception:
                            motion_raw = None

                    if motion_raw is not None:
                        motion_raw = self._apply_free_carry(motion_raw, y_raw).detach()
                        motion = self._diag_norm_x(motion_raw)  # 回到 Z 域
                    else:
                        # 兜底：仍允许在 Z 域做写回（不建议，正常有 normalizer）
                        motion = self._apply_free_carry(motion, y_raw).detach()

                    predsX.append(motion)

                if not predsY:
                    continue

                # === 归一化空间的 MSE（Y）===
                predY = torch.stack(predsY, dim=1)  # [B, T-1, Dy]
                gtY = y_seq[:, :predY.shape[1]]  # [B, T-1, Dy]
                mse_normY.append(torch.mean((predY - gtY) ** 2).item())

                # === GeoDeg（与训练同口径，直接用 rot6d_to_matrix）===
                if isinstance(rot6d_y, slice) and (rot6d_y.stop - rot6d_y.start) > 0:
                    Yraw_pred = self.normalizer.denorm(predY) if hasattr(self, 'normalizer') and (
                                self.normalizer is not None) else predY
                    Yraw_gt = self.normalizer.denorm(gtY) if hasattr(self, 'normalizer') and (
                                self.normalizer is not None) else gtY

                    py = Yraw_pred[..., rot6d_y]  # [B, T-1, 6J]
                    gy = Yraw_gt[..., rot6d_y]

                    J = py.shape[-1] // 6
                    if J > 0:
                        py6 = py.reshape(py.shape[:-1] + (J, 6))
                        gy6 = gy.reshape(gy.shape[:-1] + (J, 6))

                        # 6D 重投影（与训练端一致）
                        py6 = reproject_rot6d(py6.reshape(py6.shape[:-2] + (6 * J,))).reshape(py6.shape)
                        gy6 = reproject_rot6d(gy6.reshape(gy6.shape[:-2] + (6 * J,))).reshape(gy6.shape)

                        Rp = rot6d_to_matrix(py6)  # 项目里默认左手 + ('X','Z')
                        Rg = rot6d_to_matrix(gy6)

                        # 可选：根骨 t0 常量对齐（仅用于指标）
                        if eval_align_root and Rp.shape[1] > 0 and (0 <= root_idx < J):
                            Rpr0 = Rp[:, 0, root_idx]  # [B,3,3]
                            Rgr0 = Rg[:, 0, root_idx]
                            R_align = Rgr0 @ Rpr0.transpose(-1, -2)  # [B,3,3]
                            Rp = (R_align.view(B, 1, 1, 3, 3).expand_as(Rp)) @ Rp

                        rad = geodesic_R(Rp, Rg).mean().item()
                        geo_deg.append(rad * (180.0 / math.pi))

                        # === 角速度等指标（与训练口径一致）===
                        try:
                            w_pred = angvel_vec_from_R_seq(Rp, fps_eval)  # [B, T-1, J, 3]
                            w_gt = angvel_vec_from_R_seq(Rg, fps_eval)

                            mag_p = w_pred.norm(dim=-1)
                            mag_g = w_gt.norm(dim=-1)
                            mag_avg = 0.5 * (mag_p + mag_g)

                            tauA = 0.10  # rad/s（活动）
                            tauS = 0.10  # rad/s（静止）
                            maskA = (mag_avg > tauA)
                            maskS = (mag_avg <= tauS)

                            # 方向误差（度）
                            eps = 1e-8
                            dot = (w_pred * w_gt).sum(dim=-1)
                            cos = dot / (mag_p.clamp_min(eps) * mag_g.clamp_min(eps))
                            cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                            dir_deg = torch.arccos(cos) * (180.0 / math.pi)
                            dir_mean_j = (dir_deg * maskA).sum(dim=(0, 1)) / maskA.sum(dim=(0, 1)).clamp_min(1)
                            ang_dir_deg_list.append(float(torch.nanmedian(dir_mean_j).item()))

                            # 幅值相对误差（对称）
                            beta = 0.25
                            mag_rel = (mag_p - mag_g).abs() / (mag_avg + beta)
                            mag_rel_mean_j = (mag_rel * maskA).sum(dim=(0, 1)) / maskA.sum(dim=(0, 1)).clamp_min(1)
                            ang_mag_rel_list.append(float(torch.nanmedian(mag_rel_mean_j).item()))

                            # 向量 SRE（对称）
                            vec_sre = (w_pred - w_gt).norm(dim=-1) / (mag_avg + beta)
                            vec_sre_mean_j = (vec_sre * maskA).sum(dim=(0, 1)) / maskA.sum(dim=(0, 1)).clamp_min(1)
                            ang_vec_sre_list.append(float(torch.nanmedian(vec_sre_mean_j).item()))

                            # 静止“过多运动”
                            excess = (mag_p - mag_g).clamp_min(0.0)
                            excess_mean_j = (excess * maskS).sum(dim=(0, 1)) / maskS.sum(dim=(0, 1)).clamp_min(1)
                            static_excess_list.append(float(torch.nanmedian(excess_mean_j).item()))

                            # 全局 MAE（便于对比）
                            angvel_mae_list.append((w_pred - w_gt).abs().mean().item())
                        except Exception:
                            pass

                # === X 分支：yaw / root velocity ===
                if predsX:
                    predX = torch.stack(predsX, dim=1)  # [B, T-1, Dx]
                    gtX = motion_seq[:, 1:predX.shape[1] + 1]

                    if isinstance(yaw_x, slice):
                        # 圆差 + 常数偏置对齐（以 aligned 作为最终指标）
                        y_pred = predX[..., yaw_x]
                        y_gt = gtX[..., yaw_x]
                        dyaw = torch.atan2(torch.sin(y_pred - y_gt), torch.cos(y_pred - y_gt))
                        off = torch.atan2(dyaw.sin().mean(dim=1, keepdim=True), dyaw.cos().mean(dim=1, keepdim=True))
                        dyaw_aligned = torch.atan2(torch.sin(dyaw - off), torch.cos(dyaw - off))
                        yaw_deg_aligned.append((dyaw_aligned.abs().mean() * (180.0 / math.pi)).item())

                    if isinstance(rv_x, slice):
                        rv_err = (predX[..., rv_x] - gtX[..., rv_x]).abs().mean()
                        rv_mae.append(rv_err.item())

        def _mean_or_nan(lst):
            return float(sum(lst) / len(lst)) if lst else float('nan')

        return dict(
            AngVelMAE=_mean_or_nan(angvel_mae_list),  # rad/s
            AngVelDirDeg=_mean_or_nan(ang_dir_deg_list),  # °
            AngVelMagRel=_mean_or_nan(ang_mag_rel_list),
            AngVelSRE=_mean_or_nan(ang_vec_sre_list),
            StaticExcess=_mean_or_nan(static_excess_list),  # rad/s
            MSEnormY=_mean_or_nan(mse_normY),
            GeoDeg=_mean_or_nan(geo_deg),  # °
            YawAbsDeg=_mean_or_nan(yaw_deg_aligned),
            RootVelMAE=_mean_or_nan(rv_mae),
        )

    def _denorm(self, y):
        # 仅使用 DataNormalizer；未注入则按原值返回
        if hasattr(self, 'normalizer') and (self.normalizer is not None):
            try:
                return self.normalizer.denorm(y)
            except Exception:
                pass
        return y

    def _apply_free_carry(self, x_prev, y_denorm):
        """将模型预测的 *Y(raw)* 写回到下一步的 *X(raw)*，其余维度沿用上一帧 (carry-over)。
        改进点：
          1) 严格使用现有切片 (self.rot6d_x_slice / self.rot6d_y_slice) 与 y_to_x_map（若提供）；
          2) 写回 Rot6D 后，**同步重建 BoneAngularVelocities**（若提供 self.angvel_x_slice），避免“旧速率被沿用”；
          3) 全面形状/越界检查 + 轻量诊断打印（受 self.dbg_writeback 控制）。
        参数:
          x_prev: [B, Dx]  上一时刻的输入状态 (raw)
          y_denorm: [B, Dy]  当前时刻反归一化输出 (raw)
        返回:
          x_next: [B, Dx]  下一时刻输入 (raw)
        """
        import torch
        B, Dx = x_prev.shape[0], x_prev.shape[-1]
        Dy = y_denorm.shape[-1]
        x_next = x_prev.clone()

        dbg = bool(getattr(self, 'dbg_writeback', False))

        # 1) 先写回 Rot6D
        try:
            maps = getattr(self, 'y_to_x_map', None)
            if isinstance(maps, (list, tuple)) and len(maps) > 0:
                for m in maps:
                    xs = int(m.get('x_start', 0));
                    xz = int(m.get('x_size', 0))
                    ys = int(m.get('y_start', 0));
                    yz = int(m.get('y_size', 0))
                    if xz <= 0 or yz <= 0:
                        continue
                    xe = xs + xz;
                    ye = ys + yz
                    assert 0 <= xs < xe <= Dx and 0 <= ys < ye <= Dy, "[_apply_free_carry] y_to_x_map 越界"
                    assert xz == yz, "[_apply_free_carry] y_to_x_map 段长不一致"
                    x_next[..., xs:xe] = y_denorm[..., ys:ye]
            else:
                rx = getattr(self, 'rot6d_x_slice', None) or getattr(self, 'rot6d_slice', None)
                ry = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
                assert isinstance(rx, slice) and isinstance(ry, slice), "[_apply_free_carry] 缺少 rot6d 切片"
                len_rx = (rx.stop - rx.start);
                len_ry = (ry.stop - ry.start)
                assert len_rx == len_ry, f"[_apply_free_carry] rot6d 区间长度不等: X={len_rx} Y={len_ry}"
                x_next[..., rx] = y_denorm[..., ry]
        except Exception as e:
            if dbg:
                print(f"[_apply_free_carry][WARN] fallback copy due to: {e}")
            k = min(x_next.shape[-1], y_denorm.shape[-1])
            x_next[..., :k] = y_denorm[..., :k]

        # 2) 可选：重建 BoneAngularVelocities（避免把上一帧速率沿用到下一帧）
        try:
            av_sl = getattr(self, 'angvel_x_slice', None)
            rx = getattr(self, 'rot6d_x_slice', None) or getattr(self, 'rot6d_slice', None)
            ry = getattr(self, 'rot6d_y_slice', None) or getattr(self, 'rot6d_slice', None)
            if isinstance(av_sl, slice) and isinstance(rx, slice) and isinstance(ry, slice):
                Jx = (rx.stop - rx.start) // 6
                Jy = (ry.stop - ry.start) // 6
                if Jx == Jy and Jx > 0:
                    # prev R (from x_prev), curr R (from y_denorm)
                    prev6 = x_prev[..., rx].reshape(B, Jx, 6)
                    curr6 = x_next[..., rx].reshape(B, Jx, 6)  # 刚写回后的 rot6d
                    # 转矩阵（复用你现有的 rot6d_to_matrix 规则：columns=("X","Z")且左手系）
                    Rp = rot6d_to_matrix(prev6)  # [B,J,3,3]
                    Rc = rot6d_to_matrix(curr6)  # [B,J,3,3]
                    Rseq = torch.stack([Rp, Rc], dim=1)  # [B,2,J,3,3]
                    fps = float(getattr(self, 'bone_hz', 60.0) or 60.0)
                    w = angvel_vec_from_R_seq(Rseq, fps=fps)  # [B,1,J,3]
                    w = w[:, -1]  # [B,J,3]
                    x_next[..., av_sl] = w.reshape(B, Jx * 3)
                    if dbg:
                        l2 = torch.linalg.vector_norm(w, dim=-1).mean().item()
                        print(f"[_apply_free_carry][angvel] rebuilt |w|_mean={l2:.4f} rad/s @fps={fps}")
                else:
                    if dbg:
                        print(f"[_apply_free_carry][angvel] skip: Jx={Jx} Jy={Jy}")
        except Exception as e:
            if dbg:
                print(f"[_apply_free_carry][angvel][WARN] {e} (skip)")

        # 3) （可选）RootYaw/RootVelocity 暂维持 carry（Y 未直接预测它们）
        # 若后续你提供显式写回规则，可在此扩展。

        return x_next

    def _train_augment_if_needed(self, state_seq, gt_seq, cond_seq=None):
        """
        只在训练侧使用 augmentor：
          - time_warp: 对 X/Y 以及 cond_seq（若存在）使用一致的等长重采样
          - noise: 仅对 X 的若干切片加噪（rot6d/rootvel/angvel/yaw）
        评估路径不会调用本函数
        """
        import torch
        aug = getattr(self, 'augmentor', None)
        if aug is None:
            return state_seq, gt_seq, cond_seq

        # ---- 时间扭曲（概率触发）----
        prob = float(getattr(aug, 'time_warp_prob', 0.0) or 0.0)
        if prob > 0.0:
            if torch.rand(1, device=state_seq.device).item() < prob:
                # 轻度速度伸缩（可按需调）：[0.85, 1.15]
                scale = float(torch.empty(1, device=state_seq.device).uniform_(0.85, 1.15).item())
                state_seq = aug._time_warp(state_seq, scale)
                gt_seq = aug._time_warp(gt_seq, scale)
                if (cond_seq is not None) and (cond_seq.dim() == 3):
                    cond_seq = aug._time_warp(cond_seq, scale)
                try:
                    print(f"[TrainAug] time_warp scale={scale:.3f}")
                except Exception:
                    pass

        # ---- 加性高斯噪声（仅 X ）----
        std = float(getattr(aug, 'noise_std', 0.0) or 0.0)
        if std > 0.0:
            def _n(sl):
                if isinstance(sl, slice):
                    state_seq[:, :, sl] = state_seq[:, :, sl] + torch.randn_like(state_seq[:, :, sl]) * std

            # 复用你已注入到 Trainer 的切片
            _n(getattr(self, 'rot6d_x_slice', None))
            _n(getattr(self, 'rootvel_x_slice', None))
            _n(getattr(self, 'angvel_x_slice', None))
            _n(getattr(self, 'yaw_x_slice', None))

        return state_seq, gt_seq, cond_seq


def _contacts_to_LR_np(contacts_np):
    """
    contacts_np: (T,2) or (T,4) or None
    return L, R as (T,) float arrays in {0,1}
    """
    import numpy as np
    if contacts_np is None or contacts_np.size == 0:
        return (None, None)
    c = contacts_np.astype(np.int64)
    if c.shape[-1] >= 4:
        L = ((c[:, 0] > 0) | (c[:, 1] > 0)).astype(np.float32)
        R = ((c[:, 2] > 0) | (c[:, 3] > 0)).astype(np.float32)
    elif c.shape[-1] == 2:
        L = (c[:, 0] > 0).astype(np.float32)
        R = (c[:, 1] > 0).astype(np.float32)
    else:
        T = c.shape[0]
        L = np.zeros((T,), np.float32)
        R = np.zeros((T,), np.float32)
    return (L, R)

def _event_indices_from_LR(L, R, min_gap=6):
    """
    Rising/falling edges on either foot -> event positions.
    Coalesce events closer than min_gap frames.
    """
    import numpy as np
    if L is None or R is None:
        return []
    T = L.shape[0]
    Lb = (L > 0.5).astype(np.int32)
    Rb = (R > 0.5).astype(np.int32)
    dL = np.diff(Lb, prepend=Lb[:1])
    dR = np.diff(Rb, prepend=Rb[:1])
    cand = np.where((dL != 0) | (dR != 0))[0]
    out = []
    last = -10 ** 9
    for t in cand:
        if t - last >= int(max(1, min_gap)):
            out.append(int(t))
            last = t
    return out

def _window_starts_from_events(T, seq_len, events, pre=-1):
    import numpy as np
    if pre is None or pre < 0:
        pre = max(0, seq_len // 3)
    starts = []
    for e in events:
        s = int(e) - int(pre)
        s = max(0, min(s, T - seq_len))
        starts.append(s)
    if not starts:
        return starts
    starts = sorted(set(starts))
    if 0 not in starts:
        starts = [0] + starts
    tail = T - seq_len
    if tail >= 0 and tail not in starts:
        starts = starts + [tail]
    out = []
    last = -10 ** 9
    for s in starts:
        if s - last >= max(1, seq_len // 6):
            out.append(s)
            last = s
    return out

def _speed_from_X_layout(X, state_layout):
    """
    返回每帧“速度标量”（z 域）用于 IndexOpt 的 min_speed 过滤。
    兼容 state_layout 形态：
      - {'RootVelocity': {'start': 3, 'size': 2}}
      - {'RootVelocity': (3, 5)}   # [start, end)
      - {'RootVelocity': (3, 2)}   # [start, size]
      - {'RootVelocity': [3, 2]}   # 同上
      - 数值为 str/np 标量时自动 int()
    """
    import numpy as np

    try:
        if not isinstance(state_layout, dict):
            return None

        # ✅ 优先使用你已有的解析工具，避免重复实现
        try:
            sl = parse_layout_entry(state_layout.get('RootVelocity'), 'RootVelocity')
        except Exception:
            sl = None

        a = b = None
        if isinstance(sl, slice):
            a = int(sl.start or 0)
            b = int(sl.stop)
        else:
            rv = state_layout.get('RootVelocity')
            if rv is None:
                return None

            # dict 形式: {'start': x, 'size': y}
            if isinstance(rv, dict):
                a = int(rv.get('start', 0))
                sz = int(rv.get('size', 0))
                b = a + sz
            else:
                # 序列形式: (start, size) 或 (start, end)
                if hasattr(rv, '__iter__'):
                    r0 = int(rv[0])
                    r1 = int(rv[1])
                    # 判断 r1 是 size 还是 end（RootVelocity 通常 2 或 3 维）
                    if r1 in (1, 2, 3):
                        a, b = r0, r0 + r1
                    else:
                        a, b = r0, r1
                else:
                    return None

        if b is None or b <= a:
            return None

        v = X[:, a:b]
        spd = np.linalg.norm(v[:, :2], axis=1) if v.shape[1] >= 2 else np.abs(v[:, 0])
        return spd.astype(np.float32)
    except Exception:
        # 紧急兜底：过滤失效，但不会中断训练
        return None


def _maybe_optimize_dataset_index(ds, args):
    """
    Rebuild ds.index using stride/event/auto strategies.
    Safe to call for both train/val.
    """
    try:
        mode = getattr(args, 'index_mode', 'stride')
        stride = max(1, int(getattr(args, 'index_stride', 1)))
        min_gap = int(getattr(args, 'event_min_gap', 6))
        pre = int(getattr(args, 'window_pre', -1))
        min_speed = float(getattr(args, 'min_speed', 0.0))
    except Exception:
        mode, stride, min_gap, pre, min_speed = ('stride', 1, 6, -1, 0.0)
    new_index = []
    for cid, clip in enumerate(ds.clips):
        X = clip['X']
        T = int(X.shape[0])
        L = ds.seq_len
        contacts = clip.get('contacts', None)
        use_mode = mode
        if mode == 'auto' and (contacts is None or getattr(contacts, 'size', 0) == 0):
            use_mode = 'stride'
        if use_mode == 'event':
            Lc, Rc = _contacts_to_LR_np(contacts) if contacts is not None else (None, None)
            events = _event_indices_from_LR(Lc, Rc, min_gap=min_gap)
            starts = _window_starts_from_events(T, L, events, pre=pre)
            if not starts:
                starts = list(range(0, max(0, T - L + 1), stride))
        else:
            starts = list(range(0, max(0, T - L + 1), stride))
        if min_speed > 0.0:
            spd = _speed_from_X_layout(X, clip.get('meta', {}).get('state_layout', {}))
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
        print(f'[IndexOpt] mode={mode} stride={stride} min_gap={min_gap} pre={pre} min_speed={min_speed} -> windows={len(ds.index)}')
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
        except Exception as _hz_e:
            print(f"[HZ-ERR] hazard branch failed: {_hz_e}")
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
    se_x = _to_np(getattr(trainer, "s_eff_x", None) or getattr(trainer, "std_x", None))
    mu_y = _to_np(getattr(trainer, "mu_y", None))
    se_y = _to_np(getattr(trainer, "s_eff_y", None) or getattr(trainer, "std_y", None))

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


def train_entry():
    global GLOBAL_ARGS
    import argparse, warnings, os, glob, time, math, json
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    # ---- Slice helpers (inserted) ----
    def _safe_set_slice(obj, attr, maybe_slice):
        """Assign only if maybe_slice is a slice; avoid overwriting valid slices with None."""
        if isinstance(maybe_slice, slice):
            setattr(obj, attr, maybe_slice)

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--val_mode', type=str, default='online', choices=['online','none'])
    p.add_argument('--no_monitor', action='store_true', default=False)
    p.add_argument('--data', type=str, required=True, help='数据目录（含 *.npz）')
    p.add_argument('--out', type=str, default='./runs', help='输出目录根路径')
    p.add_argument('--run_name', type=str, default=None, help='子目录名；未给则用时间戳')
    p.add_argument('--train_files', type=str, default='', help='逗号分隔的路径/通配/或 @list.txt')
    p.add_argument('--dump_free_csv', type=str, default=None, help='若设置，则每个 epoch 在 free-run 评估时导出逐帧指标 CSV（路径或目录）')
    p.add_argument('--diag_topk', type=int, default=8, help='free-run 评估时打印 X_norm 的 |z| Top-K')

    p.add_argument('--diag_thr', type=float, default=8.0, help='|z| 阈值，统计 X_norm 爆炸比例')
    p.add_argument("--bundle_json", type=str, default=None, help='UE 导出的运行时 bundle（可含 MuY/StdY、feature_layout、MuC_other/StdC_other 等）', required=True)
    p.add_argument('--stats_json', type=str, default=None, help='包含 MuY/StdY 的 JSON')
    p.add_argument('--hz_dbg', type=int, default=0, help='>0 则打印每个 epoch 第一批次的 hazard 调试信息')
    p.add_argument('--print_hz_to_console', action='store_true', help='在控制台打印 hz_bce/hz_smooth/hz_mask_ratio')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--min_lr_ratio', type=float, default=0.05)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--tf_mode', type=str, default='epoch_linear', choices=['global', 'epoch_linear'])
    p.add_argument('--tf_warmup_epochs', type=int, default=3)
    p.add_argument('--tf_start_epoch', type=int, default=0)
    p.add_argument('--tf_end_epoch', type=int, default=10)
    p.add_argument('--pose_drop_max', type=float, default=0.0)
    p.add_argument('--pose_drop_start_epoch', type=int, default=0)
    p.add_argument('--pose_drop_end_epoch', type=int, default=0)
    p.add_argument('--rot6d_start', type=int, default=6)
    p.add_argument('--rot6d_len', type=int, default=276)
    p.add_argument('--tf_max', type=float, default=1.0)
    p.add_argument('--tf_min', type=float, default=0.1)
    p.add_argument('--use_dynamic_tf', action='store_true', default=False)
    p.add_argument('--no_dynamic_tf', dest='use_dynamic_tf', action='store_false')
    p.add_argument('--tf_warmup_steps', type=int, default=5000)
    p.add_argument('--tf_total_steps', type=int, default=200000)
    p.add_argument('--tf_start', type=float, default=0.9)
    p.add_argument('--tf_end', type=float, default=0.1)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--context_len', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--free_k', type=int, default=0, help='短地平线闭环自回归损失步数；0=禁用')
    p.add_argument('--w_free', type=float, default=0.0, help='闭环自由滚动损失权重')
    p.add_argument('--traj_hz', type=float, default=60.0)
    p.add_argument('--bone_hz', type=float, default=None)
    p.add_argument('--rot6d_eps', type=float, default=1e-06)
    p.add_argument('--w_rot_ortho', type=float, default=0.001)
    p.add_argument('--w_rot_geo', type=float, default=0.01)
    p.add_argument('--w_vel', type=float, default=0.0)
    p.add_argument('--w_acc', type=float, default=0.0)
    p.add_argument('--w_speed', type=float, default=0.0)
    p.add_argument('--w_speed_curve', type=float, default=0.2)
    p.add_argument('--w_speed_cosine', type=float, default=0.0)
    p.add_argument('--w_vel_cos', type=float, default=0.0)
    p.add_argument('--w_dir', type=float, default=0.0)
    p.add_argument('--amp_gate_k', type=float, default=8.0)
    p.add_argument('--amp_gate_th', type=float, default=0.2)
    p.add_argument('--aux_event', action='store_true', help='启用辅助接触事件头（隐式相位）')
    p.add_argument('--lambda_aux', type=float, default=0.2, help='事件头总权重')
    p.add_argument('--lambda_time', type=float, default=1.0, help='（可选）time-to-event 附加项权重')
    p.add_argument('--evt_bce_w_base', type=float, default=0.2)
    p.add_argument('--evt_bce_w_gain', type=float, default=0.8)
    p.add_argument('--evt_smooth_w', type=float, default=0.02)
    p.add_argument('--evt_hard_thresh', type=float, default=0.5)
    p.add_argument('--align_cond_to_y', action='store_true', help='将条件 C 对齐到目标 Y（即使用 C[1:] 对应 Y[1:]）')
    p.add_argument('--hz_pos_weight', type=str, default='', help='接触事件 BCE 的 pos_weight；可为 "auto"、标量，或逗号分隔的每事件权重')
    p.add_argument('--seq_len', type=int, default=120)
    p.add_argument('--yaw_aug_deg', type=float, default=0.0)
    p.add_argument('--normalize_c', action='store_true')
    p.add_argument('--aug_noise_std', type=float, default=0.0)
    p.add_argument('--aug_time_warp_prob', type=float, default=0.0)
    p.add_argument('--mixup_alpha', type=float, default=0.0)
    p.add_argument('--log_tb', action='store_true')
    p.add_argument('--tb_dir', type=str, default=None)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--offline_eval', action='store_true')
    p.add_argument('--eval_horizon', type=int, default=120)
    p.add_argument('--save_csv', type=str, default=None)
    p.add_argument('--index_mode', type=str, default='stride', choices=['stride', 'event', 'auto'], help='窗口起点策略：stride=固定步长滑窗；event=接触事件锚点；auto=优先事件，缺失退化为stride')
    p.add_argument('--index_stride', type=int, default=1, help='stride模式下的滑窗步长（>=1）')
    p.add_argument('--event_min_gap', type=int, default=6, help='事件间最小间隔帧数（去除抖动）')
    p.add_argument('--window_pre', type=int, default=-1, help='事件模式下窗口起点 = event_idx - window_pre；-1 表示使用 seq_len//3')
    p.add_argument('--min_speed', type=float, default=0.0, help='过滤“低速窗口”的阈值（m/s），0 表示不过滤')
    p.add_argument('--w_hazard_bce', type=float, default=1.0, help='BCE 权重')
    p.add_argument('--w_hazard_smooth', type=float, default=0.05, help='掩码平滑权重（未切换区间）')
    p.add_argument('--w_hazard_count', type=float, default=0.0, help='切换次数先验权重（可选）')
    p.add_argument('--smooth_edge_guard', type=int, default=0, help='切换邻域豁免帧数（平滑掩码膨胀半径）')
    p.add_argument('--smooth_on_logits', action='store_true', help='平滑项在logits空间计算而非概率空间')

    # === Boundary Consistency (BC) ===
    p.add_argument('--w_boundary', type=float, default=0.0, help='边界一致性损失权重；0 表示关闭')
    p.add_argument('--bc_stride', type=int, default=1, help='BC 的基础步长（1=相邻窗）')
    p.add_argument('--bc_stride_max', type=int, default=1, help='多尺度 BC 的最大步长；>1 启用 Multi-Stride BC')
    p.add_argument('--bc_ramp_start', type=int, default=1, help='从该 epoch 开始逐步放大 stride 上限')
    p.add_argument('--bc_ramp_end', type=int, default=0, help='达到最大 stride 的 epoch；0=不随 epoch 变化，始终用 bc_stride_max')
    p.add_argument('--events_monitor_batches', type=int, default=2, help='TrainEvents@epoch 采样批次数（默认2）')

    GLOBAL_ARGS = p.parse_args()

    def _arg(name, default=None):
        return getattr(GLOBAL_ARGS, name, default)


    train_paths = _expand_paths(_arg('train_files', ''))
    if not train_paths:
        if GLOBAL_ARGS.data and os.path.isdir(GLOBAL_ARGS.data):
            train_paths = sorted(glob.glob(os.path.join(GLOBAL_ARGS.data, '*.npz')))
        else:
            raise FileNotFoundError('No training files. Provide --train_files or --data with .npz')
    run_name = _arg('run_name') or time.strftime('%Y%m%d-%H%M%S')
    out_dir = Path(_arg('out', './runs')).expanduser() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ds_train = MotionEventDataset(GLOBAL_ARGS.data, seq_len=GLOBAL_ARGS.seq_len, paths=train_paths, align_cond_to_y=GLOBAL_ARGS.align_cond_to_y)
    ds_train = _maybe_optimize_dataset_index(ds_train, GLOBAL_ARGS)
    ds_train.is_train = True
    ds_train.yaw_aug_deg = float(_arg('yaw_aug_deg', 0.0))
    ds_train.normalize_c = bool(_arg('normalize_c', False))
    if not hasattr(ds_train, 'state_layout'):
        ds_train.state_layout = getattr(ds_train, 'state_layout', {}) or {}
    try:
        harmonize_angvel_units_inplace(ds_train, guess=False, verbose=True)
    except Exception as _e:
        print(f'[AngVelUnits] skip train: {_e}')
    pin = device.type == 'cuda'
    lkw = dict(num_workers=_arg('num_workers', 0), pin_memory=pin, persistent_workers=_arg('num_workers', 0) > 0, **{'prefetch_factor': 2} if _arg('num_workers', 0) > 0 else {})
    lkw['collate_fn'] = make_fixedlen_collate(_arg('seq_len', 60))
    train_loader = DataLoader(ds_train, batch_size=_arg('batch', 32), shuffle=True, drop_last=True, **lkw)
    Dx, Dy, Dc = (int(ds_train.Dx), int(ds_train.Dy), int(ds_train.Dc))
    L = int(_arg('depth', 2))
    H = int(_arg('width', 512))
    K = int(_arg('context_len', 16))
    print(f'[Export][Dims] Dx={Dx}, Dy={Dy}, Dc={Dc} | L={L}, H={H}, K={K}')

    model = EventMotionModel(in_state_dim=ds_train.Dx, out_motion_dim=ds_train.Dy, cond_dim=ds_train.Dc, hidden_dim=_arg('width', 512), num_layers=_arg('depth', 2), num_heads=_arg('num_heads', 4), dropout=_arg('dropout', 0.1), context_len=_arg('context_len', 16)).to(device)
    validate_and_fix_model_(model, Dx, Dc)
    validate_and_fix_model_(model)
    with torch.no_grad():
        _l0 = model.shared_encoder[0]
        if not torch.isfinite(_l0.weight).all() or (_l0.bias is not None and (not torch.isfinite(_l0.bias).all())):
            print('[Guard] first-linear became non-finite post-sanitize, reinitializing')
            torch.nn.init.kaiming_uniform_(_l0.weight, a=math.sqrt(5))
            if _l0.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(_l0.weight)
                bound = 1.0 / math.sqrt(max(fan_in, 1))
                torch.nn.init.uniform_(_l0.bias, -bound, bound)
            assert torch.isfinite(_l0.weight).all() and (_l0.bias is None or torch.isfinite(_l0.bias).all())
    with torch.no_grad():
        lin0 = model.shared_encoder[0]
        assert torch.isfinite(lin0.weight).all() and (lin0.bias is None or torch.isfinite(lin0.bias).all()), '[PostCheck] shared_encoder.0 still not finite'
    if bool(_arg('aux_event', False)):

        def _aux_store_hidden(module, inputs, output):
            try:
                h_seq = output[0] if isinstance(output, tuple) else output
            except Exception:
                h_seq = output
            setattr(model, '_last_hidden_seq', h_seq)
        try:
            if not hasattr(model, '_aux_hook') or model._aux_hook is None:
                model._aux_hook = model.temporal.register_forward_hook(_aux_store_hidden)
        except Exception as _e:
            print('[WARN] Failed to register aux hook on model.temporal:', _e)
    try:
        model._pasa_fps = float(getattr(ds_train, 'fps', 60.0))
    except Exception:
        pass
    loss_fn = MotionJointLoss(output_layout=ds_train.output_layout, fps=getattr(ds_train, 'fps', 60.0), traj_hz=_arg('traj_hz', 60.0), w_speed=_arg('w_speed', 0.0), w_speed_curve=_arg('w_speed_curve', 0.2), w_dir=_arg('w_dir', 0.0), rot6d_spec=getattr(ds_train, 'rot6d_spec', {}), w_rot_geo=_arg('w_rot_geo', 0.01), w_rot_ortho=_arg('w_rot_ortho', 0.001))
    # Hazard loss & event params from CLI
    loss_fn.w_hazard_bce = float(_arg('w_hazard_bce', 1.0))
    loss_fn.w_hazard_smooth = float(_arg('w_hazard_smooth', 0.05))
    loss_fn.w_hazard_count = float(_arg('w_hazard_count', 0.0))
    loss_fn.hz_pos_weight = _arg('hz_pos_weight', None) or None
    loss_fn.smooth_on_logits = bool(_arg('smooth_on_logits', False))
    loss_fn.smooth_edge_guard = int(_arg('smooth_edge_guard', 0))
    loss_fn.evt_hard_thresh = float(_arg('evt_hard_thresh', 0.5))
    loss_fn.evt_bce_w_base = float(_arg('evt_bce_w_base', 0.0))
    loss_fn.hz_dbg = int(_arg('hz_dbg', 0))

    print(
        f"[LossWeights] "
        f"w_speed_curve={loss_fn.w_speed_curve} "
        f"w_rot_geo={loss_fn.w_rot_geo} w_rot_ortho={loss_fn.w_rot_ortho} "
        f"w_hazard_bce={getattr(loss_fn, 'w_hazard_bce', 0.0)} "
        f"w_hazard_smooth={getattr(loss_fn, 'w_hazard_smooth', 0.0)} "
        f"w_hazard_count={getattr(loss_fn, 'w_hazard_count', 0.0)} "
        f"w_free={float(_arg('w_free', 0.0))} free_k={int(_arg('free_k', 0))}"
    )

    traj_hz = float(_arg('traj_hz', _arg('bone_hz', 60.0)))
    bone_hz = float(_arg('bone_hz', traj_hz))
    loss_fn.dt_traj = 1.0 / max(1e-6, traj_hz)
    loss_fn.dt_bone = 1.0 / max(1e-6, bone_hz)
    print(f"[Dt] dt_traj={loss_fn.dt_traj:.6f}s (traj_hz={traj_hz}) | "
          f"dt_bone={loss_fn.dt_bone:.6f}s (bone_hz={bone_hz})")

    loss_fn.w_vel = float(_arg('w_vel', 0.0))
    loss_fn.w_acc = float(_arg('w_acc', 0.0))
    loss_fn.w_vel_cos = float(_arg('w_vel_cos', 0.0))
    if hasattr(loss_fn, 'rot6d_eps'):
        loss_fn.rot6d_eps = float(_arg('rot6d_eps', 1e-06))
    augmentor = MotionAugmentation(noise_std=_arg('aug_noise_std', 0.0), time_warp_prob=_arg('aug_time_warp_prob', 0.0))
    trainer = Trainer(model=model, loss_fn=loss_fn, lr=_arg('lr', 0.0001), use_dynamic_tf=_arg('use_dynamic_tf', False), grad_clip=_arg('grad_clip', 0.0), weight_decay=_arg('weight_decay', 0.01), tf_warmup_steps=_arg('tf_warmup_steps', 5000), tf_total_steps=_arg('tf_total_steps', 200000), mixup_alpha=_arg('mixup_alpha', 0.0), augmentor=augmentor, use_swa=_arg('use_swa', False), swa_start_epoch=_arg('swa_start_epoch', 10), use_amp=_arg('amp', False), accum_steps=_arg('accum_steps', 1), scheduler=None)
    __apply_layout_center(ds_train, trainer)
    loss_fn.mu_y = getattr(trainer, "mu_y", None)
    loss_fn.std_y = getattr(trainer, "std_y", None)
    # 一次性归一化数值诊断
    _norm_debug_once(trainer, train_loader, thr=float(_arg('diag_thr', 8.0)), topk=int(_arg('diag_topk', 8)), print_to_console=True)
    trainer.free_k = int(_arg('free_k', 0) or 0)
    trainer.w_free = float(_arg('w_free', 0.0) or 0.0)
    trainer.hz_dbg = int(_arg('hz_dbg', 0))
    trainer.print_hz_to_console = bool(_arg('print_hz_to_console', True))
    trainer.aux_event = bool(_arg('aux_event', False))
    trainer.evt_bce_w_base = float(_arg('evt_bce_w_base', 0.2))
    trainer.evt_bce_w_gain = float(_arg('evt_bce_w_gain', 0.8))
    trainer.evt_smooth_w = float(_arg('evt_smooth_w', 0.02))
    trainer.evt_hard_thresh = float(_arg('evt_hard_thresh', 0.5))
    trainer.smooth_edge_guard = int(_arg('smooth_edge_guard', 0))
    trainer.smooth_on_logits = bool(_arg('smooth_on_logits', False))
    trainer.w_hazard_bce = float(_arg('w_hazard_bce', 1.0))
    trainer.w_hazard_smooth = float(_arg('w_hazard_smooth', 0.05))
    trainer.w_hazard_count = float(_arg('w_hazard_count', 0.0))

    # === Boundary Consistency config ===
    trainer.w_boundary = float(_arg('w_boundary', 0.0) or 0.0)
    trainer.bc_stride = int(_arg('bc_stride', 1) or 1)
    trainer.bc_stride_max = int(_arg('bc_stride_max', trainer.bc_stride) or trainer.bc_stride)
    trainer.bc_ramp_start = int(_arg('bc_ramp_start', 1) or 1)
    trainer.bc_ramp_end = int(_arg('bc_ramp_end', 0) or 0)
    trainer.hz_pos_weight = _arg('hz_pos_weight', None) or None
    trainer.lambda_aux = float(_arg('lambda_aux', 0.2))
    trainer.lambda_time = float(_arg('lambda_time', 1.0))
    trainer.events_head = None
    trainer.dump_free_csv = _arg('dump_free_csv', None)
    trainer.metric_for_best = _arg('metric_for_best', 'val_free')
    trainer.topk = _arg('topk', 3)


    _safe_set_slice(trainer, 'yaw_x_slice', parse_layout_entry(trainer._x_layout.get('RootYaw'), 'RootYaw'))
    _safe_set_slice(trainer, 'rootvel_x_slice', parse_layout_entry(trainer._x_layout.get('RootVelocity'), 'RootVelocity'))
    _safe_set_slice(trainer, 'angvel_x_slice', parse_layout_entry(trainer._x_layout.get('BoneAngularVelocities'), 'BoneAngularVelocities'))

    # 诊断参数（也可用命令行 --diag_topk/--diag_thr 覆盖）
    trainer.diag_topk = int(_arg('diag_topk', 8) or 8)
    trainer.diag_thr = float(_arg('diag_thr', 8.0) or 8.0)

    trainer.use_tb = bool(_arg('log_tb', False))
    trainer.tb_dir = _arg('tb_dir') or str(out_dir / f'tb_{run_name}')
    # === validation/monitor switches ===
    trainer.val_mode = _arg('val_mode', 'online')
    trainer.no_monitor = bool(_arg('no_monitor', False))
    trainer.events_monitor_batches = int(_arg('events_monitor_batches', 8) or 8)
    trainer.tf_mode = _arg('tf_mode', 'epoch_linear')
    trainer.tf_warmup_epochs = _arg('tf_warmup_epochs', 3)
    trainer.tf_start_epoch = _arg('tf_start_epoch', 0)
    trainer.tf_end_epoch = _arg('tf_end_epoch', 10)
    trainer.tf_max = _arg('tf_max', 1.0)
    trainer.tf_min = _arg('tf_min', 0.1)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, _arg('epochs', 300) * steps_per_epoch)
    effective_warmup = min(_arg('warmup_steps', 1000), int(total_steps * 0.1))
    base_lr = float(_arg('lr', 0.0001))
    min_lr = base_lr * float(max(1e-06, _arg('min_lr_ratio', 0.05)))

    def lr_lambda(step):
        # 关键：避免构造时把 LR 压到 1e-8 * base_lr
        if step <= 0:
            return 1.0
        if step < effective_warmup:
            return step / float(max(1, effective_warmup))
        t = (step - effective_warmup) / float(max(1, total_steps - effective_warmup))
        t = min(1.0, max(0.0, t))
        return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * 0.5 * (1.0 + math.cos(math.pi * t))


    lam0 = (max(1e-08, 0.0 / float(max(1, effective_warmup))) if effective_warmup > 0 else 1.0)

    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=lr_lambda)

    best_ckpt, history = trainer.fit(
        train_loader,
        epochs=_arg('epochs', 300),
        log_every=_arg('log_every', 50),
        out_dir=str(out_dir),
        patience=_arg('patience', 20),
        run_name=run_name
    )

    try:

        vloader = train_loader
        _mon_batches = int(_arg('events_monitor_batches', 8) or 8)
        _metrics = trainer.validate_autoreg_online(vloader, max_batches=_mon_batches)
        print(f"[ValFree@last] MSEnormY={_metrics['MSEnormY']:.6f} "
              f"GeoDeg={_metrics['GeoDeg']:.3f} "
              f"YawAbsDeg={_metrics['YawAbsDeg']:.3f} "
              f"RootVelMAE={_metrics['RootVelMAE']:.5f}"
              f"AngVelMAE={_metrics.get('AngVelMAE', float('nan')):.5f} rad/s")
    except Exception as _e:
        print(f"[ValFree] skipped due to error: {_e}")
        # 可选：若有 best_ckpt 就加载（保持你原有逻辑）
        try:
            if best_ckpt:
                ckpt_path = Path(best_ckpt).expanduser()
                if ckpt_path.is_file():
                    ckpt = torch.load(str(ckpt_path), map_location=device)
                    missing, unexpected = model.load_state_dict(ckpt['model'], strict=True)
                    assert not missing and (
                        not unexpected), f'state_dict mismatch: missing={missing}, unexpected={unexpected}'
                    print(f'[Load] best checkpoint loaded: {ckpt_path}')
                else:
                    print(f'[WARN] checkpoint not found: {ckpt_path}')
        except Exception as __e:
            print(f'[Load][WARN] failed to load best ckpt: {__e}')
    finally:
        # === 关键改动：无论上面成功/失败，这里都尝试导出 ONNX ===
        print('[Export][ENTER] preparing to export ONNX...')
        try:
            import os, traceback

            # 安全获取 SWA 包裹（若无 SWA 则回落到原模型）
            _swa = getattr(trainer, 'swa_model', None)
            model_to_export = getattr(_swa, 'module', _swa) if _swa is not None else model
            model_to_export = model_to_export.eval().cpu()

            onnx_path = os.path.join(str(out_dir), f'{run_name}_step_stateful_nophase.onnx')

            # 维度探测仅用于日志 & sanity；失败也不中断真正导出
            try:
                _b = next(iter(train_loader))
                Dx = int(_b['motion'].shape[-1])
                Dy = int(_b['gt_motion'].shape[-1])
                Dc = int(_b['cond_in'].shape[-1]) if 'cond_in' in _b else 0
                print(f'[Export][ProbeDims] Dx={Dx} Dy={Dy} Dc={Dc}')
                try:
                    sanity_check_model_dims(model_to_export, Dx, Dy, Dc)
                    print('[Export][Sanity] input dims check OK')
                except Exception as ee:
                    print('[Export][Sanity][WARN]', ee)
            except Exception as ee:
                print('[Export][ProbeDims][WARN] cannot read a batch for dim probe:', ee)

            os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
            export_onnx_step_stateful_nophase(
                model_to_export,
                train_loader,
                onnx_path,
                opset=18,
                dynamic_batch=False,
            )
        except Exception as e:
            print('[Export][ERROR]', e)
            traceback.print_exc()


def _first_linear_in_features(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    return None

def sanity_check_model_dims(model, Dx, Dy, Dc):
    nin = _first_linear_in_features(model)
    if nin is not None and nin != Dx + Dc:
        raise RuntimeError(f'[Guard] 模型第一层 in_features={nin}，但应为 Dx+Dc={Dx + Dc}；很可能构建时把 in_dim 设成 Dy+Dc={Dy + Dc} 了。')

@torch.no_grad()
def export_onnx_step_stateful_nophase(model: torch.nn.Module, loader, onnx_path: str, opset: int=18, dynamic_batch: bool=False):
    """
    单步(stateful)无相位 ONNX 导出：
      输入:  motion[B,Dx], h_in[L,B,H], kv_in[B,K,H], (可选) cond[B,Dc]
      输出:  motion_pred[B,Dy], h_out[L,B,H], kv_out[B,K,H]

    关键点：Dx 来自 batch['motion']/x_in_features；
          Dy 来自模型前向或 y_out_features；不再假设 Dx==Dy。
    """
    import os, torch
    if loader is None:
        raise ValueError('loader is None；需要 DataLoader 来获取示例形状。')

    def pick(*cands):
        for c in cands:
            if c is not None:
                return c
        return None
    batch = next(iter(loader))
    try:
        shape_dbg = {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, 'shape')}
        print('[Export][BatchShapes]', shape_dbg)
    except Exception:
        pass
    x_cand = pick(batch.get('motion', None), batch.get('X', None), batch.get('x_in_features', None))
    if x_cand is None:
        raise KeyError("Batch 缺少输入 X：需要 'motion' 或 'X' 或 'x_in_features'。")
    motion_seq = x_cand.float().cpu()
    c_cand = pick(batch.get('cond_in', None), batch.get('C', None), batch.get('cond_in', None))
    cond_seq = c_cand.float().cpu() if c_cand is not None else None
    y_cand = pick(batch.get('gt_motion', None), batch.get('Y', None), batch.get('y_out_features', None), batch.get('y_out_seq', None))
    y_seq = y_cand.float().cpu() if y_cand is not None else None
    if motion_seq.dim() == 3:
        _, T, Dx = motion_seq.shape
        motion0 = motion_seq[:1, 0, :]
    elif motion_seq.dim() == 2:
        T, Dx = motion_seq.shape
        motion0 = motion_seq[0:1, :]
    else:
        raise ValueError(f'Unexpected X shape: {tuple(motion_seq.shape)}')
    if cond_seq is None:
        cond0, Dc = (None, 0)
    else:
        if cond_seq.dim() == 3:
            cond0 = cond_seq[:1, 0, :]
        elif cond_seq.dim() == 2:
            cond0 = cond_seq[:1, :]
        else:
            raise ValueError(f'Unexpected C shape: {tuple(cond_seq.shape)}')
        Dc = int(cond0.shape[-1])
    device = torch.device('cpu')
    model = model.to(device).eval()
    try:
        ret = model(motion0, cond0, None, None)
    except TypeError:
        ret = model(motion0, cond0)
    except RuntimeError as e:
        nin_seen = int(motion0.shape[-1] + (0 if cond0 is None else cond0.shape[-1]))
        exp_in = _first_linear_in_features(model)
        hint = f'[Hint] 传入 in_dim={nin_seen}(=Dx+Dc)。 若模型第一层期望 in_dim≈{exp_in} 且不等于 Dx+Dc， 很可能把 in_dim 设成 Dy+Dc 了。请按 PATCH-C 修正。'
        raise RuntimeError(f'[Export] 前向失败：{e}\n{hint}') from e
    if isinstance(ret, dict):
        y0 = ret.get('out', None)
        if y0 is None:
            y0 = ret.get('motion', None)
        if y0 is None:
            y0 = ret.get('motion_pred', None)
        if y0 is None:
            for v in ret.values():
                if isinstance(v, torch.Tensor):
                    y0 = v
                    break
    elif isinstance(ret, (tuple, list)):
        y0 = ret[0] if len(ret) > 0 else None
    else:
        y0 = ret
    if not isinstance(y0, torch.Tensor):
        raise RuntimeError('无法从模型前向结果中确定主输出 (y0)。')
    Dy = int(y0.shape[-1])
    L = int(getattr(model, 'num_layers', 1))
    H = int(getattr(model, 'hidden_dim', getattr(model, 'width', y0.shape[-1])))
    K = int(getattr(model, 'context_len', 16))
    h_in_ex = torch.zeros((L, 1, H), dtype=torch.float32)
    kv_in_ex = torch.zeros((1, K, H), dtype=torch.float32)

    class _StepNoPhaseWrapper(torch.nn.Module):

        def __init__(self, core):
            super().__init__()
            self.core = core
            self.L = L
            self.H = H
            self.K = K

        def forward(self, motion, h_in, kv_in, cond=None):
            try:
                r = self.core(motion, cond, h_in, kv_in)
            except TypeError:
                r = self.core(motion, cond)
            cands = []
            if isinstance(r, dict):
                y = r.get('out', None)
                if y is None:
                    y = r.get('motion', None)
                if y is None:
                    y = r.get('motion_pred', None)
                for k, v in r.items():
                    if k in ('out', 'motion', 'motion_pred'):
                        continue
                    if isinstance(v, torch.Tensor):
                        cands.append(v)
            elif isinstance(r, (tuple, list)):
                y = r[0] if len(r) > 0 else None
                for v in r[1:]:
                    if isinstance(v, torch.Tensor):
                        cands.append(v)
            else:
                y = r
            ref_h = (self.L, motion.shape[0], self.H)
            ref_kv = (motion.shape[0], self.K, self.H)
            h_out = None
            kv_out = None
            for t in cands:
                shp = tuple(t.shape)
                if shp == ref_h and h_out is None:
                    h_out = t
                elif shp == ref_kv and kv_out is None:
                    kv_out = t
            if h_out is None:
                h_out = h_in
            if kv_out is None:
                kv_out = kv_in
            return (y, h_out, kv_out)
    wrapper = _StepNoPhaseWrapper(model).cpu().eval()
    motion_ex = motion0
    cond_ex = cond0 if Dc > 0 else None
    if cond_ex is not None:
        input_names = ['motion', 'h_in', 'kv_in', 'cond']
        example_args = (motion_ex, h_in_ex, kv_in_ex, cond_ex)
        dyn = {'motion': {0: 'B'}, 'h_in': {1: 'B'}, 'kv_in': {0: 'B'}, 'cond': {0: 'B'}, 'motion_pred': {0: 'B'}, 'h_out': {1: 'B'}, 'kv_out': {0: 'B'}}
    else:
        input_names = ['motion', 'h_in', 'kv_in']
        example_args = (motion_ex, h_in_ex, kv_in_ex)
        dyn = {'motion': {0: 'B'}, 'h_in': {1: 'B'}, 'kv_in': {0: 'B'}, 'motion_pred': {0: 'B'}, 'h_out': {1: 'B'}, 'kv_out': {0: 'B'}}
    output_names = ['motion_pred', 'h_out', 'kv_out']
    dynamic_axes = dyn if dynamic_batch else None
    print(f'[Export][Dims] Dx={Dx}, Dy={Dy}, Dc={Dc} | L={L}, H={H}, K={K}')
    if y_seq is not None:
        print(f'[Export][Check] batch Y shape={tuple(y_seq.shape)}（仅核对，不参与导出）')
    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
    torch.onnx.export(wrapper, example_args, f=onnx_path, export_params=True, opset_version=opset, do_constant_folding=True, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    print(f'[Export][OK] saved: {onnx_path}')

def __arpg__expand_paths_from_argv(flag_vals):
    """Expand one or more path/glob/dir strings into a file list of .npz"""
    import glob
    files = []
    items = []
    if not flag_vals:
        return []
    if isinstance(flag_vals, (list, tuple)):
        items = list(flag_vals)
    else:
        items = [flag_vals]
    for p in items:
        q = p
        if os.path.isdir(q):
            files.extend(sorted(glob.glob(os.path.join(q, '*.npz'))))
        elif any((ch in q for ch in '*?[')):
            files.extend(sorted(glob.glob(q)))
        elif os.path.isfile(q):
            files.append(q)
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

def _get_flag_value_from_argv(argv, flag, default=None):
    """
    从命令行参数列表中安全地获取某个标志的值。
    支持 --key value 和 --key=value 两种形式。
    """
    for tok in argv:
        if tok.startswith(flag + '='):
            return tok.split('=', 1)[1]
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv) and (not argv[i + 1].startswith('-')):
            return argv[i + 1]
    return default

def _get_flag_values_from_argv(argv, flag):
    """
    返回命令行里某个 flag 后面跟随的所有值（可能多次出现）。
    例如: --train_files a.npz b.npz --train_files /dir
    在遇到下一个以 '-' 开头的 token 时停止。
    支持逗号分隔的值。
    """
    vals = []
    for i, tok in enumerate(argv):
        if tok == flag:
            j = i + 1
            while j < len(argv) and (not argv[j].startswith('-')):
                vals.append(argv[j])
                j += 1
    out = []
    for v in vals:
        if ',' in v:
            out.extend([x for x in v.split(',') if x])
        else:
            out.append(v)
    return out

def main():
    """
    包装器主函数：
    1. 首先执行预分析并生成 bundle.json。
    2. 然后调用真正的训练函数 train_entry()。
    """
    argv0 = sys.argv[:]
    is_export_only = '--arpg_export_only' in argv0
    if is_export_only:
        rest_argv = [arg for arg in argv0 if arg != '--arpg_export_only']
    else:
        rest_argv = argv0
    out_dir_arg = _get_flag_value_from_argv(rest_argv, '--out') or _get_flag_value_from_argv(rest_argv, '-o')
    run_name_arg = _get_flag_value_from_argv(rest_argv, '--run_name')
    out_dir = out_dir_arg or './runs'
    run_name = run_name_arg or time.strftime('%Y%m%d-%H%M%S')
    train_files_flag = _get_flag_values_from_argv(rest_argv, '--train_files')
    data_dir_flag = _get_flag_value_from_argv(rest_argv, '--data')
    train_files = __arpg__expand_paths_from_argv(train_files_flag)
    if not train_files and data_dir_flag and os.path.isdir(os.path.expanduser(data_dir_flag)):
        train_files = __arpg__expand_paths_from_argv([data_dir_flag])


    from types import SimpleNamespace
    global GLOBAL_ARGS
    GLOBAL_ARGS = SimpleNamespace(out=out_dir, run_name=run_name, allow_val_on_train='--allow_val_on_train' in rest_argv, val_ratio=float(_get_flag_value_from_argv(rest_argv, '--val_ratio') or 0))
    print(f'[ARPG-PATCH] 参数准备完毕，即将进入训练入口: train_entry()')
    sys.argv = rest_argv
    try:
        train_entry()
    finally:
        sys.argv = argv0
if __name__ == '__main__':
    main()
