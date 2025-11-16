from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .layout_utils import (
    parse_layout_entry,
    normalize_layout as _normalize_layout,
    layout_span as _layout_span,
)


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
        with open(bundle_path, "r", encoding="utf-8") as f:
            b = json.load(f)
        self.bundle = b
        self.meta = b.get("meta", {}) or {}
        # required stats
        self.mu_x = np.asarray(b["MuX"], dtype=np.float32)
        self.std_x = np.asarray(b["StdX"], dtype=np.float32)
        self.mu_y = np.asarray(b["MuY"], dtype=np.float32)
        self.std_y = np.asarray(b["StdY"], dtype=np.float32)
        # raw layouts (strictly interpreted as [start,end))
        self.state_layout_raw = dict(self.meta.get("state_layout", {}))
        self.output_layout_raw = dict(self.meta.get("output_layout", {}))
        # normalized to [start,size]
        self.state_layout: Optional[Dict[str, Tuple[int, int]]] = None
        self.output_layout: Optional[Dict[str, Tuple[int, int]]] = None

        # optional
        self.y_to_x_map = list(self.meta.get("y_to_x_map", []))
        self.tanh_scales_rootvel = self.bundle.get("tanh_scales_rootvel", None)
        self.tanh_scales_angvel = self.bundle.get("tanh_scales_angvel", None)
        self.fps = float(self.meta.get("fps", 60.0))
        self.bone_names = list(self.meta.get("bone_names", []))
        self.rot6d_spec = dict(self.meta.get("rot6d_spec", {}))

    def strict_validate(self, Dx: int, Dy: int):
        # 1) stats dimension
        assert self.mu_x.size == Dx and self.std_x.size == Dx, (
            f"[FATAL] MuX/StdX length ({self.mu_x.size}/{self.std_x.size}) != Dx({Dx})"
        )
        assert self.mu_y.size == Dy and self.std_y.size == Dy, (
            f"[FATAL] MuY/StdY length ({self.mu_y.size}/{self.std_y.size}) != Dy({Dy})"
        )
        # 2) normalize to [start,size] using strict [start,end) semantics
        self.state_layout = _normalize_layout(self.state_layout_raw, Dx)
        self.output_layout = _normalize_layout(self.output_layout_raw, Dy)
        # 3) required keys
        need_x = ("RootYaw", "RootVelocity", "BoneRotations6D", "BoneAngularVelocities")
        need_y = ("BoneRotations6D",)
        for k in need_x:
            assert k in self.state_layout, f"[FATAL] meta.state_layout missing key: {k}"
        for k in need_y:
            assert k in self.output_layout, f"[FATAL] meta.output_layout missing key: {k}"
        # 4) ranges already validated above, but keep assertions
        for name, (st, sz) in self.state_layout.items():
            assert 0 <= st < Dx and 0 < sz <= Dx - st, (
                f"[FATAL] state_layout[{name}] OOR: start={st} size={sz} Dx={Dx}"
            )
        for name, (st, sz) in self.output_layout.items():
            assert 0 <= st < Dy and 0 < sz <= Dy - st, (
                f"[FATAL] output_layout[{name}] OOR: start={st} size={sz} Dy={Dy}"
            )

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
                y2x.append(
                    {
                        "name": name,
                        "x_start": int(xs),
                        "x_size": k,
                        "y_start": int(ys),
                        "y_size": k,
                    }
                )
        return y2x

    def apply_to_dataset(self, ds):
        ds.state_layout = dict(self.state_layout)
        ds.output_layout = dict(self.output_layout)
        if not hasattr(ds, "fps") or ds.fps is None:
            ds.fps = self.fps

    def apply_to_trainer(self, trainer):
        trainer._x_layout = dict(self.state_layout)
        trainer._y_layout = dict(self.output_layout)
        trainer.y_to_x_map = self.materialize_y_to_x_map()
        # stats
        trainer.mu_x = torch.tensor(np.asarray(self.mu_x).reshape(1, -1), dtype=torch.float32)
        trainer.std_x = torch.tensor(np.asarray(self.std_x).reshape(1, -1), dtype=torch.float32)
        trainer.mu_y = torch.tensor(np.asarray(self.mu_y).reshape(1, -1), dtype=torch.float32)
        trainer.std_y = torch.tensor(np.asarray(self.std_y).reshape(1, -1), dtype=torch.float32)
        # tanh scales passthrough
        trainer.tanh_scales_rootvel = self.tanh_scales_rootvel
        trainer.tanh_scales_angvel = self.tanh_scales_angvel
        # named slices
        trainer.yaw_slice = parse_layout_entry(trainer._y_layout.get("RootYaw"), "RootYaw")
        trainer.rootvel_slice = parse_layout_entry(trainer._y_layout.get("RootVelocity"), "RootVelocity")
        trainer.angvel_slice = parse_layout_entry(
            trainer._y_layout.get("BoneAngularVelocities"), "BoneAngularVelocities"
        )
        trainer.yaw_x_slice = parse_layout_entry(trainer._x_layout.get("RootYaw"), "RootYaw")
        trainer.rootvel_x_slice = parse_layout_entry(trainer._x_layout.get("RootVelocity"), "RootVelocity")
        trainer.angvel_x_slice = parse_layout_entry(
            trainer._x_layout.get("BoneAngularVelocities"), "BoneAngularVelocities"
        )
        trainer.rootpos_x_slice = parse_layout_entry(trainer._x_layout.get("RootPosition"), "RootPosition")
        # 明确注入 Rot6D 的 X/Y 切片
        trainer.rot6d_x_slice = parse_layout_entry(trainer._x_layout.get("BoneRotations6D"), "BoneRotations6D")
        trainer.rot6d_y_slice = parse_layout_entry(trainer._y_layout.get("BoneRotations6D"), "BoneRotations6D")

        # Forward axis hint for yaw reconstruction
        axis_map = {"X": 0, "Y": 1, "Z": 2}
        columns = []
        if isinstance(self.rot6d_spec, dict):
            cols_val = self.rot6d_spec.get("columns")
            if isinstance(cols_val, (list, tuple)):
                columns = [str(c).upper() for c in cols_val]
        forward_hint = columns[1] if len(columns) > 1 else (columns[0] if columns else "Z")
        trainer.yaw_forward_axis = axis_map.get(forward_hint, getattr(trainer, "yaw_forward_axis", 2))


def apply_layout_center(ds_train, trainer, bundle_path: str):
    if not bundle_path:
        raise SystemExit("[FATAL] 需要 --bundle_json 指定集中化的归一化与布局模板（不再支持猜测/回退）")

    center = LayoutCenter(bundle_path)
    center.strict_validate(int(ds_train.Dx), int(ds_train.Dy))

    center.apply_to_dataset(ds_train)
    center.apply_to_trainer(trainer)

    # 保存 bundle meta（验证用；不影响训练）
    try:
        trainer._bundle_meta = dict(center.meta)
    except Exception:
        trainer._bundle_meta = {}

    # 骨骼名（若数据集提供）
    if getattr(trainer, "_bone_names", None) is None and getattr(ds_train, "bone_names", None):
        try:
            trainer._bone_names = list(ds_train.bone_names)
        except Exception:
            pass

    # 注入 normalizer（放在布局注入之后）
    trainer.normalizer = DataNormalizer(
        mu_x=center.mu_x,
        std_x=center.std_x,
        mu_y=center.mu_y,
        std_y=center.std_y,
        y_to_x_map=center.materialize_y_to_x_map(),
        yaw_x_slice=trainer.yaw_x_slice,
        yaw_y_slice=trainer.yaw_slice,
        rootvel_x_slice=trainer.rootvel_x_slice,
        rootvel_y_slice=trainer.rootvel_slice,
        angvel_x_slice=trainer.angvel_x_slice,
        angvel_y_slice=trainer.angvel_slice,
        tanh_scales_rootvel=center.tanh_scales_rootvel,
        tanh_scales_angvel=center.tanh_scales_angvel,
        angvel_mode=getattr(ds_train, "angvel_norm_mode", None),
        angvel_mu=getattr(ds_train, "angvel_mu", None),
        angvel_std=getattr(ds_train, "angvel_std", None),
    )

    # 诊断输出（保持原格式）
    xlay = getattr(trainer, "_x_layout", getattr(ds_train, "state_layout", None))
    ylay = getattr(trainer, "_y_layout", getattr(ds_train, "output_layout", None))

    yaw = parse_layout_entry(ylay.get("RootYaw"), "RootYaw")
    rootv = parse_layout_entry(ylay.get("RootVelocity"), "RootVelocity")
    angv = parse_layout_entry(ylay.get("BoneAngularVelocities"), "BoneAngularVelocities")

    yaw_x = parse_layout_entry(xlay.get("RootYaw"), "RootYaw")
    rootv_x = parse_layout_entry(xlay.get("RootVelocity"), "RootVelocity")
    angv_x = parse_layout_entry(xlay.get("BoneAngularVelocities"), "BoneAngularVelocities")

    rot6d_x_span = _layout_span(xlay, "BoneRotations6D")
    rot6d_y_span = _layout_span(ylay, "BoneRotations6D")

    mx = getattr(trainer, "mu_x", None)
    sx = getattr(trainer, "std_x", None)
    my = getattr(trainer, "mu_y", None)
    sy = getattr(trainer, "std_y", None)
    len_mx = int(mx.numel()) if hasattr(mx, "numel") else (len(mx) if mx is not None else 0)
    len_sx = int(sx.numel()) if hasattr(sx, "numel") else (len(sx) if sx is not None else 0)
    len_my = int(my.numel()) if hasattr(my, "numel") else (len(my) if my is not None else 0)
    len_sy = int(sy.numel()) if hasattr(sy, "numel") else (len(sy) if sy is not None else 0)


class DataNormalizer:
    """封装数据规格与(反)归一化逻辑。"""

    def __init__(
        self,
        *,
        mu_x=None,
        std_x=None,
        mu_y=None,
        std_y=None,
        y_to_x_map=None,
        yaw_x_slice=None,
        yaw_y_slice=None,
        rootvel_x_slice=None,
        rootvel_y_slice=None,
        angvel_x_slice=None,
        angvel_y_slice=None,
        tanh_scales_rootvel=None,
        tanh_scales_angvel=None,
        traj_dir_slice=None,
        angvel_mode=None,
        angvel_mu=None,
        angvel_std=None,
    ):
        self.mu_x = None if mu_x is None else np.asarray(mu_x, dtype=np.float32)
        self.std_x = None if std_x is None else np.asarray(std_x, dtype=np.float32)
        self.mu_y = None if mu_y is None else np.asarray(mu_y, dtype=np.float32)
        self.std_y = None if std_y is None else np.asarray(std_y, dtype=np.float32)
        self.y_to_x_map = y_to_x_map or []
        self.yaw_x_slice = parse_layout_entry(yaw_x_slice, "RootYaw")
        self.yaw_y_slice = parse_layout_entry(yaw_y_slice, "RootYaw")
        self.rootvel_x_slice = parse_layout_entry(rootvel_x_slice, "RootVelocity")
        self.rootvel_y_slice = parse_layout_entry(rootvel_y_slice, "RootVelocity")
        self.angvel_x_slice = parse_layout_entry(angvel_x_slice, "BoneAngularVelocities")
        self.angvel_y_slice = parse_layout_entry(angvel_y_slice, "BoneAngularVelocities")
        self.traj_dir_slice = parse_layout_entry(traj_dir_slice, "TrajDir") if traj_dir_slice else None
        self.tanh_scales_rootvel = (
            None if tanh_scales_rootvel is None else np.asarray(tanh_scales_rootvel, dtype=np.float32)
        )
        self.tanh_scales_angvel = (
            None if tanh_scales_angvel is None else np.asarray(tanh_scales_angvel, dtype=np.float32)
        )
        self.angvel_mode = angvel_mode
        self.angvel_mu = None if angvel_mu is None else np.asarray(angvel_mu, dtype=np.float32)
        self.angvel_std = None if angvel_std is None else np.asarray(angvel_std, dtype=np.float32)

    # ---- public helpers -----------------------------------------------------
    def norm(self, x_raw):
        """
        支持 numpy / torch 张量的归一化，保持原行为但避免设备冲突。
        """
        if x_raw is None:
            return x_raw
        if self.mu_x is None or self.std_x is None:
            raise RuntimeError("[FATAL] DataNormalizer.norm 需要 mu_x/std_x。")

        # torch Tensor 分支：直接用同设备张量做算术，避免 .numpy() 带来的 device 错误
        try:
            import torch as _t

            if isinstance(x_raw, _t.Tensor):
                mu = _t.as_tensor(self.mu_x, device=x_raw.device, dtype=x_raw.dtype)
                std = _t.as_tensor(self.std_x, device=x_raw.device, dtype=x_raw.dtype)
                return (x_raw - mu) / std
        except Exception:
            pass

        # numpy 分支
        return (np.asarray(x_raw, dtype=np.float32) - self.mu_x) / self.std_x

    def denorm_x(self, xz: torch.Tensor, prev_raw: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        if self.mu_x is None or self.std_x is None:
            raise RuntimeError("[FATAL] DataNormalizer.denorm_x 需要 mu_x/std_x。")
        return xz * torch.as_tensor(self.std_x, device=xz.device) + torch.as_tensor(self.mu_x, device=xz.device)

    def norm_y(self, y_raw: torch.Tensor) -> torch.Tensor:
        if self.mu_y is None or self.std_y is None:
            raise RuntimeError("[FATAL] DataNormalizer.norm_y 需要 mu_y/std_y。")
        return (y_raw - torch.as_tensor(self.mu_y, device=y_raw.device)) / torch.as_tensor(
            self.std_y, device=y_raw.device
        )

    def denorm_y(self, yz: torch.Tensor) -> torch.Tensor:
        if self.mu_y is None or self.std_y is None:
            raise RuntimeError("[FATAL] DataNormalizer.denorm_y 需要 mu_y/std_y。")
        return yz * torch.as_tensor(self.std_y, device=yz.device) + torch.as_tensor(self.mu_y, device=yz.device)

    def denorm(self, yz: torch.Tensor) -> torch.Tensor:
        return self.denorm_y(yz)

    def y_to_x(self, dy_hat: torch.Tensor, xz: torch.Tensor) -> torch.Tensor:
        """
        将 Y 维度的预测结果写回 X 空间（用于混合 roll-out）。
        """
        if not self.y_to_x_map:
            return xz
        x_out = xz.clone()
        for item in self.y_to_x_map:
            xs, xk = int(item["x_start"]), int(item["x_size"])
            ys, yk = int(item["y_start"]), int(item["y_size"])
            if xk <= 0 or yk <= 0:
                continue
            x_out[..., xs : xs + xk] = dy_hat[..., ys : ys + yk]
        return x_out
