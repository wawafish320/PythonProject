from __future__ import annotations

import json
import math
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from train.geometry import rot6d_to_angle_deg_np


class DatasetProfiler:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def profile(self) -> Dict[str, Any]:
        files = sorted(self.raw_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON clips under {self.raw_dir}")
        seq_lengths: List[int] = []
        yaw_vals: List[float] = []
        speed_vals: List[float] = []
        bone_angles: List[float] = []

        for path in files:
            data = json.loads(path.read_text())
            frames = data.get("Frames") or []
            seq_lengths.append(len(frames))
            for fr in frames:
                # yaw 不是显式特征：这里用 TrajectoryDir 的中心前瞻方向估计 yaw（度），用于粗略复杂度统计。
                traj_dir = fr.get("TrajectoryDir", None)
                if isinstance(traj_dir, list) and traj_dir:
                    vec = None
                    # 支持两种格式：
                    # - [x, y]
                    # - [[x, y], [x, y], ...]（多前瞻点）
                    if len(traj_dir) >= 2 and not isinstance(traj_dir[0], (list, tuple)):
                        vec = traj_dir
                    else:
                        mid = len(traj_dir) // 2
                        cand = traj_dir[mid]
                        if isinstance(cand, (list, tuple)) and len(cand) >= 2:
                            vec = cand
                    if vec is not None:
                        try:
                            x = float(vec[0])
                            y = float(vec[1])
                            if math.isfinite(x) and math.isfinite(y) and math.hypot(x, y) > 1e-6:
                                yaw_vals.append(math.degrees(math.atan2(y, x)))
                        except Exception:
                            pass
                rv = fr.get("RootVelocityXY") or [0.0, 0.0]
                speed_vals.append(math.hypot(rv[0], rv[1]))
                rotations = fr.get("BoneRotations")
                if rotations:
                    for rot in rotations:
                        try:
                            arr = np.asarray(rot, dtype=np.float64)
                            bone_angles.append(
                                float(
                                    rot6d_to_angle_deg_np(
                                        arr,
                                        columns=("X", "Z"),
                                        canonical_axis_order=True,
                                    )
                                )
                            )
                        except Exception:
                            continue

        total_frames = sum(seq_lengths)
        yaw_mean = stats.mean(yaw_vals) if yaw_vals else 0.0
        yaw_std = stats.pstdev(yaw_vals) if len(yaw_vals) > 1 else 0.0
        speed_mean = stats.mean(speed_vals)
        speed_std = stats.pstdev(speed_vals) if len(speed_vals) > 1 else 0.0
        bone_mean = stats.mean(bone_angles) if bone_angles else 45.0
        bone_std = stats.pstdev(bone_angles) if len(bone_angles) > 1 else 0.0
        complexity = min(2.0, 0.5 * (yaw_std / 30.0 + speed_std / 0.3))

        return {
            "n_clips": len(files),
            "total_frames": total_frames,
            "avg_seq_len": stats.mean(seq_lengths) if seq_lengths else 60.0,
            "median_seq_len": stats.median(seq_lengths) if seq_lengths else 60.0,
            "yaw_mean_deg": yaw_mean,
            "yaw_std_deg": yaw_std,
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "bone_angle_mean_deg": bone_mean,
            "bone_angle_std_deg": bone_std,
            "complexity": complexity,
        }


def compute_total_epochs(total_frames: int) -> int:
    if total_frames < 2000:
        return 60
    if total_frames < 10000:
        return 45
    if total_frames < 50000:
        return 30
    return 20


def compute_batch_size(avg_seq_len: float) -> int:
    if avg_seq_len < 50:
        return 16
    if avg_seq_len < 90:
        return 8
    if avg_seq_len < 160:
        return 6
    return 4


def compute_base_lr(total_frames: int, complexity: float, batch_size: int) -> float:
    scale = max(1.5, math.log10(max(total_frames, 10))) / 5.0
    comp = 1.0 / (1.0 + complexity)
    lr = 1e-3 * scale * comp * math.sqrt(batch_size / 8.0)
    return max(1e-5, min(6e-4, lr))
