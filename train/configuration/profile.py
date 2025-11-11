from __future__ import annotations

import json
import math
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List


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
                yaw_vals.append(abs(fr.get("RootYaw", 0.0)) * (180.0 / math.pi))
                rv = fr.get("RootVelocityXY") or [0.0, 0.0]
                speed_vals.append(math.hypot(rv[0], rv[1]))
                rotations = fr.get("BoneRotations")
                if rotations:
                    for rot in rotations:
                        try:
                            bone_angles.append(_rot6d_to_angle(rot))
                        except Exception:
                            continue

        total_frames = sum(seq_lengths)
        yaw_mean = stats.mean(yaw_vals)
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


def _rot6d_to_angle(vec: List[float]) -> float:
    a = vec[:3]
    b = vec[3:6]

    def _norm(v: List[float]) -> List[float]:
        n = math.sqrt(sum(x * x for x in v))
        if n < 1e-8:
            return list(v)
        return [x / n for x in v]

    X = _norm(a)
    proj = sum(X[i] * b[i] for i in range(3))
    b_ortho = [b[i] - proj * X[i] for i in range(3)]
    Z = _norm(b_ortho)
    Y = _norm(
        [
            Z[1] * X[2] - Z[2] * X[1],
            Z[2] * X[0] - Z[0] * X[2],
            Z[0] * X[1] - Z[1] * X[0],
        ]
    )
    trace = X[0] + Y[1] + Z[2]
    val = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return math.degrees(math.acos(val))
