import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.geometry import geodesic_R, so3_exp_map, so3_log_map


def _blend(R_inc: torch.Tensor, R_dir: torch.Tensor, lam: float) -> torch.Tensor:
    R_res = torch.matmul(R_dir, R_inc.transpose(-1, -2))
    omega = so3_log_map(R_res)
    return torch.matmul(so3_exp_map(omega * lam), R_inc)


def main() -> None:
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    omega_dir = axis * 1.0

    R_inc = torch.eye(3, dtype=torch.float32)
    R_dir = so3_exp_map(omega_dir)

    R_l0 = _blend(R_inc, R_dir, 0.0)
    R_l1 = _blend(R_inc, R_dir, 1.0)
    R_lh = _blend(R_inc, R_dir, 0.5)

    deg = 180.0 / math.pi
    err_l0 = float(geodesic_R(R_l0.unsqueeze(0), R_inc.unsqueeze(0)).item()) * deg
    err_l1 = float(geodesic_R(R_l1.unsqueeze(0), R_dir.unsqueeze(0)).item()) * deg
    half_deg = float(geodesic_R(R_lh.unsqueeze(0), R_inc.unsqueeze(0)).item()) * deg
    full_deg = float(geodesic_R(R_dir.unsqueeze(0), R_inc.unsqueeze(0)).item()) * deg

    assert err_l0 < 1e-4, f"lambda=0 should keep incremental pose, got {err_l0:.6f} deg"
    assert err_l1 < 1e-4, f"lambda=1 should match direct pose, got {err_l1:.6f} deg"
    assert abs(half_deg - 0.5 * full_deg) < 1e-4, (
        f"lambda=0.5 should realize half residual angle, got {half_deg:.6f} vs expected {0.5 * full_deg:.6f} deg"
    )

    print("ok: lambda fusion SO(3) blend reaches incremental/direct endpoints and linear half-step angle")


if __name__ == "__main__":
    main()
