#!/usr/bin/env python3
from pathlib import Path
import sys

import torch


def main() -> None:
    # Minimal sanity checks for rot_local tail selection helpers.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train.models import MotionJointLoss

    J = 18
    # A small tree with multiple leaves (end-effectors).
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 0, 16]
    offsets = [[0.0, 0.0, 0.0]] + [[0.0, 0.1, 0.0] for _ in range(J - 1)]

    loss = MotionJointLoss(output_layout={}, fps=60.0, rot6d_spec={}, w_rot_local=0.2)
    # Provide some names (may or may not match limb_monitor_names; that's OK).
    loss.set_bone_names([f"bone_{i}" for i in range(J)])
    loss.set_skeleton(parents, offsets)

    # Candidate pool should be stable and cached.
    cand1 = loss._rot_local_tail_candidates("keybones", J, torch.device("cpu"), k=3)
    cand2 = loss._rot_local_tail_candidates("keybones", J, torch.device("cpu"), k=3)
    if cand1 is not None:
        assert torch.equal(cand1, cand2)
        assert cand1.dtype == torch.long
        assert cand1.numel() > 0

    # EMA scoring should smooth across calls.
    loss.rot_local_tail_select = "ema"
    loss.rot_local_tail_ema_beta = 0.9
    per_bone0 = torch.linspace(0.0, 1.0, J)
    per_bone1 = per_bone0 + 1.0
    s0 = loss._rot_local_tail_scores(per_bone0)
    s1 = loss._rot_local_tail_scores(per_bone1)
    assert torch.is_tensor(s0) and torch.is_tensor(s1)
    # With increasing input, EMA output should also increase (elementwise).
    assert torch.all(s1 >= s0)

    print("[OK] rot_local tail helpers sanity passed.")


if __name__ == "__main__":
    main()
