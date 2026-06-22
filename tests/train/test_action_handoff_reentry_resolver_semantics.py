from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "run_action_handoff_reentry_resolver_diag.py"
)
_spec = importlib.util.spec_from_file_location("reentry_resolver_diag", _MOD_PATH)
resolver = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(resolver)


def test_circular_spread_wraps_and_zero_for_singleton() -> None:
    assert resolver._circular_spread([0], 40) == 0.0
    # frames 39, 0, 1 on a 40-cycle are tightly clustered around the wrap point.
    spread = resolver._circular_spread([39, 0, 1], 40)
    assert spread < 0.05


def test_l2_to_query_scales() -> None:
    track = np.array([[0.0, 0.0], [3.0, 4.0]])
    d = resolver._l2_to_query(track, np.array([0.0, 0.0]), scale=1.0)
    assert abs(d[0]) < 1e-9
    assert abs(d[1] - 5.0) < 1e-9
    d2 = resolver._l2_to_query(track, np.array([0.0, 0.0]), scale=5.0)
    assert abs(d2[1] - 1.0) < 1e-9


def test_resolver_uses_pose_neighborhood_and_refines_with_contact() -> None:
    # Walk_F: pose is monotonic (localizes phase); contact is aliased so its
    # global min sits at a different phase than pose.
    n_f = 40
    pose_f = np.stack([np.arange(n_f, dtype=float), np.zeros(n_f)], axis=1)
    contact_f = np.ones((n_f, 2)) * 5.0
    contact_f[20] = [0.0, 0.0]  # contact global-min decoy at frame 20
    contact_f[5] = [0.1, 0.0]  # best contact within the pose neighborhood
    egovel_f = np.zeros((n_f, 2))

    # turn clip end (last frame) pose matches Walk_F frame 5; contact ~[0,0].
    pose_c = np.zeros((10, 2))
    pose_c[-1] = [5.0, 0.0]
    contact_c = np.zeros((10, 2))
    egovel_c = np.zeros((10, 2))

    res = resolver._resolve_clip(
        "TurnX",
        pose_f=pose_f,
        contact_f=contact_f,
        egovel_f=egovel_f,
        pose_c=pose_c,
        contact_c=contact_c,
        egovel_c=egovel_c,
        n_f=n_f,
        pose_scale=1.0,
    )
    assert res["pose_nn_frame"] == 5
    assert res["contact_nn_frame"] == 20  # aliased contact global-min
    assert res["resolver_reentry_frame"] == 5  # pose neighborhood + contact refine
    assert res["pose_vs_contact_disagree"] is True
    assert res["pose_sharpness_circular"] < resolver.POSE_SHARP_MAX
