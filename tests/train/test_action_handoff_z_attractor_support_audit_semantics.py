from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# The audit lives under tools/ (not an importable package); load by path.
_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "run_action_handoff_z_attractor_support_audit.py"
)
_spec = importlib.util.spec_from_file_location("z_attractor_support_audit", _MOD_PATH)
audit = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(audit)


def test_circular_index_gap_wraps_around_cycle() -> None:
    # On a cycle of length 10, frame 1 and frame 9 are 2 apart, not 8.
    assert audit._circular_index_gap(1, 9, 10) == 2
    assert audit._circular_index_gap(0, 5, 10) == 5
    assert audit._circular_index_gap(3, 3, 10) == 0


def test_cos_dist_matrix_identical_zero_orthogonal_one() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = audit._cos_dist_matrix(a, a)
    assert abs(d[0, 0]) < 1e-9
    assert abs(d[1, 1]) < 1e-9
    assert abs(d[0, 1] - 1.0) < 1e-9  # orthogonal -> cos sim 0 -> dist 1


def test_binarize_contact_maps_two_channels_to_four_states() -> None:
    contact = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    thr = np.array([0.5, 0.5])
    states = audit._binarize_contact(contact, thr)
    assert states.tolist() == [0, 2, 1, 3]


def test_ridge_recovers_linear_signal_and_rejects_noise() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    w = rng.normal(size=(6, 2))
    y_signal = x @ w
    y_noise = rng.normal(size=(200, 2))
    train_mask = np.zeros(200, dtype=bool)
    train_mask[:140] = True

    r2_signal = audit._ridge_test_r2(x, y_signal, train_mask, audit.RIDGE_LAMBDA)
    r2_noise = audit._ridge_test_r2(x, y_noise, train_mask, audit.RIDGE_LAMBDA)
    assert r2_signal["r2_mean"] > 0.95
    assert r2_noise["r2_mean"] < 0.2


def test_a2_classifies_reachable_vs_source_off_support() -> None:
    # Target clip: tight end-window anchor near direction [1, 0, ...].
    rng = np.random.default_rng(1)
    dim = 8
    t = 40
    base = np.zeros((t, dim))
    base[:, 0] = np.linspace(0.2, 1.0, t)  # converges toward [1,0,...]
    target = base + 0.001 * rng.normal(size=(t, dim))

    # Reachable source: some frames point the same direction as the anchor.
    reach = 0.001 * rng.normal(size=(30, dim))
    reach[:, 0] = np.linspace(0.1, 1.0, 30)

    # Off-support source: points in an orthogonal direction.
    off = 0.001 * rng.normal(size=(30, dim))
    off[:, 1] = np.linspace(0.1, 1.0, 30)

    z_by = {
        "Walk_F": reach,  # stand-in; not a target
        "Walk_L_To_L": target,
        "Walk_L_To_R": off,
        "Walk_R_To_L": target,
        "Walk_R_To_R": target,
    }
    out = audit.audit_a2_reachability(z_by, end_window_k=8)
    rows = {(r["source"], r["target"]): r for r in out["pairs"]}

    reach_row = rows[("Walk_F", "Walk_L_To_L")]
    off_row = rows[("Walk_L_To_R", "Walk_L_To_L")]
    assert reach_row["classification"] == "reachable"
    assert off_row["classification"] == "source_off_support"
    assert reach_row["d_min_normalized_by_anchor_radius"] < off_row["d_min_normalized_by_anchor_radius"]


def test_a3_flags_z_contact_disagreement_on_reentry_phase() -> None:
    # Walk_F: contact cycles; z is a clean ring so z-NN and contact-NN can be
    # made to disagree by construction.
    t_f = 40
    phase = np.linspace(0, 2 * np.pi, t_f, endpoint=False)
    contact_f = np.stack([np.sin(phase), np.cos(phase)], axis=1)
    z_f = np.stack([np.cos(phase), np.sin(phase)], axis=1)

    # One turn clip whose end-frame contact matches Walk_F frame ~0, but whose
    # end-frame z points opposite (matches Walk_F frame ~t_f/2).
    turn_c = contact_f[0:1].repeat(10, axis=0)
    turn_z = (-z_f[0:1]).repeat(10, axis=0)
    z_by = {"Walk_F": z_f}
    c_by = {"Walk_F": contact_f}
    for clip in audit.TURN_CLIPS:
        z_by[clip] = turn_z
        c_by[clip] = turn_c

    out = audit.audit_a3_reentry(z_by, c_by)
    assert out["aggregate"]["z_agrees_with_contact_count"] == 0
    assert out["provisional_gate"]["interpretation"] in (
        "reentry_needs_contact_phase_not_z",
        "reentry_phase_ambiguous_from_contact",
    )
