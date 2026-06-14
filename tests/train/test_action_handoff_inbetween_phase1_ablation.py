"""Pure tests for PHASE 1 head/injection ablation aggregation."""
from __future__ import annotations

from train.data.action_handoff_inbetween import TURN_CLIPS
from tools.run_action_handoff_inbetween_phase1_head_ablation import AblationSpec, summarize_run


def _fake_summary(*, dual_plateau: bool, eval_min_norm_plateau: bool, l_r_min_norm: float):
    per_clip = {
        clip: {
            "reach_rate": 0.0,
            "reach_min_norm_mean": 4.0,
            "reach_min_norm_min": 4.0,
        }
        for clip in TURN_CLIPS
    }
    per_clip["Walk_L_To_R"] = {
        "reach_rate": 0.5,
        "reach_min_norm_mean": l_r_min_norm + 0.1,
        "reach_min_norm_min": l_r_min_norm,
    }
    return {
        "lever2_hidden_pre_loss": {
            "plateau_status": {
                "plateau": dual_plateau,
                "eval_min_norm": {
                    "plateau": eval_min_norm_plateau,
                    "reason": "synthetic",
                    "relative_improvement": 0.0,
                },
            }
        },
        "per_clip": per_clip,
    }


def test_phase1_success_eligibility_uses_eval_min_norm_not_dual_plateau():
    spec = AblationSpec("toy", 512, 2, "additive", "shared_encoder.1")
    row = summarize_run(
        spec,
        _fake_summary(dual_plateau=False, eval_min_norm_plateau=True, l_r_min_norm=1.4),
        conv_norm_thr=1.5,
    )

    assert row["plateau"] is False
    assert row["eval_min_norm_plateau"] is True
    assert row["any_clip_reached"] is True
    assert row["eligible_success"] is True


def test_phase1_success_eligibility_rejects_unstable_eval_min_norm_even_if_reached():
    spec = AblationSpec("toy", 512, 2, "additive", "shared_encoder.1")
    row = summarize_run(
        spec,
        _fake_summary(dual_plateau=True, eval_min_norm_plateau=False, l_r_min_norm=1.4),
        conv_norm_thr=1.5,
    )

    assert row["plateau"] is True
    assert row["eval_min_norm_plateau"] is False
    assert row["any_clip_reached"] is True
    assert row["eligible_success"] is False
