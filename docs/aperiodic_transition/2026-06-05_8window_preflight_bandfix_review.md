> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# 8-Window Preflight Bandfix Review

Scope: debug-only stopped preflight audit. No production Trainer/runtime/gate/checkpoint mutation.

## 1. Stop condition

- Stage2 was not run because E3 did not find a strict decoder-vs-negative heading seam.
- max decoder-side heading: `0.00000005` rad.
- min available negative-control heading: `0.00000000` rad.
- Per prompt, this means heading is non-discriminative for the available negative controls and must be treated as a passenger/gross-violation check unless the negative-control set is partitioned.
- min-heading negative examples: `[{'source': 'shortcut_sequence', 'case': 'negative_control:matched_hard_seam', 'target': 'Walk_R_To_R', 'start_phase': 'phi=82;onset=0;H=2', 'heading_error_p95_rad': 0.0}, {'source': 'command_demotion_reconstructed_sequence', 'case': 'negative_control:matched_hard_seam', 'target': 'Walk_R_To_R', 'start_phase': 'phi=82;onset=0;H=2', 'heading_error_p95_rad': 0.0}, {'source': 'shortcut_sequence', 'case': 'negative_control:one_frame_angvel_root_switch', 'target': 'Walk_R_To_R', 'start_phase': 'phi=82;onset=0;H=2', 'heading_error_p95_rad': 0.0}, {'source': 'command_demotion_reconstructed_sequence', 'case': 'negative_control:one_frame_angvel_root_switch', 'target': 'Walk_R_To_R', 'start_phase': 'phi=82;onset=0;H=2', 'heading_error_p95_rad': 0.0}]`.

## 2. E1 full-family preflight

- exact GT accepted pass: `8/8`.
- decoder-replay-from-GT accepted pass: `8/8`.
- Stage1-best fail buckets: `{'fit_residual_amplification': 13, 'band_too_tight_for_cross_window_GT': 15}`.
- Stage1-best fail classes: `{'A_heading': 16, 'B_interval_gt_edge': 10, 'C_sibling_window_upper': 2}`.

## 3. E2 Class B/C relabel attempt

- relabel rows audited: `11`.
- rows that actually widened: `0`.
- Under the implemented `pooled p1/p99 + no tightening` rule, B/C rows did not widen beyond the existing inclusive GT min/max bands; see `band_relabel_classBC.csv`.
- That is a discrepancy against the intended 'GT gets genuine slack' criterion and should be resolved before any Stage2 rerun.

## 4. Artifacts

- gt_decoder_fullfamily_preflight_csv: `debug_output/_tmp_action_handoff_8window_preflight_bandfix_20260605/gt_decoder_fullfamily_preflight.csv`
- gt_decoder_fullfamily_preflight_json: `debug_output/_tmp_action_handoff_8window_preflight_bandfix_20260605/gt_decoder_fullfamily_preflight.json`
- band_relabel_classBC_csv: `debug_output/_tmp_action_handoff_8window_preflight_bandfix_20260605/band_relabel_classBC.csv`
- heading_seam_audit_csv: `debug_output/_tmp_action_handoff_8window_preflight_bandfix_20260605/heading_seam_audit.csv`
- heading_seam_audit_json: `debug_output/_tmp_action_handoff_8window_preflight_bandfix_20260605/heading_seam_audit.json`
- summary_json: `debug_output/_tmp_action_handoff_8window_preflight_bandfix_20260605/summary.json`
- summary_md: `debug_output/_tmp_action_handoff_8window_preflight_bandfix_20260605/summary.md`
- doc_md: `docs/aperiodic_transition/2026-06-05_8window_preflight_bandfix_review.md`
