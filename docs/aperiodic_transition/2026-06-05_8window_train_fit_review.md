> TRIAGE: DOWNGRADE-SUPERSEDED. Historical design/probe/planning note only; do not treat as live action-handoff delivery plan. Superseded by `2026-06-07_action_handoff_inbetween_closeout_decision_record.md` §8/§9 under its stated read-only / zero-new-injection scope.

# 8-Window Train-Fit Review

Date: 2026-06-05

Scope: debug-only fixed-oracle-schedule Layer-2 train-fit discriminator. It uses flat `state281`, a tiny deterministic decoder, oracle support/contact schedule features, and the three causal items. It is not a production Trainer/runtime/gate/checkpoint change.

## 1. Verdict

- 8-window accepted-p99 debug train-fit: `false`.
- clean pass / p99-only / fail: `0` / `0` / `8`.
- This is not generalization, deployment readiness, or schedule-learning success.
- artifact: `debug_output/_tmp_action_handoff_8window_train_fit_beststage1_20260605/summary.json`.

| window | reason | support switches |
|---|---|---|
| `Walk_L_To_L:6-21` | support_switch_high_score | `[1, 2, 13]` |
| `Walk_R_To_L:46-61` | support_switch_high_score | `[2, 11, 13]` |
| `Walk_R_To_R:37-52` | support_switch_high_score | `[8, 14]` |
| `Walk_L_To_L:17-32` | target_high_rootvel | `[2]` |
| `Walk_R_To_L:70-85` | target_high_rootvel | `[8, 14]` |
| `Walk_R_To_R:77-92` | target_high_rootvel | `[15]` |
| `Walk_L_To_L:3-18` | target_high_yaw_rate | `[4, 5]` |
| `Walk_R_To_L:0-15` | target_high_yaw_rate | `[1, 4]` |

## 2. Stage1 supervised-fit-8

- `state281 [8,16,281] float32 CPU`, `bone_angvel [8,16,138] float32 CPU`, saved `pred_raw [8,6704] float32 CPU NumPy`.
- aggregate `state_mse=0.000000000143`, `bone_angvel_aux_mse=0.000000000406`.
- best supervised checkpoint: epoch `1906`, `flat_standardized_mse=5.425982441131794e-10`.
- accepted p99 pass: `0/8`.
- rows: `debug_output/_tmp_action_handoff_8window_train_fit_beststage1_20260605/stage1_supervised_fit.csv`.
- Stage1 reached the low-MSE GT basin; accepted-p99 failure is a heading-band/derived-metric preflight block, not a capacity or representation result.

Heading preflight audit artifact: `debug_output/_tmp_action_handoff_8window_train_fit_beststage1_20260605/heading_exact_gt_vs_stage1_audit.json`.

- exact `true_raw [8,6704] float32 CPU NumPy`: command/support-side `heading_error_p95_rad` range `2.580956827951785e-08..4.746637264927455e-08`, pass `8/8`.
- Stage1 best `pred_raw [8,6704] float32 CPU NumPy`: command/support-side `heading_error_p95_rad` range `2.9952751327338012e-05..6.165633076163152e-05`, pass `0/8`.
- `max_abs_delta_stage1_true_raw=0.0003399848937988281`.
- This does not show baseline heading itself is `>>1e-5`; it shows a near-GT low-MSE prediction triggering a derived heading/support-side heading preflight block under the ultra-tight heading contract.

## 3. Stage2 minimax-8

- skipped: `true`; reason: `stage1_supervised_fit_8_failed_preflight`.
- worst final `(window x metric)`: `Walk_R_To_R:77-92 support_side.heading_error_p95_rad` with normalized slack `-1672.69720287`.
- per-window accepted pass count: `0/8`.
- p95-shadow vs p99 decision: clean `0`, p99-only `0`, fail `8`.
- step/skip log: `debug_output/_tmp_action_handoff_8window_train_fit_beststage1_20260605/stage2_minimax_step_log.csv`.
- There is no minimax trend to interpret when Stage1 preflight is blocked.
- Therefore this artifact does not answer the real 8-window minimax/generalization question.

## 4. Stage3 stall classification

| window | state | classification | evidence |
|---|---|---|---|
| Walk_L_To_L:6-21 | fail | heading-band preflight block | Stage1 reached low-MSE GT basin, but command/support-side heading metrics failed the ultra-tight band before minimax. |
| Walk_R_To_L:46-61 | fail | heading-band preflight block | Stage1 reached low-MSE GT basin, but command/support-side heading metrics failed the ultra-tight band before minimax. |
| Walk_R_To_R:37-52 | fail | heading-band preflight block | Stage1 reached low-MSE GT basin, but command/support-side heading metrics failed the ultra-tight band before minimax. |
| Walk_L_To_L:17-32 | fail | heading-band preflight block | Stage1 reached low-MSE GT basin, but command/support-side heading metrics failed the ultra-tight band before minimax. |
| Walk_R_To_L:70-85 | fail | heading-band preflight block | Stage1 reached low-MSE GT basin, but command/support-side heading metrics failed the ultra-tight band before minimax. |
| Walk_R_To_R:77-92 | fail | heading-band preflight block | Stage1 reached low-MSE GT basin, but command/support-side heading metrics failed the ultra-tight band before minimax. |
| Walk_L_To_L:3-18 | fail | heading-band preflight block | Stage1 reached low-MSE GT basin, but command/support-side heading metrics failed the ultra-tight band before minimax. |
| Walk_R_To_L:0-15 | fail | heading-band preflight block | Stage1 reached low-MSE GT basin, but command/support-side heading metrics failed the ultra-tight band before minimax. |

- p99-only rows are treated as soft-fail and included above when present.
- No true multimodality evidence is claimed: deterministic fixed-schedule train-fit did not exclude contract-width/optimization explanations for non-clean rows.

## 5. Negative controls

- shortcut controls still fail: `true`.
- command demotion controls still fail: `true`.
- command demotion pass count: `0`.

| case | n | pass count/rate | failed families |
|---|---:|---:|---|
| artifact_proxy:bone_angvel_ramp_k1 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_ramp_k2 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_ramp_k3 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_ramp_k4 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_ramp_k5 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_ramp_k6 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k1 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k2 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k3 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k4 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k5 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k6 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:mapping_state281_bone_angvel_ramp_k3 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| artifact_proxy:mapping_state281_bone_angvel_ramp_k4 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| negative_control:direct_full | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| negative_control:lambda_force1 | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| negative_control:lambda_model | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| negative_control:linear_pose_contact_proxy | 3 | 0/0.0000 | {'command_response': 3} |
| negative_control:main | 3 | 0/0.0000 | {'rate_budget': 3, 'pose_continuity': 3, 'endpoint_bridgeability': 3} |
| negative_control:matched_hard_seam | 3 | 0/0.0000 | {'rate_budget': 3, 'endpoint_bridgeability': 3, 'pose_continuity': 2, 'support_honesty': 1} |
| negative_control:one_frame_angvel_root_switch | 3 | 0/0.0000 | {'rate_budget': 3, 'endpoint_bridgeability': 3} |

Artifact proxy rows cannot be rescored from complete trajectories in this artifact; their non-rate failed-family counts remain:

- `artifact_proxy:bone_angvel_ramp_k1`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_ramp_k2`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_ramp_k3`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_ramp_k4`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_ramp_k5`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_ramp_k6`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k1`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k2`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k3`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k4`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k5`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:bone_angvel_rootvel_cmdyaw_ramp_k6`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:mapping_state281_bone_angvel_ramp_k3`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`
- `artifact_proxy:mapping_state281_bone_angvel_ramp_k4`: `{'pose_continuity': 3, 'endpoint_bridgeability': 3}`

## 6. Next decision

Audit/relabel the command/support-side heading contract or add a heading-exactness repair, then rerun Stage2 minimax. Do not escalate to sampling/multimodality from this artifact.
