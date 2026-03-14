# cp015 oldplan component ablation

- run_date: 20260314
- baseline: `cp015_bestfree` (kept as source baseline only)
- control: `cp015_with_old_planstack` (reused full old-plan challenger lane)
- stage6 config: `/Users/xingzhaorui/PycharmProjects/PythonProject/config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`

## Key groups

| group | prefixes / exact | key_count | rationale |
|---|---|---:|---|
| A. plan head rollback | `contact_plan_head., contact_plan_time_head.` | 8 | Readout layer from plan_z to contacts_plan logits, including the additive time-PE bias head. |
| B. plan init-state rollback | `contact_plan_init_head., contact_plan_init_z, contact_phase_state_init` | 8 | Cold-start state for plan_z / phase-state, including learnable init vectors and obs-conditioned init head. |
| C. planner-core rollback | `contact_plan_cell., event_clock_gate., event_clock_corrector.` | 22 | Core recurrent planner latent plus Event-Clock gate/corrector that rescales and corrects plan_z inside the loop. |
| D. phase/contact input-side rollback | `contact_plan_phase_head., contact_phase_state_delta_head.` | 12 | Phase/contact side inputs into the planner: phase residual on logits and phase-state update head driven by cond/meas/delta_meas. |

## Stage6 screening

| lane | rollback groups | all_ex_root | leg | nonleg | arm | else | delta vs control all_ex_root | delta vs control leg | delta vs cp015 | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cp015_with_old_planstack | plan_head, plan_init_state, planner_core, phase_contact_input | 0.295533 | 0.740703 | 0.199280 | 0.228705 | 0.129730 | 0.000000 | 0.000000 | -0.135844 | control_reused |
| rollback_plan_head | plan_head | 0.360335 | 0.818102 | 0.261358 | 0.304739 | 0.158822 | 0.064802 | 0.077399 | -0.071042 | screened_out |
| rollback_plan_init_state | plan_init_state | 0.414304 | 1.033133 | 0.280503 | 0.312104 | 0.205811 | 0.118772 | 0.292431 | -0.017073 | screened_out |
| rollback_planner_core | planner_core | 0.305250 | 0.766829 | 0.205449 | 0.238882 | 0.126425 | 0.009717 | 0.026126 | -0.126127 | promoted |
| rollback_phase_contact_input | phase_contact_input | 0.414361 | 1.141543 | 0.257133 | 0.297039 | 0.162807 | 0.118828 | 0.400840 | -0.017016 | screened_out |

## Stage6 ranking

| rank | lane | delta all_ex_root | delta leg | delta nonleg | rollback_effective_changed_keys |
|---:|---|---:|---:|---:|---:|
| 1 | rollback_planner_core | 0.009717 | 0.026126 | 0.006169 | 22 |
| 2 | rollback_plan_head | 0.064802 | 0.077399 | 0.062079 | 8 |
| 3 | rollback_plan_init_state | 0.118772 | 0.292431 | 0.081224 | 7 |
| 4 | rollback_phase_contact_input | 0.118828 | 0.400840 | 0.057853 | 8 |

- promote_reason: `threshold_pass`; promoted: `rollback_planner_core`
- note: Promote lanes whose Stage6 gap stays within +0.02 all_ex_root and +0.05 leg vs full oldplan control.

## Downstream

| lane | DirectGeoLocalDeg(model) | BlendGeoLocalDeg(model) | GeoLocalDeg(model) | all_ex_root(model) | leg(model) | nonleg(model) | delta vs full oldplan all_ex_root | delta vs accepted final all_ex_root |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rollback_planner_core | 0.114635 | 0.113263 | 0.462037 | 0.114635 | 0.296311 | 0.075354 | -0.005510 | 0.001688 |

| lane | strict DirectGeoLocalDeg | strict all_ex_root | strict leg | strict nonleg | strict delta vs full oldplan all_ex_root |
|---|---:|---:|---:|---:|---:|
| rollback_planner_core | 0.114862 | 0.114862 | 0.286085 | 0.077841 | -0.003021 |

## Promoted lane paths

| lane | stage6 | 70a | 70b | new70b_replace | 70R | 71 | 72 | lambda |
|---|---|---|---|---|---|---|---|---|
| rollback_planner_core | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_component_ablation_20260314/stage6/rollback_planner_core/posttrain/ckpt_last_WalkF_stage6_rollback_planner_core_20260314.pth` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_component_ablation_20260314/downstream/rollback_planner_core/70a/ckpt_last_WalkF_stage7_70a_from_rollback_planner_core_20260314.pth` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_component_ablation_20260314/downstream/rollback_planner_core/70b/ckpt_last_WalkF_stage7_70b_from_rollback_planner_core_20260314.pth` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_component_ablation_20260314/downstream/rollback_planner_core/70b_replace/ckpt_last_WalkF_stage7_70b_replace_from_rollback_planner_core_20260314.pth` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_component_ablation_20260314/downstream/rollback_planner_core/70R/ckpt_last_WalkF_stage7_70R_from_rollback_planner_core_20260314.pth` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_component_ablation_20260314/downstream/rollback_planner_core/71/ckpt_last_WalkF_stage7_71_from_rollback_planner_core_20260314.pth` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_component_ablation_20260314/downstream/rollback_planner_core/72/ckpt_last_WalkF_stage7_72_from_rollback_planner_core_20260314.pth` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_component_ablation_20260314/downstream/rollback_planner_core/lambda/ckpt_last_WalkF_stage7_lambda_from_rollback_planner_core_20260314.pth` |

## Control refs

- full oldplan control strict all_ex_root=0.117883, leg=0.280194, nonleg=0.082789
- full oldplan control model all_ex_root=0.120145, leg=0.278087, nonleg=0.085995
- accepted final model-source all_ex_root=0.112947, leg=0.274360, nonleg=0.078048

## Answers

1. Most critical rollback group: `rollback_phase_contact_input` (delta all_ex_root=0.118828, leg=0.400840).
2. Has smaller Stage6-safe version: `true`.
3. Smaller version penetrates to lambda final: `true`.
4. Smaller version beats full oldplan chain: `false` (Overall direct/all_ex_root improves vs full oldplan control, but leg regresses, so treat this as a mixed tradeoff rather than a clean win.)
5. Smaller version beats current accepted final: `false`.
6. Can propose cleaner challenger lane: `true` (Cleaner challenger/control lane only; baseline switch still requires clearing the current accepted final.)
7. Should pause further simplification: `false`.

