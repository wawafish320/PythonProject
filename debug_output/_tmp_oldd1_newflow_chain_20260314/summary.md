# old d1 basetrain + new posttrain flow

- run_date: 20260314
- base_ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/MLPL2_DirectBranch_v1/exp_phase_DirectBranch_v1_d1/ckpt_best_free_exp_phase_DirectBranch_v1_d1.pth`
- stage6_ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_stage6_basetrain_compare_20260313/old_bestfree/ckpt_last_old_bestfree_stage6_cmp_20260313.pth`
- stage6 strict: all_ex_root=0.313279, leg=0.874230, nonleg=0.191993, arm=0.217491, else=0.131724

## Checkpoints

| stage | ckpt |
|---|---|
| stage6 | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_stage6_basetrain_compare_20260313/old_bestfree/ckpt_last_old_bestfree_stage6_cmp_20260313.pth` |
| 70a | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_newflow_chain_20260314/70a/ckpt_last_WalkF_stage7_70a_from_oldd1_newflow_20260314.pth` |
| 70b | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_newflow_chain_20260314/70b/ckpt_last_WalkF_stage7_70b_from_oldd1_newflow_20260314.pth` |
| 70a_replace_warmstart | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_newflow_chain_20260314/warmstart/ckpt_last_oldd1_70a_replacecontacts_zerophase_20260314.pth` |
| new70b_replace | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_newflow_chain_20260314/70b_replace/ckpt_last_WalkF_stage7_70b_replace_from_oldd1_newflow_20260314.pth` |
| 70R | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_newflow_chain_20260314/70R/ckpt_last_WalkF_stage7_70R_from_oldd1_newflow_s180_20260314.pth` |
| 71 | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_newflow_chain_20260314/71/ckpt_last_WalkF_stage7_71_from_oldd1_newflow_20260314.pth` |
| 72 | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_newflow_chain_20260314/72/ckpt_last_WalkF_stage7_72_from_oldd1_newflow_20260314.pth` |
| lambda | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_newflow_chain_20260314/lambda/ckpt_last_WalkF_stage7_lambda_from_oldd1_newflow_20260314.pth` |

## Stage progress (model-source)

| stage | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg | arm | else |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage6 | 0.315735 | 60.282111 | 60.282111 | 0.315735 | 0.865450 | 0.196877 | 0.222863 | 0.135457 |
| 70a | 0.275083 | 60.282111 | 60.282111 | 0.275083 | 0.730911 | 0.176525 | 0.203549 | 0.112650 |
| 70b | 0.308443 | 60.282111 | 60.282111 | 0.308443 | 0.730643 | 0.217157 | 0.254408 | 0.129109 |
| new70b_replace | 0.280736 | 60.282111 | 60.282111 | 0.280736 | 0.662440 | 0.198205 | 0.226846 | 0.130508 |
| 70R | 0.158235 | 60.282111 | 60.282111 | 0.158235 | 0.556049 | 0.072222 | 0.082665 | 0.047537 |
| 71 | 0.111911 | 60.282111 | 60.282111 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.047537 |
| 72 | 0.112074 | 60.282111 | 60.282111 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.047537 |
| lambda | 0.112074 | 0.111467 | 0.534542 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.047537 |

| transition | d_all_ex_root | d_leg | d_nonleg | d_arm | d_else |
|---|---:|---:|---:|---:|---:|
| stage6_to_70a | -0.040652 | -0.134539 | -0.020352 | -0.019313 | -0.022808 |
| 70a_to_70b | 0.033361 | -0.000268 | 0.040632 | 0.050859 | 0.016459 |
| 70b_to_new70b_replace | -0.027708 | -0.068204 | -0.018952 | -0.027562 | 0.001399 |
| new70b_replace_to_70R | -0.122500 | -0.106391 | -0.125984 | -0.144181 | -0.082970 |
| 70R_to_71 | -0.046325 | -0.260576 | 0.000000 | 0.000000 | 0.000000 |
| 71_to_72 | 0.000163 | 0.000916 | 0.000000 | 0.000000 | 0.000000 |
| 72_to_lambda | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Final evals

| lane | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg | arm | else |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| model_source | 0.112074 | 0.111467 | 0.534542 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.047537 |
| strict_pretrain_contact | 0.111971 | 0.111156 | 0.530454 | 0.111971 | 0.293528 | 0.072716 | 0.083060 | 0.048265 |

## Final direct-path windows

| lane | section | legs_main | arms_main | left_arm_main | right_arm_main |
|---|---|---:|---:|---:|---:|
| model_source | overall | 0.296389 | 0.082665 | 0.109156 | 0.080301 |
| model_source | A_52_59 | 0.427246 | 0.071218 | 0.099679 | 0.057275 |
| model_source | B_76_80 | 0.225787 | 0.090594 | 0.139282 | 0.084738 |
| strict_pretrain_contact | overall | 0.293528 | 0.083060 | 0.110200 | 0.078684 |
| strict_pretrain_contact | A_52_59 | 0.400387 | 0.071439 | 0.103515 | 0.051629 |
| strict_pretrain_contact | B_76_80 | 0.217172 | 0.101627 | 0.150458 | 0.093875 |

| lane | foot_l_ball_l_SIC12_15 | calf_r_SIC2_4 |
|---|---:|---:|
| model_source | 0.812663 | 0.288880 |
| strict_pretrain_contact | 0.785841 | 0.272596 |

## Reference controls

| ref | contract | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg |
|---|---|---:|---:|---:|---:|---:|---:|
| accepted_final_anchor | model | 0.112947 | 0.491534 | 0.955117 | 0.112947 | 0.274360 | 0.078048 |
| full_oldplan_chain | model | 0.120145 | 0.119409 | 0.482860 | 0.120145 | 0.278087 | 0.085995 |
| full_oldplan_chain | strict | 0.117883 | 0.116863 | 0.475730 | 0.117883 | 0.280194 | 0.082789 |
| rollback_planner_core | model | 0.114635 | 0.113263 | 0.462037 | 0.114635 | 0.296311 | 0.075354 |
| rollback_planner_core | strict | 0.114862 | 0.112776 | 0.457103 | 0.114862 | 0.286085 | 0.077841 |

## Final deltas

| compare | contract | d_DirectGeoLocalDeg | d_BlendGeoLocalDeg | d_GeoLocalDeg | d_all_ex_root | d_leg | d_nonleg |
|---|---|---:|---:|---:|---:|---:|---:|
| accepted_anchor | model | -0.000873 | -0.380067 | -0.420575 | -0.000873 | 0.022029 | -0.005826 |
| full_oldplan_chain | model | -0.008071 | -0.007943 | 0.051682 | -0.008071 | 0.018302 | -0.013774 |
| rollback_planner_core | model | -0.002562 | -0.001797 | 0.072505 | -0.002562 | 0.000078 | -0.003132 |
| full_oldplan_chain | strict | -0.005912 | -0.005707 | 0.054724 | -0.005912 | 0.013334 | -0.010073 |
| rollback_planner_core | strict | -0.002891 | -0.001620 | 0.073351 | -0.002891 | 0.007443 | -0.005125 |

## Requested answers

1. Stage6 vs controls: old d1 Stage6 is `worse overall` than both full oldplan control and rollback_planner_core on strict Stage6 (vs full oldplan d_all_ex_root=0.017747, d_leg=0.133527; vs rollback d_all_ex_root=0.008029, d_leg=0.107401). Nonleg is slightly better in both comparisons.
2. Main improvement step: all_ex_root `new70b_replace_to_70R` (-0.122500); main regression step: `70a_to_70b` (0.033361). For leg, best=`70R_to_71` (-0.260576), worst=`71_to_72` (0.000916).
3. Final model-source status: vs full oldplan `mixed`, vs rollback_planner_core `mixed`, vs accepted anchor `mixed`.
4. Final strict status: vs full oldplan `mixed`, vs rollback_planner_core `mixed`; vs accepted anchor is `cross_contract_only` because the archived accepted anchor is model-source only.
5. Strict/model consistency: vs full oldplan `true`, vs rollback `true`.
6. Worth a challenger lane: `true`. Eligible for baseline/promote discussion now: `false` (Accepted anchor is archived only as model-source in repo; strict-vs-accepted remains cross-contract.).

