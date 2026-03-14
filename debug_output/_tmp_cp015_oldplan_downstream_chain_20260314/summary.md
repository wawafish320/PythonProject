# cp015 old-plan downstream chain

- run_date: 20260314
- stage6_case: `cp015_with_old_planstack`
- transplant_stage6_ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_stage6_plantransplant_20260314/stage6/cp015_with_old_planstack/ckpt_last_cp015_with_old_planstack_stage6_plantransplant_20260314.pth`
- stage6_exit: all_ex_root=0.295533, leg=0.740703, nonleg=0.199280

## Stage ckpts

| stage | ckpt |
|---|---|
| transplant_stage6 | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_stage6_plantransplant_20260314/stage6/cp015_with_old_planstack/ckpt_last_cp015_with_old_planstack_stage6_plantransplant_20260314.pth` |
| 70a | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_downstream_chain_20260314/70a/ckpt_last_WalkF_stage7_70a_from_cp015_oldplan_20260314.pth` |
| 70b | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_downstream_chain_20260314/70b/ckpt_last_WalkF_stage7_70b_concat_from_cp015_oldplan_20260314.pth` |
| 70a_replace_warmstart | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_downstream_chain_20260314/warmstart/ckpt_last_cp015_oldplan_70a_replacecontacts_zerophase_20260314.pth` |
| new70b_replace | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_downstream_chain_20260314/70b_replace/ckpt_last_WalkF_stage7_70b_replacecontacts_from_cp015_oldplan_20260314.pth` |
| 70R_promoted_s180 | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_downstream_chain_20260314/70R/ckpt_last_WalkF_stage7_70R_from_cp015_oldplan_trunkfull_s180_20260314.pth` |
| 71 | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_downstream_chain_20260314/71/ckpt_last_WalkF_stage7_71_from_cp015_oldplan_20260314.pth` |
| 72 | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_downstream_chain_20260314/72/ckpt_last_WalkF_stage7_72_from_cp015_oldplan_20260314.pth` |
| lambda_final | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_oldplan_downstream_chain_20260314/lambda/ckpt_last_WalkF_stage7_lambda_from_cp015_oldplan_20260314.pth` |

## Accepted references

| ref | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg |
|---|---:|---:|---:|---:|---:|---:|
| accepted_old_baseline_r5 | 0.147802 | 0.531568 | 1.030833 | 0.147802 | 0.313692 | 0.111934 |
| accepted_final_model_source | 0.112947 | 0.491534 | 0.955117 | 0.112947 | 0.274360 | 0.078048 |
| evalon_20260307_baseline | 0.131316 | 0.497677 | 0.961235 | 0.131316 | 0.292003 | 0.096573 |

## Final evals

| lane | DirectGeoLocalDeg | DirectGeoLocalDegWeighted | BlendGeoLocalDeg | BlendGeoLocalDegWeighted | GeoLocalDeg | GeoLocalDegWeighted | all_ex_root | leg | nonleg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strict_pretrain_contact | 0.117883 | 0.138167 | 0.116863 | 0.137015 | 0.475730 | 0.511086 | 0.117883 | 0.280194 | 0.082789 |
| model_source | 0.120145 | 0.141302 | 0.119409 | 0.140813 | 0.482860 | 0.523406 | 0.120145 | 0.278087 | 0.085995 |

## Window summary

| lane | section | legs_main | arms_main | left_arm_main | right_arm_main |
|---|---|---:|---:|---:|---:|
| strict_pretrain_contact | overall | 0.280194 | 0.093731 | 0.121068 | 0.086943 |
| strict_pretrain_contact | A_52_59 | 0.330085 | 0.078512 | 0.109917 | 0.054733 |
| strict_pretrain_contact | B_76_80 | 0.197513 | 0.107070 | 0.166757 | 0.083970 |
| model_source | overall | 0.278087 | 0.098362 | 0.128407 | 0.091691 |
| model_source | A_52_59 | 0.331869 | 0.075047 | 0.108804 | 0.055572 |
| model_source | B_76_80 | 0.195074 | 0.105474 | 0.159382 | 0.083871 |

| lane | foot_l_ball_l_SIC12_15 | calf_r_SIC2_4 |
|---|---:|---:|
| strict_pretrain_contact | 0.743282 | 0.133129 |
| model_source | 0.608819 | 0.136196 |

## Deltas

| compare | DirectGeoLocalDeg | BlendGeoLocalDeg | GeoLocalDeg | all_ex_root | leg | nonleg |
|---|---:|---:|---:|---:|---:|---:|
| model - accepted_final_model_source | 0.007198 | -0.372125 | -0.472257 | 0.007198 | 0.003727 | 0.007947 |
| model - accepted_old_baseline_r5 | -0.027657 | -0.412159 | -0.547973 | -0.027657 | -0.035605 | -0.025939 |
| strict - accepted_old_baseline_r5 | -0.029919 | -0.414705 | -0.555103 | -0.029919 | -0.033498 | -0.029145 |

## chain_verdict reference

- accepted lambda direct-path delta vs previous new chain: overall legs_main=-0.026292, arms_main=-0.103466, A arms=-0.236356, B arms=-0.115982
- accepted lambda blend delta vs previous new chain: BlendGeoLocalDeg=-0.020124, GeoLocalDeg=-0.014627, DirectGeoLocalDeg=-0.032109

## Answers

1. Advantage penetrates to lambda final: `true` (candidate still beats the accepted old baseline r5 anchor at lambda final).
2. Final beats current accepted mainline: `false` (current accepted compare artifact is model-source).
3. Strict eval still supports the carry claim: `true` (note: Repo docs explicitly note the accepted chain compare artifacts are model-source; a strict pretrain_contact+affine_mix08 accepted-final eval snapshot is not archived locally.)
4. Switch baseline now: `false`
5. Future simplification only on this chain: `false`

