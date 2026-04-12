# 2026-04-09 mixed-contract A1-S2 record

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

> Last updated: 2026-04-09  
> Scope: A1-S2 only / fixed-host cross-donor mixed-contract transplant assay / no new training

## 1. Scope / inherited conclusions

本轮只做 **fixed host 下的 cross-donor mixed-contract transplant assay**，直接继承以下结论，不重复证明：

- root cause not in planner semantics mainline
- root cause not in replace-entry external rollout state
- root cause not in contacts_in_t
- earliest semantic split at direct_pose_head boundary
- direct_pose_head is earliest boundary / necessary anchor but not standalone sufficient
- baseline 7-module direct branch can transfer into coadapt context
- E1-top3 is the only clearly effective upstream intervention so far
- all late/full top7 variants are worse than E1-top3
- E3A-RF further argues allocation ordering is not a sufficient lever
- current normality probe is non-discriminative and not a main criterion

同时直接继承 A1-S1：
- A1-S1 summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_partial_transplant_boundary_a1s1_20260409/summary.json`
- A1-S1 record: `/Users/xingzhaorui/PycharmProjects/PythonProject/docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_partial_transplant_boundary_a1s1_record.md`
- `A1S1-anchor_only` 不比 `E2A-R full7` 更 replace-transferable；判例更像 **Case 3**。
- 主判断：**shared_head_already_compromised**。
- residual retention 更偏向 **nonleg_expansion_candidate**，因此本轮优先测 **E1-top3 anchor + top7 nonleg expansion**。

## 2. Why A1-S2 after A1-S1

- A1-S1 已经排除了“只保住 donor 自己的 shared head 就足够”的简单解释。
- 因此 A1-S2 的唯一目标，是测试 **cross-donor anchor preservation** 是否能让某一侧 top7 expansion 变得可吸收。
- 本轮不扩成 full grid，不做新训练，不把 candidate partition 写成已证实真相。

## 3. Donor / host / target inventory

| item | artifact | path / note |
|---|---|---|
| host ckpt | coadapt replace host | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/coadapt_allrot_interface_bestlr_longer_4x/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth` |
| host config | fixed host config | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/configs/posttrain_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.json` |
| anchor donor ckpt | E1-top3 final70a | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth` |
| anchor donor eval | E1-top3 eval | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/eval_model_source/Walk_F_freerun_cycles.json` |
| expansion donor ckpt | E2A-R final70a | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth` |
| expansion donor eval | E2A-R eval | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk_curriculum_e2a_20260408/stage70a/eval_model_source/Walk_F_freerun_cycles.json` |
| baseline replace ckpt | synthetic target donor | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth` |
| target | transplant-compatible target | in-memory: fixed host + baseline replace full7 transplant |

## 4. Candidate partition reminder

下述 partition **仍然只是 hypothesis**，仅用于 A1-S2 assay inventory：

| family | modules | parameter prefixes | note |
|---|---|---|---|
| `anchor_candidate` | `direct_pose_head` | `direct_pose_head.` | code-level assay hypothesis only |
| `leg_expansion_candidate` | `direct_pose_leg_head, direct_pose_out_leg` | `direct_pose_leg_head., direct_pose_out_leg.` | code-level assay hypothesis only |
| `nonleg_expansion_candidate` | `direct_pose_arm_proj, direct_pose_else_proj, direct_pose_out_arm, direct_pose_out_else` | `direct_pose_arm_proj., direct_pose_else_proj., direct_pose_out_arm., direct_pose_out_else.` | code-level assay hypothesis only |

## 5. Mixed-contract assay inventory table

| arm | E1-top3 modules | E2A-R modules | copied key counts |
|---|---|---|---:|
| `A1S2-mix-nonleg` | `direct_pose_head, direct_pose_leg_head, direct_pose_out_leg` | `direct_pose_arm_proj, direct_pose_else_proj, direct_pose_out_arm, direct_pose_out_else` | 20 |
| `A1S2-mix-leg` | `direct_pose_head, direct_pose_arm_proj, direct_pose_else_proj, direct_pose_out_arm, direct_pose_out_else` | `direct_pose_leg_head, direct_pose_out_leg` | 20 |

## 6. Fixed transfer assay table

| arm | out_direct gap | dir_base gap | dir_leg gap | dir_nonleg gap | out closure | dir_base closure | dir_leg closure | dir_nonleg closure | aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `host-native bad reference` | 0.469847 | 1.293004 | 2.010747 | 1.137816 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `transplant-compatible target` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `E1-top3 full7` | 0.475508 | 0.794442 | 1.931073 | 0.548684 | -0.012050 | 0.385584 | 0.039624 | 0.517774 | 0.232733 |
| `E2A-R full7` | 0.461871 | 0.960091 | 2.021634 | 0.730569 | 0.016976 | 0.257472 | -0.005414 | 0.357920 | 0.156738 |
| `A1S2-mix-nonleg` | 0.468898 | 0.872376 | 1.980258 | 0.632833 | 0.002019 | 0.325311 | 0.015163 | 0.443817 | 0.196577 |
| `A1S2-mix-leg` | 0.477441 | 0.823071 | 2.092113 | 0.548684 | -0.016164 | 0.363442 | -0.040465 | 0.517774 | 0.206147 |

## 7. `dir_leg` retention interpretation

- `A1S2-mix-nonleg` 的 `dir_leg` closure = `0.015163`，相对 `E2A-R full7` delta = `0.020577`。
- `A1S2-mix-leg` 的 `dir_leg` closure = `-0.040465`，相对 `E2A-R full7` delta = `-0.035051`。
- `A1S2-mix-leg` vs `A1S2-mix-nonleg` 的 `dir_leg` closure delta = `-0.055628`。

## 8. `dir_nonleg` retention interpretation

- `A1S2-mix-nonleg` 的 `dir_nonleg` closure = `0.443817`，相对 `E2A-R full7` delta = `0.085897`。
- `A1S2-mix-nonleg` 相对 `E1-top3 full7` 的 `dir_nonleg` closure delta = `-0.073957`。
- `A1S2-mix-leg` 的 `dir_nonleg` closure = `0.517774`，与 `A1S2-mix-nonleg` 的差值 = `-0.073957`。
- aggregate 上 `A1S2-mix-leg` 只比 `A1S2-mix-nonleg` 高 `0.009570`，仍低于显式 side-preference margin。

## 9. Mixed-contract interpretation

- `Case A/B` 判读：**between_Case_A_and_Case_B_lean_A**
- `Case C/D` 判读：**no_clear_C_or_D**
- `A1S2-mix-nonleg` vs `E2A-R full7` aggregate delta = `0.039839`。
- `A1S2-mix-nonleg` vs `E1-top3 full7` aggregate delta = `-0.036156`。
- preserved anchor 下，top7 nonleg 更像：**top7_nonleg_partially_absorbable_but_not_yet_decisive**。
- side follow-up 结论：**no_clear_winner**；aggregate leader = **A1S2-mix-leg**；如果必须继续吸收一侧，优先保持 **A1S2-mix-nonleg**。
- 原因：aggregate edge stays below the explicit preference margin; mix-leg is slightly higher on aggregate, but mix-nonleg is the cleaner absorb-side because it improves both dir_nonleg and dir_leg over E2A-R。

## 10. Next-step recommendation

- 是否更明确转向 replace-side absorb-expansion：**no**
- 推荐主线：**nonleg_absorb_expansion_only_with_stronger_replace_side_absorb_or_boundary_guard**
- 说明：mix-nonleg shows partial rescue and sits closer to E1-top3 than to E2A-R, but the gain is still below the clear-win margin; do not treat plain mixed transplant as sufficient

## Final answers

- `q1_mix_nonleg_clearly_better_than_E2A_R_full7`: `{'answer': 'no', 'aggregate_delta': 0.03983898767367902, 'dir_nonleg_closure_delta': 0.08589733287332191, 'dir_leg_closure_delta': 0.020577009999426132}`
- `q2_mix_nonleg_close_to_E1_top3_full7`: `{'answer': 'yes', 'aggregate_delta': -0.03615576383638913, 'dir_nonleg_closure_delta': -0.07395676821805963}`
- `q3_under_preserved_anchor_is_top7_nonleg_absorbable`: `top7_nonleg_partially_absorbable_but_not_yet_decisive`
- `q4_preferred_followup_side`: `{'answer': 'no_clear_winner', 'aggregate_leader': 'A1S2-mix-leg', 'absorb_side_priority_if_forced': 'A1S2-mix-nonleg', 'note': 'aggregate edge stays below the explicit preference margin; mix-leg is slightly higher on aggregate, but mix-nonleg is the cleaner absorb-side because it improves both dir_nonleg and dir_leg over E2A-R'}`
- `q5_next_step_should_shift_to_replace_side_absorb_expansion_or_boundary_redesign`: `{'recommend_replace_side_absorb_expansion': False, 'next_step': 'nonleg_absorb_expansion_only_with_stronger_replace_side_absorb_or_boundary_guard', 'note': 'mix-nonleg shows partial rescue and sits closer to E1-top3 than to E2A-R, but the gain is still below the clear-win margin; do not treat plain mixed transplant as sufficient'}`
