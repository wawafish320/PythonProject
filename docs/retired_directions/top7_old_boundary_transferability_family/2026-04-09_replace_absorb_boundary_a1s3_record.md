# 2026-04-09 replace absorb boundary A1-S3 record

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

> Last updated: 2026-04-09  
> Scope: A1-S3 only / fixed-host replace-side nonleg absorb boundary assay / tri-donor / no new training

## 1. Scope / inherited conclusions

本轮只做 **fixed host 下的 replace-side nonleg absorb boundary assay**，直接继承以下结论，不重复证明：

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

直接继承 A1-S1：
- A1-S1 summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_partial_transplant_boundary_a1s1_20260409/summary.json`
- A1-S1 record: `/Users/xingzhaorui/PycharmProjects/PythonProject/docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_partial_transplant_boundary_a1s1_record.md`
- A1S1-anchor_only 不比 E2A-R full7 更 replace-transferable
- A1-S1 更像 Case 3
- 更像 shared head 本身 already compromised
- partial add-back 里 anchor_plus_nonleg residual retention 明显好于 anchor_plus_leg

直接继承 A1-S2：
- A1-S2 summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_mixed_contract_a1s2_20260409/summary.json`
- A1-S2 record: `/Users/xingzhaorui/PycharmProjects/PythonProject/docs/retired_directions/top7_old_boundary_transferability_family/2026-04-09_mixed_contract_a1s2_record.md`
- A1S2-mix-nonleg 比 E2A-R full7 更好但不够 clear-win
- A1S2-mix-nonleg aggregate 上接近 E1-top3 full7
- A1S2-mix-nonleg 同时改善了相对 E2A-R full7 的 dir_nonleg 和 dir_leg
- A1S2-mix-leg aggregate 略高，但 dir_leg 更差
- top7 nonleg 更像 partially absorbable but not yet decisive
- plain mixed transplant 还不足以支持直接进入 replace-side absorb-expansion 已经 solved

## 2. Why A1-S3 after A1-S2

- A1-S2 已经把 preserved-anchor mixed transplant 推到 `A1S2-mix-nonleg`，但仍未 clear-win。
- 因此 A1-S3 的唯一目标，是继续把 nonleg block 只拆成 `proj side` / `out side` 两个 code-level assay split，观察 fixed host 哪一侧更像 absorb boundary。
- 本轮不做 full grid，不开训练，也不把 candidate partition 写成已证实真相。

## 3. Donor / host / target inventory

| item | artifact | path / note |
|---|---|---|
| host ckpt | fixed host | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/coadapt_allrot_interface_bestlr_longer_4x/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth` |
| host config | fixed host config | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/configs/posttrain_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.json` |
| anchor donor ckpt | E1-top3 final70a | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth` |
| anchor donor eval | E1-top3 eval | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/eval_model_source/Walk_F_freerun_cycles.json` |
| expansion donor ckpt | E2A-R final70a | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth` |
| expansion donor eval | E2A-R eval | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk_curriculum_e2a_20260408/stage70a/eval_model_source/Walk_F_freerun_cycles.json` |
| baseline replace ckpt | transplant target donor | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth` |
| target | transplant-compatible target | in-memory only: fixed host + baseline replace full7 transplant |

## 4. Candidate partition reminder

下述 partition **仍然只是 hypothesis**，仅用于 A1-S3 assay inventory：

| family | modules | parameter prefixes | note |
|---|---|---|---|
| `preserved_anchor_leg_block` | `direct_pose_head, direct_pose_leg_head, direct_pose_out_leg` | `direct_pose_head., direct_pose_leg_head., direct_pose_out_leg.` | code-level assay hypothesis only |
| `nonleg_proj_candidate` | `direct_pose_arm_proj, direct_pose_else_proj` | `direct_pose_arm_proj., direct_pose_else_proj.` | code-level assay hypothesis only |
| `nonleg_out_candidate` | `direct_pose_out_arm, direct_pose_out_else` | `direct_pose_out_arm., direct_pose_out_else.` | code-level assay hypothesis only |

## 5. Tri-donor assay inventory table

| arm | E1-top3 modules | E2A-R modules | fixed-host retained modules | copied key counts |
|---|---|---|---|---:|
| `A1S3-nonleg-proj-donor_host-out` | `direct_pose_head, direct_pose_leg_head, direct_pose_out_leg` | `direct_pose_arm_proj, direct_pose_else_proj` | `direct_pose_out_arm, direct_pose_out_else` | 16 |
| `A1S3-nonleg-out-donor_host-proj` | `direct_pose_head, direct_pose_leg_head, direct_pose_out_leg` | `direct_pose_out_arm, direct_pose_out_else` | `direct_pose_arm_proj, direct_pose_else_proj` | 16 |

## 6. Fixed transfer assay table

| arm | out_direct gap | dir_base gap | dir_leg gap | dir_nonleg gap | out closure | dir_base closure | dir_leg closure | dir_nonleg closure | aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `host-native bad reference` | 0.469847 | 1.293004 | 2.010747 | 1.137816 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `transplant-compatible target` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `E1-top3 full7` | 0.475508 | 0.794442 | 1.931073 | 0.548684 | -0.012050 | 0.385584 | 0.039624 | 0.517774 | 0.232733 |
| `E2A-R full7` | 0.461871 | 0.960091 | 2.021634 | 0.730569 | 0.016976 | 0.257472 | -0.005414 | 0.357920 | 0.156738 |
| `A1S2-mix-nonleg` | 0.468898 | 0.872376 | 1.980258 | 0.632833 | 0.002019 | 0.325311 | 0.015163 | 0.443817 | 0.196577 |
| `A1S2-mix-leg` | 0.477441 | 0.823071 | 2.092113 | 0.548684 | -0.016164 | 0.363442 | -0.040465 | 0.517774 | 0.206147 |
| `A1S3-nonleg-proj-donor_host-out` | 0.478897 | 0.927958 | 1.980258 | 0.700433 | -0.019262 | 0.282324 | 0.015163 | 0.384405 | 0.165658 |
| `A1S3-nonleg-out-donor_host-proj` | 0.477357 | 1.329335 | 2.044690 | 1.174664 | -0.015985 | -0.028098 | -0.016881 | -0.032385 | -0.023337 |

## 7. `dir_leg` retention interpretation

- `A1S3-nonleg-proj-donor_host-out` 的 `dir_leg` closure = `0.015163`；相对 `A1S2-mix-nonleg` delta = `0.000000`。
- `A1S3-nonleg-out-donor_host-proj` 的 `dir_leg` closure = `-0.016881`；相对 `A1S2-mix-nonleg` delta = `-0.032044`。
- 两个 A1-S3 arms 的 `dir_leg` closure 差值（proj-host-out 减 out-host-proj）= `0.032044`。

## 8. `dir_nonleg` retention interpretation

- `A1S3-nonleg-proj-donor_host-out` 的 `dir_nonleg` closure = `0.384405`；相对 `A1S2-mix-nonleg` delta = `-0.059412`。
- `A1S3-nonleg-out-donor_host-proj` 的 `dir_nonleg` closure = `-0.032385`；相对 `A1S2-mix-nonleg` delta = `-0.476202`。
- 两个 A1-S3 arms 的 `dir_nonleg` closure 差值（proj-host-out 减 out-host-proj）= `0.416790`。

## 9. Replace-side absorb boundary interpretation

- A1-S3 判例：**Case C**
- `A1S3-nonleg-proj-donor_host-out` vs `A1S2-mix-nonleg` aggregate delta = `-0.030920`。
- `A1S3-nonleg-out-donor_host-proj` vs `A1S2-mix-nonleg` aggregate delta = `-0.219915`。
- host absorb boundary call：**no_clear_winner; weak lean host nonleg out side**。
- main incompatibility boundary call：**no_clear_single_boundary; weak lean downstream nonleg readout contract**。
- 解释：neither split arm clears the explicit improvement margin over A1S2-mix-nonleg, so plain replace-side splitting is still not decisive

## 10. Next-step recommendation

- 是否支持进入更明确的 replace-side absorb-expansion design：**no**
- 推荐主线：**shrink_back_to_earlier_boundary_or_stronger_boundary_guard**

## Final answers

- `A1S3-nonleg-proj-donor_host-out` 是否明显优于 `A1S2-mix-nonleg`：`no`
- `A1S3-nonleg-out-donor_host-proj` 是否明显优于 `A1S2-mix-nonleg`：`no`
- host absorb capacity 更像落在哪一侧：`no_clear_winner; weak lean host nonleg out side`
- 是否支持进入更明确的 replace-side absorb-expansion design：`no`
- 是否仍应先转向更早 boundary / stronger boundary guard：`yes`
