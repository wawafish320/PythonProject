# 2026-04-09 partial-transplant boundary A1-S1 record

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

> Last updated: 2026-04-09  
> Scope: A1-S1 only / fixed-host partial-transplant boundary coarse scout / no new training

## 1. Scope / inherited conclusions

本轮只做 **fixed host 下的 partial-transplant boundary assay**，直接继承以下结论，不重复证明：

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

同时明确声明：下述 `anchor_candidate / leg_expansion_candidate / nonleg_expansion_candidate` **只是 code-level candidate partition**，
不是已经被证明的真实语义边界；A1-S1 的目标只是先看这个 partition 是否提供信息增益。

## 2. Why A1-S1 before full A1

- 当前 `anchor` 仍是结构假设，不是已证实的 clean semantic partition。
- 先做 single-donor coarse scout，可以避免把 donor quality、anchor 假设、cross-donor mixing、boundary definition 一次性混在同一个大 grid 里。
- 因此本轮只测一个 donor、三个 partial assays，不扩成 full boundary sweep。

## 3. Donor / host / target inventory

| item | artifact | path / note |
|---|---|---|
| donor | E2A-R final70a ckpt | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth` |
| donor eval | fixed eval artifact | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk_curriculum_e2a_20260408/stage70a/eval_model_source/Walk_F_freerun_cycles.json` |
| donor config | stage70a config | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json` |
| host | coadapt replace host ckpt | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/coadapt_allrot_interface_bestlr_longer_4x/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth` |
| host config | fixed host config | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/configs/posttrain_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.json` |
| baseline donor | baseline replace ckpt | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth` |
| target | synthetic transplant-compatible target | in-memory: fixed host + baseline replace full7 transplant |

## 4. Candidate partition table

| family | modules | parameter prefixes | note |
|---|---|---|---|
| `anchor_candidate` | `direct_pose_head` | `direct_pose_head.` | code-level assay hypothesis only |
| `leg_expansion_candidate` | `direct_pose_leg_head, direct_pose_out_leg` | `direct_pose_leg_head., direct_pose_out_leg.` | code-level assay hypothesis only |
| `nonleg_expansion_candidate` | `direct_pose_arm_proj, direct_pose_else_proj, direct_pose_out_arm, direct_pose_out_else` | `direct_pose_arm_proj., direct_pose_else_proj., direct_pose_out_arm., direct_pose_out_else.` | code-level assay hypothesis only |

## 5. Assay inventory table

| assay | transplanted modules | parameter prefixes |
|---|---|---|
| `A1S1-anchor_only` | `direct_pose_head` | `direct_pose_head.` |
| `A1S1-anchor_plus_leg` | `direct_pose_head, direct_pose_leg_head, direct_pose_out_leg` | `direct_pose_head., direct_pose_leg_head., direct_pose_out_leg.` |
| `A1S1-anchor_plus_nonleg` | `direct_pose_head, direct_pose_arm_proj, direct_pose_else_proj, direct_pose_out_arm, direct_pose_out_else` | `direct_pose_head., direct_pose_arm_proj., direct_pose_else_proj., direct_pose_out_arm., direct_pose_out_else.` |

## 6. Fixed transfer assay table

| arm | out_direct gap | dir_base gap | dir_leg gap | dir_nonleg gap | out closure | dir_base closure | dir_leg closure | dir_nonleg closure | aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `host-native bad reference` | 0.469847 | 1.293004 | 2.010747 | 1.137816 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `transplant-compatible target` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `E1-top3 full7` | 0.475508 | 0.794442 | 1.931073 | 0.548684 | -0.012050 | 0.385584 | 0.039624 | 0.517774 | 0.232733 |
| `E2A-R full7` | 0.461871 | 0.960091 | 2.021634 | 0.730569 | 0.016976 | 0.257472 | -0.005414 | 0.357920 | 0.156738 |
| `A1S1-anchor_only` | 0.474995 | 1.426229 | 2.210307 | 1.256698 | -0.010958 | -0.103035 | -0.099247 | -0.104483 | -0.079431 |
| `A1S1-anchor_plus_leg` | 0.471948 | 1.412087 | 2.130759 | 1.256698 | -0.004472 | -0.092098 | -0.059686 | -0.104483 | -0.065185 |
| `A1S1-anchor_plus_nonleg` | 0.464984 | 0.984842 | 2.160856 | 0.730569 | 0.010349 | 0.238330 | -0.074653 | 0.357920 | 0.132987 |

## 7. `dir_leg`-focused interpretation

- `A1S1-anchor_only` dir_leg closure = `-0.099247`.
- `A1S1-anchor_plus_leg` dir_leg closure = `-0.059686`; delta vs anchor_only = `0.039561`.
- `A1S1-anchor_plus_nonleg` dir_leg closure = `-0.074653`; delta vs anchor_only = `0.024594`.
- 因此本轮把 `dir_leg` 的主要恶化边界读成：**earlier shared-head boundary**。

## 8. `dir_base` / `dir_nonleg` retention summary

- `anchor_only` 对 `dir_base` 的 closure = `-0.103035`，对 `dir_nonleg` 的 closure = `-0.104483`。
- `anchor_plus_leg` aggregate delta vs anchor_only = `0.014246`。
- `anchor_plus_nonleg` aggregate delta vs anchor_only = `0.212417`。
- `anchor_plus_nonleg` 的 `dir_nonleg` closure = `0.357920`，与 `E2A-R full7` 的 `dir_nonleg` closure 完全持平；aggregate 也接近 `E2A-R full7` (`0.132987` vs `0.156738`)。
- `anchor_plus_leg` 只比 `anchor_only` 小幅改善 aggregate (`0.014246`)，信息增益明显弱于 nonleg 侧。
- 这能帮助区分：是 shared head 自身已经坏掉，还是某一侧 expansion 进入后打破了原本还能工作的 contract。

## 9. Boundary interpretation

- 判例归类：**Case 3**
- 主判断：**shared_head_already_compromised**
- 解释：anchor_only does not clearly recover over the donor full7 reference, so the more likely picture is that the shared head itself is already compromised inside this donor; expansion mixing may still matter, but the first usable preservation target should move to cross-donor anchor preservation.
- 口径克制：这仍然只是 single-donor coarse scout，不能把 candidate partition 直接升级成已证实真相。

## 10. Whether this supports A1-S2

- 是否建议进入 `A1-S2`：**yes**
- 如果进入 `A1-S2`，优先测：**E1-top3 anchor + top7 nonleg expansion**
- 选择理由：this side retains more residual transfer once added back on top of the shared head in the single-donor scout；本轮 residual 更像保留在 **nonleg_expansion_candidate**。

## Final answers

- `q1_anchor_only_more_replace_transferable_than_E2A_R_full7`: `{'answer': 'no', 'aggregate_delta': -0.2361692107450429, 'dir_leg_closure_delta': -0.09383254103426197}`
- `q2_main_break_source`: `shared_head_already_compromised`
- `q3_dir_leg_worsening_boundary`: `earlier shared-head boundary`
- `q4_enter_A1_S2`: `yes`
- `q5_preferred_A1_S2`: `E1-top3 anchor + top7 nonleg expansion`
- `anchor_plus_leg_vs_anchor_aggregate_delta`: `0.014246020741896126`
- `anchor_plus_nonleg_vs_anchor_aggregate_delta`: `0.2124174207486`
