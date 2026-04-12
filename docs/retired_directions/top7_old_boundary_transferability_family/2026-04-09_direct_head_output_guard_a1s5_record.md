# 2026-04-09 direct-head-output guard A1-S5 record

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

## 1. Scope / inherited conclusions

- Scope: A1-S5 direct_pose_head output / nonleg consumer-entry merged diagnostic + runtime affine guard, no-train.
- Host assay remains fixed to `coadapt_allrot_interface_bestlr_longer_4x_20260406`, offset `45`, same entry, same teacher clip.
- Fixed first-forward contacts remain baseline replace native same-entry `contacts_in_t`.
- Inherited mainline: root cause not in planner semantics mainline.
- Inherited mainline: root cause not in replace-entry external rollout state.
- Inherited mainline: root cause not in contacts_in_t.
- Inherited mainline: earliest semantic split at direct_pose_head boundary.
- Inherited mainline: direct_pose_head is earliest boundary / necessary anchor but not standalone sufficient.
- Inherited mainline: baseline 7-module direct branch can transfer into coadapt context.
- Inherited mainline: E1-top3 is the only clearly effective upstream intervention so far.
- Inherited mainline: all late/full top7 variants are worse than E1-top3.
- Inherited mainline: E3A-RF further argues allocation ordering is not a sufficient lever.
- Inherited mainline: current normality probe is non-discriminative and not a main criterion.
- A1-S1 inherit: A1S1-anchor_only 不比 E2A-R full7 更 replace-transferable.
- A1-S1 inherit: A1-S1 更像 Case 3.
- A1-S1 inherit: 更像 shared head 本身 already compromised.
- A1-S1 inherit: anchor_plus_nonleg residual retention 明显好于 anchor_plus_leg.
- A1-S2 inherit: A1S2-mix-nonleg 比 E2A-R full7 更好但不够 clear-win.
- A1-S2 inherit: A1S2-mix-nonleg aggregate 上接近 E1-top3 full7.
- A1-S2 inherit: A1S2-mix-nonleg 同时改善了相对 E2A-R full7 的 dir_nonleg 和 dir_leg.
- A1-S2 inherit: plain mixed transplant 还不足以支持 replace-side absorb-expansion 已 solved.
- A1-S3 inherit: 两个 A1-S3 split arms 都不优于 A1S2-mix-nonleg.
- A1-S3 inherit: plain replace-side split 仍不足以成为 decisive absorb 路线.
- A1-S3 inherit: 若强行二选一，只有 weak lean toward host nonleg out side.
- A1-S3 inherit: 当前推荐仍偏向更早 boundary / stronger boundary guard.
- A1-S4 inherit: direct_pose_head input pairwise divergence = 0.
- A1-S4 inherit: same host + weight transplant only 口径下，所有 arms 在 direct_pose_head.0 input 收到完全相同的 activation.
- A1-S4 inherit: A1-S4 moment-matching input guard 只有数值噪声级别效果，不是有效 lever.
- A1-S4 inherit: 因此 bottleneck 不在 head 上游，而更像在 head 内部 / head output → downstream expansion contract.

## 2. Why A1-S5 after A1-S4

- A1-S4 already showed `direct_pose_head` input pairwise divergence = 0, so upstream input-side guard is not the lever.
- This round therefore moves one boundary later: shared head output for diagnosis, nonleg consumer entry for intervention.

## 3. Host / donor / target inventory

- Host ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/coadapt_allrot_interface_bestlr_longer_4x/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth`.
- E1 donor ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth`.
- E2A donor ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth`.
- Baseline replace donor ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth`.

## 4. Diagnostic hook definition

- Module: `direct_pose_head` (Sequential).
- Meaning: `direct_pose_head forward output activation (shared trunk hidden before split consumers)`.
- Why here: A1-S4 already ruled out head input; this hook captures the earliest downstream-visible shared hidden that feeds leg/nonleg expansion..

## 5. Intervention hook definition

- `direct_pose_arm_proj` (Sequential) — preferred arm nonleg consumer entry pre-hook.
- `direct_pose_else_proj` (Sequential) — preferred else nonleg consumer entry pre-hook.
- Boundary rationale: intervention stays on nonleg downstream consumer entry so shared leg side is preserved as much as possible.

## 6. Layer 0 sanity table

| left | right | l2_rms | cosine_distance |
| --- | --- | --- | --- |
| target-full7 | E1-top3-full7 | 0.329023 | 0.788794 |
| target-full7 | E2A-R-full7 | 0.332148 | 0.778403 |
| target-full7 | A1S2-mix-nonleg | 0.329023 | 0.788794 |
| E1-top3-full7 | E2A-R-full7 | 0.061945 | 0.027799 |
| E1-top3-full7 | A1S2-mix-nonleg | 0.000000 | 0.000000 |
| E2A-R-full7 | A1S2-mix-nonleg | 0.061945 | 0.027799 |

- Call: `not_trivially_zero` (max l2=`0.332148`, max cosdist=`0.788794`).

## 7. Layer 1 divergence table

| tap | pair | mean_l2_rms | mean_cosdist | max_l2_rms |
| --- | --- | --- | --- | --- |
| head_output | target-full7__vs__E1-top3-full7 | 0.337812 | 0.777750 | 0.368836 |
| head_output | target-full7__vs__E2A-R-full7 | 0.339867 | 0.774620 | 0.374241 |
| head_output | target-full7__vs__A1S2-mix-nonleg | 0.337812 | 0.777750 | 0.368836 |
| head_output | E1-top3-full7__vs__E2A-R-full7 | 0.076337 | 0.036354 | 0.088658 |
| head_output | E1-top3-full7__vs__A1S2-mix-nonleg | 0.000000 | 0.000000 | 0.000000 |
| head_output | E2A-R-full7__vs__A1S2-mix-nonleg | 0.076337 | 0.036354 | 0.088658 |
| nonleg_consumer_entry | target-full7__vs__E1-top3-full7 | 0.337812 | 0.777750 | 0.368836 |
| nonleg_consumer_entry | target-full7__vs__E2A-R-full7 | 0.339867 | 0.774620 | 0.374241 |
| nonleg_consumer_entry | target-full7__vs__A1S2-mix-nonleg | 0.337812 | 0.777750 | 0.368836 |
| nonleg_consumer_entry | E1-top3-full7__vs__E2A-R-full7 | 0.076337 | 0.036354 | 0.088658 |
| nonleg_consumer_entry | E1-top3-full7__vs__A1S2-mix-nonleg | 0.000000 | 0.000000 | 0.000000 |
| nonleg_consumer_entry | E2A-R-full7__vs__A1S2-mix-nonleg | 0.076337 | 0.036354 | 0.088658 |

- `A1S2-mix-nonleg` mean l2 to refs by tap:
  - head_output: E1=`0.000000`, E2A=`0.076337`, target=`0.337812`
  - nonleg_consumer_entry: E1=`0.000000`, E2A=`0.076337`, target=`0.337812`

## 8. Step-level correlation summary

| tap | reference | l2~dir_leg | l2~dir_nonleg | cos~dir_leg | cos~dir_nonleg |
| --- | --- | --- | --- | --- | --- |
| head_output | E1-top3-full7 | nan | nan | nan | nan |
| head_output | E2A-R-full7 | nan | nan | nan | nan |
| head_output | target-full7 | nan | nan | nan | nan |
| nonleg_consumer_entry | E1-top3-full7 | nan | nan | nan | nan |
| nonleg_consumer_entry | E2A-R-full7 | nan | nan | nan | nan |
| nonleg_consumer_entry | target-full7 | nan | nan | nan | nan |

## 9. Affine transform definition

- Guard arm: `A1S5-mm-E2Aref-on-A1S2mix-nonleg-consumer`.
- Base arm: `A1S2-mix-nonleg`.
- Reference arm: `E2A-R-full7`.
- Formula: `x_hat = (x - mu_src) / (std_src + eps) * std_ref + mu_ref`.
- Eps: `0.000001`.
- `direct_pose_arm_proj` rows=12, feat_dim=512, scale_mean=`2514.821452`.
- `direct_pose_else_proj` rows=12, feat_dim=512, scale_mean=`2514.821452`.

## 10. Layer 2 fixed transfer assay table

| arm | out_gap | dir_base_gap | dir_leg_gap | dir_nonleg_gap | agg_score |
| --- | --- | --- | --- | --- | --- |
| target-full7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| E1-top3-full7 | 0.475508 | 0.794442 | 1.931073 | 0.548684 | 0.232733 |
| E2A-R-full7 | 0.461871 | 0.960091 | 2.021634 | 0.730569 | 0.156738 |
| A1S2-mix-nonleg | 0.468898 | 0.872376 | 1.980258 | 0.632833 | 0.196577 |
| A1S5-mm-E2Aref-on-A1S2mix-nonleg-consumer | 0.468047 | 0.916970 | 1.992299 | 0.684466 | 0.175566 |

- Guard minus `A1S2-mix-nonleg` aggregate = `-0.021011`.

## 11. Boundary interpretation

- Case: `Case D`.
- Interpretation: this boundary is not strongly discriminative under the current assay, and the affine guard is not useful.
- Spillover: leg-side spillover stays limited under the consumer-only guard.

## 12. Next-step recommendation

- Recommended next step: shift effort to more-downstream contract work or training-side recipe constraints.
- Q1: yes.
- Q2: {'head_output': 'E1-top3-full7', 'nonleg_consumer_entry': 'E1-top3-full7'}.
- Q3: no.
- Q4: no.
- Q5: shift effort to more-downstream contract work or training-side recipe constraints.
