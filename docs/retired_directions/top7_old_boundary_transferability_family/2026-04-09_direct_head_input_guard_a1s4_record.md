# 2026-04-09 Direct Head Input Guard A1-S4

> Archived on 2026-04-12.  
> Current role: historical old-boundary `top7` transferability record inside the archived `E0/E1/E2A/E2C/E3A/A1-S1..S5` family, not current design policy.  
> Reader guidance: any `主线` / `推荐` / `默认下一步` / `canonical` wording below is preserved as family-local historical language.

## 1. Scope / inherited conclusions

- Scope: A1-S4 direct_pose_head input merged diagnostic + moment-matching affine guard, no-train.
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
- A1-S3 inherit: 若强行二选一，只有 weak lean toward host nonleg out side；当前推荐仍偏向更早 boundary / stronger boundary guard.

## 2. Why A1-S4 after A1-S3

- A1-S3 still did not beat `A1S2-mix-nonleg`, so the next minimal lever is a stronger boundary guard at the earliest usable anchor.
- This round stays at `direct_pose_head` input only, with runtime affine guard and no learned adapter / no retraining.

## 3. Host / donor / target inventory

- Host ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/coadapt_allrot_interface_bestlr_longer_4x/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth`.
- E1 donor ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth`.
- E2A donor ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk357ramp_stage70a_from_tailfix_e2a_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk357ramp_stage6tailfix_e2a_20260408.pth`.
- Baseline replace donor ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth`.

## 4. Hook point definition

- Resolved hook: `direct_pose_head.0` (Linear).
- Meaning: `direct_pose_head` first Linear forward pre-hook input activation, not output / not weights / not trunk hidden.

## 5. Layer 0 sanity table

| left | right | l2_rms | cosine_distance |
| --- | --- | --- | --- |
| target-full7 | E1-top3-full7 | 0.000000 | 0.000000 |
| target-full7 | E2A-R-full7 | 0.000000 | 0.000000 |
| target-full7 | A1S2-mix-nonleg | 0.000000 | 0.000000 |
| E1-top3-full7 | E2A-R-full7 | 0.000000 | 0.000000 |
| E1-top3-full7 | A1S2-mix-nonleg | 0.000000 | 0.000000 |
| E2A-R-full7 | A1S2-mix-nonleg | 0.000000 | 0.000000 |

- Call: non_discriminative_or_near_zero (max l2=`0.000000`, max cosdist=`0.000000`).

## 6. Layer 1 divergence table

| pair | mean_l2_rms | mean_cosdist | max_l2_rms |
| --- | --- | --- | --- |
| target-full7__vs__E1-top3-full7 | 0.000000 | -0.000000 | 0.000000 |
| target-full7__vs__E2A-R-full7 | 0.000000 | -0.000000 | 0.000000 |
| target-full7__vs__A1S2-mix-nonleg | 0.000000 | -0.000000 | 0.000000 |
| E1-top3-full7__vs__E2A-R-full7 | 0.000000 | -0.000000 | 0.000000 |
| E1-top3-full7__vs__A1S2-mix-nonleg | 0.000000 | -0.000000 | 0.000000 |
| E2A-R-full7__vs__A1S2-mix-nonleg | 0.000000 | -0.000000 | 0.000000 |

- `A1S2-mix-nonleg` mean l2 to refs:
  - vs E1-top3-full7: `0.000000`
  - vs target-full7: `0.000000`
  - vs E2A-R-full7: `0.000000`

## 7. Step-level correlation summary

| arm | reference | l2~dir_leg | l2~dir_nonleg | cos~dir_leg | cos~dir_nonleg |
| --- | --- | --- | --- | --- | --- |
| E2A-R-full7 | E1-top3-full7 | nan | nan | 0.000000 | 0.000000 |
| E2A-R-full7 | target-full7 | nan | nan | 0.000000 | 0.000000 |
| A1S2-mix-nonleg | E1-top3-full7 | nan | nan | 0.000000 | 0.000000 |
| A1S2-mix-nonleg | target-full7 | nan | nan | 0.000000 | 0.000000 |

## 8. Moment-matching transform definition

- Base arm: `A1S2-mix-nonleg`.
- Reference stats arm: `E1-top3-full7`.
- Formula: `x_hat = (x - mu_src) / (std_src + eps) * std_ref + mu_ref`.
- Estimation rows: `12`, feature_dim=`43`, eps=`0.000001`.
- Scale summary mean=`0.901444`, p90=`0.999999`, max=`0.999999`.

## 9. Layer 2 fixed transfer assay table

| arm | out_gap | dir_base_gap | dir_leg_gap | dir_nonleg_gap | agg_score |
| --- | --- | --- | --- | --- | --- |
| target-full7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| E1-top3-full7 | 0.475508 | 0.794442 | 1.931073 | 0.548684 | 0.232733 |
| E2A-R-full7 | 0.461871 | 0.960091 | 2.021634 | 0.730569 | 0.156738 |
| A1S2-mix-nonleg | 0.468898 | 0.872376 | 1.980258 | 0.632833 | 0.196577 |
| A1S4-mm-E1ref-on-A1S2mix | 0.468898 | 0.872374 | 1.980257 | 0.632832 | 0.196578 |

- `A1S4-mm-E1ref-on-A1S2mix` minus `A1S2-mix-nonleg` aggregate = `0.000001`.

## 10. Boundary-guard interpretation

- Case: `Case D`.
- Interpretation: direct_pose_head input is not the strongest intervention lever under this assay.
- Input-side boundary guard mainline? `false`.

## 11. Next-step recommendation

- Recommended next step: head-internal/downstream work, learned adapter, or earlier boundary.
- Q1: yes.
- Q2: tie:E1-top3-full7,E2A-R-full7,target-full7.
- Q3: no.
- Q4: no.
- Q5: head-internal/downstream work, learned adapter, or earlier boundary.
