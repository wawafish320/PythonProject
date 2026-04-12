# 2026-04-09 notail basetrain replace falsifier

> Archived on 2026-04-12.  
> Current role: historical falsifier showing `notail` did not rescue the old-boundary replace gap; it should not be read as an active mainline branch.  
> Reader guidance: `E1-top3` in this file is the legacy old-boundary-compatible anchor/control, not the current global canonical chain.

## 1. Scope / inherited conclusions

- Scope: one new matched no-tail control lane only; no sweep, no replay, no L2-SP, no new regularizer.
- Clarification update: this record now includes the real single-case `new70b_replace_lowdrift` stage, not only the fixed-host transplant assay.
- Fixed host / replace assay: `coadapt_allrot_interface_bestlr_longer_4x_20260406`, offset `45`, same teacher / same entry, fixed first-forward contacts from baseline replace native same-entry `contacts_in_t`.
- Inherited: within the matched families compared at that time, `E1-top3` remained the strongest upstream donor.
- Inherited: E2A-R improves over late/full top7 but still stays below E1-top3 on fixed-host replace transfer.
- Inherited: A1 boundary-side probes did not produce a decisive replace-side rescue over E1-top3; in the 2026-04-09 frame, the training-side recipe still remained an unresolved suspect.
- Inherited: Current direct-head output probe is supportive evidence only, not a standalone success criterion.

## 2. Why this falsifier now

- A1 boundary-side work still does not beat `E1-top3`; this round tests the stricter training-side claim that `tail-k from basetrain start` is itself a major donor-drift driver.
- The lane is matched to `E1-top3` except for disabling the active tail selection by `k=0`; rankmix/select knobs stay unchanged but become inert.

## 3. Exact no-tail config diff

| field | E1-top3 | no-tail | effect |
| --- | --- | --- | --- |
| rot_local_tail_k | 3 | 0 | top-level tail selection disabled by k=0 |
| freerun_stage_schedule.phase_b.core.rot_local_tail_k | 3 | 0 | phase-B tail selection disabled by k=0 |
| freerun_stage_schedule.phase_c.core.rot_local_tail_k | 3 | 0 | phase-C tail selection disabled by k=0 |
| freerun_stage_schedule.phase_d.core.rot_local_tail_k | 3 | 0 | phase-D tail selection disabled by k=0 |

- Actual active tail mechanism in `E1-top3`: phase-B/C/D set `rot_local_tail_weight` to `0.1/0.2` while `rot_local_tail_k=3`; `rot_local_tail_select=ema` and `rot_local_tail_reduce=rank_linear_mix` only shape that active tail term.
- This no-tail lane keeps those weights/select/reduce knobs fixed but sets `rot_local_tail_k=0` at top level and in phase-B/C/D, so `train/models.py:5543` short-circuits the tail term without introducing a second recipe change.

## 4. Basetrain / stage6 / final70a / actual new70b chain

| stage | config | ckpt | native summary |
| --- | --- | --- | --- |
| basetrain | /Users/xingzhaorui/PycharmProjects/PythonProject/config/exp_phase_DirectBranch_v1_d1_cp015_notail_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260409.json | /Users/xingzhaorui/PycharmProjects/PythonProject/models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_notail_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260409/ckpt_epoch_014.pth | /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_notail_basetrain_replace_falsifier_20260409/layer0/basetrain_epoch014_group_summary.json |
| stage6 tailfix | /Users/xingzhaorui/PycharmProjects/PythonProject/config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_tailfix_lr3e4_e8x60_wd1e4_reinit1_20260401.json | /Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_notail_stage6_tailfix_20260409/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_notail_20260409.pth | /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_notail_basetrain_replace_falsifier_20260409/stage6_tailfix/stage6_group_summary.json |
| final70a | /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json | /Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_notail_stage70a_from_tailfix_20260409/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_notail_stage6tailfix_20260409.pth | /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_notail_basetrain_replace_falsifier_20260409/stage70a/eval_model_source_group_summary.json |
| new70b_replace_lowdrift | /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_notail_basetrain_replace_falsifier_20260409/configs/posttrain_70b_replace_lowdrift_lr5e5_from_cp015_notail_70a_20260409.json | /Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_notail_replace70b_lowdrift_20260409/lr5e5/ckpt_last_WalkF_stage7_70b_replace_lowdrift_lr5e5_from_cp015_notail_70a_20260409.pth | /Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_notail_basetrain_replace_falsifier_20260409/replace70b_lowdrift/eval_model_source/lr5e5_group_summary.json |

## 5. Layer 0 native table

| case | dir_base(all_ex_root) | dir_leg | dir_nonleg | arm | else |
| --- | --- | --- | --- | --- | --- |
| notail epoch014 | 6.193658 | 10.891782 | 5.177847 | 6.525791 | 1.991798 |
| E1-top3 epoch014 | 6.224154 | 10.887709 | 5.215818 | 6.577329 | 1.997702 |
| baseline stage6 | 0.390628 | 1.042457 | 0.249692 | 0.284303 | 0.167884 |

- `notail - E1-top3` = dir_base `-0.030497`, dir_leg `0.004073`, dir_nonleg `-0.037971`.
- `notail - baseline_stage6` = dir_base `5.803030`, dir_leg `9.849325`, dir_nonleg `4.928155`.

## 6. Layer 1 final native table

| case | dir_base(all_ex_root) | dir_leg | dir_nonleg | arm | else |
| --- | --- | --- | --- | --- | --- |
| notail final70a | 0.248659 | 0.653602 | 0.161104 | 0.181944 | 0.111845 |
| E1-top3 final70a | 0.412148 | 1.058912 | 0.272307 | 0.312327 | 0.177715 |
| E2A-R final70a | 0.400640 | 1.069717 | 0.255975 | 0.289740 | 0.176167 |
| baseline 70a | 0.297430 | 0.762063 | 0.196968 | 0.227739 | 0.124236 |

- `notail - E1-top3` = dir_base `-0.163489`, dir_leg `-0.405310`, dir_nonleg `-0.111203`.
- `notail - E2A-R` = dir_base `-0.151981`, dir_leg `-0.416115`, dir_nonleg `-0.094871`.
- `notail - baseline70a` = dir_base `-0.048770`, dir_leg `-0.108461`, dir_nonleg `-0.035864`.

### Actual `new70b_replace_lowdrift` native

| case | dir_base(all_ex_root) | dir_leg | dir_nonleg | arm | else |
| --- | --- | --- | --- | --- | --- |
| notail new70b_replace_lowdrift | 0.197427 | 0.436922 | 0.145644 | 0.168483 | 0.091663 |
| notail final70a source | 0.248659 | 0.653602 | 0.161104 | 0.181944 | 0.111845 |
| baseline new70b_replace_lowdrift | 0.152126 | 0.391665 | 0.100334 | 0.116105 | 0.063055 |

- `notail new70b - notail final70a source` = dir_base `-0.051232`, dir_leg `-0.216680`, dir_nonleg `-0.015460`.
- `notail new70b - baseline new70b` = dir_base `0.045301`, dir_leg `0.045258`, dir_nonleg `0.045311`.
- Read: the real replace stage improves over the no-tail `70a` source, but it still trails the baseline `new70b_replace_lowdrift` native handoff across all tracked groups.

## 7. Layer 2 replace transfer table

| case | out_gap | dir_base_gap | dir_leg_gap | dir_nonleg_gap | out_closure | dir_base_closure | dir_leg_closure | dir_nonleg_closure | aggregate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transplant-compatible target | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| E1-top3 full7 | 0.475508 | 0.794442 | 1.931073 | 0.548684 | -0.012050 | 0.385584 | 0.039624 | 0.517774 | 0.232733 |
| E2A-R full7 | 0.461871 | 0.960091 | 2.021634 | 0.730569 | 0.016976 | 0.257472 | -0.005414 | 0.357920 | 0.156738 |
| notail full7 | 0.476243 | 0.834157 | 2.005153 | 0.580969 | -0.013614 | 0.354868 | 0.002782 | 0.489399 | 0.208359 |

- `notail - E1-top3` aggregate delta = `-0.024374`.
- `notail - E2A-R` aggregate delta = `0.051621`.
- `notail - E1-top3` gap deltas = out `0.000735`, dir_base `0.039715`, dir_leg `0.074080`, dir_nonleg `0.032285`.

## 8. Layer 3 head-output compatibility table

| left | right | l2_rms | cosine_distance |
| --- | --- | --- | --- |
| target-full7 | E1-top3-full7 | 0.329023 | 0.788794 |
| target-full7 | E2A-R-full7 | 0.332148 | 0.778403 |
| target-full7 | notail-full7 | 0.329208 | 0.785154 |
| E1-top3-full7 | E2A-R-full7 | 0.061945 | 0.027799 |
| E1-top3-full7 | notail-full7 | 0.098012 | 0.072941 |
| E2A-R-full7 | notail-full7 | 0.105830 | 0.081895 |

- `target ↔ notail` = l2 `0.329208`, cosdist `0.785154`.
- `target ↔ E1-top3` = l2 `0.329023`, cosdist `0.788794`.
- `target ↔ E2A-R` = l2 `0.332148`, cosdist `0.778403`.

## 9. Interpretation

- Case label: `Case C`.
- Fixed-host replace aggregate vs `E1-top3`: `-0.024374`; vs `E2A-R`: `0.051621`.
- Real `new70b_replace_lowdrift` native vs baseline replace: dir_base `0.045301`, dir_leg `0.045258`, dir_nonleg `0.045311`; vs no-tail `70a` source: dir_base `-0.051232`, dir_leg `-0.216680`, dir_nonleg `-0.015460`.
- Head-output divergence to target is smaller than `E1-top3` on both metrics: `False`; smaller than `E2A-R` on both metrics: `False`.
- Obvious native regression vs `E1-top3` final70a under the chosen guardrail: `False`.
- Keep this falsifier restrained: even if head-output moves closer, that is only supportive evidence unless replace transfer also improves clearly.

## 10. Historical next-step recommendation

- Tail-k from basetrain start as major drift driver: `not_supported_as_main_driver`.
- Historical next-step bucket: `hold_current_mainline_do_not_continue_tail_k_falsifier`.
- Q1. Fixed-host replace better than `E1-top3`? `False`.
- Q2. `direct_pose_head` divergence to target smaller than both `E1-top3` / `E2A-R`? `False`.
- Q3. Native shows obvious regression vs baseline / `E1-top3`? `False`.
- Q4. Data support `tail-k from basetrain start` as main drift driver? `not_supported_as_main_driver`.
- Q5. Historical recommendation bucket: `hold_current_mainline_do_not_continue_tail_k_falsifier`.
- Actual `new70b_replace_lowdrift`: improved over the no-tail `70a` source, but did not beat the baseline replace handoff.
