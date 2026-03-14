# Stage6 frozen plan-stack transplant

- run_date: 20260314
- baseline compare reused: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_stage6_basetrain_compare_20260313/compare_summary.json`
- stage6 config: `/Users/xingzhaorui/PycharmProjects/PythonProject/config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- teacher: `/Users/xingzhaorui/PycharmProjects/PythonProject/validate/teacher_batches/Walk_F_teacher.json`
- encoder bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt`
- affine stats: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`

## Stage6 exit

| case | type | backbone | plan-stack | all_ex_root | leg | nonleg | delta_vs_source all_ex_root |
|---|---|---|---|---:|---:|---:|---:|
| old_bestfree | baseline_reused | old_bestfree | old_bestfree | 0.313279 | 0.874230 | 0.191993 | 0.000000 |
| cp015_bestfree | baseline_reused | cp015_bestfree | cp015_bestfree | 0.431377 | 1.167171 | 0.272286 | 0.000000 |
| cp015_with_old_planstack | planstack_transplant | cp015_bestfree | old_bestfree | 0.295533 | 0.740703 | 0.199280 | -0.135844 |
| old_with_cp015_planstack | planstack_transplant | old_bestfree | cp015_bestfree | 0.374079 | 0.789892 | 0.284174 | 0.060800 |

## Stage6 init

| case | step1 dir_leg_base | step1 dir_nonleg_base | step1 leg/nonleg | head20 dir_leg_base | head20 dir_nonleg_base | head20 leg/nonleg | head20 grad arm/else |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_bestfree | 0.191689 | 0.061304 | 3.126852 | 0.136075 | 0.036436 | 3.785068 | 7.169632 |
| cp015_bestfree | 0.191913 | 0.061325 | 3.129458 | 0.133901 | 0.036052 | 3.760922 | 7.356860 |
| cp015_with_old_planstack | 0.191784 | 0.061369 | 3.125095 | 0.131952 | 0.036026 | 3.694466 | 7.396490 |
| old_with_cp015_planstack | 0.191577 | 0.061336 | 3.123428 | 0.134651 | 0.036528 | 3.726379 | 6.987993 |

## Init delta vs source baseline

| case | step1 leg/nonleg delta | head20 leg/nonleg delta | head20 grad arm/else delta |
|---|---:|---:|---:|
| old_bestfree | 0.000000 | 0.000000 | 0.000000 |
| cp015_bestfree | 0.000000 | 0.000000 | 0.000000 |
| cp015_with_old_planstack | -0.004363 | -0.066456 | 0.039630 |
| old_with_cp015_planstack | -0.003424 | -0.058689 | -0.181639 |

## Transplant verification

| case | transplanted keys | changed keys | verified_after_save |
|---|---:|---:|---:|
| cp015_with_old_planstack | 50 | 45 | true |
| old_with_cp015_planstack | 50 | 45 | true |

## Answers

1. `cp015_with_old_planstack` vs `cp015_bestfree`: all_ex_root improve 0.135844, leg improve 0.426469, nonleg improve 0.073006.
2. `cp015_with_old_planstack` all_ex_root = 0.295533; `old_bestfree` = 0.313279; gap = -0.017747; beats_old = `true`; closed_ratio = 1.150272.
3. `old_with_cp015_planstack` vs `old_bestfree`: all_ex_root delta 0.060800, leg delta -0.084338, nonleg delta 0.092181.
4. Hypothesis signal should be judged from the two swap directions together; see exact deltas in `summary.json`.

