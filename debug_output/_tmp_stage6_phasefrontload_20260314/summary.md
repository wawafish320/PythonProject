# Stage6 phase-hint frontload

- run_date: 20260314
- base_stage6_config: `/Users/xingzhaorui/PycharmProjects/PythonProject/config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- baseline_compare_summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_stage6_basetrain_compare_20260313/compare_summary.json`
- plantransplant_summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_stage6_plantransplant_20260314/summary.json`

## Reference Stage6 exits

| reference | all_ex_root | leg | nonleg |
|---|---:|---:|---:|
| cp015_bestfree | 0.431377 | 1.167171 | 0.272286 |
| old_bestfree | 0.313279 | 0.874230 | 0.191993 |
| cp015_with_old_planstack | 0.295533 | 0.740703 | 0.199280 |

## Case config diffs

| case | direct_pose_use_phase_z | direct_pose_phase_z_mode | direct_pose_reinit(base) |
|---|---|---|---|
| cp015_stage6_phasezin_frontload | True | concat | True |
| cp015_stage6_replacecontacts_frontload | True | replace_contacts | True |

## Stage6 exit

| case | all_ex_root | leg | nonleg | delta_vs_old | delta_vs_cp015_old_plan | improve_vs_cp015 |
|---|---:|---:|---:|---:|---:|---:|
| cp015_stage6_phasezin_frontload | 0.436702 | 1.037660 | 0.306765 | 0.123422 | 0.141169 | -0.005325 |
| cp015_stage6_replacecontacts_frontload | 0.470993 | 1.312606 | 0.289023 | 0.157714 | 0.175461 | -0.039616 |

## Stage6 init

| case | step1 leg/nonleg | head20 leg/nonleg | head20 grad arm/else |
|---|---:|---:|---:|
| cp015_stage6_phasezin_frontload | 3.108169 | 3.680103 | 6.812918 |
| cp015_stage6_replacecontacts_frontload | 3.132970 | 3.657330 | 7.429086 |

## Answers

1. `phasezin_frontload` vs `cp015_bestfree` (delta = baseline - case): all_ex_root -0.005325, leg 0.129512, nonleg -0.034479.
2. `replacecontacts_frontload` vs `cp015_bestfree` (delta = baseline - case): all_ex_root -0.039616, leg -0.145435, nonleg -0.016737.
3. Closest to `old_bestfree` is `cp015_stage6_phasezin_frontload` with all_ex_root gap 0.123422 (old_bestfree=0.313279).
4. `replacecontacts_frontload` vs `cp015_with_old_planstack`: all_ex_root gap 0.175461, leg gap 0.571904, nonleg gap 0.089743.
5. Recommended Stage2 seed: `cp015_stage6_phasezin_frontload`.

