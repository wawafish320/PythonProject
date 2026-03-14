# Stage6 probe selector

- run_tag: 20260313_directposeoff
- stage6_config: `/Users/xingzhaorui/PycharmProjects/PythonProject/config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- score: `1.000*all_ex_root + 1.500*leg + 0.500*nonleg`
- recommended: `cp015_dpoff_bestfree`

## Ranking

| rank | lane | selector | score | stage6 all_ex_root | stage6 leg | stage6 nonleg | step1 leg/nonleg | head20 leg/nonleg |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | cp015_dpoff_bestfree | best_free | 1.606140 | 0.325373 | 0.778009 | 0.227506 | 3.125625 | 3.741007 |
| 2 | old_dpoff_bestfree | best_free | 1.777942 | 0.375358 | 0.843690 | 0.274098 | 3.124421 | 3.764053 |

