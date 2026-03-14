# old d1 lowdrift replace -> 70R -> 71

- source_summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_oldd1_newflow_chain_20260314/summary.json`
- replace_summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_oldd1_skip70b_lowdrift_20260314/summary.json`
- candidate_70R_ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_skip70b_lowdrift_to71_20260314/70R/ckpt_last_WalkF_stage7_70R_from_oldd1_lowdrift_replace_20260314.pth`
- candidate_71_ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_skip70b_lowdrift_to71_20260314/71/ckpt_last_WalkF_stage7_71_from_oldd1_lowdrift_replace_20260314.pth`

## Direct-path metrics (model-source)

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else | calf_r@SIC2-4 | foot_l/ball_l@SIC12-15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_replace | 0.156709 | 0.156709 | 0.375867 | 0.109324 | 0.126458 | 0.068824 | 1.030131 | 0.551791 |
| current_70R | 0.158235 | 0.158235 | 0.556049 | 0.072222 | 0.082665 | 0.047537 | 0.613849 | 1.118483 |
| candidate_70R | 0.130926 | 0.130926 | 0.349263 | 0.083717 | 0.091849 | 0.064498 | 0.393019 | 0.860095 |
| current_71 | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.047537 | 0.440912 | 0.599272 |
| candidate_71 | 0.127787 | 0.127787 | 0.331611 | 0.083717 | 0.091849 | 0.064498 | 0.295644 | 0.540575 |

## Deltas

| compare | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_calf_r@SIC2-4 | d_foot_l/ball_l@SIC12-15 |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate_70R - current_70R | -0.027310 | -0.027310 | -0.206786 | 0.011496 | 0.009184 | -0.220830 | -0.258388 |
| candidate_71 - current_71 | 0.015877 | 0.015877 | 0.036138 | 0.011496 | 0.009184 | -0.145269 | -0.058697 |
| candidate_70R - candidate_replace | -0.025784 | -0.025784 | -0.026604 | -0.025606 | -0.034610 | -0.637112 | 0.308305 |
| candidate_71 - candidate_70R | -0.003138 | -0.003138 | -0.017651 | 0.000000 | 0.000000 | -0.097375 | -0.319520 |

## Answers

- calf recovers at 70R vs replace: `true`
- calf recovers at 71 vs 70R: `true`
- candidate 71 beats current 71 on leg: `false`
- candidate 71 beats current 71 on calf hotspot: `true`

