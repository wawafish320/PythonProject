# low-LR 71 -> 72 -> lambda

- 71 source ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_71_lowlr_sweep_20260314/lr3e4/ckpt_last_WalkF_stage7_71_lr3e4_from_candidate70R_20260314.pth`
- start lane: candidate 70R -> 71(lr=3e-4)
- eval contract: model-source only

## Metrics

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_71_lowlr | 0.107064 | 0.107064 | 0.215044 | 0.083717 | 0.091849 | 0.215044 | 0.091849 | 0.429449 | 0.099602 |
| current_72 | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate_72 | 0.121936 | 0.121936 | 0.298698 | 0.083717 | 0.091849 | 0.298698 | 0.091849 | 0.586438 | 0.191232 |
| current_lambda | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate_lambda | 0.121936 | 0.121936 | 0.298698 | 0.083717 | 0.091849 | 0.298698 | 0.091849 | 0.586438 | 0.191232 |

## Key deltas

- candidate 72 vs current 72: all_ex_root=0.009863, leg=0.002308, foot_l/ball_l@SIC12-15=-0.226225, calf_r@SIC2-4=-0.097649
- candidate lambda vs current lambda: all_ex_root=0.009863, leg=0.002308, foot_l/ball_l@SIC12-15=-0.226225, calf_r@SIC2-4=-0.097649
- candidate 72 vs candidate 71: all_ex_root=0.014872, leg=0.083653, foot_l/ball_l@SIC12-15=0.156989, calf_r@SIC2-4=0.091629
- candidate lambda vs candidate 72: all_ex_root=0.000000, leg=0.000000, foot_l/ball_l@SIC12-15=0.000000, calf_r@SIC2-4=0.000000

