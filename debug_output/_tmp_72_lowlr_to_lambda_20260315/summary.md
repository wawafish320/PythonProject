# low-LR 72 -> lambda

- 72 source ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_72_lowlr_sweep_20260314/lr1e4/ckpt_last_WalkF_stage7_72_lr1e4_from_lowlr71_20260314.pth`
- start lane: candidate 71(lr=3e-4) -> 72(lr=1e-4)
- eval contract: model-source only

## Metrics

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_72_lowlr | 0.101969 | 0.101969 | 0.186385 | 0.083717 | 0.091849 | 0.186385 | 0.091849 | 0.385267 | 0.042300 |
| current_lambda | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate_lambda | 0.101969 | 0.101969 | 0.186385 | 0.083717 | 0.091849 | 0.186385 | 0.091849 | 0.385267 | 0.042300 |

## Key deltas

- candidate lambda vs current lambda: all_ex_root=-0.010104, leg=-0.110005, foot_l/ball_l@SIC12-15=-0.427396, calf_r@SIC2-4=-0.246580
- candidate lambda vs candidate 72(lr=1e-4): all_ex_root=0.000000, leg=0.000000, foot_l/ball_l@SIC12-15=0.000000, calf_r@SIC2-4=0.000000

