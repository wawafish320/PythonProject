# Walk_F contacts_meas postprocess sweep

Artifacts:
- machine summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_72_lowlr_to_lambda_20260315/eval_lambda_model/Walk_F_gait_speed_scaling_whitebox_meas_postproc_sweep_summary.json`
- per-run JSONs: same directory, `Walk_F_gait_speed_scaling_whitebox_*.json`

## Case ranking (avg over scales)

| case | stable_scales | avg_td_count | avg_lr_delta | avg_freq_hz | avg_freq_std_hz | avg_E_cycle_speed_consistency |
|---|---:|---:|---:|---:|---:|---:|
| teacher control | 5 | 5.0 | 0.0 | 0.690 | 0.000 | 0.0003 |
| plan control | 1 | 5.6 | 2.2 | 2.422 | 2.239 | 0.0304 |
| meas baseline | 0 | 28.0 | 14.0 | 9.299 | 8.024 | 0.0967 |
| meas_logit_scale_0p5 | 0 | 28.0 | 14.0 | 9.299 | 8.024 | 0.0967 |
| meas_logit_scale_1p0 | 0 | 28.0 | 14.0 | 9.299 | 8.024 | 0.0967 |
| meas_logit_scale_1p5 | 0 | 28.0 | 14.0 | 9.299 | 8.024 | 0.0967 |
| meas_logit_scale_2p0 | 0 | 28.0 | 14.0 | 9.299 | 8.024 | 0.0967 |
| meas cond-onehot ds=0.5 | 0 | 29.4 | 3.4 | 7.900 | 5.757 | 0.0998 |
| meas_onehot_conditional_ds_0p4 | 0 | 29.6 | 3.2 | 7.966 | 5.766 | 0.1008 |
| meas combo scale=2.0 + cond-onehot ds=0.6 | 0 | 30.0 | 2.6 | 7.894 | 5.536 | 0.1008 |
| meas_combo_soft_cond_ds_0p4 | 0 | 30.2 | 3.2 | 8.628 | 7.004 | 0.0993 |
| meas strict onehot | 0 | 31.0 | 0.0 | 8.345 | 5.958 | 0.0981 |
| meas_onehot_conditional_ds_0p6 | 0 | 31.0 | 1.0 | 8.157 | 5.477 | 0.1016 |

## Most informative per-scale rows

| case | scale | td_count | L | R | td_unstable | freq_hz | freq_std_hz | stride_length | E_cycle_speed_consistency | E_cycle_leg | E_cycle_nonleg | status |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| teacher control | 0.8 | 5 | 5 | 5 | false | 0.690 | 0.000 | 1.087 | 0.0003 | 0.695 | 0.421 | pass |
| teacher control | 0.9 | 5 | 5 | 5 | false | 0.690 | 0.000 | 1.223 | 0.0003 | 0.322 | 0.186 | pass |
| teacher control | 1.0 | 5 | 5 | 5 | false | 0.690 | 0.000 | 1.359 | 0.0003 | 0.000 | 0.000 | pass |
| teacher control | 1.1 | 5 | 5 | 5 | false | 0.690 | 0.000 | 1.494 | 0.0003 | 0.347 | 0.235 | pass |
| teacher control | 1.2 | 5 | 5 | 5 | false | 0.690 | 0.000 | 1.630 | 0.0003 | 0.683 | 0.470 | pass |
| plan control | 0.8 | 4 | 4 | 2 | true | 3.825 | 2.702 | 1.399 | 0.0270 | 3.112 | 0.977 | fail |
| plan control | 0.9 | 4 | 2 | 4 | true | 1.038 | 0.607 | 1.429 | 0.0202 | 2.697 | 0.862 | fail |
| plan control | 1.0 | 6 | 6 | 7 | false | 1.848 | 1.843 | 1.006 | 0.0405 | 0.000 | 0.000 | pass |
| plan control | 1.1 | 7 | 7 | 9 | true | 2.285 | 1.977 | 0.934 | 0.0367 | 1.707 | 0.544 | fail |
| plan control | 1.2 | 7 | 7 | 11 | true | 3.113 | 4.066 | 1.019 | 0.0277 | 1.796 | 0.635 | fail |
| meas baseline | 0.8 | 30 | 44 | 30 | true | 8.829 | 7.672 | 0.178 | 0.0888 | 0.833 | 0.446 | fail |
| meas baseline | 0.9 | 31 | 44 | 31 | true | 8.768 | 7.449 | 0.194 | 0.0980 | 0.592 | 0.275 | fail |
| meas baseline | 1.0 | 26 | 39 | 26 | true | 7.779 | 7.476 | 0.258 | 0.0858 | 0.000 | 0.000 | fail |
| meas baseline | 1.1 | 27 | 45 | 27 | true | 9.038 | 7.763 | 0.269 | 0.0992 | 1.525 | 0.621 | fail |
| meas baseline | 1.2 | 26 | 38 | 26 | true | 12.079 | 9.757 | 0.305 | 0.1119 | 1.618 | 0.717 | fail |
| meas cond-onehot ds=0.5 | 0.8 | 26 | 26 | 28 | true | 6.771 | 4.296 | 0.196 | 0.0982 | 0.802 | 0.425 | fail |
| meas cond-onehot ds=0.5 | 0.9 | 26 | 26 | 31 | true | 7.740 | 6.241 | 0.219 | 0.0929 | 0.398 | 0.186 | fail |
| meas cond-onehot ds=0.5 | 1.0 | 26 | 26 | 31 | true | 6.502 | 4.153 | 0.244 | 0.0975 | 0.000 | 0.000 | fail |
| meas cond-onehot ds=0.5 | 1.1 | 35 | 35 | 38 | true | 9.779 | 7.997 | 0.198 | 0.1053 | 0.448 | 0.284 | fail |
| meas cond-onehot ds=0.5 | 1.2 | 34 | 34 | 36 | true | 8.707 | 6.099 | 0.222 | 0.1050 | 0.740 | 0.484 | fail |
| meas combo scale=2.0 + cond-onehot ds=0.6 | 0.8 | 27 | 27 | 28 | true | 7.019 | 4.228 | 0.188 | 0.0984 | 0.809 | 0.446 | fail |
| meas combo scale=2.0 + cond-onehot ds=0.6 | 0.9 | 27 | 27 | 31 | true | 7.794 | 6.098 | 0.211 | 0.0963 | 0.465 | 0.210 | fail |
| meas combo scale=2.0 + cond-onehot ds=0.6 | 1.0 | 27 | 27 | 30 | true | 6.716 | 4.062 | 0.234 | 0.0998 | 0.000 | 0.000 | fail |
| meas combo scale=2.0 + cond-onehot ds=0.6 | 1.1 | 35 | 35 | 39 | true | 9.177 | 7.117 | 0.198 | 0.1049 | 0.523 | 0.294 | fail |
| meas combo scale=2.0 + cond-onehot ds=0.6 | 1.2 | 34 | 34 | 35 | true | 8.766 | 6.174 | 0.222 | 0.1047 | 0.783 | 0.475 | fail |
| meas strict onehot | 0.8 | 28 | 28 | 28 | true | 8.925 | 7.955 | 0.192 | 0.0820 | 0.793 | 0.380 | fail |
| meas strict onehot | 0.9 | 27 | 27 | 27 | true | 6.697 | 3.824 | 0.211 | 0.0940 | 0.403 | 0.184 | fail |
| meas strict onehot | 1.0 | 28 | 28 | 28 | true | 6.773 | 3.936 | 0.226 | 0.1024 | 0.000 | 0.000 | fail |
| meas strict onehot | 1.1 | 38 | 38 | 38 | true | 10.563 | 7.901 | 0.182 | 0.1072 | 0.621 | 0.313 | fail |
| meas strict onehot | 1.2 | 34 | 34 | 34 | true | 8.766 | 6.174 | 0.222 | 0.1047 | 0.816 | 0.494 | fail |

## Readout

- `teacher` still behaves as the clean upper bound: all scales `pass`, `td_unstable=false`, `touchdown_count=5`, `freq_hz=0.689655`, `E_cycle_speed_consistency≈2.67e-4`.
- `plan` still only stabilizes `1.0x`; off-scale it fails from unstable touchdown boundaries.
- `meas` baseline still fully fails from chatter: `touchdown_count=26~31`, avg L/R delta `14.0`, `freq_hz=7.78~12.08`, `freq_std_hz=7.45~9.76`.
- Standalone `logit_scale` has exactly no observable effect under the current `0.5` touchdown threshold.
- Smallest useful meas-only setting is `--contacts_meas_model_onehot_conditional` (default `ds_thr=0.5`): it cuts avg L/R delta `14.0 -> 3.4` and avg `freq_std_hz 8.02 -> 5.76`, but all scales still fail and avg touchdown_count stays `29.4`.
- Best absolute case tested is `scale=2.0 + cond-onehot ds=0.6`: avg L/R delta reaches `2.6`, but avg touchdown_count is still `30.0` and stable scales remain `0/5`.
- Strict onehot equalizes L/R counts (`avg_lr_delta=0.0`) but does not solve over-count or jitter, so it is not a good default.

## Recommendation

- If you want a minimal meas-side default today, use `--contacts_meas_model_onehot_conditional` only; treat it as a mild cleanup, not a fix.
- Do not expect meas postprocess alone to make this lane passable: the dominant failure remains touchdown over-count, with period jitter still secondary even after L/R balancing.
- Given this sweep, continuing to squeeze `meas` postprocess is low-yield; the next higher-value lane is `plan` (or a stronger meas source redesign), because `plan` is already near interpretable at `1.0x` while `meas` remains structurally chattery on every scale.
