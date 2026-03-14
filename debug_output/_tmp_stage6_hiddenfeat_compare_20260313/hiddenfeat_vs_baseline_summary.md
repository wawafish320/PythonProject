# Hidden-feature direct branch vs baseline

- grad probe: `passed` (`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_stage6_hiddenfeat_gradprobe_20260313/hiddenfeat_backbone_grad_probe.json`)
- no-op check: `no`
- stage6 config: `/Users/xingzhaorui/PycharmProjects/PythonProject/config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- stage6 reinit: `true`
- hidden path fallback used: `no` (`direct_pose_feat_source="hidden"` for all four new lanes)
- hidden-lane aggregation json: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_stage6_hiddenfeat_compare_20260313/results/selector_summary.json`

## Answers

1. hidden-feature direct branch连到backbone后，basetrain verdict=`mixed`，Stage6 verdict=`mixed`。
2. `direct_pose_detach_feat=true` 对 Stage6 handoff 的结论=`worse`。
3. `old`: Stage6 最好的是 `old_bestfree` (baseline(cond)); basetrain 最好的是 `old_hidden_gradoff` (hidden+gradoff).
3. `cp015`: Stage6 最好的是 `cp015_dpoff_bestfree` (direct_pose=false); basetrain 最好的是 `cp015_hidden_gradon` (hidden+gradon).

## Gradient probe

| lane | detach_feat | direct_pose_head grad | shared_encoder grad | contact_plan grad |
|---|---:|---:|---:|---:|
| old_hidden_gradon | false | 3.648584 | 0.188448 | 0.000000 |
| old_hidden_gradoff | true | 7.750209 | 0.000000 | 0.000000 |
| cp015_hidden_gradon | false | 3.951730 | 0.265983 | 0.000000 |
| cp015_hidden_gradoff | true | 3.747896 | 0.000000 | 0.000000 |

## Basetrain best_free endpoint

| family | lane | variant | all_ex_root | leg | nonleg | delta_vs_baseline all_ex_root |
|---|---|---|---:|---:|---:|---:|
| cp015 | cp015_bestfree | baseline(cond) | 5.647744 | 10.826995 | 4.527905 | 0.000000 |
| cp015 | cp015_dpoff_bestfree | direct_pose=false | 40.972967 | 57.426126 | 37.415527 | 35.325223 |
| cp015 | cp015_hidden_gradon | hidden+gradon | 3.894768 | 9.720003 | 2.635257 | -1.752976 |
| cp015 | cp015_hidden_gradoff | hidden+gradoff | 6.455206 | 11.493951 | 5.365747 | 0.807462 |
| old | old_bestfree | baseline(cond) | 6.431959 | 10.517242 | 5.548655 | 0.000000 |
| old | old_dpoff_bestfree | direct_pose=false | 69.571402 | 77.849582 | 67.781525 | 63.139443 |
| old | old_hidden_gradon | hidden+gradon | 8.523103 | 15.184489 | 7.082803 | 2.091144 |
| old | old_hidden_gradoff | hidden+gradoff | 4.062449 | 10.200576 | 2.735286 | -2.369510 |

## Basetrain checkpoint contact-plan stats

| family | lane | variant | ckpt selector | train epoch | contact_plan_bce | contact_plan_mse |
|---|---|---|---|---:|---:|---:|
| cp015 | cp015_bestfree | baseline(cond) | best_free | 12 | 0.364820 | 0.022865 |
| cp015 | cp015_dpoff_bestfree | direct_pose=false | best_free | 12 | 0.390859 | 0.032500 |
| cp015 | cp015_hidden_gradon | hidden+gradon | best_free | 13 | 0.357475 | 0.019826 |
| cp015 | cp015_hidden_gradoff | hidden+gradoff | best_free | 6 | 0.451244 | 0.054434 |
| old | old_bestfree | baseline(cond) | best_free | 10 | 0.474549 | 0.064801 |
| old | old_dpoff_bestfree | direct_pose=false | best_free | 7 | 0.471200 | 0.064080 |
| old | old_hidden_gradon | hidden+gradon | best_free | 11 | 0.398204 | 0.033343 |
| old | old_hidden_gradoff | hidden+gradoff | best_free | 16 | 0.395272 | 0.031630 |

## Stage6 exit

| family | lane | variant | all_ex_root | leg | nonleg | delta_vs_baseline all_ex_root |
|---|---|---|---:|---:|---:|---:|
| cp015 | cp015_bestfree | baseline(cond) | 0.431377 | 1.167171 | 0.272286 | 0.000000 |
| cp015 | cp015_dpoff_bestfree | direct_pose=false | 0.325373 | 0.778009 | 0.227506 | -0.106004 |
| cp015 | cp015_hidden_gradon | hidden+gradon | 0.336038 | 0.763625 | 0.243587 | -0.095339 |
| cp015 | cp015_hidden_gradoff | hidden+gradoff | 0.353286 | 0.945024 | 0.225343 | -0.078091 |
| old | old_bestfree | baseline(cond) | 0.313279 | 0.874230 | 0.191993 | 0.000000 |
| old | old_dpoff_bestfree | direct_pose=false | 0.375358 | 0.843690 | 0.274098 | 0.062079 |
| old | old_hidden_gradon | hidden+gradon | 0.348316 | 0.840691 | 0.241857 | 0.035037 |
| old | old_hidden_gradoff | hidden+gradoff | 0.397575 | 1.067834 | 0.252654 | 0.084295 |

## Stage6 delta from init

- definition: `stage6_exit - basetrain`, negative is better

| family | lane | variant | delta all_ex_root | delta leg | delta nonleg |
|---|---|---|---:|---:|---:|
| cp015 | cp015_bestfree | baseline(cond) | -5.216367 | -9.659824 | -4.255619 |
| cp015 | cp015_dpoff_bestfree | direct_pose=false | -40.647593 | -56.648117 | -37.188021 |
| cp015 | cp015_hidden_gradon | hidden+gradon | -3.558730 | -8.956377 | -2.391671 |
| cp015 | cp015_hidden_gradoff | hidden+gradoff | -6.101920 | -10.548928 | -5.140405 |
| old | old_bestfree | baseline(cond) | -6.118680 | -9.643013 | -5.356662 |
| old | old_dpoff_bestfree | direct_pose=false | -69.196043 | -77.005892 | -67.507428 |
| old | old_hidden_gradon | hidden+gradon | -8.174786 | -14.343798 | -6.840946 |
| old | old_hidden_gradoff | hidden+gradoff | -3.664874 | -9.132742 | -2.482632 |

## Family deltas

- `old` hidden+gradon vs baseline: basetrain Δall=2.091144, Stage6 Δall=0.035037
- `old` hidden+gradoff vs baseline: basetrain Δall=-2.369510, Stage6 Δall=0.084295
- `old` hidden+gradoff vs hidden+gradon: basetrain Δall=-4.460654, Stage6 Δall=0.049258
- `cp015` hidden+gradon vs baseline: basetrain Δall=-1.752976, Stage6 Δall=-0.095339
- `cp015` hidden+gradoff vs baseline: basetrain Δall=0.807462, Stage6 Δall=-0.078091
- `cp015` hidden+gradoff vs hidden+gradon: basetrain Δall=2.560438, Stage6 Δall=0.017248

