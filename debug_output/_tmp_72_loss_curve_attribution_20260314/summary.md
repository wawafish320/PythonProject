# 72 loss curve attribution

## Short conclusion

- candidate `71 (lr=3e-4)` does start `72` from a better aggregate leg/all_ex_root state, but unchanged `72` immediately over-updates that cleaner start
- the stored 72 logs already show a clear early overshoot on the candidate lane: `total start20 2.739258` vs current `72` `2.263405`, `dir_group_norm_leg start20 1.706216` vs `1.192757`
- replay says the aggregate regression is introduced inside `72`, not inherited from candidate `71`: earliest snapshot crossing is `s005`
- hotspot wins survive because `foot_l/ball_l@SIC12-15` and `calf_r@SIC2-4` remain better, but broader losses on `calf_l`, `ball_l`, `ball_r`, and late mid-cycle leg windows outweigh those local gains
- best next minimal lever: `lower_lr_72_or_gentler_72`

## End-state table

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current `71` | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.295473 | 0.082665 | 0.599272 | 0.440912 |
| candidate `71` (`lr=3e-4`) | 0.107064 | 0.107064 | 0.215044 | 0.083717 | 0.091849 | 0.215044 | 0.091849 | 0.429449 | 0.099602 |
| current `72` | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate `72` | 0.121936 | 0.121936 | 0.298698 | 0.083717 | 0.091849 | 0.298698 | 0.091849 | 0.586438 | 0.191232 |

## Replay snapshots

| lane_snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current `s000` | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.295473 | 0.082665 | 0.599272 | 0.440912 |
| current `s005` | 0.129621 | 0.129621 | 0.395093 | 0.072222 | 0.082665 | 0.395093 | 0.082665 | 1.397935 | 0.463666 |
| current `s010` | 0.128126 | 0.128126 | 0.386686 | 0.072222 | 0.082665 | 0.386686 | 0.082665 | 0.827815 | 0.357110 |
| current `s020` | 0.114371 | 0.114371 | 0.309314 | 0.072222 | 0.082665 | 0.309314 | 0.082665 | 0.709409 | 0.305969 |
| current `s040` | 0.116234 | 0.116234 | 0.319793 | 0.072222 | 0.082665 | 0.319793 | 0.082665 | 0.568103 | 0.294565 |
| current `s060` | 0.120596 | 0.120596 | 0.344328 | 0.072222 | 0.082665 | 0.344328 | 0.082665 | 0.902402 | 0.264104 |
| current `s120` | 0.106344 | 0.106344 | 0.264158 | 0.072222 | 0.082665 | 0.264158 | 0.082665 | 0.480181 | 0.290744 |
| current `s180` | 0.112074 | 0.112074 | 0.296389 | 0.072222 | 0.082665 | 0.296389 | 0.082665 | 0.812663 | 0.288880 |
| candidate `s000` | 0.107064 | 0.107064 | 0.215044 | 0.083717 | 0.091849 | 0.215044 | 0.091849 | 0.429449 | 0.099602 |
| candidate `s005` | 0.151228 | 0.151228 | 0.463466 | 0.083717 | 0.091849 | 0.463466 | 0.091849 | 1.155790 | 0.246557 |
| candidate `s010` | 0.152090 | 0.152090 | 0.468314 | 0.083717 | 0.091849 | 0.468314 | 0.091849 | 1.809244 | 0.313736 |
| candidate `s020` | 0.131960 | 0.131960 | 0.355084 | 0.083717 | 0.091849 | 0.355084 | 0.091849 | 1.407482 | 0.136625 |
| candidate `s040` | 0.144241 | 0.144241 | 0.424164 | 0.083717 | 0.091849 | 0.424164 | 0.091849 | 0.776220 | 0.113536 |
| candidate `s060` | 0.140788 | 0.140788 | 0.404737 | 0.083717 | 0.091849 | 0.404737 | 0.091849 | 1.128362 | 0.286177 |
| candidate `s120` | 0.121458 | 0.121458 | 0.296009 | 0.083717 | 0.091849 | 0.296009 | 0.091849 | 0.521412 | 0.199641 |
| candidate `s180` | 0.121936 | 0.121936 | 0.298698 | 0.083717 | 0.091849 | 0.298698 | 0.091849 | 0.586438 | 0.191232 |

## 71->72 gain decomposition

| metric | inherited (candidate71-current71) | current72-current71 | candidate72-candidate71 | stage72 gain gap | final gap (candidate72-current72) |
|---|---:|---:|---:|---:|---:|
| DirectGeoLocalDeg | -0.004846 | 0.000163 | 0.014872 | 0.014709 | 0.009863 |
| all_ex_root | -0.004846 | 0.000163 | 0.014872 | 0.014709 | 0.009863 |
| leg | -0.080429 | 0.000916 | 0.083653 | 0.082737 | 0.002308 |
| nonleg | 0.011496 | 0.000000 | 0.000000 | 0.000000 | 0.011496 |
| arm | 0.009184 | 0.000000 | 0.000000 | 0.000000 | 0.009184 |
| legs_main | -0.080429 | 0.000916 | 0.083653 | 0.082737 | 0.002308 |
| arms_main | 0.009184 | 0.000000 | 0.000000 | 0.000000 | 0.009184 |
| foot_l_ball_l_SIC12_15 | -0.169823 | 0.213391 | 0.156989 | -0.056403 | -0.226225 |
| calf_r_SIC2_4 | -0.341310 | -0.152032 | 0.091629 | 0.243661 | -0.097649 |

## Final candidate72 regressions vs current72

| leg_joint | delta(candidate72-current72) | current72 | candidate72 |
|---|---:|---:|---:|
| calf_l | 0.073529 | 0.281619 | 0.355148 |
| ball_l | 0.056393 | 0.248964 | 0.305358 |
| ball_r | 0.020095 | 0.242842 | 0.262936 |
| foot_l | 0.002722 | 0.409273 | 0.411996 |
| foot_r | -0.021797 | 0.305525 | 0.283728 |
| thigh_r | -0.027533 | 0.282815 | 0.255282 |
| calf_r | -0.032569 | 0.259140 | 0.226570 |
| thigh_l | -0.052374 | 0.340936 | 0.288562 |

| leg_SIC | delta(candidate72-current72) | current72 | candidate72 |
|---|---:|---:|---:|
| SIC45 | 0.288106 | 0.208989 | 0.497094 |
| SIC48 | 0.273289 | 0.272011 | 0.545300 |
| SIC37 | 0.214321 | 0.292767 | 0.507089 |
| SIC47 | 0.205110 | 0.328796 | 0.533906 |
| SIC46 | 0.187589 | 0.380530 | 0.568118 |
| SIC36 | 0.186622 | 0.240836 | 0.427458 |
| SIC35 | 0.174651 | 0.157383 | 0.332034 |
| SIC21 | 0.170733 | 0.192716 | 0.363449 |
| SIC43 | 0.139228 | 0.243676 | 0.382905 |
| SIC03 | 0.136851 | 0.191233 | 0.328083 |
| SIC44 | 0.132862 | 0.220416 | 0.353278 |
| SIC22 | 0.102598 | 0.184627 | 0.287225 |

## Artifacts

- loss curve plot: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_72_loss_curve_attribution_20260314/72_loss_curve_compare.png`
- loss curve summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_72_loss_curve_attribution_20260314/72_loss_curve_summary.md`
- machine summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_72_loss_curve_attribution_20260314/summary.json`

