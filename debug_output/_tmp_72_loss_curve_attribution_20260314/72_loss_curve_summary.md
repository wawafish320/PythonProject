# 72 loss curve summary

- no explicit `omega`-named loss key appears in the stored 72 posttrain logs
- observed 72-specific leg-align keys: `leg_align_anchor_frac, leg_align_anchor_loss, leg_align_anchor_weight, leg_align_anchor_weighted, leg_align_distal_frac, leg_align_distal_loss, leg_align_frac, leg_align_joint_frac_ball_l, leg_align_joint_frac_ball_r, leg_align_joint_frac_calf_l, leg_align_joint_frac_calf_r, leg_align_joint_frac_foot_l, leg_align_joint_frac_foot_r, leg_align_joint_frac_thigh_l, leg_align_joint_frac_thigh_r, leg_align_joint_loss_ball_l, leg_align_joint_loss_ball_r, leg_align_joint_loss_calf_l, leg_align_joint_loss_calf_r, leg_align_joint_loss_foot_l, leg_align_joint_loss_foot_r, leg_align_joint_loss_thigh_l, leg_align_joint_loss_thigh_r, leg_align_loss, leg_align_proximal_frac, leg_align_proximal_loss, leg_align_weight, leg_align_weighted`

| key | lane | start20 | mid20 | late20 | epoch1 | epoch2 | epoch3 | peak_first20(step,value) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| total | current72 | 2.263405 | 1.924936 | 1.983918 | 2.104447 | 1.923855 | 2.001807 | s005, 3.113869 |
| total | candidate72 | 2.739258 | 1.873930 | 2.003513 | 2.276644 | 1.896688 | 2.021917 | s005, 4.478547 |
| total | cand-current | 0.475853 | -0.051006 | 0.019595 | 0.172196 | -0.027167 | 0.020111 | dstep=0.000000, dval=1.364678 |
| dir_geo | current72 | 2.261341 | 1.923878 | 1.982859 | 2.102736 | 1.922738 | 2.000763 | s005, 3.108846 |
| dir_geo | candidate72 | 2.734824 | 1.872471 | 2.001883 | 2.273151 | 1.895149 | 2.020420 | s005, 4.466986 |
| dir_geo | cand-current | 0.473483 | -0.051406 | 0.019024 | 0.170415 | -0.027589 | 0.019657 | dstep=0.000000, dval=1.358140 |
| leg_align_weighted | current72 | 0.002064 | 0.001059 | 0.001059 | 0.001711 | 0.001117 | 0.001044 | s005, 0.005022 |
| leg_align_weighted | candidate72 | 0.004434 | 0.001459 | 0.001630 | 0.003493 | 0.001538 | 0.001498 | s005, 0.011561 |
| leg_align_weighted | cand-current | 0.002370 | 0.000401 | 0.000571 | 0.001781 | 0.000422 | 0.000453 | dstep=0.000000, dval=0.006539 |
| leg_align_loss | current72 | 0.000103 | 0.000053 | 0.000053 | 0.000086 | 0.000056 | 0.000052 | s005, 0.000251 |
| leg_align_loss | candidate72 | 0.000222 | 0.000073 | 0.000081 | 0.000175 | 0.000077 | 0.000075 | s005, 0.000578 |
| leg_align_loss | cand-current | 0.000119 | 0.000020 | 0.000029 | 0.000089 | 0.000021 | 0.000023 | dstep=0.000000, dval=0.000327 |
| leg_align_distal_loss | current72 | 0.000049 | 0.000025 | 0.000025 | 0.000039 | 0.000025 | 0.000025 | s017, 0.000118 |
| leg_align_distal_loss | candidate72 | 0.000089 | 0.000034 | 0.000038 | 0.000075 | 0.000036 | 0.000034 | s012, 0.000237 |
| leg_align_distal_loss | cand-current | 0.000039 | 0.000010 | 0.000013 | 0.000036 | 0.000012 | 0.000008 | dstep=-5.000000, dval=0.000119 |
| leg_align_proximal_loss | current72 | 0.000056 | 0.000029 | 0.000029 | 0.000048 | 0.000032 | 0.000028 | s005, 0.000178 |
| leg_align_proximal_loss | candidate72 | 0.000136 | 0.000039 | 0.000045 | 0.000101 | 0.000041 | 0.000042 | s005, 0.000425 |
| leg_align_proximal_loss | cand-current | 0.000080 | 0.000010 | 0.000016 | 0.000053 | 0.000009 | 0.000015 | dstep=0.000000, dval=0.000247 |
| dir_group_norm_leg | current72 | 1.192757 | 0.924986 | 0.995050 | 1.074735 | 0.914937 | 1.003803 | s005, 1.939771 |
| dir_group_norm_leg | candidate72 | 1.706216 | 0.887224 | 1.006654 | 1.265739 | 0.891172 | 1.023311 | s005, 3.393183 |
| dir_group_norm_leg | cand-current | 0.513459 | -0.037762 | 0.011605 | 0.191004 | -0.023764 | 0.019508 | dstep=0.000000, dval=1.453412 |
| dir_leg_base | current72 | 0.005205 | 0.003865 | 0.003868 | 0.004870 | 0.003949 | 0.003816 | s005, 0.007985 |
| dir_leg_base | candidate72 | 0.007214 | 0.004584 | 0.004719 | 0.006565 | 0.004626 | 0.004547 | s005, 0.011926 |
| dir_leg_base | cand-current | 0.002009 | 0.000719 | 0.000852 | 0.001695 | 0.000677 | 0.000731 | dstep=0.000000, dval=0.003942 |
| dir_nonleg_base | current72 | 0.001242 | 0.001233 | 0.001226 | 0.001227 | 0.001243 | 0.001236 | s008, 0.001338 |
| dir_nonleg_base | candidate72 | 0.001272 | 0.001239 | 0.001234 | 0.001251 | 0.001254 | 0.001243 | s004, 0.001446 |
| dir_nonleg_base | cand-current | 0.000030 | 0.000006 | 0.000009 | 0.000024 | 0.000011 | 0.000007 | dstep=-4.000000, dval=0.000108 |
| boundary_dir_geo | current72 | 0.007636 | 0.007842 | 0.008131 | 0.007662 | 0.007812 | 0.007833 | s003, 0.008046 |
| boundary_dir_geo | candidate72 | 0.008373 | 0.008251 | 0.008767 | 0.008495 | 0.008339 | 0.008608 | s020, 0.009322 |
| boundary_dir_geo | cand-current | 0.000737 | 0.000409 | 0.000636 | 0.000833 | 0.000526 | 0.000776 | dstep=17.000000, dval=0.001276 |
