# Stage6 basetrain compare

- run_date: 20260313
- baseline: old_bestfree
- stage6_config: `/Users/xingzhaorui/PycharmProjects/PythonProject/config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- encoder_bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt`

## Basetrain endpoint

| lane | selector | all_ex_root | leg | nonleg | arm | else |
|---|---:|---:|---:|---:|---:|---:|
| old_bestfree | best_free | 6.431959 | 10.517242 | 5.548655 | 7.022836 | 2.064225 |
| cp015_bestfree | best_free | 5.647744 | 10.826995 | 4.527905 | 5.622656 | 1.940312 |
| geofix_bestteacher | best_teacher | 6.426110 | 10.931710 | 5.451926 | 6.911221 | 2.002683 |
| geofix_bestfree | best_free | 6.349273 | 10.520260 | 5.447438 | 6.894438 | 2.027256 |
| geofix_last | last | 6.426110 | 10.931710 | 5.451926 | 6.911221 | 2.002683 |

## Stage6 init

| lane | step1 leg/nonleg | head20 leg/nonleg | head20 grad arm/else | head20 arm/else |
|---|---:|---:|---:|---:|
| old_bestfree | nan | nan | 7.169632 | nan |
| cp015_bestfree | nan | nan | 7.356860 | nan |
| geofix_bestteacher | nan | nan | 7.222805 | nan |
| geofix_bestfree | nan | nan | 6.972983 | nan |
| geofix_last | nan | nan | 7.222805 | nan |

## Stage6 exit

| lane | all_ex_root | leg | nonleg | arm | else | delta all_ex_root vs old | delta leg vs old |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_bestfree | 0.313279 | 0.874230 | 0.191993 | 0.217491 | 0.131724 | +0.000000 | +0.000000 |
| cp015_bestfree | 0.431377 | 1.167171 | 0.272286 | 0.328012 | 0.140571 | +0.118097 | +0.292942 |
| geofix_bestteacher | 0.337187 | 0.901038 | 0.215273 | 0.248893 | 0.135808 | +0.023908 | +0.026809 |
| geofix_bestfree | 0.355475 | 0.974828 | 0.221561 | 0.259452 | 0.131998 | +0.042195 | +0.100599 |
| geofix_last | 0.337187 | 0.901038 | 0.215273 | 0.248893 | 0.135808 | +0.023908 | +0.026809 |

