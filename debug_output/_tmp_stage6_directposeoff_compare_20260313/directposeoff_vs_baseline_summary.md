# Basetrain direct_pose=false -> Stage6 reinit compare

- basetrain direct_pose disabled: `yes`
- Stage6 config: `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- Stage6 direct pose still initialized/trained after basetrain: `yes`
- recommended among new lanes: `cp015_dpoff_bestfree`

| lane | baseline lane | basetrain direct off verified | stage6 reinit still active | basetrain all_ex_root | baseline | delta | stage6 all_ex_root | baseline | delta | stage6 leg | baseline | delta | stage6 nonleg | baseline | delta |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_dpoff_bestfree | old_bestfree | yes | yes | 69.571402 | 6.431959 | +63.139443 | 0.375358 | 0.313279 | +0.062079 | 0.843690 | 0.874230 | -0.030540 | 0.274098 | 0.191993 | +0.082105 |
| cp015_dpoff_bestfree | cp015_bestfree | yes | yes | 40.972967 | 5.647744 | +35.325223 | 0.325373 | 0.431377 | -0.106004 | 0.778009 | 1.167171 | -0.389162 | 0.227506 | 0.272286 | -0.044780 |

## Notes

- `old_dpoff_bestfree`: best_free ckpt has `0` learned direct_pose head/proj keys; lane log=`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_stage6_directposeoff_compare_20260313/results/old_dpoff_bestfree/lane.log`
- `cp015_dpoff_bestfree`: best_free ckpt has `0` learned direct_pose head/proj keys; lane log=`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_stage6_directposeoff_compare_20260313/results/cp015_dpoff_bestfree/lane.log`
