# Stage6 true keep-weight vs reinit

- keepweight config: `debug_output/_tmp_stage6_keepweight_compare_20260313/stage6_direct_cond_anchor_splitfirst_3way_armchain_keepweight_20260313.json`
- config delta: `direct_pose_reinit=false`, `direct_pose_hidden_override=null`, `direct_pose_time_pe_dim=-1`, `direct_pose_feat_source="auto"`
- keepweight verified: `yes`
- verification rule: log must not contain `dropped 10 direct_pose_* tensors from checkpoint (reinit/override).`

| lane | keep-weight verified | reinit all_ex_root | keep all_ex_root | delta | reinit leg | keep leg | delta | reinit nonleg | keep nonleg | delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_bestfree | yes | 0.313279 | 3.338256 | +3.024977 | 0.874230 | 9.563805 | +8.689576 | 0.191993 | 1.992192 | +1.800199 |
| cp015_bestfree | yes | 0.431377 | 3.905214 | +3.473837 | 1.167171 | 10.875862 | +9.708691 | 0.272286 | 2.398047 | +2.125761 |

## Verification

- `old_bestfree`: `no drop line`; log=`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_stage6_keepweight_compare_20260313/results/old_bestfree/lane.log`
- `cp015_bestfree`: `no drop line`; log=`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_stage6_keepweight_compare_20260313/results/cp015_bestfree/lane.log`
