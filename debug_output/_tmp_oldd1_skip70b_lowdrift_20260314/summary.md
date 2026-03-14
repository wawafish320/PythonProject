# old d1 skip-raw70b replace compare

- source_summary: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_oldd1_newflow_chain_20260314/summary.json`
- candidate_config: `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_oldd1_skip70b_lowdrift_20260314/configs/posttrain_70b_replace_lowdrift_from_oldd1_20260314.json`
- candidate_ckpt: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_oldd1_skip70b_lowdrift_20260314/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_from_oldd1_20260314.pth`

## Direct-path metrics (model-source)

| lane | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else |
|---|---:|---:|---:|---:|---:|---:|
| 70a | 0.275083 | 0.275083 | 0.730911 | 0.176525 | 0.203549 | 0.112650 |
| current_new70b_replace | 0.280736 | 0.280736 | 0.662440 | 0.198205 | 0.226846 | 0.130508 |
| candidate_new70b_replace_lowdrift | 0.156709 | 0.156709 | 0.375867 | 0.109324 | 0.126458 | 0.068824 |

## Deltas

| compare | d_DirectGeoLocalDeg | d_all_ex_root | d_leg | d_nonleg | d_arm | d_else |
|---|---:|---:|---:|---:|---:|---:|
| 70a -> current_new70b_replace | 0.005653 | 0.005653 | -0.068472 | 0.021680 | 0.023297 | 0.017858 |
| 70a -> candidate_new70b_replace_lowdrift | -0.118373 | -0.118373 | -0.355044 | -0.067201 | -0.077091 | -0.043826 |
| candidate - current_new70b_replace | -0.124026 | -0.124026 | -0.286573 | -0.088881 | -0.100388 | -0.061684 |

## Decision hooks

- candidate reduces drift vs current: all_ex_root=`true`, nonleg=`true`, arm=`true`
- current leg gain vs 70a: `0.068472`
- candidate leg gain vs 70a: `0.355044`
- candidate keeps at least half the current leg gain: `true`

