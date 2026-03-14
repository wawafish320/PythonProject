# Direct pose backbone grad probe

- conclusion: current basetrain `direct_pose` branch is already detached from backbone/contact-plan in the mainline d1/cp015 configs
- code refs: `train/models.py:594`, `train/models.py:611`, `train/models.py:805`, `train/models.py:814`, `train/models.py:3482`, `train/models.py:3584`

| lane | detach_plan | feat_source | grad direct_pose_head | grad shared_encoder | grad contact_plan |
|---|---|---|---:|---:|---:|
| old_bestfree | True | cond | 0.157478 | 0.000000 | 0.000000 |
| cp015_bestfree | True | cond | 0.219820 | 0.000000 | 0.000000 |
