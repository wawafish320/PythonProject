# Hidden-feature direct grad probe

- purpose: verify `direct_pose_feat_source=hidden` creates a real backbone gradient path, and `direct_pose_detach_feat=true` removes only that path
- direct-only loss: yes
- overall_pass: yes

| lane | family | detach_feat | feat_source | grad direct_pose_head | grad shared_encoder | grad contact_plan | pass |
|---|---|---|---|---:|---:|---:|---|
| old_hidden_gradon | old | False | hidden | 3.648584 | 0.188448 | 0.000000 | yes |
| old_hidden_gradoff | old | True | hidden | 7.750209 | 0.000000 | 0.000000 | yes |
| cp015_hidden_gradon | cp015 | False | hidden | 3.951730 | 0.265983 | 0.000000 | yes |
| cp015_hidden_gradoff | cp015 | True | hidden | 3.747896 | 0.000000 | 0.000000 | yes |

## Family checks

- `old`: gradon shared=0.188448; gradoff shared=0.000000; off/on=0.000000; head stays live=yes
- `cp015`: gradon shared=0.265983; gradoff shared=0.000000; off/on=0.000000; head stays live=yes

