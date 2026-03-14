# pretrain latent information audit / oracle decoder probe

## Setup
- clip: `/Users/xingzhaorui/PycharmProjects/PythonProject/raw_data/processed_data/Walk_F.npz`
- norm_spec: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/pretrain_template.json`
- device: `cpu`
- seeds: 0, 1, 2
- oracle decoder: hidden_dim=256, layers=3, lr=0.001
- overall verdict: `H1`

## Window compare

| bundle | probe | calf_r @ 56-62 | calf_l @ 78-85 | foot_l+ball_l @ 12-15 | legs @ all sic | legs @ sic>=40 | coverage |
|---|---|---:|---:|---:|---:|---:|---|
| pt_best | baseline_frozen | 27.529 +/- 0.000 | 21.207 +/- 0.000 | 12.924 +/- 0.000 | 10.194 +/- 0.000 | 10.175 +/- 0.000 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| pt_best | oracle_soft_period_full | 0.265 +/- 0.036 | 0.370 +/- 0.032 | 1.574 +/- 0.359 | 0.617 +/- 0.032 | 0.569 +/- 0.023 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| pt_best | oracle_soft_period_calf_only | 0.231 +/- 0.018 | 0.327 +/- 0.022 | NA +/- NA | 0.502 +/- 0.046 | 0.407 +/- 0.042 | legs coverage 2/8; foot_l+ball_l coverage 0/2 |
| pt_best | oracle_h_period_full | 0.214 +/- 0.038 | 0.317 +/- 0.045 | 1.024 +/- 0.183 | 0.502 +/- 0.041 | 0.457 +/- 0.032 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| stageA | baseline_frozen | 20.641 +/- 0.000 | 17.263 +/- 0.000 | 13.022 +/- 0.000 | 9.970 +/- 0.000 | 9.649 +/- 0.000 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| stageA | oracle_soft_period_full | 0.983 +/- 0.027 | 0.811 +/- 0.042 | 4.303 +/- 0.604 | 1.481 +/- 0.085 | 1.489 +/- 0.086 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| stageA | oracle_soft_period_calf_only | 0.902 +/- 0.040 | 0.763 +/- 0.044 | NA +/- NA | 2.272 +/- 0.133 | 2.230 +/- 0.054 | legs coverage 2/8; foot_l+ball_l coverage 0/2 |
| stageA | oracle_h_period_full | 0.308 +/- 0.049 | 0.383 +/- 0.009 | 1.783 +/- 0.117 | 0.562 +/- 0.022 | 0.529 +/- 0.020 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |

## Hypothesis interpretation

- `pt_best` -> `H1`: baseline hotspot mean=24.368deg; soft_period->full=0.318deg; soft_period->calf_only=0.279deg; h_period->full=0.266deg; soft_period full decoder already removes the calf hotspot, so latent information is present.
- `stageA` -> `H1`: baseline hotspot mean=18.952deg; soft_period->full=0.897deg; soft_period->calf_only=0.833deg; h_period->full=0.346deg; soft_period full decoder already removes the calf hotspot, so latent information is present; h_period still leaves some headroom, but it is secondary because soft_period is already sufficient.

## Decision rules used

- If `soft_period -> full decoder` already pulls the calf hotspot well below 15deg, treat that as strong H1 support.
- If `soft_period -> full decoder` is limited but `soft_period -> calf-only decoder` is much better, treat that as H3 support.
- If both `soft_period` probes stay weak while `h_period -> full decoder` is much better, treat that as H2 support.
- If even `h_period -> full decoder` stays weak, treat the issue as earlier than the period bottleneck or structurally ambiguous on the single clip.

## Notes

- This is a single-clip oracle extraction probe on `Walk_F`, so it measures recoverability from frozen latent, not cross-clip generalization.
- `oracle_soft_period_calf_only` only predicts `calf_l` and `calf_r`; its leg metrics therefore cover the calf overlap only, and `foot_l + ball_l` is NA by design.
- All oracle losses use `reproject_rot6d -> rot6d_to_matrix -> geodesic_R` as the main term, with a small auxiliary 6D MSE.
