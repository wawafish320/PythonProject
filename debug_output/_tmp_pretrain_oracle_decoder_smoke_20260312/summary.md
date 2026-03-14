# pretrain latent information audit / oracle decoder probe

## Setup
- clip: `/Users/xingzhaorui/PycharmProjects/PythonProject/raw_data/processed_data/Walk_F.npz`
- norm_spec: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/pretrain_template.json`
- device: `cpu`
- seeds: 0
- oracle decoder: hidden_dim=256, layers=3, lr=0.001
- overall verdict: `H1`

## Window compare

| bundle | probe | calf_r @ 56-62 | calf_l @ 78-85 | foot_l+ball_l @ 12-15 | legs @ all sic | legs @ sic>=40 | coverage |
|---|---|---:|---:|---:|---:|---:|---|
| pt_best | baseline_frozen | 27.529 +/- 0.000 | 21.207 +/- 0.000 | 12.924 +/- 0.000 | 10.194 +/- 0.000 | 10.175 +/- 0.000 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| pt_best | oracle_soft_period_full | 0.316 +/- 0.000 | 0.346 +/- 0.000 | 1.224 +/- 0.000 | 0.573 +/- 0.000 | 0.543 +/- 0.000 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| pt_best | oracle_soft_period_calf_only | 0.224 +/- 0.000 | 0.334 +/- 0.000 | NA +/- NA | 0.446 +/- 0.000 | 0.366 +/- 0.000 | legs coverage 2/8; foot_l+ball_l coverage 0/2 |
| pt_best | oracle_h_period_full | 0.163 +/- 0.000 | 0.366 +/- 0.000 | 1.030 +/- 0.000 | 0.521 +/- 0.000 | 0.478 +/- 0.000 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| stageA | baseline_frozen | 20.641 +/- 0.000 | 17.263 +/- 0.000 | 13.022 +/- 0.000 | 9.970 +/- 0.000 | 9.649 +/- 0.000 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| stageA | oracle_soft_period_full | 0.988 +/- 0.000 | 0.795 +/- 0.000 | 3.536 +/- 0.000 | 1.443 +/- 0.000 | 1.429 +/- 0.000 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |
| stageA | oracle_soft_period_calf_only | 0.861 +/- 0.000 | 0.769 +/- 0.000 | NA +/- NA | 2.457 +/- 0.000 | 2.256 +/- 0.000 | legs coverage 2/8; foot_l+ball_l coverage 0/2 |
| stageA | oracle_h_period_full | 0.378 +/- 0.000 | 0.382 +/- 0.000 | 1.839 +/- 0.000 | 0.570 +/- 0.000 | 0.539 +/- 0.000 | legs coverage 8/8; foot_l+ball_l coverage 2/2 |

## Hypothesis interpretation

- `pt_best` -> `H1`: baseline hotspot mean=24.368deg; soft_period->full=0.331deg; soft_period->calf_only=0.279deg; h_period->full=0.265deg; soft_period full decoder already removes the calf hotspot, so latent information is present.
- `stageA` -> `H1`: baseline hotspot mean=18.952deg; soft_period->full=0.892deg; soft_period->calf_only=0.815deg; h_period->full=0.380deg; soft_period full decoder already removes the calf hotspot, so latent information is present; h_period still leaves some headroom, but it is secondary because soft_period is already sufficient.

## Decision rules used

- If `soft_period -> full decoder` already pulls the calf hotspot well below 15deg, treat that as strong H1 support.
- If `soft_period -> full decoder` is limited but `soft_period -> calf-only decoder` is much better, treat that as H3 support.
- If both `soft_period` probes stay weak while `h_period -> full decoder` is much better, treat that as H2 support.
- If even `h_period -> full decoder` stays weak, treat the issue as earlier than the period bottleneck or structurally ambiguous on the single clip.

## Notes

- This is a single-clip oracle extraction probe on `Walk_F`, so it measures recoverability from frozen latent, not cross-clip generalization.
- `oracle_soft_period_calf_only` only predicts `calf_l` and `calf_r`; its leg metrics therefore cover the calf overlap only, and `foot_l + ball_l` is NA by design.
- All oracle losses use `reproject_rot6d -> rot6d_to_matrix -> geodesic_R` as the main term, with a small auxiliary 6D MSE.
