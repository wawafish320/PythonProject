# Raw yaw vs conditioning yaw (generated 2025-11-23)
## Root yaw statistics (raw JSON)
| clip | frames | yaw_min_deg | yaw_max_deg | delta_deg | mean_speed_mps | notes |
| --- | --- | --- | --- | --- | --- | --- |
| Walk_F.json | 88 | 0.00 | 0.00 | 0.00 | 0.689 | straight |
| Walk_L_To_L.json | 55 | -45.00 | 0.00 | -45.00 | 0.743 | turn |
| Walk_L_To_R.json | 51 | 0.00 | 45.00 | 45.00 | 0.659 | turn |
| Walk_R_To_L.json | 87 | -45.00 | 0.00 | -45.00 | 0.857 | turn |
| Walk_R_To_R.json | 94 | 0.00 | 45.00 | 45.00 | 0.787 | turn |

## Cond yaw vs root yaw (processed npz, using dataset-level offset)
| clip | offset_deg_used | median_abs_diff_deg | max_abs_diff_deg | yaw_cmd0_deg | root_yaw0_deg |
| --- | --- | --- | --- | --- | --- |
| Walk_F.npz | 22.05 | 157.95 | 157.95 | 157.95 | 0.00 |
| Walk_L_To_L.npz | 22.05 | 154.34 | 174.94 | 136.59 | 0.00 |
| Walk_L_To_R.npz | 22.05 | 114.18 | 148.37 | 116.16 | 0.00 |
| Walk_R_To_L.npz | 22.05 | 173.52 | 179.93 | -171.90 | 0.00 |
| Walk_R_To_R.npz | 22.05 | 146.92 | 174.28 | 173.92 | 0.00 |
