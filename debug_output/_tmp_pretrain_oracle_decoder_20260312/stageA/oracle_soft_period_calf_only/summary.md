# oracle soft_period calf-only decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv_stageA.pt`
- probe_id: `oracle_soft_period_calf_only`
- latent_kind: `soft_period`
- output_mode: `calf_only`
- seeds: 0, 1, 2
- overall_mean_deg_ex_root: 2.272476
- coverage: legs coverage 2/8; foot_l+ball_l coverage 0/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.902 | 0.040 | 1.294 | 1/1 |
| calf_l @ sic 78-85 | 0.763 | 0.044 | 1.386 | 1/1 |
| foot_l + ball_l @ sic 12-15 | NA | NA | NA | 0/2 |
| legs @ all sic | 2.272 | 0.133 | 5.561 | 2/8 |
| legs @ sic >= 40 | 2.230 | 0.054 | 4.566 | 2/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 577 | 1.123 | 0.044662 | 0.044649 | 0.000662 |
| 1 | 790 | 1.118 | 0.039767 | 0.039756 | 0.000564 |
| 2 | 767 | 1.108 | 0.039443 | 0.039433 | 0.000503 |
