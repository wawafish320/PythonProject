# oracle soft_period full decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt`
- probe_id: `oracle_soft_period_full`
- latent_kind: `soft_period`
- output_mode: `full`
- seeds: 0, 1, 2
- overall_mean_deg_ex_root: 0.333261
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.265 | 0.036 | 0.443 | 1/1 |
| calf_l @ sic 78-85 | 0.370 | 0.032 | 0.716 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 1.574 | 0.359 | 3.075 | 2/2 |
| legs @ all sic | 0.617 | 0.032 | 1.166 | 8/8 |
| legs @ sic >= 40 | 0.569 | 0.023 | 1.027 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 775 | 0.376 | 0.004954 | 0.004954 | 0.000010 |
| 1 | 605 | 0.383 | 0.005903 | 0.005903 | 0.000013 |
| 2 | 636 | 0.359 | 0.006030 | 0.006030 | 0.000014 |
