# oracle soft_period calf-only decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt`
- probe_id: `oracle_soft_period_calf_only`
- latent_kind: `soft_period`
- output_mode: `calf_only`
- seeds: 0, 1, 2
- overall_mean_deg_ex_root: 0.502327
- coverage: legs coverage 2/8; foot_l+ball_l coverage 0/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.231 | 0.018 | 0.467 | 1/1 |
| calf_l @ sic 78-85 | 0.327 | 0.022 | 0.700 | 1/1 |
| foot_l + ball_l @ sic 12-15 | NA | NA | NA | 0/2 |
| legs @ all sic | 0.502 | 0.046 | 0.934 | 2/8 |
| legs @ sic >= 40 | 0.407 | 0.042 | 0.808 | 2/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 638 | 0.305 | 0.008788 | 0.008788 | 0.000026 |
| 1 | 382 | 0.345 | 0.010482 | 0.010481 | 0.000034 |
| 2 | 456 | 0.294 | 0.009674 | 0.009673 | 0.000029 |
