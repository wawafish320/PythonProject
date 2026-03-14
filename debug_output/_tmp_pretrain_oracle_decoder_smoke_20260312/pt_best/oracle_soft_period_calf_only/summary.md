# oracle soft_period calf-only decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt`
- probe_id: `oracle_soft_period_calf_only`
- latent_kind: `soft_period`
- output_mode: `calf_only`
- seeds: 0
- overall_mean_deg_ex_root: 0.446419
- coverage: legs coverage 2/8; foot_l+ball_l coverage 0/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.224 | 0.000 | 0.452 | 1/1 |
| calf_l @ sic 78-85 | 0.334 | 0.000 | 0.682 | 1/1 |
| foot_l + ball_l @ sic 12-15 | NA | NA | NA | 0/2 |
| legs @ all sic | 0.446 | 0.000 | 0.878 | 2/8 |
| legs @ sic >= 40 | 0.366 | 0.000 | 0.784 | 2/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 638 | 0.305 | 0.008788 | 0.008788 | 0.000026 |
