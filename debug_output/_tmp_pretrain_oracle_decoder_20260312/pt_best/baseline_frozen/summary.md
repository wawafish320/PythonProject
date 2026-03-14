# baseline frozen decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt`
- probe_id: `baseline_frozen`
- latent_kind: `soft_period`
- output_mode: `baseline`
- seeds: 0
- overall_mean_deg_ex_root: 6.595989
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 27.529 | 0.000 | 28.865 | 1/1 |
| calf_l @ sic 78-85 | 21.207 | 0.000 | 22.551 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 12.924 | 0.000 | 24.436 | 2/2 |
| legs @ all sic | 10.194 | 0.000 | 21.733 | 8/8 |
| legs @ sic >= 40 | 10.175 | 0.000 | 23.236 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 21.531 | NA | NA | NA |
