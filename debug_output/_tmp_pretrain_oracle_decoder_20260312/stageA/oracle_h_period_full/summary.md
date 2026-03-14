# oracle h_period full decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv_stageA.pt`
- probe_id: `oracle_h_period_full`
- latent_kind: `h_period`
- output_mode: `full`
- seeds: 0, 1, 2
- overall_mean_deg_ex_root: 0.313969
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.308 | 0.049 | 0.528 | 1/1 |
| calf_l @ sic 78-85 | 0.383 | 0.009 | 0.625 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 1.783 | 0.117 | 3.776 | 2/2 |
| legs @ all sic | 0.562 | 0.022 | 1.096 | 8/8 |
| legs @ sic >= 40 | 0.529 | 0.020 | 1.017 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 788 | 0.415 | 0.005762 | 0.005762 | 0.000013 |
| 1 | 797 | 0.369 | 0.005602 | 0.005602 | 0.000011 |
| 2 | 757 | 0.374 | 0.005571 | 0.005571 | 0.000013 |
