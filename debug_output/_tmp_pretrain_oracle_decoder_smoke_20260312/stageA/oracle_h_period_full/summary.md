# oracle h_period full decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv_stageA.pt`
- probe_id: `oracle_h_period_full`
- latent_kind: `h_period`
- output_mode: `full`
- seeds: 0
- overall_mean_deg_ex_root: 0.306793
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.378 | 0.000 | 0.646 | 1/1 |
| calf_l @ sic 78-85 | 0.382 | 0.000 | 0.562 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 1.839 | 0.000 | 4.196 | 2/2 |
| legs @ all sic | 0.570 | 0.000 | 1.077 | 8/8 |
| legs @ sic >= 40 | 0.539 | 0.000 | 0.989 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 788 | 0.415 | 0.005762 | 0.005762 | 0.000013 |
