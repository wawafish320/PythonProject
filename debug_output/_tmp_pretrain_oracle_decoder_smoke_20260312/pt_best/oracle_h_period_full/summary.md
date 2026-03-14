# oracle h_period full decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt`
- probe_id: `oracle_h_period_full`
- latent_kind: `h_period`
- output_mode: `full`
- seeds: 0
- overall_mean_deg_ex_root: 0.315307
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.163 | 0.000 | 0.221 | 1/1 |
| calf_l @ sic 78-85 | 0.366 | 0.000 | 0.582 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 1.030 | 0.000 | 1.990 | 2/2 |
| legs @ all sic | 0.521 | 0.000 | 0.963 | 8/8 |
| legs @ sic >= 40 | 0.478 | 0.000 | 0.811 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 538 | 0.312 | 0.005951 | 0.005951 | 0.000012 |
