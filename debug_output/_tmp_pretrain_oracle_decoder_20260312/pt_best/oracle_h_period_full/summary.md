# oracle h_period full decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt`
- probe_id: `oracle_h_period_full`
- latent_kind: `h_period`
- output_mode: `full`
- seeds: 0, 1, 2
- overall_mean_deg_ex_root: 0.307139
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.214 | 0.038 | 0.343 | 1/1 |
| calf_l @ sic 78-85 | 0.317 | 0.045 | 0.520 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 1.024 | 0.183 | 2.003 | 2/2 |
| legs @ all sic | 0.502 | 0.041 | 0.913 | 8/8 |
| legs @ sic >= 40 | 0.457 | 0.032 | 0.809 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 538 | 0.312 | 0.005951 | 0.005951 | 0.000012 |
| 1 | 581 | 0.323 | 0.005883 | 0.005883 | 0.000011 |
| 2 | 745 | 0.290 | 0.005002 | 0.005002 | 0.000008 |
