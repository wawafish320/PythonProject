# oracle soft_period full decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv.pt.best.pt`
- probe_id: `oracle_soft_period_full`
- latent_kind: `soft_period`
- output_mode: `full`
- seeds: 0
- overall_mean_deg_ex_root: 0.304402
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.316 | 0.000 | 0.492 | 1/1 |
| calf_l @ sic 78-85 | 0.346 | 0.000 | 0.669 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 1.224 | 0.000 | 2.587 | 2/2 |
| legs @ all sic | 0.573 | 0.000 | 1.081 | 8/8 |
| legs @ sic >= 40 | 0.543 | 0.000 | 0.996 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 775 | 0.376 | 0.004954 | 0.004954 | 0.000010 |
