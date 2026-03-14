# oracle soft_period full decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv_stageA.pt`
- probe_id: `oracle_soft_period_full`
- latent_kind: `soft_period`
- output_mode: `full`
- seeds: 0, 1, 2
- overall_mean_deg_ex_root: 0.601154
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.983 | 0.027 | 1.757 | 1/1 |
| calf_l @ sic 78-85 | 0.811 | 0.042 | 1.399 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 4.303 | 0.604 | 9.532 | 2/2 |
| legs @ all sic | 1.481 | 0.085 | 3.222 | 8/8 |
| legs @ sic >= 40 | 1.489 | 0.086 | 3.121 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 737 | 1.001 | 0.010016 | 0.010014 | 0.000065 |
| 1 | 695 | 1.008 | 0.012152 | 0.012150 | 0.000089 |
| 2 | 776 | 1.036 | 0.010540 | 0.010539 | 0.000071 |
