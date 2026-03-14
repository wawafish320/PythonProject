# oracle soft_period full decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv_stageA.pt`
- probe_id: `oracle_soft_period_full`
- latent_kind: `soft_period`
- output_mode: `full`
- seeds: 0
- overall_mean_deg_ex_root: 0.585437
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.988 | 0.000 | 1.538 | 1/1 |
| calf_l @ sic 78-85 | 0.795 | 0.000 | 1.228 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 3.536 | 0.000 | 6.873 | 2/2 |
| legs @ all sic | 1.443 | 0.000 | 3.198 | 8/8 |
| legs @ sic >= 40 | 1.429 | 0.000 | 2.984 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 737 | 1.001 | 0.010016 | 0.010014 | 0.000065 |
