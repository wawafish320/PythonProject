# oracle soft_period calf-only decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv_stageA.pt`
- probe_id: `oracle_soft_period_calf_only`
- latent_kind: `soft_period`
- output_mode: `calf_only`
- seeds: 0
- overall_mean_deg_ex_root: 2.457200
- coverage: legs coverage 2/8; foot_l+ball_l coverage 0/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 0.861 | 0.000 | 1.240 | 1/1 |
| calf_l @ sic 78-85 | 0.769 | 0.000 | 1.483 | 1/1 |
| foot_l + ball_l @ sic 12-15 | NA | NA | NA | 0/2 |
| legs @ all sic | 2.457 | 0.000 | 6.048 | 2/8 |
| legs @ sic >= 40 | 2.256 | 0.000 | 4.770 | 2/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 577 | 1.123 | 0.044662 | 0.044649 | 0.000662 |
