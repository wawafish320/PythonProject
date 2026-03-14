# baseline frozen decoder

- bundle: `/Users/xingzhaorui/PycharmProjects/PythonProject/models/motion_encoder_equiv_stageA.pt`
- probe_id: `baseline_frozen`
- latent_kind: `soft_period`
- output_mode: `baseline`
- seeds: 0
- overall_mean_deg_ex_root: 6.353083
- coverage: legs coverage 8/8; foot_l+ball_l coverage 2/2

## Key windows

| window | mean_deg | std_deg | p90_deg | coverage |
|---|---:|---:|---:|---|
| calf_r @ sic 56-62 | 20.641 | 0.000 | 21.937 | 1/1 |
| calf_l @ sic 78-85 | 17.263 | 0.000 | 18.370 | 1/1 |
| foot_l + ball_l @ sic 12-15 | 13.022 | 0.000 | 24.936 | 2/2 |
| legs @ all sic | 9.970 | 0.000 | 20.406 | 8/8 |
| legs @ sic >= 40 | 9.649 | 0.000 | 20.284 | 8/8 |

## Seed runs

| seed | best_epoch | stop_metric_deg | train_loss | train_geodesic_rad | train_aux_mse |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 17.124 | NA | NA | NA |
