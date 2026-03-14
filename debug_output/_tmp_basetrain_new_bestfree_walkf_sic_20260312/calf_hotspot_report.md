# Walk_F / Stage7: SIC hotspots vs GT knee angular-velocity overlay

Date: 2026-03-12

Goal: relate phase-locked SO(3) error bias at specific sic to GT angular-velocity peaks/sign flips.

## Inputs

- freerun json: `debug_output/_tmp_basetrain_new_bestfree_walkf_sic_20260312/Walk_F_freerun_cycles.json`
- npz: `raw_data/processed_data/Walk_F.npz` (FPS=60.0)
- branch/space: `direct` / `body`

## Mask / protocol

- cycle >= 1
- exclude_wrap = True
- exclude_root = False (root_idx=0)
- projection_diag = True
- projection gate: sign(mu*omega)>0, ||mu||<5.0 deg, ||omega||>=30.0 deg/s

## Figure

- png: `debug_output/_tmp_basetrain_new_bestfree_walkf_sic_20260312/calf_hotspot_report.png`
- projection scatter(s):
  - calf_l: `debug_output/_tmp_basetrain_new_bestfree_walkf_sic_20260312/calf_hotspot_report_projection/calf_l_mu_parallel_vs_omega.png`
  - calf_r: `debug_output/_tmp_basetrain_new_bestfree_walkf_sic_20260312/calf_hotspot_report_projection/calf_r_mu_parallel_vs_omega.png`

## Quick summary

Thresholds: `||mu_sic|| >= 0.500 deg`, `|omega_z| >= 30.0 deg/s`.

- calf_l:
  - sign(mu_z * omega_z) > 0 fraction = 0.482 (N_mu=85)
  - dt_frames = (mu_z/omega_z) * FPS: median=1.919, IQR=[-3.747, 9.452] (N_dt=61)
  - proj_gate(sign(mu*omega)>0, ||mu||<5.0deg, ||omega||>=30.0deg/s): N=4/86 (0.047)
  - dt* = (mu*omega/||omega||^2)*FPS: median=4.546, IQR=[3.435, 4.859] (N=4)
  - r_perp = ||mu_perp||/||mu||: median=0.000, p90=0.000
  - mu_parallel_scalar = mu*omega_hat: median=3.495 deg, std=0.862
  - H1/H2 fit (mu_parallel=a+b*||omega||): a=4.2457, b=-0.0071, R^2=0.296 (N=4)
- calf_r:
  - sign(mu_z * omega_z) > 0 fraction = 0.500 (N_mu=86)
  - dt_frames = (mu_z/omega_z) * FPS: median=-2.937, IQR=[-11.712, 4.745] (N_dt=52)
  - proj_gate(sign(mu*omega)>0, ||mu||<5.0deg, ||omega||>=30.0deg/s): N=2/86 (0.023)
  - dt* = (mu*omega/||omega||^2)*FPS: median=0.434, IQR=[0.383, 0.484] (N=2)
  - r_perp = ||mu_perp||/||mu||: median=0.000, p90=0.000
  - mu_parallel_scalar = mu*omega_hat: median=1.743 deg, std=0.263
  - H1/H2 fit (mu_parallel=a+b*||omega||): a=4.8204, b=-0.0125, R^2=1.000 (N=2)

Interpretation note: if the small-angle approximation `mu ~= omega * dt` holds, then `dt_frames>0` suggests phase-lag (pred behind GT along motion direction), while `dt_frames<0` suggests phase-lead / opposite-sign bias.
Projection note: `dt*` is only interpreted on the gated subset; when `r_perp(median)` is high, most error energy is orthogonal to omega and lag interpretation is weak.

## GT symmetry sanity (data-only)

- Method: synthetic lag on GT (`pred(t)=gt(t-k)` with edge padding), then apply the same dt estimator.
- thresholds: `||mu||>=0.500 deg`, `|omega_z|>=30.0 deg/s`
- joints: `calf_l,calf_r`

|k (frames)|calf_l dt_med|calf_r dt_med|calf_l align|calf_r align|same_sign|
|---:|---:|---:|---:|---:|:---:|
|1|0.999|1.004|1.000|1.000|Y|
|2|1.919|1.942|0.956|0.973|Y|
|3|2.736|2.766|0.918|0.962|Y|
|-1|-1.022|-1.009|0.000|0.000|Y|
|-2|-2.031|-1.962|0.000|0.014|Y|
|-3|-3.081|-2.872|0.027|0.026|Y|
- mean(|omega_z|) calf_l = 104.091 deg/s
- mean(|omega_z|) calf_r = 91.256 deg/s
- best corr(omega_calf_l, roll(omega_calf_r, s)) = 0.901 at s=45 (cycle len=88)

## Projection diagnostics summary (gated subset)

|joint|N_gate/N_valid|dt* median (IQR)|r_perp median/p90|mu_parallel median+/-std (deg)|fit a|fit b|R^2|reliability|
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
|calf_l|4/86|4.546 [3.435, 4.859]|0.000/0.000|3.495+/-0.862|4.2457|-0.0071|0.296|usable|
|calf_r|2/86|0.434 [0.383, 0.484]|0.000/0.000|1.743+/-0.263|4.8204|-0.0125|1.000|usable|

## Global SIC hotspots (Top 15 by max_j ||mu_sic,j||)

|rank|sic|N|mean||mu|| (deg)|max||mu|| (deg)|top_joint|top mu_xyz (deg)|dom|
|---:|---:|---:|---:|---:|:---|:---|:---:|
|1|46|4|9.017|98.312|pinky_01_l|`[-96.603, -8.516, -16.140]`|x-|
|2|45|4|8.799|92.326|pinky_01_l|`[-90.709, -7.371, -15.544]`|x-|
|3|5|4|10.101|92.314|pinky_01_l|`[-90.696, -7.374, -15.546]`|x-|
|4|47|4|8.524|88.795|pinky_01_l|`[-87.224, -6.741, -15.203]`|x-|
|5|48|4|8.182|82.727|pinky_01_l|`[-81.219, -5.743, -14.636]`|x-|
|6|6|4|9.214|78.530|pinky_01_l|`[-77.057, -5.104, -14.251]`|x-|
|7|44|4|8.270|78.403|pinky_01_l|`[-76.931, -5.085, -14.240]`|x-|
|8|4|4|9.440|77.741|pinky_01_l|`[-76.270, -5.000, -14.195]`|x-|
|9|3|4|9.506|76.616|pinky_01_l|`[-75.151, -4.844, -14.105]`|x-|
|10|9|4|8.753|66.316|pinky_01_l|`[-64.917, -3.431, -13.104]`|x-|
|11|49|4|7.279|62.984|pinky_01_l|`[-61.588, -3.052, -12.827]`|x-|
|12|7|4|8.268|60.048|pinky_01_l|`[-58.665, -2.713, -12.524]`|x-|
|13|8|4|8.323|59.354|pinky_01_l|`[-57.975, -2.632, -12.448]`|x-|
|14|2|4|8.507|57.238|pinky_01_l|`[-55.848, -2.433, -12.297]`|x-|
|15|43|4|7.484|57.191|pinky_01_l|`[-55.812, -2.406, -12.248]`|x-|

## Per-joint SIC tables (mu vs omega)

### calf_l

- corr(mu_z, omega_z) = 0.108
- corr(||mu||, |omega|) = 0.032
- lag estimate uses |omega_z| >= 30.0 deg/s; dt_frames = (mu_z/omega_z) * FPS

|rank|sic|||mu|| (deg)|mu_axis (deg)|mu_xyz (deg)|mu dom|omega_axis (deg/s)||omega| (deg/s)|dt (frames)|dt* (frames)|r_perp|mu_parallel (deg)|sign(mu_axis*omega_axis)|
|---:|---:|---:|---:|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
|1|18|42.306|42.306|`[-0.000,  0.000,  42.306]`|z+|25.5|25.5|NA|99.496|0.000|42.306|+|
|2|17|42.075|42.075|`[-0.000,  0.000,  42.075]`|z+|58.7|58.7|43.031|43.031|0.000|42.075|+|
|3|19|41.918|41.918|`[-0.000,  0.000,  41.918]`|z+|-7.9|7.9|NA|-319.442|0.000|-41.918|-|
|4|16|41.359|41.359|`[-0.000,  0.000,  41.359]`|z+|99.9|99.9|24.848|24.848|0.000|41.359|+|
|5|20|40.964|40.964|`[-0.000,  0.000,  40.964]`|z+|-42.9|42.9|-57.256|-57.256|0.000|-40.964|-|
|6|15|40.046|40.046|`[-0.000,  0.000,  40.046]`|z+|140.6|140.6|17.084|17.084|0.000|40.046|+|
|7|21|39.444|39.444|`[-0.000,  0.000,  39.444]`|z+|-76.0|76.0|-31.138|-31.138|0.000|-39.444|-|
|8|14|37.942|37.942|`[-0.000,  0.000,  37.942]`|z+|179.2|179.2|12.701|12.701|0.000|37.942|+|
|9|22|37.351|37.351|`[-0.000,  0.000,  37.351]`|z+|-109.4|109.4|-20.488|-20.488|0.000|-37.351|-|
|10|13|35.248|35.248|`[-0.000,  0.000,  35.248]`|z+|223.7|223.7|9.452|9.452|0.000|35.248|+|
|11|23|34.626|34.626|`[-0.000,  0.000,  34.626]`|z+|-145.1|145.1|-14.322|-14.322|0.000|-34.626|-|
|12|12|31.802|31.802|`[-0.000,  0.000,  31.802]`|z+|250.6|250.6|7.613|7.613|0.000|31.802|+|
|13|24|31.356|31.356|`[-0.000,  0.000,  31.356]`|z+|-178.6|178.6|-10.537|-10.537|0.000|-31.356|-|
|14|11|27.574|27.574|`[-0.000,  0.000,  27.574]`|z+|281.7|281.7|5.873|5.873|0.000|27.574|+|
|15|25|27.358|27.358|`[-0.000,  0.000,  27.358]`|z+|-216.5|216.5|-7.582|-7.582|0.000|-27.358|-|

### calf_r

- corr(mu_z, omega_z) = 0.085
- corr(||mu||, |omega|) = -0.082
- lag estimate uses |omega_z| >= 30.0 deg/s; dt_frames = (mu_z/omega_z) * FPS

|rank|sic|||mu|| (deg)|mu_axis (deg)|mu_xyz (deg)|mu dom|omega_axis (deg/s)||omega| (deg/s)|dt (frames)|dt* (frames)|r_perp|mu_parallel (deg)|sign(mu_axis*omega_axis)|
|---:|---:|---:|---:|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
|1|82|31.934|-31.934|`[-0.000, -0.000, -31.934]`|z-|-55.0|55.0|34.863|34.863|0.000|31.934|+|
|2|81|31.376|-31.376|`[-0.000, -0.000, -31.376]`|z-|-88.1|88.1|21.369|21.369|0.000|31.376|+|
|3|83|30.670|-30.670|`[-0.000, -0.000, -30.670]`|z-|19.8|19.8|NA|-92.818|0.000|-30.670|-|
|4|80|29.856|-29.856|`[-0.000, -0.000, -29.856]`|z-|-103.2|103.2|17.356|17.356|0.000|29.856|+|
|5|59|29.146|29.146|`[-0.000,  0.000,  29.146]`|z+|17.4|17.4|NA|100.479|0.000|29.146|+|
|6|84|28.953|-28.953|`[-0.000, -0.000, -28.953]`|z-|79.8|79.8|-21.769|-21.769|0.000|-28.953|-|
|7|60|28.745|28.745|`[-0.000,  0.000,  28.745]`|z+|-13.0|13.0|NA|-132.647|0.000|-28.745|-|
|8|58|28.657|28.657|`[-0.000,  0.000,  28.657]`|z+|44.6|44.6|38.538|38.538|0.000|28.657|+|
|9|79|28.226|-28.226|`[-0.000, -0.000, -28.226]`|z-|-122.2|122.2|13.864|13.864|0.000|28.226|+|
|10|57|28.074|28.074|`[-0.000,  0.000,  28.074]`|z+|68.5|68.5|24.604|24.604|0.000|28.074|+|
|11|61|27.964|27.964|`[-0.000,  0.000,  27.964]`|z+|-38.2|38.2|-43.960|-43.960|0.000|-27.964|-|
|12|85|27.212|-27.212|`[-0.000, -0.000, -27.212]`|z-|97.4|97.4|-16.761|-16.761|0.000|-27.212|-|
|13|62|26.645|26.645|`[-0.000,  0.000,  26.645]`|z+|-62.8|62.8|-25.472|-25.472|0.000|-26.645|-|
|14|78|26.346|-26.346|`[-0.000, -0.000, -26.346]`|z-|-144.2|144.2|10.960|10.960|0.000|26.346|+|
|15|56|26.318|26.318|`[ 0.000,  0.000,  26.318]`|z+|97.9|97.9|16.127|16.127|0.000|26.318|+|

