# Walk_F / Stage7: SIC hotspots vs GT knee angular-velocity overlay

Date: 2026-03-12

Goal: relate phase-locked SO(3) error bias at specific sic to GT angular-velocity peaks/sign flips.

## Inputs

- freerun json: `debug_output/_tmp_basetrain_old_bestfree_walkf_sic_20260312/Walk_F_freerun_cycles.json`
- npz: `raw_data/processed_data/Walk_F.npz` (FPS=60.0)
- branch/space: `direct` / `body`

## Mask / protocol

- cycle >= 1
- exclude_wrap = True
- exclude_root = False (root_idx=0)
- projection_diag = True
- projection gate: sign(mu*omega)>0, ||mu||<5.0 deg, ||omega||>=30.0 deg/s

## Figure

- png: `debug_output/_tmp_basetrain_old_bestfree_walkf_sic_20260312/calf_hotspot_report.png`
- projection scatter(s):
  - calf_l: `debug_output/_tmp_basetrain_old_bestfree_walkf_sic_20260312/calf_hotspot_report_projection/calf_l_mu_parallel_vs_omega.png`
  - calf_r: `debug_output/_tmp_basetrain_old_bestfree_walkf_sic_20260312/calf_hotspot_report_projection/calf_r_mu_parallel_vs_omega.png`

## Quick summary

Thresholds: `||mu_sic|| >= 0.500 deg`, `|omega_z| >= 30.0 deg/s`.

- calf_l:
  - sign(mu_z * omega_z) > 0 fraction = 0.482 (N_mu=85)
  - dt_frames = (mu_z/omega_z) * FPS: median=2.084, IQR=[-3.544, 9.343] (N_dt=61)
  - proj_gate(sign(mu*omega)>0, ||mu||<5.0deg, ||omega||>=30.0deg/s): N=4/86 (0.047)
  - dt* = (mu*omega/||omega||^2)*FPS: median=2.937, IQR=[0.638, 5.408] (N=4)
  - r_perp = ||mu_perp||/||mu||: median=0.000, p90=0.000
  - mu_parallel_scalar = mu*omega_hat: median=3.299 deg, std=1.583
  - H1/H2 fit (mu_parallel=a+b*||omega||): a=4.8529, b=-0.0121, R^2=0.922 (N=4)
- calf_r:
  - sign(mu_z * omega_z) > 0 fraction = 0.500 (N_mu=86)
  - dt_frames = (mu_z/omega_z) * FPS: median=-2.043, IQR=[-7.486, 4.703] (N_dt=52)
  - proj_gate(sign(mu*omega)>0, ||mu||<5.0deg, ||omega||>=30.0deg/s): N=2/86 (0.023)
  - dt* = (mu*omega/||omega||^2)*FPS: median=0.846, IQR=[0.812, 0.880] (N=2)
  - r_perp = ||mu_perp||/||mu||: median=0.000, p90=0.000
  - mu_parallel_scalar = mu*omega_hat: median=2.858 deg, std=0.616
  - H1/H2 fit (mu_parallel=a+b*||omega||): a=0.7367, b=0.0102, R^2=1.000 (N=2)

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
|calf_l|4/86|2.937 [0.638, 5.408]|0.000/0.000|3.299+/-1.583|4.8529|-0.0121|0.922|usable|
|calf_r|2/86|0.846 [0.812, 0.880]|0.000/0.000|2.858+/-0.616|0.7367|0.0102|1.000|usable|

## Global SIC hotspots (Top 15 by max_j ||mu_sic,j||)

|rank|sic|N|mean||mu|| (deg)|max||mu|| (deg)|top_joint|top mu_xyz (deg)|dom|
|---:|---:|---:|---:|---:|:---|:---|:---:|
|1|46|4|9.712|111.378|pinky_01_l|`[-108.969, -14.489, -17.914]`|x-|
|2|5|4|11.535|109.368|pinky_01_l|`[-107.092, -13.737, -17.434]`|x-|
|3|45|4|9.512|109.219|pinky_01_l|`[-106.949, -13.683, -17.416]`|x-|
|4|47|4|9.401|107.321|pinky_01_l|`[-105.151, -13.034, -17.063]`|x-|
|5|48|4|9.282|104.622|pinky_01_l|`[-102.585, -12.159, -16.558]`|x-|
|6|44|4|9.029|103.712|pinky_01_l|`[-101.733, -11.858, -16.307]`|x-|
|7|4|4|11.108|103.600|pinky_01_l|`[-101.623, -11.834, -16.297]`|x-|
|8|3|4|11.223|103.404|pinky_01_l|`[-101.437, -11.773, -16.255]`|x-|
|9|6|4|10.813|103.392|pinky_01_l|`[-101.421, -11.774, -16.279]`|x-|
|10|9|4|10.350|96.894|pinky_01_l|`[-95.171, -9.989, -15.203]`|x-|
|11|49|4|8.633|94.841|pinky_01_l|`[-93.168, -9.488, -14.978]`|x-|
|12|2|4|10.470|93.955|pinky_01_l|`[-92.327, -9.283, -14.736]`|x-|
|13|7|4|10.008|93.866|pinky_01_l|`[-92.235, -9.261, -14.755]`|x-|
|14|8|4|10.027|93.254|pinky_01_l|`[-91.641, -9.121, -14.667]`|x-|
|15|43|4|8.297|93.231|pinky_01_l|`[-91.628, -9.109, -14.605]`|x-|

## Per-joint SIC tables (mu vs omega)

### calf_l

- corr(mu_z, omega_z) = 0.120
- corr(||mu||, |omega|) = -0.002
- lag estimate uses |omega_z| >= 30.0 deg/s; dt_frames = (mu_z/omega_z) * FPS

|rank|sic|||mu|| (deg)|mu_axis (deg)|mu_xyz (deg)|mu dom|omega_axis (deg/s)||omega| (deg/s)|dt (frames)|dt* (frames)|r_perp|mu_parallel (deg)|sign(mu_axis*omega_axis)|
|---:|---:|---:|---:|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
|1|18|41.378|41.378|`[-0.000, -0.000,  41.378]`|z+|25.5|25.5|NA|97.312|0.000|41.378|+|
|2|17|41.214|41.214|`[-0.000, -0.000,  41.214]`|z+|58.7|58.7|42.150|42.150|0.000|41.214|+|
|3|19|40.942|40.942|`[-0.000, -0.000,  40.942]`|z+|-7.9|7.9|NA|-312.004|0.000|-40.942|-|
|4|16|40.544|40.544|`[-0.000, -0.000,  40.544]`|z+|99.9|99.9|24.358|24.358|0.000|40.544|+|
|5|20|39.942|39.942|`[-0.000, -0.000,  39.942]`|z+|-42.9|42.9|-55.827|-55.827|0.000|-39.942|-|
|6|15|39.276|39.276|`[-0.000, -0.000,  39.276]`|z+|140.6|140.6|16.756|16.756|0.000|39.276|+|
|7|21|38.405|38.405|`[-0.000, -0.000,  38.405]`|z+|-76.0|76.0|-30.318|-30.318|0.000|-38.405|-|
|8|14|37.195|37.195|`[-0.000, -0.000,  37.195]`|z+|179.2|179.2|12.451|12.451|0.000|37.195|+|
|9|22|36.292|36.292|`[-0.000, -0.000,  36.292]`|z+|-109.4|109.4|-19.907|-19.907|0.000|-36.292|-|
|10|13|34.517|34.517|`[-0.000, -0.000,  34.517]`|z+|223.7|223.7|9.256|9.256|0.000|34.517|+|
|11|23|33.563|33.563|`[-0.000, -0.000,  33.563]`|z+|-145.1|145.1|-13.882|-13.882|0.000|-33.563|-|
|12|12|31.075|31.075|`[-0.000, -0.000,  31.075]`|z+|250.6|250.6|7.439|7.439|0.000|31.075|+|
|13|24|30.321|30.321|`[-0.000, -0.000,  30.321]`|z+|-178.6|178.6|-10.189|-10.189|0.000|-30.321|-|
|14|11|26.741|26.741|`[-0.000, -0.000,  26.741]`|z+|281.7|281.7|5.696|5.696|0.000|26.741|+|
|15|25|26.330|26.330|`[-0.000, -0.000,  26.330]`|z+|-216.5|216.5|-7.297|-7.297|0.000|-26.330|-|

### calf_r

- corr(mu_z, omega_z) = 0.105
- corr(||mu||, |omega|) = 0.214
- lag estimate uses |omega_z| >= 30.0 deg/s; dt_frames = (mu_z/omega_z) * FPS

|rank|sic|||mu|| (deg)|mu_axis (deg)|mu_xyz (deg)|mu dom|omega_axis (deg/s)||omega| (deg/s)|dt (frames)|dt* (frames)|r_perp|mu_parallel (deg)|sign(mu_axis*omega_axis)|
|---:|---:|---:|---:|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
|1|59|35.926|35.926|`[ 0.000, -0.000,  35.926]`|z+|17.4|17.4|NA|123.854|0.000|35.926|+|
|2|58|35.701|35.701|`[ 0.000, -0.000,  35.701]`|z+|44.6|44.6|48.010|48.010|0.000|35.701|+|
|3|60|35.518|35.518|`[ 0.000, -0.000,  35.518]`|z+|-13.0|13.0|NA|-163.900|0.000|-35.518|-|
|4|57|35.113|35.113|`[ 0.000, -0.000,  35.113]`|z+|68.5|68.5|30.772|30.772|0.000|35.113|+|
|5|61|34.737|34.737|`[ 0.000, -0.000,  34.737]`|z+|-38.2|38.2|-54.608|-54.608|0.000|-34.737|-|
|6|56|33.607|33.607|`[ 0.000, -0.000,  33.607]`|z+|97.9|97.9|20.593|20.593|0.000|33.607|+|
|7|62|33.477|33.477|`[ 0.000, -0.000,  33.477]`|z+|-62.8|62.8|-32.003|-32.003|0.000|-33.477|-|
|8|55|32.153|32.153|`[ 0.000, -0.000,  32.153]`|z+|139.9|139.9|13.786|13.786|0.000|32.153|+|
|9|63|31.850|31.850|`[ 0.000, -0.000,  31.850]`|z+|-87.7|87.7|-21.798|-21.798|0.000|-31.850|-|
|10|54|30.414|30.414|`[ 0.000, -0.000,  30.414]`|z+|194.1|194.1|9.402|9.402|0.000|30.414|+|
|11|64|29.840|29.840|`[ 0.000, -0.000,  29.840]`|z+|-109.4|109.4|-16.370|-16.370|0.000|-29.840|-|
|12|53|27.638|27.638|`[ 0.000, -0.000,  27.638]`|z+|240.8|240.8|6.886|6.886|0.000|27.638|+|
|13|65|27.541|27.541|`[ 0.000, -0.000,  27.541]`|z+|-130.2|130.2|-12.690|-12.690|0.000|-27.541|-|
|14|66|24.791|24.791|`[ 0.000, -0.000,  24.791]`|z+|-151.8|151.8|-9.800|-9.800|0.000|-24.791|-|
|15|82|24.660|-24.660|`[ 0.000, -0.000, -24.660]`|z-|-55.0|55.0|26.921|26.921|0.000|24.660|+|

