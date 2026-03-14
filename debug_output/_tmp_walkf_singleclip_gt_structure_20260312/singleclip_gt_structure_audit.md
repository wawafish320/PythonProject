# Walk_F single-clip GT structure / ambiguity audit

Date: 2026-03-12

## Scope and caution

- Inputs: `raw_data/processed_data/Walk_F.npz` and `raw_data/Walk_F.json`.
- GT-only audit: only `bone_ang_vel`, no prediction, no model change, no training.
- This is **not** a GT variance audit. `Walk_F` is a single natural cycle, so the result can only be interpreted as a single-clip structural ambiguity / sharpness audit.
- Main signal: local/body-space `omega_z`; `||omega||` was checked only as a support signal.

## Data confirmation

- Raw JSON: `NumFrames=88`, `len(Frames)=88`.
- Processed NPZ: `bone_ang_vel.shape=(88, 46, 3)`, `bone_rot6d.shape=(88, 46, 6)`.
- There is no extra sample/cycle axis in either file; the clip is a single flat time axis with SIC treated as frame index `0..87`.
- Cycle closure check (frame 0 vs 87, local pose): mean=0.016 deg, median=0.001 deg, max=0.148 deg.
- Mid-cycle contrast (frame 0 vs 44): mean=6.375 deg, median=2.755 deg, max=35.547 deg.
- Root displacement over the clip: `dx,dy,dz = [-0.999, -0.000, 0.000]` m.
- Interpretation: start/end local pose nearly closes while root translates forward, which is consistent with one locomotion cycle rather than a multi-cycle stack.

## omega_z sufficiency check

- `calf_l`: on non-zero samples, `|omega_z| / ||omega||` min/mean/max = 1.000 / 1.000 / 1.000; the motion is effectively pure z-axis rotation in these windows.
- `calf_r`: on non-zero samples, `|omega_z| / ||omega||` min/mean/max = 1.000 / 1.000 / 1.000; the motion is effectively pure z-axis rotation in these windows.

## Outputs

- `debug_output/_tmp_walkf_singleclip_gt_structure_20260312/calf_l_gt_omega_z.png`
- `debug_output/_tmp_walkf_singleclip_gt_structure_20260312/calf_r_gt_omega_z.png`
- `debug_output/_tmp_walkf_singleclip_gt_structure_20260312/singleclip_gt_structure_audit.md`
- `debug_output/_tmp_walkf_singleclip_gt_structure_20260312/singleclip_gt_structure_audit_metrics.json`

## Window diagnostics

## calf_l @ sic=78-85

- Figure: `debug_output/_tmp_walkf_singleclip_gt_structure_20260312/calf_l_gt_omega_z.png`
- Context shown: sic=72-88 (data available through sic=87).
- GT read: `broad / phase-unfriendly`; lean: `more like ambiguity / observability`.

|metric|value|
|---|---|
|max positive peak|34.113 deg/s at sic=80; prom=20.580; half-prom width=4.892 sic|
|max negative peak|none (window never goes negative)|
|zero-crossing in window|no zero-crossing inside window; nearest context event is zero plateau sic=74-77 (entry slope=77.526, exit slope=1.187)|
|omega_z dynamic range|32.926 deg/s|
|d omega_z / d sic stats|median=4.216, mean=8.527, p90=18.380, max=20.580|
|curvature (d^2 omega_z / d sic^2)|median=7.770, mean=10.057, p90=19.741, max=22.844|
|plateau test (>=95% / >=90% of dominant)|positive: 80-80 (len=1) / 80-81 (len=2), 83-83 (len=1); negative: none / none|
|multi-peak competition|positive: secondary pos bump exists, but prominence ratio is only 0.150; negative: no neg peak in window|

Qualitative read:
- The main window is a low-to-moderate positive shoulder after an adjacent zero plateau (context sic=74-77), not an isolated sharp spike.
- There is a dominant positive peak at sic=80, but its half-prominence width is broad and a secondary bump around sic=83 behaves more like a shoulder than a clean second mode.
- Because the onset emerges out of several exact-zero samples and the immediate exit slope from the zero plateau is weak, this window is comparatively phase-unfriendly.

## calf_r @ sic=56-62

- Figure: `debug_output/_tmp_walkf_singleclip_gt_structure_20260312/calf_r_gt_omega_z.png`
- Context shown: sic=50-68 (data available through sic=68).
- GT read: `sharp / deterministic`; lean: `more like capacity / temporal-resolution`.

|metric|value|
|---|---|
|max positive peak|97.917 deg/s at sic=56; prom=160.680; half-prom width=2.994 sic (edge-clipped)|
|max negative peak|-62.763 deg/s at sic=62; prom=160.680; half-prom width=3.006 sic (edge-clipped)|
|zero-crossing in window|sic=59.572, slope=30.406|
|omega_z dynamic range|160.680 deg/s|
|d omega_z / d sic stats|median=26.189, mean=26.780, p90=29.930, max=30.406|
|curvature (d^2 omega_z / d sic^2)|median=3.367, mean=3.595, p90=5.460, max=5.606|
|plateau test (>=95% / >=90% of dominant)|positive: 56-56 (len=1) / 56-56 (len=1); negative: 62-62 (len=1) / 62-62 (len=1)|
|multi-peak competition|positive: no interior peak; window is edge-dominated / monotonic; negative: no interior peak; window is edge-dominated / monotonic|

Qualitative read:
- The window is almost a monotonic high-slope descent from strong positive omega_z to strong negative omega_z.
- The interpolated zero-crossing sits cleanly inside the window, and the crossing slope is large relative to the rest of the clip.
- There is no real plateau and no meaningful peak competition inside the window; the structure looks phase-sharp and deterministic.

## Final read

|window|GT structure|lean|why|
|---|---|---|---|
|`calf_l @ sic=78-85`|broad / phase-unfriendly|more like ambiguity / observability|adjacent zero plateau, soft onset, broad shoulder, no clean in-window zero-crossing|
|`calf_r @ sic=56-62`|sharp / deterministic|more like capacity / temporal-resolution|clean in-window sign flip, large crossing slope, monotonic ramp, no plateau / no peak competition|

- The window that most looks like `model should learn this but currently has not` is `calf_r @ sic=56-62`.
- The window that more looks like `GT is itself less phase-friendly in this local region` is `calf_l @ sic=78-85`, but the claim should stay modest: this is only a single-clip structural read, not a cross-sample variance floor argument.
- So the practical read is: if capacity / temporal-resolution follow-up is expensive, `calf_r` is the cleaner target for that direction; `calf_l` still carries a stronger ambiguity / observability caveat.

