# Standard Rotvec Semantics

This debug repo now uses one SO(3) contract everywhere:

- `train.geometry.so3_log_map(R)` returns the standard axis-angle / rotvec `axis * angle`.
- `train.geometry._matrix_log_map(R)` uses the same semantics.
- `train.geometry.angvel_vec_from_delta_R` and `train.geometry.angvel_vec_from_R_seq` therefore produce standard rotvec angular velocity in `rad/s`.
- Lambda blend, direct-leg omega diagnostics, exported `rotvec_deg_xyz`, `omega_deg_xyz`, and related JSON fields no longer apply any external `*2` compensation.

## Asset Migration

This repo keeps normalized angvel features numerically stable by migrating the scale-carrying assets:

- `raw_data/processed_data/norm_template.json`
- `models/pretrain_template.json`
- `temp_pretrain_template.json`
- `models/motion_encoder_equiv_stageA.pt`
- `models/motion_encoder_equiv.pt.best.pt`

Policy:

- Raw angvel semantics change from legacy half-angle to standard rotvec.
- `tanh_scales_angvel` is doubled to match the new raw angvel magnitude.
- Tanh-domain z-score stats (`MuAngVel`, `StdAngVel`) stay unchanged because `tanh((2w)/(2s)) = tanh(w/s)`.
- Frozen motion-encoder bundles are stamped with the same standard rotvec metadata and runtime now fail-fast on untagged / legacy assets.

## Verification

Recommended checks:

```bash
python tools/check_standard_rotvec_semantics.py
python tools/check_lambda_fusion_blend_geometry.py
```

For freerun validation, the canonical pretrain template path is now `models/pretrain_template.json`.
