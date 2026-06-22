> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Lifted Contract Exactness Repair Audit

Date: 2026-06-03

## 1. Scope / Non-goals

本轮只做 GT-only / read-only exactness repair audit，目标是把
`support_anchor_keep_inter_anchor` 的 reconstructed-domain exactness 从 `0.9468`
审到 acceptance-grade，并判定 hard-band residual 是否是真表征损失。

Probe 输入/输出均为 CPU `float32`：`state281 [16,281]`、`root_pos [16,3]`、
`root_vel [16,2]`、`contact [16,2]`、`bone_angvel [16,138]`。覆盖 matched windows
`n=188`，per-clip windows 为 `Walk_L_To_L:39`、`Walk_R_To_L:71`、
`Walk_R_To_R:78`。

Non-goals:

- 不训练 decoder / generator。
- 不 forward production runtime / trainer / gate。
- 不改 production trainer / runtime / gate。
- 不改 checkpoint。
- 不做 residual head。
- 不继续 endpoint / yaw / discriminator 仪器。
- 不推进 anchored/lifted decoder toy smoke。
- yaw / `cond_dir` 只作为 commanded cue。
- 不把 flat decoder failure 写成 diffusion required。
- 不把 support-foot-anchor 预设为正确答案。

Artifacts:

- `tools/run_action_handoff_lifted_contract_exactness_repair.py`
- `debug_output/_tmp_action_handoff_lifted_contract_exactness_repair_20260603/lifted_contract_exactness_repair_summary.md`
- `debug_output/_tmp_action_handoff_lifted_contract_exactness_repair_20260603/lifted_contract_exactness_repair_summary.json`
- `debug_output/_tmp_action_handoff_lifted_contract_exactness_repair_20260603/lifted_contract_exactness_repair_rows.csv`

## 2. Preflight Caveat Consistency

Preflight 对 2026-06-03 signal audit 做了最小一致性修正：`1e-3` perturbation 数
只标为 per-frame independent Gaussian / high-frequency sensitivity diagnostic，不作为
anchored-vs-flat conditioning verdict。

Caveat 固定为：

- current noise = per-frame independent Gaussian / high-frequency。
- flat velocity integration low-passes 该噪声。
- lifted position finite-diff high-passes 该噪声。
- 这些数不能作为 conditioning 判决，也不能写成 anchored 不稳 / anchored 更敏感的结论。
- fair perturbation gate 需要 native-space correlated/bias noise、equal reconstructed-`state281`
  MSE，以及 position-side / velocity-side 双侧指标。

本轮只做 caveat consistency，不扩展为新 signal 文档任务。

## 3. Current Blocker

已认可 baseline：

- `support_anchor_keep_inter_anchor`: `n=188`、`H=16`、demoted pass `0.9468`。
- failed family counts: `rate_budget:8`、`support_honesty:3`，其中
  `Walk_L_To_L[14:29]` overlap。
- `command_compatibility=1.0000`，`support_side_core=1.0000`。
- root path p95 error mean `1.44e-8 m`，support-foot world displacement p95 mean
  `1.96e-8 m`。
- heading error p95 mean `0.0026155 rad`，foot-slip ratio mean `0.7356205`。

因此 blocker 不在 command demotion，也不在 hard support-side core。hard-band 视图下的
剩余问题分两层：

- baseline 当前 forward FD 的 `root_pos -> state281 root_vel` exactness / calibration。
- committed `endpoint_consistent_fd` 修掉 rootvel 后，只剩 support/FK foot-slip band-edge
  float32 round-trip precision residual。

## 4. Failure Row Decomposition

Baseline `support_anchor_keep_inter_anchor` 的 10 个 failed windows 如下。每行列出实际
超过 band 的子项；未列入 `exceeded subitems` 的 rate/support 子项没有超过对应 band。

| clip | start | end | failed family | exceeded subitems | rootvel p95/band | foot slip p95/band | root path p95 m | support foot disp p95 m | heading p95 rad | max abs state delta |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| Walk_L_To_L | 5 | 20 | rate_budget | rootvel_step_l2_p95 | 0.042420636 / 0.042419876 | 1.797852433 / 2.766105282 | 2.28e-8 | 2.68e-8 | 0.000543771 | 0.023647070 |
| Walk_L_To_L | 6 | 21 | rate_budget | rootvel_step_l2_p95 | 0.042419995 / 0.042419876 | 1.822326487 / 2.766105282 | 1.53e-8 | 1.49e-8 | 0.000399353 | 0.012094498 |
| Walk_L_To_L | 7 | 22 | rate_budget | rootvel_step_l2_p95 | 0.042419995 / 0.042419876 | 1.817997348 / 2.766105282 | 1.09e-8 | 1.49e-8 | 0.000327940 | 0.010948777 |
| Walk_L_To_L | 10 | 25 | rate_budget | rootvel_step_l2_p95 | 0.042420636 / 0.042419876 | 2.114327520 / 2.766105282 | 2.24e-8 | 1.49e-8 | 0.000083415 | 0.007525682 |
| Walk_L_To_L | 14 | 29 | rate_budget,support_honesty | rootvel_step_l2_p95,foot_slip_p95_mps | 0.042419995 / 0.042419876 | 2.766106200 / 2.766105282 | 1.68e-8 | 2.98e-8 | 0.000286328 | 0.002964020 |
| Walk_L_To_L | 15 | 30 | rate_budget | rootvel_step_l2_p95 | 0.042420562 / 0.042419876 | 2.765579104 / 2.766105282 | 1.49e-8 | 2.98e-8 | 0.000380300 | 0.001821995 |
| Walk_L_To_L | 16 | 31 | rate_budget | rootvel_step_l2_p95 | 0.042420562 / 0.042419876 | 2.765051663 / 2.766105282 | 2.98e-8 | 2.98e-8 | 0.000475135 | 0.001256245 |
| Walk_L_To_L | 17 | 32 | rate_budget | rootvel_step_l2_p95 | 0.042419995 / 0.042419876 | 2.764526033 / 2.766105282 | 2.98e-8 | 4.02e-8 | 0.000562200 | 0.001489722 |
| Walk_R_To_L | 16 | 31 | support_honesty | foot_slip_p95_mps | 0.027248474 / 0.048021401 | 3.760815430 / 3.760814953 | 2.24e-8 | 2.98e-8 | 0.008493788 | 0.023274329 |
| Walk_R_To_L | 21 | 36 | support_honesty | foot_slip_p95_mps | 0.027248474 / 0.048021401 | 3.760815263 / 3.760814953 | 7.45e-9 | 1.49e-8 | 0.008177662 | 0.023952005 |

Rate failure margin is tiny: `rootvel_step_l2_p95` over-band count `8`, mean margin
`4.21e-7`, max margin `7.60e-7`. Baseline support failure is also tiny:
`foot_slip_p95_mps` over-band count `3`, mean margin `5.68e-7 m/s`, max margin
`9.18e-7 m/s`.

## 5. Exactness Variants

All rows use the same `support_anchor_keep_inter_anchor` GT reconstructed seq family.

| variant | n | hard pass | float32 pass | rate | support honest / float32 | support core | command compat | pose | endpoint | hard failures | float32 failures | root p95 mean/p95/max m | support foot disp mean/p95/max m | foot ratio mean/p95/max | heading mean/p95/max rad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| baseline_current | 188 | 0.9468 | 0.9574 | 0.9574 | 0.9840 / 0.9947 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `rate_budget:8, support_honesty:3` | `rate_budget:8, support_honesty:1` | 1.44e-8 / 3.07e-8 / 4.47e-8 | 1.96e-8 / 4.77e-8 / 5.96e-8 | 0.735621 / 1.000000 / 1.000000 | 0.002615519 / 0.008146047 / 0.009420923 |
| copied_gt_root_vel | 188 | 0.9734 | 1.0000 | 1.0000 | 0.9734 / 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `support_honesty:5` | `{}` | 1.23e-8 / 3.00e-8 / 3.34e-8 | 2.42e-8 / 1.19e-7 / 1.23e-7 | 0.735621 / 1.000000 / 1.000000 | 3.30e-8 / 5.10e-8 / 5.10e-8 |
| endpoint_consistent_fd | 188 | 0.9734 | 1.0000 | 1.0000 | 0.9734 / 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `support_honesty:5` | `{}` | 1.23e-8 / 3.00e-8 / 3.34e-8 | 2.42e-8 / 1.19e-7 / 1.23e-7 | 0.735621 / 1.000000 / 1.000000 | 0.009080730 / 0.021297708 / 0.022446727 |
| contact_passthrough_check | 188 | 0.9468 | 0.9574 | 0.9574 | 0.9840 / 0.9947 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `rate_budget:8, support_honesty:3` | `rate_budget:8, support_honesty:1` | 1.44e-8 / 3.07e-8 / 4.47e-8 | 1.96e-8 / 4.77e-8 / 5.96e-8 | 0.735621 / 1.000000 / 1.000000 | 0.002615519 / 0.008146047 / 0.009420923 |
| support_side_core_only | 188 | 1.0000 | 1.0000 | 0.9574 | 0.9840 / 0.9840 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `{}` | `{}` | 1.44e-8 / 3.07e-8 / 4.47e-8 | 1.96e-8 / 4.77e-8 / 5.96e-8 | 0.735621 / 1.000000 / 1.000000 | 0.002615519 / 0.008146047 / 0.009420923 |

`support_side_core_only=1.0000` is diagnostic-only. It excludes rate/support_honesty
from the pass definition and therefore cannot license decoder entry by itself.

`copied_gt_root_vel` is also diagnostic-only: it copies GT `root_vel [16,2] float32 cpu`
and is an oracle upper bound, not a deployable reconstruction contract. The committed
contract is `endpoint_consistent_fd`, because it computes `root_vel` from the lifted
`root_pos [16,3] float32 cpu` path.

## 6. Root Velocity Reconstruction Finding

`copied_gt_root_vel` raises hard-band demoted pass from `0.9468` to `0.9734`, and
`rate_budget_pass_rate` from `0.9574` to `1.0000`. Because it copies GT
`root_vel [16,2] float32 cpu`, it is only an oracle upper bound. Its role is to
isolate root-path-to-root-velocity exactness / calibration, not to define the contract.

`endpoint_consistent_fd` is the committed debug reconstruction contract. It reaches
hard-band `0.9734`, float32 precision-tolerant `1.0000`, and `rate_budget=1.0000`.
The debug
scheme uses anchored keep `root_pos [16,3] float32 cpu` as the authoritative path and
derives `root_vel [16,2] float32 cpu` by central finite difference with endpoint
one-sided copies. This is a recommended debug reconstruction contract for subsequent
read-only audits, but it does not change production gate/runtime.

The previous baseline rate failures were all `rootvel_step_l2_p95` only. No
`angvel_step_rms_p95`、`angvel_component_p95_p95`、or `yaw_rate_step_abs_p95`
subitem exceeded band in those rate-only rows.

## 7. Support Honesty / FK Finding

`contact_passthrough_check` keeps oracle contact exact: `max_abs_contact_delta=0.0`
on all audited rows. Its remaining support failures still have `contact_step_l2_p95`
within band and fail only on `foot_slip_p95_mps`.

After root_vel repair, the remaining hard-band `copied_gt_root_vel` /
`endpoint_consistent_fd` failures are `support_honesty:5`, all due to
`foot_slip_p95_mps` over band by `1.43e-7..9.18e-7 m/s`. The configured float32
round-trip tolerance is `max(1e-6 m/s, 2e-6 * foot_slip_band)`, i.e. `5.53e-6`
to `7.52e-6 m/s` on these bands. All 5 residuals are inside that tolerance.
Their root path p95 error remains `<=3.02e-8 m`, and support-foot world displacement
p95 remains `<=1.23e-7 m`.

This is `(band-edge tail) x (float32 round-trip precision)`, not evidence of a
real representation contract gap. The lifted GT reconstruction is lossless at the
same practical precision level as flat GT reconstructability. It also does not prove
support-foot-anchor is the right final representation; it only shows that the
keep-inter-anchor GT path is lossless under the committed reconstruction contract.

## 8. Decision Boundary

Decision criteria applied:

- `copied_gt_root_vel >=0.95`: yes, hard-band `0.9734`, float32 pass `1.0000`;
  oracle upper bound only.
- `endpoint_consistent_fd >=0.95`: yes, hard-band `0.9734`, float32 pass `1.0000`;
  committed debug reconstruction contract, no production gate change.
- `copied_gt_root_vel <0.95`: no; no current evidence of support/FK/contact gap large
  enough to block GT-only reconstructability.
- only `support_side_core_only` passes: no; copied/root FD variants also pass. In any
  future case where only support-side-core passes while rate/support_honesty fail, decoder
  remains blocked.
- Prior command demotion negative controls still fail from
  `debug_output/_tmp_action_handoff_command_demotion_replay_20260603/command_demotion_replay_summary.json`.

Result: `support_anchor_keep_inter_anchor` is GT-lossless under the committed
`endpoint_consistent_fd` reconstruction contract. The apparent hard-band `0.9734`
is not a 2.6% representation loss; it is float32 band-edge precision noise. The next
permitted step is fair perturbation, not decoder toy smoke.

## 9. Next Step: Fair Perturbation Gate Spec

Fair perturbation should now run through the consolidated Layer-2 harness, not another
one-off probe:

- `tools/run_action_handoff_layer2_harness.py`
- `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_summary.md`
- `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_summary.json`
- `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_rows.csv`

Harness contract:

- committed lifted arm: `endpoint_consistent_fd`。
- oracle-only excluded arm: `copied_gt_root_vel`。
- mode flags: `data_line`, `fair_perturbation`。
- fair perturbation arms: `flat_velocity_state281`, `endpoint_consistent_fd`。

Fair perturbation gate requirements:

- Inject noise in each representation's native space.
- Use correlated/bias noise over time, not only per-frame independent Gaussian.
- Calibrate amplitudes to equal reconstructed-`state281 [16,281] float32 cpu` MSE.
- Report both position-side metrics (`root_path_error_p95_m`,
  `support_foot_world_displacement_p95_m`) and velocity-side metrics
  (`rootvel_step_l2_p95`, `heading_error_p95_rad`, `rate_budget`).
- Include finite-difference native-noise-to-acceptance sensitivity proxy as gradient
  diagnostic in read-only mode; model gradient diagnostics remain disabled until a
  non-toy decoder is explicitly in scope.
- Keep command demotion negative controls in the gate: linear / one-frame / direct-family
  rows must remain fail.
- Do not start decoder toy smoke until fair perturbation passes and no new GT-only
  exactness blocker appears.

## 10. Full Fair Perturbation Verdict

The consolidated harness full sweep has been run:

- summary: `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_summary.json`
- rows: `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_rows.csv`
- compact verdict: `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_full_verdict.md`
- verdict json: `debug_output/_tmp_action_handoff_layer2_harness_20260603/layer2_harness_full_verdict.json`

Run matrix:

- windows `188`, switch windows `100`, rows `22560`, summary groups `60`。
- arms: `flat_velocity_state281`, `endpoint_consistent_fd_native`,
  `endpoint_consistent_fd_roundtrip`。
- equalization: `state_mse`, `root_path_p95`。
- state MSE targets: `1e-7`, `1e-6`, `1e-5`。
- root p95 targets: `1e-3 m`, `3e-3 m`。
- noise: `correlated` with rho `0.0/0.5/0.9`, plus `bias`。
- trials: `2`。

Decision: `no_robust_anchored_over_flat_win_full_sweep`。

Interpretation:

- Harness wiring remains valid: anchored native foot-slip is evaluated directly from
  `seq["root_pos"]`, not through root_vel re-integration.
- Anchored native shows real position-side wins in several state-MSE correlated groups,
  but the ordering is not stable across equalization mode, noise kind, rho, and target
  magnitude.
- Against flat over comparable groups (`calibration_valid_rate >= 0.9`), native anchored
  wins root-path error in `9/14` groups, but loses valid float32 pass in `8/14`, heading
  error in `11/14`, rate budget in `8/14`, and support-side core in `12/14` groups.
- Round-trip anchored is not a winning consumption path in this sweep: it wins root-path
  error in `6/9` comparable groups, but loses valid float32 pass in `5/9`, heading in
  `7/9`, and rate budget in `5/9` groups.
- `state_mse + anchored bias` is calibration-degenerate because constant root-position
  bias is mostly invisible to state281 velocity channels; those groups are marked invalid
  by `scale_exceeds_cap`.
- Some round-trip root-path targets are invalid because zero-noise round-trip pipeline
  cost already exceeds the target; those groups are marked `zero_or_baseline_exceeds_target`。

Conclusion: the lifted GT representation is lossless, and the harness is now a credible
fair perturbation tool. But the full perturbation sweep does not establish a robust
anchored-over-flat conditioning advantage. The current evidence says the claimed
"升维解耦" advantage is at best context-dependent; it is not stable enough to justify
decoder toy smoke or a production-path change.
