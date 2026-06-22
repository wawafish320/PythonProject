> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# no-FK between GRU bridge probe

Date: 2026-06-06

Status: debug-only train-fit. No production trainer, checkpoint, or `EventMotionModel` path changed.

## 目的

上一轮 `tools/run_action_handoff_between_feature_bridge_preflight.py` 的 train-fit 仍是 flat-concat MLP：`ctx_state [B,C,281] float32 cpu` 被展平成 4496 维，再直接 free-emit raw `[B,H,281] + [B,H,138]`。四个 variant 指标几乎一样，因此结论只能是旧 harness null，不能解释成 soft contact 无效。

本轮新增 `tools/run_action_handoff_between_gru_bridge_probe.py`，把输入改成同量级 latent：

- `ctx_state [B,16,281] float32 cpu` -> `z_ctx [B,128] float32 cpu`
- `goal_intent [B,54] float32 cpu` -> `z_goal [B,128] float32 cpu`
- `soft_contact [B,16,2] float32 cpu`，debug min/max 映射到 `[0,1]`
- decoder split heads 输出 root motion、root/nonroot rot6d residual、bone0/nonroot bone_angvel residual，并把 contact plan 写回 `state281[...,279:281]`

实现位置：

- variant / controls: `tools/run_action_handoff_between_gru_bridge_probe.py:103`
- contact `[0,1]` 映射与 clip load: `tools/run_action_handoff_between_gru_bridge_probe.py:274`
- low-dim goal intent: `tools/run_action_handoff_between_gru_bridge_probe.py:351`
- `ContactPlanGRU`: `tools/run_action_handoff_between_gru_bridge_probe.py:501`
- `GRUBridgeProbe` split heads: `tools/run_action_handoff_between_gru_bridge_probe.py:524`
- no-FK metrics: `tools/run_action_handoff_between_gru_bridge_probe.py:984`
- dumb baselines: `tools/run_action_handoff_between_gru_bridge_probe.py:913`

## 约束

- 不使用 FK 作为 generator feature、loss 或 success metric。
- 不 import/load skeleton，不调用 legacy acceptance/reconstruction helper。
- `soft_contact [B,H,2] float32 device=model.device` 只作为 continuous cycle/control signal。
- `oracle_contact_upper_bound` 只作为 upper bound；真实 candidate 是 `predicted_contact`，contact plan 从 `ctx_contact [B,C,2] float32 cpu` + `z_ctx/z_goal` 自回归生成。

## 实验设置

Command:

```bash
python3 tools/run_action_handoff_between_gru_bridge_probe.py \
  --epochs 300 \
  --torch-num-threads 8 \
  --out-dir debug_output/_tmp_action_handoff_between_gru_bridge_probe_20260606
```

Artifacts:

- `debug_output/_tmp_action_handoff_between_gru_bridge_probe_20260606/summary.md`
- `debug_output/_tmp_action_handoff_between_gru_bridge_probe_20260606/summary.json`
- `debug_output/_tmp_action_handoff_between_gru_bridge_probe_20260606/variants.csv`
- `debug_output/_tmp_action_handoff_between_gru_bridge_probe_20260606/baselines.csv`
- `debug_output/_tmp_action_handoff_between_gru_bridge_probe_20260606/grad_usage.csv`
- `debug_output/_tmp_action_handoff_between_gru_bridge_probe_20260606/per_window.csv`

Dataset / split:

- matched windows: `188` from `Walk_L_To_L`, `Walk_R_To_L`, `Walk_R_To_R`
- unmatched diagnostic target excluded: `Walk_L_To_R`, `35` windows
- contiguous split: train `111`, test `53`
- split detail: `Walk_L_To_L: train=23 gap=8 test=8; Walk_R_To_L: train=42 gap=8 test=21; Walk_R_To_R: train=46 gap=8 test=24`
- finite_fraction_mean: `1.0`
- contact source min/max before `[0,1]` mapping: right `[-0.95149958, 0.63231367]`, left `[-0.66951746, 0.96048182]`

Tensor contract on the train batch:

| tensor | shape | dtype | device |
|---|---:|---|---|
| `ctx_state` | `[111,16,281]` | `float32` | `cpu` |
| `goal_intent` | `[111,54]` | `float32` | `cpu` |
| `soft_contact` | `[111,16,2]` | `float32` | `cpu` |
| `state_output` | `[111,16,281]` | `float32` | `cpu` |
| `bone_angvel_output` | `[111,16,138]` | `float32` | `cpu` |

## Test metrics

Lower is better except `contact_phase_consistency`.

| variant | root_intent_mse | contact_plan_mse | contact_cycle_delta_mse | contact_phase | pose_nonroot_rot6d_mse | seam_c1_directlocal_mse |
|---|---:|---:|---:|---:|---:|---:|
| `ctx_only` | `2.42439481` | `0.46398839` | `0.00101577` | `0.000000` | `1.13056231` | `0.02442694` |
| `no_contact` | `0.81982507` | `0.46398839` | `0.00101577` | `0.000000` | `1.77107962` | `0.03346882` |
| `predicted_contact` | `0.58217311` | `0.03038624` | `0.00380165` | `0.141896` | `1.37554580` | `0.02594009` |
| `oracle_contact_upper_bound` | `0.78976809` | `0.00000000` | `0.00000000` | `1.000000` | `1.71547744` | `0.02536170` |
| `shifted_or_random_contact_control` | `0.66457787` | `0.30786149` | `0.17083637` | `0.072933` | `1.63160317` | `0.03176143` |

Dumb baselines:

| baseline | root_intent_mse | root_pos_mse | contact_plan_mse | pose_nonroot_rot6d_mse | seam_c1_directlocal_mse |
|---|---:|---:|---:|---:|---:|
| `ctx_last_hold` | `0.20617006` | `0.00001626` | `0.02848499` | `0.00328617` | `0.00663959` |
| `root_linear_to_goal_pose_hold` | `0.12569711` | `0.00000359` | `0.02848499` | `0.00328617` | `0.00663959` |

## Grad / activation usage

`grad_usage.csv` confirms the main learned path is not dead:

| variant | z_ctx_grad | z_goal_grad | contact_used_grad | contact_plan_grad | root_head_grad | pose_local_head_grad |
|---|---:|---:|---:|---:|---:|---:|
| `ctx_only` | `0.00014939` | `0.00000000` | `0.00000000` | `0.00000000` | `0.13434314` | `0.00237691` |
| `no_contact` | `0.00010096` | `0.00010853` | `0.00000000` | `0.00000000` | `0.00450773` | `0.00091851` |
| `predicted_contact` | `0.00011063` | `0.00011488` | `0.00011329` | `0.00001932` | `0.21388554` | `0.00555982` |
| `oracle_contact_upper_bound` | `0.00011167` | `0.00011999` | `0.00000000` | `0.00000306` | `0.00554950` | `0.00110148` |
| `shifted_or_random_contact_control` | `0.00012788` | `0.00013194` | `0.00000000` | `0.00000266` | `0.00486632` | `0.00104922` |

`predicted_contact` has non-zero `z_goal_grad_l2_mean`, `contact_used_grad_l2_mean`, contact planner grad, root head grad, and local head grad. The learned contact path is therefore active; the weak result is not simply an unused-conditioning bug.

## 结论

1. **Goal intent is not ignored inside the learned GRU harness.**
   `predicted_contact` test `root_intent_mse=0.58217311`, better than `ctx_only=2.42439481` by `-1.84222169` and better than `no_contact=0.81982507` by `-0.23765195`.

2. **Soft contact/cycle has discriminative signal, but only the negative-control comparison is valid evidence.**
   `no_contact` sets `contact_used = 0`, so its `contact_plan_mse=0.46398839` is not evidence that contact has no information; it only says the contact path is disabled. The valid contact-phase evidence is `predicted_contact=0.03038624` vs random control `0.30786149` (10.13x lower), with `oracle_contact_upper_bound=0.0` as the upper bound. Also note that `ctx_last_hold contact_plan_mse=0.02848499` is slightly better than `predicted_contact=0.03038624`, so the learned contact plan has not beaten the trivial hold contact baseline in absolute precision.

3. **The learned parameterization fails structurally against dumb baselines.**
   `root_linear_to_goal_pose_hold` has `root_intent_mse=0.12569711`, 4.63x better than the best learned root intent (`predicted_contact=0.58217311`). `ctx_last_hold` has `pose_nonroot_rot6d_mse=0.00328617`, while the best learned pose-local result is `ctx_only=1.13056231` and `predicted_contact=1.37554580`, roughly 344x and 419x worse. This is not "decoder capacity needs a bit more work"; the learned AR delta bridge is worse than doing nothing on pose/state.

4. **Root cause: AR delta drift + no global anchor + overfitting.**
   The decoder updates `cur = prev_state + head(h)` for 16 recurrent steps (`tools/run_action_handoff_between_gru_bridge_probe.py:604`), so per-step residual noise accumulates into pose/root drift. The root path has no explicit linear-to-goal anchor even though the goal intent is available, so it must learn a strong endpoint prior that the dumb baseline gets by construction. The model has `786851` parameters for `111` train windows; train/test for `predicted_contact` has `state_mse 0.08945012 -> 1.32616919` and `pose_nonroot_rot6d_mse 0.09290310 -> 1.37554580` (~14.8x gap). Even the train pose-local loss is ~28.3x worse than the test dumb hold pose-local `0.00328617`, so the representation itself is mismatched, not just under-regularized.

## Anchored residual follow-up

Implemented:

- tool: `tools/run_action_handoff_between_anchor_residual_probe.py`
- root/pose anchor construction: `tools/run_action_handoff_between_anchor_residual_probe.py:68`
- parallel residual decoder: `tools/run_action_handoff_between_anchor_residual_probe.py:123`
- anchor + residual writeback: `tools/run_action_handoff_between_anchor_residual_probe.py:201`
- residual L2 regularized training loop: `tools/run_action_handoff_between_anchor_residual_probe.py:253`

Command:

```bash
python3 tools/run_action_handoff_between_anchor_residual_probe.py \
  --epochs 300 \
  --torch-num-threads 8 \
  --out-dir debug_output/_tmp_action_handoff_between_anchor_residual_probe_20260606
```

Artifacts:

- `debug_output/_tmp_action_handoff_between_anchor_residual_probe_20260606/summary.json`
- `debug_output/_tmp_action_handoff_between_anchor_residual_probe_20260606/variants.csv`
- `debug_output/_tmp_action_handoff_between_anchor_residual_probe_20260606/baselines.csv`
- `debug_output/_tmp_action_handoff_between_anchor_residual_probe_20260606/grad_usage.csv`
- `debug_output/_tmp_action_handoff_between_anchor_residual_probe_20260606/root_invariant.csv`
- `debug_output/_tmp_action_handoff_between_anchor_residual_probe_20260606/per_window.csv`

Tensor contract remains:

| tensor | shape | dtype | device |
|---|---:|---|---|
| `ctx_state` | `[111,16,281]` | `float32` | `cpu` |
| `goal_intent` | `[111,54]` | `float32` | `cpu` |
| `soft_contact` | `[111,16,2]` | `float32` | `cpu` |
| `state_output` | `[111,16,281]` | `float32` | `cpu` |
| `bone_angvel_output` | `[111,16,138]` | `float32` | `cpu` |

300-epoch anchored test results:

| variant | state_mse | root_intent_mse | root_pos_mse | root_disp_mse | yaw_traj_mse | contact_plan_mse | pose_nonroot_rot6d_mse |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ctx_only` | `0.00416819` | `0.17927075` | `0.00000579` | `0.00000483` | `0.24555486` | `0.02848499` | `0.00318504` |
| `no_contact` | `0.00381000` | `0.17695042` | `0.00001026` | `0.00001512` | `0.24250963` | `0.02848499` | `0.00281857` |
| `predicted_contact` | `0.00395877` | `0.18147061` | `0.00002406` | `0.00007381` | `0.24809469` | `0.03637207` | `0.00288047` |
| `oracle_contact_upper_bound` | `0.00348027` | `0.14345171` | `0.00001686` | `0.00004886` | `0.19607310` | `0.00000000` | `0.00286505` |
| `shifted_or_random_contact_control` | `0.00574118` | `0.16695417` | `0.00000697` | `0.00000454` | `0.22871479` | `0.30786149` | `0.00281007` |
| `ctx_last_hold` | `0.00428114` | `0.20617006` | `0.00001626` | `0.00006230` | `0.25187910` | `0.02848499` | `0.00328617` |
| `root_linear_to_goal_pose_hold` | `0.00399263` | `0.12569711` | `0.00000359` | `0.00000000` | `0.17235239` | `0.02848499` | `0.00328617` |

Root invariant:

- `root_invariant.csv` rows: `10`
- all rows: `ok=True`
- max absolute delta between loss-style torch root integration and eval-style per-window root aggregation: `7.06887883e-11`
- max row: `predicted_contact/test`, `loss_style_root_pos_mse=2.40637146e-05`, `eval_style_root_pos_mse=2.40637853e-05`

Reading:

- The anchor fixed the catastrophic AR drift: `predicted_contact state_mse` went from `1.32616919` (AR) to `0.00395877` (anchored), and `pose_nonroot_rot6d_mse` went from `1.37554580` to `0.00288047`.
- Pose/local now beats dumb hold: `predicted_contact pose_nonroot_rot6d_mse=0.00288047` vs `ctx_last_hold=0.00328617`.
- Root position eval is not broken. The loss-style torch integration and eval-style numpy aggregation match to `7.07e-11` absolute error. The earlier "root fails" reading came from the aggregate `root_intent_mse`, not from `root_pos_mse`.
- Root position/displacement is solved for this debug scope and should be treated as a diagnostic, not a modeling target to keep tuning. For `predicted_contact`, `root_pos_mse=2.40637853e-05` and `root_disp_mse=7.38074830e-05`.
- The open root-control problem is yaw-rate. The current yaw residual is net-harmful: the dumb `root_linear_to_goal_pose_hold` anchor has `yaw_traj_mse=0.17235239`, while learned variants are worse (`oracle_contact_upper_bound=0.19607310`, `shifted_or_random_contact_control=0.22871479`, `no_contact=0.24250963`, `ctx_only=0.24555486`, `predicted_contact=0.24809469`). A zero yaw residual would be a stronger baseline than the learned yaw residual for every learned variant here.
- Oracle contact gives a plausible yaw-control signal, but predicted contact does not realize it. `oracle_contact_upper_bound yaw_traj_mse=0.19607310` improves over `no_contact=0.24250963` by `0.04643653` (~19.1% relative), while `predicted_contact=0.24809469` is slightly worse than `no_contact` and has worse contact precision than hold (`contact_plan_mse=0.03637207` vs `0.02848499`). Treat this as a hypothesis until multi-seed confidence intervals confirm the oracle-vs-no-contact gap.

## 下一步

- Keep the anchored parallel residual parameterization; it solves the AR drift failure.
- Keep the root invariant in every debug run; loss-style root integration and eval-style root aggregation must stay aligned before interpreting any root metric.
- Stop using aggregate `root_intent_mse` alone for root conclusions. Report `root_disp_mse`, `endpoint_ego_vel_mse`, `endpoint_yaw_rate_mse`, and `yaw_traj_mse` separately, and treat root position/displacement as solved unless the invariant fails.
- Add a yaw-anchor floor before changing capacity: zero yaw residual must be a baseline, and any learned yaw residual must report delta vs that anchor. Candidate mechanisms are a per-channel residual gate initialized near zero, stronger yaw residual L2, or a better gait-aware yaw-rate anchor.
- Validate the oracle-contact yaw gain before building on it: run at least a small seed sweep / CI for `oracle_contact_upper_bound yaw_traj_mse < no_contact yaw_traj_mse`. If stable, improve the contact planner before increasing yaw decoder capacity, because predicted contact currently fails to capture the oracle gain.
