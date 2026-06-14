> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# Command Response Demotion Replay

Date: 2026-06-03

## 1. Scope / Non-goals

本轮只做 debug-only / read-only acceptance replay。目标是审 `command_response`
从 locomotion-style per-frame hard gate 降级为 middle/inbetween bridge 的
diagnostic 是否会放过 shortcut。

Non-goals:

- 不训练 generator / decoder。
- 不 forward production runtime / trainer。
- 不改 production trainer / runtime / gate。
- 不改 checkpoint。
- 不把 command 降级写成 anchored/lifted 必然正确。

输入序列在 probe 内统一按 CPU `float32` 处理：`rot6d [H,276]`、
`root_pos [H,3]`、`root_vel [H,2]`、`bone_angvel [H,138]`、
`cond_dir [H,2]`、`contact [H,2]`、`yaw_rate [H]`，本轮 `H=16`。

## 2. Why Demote Per-frame Command Response

旧 contract 把 command response 写成 “realized yaw/root direction follows the
commanded-yaw path”，并要求 heading/yaw alignment、heading error、root-direction
consistency 等指标；这在实现上落成了 per-frame `root_vel` 对 `cond_dir` 的
`heading_error_p95` hard gate。

middle/inbetween 的主职责不是逐帧像 locomotion controller 一样追 command。
start/end 或 planner 应负责选择目标 regime / endpoint / support schedule；
middle 负责生成物理上可接的过渡。它不能明显反 command，但允许踏步、重心转移
中的瞬时偏离。

已有 2026-06-03 signal audit 已把 `cond_dir`/yaw 归类为 commanded / available
cue，而不是 prediction success target。因此本 replay 不再把 per-frame
`command_response` 作为 hard blocker，而改用 net/integral anti-shortcut check。

## 3. Reformulated Command Family

新 hard family 是 `command_compatibility`，定义为：

- `net_integral_ok`: `abs(sum(yaw_rate) / FPS - wrapped_net_angle(cond_dir))`
  不超过同 clip 连续 GT 窗口校准 band。
- `root_not_counter_command`: 整段 root displacement 在平均 `cond_dir` 上的投影
  不为明显反向。

这不是 endpoint heading match。它检查整段 horizon 到底有没有按 commanded 方向
产生净 turn，同时不要求每一帧 velocity 都贴合 command。

为避免 exact GT 被 metric calibration 误拦，`command_quantile=100.0`。这使
`flat_state281` GT reconstruct 的 `command_compatibility_pass_rate=1.0000`，
符合 acceptance-grade reconstructability 的校准要求。

## 4. Anti-gaming Negative Controls

负控必须仍 fail；否则 demotion 就是在删掉唯一抓 shortcut 的 gate。

结果路径：
`debug_output/_tmp_action_handoff_command_demotion_replay_20260603/command_demotion_replay_summary.md`

| case | n | demoted pass | command compat | failed families |
|---|---:|---:|---:|---|
| `negative_control:linear_pose_contact_proxy` | 3 | 0.0000 | 0.0000 | `command_compatibility:3` |
| `negative_control:one_frame_angvel_root_switch` | 3 | 0.0000 | 0.0000 | `rate_budget:3, command_compatibility:3, endpoint_bridgeability:3` |
| `negative_control:matched_hard_seam` | 3 | 0.0000 | 0.0000 | `rate_budget:3, command_compatibility:3, endpoint_bridgeability:3` |
| `negative_control:direct_full` | 3 | 0.0000 | 0.0000 | `rate_budget:3, command_compatibility:3, pose_continuity:3, endpoint_bridgeability:3` |
| `negative_control:lambda_model` | 3 | 0.0000 | 0.0000 | `rate_budget:3, command_compatibility:3, pose_continuity:3, endpoint_bridgeability:3` |

Interpretation: demotion 没有让 linear / one-frame / hard-seam / direct-family
负控通过。尤其 linear proxy 仍只靠 `command_compatibility` 被拦住，保留了
“平滑但没真的进入 turn regime”的反 gaming 防线。

## 5. Reconstructability Replay

Reconstructability 覆盖 `188` 个 `H=16` windows，包含 support switch windows。
所有 rows 走 reconstructed-domain acceptance path；per-frame `command_response`
只保留 diagnostic。

| representation | n | legacy pass | demoted pass | command compat | support core | support full legacy | support honest | rate | heading err mean | foot ratio | root p95 err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat_state281` | 188 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.7356 | 0.000000 |
| `root_position_lifted` | 188 | 0.0000 | 0.9681 | 1.0000 | 1.0000 | 0.0053 | 1.0000 | 0.9681 | 0.0026 | 0.7356 | 0.000000 |
| `support_anchor_keep_inter_anchor` | 188 | 0.0000 | 0.9468 | 1.0000 | 1.0000 | 0.0053 | 0.9840 | 0.9574 | 0.0026 | 0.7356 | 0.000000 |
| `support_anchor_drop_inter_anchor` | 188 | 0.0000 | 0.7394 | 0.9202 | 0.9628 | 0.0053 | 0.9468 | 0.7500 | 0.1301 | 0.9581 | 0.145956 |

Key points:

- `flat_state281` exact GT reconstruct is acceptance-grade after command calibration:
  `demoted_pass_rate=1.0000`。
- `support_anchor_keep_inter_anchor` no longer fails because of old per-frame command:
  `command_compatibility_pass_rate=1.0000`。
- keep-inter-anchor still misses the `0.95` reconstruct threshold by `0.0032`:
  `demoted_pass_rate=0.9468`，remaining failures are `rate_budget:8` and
  `support_honesty:3` windows.
- `support_side_core=1.0000` but full legacy support-side is `0.0053` because the
  legacy support-side feature set includes command-ish / velocity-side keys:
  `yaw_sum_rad`, `yaw_abs_sum_rad`, `heading_error_p95_rad`, `root_speed_mean`,
  `root_lateral_mean`, `support_yaw_product`, `support_lateral_product`。
- Dropping inter-anchor / footstep placement remains a negative control:
  root p95 error `0.145956 m` and `demoted_pass_rate=0.7394`。

## 6. Dual-sided Perturbation Sensitivity

本 replay 同时报 position-derived 与 velocity-derived 量，避免只展示 anchored
在 position side 的收益而隐藏 velocity side 代价。

At `noise_mse=1e-3` (`n=564`, 188 windows x 3 trials):

| representation | demoted pass | command compat | support core | support honest | rate | heading err mean | foot ratio | root p95 err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat_velocity_state281` | 0.0124 | 1.0000 | 0.7057 | 0.8759 | 0.0142 | 0.0707 | 0.7375 | 0.002885 |
| `root_position_lifted` | 0.0000 | 1.0000 | 0.0053 | 0.0213 | 0.0000 | 2.7097 | 2.2342 | 0.092092 |
| `support_anchor_keep_inter_anchor` | 0.0000 | 1.0000 | 0.0035 | 0.0230 | 0.0000 | 2.6843 | 2.2470 | 0.092942 |
| `support_anchor_drop_inter_anchor` | 0.0000 | 0.9113 | 0.0035 | 0.0177 | 0.0000 | 2.6847 | 2.4355 | 0.226018 |

Interpretation / noise-model caveat:

- 本表使用 per-frame independent Gaussian noise。它是高频噪声模型，不是 decoder
  真实误差频谱的证明。
- Flat 路径从 velocity 积分到 root position；积分会低通高频噪声。因此 flat 在本表中
  root/foot position-side 指标较稳定：`root p95 err=0.002885 m`，`foot ratio=0.7375`。
- Root-position / anchored 路径从 position finite-diff 回 root velocity；微分会放大高频噪声。
  因此 lifted/anchored 的 heading error 跳到约 `2.68-2.71 rad`，rate budget 是 `0.0000`。
- 这些数字不能作为 anchored conditioning 的反证，也不能作为 anchored 稳定性的证明。它们只说明：
  independent high-frequency position noise 对 lifted velocity-side metrics 是最坏情形之一。
- Drop-inter-anchor is strictly worse on root path (`0.226018 m` at `1e-3`) and
  command compatibility (`0.9113`) than keep-inter-anchor, so inter-anchor /
  footstep placement stays in the representation contract.

## 7. Decision Boundary

Demotion guard status:

- Negative controls still fail: pass。
- Linear proxy remains blocked by net-integral command: pass。
- GT exact flat reconstruct remains acceptance-grade: pass。
- Anchored keep no longer blocked by old per-frame command: pass。
- Anchored keep reconstructability is close but still below threshold:
  `0.9468 < 0.95`，blocked by rate/support-honesty exactness。
- Current perturbation is not a fair conditioning verdict: the independent high-frequency
  noise model favors flat's integration path and penalizes lifted finite-diff velocity.
  Conditioning advantage remains untested under equal-state-MSE correlated noise。

Decision:

`command_response` can be demoted from hard per-frame locomotion-style acceptance to
diagnostic for middle/inbetween, provided the new hard command family is net/integral
`command_compatibility` and negative-control replay remains part of the gate.

Do not enter anchored/lifted decoder toy smoke yet. The next narrow step is a
GT-only support-side / rate exactness audit for reconstructed lifted paths. After
anchored keep reaches acceptance-grade reconstructability, rerun perturbation with
native-space correlated/bias noise, equal reconstructed-`state281` MSE calibration,
and dual position-side / velocity-side reporting.

## 8. Artifacts

- `tools/run_action_handoff_command_demotion_replay.py`
- `debug_output/_tmp_action_handoff_command_demotion_replay_20260603/command_demotion_replay_summary.md`
- `debug_output/_tmp_action_handoff_command_demotion_replay_20260603/command_demotion_replay_summary.json`
- `debug_output/_tmp_action_handoff_command_demotion_replay_20260603/command_demotion_replay_rows.csv`
