# 2026-04-09 replace handoff distribution pathology probe

> Archived on 2026-04-12.  
> Current role: historical negative evidence against treating activation-distribution pathology as the main explanation for the old-boundary replace gap.  
> Reader guidance: labels like `a1_then_mainline` and “本记录的历史主结论” below are preserved as local historical shorthand, not present-tense policy.

> Scope: B1 only / 70a donor-state -> 70b replace entry activation distribution pathology profile / no-train

## 1. Scope / inherited conclusions

直接继承，不重证：
- `a1_then_mainline`:
  - root cause 不在 planner semantics 主线
  - root cause 不在 replace entry 外部 rollout state
  - root cause 不在 contacts_in_t
  - earliest semantic split 在 direct_pose_head boundary
  - direct_pose_head 是 earliest boundary / necessary anchor，但不是 standalone sufficient
  - normality probe 在 A1 口径下 non-discriminative，不要当主判据
  - A1-S5: donor family 的 direct_pose_head output 与 target/baseline manifold 有大幅 drift
  - A1-S5: donor-vs-donor divergence 很小，但 donor-vs-target divergence 很大
  - A1-S5: affine guard 对 replace aggregate 无明显帮助
- `notail_falsifier`:
  - tail-k 不是 representation drift 的必要原因
  - notail 的 head-output cosdist to target 仍约 0.785
  - notail native direct/freerun 指标优于 E1-top3
  - 当前更像 donor-family / 70a->70b handoff 机制问题，而不是 tail-k 特有问题
- `replace_closed_loop_falsifier`:
  - step0/1 local optizability 版本的解释已基本不支持
  - 当前更像 initial loss profile -> group-norm EMA seed -> rollout feedback
  - 本轮待答：是什么 distribution shape 把 initial loss profile 弄歪

## 2. Why this probe after A1 + notail falsifier

- A1 已经把 earliest usable boundary 锁到 `direct_pose_head`，但还没回答 replace entry 的 activation shape 是否已病态。
- notail falsifier 说明问题不像 tail-k 特有；更像 donor-family handoff 机制。
- 因此本轮只看 `70a donor -> 70b replace objective` 入口时 direct branch activation distribution / grad-share / early EMA support。

## 3. Compared arms

| arm | ckpt |
|---|---|
| `baseline-raw70a` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_posttrain_pipeline_from_bestfree_20260317/70a/ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth` |
| `E1-top3-raw70a` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk3_rankmix_tw020_stage70a_from_tailfix_e1_20260408/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk3_rankmix_tw020_stage6tailfix_e1_20260408.pth` |
| `notail-raw70a` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_notail_stage70a_from_tailfix_20260409/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_notail_stage6tailfix_20260409.pth` |

## 4. Hook definitions

| hook_key | selected module | kind | reason | shapes |
|---|---|---|---|---|
| `direct_pose_head_out` | `direct_pose_head` | `forward_hook_output` | earliest shared trunk boundary; A1 inherited earliest necessary anchor | `[[1, 512]]` |
| `direct_pose_leg_head_out` | `direct_pose_leg_head` | `forward_hook_output` | leg-only branch hidden state before leg readout | `[[1, 24]]` |
| `direct_pose_arm_proj_out` | `direct_pose_arm_proj` | `forward_hook_output` | arm-side nonleg adapter/proj before arm readout | `[[1, 256]]` |
| `direct_pose_else_proj_out` | `direct_pose_else_proj` | `forward_hook_output` | else-side nonleg adapter/proj before else readout | `[[1, 256]]` |
| `direct_pose_out_leg_in` | `direct_pose_out_leg` | `forward_pre_hook_input` | leg readout contract input; helps separate trunk-vs-leg-readout issues | `[[1, 512]]` |
| `direct_pose_out_arm_in` | `direct_pose_out_arm` | `forward_pre_hook_input` | arm readout contract input; primary nonleg branch readout tap | `[[1, 256]]` |
| `direct_pose_out_else_in` | `direct_pose_out_else` | `forward_pre_hook_input` | else readout contract input; complements arm-side nonleg tap | `[[1, 256]]` |

## 5. Statistics definition

- Channel axis = tensor last dim; other dims全部 flatten 后聚合。
- 主统计：`mean/std/min/max/p01/p99/median/MAD/Fisher excess kurtosis`。
- Grad 统计：每个 channel 记录 `mean abs grad wrt dir_leg_base` 与 `wrt dir_nonleg_base`，再算 `leg_grad_share / nonleg_grad_share`。

## 6. Dead / heavy-tail criteria

- near-dead: `std <= max(abs_dead_std, rel_dead_factor * layer_median_std) AND (p99-p01) <= max(abs_dead_span, rel_dead_factor * layer_median_span)`
- low-diversity: `MAD <= max(abs_dead_mad, rel_dead_factor * layer_median_mad)`
- heavy-tail: `Fisher excess kurtosis >= heavy_tail_excess_kurtosis`
- scale-outlier: `max(|mean|/layer_median_|mean|, std/layer_median_std, p99_abs/layer_median_p99_abs) >= scale_outlier_ratio`

## 7. Per-hook summary table

| arm | hook | near_dead | low_div | heavy_tail | scale_outlier | anomaly_leg_mass | anomaly_nonleg_mass |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline-raw70a` | `direct_pose_head_out` | 253 | 435 | 37 | 251 | 0.7664 | 0.2336 |
| `baseline-raw70a` | `direct_pose_leg_head_out` | 3 | 4 | 0 | 1 | 1.0000 | 0.0000 |
| `baseline-raw70a` | `direct_pose_arm_proj_out` | 152 | 183 | 2 | 104 | 0.0000 | 1.0000 |
| `baseline-raw70a` | `direct_pose_else_proj_out` | 103 | 164 | 3 | 0 | 0.0000 | 1.0000 |
| `baseline-raw70a` | `direct_pose_out_leg_in` | 253 | 435 | 37 | 251 | 0.7664 | 0.2336 |
| `baseline-raw70a` | `direct_pose_out_arm_in` | 152 | 183 | 2 | 104 | 0.0000 | 1.0000 |
| `baseline-raw70a` | `direct_pose_out_else_in` | 103 | 164 | 3 | 0 | 0.0000 | 1.0000 |
| `E1-top3-raw70a` | `direct_pose_head_out` | 124 | 300 | 50 | 14 | 0.8251 | 0.1749 |
| `E1-top3-raw70a` | `direct_pose_leg_head_out` | 4 | 4 | 0 | 2 | 1.0000 | 0.0000 |
| `E1-top3-raw70a` | `direct_pose_arm_proj_out` | 105 | 129 | 3 | 9 | 0.0000 | 1.0000 |
| `E1-top3-raw70a` | `direct_pose_else_proj_out` | 56 | 111 | 5 | 0 | 0.0000 | 1.0000 |
| `E1-top3-raw70a` | `direct_pose_out_leg_in` | 124 | 300 | 50 | 14 | 0.8251 | 0.1749 |
| `E1-top3-raw70a` | `direct_pose_out_arm_in` | 105 | 129 | 3 | 9 | 0.0000 | 1.0000 |
| `E1-top3-raw70a` | `direct_pose_out_else_in` | 56 | 111 | 5 | 0 | 0.0000 | 1.0000 |
| `notail-raw70a` | `direct_pose_head_out` | 127 | 293 | 39 | 14 | 0.8477 | 0.1523 |
| `notail-raw70a` | `direct_pose_leg_head_out` | 4 | 5 | 0 | 2 | 1.0000 | 0.0000 |
| `notail-raw70a` | `direct_pose_arm_proj_out` | 114 | 134 | 3 | 15 | 0.0000 | 1.0000 |
| `notail-raw70a` | `direct_pose_else_proj_out` | 60 | 111 | 6 | 0 | 0.0000 | 1.0000 |
| `notail-raw70a` | `direct_pose_out_leg_in` | 127 | 293 | 39 | 14 | 0.8477 | 0.1523 |
| `notail-raw70a` | `direct_pose_out_arm_in` | 114 | 134 | 3 | 15 | 0.0000 | 1.0000 |
| `notail-raw70a` | `direct_pose_out_else_in` | 60 | 111 | 6 | 0 | 0.0000 | 1.0000 |

## 8. Anomaly vs grad alignment summary

- `E1-top3-raw70a`: dominant family=`leg_branch`, dominant anomaly type=`heavy_tail`, grad alignment=`leg_aligned`, support=`supportive_but_not_causal`。
- `notail-raw70a`: dominant family=`nonleg_branch`, dominant anomaly type=`heavy_tail`, grad alignment=`nonleg_aligned`, support=`supportive_but_not_causal`。
- 但这两个 arms **没有给出 shared dominant family**：`E1-top3` 更偏 leg-aligned，`notail` 更偏 nonleg-aligned。
- 同时 baseline 在 `near_dead / low_diversity / scale_outlier` 计数上普遍更高；donor 侧相对 baseline 真正稳定增加的，主要只是 `heavy_tail` 计数。

## 9. Optional EMA-seed support

- 记录了前 N batch 的 `dir_leg_base / dir_nonleg_base / dir_group_norm_leg_raw / dir_group_norm_nonleg_raw / dir_group_norm_*_ema` 序列。
- 这些序列在 `summary.json` 的 `arms.*.per_batch_sequence` 下可直接做 step0/step1 / seed trajectory 对比。
- 本配置下 `loader_len=1`，因此本轮 `N=32` 实际口径是：**同一个 training loader batch 重复 32 次 + 每次新的 rollout random offset**，这与真实 training entry 更一致。

## 10. Interpretation

- `step0/step1` raw loss 与 EMA seed 仍然明显分开：`E1-top3` / `notail` 的 `dir_leg_base`、`dir_nonleg_base`、对应 EMA seed 全都低于 baseline。
- 但 activation pathology 的**共享形态**并不成立：
  - baseline 反而有更多 `near_dead / low_diversity / scale_outlier`；
  - donor 侧额外增加的主要是 `heavy_tail`；
  - `E1-top3` 与 `notail` 的 dominant family / grad alignment 不一致。
- 因此本轮更像 **Case B/C 之间**：
  - 分布差异存在，但不足以给出一个 shared donor-family activation pathology 主线；
  - 目前更不像“同一类 direct-branch distribution 病态把 70b replace 弄歪”。
- overall distribution pathology support: `False`
- historical recommendation: `do_not_promote_distribution_pathology_as_mainline`
- Q1: `not_both_clearly_supported`
- Q2: `no_clear_dominant_family`
- Q3: `{'e1_top3': 'leg_aligned', 'notail': 'nonleg_aligned'}`
- Q4: `not_supported_as_mainline`
- Q5: `do_not_promote_distribution_pathology_as_mainline`

## 11. Historical next-step recommendation

- 不建议直接进入 `robust normalization / clipping / branchwise rescale / whitening` 作为主线修正。
- 若要保留本线，只建议把它降级成 **follow-up**：
  - `notail` 偏 nonleg-aligned heavy-tail，可做极轻量 `robust clipping` smoke；
  - `E1-top3` 偏 leg-aligned heavy-tail，可做极小范围 branch-specific guard；
  - 但这些都不应升级成主判据。
- 本记录的历史主结论：`do_not_promote_distribution_pathology_as_mainline`
