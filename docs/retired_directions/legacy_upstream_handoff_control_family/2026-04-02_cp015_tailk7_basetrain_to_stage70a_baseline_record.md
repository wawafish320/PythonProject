# CP015 tailk7：固定 basetrain -> Stage6 tail-fix -> 当前 70a 基线记录

> Status: archived legacy upstream / handoff / control record
> Reader note: this file belongs to the old-boundary upstream-control investigation; any `current`, `default`, `canonical`, `recommend`, or `mainline` wording below is historical context, not present-tense repo policy.
> Current entry points: `docs/posttrain_pipeline.md`, `docs/train_design/2026-04-11_donor_contract_basin_universal_debug_synthesis.md`, `docs/retired_directions/legacy_upstream_handoff_control_family/README.md`

> Last updated: 2026-04-02
> 目的：把这条 `cp015 tailk7 rankmix tw020` lane 从固定 basetrain 入口到当前 `70a` 的有效状态写清楚，作为后续 debug / downstream handoff 的统一基线。

当前冻结链路应写成：

`basetrain(epoch014 fixed entry) -> Stage6 tail-fix best -> 70a(lr=3e-4)`

其中：

- `basetrain` 不再改、不重跑
- `Stage6` 当前 canonical winner 是 `lr3e4_e8x60_wd1e4_reinit1`
- `70a` 当前采用 plain cleanup 配置，沿用既有 `70a(lr=3e-4, 5x60)` 口径

---

## 0. TL;DR

本轮应固定的结论有五条：

1. 当前固定入口 basetrain ckpt 是：
   - `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth`
2. 当前 Stage6 canonical winner 是：
   - `lr3e4_e8x60_wd1e4_reinit1`
   - `all_ex_root = 0.250873`
   - `leg = 0.566078`
3. 当前 70a plain cleanup 结果是：
   - `all_ex_root = 0.213743`
   - `leg = 0.502952`
   - 相对 Stage6 winner 继续下降
4. 当前 70a 相对 baseline 链中的 `70a` 全面更好：
   - `all_ex_root / leg / arm / else` 的 `mean / p50 / p90 / p95` 全部更低
5. 未来 debug 时不要误读两件事：
   - `Stage6 reinit0` 不是 clean ablation
   - `lane.log` 的 `legω` 不是角度修正证据

---

## 1. 固定产物路径

### 1.1 固定 basetrain 入口

- fixed basetrain ckpt:
  - `models/cp015_phasecd_tailk_probe_20260331/exp_phase_DirectBranch_v1_d1_cp015_tailk7_rankmix_tw020_corridor_hold_tail15_phasea050_fixedsched_ep014center_control_denseckpt_seed2024_20260401/ckpt_epoch_014.pth`
- fixed basetrain entry summary:
  - `debug_output/_tmp_cp015_tailk7_rankmix_tw020_20260401/stage6_exact/epoch014/basetrain_group_summary.json`

### 1.2 Stage6 reference 与当前 winner

- Stage6 reference (`k7 ep014`):
  - `debug_output/_tmp_cp015_tailk7_rankmix_tw020_20260401/stage6_exact/epoch014/stage6_group_summary.json`
- Stage6 reference init stats:
  - `debug_output/_tmp_cp015_tailk7_rankmix_tw020_20260401/stage6_exact/epoch014/posttrain_stage6_init_stats.json`
- current Stage6 winner ckpt:
  - `models/__tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.pth`
- current Stage6 winner log:
  - `models/__tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit1/posttrain_log_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.json`
- current Stage6 winner init stats:
  - `debug_output/_tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit1/posttrain_stage6_init_stats.json`
- current Stage6 winner summary:
  - `debug_output/_tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit1/stage6_group_summary.json`

### 1.3 当前 70a 与 baseline 70a

- current 70a ckpt:
  - `models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth`
- current 70a log:
  - `models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/posttrain_log_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.json`
- current 70a eval:
  - `debug_output/_tmp_cp015_tailk7_stage70a_from_tailfix_20260402/eval_model_source/Walk_F_freerun_cycles.json`
- current 70a summary:
  - `debug_output/_tmp_cp015_tailk7_stage70a_from_tailfix_20260402/eval_model_source_group_summary.json`
- baseline-chain 70a ckpt:
  - `models/__tmp_posttrain_pipeline_from_bestfree_20260317/70a/ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth`
- baseline-chain 70a eval:
  - `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/70a/Walk_F_freerun_cycles.json`
- baseline-chain 70a summary:
  - `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/70a_group_summary.json`

### 1.4 历史参考 70a

这不是本轮 baseline compare 的主对照，但未来 debug 时经常会拿来参考：

- old ep014center `70a(lr=3e-4)` summary:
  - `debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/eval_model_source/lr3e4_group_summary.json`

---

## 2. 当前链路关键超参

### 2.1 Stage6 winner

- config family:
  - `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- winning override:
  - `epochs = 8`
  - `steps_per_epoch = 60`
  - `lr = 3e-4`
  - `weight_decay = 1e-4`
  - `direct_pose_reinit = true`

### 2.2 当前 70a

当前 70a 直接复用了既有 `70a(lr=3e-4)` plain cleanup 口径：

- source config:
  - `debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json`
- effective config:
  - `epochs = 5`
  - `steps_per_epoch = 60`
  - `lr = 3e-4`
  - `weight_decay = 0.0`
  - `time_index_mode = auto`
  - `phase_reset_source = none`

---

## 3. 统一口径指标快照

### 3.1 `all_ex_root`

| stage | mean | p50 | p90 | p95 |
|---|---:|---:|---:|---:|
| basetrain entry | 6.197854 | 3.225943 | 14.787715 | 20.130947 |
| Stage6 reference (`k7 ep014`) | 0.325408 | 0.155129 | 0.837841 | 1.249820 |
| Stage6 winner (`tail-fix`) | 0.250873 | 0.152024 | 0.610601 | 0.836182 |
| current 70a | 0.213743 | 0.112181 | 0.566093 | 0.783662 |
| baseline-chain 70a | 0.297430 | 0.135222 | 0.833314 | 1.124378 |

### 3.2 `leg`

| stage | mean | p50 | p90 | p95 |
|---|---:|---:|---:|---:|
| basetrain entry | 10.890589 | 9.109726 | 23.992111 | 28.185976 |
| Stage6 reference (`k7 ep014`) | 0.882395 | 0.692138 | 1.835970 | 2.304572 |
| Stage6 winner (`tail-fix`) | 0.566078 | 0.465752 | 1.081593 | 1.324971 |
| current 70a | 0.502952 | 0.423873 | 0.970880 | 1.175182 |
| baseline-chain 70a | 0.762063 | 0.640602 | 1.478543 | 1.911011 |

### 3.3 `arm`

| stage | mean | p50 | p90 | p95 |
|---|---:|---:|---:|---:|
| basetrain entry | 6.533367 | 3.720486 | 14.143064 | 18.183880 |
| Stage6 reference (`k7 ep014`) | 0.237797 | 0.117192 | 0.610986 | 0.839315 |
| Stage6 winner (`tail-fix`) | 0.201131 | 0.100175 | 0.524390 | 0.687248 |
| current 70a | 0.177822 | 0.087443 | 0.445496 | 0.629841 |
| baseline-chain 70a | 0.227739 | 0.100946 | 0.655968 | 0.907438 |

### 3.4 `else`

| stage | mean | p50 | p90 | p95 |
|---|---:|---:|---:|---:|
| basetrain entry | 1.991925 | 1.227782 | 5.415772 | 6.838516 |
| Stage6 reference (`k7 ep014`) | 0.127405 | 0.106056 | 0.282156 | 0.349386 |
| Stage6 winner (`tail-fix`) | 0.139205 | 0.120049 | 0.289054 | 0.351424 |
| current 70a | 0.088314 | 0.066898 | 0.201317 | 0.259110 |
| baseline-chain 70a | 0.124236 | 0.100114 | 0.269909 | 0.347112 |

---

## 4. 关键增量

### 4.1 `Stage6 winner - Stage6 reference`

这是 tail-fix 相对当前 `k7 ep014 Stage6 baseline` 的净收益：

| group | d_mean | d_p50 | d_p90 | d_p95 |
|---|---:|---:|---:|---:|
| `all_ex_root` | -0.074534 | -0.003105 | -0.227239 | -0.413638 |
| `leg` | -0.316317 | -0.226387 | -0.754377 | -0.979602 |
| `arm` | -0.036666 | -0.017017 | -0.086596 | -0.152067 |
| `else` | +0.011800 | +0.013993 | +0.006898 | +0.002037 |

结论：

- 主要收益来自 `leg` 和 `all_ex_root` 的右尾下降
- `arm` 也同步改善
- `else` 略有回退，但幅度很小

### 4.2 `current 70a - Stage6 winner`

这是当前 plain `70a(lr=3e-4)` 在新 Stage6 winner 上继续拿到的 cleanup：

| group | d_mean | d_p50 | d_p90 | d_p95 |
|---|---:|---:|---:|---:|
| `all_ex_root` | -0.037130 | -0.039844 | -0.044509 | -0.052520 |
| `leg` | -0.063126 | -0.041879 | -0.110713 | -0.149789 |
| `arm` | -0.023310 | -0.012732 | -0.078894 | -0.057407 |
| `else` | -0.050891 | -0.053151 | -0.087737 | -0.092313 |

结论：

- 新 70a 不是仅仅“保住 Stage6 收益”
- 它仍然是有效 cleanup step，且四个 group 全部继续下降

### 4.3 `current 70a - baseline-chain 70a`

这是当前最重要的 downstream compare：

| group | d_mean | d_p50 | d_p90 | d_p95 |
|---|---:|---:|---:|---:|
| `all_ex_root` | -0.083686 | -0.023041 | -0.267222 | -0.340717 |
| `leg` | -0.259111 | -0.216729 | -0.507663 | -0.735829 |
| `arm` | -0.049918 | -0.013503 | -0.210473 | -0.277598 |
| `else` | -0.035922 | -0.033215 | -0.068592 | -0.088002 |

结论：

- 当前 70a 明确 beat baseline 链中的 70a
- 不是单 group 侥幸，而是 `all_ex_root / leg / arm / else` 全面更低
- 主收益仍然最集中在 `leg` 和 `all_ex_root`

---

## 5. 当前应如何理解这条链

### 5.1 basetrain 不再是当前首要瓶颈

同一颗固定 basetrain 入口上：

- `Stage6 tail-fix` 已经显著优于旧 `Stage6 baseline`
- 当前 `70a` 又继续优于 baseline-chain `70a`

所以这条 lane 的主结论不是“需要先回 basetrain 才能继续看 downstream”，而是：

> 当前 basetrain 入口已经足够支撑后续 debug；真正决定效果的是 `Stage6 exit` 和其后 plain cleanup 的具体状态。

### 5.2 当前 canonical downstream handoff

如果后续要从这条 lane 往下接 replace / 70R / 71 / 72 / lambda，当前应固定使用：

`basetrain epoch014 fixed entry -> Stage6 tail-fix best -> 70a(lr=3e-4)`

不要再把 downstream 入口写成：

- `Stage6 reference k7 ep014`
- baseline-chain `70a`
- 或任何 `reinit0` 误判出来的伪对照

---

## 6. Debug caveats

### 6.1 `reinit0` 不是 clean Stage6 ablation

本轮 `Stage6 reinit0` 与 `reinit1` 的最终 exit 指标完全重合，原因不是“`direct_pose_reinit` 无关紧要”，而是：

- 入口 ckpt 的 direct head 形状是：
  - `direct_pose_hidden = 256`
  - `direct_pose_split_enable = false`
  - `direct_pose_arm_split_enable = false`
  - `direct_pose_time_pe_dim = None`
  - `direct_pose_nonleg_proj_dim = 0`
- 当前 Stage6 base config 强制的是：
  - `direct_pose_hidden_override = 512`
  - `direct_pose_split_enable = true`
  - `direct_pose_arm_split_enable = true`
  - `direct_pose_time_pe_dim = 32`
  - `direct_pose_nonleg_proj_dim = 256`

因此即使 `direct_pose_reinit=false`，`train.posttrain` 仍会因为 shape / split override 走到 `drop_direct_pose_weights`，所以：

> `lr3e4_e8x60_wd1e4_reinit0` 不能被当作 clean keep-head ablation。

### 6.2 `lane.log` 的 `legω` 不是角度 clip 证据

后续如果再看 Stage6 / 70a / downstream lane log，必须继续遵守这个解释约束：

- `legω` 表示的是 `direct_grad_norm_leg_branch`
- 它不是 `so3_corr_max_deg`
- 它也不是角度 ceiling 命中证据

如果要声称“撞 ceiling”，必须回到：

- 真实角度字段
- 或明确 config / 代码证据

### 6.3 Stage6 winner 的梯度分配仍然不均

当前 winner 的 Stage6 init stats 仍显示：

- `step1 leg_over_nonleg = 2.678241`
- `step1 grad_arm_over_else = 8.227071`
- `head20 leg_over_nonleg = 3.224530`
- `head20 grad_arm_over_else = 7.527935`

这说明 tail-fix 虽然已经把 exit 压得很好，但并不意味着训练内梯度分配已经“完全健康”。

未来如果 downstream 再出现新的 group-specific regression，这里仍然是优先检查点。

---

## 7. 本轮 70a 复现实跑命令

### 7.1 train

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_ep014center_70a_lowlr_sweep_20260328/configs/posttrain_70a_lr3e4_from_ep014center_20260328.json \
  --ckpt_in models/__tmp_cp015_tailk7_stage6_tailfix_20260401/lr3e4_e8x60_wd1e4_reinit1/ckpt_last_lr3e4_e8x60_wd1e4_reinit1_stage6_tailfix_tailk7_stage6_tailfix_20260401.pth \
  --out_dir models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402 \
  --run_name WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0
```

### 7.2 eval

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth \
  --rounds 5 \
  --depth 3 \
  --time-index-mode cycle \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source model \
  --lambda_fusion_apply \
  --log_contacts \
  --export_direct_arm_probe \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_cp015_tailk7_stage70a_from_tailfix_20260402/eval_model_source \
  --force
```

### 7.3 group summary

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  debug_output/_tmp_cp015_tailk7_stage70a_from_tailfix_20260402/eval_model_source/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out debug_output/_tmp_cp015_tailk7_stage70a_from_tailfix_20260402/eval_model_source_group_summary.json
```

---

## 8. 当前 baseline 记录的一句话版本

未来如果只需要一句话提醒当前状态，可以直接写：

> 当前 `cp015 tailk7 rankmix tw020` lane 已固定为 `basetrain epoch014 -> Stage6 tail-fix best(lr3e-4,e8x60,wd1e-4,reinit1) -> 70a(lr=3e-4)`，且该 `70a` 已明确优于 baseline-chain `70a`。
