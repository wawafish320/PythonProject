# CP015 tailk7 replace：factorized interface translation / no-shared-adapter falsifier

> Archived on 2026-04-12.  
> Current role: historical negative evidence for the factorized-interface / split-adapter branch, not a live optimization direction.  
> Reader guidance: phrases like “当前 control / 固定判定” below refer to the 2026-04-04 experiment context only; current posttrain policy should be read from `docs/posttrain_pipeline.md`.

> Last updated: 2026-04-04
> 目的：在当前 `e3x60_adapter_factorized` control 上，只改一处——把 shared direct input adapter 换成 `leg/arm/else` 三路 factorized interface translation，回答 shared translated feature / upstream hidden 是否仍是主要剩余问题。

---

## 0. TL;DR

本轮只做了一个最小新实验：

- 基线：`tailk7 e3x60 adapter factorized readout`
- 唯一变量：`shared direct input adapter -> leg/arm/else factorized adapters`
- 其余保持不变：
  - 不换 donor
  - 不改 replace / 70R 语义
  - 不改 warmstart
  - 不改 loss / optimizer / wd / data / batch / seq_len / seed / epochs / rollout_cycles / event_clock / phase_reset_source / contacts_source
  - 不改 factorized readout trunk hidden / output head / proj dim / target 语义

结果：

- 相对当前 control `e3x60_adapter_factorized`
  - `arm p95`: `0.544815 -> 0.589489` 变差
  - `all_ex_root p95`: `0.610157 -> 0.651536` 变差
  - `leg p95`: `0.834997 -> 0.964328` 明显变差
- 虽然 follow-up 里仍能看到少数 `step_in_cycle` / 少数 arm bones 的局部改善，但全局 freerun 没有变好，且主判定指标明显回退。

因此本轮固定判定：

- **B. factorized interface translation 也基本救不回来，single shared adapter / translated feature 不是主要剩余问题**

---

## 1. 新 case 与产物

- config:
  - `debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_factorized_adaptersplit_lr5e5_from_cp015_tailk7_70a_20260404.json`
- ckpt:
  - `models/__tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/e3x60_adapter_factorized_adaptersplit/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_adaptersplit_lr5e5_from_cp015_tailk7_70a_20260404.pth`
- train log:
  - `models/__tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/e3x60_adapter_factorized_adaptersplit/posttrain_log_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_adaptersplit_lr5e5_from_cp015_tailk7_70a_20260404.json`
- eval json:
  - `debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_adaptersplit/Walk_F_freerun_cycles.json`
- group summary:
  - `debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_adaptersplit_group_summary.json`
- follow-up analysis:
  - `debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/analysis/e3x60_adapter_factorized_vs_adaptersplit_followup.json`
- 汇总:
  - `debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/analysis/summary.json`

---

## 2. 唯一实现改动

### 2.1 精确插入点

插在当前 shared adapter 原位置，即：

- `direct_pose_feat_source` 选出 `direct_feat`
- **在拼接 `time_pe / plan / meas` 之前**
- 用 `leg/arm/else` 三个 zero-init residual adapters 分别生成三路 translated feature
- 再分别拼成三路 `direct_flat`
- 送入现有 factorized readout 的 `direct_pose_head_leg / arm / else`

代码位置：

- `train/models.py:3751`
- `train/models.py:3787`
- `train/models.py:2177`

### 2.2 输入输出 shape

当前 case `direct_pose_feat_source=cond`，所以三路 adapter 都直接读同一个 pre-adapter donor feature：

- `arm adapter`: `(B,T,7) -> (B,T,7)`
- `leg adapter`: `(B,T,7) -> (B,T,7)`
- `else adapter`: `(B,T,7) -> (B,T,7)`

当前 factorized case 的 downstream 输入不变：

- `translated_feat`: `7`
- `time_pe`: `32`
- `plan`: `2`
- `meas`: `2`
- 所以 `arm/leg/else` 各自 `direct_flat shape = (B*T,43)`

当前 factorized readout 输出不变：

- `leg_out_dim = 48`
- `arm_out_dim = 156`
- `else_out_dim = 74`

### 2.3 被移除/绕开的 shared path

被绕开的 shared path 只有一条：

- `direct_pose_input_adapter.*`

新 ckpt 中：

- 不再有 `direct_pose_input_adapter.*`
- 只有：
  - `direct_pose_input_adapter_leg.*`
  - `direct_pose_input_adapter_arm.*`
  - `direct_pose_input_adapter_else.*`

### 2.4 为什么这是最小 no-shared-adapter

因为它只替换了：

- `shared translated feature`

而没有改：

- factorized readout trunk / head
- branch input 语义
- `time_pe / plan / meas` 拼接语义
- loss / target / routing / stopgrad / bypass

也就是说，这不是新的 routing 变体，而是当前 factorized readout control 上最小的 “去 shared adapter” 替换。

---

## 3. 参数量

单个 shared adapter 参数量：

- `3861`

三路 factorized adapters：

- 每路 `3861`
- 合计 `11583`

相对 `e3x60_adapter_factorized`：

- 新增参数量：`+7722`

总 trainable 参数：

- control `e3x60_adapter_factorized`: `1503299`
- new `e3x60_adapter_factorized_adaptersplit`: `1511021`

---

## 4. 实际运行命令

### 4.1 train

```bash
PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_factorized_adaptersplit_lr5e5_from_cp015_tailk7_70a_20260404.json \
  --ckpt_in models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/warmstart/ckpt_last_cp015_tailk7_70a_replace_zerophase_20260402.pth \
  --out_dir models/__tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/e3x60_adapter_factorized_adaptersplit \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_adaptersplit_lr5e5_from_cp015_tailk7_70a_20260404 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 4.2 eval

```bash
PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/e3x60_adapter_factorized_adaptersplit/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_adaptersplit_lr5e5_from_cp015_tailk7_70a_20260404.pth \
  --rounds 5 --depth 3 --time-index-mode cycle --event_clock auto --phase_reset_source none \
  --contacts_meas_source model --lambda_fusion_apply --log_contacts \
  --export_direct_arm_probe --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_adaptersplit \
  --force
```

### 4.3 group summary / follow-up

```bash
python3 tools/phasea_group_summary.py \
  debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_adaptersplit/Walk_F_freerun_cycles.json \
  --cycle_gte 1 --drop_wrap \
  --out debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_adaptersplit_group_summary.json

python3 tools/compare_factorized_readout_followup.py \
  --baseline debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/eval_model_source/e3x60_adapter_factorized/Walk_F_freerun_cycles.json \
  --candidate debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_adaptersplit/Walk_F_freerun_cycles.json \
  --cycle_gte 1 --drop_wrap --topk 10 \
  --out debug_output/_tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/analysis/e3x60_adapter_factorized_vs_adaptersplit_followup.json
```

---

## 5. Train 结果

stdout epoch average：

- epoch1: `1.491008`
- epoch2: `1.841523`
- epoch3: `1.969442`

final logged row（step=180）：

- `total = 1.773771`
- `blend_loss = 0.327080`
- `dir_leg_base = 0.002671`
- `dir_nonleg_base = 0.000924`
- `leg_over_nonleg = 2.891333`

---

## 6. Freerun 指标与并表

下表统一为 `mean / p90 / p95`。

说明：

- 这里的 `tailk7 e3x60 lr=5e-5` 与前一份 adapter / routing 文档保持一致，
  固定引用：
  - `debug_output/_tmp_cp015_tailk7_replace_schedule_ablation_20260402/eval_model_source/e3x60_group_summary.json`
- **不是**
  - `debug_output/_tmp_cp015_tailk7_replace_from_70a_20260402/eval_model_source/lr5e5_group_summary.json`
  那条 `plain replace-from-70a lr5e5` 结果。

| case | arm | all_ex_root | leg | else |
|---|---|---|---|---|
| tailk7 70a | 0.177822 / 0.445496 / 0.629841 | 0.213743 / 0.566093 / 0.783662 | 0.502952 / 0.970880 / 1.175182 | 0.088314 / 0.201317 / 0.259110 |
| tailk7 e3x60 lr=5e-5 | 0.160056 / 0.414565 / 0.554550 | 0.182774 / 0.461205 / 0.622943 | 0.379237 / 0.745277 / 0.890515 | 0.093588 / 0.200817 / 0.288886 |
| tailk7 e3x60 lr=1e-5 falsifier | 0.168610 / 0.430735 / 0.608705 | 0.192825 / 0.484252 / 0.690631 | 0.407328 / 0.808446 / 0.970266 | 0.094060 / 0.201514 / 0.296829 |
| tailk7 e3x60 adapter | 0.156052 / 0.415701 / 0.550443 | 0.175801 / 0.438493 / 0.611407 | 0.364820 / 0.738802 / 0.882486 | 0.085013 / 0.187801 / 0.261784 |
| tailk7 e3x60 adapter leg-bypass | 0.154278 / 0.400737 / 0.562478 | 0.178262 / 0.442584 / 0.630359 | 0.384593 / 0.775811 / 0.971611 | 0.084890 / 0.190240 / 0.274835 |
| tailk7 e3x60 adapter factorized readout | 0.160344 / 0.425562 / 0.544815 | 0.177592 / 0.450439 / 0.610157 | 0.351382 / 0.712495 / 0.834997 | 0.091967 / 0.198024 / 0.288770 |
| tailk7 e3x60 adapter factorized adaptersplit | 0.171442 / 0.444734 / 0.589489 | 0.188930 / 0.473916 / 0.651536 | 0.388438 / 0.751077 / 0.964328 | 0.085167 / 0.200968 / 0.285388 |
| baseline replace | 0.116105 / 0.303279 / 0.423788 | 0.152126 / 0.402586 / 0.567555 | 0.391665 / 0.795561 / 0.998558 | 0.063055 / 0.140461 / 0.177474 |

相对当前 control `e3x60_adapter_factorized`：

- `arm`: `+0.011098 / +0.019172 / +0.044674`
- `all_ex_root`: `+0.011338 / +0.023476 / +0.041379`
- `leg`: `+0.037056 / +0.038581 / +0.129331`
- `else`: `-0.006800 / +0.002945 / -0.003382`

---

## 7. 证明 factorized adapter 确实启用

### 7.1 log/config 证据

见：

- `models/__tmp_cp015_tailk7_replace_factorized_adapter_falsifier_20260404/e3x60_adapter_factorized_adaptersplit/posttrain_log_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_adaptersplit_lr5e5_from_cp015_tailk7_70a_20260404.json:95`
  - `direct_pose_input_adapter_enable = false`
- 同文件 `:96`
  - `direct_pose_factorized_input_adapter_enable = true`
- 同文件 `:160`-`:165`
  - `direct_pose_factorized_input_adapter_enabled = true`
  - `in_dim = 7`
  - `total_param_count = 11583`
  - `delta_vs_shared_adapter = 7722`
- 同文件 `:173`
  - `direct_pose_factorized_readout_in_dim = 43`

### 7.2 ckpt key 证据

新 ckpt 中：

- `direct_pose_input_adapter_leg.*` 存在
- `direct_pose_input_adapter_arm.*` 存在
- `direct_pose_input_adapter_else.*` 存在
- `direct_pose_input_adapter.*` 不存在

这正对应 “去 shared adapter，只保留 arm/leg/else 三路 translated feature”。

---

## 8. Follow-up（不加新实验）

相对 `e3x60_adapter_factorized`，仍能观察到少数局部改善：

- arm `step_in_cycle` 最佳 p95 改善 top5：
  - `83 (-0.1148)`
  - `22 (-0.1087)`
  - `84 (-0.0953)`
  - `85 (-0.0883)`
  - `42 (-0.0623)`
- `all_ex_root` 最佳 p95 改善 top5：
  - `83 (-0.1516)`
  - `4 (-0.1449)`
  - `84 (-0.1395)`
  - `22 (-0.1087)`
  - `21 (-0.1031)`
- arm bone p95 最佳 top5：
  - `hand_l (-0.0826)`
  - `thumb_01_l (-0.0812)`
  - `RUpArmTwist_l_01 (-0.0571)`
  - `hand_r (-0.0477)`
  - `RUpArmTwist_r_01 (-0.0426)`
- arm bone p95 最差 top5：
  - `lowerarm_l (+0.1172)`
  - `middle_01_l (+0.0188)`
  - `upperarm_l (+0.0136)`
  - `upperarm_r (+0.0123)`
  - `clavicle_r (+0.0083)`

`direct_arm_probe` norm 对比：

- `direct_in`: 基本不变（`+0.00031`）
- `trunk_hidden`: `-0.0221`
- `proj_pre0`: `-0.0316`
- `out_in`: `-0.0284`
- `arm_out`: `-0.0295`

结论仍然一样：有局部 patch，但不构成全局 freerun 改善。

---

## 9. 最终结论

- 按约定判定标准，必须看 freerun，不能看 train loss 小幅变化。
- 本轮 `arm p95`、`all_ex_root p95` 都没有继续下降，`leg` 还明显恶化。

因此最终只保留一个结论：

- **B. factorized interface translation 也基本救不回来，single shared adapter / translated feature 不是主要剩余问题**
