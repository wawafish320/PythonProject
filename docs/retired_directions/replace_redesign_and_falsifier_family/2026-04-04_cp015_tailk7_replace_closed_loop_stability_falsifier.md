# CP015 tailk7 replace：minimal closed-loop stability falsifier analysis

> Archived on 2026-04-12.  
> Current role: historical mechanism-analysis record for the retired tailk7 closed-loop falsifier branch, not a live debugging directive.  
> Reader guidance: “当前最该优先推进 / 下一步唯一合理” wording below should be read as 2026-04-07 local conclusions inside the old-boundary investigation, not as current repo-level policy.

> Last updated: 2026-04-07
> 目的：不再做新的 downstream architecture falsifier，只回答一个最小问题：
> `tailk7` 当前主要剩余问题，是否更像 closed-loop robustness / bad geometry / low plasticity，而不是静态信息不足。

---

## 0. TL;DR

本轮只做分析，不开新训练，不改结构，不改 donor，不改 warmstart，不改 replace / 70R 语义，不改 loss / optimizer / data / seed / epochs。

固定对照只看两个 case：

1. `tailk7 e3x60 adapter factorized readout`
2. `baseline replace`

本轮最稳妥的结论是：

- **A. 现有证据强支持 tailk7 的剩余 gap 主要是 closed-loop 下对 hints/signals 的使用更 brittle，而不是静态信息不足；bad geometry / low plasticity / robustness 是当前最被支持的解释族，但还不是被单独锁死的唯一根因。**

支持点有两条，而且两条是同方向的：

1. `teacher-conditioned one-step` 不是 tailk7 的主要矛盾。
   - tailk7 在 teacher-conditioned runtime path 上只表现为轻微更差；
   - 但在 free-run `d0` 上反而 **不差于** baseline，甚至更好。
2. 真正的差距出现在 rollout 深度 `10+` 和 mid-cycle 区间；
   同时 tailk7 对 `contacts_meas` / `plan` 的局部小扰动 gain 远大于 baseline。

这更符合：

- static readable information 足够
- 但 closed-loop 使用这些信号时更 brittle

而不符合：

- “只是静态一步信息不足”

---

## 1. 本轮边界

本轮明确不做：

- 新训练结构实验
- 新 readout / adapter / routing / stopgrad / bypass 变体
- donor 变更
- replace / 70R 语义改动
- warmstart 改动
- loss / optimizer / data / batch / seq_len / seed / epochs 改动

已知前提，本轮不重复证明：

- strict leg linear probe 已否掉 “tailk7 donor trunk hidden 被破坏”
- donor-specific trunk hidden -> arm probe 说明 tailk7 的 arm 静态可读性更强，不是更差
- LR falsifier 已否掉 “只是 LR 太大”
- adapter 说明 interface translation 有帮助
- factorized readout 只做局部 tail 改善，不是全局解
- factorized adapter 变差，说明 split adapter 不是方向

因此这轮只做两类 analysis：

- A. teacher-forced vs free-run gap 分解
- B. local sensitivity / gain audit

---

## 2. 固定输入与产物

### 2.1 两个 case

#### tailk7 factorized control

- ckpt:
  `models/__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/e3x60_adapter_factorized/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth`
- eval:
  `debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/eval_model_source/e3x60_adapter_factorized/Walk_F_freerun_cycles.json`
- group summary:
  `debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_group_summary.json`

#### baseline replace

- ckpt:
  `models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth`
- eval:
  `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/new70b_replace_lowdrift/Walk_F_freerun_cycles.json`
- group summary:
  `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/new70b_replace_lowdrift_group_summary.json`

### 2.2 共同输入 teacher batch

- teacher batch:
  `validate/teacher_batches/Walk_F_teacher.json`

### 2.3 本轮新增最小脚本

- gap analysis:
  `tools/analyze_cp015_tailk7_closed_loop_gap.py`
- local sensitivity:
  `tools/analyze_cp015_tailk7_local_sensitivity.py`

### 2.4 结果产物

- A 结果:
  `debug_output/_tmp_cp015_tailk7_closed_loop_stability_analysis_20260404/gap_analysis.json`
- B 结果:
  `debug_output/_tmp_cp015_tailk7_closed_loop_stability_analysis_20260404/local_sensitivity.json`

---

## 3. 实际运行命令

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_closed_loop_gap.py
python3 tools/analyze_cp015_tailk7_closed_loop_gap.py
python3 -m py_compile tools/analyze_cp015_tailk7_local_sensitivity.py
python3 tools/analyze_cp015_tailk7_local_sensitivity.py
```

说明：

- 本轮没有新训练命令。
- A/B 都是直接读取现有 ckpt + teacher/eval 产物做最小分析。

---

## 4. A. Teacher-forced vs Free-run gap 分解

### 4.1 用到的工具与定义

优先复用了现有 runtime eval 工具：

- `train/validate/run_freerun_cycles.py`

而不是重写另一套 direct metric path。

原因：

- `DirectGeoLocalDeg` 的 freerun metric 不只是裸 `out_direct`
- 它还经过 runtime direct hint source、leg correction、arm residual correction 等路径
- 如果 teacher-side 不走同一 runtime wrapper，数字会不可比

所以本轮 teacher 指标定义为：

- **teacher-conditioned runtime pass**
- driver:
  `_run_freerun_cycles`
- 参数：
  - `rounds=1`
  - `freerun_x_gt=True`
  - `pose_hist_source='seq'`
  - `pose_hist_update_source='gt'`
- metric:
  `DirectGeoLocalDeg`

free-run 指标定义为：

- driver:
  `_run_freerun_cycles`
- 参数：
  - `rounds=5`
  - `pose_hist_source='buffer'`
  - `pose_hist_update_source='pred'`
- metric:
  `DirectGeoLocalDeg`

因此 A 部分 teacher / freerun 现在走的是同一 runtime metric path。

### 4.2 one-step 并表

单位：`deg`

#### mean

| metric | arm tail | arm base | arm Δ | all_ex_root tail | all_ex_root base | all_ex_root Δ | leg tail | leg base | leg Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher-conditioned one-step | 0.159679 | 0.114374 | +0.045305 | 0.177395 | 0.150446 | +0.026949 | 0.352989 | 0.388808 | -0.035819 |
| free-run `d0` | 0.168583 | 0.199847 | -0.031263 | 0.153638 | 0.207496 | -0.053858 | 0.200736 | 0.401127 | -0.200391 |

#### p95

| metric | arm tail | arm base | arm Δ | all_ex_root tail | all_ex_root base | all_ex_root Δ | leg tail | leg base | leg Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher-conditioned one-step | 0.538378 | 0.418585 | +0.119793 | 0.610819 | 0.567169 | +0.043650 | 0.834007 | 0.993021 | -0.159014 |
| free-run `d0` | 0.575457 | 0.773901 | -0.198444 | 0.471748 | 0.823264 | -0.351516 | 0.401347 | 0.862810 | -0.461463 |

直接读数：

- tail 在 teacher-conditioned one-step 上，`arm` / `all_ex_root` 只轻微更差
- 但在 free-run step 0，tail 反而 **优于** baseline
- 所以 “tailk7 一开始就静态一步更差” 不是主导事实

### 4.3 rollout depth bucket

单位：`deg`，表中为 `mean`

| bucket | arm tail | arm base | arm Δ | all_ex_root tail | all_ex_root base | all_ex_root Δ | leg tail | leg base | leg Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `d10_19` | 0.199958 | 0.109416 | +0.090541 | 0.216169 | 0.142106 | +0.074063 | 0.379205 | 0.373331 | +0.005874 |
| `d20_43` | 0.176291 | 0.091524 | +0.084767 | 0.207636 | 0.121772 | +0.085864 | 0.440297 | 0.322128 | +0.118170 |
| `d44_86` | 0.149632 | 0.115420 | +0.034212 | 0.165128 | 0.155836 | +0.009293 | 0.337277 | 0.406361 | -0.069084 |
| `d87_173` | 0.164548 | 0.117301 | +0.047247 | 0.183552 | 0.153681 | +0.029871 | 0.366462 | 0.394832 | -0.028371 |
| `d174_346` | 0.162458 | 0.115867 | +0.046592 | 0.180589 | 0.152096 | +0.028493 | 0.358965 | 0.391892 | -0.032927 |
| `d347_433` | 0.164548 | 0.117301 | +0.047247 | 0.183552 | 0.153681 | +0.029871 | 0.366461 | 0.394832 | -0.028371 |

关键点：

- 从 `d10_19` 开始，tail 在 `arm` / `all_ex_root` 上持续高于 baseline
- `d20_43` 是最清楚的发散区间：
  - arm `+0.0848`
  - all_ex_root `+0.0859`
  - leg `+0.1182`

### 4.4 step-in-cycle bucket

单位：`deg`，表中为 `mean`

| bucket | arm tail | arm base | arm Δ | all_ex_root tail | all_ex_root base | all_ex_root Δ | leg tail | leg base | leg Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sic0_10` | 0.163668 | 0.183914 | -0.020246 | 0.174498 | 0.226209 | -0.051711 | 0.313354 | 0.542689 | -0.229334 |
| `sic11_21` | 0.205786 | 0.107637 | +0.098149 | 0.227415 | 0.136931 | +0.090484 | 0.416384 | 0.356306 | +0.060078 |
| `sic22_43` | 0.172621 | 0.091715 | +0.080906 | 0.202974 | 0.122364 | +0.080609 | 0.432192 | 0.323892 | +0.108300 |
| `sic44_64` | 0.179374 | 0.109927 | +0.069447 | 0.187074 | 0.150054 | +0.037020 | 0.362709 | 0.394581 | -0.031872 |
| `sic65_86` | 0.117518 | 0.118418 | -0.000900 | 0.139117 | 0.158909 | -0.019792 | 0.300650 | 0.413147 | -0.112497 |

关键点：

- 主要差距集中在 mid-cycle：
  - `sic11_21`
  - `sic22_43`
- 而不是 cycle 一开始就落后

### 4.5 teacher -> rollout 的变化量

拿 `teacher-conditioned one-step` 与 `d20_43` 比较：

| group | tail: teacher -> d20_43 | baseline: teacher -> d20_43 | delta of deltas |
|---|---:|---:|---:|
| arm | +0.016612 | -0.022850 | +0.039463 |
| all_ex_root | +0.030241 | -0.028674 | +0.058915 |
| leg | +0.087308 | -0.066681 | +0.153989 |

解释：

- baseline 从 teacher-conditioned one-step 到 `d20_43` 是下降的
- tail 反而是上升的
- 因此 tail 的主要问题不是 “teacher-conditioned step 就更差很多”
- 而是 rollout 中更容易放大误差

### 4.6 A 部分最小判读

固定结论：

- tailk7 **不是** 静态一步就明显更差
- tailk7 的主要差距更像出现在闭环 rollout 中对 hints/signals 的使用更 brittle

---

## 5. B. Local sensitivity / gain audit

### 5.1 用到的工具与定义

新脚本：

- `tools/analyze_cp015_tailk7_local_sensitivity.py`

工作点：

- 同一 `Walk_F_teacher.json`
- 在 teacher-conditioned rollout working point 上评估 `out_direct`

扰动对象：

- `pose_history`
- `contacts_meas`
- `plan`

输出对象：

- `out_direct`

metric：

- perturbed `out_direct` 相对 base `out_direct` 的 group-mean local geodesic delta，单位 `deg`

gain 定义：

- `0.5 * (d_plus + d_minus) / ||delta_input||_2`

扰动规模：

- `||delta_input||_2 = 0.05`

说明：

- `contacts_meas` 通过 `model.direct_pose_meas_override`
- `plan` 通过 `model.direct_pose_plan_override`
- 这样做是因为两个 eval json 都明确是：
  - `direct_pose_meas_source=model`
  - `direct_pose_plan_source=model`
  - `contacts_meas_source=model`

### 5.2 gain 并表

单位：`deg / input_l2`

| input -> out_direct | arm tail | arm base | arm Δ | all_ex_root tail | all_ex_root base | all_ex_root Δ | leg tail | leg base | leg Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pose_history` | 2.886e-06 | 6.900e-08 | +2.817e-06 | 3.164e-06 | 6.784e-08 | +3.096e-06 | 5.373e-06 | 8.829e-08 | +5.285e-06 |
| `contacts_meas` | 0.129316 | 0.004831 | +0.124485 | 0.139066 | 0.005479 | +0.133587 | 0.244495 | 0.010735 | +0.233760 |
| `plan` | 0.123823 | 0.003451 | +0.120371 | 0.135672 | 0.003922 | +0.131750 | 0.248856 | 0.007737 | +0.241119 |

### 5.3 p95 量级

| input -> out_direct | arm tail p95 | arm base p95 | all_ex_root tail p95 | all_ex_root base p95 |
|---|---:|---:|---:|---:|
| `contacts_meas` | 0.207213 | 0.010394 | 0.207550 | 0.011891 |
| `plan` | 0.226088 | 0.006903 | 0.225639 | 0.007681 |

### 5.4 B 部分最小判读

固定事实：

- `pose_history` 局部 gain 对两边都几乎是 0，不是主要区分点
- tail 对 `contacts_meas` / `plan` 的局部 gain 明显高于 baseline
- 而且这个现象同时出现在：
  - `arm`
  - `all_ex_root`
  - `leg`

因此 tail 的 direct branch 对 closed-loop hint 的使用更 brittle。

---

## 6. 事实优先的总判定

把 A/B 合起来，只保留最小判读：

1. teacher-conditioned one-step 上，tail 只轻微更差；
   但 free-run `d0` 上 tail 反而不差于 baseline。
2. 从 rollout depth `10+` 开始，tail 在 `arm` / `all_ex_root` 上稳定落后。
3. mid-cycle `sic11_21` / `sic22_43` 是最明显的发散区。
4. tail 对 `contacts_meas` / `plan` 的局部 gain 远高于 baseline。

因此本轮最稳妥的判定是：

- **A. 现有证据更支持 tailk7 的剩余问题主要是 closed-loop hint-usage brittleness / robustness，而不是静态信息不足；bad geometry / low plasticity 是当前最合理的解释族，但还没有被单独证明为唯一根因。**

不是：

- **B. teacher-forced 一步就已经明显差**

也不是：

- **C. A/B 两边 sensitivity / gap 没有清晰差异**

---

## 7. 下一步约束

如果后续只允许开一个训练 falsifier，方向应当限定为：

- 直接针对 closed-loop robustness / plasticity

而不是再回头做：

- static probe
- downstream readout / adapter / routing 变体

## 8. Direct-Only Hint Robustness Falsifier Follow-up

### 8.1 动机

这轮只测试一个最小问题：

- 如果只降低 direct branch 对 `contacts_meas` / `plan` 的高增益依赖，当前 `tailk7 e3x60 adapter factorized readout` 的 freerun 剩余 gap 是否会收敛。

固定边界：

- 不改 donor
- 不改 replace / 70R 语义
- 不改 warmstart
- 不改 loss / optimizer / wd / data / batch / seq_len / seed / epochs / rollout_cycles
- 不改 event_clock / phase_reset_source / contacts source
- 不改 eval contract
- 不开第二个配置

### 8.2 唯一变量

control：

- `debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.json`

new config：

- `debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_factorized_directhintrobust_lr5e5_from_cp015_tailk7_70a_20260404.json`

相对 control，只加三项 direct-only hint robustness regularization：

- `direct_pose_meas_noise_std = 0.03`
- `direct_pose_meas_drop_prob = 0.05`
- `direct_pose_plan_drop_prob = 0.05`

补充说明：

- 首次按上述 config 开跑时，发现 `train/posttrain.py` 在 model 构建处把这三项硬编码成 `0.0`，导致 run 实际等价于 control。
- 因此本轮做了一个最小 plumbing 修复：只把这三项现有参数从 posttrain config 透传到 `EventMotionModel`，不新增机制、不改其它语义，然后用同一份单 config 重新跑完 train / eval / group summary。

### 8.3 实际命令

```bash
python3 -m py_compile train/posttrain.py

PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_factorized_directhintrobust_lr5e5_from_cp015_tailk7_70a_20260404.json \
  --ckpt_in models/__tmp_cp015_tailk7_replace_schedule_ablation_20260402/warmstart/ckpt_last_cp015_tailk7_70a_replace_zerophase_20260402.pth \
  --out_dir models/__tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/e3x60_adapter_factorized_directhintrobust \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_directhintrobust_lr5e5_from_cp015_tailk7_70a_20260404 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json

PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/e3x60_adapter_factorized_directhintrobust/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_directhintrobust_lr5e5_from_cp015_tailk7_70a_20260404.pth \
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
  --out debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_directhintrobust \
  --force

PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  tools/phasea_group_summary.py \
  debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_directhintrobust/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_directhintrobust_group_summary.json
```

### 8.4 产物路径

train ckpt：

- `models/__tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/e3x60_adapter_factorized_directhintrobust/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_directhintrobust_lr5e5_from_cp015_tailk7_70a_20260404.pth`

train log：

- `models/__tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/e3x60_adapter_factorized_directhintrobust/posttrain_log_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_directhintrobust_lr5e5_from_cp015_tailk7_70a_20260404.json`

eval：

- `debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_directhintrobust/Walk_F_freerun_cycles.json`

group summary：

- `debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_directhintrobust_group_summary.json`

comparison cache：

- `debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/analysis/summary.json`

### 8.5 训练事实

effective run 的 stdout epoch average：

- epoch1: `1.552665`
- epoch2: `1.853396`
- epoch3: `1.980315`

final logged row（step=`180`）：

- `total = 1.616433`
- `blend_loss = 0.341223`
- `dir_leg_base = 0.002213`
- `dir_nonleg_base = 0.000991`
- `leg_over_nonleg = 2.232025`

透传生效确认：

- saved posttrain config 中三项值分别为 `0.03 / 0.05 / 0.05`
- new ckpt 与 control ckpt 的 `model` 权重已有差异：
  - diff tensor keys: `34`
  - max abs diff: `0.008671`
  - max diff key: `direct_pose_head_arm.3.weight`

### 8.6 与 control 并表

单位：`deg`

| group | metric | control | new | baseline replace | new - control |
|---|---|---:|---:|---:|---:|
| arm | mean | 0.160344 | 0.169649 | 0.116105 | +0.009305 |
| arm | p90 | 0.425562 | 0.437013 | 0.303279 | +0.011451 |
| arm | p95 | 0.544815 | 0.601363 | 0.423788 | +0.056548 |
| all_ex_root | mean | 0.177592 | 0.180089 | 0.152126 | +0.002497 |
| all_ex_root | p90 | 0.450439 | 0.445548 | 0.402586 | -0.004891 |
| all_ex_root | p95 | 0.610157 | 0.616924 | 0.567555 | +0.006767 |
| leg | mean | 0.351382 | 0.347248 | 0.391665 | -0.004134 |
| leg | p90 | 0.712495 | 0.662923 | 0.795561 | -0.049572 |
| leg | p95 | 0.834997 | 0.850677 | 0.998558 | +0.015680 |

直接读数：

- `arm` 三个槽位全部变差，尤其 `p95` 从 `0.544815 -> 0.601363`
- `all_ex_root` 只有 `p90` 小幅变好，`mean/p95` 仍变差
- `leg mean/p90` 有改善，但 `leg p95` 反而变差
- 相对 baseline 的 arm / all_ex_root gap 没有收敛，反而扩大

### 8.7 最终判定

- **B. 改善不明显，说明 high-gain hint usage 更像症状，不是主导瓶颈。**

理由只保留最小事实：

- 这次 direct-only robustness regularization 已经真实生效，不是“参数没进模型”
- 但 freerun 并没有出现面向剩余 gap 的一致性改善
- 非腿尤其 `arm` / `all_ex_root` 没有收敛，主要尾部槽位还更差

如果后续还允许跟进一次 analysis，建议只补一个 “生效后训练态 local sensitivity/gain 是否真的被压下去” 的对照读数，不要再开第二个结构 sweep。

### 8.8 Gain Follow-up

为了解开上一轮 `B` 判定里的剩余歧义，这里补一个最小 gain 对照读数：

- 问题不是再看 freerun 指标，而是直接看 new run 相对 control，`contacts_meas / plan -> out_direct` 的 local gain 是否真的下降。

实际命令：

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_local_sensitivity.py

PYTHONPATH=. python3 tools/analyze_cp015_tailk7_local_sensitivity.py \
  --tail-ckpt models/__tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/e3x60_adapter_factorized_directhintrobust/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_directhintrobust_lr5e5_from_cp015_tailk7_70a_20260404.pth \
  --tail-eval debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/eval_model_source/e3x60_adapter_factorized_directhintrobust/Walk_F_freerun_cycles.json \
  --baseline-ckpt models/__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/e3x60_adapter_factorized/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth \
  --baseline-eval debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/eval_model_source/e3x60_adapter_factorized/Walk_F_freerun_cycles.json \
  --out debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/analysis/local_sensitivity_vs_control.json
```

产物：

- `debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/analysis/local_sensitivity_vs_control.json`
- `debug_output/_tmp_cp015_tailk7_replace_direct_hint_robustness_falsifier_20260404/analysis/gain_followup_summary.json`

单位：`deg / input_l2`

#### contacts_meas gain mean

| group | control | new | new/control | delta |
|---|---:|---:|---:|---:|
| arm | 0.125155 | 0.112731 | 0.9007 | -0.012424 |
| all_ex_root | 0.136204 | 0.125277 | 0.9198 | -0.010926 |
| leg | 0.242366 | 0.229941 | 0.9487 | -0.012425 |

#### plan gain mean

| group | control | new | new/control | delta |
|---|---:|---:|---:|---:|
| arm | 0.122415 | 0.116968 | 0.9555 | -0.005447 |
| all_ex_root | 0.136498 | 0.132126 | 0.9680 | -0.004372 |
| leg | 0.256815 | 0.251919 | 0.9809 | -0.004896 |

#### p95 补充

| target | group | control p95 | new p95 | delta |
|---|---|---:|---:|---:|
| contacts_meas | arm | 0.206639 | 0.180421 | -0.026218 |
| contacts_meas | all_ex_root | 0.204074 | 0.185037 | -0.019037 |
| contacts_meas | leg | 0.329477 | 0.318757 | -0.010721 |
| plan | arm | 0.229792 | 0.217562 | -0.012230 |
| plan | all_ex_root | 0.239002 | 0.227502 | -0.011499 |
| plan | leg | 0.362698 | 0.374014 | +0.011316 |

最小判读：

- 这次 direct-only robustness regularization **确实把 gain 压下去了一点**，不是“完全没打中”。
- 下降幅度主要是：
  - `contacts_meas`: 约 `5%` 到 `10%`
  - `plan`: 约 `2%` 到 `5%`
- 但 freerun 并没有同步改善，反而 `arm` / `all_ex_root` 更差。

因此这一步让上一轮结论更具体：

- `high-gain hint usage` 不是完全无关现象；
- 但在当前 tailk7 剩余 gap 里，它更像伴随症状或次级放大项，而不是主导瓶颈。

更直接地说：

- **gain 被压下去一些了，但 gap 没收敛。**
- 所以后续主线不应再优先放在 downstream hint-path regularization 上，而应转向 donor closed-loop dynamics / rollout state propagation 本身。

## 9. Single-Step Rescue Audit Follow-up

上一节把 `hint-path high gain` 基本降级为症状之后，剩余最值得直接测试的问题变成：

- 当前 tailk7 的 mid-cycle / deep-rollout 失败，是否主要来自 **donor closed-loop state propagation** 本身。

为此，这里只做一个最小 `no-train` rescue audit：

- 固定 control：
  `tailk7 e3x60 adapter factorized readout`
- 只在已知坏窗口中，对单个内部状态做 **one-step teacher rescue**
- 看 rescue 之后接下来 `5 / 20` 步的 `used_local_geo_deg` 是否稳定改善

如果某个状态的一步 teacher rescue 能一致改善后续 freerun，那么它更像是当前闭环失败的因果状态；反之则更像旁路症状或次级变量。

### 9.1 固定输入与新增脚本

control ckpt：

- `models/__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/e3x60_adapter_factorized/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth`

control eval：

- `debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/eval_model_source/e3x60_adapter_factorized/Walk_F_freerun_cycles.json`

teacher batch：

- `validate/teacher_batches/Walk_F_teacher.json`

新增脚本：

- `tools/analyze_cp015_tailk7_single_step_rescue.py`

输出：

- `debug_output/_tmp_cp015_tailk7_single_step_rescue_audit_20260404/summary.json`

### 9.2 实际运行命令

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_single_step_rescue.py

PYTHONPATH=. python3 tools/analyze_cp015_tailk7_single_step_rescue.py \
  --ckpt models/__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/e3x60_adapter_factorized/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth \
  --eval debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/eval_model_source/e3x60_adapter_factorized/Walk_F_freerun_cycles.json \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --drop-wrap \
  --out debug_output/_tmp_cp015_tailk7_single_step_rescue_audit_20260404/summary.json
```

### 9.3 选择窗口与 rescue 定义

固定只看 control 中已知坏区间：

- `depth >= 10`
- `step_in_cycle in [11, 43]`
- `drop_wrap = True`

最终共选中：

- `165` 个 step
- 聚合为 `33` 个 window

metric 定义：

- `used_local_geo_deg`
- 即 carried rollout pose `y_used_raw` 对 GT `y[t+1]` 的 local geodesic error
- group 只看 `arm / all_ex_root / leg`

单步 rescue 类型：

- `none`：不做 rescue，作为实现自洽性 sanity check
- `pose_history`：当前 step 前把 `PoseHistState` 换成 teacher-conditioned 版本
- `plan_z`：当前 step 前把 planner latent `plan_z` 换成 teacher-conditioned 版本
- `h_final`：在当前 step 用 forward hook 把 `model.coupling_norm` 后的 `h_final` 换成 teacher-conditioned 版本，然后正常继续 freerun

### 9.4 Horizon 5：window-level improvement mean

单位：`deg`，正数表示 rescue 后误差下降

| rescue | arm | all_ex_root | leg |
|---|---:|---:|---:|
| none | +0.0000 | +0.0000 | +0.0000 |
| pose_history | -0.0033 | -0.0001 | +0.0153 |
| plan_z | -0.0102 | -0.0096 | -0.0226 |
| h_final | +0.0191 | +0.0130 | +0.0199 |

### 9.5 Horizon 20：window-level improvement mean

单位：`deg`，正数表示 rescue 后误差下降

| rescue | arm | all_ex_root | leg |
|---|---:|---:|---:|
| none | +0.0000 | +0.0000 | +0.0000 |
| pose_history | -0.0074 | -0.0010 | +0.0203 |
| plan_z | -0.0229 | -0.0188 | -0.0278 |
| h_final | +0.0320 | +0.0211 | +0.0178 |

### 9.6 Horizon 20：positive-rate

定义：`33` 个 window 里，improvement `> 0` 的比例

| rescue | arm | all_ex_root | leg |
|---|---:|---:|---:|
| none | 0.0000 | 0.0000 | 0.0000 |
| pose_history | 0.0303 | 0.5152 | 0.8788 |
| plan_z | 0.0303 | 0.0606 | 0.0000 |
| h_final | 0.9091 | 0.8485 | 0.7879 |

### 9.7 最小判读

先看实现自洽性：

- `none` rescue 在 `horizon 5 / 20` 上 improvement 全为 `0`
- 说明 snapshot / restore / continuation 逻辑是对的，这个 audit 没有被实现误差污染

再看三类内部状态：

- `pose_history` 基本不能 rescue `arm / all_ex_root`，只对 `leg` 有小幅帮助
- `plan_z` 在两个 horizon 上都稳定变差，说明把 planner latent 拉回 teacher manifold 不是一个有效恢复杆
- `h_final` 是唯一一个在所有 group 上都给出一致正向 rescue 的状态，且 `horizon 20` 的 positive-rate 很高

所以这一步把假设空间进一步缩窄到：

- 当前 tailk7 剩余 gap 更像是 **donor hidden-state trajectory / shared hidden rollout state** 的闭环传播问题
- 而不是 `pose_history` 状态本身
- 也不是 `plan_z` 本身
- 更不是下游 hint-path gain 这类已经被前面 falsifier 降级的症状项

### 9.8 本节结论

这轮 `single-step rescue audit` 给出的方向性信号很干净：

- **真正能救回来的是 `h_final`，不是 `pose_history`，也不是 `plan_z`。**

因此按这份 2026-04-07 记录当时的判断，最该优先推进的方向，不再是新增 downstream 训练 falsifier，而是进一步直接审计：

- donor 在 closed-loop rollout 中的 hidden-state drift / trajectory geometry / state-manifold stability

如果只允许再跟进一个最小 analysis，建议做：

- 对 bad windows 里的 `h_final` 做 teacher-vs-freerun drift audit，直接量化 hidden 轨迹从 teacher manifold 偏离的深度、速度与可恢复性。

## 10. h_final Drift Audit Follow-up

上一节把唯一还没直接量化的窄假设收敛到：

- tailk7 的剩余 freerun gap，是否已经主要表现为 donor shared hidden `h_final` 在 closed-loop rollout 中偏离 teacher manifold 后持续传播。

这轮仍然是最小 `no-train` follow-up：

- 不开新训练
- 不改 donor
- 不改任何模型结构 / readout / adapter / routing / stopgrad / bypass / hint path
- 不改 replace / 70R / warmstart / loss / optimizer / data / seed / eval contract

唯一变量：

- 在 **同一 multicycle runtime path** 上，对同一 step 比较：
  - `h_final_free(t)`：实际 freerun 当前步 hidden
  - `h_final_teacher(t)`：teacher-conditioned 当前步 hidden
- 并额外导出 `plan_z_in` 的 teacher-vs-freerun trace，只用于检查它是否时间上领先于 `h_final` drift。

### 10.1 固定输入、脚本与产物

control ckpt：

- `models/__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/e3x60_adapter_factorized/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth`

control eval：

- `debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/eval_model_source/e3x60_adapter_factorized/Walk_F_freerun_cycles.json`

teacher batch：

- `validate/teacher_batches/Walk_F_teacher.json`

新增脚本：

- `tools/analyze_cp015_tailk7_hfinal_drift.py`

summary：

- `debug_output/_tmp_cp015_tailk7_hfinal_drift_audit_20260404/summary.json`

### 10.2 实际运行命令

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_hfinal_drift.py

PYTHONPATH=. python3 tools/analyze_cp015_tailk7_hfinal_drift.py \
  --ckpt models/__tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/e3x60_adapter_factorized/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.pth \
  --eval debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/eval_model_source/e3x60_adapter_factorized/Walk_F_freerun_cycles.json \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --drop-wrap \
  --out debug_output/_tmp_cp015_tailk7_hfinal_drift_audit_20260404/summary.json
```

说明：

- 这次没有复用上一节那个单周期 rescue loop 去做主统计。
- 为了让 step 轴真正对齐 `Walk_F_freerun_cycles.json` 的 `434` 个 rollout steps，这次直接复用了 `train/validate/run_freerun_cycles.py::_run_freerun_cycles` 的 **multicycle runtime path**。
- `h_final` 通过 `model.coupling_norm` forward hook 抓取；`plan_z` 通过现成的 `plan_state_series.plan_z_in` 导出。

### 10.3 选择窗口

与 single-step rescue follow-up 保持同一 aperture：

- `depth >= 10`
- `step_in_cycle in [11, 43]`
- `drop_wrap = True`

最终共选中：

- `165` 个 steps（`5` 个 cycle，每个 cycle `33` 个 step）

### 10.4 h_final drift：offset 0 / 1 / 5 / 20

单位：

- `normalized L2 = ||h_free - h_teacher||_2 / sqrt(D)`
- `cosine distance = 1 - cos(h_free, h_teacher)`

| offset | samples | norm L2 mean | norm L2 p90 | norm L2 p95 | cos mean | cos p90 | cos p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 165 | 0.4382 | 0.6303 | 0.6421 | 0.1123 | 0.1995 | 0.2071 |
| 1 | 165 | 0.4385 | 0.6303 | 0.6421 | 0.1125 | 0.1995 | 0.2071 |
| 5 | 165 | 0.4537 | 0.6354 | 0.6648 | 0.1191 | 0.2028 | 0.2220 |
| 20 | 165 | 0.4861 | 0.6680 | 0.6952 | 0.1334 | 0.2242 | 0.2428 |

直接读数：

- `h_final` drift 不是只在 bad step 当下出现；它在后续 rollout 中继续增大。
- `0 -> 5 -> 20` 两个指标都单调变大。

### 10.5 h_final drift 增长率

单位同上，表中为 `delta = drift(later) - drift(earlier)`。

| growth span | norm L2 mean | norm L2 p90 | norm L2 p95 | cos mean | cos p90 | cos p95 |
|---|---:|---:|---:|---:|---:|---:|
| `0 -> 5` | +0.0155 | +0.0751 | +0.1093 | +0.0068 | +0.0359 | +0.0474 |
| `5 -> 20` | +0.0324 | +0.1348 | +0.1785 | +0.0142 | +0.0566 | +0.0655 |
| `0 -> 20` | +0.0479 | +0.1535 | +0.1791 | +0.0211 | +0.0578 | +0.0599 |

关键点：

- 增长主要不是停留在 `0 -> 1` 的瞬时跳变；
- 更强的是 `5 -> 20` 继续放大，这更像闭环 state propagation，而不是一步局部 noise。

### 10.6 deep-rollout 对齐：按 cycle 看 selected-window drift

单位：`normalized L2 mean`

| cycle | offset 0 | offset 5 | offset 20 |
|---|---:|---:|---:|
| 0 | 0.1699 | 0.1905 | 0.2271 |
| 1 | 0.3104 | 0.3330 | 0.3812 |
| 2 | 0.4947 | 0.5106 | 0.5444 |
| 3 | 0.5854 | 0.5955 | 0.6194 |
| 4 | 0.6305 | 0.6388 | 0.6584 |

这组读数非常直接：

- 同一 bad-window aperture 下，随着 closed-loop 深度增加，`h_final` drift 系统性升高；
- 不是某个单一 cycle 的偶发异常。

### 10.7 时间读数：d10-d20 / sic11-sic43

这部分单独回答“是否在 `d10-d20` 加速、`sic11-sic43` 最严重”。

#### 选中窗口按 depth bucket

单位：`normalized L2 mean`

| base bucket | samples | offset 0 | offset 5 | offset 20 | growth 0->5 | growth 5->20 |
|---|---:|---:|---:|---:|---:|---:|
| `d10_20` | 10 | 0.1749 | 0.1368 | 0.1447 | -0.0381 | +0.0079 |
| `d21_43` | 23 | 0.1677 | 0.2138 | 0.2629 | +0.0461 | +0.0491 |
| `d87_173` | 33 | 0.3104 | 0.3330 | 0.3812 | +0.0226 | +0.0482 |
| `d174_433` | 99 | 0.5702 | 0.5816 | 0.6074 | +0.0114 | +0.0258 |

结论：

- **`d10_20` 不是最干净的加速段。**
- 更清楚的窗口相对加速出现在：
  - `d21_43`（首个 cycle 的 mid-window 后半段）
  - 后续更深 cycles 里持续维持正增长

#### 选中窗口按 `step_in_cycle` 子区间

单位：`normalized L2 mean`；`future error` 为 freerun `used_local_geo_deg` 的 `horizon 20` mean，单位 `deg`

| sic bucket | samples | offset 0 | offset 5 | offset 20 | growth 0->5 | growth 5->20 | arm h20 err | all_ex_root h20 err | leg h20 err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sic11_21` | 55 | 0.4246 | 0.4116 | 0.4395 | -0.0131 | +0.0280 | 69.95 | 71.04 | 71.86 |
| `sic22_43` | 110 | 0.4449 | 0.4747 | 0.5094 | +0.0298 | +0.0347 | 74.28 | 75.31 | 76.08 |

结论：

- 在当前审计 aperture 内，**更重的其实是后半段 `sic22_43`，不是前半段 `sic11_21`。**
- 所以“`sic11_43` 是坏窗口”这个大口径仍然成立；
- 但更细一点看，严重度主要压在 `22-43`，不是整个区间均匀一致。

### 10.8 drift 与后续误差的关联

这里直接看 `h_final drift` 和后续 freerun `used_local_geo_deg` 的相关性。

#### Pearson r：drift at offset 0 vs future horizon-20 error

| group | norm L2 r | cosine r |
|---|---:|---:|
| arm | 0.9729 | 0.9695 |
| all_ex_root | 0.9744 | 0.9717 |
| leg | 0.9725 | 0.9704 |

#### Pearson r：drift at offset 20 vs future horizon-20 error

| group | norm L2 r | cosine r |
|---|---:|---:|
| arm | 0.9329 | 0.9363 |
| all_ex_root | 0.9368 | 0.9472 |
| leg | 0.9327 | 0.9479 |

补一个更直观的对齐读数：

- 以 `offset 0 norm-L2 drift` 做四分位分桶后，
  top quartile 相比 bottom quartile 的 `horizon 20` freerun error 平均高出：
  - arm：`+88.14 deg`
  - all_ex_root：`+89.60 deg`
  - leg：`+90.70 deg`

这说明：

- 不只是“相关系数好看”；
- 高 drift window 和后续误差恶化在量级上也是强对齐的。

### 10.9 plan_z 补充导出

这轮按要求也导出了 `plan_z` 的 teacher-vs-freerun drift。

#### plan_z offset drift

| offset | samples | norm L2 mean | norm L2 p90 | norm L2 p95 |
|---|---:|---:|---:|---:|
| 0 | 165 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 165 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 165 | 0.0000 | 0.0000 | 0.0000 |
| 20 | 165 | 0.0000 | 0.0000 | 0.0000 |

lead/lag 结果：

- `best_lag = 0`
- `best_pearson_r = NaN`

解释：

- 不是导出缺失；summary 里 `plan_z` 有 `433` 个有效 steps
- 但 freerun 与 teacher-conditioned 的 `plan_z_in` 在这个 control 上是 **数值上相同**
- 因此不存在“`plan_z` drift 领先 `h_final` drift”这条证据

这和上一节 rescue 的结论是同方向的：

- `plan_z` 不是当前有效恢复杆
- 现在也看不到它在时间上先坏掉

### 10.10 最终判定

- **A. `h_final` drift 在 bad windows 中显著增长，且和后续误差恶化一致，支持 donor hidden-state rollout dynamics 是主导瓶颈。**

严格限定一下这条 A 的含义：

- 最强证据来自：
  - selected-window `offset 0 / 5 / 20` 的单调 drift 增长
  - 按 cycle 的系统性升高
  - `h_final drift` 与后续 `horizon 5 / 20` freerun error 的超高相关
  - `plan_z` drift 为零，因此没有可替代的领先解释
- 但子问题里“是否恰好在 `d10_20` 加速、`sic11_43` 整段最严重”并不是完全成立：
  - `d10_20` 不是最干净的加速段
  - 在审计 aperture 内，更重的是 `sic22_43`

不影响主结论的原因是：

- 当前最核心的问题已经不是“哪一个 phase bin 最糟”；
- 而是 **teacher manifold 上的 donor shared hidden trajectory 一旦偏离，后续 closed-loop 误差会持续传播并且高度对齐到实际 freerun 恶化。**

## 11. Donor Hidden-Dynamics Follow-up

### 11.1 动机

这轮按 follow-up 要求只做一个 donor-side continuation：

- 起点固定为当前 `tailk7 70a` donor
- 不改 downstream 结构，不重训 downstream
- 只训练形成 `h_final` 的 donor shared temporal/coupling trunk
- 在保留 donor 现有 rollout objective 的前提下，额外加一个最小 `h_final` teacher-vs-freerun consistency` 约束
- 然后把 **完全相同的 current downstream control** 重新挂到新 donor 上，检查 freerun 是否自行改善

这轮要回答的不是 “drift 能不能被压下去”，而是：

- **压下去以后，固定 control 的 freerun gap 会不会同步下降。**

### 11.2 Donor Trainable Subset

实际只解冻了下面这组 donor trunk 模块：

- `shared_encoder`
- `residual_proj`
- `_pasa_lnq`
- `_pasa_q`
- `_pasa_k`
- `_pasa_v`
- `_pasa_o`
- `_pasa_film`
- `coupling_norm`

实际 trainable 参数量：

- `2,477,568`

确认未训练的 donor/downstream 模块：

- `motion_head`: unchanged
- `contact_plan_*`: unchanged
- `direct_pose_*`: unchanged

这满足“只训练形成 `h_final` 的 shared temporal/coupling trunk；尽量冻结下游 heads”。

### 11.3 新增 Donor-Side Loss 定义

这轮 continuation 保留 donor 70a 原有的 direct rollout objective，但把 direct/readout heads 全冻结，只让梯度回到 trunk。

新增的 donor hidden-dynamics auxiliary 写成：

```text
L_total
  = L_direct_frozen_head
  + w_h * L_hfinal_aux

L_hfinal_aux
  = mean_{Δ in {1,5,20}} L_focus(Δ)
  + w_global * L_global

L_focus(Δ)
  = mean over anchor t:
      cycle(t) >= 1
      11 <= step_in_cycle(t) <= 43
      of  MSE(h_free[t+Δ], stopgrad(h_teacher[t+Δ]))
        + α * (1 - cos(h_free[t+Δ], stopgrad(h_teacher[t+Δ])))

L_global
  = mean over all rollout steps of the same hidden discrepancy
```

实际超参：

- `w_h = 1.0`
- `w_global = 0.15`
- `α = 0.25`
- focus offsets: `1 / 5 / 20`
- focus mask: `cycle >= 1`, `step_in_cycle in [11, 43]`

这符合“priority 放在 bad aperture，但保留小的全局权重避免只修局部窗口”。

### 11.4 实际训练命令

```text
historical note: this stage originally used a dedicated donor-hidden-dynamics follow-up script; that probe was later deleted during the 2026-04-18 posttrain compat cleanup.
```

这个脚本实际做了两件事：

- donor-only continuation training
- 训练结束后，把 **当前 control ckpt 的 `direct_pose_*` 权重原样 transplant 到新 donor 上**，生成固定 downstream control 的 composite ckpt

### 11.5 实际 Eval 命令

```bash
python3 -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/e3x60_adapter_factorized_control_on_donor_hfinal/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_control_on_donor_hfinal_20260404.pth \
  --rounds 5 \
  --depth 3 \
  --device cpu \
  --time-index-mode cycle \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source model \
  --direct_pose_meas_source model \
  --direct_pose_plan_source model \
  --pose_hist_source buffer \
  --pose_hist_update_source pred \
  --lambda_fusion_apply \
  --log_contacts \
  --export_direct_arm_probe \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/eval_model_source/e3x60_adapter_factorized_control_on_donor_hfinal \
  --force
```

group summary：

```bash
python3 tools/phasea_group_summary.py \
  debug_output/_tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/eval_model_source/e3x60_adapter_factorized_control_on_donor_hfinal/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --out debug_output/_tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/eval_model_source/e3x60_adapter_factorized_control_on_donor_hfinal_group_summary.json
```

### 11.6 实际 Drift Audit 命令

```bash
python3 tools/analyze_cp015_tailk7_hfinal_drift.py \
  --ckpt models/__tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/e3x60_adapter_factorized_control_on_donor_hfinal/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_control_on_donor_hfinal_20260404.pth \
  --eval debug_output/_tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/eval_model_source/e3x60_adapter_factorized_control_on_donor_hfinal/Walk_F_freerun_cycles.json \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --device cpu \
  --out debug_output/_tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/hfinal_drift_summary.json
```

### 11.7 新 Artifact 路径

- 新 donor ckpt  
  `models/__tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/donor_hfinal_trunk_anchor/ckpt_last_WalkF_stage7_70a_hfinal_dynamics_trunk_anchor_lr3e-05_e1x60_20260404.pth`
- 新 downstream composite ckpt  
  `models/__tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/e3x60_adapter_factorized_control_on_donor_hfinal/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_control_on_donor_hfinal_20260404.pth`
- 新 downstream eval  
  `debug_output/_tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/eval_model_source/e3x60_adapter_factorized_control_on_donor_hfinal/Walk_F_freerun_cycles.json`
- 新 downstream group summary  
  `debug_output/_tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/eval_model_source/e3x60_adapter_factorized_control_on_donor_hfinal_group_summary.json`
- 新 drift summary  
  `debug_output/_tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/hfinal_drift_summary.json`
- donor follow-up summary  
  `debug_output/_tmp_cp015_tailk7_donor_hidden_dynamics_followup_20260404/summary.json`

### 11.8 关键对比表

#### donor-side `h_final` drift：offset `0 / 5 / 20`

单位：`normalized L2 mean`

| metric | current control | donor-stabilized control | delta |
|---|---:|---:|---:|
| offset `0` | 0.4382 | 0.1662 | -0.2719 |
| offset `5` | 0.4537 | 0.1792 | -0.2745 |
| offset `20` | 0.4861 | 0.2105 | -0.2756 |

#### donor-side `h_final` drift growth：`0->5 / 5->20`

单位：`normalized L2 delta mean`

| metric | current control | donor-stabilized control | delta |
|---|---:|---:|---:|
| growth `0->5` | 0.0155 | 0.0130 | -0.0025 |
| growth `5->20` | 0.0324 | 0.0313 | -0.0012 |

#### downstream freerun 指标前后对比

单位：`DirectGeoLocalDeg`

| group | mean before | mean after | delta | p90 before | p90 after | delta | p95 before | p95 after | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arm | 0.1603 | 0.1635 | +0.0032 | 0.4256 | 0.4300 | +0.0045 | 0.5448 | 0.5560 | +0.0112 |
| all_ex_root | 0.1776 | 0.1821 | +0.0045 | 0.4504 | 0.4562 | +0.0058 | 0.6102 | 0.6169 | +0.0067 |
| leg | 0.3514 | 0.3627 | +0.0113 | 0.7125 | 0.7277 | +0.0152 | 0.8350 | 0.8736 | +0.0386 |

#### drift-vs-error 相关性前后对比

这里取和主诊断一致的读数：`offset 0 drift` vs `future horizon-20 error` 的 `Pearson r (norm_l2)`

| group | r before | r after | delta |
|---|---:|---:|---:|
| arm | 0.9729 | 0.5905 | -0.3824 |
| all_ex_root | 0.9744 | 0.5964 | -0.3780 |
| leg | 0.9725 | 0.5935 | -0.3790 |

### 11.9 结果解释

这轮 follow-up 给出了一个很硬的事实组合：

- donor hidden-dynamics auxiliary **确实显著压低了 `h_final` drift**
- 压低幅度不是只在单一点上：
  - `offset 0 / 5 / 20` 全部下降约 `0.27`
  - `5->20` 的增长也略降
- 但在 **完全不改 downstream 结构、只回贴同一个 control 的 `direct_pose_*` 权重** 之后：
  - `arm / all_ex_root / leg` freerun 指标都没有改善
  - 三组 `mean / p90 / p95` 反而全部小幅变差

也就是说：

- “把 donor shared hidden 拉回 teacher manifold” 这件事本身是能做成的；
- 但它 **没有自动转化为当前 fixed control 的 freerun 恢复**。

这轮还带来一个附加信号：

- `drift-vs-error` 的相关性从 `~0.97` 掉到 `~0.59`
- 说明这次 donor continuation 改变了 hidden closed-loop behavior，
  但 downstream error 的主瓶颈并没有被一起解除

更直接地说：

- 这轮不是 “donor hidden drift 根本压不下去”；
- 而是 **“drift 压下去了，但固定 control 还是没吃到收益”**。

### 11.10 最终 A/B 判定

- **B. donor-side hidden-dynamics stabilization 没有带来对应的 fixed-control freerun 改善。**

因此这轮的结论不是回到 downstream readout 大方向重做，而是更具体：

- 单靠 donor shared hidden 的闭环稳定化，**不足以解释并修复** 当前 `tailk7 e3x60 adapter factorized readout` 的剩余 freerun gap
- donor-side 假设还需要继续细分 donor 内部机制，至少要进一步区分：
  - “shared hidden drift” 本身
  - 与当前 fixed control `direct_pose_*` readout 所需几何方向之间的兼容性
  - donor trunk 内部哪些子块在 stabilizing 后改变了 control-facing feature geometry，但没有转成 lower freerun error

所以：

- 当前证据不支持 A
- 这轮 follow-up 支持 **B**

## 12. Donor Dynamics Gain Follow-up

### 12.1 本轮动机

上一轮 donor-only `h_final` anchoring 已经给出清楚的 negative result：

- `offset drift` 能明显压低
- 但 `0->5 / 5->20` growth 基本没动
- 在固定当前 downstream control 下，freerun 没改善

所以这轮不再做 absolute `||h_free - h_teacher||` anchoring，而是只做一个最小 objective-only donor continuation，直接把 donor 目标改成 **teacher-conditioned vs freerun 的 hidden transition/span consistency**，优先尝试动 closed-loop gain / growth dynamics。

同时按要求把评估拆成两条 lane：

- `lane F`: 新 donor + frozen current downstream control
- `lane C`: 新 donor + 同配置 downstream posttrain rerun（donor frozen）

本轮没有引入 donor internal low-gain 结构改动。原因很简单：

- objective-only lane 能正常跑通
- 用户要求不要顺手开第二个 donor 变体
- 因此这轮先把“objective-only 是否真能动 growth”这个问题做成闭环

### 12.2 Donor Trainable Subset

这轮 donor continuation 仍然只训练形成 `h_final` 的最小 trunk 子集：

- `shared_encoder`
- `residual_proj`
- `_pasa_lnq`
- `_pasa_q`
- `_pasa_k`
- `_pasa_v`
- `_pasa_o`
- `_pasa_film`
- `coupling_norm`

实际 trainable 参数量：

- `2,477,568`

对应 trainable tensor 共 `28` 个：

- `shared_encoder.0.weight`, `shared_encoder.0.bias`
- `shared_encoder.1.weight`, `shared_encoder.1.bias`
- `shared_encoder.4.weight`, `shared_encoder.4.bias`
- `shared_encoder.5.weight`, `shared_encoder.5.bias`
- `shared_encoder.8.norm.weight`, `shared_encoder.8.norm.bias`
- `shared_encoder.8.fc1.weight`, `shared_encoder.8.fc1.bias`
- `shared_encoder.8.fc2.weight`, `shared_encoder.8.fc2.bias`
- `residual_proj.weight`, `residual_proj.bias`
- `_pasa_q.weight`, `_pasa_k.weight`, `_pasa_v.weight`, `_pasa_o.weight`
- `_pasa_lnq.weight`, `_pasa_lnq.bias`
- `_pasa_film.fc1.weight`, `_pasa_film.fc1.bias`
- `_pasa_film.fc2.weight`, `_pasa_film.fc2.bias`
- `coupling_norm.weight`, `coupling_norm.bias`

这仍然满足“只动 donor hidden transition dynamics，不动 downstream 结构”的边界。

### 12.3 新 Donor-Side Dynamics Loss 定义

这轮 donor objective 写成：

```text
L_total
  = L_direct_frozen_head
  + w_tr * L_transition

L_transition
  = mean_{Δ in {1,5,20}} L_focus(Δ)
  + w_global * L_global

L_focus(Δ)
  = mean over anchor t:
      cycle(t) >= 1
      11 <= step_in_cycle(t) <= 43
      of [
        MSE(Δh_free[t,Δ], stopgrad(Δh_teacher[t,Δ]))
        + α * (1 - cos(Δh_free[t,Δ], stopgrad(Δh_teacher[t,Δ])))
        + β * abs(||Δh_free[t,Δ]|| - ||Δh_teacher[t,Δ]||)
      ]

Δh[t,Δ] = h[t+Δ] - h[t]

L_global
  = same transition/span discrepancy
    averaged over all rollout anchors and all Δ in {1,5,20}
```

实际超参：

- `w_tr = 2.0`
- `w_global = 0.10`
- `α = 0.25`
- `β = 0.50`
- focus offsets: `Δ1 / Δ5 / Δ20`
- focus mask: `cycle >= 1`, `step_in_cycle in [11, 43]`

关键点：

- 主目标明确转向 `transition / span`
- 没再把 absolute hidden anchoring 当主目标
- 额外的 `||Δh||` gap 项是为了更直接约束 gain / growth magnitude，而不是只对齐方向

### 12.4 实际运行命令

#### donor continuation + lane F 组装

```text
historical note: this stage originally used a dedicated donor-dynamics-gain follow-up script; that probe was later deleted during the 2026-04-18 posttrain compat cleanup.
```
  --focus-cycle-min 1 \
  --focus-sic-lo 11 \
  --focus-sic-hi 43 \
  --force
```

这条命令实际做了三件事：

- 跑 donor objective-only continuation
- 生成 `lane F` composite ckpt（新 donor + frozen current `direct_pose_*`）
- 生成 `lane C` copy-only warmstart

#### lane F eval

```bash
PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/laneF_frozen_current_control/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_frozen_current_control_on_transition_gain_20260404.pth \
  --rounds 5 \
  --depth 3 \
  --time-index-mode cycle \
  --event_clock auto \
  --phase_reset_source none \
  --contacts_meas_source model \
  --direct_pose_meas_source model \
  --direct_pose_plan_source model \
  --pose_hist_source buffer \
  --pose_hist_update_source pred \
  --lambda_fusion_apply \
  --log_contacts \
  --export_direct_arm_probe \
  --export_joint_direct_geolocal_series \
  --out debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneF_frozen_current_control \
  --force
```

lane F group summary：

```bash
python3 tools/phasea_group_summary.py \
  debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneF_frozen_current_control/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneF_frozen_current_control_group_summary.json
```

#### lane C downstream posttrain

```bash
PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  -m train.posttrain \
  --config debug_output/_tmp_cp015_tailk7_replace_factorized_readout_falsifier_20260404/configs/posttrain_70b_replace_lowdrift_e3x60_adapter_factorized_lr5e5_from_cp015_tailk7_70a_20260404.json \
  --ckpt_in models/__tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/warmstart/ckpt_last_cp015_tailk7_70a_transition_gain_replace_zerophase_20260404.pth \
  --out_dir models/__tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/laneC_coadapt_posttrain \
  --run_name WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_coadapt_on_transition_gain_20260404 \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

实际 log 明确显示：

- `mode=train_direct_pose`
- `trainable=34 params`
- `direct_pose_feat_source='cond'`
- `direct_pose_factorized_readout_enable=true`

也就是说 `lane C` 确实是 donor frozen，只让当前 downstream factorized control 按原 config/budget 重新适配。

#### lane C eval

```bash
PYTHONPATH=. python3 debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py \
  -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/__tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/laneC_coadapt_posttrain/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_coadapt_on_transition_gain_20260404.pth \
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
  --out debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneC_coadapt_posttrain \
  --force
```

lane C group summary：

```bash
python3 tools/phasea_group_summary.py \
  debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneC_coadapt_posttrain/Walk_F_freerun_cycles.json \
  --cycle_gte 1 \
  --drop_wrap \
  --out debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneC_coadapt_posttrain_group_summary.json
```

#### 复跑现有 h_final drift audit

```bash
python3 tools/analyze_cp015_tailk7_hfinal_drift.py \
  --ckpt models/__tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/laneF_frozen_current_control/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_frozen_current_control_on_transition_gain_20260404.pth \
  --eval debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneF_frozen_current_control/Walk_F_freerun_cycles.json \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --device cpu \
  --drop-wrap \
  --out debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/hfinal_drift_summary.json
```

### 12.5 新 Artifact 路径

- 新 donor ckpt  
  `models/__tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/donor_transition_gain/ckpt_last_WalkF_stage7_70a_transition_gain_objective_lr3e-05_e2x60_20260404.pth`
- lane F frozen composite ckpt  
  `models/__tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/laneF_frozen_current_control/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_frozen_current_control_on_transition_gain_20260404.pth`
- lane F eval  
  `debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneF_frozen_current_control/Walk_F_freerun_cycles.json`
- lane F group summary  
  `debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneF_frozen_current_control_group_summary.json`
- lane C warmstart  
  `models/__tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/warmstart/ckpt_last_cp015_tailk7_70a_transition_gain_replace_zerophase_20260404.pth`
- lane C ckpt  
  `models/__tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/laneC_coadapt_posttrain/ckpt_last_WalkF_stage7_70b_replace_lowdrift_e3x60_adapter_factorized_coadapt_on_transition_gain_20260404.pth`
- lane C eval  
  `debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneC_coadapt_posttrain/Walk_F_freerun_cycles.json`
- lane C group summary  
  `debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/eval_model_source/laneC_coadapt_posttrain_group_summary.json`
- 新 drift summary  
  `debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/hfinal_drift_summary.json`
- follow-up summary  
  `debug_output/_tmp_cp015_tailk7_donor_dynamics_gain_followup_20260404/summary.json`

### 12.6 关键对比表

#### donor-side `h_final` drift：offset `0 / 5 / 20`

单位：`normalized L2 mean`

| metric | current control | dynamics gain follow-up | delta |
|---|---:|---:|---:|
| offset `0` | 0.4382 | 0.4018 | -0.0364 |
| offset `5` | 0.4537 | 0.4291 | -0.0246 |
| offset `20` | 0.4861 | 0.4842 | -0.0020 |

#### donor-side `h_final` drift growth：`0->5 / 5->20`

单位：`normalized L2 delta mean`

| metric | current control | dynamics gain follow-up | delta |
|---|---:|---:|---:|
| growth `0->5` | 0.0155 | 0.0273 | +0.0118 |
| growth `5->20` | 0.0324 | 0.0551 | +0.0226 |

结论先写在表下：

- 这轮 objective-only 虽然把 `offset 0 / 5` 稍微往下拉了
- 但真正要打的 `growth` 不但没降，反而明显变大
- 所以它没有命中 donor closed-loop gain / transition dynamics 这个目标

#### lane F freerun：`arm / all_ex_root / leg`

单位：`DirectGeoLocalDeg`

| group | mean before | mean lane F | delta | p90 before | p90 lane F | delta | p95 before | p95 lane F | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arm | 0.1603 | 0.1603 | +0.0000 | 0.4256 | 0.4256 | +0.0000 | 0.5448 | 0.5448 | +0.0000 |
| all_ex_root | 0.1776 | 0.1776 | +0.0000 | 0.4504 | 0.4504 | +0.0000 | 0.6102 | 0.6102 | +0.0000 |
| leg | 0.3514 | 0.3514 | +0.0000 | 0.7125 | 0.7125 | +0.0000 | 0.8350 | 0.8350 | +0.0000 |

这不是统计巧合，而是一个结构事实：

- `lane C` 日志里当前 control config 仍然是 `direct_pose_feat_source='cond'`
- 所以 frozen lane 基本没有把 donor hidden 改动重新暴露到当前 direct readout contract 里
- 换句话说，`lane F` 在这个具体 control contract 下几乎退化成 “no-op compatibility check”

#### lane C freerun：`arm / all_ex_root / leg`

单位：`DirectGeoLocalDeg`

| group | mean before | mean lane C | delta | p90 before | p90 lane C | delta | p95 before | p95 lane C | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arm | 0.1603 | 0.1638 | +0.0035 | 0.4256 | 0.4291 | +0.0036 | 0.5448 | 0.5803 | +0.0355 |
| all_ex_root | 0.1776 | 0.1810 | +0.0034 | 0.4504 | 0.4561 | +0.0057 | 0.6102 | 0.6194 | +0.0092 |
| leg | 0.3514 | 0.3582 | +0.0068 | 0.7125 | 0.7098 | -0.0027 | 0.8350 | 0.8612 | +0.0262 |

#### frozen vs co-adapt 差值

单位：`lane C - lane F`

| group | mean delta | p90 delta | p95 delta |
|---|---:|---:|---:|
| arm | +0.0035 | +0.0036 | +0.0355 |
| all_ex_root | +0.0034 | +0.0057 | +0.0092 |
| leg | +0.0068 | -0.0027 | +0.0262 |

### 12.7 结果解释

这轮最关键的事实组合是：

1. objective-only transition/span loss 没有把 donor `growth` 压下去
   - `offset` 有小幅下降
   - 但 `0->5 / 5->20` growth 明显上升

2. `lane F` 与当前 control freerun 指标完全重合
   - 这说明在 **当前 factorized control contract** 下，frozen lane 几乎没有把 donor 改动暴露到 `DirectGeoLocalDeg`
   - 具体证据是当前 rerun config 仍然是 `direct_pose_feat_source='cond'`

3. `lane C` 在 donor frozen、只做 downstream re-adapt 的前提下，freerun 仍没有改善
   - `arm / all_ex_root / leg mean` 全部变差
   - `arm / leg p95` 也都变差

所以这轮不支持“donor dynamics 已经打中，只是 downstream mismatch 挡住了收益”。

更准确地说：

- 新 objective 的确改变了 donor hidden 行为
- 但它改变的方向不是 lower-growth
- downstream 即使按同配置重适配，也没有从这个 donor substrate 上拿到 freerun 收益

### 12.8 最终 A / B / C 判定

- **C. 这轮 intervention 连 growth rate 都没能实质动到，说明必须上更直接的 donor internal gain mechanism，而不是继续在 objective 上打转。**

理由：

- 目标中的核心量是 growth，而不是 absolute offset
- 本轮 `growth_0_to_5` 从 `0.0155` 升到 `0.0273`
- 本轮 `growth_5_to_20` 从 `0.0324` 升到 `0.0551`
- `lane C` 也没有给出 freerun 改善

因此下一步如果还沿 donor 主线推进，应该直接去 donor internal gain mechanism：

- 优先检查 `h_final = coupling_norm((h_temporal + attn_out) * (1 + g) + b)` 中
  - `attn_out`
  - `g`
  - 以及两者在 closed-loop 下对 `Δ20` span 增益的放大路径
- 不建议继续只靠 objective 层面对齐 `Δh`

## 13. Plan / Meas Drift Audit

### 13.1 本轮动机

前两轮 follow-up 已经把两条主线钉死：

- `h_final` 在 freerun vs teacher 间虽然会 drift，但 donor dynamics follow-up 里的 `lane F` 与当前 control 完全同指标重合，说明当前 direct contract **并不消费**这条 hidden 变化。
- `cond` 在当前 runtime contract 下 freerun vs teacher drift = `0.0000`，因此也不是当前剩余 freerun gap 的漂移变量。

所以这轮只补一个最小审计：

- direct branch 仍然实际吃到的 `contacts_plan` / `contacts_meas`
- freerun vs teacher 的 drift 幅度
- tailk7 vs baseline 的幅度对比
- drift 时间 pattern 是否和 `d10+` / `sic11_21` / `sic22_43` 的 freerun gap 对齐

### 13.2 Runtime 路径与实际 consumed tensor

#### `contacts_plan` 的 runtime 路径

- `train/models.py:1491-1546`
  - 定义 cond-only `contact_plan_cell` + `contact_plan_head`
  - 当前实现里 `contacts_plan` 的主生成器是 `cond` 驱动的 GRUCell
- `train/models.py:3224-3276` / `train/models.py:3357-3384`
  - `plan_in_t = cond_seq[:, _t]`
  - `plan_z_t = self.contact_plan_cell(plan_in_t, plan_z_t)`
  - `logits = self.contact_plan_head(...) [+ time term]`
  - `contacts_plan = torch.sigmoid(logits)`

直接结论：

- 在当前 code path 下，`contacts_plan` 是 **cond/time 驱动** 的 plan signal
- 它不读 `h_final`
- 它也不直接读 pose

#### `contacts_meas` 的 runtime 路径

- `train/models.py:1965`
  - 代码注释已经明确：`Contacts are provided externally via contacts_input; internal meas/hazard heads are retired.`
- `train/models.py:3119-3142` / `train/models.py:3291-3313`
  - 仅当 `contacts_input` 存在时，把它 canonicalize 成 `contacts_meas`
  - 否则直接填 `zeros`
- `train/models.py:3570-3584`
  - `result['contacts_meas']` 返回的就是上面那条 `contacts_input-or-zero` 路径
- `train/validate/run_freerun_cycles.py:4479-4519`
  - `contacts_meas_source` 决定 `contacts_in_t`
  - 若 `contacts_meas_source=model`，但模型没有内部 meas head，则 runtime 只能落到 “no external meas provided”

直接结论：

- 当前 repo 的 runtime 下，`contacts_meas` 不是从 `h_final` 推出来的 learned meas
- 它是 rollout 侧 `contacts_input` 的外部注入；没有注入就回落成全零

#### direct head 实际消费的 tensor

- `train/validate/run_freerun_cycles.py:4661`
  - 只有 `direct_pose_meas_source != model` 时才会给 `model.direct_pose_meas_override`
- `train/validate/run_freerun_cycles.py:4690`
  - 只有 `direct_pose_plan_source != model` 时才会给 `model.direct_pose_plan_override`
- 两个固定参考 eval JSON 实际记录：
  - `direct_pose_meas_source=model`
  - `direct_pose_plan_source=model`
  - `DirectMeasOverridePerC` 非空步数 = `0`
  - `DirectPlanOverridePerC` 非空步数 = `0`
- `train/models.py:3636-3673`
  - eval 态下，`plan_in = contacts_plan.detach()`；detach 不改数值
- `train/models.py:3676-3731`
  - eval 态下，`meas_in = clamp(contacts_meas, 0, 1)`；无 drop/noise
- `train/models.py:3735-3750`
  - `direct_pose_feat_source='cond'` 时，`direct_feat = cond`
- `train/models.py:3785-3801`
  - `direct_pose_meas_mode='concat'` 时，实际 direct 输入就是 `[cond, plan_in, meas_in]`

因此本轮主表严格以：

- `ContactPlanPerC`
- `ContactMeasPerC`

作为 direct head 真正 consumed version 的审计对象。

#### 当前两个 case 的 runtime 事实

| case | direct feat source | direct meas mode | internal meas head present | legacy phase keys in ckpt | freerun `ContactsMeasSourceApplied` | direct meas override steps | direct plan override steps |
|---|---|---|---:|---:|---|---:|---:|
| `tailk7_current_control` | `cond` | `concat` | 0 | 0 | `model_missing` | 0 | 0 |
| `baseline_replace` | `cond` | `concat` | 0 | 13 | `whitebox_fallback` | 0 | 0 |

这里有一个必须单列的 caveat：

- baseline ckpt 里仍带有 `13` 个 legacy `contact_plan_phase_head` / `contact_phase_state_*` 参数
- 当前代码加载 baseline 时把这些 key 作为 `unexpected` 丢掉
- 所以 baseline 的 `contacts_plan` teacher rerun **不是** 旧 runtime 的精确回放
- 这意味着 baseline `contacts_plan` 的 teacher-vs-freerun 差值只能作为低置信 side note，不能拿来压过 tail 当前 control 的直接事实

### 13.3 审计方法

- 新脚本：
  `tools/analyze_cp015_tailk7_plan_meas_drift.py`
- freerun 侧：
  直接读取固定参考 eval JSON，不重跑 freerun
- teacher 侧：
  用同一个 ckpt + 同一 runtime override，补一条 teacher-conditioned `_run_freerun_cycles`
  - `rounds=5`
  - `freerun_x_gt=True`
  - `pose_hist_source='seq'`
  - `pose_hist_update_source='gt'`
- 主 drift 指标继续复用前面 `h_final` / `cond` audit 口径：
  - `norm_l2 = ||free - teacher||_2 / sqrt(D)`
  - offsets: `0 / 5 / 20`
  - growth: `0->5 / 5->20`
- freerun error 对照口径：
  - 用现有 eval JSON 的 `per_step_direct_geolocal_deg['DirectGeoLocalDeg']`
  - 分组方式与 `tools/phasea_group_summary.py` 完全一致
  - 对照组：`arm / all_ex_root / leg`

### 13.4 实际命令与产物

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_plan_meas_drift.py
PYTHONPATH=. python3 tools/analyze_cp015_tailk7_plan_meas_drift.py
```

- 新脚本路径：
  `tools/analyze_cp015_tailk7_plan_meas_drift.py`
- summary 输出：
  `debug_output/_tmp_cp015_tailk7_plan_meas_drift_audit_20260404/summary.json`

### 13.5 关键对比表

#### `contacts_plan`：selected-window offset / growth

主表口径：`depth>=10`, `sic11_43`, `drop_wrap=false`

| metric | tailk7 | baseline | tail-base |
|---|---:|---:|---:|
| offset `0` | 0.0000 | 0.0407 | -0.0407 |
| offset `5` | 0.0000 | 0.0402 | -0.0402 |
| offset `20` | 0.0000 | 0.0495 | -0.0495 |
| growth `0->5` | 0.0000 | -0.0005 | +0.0005 |
| growth `5->20` | 0.0000 | +0.0093 | -0.0093 |

解释：

- tail current control 的 `contacts_plan` freerun-vs-teacher drift 是 **严格 0**
- baseline 这里有小幅非零，但它受上面 legacy phase-state loader mismatch 影响，只能低置信看待

#### `contacts_meas`：selected-window offset / growth

| metric | tailk7 | baseline | tail-base |
|---|---:|---:|---:|
| offset `0` | 0.0000 | 0.0000 | +0.0000 |
| offset `5` | 0.0000 | 0.0000 | +0.0000 |
| offset `20` | 0.0000 | 0.0000 | +0.0000 |
| growth `0->5` | 0.0000 | 0.0000 | +0.0000 |
| growth `5->20` | 0.0000 | 0.0000 | +0.0000 |

解释：

- tail `contacts_meas` 在当前 contract 下就是 `model_missing -> zeros`
- baseline archived freerun 虽然保留了 `whitebox_fallback` 痕迹，但坏窗口主区间 `d10+ / sic11_21 / sic22_43` 上仍然是零 drift

#### depth bucket 对照：drift vs freerun error

单位：

- drift = `norm_l2`
- error = `used_local_geo_deg mean`

| bucket | plan drift tail | plan drift base | meas drift tail | meas drift base | error gap all_ex_root | error gap arm | error gap leg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d0_9` | 0.0000 | 0.1118 | 0.0000 | 0.0636 | -0.0513 | -0.0273 | -0.1976 |
| `d10_20` | 0.0000 | 0.0803 | 0.0000 | 0.0000 | +0.0823 | +0.0945 | +0.0341 |
| `d21_43` | 0.0000 | 0.0341 | 0.0000 | 0.0000 | +0.0829 | +0.0821 | +0.1138 |
| `d44_86` | 0.0000 | 0.0590 | 0.0000 | 0.0000 | +0.0087 | +0.0337 | -0.0686 |
| `d87_433` | 0.0000 | 0.0568 | 0.0000 | 0.0000 | +0.0276 | +0.0452 | -0.0334 |

直接读表：

- tail 的 `plan/meas` drift 在 `d10+` 没有任何抬升，仍然是 `0.0000`
- 真正的 freerun error gap 反而是从 `d10_20` / `d21_43` 才开始显著转正
- 所以 `d10+` 的坏窗口 **不是** 由 tail 当前 contract 下的 `plan/meas drift` 驱动

#### `step_in_cycle` 对照：必须拆 `sic11_21` vs `sic22_43`

| bucket | plan drift tail | plan drift base | meas drift tail | meas drift base | error gap all_ex_root | error gap arm | error gap leg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sic0_10` | 0.0000 | 0.0614 | 0.0000 | 0.0000 | -0.0521 | -0.0214 | -0.2275 |
| `sic11_21` | 0.0000 | 0.0447 | 0.0000 | 0.0000 | +0.0879 | +0.0961 | +0.0517 |
| `sic22_43` | 0.0000 | 0.0388 | 0.0000 | 0.0000 | +0.0802 | +0.0808 | +0.1062 |
| `sic44_86` | 0.0000 | 0.0696 | 0.0000 | 0.0000 | +0.0018 | +0.0294 | -0.0888 |

直接读表：

- tail 在 `sic11_21` / `sic22_43` 的 `plan/meas` drift 都是 `0.0000`
- 但 freerun error gap 恰恰就在这两个 split bucket 最显著
- 因而 `sic11_43` 总表如果只看粗平均会误导；拆开以后更清楚：
  - gap 在 `sic11_21` / `sic22_43` 都存在
  - tail 的 `plan/meas drift` 在这两个 bucket 仍然都不存在

#### split table：`contacts_plan` 在 `sic11_21` vs `sic22_43`

| bucket | case | offset `0` | offset `5` | offset `20` | growth `0->5` | growth `5->20` |
|---|---|---:|---:|---:|---:|---:|
| `sic11_21` | tailk7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `sic22_43` | tailk7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `sic11_21` | baseline | 0.0447 | 0.0414 | 0.0381 | -0.0033 | -0.0032 |
| `sic22_43` | baseline | 0.0388 | 0.0396 | 0.0552 | +0.0009 | +0.0156 |

补充：

- `contacts_meas` 在 `sic11_21` / `sic22_43` 的 split offset/growth 对 tail 和 baseline 都是 `0.0000`
- 因此这里没有再单独展开一张重复表

### 13.6 明确判定

先回答这轮的四个问题：

1. `contacts_plan` 是否在 freerun vs teacher 间显著 drift？
   - tailk7 current control：**否，严格 0**
2. `contacts_meas` 是否在 freerun vs teacher 间显著 drift？
   - tailk7 current control：**否，严格 0**
3. 这些 drift 在 tailk7 是否显著大于 baseline？
   - **否**
   - 若按机器输出直读，反而是 baseline side note 更大
   - 但 baseline `contacts_plan` 的非零值受 legacy phase-state loader mismatch 影响，不能上升到主结论
4. drift 的时间 pattern 是否和 `d10+` / `sic11_21` / `sic22_43` 的 freerun gap 对齐？
   - **否**
   - tail 在这些坏窗口上的 `plan/meas drift` 依然都是 `0.0000`

所以这轮最终判定是：

- **C. `plan/meas` drift 也很弱或接近零，说明当前 direct-hint drift 假设仍不成立，需要转向其它 runtime-generated variables。**

更精确地说：

- `contacts_plan`
  - 在 target case `tailk7 current control` 下根本没有 freerun-vs-teacher drift
  - baseline 那条小非零也不和 tail 的残余 gap 提供同向证据
- `contacts_meas`
  - 在当前 contract 下不是一个真正随闭环发散的 runtime variable
  - tail 是 `model_missing -> zero`
  - baseline 只保留了一个 archived `whitebox_fallback` 痕迹，而且不落在 `d10+ / sic11_21 / sic22_43` 主坏窗口

因此这轮不支持：

- A. “tail 的 `plan/meas` drift 更大且时间 pattern 对齐，所以它们更接近因果链”
- 也不太像 B. “drift 有，但只是 tail/base 幅度不够”

当前更直接的结论是：

- 在 **当前 direct contract** 里，`contacts_plan` / `contacts_meas` 不是解释 tail 剩余 freerun gap 的漂移变量
- 如果还沿 runtime-generated variable 主线继续查，下一步应转向别的真正闭环生成量，而不是继续围绕这两个 direct hints 打转

## 14. Main Trunk Drift Audit

### 14.1 本轮动机

前面已经成立的事实是：

- `cond / contacts_plan / contacts_meas / time_pe / plan_z` 在 teacher vs freerun 间不构成 tail 当前残余 gap 的主因
- `DirectGeoLocalDeg` 不是 rollout used 指标
- 当前 `tailk7 current control` 虽然 eval JSON 里 `lambda_fusion_apply=true`，但 ckpt/runtime 都没有实际生效的 lambda/fusion 权重

所以这轮只追主干闭环链：

`out -> _compose_delta_to_raw -> y_inc_raw -> y_used_raw -> _apply_free_carry -> motion / pose_history`

目标是回答：

- `out` 是否显著 drift
- `y_inc_raw` 是否显著 drift
- 下一步真正消费的主干 state 是否显著 drift
- `pose_history` 是否显著 drift
- 上述 drift 在 tailk7 是否显著大于 baseline
- 这些 drift 的时间 pattern 是否和当前真实 freerun gap 对齐

### 14.2 先确认代码事实

#### `out` 的 runtime 路径

- `model(...)` 返回 dict，主干输出直接取 `ret["out"]`
- 代码位置：
  - `train/eval_utils.py:394-407`
  - freerun 主循环镜像逻辑在 `train/validate/run_freerun_cycles.py:5912-5918`

结论：

- `out` 就是主 trunk 在当前步产出的 normalized delta / residual output

#### `y_inc_raw` 的 runtime 路径

- `delta_norm = out`
- 若存在上一帧 `y_raw_prev`，则 `y_inc_raw = trainer._compose_delta_to_raw(y_raw_prev, delta_norm, ...)`
- 否则回退到 `trainer._denorm(delta_norm)`
- 代码位置：
  - `train/validate/run_freerun_cycles.py:5912-5927`
  - `_compose_delta_to_raw` 定义在 `train/training_MPL.py:2842-2955`

`_compose_delta_to_raw` 的实际语义：

- 对 `rot6d` slice 用 `compose_rot6d_delta(...)` 做增量合成
- 非旋转 tail 通道做 `tail_prev + tail_delta`

所以 `y_inc_raw` 是 incremental/main-trunk 的 **absolute next pose in raw space**

#### `y_inc_raw / y_blend_raw / y_used_raw` 的关系

- `y_blend_raw` 先初始化为 `y_inc_raw`
- 只有在 `lambda_fusion_apply=true` 且 `direct_norm_step`、`lam_for_blend` 都是 tensor 时，才调用 `_apply_lambda_fusion_to_raw(...)`
- 最终 rollout state update 取：
  - `y_used_raw = y_blend_raw if lambda_fusion_apply else y_inc_raw`
- 代码位置：
  - `train/training_MPL.py:3097-3180`
  - `train/validate/run_freerun_cycles.py:6219-6248`

本轮实际运行结果：

- tailk7 / baseline 两个 case 都满足：
  - `LambdaMean` non-null steps = `0`
  - `LambdaEffMean` non-null steps = `0`
  - `BlendGeoLocalDeg == GeoLocalDeg`，434 个有限 step 上 `max_abs_diff = 0`
  - freerun 内部检查里 `y_used_raw vs y_inc_raw` 是严格零差

所以当前这两条 run 上：

- `y_blend_raw == y_inc_raw`
- `y_used_raw == y_inc_raw`

#### `_apply_free_carry` 实际把什么写回 `motion / motion_raw`

定义在 `train/training_MPL.py:3345-3415`。

它做了四件事：

1. `x_next[..., rot6d_x_slice] = y_denorm[..., rot6d_y_slice]`
2. 用上一帧/当前帧 rotation 推导 `angvel_x_slice`
3. 用 `cond_next_raw` 写 `rootvel_x_slice`
4. 用 `cond_next_raw` 累积 `rootpos_x_slice`

因此：

- 进入下一步的主干 carry 里，真正由 `y_used_raw` 驱动的核心部分是
  - pose rotation
  - 由 pose 差分导出的 angvel
- root velocity / root position 是 cond-driven deterministic carry

本轮主表里把 **`motion_in`** 作为 “rollout carried state 里真正被下一步消费的主干变量”：

- 它是下一次 `model(...)` 真正收到的 `motion`
- 比 `motion_raw_after_carry` 更贴近“实际 consumed tensor”
- `motion_raw_after_carry` 仍然零额外成本导出到 summary appendix

#### `pose_history` 在 freerun 中实际写入的是哪一路 pose

- `resolve_pose_hist_input(...)` 优先读 `state.buffer_norm`，否则才回退到 `pose_hist_seq`
  - `train/history.py:133-146`
- freerun buffer 模式下，`pose_hist_update_source == "pred"` 时：
  - `rot_write = y_used_raw[..., rot_slice]`
  - 再调用 `advance_pose_hist_state_with_tail(..., rot_tail_raw=rot_write)`
  - `advance_pose_hist_state_with_tail` 把 `rot_tail_raw` 写到 buffer 末尾
  - 代码位置：
    - `train/validate/run_freerun_cycles.py:6296-6332`
    - `train/history.py:170-190`

本轮实际 freerun 检查：

- `pose_hist_write_raw vs y_used_raw[..., rot_slice]` 也是 434 step 严格零差

所以当前 current control / baseline 这两条 run 上：

- `pose_history` write path 实际写入的就是 `y_used_raw[..., rot_slice]`
- 由于本轮 `y_used_raw == y_inc_raw`，它本质上也是主干 incremental pose

#### `GeoLocalDeg / BlendGeoLocalDeg / DirectGeoLocalDeg` 的语义区分

freerun 主循环里：

- `predsY.append(y_inc_norm)` 对应 incremental/main trunk
- `predsY_blend.append(y_blend_norm)` 对应 blend path
- `predsY_direct.append(direct_norm_step)` 对应 direct head path
- 代码位置：`train/validate/run_freerun_cycles.py:6233-6253`

之后：

- `predY_full = stack(predsY)`
- `predY_blend_full = stack(predsY_blend)`
- `predY_direct_full = stack(predsY_direct)`
- 再分别 denorm 成：
  - `pred_raw_full`
  - `pred_blend_raw_full`
  - `pred_direct_raw_full`
- 代码位置：`train/validate/run_freerun_cycles.py:6341-6413`

最后 per-step metric entry 里：

- `GeoLocalDeg` 绑定 incremental path
- `BlendGeoLocalDeg` 绑定 blend path
- `DirectGeoLocalDeg` 绑定 direct path
- 代码位置：`train/validate/run_freerun_cycles.py:8814-8841`

本轮没有显式 `used_local_geo_deg` 字段，所以主对照指标取 **现有 `GeoLocalDeg`**，原因是：

- 它来自 `predsY <- y_inc_norm`
- 而本轮实际 runtime 又满足 `y_used_raw == y_inc_raw`
- 因此当前 `GeoLocalDeg` 就是 main-trunk / rollout-used 的正确对照指标

### 14.3 审计方法

新脚本：

- `tools/analyze_cp015_tailk7_main_trunk_drift.py`

方法：

- 复用 `_load_case` + `_run_freerun_cycles`
- 不改 eval contract，不改 runtime 语义
- 只在脚本外层 monkeypatch 记录：
  - `model.forward`：抓 `motion_in` / `pose_history_in` / `out`
  - `trainer._compose_delta_to_raw`：抓 `y_inc_raw`
  - `trainer._apply_free_carry`：抓 `y_used_raw` / `motion_raw_after_carry`
  - `advance_pose_hist_state_with_tail`：抓 `pose_hist_write_raw`
- teacher 对照跑法：
  - 同样 5-cycle `_run_freerun_cycles`
  - `pose_hist_source='seq'`
  - `pose_hist_update_source='gt'`
  - `freerun_x_gt=True`
- freerun 跑法：
  - 完全沿用各自 eval JSON runtime override
  - 也就是 `buffer/pred`, `lambda_fusion_apply=true`, `time_index_mode=cycle`

drift metric：

- `norm_l2 = ||free - teacher||_2 / sqrt(D)`
- 辅助导出 `mean_abs`, `cosine_distance`

主窗口口径：

- `depth >= 10`
- `step_in_cycle in [11, 43]`
- drop wrap
- 必须能看 `+20` horizon

pattern table 口径：

- depth: `d0_9 / d10_20 / d21_43`
- step_in_cycle: `sic0_10 / sic11_21 / sic22_43`

注意：

- 不同 signal 的维度和归一化不同，**不同 signal 之间的绝对数值不可横向比较**
- 这轮只比较：
  - 同一 signal 内 tail vs baseline
  - 同一 signal 内不同 bucket 的抬升 pattern

### 14.4 实际跑的命令

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_main_trunk_drift.py
python3 tools/analyze_cp015_tailk7_main_trunk_drift.py --device cpu
```

summary 输出：

- `debug_output/_tmp_cp015_tailk7_main_trunk_drift_audit_20260404/summary.json`

### 14.5 关键结果

先给当前真实坏窗口口径：

| metric | tailk7 current control | baseline replace | tail-base |
|---|---:|---:|---:|
| selected-window `GeoLocalDeg mean` | 60.3370 | 43.9433 | +16.3937 |

补充：

- 这里的 selected window 就是 `d10+` 且 `sic11_43`、drop wrap、要求 `+20` horizon
- 这也是后面 offset/growth 主表的口径

#### selected-window 主表：`out / y_inc_raw / motion_in / pose_history_in`

| signal | tail off0 | tail off5 | tail off20 | tail g0->5 | tail g5->20 | base off0 | base off5 | base off20 | base g0->5 | base g5->20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `out` | 0.236313 | 0.243812 | 0.266180 | 0.007498 | 0.022368 | 0.032598 | 0.033208 | 0.035534 | 0.000610 | 0.002326 |
| `y_inc_raw` | 0.398266 | 0.407802 | 0.438716 | 0.009536 | 0.030913 | 0.183048 | 0.188674 | 0.207610 | 0.005626 | 0.018936 |
| `motion_in` | 3.921657 | 4.112583 | 4.378345 | 0.190926 | 0.265762 | 3.010070 | 3.189506 | 3.356302 | 0.179436 | 0.166796 |
| `pose_history_in` | 33877.680687 | 34784.045449 | 37550.634267 | 906.364761 | 2766.588818 | 12343.089574 | 12659.321890 | 13596.099566 | 316.232316 | 936.777677 |

直接读表：

- `out`：tail 明显大于 baseline，且 `5->20` growth 差最明显
- `y_inc_raw`：tail 也明显大于 baseline，`offset20` 与 `5->20` growth 都拉开
- `motion_in`：tail 依然更大，但 pattern 比 `out / y_inc_raw / pose_history_in` 更钝
- `pose_history_in`：tail 显著大于 baseline，且后段 growth 明显抬升

#### tail vs baseline：selected-window 幅度差

| signal | Δoff0 | Δoff5 | Δoff20 | Δg0->5 | Δg5->20 |
|---|---:|---:|---:|---:|---:|
| `out` | +0.203715 | +0.210603 | +0.230645 | +0.006888 | +0.020042 |
| `y_inc_raw` | +0.215219 | +0.219129 | +0.231106 | +0.003910 | +0.011977 |
| `motion_in` | +0.911586 | +0.923077 | +1.022043 | +0.011490 | +0.098966 |
| `pose_history_in` | +21534.591114 | +22124.723559 | +23954.534701 | +590.132445 | +1829.811141 |

#### depth bucket 对照：drift pattern vs `GeoLocalDeg`

| bucket | `GeoLocalDeg` tail/base | `out` tail/base | `y_inc_raw` tail/base | `motion_in` tail/base | `pose_history_in` tail/base |
|---|---:|---:|---:|---:|---:|
| `d0_9` | 2.3881 / 2.6792 | 0.026579 / 0.021079 | 0.007071 / 0.004991 | 2.871526 / 2.872162 | 432.066 / 208.880 |
| `d10_20` | 7.6484 / 8.4032 | 0.029549 / 0.014555 | 0.021095 / 0.013635 | 2.264500 / 2.261921 | 1650.835 / 792.571 |
| `d21_43` | 10.9665 / 12.1860 | 0.045962 / 0.012114 | 0.023979 / 0.017352 | 2.453928 / 2.450719 | 3733.257 / 1783.284 |

直接读表：

- tail 内部的 `GeoLocalDeg` 从 `d0_9 -> d10_20 -> d21_43` 明显升高
- `out` 也从 `0.0266 -> 0.0295 -> 0.0460` 抬升
- `y_inc_raw` 从 `0.0071 -> 0.0211 -> 0.0240`，在 `d10+` 才真正变大
- `pose_history_in` 从 `432 -> 1651 -> 3733`，与 `d10+` 的坏窗口非常一致
- `motion_in` depth 抬升没有前面三条那么干净，说明 full carried state 被 cond/root 通道稀释后，depth selectivity 变弱

#### `step_in_cycle` bucket 对照：必须拆 `sic11_21` vs `sic22_43`

| bucket | `GeoLocalDeg` tail/base | `out` tail/base | `y_inc_raw` tail/base | `motion_in` tail/base | `pose_history_in` tail/base |
|---|---:|---:|---:|---:|---:|
| `sic0_10` | 53.3908 / 38.3757 | 0.232359 / 0.041403 | 0.355698 / 0.157816 | 4.324509 / 3.653516 | 30014.156 / 11004.313 |
| `sic11_21` | 57.6182 / 41.5557 | 0.227418 / 0.033143 | 0.378908 / 0.171738 | 3.645511 / 2.779015 | 31903.335 / 11641.418 |
| `sic22_43` | 61.6965 / 45.1371 | 0.240761 / 0.032325 | 0.407946 / 0.188703 | 4.059730 / 3.125598 | 34864.853 / 12693.925 |

直接读表：

- 当前真实坏窗口 `GeoLocalDeg` 在 tail 上确实是：
  - `sic11_21 > sic0_10`
  - `sic22_43 > sic11_21`
- `y_inc_raw` 和 `pose_history_in` 也呈现同方向抬升
- `out` 在 `sic11_21` 没有比 `sic0_10` 更高，但在 `sic22_43` 明显更高；说明它更像“持续高位 + 末段再抬升”
- `motion_in` 依旧比 baseline 大，但 split-bucket pattern 没有 `y_inc_raw / pose_history_in` 那么干净

#### 本轮额外 runtime check

| check | tailk7 | baseline |
|---|---:|---:|
| `LambdaMean` non-null steps | 0 | 0 |
| `LambdaEffMean` non-null steps | 0 | 0 |
| `BlendGeoLocalDeg == GeoLocalDeg` | yes | yes |
| `y_used_raw == y_inc_raw` | yes | yes |
| `pose_hist_write_raw == y_used_raw[..., rot_slice]` | yes | yes |

这意味着：

- 当前 run 上不存在 “fusion 污染了 main-trunk metric” 这个解释
- `pose_history` 的 drift 也不是来自额外支路，它就是主干 `y_used_raw` 写进去以后逐步累积出来的

### 14.6 最终判定

先按问题逐条回答：

1. `out` 在 teacher vs freerun 间是否显著 drift？
   - **是**
   - 而且 tailk7 明显大于 baseline
2. `y_inc_raw` 在 teacher vs freerun 间是否显著 drift？
   - **是**
   - 且在 `d10+`、尤其 `sic11_21 / sic22_43` 上升高明显
3. rollout carried state 里真正进入下一步消费的主干变量是否显著 drift？
   - **是**
   - 以实际 consumed 的 `motion_in` 看，tail 仍然显著大于 baseline
   - 但这个 full-state 信号的时间 pattern 比 `out / y_inc_raw / pose_history_in` 更钝
4. `pose_history` 是否在 freerun 中显著 drift？
   - **是，且非常强**
   - 并且 write path 已确认就是 `y_used_raw[..., rot_slice]`
5. 这些 drift 在 tailk7 是否显著大于 baseline？
   - **是**
   - selected-window 主表四个信号全部 tail > baseline
6. drift 的时间 pattern 是否和真实 freerun gap 对齐？
   - **整体是**
   - 最干净的是 `y_inc_raw` 与 `pose_history_in`
   - `out` 也对齐，但更像高位持续并在 `sic22_43` 进一步抬升
   - `motion_in` 由于混入 full carry channels，pattern 相对不如前面三条干净

所以本轮最终判定是：

- **A. main trunk drift 在 tailk7 明显大于 baseline，且时间 pattern 与 `d10+` / `sic11_21` / `sic22_43` 的 `GeoLocalDeg` 坏窗口整体对齐，当前更接近因果链的是主干 rollout path，而不是 direct hint path。**

更精确地说：

- 如果只看最直接的主干变量：
  - `out`
  - `y_inc_raw`
  - `pose_history_in`
  这三条已经足够给出 A
- `motion_in` 不是反例，只是它作为 full carried state，会被 cond/root deterministic carry 稀释，所以 pattern 没前面三条干净
- 在 fusion 明确失效、`DirectGeoLocalDeg` 又已排除为 rollout-used proxy 的前提下，这轮证据最自然地把因果优先级推回：
  - `main trunk rollout path`
  - 而不是 `direct hint path`

## 15. Matched-Input Trunk Gain Audit

### 15.1 本轮动机

上一轮 main-trunk drift audit 已经证明：

- tailk7 freerun 上的 `out / y_inc_raw / pose_history_in` drift 明显大于 baseline
- 但那一轮还没有拆清：
  - 是 tailk7 的 incoming drift 本身更大
  - 还是在相同输入扰动下，tailk7 的 trunk / closed-loop gain 更高

所以这轮只做一个最小的 matched-input one-step counterfactual：

- 仍然用 teacher-conditioned base step
- 扰动方向不再用随机噪声，而是直接取 tailk7 freerun vs teacher 的真实 observed drift direction
- 用同一个 `Δmotion_in` / `Δpose_history_in` 同时喂 tailk7 current control 和 baseline replace
- 只比较一步响应：
  - `h_final`
  - `out`
  - `y_inc_raw`
  - 一步后的 main-trunk local geo proxy

### 15.2 Trunk 实际输入路径 / hook 点 / 注入点定义

先把这轮用到的代码事实固定下来：

1. validate runtime 的下一步 model call 是：

   - `model(motion, cond_input, contacts, angvel, pose_history, plan_z, phase_event_age, meas_logits_prev, time_index, rollout_step)`
   - 对应实现：`train/validate/run_freerun_cycles.py`

2. 共享 trunk 的直接拼接输入不是全部 runtime args，而是：

   - `x = concat([state, cond, plan_feat_for_inject])`
   - 当前两个 ckpt 都是 `contact_plan_inject='plan_z'`
   - 对应实现：`train/models.py`

3. `pose_history` 不会直接拼进 shared trunk `x`，它通过两条间接路径影响 `h_final`：

   - frozen contact / frozen period side path
   - contact-plan / event-clock path

4. `h_final` 这轮不再需要拿 `coupling_norm` 当 proxy：

   - 当前前向里就是 `h_final = self.coupling_norm(...)`
   - 随后直接 `result['h_final'] = h_final`
   - 所以 `ret['h_final']` 就是 exact `h_final`
   - 如果沿用 `coupling_norm` hook，它捕到的也是 exact `h_final`，不是近似 witness

5. `out` 的生成路径：

   - `out = self.motion_head(h_final)`

6. `y_inc_raw` 的生成路径：

   - `y_inc_raw = trainer._compose_delta_to_raw(y_raw_prev, out, ...)`
   - 当前这两条 run 已知：
     - `LambdaMean / LambdaEffMean == None`
     - `BlendGeoLocalDeg == GeoLocalDeg`
     - `y_used_raw == y_inc_raw`
   - 所以这轮一步 main-trunk error proxy 可以直接绑定到 `y_inc_raw`

7. 一步后的 main-trunk local geo proxy：

   - 这轮显式重算 `y_inc_raw` 对 tiled `gtY` 的 root-relative local geodesic error
   - 口径对齐 eval 的 `GeoLocalDeg`
   - 因为当前 runtime 已确认 `y_used_raw == y_inc_raw`，所以它就是一步后的 rollout-used main-trunk local geo proxy

8. 扰动注入点定义：

   - `motion only`：只在 model input `motion` 注入 `Δmotion_in`
   - `pose_history only`：只在 model input `pose_history` 注入 `Δpose_history_in`
   - `motion + pose_history`：两个通道同时注入

9. 为了不破坏 runtime contract，这轮不是简单把其它输入全 freeze 成旧值：

   - 固定：
     - `cond_input`
     - `plan_z`
     - `phase_event_age`
     - `meas_logits_prev`
     - `time_index`
     - `rollout_step`
   - 但会对 injected state 重新派生：
     - `angvel_t`：当 `use_freerun_state_sync=True` 时，从 injected `motion` 的 angvel slice 重取
     - `contacts_in_t`：调用 `trainer._predict_pretrain_contacts_from_frozen(injected motion, injected pose_history)` 重新计算

### 15.3 Matched-Input 方法

- delta source：
  - 直接取 tailk7 current control 在同一个 global step 上的
    - `freerun motion_in - teacher motion_in`
    - `freerun pose_history_in - teacher pose_history_in`
- base state：
  - 对 tailk7 / baseline 各自先跑一个 teacher-conditioned multicycle pass
  - 然后在同一个 global step 上取各自的 teacher-conditioned base input/state
- counterfactual：
  - 把同一个 observed `Δ` 注入到两条 case 的 teacher-conditioned base state
  - 只做 one-step，不继续 rollout
- 本轮参数：
  - `rounds = 5`
  - `alpha = 1.0`

### 15.4 实际跑的命令 / 脚本 / 输出

新脚本：

- `tools/analyze_cp015_tailk7_matched_input_trunk_gain.py`

summary 输出：

- `debug_output/_tmp_cp015_tailk7_matched_input_trunk_gain_audit_20260405/summary.json`

实际跑的命令：

```bash
python3 tools/analyze_cp015_tailk7_matched_input_trunk_gain.py --device cpu
```

### 15.5 Overall 输入幅度 / 响应幅度

| channel | input_norm | tail resp_h | base resp_h | tail resp_out | base resp_out | tail resp_y | base resp_y |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pose_history_only` | 36761.1 | 0.0383487 | 0.0526544 | 0.00290308 | 0.00416483 | 0.000128694 | 0.000207828 |
| `motion_only` | 4.21615 | 0.461604 | 0.532143 | 0.256910 | 0.0467424 | 0.00427581 | 0.00237205 |
| `motion_plus_pose_history` | 29955.1 | 0.465944 | 0.538179 | 0.257534 | 0.0477450 | 0.00427602 | 0.00240957 |

先直接读表：

- `pose_history only` 下，tail 的响应幅度反而普遍小于 baseline
- `motion only` 下，tail 的 `out / y_inc_raw` 响应明显大于 baseline，但 `h_final` 不是
- `motion + pose_history` 的响应幅度几乎和 `motion only` 一样：
  - tail `resp_out`: `0.256910 -> 0.257534`
  - tail `resp_y`: `0.00427581 -> 0.00427602`
  - baseline 也同样几乎不变
- 这说明 joint 注入的增量解释力主要还是来自 `motion`，不是 `pose_history`

### 15.6 pose_history only：matched-input gain 表

| metric | input_norm | tail gain_h | base gain_h | tail gain_out | base gain_out | tail gain_y | base gain_y | tail gain_geo | base gain_geo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 36761.1 | 6.81011e-06 | 9.51252e-06 | 6.57679e-07 | 6.81747e-07 | 2.39091e-08 | 3.79457e-08 | 2.89159e-07 | 5.81931e-07 |
| ratio tail/base | - | 0.71591 | - | 0.964695 | - | 0.630088 | - | 0.496896 | - |

这张表只说明一件事：

- 相同 `Δpose_history_in` 下，tail 并没有表现出更高 gain
- 尤其 `y_inc_raw` / local-geo 方向，tail 还是更小
- 所以上一轮 tail `pose_history_in` drift 更大，更像 incoming drift amplitude 问题，不是 pose-history sensitivity 更高

### 15.7 motion only：matched-input gain 表

| metric | input_norm | tail gain_h | base gain_h | tail gain_out | base gain_out | tail gain_y | base gain_y | tail gain_geo | base gain_geo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4.21615 | 0.110774 | 0.126478 | 0.0598161 | 0.0101473 | 0.000995189 | 0.000521322 | 0.0383089 | 0.00831585 |
| ratio tail/base | - | 0.875839 | - | 5.89480 | - | 1.90897 | - | 4.60673 | - |

这张表是本轮最关键的 matched-input 结果：

- `h_final`：tail 没有更大，反而略小（`0.876x`）
- `out`：tail 明显更大（`5.89x`）
- `y_inc_raw`：tail 仍更大（`1.91x`）
- local-geo 响应：tail 也更大（`4.61x`）

所以：

- “higher matched-input gain” 这句话如果绑定到 `h_final`，**不成立**
- 但如果绑定到 `out / y_inc_raw / one-step local geo`，**成立，而且很强**

### 15.8 motion + pose_history：matched-input gain 表

| metric | input_norm | tail gain_h | base gain_h | tail gain_out | base gain_out | tail gain_y | base gain_y | tail gain_geo | base gain_geo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 29955.1 | 4.76862e-05 | 5.52355e-05 | 1.26437e-05 | 3.83714e-06 | 3.21625e-07 | 2.09687e-07 | 7.89243e-06 | 3.80226e-06 |
| ratio tail/base | - | 0.863326 | - | 3.29509 | - | 1.53383 | - | 2.07572 | - |

joint 注入和 `motion only` 的读法一致：

- `h_final` 仍不是 tail 更大
- `out / y_inc_raw / local geo` 仍是 tail 更大
- 但相比 `motion only`，joint 注入并没有带来更强的 tail/base 分离，反而略弱
- 这再次说明主因不是 `pose_history` 补上去以后才出现，而是 `motion` 已经足够触发主要差异

### 15.9 tailk7 vs baseline：overall gain ratio 对比

| channel | gain_h ratio | gain_out ratio | gain_y ratio | gain_geo ratio |
|---|---:|---:|---:|---:|
| `pose_history_only` | 0.7159 | 0.9647 | 0.6301 | 0.4969 |
| `motion_only` | 0.8758 | 5.8948 | 1.9090 | 4.6067 |
| `motion_plus_pose_history` | 0.8633 | 3.2951 | 1.5338 | 2.0757 |

按 channel 总结：

- `pose_history_only`：不支持 tail 高 gain
- `motion_only`：最能解释 tail 的 higher matched-input gain，尤其 `out`
- `motion + pose_history`：也支持，但解释力比 `motion only` 更弱

### 15.10 depth bucket 对照表

| bucket | GeoLocalDeg tail/base | pose gain_y ratio | motion gain_y ratio | joint gain_y ratio | pose gain_out ratio | motion gain_out ratio | joint gain_out ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d0_9` | 2.3881 / 2.6792 | 0.6568 | 1.4620 | 1.3281 | 1.3026 | 1.3045 | 1.2309 |
| `d10_20` | 7.6484 / 8.4032 | 0.5918 | 1.3345 | 1.1684 | 0.6345 | 2.1883 | 1.8885 |
| `d21_43` | 10.9665 / 12.1860 | 0.5514 | 1.7889 | 1.2582 | 0.5406 | 4.4727 | 3.4357 |

depth 读法：

- `pose_history only` 在 `d10+` 反而持续变弱
- `motion only` 的 `gain_out` ratio 从 `1.30x -> 2.19x -> 4.47x`
- `motion only` 的 `gain_y` ratio 从 `1.46x -> 1.33x -> 1.79x`
- `motion + pose_history` 也随 depth 增强，但不如 `motion only` 明显

所以 matched-input pattern 仍然对齐 `d10_20 / d21_43`，而且最对齐的是 `motion`

### 15.11 step_in_cycle bucket 对照表

| bucket | GeoLocalDeg tail/base | pose gain_y ratio | motion gain_y ratio | joint gain_y ratio | pose gain_out ratio | motion gain_out ratio | joint gain_out ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sic0_10` | 53.3908 / 38.3757 | 0.6599 | 2.0386 | 1.3993 | 1.2510 | 5.5769 | 1.6188 |
| `sic11_21` | 57.6182 / 41.5557 | 0.5687 | 1.8153 | 1.3406 | 0.6382 | 5.5276 | 3.5596 |
| `sic22_43` | 61.6965 / 45.1371 | 0.5560 | 1.8784 | 1.6246 | 0.5466 | 5.6620 | 4.7990 |

cycle-phase 读法：

- `pose_history only` 在 `sic11_21 / sic22_43` 都明显小于 `1.0`
- `motion only` 的 `gain_out` ratio 在三个 bucket 都稳定约 `5.5x+`
- `motion only` 的 `gain_y` ratio 在 `sic11_21 / sic22_43` 仍保持 `1.8x+`
- `motion + pose_history` 在 `sic22_43` 有增强，但仍然没有超过 `motion only`

所以 matched-input gain pattern 仍然对齐：

- `sic11_21`
- `sic22_43`

而且最稳定的解释通道仍然是 `motion`

### 15.12 response pattern vs GeoLocalDeg 坏窗口并表

| bad window | freerun GeoLocalDeg tail/base | pose y ratio | motion y ratio | joint y ratio | motion out ratio | joint out ratio |
|---|---:|---:|---:|---:|---:|---:|
| `d10_20` | 7.6484 / 8.4032 | 0.5918 | 1.3345 | 1.1684 | 2.1883 | 1.8885 |
| `d21_43` | 10.9665 / 12.1860 | 0.5514 | 1.7889 | 1.2582 | 4.4727 | 3.4357 |
| `sic11_21` | 57.6182 / 41.5557 | 0.5687 | 1.8153 | 1.3406 | 5.5276 | 3.5596 |
| `sic22_43` | 61.6965 / 45.1371 | 0.5560 | 1.8784 | 1.6246 | 5.6620 | 4.7990 |

这张并表把本轮的因果更新压得最清楚：

- 真正的坏窗口越差，`motion only` 的 matched-input `out / y_inc_raw` ratio 越稳定、越强
- `pose_history only` 不但不增强，反而在坏窗口里始终 `< 1`
- `motion + pose_history` 不是零，但其解释力主要就是跟着 `motion` 走

### 15.13 最终判定

先按用户要求逐条回答：

1. 在完全相同的输入扰动下，tailk7 的 `h_final` 响应是否大于 baseline？

   - **否**
   - 三种注入里整体都是 baseline `h_final` 响应更大
   - 所以“tail 的 higher matched-input gain”不能写成 `h_final` 结论

2. 在完全相同的输入扰动下，tailk7 的 `out` 响应是否大于 baseline？

   - **是，但只在 `motion only` / `motion + pose_history` 上成立**
   - 其中 `motion only` 最强：
     - overall `gain_out ratio = 5.89x`
     - `d21_43 = 4.47x`
     - `sic11_21 = 5.53x`
     - `sic22_43 = 5.66x`

3. 在完全相同的输入扰动下，tailk7 的 `y_inc_raw` 响应是否大于 baseline？

   - **是，但同样主要来自 `motion only` / `motion + pose_history`**
   - `motion only`：
     - overall `gain_y ratio = 1.91x`
     - `d21_43 = 1.79x`
     - `sic11_21 = 1.82x`
     - `sic22_43 = 1.88x`

4. gain 差异主要来自哪条通道？

   - **不是 `pose_history only`**
   - **主要来自 `motion only`**
   - `motion + pose_history` 也成立，但没有比 `motion only` 更强

5. matched-input gain pattern 是否仍对齐：
   - `d10_20 / d21_43`
   - `sic11_21 / sic22_43`

   - **是**
   - 且这种对齐主要出现在 `motion only` 的 `out / y_inc_raw`

因此这轮最终判定写成：

- **A：部分成立，但只对 `out / y_inc_raw` 成立，不对 `h_final` 成立**
- **B：不成立；matched-input 后差异没有收缩到消失，尤其 `motion only` 仍明显 tail > baseline**
- **C：成立，而且是这轮主结论；`pose_history only` 不足以解释，因果链必须从 “pose_history-centric” 改写成更宽的 recurrent carried inputs，其中以 `motion` 为主**
- **D：不成立；三种注入并非都解释力有限，`motion only` 已经足够给出可判定结果**

如果只保留一句最终总结：

- **本轮最小 matched-input audit 的结论是：tailk7 并不是对相同输入表现出更大的 `h_final` 增益，而是对相同的 motion-direction 扰动表现出明显更高的 `out / y_inc_raw` 闭环增益；`pose_history only` 不能解释当前 gap，主表述应改成更宽的 recurrent carried inputs，并且重点落在 `motion` 而不是 `pose_history`。**

## 16. Motion-Head Gain Audit

### 16.1 本轮动机

上一轮 matched-input trunk gain audit 已经把主因收缩到：

- `h_final` 不是 tailk7 的主 amplifier
- 真正异常更像在 `h_final -> out -> y_inc_raw`

所以这轮只做最小 head-side audit，目标是回答：

- 在完全相同的 `Δh_final` 下，tailk7 的 `out` 响应是否仍显著大于 baseline
- 在完全相同的 `Δh_final` 下，tailk7 的 `y_inc_raw` 响应是否仍显著大于 baseline
- 这种差异更像静态权重尺度，还是更像 local Jacobian / activation regime
- amplification 是否仍对齐 `d10_20 / d21_43 / sic11_21 / sic22_43`
- amplification 是否主要集中在 rot slice

本轮不改 donor，不改 downstream，不改结构，不开训练线，只追加一个最小分析脚本。

### 16.2 必须先确认的代码事实

#### 16.2.1 `motion_head` 的实际定义

源码上，`EventMotionModel.__init__` 在 `train/models.py:1481-1488` 定义：

- `self.motion_head = build_mlp(hidden_dim, hidden_dim, num_layers=1, activation=nn.ReLU, dropout=dropout, final_dim=out_motion_dim)`

实际恢复后的两个 ckpt 上，`motion_head` 实例完全一致：

```python
Sequential(
  (0): Linear(in_features=512, out_features=512, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.1, inplace=False)
  (3): Linear(in_features=512, out_features=278, bias=True)
)
```

所以主 readout 本体不是单层线性，而是：

- `Linear(512 -> 512)`
- `ReLU`
- `Dropout(0.1)`，eval 时等价 identity
- `Linear(512 -> 278)`

但这还不是完整 `h_final -> out` 路径。`train/models.py:3548-3554` 还会在 `motion_head(h_final)` 之后叠加 `_bone_adapters`：

- `slice 192:198` -> `thigh_l`
- `slice 198:204` -> `calf_l`
- `slice 222:228` -> `foot_l`
- `slice 234:240` -> `thigh_r`
- `slice 240:246` -> `calf_r`
- `slice 264:270` -> `foot_r`

因此本轮 audit 里的 “motion_head / head-side readout” 实际上指：

- `out = motion_head(h_final) + bone_adapter_deltas`

#### 16.2.2 `h_final -> out` 的精确路径

`train/models.py:3548-3566` 的实际路径是：

- `hidden_out = h_final`
- `out = self.motion_head(h_final)`
- 若 `_bone_adapters` 生效，则对若干 rot sub-slice 再加 adapter residual
- `result['out'] = out`
- `result['h_final'] = hidden_out`

所以最干净的 head-side 注入点就是：

- `coupling_norm(...)` 之后
- `motion_head` 与 `_bone_adapters` 之前

也就是本轮定义的 `Δh_final` 注入点。

#### 16.2.3 `out_motion_dim` 与 output slice

layout helper 不是手写魔法数字，而是复用：

- `train/layout.py:271-282`
- `train/models.py:4717-4720`

实际恢复后的两条 case 都得到同一个 output layout：

- `BoneRotations6D: slice(0, 276)`
- `RootVelocity: slice(276, 278)`

因此：

- `out_motion_dim = 278 = 276 rot + 2 root_vel`
- 当前实现里 **不存在** output-side `angvel`
- 当前实现里 **不存在** output-side `contacts`
- 当前实现里 **不存在** output-side `RootPosition`
- 除 `rot / root_vel` 外也没有稳定可切的其它 output slice

所以本轮 slice-wise 拆分只能稳定做：

- `rot`
- `root_vel`

不能假装有 `angvel / contacts / other`。

#### 16.2.4 `y_inc_raw` 的生成路径

自由运行里，`train/validate/run_freerun_cycles.py:5912-5927` 做：

- `delta_norm = out`
- `y_inc_raw = trainer._compose_delta_to_raw(y_raw_prev, delta_norm, ...)`

而 `Trainer._compose_delta_to_raw` 在 `train/training_MPL.py:2842-2955` 做：

- 先把 `delta_norm` 乘 `std_y` 变成 `delta_raw`
- `rot_slice` 上走 `compose_rot6d_delta(...)`
- 尾部非-rot 通道直接 residual add

在当前两条 eval contract 下，前面已成立：

- `LambdaMean / LambdaEffMean == None`
- `BlendGeoLocalDeg == GeoLocalDeg`
- `y_used_raw == y_inc_raw`
- `pose_hist_write_raw == y_used_raw[..., rot_slice]`

对应源码位置：

- `train/validate/run_freerun_cycles.py:6247-6248`
- `train/validate/run_freerun_cycles.py:6313-6324`

所以本轮 `y_inc_raw` / `pose_history` 绑定路径没有新歧义。

#### 16.2.5 一步后的 `GeoLocalDeg` 绑定路径

本轮 one-step proxy 没有走多步 rollout，而是：

- 先得到 head-side 注入后的 `y_inc_raw`
- 再与同一步 `gt_raw` 做 root-relative local SO(3) geodesic

实现上复用了上一轮 `_geo_local_deg_from_raw(...)` 的同一条 root-relative local geo 路径，所以它就是：

- “same clean teacher-conditioned base step”
- “same one-step composed `y_inc_raw`”
- “same-step GT raw”

#### 16.2.6 local Jacobian 的计算点

本轮没有用 input-side 有限差分，也没有去重跑 full rollout Jacobian，而是直接对部署态 readout

- `out(h_final) = motion_head(h_final) + adapters(h_final)`

做 directional JVP：

- `J u = d out / d h_final · u`

其中 `u = Δh_final / ||Δh_final||`。

这么做更干净，原因是：

- 它直接隔离了 `h_final -> out`
- 它包含了实际生效的 adapter 分支
- 它不混入 trunk / state-sync / pose_history / carry
- 比 input-side finite difference 更符合本轮“只拆 head-side amplifier”的目标

### 16.3 新脚本、命令与输出

新脚本：

- `tools/analyze_cp015_tailk7_motion_head_gain.py`

summary 输出：

- `debug_output/_tmp_cp015_tailk7_motion_head_gain_audit_20260405/summary.json`

smoke 输出：

- `debug_output/_tmp_cp015_tailk7_motion_head_gain_audit_20260405/smoke_summary.json`

本轮实际跑过的命令：

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_motion_head_gain.py
python3 tools/analyze_cp015_tailk7_motion_head_gain.py --rounds 1 --device cpu --out debug_output/_tmp_cp015_tailk7_motion_head_gain_audit_20260405/smoke_summary.json
python3 tools/analyze_cp015_tailk7_motion_head_gain.py --rounds 5 --device cpu --out debug_output/_tmp_cp015_tailk7_motion_head_gain_audit_20260405/summary.json
```

### 16.4 方法

#### 16.4.1 static spectral audit

对每个 case，分别计算：

- `motion_head` 每个 `Linear` 层的 `weight shape`
- 每层 `sigma_max`
- `motion_head` 的简单 upper bound = 两层 `sigma_max` 乘积
- 每个 `_bone_adapter` 分支的 `alpha_effective`
- 每个 adapter 内两层 `Linear` 的 `sigma_max`
- 每个 adapter 的 upper bound = `|alpha| * sigma_max(fc0) * sigma_max(fc1)`
- `full_head_conservative_upper_bound = motion_head_upper_bound + sum(adapter_upper_bound)`

这不是为了替代主结论，而是为了判断：

- “静态权重尺度差” 到底有多大
- 它是否足够解释 empirical matched-hidden gain

#### 16.4.2 matched-hidden one-step counterfactual

对同一个 teacher-conditioned clean base step：

1. 先从 tail 的 `motion_only` counterfactual 构造 observed hidden drift direction：
   - `Δmotion = motion_freerun - motion_teacher`
   - 固定其它 teacher-conditioned 输入
   - 只注入 `Δmotion`
   - 得到 observed `Δh_final = h_final(motion_only_perturbed) - h_final(clean)`

2. 再把这个 **完全相同的** `Δh_final` 同时注入：
   - tail clean `h_final`
   - baseline clean `h_final`

3. 对两个 clean base state 做同样的 head-side readout：
   - `out`
   - `y_inc_raw`
   - optional one-step `GeoLocalDeg`

4. 记录：
   - `||Δh_final||`
   - `||Δout||`
   - `||Δy_inc_raw||`
   - `gain_head_out = ||Δout|| / ||Δh_final||`
   - `gain_head_y = ||Δy_inc_raw|| / ||Δh_final||`
   - `local directional gain_out = ||J u||`

5. 再做 bucket-wise / slice-wise 聚合：
   - depth: `d0_9 / d10_20 / d21_43`
   - step_in_cycle: `sic0_10 / sic11_21 / sic22_43`
   - output slices: `rot / root_vel`

### 16.5 Motion-Head Static Spectral Table

| case | `motion_head.0` shape | `sigma_max` | `motion_head.3` shape | `sigma_max` | motion_head product | adapter upper sum | full conservative upper |
|---|---|---:|---|---:|---:|---:|---:|
| `tailk7_current_control` | `(512, 512)` | 1.9331 | `(278, 512)` | 3.6013 | 6.9618 | 0.1581 | 7.1198 |
| `baseline_replace` | `(512, 512)` | 1.8302 | `(278, 512)` | 3.1054 | 5.6836 | 0.0815 | 5.7652 |
| tail/base ratio | `-` | 1.0562 | `-` | 1.1597 | 1.2249 | 1.9384 | 1.2350 |

这张表先给出最重要的静态结论：

- 主 `motion_head` 的静态谱差异只有约 `1.22x`
- 把 adapter 分支也保守加进去，full head conservative upper 也只有约 `1.24x`
- adapter upper sum 的 ratio 虽然到 `1.94x`，但绝对量很小：`0.1581 vs 0.0815`

所以光靠“静态权重谱更大”并不能解释后面将看到的 `7x+` empirical gain ratio。

补一张 adapter 分支表：

| slice | joints | tail `alpha` | tail upper | base `alpha` | base upper | tail/base |
|---|---|---:|---:|---:|---:|---:|
| `192:198` | `thigh_l` | 0.0419 | 0.0099 | 0.0480 | 0.0121 | 0.8177 |
| `198:204` | `calf_l` | 0.0569 | 0.0326 | 0.0483 | 0.0098 | 3.3231 |
| `222:228` | `foot_l` | 0.0502 | 0.0327 | 0.0549 | 0.0176 | 1.8565 |
| `234:240` | `thigh_r` | 0.0385 | 0.0077 | 0.0429 | 0.0098 | 0.7824 |
| `240:246` | `calf_r` | 0.0599 | 0.0461 | 0.0509 | 0.0200 | 2.3046 |
| `264:270` | `foot_r` | 0.0635 | 0.0291 | 0.0478 | 0.0122 | 2.3811 |

这说明 adapter 确实也参与 head-side readout，但它们更多是放大 rot 的局部通道，不是单独就能解释整个 `7x+` overall gain ratio。

### 16.6 Same `Δh_final` 的 Overall Gain Table

`rows = 434`，并且 `mean ||Δh_final||` 在 tail / baseline 完全 matched：

| metric | tail | baseline | tail/base |
|---|---:|---:|---:|
| mean `L2(Δh_final)` | 10.4449 | 10.4449 | 1.0000 |
| mean `L2(Δout)` | 4.2836 | 0.5443 | 7.8702 |
| mean `L2(Δy_inc_raw)` | 0.0713 | 0.0285 | 2.5040 |
| `gain_head_out = L2(Δout) / L2(Δh_final)` | 0.3952 | 0.0514 | 7.6857 |
| `gain_head_y = L2(Δy_inc_raw) / L2(Δh_final)` | 0.006631 | 0.002772 | 2.3924 |
| `local directional gain_out = L2(Ju)` | 0.3651 | 0.0430 | 8.4850 |
| one-step `abs(ΔGeoLocalDeg)` | 0.1628 | 0.0235 | 6.9408 |

这张表已经足够回答本轮两个核心问题：

- 在完全相同的 `Δh_final` 下，tailk7 的 `out` 响应 **明显大于** baseline
- 在完全相同的 `Δh_final` 下，tailk7 的 `y_inc_raw` 响应 **明显大于** baseline

而且最关键的是：

- static full-head bound ratio 只有 `1.2350x`
- local directional gain ratio 却有 `8.4850x`

因此主因更像：

- `local Jacobian / activation regime`

而不是单纯：

- “静态权重尺度更大”

### 16.7 Depth Bucket 对照

| bucket | rows | tail `gain_out` | base `gain_out` | ratio | tail `gain_y` | base `gain_y` | ratio | tail local `L2(Ju)` | base local `L2(Ju)` | ratio | tail freerun `GeoLocalDeg` | base freerun `GeoLocalDeg` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `d0_9` | 10 | 0.0682 | 0.0668 | 1.0209 | 0.004185 | 0.003282 | 1.2753 | 0.0696 | 0.0636 | 1.0944 | 2.3881 | 2.6792 |
| `d10_20` | 11 | 0.1483 | 0.0607 | 2.4434 | 0.003940 | 0.003191 | 1.2346 | 0.1472 | 0.0666 | 2.2106 | 7.6484 | 8.4032 |
| `d21_43` | 23 | 0.2458 | 0.0702 | 3.5000 | 0.005165 | 0.003446 | 1.4988 | 0.2398 | 0.0737 | 3.2522 | 10.9665 | 12.1860 |

depth 读法：

- `d0_9` 基本还没有差异
- 到 `d10_20`，head-side `gain_out` ratio 已经到 `2.44x`
- 到 `d21_43`，进一步升到 `3.50x`

所以 depth 维度上也成立：

- amplifier pattern 从 `d10+` 开始抬头
- `d21_43` 比 `d10_20` 更强

但它没有 step-in-cycle bucket 那么尖锐。

### 16.8 Step-In-Cycle Bucket 对照

| bucket | rows | tail `gain_out` | base `gain_out` | ratio | tail `gain_y` | base `gain_y` | ratio | tail local `L2(Ju)` | base local `L2(Ju)` | ratio | tail freerun `GeoLocalDeg` | base freerun `GeoLocalDeg` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sic0_10` | 50 | 0.3085 | 0.0583 | 5.2923 | 0.006231 | 0.002972 | 2.0964 | 0.2912 | 0.0425 | 6.8552 | 53.3908 | 38.3757 |
| `sic11_21` | 55 | 0.3716 | 0.0474 | 7.8325 | 0.006289 | 0.002867 | 2.1936 | 0.3497 | 0.0466 | 7.4993 | 57.6182 | 41.5557 |
| `sic22_43` | 110 | 0.3840 | 0.0498 | 7.7079 | 0.006555 | 0.002849 | 2.3008 | 0.3523 | 0.0511 | 6.8972 | 61.6965 | 45.1371 |

这张表是本轮最关键的 pattern 证据：

- `sic11_21` 和 `sic22_43` 都稳定在 `gain_out ratio ~ 7.7x-7.8x`
- `gain_y ratio` 也稳定在 `2.19x-2.30x`
- local directional gain ratio 也同样高，说明不是 full-step 偶然现象

因此 matched-hidden gain pattern 仍然明确对齐：

- `sic11_21`
- `sic22_43`

### 16.9 Slice-Wise Amplification

当前 output 只有两个稳定 slice：

- `rot`：`width = 276`
- `root_vel`：`width = 2`

没有 `angvel / contacts / root_pos / other` output slice。

| slice | width | tail `gain_out` | base `gain_out` | ratio | tail local `L2(Ju)` | base local `L2(Ju)` | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rot` | 276 | 0.395042 | 0.051378 | 7.6889 | 0.364930 | 0.042971 | 8.4924 |
| `root_vel` | 2 | 0.010505 | 0.001807 | 5.8129 | 0.010138 | 0.001966 | 5.1574 |
| `angvel` | 0 | NA | NA | NA | NA | NA | NA |
| `contacts` | 0 | NA | NA | NA | NA | NA | NA |
| `root_pos` | 0 | NA | NA | NA | NA | NA | NA |
| `other` | 0 | NA | NA | NA | NA | NA | NA |

这张 slice 表有两个结论：

- amplification **主要集中在 `rot`**
- `root_vel` 也有放大，但无论绝对 gain 还是 ratio 都低于 `rot`

尤其是绝对量级上：

- `rot gain_out mean = 0.3950`
- `root_vel gain_out mean = 0.0105`

所以即便 `root_vel` 也有 head-side 放大，真正主导 `out` / `y_inc_raw` 差异的仍然是 `rot` readout。

### 16.10 Response Pattern vs GeoLocalDeg 坏窗口并表

| bad window | freerun `GeoLocalDeg` tail/base | `gain_out` ratio | `gain_y` ratio | local `L2(Ju)` ratio |
|---|---:|---:|---:|---:|
| `d10_20` | 7.6484 / 8.4032 | 2.4434 | 1.2346 | 2.2106 |
| `d21_43` | 10.9665 / 12.1860 | 3.5000 | 1.4988 | 3.2522 |
| `sic11_21` | 57.6182 / 41.5557 | 7.8325 | 2.1936 | 7.4993 |
| `sic22_43` | 61.6965 / 45.1371 | 7.7079 | 2.3008 | 6.8972 |

这张并表说明：

- 坏窗口最强的对齐仍然在 `sic11_21 / sic22_43`
- `d10_20 / d21_43` 也成立，但强度明显低于 cycle-phase bucket
- 所以如果后续 intervention 要优先打窗口，仍然应先盯：
  - `sic11_21`
  - `sic22_43`

### 16.11 最终判定

先按本轮唯一任务逐条回答：

1. 在完全相同的 `Δh_final` 下，tail 的 `out` 响应是否大于 baseline？

   - **是**
   - overall `gain_head_out ratio = 7.6857x`
   - `local directional gain_out ratio = 8.4850x`

2. 在完全相同的 `Δh_final` 下，tail 的 `y_inc_raw` 响应是否大于 baseline？

   - **是**
   - overall `gain_head_y ratio = 2.3924x`

3. 这种 head-side gain 差异主要是静态权重谱，还是 local Jacobian / activation regime？

   - **更像 local Jacobian / activation regime**
   - static full-head conservative upper ratio 只有 `1.2350x`
   - empirical local directional gain ratio 却达到 `8.4850x`
   - 所以静态权重尺度可能有贡献，但远不足以解释整体差异

4. matched-hidden gain pattern 是否仍对齐：
   - `d10_20 / d21_43`
   - `sic11_21 / sic22_43`

   - **是**
   - 其中：
     - depth 方向是 `d10_20 -> d21_43` 逐步增强
     - cycle-phase 方向则在 `sic11_21 / sic22_43` 最稳定、最强

5. amplification 是否集中在特定 output slice？

   - **是，主要集中在 `rot`**
   - `root_vel` 也有放大，但明显次级
   - 当前 output layout 下没有 `angvel / contacts / root_pos / other` 可切分 slice

因此本轮 A / B / C / D 判定为：

- **A：成立**
  - same `Δh_final` 下，tail 的 `out / y_inc_raw` gain 仍显著大于 baseline
  - 直接验证 amplifier 在 `motion_head / head-side readout`

- **B：不成立**
  - same `Δh_final` 下，tail/base 差异没有明显收缩
  - 所以不需要回退到 “更早层才是主因”

- **C：成立**
  - 静态谱差只有 `~1.24x`
  - empirical local head gain 却到 `~8.49x`
  - 主因更像 `activation regime / local Jacobian`

- **D：成立**
  - amplification 主要集中在 `rot`
  - 后续若要 intervention，更该优先打：
    - `rot readout`
    - `rot-related blend / regularization`
  - 而不是再优先回去改 donor hidden dynamics

如果只保留一句最终总结：

- **本轮 matched-hidden audit 直接确认了 amplifier 就在 head-side readout：same `Δh_final` 下，tailk7 的 `out` gain 约为 baseline 的 `7.69x`，`y_inc_raw` gain 约为 `2.39x`；静态谱差只有 `~1.24x`，所以主因不是简单权重变大，而是更强的 local Jacobian / activation regime，并且放大主要落在 rot slice。**

## 17. Rot Readout Main-vs-Adapter Decomposition

本轮目标不是再回到 trunk / donor hidden dynamics，而是把已经确认存在的 head-side `rot` amplification 继续拆成：

- `motion_head` 主干本体
- `_bone_adapters` 路径

核心问题只有一个：

- same `Δh_final` 下，tailk7 相比 baseline 的 matched-hidden `rot` amplification，主要来自 `main head` 还是 `adapters`

### 17.1 Code Facts And Definitions

代码里的精确实现点仍然在 `train/models.py`：

- `h_final = coupling_norm(...)`
- `out_main = motion_head(h_final)`
- 对每个 `(slice_i, adapter_i)`：
  - `delta_full[..., slice_i] = adapter_i(h_final)`
- `out_total = out_main + delta_full`

因此本轮定义：

- `out_main = motion_head(h_final)`
- `out_adapter_total = sum_i scatter(adapter_i(h_final), slice_i -> full out dim)`
- `out_total = out_main + out_adapter_total`

其中本轮只看 `rot` readout：

- `rot = [0:276]`
- `root_vel = [276:278]`

adapter slice 与 joint 对应关系如下，并且这 6 个 slice **全部都落在 `rot` slice 内部**：

| adapter | full out slice | joint |
|---|---|---|
| `adapter_thigh_l` | `[192:198]` | `thigh_l` |
| `adapter_calf_l` | `[198:204]` | `calf_l` |
| `adapter_foot_l` | `[222:228]` | `foot_l` |
| `adapter_thigh_r` | `[234:240]` | `thigh_r` |
| `adapter_calf_r` | `[240:246]` | `calf_r` |
| `adapter_foot_r` | `[264:270]` | `foot_r` |

最干净的 decomposition 点也就明确了：

- `out_main_rot = out_main[..., 0:276]`
- `out_adapter_rot = out_adapter_total[..., 0:276]`
- `out_total_rot = out_total[..., 0:276]`

对单个 adapter `i`，本轮记：

- `out_adapter_i` = 该 adapter 的 full-rot scattered contribution
- 它只有自己的 6D slice 非零，所以 `L2(scattered full-rot)` 与 `L2(local 6D slice)` 完全等价

### 17.2 Matched-Hidden Method

仍然沿用上一轮已经验证过的 same `Δh_final` setup：

1. 在 tail teacher-conditioned clean step 上测 observed motion-induced hidden drift：
   - `Δmotion = motion_freerun - motion_teacher`
   - 固定其它 teacher-conditioned 输入
   - 得到 observed `Δh_final`
2. 把 **完全相同的** `Δh_final` 同时加到：
   - tail clean `h_final`
   - baseline clean `h_final`
3. 分别只做 head-side rot readout decomposition，记录：
   - finite response：
     - `L2(Δout_rot_total)`
     - `L2(Δout_rot_main)`
     - `L2(Δout_rot_adapter_total)`
     - `L2(Δout_rot_adapter_i)`
   - gain：
     - `gain_rot_part = L2(Δout_rot_part) / L2(Δh_final)`
   - local directional Jacobian：
     - 令 `u = Δh_final / ||Δh_final||`
     - 计算 `L2(J_part,rot u)`

注意：

- `Δout_rot_total = Δout_rot_main + Δout_rot_adapter_total`
- 但 **L2 norm 不可加**，所以会出现 `gain_rot_main` 略大于 `gain_rot_total`
- 这表示 adapter 分量与 main 分量在向量上有轻微抵消，不表示定义有误

### 17.3 Run Artifacts

新脚本：

- `tools/analyze_cp015_tailk7_rot_readout_decomposition.py`

summary 输出：

- `debug_output/_tmp_cp015_tailk7_rot_readout_decomposition_20260405/summary.json`

smoke 输出：

- `debug_output/_tmp_cp015_tailk7_rot_readout_decomposition_20260405/smoke_summary.json`

本轮实际跑过的命令：

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_rot_readout_decomposition.py
python3 tools/analyze_cp015_tailk7_rot_readout_decomposition.py --rounds 1 --device cpu --out debug_output/_tmp_cp015_tailk7_rot_readout_decomposition_20260405/smoke_summary.json
python3 tools/analyze_cp015_tailk7_rot_readout_decomposition.py --rounds 5 --device cpu --out debug_output/_tmp_cp015_tailk7_rot_readout_decomposition_20260405/summary.json
```

### 17.4 Static Decomposition Table

先给静态 conservative upper bound 分解：

| component | tail | baseline | tail/base |
|---|---:|---:|---:|
| `motion_head` upper bound | 6.9618 | 5.6836 | 1.2249 |
| `bone_adapters_sum` upper bound | 0.1581 | 0.0815 | 1.9384 |
| `full_head_conservative_upper` | 7.1198 | 5.7652 | 1.2350 |

静态 adapter 分支表：

| joint | slice | tail `alpha` | tail upper | base `alpha` | base upper | tail/base |
|---|---|---:|---:|---:|---:|---:|
| `thigh_l` | `[192:198]` | 0.0419 | 0.0099 | 0.0480 | 0.0121 | 0.8177 |
| `calf_l` | `[198:204]` | 0.0569 | 0.0326 | 0.0483 | 0.0098 | 3.3231 |
| `foot_l` | `[222:228]` | 0.0502 | 0.0327 | 0.0549 | 0.0176 | 1.8565 |
| `thigh_r` | `[234:240]` | 0.0385 | 0.0077 | 0.0429 | 0.0098 | 0.7824 |
| `calf_r` | `[240:246]` | 0.0599 | 0.0461 | 0.0509 | 0.0200 | 2.3046 |
| `foot_r` | `[264:270]` | 0.0635 | 0.0291 | 0.0478 | 0.0122 | 2.3811 |

静态结论先不变：

- `motion_head` 静态谱差只有 `1.2249x`
- adapters 静态 ratio 更大，但绝对量仍然很小
- 所以 static bound 依旧不足以解释后面的 `7x+` empirical amplification

### 17.5 Same `Δh_final` Overall Rot Decomposition

`rows = 434`，并且 same `Δh_final` 完全 matched：

- mean `L2(Δh_final) = 10.4449`

整体 rot decomposition 表：

| part | tail `L2(Δout_rot)` | baseline `L2(Δout_rot)` | tail/base | tail `gain_rot` | baseline `gain_rot` | tail/base | tail `L2(J_rot u)` | baseline `L2(J_rot u)` | tail/base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `total` | 4.2821 | 0.5439 | 7.8734 | 0.395042 | 0.051378 | 7.6889 | 0.364930 | 0.042971 | 8.4924 |
| `main` | 4.3127 | 0.5423 | 7.9524 | 0.397477 | 0.051244 | 7.7566 | 0.367289 | 0.042953 | 8.5510 |
| `adapter_total` | 0.1263 | 0.0136 | 9.2782 | 0.010301 | 0.001265 | 8.1412 | 0.010355 | 0.001208 | 8.5728 |

这张表是本轮最关键的事实：

- adapter 路径的 **ratio** 的确也高
- 但 adapter 的 **绝对 gain** 极小：
  - tail `gain_rot_main = 0.397477`
  - tail `gain_rot_adapter_total = 0.010301`
- overall excess 也极不对称：
  - main excess gain = `0.346233`
  - adapter excess gain = `0.009036`
  - main / adapter excess gain = `38.32x`
- local Jacobian 也是同样结论：
  - main excess local-dir = `0.324336`
  - adapter excess local-dir = `0.009147`
  - main / adapter excess local-dir = `35.46x`

也就是说：

- **不能只看 tail/base ratio**
- 因为 baseline adapter path 本来就几乎为零，所以 adapter ratio 会被放大
- 但真正解释绝大多数 `rot amplification` 的，仍然是 `main motion_head`

### 17.6 Depth Bucket Table

| bucket | rows | tail/base freerun `GeoLocalDeg` | `gain_rot_total` ratio | `gain_rot_main` ratio | `gain_rot_adapter` ratio | local `total` ratio | local `main` ratio | local `adapter` ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `d0_9` | 10 | 2.3881 / 2.6792 | 1.0170 | 1.0226 | 1.5012 | 1.0908 | 1.0971 | 1.5219 |
| `d10_20` | 11 | 7.6484 / 8.4032 | 2.4431 | 2.4443 | 1.5823 | 2.2099 | 2.2117 | 1.5720 |
| `d21_43` | 23 | 10.9665 / 12.1860 | 3.4982 | 3.5043 | 1.6330 | 3.2501 | 3.2551 | 1.6543 |

depth 上的读法很直接：

- `d10_20 / d21_43` 的 pattern 仍然成立
- 但增长主因明显落在 `main`
- 用 excess gain 看更清楚：
  - `d10_20`：main excess / adapter excess = `93.28x`
  - `d21_43`：main excess / adapter excess = `258.79x`

所以 depth bucket 里 adapter 不是主因，只是附带放大。

### 17.7 Step-In-Cycle Bucket Table

| bucket | rows | tail/base freerun `GeoLocalDeg` | `gain_rot_total` ratio | `gain_rot_main` ratio | `gain_rot_adapter` ratio | local `total` ratio | local `main` ratio | local `adapter` ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sic0_10` | 50 | 53.3908 / 38.3757 | 5.2984 | 5.3549 | 5.9481 | 6.8729 | 6.9374 | 6.2643 |
| `sic11_21` | 55 | 57.6182 / 41.5557 | 7.8394 | 7.8970 | 10.2222 | 7.5027 | 7.5535 | 10.5993 |
| `sic22_43` | 110 | 61.6965 / 45.1371 | 7.7085 | 7.7710 | 9.4412 | 6.8986 | 6.9418 | 9.9222 |

这张表说明两点要同时读：

- `sic11_21 / sic22_43` 仍然是最强坏窗口，对齐关系没有变
- adapter ratio 在这两个坏窗口里确实更尖

但如果看绝对量，结论仍然没变：

- `sic11_21`：main excess / adapter excess = `37.33x`
- `sic22_43`：main excess / adapter excess = `37.31x`

所以 cycle-phase 坏窗口里的放大，仍然是：

- **主因 = main head**
- **adapter = 次级、随窗口同步增强的局部 leg 通道**

### 17.8 Adapter-Wise Amplification Table

overall adapter-wise 表：

| joint | slice | tail `gain_adapter_i` | baseline `gain_adapter_i` | tail/base | gain excess | tail local `L2(J_i u)` | baseline local `L2(J_i u)` | tail/base | local excess |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `thigh_l` | `[192:198]` | 0.000040 | 0.000262 | 0.1518 | -0.000222 | 0.000025 | 0.000249 | 0.1010 | -0.000224 |
| `calf_l` | `[198:204]` | 0.004804 | 0.000138 | 34.9261 | 0.004666 | 0.004844 | 0.000138 | 35.1891 | 0.004706 |
| `foot_l` | `[222:228]` | 0.004529 | 0.000888 | 5.0994 | 0.003641 | 0.004539 | 0.000871 | 5.2107 | 0.003668 |
| `thigh_r` | `[234:240]` | 0.000061 | 0.000269 | 0.2257 | -0.000208 | 0.000063 | 0.000206 | 0.3065 | -0.000143 |
| `calf_r` | `[240:246]` | 0.006958 | 0.000650 | 10.7099 | 0.006309 | 0.006999 | 0.000592 | 11.8207 | 0.006407 |
| `foot_r` | `[264:270]` | 0.003495 | 0.000168 | 20.7619 | 0.003327 | 0.003503 | 0.000158 | 22.1120 | 0.003344 |

adapter-wise 的主次也很清楚：

- 正贡献集中在：
  - `calf_r`
  - `calf_l`
  - `foot_l`
  - `foot_r`
- `thigh_l / thigh_r` 不但不是主因，反而在 overall 上弱于 baseline

坏窗口里这个排序仍然基本不变：

- `sic11_21`：正的 adapter excess 主要是 `calf_r > foot_l ~= calf_l > foot_r`
- `sic22_43`：正的 adapter excess 主要是 `calf_r > calf_l > foot_l > foot_r`
- `thigh_l / thigh_r` 在这些坏窗口里仍然是负贡献

所以如果只问 “adapter 有没有贡献，并且集中在哪些 joint”：

- **有**
- **主要集中在双侧 calf / foot，而不是 thigh**

### 17.9 Response Pattern vs GeoLocalDeg Bad Windows

| bad window | tail/base freerun `GeoLocalDeg` | `gain_rot_total` ratio | `gain_rot_main` ratio | `gain_rot_adapter` ratio | local `total` ratio | local `main` ratio | local `adapter` ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d10_20` | 7.6484 / 8.4032 | 2.4431 | 2.4443 | 1.5823 | 2.2099 | 2.2117 | 1.5720 |
| `d21_43` | 10.9665 / 12.1860 | 3.4982 | 3.5043 | 1.6330 | 3.2501 | 3.2551 | 1.6543 |
| `sic11_21` | 57.6182 / 41.5557 | 7.8394 | 7.8970 | 10.2222 | 7.5027 | 7.5535 | 10.5993 |
| `sic22_43` | 61.6965 / 45.1371 | 7.7085 | 7.7710 | 9.4412 | 6.8986 | 6.9418 | 9.9222 |

这张并表的真正读法是：

- pattern 对齐仍然最强在：
  - `sic11_21`
  - `sic22_43`
- `d10_20 / d21_43` 也成立，但明显更弱
- bad window 里 adapter ratio 确实更尖
- 但 bad window 的 **大头绝对放大量** 仍然由 main head 提供

### 17.10 Final Conclusion

先按本轮唯一问题直接判定：

- tailk7 相比 baseline 的 matched-hidden `rot amplification`：
  - **C. 两者都有，但 main `motion_head` 是主因**

为什么不是 D：

- adapter 的 tail/base ratio 也高，尤其在 `sic11_21 / sic22_43`
- 但 adapter 的绝对 matched-hidden gain 只有 `0.010301`
- main 的绝对 matched-hidden gain 是 `0.397477`
- main excess gain / adapter excess gain = `38.32x`
- main excess local Jacobian / adapter excess local Jacobian = `35.46x`

因此这轮最稳的解释是：

- amplification 的**决定性来源**是 `motion_head` 主干的 local rot readout Jacobian / activation regime
- adapters 只是把这件事进一步集中到局部 leg rot channels，主要是：
  - `calf_r`
  - `calf_l`
  - `foot_l`
  - `foot_r`

面向下一步 intervention，只保留一句建议：

- **应优先打 global rot readout**

如果后续第一轮全局 rot readout intervention 还不能明显收敛，再第二优先考虑：

- `adapter / local leg rot channels`

## 18. No-Train Main-Head Causal Swaps

上一节已经确认：

- matched-hidden `rot amplification` 的主因在 `main motion_head`
- adapters 只有次级贡献

但还剩一个更关键的问题：

- 后续 intervention 应该先打 **整个 `motion_head` nonlinear regime**
- 还是更窄地先打 **final global `rot` readout rows `[0:276]`**

所以本轮不训练，只做 no-train in-memory causal swaps。

### 18.1 Swap Definitions

所有 swap 都以 tail current control 为底座，且不改 runtime contract，不改 trunk，不改 donor，不改 downstream。

本轮实际评估的 6 个 case：

| case | 定义 |
|---|---|
| `baseline_reference` | baseline 原模型 reference |
| `tail_current_control` | tail current control |
| `swap_full_head_keep_adapters` | tail current + baseline full `motion_head` + tail adapters |
| `swap_final_rot_rows_keep_adapters` | tail current + baseline final `rot` rows `[0:276]` + 其余保持 tail |
| `zero_adapters_tail_head` | tail current + tail `motion_head` + adapters 置零 |
| `swap_full_head_zero_adapters` | tail current + baseline full `motion_head` + adapters 置零 |

其中：

- “adapters 置零” 的实现是把每个 `_bone_adapters[i].alpha := 0`
- `final rot rows [0:276]` 只替换 `motion_head` 最后一层 Linear 的前 `276` 个输出 rows

### 18.2 Metric Semantics And Pose-Primary Reading

这里需要先纠正阅读口径，避免后续误解。

本轮 no-train causal swap 的**主判据应是 pose-side metric**，不是 `DirectGeoLocalDeg`，也不应该把 `GeoLocalDeg` 误读成“本轮最终目标指标”。

运行时代码里的语义是：

- `DirectGeoLocalDeg` 走 `predY_direct`
- `GeoLocalDeg` 走 rollout 实际使用的 `y_used_raw`
- 当前要打的是 `motion_head` incremental readout，而不是 direct branch

因此：

- `DirectGeoLocalDeg` 在所有 `motion_head` swap 下几乎完全不变，这个现象是**符合代码语义**的
- 但如果本轮目标是回答 **pose error**，第 18 节的主阅读口径应切到 `Rot6dLocalL2`
- `GeoLocalDeg` 只保留为 **secondary closed-loop proxy / corroboration**

另一个需要写清楚的事实是：

- `RootPosErr` 在这轮 swap 下完全不变
- 原因不是“root 没问题”，而是本轮操作只改 `rot[0:276]` rows，没有动 root-position path
- 所以 `RootPosErr` 不能作为本轮 causal ranking metric

### 18.3 Run Artifacts

新脚本：

- `tools/analyze_cp015_tailk7_motion_head_causal_swaps.py`

summary 输出：

- `debug_output/_tmp_cp015_tailk7_motion_head_causal_swaps_20260405/summary.json`

smoke 输出：

- `debug_output/_tmp_cp015_tailk7_motion_head_causal_swaps_20260405/smoke_summary.json`

本轮实际跑过的命令：

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_motion_head_causal_swaps.py
python3 tools/analyze_cp015_tailk7_motion_head_causal_swaps.py --rounds 1 --device cpu --out debug_output/_tmp_cp015_tailk7_motion_head_causal_swaps_20260405/smoke_summary.json
python3 tools/analyze_cp015_tailk7_motion_head_causal_swaps.py --rounds 5 --device cpu --out debug_output/_tmp_cp015_tailk7_motion_head_causal_swaps_20260405/summary.json
```

### 18.4 Overall Pose Table

如果把本轮目标明确写成 **pose-side rotation error**，主表应先看 `Rot6dLocalL2`：

| case | overall `Rot6dLocalL2` | tail→baseline gap closed |
|---|---:|---:|
| `baseline_reference` | 0.7692 | 1.0000 |
| `tail_current_control` | 0.9966 | 0.0000 |
| `swap_full_head_keep_adapters` | 1.3376 | -1.4994 |
| `swap_final_rot_rows_keep_adapters` | 0.8852 | 0.4899 |
| `zero_adapters_tail_head` | 0.9837 | 0.0570 |
| `swap_full_head_zero_adapters` | 1.3343 | -1.4848 |

按 pose metric 直接读，这张表给出的结论其实和上一轮 head-side amplification 是一致的：

1. `baseline full motion_head` 直接塞进 tail current **明显更差**
2. 只换 `final rot rows [0:276]` 就能收掉约 **49.0%** 的 pose gap
3. 单独把 adapters 置零只带来约 **5.7%** 的弱改善

因此如果问题定义是“pose 误差优先”，这里仍然不支持：

- whole-head transplant
- adapter-first intervention

它支持的是：

- **优先打 final global rot readout rows**

### 18.5 Pose Bucket Tables

先看 depth bucket 上的 `Rot6dLocalL2`：

| case | `d10_20` `Rot6dLocalL2` | `d21_43` `Rot6dLocalL2` |
|---|---:|---:|
| `baseline_reference` | 0.1532 | 0.2255 |
| `tail_current_control` | 0.1395 | 0.2043 |
| `swap_full_head_keep_adapters` | 0.6029 | 0.9337 |
| `swap_final_rot_rows_keep_adapters` | 0.1800 | 0.3017 |
| `zero_adapters_tail_head` | 0.1360 | 0.2022 |
| `swap_full_head_zero_adapters` | 0.6014 | 0.9295 |

这里的读法和前面一致：

- `d10_20 / d21_43` 不是当前最核心的 pose 失败窗口
- `swap_final_rot_rows_keep_adapters` 不是“所有 bucket 都变好”
- whole-head swap 会显著破坏 tail 现有 hidden/head 适配

真正关键的还是坏 cycle windows：

| case | `sic11_21` `Rot6dLocalL2` | `sic22_43` `Rot6dLocalL2` |
|---|---:|---:|
| `baseline_reference` | 0.6947 | 0.7528 |
| `tail_current_control` | 0.9184 | 0.9767 |
| `swap_full_head_keep_adapters` | 1.2649 | 1.3318 |
| `swap_final_rot_rows_keep_adapters` | 0.8234 | 0.8695 |
| `zero_adapters_tail_head` | 0.9096 | 0.9647 |
| `swap_full_head_zero_adapters` | 1.2686 | 1.3296 |

对 pose error 的关键读法是：

- `swap_full_head_keep_adapters`
  - `sic11_21`: `0.9184 -> 1.2649`
  - `sic22_43`: `0.9767 -> 1.3318`
  - 明显是 **错方向**
- `swap_final_rot_rows_keep_adapters`
  - `sic11_21`: `0.9184 -> 0.8234`
  - `sic22_43`: `0.9767 -> 0.8695`
  - 是 **明显对方向**
  - 分别收掉这些坏窗口 pose gap 的：
    - `42.5%`
    - `47.9%`
- `zero_adapters_tail_head`
  - `sic11_21`: `0.9184 -> 0.9096`
  - `sic22_43`: `0.9767 -> 0.9647`
  - 只有很弱的改善

所以如果第 18 节按 pose metric 来读，真正重要的 causal 结论仍然是：

- **baseline final rot rows 更像正确 intervention 方向**

### 18.6 Closed-Loop Proxy Table

`GeoLocalDeg` 不再作为第 18 节的主排序指标，但可以作为 secondary closed-loop proxy 留存，检查方向是否一致。

全局 `GeoLocalDeg` 如下：

| case | overall `GeoLocalDeg` | tail→baseline gap closed |
|---|---:|---:|
| `baseline_reference` | 46.0819 | 1.0000 |
| `tail_current_control` | 63.3033 | 0.0000 |
| `swap_full_head_keep_adapters` | 86.8998 | -1.3702 |
| `swap_final_rot_rows_keep_adapters` | 54.6050 | 0.5051 |
| `zero_adapters_tail_head` | 62.3180 | 0.0572 |
| `swap_full_head_zero_adapters` | 86.6102 | -1.3534 |

这里的价值只是在于：

- 它与 pose-side `Rot6dLocalL2` 的方向**没有冲突**
- 但后续不应该把这一表当成“下一轮 intervention 的唯一主表”

### 18.7 Adapter Necessity Check

adapter 是否是必需条件，也应按 pose metric 直接回答。

先看在 tail 现有 head 下：

| compare | overall `Rot6dLocalL2` delta | `sic11_21` delta | `sic22_43` delta |
|---|---:|---:|---:|
| `tail_current_control -> zero_adapters_tail_head` | `-0.0130` | `-0.0088` | `-0.0120` |

再看在 baseline full head 下：

| compare | overall `Rot6dLocalL2` delta | `sic11_21` delta | `sic22_43` delta |
|---|---:|---:|---:|
| `swap_full_head_keep_adapters -> swap_full_head_zero_adapters` | `-0.0033` | `+0.0037` | `-0.0022` |

这两组对比说明：

- adapter 不是主因
- adapter 也不是 full-head swap 变差的必需条件
- 它更像是 **弱二阶项**

### 18.8 Whole Head vs Final Rot Rows

这一步的主结论也应该先按 pose metric 来写。

直接对比：

| compare | overall `Rot6dLocalL2` delta | `d10_20` delta | `d21_43` delta | `sic11_21` delta | `sic22_43` delta |
|---|---:|---:|---:|---:|---:|
| `swap_full_head_keep_adapters -> swap_final_rot_rows_keep_adapters` | `-0.4523` | `-0.4229` | `-0.6320` | `-0.4416` | `-0.4624` |

也就是说：

- 只换 `final rot rows` 比换整颗 baseline `motion_head` **好非常多**
- whole-head baseline transplant 与 tail hidden regime 明显不兼容
- 但 baseline 的 final global rot readout rows 确实能缓解 tail current 的 pose error，尤其是在坏 cycle windows

更接近的因果解释是：

- 问题**不主要是**“整颗 `motion_head` 的 nonlinear regime 都必须换回 baseline”
- 问题更像是：
  - tail 当前 hidden regime 已经与自己的前段 head 共同适配
  - 但 **final global rot readout rows** 在坏 cycle windows 上把 `rot` 读出方向推得过激

### 18.9 Final Conclusion

把第 18 节严格按 pose-primary 口径压缩成一句话，结论是：

- **下一步应优先做 final global `rot` readout rows `[0:276]` intervention，而不是整颗 `motion_head` transplant，也不是 adapter-first intervention。**

展开成三条就是：

1. **不要优先打 whole `motion_head` transplant**
   - baseline full `motion_head` swap 在 pose metric 上明显恶化

2. **应该优先打 final global `rot` readout rows `[0:276]`**
   - 只换 final rot rows 就能在 `Rot6dLocalL2` 上收掉约 `49.0%` 的 overall tail→baseline gap
   - 在最坏的 `sic11_21 / sic22_43` 上也能收掉约 `42.5% / 47.9%`

3. **adapter 不是必需条件，也不是主因**
   - 单独 zero adapters 只有弱改善
   - 在 full-head swap 下 zero adapters 也几乎不改变结论

如果后续要面向其他人复述这一节，推荐直接这么说：

- 本轮 no-train causal swap 的**主表是 `Rot6dLocalL2`**
- `GeoLocalDeg` 只是闭环 proxy，不是本轮 pose-target ranking metric
- 结论仍然是：**先打 final rot rows，不先打 adapters**

### 18.10 Next-Round Pose-Primary Prompt

下面这段 prompt 可以直接用于下一轮：

```text
继续做 cp015 tailk7 的单一 analysis follow-up，这一轮严格以 pose-side metric 为主，不要再让 GeoLocalDeg 主导结论。

核心目标：
在 no-train 前提下，把 tail current control 的 final global rot readout rows [0:276] 进一步做 joint-group causal falsifier，回答：

1. 哪些 final rot row groups 对 pose error 的改善贡献最大？
2. 是否存在“只换腿部 rows 就够”的更小 intervention？
3. adapter 是否仍然只是二阶项，而不是下一步第一优先级？

固定前提：
- 已经成立：whole motion_head transplant 是错方向
- 已经成立：baseline final rot rows [0:276] swap 能明显改善
- 已经成立：adapters 只有弱二阶贡献
- 这轮不要回 trunk / donor / downstream
- 这轮不要训练
- 不改 runtime contract
- 不新建平行文档，只追加主文档

本轮 primary metrics：
- Rot6dLocalL2
- Rot6dLocalL2Weighted
- GeoDeg
- KeyBoneGeoDegMean
- KeyBoneGeoLocalDegMean

secondary only：
- GeoLocalDeg

不要把下列指标当主排序指标：
- DirectGeoLocalDeg
- RootPosErr

需要做的 no-train variants：
- tail current control
- swap final rot rows for thigh_l + thigh_r
- swap final rot rows for calf_l + calf_r
- swap final rot rows for foot_l + foot_r
- swap final rot rows for all leg rows
- swap final rot rows for non-leg rows

如果计算量允许，再做 alpha-blend：
- alpha in {0.25, 0.5, 0.75, 1.0}
- 只对最有效的 1-2 个 row groups 做

bucket-wise 必做：
- depth: d0_9 / d10_20 / d21_43
- step_in_cycle: sic0_10 / sic11_21 / sic22_43

重点结论必须优先回答：
- pose error 的主要改善是否集中在 specific leg row groups？
- 下一步应该：
  - A. 直接训练 final rot rows [0:276] 全部
  - B. 只训练特定 leg row groups
  - C. 先打 adapters

建议新脚本名：
- tools/analyze_cp015_tailk7_rot_row_group_pose_swaps.py

文档追加到主文档：
- docs/retired_directions/replace_redesign_and_falsifier_family/2026-04-04_cp015_tailk7_replace_closed_loop_stability_falsifier.md

新 section 标题：
- ## 19. Pose-Primary Final-Rot-Row Group Causal Swaps

输出要求：
1. 先给 code facts 和 row-group mapping
2. 再给 primary pose tables
3. 再给 closed-loop proxy table
4. 最后给一句明确 intervention 建议

最终必须明确写：
- 本轮 primary metric 是 pose-side metric，不是 GeoLocalDeg
- 下一步应优先打：
  - final rot rows 全局
  - 或者特定 leg row groups
  - 不能含糊
```

## 19. Pose-Primary Final-Rot-Row Group Causal Swaps

> Last updated: 2026-04-06
> 本节严格按 pose-side primary metrics 排序：
> `Rot6dLocalL2 / Rot6dLocalL2Weighted / GeoDeg / KeyBoneGeoDegMean / KeyBoneGeoLocalDegMean`
> `GeoLocalDeg` 只作 secondary proxy，不参与主排序。

### 19.1 运行与产物

新增脚本：

- `tools/analyze_cp015_tailk7_rot_row_group_pose_swaps.py`

summary：

- `debug_output/_tmp_cp015_tailk7_rot_row_group_pose_swaps_20260405/summary.json`

实际命令：

```bash
python3 -m py_compile tools/analyze_cp015_tailk7_rot_row_group_pose_swaps.py
python3 tools/analyze_cp015_tailk7_rot_row_group_pose_swaps.py --rounds 5 --device cpu --alpha-top-k 2
```

固定实现事实：

- 所有 case 都从同一个 `tail current control` checkpoint 出发，只在内存里改 final `motion_head` 最后一层对 `rot[0:276]` 的 row。
- row ranking 最终改成了“相对 `tail_current_control` 的 primary relative improvement”，而不是“相对 `swap_final_rot_rows_all` 的 gap-closure”。
- 原因是：`swap_final_rot_rows_all` 对 `GeoDeg / Rot6dLocalL2` 是改进，但对 `Rot6dLocalL2Weighted / KeyBoneGeoDegMean / KeyBoneGeoLocalDegMean` 不是单调改进；如果强行拿它当唯一分母，会把排序扭曲。

### 19.2 Code Facts 与 Row Mapping

本轮 `rot_slice=[0,276)`，共 `46` joints，root=`pelvis`。

关键 row-group mapping：

| group | joints | rows | row count |
|---|---|---:|---:|
| `thigh_pair` | `thigh_l`, `thigh_r` | `[192:198) + [234:240)` | `12` |
| `calf_pair` | `calf_l`, `calf_r` | `[198:204) + [240:246)` | `12` |
| `foot_pair` | `foot_l`, `foot_r` | `[222:228) + [264:270)` | `12` |
| `all_leg_rows` | `thigh/calf/calf_twist/foot/ball` | `[192:276)` | `84` |
| `non_leg_rows` | pelvis/spine/arms/head 等 | `[0:192)` | `192` |
| anchor `all_rot_rows` | 全部 rot rows | `[0:276)` | `276` |

补充：

- `all_leg_rows` 占全部 rot rows 的 `30.4%`
- `non_leg_rows` 占 `69.6%`
- `all_leg_rows` 不是只含 `thigh/calf/foot` 三对骨，而是包含 `L/RCalfTwist_*` 与 `ball_*`

### 19.3 Primary Pose Tables

#### 19.3.1 Hard swaps：overall 相对 `tail current` 的 primary 改善率

单位：相对 `tail current control` 的 improvement ratio；正值表示更好。

| variant | rows | primary mean | wins / 5 | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `swap_final_rot_rows_non_leg_rows` | `192` | `+11.17%` | `4/5` | `+18.14%` | `-5.55%` | `+17.59%` | `+10.79%` | `+14.87%` |
| `swap_final_rot_rows_all_leg_rows` | `84` | `+5.97%` | `5/5` | `+2.06%` | `+16.45%` | `+4.80%` | `+3.28%` | `+3.29%` |
| `swap_final_rot_rows_all` | `276` | `+1.72%` | `2/5` | `+11.18%` | `-12.52%` | `+12.84%` | `-2.62%` | `-0.29%` |
| `swap_final_rot_rows_calf_pair` | `12` | `-0.40%` | `0/5` | `-0.08%` | `-0.48%` | `-0.20%` | `-0.60%` | `-0.66%` |
| `swap_final_rot_rows_thigh_pair` | `12` | `-1.94%` | `0/5` | `-0.62%` | `-1.61%` | `-0.65%` | `-3.40%` | `-3.41%` |
| `swap_final_rot_rows_foot_pair` | `12` | `-2.06%` | `0/5` | `-2.08%` | `-2.12%` | `-1.30%` | `-2.42%` | `-2.37%` |

直接结论：

1. **hard swap 里贡献最大的不是 isolated leg pair，而是 `non_leg_rows`。**
2. **`all_leg_rows` hard swap 是第二名，而且 5 个 primary metrics 全正。**
3. **精确到 `thigh/calf/foot` 三个 12-row pair 都不成立，连 `calf_pair` 都只是在 0 附近微负。**
4. `swap_final_rot_rows_all` 本身不是最优 hard intervention，说明“把全部 `[0:276]` 一次性硬换掉”并不是最干净的读出修正。

#### 19.3.2 Alpha-blend：最有效 2 个 row groups

这里按上面的 hard ranking 只补做了 `non_leg_rows` 和 `all_leg_rows` 的 alpha。

| variant | rows | primary mean | wins / 5 | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alpha_blend_all_leg_rows_a025` | `84` | `+11.86%` | `5/5` | `+4.68%` | `+10.54%` | `+7.91%` | `+17.97%` | `+18.20%` |
| `alpha_blend_non_leg_rows_a050` | `192` | `+9.39%` | `5/5` | `+13.95%` | `+3.37%` | `+14.84%` | `+6.52%` | `+8.28%` |
| `swap_final_rot_rows_non_leg_rows` (`alpha=1.0`) | `192` | `+11.17%` | `4/5` | `+18.14%` | `-5.55%` | `+17.59%` | `+10.79%` | `+14.87%` |
| `swap_final_rot_rows_all_leg_rows` (`alpha=1.0`) | `84` | `+5.97%` | `5/5` | `+2.06%` | `+16.45%` | `+4.80%` | `+3.28%` | `+3.29%` |

alpha 读法：

- `non_leg_rows` 如果追求 raw overall 改善，`alpha=1.0` 仍然最大，但它会把 `Rot6dLocalL2Weighted` 拉坏。
- `all_leg_rows` 的 **best balanced alpha 是 `0.25`**：
  - overall primary mean `+11.86%`
  - `5/5` primary metrics 全正
  - 这是本轮最像“更小但够用”的 intervention。

### 19.4 Bucket-Wise Primary Table

下表是 5 个 primary metrics 在各 bucket 上的平均 relative improvement（相对 `tail current`）：

| variant | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |
|---|---:|---:|---:|---:|---:|---:|
| `swap_final_rot_rows_all` | `-42.91%` | `-28.13%` | `-50.28%` | `-2.58%` | `-0.86%` | `+1.80%` |
| `swap_final_rot_rows_non_leg_rows` | `-24.83%` | `-19.32%` | `-30.54%` | `+10.65%` | `+8.45%` | `+9.59%` |
| `swap_final_rot_rows_all_leg_rows` | `-17.11%` | `-8.35%` | `-19.95%` | `+4.85%` | `+7.31%` | `+5.86%` |
| `alpha_blend_all_leg_rows_a025` | `+0.15%` | `+1.46%` | `+3.78%` | `+11.84%` | `+11.54%` | `+12.86%` |
| `alpha_blend_non_leg_rows_a050` | `-3.46%` | `-2.00%` | `-2.64%` | `+8.82%` | `+8.59%` | `+10.55%` |
| `tail_current_zero_adapters` | `+3.95%` | `+3.44%` | `+1.54%` | `+3.22%` | `+2.17%` | `+2.30%` |

bucket-wise 的关键信息很清楚：

1. **hard `all_rot_rows / non_leg_rows / all_leg_rows` 都没有同时解决 `d10_20 / d21_43`。**
2. **`all_leg_rows @ alpha=0.25` 是唯一一个在 `d0_9 / d10_20 / d21_43 / sic0_10 / sic11_21 / sic22_43` 六个必做 bucket 上全部为正的候选。**
3. `non_leg_rows` hard swap 虽然 overall 最强，但在绝对 depth bucket 上仍然明显为负，说明它更像“后段总体拉回”，不是“前 43 step 就稳住”的 intervention。

### 19.5 Secondary Proxy Table

`GeoLocalDeg` 只作 secondary proxy，不参与主排序。

| variant | GeoLocalDeg mean | rel vs tail |
|---|---:|---:|
| `tail_current_control` | `63.303304` | `0.00%` |
| `swap_final_rot_rows_all` | `54.605003` | `+13.74%` |
| `swap_final_rot_rows_non_leg_rows` | `51.219016` | `+19.09%` |
| `swap_final_rot_rows_all_leg_rows` | `60.258275` | `+4.81%` |
| `alpha_blend_all_leg_rows_a025` | `58.269926` | `+7.95%` |
| `alpha_blend_non_leg_rows_a050` | `53.482938` | `+15.51%` |
| `tail_current_zero_adapters` | `62.318008` | `+1.56%` |

proxy 与 primary 的关系：

- `non_leg_rows` hard swap 在 proxy 上也最强；
- 但它并没有 primary bucket-wise 的“全桶为正”性质；
- `all_leg_rows @ alpha=0.25` 虽然 proxy 不如 `non_leg_rows`，但 primary pose table 更平衡。

### 19.6 Final Answer

按问题直接回答：

1. **哪些 final rot row groups 对 pose error 改善贡献最大？**
   - 如果只看 hard swaps，**最大贡献来自 `non_leg_rows [0:192)`**，不是 isolated leg pair。
   - 第二名是 **`all_leg_rows [192:276)`**。
   - `thigh_pair / calf_pair / foot_pair` 都不足以解释改善，甚至整体是微负。

2. **是否存在“只换腿部 rows 就够”的更小 intervention？**
   - **如果限制为 hard swap：没有。**
     - `all_leg_rows` hard swap 虽然 overall 为正，但明显弱于 `non_leg_rows`，且 `d10_20 / d21_43` 仍为负。
   - **如果允许 alpha-blend：有。**
     - `all_leg_rows @ alpha=0.25` 是当前最强的“更小且平衡”的候选：
       - 只动 `84 / 276` rows
       - overall primary mean `+11.86%`
       - `5/5` primary metrics 全正
       - 6 个必做 buckets 全正

3. **adapter 是否仍然只是二阶项？**
   - **是。**
   - `tail_current_zero_adapters` 的 overall primary mean 只有 `+2.61%`；
   - 明显小于：
     - `swap_final_rot_rows_non_leg_rows` 的 `+11.17%`
     - `alpha_blend_all_leg_rows_a025` 的 `+11.86%`
   - 因此 adapter 仍然不是下一步第一优先级。

### 19.7 Intervention Recommendation

本轮 primary metric 是 pose-side metric，不是 `GeoLocalDeg`。基于这轮结果，下一步不要选 `C`。

如果必须在 prompt 里的 `A / B / C` 三选一，我会选：

- **B. 先打特定 leg row groups**

但这里的 “specific leg row groups” 不是 `thigh/calf/foot` 这种 12-row exact pair，而是：

- **broad `all_leg_rows [192:276)` block**
- 并且优先考虑 **soft / blended / low-rank** 形式，而不是 hard full swap

原因只有一句：

- `all_leg_rows @ alpha=0.25` 是这轮唯一同时满足
  - overall primary 明显改善
  - 5 个主指标全正
  - 6 个必做 bucket 全正
  的更小 intervention。

因此这节最后压缩成一句话就是：

- **hard-swap 最大贡献来自 `non_leg_rows`，但下一步真正值得优先训练/干预的是 broad `all_leg_rows [192:276)` 的 soft intervention，而不是 adapter-first，也不是 isolated thigh/calf/foot pair。**

## 20. Replace-Time Interface Co-Adaptation Falsifier

> Last updated: 2026-04-05
> 本节不是继续做 Jacobian / spectral-norm / noise regularization，也不是继续做 donor-only / downstream-only 单边修补。
> 本轮只回答一个 pipeline-level 假设：
> **当前主要问题不是单纯高增益症状，而是 frozen-trunk replace 是否阻止了 coupling interface 与新 reader 协商。**

### 20.1 Code Facts

先写清楚代码事实。

#### 20.1.1 这轮之前 posttrain 能训什么，不能训什么

在这轮改动之前，`train/posttrain.py` 的 active train modes 只有：

- `train_direct_pose`
- `train_lambda_head`
- `train_arm_residual`
- `train_arm_leg_residual`

对应可训练模块也只覆盖：

- `direct_pose_*`
- `lambda_fusion_head`
- `arm_residual_corrector`

**并没有一条 mode 能直接训练 deployed incremental path 的 `shared_encoder -> PASA -> coupling_norm -> motion_head`。**

#### 20.1.2 本轮新增的 minimal plumbing

本轮只加了两类最小 plumbing，没有改 runtime contract，没有改 loss family，没有新开大架构。

1. **新增 `train_incremental_replace`**
   - rollout objective 走 `objective="inc"`
   - 但 rollout carry 仍保持现有 deployed blend path，不另造训练时 runtime 分支

2. **新增 final rot row-mask**
   - 配置键：`incremental_motion_head_row_ranges`
   - 只对 `motion_head` 最后一个 `Linear` 的 `weight / bias` 做 grad mask
   - 这轮实际命中的是 `motion_head.3`
   - `readout_only_allrot`: `[[0,276]]`
   - `readout_only_allleg`: `[[192,276]]`

3. **新增 interface-tail tiny-LR param group**
   - 配置键：
     - `incremental_interface_mode`
     - `incremental_interface_lr_scale`
   - 支持：
     - `off`
     - `tail`
     - `coupling_norm_only`
   - `tail` 实际解析到：
     - `shared_encoder.8`
     - `residual_proj`
     - `_pasa_lnq`
     - `_pasa_q`
     - `_pasa_k`
     - `_pasa_v`
     - `_pasa_o`
     - `_pasa_film`
     - `coupling_norm`
   - 这轮 readout LR 固定 `5e-5`
   - interface LR = `1e-6` = `readout_lr / 50`

#### 20.1.3 这轮实际训练日志里最关键的 code fact

这轮所有 case 的训练日志里，`λ≈0.000`。

这意味着：

- **这一条 replace family 在当前 runtime 下实际上就是 incremental path**
- 所以这轮优先打 `motion_head rot readout + coupling tail` 是对题的
- 不是在拿 direct-only 头部实验偷换问题

新增 runner / 产物：

- runner:
  `tools/run_cp015_tailk7_replace_interface_coadapt_ablation.py`
- summary:
  `debug_output/_tmp_cp015_tailk7_replace_interface_coadapt_ablation_20260405/summary.json`

### 20.2 Experiment Matrix

固定 reference：

- `current_frozen_trunk_replace_control`
- `baseline_replace`

本轮最小训练矩阵：

| variant | train rows | interface subset | interface LR |
|---|---|---|---:|
| `readout_only_allrot` | `[0:276]` | off | `0` |
| `readout_only_allleg` | `[192:276)` | off | `0` |
| `coadapt_allrot_interface` | `[0:276]` | broad tail | `1e-6` |
| `coadapt_allleg_interface` | `[192:276)` | broad tail | `1e-6` |
| `coadapt_allrot_couplingnorm_only` | `[0:276]` | `coupling_norm` only | `1e-6` |

说明：

- 没有打开整个 trunk
- 没有 full co-train
- 没有 adapter-first
- 没有 Jacobian / spectral / noise regularization

### 20.3 Primary Pose Tables

本轮 **primary metric 是 pose-side metric，不是 `GeoLocalDeg`**。

primary：

- `Rot6dLocalL2`
- `Rot6dLocalL2Weighted`
- `GeoDeg`
- `KeyBoneGeoDegMean`
- `KeyBoneGeoLocalDegMean`

secondary only：

- `GeoLocalDeg`

#### 20.3.1 Absolute table

| variant | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean | GeoLocalDeg | primary rel vs current |
|---|---:|---:|---:|---:|---:|---:|---:|
| `current_frozen_trunk_replace_control` | `0.996639` | `0.815900` | `62.188209` | `69.351997` | `73.447931` | `63.303304` | `+0.00%` |
| `readout_only_allrot` | `0.812088` | `0.571066` | `49.685799` | `53.582467` | `56.811756` | `50.603874` | `+22.80%` |
| `readout_only_allleg` | `0.946547` | `0.783498` | `58.811926` | `62.432718` | `66.120880` | `59.878892` | `+6.88%` |
| `coadapt_allrot_interface` | `0.785936` | `0.540126` | `47.833400` | `51.032129` | `54.106550` | `48.718804` | `+26.15%` |
| `coadapt_allleg_interface` | `0.916061` | `0.737045` | `56.774141` | `60.386000` | `63.951574` | `57.803166` | `+10.46%` |
| `coadapt_allrot_couplingnorm_only` | `0.811950` | `0.570919` | `49.676074` | `53.569287` | `56.797785` | `50.593980` | `+22.82%` |
| `baseline_replace` | `0.769215` | `0.544482` | `45.361890` | `44.519712` | `46.769045` | `46.080397` | `+31.05%` |

#### 20.3.2 Direct answers from the primary table

1. **tiny-LR interface co-adaptation 优于对应的 readout-only。**
   - `allrot`: `+26.15%` vs `+22.80%`
   - `allleg`: `+10.46%` vs `+6.88%`

2. **`coupling_norm_only` 不够。**
   - `coadapt_allrot_couplingnorm_only = +22.82%`
   - 几乎等于 `readout_only_allrot = +22.80%`
   - 说明“只开规范化层”基本没有提供额外收益

3. **这轮不支持把 `all_leg_rows [192:276)` 当成比 full `[0:276]` 更好的 first co-adapt target。**
   - `coadapt_allrot_interface = +26.15%`
   - `coadapt_allleg_interface = +10.46%`
   - full `[0:276]` 明显更强

### 20.4 Bucket-Wise Tables

下表是相对 `current_frozen_trunk_replace_control` 的 5 个 primary metrics平均 relative improvement：

| variant | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |
|---|---:|---:|---:|---:|---:|---:|
| `readout_only_allrot` | `+5.16%` | `+6.93%` | `+13.47%` | `+22.58%` | `+21.67%` | `+23.01%` |
| `readout_only_allleg` | `+0.45%` | `+2.82%` | `+4.83%` | `+5.03%` | `+5.28%` | `+6.83%` |
| `coadapt_allrot_interface` | `+5.53%` | `+7.08%` | `+13.61%` | `+27.03%` | `+24.85%` | `+26.33%` |
| `coadapt_allleg_interface` | `+1.60%` | `+3.72%` | `+6.38%` | `+9.53%` | `+9.08%` | `+10.72%` |
| `coadapt_allrot_couplingnorm_only` | `+5.16%` | `+6.93%` | `+13.47%` | `+22.60%` | `+21.69%` | `+23.03%` |
| `baseline_replace` | `-4.27%` | `-3.39%` | `-0.20%` | `+34.62%` | `+33.60%` | `+31.20%` |

bucket-wise 读法：

1. **co-adapt 的增益不是只出现在 overall 均值；在 6 个必做 bucket 上也都压过对应的 readout-only。**
   - `coadapt_allrot_interface > readout_only_allrot`
   - `coadapt_allleg_interface > readout_only_allleg`

2. **最值得注意的额外收益主要出现在 bad cycle windows。**
   - `allrot` 的 co-adapt 相比 readout-only：
     - `sic0_10`: `+27.03%` vs `+22.58%`
     - `sic11_21`: `+24.85%` vs `+21.67%`
     - `sic22_43`: `+26.33%` vs `+23.01%`

3. **`coupling_norm_only` 再次几乎完全贴着 `readout_only_allrot`。**
   - 这进一步支持：
     - 不是“只把 norm 层放开就够”
     - 而是要让 broad coupling tail 做低幅协商

### 20.5 Mandatory Judgement

按题目要求，直接回答判断题：

1. **tiny-LR interface co-adaptation 是否优于 frozen-trunk readout-only？**
   - **是。**
   - 两组对照都成立：
     - `coadapt_allrot_interface > readout_only_allrot`
     - `coadapt_allleg_interface > readout_only_allleg`

2. **`all_leg_rows` 是否比 full `[0:276]` 更适合做 first co-adapt target？**
   - **不是。**
   - 这轮训练结果明确支持 full `[0:276]` 先于 `all_leg_rows [192:276)`。

3. **如果必须二选一：**
   - “读出层继续单边适应”
   - “replace 时允许 interface 双向协商”
   - 哪个更接近根因？
   - **后者。**
   - 因为在完全相同 row target 下，只要允许 tiny-LR interface co-adaptation，pose-primary metrics 就会系统性继续下降。

4. **adapter 是否仍然不是下一步第一优先级？**
   - **是。**
   - 本轮没有任何证据把优先级推回 adapter-first。

### 20.6 Final Recommendation

把本轮压缩成一句话：

- **本轮主问题不是单纯高增益症状，而是 frozen-trunk replace 阻止了 coupling interface 与新 reader 协商。**

再展开成执行建议：

1. **下一步应优先推进 tiny-LR interface co-adaptation during replace。**
2. **第一训练目标优先 full final rot rows `[0:276]` + interface tail 联训。**
3. `all_leg_rows [192:276)` 仍然是有效 smaller target，但这轮不支持把它排到 full `[0:276]` 前面。
4. `coupling_norm_only` 不是足够的替代品。
5. **不是 adapter-first。**

因此本节最后的落地建议是：

- **如果下一轮只做一个 follow-up，就做 `full [0:276] final rot rows + broad interface tail` 的 tiny-LR co-adaptation replace，而不是 adapter-first，也不是只开 `coupling_norm`。**

## 21. Replace-Time Co-Adaptation Saturation Sweep

> Last updated: 2026-04-06
> **本轮主问题不是“co-adapt 是否有效”，而是“当前 replace-time co-adapt 还没调到位，还是已经饱和”。**
> 本轮 primary metric 仍然只看 pose-side metric，不拿 `GeoLocalDeg` 当主排序指标。

### 21.1 Code Facts

#### 21.1.1 当前 `train_incremental_replace` / row-mask / interface param-group 仍然是什么

- `train/posttrain.py` 里的 `train_incremental_replace` 仍然是：
  - rollout objective 用 `objective="inc"`
  - runtime contract 不变
  - 不改 loss family
  - 不新开 adapter / Jacobian / spectral / whitening 路线
- `incremental_motion_head_row_ranges` 仍然只对 `motion_head` 最后一个 `Linear` 做 grad mask。
  - 这轮实际命中仍是 `motion_head.3`
  - full final rot rows = `[[0, 276]]`
  - 未选中的 `weight / bias` 行梯度会被 hook 置零
- interface tiny-LR param-group 仍然是自动创建的 `incremental_interface` group：
  - readout LR 固定 `5e-5`
  - interface LR = `5e-5 * incremental_interface_lr_scale`
- broad tail 仍然解析到：
  - `shared_encoder.8`
  - `residual_proj`
  - `_pasa_lnq`
  - `_pasa_q`
  - `_pasa_k`
  - `_pasa_v`
  - `_pasa_o`
  - `_pasa_film`
  - `coupling_norm`
- 这轮训练日志里仍然看到 `λ≈0.000`。
  - 所以这条 replace family 在当前 runtime 下仍然主要就是 incremental path
  - 继续优先打 deployed incremental path / motion_head rot readout 是对题的

#### 21.1.2 本轮新增 runner / config plumbing 是什么

- 新 runner：
  - `tools/run_cp015_tailk7_replace_interface_coadapt_saturation_sweep.py`
- 新 summary：
  - `debug_output/_tmp_cp015_tailk7_replace_interface_coadapt_saturation_sweep_20260406/summary.json`
- 这轮 runner 明确把决策顺序写死为：
  1. `LR scale`
  2. `longer training`
  3. `interface subset`
- mandatory LR sweep 固定：
  - final rot rows: full `[0:276]`
  - interface subset: broad tail
  - `0.02` 直接复用 2026-04-05 的 best anchor 产物
- 为了支持只有在平台化后才做的 subset probe，这轮给 `train/posttrain.py` 的 `incremental_interface_mode` 额外加了两个最小 mode：
  - `tail_no_sharedenc_lastblock`
  - `tail_no_pasa_stack`
- longer training 没有改训练语义，只是把总步数从 `60 -> 90 -> 120`。

### 21.2 Experiment Matrix

固定 reference：

- `current_frozen_trunk_replace_control`
- `baseline_replace`

本轮实际 sweep matrix：

| variant | row_ranges | interface subset | interface LR scale | epochs | steps/epoch | total steps | reuse |
|---|---|---|---:|---:|---:|---:|---|
| `coadapt_allrot_interface_lrscale_0p01` | `[0:276]` | broad tail | `0.01` | `1` | `60` | `60` | no |
| `coadapt_allrot_interface_lrscale_0p02` | `[0:276]` | broad tail | `0.02` | `1` | `60` | `60` | yes |
| `coadapt_allrot_interface_lrscale_0p04` | `[0:276]` | broad tail | `0.04` | `1` | `60` | `60` | no |
| `coadapt_allrot_interface_bestlr_longer_1p5x` | `[0:276]` | broad tail | best LR = `0.04` | `1` | `90` | `90` | no |
| `coadapt_allrot_interface_bestlr_longer_2x` | `[0:276]` | broad tail | best LR = `0.04` | `1` | `120` | `120` | no |

这轮 **没有** 进入 subset probe：

- `coadapt_allrot_interface_no_sharedenc_lastblock`
- `coadapt_allrot_interface_no_pasa_stack`

原因很直接：

- `0.04` 明确是 best LR
- `1.5x` 继续变好
- `2x` 继续变好
- 所以 replace-stage broad-tail co-adapt 还没有平台化，subset 还不是第一优先级

### 21.3 Primary Pose Tables

再次强调：

- **本轮 primary metric 是 pose-side metric，不是 `GeoLocalDeg`。**

primary：

- `Rot6dLocalL2`
- `Rot6dLocalL2Weighted`
- `GeoDeg`
- `KeyBoneGeoDegMean`
- `KeyBoneGeoLocalDegMean`

secondary only：

- `GeoLocalDeg`

#### 21.3.1 Absolute table

| variant | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean | GeoLocalDeg | primary rel vs current |
|---|---:|---:|---:|---:|---:|---:|---:|
| `current_frozen_trunk_replace_control` | `0.996639` | `0.815900` | `62.188209` | `69.351997` | `73.447931` | `63.303304` | `+0.00%` |
| `coadapt_allrot_interface_lrscale_0p01` | `0.799269` | `0.555740` | `48.783708` | `52.318938` | `55.471092` | `49.685817` | `+24.46%` |
| `coadapt_allrot_interface_lrscale_0p02` | `0.785936` | `0.540126` | `47.833400` | `51.032129` | `54.106550` | `48.718804` | `+26.15%` |
| `coadapt_allrot_interface_lrscale_0p04` | `0.757507` | `0.507829` | `45.827169` | `48.286806` | `51.191792` | `46.676284` | `+29.75%` |
| `coadapt_allrot_interface_bestlr_longer_1p5x` | `0.661031` | `0.425022` | `39.168850` | `39.869962` | `42.183952` | `39.873178` | `+40.73%` |
| `coadapt_allrot_interface_bestlr_longer_2x` | `0.572259` | `0.364053` | `33.475927` | `33.636316` | `35.535482` | `34.063214` | `+49.45%` |
| `baseline_replace` | `0.769215` | `0.544482` | `45.361890` | `44.519712` | `46.769045` | `46.080397` | `+31.05%` |

#### 21.3.2 直接读表

1. **`0.02` 确实偏保守。**
   - 同样都是 `60 steps`：
     - `0.02 = +26.15%`
     - `0.04 = +29.75%`
   - 所以原先剩余 gap：
     - `31.05 - 26.15 = 4.90pp`
   - 只靠把 LR scale 从 `0.02 -> 0.04`，就在相同训练长度下把 gap 缩到：
     - `31.05 - 29.75 = 1.30pp`

2. **`0.01 / 0.02 / 0.04` 里明确 best LR 是 `0.04`。**
   - `0.01 < 0.02 < 0.04`
   - 不是 plateau around `0.02`

3. **更长训练远没饱和。**
   - `0.04 @ 60 steps = +29.75%`
   - `0.04 @ 90 steps = +40.73%`
   - `0.04 @ 120 steps = +49.45%`

4. **replace-stage broad-tail co-adapt 目前不仅没平台化，而且已经显著超过 `baseline_replace`。**
   - `1.5x = +40.73% > +31.05%`
   - `2x = +49.45% > +31.05%`

### 21.4 Bucket-Wise Tables

#### 21.4.1 相对 `current_frozen_trunk_replace_control` 的 primary improvement

| variant | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |
|---|---:|---:|---:|---:|---:|---:|
| `coadapt_allrot_interface_lrscale_0p01` | `+5.33%` | `+7.00%` | `+13.54%` | `+24.71%` | `+23.24%` | `+24.59%` |
| `coadapt_allrot_interface_lrscale_0p02` | `+5.53%` | `+7.08%` | `+13.61%` | `+27.03%` | `+24.85%` | `+26.33%` |
| `coadapt_allrot_interface_lrscale_0p04` | `+5.94%` | `+7.24%` | `+13.68%` | `+32.11%` | `+28.46%` | `+29.79%` |
| `coadapt_allrot_interface_bestlr_longer_1p5x` | `+6.67%` | `+7.90%` | `+14.53%` | `+45.11%` | `+40.05%` | `+39.98%` |
| `coadapt_allrot_interface_bestlr_longer_2x` | `+7.65%` | `+9.35%` | `+16.83%` | `+53.59%` | `+48.49%` | `+48.72%` |
| `baseline_replace` | `-4.27%` | `-3.39%` | `-0.20%` | `+34.62%` | `+33.60%` | `+31.20%` |

#### 21.4.2 相对 `0.02` anchor 的增量

| variant | overall primary delta vs `0.02` | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `coadapt_allrot_interface_lrscale_0p01` | `-1.70pp` | `-0.20pp` | `-0.09pp` | `-0.06pp` | `-2.32pp` | `-1.61pp` | `-1.74pp` |
| `coadapt_allrot_interface_lrscale_0p04` | `+3.59pp` | `+0.41pp` | `+0.16pp` | `+0.07pp` | `+5.08pp` | `+3.62pp` | `+3.46pp` |
| `coadapt_allrot_interface_bestlr_longer_1p5x` | `+14.58pp` | `+1.14pp` | `+0.82pp` | `+0.92pp` | `+18.08pp` | `+15.20pp` | `+13.65pp` |
| `coadapt_allrot_interface_bestlr_longer_2x` | `+23.30pp` | `+2.12pp` | `+2.27pp` | `+3.23pp` | `+26.56pp` | `+23.64pp` | `+22.39pp` |

bucket-wise 读法：

1. **`0.04` 相对 `0.02` 的额外收益，不只是 overall 均值更好。**
   - 它最明显的增量仍然集中在 cycle buckets：
     - `sic0_10: +5.08pp`
     - `sic11_21: +3.62pp`
     - `sic22_43: +3.46pp`
   - depth buckets 的增量明显更小

2. **更长训练进一步放大了同一类窗口收益。**
   - `2x` 相对 `0.02`：
     - `sic11_21: +23.64pp`
     - `sic22_43: +22.39pp`
   - 说明额外 headroom 仍然主要落在 bad cycle windows，而不是早已转移到别处

3. **`baseline_replace` 依然呈现 old pattern：**
   - early depth 并不占优
   - 但 cycle buckets 比 current control 强
   - 而这次 `0.04 + longer` 已经连这些 cycle buckets 也一起超过去了

### 21.5 Mandatory Judgement

按题目要求，直接回答 6 个判断题：

1. **当前 broad-tail co-adapt 的 `0.02` 是否只是 LR 太保守？**
   - **是。**
   - 证据：
     - `0.04 @ 60 steps = +29.75%`
     - `0.02 @ 60 steps = +26.15%`
   - 所以这轮首先支持的是：
     - 当前 best co-adapt 之前确实太保守、还没调到位

2. **`0.01 / 0.02 / 0.04` 中哪个最优？提升主要发生在 overall，还是主要集中在 `sic11_21 / sic22_43`？**
   - **最优是 `0.04`。**
   - 提升既体现在 overall，也明显集中在 bad cycle windows。
   - 相对 `0.02`，`0.04` 在：
     - `sic11_21`
     - `sic22_43`
   - 上都继续明显变好

3. **更长训练（`1.5x / 2x`）是否还能继续改善 pose-side primary metrics，还是已经基本饱和？**
   - **还能继续改善，且这轮完全没有看到饱和。**
   - `1.5x` 比 `60 steps` 更好
   - `2x` 又比 `1.5x` 更好

4. **如果 LR / epochs 已经平台化，broad tail 里更不可或缺的是 `shared_encoder` 最后一个 block，还是 `_pasa_*` stack？**
   - **这轮还不能回答。**
   - 因为 LR / epochs **没有** 平台化
   - 所以 subset probe 按预设没有执行
   - 当前还不该把优先级提前到 `shared_encoder last block vs PASA stack` 的裁决

5. **基于这轮结果，是否仍然不应该优先跳回 basetrain / 70a？**
   - **是，仍然不应该。**
   - 因为 replace-stage broad-tail co-adapt 还有很大 headroom
   - 而且在不回 upstream 的前提下，`1.5x / 2x` 已经明显超过 `baseline_replace`

6. **adapter 是否仍然不是下一步第一优先级？**
   - **是。**
   - 这轮最强信号来自：
     - `full [0:276] final rot rows + broad interface tail`
     - `interface_lr_scale`
     - `longer training`
   - 不是 adapter-first

### 21.6 Final Recommendation

把本轮压缩成一句话：

- **当前 replace-time broad-tail co-adaptation 还没调到位，远没有饱和。**

明确建议：

1. **继续优先推进 `full [0:276] final rot rows + broad interface tail` 的 replace-time co-adaptation 调优。**
2. **优先调 `interface_lr_scale`。**
   - 这轮已确认：`0.04 > 0.02 > 0.01`
3. **然后继续看 longer training。**
   - 这轮已确认：`60 -> 90 -> 120 steps` 还在持续改善 pose-side primary metrics
4. **只有当 replace-stage co-adapt 明确平台化，才考虑回 basetrain / 70a 做 interface robustness。**
5. **不是 adapter-first。**

## 22. Replace-Time Co-Adaptation Longer Push And Donor Integrity

> Last updated: 2026-04-06
> **本轮主问题不是“co-adapt 是否有效”，而是“best-LR replace-stage longer training 是否还在继续改善，以及 donor integrity 是否仍然安全”。**
> 本轮 primary metric 仍然只看 pose-side metric，不拿 `GeoLocalDeg` 当主排序指标。

### 22.1 Code Facts

#### 22.1.1 当前 `train_incremental_replace` / row-mask / interface param-group 仍然是什么

- `train/posttrain.py` 里的 `train_incremental_replace` 仍然是 deployed incremental path：
  - objective 仍然是 `inc`
  - runtime contract 不变
  - 不改 loss family
  - 不新开 adapter / Jacobian / spectral / whitening / basetrain 回路
- `incremental_motion_head_row_ranges` 仍然只对 `motion_head` 最后一个 `Linear` 做 grad mask。
  - 这轮锁定 full final rot rows：`[[0, 276]]`
  - 未选中的 `weight / bias` 行梯度继续由 hook 置零
- broad interface tail 仍然锁定为：
  - `shared_encoder.8`
  - `residual_proj`
  - `_pasa_lnq`
  - `_pasa_q`
  - `_pasa_k`
  - `_pasa_v`
  - `_pasa_o`
  - `_pasa_film`
  - `coupling_norm`
- readout LR 仍固定 `5e-5`
- interface LR scale 仍固定 `0.04`
  - 所以 interface LR 仍然是 `2e-6`
- 训练日志里仍然看到 `λ≈0.000`
  - 所以当前 replace family 在 runtime 下仍然主要就是 incremental path
  - 继续优先 deployed incremental path / motion_head rot readout 是对题的

#### 22.1.2 本轮 longer-push / integrity-monitor runner 是什么

- 新 runner：
  `tools/run_cp015_tailk7_replace_interface_coadapt_longer_push.py`
- 新 summary：
  `debug_output/_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/summary.json`
- runner 的决策顺序是固定的：
  1. 复用 `0.04 @ 60 / 90 / 120`
  2. 必做 `0.04 @ 180`
  3. 只有在 `180` 继续改善 pose-side primary metrics 且 donor integrity 平滑安全时，才追加 `0.04 @ 240`
- donor integrity monitor 固定只看 broad-tail interface params，相对原始 `70a` donor 输出：
  - overall:
    - `max_abs_diff`
    - `mean_abs_diff`
    - `rms_diff`
    - `rel_rms_vs_base`
  - per-prefix:
    - 同样 4 个 drift 指标
  - trajectory:
    - `60 -> 90 -> 120 -> 180 -> 240`

### 22.2 Experiment Matrix

固定 anchors：

- `current_frozen_trunk_replace_control`
- `baseline_replace`
- `coadapt_allrot_interface_lrscale_0p04`
- `coadapt_allrot_interface_bestlr_longer_1p5x`
- `coadapt_allrot_interface_bestlr_longer_2x`

本轮实际矩阵：

| variant | row_ranges | interface subset | interface LR scale | total steps | reuse |
|---|---|---|---:|---:|---|
| `coadapt_allrot_interface_lrscale_0p04` | `[0:276]` | broad tail | `0.04` | `60` | yes |
| `coadapt_allrot_interface_bestlr_longer_1p5x` | `[0:276]` | broad tail | `0.04` | `90` | yes |
| `coadapt_allrot_interface_bestlr_longer_2x` | `[0:276]` | broad tail | `0.04` | `120` | yes |
| `coadapt_allrot_interface_bestlr_longer_3x` | `[0:276]` | broad tail | `0.04` | `180` | no |
| `coadapt_allrot_interface_bestlr_longer_4x` | `[0:276]` | broad tail | `0.04` | `240` | no |

这轮没有做：

- subset probe
- basetrain / `70a` robustness 回跳
- proximity / EWC-style constraint
- adapter-first

原因很直接：

- `3x` 继续改善
- `60 -> 90 -> 120 -> 180` donor integrity trajectory 平滑小幅增长
- 因此 `4x` gate 被放行，且 `4x` 也确实继续改善

### 22.3 Primary Pose Tables

再次强调：

- **本轮 primary metric 是 pose-side metric，不是 `GeoLocalDeg`。**

primary：

- `Rot6dLocalL2`
- `Rot6dLocalL2Weighted`
- `GeoDeg`
- `KeyBoneGeoDegMean`
- `KeyBoneGeoLocalDegMean`

secondary only：

- `GeoLocalDeg`

#### 22.3.1 Absolute table

| variant | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean | GeoLocalDeg | primary rel vs current |
|---|---:|---:|---:|---:|---:|---:|---:|
| `current_frozen_trunk_replace_control` | `0.996639` | `0.815900` | `62.188209` | `69.351997` | `73.447931` | `63.303304` | `+0.00%` |
| `baseline_replace` | `0.769215` | `0.544482` | `45.361890` | `44.519712` | `46.769045` | `46.080397` | `+31.05%` |
| `coadapt_allrot_interface_lrscale_0p04` | `0.757507` | `0.507829` | `45.827169` | `48.286806` | `51.191792` | `46.676284` | `+29.75%` |
| `coadapt_allrot_interface_bestlr_longer_1p5x` | `0.661031` | `0.425022` | `39.168850` | `39.869962` | `42.183952` | `39.873178` | `+40.73%` |
| `coadapt_allrot_interface_bestlr_longer_2x` | `0.572259` | `0.364053` | `33.475927` | `33.636316` | `35.535482` | `34.063214` | `+49.45%` |
| `coadapt_allrot_interface_bestlr_longer_3x` | `0.458169` | `0.285701` | `25.932796` | `26.060334` | `27.416103` | `26.351756` | `+60.48%` |
| `coadapt_allrot_interface_bestlr_longer_4x` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` | `21.761820` | `+65.89%` |

#### 22.3.2 `3x / 4x` 相对 `2x` 的新增收益

| variant | overall primary delta vs `2x` | Rot6dLocalL2 Δ | Rot6dLocalL2Weighted Δ | GeoDeg Δ | KeyBoneGeoDegMean Δ | KeyBoneGeoLocalDegMean Δ |
|---|---:|---:|---:|---:|---:|---:|
| `3x` | `+11.03pp` | `-0.114090` | `-0.078352` | `-7.543131` | `-7.575982` | `-8.119379` |
| `4x` | `+16.44pp` | `-0.183396` | `-0.111592` | `-12.032559` | `-10.602350` | `-11.364306` |

直接读法：

1. **`0.04 @ 3x` 没有平台化，仍在继续明显改善 pose-side primary metrics。**
2. **`4x` 也继续改善。**
   - 相对 `3x`，`4x` 的 absolute overall primary 仍然从 `+60.48%` 继续升到 `+65.89%`
3. **到 `4x` 为止，best-LR longer training 还没有出现“已经吃完 headroom”的证据。**

### 22.4 Bucket-Wise Tables

#### 22.4.1 Absolute table

下表是相对 `current_frozen_trunk_replace_control` 的 5 个 primary metrics平均 relative improvement：

| variant | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_replace` | `-4.27%` | `-3.39%` | `-0.20%` | `+34.62%` | `+33.60%` | `+31.20%` |
| `0.04 @ 60` | `+5.94%` | `+7.24%` | `+13.68%` | `+32.11%` | `+28.46%` | `+29.79%` |
| `0.04 @ 90` | `+6.67%` | `+7.90%` | `+14.53%` | `+45.11%` | `+40.05%` | `+39.98%` |
| `0.04 @ 120` | `+7.65%` | `+9.35%` | `+16.83%` | `+53.59%` | `+48.49%` | `+48.72%` |
| `0.04 @ 180` | `+8.91%` | `+11.48%` | `+19.90%` | `+64.35%` | `+58.91%` | `+59.53%` |
| `0.04 @ 240` | `+10.15%` | `+12.28%` | `+20.49%` | `+69.98%` | `+64.15%` | `+63.65%` |

#### 22.4.2 `3x / 4x` 相对 `2x` 的新增 bucket 收益

| variant | overall primary delta vs `2x` | d0_9 | d10_20 | d21_43 | sic0_10 | sic11_21 | sic22_43 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `3x` | `+11.03pp` | `+1.26pp` | `+2.13pp` | `+3.06pp` | `+10.76pp` | `+10.42pp` | `+10.81pp` |
| `4x` | `+16.44pp` | `+2.50pp` | `+2.93pp` | `+3.66pp` | `+16.39pp` | `+15.67pp` | `+14.93pp` |

bucket-wise 读法：

1. **`3x` 的新增收益主要仍然集中在 `sic11_21 / sic22_43`，没有改写主导位置。**
   - `sic11_21: +10.42pp`
   - `sic22_43: +10.81pp`
   - 同时 depth buckets 虽然继续为正，但明显更小：
     - `d10_20: +2.13pp`
     - `d21_43: +3.06pp`
2. **`4x` 继续放大同一模式。**
   - `sic0_10 / sic11_21 / sic22_43` 的增量仍然显著大于 depth buckets
3. 因此当前 broad-tail co-adapt 的 longer training **仍然主要在修 mid-cycle negotiation mismatch，不是主要在修 early-depth pose quality。**

### 22.5 Donor Integrity Tables

#### 22.5.1 Overall drift trajectory

对比对象固定为原始 `70a` donor，只看 broad-tail interface params。

| variant | total_steps | max_abs_diff | mean_abs_diff | rms_diff | rel_rms_vs_base | `shared_encoder.8` max_abs_diff |
|---|---:|---:|---:|---:|---:|---:|
| `0.04 @ 60` | `60` | `1.37e-4` | `3.16e-5` | `5.09e-5` | `0.13%` | `1.24e-4` |
| `0.04 @ 90` | `90` | `2.15e-4` | `4.25e-5` | `6.93e-5` | `0.18%` | `1.87e-4` |
| `0.04 @ 120` | `120` | `3.05e-4` | `5.18e-5` | `8.54e-5` | `0.22%` | `2.46e-4` |
| `0.04 @ 180` | `180` | `4.69e-4` | `6.77e-5` | `1.13e-4` | `0.29%` | `3.67e-4` |
| `0.04 @ 240` | `240` | `6.13e-4` | `8.09e-5` | `1.37e-4` | `0.35%` | `4.98e-4` |

trajectory 判读：

- `60 -> 90 -> 120 -> 180 -> 240` 是**平滑小幅增长**，没有出现加速
- `rel_rms_vs_base` 的 per-step slope：
  - `90 -> 120`: `1.38e-5 / step`
  - `120 -> 180`: `1.19e-5 / step`
  - `180 -> 240`: `1.03e-5 / step`
- 也就是说 trajectory 不仅没加速，反而在 per-step slope 上略微放缓

#### 22.5.2 Per-prefix table

##### `0.04 @ 180`

| prefix | max_abs_diff | mean_abs_diff | rms_diff | rel_rms_vs_base |
|---|---:|---:|---:|---:|
| `shared_encoder.8` | `3.67e-4` | `1.53e-5` | `4.89e-5` | `0.13%` |
| `residual_proj` | `4.69e-4` | `1.21e-4` | `1.53e-4` | `0.58%` |
| `_pasa_lnq` | `0` | `0` | `0` | `0%` |
| `_pasa_q` | `0` | `0` | `0` | `0%` |
| `_pasa_k` | `0` | `0` | `0` | `0%` |
| `_pasa_v` | `4.24e-4` | `1.59e-4` | `1.82e-4` | `0.70%` |
| `_pasa_o` | `4.16e-4` | `1.39e-4` | `1.59e-4` | `0.61%` |
| `_pasa_film` | `3.68e-4` | `1.22e-4` | `1.42e-4` | `0.26%` |
| `coupling_norm` | `3.73e-4` | `1.35e-4` | `1.54e-4` | `0.02%` |

##### `0.04 @ 240`

| prefix | max_abs_diff | mean_abs_diff | rms_diff | rel_rms_vs_base |
|---|---:|---:|---:|---:|
| `shared_encoder.8` | `4.98e-4` | `1.85e-5` | `6.05e-5` | `0.17%` |
| `residual_proj` | `6.13e-4` | `1.43e-4` | `1.83e-4` | `0.69%` |
| `_pasa_lnq` | `0` | `0` | `0` | `0%` |
| `_pasa_q` | `0` | `0` | `0` | `0%` |
| `_pasa_k` | `0` | `0` | `0` | `0%` |
| `_pasa_v` | `5.73e-4` | `1.93e-4` | `2.24e-4` | `0.86%` |
| `_pasa_o` | `5.54e-4` | `1.65e-4` | `1.92e-4` | `0.74%` |
| `_pasa_film` | `4.79e-4` | `1.43e-4` | `1.69e-4` | `0.31%` |
| `coupling_norm` | `4.66e-4` | `1.59e-4` | `1.85e-4` | `0.03%` |

per-prefix 读法：

1. **`shared_encoder.8` 到 `180` 为止仍然很小。**
   - `max_abs_diff = 3.67e-4`
2. **到 `240` 也仍然没有出现“明显破坏 donor trunk / interface 质量”的证据。**
   - `shared_encoder.8 max_abs_diff = 4.98e-4`
   - 这已经开始接近需要持续盯的量级，但仍然没有出现明显断点或加速
3. `_pasa_lnq / _pasa_q / _pasa_k` 仍然基本 `0 drift`
4. 当前最大漂移仍主要落在：
   - `residual_proj`
   - `_pasa_v`
   - `_pasa_o`
   - `_pasa_film`
   但总体量级仍然小，且 trajectory 平滑

### 22.6 Mandatory Judgement

按题目要求，直接回答 8 个判断题：

1. **`0.04 @ 3x` 是否继续改善 pose-side primary metrics，还是开始平台化？**
   - **继续改善。**
   - 相对 `2x`：
     - overall primary `+11.03pp`
     - 5 个 primary metrics 全部继续下降

2. **如果 `3x` 继续改善，改善主要仍然集中在 `sic11_21 / sic22_43`，还是已经扩散到 depth buckets？**
   - **主要仍然集中在 `sic11_21 / sic22_43`。**
   - depth buckets 也继续为正，但明显更小

3. **donor integrity 从 `60 -> 90 -> 120 -> 180` 的 drift trajectory 是平滑小幅增长，还是开始出现加速？**
   - **平滑小幅增长，没有出现加速。**
   - `rel_rms_vs_base` 的 per-step slope 还略微放缓

4. **`shared_encoder.8` 的 drift 是否仍然很小，还是已经开始接近需要警惕的量级？**
   - **到 `180` 仍然很小。**
   - 到 `240` 仍未越过明确警戒线，但已经值得继续盯住

5. **是否值得继续做 `4x`？**
   - **值得。**
   - 这轮 gate 放行后实际做了 `4x`
   - 而且 `4x` 也继续改善 pose-side primary metrics

6. **基于这轮结果，下一步是否仍然应该优先继续 replace-stage longer training，而不是回 basetrain / 70a？**
   - **是。**
   - 当前证据仍支持继续优先 replace-stage longer training
   - 还不支持回 basetrain / `70a`

7. **proximity / EWC-style constraint 是否已经需要进入下一优先级，还是还不该上？**
   - **还不该上。**
   - 只有当 longer training 明确平台化，或者 donor integrity 明显恶化，才需要提优先级

8. **adapter 是否仍然不是下一步第一优先级？**
   - **是。**
   - 这轮最强信号仍然来自：
     - `full [0:276] final rot rows + broad interface tail`
     - `interface_lr_scale = 0.04`
     - longer training
   - **不是 adapter-first**

### 22.7 Final Recommendation

把本轮压缩成一句话：

- **best-LR replace-stage longer training 还在继续明显改善，而 donor integrity 到 `240` 为止仍然平滑安全；因此下一步仍应优先继续 `full [0:276] final rot rows + broad interface tail`、`interface_lr_scale = 0.04` 的 longer training，不是 adapter-first。**

明确写结论：

1. **本轮主问题不是“co-adapt 是否有效”，而是“best-LR replace-stage longer training 是否还在继续改善，以及 donor integrity 是否仍然安全”。**
2. **本轮 primary metric 是 pose-side metric，不是 `GeoLocalDeg`。**
3. 当前结果支持继续优先推进：
   - `full [0:276] final rot rows + broad interface tail`
   - `interface_lr_scale = 0.04`
   - longer training
4. **只有当 longer training 明确平台化，或者 donor integrity 明显恶化，才考虑：**
   - proximity / EWC-style constraint
   - 或回 basetrain / `70a` 做 interface robustness
5. **不是 adapter-first。**

## 23. Direct Path Recovery For Baseline Replacement

这一轮不再回答：

- co-adapt 是否有效
- longer training 还能不能继续涨

这些都已经成立。

这一轮唯一主问题是：

- **在保留当前 best pose-side co-adapt 收益的前提下，能不能把 direct path 拉回到 `baseline_replace` 水平，从而让 co-adapt candidate 真正具备 baseline replacement 资格。**

本节新增 hard gate：

- **`DirectGeoLocalDeg` 必须作为 baseline replacement hard gate 报告。**
- 本轮操作性判据写死为：
  `DirectGeoLocalDeg <= baseline_replace + 0.01`

本轮新增脚本与产物：

- bridge runner:
  `tools/run_cp015_tailk7_replace_direct_recovery_bridge.py`
- gate compare:
  `tools/compare_cp015_tailk7_replace_baseline_replacement_gate.py`
- summary:
  `debug_output/_tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406/summary.json`
- gate summary:
  `debug_output/_tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406/baseline_replacement_gate.json`

### 23.1 Code Facts

先写本轮真正相关的 code facts，不回旧方向：

1. **direct head 实际读的是 `cond`，不是 `h_final`。**
   - `train/models.py` 中 direct head 的输入路由明确写成：
     - 默认 `direct_pose_feat_source = 'cond'`
     - 只有显式切到 `hidden / hidden_pre / cond+hidden / cond+hidden_pre` 才会改输入
   - 当前 `coadapt_4x` 配置文件里也明确是：
     - `direct_pose_feat_source = cond`

2. **`train_incremental_replace` 实际 trainable set 不包含 `direct_pose_*`。**
   - `train/posttrain.py` 的流程是：
     - `_freeze_all(model)`
     - `_unfreeze_incremental_replace(...)`
   - `_unfreeze_incremental_replace(...)` 只会解冻：
     - `motion_head` 最后一个 `Linear`
     - 以及 `incremental_interface_mode=tail` 下的 broad interface tail
   - 当前 `coadapt_4x` 对应配置是：
     - `incremental_motion_head_row_ranges = [[0, 276]]`
     - `incremental_interface_mode = tail`
     - `incremental_interface_lr_scale = 0.04`
   - **结论：当前 co-adapt 线根本没有训练 direct path。**

3. **ckpt 级别 `direct_pose_*` identical facts 直接成立。**
   - `coadapt60 vs 120`: `25/25` common direct tensors，`diff_key_count = 0`
   - `coadapt60 vs 180`: `25/25` common direct tensors，`diff_key_count = 0`
   - `coadapt60 vs 240`: `25/25` common direct tensors，`diff_key_count = 0`
   - `coadapt60 vs warmstart`: `25/25` common direct tensors，`diff_key_count = 0`
   - 所以：
     - **`0.213743` 不是 co-adapt 把 direct 训坏了**
     - **而是 co-adapt 根本没训 direct**

4. **`baseline_replace` 的 direct path 则确实训练过。**
   - `baseline_replace vs its warmstart`：
     - `25/25` common direct tensors
     - `diff_key_count = 20`
   - 所以 baseline 的更好 direct，不是 warmstart 继承假象，而是 direct ownership 真被校准过。

5. **current control 与 coadapt4x 的 direct architecture 不完全同构。**
   - `current control` 有 `39` 个 `direct_pose_*` tensors
   - `coadapt_4x` 有 `25` 个
   - 两者只共享 `21` 个兼容 direct tensors
   - 因此 `coadapt_4x_plus_control_directpose_swap` 只能做：
     - **compatible shared direct tensors transplant**
     - 不是 full exact transplant

### 23.2 Transplant Experiment Matrix

anchors：

- `current_frozen_trunk_replace_control`
- `baseline_replace`
- `coadapt_allrot_interface_bestlr_longer_4x`

本轮实际执行矩阵：

| candidate | pose source | direct source | operation | note |
|---|---|---|---|---|
| `coadapt_4x_plus_baseline_directpose_swap` | `coadapt_4x` | `baseline_replace` | exact transplant | `25/25` direct tensors 全量替换 |
| `coadapt_4x_plus_control_directpose_swap` | `coadapt_4x` | `current control` | compatible transplant | 只替换 `21` 个 shared direct tensors |
| `baseline_plus_coadapt4x_directpose_swap` | `baseline_replace` | `coadapt_4x` | exact transplant | 反向 probe |
| `coadapt_4x_directonly_calibration_short` | `coadapt_4x` warmstart | self | direct-only train | `60` steps；只训 `direct_pose_*` |

条件追加的 slightly longer calibration 没有执行，因为：

- short direct-only calibration 虽然改善了 direct
- **但仍然没有过 baseline hard gate**

### 23.3 Final Intuitive Compare

| candidate | DirectGeoLocalDeg | GeoLocalDeg mean | BlendGeoLocalDeg mean | lambda_present | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| current_frozen_trunk_replace_control | 0.177592 | 76.058629 | 76.058629 | false | 0.996639 | 0.815900 | 62.188209 | 69.351997 | 73.447931 |
| baseline_replace | 0.153746 | 54.310389 | 54.310389 | false | 0.769215 | 0.544482 | 45.361890 | 44.519712 | 46.769045 |
| coadapt_allrot_interface_bestlr_longer_4x | 0.213743 | 25.731748 | 25.731748 | false | 0.388863 | 0.252461 | 21.443368 | 23.033965 | 24.171176 |
| coadapt_4x_plus_baseline_directpose_swap | 0.150419 | 25.731748 | 25.731748 | false | 0.388863 | 0.252461 | 21.443368 | 23.033965 | 24.171176 |
| coadapt_4x_plus_control_directpose_swap | 0.188263 | 25.731748 | 25.731748 | false | 0.388863 | 0.252461 | 21.443368 | 23.033965 | 24.171176 |
| baseline_plus_coadapt4x_directpose_swap | 0.215619 | 54.311840 | 54.311840 | false | 0.769241 | 0.544503 | 45.363390 | 44.522049 | 46.771412 |
| coadapt_4x_directonly_calibration_short | 0.191253 | 25.731748 | 25.731748 | false | 0.388863 | 0.252461 | 21.443368 | 23.033965 | 24.171176 |

直接读这张表，结论已经很硬：

- **`coadapt_4x + baseline direct_pose_*` 把 direct 从 `0.213743` 修到 `0.150419`，而 pose-side primary metrics 一点没动。**
- 反向 swap 则把 baseline 的 direct 从 `0.153746` 拉坏到 `0.215619`。
- `lambda_present` 全部仍然是 `false`，`BlendGeoLocalDeg mean == GeoLocalDeg mean` 继续成立。
- 所以这轮修复 direct 的最小解释，不需要先动 lambda path。

### 23.4 Pose-Side Primary Table

pose-side primary metrics 仍然是主排序指标，因此单独列一张表：

| candidate | pose source | direct source | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean | pose better than baseline? | pose preserved vs coadapt_4x? |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| baseline_replace | baseline | baseline | 0.769215 | 0.544482 | 45.361890 | 44.519712 | 46.769045 | ref | no |
| coadapt_allrot_interface_bestlr_longer_4x | coadapt_4x | coadapt_4x | 0.388863 | 0.252461 | 21.443368 | 23.033965 | 24.171176 | yes | ref |
| coadapt_4x_plus_baseline_directpose_swap | coadapt_4x | baseline | 0.388863 | 0.252461 | 21.443368 | 23.033965 | 24.171176 | yes | yes |
| coadapt_4x_plus_control_directpose_swap | coadapt_4x | current control | 0.388863 | 0.252461 | 21.443368 | 23.033965 | 24.171176 | yes | yes |
| coadapt_4x_directonly_calibration_short | coadapt_4x | short calib | 0.388863 | 0.252461 | 21.443368 | 23.033965 | 24.171176 | yes | yes |

本轮 pose-side 结论非常干净：

- **所有 direct recovery probe 都没有伤到当前 `4x` 的 pose-side primary gains。**
- 换言之，这轮确实不是 pose-side 收益问题，而是 direct ownership / calibration 问题。

### 23.5 Direct Swap Table

| candidate | direct source | pose source | DirectGeoLocalDeg | delta vs baseline_replace | delta vs coadapt_4x |
|---|---|---|---:|---:|---:|
| coadapt_4x_plus_baseline_directpose_swap | baseline_replace | coadapt_allrot_interface_bestlr_longer_4x | 0.150419 | -0.003326 | -0.063324 |
| coadapt_4x_plus_control_directpose_swap | current_frozen_trunk_replace_control | coadapt_allrot_interface_bestlr_longer_4x | 0.188263 | +0.034518 | -0.025480 |
| baseline_plus_coadapt4x_directpose_swap | coadapt_allrot_interface_bestlr_longer_4x | baseline_replace | 0.215619 | +0.061873 | +0.001875 |

这张表直接回答 ownership 问题：

1. **baseline direct head 一换进去，coadapt direct 立刻修好。**
2. **coadapt direct head 一换到 baseline 上，baseline direct 立刻坏掉。**
3. `current control` 的兼容子集 transplant 只能部分改善，修不到 baseline。

因此这轮最被支持的解释是：

- **direct regression 主要就是 `direct_pose_*` stale / untrained / ownership 未校准。**

### 23.6 Baseline Replacement Gate

| candidate | pose better than baseline? | direct non-regression vs baseline? | replace baseline eligible? |
|---|---|---|---|
| current_frozen_trunk_replace_control | no | no | no |
| coadapt_allrot_interface_bestlr_longer_4x | yes | no | no |
| coadapt_4x_plus_baseline_directpose_swap | yes | yes | yes |
| coadapt_4x_plus_control_directpose_swap | yes | no | no |
| coadapt_4x_directonly_calibration_short | yes | no | no |

这里要把一个关键区分写死：

1. **raw `coadapt_4x` 本身，仍然不能替 baseline 上位。**
   - pose-side 明显更好
   - 但 direct hard gate 失败

2. **存在一个最小 recovery candidate 可以替 baseline 上位：**
   - `coadapt_4x_plus_baseline_directpose_swap`

3. **`60` step short direct-only calibration 还不够。**
   - `DirectGeoLocalDeg: 0.213743 -> 0.191253`
   - 有改善
   - 但仍显著差于 baseline `0.153746`
   - 因此没有继续跑 slightly longer direct-only calibration

### 23.7 Mandatory Judgement

按本轮要求，直接回答 8 个判断题：

1. **`coadapt_4x + baseline direct_pose_*` 能否显著修复 `DirectGeoLocalDeg`？**
   - **能。**
   - `0.213743 -> 0.150419`
   - 不只是接近 baseline，甚至略优于 baseline `0.153746`

2. **如果能修复，pose-side primary metrics 是否基本保持不变？**
   - **是。**
   - 5 个 primary metrics 与 `coadapt_4x` 完全相同

3. **direct regression 是否主要就是 `direct_pose_*` stale / untrained 导致的？**
   - **是，当前证据强支持。**
   - 正向 swap 修好
   - 反向 swap 拉坏
   - coadapt `direct_pose_*` 又与 warmstart 完全 identical

4. **是否需要 contact-plan / lambda path 才能一起修 direct？**
   - **不需要，至少不是第一 blocker。**
   - 当前 `lambda_present = false`
   - 但 baseline direct transplant 已经能单独修回 direct

5. **是否存在一个 candidate，既明显优于 baseline 的 pose-side primary metrics，又不再在 `DirectGeoLocalDeg` 上落后？**
   - **存在。**
   - `coadapt_4x_plus_baseline_directpose_swap`

6. **如果没有这样的 candidate，当前 co-adapt 是否仍然不能替换 baseline？**
   - 这一题本轮变成：
   - **raw `coadapt_4x` 仍然不能替换 baseline；但最小 direct recovery 后，存在可以替换 baseline 的 candidate。**

7. **下一步优先级是否应该是 direct-only calibration，而不是回 basetrain / 70a？**
   - **是。**
   - 但要补一句：
   - 当前 `60` step short direct-only calibration 还不够
   - 所以下一步仍应继续 direct ownership / calibration 路线，而不是跳回 basetrain / `70a`

8. **adapter 是否仍然不是下一步第一优先级？**
   - **是。**
   - 这轮已经证明最小 direct recovery path 存在
   - 不是 adapter-first

### 23.8 Final Recommendation

把本轮压缩成一句话：

- **当前问题不是“co-adapt pose-side 有没有收益”，这个已经成立；当前问题是“它能不能在不丢掉 pose-side 收益的前提下，通过最小 direct-path recovery 真正替换 baseline”。本轮答案是：可以通过最小 `direct_pose_* ownership` recovery 达到，但 raw `coadapt_4x` 本身还没过 gate。**

明确写最终建议：

1. **`DirectGeoLocalDeg` 这轮必须作为 baseline replacement 的 hard gate 报告。**
2. **最小 ownership path 已被 exact transplant 直接证实：**
   - `coadapt_4x + baseline direct_pose_*`
   - 在不损失任何 pose-side primary gains 的前提下
   - 直接把 direct 拉回到 baseline 水平之上
3. **因此当前最优先的后续方向仍然是 direct-only calibration / ownership path，不是回 basetrain / `70a`，也不是 adapter-first。**
4. **但 short direct-only calibration (`60` steps) 还不够，说明“方向对”已经成立，“最小训练配方已足够”还没有成立。**
5. **只有当 direct-pose transplant / direct-only calibration 这条最小 recovery 路径整体失败时，才允许回更上游怀疑：**
   - contact-plan ownership
   - lambda path
   - basetrain / `70a` robustness
6. **不是 adapter-first。**

## 24. Direct-Only Calibration After Direct-Pose Ownership Proof

这一轮不再回旧方向，也不再重证已经成立的事实。

本节唯一回答：

- **在已经证明 static baseline direct transplant 可以修好 direct 之后，能不能只靠最小 direct-only calibration，把 `coadapt_4x` 自己的 `direct_pose_*` 训回 baseline replacement gate 以内。**

本节 hard gate 继续固定为：

- `DirectGeoLocalDeg <= baseline_replace + 0.01`
- 也就是：
  `DirectGeoLocalDeg <= 0.163746`

### 24.1 Code Facts

这一轮 relevant code facts 只保留 direct-only train lane：

1. **direct-only trainable set 实际就是 `train_direct_pose` lane。**
   - `train/posttrain.py`
     - `_resolve_train_mode(...)` 只允许单一目标
     - 本轮配置固定为：
       - `train_direct_pose = true`
       - `train_incremental_replace = false`
       - `train_lambda_head = false`
       - `train_arm_residual = false`
       - `train_arm_leg_residual = false`
   - 然后执行：
     - `_freeze_all(model)`
     - `_unfreeze_direct_pose(...)`

2. **四个 calibration case 的 runtime log 都证明：实际只动了 `direct_pose_*`。**
   - `coadapt_4x_directonly_calibration_short`
   - `coadapt_4x_directonly_calibration_120`
   - `coadapt_4x_directonly_calibration_180`
   - `coadapt_4x_directonly_calibration_240`
   - 四个 case 的 log 都是：
     - `mode=train_direct_pose`
     - `trainable=20 params`
   - trainable sample 全都落在 `direct_pose_*`：
     - `direct_pose_head.0.*`
     - `direct_pose_head.3.*`
     - `direct_pose_out_leg.*`
     - `direct_pose_out_arm.*`
     - 以及同组 direct readout tensors

3. **因此 donor trunk / interface / motion_head / lambda 在这轮都没有动。**
   - `incremental_interface_mode = off`
   - `incremental_interface_lr_scale = 0.0`
   - runtime contract 没变
   - loss family 没变
   - `lambda_present` 在所有 candidate 上仍然是 `false`

结论简写：

- **这轮 direct-only calibration 是真实的 `direct_pose_* ownership` 微调，不是任何 trunk / adapter / lambda 混入。**

### 24.2 Experiment Matrix

固定约束：

- warmstart = `coadapt_allrot_interface_bestlr_longer_4x`
- 只训练 `direct_pose_*`
- 冻结 donor trunk / interface / motion_head / lambda
- 不改 runtime contract
- 不改 loss family

本轮实际执行矩阵：

| candidate | warmstart | steps | lr | wd | actual trainable set |
|---|---|---:|---:|---:|---|
| `coadapt_4x_directonly_calibration_short` | `coadapt_4x` | 60 | `5e-5` | `0.0` | `train_direct_pose` only |
| `coadapt_4x_directonly_calibration_120` | `coadapt_4x` | 120 | `5e-5` | `0.0` | `train_direct_pose` only |
| `coadapt_4x_directonly_calibration_180` | `coadapt_4x` | 180 | `5e-5` | `0.0` | `train_direct_pose` only |
| `coadapt_4x_directonly_calibration_240` | `coadapt_4x` | 240 | `5e-5` | `0.0` | `train_direct_pose` only |

为了保持本轮单一 follow-up 最小化：

- **没有扩散成额外 lr / wd / schedule sweep。**
- 先让 `120 / 180 / 240` 自身把 trajectory 说清楚。

### 24.3 Final Intuitive Compare

| candidate | DirectGeoLocalDeg | delta vs baseline gate | GeoLocalDeg mean | BlendGeoLocalDeg mean | lambda_present | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `current_frozen_trunk_replace_control` | `0.177592` | `+0.013846` | `76.058629` | `76.058629` | `false` | `0.996639` | `0.815900` | `62.188209` | `69.351997` | `73.447931` |
| `baseline_replace` | `0.153746` | `-0.010000` | `54.310389` | `54.310389` | `false` | `0.769215` | `0.544482` | `45.361890` | `44.519712` | `46.769045` |
| `coadapt_allrot_interface_bestlr_longer_4x` | `0.213743` | `+0.049998` | `25.731748` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_plus_baseline_directpose_swap` | `0.150419` | `-0.013326` | `25.731748` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_plus_control_directpose_swap` | `0.188263` | `+0.024518` | `25.731748` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `baseline_plus_coadapt4x_directpose_swap` | `0.215619` | `+0.051873` | `54.311840` | `54.311840` | `false` | `0.769241` | `0.544503` | `45.363390` | `44.522049` | `46.771412` |
| `coadapt_4x_directonly_calibration_short` | `0.191253` | `+0.027507` | `25.731748` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_directonly_calibration_120` | `0.188647` | `+0.024901` | `25.731748` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_directonly_calibration_180` | `0.181511` | `+0.017765` | `25.731748` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_directonly_calibration_240` | `0.172359` | `+0.008613` | `25.731748` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |

直接读这张表，本轮新增结论非常明确：

- **direct-only calibration 确实持续修复 `DirectGeoLocalDeg`。**
- 轨迹是单调的：
  - `0.191253`
  - `0.188647`
  - `0.181511`
  - `0.172359`
- 但 **`240` steps 仍然没有跨过 `0.163746` hard gate。**
- pose-side primary metrics 继续完全不动。

### 24.4 Baseline Replacement Gate

| candidate | pose better than baseline? | direct non-regression vs baseline? | replace baseline eligible? |
|---|---|---|---|
| `baseline_replace` | no | yes | no |
| `current_frozen_trunk_replace_control` | no | no | no |
| `coadapt_allrot_interface_bestlr_longer_4x` | yes | no | no |
| `coadapt_4x_plus_baseline_directpose_swap` | yes | yes | yes |
| `coadapt_4x_plus_control_directpose_swap` | yes | no | no |
| `coadapt_4x_directonly_calibration_short` | yes | no | no |
| `coadapt_4x_directonly_calibration_120` | yes | no | no |
| `coadapt_4x_directonly_calibration_180` | yes | no | no |
| `coadapt_4x_directonly_calibration_240` | yes | no | no |

这张 gate table 要把两个 candidate 类型分开看：

1. **static transplant candidate 已经存在。**
   - `coadapt_4x_plus_baseline_directpose_swap`
   - pose better than baseline = `yes`
   - direct non-regression vs baseline = `yes`

2. **但 trainable direct-only candidate 这轮还不存在。**
   - `60 / 120 / 180 / 240` 全部：
     - pose better than baseline = `yes`
     - direct non-regression vs baseline = `no`

### 24.5 Calibration Trajectory

| candidate | steps | DirectGeoLocalDeg | delta vs baseline_replace | delta vs coadapt_4x | pose preserved vs coadapt_4x? |
|---|---:|---:|---:|---:|---|
| `coadapt_4x_directonly_calibration_short` | `60` | `0.191253` | `+0.037507` | `-0.022490` | `yes` |
| `coadapt_4x_directonly_calibration_120` | `120` | `0.188647` | `+0.034901` | `-0.025096` | `yes` |
| `coadapt_4x_directonly_calibration_180` | `180` | `0.181511` | `+0.027765` | `-0.032233` | `yes` |
| `coadapt_4x_directonly_calibration_240` | `240` | `0.172359` | `+0.018613` | `-0.041384` | `yes` |

这张 trajectory table 本轮最关键：

1. **curve 是稳定单调改善，不是随机抖动。**
2. **pose-side primary metrics 在整个 trajectory 上完全保留。**
3. **best trainable case 是 `240` steps。**
4. **best case 距离 hard gate 只剩：**
   - `0.172359 - 0.163746 = 0.008613`

### 24.6 Mandatory Judgement

直接回答本轮 6 个判断题：

1. **`120 / 180 / 240` 的 direct-only calibration 中，是否有任何一个过了 baseline hard gate？**
   - **没有。**
   - best case 是 `coadapt_4x_directonly_calibration_240`
   - `DirectGeoLocalDeg = 0.172359`
   - 仍高于 gate `0.163746`，差 `0.008613`

2. **如果过了，pose-side primary metrics 是否仍然基本不变？**
   - 这一题本轮变成：
   - **虽然没有过 gate，但 pose-side primary metrics 在 `120 / 180 / 240` 上都基本完全不变。**
   - 所以 direct-only calibration 的副作用不是 pose-side regression

3. **是否已经存在一个 trainable candidate，可以真正替换 `baseline_replace`？**
   - **还没有。**
   - 当前只有 static transplant candidate 通过了 replace gate
   - trainable direct-only candidate 仍未过 hard gate

4. **如果还没有，问题更像 calibration strength 不够，还是 direct-only ownership 本身不够？**
   - **更像 calibration strength 还不够。**
   - 原因不是一句“steps 不够”这么窄，而是更大的 calibration-strength bucket：
     - 现有 `60 -> 120 -> 180 -> 240` 是单调改善
     - pose-side 完全保持
     - 剩余 gap 只剩 `0.008613`
   - 因而当前证据**不支持**“direct-only ownership 本身无效”
   - 当前证据更支持：
     - **`direct_pose_*` ownership 是对的**
     - **但现有 `1 epoch / lr=5e-5 / wd=0 / fixed-step` recipe 仍略微 underpowered**
   - 仅凭本轮矩阵，**还不能精确区分**：
     - 是纯 steps 不够
     - 还是需要一个稍微更稳的 lr / schedule
   - 但这两者都属于 calibration strength，不属于 ownership 失效

5. **是否还应该继续优先 direct-only calibration，而不是回 basetrain / 70a？**
   - **是。**
   - 因为本轮已经把 failure mode 收缩到 very small residual gap：
     - direct-only 是有效的
     - 只是还差最后一截
   - 所以还不该回 basetrain / `70a`

6. **adapter 是否仍然不是下一步第一优先级？**
   - **是。**
   - 这轮更进一步说明：
     - 问题不是 adapter-first
     - 问题是 direct-only calibration recipe 还没把 ownership fully calibrate 到 gate 内

### 24.7 Final Recommendation

把这一轮压成一句话：

- **`coadapt_4x` 的 `direct_pose_*` 确实是 trainable 的，而且 direct-only calibration 能稳定把 `DirectGeoLocalDeg` 从 `0.213743` 一路压到 `0.172359`，同时 pose-side primary metrics 完全不动；但在当前最小 `60/120/180/240 @ lr=5e-5` recipe 下，仍然没有 candidate 过 `0.163746` hard gate。**

明确建议只留一句：

- **下一步仍然优先 direct-only calibration，不回 basetrain / `70a`，也不做 adapter-first；如果要加一轮新实验，只加一个 low-risk direct-only strength tweak 去补掉剩余 `0.008613` gate gap。**

## 25. 240+120 Low-LR Direct-Only Continuation

这一轮只回答一个更窄的问题：

- **从当前 best direct-only case (`240` steps) 继续 low-LR 轻推 `120` steps，能不能把最后 `0.008613` gate gap 补掉。**

本轮设置完全按最小 continuation 执行：

- candidate:
  `coadapt_4x_directonly_calibration_240plus120_lowlr`
- warmstart:
  `coadapt_4x_directonly_calibration_240`
- train target:
  `train_direct_pose`
- 实际 trainable set:
  `20` 个 `direct_pose_*` 参数
- trunk / interface / motion_head / lambda:
  全冻结
- steps:
  `120`
- lr:
  `3e-5`
- wd:
  `0`
- runtime contract / loss family:
  不变

### 25.1 Code Facts

这轮先把 code facts 写死：

1. **warmstart 确实来自 `240` direct-only ckpt，而不是回到 raw `coadapt_4x`。**
   - config 里的 `ckpt_in` 指向：
     `coadapt_4x_directonly_calibration_240/ckpt_last_...pth`

2. **本轮仍然是 strict direct-only lane。**
   - `train_direct_pose = true`
   - `train_incremental_replace = false`
   - `train_lambda_head = false`
   - `train_arm_residual = false`
   - `train_arm_leg_residual = false`
   - `incremental_interface_mode = off`

3. **runtime log 继续证明只动了 `direct_pose_*`。**
   - `mode=train_direct_pose`
   - `trainable=20 params`
   - sample names 仍然全是 `direct_pose_*`

结论简写：

- **这不是 trunk continuation，也不是 adapter continuation，而是从 best direct-only case 出发的 low-LR direct-only continuation。**

### 25.2 Minimal Compare

| candidate | DirectGeoLocalDeg | delta vs baseline gate | delta vs `coadapt_4x_directonly_calibration_240` | GeoLocalDeg mean | lambda_present | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `coadapt_allrot_interface_bestlr_longer_4x` | `0.213743` | `+0.049998` | `+0.041384` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_directonly_calibration_240` | `0.172359` | `+0.008613` | `ref` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_directonly_calibration_240plus120_lowlr` | `0.177308` | `+0.013562` | `+0.004949` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_plus_baseline_directpose_swap` | `0.150419` | `-0.013326` | `-0.021939` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |

直接读这张表，本轮结论很硬：

- **low-LR continuation 没有继续改善，反而相对 `240` case 回弹。**
- `0.172359 -> 0.177308`
- 回弹幅度：
  `+0.004949`
- 但 pose-side primary metrics 继续完全不动

### 25.3 Gate Read

| candidate | pose better than baseline? | direct non-regression vs baseline? | replace baseline eligible? |
|---|---|---|---|
| `coadapt_4x_directonly_calibration_240` | yes | no | no |
| `coadapt_4x_directonly_calibration_240plus120_lowlr` | yes | no | no |

这里的 gate 含义很明确：

- **trainable direct-only replacement candidate 仍然不存在。**
- 但 failure mode 已经从“ownership 可能不成立”收缩成：
  - **late-stage continuation recipe 没对上**

### 25.4 Mandatory Judgement

按你给的判读规则，直接回答：

1. **能不能拿到一个真正 trainable 的 replacement candidate？**
   - **这轮没有。**
   - `240plus120_lowlr` 仍未过 hard gate

2. **当前差的最后一截，是不是只是 calibration strength / late-stage optimization 问题？**
   - **更像是 late-stage optimization / schedule mismatch。**
   - 因为：
     - 先前 `60 -> 120 -> 180 -> 240` 是稳定单调改善
     - 但从 best `240` case 再接 `120 @ 3e-5` 并没有继续下去
     - 而是小幅回弹
   - 所以当前最合理的解释不是：
     - trunk / adapter / upstream ownership 缺失
   - 当前更像：
     - **direct-only ownership 本身是可训练的**
     - **但 late-stage continuation 不能靠这一个简单 low-LR recipe 直接过线**

3. **是否应该回 basetrain / `70a`？**
   - **不应该。**
   - 这轮失败不支持回上游

4. **adapter 是否变成第一优先级？**
   - **没有。**
   - 这轮仍然把问题收敛在 direct-only recipe 层

### 25.5 Final Recommendation

把这一轮压成一句话：

- **`240plus120_lowlr` 没把 `240` case 推过 gate，反而从 `0.172359` 回到 `0.177308`；因此当前 failure 更像 late-stage lr / schedule mismatch，而不是 direct-only ownership 无效，更不是 trunk / adapter 问题。**

明确建议：

- **下一步仍然只做 direct-only recipe 调整；不要回 basetrain / `70a`，也不要跳到 adapter-first。**

## 26. Continuous 360-Step Direct-Only Check

这一轮只回答一个 falsifier：

- **`240plus120_lowlr` 的回弹，是否主要只是 optimizer-state reset artifact。**

最直接的验证不是继续 warmstart continuation，而是：

- **从 raw `coadapt_4x` 再跑一条连续的 `360 @ 5e-5` direct-only 轨迹。**

如果 `360` 明显优于 `240`，那 optimizer-reset 解释会很强。
如果 `360` 也不如 `240`，那就说明：

- **问题不只是 continuation optimizer-state reset。**

### 26.1 Experiment Matrix

本轮新增 case：

| candidate | warmstart | steps | lr | wd | trainable set |
|---|---|---:|---:|---:|---|
| `coadapt_4x_directonly_calibration_360` | `coadapt_allrot_interface_bestlr_longer_4x` | `360` | `5e-5` | `0` | `train_direct_pose` only |

保持不变：

- 只训 `direct_pose_*`
- 冻结 trunk / interface / motion_head / lambda
- 不改 runtime contract
- 不改 loss family

runtime log 继续确认：

- `mode=train_direct_pose`
- `trainable=20 params`
- sample names 全在 `direct_pose_*`

### 26.2 Minimal Compare

| candidate | DirectGeoLocalDeg | delta vs baseline gate | delta vs `coadapt_4x_directonly_calibration_240` | GeoLocalDeg mean | lambda_present | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `coadapt_allrot_interface_bestlr_longer_4x` | `0.213743` | `+0.049998` | `+0.041384` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_directonly_calibration_240` | `0.172359` | `+0.008613` | `ref` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_directonly_calibration_360` | `0.176945` | `+0.013199` | `+0.004586` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_directonly_calibration_240plus120_lowlr` | `0.177308` | `+0.013562` | `+0.004949` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |
| `coadapt_4x_plus_baseline_directpose_swap` | `0.150419` | `-0.013326` | `-0.021939` | `25.731748` | `false` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` |

这张表的结论比 `240plus120_lowlr` 更强：

- **连续 `360` 也没有优于 `240`。**
- `240 -> 360`：
  - `0.172359 -> 0.176945`
  - 回弹 `+0.004586`
- `240 -> 240plus120_lowlr`：
  - `0.172359 -> 0.177308`
  - 回弹 `+0.004949`
- 两条不同 late-stage recipe 都比 `240` 差，而且差值量级非常接近

### 26.3 What This Falsifies

这一轮最重要的不是“又没过 gate”，而是它 falsify 了什么：

1. **不能再把 `240plus120_lowlr` 的回弹主要归因于 optimizer-state reset。**
   - reset 可能仍然有贡献
   - 但它已经不是充分解释

2. **因为连续 `360 @ 5e-5` 也回弹了。**
   - 这条 run 没有 warmstart continuation
   - 没有换 optimizer state
   - 也没有换 LR
   - 仍然比 `240` 更差

3. **因此当前更像是 late-stage objective / schedule mismatch 或 overrun。**
   - `240` 附近更像是这条 recipe 的 sweet spot
   - 再继续堆 steps，不管是 continuous 还是 restarted，都没有继续受益

### 26.4 Mandatory Judgement

直接回答当前判断：

1. **“回弹主要是 optimizer-state reset 问题，不是 LR 本身的问题”这个解释对吗？**
   - **不完全对。**
   - 更准确地说：
     - optimizer-state reset **可能是 `240plus120_lowlr` 回弹的一个因素**
     - 但 **不是主要充分解释**
   - 因为 continuous `360` 也回弹了

2. **现在更像什么？**
   - **更像 direct-only late-stage recipe 已经过了 sweet spot。**
   - 也就是：
     - 不是 ownership 无效
     - 不是必须回 trunk / adapter
     - 也不是“只要继续堆 steps 就会自然过线”

3. **是否还应该回 basetrain / `70a`？**
   - **仍然不应该。**

4. **adapter 是否变成第一优先级？**
   - **仍然不是。**

### 26.5 Final Recommendation

把这一轮压成一句话：

- **`360 @ 5e-5` 连续直跑也没有超过 `240`，所以 `240plus120_lowlr` 的回弹不能主要归因于 optimizer-state reset；当前更像 `240` 已经接近这条 direct-only recipe 的 sweet spot，之后出现 late-stage mismatch / overrun。**

明确建议：

- **下一步如果还留在 direct-only，就不要再做“纯加 steps”或“240 后简单续跑”了；应该改成 direct-only recipe 级别的 late-stage schedule 设计，而不是回 basetrain / `70a`，也不是 adapter-first。**

## 27. Plan/Meas Causal Probe For Direct Replacement Gap

这一轮严格按 replacement decision debug 来做：

- 不新训练
- 不回 basetrain / `70a`
- 不做 adapter / trunk / lambda 路线
- 只复用现有 `run_freerun_cycles.py` 做 direct-only causal probe

产物目录：

- `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/summary.json`
- `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/compare_table.md`
- `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/ranking_table.md`
- `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/matrix_table.md`

### 27.1 Code Facts

- `direct_pose_feat_source` 继承仍是 `cond`，但 direct head 实际消费的不是纯 `cond`，而是 `direct_feat + plan_in + meas_in` 的 concat。
  - `direct_feat_source` 解析在 `train/models.py:3735-3750`
  - direct concat 在 `train/models.py:3785-3801`
- `contacts_plan` 的生成路径仍然是 `cond_seq -> contact_plan_cell(GRUCell) -> contact_plan_head -> contacts_plan`。
  - Event-Clock on 时先做 `plan_z_raw -> event_clock_corrector -> logits_base -> plan_probs`
  - Event-Clock off 时也是 `contact_plan_cell -> contact_plan_head -> plan_probs`
  - 关键路径在 `train/models.py:3224-3384`
- `run_freerun_cycles.py` 里的 `--direct_pose_plan_source` / `--direct_pose_meas_source` 是 direct-only override。
  - CLI 入口和 trainer 字段在 `train/validate/run_freerun_cycles.py:719-748`
  - per-step meas override 明确注明“**This does NOT affect contacts_err/Event-Clock/λ unless you also override --contacts_meas_source**”，见 `train/validate/run_freerun_cycles.py:4469-4478`
  - per-step plan override 明确注明“**This does NOT affect contacts_plan/contacts_err/lambda (only direct hint)**”，见 `train/validate/run_freerun_cycles.py:4665-4690`
- 因此本轮 probe 改的是 direct hint 消费，不是全局 `contacts_plan / contacts_meas / contacts_err / lambda` contract。

### 27.2 Experiment Matrix

复用策略：

- `model/model` 直接复用现有 eval JSON：
  - `coadapt_4x_plus_baseline_directpose_swap`
  - `coadapt_4x_directonly_calibration_240`
- 其余 12 个 setting 用同一 eval contract 重跑：
  - `teacher=Walk_F_teacher`
  - `rounds=5`
  - `time_index_mode=cycle`
  - `phase_reset_source=none`
  - `event_clock=auto`
  - `pose_hist_source=buffer`
  - `pose_hist_update_source=pred`
  - `lambda_fusion_apply=true`
- 只改：
  - `--direct_pose_plan_source`
  - `--direct_pose_meas_source`
- runtime contract change：`no`
- coarse matrix 已经足够区分主因，本轮不再补 local sensitivity

| candidate | direct_pose_plan_source | direct_pose_meas_source | runtime contract changed? |
|---|---|---|---|
| `coadapt_4x_plus_baseline_directpose_swap` | `model` | `model` | `no` |
| `coadapt_4x_plus_baseline_directpose_swap` | `gt` | `model` | `no` |
| `coadapt_4x_plus_baseline_directpose_swap` | `model` | `gt` | `no` |
| `coadapt_4x_plus_baseline_directpose_swap` | `gt` | `gt` | `no` |
| `coadapt_4x_plus_baseline_directpose_swap` | `zero` | `model` | `no` |
| `coadapt_4x_plus_baseline_directpose_swap` | `model` | `zero` | `no` |
| `coadapt_4x_plus_baseline_directpose_swap` | `zero` | `zero` | `no` |
| `coadapt_4x_directonly_calibration_240` | `model` | `model` | `no` |
| `coadapt_4x_directonly_calibration_240` | `gt` | `model` | `no` |
| `coadapt_4x_directonly_calibration_240` | `model` | `gt` | `no` |
| `coadapt_4x_directonly_calibration_240` | `gt` | `gt` | `no` |
| `coadapt_4x_directonly_calibration_240` | `zero` | `model` | `no` |
| `coadapt_4x_directonly_calibration_240` | `model` | `zero` | `no` |
| `coadapt_4x_directonly_calibration_240` | `zero` | `zero` | `no` |

### 27.3 Final Intuitive Compare

baseline gate 仍取：

- `baseline_replace`
- `DirectGeoLocalDeg = 0.153746`

> 注：这一轮 coarse probe 的辨别指标是 `DirectGeoLocalDeg`。`GeoLocalDeg / BlendGeoLocalDeg / Rot6dLocalL2 / GeoDeg` 在 probe rows 上几乎一起平移到同一个运行点，所以 replacement judgement 仍然以 direct 指标为主。

| candidate | plan_source | meas_source | DirectGeoLocalDeg | delta vs candidate baseline | delta vs baseline gate | GeoLocalDeg mean | BlendGeoLocalDeg mean | lambda_present | Rot6dLocalL2 | Rot6dLocalL2Weighted | GeoDeg | KeyBoneGeoDegMean | KeyBoneGeoLocalDegMean |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `baseline_replace` | `model` | `model` | `0.153746` | `+0.000000` | `+0.000000` | `73.614296` | `73.614296` | `yes` | `1.188174` | `0.992915` | `72.508720` | `85.748924` | `90.248283` |
| `coadapt_allrot_interface_bestlr_longer_4x` | `model` | `model` | `0.213743` | `+0.000000` | `+0.059998` | `44.006924` | `44.006924` | `yes` | `0.750313` | `0.457116` | `43.212677` | `40.774555` | `43.153343` |
| `coadapt_4x_plus_baseline_directpose_swap` | `model` | `model` | `0.150419` | `+0.000000` | `-0.003326` | `44.006924` | `44.006924` | `yes` | `0.750313` | `0.457116` | `43.212677` | `40.774555` | `43.153343` |
| `coadapt_4x_plus_baseline_directpose_swap` | `gt` | `model` | `0.150220` | `-0.000200` | `-0.003526` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_plus_baseline_directpose_swap` | `model` | `gt` | `0.151791` | `+0.001372` | `-0.001955` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_plus_baseline_directpose_swap` | `gt` | `gt` | `0.151570` | `+0.001151` | `-0.002175` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_plus_baseline_directpose_swap` | `zero` | `model` | `0.151184` | `+0.000764` | `-0.002562` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_plus_baseline_directpose_swap` | `model` | `zero` | `0.150419` | `+0.000000` | `-0.003326` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_plus_baseline_directpose_swap` | `zero` | `zero` | `0.151184` | `+0.000764` | `-0.002562` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_directonly_calibration_240` | `model` | `model` | `0.172359` | `+0.000000` | `+0.018613` | `44.006924` | `44.006924` | `yes` | `0.750313` | `0.457116` | `43.212677` | `40.774555` | `43.153343` |
| `coadapt_4x_directonly_calibration_240` | `gt` | `model` | `0.153369` | `-0.018990` | `-0.000377` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_directonly_calibration_240` | `model` | `gt` | `0.163949` | `-0.008410` | `+0.010203` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_directonly_calibration_240` | `gt` | `gt` | `0.148023` | `-0.024335` | `-0.005722` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_directonly_calibration_240` | `zero` | `model` | `0.198338` | `+0.025979` | `+0.044592` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_directonly_calibration_240` | `model` | `zero` | `0.172359` | `+0.000000` | `+0.018613` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |
| `coadapt_4x_directonly_calibration_240` | `zero` | `zero` | `0.198338` | `+0.025979` | `+0.044592` | `44.277092` | `44.277092` | `yes` | `0.754423` | `0.461214` | `43.478043` | `41.064270` | `43.460224` |

最直观的读法：

- `self240` 从 `model/model = 0.172359` 换到 `plan=gt, meas=model = 0.153369`，直接吃掉了 `-0.018990`。
  - 它与 transplant 的 gap 从 `0.021939` 缩到 `0.002949`
  - 也就是已经关掉了约 `86.6%` 的原始 gap
- `self240` 只换 `meas=gt` 时只有 `0.163949`
  - 相对自身 baseline 只拿到 `-0.008410`
  - 仍然比 transplant 差 `0.013530`
- `self240` 的 `gt/gt = 0.148023`
  - 已经不只是追平 baseline gate
  - 还略优于 transplant `0.150419`
  - 这说明 direct head 本身并不是“完全不会训”或“根本没有 readout capacity”
- transplant 基本不动：
  - `model/model = 0.150419`
  - `gt/model = 0.150220`
  - `zero/zero = 0.151184`
  - 全矩阵波动不到 `0.0014`
- `self240` 的 `model/zero = 0.172359` 与 `model/model = 0.172359` 完全相同。
  - 也就是 learned `meas` 在当前 self-trained best direct-only 里几乎没有提供可见收益
  - 真正拉分的是 `plan`

### 27.4 Causal Ranking

| candidate | best plan/meas setting by DirectGeoLocalDeg | gain from plan=model -> plan=gt | gain from meas=model -> meas=gt | gain from model/model -> zero/zero | which signal is more causal |
|---|---|---:|---:|---:|---|
| `coadapt_4x_plus_baseline_directpose_swap` | `gt/model (0.150220)` | `+0.000200` | `-0.001372` | `-0.000764` | `neither` |
| `coadapt_4x_directonly_calibration_240` | `gt/gt (0.148023)` | `+0.018990` | `+0.008410` | `-0.025979` | `plan` |

这里可以再压成一句话：

- transplant 对 `plan/meas` override 几乎不敏感
- `self240` 对 `plan` 明显敏感，对 `meas` 次敏感，对 `zero/zero` 明显脆弱
- 而且 `zero/model` 与 `zero/zero` 完全相同，`model/zero` 与 `model/model` 完全相同
  - 所以 `zero/zero` 的损伤基本全部来自 `plan`
  - `meas` 单独清零几乎不伤 `self240`

### 27.5 Replacement Decision

| candidate | pose better than baseline? | direct non-regression vs baseline? | self-contained trainable? | production replacement eligible? | recommended role |
|---|---|---|---|---|---|
| `baseline_replace` | `ref/current` | `yes` | `yes` | `yes` | `production` |
| `coadapt_allrot_interface_bestlr_longer_4x` | `yes` | `no` | `yes` | `no` | `reject` |
| `coadapt_4x_plus_baseline_directpose_swap` | `yes` | `yes` | `no` | `yes` | `production` |
| `coadapt_4x_directonly_calibration_240` | `yes` | `no` | `yes` | `no` | `research-only` |

这张表对应的 decision 很干净：

- production replacement 看的是“pose 好于 baseline，同时 direct 不回归”
- 这个条件当前只有 `coadapt_4x_plus_baseline_directpose_swap` 满足
- `self240` 虽然是 self-contained trainable，但在真实 `model/model` 下仍然 direct 回归
- 所以 self-contained trainable replacement 不能再作为 production blocker

### 27.6 Mandatory Judgement

直接回答本轮必须判断的 6 个问题：

1. **对 `coadapt_4x_directonly_calibration_240` 来说，`plan=gt` 是否能显著缩小它与 transplant candidate 的 gap？**
   - **能，而且是主要量级。**
   - `0.172359 -> 0.153369`
   - 原始 gap：`0.021939`
   - `plan=gt` 后剩余 gap：`0.002949`

2. **`meas=gt` 是否也有类似量级的收益，还是明显更弱？**
   - **明显更弱。**
   - `0.172359 -> 0.163949`
   - gain 只有 `0.008410`
   - 约等于 `plan` gain 的 `44%`

3. **`zero/zero` 是否会同时伤害 transplant candidate 和 self-trained 240，还是只对其中一个更敏感？**
   - **主要只对 self-trained 240 更敏感。**
   - transplant：`0.150419 -> 0.151184`，只坏 `+0.000764`
   - `self240`：`0.172359 -> 0.198338`，坏 `+0.025979`

4. **当前 `0.150 vs 0.172` 的 gap，更像什么？**
   - **更像 `plan/meas` signal mismatch，且主因是 `plan`。**
   - 更准确地说：
     - 主因是 direct head 实际消费到的 `plan` hint 分布不对
     - `meas` 也有次级帮助
     - `direct_pose_*` 自身 readout capacity 不足不是主要解释
   - 证据是：
     - `self240 gt/gt = 0.148023`
     - 也就是同一 direct head 在理想 hint 下并不弱
   - 但也不该把结论写成“只有 upstream signal mismatch，没有任何 readout interaction”。
     - 因为 `gt/model = 0.153369` 还没完全到 transplant 的 `0.150419`
     - full best 仍然出现在 `gt/gt`
   - 所以最准确的表述是：
     - **主要是 `plan/meas` mismatch，尤其 `plan`**
     - **有小的 secondary interaction**

5. **在这个 probe 后，production replacement 是否应该直接采用 `coadapt_4x_plus_baseline_directpose_swap`？**
   - **应该。**

6. **self-contained trainable replacement 是否应该从 production blocker 降级为 research follow-up？**
   - **应该。**

### 27.7 Final Recommendation

把这一轮压成一句话：

- **`0.150419 vs 0.172359` 的主差异，主要不是“self-trained direct head 根本训不出来”，而是 `self240` 实际消费到的 direct `plan/meas` hint，尤其 `plan`，不如 transplant 稳；在 `plan=gt` 下它几乎追平 transplant，在 `gt/gt` 下甚至反超。**

明确建议：

- **production replacement 直接采用 `coadapt_4x_plus_baseline_directpose_swap`。**
- **self-contained trainable replacement 从 production blocker 降级为 research follow-up。**
- **如果后续还研究 self-contained 方案，优先盯 direct head 实际吃到的 `contacts_plan` 分布与其可用性校准，`meas` 次之；不要把主问题再退回成“direct_pose_* 自身不会训练”。**

---

## 28. 2026-04-06 update: minimal self-contained `direct + contacts_plan ownership` lane

这一节是对第 27 节 production 结论的 superseding update。

先继承本轮固定前提，不回头重证：

- `coadapt_4x_plus_baseline_directpose_swap` 仍然只算 oracle / direction validator。
- 它不是 self-contained tailk7 replacement，所以**不再作为 production 候选**。
- 本轮唯一目标是实现并验证一个最小 self-contained trainable lane：
  - 只让 `direct_pose_*` 和它实际消费的 `contacts_plan` 生成路径一起 co-calibrate
  - 不改 trunk / donor / adapter / lambda route
  - 尽量复用现有 `train_direct_pose` posttrain recipe

### 28.1 Code facts

- 最小侵入实现点放在 `train/posttrain.py`，**不是**新开一个 XOR train mode，而是扩展既有 `train_direct_pose`。
- ownership lane 通过新 flags 挂到现有 direct mode：
  - `direct_pose_detach_plan`
  - `direct_pose_plan_ownership_enable`
  - `direct_pose_plan_ownership_lr_scale`
  - `direct_pose_plan_ownership_include_init`
  - `direct_pose_plan_ownership_event_clock`
- 开 ownership 时，`direct_pose_detach_plan` 会被强制改成 `false`，保证 direct loss 能回传到 consumed `contacts_plan` path。
- 参数解算范围：
  - 基础 plan path：`contact_plan_cell` / `contact_plan_head` / `contact_plan_time_head`
  - 可选 init：`contact_plan_init_z` / `contact_plan_init_head`
  - 可选 Event-Clock：`event_clock_gate` / `event_clock_corrector`
- optimizer 仍走现有 prefix-based param-group 机制，只新增两档 LR：
  - direct group: `lr`
  - ownership group: `lr * direct_pose_plan_ownership_lr_scale`
- old recipe 默认行为不变：
  - `direct_pose_plan_ownership_enable=false`
  - `direct_pose_detach_plan` 默认仍是 legacy direct-only 的 `true`
  - 不会影响旧 config / runtime contract
- 现有 contact-plan regularization path 保留了：
  - `lambda_plan_entropy_weight`
  - `lambda_plan_dyn_weight`
  - 但它们在 rollout 中仍作用于 `plan_step.detach()`，而且本轮两个 run 都出现 `lambda_mean ~= 0`
  - 所以这条“语义约束”在实现上保留了，在实测上几乎是 inert

### 28.2 Implementation summary

| item | summary |
|---|---|
| changed files | `train/posttrain.py`; 本文档 |
| new mode? | 否。复用 `train_direct_pose`，增加 ownership 子开关 |
| new config / flags | `direct_pose_detach_plan`, `direct_pose_plan_ownership_enable`, `direct_pose_plan_ownership_lr_scale`, `direct_pose_plan_ownership_include_init`, `direct_pose_plan_ownership_event_clock` |
| optimizer hookup | 复用 `optimizer_param_group_overrides` 解析链，在 direct mode 下自动注入 `direct_plan_ownership` low-LR group |
| runtime contract changed? | 否。freerun eval 仍是同一 `model/model` contract |
| old recipes impacted? | 否。只有显式打开 ownership flags 才进入新 lane |
| verification | 1-step smoke run 成功；`python3 -m py_compile train/posttrain.py` 通过；两条 240-step train run + freerun eval 完成 |

### 28.3 Trainable params table

| module group | sample param names | trainable / frozen | lr group |
|---|---|---|---|
| direct pose readout trunk | `direct_pose_head.0.weight`, `direct_pose_head.3.bias` | trainable | main (`5.0e-5`) |
| direct leg branch | `direct_pose_leg_head.0.weight`, `direct_pose_leg_head.6.bias`, `direct_pose_out_leg.weight` | trainable | main (`5.0e-5`) |
| direct arm/else branch | `direct_pose_arm_proj.0.weight`, `direct_pose_else_proj.0.bias`, `direct_pose_out_arm.weight`, `direct_pose_out_else.bias` | trainable | main (`5.0e-5`) |
| plan core path | `contact_plan_cell.weight_ih`, `contact_plan_cell.bias_hh`, `contact_plan_head.0.weight`, `contact_plan_head.4.bias`, `contact_plan_time_head.weight` | trainable | ownership (`1.25e-5`) |
| plan init path | `contact_plan_init_z`, `contact_plan_init_head.0.weight`, `contact_plan_init_head.4.bias` | trainable when `direct_pose_plan_ownership_include_init=true` | ownership (`1.25e-5`) |
| Event-Clock correction path | `event_clock_gate.confidence_head.0.weight`, `event_clock_gate.prior_head.2.bias`, `event_clock_corrector.correction_head.4.weight`, `event_clock_corrector.layer_norm.bias` | trainable only when ownership scope includes Event-Clock | ownership (`1.25e-5`) |
| shared pose trunk | `shared_encoder.0.weight`, `shared_encoder.5.bias`, `motion_head.0.weight`, `residual_proj.weight` | frozen | n/a |
| frozen donor / frozen feature path | `frozen_encoder.mlp.0.weight`, `frozen_period_head.fc.weight` | frozen | n/a |
| other unrelated heads | `so3_delta_corrector.0.weight` and all non-selected heads | frozen | n/a |
| `contact_meas_*` | no active trainable `contact_meas_*` params resolved in this lane | frozen / absent | n/a |

ownership scope差异：

- `direct_pose_plan_ownership_event_clock=auto`
  - 训练总 tensor 数：`57`
  - ownership group：`37 tensors / 167,094 params`
  - 包含 Event-Clock ownership
- `direct_pose_plan_ownership_event_clock=off`
  - 训练总 tensor 数：`39`
  - ownership group：`19 tensors / 152,756 params`
  - 不包含 Event-Clock ownership

### 28.4 Experiment table

所有 run 都从同一个 warmstart 起步：

- warmstart ckpt:
  `models/__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/coadapt_allrot_interface_bestlr_longer_4x/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth`

eval contract 统一为：

- self-contained `model/model`
- `train.validate.run_freerun_cycles`
- `--contacts_meas_source model`
- `--event_clock auto`
- `--lambda_fusion_apply`
- cycle mask: `cycle>=1`, `drop_wrap=True`

| run name | warmstart ckpt | trainable scope | direct LR | plan LR | event_clock ownership | training steps | eval contract |
|---|---|---|---:|---:|---|---:|---|
| `coadapt_4x_direct_plus_plan_ownership_240` | `coadapt_allrot_interface_bestlr_longer_4x` | `direct_pose_*` + `contact_plan_cell/head/time` + `contact_plan_init_*` + `event_clock_gate/corrector` | `5.0e-5` | `1.25e-5` | on (`auto` resolved true) | `240` | self-contained `model/model` freerun |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `coadapt_allrot_interface_bestlr_longer_4x` | `direct_pose_*` + `contact_plan_cell/head/time` + `contact_plan_init_*` | `5.0e-5` | `1.25e-5` | off | `240` | self-contained `model/model` freerun |

artifact paths：

- config:
  - `debug_output/_tmp_cp015_tailk7_plan_ownership_calibration_20260406/configs/coadapt_4x_direct_plus_plan_ownership_240_20260406.json`
  - `debug_output/_tmp_cp015_tailk7_plan_ownership_calibration_20260406/configs/coadapt_4x_direct_plus_plan_ownership_240_noeventclock_20260406.json`
- train logs:
  - `debug_output/_tmp_cp015_tailk7_plan_ownership_calibration_20260406/train_coadapt_4x_direct_plus_plan_ownership_240.log`
  - `debug_output/_tmp_cp015_tailk7_plan_ownership_calibration_20260406/train_coadapt_4x_direct_plus_plan_ownership_240_noeventclock.log`
- eval json:
  - `debug_output/_tmp_cp015_tailk7_plan_ownership_calibration_20260406/eval_model_source/coadapt_4x_direct_plus_plan_ownership_240/Walk_F_freerun_cycles.json`
  - `debug_output/_tmp_cp015_tailk7_plan_ownership_calibration_20260406/eval_model_source/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/Walk_F_freerun_cycles.json`

### 28.5 Result compare table

单位：

- direct / geo metrics: `deg`
- `Rot6dLocalL2*`: raw scalar

| candidate / run | self-contained? | DirectGeoLocalDeg | delta vs `baseline_replace` | delta vs `coadapt_4x_directonly_calibration_240` | `GeoLocalDeg mean` | `BlendGeoLocalDeg mean` | `Rot6dLocalL2` | `Rot6dLocalL2Weighted` | `GeoDeg` | `KeyBoneGeoDegMean` | `KeyBoneGeoLocalDegMean` | `lambda_present` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_replace` | yes | `0.153746` | `+0.000000` | `-0.018613` | `54.310389` | `54.310389` | `0.769215` | `0.544482` | `45.361890` | `44.519712` | `46.769045` | `false` |
| `coadapt_4x` | yes | `0.213743` | `+0.059998` | `+0.041384` | `25.731748` | `25.731748` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` | `false` |
| `coadapt_4x_directonly_calibration_240` | yes | `0.172359` | `+0.018613` | `+0.000000` | `25.731748` | `25.731748` | `0.388863` | `0.252461` | `21.443368` | `23.033965` | `24.171176` | `false` |
| `coadapt_4x_direct_plus_plan_ownership_240` | yes | `0.174185` | `+0.020439` | `+0.001826` | `25.766133` | `25.766133` | `0.389364` | `0.252846` | `21.470860` | `23.165714` | `24.307935` | `false` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | `0.171490` | `+0.017744` | `-0.000869` | `25.742469` | `25.742469` | `0.389050` | `0.252565` | `21.451981` | `23.071112` | `24.209931` | `false` |

直接读数：

- ownership + Event-Clock：
  - 比 direct-only `240` **更差**
  - `0.172359 -> 0.174185`，坏 `+0.001826`
- ownership without Event-Clock：
  - 比 direct-only `240` **小幅更好**
  - `0.172359 -> 0.171490`，好 `-0.000869`
- 但两个 ownership run 都仍明显落后 `baseline_replace=0.153746`
- pose side 两个 ownership run 都远好于 `baseline_replace`，并且与 `coadapt_4x` 基本持平
  - `noeventclock` 相对 `coadapt_4x` 的 pose 代价只有：
    - `GeoLocalDeg +0.008480`
    - `GeoDeg +0.008614`
    - `KeyBoneGeoLocalDegMean +0.038755`

### 28.6 Replacement decision table

| candidate | self-contained? | pose better than baseline? | direct non-regression vs baseline? | production eligible? | recommended role |
|---|---|---|---|---|---|
| `baseline_replace` | yes | `ref/current` | yes | yes | `production` |
| `coadapt_4x` | yes | yes | no | no | `reject` |
| `coadapt_4x_plus_baseline_directpose_swap` | no | yes | yes | no | `research-only` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | no | no | `research-only` |
| `coadapt_4x_direct_plus_plan_ownership_240` | yes | yes | no | no | `reject` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | yes | no | no | `research-only` |

这里的 production decision 现在是：

- transplant / swap 继续只保留为 oracle reference
- 当前**没有**新的 self-contained production candidate
- `noeventclock` lane 只能算 research positive

### 28.7 Mandatory judgements

1. **这个最小 lane 是否验证了“主要问题是 `contacts_plan ownership mismatch`”？**
   - **只验证了“这件事确实存在并且有小幅正收益”，但没有验证到“它单独就是主 blocker”。**
   - 证据：
     - raw plan-path ownership（不含 Event-Clock ownership）确实把 direct 从 `0.172359` 拉到 `0.171490`
     - 但 gain 只有 `-0.000869`
     - 远小于它相对 `baseline_replace` 的剩余 gap `+0.017744`
   - 所以更准确的表述是：
     - **`contacts_plan` ownership mismatch 是 real factor**
     - **但只靠这条最小 lane，不足以解释或关闭主要 production gap**

2. **只放开 direct + plan path，是否足以显著优于 `0.172359`？**
   - **不够。**
   - 最好结果是 `0.171490`
   - 这是小幅优于，不是显著优于

3. **Event-Clock ownership 是否需要一起放开，还是 raw `contact_plan_cell/head` 就够？**
   - **本轮证据支持：先不要放 Event-Clock ownership。**
   - 因为：
     - `raw plan path + init/time` 且 `event_clock ownership=off` 给出了唯一正向结果 `0.171490`
     - 把 `event_clock_gate/corrector` 一起放开后反而变成 `0.174185`
   - 也就是说，本轮最小 lane 下：
     - **raw `contact_plan_cell/head/time/init` 比 “再加 Event-Clock ownership” 更稳**

4. **当前是否已经出现 self-contained production candidate？**
   - **没有。**
   - `noeventclock` 虽然 self-contained 且优于 direct-only `240`
   - 但 direct 仍比 `baseline_replace` 坏 `+0.017744`
   - 所以只能归类为 `research-only`

5. **如果还没有，下一步最应该继续加的是什么？**
   - **主优先级只选一个：`other`**
   - 具体是：
     - **让 consumed `contacts_plan` path 在这个 lane 里拿到真正 active 的 plan objective / supervision**
   - 原因：
     - 当前已试过 raw plan ownership，本身只有小增益
     - Event-Clock ownership 当前是负贡献
     - `contact_plan init/time head` 已经在本轮 lane 里放开
     - `contact_meas` 仍是次因，而且当前 lane 里没有 active `contact_meas_*` ownership 可直接收敛
     - 现有 plan regularization 虽然保留了，但因为 `lambda_mean ~= 0`，实测几乎不起作用

### 28.8 Final judgement

把本轮压成一句话：

- **最小 self-contained `direct + contacts_plan ownership` lane 已经实现、可运行、也给出了一点正信号，但它没有把 tailk7 送过 direct gate；唯一正向变体是“不开 Event-Clock ownership”的 raw plan-path co-calibration，结果 `0.171490` 只是略优于 `0.172359`，还远不是 production replacement。**

因此当前最稳妥的结论是：

- 本轮**没有**产生新的 self-contained production candidate
- `contacts_plan` ownership mismatch 不是假问题，但它在这个最小 lane 下只解释了很小一部分 gap
- Event-Clock ownership 现在不该作为第一优先级继续放开
- production 仍维持 `baseline_replace`
- research follow-up 的主方向不是 transplant，也不是再做纯 direct-only probe，而是：
  - **让 self-contained consumed plan path 在 direct lane 中获得真正有效的 plan-side训练约束**

## 29. `contacts_plan` white-box semantic audit（self-contained only）

这一节只做 user 要求的最小 white-box audit，不回 basetrain / 70a，不做 trunk / donor / adapter redesign，也不开新的大训练框架。

本节新增 artifact：

- audit root:
  - `debug_output/_tmp_cp015_tailk7_contact_plan_semantic_audit_20260406/summary.json`
  - `debug_output/_tmp_cp015_tailk7_contact_plan_semantic_audit_20260406/summary.md`
- teacher artifacts:
  - `debug_output/_tmp_cp015_tailk7_contact_plan_semantic_audit_20260406/teacher_artifacts/*.json`
- minimal helper:
  - `train/validate/run_contact_plan_semantic_audit.py`

### 29.1 Code facts（继承后再收口）

1. **`contacts_plan` 的生成链**
   - `contacts_plan` 来自 `cond_seq -> contact_plan_cell -> contact_plan_head -> contacts_plan`
   - Event-Clock 打开时，会先经过 `event_clock_gate / event_clock_corrector` 再回到 `contact_plan_head`
   - code path:
     - `train/models.py:3224`
     - `train/models.py:3238`
     - `train/models.py:3244`
     - `train/models.py:3357`
     - `train/models.py:3384`

2. **direct head 确实消费 `plan + meas`，不是只看 cond**
   - direct head 会把 `plan_in=contacts_plan` 与 `meas_in=contacts_meas` 一起送入 direct 分支
   - code path:
     - `train/models.py:3633`
     - `train/models.py:3636`
     - `train/models.py:3676`
     - `train/models.py:3679`

3. **GT contacts / transition 信息 repo 里本来就有**
   - dataset sample 已经带 `contacts`、`ttc_td`、`ttc_td_valid`、`ttc_td_events`
   - code path:
     - `train/dataset.py:1003`
     - `train/dataset.py:1013`
   - GT contact channel 顺序是 `[L, R]`
     - `train/io.py:27`
     - `train/io.py:45`
   - transition 定义直接复用 repo-native threshold crossing：
     - `train/ttc.py:12`
     - `train/ttc.py:23`
     - `train/ttc.py:24`

4. **freerun 旧 export 已经足够做 white-box audit**
   - `run_freerun_cycles` 在 `log_contacts` 路径里已经导出：
     - `ContactGTPerC`
     - `ContactPlanPerC`
     - `ContactMeasPerC`
     - `ContactPlanLogits*`
     - `TTC*`
   - 并且 `metrics_per_step` 已经带：
     - `cycle`
     - `step_in_cycle`
     - `wrap_boundary_step`
     - `DirectGeoLocalDeg`
   - code path:
     - `train/validate/run_freerun_cycles.py:5619`
     - `train/validate/run_freerun_cycles.py:5622`
     - `train/validate/run_freerun_cycles.py:5630`
     - `train/validate/run_freerun_cycles.py:5633`
     - `train/validate/run_freerun_cycles.py:5744`
     - `train/validate/run_freerun_cycles.py:7204`
     - `train/validate/run_freerun_cycles.py:7205`
     - `train/validate/run_freerun_cycles.py:7206`

5. **这轮为了拿到 white-box 统计，没有改旧 runtime contract**
   - 新增的是 `train/validate/run_contact_plan_semantic_audit.py`
   - freerun 侧直接读取已有 `Walk_F_freerun_cycles.json`
   - teacher 侧复用 `FreeRunCycleRunner + _build_full_cycle_sample + trainer._rollout_sequence`
   - code path:
     - `train/validate/run_contact_plan_semantic_audit.py:17`
     - `train/validate/run_contact_plan_semantic_audit.py:534`
     - `train/validate/run_contact_plan_semantic_audit.py:622`
     - `train/validate/run_contact_plan_semantic_audit.py:745`
     - `train/validate/run_contact_plan_semantic_audit.py:755`
     - `train/validate/run_contact_plan_semantic_audit.py:1038`
   - 结论：
     - **没有改变旧 runtime/export contract**
     - **没有影响旧 recipe 默认行为**

6. **active semantic supervision hook 其实已经存在**
   - `w_contact_plan` 已能直接对 `contacts_plan_logits` vs `batch['contacts']` 做 BCE
   - code path:
     - `train/models.py:5976`
     - `train/models.py:5981`
     - `train/models.py:5986`
     - `train/models.py:5993`
   - 但这轮按 scope 先停在 audit，不新开 train lane

### 29.2 Audit design

teacher / freerun 视角分开做，目的就是把：

- planner 在 teacher-conditioned 下本来就 semantic 不准
- 还是 freerun closed-loop 才 drift

拆开。

本轮 audit 具体定义如下：

1. **candidates**
   - `baseline_replace`
   - `coadapt_4x`
   - `coadapt_4x_directonly_calibration_240`
   - `coadapt_4x_direct_plus_plan_ownership_240_noeventclock`
   - `coadapt_4x_direct_plus_plan_ownership_240`（作为 Event-Clock 反例）

2. **teacher-conditioned / teacher-driven**
   - source: `validate/teacher_batches/Walk_F_teacher.json`
   - 通过 `trainer._rollout_sequence(..., mode="mixed", tf_ratio=1.0)` 跑 teacher-conditioned white-box
   - 对 `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` 强制 runtime `event_clock=off`
   - 目的是让 teacher audit 对齐它真正的 no-Event-Clock consumed path

3. **freerun `model/model`**
   - 直接复用已有 `Walk_F_freerun_cycles.json`
   - 不重新改 eval pipeline
   - linkage 使用的是 `metrics_per_step[].DirectGeoLocalDeg`
   - 注意：这里的 linkage 是 per-step relation，不是重算 candidate-level 总表

4. **semantic metrics**
   - 主指标：`Brier`
   - 同时报告：`BCE`, `accuracy`, `precision`, `recall`, `F1`, `entropy`, `ECE10`
   - threshold:
     - `thr = 0.5`
   - calibration:
     - `ECE10` = 10-bin expected calibration error
   - breakdown:
     - 按 `[L, R]` 分开报告 channel F1

5. **transition / timing metrics**
   - event 定义：
     - touchdown: `prev < thr-hyst` 且 `cur >= thr`
     - liftoff: `prev >= thr+hyst` 且 `cur < thr`
   - 参数：
     - `thr = 0.5`
     - `hysteresis = 0.0`
     - `event_match_window = 2`
   - 指标：
     - precision / recall / F1
     - timing offset MAE
     - 必要时给出 `L/R` F1 breakdown

6. **linkage analysis**
   - step-wise `plan_brier_step` vs `DirectGeoLocalDeg` Pearson correlation
   - 按 plan error quantile 分 4 桶（Q1~Q4），报告 `Q_last / Q1` 的 direct error ratio
   - 看 `cycle0 -> last` 的共同漂移
   - 看 top-5 `step_in_cycle` hotspot 是否重合

7. **command**
   - 本轮实际跑法是：
   - `python3 -m train.validate.run_contact_plan_semantic_audit ... --event-clock-override coadapt_4x_direct_plus_plan_ownership_240_noeventclock=off --out debug_output/_tmp_cp015_tailk7_contact_plan_semantic_audit_20260406 --force`

### 29.3 Candidate table

| candidate / run | self-contained? | event_clock enabled? | eval mode | eval artifact path |
|---|---|---|---|---|
| `baseline_replace` | yes | `true` | `teacher` | `debug_output/_tmp_cp015_tailk7_contact_plan_semantic_audit_20260406/teacher_artifacts/baseline_replace.json` |
| `baseline_replace` | yes | `true` | `freerun` | `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/new70b_replace_lowdrift/Walk_F_freerun_cycles.json` |
| `coadapt_4x` | yes | `true` | `teacher` | `debug_output/_tmp_cp015_tailk7_contact_plan_semantic_audit_20260406/teacher_artifacts/coadapt_4x.json` |
| `coadapt_4x` | yes | `true` | `freerun` | `debug_output/_tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/eval_model_source/coadapt_allrot_interface_bestlr_longer_4x/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | `true` | `teacher` | `debug_output/_tmp_cp015_tailk7_contact_plan_semantic_audit_20260406/teacher_artifacts/coadapt_4x_directonly_calibration_240.json` |
| `coadapt_4x_directonly_calibration_240` | yes | `true` | `freerun` | `debug_output/_tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406/eval_model_source/coadapt_4x_directonly_calibration_240/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | `false` | `teacher` | `debug_output/_tmp_cp015_tailk7_contact_plan_semantic_audit_20260406/teacher_artifacts/coadapt_4x_direct_plus_plan_ownership_240_noeventclock.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | `false` | `freerun` | `debug_output/_tmp_cp015_tailk7_plan_ownership_calibration_20260406/eval_model_source/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240` | yes | `true` | `teacher` | `debug_output/_tmp_cp015_tailk7_contact_plan_semantic_audit_20260406/teacher_artifacts/coadapt_4x_direct_plus_plan_ownership_240.json` |
| `coadapt_4x_direct_plus_plan_ownership_240` | yes | `true` | `freerun` | `debug_output/_tmp_cp015_tailk7_plan_ownership_calibration_20260406/eval_model_source/coadapt_4x_direct_plus_plan_ownership_240/Walk_F_freerun_cycles.json` |

### 29.4 Semantic accuracy table

主指标记法：

- `plan prob err` = `Brier`
- `calibration` = `ECE10`

| candidate | eval mode | plan prob err (`Brier`) | `BCE` | accuracy | precision | recall | F1 | entropy | calibration (`ECE10`) | L F1 | R F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_replace` | `teacher` | `0.388564` | `1.214745` | `0.1552` | `0.2000` | `0.1771` | `0.1878` | `0.5654` | `0.5153` | `0.311` | `0.026` |
| `baseline_replace` | `freerun` | `0.232466` | `0.806534` | `0.4953` | `0.5492` | `0.4821` | `0.5135` | `0.6471` | `0.2958` | `0.693` | `0.009` |
| `coadapt_4x` | `teacher` | `0.577128` | `1.843493` | `0.0690` | `0.1163` | `0.1042` | `0.1099` | `0.4020` | `0.7130` | `0.110` | `0.110` |
| `coadapt_4x` | `freerun` | `0.198783` | `0.734798` | `0.5221` | `0.5816` | `0.4800` | `0.5260` | `0.6365` | `0.1429` | `0.634` | `0.284` |
| `coadapt_4x_directonly_calibration_240` | `teacher` | `0.577128` | `1.843493` | `0.0690` | `0.1163` | `0.1042` | `0.1099` | `0.4020` | `0.7130` | `0.110` | `0.110` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `0.198783` | `0.734798` | `0.5221` | `0.5816` | `0.4800` | `0.5260` | `0.6365` | `0.1429` | `0.634` | `0.284` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher` | `0.187829` | `0.708693` | `0.5230` | `0.5747` | `0.5208` | `0.5464` | `0.6722` | `0.0809` | `0.730` | `NA` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `0.194359` | `0.724637` | `0.5244` | `0.5859` | `0.4737` | `0.5239` | `0.6416` | `0.1297` | `0.642` | `0.258` |
| `coadapt_4x_direct_plus_plan_ownership_240` | `teacher` | `0.566309` | `1.790986` | `0.0632` | `0.1149` | `0.1042` | `0.1093` | `0.4172` | `0.7072` | `0.109` | `0.110` |
| `coadapt_4x_direct_plus_plan_ownership_240` | `freerun` | `0.194578` | `0.724890` | `0.5093` | `0.5714` | `0.4463` | `0.5012` | `0.6396` | `0.1328` | `0.642` | `0.167` |

semantic table 直接给出的白盒结论：

- `coadapt_4x` 与 `coadapt_4x_directonly_calibration_240` 的 teacher semantics **完全一样**
  - 说明 direct-only calibration **没有**改进 planner semantics
- raw coadapt family 在 teacher-conditioned 下非常差
  - `Brier ~ 0.577`
  - `F1 ~ 0.11`
  - `ECE ~ 0.71`
- `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` 在 teacher-conditioned 下语义质量**显著变好**
  - `Brier 0.187829`
  - `F1 0.546448`
  - `ECE10 0.080906`
- 但它的 thresholded channel 行为并不完美：
  - L channel 基本全被推到正侧，`L recall=1.0`
  - R channel 在 threshold 后基本不出正例，`R F1 = NA`
  - 这意味着：
    - **平均概率语义明显变好**
    - **但 event sharpness / threshold crossing 仍然弱**

### 29.5 Transition / timing table

说明：

- event count 很小，所以这张表只作为 secondary evidence
- `NA` 通常表示没有匹配事件或根本没有预测到 threshold crossing

| candidate | eval mode | TD precision | TD recall | TD F1 | TD offset MAE | TD F1 L/R | LO precision | LO recall | LO F1 | LO offset MAE | LO F1 L/R |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `baseline_replace` | `teacher` | `0.0000` | `0.0000` | `NA` | `NA` | `NA / NA` | `0.0000` | `0.0000` | `NA` | `NA` | `NA / NA` |
| `baseline_replace` | `freerun` | `0.0000` | `0.0000` | `NA` | `NA` | `NA / NA` | `0.0000` | `0.0000` | `NA` | `NA` | `NA / NA` |
| `coadapt_4x` | `teacher` | `0.0000` | `0.0000` | `NA` | `NA` | `NA / 0.667` | `0.3333` | `0.5000` | `0.4000` | `1.0000` | `NA / NA` |
| `coadapt_4x` | `freerun` | `0.0500` | `0.1000` | `0.0667` | `2.0000` | `NA / 0.667` | `0.2500` | `0.5000` | `0.3333` | `0.0000` | `NA / 0.667` |
| `coadapt_4x_directonly_calibration_240` | `teacher` | `0.0000` | `0.0000` | `NA` | `NA` | `NA / 0.667` | `0.3333` | `0.5000` | `0.4000` | `1.0000` | `NA / NA` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `0.0500` | `0.1000` | `0.0667` | `2.0000` | `NA / 0.667` | `0.2500` | `0.5000` | `0.3333` | `0.0000` | `NA / 0.667` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher` | `NA` | `0.0000` | `NA` | `NA` | `NA / NA` | `NA` | `0.0000` | `NA` | `NA` | `NA / NA` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `0.0500` | `0.1000` | `0.0667` | `2.0000` | `NA / 0.667` | `0.2500` | `0.5000` | `0.3333` | `1.6000` | `NA / 0.667` |
| `coadapt_4x_direct_plus_plan_ownership_240` | `teacher` | `0.0000` | `0.0000` | `NA` | `NA` | `NA / 0.667` | `0.3333` | `0.5000` | `0.4000` | `1.0000` | `NA / NA` |
| `coadapt_4x_direct_plus_plan_ownership_240` | `freerun` | `0.0500` | `0.1000` | `0.0667` | `2.0000` | `NA / NA` | `0.4500` | `0.9000` | `0.6000` | `1.5556` | `NA / 0.667` |

timing table 需要特别注意的点：

- `ownership_noeventclock` teacher 模式下，**虽然概率语义显著变好，但 thresholded touchdown / liftoff 事件几乎没被打出来**
- 也就是说：
  - semantic probability quality improved
  - 但 transition sharpness 还不够
- Event-Clock 版本在 freerun `liftoff` 上给出一个更高的 `LO F1 = 0.6`
  - 但这个 gain 没有转化成更好的 overall semantic calibration
  - 也没有转化成更好的 direct
  - 所以不能据此判定 Event-Clock 在这个 lane 里是净正贡献

### 29.6 Linkage table

说明：

- 这里 direct 关系看的都是 **per-step** `DirectGeoLocalDeg`
- `Q_last/Q1 direct > 1` 表示 plan semantic error 高分桶里 direct error 更大
- `cycle drift` 看的是 cycle0 到最后一个 cycle 的共同漂移

| candidate | eval mode | corr(plan error, direct) | `Q_last/Q1 direct` | cycle drift | top-5 plan-error SIC | top-5 direct-error SIC | overlap | linkage judgement |
|---|---|---:|---:|---|---|---|---|---|
| `baseline_replace` | `teacher` | `-0.3568` | `0.5810` | `Δplan 0.0000 / Δdirect 0.0000` | `73,72,74,71,75` | `48,47,46,49,45` | `NA` | 反向 / 不同位点 |
| `baseline_replace` | `freerun` | `-0.2470` | `0.8093` | `Δplan 0.0195 / Δdirect 0.0042` | `38,39,14,13,15` | `4,3,77,76,52` | `NA` | 不同位点，无同向热点 |
| `coadapt_4x` | `teacher` | `0.1731` | `0.9992` | `Δplan 0.0000 / Δdirect 0.0000` | `59,56,58,60,61` | `40,42,38,39,41` | `NA` | 弱相关，不共位 |
| `coadapt_4x` | `freerun` | `0.1093` | `1.0053` | `Δplan -0.0001 / Δdirect 0.0001` | `17,16,18,15,19` | `52,59,60,58,51` | `NA` | 基本无关系，热点分离 |
| `coadapt_4x_directonly_calibration_240` | `teacher` | `0.1724` | `0.9962` | `Δplan 0.0000 / Δdirect 0.0000` | `59,56,58,60,61` | `40,42,38,39,41` | `NA` | 弱相关，不共位 |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `0.4781` | `1.1747` | `Δplan -0.0001 / Δdirect 0.0003` | `17,16,18,15,19` | `22,24,58,31,27` | `NA` | 中等相关，但 SIC 热点不重合 |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher` | `0.4554` | `1.3167` | `Δplan 0.0000 / Δdirect 0.0000` | `12,13,11,14,41` | `40,42,38,39,41` | `41` | 有关联，但只有单个 SIC 交点 |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `0.4327` | `1.1294` | `Δplan 0.0019 / Δdirect 0.0006` | `17,16,18,15,19` | `58,22,24,31,59` | `NA` | 有均值相关，但热点分离 |
| `coadapt_4x_direct_plus_plan_ownership_240` | `teacher` | `0.1819` | `0.9640` | `Δplan 0.0000 / Δdirect 0.0000` | `59,62,61,60,78` | `40,42,38,39,41` | `NA` | 弱相关，不共位 |
| `coadapt_4x_direct_plus_plan_ownership_240` | `freerun` | `0.6114` | `1.3797` | `Δplan 0.0017 / Δdirect 0.0006` | `17,18,16,19,15` | `22,28,24,27,31` | `NA` | 相关增强，但热点仍分离 |

linkage table 收口成三句：

- freerun planner semantic drift **很小**
  - `coadapt_4x`: `Δplan -0.000061`
  - `directonly_240`: `Δplan -0.000061`
  - `ownership_noeventclock`: `Δplan +0.001928`
  - `ownership_eventclock`: `Δplan +0.001709`
- 也就是说，planner 的主要问题**不是** freerun closed-loop collapse
- 更关键的是：
  - 即使 `ownership_noeventclock` 把平均 planner semantics 拉上去
  - **direct hotspots 仍不和 plan-error hotspots 共位**
  - 所以 residual direct gap 不能用“平均 semantic error 还大”一条就解释完

### 29.7 Final judgement table

这里的 direct 参考结论仍继承 28.5 / 28.7：

- `coadapt_4x_directonly_calibration_240`: `DirectGeoLocalDeg = 0.172359`
- `coadapt_4x_direct_plus_plan_ownership_240_noeventclock`: `0.171490`
- `coadapt_4x_direct_plus_plan_ownership_240`: `0.174185`

| candidate | planner semantic quality acceptable? | planner timing quality acceptable? | semantic plan error 足以解释 direct gap? | ownership lane 是否提升 planner semantics? | active supervision likely needed? | recommended role |
|---|---|---|---|---|---|---|
| `baseline_replace` | `mixed`: teacher 差、freerun 中等 | no | `ref / n.a.` | `n.a.` | no | `production` |
| `coadapt_4x` | no | no | yes, primary on raw planner | `n.a.` | yes | `reject` |
| `coadapt_4x_directonly_calibration_240` | no | no | yes, primary on raw planner | no | yes | `research-only` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes on teacher average semantics; freerun moderate | no | no, not by average semantics alone | yes, strong on teacher / marginal on freerun | yes | `research-only` |
| `coadapt_4x_direct_plus_plan_ownership_240` | no | no / mixed | no | no, Event-Clock 抵消掉 semantic gain | yes | `reject` |

这张表对应的收口判断是：

- 对 raw coadapt family（`coadapt_4x` / `directonly_240`）：
  - **planner semantic anchor 缺失就是主问题**
- 但对 best ownership lane（`ownership_noeventclock`）：
  - planner semantics 平均上已经明显更好
  - direct 只从 `0.172359 -> 0.171490`
  - 说明 residual gap **不是平均 semantic error 一项就能解释**
  - secondary factor 已经变成：
    - semantic signal 的 threshold sharpness / usable event form 不足
    - 或 direct path 对 improved plan semantics 的利用仍不够高效

### 29.8 Mandatory judgement answers

1. **当前 `contacts_plan` 的主要问题到底是什么？**
   - **主判断：semantic 不准。**
   - 但要加一个 secondary factor：
     - 对 raw coadapt planner 来说，primary 就是 semantic 不准
     - 对 `ownership_noeventclock` 来说，semantic 平均质量已经明显改善，但 residual direct gap 仍在
     - 所以 secondary factor 是：
       - **semantic 虽然变好了，但还没有被 direct path 有效转成 probe 级 direct gain**

2. **`coadapt_4x_direct_plus_plan_ownership_240_noeventclock` 的小幅 direct 改善，是否真的来自 planner semantics 改善？**
   - **是，但只是一点点。**
   - 它的 teacher semantics 提升非常明显，说明 ownership 确实改变了 planner
   - 但 direct 只小幅改善，说明：
     - semantic improvement 是 real
     - 但它不是当前 residual direct gap 的完整解释

3. **planner 的问题主要出现在 teacher-conditioned 还是 freerun closed-loop？**
   - **主要在 teacher-conditioned。**
   - freerun semantic drift 很小，不支持“planner 一到 closed-loop 就明显 collapse”这个说法

4. **Event-Clock 在 planner semantics 上是帮忙还是添乱？**
   - **主判断：添乱。**
   - 证据：
     - `ownership_noeventclock` teacher semantics 显著更好
     - 同 lane 一旦把 Event-Clock 开回去，semantic gain 基本被打掉
     - direct 也比 noeventclock 更差

5. **基于这轮证据，是否已经足够支持“需要 active supervision / semantic anchor”？**
   - **是，已经足够。**
   - 而且 repo 里已有最小可复用 hook：
     - `w_contact_plan`
     - 直接对 `contacts_plan_logits` vs `batch['contacts']` 做 BCE

6. **下一步主优先级只能选一个，选什么？**
   - **只选：给 consumed `contact_plan` path 加 active semantic supervision。**
   - 不选：
     - `contact_meas`
     - direct head 结构
     - transplant 路线
   - 这也是对 28.7 里 `other` 的进一步收口：
     - **`other` 现在可以明确落成已有 `w_contact_plan` hook 的最小 active semantic anchor lane**

7. **为什么 `baseline_replace` 不需要这部分？**
   - 这轮证据支持的不是“所有能过 direct gate 的模型都必须依赖高质量 semantic `contacts_plan`”。
   - 更准确地说：
     - `baseline_replace` 的 direct 之所以好，**并不是因为它有一个特别强的 planner semantic path**
     - 相反，本轮 white-box audit 看到：
       - `baseline_replace` 的 teacher planner semantics 也不强
       - 它的 plan-error hotspot 与 direct-error hotspot 也不共位
     - 所以对 baseline 来说，`contacts_plan` 更像是 weak side-signal，而不是主承载通道
   - 但对当前 self-contained `coadapt` replacement family，不一样：
     - 继承 probe 已经证明 `plan=gt` 会显著救 direct
     - 这说明 **这条 family 的 direct path 对高质量 `contacts_plan` 依赖更强**
   - 因此：
     - “active semantic supervision” 不是 universal necessity
     - 它是 **当前 `coadapt` replacement lane 的最小因果修复杠杆**

8. **这是否意味着后续 debug 应该顺着这个信息继续？**
   - **是。**
   - 但后续 debug 的问题表述要更精确：
     - 不是再问“planner 需不需要 semantic anchor”这个总问题
     - 而是问：
       - **为什么 baseline 不依赖强 planner semantics 也能把 direct 做好，而 `coadapt` family 却明显依赖它？**
   - 这会把下一轮 debug 收紧成两个最小方向：
     - `A)` `coadapt` 里除了 `contacts_plan` 之外，本来应该支撑 direct 的内部表征 / cond dynamics 是否退化了
     - `B)` `coadapt` 的 direct head 是否把 `contacts_plan` 当成了过强捷径，而没有学到 baseline 那种更稳的非-plan path
   - 所以：
     - **可以继续沿这个信息 debug**
     - 但目标应从“继续证明要不要 supervision”切到
     - **解释 baseline-vs-coadapt 的 causal asymmetry**

### 29.9 Final one-paragraph judgement

把这轮 white-box audit 压成一句话：

- **raw self-contained coadapt planner 的 primary blocker 确实是 semantic anchor 缺失；`ownership_noeventclock` 已经证明 planner semantics 可以被明显拉好，但 direct 只小幅改善，且 direct hotspot 与 plan-error hotspot 基本不共位，因此当前证据已经足够支持“下一步优先给 consumed `contact_plan` path 加 active semantic supervision”，同时也说明 residual direct gap 已经不再是“平均 planner semantic error 还很差”这一条就能解释。**

## 30. baseline-vs-coadapt direct dependency asymmetry audit (2026-04-07)

这轮只回答一个问题：

- **为什么 `baseline_replace` 不依赖强 semantic planner 也能把 direct 做好，而 `coadapt` family 却明显依赖 `contacts_plan`？**

### 30.1 code facts 简写

- direct head 的 consumed hint 入口仍是：
  - `train/models.py`
  - `plan_in = contacts_plan`
  - `meas_in = contacts_meas`
  - plan/meas 都是 **concat 到 direct path**，不是纯 cond-side gating
- model 侧 direct override 仍走：
  - `model.direct_pose_plan_override`
  - `model.direct_pose_meas_override`
  - 见 `train/models.py` 的 direct head debug canonicalization
- freerun runtime 已原生支持：
  - `--direct_pose_plan_source {model,gt,softgt,zero}`
  - `--direct_pose_meas_source {model,gt,softgt,zero}`
  - 它们只改 **direct hint**，不改 `contacts_plan / contacts_err / lambda`
  - 见 `train/validate/run_freerun_cycles.py`
- 这轮 teacher-conditioned **没有改 `run_teacher_rollout.py`**：
  - 该脚本当前只显式暴露 `angvel / pose_hist` debug sourcing
  - 没有 direct `plan/meas` override CLI
  - 所以本轮 teacher-conditioned 最小口径直接采用：
    - `run_freerun_cycles --freerun_x_gt`
  - 这样 teacher 与 freerun 共用同一条 direct override runtime path，只把 carried `X` 切成 GT
- 这轮新增最小 helper：
  - `tools/audit_cp015_tailk7_direct_dependency_asymmetry.py`
  - 作用：
    - 复用现有 `run_freerun_cycles`
    - 批量跑 dependency matrix
    - 汇总 `summary.json / summary.md`
    - 对旧 probe JSON 自动做 metrics fallback（优先 joint direct series，没有则退回 `metrics_per_step.DirectGeoLocalDeg`）
- **没有改旧 runtime contract**
- **没有改旧 recipe 默认行为**
- artifact root：
  - `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/summary.json`
  - `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/summary.md`

### 30.2 audit design

- 比较 candidate：
  - `baseline_replace`
  - `coadapt_4x_directonly_calibration_240`
  - `coadapt_4x_direct_plus_plan_ownership_240_noeventclock`
- eval mode：
  - `freerun`
    - 正常 `model/model` closed loop
  - `teacher-conditioned`
    - 不是单独改 teacher exporter
    - 而是 `run_freerun_cycles --freerun_x_gt`
    - 让 direct override 走和 freerun 完全同一条 runtime path
- dependency matrix：
  - `plan=model, meas=model`
  - `plan=zero, meas=model`
  - `plan=gt, meas=model`
  - `plan=model, meas=zero`
  - `plan=model, meas=gt`
  - `plan=zero, meas=zero`
  - `plan=gt, meas=gt`
- 语义落点：
  - `model`：direct head 吃当前 runtime 自己产出的 hint
  - `gt`：direct head 吃 teacher soft contacts
  - `zero`：CLI 层传 `zero`，runtime canonicalize 成 direct override `"ignore"`，model 里最终落成 zero hint
- 强约束保持：
  - `contacts_meas_source` 固定 `model`
  - 不去动 `contacts_err / lambda / event loop`
  - 所以这轮看的是 **direct consumed hint dependency**，不是全系统 contact loop

### 30.3 candidate table

| candidate / run | self-contained? | event_clock enabled? | eval mode | override mode | eval artifact path |
|---|---|---|---|---|---|
| `baseline_replace` | yes | yes | `freerun` | `model/model` | `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/new70b_replace_lowdrift/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `freerun` | `zero/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/freerun/plan_zero__meas_model/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `freerun` | `gt/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/freerun/plan_gt__meas_model/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `freerun` | `model/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/freerun/plan_model__meas_zero/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `freerun` | `model/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/freerun/plan_model__meas_gt/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `freerun` | `zero/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/freerun/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `freerun` | `gt/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/freerun/plan_gt__meas_gt/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `teacher-conditioned` | `model/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `teacher-conditioned` | `zero/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/teacher_x_gt/plan_zero__meas_model/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `teacher-conditioned` | `gt/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/teacher_x_gt/plan_gt__meas_model/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `teacher-conditioned` | `model/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/teacher_x_gt/plan_model__meas_zero/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `teacher-conditioned` | `model/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/teacher_x_gt/plan_model__meas_gt/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `teacher-conditioned` | `zero/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/teacher_x_gt/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| `baseline_replace` | yes | yes | `teacher-conditioned` | `gt/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/teacher_x_gt/plan_gt__meas_gt/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `freerun` | `model/model` | `debug_output/_tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406/eval_model_source/coadapt_4x_directonly_calibration_240/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `freerun` | `zero/model` | `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/coadapt_4x_directonly_calibration_240/plan_zero__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `freerun` | `gt/model` | `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/coadapt_4x_directonly_calibration_240/plan_gt__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `freerun` | `model/zero` | `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/coadapt_4x_directonly_calibration_240/plan_model__meas_zero/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `freerun` | `model/gt` | `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/coadapt_4x_directonly_calibration_240/plan_model__meas_gt/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `freerun` | `zero/zero` | `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/coadapt_4x_directonly_calibration_240/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `freerun` | `gt/gt` | `debug_output/_tmp_cp015_tailk7_replace_plan_meas_causal_probe_20260406/coadapt_4x_directonly_calibration_240/plan_gt__meas_gt/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `teacher-conditioned` | `model/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `teacher-conditioned` | `zero/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_zero__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `teacher-conditioned` | `gt/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_gt__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `teacher-conditioned` | `model/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_model__meas_zero/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `teacher-conditioned` | `model/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_model__meas_gt/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `teacher-conditioned` | `zero/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| `coadapt_4x_directonly_calibration_240` | yes | yes | `teacher-conditioned` | `gt/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_gt__meas_gt/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `freerun` | `model/model` | `debug_output/_tmp_cp015_tailk7_plan_ownership_calibration_20260406/eval_model_source/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `freerun` | `zero/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/freerun/plan_zero__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `freerun` | `gt/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/freerun/plan_gt__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `freerun` | `model/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/freerun/plan_model__meas_zero/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `freerun` | `model/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/freerun/plan_model__meas_gt/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `freerun` | `zero/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/freerun/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `freerun` | `gt/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/freerun/plan_gt__meas_gt/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `teacher-conditioned` | `model/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `teacher-conditioned` | `zero/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_zero__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `teacher-conditioned` | `gt/model` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_gt__meas_model/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `teacher-conditioned` | `model/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_model__meas_zero/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `teacher-conditioned` | `model/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_model__meas_gt/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `teacher-conditioned` | `zero/zero` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | yes | no | `teacher-conditioned` | `gt/gt` | `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_gt__meas_gt/Walk_F_freerun_cycles.json` |

### 30.4 dependency result table

| candidate | eval mode | `plan/meas` override | `DirectGeoLocalDeg` | delta vs default | conclusion label |
|---|---|---|---:|---:|---|
| `baseline_replace` | `freerun` | `gt/gt` | `0.151570` | `-0.002175` | `non-plan path robust` |
| `baseline_replace` | `freerun` | `gt/model` | `0.150220` | `-0.003526` | `mild sensitivity` |
| `baseline_replace` | `freerun` | `model/gt` | `0.151869` | `-0.001877` | `non-plan path robust` |
| `baseline_replace` | `freerun` | `model/model` | `0.153746` | `+0.000000` | `default` |
| `baseline_replace` | `freerun` | `model/zero` | `0.150492` | `-0.003253` | `mild sensitivity` |
| `baseline_replace` | `freerun` | `zero/model` | `0.151184` | `-0.002562` | `non-plan path robust` |
| `baseline_replace` | `freerun` | `zero/zero` | `0.151184` | `-0.002562` | `non-plan path robust` |
| `baseline_replace` | `teacher-conditioned` | `gt/gt` | `0.151570` | `+0.001078` | `non-plan path robust` |
| `baseline_replace` | `teacher-conditioned` | `gt/model` | `0.150220` | `-0.000273` | `non-plan path robust` |
| `baseline_replace` | `teacher-conditioned` | `model/gt` | `0.151869` | `+0.001376` | `non-plan path robust` |
| `baseline_replace` | `teacher-conditioned` | `model/model` | `0.150492` | `+0.000000` | `default` |
| `baseline_replace` | `teacher-conditioned` | `model/zero` | `0.150492` | `+0.000000` | `non-plan path robust` |
| `baseline_replace` | `teacher-conditioned` | `zero/model` | `0.151184` | `+0.000691` | `non-plan path robust` |
| `baseline_replace` | `teacher-conditioned` | `zero/zero` | `0.151184` | `+0.000691` | `non-plan path robust` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `gt/gt` | `0.147868` | `-0.023622` | `plan-sensitive` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `gt/model` | `0.154513` | `-0.016976` | `plan-sensitive` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `model/gt` | `0.161886` | `-0.009603` | `mild sensitivity` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `model/model` | `0.171490` | `+0.000000` | `default` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `model/zero` | `0.171490` | `+0.000000` | `non-plan path robust` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `zero/model` | `0.196174` | `+0.024685` | `collapsed without plan` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `zero/zero` | `0.196174` | `+0.024685` | `collapsed without plan` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `gt/gt` | `0.147868` | `-0.023622` | `plan-sensitive` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `gt/model` | `0.154513` | `-0.016976` | `plan-sensitive` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `model/gt` | `0.161886` | `-0.009603` | `mild sensitivity` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `model/model` | `0.171490` | `+0.000000` | `default` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `model/zero` | `0.171490` | `+0.000000` | `non-plan path robust` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `zero/model` | `0.196174` | `+0.024685` | `collapsed without plan` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `zero/zero` | `0.196174` | `+0.024685` | `collapsed without plan` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `gt/gt` | `0.148023` | `-0.024335` | `plan-sensitive` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `gt/model` | `0.153369` | `-0.018990` | `plan-sensitive` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `model/gt` | `0.163949` | `-0.008410` | `mild sensitivity` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `model/model` | `0.172359` | `+0.000000` | `default` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `model/zero` | `0.172359` | `+0.000000` | `non-plan path robust` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `zero/model` | `0.198338` | `+0.025979` | `collapsed without plan` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `zero/zero` | `0.198338` | `+0.025979` | `collapsed without plan` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `gt/gt` | `0.148023` | `-0.024335` | `plan-sensitive` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `gt/model` | `0.153369` | `-0.018990` | `plan-sensitive` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `model/gt` | `0.163949` | `-0.008410` | `mild sensitivity` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `model/model` | `0.172359` | `+0.000000` | `default` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `model/zero` | `0.172359` | `+0.000000` | `non-plan path robust` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `zero/model` | `0.198338` | `+0.025979` | `collapsed without plan` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `zero/zero` | `0.198338` | `+0.025979` | `collapsed without plan` |

### 30.5 asymmetry table

| candidate | baseline-vs-coadapt 差异 | non-plan direct path 是否强 | 对 `plan` 的依赖是否主导 | 对 `meas` 的依赖是否主导 | 主要问题发生在 teacher 还是 freerun | 是否支持 “coadapt 把 `plan` 当成过强捷径” |
|---|---|---|---|---|---|---|
| `baseline_replace` | baseline reference | `strong` | no | no | `similar` | no |
| `coadapt_4x_directonly_calibration_240` | weaker non-plan path than baseline | `weak` | yes | no | `similar` | yes |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | weaker non-plan path than baseline | `weak` | yes | no | `similar` | yes |

### 30.6 supporting profile table

这里只保留最有解释力的 step-wise / SIC-wise supporting rows。

| candidate | eval mode | override | cycle mean deltas | top `|Δ|` SICs |
|---|---|---|---|---|
| `baseline_replace` | `freerun` | `gt/model` | `c1:-0.0001, c2:-0.0013, c3:-0.0026, c4:-0.0035` | `11:-0.0095, 9:-0.0074, 7:-0.0072, 17:-0.0068, 54:-0.0062` |
| `baseline_replace` | `freerun` | `model/gt` | `c1:+0.0015, c2:+0.0003, c3:-0.0010, c4:-0.0019` | `3:+0.0101, 71:-0.0086, 4:+0.0085, 70:-0.0075, 72:-0.0073` |
| `baseline_replace` | `freerun` | `zero/model` | `c1:+0.0008, c2:-0.0004, c3:-0.0017, c4:-0.0026` | `71:-0.0075, 70:-0.0065, 72:-0.0062, 66:-0.0057, 11:-0.0051` |
| `baseline_replace` | `teacher-conditioned` | `gt/model` | `c1:-0.0003, c2:-0.0003, c3:-0.0003, c4:-0.0003` | `3:-0.0036, 4:-0.0033, 38:-0.0026, 5:-0.0024, 7:-0.0023` |
| `baseline_replace` | `teacher-conditioned` | `model/gt` | `c1:+0.0014, c2:+0.0014, c3:+0.0014, c4:+0.0014` | `4:+0.0113, 3:+0.0112, 8:+0.0106, 9:+0.0099, 7:+0.0088` |
| `baseline_replace` | `teacher-conditioned` | `zero/model` | `c1:+0.0007, c2:+0.0007, c3:+0.0007, c4:+0.0007` | `53:+0.0041, 63:+0.0040, 66:-0.0038, 51:+0.0038, 52:+0.0038` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `gt/model` | `c1:-0.0170, c2:-0.0170, c3:-0.0170, c4:-0.0170` | `22:-0.0869, 28:-0.0810, 21:-0.0806, 31:-0.0774, 23:-0.0717` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `model/gt` | `c1:-0.0096, c2:-0.0096, c3:-0.0096, c4:-0.0096` | `18:-0.0818, 17:-0.0793, 19:-0.0774, 24:-0.0628, 11:-0.0601` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `freerun` | `zero/model` | `c1:+0.0247, c2:+0.0247, c3:+0.0247, c4:+0.0247` | `85:+0.1053, 13:+0.0810, 11:+0.0745, 83:+0.0709, 84:+0.0654` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `gt/model` | `c1:-0.0170, c2:-0.0170, c3:-0.0170, c4:-0.0170` | `22:-0.0869, 28:-0.0810, 21:-0.0806, 31:-0.0774, 23:-0.0717` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `model/gt` | `c1:-0.0096, c2:-0.0096, c3:-0.0096, c4:-0.0096` | `18:-0.0818, 17:-0.0793, 19:-0.0774, 24:-0.0628, 11:-0.0601` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned` | `zero/model` | `c1:+0.0247, c2:+0.0247, c3:+0.0247, c4:+0.0247` | `85:+0.1053, 13:+0.0810, 11:+0.0745, 83:+0.0709, 84:+0.0654` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `gt/model` | `c1:-0.0190, c2:-0.0190, c3:-0.0190, c4:-0.0190` | `22:-0.0907, 21:-0.0879, 28:-0.0871, 23:-0.0753, 31:-0.0706` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `model/gt` | `c1:-0.0084, c2:-0.0084, c3:-0.0084, c4:-0.0084` | `17:-0.0831, 18:-0.0830, 19:-0.0759, 11:-0.0665, 24:-0.0638` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `zero/model` | `c1:+0.0260, c2:+0.0260, c3:+0.0260, c4:+0.0260` | `85:+0.1101, 13:+0.0842, 83:+0.0728, 84:+0.0723, 11:+0.0702` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `gt/model` | `c1:-0.0190, c2:-0.0190, c3:-0.0190, c4:-0.0190` | `22:-0.0907, 21:-0.0879, 28:-0.0871, 23:-0.0753, 31:-0.0706` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `model/gt` | `c1:-0.0084, c2:-0.0084, c3:-0.0084, c4:-0.0084` | `17:-0.0831, 18:-0.0830, 19:-0.0759, 11:-0.0665, 24:-0.0638` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `zero/model` | `c1:+0.0260, c2:+0.0260, c3:+0.0260, c4:+0.0260` | `85:+0.1101, 13:+0.0842, 83:+0.0728, 84:+0.0723, 11:+0.0702` |

这张 supporting table 的关键收口是：

- `coadapt` 两个 self-contained candidate 的 sensitivity **按 cycle 几乎是常数平移**
  - 不是只在 later-cycle 才突然放大
  - 所以这不是 freerun drift 才暴露出来的 dependency
- `baseline_replace` 则所有 override 都只有 millidegree 级变化
  - `zero/model`
  - `model/gt`
  - `zero/zero`
  - 全都不构成 structural dependency

### 30.7 asymmetry interpretation

这轮 causal asymmetry 的最小白盒结论可以直接压成四句：

1. **`baseline_replace` 的 non-plan direct path 明显更强。**
   - freerun：
     - `plan_score = 0.003526`
     - `meas_score = 0.003253`
     - `zero/zero delta = -0.002562`
   - teacher-conditioned：
     - `plan_score = 0.001078`
     - `meas_score = 0.001376`
     - `zero/zero delta = +0.000691`
   - 也就是说：
     - 把 `plan/meas` 全清掉，direct 基本还成立
     - `baseline` 确实主要靠 **non-plan path / cond-direct_feat**

2. **`coadapt_4x_directonly_calibration_240` 已经对 consumed `contacts_plan` 形成过强依赖。**
   - freerun / teacher-conditioned 两边完全同口径：
     - `plan=gt, meas=model -> -0.018990`
     - `plan=zero, meas=model -> +0.025979`
     - `model=zero` 对 meas：
       - `model/zero -> +0.000000`
     - `zero/zero` 与 `zero/model` 完全相同
   - 这说明：
     - 当前 direct lane **几乎不吃 `meas`**
     - 没有 `plan` 时，剩下的 non-plan path 直接失效

3. **`ownership_noeventclock` 并没有把这种 dependency structure 真正改掉。**
   - 它只是把数值稍微缓和了一点：
     - `plan=gt, meas=model`：
       - `-0.018990 -> -0.016976`
     - `plan=zero, meas=model`：
       - `+0.025979 -> +0.024685`
   - 但结构完全没变：
     - `plan` 仍是 dominant dependency
     - `meas` 仍基本不主导
     - `zero/zero` 仍等于 `zero/model`

4. **这种 asymmetry 主要不是 freerun drift 问题。**
   - 在这轮口径里：
     - `freerun`
     - `teacher-conditioned (--freerun_x_gt)`
   - 对 `coadapt` 两个 candidate，dependency matrix 基本逐项相同
   - 因此更自然的解释是：
     - **shortcut structure 已经写进了 direct path 本身**
     - 不是 closed-loop 才把它放大的 secondary failure

### 30.8 final judgement table

| candidate | direct 主要依赖 source 是什么 | baseline 为什么能不靠 semantic planner | coadapt 为什么不行 | `ownership_noeventclock` 是否缓和了这个 asymmetry | recommended next role |
|---|---|---|---|---|---|
| `baseline_replace` | `non-plan path / cond-direct_feat`；`plan/meas` 只是 weak side-signal | 因为 `zero/zero` 仍只动 `~0.0026`（freerun）/ `~0.0007`（teacher） | `n.a.` | `n.a.` | `production` |
| `coadapt_4x_directonly_calibration_240` | `contacts_plan` dominant；`meas` 基本不被用到 | `n.a.` | 因为 non-plan path 已经弱化，direct head 把 `plan` 当成主捷径；一旦 `plan=zero` 直接 `+0.025979` collapse | no | `research-only` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | 仍是 `contacts_plan` dominant；`meas` 仍非主导 | `n.a.` | planner semantics 虽然更好，但 direct dependency structure 没被改掉；non-plan path 仍弱 | marginal only (`~0.001-0.002` 级缓和，不是结构性修复) | `research-only` |

### 30.9 mandatory judgement answers

1. **`baseline_replace` 的 direct，主要是不是靠 non-plan path 就能成立？**
   - **是。**
   - 最强证据是：
     - freerun `zero/zero delta = -0.002562`
     - teacher-conditioned `zero/zero delta = +0.000691`
   - 这已经足够把它归到：
     - **non-plan path robust**

2. **`coadapt_4x_directonly_calibration_240` 的 direct，是否已经对 `contacts_plan` 形成过强依赖？**
   - **是，而且依赖已经是 structural，不是轻微偏好。**
   - 证据：
     - `plan=zero, meas=model -> +0.025979`
     - `plan=gt, meas=model -> -0.018990`
     - `model/zero -> +0.000000`
   - 所以这不是 “plan 有帮助”；
   - 而是 **没有 plan 就明显坏，有 GT plan 就明显被救。**

3. **这种依赖更偏 `plan` 还是更偏 `meas`？**
   - **明显更偏 `plan`。**
   - `coadapt_directonly_240`：
     - `plan_score = 0.025979`
     - `meas_score = 0.008410`
   - `ownership_noeventclock`：
     - `plan_score = 0.024685`
     - `meas_score = 0.009603`
   - 同时：
     - `model/zero == model/model`
     - `zero/zero == zero/model`
   - 这进一步说明：
     - **direct 的 consumed `meas` path 在当前 lane 上几乎不是主因。**

4. **主要问题出现在 teacher-conditioned 还是 freerun？**
   - **不是 freerun-specific；teacher-conditioned 已经足够暴露。**
   - 更准确地说：
     - 两个 `coadapt` candidate 在 `freerun` 与 `teacher-conditioned` 上的 dependency matrix 几乎完全相同
   - 所以如果一定二选一：
     - **teacher-conditioned 就已经成立**
   - freerun 并没有把这种 dependency structure 再明显放大

5. **`ownership_noeventclock` 改善的到底是 planner semantics，还是也顺带降低了 direct 对 `plan` 的脆弱依赖？**
   - **主要改善的是 planner semantics；对 direct plan fragility 只带来极小缓和。**
   - 它没有把结构从：
     - `plan-dominant`
   - 变成：
     - `non-plan robust`
   - 所以这轮不能把 ownership 解释成 “已经修好了 direct dependency asymmetry”

6. **基于这轮证据，下一步主优先级只能选一个，选什么？**
   - **只选：debug non-plan direct path / cond direct feat 退化。**
   - 不选：
     - `contact_meas`
       - 因为 `meas_zero` 基本不伤 direct
     - 只给 consumed `contact_plan` path 加 active semantic supervision
       - 这也许还是 repair lever
       - 但**不能解释 baseline-vs-coadapt asymmetry**
       - ownership 已经证明“平均 semantics 更好”并不会自动把这种 dependency structure 改掉
   - 这轮最自然的 root-cause priority 已经变成：
     - **为什么 `coadapt` 的 non-plan direct path 比 baseline 弱这么多**
     - **为什么 direct head 会把 `plan` 学成过强捷径**

### 30.10 one-paragraph judgement

- **这轮 dependency asymmetry audit 已经把 baseline-vs-coadapt 的差异压实了：`baseline_replace` 的 direct 基本由 non-plan path / cond-direct_feat 支撑，`plan/meas` 只起弱辅助作用；而 `coadapt_4x_directonly_calibration_240` 与 `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` 都表现出强 `plan` 依赖、弱 non-plan path、近乎无 `meas` 依赖，且这种结构在 teacher-conditioned 与 freerun 下几乎完全相同，因此 residual 问题更像是 direct path 内部的 shortcut / non-plan degeneration，而不是 planner freerun drift 或 `contact_meas` 路线。**

### 30.11 minimal white-box code facts (`non-plan direct path / direct_feat`)

1. **direct head 的 non-plan path 现在具体怎么进来的**
   - `train/models.py`
     - `direct_pose_feat_source` 默认值就是 `cond`
     - relevant ckpt/runtime 解析结果也都是 `cond`
     - `direct_feat` 先取 `cond`，再拼上 `time_pe_direct`
   - 当前三条 candidate 的 runtime layout 一致：
     - `direct_pose_feat_source = cond`
     - `direct_pose_meas_mode = concat`
     - `direct_pose_split_enable = true`
     - `direct_pose_arm_split_enable = true`
     - `direct_pose_factorized_readout_enable = false`
     - `direct_pose_input_adapter_enable = false`
   - 因此当前 direct head 的主链路是：
     - `cond (+ direct time pe) -> direct_feat`
     - `direct_in = concat([direct_feat, plan_in, meas_in])`
     - `direct_pose_head(_arm) -> trunk_hidden`
     - `direct_pose_arm_proj / direct_pose_out_arm -> out_direct`

2. **teacher / freerun 各走哪条 runtime path**
   - 入口都复用 `train.validate.run_freerun_cycles`
   - `teacher-conditioned`：
     - `train.validate.run_freerun_cycles --freerun_x_gt`
   - `freerun model/model`：
     - 同一个 entry，不加 `--freerun_x_gt`
   - 这轮没有新开任何训练 lane，也没有改 runtime data flow

3. **这轮复用了哪些现有 probe / override / export**
   - 已有 override：
     - `--direct_pose_plan_source`
     - `--direct_pose_meas_source`
   - 已有 direct probe/export：
     - `--export_direct_arm_probe`
     - `--export_joint_direct_geolocal_series`
   - `direct_arm_probe` 已经能导出：
     - `direct_in`
     - `trunk_hidden`
     - `proj_pre0`
     - `out_in`
     - `arm_out`
   - 这轮 white-box 主用的是：
     - `direct_in`
     - `trunk_hidden`
     - `per_step_direct_geolocal_deg`

4. **为了拿到统计，这轮新增了什么**
   - 只新增一个最小 helper：
     - `tools/audit_cp015_tailk7_nonplan_direct_path_whitebox.py`
   - 它只做两件事：
     - 用现有 `run_freerun_cycles` 跑 fresh probe artifacts
     - 解析现有 json/export，汇总 `direct_feat` / `direct_in` / `trunk_hidden` 统计
   - 它不会改旧 runtime contract
   - 不影响旧 recipe 默认行为
   - 也没有改 model / trainer 默认逻辑

### 30.12 audit design

1. **candidate**
   - `baseline_replace`
   - `coadapt_4x_directonly_calibration_240`
   - `coadapt_4x_direct_plus_plan_ownership_240_noeventclock`（对照）

2. **teacher / freerun 怎么跑**
   - dependency matrix：
     - 继承既有 `tools/audit_cp015_tailk7_direct_dependency_asymmetry.py`
     - artifact root：
       - `debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/`
   - white-box：
     - fresh rerun `tools/audit_cp015_tailk7_nonplan_direct_path_whitebox.py --force`
     - artifact root：
       - `debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/`
   - 这里 white-box 特意强制 fresh probe，不再复用历史 `baseline_replace` base eval，避免把旧 artifact 混进来

3. **override matrix 怎么定义**
   - dependency audit 主矩阵：
     - `plan=model, meas=model`
     - `plan=zero, meas=model`
     - `plan=model, meas=zero`
     - `plan=zero, meas=zero`
   - 同时继承 support 证据：
     - `plan=gt, meas=model`
     - `plan=model, meas=gt`
     - `plan=gt, meas=gt`
   - white-box 最小矩阵只跑：
     - `model/model`
     - `zero/zero`
   - 这是因为 A 部分 dependency structure 已经有完整矩阵；B 部分只需要最小 non-plan-only vs default 对照

4. **`plan=zero` / `meas=zero` 在代码里具体落到哪里**
   - `plan=zero`：
     - `train.validate.run_freerun_cycles --direct_pose_plan_source zero`
     - 最终让 direct head 消费的 `plan_in` 变成零向量
   - `meas=zero`：
     - `train.validate.run_freerun_cycles --direct_pose_meas_source zero`
     - 最终让 `meas_in` 变成零向量
   - 因为 `direct_pose_meas_mode = concat`
     - 所以它们都直接落在 `direct_in = concat([direct_feat, plan_in, meas_in])` 这一层

5. **white-box 统计怎么定义**
   - `direct_feat`
     - 直接从 `direct_in` 前半段切片出来
     - 维度由 runtime layout 推断：
       - `cond_dim = 7`
       - `direct_pose_time_pe_dim = 32`
       - `direct_feat_dim = 39`
   - `plan` / `meas`
     - 都是 `contact_dim = 2`
   - `trunk_hidden`
     - 来自 `direct_pose_head_arm`（若无则回退 `direct_pose_head`）输出
   - 主统计：
     - `rms`
     - `std`
     - `step_delta = rms(x_t - x_{t-1})`
     - teacher-vs-freerun paired drift（同 `cycle`, `step_in_cycle` 对齐）
   - 额外最小 causality support：
     - `SIC 0-10 / 11-21 / 22-43` bucket direct error delta

### 30.13 candidate table

| candidate / run | self-contained? | event_clock enabled? | eval mode | override mode | eval artifact path |
|---|---|---|---|---|---|
| baseline_replace | yes | yes | teacher-conditioned | model/model | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/baseline_replace/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` |
| baseline_replace | yes | yes | teacher-conditioned | zero/zero | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/baseline_replace/teacher_x_gt/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| baseline_replace | yes | yes | freerun | model/model | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/baseline_replace/freerun/plan_model__meas_model/Walk_F_freerun_cycles.json` |
| baseline_replace | yes | yes | freerun | zero/zero | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/baseline_replace/freerun/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| coadapt_4x_directonly_calibration_240 | yes | yes | teacher-conditioned | model/model | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` |
| coadapt_4x_directonly_calibration_240 | yes | yes | teacher-conditioned | zero/zero | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| coadapt_4x_directonly_calibration_240 | yes | yes | freerun | model/model | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/coadapt_4x_directonly_calibration_240/freerun/plan_model__meas_model/Walk_F_freerun_cycles.json` |
| coadapt_4x_directonly_calibration_240 | yes | yes | freerun | zero/zero | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/coadapt_4x_directonly_calibration_240/freerun/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | yes | no | teacher-conditioned | model/model | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | yes | no | teacher-conditioned | zero/zero | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | yes | no | freerun | model/model | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/freerun/plan_model__meas_model/Walk_F_freerun_cycles.json` |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | yes | no | freerun | zero/zero | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_nonplan_direct_whitebox_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/freerun/plan_zero__meas_zero/Walk_F_freerun_cycles.json` |

### 30.14 non-plan dependency table

| candidate | eval mode | override mode | DirectGeoLocalDeg | delta vs default | label |
|---|---|---|---:|---:|---|
| baseline_replace | teacher-conditioned | model/model | 0.150492 | 0.000000 | default |
| baseline_replace | teacher-conditioned | zero/model | 0.151184 | 0.000691 | non-plan path robust |
| baseline_replace | teacher-conditioned | model/zero | 0.150492 | 0.000000 | non-plan path robust |
| baseline_replace | teacher-conditioned | zero/zero | 0.151184 | 0.000691 | non-plan path robust |
| baseline_replace | freerun | model/model | 0.153746 | 0.000000 | default |
| baseline_replace | freerun | zero/model | 0.151184 | -0.002562 | non-plan path robust |
| baseline_replace | freerun | model/zero | 0.150492 | -0.003253 | mild sensitivity |
| baseline_replace | freerun | zero/zero | 0.151184 | -0.002562 | non-plan path robust |
| coadapt_4x_directonly_calibration_240 | teacher-conditioned | model/model | 0.172359 | 0.000000 | default |
| coadapt_4x_directonly_calibration_240 | teacher-conditioned | zero/model | 0.198338 | 0.025979 | collapsed without plan |
| coadapt_4x_directonly_calibration_240 | teacher-conditioned | model/zero | 0.172359 | 0.000000 | non-plan path robust |
| coadapt_4x_directonly_calibration_240 | teacher-conditioned | zero/zero | 0.198338 | 0.025979 | collapsed without plan |
| coadapt_4x_directonly_calibration_240 | freerun | model/model | 0.172359 | 0.000000 | default |
| coadapt_4x_directonly_calibration_240 | freerun | zero/model | 0.198338 | 0.025979 | collapsed without plan |
| coadapt_4x_directonly_calibration_240 | freerun | model/zero | 0.172359 | 0.000000 | non-plan path robust |
| coadapt_4x_directonly_calibration_240 | freerun | zero/zero | 0.198338 | 0.025979 | collapsed without plan |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | teacher-conditioned | model/model | 0.171490 | 0.000000 | default |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | teacher-conditioned | zero/model | 0.196174 | 0.024685 | collapsed without plan |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | teacher-conditioned | model/zero | 0.171490 | 0.000000 | non-plan path robust |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | teacher-conditioned | zero/zero | 0.196174 | 0.024685 | collapsed without plan |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | freerun | model/model | 0.171490 | 0.000000 | default |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | freerun | zero/model | 0.196174 | 0.024685 | collapsed without plan |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | freerun | model/zero | 0.171490 | 0.000000 | non-plan path robust |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | freerun | zero/zero | 0.196174 | 0.024685 | collapsed without plan |

最小读法可以直接压成三条：

1. **`baseline_replace` 在 `plan/meas` 被拿掉后几乎不塌。**
   - teacher-conditioned：
     - `zero/zero delta = +0.000691`
   - freerun：
     - `zero/zero delta = -0.002562`
   - 所以它可以归入：
     - `non-plan robust`

2. **`coadapt_4x_directonly_calibration_240` 与 `ownership_noeventclock` 都是“去掉 `plan` 就塌”。**
   - `directonly_240`：
     - `zero/model = zero/zero = +0.025979`
     - `model/zero = model/model = +0.000000`
   - `ownership_noeventclock`：
     - `zero/model = zero/zero = +0.024685`
     - `model/zero = model/model = +0.000000`
   - 所以当前主要拐杖是：
     - `plan`
   - 不是：
     - `meas`

3. **这不是 freerun-only failure。**
   - 三个 candidate 的 teacher-conditioned / freerun dependency matrix 几乎逐项一致
   - 所以 shortcut structure 在 teacher-conditioned 就已经成立

补充 support（沿用这轮已知 direct hint probe，不回头重证）：

- `coadapt_4x_directonly_calibration_240`
  - `plan=gt, meas=model -> 0.153369`
  - `meas=gt -> 0.163949`
  - `gt/gt -> 0.148023`
- 对应 delta 结构也一致：
  - `plan_gt_delta = -0.018990`
  - `meas_gt_delta = -0.008410`
- 所以即使加 GT 支持，`plan` 仍比 `meas` 更像 direct 的主拐杖

### 30.15 direct feature white-box table

| candidate | eval mode | override | direct_feat rms/std | trunk_hidden rms/std | trunk_hidden step Δ | dynamic range shrink? | step instability? | freerun drift? | supports non-plan degeneration? |
|---|---|---|---|---|---:|---|---|---|---|
| baseline_replace | teacher-conditioned | model/model | 0.6601 / 0.5865 | 0.2745 / 0.2537 | 0.0363 | ref | ref | feat 0.0000, trunk 0.0000 | no |
| baseline_replace | teacher-conditioned | zero/zero | 0.6601 / 0.5865 | 0.2746 / 0.2538 | 0.0363 | ref | ref | feat 0.0000, trunk 0.0000 | no |
| baseline_replace | freerun | model/model | 0.6601 / 0.5865 | 0.2745 / 0.2537 | 0.0363 | ref | ref | feat 0.0000, trunk 0.0000 | no |
| baseline_replace | freerun | zero/zero | 0.6601 / 0.5865 | 0.2746 / 0.2538 | 0.0363 | ref | ref | feat 0.0000, trunk 0.0000 | no |
| coadapt_4x_directonly_calibration_240 | teacher-conditioned | model/model | 0.6601 / 0.5865 | 0.2621 / 0.2139 | 0.0359 | yes | no | feat 0.0000, trunk 0.0000 | yes |
| coadapt_4x_directonly_calibration_240 | teacher-conditioned | zero/zero | 0.6601 / 0.5865 | 0.2613 / 0.2135 | 0.0359 | yes | no | feat 0.0000, trunk 0.0000 | yes |
| coadapt_4x_directonly_calibration_240 | freerun | model/model | 0.6601 / 0.5865 | 0.2621 / 0.2139 | 0.0359 | yes | no | feat 0.0000, trunk 0.0000 | yes |
| coadapt_4x_directonly_calibration_240 | freerun | zero/zero | 0.6601 / 0.5865 | 0.2613 / 0.2135 | 0.0359 | yes | no | feat 0.0000, trunk 0.0000 | yes |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | teacher-conditioned | model/model | 0.6601 / 0.5865 | 0.2624 / 0.2141 | 0.0359 | yes | no | feat 0.0000, trunk 0.0000 | yes |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | teacher-conditioned | zero/zero | 0.6601 / 0.5865 | 0.2616 / 0.2137 | 0.0359 | yes | no | feat 0.0000, trunk 0.0000 | yes |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | freerun | model/model | 0.6601 / 0.5865 | 0.2624 / 0.2141 | 0.0359 | yes | no | feat 0.0000, trunk 0.0000 | yes |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | freerun | zero/zero | 0.6601 / 0.5865 | 0.2616 / 0.2137 | 0.0359 | yes | no | feat 0.0000, trunk 0.0000 | yes |

这个 white-box 表最重要的不是 “数字更好看”，而是 **差异出现在什么位置**：

1. **`direct_feat` 本身没有塌。**
   - 三个 candidate、两种 mode、两种 override 下：
     - `direct_feat rms/std` 都是 `0.6601 / 0.5865`
   - 所以这轮没有证据支持：
     - `cond-direct input` 本身已经变弱

2. **差异出现在 `direct_feat -> direct_pose_head -> trunk_hidden` 这一段。**
   - `baseline_replace`：
     - `trunk_hidden rms/std ~= 0.2745 / 0.2537`
   - `directonly_240`：
     - `trunk_hidden rms/std ~= 0.2621 / 0.2139`
   - `ownership_noeventclock`：
     - `trunk_hidden rms/std ~= 0.2624 / 0.2141`
   - 换成 baseline 比例看：
     - `directonly_240`
       - `trunk_hidden_rms ~= 95.5%`
       - `trunk_hidden_std ~= 84.3%`
     - `ownership_noeventclock`
       - `trunk_hidden_rms ~= 95.6%`
       - `trunk_hidden_std ~= 84.4%`
   - 这更像：
     - **direct trunk/readout dynamic range shrinkage**
   - 不是：
     - raw `direct_feat` 输入幅度不足

3. **也没有看到 rollout instability / freerun drift。**
   - `trunk_hidden step Δ`
     - baseline `0.0363`
     - coadapt `0.0359`
   - teacher-vs-freerun paired drift：
     - `direct_feat = 0.0000`
     - `trunk_hidden = 0.0000`
   - 所以 residual 问题不是：
     - freerun 才突然漂掉
   - 而是：
     - **teacher-conditioned 就已经是“弱 non-plan path + 强 plan shortcut”**

### 30.16 minimal non-plan-only causality support (`SIC` bucket)

| candidate | freerun override | SIC0-10 Δ vs model/model | SIC11-21 Δ | SIC22-43 Δ | readout |
|---|---|---:|---:|---:|---|
| baseline_replace | zero/zero | 0.001986 | 0.001285 | 0.000052 | flat / non-collapse |
| coadapt_4x_directonly_calibration_240 | zero/zero | 0.043364 | 0.036322 | 0.016559 | all buckets worse; strongest early-mid |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | zero/zero | 0.038401 | 0.034732 | 0.015611 | all buckets worse; strongest early-mid |

最小读法：

- `baseline_replace`
  - non-plan-only 条件下几乎所有 SIC bucket 都只是 `~1e-3`
  - 没有 “越滚越炸” 的信号
- `coadapt` 两条 lane
  - 在 `zero/zero` 下所有 bucket 全面变差
  - 早中段最明显，后段仍坏
  - 更像是 **non-plan-only direct readout 本来就弱**
  - 不是单纯 late-cycle drift

### 30.17 baseline-vs-coadapt asymmetry table

| candidate | baseline-vs-coadapt difference | non-plan path strength | mainly depends on plan? | mainly depends on meas? | issue mode | supports key claim |
|---|---|---|---|---|---|---|
| baseline_replace | reference | strong | no | no | similar | baseline stands on non-plan path |
| coadapt_4x_directonly_calibration_240 | direct trunk weaker + plan shortcut | weak | yes | no | similar | coadapt non-plan path weak; plan shortcut supported |
| coadapt_4x_direct_plus_plan_ownership_240_noeventclock | ownership softens planner semantics only | weak | yes | no | similar | coadapt non-plan path weak; plan shortcut supported |

可以把 baseline-vs-coadapt asymmetry 再压成一句话：

- **`baseline_replace` 靠的是“强 non-plan direct path + 弱 plan side signal”；`coadapt` family 靠的是“弱 non-plan path + 过强 plan shortcut”，而且这个结构在 teacher-conditioned 就已经写死，不是 freerun 才出现。**

### 30.18 final judgement table

| candidate | direct 的主要依赖 source | non-plan direct path 是否可接受 | 这是否足以解释 baseline-vs-coadapt 差异 | recommended next role |
|---|---|---|---|---|
| `baseline_replace` | `non-plan path / cond-direct_feat`；`plan/meas` 只是弱 side-signal | yes | yes | `production` |
| `coadapt_4x_directonly_calibration_240` | `contacts_plan` dominant；`meas` 次要；non-plan trunk/readout 偏弱 | no | yes | `research-only` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | 仍然 `contacts_plan` dominant；只是 planner semantics 更好一点 | no | yes | `research-only` |

### 30.19 mandatory judgement answers

1. **`baseline_replace` 的 direct，主要是不是靠 non-plan path 就能成立？**
   - **是。**
   - 证据：
     - teacher-conditioned `zero/zero delta = +0.000691`
     - freerun `zero/zero delta = -0.002562`
   - 这已经足够把它归到：
     - `non-plan robust`

2. **`coadapt_4x_directonly_calibration_240` 的 non-plan direct path，是否已经明显退化？**
   - **是。**
   - 证据不只是最终 pose 更差，而是 white-box 直接显示：
     - `direct_feat` 输入段没有缩
     - 但 `trunk_hidden` 的 `rms/std` 相比 baseline 明显缩
     - 同时一旦 `plan=zero`，`DirectGeoLocalDeg` 直接 `+0.025979`
   - 所以更准确的说法是：
     - **non-plan direct path 的退化主要出现在 `direct_feat -> direct_pose_head -> out_direct` 的 trunk/readout 响应上**

3. **这种退化主要体现在什么？**
   - 主要是三件事叠在一起：
     1. **dynamic range shrinkage**
        - `trunk_hidden_std` 只有 baseline 的 `~84%`
     2. **对 `plan` 的过强依赖**
        - `zero/model = zero/zero`
        - `model/zero = model/model`
     3. **不是 rollout instability**
        - `step_delta` 没放大
        - teacher-vs-freerun drift 约等于 `0`
   - 所以主因排序更像：
     - **plan shortcut + trunk/readout weakness**
     - 而不是：
     - freerun instability

4. **`plan` 和 `meas` 里，哪个更像是 `coadapt` 的主要拐杖？**
   - **明确是 `plan`。**
   - `directonly_240`
     - `plan_score = 0.025979`
     - `meas_score = 0.008410`
   - `ownership_noeventclock`
     - `plan_score = 0.024685`
     - `meas_score = 0.009603`
   - 同时：
     - `model/zero == model/model`
     - `zero/zero == zero/model`
   - 所以 consumed `meas` path 在这条 lane 上不是主拐杖

5. **问题主要出现在 teacher-conditioned 还是 freerun？**
   - **teacher-conditioned 就已经成立；不是 freerun-specific。**
   - 更准确地说：
     - 这轮两个 `coadapt` candidate 在 teacher-conditioned 与 freerun 上的 dependency matrix 基本同构
   - freerun 没有再额外制造一个新的 failure mode

6. **`ownership_noeventclock` 是否只修了 planner semantics，还是也顺带缓和了 non-plan path 的脆弱性？**
   - **主要还是修 planner semantics；non-plan fragility 只被轻微缓和。**
   - 它确实把：
     - `plan_zero_delta`
       - 从 `+0.025979`
       - 降到 `+0.024685`
   - 但 white-box 仍显示：
     - `trunk_hidden rms/std` 基本还在 coadapt 水平
     - `plan` 仍是 dominant dependency
   - 所以不能把它解释成：
     - “已经顺带修好 non-plan direct path”

7. **基于这轮证据，下一步主优先级只能选一个，选什么？**
   - **只选：debug non-plan direct path / cond-direct feature downstream degeneration。**
   - 更具体地说：
     - 优先 debug `direct_feat -> direct_pose_head -> out_direct` 这段为什么比 baseline 更弱
     - 而不是再回去泛泛做 planner semantics
   - 这轮不支持把主优先级放到：
     - consumed `contact_plan` active semantic supervision
     - `contact_meas`
     - 其他 trunk / head 大改

### 30.20 one-paragraph white-box judgement

- **这轮最小 white-box audit 已经把“baseline-vs-coadapt 的 non-plan asymmetry”钉实：三条 candidate 的 `direct_feat` 输入段都一样，说明 raw `cond + time_pe` 不是问题；真正拉开差距的是 `direct_feat -> direct_pose_head -> out_direct` 这一段的 non-plan trunk/readout 响应，`baseline_replace` 保持了更高的 `trunk_hidden` 动态范围并在 `plan/meas` 清零后几乎不受伤，而 `coadapt_4x_directonly_calibration_240` 与 `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` 都表现为 non-plan path 偏弱、对 `plan` 的结构性依赖、对 `meas` 的近乎无依赖，且这种结构在 teacher-conditioned 与 freerun 下一致，因此这份记录当时给出的下一步主优先级是 debug non-plan direct path / cond-direct downstream degeneration，而不是再做一轮 planner semantics 复读。**

## 31. plan-shortcut takeover mechanism audit

这一节只做本轮要求的最小 frozen-checkpoint white-box audit，目标不是再证明 baseline 更好，而是把：

- `coadapt` 的 consumed `plan` path 如何在 frozen checkpoint 里表现出 **挤掉 non-plan direct path**
- 现有证据更支持：
  - upstream trunk weakening
  - 还是 direct head 内部的 shortcut takeover / non-plan starvation

压到 **head-level / branch-level**。

artifact root：

- `debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407/summary.json`
- `debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407/summary.md`
- per-candidate detail:
  - `debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407/candidates/baseline_replace.json`
  - `debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407/candidates/coadapt_4x_directonly_calibration_240.json`
  - `debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407/candidates/coadapt_4x_direct_plus_plan_ownership_240_noeventclock.json`
- minimal freerun spot-check:
  - `debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407/freerun_spotcheck_summary.json`

新增 helper：

- `tools/audit_cp015_tailk7_plan_shortcut_takeover_mechanism.py`

runtime / recipe contract：

- runtime contract change：`no`
- old recipe default behavior changed：`no`

### 31.1 code facts 简写

#### direct head consumed concat 的确切代码路径

- `train/models.py`
  - `direct_pose_feat_source='cond'`
  - `direct_pose_time_pe_dim=32`
  - runtime 先构造：
    - `direct_feat = cond`
    - 再 append `time_pe_direct`
  - 然后在 direct bridge 里做：
    - `direct_in = torch.cat([direct_feat, plan_use, meas_in], dim=-1)`
  - 再进入：
    - `_forward_direct_pose_readout(...)`

#### 这轮实际 hook 到的 module / 参数名

- shared split-head trunk：
  - `direct_pose_head.0`
  - `Linear(43 -> 512)`
- readout helper：
  - `_forward_direct_pose_readout`
- split readout modules：
  - `direct_pose_out_leg`
  - `direct_pose_out_arm`
  - `direct_pose_out_else`

#### branch slice / dim 边界

这三条 candidate 的 direct head 拓扑完全同构；首层输入固定为 `43D`，切块如下：

- `direct_feat = [0:39]`
  - 其中：
    - `cond = 7D`
    - `direct_pose_time_pe = 32D`
- `plan = [39:41]`
- `meas = [41:43]`

这个点很关键：

- **这条 lane 的 direct head 并不消费 `h_final/trunk_hidden`**
- 它消费的是：
  - `cond + direct_pose_time_pe + contacts_plan + contacts_meas`

所以：

- “`trunk_hidden std` 缩了 `16%`” 这件事，至多是 side symptom / correlated signal
- **不能直接作为这条 consumed lane 的 root cause 解释**

#### 这轮复用哪些现成工具

review 过但没有直接复用实验主体：

- `tools/diagnose_direct_head_jacobian_one_step.py`
  - 更偏 left/right Jacobian flip probe
- `tools/analyze_cp015_tailk7_motion_head_gain.py`
  - 更偏 trunk/output gain，不是 consumed-branch audit
- `tools/analyze_cp015_tailk7_rot_readout_decomposition.py`
  - 更偏 readout slice decomposition，不是 direct input branch competition

实际直接复用：

- `tools/analyze_cp015_tailk7_closed_loop_gap.py::_load_case`
- `train.validate.run_freerun_cycles::_run_freerun_cycles`
- 以及上一轮 direct-dependency asymmetry 的 candidate / eval path

### 31.2 audit design

#### compare candidates

主比较：

- `baseline_replace`
- `coadapt_4x_directonly_calibration_240`

辅助对照：

- `coadapt_4x_direct_plus_plan_ownership_240_noeventclock`

#### 主 eval 视角

主口径：

- teacher-conditioned
- 继续用：
  - `run_freerun_cycles --freerun_x_gt`

固定选择：

- `rounds=5`
- `cycle>=1`
- `drop_wrap=True`
- selected rows = `344`
- total rows = `434`

补了一个最小 freerun spot-check：

- 只看：
  - `baseline_replace`
  - `coadapt_4x_directonly_calibration_240`
- qualitative ordering 与 teacher-conditioned 一致；没有出现 reversal

#### sensitivity 指标定义

head-level sensitivity 用的是 **head-only local Jacobian**：

- metric:
  - `jacobian_fro_per_input_dim`
  - `= ||∂ out_direct_rot_norm / ∂ branch||_F / sqrt(branch_dim)`
- output 只取 direct rotation slice
- Jacobian 在 selected rows 上均匀抽 `96` 个 step 做同口径比较

这里用的是 head-only replay，而不是全模型 Jacobian，原因是本轮要 isolate：

- consumed `direct_feat`
- consumed `plan`
- consumed `meas`

在 direct head 里的竞争关系

#### block-wise decomposition 定义

只抓最有解释力的 consumed first layer：

- `direct_pose_head.0`

对三个 block 统计：

- block weight norm
- branch input std / rms norm
- effective contribution proxy：
  - `mean ||x_branch @ W_branch^T|| / sqrt(512)`

因为：

- `direct_feat` 是 `39D`
- `plan/meas` 各是 `2D`

所以除了 raw Fro norm，也看 per-dim 和 effective contribution，避免只被维度数误导。

#### causal ablation 定义

只在 frozen checkpoint 下做 **in-head ablation**：

- forward pre-hook 直接挂在：
  - `direct_pose_head.0`
- 只 zero 指定 branch slice：
  - `direct_feat`
  - `plan`
  - `meas`
- 不改：
  - global contacts loop
  - `contacts_err`
  - `lambda`
  - `event loop`

比较两个量：

1. direct output 自身 delta
   - 用 head-only replay 后的 direct-vs-ablated local geodesic delta
2. downstream `DirectGeoLocalDeg` delta
   - 用同一个 runtime contract 重跑 `_run_freerun_cycles`

### 31.3 candidate table

| candidate / run | self-contained? | event_clock enabled? | eval mode | checkpoint / eval artifact path | analysis artifact path |
|---|---|---|---|---|---|
| `baseline_replace` | `yes` | `yes` | `teacher-conditioned / freerun_x_gt` | ckpt=`models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth`；eval=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/baseline_replace/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407/candidates/baseline_replace.json` |
| `coadapt_4x_directonly_calibration_240` | `yes` | `yes` | `teacher-conditioned / freerun_x_gt` | ckpt=`models/__tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406/coadapt_4x_directonly_calibration_240/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_4x_directonly_calibration_240_20260406.pth`；eval=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_directonly_calibration_240/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407/candidates/coadapt_4x_directonly_calibration_240.json` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `yes` | `no` | `teacher-conditioned / freerun_x_gt` | ckpt=`models/__tmp_cp015_tailk7_plan_ownership_calibration_20260406/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_4x_direct_plus_plan_ownership_240_noeventclock_20260406.pth`；eval=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407/coadapt_4x_direct_plus_plan_ownership_240_noeventclock/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407/candidates/coadapt_4x_direct_plus_plan_ownership_240_noeventclock.json` |

### 31.4 sensitivity result table

metric definition：

- `jacobian_fro_per_input_dim.mean`
- `= mean_selected ||∂ out_direct_rot_norm / ∂ branch||_F / sqrt(branch_dim)`

| candidate | eval mode | metric definition | `direct_feat` sensitivity | `plan` sensitivity | `meas` sensitivity | `plan/direct_feat` ratio | `plan/direct ratio vs baseline` | 结论标签 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `baseline_replace` | `teacher-conditioned / freerun_x_gt` | head-only local Jacobian | `1.062731` | `0.012334` | `0.016853` | `0.011606` | `1.000000` | `direct_feat-preserved`, `meas-negligible` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned / freerun_x_gt` | head-only local Jacobian | `0.762544` | `0.270787` | `0.335545` | `0.355110` | `30.596936` | `plan-dominant`, `direct_feat-compressed`, `meas-negligible`, `shortcut-takeover-like` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned / freerun_x_gt` | head-only local Jacobian | `0.763420` | `0.270440` | `0.335651` | `0.354247` | `30.522651` | `plan-dominant`, `direct_feat-compressed`, `meas-negligible`, `shortcut-takeover-like` |

最关键的读法不是看 `plan` 是否绝对超过 `direct_feat`，而是看它相对 baseline 进入了完全不同的 regime：

- `baseline_replace`
  - `plan/direct = 0.011606`
- `directonly_240`
  - `plan/direct = 0.355110`
  - 是 baseline 的 `30.60x`
- `ownership_noeventclock`
  - `plan/direct = 0.354247`
  - 是 baseline 的 `30.52x`

也就是说：

- **coadapt` 的 frozen checkpoint 已经表现出明显更高的 `plan/direct_feat sensitivity ratio`**

而且：

- `direct_feat` sensitivity 自身也降了
  - `0.762544 / 1.062731 = 0.7175`
  - 约为 baseline 的 `71.8%`

### 31.5 weight / effective-gain table

effective contribution proxy：

- `mean ||x_branch @ W_branch^T|| / sqrt(512)`

| candidate | layer / module | `direct_feat` block weight norm | `plan` block weight norm | `meas` block weight norm | branch input std / norm | effective contribution proxy | 结论 |
|---|---|---:|---:|---:|---|---|---|
| `baseline_replace` | `direct_pose_head.0` | `13.123324` (`2.101414/dim`) | `0.039478` (`0.027915/dim`) | `0.042946` (`0.030368/dim`) | direct=`0.591593 / 0.660050`；plan=`0.161827 / 0.504454`；meas=`0.000000 / 0.000000` | direct=`0.445919`；plan=`0.000875`；meas=`0.000000` | no strong skew；plan block 远小于 direct |
| `coadapt_4x_directonly_calibration_240` | `direct_pose_head.0` | `12.727118` (`2.037970/dim`) | `2.841486` (`2.009234/dim`) | `2.753941` (`1.947330/dim`) | direct=`0.591593 / 0.660050`；plan=`0.163649 / 0.515760`；meas=`0.000000 / 0.000000` | direct=`0.389511`；plan=`0.065683`；meas=`0.000000` | `weight skew + effective-gain skew toward plan` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `direct_pose_head.0` | `12.727335` (`2.038005/dim`) | `2.841722` (`2.009401/dim`) | `2.753981` (`1.947359/dim`) | direct=`0.591593 / 0.660050`；plan=`0.156171 / 0.512467`；meas=`0.000000 / 0.000000` | direct=`0.389555`；plan=`0.065272`；meas=`0.000000` | `weight skew + effective-gain skew toward plan` |

这张表最关键的 mechanistic fact 有三个：

1. **`direct_feat` 输入统计几乎没变**
   - 三条 candidate 的 direct branch `input_std` 都是 `0.591593`
   - 也就是 consumed non-plan input 自身没有 collapse

2. **baseline 的 plan block 在首层几乎不存在**
   - per-dim weight 只有 `0.027915`
   - effective proxy 只有 `0.000875`

3. **coadapt 两条 lane 都把 plan block 抬到与 direct block 同一量级**
   - `directonly_240`
     - plan per-dim weight = `2.009234`
     - 相对 baseline 是 `71.98x`
   - `ownership_noeventclock`
     - plan per-dim weight = `2.009401`
     - 相对 baseline 也是 `71.98x`

所以这轮 first-layer 白盒更支持：

- **不是 input skew**
- 而是：
  - **head 内部把 plan block 学成了强 shortcut**

### 31.6 causal ablation table

说明：

- `direct output delta` = head-only replay 下，原始 direct output 与 branch-zeroed direct output 的 local geodesic delta
- `DirectGeoLocalDeg delta` = 用同一 runtime contract 重跑 `_run_freerun_cycles` 后的 downstream delta

| candidate | eval mode | ablated branch | direct output delta | `DirectGeoLocalDeg` delta | 结论标签 |
|---|---|---|---:|---:|---|
| `baseline_replace` | `teacher-conditioned / freerun_x_gt` | `direct_feat` | `3.989588` | `+3.518646` | `direct_feat ablation dominant` |
| `baseline_replace` | `teacher-conditioned / freerun_x_gt` | `plan` | `0.003510` | `+0.000456` | `plan ablation mild` |
| `baseline_replace` | `teacher-conditioned / freerun_x_gt` | `meas` | `0.000000` | `+0.000000` | `meas ablation negligible` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned / freerun_x_gt` | `direct_feat` | `3.692700` | `+3.289845` | `direct_feat ablation dominant` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned / freerun_x_gt` | `plan` | `0.081561` | `+0.021872` | `plan ablation catastrophic` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned / freerun_x_gt` | `meas` | `0.000000` | `+0.000000` | `meas ablation negligible` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned / freerun_x_gt` | `direct_feat` | `3.698699` | `+3.290079` | `direct_feat ablation dominant` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned / freerun_x_gt` | `plan` | `0.081997` | `+0.022583` | `plan ablation catastrophic` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `teacher-conditioned / freerun_x_gt` | `meas` | `0.000000` | `+0.000000` | `meas ablation negligible` |

这一张表说明：

- `coadapt` 并不是 “direct_feat 完全没了”
  - 因为 zero `direct_feat` 依然是 catastrophic
- 但它已经不是 baseline 那种：
  - `plan` 只是弱 side-signal
- 它变成：
  - **direct_feat 还重要**
  - 但 `plan` 也被学成了一个足以显著主导结果的 shortcut**

也就是更准确的说法：

- **non-plan starvation / shortcut takeover**
- 不是：
  - pure plan-only replacement

### 31.7 mechanism verdict table

| candidate | upstream trunk weakening 是否足以解释现象 | downstream head branch competition 是否主导 | 是否支持 `plan-shortcut takeover` | 是否支持 `non-plan starvation` | 证据强度 |
|---|---|---|---|---|---|
| `baseline_replace` | `no` | `no` | `no` | `no` | `strong` |
| `coadapt_4x_directonly_calibration_240` | `no` | `yes` | `yes` | `yes` | `strong` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `no` | `yes` | `yes` | `yes` | `strong` |

为什么 upstream weakening 不足以解释：

1. consumed lane 的 `direct_feat` 根本不是 `h_final`
   - 而是 `cond + time_pe`
2. direct branch input std 完全没缩
   - `0.591593` across all three candidates
3. 变化真正发生在 consumed head 内部：
   - plan block weight 从 baseline 的 `0.027915/dim`
   - 直接抬到 coadapt 的 `~2.009/dim`
4. 因果上也不是 upstream-only 能解释：
   - baseline `plan ablation = +0.000456`
   - coadapt `plan ablation ≈ +0.022`

所以本轮 mechanistic verdict 很明确：

- **主导矛盾在 downstream head branch competition / shortcut takeover**
- 不是：
  - “上游 hidden 先弱了，所以 direct 才坏”

### 31.8 final judgement table

| candidate | 当前 direct head 主要依赖 source | baseline 为什么没被 shortcut takeover | coadapt 为什么被 takeover | `ownership_noeventclock` 是否改变 takeover 结构 | 如果下一步只能选一个 fix family，选什么 | recommended next role |
|---|---|---|---|---|---|---|
| `baseline_replace` | `non-plan/direct_feat` | `plan` block 始终极小，`plan` ablation 几乎 `0`；direct path 自己就能成立 | n/a | n/a | keep as reference | `production` |
| `coadapt_4x_directonly_calibration_240` | `plan-shortcut-biased` | baseline comparator only | `plan/direct ratio` 相对 baseline = `30.60x`；`plan` sensitivity = baseline 的 `21.95x`；`plan` per-dim first-layer weight = baseline 的 `71.98x`；`plan` ablation = `+0.021872` | n/a | `debug training-time plan competition / shortcut takeover` | `research-only` |
| `coadapt_4x_direct_plus_plan_ownership_240_noeventclock` | `plan-shortcut-biased` | baseline comparator only | 与 directonly_240 几乎同构；`plan/direct ratio` 仍是 baseline 的 `30.52x`；`plan` ablation = `+0.022583` | **没有**实质改变 takeover 结构；只是在 planner semantics 上略缓和 | `debug training-time plan competition / shortcut takeover` | `research-only` |

### 31.9 mandatory judgement answers

1. **`coadapt` 的 frozen checkpoint，是否已经表现出明显更高的 `plan/direct_feat` sensitivity ratio？**
   - **是。**
   - `baseline_replace`
     - `plan/direct = 0.011606`
   - `coadapt_4x_directonly_calibration_240`
     - `plan/direct = 0.355110`
     - 是 baseline 的 `30.60x`
   - `coadapt_4x_direct_plus_plan_ownership_240_noeventclock`
     - `plan/direct = 0.354247`
     - 是 baseline 的 `30.52x`

2. **`coadapt` 的问题，更像 upstream trunk 先变弱，还是 downstream head 先学会 plan shortcut？**
   - **更像 downstream head 先学会了 plan shortcut，并在 head 内部形成 branch competition takeover。**
   - 理由：
     - 这条 consumed lane 的 `direct_feat` 是 `cond + time_pe`
     - 不是 `h_final`
     - direct branch input std 完全没缩
     - 但 first-layer 的 `plan` block weight / effective gain 暴涨
     - `plan` ablation 直接变成 catastrophic

3. **`trunk_hidden std` 缩 `16%` 这件事，本轮证据下更像 root cause 还是 symptom？**
   - **更像 symptom / side correlate，而不是这条 direct lane 的 root cause。**
   - 因为这条 consumed direct head 根本不吃 `h_final`
   - 所以它最多能解释 broader model state 的相关变化
   - 不能解释当前 observed head-level takeover

4. **`meas` 在这个机制里是不是仍然次要？**
   - **是，仍然次要。**
   - selected rows 上：
     - consumed `meas` slice `input_std = 0`
     - effective proxy = `0`
     - meas ablation = `0`
   - 所以这轮结构性问题仍然主要不是 `meas`

5. **`ownership_noeventclock` 改善的是 planner semantics，还是也实质改变了 shortcut takeover 结构？**
   - **主要是 planner semantics；没有实质改变 shortcut takeover 结构。**
   - 它和 `directonly_240` 在下面几乎完全同构：
     - `plan/direct ratio`
     - `plan` sensitivity
     - `plan` first-layer weight
     - `plan` ablation
   - downstream `plan` ablation 甚至略高：
     - `directonly_240 = +0.021872`
     - `ownership_noeventclock = +0.022583`

6. **基于这轮证据，下一步主优先级如果只能选一个，选什么？**
   - **只选：debug training-time plan competition / shortcut takeover。**
   - 不是：
     - 继续泛化 debug non-plan trunk hidden
     - 转去做 `contact_meas`
     - trunk / donor / adapter redesign

### 31.10 one-paragraph mechanism judgement

- **这轮 frozen-checkpoint mechanism audit 已经把主矛盾从“泛泛 debug trunk hidden”收敛到了“debug training-time plan competition / shortcut takeover”：这些 candidate 的 direct head consumed lane 固定是 `cond + time_pe + plan + meas`，而不是 `h_final`；baseline 的 `plan` block 在首层几乎不存在（`0.027915/dim`，effective proxy `0.000875`），所以 `plan` ablation 只有 `+0.000456`，direct 基本靠 non-plan path 成立；coadapt 两条 lane 则把 `plan` block 抬到与 `direct_feat` 同量级（`~2.009/dim`），使 `plan/direct` sensitivity ratio 相对 baseline 放大到 `~30.5x`，同时 `plan` ablation 变成 `+0.0219~+0.0226` 的 catastrophic 级，而 `meas` 仍然几乎完全不起作用。因此这轮证据更支持：coadapt 的 residual gap 主因不是 upstream trunk 先弱，而是 direct head 在训练中把 consumed `plan` 学成了更容易的 shortcut，并由此挤压了 non-plan path；`ownership_noeventclock` 主要只是轻微改善 planner semantics，没有修掉 takeover 结构。**

## 32. Minimal `direct_pose_plan_drop_prob` competition probe (2026-04-07)

本节只做一个最小 falsifier：**不改 architecture / donor / trunk / lambda / meas / planner semantics**，只在 `train_direct_pose` lane 上把 `direct_pose_plan_drop_prob` 从旧 replace recipe 的 `0.0` 提到非零，检查 training-time branch competition 是否可被打断。

### 32.1 training design

固定约束：

- warmstart 固定为 `coadapt_allrot_interface_bestlr_longer_4x`
- trainable scope 固定为 `train_direct_pose` only；不扩大到 trunk / interface / donor / lambda
- runtime contract 不变；event-clock 口径保持与 `coadapt_4x_directonly_calibration_240` 同口径
- 旧 recipe 默认行为**没有**改掉；`train/posttrain.py` 默认仍是 `direct_pose_plan_drop_prob = 0.0`，只在新生成 config 上显式覆写

| lane | warmstart | trainable scope | only changed config | `direct_pose_plan_drop_prob` | old recipe default changed? |
|---|---|---|---|---:|---|
| `coadapt_plan_drop_0p3` | `coadapt_allrot_interface_bestlr_longer_4x` | `train_direct_pose` only | `direct_pose_plan_drop_prob` | `0.3` | `no` |
| `coadapt_plan_drop_0p5` | `coadapt_allrot_interface_bestlr_longer_4x` | `train_direct_pose` only | `direct_pose_plan_drop_prob` | `0.5` | `no` |

训练侧证据链：

- config summary: `debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/summary.json`
- generated configs:
  - `debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/configs/coadapt_plan_drop_0p3_20260407.json`
  - `debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/configs/coadapt_plan_drop_0p5_20260407.json`
- manual train runtime 确认两条 lane 仍是：
  - `mode=train_direct_pose`
  - `trainable=20 params`
  - 只动 `direct_pose_*`，没有 interface/trunk/lambda 混入

### 32.2 candidate table

| candidate | warmstart | `plan_drop_prob` | self-contained? | event_clock enabled? | checkpoint / eval artifact path | analysis artifact path |
|---|---|---:|---|---|---|---|
| `baseline_replace` | `70a_replace_zerophase_20260317` | `0.0` | `yes` | `yes` | ckpt=`models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth`；eval=`debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/new70b_replace_lowdrift/Walk_F_freerun_cycles.json` | behavior=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407_plan_drop/summary.json`；mechanism=`debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407_plan_drop/candidates/baseline_replace.json` |
| `coadapt_4x_directonly_calibration_240` | `coadapt_allrot_interface_bestlr_longer_4x` | `0.0` | `yes` | `yes` | ckpt=`models/__tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406/coadapt_4x_directonly_calibration_240/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_4x_directonly_calibration_240_20260406.pth`；eval=`debug_output/_tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406/eval_model_source/coadapt_4x_directonly_calibration_240/Walk_F_freerun_cycles.json` | behavior=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407_plan_drop/summary.json`；mechanism=`debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407_plan_drop/candidates/coadapt_4x_directonly_calibration_240.json` |
| `coadapt_plan_drop_0p3` | `coadapt_allrot_interface_bestlr_longer_4x` | `0.3` | `yes` | `yes` | ckpt=`models/__tmp_cp015_tailk7_plan_drop_competition_probe_20260407/coadapt_plan_drop_0p3/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_plan_drop_0p3_20260407.pth`；eval=`debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/eval_model_source/coadapt_plan_drop_0p3/Walk_F_freerun_cycles.json` | training=`debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/summary.json`；behavior=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407_plan_drop/summary.json`；mechanism=`debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407_plan_drop/candidates/coadapt_plan_drop_0p3.json` |
| `coadapt_plan_drop_0p5` | `coadapt_allrot_interface_bestlr_longer_4x` | `0.5` | `yes` | `yes` | ckpt=`models/__tmp_cp015_tailk7_plan_drop_competition_probe_20260407/coadapt_plan_drop_0p5/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_plan_drop_0p5_20260407.pth`；eval=`debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/eval_model_source/coadapt_plan_drop_0p5/Walk_F_freerun_cycles.json` | training=`debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/summary.json`；behavior=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407_plan_drop/summary.json`；mechanism=`debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407_plan_drop/candidates/coadapt_plan_drop_0p5.json` |

### 32.3 behavior result table

| candidate | eval mode | `model/model` | `plan=zero, meas=model` | `plan=gt, meas=model` | `model/zero` | `zero/zero` | `plan_score` | `meas_score` | `zero/zero delta` | 结论标签 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_replace` | `teacher-conditioned` | `0.150492` | `0.151184` | `0.150220` | `0.150492` | `0.151184` | `0.001078` | `0.001376` | `+0.000691` | `non-plan robust / no shortcut` |
| `baseline_replace` | `freerun` | `0.153746` | `0.151184` | `0.150220` | `0.150492` | `0.151184` | `0.003526` | `0.003253` | `-0.002562` | `non-plan robust / no shortcut` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `0.172359` | `0.198338` | `0.153369` | `0.172359` | `0.198338` | `0.025979` | `0.008410` | `+0.025979` | `plan-zero collapse / takeover` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `0.172359` | `0.198338` | `0.153369` | `0.172359` | `0.198338` | `0.025979` | `0.008410` | `+0.025979` | `plan-zero collapse / takeover` |
| `coadapt_plan_drop_0p3` | `teacher-conditioned` | `0.170843` | `0.170028` | `0.157163` | `0.170843` | `0.170028` | `0.016193` | `0.000920` | `-0.000814` | `takeover reduced / no zero collapse` |
| `coadapt_plan_drop_0p3` | `freerun` | `0.170843` | `0.170028` | `0.157163` | `0.170843` | `0.170028` | `0.016193` | `0.000920` | `-0.000814` | `takeover reduced / no zero collapse` |
| `coadapt_plan_drop_0p5` | `teacher-conditioned` | `0.170569` | `0.165486` | `0.158119` | `0.170569` | `0.165486` | `0.016003` | `0.002843` | `-0.005082` | `takeover reduced / plan-free slightly better` |
| `coadapt_plan_drop_0p5` | `freerun` | `0.170569` | `0.165486` | `0.158119` | `0.170569` | `0.165486` | `0.016003` | `0.002843` | `-0.005082` | `takeover reduced / plan-free slightly better` |

behavior 层直接读出来的最关键变化：

- `coadapt_4x_directonly_calibration_240`：teacher-conditioned `zero/zero delta = +0.025979`，仍是典型的 `plan-zero collapse`。
- `coadapt_plan_drop_0p3`：teacher-conditioned `zero/zero delta = -0.000814`，已经基本消掉 “没 plan 就塌” 的现象。
- `coadapt_plan_drop_0p5`：teacher-conditioned `zero/zero delta = -0.005082`，`plan=zero` 反而略优于 `model/model`；但 `plan=gt` 仍可再给 `-0.012449` 改善，说明 `plan` 更像 auxiliary signal 而不是 shortcut。
- 两条新 lane 的 `meas_score` 仍然很小（`0.000920` / `0.002843`），继续支持 `meas` 不是主矛盾。

### 32.4 mechanism result table

| candidate | `direct_feat` sensitivity | `plan` sensitivity | `meas` sensitivity | `plan/direct ratio` | `plan` block weight / dim | `direct` block weight / dim | effective contribution proxy | `plan` ablation delta | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `baseline_replace` | `1.062731` | `0.012334` | `0.016853` | `0.011606` | `0.027915` | `2.101414` | direct=0.445919；plan=0.000875；meas=0.000000 | `+0.000456` | `reference / no takeover` |
| `coadapt_4x_directonly_calibration_240` | `0.762544` | `0.270787` | `0.335545` | `0.355110` | `2.009234` | `2.037970` | direct=0.389511；plan=0.065683；meas=0.000000 | `+0.021872` | `takeover unchanged` |
| `coadapt_plan_drop_0p3` | `0.765678` | `0.231673` | `0.337984` | `0.302573` | `2.002692` | `2.038116` | direct=0.389730；plan=0.065331；meas=0.000000 | `+0.006273` | `takeover reduced, still plan-dominant` |
| `coadapt_plan_drop_0p5` | `0.766480` | `0.227979` | `0.340734` | `0.297436` | `2.001207` | `2.038190` | direct=0.389843；plan=0.065238；meas=0.000000 | `+0.002094` | `takeover reduced, still plan-dominant` |

mechanism 层最关键的对照：

- `plan/direct ratio`：`0.355110 -> 0.302573 -> 0.297436`，相对 `directonly_240` 下降约 `14.8% / 16.2%`，但仍远高于 baseline 的 `0.011606`。
- `plan` sensitivity：`0.270787 -> 0.231673 -> 0.227979`，有实质回落，但仍处在明显 plan-dominant regime。
- **`plan` block weight / dim 基本没动：**`2.009234 -> 2.002692 -> 2.001207`，仍是 `~2.0/dim`。
- **`plan` ablation downstream delta 明显回落：**`+0.021872 -> +0.006273 -> +0.002094`，其中 `0p5` 相对 `directonly_240` 下降约 `90.4%`。
- effective contribution proxy 的 `plan/direct` 比值几乎没变（约 `0.1686 -> 0.1676 / 0.1673`），所以缓解主要不是 first-layer weight skew 被修掉，而是 downstream causal dependence 被削弱。

### 32.5 final judgement table

| candidate | 是否支持 training-time competition 假设 | 是否支持 “plan drop 可缓解 shortcut takeover” | 是否值得进入下一步 fix family | recommended next role |
|---|---|---|---|---|
| `baseline_replace` | `reference` | `n/a` | `reference only` | `production` |
| `coadapt_4x_directonly_calibration_240` | `yes`（positive control） | `no` | `as control only` | `research-only` |
| `coadapt_plan_drop_0p3` | `yes` | `yes, partial` | `yes` | `research-only` |
| `coadapt_plan_drop_0p5` | `yes` | `yes, strongest this round` | `yes` | `research-only` |

这里的排序很清楚：

1. **`coadapt_plan_drop_0p5` 是本轮最强 signal。**
   - `DirectGeoLocalDeg(model/model) = 0.170569`，比 `directonly_240` 的 `0.172359` 略好。
   - `plan` ablation delta 已压到 `+0.002094`，几乎回到 mild regime。
   - 但 `plan/direct ratio = 0.297436`、`plan` block weight `~2.001/dim` 仍说明 takeover 没有被结构性修干净。

2. **`coadapt_plan_drop_0p3` 也支持同一个方向，但比 `0p5` 弱。**
   - 行为层已经摆脱 `plan-zero collapse`。
   - 但 `plan` ablation 仍有 `+0.006273`，明显高于 `0p5`。

3. **两条 plan-drop lane 都还不够进入 production / replacement gate。**
   - 直接指标仍明显高于 `baseline_replace` 的 `0.150492 ~ 0.153746`。
   - mechanism 上也仍远离 baseline regime。

### 32.6 mandatory judgement answers

1. **非零 `direct_pose_plan_drop_prob` 是否有效缓解了 shortcut takeover？**
   - **是，但只是 partial mitigation，不是 full repair。**
   - strongest candidate 是 `coadapt_plan_drop_0p5`。

2. **缓解幅度主要体现在哪？**
   - **最主要体现在 causal ablation 与 behavior。**
   - behavior：`zero/zero delta` 从 `+0.025979` 直接回到 `-0.000814 / -0.005082`。
   - causal ablation：`plan` downstream delta 从 `+0.021872` 降到 `+0.006273 / +0.002094`。
   - sensitivity ratio 也有回落，但幅度中等：`0.355110 -> 0.302573 / 0.297436`。
   - **weight skew 几乎没变**，所以这不是 “首层 plan block 已经修好”，而更像 “固定 plan shortcut 的可训练依赖被部分打断”。

3. **非零 `direct_pose_plan_drop_prob` 是否能显著压低 `plan/direct sensitivity ratio`？**
   - **能压低，但还谈不上把它拉回 healthy regime。**
   - 相对 `directonly_240` 下降约 `15~16%`，但仍是 baseline 的 `~25.6x / 26.1x`。

4. **非零 `direct_pose_plan_drop_prob` 是否能阻止 `plan` block weight 暴涨到 `~2.0/dim`？**
   - **不能。**
   - `plan` block per-dim weight 基本保持在 `2.00/dim`，与 `directonly_240` 几乎重合。

5. **`plan` ablation downstream delta 是否明显回落？**
   - **是，且这是本轮最强证据。**
   - `0p3`: `+0.006273`
   - `0p5`: `+0.002094`
   - 对比 `directonly_240 = +0.021872`，回落非常明显。

6. **direct 最终指标是否仍保持可接受，至少不明显劣于 `coadapt_4x_directonly_calibration_240`？**
   - **是。**
   - 两条新 lane 的 `DirectGeoLocalDeg(model/model)` 都略优于 `0.172359`。
   - 但离 `baseline_replace` 仍有明显 gap，所以只能算 “non-regressing / slightly better”，还不是 replacement-ready。

7. **这些结果是否足以把 root cause 更明确地钉到 training-time plan competition / shortcut takeover，而不是 head 结构性天然偏好 plan？**
   - **更支持 “training-time competition 是 major cause”，但还不能说 head structural bias 完全不存在。**
   - 反证是：**不改 architecture，只改 training-time plan dropout，就能显著改掉 behavior collapse 与 causal ablation。** 这已经排除了 “纯结构不可改”的解释。
   - 但因为 `plan` first-layer weight skew 仍钉在 `~2.0/dim`，也说明 **simple fixed dropout 还不足以完全消掉 plan-favored solution**。更准确的说法是：
     - root cause 主要在 training-time competition / shortcut takeover
     - 但这个 competition 问题**不止**靠固定 `plan_drop` 就能完全解完

8. **基于这轮结果，下一步是否还值得继续沿 training-time competition 这条线推进？**
   - **值得，而且应继续。**
   - 但这轮结果已经说明：下一步不该把希望放在“固定 `plan_drop` 一招吃透”；更合理的是把它当成一个 confirmed direction，后续继续在 training-time competition family 内做更精确的 non-plan ownership / curriculum / asymmetry control。

### 32.7 one-paragraph judgement

- **这轮最小 `direct_pose_plan_drop_prob` probe 给出的结论是：training-time plan competition / shortcut takeover 确实是可干预的，而且只靠 training-time 改动就能把最关键的 behavior collapse 与 causal plan-ablation 大幅压下去；因此 root cause 已经更明确地指向 training-time branch competition，而不是“head 结构天然注定偏好 plan”。但同一轮证据也同样清楚地说明，固定 `plan_drop` 只能 partial repair：`plan/direct` sensitivity ratio 只从 `0.355110` 降到 `0.302573 / 0.297436`，`plan` first-layer weight 仍然钉在 `~2.0/dim`，所以 takeover regime 仍在，离 baseline 的 `0.011606` 还差得很远。综合看，`coadapt_plan_drop_0p5` 是本轮最强研究候选：它把 `plan` ablation 从 `+0.021872` 压到 `+0.002094`，并且 `DirectGeoLocalDeg` 还略优于 `directonly_240`；但它仍然只能是 `research-only`，还不能进 production。**

## 33. Minimal `direct_pose_plan_drop_prob` schedule probe (2026-04-07)

本节只追一个更窄的问题：**如果 shortcut takeover 的关键是 training early stage 的 branch competition ordering，那么“早期高 `plan_drop`、后期再放开”是否能比 fixed `plan_drop=0.5` 更进一步改变 ownership structure。**

边界继续固定：

- 不改 architecture
- 不改 donor / adapter / trunk / lambda / meas / contact semantics
- warmstart 继续用 `coadapt_allrot_interface_bestlr_longer_4x`
- trainable scope 继续固定为 `train_direct_pose` only
- runtime contract 不变
- 旧 recipe 默认行为不变：
  - `direct_pose_plan_drop_prob` 默认仍是 `0.0`
  - 新增的 `direct_pose_plan_drop_schedule` 默认是 `None`
  - schedule 只在新 probe config 上显式覆写

代码侧只做了最小支持：

- `train/posttrain.py`
  - 新增 `direct_pose_plan_drop_schedule` 的 config parse / validation
  - 在 posttrain step loop 中按 `global_step` 解析当前 `plan_drop_prob`
  - `train/models.py` 不改
- audit helper 只补了一个最小路径参数：
  - `tools/audit_cp015_tailk7_plan_shortcut_takeover_mechanism.py --direct-dependency-root`
  - 只为复用同一套 helper 指向本轮新的 behavior audit root

### 33.1 training design

| lane | warmstart | trainable scope | only changed config | schedule definition | old recipe default changed? |
|---|---|---|---|---|---|
| `coadapt_plan_drop_sched_1p0_to_0p3_240` | `coadapt_allrot_interface_bestlr_longer_4x` | `train_direct_pose` only | `direct_pose_plan_drop_schedule` | `[0,80)->1.0; [80,160)->0.7; [160,240)->0.3` | `no` |
| `coadapt_plan_drop_sched_1p0_to_0p0_240` | `coadapt_allrot_interface_bestlr_longer_4x` | `train_direct_pose` only | `direct_pose_plan_drop_schedule` | `[0,80)->1.0; [80,160)->0.5; [160,240)->0.0` | `no` |

训练产物：

- training summary:
  `debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/summary.json`
- generated configs:
  - `debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/configs/coadapt_plan_drop_sched_1p0_to_0p3_240_20260407.json`
  - `debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/configs/coadapt_plan_drop_sched_1p0_to_0p0_240_20260407.json`
- step log 明确显示 schedule 真正在训练中生效：
  - `1.0 -> 0.7 -> 0.3`
  - `1.0 -> 0.5 -> 0.0`

### 33.2 candidate table

| candidate | warmstart | schedule | self-contained? | event_clock enabled? | checkpoint / eval artifact path | analysis artifact path |
|---|---|---|---|---|---|---|
| `baseline_replace` | `70a_replace_zerophase_20260317` | `fixed 0.0` | `yes` | `yes` | ckpt=`models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth`；eval=`debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/eval_model_source/new70b_replace_lowdrift/Walk_F_freerun_cycles.json` | behavior=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407_plan_drop_schedule/summary.json`；mechanism=`debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407_plan_drop_schedule/candidates/baseline_replace.json` |
| `coadapt_4x_directonly_calibration_240` | `coadapt_allrot_interface_bestlr_longer_4x` | `fixed 0.0` | `yes` | `yes` | ckpt=`models/__tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406/coadapt_4x_directonly_calibration_240/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_4x_directonly_calibration_240_20260406.pth`；eval=`debug_output/_tmp_cp015_tailk7_replace_direct_recovery_bridge_20260406/eval_model_source/coadapt_4x_directonly_calibration_240/Walk_F_freerun_cycles.json` | behavior=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407_plan_drop_schedule/summary.json`；mechanism=`debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407_plan_drop_schedule/candidates/coadapt_4x_directonly_calibration_240.json` |
| `coadapt_plan_drop_0p5` | `coadapt_allrot_interface_bestlr_longer_4x` | `fixed 0.5` | `yes` | `yes` | ckpt=`models/__tmp_cp015_tailk7_plan_drop_competition_probe_20260407/coadapt_plan_drop_0p5/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_plan_drop_0p5_20260407.pth`；eval=`debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/eval_model_source/coadapt_plan_drop_0p5/Walk_F_freerun_cycles.json` | behavior=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407_plan_drop_schedule/summary.json`；mechanism=`debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407_plan_drop_schedule/candidates/coadapt_plan_drop_0p5.json` |
| `coadapt_plan_drop_sched_1p0_to_0p3_240` | `coadapt_allrot_interface_bestlr_longer_4x` | `[0,80)->1.0; [80,160)->0.7; [160,240)->0.3` | `yes` | `yes` | ckpt=`models/__tmp_cp015_tailk7_plan_drop_competition_probe_20260407/coadapt_plan_drop_sched_1p0_to_0p3_240/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_plan_drop_sched_1p0_to_0p3_240_20260407.pth`；eval=`debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/eval_model_source/coadapt_plan_drop_sched_1p0_to_0p3_240/Walk_F_freerun_cycles.json` | training=`debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/summary.json`；behavior=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407_plan_drop_schedule/summary.json`；mechanism=`debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407_plan_drop_schedule/candidates/coadapt_plan_drop_sched_1p0_to_0p3_240.json` |
| `coadapt_plan_drop_sched_1p0_to_0p0_240` | `coadapt_allrot_interface_bestlr_longer_4x` | `[0,80)->1.0; [80,160)->0.5; [160,240)->0.0` | `yes` | `yes` | ckpt=`models/__tmp_cp015_tailk7_plan_drop_competition_probe_20260407/coadapt_plan_drop_sched_1p0_to_0p0_240/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_plan_drop_sched_1p0_to_0p0_240_20260407.pth`；eval=`debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/eval_model_source/coadapt_plan_drop_sched_1p0_to_0p0_240/Walk_F_freerun_cycles.json` | training=`debug_output/_tmp_cp015_tailk7_plan_drop_competition_probe_20260407/summary.json`；behavior=`debug_output/_tmp_cp015_tailk7_direct_dependency_asymmetry_audit_20260407_plan_drop_schedule/summary.json`；mechanism=`debug_output/_tmp_cp015_tailk7_plan_shortcut_takeover_mechanism_audit_20260407_plan_drop_schedule/candidates/coadapt_plan_drop_sched_1p0_to_0p0_240.json` |

### 33.3 behavior result table

| candidate | eval mode | `model/model` | `plan=zero, meas=model` | `plan=gt, meas=model` | `model/zero` | `zero/zero` | `plan_score` | `meas_score` | `zero/zero delta` | 结论标签 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_replace` | `teacher-conditioned` | `0.150492` | `0.151184` | `0.150220` | `0.150492` | `0.151184` | `0.001078` | `0.001376` | `+0.000691` | `reference / non-plan robust` |
| `baseline_replace` | `freerun` | `0.153746` | `0.151184` | `0.150220` | `0.150492` | `0.151184` | `0.003526` | `0.003253` | `-0.002562` | `reference / non-plan robust` |
| `coadapt_4x_directonly_calibration_240` | `teacher-conditioned` | `0.172359` | `0.198338` | `0.153369` | `0.172359` | `0.198338` | `0.025979` | `0.008410` | `+0.025979` | `plan-zero collapse / takeover` |
| `coadapt_4x_directonly_calibration_240` | `freerun` | `0.172359` | `0.198338` | `0.153369` | `0.172359` | `0.198338` | `0.025979` | `0.008410` | `+0.025979` | `plan-zero collapse / takeover` |
| `coadapt_plan_drop_0p5` | `teacher-conditioned` | `0.170569` | `0.165486` | `0.158119` | `0.170569` | `0.165486` | `0.016003` | `0.002843` | `-0.005082` | `best fixed-drop reference / no zero collapse` |
| `coadapt_plan_drop_0p5` | `freerun` | `0.170569` | `0.165486` | `0.158119` | `0.170569` | `0.165486` | `0.016003` | `0.002843` | `-0.005082` | `best fixed-drop reference / no zero collapse` |
| `coadapt_plan_drop_sched_1p0_to_0p3_240` | `teacher-conditioned` | `0.177076` | `0.173451` | `0.164189` | `0.177076` | `0.173451` | `0.018849` | `0.004665` | `-0.003625` | `robustness kept / worse than fixed 0.5` |
| `coadapt_plan_drop_sched_1p0_to_0p3_240` | `freerun` | `0.177076` | `0.173451` | `0.164189` | `0.177076` | `0.173451` | `0.018849` | `0.004665` | `-0.003625` | `robustness kept / worse than fixed 0.5` |
| `coadapt_plan_drop_sched_1p0_to_0p0_240` | `teacher-conditioned` | `0.184031` | `0.188273` | `0.165059` | `0.184031` | `0.188273` | `0.030916` | `0.012814` | `+0.004242` | `partial robustness regression / worse than fixed 0.5` |
| `coadapt_plan_drop_sched_1p0_to_0p0_240` | `freerun` | `0.184031` | `0.188273` | `0.165059` | `0.184031` | `0.188273` | `0.030916` | `0.012814` | `+0.004242` | `partial robustness regression / worse than fixed 0.5` |

behavior 层直接结论：

- `1.0 -> 0.7 -> 0.3` **没有**比 fixed `0.5` 更进一步压低 behavior-side plan dependence：
  - `plan_score`: `0.018849` > fixed `0.016003`
  - `zero/zero delta`: `-0.003625` 仍然稳，但也**没有**优于 fixed `-0.005082`
  - `model/model`: `0.177076` 明显差于 fixed `0.170569`
- `1.0 -> 0.5 -> 0.0` 更差：
  - `zero/zero delta = +0.004242`
  - `plan_score = 0.030916`
  - `model/model = 0.184031`

也就是说，schedule lane 仍然能保持某种“plan 缺失时可工作”的 dual-mode 特征，但**没有**把 behavior 推到比 fixed `0.5` 更好的 regime；`1.0 -> 0.0` 甚至在 robustness 和最终 direct 指标上同时回退。

### 33.4 mechanism result table

| candidate | `direct_feat` sensitivity | `plan` sensitivity | `meas` sensitivity | `plan/direct ratio` | `plan` block weight / dim | `direct` block weight / dim | effective contribution proxy | `plan` ablation delta | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `baseline_replace` | `1.062731` | `0.012334` | `0.016853` | `0.011606` | `0.027915` | `2.101414` | direct=`0.445919`; plan=`0.000875`; meas=`0.000000` | `+0.000456` | `reference / no takeover` |
| `coadapt_4x_directonly_calibration_240` | `0.762544` | `0.270787` | `0.335545` | `0.355110` | `2.009234` | `2.037970` | direct=`0.389511`; plan=`0.065683`; meas=`0.000000` | `+0.021872` | `takeover unchanged / still plan-dominant` |
| `coadapt_plan_drop_0p5` | `0.766480` | `0.227979` | `0.340734` | `0.297436` | `2.001207` | `2.038190` | direct=`0.389843`; plan=`0.065238`; meas=`0.000000` | `+0.002094` | `takeover reduced / still plan-dominant` |
| `coadapt_plan_drop_sched_1p0_to_0p3_240` | `0.764584` | `0.234604` | `0.339507` | `0.306839` | `1.996744` | `2.038112` | direct=`0.389838`; plan=`0.064990`; meas=`0.000000` | `+0.003951` | `ownership not improved vs fixed 0.5 / still plan-dominant` |
| `coadapt_plan_drop_sched_1p0_to_0p0_240` | `0.765116` | `0.232597` | `0.335772` | `0.304002` | `2.000403` | `2.038158` | direct=`0.389765`; plan=`0.065221`; meas=`0.000000` | `+0.009076` | `takeover reduced vs directonly but regressed vs fixed 0.5 / still plan-dominant` |

mechanism 层的关键信号非常一致：

1. **schedule 没有比 fixed `0.5` 更进一步压低 `plan/direct ratio`。**
   - fixed `0.5`: `0.297436`
   - `1.0 -> 0.7 -> 0.3`: `0.306839`
   - `1.0 -> 0.5 -> 0.0`: `0.304002`

2. **schedule 没有第一次把 `plan` block weight / dim 拉出 `~2.0/dim`。**
   - fixed `0.5`: `2.001207`
   - `1.0 -> 0.7 -> 0.3`: `1.996744`
   - `1.0 -> 0.5 -> 0.0`: `2.000403`

   `1.996744` 只是极小变化，不构成脱离 `~2.0/dim` regime 的证据。

3. **schedule 没有比 fixed `0.5` 保持更低的 causal plan ablation。**
   - fixed `0.5`: `+0.002094`
   - `1.0 -> 0.7 -> 0.3`: `+0.003951`
   - `1.0 -> 0.5 -> 0.0`: `+0.009076`

4. effective contribution proxy 也没有出现新的 ownership shift：
   - fixed `0.5`: plan proxy `0.065238`
   - `1.0 -> 0.7 -> 0.3`: `0.064990`
   - `1.0 -> 0.5 -> 0.0`: `0.065221`

也就是说，**early high-drop 这条最小 schedule probe 并没有把 head 从 plan-dominant ownership regime 里真正推出来。**

### 33.5 final judgement table

| candidate | 是否支持 early competition ordering 假设 | 是否支持 “schedule 比 fixed drop 更能改 ownership” | 是否值得进入下一步 fix family | recommended next role |
|---|---|---|---|---|
| `baseline_replace` | `reference` | `n/a` | `reference only` | `production` |
| `coadapt_4x_directonly_calibration_240` | `positive control` | `no` | `as control only` | `research-only` |
| `coadapt_plan_drop_0p5` | `yes, for competition family in general` | `baseline comparator` | `yes` | `research-only` |
| `coadapt_plan_drop_sched_1p0_to_0p3_240` | `no stronger support` | `no` | `no` | `reject` |
| `coadapt_plan_drop_sched_1p0_to_0p0_240` | `no` | `no` | `no` | `reject` |

这里的结论很直接：

- **如果只问“schedule 是否比 fixed `0.5` 更进一步有效？”答案是否。**
- `1.0 -> 0.7 -> 0.3` 是这轮更好的 schedule lane，但它依然：
  - `plan/direct ratio` 更高
  - `plan` ablation 更差
  - `model/model` 更差
  - `plan` weight skew 没有实质变化
- `1.0 -> 0.5 -> 0.0` 则基本构成负例：后期完全放开 `plan` 不但没有改 ownership，反而在 behavior / causal dependence / direct 指标上都更差。

### 33.6 mandatory judgement answers

1. **schedule 是否比 fixed `plan_drop=0.5` 更进一步压低了 `plan/direct sensitivity ratio`？**
   - **否。**
   - fixed `0.5 = 0.297436`
   - `1.0 -> 0.7 -> 0.3 = 0.306839`
   - `1.0 -> 0.5 -> 0.0 = 0.304002`

2. **schedule 是否第一次让 `plan` block weight / dim 明显脱离 `~2.0/dim`？**
   - **否。**
   - 两条 schedule lane 都仍然钉在 `~2.0/dim`。

3. **schedule 是否继续保持低 `plan` ablation delta`？**
   - `1.0 -> 0.7 -> 0.3`：**是，但不如 fixed `0.5`。**
     - `+0.003951` 仍在低位，但高于 fixed `+0.002094`
   - `1.0 -> 0.5 -> 0.0`：**部分失守。**
     - `+0.009076` 已明显回升

4. **schedule 是否继续保持 `zero/zero delta` 不塌？**
   - `1.0 -> 0.7 -> 0.3`：**是。**
     - `-0.003625`
   - `1.0 -> 0.5 -> 0.0`：**不够稳。**
     - `+0.004242`

5. **direct 最终指标是否至少不明显差于 `coadapt_plan_drop_0p5`？**
   - **否。**
   - fixed `0.5`: `0.170569`
   - `1.0 -> 0.7 -> 0.3`: `0.177076`
   - `1.0 -> 0.5 -> 0.0`: `0.184031`

6. **如果 schedule 有效，是否足以更明确地把 root cause 钉到 early training competition ordering？**
   - **本轮不能这么说。**
   - 这个最小 falsifier **没有**给出 “early ordering 比 fixed drop 更能改 ownership” 的正证据。

7. **如果 schedule 无效，是否更支持：fixed dropout 只能修 robustness，但 ownership 还需要更显式的 shaping？**
   - **是，而且这是本轮最强结论。**
   - schedule lane 仍然保持“plan 缺失时可工作”的一部分 robustness
   - 但 ownership structure 仍然锁在：
     - `plan/direct ratio ~0.30`
     - `plan weight ~2.0/dim`
     - plan-dominant labels 不变

8. **基于这轮结果，training-time competition 这条线是否仍值得继续推进？**
   - **值得，但这轮更明确地说明：不能把希望放在“简单 early high-drop schedule”上。**
   - 更准确的表述是：
     - competition family 仍然值得推进
     - 但下一步需要更显式的 ownership shaping / asymmetry control
     - 而不是继续微调 fixed-vs-schedule dropout 这一维

### 33.7 one-paragraph judgement

- **这轮最小 `direct_pose_plan_drop_prob` schedule falsifier 给出的结论是否定的：`early high-drop -> later release` 并没有比 fixed `plan_drop=0.5` 更进一步改变 ownership structure。`1.0 -> 0.7 -> 0.3` 仍然能保持较好的 zero-plan robustness，但它的 `plan/direct sensitivity ratio` (`0.306839`) 反而高于 fixed `0.5` (`0.297436`)，`plan` first-layer weight 也没有脱离 `~2.0/dim`，`plan` ablation (`+0.003951`) 还比 fixed `+0.002094` 更差，最终 direct 指标也明显回退到 `0.177076`。`1.0 -> 0.5 -> 0.0` 更进一步说明：如果只是让模型早期在没有 plan 的条件下也能学会工作，而后期重新把 plan 放回去，它仍然会回到 plan-dominant regime。综合看，这轮结果更支持：fixed dropout 确实能修 robustness，但 ownership 还需要更显式的 shaping；training-time competition 这条线仍值得推进，但 simple schedule 已经基本被这轮 falsify 掉了。**

## 2026-04-07 raw70a non-plan readiness falsifier（C 优先）

### (1) Corrected factual basis

- baseline 的 `plan collapse` 主体发生在 `70a_replace_zerophase` warmstart surgery，不是后续 `70b` 60-step loss-driven collapse。
- 以下同口径 `direct_pose_head.0.weight` first-layer block weight / dim 数字直接继承，不在本轮重证：

| candidate | stage type | direct / dim | plan / dim | meas / dim |
|---|---|---:|---:|---:|
| `baseline raw 70a` | `raw70a` | `2.100333` | `2.020063` | `2.094611` |
| `baseline 70a_replace_zerophase` | `warmstart/zerophase` | `2.100333` | `0.000000` | `0.000000` |
| `baseline_replace final` | `70b final` | `2.101414` | `0.027915` | `0.030368` |
| `tailk7 raw 70a` | `raw70a` | `2.038038` | `2.011084` | `1.948133` |
| `tailk7 70a_replace_zerophase` | `warmstart/copy-only zerophase` | `2.038038` | `2.011084` | `1.948133` |
| `tailk7 coadapt final` | `70b final/coadapt` | `2.038038` | `2.011084` | `1.948133` |

### (2) Candidate table

| candidate | stage type | checkpoint path | purpose | self-contained? | eval artifact path | analysis artifact path |
|---|---|---|---|---|---|---|
| `baseline raw 70a` | `raw70a` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_posttrain_pipeline_from_bestfree_20260317/70a/ckpt_last_WalkF_stage7_70a_fromfresh_20260317.pth` | raw 70a anchor；检验 baseline 的 non-plan path 在 surgery 前是否已可用 | `true` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/eval_matrix/baseline_raw_70a/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/behavior/candidates/baseline_raw_70a.json` |
| `baseline 70a_replace_zerophase` | `warmstart/zerophase` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_posttrain_pipeline_from_bestfree_20260317/warmstart/ckpt_last_70a_replace_zerophase_20260317.pth` | 检验 baseline warmstart surgery 是创造能力还是暴露已 ready 的 non-plan path | `true` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/eval_matrix/baseline_70a_replace_zerophase/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/behavior/candidates/baseline_70a_replace_zerophase.json` |
| `baseline_replace final` | `70b final` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/ckpt_last_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.pth` | production reference；只作为 low-plan basin 最终参考锚点 | `true` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/eval_matrix/baseline_replace_final/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/behavior/candidates/baseline_replace_final.json` |
| `tailk7 raw 70a` | `raw70a` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_stage70a_from_tailfix_20260402/ckpt_last_WalkF_stage7_70a_lr3e4_from_cp015_tailk7_stage6tailfix_20260402.pth` | raw 70a falsifier 主角；检验 tailk7 的 non-plan readiness 是否已在 70a 出口更弱 | `true` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/eval_matrix/tailk7_raw_70a/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/behavior/candidates/tailk7_raw_70a.json` |
| `tailk7 70a_replace_zerophase` | `warmstart/copy-only zerophase` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/warmstart/ckpt_last_cp015_tailk7_70a_replace_zerophase_20260406.pth` | copy-only warmstart；检验 tailk7 只是进入 replace 后 copy 版本能否暴露 non-plan path | `true` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/eval_matrix/tailk7_70a_replace_zerophase/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/behavior/candidates/tailk7_70a_replace_zerophase.json` |
| `tailk7 baseline-style adapted warmstart` | `warmstart/baseline-style adapted` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_warmstart_contract_sentinel_20260402_warmstart_contract_sentinel/warmstart/ckpt_last_cp015_tailk7_70a_replace_baseline_style_20260402_warmstart_contract_sentinel.pth` | warmstart contract sentinel；检验 baseline-style adaptation 是否足以把 tailk7 拉进 baseline basin | `true` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/eval_matrix/tailk7_baseline_style_adapted_warmstart/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/behavior/candidates/tailk7_baseline_style_adapted_warmstart.json` |
| `tailk7 coadapt final` | `70b final/coadapt` | `/Users/xingzhaorui/PycharmProjects/PythonProject/models/__tmp_cp015_tailk7_replace_interface_coadapt_longer_push_20260406/coadapt_allrot_interface_bestlr_longer_4x/ckpt_last_WalkF_stage7_70b_replace_lowdrift_coadapt_allrot_interface_bestlr_longer_4x_20260406.pth` | tailk7 final reference；只作为 no-collapse 终态 readout / symptom 对照 | `true` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/eval_matrix/tailk7_coadapt_final/teacher_x_gt/plan_model__meas_model/Walk_F_freerun_cycles.json` | `/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_cp015_tailk7_raw70a_nonplan_readiness_falsifier_20260407/behavior/candidates/tailk7_coadapt_final.json` |

### (3) Behavior result table

| candidate | eval mode | model/model | plan=zero, meas=model | plan=gt, meas=model | model/zero | zero/zero | plan_score | meas_score | zero/zero delta | DirectGeoLocalDeg | 结论标签 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline 70a_replace_zerophase` | `freerun spot-check` | `0.341061` | `0.341061` | `0.341061` | `0.341061` | `0.341061` | `0.000000` | `0.000000` | `+0.000000` | `0.341061` | `non-plan ready` |
| `baseline 70a_replace_zerophase` | `teacher-conditioned / freerun_x_gt` | `0.341061` | `0.341061` | `0.341061` | `0.341061` | `0.341061` | `0.000000` | `0.000000` | `+0.000000` | `0.341061` | `non-plan ready` |
| `baseline raw 70a` | `freerun spot-check` | `0.295911` | `0.341061` | `0.291285` | `0.294252` | `0.341061` | `0.045150` | `0.045150` | `+0.045150` | `0.295911` | `plan-compensatory dependency` |
| `baseline raw 70a` | `teacher-conditioned / freerun_x_gt` | `0.294252` | `0.341061` | `0.291285` | `0.294252` | `0.341061` | `0.046808` | `0.046808` | `+0.046808` | `0.294252` | `plan-compensatory dependency` |
| `baseline_replace final` | `teacher-conditioned / freerun_x_gt` | `0.150492` | `0.151184` | `0.150220` | `0.150492` | `0.151184` | `0.000691` | `0.000691` | `+0.000691` | `0.150492` | `non-plan ready` |
| `tailk7 70a_replace_zerophase` | `freerun spot-check` | `0.213743` | `0.233657` | `0.212700` | `0.213743` | `0.233657` | `0.019914` | `0.019914` | `+0.019914` | `0.213743` | `plan-compensatory dependency` |
| `tailk7 70a_replace_zerophase` | `teacher-conditioned / freerun_x_gt` | `0.213743` | `0.233657` | `0.212700` | `0.213743` | `0.233657` | `0.019914` | `0.019914` | `+0.019914` | `0.213743` | `plan-compensatory dependency` |
| `tailk7 baseline-style adapted warmstart` | `freerun spot-check` | `0.228227` | `0.233657` | `0.242692` | `0.228227` | `0.233657` | `0.014464` | `0.005430` | `+0.005430` | `0.228227` | `mixed` |
| `tailk7 baseline-style adapted warmstart` | `teacher-conditioned / freerun_x_gt` | `0.228227` | `0.233657` | `0.242692` | `0.228227` | `0.233657` | `0.014464` | `0.005430` | `+0.005430` | `0.228227` | `mixed` |
| `tailk7 coadapt final` | `teacher-conditioned / freerun_x_gt` | `0.213743` | `0.233657` | `0.212700` | `0.213743` | `0.233657` | `0.019914` | `0.019914` | `+0.019914` | `0.213743` | `plan-compensatory dependency` |
| `tailk7 raw 70a` | `freerun spot-check` | `0.213743` | `0.233657` | `0.212700` | `0.213743` | `0.233657` | `0.019914` | `0.019914` | `+0.019914` | `0.213743` | `plan-compensatory dependency` |
| `tailk7 raw 70a` | `teacher-conditioned / freerun_x_gt` | `0.213743` | `0.233657` | `0.212700` | `0.213743` | `0.233657` | `0.019914` | `0.019914` | `+0.019914` | `0.213743` | `plan-compensatory dependency` |

### (4) Mechanism result table

| candidate | direct_feat sensitivity | plan sensitivity | meas sensitivity | plan/direct ratio | plan block weight / dim | direct block weight / dim | effective contribution proxy | plan ablation delta | 结论标签 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `baseline 70a_replace_zerophase` | `1.042844` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `2.100333` | `direct=0.444910; plan=0.000000; meas=0.000000` | `+0.000000` | `non-plan owned / low-plan` |
| `baseline raw 70a` | `1.041535` | `0.328888` | `0.741491` | `0.315773` | `2.020063` | `2.100333` | `direct=0.444910; plan=0.067373; meas=0.000000` | `+0.045673` | `direct path still essential` |
| `baseline_replace final` | `1.062731` | `0.012334` | `0.016853` | `0.011606` | `0.027915` | `2.101414` | `direct=0.445919; plan=0.000875; meas=0.000000` | `+0.000456` | `non-plan owned / low-plan` |
| `tailk7 70a_replace_zerophase` | `0.756979` | `0.269853` | `0.329924` | `0.356487` | `2.011084` | `2.038038` | `direct=0.389713; plan=0.065724; meas=0.000000` | `+0.017363` | `direct path still essential` |
| `tailk7 baseline-style adapted warmstart` | `0.768055` | `0.591559` | `0.674282` | `0.770204` | `2.796499` | `2.038038` | `direct=0.389713; plan=0.091479; meas=0.000000` | `+0.017958` | `plan-compensatory takeover` |
| `tailk7 coadapt final` | `0.756979` | `0.269853` | `0.329924` | `0.356487` | `2.011084` | `2.038038` | `direct=0.389713; plan=0.065724; meas=0.000000` | `+0.017363` | `direct path still essential` |
| `tailk7 raw 70a` | `0.756979` | `0.269853` | `0.329924` | `0.356487` | `2.011084` | `2.038038` | `direct=0.389713; plan=0.065724; meas=0.000000` | `+0.017363` | `direct path still essential` |

### (5) Hypothesis judgement table

| hypothesis | support level | strongest evidence | weakest point | baseline low-plan basin? | tailk7 no-collapse? | high-LR tailk7 worse? |
|---|---|---|---|---|---|---|
| `A` | `weak` | 没有足够强的新证据把 A 提到和 C 同级 | 即使 baseline planner 稍 noisy，也解释不了 tailk7 adapted warmstart 仍然不进入 low-plan basin | `no` | `no` | `no` |
| `B` | `partially supported` | raw 70a 的 zero/zero SIC / joint residual profile 确实不同，说明 replace step0 的 error mass 组成并不相同 | 目前只证明 residual composition 不同，还没用新的 step0/1 gradient audit 把它锁成主因 | `partial` | `partial` | `partial` |
| `C` | `partially supported` | tailk7 adapted warmstart 仍未进入 baseline low-plan basin，且 raw 70a 的 direct_feat sensitivity 低于 baseline raw 70a | 本轮 primary falsifier 是反向的：tailk7 raw 70a 的 plan=zero 并没有比 baseline raw 70a 更差 | `yes` | `partial` | `partial` |

### (6) Final judgement table

| candidate / hypothesis | root cause 前移到 stage6/70a | plan 更像 symptom / readout | warmstart surgery 不是主因 | non-plan readiness 是主矛盾 | recommended next role |
|---|---|---|---|---|---|
| `baseline raw 70a` | `baseline reference` | `yes` | `n/a` | `partial` | `research-only` |
| `baseline 70a_replace_zerophase` | `baseline reference` | `yes` | `n/a` | `partial` | `research-only` |
| `baseline_replace final` | `baseline reference` | `yes` | `n/a` | `partial` | `production` |
| `tailk7 raw 70a` | `partial` | `yes` | `n/a` | `partial` | `research-only` |
| `tailk7 70a_replace_zerophase` | `partial` | `yes` | `yes` | `partial` | `research-only` |
| `tailk7 baseline-style adapted warmstart` | `partial` | `yes` | `yes` | `partial` | `research-only` |
| `tailk7 coadapt final` | `partial` | `yes` | `n/a` | `partial` | `research-only` |
| `Hypothesis C` | `partial` | `yes` | `yes` | `yes` | `research-only` |

### (7) Direct answers

1. 不是。`tailk7 raw 70a` 的 `plan=zero` 并没有明显差于 `baseline raw 70a`；在 teacher audit 下反而更小：`+0.019914` vs `+0.046808`。
2. 因此，仅凭这轮 primary falsifier，还不足以把 root cause 更明确地前移到 `stage6/70a`；这轮更像是否掉了“强 C 版本”，而不是把 C 彻底证死。
3. `baseline 70a_replace_zerophase` 在本轮 `DirectGeoLocalDeg` 口径下，更像一个把 `plan/meas` 直接清零的 basin-entry surgery，不是“原本就 ready 的 non-plan path 被直接裸露出来”；因为它的 teacher `model/model=0.341061` 并不优于 baseline raw 70a。
4. 是。`tailk7 baseline-style adapted warmstart` 仍没有进入 baseline low-plan basin，且 teacher `zero/zero delta=+0.005430`、`plan/direct ratio=0.770204`；这更支持问题不在 warmstart surgery 本身，而在 donor-state / 70a exit basin。
5. 这轮我不再维持 `C > B >> A`。更像 `B ~ C >> A`；若必须排序，我会给 `B ≳ C >> A`，当前支持度分别是 `B=partially supported`、`C=partially supported`、`A=weak`。
6. 由于这轮没有继续把 stronger C 拉高，我现在更把 B 看成需要下一步直接验证的中层机制 / 共同驱动项，而不只是已经从属于 C 的附属解释。
7. 基于这轮结果，下一步最该打的仍是 `70a exit basin / donor-state` 线路，但最小新增 probe 应该是 raw70a 的 `step0/1 gradient composition audit`，不是 planner semantics 线路。

## 2026-04-07 raw70a C-only step0/1 optizability probe

这轮严格按你刚刚收窄后的要求，只打 `C` 的最小验证点：

- 不开新 60-step / 240-step lane
- 不做 per-SIC / per-joint 形态主分析
- 只做：
  - same-batch / same-loss / same-optimizer 的 `step0` gradient composition
  - same-batch / same-loss / same-optimizer 的 **1-step in-memory AdamW** optizability probe
  - 同时看 `model` 和 `plan=zero` 条件

本轮新产物：

- script:
  - `tools/run_cp015_tailk7_raw70a_c_optizability_probe.py`
- summary:
  - `debug_output/_tmp_cp015_tailk7_raw70a_c_optizability_probe_20260407/summary.json`
  - `debug_output/_tmp_cp015_tailk7_raw70a_c_optizability_probe_20260407/summary.md`

使用的 replace probe template 继续固定为：

- `debug_output/_tmp_cp015_tailk7_replace_3way_objective_ablation_20260402_3way_objective/configs/posttrain_70b_replace_lowdrift_e2x60_3way_arm125_lr5e5_from_cp015_tailk7_70a_20260402_3way_objective.json`

只比较三例：

- `baseline raw 70a`
- `tailk7 raw 70a`
- `tailk7 baseline-style adapted warmstart`

### C.1 step0 gradient composition（只看 local step0）

| candidate | `direct_grad/dim` | `plan_grad/dim` | `meas_grad/dim` | `plan/direct` | `direct signed proj / dim` | `plan signed proj / dim` | C 视角读法 |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline raw 70a` | `3.191341` | `1.556533` | `1.724358` | `0.487736` | `-0.215904` | `-0.121387` | `plan` 梯度占比**更高** |
| `tailk7 raw 70a` | `3.821238` | `1.108544` | `1.086417` | `0.290101` | `-0.044356` | `-0.046451` | `plan` 梯度占比**更低** |
| `tailk7 baseline-style adapted warmstart` | `3.190354` | `1.174065` | `1.341832` | `0.368005` | `+0.191000` | `+0.120007` | warmstart 后仍不是 baseline low-plan 几何 |

这里先给出最重要的直接判词：

- **如果 C 的局部版本是：`tailk7 raw 70a` 在 replace `step0` 上对 non-plan 更不友好，那么这个 probe 不支持。**
- 相反，`tailk7 raw 70a` 的 `step0 plan/direct grad ratio = 0.290101`，**低于** `baseline raw 70a = 0.487736`。
- 也就是说，单看 replace `step0` 的 first-layer local gradient composition，**看不出 tailk7 比 baseline 更“被迫依赖 plan”**。

### C.2 one-step optizability（只看 1 步）

| candidate | `step0 model arm` | `step1 model arm` | `model impr` | `step0 zero-plan arm` | `step1 zero-plan arm` | `zero-plan impr` | `step0 gap` | `step1 gap` | `gap shrink` | 归一化读法 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `baseline raw 70a` | `0.003955` | `0.003618` | `+0.000337` | `0.004278` | `0.003947` | `+0.000332` | `0.000324` | `0.000329` | `-0.000005` | zero-plan 改善约 `7.76%`，gap 基本不动 | **不支持 stronger C** |
| `tailk7 raw 70a` | `0.002056` | `0.001852` | `+0.000205` | `0.002305` | `0.001993` | `+0.000312` | `0.000248` | `0.000141` | `+0.000107` | zero-plan 改善约 `13.53%`，gap 反而缩小 | **更像可被一步推向 non-plan 一点点** |
| `tailk7 baseline-style adapted warmstart` | `0.003453` | `0.002518` | `+0.000935` | `0.002975` | `0.002543` | `+0.000432` | `-0.000478` | `0.000025` | `-0.000503` | zero-plan 也改善，但 `model` 改善更多，gap 恶化 | **warmstart 仍不是 basin 入口解** |

这张表对 `C` 的信息量比上一轮 primary falsifier 更直接：

1. **`tailk7 raw 70a` 并没有表现出“更难被一步更新推向 zero-plan 方向”。**
   - `zero-plan impr`：
     - baseline raw 70a: `+0.000332`
     - tailk7 raw 70a: `+0.000312`
   - 绝对值已经非常接近。
   - 归一化后甚至是：
     - baseline: `7.76%`
     - tailk7: `13.53%`

2. **`tailk7 raw 70a` 的 plan gap 在一步后是收缩的。**
   - `gap shrink = +0.000107`
   - baseline raw 70a 则几乎不变，甚至略负：`-0.000005`

3. 因此，**“tailk7 raw 70a 的 non-plan path 在 replace step0/1 上更难优化出来”这一版 C，也拿不到支持。**

### C.3 warmstart surgery 仍然不是主因，但理由要更精确

`tailk7 baseline-style adapted warmstart` 这例很关键，它给出的信号是：

- 一步更新后它的绝对 `arm` 指标确实能改善：
  - `model impr = +0.000935`
  - `zero-plan impr = +0.000432`
- 但它**不是**在往 baseline 式 non-plan basin 靠：
  - `step0 gap = -0.000478`
  - `step1 gap = +0.000025`
  - `gap shrink = -0.000503`

换句话说：

- baseline-style surgery **可以改变局部数值**
- 但**不能把 tailk7 直接送进 baseline low-plan basin 的局部几何**
- 所以“warmstart surgery 不是主因”这个 inherited judgement 仍成立
- 只是这轮更准确的说法应是：
  - **warmstart 不足以制造 baseline-style non-plan optizability geometry**

### C.4 对 C 的更新结论（只按这轮 C-only probe）

这轮应该把 `C` 再往下修正一层：

- **已经不支持：**
  - `tailk7 raw 70a` 更 zero-plan 不可用
  - `tailk7 raw 70a` 更难被一步更新推向 non-plan regime

- **仍可保留但必须重写的版本，只能是更高阶的：**
  - 如果 `C` 还要成立，它不能再是
    - raw 行为 ready-ness
    - 或 step0/1 local optizability
  - 它只能是更高阶的
    - **exit-basin / multi-step trajectory geometry**
    - **不是 1-step local geometry**

也就是说，当前最稳妥的表述已经不是：

- `C > B >> A`

而是：

- **这轮 C-only local falsifier 对 stronger C 是负的**
- `A` 仍弱
- `warmstart surgery` 仍不是主因
- 但 **C 若要保留，必须改写成“高阶 basin 几何问题”，不能再用 `raw70a zero-plan readiness` 或 `step0/1 local optizability` 来支撑**

### C.5 只回答这轮你要的 C 验证点

1. **`tailk7 raw 70a` 的 `step0/1` C probe，是否支持“它比 baseline raw 70a 更难进入 non-plan direction”？**
   - **不支持。**

2. **step0 gradient composition 是否支持 stronger C？**
   - **不支持。**
   - `tailk7 raw 70a plan/direct = 0.290101 < baseline raw 70a 0.487736`

3. **1-step zero-plan optizability 是否支持 stronger C？**
   - **不支持。**
   - `tailk7 raw 70a zero-plan impr = +0.000312`
   - `baseline raw 70a zero-plan impr = +0.000332`
   - 差距不构成 “tail 明显更难优化” 的证据

4. **tailk7 adapted warmstart 是否说明 warmstart surgery 不是主因？**
   - **是。**
   - 因为它虽能改善绝对指标，但没有把局部几何推向 baseline-style non-plan basin

5. **这轮之后，C 还剩下什么版本值得保留？**
   - 只剩：
     - **higher-order exit-basin / multi-step geometry**
   - 不再是：
     - zero-plan ready-ness
     - step0/1 local optizability

### C.6 one-paragraph judgement

- **这轮只针对 `C` 做的 same-batch `step0/1` probe 给出的结论也是负的：`tailk7 raw 70a` 并没有表现出比 `baseline raw 70a` 更差的 local non-plan optizability。相反，`step0` first-layer `plan/direct` gradient ratio 在 tailk7 raw 70a 上更低 (`0.290101 < 0.487736`)，而一步更新后 tailk7 raw 70a 的 zero-plan arm 指标也能改善 (`+0.000312`)，甚至其 plan-gap 还出现轻微收缩 (`+0.000107`)，baseline raw 70a 则几乎不动 (`-0.000005`)。因此，如果 C 要继续保留，就不能再写成“tailk7 raw 70a 的 non-plan path 在 replace step0/1 上更难被优化出来”；这条局部版本现在基本被 falsify。与此同时，`tailk7 baseline-style adapted warmstart` 虽然能改善绝对 arm 指标，但没有把局部几何推向 baseline-style non-plan basin，因此 warmstart surgery 仍不是主因。当前更精确的更新是：C 若存在，只能是一个 higher-order 的 exit-basin / multi-step trajectory geometry 问题，而不是 raw readiness 或 1-step local optizability 问题。**

## D. group-norm EMA initial-condition + rollout feedback audit（2026-04-07）

这轮严格只复用了：

- baseline full replace log：
  - `models/__tmp_posttrain_pipeline_from_bestfree_20260317/70b_replace_lowdrift/posttrain_log_WalkF_stage7_70b_replace_lowdrift_fromfresh_20260317.json`
- tailk7 matched copy-only replace log：
  - `models/__tmp_cp015_tailk7_replace_from_70a_20260402/lr5e5/posttrain_log_WalkF_stage7_70b_replace_lowdrift_lr5e5_from_cp015_tailk7_70a_20260402.json`
- replace proxy snapshot summary：
  - `debug_output/_tmp_cp015_tailk7_replace_efficiency_audit_20260402_arm_efficiency_audit/summary.json`
- adapted warmstart auxiliary summary：
  - `debug_output/_tmp_cp015_tailk7_warmstart_contract_sentinel_20260402_warmstart_contract_sentinel/summary.json`

本轮继承已知结论，不回头重证：

- 不是 `step0 gradient direction`
- 不是 LR 本身
- 不是 raw non-plan local readiness
- 不把 `B`（per-SIC / per-joint residual 形态）当主验证点

### D.1 corrected causal basis（本轮 code-grounded clarification）

| term | definition in code | optimized? | rollout observable? | where computed |
|---|---|---|---|---|
| `blend_loss_total` | `torch.stack(loss_terms).sum()`；是 rollout-side blend observable，不是 direct replace 的主优化目标 | No（在本 run 不是） | Yes | `train/posttrain.py:3650`, `train/posttrain.py:3943` |
| `dir_leg_base` / `dir_nonleg_base` | group-norm 前的 raw direct base losses；2-way direct replace 下后续 ratio 的分子 | Indirectly | Partly（logged raw terms） | `train/posttrain.py:3654-3656`, `train/posttrain.py:3979-3983` |
| `dir_group_norm_*_ema` | `_ema_prev()` 取上一轮 EMA；若不存在则退回当前 `base.detach()`，因此 `step1` seed = `step1 raw base` | State only | No（internal state, but logged） | `train/posttrain.py:3701-3713`, `train/posttrain.py:3770-3794`, `train/posttrain.py:4017-4024` |
| `dir_group_norm_*_raw` / `clamped` | `ratio_raw = base / ema_prev.clamp_min(eps)`；再 clamp 到 `[ratio_min, ratio_max]` | Yes，via `dir_geo` | Yes | `train/posttrain.py:3714-3719`, `train/posttrain.py:4027-4038` |
| `dir_geo` | 2-way replace 下 `w_leg * dir_group_norm_leg + w_nonleg * dir_group_norm_nonleg`；本 run 两个权重都为 `1.0` | Yes | Yes | `train/posttrain.py:3787`, `train/posttrain.py:3977` |
| `total` | `objective in ("direct", ...)` 时先设为 `dir_geo`；本两条日志里逐步满足 `total == dir_geo` | Yes | Yes | `train/posttrain.py:3798-3804`, logs verified |

补一条必须纠正的 factual basis：

- direct replace 里被优化的是 `total = dir_geo`，不是 `blend_loss_total`
- 因此 `step1 blend_loss` 的 baseline/tail 差异，首先只能证明 rollout-side observable 已经分开
- 不能直接等同于 `step1 effective objective` 已经分开

### D.2 trajectory divergence tables（delta = tail - baseline）

**blend_loss**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.273066 | 0.320877 | 0.047811 | 0.047811 | 0.047811 |
| 2 | 0.271454 | 0.401537 | 0.130083 | 0.088947 | 0.088947 |
| 3 | 0.266247 | 0.335480 | 0.069233 | 0.082376 | 0.082376 |
| 5 | 0.269626 | 0.341235 | 0.071609 | 0.075956 | 0.075956 |
| 10 | 0.266709 | 0.343795 | 0.077086 | 0.068122 | 0.072039 |
| 15 | 0.269133 | 0.343019 | 0.073886 | 0.063095 | 0.069058 |
| 20 | 0.262234 | 0.338987 | 0.076753 | 0.075857 | 0.070757 |
| 30 | 0.268851 | 0.329854 | 0.061003 | 0.066039 | 0.069563 |
| 40 | 0.263468 | 0.326018 | 0.062550 | 0.067804 | 0.069551 |
| 50 | 0.264832 | 0.323477 | 0.058645 | 0.065661 | 0.070207 |
| 60 | 0.256634 | 0.320264 | 0.063630 | 0.051988 | 0.067663 |

**dir_geo**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.000000 | 2.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 1.513519 | 2.302599 | 0.789081 | 0.394540 | 0.394540 |
| 3 | 1.638699 | 1.305490 | -0.333209 | 0.151957 | 0.151957 |
| 5 | 1.492925 | 1.416211 | -0.076714 | -0.016507 | -0.016507 |
| 10 | 1.491868 | 1.152068 | -0.339800 | -0.196625 | -0.106566 |
| 15 | 1.429623 | 1.201427 | -0.228196 | 0.022747 | -0.063462 |
| 20 | 1.189661 | 1.263412 | 0.073751 | 0.077832 | -0.028138 |
| 30 | 1.296521 | 1.387158 | 0.090638 | 0.241387 | 0.031591 |
| 40 | 1.414726 | 1.585197 | 0.170471 | 0.128248 | 0.063123 |
| 50 | 1.387699 | 1.461481 | 0.073781 | -0.106359 | 0.035211 |
| 60 | 1.420448 | 1.545607 | 0.125159 | 0.138268 | 0.036542 |

**total**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.000000 | 2.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 1.513519 | 2.302599 | 0.789081 | 0.394540 | 0.394540 |
| 3 | 1.638699 | 1.305490 | -0.333209 | 0.151957 | 0.151957 |
| 5 | 1.492925 | 1.416211 | -0.076714 | -0.016507 | -0.016507 |
| 10 | 1.491868 | 1.152068 | -0.339800 | -0.196625 | -0.106566 |
| 15 | 1.429623 | 1.201427 | -0.228196 | 0.022747 | -0.063462 |
| 20 | 1.189661 | 1.263412 | 0.073751 | 0.077832 | -0.028138 |
| 30 | 1.296521 | 1.387158 | 0.090638 | 0.241387 | 0.031591 |
| 40 | 1.414726 | 1.585197 | 0.170471 | 0.128248 | 0.063123 |
| 50 | 1.387699 | 1.461481 | 0.073781 | -0.106359 | 0.035211 |
| 60 | 1.420448 | 1.545607 | 0.125159 | 0.138268 | 0.036542 |

**dir_leg_base**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.013882 | 0.009822 | -0.004060 | -0.004060 | -0.004060 |
| 2 | 0.009130 | 0.009261 | 0.000132 | -0.001964 | -0.001964 |
| 3 | 0.009625 | 0.006170 | -0.003455 | -0.002461 | -0.002461 |
| 5 | 0.008846 | 0.006258 | -0.002588 | -0.002870 | -0.002870 |
| 10 | 0.007864 | 0.003931 | -0.003933 | -0.003262 | -0.003066 |
| 15 | 0.007481 | 0.003891 | -0.003590 | -0.002388 | -0.002840 |
| 20 | 0.005857 | 0.003815 | -0.002041 | -0.002006 | -0.002631 |
| 30 | 0.005234 | 0.003461 | -0.001773 | -0.001083 | -0.002234 |
| 40 | 0.004544 | 0.003971 | -0.000573 | -0.001187 | -0.002036 |
| 50 | 0.003879 | 0.002612 | -0.001266 | -0.002029 | -0.002057 |
| 60 | 0.003785 | 0.002743 | -0.001042 | -0.001018 | -0.001957 |

**dir_nonleg_base**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.003327 | 0.002217 | -0.001110 | -0.001110 | -0.001110 |
| 2 | 0.002847 | 0.003014 | 0.000166 | -0.000472 | -0.000472 |
| 3 | 0.003082 | 0.001524 | -0.001558 | -0.000834 | -0.000834 |
| 5 | 0.002718 | 0.001641 | -0.001077 | -0.001062 | -0.001062 |
| 10 | 0.002606 | 0.001384 | -0.001222 | -0.001026 | -0.001044 |
| 15 | 0.002139 | 0.001293 | -0.000846 | -0.000599 | -0.000895 |
| 20 | 0.001623 | 0.001261 | -0.000362 | -0.000526 | -0.000803 |
| 30 | 0.001453 | 0.001220 | -0.000233 | -0.000196 | -0.000642 |
| 40 | 0.001455 | 0.001113 | -0.000343 | -0.000231 | -0.000522 |
| 50 | 0.001355 | 0.001135 | -0.000221 | -0.000335 | -0.000474 |
| 60 | 0.001206 | 0.001029 | -0.000177 | -0.000254 | -0.000444 |

**dir_group_norm_leg_raw**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 0.657692 | 0.942969 | 0.285277 | 0.142638 | 0.142638 |
| 3 | 0.705445 | 0.630019 | -0.075426 | 0.069950 | 0.069950 |
| 5 | 0.666383 | 0.664567 | -0.001816 | 0.010333 | 0.010333 |
| 10 | 0.650991 | 0.467061 | -0.183930 | -0.106867 | -0.048267 |
| 15 | 0.680956 | 0.511382 | -0.169574 | -0.020427 | -0.038987 |
| 20 | 0.582154 | 0.550646 | -0.031508 | 0.005963 | -0.027750 |
| 30 | 0.647139 | 0.611404 | -0.035735 | 0.069010 | -0.008137 |
| 40 | 0.658817 | 0.823548 | 0.164732 | 0.048010 | -0.001609 |
| 50 | 0.619157 | 0.627967 | 0.008810 | -0.104123 | -0.024656 |
| 60 | 0.681956 | 0.737116 | 0.055159 | 0.102853 | -0.016576 |

**dir_group_norm_nonleg_raw**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 0.855826 | 1.359631 | 0.503804 | 0.251902 | 0.251902 |
| 3 | 0.933254 | 0.675472 | -0.257782 | 0.082007 | 0.082007 |
| 5 | 0.826542 | 0.751644 | -0.074898 | -0.026840 | -0.026840 |
| 10 | 0.840877 | 0.685008 | -0.155870 | -0.089757 | -0.058299 |
| 15 | 0.748667 | 0.690045 | -0.058622 | 0.043174 | -0.024475 |
| 20 | 0.607507 | 0.712766 | 0.105259 | 0.071869 | -0.000389 |
| 30 | 0.649381 | 0.775754 | 0.126373 | 0.172377 | 0.039728 |
| 40 | 0.755910 | 0.761649 | 0.005739 | 0.080238 | 0.064732 |
| 50 | 0.768542 | 0.833514 | 0.064972 | -0.002236 | 0.059867 |
| 60 | 0.738492 | 0.808491 | 0.070000 | 0.035415 | 0.053119 |

**dir_group_norm_leg**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 0.657692 | 0.942969 | 0.285277 | 0.142638 | 0.142638 |
| 3 | 0.705445 | 0.630019 | -0.075426 | 0.069950 | 0.069950 |
| 5 | 0.666383 | 0.664567 | -0.001816 | 0.010333 | 0.010333 |
| 10 | 0.650991 | 0.467061 | -0.183930 | -0.106867 | -0.048267 |
| 15 | 0.680956 | 0.511382 | -0.169574 | -0.020427 | -0.038987 |
| 20 | 0.582154 | 0.550646 | -0.031508 | 0.005963 | -0.027750 |
| 30 | 0.647139 | 0.611404 | -0.035735 | 0.069010 | -0.008137 |
| 40 | 0.658817 | 0.823548 | 0.164732 | 0.048010 | -0.001609 |
| 50 | 0.619157 | 0.627967 | 0.008810 | -0.104123 | -0.024656 |
| 60 | 0.681956 | 0.737116 | 0.055159 | 0.102853 | -0.016576 |

**dir_group_norm_nonleg**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 0.855826 | 1.359631 | 0.503804 | 0.251902 | 0.251902 |
| 3 | 0.933254 | 0.675472 | -0.257782 | 0.082007 | 0.082007 |
| 5 | 0.826542 | 0.751644 | -0.074898 | -0.026840 | -0.026840 |
| 10 | 0.840877 | 0.685008 | -0.155870 | -0.089757 | -0.058299 |
| 15 | 0.748667 | 0.690045 | -0.058622 | 0.043174 | -0.024475 |
| 20 | 0.607507 | 0.712766 | 0.105259 | 0.071869 | -0.000389 |
| 30 | 0.649381 | 0.775754 | 0.126373 | 0.172377 | 0.039728 |
| 40 | 0.755910 | 0.761649 | 0.005739 | 0.080238 | 0.064732 |
| 50 | 0.768542 | 0.833514 | 0.064972 | -0.002236 | 0.059867 |
| 60 | 0.738492 | 0.808491 | 0.070000 | 0.035415 | 0.053119 |

**dir_group_norm_leg_ema**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.013882 | 0.009822 | -0.004060 | -0.004060 | -0.004060 |
| 2 | 0.013882 | 0.009822 | -0.004060 | -0.004060 | -0.004060 |
| 3 | 0.013644 | 0.009794 | -0.003850 | -0.003990 | -0.003990 |
| 5 | 0.013275 | 0.009417 | -0.003858 | -0.003932 | -0.003932 |
| 10 | 0.012080 | 0.008417 | -0.003663 | -0.003732 | -0.003832 |
| 15 | 0.010987 | 0.007610 | -0.003377 | -0.003545 | -0.003736 |
| 20 | 0.010060 | 0.006929 | -0.003131 | -0.003250 | -0.003615 |
| 30 | 0.008088 | 0.005661 | -0.002427 | -0.002641 | -0.003345 |
| 40 | 0.006897 | 0.004822 | -0.002075 | -0.002165 | -0.003066 |
| 50 | 0.006264 | 0.004160 | -0.002105 | -0.002062 | -0.002856 |
| 60 | 0.005550 | 0.003721 | -0.001829 | -0.001941 | -0.002714 |

**dir_group_norm_nonleg_ema**

| step | baseline | tail | delta | 5-step MA delta | cumulative mean delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.003327 | 0.002217 | -0.001110 | -0.001110 | -0.001110 |
| 2 | 0.003327 | 0.002217 | -0.001110 | -0.001110 | -0.001110 |
| 3 | 0.003303 | 0.002256 | -0.001046 | -0.001089 | -0.001089 |
| 5 | 0.003289 | 0.002184 | -0.001105 | -0.001089 | -0.001089 |
| 10 | 0.003099 | 0.002020 | -0.001079 | -0.001095 | -0.001092 |
| 15 | 0.002857 | 0.001873 | -0.000983 | -0.001038 | -0.001074 |
| 20 | 0.002672 | 0.001769 | -0.000902 | -0.000930 | -0.001038 |
| 30 | 0.002238 | 0.001573 | -0.000665 | -0.000731 | -0.000954 |
| 40 | 0.001925 | 0.001461 | -0.000464 | -0.000488 | -0.000848 |
| 50 | 0.001763 | 0.001361 | -0.000402 | -0.000403 | -0.000761 |
| 60 | 0.001634 | 0.001273 | -0.000360 | -0.000362 | -0.000697 |

这组表的最关键读法不是单点，而是分三层：

1. `blend_loss`：
   - `step1` 就已经分开，而且之后 60 步都稳定保持 tail > baseline
   - 所以 rollout-side observable 的分叉非常早

2. `dir_leg_base + dir_nonleg_base`（raw objective proxy）：
   - 差异一直非常小，而且大部分时候还是 tail < baseline
   - 没有出现与 `dir_geo` 同量级的 trajectory amplification

3. `dir_geo / total` 与 group-norm ratios：
   - `step2` 先出现一个很大的正脉冲
   - `step3-21` 仍有符号摆动
   - 更像从 `step22` 左右起，5-step MA 转成持续正偏；`step29-33` 最明显
   - cumulative mean delta 在 `step30` 首次转正

这比“step1 cliff”更像：

- `step1 observable already split`
- `step2 effective-objective impulse`
- `step22-30 stable positive trajectory emergence`

### D.3 EMA transmission table（step1 -> step2）

| candidate | step1 `dir_leg_base` | step1 `dir_nonleg_base` | step1 EMA seed | step2 raw ratio | step2 clamped ratio | reconstructed step2 `dir_geo` | logged step2 `dir_geo` | reconstruction error | conclusion tag |
|---|---:|---:|---|---|---|---:|---:|---:|---|
| baseline | 0.013882 | 0.003327 | leg=0.013882; nonleg=0.003327 | leg=0.657692; nonleg=0.855826 | leg=0.657692; nonleg=0.855826 | 1.513519 | 1.513519 | 1.10e-07 | exact log reconstruction |
| tail | 0.009822 | 0.002217 | leg=0.009822; nonleg=0.002217 | leg=0.942969; nonleg=1.359631 | leg=0.942969; nonleg=1.359631 | 2.302600 | 2.302599 | 8.62e-08 | exact log reconstruction |

这张表已经说明：

- `step2 dir_geo` 的 log 值，可以被 `step1 EMA seed + step2 raw base` **同口径精确反算**
- 所以 `step2` 的 effective-objective 分歧，不需要引入额外假说
- 它就是 group-norm transmission 在工作

但还可以更进一步，问一个更尖锐的问题：

- `step2 gap` 里，到底多少是 `step1 seed` 的锅？
- 多少才是 `step2 raw base` 自身已经分开？

Cross-swap decomposition（`step2 dir_geo` gap = tail - baseline）：

| decomposition | value | share of actual step2 gap |
|---|---:|---:|
| actual step2 gap | 0.789081 | 1.000000 |
| seed effect on baseline step2 base (`baseline step2 base + tail seed` minus baseline actual) | 0.700578 | 0.887840 |
| seed effect on tail step2 base (tail actual minus `tail step2 base + baseline seed`) | 0.729562 | 0.924572 |
| raw-base effect under baseline seed (`tail step2 base + baseline seed` minus baseline actual) | 0.059519 | 0.075428 |
| raw-base effect under tail seed (tail actual minus `baseline step2 base + tail seed`) | 0.088503 | 0.112160 |

这基本给出本轮最关键的 falsifier 结果：

- `step2 dir_geo gap` 里，**约 89%~92% 来自 `step1 EMA seed` 差异**
- `step2 raw base` 自身只解释 **约 8%~11%**

因此，若问：

- `step1 raw group losses + EMA seed` 是否足以解释 `step2 dir_geo / total` 分歧？

答案不是“部分”，而是：

- **大部分是，而且其中主导项就是 seed。**

### D.4 group-norm counterfactual table（analysis-only）

counterfactual 只做 analysis，不改模型、不重训：

- 用 `raw-objective proxy = dir_leg_base + dir_nonleg_base`
- 对比原始 `dir_geo` gap 与去掉 group-norm 后的 raw gap

| candidate / pair | original `dir_geo` gap (mean abs) | raw-objective gap (mean abs) | gap reduction ratio | whether divergence mostly appears after group-norm |
|---|---:|---:|---:|---|
| baseline vs tail / early step1-10 | 0.270606 | 0.004169 | 0.984593 | Yes |
| baseline vs tail / mid step11-30 | 0.202579 | 0.002332 | 0.988489 | Yes |
| baseline vs tail / late step31-60 | 0.222768 | 0.001941 | 0.991287 | Yes |
| baseline vs tail / all step1-60 | 0.224011 | 0.002443 | 0.989096 | Yes |

这张表几乎把 root-cause localization 直接钉死了：

- 如果把 group-norm 反算掉，trajectory gap 会缩到原来的 **~1%**
- 也就是说，baseline vs tail 的 replace divergence **并不是 raw objective 层已经同样分开**
- 它主要是 **group-norm 之后** 才被放大出来

### D.5 final judgement table

| hypothesis | support level | strongest evidence | weakest point | explains baseline low-plan basin? | explains tailk7 no-collapse? | explains high-LR tailk7 worse? |
|---|---|---|---|---|---|---|
| `step1 initial loss profile -> EMA seed` 已足以把 `step2` effective objective 拉开 | strongly supported | `step2 dir_geo` 可由 `step1 seed + step2 raw base` exact reconstruction；cross-swap 显示约 `89%~92%` 的 `step2 gap` 来自 seed 差异 | 只覆盖 `step1 -> step2`，不单独证明后续所有步 | Partly | Partly | Partly |
| 稳定轨迹分叉首先出现于 rollout observable，而不是 raw objective | strongly supported | `blend_loss` 从 `step1` 起始终正偏；raw-objective proxy gap 全程只有约 `0.001~0.005` | `dir_geo` 本身在 `step2-21` 仍有符号摆动 | Yes | Yes | Partly |
| 20~30 steps 的 replace divergence 主要是 group-norm EMA + rollout feedback 累积放大 | strongly supported | `dir_geo` 5-step MA 在 `step22-30` 转为持续正偏，cummean 在 `step30` 由负转正；raw base gap 不同步放大 | 未直接观测同一步 plan-weight readout 时间线 | Yes | Yes | Yes (mechanistically consistent) |
| `plan weight collapse / no-collapse` 更像 side effect / readout，而非 root cause 本体 | partially supported | 本轮最早分叉先出现在 `blend_loss` / group-norm ratio / `dir_geo`，早于任何本轮直接 plan readout | 本轮 artifacts 不含同步 plan-weight 日志，只能间接支持 | Yes | Yes | Partly |

### D.6 direct answers

1. **stable divergence 最早在第几步出现？**
   - 如果按 rollout observable 记：`blend_loss` 在 `step1` 就已出现稳定分叉，`step2` 进一步放大
   - 如果按 effective objective (`dir_geo / total`) 记：`step2` 有首个大脉冲，但稳定正偏窗口更像从 `step22` 左右开始，并在 `step29-33` 最明显；cumulative mean 于 `step30` 转正

2. **它首先体现在 `blend_loss`、`dir_geo`，还是 group-norm ratio？**
   - 最早是 `blend_loss`
   - 若限定 direct replace 的真正优化链，则首先是 `step2` 的 group-norm ratio / `dir_geo`
   - 不是 raw base objective

3. **`step1 -> step2` 的分歧是否大部分可由 EMA seed 解释？**
   - **是。**
   - cross-swap 显示 `step2 dir_geo gap` 约 `89%~92%` 由 `step1 seed` 差异贡献，`step2 raw base` 只解释约 `8%~11%`

4. **如果把 group-norm 反算掉，baseline vs tail 的 trajectory gap 是否明显变小？**
   - **是，而且是数量级级别地变小。**
   - `dir_geo` mean-abs gap 为 `0.202579~0.270606`
   - raw-objective proxy 仅 `0.001941~0.004169`
   - 缩减约 `98.46%~99.13%`

5. **若是，是否可以把 root cause 更精确地定位到：`initial loss profile -> group-norm EMA initial condition -> rollout feedback`？**
   - **可以，而且这是本轮最强支持的改写。**
   - 更精确地说：
     - `step1` raw loss profile 先不同
     - 直接写入 `EMA seed`
     - `step2` 通过 group-norm ratio 立刻放大
     - 后续在 multi-cycle rollout feedback 中逐步累积
     - 到 `step22-30` 形成稳定的 effective-objective trajectory divergence

6. **在这个改写下，`plan weight collapse` 是否更像 side effect / readout 而非 root cause？**
   - **更像 side effect / readout。**
   - 但要保守表述为：
     - 本轮是**间接强支持**
     - 因为最早分叉已在 `blend_loss` / group-norm ratio / `dir_geo` 里出现
     - 而本轮 artifacts 没有同步的 plan-weight trajectory 日志，所以“不是 root cause”还不算 direct proof，只是目前最贴合数据的 causal rewrite

### D.7 one-paragraph judgement

- **这轮 analysis-only audit 对 “`group-norm EMA initial-condition + rollout feedback` 是 replace trajectory divergence 起点” 给出强支持。首先，code 口径上 direct replace 优化的是 `total = dir_geo`，不是 `blend_loss_total`；因此 `blend_loss` 只是更早分开的 rollout observable，而不是 objective 本体。其次，`step1 -> step2` 的最小传导链可以被现有日志精确反算：`step1` 的 raw `dir_leg_base / dir_nonleg_base` 直接成为 `EMA seed`，而 `step2 dir_geo` 几乎可以完全由 `step1 seed + step2 raw base` 重构；更关键的是，cross-swap 显示 `step2 dir_geo gap` 约 `89%~92%` 由 seed 差异贡献，`step2 raw base` 自身只解释约 `8%~11%`。再次，把 group-norm 从 logged raw terms 中反算掉后，baseline vs tail 的 trajectory gap 会缩减 `98%~99%`，说明稳定分叉并不是 raw objective 层原本就同等分开，而是主要出现在 group-norm 之后。最后，`dir_geo` 本身虽然在 `step2-21` 仍有摆动，但从 `step22` 左右起其 5-step moving average 转为持续正偏，且 cumulative mean 在 `step30` 转正，吻合“20~30 steps 后形成稳定 trajectory divergence”的改写。因此，目前最像的 root cause 应定位为：`initial loss profile -> group-norm EMA initial condition -> rollout feedback`；在这一改写下，`plan weight collapse / no-collapse` 更像是这条 trajectory divergence 的伴随 readout / side effect，而不是 root cause 本体。**

## E. P1 entry-state contract audit（2026-04-07）

本节目标不是再看 20~60 step trajectory，而是把分叉继续往前推到：

- 同一个 replace config
- 同一个 batch
- 同一个 seed
- 同一个 `idx`
- 只看 `t=0` 的单步 rollout entry

并回答：

- baseline vs tail 是否在 **外部 rollout state** 上已经不同？
- 如果不同，最早不同的是哪一层？
- 这个最早不同是否足以解释 `step1 raw loss profile`

### E.1 exact probe setup

本轮 P1 只对当前 2-way full replace 本体做 deterministic single-step audit：

- baseline config：
  - `debug_output/_tmp_posttrain_pipeline_from_bestfree_20260317/configs/posttrain_70b_replace_lowdrift_fromfresh_20260317.json`
- tail config：
  - `debug_output/_tmp_cp015_tailk7_replace_from_70a_20260402/configs/posttrain_70b_replace_lowdrift_lr5e5_from_cp015_tailk7_70a_20260402.json`

它们共享：

- `batch = 1`
- `seed = 0`
- `seq_len = 87`
- `rollout_steps = 0`
- `rollout_cycles = 5`
- `rollout_include_boundary = True`
- `objective = direct`
- `direct_pose_loss_leg_split = True`

本轮通过拦截以下代码路径做同口径单步 probe：

- `_rollout_step_common`
- `_prepare_rollout_cond`
- `_prepare_rollout_contacts_input`
- `_lambda_rollout_unroll_single_step`

对应主链代码位置：

- `train/posttrain.py:1780`
- `train/posttrain.py:1852`
- `train/posttrain.py:1982`
- `train/posttrain.py:3000`
- `train/posttrain.py:3185`

### E.2 stage-by-stage entry audit

#### E.2.1 batch contract

baseline vs tail 在 dataset / batch 输入上完全一致：

- `motion`, `gt_motion`, `cond_in`, `cond_tgt_raw`
- `cond_norm_mu`, `cond_norm_std`
- `contacts`, `angvel`, `pose_hist`, `start`

全部是 bitwise equal。

#### E.2.2 pre-step external rollout state

在进入 first forward 之前，baseline vs tail 的外部 rollout state 也完全一致：

| item | result |
|---|---|
| `state["motion"]` | identical |
| `state["motion_raw"]` | identical |
| `state["y_prev_raw"]` | identical |
| `state["plan_z"]` | both `None` |
| `state["meas_logits_prev"]` | both `None` |
| `idx` | both `45` |
| `total_steps` | both `434` |
| `cycle_len` | both `87` |

这一步非常关键，因为它直接排除了更强版本的：

- “baseline/tail 在 replace entry 之前就已经有不同的外部 rollout state”

至少在当前 full replace 本体上，这个版本 **不成立**。

#### E.2.3 prepared entry inputs

在 `t=0` 时，`cond` 和时间索引仍完全一致：

| item | result |
|---|---|
| `cond_t` | identical |
| `cond_raw_step` | identical |
| `time_index_t` | identical |
| `rollout_step_t` | identical |
| `pose_hist_t` | identical |

唯一最早出现的 prepared-input 差异是：

| item | baseline | tail | delta |
|---|---:|---:|---:|
| `contacts_in_t` mean abs | `0.481094` | `0.464612` | `ΔL2 = 0.033952` |

也就是说，严格按 stage 划分：

- **第一个不相同的 tensor 是 `contacts_in_t`**
- 但它只是一个很小的差异

### E.3 same-step first forward divergence

尽管 prepared input 里只有 `contacts_in_t` 有小差异，但 first forward 输出已经大幅分叉：

| tensor | delta L2 | cosine | readout |
|---|---:|---:|---|
| `ret.out_direct` | `8.317197` | `0.864391` | direct readout 已大幅不同 |
| `ret.out` | `5.811278` | `0.056968` | inc/blend side output 也显著不同 |
| `ret.plan_z_next` | `6.059849` | `-0.259393` | plan latent 也同步分叉 |
| `ret.contacts_plan_logits` | `1.117334` | `0.999331` | 绝对值变了，但方向相近 |
| `ret.direct_leg_omega` | `0.112198` | `0.928614` | leg residual head 也不同，但量级远小于 `out_direct` |

所以从 causal ordering 看：

- `contacts_in_t` 的小差异先出现
- 但真正大的分叉出现在 **same-input first forward semantics**
- 尤其是 `ret.out_direct`

### E.4 first-step raw loss profile

`t=0` 单步 accum 里，baseline vs tail 的 raw loss profile 已经分开：

| item | baseline | tail | tail - baseline |
|---|---:|---:|---:|
| `dir_base_terms[0]` | `0.00307049` | `0.00123503` | `-0.00183546` |
| `dir_leg_base_terms[0]` | `0.00229380` | `0.00103538` | `-0.00125842` |
| `dir_nonleg_base_terms[0]` | `0.00077670` | `0.00019965` | `-0.00057704` |
| `loss_terms[0]` | `0.00139569` | `0.00131429` | `-0.00008140` |
| `inc_terms[0]` | `0.00139569` | `0.00131429` | `-0.00008140` |

这和前面 full-log 看到的事实一致：

- tail 的 `step1 raw base losses` 更小
- 这些更小的 raw losses 随后被写成更小的 EMA seed
- 再在 `step2` 被 group-norm 放大成更大的 `dir_geo`

### E.5 contacts counterfactual（analysis-only）

P1 里最重要的一个反问是：

- 如果最早不同的 prepared input 是 `contacts_in_t`
- 那它是不是就是 first-step raw loss profile 分叉的根因？

为此做了最小 counterfactual：

- baseline model 保持不变，只把 `contacts_in_t` 换成 tail 的
- tail model 保持不变，只把 `contacts_in_t` 换成 baseline 的
- 不改任何权重，不训练

结果：

| metric | baseline native | baseline swapped | tail native | tail swapped |
|---|---:|---:|---:|---:|
| `dir_base_terms[0]` | `0.00307049` | `0.00307049` | `0.00123503` | `0.00123992` |
| `dir_leg_base_terms[0]` | `0.00229380` | `0.00229380` | `0.00103538` | `0.00103574` |
| `dir_nonleg_base_terms[0]` | `0.00077670` | `0.00077670` | `0.00019965` | `0.00020418` |

关键读法：

- actual `dir_base_terms[0]` gap = `-0.00183546`
- swap `contacts_in_t` 对 tail 的影响只有 `-4.89e-06`
- 占 actual gap 的比例约 `0.27%`

而对 `ret.out_direct` 也是一样：

| comparison | `out_direct` L2 |
|---|---:|
| baseline native vs tail native | `8.317197` |
| baseline native vs baseline swapped | `0.000000` |
| tail native vs tail swapped | `0.011759` |

因此：

- `contacts_in_t` 确实是第一个不同的 prepared input
- **但它对 first-step raw loss profile 的贡献几乎可以忽略**
- 它不是当前 baseline vs tail 分叉的主因

### E.6 same-input activation audit

进一步把 `contacts_in_t` 固定成同一个（baseline contacts），再做 first-forward activation 对比。

结果：

| module | baseline mean abs | tail mean abs | delta L2 | cosine |
|---|---:|---:|---:|---:|
| `direct_pose_head` | `0.108106` | `0.151396` | `7.784381` | `0.211633` |
| `direct_pose_leg_head` | `0.030592` | `0.040816` | `0.111951` | `0.928714` |
| `direct_pose_arm_proj` | `0.137932` | `0.205214` | `7.050559` | `0.227254` |
| `direct_pose_else_proj` | `0.149016` | `0.187998` | `6.041028` | `0.288498` |
| `direct_pose_out_leg` | `0.736022` | `0.338193` | `5.348294` | `0.916738` |
| `direct_pose_out_arm` | `0.434289` | `0.425991` | `3.855028` | `0.955342` |
| `direct_pose_out_else` | `0.398578` | `0.293830` | `5.067666` | `0.383884` |

这个表把 P1 的判词再收紧了一层：

- 当 external state 和 `contacts_in_t` 都对齐后
- **最早的大分叉已经出现在 `direct_pose_head` shared trunk features**
- 不是先出现在 leg residual head
- 也不是只在最后 readout 才突然分开

因此 P1 之后更精确的定位是：

- 不是 “rollout external state contract broken”
- 也不是 “contacts_in_t mismatch drives everything”
- 而是：
  - **same-entry same-input 下，baseline vs tail 的 first-forward shared representation contract 已经不同**
  - `step1 raw loss profile` 只是这个 same-forward semantic split 的直接 readout

### E.7 P1 judgement

| hypothesis | support level | evidence |
|---|---|---|
| baseline vs tail 在 replace entry 前就有不同的外部 rollout state | not supported | `batch`, `motion`, `motion_raw`, `y_prev_raw`, `cond_t`, `time_index_t`, `rollout_step_t` 全部相同 |
| 最早不同的是 prepared input `contacts_in_t` | weak but true | `contacts_in_t` 是第一个不同的 tensor，但量级很小 |
| `contacts_in_t` 差异足以解释 first-step raw loss profile gap | not supported | contact-swap 只改变 `dir_base_terms[0]` 约 `4.9e-06`，远小于 actual gap `1.835e-03` |
| 真正关键的分叉来自 same-input first forward semantics | strongly supported | `ret.out_direct` L2=`8.317`；same-contacts 条件下 `direct_pose_head` L2=`7.784` / cosine=`0.212` |
| P1 之后的下一步该看 direct head / shared trunk semantic contract，而不是 planner 或 long rollout | strongly supported | earliest meaningful divergence 已锁定在 `direct_pose_head` shared features |

### E.8 one-paragraph judgement

- **P1 的 deterministic entry-state audit 把问题继续前推了一层，而且结论比“initial rollout-state 不同”更精确：baseline vs tail 在当前 2-way full replace 本体上，并不存在显著的外部 rollout entry-state mismatch。`batch`、`motion`、`motion_raw`、`y_prev_raw`、`cond_t`、`time_index_t`、`rollout_step_t` 都完全一致，最早出现的 prepared-input 差异只是一个很小的 `contacts_in_t` 偏移（`ΔL2 = 0.03395`）。但 contact-swap counterfactual 进一步表明，这个差异对 `step0` raw loss profile 几乎没有贡献：`dir_base_terms[0]` 的 actual gap 是 `1.835e-03`，而交换 `contacts_in_t` 只能改变约 `4.9e-06`。真正大的分叉已经在 same-input first forward 里出现，尤其是 `ret.out_direct` (`ΔL2 = 8.317`)；当把 `contacts_in_t` 也固定为同一个时，`direct_pose_head` shared trunk activation 仍然有 `ΔL2 = 7.784`、cosine 仅 `0.212`。因此，P1 把 root-cause localization 从“EMA / rollout feedback”更早地前推到了“same-entry same-input 下的 first-forward shared representation contract 已经不同”；`step1 raw loss profile` 是这个 semantic split 的直接 readout，后面的 `EMA seed -> group-norm -> rollout feedback` 只是把它持续放大。基于这个结果，下一步最值得做的不是 planner 主线，也不是再查外部 rollout state，而是做 same-input 的 direct-head / shared-trunk module attribution。**

### E.9 same-input module attribution / module-swap inference

运行脚本（历史记录）：

- same-input module attribution originally used a dedicated probe script that was later deleted during the 2026-04-18 posttrain compat cleanup.
- summary: `debug_output/_tmp_cp015_tailk7_same_input_module_attribution_20260407/summary.json`

关键结果（这里的 `delta_l2` 是当前 probe 使用的 **RMS-normalized L2**，与上文 raw L2 量纲不同）：

- same-input controls 全对齐：`state/cond/contacts/angvel/pose_history/rollout_step` gap 都为 `0`
- internal plan control 仍有差异，但 direct-head plan 输入贡献很小：
  - `contacts_plan(control)` `delta_l2 = 0.044554`, cosine `= 0.998952`
  - `plan_input_override` 后 `dir_base_terms[0]` closure 只有 `0.0518`
- 最早的低-cosine semantic split 仍在 direct trunk 边界：
  - `direct_pose_head` `delta_l2 = 0.334611`, cosine `= 0.206900`
  - `direct_pose_arm_proj` `delta_l2 = 0.425278`, cosine `= 0.228903`
  - `direct_pose_else_proj` `delta_l2 = 0.364296`, cosine `= 0.292336`
- `step0` raw loss gap 几乎全是 leg slice：
  - `|dir_leg_base_gap| / |dir_base_gap| = 0.9366`
  - `|dir_nonleg_base_gap| / |dir_base_gap| = 0.0634`
- 但这不意味着 leg module 是 root cause：
  - `activation:direct_pose_out_leg + direct_pose_leg_head` 可把 `dir_leg_base_gap` 关到 `0`，`dir_base_gap` closure `= 0.9366`
  - 可是 `weight:direct_pose_out_leg`、`weight:direct_pose_leg_head` 单独都**不能**关闭 gap，反而把 `dir_base_gap` 放大为正偏移
- `direct_pose_head` 不是单模块 sufficient：
  - `activation:direct_pose_head` 与 `weight:direct_pose_head` 都把 `out_direct` / `dir_base` gap **放大**
  - 说明不是“只换 head 就能回到 baseline”，而是 **head 与下游 readouts 已形成 joint contract**
- exhaustive 7-module weight-subset search 显示：
  - 惟一能同时把 `out_direct` 与 `dir_base/leg/nonleg` gap 都关到 `>0.99 closure` 的集合是  
    `direct_pose_head + direct_pose_arm_proj + direct_pose_else_proj + direct_pose_out_leg + direct_pose_out_arm + direct_pose_out_else + direct_pose_leg_head`
  - 去掉 `direct_pose_head` 的 “all-direct-modules-no-head” 组合仍明显失败

当前判词：

- **earliest large split 更像是 `direct_pose_head` 边界开始出现的 shared-representation / readout-contract split**
- **`direct_pose_head` 自身不是近似充分条件；必须连同 downstream direct readouts 一起看**
- **arm/else proj 更像 contract-coupled amplifier / branch adapter，不是可单独抽离的 root module**
- **leg readout / leg head 主要是 dominant readout（因为当前 gap 主要落在 leg loss），不是 earliest source**
- **下一步最值得做的最小 intervention 不是 contacts，也不是 planner，而是 direct branch 的 contract-preserving intervention：优先从 `direct_pose_head` 出发，但必须配合一小组下游 readout（至少 leg / arm / else readout，必要时连同 arm/else proj 与 leg head）。**

### E.10 Phase-2 staged sufficiency / staged intervention decomposition

这一节**不回头重证 E.9**，只在其上继续做 staged intervention design。

继承不变的前提是：

- earliest semantic split 起点仍在 `direct_pose_head` boundary
- `step0` raw gap 仍是明显的 leg-dominant readout
- `direct_pose_head` 不是单模块 sufficient
- 最强 sufficiency 结论仍是 whole direct-branch contract mismatch，而不是 isolated single-module root cause

本节新增问题不是“能不能全关掉”，而是：

- 是否存在解释力更强的 **leg-first staged path**
- 是否存在解释力更强的 **nonleg-first staged path**
- `direct_pose_head`、arm/else proj、leg modules 在 staged graph 中到底分别扮演什么角色

运行脚本（历史记录，同一 deterministic single-step / first-forward probe）：

- same-input attribution / staged-search originally used a dedicated probe script that was later deleted during the 2026-04-18 posttrain compat cleanup.
- updated summary: `debug_output/_tmp_cp015_tailk7_same_input_module_attribution_20260407/summary.json`

脚本在原有 same-input attribution 基础上新增了：

- 7 个 direct modules 的 exhaustive weight-subset search
- 7 个 direct modules 的 exhaustive activation-subset search
- staged candidate table
- leg-first / nonleg-first best path synthesis
- head-anchor analysis
- interaction / synergy judgement

#### E.10.1 stage candidates table

先把当前最有解释力的 candidate subset 摘出来：

| candidate set | size | out_direct closure | dir_base closure | dir_leg closure | dir_nonleg closure | staged interpretation |
|---|---:|---:|---:|---:|---:|---|
| `activation:{direct_pose_out_leg, direct_pose_leg_head}` | 2 | `0.2314` | `0.9366` | `1.0000` | `0.0000` | 最干净的 `leg-readout` stage-1；只关掉 dominant leg readout，不碰 nonleg |
| `activation:{direct_pose_out_arm, direct_pose_out_else}` | 2 | `0.3602` | `0.0634` | `0.0000` | `1.0000` | 最干净的 `nonleg-readout` stage-1；只关掉 residual nonleg readout，不碰 leg |
| `activation:{direct_pose_out_leg, direct_pose_leg_head, direct_pose_out_arm, direct_pose_out_else}` | 4 | `1.0000` | `1.0000` | `1.0000` | `1.0000` | 最小 pure-readout `near-sufficient` activation intervention |
| `weight:{direct_pose_head}` | 1 | `-0.5143` | `-35.2178` | `-22.6848` | `-220.3522` | `head-anchor` 但单独换会放大 gap；不是 standalone fix |
| `weight:{direct_pose_head, direct_pose_out_leg, direct_pose_leg_head}` | 3 | `-0.2370` | `-13.0321` | `0.9972` | `-220.3522` | head+leg path 在 weight 空间只关掉 leg，nonleg 仍严重错配 |
| `weight:{direct_pose_head, direct_pose_arm_proj, direct_pose_out_leg, direct_pose_out_arm, direct_pose_leg_head}` | 5 | `0.3268` | `-1.3719` | `0.9972` | `-36.4503` | 当前最像 head-anchored leg-biased contract subset |
| `weight:{direct_pose_head, direct_pose_arm_proj, direct_pose_else_proj, direct_pose_out_arm, direct_pose_out_else}` | 5 | `0.1265` | `-21.1833` | `-22.6848` | `0.9967` | 当前最像 head-anchored nonleg contract subset |
| `weight:{direct_pose_head, direct_pose_out_leg, direct_pose_out_arm, direct_pose_out_else, direct_pose_leg_head}` | 5 | `-0.2487` | `-13.7667` | `0.9972` | `-231.9378` | 只换 head+all readouts 仍失败；缺 arm/else adapters |
| `weight:{all_direct_modules_no_head}` | 6 | `0.0674` | `-38.0112` | `-28.7481` | `-174.8429` | 去掉 `head` 后整体 contract 仍进不了 high-closure regime |
| `weight:{all 7 direct modules}` | 7 | `0.9999` | `0.9976` | `0.9972` | `0.9967` | 当前唯一 weight-level `near-sufficient` set |

这里最重要的结构信号有两个：

- **activation 空间里，存在非常干净的 staged readout decomposition**
- **weight 空间里，high-closure regime 仍然必须依赖 `head-anchor + adapters + readouts` 的 joint contract**

#### E.10.2 leg-first best path

如果从 staged intervention 角度强行做 **leg-first**，当前最自然的路径是：

**Stage A1: leg-readout first**

- set: `activation:{direct_pose_out_leg, direct_pose_leg_head}`
- closure:
  - `out_direct = 0.2314`
  - `dir_base = 0.9366`
  - `dir_leg = 1.0000`
  - `dir_nonleg = 0.0000`

为什么这是当前最好的 leg-first stage1：

- 它用**最小的 2-module block**
- 把 dominant leg readout **完全关掉**
- 不误伤 nonleg
- 同时 `dir_base` closure 恰好停在 `0.9366`，与当前 leg 占比完全一致

因此它的解释很直接：

- **当前 step0 split 的“读出层面主像”就是 leg readout**
- 但这个结论只说明 dominant readout，不等于 leg block 是 earliest source

**Stage A2: 补 nonleg readouts**

- incremental modules: `direct_pose_out_arm`, `direct_pose_out_else`
- final set: `activation:{direct_pose_out_leg, direct_pose_leg_head, direct_pose_out_arm, direct_pose_out_else}`
- final closure:
  - `out_direct = 1.0000`
  - `dir_base = 1.0000`
  - `dir_leg = 1.0000`
  - `dir_nonleg = 1.0000`

所以：

- **在 semantic / readout decomposition 上，leg-first path 是成立的**
- 而且它是当前最自然、最干净的 staged path

但如果把同样问题搬到 weight-space contract sufficiency：

- 最好的 leg-biased head-anchored stage1 是  
  `weight:{direct_pose_head, direct_pose_arm_proj, direct_pose_out_leg, direct_pose_out_arm, direct_pose_leg_head}`
- 其 closure 是：
  - `out_direct = 0.3268`
  - `dir_base = -1.3719`
  - `dir_leg = 0.9972`
  - `dir_nonleg = -36.4503`

只有再补：

- `direct_pose_else_proj`
- `direct_pose_out_else`

才会进入真正的 high-closure regime（回到 7-module full direct set）。

因此 leg-first 的更精确判词是：

- **leg-first 在 readout/semantic 层面非常成立**
- **leg-first 在 weight/contract 层面不是 standalone sufficiency path**

#### E.10.3 nonleg-first best path

如果做 **nonleg-first**，当前最自然的 semantic path 是：

**Stage B1: nonleg-readout first**

- set: `activation:{direct_pose_out_arm, direct_pose_out_else}`
- closure:
  - `out_direct = 0.3602`
  - `dir_base = 0.0634`
  - `dir_leg = 0.0000`
  - `dir_nonleg = 1.0000`

它的解释也很干净：

- residual nonleg slice 可以被一个极小的 2-module readout block 完全解释
- 而 leg 完全不动

但它的问题也同样明显：

- **它只能解释 residual nonleg slice**
- 不能解释当前 main split 的主体，因为主体本来就不是 nonleg

**Stage B2: 补 leg block**

- incremental modules: `direct_pose_out_leg`, `direct_pose_leg_head`
- final set: `activation:{direct_pose_out_arm, direct_pose_out_else, direct_pose_out_leg, direct_pose_leg_head}`
- final closure:
  - `out_direct = 1.0000`
  - `dir_base = 1.0000`
  - `dir_leg = 1.0000`
  - `dir_nonleg = 1.0000`

所以 nonleg-first 的更准确读法是：

- **它是一个很干净的 residual-branch staged path**
- **但不是当前最自然的主叙事**

对应的 weight-space contract 版本：

- stage1:
  `weight:{direct_pose_head, direct_pose_arm_proj, direct_pose_else_proj, direct_pose_out_arm, direct_pose_out_else}`
- closure:
  - `out_direct = 0.1265`
  - `dir_base = -21.1833`
  - `dir_leg = -22.6848`
  - `dir_nonleg = 0.9967`

这里可以很清楚地看到：

- nonleg contract path 的确能在 weight-space 里把 nonleg 关到接近 1
- 但仍然**完全无法解释 leg-dominant main gap**

#### E.10.4 head-anchor analysis

把 `direct_pose_head` 单独拿出来看，当前最关键的是下面这五个对照：

| path | out_direct closure | dir_base closure | dir_leg closure | dir_nonleg closure | 解释 |
|---|---:|---:|---:|---:|---|
| `head only` | `-0.5143` | `-35.2178` | `-22.6848` | `-220.3522` | 单独换 head 会显著放大 gap |
| `head + leg path` | `-0.2370` | `-13.0321` | `0.9972` | `-220.3522` | leg 能关，但 nonleg 仍彻底错配 |
| `head + nonleg path` | `0.1265` | `-21.1833` | `-22.6848` | `0.9967` | nonleg 能关，但 leg 仍保留主 gap |
| `head + all readouts` | `-0.2487` | `-13.7667` | `0.9972` | `-231.9378` | 只补 readouts 仍不够，adapter mismatch 还在 |
| `head + all direct modules` | `0.9999` | `0.9976` | `0.9972` | `0.9967` | 第一个真正进入 high-closure regime 的集合 |

因此 `direct_pose_head` 的 staged graph 角色现在可以更明确地说成：

- **earliest source boundary**：最早的大 split 边界仍在这里
- **necessary anchor**：没有它，weight-space sufficiency path 进不了 high-closure regime
- **但不是 standalone sufficient explanation**

也就是说：

- `direct_pose_head` 不是“一个单独的 root-cause module”
- 更像 **joint contract 的必要锚点**

#### E.10.5 interaction / synergy judgement

这一轮最重要的 interaction 结论有三条：

**(1) `direct_pose_head` 与 downstream readouts 存在明显 synergy**

最简单的证据是：

- `weight:{direct_pose_head}` 的 `dir_base` closure = `-35.2178`
- `weight:{all_direct_modules_no_head}` 的 `dir_base` closure = `-38.0112`
- 但 `weight:{all 7}` 的 `dir_base` closure = `0.9976`

也就是：

- head-only 不行
- no-head all-direct 也不行
- **只有 head + downstream modules 一起换，才进入高 closure**

所以这不是 additive readout patch，而是明显的 **joint contract synergy**

**(2) `direct_pose_out_leg + direct_pose_leg_head` 构成了一个可解释的 leg-stage block**

在 activation 层面，它非常干净：

- `dir_leg closure = 1.0`
- `dir_nonleg closure = 0.0`
- `dir_base closure = 0.9366`

但一旦换到 weight-space：

- 即便加上 `head`
- 它也只能关掉 leg，无法让 nonleg / total 回到 baseline

所以它应被理解为：

- **dominant readout block**
- 不是 standalone contract-sufficient block

**(3) `direct_pose_out_arm + direct_pose_out_else` 是干净的 nonleg readout block；arm/else proj 则是 adapter/amplifier**

activation 层面：

- 只换 `out_arm + out_else` 就能把 `dir_nonleg` 关到 `1.0`

weight-space：

- `head + all readouts` 仍失败
- 必须把 `direct_pose_arm_proj + direct_pose_else_proj` 也一起补上，nonleg contract 才真正对齐

所以 arm/else proj 当前最像：

- **nonleg branch adapter**
- 同时也起到 **downstream amplifier** 作用

不是纯 late readout，也不是孤立 root cause。

#### E.10.6 staged final judgement

| hypothesis | support level | strongest evidence | weakest point | next best minimal intervention |
|---|---|---|---|---|
| 最自然的 staged decomposition 是 hybrid：readout-first + head-anchored contract closure | strongly supported | `activation:{out_leg+leg_head}` 可干净解释 leg-dominant split；weight 高 closure 只在 full 7-module set 出现 | 单一 swap grammar 无法同时表达 semantic decomposition 与 contract sufficiency | 先用 readout-level staged path 解释，再用 full 7-module weight set 验证 sufficiency |
| `direct_pose_head` 是 necessary anchor / earliest boundary，但不是 standalone sufficient explanation | strongly supported | `head only` 明显放大 gap；`all 7` 才进入 high-closure regime | activation 直接覆写 readouts 时可不碰 head 就关掉 step0 split | 把 head 当作 weight-space anchor，而不是单模块 patch |
| 存在强的 “leg-only” 版本，但只在 readout level 成立 | partially supported | `activation:{out_leg+leg_head}` 的 `dir_base closure = 0.9366`, `dir_leg closure = 1.0` | 它不能单独解释 earliest source / contract break | 把 leg block 作为 stage-1 semantic readout block |
| 存在强的 “nonleg-only earliest split” 版本 | not supported | nonleg-first 只能解释 residual nonleg slice | 当前 main split 不是 nonleg-dominant | 只把 nonleg-first 当 residual branch diagnosis |
| arm/else proj 更像 adapter / amplifier，而不是 pure readout | strongly supported | `head + all readouts` 仍失败，必须补 arm/else proj 才能进高 closure | 它们仍位于 earliest boundary 附近，不能被视为纯 late module | 在 contract intervention 中把它们视为 nonleg branch adapter |

#### E.10.7 one-paragraph judgement

- **当前最自然的 staged intervention 分解不是“只能整体换、完全不能分 stage”，也不是“纯 head-first”或“纯 nonleg-first”。更准确地说，它是一个 hybrid decomposition：如果目标是解释当前 step0 split 的语义读出结构，最自然的 first-stage 是 leg-readout first——`direct_pose_out_leg + direct_pose_leg_head` 这个 2-module block 就能把 `dir_leg` 完全关掉，并把 `dir_base` closure 精确推进到 `0.9366`，直接对应当前 leg-dominant raw gap；然后再补 `direct_pose_out_arm + direct_pose_out_else`，即可在 activation/readout 层面把 total gap 全关掉。但如果目标是做 weight-space sufficiency / causal contract closure，那么 `direct_pose_head` 仍然是必要锚点：head-only 会放大 gap，all-direct-no-head 也会失败，只有 `direct_pose_head + arm/else proj + leg/arm/else readouts + direct_pose_leg_head` 这套 full 7-module direct-branch contract 才真正进入 `>0.99` 的 high-closure regime。由此看，`direct_pose_head` 最像 earliest source boundary + necessary anchor，而不是 standalone sufficient explanation；leg modules 最像 dominant readout；arm/else proj 最像 nonleg branch adapter / amplifier；所以 staged causal chain 的最佳表述应是：earliest semantic split 出现在 `direct_pose_head` boundary，主 readout 落在 leg block，nonleg 路径主要承担 residual readout 与 adapter amplification，而真正的 sufficiency 仍需要 head-anchored whole direct-branch contract closure。**
