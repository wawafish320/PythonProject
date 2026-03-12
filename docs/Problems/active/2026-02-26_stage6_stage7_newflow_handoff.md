# 2026-02-26 Stage6→Stage7 新流程实验交接（handoff）

Last updated: 2026-02-27

## 1) 对“Stage6 早引入 leg/non-leg split”的判断

你的分析方向是对的：**尽早把 leg / non-leg 放进相对独立参数空间**，可以减少后续 7.1/7.2 leg-only 训练对 non-leg 的干扰。

但补充两点：
- 本轮链路里 Stage6 已经是 `direct_pose_split_enable=True`（有 split 结构），只是当时还没启用 70R 那套 non-leg 投影恢复设置。
- 目前主要破坏源仍是 **lambda final calibration**（分布迁移后的路由误学），所以 split 只能“降干扰”，不能单独解决 λ 回退。

---

## 2) 本轮目标与方向（供新对话直接续跑）

目标：验证“Base ckpt → Stage6 anchor → Stage7(70a/b/c→70R→71→72) → 单次 λ 校准”是否可作为新主流程，并定位回退点。

当前方向：
1. **先把 direct 分布稳定**（70R/71/72）
2. **再做 λ calibration**（只做一次，且在最终分布上）
3. 出现回退时先判别：是 direct 裸输出问题，还是 λ 混合路由问题

---

## 3) 已有产物总览（ckpt / log / 验证输出）

### 3.1 训练链路主产物

模型目录：`models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu`

- Stage6 anchor ckpt: `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage6_direct_cond_anchor_nohinge_pe32_h512_20260226_frombase.pth`
- Stage7.0a ckpt: `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70a_splitB2_pe32h512_20260226_frombase.pth`
- Stage7.0b ckpt: `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260226_frombase.pth`
- Stage7.0c ckpt: `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260226_frombase.pth`
- Stage7.0R ckpt: `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_frombase.pth`
- Stage7.1 ckpt: `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_71_legonly_after_nonlegproj256_20260226_frombase.pth`
- Stage7.2 ckpt: `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_72_legomega_after_nonlegproj256_20260226_frombase.pth`
- Stage6 armchain ckpt（当前推荐起点）: `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.pth`
- λ final ckpt（三种口径）:
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_lambda_final_calib_20260226_frombase.pth`
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_lambda_final_calib_20260226_frombase_fixphase.pth`
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_lambda_final_calib_20260226_frombase_fullcompat.pth`

对应日志（含完整 config）：同目录下 `posttrain_log_*.json`。

### 3.2 已完成的检查产物

- 7.1 freeze-guard 梯度检查日志：`models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage7_71_legonly_gradcheck_20260226.json`
- freerun（S6 baseline / 70c / 70R）：
  - `debug_output/verify_stage6_direct_cond_anchor_pe32_h512_20260223_nonlegcheck_v2/Walk_F_freerun_cycles.json`
  - `debug_output/verify_stage7_70c_pre70R_nonleg_check_v2/Walk_F_freerun_cycles.json`
  - `debug_output/verify_stage7_70R_nonleg_check_v2/Walk_F_freerun_cycles.json`
- pairwise 汇总：
  - `debug_output/verify_stage6_vs_70c_nonleg_check_20260226_v2/`
  - `debug_output/verify_stage6_vs_70R_nonleg_check_20260226_v2/`
  - `debug_output/verify_stage70c_vs_70R_nonleg_check_20260226_v2/`

---

## 4) 每一步关键配置（精简版）

以下字段来自对应 `posttrain_log_*.json` 中 `config`：

- Stage6 (`WalkF_stage6_direct_cond_anchor_nohinge_pe32_h512_20260226_frombase`)
  - `direct_pose_reinit=True`
  - `direct_pose_feat_source=cond`
  - `direct_pose_split_enable=True`
  - `direct_pose_loss_leg_split=False`

- Stage6 split-first 2way（当前 Stage6 对照起点）
  - `run_name=WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226`
  - `direct_pose_split_enable=True`
  - `direct_pose_leg_train_only=False`
  - `direct_pose_nonleg_train_only=False`
  - `direct_pose_loss_leg_split=True`

- Stage6 split-first 3way armchain（本轮新增起点）
  - `run_name=WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227`
  - `direct_pose_split_enable=True`
  - `direct_pose_arm_split_enable=True`
  - `direct_pose_arm_bones=clavicle_l,upperarm_l,RUpArmTwist_l_01,RUpArmTwist_l_02,lowerarm_l,L_ForeTwist_01,L_ForeTwist_02,hand_l,index_01_l,middle_01_l,ring_01_l,pinky_01_l,thumb_01_l,clavicle_r,upperarm_r,RUpArmTwist_r_01,RUpArmTwist_r_02,lowerarm_r,R_ForeTwist_01,R_ForeTwist_02,hand_r,index_01_r,middle_01_r,ring_01_r,pinky_01_r,thumb_01_r`
  - `direct_pose_leg_train_only=False`
  - `direct_pose_nonleg_train_only=False`
  - 说明：Stage6 仍是全量训练口径（不是 leg-only / non-leg-only）。

- 70a (`WalkF_stage7_70a_splitB2_pe32h512_20260226_frombase`)
  - `direct_pose_split_enable=True`
  - `direct_pose_loss_leg_split=True`
  - `direct_pose_use_phase_z=False`

- 70b (`WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260226_frombase`)
  - 继承 70a
  - `direct_pose_use_phase_z=True`
  - `direct_pose_phase_z_mode=concat`

- 70c (`WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260226_frombase`)
  - `epochs=0`（配置切换型阶段）
  - `direct_pose_phase_z_mode=replace_contacts`

- 70R (`WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_frombase`)
  - `direct_pose_nonleg_proj_dim=256`
  - `direct_pose_nonleg_train_only=True`
  - `direct_pose_leg_enable=True`
  - `direct_pose_leg_mode=so3`
  - `direct_pose_grad_monitor_enable=True`

- 71 (`WalkF_stage7_71_legonly_after_nonlegproj256_20260226_frombase`)
  - `direct_pose_leg_train_only=True`
  - `direct_pose_nonleg_train_only=False`

- 72 (`WalkF_stage7_72_legomega_after_nonlegproj256_20260226_frombase`)
  - `direct_pose_leg_train_only=True`
  - 其余基本继承 71

- λ final (`WalkF_stage7_lambda_final_calib_*`)
  - `train_lambda_head=True`
  - `train_direct_pose=False`
  - `time_index_mode=cycle`
  - `rollout_cycles=2`

- Stage7 from armchain（下一轮建议口径）
  - 70a/70b/70c/70R/71/72 全链路保持：
    - `direct_pose_arm_split_enable=True`
    - `direct_pose_arm_bones=<与 Stage6 armchain 完全一致>`
  - 71/72 继续使用 leg-only 更新：
    - `direct_pose_leg_train_only=True`
    - `direct_pose_nonleg_train_only=False`
  - 目的：在 armchain 起点上验证 leg8 是否被 7.1/7.2 拉回，同时保持 non-leg 收益。

---

## 5) 指令清单（可直接复现/续跑）

## 5.1 从日志回放任一 posttrain 步骤（推荐）

```bash
# 用法: replay_posttrain <posttrain_log_xxx.json>
replay_posttrain () {
  local LOG_JSON="$1"
  local TMP_JSON="/tmp/$(basename "${LOG_JSON%.json}")__config.json"
  python - <<PY
import json
src = r"$LOG_JSON"
dst = r"$TMP_JSON"
obj = json.load(open(src, "r"))
json.dump(obj["config"], open(dst, "w"), indent=2)
print(dst)
PY
  PYTHONPATH=. python -m train.posttrain --config "$TMP_JSON"
}
```

示例：

```bash
replay_posttrain models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_frombase.json
replay_posttrain models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage7_71_legonly_after_nonlegproj256_20260226_frombase.json
```

## 5.2 Check-1：7.1 freeze guard 梯度泄漏检查（已验证通过）

```bash
python - <<'PY'
import json
p='models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage7_71_legonly_gradcheck_20260226.json'
log=json.load(open(p))['log']
print('n=',len(log))
print('step1:', log[0]['direct_grad_norm_out_nonleg'], log[0]['direct_grad_norm_trunk'], log[0]['direct_grad_norm_leg_head'])
print('last :', log[-1]['direct_grad_norm_out_nonleg'], log[-1]['direct_grad_norm_trunk'], log[-1]['direct_grad_norm_leg_head'])
PY
```

期望：`direct_grad_norm_out_nonleg=NaN` 且 `direct_grad_norm_trunk=NaN`，`direct_grad_norm_leg_head` 有效非零。

## 5.3 Check-2：S6 vs 70c vs 70R freerun 对比

```bash
# Stage6 baseline
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_pe32_h512_20260223.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage6_direct_cond_anchor_pe32_h512_20260223_nonlegcheck_v2 \
  --rounds 5 --depth 3 \
  --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none \
  --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series \
  --force

# Stage7.0c
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260226_frombase.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage7_70c_pre70R_nonleg_check_v2 \
  --rounds 5 --depth 3 \
  --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none \
  --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series \
  --force

# Stage7.0R
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_frombase.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage7_70R_nonleg_check_v2 \
  --rounds 5 --depth 3 \
  --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none \
  --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series \
  --force
```

## 5.4 对比报告构建

```bash
python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_direct_cond_anchor_pe32_h512_20260223_nonlegcheck_v2/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_70c_pre70R_nonleg_check_v2/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6_vs_70c_nonleg_check_20260226_v2

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_direct_cond_anchor_pe32_h512_20260223_nonlegcheck_v2/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_70R_nonleg_check_v2/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6_vs_70R_nonleg_check_20260226_v2

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage7_70c_pre70R_nonleg_check_v2/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_70R_nonleg_check_v2/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage70c_vs_70R_nonleg_check_20260226_v2
```

---

## 6) 关键结论（截至当前）

- Freeze guard（7.1）通过：non-leg/trunk 无梯度更新。
- Check-2（S6/70c/70R）显示：
  - `non-leg mean`: `S6 0.3009 -> 70c 0.2389 -> 70R 0.1538`
  - `70c -> 70R` 进一步改善 non-leg（`-0.0852`）
- 在这组 ckpt 上，没观察到“70R non-leg 恢复不足”。
- 现阶段最大风险点仍是 **λ final calibration 回退**，需要在“最终 direct 分布”上做更稳健的 λ 校准策略。

> 注意：本次 S6 baseline 来自 `models/MLPL2_DirectBranch_v1/ckpt_last_WalkF_stage6_direct_cond_anchor_pe32_h512_20260223.pth`，与 frombase 链路并非同 lineage，绝对值对比需谨慎，趋势对比可用。

---

## 7) 下一步建议（按最新决策：先跑通，再看热点）

1. **先做 Stage6 split-first 版本**：在 Stage6 就固定 leg/non-leg 拆分训练口径，再从这个新 Stage6 起点串跑完整 Stage7 与 λ。  
2. **先看链路是否跑通**：`Stage6(split-first) -> 70a -> 70b -> 70c -> 70R -> 71 -> 72 -> lambda_final`，先不提前放大 calf_r@SIC2-4。  
3. **跑通后再做热点定位**：再回到 no-lambda / SIC 分桶 / calf_r 局部分析，避免在不稳定起点上反复调参。  

### 7.1 Stage6 split-first（建议配置）

基于 `posttrain_log_WalkF_stage6_direct_cond_anchor_nohinge_pe32_h512_20260226_frombase.json` 生成一个新配置并运行：

```bash
python - <<'PY'
import json
src='models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage6_direct_cond_anchor_nohinge_pe32_h512_20260226_frombase.json'
dst='/tmp/stage6_splitfirst_20260226.json'
cfg=json.load(open(src))['config']
cfg['run_name']='WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226'
cfg['direct_pose_split_enable']=True
cfg['direct_pose_loss_leg_split']=True
cfg['direct_pose_nonleg_proj_dim']=256
cfg['direct_pose_leg_enable']=True
cfg['direct_pose_leg_mode']='so3'
cfg['direct_pose_leg_train_only']=False
cfg['direct_pose_nonleg_train_only']=False
json.dump(cfg, open(dst,'w'), indent=2)
print(dst)
PY
PYTHONPATH=. python -m train.posttrain --config /tmp/stage6_splitfirst_20260226.json
```

产物目标：
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226.pth`
- 对应 `posttrain_log_WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226.json`

### 7.2 Stage6 split-first（已执行，2026-02-26）

已完成实跑（基于 7.1 配置），并额外打开了两项早期稳定性观察开关：
- `direct_pose_grad_monitor_enable=True`
- `direct_pose_loss_group_norm_enable=True`（`w_leg=w_nonleg=1.0`）

生成产物：
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226.pth`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226.json`

关键信号（来自 log）：
- 训练完成：`epochs=5, steps_per_epoch=60`，无 skip（`ok_steps=60` 每个 epoch）。
- 早期（step0）梯度：`g(trunk/leg/non)=9.126e-01 / 9.103e-01 / 5.810e-01`，leg 与 non-leg 均为有效非零。
- 前 20 step 梯度比（`nonleg/leg`）中位数约 `0.49`（范围约 `0.32~0.71`），未见“某分支近零停滞”。
- 300 step 内 `direct_grad_ratio_alert` 触发 `39/300`（提示有不均衡窗口，但非单侧失活）。

结论（针对“共享权重复制后早期分支过近”的担忧）：
- 担忧成立且值得监控；本次配置下两个分支都拿到了持续监督信号。
- 建议保留以上两项开关作为 split-first 默认 guard，再进入 70a/70b/70c/70R/71/72/λ 串跑。

---

## 8) 2026-02-27 执行记录：split-first 起点接入 Stage7 全链路（已完成）

### 8.1 固化配置文件（可直接复现）

本轮实际使用配置已落盘到 `config/`：

- `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226.json`
- `config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260226_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260226_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260226_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260226_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260226_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_lambda_final_calib_20260226_fromsplitfirst_fullcompat.json`

关键说明（与 frombase 原链路差异）：
- 70a/70b/70c 也保持 `direct_pose_nonleg_proj_dim=256 + direct_pose_leg_enable=True + direct_pose_leg_mode=so3`，保证从 split-first Stage6 结构连续接入。
- 后续 70R/71/72/λ 与 split-first 链路一致延续。

### 8.2 训练指令（按顺序）

```bash
# Stage6 split-first（若已完成可跳过）
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226.json

# Stage7 chain from split-first
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260226_fromsplitfirst.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260226_fromsplitfirst.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260226_fromsplitfirst.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_fromsplitfirst.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260226_fromsplitfirst.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260226_fromsplitfirst.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_lambda_final_calib_20260226_fromsplitfirst_fullcompat.json
```

产物（主路径）：
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_fromsplitfirst.pth`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_72_legomega_after_nonlegproj256_20260226_fromsplitfirst.pth`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_lambda_final_calib_20260226_fromsplitfirst_fullcompat.pth`

对应日志：
- 同目录下 `posttrain_log_*_fromsplitfirst*.json`

### 8.3 验证与对比指令

```bash
# freerun 导出（split-first链路关键节点）
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_fromsplitfirst.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage7_70R_fromsplitfirst_20260227 \
  --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force

PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_72_legomega_after_nonlegproj256_20260226_fromsplitfirst.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage7_72_fromsplitfirst_20260227 \
  --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force

PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_lambda_final_calib_20260226_fromsplitfirst_fullcompat.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage7_lambda_final_fromsplitfirst_20260227 \
  --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force

# 对照：旧主链 lambda_final(frombase_fullcompat) freerun
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_lambda_final_calib_20260226_frombase_fullcompat.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage7_lambda_final_frombase_fullcompat_20260227 \
  --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force
```

```bash
# split-first Stage6 vs Stage7(70R/72/lambda) 关键对比
python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_splitfirst_frombase_20260226/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_70R_fromsplitfirst_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split_vs_stage70R_fromsplitfirst_20260227

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_splitfirst_frombase_20260226/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_72_fromsplitfirst_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split_vs_stage72_fromsplitfirst_20260227

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_splitfirst_frombase_20260226/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_lambda_final_fromsplitfirst_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split_vs_stage7lambda_fromsplitfirst_20260227

# 旧主链 lambda_final（frombase_fullcompat）对照
python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage7_lambda_final_frombase_fullcompat_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_lambda_final_fromsplitfirst_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage7lambda_frombase_vs_fromsplitfirst_20260227
```

### 8.4 结果摘要（本轮）

1) `Stage6(split-first) -> 70R`（`debug_output/verify_stage6split_vs_stage70R_fromsplitfirst_20260227/gate_metrics.json`）  
- `global_mean`: `0.3452 -> 0.3025`（`-12.38%`）  
- `leg8_mean_delta=+0.2177`（腿部短期退化）  
- `non_leg_mean_delta=-0.0990`（non-leg 恢复）  
- `calf_r@SIC2-4`: `0.8291 -> 2.2745`（恶化）  
- `calf_r@SIC35-42`: `1.9865 -> 0.8384`（改善）  
- `calf_r@SIC53-63`: `1.8088 -> 0.7162`（改善）  

2) `Stage6(split-first) -> 72`（`debug_output/verify_stage6split_vs_stage72_fromsplitfirst_20260227/gate_metrics.json`）  
- `global_mean`: `0.3452 -> 0.1937`（`-43.90%`）  
- `leg8_mean_delta=-0.3944`，`non_leg_mean_delta=-0.0990`（双侧改善）  
- `calf_r@SIC2-4`: `0.8291 -> 0.3592`（改善）  
- `calf_r@SIC35-42`: `1.9865 -> 0.2057`（改善）  
- `calf_r@SIC53-63`: `1.8088 -> 0.2642`（改善）  

3) `72 -> lambda_final`  
- Direct 指标基本不变（`debug_output/verify_stage6split_vs_stage7lambda_fromsplitfirst_20260227/gate_metrics.json` 与 72 一致）  
- 融合侧恢复到正常：`lambda_mean≈0.974`，见 `debug_output/verify_stage7_splitfirst_chain_20260227/fused_metrics_summary.txt`  

---

## 9) 用户指定 old_json 对比（用于退化分布判断）

对比命令：

```bash
python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/_stage7_recheck_20260224_preleg/compare_vs_initial/freerun_old.json/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_lambda_final_fromsplitfirst_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_oldrecheck_vs_stage7lambda_fromsplitfirst_20260227
```

主结果（同你给的格式；old=指定 old_json，new=本轮最终 lambda）：

```text
[overall]
mean_old=0.176346
mean_new=0.193675
mean_delta=+0.017329
bones_excl_root=45
bones_regress_by_mean=25
bones_improve_by_mean=20

[region_split]
leg8_mean_old=0.399370
leg8_mean_new=0.403514
leg8_mean_delta=+0.004144
non_leg_mean_old=0.128125
non_leg_mean_new=0.148305
non_leg_mean_delta=+0.020180

[pointwise_signal]
points=15480
improved_ratio=0.475194
worse_ratio=0.524806
median_delta=+0.002258
```

附加热点（`debug_output/verify_oldrecheck_vs_stage7lambda_fromsplitfirst_20260227/summary_metrics.txt`）：
- `calf_r global`: `0.3478 -> 0.2780`（改善）
- `calf_r@SIC2-4`: `0.1574 -> 0.3592`（恶化）
- `calf_r@SIC35-42`: `0.2929 -> 0.2057`（改善）
- `calf_r@SIC53-63`: `0.3927 -> 0.2642`（改善）

---

## 10) 2026-02-27 追加记录：扩容/增轮次 + Probe + 监督信号迭代

### 10.1 扩容与增轮次（70R/71/72/λ，fromsplitfirst）

#### 10.1.1 固化配置

- proj256 + ep5：
  - `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_ep5_20260227_fromsplitfirst.json`
  - `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_ep5_20260227_fromsplitfirst.json`
  - `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_ep5_20260227_fromsplitfirst.json`
  - `config/posttrain_WalkF_stage7_lambda_final_calib_ep5_20260227_fromsplitfirst_fullcompat.json`

- proj512 + ep5：
  - `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj512_preleg_ep5_20260227_fromsplitfirst.json`
  - `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj512_ep5_20260227_fromsplitfirst.json`
  - `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj512_ep5_20260227_fromsplitfirst.json`
  - `config/posttrain_WalkF_stage7_lambda_final_calib_proj512_ep5_20260227_fromsplitfirst_fullcompat.json`

#### 10.1.2 结果（核心对比）

1) 增轮次（proj256: ep3 -> ep5）
- 对比文件：`debug_output/verify_stage7lambda_fromsplitfirst_ep3_vs_ep5_20260227/global_signal_summary.txt`
- 关键值：
  - `overall mean: 0.193675 -> 0.193306 (delta=-0.000370)`
  - `non_leg_mean: 0.148305 -> 0.147855 (delta=-0.000450)`
- 结论：增轮次收益很小，但方向略好（并未恶化）。

2) 扩容（proj256 ep5 -> proj512 ep5）
- 对比文件：`debug_output/verify_stage7lambda_proj256ep5_vs_proj512ep5_fromsplitfirst_20260227/global_signal_summary.txt`
- 关键值：
  - `overall mean: 0.193306 -> 0.203013 (delta=+0.009708)`
  - `leg8_mean: 0.403514 -> 0.425202 (delta=+0.021688)`
  - `non_leg_mean: 0.147855 -> 0.154973 (delta=+0.007117)`
- 结论：proj512 在当前训练预算下明显变差，不是“容量不够”的证据。

3) 相对 old_json 的主对比（proj256 ep5 vs proj512 ep5）
- `debug_output/verify_oldrecheck_vs_stage7lambda_ep5_fromsplitfirst_20260227/global_signal_summary.txt`
  - `mean_delta=+0.016960`
- `debug_output/verify_oldrecheck_vs_stage7lambda_proj512_ep5_fromsplitfirst_20260227/global_signal_summary.txt`
  - `mean_delta=+0.026667`
- 结论：proj512 相比 proj256 ep5 进一步远离 old 基线。

---

### 10.2 Experiment 3：non-leg feature probe（先验验证）

#### 10.2.1 产物

- freerun probe 导出：
  - `debug_output/verify_stage7_70R_nonlegprobe_proj256_ep5_20260227/Walk_F_freerun_cycles.json`
  - `debug_output/verify_stage7_70R_nonlegprobe_proj512_ep5_20260227/Walk_F_freerun_cycles.json`

- probe 分析工具：
  - `tools/diag_direct_nonleg_feature_probe.py`

- 汇总：
  - `debug_output/verify_stage7_70R_nonlegprobe_compare_proj256_vs_proj512_20260227/summary.md`

#### 10.2.2 结论

probe 结果显示：
- `pre_proj_in`（shared trunk）上线几乎相同：
  - linear `r2_probe=0.773100`（256/512 一样）
  - mlp `r2_probe=0.997785`（256/512 一样）
- `out_in` 上：
  - linear: `proj256 0.900497 > proj512 0.796927`
  - mlp: `proj512 0.998144 > proj256 0.975415`

解释：信息在 feature 里（不是“信息缺失”），问题更像 extraction/监督信号/优化路径，而非单纯容量。

---

### 10.3 Experiment 1：监督信号迭代（non-leg 目标骨骼 focus）

#### 10.3.1 代码改动（新增监督开关）

文件：`train/posttrain.py`

新增配置项：
- `direct_pose_nonleg_focus_bones`
- `direct_pose_nonleg_focus_weight`

并新增日志统计：
- `dir_nonleg_plain`
- `dir_nonleg_focus_requested`
- `dir_nonleg_focus_resolved`
- `dir_nonleg_focus_applied`

同时修复了 focus bone 映射来源：
- 在 trainer 初始化时把 `ds.bone_names` 注入 `loss_fn`（`set_bone_names`）
- 在 focus 解析时 fallback 到 `loss_fn.meta.skeleton.bone_names`

> 中间轮次说明：
> - `supfocusULw4` / `supfocusULw4v2` 首次跑时 `dir_nonleg_focus_resolved=0`（配置生效但映射未命中）
> - 修复后 `supfocusULw4v3` 达到 `resolved=8, applied=1.0`，并以 v3 作为最终结果

#### 10.3.2 最终配置（v3）

- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_supfocusULw4v3_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_supfocusULw4v3_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_supfocusULw4v3_ep5_20260227_fromsplitfirst.json`
- `config/posttrain_WalkF_stage7_lambda_final_calib_supfocusULw4v3_ep5_20260227_fromsplitfirst_fullcompat.json`

focus bones：
- `upperarm_l,lowerarm_l,hand_l,pinky_01_l,upperarm_r,lowerarm_r,hand_r,pinky_01_r`
- `direct_pose_nonleg_focus_weight=4.0`

#### 10.3.3 训练/验证产物（v3）

- ckpt/log（models）：
  - `ckpt_last_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_supfocusULw4v3_ep5_20260227_fromsplitfirst.pth`
  - `posttrain_log_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_supfocusULw4v3_ep5_20260227_fromsplitfirst.json`
  - `ckpt_last_WalkF_stage7_72_legomega_after_nonlegproj256_supfocusULw4v3_ep5_20260227_fromsplitfirst.pth`
  - `ckpt_last_WalkF_stage7_lambda_final_calib_supfocusULw4v3_ep5_20260227_fromsplitfirst_fullcompat.pth`

- freerun 导出：
  - `debug_output/verify_stage7_70R_supfocusULw4v3_ep5_fromsplitfirst_20260227/Walk_F_freerun_cycles.json`
  - `debug_output/verify_stage7_72_supfocusULw4v3_ep5_fromsplitfirst_20260227/Walk_F_freerun_cycles.json`
  - `debug_output/verify_stage7_lambda_final_supfocusULw4v3_ep5_fromsplitfirst_20260227/Walk_F_freerun_cycles.json`

- 对比报告：
  - `debug_output/verify_stage6split_vs_stage70R_supfocusULw4v3_ep5_fromsplitfirst_20260227/`
  - `debug_output/verify_stage6split_vs_stage72_supfocusULw4v3_ep5_fromsplitfirst_20260227/`
  - `debug_output/verify_stage6split_vs_stage7lambda_supfocusULw4v3_ep5_fromsplitfirst_20260227/`
  - `debug_output/verify_oldrecheck_vs_stage7lambda_supfocusULw4v3_ep5_fromsplitfirst_20260227/`
  - `debug_output/verify_stage7lambda_ep5_baseline_vs_supfocusULw4v3_fromsplitfirst_20260227/`

#### 10.3.4 结果摘要（v3）

1) 70R 训练段信号（对应“是否仍在学”）
- log：`posttrain_log_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_supfocusULw4v3_ep5_20260227_fromsplitfirst.json`
- `dir_nonleg_focus_resolved=8`，`dir_nonleg_focus_applied=1.0`
- epoch3 仍有学习：
  - `total: 1.968978 -> 1.978128`（近平台）
  - `dir_nonleg_base: 0.004088 -> 0.004415`
  - `dir_nonleg_plain: 0.002901 -> 0.003009`
- non-leg head 梯度仍高位：`direct_grad_norm_out_nonleg mean≈12.39, p50≈13.63`

2) split-first Stage6 -> 本轮 λ 最终（v3）
- 文件：`debug_output/verify_stage6split_vs_stage7lambda_supfocusULw4v3_ep5_fromsplitfirst_20260227/gate_metrics.json`
- 关键值：
  - `global_mean: 0.345213 -> 0.187010`（`-45.83%`）
  - `leg8_mean_delta=-0.394362`
  - `non_leg_mean_delta=-0.107141`
  - `calf_r@SIC2-4: 0.829061 -> 0.359193`（改善）

3) baseline(ep5) -> supfocus(v3)
- 文件：`debug_output/verify_stage7lambda_ep5_baseline_vs_supfocusULw4v3_fromsplitfirst_20260227/global_signal_summary.txt`
- 关键值：
  - `overall mean: 0.193306 -> 0.187010 (delta=-0.006295)`
  - `non_leg_mean: 0.147855 -> 0.140199 (delta=-0.007656)`
- λ稳定性：
  - baseline `LambdaMean≈0.974473`
  - supfocus v3 `LambdaMean≈0.973970`

4) 你指定 old_json 的主结果（old=指定 old_json，new=本轮最终 lambda）
- 来源：`debug_output/verify_oldrecheck_vs_stage7lambda_supfocusULw4v3_ep5_fromsplitfirst_20260227/global_signal_summary.txt`

```text
[overall]
mean_old=0.176346
mean_new=0.187010
mean_delta=+0.010664
bones_excl_root=45
bones_regress_by_mean=32
bones_improve_by_mean=13

[region_split]
leg8_mean_old=0.399370
leg8_mean_new=0.403514
leg8_mean_delta=+0.004144
non_leg_mean_old=0.128125
non_leg_mean_new=0.140199
non_leg_mean_delta=+0.012074

[pointwise_signal]
points=15480
improved_ratio=0.466021
worse_ratio=0.533979
median_delta=+0.003003
```

---

### 10.4 当前结论（更新，含因果链）

1. **扩容结论**：`proj512` 在当前训练预算下比 `proj256` 更差（含 overall/leg/non-leg），不支持“先扩容即可恢复”。
2. **增轮次结论**：`ep3 -> ep5` 对 `proj256` 仅小幅改善，说明“单纯加步数”不是主瓶颈。
3. **显式因果链**：  
   `Probe(信息在feature)` + `扩容/增步无明显收益` + `同容量下改监督可变好` ⇒ 当前瓶颈更接近**监督目标/梯度分配**，而不是 feature 信息缺失或纯容量不足。
4. **监督信号迭代结论**：non-leg 目标骨骼 focus（v3）在不改容量的前提下，能把 `lambda_final(ep5)` 从 `0.193306` 拉到 `0.187010`，验证监督信号方向有效。
5. **thumb_01_l / hand_r 副作用性质（当前判断）**：  
   - 观测：baseline(ep5) -> supfocus(v3) 时，`thumb_01_l: 0.258710 -> 0.426931 (+0.168220)`，`hand_r: 0.335775 -> 0.428421 (+0.092646)`；同时 `upperarm_l/lowerarm_l/pinky_01_l/lowerarm_r` 等显著改善。  
   - 含义：更像 **shared non-leg 参数空间内的梯度重分配/组内竞争**（tuning-sensitive），不是“上肢信息不存在”。  
   - 但是否为**结构性上限**（需 arm split 才能彻底消除冲突）尚未被证伪，需判别实验确认。
6. **仍未完全回到 old_json**：`mean_delta` 仍为 `+0.010664`，说明当前方案虽优于 baseline(ep5)，但离目标分布仍有差距。

### 10.5 结构性 vs 调参问题：判别实验标准（下一步）

1) **先做调参判别（低成本）**  
- 固定 v3 其余配置，仅扫：  
  - `direct_pose_nonleg_focus_weight ∈ {1.5, 2.0, 3.0, 4.0}`  
  - focus 集合：`(当前8骨)` vs `(当前8骨 + thumb_01_l + hand_r)`  
- 判据：若能同时保持上肢主目标改善且显著回收 `thumb_01_l/hand_r`，则归因为**调参问题**。

2) **再做结构判别（必要时）**  
- 若上述 sweep 全部落在“修一处坏一处”Pareto 前沿（无法同时兼顾），则支持**结构性耦合**。  
- 进入架构改动：在 non-leg 内做 arm/finger 拆分（例如 arm split head），把冲突从同一 readout 参数空间中解耦。

---

## 11) 2026-02-27 追加记录：Stage6 三路拆分（armchain）验证

### 11.1 目标与重定位

- 背景：`supfocus v3` 已验证监督信号方向正确，但 `thumb/hand` 副作用更像 shared 参数空间竞争，不是单纯调 focus weight 可解。
- 本轮重定位：左右异常更可能来自 arm/else 边界切割（twist 链条被放在 else，和 upper/lowerarm 强耦合被切断）。
- 验证策略：在 Stage6 直接做三路 split，比较 `2way` vs `3way_arm` vs `3way_armchain`。

### 11.2 本轮配置（关键字段）

- 2way Stage6 baseline（对照）  
  `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226.json`
- 3way arm（旧边界）  
  `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_arm_pe32h512_20260227.json`
- 3way armchain（新边界）  
  `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`

三者共同训练口径（已在 config 与 posttrain_log 双侧确认）：
- `direct_pose_split_enable=true`
- `direct_pose_leg_train_only=false`
- `direct_pose_nonleg_train_only=false`

3way 特有：
- `direct_pose_arm_split_enable=true`
- `direct_pose_arm_bones`：
  - `3way_arm`：仅 upper/lowerarm + hand + thumb/pinky（简版）
  - `3way_armchain`：clavicle + upperarm + up-arm twist + lowerarm + foretwist + hand + 五指基骨（左右对称链）

### 11.3 训练指令（Stage6）

```bash
# 3way_arm
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_arm_pe32h512_20260227.json

# 3way_armchain
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json
```

### 11.4 验证/对比指令

```bash
# A) freerun 导出（2way / 3way_arm / 3way_armchain）
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage6_direct_cond_anchor_splitfirst_pe32h512_20260226.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage6_split2way_recheck_20260227 \
  --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force

PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage6_direct_cond_anchor_splitfirst_3way_arm_pe32h512_20260227.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage6_split3way_arm_20260227 \
  --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force

PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage6_split3way_armchain_20260227 \
  --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force

# B) old/new 汇总报告
python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage6_split3way_arm_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6_split2way_vs_split3way_arm_20260227

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage6_split3way_armchain_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6_split2way_vs_split3way_armchain_20260227

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split3way_arm_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage6_split3way_armchain_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6_split3way_arm_vs_armchain_20260227
```

### 11.5 本轮产物清单

- 配置：
  - `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_arm_pe32h512_20260227.json`
  - `config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- 训练 ckpt/log：
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage6_direct_cond_anchor_splitfirst_3way_arm_pe32h512_20260227.pth`
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage6_direct_cond_anchor_splitfirst_3way_arm_pe32h512_20260227.json`
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.pth`
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/posttrain_log_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json`
- freerun：
  - `debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json`
  - `debug_output/verify_stage6_split3way_arm_20260227/Walk_F_freerun_cycles.json`
  - `debug_output/verify_stage6_split3way_armchain_20260227/Walk_F_freerun_cycles.json`
- 对比与审计：
  - `debug_output/verify_stage6_split2way_vs_split3way_arm_20260227/global_signal_summary.txt`
  - `debug_output/verify_stage6_split2way_vs_split3way_armchain_20260227/global_signal_summary.txt`
  - `debug_output/verify_stage6_split3way_arm_vs_armchain_20260227/global_signal_summary.txt`
  - `debug_output/verify_stage6_split3way_arm_vs_armchain_20260227/branch_mapping_check_armchain.txt`
  - `debug_output/verify_stage6_split3way_arm_vs_armchain_20260227/train_grad_compare.txt`

### 11.6 结果摘要（核心结论）

1) `2way -> 3way_arm`（旧边界）  
- `overall mean: 0.345213 -> 0.382701 (delta=+0.037489)`  
- `leg8_mean_delta=+0.082533`，`non_leg_mean_delta=+0.027749`  
- 左右异常明显：`lowerarm_l -0.524820`，`lowerarm_r +0.331387`；`calf_l -0.261523`，`calf_r +0.455661`

2) `2way -> 3way_armchain`（新边界）  
- `overall mean: 0.345213 -> 0.315243 (delta=-0.029970)`  
- `leg8_mean_delta=+0.117164`（腿部仍有退化）  
- `non_leg_mean_delta=-0.061783`（non-leg 明显改善）  
- `bones_regress_by_mean=14`，`bones_improve_by_mean=31`

3) `3way_arm -> 3way_armchain`（仅看边界修复收益）  
- `overall mean_delta=-0.067459`  
- `non_leg_mean_delta=-0.089532`  
- 说明 chain-consistent arm grouping 对 non-leg 改善显著。

4) 梯度均衡（Stage6）  
- `2way`: `ratio20_p50=0.4863`, `grad_ratio_alert=39/300`  
- `3way_arm`: `ratio20_p50=0.9159`, `grad_ratio_alert=0/300`  
- `3way_armchain`: `ratio20_p50=0.9821`, `grad_ratio_alert=0/300`

5) 分组/迁移映射审计  
- `old_nonleg_idx_equals_new_nonleg_idx=True`  
- `arm_copy_mapping_mismatch_count=0`  
- 结论：未发现左右 index 映射 bug，主要问题在 arm/else 边界定义。

### 11.7 当前判断（含 leg 退步性质）

- 结论：twist 链条边界修复（armchain）方向成立，且是本轮 Stage6 起点效果提升的关键因素。
- leg 退步性质判断：当前更像 **Stage6 全量训练下的共享 trunk 梯度重分配副作用**，不是映射 bug。依据：
  - Stage6 配置是全量训练：`direct_pose_leg_train_only=false` 且 `direct_pose_nonleg_train_only=false`。
  - 3way_armchain 中 arm 侧梯度占比明显提升（`g_out_arm_p50_first20=1.0784` vs `g_out_else_p50_first20=0.1407`），说明 trunk 梯度分配发生了结构性变化。
  - 同时映射审计通过（`arm_copy_mapping_mismatch_count=0`）。
- 为什么不在 Stage6 加 leg 保护：Stage6 目标是建立更干净的分支初始分工；leg 的最终回收本来由 7.1/7.2 的 leg-only 阶段负责，过早在 Stage6 做 leg 保护会削弱 armchain 对 non-leg 的起点收益。

### 11.8 下一步执行（具体操作 + 验收标准）

#### 11.8.1 先生成 Stage7 from-armchain 配置（70a→72→λ）

```bash
python - <<'PY'
import json
from pathlib import Path

ARM_BONES = "clavicle_l,upperarm_l,RUpArmTwist_l_01,RUpArmTwist_l_02,lowerarm_l,L_ForeTwist_01,L_ForeTwist_02,hand_l,index_01_l,middle_01_l,ring_01_l,pinky_01_l,thumb_01_l,clavicle_r,upperarm_r,RUpArmTwist_r_01,RUpArmTwist_r_02,lowerarm_r,R_ForeTwist_01,R_ForeTwist_02,hand_r,index_01_r,middle_01_r,ring_01_r,pinky_01_r,thumb_01_r"
MODEL_DIR = "models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu"

chain = [
    ("posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260226_fromsplitfirst.json",
     "posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json",
     "ckpt_last_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.pth"),
    ("posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260226_fromsplitfirst.json",
     "posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json",
     "ckpt_last_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.pth"),
    ("posttrain_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260226_fromsplitfirst.json",
     "posttrain_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260227_fromarmchain.json",
     "ckpt_last_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.pth"),
    ("posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260226_fromsplitfirst.json",
     "posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json",
     "ckpt_last_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260227_fromarmchain.pth"),
    ("posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260226_fromsplitfirst.json",
     "posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json",
     "ckpt_last_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.pth"),
    ("posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260226_fromsplitfirst.json",
     "posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json",
     "ckpt_last_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.pth"),
    ("posttrain_WalkF_stage7_lambda_final_calib_20260226_fromsplitfirst_fullcompat.json",
     "posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json",
     "ckpt_last_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.pth"),
]

for src_name, dst_name, ckpt_name in chain:
    src = Path("config") / src_name
    dst = Path("config") / dst_name
    cfg = json.load(open(src, "r"))
    cfg["run_name"] = cfg["run_name"].replace("20260226_fromsplitfirst", "20260227_fromarmchain")
    cfg["ckpt_in"] = str(Path(MODEL_DIR) / ckpt_name)
    cfg["direct_pose_arm_split_enable"] = True
    cfg["direct_pose_arm_bones"] = ARM_BONES
    json.dump(cfg, open(dst, "w"), indent=2)
    print(dst)
PY
```

#### 11.8.2 训练指令（按顺序串跑）

```bash
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260227_fromarmchain.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json
```

#### 11.8.3 验证与验收（必须做）

1) freerun 导出：至少导出 `70R / 71 / 72` 三个节点。  
2) 用 `tools/build_stage7_old_new_summary.py` 统一对比到 `2way Stage6` 基线：  
   - old: `debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json`
   - new: `70R/71/72` 各自 freerun json

```bash
# freerun: 70R / 71 / 72
for TAG in \
  "70R_nonleg_recovery_proj256_preleg" \
  "71_legonly_after_nonlegproj256" \
  "72_legomega_after_nonlegproj256"; do
  PYTHONPATH=. python -m train.validate.run_freerun_cycles \
    --teacher validate/teacher_batches/Walk_F_teacher.json \
    --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_${TAG}_20260227_fromarmchain.pth \
    --bundle raw_data/processed_data/norm_template.json \
    --pretrain-template models/pretrain_template.json \
    --encoder-bundle models/motion_encoder_equiv_stageA.pt \
    --npz-root raw_data/processed_data \
    --out debug_output/verify_stage7_${TAG}_fromarmchain_20260227 \
    --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
    --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
    --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force
done

# summary: 2way Stage6 baseline vs 70R / 71 / 72
python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_70R_nonleg_recovery_proj256_preleg_fromarmchain_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split2way_vs_stage70R_fromarmchain_20260227

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_71_legonly_after_nonlegproj256_fromarmchain_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split2way_vs_stage71_fromarmchain_20260227

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_72_legomega_after_nonlegproj256_fromarmchain_20260227/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split2way_vs_stage72_fromarmchain_20260227
```

验收标准（leg8 回收）：
- **硬验收**：`72` 相对 `2way Stage6` 的 `leg8_mean_delta <= 0`（完全回收 Stage6 的 `+0.117` 退步）。
- **软验收**：若未到 0，要求 `72` 至少回收 70% 以上退步（即 `leg8_mean_delta <= +0.035`）。
- 同时要求 non-leg 不塌陷：`72` 相对 `2way Stage6` 的 `non_leg_mean_delta < 0`。

#### 11.8.4 实跑结果（2026-02-27）

已按 11.8.1~11.8.3 全部执行完毕：

1) 配置生成（from-armchain）  
- 已生成 7 份配置：`70a/70b/70c/70R/71/72/lambda`，文件名均为 `20260227_fromarmchain` 版本。  

2) 串行训练（70a→70b→70c→70R→71→72→lambda）  
- 7 个节点全部完成并产出 ckpt/log：  
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70a_splitB2_pe32h512_20260227_fromarmchain.pth`  
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.pth`  
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70c_replacecontacts_splitB2_pe32h512_20260227_fromarmchain.pth`  
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260227_fromarmchain.pth`  
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_71_legonly_after_nonlegproj256_20260227_fromarmchain.pth`  
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_72_legomega_after_nonlegproj256_20260227_fromarmchain.pth`  
  - `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.pth`  

3) freerun + summary 对比（old=`2way Stage6`）  
- freerun 输出：  
  - `debug_output/verify_stage7_70R_nonleg_recovery_proj256_preleg_fromarmchain_20260227/Walk_F_freerun_cycles.json`  
  - `debug_output/verify_stage7_71_legonly_after_nonlegproj256_fromarmchain_20260227/Walk_F_freerun_cycles.json`  
  - `debug_output/verify_stage7_72_legomega_after_nonlegproj256_fromarmchain_20260227/Walk_F_freerun_cycles.json`  
- 对比摘要：  
  - `debug_output/verify_stage6split2way_vs_stage70R_fromarmchain_20260227/global_signal_summary.txt`  
  - `debug_output/verify_stage6split2way_vs_stage71_fromarmchain_20260227/global_signal_summary.txt`  
  - `debug_output/verify_stage6split2way_vs_stage72_fromarmchain_20260227/global_signal_summary.txt`  

关键指标（相对 `2way Stage6`）：

| Node | mean_delta | leg8_mean_delta | non_leg_mean_delta | 判定 |
|---|---:|---:|---:|---|
| 70R | -0.009564 | +0.540423 | -0.128480 | non-leg 改善，但 leg 明显退化 |
| 71 | -0.161938 | -0.316680 | -0.128480 | leg 回收通过（硬验收通过） |
| 72 | -0.175031 | -0.390328 | -0.128480 | 最优，硬验收通过 |

验收结论：
- `72 leg8_mean_delta=-0.390328 <= 0`，**硬验收通过**。  
- `72 leg8_mean_delta=-0.390328 <= +0.035`，**软验收通过**。  
- `72 non_leg_mean_delta=-0.128480 < 0`，**non-leg guard 通过**。  
- 结论：11.8 目标完成，且 `72(from-armchain)` 已达到并超过预设回收标准。

#### 11.8.5 待验证路线：跳过 7.0c（`70b -> 70R -> 71 -> 72 -> lambda`）

定位：
- 该路线作为**待验证/备选链路**保留，不替代当前主链（含 `70c`）默认执行顺序。  
- 核心用途：验证 `70b(concat)` 直接切 `70R(replace_contacts)` 时，leg/gate warm-start 兼容性是否稳定。

前置代码（已具备）：
- `train/posttrain.py` 已补充 phase 输入维度适配到以下 first-layer 权重：  
  - `direct_pose_leg_head.0.weight`  
  - `direct_pose_leg_head_shared.0.weight`  
  - `direct_pose_leg_gate_head.0.weight`  
  - `direct_pose_leg_gate_head_shared.0.weight`  
- 目的：避免 `70b -> 70R` 时因 in-dim 变化触发 shape mismatch，导致 leg 侧权重静默掉载。

1) 配置（已生成）
- `config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260228_fromarmchain_skip70c.json`
- `config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260228_fromarmchain_skip70c.json`
- `config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260228_fromarmchain_skip70c.json`
- `config/posttrain_WalkF_stage7_lambda_final_calib_20260228_fromarmchain_skip70c_fullcompat.json`
- 关键起点：`70R.ckpt_in = .../ckpt_last_WalkF_stage7_70b_phasezin_splitB2_pe32h512_20260227_fromarmchain.pth`（直接跳过 `70c`）。

2) 训练指令（skip70c）

```bash
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260228_fromarmchain_skip70c.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_71_legonly_after_nonlegproj256_20260228_fromarmchain_skip70c.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_72_legomega_after_nonlegproj256_20260228_fromarmchain_skip70c.json
PYTHONPATH=. python -m train.posttrain --config config/posttrain_WalkF_stage7_lambda_final_calib_20260228_fromarmchain_skip70c_fullcompat.json
```

3) 对应模型产物（当前）
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_70R_nonleg_recovery_proj256_preleg_20260228_fromarmchain_skip70c.pth`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_71_legonly_after_nonlegproj256_20260228_fromarmchain_skip70c.pth`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_72_legomega_after_nonlegproj256_20260228_fromarmchain_skip70c.pth`
- `models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_lambda_final_calib_20260228_fromarmchain_skip70c_fullcompat.pth`

4) 验证指令（建议固定使用 `skip70c_fixlegadapt_20260228` 输出目录）

```bash
# freerun: 70R / 71 / 72
for TAG in \
  "70R_nonleg_recovery_proj256_preleg" \
  "71_legonly_after_nonlegproj256" \
  "72_legomega_after_nonlegproj256"; do
  PYTHONPATH=. python -m train.validate.run_freerun_cycles \
    --teacher validate/teacher_batches/Walk_F_teacher.json \
    --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_${TAG}_20260228_fromarmchain_skip70c.pth \
    --bundle raw_data/processed_data/norm_template.json \
    --pretrain-template models/pretrain_template.json \
    --encoder-bundle models/motion_encoder_equiv_stageA.pt \
    --npz-root raw_data/processed_data \
    --out debug_output/verify_stage7_${TAG}_fromarmchain_skip70c_fixlegadapt_20260228 \
    --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
    --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
    --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force
done

# freerun: lambda
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__base_to_stage7_lambdafinal_20260226_cpu/ckpt_last_WalkF_stage7_lambda_final_calib_20260228_fromarmchain_skip70c_fullcompat.pth \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/verify_stage7_lambda_final_calib_fromarmchain_skip70c_fixlegadapt_20260228 \
  --rounds 5 --depth 3 --time-index-mode cycle --time-index-cycle-minus1 \
  --phase_reset_source none --lambda_fusion_apply --so3_corr_apply \
  --log_contacts --export_joint_geolocal --export_joint_direct_geolocal_series --force

# old(2way Stage6) vs skip70c chain
python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_70R_nonleg_recovery_proj256_preleg_fromarmchain_skip70c_fixlegadapt_20260228/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split2way_vs_stage70R_fromarmchain_skip70c_fixlegadapt_20260228

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_71_legonly_after_nonlegproj256_fromarmchain_skip70c_fixlegadapt_20260228/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split2way_vs_stage71_fromarmchain_skip70c_fixlegadapt_20260228

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_72_legomega_after_nonlegproj256_fromarmchain_skip70c_fixlegadapt_20260228/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split2way_vs_stage72_fromarmchain_skip70c_fixlegadapt_20260228

python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage6_split2way_recheck_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_lambda_final_calib_fromarmchain_skip70c_fixlegadapt_20260228/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage6split2way_vs_stage7lambda_fromarmchain_skip70c_fixlegadapt_20260228

# 主链 lambda(fromarmchain) vs skip70c lambda（检验 non-leg 是否保持不变）
python tools/build_stage7_old_new_summary.py \
  --old-json debug_output/verify_stage7_lambda_final_calib_fromarmchain_20260227/Walk_F_freerun_cycles.json \
  --new-json debug_output/verify_stage7_lambda_final_calib_fromarmchain_skip70c_fixlegadapt_20260228/Walk_F_freerun_cycles.json \
  --out-dir debug_output/verify_stage7lambda_fromarmchain_vs_skip70c_fixlegadapt_20260228
```

5) 当前观察（2026-02-28，作为“可继续验证”依据）
- `70R`（old=`2way Stage6`）：
  - 未修复适配前：`leg8_mean_delta=+1.322617`
  - 修复后：`leg8_mean_delta=+0.009716`
- `lambda`（old=`2way Stage6`）：`leg8_mean_delta=-0.363599`，仍显著优于基线。  
- `lambda`（old=`主链 fromarmchain lambda`）：`non_leg_mean_delta=+0.000000`（non-leg 聚合保持不变），逐骨也全为 `0`。

建议验收（作为是否升级为正式路线的门槛）：
- A. 相对 `2way Stage6`：`lambda leg8_mean_delta <= -0.35` 且 `non_leg_mean_delta < 0`。  
- B. 相对主链 `fromarmchain lambda`：`non_leg_mean_delta == 0`（允许 `abs(delta)<=1e-6`）且 `leg8_mean_delta <= +0.03`。  
- 满足 A+B 时，可将 skip70c 由“待验证路线”提升为“可切换路线”；否则继续保持主链默认含 `70c`。
