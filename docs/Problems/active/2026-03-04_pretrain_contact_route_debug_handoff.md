# 2026-03-04 Pretrain Contact 路由到 Post-train 的调试交接

Last updated: 2026-03-04

## 1) 问题定义（当前焦点）

目标不是“立刻替换 trainbase contact”，而是先验证：

1. `pretrain contact_head` 是否在 post-train freerun 里可用；
2. 若不可用，问题是“信号无信息”还是“信号分布/scale 不匹配”；
3. 在最小改动下，是否能把它变成可用的 additive signal（减轻 trainbase 职责）。

当前关键判断（本轮结论）：
- `pretrain_contact_raw ≈ zero` 更像是 **scale/分布失配**，不是“pretrain 信号本身无信息”。
- Phase B（freerun 分布拟合的 affine）已成为当前主线赢家：`ContactErrAbsMean` 显著优于 `pretrain_clamp1`，且不伤 pose。
- Phase B 在当前 5 clip 聚合口径下已 **优于 model baseline**（`ContactErrAbsMean 0.051464 < 0.069735`），因此本阶段结论不是“仅可用”，而是“在目标指标上更优”。
- Phase C（tiny GRU anchor）当前版本虽可提升 `ContactMeasGtAbsMean`，但 `ContactErrAbsMean` 未超过 Phase B。

---

## 2) 已落地改动（便于新会话接管）

代码侧已接通 runtime source：

1. `EventMotionModel.attach_motion_encoder()` 支持加载并冻结 `contact_head`
   - `train/models.py`
   - 新字段：`frozen_contact_head`

2. `run_freerun_cycles` 新增 contact source
   - `train/validate/run_freerun_cycles.py`
   - 新选项：`--contacts_meas_source pretrain_contact`
   - 新选项：`--contacts_meas_pretrain_clamp`（默认 `1.0`）
   - 新选项：`--contacts_meas_pretrain_affine_stats`（Phase B：logit-space affine 校准）
   - 新选项：`--contacts_meas_pretrain_anchor_ckpt`（Phase C：tiny GRU anchor）
   - 实现细节：pretrain source 输入时将 contact 通道置零，避免 trivial leakage。

3. 新增离线 AUC 诊断脚本
   - `tools/diag_pretrain_contact_vs_period_auc.py`
   - 用于比较 `contact_head` vs `soft_period[:2]`。

4. 新增 Phase B / Phase C 拟合脚本
   - `tools/fit_pretrain_contact_affine_from_freerun.py`
   - `tools/fit_pretrain_contact_tinygru_from_freerun.py`
   - 两者都以 freerun `round>=1` 样本为拟合分布（不是 teacher-forcing）。

---

## 3) 数据现状

### 3.1 离线（teacher分布）AUC：`contact_head` vs `soft_period[:2]`

产物：
- `debug_output/_tmp_pretrain_contact_auc_stageA_full_20260304/summary.md`
- `debug_output/_tmp_pretrain_contact_auc_stageA_nocontact_20260304/summary.md`
- `debug_output/_tmp_pretrain_contact_auc_stageA_nocontact_proj_20260304/summary.md`

结论（stageA bundle）：
- 两者几乎等价，`contact_head` 仅微弱领先：
  - soft AUC mean: `contact=0.915870` vs `period=0.913385`，`delta=+0.002484`
- 去掉输入 contact 通道后，结论基本不变（delta 仍约 `+0.002~0.003`）。

解释：
- 这支持“信息高度重叠”，也解释了为什么直接替换不会带来显著收益。

### 3.2 在线（freerun分布）A/B：`contacts_meas_source`

口径：
- clip: `Walk_F`
- rounds=3, depth=3, `phase_reset_source=none`, `time-index-mode=cycle`
- 统计 `round>=1` 均值

总汇总：
- `debug_output/_tmp_pretrain_contact_route_ab_20260304/summary_contact_source_ablation.md`

关键结果：

| run | GeoLocalDegWeighted | GeoDeg | DirectGeoLocalDegWeighted | ContactErrAbsMean | ContactMeasGtAbsMean |
|---|---:|---:|---:|---:|---:|
| model | 49.872872 | 63.902237 | 5.025246 | 0.074663 | 0.414027 |
| pretrain_contact_raw | 48.882255 | 63.126816 | 5.071418 | 0.486667 | 0.485791 |
| pretrain_contact_clamp1 | 49.845362 | 63.826471 | 5.030010 | 0.126251 | 0.473949 |
| zero | 48.878452 | 63.121481 | 5.070600 | 0.485914 | 0.485791 |

关键观察：

1. `pretrain_contact_raw` 与 `zero` 在 contact 指标上几乎重合  
   (`ContactErrAbsMean 0.486667 vs 0.485914`)。
2. `pretrain_contact_clamp1` 后，pose 主指标恢复到接近 model  
   (`GeoLocalDegWeighted` 仅 `-0.0275`)。
3. 但 contact 对齐仍明显弱于 model  
   (`ContactErrAbsMean +0.0516`, `ContactMeasGtAbsMean +0.0599`)。

补充行为证据（cycle>=1）：
- model 的 `ContactMeasPerC` 近似 `[0.497, 0.501]`（窄分布）；
- pretrain raw 出现快速塌缩（大量趋近 0）；
- pretrain clamp1 变为可变分布（std明显增大），但与 GT 对齐仍偏弱。

### 3.3 本轮新增（5 clip 聚合）：Phase B vs Phase C

口径：
- clips=`Walk_F/L2L/L2R/R2L/R2R`
- rounds=3, depth=3, `phase_reset_source=none`, `time-index-mode=cycle`
- 统计 `round>=1` 的 per-clip 均值再聚合

汇总产物：
- `debug_output/_tmp_phasec_anchor_20260304/summary_phasec_anchor_vs_phaseb_v2.md`
- `debug_output/_tmp_phaseb_affine_20260304/summary_phaseb_affine_sweep.md`

关键结果：

| run | GeoLocalDegWeighted | ContactErrAbsMean | ContactMeasGtAbsMean |
|---|---:|---:|---:|
| model | 43.321617 | 0.069735 | 0.392455 |
| pretrain_clamp1 | 43.305950 | 0.090227 | 0.433522 |
| phaseb_affine_mix08 | 43.282489 | **0.051464** | 0.392850 |
| phasec_anchor_mix08 | 43.274554 | 0.085538 | **0.361187** |
| phasec_anchor_mix08_plus_affine | 43.306068 | 0.110214 | 0.390871 |

解读：
1. **Phase B affine 明确胜出**：`ContactErrAbsMean` 显著低于 `pretrain_clamp1`，且优于 `model`，pose 无劣化。
2. Phase C anchor 当前版本更偏向时序平滑（`ContactMeasGtAbsMean` 最好），但对 `ContactErrAbsMean` 帮助有限。
3. anchor + affine 非互补，当前组合反而恶化 `ContactErrAbsMean`。

指标语义提醒（避免误判）：
- `ContactErrAbsMean` 是 `|ContactPlanPerC - ContactMeasPerC|`；
- `ContactMeasGtAbsMean` 是 `|ContactMeasPerC - ContactGTPerC|`；
- 二者优化方向不必然一致，因此会出现“Meas 更像 GT，但 Plan-Meas 残差变差”的情况。

---

## 4) 当前解释框架（用于后续 debug）

### 4.1 “raw≈zero”的含义

更可能是 **输入 OOD + 值域失配** 导致 pretrain 支路饱和/失效，而不是“信号语义缺失”。

证据链：
- `raw` 与 `zero` 几乎一致；
- 仅加 clamp（不改模型权重）后，主 pose 指标立刻回到接近 baseline。

### 4.2 为什么 `ContactErrAbsMean` 仍偏高（+0.052）

当前两个高优先级原因：

1. 分布偏移（teacher-forcing -> freerun）
   - pretrain contact_head 训练时输入更干净；
   - freerun 下 `motion/pose_hist` 漂移，未做鲁棒适配。

2. 缺少时序记忆
   - pretrain StepHead 是逐帧 MLP；
   - 在 swing/stance 边界易抖动；
   - 对比 trainbase 链路的时序成分，边界稳定性可能不足（历史假设，优先级已下调）。

本轮更新（2026-03-04）：
- 从实测结果看，**主因是分布/scale 失配**，不是时序能力短板：Phase B affine 可将 `ContactErrAbsMean` 从 `0.090227` 降到 `0.051464`（并优于 `model=0.069735`）；而 Phase C GRU 未在该指标上超过 Phase B。
- 因此当前阶段“缺少时序记忆”不再作为主瓶颈假设，仅保留为后续备选方向。

---

## 5) 当前结论（更新到本轮）

1. **Phase B affine 已满足并超过当前门槛**：在 5 clip 聚合下，`ContactErrAbsMean` 明显优于 `pretrain_clamp1`，且优于 `model`，pose 不劣化。
2. Phase C tiny GRU anchor 本轮不是主线赢家：虽然 `ContactMeasGtAbsMean` 更低，但 `ContactErrAbsMean` 未超过 Phase B。
3. 结论从“先上 GRU”调整为“**先固化 Phase B 主线**，Phase C 暂作备选研究分支”。

---

## 6) 下一步（按主线流程一致性推进）

### 6.1 与现有流程一致性（必须遵守）

本节保留的是 2026-03-04 当时的 handoff 语境；截至 2026-03-09，当前主线 policy（见 `docs/posttrain_pipeline.md`）已经更新为：

1. `posttrain` rollout contacts 固定为 `pretrain_contact`，不再保留 `whitebox` 默认线。
2. `whitebox` 已退休为历史 reference；validate/control 执行建议只保留 `pretrain_contact|model|gt|zero`。

### 6.2 下一步执行策略（更新版）

1. 下面的 `whitebox vs pretrain+affine` A/B 设计仅作为历史记录保留，不再作为当前执行建议。
2. 当前主线只保留 `posttrain_contacts_source=pretrain_contact` 所需字段：
   - `posttrain_contacts_source=pretrain_contact`
   - `posttrain_contacts_pretrain_clamp`
   - `posttrain_contacts_pretrain_affine_stats`
3. 当时的结论已经固定：`pretrain_contact + clamp1 + affine_mix08` 优于历史 `whitebox` 对照，无需继续保留 control lane。
4. 当前如需复跑，直接按 `docs/posttrain_pipeline.md` 的 `pretrain_contact` 主线命令执行。

### 6.3 新的验收门槛（Phase D / posttrain 接入）

在 `round>=1` + 5 clip 聚合口径下：

1. `GeoLocalDegWeighted`：不劣化（相对 whitebox 基线容忍 `<= +0.05`）。
2. `ContactErrAbsMean`：相对 whitebox 基线显著下降（目标至少 `-0.02`）。
3. source 行为可追溯：日志中 `ContactsMeasSourceApplied` 必须稳定落在预期分支。
4. 训练流程兼容：不开实验开关时，结果与当前主线一致。

---

## 7) 复现实验命令（本轮核心）

说明：当前 `run_freerun_cycles` 中 `--contacts_meas_pretrain_clamp` 默认值为 `1.0`。  
若要复现本文 “pretrain_contact_raw≈zero” 的结论，raw 实验需显式传 `--contacts_meas_pretrain_clamp 0`。

```bash
# baseline
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__regcheck_20260301/exp_phase_DirectBranch_v1_d1_regcheck_20260301/ckpt_last_exp_phase_DirectBranch_v1_d1_regcheck_20260301.pth \
  --rounds 3 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source model --log_contacts \
  --out debug_output/_tmp_pretrain_contact_route_ab_20260304/model_log --force

# pretrain_contact raw
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__regcheck_20260301/exp_phase_DirectBranch_v1_d1_regcheck_20260301/ckpt_last_exp_phase_DirectBranch_v1_d1_regcheck_20260301.pth \
  --rounds 3 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 0 --log_contacts \
  --out debug_output/_tmp_pretrain_contact_route_ab_20260304/pretrain_contact_log --force

# pretrain_contact + clamp
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__regcheck_20260301/exp_phase_DirectBranch_v1_d1_regcheck_20260301/ckpt_last_exp_phase_DirectBranch_v1_d1_regcheck_20260301.pth \
  --rounds 3 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 1.0 --log_contacts \
  --out debug_output/_tmp_pretrain_contact_route_ab_20260304/pretrain_contact_log_clamp1 --force

# zero reference
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1__regcheck_20260301/exp_phase_DirectBranch_v1_d1_regcheck_20260301/ckpt_last_exp_phase_DirectBranch_v1_d1_regcheck_20260301.pth \
  --rounds 3 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source zero --log_contacts \
  --out debug_output/_tmp_pretrain_contact_route_ab_20260304/zero_log --force
```

---

## 8) Phase B 实测落地（freerun 分布拟合）

### 8.1 口径（必须）

1. **拟合数据必须来自 freerun rollout**（不是 teacher-forcing）。
2. 使用 `round>=1` 样本拟合（规避 round0 冷启动偏差）。
3. `pretrain_contact` 路由显式写 `--contacts_meas_pretrain_clamp 1.0`。

### 8.2 推荐参数（当前最佳）

基于 5 个 walk clip (`Walk_F/L2L/L2R/R2L/R2R`) 的 sweep，当前推荐：

- Phase B target: `mix`
- `mix_alpha=0.8`
- affine 形式：logit-space `p' = sigmoid(b + s * logit(p))`

相对 `pretrain_clamp1` 的聚合收益（`round>=1`）：
- `ContactErrAbsMean`: `0.090227 -> 0.051464`（`-0.038763`）
- `ContactMeasGtAbsMean`: `0.433522 -> 0.392850`（`-0.040672`）
- `GeoLocalDegWeighted`: `43.305950 -> 43.282489`（无劣化）

### 8.3 复现命令（5 clip）

```bash
# A) 生成拟合源（pretrain_contact + clamp1）
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
           validate/teacher_batches/Walk_L_To_L_teacher.json \
           validate/teacher_batches/Walk_L_To_R_teacher.json \
           validate/teacher_batches/Walk_R_To_L_teacher.json \
           validate/teacher_batches/Walk_R_To_R_teacher.json \
  --model models/MLPL2_DirectBranch_v1__regcheck_20260301/exp_phase_DirectBranch_v1_d1_regcheck_20260301/ckpt_last_exp_phase_DirectBranch_v1_d1_regcheck_20260301.pth \
  --rounds 3 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 1.0 --log_contacts \
  --out debug_output/_tmp_phaseb_affine_20260304/pretrain_clamp1_fit_source --force

# B) 在 freerun 分布上拟合 affine（推荐 mix_alpha=0.8）
PYTHONPATH=. python tools/fit_pretrain_contact_affine_from_freerun.py \
  --json debug_output/_tmp_phaseb_affine_20260304/pretrain_clamp1_fit_source \
  --round-gte 1 --require-source-prefix pretrain_contact \
  --target mix --mix-alpha 0.8 \
  --out-json debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --out-md   debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/summary.md

# C) 评估 Phase B（pretrain_contact + clamp1 + affine）
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
           validate/teacher_batches/Walk_L_To_L_teacher.json \
           validate/teacher_batches/Walk_L_To_R_teacher.json \
           validate/teacher_batches/Walk_R_To_L_teacher.json \
           validate/teacher_batches/Walk_R_To_R_teacher.json \
  --model models/MLPL2_DirectBranch_v1__regcheck_20260301/exp_phase_DirectBranch_v1_d1_regcheck_20260301/ckpt_last_exp_phase_DirectBranch_v1_d1_regcheck_20260301.pth \
  --rounds 3 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json \
  --log_contacts --out debug_output/_tmp_phaseb_affine_20260304/eval_pretrain_affine_mix08 --force
```

---

## 9) Phase C 启动（Pretrain -> TinyGRU Anchor）[暂停]

新增 runtime 开关：

- `--contacts_meas_pretrain_anchor_ckpt <ckpt.pt>`
  - 仅在 `--contacts_meas_source pretrain_contact` 下生效
  - 在 pretrain_contact（及可选 affine）之后应用 tiny GRU anchor

### 9.1 推荐首轮配置（MVP）

- 输入：`[c_pre, Δc_pre]`（4D）
- `hidden_dim=16`
- 损失：`1.0 * BCE + 0.05 * smooth + 0.02 * consistency`
- 监督目标：`target=mix, mix_alpha=0.8`
- 训练样本：**base freerun rollout 的 `round>=1`**

### 9.2 Phase C 命令模板

```bash
# A) 在 freerun 分布上训练 tiny GRU anchor
PYTHONPATH=. python tools/fit_pretrain_contact_tinygru_from_freerun.py \
  --json debug_output/_tmp_phaseb_affine_20260304/pretrain_clamp1_fit_source \
  --round-gte 1 --require-source-prefix pretrain_contact \
  --target mix --mix-alpha 0.8 \
  --hidden-dim 16 --epochs 300 --lr 1e-2 \
  --w-bce 1.0 --w-smooth 0.05 --w-consistency 0.02 \
  --out-ckpt debug_output/_tmp_phasec_anchor_20260304/anchor_mix08_h16_e300.pt \
  --out-md   debug_output/_tmp_phasec_anchor_20260304/anchor_mix08_h16_e300.md

# B) 在 freerun 中接入 anchor 评估
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
           validate/teacher_batches/Walk_L_To_L_teacher.json \
           validate/teacher_batches/Walk_L_To_R_teacher.json \
           validate/teacher_batches/Walk_R_To_L_teacher.json \
           validate/teacher_batches/Walk_R_To_R_teacher.json \
  --model models/MLPL2_DirectBranch_v1__regcheck_20260301/exp_phase_DirectBranch_v1_d1_regcheck_20260301/ckpt_last_exp_phase_DirectBranch_v1_d1_regcheck_20260301.pth \
  --rounds 3 --depth 3 --time-index-mode cycle --phase_reset_source none \
  --contacts_meas_source pretrain_contact --contacts_meas_pretrain_clamp 1.0 \
  --contacts_meas_pretrain_anchor_ckpt debug_output/_tmp_phasec_anchor_20260304/anchor_mix08_h16_e300.pt \
  --log_contacts --out debug_output/_tmp_phasec_anchor_20260304/eval_anchor_mix08_h16_e300 --force
```

### 9.3 首轮结果（2026-03-04，当天实测）

汇总：`debug_output/_tmp_phasec_anchor_20260304/summary_phasec_anchor_vs_phaseb_v2.md`

聚合（5 clip，`round>=1`）：

| run | GeoLocalDegWeighted | ContactErrAbsMean | ContactMeasGtAbsMean |
|---|---:|---:|---:|
| pretrain_clamp1 | 43.305950 | 0.090227 | 0.433522 |
| phaseb_affine_mix08 | 43.282489 | **0.051464** | 0.392850 |
| phasec_anchor_mix08 | 43.274554 | 0.085538 | **0.361187** |
| phasec_anchor_mix08_plus_affine | 43.306068 | 0.110214 | 0.390871 |

阶段性判断：

1. Phase C tiny GRU anchor 已成功接入（`ContactsMeasSourceApplied=pretrain_contact_anchor`）。
2. 当前这版 anchor 明显改善 `ContactMeasGtAbsMean`（`-0.072335` vs clamp1）。
3. 但 `ContactErrAbsMean` 仅小幅改善（`-0.004689` vs clamp1），尚未超过 Phase B affine（`-0.038763`）。
4. anchor + affine 组合在本轮并非互补，`ContactErrAbsMean` 进一步变差（`0.110214`）。
5. 结论：本轮决策为 **Phase B 主线、Phase C 暂停主线推进**。
6. 重启条件：仅当 Phase B 在 Phase D（posttrain 接入）A/B 中未达到 6.3 门槛，才恢复 Phase C 线继续优化。

---

## 10) 当前下一步（执行清单，Phase D 接入计划）

计划执行日期：`2026-03-05`

### 10.1 D0：冻结实验资产（开工前）

1. 固定 control/experiment 共用资产（避免实验漂移）：
   - `encoder_bundle`: `models/motion_encoder_equiv.pt.best.pt`
   - `affine_stats`: `debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json`
   - clips: `Walk_F/L2L/L2R/R2L/R2R`
2. 固定输出目录前缀：
   - `debug_output/_tmp_phaseD_posttrain_contact_route_20260305/`
3. 本轮当时只比较两组：
   - A（historical control at the time）: `whitebox`
   - B（experiment / current accepted route）: `pretrain_contact + clamp1 + affine_mix08`

### 10.2 D1：代码接入（历史记录，现已收缩）

目标（当时）：先以增量方式接入 `pretrain_contact`。当前主线已进一步收缩为仅接受 `pretrain_contact`。

改动点（文件级）：

1. `train/posttrain.py` 的 `PostTrainConfig` 当时新增：
   - `posttrain_contacts_source`（历史过渡期曾允许 source 切换；当前主线仅接受 `pretrain_contact`）
   - `posttrain_contacts_pretrain_clamp`（float，默认 `1.0`）
   - `posttrain_contacts_pretrain_affine_stats`（Optional[str]）
2. `train/posttrain.py` 的 `_cfg_parse_lambda_rollout` 与 `_build_posttrain_arg_parser` 接入了这些键。
3. 过渡期 source 路由对比已完成；当前主线只保留：
   - `pretrain_contact`: 使用 frozen encoder + frozen contact_head 预测，输入 contact 通道置零，支持 clamp + affine（与 `run_freerun_cycles` 逻辑对齐）。
4. 当前仍保留的 fail-fast contract：
   - 当 `posttrain_contacts_source=pretrain_contact` 且缺失 `encoder_bundle`/`frozen_contact_head` 时直接报错退出（避免 silent fallback 污染结论）。

### 10.3 D2：最小回归（smoke，先通路再长跑）

1. 代码静态检查：

```bash
python3 -m py_compile train/posttrain.py train/validate/run_freerun_cycles.py train/models.py
```

2. historical whitebox smoke（control）：

该命令对应的过渡期 control lane 已退休，不再建议在当前 mainline 复跑。

3. pretrain+affine smoke（experiment）：

```bash
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_WalkF_stage6_direct_cond_anchor_splitfirst_3way_armchain_pe32h512_20260227.json \
  --epochs 1 --steps_per_epoch 5 \
  --encoder_bundle models/motion_encoder_equiv.pt.best.pt \
  --out_dir models/__tmp_phaseD_route_smoke --run_name phaseD_smoke_pretrain_affine \
  --posttrain_contacts_source pretrain_contact \
  --posttrain_contacts_pretrain_clamp 1.0 \
  --posttrain_contacts_pretrain_affine_stats debug_output/_tmp_phaseb_affine_20260304/affine_fit_mix08/affine_stats.json
```

### 10.4 D3：正式 A/B（同 seed、同配置）

1. `direct` 配置跑 A/B（历史记录）：
   - historical control：原配置 + retired whitebox route
   - experiment：原配置 + `posttrain_contacts_source=pretrain_contact` + clamp/affine
2. `lambda` 配置跑 A/B（同上；当前不再建议保留 whitebox control lane）：
   - `config/posttrain_WalkF_stage7_lambda_final_calib_20260227_fromarmchain_fullcompat.json`
3. 输出目录建议：
   - `.../ab_direct_whitebox`
   - `.../ab_direct_pretrain_affine`
   - `.../ab_lambda_whitebox`
   - `.../ab_lambda_pretrain_affine`

### 10.5 D4：评估与门槛判定（5 clip，`round>=1`）

1. 对每个 A/B ckpt 运行同口径 freerun（clips=5，rounds=3，depth=3，`phase_reset_source=none`，`time-index-mode=cycle`）。
2. 当时的 control 评估源固定 `whitebox`；该路线现已退休，当前复跑应直接使用 `pretrain_contact + clamp1 + affine_mix08`。
3. 判定门槛（沿用 6.3）：
   - `GeoLocalDegWeighted`：相对 control 不劣化（`<= +0.05`）
   - `ContactErrAbsMean`：相对 control 至少 `-0.02`
   - `ContactsMeasSourceApplied`：稳定命中预期分支（whitebox / pretrain_contact_affine）
4. 产物：
   - `summary_phaseD_posttrain_ab.md`
   - `summary_phaseD_posttrain_ab.json`

### 10.6 失败分支与回滚策略

1. 当时的回滚方案是保持 `whitebox` 主线不动；该回滚线现已退休，仅保留为历史记录。
2. 若 pose 不劣化但 `ContactErrAbsMean` 未达 `-0.02`：回到 Phase B 参数微调（`mix_alpha`/per-clip 拟合），不进入主线。
3. 若 source 命中不稳定：先修 runtime route 与日志追踪，再重跑 A/B；禁止带不确定 source 下结论。
4. 仅当 D4 全门槛通过，才进入“将 Phase B 升级到 posttrain 主线 contract”的评审。

---

## 11) Phase D 实际执行进展（2026-03-04 当晚）

### 11.1 D1 已落地（posttrain 接入完成）

已在 `train/posttrain.py` 落地以下能力。注意：当前 mainline 已不再保留 whitebox 默认行为。

1. 新增配置/CLI：
   - `posttrain_contacts_source`
   - `posttrain_contacts_pretrain_clamp`
   - `posttrain_contacts_pretrain_affine_stats`
2. rollout contacts source 新增 `pretrain_contact` 路由：
   - 使用 frozen `encoder + contact_head`；
   - 输入 contact 通道置零；
   - 支持 clamp + logit-space affine。
3. fail-fast：
   - `posttrain_contacts_source=pretrain_contact` 但缺失可用 `encoder_bundle/contact_head` 时直接报错退出（禁止 silent fallback）。

### 11.2 D2 smoke（通过）

输出目录：`models/__tmp_phaseD_route_smoke`

1. `phaseD_smoke_whitebox`：通过
   - `avg_total=1.762837`, `ok_steps=5`, `skipped=0`
2. `phaseD_smoke_pretrain_affine`：通过
   - `avg_total=1.783996`, `ok_steps=5`, `skipped=0`
3. 负路径校验：通过
   - 传入不存在 `encoder_bundle` 时触发 `[FATAL]`（fail-fast 生效）

### 11.3 D3 正式 A/B 训练（完成）

输出目录：`models/__tmp_phaseD_posttrain_ab_20260305`

1. `phaseD_ab_direct_whitebox`
   - `epoch5 avg_total=1.871174`
2. `phaseD_ab_direct_pretrain_affine`
   - `epoch5 avg_total=1.918729`
3. `phaseD_ab_lambda_whitebox`
   - `epoch1 avg_total=0.022339`
4. `phaseD_ab_lambda_pretrain_affine`
   - `epoch1 avg_total=0.022374`

四组 ckpt 已保存：
- `ckpt_last_phaseD_ab_direct_whitebox.pth`
- `ckpt_last_phaseD_ab_direct_pretrain_affine.pth`
- `ckpt_last_phaseD_ab_lambda_whitebox.pth`
- `ckpt_last_phaseD_ab_lambda_pretrain_affine.pth`

### 11.4 D4 评估与门槛判定（完成）

评估口径：
- clips=`Walk_F/L2L/L2R/R2L/R2R`
- rounds=3, depth=3, `phase_reset_source=none`, `time-index-mode=cycle`
- 统计 `round>=1`，先 per-clip 再跨 clip 聚合

汇总产物：
- `debug_output/_tmp_phaseD_posttrain_ab_20260305/summary_phaseD_posttrain_ab.md`
- `debug_output/_tmp_phaseD_posttrain_ab_20260305/summary_phaseD_posttrain_ab.json`

聚合结果：

| run | GeoLocalDegWeighted | ContactErrAbsMean | ContactMeasGtAbsMean | SourceMatchRate |
|---|---:|---:|---:|---:|
| direct_whitebox | 34.363971 | 0.289088 | 0.354103 | 1.0000 |
| direct_pretrain_affine | 34.397649 | 0.134621 | 0.425230 | 1.0000 |
| lambda_whitebox | 34.363971 | 0.289088 | 0.354103 | 1.0000 |
| lambda_pretrain_affine | 34.397649 | 0.134621 | 0.425230 | 1.0000 |

门槛判定（experiment - control）：

1. direct：
   - `GeoLocalDegWeighted = +0.033678`（通过 `<= +0.05`）
   - `ContactErrAbsMean = -0.154467`（通过 `<= -0.02`）
   - source 命中率 `1.0000`（通过）
   - **All gate pass**
2. lambda：
   - `GeoLocalDegWeighted = +0.033678`（通过）
   - `ContactErrAbsMean = -0.154467`（通过）
   - source 命中率 `1.0000`（通过）
   - **All gate pass**

### 11.5 当前结论（基于本轮实跑）

1. Phase D A/B 在既定门槛下已通过：`pretrain_contact + affine_mix08` 相对 whitebox 显著降低 `ContactErrAbsMean`，且 pose 未劣化。
2. source 路由追踪稳定：`ContactsMeasSourceApplied` 与配置一致（聚合命中率 1.0）。
3. 可进入下一步评审：是否将 Phase B 从 validate-lane 升级到 posttrain 主线 contract（仍建议先做一次主线 replay 复核）。
4. 本轮 `direct/lambda` 的聚合结果数值相同；这说明在当前评估口径下，结论主要由 contacts source A/B 主导。若后续需要分离“训练目标差异”影响，建议补一轮带 `lambda_fusion_apply` 的对照评估。

---

## 12) 追加实验：`--lambda_fusion_apply` 敏感性对照（已完成）

执行日期：`2026-03-04`

### 12.1 目标

在不改训练产物（复用 D3 四个 ckpt）的前提下，分离两类影响：

1. 固定 contacts source，观察 `direct vs lambda`（训练目标差异）。
2. 固定训练目标，观察 `whitebox vs pretrain_affine`（source 差异）。

### 12.2 评估口径

1. 复用 D3 四个 ckpt：
   - `direct_whitebox`
   - `direct_pretrain_affine`
   - `lambda_whitebox`
   - `lambda_pretrain_affine`
2. freerun 统一开启 `--lambda_fusion_apply`。
3. clips=`Walk_F/L2L/L2R/R2L/R2R`，`rounds=3`，`depth=3`，`phase_reset_source=none`，`time-index-mode=cycle`。
4. 统计 `round>=1`（实现中对应 `cycle>=1`），先 per-clip 均值，再跨 clip 聚合。

新增产物：
- `debug_output/_tmp_phaseD_posttrain_ab_20260305_lambda_apply/summary_phaseD_posttrain_ab_lambda_apply_sensitivity.md`
- `debug_output/_tmp_phaseD_posttrain_ab_20260305_lambda_apply/summary_phaseD_posttrain_ab_lambda_apply_sensitivity.json`

### 12.3 聚合结果（`--lambda_fusion_apply` 打开）

| run | GeoLocalDegWeighted | ContactErrAbsMean | ContactMeasGtAbsMean | SourceMatchRate |
|---|---:|---:|---:|---:|
| direct_whitebox | 34.363971 | 0.289088 | 0.354103 | 1.0000 |
| direct_pretrain_affine | 34.397649 | 0.134621 | 0.425230 | 1.0000 |
| lambda_whitebox | 7.313585 | 0.291946 | 0.445916 | 1.0000 |
| lambda_pretrain_affine | 7.311087 | 0.143879 | 0.432434 | 1.0000 |

### 12.4 差分拆解（本轮关键）

固定 source，看训练目标差异（`lambda - direct`）：

1. `whitebox`：
   - `GeoLocalDegWeighted = -27.050387`
   - `ContactErrAbsMean = +0.002858`
   - `ContactMeasGtAbsMean = +0.091813`
2. `pretrain_affine`：
   - `GeoLocalDegWeighted = -27.086562`
   - `ContactErrAbsMean = +0.009258`
   - `ContactMeasGtAbsMean = +0.007204`

固定训练目标，看 source 差异（`pretrain_affine - whitebox`）：

1. `direct`：
   - `GeoLocalDegWeighted = +0.033678`
   - `ContactErrAbsMean = -0.154467`
   - `ContactMeasGtAbsMean = +0.071127`
2. `lambda`：
   - `GeoLocalDegWeighted = -0.002498`
   - `ContactErrAbsMean = -0.148066`
   - `ContactMeasGtAbsMean = -0.013482`

### 12.5 结论更新（用于后续决策）

1. 这轮成功把“训练目标差异”从前一轮（`lambda_fusion_apply=off`）中分离出来：开启 `--lambda_fusion_apply` 后，`direct vs lambda` 在 `GeoLocalDegWeighted` 上出现大幅差异（约 `-27`）。
2. 同时，`ContactErrAbsMean` 上的训练目标差异仍较小（`+0.003 ~ +0.009`），说明本轮 contact 残差主结论依然主要由 source A/B 驱动。
3. `pretrain_affine` 相对 `whitebox` 在 `direct/lambda` 两条训练目标下都稳定降低 `ContactErrAbsMean`（约 `-0.15`），主线结论不变。

---

## 13) Phase D 全流程复跑（收尾验证，2026-03-05）

执行目的：
- 作为方向收尾，按同口径完整复跑 `posttrain` 接入 A/B（训练 + 评估），确认结论稳定性。

### 13.1 训练复跑（4 臂）

输出目录：
- `models/__tmp_phaseD_posttrain_ab_20260304_fullrerun`

ckpt：
- `ckpt_last_phaseD_full_direct_whitebox.pth`
- `ckpt_last_phaseD_full_direct_pretrain_affine.pth`
- `ckpt_last_phaseD_full_lambda_whitebox.pth`
- `ckpt_last_phaseD_full_lambda_pretrain_affine.pth`

训练收口（控制台末轮）：
1. direct_whitebox：`epoch5 avg_total=1.871174`
2. direct_pretrain_affine：`epoch5 avg_total=1.966494`
3. lambda_whitebox：`epoch1 avg_total=0.022339`
4. lambda_pretrain_affine：`epoch1 avg_total=0.022319`

### 13.2 D4 主口径复跑（`lambda_fusion_apply=off`）

评估目录（5 clips，`rounds=3/depth=3/cycle>=1`）：
- `debug_output/_tmp_phaseD_posttrain_ab_20260304_fullrerun/eval_nolambda_*`

汇总产物：
- `debug_output/_tmp_phaseD_posttrain_ab_20260304_fullrerun/summary_phaseD_posttrain_ab_fullrerun_nolambda.md`
- `debug_output/_tmp_phaseD_posttrain_ab_20260304_fullrerun/summary_phaseD_posttrain_ab_fullrerun_nolambda.json`

聚合结果：

| run | GeoLocalDegWeighted | ContactErrAbsMean | ContactMeasGtAbsMean | SourceMatchRate |
|---|---:|---:|---:|---:|
| direct_whitebox | 34.363971 | 0.289088 | 0.354103 | 1.0000 |
| direct_pretrain_affine | 34.394942 | 0.134328 | 0.424040 | 1.0000 |
| lambda_whitebox | 34.363971 | 0.289088 | 0.354103 | 1.0000 |
| lambda_pretrain_affine | 34.394942 | 0.134328 | 0.424040 | 1.0000 |

门槛判定（experiment-control）：
1. direct：`ΔGeo=+0.030971`，`ΔContactErr=-0.154760`，source 命中率 1.0，**all pass**
2. lambda：`ΔGeo=+0.030971`，`ΔContactErr=-0.154760`，source 命中率 1.0，**all pass**

### 13.3 敏感性复跑（`lambda_fusion_apply=on`）

评估目录（5 clips，带 `--lambda_fusion_apply`）：
- `debug_output/_tmp_phaseD_posttrain_ab_20260304_fullrerun/eval_lambda_*`

汇总产物：
- `debug_output/_tmp_phaseD_posttrain_ab_20260304_fullrerun/summary_phaseD_posttrain_ab_fullrerun_lambda_apply_sensitivity.md`
- `debug_output/_tmp_phaseD_posttrain_ab_20260304_fullrerun/summary_phaseD_posttrain_ab_fullrerun_lambda_apply_sensitivity.json`

聚合结果：

| run | GeoLocalDegWeighted | ContactErrAbsMean | ContactMeasGtAbsMean | SourceMatchRate |
|---|---:|---:|---:|---:|
| direct_whitebox | 34.363971 | 0.289088 | 0.354103 | 1.0000 |
| direct_pretrain_affine | 34.394942 | 0.134328 | 0.424040 | 1.0000 |
| lambda_whitebox | 7.313585 | 0.291946 | 0.445916 | 1.0000 |
| lambda_pretrain_affine | 7.311134 | 0.144257 | 0.431037 | 1.0000 |

固定 source 的训练目标差异（`lambda - direct`）：
1. `whitebox`：`ΔGeo=-27.050387`，`ΔContactErr=+0.002858`，`ΔContactMeasGt=+0.091813`
2. `pretrain_affine`：`ΔGeo=-27.083808`，`ΔContactErr=+0.009928`，`ΔContactMeasGt=+0.006998`

固定训练目标的 source 差异（`pretrain_affine - whitebox`）：
1. `direct`：`ΔGeo=+0.030971`，`ΔContactErr=-0.154760`，`ΔContactMeasGt=+0.069936`
2. `lambda`：`ΔGeo=-0.002450`，`ΔContactErr=-0.147689`，`ΔContactMeasGt=-0.014879`

### 13.4 逐骨骼诊断产物（`DirectGeoLocalDeg`）

产物目录：
- `debug_output/_tmp_phaseD_posttrain_ab_20260304_fullrerun/pairwise_global_signal_lambda_apply/`

包含：
1. `source_ab_direct.txt/.json`
2. `source_ab_lambda.txt/.json`
3. `objective_ab_whitebox.txt/.json`
4. `objective_ab_pretrain_affine.txt/.json`

上述文件已包含：
- `[overall]/[region_split]/[pointwise_signal]`
- `worst8_bones_by_mean_delta`
- `worst_points_for_worst8`

### 13.5 收尾结论（本次复跑）

1. `posttrain_contacts_source=pretrain_contact + affine_mix08` 的方向在本轮复跑下稳定成立：主口径 `ContactErrAbsMean` 相对白盒大幅下降（约 `-0.155`），且 `GeoLocalDegWeighted` 未越界（`+0.031 < +0.05`）。
2. `lambda_fusion_apply=on` 后，训练目标差异主要体现在 pose 主指标（`Geo` 大幅变化），而 contact 残差差异仍较小；source A/B 的主结论保持一致。
3. 该方向随后已经完成主线升级；当前 contract 已收缩为 `pretrain_contact`，`whitebox` 仅保留为历史 reference。
