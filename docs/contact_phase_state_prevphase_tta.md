# Contact Phase Anchor（prev_phase / TTA）现状结论与后续路线图

> Last updated: 2026-01-07  
> 关联文档：`docs/contact_meas_head_redesign_lowerbody_nohist.md`、`docs/contact_loop_closure_design.md`
> Update (2026-03-09): 当前 mainline 已移除 `contact_phase_state` 主链，并且 `train.validate.run_freerun_cycles` 不再支持 `contacts_meas_source=whitebox`。本文保留作历史 phase/TTA 研究记录；下面命令仅能在历史快照上复现，不能再作为当前执行建议。

本文档记录当前阶段（截至 2026-01-07）我们对 “contact-loop / phase anchor / TTA(learned phase advance)” 的共识：
- **现在的问题是什么**（现象与影响）
- **哪些已经验证没问题**（数据对齐/信号定义/指标一致性）
- **我们认为的根因是什么**（为什么会塌、为什么会被吸附）
- **后续怎么改**（已确定方向 + 仍需额外处理的部分）

> 注：本文不是“继续 debug 的流水账”；debug 细节只作为证据与验收手段出现（附录）。

---

## 0) TL;DR

1) **已确认 `contacts_meas_t` / GT contacts 没有错位一帧**：数据管线对齐正常（第 2 节）。
2) **现有 `contacts_plan = GRU(cond)` 在 `cond≈常量` 时不可辨识相位**：会自然塌到均值/偏置，导致 Plan 不交替（第 3 节）。
3) **改进方向已明确**：把 phase 做成“带状态的时钟”（TTA / prev_phase），并让其在推理时接受 `contacts_meas_t` 的闭环纠偏；而不是指望 `GRU(cond)` 自发振荡（第 4 节）。
4) **先钉死 “contacts_meas_t 到底来自哪里”**：`--contacts_meas_source` 才是闭环里真实使用的 meas（影响 `contacts_err` / event_clock / λ），而 `--direct_pose_meas_source` 只影响 direct head 的 phase hint（不等价）。
5) **在 d1_phaseclk 这轮上，learned/whitebox meas 在 freerun 下都不稳定**（第 5.3 节）：  
   - `contacts_meas_source=gt`：events=5，period=87（稳定，可用于验证 phase/TTA 定义与接线）  
   - `contacts_meas_source=model`（未训 meas head）：events≈0/1（几乎无 reset）  
   - `Stage0(meas-only)` 后 `contacts_meas_source=model`：events≈40–60，period mean≈7–10（过密抖动，典型 freerun OOD/drift）  
   - `contacts_meas_source=whitebox`（pose-derived）：同样会 events 过密（不等价于“deploy 外部传入的 whitebox meas”）

建议的最小输入形态（接口层）：
```
[ cond_t, contacts_meas_t, Δcontacts_meas_t, prev_phase_vec_t ]      # phase_vec = [sinφ, cosφ]
# 或
[ cond_t, contacts_meas_t, Δcontacts_meas_t, prev_TTA_t ]           # 更像 “内部状态”，通常不建议当纯外部输入
```

其中只有 `prev_phase / prev_TTA` 需要“状态定义 + 初始化策略”；`contacts_meas_t`、`Δcontacts_meas_t` 是当步可得。

---

## 1) 当前问题（现象 → 影响）

我们目前遇到的核心现象（在 freerun / multi-cycle 下最明显）：

1) **contacts_plan 不交替 / 振幅很低（塌陷到中间值）**  
   直接后果：`contacts_err = plan - meas` 退化成偏置或弱信号，闭环无法利用它去 gate/correct。

2) **time-PE 驱动力≈0**（time term 幅度小，或被缩放压没）  
   直接后果：即使提供 `time_index`，plan logits 也很难跨过 base 偏置翻符号。

3) **event_clock 在某些设置下表现为“吸附器”**（ec=on 时更难交替）  
   直接后果：time drive 被进一步抑制，plan 更稳定地塌到一个固定点。

这些现象会连锁影响 `docs/contact_loop_closure_design.md` 中的两类机制：
- **SO(3) corrector / λ fusion** 的输入侧信号质量（`contacts_err` 退化）
- **direct vs inc** 的 early 行为（Round0 只能“不被混坏”，无法“拉回”）

---

## 2) 已确认没问题的部分（证据/指标）

### 2.1 数据时间轴对齐：contacts 没有错位一帧（✅）

对齐审计输出：
- `debug_output/_tmp_teacher_debug/v1_d1_lbnohist_v1/alignment_audit/audit_alignment.md`（一次性审计，证明 data/fit/teacher 对齐）

审计结论（多 clip 一致）：
- `raw_frames_T = pair_T + 1` 且 `contact_fit = trunc(-1)`：为对齐训练 pair 截掉 raw 最后一帧（正常行为，不是 shift）
- `teacher_state_max_abs = 0.0`、`teacher_target_max_abs = 0.0`：teacher batch 的 state/target 与 npz 一致
- `teacher_pred_contact_max_abs = 0.0`：teacher_rollout 里的 `aux_inputs.contacts` 与 raw FootEvidence soft_contact_score（fit 后）逐帧一致

因此可以排除“contacts 标签与 state/pose 错位”的方向。

### 2.2 freerun 输入侧：`contacts_meas_t` 确实是当步（✅）

当 `--contacts_meas_source gt`：
- freerun 每步喂给模型的 `contacts_in_t = contacts_seq[:, t]`（不是 `t+1`）：`train/validate/run_freerun_cycles.py:1419`
- 模型 forward 中该 `contacts` 被作为 `contacts_meas` 的 override：`train/models.py:1610`
- `contacts_err = contacts_plan - contacts_meas` 使用同一步的 `contacts_meas_t`：`train/models.py:1651`

### 2.2.1 JSON 自检：快速排除 off-by-one（✅）

当你用 `--contacts_meas_source gt`（且 `--log_contacts`）跑 freerun JSON 时：
- `ContactMeasGtAbsMean` 应该≈0
- 且 `ContactMeasPerC` 应更接近 `ContactGTPerC` 而不是 `ContactGTNextPerC`

如果发生“错位一帧”，最常见表现是：`ContactMeasPerC` 反而更像 `ContactGTNextPerC`。

这些字段来自 `train/validate/run_freerun_cycles.py` 每步同时记录 `gt_contacts(t)` 与 `gt_contacts(t+1)`（`ContactGTPerC` / `ContactGTNextPerC`），因此非常适合做快速 sanity check。

### 2.2.2 先钉死 “deploy 在用哪种 contacts_meas” （✅ 必做）

不要凭感觉推断。用 JSON 里的 `ContactsMeasSourceApplied` 直接读出本次 rollout 每一步到底喂给模型的 `contacts_meas` 来源：
- `gt`：来自 teacher batch（oracle；用于对齐/定义验收或模拟外部 override）
- `whitebox`：由 `run_freerun_cycles` 从 **当前预测 pose** 推导的白盒 contacts（会受 drift 影响；不是“部署外部白盒输入”）
- `model`：learned meas head 输出（若没训练会接近常量）

注意一个容易误判的情况：当 `contact_plan_init_mode=learnable+obs` 且 `plan_z` 在 `t=0` 还没 init 时，`ContactsMeasSourceApplied` 可能在第 0 步短暂显示为 `whitebox_init`（仅用于 plan_z 初始化，不代表整个序列都在用 whitebox）。

快速查看：
```bash
python - <<'PY'
import json
path='debug_output/posttrain_d1_phaseclk/verify_final_contacts_meas_gt/Walk_F_freerun_cycles.json'
j=json.load(open(path))
src=[m.get('ContactsMeasSourceApplied') for m in j.get('metrics_per_step',[])]
print('counts', {s:src.count(s) for s in sorted(set(src))})
print('first10', src[:10])
PY
```

---

### 2.3 phase/TTA 的“定义一致性”已通过（GT 上）（✅）

我们已经用 `run_freerun_cycles` 的 JSON 做了事件检测与 TTA/phase 导出，并验证：
- event 周期稳定（Walk_F 上 L/R touchdown 间隔恒定）
- TTA 在两次 reset 之间严格每步 -1（无 off-by-one）

使用的验收脚本是 `tools/diagnose_phase_tta_inputs.py`（从 `Contact*PerC` 做阈值过零检测），重点关注两行：
- `period[min/mean/max]`：周期是否稳定（不稳通常是 event 定义或 contacts 噪声问题）
- `TTA_consistency_bad/total`：若非 0，基本就是 TTA 定义/对齐有 bug（或 contacts 抖到频繁过阈）

工具与命令见附录 A（并包含一个 Walk_F 的示例结论）。

### 2.4 输入可观测性审计：`cond` 在 Walk_F 上近常量（✅）

以 `validate/teacher_batches/Walk_F_teacher.json` 为例（teacher.cond 形状为 `(T=87, Dc=7)`）：
- 绝大多数维度 std≈0（常量）
- 只有 1 个维度有明显变化（std≈0.167）

这解释了为什么仅靠 `GRU(cond)` 很难学出稳定 L/R 交替：同一个 `cond` 对应多种相位轨迹，问题不可辨识。

复现命令：
```bash
python - <<'PY'
import json, numpy as np
j=json.load(open('validate/teacher_batches/Walk_F_teacher.json'))
cond=np.array(j['teacher']['cond'],dtype=np.float32)
std=cond.std(0)
print('cond shape',cond.shape)
print('std',std)
print('top std idx',np.argsort(-std)[:10])
PY
```

### 2.5 time-PE 量纲审计：`contact_plan_time_head` 在当前 ckpt 幅度偏小（✅）

对当前 ckpt：
- `models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth`

检查发现 `contact_plan_time_head.weight` 的范数与 `absmax` 都比较小（time-drive 天花板低），这与 “需要把 `--contact_plan_time_bias_scale` 放大到 10~20 才能看到明显交替” 的现象一致。

复现命令：
```bash
python - <<'PY'
import torch
ckpt='models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth'
obj=torch.load(ckpt, map_location='cpu')
state=None
if isinstance(obj, dict):
    for k in ('state_dict','model','model_state','model_state_dict'):
        if k in obj and isinstance(obj[k], dict):
            state=obj[k]; break
    if state is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
        state=obj
else:
    state=obj
w=state.get('contact_plan_time_head.weight')
b=state.get('contact_plan_time_head.bias')
print('w',tuple(w.shape), 'norm', float(w.norm()), 'absmax', float(w.abs().max()))
print('b',tuple(b.shape), 'norm', float(b.norm()), 'absmax', float(b.abs().max()))
PY
```

---

## 3) 已定位的根因（为什么会这样）

### 3.1 `cond≈常量` 时：`GRU(cond)` 的相位不可辨识（核心根因）

在 walk/locomotion 这类片段上，如果 `cond` 基本不变化，那么：
- `contacts_plan = GRU(cond)` 无法从输入区分“左先/右先/当前相位”
- 学习上最稳的解就是输出边际均值（或带偏置的常数）→ plan 不交替

这不是“数据不够”，而是 **同一个 cond 对应多个相位轨迹** 的不可辨识问题。

### 3.2 time-PE / event_clock 为什么容易“帮倒忙”

在当前实现中：
- time-PE 是一个额外 logit term（幅度可能很小）
- event_clock 开启后 time term 还可能被 `lambda_corr` 进一步缩放（相当于“观测不可信就别用 time”）

若 `lambda_corr` 长期偏小/偏保守，就会出现：
- ec=off：time 有摆幅，但不足以翻符号
- ec=on：time 被压到更小，plan 更像固定点（“吸附器”效果）

> 经验上还需要额外检查：当前 ckpt 的 `contact_plan_time_head` 权重是否本来就很小（导致 time-drive 天花板低）。这属于“量纲/初始化/训练信号”问题，而非数据对齐问题。

---

## 4) 已确定的改进方向（TTA / learned phase advance）

### 4.1 目标：把“相位/节奏”变成显式状态，而不是隐式振荡

我们不再依赖 `GRU(cond)` 自发产生交替节奏，而是采用“带状态的时钟”：
- 状态：`prev_phase_vec`（推荐）或 `TTA`（更像内部预测量）
- 推进：由网络预测 `Δφ / freq / ΔTTA`（learned phase advance）
- 纠偏：由 `contacts_meas_t` 与 `Δcontacts_meas_t` 提供闭环创新

### 4.2 最小接口（工程上先把线接对）

推荐先落地 `prev_phase_vec`（sin/cos）：
```
Inputs:
  cond_t
  contacts_meas_t
  dcontacts_meas_t
  prev_phase_vec_t    # [sinφ_{t-1}, cosφ_{t-1}] per foot（或 shared）

Outputs:
  contacts_plan_t
  (optional) phase_delta/freq, tta_pred
```

经验建议：
- `prev_phase_vec` 必须进入 plan 分支主路径（决定 logits/plan），而不是仅作为 trunk 辅助特征
  - 实现上建议用 **logits residual**：`logits = logits_base(GRU(cond)) + logits_phase(prev_phase_vec)`（见 `train/models.py`）。
- `TTA` 更适合作为“内部状态/输出监督”，不建议一开始就把 “prev_TTA” 当外部输入（它在线不可直接观测）

---

## 5) 仍需额外处理/仍未验收的部分（下一阶段）

下面这些不是“数据对齐问题”，而是下一阶段必须补齐的工程/建模点：

### 5.1 `t=0` 初始化策略（未定，必须做）

`prev_phase` 在 `t=0` 怎么来：
- zeros / learnable init
- obs-init（用 `contacts_meas_0` / `Δmeas_0` / `lr_diff_0` 等）  
这一项会直接决定 Round0 early 行为是否稳定（与 `docs/contact_loop_closure_design.md` 的 “direct early 精度” 强耦合）。

### 5.2 “弱事件/无事件”片段（非周期动作）的 fallback（未定）

例如攻击/受击/急停：
- soft contact 可能长期平台（双脚接触都接近 1）
- event 不发生或极少发生 → phase/TTA 需要 fallback（保持、或靠 learned freq 推进）

这会决定“泛化到非 locomotion 动作”是否可靠。

### 5.3 用 `contacts_meas_source=model` 时，meas 的事件稳定性（当前不通过）

本文档第 2 节里“稳定性”主要来自 `contacts_meas_source=gt`（oracle override）。  
在 d1_phaseclk 的实测里，**learned/whitebox meas 在 freerun 下目前都不满足 event reset 的稳定性要求**：

- final ckpt（meas head 未训，输出≈0.5）：`contacts_meas_source=model` → events≈0/1（几乎无 reset）  
  复现 JSON：`debug_output/_tmp_prev_phase_check_d1_phaseclk_meas_model/Walk_F_freerun_cycles.json`
- Stage0(meas-only) 从 final ckpt 训练 meas head 后：`contacts_meas_source=model` → events≈40–60，period mean≈7–10（过密抖动）  
  复现 JSON：  
  - `debug_output/posttrain_d1_phaseclk/verify_stage0_meas_only/Walk_F_freerun_cycles.json`  
  - `debug_output/posttrain_d1_phaseclk/verify_stage0_meas_only_e5/Walk_F_freerun_cycles.json`
- `contacts_meas_source=whitebox`（由预测 pose 推导）同样 events 过密：  
  复现 JSON：`debug_output/posttrain_d1_phaseclk/verify_final_contacts_meas_whitebox/Walk_F_freerun_cycles.json`
- `contacts_meas_source=gt`（模拟外部 override）是稳定的：events=5，period=87  
  复现 JSON：`debug_output/posttrain_d1_phaseclk/verify_final_contacts_meas_gt/Walk_F_freerun_cycles.json`

因此：
- **如果你后续验证“phase/TTA 定义 + 接线 + 闭环机制”**，可以直接用 `--contacts_meas_source gt` 先把 meas 这条不确定性摘掉（这是“上限/外部 meas 假设”，不是“learned meas 已经可用”的证明）。
- **如果你最终部署要依赖 learned meas head**，那当前问题本质是 freerun OOD/drift：teacher 下校准 OK，但 freerun 里 drift 把 meas 输入分布推飞，导致 “无 crossing” 或 “过密 crossing” 两种失败模式。

#### 5.3.1 Debug playbook：区分“meas head 校准” vs “freerun drift/OOD”

1) teacher 下看 meas：  
   - 例：`debug_output/_tmp_teacher_meas_d1_phaseclk/Walk_F_teacher_pred.json` 跑 `--source meas`，检查是否至少有合理的 crossing（单周期只会有 1 次 touchdown）。
2) freerun 下看 meas：  
   - 对比 `debug_output/_tmp_prev_phase_check_d1_phaseclk_meas_model/Walk_F_freerun_cycles.json`（无事件）与 Stage0 后的 freerun（事件过密），判断是 “未训练” 还是 “OOD 抖动”。
3) 钉死来源：确认 `ContactsMeasSourceApplied`（第 2.2.2 节）。  
   - 很多“以为在用外部/whitebox”其实在用 model 或只在 t=0 用了 `whitebox_init`。
4) 若结论是 OOD/drift：下一步优先 debug “为什么 pose drift → meas 被吸附/抖动”，而不是继续加大 `contact_meas_weight`。  
   - 实用做法：用 scheduled sampling / freerun-like rollout 去训练 meas head（让它见过 drift 分布），或给 meas head 加 history（短时卷积/GRU）与 hysteresis（阈值滞回 + min-interval）。

**实现提示（已接线）**：`train/posttrain.py` 新增 `--contact_meas_rollout true`，当 `train_contact_meas_only` 时用闭环 rollout 监督 meas（而不是 teacher forcing），用于缓解 freerun OOD。
```bash
PYTHONPATH=. python -m train.posttrain \
  --ckpt_in <FINAL_OR_STAGE0_CKPT.pth> \
  --out_dir debug_output/_tmp_posttrain_meas_rollout \
  --train_so3_corrector false --train_contact_plan_init false --train_contact_plan false \
  --train_direct_pose false --train_lambda_head false \
  --train_contact_meas true --contact_meas_rollout true --contact_meas_weight 1.0 \
  --seq_len 87 --rollout_steps 0 --rollout_cycles 2 \
  --epochs 1 --steps_per_epoch 200
```
> 注：当 `rollout_cycles>1` 时默认 `rollout_include_boundary=true`；wrap 边界步默认 `lambda_boundary_weight=0`（只用于 state 更新，不做监督），避免非严格周期数据在边界处产生错误监督信号。

---

## 6) 验收指标清单（哪些已绿、哪些还需补）

### 6.1 Green（已确认）

- 数据对齐：`audit_alignment.md` 中 `teacher_*_max_abs=0`、`teacher_pred_contact_max_abs=0`  
  → `debug_output/_tmp_teacher_debug/v1_d1_lbnohist_v1/alignment_audit/audit_alignment.md`
- freerun 当步对齐（meas=gt）：`ContactMeasPerC == ContactGTPerC`，`ContactMeasGtAbsMean≈0`  
  → 运行 `--contacts_meas_source gt --log_contacts` 后检查 JSON
- phase/TTA 定义一致性（GT）：period 稳定 + `TTA_consistency_bad/total=0/...`  
  → `tools/diagnose_phase_tta_inputs.py`（附录 A；示例 JSON 见 `debug_output/_tmp_prev_phase_check_d1_phaseclk_gt`）
- `cond` 可观测性：Walk_F teacher.cond 多维度 std≈0（近常量）  
  → 第 2.4 节命令

### 6.2 Yellow（需要额外处理/验收）

- meas=deploy 形态的事件稳定性（`--contacts_meas_source model` / `whitebox`）  
  - d1_phaseclk 实测：model/whitebox 在 freerun 下都不稳定（第 5.3 节），需要先解决 OOD/drift 才能作为 event reset 依赖
- `t=0` 初始化策略对 Round0 的影响（learnable vs obs-init）  
  - 需要明确验收：Round0 early 的 direct/inc/λ 关系是否被改善（见 `docs/contact_loop_closure_design.md`）
- 非周期动作的 fallback（无事件/弱事件段）  
  - 需要定义：保持 / learned freq 推进 / 置信度下降策略

---

## 附录 A：验收/复现命令（用于“确认没问题”与后续回归）

### A.1 生成带 GT meas 的 freerun JSON（用于定义/对齐验证）
```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth \
  --rounds 5 \
  --contacts_meas_source gt \
  --event_clock off \
  --log_contacts \
  --out debug_output/_tmp_prev_phase_check_d1_phaseclk_gt \
  --force
```

### A.2 从 JSON 导出并验证 phase/TTA（定义一致性）
```bash
python tools/diagnose_phase_tta_inputs.py \
  --json debug_output/_tmp_prev_phase_check_d1_phaseclk_gt/Walk_F_freerun_cycles.json \
  --source gt \
  --event-kind touchdown \
  --thr 0.5
```

重点看脚本输出中的两行：
- `period[min/mean/max]`（周期稳定性）
- `TTA_consistency_bad/total`（TTA 定义一致性）

示例：在 `debug_output/_tmp_prev_phase_check_d1_phaseclk_gt/Walk_F_freerun_cycles.json` 上，
`touchdown` 事件间隔为 87 帧且 `TTA_consistency_bad/total=0/...`，说明当前时间轴下没有 off-by-one。

### A.3 对齐审计（已有结果位置）

- `debug_output/_tmp_teacher_debug/v1_d1_lbnohist_v1/alignment_audit/audit_alignment.md`

### A.4 钉死 `contacts_meas_source`（gt / model / whitebox）并检查事件周期

```bash
# 1) oracle / 外部 override（稳定）：events=5, period=87
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth \
  --rounds 5 \
  --contacts_meas_source gt \
  --log_contacts \
  --out debug_output/posttrain_d1_phaseclk/verify_final_contacts_meas_gt \
  --force

# 2) whitebox（由预测 pose 推导；通常不稳定）
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_phaseclk_lambda_cycles2_after_direct_pose.pth \
  --rounds 5 \
  --contacts_meas_source whitebox \
  --log_contacts \
  --out debug_output/posttrain_d1_phaseclk/verify_final_contacts_meas_whitebox \
  --force
```

然后用诊断脚本看 `events/period`：
```bash
python tools/diagnose_phase_tta_inputs.py \
  --json debug_output/posttrain_d1_phaseclk/verify_final_contacts_meas_gt/Walk_F_freerun_cycles.json \
  --source meas --event-kind touchdown --thr 0.5

python tools/diagnose_phase_tta_inputs.py \
  --json debug_output/posttrain_d1_phaseclk/verify_final_contacts_meas_whitebox/Walk_F_freerun_cycles.json \
  --source meas --event-kind touchdown --thr 0.5
```
