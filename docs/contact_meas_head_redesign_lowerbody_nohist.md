## ContactMeas Head 重设计对接文档（v1）

> Status (2026-05-04): archived design / debugging note. 本文记录 lower-body + no-hist meas head 的历史设计和调试结论，不是 current implementation contract。当前可执行 CLI 选项以 `train/validate/run_freerun_cycles.py` parser 为准；`ContactMeasHeadLowerBodyNoHistV1` 与 `LOWER_BODY_INDICES_V1` 已于 2026-04-17 从 `train/models.py` 删除。
> Historical update (2026-01): 当时默认切到 v1（lower-body + no-hist）meas head；本文保留 legacy head 的问题复盘，并补充 freerun/OOD 与 “contacts_meas 来源钉死” 的调试/验证流程。
> Update (2026-01-08): 目前已确认 **teacher 上 learned meas 近乎完美，但 freerun 下 learned/whitebox meas 都可能失稳**，表现为两类互斥的失败模式：  
> 1) **阈值抖动（event 过密）**：`period_min` 可到 2 帧，导致 phase reset/event-clock 被“抖”坏；  
> 2) **event starvation（几乎无事件）**：meas 被推到阈值同侧的常量附近，phase reset 失效。  
> 注意：简单的 closed-loop rollout BCE 监督（不做对齐/约束）可能把模型从 (1) 推向 (2)（看似 flips 少了，但事件直接消失，且 meas-vs-GT 误差不一定下降）。
> Update (2026-05-04): `train.validate.run_freerun_cycles` 当前仍支持 `contacts_meas_source=whitebox`，但本文的 whitebox 结论只作为历史诊断记录；不要把本文当作 current source policy。

### 背景
历史 `contact_meas_head` 的设计目标是提供一个**便宜/可微/不依赖物理引擎**的 contact measurement（`contacts_meas`），并在闭环中形成残差：

> `contacts_err = contacts_plan - contacts_meas`

该残差后续会影响 rollout 可靠性/纠错（如 `lambda_reliability_mode=contacts_err` 等），因此 `contacts_meas` 的**因果性与跨动作泛化**比“在某些 clip 上拟合得很像 GT”更重要。

### Legacy 实现（已废弃：pose_hist-based head）
- meas head 结构：`LayerNorm -> Linear -> ReLU -> Dropout -> Linear -> sigmoid`
- meas head 输入：`[pose_hist, angvel]`
  - `pose_hist`：全身姿态历史窗口（默认 `pose_hist_len=3`），来自数据集预处理（纯过去窗口）
  - `angvel`：骨骼角速度（来自 state 或 seq，取决于 rollout 配置）
- 相关代码入口：
  - 模型定义/forward：`train/models.py`（`contact_meas_head` 与 `contacts_err` 计算）
  - rollout 输入来源：`train/training_MPL.py`（`pose_hist` buffer/seq、`angvel` state/seq）
  - teacher 调试脚本：`train/validate/run_teacher_rollout.py`

> 说明：legacy `pose_hist`-based meas head 已不再支持；如需使用请回退代码/重新训练。下节 v1 也是历史设计记录，不是 current 默认实现。

### v1 实现（历史设计：lower-body + no-hist）
- meas head 结构：`LN_pose + LN_angvel -> concat -> MLP -> logits`（prob 用 `sigmoid(logits)`）
- meas head 输入：`[pose_lower(t), angvel_lower(t)]`（仅当前帧 `state_t`，不使用 `pose_hist`）
- 对接位置：
  - 历史 head 定义：`ContactMeasHeadLowerBodyNoHistV1`（已从 `train/models.py` 删除）
  - 历史 lower-body 关节子集：`LOWER_BODY_INDICES_V1`（已从 `train/models.py` 删除）
  - 推理/验证当前可用 `--contacts_meas_source {model,whitebox,gt,zero,pretrain_contact}` 固定 meas 来源：`train/validate/run_freerun_cycles.py`

### 已定位的问题（机制层结论）
历史调试确认：该 head 在训练阶段就学到了“姿态模式 ↔ 步态阶段”的 **spurious correlation**，且该相关性在不同动作/转向下会失效甚至反向，导致推理端出现 clip-dependent 的系统性问题。

核心表现（已通过工具链与归因确认）：
1) **上肢/躯干作为 phase proxy**
   - 归因显示：falling 边沿时刻 top contributors 多为上肢/手指/躯干骨骼，foot bones 占比低。
   - arms-only 仍可把 contact logit 推回 contact 区域（说明 head 的主要判别路径不是 foot/leg）。

2) **R-fall “长尾/滞回”不是简单的“历史帧确认策略”**
   - keep_last/replicate 等 ablation 主要影响 `pred@GT_fall` 的偏置，但对 `dt<=mid_th`（time-to-threshold）影响不稳定；
   - 更像是：在 GT_fall 时刻，输入已经把 logit 推到 contact 区域，随后缓慢衰减（而不是靠 block 间差分做确认）。

3) **推理端 shift/mask 只能作为 workaround**
   - `pose_hist_time_shift≈-30` 能“修复”部分 clip 的统计，但强依赖边界 padding / clip 分布；
   - 对 `Walk_R_To_L` 等反例，全域 shift 无效（机制上说明 head 的决策边界已错，而非单纯对齐误差）。

结论：当前问题不是“推理端对齐/滤波没做好”，而是 **meas head 学错了输入依赖与判别依据**；推理端 ablation 只会把问题从一个子集转移到另一个子集。

---

## 本轮改进目标
用最小改动把问题从“不断打转的 workaround 调参”转为“结构上禁止 spurious proxy 的正确建模”，并尽量把后续问题拆解为独立维度（对齐 vs 时序一致）。

本轮只做三件事：
1) **去掉 pose_hist 输入**
2) **输入下肢化（Lower-body Pose + Lower-body AngVel）**
3) **分支 LayerNorm（避免 LN coupling / mask 引起的重标定副作用）**

明确不做：
- 不引入 Schmitt/EMA 等“迟滞/平滑”（避免用滤波掩盖对齐/因果性问题；也避免引入不可接受的推理延迟）
- 不引入 whitebox/FK 特征作为推理输入（避免 freerun 中“自己预测的函数”引发 cascading）

---

## 设计方案（v1）

### 1) meas head 输入：Lower-body Pose + Lower-body AngVel（t 时刻）
**输入只来自当前时刻 `state_t`（因果测量）**，不使用 `y_raw_local` / “下一帧输出”。

- `pose_lower(t)`：从 `state_t` 的 `BoneRotations6D`（或等价旋转表征）抽取下肢骨骼子集
- `angvel_lower(t)`：从 `state_t` 的 `BoneAngularVelocities` 抽取下肢骨骼子集

下肢骨骼索引（本项目默认 46-bone skeleton 的 `bone_names` 顺序；可参考 `raw_data/Walk_F.json` / `*.npz` 的 `bone_names`）：

```python
LOWER_BODY_INDICES_V1 = [0, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
# pelvis + thigh/calf/twist/foot/ball (L+R), 共 15 个关节
```

说明：
- 下肢化的目标是**结构性禁止** upper-body phase proxy（而不是靠 loss/penalty 去“劝退”捷径）。
- 选择 `state_t`（而非 `y_raw_local`）的目标是避免引入 1-step 对齐偏差，把 meas 明确成“当前系统状态的测量”，使 `contacts_err` 语义清晰。

### 2) 去 pose_hist
`pose_hist`（全身多帧历史）是 spurious correlation 的主要入口，同时也把时间对齐问题放大为“需要 lookahead”的假象。

本轮直接移除它作为 meas 输入：
- meas head 不再接收 `pose_history`
- 时序信息由主干/GRU 的状态演化提供（如果仍不足，再单独评估是否需要显式的 `prev_meas` 记忆，而不是回退到全身 pose_hist）

### 3) 分支 LayerNorm（Branch LN）
为避免「mask/置零 → LN 重标定 → clip 反例」这类分布效应：

```
z_pose = LN_pose(pose_lower)
z_w    = LN_angvel(angvel_lower)
meas_in = concat([z_pose, z_w])
logits  = MLP(meas_in)
contacts_meas = sigmoid(logits)
```

---

## 预期收益
1) **切断 upper-body proxy 路径**：从输入空间结构上禁止 spurious gait-phase correlation。
2) **减少 clip-dependent workaround**：shift/mask 不再是“必要步骤”，而变为可选的诊断手段。
3) **把问题拆解成可独立验证的两类**：
   - 若仍有 lag：更可能是监督/目标定义（contacts 本身的时域定义）或闭环使用方式问题；
   - 若出现抖动：再单独评估是否需要迟滞/平滑（且只在使用侧引入，避免影响对齐判断）。

---

## 风险与注意事项
1) **需要重训**：输入维度与依赖关系变化，无法复用旧 head 权重。
2) **rising/on 可能变钝**：纯 lower-body angvel 对 landing/on 的可分性可能不足；因此保留 lower-body pose 作为补充信息。
3) **闭环安全性**：在新 head 未收敛前，`contacts_err` 可能变噪，建议在训练/早期评估阶段对 `contacts_err` 的下游使用做保护（例如仅用于 logging/reliability，不强驱动纠错）。

---

## 训练/校准（Stage0：meas-only posttrain）
当 ckpt 启用了依赖 `contacts_meas` 的模块（如 phase clock reset / Event-Clock / `contacts_err`），但训练阶段没有监督 meas head（例如 `train_contact_meas=false` 或 `contact_meas_weight=0`），常见现象是：
- `contacts_meas≈0.5`（`sigmoid(0)`），阈值 0.5 附近几乎无 crossing
- `contacts_err` 语义崩溃，导致下游 reliability/纠错失效或方向混乱

此时建议先补齐一个 **只训练 meas head** 的 posttrain（不动 λ/so3/direct/plan 等其他头）：

```bash
PYTHONPATH=. python -m train.posttrain \
  --config config/posttrain_lambda_fusion.json \
  --ckpt_in <CKPT.pth> \
  --out_dir <OUT_DIR> \
  --run_name posttrain_<tag>_meas_only \
  --train_contact_meas true --contact_meas_weight 1.0 \
  --train_lambda_head false --train_so3_corrector false --train_direct_pose false \
  --train_contact_plan_init false --train_contact_plan false
```

> 注意：meas-only posttrain 是 teacher-supervised（对齐 GT contacts），“teacher 很准”并不保证 freerun 下稳定；若出现 “teacher OK, freerun 崩”，按文末 Debug Playbook 判断是否 drift/OOD。

## 验证计划（不引入推理延迟）
建议分三层验证，尽量复用现有工具链：

### A) Teacher-rollout（对齐与边沿质量）
对每个 clip（尤其 `Walk_L_To_R`, `Walk_R_To_L`）统计：
- falling：`pred@GT_fall`、`dt<=mid_th`（你现有的 `mid_th=0.55, window=30`）
- rising：对称的 `pred@GT_rise`、`dt>=mid_th`（同窗口）
- regime：`P(L>R | Lsup)`、`P(R>L | Rsup)`（避免 L/R 混淆复发）

工具：
- `tools/analyze_contact_meas_head.py`
- `tools/analyze_contact_meas_lag.py`
- `tools/plot_contact_meas_event_curves.py`

### B) Freerun（闭环稳定性）
关注两类信号：
- `contacts_err` 的幅度/稳定性是否合理（避免驱动纠错乱修）
- 关键下游模块（例如 SO(3) corrector / lambda reliability）是否出现异常放大

### C) 对照实验（定位“剩余问题属于哪一类”）
- 若 teacher 已无长尾但 freerun 有：更像闭环/漂移耦合问题（与 meas 设计无关）
- 若 teacher 仍长尾：更像监督定义/时域标签本身滞后（需要回到 contacts 的生成/对齐）

### D) Deploy/验证：先“钉死” contacts_meas 来源（强烈建议）
`contacts_meas` 在模型里有明确的优先级：**外部 override（`contacts_input`）> learned meas head > zeros**。
因此你必须先确认 deploy/验证到底走哪条路径，否则“meas 不稳定/无 crossing”这类问题会被接线差异掩盖。

1) **区分两种完全不同的“外部 meas”**
- **真实外部 meas（deploy 期望）**：来自传感器/上游模块，**不依赖模型预测 pose**，理论上可稳定提供 event reset。
- **whitebox meas**：由预测 pose 通过 FK/高度/速度规则算出的 contact 分数；在 freerun 下会随 drift 放大，通常并不稳定（这不是 learned meas 的锅）。

2) **验证脚本里固定 meas 来源**
`run_freerun_cycles` 支持用参数固定 `contacts_meas`（影响 Event-Clock / phase clock reset / contacts_err / λ 等）：

```bash
PYTHONPATH=. python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <CKPT.pth> \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --rounds 5 --time-index-mode auto --depth 3 \
  --lambda_fusion_apply --so3_corr_apply \
  # contacts_meas_source: model|whitebox|gt|zero|pretrain_contact
  --contacts_meas_source model \
  --log_contacts \
  --out debug_output/<out_dir> --force
```

输出 JSON 里检查 `metrics_per_step[*].ContactsMeasSourceApplied`：
- `model`：learned meas head 生效（通常也会有 `ContactMeasLogitsPerC`）
- `gt`：oracle 外部 meas（只用于离线验收/上限对照，deploy 不存在）
- `pretrain_contact`：冻结的 pretrain contact head 生效（当前主线外部 contacts 路由）
- `whitebox`：运行时 whitebox contacts 分支；适合诊断，不代表 deploy contract
- `whitebox_init`：历史/特定 fallback 标记，读旧 JSON 时按当时配置解释

> 注意：`--direct_pose_meas_source` 只影响 direct head 的 meas hint（`direct_pose_meas_override`），**不等价于** `contacts_meas_source`（后者才驱动 phase reset/contacts_err）。

3) **快速 sanity：用 GT 作为“外部 meas 上限”**
如果 `--contacts_meas_source gt` 事件周期正常，但 `--contacts_meas_source model`/`pretrain_contact` 不正常，
基本就能把问题定位到“learned meas freerun OOD”或“frozen pretrain contact route under drift/OOD 不稳”，而不是 phase/TTA 定义或 off-by-one。

---

## 后续可选增强（本轮不做）
若 v1 改动后仍存在明确问题，再按“最小增量”补：
1) 只在使用侧引入 Schmitt/EMA（只用于 gating/reliability，不改变 meas 概率本体）
2) 训练侧增强：upper-body dropout/randomization（即使输入已下肢化，也可防止模型走 pelvis/spine proxy）
3) whitebox/FK 仅作为离线评估基准（或训练 regularizer，但需要处理 GT pose vs pred pose 的分布差）

---

## Debug Playbook：learned meas “teacher OK, freerun 崩”
当你发现：
- teacher（`run_teacher_rollout`）上 `contacts_meas` 很像 GT
- 但 freerun（`run_freerun_cycles`）上 `contacts_meas` 抖动 crossing / 错相 / 变常数

这通常意味着：**闭环 drift 把 meas head 的输入分布推到 OOD 区域**，而不是 “head 没训练/结构错”。

建议按下面顺序做最小诊断：

### 1) teacher 侧：确认 head 本身学对了（对齐/滞后）
```bash
PYTHONPATH=. python -m train.validate.run_teacher_rollout \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model <CKPT.pth> --depth 3 \
  --bundle raw_data/processed_data/norm_template.json \
  --pretrain-template models/pretrain_template.json \
  --encoder-bundle models/motion_encoder_equiv_stageA.pt \
  --npz-root raw_data/processed_data \
  --out debug_output/_tmp_teacher_meas --force
python tools/analyze_contact_meas_head.py --json debug_output/_tmp_teacher_meas/Walk_F_teacher_pred.json
python tools/analyze_contact_meas_lag.py  --json debug_output/_tmp_teacher_meas/Walk_F_teacher_pred.json --max-lag 30
```

### 2) freerun 侧：固定 meas 来源为 model，并观察误差与 crossing
- `metrics_per_step[*].ContactMeasGtAbsMean`：meas vs GT 的平均 abs（越大越像 OOD）
- `tools/diagnose_phase_tta_inputs.py --source meas`：events 数量/period 是否异常

### 3) 判定是不是“阈值抖动”还是“输入 OOD”
`contact_phase_state_event_*` 这组 phase-state reset knobs 已从主线移除，不再作为当前执行建议。

当前口径下，若要区分 crossing jitter vs 输入 OOD，应直接看：
- `metrics_per_step[*].ContactMeasGtAbsMean`
- `tools/diagnose_phase_tta_inputs.py --source meas`

如果这两类统计都稳定，却仍出现周期错乱，再回到离线问题单里看历史 phase-reset 诊断结论。

### 4) 定位 drift 是否来自跨 cycle 累积
用 freerun 的 ablation 把“跨 cycle carry”切断：
- `--multicycle-sync-state-on-cycle-start`：每个 cycle 起点用 teacher state 同步，避免跨周期 carry
- `--multicycle-reset-plan-z-on-cycle-start`：每个 cycle 起点把 plan_z/phase_z 置空，隔离 GRU 状态累积

若这些 ablation 能显著改善 learned meas，基本可判定是 drift/OOD；此时应该把精力放在：
- deploy 期使用真实外部 meas override（推荐）
- 或为 learned meas 做 freerun-robust（需要引入闭环分布/扰动训练，而不是继续加 teacher BCE）
