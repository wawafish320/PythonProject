# [2026-04-24] `train/models.py` `forward(...)` 单文件 de-maze 计划（pre-Phase-E）

Date: 2026-04-24  
Status: Draft  
Scope: `train/models.py:3997` `EventMotionModel.forward(...)`（当前阶段**只做单文件内去迷宫化，不拆新文件**）  
Goal: 在**不改变训练/推理语义、output key、checkpoint contract、默认超参行为**前提下，把 `forward(...)` 从 feature-maze 收成 single-file stage graph，降低后续 agent / 人类修改时对隐含耦合的猜测成本。  
Non-goal:
- 不按 posttrain `70a / 70R / 71 / 72 / lambda` stage 名拆代码。
- 不在本轮进入 Phase E 跨文件迁移。
- 不改核心算法 / 数学定义 / loss 数值逻辑。
- 不把 `forward(...)` 再切成一串薄 helper 网络。
- 不引入 compat shim、warning-only debt、silent fallback、kwargs-heavy mega-helper、15-tuple return、dataclass-return 雨。

关联参考：

- 主路线图：`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`
- cleanup inventory：`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`
- posttrain canonical mental model：`docs/posttrain_pipeline.md`
- removal / fail-fast 准则：`docs/removal_policy.md`

---

## 0) 必须遵守的纪律（本计划受 `docs/removal_policy.md` 约束）

本计划虽然不是“删除分支”专项，但**任何 de-maze / pre-split 收口动作**，只要涉及分支收窄、入口改写、runtime attr 不再继续向下游传播，仍必须遵守 `docs/removal_policy.md`。

本计划中的不可协商项：

- **不得新增** silent fallback、`.get(..., .get(...))`、`warnings.warn(...)`、ckpt key silent rename、`try/except Exception: fallback`。
- **若 de-maze 顺手退休某条支路**，必须在第一次读取入口 fail-fast raise，而不是“先整理结构，行为以后再说”。
- **不得把 posttrain stage 名直接当代码边界**。`docs/posttrain_pipeline.md:95` 已明确：posttrain 的 canonical mental model 是 `Stage6-StepC handoff` 被 downstream 链消费，不是“Stage6 一段、Stage7 一段”的代码 ownership。
- **允许的提纯方式**：single-file 内 stage shell、nested def、集中 writer、显式 read-set / write-set。
- **不允许的提纯方式**：把主 branch 的迷宫拆成跨多 helper / 多文件的调用迷宫。

建议在后续每次实际落代码前继续 grep touched diff，确认未新增：

```bash
rg -n "\.get\(.*\.get\(|warnings\.warn\(|state_dict\[.*\]\s*=\s*state_dict\.pop\(|# .* compat|# .* legacy"
```

---

## 1) 一页版结论

当前 `forward(...)` 的主要问题，不是“单文件太大”本身，而是：

1. **主 branch 同时承担 stage ordering 与 feature branching**  
   顶层既在决定先后顺序，又在展开 `event_clock on/off`、`side_routing/non-side`、`leg_mode`、`direct_pose_meas_mode` 等分支。
2. **很多 local state 的 owner 不显式**  
   如 `plan_z_next`、`plan_feat_for_inject`、`phase_z_in_direct`、`leg_side_cue_in`、`soft_period`、`e_t`、`direct_leg_*` 等，跨数百行被生产、变形、最终写回。
3. **对于 agent 来说，修改成本来自“隐含推测”而不是“代码量”**  
   改 contact-plan 时要反推 direct-pose 读了什么；改 direct-pose 时要猜 contact-plan finalize 之后哪些字段才稳定。

因此，本计划的目标不是“继续 helper 化”，而是：

- **先在当前文件里把 `forward(...)` 改成 stage graph**
- **再考虑 Phase E 的 mechanical move**

一句话：

> 先 single-file de-maze，再 cross-file split。  
> 先消灭隐含 ownership，再做物理迁移。

---

## 2) 为什么“按形态拆”比“按 stage 名拆”更自然

`docs/posttrain_pipeline.md` 约束的是 **runbook / validation boundary**，不是 `EventMotionModel.forward(...)` 的代码 ownership。

如果直接按 posttrain stage 名切代码，例如：

- `stage6_forward.py`
- `stage70a_forward.py`
- `stage71_forward.py`

那么结果大概率是：

- runtime branch 仍旧存在，只是换了文件位置；
- 同一套 `EventMotionModel` forward 行为被人为复制成多个 stage 视角；
- 主 branch 表面变短，但真正的控制流被转移到“哪个 stage 文件该改”的认知成本里。

当前更自然的边界不是 stage family，而是 **runtime 形态**：

1. **Boundary / runtime normalization**
2. **Contact-plan state machine**
3. **Motion core**
4. **Direct-pose state machine**
5. **Aux output writers**

这五块才是 `forward(...)` 真实运行时的形态边界。

---

## 3) 当前 baseline（2026-04-24 code anchors）

当前 `forward(...)` 的已知边界 anchor：

| 区块 | 当前位置 | 备注 |
|---|---|---|
| 输入准备 | `train/models.py:2401` | `_prepare_forward_inputs(...)` 已是单独入口 |
| base result writer | `train/models.py:2517` | `_build_forward_base_result(...)` |
| direct output writer | `train/models.py:2531` | `_write_forward_direct_pose_outputs(...)` |
| contact-plan finalize | `train/models.py:2754` | `_finalize_contact_plan_outputs(...)` |
| side-routed leg shell | `train/models.py:3340` | `_forward_side_routed_leg_residual(...)` |
| non-side leg shell | `train/models.py:3460` | `_forward_non_side_leg_residual(...)` |
| `forward(...)` 主入口 | `train/models.py:3997` | 当前 maze 仍集中在这里 |
| event-clock step append shell | `train/models.py:4499` | batch 7 新增 nested def |
| event-clock step shell | `train/models.py:4620` | batch 7 新增 nested def |

当前最重要的结构事实：

- `contact-plan / event-clock` 已经具备 loop-local shell 雏形；
- `direct-pose` 已经具备 side-routed / non-side 两个 branch-sized shell；
- 但这些边界**还没有提升成顶层 stage graph**，所以 `forward(...)` 外层仍然需要读者穿过大量 local state 才能理解 owner。

---

## 4) 目标形状：single-file stage graph，而不是 helper graph

目标不是把 `forward(...)` 拆成更多 helper，而是让顶层只承担 **stage ordering**：

```python
def forward(...):
    forward_inputs = self._prepare_forward_inputs(...)

    def _run_contact_plan_stage() -> None:
        ...

    def _run_motion_core_stage() -> None:
        ...

    def _run_direct_pose_stage() -> None:
        ...

    def _run_aux_output_stage() -> None:
        ...

    _run_contact_plan_stage()
    _run_motion_core_stage()
    result = self._build_forward_base_result(...)
    _run_direct_pose_stage()
    _run_aux_output_stage()
    return result
```

关键点：

- 顶层 `forward(...)` **不再直接展开** `use_event_clock` / `side_routing` / `leg_mode` 等 feature branching。
- 这些 branching **全部留在 stage 内部**。
- stage 之间通过 single-file 局部 state 交接，不马上做 cross-file carrier 设计。

一句话：

> 顶层只看 stage 顺序；分支只在 stage 内部展开。

---

## 5) 拟议 stage 划分（按形态，不按 stage family）

### 5.1 Stage A — Boundary / runtime normalization

**owner**

- `forward_inputs`
- `state/cond/contacts/angvel/pose_history/plan_z/phase_z/phase_event_age`
- `is_single/device/dtype/B/Tq`
- `runtime_controls`

**现状**

- 这块大部分已经在 `train/models.py:2401` `_prepare_forward_inputs(...)` 中完成。
- `forward(...)` 内仍有少量 `_expand_state_sequence(...)` / `time_index -> t_grid` / `time PE` 构造逻辑残留。

**目标**

- 维持 Stage A 为 contract owner。
- 不让 Stage B / C / D 重新推断 batch/time/shape normalization。

**不做**

- 本轮不把 `time_index` / `time PE` 进一步拆成 module helper；优先保留 single-file local contract。

### 5.2 Stage B — Contact-plan state machine

**owner**

- `contacts_plan`
- `contacts_plan_logits`
- `contact_plan_debug_logits`
- `plan_z_next`
- `plan_feat_for_inject`
- `contacts_meas`
- `event_clock_delta_meas`
- `event_clock_lr_diff`
- `event_clock_lambda_corr`
- `event_clock_lambda_logit`
- `event_clock_dynamic_prior`
- `event_clock_delta_z`
- `phase_z_in_direct`
- `leg_side_cue_in`

**内部合法分支**

- `event_clock=on`
- `event_clock=off`

**必须保持在本 stage 内的逻辑**

- `plan_z` init / update
- `contacts_meas` canonicalization
- time-bias / debug logits / finalize handoff
- phase / cue 的 direct-pose bridge payload

**禁止泄漏到顶层的逻辑**

- `if self.use_event_clock ... else ...`
- `for _t in range(Tq)` 的 per-step contract 细节

### 5.3 Stage C — Motion core

**owner**

- `encoder_input`
- `soft_period`（若由 frozen encoder 产出）
- `h_temporal`
- `h_final`
- `attn`
- `out`
- `hidden_out`

**特点**

- 这块很长，但主要是线性算子链，不是 feature maze 的主因。

**目标**

- 把它变成单独 stage shell 后，顶层物理长度会显著下降；
- 但本 stage 内部不急着再细切。

### 5.4 Stage D — Direct-pose state machine

**owner**

- `direct_out`
- `direct_leg_omega`
- `direct_leg_omega_raw`
- `direct_leg_gate`
- `direct_leg_gate_logits`
- `direct_leg_scale`
- `direct_leg_scale_log`
- `direct_leg_scale_log_raw`
- `direct_leg_side_sign_gate`

**内部合法分支**

- `direct_pose_meas_mode`
- `direct_pose_feat_source`
- `side_routing=True/False`
- `leg_mode='so3'|'rot6d_add'`

**必须保持在本 stage 内的逻辑**

- direct feature source resolve
- direct readout
- side-routed / non-side residual dispatch
- final direct writer payload assemble

**禁止泄漏到顶层的逻辑**

- `if self._should_run_direct_pose_forward(...)`
- `if side_routing ... elif direct_pose_leg_head ...`

### 5.5 Stage E — Aux output writers

**owner**

- `lambda_fusion*`
- `omega_hat`
- `period_pred`

**目标**

- 继续保持 tail-only writeback 角色；
- 不反向控制前面 stage 的 branching。

---

## 6) 最小 de-maze 执行顺序（仍在单文件内）

### Step DM1 — 抽 `_run_contact_plan_stage(...)`

**为什么先做它**

- batch 7 已经把 event-clock loop 压成 closure-local shell；
- 这块是当前最成熟、最像独立 state machine 的部分；
- 也是 direct-pose bridge payload（`phase_z_in_direct` / `leg_side_cue_in`）的真正 owner。

**完成标志**

- 顶层不再直接看到 `if self.use_event_clock ... else ...`
- 顶层不再直接看到 contact-plan GRU loop body

### Step DM2 — 抽 `_run_motion_core_stage(...)`

**为什么第二个做**

- 它线性、稳定、验证成本低；
- 抽完以后 `forward(...)` 主体长度会显著下降；
- 也能让 contact-plan inject / shared encoder / PASA / motion head 的 owner 更清楚。

**完成标志**

- 顶层不再直接看到 encoder/PASA 细节
- 顶层只接收 `out/h_final/attn/soft_period`

### Step DM3 — 抽 `_run_direct_pose_stage(...)`

**为什么第三个做**

- 当前这块 branch-heavy，但 shell 已有基础：
  - `train/models.py:3340`
  - `train/models.py:3460`
- 真正需要的不是更多 helper，而是把顶层 dispatch 收走。

**完成标志**

- 顶层不再直接看到 `direct_pose_meas_mode` / `side_routing` / `leg_mode`
- 顶层只负责调用 stage，不参与 direct branch dispatch

### Step DM4 — 保持 `_run_aux_output_stage(...)` 尾段化

**说明**

- 这块可以最后做，也可以先留在现状 writer helper 组合；
- 重点不是细切，而是保持它只做 writeback，不污染前面阶段的 ownership。

---

## 7) 为了避免“为了拆而拆”，本计划明确禁止的做法

### 7.1 禁止按 posttrain stage 名拆代码

不允许做：

- `_run_stage70a_logic(...)`
- `_run_stage71_logic(...)`
- `train/models_stage6.py`

原因：

- stage family 是验证链，不是 runtime ownership。

### 7.2 禁止薄 helper 雨

不允许继续新增：

- `_append_phase_step(...)`
- `_append_cue_step(...)`
- `_compute_time_bias(...)`
- `_write_event_clock_stats(...)`

这类 helper 除非仍留在 single-file nested closure 内，且明显压平 host path，否则都属于“看起来整齐，实际更散”。

### 7.3 禁止 writer ownership 分裂

不允许把同一类 output key 写回散到多个 stage：

- `contacts_plan*` 只应由 Stage B owner 管理
- `direct_leg_*` 只应由 Stage D owner 管理
- `lambda/so3/period` 只应由 Stage E owner 管理

### 7.4 禁止把 de-maze 变成 compat 工程

不允许在 single-file de-maze 过程中顺手加入：

- old/new attr 兼容层
- deprecation warning
- runtime attr alias
- “先兼容着，Phase E 再删”

这些都违反 `docs/removal_policy.md` 的精神。

---

## 8) 每个 stage 的顶层 owner / read-set / write-set（草案）

| Stage | 顶层输入 | 顶层写出 | 顶层不应知道的分支 |
|---|---|---|---|
| Stage A `prepare` | 原始 `forward(...)` args | normalized inputs / runtime controls | 无 |
| Stage B `contact_plan` | normalized inputs, `meas_logits_prev`, `time_index` | `contacts_plan*`, `plan_z_next`, `plan_feat_for_inject`, `phase_z_in_direct`, `leg_side_cue_in`, `event_clock_*` | `event_clock on/off` |
| Stage C `motion_core` | `state`, `cond`, `pose_history`, `angvel`, `plan_feat_for_inject` | `out`, `h_final`, `attn`, `soft_period` | PASA / inject / period-hint 细节 |
| Stage D `direct_pose` | `contacts_plan`, `contacts_meas`, `phase_z_in_direct`, `leg_side_cue_in`, `h_final/h_temporal/cond` | `out_direct`, `direct_leg_*` | `meas_mode`, `feat_source`, `side_routing`, `leg_mode` |
| Stage E `aux_outputs` | `h_final`, `contacts_err`, `rollout_step`, `soft_period` | `lambda_fusion*`, `omega_hat`, `period_pred` | `lambda rollout feature` 细节 |

本表的目的不是现在就设计 cross-file API，而是先把 **顶层 owner** 写死，减少“下一次改代码时要靠猜”的空间。

---

## 9) 验证门禁（后续实际落代码时复用）

每次 single-file de-maze 落代码，至少跑：

- `python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`
- `python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`
- `python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`
- `python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`
- stage6 deterministic smoke
- AST broad-handler count
- removal-policy grep

本计划特别要求：

- 若只是在顶层 stage ordering 上重排，不应改变 snapshot / fingerprint / stage6 smoke 结果。
- 若某一步必须新增测试，优先加 **stage-shell / owner / branch-shape regression**，而不是对 implementation detail 过耦合的 helper-dispatch test。

---

## 10) 停止条件（避免无限单文件 batching）

single-file de-maze 到以下条件时，应停止并转入 Phase E：

1. `forward(...)` 顶层只剩 `prepare -> contact_plan -> motion_core -> direct_pose -> aux -> return`
2. 顶层不再直接出现 `event_clock on/off` / `side_routing` / `leg_mode` 等 branching
3. 每个 output family 的 owner 已唯一化
4. 再继续整理会明显滑向“薄 helper 增多、认知成本转移到调用图”

换言之：

> 当剩余问题从“主 branch 是迷宫”变成“某个 stage 内部还有点长”时，就该停。

---

## 11) Phase E 入口条件（不是现在就做，但要知道何时算 ready）

只有满足以下条件，才建议开始 cross-file split：

- Stage graph 已在单文件内稳定一轮以上；
- snapshot / fingerprint / stage6 smoke 对该结构稳定；
- stage owner / read-set / write-set 在文档与代码里都清楚；
- 没有再依赖“顺手猜一个 local”才能改动；
- 没有新增 removal-policy 反模式。

这时的跨文件迁移才是 **mechanical move**，而不是 structural rewrite。

---

## 12) 当前建议

当前建议不是继续 batch 8 式“局部薄壳 cleanup”，而是：

1. 先按本文执行 **single-file de-maze**
2. 只把 `forward(...)` 顶层收成 stage graph
3. 暂时不拆文件
4. 等 stage graph 稳定后，再评估 Phase E

一句话总结：

> 当前最值钱的下一步，不是“把代码搬到别的文件”，  
> 而是“先让 `forward(...)` 自己不再需要隐含推测才能修改”。

---

## Implementation Update — 2026-04-24 single-file de-maze pass 1

- **本轮目标**：只在 `train/models.py` 内把 `EventMotionModel.forward(...)` 顶层收成更线性的 stage ordering；把 `contact-plan / event-clock` 继续留在单个 stage shell 内；顶层不再直接展开 `event_clock on/off`。
- **实际完成项**：`forward(...)` 现已用 4 个 nested stage shell 组织主路径：`_run_contact_plan_stage()` → `_run_motion_core_stage()` → `_run_direct_pose_stage()` → `_run_output_writeback_stage()`。`event_clock on/off` 分叉被内收到 `_run_contact_plan_stage()`，direct-pose 的 feature source / meas mode / side-routing / leg residual dispatch 继续留在 `_run_direct_pose_stage()` 内部；顶层只保留 stage 顺序与最终 `return`。未拆新文件，未新增 dataclass-return / 15-tuple return / helper 雨，未改 output key、checkpoint contract、默认超参。
- **修改文件列表**：`train/models.py`；`docs/changes/2026-04-24_train_models_forward_single_file_de_maze_plan.md`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`；`python3 - <<'PY' ... ast.parse(train/models.py) ... except-handler count ... PY`；`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch8_20260424 --run_name train_models_pre_phase_e_simplify_batch8_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`；touched diff removal-policy grep。
- **验证结果**：全部通过。联合 `unittest` 仍为 `135` 个用例通过；snapshot regression 通过；state_dict fingerprint regression 通过；AST 计数保持 broad=`0` / exact=`0` / as_exc=`0`；stage6 deterministic smoke 结果 `ok_steps=5 skipped=0`。
- **阻塞项 / 风险**：顶层 stage graph 已落地，但 Stage B / Stage D 内部仍各自偏长；如果继续整理，必须继续坚持“单文件 stage shell > 薄 helper graph”，避免把认知负担从主 branch 转移到调用图。当前 `_run_output_writeback_stage()` 仍承接 contact-plan writeback 与 aux writer；是否进一步收窄 owner，留待下一轮按可读性收益再判断。
- **下一轮建议动作**：先观察这一版 stage graph 在 snapshot / fingerprint / stage6 smoke 下稳定一轮；若继续推进，优先只做 stage 内部 read-set / write-set 的收口，不做跨文件迁移。若下一轮已经无法继续提升顶层线性度而不引入薄 helper，则停止 single-file de-maze，转入 Phase E preflight 评估。

## Implementation Update — 2026-04-25 single-file de-maze pass 2 owner micro-fix

- **本轮目标**：在不增加 helper graph 的前提下，把 `contacts_meas` shape normalize 与 `contacts_err` derive 从 `_run_output_writeback_stage()` 挪回 `_run_contact_plan_stage()`，让 Stage E 更接近纯 write-only tail。
- **实际完成项**：`train/models.py` 中 `e_t = contacts_plan - contacts_meas.to(...)` 现已在 Stage B 内完成，且与 `contacts_meas` canonicalization 保持同域；`_run_output_writeback_stage()` 不再执行 `contacts_meas.unsqueeze(1)`，也不再声明 `nonlocal contacts_meas` / `nonlocal e_t`。Stage E 继续只负责 `contacts_plan*` / `contacts_err` / `lambda_fusion*` / `omega_hat` / `period_pred` 的 writeback，未新增 helper、未改数值语义、output key、checkpoint contract、默认超参。
- **修改文件列表**：`train/models.py`；`docs/changes/2026-04-24_train_models_forward_single_file_de_maze_plan.md`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`；`python3 - <<'PY' ... ast.parse(train/models.py) ... except-handler count ... PY`；`sed -n '4230,5290p' train/models.py | rg -n '<removal-policy §6 patterns>'`；`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch9_20260425 --run_name train_models_pre_phase_e_simplify_batch9_20260425 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`。
- **验证结果**：全部通过。联合 `unittest` 仍为 `135` 个用例通过；snapshot regression 通过；state_dict fingerprint regression 通过；AST 计数保持 broad=`0` / exact=`0` / as_exc=`0`；touched-range removal-policy grep 零命中；stage6 deterministic smoke 结果 `ok_steps=5 skipped=0`。
- **结果解释**：这一步不是继续切分，而是把 Stage B 的 derive owner 放回 Stage B，同时保留 Stage E 的物理 colocation。主 branch 形状保持不变，但 `_run_output_writeback_stage()` 更接近名副其实的 tail writeback shell。
