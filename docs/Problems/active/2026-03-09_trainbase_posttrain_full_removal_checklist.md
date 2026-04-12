# 2026-03-09 Trainbase / Posttrain 最后一轮整段移除清单（含 `whitebox` / 计算分支）

## 目标

这份清单对应
`docs/Problems/active/2026-03-07_trainbase_posttrain_unused_branch_inventory.md`
里“第 5 步”的正式执行版。

本轮目标不是再加一层 guard / fail-fast / inert default，而是做**整段移除**：

- 删除当前 mainline 已不再需要的运行时计算分支；
- 删除仅为 fallback / reference control 保留的 `whitebox` 路由；
- 删除 `contact_phase_state` state core 本体；
- 删除 `lambda final` 在 `train_lambda_head` 模式下仅作 compat-read 的 direct/leg 字段读取链。

一句话口径：

> 这轮做的是 **hard delete + contract shrink**，不是“保留代码、只靠配置禁止启用”。

---

## 前置条件状态（已满足）

截至 2026-03-09，下面几个前置条件已经成立：

- posttrain mainline 已同步到当前 accepted 运行口径：
  `docs/posttrain_pipeline.md`
- accepted downstream chain 已固定为：
  `70R(s180 low-LR trunkfull) -> 71 -> 72 -> lambda final`
- `pretrain_contact + clamp1 + affine_mix08` 已成为当前 accepted contact 主线；
- 高阶 direct-pose 支线已完成独立清理轮，不再阻塞最后一轮整段移除。

因此，这轮可以直接进入代码删除，而不是继续停留在“最后处理，暂不动”。

---

## 本轮保留 / 本轮删除

### 保留（当前 active mainline）

- `train.posttrain` 的 XOR 主链：`train_direct_pose` / `train_lambda_head`
- `posttrain_contacts_source=pretrain_contact`
- `pretrain_contact + clamp1 + affine_mix08`
- Stage6 / 70a / 70b / 70c / promoted 70R / 71 / 72 / `lambda final`
- `phase_z concat`
- `replace_contacts`
- `lambda_fusion_head` 及其真实 active 配置（包括 `lambda_fusion_use_rollout_step`）
- `direct_pose_head` 作为 `lambda final` 上游 expert 结构本体

### 删除（本清单覆盖范围）

- trainbase 的 `whitebox` rollout 输入与 fallback 解析
- validate / control lane 的 `whitebox` contacts source
- `contact_phase_state` state core 与其 event/reset/state update 计算链
- `lambda final` 下仅作 compat-read 的 direct/leg 字段与相应兼容读取逻辑
- 与以上分支绑定的 parser / config key / metrics / docs / control scripts

---

## A. `whitebox` / fallback 整段移除

### A1. trainbase runtime

- 状态更新（2026-03-09 代码核对）：
  `whitebox/fallback` runtime 主链已从 `trainbase` 入口退休；仅 `w_contact_meas` supervision shell 仍待单独清理。

- [x] `train/training_MPL.py`：删除 `_contact_meas_whitebox` 及其 debug payload / logging 分支。
- [x] `train/training_MPL.py`：删除 `trainbase_contacts_source=auto|whitebox`，收缩为仅允许 `pretrain_contact`。
- [x] `train/training_MPL.py`：删除“缺失 frozen bundle 时 fallback 到 `whitebox`”的解析逻辑。
- [x] `train/training_MPL.py`：删除 whitebox 专属 CLI/config knobs（ground-z / gate-by-hit / debug whitebox logging 等）。
- [ ] `train/training_MPL.py`：若 `w_contact_meas` loss 壳尚未完全清掉，则与 whitebox runtime 一并删除，不再保留 supervision-off shell。
- [x] `train/eval_utils.py`：删除对 `trainer._contact_meas_whitebox` 的兜底调用。

### A2. validate / reference control

- [x] `train/validate/run_freerun_cycles.py`：删除 `contacts_meas_source=whitebox` 选项与 lazy whitebox 计算路径。
- [x] `train/validate/run_freerun_cycles.py`：删除 `whitebox_missing` / `wb` alias / `log_contacts_whitebox` 相关状态分支。
- [x] `tools/run_stage67_transition.py`：删除 `whitebox` 相关 choices、summary 字段与 compare wording。
- [x] `docs/posttrain_pipeline.md`：删除 “Optional historical control (`whitebox` source)” 命令段。
- [x] active docs / handoff 中把 `whitebox` 从“保留 control lane”改成“已退休历史口径”。

### A3. 配置 / 文档 / 资产

- [x] 清点 active configs，删除所有 `trainbase_contacts_source=whitebox|auto` 残留。
- [ ] 清点脚本 / README / 问题单中的 `whitebox` 主线表述，只保留历史记录，不再保留执行建议。
  2026-03-09 A3 复核：当前主要残留是 `docs/contact_loop_closure_design.md`，它仍写着 “默认推荐 white-box meas”，且该文件已在当前工作树中被修改，暂未自动改动。
- [x] 对仍以 `whitebox` 命名的历史 ckpt / 输出目录补一条 archive 说明，避免被误判为现役路径。
  见 `docs/changes/2026-03-09_whitebox_artifact_archive_note.md`。

核对备注（2026-03-09）：

- `config/exp_phase_mpl.clean.json` 已固定为 `trainbase_contacts_source=pretrain_contact`。
- 主树 `config/ train/ tools/ docs/` 下未再发现 `trainbase_contacts_source=whitebox|auto` 实际配置残留；当前 grep 命中仅剩本 checklist 自身。

### A4. 完成定义

- [ ] 主树中不再存在 `trainbase_contacts_source=whitebox|auto`
- [ ] 主树中不再存在 `contacts_meas_source=whitebox`
- [ ] 主树中不再存在 `_contact_meas_whitebox`

2026-03-09 状态复核：

- `train/` runtime 代码范围内，上述 3 项 grep 已为 0；就运行入口而言，`whitebox` / fallback 退休状态已成立。
- 但按当前 checklist 的 repo 口径（`train docs config tools`）看，A4 仍未完成：
  - `trainbase_contacts_source=whitebox|auto` 仅剩本 checklist 自身的历史记录表述；
  - `contacts_meas_source=whitebox` 仍残留在历史问题单 / 研究文档；
  - `_contact_meas_whitebox` 仍残留在历史设计文档 / 离线分析说明中。
- 因此 A4 现在应解读为：**runtime 已完成，主树文本级清理未完成**。

---

## B. `contact_phase_state` state core 整段移除

注意：这里删的是 trainbase / model 内的 `contact_phase_state` state core，**不是**删除 posttrain 当前 active 的
`phase_z concat` / `replace_contacts`。

### B1. trainbase parser / runtime config

- [x] `train/training_MPL.py`：删除 `contact_phase_state_enable`
- [x] `train/training_MPL.py`：删除 `contact_phase_state_init_mode`
- [x] `train/training_MPL.py`：删除 `contact_phase_state_hidden`
- [x] `train/training_MPL.py`：删除 `contact_phase_state_delta_max`
- [x] `train/training_MPL.py`：删除 `contact_phase_state_delta_init`
- [x] `train/training_MPL.py`：删除与 `contact_phase_state_event_*` / `phase_reset_source` 相关的 removed-keys compatibility 壳，因主链中该概念整体下线。
- [x] `train/training_MPL.py`：删除 runtime_cfg/model build 中对 `contact_phase_state_*` 的传递。

### B1.5 风险备注（2026-03-09）

- `B1` 属于 trainbase 入口 contract 收缩，风险相对低；它删除的是 parser/runtime config 暴露面，**不是**
  当前 active 的 `pretrain_contact` 输入链本体。
- 从 `B2` 开始，改动会进入共享的 `train/models.py::EventMotionModel`，不再是 trainbase-only 清理。
- `pretrain_contact` 的输入路径本身独立于 `contact_phase_state`；但当前 posttrain active 链仍通过
  `contact_phase_state -> phase_z -> direct_pose_use_phase_z` 来消费 phase/contact 提示。
- 当前 accepted downstream configs 中：
  - `70b` 仍在使用 `direct_pose_use_phase_z=true` + `direct_pose_phase_z_mode=concat`
  - `70c / 71 / 72 / lambda final` 仍在使用
    `direct_pose_use_phase_z=true` + `direct_pose_phase_z_mode=replace_contacts`
- 因此，若在 `B2` 里直接删除 state core，本质上影响的是 **posttrain 如何消费 contact 提示**，而不只是
  trainbase 是否还能接入 `pretrain_contact`。
- `phase_reset_source=none` 只表示 reset 分支关闭，**不等于** `contact_phase_state` state core 已停用；
  `delta integrate`、`phase_z_next`、`phase_event_age_next` 这条 recurrent contract 当前仍在跑。
- 直接风险面包括：
  - `train/posttrain.py` 仍会从 ckpt/config 推断并传递 `contact_phase_state_*` / `phase_reset_source`，
    并维护 `phase_z` / `phase_event_age` rollout state；
  - `train/training_MPL.py` rollout / ONNX export 仍保留 `plan_z_next + phase_z_next` contract；
  - `train/validate/run_freerun_cycles.py` 与 `tools/run_stage67_transition.py` 仍有 phase-reset /
    phase-state 相关 CLI/summary 待清。
- 当前风险判断应记为：
  - `A` / `B1`：低风险，属于主链 contract 收缩；
  - `B2` 及以后：高风险，属于 shared model / posttrain active mainline 解耦。
- 执行建议：
  1. 先完成 `B3` 的 validate/tools phase 依赖清点；
  2. 再确认 active posttrain 是否已有无 `phase_z` 的替代路径；
  3. 最后再删除 `EventMotionModel` 内的 `contact_phase_state` state core。

### B2. model 结构与前向计算

- [ ] `train/models.py`：删除构造参数 `contact_phase_state_*` 与 `phase_reset_source`。
- [ ] `train/models.py`：删除 `contact_phase_state_init`、`contact_phase_state_delta_head`、`_contact_phase_state_dim`。
- [ ] `train/models.py`：删除 prev-phase 初始化、state update、event reset、delta integrate 逻辑。
- [ ] `train/models.py`：删除因 `contact_phase_state` 产生的额外输入拼接与状态缓存。
- [ ] `train/models.py`：删除只服务于 `contact_phase_state` 的 debug / metric / auxiliary return 字段。

### B3. validate / analysis / scripts

- [ ] `train/validate/run_freerun_cycles.py`：删除依赖 `contact_phase_state` 的 phase-reset 运行逻辑与导出字段。
- [ ] `train/validate/run_freerun_cycles.py`：删除 `phase_reset_source`、`phase_reset_source_strict` 等仅服务于该状态机的 CLI/summary。
- [ ] `tools/run_stage67_transition.py`：删除 phase-reset compare 字段与命令透传。
- [x] 历史 phase/TTA 说明文档已从主文档集中移除。

### B4. 完成定义

- [ ] 主树中不再存在 `contact_phase_state_enable`
- [ ] 主树中不再存在 `contact_phase_state_event_*`
- [ ] 主树中不再存在 `phase_reset_source`（若仅用于这套状态机）
- [ ] 主树中不再存在 `_contact_phase_state_dim` / `contact_phase_state_delta_head`

---

## C. `lambda final` compat-read 整段移除

注意：这里删的是 **`train_lambda_head` 模式下不形成有效训练目标、只做兼容读取的 direct/leg 字段**。

不删除：

- `direct_pose_head` 本体
- `lambda_fusion_head`
- 当前 active 的 `lambda_fusion_*` 真实配置
- 70R / 71 / 72 仍在使用的 direct/leg 正式主链能力

### C1. config contract 收缩

- [ ] lambda-final active config 只保留 `train_lambda_head` 真正需要的字段。
- [ ] 从 lambda-final config 中移除 inert 的 `direct_pose_*train_only` / `direct_pose_reinit` / direct-leg 对齐监督类字段。
- [ ] 对仍沿用 `fullcompat` 命名的 lambda-final config，补一轮重命名或注释，避免继续暗示“兼容壳仍是现役 contract”。

### C1.5 风险备注（2026-03-09）

- `C` 的风险整体低于 `B2`，因为当前 `train_lambda_head` 模式下真正可训练参数已经收敛到
  `lambda_fusion_head`；`direct/leg train_only` 字段并不会像 `B2` 那样直接改动 shared state core。
- 但 `C` 不能做成“把 lambda final config 里的所有 `direct_pose_*` 一口气删光”，因为其中一部分字段虽然
  不再形成训练目标，仍然决定 **冻结的 direct expert 结构与 ckpt 形状**：
  - `direct_pose_use_phase_z`
  - `direct_pose_phase_z_mode`
  - `direct_pose_split_enable`
  - `direct_pose_nonleg_proj_dim`
  - `direct_pose_leg_enable`
  - `direct_pose_leg_bones`
  - `direct_pose_leg_mode`
  - `direct_pose_leg_gate_mode` / `direct_pose_leg_gate_power`
- 上述字段当前仍会影响 `train/posttrain.py` 中对 `EventMotionModel` 的实例化、direct head shape 适配、
  ckpt tensor drop / warm-start 路径；如果误删，会把 `lambda final` 从“冻结 direct expert + 训练 lambda”
  变成“改动 direct expert 结构”的另一件事。
- 相对安全、应优先清理的是 **lambda 模式下已 inert 的 direct/leg 训练语义字段**：
  - `direct_pose_leg_train_only`
  - `direct_pose_leg_gate_train_only`
  - `direct_pose_nonleg_train_only`
  - `direct_pose_reinit`
  - `direct_pose_leg_gate_sup_weight`
  - `direct_pose_leg_align_weight` 及其 schedule / oracle / mode / mag / res / sign / thresh 附属字段
- 当前代码口径下，`train_lambda_head` 模式：
  - 只会 `unfreeze lambda_fusion_head`
  - rollout objective 固定为 `blend`
  - direct/leg align 与 gate-supervision 权重不会进入 lambda-mode rollout kwargs
- 因此，`C` 的真实风险点不在“训练目标回路”，而在：
  - parser / config 是否仍把 inert direct/leg 字段伪装成 active contract；
  - model build / ckpt adapt 是否仍为 lambda 模式保留过宽的 direct-leg compat-read；
  - 保存出的 runtime/log 是否继续回显这些 inert 字段，制造“仍在生效”的错觉。
- 当前风险判断应记为：
  - 删除 lambda-mode 的 direct/leg train-only / supervise compat-read：中低风险；
  - 误删 frozen direct expert 结构字段或 shape-adaptation 路径：中高风险。
- 执行建议：
  1. 先把 `train_lambda_head` 路径下不参与 objective / unfreeze 的 direct-leg 字段从 parser/logging 中剥离；
  2. 再单独核对哪些 direct expert 结构字段仍需保留到 ckpt fully reconstruct；
  3. 最后再收 `fullcompat` 命名与历史 shape-adaptation 壳。

### C1.6 执行分层（2026-03-09）

#### 可立即执行（低风险）

- 从 lambda-final active config / parser / logging 中删除或忽略以下 **lambda-mode inert** 字段：
  - `direct_pose_leg_train_only`
  - `direct_pose_leg_gate_train_only`
  - `direct_pose_nonleg_train_only`
  - `direct_pose_reinit`
  - `direct_pose_leg_gate_sup_weight`
  - `direct_pose_leg_align_weight`
  - `direct_pose_leg_align_schedule`
  - `direct_pose_leg_align_start_weight`
  - `direct_pose_leg_align_warmup_steps`
  - `direct_pose_leg_align_ramp_steps`
  - `direct_pose_leg_align_oracle_min_deg`
  - `direct_pose_leg_align_oracle_weight_deg`
  - `direct_pose_leg_align_mode`
  - `direct_pose_leg_align_mag_weight`
  - `direct_pose_leg_align_res_weight`
  - `direct_pose_leg_align_sign_weight`
  - `direct_pose_leg_align_cos_thresh`
- 目标语义：`train_lambda_head=true` 时，只保留“冻结 direct expert + 训练 `lambda_fusion_head`”必需字段。
- 预期收益：
  - active config 不再误导为“lambda 模式还在训练 leg/nonleg direct 分支”
  - runtime/log 不再出现 inert direct-leg 伪 active 信号
  - 不改变 frozen direct expert 的 shape / instantiation contract

#### 当前必须保留（否则会误伤 accepted mainline）

- 下面这些字段虽然不属于 lambda-mode train-only objective，但当前仍定义 frozen direct expert 结构：
  - `direct_pose_use_phase_z`
  - `direct_pose_phase_z_mode`
  - `direct_pose_split_enable`
  - `direct_pose_nonleg_proj_dim`
  - `direct_pose_leg_enable`
  - `direct_pose_leg_bones`
  - `direct_pose_leg_mode`
  - `direct_pose_leg_gate_mode`
  - `direct_pose_leg_gate_power`
- 保留原因：
  - `train/posttrain.py` 仍用它们实例化 `EventMotionModel`
  - direct trunk / leg head / phase input shape 仍依赖这些字段
  - accepted `72 -> lambda final` handoff 仍需要按现有 frozen expert 结构重建 ckpt

#### 最后再动（需单独核对 ckpt/shape compat）

- 以下内容不要和上面的“低风险收缩”混做一轮：
  - direct head input shape adaptation
  - leg/non-leg split 相关 tensor drop
  - `load_state_dict` 前的 compat tensor pop / warm-start 分支
  - `fullcompat` 命名与历史 shape 兼容壳
- 进入这一步前应满足：
  1. lambda-final config 已先完成 inert 字段收缩
  2. `train_lambda_head` 路径下日志/指标已不再依赖 direct-leg train-only 键
  3. 已确认当前 accepted `72 -> lambda final` ckpt reconstruct 不再依赖额外 compat-read

### C2. posttrain 入口 / build / loss

- [ ] `train/posttrain.py`：在 `train_lambda_head=true` 路径下，不再解析/消费 direct-leg train-only compat 字段。
- [ ] `train/posttrain.py`：删除只为 lambda 模式吞下旧 ckpt/config 形状而保留的 direct/leg compat-read 分支。
- [ ] `train/posttrain.py`：清理 lambda 模式下不会生效的 logging / accum / metrics 字段，避免伪 active 信号。
- [ ] `train/posttrain.py`：把 lambda 模式收缩为“冻结上游 direct expert + 训练 `lambda_fusion_head`”的最小语义面。

### C3. checkpoint / historical compatibility

- [ ] 清点 `load_state_dict` / shape adaptation / tensor drop 中仅服务于 lambda compat-read 的分支；若不再影响 accepted mainline，则删除。
- [ ] 对仍需保留的历史 ckpt 兼容，迁移到单独 archive tool / offline conversion，而不是继续留在 main runtime。

### C4. 完成定义

- [ ] lambda-final active config 中不再出现 inert direct/leg compat 字段
- [ ] `train/posttrain.py` 的 `train_lambda_head` 路径不再读取 direct/leg train-only compat 键
- [ ] runtime/log 中不再出现“lambda mode 读了 direct/leg compat 字段但未生效”的伪 active 现象

---

## D. 建议执行顺序（按依赖）

1. 先删 trainbase `whitebox` / fallback runtime
2. 再删 validate / control 的 `whitebox`
3. 再删 `contact_phase_state` state core
4. 最后删 `lambda final` compat-read
5. 收尾清 configs / docs / tools / archive 说明

这个顺序的原因：

- `whitebox` 移除后，contacts 主链 contract 可以彻底收缩到 `pretrain_contact`；
- `contact_phase_state` state core 会牵动 `train/models.py` 与 validate 逻辑，适合单独一轮；
- `lambda final` compat-read 最容易牵到 ckpt 兼容，放最后收最稳。

---

## E. 每一轮删除后的最小复核

### E1. trainbase

- [ ] 复跑当前 accepted `train.training_MPL` 主命令
- [ ] 确认 contacts runtime 固定命中 `pretrain_contact`
- [ ] 确认不再存在 `whitebox` / fallback applied log

### E2. posttrain mainline

- [ ] 按 `docs/posttrain_pipeline.md` 复跑 accepted 链：`70R(s180) -> 71 -> 72 -> lambda final`
- [ ] 确认 `train.posttrain` 入口只走 `pretrain_contact`
- [ ] 确认 `lambda final` 指标未因 compat-read 删除而回退

### E3. 代码面 grep 复核

- [ ] `rg -n "_contact_meas_whitebox|contacts_meas_source=.*whitebox|trainbase_contacts_source=.*whitebox|trainbase_contacts_source=.*auto" train docs config tools`
- [ ] `rg -n "contact_phase_state|phase_reset_source|contact_phase_state_event_" train docs config tools`
- [ ] `rg -n "fullcompat|train_lambda_head.*direct_pose_leg|direct_pose_leg_train_only|direct_pose_nonleg_train_only" train docs config`

---

## F. Done 定义

满足下面 4 条时，这轮可以算真正完成：

1. `whitebox` 已不再出现在主树 runtime / validate / control 入口；
2. `contact_phase_state` state core 已从主树代码删除，而不是仅 hard-disable；
3. `lambda final` 已收缩为最小必要 contract，不再携带 direct/leg compat-read 壳；
4. accepted `trainbase -> posttrain` 主链复跑通过，结论不回退。
