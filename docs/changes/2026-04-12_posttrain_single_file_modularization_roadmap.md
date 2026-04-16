# [2026-04-12] `train/posttrain.py` 单文件模块化整理路线图（Phase 4 已收口）

Date: 2026-04-12  
Status: Phase 1-4 complete / validation-only follow-up  
Scope: `train/posttrain.py`（只做单文件内整理，不跨文件迁移）  
Goal: 按 `docs/refactor/posttrain_single_file_modularization_note.md` 的 Phase 规划，把文件内部职责边界整理清楚；当前 **Phase 1-4 已完成首轮收口**，后续只建议做验证或极小净减。  
Non-goal: 不改 CLI 参数名、不改 config key、不改 checkpoint 格式、不改训练语义、不新建公共 util/module。

---

## 0) 当前策略（单文件 Phase 已收口）

本路线图用于承接 `docs/refactor/posttrain_single_file_modularization_note.md`，但记录形式按
`docs/changes/2026-03-01_posttrain_minimal_refactor_roadmap.md` 风格维护，避免后续变得零散。

当前执行原则：

1. **只整理 `train/posttrain.py` 内部结构**
   - 不跨脚本抽公共逻辑。
   - 不拆新文件。
2. **优先保持已形成的职责边界**
   - Build & Checkpoint、Config Contract、Rollout Kernel、Train Runtime、CLI Entry 已完成首轮单文件内收口。
   - 后续如继续处理，优先验证或极小净减；不再扩拆 helper / schema / runtime 框架。
3. **保持行为不变**
   - 不能改 ckpt 读写字段。
   - 不能改 direct/lambda/event_clock 的启停语义。
   - 不能改现有 wrapper 调用方式。

本阶段额外约束：
- 如无必要，不新增新的“大 helper”。
- 不顺手清理其他文件或其他训练入口。
- 允许 LOC 持平或小幅下降；重点是结构连续性与后续可维护性。

---

## 1) 当前状态（Phase 1 已完成）

Phase 1 已完成内容：
- `_build_posttrain_model_from_ckpt(...)` 已由多返回值切换为单文件 dataclass：`PostTrainModelArtifacts`
- `main()` 与 `_save_posttrain_outputs(...)` 已改为通过 dataclass 读写
- 外部行为保持不变

当前工作树基线（2026-04-12）：
- LOC：`5798`
- `def` 数：`91`
- `py_compile`：通过

当前工作树（Phase 4 收口压缩后）：
- LOC：`5884`
- `def` 数：`98`
- 新增单文件 helper：`6`
- `py_compile`：通过
- `python3 -m train.posttrain --help`：通过
- 1-step smoke：通过（现有 direct-pose ckpt，CPU，输出 `/tmp/posttrain_phase2_5_smoke/ckpt_last_phase2_5_smoke.pth`）

说明：
- 第一版 Phase 2 先优先收拢 Build & Checkpoint 边界，因此 LOC / `def` 数相对基线有上升。
- 后续 Phase 3/4 已完成收口：Config Contract 主线更显式，Runtime / Rollout 主线已按 prepare / unroll / finalize 与 train runtime 边界整理。
- 最近一轮 Phase 4 压缩把 LOC 从 `5919` 压回 `5901`，`def` 数保持 `100`。
- Phase 2.5-style 小回收已内联两个过薄 build-only helper（contact-plan init 与 direct-pose leg gate），LOC 继续压到 `5884`，`def` 数降到 `98`。
- 增量主要来自单文件内 infer / compat / adapt 小 helper，目的是让 `_build_posttrain_model_from_ckpt(...)`
  先形成稳定的职责骨架；当前不建议继续扩展结构性重构。

当前已收口的主要热点：
- Config Contract：`_cfg_from_payload(...)`、`_cfg_parse_direct_pose(...)`、`_cfg_parse_lambda_rollout(...)`
- Rollout Kernel：`_lambda_fusion_loss_rollout(...)`
- Train Runtime：`_build_rollout_mode_kwargs(...)`、`_run_training_loop(...)`
- CLI Entry：`main()`

现状问题：
- 单文件职责边界已基本清楚。
- 继续结构性整理的收益开始递减，风险主要是误改训练语义。
- 后续建议优先做更标准 ckpt/config 的 targeted smoke，而不是继续拆 runtime/rollout。

---

## 2) Phase 2：收拢 Build & Checkpoint 区块（本轮目标）

目标：
- 不改变 `_build_posttrain_model_from_ckpt(...)` 的外部行为。
- 只在函数内部把 checkpoint 相关逻辑整理成更连续的职责区块。
- 让后续 Phase 3（Config Contract）和 Phase 4（Runtime / Rollout 排序）可以在更稳定的边界上继续做。

建议分步如下。

当前进展（2026-04-12）：
- 已完成第一版 Phase 2 骨架：`_build_posttrain_model_from_ckpt(...)` 主体已形成
  `infer-only -> build instance -> pre-load compat/adapt -> load -> post-load guards -> artifacts return`
  的连续结构。
- 已完成 direct-pose compat / drop / adapt 的第一版收拢，相关子流程已进入单文件 helper。
- 已完成 attach/load 顺序显式化，`attach_motion_encoder(...)` 必须先于 `load_state_dict(...)` 的依赖已写入注释。
- 已完成 build-only infer 的第一版收拢：contact-plan init / event-clock / lambda-fusion / leg-gate
  的推断已抽到单文件 helper。
- 已对 `_resolve_direct_pose_build_cfg(...)` 做一轮“净减法”压缩：收口了 direct readout shape 推断、phase-z/time-pe 输入维度推断、以及 split/stepc/arm-split compat 分支。
- 当前若继续停留在 Phase 2，收益将明显递减；除非后续再发现新的 checkpoint 兼容热点，否则更适合转入 Phase 3。

---

### Step P2-1 — 建立 `_build_posttrain_model_from_ckpt(...)` 的连续区块骨架（低风险）

当前状态（2026-04-12）：已完成第一版。

先不急着抽 helper，先把函数内部顺序固定成以下大段：

1. **Checkpoint Read & Shape Infer**
   - `torch.load`
   - `posttrain_cfg` 提取
   - `state_dict` 过滤
   - width / period_dim / nin / contact_plan / direct_pose / event_clock / lambda 的 shape 推断
2. **Instantiation Resolve**
   - 决定模型实例化参数
   - 构造 `EventMotionModel`
   - `validate_and_fix_model_`
3. **Pre-load Compat / Adapt / Drop**
   - encoder attach（如果必须在 load 前发生）
   - direct-pose reinit/drop
   - phase-z 输入维度适配
   - retired tensor drop
   - leg override / shape mismatch drop
4. **State Load**
   - `model.load_state_dict(..., strict=False)`
5. **Post-load Runtime Guards**
   - `train_direct_pose` / `train_lambda_head` 等 fail-fast
   - optional gate logit reset
6. **Artifacts Return**
   - `PostTrainModelArtifacts(...)`

要求：
- 本步主要是**顺序连续化**与注释标题澄清。
- 不引入行为变化。
- 不做跨函数抽取。

验收：
- 读者可以仅靠分段标题快速定位“infer / instantiate / pre-load compat / load / post-load guard”。

---

### Step P2-2 — 收拢 direct-pose compat / drop / adapt 子区块（中风险）

当前状态（2026-04-12）：已完成第一版。
- 已落地 helper：
  - `_drop_direct_pose_ckpt_tensors(...)`
  - `_adapt_direct_pose_phase_z_ckpt_inputs(...)`
  - `_drop_retired_direct_pose_highorder_ckpt_tensors(...)`
  - `_drop_incompatible_direct_pose_leg_ckpt_tensors(...)`

将当前分散但强相关的 direct-pose checkpoint 兼容逻辑整理为一个连续子区块，建议内部顺序如下：

1. **是否需要 drop direct-pose 权重**
   - `direct_pose_reinit`
   - `shape_override`
   - `split_mismatch`
   - `stepc_leg_terminal_mismatch`
   - `arm_split_mismatch`
2. **执行 direct-pose 大类 drop**
   - `drop_direct_pose_weights`
   - 相关 `direct_pose_*` tensor 清除
3. **phase-z 输入适配**
   - 仅处理 direct trunk / leg / gate 第一层权重的 in-dim 兼容
4. **retired high-order direct-pose tensor 清理**
5. **leg override / leg shape mismatch 清理**

目标：
- 让“direct-pose 兼容”看起来像一个完整子流程，而不是散落的几段特判。
- 保持所有 warning / fatal / fallback 行为不变。

允许的小改动：
- 仅限局部变量重命名、连续注释标题、必要的小范围 helper。

不允许：
- 改 `direct_pose_*` override 的 fail-fast 条件。
- 改任何 warm-start / reinit / drop 的触发语义。

---

### Step P2-3 — 收拢 attach/load 相关顺序并显式化依赖（中风险）

当前状态（2026-04-12）：已完成。

当前一个关键隐含约束是：
- `attach_motion_encoder(...)` 必须在 `load_state_dict(...)` 前发生，
  因为 `period_dim/period_encoder` 可能被 encoder bundle 影响。

这一点 Phase 2 里应该显式化：

- 把 `attach_motion_encoder(...)` 放到明确的 **Pre-load Compat** 小节开头
- 在相邻注释中写清楚“为什么必须先 attach 再 load”
- 保持现有时序不变

目标：
- 后续继续整理时，不会误把 attach/load 顺序打乱。
- 把“结构依赖”从隐性知识变成代码中的显式边界。

---

### Step P2-4 — 最小命名整理（低风险）

当前状态（2026-04-12）：最小版已完成。
- 已新增的 build-only infer helper：
  - `_infer_contact_plan_init_build_cfg(...)`
  - `_infer_event_clock_build_cfg(...)`
  - `_infer_lambda_fusion_build_cfg(...)`
  - `_resolve_direct_pose_leg_gate_build_cfg(...)`
- 当前仍不做大规模 rename；只保证局部区块命名和职责标题更易读。

只处理本次直接涉及的局部命名，使区块意图更清楚，例如：
- `state_dict` 继续表示“待加载的可变 ckpt state”
- `raw_state` 保持“原始 ckpt state”
- `removed` / `removed_shape` / `removed_highorder` 等变量尽量在各自局部语义闭环内使用

限制：
- 不做大规模 rename。
- 不顺手改其他无关逻辑命名。

---

## 3) 本阶段不做的事

Phase 2 明确不做：
- 不整理 `_cfg_*` / alias / reject 区块（那是 Phase 3）
- 不整理 runtime / rollout / training loop 区块（那是 Phase 4）
- 不修改 `main()` 编排
- 不抽到 `train/models.py` / `train/training_MPL.py`
- 不新增公共模块

---

## 4) 回归与验收标准

最低回归门：
1. `python3 -m py_compile train/posttrain.py`
2. `python3 -m train.posttrain --help`

若本轮改动触及 `_build_posttrain_model_from_ckpt(...)` 的执行顺序，建议补充：
3. 至少一次最小 ckpt 加载 smoke（使用现有可用 bundle / ckpt 组合）

每次提交建议记录：
1. `wc -l train/posttrain.py`
2. `rg -n "^def " train/posttrain.py | wc -l`
3. `_build_posttrain_model_from_ckpt(...)` 的区块顺序是否已满足：
   - infer
   - instantiate
   - pre-load compat/adapt/drop
   - load
   - post-load guard
   - artifacts return

本阶段验收标准：
- `_build_posttrain_model_from_ckpt(...)` 的 checkpoint 相关逻辑比当前更连续。
- 没有引入跨脚本依赖。
- 没有改变 CLI / config / ckpt 行为。
- 代码已为后续 Phase 3 / 4 预留清晰边界。

---

## 5) 后续建议

本文件当前更适合进入“验证 / 维护”阶段：
- 优先补更标准的 ckpt/config smoke（而不是继续重构）
- 如需继续改动，只做 very small 的净减法或注释澄清
- 若后续转去 `train/training_MPL.py`，建议单开独立 roadmap，不与 posttrain 混做
