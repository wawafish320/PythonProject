# [2026-04-12] `train/training_MPL.py` 单文件模块化整理路线图（Phase 1-4 已完成 / Phase 5 in progress）

Date: 2026-04-12  
Status: Active / Phase 4 completed, Phase 5 in progress  
Scope: `train/training_MPL.py`（只做单文件内整理，不跨文件迁移）  
Goal: 在**不改变训练语义**前提下，先把 `train/training_MPL.py` 整理成可维护的 layered monolith：入口编排清晰、builder 边界 typed 化、resume/export/debug 区块收口，再决定哪些逻辑值得跨脚本复用。  
Non-goal: 不改 CLI 参数名、不改 config key、不改 checkpoint 格式、不改 loss 数学定义、不改默认训练/验证行为、不提前抽成 `posttrain`/`pretrain` 公共模块。

---

## 0) 当前策略（先单文件分层，再决定是否跨文件）

当前结论：

1. **先只整理 `train/training_MPL.py` 内部结构**
   - 不急着拆新脚本。
   - 不急着做 shared util / shared builder。
2. **先把“边界”整理清楚，再考虑“物理拆分”**
   - 先让 parser / entry context / runtime config / dataset-model-loss builder / resume compat / postfit export
     在单文件内形成稳定边界。
   - 等这些边界稳定后，再判断哪些值得给 `train/posttrain.py` 或 `train/pretrain_mpl_min.py` 复用。
3. **优先做 typed boundary，不优先做跨文件复用**
   - 当前很多 builder 之间仍通过 `SimpleNamespace` 传递。
   - 这类边界如果在还不稳定时提前抽出去，容易把 MPL-specific 偶然形状固化成共享 API。
4. **`Trainer` 暂缓大拆**
   - `Trainer` 是最大结构债，但不是第一刀。
   - 先整理外围 entry/build/runtime 层，让 `train_entry()` 成为干净编排入口，再评估 `Trainer` 内部分层。

本阶段额外约束：
- 不新增公共 module。
- 不跨文件迁移逻辑。
- 不顺手统一 `posttrain` / `pretrain` 行为。
- 允许新增少量单文件 dataclass / helper，但必须直接服务于边界收口。

---

## 1) 当前状态与热点

当前工作树基线（2026-04-12）：
- LOC：`6768`
- 顶层 `def` 数：`31`
- `class Trainer` 区间：`train/training_MPL.py:483` 到 `train/training_MPL.py:4349`
- `Trainer` 体量：约 `3867` 行 / `99` 个方法
- `_build_train_parser(...)` 参数规模：`133` 个 `add_argument`
- `train_entry()` 长度：`50` 行
- `main()` 长度：`33` 行
- `py_compile`：本路线图编写时未额外执行；当前文档只记录结构计划

当前工作树（Phase 1 完成后）：
- LOC：`6884`（`+116`）
- 顶层 `def` 数：`33`（`+2`）
- 新增单文件 dataclass：`6`
- `py_compile`：`python3 -m py_compile train/training_MPL.py` 通过

当前工作树（Phase 2 完成后）：
- LOC：`6937`（相对 Phase 1 `+53`）
- 顶层 `def` 数：`39`（相对 Phase 1 `+6`）
- parser 主题 helper：`6`
- `_build_train_parser(...)` 长度：`17` 行
- `_build_train_parser(...)` 参数规模：`133` 个 `add_argument`（不变）
- `train_entry()` 长度：`50` 行（不变）
- 验证：
  - `python3 -m py_compile train/training_MPL.py` 通过
  - `python3 -m train.training_MPL --help` 通过
  - `--help` 输出 diff：无变化

当前工作树（Phase 3 完成后）：
- LOC：`6958`（相对 Phase 2 `+21`）
- 顶层 `def` 数：`43`（相对 Phase 2 `+4`）
- model runtime prep helper：`0 -> 4`
- `_build_train_model(...)` 长度：`78 -> 79` 行
- `_prepare_train_model_runtime(...)` 长度：`128 -> 25` 行
- 验证：
  - `python3 -m py_compile train/training_MPL.py` 通过

当前工作树（Phase 4 完成后）：
- LOC：`6978`（相对 Phase 3 `+20`）
- 顶层 `def` 数：`45`（相对 Phase 3 `+2`）
- 新增 postfit helper：`0 -> 2`
- `_run_postfit_actions(...)` 长度：`41 -> 26` 行（由原 `_run_postfit_validation_and_export(...)` 演进）
- `_export_postfit_onnx(...)` 长度：`38 -> 38` 行（不变）
- 验证：
  - `python3 -m py_compile train/training_MPL.py` 通过

当前工作树（Phase 5 当前增量）：
- LOC：`6978 -> 6978`（不变）
- 顶层 `def` 数：`45 -> 45`（不变）
- state / runtime helper 前导区块：`2444-2591 + 3492-3517` → `484-786`
- rollout / freerun 支撑区块：`1068-1693 + 2542-2591 + 3492-4210` → `788-2103`
- validation / metrics / fit / autoreg validate 连续区块：`3983-4238` → `2845-3477`
- diagnostics / whitebox 尾块：`507-958 + 1747-1963 + 2292-2386 + 4239-4347` → `3479-4347`
- train-step 邻接：旧 augment helper 后续已删除，`_run_one_train_batch(...)` 保持主入口
- `__init__(...)`：`2444 -> 484`
- fit helper 邻接：`_compute_fit_drift_slope(...)` `2542 -> 2849`，`_fit_checkpoint_payload(...)` `2568 -> 3345`
- `validate_autoreg_online(...)`：`3500 -> 2851`
- 验证：
  - `python3 -m py_compile train/training_MPL.py` 通过

Phase 1 已落地内容：
- builder/artifact 边界已从 `SimpleNamespace` 收口到单文件 dataclass：
  - `TrainEntryContext`：`train/training_MPL.py:5333`
  - `TrainDataArtifacts`：`train/training_MPL.py:5345`
  - `DirectPoseBuildOptions`：`train/training_MPL.py:5355`
  - `ResumeLoadReport`：`train/training_MPL.py:5363`
  - `TrainModelArtifacts`：`train/training_MPL.py:5374`
  - `TrainBuildArtifacts`：`train/training_MPL.py:5384`
- direct-pose build 配置已收口到 `_build_direct_pose_options(...)`：`train/training_MPL.py:5998`
- resume compat 已收口到 `_resume_load_weights_compat(...)`：`train/training_MPL.py:6021`
- 以下 builder 已切换为 typed 返回值：
  - `_build_train_components(...)`：`train/training_MPL.py:6105`
  - `_build_train_loaders(...)`：`train/training_MPL.py:6187`
  - `_build_train_model(...)`：`train/training_MPL.py:6225`
  - `_build_train_loss_and_trainer(...)`：`train/training_MPL.py:6431`

说明：
- 本轮 Phase 1 允许 LOC 小幅上升；目标是先建立稳定 typed boundary，而不是立即做净减法。
- 当前 `train_entry()` 外部行为、`Trainer(...)` 构造方式、checkpoint load 策略均保持不变。

当前已存在、但尚未完全收口的“天然 seam”：
- `TrainerRuntimeConfig`：`train/training_MPL.py:5275`
- `_build_train_parser(...)`：`train/training_MPL.py:5460`
- `_parse_train_entry_args(...)`：`train/training_MPL.py:5739`
- `_resolve_trainer_runtime_config(...)`：`train/training_MPL.py:5777`
- `_apply_trainer_runtime_config(...)`：`train/training_MPL.py:5935`
- `_build_train_components(...)`：`train/training_MPL.py:6105`
- `_build_train_loaders(...)`：`train/training_MPL.py:6187`
- `_build_train_model(...)`：`train/training_MPL.py:6225`
- `_prepare_train_model_runtime(...)`：`train/training_MPL.py:6312`
- `_build_train_loss_and_trainer(...)`：`train/training_MPL.py:6431`
- `_run_postfit_validation_and_export(...)`：`train/training_MPL.py:6565`
- `_export_postfit_onnx(...)`：`train/training_MPL.py:6606`
- `train_entry()`：`train/training_MPL.py:6644`

当前结构热点：

1. **Trainer / 训练核心过重**
   - `class Trainer` 同时承担 train loop、rollout、validation、contact runtime、history debug、metrics 汇总。
   - 当前最大的 Trainer 内部热点包括：
     - `_contact_meas_whitebox(...)`：约 `319` 行
     - `_run_epoch_validation(...)`：约 `154` 行
     - `_lambda_fusion_apply_reliability(...)`：约 `140` 行
     - `_run_one_train_epoch(...)`：约 `136` 行

2. **Parser / runtime schema 过大**
   - `_build_train_parser(...)` 自身约 `279` 行。
   - 目前 CLI 参数集合过大，语义仍主要分散在 parser 体内，而不是清晰的 config contract。

3. **Builder 链路第一轮已 typed 化，但高层编排仍可继续收口**
   - `_build_train_components(...)`
   - `_build_train_loaders(...)`
   - `_build_train_model(...)`
   - `_build_train_loss_and_trainer(...)`
   - Phase 1 已完成 dataclass 化，但 `train_entry()`、parser 与 runtime config 之间的编排仍可继续压平。

4. **resume compat 已 helper 化，但 postfit / export 仍在主训练文件内**
   - `resume` 兼容加载已收口为 `_resume_load_weights_compat(...)`：`train/training_MPL.py:6021`
   - postfit validate / ONNX export 仍位于训练入口尾部：`train/training_MPL.py:6565`、`train/training_MPL.py:6606`
   - 当前状态是“resume 已形成单点 helper，postfit/export 还未完成职责闭环”。

5. **入口层仍有全局状态痕迹**
   - `GLOBAL_ARGS` / `set_global_args(...)` / `_arg(...)` 仍参与部分运行时行为。
   - 这说明入口层与运行时配置仍有泄漏，但暂时不作为第一刀处理对象。

---

## 2) 总体目标（单文件 layered monolith）

本路线图的目标不是马上把 `train/training_MPL.py` 拆成多个文件，而是先让它具备以下特征：

1. **入口编排可顺读**
   - `train_entry()` 只保留高层 orchestration。
   - 读者能一眼看懂 parse → context → loaders → model → runtime → fit → postfit 的主链路。

2. **builder 边界 typed 化**
   - builder 之间优先传 dataclass，而不是 `SimpleNamespace`。
   - 这样即使暂时不跨文件，后续抽离时也不会把隐式字段扩散出去。

3. **resume / postfit / export 成为职责闭环**
   - 不一定物理搬走，但要在单文件内变成清晰 helper，而不是夹在 builder 内部的随机逻辑段。

4. **`Trainer` 先保持为单体，但外围噪声降低**
   - 先减少“入口 + builder + export + compat”对阅读面的干扰。
   - 再决定 `Trainer` 内部究竟按 rollout / eval / debug / fit 哪种方式继续整理。

---

## 3) Phase 1 — Artifact / Context 边界 typed 化（低风险，已完成）

这是第一刀，现已完成。

目标：
- 不改 builder 行为。
- 只把 builder 之间传递的 `SimpleNamespace` 收敛成 dataclass。
- 为后续 parser 分组、resume helper 化、postfit 收口打基础。

已落地的单文件 dataclass：
- `TrainEntryContext`
- `TrainDataArtifacts`
- `DirectPoseBuildOptions`
- `TrainModelArtifacts`
- `TrainBuildArtifacts`
- `ResumeLoadReport`

已覆盖的现有函数：
- `_build_train_components(...)`
- `_build_train_loaders(...)`
- `_build_train_model(...)`
- `_build_train_loss_and_trainer(...)`

已完成内容：
1. `TrainEntryContext`
   - 已替代 `_build_train_components(...)` 的 `SimpleNamespace`
   - 当前字段包括：`args` / `train_paths` / `run_name` / `out_dir` / `device` / `norm_template_path` / `norm_spec` / `pose_hist_len`
2. `TrainDataArtifacts`
   - 已替代 `_build_train_loaders(...)` 返回值
   - 当前字段包括：`ds_train` / `train_loader` / `pin_memory` / `dx` / `dy` / `dc`
3. `DirectPoseBuildOptions`
   - 已收拢 `_build_train_model(...)` 中 direct-pose arm/split/nonleg 相关解析
   - `EventMotionModel(...)` 调参行为保持不变
4. `TrainModelArtifacts`
   - 已替代 `_build_train_model(...)` 返回值
   - 当前包含 `model` 与 direct-pose / history 相关 build 结果
5. `TrainBuildArtifacts`
   - 已替代 `_build_train_loss_and_trainer(...)` 返回值
   - 当前包含 `model` / `loss_fn` / `trainer` / `bundle_json_path` / `resolved_config`
6. `ResumeLoadReport`
   - 已用于 `_resume_load_weights_compat(...)`
   - 作为 Phase 3 进一步收口 resume / guard 顺序的基础结构

本阶段约束：
- 不更改 `train_entry()` 的外部行为。
- 不更改 checkpoint load 策略。
- 不更改 `Trainer` 构造参数签名。
- 不改日志 key。

Phase 1 指标（before/after）：
- LOC：`6768 -> 6884`（`+116`）
- 顶层 `def` 数：`31 -> 33`（`+2`）
- 单文件新增 dataclass：`0 -> 6`
- builder 主返回值中的 `SimpleNamespace`：`4 -> 0`

Phase 1 验收：
- builder 主返回值已不再使用 `SimpleNamespace`
- 类型边界已清晰
- `train_entry()` 调用链已可通过 dataclass 字段直读
- `python3 -m py_compile train/training_MPL.py` 已通过

---

## 4) Phase 2 — Parser / Entry Context 分组（低/中风险）

目标：
- 不改 CLI 参数集合。
- 只把 `_build_train_parser(...)` 的内部组织从“长串 add_argument”整理成按主题分组的单文件 helper。

建议边界：
- `_parser_add_io_args(...)`
- `_parser_add_data_args(...)`
- `_parser_add_model_args(...)`
- `_parser_add_loss_args(...)`
- `_parser_add_runtime_args(...)`
- `_parser_add_debug_export_args(...)`

保持不变的内容：
- `_load_train_entry_config_defaults(...)`
- `_apply_train_entry_config_overrides(...)`
- `_parse_train_entry_args(...)`

为什么 Phase 2 先不跨文件：
- 当前 parser 仍强绑定 `training_MPL` 的语义和 flag 组合。
- 如果现在抽到公共 parser module，后续很容易变成“为了共享而共享”。

本阶段验收：
- `_build_train_parser(...)` 仍是唯一组装入口
- 参数语义与默认值完全不变
- `--config_json` 与 `--config_override` 优先级保持不变
- parser 主体可通过主题 helper 快速跳读

Phase 2 已落地内容：
1. `_build_train_parser(...)` 已压缩为单一组装入口
   - 仅保留 `config_parser` 创建与 helper 编排
   - `--config_json` 默认值加载链路未改
2. 新增 6 个单文件 helper，并按主题承载原有 `add_argument(...)`
   - `_parser_add_io_args(...)`
   - `_parser_add_runtime_args(...)`
   - `_parser_add_loss_args(...)`
   - `_parser_add_model_args(...)`
   - `_parser_add_data_args(...)`
   - `_parser_add_debug_export_args(...)`
3. `_load_train_entry_config_defaults(...)` / `_apply_train_entry_config_overrides(...)` / `_parse_train_entry_args(...)` 保持原行为
4. parser 元数据已做精确对照
   - 基于 `HEAD` 原版与当前版本的 action snapshot 比较，`133/133` 参数定义 **exact match**
   - `--help` 输出 diff 结果为无变化

Phase 2 指标（before/after）：
- LOC：`6884 -> 6937`（`+53`）
- 顶层 `def` 数：`33 -> 39`（`+6`）
- `_build_train_parser(...)` 长度：`279 -> 17`（`-262`）
- parser 主题 helper：`0 -> 6`
- `_build_train_parser(...)` 参数规模：`133 -> 133`（不变）
- `train_entry()` 长度：`50 -> 50`（不变）

Phase 2 备注：
- `adaptive_bone_weights` / `unified_*` / `rot_local_tail_*` 的主题归属存在一定歧义。
- 本轮将它们保留在 `_parser_add_data_args(...)` 所在的后置区块，以尽量贴近原始参数注册顺序并降低 help/parse 漂移风险。

---

## 5) Phase 3 — Model Build / Resume Compat / Runtime Guard 收口（中风险）

目标：
- 让 `_build_train_model(...)` 更像真正的 model builder
- 把 resume compat、build-time direct-pose 解析、post-build guard 从主干中收口

建议分步：

### Step P3-1 — 收拢 direct-pose build 配置

把当前分散在 `_build_train_model(...)` 头部的：
- `direct_pose_split_enable`
- `direct_pose_arm_split_enable`
- `direct_pose_arm_bones_resolved`
- `direct_pose_nonleg_proj_dim`

整理为：
- `_build_direct_pose_options(args) -> DirectPoseBuildOptions`

目标：
- 让 `EventMotionModel(...)` 实例化参数准备更集中
- 降低 `_build_train_model(...)` 前半段的噪声

### Step P3-2 — 收拢 resume compat

把当前 `_build_train_model(...)` 中的 checkpoint 兼容加载逻辑整理为：
- `_resume_load_weights_compat(model, resume_path) -> ResumeLoadReport`

保持不变：
- `adapt_legacy_state_dict_`
- shape match 过滤
- `strict=False`
- warn / fallback 行为

目标：
- 让 “build model” 与 “load compatible weights” 成为相邻但职责独立的区块
- 为后续是否给其他入口复用，保留判断空间

### Step P3-3 — 收拢 build 后 guard

把当前模型构建完成后的：
- first-linear finite guard
- dataset fps / pasa 相关 runtime 注入
- history adaptive runtime attach

按 “build only / post-build runtime prep / guard” 顺序排成连续区块。

本阶段验收：
- model build/runtime 链路顺序清晰：
  - `_build_train_model(...)`：resolve options → instantiate model → build artifacts return
  - `_prepare_train_model_runtime(...)`：adaptive/runtime attach → resume compat → post-resume finite guard → dataset runtime attach

Phase 3 已落地内容：
1. `_build_train_model(...)` 已进一步保持为纯 build 区块
   - direct-pose 仍通过 `_build_direct_pose_options(args)` 解析
   - `EventMotionModel(...)` 参数被收口到 `model_kwargs` 后一次实例化
   - `TrainModelArtifacts` 返回字段不改语义，只继续承载 history/direct-pose build 结果
2. `_prepare_train_model_runtime(...)` 已压缩为顺读的 runtime prep 编排
   - `_attach_adaptive_history_runtime(...)`
   - `_sanitize_train_model_post_build(...)`
   - `_prepare_motion_encoder_and_contacts_runtime(...)`
   - `_resume_load_weights_compat(...)`
   - `_guard_first_linear_finite_(...)`
   - dataset `_pasa_fps` runtime attach
3. resume compat 路径保持单点 helper
   - `adapt_legacy_state_dict_`
   - shape-match filter
   - `strict=False`
   - warn/fallback behavior
4. build 后 guard 已从 runtime 主体中命名收口
   - `validate_and_fix_model_(model, dx, dc)` 与 `validate_and_fix_model_(model)` 的执行顺序保持不变
   - first-linear finite reinit/assert 行为保持不变

Phase 3 指标（before/after）：
- LOC：`6937 -> 6958`（`+21`）
- 顶层 `def` 数：`39 -> 43`（`+4`）
- `_build_train_model(...)` 长度：`78 -> 79` 行（以 `model_kwargs` 明确 resolve options / instantiate 边界）
- `_prepare_train_model_runtime(...)` 长度：`128 -> 25` 行
- 新增 runtime prep helper：`0 -> 4`

Phase 3 备注：
- 为保持 `train_entry()` 现有调用方式，`TrainModelArtifacts` 仍在 `_build_train_model(...)` 完成 pure build 后返回；runtime prep 随后对同一个 `model` 做 in-place attach/resume/guard，本轮不把 `_prepare_train_model_runtime(...)` 内联回 builder。
- `_prepare_motion_encoder_and_contacts_runtime(...)` 同时保留 MotionEncoder bundle attach 与 contact source resolution，因为 `contact_plan_enable` 的 fatal guard 依赖 bundle attach 后的 frozen contact head；为保持行为，本轮不拆成两个更细 helper。
- dataset `_pasa_fps` attach 仍保留在 `_prepare_train_model_runtime(...)` 尾部，作为最后一个 dataset/runtime 注入动作。

---

## 6) Phase 4 — Postfit / Export 区块收口（中风险，已完成）

目标：
- 保持仍在 `train/training_MPL.py` 内
- 但把训练结束后的行为整理成单独职责带

建议边界：
- `_run_postfit_actions(...)`
  - teacher / freerun validate
  - best ckpt reload（若保留）
  - ONNX export
- `_export_postfit_onnx(...)`
- `export_onnx_step_stateful_nophase(...)`

需要特别注意：
- `train/export_onnx_from_ckpt.py` 当前仍依赖 `train.training_MPL` 中的 `export_onnx_step_stateful_nophase(...)`
- 本路线图阶段内**不改变这一外部依赖关系**
- 先把 `training_MPL.py` 内部结构整理稳定，再决定是否迁移导出接口

本阶段验收：
- `train_entry()` 尾部不再混入 postfit 细节
- ONNX export 相关逻辑在单文件内连续可定位
- 不改导出签名、不改外部脚本调用方式

Phase 4 已落地内容：
1. `train_entry()` 尾部继续保持单一 orchestration
   - `fit(...)` 后仅调用 `_run_postfit_actions(...)`
   - postfit validate / reload / export 不再散落在入口尾部
2. postfit validate / reload 已拆成明确 helper
   - `_run_postfit_valfree_eval(...)`
   - `_try_reload_postfit_best_ckpt(...)`
   - `_run_postfit_actions(...)`
3. `_export_postfit_onnx(...)` 保持导出逻辑单点
   - 导出签名保持不变
   - `export_onnx_step_stateful_nophase(...)` 仍留在原位置，外部依赖关系不变
4. best checkpoint reload 语义保持不变
   - 仅在 postfit valfree 失败时尝试 strict reload `best_ckpt`
   - reload 成功/失败/缺文件的日志行为保持一致

Phase 4 指标（before/after）：
- LOC：`6958 -> 6978`（`+20`）
- 顶层 `def` 数：`43 -> 45`（`+2`）
- postfit 细分 helper：`0 -> 2`
- `_run_postfit_validation_and_export(...)`：`41` 行 → `_run_postfit_actions(...)`：`26` 行
- `_export_postfit_onnx(...)`：`38 -> 38` 行（不变）

Phase 4 备注：
- `train_entry()` 总长度略有增加（`51 -> 53`），主要来自函数间空行与尾部 helper 调用的排版，不涉及训练或导出语义变化。
- 本轮没有把 `_export_postfit_onnx(...)` 再拆为 dim-probe/export 两段 helper，优先维持 ONNX 导出逻辑连续可定位。

---

## 7) Phase 5 — `Trainer` 内部顺序整理（高风险，最后做）

这是最后做的步骤。

当前判断：
- `Trainer` 是本文件最大结构债，但在外围边界收口之前，不适合先做大拆。
- 先整理 entry/build/runtime 层，可以显著降低 `Trainer` 拆分时的 blast radius。

建议先做“区块顺序整理”，不急着做多类 / 多文件：

1. **state / runtime helper**
2. **rollout / forward step**
3. **train step / epoch fit**
4. **validation / metrics**
5. **diagnostics / debug / whitebox**

当前已知大块热点：
- `_contact_meas_whitebox(...)`
- `_run_epoch_validation(...)`
- `_lambda_fusion_apply_reliability(...)`
- `_run_one_train_epoch(...)`
- `_apply_stage_schedule(...)`
- `_history_drift_debug(...)`

本阶段优先原则：
- 先重排与命名澄清
- 再考虑是否引入小 helper
- 暂不做多文件 mixin 化

Phase 5 当前进展：
1. 已先做一刀低风险“validation / metrics 邻接重排”
   - `_run_epoch_validation(...)`
   - `_persist_epoch_validation_outputs(...)`
   - `_update_best_ckpts(...)`
   - `_metrics_json_safe(...)`
   - epoch metrics runtime hooks（后续已删除）
   - `_persist_epoch_metrics_artifacts(...)` / `_dump_metrics_json(...)`
   - `_write_basetrain_keybone_group_summary(...)`
   - `_save_val_metrics(...)`
   - `fit(...)`
2. 已完成 rollout / freerun / diagnostics 第一轮邻接重排
   - rollout 前置 helper 已收口到连续区块：
     - `_diag_norm_x(...)` / `_format_template_hint(...)` / `_require_normalizer(...)` / `_raise_norm_error(...)`
     - `_commit_rollout_diag_update(...)`
     - `_denorm(...)` / `_cached_norm_param(...)` / `_norm_y(...)`
     - `_prepare_cond_stat(...)` / `_normalize_cond_from_raw(...)`
     - `_pose_hist_params(...)` / `_infer_root_yaw_from_rot6d(...)`
     - `_compose_delta_to_raw(...)`
     - `_lambda_fusion_apply_reliability(...)` / `_apply_lambda_fusion_to_raw(...)`
     - `_reproject_cond_to_local_frame(...)` / `_apply_free_carry(...)`
   - rollout 核心区块保持连续：
     - `_prepare_pose_hist_state(...)`
     - `_resolve_rollout_step_inputs(...)`
     - `_update_rollout_carry_state(...)`
     - `_init_rollout_state(...)`
     - `_get_rollout_step_tensor(...)`
     - `_update_rollout_plan_state(...)`
     - `_record_rollout_step_outputs(...)`
     - `_compute_rollout_step_debug_stats(...)`
     - `_rollout_forward_step(...)`
     - `_apply_scheduled_sampling_update(...)`
     - `_rollout_sequence(...)`
     - `_freerun_traj_loss(...)`
   - train-step 辅助旧 augment helper 后续已删除
   - diagnostics / debug 尾块已收口为：
     - `_history_drift_debug(...)`
     - `_joint_group_masks(...)`
     - `_summarize_angvel_dir(...)`
     - `test_gradient_connection(...)`
     - `_diagnose_free_run(...)`
     - `_dump_nan_grad_report(...)`
3. 本轮继续完成 state / runtime helper 与 whitebox 收口
   - `__init__(...)` 已提到 `Trainer` 最前段，`_pick_first(...)` 也回收到 state/runtime helper 区
   - `_resolve_contact_meas_runtime(...)` / `_resolve_contact_meas_cfg(...)` / `_resolve_contact_meas_foot_indices(...)` / `_contact_meas_whitebox(...)` 已并入 diagnostics / whitebox 尾块
   - `_predict_pretrain_contacts_from_frozen(...)` 保持留在 rollout 支撑区，不与 whitebox 混放
4. 本轮继续完成 fit / validation seam 收口
   - `_compute_fit_drift_slope(...)` 已移到 `eval_epoch(...)` / `_run_epoch_validation(...)` 邻域
   - `_fit_checkpoint_payload(...)` / `_save_fit_checkpoint_payload(...)` 已移到 `fit(...)` 前，checkpoint 持久化 helper 不再滞留在 train-step 前段
   - `validate_autoreg_online(...)` 已并回 validation 区块，紧邻 `eval_epoch(...)` / `_run_epoch_validation(...)`
5. 当前效果
   - rollout 核心与其 normalizer / cond / yaw / lambda / carry 依赖已成为连续阅读区块
   - train step 主路径围绕 `_run_one_train_batch(...)` 连续阅读
   - `Trainer` 开头更接近 roadmap 的 `state / runtime helper` 入口
   - validation / metrics / fit / autoreg validate / diagnostics / whitebox 的顺序更接近最终目标
   - `fit(...)` 主循环与其 drift/checkpoint helper 的阅读跳转更少
   - autoreg online validate 不再悬挂在 diagnostics seam 上
6. 明确保持不变
   - 仅重排 `Trainer` 内方法位置
   - 不改 fit / validation / metrics 的数据流、日志 key、checkpoint 选择逻辑
   - 不改 rollout / scheduled sampling / free-run / teacher 模式语义
   - 不改 `_rollout_forward_step(...)`、`_rollout_sequence(...)`、`validate_autoreg_online(...)`、`_diagnose_free_run(...)`、`_contact_meas_whitebox(...)` 的行为

Phase 5 当前指标（before/after）：
- LOC：`6978 -> 6978`（不变）
- 顶层 `def` 数：`45 -> 45`（不变）
- state / runtime helper 前导区块：`2444-2591 + 3492-3517` → `484-786`
- rollout / freerun 支撑区块：`1068-1693 + 2542-2591 + 3492-4210` → `788-2103`
- `Trainer` validation / metrics / fit / autoreg validate 连续区块：`3983-4238` → `2845-3477`
- diagnostics / whitebox 尾块：`507-958 + 1747-1963 + 2292-2386 + 4239-4347` → `3479-4347`
- train-step 邻接：旧 augment helper 后续已删除，`_run_one_train_batch(...)` 为连续主入口
- `__init__(...)`：`2444 -> 484`
- fit helper 邻接：`_compute_fit_drift_slope(...)` `2542 -> 2849`，`_fit_checkpoint_payload(...)` `2568 -> 3345`
- `validate_autoreg_online(...)`：`3500 -> 2851`

Phase 5 当前备注：
- 本轮只做 `Trainer` 内部顺序整理，不新增 helper，不触碰 train loop 语义。
- `_contact_meas_whitebox(...)` 已下沉到 diagnostics / whitebox 尾块；与之强绑定的 contact runtime helper 同步移动。
- `_predict_pretrain_contacts_from_frozen(...)` 仍留在 rollout 支撑区，因为它直接参与 basetrain rollout contact resolution，不属于 debug-only whitebox。

---

## 8) 当前明确不做的事

本路线图明确不做：
- 不新建 `train/training_entry.py`、`train/builders.py`、`train/export.py`
- 不把 `training_MPL` 逻辑提前抽给 `posttrain` / `pretrain`
- 不修改 `GLOBAL_ARGS` / `set_global_args(...)` 兼容路径（除非某一步整理已被其直接阻塞）
- 不修改 `Trainer(..., args=args)` 的外部构造方式
- 不修改 loss / rollout / eval 数学定义
- 不统一 validate / posttrain / pretrain 的 shared API

---

## 9) 什么时候才允许跨文件抽离

只有同时满足以下条件，才建议从本路线图切换到“跨文件抽离”：

1. 已经存在**第二个真实调用方**
   - 不是“未来可能复用”，而是 `posttrain` / `pretrain` / validate 脚本已明确需要同一逻辑
2. 接口已经 typed 化
   - 输入输出不再依赖 `SimpleNamespace`
   - 尽量不直接暴露裸 `argparse.Namespace`
3. helper 不再依赖入口侧副作用
   - 不依赖 `GLOBAL_ARGS`
   - 不依赖 `sys.argv` 改写
   - 不依赖训练入口临时 print / path side effect

如果以上条件不满足，则继续留在单文件内整理更稳。

---

## 10) 回归与验收标准

最低回归门（建议每步至少执行）：
1. `python3 -m py_compile train/training_MPL.py`
2. `python3 -m train.training_MPL --help`

若某步触及 builder / resume / export，建议补充：
3. 最小 `--help` / parser 语义 spot-check
4. 需要时执行一次最小 smoke（仅当本地已有稳定数据与模板）

每次提交建议记录的结构指标：
1. `wc -l train/training_MPL.py`
2. `rg -n "^def " train/training_MPL.py | wc -l`
3. `class Trainer` 行数与方法数
4. `_build_train_parser(...)` 的 `add_argument` 数
5. `train_entry()` 长度

本路线图阶段性验收标准：
- `train_entry()` 成为顺读的 orchestration
- builder 之间优先使用 dataclass 传递
- resume compat 成为单点逻辑区块
- postfit / export 在单文件内形成连续职责边界
- 整个整理过程不引入跨文件依赖，不改变训练行为

---

## 11) 后续预留

后续如果继续推进，可在本文件追加：
- Phase 2 执行记录（parser 分组）
- Phase 3 执行记录（resume / model build 收口）
- Phase 4 执行记录（postfit / export 收口）
- Phase 5 执行记录（`Trainer` 内部顺序整理）
