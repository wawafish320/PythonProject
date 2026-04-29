# [2026-04-18] `posttrain.py` / `training_MPL.py` 联合 zone map

Date: 2026-04-18  
Status: Draft / executable boundary map  
Scope: `train/posttrain.py`, `train/training_MPL.py`  
Pass contract:

- 本轮只建立后续拆分的职责边界，不正式拆文件。
- 本轮不改训练语义，不改 checkpoint contract / compat。
- 本轮不动 `train/models.py`、`train/pretrain_mpl_min.py`。
- 本轮不做 reorder-only diff；原因是两个文件都在 `4k+` 行量级，单纯重排会制造大面积无语义 diff，降低后续真实拆分的 review signal。

---

## 0. 当前进度

| 项目 | 状态 | 说明 |
|---|---|---|
| `commonization_execution_plan` | ✅ landed | `train/configuration/norm_spec.py` 已落地，`posttrain` / `training_MPL` 已接入。 |
| `runtime_attach` shared-half | ✅ landed | `train/runtime_attach.py` 已被 basetrain / posttrain 共同调用；shared helper 现已同时承接 trainer runtime attach 与 `loss_fn` runtime sync。 |
| `runtime_attach` overlay-half | ⏳ pending | overlay 仍留在 caller-local adapter / helper，尚未提升为 shared contract。 |
| neutral contact-pretrain runtime attrs | ✅ landed | basetrain / posttrain owner 继续保留各自前缀 attrs，同时共同写 `contacts_pretrain_*` 中性 attrs；shared `Trainer` contact-pretrain 读取已切到中性 contract；已解析 payload 内部统一为 `ContactPretrainRuntime`，owner prefix 只在最终 attach 点使用；`tools/run_training_mpl_entry_shell_smoke.py` 与 `tools/run_posttrain_runtime_overlay_smoke.py` 已覆盖。 |
| Step 2 shell/core exposure | ✅ landed | `training_MPL.py` 已把 basetrain entry shell 收成显式 shell band；runtime attach helpers 保留在 file-local core side，`train_entry()` 中的 dataset-runtime attach / loss sync 已收口为 thin shell helper；`tools/run_training_mpl_entry_shell_smoke.py` + `debug_output/_training_mpl_entry_shell_smokes_20260418/training_mpl_entry_shell_smoke_summary.json` 已通过。 |
| Step 3 posttrain build shell | ✅ landed | `train/posttrain_build_shell.py` 已承接 checkpoint rebuild shell；`tools/run_posttrain_build_shell_smoke.py` + `debug_output/_posttrain_build_shell_smokes_20260418/posttrain_build_shell_smoke_summary.json` 已覆盖 direct/lambda build-shell smoke；tools-side build-shell rewiring 已完成，`train/posttrain.py` 顶层 re-export 子集已删除。 |
| Step 4 rollout kernel extraction | ✅ thin-slice landed | `train/rollout_kernel.py` 已承接 shared low-level rollout DTO / state-init / input-prepare / carry / buffer helpers；basetrain fit/eval shell 与 posttrain objective 继续留在原文件。 |

---

## 1. 联合视图

### 1.1 联合系统结论

`train/posttrain.py` 和 `train/training_MPL.py` 不是两个互不相关的大脚本；更准确地说，它们是同一套 motion runtime 的两层：

- `train/training_MPL.py` 提供 **base train entry + generic Trainer rollout/runtime substrate**。
- `train/posttrain.py` 提供 **checkpoint-derived model rebuild + posttrain-only policy/loss loop**。

因此后续拆分不能按“先把两个文件分别均匀切块”的方式做；必须先围绕下面这些联合 seam 建边界：

1. `Trainer` runtime / rollout vocabulary 是共享底座。  
2. `DatasetRuntimeArtifacts -> trainer/loss runtime attach` 是共享 wiring。  
3. `pose history / cond normalization / contact-pretrain hydration` 是事实共享状态面。  
4. checkpoint build/load orchestration 只在低层 helper 上共享，高层流程并不共享。  
5. posttrain 的 lambda/direct objective 是 **policy layer**，不是 basetrain 的 generic rollout layer。  

### 1.2 shared seam / mirrored flow

| Seam | 位置锚点 | 结论 | 为什么是联合 seam |
|---|---|---|---|
| Dataset / loader / runtime attach | `train/posttrain.py:3125`, `train/posttrain.py:3162`, `train/training_MPL.py:3417`, `train/training_MPL.py:3461`, `train/training_MPL.py:3850` | 强 shared seam | 两边都走 `merge_norm_spec` → `build_motion_dataset` → `build_motion_dataloader` → `build_and_attach_dataset_runtime`，只是 entry policy 不同。 |
| Trainer runtime attach | `train/posttrain.py:3162`, `train/training_MPL.py:2830`, `train/training_MPL.py:3291`, `train/training_MPL.py:3375` | 强 shared seam | 两边都把 normalizer / pose history / contact-pretrain / diagnostics 绑定到同一个 `Trainer` 运行时对象上。 |
| Rollout state vocabulary | `train/posttrain.py:769`, `train/posttrain.py:899`, `train/posttrain.py:1037`, `train/training_MPL.py:1111`, `train/training_MPL.py:1271`, `train/training_MPL.py:1702`, `train/training_MPL.py:1779` | 镜像 seam | 两边都围绕 cond、contacts、pose_hist、time_index、carry state、output buffers 运作；差别在于高层 objective 和 loop policy。 |
| Pose history state | `train/posttrain.py:873`, `train/posttrain.py:1037`, `train/training_MPL.py:1111`, `train/training_MPL.py:3291` | 强 shared seam | 都依赖 `PoseHistState`/runtime tensors，只是一个在 free functions 中推进，一个在 `Trainer` methods 中推进。 |
| Contact-pretrain hydration | `train/posttrain.py:3200`, `train/training_MPL.py:3318` | 强 shared seam | 已经共享 `resolve_contact_pretrain_runtime(...)`，只是字段前缀不同。 |
| Checkpoint compat helpers | `train/posttrain_build_shell.py:108`, `train/posttrain_build_shell.py:346`, `train/training_MPL.py:3581`, `train/training_MPL.py:3656` | 低层 shared seam | 两边都调用 compat / bundle attach helper，但高层 build path 并不相同。 |

### 1.3 单边入口，不应提前抽成公共层

这些逻辑虽然体量大，但目前仍应保持 local ownership：

- `train/posttrain.py` 的 `PostTrainConfig`、`_cfg_*`  
  - 原因：它们服务的是 **posttrain entry/config policy**，不是 basetrain 的构建入口。
- `train/posttrain_build_shell.py` 的 `_resolve_posttrain_model_build_state(...)`、`_instantiate_posttrain_model(...)`、`_load_posttrain_checkpoint_into_model(...)`、`_build_posttrain_model_from_ckpt(...)`
  - 原因：它们服务的是 **checkpoint-derived rebuild shell**，属于 posttrain-only build orchestration，不是 basetrain 的构建入口。
- `train/posttrain.py` 的 `_lambda_fusion_loss_rollout(...)` 及其 `_lambda_*` family  
  - 原因：这里是 **posttrain-only policy/objective**，虽然重用了 rollout vocabulary，但不该被误抽成 generic runtime。
- `train/training_MPL.py` 的 parser/defaults、`train_entry()`, `Trainer.fit(...)`, teacher eval、metrics artifacts、ONNX export  
  - 原因：这是 basetrain shell，posttrain 不消费这些行为。
- `train/training_MPL.py` 的 stage schedule / plateau / summary panel  
  - 原因：它们是 basetrain training-management 逻辑，不是 posttrain shared seam。

### 1.4 keep-local / do-not-extract areas

本轮和下一轮 early split 都不建议抽公共层的区域：

- `train/posttrain.py:2440` `_lambda_fusion_loss_rollout(...)` 整段高层目标定义
- `train/posttrain.py:2920` `_run_training_loop(...)` 的 step-snapshot / L2-SP / grad probe 逻辑
- `train/training_MPL.py:2788` `Trainer.fit(...)` 的 epoch management / validation / checkpoint finalize
- `train/training_MPL.py:3803` `_export_postfit_onnx(...)`
- `train/posttrain_build_shell.py:108` 与 `train/training_MPL.py:3497` 以上的高层 model build orchestration

这些区域看起来都“可以拆”，但它们不是 **shared seam**；如果过早抽公共层，后面大概率还会再拆回去。

---

## 2. 逐文件 zone map

### 2.1 `train/posttrain.py`

| Zone | 主要函数 / 入口 | 职责 | 依赖的外部模块 | 依赖的内部状态 | 与 `training_MPL.py` 的耦合点 |
|---|---|---|---|---|---|
| Entry / CLI shell | `train/posttrain.py:3598`, `train/posttrain.py:3964` | 解析 CLI + config，选择 posttrain mode，串起 dataset/build/runtime/train/save 整体流程。 | `argparse`, `train.configuration.io`, `train.utils` | `args`, `payload`, `cfg`, `train_mode`, `device` | 最终实例化并消费 `Trainer`，但不共享 basetrain entry shell。 |
| Config resolution | `train/posttrain.py:426`, `train/posttrain.py:724`, `train/posttrain.py:2697`, `train/posttrain.py:2704`, `train/posttrain.py:2908` | 解析 JSON/CLI 为 `PostTrainConfig`，并派生 rollout/direct/lambda mode 配置。 | `train.utils`, `pathlib`, `argparse` | `PostTrainConfig`, 各类 rollout/direct_pose 字段 | 与 basetrain 的 arg schema 有概念重叠，但 contract 不同，不应直接合并。 |
| Rollout step adapters | `train/posttrain.py:769`, `train/posttrain.py:824`, `train/posttrain.py:850`, `train/posttrain.py:873`, `train/posttrain.py:899` | 把 batch/sequence/runtime 状态转成每一步 rollout 输入，维护 carry state 与 pose history。 | `train.history`, `train.geometry`, `train.training_MPL.Trainer` | `state` dict, `PoseHistState`, `meas_logits_prev`, `cond_raw_prev` | 和 `Trainer` 的 rollout methods 共享同一套 runtime vocabulary，是后续低层抽取的核心 seam。 |
| Posttrain-only policy / objective | `train/posttrain.py:1037`, `train/posttrain.py:1204`, `train/posttrain.py:1277`, `train/posttrain.py:1759`, `train/posttrain.py:1973`, `train/posttrain.py:2178`, `train/posttrain.py:2440` | 定义 lambda/direct posttrain rollout loss、leg-align、gate supervision、EMA payload 等。 | `train.geometry`, `torch`, `train.models.EventMotionModel` | `runtime_ctx`, `weights_ctx`, `accum_ctx`, `state_vars`, `aux_payload` | 强依赖 `Trainer` 暴露的 runtime attr / helper 行为与 model output schema；但逻辑本身不应进入 basetrain 公共层。 |
| Training loop / diagnostics / artifacts | `train/posttrain.py:2793`, `train/posttrain.py:2920`, `train/posttrain.py:3101` | 自定义 optimizer loop、step snapshot、grad monitor、log rows、最终 ckpt/log 导出。 | `torch`, `train.data.io`, `train.utils` | `global_step`, `log_rows`, `save_step_set`, `l2sp_pairs` | 复用 `Trainer` 运行时属性，但不走 `Trainer.fit(...)`。 |
| Dataset / runtime wiring | `train/posttrain.py:3125`, `train/posttrain.py:3162` | 构建 dataset/loader，attach dataset runtime，把 normalizer/contact-pretrain 等注入 `Trainer` / `loss_fn`。 | `train.configuration.norm_spec`, `train.data.dataset`, `train.models`, `train.runtime_attach`, `train.training_MPL.Trainer` | `norm_spec`, `dataset_artifacts`, trainer attrs (`posttrain_contacts_pretrain_*`, neutral `contacts_pretrain_*`, `lambda_reliability_*`) | 这是与 basetrain 最稳定的共享 wiring seam；`apply_contacts_pretrain_runtime(...)` 之后，owner dual-write 已集中化。 |
| Checkpoint / build-state shell | `train/posttrain_build_shell.py:108`, `train/posttrain_build_shell.py:269`, `train/posttrain_build_shell.py:346`, `train/posttrain_build_shell.py:441` | 从 checkpoint 推断 build-state，实例化 model，应用 strict contract/load 校验和 selective runtime guard。 | `train.checkpoint.load_schema`, `train.checkpoint.contract`, `torch.load`, `train.models` | `PostTrainModelBuildState`, `state_dict`, `ckpt_posttrain_cfg`, inferred head/runtime dims | 与 basetrain 共享低层 load-schema / contract helper，但高层流程必须 keep-local。 |

### 2.2 `train/training_MPL.py`

| Zone | 主要函数 / 入口 | 职责 | 依赖的外部模块 | 依赖的内部状态 | 与 `posttrain.py` 的耦合点 |
|---|---|---|---|---|---|
| Rollout DTOs / buffers | `train/training_MPL.py:124`, `train/training_MPL.py:164`, `train/training_MPL.py:198`, `train/training_MPL.py:229` | 定义 rollout 输入、执行状态、prediction buffers、fit checkpoint state 等 runtime carriers。 | `dataclasses`, `torch`, `train.history.PoseHistState` | `RolloutExecutionState`, `RolloutPredictionBuffers` | posttrain 自己没有同等 dataclass，但其 free-function rollout 逻辑复用了相同概念边界。 |
| Shared Trainer rollout kernel | `train/training_MPL.py:382`, `train/training_MPL.py:1111`, `train/training_MPL.py:1271`, `train/training_MPL.py:1314`, `train/training_MPL.py:1592`, `train/training_MPL.py:1702`, `train/training_MPL.py:1779` | generic rollout state machine：准备 cond/contact/pose_hist/time inputs，运行 model step，更新 carry state，汇总 outputs。 | `train.geometry`, `train.history`, `train.diagnostics`, `train.data.normalizers` | `Trainer` attrs：normalizer、pose_hist_*、lambda_reliability_*、teacher_forcing_ratio、diag state | posttrain 直接 import `Trainer`，并在自身 rollout helpers 中假设这些 runtime attr / 行为存在。 |
| Trainer fit / validation / metrics / ckpt | `train/training_MPL.py:1849`, `train/training_MPL.py:2397`, `train/training_MPL.py:2503`, `train/training_MPL.py:2720`, `train/training_MPL.py:2788` | basetrain epoch loop、stage schedule、teacher eval、metrics JSON、fit checkpoint payload。 | `evaluate_teacher`, `train.diagnostics`, `time`, `torch` | optimizer, schedulers, `_cached_train_batch`, summary rows, `full_config` | posttrain 不使用这层 orchestration；这块是 basetrain-only shell。 |
| Runtime config attach | `train/training_MPL.py:2830`, `train/training_MPL.py:3291`, `train/training_MPL.py:3375` | 解析 `DatasetRuntimeArtifacts` + args，统一 attach 到 `Trainer`。 | `train.configuration.norm_spec`, `train.history`, `DatasetRuntimeArtifacts` | `TrainerRuntimeConfig`、attr rename map、field groups | 与 posttrain `_build_model_and_trainer(...)` 形成最明确的 mirrored structure。 |
| Entry / CLI / defaults | `train/training_MPL.py:2917`, `train/training_MPL.py:2987`, `train/training_MPL.py:3232`, `train/training_MPL.py:3246` | basetrain CLI、defaults merge、参数解析。 | `argparse`, `json`, `ast` | `args`, parser defaults, `TRAIN_ENTRY_CONFIG_META_KEYS` | 和 posttrain 一样有 entry shell，但 contract/flag surface 不相同。 |
| Dataset / loader shell | `train/training_MPL.py:3417`, `train/training_MPL.py:3461` | 解析 train paths、build dataset/loader、创建 run dir / device / norm spec。 | `glob`, `os`, `pathlib`, `train.data.dataset`, `train.configuration.norm_spec` | `TrainEntryContext`, `TrainDataArtifacts` | 与 posttrain 的 `_build_dataset_and_loader(...)` 是强 mirrored seam，但 loader policy 不同。 |
| Model build / runtime prepare | `train/training_MPL.py:3404`, `train/training_MPL.py:3497`, `train/training_MPL.py:3581`, `train/training_MPL.py:3631` | 构建 `EventMotionModel`、attach encoder bundle、resume weights、history runtime。 | `train.models`, `train.checkpoint.load_schema`, `train.checkpoint.contract`, `train.history` | `DirectPoseBuildOptions`, model kwargs, attached frozen encoder/runtime | 与 posttrain 的 checkpoint-derived build shell 共用 constructor vocabulary，但不共用 orchestration。 |
| Loss / trainer build + train entry orchestration | `train/training_MPL.py:3669`, `train/training_MPL.py:3841` | 构建 `MotionJointLoss`、`Trainer`、attach dataset runtime、运行 fit。 | `train.models.MotionJointLoss`, `Trainer`, `build_and_attach_dataset_runtime` | `TrainBuildArtifacts`, `resolved_config`, trainer/loss runtime attrs | posttrain 同样构建 `MotionJointLoss` + `Trainer`，但后续 loop 不同。 |
| Export / postfit | `train/training_MPL.py:3794`, `train/training_MPL.py:3803`, `train/training_MPL.py:3897` | basetrain-only ONNX export / wrapper。 | `torch.onnx`, `os` | export batch probes, `onnx_path` | posttrain 不复用，应保持 local。 |

---

## 3. shared seam 清单

| 类型 | 是否存在 | 边界在哪里 | 本轮判断 |
|---|---|---|---|
| dataset / loader / runtime artifacts | 是，强 seam | `merge_norm_spec(...)` + `build_motion_dataset(...)` + `build_motion_dataloader(...)` + `build_and_attach_dataset_runtime(...)`；锚点见 `train/posttrain.py:3125`, `train/posttrain.py:3162`, `train/training_MPL.py:3417`, `train/training_MPL.py:3461`, `train/training_MPL.py:3850` | 可以先沿这个边界做后续拆分；但 loader policy（infinite iterator / collate / workers）继续留在各自 entry shell。 |
| rollout / step / eval loop | 部分存在 | 低层 step-state / carry / buffer 是 seam；高层 loop policy 不是。锚点见 `train/posttrain.py:769`, `train/posttrain.py:899`, `train/posttrain.py:1037`, `train/posttrain.py:2440`, `train/training_MPL.py:1271`, `train/training_MPL.py:1702`, `train/training_MPL.py:1779`, `train/training_MPL.py:2476` | 先切 low-level kernel，暂不切 full loop。否则会把 posttrain objective 和 basetrain eval 再揉回一起。 |
| pose history state | 是，强 seam | `PoseHistState` 初始化、runtime tensor attach、per-step resolve/advance；锚点见 `train/posttrain.py:873`, `train/posttrain.py:1037`, `train/training_MPL.py:1111`, `train/training_MPL.py:3291` | 这是最适合形成 shared kernel 的低层状态面之一。 |
| checkpoint load / build-state resolution | 只有低层 seam | 低层共享于 `train.checkpoint.load_schema` / `train.checkpoint.contract` / bundle attach；高层 orchestration 分别在 `train/posttrain_build_shell.py:108` 和 `train/training_MPL.py:3497`/`train/training_MPL.py:3631` | 不应把“从 checkpoint 反推 model build-state”和“basetrain resume load”抽成一个共同入口。 |
| config / norm_spec / runtime attach | 是，最高优先级 seam | `merge_norm_spec(...)`、`resolve_contact_pretrain_runtime(...)`、`TrainerRuntimeConfig` 风格 attach；锚点见 `train/posttrain.py:3162`, `train/training_MPL.py:2830`, `train/training_MPL.py:3291`, `train/training_MPL.py:3375` | 后续最先切这里最稳，因为这里本来就是 joint runtime contract。 |
| logging / diagnostics / metrics | 部分存在 | JSON writer / runtime diag / grad stats 概念共享，但 epoch metrics、teacher eval、posttrain step log 不共享；锚点见 `train/posttrain.py:2793`, `train/posttrain.py:2920`, `train/training_MPL.py:2503`, `train/training_MPL.py:2579`, `train/training_MPL.py:2608` | 可以共享 writer/helper，不要共享 log policy。 |
| export / resume / trainer helper | mixed | `Trainer` 本身已是 shared helper；resume/bundle compat 是低层 seam；导出/最终 artifact policy 不共享。锚点见 `train/posttrain.py:3162`, `train/posttrain_build_shell.py:346`, `train/training_MPL.py:3581`, `train/training_MPL.py:3656`, `train/training_MPL.py:3803` | 只抽 helper，不抽 export/resume orchestration。 |

---

## 4. 建议切分顺序

下面的顺序故意先沿 **joint seam** 走，再去拆各自独有的大块；这样能避免“先分别切两个文件，后面又为了共享 runtime 再合一次”。

### 4.1 开工前先锁定的决策

Step 1 详细接口草案见：

- `docs/refactor/2026-04-18_runtime_attach_api_draft.md`

- **Step 1 attr 命名策略：采用“保留前缀 + 中性映射层”**  
  - 不在 Step 1 统一 `posttrain_*` / `trainbase_*` live attr 名。  
  - shared helper / dataclass 使用中性字段名；caller 端继续各自映射到 `posttrain_*` / `trainbase_*`。  
  - 这样可以避免在最早阶段引入 runtime attr rename / checkpoint 兼容风险。
- **Step 2 和 Step 4 的界限**  
  - Step 2 只做 **shell ↔ core 边界显式化**：允许重排、分段、局部 façade，但**不抽 shared rollout kernel module**。  
  - Step 4 才是 **第一次正式抽 shared low-level rollout kernel module**。  
  - 执行时如果 Step 2 开始搬 shared kernel，就说明已经越界，应立即停下重评。
- **checkpoint round-trip smoke gate 前移到 Step 3**  
  - Step 3 一旦碰 `posttrain` checkpoint rebuild path，就必须在该 Step 结束时跑 smoke。  
  - Step 6 仍然继续把这条 smoke 当成 blocker，但不再等到 Step 6 才首次发现 contract 回归。

| Step | 先切什么 | 产出物 | 为什么先切它 | 主要风险 | 验收 / stop-rule | 为什么不会导致后面再合一次 |
|---|---|---|---|---|---|---|
| 1 | `config / norm_spec / runtime attach` seam | **新增 module**：`train/runtime_attach.py`。**搬走**：`training_MPL.py` 中 runtime attach 解析/应用内核；`posttrain.py` 中 dataset-runtime attach 的中性部分。**保留原文件**：caller-specific 前缀映射、entry 参数读取。 | 这是两边已经事实共享、且不改训练语义的最稳边界；`DatasetRuntimeArtifacts`、pose_hist runtime、contact-pretrain hydration 都在这里会合。 | attr 命名 drift（`posttrain_*` vs `trainbase_*`）。 | 验收：完成“中性字段 + caller 映射”策略，不重命名 live attrs。若 Step 1 需要直接 rename 现有 runtime attr 或改 checkpoint-facing key，立即停手，回到 zone map 重评。 | 因为后续不管怎么拆，两边都仍然需要同一批 runtime metadata；这里只会越来越稳定，不会回流。 |
| 2 | `training_MPL.py` 的 basetrain shell（parser/defaults/entry/export）从 `Trainer` core 外围剥离 | **新增 module**：无（刻意不抽 shared module）。**搬走**：无跨文件搬运，只在 `training_MPL.py` 内把 `_load_train_entry_config_defaults(...)`、parser adders、`_build_train_components(...)`、`_build_train_loaders(...)`、`_build_train_model(...)`、`_prepare_train_model_runtime(...)`、`_build_train_loss_and_trainer(...)`、`_run_postfit_actions(...)`、`_export_postfit_onnx(...)`、`train_entry()` 收拢为 contiguous shell band。**保留原文件**：`Trainer` class、rollout DTOs、`TrainerRuntimeConfig`、runtime attach helpers。 | 先把真正可复用的 shared core（`Trainer` runtime kernel）从 basetrain 壳层里显式露出来。 | import churn；容易顺手把 fit 细节和 core 混在一起搬。 | 验收：只暴露 shell/core 边界，不引入 shared rollout module。若出现跨文件抽取 low-level rollout helper 的冲动，说明已越过 Step 4 边界，必须停手。 | basetrain shell 是单边入口，posttrain 不会反向依赖它；拆开后没有“再合”的理由。 |
| 3 | `posttrain.py` 的 checkpoint/build-state shell | **新增 module**：`train/posttrain_build_shell.py`。**搬走**：`PostTrainModelBuildState`、`_resolve_posttrain_model_build_state(...)`、`_instantiate_posttrain_model(...)`、`_load_posttrain_checkpoint_into_model(...)`、`_build_posttrain_model_from_ckpt(...)`。**保留原文件**：`main()`、`_build_dataset_and_loader(...)`、`_build_model_and_trainer(...)`、`_run_training_loop(...)`、`_save_posttrain_outputs(...)`、posttrain-only objective 区。 | 这是 posttrain 最大的单边区块之一，而且和 basetrain 共享的只是低层 compat helper，不是高层流程。 | checkpoint compat 误改；容易把 payload contract 一起碰到。 | 验收：必须通过 checkpoint round-trip smoke；至少确认 payload shape / top-level key contract 不变，load→save→reload 路径在 compat 预期内。若 smoke 失败，立即回滚 Step 3 并重做该 Step 的 zone map。 | posttrain rebuild 路径长期只会属于 posttrain，不是共享层，所以不会再并回 `training_MPL.py`。 |
| 4 | 低层 rollout state kernel（pose history / cond-contact-time input / carry / output buffers） | **新增 module**：`train/rollout_kernel.py`。**搬走**：`training_MPL.py` 中 rollout DTOs + low-level step/carry/buffer helpers；`posttrain.py` 中 `_prepare_rollout_cond(...)`、`_prepare_rollout_contacts_input(...)`、`_update_rollout_recurrent_state(...)`、`_apply_rollout_carry_state(...)`、`_rollout_step_common(...)` 的 shared low-level 部分。**保留原文件**：`Trainer.fit(...)`、`evaluate_teacher(...)` 调度、`_lambda_fusion_loss_rollout(...)` 与其高层 posttrain policy。 | 只有在 Step 1-3 之后，policy 与 shell 才足够分明，低层 shared kernel 才不会夹带 basetrain/posttrain 高层语义。 | 一旦误把 teacher eval 或 lambda/direct objective 一起搬，训练语义就会 drift。 | 验收：新 module 只承载 low-level state kernel，不直接拥有 basetrain epoch loop 或 posttrain objective。若 diff 开始吞入 teacher eval、stage schedule、lambda/direct loss 聚合，立即停手拆小该 Step。 | kernel/policy 分层是结构性边界；一旦稳定，后面只会让 posttrain 和 basetrain 各自站在同一 kernel 上，不会反向合并。 |
| 5 | `posttrain.py` 的 posttrain-only policy/loss module | **新增 module**：`train/posttrain_objective.py`。**搬走**：`_lambda_rollout_prepare_context(...)`、`_lambda_rollout_*` family、`_lambda_fusion_run_unroll(...)`、`_lambda_fusion_finalize(...)`、`_lambda_fusion_loss_rollout(...)`。**保留原文件**：config parse、dataset/build shell、training loop shell、artifact save。 | 到这时 shared kernel 已稳定，posttrain 的 `_lambda_*` family 可以作为纯 policy zone 独立出去。 | model output key 假设、aux payload 约定散落。 | 验收：posttrain policy module 仍只依赖 kernel contract 和 model output schema，不重新吸入 checkpoint/build shell。若需要回头改 Step 3 shell 才能落地，先停手回看 zone map 是否切点错误。 | 这块本来就不应共用；独立后只会继续局部演化，不会再并回 generic runtime。 |
| 6 | `models.py` / constructor-level reconciliation（**最后**） | **新增 module**：`train/model_build_contract.py`（仅在 smoke 绿灯后）。**搬走**：caller-side constructor kwarg normalization / build contract helpers；**保留原文件**：`train/models.py` 中真实模型定义与 head implementation。 | 这是最大 contract surface；必须在 runtime seam 和 checkpoint seam 都稳定之后再碰。 | checkpoint contract / compat 回归。 | 验收：Step 3 的 checkpoint round-trip smoke 仍为 blocker，并在改 `models.py` 前再次通过。若 smoke 再次失败，停止 Step 6，不进入 model internals。 | 放到最后，并以前置 smoke gate 限制，避免一边拆 shell 一边动最宽契约面。 |

### 4.2 全局 stop-rule / rollback 条款

- 任一 Step 如果出现以下任一情况，立即停手、回滚该 Step、重做该 Step 的 zone map：
  - 改动面超出预期，且需要触碰 **400 行以上** 的非目标区代码；
  - 触碰了该 Step 未声明的额外模块，且超出 **3 个文件**；
  - smoke / compile / checkpoint round-trip gate 失败；
  - 为了落地当前 Step，不得不改 live runtime attr 名或 checkpoint top-level contract。
- Step 2-3 属于 **boundary exposure / shell extraction** 阶段；若执行中开始吸入 shared kernel、policy logic 或 `models.py` contract 变更，视为越界，必须停下。
- Step 4-5 属于 **kernel / policy extraction** 阶段；若执行中重新需要把 shell orchestration 吸回新 module，说明前置切点不可信，也必须回 zone map 重评。

### 为什么这个顺序比“分别切两个文件”更稳

如果先分别对 `posttrain.py` 和 `training_MPL.py` 做 file-local split，会立刻遇到两个问题：

1. **shared runtime seam 会被切穿两次**  
   - 一次在 basetrain 里切；
   - 一次在 posttrain 里切；  
   最后为了统一 `Trainer` runtime / pose_history / contact-pretrain / norm attach，还得再合回来。

2. **policy 和 kernel 会被混着搬**  
   - `posttrain` 的 `_lambda_*` 高层目标定义，和
   - `training_MPL` 的 generic rollout kernel  
   使用了相同 vocabulary，但不是同一职责。  
   如果先按文件拆，很容易把“名字相似的函数”误判成应该共用，后面再返工拆开。

因此更稳的路线是：

- 先锁 joint seam，
- 再把 shell 分离，
- 最后才动 policy zone。

---

## 5. 非目标

本轮明确不做：

- 不正式拆出新模块。
- 不修改任何训练语义。
- 不修改 checkpoint 保存/加载 contract。
- 不修改 `train/models.py`。
- 不修改 `train/pretrain_mpl_min.py`。
- 不统一 `PostTrainConfig` 与 basetrain argparse schema。
- 不把 `Trainer.fit(...)` 和 posttrain 自定义 loop 合并。
- 不把 `posttrain` 的 `_lambda_fusion_loss_rollout(...)` 抽成 generic common layer。
- 不把 posttrain checkpoint rebuild 路径与 basetrain resume 路径统一成一个入口。
- 不做 reorder-only diff。

### 那些“看起来能拆，但现在不该拆”的部分

- `train/posttrain.py:2440` `_lambda_fusion_loss_rollout(...)`  
  - 体量大，但它是 posttrain policy，不是 shared runtime。
- `train/training_MPL.py:1779` `_rollout_sequence(...)`  
  - 可以为 shared kernel 做基础，但不能直接等同于 posttrain 的整个 rollout objective。
- `train/posttrain_build_shell.py:108` `_resolve_posttrain_model_build_state(...)`  
  - 可以独立成 posttrain-only shell，但不应因为“都在 build model”就跟 basetrain builder 抽成同层公共入口。
- `train/training_MPL.py:2788` `Trainer.fit(...)`  
  - 是 basetrain 管理流程，不是 posttrain helper。

---

## 6. `models.py` 前置条件

在动 `train/models.py` 之前，先跑 **checkpoint round-trip smoke**，确认 contract / compat 边界稳定。

补充约束：

- 这条 smoke 从 Step 3 起就是硬验收，不再等到 Step 6 才第一次执行。
- Step 6 只是再次把它当 blocker；如果 Step 3 没过，就不进入 `models.py` 阶段。

明确 gate：

> **先做 checkpoint round-trip smoke，再动 `models.py`。**

本轮只记录这个前置条件，不在本轮实现它。
