# Basetrain / posttrain build skeleton inventory

Date: 2026-04-25  
Status: Draft / pre-`compute_build_order_hash()` paper artifact  
Scope: `train/training_MPL.py`, `train/posttrain.py`, `train/posttrain_build_shell.py`, `train/runtime_attach.py`  
Goal: 在真正落 `compute_build_order_hash()` 之前，先冻结“哪些 build chunk 算 semantic step、每步 consumes/produces 什么、哪些字段是 volatile 噪声”。

---

## 0. 约定

### 0.1 这份 inventory 记录的是什么

这里的 `step_id` 指 **semantic build skeleton step**，不是某个具体 helper 的文件位置。

也就是说：

- helper 可以挪文件
- wrapper 可以重命名
- 只要语义 chunk 没变，`step_id` 就不该变

### 0.2 粒度约定

`build_order_hash` 不按 leaf function 粒度做；它按 **semantic chunk** 做。

例如：

- `basetrain.attach_entry_runtime`
- `posttrain.attach_trainer_runtime`

这两个 step 下面可以继续展开内部 helper，但 hash-level skeleton 应保持 big-chunk 粒度，否则会对实现细节过敏。

**"folded into parent chunk" 的精确含义**：substep 内部的 `consumes` / `produces` / `attached_attrs` **细节**折进 parent chunk hash，不单独生成 hash 段；但 substep `step_id` 的 **ordered sequence 仍是 parent chunk 的 hash 输入**。这样 `sync_loss_runtime` 被挪到 `apply_runtime_cfg` 之前、或 `resolve_shared_runtime` 与 `attach_dataset_runtime` 对调这类 ordering drift 不会被漏掉。

### 0.3 产物命名

这份 inventory 统一使用以下 artifact 名称：

- `TrainEntryContext`
- `TrainDataArtifacts`
- `TrainModelArtifacts`
- `TrainBuildArtifacts`
- `DatasetRuntimeArtifacts`
- `TrainerRuntimeConfig`
- `PostTrainModelBuildState`
- `PostTrainModelArtifacts`
- `Trainer`

---

## 1. Basetrain skeleton

### 1.1 Hash-level steps

| order | step_id | current owner | consumes | produces | attached / mutated state | volatile_excluded | hash_scope | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | `basetrain.parse_context` | `train/training_MPL.py:_build_train_components` | `argv`, config defaults, CLI overrides | `TrainEntryContext` | resolves `args`, `train_paths`, `device`, `norm_template_path`；**volatile / non-hash**: `out_dir`, `run_name` | raw argv order, cwd-dependent incidental path spelling | required | 这里只锁”解析发生在 dataset/model build 之前”，不锁 argparse 实现细节 |
| 2 | `basetrain.build_dataset_loader` | `train/training_MPL.py:_build_train_loaders` | `TrainEntryContext` | `TrainDataArtifacts` | constructs `ds_train`, `train_loader`, `dx/dy/dc` | DataLoader worker runtime noise, loader object id | required | 这是 dataset/loader build chunk，不把 sample order / worker pid 算进 hash |
| 3 | `basetrain.instantiate_model` | `train/training_MPL.py:_build_train_model` | `TrainEntryContext`, `TrainDataArtifacts` | `TrainModelArtifacts` | instantiates `EventMotionModel` with resolved structural options | device placement detail, raw out_dir paths | required | `direct_pose_options` / history export dim 解析属于这个 chunk |
| 4 | `basetrain.prepare_model_runtime` | `train/training_MPL.py:_prepare_train_model_runtime` | `TrainEntryContext`, `TrainDataArtifacts`, `TrainModelArtifacts` | same `TrainModelArtifacts` (mutated model runtime) | attaches adaptive history, validates model, attaches external motion/contact bundle, resume-loads weights, sets `_pasa_fps` | bundle file path string, raw checkpoint path string, RNG advancement side effects | required | 这是 build-order 上非常关键的一步；顺序错位会直接改变 model semantic graph |
| 5 | `basetrain.build_loss_and_trainer` | `train/training_MPL.py:_build_train_loss_and_trainer` | `TrainEntryContext`, `TrainDataArtifacts`, `TrainModelArtifacts` | `TrainBuildArtifacts` | instantiates `MotionJointLoss`, `Trainer`, writes resolved config snapshot | full `resolved_config` blob path, json dump path | required | hash 只关心“loss/trainer 在 runtime attach 之前 build 完成”这件事 |
| 6 | `basetrain.attach_entry_runtime` | `train/training_MPL.py:_attach_train_entry_runtime` | `TrainEntryContext`, `TrainDataArtifacts`, `TrainBuildArtifacts` | ready-to-fit `Trainer` / `loss_fn` runtime | dataset runtime attach, shared trainer runtime, contact-pretrain runtime, history schedule attrs, loss runtime sync | `out_dir`, `run_name`, `bundle_json_path`, full config blob, trainer object id | required | 这个 chunk 内部还会展开为 shared attach helpers；hash 以 chunk 顺序 + attached attr families 为主 |

### 1.2 `basetrain.attach_entry_runtime` 内部展开

| substep | current owner | consumes | produces | attached attrs / side effects | hash inclusion |
|---|---|---|---|---|---|
| `basetrain.attach_dataset_runtime` | `build_and_attach_dataset_runtime(...)` | `trainer`, `ds_train`, optional `bundle_json_path` | `DatasetRuntimeArtifacts` | trainer dataset-normalization runtime, yaw/bundle metadata base attach | folded into parent chunk |
| `basetrain.resolve_runtime_cfg` | `_resolve_trainer_runtime_config(...)` | `args`, `trainer`, `DatasetRuntimeArtifacts`, path/meta inputs | `TrainerRuntimeConfig` | resolves shared runtime + contact-pretrain runtime + eval/history schedule | folded into parent chunk |
| `basetrain.apply_runtime_cfg` | `_apply_trainer_runtime_config(...)` | `trainer`, `TrainerRuntimeConfig` | mutated trainer | `pose_hist_*`, yaw attrs, `trainbase_contacts_pretrain_*`, `tf_*`, `freerun_stage_schedule`, diagnostics attrs | folded into parent chunk |
| `basetrain.sync_loss_runtime` | `_sync_train_entry_loss_runtime(...)` | `loss_fn`, `trainer` | mutated `loss_fn` | copies `mu_y/std_y`, optional bundle meta | folded into parent chunk |

### 1.3 Basetrain step ordering constraints

后续 `compute_build_order_hash()` 至少要锁住这些约束：

1. dataset/loader build 在 model build 之前  
2. model instantiate 在 model runtime prepare 之前  
3. loss/trainer build 在 trainer runtime attach 之前  
4. loss runtime sync 在 trainer runtime attach 之后  
5. adaptive-history / frozen-bundle attach 发生在进入训练循环之前  

---

## 2. Posttrain skeleton

### 2.1 Hash-level steps

| order | step_id | current owner | consumes | produces | attached / mutated state | volatile_excluded | hash_scope | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | `posttrain.parse_cfg_and_seed` | `train/posttrain.py` entry band | config json payload, CLI overrides | `PostTrainConfig`, `train_mode`, RNG seed setup | resolves typed cfg, train mode, device intent；**volatile / non-hash**: output dir, run directory | raw config path, run directory path spelling, RNG state values | required (config subset only) | hash 不记录 RNG 当前值，只记录 seed/setup 这一语义 chunk 存在 |
| 2 | `posttrain.build_dataset_loader` | `train/posttrain.py:_build_dataset_and_loader` | `PostTrainConfig` | `norm_spec`, `MotionEventDataset`, infinite `batch_iter` | dataset + loader build | iterator object id, batch iterator state | required | 与 basetrain 一样，只锁 dataset/loader build chunk |
| 3 | `posttrain.resolve_ckpt_build_state` | `train/posttrain_build_shell.py:_resolve_posttrain_model_build_state` | `cfg`, `ds`, checkpoint payload | `PostTrainModelBuildState` | resolves structural enables（7 flags）：`direct_pose_enable`, `lambda_fusion_enable`, `contact_plan_enable`, `use_event_clock`, `direct_pose_leg_enable`, `direct_pose_leg_side_routing`, `direct_pose_arm_split_enable`；另解析 1 个 structural mode：`direct_pose_leg_mode`；并补齐由 ckpt shape 反推的 head/branch 维度 | checkpoint file path, raw dict insertion order | required | posttrain 最关键的 semantic inference step；step 4 `instantiate_model` 的结构由本步结果决定，故 `module_graph_hash` 中各 slot 的 `enabled_when` / mode 取值必须与本步产出一致（同一信息进两段 hash 时，canonical 序列化须一致） |
| 4 | `posttrain.instantiate_model` | `train/posttrain_build_shell.py:_instantiate_posttrain_model` | `cfg`, `ds`, `device`, `PostTrainModelBuildState` | instantiated `EventMotionModel` | model placed on device, validated | device id / storage ptr | required | 必须位于 ckpt load 之前 |
| 5 | `posttrain.load_ckpt_into_model` | `train/posttrain_build_shell.py:_load_posttrain_checkpoint_into_model` | `cfg`, instantiated model, `PostTrainModelBuildState` | mutated model + loaded state | optional encoder bundle attach, compat transform, `strict=False` state load, direct-pose/lambda guards | bundle file path, raw checkpoint path | required | 这是最容易发生“能 load 但语义漂”的步骤之一 |
| 6 | `posttrain.verify_rollout_contracts` | `train/posttrain.py` entry band | `cfg`, loaded model | verified model/runtime contract | checks required bundle slots for contact rollout | raw file path spelling | required | 这是 fail-fast boundary；不应被 silent fallback 替代 |
| 7 | `posttrain.build_loss_and_trainer` | `train/posttrain.py:_build_posttrain_loss_and_trainer` | `cfg`, `ds`, loaded model | `Trainer` | instantiates `MotionJointLoss`, `Trainer`, syncs bone names | none beyond object ids | required | posttrain loss/trainer 必须在 runtime overlay 前建立 |
| 8 | `posttrain.attach_trainer_runtime` | `train/posttrain.py:_attach_posttrain_trainer_runtime` | `cfg`, `ds`, `trainer`, `norm_spec` | ready-to-train `Trainer` | dataset runtime attach, shared runtime attach, loss runtime sync, posttrain local overlay | `bundle_json_path`, `out_dir`, full config blob, current run name | required | 这是 posttrain 的 trainer/runtime seam |
| 9 | `posttrain.configure_trainable_slots` | `train/posttrain.py` entry band (`_freeze_all` + `_unfreeze_for_train_mode`) | `cfg`, `train_mode`, model | trainable parameter mask / train mode selection | freezes all params, then selectively unfreezes direct/λ/leg-gate/nonleg subsets | exact parameter object ids | required | 这是 training semantics，不应被误归到纯 runtime 噪声 |

### 2.2 `posttrain.attach_trainer_runtime` 内部展开

| substep | current owner | consumes | produces | attached attrs / side effects | hash inclusion |
|---|---|---|---|---|---|
| `posttrain.attach_dataset_runtime` | `build_and_attach_dataset_runtime(...)` | `trainer`, `ds`, `bundle_json`, `norm_spec` | `DatasetRuntimeArtifacts` | trainer dataset runtime | folded into parent chunk |
| `posttrain.apply_shared_runtime` | `resolve_shared_trainer_runtime(...)` + `apply_shared_trainer_runtime(...)` | `DatasetRuntimeArtifacts`, cfg meta | mutated trainer | `pose_hist_*`, yaw attrs, output/meta attrs | folded into parent chunk |
| `posttrain.sync_loss_runtime` | `apply_loss_runtime_from_trainer(...)` | `trainer.loss_fn`, `trainer` | mutated `loss_fn` | normalization stats mirrored onto loss | folded into parent chunk |
| `posttrain.apply_local_overlay` | `_resolve_posttrain_local_runtime_overlay(...)` + `_apply_posttrain_local_runtime_overlay(...)` | `cfg`, `trainer` | mutated trainer | `posttrain_contacts_pretrain_*`, contact meas policy, lambda reliability policy, rollout-local runtime knobs | folded into parent chunk |

### 2.3 Posttrain step ordering constraints

后续 `compute_build_order_hash()` 至少要锁住这些约束：

1. checkpoint-backed build-state resolve 在 model instantiate 之前  
2. model instantiate 在 ckpt load / compat 之前  
3. bundle attach / compat / state load 在 trainer build 之前  
4. trainer runtime attach 在 trainable-parameter selection 之前  
5. fail-fast rollout contract verification 发生在进入训练循环之前  

---

## 3. Shared attach seam inventory

以下 shared seam 同时被 basetrain / posttrain 消费，因此推荐在 `build_order_hash` 中把它们当作稳定语义动作，而不是 caller-local 偶然实现：

| semantic action | current owner | consumes | produces | exclude from hash |
|---|---|---|---|---|
| `shared.attach_dataset_runtime` | `build_and_attach_dataset_runtime(...)` | trainer + dataset + optional bundle/norm spec | `DatasetRuntimeArtifacts` + trainer dataset runtime attrs | loader object id, path spelling |
| `shared.resolve_shared_runtime` | `resolve_shared_trainer_runtime(...)` | `DatasetRuntimeArtifacts`, caller overrides/meta | `SharedTrainerRuntime` | `out_dir`, `current_run_name`, raw full config blob |
| `shared.apply_shared_runtime` | `apply_shared_trainer_runtime(...)` | trainer + `SharedTrainerRuntime` | trainer shared runtime attrs | none beyond object ids |
| `shared.apply_contacts_pretrain_runtime` | `apply_contacts_pretrain_runtime(...)` | trainer + normalized contact-pretrain runtime | trainer owner-prefixed contact-pretrain attrs | raw caller prefix string formatting noise |
| `shared.apply_loss_runtime` | `apply_loss_runtime_from_trainer(...)` | loss_fn + trainer | mirrored loss runtime attrs | bundle meta dict ordering |

---

## 4. Volatile fields that must stay out of `build_order_hash`

无论 basetrain 还是 posttrain，下列内容都属于 **runtime metadata / environment noise**，不应直接进入 `build_order_hash`：

- `out_dir`
- `run_name`
- `bundle_json_path`
- raw config file path
- checkpoint path
- temporary directory path
- Python object id / memory address
- DataLoader worker runtime details
- RNG current state / generator advance count
- `full_config` 原始大字典的插入顺序

如果这些信息需要保留，应：

- 写进 manifest
- 但不进入 required build-order hash input

相反地，以下项**不**属于 volatile（常被误判）：

- normalization stat **identity**（`mu_y / std_y` 的来源、norm mode、bundle-version-anchored stat set）——这些属于 per-slot `normalized_config` 或独立的 `normalization_policy_hash`，不能因为"通过 runtime attach 传递"就被当成噪声。
- `configure_trainable_slots` 的 trainable-parameter 选集（见 §2.1 step 9）——按 sorted `{slot_name: trainable_bool}` 入 hash，不按 parameter object id。
- resolved structural enables（见 §2.1 step 3）——即使通过 CLI override 进入，也属于语义输入。

---

## 5. 对后续 `compute_build_order_hash()` 的直接约束

后续实现时应遵守：

1. hash-level step 使用本文件的 `step_id`，不是函数 import path  
2. basetrain / posttrain 统一按 semantic chunk 顺序散列，不按 leaf helper 散列  
3. 内部 helper 展开主要用于生成 `short_diff_hint`，不是直接决定 hash 粒度  
4. `configure_trainable_slots` 这类影响训练语义的步骤不能被当成纯噪声跳过  
5. 路径、run-meta、RNG state 只做 manifest 审计，不做 required hash input
