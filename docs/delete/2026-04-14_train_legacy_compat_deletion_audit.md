# 2026-04-14 `train/*` legacy / compat 分支删除审计清单

Date: 2026-04-14  
Status: Updated with one fresh basetrain->posttrain rerun / checklist-ready; A/B code deletions landed on 2026-04-15, C/D/E remain as documented  
Scope: `train/*.py`，**显式忽略** `train/configuration/*` 与 `train/validate/*`  
Method: 静态引用分析（`rg` + 代码阅读） + 一次真实 fresh chain 回归：`debug_output/_tmp_tail_top7_fresh_chain_20260414_235208`。  
Goal: 给后续“删 compat / legacy 壳”提供一份**函数级** inventory + checklist，方便你后面按流程打勾/打叉。

2026-04-15 update:

- 已删除 A 类项：`train/debug_contact_plan_stability.py`、`train/train_configurator.py`、`train/__init__.py::__getattr__`、`direct_pose_leg_gate_loss_weight` alias、`direct_pose_leg_gate_mode='auto'` deprecated 分支、`dataset_index_mode <- index_mode` alias
- 已删除 B 类项：`train/layout.py` 的 `fallback_to_bone_names`、`EventMotionModel.__init__` 的 `contact_phase_state_*` legacy kwargs 壳、`adapt_legacy_event_motion_state_dict`、`preprocess_event_motion_state_dict_for_load`
- 当前文档其余条目仍保留原始审计结论，作为后续 C/D/E 清理参考

本轮 fresh-chain 结构化产物：

- `debug_output/_tmp_tail_top7_fresh_chain_20260414_235208/run_context.json`
- `debug_output/_tmp_tail_top7_fresh_chain_20260414_235208/runtime_compat_hits.json`

本轮真实 rerun 中顺手修掉的 live blocker（它们不是 legacy/compat 分支本身，但会直接阻塞 runbook）：

1. `train/posttrain.py` 缺少 `_cfg_pick` import，导致 `stage6` 首次启动直接 `NameError`
2. `train/posttrain.py` 缺少 `_enable_modules` import，导致 `lambda` 首次启动直接 `NameError`

额外观察到但暂未在本报告内处理的 live issue：

- basetrain 每个 epoch 的 teacher validation 仍会因为 `Trainer` 缺少 `_diagnose_free_run` 而被 skip
- `70R` 仍需要临时 shim，当前 runbook 不能直接裸跑 `tools/run_posttrain_nonleg_trunk_ablation.py`

---

## 0. 结论先看

基于这轮 fresh chain，建议先按“当前主链是否真的会碰到”来分：

### A. 对当前 fresh chain 可直接视为可删

1. `train/debug_contact_plan_stability.py`
2. `train/train_configurator.py`
3. `train/__init__.py::__getattr__`
4. `train/posttrain.py` 中 `direct_pose_leg_gate_loss_weight` alias
5. `train/posttrain.py` 中 `direct_pose_leg_gate_mode='auto'` deprecated 分支
6. `train/posttrain.py` 中 `dataset_index_mode <- index_mode` alias

### B. 这轮 fresh chain 没打到，但删前仍建议清理调用面 / 测试 / archive

1. `train/layout.py::{resolve_rot6d_slice, infer_rot_joint_count}` 的 `fallback_to_bone_names`
2. `train/models.py::EventMotionModel.__init__` 中 `contact_phase_state_*` legacy kwargs 壳
3. `train/model_ckpt_compat.py::adapt_legacy_event_motion_state_dict`
4. `train/model_ckpt_compat.py::preprocess_event_motion_state_dict_for_load`

### C. 真实 rerun 明确证明不能删

1. `train/training_MPL.py` 的 legacy train-entry ignored keys/flags
2. `train/model_ckpt_compat.py::resume_load_weights_compat`
3. `train/model_ckpt_compat.py::_resolve_direct_pose_ckpt_compat_policy`
4. `train/model_ckpt_compat.py::apply_direct_pose_ckpt_compat`
5. `train/model_ckpt_compat.py::maybe_upgrade_direct_pose_split_state_dict`
6. `train/model_ckpt_compat.py::maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict`

### D. 这轮没用到，但更像 guard；删不删取决于你还要不要 fail-fast 保护

1. `train/training_MPL.py::{_legacy_loss_keys_msg, _assert_no_legacy_loss_keys_in_schedule}`
2. `train/models.py::MotionJointLoss.__init__` 里的 deprecated loss keys 拒绝逻辑

### E. fresh-chain runtime override

- basetrain 仍然真实依赖 parser swallow layer：`diag_input_stats`, `eval_angvel_dir_percentile`, `eval_horizon`, `eval_warmup`, `force_valfree_eval`, `monitor_batches`, `no_monitor`, `val_mode`
- active posttrain config 全部使用新字段：
  - `dataset_index_mode`
  - `direct_pose_leg_gate_sup_weight`
  - `direct_pose_leg_gate_mode='none'`
- runtime hits（`runtime_compat_hits.json`）显示：
  - `EventMotionModel` 初始化 9 次，`contact_phase_state_*` legacy kwargs 命中 0 次
  - `MotionJointLoss` 初始化 9 次，deprecated loss kwargs 命中 0 次
  - `train/layout.py::{resolve_rot6d_slice, infer_rot_joint_count}` 命中 0 次
  - `resume_load_weights_compat` 命中 1 次（basetrain）
  - `_resolve_direct_pose_ckpt_compat_policy` 命中 8 次，其中 truthy 返回 1 次
  - `apply_direct_pose_ckpt_compat` 命中 8 次
  - `maybe_upgrade_direct_pose_split_state_dict` 命中 8 次，其中真正发生升级 1 次（fresh basetrain -> stage6 donor 边界）
  - `maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict` 命中 8 次，但本轮 0 次真正改写
  - `adapt_legacy_event_motion_state_dict` / `preprocess_event_motion_state_dict_for_load` 命中 0 次

---

## 1. 状态标记

- `Remove-Now`：当前仓内静态引用已基本清空，删前只需要做轻量 smoke。
- `Remove-If-Clean`：需要先确认 config / docs / tools / 外部调用面已经清理。
- `Keep-Guard`：虽然是 legacy/compat 相关，但现在扮演“显式拒绝旧配置”的 guard，不建议先删。
- `Keep-Active`：仍在主链实际工作，删掉会直接破坏训练/后训练/导出/恢复加载。
- `Cold-In-Chain`：这轮 fresh basetrain->posttrain rerun 没打到；若你的目标只是保当前主链，可优先怀疑它可删，但 repo 级删除前仍要清理外部调用面。

---

## 2. 总表

| Item | Location | 类型 | Fresh-chain 证据 | 建议 |
|---|---|---|---|---|
| `debug_contact_plan_stability` | `train/debug_contact_plan_stability.py:1` | 独立诊断脚本 | rerun 未触发；repo 内也未搜到 consumer | `Remove-Now` |
| `train_configurator.main` | `train/train_configurator.py:9` | CLI 转发壳 | 当前真实链路只用 `-m train.training_MPL` / `-m train.posttrain` | `Remove-Now`（若只保当前主链） |
| `__getattr__` lazy export | `train/__init__.py:28` | 包级兼容导出面 | 当前真实链路未触发；repo 内未发现 live consumer | `Remove-Now`（若只保当前主链） |
| `direct_pose_leg_gate_loss_weight` alias | `train/posttrain.py:679`, `train/posttrain.py:3684` | config/CLI alias | 活跃 stage6/70a/replace/70R/71/72/lambda config 全用新键 `direct_pose_leg_gate_sup_weight` | `Remove-Now` |
| `direct_pose_leg_gate_mode='auto'` | `train/posttrain.py:4141` | deprecated enum alias | 活跃 config 全用 `none` / `cycle`，当前 fresh chain 未传 `auto` | `Remove-Now` |
| `dataset_index_mode <- index_mode` | `train/posttrain.py:786` | config alias | 活跃 config 全用 `dataset_index_mode=start0` | `Remove-Now` |
| `fallback_to_bone_names` | `train/layout.py:107`, `train/layout.py:130` | layout fallback | `runtime_compat_hits.json` 中两处 helper 命中均为 0 | `Cold-In-Chain` / `Remove-If-Clean` |
| `contact_phase_state_*` ctor kwargs | `train/models.py:609`, `train/models.py:622` | legacy kwargs shell | `EventMotionModel` 初始化 9 次，legacy kwargs 命中 0 次 | `Cold-In-Chain` / `Remove-If-Clean` |
| legacy train-entry ignored keys/flags | `train/training_MPL.py:368`, `train/training_MPL.py:379`, `train/training_MPL.py:4241` | parser compat swallow layer | 本轮 basetrain config 仍真实携带 8 个 ignored keys | `Keep-Active` |
| legacy loss rejection | `train/training_MPL.py:293`, `train/training_MPL.py:318`, `train/training_MPL.py:327` | fail-fast guard | 当前链路未触发，但仍是更早更友好的报错层 | `Keep-Guard` |
| `MotionJointLoss` deprecated kwargs rejection | `train/models.py:3688` | fail-fast guard | `MotionJointLoss` 初始化 9 次，legacy kwargs 命中 0 次 | `Keep-Guard` |
| `resume_load_weights_compat` | `train/model_ckpt_compat.py:441` | resume compat | 本轮命中 1 次（basetrain） | `Keep-Active` |
| `_resolve_direct_pose_ckpt_compat_policy` | `train/model_ckpt_compat.py:914` | build/load policy | 本轮命中 8 次 | `Keep-Active` |
| `apply_direct_pose_ckpt_compat` | `train/model_ckpt_compat.py:1391` | load-time compat | 本轮命中 8 次 | `Keep-Active` |
| `maybe_upgrade_direct_pose_split_state_dict` | `train/model_ckpt_compat.py:1420` | split-head ckpt 升级 | 本轮命中 8 次，真实升级 1 次 | `Keep-Active` |
| `maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict` | `train/model_ckpt_compat.py:1643` | StepC leg-terminal 升级检查 | 本轮命中 8 次，真实升级 0 次 | `Keep-Active`（热检查 / 冷变换） |
| `adapt_legacy_event_motion_state_dict` | `train/model_ckpt_compat.py:1681` | 旧 EventMotionModel state adapter | 本轮命中 0 次 | `Cold-In-Chain` / `Revisit` |
| `preprocess_event_motion_state_dict_for_load` | `train/model_ckpt_compat.py:1690` | load 前 state preprocess | 本轮命中 0 次 | `Cold-In-Chain` / `Revisit` |

---

## 3. 详细审计条目

## 3.1 `train/debug_contact_plan_stability.py`

**位置**

- `train/debug_contact_plan_stability.py:1`
- 入口 `main()` 在 `train/debug_contact_plan_stability.py:311`

**它是什么**

- 一个独立诊断脚本，用于固定 `cond` 后长步 rollout，观察 `contact_plan GRU` 是否塌缩/发散。
- 代码上依赖 `train.model_ckpt_compat` 和 `train.models.EventMotionModel`，但没有反向 consumer。

**静态证据**

- 本轮没有搜到：
  - `from train.debug_contact_plan_stability import ...`
  - `python -m train.debug_contact_plan_stability`
  - 文档/工具/测试文本引用

**当前建议**

- `Remove-Now`

**删除前 checklist**

- [ ] `rg -n "debug_contact_plan_stability|python -m train\\.debug_contact_plan_stability" .` 无有效命中
- [ ] 确认没有你本地 shell alias / runbook / IDE task 在手工调用它
- [ ] `python3 -m py_compile train/models.py train/model_ckpt_compat.py`
- [ ] `python3 -m train.posttrain --help`

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

---

## 3.2 `train/train_configurator.py::main`

**位置**

- `train/train_configurator.py:9`

**它是什么**

- 一个非常薄的 CLI wrapper：
  - 兼容 `python -m train.train_configurator ...`
  - 兼容 `python train/train_configurator.py ...`
- 最终只转发到 `train.configuration.cli.main`，见 `train/train_configurator.py:16`。

**静态证据**

- 在忽略 `train/configuration/*` 后，剩余主树里没有模块 import 它。
- 文本引用只剩它自己的注释说明。

**风险点**

- 虽然在 `train` 主链里游离，但它仍然代表一个命令入口。
- 如果你还有人/脚本使用 `python -m train.train_configurator`，删掉就会 break。

**当前建议**

- `Remove-Now`（如果目标只保当前 fresh basetrain->posttrain 主链）

**Fresh-chain update**

- 本轮真实链路没有经过 `train/train_configurator.py`：
  - basetrain 入口是 `cpu_nomps_exec.py -m train.training_MPL`
  - posttrain 入口是 `cpu_nomps_exec.py -m train.posttrain`
- 因此它对当前主链是 runtime-cold 的。

**删除前 checklist**

- [ ] `rg -n "train\\.train_configurator|python -m train\\.train_configurator|python train/train_configurator.py" .` 无 repo 内命中
- [ ] 确认以后不再保留 `train.configuration` 这套 CLI 入口
- [ ] 如仍需入口，先把调用方统一改成目标模块正式入口，再删 wrapper

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

---

## 3.3 `train/__init__.py::__getattr__` lazy export 兼容层

**位置**

- `train/__init__.py:28`

**它是什么**

- 包级懒导出层，允许类似：
  - `from train import geodesic_R`
  - `from train import normalize_layout`
  - `from train import evaluate_teacher`

**静态证据**

- 本轮未搜到 repo 内对这些懒导出 symbol 的实际消费。
- repo 里确实有很多 `from train import posttrain` / `from train import training_MPL`，但那是子模块导入，不依赖这里的 `__getattr__`。

**当前建议**

- `Remove-Now`（如果目标只保当前 fresh basetrain->posttrain 主链）

**Fresh-chain update**

- 本轮真实链路没有经过包级 lazy export。
- 命令入口全部是显式模块执行，不需要 `from train import ...` 这层 compat 面。

**删除前 checklist**

- [ ] `rg -n "from train import (reproject_rot6d|rot6d_to_matrix|angvel_vec_from_R_seq|geodesic_R|compose_rot6d_delta|root_relative_matrices|so3_log_map|evaluate_teacher|evaluate_freerun|FreeRunSettings|parse_layout_entry|normalize_layout|canonicalize_state_layout|load_soft_contacts_from_json|direction_yaw_from_array|velocity_yaw_from_array)" .` 仍无命中
- [ ] 若存在外部 notebook / 私有脚本，先统一改成显式子模块导入
- [ ] 删除后跑一次 import smoke：`python3 - <<'PY'` / `import train` / `import train.posttrain` / `import train.training_MPL`

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

---

## 3.4 `train/posttrain.py` 的微型 alias / deprecated 分支

### 3.4.1 `direct_pose_leg_gate_loss_weight` alias

**位置**

- config alias: `train/posttrain.py:679`
- CLI alias: `train/posttrain.py:3684`

**它是什么**

- 新名字是 `direct_pose_leg_gate_sup_weight`
- 旧名字 `direct_pose_leg_gate_loss_weight` 只是兼容 alias

**静态证据**

- 本轮没有搜到 config / tools / tests 的实际使用。
- 命中基本只剩：
  - `train/posttrain.py`
  - 一份变更文档 `docs/changes/2026-03-01_posttrain_minimal_refactor_roadmap.md`

**当前建议**

- `Remove-Now`

**Fresh-chain update**

- 本轮 stage6/70a/replace/70R/71/72/lambda 活跃 config 全部使用 `direct_pose_leg_gate_sup_weight`。
- fresh-chain 中未观察到任何 `direct_pose_leg_gate_loss_weight` 输入。

**删除前 checklist**

- [ ] `rg -n "direct_pose_leg_gate_loss_weight" config tools docs tests train` 只剩文档历史记录
- [ ] `python3 -m train.posttrain --help`
- [ ] 任选 1 个当前 posttrain config 跑 `--help` / config parse smoke

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

### 3.4.2 `direct_pose_leg_gate_mode='auto'`

**位置**

- CLI 允许 `auto`: `train/posttrain.py:3665`
- runtime deprecated 处理: `train/posttrain.py:4141`

**它是什么**

- 旧逻辑里 `auto` 代表从 checkpoint 推断 leg gate 模式。
- 当前实现已经明确把 `auto` 当成 deprecated，并且统一降成 `"none"`。

**静态证据**

- 本轮没有搜到 config / tools / docs 中实际传入 `auto` 的使用。
- 唯一命中是 warning 文案本身。

**当前建议**

- `Remove-Now`

**Fresh-chain update**

- 本轮活跃 config 全部使用 `direct_pose_leg_gate_mode='none'`。
- fresh-chain 中未观察到任何 `direct_pose_leg_gate_mode='auto'` 输入。

**删除前 checklist**

- [ ] `rg -n "direct_pose_leg_gate_mode.*auto|--direct_pose_leg_gate_mode(=|\\s+)auto" config tools docs tests train` 无命中
- [ ] 删除后 `python3 -m train.posttrain --help`
- [ ] 若你有 stage6/70a 当前活跃 config，做一次最小 config parse smoke

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

### 3.4.3 `dataset_index_mode <- index_mode` alias

**位置**

- `train/posttrain.py:786`

**它是什么**

- posttrain config 读取时，优先读 `dataset_index_mode`，兼容读旧键 `index_mode`。

**静态证据**

- 没搜到现有 config 直接使用旧键 `index_mode`。
- 但有 tool 代码在做“`cfg.get("dataset_index_mode") or cfg.get("index_mode")`”式兼容读取，见 `tools/diagnose_stage7_sampling_grad_closure.py:1145`。

**当前建议**

- `Remove-Now`

**Fresh-chain update**

- 本轮 stage6/70a/replace/70R/71/72/lambda 活跃 config 全部使用 `dataset_index_mode=start0`。
- 当前真实链路没有任何一步回退到旧键 `index_mode`。

**删除前 checklist**

- [ ] `rg -n "\\bindex_mode\\b" config tools docs tests train` 只剩 tool 侧兼容读取
- [ ] 活跃 posttrain config 全部只使用 `dataset_index_mode`
- [ ] 如有 tool 仍回读 `index_mode`，先一起清掉 tool 的旧键回退

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

### 3.4.4 暂时不要动的 posttrain compat 分支

以下不建议现在删：

- `time_index_mode="auto"` -> runtime 归一到 `"global"`，见 `train/posttrain.py:2205`
  - 原因：当前很多 config 仍显式写 `time_index_mode: "auto"`
- `direct_pose_leg_align_mode` / `direct_pose_leg_align_schedule` 的合法值体系
  - `proj` / `cos` / `linear` / `none` 仍有大量 config 在用
- `direct_pose_leg_mode="rot6d_add"` 文案里虽标记 compat，见 `train/posttrain.py:3652`
  - 但这仍是一个真正可选模式，不是纯死分支

---

## 3.5 `train/layout.py::{resolve_rot6d_slice, infer_rot_joint_count}` 的旧布局 fallback

**位置**

- `train/layout.py:107`
- `train/layout.py:130`

**它是什么**

- 当 layout 缺失 `BoneRotations6D` 时，退回到旧约定：
  - `rot_slice = slice(0, len(bone_names) * 6)`
  - joint count 退回到 `len(bone_names)`

**静态证据**

- 这些函数仍被 `train/models.py` 多处调用。
- 有些调用显式传 `fallback_to_bone_names=False`，说明主链已经在逐步摆脱这个旧兜底。
- 但另一些调用还保留“必要时 fallback”的模式，说明元数据还没完全规范化。

**风险点**

- 一旦删掉 fallback，老 bundle / 老 output_layout / 老 norm template 可能直接炸在建模阶段，而不是被平滑兜住。

**当前建议**

- `Remove-If-Clean`

**Fresh-chain update**

- `runtime_compat_hits.json` 里：
  - `resolve_rot6d_slice_calls = 0`
  - `infer_rot_joint_count_calls = 0`
- 也就是说，这轮 fresh chain 连 helper 本身都没打到，更不用说真的走 fallback 分支。

**删除前 checklist**

- [ ] 抽样审计当前活跃 `norm_template.json` / bundle / dataset meta，确认都带明确 `BoneRotations6D`
- [ ] `train/models.py` 所有调用点都切到 `fallback_to_bone_names=False`
- [ ] 至少跑 1 次 `train.training_MPL` 入口 smoke + 1 次 `train.posttrain` 入口 smoke

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

---

## 3.6 `train/models.py::EventMotionModel.__init__` 的 `contact_phase_state_*` legacy kwargs 壳

**位置**

- legacy name helper: `train/models.py:424`
- 兼容剥离逻辑: `train/models.py:609`, `train/models.py:622`

**它是什么**

- 构造器接受 `**legacy_kwargs`
- 会把下列旧参数名静默 `pop` 掉，然后只对剩余未知参数报错：
  - `contact_phase_state_enable`
  - `contact_phase_state_init_mode`
  - `contact_phase_state_hidden`
  - `contact_phase_state_delta_max`
  - `contact_phase_state_delta_init`
  - `contact_phase_state_event_kind`
  - `contact_phase_state_event_thr`
  - `contact_phase_state_event_hyst`
  - `contact_phase_state_event_min_interval`

**静态证据**

- 测试仍显式传入 `contact_phase_state_enable`，见 `tests/train/test_event_motion_model_refactor_phase_d.py:80`
- archive config / 历史文档仍多次提到这些字段

**当前建议**

- `Remove-If-Clean`

**Fresh-chain update**

- `runtime_compat_hits.json` 里 `EventMotionModel` 初始化 9 次。
- `legacy_kwargs_calls = 0`，说明当前 fresh chain 没有任何一次真正传入 `contact_phase_state_*`。

**风险点**

- 删掉后，旧测试会直接 fail
- 旧 archive config / 历史 replay 脚本如果还走这个 ctor，也会从“兼容忽略”变成 `TypeError`

**删除前 checklist**

- [ ] 先改测试：`tests/train/test_event_motion_model_refactor_phase_d.py` 不再传 `contact_phase_state_*`
- [ ] `rg -n "contact_phase_state_(enable|init_mode|hidden|delta_max|delta_init|event_kind|event_thr|event_hyst|event_min_interval)" .` 只剩 archive docs/config
- [ ] 明确接受“旧 archive config 不再可 replay”，或先做批量 config 清洗
- [ ] 删除后跑 `import train.models` + `EventMotionModel(...)` 最小实例化 smoke

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

---

## 3.7 `train/training_MPL.py` 的 legacy train-entry parser 层

### 3.7.1 `LEGACY_IGNORED_TRAIN_ENTRY_KEYS` / `LEGACY_IGNORED_TRAIN_ENTRY_FLAGS`

**位置**

- keys: `train/training_MPL.py:368`
- flags: `train/training_MPL.py:379`
- 清洗逻辑：`train/training_MPL.py:3857`, `train/training_MPL.py:4241`

**它是什么**

- 这一层不会报错，而是**静默吞掉**旧 key / 旧 CLI flag。
- 当前包含：
  - `diag_input_stats`
  - `eval_angvel_dir_percentile`
  - `eval_horizon`
  - `eval_warmup`
  - `force_valfree_eval`
  - `monitor_batches`
  - `no_monitor`
  - `val_mode`

**静态证据**

- 当前多个活跃 `config/*.json` 仍在使用这些键。
- 文档和 smoke handoff 里也还有旧 CLI flag 示例。

**当前建议**

- `Keep-Active`

**为什么现在不能删**

- 因为它不是“只服务 archive”，而是仍在兜当前 config 载荷。
- 直接删会让现有 training entry 从“容忍旧字段”变成 parse error。

**Fresh-chain update**

- 本轮 basetrain sanitized config 里**仍保留**了以下 8 个 parser-swallowed keys：
  - `diag_input_stats`
  - `eval_angvel_dir_percentile`
  - `eval_horizon`
  - `eval_warmup`
  - `force_valfree_eval`
  - `monitor_batches`
  - `no_monitor`
  - `val_mode`
- 因此这组 swallow layer 对当前主链不是抽象上的“也许还在用”，而是这轮 fresh basetrain 的真实输入面。

**后续转删条件**

- [ ] 活跃 `config/*.json` 批量清理这些 key
- [ ] 活跃 runbook / tools 不再传这些 CLI flag
- [ ] 再把这层从“静默忽略”改成“硬错误”或直接删除

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

### 3.7.2 legacy loss keys 的 reject guard

**位置**

- key set: `train/training_MPL.py:293`
- 报错消息：`train/training_MPL.py:318`
- schedule 审计：`train/training_MPL.py:327`
- 应用点：`train/training_MPL.py:2096`, `train/training_MPL.py:3864`, `train/training_MPL.py:4268`

**它是什么**

- 不是兼容旧行为，而是**明确拒绝**已移除的旧 loss 配置：
  - `ignore_motion_groups`
  - `bone_prior_stds`
  - `use_hierarchy_weights`
  - `hierarchy_mode`
  - `hierarchy_alpha`
  - `max_weight_ratio`
  - `weight_gamma`
  - `bone_prior_mode`
  - `bone_prior_samples`

**静态证据**

- 这些键目前仍在诊断脚本/历史材料中出现。
- 删除 reject guard 会让旧配置更晚炸，或者以更隐蔽方式失败。

**当前建议**

- `Keep-Guard`

**Fresh-chain update**

- 这轮 fresh chain 没有因为 legacy loss keys 被拒绝而报错。
- 结合活跃 config 与日志，当前主链并没有把这组旧 loss keys 喂进 `train.training_MPL`。
- 它对本轮主链是 runtime-cold guard，而不是 runtime-hot compatibility path。

**后续转删条件**

- [ ] 整个 repo（含 tools / archive config）都不再出现这批键
- [ ] 确认你不再需要“友好报错”，可接受 argparse / ctor 其他层级报错

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

---

## 3.8 `train/models.py::MotionJointLoss.__init__` 的 deprecated loss key guard

**位置**

- `train/models.py:3688`

**它是什么**

- 与 `training_MPL` 的 legacy loss 审计配套。
- 如果有人绕过 `training_MPL` 直接实例化 `MotionJointLoss(**legacy_kwargs)`，这里会 fail-fast。

**当前建议**

- `Keep-Guard`

**Fresh-chain update**

- `runtime_compat_hits.json` 里 `MotionJointLoss` 初始化 9 次。
- `legacy_kwargs_calls = 0`，说明当前 fresh chain 没有任何一次真正通过 `MotionJointLoss(**legacy_kwargs)` 打到这层拒绝逻辑。

**删除前 checklist**

- [ ] 确认没有任何直接实例化路径会再传旧 loss keys
- [ ] 确认上游 parser / config loader 已足够严格

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

---

## 3.9 `train/model_ckpt_compat.py`：不要提前删的 direct-pose checkpoint compat 主链

### 3.9.1 这些函数当前仍在主链

- `resume_load_weights_compat` — `train/model_ckpt_compat.py:441`
  - 由 `train/training_MPL.py:4762` 调用
- `_resolve_direct_pose_ckpt_compat_policy` — `train/model_ckpt_compat.py:914`
  - 由 `resolve_direct_pose_build_cfg` 驱动
- `apply_direct_pose_ckpt_compat` — `train/model_ckpt_compat.py:1391`
  - 由 `train/posttrain.py:4234` 调用
- `maybe_upgrade_direct_pose_split_state_dict` — `train/model_ckpt_compat.py:1420`
  - 由 `train/models.py` 里的 adapter 方法消费
- `maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict` — `train/model_ckpt_compat.py:1643`
- `adapt_legacy_event_motion_state_dict` — `train/model_ckpt_compat.py:1681`
- `preprocess_event_motion_state_dict_for_load` — `train/model_ckpt_compat.py:1690`

### 3.9.2 它们在兼容什么

**A. resume 兼容加载**

- `resume_load_weights_compat` 会：
  - 尝试 `model.adapt_legacy_state_dict_(state_dict)`
  - 只加载 shape 能对上的 tensor
  - 输出 `ResumeLoadReport`

**B. direct-pose 头结构升级**

- `maybe_upgrade_direct_pose_split_state_dict`
  - 处理老的 `direct_pose_head.6.*`
  - 处理旧 `direct_pose_out_nonleg.*`
  - 在 split / arm-split / nonleg-proj 结构间迁移权重

- `maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict`
  - 处理旧 `direct_pose_out_leg.*` 到新 `direct_pose_leg_terminal.*`

**C. build/load policy**

- `_resolve_direct_pose_ckpt_compat_policy`
  - 当 ckpt 结构与当前构型 mismatch 时，决定：
    - fatal
    - 还是 drop weights 并重新初始化

- `apply_direct_pose_ckpt_compat`
  - 在 `posttrain` 载入前统一清理/删弃不兼容权重

### 3.9.3 为什么现在不要删

- 这不是“没人用的老代码”，而是现在 `posttrain` / `training_MPL` / `export_onnx` 真在走的兼容迁移层。
- 真正决定能不能删的，不是源码引用，而是**你手上的 checkpoint 语料是否已经全部升级**。

**Fresh-chain update**

- 这轮 fresh chain 的 runtime 命中是：
  - `resume_load_weights_compat = 1`
  - `_resolve_direct_pose_ckpt_compat_policy = 8`
  - `apply_direct_pose_ckpt_compat = 8`
  - `maybe_upgrade_direct_pose_split_state_dict = 8`，其中真正升级 1 次
  - `maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict = 8`，其中真正升级 0 次
  - `adapt_legacy_event_motion_state_dict = 0`
  - `preprocess_event_motion_state_dict_for_load = 0`
- 这说明当前 fresh chain 的“热路径”其实已经能进一步细分：
  - `resume_load_weights_compat` / `_resolve_direct_pose_ckpt_compat_policy` / `apply_direct_pose_ckpt_compat` / `maybe_upgrade_direct_pose_split_state_dict` 仍是硬活跃路径
  - `maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict` 是热检查但本轮没真正改写
  - `adapt_legacy_event_motion_state_dict` / `preprocess_event_motion_state_dict_for_load` 至少对这轮主链是 cold-in-chain

### 3.9.4 删除前必须做的 checkpoint 审计

- [ ] 抽样扫描当前活跃 `.pth`，确认不再包含这些旧 key：
  - `direct_pose_head.6.weight`
  - `direct_pose_head.6.bias`
  - `direct_pose_out_leg.weight`
  - `direct_pose_out_leg.bias`
  - `direct_pose_out_nonleg.weight`
  - `direct_pose_out_nonleg.bias`
- [ ] 代表性基线 ckpt 至少覆盖：
  - basetrain resume ckpt
  - stage6 posttrain ckpt
  - stage70a / 70R / 71 / 72 / lambda ckpt
  - export ONNX 用的代表性 ckpt
- [ ] 兼容层移除后，复跑以下入口：
  - `python3 -m train.training_MPL --help`
  - `python3 -m train.posttrain --help`
  - `python3 -m train.export_onnx_from_ckpt --help`
  - 至少 1 条真实 `posttrain` resume/load 链路

### 3.9.5 建议的 checkpoint key 审计脚本

```python
python3 - <<'PY'
from pathlib import Path
import torch

legacy_keys = {
    "direct_pose_head.6.weight",
    "direct_pose_head.6.bias",
    "direct_pose_out_leg.weight",
    "direct_pose_out_leg.bias",
    "direct_pose_out_nonleg.weight",
    "direct_pose_out_nonleg.bias",
}

for path in Path("models").rglob("*.pth"):
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[skip] {path}: {exc}")
        continue
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        continue
    hits = sorted(k for k in legacy_keys if k in sd)
    if hits:
        print(path)
        for key in hits:
            print("  -", key)
PY
```

**最终勾选**

- [ ] Remove
- [ ] Keep
- [ ] Revisit

---

## 4. 建议的全局删前流程

建议每次只删一类 compat，不要一次性连删。

### Step 1. 静态搜索

- [ ] `rg -n "<target-keyword>" config tools docs tests train`
- [ ] 区分活跃 config / archive config / 文档历史记录
- [ ] 把“archive only” 与 “active path” 分开

### Step 2. 入口 smoke

- [ ] `python3 -m py_compile train/models.py train/model_ckpt_compat.py train/training_MPL.py train/posttrain.py train/layout.py train/utils.py`
- [ ] `python3 -m train.training_MPL --help`
- [ ] `python3 -m train.posttrain --help`
- [ ] `python3 -m train.export_onnx_from_ckpt --help`

### Step 3. 真实最小 replay

- [ ] 1 条 basetrain / resume 链路
- [ ] 1 条 stage6 或 70a `train.posttrain` load 链路
- [ ] 如涉及 ckpt compat，再做 1 条 ONNX 导出或 posttrain warmstart

### Step 4. 决策落表

- [ ] Remove
- [ ] Keep
- [ ] Revisit with more evidence

---

## 5. 我建议的清理顺序

### 第一批：低风险 / 高收益

1. `train/debug_contact_plan_stability.py`
2. `train/posttrain.py` 的 `direct_pose_leg_gate_loss_weight` alias
3. `train/posttrain.py` 的 `direct_pose_leg_gate_mode='auto'`
4. `train/__init__.py::__getattr__`（如果确认没有外部脚本依赖）

### 第二批：需要先清 config/tools

1. `train/train_configurator.py`
2. `train/posttrain.py` 的 `dataset_index_mode <- index_mode`
3. `train/models.py` 的 `contact_phase_state_*` kwargs 壳
4. `train/training_MPL.py` 的 legacy ignored keys/flags

### 最后一批：checkpoint/bundle/layout 契约完全收口之后

1. `train/layout.py` 的 `fallback_to_bone_names`
2. `train/model_ckpt_compat.py` 整个 direct-pose compat family

---

## 6. 备注

- 本文档是**静态审计基线**，不是最终删除授权。
- 对 `model_ckpt_compat.py` 这类逻辑，**源码“没人 import”不等于真的能删**；关键是历史 ckpt 是否还存在、是否还要 replay。
- 对 parser / config alias 类逻辑，优先删“没有 repo 命中”的小 alias，不要一开始就动当前活跃 config 仍在用的 `time_index_mode=auto` 或 train-entry ignored keys。
