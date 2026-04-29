# 2026-04-28 `legacy_checkpoint_compat` 双轨删除计划

Date: 2026-04-28  
Status: Completed locally / mainline strict-current cut landed locally; Gate A0 / Gate A-scope / Gate A1 / Gate B completed locally; post-cut rerun Stage 2 / 3 completed locally; Step 6 migration-only / shared compat 拆层已收口  
Scope: `train/posttrain.py`, `train/posttrain_build_shell.py`, `train/configuration/model_build.py`, `train/checkpoint/load_schema.py`, `train/checkpoint/contract.py`, related `tools/`, `tests/`, `configs/`, `runners*.py` callsites  
Goal: 按 `docs/removal_policy.md` 的 fail-fast 契约，一次提交 / 一个 PR 删除 `legacy_checkpoint_compat` 双轨，不留 silent fallback / warning-only deprecation / runtime compat shim。  
Non-goal:

- 不顺手重命名 `strict_current_model_build`
- 不做 compat/model/runtime semantic cleanup
- 不改 `tools/migrate_legacy_posttrain_ckpt.py` 名字或位置
- 不碰 `archive_posttrain_legacy/**`

---

## 0) TL;DR

这次删除不是“把 legacy 分支代码删掉”这么简单，而是要先证明：

1. 代表性 legacy checkpoint 可以经 `tools/migrate_legacy_posttrain_ckpt.py` 转成 strict checkpoint，并且 strict 路径能真实吃掉。
2. 用户侧所有显式 legacy 开关入口都被 inventory 出来，并标清是“可直接删”还是“需先迁 ckpt 再删”。
3. parse 边界能够在同一提交里钉死 legacy 入口，下游 build/load/metadata 分支同步删掉，不留下 silent fallback。

只有以上 3 条成立，这个删除才符合 `docs/removal_policy.md`。

当前本地执行状态（2026-04-28）：

- 已落地：`train/posttrain.py` parse boundary fail-fast；`--legacy_checkpoint_compat` / `--strict_current_model_build` CLI removal；`train/posttrain_build_shell.py` strict/current 单路 load；strict manifest / shape contract error text 改为 migrate 指引；新 checkpoint 停止写顶层 `legacy_checkpoint_compat`。
- 已落地：共享 `model_build.py` 不再暴露 legacy checkpoint-dependent resolver；legacy-only build resolution 已下沉到 `tools/migrate_legacy_posttrain_ckpt.py`。
- 已验证：`python3 -m unittest tests.train.test_checkpoint_compat_removal` 通过；`python3 tools/check_strict_checkpoint_contract_smoke.py` 通过；相关文件 `py_compile` 通过。
- 已完成：Gate A1 代表 ckpt 已在 `debug_output/_tmp_legacy_ckpt_gateA1_stage1_20260428_220911/` 下完成 migrate、strict contract smoke、最小 `train.posttrain` 启动 smoke；migrated ckpt 不再携带顶层 `legacy_checkpoint_compat`，且带 `resolved_build_manifest` / `_hash`。
- 已完成：Stage 1 幂等性边界已收口为 fail-fast；对已 strict 的 ckpt 再次运行 `tools/migrate_legacy_posttrain_ckpt.py` 现在会报 `[FATAL][AlreadyStrict]`，不再产生 double-migrate 漂移。
- 已完成：post-cut rerun `Stage 2` 已在 `debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/` 下跑完完整 lambda posttrain、freerun eval 与 `group_summary.json`。
- 已完成：post-cut rerun `Stage 3` 已生成 `debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/stage3_compare_lambda_step200.json`；`post-cut` 与 `0425 baseline` 在 5 组 x 4 指标上完全一致，故全量 `delta_vs_0425 = 0.0`。
- Audit 备注：post-cut group_summary 与 0425 group_summary 经 `json.load` 字段级对比，差异仅在 `source` 字段（run-root 路径必然不同），`groups / group_names / mask` 全部 bit-identical（所有 mean/p50/p90/p95 浮点 16 位完全一致）。这是本轮删除"strict-current cut 是接口/契约层改动，不动 model build / train / eval semantic"的最强 audit 证据；未来 reviewer 不应将该一致性误读为压数。
- 已完成：Gate B inventory 已回填；扫描范围内未发现仍显式传 `--legacy_checkpoint_compat` / `--strict_current_model_build` 的 live caller，也未发现 config / runner 侧阻断项。结构化扫描产物见 `debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/gateB_inventory.json`。
- Inventory 已知 stale 项（不影响结论，记录以避免未来 reviewer 误读）：
  - 该 inventory 是 cut 进行中某一时刻的快照；其中 `tools/migrate_legacy_posttrain_ckpt.py:22:from train.checkpoint.compat import ...` 命中已在后续 import 收口中改为 `from train.checkpoint.load_schema import ...`，当前文件不再引用已删除的 `compat` 模块。
  - `tools/run_strict_70r_trunkfull_probe.py` / `run_strict_replace_phasez_boundary_probe.py` / `run_strict_stageb_resolvedcfg_rerun.py` 各 1 处 `"legacy_checkpoint_compat=true"` 命中是**反向探针**（log 关键词扫描），不是 legacy caller，按 scanner 分类不阻断删除。
- 已完成：strict/shared checkpoint load schema helper 已拆到 `train/checkpoint/load_schema.py`；`validate/export/training/models/configuration` 侧 strict/shared caller 已改从新模块 import。
- 已完成：migration-only direct-pose helper 已内联收口到 `tools/migrate_legacy_posttrain_ckpt.py` 所经由的 `train/checkpoint/load_schema.py` shared validator；原 `train/checkpoint/compat.py` / `train/checkpoint/migration_compat.py` 已退役删除。
- 已完成：Step 6 目前不再存在 shared strict-load 与 compat 混层，也没有单独的 migration compat 模块残留。

---

## 1) 删除目标

本次要删除的不是单个函数，而是一整条 legacy 双轨：

- config / payload 层：
  - `legacy_checkpoint_compat`
  - `strict_current_model_build=false`
- CLI 层：
  - `--legacy_checkpoint_compat`
  - `--strict_current_model_build false`
- build resolver 双轨：
  - `resolve_posttrain_model_build_config(...)`
  - `resolve_model_build_config_with_trace(...)`
  - strict/current 与 legacy checkpoint-dependent resolver 双分流
- shell load 双轨：
  - `_strict_current_model_build_enabled(...)`
  - `_LEGACY_STRIPPED_CHECKPOINT_PREFIXES`
  - width-from-checkpoint-tensor 推断
  - legacy warning / fallback path
- ckpt load compat：
  - `_apply_direct_pose_ckpt_compat(...)`
  - `compat` 模块对 `DirectPoseLoadCompatOptions` 的持有/导出
  - 其历史孤儿 helper / 常量
- ckpt metadata 双轨：
  - 顶层 `legacy_checkpoint_compat`
  - 所有提示 “or explicitly set legacy_checkpoint_compat=true” 的错误文案

严格保留的新契约：

- `resolved_build_manifest`
- `resolved_build_manifest_hash`
- strict/current 的 fail-fast manifest / shape contract
- `tools/migrate_legacy_posttrain_ckpt.py`

---

## 2) 删除前必须通过的 gates

### Gate A0. 现场审计事实（2026-04-28）

下列三条作为本次删除计划 `Gate A0` 的现场审计事实，不是推断：

1. `resolved_build_manifest` / `resolved_build_manifest_hash` / `_LEGACY_STRIPPED_CHECKPOINT_PREFIXES` / `strict_current_model_build` 整套 strict-current 契约**截至 2026-04-28 仅存在于本地 working tree，在 `git log` 可追溯 code history 中缺席**。结论：仓内已存在的所有 ckpt 都缺 manifest。
2. `contact_plan_input_proj.` 在 `git log` 全历史中**从未作为 `self.<x>` 活属性出现**；首次出现于 commit `bbce618`（2026-04-18）即作为 strip prefix。审计结论：`contact_plan_input_proj.` 从无活属性、无 live owner，本仓库代码从未产出该 prefix。
3. `debug_output/**/*.pt` 全量扫描共 **328 例**：
   - 含 `frozen_encoder.*` state-dict key：**0 例**
   - 含 `contact_plan_input_proj.*` state-dict key：**0 例**
   - 缺 `resolved_build_manifest`：**328 / 328**

以上事实共同指向：截至本轮，**没有找到任何“必须依赖 C2 / C3 strip 或 `_apply_direct_pose_ckpt_compat` 才能继续活着”的当前主线对象证据**。

### Gate A-scope. Off-repo legacy ckpt 范围决定

本计划以如下作用域假设作为硬边界：

- post-cut 当前主线**不会有 off-repo legacy ckpt 流入** strict/current load 路径（无训练服务器残留 ckpt 待加载，`archive_posttrain_legacy/**` 已在 non-goal 中排除）。
- 因而 C2 / C3（`frozen_encoder.*` / `contact_plan_input_proj.*` strip）与 `_apply_direct_pose_ckpt_compat` **一律按 dead-protection 处理**，不再作为 live object 保护面。
- 如果这个作用域假设被打破，**删除工作必须立即停止**；不得继续按当前计划推进，也不得把 optional gate 反向升级成删除前提来替代该边界。

### Gate A1. Manifest-only live smoke（仅此一条仍要求 live smoke）

代表性 ckpt（已固定，不允许执行人临时挑选）：

- 路径：`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_tail_top7_fresh_chain_step360_20260425_030401/72_lr_branch_cmp/from71_lr1e4_s20_s120/checkpoints/72_main/ckpt_step_000150_WalkF_stage7_72_from71_lr1e4_s20_s120_20260425_030401.pth`
- 选择理由：该 ckpt 来自 `debug_output/**`，且与 `0425 / 0426 / 0427` 参考链同源；它同时是 post-cut rerun plan §1.A `0425 baseline` 字段 `ckpt=` 指向的 donor，也是 Stage 2 完整 posttrain rerun 的输入。Gate A1 migrate 后产出的 strict ckpt 直接被 Stage 2 复用，"Gate A1 / Stage 1 / Stage 2 同一对象"链路在物理路径层面闭合。
- 候选 alternatives 已显式排除：
  - `lambda_main/ckpt_step_000200_*`（0425 lambda lane endpoint）：是 0425 group_summary 的产出 artifact，但不是 Stage 2 的输入；选它会让 Gate A1 与 Stage 2 落到不同对象上。
  - 其他 `debug_output/**/*.pt`：与 0425/0426/0427 比对链路无血缘，post-cut rerun 无法保持同源。
- Gate A1 的 live smoke 入场条件**只保留一条**：该 ckpt 缺 `resolved_build_manifest` / `resolved_build_manifest_hash`。`frozen_encoder.*` / `contact_plan_input_proj.*` 不再作为 Gate A1 前提；这两项仅由 Gate A0 审计事实覆盖。

必跑链路：

1. `tools/migrate_legacy_posttrain_ckpt.py`
2. `tools/check_strict_checkpoint_contract_smoke.py`
3. 最小 `train.posttrain` 启动 smoke
4. 确认 strict/current load 路径命中，而不是 legacy fallback

通过条件：

- migrate 工具成功产出 strict checkpoint（带 `resolved_build_manifest` + `_hash`）
- strict contract smoke 成功
- `train.posttrain` 最小启动成功
- 新 ckpt 不再依赖 legacy branch

如果 Gate A1 过不了，**本轮不允许删 legacy 双轨**。

Gate A1 与 post-cut rerun plan 中的 `Stage 1` 必须使用**同一个**代表性 ckpt、同一份 migrate 产物、同一组 smoke output 路径；不允许“删除前 gate 用 A ckpt，删除后 rerun 用 B ckpt”。

### Gate A-optional. 合成 ckpt 兜底（可选，不阻断本轮）

仅当未来发现仍有 off-repo legacy ckpt 流入需求时，再补：用 `attach_motion_encoder_bundle` 注入合成的 `frozen_encoder.*` 张量，构造合成 legacy ckpt，跑一次 migrate smoke 验证 strip 路径。本轮可做，但不阻断；不得把该 optional gate 变成删除前提。

### Gate B. 用户侧 callsite inventory

必须全 repo grep 并分类：

- `legacy_checkpoint_compat=true`
- `--strict_current_model_build false`
- `_apply_direct_pose_ckpt_compat`
- `_LEGACY_STRIPPED_CHECKPOINT_PREFIXES`
- `_strict_current_model_build_enabled`
- `resolve_posttrain_model_build_config`
- `resolve_model_build_config_with_trace`
- `_filter_checkpoint_state_dict`
- `_RETIRED_DIRECT_POSE_STEPC_LEG_PREFIXES`
- `_DIRECT_POSE_WEIGHT_PREFIXES`
- `STRICT_BRANCH_UNLOAD_CHANGE`
- `STRICT_DIRECT_POSE_SHAPE_INFERENCE_UNLOAD_CHANGE`
- `"legacy_checkpoint_compat=true"`（错误文案）
- `"--legacy_checkpoint_compat"`
- `"--strict_current_model_build"`

扫描范围：

- `tools/`
- `configs/`
- `runners*.py`
- `tests/`

每个命中都要标成两类之一：

- `Delete-Now`: 可随代码同提交删除
- `Migrate-First`: 需要先迁 ckpt / 改 fixture / 改 caller，然后才能删

额外要求：

- `runners*.py` / `tools/*.py` 里凡是还显式传 `--legacy_checkpoint_compat` 或 `--strict_current_model_build` 的活脚本，必须在删 flag 前先改 caller；不能指望删完以后让这些 live caller 靠 argparse unknown argument 才暴露。
- `STRICT_BRANCH_UNLOAD_CHANGE` / `STRICT_DIRECT_POSE_SHAPE_INFERENCE_UNLOAD_CHANGE` 这两个 marker 字符串默认**保留**，作为 post-cut `[Removed]` 报错锚点；若有人主张删除，必须单独说明替代锚点。
- `_filter_checkpoint_state_dict` 的 inventory 结果必须回答：删完 legacy 分支后它是 inline 掉，还是仍保留为多调用点 helper。

没有这份 inventory，不进入删除实现阶段。

当前状态（2026-04-28）：**completed**

- 结构化扫描产物：`/Users/xingzhaorui/PycharmProjects/PythonProject/debug_output/_tmp_legacy_ckpt_postcut_stage2_20260428_223017/gateB_inventory.json`
- 结论：扫描范围 `tools/ / configs/ / tests/ / runners*.py` 内，未发现仍显式传 `--legacy_checkpoint_compat`、`--strict_current_model_build`、或 `strict_current_model_build false` 的 live caller；`configs/` 与 `runners*.py` 在本 gate 关键词下均无阻断命中。
- 仍有命中的只剩 3 类非阻断项：
  - removed-boundary 负测：`tests/train/test_checkpoint_compat_removal.py`、`tools/check_strict_checkpoint_contract_smoke.py`
  - migration-only helper：`tools/migrate_legacy_posttrain_ckpt.py`
  - audit-only log-scan 字段：`tools/run_strict_70r_trunkfull_probe.py`、`tools/run_strict_stageb_resolvedcfg_rerun.py`、`tools/run_strict_replace_phasez_boundary_probe.py`
- 因此 Gate B 的“先识别 live caller、再判断是否阻断删除”的目标已经满足；剩余 shared compat 收尾是否继续做，转入 Step 6 范围，不再阻断 Gate B。

---

## 3) 被删双轨当前仍消费的状态

按 `docs/removal_policy.md`，先列出 legacy 双轨消费的状态，而不是先改代码。

### 3.1 Config / payload

- `legacy_checkpoint_compat`
- `strict_current_model_build`
- `load_context`
- strict rejectors 当前仍按 `strict_current` 条件早退

### 3.2 CLI flags

- `--legacy_checkpoint_compat`
- `--strict_current_model_build`

### 3.3 Checkpoint top-level metadata

- `legacy_checkpoint_compat`
- `strict_current_model_build`
- 缺失 `resolved_build_manifest`
- 缺失 `resolved_build_manifest_hash`
- legacy 路径仍可依赖 `posttrain_cfg`

### 3.4 Checkpoint state / load-time behavior

- `_LEGACY_STRIPPED_CHECKPOINT_PREFIXES`
- width-from-`shared_encoder.0.weight`
- direct-pose checkpoint compat patching
- strict=False legacy load

### 3.5 Runtime / helper selectors

- `_strict_current_model_build_enabled(...)`
- strict / legacy build resolver split
- strict / legacy load split
- `_filter_checkpoint_state_dict(...)`

### 3.6 Errors / operator guidance

- 所有 “or explicitly set legacy_checkpoint_compat=true”
- legacy warning / fallback 提示
- `shape/posttrain_cfg inference` 相关 legacy 文案
- `STRICT_BRANCH_UNLOAD_CHANGE`
- `STRICT_DIRECT_POSE_SHAPE_INFERENCE_UNLOAD_CHANGE`

---

## 4) 删除形状（一次提交 / 一个 PR）

这次删除按“先在 parse 边界钉死 legacy，再同步清掉下游实现”的形状做。

### Step 1. Parse boundary fail-fast

目标：

- `legacy_checkpoint_compat=true` 直接 `[Removed]`
- `strict_current_model_build=false` 直接 `[Removed]`

要求：

- 不再保留默认耦合：
  - 不再通过 `default=(not legacy_checkpoint_compat)` 推导 strict
  - 不再通过 `if not strict_current: legacy_checkpoint_compat=True` 回写 legacy
- `PostTrainConfig` 中删除 `legacy_checkpoint_compat`
- `strict_current_model_build` 可以保留为名字冗余但单轨常量语义；不在本轮顺手改名

当前状态（2026-04-28）：**已落地**

### Step 2. CLI removal

目标：

- 删除 `--legacy_checkpoint_compat`
- 删除 `--strict_current_model_build`

要求：

- 不加软兼容
- 老脚本继续传这些参数时，让 argparse 默认报 unknown argument
- 但在真正删 flag 之前，Gate B 里识别出的 live caller 必须先改完

当前状态（2026-04-28）：**已落地**

### Step 3. Rejectors become unconditional

目标：

- 当前 `_cfg_reject_strict_*` rejectors 去掉 `if not strict_current: return`
- 名字去掉 `_strict_` 前缀，避免单轨后残留双轨术语

要求：

- rejectors 现在永远生效
- 不引入默认兜底

当前状态（2026-04-28）：**已基本落地**

- 当前主线已无合法 non-strict 入口；`legacy_checkpoint_compat` 与 `strict_current_model_build=false` 在 parse boundary 即 fail-fast。
- `_cfg_reject_strict_*` 命名尚未统一去掉 `_strict_` 前缀；这是语义收尾，不阻断当前单轨主线。

### Step 4. Build resolver单路化

目标：

- 删除 `resolve_posttrain_model_build_config(...)`
- 删除 `resolve_model_build_config_with_trace(...)`
- shell 只走 `resolve_current_model_build_config_with_trace(...)`

要求：

- 不保留 legacy wrapper 壳
- 不保留 checkpoint-dependent semantic inference

当前状态（2026-04-28）：**已落地**

### Step 5. Shell load单路化

目标：

- 删除 `_strict_current_model_build_enabled(...)`
- 删除 `_LEGACY_STRIPPED_CHECKPOINT_PREFIXES`
- 删除 legacy width-from-tensor 推断
- 删除 legacy warning / fallback 文案

要求：

- 若 `_filter_checkpoint_state_dict(..., ignored_prefixes=())` 只剩单个空调用点，则 inline
- strict/current 边界继续 fail-fast
- inventory 必须先回答 `_filter_checkpoint_state_dict` 是否仍有多个非 legacy 调用点

当前状态（2026-04-28）：**已落地**

### Step 6. Ckpt compat load分支删除

目标：

- `_load_posttrain_checkpoint_into_model(...)` 只剩 strict/current 单分支
- 删除 compat-owned `_apply_direct_pose_ckpt_compat(...)` wrapper / 中转层
- 把 `DirectPoseLoadCompatOptions` 收口为 `train/checkpoint/load_schema.py` 的 shared load 输入，而不是 compat 模块资产

要求：

- 先确认没有其他 live caller
- 如果 `compat.py` 因此出现孤儿 helper / 前缀常量，同 commit 一起清理
- `compat.py` 的去留必须二选一并在实施前钉死：
  - 保留为 retired-key / removed-boundary 的薄壳
  - 或连壳一起删，只保留仍有 live owner 的 helper
- 不允许删掉 `_apply_direct_pose_ckpt_compat` 后，把 `compat.py` 留成 re-export / import 中转站

当前状态（2026-04-28）：**已落地**

- `train/posttrain_build_shell.py` 的 runtime load 分支已不再调用 `_apply_direct_pose_ckpt_compat(...)`，主线只剩 strict/current 单分支。
- 本轮已把 strict/shared load schema helper 拆到 `train/checkpoint/load_schema.py`；shared strict caller 不再经过 legacy compat 模块。
- migration-only 调用点已收口为 `tools/migrate_legacy_posttrain_ckpt.py` 直接调用 `train/checkpoint/load_schema.py` 中的 shared validator，不再保留单独 migration compat 模块。
- strict load 入口 `prepare_event_motion_ckpt_state_for_load(...)` 已直接走 `train/checkpoint/load_schema.py` 内的 shared schema-only normalize / retired-boundary reject helper。
- `train/checkpoint/compat.py` / `train/checkpoint/migration_compat.py` 均已删除；仓内不再存在 compat 中转层。
- 已验证：`py_compile`、`tests.train.test_checkpoint_compat_removal`、`tests.train.test_event_motion_model_refactor_phase_d`、`tools/check_strict_checkpoint_contract_smoke.py`、shared/migration import smoke 均通过。

### Step 7. Checkpoint metadata收口

目标：

- 停止写顶层 `legacy_checkpoint_compat`
- 所有 post-cut 新 checkpoint **无条件**写出：
  - `resolved_build_manifest`
  - `resolved_build_manifest_hash`

要求：

- 不允许残留 “strict 才写、legacy 不写” 的条件 save 逻辑
- post-cut 后不存在合法的新 ckpt 缺失这两项 strict contract 字段

当前状态（2026-04-28）：**已落地**

### Step 8. Error text统一迁移口径

目标：

- 所有 `legacy_checkpoint_compat=true` operator hint 改成 migrate 指引

要求：

- 不再暗示“还能开 legacy 开关兜底”
- 文案统一指向 `tools/migrate_legacy_posttrain_ckpt.py`

当前状态（2026-04-28）：**已落地**

---

## 5) 删除实现前的 callsite 分类模板

本轮 inventory 实际结果如下：

| Entry | File | Kind | Classification | Action before deletion |
|---|---|---|---|---|
| `legacy_checkpoint_compat=true` | `tests/train/test_checkpoint_compat_removal.py` | removed-boundary negative test | `Delete-Now` | 非 live caller；可继续保留为 fail-fast regression，也可在测试退役时一并删除。 |
| `legacy_checkpoint_compat=true` | `tools/check_strict_checkpoint_contract_smoke.py` | smoke negative case | `Delete-Now` | 非 live caller；继续保留可覆盖 removed-boundary，自身不阻断删除。 |
| `"legacy_checkpoint_compat=true"` | `tools/run_strict_70r_trunkfull_probe.py`, `tools/run_strict_stageb_resolvedcfg_rerun.py`, `tools/run_strict_replace_phasez_boundary_probe.py` | audit-only log-scan field name | `Delete-Now` | 仅用于 grep/log 审计；不参与 runtime。可在后续 probe 清理时统一改名或删除。 |
| `_apply_direct_pose_ckpt_compat` | `tools/migrate_legacy_posttrain_ckpt.py` | 历史 migration-only helper 名称 / 报错上下文 | `Delete-Now` | 已不再作为独立 helper 存在；迁移工具直接调用 `train/checkpoint/load_schema.py` shared validator，并保留该 context 字符串。 |
| `_LEGACY_STRIPPED_CHECKPOINT_PREFIXES` | `tools/migrate_legacy_posttrain_ckpt.py` 中 `_MIGRATION_LEGACY_STRIPPED_CHECKPOINT_PREFIXES` | migration-only strip-list constant | `Migrate-First` | 保留在迁移工具内部；无 runtime caller。 |
| `_filter_checkpoint_state_dict` | `tools/migrate_legacy_posttrain_ckpt.py` | shared helper consumed by migration tool | `Migrate-First` | 当前仍有多调用点且被迁移工具使用；只有在迁移工具退役时才考虑 inline/remove。 |

说明：

- `Delete-Now`：删代码时可以一起处理
- `Migrate-First`：必须先迁 ckpt / 改 fixture / 改生成链路，否则本轮不能合并
- 零命中项：`--strict_current_model_build false`、`"--legacy_checkpoint_compat"`、`"--strict_current_model_build"`、`_strict_current_model_build_enabled`、`resolve_posttrain_model_build_config`、`resolve_model_build_config_with_trace`、`_RETIRED_DIRECT_POSE_STEPC_LEG_PREFIXES`、`_DIRECT_POSE_WEIGHT_PREFIXES`、`STRICT_BRANCH_UNLOAD_CHANGE`、`STRICT_DIRECT_POSE_SHAPE_INFERENCE_UNLOAD_CHANGE`。

---

## 6) 必须新增 / 改写的 fail-fast 边界

按删除政策，至少要在以下入口显式 reject：

1. `train/posttrain.py` payload parse
   - `legacy_checkpoint_compat=true`
   - `strict_current_model_build=false`
2. argparse 边界
   - 老 flags 直接 unknown argument
3. 任何仍会首次读取 legacy top-level metadata 的地方
   - 若保留 metadata field 读取逻辑，应明确 reject，而不是 silent ignore
4. 任何测试 / tool fixture 若还试图构造 legacy-only config
   - 改成断言 `[Removed]`

错误信息必须包含：

- 被删字段
- 删除日期 / 提交边界
- 迁移路径：`tools/migrate_legacy_posttrain_ckpt.py`

---

## 7) 测试变更计划

本轮需要同步处理的测试：

- `tests/train/test_checkpoint_compat_removal.py`
  - 任何仍断言 legacy 行为的 case：
    - 改成断言 parse `[Removed]`
    - 或直接删除
- 任何 fixture 若依赖 `legacy_checkpoint_compat=true`
  - 先经 migrate 工具升级
  - 不允许改成 strict 下 silent shim

切后最少要跑：

1. `tests.train.test_checkpoint_compat_removal`
2. `tests.train.test_checkpoint_fingerprint_phase5_enforce`
3. `tests.train.test_strict_replace_phasez_boundary_probe`
4. 新增一条：
   - `legacy_checkpoint_compat=true` 必须 fail-fast

当前状态（2026-04-28）：

- 已跑：`tests.train.test_checkpoint_compat_removal`
- 已跑：`tools/check_strict_checkpoint_contract_smoke.py`
- 尚未回填：`tests.train.test_checkpoint_fingerprint_phase5_enforce` / `tests.train.test_strict_replace_phasez_boundary_probe`

---

## 8) 完成条件

满足以下条件后，这个删除才算可提交：

1. Gate A0 / Gate A-scope / Gate A1 / Gate B 全部完成
2. legacy config / CLI 入口全部 fail-fast
3. build/load/metadata 单轨化完成
4. 仓内不再出现 live operator hint：
   - `legacy_checkpoint_compat=true`
   - `--strict_current_model_build false`
5. migrate 工具保留且可用
6. post-cut 完整 rerun 按单独 rerun plan 跑完，并出对比回报

---

## 9) 回滚条款

post-cut rerun 若出现需要阻断的行为回退：

- 回滚整个删除 PR
- 不允许用 silent compat shim / warning-only deprecation / default fallback 续命
- 若确需保留迁移能力，只能回到显式 migrate 工具或新的 fail-fast 边界设计，不回填 legacy runtime branch

---

## 10) PR 描述里必须带的 removal boundary

提交 / PR 描述必须显式写：

```text
removal boundary:
- removed: legacy_checkpoint_compat dual-track (config/CLI/build/load/ckpt metadata)
- rejection enforced at: train/posttrain.py parse boundary + argparse boundary
- migration: tools/migrate_legacy_posttrain_ckpt.py
- last-supported-commit: <fill before merge>
```

---

## 11) 本文档不做的事

- 不定义最终指标阈值
- 不替代 post-cut rerun report
- 不在本轮顺手做 `strict_current_model_build` 命名清理
- 不替代 migrate 工具本身的实现文档
