# [2026-04-18] `train/` 文件夹重分层执行计划

Date: 2026-04-18  
Status: Active / Draft v1  
Owner: train structure cleanup  
Scope: `train/`, `tools/`, compatibility shims, import rewiring  
Goal: 在**不提前拆大文件**、**不改变训练语义**的前提下，先把 `train/` 的目录层次整理清楚，让 runtime/config/data/contracts/checkpoint 的边界更明确，并为后续“清理无用分支”后再拆巨文件创造条件。  
Non-goals: 本轮不拆 `train/models.py` / `train/posttrain.py` / `train/training_MPL.py` / `train/pretrain_mpl_min.py`；不为了拆分而拆分；不保留无限期 shim；不改变 checkpoint contract。

---

## 0. 一页版结论

本计划采用 **shim-first + phased relocation**：

1. 先迁**边界已经清楚**的小模块。  
2. 先改**低风险调用点**，只有在影响面太大时才保留 shim。  
3. 每个 phase 都必须定义 **shim 清理门槛**，避免 re-export 永久残留。  
4. `models.py` 的未来目录结构**先不定型**，等无用分支清理后再判断自然边界。  

当前推荐 phase 顺序：

- Phase 0: 清 `train/configuration/` 中的离线 config-builder
- Phase 1: 引入 `train/runtime/freeze.py`
- Phase 2: 引入 `train/contracts/asset_semantics.py`
- Phase 3: 引入 `train/data/`（先迁小边界，再看 `dataset.py`）
- Phase 4: 引入 `train/checkpoint/`
- Phase 5: 无用分支清理完成后，再评估大文件拆分

---

## 1. 总原则

### 1.1 不为了拆分而拆分

只在以下条件同时满足时才迁模块/改目录：

- 目标模块职责相对稳定；
- 迁移后边界更清晰；
- 调用方 rewiring 风险可控；
- 不会干扰后续的无用分支清理。

以下情况**不做**目录拆分：

- 只是因为文件短；
- 只是因为根目录看起来“平铺”；
- 只是为了凑一个更“漂亮”的树；
- 迁移后需要长期依赖 shim 才能工作；
- 当前代码形状明显仍会被后续清分支大幅改变。

### 1.2 优先改调用点，而不是优先保 shim

compatibility shim 不是默认策略，只是 fallback。

优先级：

1. **直接改调用点**（低风险、调用点少、可一轮改完）  
2. **短期 shim**（调用面较广、当轮不适合全部 rewiring）  
3. **长期保留旧路径**（原则上禁止）  

判定建议：

- 如果 import 调用点 `<= 5`，优先直接改调用点；
- 如果 import 调用点较广，允许 shim，但必须登记清理门槛；
- 如果模块是主链 runtime API，则优先让新路径成为唯一正式路径，旧路径最多保留过渡层。

### 1.3 Shim 不是永久资产

每个 phase 必须写清：

- 哪些文件允许用 shim；
- shim 最多保留到哪个阶段；
- 何时删除；
- 用什么 grep/计数作为删除门槛。

### 1.4 同名模块必须写清职责 docstring

对于文件名相同、但语义完全不同的模块，落地迁移时必须在文件顶部补 1~2 行 docstring，明确职责边界。

本计划至少包括：

- `train/configuration/io.py`：只负责 config JSON I/O
- `train/data/io.py`：只负责 dataset/source metadata / trajectory / runtime data helpers

如果缺失这类 docstring，视为对应 phase 未完全落地。

---

## 2. 当前建议目标结构

说明：这是**分层方向**，不是要求一次性全部落地；其中大文件先不动。

```text
train/
  __init__.py

  configuration/
    __init__.py
    io.py
    norm_spec.py

  data/
    __init__.py
    io.py
    layout.py
    normalizers.py
    contact_signals.py
    dataset.py                # later

  contracts/
    __init__.py
    asset_semantics.py

  checkpoint/
    __init__.py
    contract.py
    compat.py

  motion/
    __init__.py
    geometry.py               # later
    history.py                # later

  runtime/
    __init__.py
    freeze.py
    diagnostics.py            # later
    evaluation.py             # later

  models.py
  training_MPL.py
  posttrain.py
  pretrain_mpl_min.py
  convert_json_to_npz.py
  export_onnx_from_ckpt.py
```

明确约束：

- 本计划**不预先承诺** `train/models.py` 将来一定拆成 `train/modeling/`。
- 要等 Phase 5 无用分支清理后，再根据剩余代码的自然边界决定。

---

## 3. 模块分层判断

### 3.1 应保留在 `train/configuration/` 的

- `train/configuration/io.py`
- `train/configuration/norm_spec.py`

它们属于 runtime config 语义，已经进入主链。

### 3.2 应迁出 `train/configuration/` 的

- `train/configuration/cli.py`
- `train/configuration/profile.py`
- `train/configuration/stages.py`
- `train/configuration/__main__.py`

这套本质是 **offline config builder**，更适合迁到：

```text
tools/config_builder/
  __init__.py
  __main__.py
  cli.py
  profile.py
  stages.py
```

### 3.3 应收敛为新命名空间但暂不拆巨文件的

- `train/posttrain_common.py` → `train/runtime/freeze.py`
- `train/rotvec_semantics.py` → `train/contracts/asset_semantics.py`
- `train/io.py` → `train/data/io.py`
- `train/layout.py` → `train/data/layout.py`
- `train/normalizers.py` → `train/data/normalizers.py`
- `train/ttc.py` → `train/data/contact_signals.py`
- `train/model_ckpt_contract.py` → `train/checkpoint/contract.py`
- `train/model_ckpt_compat.py` → `train/checkpoint/compat.py`

### 3.4 先不动的巨文件

- `train/models.py`
- `train/posttrain.py`
- `train/training_MPL.py`
- `train/pretrain_mpl_min.py`

理由：

- 这些文件后续还要配合“无用分支清理”一起看；
- 现在先搬目录容易按旧形状切坏未来边界；
- 当前优先级应是目录收层次，而不是提前切大块。

---

## 4. Phase 设计

### Phase 0 — 清理 `train/configuration/`

**目标**

把 runtime config 与 offline config builder 分开。

**迁移**

- `train/configuration/cli.py` → `tools/config_builder/cli.py`
- `train/configuration/profile.py` → `tools/config_builder/profile.py`
- `train/configuration/stages.py` → `tools/config_builder/stages.py`
- `train/configuration/__main__.py` → `tools/config_builder/__main__.py`

**保留**

- `train/configuration/io.py`
- `train/configuration/norm_spec.py`

**对 `train/configuration/__init__.py` 的要求**

只 re-export runtime config API，例如：

- `load_json`
- `dump_json`
- `NORM_SPEC_RUNTIME_PRETRAIN_KEYS`
- `ContactPretrainRuntime`
- `merge_norm_spec`
- `parse_pretrain_contact_affine_spec`
- `resolve_contact_pretrain_runtime`

**本 phase 原则**

- 尽量直接改 builder 相关调用点；
- `train/configuration.__main__` 可以保留一个短期 wrapper，以兼容 `python -m train.configuration`；
- 不要再从 `train.configuration.__init__` 暴露 `DatasetProfiler` / `TrainingConfigBuilder`。

**验收**

- `python3 -m py_compile train/configuration/__init__.py train/configuration/io.py train/configuration/norm_spec.py`
- `python3 -m py_compile tools/config_builder/cli.py tools/config_builder/profile.py tools/config_builder/stages.py`
- `python3 -m tools.config_builder --dry-run`
- 如果保留 wrapper：`python3 -m train.configuration --dry-run`
- import smoke：

```bash
python3 - <<'PY'
from train.configuration import load_json, dump_json
from train.configuration.norm_spec import merge_norm_spec, resolve_contact_pretrain_runtime
print("runtime configuration imports ok")
PY
```

**Shim 清理门槛**

- `rg -n "from train\\.configuration import .*DatasetProfiler|TrainingConfigBuilder|STAGE_TEMPLATE|compute_total_epochs|compute_batch_size|compute_base_lr" train tools tests`
- `rg -n "python -m train\\.configuration|train/configuration/__main__" tools tests docs`
- 当结果为 `0` 时：
  - 删除 `train/configuration/__main__.py` wrapper（若仍存在）
  - 保持 `train/configuration/__init__.py` 只暴露 runtime API

---

### Phase 1 — 引入 `train/runtime/freeze.py`

**目标**

让 `posttrain_common.py` 从“名字不准”转为明确的 runtime/freeze 命名空间。

**迁移**

- `train/posttrain_common.py` → `train/runtime/freeze.py`

**保留策略**

- 本阶段允许 `train/posttrain_common.py` 保留 shim
- shim 内容只允许：
  - `from train.runtime.freeze import ...`

**迁移内容**

- `_freeze_all`
- `_enable_modules`
- `_unfreeze_direct_pose`
- `_select_trainable_params`
- `_unfreeze_for_train_mode`

**本 phase 原则**

- 如果 import 调用点很少，直接改调用点；
- 如果 tools/脚本较多，先用 shim 顶住，但只允许短期存在；
- 不在本阶段再往这个模块塞 config helper。

**验收**

- `python3 -m py_compile train/runtime/freeze.py train/posttrain_common.py`
- `rg -n "from train\\.posttrain_common|from \\.posttrain_common" train tools tests`
- 最小 smoke：复用现有 posttrain 相关最小路径或独立 Python smoke，覆盖 `_freeze_all(...)` + `_unfreeze_for_train_mode(...)` + `_select_trainable_params(...)` 的组合行为
- 验收标准：
  - `train_mode="direct"` 时只解冻 direct-pose 目标模块
  - `train_mode="lambda"` 时只解冻 `lambda_fusion_head`

**Shim 清理门槛**

- 当 `from train.posttrain_common` 的结果只剩 `0~2` 处、且都在 archive/临时工具之外时：
  - 全部改到 `train.runtime.freeze`
  - 删除 `train/posttrain_common.py`

---

### Phase 2 — 引入 `train/contracts/asset_semantics.py`

**目标**

让 `rotvec_semantics.py` 的职责用更准确的名字表达。

**迁移**

- `train/rotvec_semantics.py` → `train/contracts/asset_semantics.py`

**保留策略**

- 允许 `train/rotvec_semantics.py` 保留 shim

**迁移内容**

- `STANDARD_ROTVEC_SEMANTICS`
- `STANDARD_ANGVEL_SEMANTICS`
- `LEGACY_ROTVEC_SEMANTICS`
- `get_rotvec_semantics`
- `get_angvel_semantics`
- `stamp_standard_rotvec_spec`
- `require_standard_rotvec_spec`
- `require_standard_rotvec_bundle`

**本 phase 原则**

- 不把它并进 `normalizers.py`
- 不把它并进 `configuration/`
- 它属于 cross-cutting contract，不属于某个单点功能附属

**验收**

- `python3 -m py_compile train/contracts/asset_semantics.py train/rotvec_semantics.py`
- `rg -n "from train\\.rotvec_semantics|from \\.rotvec_semantics" train tools tests`

**Shim 清理门槛**

- 当非 archive / 非 docs 的 `from train.rotvec_semantics` 结果为 `0` 时：
  - 删除 root shim

---

### Phase 3 — 引入 `train/data/`

**目标**

把 data/feature runtime 收到同一命名空间下，但先不强拆 `dataset.py`。

**迁移顺序**

先迁小模块：

- `train/io.py` → `train/data/io.py`
- `train/layout.py` → `train/data/layout.py`
- `train/normalizers.py` → `train/data/normalizers.py`
- `train/ttc.py` → `train/data/contact_signals.py`

后迁大模块（可延期）：

- `train/dataset.py` → `train/data/dataset.py`

**保留策略**

- root 旧文件允许短期 shim
- `dataset.py` 可晚于其依赖模块迁移

**本 phase 原则**

- `ttc.py` 改名为 `contact_signals.py`，强调它不只是 TTC
- 不把 `ttc.py` 并回 `dataset.py`
- 不把 `io.py` 与 `configuration/io.py` 物理合并
- `train/configuration/io.py` 与 `train/data/io.py` 必须各自补职责 docstring
- 如果 `dataset.py` 调整过大，允许延后到 Phase 3b

**验收**

- `python3 -m py_compile train/data/io.py train/data/layout.py train/data/normalizers.py train/data/contact_signals.py`
- `python3 -m py_compile train/io.py train/layout.py train/normalizers.py train/ttc.py`
- `rg -n "from train\\.(io|layout|normalizers|ttc)|from \\.(io|layout|normalizers|ttc)" train tools tests`
- 最小 smoke：至少跑一次 dataset / normalizer 相关最小路径，确认迁移后 `train/posttrain.py` 或 `train/training_MPL.py` 能拿到非空 `norm_spec`、layout 解析结果和 normalizer 构造结果；优先到 `_build_dataset_and_loader(...)` 或等价最小构造处止步，不进入真训练

**Shim 清理门槛**

- 对每个 shim 单独计数，不要求同一轮全部删：
  - `train/io.py`
  - `train/layout.py`
  - `train/normalizers.py`
  - `train/ttc.py`
- 当某个旧路径在非 archive / 非 docs 代码中被引用次数为 `0` 时，立即删对应 shim

---

### Phase 4 — 引入 `train/checkpoint/`

**目标**

把 checkpoint contract / compat 收为明确命名空间。

**迁移**

- `train/model_ckpt_contract.py` → `train/checkpoint/contract.py`
- `train/model_ckpt_compat.py` → `train/checkpoint/compat.py`

**保留策略**

- root 旧文件保留 shim

**本 phase 原则**

- 不改 checkpoint 语义
- 不顺手改 `models.py` build/load contract
- 只做 namespace 收口

**验收**

- `python3 -m py_compile train/checkpoint/contract.py train/checkpoint/compat.py`
- `python3 -m py_compile train/model_ckpt_contract.py train/model_ckpt_compat.py`
- `rg -n "from train\\.model_ckpt_(contract|compat)|from \\.model_ckpt_(contract|compat)" train tools tests`

**Shim 清理门槛**

- 当旧 import 只剩 `0~3` 处并可一轮改完时，直接 rewiring 并删 shim

---

### Phase 5 — 无用分支清理后，再评估巨文件拆分

**明确要求**

- 这一阶段之前，不对 `models.py` 的将来目录结构定型；
- 不预设 `train/modeling/`、`train/model/`、`train/architectures/` 哪种更优；
- 必须先完成无用分支清理，再看剩余代码自然边界。

**届时再评估的对象**

- `train/models.py`
- `train/posttrain.py`
- `train/training_MPL.py`
- `train/pretrain_mpl_min.py`

---

## 5. 迁移与 shim 的统一规则

### Rule A — 每次迁模块都必须同时写 shim 删除条件

禁止：

- “先放个 shim，以后再说”

必须：

- 写出 grep 命令
- 写出清理门槛
- 写出最晚删除 phase

### Rule B — shim 只允许 re-export

shim 文件不允许：

- 新增逻辑
- 新增 warning/fallback
- 悄悄修改参数行为
- 做 path-dependent branching

只允许：

- `from new.path import *`
- 或显式列名导出

### Rule C — 优先在活跃代码里 rewiring

活跃代码包括：

- `train/`
- `tools/`
- `tests/`

不优先清理：

- `docs/`
- archive / retired / delete 文档中的路径文字

**执行更新（2026-04-18）**

- `train.posttrain -> train.posttrain_build_shell` 的 build-shell re-export shim 已按本规则完成一次本地清理示例：
  - 先 rewiring 活跃代码中的 `tools/` caller；
  - 再删除 `train/posttrain.py` 顶层对 `_build_posttrain_model_from_ckpt(...)` / `_instantiate_posttrain_model(...)` / `_load_posttrain_checkpoint_into_model(...)` / `_resolve_posttrain_model_build_state(...)` 的 re-export 面。
- 该轮验证使用：
  - `python3 -m py_compile train/posttrain.py train/posttrain_build_shell.py ...`
  - `python3 tools/run_posttrain_build_shell_smoke.py`
- 这类 shim 删除应优先在活跃调用方收口，而不是等 docs/历史说明同步完再删。

---

## 6. 提交策略

原则：

- 不把“已经达到删除门槛的 shim”攒到最后一起删；
- 每个 phase 一旦达到 shim 清理门槛，就地单独提交一次 shim 删除；
- 禁止出现“主迁移已完成，但 shim 还堆在 root 等以后统一收”的情况。

### 建议提交顺序

1. Commit 1: `train/configuration/` runtime-only 收口 + `tools/config_builder/`  
2. Commit 2: 若 Phase 0 门槛已满足，删除 `train/configuration` wrapper / builder 旧暴露  
3. Commit 3: `train/runtime/freeze.py` + `posttrain_common.py` shim  
4. Commit 4: 若 Phase 1 门槛已满足，删除 `train/posttrain_common.py` shim  
5. Commit 5: `train/contracts/asset_semantics.py` + `rotvec_semantics.py` shim  
6. Commit 6: 若 Phase 2 门槛已满足，删除 `train/rotvec_semantics.py` shim  
7. Commit 7: `train/data/` phase 1（`io/layout/normalizers/contact_signals`）  
8. Commit 8: 若 Phase 3 中某个 shim 已达门槛，就地删除对应 shim  
9. Commit 9: `train/checkpoint/` phase 1  
10. Commit 10: 若 Phase 4 门槛已满足，删除 checkpoint shim  

---

## 7. 最小执行建议

如果只做一轮最小结构整理，我建议只做：

1. Phase 0  
2. Phase 1  
3. Phase 2  

原因：

- 这三步边界最稳定；
- 对大文件零侵入；
- 对后续无用分支清理干扰最小；
- 可以最快把 `train/` 的“命名不准”问题先解决一半。

---

## 8. 本计划的成功标准

执行完成后，应能回答以下问题且答案清楚：

1. runtime config 在哪里？  
2. data feature/runtime helpers 在哪里？  
3. asset semantics contract 在哪里？  
4. checkpoint contract/compat 在哪里？  
5. 哪些旧路径只是过渡 shim，何时删除？  

如果做完后 root 目录仍残留大量长期 shim、且没有明确删除门槛，则视为本计划失败。
