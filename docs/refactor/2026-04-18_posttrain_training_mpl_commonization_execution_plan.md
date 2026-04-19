# [2026-04-18] `posttrain` / `training_MPL` 共享 runtime/config 收敛执行计划

Date: 2026-04-18  
Status: ✅ Phase A-C landed on 2026-04-18 / Phase D historical-superseded in current tree  
Owner: train refactor cleanup  
Scope: `train/posttrain.py`, `train/runtime/freeze.py`, `train/training_MPL.py`, `train/configuration/`  
Goal: 先收敛 `posttrain` 与 `training_MPL` 之间已经事实共享的 runtime/config 逻辑，消除小范围 drift 与 inline 重实现，再决定是否做第二阶段细拆。  
Non-goals: 本轮不拆 rollout/loss kernel，不统一 `TrainerRuntimeConfig`，不改变训练语义，不改 checkpoint contract，不引入过细碎的小模块。

注：

- 本文中的 `train/posttrain_common.py` 引用属于历史计划痕迹；当前工作树已改用 `train/runtime/freeze.py`。
- 下述 checklist 中，已显式落地/验证的事项会勾选；未单独跑过的 smoke / diff 验证保持未勾选。
- 后续进展补记：Step 4 的 thin-slice rollout kernel 已另行落地到 `train/rollout_kernel.py`；本文件仍主要记录 Phase A-C 的 shared runtime/config 收敛背景。

---

## 0. 一页版结论

本轮采用“**先杂糅收敛，再二拆**”策略，而不是直接把 shared 逻辑拆成很多小文件。

本轮只做三件事：

1. 把 `spec load + norm merge` 收到共享模块。  
2. 把 `contact-pretrain hydration` 收到共享模块。  
3. 保持 `TrainerRuntimeConfig` 与 `Trainer` runtime apply 路径不动。  

已经完成的预备动作：

- `train/posttrain.py` 中的 `_unfreeze_for_train_mode(...)` 已迁到 `train/runtime/freeze.py`，说明 freeze/train-mode policy 已开始从 entry file 收口。

本轮推荐最终落点：

- `train/configuration/norm_spec.py`
  - `NORM_SPEC_RUNTIME_PRETRAIN_KEYS`
  - `merge_norm_spec(..., strict=...)`
  - `parse_pretrain_contact_affine_spec(...)`
  - `resolve_contact_pretrain_runtime(...)`
- `train/posttrain.py`
  - 改为调用共享 helper
- `train/training_MPL.py`
  - 改为调用共享 helper
- `train/posttrain_common.py`
  - 本轮允许保留兼容 re-export；后续再决定是否继续瘦身或废弃

明确暂缓：

- `TrainerRuntimeConfig`
- `_resolve_trainer_runtime_config(...)`
- `_apply_trainer_runtime_config(...)`
- `Trainer` 内部 rollout / pose_history / angvel runtime 逻辑

---

## 1. 为什么这一轮先这样切

### 1.1 当前已经存在两类“共享但未收敛”的逻辑

#### A. `spec load + norm merge`

当前状态：

- `train/posttrain_common.py` 已有 `_load_json_object(...)`、`merge_norm_spec(...)`
- `train/training_MPL.py` 仍然保留 inline `_load_json_spec(...)` + pretrain-template merge

这两处是同一语义的两套实现。

当前已知 drift：

- `training_MPL` inline merge 的 key list 比 `NORM_SPEC_RUNTIME_PRETRAIN_KEYS` 少一个 `pose_hist_dim`
- 因此两条路径虽然接近，但不完全等价

这属于典型的“适合立即 dedup”的低风险目标。

#### B. `contact-pretrain hydration`

当前状态：

- `posttrain` 路径维护：
  - `posttrain_contacts_pretrain_clamp`
  - `posttrain_contacts_pretrain_affine_stats`
  - `posttrain_contacts_pretrain_affine`
- `training_MPL` 路径维护：
  - `trainbase_contacts_pretrain_clamp`
  - `trainbase_contacts_pretrain_affine_stats`
  - `trainbase_contacts_pretrain_affine`

两边本质上都在做同一个三元组解析：

- clamp
- affine_stats 原始 spec
- affine_stats 解析结果

注意：`posttrain` 侧当前 raw 字段名是 `posttrain_contacts_pretrain_affine_stats_spec`，`training_MPL` 侧当前 raw 字段名是 `trainbase_contacts_pretrain_affine_stats`。共享 helper 统一命名为 `affine_stats`，含义是 parse 前的原始 spec；调用方继续映射到各自字段名。

唯一需要保留的差异：

- `training_MPL` 在解析失败时会给 warn
- `posttrain` 当前更偏静默

统一时应保留 **warn 行为能力**，但不强制统一字段前缀。

### 1.2 为什么暂缓 `TrainerRuntimeConfig`

这一块不是 simple dedup，而是结构性重构：

- `training_MPL` 走 dataclass + apply 路径
- `posttrain` 走 `_build_model_and_trainer(...)` 中逐项 `setattr` 路径

如果本轮硬做，会变成“两边一起 churn”：

1. 先改 shared helper
2. 再改 posttrain trainer 构造
3. 再改 training_MPL runtime apply

这一步的回归风险明显高于前两类 shared helper dedup，因此本轮不做。

---

## 2. 本轮目标状态

本轮完成后，应满足：

1. `spec load + norm merge` 只有一套共享实现。  
2. `contact-pretrain hydration` 只有一套共享实现。  
3. `posttrain` 与 `training_MPL` 继续保留各自字段前缀与 runtime wiring。  
4. `training_MPL` 不再保留 inline norm merge key list。  
5. `pose_hist_dim` merge drift 被修正。  
6. `TrainerRuntimeConfig` 路径保持不变。  
7. `rollout` / `loss` / `Trainer` 核心逻辑不受影响。  

---

## 3. 本轮目标模块边界

### 3.1 `train/configuration/norm_spec.py`

这是本轮新增/收敛中心。

职责：

- 处理 JSON spec 读取与 merge
- 处理 pretrain-template runtime key 注入
- 处理 contact-pretrain affine/clamp runtime normalization
- 暴露两条训练入口可直接复用的 helper

本轮建议 API：

```python
from typing import Literal, overload

NORM_SPEC_RUNTIME_PRETRAIN_KEYS: tuple[str, ...]

def parse_pretrain_contact_affine_spec(spec: Any) -> Optional[Dict[str, Any]]: ...

@dataclass(frozen=True)
class ContactPretrainRuntime:
    clamp: float
    # Raw pre-parse spec. Maps to posttrain_contacts_pretrain_affine_stats_spec
    # or trainbase_contacts_pretrain_affine_stats at each caller.
    affine_stats: Optional[str]
    affine: Optional[Dict[str, Any]]

def resolve_contact_pretrain_runtime(
    *,
    clamp_raw: Any,
    affine_stats_raw: Any,
    warn: bool = False,
    warn_prefix: str = "",
) -> ContactPretrainRuntime: ...

@overload
def merge_norm_spec(
    bundle_path: Path,
    pretrain_path: Optional[Path],
    *,
    pretrain_keys: Optional[tuple[str, ...]] = None,
    strict: Literal[True] = True,
    warn: bool = False,
    warn_prefix: str = "",
) -> Dict[str, Any]: ...

@overload
def merge_norm_spec(
    bundle_path: Path,
    pretrain_path: Optional[Path],
    *,
    pretrain_keys: Optional[tuple[str, ...]] = None,
    strict: Literal[False],
    warn: bool = False,
    warn_prefix: str = "",
) -> Optional[Dict[str, Any]]: ...

def merge_norm_spec(
    bundle_path: Path,
    pretrain_path: Optional[Path],
    *,
    pretrain_keys: Optional[tuple[str, ...]] = None,
    strict: bool = True,
    warn: bool = False,
    warn_prefix: str = "",
) -> Optional[Dict[str, Any]]: ...
```

设计原则：

- `strict=True`：服务 `posttrain`
- `strict=False`：服务 `training_MPL`
- `strict=True` 的 overload 返回 `Dict[str, Any]`；`strict=False` 的 overload 返回 `Optional[Dict[str, Any]]`
- `warn=False` 表示静默；`warn=True` 表示打印 warning；`warn_prefix` 只控制 warn 文案前缀，不改变 merge / parse 语义
- 不在 helper 中硬编码 `posttrain_` / `trainbase_` 前缀

### 3.2 `train/posttrain.py`

本轮只改调用点，不改总体结构。

收敛目标：

- `_build_dataset_and_loader(...)` 改用共享 `merge_norm_spec(...)`
- `_build_model_and_trainer(...)` 中的 contact-pretrain hydration 改用 `resolve_contact_pretrain_runtime(...)`

仍然保留：

- `PostTrainConfig`
- trainer 属性命名 `posttrain_contacts_pretrain_*`
- 当前逐项 `setattr` 风格

### 3.3 `train/training_MPL.py`

本轮只改两块：

1. 训练入口里的 inline spec load + merge  
2. `_resolve_trainer_runtime_config(...)` 里的 contact-pretrain 三元组解析  

仍然保留：

- `TrainerRuntimeConfig`
- `_resolve_trainer_runtime_config(...)`
- `_apply_trainer_runtime_config(...)`
- trainer 属性命名 `trainbase_contacts_pretrain_*`

### 3.4 `train/posttrain_common.py`

本轮不要求彻底删除。

本轮明确终态：

- 迁入 `train/configuration/norm_spec.py`，但在 `posttrain_common.py` 保留 re-export shim：
  - `NORM_SPEC_RUNTIME_PRETRAIN_KEYS`
  - `merge_norm_spec`
  - `_merge_norm_spec`
  - `_load_json_object`
  - `_parse_pretrain_contact_affine_spec`
- 本轮继续留在 `posttrain_common.py`：
  - `_freeze_all`
  - `_enable_modules`
  - `_unfreeze_direct_pose`
  - `_select_trainable_params`
  - `_unfreeze_for_train_mode`

原则：

- 不再继续向 `posttrain_common.py` 添加新的 config parser 重实现
- 新 shared config/runtime helper 统一进入 `train/configuration/norm_spec.py`
- Commit 4 只做 shim / import 收口，不在本轮删除 validate/tools 仍可能依赖的旧符号

---

## 4. 分阶段执行

### Phase A — 引入共享 `norm_spec` 模块

**要做什么**

- [x] A1. 新建 `train/configuration/norm_spec.py`
- [x] A2. 迁入 `NORM_SPEC_RUNTIME_PRETRAIN_KEYS`
- [x] A3. 迁入并重命名 `parse_pretrain_contact_affine_spec(...)`
- [x] A4. 迁入/改造 `merge_norm_spec(...)`
- [x] A5. 新增 `resolve_contact_pretrain_runtime(...)`

**实现要求**

- `merge_norm_spec(..., strict=True)` 保持 `posttrain` 当前 hard-fail 行为
- `merge_norm_spec(..., strict=False)` 支持 `training_MPL` 当前 soft-load 行为
- shared merge key 默认使用 `NORM_SPEC_RUNTIME_PRETRAIN_KEYS`
- 必须覆盖 `pose_hist_dim`
- `warn=False` 为静默；`warn=True` 才打印 warning；`warn_prefix` 允许为空字符串

**验收**

- [x] `train/configuration/norm_spec.py` 可以独立被 `posttrain` / `training_MPL` import
- [x] key merge 列表无遗漏 `pose_hist_dim`
- [x] 两种 strict 模式都具备明确行为
- [ ] 对同一组 `bundle_json` / `pretrain_template`，新 `merge_norm_spec(strict=True)` 与旧 `_merge_norm_spec(...)` 逐 key diff 为空
- [x] resolver 对 invalid affine spec 在 `warn=False` 时静默、`warn=True` 时发 warn

### Phase B — 接入 `posttrain.py`

**要做什么**

- [x] B1. `train/posttrain.py` 改从 `train/configuration/norm_spec.py` import
- [x] B2. `_build_dataset_and_loader(...)` 使用共享 `merge_norm_spec(...)`
- [x] B3. `_build_model_and_trainer(...)` 使用 `resolve_contact_pretrain_runtime(...)`

**保留约束**

- 不统一字段前缀
- 不改 trainer attr 名称
- 不改 `_build_model_and_trainer(...)` 其余 runtime wiring

**验收**

- [x] `posttrain` 不再手写 clamp/affine parse 逻辑
- [x] `posttrain` trainer attr 仍然是 `posttrain_contacts_pretrain_*`
- [x] 最小 smoke 到 `_build_dataset_and_loader(...)` 返回：`norm_spec` 非空，且包含 pretrain runtime keys（含 `pose_hist_dim`，若 pretrain template 提供）
- [x] 最小 smoke 到 `_build_model_and_trainer(...)` 返回：`trainer.posttrain_contacts_pretrain_clamp`、`trainer.posttrain_contacts_pretrain_affine_stats_spec`、`trainer.posttrain_contacts_pretrain_affine` 三个 attr 均存在
- [x] `python3 -m py_compile train/posttrain.py` 通过

### Phase C — 接入 `training_MPL.py`

**要做什么**

- [x] C1. 删除训练入口中的 inline `_load_json_spec(...) + merge` 重实现
- [x] C2. 改用共享 `merge_norm_spec(..., strict=False)`
- [x] C3. `_resolve_trainer_runtime_config(...)` 改用 `resolve_contact_pretrain_runtime(...)`

**保留约束**

- 保留 `TrainerRuntimeConfig`
- 保留 `trainbase_contacts_pretrain_*` 字段
- 保留 warn 语义

**验收**

- [x] `training_MPL` 不再保留独立的 runtime pretrain merge key list
- [x] `training_MPL` 仍在解析失败时发 warn
- [x] 最小 smoke 到 `_resolve_trainer_runtime_config(...)` 返回：`TrainerRuntimeConfig.trainbase_contacts_pretrain_clamp`、`trainbase_contacts_pretrain_affine_stats`、`trainbase_contacts_pretrain_affine` 三个字段均存在
- [ ] `merge_norm_spec(..., strict=False, warn=True)` 在缺失可选 pretrain template 时保持 soft path；缺失必需 norm template 时仍按训练入口逻辑 fail
- [x] `python3 -m py_compile train/training_MPL.py` 通过

### 4.1 已验证 fixture（2026-04-18）

- basetrain entry-build smoke
  - `config/exp_phase_mpl.clean.json`
  - clip: `raw_data/processed_data/Walk_F.npz`
  - encoder: `models/motion_encoder_equiv_stageA.pt`
  - coverage: `_build_train_components(...)` → `_build_train_loaders(...)` → `_build_train_model(...)` → `_prepare_train_model_runtime(...)` → `_build_train_loss_and_trainer(...)` → `build_and_attach_dataset_runtime(...)` → `_resolve_trainer_runtime_config(...)`
  - Step 2 shell smoke:
    - `tools/run_training_mpl_entry_shell_smoke.py`
    - `debug_output/_training_mpl_entry_shell_smokes_20260418/training_mpl_entry_shell_smoke_summary.json`
- posttrain entry-build smoke
  - `config/posttrain_direct_pose_walkf.json`
  - ckpt override: `models/MLPL2_DirectBranch_v1_20260317/exp_phase_DirectBranch_v1_d1_20260317/ckpt_best_free_exp_phase_DirectBranch_v1_d1_20260317.pth`
  - clip: `raw_data/processed_data/Walk_F.npz`
  - encoder: `models/motion_encoder_equiv_stageA.pt`
  - coverage: `_cfg_from_payload(...)` → `_build_dataset_and_loader(...)` → `_build_posttrain_model_from_ckpt(...)` → `_build_model_and_trainer(...)`
- smoke artifacts summary:
  - `debug_output/_runtime_attach_entry_smokes_20260418/entry_build_smoke_summary.json`

### Phase D — 兼容层与后续观察

说明：当前工作树已无 `train/posttrain_common.py`；freeze/train-mode helper 已位于 `train/runtime/freeze.py`。
因此 D1-D2 在现态下属于历史计划尾项，不再是本轮 blocker。

**要做什么**

- [ ] D1. 将 `train/posttrain_common.py` 中已被 `norm_spec.py` 接管的 parser/config helper 改成 re-export shim
- [ ] D2. 保留旧符号，不删除 validate/tools 仍可能 import 的名称
- [ ] D3. 记录后续是否继续推进 `TrainerRuntimeConfig` 统一

**本轮不要求**

- 不要求删除所有兼容 import
- 不要求同步拆 `train/freeze.py`
- 不要求把 trainer runtime dataclass 抬到 common
- 不要求迁移 validate/tools 到 `train.configuration.norm_spec`

---

## 5. 影响面与调用点

### 5.1 直接受影响文件

- `train/configuration/norm_spec.py`（新增）
- `train/posttrain.py`
- `train/training_MPL.py`
- `train/posttrain_common.py`（可选兼容清理）

### 5.2 可能间接受影响文件

- `train/validate/run_freerun_cycles.py`
- `train/validate/run_teacher_rollout.py`
- 若仍从 `posttrain_common.py` import merge helper 的 `tools/`

处理原则：

- 本轮优先通过 re-export 或 import 改线维持兼容
- 不把 validate/tools 的适配强行并入本轮大改

---

## 6. 风险与回退

| 风险 | 触发信号 | 回退策略 |
|---|---|---|
| `training_MPL` soft-load 行为被误改成 hard-fail | 缺失 pretrain/norm 文件时直接 `SystemExit` 或 `RuntimeError` | 保留 `strict=False` 路径，恢复 warn + `None` |
| `posttrain` 行为被误改成 silent fallback | bundle/pretrain 模板损坏时不再 fail-fast | `posttrain` 强制使用 `strict=True` |
| contact-pretrain warn 行为丢失 | `training_MPL` 解析失败时无日志 | 在 resolver 保留 `warn=True` + `warn_prefix` |
| 字段前缀被错误统一 | `posttrain_*` 或 `trainbase_*` attr 丢失 | helper 只返回 normalized payload，不写 attr 名 |
| validate/tools import 断裂 | `posttrain_common` 删除过早 | 先保留兼容 re-export |

---

## 7. 建议提交拆分

1. Commit 1: 新增 `train/configuration/norm_spec.py`，迁入 shared helper  
2. Commit 2: `posttrain.py` 改接共享 helper  
3. Commit 3: `training_MPL.py` 改接共享 helper  
4. Commit 4: `posttrain_common.py` 加 re-export shim、不删旧符号；validate/tools 迁移留到下一轮  

---

## 8. 下一轮再决定的事情

以下事项明确推迟到下一轮：

- 是否把 `TrainerRuntimeConfig` 抬到 shared/common
- 是否把 `posttrain` 的 trainer hydration 也重构成 dataclass + apply 路径
- 是否把 freeze helper 进一步迁到 `train/freeze.py`
- 是否正式废弃 `train/posttrain_common.py`

下一轮的前提是：

1. 本轮 `(spec merge + contact-pretrain hydration)` 收敛完成  
2. `posttrain` / `training_MPL` 编译与最小训练路径无回归  
3. validate/tools 的 import 兼容层已稳定  

---

## 9. 执行原则

- 优先消除“两处共享逻辑 + 两处 drift”的问题，不追求一步到位模块化
- 本轮新增的 shared 模块应当是**稍微杂糅但语义一致**，而不是过细拆分
- 所有 helper 以“返回 normalized payload，由调用方自行命名 attr”为原则
- 任何跨 `posttrain` / `training_MPL` 的统一，如果需要两边都大改 wiring，则自动延期到下一轮
