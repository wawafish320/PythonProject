# [2026-04-21] `train/models.py` 单文件重构收敛计划（v2）

Date: 2026-04-21
Status: Draft v2（v1 于同日产出，v2 基于 review 修订）
Scope: `train/models.py`（允许同步修改最小必要测试与文档；当前阶段**不要求先拆新文件**）
Goal: 在**不改变训练/推理语义、checkpoint contract、默认超参行为**前提下，先把 `train/models.py` 做成“边界清楚、异常可见、后续可机械拆分”的单文件模块化结构。
Non-goal:
- 不改核心算法 / 数学定义、不恢复 legacy compat、不调整 default init / optimizer 行为、不顺手清 unrelated 模块。
- 不迁入 `train/geometry.py`：rot6d 子系统是“对归一化后的 rot6d slice 做 denorm + reproject + loss”的上层 loss 逻辑，已经 *消费* `train/geometry.py` 的 pure primitives，不应污染 geometry 层。参见 §2.4。
- 不把 `_warn_once` 签名重写纳入 Phase B（影响面过大，挪到 Phase E）。参见 §4 Phase B。

---

## 0) 必须遵守的纪律（读任何一条 Phase 前先读这个）

本 roadmap 的所有清理动作（尤其 Phase A fail-fast、Phase C contract 收口、Phase E 跨文件迁移）**强制遵守** [`docs/removal_policy.md`](../removal_policy.md)。

以下条款在本次重构中 **不可协商**：

- **§2 Step 2**：每一处被删 / 收窄的分支，必须在第一次被读入的入口显式 `raise`，错误信息必须包含「字段名 + 移除提交 + 迁移路径（或 "no replacement"）」。
- **§4 反模式清单**：不得新增 silent fallback、`.get(.., .get(.., default))` 兜底、`warnings.warn` 替代 raise、ckpt key silent rename、`# kept for compat` 注释死代码、`try/except Exception: fallback` 吞错。收窄 broad `except Exception` 时，**不得**把它改写成"warn + 继续"——要么 raise，要么窄异常 + 明确 fallback 值。
- **§5 Ckpt compat 边界**：本次重构涉及任何 direct-pose init / split state / weight cache 改动时，compat 层**只允许** schema reshape，不允许 semantic mapping。
- **§6 LLM 协作约束**：每次 commit 前必须 grep diff，确认未新增 §6 的 4 条反模式正则。
- **§7 评审 checklist**：每个 PR 都必须跑一遍 §7 的 7 项自检。
- **每轮强制回填**：本轮如果有任何实际执行结果，必须同步回填到对应的 `docs/changes` 文档中，至少包括日期、本轮目标、实际完成项、修改文件列表、运行过的命令、验证结果、阻塞项 / 风险、下一轮建议动作；禁止只改代码不回填文档，禁止只更新计划不写执行结果。

**范本**（照抄，不要发明新风格）：
- `train/training_MPL.py:_assert_no_legacy_loss_keys_in_schedule`
- `train/training_MPL.py:_assert_no_removed_trainbase_stage_keys`
- `train/posttrain.py:_cfg_reject_retired_direct_pose_highorder`

命名继续沿用 `_assert_no_<scope>_<what>_keys` / `_<scope>_reject_<what>`。

---

## 1) 一页版结论（先看这个）

**当前最值得先做的三件事（P0）**：

1. **先把 loss / direct-pose 主路径里的 broad `except Exception` 按 removal policy §4 收窄成 fail-fast 或窄 fallback**。
2. **先在当前文件内提 pure helpers / 建 cluster（skeleton / rot6d / direct-pose-loss / applicators），不急着跨文件拆分**。
3. **先把 direct-pose / skeleton / weight-cache contract 显式化、类型化，再做大函数拆分。**

**执行顺序**：fail-fast 预清障 → 单文件 helper 化 / cluster 化 → direct-pose contract 收口（model 侧 + loss 侧）→ 拆 `forward` / `MotionJointLoss` → 最后才考虑跨文件迁移。

**验收原则**：每一步都必须保持 `py_compile`、最小单测通过；凡触及 direct-pose 语义路径，必须跑 `stage6 deterministic smoke` + **forward output snapshot 对照**。Phase 结束时对照的"硬指标"见 §5。

---

## 2) 当前问题（按优先级）

### 2.1 P0（必须先解决）

| 类别 | 位置 | 当前现象 | 风险 |
|---|---|---|---|
| 巨类 / 职责过载 | `train/models.py:405` | `EventMotionModel` 类跨度约 3368 行，构造、前向、runtime control、ablation、direct-pose、event-clock 全混在一起 | 后续任何局部改动都容易误伤主路径 |
| 巨函数 | `train/models.py:2341` | `EventMotionModel.forward(...)` 约 1432 行，既做输入整形，也做 contact-plan、direct-pose、leg routing、runtime ablation、输出拼装 | 无法局部证明正确性；回归定位成本极高 |
| 构造耦合 | `train/models.py:410` | `EventMotionModel.__init__(...)` 约 539 行，兼做 config 归一化、合法化、state 初始化、module build | "参数解析错误"和"结构错误"混在一起 |
| direct-pose 热点（model 侧） | `train/models.py:1222` | `_build_direct_pose_modules()` 同时负责 shape 推导、layout contract、branch build、显式 init 语义、deterministic init 策略 | 是当前最容易继续长大的热点 |
| direct-pose 热点（loss 侧） | `train/models.py:4870`–`5215` | 初始热点为 `_direct_pose_default_stats` / `_direct_pose_extra_defaults` / `_prepare_direct_pose_pair` / `_compute_direct_pose_group_norm_payload` / `_compute_direct_pose_group_base_payload` / `_compute_direct_pose_group_norm_shared` / `_compute_direct_pose_payload` / `_apply_direct_pose_component` 这一簇；后续已在 `train/losses.py` 收敛为更薄的 `group_base_payload -> group_norm_shared -> group_norm_result -> direct_pose_payload` 主链 | roadmap v1 只覆盖了 model 侧，loss 侧 direct-pose 语义同样复杂且耦合 EMA / weight cache |
| 静默异常 | `train/models.py:1533`, `train/models.py:2406`, `train/models.py:3653` 等 | 文件内共有 `72` 个 broad `Exception` handler（`71` 个精确 `except Exception:` + `1` 个 `except Exception as exc`）；`forward` 区 `32`，`MotionJointLoss` 区 `12` | 违反 removal_policy §4；结构错误被吞掉，后续重构只能"盲猜" |
| silent contract failure | `train/models.py:1591` | `_direct_pose_split_state()` 失败时直接返回 `None`，调用方再间接判断 | contract 断裂时无明确声音，违反 removal_policy §2 Step 2 |

### 2.2 P1（建议随后解决）

| 类别 | 位置 | 当前现象 | 风险 |
|---|---|---|---|
| buried pure helpers | `train/models.py:4055`, `:5451`, `:5464`, `:5480` | `_masked_group_mean`、`_masked_group_weighted_mean`、`_stats_float`、`_stats_float_or`、`_ensure_temporal_axis`、`_setdefault_stats` 埋在类体内部 | helper 无法形成稳定边界，类体持续膨胀 |
| stateful skeleton / cache helpers | `train/models.py:4084`, `:4095`, `:4173`, `:4205`, `:4261`, `:4286` | skeleton 元数据、权重缓存、tail-risk 候选与局部 stats 逻辑混在 `MotionJointLoss` 内 | loss 逻辑与 skeleton state 高耦合，不利于后续独立测试 |
| rot6d 子系统 cluster 缺失 | `train/models.py:4474`–`4642` | `_maybe_get_rot6d` / `_denorm_rot6d_flat` / `_extract_rot6d_flat` / `_extract_rot6d_mats` / `compute_rot6d_geo_loss` / `compute_rot6d_ortho_loss` / `_rot6d_matrices` / `compute_rot6d_log_loss` 共 8 方法散落 | 无清晰边界；v1 漏列 |
| applicators 簇无边界 | `train/models.py:4678`–`5407` | `_apply_rot_ortho_component` / `_apply_rot_local_tail_component` / `_apply_rot_local_component` / `_apply_root_velocity_components` / `_apply_motion_components` / `_apply_direct_pose_component` / `_apply_contact_plan_component` / `_apply_event_clock_components` / `_apply_contact_meas_component` / `_apply_omega_l2_component` / `_apply_aux_components` 共 11 方法，签名一致、行为独立 | 天然 cluster 但没显式划区 |
| direct-pose runtime / ablation 混入核心模型 | `train/models.py:2079`, `:2210`, `:2230` | eval/runtime override 与训练主语义共存 | runtime policy 和 core model 边界不清楚 |
| loss 组件过载 | `train/models.py:3774` | `MotionJointLoss` 类跨度约 1738 行，包含 config parse、skeleton state、component apply、stats 聚合 | 后续拆 loss 时容易引入行为漂移 |

### 2.3 P2（清理与治理）

| 类别 | 位置 | 当前现象 | 风险 |
|---|---|---|---|
| eventual file split blocked by semantics | `train/models.py` | 当前如果直接拆文件，会把"结构变化"和"语义变化"绑在一起 | 回归面过大，难以做 mechanical move |
| helper purity 混淆 | `train/models.py:3962`, `:4153`, `:4168` | `_warn_once`、`_parent_relative_matrices`、`_root_relative` 看似小函数，但仍隐式读写 state/cache | 若误当 pure helper 提前抽走，会形成新债 |
| validation coverage gap | `tests/train/test_event_motion_model_refactor_phase_d.py:178` | 目前 direct-pose 回归已覆盖一部分，但 loss / skeleton / fail-fast 路径仍缺最小化回归；forward output snapshot 完全缺失 | 重构后可能"结构更漂亮，但失败更安静" |

### 2.4 rot6d 子系统归属决策（v2 新增）

8 个方法依赖的 `self.*` 状态：

| 依赖 | 谁用 |
|---|---|
| `group_slices`（slice 契约） | `_denorm_rot6d_flat`, `compute_rot6d_geo_loss` |
| `mu_y`, `std_y`（normalization stats） | `_denorm_rot6d_flat` |
| `_warned_bad_rot6d*`, `_train_denorm_hit`（warn-once flags） | `_extract_rot6d_flat`, `compute_rot6d_geo*`, `compute_rot6d_ortho*` |
| `_joint_weight_vector(...)`（skeleton weights） | `compute_rot6d_geo_loss` |

**结论**：这批方法是"归一化 rot6d slice 上的 loss 层逻辑"，**不是**纯几何；它们 *消费* `train/geometry.py` 的 `rot6d_to_matrix` / `geodesic_R` / `angvel_vec_from_R_seq` / `reproject_rot6d`，不应污染 geometry 层。

- **现在（Phase B.B5）**：in-file 分区 + 分区注释 `# === future: train/loss_rot6d.py ===`。
- **将来（Phase E.E3）**：迁到 `train/loss_rot6d.py`（新建），以 `_Rot6DLossHelpers` 持有 `(group_slices, mu_y, std_y, warn_registry)`。`compute_rot6d_geo_loss` 对 `_joint_weight_vector` 的调用改成外部传入 `weights` 参数，切开 skeleton 耦合。

---

## 3) 目标状态（Done 后应满足）

1. `train/models.py` 先在**单文件内部**形成清晰边界：`EventMotionModel build/runtime`、`MotionJointLoss core`、`skeleton/weight cache helper cluster`、`rot6d helper cluster`、`direct-pose loss cluster`、`component applicators cluster`、`loss tracker cluster`。
2. direct-pose（model 侧 + loss 侧）/ loss 主路径里的 broad `except Exception` 不再吞结构错误；必须是 **fail-fast** 或 **窄异常 + 明确 fallback**，且通过 removal_policy §4 grep 自检。
3. pure helpers 变成模块级 free functions；stateful helpers 变成显式 helper cluster / dataclass / state object，而不是"伪 pure"。
4. `_direct_pose_split_state()` 这类 contract 改成**显式类型 + 明确失败路径**，不再靠 `None` 静默退化（removal_policy §2 Step 2）。
5. 只有当单文件边界稳定后，跨文件拆分才变成 mechanical move，而不是高风险语义重写。
6. **每个 Phase 结束都留下 forward output snapshot 与 state_dict key-set / checksum**，用于下一 Phase 对照。

---

## 4) 分阶段执行（每阶段都有输入/输出/验收）

### Phase A — fail-fast 预清障与语义冻结（不先拆文件）

**要做什么**
- [x] A1. 记录 `train/models.py` 当前结构基线：LOC=`5511`，`EventMotionModel.forward`≈`1432` 行，broad `Exception` handler=`72`（`71` 个精确 `except Exception:` + `1` 个 `except Exception as exc`；见 §9 与 §12）。
- [x] A2. **产出 fail-fast inventory**：扫描全部 72 处 broad `Exception` handler，分三类登记到独立文档 `docs/changes/2026-04-21_train_models_fail_fast_inventory.md`：
  - 类别 A（**立即 fail-fast**）：loss / direct-pose / forward 主路径上吞结构错误的（含 `:1533` / `:2406` / `:3653` / `MotionJointLoss.__init__` 中 `:3850` / `:3868` / `:3945` 等）
  - 类别 B（**收窄为具体异常 + 明确 fallback 值**）：外围 diagnostic / stats 计算路径
  - 类别 C（**本轮不动**）：IO / serialization / 第三方接口边界
  - 每一条必须标：行号、上下文（4 行代码）、归类理由、目标处理方式。
- [ ] A3. 按 A2 类别 A 的条目逐个收窄（改 raise 或窄异常）；类别 B 条目改异常类型并给明确默认值；类别 C 不动。所有 raise 的错误信息按 removal_policy §2 Step 2 模板（字段名 + 移除提交 + 迁移路径）。
  - 2026-04-21 update: A01–A29（除 A30）已处理（direct-pose leg residual outer branch + gate / scale / sign-gate inner fallback + phase/side-cue contract + side-routed phase view/side-embed contract + direct-pose feature/time cluster + contact-plan init/append/stack/time-bias cluster + auxiliary/lambda-fusion cluster）；当前 broad `Exception` handler=`43`，剩余类别 A 原始条目 `22` 个待处理。
  - 2026-04-21 update: Batch 3（A31–A40）已处理（build-time layout / deterministic init / routing metadata cluster）；当前 broad `Exception` handler=`33`，剩余类别 A 原始条目 `12` 个待处理（A30 + A41–A51）。
  - 2026-04-21 update: Batch 4（A41–A51）已处理（MotionJointLoss config / skeleton / payload cluster）；当前 broad `Exception` handler=`22`，剩余类别 A 原始条目 `1` 个待处理（A30）。
  - 2026-04-21 update: Batch 5（A30）已处理；当前 broad `Exception` handler=`21`，`except Exception as exc`=`0`，Phase A Category A 原始条目剩余 `0`。剩余 broad handler 均为 Category B / C。
  - 2026-04-22 update: Phase A Category B constructor cluster B01–B11 已处理；`EventMotionModel.__init__` 中 11 个 broad `except Exception:` 已全部替换为 typed normalization / explicit default；当前 broad `Exception` handler=`10`，剩余 broad handler 为 Category B `8`（B12–B19）+ Category C `2`（C01–C02）。
  - 2026-04-23 update: Phase A Category B runtime/build helper cluster B12–B18 已处理；当前 broad `Exception` handler=`3`，剩余 broad handler 为 Category B `1`（B19）+ Category C `2`（C01–C02）。
  - 2026-04-23 update: Phase A Category B B19 已处理；当前 broad `Exception` handler=`2`，剩余 broad handler 均为 Category C `2`（C01–C02）。
  - 2026-04-23 update: Phase C.loss.3 与 Category C（C01/C02）已完成；当前 broad `Exception` handler=`0`，精确 `except Exception:`=`0`，`except Exception as exc`=`0`。
- [ ] A4. 为 contract failure 建最小可观测回归：覆盖 `_direct_pose_split_state()` strict failure、loss skeleton 输入异常、forward 输入 shape 异常；新增位置 `tests/train/test_train_models_failfast.py`。
  - 2026-04-21 update: 已创建 `tests/train/test_train_models_failfast.py`，当前覆盖 `30` 个 fail-fast 场景（direct-pose leg residual head failure + gate / scale / sign-gate failure + phase/side-cue contract + side-routed phase/side-embed contract + direct-pose feature/time cluster + contact-plan init/append/stack/time-bias cluster + auxiliary/lambda-fusion cluster）；strict split / skeleton / generic forward shape 覆盖仍待补。
  - 2026-04-21 update: Batch 3 新增 `10` 个 constructor/build-time fail-fast 回归；当前覆盖 `40` 个 fail-fast 场景，但 `_direct_pose_split_state()` strict failure、MotionJointLoss skeleton / payload fail-fast 与 generic forward shape 覆盖仍待补。
  - 2026-04-21 update: Batch 4 新增 `11` 个 MotionJointLoss fail-fast 回归；当前覆盖 `51` 个 fail-fast 场景，但 `_direct_pose_split_state()` strict failure 与 generic forward shape 覆盖仍待补。
  - 2026-04-21 update: Batch 5 未新增测试；现有 `51` 个 fail-fast 场景继续覆盖 A30 contextual wrap，`tests.train.test_train_models_failfast` 总用例数为 `51`，联合验证总用例数为 `57`。
  - 2026-04-22 update: A4 覆盖补齐后继续扩到 constructor normalization cluster；`tests/train/test_train_models_failfast.py` 新增 `15` 个 B01–B11 constructor regression tests，当前 fail-fast regression 总数为 `70`，与 `tests.train.test_event_motion_model_refactor_phase_d` 联合验证总用例数为 `76`。
  - 2026-04-23 update: B12–B18 新增 `12` 个 runtime/build helper regression tests，当前 fail-fast regression 总数为 `82`，与 `tests.train.test_event_motion_model_refactor_phase_d` 联合验证总用例数为 `88`。
  - 2026-04-23 update: B19 新增 `5` 个 attention regularization regression tests，当前 fail-fast regression 总数为 `87`，与 `tests.train.test_event_motion_model_refactor_phase_d` 联合验证总用例数为 `93`。

**产出物**
- [ ] 本 roadmap 文档（当前文件）
- [x] `docs/changes/2026-04-21_train_models_fail_fast_inventory.md`（72 处三分类表：A=`51`，B=`19`，C=`2`）
- [x] `tests/train/test_train_models_failfast.py`（已创建；A4 覆盖已补齐，且已扩到 Phase A Category B B01–B19）
- [ ] 基线 forward output snapshot（固定 seed / batch，dump `pred_motion` / `losses` scalar checksum）→ `tests/train/snapshots/baseline_20260421.json`

**验收**
- [ ] `python3 -m py_compile train/models.py tests/train/test_event_motion_model_refactor_phase_d.py tests/train/test_train_models_failfast.py`
- [ ] `python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d tests.train.test_train_models_failfast`
- [ ] 类别 A 条目清零；broad `except Exception` 总数 ≤ 40（从 71 降到类别 B + C）
- [ ] `git diff` grep 未命中 removal_policy §6 的 4 条反模式正则
- [ ] forward output snapshot 与基线 bitwise 相等（此 Phase 不应改变语义）

---

### Phase B — 单文件 helper 化 / cluster 化（低风险重构）

**要做什么**
- [x] B1. 把真正 pure 的小函数提成**模块级 free functions**：`_masked_group_mean`、`_masked_group_weighted_mean`、`_stats_float`、`_stats_float_or`、`_ensure_temporal_axis`、`_setdefault_stats`。（2026-04-23 已完成；类上保留同名 `staticmethod` alias 以维持仓内既有私有调用点）
- [ ] B2. **`_warn_once` 本 Phase 不动**（调用面广，挪到 Phase E 统一处理）。`_parent_relative_matrices` / `_root_relative` 同样暂不 free function 化（仍依赖 `parents` / `root_idx` / `_parents_tensor`），并入 B3 skeleton cluster。
- [x] B3. 在**当前文件内**建立 **skeleton / weight-cache cluster**（分区注释 `# === future: train/loss/skeleton_weights.py ===`）：`set_skeleton`、`_invalidate_weight_cache`、`_resolve_limb_masks`、`_resolve_direct_group_masks`、`_resolve_named_joint_indices`、`_joint_weight_vector`、`_compute_unified_weights_cpu`、`_collect_limb_local_stats`、`_rot_local_tail_scores`、`_rot_local_tail_candidates`、`_parent_relative_matrices`、`_root_relative`。（2026-04-23 已完成：cluster 分区注释 + 顺序整理 + direct masks / limb stats / weight cache / tail candidates / relative rotation regression 锁定）
- [x] B4. 在**当前文件内**建立 **component applicators cluster**（分区注释 `# === future: train/loss/components.py ===`）：`_apply_rot_ortho_component` / `_apply_rot_local_tail_component` / `_apply_rot_local_component` / `_apply_root_velocity_components` / `_apply_motion_components` / `_apply_contact_plan_component` / `_apply_event_clock_components` / `_apply_contact_meas_component` / `_apply_omega_l2_component` / `_apply_aux_components`。**本 Phase 只做 in-file 分区，不抽 free function**（依赖 `self._submit_component_loss` / `self._accumulate_loss_contrib`，过早抽离会暴露一堆参数）。（2026-04-23 已完成：future 分区注释 + motion/direct-pose/aux 子区注释 + forward-level stats/key regression + direct-pose default stats regression）
- [x] B5. （v2 新增）在**当前文件内**建立 **rot6d helper cluster**（分区注释 `# === future: train/loss_rot6d.py ===`）：`_maybe_get_rot6d` → `_denorm_rot6d_flat` → `_extract_rot6d_flat` → `_extract_rot6d_mats` / `_rot6d_matrices` → `compute_rot6d_ortho_loss` → `compute_rot6d_geo_loss` → `compute_rot6d_log_loss`，按"从底层到外层"顺序排列。**本 Phase 不做**：`_Rot6DLossHelpers` 类化、`_joint_weight_vector` 外部化（留到 Phase E.E3）。（2026-04-23 已完成：future 分区注释 + helper/objective 子区注释 + extract/matrix/objective regression）

**验收**
- [ ] 类体长度显著下降，4 个 cluster（skeleton / rot6d / applicators / direct-pose-loss 预留）分区注释就位
- [ ] `MotionJointLoss` 内部不再同时承载"纯数值 helper"和"state/cache 管理"两类职责
- [ ] 对外行为、日志 key、checkpoint contract 不变
- [ ] `forward output snapshot` 与 Phase A 结束时 bitwise 相等
- [ ] `git diff` grep 未命中 removal_policy §6 的 4 条反模式正则

---

### Phase C — direct-pose contract 收口（中风险）

**C 本身拆成两条独立路径**，两边同名但语义独立，必须分开做：

#### Phase C.model — EventMotionModel 侧 direct-pose

- [ ] C.model.1 **typed contract / strict state**：将 `_direct_pose_split_state()` 从"返回 dict / `None`"改成显式结构（dataclass / NamedTuple）。提供 strict builder（训练主路径）与 optional probe builder（诊断路径），**禁止主路径 silent `None`**（removal_policy §2 Step 2）。
- [ ] C.model.2 **build vs init vs runtime 边界**：将 direct-pose 的 topology / index / branch spec 和 deterministic init policy 显式分层；保持当前 deterministic init 行为、canonical checkpoint contract、stage6 step0 对齐结论不变。
- [ ] C.model.3 **runtime hot path 下沉**：从 `EventMotionModel.forward(...)` 中下沉 direct-pose feature assemble、readout、leg routing、gate/ablation 逻辑，保持编排层只负责调度。

#### Phase C.loss — MotionJointLoss 侧 direct-pose（v2 新增）

- [x] C.loss.1 把 loss 侧 direct-pose cluster 收成一个 in-file cluster（分区注释 `# === future: train/loss/direct_pose.py ===`）。（2026-04-23 已完成：single-file cluster 注释边界 + 子区注释 + focused regression coverage；2026-04-25 已进一步删去 `_direct_pose_extra_defaults` 与 `_compute_direct_pose_group_norm_payload` 这类机械壳）
- [x] C.loss.2 把 payload 层（`_compute_direct_pose_payload` 的输入）类型化成 dataclass；EMA / weight cache 访问显式化。（2026-04-23 已完成：新增 direct-pose pair / payload / group-norm request-result dataclasses，外部 helper 返回契约保持不变）
- [x] C.loss.3 `_apply_direct_pose_component` 的 stats 写入 key 集合冻结为 contract（单独一张表），后续任何新增 key 必须走 contract bump。（2026-04-23 已完成：module-level contract key tuple + exact regression coverage + roadmap contract table）

| Contract Group | Keys |
|---|---|
| Direct core | `direct_pose_geo`, `direct_pose_geo_deg`, `direct_pose_objective`, `direct_pose_weighted` |
| Split / base | `direct_pose_split_active`, `direct_pose_arm_split_active`, `dir_base`, `dir_leg_base`, `dir_nonleg_base`, `dir_nonleg_effective_base`, `dir_arm_base`, `dir_else_base` |
| Balance / weights | `leg_over_nonleg`, `leg_over_nonleg_effective`, `arm_over_else`, `direct_pose_arm_else_balance_active`, `direct_pose_loss_arm_weight`, `direct_pose_loss_else_weight` |
| Group norm core | `dir_group_norm_used`, `dir_group_norm_leg_raw`, `dir_group_norm_nonleg_raw`, `dir_group_norm_leg_clamped`, `dir_group_norm_nonleg_clamped`, `dir_group_norm_leg`, `dir_group_norm_nonleg`, `dir_group_norm_leg_ema`, `dir_group_norm_nonleg_ema`, `dir_group_norm_leg_hit_min`, `dir_group_norm_leg_hit_max`, `dir_group_norm_nonleg_hit_min`, `dir_group_norm_nonleg_hit_max`, `dir_group_norm_leg_hit_any`, `dir_group_norm_nonleg_hit_any` |
| Group norm config | `dir_group_norm_w_leg`, `dir_group_norm_w_nonleg` |

**阶段验收**
- [ ] `train/models.py:1222` 与 `train/models.py:1808` 的职责不再交叉（model 侧）
- [ ] loss 侧 direct-pose cluster 分区就位，stats key contract 表产出
- [ ] **必跑** stage6 deterministic smoke（model 侧 C 改动时）
- [ ] **必跑** forward output snapshot + loss scalar snapshot（loss 侧 C 改动时）
- [ ] step0 state_dict **key-set bitwise 相等**，weight tensor checksum **漂移 < 1e-12**
- [ ] `git diff` grep 未命中 removal_policy §6 的 4 条反模式正则

---

### Phase D — `forward` / `MotionJointLoss` 大函数收口（中高风险）

**要做什么**
- [x] D1. 将 `EventMotionModel.forward(...)` 拆成"输入准备 → contact-plan/event-clock → direct-pose → finalize"四层编排。
- [x] D2. 将 `MotionJointLoss` 拆成"config/init / skeleton state / payload builders / component apply / stats finalize"五层；五层的物理区块在 Phase B / C 的 cluster 基础上**只做顺序整理**，不新增 helper。
- [x] D3. 让 applicators cluster（Phase B.B4 建立）外观不变；Phase D 只关心 forward 编排层如何调度，不再把 applicator 内部动摇一遍。

**验收**
- [x] `EventMotionModel.forward(...)` 编排壳已显式成层，且失效点更可见；当前 line inventory 保留到 Phase E 继续缩短
- [x] `MotionJointLoss` 可以在单文件内被阅读为"编排壳 + 组件 helper"，而不是一整片连续实现
- [x] broad `except Exception` 数进一步下降（当前为 `0`）
- [x] **必跑** stage6 deterministic smoke + forward output snapshot（Phase D 改了 forward 编排层）
- [x] `git diff` grep 未命中 removal_policy §6 的 4 条反模式正则

---

### Phase E — 清理与可选跨文件迁移（只在前面稳定后再做）

**要做什么（跨文件目标文件清单，v2 补全）**
- [ ] E1. `train/loss/skeleton_weights.py`（新建）← 迁移 Phase B.B3 的 skeleton / weight-cache cluster。
- [ ] E2. `train/models/direct_pose/*`（新建目录）← 迁移 Phase C.model 的 direct-pose build / init / runtime 子系统。
- [ ] E3. `train/loss_rot6d.py`（新建）← 迁移 Phase B.B5 的 rot6d cluster；同时做 `_Rot6DLossHelpers` 类化与 `_joint_weight_vector` 外部化。
- [ ] E4. `train/loss/direct_pose.py`（新建）← 迁移 Phase C.loss 的 loss 侧 direct-pose cluster。
- [ ] E5. `train/loss/components.py`（新建；或按组件拆 `train/loss/components/*.py`）← 迁移 Phase B.B4 的 applicators cluster。
- [ ] E6. `train/loss/tracker.py`（新建）← 迁移 `_init_loss_group_tracker` / `_accumulate_loss_contrib` / `_loss_group_stats` / `_submit_component_loss` / `_prepare_aux_supervision_pair`。
- [ ] E7. `_warn_once` 重构（v1 的 B2，本 v2 挪到这里）：改成显式 `warn_once(warned_set, key, msg, ...)` free function；扫描全部调用点批量改写。
- [ ] E8. 将剩余外围 broad exception 做系统性清理（类别 B / C 中的保留项），保留少数合理的窄 fallback。
- [ ] E9. 更新 `train/MODULE_BOUNDARIES.md`、相关 roadmap / report / test 文档，形成长期维护入口。

**验收**
- [ ] 跨文件迁移成为 mechanical move，而不是伴随语义再设计
- [ ] file split 前后行为对齐可用现有 smoke / replay / stage6 artifact + forward output snapshot 证明
- [ ] 不新增新的 implicit context / silent compat / broad swallow
- [ ] removal_policy §7 checklist 全通过

---

## 5) 验收硬指标（Phase 无关）

每个 Phase 收尾必须产出 / 对照以下硬指标，不是口头"行为不变"：

### 5.1 Forward output snapshot
- 固定 seed（`--seed 0`）、固定输入 batch（体积 ≤ 4 帧）、固定 config（建议 `config_stage6_offset0_e8x60.json`）
- dump：`pred_motion` 的 scalar checksum（元素求和 + L2 norm）+ 全部 `losses[key]` scalar 值
- 存 `tests/train/snapshots/<phase>_<yyyymmdd>.json`
- 本 Phase 对照上一 Phase bitwise 相等（Phase A / B / D）或容差 < 1e-12（Phase C 因 init 顺序可能有浮点差异）

### 5.2 State dict 指纹
- step0 ckpt 的 `state_dict.keys()` 集合（排序后）
- 每个 key 对应 tensor 的 `sha256(tensor.cpu().numpy().tobytes())`
- 本 Phase 对照上一 Phase **key-set bitwise 相等**，Phase C 允许 tensor checksum 漂移 < 1e-12

### 5.3 stage6 deterministic smoke 触发条件（精确化）
| Phase | 是否必跑 stage6 smoke |
|---|---|
| A | 否（只改异常处理，不碰语义） |
| B | 否（只做搬移 / 分区） |
| C.model | **是** |
| C.loss | 否（跑 forward output snapshot 即可） |
| D | **是**（forward 编排层改了） |
| E | **是**（跨文件迁移） |

stage6 smoke 命令：

```bash
PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain \
  --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json \
  --out_dir <new_out_dir> \
  --run_name <new_run_name> \
  --epochs 1 \
  --steps_per_epoch 5 \
  --save_step_ckpts 0,1,5 \
  --rollout_random_offset false \
  --seed 0
```

### 5.4 Removal policy 自检（每个 commit）
每个 commit 前跑：

```bash
git diff --cached | grep -E '\.get\(.*\.get\(|warnings?\.warn\(|state_dict\[.*\]\s*=\s*state_dict\.pop\(|#.*(compat|legacy)'
```

命中任意一条即回到 removal_policy §2 Step 1 重做。

---

## 6) 风险与回退

| 风险 | 触发信号 | 回退策略 |
|---|---|---|
| fail-fast 清理过猛，误伤合法 fallback | 原本可运行路径开始直接 raise | 只先清理 A2 inventory 类别 A；类别 B 保守处理为窄异常 + 明确默认值 |
| pure helper / stateful helper 误分类 | helper 被抽走后需要回读/回写 `self` 状态 | 先在同文件内聚类，不强行跨文件；对 `_warn_once` / `_parent_relative_matrices` / `_root_relative` 明确延后 |
| direct-pose contract 重构引入 step0 偏差 | stage6 deterministic smoke 的 step0 / state diff 漂移 | 任何 direct-pose contract / init 边界变化都必须跑 stage6 smoke + state_dict 指纹对照；不过线就停 |
| forward / loss 拆分后定位更难 | 帮助函数增多但边界没有变清楚 | 强制"编排壳 + 纯计算块"模式；禁止纯转发 wrapper 与新黑盒 context |
| 拆分中无声引入 compat shim | `git diff` grep 命中 removal_policy §6 正则 | 立即回退 commit，按 removal_policy §2 Step 1 重做 |
| rot6d 子系统误迁入 geometry.py | `train/geometry.py` 出现 `self._warn_once` / `group_slices` 引用 | 立即回退；参照 §2.4 归属决策 |

---

## 7) 建议提交拆分（Commit Plan）

1. Commit 1: Phase A — fail-fast inventory 文档 + 类别 A 热点收窄 + 最小回归测试 + 基线 snapshot
2. Commit 2: Phase B.B1 + B3 — module-level pure helper 提取 + skeleton/weight-cache cluster（**合并**，避免 helper 调用点跨 commit 改两次）
3. Commit 3: Phase B.B4 — applicators cluster in-file 分区
4. Commit 4: Phase B.B5 — rot6d cluster in-file 分区
5. Commit 5: Phase C.model — direct-pose model 侧 typed contract + runtime 下沉
6. Commit 6: Phase C.loss — direct-pose loss 侧 cluster + stats key contract 表
7. Commit 7: Phase D — `EventMotionModel.forward(...)` 与 `MotionJointLoss` 编排层收口
8. Commit 8+: Phase E — 跨文件迁移 + `_warn_once` 重构 + 外围 exception 清理 + 文档更新

---

## 8) 本轮优先级

- **P0（立即）**: Phase A 全部 —— fail-fast inventory 产出 + 类别 A 热点收窄 + 基线 snapshot。
- **P1**: Phase B 全部（B1 / B3 / B4 / B5）—— 在当前文件内完成 pure helper + 四个 cluster 就位。
- **P2**: Phase C（model 侧 + loss 侧）+ Phase D 编排收口。
- **P3**: Phase E 跨文件迁移。

---

## 9) 当前基线（便于后续 update）

- `train/models.py` LOC：`5511`
- broad `Exception` handler：当前 `65`（初始 inventory 为 `72`；本轮累计已收窄 A01–A07 七处；当前精确 `except Exception:`=`64`，`except Exception as exc`=`1`）
- `EventMotionModel.forward(...)` 规模：约 `1432` 行（`train/models.py:2341`）
- `EventMotionModel.__init__(...)` 规模：约 `539` 行（`train/models.py:410`）
- `_build_direct_pose_modules()` 规模：约 `292` 行（`train/models.py:1222`）
- `MotionJointLoss` 类跨度：约 `1738` 行（`train/models.py:3774`）
- direct-pose 相关方法数：model 侧 `7`，loss 侧 `8`
- rot6d 核心 cluster 方法数：`8`（`train/models.py:4474`–`4642`；不含 `train/models.py:3975` 的 `_resolve_rot6d_columns`）
- component applicators 核心 cluster 方法数：`11`（`train/models.py:4678`–`5407`）

建议固定回归命令：

```bash
python3 -m py_compile train/models.py tests/train/test_event_motion_model_refactor_phase_d.py tests/train/test_train_models_failfast.py
python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d tests.train.test_train_models_failfast
```

每个 commit 前必跑 removal_policy §6 反模式 grep：

```bash
git diff --cached | grep -E '\.get\(.*\.get\(|warnings?\.warn\(|state_dict\[.*\]\s*=\s*state_dict\.pop\(|#.*(compat|legacy)'
```

---

## 10) 与配套文档关系

| 文档 | 关系 |
|---|---|
| [`docs/removal_policy.md`](../removal_policy.md) | **强制遵守**（§0、§4 Phase A、§5.4、§6、§7）——本次重构所有删除 / 收窄 / 迁移动作的行为契约 |
| `train/MODULE_BOUNDARIES.md` | 代码归属红线；Phase E 跨文件迁移前必须对照 |
| `docs/changes/2026-03-16_event_motion_model_refactor_phaseD_report.md` | direct-pose 历史背景 |
| `docs/changes/2026-04-15_pretrain_contact_contract_tightening.md` | contract tightening 范本 |
| `docs/changes/2026-04-21_train_models_fail_fast_inventory.md` | Phase A inventory 产出物（72 处 broad handler 三分类；A=`51`，B=`19`，C=`2`）；已于 2026-04-21 创建 |
| `docs/basetrain_pipeline.md` / `docs/posttrain_pipeline.md` | canonical pipeline；Phase C / D 改动前对照 |

---

## 11) v2 相对 v1 的变更摘要

- **§0 必须遵守的纪律**：新增，引入 `docs/removal_policy.md` 为本次重构的硬约束，贯穿 Phase A–E 和每个 commit。
- **§2.1 P0**：新增"direct-pose 热点（loss 侧）"一行，补足 v1 遗漏。
- **§2.2 P1**：新增 "rot6d 子系统 cluster 缺失" 与 "applicators 簇无边界"，v1 漏列。
- **§2.4**：新增 rot6d 归属决策，明确**不迁入 `train/geometry.py`**。
- **Phase A**：A2 从 3 个点位扩展为 72 处全量 broad handler inventory（三分类）+ 独立文档产出物。
- **Phase B**：`_warn_once` 从 B2 移到 E7；新增 B4 applicators cluster、B5 rot6d cluster。
- **Phase C**：拆成 C.model + C.loss 两条独立路径，补足 loss 侧 direct-pose 8 方法。
- **Phase E**：跨文件目标从 2 个扩展为 7 个（E1–E7），补足 rot6d / loss 侧 direct-pose / applicators / tracker。
- **§5 验收硬指标**：新增 forward output snapshot、state_dict 指纹、stage6 smoke 触发条件表、removal_policy §6 grep 自检。
- **§7 Commit Plan**：从 7 条扩展为 8 条，合并 v1 的第 2 + 3 条（pure helper 与 skeleton cluster），拆分 B4 / B5 / C.model / C.loss。

---

## 12) Execution Log

### 2026-04-21 — Phase A fail-fast inventory landed

- **本轮目标**：只落地 Phase A 第一项 fail-fast inventory，不修改 `train/models.py` 行为，不做 Phase B/C/D 结构重构。
- **实际完成项**：创建 `docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；完成 `train/models.py` broad `Exception` handler 全量 inventory；按 A/B/C 分类为 A=`51`、B=`19`、C=`2`。
- **修改文件列表**：`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：AST 口径 broad `Exception` handler 共 `72`；其中精确 `except Exception:` 为 `71`，另有 `except Exception as exc` 为 `1`（`train/models.py:3685`）；`forward` 区 `32`，`MotionJointLoss` 区 `12`，其余 `28`。
- **热点对照**：direct-pose / `EventMotionModel.forward(...)` / `MotionJointLoss` 热点与 roadmap 判断一致；新增明确最高风险点 `train/models.py:3653`、`train/models.py:3546`、`train/models.py:2406`、`train/models.py:1533`、`train/models.py:3852`–`train/models.py:3887`。
- **运行过的命令**：`rg --files -g 'AGENTS.md' ...` 读取约束文档；`python3 - <<'PY' ... ast.parse(train/models.py) ...` 统计 broad handlers；`rg -n "except Exception( as ...)?:" train/models.py` 交叉检查行号；`wc -l train/models.py` 验证 LOC；`python3 - <<'PY' ... inventory_table_counts_ok ... roadmap_backfill_markers_ok ...` 验证文档可读、引用路径存在、统计结果一致、inventory 表计数一致、roadmap 回填 marker 存在。
- **验证结果**：通过；新 inventory 文档可读，roadmap 回填已落盘，引用路径存在，AST 统计与源码 grep 口径一致；未写入临时辅助脚本，无临时文件残留。
- **阻塞项 / 风险**：roadmap 原基线 `71` 是精确 `except Exception:` 文本计数；后续应统一使用 `72` broad handler 口径，避免漏掉 `except Exception as exc`。
- **下一轮建议动作**：按 inventory 类别 A 优先处理 direct-pose forward 大吞错点，先改 `train/models.py:3653` 与 `train/models.py:3546`，随后处理 `3501`、`3539`、`3614`、`3637`、`2406`；仍不要提前做 helper 提取或文件拆分。

### 2026-04-21 — Phase A A01/A02 fail-fast patch

- **本轮目标**：只处理 inventory 类别 A 的前两个 highest-risk direct-pose `forward` 大吞错点，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 side-routed 与 non-side direct-pose leg residual 外层 `except Exception: pass` 改为窄异常捕获并 `RuntimeError` fail-fast；新增两个 regression tests 覆盖 leg residual head failure 不再 silent skip。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `72` 降到 `70`；精确 `except Exception:` 从 `71` 降到 `69`；`except Exception as exc` 仍为 `1`（`direct_pose forward failed` outer wrapper）。
- **运行过的命令**：`python3 - <<'PY' ... ast.parse(train/models.py) ...` 验证 broad handler 计数；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py`；`python3 -m unittest tests.train.test_train_models_failfast`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`。
- **验证结果**：通过；新增 fail-fast tests `2` 个通过，相邻 direct-pose refactor tests `6` 个通过，`train/models.py` 与新增测试文件 py_compile 通过。
- **阻塞项 / 风险**：A03–A07 的 gate / scale / sign-gate 内层 fallback 仍会把部分 direct-pose leg residual 子路径降级为 ungated / unscaled 输出；A12 `_expand_state_sequence(...)` 仍会返回 `None`。
- **下一轮建议动作**：继续处理 A03–A07，优先把 `direct_pose_leg_gate_mode in {"learned", "scale"}` 与 `direct_pose_leg_side_sign_gate` 的 inner broad catches 改成窄异常 + fail-fast，并补充同文件最小回归。

### 2026-04-21 — Phase A A03–A07 inner gate/scale/sign-gate fail-fast patch

- **本轮目标**：只处理 direct-pose leg residual inner gate / scale / sign-gate 的 5 个 silent fallback，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 `train/models.py:3450`、`train/models.py:3501`、`train/models.py:3539`、`train/models.py:3618`、`train/models.py:3641` 对应的 inner broad catch 已改为窄异常捕获并 fail-fast raise；新增 5 个 regression tests 覆盖 side-routed learned gate / scale gate / sign gate 以及 non-side learned gate / scale gate failure。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `70` 降到 `65`；精确 `except Exception:` 从 `69` 降到 `64`；`except Exception as exc` 仍为 `1`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py`；`python3 -m unittest tests.train.test_train_models_failfast`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...` 复核 broad handler 计数。
- **验证结果**：通过；`tests.train.test_train_models_failfast` 共 `7` 个用例通过，`tests.train.test_event_motion_model_refactor_phase_d` 共 `6` 个用例通过，broad handler 计数与文档回填一致。
- **阻塞项 / 风险**：A12 `_expand_state_sequence(...)` 仍会把 phase / side-cue contract 退化为 `None`；A08–A11 的 direct-pose feature / time / phase contract 仍可能静默降级。
- **下一轮建议动作**：优先处理 `train/models.py:2406`，把 `_expand_state_sequence(...)` 从 silent `None` 改为带字段名和 shape 上下文的 fail-fast；随后处理 A08–A11。

### 2026-04-21 — Phase A A12 `_expand_state_sequence(...)` fail-fast patch

- **本轮目标**：只处理 inventory A12 `train/models.py:2406`，把 `_expand_state_sequence(...)` 的 silent `None` 改成 fail-fast，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 `_expand_state_sequence(...)` 现对 `phase_z` / `phase_event_age` 执行显式 shape+broadcast contract 校验；失败时抛出带字段名、期望 broadcast 语义、实际 `shape` / `ndim`、`B` / `Tq` / `feat_dim` 上下文的 `RuntimeError`；调用点最小调整为显式传入 `field_name`；`tests/train/test_train_models_failfast.py` 新增 `phase_z` 与 `phase_event_age` 的 regression tests。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `65` 降到 `64`；精确 `except Exception:` 从 `64` 降到 `63`；`except Exception as exc` 仍为 `1`；Phase A 当前累计已处理 A01–A07 + A12，类别 A 原始条目剩余 `43` 个。
- **A3 / A4 推进**：A3 已把 direct-pose leg residual outer/inner fallback 与 `_expand_state_sequence(...)` 收窄为 fail-fast；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `9` 个场景，但 `_direct_pose_split_state()` strict failure、loss skeleton 输入异常、generic forward shape 覆盖仍待补。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...` 复核 broad handler 计数。
- **验证结果**：通过；`tests.train.test_train_models_failfast` 共 `9` 个用例通过，`tests.train.test_event_motion_model_refactor_phase_d` 共 `6` 个用例通过，broad handler 计数与源码一致，文档回填已落盘。
- **阻塞项 / 风险**：A08–A11 的 direct-pose per-side phase / side-embed / time PE / feature-source contract 仍可能静默降级；当前 worktree 还有与本轮无关的既有改动，removal-policy grep 需按本轮触达 hunk 局部复核。
- **下一轮建议动作**：优先处理 A08 `train/models.py:3294`，把 per-side phase view failure 从零填充改为 fail-fast；随后串行处理 A09 `:3383`、A10 `:3137`、A11 `:3122`。

### 2026-04-21 — Phase A A08 side-routed phase view fail-fast patch

- **本轮目标**：只处理 inventory A08 `train/models.py:3294`，把 side-routed direct-pose per-side phase view 的 zero-fill fallback 改成 fail-fast，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 side-routed leg branch 现对 `phase_z_in_direct` 执行严格 view contract 校验，要求 per-side view 前为 `(B, Tq, 2*contact_channels)`；失败时抛出包含字段名、期望 `(B, Tq, contact_channels, 2)` 语义、实际 `shape` / `ndim`、`ch_r` / `ch_l` / `contact_dim` 的 `RuntimeError`；`tests/train/test_train_models_failfast.py` 新增 `test_side_routed_phase_z_view_contract_failure_raises`。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `64` 降到 `63`；精确 `except Exception:` 从 `63` 降到 `62`；`except Exception as exc` 仍为 `1`；Phase A 当前累计已处理 A01–A08 + A12，类别 A 原始条目剩余 `42` 个。
- **A3 / A4 推进**：A3 已把 direct-pose leg residual outer/inner fallback、`_expand_state_sequence(...)`、side-routed phase view fallback 收窄为 fail-fast；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `10` 个场景，但 `_direct_pose_split_state()` strict failure、loss skeleton 输入异常、generic forward shape 覆盖仍待补。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...` 复核 broad handler 计数。
- **验证结果**：通过；`tests.train.test_train_models_failfast` 共 `10` 个用例通过，`tests.train.test_event_motion_model_refactor_phase_d` 共 `6` 个用例通过，broad handler 计数与源码一致，文档回填已落盘。
- **阻塞项 / 风险**：A09 `train/models.py:3383` side embedding failure、A10 `train/models.py:3137` direct-pose time PE concat failure、A11 `train/models.py:3122` feature-source fallback 仍可能静默降级。
- **下一轮建议动作**：优先处理 A09 `train/models.py:3383`，把 enabled side embedding failure 从 drop feature 改为 fail-fast；随后处理 A10 / A11。

### 2026-04-21 — Phase A A09 side embedding fail-fast patch

- **本轮目标**：只处理 inventory A09 `train/models.py:3383`，把 enabled side embedding failure 从 silent drop 改成 fail-fast，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 side-routed leg branch 现对 `direct_pose_leg_side_embed` 执行严格 broadcast contract 校验，要求 right/left embedding 可扩展到 `(B, Tq, D)`；失败时抛出包含期望 broadcast 语义、`embed_weight_shape`、实际 `emb_r.shape` / `emb_l.shape` 的 `RuntimeError`；`tests/train/test_train_models_failfast.py` 新增 `test_side_routed_side_embedding_failure_raises`。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `63` 降到 `62`；精确 `except Exception:` 从 `62` 降到 `61`；`except Exception as exc` 仍为 `1`；Phase A 当前累计已处理 A01–A09 + A12，类别 A 原始条目剩余 `41` 个。
- **A3 / A4 推进**：A3 已把 direct-pose leg residual outer/inner fallback、`_expand_state_sequence(...)`、side-routed phase view/side-embed fallback 收窄为 fail-fast；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `11` 个场景，但 `_direct_pose_split_state()` strict failure、loss skeleton 输入异常、generic forward shape 覆盖仍待补。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...` 复核 broad handler 计数。
- **验证结果**：通过；`tests.train.test_train_models_failfast` 共 `11` 个用例通过，`tests.train.test_event_motion_model_refactor_phase_d` 共 `6` 个用例通过，broad handler 计数与源码一致，文档回填已落盘。
- **阻塞项 / 风险**：A10 `train/models.py:3137` direct-pose time PE concat failure、A11 `train/models.py:3122` feature-source fallback 仍可能静默降级。
- **下一轮建议动作**：优先处理 A10 `train/models.py:3137`，把 direct-pose time PE concat failure 从 ignore 改成 fail-fast；随后处理 A11 `:3122`。

### 2026-04-21 — Phase A A10 direct-pose time PE concat fail-fast patch

- **本轮目标**：只处理 inventory A10 `train/models.py:3137`，把 direct-pose time PE concat failure 从 ignore 改成 fail-fast，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 direct-pose 分支现在对 `direct_feat` 与 `time_pe_direct` 执行严格 concat contract 校验，要求分别满足 `(B, Tq, F)` 与 `(B, Tq, time_pe_dim)`；`to(...)` 或 `torch.cat(...)` 失败时抛出包含 `direct_feat.shape`、`time_pe_direct.shape`、`time_pe_dim` 的 `RuntimeError`；`tests/train/test_train_models_failfast.py` 新增 `test_direct_pose_time_pe_concat_failure_raises`。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `62` 降到 `61`；精确 `except Exception:` 从 `61` 降到 `60`；`except Exception as exc` 仍为 `1`；Phase A 当前累计已处理 A01–A10 + A12，类别 A 原始条目剩余 `40` 个。
- **A3 / A4 推进**：A3 已把 direct-pose leg residual outer/inner fallback、`_expand_state_sequence(...)`、side-routed phase view/side-embed fallback、direct-pose time PE concat fallback 收窄为 fail-fast；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `12` 个场景，但 `_direct_pose_split_state()` strict failure、loss skeleton 输入异常、generic forward shape 覆盖仍待补。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...` 复核 broad handler 计数。
- **验证结果**：通过；`tests.train.test_train_models_failfast` 共 `12` 个用例通过，`tests.train.test_event_motion_model_refactor_phase_d` 共 `6` 个用例通过，broad handler 计数与源码一致，文档回填已落盘。
- **阻塞项 / 风险**：A11 `train/models.py:3122` `direct_pose_feat_source` fallback 仍可能静默降级；其后是 contact-plan / time feature 条目。
- **下一轮建议动作**：优先处理 A11 `train/models.py:3122`，把 `direct_pose_feat_source` invalid attr fallback 从 `cond` 默认改为 fail-fast；随后再处理 contact-plan / time feature 条目。

### 2026-04-21 — Phase A A11 + A14–A16 clustered fail-fast patch

- **本轮目标**：按相邻 `forward` cluster 一起处理 A11、A14、A15、A16：收紧 `direct_pose_feat_source`、`time_index` 规范化、contact/direct time PE 构造，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 `direct_pose_feat_source` 不再在 init 或 forward 中 silent fallback 到 `cond`；`time_index` 现在要求显式满足 broadcast 到 `(B, Tq)` 的 contract；contact-plan 与 direct-pose time PE 构造失败时都会抛出带 `pe_dim`、`t_grid.shape`、`base` 上下文的 `RuntimeError`；`tests/train/test_train_models_failfast.py` 新增 `test_direct_pose_feat_source_contract_failure_raises`、`test_time_index_contract_failure_raises`、`test_contact_plan_time_pe_construction_failure_raises`、`test_direct_pose_time_pe_construction_failure_raises`。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `61` 降到 `57`；精确 `except Exception:` 从 `60` 降到 `56`；`except Exception as exc` 仍为 `1`；Phase A 当前累计已处理 A01–A12、A14–A16，类别 A 原始条目剩余 `36` 个。
- **A3 / A4 推进**：A3 已把 direct-pose leg residual、phase/side-cue contract、side-routed phase/side-embed contract、direct-pose feature/time cluster 收窄为 fail-fast；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `16` 个场景，但 `_direct_pose_split_state()` strict failure、loss skeleton 输入异常、generic forward shape 覆盖仍待补。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...` 复核 broad handler 计数。
- **验证结果**：通过；`tests.train.test_train_models_failfast` 共 `16` 个用例通过，`tests.train.test_event_motion_model_refactor_phase_d` 共 `6` 个用例通过，broad handler 计数与源码一致，文档回填已落盘。
- **阻塞项 / 风险**：下一批主热点已切到 contact-plan init / phase append / side-cue append / stack failures；这些点位分散在 event-clock on/off 两条分支里，适合继续按 cluster 一次处理。
- **下一轮建议动作**：按 cluster 处理 `2513`、`2721`、`2730`、`2810`、`2819`、`2847`、`2852`、`2856`，随后再处理 `2767` / `2830` 的 enabled time bias 条目。

### 2026-04-21 — Phase A A13 + A19–A27 clustered fail-fast patch

- **本轮目标**：按相邻 `forward` cluster 一起处理 A13、A19–A27：收紧 contact-plan observed init、event-clock on/off 的 phase append / side-cue append / time bias，以及 phase/cue/logits stack，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 `contact_plan_init_mode in {'obs', 'learnable+obs'}` 不再在 init head 失败时 silent fallback；event-clock on/off 的 phase / side-cue append 不再 `pass`；enabled contact-plan time bias 不再静默丢掉；`phase_in_direct_seq`、`leg_side_cue_seq`、`plan_logits` 的 stack 失败现在都会带 step / element shape 上下文 fail-fast；`tests/train/test_train_models_failfast.py` 新增 `test_contact_plan_observed_init_failure_raises`、`test_event_clock_phase_append_failure_raises`、`test_event_clock_side_cue_append_failure_raises`、`test_non_event_clock_phase_append_failure_raises`、`test_non_event_clock_side_cue_append_failure_raises`、`test_event_clock_time_bias_failure_raises`、`test_non_event_clock_time_bias_failure_raises`、`test_phase_sequence_stack_failure_raises`、`test_side_cue_sequence_stack_failure_raises`、`test_contacts_plan_logits_stack_failure_raises`。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `57` 降到 `47`；精确 `except Exception:` 从 `56` 降到 `46`；`except Exception as exc` 仍为 `1`；Phase A 当前累计已处理 A01–A16、A19–A27，类别 A 原始条目剩余 `26` 个。
- **A3 / A4 推进**：A3 已把 direct-pose leg residual、phase/side-cue contract、side-routed phase/side-embed contract、direct-pose feature/time cluster、contact-plan init/append/stack/time-bias cluster 收窄为 fail-fast；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `26` 个场景，但 `_direct_pose_split_state()` strict failure、loss skeleton 输入异常、generic forward shape 覆盖仍待补。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...` 复核 broad handler 计数。
- **验证结果**：通过；`tests.train.test_train_models_failfast` 共 `26` 个用例通过，`tests.train.test_event_motion_model_refactor_phase_d` 共 `6` 个用例通过，broad handler 计数与源码一致，文档回填已落盘。
- **阻塞项 / 风险**：下一批主热点已切到 `forward` auxiliary / late-forward cluster（adaptive-history、period feature、lambda-fusion）；再之后才建议切 build-time cluster。
- **下一轮建议动作**：按 cluster 处理 `2664`、`2708`、`3740`、`3755`，然后再处理 build-time layout / deterministic init / routing metadata cluster。

### 2026-04-21 — Phase A A17/A18/A28/A29 auxiliary clustered fail-fast patch

- **本轮目标**：按 `forward` auxiliary / late-forward cluster 一起处理 A17、A18、A28、A29：收紧 adaptive-history、frozen period feature、lambda-fusion rollout-step 与 lambda-fusion head，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 event-clock adaptive-history 与 frozen period feature 失败不再 silent degradation；lambda-fusion rollout-step 输入不再退化为零 step feature；lambda-fusion head 失败不再吞掉 `lambda_fusion` / `lambda_fusion_logits` 输出；`tests/train/test_train_models_failfast.py` 新增 `test_adaptive_history_failure_raises`、`test_frozen_period_feature_failure_raises`、`test_lambda_fusion_rollout_step_contract_failure_raises`、`test_lambda_fusion_forward_failure_raises`。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `47` 降到 `43`；精确 `except Exception:` 从 `46` 降到 `42`；`except Exception as exc` 仍为 `1`；Phase A 当前累计已处理 A01–A29（除 A30），类别 A 原始条目剩余 `22` 个。
- **A3 / A4 推进**：A3 已把 `forward` 主要 direct-pose / contact-plan / auxiliary clusters 收窄为 fail-fast；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `30` 个场景，但 `_direct_pose_split_state()` strict failure、loss skeleton 输入异常、generic forward shape 覆盖仍待补。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...` 复核 broad handler 计数。
- **验证结果**：通过；`tests.train.test_train_models_failfast` 共 `30` 个用例通过，`tests.train.test_event_motion_model_refactor_phase_d` 共 `6` 个用例通过，broad handler 计数与源码一致，文档回填已落盘。
- **阻塞项 / 风险**：下一批建议切 build-time layout / deterministic init / routing metadata cluster；该批可能影响模型构造与 deterministic init 行为，测试应更偏向构造失败与最小 state_dict/forward smoke。
- **下一轮建议动作**：按 cluster 处理 `1533`、`1556`、`1568`、`1588`、`1190`、`1206`、`980`、`1019`、`1144`、`1151`。

### 2026-04-21 — Phase A Batch 3 build-time layout / init / routing metadata cluster

- **本轮目标**：按 build-time cluster 一起处理 A31–A40：收紧 lambda/so3 rot6d layout 推断、contact-plan deterministic init、direct-pose routing metadata register / side-embed init，禁止 Phase B/C/D 结构重构。
- **实际完成项**：`train/models.py` 中 A31 / A32 不再在 rot6d layout 推断失败时把 joint count 静默置 `0`；A33 / A34 / A35 / A36 不再吞掉 deterministic zero/logit init 失败；A37 / A38 / A39 不再吞掉 routing metadata buffer register/update 失败；A40 不再吞掉 side embedding zero-init 失败。`tests/train/test_train_models_failfast.py` 新增 10 个 constructor/build-time regression tests，覆盖 Batch 3 全部 inventory 条目。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `43` 降到 `33`；精确 `except Exception:` 从 `42` 降到 `32`；`except Exception as exc` 仍为 `1`；Phase A 当前累计已处理 A01–A29（除 A30）+ A31–A40，类别 A 原始条目剩余 `12` 个（A30 + A41–A51）。
- **A3 / A4 推进**：A3 已把 `forward` 主路径与 build-time layout / deterministic init / routing metadata cluster 收窄为 fail-fast；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `40` 个场景，但 `_direct_pose_split_state()` strict failure、MotionJointLoss skeleton / payload fail-fast 与 generic forward shape 覆盖仍待补。
- **新增 / 更新测试**：`test_lambda_fusion_rot6d_layout_failure_raises_at_build_time`、`test_so3_corrector_rot6d_layout_failure_raises_at_build_time`、`test_lambda_fusion_deterministic_init_failure_raises_at_build_time`、`test_so3_corrector_deterministic_init_failure_raises_at_build_time`、`test_contact_plan_init_head_deterministic_init_failure_raises_at_build_time`、`test_contact_plan_time_head_deterministic_init_failure_raises_at_build_time`、`test_direct_pose_leg_joint_index_buffer_registration_failure_raises`、`test_direct_pose_split_leg_index_buffer_registration_failure_raises`、`test_direct_pose_side_position_buffer_registration_failure_raises`、`test_direct_pose_side_embedding_deterministic_init_failure_raises_at_build_time`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`(sed -n '970,1265p' train/models.py; sed -n '1520,1685p' train/models.py; sed -n '1,220p' tests/train/test_train_models_failfast.py; sed -n '760,980p' tests/train/test_train_models_failfast.py) | rg ...`。
- **验证结果**：通过；`py_compile` 通过，`unittest` 共 `46` 个用例通过，AST 计数得到 broad=`33` / exact=`32` / as_exc=`1`，Batch 3 touched ranges 的 removal-policy 反模式 grep 未命中新增 nested fallback / warning-only / silent rename / compat 注释。
- **阻塞项 / 风险**：Batch 4 将首次触碰 `MotionJointLoss` config / skeleton / payload scalar resolver；arm/else weight、EMA beta、ratio min/max、epsilon 的 fail-fast 需要保持 valid-input loss 数学语义不变。
- **下一轮建议动作**：进入 Batch 4，处理 A41–A51 的 MotionJointLoss config / skeleton / payload cluster；Batch 4 完成并验证后，再判断 A30 outer direct-pose wrapper 是否保留 contextual wrap。

### 2026-04-21 — Phase A Batch 4 MotionJointLoss config / skeleton / payload cluster

- **本轮目标**：按 MotionJointLoss cluster 一起处理 A41–A51：收紧 direct-pose arm/else weight、group-norm beta / ratio / eps、skeleton offsets、payload override scalar resolver，不改 loss 数学语义，不加 default fallback / warning-only。
- **实际完成项**：`train/models.py` 中 A41 / A42 / A43 / A44 / A45 / A46 现对 invalid scalar / range 直接 fail-fast；A47 / A48 现对 invalid skeleton offsets type / shape / finite 性质直接 fail-fast；A49 / A50 现对 invalid payload `arm_weight` / `else_weight` override 直接 fail-fast；A51 现把 runtime scalar resolver 改成 typed scalar 校验，不再吞掉 bad override。`tests/train/test_train_models_failfast.py` 新增 11 个 MotionJointLoss regression tests，覆盖 Batch 4 全部 inventory 条目。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `33` 降到 `22`；精确 `except Exception:` 从 `32` 降到 `21`；`except Exception as exc` 仍为 `1`；Phase A 当前累计已处理 A01–A29（除 A30）+ A31–A51，类别 A 原始条目剩余 `1` 个（A30）。
- **A3 / A4 推进**：A3 已把 `forward` 主路径、build-time cluster、MotionJointLoss config / skeleton / payload cluster 收窄为 fail-fast；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `51` 个场景，但 `_direct_pose_split_state()` strict failure 与 generic forward shape 覆盖仍待补。
- **新增 / 更新测试**：`test_motion_joint_loss_arm_weight_invalid_raises`、`test_motion_joint_loss_else_weight_invalid_raises`、`test_motion_joint_loss_group_norm_beta_invalid_raises`、`test_motion_joint_loss_group_norm_ratio_min_invalid_raises`、`test_motion_joint_loss_group_norm_ratio_max_invalid_raises`、`test_motion_joint_loss_group_norm_eps_invalid_raises`、`test_motion_joint_loss_ctor_skeleton_offsets_invalid_raises`、`test_motion_joint_loss_set_skeleton_offsets_invalid_raises`、`test_motion_joint_loss_payload_arm_weight_override_invalid_raises`、`test_motion_joint_loss_payload_else_weight_override_invalid_raises`、`test_motion_joint_loss_group_norm_runtime_scalar_invalid_raises`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`(sed -n '4260,4320p' train/models.py; sed -n '4356,4528p' train/models.py; sed -n '5390,5515p' train/models.py; sed -n '1,220p' tests/train/test_train_models_failfast.py; sed -n '980,1140p' tests/train/test_train_models_failfast.py) | rg ...`。
- **验证结果**：通过；`py_compile` 通过，`unittest` 共 `57` 个用例通过，AST 计数得到 broad=`22` / exact=`21` / as_exc=`1`，Batch 4 touched ranges 经局部人工复核后未新增 nested fallback / warning-only / silent rename / compat 注释（唯一 `.get(...get...)` grep 命中为既有 `layout.get('slices') if isinstance(layout.get('slices'), dict)`，非 nested fallback）。
- **阻塞项 / 风险**：仅剩 A30 outer direct-pose wrapper；若保留 contextual wrap，需要把 broad `Exception as exc` 收窄为 typed contextual re-raise，并同步确认现有 direct-pose fail-fast 测试的上下文消息仍保留。
- **下一轮建议动作**：进入 Batch 5，处理 A30 并重新扫描 `train/models.py` broad handler，确认 Phase A 是否已无 Category A 剩余。

### 2026-04-21 — Phase A Batch 5 sweep / A30 typed contextual wrap

- **本轮目标**：重新扫描 `train/models.py` broad handler，对照 inventory 找出剩余 Phase A Category A；重点判断 A30 outer direct-pose wrapper 是否保留 contextual wrap。
- **实际完成项**：`train/models.py` 中 A30 已处理；保留 `direct_pose forward failed` contextual wrap，但把 `except Exception as exc` 收窄为 typed contextual re-raise：`(AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError)`。重新扫描确认剩余 broad handler 全部对应 Category B / C。
- **修改文件列表**：`train/models.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `22` 降到 `21`；精确 `except Exception:` 为 `21`；`except Exception as exc` 从 `1` 降到 `0`；Phase A Category A 原始条目剩余 `0`。
- **A3 / A4 推进**：A3 的 Category A 部分已清零；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `51` 个 fail-fast 场景，但 `_direct_pose_split_state()` strict failure 与 generic forward shape 覆盖仍待补。
- **新增 / 更新测试**：Batch 5 未新增测试；现有 direct-pose fail-fast tests 继续覆盖 `direct_pose forward failed` contextual wrap。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`sed -n '4166,4180p' train/models.py | rg ...`。
- **验证结果**：通过；`py_compile` 通过，`unittest` 共 `57` 个用例通过，AST 计数得到 broad=`21` / exact=`21` / as_exc=`0`，A30 touched hunk 的 removal-policy 反模式 grep 未命中。
- **阻塞项 / 风险**：Phase A Category A 已无剩余；剩余 broad handler 为 Category B `19` + Category C `2`，不在本轮范围内。
- **下一轮建议动作**：优先按 Category B cluster 处理 constructor/runtime numeric fallback，或先补 `_direct_pose_split_state()` strict failure / generic forward shape 的 A4 coverage 缺口后再进入 Phase B。

### 2026-04-22 — Phase A A4 coverage completion

- **本轮目标**：补齐 A4 回归覆盖缺口，只验证 `_direct_pose_split_state()` strict failure 与 `EventMotionModel.forward(...)` generic input shape contract，不扩展到 Phase B/C/D。
- **实际完成项**：`tests/train/test_train_models_failfast.py` 新增 4 个 fail-fast regression tests，锁定 `_direct_pose_split_state()` 的 missing-index / disjoint-coverage contract，以及 generic `state` / `cond` shape precondition。
- **修改文件列表**：`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `21 -> 21`；精确 `except Exception:` 维持 `21 -> 21`；`except Exception as exc` 维持 `0 -> 0`；Phase A Category A 原始条目仍为 `0`。
- **A3 / A4 推进**：A3 Category A 已清零；A4 当前 `tests.train.test_train_models_failfast` 覆盖 `55` 个 fail-fast 场景，`_direct_pose_split_state()` strict failure 与 generic forward shape 覆盖已补齐。
- **新增 / 更新测试**：`test_direct_pose_split_state_missing_leg_index_tensor_raises`、`test_direct_pose_split_state_disjoint_coverage_mismatch_raises`、`test_event_motion_model_forward_state_shape_contract_failure_raises`、`test_event_motion_model_forward_cond_shape_contract_failure_raises`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`(sed -n '1684,1765p' train/models.py; sed -n '2470,2575p' train/models.py; sed -n '1035,1115p' tests/train/test_train_models_failfast.py) | rg ...`。
- **验证结果**：通过；`py_compile` 通过，`unittest` 共 `61` 个用例通过，AST 计数得到 broad=`21` / exact=`21` / as_exc=`0`，A4 touched ranges 的 removal-policy 反模式 grep 零命中（`rg` exit code `1`）。
- **阻塞项 / 风险**：Phase A 已完成；剩余 broad handler 为 Category B `19` + Category C `2`。若继续清理 Category B，需要保持 explicit typed fallback，不可回退成 silent fallback / warning-only。
- **下一轮建议动作**：优先按 Category B cluster 处理 constructor/runtime numeric fallback；若暂不动 broad handler，可进入 Phase B 的 in-file helper / cluster 化准备。

### 2026-04-22 — Phase A Category B constructor normalization B01–B11

- **本轮目标**：继续 Phase A，但只处理 `EventMotionModel.__init__` 的 Category B constructor normalization cluster（B01–B11），不进入 runtime / `MotionJointLoss` / helper 提取 / 文件拆分。
- **实际完成项**：`train/models.py` 中 B01–B11 已全部收窄：mode 字段改为 string-or-`None` typed normalization，unknown string 继续显式回落到设计默认值；numeric 字段去掉 broad `except Exception`，对 omission / `None` 保留默认，对 parse failure 与不合法 range 改为 constructor-time `TypeError` / `ValueError`；保留 `direct_pose_leg_scale_clamp_k <= 1` 的 explicit disable 语义与 `direct_pose_leg_side_embed_dim < 0` 的 explicit clamp 语义。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `21` 降到 `10`；精确 `except Exception:` 从 `21` 降到 `10`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 为 Category B `8`（B12–B19）+ Category C `2`（C01–C02）。
- **A3 / A4 推进**：A3 已从 Category A 扩展到 Category B constructor cluster；A4 新增 `15` 个 constructor regression tests，`tests.train.test_train_models_failfast` 总数升至 `70`，联合验证总用例数为 `76`。
- **新增 / 更新测试**：新增 `test_direct_pose_ctor_unknown_string_modes_use_explicit_defaults`、`test_direct_pose_ctor_none_defaults_preserved_for_constructor_cluster`、`test_direct_pose_phase_z_mode_exotic_object_raises`、`test_direct_pose_leg_mode_exotic_object_raises`、`test_direct_pose_leg_max_deg_invalid_range_raises`、`test_direct_pose_leg_gate_mode_exotic_object_raises`、`test_direct_pose_leg_gate_power_invalid_range_raises`、`test_direct_pose_leg_scale_log_clip_invalid_range_raises`、`test_direct_pose_leg_scale_clamp_k_invalid_type_raises`、`test_direct_pose_leg_scale_clamp_k_values_le_one_disable_explicitly`、`test_direct_pose_leg_contact_order_exotic_object_raises`、`test_direct_pose_leg_side_embed_dim_invalid_type_raises`、`test_direct_pose_leg_side_embed_dim_negative_clamps_to_zero`、`test_direct_pose_leg_side_cue_exotic_object_raises`、`test_direct_pose_leg_side_cue_tau_invalid_range_raises`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py`；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py && python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`(sed -n '650,835p' train/models.py; sed -n '1,260p' tests/train/test_train_models_failfast.py; tail -n 120 tests/train/test_train_models_failfast.py) | rg ...`。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `76` 个用例通过，AST 计数 broad=`10` / exact=`10` / as_exc=`0`，constructor touched hunks 的 removal-policy 反模式 grep 零命中（`rg` exit code `1`）。
- **阻塞项 / 风险**：本轮没有把 unknown string mode fallback 一刀切改成 fail-fast，而是保留 typed fallback；同时 `scale_clamp_k <= 1` 与 negative `side_embed_dim` 仍是设计内的显式 disable/clamp 语义。若下一轮继续收紧 explicit bad value，需要与 posttrain / ckpt contract 一起审视，避免 parser/model contract 分叉。
- **下一轮建议动作**：优先继续 Category B 剩余 broad handler：B12 `_init_bone_residual_adapters(...)`、B13 routing metadata names、B14/B15 eval runtime scalar fallback、B16–B18 debug/ablation helper；B19 保持最后单独评估。

### 2026-04-23 — Phase A Category B runtime/build helper B12–B18

- **本轮目标**：继续 Phase A Category B，但只处理 B12–B18，不进入 B19 / Category C / Phase B，不做 helper 提取和文件拆分。
- **实际完成项**：`train/models.py` 中 B12–B18 已全部收窄：adapter metadata fallback 仅捕获 metadata 类型异常；split name lookup 仅捕获 `TypeError` / `IndexError`；eval runtime scalar fallback 仅捕获 `TypeError` / `ValueError`；contact-plan debug stack 仅捕获 `torch.stack` shape `RuntimeError`；cross-leg ablation `contact_dim`/`joint_names` fallback 仅捕获预期 parse/iteration异常，runtime error 不再被吞。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `10` 降到 `3`；精确 `except Exception:` 从 `10` 降到 `3`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 为 Category B `1`（B19）+ Category C `2`（C01–C02）。
- **A3 / A4 推进**：A3 已完成 Category B B01–B18；A4 新增 `12` 个 B12–B18 regression tests，`tests.train.test_train_models_failfast` 总数升至 `82`，联合验证总用例数为 `88`。
- **新增 / 更新测试**：新增 `test_bone_residual_adapter_metadata_failure_disables_adapters`、`test_bone_residual_adapter_runtime_failure_raises`、`test_direct_pose_split_leg_name_type_error_falls_back_to_empty_names`、`test_direct_pose_split_leg_name_runtime_error_not_swallowed`、`test_eval_runtime_control_scalar_parse_failure_falls_back_to_default`、`test_eval_runtime_control_scalar_runtime_error_not_swallowed`、`test_contact_plan_debug_stack_shape_mismatch_falls_back_to_none`、`test_contact_plan_debug_stack_type_error_not_swallowed`、`test_direct_pose_leg_cross_leg_ablation_contact_dim_parse_failure_returns_none`、`test_direct_pose_leg_cross_leg_ablation_contact_dim_runtime_error_not_swallowed`、`test_direct_pose_leg_cross_leg_ablation_joint_names_parse_failure_returns_none`、`test_direct_pose_leg_cross_leg_ablation_joint_names_runtime_error_not_swallowed`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py && python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d && python3 - <<'PY' ... ast.parse(train/models.py) ...`；`(sed -n '930,1035p' train/models.py; sed -n '1098,1112p' train/models.py; sed -n '2180,2265p' train/models.py; sed -n '2470,2490p' train/models.py; sed -n '1,260p' tests/train/test_train_models_failfast.py; sed -n '470,640p' tests/train/test_train_models_failfast.py) | rg ...`。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `88` 个用例通过，AST 计数 broad=`3` / exact=`3` / as_exc=`0`，B12–B18 touched hunks 的 removal-policy 反模式 grep 零命中（`rg` exit code `1`）。
- **阻塞项 / 风险**：B12/B16/B17/B18 仍保留 optional/debug fallback 语义；这些 fallback 均已由测试锁定为 typed fallback，不再吞 arbitrary runtime error。
- **下一轮建议动作**：单独处理 B19 `MotionJointLoss.compute_attention_regularization`；若 B19 完成，剩余 broad handler 将只剩 C01/C02 compiler/export probes。

### 2026-04-23 — Phase A Category B attention regularization B19

- **本轮目标**：只处理 B19 `MotionJointLoss.compute_attention_regularization` 的 geomask broad fallback，不进入 Category C / Phase B，不改 attention 正则主算法。
- **实际完成项**：`train/models.py` 中 B19 已收窄：tensor `geomask` 仅接受 rank `2` / `3` / `4`；直接 broadcast 失败时只捕获 `RuntimeError`，rank-2 fallback 限定为 `view(1,T,T)`，rank-4 fallback 限定为 `mean(0)`，rank-3 / fallback 后仍不 broadcast 均显式报错；non-tensor `geomask` 保留 explicit distance-prior path。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 从 `3` 降到 `2`；精确 `except Exception:` 从 `3` 降到 `2`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 均为 Category C `2`（C01–C02）。
- **A3 / A4 推进**：A3 已完成全部 Category B B01–B19；A4 新增 `5` 个 B19 regression tests，`tests.train.test_train_models_failfast` 总数升至 `87`，联合验证总用例数为 `93`。
- **新增 / 更新测试**：新增 `test_attention_regularization_non_tensor_geomask_uses_distance_prior`、`test_attention_regularization_geomask_invalid_rank_raises`、`test_attention_regularization_geomask_rank2_bad_shape_raises`、`test_attention_regularization_geomask_rank3_bad_broadcast_raises`、`test_attention_regularization_geomask_rank4_bad_fallback_raises`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py && python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d && python3 - <<'PY' ... ast.parse(train/models.py) ...`；`(sed -n '5245,5305p' train/models.py; sed -n '1545,1645p' tests/train/test_train_models_failfast.py; sed -n '171,210p' docs/changes/2026-04-21_train_models_fail_fast_inventory.md; sed -n '559,610p' docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md) | rg ...`。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `93` 个用例通过，AST 计数 broad=`2` / exact=`2` / as_exc=`0`，B19 touched hunks 的 removal-policy 反模式 grep 零命中（`rg` exit code `1`）。
- **阻塞项 / 风险**：B19 保留 non-tensor `geomask` 的 distance-prior explicit fallback 与 rank-4 `mean(0)` fallback；现在两者都不再依赖 broad catch，fallback 失败会显式报错。
- **下一轮建议动作**：Phase A Category B 已清零；建议保留 C01/C02 到 compiler/export 专项，或进入 Phase B single-file helper / cluster 化。

### 2026-04-23 — Phase B.B1 pure helper free-function batch

- **本轮目标**：进入 Phase B，但只做 `train/models.py` 单文件 helper / cluster 化准备；本 batch 仅处理 `MotionJointLoss` 的 pure helper：`_masked_group_mean`、`_masked_group_weighted_mean`、`_stats_float`、`_stats_float_or`、`_ensure_temporal_axis`、`_setdefault_stats`；不处理 Category C，不做跨文件拆分，不改 `EventMotionModel.forward` 主逻辑，不改 loss 语义。
- **实际完成项**：`train/models.py` 已把上述 6 个 helper 提成模块级 private free functions；`MotionJointLoss` 内部调用点改为直接依赖模块级 helper；为保持仓内既有私有调用面稳定，类上保留同名 `staticmethod` alias，因此 `train/posttrain.py` 等现有调用点无需联动修改；默认超参、loss 数值语义、stats key、checkpoint contract 均保持不变。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `2 -> 2`；精确 `except Exception:` 维持 `2 -> 2`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 仍仅为 Category C `2`（C01–C02）。
- **A3 / A4 推进**：Phase B.B1 首批 pure helper 已就位；A4 新增 `4` 个 helper regression tests，`tests.train.test_train_models_failfast` 总数升至 `91`，联合验证总用例数为 `97`。
- **新增 / 更新测试**：新增 `test_motion_joint_loss_module_group_helpers_preserve_weighted_means`、`test_motion_joint_loss_module_stats_helpers_preserve_scalar_contract`、`test_motion_joint_loss_module_temporal_and_stats_default_helpers`、`test_motion_joint_loss_helper_aliases_remain_compatible`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '4496,4560p' train/models.py; sed -n '5574,5590p' train/models.py; sed -n '5678,5688p' train/models.py; sed -n '5838,5852p' train/models.py; sed -n '6136,6172p' train/models.py; sed -n '6266,6272p' train/models.py; sed -n '6384,6430p' train/models.py; sed -n '1,20p' tests/train/test_train_models_failfast.py; sed -n '310,355p' tests/train/test_train_models_failfast.py) | if rg ...`.
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `97` 个用例通过（`tests.train.test_train_models_failfast=91`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数 broad=`2` / exact=`2` / as_exc=`0`，touched hunks 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：本轮只完成 B1；B2 `_warn_once` 继续冻结，B3 skeleton / weight-cache cluster、B4 applicators cluster、B5 rot6d cluster 仍未开始；Category C 的 C01/C02 compiler / export probe broad handler 继续保留。
- **下一轮建议动作**：优先进入 Phase B.B3，在 `train/models.py` 内建立 skeleton / weight-cache cluster，先做分区与顺序整理，再补最小 cluster regression；Category C 继续留到 compiler/export 专项。

### 2026-04-23 — Phase B.B3 skeleton / weight-cache small batch

- **本轮目标**：继续 Phase B，但只做 `train/models.py` 单文件内的 skeleton / weight-cache cluster 小批次准备；本 batch 仅处理 `MotionJointLoss` 中 `set_skeleton` / `_resolve_direct_group_masks` / `_joint_weight_vector` / `_parent_relative_matrices` / `_root_relative` 一组的边界显式化，不处理 Category C，不做文件拆分。
- **实际完成项**：在 `train/models.py` 的 skeleton / weight-cache helper 起点新增 `# === future: train/loss/skeleton_weights.py ===` 分区注释，明确 B3 cluster 边界；未改 helper 行为与数值语义，仅新增 focused regression 锁定 direct-group root/overlap masking、weight cache invalidate 后 recompute 等价性，以及 parent/root relative rotation helper 的当前 contract。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `2 -> 2`；精确 `except Exception:` 维持 `2 -> 2`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 仍仅为 Category C `2`（C01–C02）。
- **A3 / A4 推进**：B3 已完成首个小批次边界标注与语义锁定；A4 新增 `3` 个 skeleton-cluster regression tests，`tests.train.test_train_models_failfast` 总数升至 `94`，联合验证总用例数为 `100`。
- **新增 / 更新测试**：新增 `test_motion_joint_loss_skeleton_cluster_direct_group_masks_exclude_root_and_overlaps`、`test_motion_joint_loss_skeleton_cluster_weight_cache_invalidation_preserves_values`、`test_motion_joint_loss_skeleton_cluster_relative_rotation_helpers`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '4840,4852p' train/models.py; sed -n '1780,1870p' tests/train/test_train_models_failfast.py; sed -n '195,207p' docs/changes/2026-04-21_train_models_fail_fast_inventory.md; sed -n '158,159p' docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md; sed -n '587,599p' docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md) | if rg ...`.
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `100` 个用例通过（`tests.train.test_train_models_failfast=94`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数 broad=`2` / exact=`2` / as_exc=`0`，touched hunks 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：本轮只完成 B3 的小批次准备与行为锁定，尚未做更完整的方法顺序整理；B3 仍剩 `_resolve_limb_masks` / `_collect_limb_local_stats` / `_compute_unified_weights_cpu` / `_rot_local_tail_scores` / `_rot_local_tail_candidates` 的系统整理，B4 / B5 仍未开始。
- **下一轮建议动作**：继续 B3，把 skeleton / weight-cache cluster 余下方法按“state update → mask resolution → weight compute → tail helpers → FK relative”顺序整理到同一块；完成后再进入 B4 applicators cluster。

### 2026-04-23 — Phase B.B3 skeleton / weight-cache cluster completion

- **本轮目标**：完成 Phase B.B3 剩余 in-file 顺序整理，把 skeleton / weight-cache cluster 按“state update / cache invalidation → mask resolution / skeleton stats → weight computation → tail-risk selection → FK-relative rotation views”整理成连续块；不处理 Category C，不做文件拆分，不改 `EventMotionModel.forward` 主逻辑。
- **实际完成项**：`train/models.py` 中 B3 cluster 已完成顺序整理：`set_bone_names` / `set_skeleton` / `_invalidate_weight_cache` 归入 state update；`_resolve_named_joint_indices` / `_resolve_limb_masks` / `_resolve_direct_group_masks` / `_collect_limb_local_stats` 归入 mask/stats；`_joint_weight_vector` / `_compute_unified_weights_cpu` 归入 weight；`_rot_local_tail_scores` / `_rot_local_tail_candidates` 归入 tail-risk；`_parent_relative_matrices` / `_root_relative` 归入 FK-relative。函数体语义未改，默认超参、loss 数值、stats key、checkpoint contract 均保持不变。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `2 -> 2`；精确 `except Exception:` 维持 `2 -> 2`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 仍仅为 Category C `2`（C01–C02）。
- **A3 / A4 推进**：B3 已完成 in-file cluster；A4 新增 `2` 个 B3 completion regression tests，`tests.train.test_train_models_failfast` 总数升至 `96`，联合验证总用例数为 `102`。
- **新增 / 更新测试**：新增 `test_motion_joint_loss_skeleton_cluster_limb_stats_helpers`、`test_motion_joint_loss_skeleton_cluster_tail_candidates_and_scores`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '4840,5245p' train/models.py; sed -n '1790,1925p' tests/train/test_train_models_failfast.py) | if rg ...`.
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `102` 个用例通过（`tests.train.test_train_models_failfast=96`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数 broad=`2` / exact=`2` / as_exc=`0`，touched hunks 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：本轮仍限定在 single-file cluster / mechanical reordering，没有抽文件，也没有把 stateful helpers free-function 化；后续 Phase E 迁移需要同步处理 `MotionJointLoss` 私有调用点与 `train/posttrain.py` 调用面。
- **下一轮建议动作**：进入 Phase B.B4 applicators cluster，在 `train/models.py` 内为 `_apply_rot_ortho_component` / `_apply_rot_local_tail_component` / `_apply_rot_local_component` / `_apply_root_velocity_components` / `_apply_motion_components` / `_apply_contact_plan_component` / `_apply_event_clock_components` / `_apply_contact_meas_component` / `_apply_omega_l2_component` / `_apply_aux_components` 建立 `# === future: train/loss/components.py ===` 分区，并补最小 stats/key regression；Category C 继续留到 compiler/export 专项。

### 2026-04-23 — Phase B.B4 applicators cluster first batch

- **本轮目标**：进入 Phase B.B4，但只做 applicators cluster 的首个小批次：在 `train/models.py` 内建立 `# === future: train/loss/components.py ===` 分区边界，并补最小 forward-level stats/key regression；不做跨文件拆分，不改 `EventMotionModel.forward` 主逻辑，不处理 Category C。
- **实际完成项**：`train/models.py` 在 `_apply_rot_ortho_component` 前新增 `# === future: train/loss/components.py ===`，明确 component applicators 的 future split 边界；未移动或改写 applicator 函数体。新增回归通过真实 `MotionJointLoss.forward(...)` 同时触发 rot ortho / rot local / root velocity / direct pose / contact plan / event clock / contact meas / omega L2 applicators，锁定关键 stats key 与 root velocity 数值。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `2 -> 2`；精确 `except Exception:` 维持 `2 -> 2`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 仍仅为 Category C `2`（C01–C02）。
- **A3 / A4 推进**：B4 已完成首个小批次边界标注与 stats/key 合同锁定；A4 新增 `1` 个 B4 regression test，`tests.train.test_train_models_failfast` 总数升至 `97`，联合验证总用例数为 `103`。
- **新增 / 更新测试**：新增 `test_motion_joint_loss_applicator_cluster_forward_stats_contract`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py && python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '5560,5576p' train/models.py; sed -n '1920,1995p' tests/train/test_train_models_failfast.py) | if rg ...`.
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `103` 个用例通过（`tests.train.test_train_models_failfast=97`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数 broad=`2` / exact=`2` / as_exc=`0`，touched hunks 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：B4 当前仍与 direct-pose payload helpers 物理相邻；这是刻意保留的单文件边界，后续应由 Phase C.loss 单独收口 direct-pose loss cluster，而不是在 B4 中混合做语义重排。
- **下一轮建议动作**：Phase B 下一步建议进入 B5 rot6d cluster；Category C 继续留到 compiler/export 专项。

### 2026-04-23 — Phase B.B4 applicators cluster completion

- **本轮目标**：完成 Phase B.B4，把 applicators cluster 明确拆成 motion applicators / direct-pose payload+applicator / auxiliary applicators 三个 in-file 子区，并补一条默认 direct-pose stats regression；不做跨文件拆分，不改 `EventMotionModel.forward` 主逻辑，不处理 Category C。
- **实际完成项**：`train/models.py` 的 applicators 区现在显式分成 `Motion applicators.`、`Direct-pose payload helpers stay colocated until Phase C.loss.`、`Direct-pose applicator.`、`Auxiliary applicators.` 四个连续子区；未移动 helper 出文件，也未改任何 applicator 数值逻辑。新增回归锁定：`pred_motion` 非 dict 时，`_apply_direct_pose_component` 仍写入完整 direct-pose default stats 集合且总 loss 为 `0.0`。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `2 -> 2`；精确 `except Exception:` 维持 `2 -> 2`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 仍仅为 Category C `2`（C01–C02）。
- **A3 / A4 推进**：B4 已完成 in-file cluster；A4 新增 `1` 个 B4 completion regression test，`tests.train.test_train_models_failfast` 总数升至 `98`，联合验证总用例数为 `104`。
- **新增 / 更新测试**：新增 `test_motion_joint_loss_applicator_cluster_direct_pose_defaults_when_pred_not_dict`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py && python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '5564,5574p' train/models.py; sed -n '1924,2030p' tests/train/test_train_models_failfast.py) | if rg ...`.
- **验证结果**：通过；联合 `unittest` 共 `104` 个用例通过（`tests.train.test_train_models_failfast=98`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数 broad=`2` / exact=`2` / as_exc=`0`，touched hunks 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：B4 已完成 in-file cluster 目标，但 direct-pose payload helpers 仍保留在同一文件内；后续如继续收口，必须在 Phase C.loss 中单独处理 payload typing / stats contract，避免把结构调整和语义调整绑定。
- **下一轮建议动作**：进入 Phase B.B5 rot6d cluster，按 `_maybe_get_rot6d` → `_denorm_rot6d_flat` → `_extract_rot6d_flat` → `_extract_rot6d_mats` / `_rot6d_matrices` → `compute_rot6d_ortho_loss` → `compute_rot6d_geo_loss` → `compute_rot6d_log_loss` 的顺序做 in-file cluster 化；Category C 继续留到 compiler/export 专项。

### 2026-04-23 — Phase B.B5 rot6d cluster completion

- **本轮目标**：完成 Phase B.B5，在 `train/models.py` 内建立 rot6d helper cluster，按 `_maybe_get_rot6d` → `_denorm_rot6d_flat` → `_extract_rot6d_flat` → `_extract_rot6d_mats` / `_rot6d_matrices` → `compute_rot6d_ortho_loss` → `compute_rot6d_geo_loss` → `compute_rot6d_log_loss` 的顺序整理；不做跨文件拆分，不改数学定义，不处理 Category C。
- **实际完成项**：`train/models.py` 已在 rot6d helpers 起点新增 `# === future: train/loss_rot6d.py ===` 与 `Rot6D slice / denorm / matrix helpers.` / `Rot6D objective helpers.` 子区注释；rot6d 相关方法已按 roadmap 顺序重排为 `_maybe_get_rot6d` → `_denorm_rot6d_flat` → `_extract_rot6d_flat` → `_extract_rot6d_mats` → `_rot6d_matrices` → `compute_rot6d_ortho_loss` → `compute_rot6d_geo_loss` → `compute_rot6d_log_loss`。函数体与数值语义未改。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `2 -> 2`；精确 `except Exception:` 维持 `2 -> 2`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 仍仅为 Category C `2`（C01–C02）。
- **A3 / A4 推进**：B5 已完成 in-file cluster；A4 新增 `2` 个 B5 regression tests，`tests.train.test_train_models_failfast` 总数升至 `100`，联合验证总用例数为 `106`。
- **新增 / 更新测试**：新增 `test_motion_joint_loss_rot6d_cluster_extract_and_matrix_helpers`、`test_motion_joint_loss_rot6d_cluster_objective_helpers`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py && python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '5338,5518p' train/models.py; sed -n '2020,2088p' tests/train/test_train_models_failfast.py) | if rg ...`.
- **验证结果**：通过；联合 `unittest` 共 `106` 个用例通过（`tests.train.test_train_models_failfast=100`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数 broad=`2` / exact=`2` / as_exc=`0`，touched hunks 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：Phase B 已完成 B1/B3/B4/B5，但 `_warn_once` 仍按 roadmap 保持冻结；rot6d cluster 仍留在单文件内，未做 `_Rot6DLossHelpers` 类化或 `_joint_weight_vector` 外部化，这些都明确留到 Phase E。
- **下一轮建议动作**：若继续单文件收口，下一步应进入 Phase C.loss，单独整理 direct-pose loss cluster 与 payload typing；Category C compiler/export probes 继续留到专项处理。

### 2026-04-23 — Phase C.loss.1 direct-pose loss cluster prep

- **本轮目标**：进入 Phase C.loss，但只做 C.loss.1 的 single-file cluster 收口准备；范围限定在 `MotionJointLoss` 的 direct-pose loss helpers，不处理 C01/C02，不做跨文件拆分，不改 `EventMotionModel.forward` 主逻辑，不改 loss 数值语义。
- **实际完成项**：`train/models.py` 已在 direct-pose loss 区新增 `# === future: train/loss/direct_pose.py ===` 分区注释，并补上 `Direct-pose default stats and pair normalization.`、`Direct-pose group payload builders.`、`Direct-pose payload assembly.`、`Direct-pose component applicator.` 子区注释；当时的 direct-pose cluster 继续保持连续 in-file 排布。后续在 `2026-04-25` 已于 `train/losses.py` 删除 `_direct_pose_extra_defaults` / `_compute_direct_pose_group_norm_payload` 等机械壳，group-norm 公开入口收敛为 `_compute_direct_pose_group_norm_shared(...) -> _compute_direct_pose_group_norm_result(...)`。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `2 -> 2`；精确 `except Exception:` 维持 `2 -> 2`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 仍仅为 Category C `2`（C01–C02）。
- **新增 / 更新测试**：`tests.train.test_train_models_failfast` 新增 `4` 个 C.loss.1 regression tests：`test_motion_joint_loss_direct_pose_default_stats_key_contract`、`test_motion_joint_loss_prepare_direct_pose_pair_normalizes_2d_3d_inputs`、`test_motion_joint_loss_group_base_payload_arm_else_balance_contract`、`test_motion_joint_loss_group_norm_shared_can_skip_ema_update`；`tests.train.test_train_models_failfast` 总数增至 `104`，联合 `tests.train.test_event_motion_model_refactor_phase_d` 总数为 `110`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '5760,6168p' train/models.py; sed -n '2000,2170p' tests/train/test_train_models_failfast.py) | rg -n \"\\.get\\(.*\\.get\\(|warnings?\\.warn\\(|state_dict\\[.*\\]\\s*=\\s*state_dict\\.pop\\(|# .* compat|# .* legacy\"`。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `110` 个用例通过（`tests.train.test_train_models_failfast=104`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数得到 broad=`2` / exact=`2` / as_exc=`0`，本轮 touched hunks 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：本轮只完成 C.loss.1 的 cluster 边界和 contract smoke，尚未进入 C.loss.2 的 payload dataclass typing，也未冻结 C.loss.3 stats key contract 表；Category C 的 C01/C02 compiler / export probe broad handler 仍保留。
- **下一轮建议动作**：优先进入 C.loss.2，把 `_compute_direct_pose_payload` 相关 payload 输入/输出类型化为 dataclass，并显式化 EMA / group-norm state 访问；保持现有 stats key、默认超参和 loss 数值语义不变。

### 2026-04-23 — Phase C.loss.2 direct-pose payload typing

- **本轮目标**：完成 C.loss.2；把 direct-pose payload 层类型化成 dataclass，并把 EMA / group-norm state 访问显式化；保持 `train/models.py` 单文件内收口，不做跨文件拆分，不改默认超参、stats key、checkpoint contract、loss 数值语义，也不处理 Category C。
- **实际完成项**：`train/models.py` 新增 `_DirectPosePair`、`_DirectPoseGroupBaseRequest`、`_DirectPoseGroupNormRequest`、`_DirectPoseGroupNormResult`、`_DirectPosePayloadRequest`、`_DirectPosePayloadResult` 六个 internal dataclass，用于 direct-pose pair normalize、group-base payload、group-norm payload、direct-pose payload 的 typed request/result 表达；当时新增了 `_compute_direct_pose_group_norm_from_request(...)`、`_compute_direct_pose_payload_from_request(...)` 两个 typed 内部入口，以及 `_direct_pose_group_norm_ema_snapshot(...)` / `_direct_pose_group_norm_ema_value(...)` / `_store_direct_pose_group_norm_ema(...)` 三个 EMA state helper。后续在 `2026-04-25` 已进一步简化：删除 `_compute_direct_pose_group_norm_from_request(...)`、`_direct_pose_group_norm_ema_snapshot(...)`、`_store_direct_pose_group_norm_ema(...)`，保留 `_compute_direct_pose_group_norm_shared(...)` 作为 tuple 兼容 seam、`_compute_direct_pose_group_norm_result(...)` 作为核心实现，`train/posttrain.py` 调用面保持不变。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `2 -> 2`；精确 `except Exception:` 维持 `2 -> 2`；`except Exception as exc` 维持 `0 -> 0`；剩余 broad handler 仍仅为 Category C `2`（C01–C02）。
- **新增 / 更新测试**：`tests.train.test_train_models_failfast` 新增 `2` 个 C.loss.2 regression tests：`test_motion_joint_loss_direct_pose_payload_request_result_types`、`test_motion_joint_loss_group_norm_request_result_types`；并把 `_prepare_direct_pose_pair` 回归改为验证 dataclass field 访问。`tests.train.test_train_models_failfast` 总数增至 `106`，联合 `tests.train.test_event_motion_model_refactor_phase_d` 总数为 `112`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '4488,4568p' train/models.py; sed -n '4568,4658p' train/models.py; sed -n '5760,6175p' train/models.py; sed -n '2030,2325p' tests/train/test_train_models_failfast.py) | rg -n \"\\.get\\(.*\\.get\\(|warnings?\\.warn\\(|state_dict\\[.*\\]\\s*=\\s*state_dict\\.pop\\(|# .* compat|# .* legacy\"`。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `112` 个用例通过（`tests.train.test_train_models_failfast=106`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数得到 broad=`2` / exact=`2` / as_exc=`0`，本轮 touched hunks 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：本轮已完成 C.loss.2，但 C.loss.3 的 stats key contract 表仍未单独冻结；typed dataclass 当前仍为 `train/models.py` 内部实现细节，后续跨文件迁移时要保持 public tuple/dict helper 契约不漂移；Category C 的 C01/C02 compiler / export probe broad handler 仍保留。
- **下一轮建议动作**：进入 C.loss.3，产出 direct-pose stats key contract 表，并用最小 regression 锁定 `_apply_direct_pose_component` 最终写入集合；Category C 继续留到 compiler/export 专项。

### 2026-04-23 — Phase C.loss.3 stats contract + Category C probe cleanup

- **本轮目标**：一起完成 C.loss.3 与 Category C；冻结 `_apply_direct_pose_component` stats 写入 key contract，并把 `torch._dynamo.is_compiling()` / `torch.onnx.is_in_onnx_export()` 两个 broad compiler/export probe 收窄为显式 helper；保持单文件内完成，不改 loss 数值语义、默认超参、checkpoint contract，不引入 warning-only debt。
- **实际完成项**：`train/models.py` 新增 `_DIRECT_POSE_DEFAULT_STAT_KEYS`、`_DIRECT_POSE_COMPONENT_STAT_KEYS` 两个 module-level contract tuple；`_apply_direct_pose_component` 的 default-path 与 group-norm path 现均有 exact stats-key regression 锁定。后续在 `2026-04-25` 已删除 `_direct_pose_default_stat_keys()` / `_direct_pose_component_stat_keys()` getter 薄壳，测试改为直接绑定模块常量。`EventMotionModel.forward(...)` 中的 `_skip_guard` probe 已改为调用 `_torch_dynamo_is_compiling_safe()` 与 `_torch_onnx_is_in_export_safe()`，仅对 `AttributeError` / `RuntimeError` 做显式 fallback，broad `except Exception` 已完全清零。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler `2 -> 0`；精确 `except Exception:` `2 -> 0`；`except Exception as exc` 维持 `0 -> 0`；Category C `2 -> 0`。
- **新增 / 更新测试**：`tests.train.test_train_models_failfast` 新增 `4` 个 regression tests：`test_motion_joint_loss_direct_pose_component_stats_contract_default_path`、`test_motion_joint_loss_direct_pose_component_stats_contract_group_norm_path`、`test_torch_dynamo_probe_safe_handles_missing_and_runtime_failure`、`test_torch_onnx_probe_safe_handles_missing_and_runtime_failure`；并更新 `test_motion_joint_loss_direct_pose_default_stats_key_contract` 使其显式绑定 module-level contract tuple。`tests.train.test_train_models_failfast` 总数增至 `110`，联合 `tests.train.test_event_motion_model_refactor_phase_d` 总数为 `116`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '56,110p' train/models.py; sed -n '3498,3522p' train/models.py; sed -n '5820,6305p' train/models.py; sed -n '1960,2335p' tests/train/test_train_models_failfast.py) | rg -n \"\\.get\\(.*\\.get\\(|warnings?\\.warn\\(|state_dict\\[.*\\]\\s*=\\s*state_dict\\.pop\\(|# .* compat|# .* legacy\"`。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `116` 个用例通过（`tests.train.test_train_models_failfast=110`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`，本轮 touched hunks 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：Phase C.loss 已完成 1/2/3，但 stats contract table 当前仍留在 roadmap 文档与 module-level tuple；若将来跨文件迁移到 `train/loss/direct_pose.py`，必须原样迁移 `_DIRECT_POSE_DEFAULT_STAT_KEYS` / `_DIRECT_POSE_COMPONENT_STAT_KEYS` 并保持现有 public helper 返回契约。`_warn_once` 仍按 roadmap 留在 Phase E。
- **下一轮建议动作**：Phase C.loss 与 Category C 已清零；若继续推进，最推荐进入 Phase D，开始 `MotionJointLoss` / `EventMotionModel.forward` 的编排层收口，或单独开启 Phase E 的 `_warn_once` / 跨文件迁移准备。

### 2026-04-23 — Phase D.D2 batch 1 MotionJointLoss orchestration shell

- **本轮目标**：进入 Phase D，但只做 D2 的第一批：在 `train/models.py` 内收口 `MotionJointLoss.forward(...)` 周边编排层；范围限定为 `forward(...)`、`_prepare_forward_inputs(...)`、`_init_loss_group_tracker(...)`、`_accumulate_loss_contrib(...)`、`_loss_group_stats(...)`、`_submit_component_loss(...)`、`_prepare_aux_supervision_pair(...)` 一组，不改 applicator 内部数值语义，不碰 `EventMotionModel.forward(...)` 主逻辑，不做跨文件迁移。
- **实际完成项**：`train/models.py` 新增 `# === future: train/loss/orchestration.py ===` orchestration 分区；将 `MotionJointLoss` 的 forward-shell 重新整理为 `Forward input prep / base loss.`、`Loss tracker / stats finalize.`、`Applicator dispatch shell.` 三个连续子区。`_prepare_forward_inputs(...)` / `_run_forward_base(...)`、`_init_loss_group_tracker(...)` / `_accumulate_loss_contrib(...)` / `_loss_group_stats(...)` / `_prepare_aux_supervision_pair(...)` / `_submit_component_loss(...)` 现已物理聚拢到 class 尾部 orchestration 区。后续在 `2026-04-25` 已把 `_coerce_forward_base_output(...)` inline 回 `_run_forward_base(...)`，但保留 `_run_forward_base(...)` / `_init_loss_group_tracker(...)` / `_loss_group_stats(...)` 作为 orchestration seams。新增薄包装 `_dispatch_forward_components(...)`，把 dispatch 顺序显式收口为 `motion -> direct_pose -> aux`，`forward(...)` 变成“init tracker → prepare inputs → run base → dispatch applicators → finalize stats”的编排壳。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_train_models_failfast` 新增 `4` 个 Phase D.D2 orchestration regression tests：`test_motion_joint_loss_prepare_forward_inputs_preserves_dict_and_tensor_contract`、`test_motion_joint_loss_prepare_aux_supervision_pair_aligns_steps_dtype_and_device`、`test_motion_joint_loss_submit_component_loss_tracks_group_stats_contract`、`test_motion_joint_loss_forward_dispatch_order_regression`。`tests.train.test_train_models_failfast` 总数增至 `114`，联合 `tests.train.test_event_motion_model_refactor_phase_d` 总数为 `120`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '5635,6685p' train/models.py; sed -n '2240,2490p' tests/train/test_train_models_failfast.py) | rg -n \"\\.get\\(.*\\.get\\(|warnings?\\.warn\\(|state_dict\\[.*\\]\\s*=\\s*state_dict\\.pop\\(|# .* compat|# .* legacy\"`。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `120` 个用例通过（`tests.train.test_train_models_failfast=114`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`，本轮 touched ranges 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：本轮只完成 D2 的第一批 orchestration 壳收口，`MotionJointLoss` 的 applicators cluster 本体未再细拆；`EventMotionModel.forward(...)` 的 D1 尚未开始；Phase D roadmap 验收里的 stage6 deterministic smoke / forward output snapshot 本 batch 尚未补跑。
- **下一优先级热点**：继续做 D2 后续批次时，建议把 `MotionJointLoss` 剩余“config/init / skeleton state / payload builders / component apply / stats finalize”阅读路径再压平一轮，但继续避免改 applicator 数值语义；或者转入 D1，只收 `EventMotionModel.forward(...)` 的输入准备 / dispatch / finalize 外壳。

### 2026-04-23 — Phase D.D2 batch 2 MotionJointLoss init/skeleton read path

- **本轮目标**：继续 Phase D.D2，但只做第二批：压平 `MotionJointLoss.__init__(...)` 与 skeleton/cache bootstrap 的单文件阅读路径；不改默认超参、checkpoint/output key、loss 数值语义，不碰 `EventMotionModel.forward(...)`，不做跨文件拆分。
- **实际完成项**：`train/models.py` 在 `MotionJointLoss` 类头新增 `# === future: train/loss/init.py ===`，把 constructor 读法分成 `Fail-fast retired-key boundary`、`Local scalar normalization`、`Core loss weights / direct-pose config`、`Layout / rot6d contract`、`Skeleton / cache bootstrap`、`Orchestration tracker defaults`。同时把 `layout.get('slices')` 同行双调用拆为 `slices_layout`，避免 removal-policy grep 误中 nested fallback 形态；skeleton cluster 注释从 `State updates / cache invalidation.` 收窄为 `Skeleton state / cache invalidation.`，与 Phase B.B3 边界一致。未改 `_warn_once` 签名和实现，未新增 fallback / warning-only debt。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：新增 `test_motion_joint_loss_skeleton_cluster_init_state_contract`，锁定 constructor 的 core weights / skeleton state / cache containers / loss-group alias 初始 contract；新增 `test_motion_joint_loss_finalize_forward_outputs_adds_loss_group_stats`，锁定 stats finalize 合并 loss-group 行为。`tests.train.test_train_models_failfast` 总数增至 `116`，联合 `tests.train.test_event_motion_model_refactor_phase_d` 总数为 `122`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py && python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '4677,4955p' train/models.py; sed -n '4979,5045p' train/models.py; sed -n '1810,1888p' tests/train/test_train_models_failfast.py; sed -n '2300,2452p' tests/train/test_train_models_failfast.py) | rg -n <removal-policy §6 regex>`。
- **验证结果**：通过；联合 `unittest` 共 `122` 个用例通过（`tests.train.test_train_models_failfast=116`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`，本轮 touched ranges 的 removal-policy 反模式 grep 零命中。
- **阻塞项 / 风险**：D2 batch 2 仍是单文件内阅读路径整理，没有抽 helper class，也没有改 skeleton/cache semantics；Phase D roadmap 验收里的 stage6 deterministic smoke / forward output snapshot 本 batch 仍未补跑。
- **下一优先级热点**：继续 D2 时建议进入 payload/applicator/read-path 最后一轮，把 `MotionJointLoss` 的 direct-pose payload builders 与 component applicators 的物理边界再对齐；如果改动扩大到 forward 主路径，应先补 forward snapshot。

### 2026-04-23 — Phase D.D2 batch 3 payload/applicator boundary alignment

- **本轮目标**：继续 Phase D.D2，但只做第三批：对齐 `MotionJointLoss` 的 direct-pose payload builders 与 component applicators 的物理边界；保持单文件内完成，不改 loss 数值语义、默认超参、stats key、checkpoint/output key，不碰 `EventMotionModel.forward(...)`。
- **实际完成项**：`train/models.py` 中 direct-pose cluster 现在按 `Stats contract / pair normalization` → `Group base payload` → `Group norm public wrapper / EMA helpers` → `Group norm typed implementation` → `Direct-pose payload public wrapper / typed assembly` → `Direct-pose applicator` 的顺序阅读；`_compute_direct_pose_group_base_payload(...)` 物理前移到 group-norm wrapper 之前，便于先看 base payload 再看 group-norm overlay。component applicators 增补 `Motion component dispatch`、`Contact-plan applicator`、`Event-clock applicators`、`Contact-measurement applicator`、`Omega regularization applicator`、`Auxiliary component dispatch` 子区注释；未新增 fallback，未改 `_warn_once`，未调整 applicator 内部公式。
- **修改文件列表**：`train/models.py`；`tests/train/test_train_models_failfast.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：新增 `test_motion_joint_loss_motion_component_dispatch_order_regression`，锁定 `_apply_motion_components(...)` 内部 `rot_ortho -> rot_local -> root_velocity` 调度顺序；新增 `test_motion_joint_loss_aux_component_dispatch_order_regression`，锁定 `_apply_aux_components(...)` 内部 `contact_plan -> event_clock -> contact_meas -> omega_l2` 调度顺序。`tests.train.test_train_models_failfast` 总数增至 `118`，联合 `tests.train.test_event_motion_model_refactor_phase_d` 总数为 `124`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`python3 - <<'PY' ... unittest.defaultTestLoader.loadTestsFromName(...) ...`；`(sed -n '5830,6555p' train/models.py; sed -n '2380,2535p' tests/train/test_train_models_failfast.py) | rg -n <removal-policy §6 regex>`；`rg -n "def _compute_direct_pose_group_base_payload|def _compute_direct_pose_group_norm_payload|def _compute_direct_pose_payload\\(|def _apply_motion_components|def _apply_aux_components" train/models.py`。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `124` 个用例通过（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=6`），AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`，本轮 touched ranges 的 removal-policy 反模式 grep 零命中，direct-pose group base payload 定义确认单一且位于 group-norm wrapper 前。
- **阻塞项 / 风险**：D2 batch 3 仍是物理边界与注释层整理，未做跨文件迁移，也未补跑 stage6 deterministic smoke / forward output snapshot；后续若开始 D1 或移动 `EventMotionModel.forward(...)`，应先补 snapshot。
- **下一优先级热点**：D2 已完成 orchestration shell、init/skeleton read path、payload/applicator boundary 三批；下一轮最推荐转入 Phase D.D1 的 `EventMotionModel.forward(...)` 外壳收口，但第一步只做 forward input prep / finalize 的薄壳和 snapshot regression。

### 2026-04-23 — Phase D.D1 batch 1 EventMotionModel forward shell + snapshot smoke

- **本轮目标**：进入 Phase D.D1 第一批，只整理 `EventMotionModel.forward(...)` 外围编排壳：先建立 deterministic forward snapshot baseline，再做 input prep / runtime-control prep 与 final output assembly 的最小薄包装；同时补齐 Phase D 的 stage6 deterministic smoke 可执行验证路径与 forward output snapshot / dispatch smoke regression。
- **实际完成项**：改动前用 `tests.train.test_event_motion_model_refactor_phase_d` 的 `_build_model(...)` / `_make_io(...)` 固定 seed 生成 forward snapshot；改动后 snapshot key set、shape、finite、sum/mean/L2 checksum 保持一致。`train/models.py` 新增 `_EventMotionForwardInputPrep` dataclass、`_prepare_forward_inputs(...)`、`_forward_input_shape_error(...)`，把 `state/cond/contacts/angvel/pose_history/plan_z/phase_z/phase_event_age` 的输入归一化与 `_eval_runtime_controls_bundle()` 读取收成薄壳；新增 `_build_forward_base_result(...)` 与 `_write_forward_direct_pose_outputs(...)`，只收口 `out/delta/attn/h_final` base result 与 direct-pose final key 写入。未进入 contact-plan / event-clock / direct-pose 内部算法重写，未改 loss 数值语义、模型输出 key、checkpoint contract 或默认超参。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 新增 `test_forward_output_snapshot_deterministic_regression` 与 `test_forward_shell_dispatch_smoke_regression`。前者锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum；后者锁定 D1 shell 调度会经过 `_prepare_forward_inputs(...)` → `_build_forward_base_result(...)` → `_write_forward_direct_pose_outputs(...)`。`tests.train.test_event_motion_model_refactor_phase_d` 总数增至 `8`，联合 `tests.train.test_train_models_failfast` 总数为 `126`。
- **stage6 deterministic smoke / snapshot 验证方式**：复用 roadmap §5.3 现有路径：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_phase_d_d1_stage6_smoke_20260423 --run_name train_models_phase_d_d1_forward_shell_smoke_20260423 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`；forward snapshot 用固定 `torch.manual_seed(12345)` 构造输入、固定 `torch.manual_seed(999)` 执行 eval forward，对比 key/shape/finite/sum/mean/L2。
- **运行过的命令**：pre-change inline snapshot Python；post-change inline snapshot Python；`python3 -m py_compile train/models.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`；stage6 deterministic smoke 命令；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；AST broad-handler inline Python；touched ranges removal-policy §6 grep。
- **验证结果**：focused Phase D test 通过（`8` tests）；stage6 smoke 通过，`ok_steps=5 skipped=0` 并保存 `debug_output/_tmp_train_models_phase_d_d1_stage6_smoke_20260423/ckpt_last_train_models_phase_d_d1_forward_shell_smoke_20260423.pth`；最终联合验证通过后记录为 `tests.train.test_train_models_failfast=118`、`tests.train.test_event_motion_model_refactor_phase_d=8`、total=`126`，AST broad=`0` / exact=`0` / as_exc=`0`，touched ranges removal-policy grep 零命中。
- **阻塞项 / 风险**：本 batch 只完成 D1 外壳第一层；`EventMotionModel.forward(...)` 内部的 contact-plan / event-clock / direct-pose 主体仍是原地大块逻辑。stage6 smoke 输出中的 checkpoint override warning 来自 `train.posttrain` 既有加载路径，本轮未新增 warning-only debt。
- **下一轮建议动作**：继续 Phase D.D1 第二批，但仍限制在 `EventMotionModel.forward(...)` 外壳：优先为 contact-plan/event-clock 大块加明确子区边界或薄 dispatch helper，并用同一 forward snapshot regression 证明 output key / checksum 不漂移；不要进入 Phase E 跨文件迁移。

### 2026-04-24 — Phase D.D1 batch 2 contact-plan / Event-Clock shell boundary

- **本轮目标**：继续 Phase D.D1，只收 `EventMotionModel.forward(...)` 中 contact-plan / Event-Clock 的外壳边界；保持 snapshot 先行，不进入 contact-plan GRU loop、event-clock gate/corrector、direct-pose readout 或 loss 数值逻辑重写。
- **实际完成项**：改动前复用 `test_event_motion_model_refactor_phase_d._make_forward_snapshot_output()` 记录 forward snapshot；改动后 key set、shape、finite、sum/mean/L2 checksum 保持一致。`train/models.py` 新增 `_ContactClockForwardDefaults`、`_ContactPlanForwardFinal` 两个 internal dataclass，以及 `_init_contact_clock_forward_defaults(...)` / `_finalize_contact_plan_outputs(...)` 两个薄壳：前者集中初始化 Event-Clock 预计算信号与 contact-plan 输出默认值，后者只集中原有 `plan_probs` / `plan_logits` / phase direct seq / side cue seq / debug logits / inject feature 的最终 stack 与写回。contact-plan / Event-Clock 主循环公式与异常语义保持原样。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：更新 `test_forward_shell_dispatch_smoke_regression`，将 D1 shell 调度锁定为 `_prepare_forward_inputs(...)` → `_init_contact_clock_forward_defaults(...)` → `_finalize_contact_plan_outputs(...)` → `_build_forward_base_result(...)` → `_write_forward_direct_pose_outputs(...)`；`test_forward_output_snapshot_deterministic_regression` 继续锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum。测试数量不变：`tests.train.test_event_motion_model_refactor_phase_d=8`，联合 total=`126`。
- **stage6 deterministic smoke / snapshot 验证方式**：stage6 继续复用 roadmap §5.3 现有 executable path：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_phase_d_d1_contact_clock_smoke_20260424 --run_name train_models_phase_d_d1_contact_clock_smoke_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`；forward snapshot 继续复用 Phase D test builder / fixture，固定 `torch.manual_seed(12345)` 造输入与 `torch.manual_seed(999)` 执行 eval forward。
- **运行过的命令**：pre-change inline snapshot Python；post-change inline snapshot Python；`python3 -m py_compile train/models.py`；`python3 -m py_compile train/models.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`；stage6 deterministic smoke 命令；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；AST broad-handler inline Python；touched ranges removal-policy §6 grep。
- **验证结果**：focused Phase D test 通过（`8` tests）；stage6 smoke 通过，`ok_steps=5 skipped=0` 并保存 `debug_output/_tmp_train_models_phase_d_d1_contact_clock_smoke_20260424/ckpt_last_train_models_phase_d_d1_contact_clock_smoke_20260424.pth`；最终联合验证通过后记录为 `tests.train.test_train_models_failfast=118`、`tests.train.test_event_motion_model_refactor_phase_d=8`、total=`126`，AST broad=`0` / exact=`0` / as_exc=`0`，touched ranges removal-policy grep 零命中。
- **阻塞项 / 风险**：本 batch 仍是单文件外壳收口；contact-plan init observed branch、Event-Clock on/off loop、direct-pose readout 与 leg residual 大块仍在 `forward(...)` 内部。stage6 smoke 输出中的 checkpoint override warning 仍来自既有 `train.posttrain` 加载路径，本轮未新增 warning-only debt。
- **下一轮建议动作**：继续 Phase D.D1 第三批时，优先收 `EventMotionModel.forward(...)` 的 direct-pose readout / leg residual 外壳边界，但必须保持 direct-pose stats contract key 集不变，并继续用同一 snapshot regression + stage6 smoke 对照。

### 2026-04-24 — Phase D.D1 batch 3 direct-pose shell completion

- **本轮目标**：收完 Phase D.D1 的 `EventMotionModel.forward(...)` 外壳边界；只补 direct-pose entry / runtime-control shell 与 dispatch smoke，不迁移文件、不改 direct-pose readout / leg residual 内部算法、不改 stats key / output key / checkpoint contract / 默认超参。
- **实际完成项**：改动前复用 `_make_forward_snapshot_output()` 记录 deterministic forward snapshot；改动后 key set、shape、finite、sum/mean/L2 checksum 保持一致。`train/models.py` 新增 `_DirectPoseForwardRuntime` dataclass、`_should_run_direct_pose_forward(...)`、`_init_direct_pose_forward_runtime(...)`，把 direct-pose 是否执行、plan/meas override、leg side ablation mode、cross-leg ablation mode 与 direct-leg optional output defaults 收成 direct-pose shell；`forward(...)` 现在外层阅读顺序为 input prep → contact/Event-Clock defaults/finalize → base result finalize → direct-pose entry/runtime shell → direct output finalize → lambda/so3/period tail。Direct-pose 主体公式与异常上下文保持原样。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **统计结果**：broad `Exception` handler 维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：更新 `test_forward_shell_dispatch_smoke_regression`，将 D1 shell 调度锁定为 `_prepare_forward_inputs(...)` → `_init_contact_clock_forward_defaults(...)` → `_finalize_contact_plan_outputs(...)` → `_build_forward_base_result(...)` → `_should_run_direct_pose_forward(...)` → `_init_direct_pose_forward_runtime(...)` → `_write_forward_direct_pose_outputs(...)`；`test_forward_output_snapshot_deterministic_regression` 继续锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum。测试数量不变：`tests.train.test_event_motion_model_refactor_phase_d=8`，联合 total=`126`。
- **stage6 deterministic smoke / snapshot 验证方式**：stage6 继续复用 roadmap §5.3 executable path：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_phase_d_d1_complete_smoke_20260424 --run_name train_models_phase_d_d1_complete_smoke_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`；forward snapshot 继续复用 Phase D test builder / fixture，固定 `torch.manual_seed(12345)` 造输入与 `torch.manual_seed(999)` 执行 eval forward。
- **运行过的命令**：pre-change inline snapshot Python；post-change inline snapshot Python；`python3 -m py_compile train/models.py`；`python3 -m py_compile train/models.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`；stage6 deterministic smoke 命令；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；AST broad-handler inline Python；touched ranges removal-policy §6 grep。
- **验证结果**：focused Phase D test 通过（`8` tests）；stage6 smoke 通过，`ok_steps=5 skipped=0` 并保存 `debug_output/_tmp_train_models_phase_d_d1_complete_smoke_20260424/ckpt_last_train_models_phase_d_d1_complete_smoke_20260424.pth`；最终联合验证通过后记录为 `tests.train.test_train_models_failfast=118`、`tests.train.test_event_motion_model_refactor_phase_d=8`、total=`126`，AST broad=`0` / exact=`0` / as_exc=`0`，touched ranges removal-policy grep 零命中。
- **阻塞项 / 风险**：D1 已形成四层外壳，但 direct-pose readout / side-routed leg residual 主体仍在 `forward(...)` 内作为算法块保留，避免本轮改数值语义；stage6 smoke 输出中的 checkpoint override warning 仍来自既有 `train.posttrain` 加载路径，本轮未新增 warning-only debt。
- **下一轮建议动作**：Phase D.D1 可视为完成；下一步建议进入 Phase D.D3 / Phase D 收尾，补一轮 `EventMotionModel.forward(...)` line-count/structure inventory 与 final Phase D snapshot/state_dict 指纹；Phase E 跨文件迁移仍建议等 D 收尾验证稳定后再开。

### 2026-04-24 — Phase D.D3 final inventory / fingerprint closeout

- **本轮目标**：完成 Phase D 收尾，只做 `train/models.py` 结构 inventory、`EventMotionModel.forward(...)` / `MotionJointLoss` Phase D 后结构对照、final deterministic forward snapshot 复核、final state_dict key-set / checksum fingerprint 验证，以及 roadmap / inventory 文档回填；不进入 Phase E，不改核心算法 / loss 数值语义 / 输出 key / checkpoint contract / 默认超参。
- **实际完成项**：本轮未改 `train/models.py`；只新增最小 state_dict 指纹回归并回填文档。结构 inventory 结果：`train/models.py` 当前 `LOC=6963`；`EventMotionModel.forward(...)` 位于 `train/models.py:3033`–`train/models.py:4820`，共 `1788` 行；`MotionJointLoss.forward(...)` 位于 `train/models.py:6947`–`train/models.py:6963`，共 `17` 行。当前 in-file shell / cluster 为：`EventMotionModel` input prep helper `train/models.py:2382`–`train/models.py:2494` + forward dispatch `train/models.py:3047`–`train/models.py:3247`；contact-plan / Event-Clock shell `train/models.py:2551`–`train/models.py:2655` + runtime body `train/models.py:3249`–`train/models.py:3933`；final output assembly `train/models.py:2498`–`train/models.py:2547` + base-result/direct-write dispatch `train/models.py:3935`–`train/models.py:3996`；direct-pose shell `train/models.py:2659`–`train/models.py:3032` + runtime body `train/models.py:3997`–`train/models.py:4698`；`MotionJointLoss` init cluster `train/models.py:4948`–`train/models.py:5240`；skeleton / weight-cache cluster `train/models.py:5241`–`train/models.py:5762`；payload cluster `train/models.py:6129`–`train/models.py:6595`；applicator cluster `train/models.py:5934`–`train/models.py:6128` 与 `train/models.py:6596`–`train/models.py:6814`；orchestration shell `train/models.py:6815`–`train/models.py:6963`。Phase D.D1 完成项保持为三批：input prep / runtime-control shell、contact-plan / Event-Clock shell boundary、direct-pose shell completion；Phase D.D2 完成项保持为三批：orchestration shell、init/skeleton read path、payload/applicator boundary alignment。明确留给 Phase E / 后续专项的事项：跨文件迁移（`train/loss/skeleton_weights.py`、`train/loss_rot6d.py`、`train/loss/direct_pose.py`、`train/loss/components.py`、`train/loss/tracker.py`、`train/models/direct_pose/*`）、`_warn_once` 重构、以及 `EventMotionModel.forward(...)` / direct-pose 主体的进一步物理拆分。
- **修改文件列表**：`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 新增 `test_state_dict_fingerprint_repeated_construction_regression`，复用现有 `_build_model(...)` builder，并新增 `_state_dict_fingerprint(...)` / `_tensor_sha256(...)` helper；该回归固定 builder seed，验证 repeated construction 下 `state_dict` 的 sorted key set、每个 tensor key 的 `shape` / `dtype` / `sha256(tensor.cpu().numpy().tobytes())` 与 aggregate fingerprint 均不漂移。`tests.train.test_event_motion_model_refactor_phase_d` 总数增至 `9`，联合 `tests.train.test_train_models_failfast` 总数为 `127`。
- **final forward snapshot 验证方式**：直接复用 `tests.train.test_event_motion_model_refactor_phase_d.test_forward_output_snapshot_deterministic_regression`；其内部继续复用 `_make_forward_snapshot_output()` / `_build_model(...)` / `_make_io(...)`，固定 `torch.manual_seed(12345)` 构造输入与 `torch.manual_seed(999)` 执行 eval forward，锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum。
- **final state_dict fingerprint 验证方式**：运行 `tests.train.test_event_motion_model_refactor_phase_d.test_state_dict_fingerprint_repeated_construction_regression`；固定同一最小 split direct-pose builder，两次构造之间插入额外 RNG 消耗，确认 repeated construction 后 `state_key_count=93`、sample aggregate fingerprint 为 `b6fc8c171a8855daca03735bba97dad40888c09edb2885822f7e7c6adfef2c80`，且 per-key `shape` / `dtype` / `sha256` 全量一致。
- **stage6 deterministic smoke / snapshot 验证方式**：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_phase_d_final_smoke_20260424 --run_name train_models_phase_d_final_smoke_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`；forward snapshot 继续复用上述 Phase D test。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 - <<'PY' ... ast.parse(train/models.py) ...`；`(sed -n '1,260p' tests/train/test_event_motion_model_refactor_phase_d.py) | rg -n <removal-policy §6 regex>`；`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_phase_d_final_smoke_20260424 --run_name train_models_phase_d_final_smoke_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`；`python3 - <<'PY' ... _state_dict_fingerprint(model.state_dict()) ...`；`python3 - <<'PY' ... ast.parse(train/models.py) -> LOC/forward spans ...`。
- **验证结果**：`py_compile` 通过；联合 `unittest` 共 `127` 个用例通过（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=9`）；AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`；touched code/test ranges removal-policy 反模式 grep 零命中；stage6 smoke 通过，`ok_steps=5 skipped=0`，输出目录 `debug_output/_tmp_train_models_phase_d_final_smoke_20260424` 并保存 `debug_output/_tmp_train_models_phase_d_final_smoke_20260424/ckpt_last_train_models_phase_d_final_smoke_20260424.pth`。
- **阻塞项 / 风险**：Phase D 已完成 shell 化与 final verification，但 `EventMotionModel.forward(...)` 仍保留 `1788` 行，direct-pose readout / leg residual 与 contact-plan / Event-Clock 主体算法块仍在单文件内；这是刻意保留的语义冻结边界，不属于本轮 scope。stage6 smoke 中出现的 checkpoint override warning 仍来自 `train.posttrain` 既有加载路径，本轮未新增 warning-only debt。
- **当前 Phase D 是否可收尾**：可以。D1 / D2 / D3 的既定目标均已完成，broad handler 维持 `0`，forward snapshot / state_dict fingerprint / stage6 deterministic smoke / removal-policy grep 均已补齐。
- **下一步建议动作（Phase E 前置条件）**：优先先冻结本轮 snapshot / fingerprint artifact 与 line inventory；随后进入 Phase E 前建议补三项前置检查：1) 按当前 inventory 再跑一次 focused `EventMotionModel.forward(...)` / `MotionJointLoss` diff review，确认迁移 write-set 边界；2) 明确 `_warn_once` 的替换策略与全调用点清单；3) 为 planned cross-file moves 预先列出 import surface / ckpt contract / test ownership，确保 Phase E 保持 purely mechanical move。

### 2026-04-24 — pre-Phase-E simplification batch 1 forward tail shell

- **本轮目标**：只做 `EventMotionModel.forward(...)` 尾段 `lambda_fusion` / `so3_delta_corrector` / `period_pred` 的 shell 收口，继续压平单文件结构，让 Phase E 更接近 mechanical move；不改核心算法 / loss 数值语义 / 输出 key / checkpoint contract / 默认超参，不进入跨文件迁移。
- **实际完成项**：`train/models.py` 新增 `_lambda_fusion_rollout_step_feature(...)`、`_write_forward_lambda_fusion_outputs(...)`、`_write_forward_so3_delta_outputs(...)`、`_write_forward_period_output(...)` 四个极薄 helper。`forward(...)` 尾段从内联 `lambda_fusion` tail / 内联 `so3` writeback / 内联 `period_pred` writeback，收口为三段连续 shell dispatch；`lambda_fusion` 的 rollout_step contract 与 contextual `RuntimeError` 保持不变，`omega_hat` / `period_pred` key 写入行为不变。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：简化前 tail 阅读路径为 direct-pose output finalize 后继续读完整 `lambda_fusion` block，再读 `so3_delta_corrector` block，再读 `period_pred` 单行写回；简化后为 `_write_forward_direct_pose_outputs(...)` → `_write_forward_lambda_fusion_outputs(...)` → `_write_forward_so3_delta_outputs(...)` → `_write_forward_period_output(...)`。当前 `forward(...)` 位于 `train/models.py:3194`–`train/models.py:4896`，共 `1703` 行；tail helper 分别位于 `train/models.py:2549`、`train/models.py:2619`、`train/models.py:2678`、`train/models.py:2702`。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：更新 `tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_shell_dispatch_smoke_regression`，新增 tail dispatch 断言，锁定 `_write_forward_lambda_fusion_outputs(...)`、`_write_forward_so3_delta_outputs(...)`、`_write_forward_period_output(...)` 的调用顺序；`test_forward_output_snapshot_deterministic_regression` 与 `test_state_dict_fingerprint_repeated_construction_regression` 直接复用。Phase D tests 维持 `9`，联合 `tests.train.test_train_models_failfast` 总数维持 `127`。
- **forward snapshot 验证方式**：复用 `tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`；固定 `torch.manual_seed(12345)` 构造输入与 `torch.manual_seed(999)` 执行 eval forward，锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum。
- **state_dict fingerprint 验证方式**：复用 `tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`；确认 repeated construction 下 `state_dict` sorted key set、per-key `shape` / `dtype` / `sha256` 与 aggregate fingerprint 不漂移。
- **stage6 deterministic smoke 验证方式**：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch1_20260424 --run_name train_models_pre_phase_e_simplify_batch1_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；snapshot 单测命令；state_dict fingerprint 单测命令；AST broad-handler inline Python；stage6 deterministic smoke 命令；touched diff removal-policy grep。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `127` 个用例通过（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=9`）；snapshot / state_dict fingerprint 单测均通过；AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`；stage6 smoke 通过，`ok_steps=5 skipped=0`，输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch1_20260424`。
- **阻塞项 / 风险**：本轮只做尾段 shell 收口，没有动 direct-pose leg gate/scale duplicated runtime block，也没有进一步拆 contact-plan / Event-Clock 主体算法。仍需避免把结构调整与 direct-pose 数值语义/统计 key 绑定在同一批次。
- **为什么此时仍适合继续单文件 simplification**：当前 `forward(...)` 尾段已具备线性 dispatch 形态，但 direct-pose leg gate/scale duplicated runtime block 仍在单文件内部、且可通过现有 snapshot/fingerprint/stage6 gate 继续机械化收口；如果现在直接进入 Phase E，会把“仍可在单文件内验证的结构压平”与“跨文件 import/write-set 迁移”混在一批，回归面会变大。
- **下一步建议动作**：继续 simplification batch 2，优先收 direct-pose leg gate / scale duplicated runtime block 的壳层与 dispatch，仍暂不进入 Phase E。

### 2026-04-24 — pre-Phase-E simplification batch 2 direct-pose leg gate/scale shell

- **本轮目标**：继续单文件 simplification，只收 `EventMotionModel.forward(...)` 中 side-routed / non-side 共享的 direct-pose leg learned gate / scale gate runtime 壳层；不改 direct-pose 数值语义、输出 key、checkpoint contract、默认超参，不进入跨文件迁移。
- **实际完成项**：`train/models.py` 新增 `_DirectPoseLegGateOutputs` dataclass 与 `_apply_direct_pose_leg_gate_outputs(...)` helper。`forward(...)` 里原先 duplicated 的两套 gate/scale apply 逻辑（shared head 与 non-shared head）都改成“分支内准备 `omega_leg` + `head_inputs` + 可选 `side_positions` → 统一调用 helper → 回填 `direct_leg_gate*` / `direct_leg_scale*` / `omega_eff`”。side-routed 与 non-side 仍保留各自的 error prefix 与 gate-head 名称，因此 failure context 不弱化。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：简化前为两套 duplicated 片段：`direct_pose_leg_gate_head_shared` 分支重复 `sigmoid/power` 与 `exp/clamp/log`，`direct_pose_leg_gate_head` 分支再重复一次。简化后 duplicated 部分收敛到 `train/models.py:2852` 的 `_apply_direct_pose_leg_gate_outputs(...)`；分支内仅保留 feature assembly、omega scatter 和 helper 调用。新增 contract helper 位于 `train/models.py:243` 与 `train/models.py:2852`。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 新增 `test_forward_leg_gate_helper_dispatch_regression` 与 `test_forward_leg_scale_helper_dispatch_regression`，分别覆盖 learned / scale gate mode，并在 `side_routing=False/True` 下锁定 `_apply_direct_pose_leg_gate_outputs(...)` dispatch。Phase D tests 总数增至 `11`，联合 `tests.train.test_train_models_failfast` 总数增至 `129`。
- **forward snapshot 验证方式**：继续复用 `tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`；固定 `torch.manual_seed(12345)` 构造输入与 `torch.manual_seed(999)` 执行 eval forward，锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum。
- **state_dict fingerprint 验证方式**：继续复用 `tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`；确认 repeated construction 下 sorted key set、per-key `shape` / `dtype` / `sha256` 与 aggregate fingerprint 不漂移。
- **stage6 deterministic smoke 验证方式**：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch2_20260424 --run_name train_models_pre_phase_e_simplify_batch2_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；snapshot 单测命令；state_dict fingerprint 单测命令；AST broad-handler inline Python；stage6 deterministic smoke 命令；touched ranges removal-policy grep。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `129` 个用例通过（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=11`）；snapshot / state_dict fingerprint 单测均通过；AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`；stage6 smoke 通过，`ok_steps=5 skipped=0`，输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch2_20260424`。
- **阻塞项 / 风险**：本轮只统一了 gate/scale apply 壳层，没有继续收 side-routed / non-side 的 feature assembly 与 omega scatter。仍要避免把下一轮的 feature assembly 收口与 direct-pose 数值语义调整绑在一起。
- **为什么此时仍适合继续单文件 simplification**：当前 duplicated gate/scale apply 已被压平，但 upstream 的 leg feature assembly、per-side cue/embedding、omega scatter 仍完全位于单文件内、且还能继续被现有 snapshot/fingerprint/stage6 gate 机械验证。此时直接进入 Phase E，仍会把未压平的单文件热点与跨文件迁移一起放大。
- **下一步建议动作**：继续 simplification batch 3，优先收 side-routed / non-side leg feature assembly 与 `omega_leg` pre-gate shell，仍暂不进入 Phase E。

### 2026-04-24 — pre-Phase-E simplification batch 3 leg assembly / cue-embed / omega pre-gate shell

- **本轮目标**：继续停留在 `train/models.py` 单文件内，只收 side-routed / non-side leg feature assembly、side-routed cue / embedding shell、以及 `omega_leg` 进入 gate/scale apply helper 之前的公共准备；不进入 Phase E，不改核心算法 / loss 数值语义 / 输出 key / checkpoint contract / 默认超参。
- **实际完成项**：`train/models.py` 新增 `_DirectPoseSideLegAssembly`、`_prepare_direct_pose_leg_head_input(...)`、`_prepare_direct_pose_side_cues(...)`、`_prepare_direct_pose_side_embeddings(...)`、`_prepare_direct_pose_leg_omega(...)`、`_assemble_direct_pose_side_leg_features(...)`。其中 `_prepare_direct_pose_leg_head_input(...)` 统一 non-side 与 side-routed 两条路径的 leg head 输入 flatten/detach/shape contract；`_prepare_direct_pose_side_cues(...)` 与 `_prepare_direct_pose_side_embeddings(...)` 把 per-side cue / embedding 约束收成显式入口；`_prepare_direct_pose_leg_omega(...)` 统一 non-side reshape/max-rad clamp 与 side-routed scatter/max-rad clamp；`_assemble_direct_pose_side_leg_features(...)` 只负责 side-routed 分支的 plan/meas/phase/cue/embedding feature assembly，不碰 gate/scale apply 语义。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：batch 2 结束时，`forward(...)` 仍在 side-routed 分支内内联处理 plan/meas canonicalization、phase view、cue clamp、embedding broadcast、leg input flatten，以及 side omega scatter/max-rad clamp；non-side 分支则单独保留 `direct_flat.detach()` 与 `leg_delta.view(..., 3)` + max-rad clamp。batch 3 后，`forward(...)` 外层阅读路径收敛为：side-routed `feature assembly -> side head forward -> optional sign gate -> omega pre-gate helper -> gate/scale helper`，non-side `shared leg-input helper -> leg head/cross-leg ablation -> omega pre-gate helper -> gate/scale helper`。本轮没有把 side/non-side 强行并成 mega-helper，仍保留各自 branch-specific 数值路径。
- **helper 取舍说明**：本轮只加了 `1` 个 side-assembly contract dataclass 和 `5` 个硬 helper；没有新增单行 wrapper，也没有把 side-routed / non-side 全部揉成大而复杂的统一入口。`_prepare_direct_pose_leg_head_input(...)` 与 `_prepare_direct_pose_leg_omega(...)` 的收益分别是“共享 detach/shape contract”和“共享 pre-gate omega 形状/limit contract”；其余 helper 都承担 cue/embedding/error context 或 side assembly 去重职责。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 新增 `test_forward_leg_input_helper_dispatch_regression`、`test_forward_side_leg_cue_embedding_shell_dispatch_regression`、`test_forward_leg_omega_pre_gate_helper_dispatch_regression`；结合既有 `test_forward_leg_gate_helper_dispatch_regression` / `test_forward_leg_scale_helper_dispatch_regression`，最小锁定 shared leg-input helper、side cue/embedding 入口、omega pre-gate helper、learned/scale gate contract。Phase D tests 总数增至 `14`，联合 `tests.train.test_train_models_failfast` 总数增至 `132`。
- **forward snapshot 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`；继续复用 `_make_forward_snapshot_output()`，固定 `torch.manual_seed(12345)` 造输入、`torch.manual_seed(999)` 执行 eval forward，锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum。
- **state_dict fingerprint 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`；继续验证 repeated construction 下 sorted key set、per-key `shape` / `dtype` / `sha256` 与 aggregate fingerprint 不漂移。
- **stage6 deterministic smoke 验证方式**：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch3_20260424 --run_name train_models_pre_phase_e_simplify_batch3_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；snapshot 单测命令；state_dict fingerprint 单测命令；AST broad-handler inline Python；stage6 deterministic smoke 命令；touched code/test ranges removal-policy grep。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `132` 个用例通过（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=14`）；snapshot / state_dict fingerprint 单测均通过；AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`；stage6 smoke 通过，`ok_steps=5 skipped=0`，输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch3_20260424`。
- **阻塞项 / 风险**：本轮故意没有继续重做 gate/scale apply helper，也没有统一成 side/non-side mega-helper。remaining hotspot 主要是 side-routed head forward / optional sign gate 与 non-side cross-leg ablation + readout 仍留在 `forward(...)` 内；这些块已经比 batch 2 更线性，但仍不适合与 Phase E 跨文件迁移绑在同一批。
- **为什么此时仍适合继续单文件 simplification**：当前 direct-pose leg residual 路径里最嘈杂的 assembly / cue / embedding / omega pre-gate contract 已被单文件内机械压平，且 snapshot / fingerprint / stage6 smoke 都能继续证明“结构更线性、行为不变”。这说明余下工作仍可在单文件里低风险推进；如果现在直接切 Phase E，会把尚未完全压平的 forward 局部热点与 import/write-set 迁移耦合。
- **下一步建议动作**：继续 simplification batch 4，但仍暂不进入 Phase E；最推荐收 side-routed / non-side leg head forward 后的 residual writeback/readout 壳层，优先看 sign-gate / rank1 / cross-leg ablation 之后还能否再压出一个不改变数值语义的 dispatch 层。

### 2026-04-24 — pre-Phase-E simplification batch 4 side omega / non-side delta dispatch shell

- **本轮目标**：继续停留在 `train/models.py` 单文件内，只收 side-routed sign-gate / rank1 / side-omega resolver，以及 non-side cross-leg ablation → head fallback → rot6d residual writeback 的 dispatch shell；不进入 Phase E，不改核心算法 / loss 数值语义 / 输出 key / checkpoint contract / 默认超参。
- **实际完成项**：`train/models.py` 新增 `_DirectPoseSideLegOmegaOutputs`、`_resolve_direct_pose_side_leg_omegas(...)`、`_resolve_direct_pose_non_side_leg_delta(...)`、`_apply_direct_pose_rot6d_leg_delta(...)`。其中 `_resolve_direct_pose_side_leg_omegas(...)` 把 side-routed rank1 / per-joint / optional sign-gate 收到统一入口；`_resolve_direct_pose_non_side_leg_delta(...)` 把 cross-leg ablation 与 `direct_pose_leg_head(...)` fallback 收到一个带 shape contract 的入口；`_apply_direct_pose_rot6d_leg_delta(...)` 把 non-side `rot6d_add` residual writeback 收成独立壳层。`forward(...)` 外层现在更接近“assembly -> omega resolver -> pre-gate -> gate/scale”与“leg-input -> leg-delta resolver -> so3/rot6d dispatch”两条线性路径。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：batch 3 后，side-routed 分支仍在 `forward(...)` 内内联处理 rank1/per-joint 分叉与 optional sign-gate；non-side 分支仍先 inline 调 cross-leg ablation / head fallback，再在 `leg_mode` 分支里内联 `rot6d_add` writeback。batch 4 后，side-routed 这段收敛为 `_resolve_direct_pose_side_leg_omegas(...)`，non-side 这段收敛为 `_resolve_direct_pose_non_side_leg_delta(...)` + `_apply_direct_pose_rot6d_leg_delta(...)`。本轮仍未把 side/non-side 合并成一个大 helper，而是按 branch-specific contract 收口。
- **helper 取舍说明**：本轮新增的 3 个 helper 都不是单行 wrapper。`_resolve_direct_pose_side_leg_omegas(...)` 收拢了 rank1 / sign-gate 的真实分支逻辑与错误上下文；`_resolve_direct_pose_non_side_leg_delta(...)` 提供了 cross-leg ablation/head fallback 的 shared dispatch 与 shape contract；`_apply_direct_pose_rot6d_leg_delta(...)` 收拢了 `rot6d_add` writeback contract。没有为了“切小”而机械包一层。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 新增 `test_forward_side_leg_omega_resolver_dispatch_regression`、`test_forward_non_side_leg_delta_dispatch_regression`、`test_forward_non_side_rot6d_residual_writeback_dispatch_regression`。结合 batch 2/3 已有 dispatch regression，最小锁定 side omega resolver、non-side leg-delta resolver、rot6d residual writeback 壳层。Phase D tests 总数增至 `17`，联合 `tests.train.test_train_models_failfast` 总数增至 `135`。
- **forward snapshot 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`；继续复用 `_make_forward_snapshot_output()`，固定 `torch.manual_seed(12345)` 造输入、`torch.manual_seed(999)` 执行 eval forward，锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum。
- **state_dict fingerprint 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`；继续验证 repeated construction 下 sorted key set、per-key `shape` / `dtype` / `sha256` 与 aggregate fingerprint 不漂移。
- **stage6 deterministic smoke 验证方式**：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch4_20260424 --run_name train_models_pre_phase_e_simplify_batch4_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；snapshot 单测命令；state_dict fingerprint 单测命令；AST broad-handler inline Python；stage6 deterministic smoke 命令；touched code/test ranges removal-policy grep。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `135` 个用例通过（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=17`）；snapshot / state_dict fingerprint 单测均通过；AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`；stage6 smoke 通过，`ok_steps=5 skipped=0`，输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch4_20260424`。
- **阻塞项 / 风险**：本轮仍没有碰 output writer、stats contract 或 gate/scale 数值语义。remaining hotspot 主要是 side-routed / non-side 分支的 final writeback/readout orchestration 仍在 `forward(...)` 内，但此时已经比 batch 3 更线性。若直接进入 Phase E，仍会把这些未完全收平的 branch tail 与跨文件迁移绑在一起。
- **为什么此时仍适合继续单文件 simplification**：batch 4 继续证明 direct-pose leg residual 的 branch-specific tail 还可以在单文件里机械压平，并且现有 snapshot / fingerprint / stage6 smoke 足够锁定不变性。此时继续 batch 5 的风险仍低于直接跨文件迁移。
- **下一步建议动作**：继续 simplification batch 5，但仍暂不进入 Phase E；最推荐收 final direct-pose leg residual writeback / result-assignment dispatch，看是否还能把 side/non-side tail 对齐成更线性的壳层而不触碰数值语义。

### 2026-04-24 — pre-Phase-E simplification batch 5 leg residual final writeback / result-assignment shell

- **本轮目标**：继续停留在 `train/models.py` 单文件内，只收 direct-pose leg residual 的 final writeback / result-assignment dispatch shell；范围限定为 side-routed / non-side 两条 leg residual 分支在最终输出写回前的薄 orchestration，不进入 Phase E，不改核心算法 / loss 数值语义 / 输出 key / checkpoint contract / 默认超参。
- **实际完成项**：`train/models.py` 新增 `_DirectPoseLegWritebackOutputs` 与 `_dispatch_direct_pose_leg_residual_writeback(...)`，并让 `_write_forward_direct_pose_outputs(...)` 统一消费该 writeback contract。batch 4 之后留在 `forward(...)` 内的 `direct_leg_omega*` / `direct_leg_gate*` / `direct_leg_scale*` / `direct_leg_side_sign_gate` 手工回填与 `rot6d_add` final writeback dispatch，现已统一收敛到同一个 final shell；side-routed 路径固定以 shared omega 语义进入该 shell，non-side 路径则带着 `leg_mode='so3'|'rot6d_add'` 进入同一个 final dispatch 壳。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：batch 4 后，side-routed 分支在 `_resolve_direct_pose_side_leg_omegas(...)` / gate helper 之后仍内联回填 `direct_leg_*` locals，non-side 分支在 `_resolve_direct_pose_non_side_leg_delta(...)` 之后仍内联 `so3` / `rot6d_add` final writeback 与 result-assignment 参数展开。batch 5 后，`forward(...)` 外层阅读路径更接近：side-routed `assembly -> omega resolver -> pre-gate helper -> gate/scale helper -> final writeback shell`，non-side `leg-input -> delta resolver -> so3/rot6d dispatch -> final writeback shell`。本轮没有把 side/non-side 强行并成 mega-helper，只把最终 output contract 与 writeback dispatch 收口。
- **helper 取舍说明**：本轮只新增 `1` 个 writeback contract dataclass 和 `1` 个 final dispatch helper。它们都不是单行包装：`_DirectPoseLegWritebackOutputs` 明确冻结 `out_direct` 与 `direct_leg_*` output key contract，`_dispatch_direct_pose_leg_residual_writeback(...)` 则真实承载 side-routed / non-side 共同的 final writeback dispatch。没有出现“为了拆分而拆分”的单行 wrapper。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d` 新增 `test_forward_leg_residual_final_writeback_shell_regression`，最小锁定 side-routed `scale` / `sign-gate` / `rank1`、non-side `learned` / `rot6d_add` 四类分支在 resolver 之后都会进入 `_dispatch_direct_pose_leg_residual_writeback(...)`，并继续保持 `direct_leg_omega` / `direct_leg_gate` / `direct_leg_scale*` / `direct_leg_side_sign_gate` / `out_direct` 的 output key contract。Phase D tests 总数增至 `18`，联合 `tests.train.test_train_models_failfast` 总数增至 `136`。
- **forward snapshot 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`；继续复用 `_make_forward_snapshot_output()`，固定 `torch.manual_seed(12345)` 造输入、`torch.manual_seed(999)` 执行 eval forward，锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum。
- **state_dict fingerprint 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`；继续验证 repeated construction 下 sorted key set、per-key `shape` / `dtype` / `sha256` 与 aggregate fingerprint 不漂移。
- **stage6 deterministic smoke 验证方式**：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch5_20260424 --run_name train_models_pre_phase_e_simplify_batch5_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`；AST broad-handler inline Python；stage6 deterministic smoke 命令；touched code/test ranges removal-policy grep。
- **验证结果**：通过；`py_compile` 通过，联合 `unittest` 共 `136` 个用例通过（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=18`）；snapshot / state_dict fingerprint 单测均通过；AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`；stage6 smoke 通过，`ok_steps=5 skipped=0`，输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch5_20260424`。
- **阻塞项 / 风险**：本轮故意没有重做 gate/scale helper、cue/embedding helper、也没有触碰 direct-pose 数学语义；remaining hotspot 已不再是 final writeback，而是 `forward(...)` 里仍偏长的 direct-pose 主体 readout/orchestration 体量。若此时直接进入 Phase E，仍会把尚可在单文件内继续机械压平的 forward 局部热点与跨文件迁移耦合。
- **为什么此时仍适合继续单文件 simplification**：batch 5 继续证明 direct-pose leg residual 的尾段 contract 仍能在单文件里机械压平，并且 snapshot / fingerprint / stage6 smoke 足够锁定“结构更线性、行为不变”。此时继续 batch 6 的边际风险仍低于立刻进入 Phase E。
- **下一步建议动作**：继续 simplification batch 6，但仍暂不进入 Phase E；最推荐看 direct-pose readout / `leg_outputs` 初始化 / final `out_direct` writer 之间还有没有一层可继续压平的编排壳，同时保持 helper 数量少而硬。

### 2026-04-24 — pre-Phase-E simplification batch 6 corrective shell + thin-helper retirement

- **本轮目标**：对 batch 5 做一次保守 corrective pass，撤回“长换成散”的薄 helper / kwargs-heavy dispatch 方向，改为只保留 branch-sized shell 与少而硬的 contract helper；仍只改 `train/models.py` 单文件内的 direct-pose leg residual orchestration，不进入 Phase E，不改核心算法 / loss 数值语义 / 输出 key / checkpoint contract / 默认超参。
- **实际完成项**：`train/models.py` 退休 `_prepare_direct_pose_side_cues(...)`、`_prepare_direct_pose_side_embeddings(...)`、`_prepare_direct_pose_leg_head_input(...)`、`_apply_direct_pose_rot6d_leg_delta(...)`、`_dispatch_direct_pose_leg_residual_writeback(...)`、`_DirectPoseLegWritebackOutputs`；保留 `_assemble_direct_pose_side_leg_features(...)`、`_prepare_direct_pose_leg_omega(...)`、`_apply_direct_pose_leg_gate_outputs(...)`、`_resolve_direct_pose_side_leg_omegas(...)`、`_resolve_direct_pose_non_side_leg_delta(...)` 这些真实硬 helper。`forward(...)` 外层现改为通过 `_forward_side_routed_leg_residual(...)` 与 `_forward_non_side_leg_residual(...)` 两个 branch-sized shell 承接 side-routed / non-side residual path，其中 non-side shell 内部用 guard clause 展平旧金字塔控制流。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：batch 5 的写法把 final writeback / dispatch 切成多个薄 helper，并引入 dataclass + kwargs-heavy dispatch，虽然压平了 inner body，但没有压平 branch control flow。batch 6 后，`forward(...)` 外层改回更线性的两条宿主路径：side-routed `assembly -> omega resolver -> pre-gate helper -> gate/scale helper -> side shell return`，non-side `guard clauses -> leg-delta resolver -> so3/rot6d branch -> non-side shell return`；最终 `result` 写回继续通过已有 `_write_forward_direct_pose_outputs(...)` 显式参数完成，不再跨多处跳转追踪 writeback contract。
- **这轮具体压平了哪些 duplicated shell / dispatch 壳**：压平了 side-routed 与 non-side 两条 direct-pose leg residual 分支在 `forward(...)` 里的 branch-local orchestration 壳；移除了 batch 5 造成的 final writeback dispatch 壳、rot6d residual 单行壳、leg-input 单行壳，以及 side cue / embedding 的单-call-site 薄壳。保留下来的 duplication 只剩 branch-local、可就地阅读的结果赋值与 mode-specific 写回。
- **helper 取舍说明**：本轮明确把“extract-for-host-readability”与“真实去重/contract helper”分开处理。`_assemble_direct_pose_side_leg_features(...)` 因为 inline 回去会把 `forward(...)` 外层重新拉长约百余行而保留；其余退役 helper 都没有跨分支去重收益，或只是单行包装，因此回收到 caller/shell。没有再新增看起来像“为了拆分而拆分”的 helper。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests.train.test_event_motion_model_refactor_phase_d.py` 删除对已退休薄 helper 的 dispatch regression，改为新增 `test_forward_side_routed_leg_residual_shell_dispatch_regression` 与 `test_forward_non_side_leg_residual_shell_dispatch_regression`，最小锁定 side-routed / non-side 两个 branch-sized shell 会被 `forward(...)` 调用，并继续保持 learned / scale / rank1 / sign-gate / rot6d_add / so3 下的 output key contract。Phase D tests 当前为 `16`，联合 `tests.train.test_train_models_failfast` 当前为 `134`。
- **forward snapshot 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`；继续复用 `_make_forward_snapshot_output()`，固定 `torch.manual_seed(12345)` 造输入、`torch.manual_seed(999)` 执行 eval forward，锁定 output key set、关键 tensor shape、finite、sum/mean/L2 checksum。
- **state_dict fingerprint 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`；继续验证 repeated construction 下 sorted key set、per-key `shape` / `dtype` / `sha256` 与 aggregate fingerprint 不漂移。
- **stage6 deterministic smoke 验证方式**：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch6_20260424 --run_name train_models_pre_phase_e_simplify_batch6_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`。
- **运行过的命令**：`python3 -m py_compile train/models.py tests/train/test_train_models_failfast.py tests/train/test_event_motion_model_refactor_phase_d.py`；`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d`；`python3 -m unittest tests.train.test_train_models_failfast tests.train.test_event_motion_model_refactor_phase_d`；snapshot 单测命令；state_dict fingerprint 单测命令；AST broad-handler inline Python；touched code/test ranges removal-policy grep；stage6 deterministic smoke 命令。
- **验证结果**：通过；`py_compile` 通过，`tests.train.test_event_motion_model_refactor_phase_d=16`，联合 `unittest` 共 `134` 个用例通过（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=16`）；snapshot / state_dict fingerprint 单测均通过；AST 计数得到 broad=`0` / exact=`0` / as_exc=`0`；touched ranges removal-policy grep 对 nested fallback / `warnings.warn(...)` / `state_dict[...] = state_dict.pop(...)` / `# compat` / `# legacy` 均为 `0`；stage6 smoke 通过，`ok_steps=5 skipped=0`，输出目录 `debug_output/_tmp_train_models_pre_phase_e_simplify_batch6_20260424`。
- **阻塞项 / 风险**：本轮没有动 gate/scale 数学语义、cue/embedding 公式、loss 数值逻辑或 checkpoint contract；风险主要来自“回退 batch 5 薄 helper”时是否漏掉 output key / writeback 细节。snapshot / fingerprint / stage6 smoke 已覆盖这部分风险。当前 remaining hotspot 已不再是 leg residual branch tail，而是更高一层的 direct-pose 主 readout/orchestration 体量。
- **为什么此时仍适合继续单文件 simplification**：这次 corrective pass 说明“localized duplication > scattered cognition”在当前阶段更稳妥：把控制流收回两个 branch-sized shell 后，后续如果继续单文件 simplification，将面对的是可局部搬运的宿主壳，而不是跨 1000 行跳转的薄 helper 网络。这使 Phase E 前的最后一小步单文件整理仍然具备低风险、可机械验证的特征。
- **下一步建议动作**：batch 6 完成后先暂停评估；除非还能找到一个同级别的 branch-sized shell，否则不建议继续开 batch 7。若必须再做一批，最推荐只看 direct-pose 主 readout/orchestration 中剩余的单个宿主壳；否则应开始准备 Phase E 跨文件迁移。

### 2026-04-24 — pre-Phase-E simplification batch 7 event-clock loop shell

- **本轮目标**：只压平 `train/models.py` 中 event-clock on 分支 `for _t in range(Tq)` 的 loop body，重点收 phase append / cue append / `contact_plan_time_head(...)` per-step 壳与 gate/corrector/readout 宿主阅读路径；不碰 `__init__`、`_canonicalize_contacts_meas_inputs(...)`、direct-pose leg residual、forward 大 stage 切分或任何跨文件迁移。
- **实际完成项**：`forward(...)` 内新增共享 nested def `_append_contact_plan_direct_step_inputs(...)` 与 event-clock on 专用 nested def `_step_contact_plan_event_clock(...)`。前者把 `phase_in_direct_seq.append(...)` / `leg_side_cue_seq.append(...)` 的 per-step shape contract 与错误包装收回 loop-local closure；后者把 event-clock on 的 `plan_z raw/logits/err -> gate -> corrector -> optional time bias -> debug/logit/prob append -> lambda/dyn/delta append` 压成单个 step shell。event-clock off 路径只复用 append helper，因此 loop 外层更短，但没有新增 module-level helper、dataclass return、15-tuple return 或 kwargs-heavy dispatch。
- **修改文件列表**：`train/models.py`；`tests/train/test_event_motion_model_refactor_phase_d.py`；`docs/changes/2026-04-21_train_models_fail_fast_inventory.md`；`docs/changes/2026-04-21_train_models_single_file_refactor_roadmap.md`；`docs/delete/2026-04-24_train_models_pre_phase_e_cleanup_audit.md`。
- **简化前后结构对照**：batch 6 之后，event-clock on loop 仍把 phase/cue contract、gate/corrector 与 time-bias try/except 混在宿主路径里。batch 7 后，宿主阅读路径基本变为 `for _t in range(Tq): _step_contact_plan_event_clock(_t)`；shell 内顺序固定为 `phase append -> cue append -> plan_z_raw/logits/err -> gate -> corrector -> optional time bias -> debug append -> logits/probs -> lambda/dyn/delta append`。这是 host-path 变短，而不是把同一逻辑散到很多薄 helper。
- **helper 取舍说明**：本轮只增加 `2` 个 nested def，且都停留在 `forward(...)` closure 内。`_append_contact_plan_direct_step_inputs(...)` 有明确 contract 收益；`_step_contact_plan_event_clock(...)` 有明确 host-readability 收益。没有新增看起来像“为了拆分而拆分”的 module-level method，也没有把 time-bias 再单独拆成薄 wrapper。
- **broad handler 计数变化**：维持 `0 -> 0`；精确 `except Exception:` 维持 `0 -> 0`；`except Exception as exc` 维持 `0 -> 0`。
- **新增 / 更新测试**：`tests/train/test_event_motion_model_refactor_phase_d.py` 新增 `test_event_clock_loop_shell_phase_cue_time_bias_regression`，最小锁定 event-clock on 路径中 phase/cue/time-bias 三者同时工作，且 `event_clock_*` / `contacts_plan*` / `out_direct` key contract 保持不变。Phase D tests 更新为 `17`，联合 `tests.train.test_train_models_failfast` 更新为 `135`。
- **forward snapshot 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_forward_output_snapshot_deterministic_regression`。
- **state_dict fingerprint 验证方式**：`python3 -m unittest tests.train.test_event_motion_model_refactor_phase_d.EventMotionModelRefactorPhaseDTest.test_state_dict_fingerprint_repeated_construction_regression`。
- **stage6 deterministic smoke 验证方式**：`PYTHONPATH=. debug_output/_tmp_phasecd_min_ablation_20260330/cpu_nomps_exec.py -m train.posttrain --config debug_output/_target_stage6_same_donor_det_20260421/current_code/config_stage6_offset0_e8x60.json --out_dir debug_output/_tmp_train_models_pre_phase_e_simplify_batch7_20260424 --run_name train_models_pre_phase_e_simplify_batch7_20260424 --epochs 1 --steps_per_epoch 5 --save_step_ckpts 0,1,5 --rollout_random_offset false --seed 0`。
- **运行过的命令**：compile / focused event-clock regression / full joint `unittest` / snapshot / state_dict fingerprint / AST broad-count / stage6 deterministic smoke / touched ranges removal-policy grep。
- **验证结果**：全部通过；联合 `unittest` 共 `135` 个用例通过（`tests.train.test_train_models_failfast=118`，`tests.train.test_event_motion_model_refactor_phase_d=17`）；snapshot、fingerprint、stage6 smoke 均证明 batch 7 前后一致；AST 计数保持 broad=`0` / exact=`0` / as_exc=`0`。
- **阻塞项 / 风险**：batch 7 之后，显而易见的“少而硬 nested def 就能压平宿主路径”的单文件收益基本耗尽。再往前推，很容易回到 batch 5 的“长换成散”或被迫做 forward 大 stage 切分，这已经越过本轮红线。
- **为什么此时仍适合继续单文件 simplification**：适合的是“做到 batch 7 为止”——因为 event-clock loop 仍是一个 closure-local、branch-sized、可用 nested def 收口的热点。完成它之后，继续单文件 simplification 的理由明显变弱。
- **下一步建议动作**：优先评估停在 batch 7 并准备 Phase E。若必须再做一小步，只能挑 contact-plan non-event-clock / finalize handoff 的单个宿主壳；否则默认进入 Phase E readiness，而不是继续批量化单文件拆壳。
