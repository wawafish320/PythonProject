# 2026-04-14 `train/models.py` 热点重构路线图（v2）

Date: 2026-04-14  
Status: Active v2 / in-progress（已回填 A1 / A2 / A3 / B1 / B2 / B3 / B4；采用“先做最小必要清理，再做功能拆分”作为第一优先策略；明确**不新增 Python 脚本/模块**，只复用现有承载文件）  
Scope: `train/models.py`（本轮只做结构去耦、职责拆分、兼容边界收口与异常边界收紧；不改模型语义、不改默认超参、不改 rotvec / SO(3) 几何约定、不改输出 key/schema）  
Goal: 在**不改变语义/行为**前提下，优先降低“超长 `EventMotionModel` / `MotionJointLoss`、checkpoint compat 与 bundle attach 混入模型类、`forward` 内调试/ablation 路径堆积、helper 与几何/工具逻辑边界不清”的维护风险。  
Non-goal: 不改核心算法/数学定义、不引入新的模型架构、不新增 `train/*.py` / `train/validate/*.py` 文件、不把重构变成全量风格清理。

关联参考文档：
- 参考格式：`docs/delete/2026-04-13_train_training_MPL_refactor_hotspots_roadmap.md`
- 上一版路线图：`docs/delete/2026-03-10_train_models_refactor_hotspots_roadmap.md`
- 关联兼容承载：`train/model_ckpt_compat.py`

---

## 0) 当前策略（先做最小必要清理，再做功能拆分；且只复用现有模块）

本轮总原则先明确：

- **第一优先策略不是先做一轮全面清理，也不是立刻大规模搬文件，而是先做“为拆分服务的最小必要清理”。**
- `train/models.py` 当前的核心问题不是某一个 100 行 helper，而是**模型定义 / 运行时分支路由 / checkpoint compat / bundle attach / loss 统计**五类职责交织。
- 如果先做全面重命名、格式化、注释清洗，会放大 diff 面；如果直接搬代码出去，又会因为边界还没收紧而把耦合原样搬走。
- 因此统一顺序必须是：**边界清理 -> 复用现有模块抽离独立职责 -> 类内壳层化 -> 收紧异常边界**。
- 本轮明确遵守你的偏好：**不新增 Python 脚本/模块**；只允许把职责移动到已有文件，例如 `train/model_ckpt_compat.py`、`train/geometry.py`、`train/utils.py`。

统一执行顺序：

1. **Phase A: 最小必要清理与边界收口（低/中风险）**
   - 先清掉 `models.py` 内明显的脚本残留与边界混淆点。
   - 先把 `__init__` / `forward` 中的职责块命名出来，而不是直接搬走。
2. **Phase B: 只向现有模块抽离独立职责（中风险）**
   - 优先抽离 checkpoint compat / bundle attach / 纯几何 helper / 通用输入 helper。
   - 不新建 `train/losses.py`、`train/model_components.py`、`train/joint_specs.py`。
3. **Phase C: `EventMotionModel` 类内壳层化（中/高风险）**
   - 先把 `forward` 拆成 orchestration 壳 + 具名私有 helper。
   - 再把 `__init__` 拆成配置归一化 + 子模块构建两层。
4. **Phase D: `MotionJointLoss` 结构收口（中风险）**
   - 本轮优先在原文件内收口，不新建独立 loss 文件。
5. **Phase E: 异常边界与 debug/ablation 路径收紧（中风险）**
   - 区分 active path、compat path、debug-only path，减少热路径宽泛吞没。

核心原则：

- one step, one commit
- 每步必须有 before/after 结构指标
- 任何一步回归失败，立即停在当前 commit，不继续后续步骤
- 每步固定汇报 4 项：LOC、`def`/函数数、最大函数长度、`except Exception` 数量
- 每步还必须汇报至少 **1 项本轮主题相关的结构债指标**
- **单步硬门禁**：以 Step 收尾为准，必须满足 `LOC_after <= LOC_before`

新增约束（本路线图强制）：

- 不允许新增 Python 文件来“掩盖” `models.py` 复杂度
- 不允许继续把新逻辑直接堆进 `EventMotionModel.forward`
- 不允许新增新的“compat 壳 + 主实现”双轨共存
- 不允许“只抽函数不删旧逻辑”
- 新增 helper 必须带来至少 1 项可量化净收益（最大函数长度下降 / `except Exception` 下降 / 明显职责块外移 / 重复 guard 收敛）

### 拆分约束（v2，Phase B / C 强制执行）

拆分前准入（满足任一即可进入候选）：

- 被调用 >=2 次（或下一步已明确会复用 >=2 次）
- 封装了可独立命名的模型领域概念
- 拆出后调用处更易读（调用点行数或嵌套层级下降）
- 能稳定落到**已有模块**的职责边界中

硬禁止（命中任一则不允许拆）：

- 纯转发 wrapper / 无实质边界的间接层
- 函数名仅复述代码字面行为（无领域语义）
- 再次引入黑盒上下文（隐式依赖一长串 `getattr(self, ...)` 却不声明输入）
- 只抽函数、不删除原地旧逻辑（双实现并存）
- **Step 收尾时** `LOC_after > LOC_before`
- 参数爆炸：helper 参数 > 8 且未收敛到结构化 context / payload

单次调用 helper 的例外规则（替代绝对禁止）：

- 调用次数 = 1 允许，但必须同时满足：
  - 具备独立概念命名
  - 调用点可读性提升
  - 至少 1 项结构指标下降（最大函数长度 / `except Exception` / 重复 guard）

拆分后验流程（强制）：

1. 先按“独立职责 + 现有承载文件匹配度”拆分
2. Step 收尾前检查 `LOC_after <= LOC_before`
3. 若不满足：回收无概念 helper / 参数搬运层 / 纯 wrapper
4. 在**当前 step**完成净减回收后，才允许进入下一步

---

## 0.5) 本轮进展（2026-04-14 / baseline-0 -> A3+B3）

当前已完成 `A1` / `A2` / `A3` / `B1`；以下先保留 baseline-0 快照，随后记录当前执行快照：

- 文件总行数：`5886`
- `def` / 函数数：`113`
- `class` 数：`10`
- 最大类跨度：`EventMotionModel = 3631` 行（`train/models.py:588` 起）
- 次大类跨度：`MotionJointLoss = 1667` 行（`train/models.py:4220` 起）
- `EventMotionModel` 方法数：`21`
- `MotionJointLoss` 方法数：`64`
- 最大函数长度：`EventMotionModel.forward = 1767` 行（`train/models.py:2364`）
- 次大函数长度：`EventMotionModel.__init__ = 1181` 行（`train/models.py:593`）
- 宽泛异常吞没：`except Exception = 98`
- `print(...)` 调用：`5`
- `getattr(...)` 调用：`173`
- `direct_pose_` token count：`646`
- `contact_plan_` token count：`119`
- `event_clock_` token count：`89`
- `lambda_fusion_` token count：`44`
- `legacy` marker count：`31`
- `fallback` marker count：`18`

当前执行快照（through `A3+B4`）：

- 已完成：
  - `A1`：中段脚本残留 import / `tqdm` fallback 清理
  - `A2`：`EventMotionModel.__init__` 类内职责块命名与 builder 化
  - `A3`：`forward` runtime override / ablation 具名 helper 收口
  - `B1`：direct-pose ckpt compat 主实现迁回 `train/model_ckpt_compat.py`
  - `B2`：`attach_motion_encoder` 主实现迁回 `train/model_ckpt_compat.py`
  - `B3`：`_apply_so3_correction_to_delta_raw` 主实现迁回 `train/geometry.py`
  - `B4`：通用输入拼装 / joint-spec 解析 helper 主实现迁回 `train/utils.py`
- 当前 `train/models.py`：
  - LOC：`5313`
  - `def` / 函数数：`117`
  - 最大函数长度：`EventMotionModel.forward = 1561`
  - 宽泛异常吞没：`except Exception = 90`
  - `EventMotionModel` 类跨度：`3215`
- 当前 `train/utils.py`：
  - LOC：`723`
  - `def` / 函数数：`38`
  - 最大函数长度：`42`
  - 宽泛异常吞没：`except Exception = 10`
- 当前 `train/geometry.py`：
  - LOC：`1066`
  - `def` / 函数数：`34`
  - 最大函数长度：`74`
  - 宽泛异常吞没：`except Exception = 4`
- 当前 `train/model_ckpt_compat.py`：
  - LOC：`2163`
  - `def` / 函数数：`29`
  - 最大函数长度：`223`
  - 宽泛异常吞没：`except Exception = 39`
- 双文件合计（`models.py` + `model_ckpt_compat.py`）：
  - LOC：`7625`
  - `def` / 函数数：`150`
  - `except Exception`：`130`
- 已验证：
  - `python3 -m py_compile train/models.py train/model_ckpt_compat.py`
  - `python3 -m py_compile train/models.py train/geometry.py train/training_MPL.py train/model_ckpt_compat.py`
  - `python3 -m py_compile train/models.py train/utils.py train/training_MPL.py train/geometry.py train/model_ckpt_compat.py`
  - import smoke：`import train.models`、`import train.model_ckpt_compat`、`import train.utils`、`import train.training_MPL`
  - object smoke：`EventMotionModel(in_state_dim=1, out_motion_dim=1)` 可实例化，`attach_motion_encoder` 方法仍存在

局部热点长度（baseline-0）：

- `train/models.py:2364` `EventMotionModel.forward`：`1767`
- `train/models.py:593` `EventMotionModel.__init__`：`1181`
- `train/models.py:1993` `_maybe_upgrade_direct_pose_split_state_dict`：`250`
- `train/models.py:4221` `MotionJointLoss.__init__`：`194`
- `train/models.py:4132` `attach_motion_encoder`：`87`
- `train/models.py:5426` `_compute_direct_pose_payload`：`86`
- `train/models.py:4830` `compute_attention_regularization`：`76`
- `train/models.py:4719` `_rot_local_tail_candidates`：`71`
- `train/models.py:5603` `_apply_event_clock_components`：`65`
- `train/models.py:5363` `_compute_direct_pose_group_norm_payload`：`62`
- `train/models.py:4638` `_compute_unified_weights_cpu`：`55`
- `train/models.py:1939` `_forward_direct_pose_readout`：`53`

当前新增观察（本轮需要明确写进策略）：

- `train/models.py:569-577` 的**脚本残留 import / tqdm fallback** 已在 `A1` 清理；这是后续拆分前必要的边界降噪。
- direct-pose ckpt compat 主实现已在 `B1` 迁回 `train/model_ckpt_compat.py`；`EventMotionModel` 仅保留薄入口壳。
- `train/models.py:3793-3794` 的 `attach_motion_encoder` 已收敛为薄 wrapper；bundle 解析 / 维度推断 / frozen module 安装主实现已迁至 `train/model_ckpt_compat.py:340-419`。
- `train/geometry.py:403` 现在承载 `_apply_so3_correction_to_delta_raw`；`train/training_MPL.py` 已直接从 `geometry.py` 引用该 helper，`train/models.py` 不再公开该符号。
- `train/utils.py:54-148` 现在承载 `_normalize_joint_spec_items` / `_resolve_joint_spec_indices` / `_build_pretrain_contact_encoder_input`；`train/training_MPL.py` 已不再从 `models.py` 导入这些非模型语义 helper。

当前结论：

- 2026-03-10 的旧路线图已经不足以覆盖当前 `direct_pose / contact_plan / compat` 的复杂度。
- 新版 v2 路线图必须把“**不新增文件**”和“**优先复用现有模块**”作为硬约束写死。
- 当前最优策略不是“先全面清理”也不是“先全面拆分”，而是**先做最小必要清理，再做职责外移**。

---

## 1) 基线现状（针对本轮热点问题）

baseline-0 代码快照核对（`train/models.py`）：

- **热点问题 1**：`train/models.py:588` 起的 `EventMotionModel` 类跨度达到 `3631` 行，模型定义、运行时路由、compat 适配、bundle attach 全部缠在一起。
- **热点问题 2**：`train/models.py:2364` 的 `EventMotionModel.forward` 长达 `1767` 行，当前最大的不是单一算法复杂度，而是 `contact_plan / direct_pose / leg residual / lambda fusion / SO(3) corrector / runtime override` 六类逻辑堆叠。
- **热点问题 3**：`train/models.py:593` 的 `EventMotionModel.__init__` 长达 `1181` 行，配置归一化、buffer 注册、joint 解析、head 构建、compat 元数据准备没有清晰分段。
- **热点问题 4**：`train/models.py:1993` 到 `train/models.py:2294` 的 direct-pose ckpt compat 逻辑本质属于加载/兼容层，继续挂在模型类里会放大维护成本。
- **热点问题 5**：`train/models.py:4132` 的 `attach_motion_encoder` 是 bundle loader + frozen module installer，不属于模型主语义。
- **热点问题 6**：`train/models.py:4220` 起的 `MotionJointLoss` 类跨度 `1667` 行，虽然 `forward` 已经不厚，但 `__init__`、group mask、骨骼权重、direct-pose payload、aux losses 仍混在一起。
- **热点问题 7**：`train/models.py:139`、`train/models.py:180` 这类 helper 已被训练入口直接复用，说明其职责应当沉到现有工具模块，而不是继续公开挂在 `models.py`。
- **热点问题 8**：`train/models.py:569-577` 的中段 import / `tqdm` fallback 强烈暗示文件历史上混入过脚手架逻辑，应该先清掉这类边界噪音。
- **热点问题 9**：`except Exception = 98` 说明当前文件仍有较重的“热路径与 debug / fallback 一起吞异常”问题，拆分前必须分层。

结构指标基线（本路线图起点）：

- LOC：`5886`
- `def` / 函数数：`113`
- `class` 数：`10`
- 最大函数长度：`EventMotionModel.forward = 1767`
- 主题结构债指标 #1：`direct_pose_ token count = 646`
- 主题结构债指标 #2：`except Exception = 98`
- 主题结构债指标 #3：`getattr(...) = 173`

为什么本轮仍优先处理 `train/models.py`：

- `train/training_MPL.py` 已经有新的路线图和阶段性结构收口，但其热点更多是 trainer / rollout / diagnostics orchestration。
- `train/models.py` 则是**模型主语义、compat、loss、runtime 分支**的共同耦合点，继续积累会直接抬高 posttrain / freerun / export 三条链路的回归风险。
- 而且这轮已有明确约束：**不新增 Python 文件**，因此最有价值的工作就是先把 `models.py` 压回合理边界，并把独立职责送回现有模块。

---

## 2) 模块落点决策（不新增文件版本）

本轮明确采用以下落点，不新增新的 Python 文件：

- **落到 `train/model_ckpt_compat.py`**
  - `train/models.py:1993` `_maybe_upgrade_direct_pose_split_state_dict`
  - `train/models.py:2244` `_maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict`
  - `train/models.py:2286` `adapt_legacy_state_dict_`
  - `train/models.py:2294` `load_state_dict` 中的 compat 预处理部分
  - `train/models.py:4132` `attach_motion_encoder`

- **落到 `train/geometry.py`**
  - `train/models.py:180` `_apply_so3_correction_to_delta_raw`

- **落到 `train/utils.py`**
  - `train/models.py:139` `_build_pretrain_contact_encoder_input`
  - `train/models.py:71` `_normalize_joint_spec_items`
  - `train/models.py:99` `_resolve_joint_spec_indices`

- **本轮先留在 `train/models.py`**
  - `ContactMeasHeadLowerBodyNoHistV1`
  - `MotionEncoder`
  - `PeriodHead`
  - `_CondFiLM`
  - `_ResidualMLPBlock`
  - `_BoneSliceResidualAdapter`
  - `PlanZCorrector`
  - `PeriodicityGate`
  - `EventMotionModel`
  - `MotionJointLoss`

保留在 `models.py` 的原因：

- 当前没有合适的**现有**组件承载文件可接它们；
- 本轮约束是不新增 `train/model_components.py`、`train/losses.py` 之类的新文件；
- 因此正确策略是：**先做类内壳层化，再决定是否有必要在后续版本新增专用承载文件**。

---

## 3) 具体改动流程

## Phase A — 最小必要清理与边界收口（A1 + A2 + A3）

### Step A1 — 清理 `models.py` 中的脚本残留与导入边界（低风险）

状态：**已完成（2026-04-14）**

目标：先去掉最明显的边界噪音，让后续拆分不建立在脏边界上。

实施：

- 清理 `train/models.py:569-577` 的中段 import / `tqdm` fallback 残留。
- 删除未在模型语义中实际使用的脚本级 import。
- 确保模型文件顶部 import 集中、含义明确。

约束：

- 不改变任何类 / 函数对外签名。
- 不顺手做全文件风格重排。

验收门：

- `train/models.py` 仍可正常 import
- 中段 import 残留归零
- `LOC_after <= LOC_before`

回填结果：

- 已删除 `train/models.py` 中段脚本残留：
  - `import os, json, math, glob, time, argparse`
  - `from torch.utils.data import DataLoader`
  - `tqdm` fallback（含 warning `print(...)` 与本地 `def tqdm(...)`）
- 顶部 `import os` 与 `import math as _math` 保留；它们仍分别用于 `os.PathLike` 和 `_math` 计算路径。
- 验证：`python3 -m py_compile train/models.py`
- 指标：
  - LOC：`5886 -> 5873`
  - `def` / 函数数：`113 -> 112`
  - 最大函数长度：`1767 -> 1767`
  - `except Exception`：`98 -> 98`

### Step A2 — 先在类内给 `EventMotionModel.__init__` 建立显式职责块（低/中风险）

状态：**已完成（2026-04-14）**

目标：不急着搬文件，先把 `__init__` 的职责分段明确下来。

实施：

- 将 `train/models.py:593-1773` 按职责切成 file-local / method-local builder：
  - feature/config normalization
  - contact-plan module build
  - direct-pose module build
  - leg-routing metadata build
  - lambda / SO(3) aux head build
- 第一轮允许 helper 仍留在 `models.py`。

约束：

- 不改变任何 `__init__` 参数集合
- 不改变 buffer 名称 / `state_dict` key

验收门：

- `EventMotionModel.__init__` 长度明显下降
- `state_dict().keys()` 集合不变
- `LOC_after <= LOC_before`

回填结果：

- 已在 `models.py` 内收口为具名 builder / metadata helper：
  - `leg-routing metadata build`
  - `contact-plan module build`
  - `direct-pose module build`
  - `lambda / SO(3) aux head build`
- 第一轮仍保留在 `train/models.py`，没有进入跨文件功能拆分。
- 验证：`python3 -m py_compile train/models.py`
- 指标：
  - LOC：`5873 -> 5852`
  - `def` / 函数数：`112 -> 116`
  - 最大函数长度：`1767 -> 1767`
  - `except Exception`：`98 -> 98`
  - 主题结构债指标：`EventMotionModel.__init__` 长度 `1181 -> 574`

### Step A3 — 先把 runtime override / ablation 收口成具名 helper（中风险）

状态：**已完成（2026-04-14）**

目标：在 `forward` 大拆分前，先把最容易污染主路径的 runtime override 收口。

实施：

- 将 `direct_pose_plan_override` / `direct_pose_meas_override` / cross-leg ablation / side-plan-other ablation 等 runtime 分支收进具名 helper。
- 区分：
  - active model path
  - eval-only ablation path
  - debug fallback path

约束：

- 不改变默认行为
- override 未显式启用时，执行图必须与当前一致

验收门：

- `forward` 主干嵌套层级下降
- debug/ablation 逻辑不再散落在主路径多处

回填结果：

- 已收口为具名 helper：
  - `direct_pose_plan_override`
  - `direct_pose_meas_override`
  - `direct_pose_leg_side_plan_other_ablate`
  - `direct_pose_leg_cross_leg_ablate`
- 保持默认行为不变，仅把 eval/debug 分支从 `forward` 主路径中抽离。
- 验证：`python3 -m py_compile train/models.py`
- 指标：
  - LOC：`5852 -> 5842`
  - `def` / 函数数：`116 -> 121`
  - 最大函数长度：`1767 -> 1561`
  - `except Exception`：`98 -> 98`
  - 主题结构债指标：`EventMotionModel.forward` 长度 `1767 -> 1561`

---

## Phase B — 向现有模块抽离独立职责（B1 + B2 + B3 + B4）

### Step B1 — 将 ckpt compat 从模型类中抽回 `train/model_ckpt_compat.py`（中风险）

状态：**已完成（2026-04-14）**

目标：让 `EventMotionModel` 只保留模型定义，不直接承担大段 checkpoint 兼容升级。

实施：

- 将以下逻辑迁回或并入 `train/model_ckpt_compat.py`：
  - `_maybe_upgrade_direct_pose_split_state_dict`
  - `_maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict`
  - `adapt_legacy_state_dict_`
  - `load_state_dict` 内的 compat 预处理
- `EventMotionModel` 最多保留很薄的入口壳，或者完全删除薄壳改由 compat 模块外部调用。

约束：

- 不改变历史 ckpt 的可加载性
- 不改变 warning / failure 的基本语义

验收门：

- `model_ckpt_compat.py` 与 `models.py` 的职责重叠下降
- `EventMotionModel` 类跨度下降
- `LOC_after <= LOC_before`

回填结果：

- 已迁回 `train/model_ckpt_compat.py` 的主实现：
  - `_maybe_upgrade_direct_pose_split_state_dict`
  - `_maybe_upgrade_direct_pose_stepc_leg_terminal_state_dict`
  - `adapt_legacy_state_dict_`
  - `load_state_dict` compat 预处理
- `EventMotionModel` 当前仅保留薄入口壳，调用 compat 模块中的主实现。
- 同时将 `train/model_ckpt_compat.py` 对 `EventMotionModel` 的依赖改为 `TYPE_CHECKING` 路径，避免运行时循环导入。
- 验证：
  - `python3 -m py_compile train/models.py train/model_ckpt_compat.py`
  - import smoke：`import train.models`、`import train.model_ckpt_compat`
- 指标（按 B1 主题使用双文件汇报）：
  - `train/models.py` LOC：`5842 -> 5548`
  - `train/model_ckpt_compat.py` LOC：`1794 -> 2080`
  - 总 LOC：`7636 -> 7628`
  - 总 `def` / 函数数：`145 -> 149`
  - 最大函数长度：`1561 -> 1561`
  - 总 `except Exception`：`130 -> 130`
  - 主题结构债指标：`EventMotionModel` 类跨度 `3631 -> 3300`

### Step B2 — 将 bundle attach 逻辑从模型类抽回兼容/装配层（中风险）

目标：把 `attach_motion_encoder` 从模型主语义中剥离。

实施：

- 将 `train/models.py:4132-4218` 的 bundle 解析、维度推断、冻结安装逻辑挪到 `train/model_ckpt_compat.py`。
- `EventMotionModel` 若必须保留入口，只保留薄 wrapper，不保留主要实现。

约束：

- 不改变 `posttrain.py` / `training_MPL.py` 的现有调用语义
- 不改变 frozen module 的设备与 `requires_grad_(False)` 语义

验收门：

- `attach_motion_encoder` 主实现不再位于 `models.py`
- 相关调用点仍保持可读

执行回填（2026-04-14）：

- 状态：`done`
- 实际改动：
  - 新增 `train/model_ckpt_compat.py:340` `attach_motion_encoder_bundle(model, bundle, map_location=...)`，承载 bundle 解析、hint mode 校验、维度推断、frozen encoder/heads 安装与 `period_encoder` 补建。
  - `train/models.py:3793` `EventMotionModel.attach_motion_encoder(...)` 仅保留 1 行 wrapper，继续保持原对外调用入口。
  - 未改 `posttrain.py` / `training_MPL.py` / 其它调用点；`bundle` path/dict 双入口、`require_standard_rotvec_bundle(...)` 校验、设备迁移、`eval()`、`requires_grad_(False)`、`self.frozen_encoder` / `self.frozen_period_head` / `self.frozen_contact_head` 挂载语义保持不变。
- 最小验证：
  - `python3 -m py_compile train/models.py train/model_ckpt_compat.py`
  - `python3 - <<'PY'` / `import train.models` / `import train.model_ckpt_compat`
  - `python3 - <<'PY'` / `EventMotionModel(in_state_dim=1, out_motion_dim=1)` / `hasattr(model, 'attach_motion_encoder')`
- before / after 指标：
  - `train/models.py`：LOC `5548 -> 5462`；`def` `121 -> 121`；最大函数长度 `1561 -> 1561`；`except Exception` `92 -> 91`
  - `train/model_ckpt_compat.py`：LOC `2080 -> 2163`；`def` `28 -> 29`；最大函数长度 `223 -> 223`；`except Exception` `38 -> 39`
  - 双文件合计：LOC `7628 -> 7625`；`def` `149 -> 150`；最大函数长度 `1561 -> 1561`；`except Exception` `130 -> 130`
  - B2 主题指标：`attach_motion_encoder` 主实现是否仍由 `models.py` 承担：`no`
- 结论 / 下一步：
  - `B2` 达成验收门；本 step 保持 `LOC_after <= LOC_before`。
  - 下一推荐 step 保持为 `B3`：将 `_apply_so3_correction_to_delta_raw` 下沉到 `train/geometry.py`；除非在执行前发现外部调用面比当前观察更宽，需要先补一轮引用核对。

### Step B3 — 将纯几何修正 helper 下沉到 `train/geometry.py`（低/中风险）

目标：把 `_apply_so3_correction_to_delta_raw` 归回几何单一真源。

实施：

- 将 `train/models.py:180-215` 移至 `train/geometry.py`
- `training_MPL.py` 与 `models.py` 统一从 `geometry.py` 引用

约束：

- 不改数值路径
- 不改 columns / clamp / omega detach 语义

验收门：

- 几何 helper 不再公开挂在 `models.py`
- `training_MPL.py` 引用链变短

执行回填（2026-04-14）：

- 状态：`done`
- 实际改动：
  - 将 `_apply_so3_correction_to_delta_raw(...)` 从 `train/models.py` 移至 `train/geometry.py:403`，保持函数签名与内部数值路径不变。
  - `train/training_MPL.py` 改为直接从 `train/geometry.py` 导入该 helper；`train/models.py` 同时移除该 helper 的定义与公开导出。
  - 未改 `columns` fallback、`gate_val` / `max_deg` 逻辑、`omega_detach` 语义、SO(3) 修正数值路径。
- 最小验证：
  - `python3 -m py_compile train/models.py train/geometry.py train/training_MPL.py train/model_ckpt_compat.py`
  - `python3 - <<'PY'` / `import train.geometry` / `import train.models` / `import train.training_MPL`
  - `python3 - <<'PY'` / `assert not hasattr(train.models, '_apply_so3_correction_to_delta_raw')` / `assert hasattr(train.geometry, '_apply_so3_correction_to_delta_raw')`
  - `python3 - <<'PY'` / helper smoke：`omega_hat is None` 与 `gate_val == 0` 均保持原返回早退语义
- before / after 指标：
  - `train/models.py`：LOC `5462 -> 5419`；`def` `121 -> 120`；最大函数长度 `1561 -> 1561`；`except Exception` `91 -> 91`
  - `train/geometry.py`：LOC `1027 -> 1066`；`def` `33 -> 34`；最大函数长度 `74 -> 74`；`except Exception` `4 -> 4`
  - `train/training_MPL.py`：LOC `5201 -> 5201`；`def` `159 -> 159`；最大函数长度 `153 -> 153`；`except Exception` `53 -> 53`
  - 三文件合计：LOC `11690 -> 11686`；`def` `313 -> 313`；最大函数长度 `1561 -> 1561`；`except Exception` `148 -> 148`
  - B3 主题指标：
    - `_apply_so3_correction_to_delta_raw` 是否仍由 `models.py` 承担主实现：`no`
    - `training_MPL.py` 是否已直接从 `geometry.py` 引用该 helper：`yes`
- 结论 / 下一步：
  - `B3` 达成验收门；本 step 保持 `LOC_after <= LOC_before`。
  - 下一推荐 step 为 `B4`：将 `_build_pretrain_contact_encoder_input` 与 joint-spec 解析 helper 下沉到 `train/utils.py`。

### Step B4 — 将通用输入拼装 / joint-spec 解析下沉到 `train/utils.py`（低风险）

目标：把已被训练入口复用的 helper 挪到公共承载文件。

实施：

- 将 `_build_pretrain_contact_encoder_input` 移至 `train/utils.py`
- 将 `_normalize_joint_spec_items` / `_resolve_joint_spec_indices` 移至 `train/utils.py`
- `DEFAULT_DIRECT_POSE_LEG_BONES` / `STAGE6_3WAY_ARMCHAIN_BONES` 本轮可先保留在 `models.py`，避免额外 import churn

约束：

- 不新建 `joint_specs.py`
- 不为了“纯净”而增加额外文件

验收门：

- `models.py` 顶层 helper 数下降
- `training_MPL.py` 不再从 `models.py` 导入非模型语义 helper

执行回填（2026-04-14）：

- 状态：`done`
- 实际改动：
  - 将 `_normalize_joint_spec_items(...)`、`_resolve_joint_spec_indices(...)`、`_build_pretrain_contact_encoder_input(...)` 从 `train/models.py` 移至 `train/utils.py`。
  - `train/models.py` 改为从 `train/utils.py` 引用 `_resolve_joint_spec_indices` 与 `_build_pretrain_contact_encoder_input`，模型内部调用语义不变；当时 `_build_pretrain_contact_encoder_input` 继续通过 `train.models` 暴露，避免现有外部导入立刻失效。
  - `train/training_MPL.py` 改为直接从 `train/utils.py` 导入 `_build_pretrain_contact_encoder_input`，不再从 `models.py` 引入非模型语义 helper。
  - 后续跟进（`2026-04-18`）：`train/models.py` 中 `_build_pretrain_contact_encoder_input` 的 compat import / re-export 已删除，helper 现仅从 `train.utils` 暴露。
- 最小验证：
  - `python3 -m py_compile train/models.py train/utils.py train/training_MPL.py train/geometry.py train/model_ckpt_compat.py`
  - `python3 - <<'PY'` / `import train.utils` / `import train.models` / `import train.training_MPL`
  - `python3 - <<'PY'` / `assert train.models._resolve_joint_spec_indices is train.utils._resolve_joint_spec_indices`
  - 文本检查：`train/training_MPL.py` 的 `from .models import (...)` 已不再包含 `_build_pretrain_contact_encoder_input`
- before / after 指标：
  - `train/models.py`：LOC `5419 -> 5313`；`def` `120 -> 117`；最大函数长度 `1561 -> 1561`；`except Exception` `91 -> 90`
  - `train/utils.py`：LOC `620 -> 723`；`def` `35 -> 38`；最大函数长度 `42 -> 42`；`except Exception` `9 -> 10`
  - `train/training_MPL.py`：LOC `5201 -> 5201`；`def` `159 -> 159`；最大函数长度 `153 -> 153`；`except Exception` `53 -> 53`
  - 三文件合计：LOC `11240 -> 11237`；`def` `314 -> 314`；最大函数长度 `1561 -> 1561`；`except Exception` `153 -> 153`
  - B4 主题指标：
    - `models.py` 中这 3 个顶层共享 helper 定义数：`3 -> 0`
    - `training_MPL.py` 是否仍从 `models.py` 导入非模型语义 helper：`yes -> no`
- 结论 / 下一步：
  - `B4` 达成验收门；本 step 保持 `LOC_after <= LOC_before`。
  - 下一推荐 step 为 `C1`：将 `EventMotionModel.forward` 收敛为 orchestration 壳。

---

## Phase C — `EventMotionModel` 类内壳层化（C1 + C2 + C3）

### Step C1 — 将 `forward` 收敛为 orchestration 壳（高风险）

目标：将 `train/models.py:2364-4130` 从 1767 行压成“主编排 + 具名职责块”。

实施：

- 优先拆出以下私有 helper（仍在 `models.py` 内）：
  - `_prepare_forward_context`
  - `_forward_contact_plan_path`
  - `_forward_direct_pose_path`
  - `_forward_leg_residual_path`
  - `_forward_lambda_fusion_path`
  - `_forward_so3_corrector_path`
  - `_finalize_forward_result`

约束：

- 不改变输出 dict key 集合
- 不改变单步 / 序列两种 shape 语义
- 不改变 teacher / freerun / posttrain 现有调用约定

验收门：

- `EventMotionModel.forward` 显著下降
- 主要职责块可单独命名与 review
- `LOC_after <= LOC_before`

### Step C2 — 把 direct-pose leg routing / cross-leg ablation 继续压进专用 helper（中/高风险）

目标：收口 `forward` 中最密集、最易回归的一段逻辑。

实施：

- 将 `train/models.py:3433-3999` 左右的 leg residual / side routing / rank1 / sign gate / cross-leg ablation 进一步分层：
  - routed shared head path
  - non-routed leg head path
  - optional gating / scaling path
  - eval-only ablation patching path

约束：

- 不更改 direct leg 输出 schema
- 不更改 side routing / gate mode / scale mode 的兼容语义

验收门：

- `forward` 中 leg residual 主块厚度下降
- eval-only 逻辑与主路径边界更清晰

### Step C3 — 将 `__init__` 压成“配置归一化 + builder”两层（中风险）

目标：让 `EventMotionModel.__init__` 从“超长配置工厂”回到可读边界。

实施：

- 将 contact-plan、direct-pose、leg-route、lambda-fusion、SO(3) corrector 分别沉到 builder helper。
- 明确把“纯配置规范化”和“模块实例化”分开。

约束：

- 不改变参数默认值
- 不改变任何模块名 / buffer 名 / state_dict key

验收门：

- `EventMotionModel.__init__` 明显下降
- 主要 builder 命名后可被独立 review

---

## Phase D — `MotionJointLoss` 结构收口（当前轮次不新建 loss 文件）

### Step D1 — 先拆 `MotionJointLoss.__init__`，不拆文件（中风险）

目标：在不新增 `train/losses.py` 的前提下，先降低 loss 配置入口复杂度。

实施：

- 将 `train/models.py:4221-4414` 分成：
  - direct-pose supervision config
  - event/aux loss config
  - skeleton/meta/init cache
  - weighting/tail-loss config

约束：

- `MotionJointLoss` 仍保留在 `models.py`
- 不改变构造参数集合

验收门：

- `MotionJointLoss.__init__` 长度下降
- `forward` 调用语义不变

### Step D2 — 收口骨骼权重 / group-mask / direct-pose payload 的内部边界（中风险）

目标：把 `MotionJointLoss` 中仍然跨职责的计算块分层。

实施：

- 重点收口：
  - `_resolve_direct_group_masks`
  - `_compute_unified_weights_cpu`
  - `_compute_direct_pose_group_norm_payload`
  - `_compute_direct_pose_payload`
- 区分：
  - pure geometry/statistics
  - config-dependent weighting
  - stats payload assembly

约束：

- 不改变 loss 数学定义
- 不改变 stats key

验收门：

- 直接 supervision 与 skeleton weighting 边界更清晰
- helper 之间不再共享过多隐式状态

### Step D3 — 维持 `forward` 壳薄，但继续减少 helper 内部宽泛异常（中风险）

目标：把当前 loss 侧的“能 fail-fast 的地方”恢复成显式 guard。

实施：

- 审核 active path 的 `except Exception`
- 对 shape / dtype / key 缺失可预期场景用显式 guard
- 把 debug-only 标量化 fallback 限定在 stats 路径

验收门：

- `except Exception` 数下降
- 热路径错误不再被静默吞没

---

## Phase E — 验证与收尾

每个 step 的最小验证集：

- `python -m py_compile train/models.py train/model_ckpt_compat.py train/geometry.py train/utils.py`
- 关键 import smoke：
  - `train/training_MPL.py`
  - `train/posttrain.py`
  - `train/validate/run_freerun_cycles.py`
- 关键行为 smoke：
  - 历史 ckpt 仍可走 compat 预处理
  - encoder bundle 仍可 attach
  - `EventMotionModel.forward` 输出 key 不变
  - `MotionJointLoss` 输出 `(loss, stats)` 约定不变

每步汇报固定格式：

- LOC
- `def` / 函数数
- 最大函数长度
- `except Exception` 数量
- 主题结构债指标（至少 1 项，例如 `direct_pose_ token count` / `EventMotionModel.__init__` 长度 / `EventMotionModel.forward` 长度）

---

## 4) 当前推荐执行顺序（按收益 / 风险排序）

建议严格按下面顺序推进：

1. `[done]` `A1`：清掉 `models.py` 中段脚本残留 import
2. `[done]` `A2`：`EventMotionModel.__init__` 类内 builder 化第一轮
3. `[done]` `A3`：runtime override / ablation 具名 helper 收口
4. `[done]` `B1`：compat 逻辑迁回 `train/model_ckpt_compat.py`
5. `[done]` `B2`：`attach_motion_encoder` 迁回 `train/model_ckpt_compat.py`
6. `[done]` `B3`：SO(3) 修正 helper 迁到 `train/geometry.py`
7. `[done]` `B4`：通用输入拼装 / joint-spec helper 迁到 `train/utils.py`
8. `[next]` `C1`：`EventMotionModel.forward` 壳层化
9. `[pending]` `C3`：`EventMotionModel.__init__` builder 化剩余收口
10. `[pending]` `D1`：`MotionJointLoss.__init__` 拆段
11. `[pending]` `D2` + `D3`：loss 内部边界与异常边界收口

原因：

- 前 7 个边界步骤收益最高、语义最稳定，且完全符合“只复用现有模块、不新增文件”的约束。
- `forward` / `__init__` 的大拆分必须建立在 compat / bundle / geometry / util 已先归位的前提上。
- `MotionJointLoss` 本轮不急着独立成新文件，先在原地把结构压顺即可。

---

## 5) 一句话结论

本轮 `train/models.py` 的正确方向不是“先全量清理”也不是“先新建更多模块”，而是：

- **先做最小必要清理**
- **再把独立职责送回已有模块**
- **最后把 `EventMotionModel` / `MotionJointLoss` 压成清晰的 orchestration 壳**

并且整个过程严格遵守：**不新增 Python 文件，不改变模型语义，不扩大默认接口面。**
