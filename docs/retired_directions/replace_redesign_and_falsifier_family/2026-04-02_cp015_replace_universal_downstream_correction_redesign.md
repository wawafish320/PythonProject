# CP015 replace 重设计草案：从 donor-local refine 到 universal downstream correction

> Retired from active mainline use on 2026-04-12.
> This document is kept as historical redesign context / ruled-out direction evidence, not as the current posttrain canonical driver.
> Current replacement:
> - `docs/posttrain_pipeline.md`
> - `docs/train_design/2026-04-12_top7_clean_stage6_stepc_causality_record.md`

> Last updated: 2026-04-02
> 目标：明确当前 `replace` 路径为什么是结构性 donor/basin-sensitive，并提出一个更通用的 downstream correction 设计。
>
> 本文的重点不是“继续磨 current replace”，而是重新定义 `70a -> downstream` 的 stage interface。
> 同时，本文**不把“先做更多同产物 clip 实验”当作 redesign 前提**。当前虽然只有 `5` 个 clip，且都来自相近产物 family，无法充分覆盖跨产物分布；但不同产物/动画天然落入不同 basin 这一点，本身就是 first-principles 事实，已经足够支持把当前问题视为结构性接口问题，而不是单一 clip 偶发现象。

关联输入：

- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-03-19_stage70a_plain_handoff_design.md`
- `docs/retired_directions/legacy_upstream_handoff_control_family/2026-04-02_cp015_tailk7_basetrain_to_stage70a_baseline_record.md`
- `debug_output/_tmp_cp015_tailk7_replace_efficiency_audit_20260402_arm_efficiency_audit/summary.md`
- `debug_output/_tmp_cp015_tailk7_warmstart_contract_sentinel_20260402_warmstart_contract_sentinel/summary.md`
- `debug_output/_tmp_cp015_tailk7_exit_optizability_audit_20260402_exit_optizability_audit/summary.md`
- `train/models.py`
- `train/posttrain.py`

---

## 0) TL;DR

1. 当前 `replace` 的真实建模对象不是“通用 arm correction”，而是 **donor-local parameter-space continuation**。
2. 这使它天然依赖 donor 的局部 state geometry；一旦 donor basin 改变，即使 freerun 更好、梯度更大，下游 replace 仍可能显著失效。
3. 因此当前问题不应再理解为“再找一个更好的 replace case”，而应理解为：
   - `replace` 这一步的 **stage interface 设计有结构性问题**
   - 它把 donor basin 当作隐式前提
4. 新方向不应是“让 replace 更适应某些形态”，而应是：
   - **让 downstream correction 尽量不依赖 donor 形态**
   - 从 `weight-space continuation` 改成 `behavior-space frozen-donor correction`
5. 推荐的新角色定义是：
   - `70a`：继续作为 donor/base predictor
   - `replace v1`：不再是 donor-specific warmstart finetune，而是 **frozen-donor arm residual corrector**
6. 当前 `5` 个 clip 都来自相近产物 family，不能证明跨产物泛化；但 redesign 的逻辑不应建立在“必须先做更多同产物 clip 实验”上。
   - 若未来要补验证，优先级应是 **跨产物 / 跨 donor family**
   - 而不是继续堆同产物 clip 数量

---

## 1) 当前问题的准确表述

当前最值得固定的结论，不是：

- `tailk7 donor 不够好`
- 或 `replace recipe 还没调到位`

而是：

> 当前 `replace` 的成功条件隐含依赖 donor 的局部 basin。
> 这意味着它不是一个通用 downstream stage，而是一个只对特定 donor geometry 稳定的 local refiner。

从最新证据看：

1. `tailk7 70a` 本身的 freerun 已优于 baseline `70a`，因此问题不再是“谁的 `70a` 更好”。
2. `warmstart contract` 基本已排除：
   - 历史 baseline warmstart 只改两个 tensor
   - 只动 `39..42` 列
   - legacy `direct_pose_use_phase_z` / `direct_pose_phase_z_mode` 仍是 raw-config 可见但 parser-dead
3. 即使对 tailk7 donor 施加 baseline-style warmstart adaptation：
   - early local train-side efficiency 有部分改善
   - 但 step60 同口径 replace probe 仍明显落后 baseline
4. 本轮最小 `exit optizability audit` 进一步说明：
   - copy-only 不是“没梯度”
   - adapted 也不是“没局部改善”
   - 真正的问题是：**优化几何与 baseline 不同，且这种不同不会被 warmstart contract 自动修复**

这说明问题核心已经从：

- `replace 配置是否写对`

转为：

- `当前 replace stage 是否在错误的抽象层面上工作`

---

## 2) 为什么这是结构性设计问题，而不是 clip 数量问题

需要把两件事分开：

### 2.1 当前证据还不能回答什么

当前手里虽然有 `5` 个 clip，但它们都来自相近产物 family，因此**还不能**回答：

- 这个问题在真实跨产物分布上覆盖率有多高
- 哪些 product family 最容易触发这种 basin sensitivity

换句话说，当前证据对“问题分布有多广”仍是不充分的。

### 2.2 但当前证据已经足够回答什么

当前证据已经足够回答：

- 当前 `replace` 的成功前提依赖 donor basin
- 而 donor basin across products 本来就天然会不同

因此，**即使没有更多 clip，也已经足够把这视为结构性设计问题**。

这里的逻辑是 first-principles：

1. 不同产物 / 动画本来就会学到不同 exit basin
2. 如果 downstream stage 的成立条件是：
   - donor 恰好落在某类 basin
3. 那这个 downstream stage 就不是通用 stage，而是 conditional local repair

所以真正的问题不是：

- “我们现在是否已用足够多 clip 证明它一定普遍失败”

而是：

- “当前设计是否把一个不该依赖 donor basin 的接口，做成了 donor-basin-dependent 接口”

对这个问题，当前答案已经足够明确：**是的**。

---

## 3) 当前 replace 的结构性问题在哪里

## 3.1 它实际上是 weight-space continuation

从当前实现看，`replace` 的工作方式本质上是：

1. 从 donor ckpt warmstart
2. 继续沿 donor 已有的 `direct_pose_head / direct_pose_arm_proj / direct_pose_out_arm` 参数空间做局部优化
3. 期待小步更新能稳定把 arm objective 压下去，并传到 freerun

这意味着它优化的是：

> donor 已有 head geometry 附近的局部参数方向

而不是：

> 跨 donor 都成立的通用 arm failure mechanism

这就是当前最大的问题。

## 3.2 它把 donor state geometry 当成了隐式接口

如果 downstream 真的是通用 correction，那么它应该主要依赖：

- 当前行为状态
- 当前时序上下文
- 当前 arm failure pattern
- 当前 canonicalized feature

而不是依赖：

- donor head 权重长成什么样
- donor trunk feature 当前的尺度/方向恰好是什么
- donor 局部梯度在这个参数坐标里如何解释

现在的 `replace` 恰恰相反。它的成功与否，高度依赖：

- donor 进入 step0 时，局部 feature geometry 是否“像 baseline”
- 小步 trunk/arm update 是否仍能对应到相似的 arm 行为改善

这就是典型的 **stage interface leak**：

- 上游 donor 的内部几何，被 downstream 当成了外部 contract 的一部分

## 3.3 它更像 local adapter，而不是 universal corrector

当前 `replace` 若继续存在，最准确的角色描述应是：

- `donor-specific local adapter`

而不是：

- `universal downstream corrector`

这一区分非常重要，因为它直接决定了后续设计方向：

- 如果承认它只是 local adapter，那么继续磨它只是延长一条 donor-specific 路线
- 如果目标是通用性，就必须换抽象层，不再让它以 donor parameter neighborhood 为主要工作空间

---

## 4) 新的设计目标

新的 downstream correction 设计，应明确满足以下目标。

| 维度 | 当前 replace | v1 新目标 |
|---|---|---|
| 依赖对象 | donor 参数局部形态 | behavior-space observable |
| 优化空间 | weight-space continuation | task-space / behavior-space correction |
| 初始化 | donor warmstart | fresh init, zero-correction start |
| 训练目标 | 修当前 donor 的 head | 学 donor-robust arm correction rule |
| 稳定性前提 | donor basin 接近 baseline | donor basin 可变，但输入尽量来自标准化行为量 |
| stage 角色 | local refine | frozen-donor residual corrector |

因此 v1 的核心不是：

- “怎样让 replace 更适应某类 donor 形态”

而是：

- “怎样让 downstream correction 尽量不把 donor 内部形态当作输入”

---

## 5) 核心重设计方向

## 5.1 v1 不引入 canonicalizer，直接跳到 behavior-space residual correction

上一版草案里提出了：

- `canonical handoff bundle`
- `ReplaceCanonicalizer`
- `universal corrector`

这个方向在长期上仍成立，但对当前信息条件来说，v1 偏重。原因是：

1. 当前最强证据已经说明 donor 内部几何差异很大：
   - baseline vs tailk7 的 local gradient cosine 近似为 `0`
   - trunk `~0.017`
   - arm `~0.056`
2. 这意味着 donor hidden/trunk/branch feature subspace 近乎正交。
3. 在只有一个 product family、且只有 `5` 个 clip 的前提下，去学一个能把这些 donor-internal subspace 统一起来的 `canonicalizer`，风险很高。
4. `LayerNorm` / affine projection 最多能消 scale/shift，不能天然消去方向差异。

因此 v1 更合理的设计不是：

`donor internal feature -> canonicalizer -> corrector`

而是直接：

`behavior-space observable -> residual corrector`

也就是：

> v1 跳过 canonicalizer，不消费 donor 内部 feature，只消费任务空间 / 行为空间量。

## 5.2 从 donor continuation 改成 frozen-donor behavior-space correction

建议把当前链路：

`70a donor ckpt -> replace warmstart -> 继续调 donor head`

改成：

`frozen donor -> base arm prediction + behavior obs -> ArmResidualCorrector -> delta_so3`

其中：

- donor 全模型冻结
- corrector 全新初始化，不从 donor warmstart
- corrector 最后一层 zero-init，使初始行为 = identity correction
- downstream 学的是：
  - “看到这种 arm base prediction + contact/phase/event 状态，该输出怎样的 on-manifold arm residual”

而不是：
  - “在 donor 现有 head 权重附近，再往哪里推”

## 5.3 为什么这条路线更适合 v1

它解决的是和完整版方案同一个核心问题：

- 不再把 donor parameter geometry 当成主要变量

但工程上更轻：

- 不需要 `build_replace_handoff_bundle`
- 不需要 `ReplaceCanonicalizer`
- 不需要 bundle schema / versioning 先铺开
- 不需要在 donor internal feature 上定义稳定 contract

同时它更符合当前证据约束：

- 当前已经知道 donor internal geometry 差异巨大
- 但 arm failure 在行为空间上是可观测的
- baseline replace 能把 arm p95 从 `~0.91` 压到 `~0.42`
- 这说明“要不要修 / 怎么修 arm”很可能主要在 behavior-space 可见

---

## 6) 推荐的 v1 接口定义

## 6.1 输入只使用 behavior-space observable

v1 不建议让 corrector 读取 donor 内部特征，例如：

- `shared_feature`
- `arm_feature`
- donor trunk hidden

因为这些量当前没有证据表明对跨 donor 是稳定可解释的。

v1 推荐输入应全部来自模型输出或标准化观测量：

1. `arm_base_rot6d`
   - donor 当前 arm base prediction
   - 这是任务空间量，本身比 donor 内部 feature 更 donor-robust
2. `contacts_plan`
3. `contacts_meas`
4. `contacts_err`
5. `event_clock_delta_meas`
6. `event_clock_lambda_corr`
7. `event_clock_delta_z`
8. `phase/event-related normalized obs`
9. 可选：短窗 arm 自身 delta / consistency proxy

这里的原则不是“这些量绝对 invariant”，而是：

> 它们比 donor hidden / head feature 更接近共享语义空间，更适合作为 v1 的通用 correction 输入。

## 6.2 输出只预测 arm task-space residual

v1 建议 corrector 只输出：

- `delta_so3`（每个 arm joint 一个 `omega`）

再在 arm slice 上做：

`R_final = exp(delta_so3) @ R(base_arm_rot6d)`

注意：

- 在当前实现里，`out_direct` 位于 normalized Y space
- 因此不建议直接在模型内部把 `out_direct` 原位做 SO(3) compose

更稳的方式是：

1. 模型输出 `arm_omega_hat`
2. `train/posttrain.py` 的 loss / rollout harness 在 arm slice 上做：
   - `denorm -> compose -> renorm`

这与现有 leg `so3` 路径的语义更一致，也更容易复用现有 rotation utility / loss 逻辑。

## 6.3 v2 fallback：只有在 behavior-space 不够时，才补最小 canonicalizer

若未来证明：

- 纯 behavior-space corrector 修正力不够
- 或跨 donor transfer 明显失败

再进入 v2：

- 只补最小 feature-side context
- 例如 `detach + LayerNorm` 之后的 trunk hidden
- 而不是一开始就铺完整 canonicalizer/bundle 基建

也就是说：

> `canonicalizer` 不是 v1 前提，而是 v2 fallback。

---

## 7) 训练与验收方式应如何改变

## 7.1 v1 的训练对象

第一版建议：

- donor 原有 direct 路径冻结
- 只训练 `ArmResidualCorrector`
- 不让 donor head / trunk 继续被改写

这样做的意义很直接：

- base prediction 分布稳定
- corrector 学的是 correction rule
- 不会退回 donor-local continuation

## 7.2 v1 的训练目标

建议 objective 至少包含两层：

1. `arm task-space correction loss`
   - 直接作用在 arm correction 后的输出上
2. `short-horizon rollout transfer loss`
   - 防止 train-side 改善不传到 freerun-side

如果这两层已经足够，不必在 v1 里先加 donor-invariance regularizer。

原因是：

- 当前最大风险不是 regularizer 不够
- 而是设计一开始就选错输入空间

## 7.3 v1 的验收标准

未来 v1 的验收应优先看：

1. `zero-corrector sanity`
   - zero-init corrector 挂上后，输出应与 donor 原始输出一致
2. single-donor correction power
   - 在 frozen tailk7 donor 上，arm p95 是否显著下降
3. cross-donor transfer
   - 同一个 corrector 挂到 baseline donor 上，是否仍能改善或至少不崩

当前不建议把“同产物多 clip 数量”当成首要 gate。  
未来若补验证，优先级也应是：

- 跨 donor / 跨 product family

而不是：

- 同产物内部继续加更多 clip

---

## 8) 对当前 `train/models.py` / `train/posttrain.py` 的具体改造建议

## 8.1 `train/models.py`

v1 只建议新增一个模块：

### `ArmResidualCorrector`

职责：

- 消费 `arm_base_rot6d + behavior obs`
- 输出 arm joint 的 `delta_so3`

建议接口：

```python
class ArmResidualCorrector(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, arm_joint_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, arm_joint_count * 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, arm_base_rot6d, contact_obs, phase_obs):
        x = torch.cat([arm_base_rot6d, contact_obs, phase_obs], dim=-1)
        return self.net(x)
```

更具体地说，模型侧需要：

1. 利用现有 `direct_pose_arm_out_idx` 拿到 arm slice
2. 导出 `arm_base_rot6d`
3. 导出或复用已有 behavior-space observables：
   - `contacts_plan`
   - `contacts_meas`
   - `contacts_err`
   - `event_clock_delta_meas`
   - `event_clock_lambda_corr`
   - `event_clock_delta_z`
4. 输出 `arm_omega_hat`

不建议在 v1 新增：

- `ReplaceCanonicalizer`
- `replace_handoff_bundle`
- donor internal feature export contract

## 8.2 `train/posttrain.py`

建议新增一个独立 train mode，例如：

- `arm_residual`

它的行为应是：

1. donor 全模型冻结
2. 只解冻 `arm_residual_corrector`
3. rollout/loss 时只在 arm slice 上应用 `omega_hat`
4. compose 放在 harness 内完成，而不是直接改模型中的 normalized `out_direct`

在实现上，最小改造点是：

- `_resolve_train_mode` 新增第三种 mode
- `_unfreeze_for_train_mode` 只 `_enable_modules(model, ("arm_residual_corrector",))`

这样不需要改动现有：

- `train_direct_pose`
- `train_lambda_head`
- 旧 replace 语义

也就是说，v1 可以作为一条平行实验 lane 存在，不污染旧路径。

## 8.3 配置层建议

比起新的 bundle schema，更建议先加最小 config：

```json
{
  "train_arm_residual": true,
  "arm_residual_hidden": 256,
  "arm_residual_mode": "so3",
  "arm_residual_zero_init": true,
  "arm_residual_use_donor_weight_continuation": false
}
```

其中最关键的语义是：

- `arm_residual_use_donor_weight_continuation=false`

它明确表示：

- 这条线不是 current replace 的 warmstart continuation
- 而是 frozen-donor behavior-space correction

---

## 9) 验证路径

建议按三步做，且每一步都有明确 pass/fail 判据。

## Step 1：Zero-corrector sanity

目标：

- 冻结 donor 全模型
- 挂一个 zero-init `ArmResidualCorrector`
- 验证 freerun 输出与 donor 原始输出一致

这一步只验证工程正确性，不验证修正力。

## Step 2：Single-donor train on tailk7

目标：

- 在 tailk7 `70a` donor 上冻结全模型
- 只训练 `ArmResidualCorrector`
- 用现有 arm objective 看 arm p95 能否明显下降

推荐判据分两档：

- `arm p95 < 0.55`：证明路线有明确修正力
- `arm p95 < 0.50`：强 pass

这样比直接拿 baseline replace 的 `~0.42` 当第一版硬门槛更稳，因为 frozen-donor residual 的自由度更小。

## Step 3：Cross-donor transfer

目标：

- 把在 tailk7 上训练好的 corrector 直接挂到 baseline donor 上
- 不重新训练
- 看 arm 指标是否也改善或至少不崩

解读：

- 若改善：说明 corrector 确实学到了 donor-robust 的 correction rule
- 若不改善：说明即使 behavior-space 输入也仍有 donor 分布差异
- 这时再考虑：
  - 增加 rollout-local signal
  - 或进入 v2 最小 feature-side fallback

---

## 10) 需要明确放弃的旧前提

后续文档和讨论里，建议明确停止使用以下前提：

1. “只要 donor freerun 更好，replace 应该自然更好”
2. “只要再找更好的 warmstart patch，就能解决 donor 差异”
3. “replace 本来就应该吸收 donor basin 差异”
4. “必须先铺 canonicalizer / bundle 基建，才能做通用 correction”
5. “当前 clip 还不够多，所以还不能说是结构问题”

其中第 4 和第 5 条尤其需要固定：

> v1 完全可以先走 behavior-space residual correction；  
> clip 覆盖不足只能限制对问题分布范围的估计，不能否定当前已暴露出的接口设计问题。

---

## 11) Historical redesign conclusion at the time

1. At the time, the preferred reading was that current `replace` should no longer be treated as a generalized downstream corrector.
2. 它暴露出来的是：
   - `70a -> downstream` 接口把 donor basin 当成隐式契约
3. 对于真正的通用 downstream correction，当时最合理的 v1 起点不是完整 `canonicalizer + bundle` 基建。
4. 当时更合理的 v1 proposal 是：
   - frozen donor
   - behavior-space observable input
   - freshly initialized `ArmResidualCorrector`
   - arm-only `delta_so3` correction
5. `canonicalizer` 不是被否定，而是应降级为：
   - 只有在 v1 已证明 behavior-space 不够时，才进入的 v2 fallback

Historical one-line summary:

> At the time, this memo argued that v1 did not need a full canonical bundle + canonicalizer first; it proposed behavior-space observable frozen-donor arm residual correction as the lowest-risk redesign hypothesis then available.
