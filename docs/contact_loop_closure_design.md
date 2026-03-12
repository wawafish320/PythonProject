# Contact Loop Closure & Stage2 λ Fusion（单一设计文档）

> Last updated: 2025-12-28  
> 本文档是 **contact-loop / SO(3) corrector / Stage2 λ fusion** 的唯一设计与诊断入口；旧的迭代/实验文档已合并（见本文“历史问题与修复”与“路线图”）。

---

## 0. TL;DR（最新结论：为什么 Round0“拉不回”）

**Stage2（λ）本质是“选择/插值”，不是“创造精度”。**

- 当某段区间 `direct` 的绝对误差 **系统性大于** `incremental`（典型：plan_z 冷启动导致 direct early 很差）时：
  - 任何 `λ_eff > 0` 的 on-manifold 融合都会把结果往更差的 direct 拉；
  - 所以 **Round0 的上限≈incremental 本身的误差**（最多做到“不被混坏”）。
- 你观察到“Round1 才开始救火”并不矛盾：
  - 当 `incremental` drift 后变得 **比 direct 更差**，融合才有“拉回空间”，观感就像 Round1 才开始生效。

因此当前优化优先级应是：
1) **提升 direct 的 early/round0 绝对精度**（否则 Round0 永远只能≈inc）  
2) **把 λ 从 time schedule 升级为 reliability gate**（在 direct 更差区间把 `λ_eff≈0`）  
3) 再做 per-joint warmup / monotonic / plan stability / disagreement 等稳态细化

---

## 1. 目标与失败模式

目标现象：
- Teacher forcing 单步误差小
- Free-run 长序列误差累积（drift），在 `freerun_cycles` 多轮循环下被放大（Round1/2 更差）

我们要的不是“把单帧做更准”，而是让系统具备：
- **短期精度**（early steps 不被破坏）
- **长期稳定**（多轮 drift 被压住）
- **可解释信号**（能诊断“谁在拖后腿：plan / meas / direct / λ / corrector”）

---

## 2. 组件总览（系统里有哪些“杠杆”）

### 2.1 两个专家（experts）

- **Incremental expert（inc / Δ 分支）**
  - 输出 `delta`（normalized）
  - 由 `compose(y_prev_raw, delta)` 得到 `y_inc_raw`
  - 特性：early 往往更准，但自回归 drift 会累积

- **Direct expert（dir / out_direct 分支）**
  - 输出 `out_direct`（normalized absolute prior）
  - 反归一化得到 `y_dir_raw`
  - 特性：不累积 drift，但绝对精度取决于 anchor/phase 信息量（很容易 early 偏）

> 代码位置：`train/models.py`（`direct_pose_head` 输入 = `cond + contacts_plan`）

### 2.2 Contact loop（plan / meas / err）

闭环的关键是构造“innovation-like”信号：
- `contacts_plan`：cond-only GRU 产生的独立锚点（不依赖姿态）
- `contacts_meas`：从当前姿态派生的观测/传感器（依赖姿态）
- `contacts_err = contacts_plan - contacts_meas`

如果 `contacts_meas` 退化成常数/均值解，那么 `contacts_err` 就会变成“偏置”，几乎不随 drift 变化 → 闭环就不“闭”。

> 当前默认推荐 **white-box meas**：`train/training_MPL.py:Trainer._contact_meas_whitebox()`  
> `contact_meas_head` 可选保留（轻量化/泛化/无 FK 部署），但需防均值解。

### 2.3 SO(3) corrector（短期纠偏）

在 `SO(3)` 上对 `ΔR_pred` 做小角度纠偏：

```
omega_hat = f(h_final, contacts_err)      # 小角度
ΔR_used = Exp(gate * omega_hat) @ ΔR_pred
```

注意：
- corrector 只影响旋转（`BoneRotations6D`），不会直接修 root translation
- 推理时硬开 gate 通常会爆炸：需要训练时 curriculum + 正则来支撑

### 2.4 Stage2：λ gate + on-manifold fusion（长期切换）

Stage2 的核心是：**rollout state 真正用融合后的姿态更新**（不是只算 direct 指标）。

在 `SO(3)` 上做 geodesic 插值（每 joint 独立）：

```
R_res = R_dir @ R_inc^T
ω = Log(R_res)
R_blend = Exp(λ_eff * ω) @ R_inc
```

其中 `λ_eff = λ * r_t`：
- `λ`：模型输出（sigmoid 后在 [0,1]）
- `r_t`：deterministic reliability factor（现实现：warmup + 可选 contacts_err）

> 代码位置：`train/training_MPL.py:Trainer._apply_lambda_fusion_to_raw()` + `Trainer._lambda_fusion_apply_reliability()`

---

## 3. 评估口径（避免被 JSON 误导）

推荐统一用 `train/validate/run_freerun_cycles.py`：

### 3.1 multi-cycle 的时间与 round 切片

- freerun_cycles 的 round 切片固定为 **cycle 内部 transition（len=cycle_len-1）**，自动跳过 wrap boundary（不再提供 round-seg-mode 开关）
- `--time-index-mode cycle`：多轮循环时给 model 的 `time_index` 用 `t % cycle_len`，避免 Round1 time-PE OOD（如果启用了 time-PE）

### 3.2 诊断 direct 是否真的准：请先关掉 apply

`--lambda_fusion_apply` 会把 blend 作为 rollout state → 会改变后续输入分布。

因此：
- 想判断 “direct 到底准不准 / inc 到底漂没漂”，必须 **先跑 `--lambda_fusion_apply` 关闭** 的版本；
- 想看“系统最终表现”，才开 apply。

---

## 4. 历史问题与修复（已合并到主线）

### 4.1 `so3_corr_apply` step0 爆炸（已修）

典型根因是把 “delta 表示” 当作 “absolute 6D rotation” 写回导致首帧炸裂；已修复（首帧回到 ~0.x° 量级）。

### 4.2 `contacts_meas_head` 均值解 → 白盒 meas（已默认）

当 meas 要隐式学习 “FK + foot evidence” 时很容易退化成常数输出，`contacts_err` 失真。

当前默认策略：
- rollout / freerun_cycles 使用 white-box meas（确定性、无拟合误差）
- ML meas head 仅作为可选项（部署/泛化需求出现时再做）

补充：white-box meas 的稳定性/连续性 TODO 见 `docs/contact_meas_whitebox_stability.md`（Walk_F step1 崩溃类问题优先从这里排查）。

### 4.3 time-PE 在 multi-cycle 下 OOD（需严格对齐）

如果 contact_plan 使用 time-PE 且 time_index 是 global t：
- 训练域里 time_index 通常只覆盖 `0..cycle_len-1`
- freerun_cycles Round1 会进入 `t>=cycle_len` → time-PE OOD → plan 跑偏

解决方案：
- multi-cycle 评估统一 `--time-index-mode cycle`

---

## 5. 2025-12-28 Walk_F 诊断：为什么 Round0 不会“拉回”

把“融合是否有效”与“direct 本身是否准”拆开看：

在 `--lambda_fusion_apply False` 的诊断中（同 ckpt、同 clip）可以直接看到两专家对比：

- 示例输出：`debug_output/freerun_cycles/_tmp_nofusion/Walk_F_freerun_cycles.json`
  - Round0（86 steps mean）：
    - `GeoLocalDeg(inc) = 13.38°`
    - `DirectGeoLocalDeg(dir) = 18.60°`（比 inc 差 5.21°）
  - Round1：
    - `GeoLocalDeg(inc) = 39.40°`（drift 爆炸）
    - `DirectGeoLocalDeg(dir) = 18.31°`（几乎不变）
  - Step0：
    - `GeoLocalDeg(inc) = 0.2306°`
    - `DirectGeoLocalDeg(dir) = 6.1055°`

解释：
- Round0/early：dir 绝对精度差 → 混合只会更差 → “不可能拉回”
- Round1：inc drift 后变差 → dir 反而更接近 GT → 融合才开始“救火”

结论：
- 如果目标是 “Round0 明显优于 inc”，那不是 λ 能解决的，必须提升 **direct 的 early 绝对精度**。

---

## 6. λ 的问题：从“时间 schedule”到“可靠性 gate”

### 6.1 为什么 warmup_steps=10 会伤 Round0

ckpt 里默认 `lambda_reliability_mode=warmup, warmup_steps=10` 的含义是：
- 只有 step0 `r_t=0`，从 step1 开始线性 ramp；
- 当 direct early 很差时，哪怕 `λ_eff` 很小也足够把 early 拉坏（Step0/1 的误差倍率太夸张）。

结论：warmup=10 只是“形式上保护 step0”，不等价于“保护整个 Round0”。

### 6.2 一个有效的工程策略：把 warmup 拉到一个 cycle 的量级

把 warmup_steps 设到 ≈`cycle_len`，相当于：
- Round0 主要走 inc（保住短期精度）
- Round1 开始逐渐允许 λ_eff 变大（用 dir 抑制 drift）

示例输出：`debug_output/freerun_cycles/_tmp_fusion_warmup87/Walk_F_freerun_cycles.json`

### 6.3 per-joint warmup scales：drift-based 会踩坑，reliability-based 更稳

已复现的坑：
- drift-based（按 inc early drift delta）会把“漂得快的关节”更早切到 direct
- 但若这些关节恰好是 direct early 最差的（腿/脚常见），等于“把最差 expert 更早加权上来” → Round0/early 变差

更稳的替代：reliability-based scales
- 核心约束：如果 `direct_early > inc_early`，该 joint **绝不加速**（scale>1）
- 可用启发式：

```
scale_j = clamp( (inc_early_j / direct_early_j) ** alpha, min_scale, max_scale )
```

脚本（从带 `--export_joint_geolocal` 的 freerun_cycles JSON 生成）：
- `tools/make_lambda_reliability_joint_scales.py`

用法：
```bash
python tools/make_lambda_reliability_joint_scales.py \
  --in_diag debug_output/freerun_cycles/<diag_dir>/Walk_F_freerun_cycles.json \
  --out    debug_output/freerun_cycles/<diag_dir>/Walk_F_joint_scales_reliability.json
```

注意：
- 当前实现里 per-joint warmup 会用 `r_w_base = idx/(K-1)`（未提前 clamp）乘以 scale，再 clamp 到 [0,1]；
- 当 scale<1 且 rollout 总步数不够长时，可能导致某些 joint 到最后都达不到 1（需要用 `LambdaRelMean`/per-joint 指标确认是不是你想要的行为）。

### 6.4 重要放大器：`--lambda_fusion_apply` 会改 rollout state

这不是“只改输出显示”，而是会把融合后的姿态喂回下一步：
- per-joint 不同速度切换会产生混合姿态分布（下肢更 direct，上肢更 inc）
- incremental 分支下一步看到 OOD 输入 → `GeoLocalDeg(inc)` 自身也可能上升

因此：
- 专家质量对比：关 apply
- 系统效果评估：开 apply

---

## 7. 真正瓶颈：Direct early 绝对精度（需要 phase/anchor 消歧义）

### 7.1 是否必须用“显式 phase / clip 绝对位置”这类强先验？

不一定要用 **clip 绝对位置** 这种强先验，但必须解决一个事实：

> 在 locomotion 等场景，仅凭 `cond`（甚至加上弱 contacts_plan）去预测“下一帧绝对姿态”往往是多模态：同样速度/方向指令，对应不同相位（左脚/右脚起步）。  
> 若没有足够信息消歧义，direct 会学成“平均相位” → early 必差。

泛化风险从低到高的输入方案：
1) **观测驱动的 phase/anchor（推荐）**：从 `pose_history / contacts_meas / foot state` 估一个起步态/相位，作为 direct 输入或用于初始化 plan_z  
2) **弱时钟（相对）**：`rollout_step_norm`、或 multi-cycle 下的 `t % cycle_len`（只在明确周期任务里用）  
3) **强先验（风险高）**：clip 绝对时间/绝对帧号/固定相位标签（最容易按时间背答案，变速/非周期会崩）

### 7.2 “A 的天花板≈inc”是结构决定的

当 Round0 `dir > inc` 时，Stage2 的最优策略就是 `λ_eff≈0`，所以 Round0 最好也只能到 inc 的水平；要突破必须让 dir 在 Round0 的某些区间/关节变得不劣于 inc。

---

## 8. 建议的迭代路线（按收益/风险排序）

### 8.1 先把评估与输入域固定住（避免无效 ablation）

- freerun_cycles 统一：
  - round 切片固定为 **cycle 内部 transition（len=cycle_len-1）**
  - multi-cycle 时 `--time-index-mode cycle`
- meas 统一 white-box（先保证闭环信号“能用”）

### 8.2 提升 direct early：优先做“观测驱动的 anchor”

低风险优先级建议：
- 用 rollout 状态（`pose_history/contacts_meas`）构造起步/相位提示，注入：
  - 初始化 `plan_z`（比纯 learnable init 更不欠定）
  - 或作为 direct_pose_head 的额外输入（比塞 clip 绝对 time 更稳）

### 8.3 训练 λ：把它从 schedule 变成 reliability gate

posttrain（冻结两专家，只训 λ）逐步 ablation：
1) `uniform + 长 horizon（≥cycle_len-1）`：先学出 early→late 上升趋势
2) 加 early 保护（`lambda_fusion_early_*`）与 monotonic（`lambda_fusion_monotonic_weight`）
3) 加 plan stability 惩罚（`lambda_plan_entropy_weight / lambda_plan_dyn_weight`）
4) 再考虑 disagreement `||ω_res||`：优先 stop-grad 或 lagged，避免鸡生蛋投机解

### 8.4 SO(3) corrector：在闭环信号可信后再做 curriculum

不要用“推理时硬开 gate”替代训练；应该在 posttrain 中用 warmup/正则逐步提高纠偏幅度。

---

## 9. Quickstart：一套命令复现实验与诊断

### 9.1 诊断两专家（关 apply）

```bash
PYTHONPATH=. python train/validate/run_freerun_cycles.py \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model  <CKPT.pth> \
  --rounds 2 \
  --out debug_output/freerun_cycles/<diag_dir> --force
```

### 9.2 验证融合（开 apply + 调整 warmup）

```bash
PYTHONPATH=. python train/validate/run_freerun_cycles.py \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model  <CKPT.pth> \
  --rounds 2 \
  --lambda_fusion_apply \
  --lambda_reliability_mode warmup --lambda_reliability_warmup_steps 87 \
  --out debug_output/freerun_cycles/<out_dir> --force
```

### 9.3 导出 per-joint GeoLocal（用于生成 per-joint warmup scales）

```bash
PYTHONPATH=. python train/validate/run_freerun_cycles.py \
  --teacher validate/teacher_batches/Walk_F_teacher.json \
  --model  <CKPT.pth> \
  --rounds 2 \
  --lambda_fusion_apply \
  --export_joint_geolocal \
  --out debug_output/freerun_cycles/<diag_dir> --force
```

---

## 10. 验收指标（看什么，避免被均值骗）

- 专家质量（关 apply 看）：
  - `GeoLocalDeg` vs `DirectGeoLocalDeg`
  - “direct 是否在 drift 区间真的更好”
- 系统效果（开 apply 看）：
  - `BlendGeoLocalDeg` 的 Round1 drift 是否显著下降
  - Round0 是否被拉坏（若被拉坏，优先延长 warmup / 压 λ_eff）
- λ 行为是否合理：
  - `LambdaMean/LambdaStd`、`LambdaRelMean`、`LambdaEffMean`
  - 分段相关性：`corr(step, λ_eff)`、`corr(ContactErrAbsMean, λ_eff)`（按 round/按 early-mid-late 分段）

---

## 11. 配置/开关速查（把被删掉的“散落知识”收敛到这里）

### 11.1 `freerun_cycles`（`train/validate/run_freerun_cycles.py`）

- 专家诊断（不改 rollout state）：不要传 `--lambda_fusion_apply`
- 系统效果评估（会改 rollout state）：加 `--lambda_fusion_apply`
- multi-cycle + time-PE：优先 `--time-index-mode cycle`（round 切片固定为 intra-cycle）
- per-joint 统计导出：`--export_joint_geolocal`（输出 `per_joint_geolocal` 字段）

### 11.2 Reliability factor `r_t`（`λ_eff = λ * r_t`）

`r_t` 当前在 `Trainer._lambda_fusion_apply_reliability()` 里实现，核心开关：

- `lambda_reliability_mode`：
  - `none`：不启用（`λ_eff = λ`）
  - `warmup`：按 step ramp `r_t: 0→1`
  - `contacts_err`：按 `contacts_err` 大小调制（需要模型输出 `contacts_err`）
  - `warmup+contacts_err`：两者相乘
- `lambda_reliability_warmup_steps`：warmup 的 K
- `lambda_reliability_contact_err_max`：contacts_err 的归一化尺度（`r=clamp(1-err/max,0,1)`）
- `lambda_reliability_warmup_joint_scales`：
  - 支持 JSON list（长度=J）或 JSON 文件路径
  - 也支持 dict：`{"scales":[...], "meta":{...}}`（本 repo 的生成脚本就是这个格式）

### 11.3 Stage2 λ 的训练入口（推荐 posttrain 形态）

当前最稳妥的 Stage2 训练形态是：**从 Stage1 ckpt 初始化，冻结两专家，只训练 λ head**（避免把 direct/Δ 一起训坏）。

对应入口：`train/posttrain.py`（模式由 config 控制）：
- `train_lambda_head=true`：训练 λ head（Stage2）
- `train_so3_corrector=true`：训练 corrector（短期纠偏）
- `train_contact_plan_init=true`：训练 `contact_plan_init_z`（缓解 plan_z 冷启动）

训练时常用的两个“形状”开关（决定 λ 学成什么样）：
- `lambda_time_weight_mode`（posttrain 里对 rollout loss 的 step 权重）：
  - `uniform`：更容易学出“late 更偏 direct”的趋势（但会伤 early，需要再加 early 保护）
  - `inv`：强保护 early（但如果 horizon 不够长，λ 容易塌成很小的常数）
- `lambda_fusion_use_rollout_step=true`：让 λ head 显式看到 `rollout_step_norm=t/(H-1)`（否则它很难学出时序变化）

### 11.4 direct early 精度提升（你需要的不是“更强 λ”，而是“更准 direct”）

建议优先从“低风险、泛化更稳”的方向加信息量：
- 让 direct 能看到 **观测驱动的 anchor**（`pose_history / contacts_meas / foot state` → phase/起步态提示）
- 或先把 `contacts_plan` 做到真正稳定且可泛化（multi-cycle 时域对齐 + 更可靠的初始化/输入）

代码落地（本 repo 已实现）：

- `EventMotionModel.contact_plan_init_mode=learnable+obs`：cold-start 时用 `plan_z0 = init_z + init_head(obs0)`  
  其中 `obs0 = [contacts_meas0, angvel0, pose_history0]`（都来自当前可观测，不需要显式 global phase / clip 绝对位置）。
- 对应训练开关：
  - 主训练：`--contact_plan_init_mode learnable+obs --contact_plan_init_hidden 128`
  - posttrain：`--train_contact_plan_init true --contact_plan_init_mode learnable+obs`

同时建议保留一个硬约束意识：只要 `dir > inc` 的区间还很长，Round0 就不可能靠融合变得比 inc 更好。
