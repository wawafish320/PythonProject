# White-box `contacts_meas` 稳定性分析与改进 TODO（Walk_F case）

> Last updated: 2025-12-28  
> 目标：把 `contacts_meas` 从“会离散翻车的伪观测”提升为“可连续使用的观测信号”，让 `contacts_err = plan - meas` 真正接近 innovation，从而支撑 Stage2 `r_t`/闭环诊断。
> Update (2026-03-09): `whitebox` runtime/validate lane 与 `--log_contacts_whitebox*` 调试开关已从当前 mainline 退休。本文仅保留作历史分析记录；下面的命令和代码路径需要在历史快照上复现，不能再当作当前执行建议。

## TL;DR（先做什么）

- **先定位**：确认 step1 归零是 `hit_flag` 翻转还是 `ground_z` 记忆下沉（而不是渐进 drift）。  
- **再 ablation**：优先跑 `gate_by_hit=False`（去掉离散门控；freerun 可用 `--contact_meas_gate_by_hit false`）看是否立刻稳定；再改 `ground_z` 更新策略（EMA/限速/滑窗）。  
- **补充（推荐）**：如果你的序列存在明显的全局平移（walk/run/basketball），white-box 用 **root-relative planar speed** 更稳：`--contact_meas_vxy_mode root_rel`。它能避免“脚在支撑相位随 root 平移也被当成高速滑动”导致的 vxy gate 误杀。  
- **验收口径**：`ContactMeasAbsMean` 不应出现单步 0.4→0.0 跳变；`ContactMeasGtAbsMean` 不应从 ~0.02 突然跃迁到 ~0.45+。

---

## 0. 背景：Walk_F 诊断里发生了什么

在 `--lambda_fusion_apply False`（不改 rollout state、只看两专家质量）的 freerun 诊断里，出现了两类问题：

1) **direct early 很差（Round0 没锚点）**
- `step0`: `GeoLocalDeg(inc)≈0.319°`，`DirectGeoLocalDeg≈5.58°`
- `first10`: `DirectGeoLocalDeg` 远大于 `GeoLocalDeg`，`direct<=inc` 覆盖率≈0

2) **`contacts_meas` 从 step1 开始崩（导致 `contacts_err` 失真）**
- `step0`: `ContactMeasGtAbsMean≈0.026`（meas 对 GT 很准）
- `step1`: `ContactMeasAbsMean≈0.012` 但 `ContactGTAbsMean≈0.466`（meas 接近 0、与 GT 反相）
- 后续 step：`ContactMeasGtAbsMean` 继续很大（≈0.6~0.85）

此时 `contacts_err = contacts_plan - contacts_meas` 变成了“传感器故障差”，而不是 drift innovation；用它做 `r_t(contacts_err)` 或闭环诊断都会误导。

---

## 1. 系统语义澄清：你以为的“meas 不递归”，在当前实现里并不成立

### 1.1 `contacts_plan`：确实是 pose-independent prior（不会直接被 drift 污染）

当前 `EventMotionModel` 的 `contact_plan_cell` 是 `GRUCell(cond_dim, h_plan)`：

- 递推：`plan_z(t) = GRU(cond(t), plan_z(t-1))`
- 输出：`contacts_plan(t) = sigmoid(head(plan_z(t)))`

代码：`train/models.py`（plan 分支在 forward 的 ~712 行之后）。

### 1.2 `contacts_meas`：即使是 MLP，也会在 free-run 中变成“整体递归系统”

在 freerun 评估/诊断脚本里，`contacts_meas` 默认走 **white-box override**：

- `run_freerun_cycles` 构建 model 时：当 `contact_plan_enable=True` **或** `contact_plan_inject != "none"` 时，会设置 `contacts_as_meas_override=True`  
  代码：`train/validate/run_freerun_cycles.py:288-337`（推断 plan/inject）与 `train/validate/run_freerun_cycles.py:392-430`（实例化 `EventMotionModel`）
- 每步把 `Trainer._contact_meas_whitebox(motion_raw, prev_foot_pos_meas)` 的输出作为 `contacts` 喂进 model  
  代码：`train/validate/run_freerun_cycles.py:1008`
- model forward 内部会用 `contacts_input` 直接覆盖 `contacts_meas`（而不是用 `contact_meas_head`）  
  代码：`train/models.py:1025`

而 white-box 本身带 **状态**：
- `prev_foot_pos`（用于速度）
- `Trainer._contact_meas_ground_z`（地面估计记忆）

代码：`train/training_MPL.py:195` 起。

此外，即便不用 white-box override，ML `contact_meas_head` 也吃 `pose_history`；而 `pose_history` 在 free-run 中由预测姿态每步滚动更新（buffer），因此依然会随 drift 递归累积。

**结论**：当前 pipeline 下，`contacts_meas` 并不是“纯 feed-forward、不累积”的观测；它在系统层面会递归，并且对离散门控/地面估计非常敏感。

### 1.3 White-box `contacts_meas` 当前实现（公式/单位/阈值，对齐用）

实现入口：`Trainer._contact_meas_whitebox`（`train/training_MPL.py:195`）。输出 `(B, C)` 的 `contacts_meas`，并缓存 `prev_foot_pos` 与 `self._contact_meas_ground_z`。

**(a) 观测构建（rot6d→FK→脚端状态）**

- `rot6d -> reproject -> FK -> foot_pos`  
- `vel = (foot_pos - prev_foot_pos) * fps`（m/s）
- `vxy_mps = ||vel_xy||`，`vz_mps = |vel_z|`（up_axis 可配置，默认 z）

**(b) 地面估计（当前是 “min 记忆”）**

- `bottom_z = foot_pos[z] - radius_m`（球底部高度）  
- `ground_z_now = min_C(bottom_z)`（每 batch 一条 ground）  
- `ground_z = min(ground_z_prev, ground_z_now)`（当 `prev_foot_pos` 有效时）  
  代码：`train/training_MPL.py:330-342`

这会让 `ground_z` **单调不升**，一次异常穿地/抖动会把地面估计永久拉低（对应本文 H2）。

**(c) Sweep hit gate（离散门控）**

当 `gate_by_hit=True` 时：

- `start_z = foot_pos[z] + up_offset_m`  
- `sweep_target_z = ground_z + radius_m`（命中条件看球心是否穿越该平面）  
- `hit_flag = (start_z >= target) & ((start_z - down_distance_m) <= target)`  
  代码：`train/training_MPL.py:346-355`

最终 `contacts_meas *= hit_flag`（对应本文 H1）：

- 代码：`train/training_MPL.py:397-398`

**(d) Soft score（cm / cmps）**

把几何/速度变成可解释的连续 score（阈值以下不惩罚）：

- `dist_cm = clamp(bottom_z - ground_z, 0, +inf) * 100`  
- `vz_cmps = |vz_mps| * 100`，`vxy_cmps = vxy_mps * 100`  
- `dist_score = sigmoid(alpha_dist*(dist0_cm - dist_cm)/dist0_cm) / sigmoid(alpha_dist)`  
- `vz_score = exp(-relu(vz_cmps - vz0_cmps) / (alpha_vz * vz0_cmps))`  
- `vxy_score = exp(-relu(vxy_cmps - vxy0_cmps) / (alpha_vxy * vxy0_cmps))`  
- `contacts_meas = clamp(scale * dist_score * vz_score * vxy_score * hit_flag, 0, max_score)`  
  并对 hit 的部分做 `min_score` 下界（避免接近 0 的抖动）。  
  代码：`train/training_MPL.py:356-407`

**(e) 参数来源**

- `meta['foot_evidence']['sweep']`：`sphere_radius_cm / up_offset_cm / down_distance_cm`
- `meta['foot_evidence']['soft_score_spec']`：`dist0_cm / alpha_dist / vz0_cmps / alpha_vz / vxy0_cmps / alpha_vxy / gate_by_hit`

此外，当前实现还内置了经验值（若想做严格 ablation，建议也外置化/记录到日志）：  
`min_score=1e-4, max_score=0.9, scale=0.92`（`train/training_MPL.py:300-305`）

### 1.4 freerun_cycles 的 contact 指标与复现方式（对齐口径）

`run_freerun_cycles --log_contacts` 会在 per-step JSON 里记录（`train/validate/run_freerun_cycles.py:1096-1170`）：

- `ContactGT*`：teacher/dataset 的 soft contacts（对齐到当前 free-run timeline）
- `ContactPlan*`：`contacts_plan` 的均值/绝对均值
- `ContactMeas*`：`contacts_meas` 的均值/绝对均值（当启用 override 时就是 white-box 输出）
- `ContactErrAbs*`：`|contacts_plan - contacts_meas|` 的均值，以及 per-channel mean abs
- `Contact*GtAbs*`：plan/meas 相对 GT 的 mean abs（用于检查 meas head/white-box 是否“坏了”）

最小复现示例（按你实际 teacher / ckpt 路径替换）：

```bash
python -m train.validate.run_freerun_cycles \
  --teacher validate/teacher_batches/*.json \
  --model output_models/xxx.pth \
  --rounds 5 \
  --log_contacts
```

如果要把 white-box 的中间量也写进 JSON（用于定位 `hit_flag` 翻转/`ground_z` 漂移），加上：

```bash
  --log_contacts_whitebox \
  --log_contacts_whitebox_first_steps 4
```

此时输出 JSON 的 `metrics_per_step[*]`（每步条目）里会出现 `ContactMeasWhitebox`，包含 per-foot 的 `dist/vz/vxy/sweep` 等均值统计。

---

## 2. 根因假设（优先排查顺序）

你观察到的“step1 瞬间塌到接近 0”形态，非常像 white-box 的离散门控触发，而不是渐进 drift。

### H1：`hit_flag` hard gate 导致 0/1 跳变

white-box 里会把 soft score 乘上 `hit_flag`：

- `contacts_meas = contacts_meas * hit_flag`  
  代码：`train/training_MPL.py:397-398`

当 `start_z / ground_z / down_dist / radius` 任一项略有偏差，`hit_flag` 会直接从 True 变 False，导致 `contacts_meas` 从 ~0.4 量级瞬间归零。

### H2：`ground_z` 更新策略（min 记忆）导致地面下沉/漂移

white-box 当前会把 ground 设成 `min(prev, now)` 一类的保守记忆：

- 一次异常（脚瞬间穿地/pose 抖动）会把 ground 拉得过低；
- 之后 sweep 永远 hit 不到 → `hit_flag` 长时间 False；
- meas 长时间贴近 0，彻底失去观测意义。

代码：`train/training_MPL.py:330-343` 附近。

### H3：阈值/softness + clamp 过强，导致平台期“饱和”

即使不 hard gate，`min_score/max_score/clamp` 也会让 `contacts_meas` 平台期信息量过低，进而：
- `dc/dt` 不可靠（全 0）
- 事件检测（touchdown/toeoff）抖动或延迟

---

## 3. 改进目标（保持当前系统语义，不引入额外复杂度）

短期目标（为闭环信号服务）：
- **连续性**：`contacts_meas(t)` 不应因为一步小误差就从 0.4→0.0（至少应平滑退化）
- **鲁棒性**：对小的姿态噪声、脚速噪声、root yaw 重投影误差不敏感
- **可解释性**：能在日志里定位“哪个脚、哪个中间量导致崩溃”

非目标（暂不做）：
- 不引入 clip 绝对时间/绝对帧号
- 不把 plan 改成观测驱动递推（保持 `contacts_plan` 为 pose-independent prior）
- 不在测试阶段扩展到手（后续可按相同框架扩展）

---

## 4. TODO（按优先级，建议每步都可单独做 ablation）

### P0：把白盒 meas 的“离散翻车”定位成可复现的数值条件

直接用 `run_freerun_cycles --log_contacts_whitebox` 打开 white-box debug：

- `train/training_MPL.py:Trainer._contact_meas_whitebox` 会把中间量写到 `Trainer._contact_meas_whitebox_debug`
- `train/validate/run_freerun_cycles.py` 会把该 payload 选择性附加到 per-step JSON 的 `ContactMeasWhitebox`
  - 默认：前 `--log_contacts_whitebox_first_steps` 步必打
  - 额外：当检测到疑似“meas 归零但 GT 很大”的崩溃形态时也会打

验收：能解释“为什么 step1 hit_flag 变了 / ground_z 跳了 / 速度门控触发了”。

### P1：移除/软化 hard gate（`hit_flag`）或引入 hysteresis

选一个低风险切入点做 ablation：

- 方案 A：提供一个开关让 `gate_by_hit=False`（只用连续 score；freerun 可用 `--contact_meas_gate_by_hit false`）
- 方案 B：把 `hit_flag` 从 bool 变成 soft gate（例如基于 `start_z - sweep_target_z` 的 sigmoid）
- 方案 C：引入滞回：hit 的进入/退出用不同阈值，避免抖动

验收：`ContactMeasAbsMean` 不再出现 step 级别的“归零跳变”，`ContactMeasGtAbsMean` 不再从 ~0.02 突然变 ~0.45+。

### P2：改造 `ground_z` 更新为“限速/稳健”估计

优先用最易控的工程策略：

- EMA：`ground_z = (1-β)*ground_z + β*ground_z_now`（β 小）
- 限速：每步 `Δground_z` clamp（避免单步异常拉爆）
- 稳健统计：用 per-foot 的低分位数/滑窗而不是 min

现已在 freerun 中提供可控开关（默认仍是 legacy 的 `min` 记忆，避免影响已有实验）：

```bash
# 1) 滑窗低分位数（推荐先试）：忽略单次“穿地”尖峰
python -m train.validate.run_freerun_cycles ... \
  --contact_meas_ground_z_mode window \
  --contact_meas_ground_z_window 5 \
  --contact_meas_ground_z_quantile 0.2

# 2) EMA：连续平滑跟踪 ground
python -m train.validate.run_freerun_cycles ... \
  --contact_meas_ground_z_mode ema \
  --contact_meas_ground_z_beta 0.05

# 3) 限速：限制 ground_z 每步最大上升/下降（cm/step）
python -m train.validate.run_freerun_cycles ... \
  --contact_meas_ground_z_mode window \
  --contact_meas_ground_z_slew_up_cm 0.2 \
  --contact_meas_ground_z_slew_down_cm 1.0
```

验收：长序列里 `ground_z` 不会单调下沉导致永久 miss。

### P3：保留“软接触≈相位 proxy”的同时，让边沿方向可辨识（为后续 direct early 消歧准备）

在不引入手/全身复杂度的前提下，先定义最小的 per-foot phase feature（不一定立刻接入网络，先在日志里验证稳定性）：

- `c(t)`（soft contact）
- `dc/dt`（差分/EMA 差分）
- `time_since_touchdown / time_since_toeoff`（基于阈值 + hysteresis 的事件计时器）
- 可选：`foot_height`、`vz`、`vxy`

验收：double-support 或平台期也能提供“相位推进”的信息，而不是全靠 GRU 猜。

---

## 5. 与 Stage2/闭环的关系（避免目标错位）

一旦 white-box meas 稳定：
- `contacts_err = plan - meas` 才有机会在 drift 发生时呈现“innovation-like”变化；
- `r_t(contacts_err)` 才能作为可靠性 gate，避免在 `dir>inc` 区间把 Round0 拉坏（详见主文档 `docs/contact_loop_closure_design.md` 的 2.2 / 6.2）。

在此之前，任何“让 plan 更观测驱动 / 让 direct 看更多 obs”的改动都可能把系统带入“用故障观测自洽”的闭环，诊断会更困难。
