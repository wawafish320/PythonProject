# 2026-03-08 `s180` promote 进步 / 后退窗口清单

## 目标

把当前 accepted `s180 low-LR trunkfull -> 71 -> 72 -> lambda final` 链路，相对两份基线的**进步点、后退点、热点窗口**单独整理出来，供后续做：

- mainline 文档更新；
- 回归 watchlist；
- 后续是否需要回切 `s60` / 引入 LR decay 的判断。

这里默认 sign 口径为：

- `candidate - baseline`
- 负数更好，正数更差

---

## 数据源

### 当前 accepted promote 链

- `70R` promote verdict：`debug_output/_tmp_70R_lowlr_trunkfull_s180_rounds5_20260308/s180_verdict.md`
- downstream chain verdict：`debug_output/_tmp_chain_s180promote_20260308/chain_verdict.md`
- final ckpt eval：`debug_output/_tmp_lambda_from_s180_72_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`

### 基线 A：accepted `new_fullchain_pretrain`（rounds=5）

- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305_r5/new_fullchain_pretrain/Walk_F_freerun_cycles.json`
- direct compare：`debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_direct/global_signal_summary.txt`
- blend compare：`debug_output/_tmp_chain_s180promote_20260308/compare_vs_accepted_r5_blend/summary_metrics.txt`

### 基线 B：`2026-03-07 eval_on` baseline

- `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_Walk_F_series/Walk_F_freerun_cycles.json`
- direct compare：`debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_direct/global_signal_summary.txt`
- blend compare：`debug_output/_tmp_chain_s180promote_20260308/compare_vs_evalon_20260307_blend/summary_metrics.txt`

---

## 一、相对旧 `new` chain：这次 promote 为什么可以通过

这是当前最重要的“accept / reject”口径。

- `71`: overall `legs_main=-0.065894`, `arms_main=-0.103466`; A=`-0.236356`, B=`-0.115982`
- `72`: overall `legs_main=-0.026292`, `arms_main=-0.103466`; A=`-0.236356`, B=`-0.115982`
- `lambda`（blend-aware）：`BlendGeoLocalDeg=-0.020124`, `GeoLocalDeg=-0.014627`, `DirectGeoLocalDeg=-0.032109`

**结论**：promote 后的 upstream 修复没有在 downstream 被洗掉，反而稳定传递到了 `71 -> 72 -> lambda`。

---

## 二、相对基线 A（accepted `new_fullchain_pretrain` r5）

### 2.1 明显进步

- overall direct mean：`0.147802 -> 0.112947`，`delta=-0.034855`
- `leg8_mean`：`0.313692 -> 0.274360`，`delta=-0.039333`
- `non_leg_mean`：`0.111934 -> 0.078048`，`delta=-0.033887`
- `improved_ratio=0.630103`
- `legs_main=-0.039333`, `arms_main=-0.098183`
- A window：`legs_main=-0.004081`, `arms_main=-0.059654`
- B window：`legs_main=-0.043974`, `arms_main=-0.108904`
- blend-aware：
  - `BlendGeoLocalDeg=-0.040034`
  - `GeoLocalDeg=-0.075716`
  - `DirectGeoLocalDeg=-0.034855`
- per-bone 代表性进步：
  - `lowerarm_l: -0.271776`
  - `upperarm_l: -0.129537`
  - `thumb_01_l: -0.128867`
  - `pinky_01_l: -0.111607`
  - `ball_l: -0.093798`
  - `thumb_01_r: -0.090921`

### 2.2 明显后退 / 需要继续盯

- `foot_l/ball_l` hotspot (`SIC12-15`)：
  - blend：`+0.248285`
  - direct：`+0.046298`
- `calf_r @ SIC2-4`：
  - blend：`+0.119355`
  - direct：`+0.132775`
- `calf_r @ SIC53-63`：`+0.036184`
- mean 层面的小回退：
  - `hand_l: +0.002727`
  - `RUpArmTwist_r_02: +0.002228`
  - `L_ForeTwist_02: +0.001471`

### 2.3 对这份基线的判定

- **可以明确判定为更好**；
- 唯一要保留的 watchlist 是：`foot_l/ball_l@SIC12-15` 与 `calf_r@SIC2-4`。

---

## 三、相对基线 B（`2026-03-07 eval_on`）

### 3.1 明显进步

- overall direct mean：`0.131316 -> 0.112947`，`delta=-0.018368`
- `leg8_mean`：`0.292003 -> 0.274360`，`delta=-0.017644`
- `non_leg_mean`：`0.096573 -> 0.078048`，`delta=-0.018525`
- `improved_ratio=0.615698`
- overall group：`legs_main=-0.017644`, `arms_main=-0.011768`
- lower-body 代表性进步：
  - `calf_r global: -0.047878`
  - `calf_r @ SIC53-63: -0.133938`
  - `thigh_r: -0.048562`
  - `foot_r: -0.059784`
- blend-aware：
  - `BlendGeoLocalDeg=-0.006143`
  - `BlendGeoLocalDegWeighted=-0.017571`
  - `GeoLocalDeg=-0.006118`
  - `DirectGeoLocalDeg=-0.018368`

### 3.2 明显后退 / 需要继续盯

这是当前最关键的 residual regression 清单：

- A window：`arms_main=+0.061042`
- B window：`arms_main=+0.047092`
- `foot_l/ball_l` hotspot (`SIC12-15`)：
  - blend：`+0.167161`
  - direct：`+0.024892`
- `calf_r @ SIC2-4`：
  - blend：`+0.219655`
  - direct：`+0.138344`
- mean 层面前几项回退：
  - `ball_r: +0.018133`
  - `hand_l: +0.011294`
  - `thigh_l: +0.011023`
  - `foot_l: +0.005115`

### 3.3 对这份基线的判定

- **整体 / global 已经赢了**；
- **但 upper-body 的 A/B hotspot 还没赢**；
- 因此它的正确定位是：`pass with watchlist`，不是“严格支配这份最强当前 baseline”。

---

## 四、当前保留的 watchlist

后续继续迭代时，优先盯这几个点：

1. `arms_main @ A_52_59`
2. `arms_main @ B_76_80`
3. `foot_l/ball_l @ SIC12-15`
4. `calf_r @ SIC2-4`
5. `ball_r / hand_l / thigh_l` 的 mean regression 是否会继续放大

建议口径：

- `70R` / `71` / `72` 阶段继续看 `DirectGeoLocalDeg + legs_main/arms_main/A/B`
- `lambda final` 阶段必须同时看：
  - `BlendGeoLocalDeg`
  - `GeoLocalDeg`
  - `DirectGeoLocalDeg`

---

## 五、当前建议

- 文档主线可以更新为：`promoted 70R (s180 low-LR trunkfull) -> 71 -> 72 -> lambda final`
- 这条线已经足以作为当前 accepted flow 通过
- 但不要把它表述成“已经严格优于所有 baseline”
- 更准确的表述应该是：
  - **明显优于旧 accepted baseline**
  - **整体优于 `2026-03-07 eval_on` baseline，但仍保留 upper-body A/B 与部分 hotspot 的 watchlist**
