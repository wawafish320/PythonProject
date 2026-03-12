# 2026-03-09 `direct / blend / lambda` pointwise hotspot 解释

## 1) 目的

这份文档回答的问题是：

- 在 `docs/Problems/active/2026-03-09_direct_blend_lambda_pointwise_snapshot.md` 里列出来的
  `p95 / p99 / max` 这些 step 点，
- 它们具体是由哪些 `bone/joint` 在拉高误差。

## 2) 口径说明

当前 `Walk_F` final eval artifact 里：

- 有 `per_step_direct_geolocal_deg`
- 没有 `per_step_blend_geolocal_deg`
- 也没有 `per_step_geolocal_deg`

所以这里的 joint-level hotspot 解释采用的是：

- 对于 `DirectGeoLocalDeg`：直接使用该 step 的 joint error 排名
- 对于 `BlendGeoLocalDeg / GeoLocalDeg / LambdaMean`：使用**同一个 step**上的 `direct` joint 排名来解释该时刻的 pose hotspot

也就是说：

- 这是 **same-step joint hotspot**
- 不是 blend/global 各自的独立 joint 分解

数据源：`debug_output/_tmp_lambda_from_s180_72_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`

---

## 3) unique hotspot steps（current）

### 3.1 step `307` (`cycle=3, sic=46`) — `DirectGeoLocalDeg p95`

top joints:

1. `ball_r`: `1.082470`
2. `ball_l`: `0.956572`
3. `foot_l`: `0.781149`
4. `calf_l`: `0.577490`
5. `thigh_r`: `0.461850`
6. `upperarm_l`: `0.381322`
7. `calf_r`: `0.307886`
8. `RUpArmTwist_r_01`: `0.296607`
9. `thigh_l`: `0.293383`
10. `hand_r`: `0.264396`

解读：

- 这是一个明显的 **双脚末端 + 小腿** 驱动点
- 其中 `ball_r / ball_l / foot_l` 是最主要贡献者
- 同时带有轻度 upper-body 残差（`upperarm_l`, `hand_r`）

---

### 3.2 step `91` (`cycle=1, sic=4`) — `DirectGeoLocalDeg p99`

top joints:

1. `ball_l`: `0.759482`
2. `foot_r`: `0.754832`
3. `thigh_l`: `0.722343`
4. `upperarm_r`: `0.721642`
5. `lowerarm_l`: `0.620497`
6. `foot_l`: `0.482093`
7. `upperarm_l`: `0.422332`
8. `RUpArmTwist_r_01`: `0.420382`
9. `thigh_r`: `0.382634`
10. `spine_02`: `0.354068`

解读：

- 这是一个 **下肢 + 上肢同时抬高** 的 mixed hotspot
- 下肢里 `ball_l / foot_r / thigh_l` 很突出
- 上肢里 `upperarm_r / lowerarm_l / upperarm_l` 也明显参与

---

### 3.3 step `352` (`cycle=4, sic=4`) — `DirectGeoLocalDeg max`

top joints:

1. `ball_l`: `0.901353`
2. `foot_r`: `0.809551`
3. `thigh_l`: `0.768048`
4. `upperarm_r`: `0.712592`
5. `lowerarm_l`: `0.550964`
6. `foot_l`: `0.539135`
7. `RUpArmTwist_r_01`: `0.421631`
8. `thigh_r`: `0.404507`
9. `upperarm_l`: `0.375363`
10. `spine_02`: `0.351836`

解读：

- `Direct max` 与 `Direct p99` 的形态非常接近
- 核心模式仍是：`ball_l + foot_r + thigh_l` 主导，叠加右上肢/左前臂残差
- 说明 `sic=4` 附近是一个稳定的 mixed hotspot 区段

---

### 3.4 step `270` (`cycle=3, sic=9`) — `BlendGeoLocalDeg p95` / `GeoLocalDeg p95`

top joints:

1. `calf_l`: `0.790371`
2. `foot_l`: `0.645318`
3. `thigh_l`: `0.439434`
4. `thigh_r`: `0.380820`
5. `upperarm_r`: `0.278650`
6. `head`: `0.277594`
7. `lowerarm_r`: `0.274460`
8. `foot_r`: `0.260180`
9. `upperarm_l`: `0.213417`
10. `thumb_01_l`: `0.209783`

解读：

- 这是一个 **左腿链主导** 的 tail 点
- 主体顺序非常清楚：`calf_l -> foot_l -> thigh_l`
- 同时存在较轻的右臂/头部/右前臂扰动

---

### 3.5 step `273` (`cycle=3, sic=12`) — `BlendGeoLocalDeg p99` / `GeoLocalDeg p99`

top joints:

1. `foot_l`: `1.642255`
2. `ball_l`: `0.564793`
3. `calf_l`: `0.471606`
4. `thigh_l`: `0.421050`
5. `hand_r`: `0.367545`
6. `ball_r`: `0.326781`
7. `foot_r`: `0.239423`
8. `thigh_r`: `0.227812`
9. `upperarm_r`: `0.176284`
10. `calf_r`: `0.151521`

解读：

- 这是最典型的 **`foot_l` 单点爆高** hotspot
- `foot_l` 明显是绝对主导项，后面才是 `ball_l / calf_l / thigh_l`
- 这和此前 watchlist 里的 `foot_l/ball_l` 热点完全一致

---

### 3.6 step `99` (`cycle=1, sic=12`) — `BlendGeoLocalDeg max` / `GeoLocalDeg max`

top joints:

1. `foot_l`: `1.582562`
2. `calf_l`: `0.531018`
3. `ball_l`: `0.504820`
4. `thigh_l`: `0.392108`
5. `hand_r`: `0.365986`
6. `ball_r`: `0.317781`
7. `foot_r`: `0.264134`
8. `thigh_r`: `0.238816`
9. `upperarm_r`: `0.176457`
10. `calf_r`: `0.155658`

解读：

- 与 step `273` 本质同型，也是 **左脚链主导**
- 差别在于这里 `calf_l` 比 `ball_l` 更高，说明左腿整条链都在被拉高
- `cycle=1, sic=12` 是当前最明确的 `blend/global` watch step

---

### 3.7 step `401` (`cycle=4, sic=53`) — `LambdaMean p95`

top joints:

1. `foot_r`: `0.590957`
2. `upperarm_l`: `0.382440`
3. `thigh_l`: `0.305874`
4. `thigh_r`: `0.288958`
5. `lowerarm_l`: `0.207271`
6. `calf_r`: `0.205509`
7. `hand_l`: `0.186182`
8. `RUpArmTwist_l_01`: `0.160387`
9. `lowerarm_r`: `0.157347`
10. `thumb_01_l`: `0.141809`

解读：

- `Lambda p95` 的 same-step pose hotspot 不是单纯 foot-only
- 更像是 **`foot_r + 左臂 + 双 thigh`** 的组合点
- 说明 λ 高位点并不直接等价于左脚热点那一类 tail

---

### 3.8 step `174` (`cycle=2, sic=0`) — `LambdaMean p99`

top joints:

1. `thigh_l`: `0.551534`
2. `thigh_r`: `0.469205`
3. `hand_l`: `0.385464`
4. `foot_r`: `0.378879`
5. `hand_r`: `0.292210`
6. `ball_l`: `0.268707`
7. `foot_l`: `0.263748`
8. `upperarm_l`: `0.207088`
9. `lowerarm_l`: `0.183850`
10. `calf_r`: `0.139429`

解读：

- `Lambda p99` 在 `sic=0`，更像是 **双 thigh + 双手 + foot_r** 的起始帧组合点
- 和 `blend/global` 在 `sic=12` 的左脚链主导热点不是同一种形态

---

### 3.9 step `261` (`cycle=3, sic=0`) — `LambdaMean max`

top joints:

1. `thigh_l`: `0.536485`
2. `thigh_r`: `0.448822`
3. `hand_l`: `0.384034`
4. `foot_r`: `0.383060`
5. `hand_r`: `0.287143`
6. `ball_l`: `0.252904`
7. `foot_l`: `0.247486`
8. `upperarm_l`: `0.201520`
9. `calf_r`: `0.175026`
10. `lowerarm_l`: `0.169293`

解读：

- `Lambda max` 与 `Lambda p99` 几乎同型
- 说明 λ 高位主要集中在 `sic=0` 附近的 **双 thigh + upper-body 混合点**
- 它和 `Direct/Blend/Geo` 的尾部热点窗口并不完全重合

---

## 4) 当前最值得记住的 hotspot 模式

### 模式 A：`sic=12` 左脚链主导

代表 step：

- `99` (`Blend/Geo max`)
- `273` (`Blend/Geo p99`)

共同特征：

- `foot_l` 是绝对第一热点
- 其次通常是 `ball_l / calf_l / thigh_l`
- 这是当前最稳定、最像 watchlist 根部的 hotspot 模式

### 模式 B：`sic=4` mixed hotspot

代表 step：

- `91` (`Direct p99`)
- `352` (`Direct max`)

共同特征：

- `ball_l / foot_r / thigh_l` 持续排前
- 同时带 `upperarm_r / lowerarm_l` 等 upper-body 残差
- 说明 direct tail 不只是腿部问题，而是 mixed residual

### 模式 C：`sic=0` lambda 高位点

代表 step：

- `174` (`Lambda p99`)
- `261` (`Lambda max`)

共同特征：

- `thigh_l / thigh_r` 稳定居前
- `hand_l / hand_r / foot_r` 也显著
- 这类点更像 cycle 起点附近的全身混合态，而不是单个 foot hotspot

---

## 5) 下一步建议

如果继续往下收口，优先顺序建议是：

1. 先盯 `sic=12`：`foot_l / ball_l / calf_l`
2. 再盯 `sic=4`：`ball_l / foot_r / thigh_l / upperarm_r`
3. 最后单独看 `sic=0`：确认 λ 高位是否只是伴随信号，还是会放大起始帧 mixed residual

如果需要，我下一步可以继续把这些 hotspot step 再展开成：

- **对应 baseline A / B 同 step 的 top joints 对照**
- 或者直接做成 **watchlist 表：step -> top joints -> 建议关注模块**
