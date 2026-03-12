# 2026-03-11 `exp_phase_DirectBranch_v1_d1 -> posttrain mainline` pointwise hotspot 解释

## 1) 目的

这份文档回答的问题是：

- 对 `docs/Problems/active/2026-03-11_trainbase_posttrain_mainline_evalon_snapshot.md` 里列出的
  `p95 / p99 / max` step 点，
- 它们具体是由哪些 `bone/joint` 在拉高误差；
- 并且只看这条单源 anchor，不再带 baseline section。

---

## 2) 口径说明

数据源：`debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_Walk_F_series/Walk_F_freerun_cycles.json`

当前 eval artifact 里：

- 有 `per_step_direct_geolocal_deg`
- 没有 `per_step_blend_geolocal_deg`
- 也没有 `per_step_geolocal_deg`

所以这里沿用 2026-03-09 模板的解释规则：

- 对 `DirectGeoLocalDeg`：直接使用该 step 的 joint 排名；
- 对 `BlendGeoLocalDeg / GeoLocalDeg / LambdaMean`：使用**同一个 step**上的 direct joint 排名来解释该时刻的 pose hotspot。

也就是说：

- 这是 same-step joint hotspot；
- 不是 blend/global 各自独立的 joint 分解。

---

## 3) unique hotspot steps（anchor）

### 3.1 step `87` (`cycle=1, sic=0`) - `DirectGeoLocalDeg p99`
top joints:

1. `calf_l`: `1.111619`
2. `thigh_l`: `0.748296`
3. `foot_l`: `0.676364`
4. `foot_r`: `0.580803`
5. `upperarm_l`: `0.497270`
6. `hand_l`: `0.448934`
7. `thigh_r`: `0.425269`
8. `lowerarm_l`: `0.361672`
9. `calf_r`: `0.355498`
10. `head`: `0.349979`

解读：

- 这是一个 `sic=0` 的 cycle-start lower-body hotspot。
- `calf_l / thigh_l / foot_l / foot_r` 主导，说明左腿链最强，同时右脚也被带起。
- 上肢里 `upperarm_l / hand_l / lowerarm_l` 同时偏高，但仍是次级贡献。

---

### 3.2 step `99` (`cycle=1, sic=12`) - `BlendGeoLocalDeg max` / `GeoLocalDeg max`
top joints:

1. `foot_l`: `0.849918`
2. `hand_r`: `0.425279`
3. `thigh_r`: `0.410574`
4. `foot_r`: `0.410502`
5. `ball_l`: `0.323989`
6. `calf_l`: `0.314873`
7. `upperarm_l`: `0.309980`
8. `calf_r`: `0.286821`
9. `ball_r`: `0.259271`
10. `thumb_01_l`: `0.258693`

解读：

- 这是最典型的 `sic=12` 左脚链热点之一。
- `foot_l` 明显是第一主导项，但 `hand_r / thigh_r / foot_r` 也一起抬高，所以不是纯 foot-only。
- 它和后面的 `step 273 / 360` 属于同型窗口。

---

### 3.3 step `101` (`cycle=1, sic=14`) - `GeoLocalDeg p95`
top joints:

1. `hand_r`: `0.840770`
2. `ball_l`: `0.686113`
3. `lowerarm_l`: `0.559721`
4. `calf_r`: `0.524632`
5. `upperarm_l`: `0.478211`
6. `upperarm_r`: `0.475037`
7. `thigh_l`: `0.375702`
8. `foot_l`: `0.344447`
9. `thigh_r`: `0.328755`
10. `calf_l`: `0.207402`

解读：

- 这是 `GeoLocalDeg p95` 对应的 mixed hotspot。
- `hand_r` 突然成为绝对第一项，同时 `ball_l / calf_r / upperarm_l / upperarm_r` 一起升高。
- 说明 `sic=14` 更像 upper-limb + lower-limb crossover，而不是单纯左脚窗口。

---

### 3.4 step `183` (`cycle=2, sic=9`) - `BlendGeoLocalDeg p95`
top joints:

1. `ball_l`: `0.824782`
2. `hand_r`: `0.425283`
3. `foot_l`: `0.384625`
4. `thumb_01_l`: `0.299252`
5. `RUpArmTwist_r_01`: `0.288213`
6. `foot_r`: `0.212414`
7. `clavicle_r`: `0.204760`
8. `upperarm_r`: `0.201777`
9. `calf_r`: `0.191974`
10. `hand_l`: `0.186285`

解读：

- 这是 `cycle=2, sic=9` 的左脚前掌窗口。
- `ball_l` 是绝对主导，后面才是 `hand_r / foot_l / thumb_01_l`。
- 相比 `sic=12` 窗口，它更像 `ball_l` 单点先起，再带出少量上肢残差。

---

### 3.5 step `259` (`cycle=2, sic=85`) - `DirectGeoLocalDeg p95`
top joints:

1. `thigh_l`: `0.939127`
2. `calf_l`: `0.703644`
3. `foot_l`: `0.668731`
4. `foot_r`: `0.650198`
5. `upperarm_l`: `0.553370`
6. `ball_l`: `0.425530`
7. `ball_r`: `0.365876`
8. `RUpArmTwist_r_01`: `0.354746`
9. `hand_r`: `0.305922`
10. `thumb_01_l`: `0.247169`

解读：

- 这是 `DirectGeoLocalDeg p95` 对应的尾段 lower-body hotspot。
- `thigh_l / calf_l / foot_l / foot_r` 一起抬高，左腿链最强，右脚链同步偏高。
- `upperarm_l / hand_r / RUpArmTwist_r_01` 说明尾段仍带有 upper-body mixed residual。

---

### 3.6 step `261` (`cycle=3, sic=0`) - `DirectGeoLocalDeg max` / `LambdaMean p99`
top joints:

1. `calf_l`: `1.195993`
2. `thigh_l`: `0.791169`
3. `foot_l`: `0.642206`
4. `foot_r`: `0.613874`
5. `upperarm_l`: `0.546342`
6. `hand_l`: `0.471890`
7. `thigh_r`: `0.449031`
8. `lowerarm_l`: `0.431025`
9. `calf_r`: `0.389298`
10. `head`: `0.359102`

解读：

- 这是 `DirectGeoLocalDeg max` 和 `LambdaMean p99` 共享的 `sic=0` 窗口。
- `calf_l / thigh_l / foot_l / foot_r` 仍是主体，属于稳定重复的 cycle-start lower-body 模式。
- `Lambda` 在这里抬高，说明它并不只盯 `sic=12` 左脚热点，也会在 cycle start 跟随下肢链抬升。

---

### 3.7 step `273` (`cycle=3, sic=12`) - `GeoLocalDeg p99` / `LambdaMean max`
top joints:

1. `foot_l`: `0.792370`
2. `hand_r`: `0.406638`
3. `thigh_r`: `0.387380`
4. `foot_r`: `0.379341`
5. `upperarm_l`: `0.334026`
6. `calf_r`: `0.317756`
7. `thumb_01_l`: `0.289113`
8. `ball_l`: `0.273019`
9. `ball_r`: `0.227766`
10. `spine_02`: `0.196449`

解读：

- 这是 `GeoLocalDeg p99` 与 `LambdaMean max` 共享的 `sic=12` 窗口。
- `foot_l` 是绝对第一项，随后是 `hand_r / thigh_r / foot_r / upperarm_l`。
- 相比 current mainline 的 `foot_l + ball_l + calf_l` 纯左脚链，这个 anchor 更混合一些。

---

### 3.8 step `360` (`cycle=4, sic=12`) - `BlendGeoLocalDeg p99`
top joints:

1. `foot_l`: `0.765443`
2. `hand_r`: `0.401727`
3. `thigh_r`: `0.379522`
4. `foot_r`: `0.366941`
5. `upperarm_l`: `0.343092`
6. `calf_r`: `0.325075`
7. `thumb_01_l`: `0.299839`
8. `ball_l`: `0.224511`
9. `ball_r`: `0.207073`
10. `spine_02`: `0.193032`

解读：

- 这是 `BlendGeoLocalDeg p99` 的 `sic=12` 重复热点。
- 排名与 `step 273` 几乎同型：`foot_l` 第一，随后是 `hand_r / thigh_r / foot_r / upperarm_l / calf_r`。
- 说明这个 anchor 在多 cycle 的 `sic=12` 上存在稳定重复的 mixed hotspot。

---

### 3.9 step `361` (`cycle=4, sic=13`) - `LambdaMean p95`
top joints:

1. `calf_l`: `0.829385`
2. `foot_l`: `0.769126`
3. `RUpArmTwist_r_01`: `0.642844`
4. `ball_l`: `0.621792`
5. `upperarm_r`: `0.523768`
6. `calf_r`: `0.450083`
7. `thigh_r`: `0.418352`
8. `upperarm_l`: `0.397025`
9. `lowerarm_l`: `0.376207`
10. `hand_r`: `0.323947`

解读：

- 这是 `LambdaMean p95` 的相邻一步 `sic=13` 窗口。
- `calf_l / foot_l / ball_l` 先抬高，再叠加 `RUpArmTwist_r_01 / upperarm_r / calf_r`。
- 说明 lambda 高位尾部会从 `sic=12` 左脚窗口向后拖到 `sic=13`，并混入右臂残差。

---

## 4) 当前最值得记住的 hotspot 模式

### 模式 A：`sic=12` 重复 mixed hotspot

代表 step：

- `99` (`Blend max` / `Geo max`)
- `273` (`Geo p99` / `Lambda max`)
- `360` (`Blend p99`)

共同特征：

- `foot_l` 基本都排第一；
- 但后面不是纯 `ball_l / calf_l`，而是常常混入 `hand_r / thigh_r / foot_r / upperarm_l`；
- 所以这条 anchor 的 `sic=12` 更像 mixed hotspot，而不是纯左脚链单型。

### 模式 B：`sic=0` cycle-start lower-body hotspot

代表 step：

- `87` (`Direct p99`)
- `261` (`Direct max` / `Lambda p99`)

共同特征：

- `calf_l / thigh_l / foot_l / foot_r` 稳定排在前列；
- 左腿链是主因，右脚链同步抬高；
- `Lambda` 在这里也会跟着抬高，说明 lambda 高位并不只发生在 `sic=12`。

### 模式 C：`sic=9-14` crossover / tail hotspot

代表 step：

- `183` (`Blend p95`)
- `101` (`Geo p95`)
- `361` (`Lambda p95`)

共同特征：

- `ball_l`、`hand_r`、`upperarm_r`、`calf_r` 会以不同组合出现；
- 它们更像主热点窗口前后的 crossover/tail，而不是最核心的根部窗口；
- 但这些点能解释为什么这个 anchor 的 tail quantile 仍带有 upper-body mixed residual。

---

## 5) 结论

如果把这条 anchor 只压缩成一句话，可以记成：

- lower-body 主要有两类热点：`sic=0` 的 cycle-start 左腿链，以及反复出现的 `sic=12` 左脚主导 mixed hotspot；
- 同时在 `sic=9-14` 的邻域，会夹带 `hand_r / upperarm_r / calf_r` 等 crossover residual；
- 这也解释了为什么后来 accepted `s180-promote -> 71 -> 72 -> lambda final` 还需要继续清理 arms/calf watchlist，而不能只盯 `foot_l/ball_l`。
