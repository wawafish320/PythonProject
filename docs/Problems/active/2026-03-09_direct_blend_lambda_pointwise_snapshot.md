# 2026-03-09 `direct / blend / lambda` pointwise 快照

## 1) 目的

这份文档只整理一件事：

- 不看均值；
- 只看 `p50 / p90 / p95 / p99 / max` 对应的**实际索引点**；
- 对每个分位点给出：`global_step / cycle / sic / value`。

这里的 `sic` 定义为：

- `sic = global_step % cycle_len`
- 当前 `Walk_F` 的 `cycle_len = 87`

统一 mask 口径：

- `cycle >= 1`
- `drop_wrap = True`

因此当前有效 step 数统一为 `n = 344`。

> 注意：这里的 percentile 不是插值后的统计分位数，而是 **nearest-rank 对应的真实 step 点**。
> 也就是按升序排序后，直接取 `ceil(q * n)` 对应的那个观测点。

---

## 2) 数据源

### 当前 current eval

- `debug_output/_tmp_lambda_from_s180_72_eval_20260308_Walk_F/Walk_F_freerun_cycles.json`

### 基线 A

- `debug_output/_tmp_phaseD_direct_geolocal_compare_20260305_r5/new_fullchain_pretrain/Walk_F_freerun_cycles.json`

### 基线 B

- `debug_output/_tmp_posttrain_from_exp_phase_DirectBranch_v1_d1_20260307_eval_on_Walk_F_series/Walk_F_freerun_cycles.json`

---

## 3) current：实际索引点

### 3.1 `DirectGeoLocalDeg`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 109 | 1 | 22 | 0.108593 |
| `p90` | 310 | 182 | 2 | 8 | 0.168138 |
| `p95` | 327 | 307 | 3 | 46 | 0.179283 |
| `p99` | 341 | 91 | 1 | 4 | 0.206879 |
| `max` | - | 352 | 4 | 4 | 0.210580 |

### 3.2 `BlendGeoLocalDeg`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 305 | 3 | 44 | 0.468811 |
| `p90` | 310 | 179 | 2 | 5 | 0.647782 |
| `p95` | 327 | 270 | 3 | 9 | 0.737072 |
| `p99` | 341 | 273 | 3 | 12 | 0.817071 |
| `max` | - | 99 | 1 | 12 | 0.821959 |

### 3.3 `GeoLocalDeg`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 379 | 4 | 31 | 0.924266 |
| `p90` | 310 | 314 | 3 | 53 | 1.242373 |
| `p95` | 327 | 270 | 3 | 9 | 1.382714 |
| `p99` | 341 | 273 | 3 | 12 | 1.566507 |
| `max` | - | 99 | 1 | 12 | 1.570373 |

### 3.4 `LambdaMean`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 254 | 2 | 80 | 0.973611 |
| `p90` | 310 | 141 | 1 | 54 | 0.973853 |
| `p95` | 327 | 401 | 4 | 53 | 0.973919 |
| `p99` | 341 | 174 | 2 | 0 | 0.974037 |
| `max` | - | 261 | 3 | 0 | 0.974067 |

---

## 4) baseline A：实际索引点

### 4.1 `DirectGeoLocalDeg`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 139 | 1 | 52 | 0.141481 |
| `p90` | 310 | 89 | 1 | 2 | 0.195596 |
| `p95` | 327 | 179 | 2 | 5 | 0.218209 |
| `p99` | 341 | 270 | 3 | 9 | 0.317730 |
| `max` | - | 96 | 1 | 9 | 0.321332 |

### 4.2 `BlendGeoLocalDeg`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 404 | 4 | 56 | 0.510423 |
| `p90` | 310 | 268 | 3 | 7 | 0.696213 |
| `p95` | 327 | 187 | 2 | 13 | 0.827272 |
| `p99` | 341 | 96 | 1 | 9 | 0.881129 |
| `max` | - | 97 | 1 | 10 | 0.901127 |

### 4.3 `GeoLocalDeg`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 118 | 1 | 31 | 0.989825 |
| `p90` | 310 | 181 | 2 | 7 | 1.357094 |
| `p95` | 327 | 183 | 2 | 9 | 1.523750 |
| `p99` | 341 | 272 | 3 | 11 | 1.676843 |
| `max` | - | 185 | 2 | 11 | 1.683177 |

### 4.4 `LambdaMean`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 336 | 3 | 75 | 0.973638 |
| `p90` | 310 | 299 | 3 | 38 | 0.973801 |
| `p95` | 327 | 176 | 2 | 2 | 0.973861 |
| `p99` | 341 | 142 | 1 | 55 | 0.973925 |
| `max` | - | 174 | 2 | 0 | 0.973971 |

---

## 5) baseline B：实际索引点

### 5.1 `DirectGeoLocalDeg`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 336 | 3 | 75 | 0.127522 |
| `p90` | 310 | 178 | 2 | 4 | 0.172469 |
| `p95` | 327 | 259 | 2 | 85 | 0.179868 |
| `p99` | 341 | 87 | 1 | 0 | 0.212547 |
| `max` | - | 261 | 3 | 0 | 0.221648 |

### 5.2 `BlendGeoLocalDeg`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 411 | 4 | 63 | 0.479669 |
| `p90` | 310 | 315 | 3 | 54 | 0.613184 |
| `p95` | 327 | 183 | 2 | 9 | 0.704609 |
| `p99` | 341 | 360 | 4 | 12 | 0.796576 |
| `max` | - | 99 | 1 | 12 | 0.824611 |

### 5.3 `GeoLocalDeg`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 394 | 4 | 46 | 0.918294 |
| `p90` | 310 | 315 | 3 | 54 | 1.236477 |
| `p95` | 327 | 101 | 1 | 14 | 1.348158 |
| `p99` | 341 | 273 | 3 | 12 | 1.547721 |
| `max` | - | 99 | 1 | 12 | 1.574776 |

### 5.4 `LambdaMean`

| quantile | rank | global_step | cycle | sic | value |
|---|---:|---:|---:|---:|---:|
| `p50` | 172 | 250 | 2 | 76 | 0.972846 |
| `p90` | 310 | 229 | 2 | 55 | 0.973102 |
| `p95` | 327 | 361 | 4 | 13 | 0.973131 |
| `p99` | 341 | 261 | 3 | 0 | 0.973286 |
| `max` | - | 273 | 3 | 12 | 0.973315 |

---

## 6) 当前最值得记住的点

1. current 的 `DirectGeoLocalDeg`：
   - `p95` 在 `cycle=3, sic=46`
   - `p99` 在 `cycle=1, sic=4`
   - `max` 在 `cycle=4, sic=4`

2. current 的 `BlendGeoLocalDeg` 与 `GeoLocalDeg`：
   - `p95` 都落在 `cycle=3, sic=9`
   - `p99` 都落在 `cycle=3, sic=12`
   - `max` 都落在 `cycle=1, sic=12` 或极近邻位置

3. current 的 `LambdaMean`：
   - 整体变化很小
   - `p99 / max` 分别落在 `cycle=2, sic=0` 与 `cycle=3, sic=0`

4. 如果后面要追 hotspot，最值得优先盯的 current 点位是：
   - `DirectGeoLocalDeg`: `cycle=4, sic=4`
   - `BlendGeoLocalDeg`: `cycle=1, sic=12`
   - `GeoLocalDeg`: `cycle=1, sic=12`

---

## 7) 备注

- 这份文档是 pointwise 版本，专门回答“某个 quantile 落在哪个 step / index 点上”。
- 如果后续还需要，我可以继续补：
  - 这些 index 点对应的 `bone-wise top joints`
  - 或者把这些点直接转成 `SIC hotspot watchlist`
