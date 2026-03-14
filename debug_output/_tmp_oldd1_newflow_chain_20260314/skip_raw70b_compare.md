# old d1 newflow: `70a -> new70b_replace` focused note

## Key clarification

For the run completed on 2026-03-14, `new70b_replace` already skips the raw `70b` checkpoint as its training input.

- warmstart is created from `70a`
- `new70b_replace` is trained from that `70a`-derived warmstart
- raw `70b` is only an extra diagnostic checkpoint in the chain log

Code path:

- `tools/run_oldd1_newflow_chain.py:704`
- `tools/run_oldd1_newflow_chain.py:711`

Concretely:

- `create_replace_zerophase_warmstart(src_ckpt=ckpt_70a, ...)`
- `run_posttrain_stage(... ckpt_in=warmstart_ckpt, ... new70b_replace ...)`

So there is no separate "re-run skip70b" training job needed for this lane: the existing `new70b_replace` artifact is already the `70a -> new70b_replace` result.

---

## Checkpoints

- `70a`: `models/__tmp_oldd1_newflow_chain_20260314/70a/ckpt_last_WalkF_stage7_70a_from_oldd1_newflow_20260314.pth`
- `70b`: `models/__tmp_oldd1_newflow_chain_20260314/70b/ckpt_last_WalkF_stage7_70b_from_oldd1_newflow_20260314.pth`
- `new70b_replace`: `models/__tmp_oldd1_newflow_chain_20260314/70b_replace/ckpt_last_WalkF_stage7_70b_replace_from_oldd1_newflow_20260314.pth`

Source summary:

- `debug_output/_tmp_oldd1_newflow_chain_20260314/summary.json`

---

## Direct-path metrics (model-source)

| stage | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | else |
|---|---:|---:|---:|---:|---:|---:|
| 70a | 0.275083 | 0.275083 | 0.730911 | 0.176525 | 0.203549 | 0.112650 |
| 70b | 0.308443 | 0.308443 | 0.730643 | 0.217157 | 0.254408 | 0.129109 |
| new70b_replace | 0.280736 | 0.280736 | 0.662440 | 0.198205 | 0.226846 | 0.130508 |

---

## Delta: `70a -> 70b`

| metric | delta |
|---|---:|
| all_ex_root | +0.033361 |
| leg | -0.000268 |
| nonleg | +0.040632 |
| arm | +0.050859 |
| else | +0.016459 |

Interpretation:

- this is the main regression step
- leg is basically flat
- nonleg and arm get noticeably worse

## Delta: `70a -> new70b_replace`

| metric | delta |
|---|---:|
| all_ex_root | +0.005653 |
| leg | -0.068472 |
| nonleg | +0.021680 |
| arm | +0.023297 |
| else | +0.017858 |

Interpretation:

- compared with staying at `70a`, `new70b_replace` buys a real leg improvement
- but it pays for that with worse overall direct, worse nonleg, and worse arm

## Delta: `70b -> new70b_replace`

| metric | delta |
|---|---:|
| all_ex_root | -0.027708 |
| leg | -0.068204 |
| nonleg | -0.018952 |
| arm | -0.027562 |
| else | +0.001399 |

Interpretation:

- `new70b_replace` clearly recovers most of the raw `70b` damage
- but it is not simply "strictly better than 70a"; it is a leg-favoring tradeoff stage

---

## Practical conclusion

For this old d1 lane:

- raw `70b` should be treated as a diagnostic-only stage
- the meaningful handoff is `70a -> new70b_replace`
- if the goal is leg optimization, the right question is:
  - why does `new70b_replace` improve leg relative to `70a`
  - while simultaneously worsening `all_ex_root / nonleg / arm`

That makes the next useful A/B:

1. `70a` vs `new70b_replace`
2. `new70b_replace` vs `70R`
3. `70R` vs `71`

