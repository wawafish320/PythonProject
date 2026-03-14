# 2026-03-14 cp015 restore `w_contact_plan` quick summary

Primary mask: `cycle>=1 && !wrap_boundary_step`

| lane | basetrain all_ex_root | basetrain leg | plan_bce | |logit| mean | |Δprob| mean | Stage6 all_ex_root | Stage6 leg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `old_bestfree` | 6.431959 | 10.517242 | 0.611716 | 0.372866 | 0.009323 | 0.313279 | 0.874230 |
| `cp015_bestfree` | 5.647744 | 10.826995 | 0.877855 | 1.566154 | 0.003717 | 0.431377 | 1.167171 |
| `cp015_restorewcontact` | 6.732179 | 10.475876 | 0.634094 | 0.382409 | 0.005077 | 0.331054 | 0.880223 |

Reads:

- Restoring `w_contact_plan` pulls `|logit|` back near old (`0.382409` vs old `0.372866`; cp015 was `1.566154`).
- But temporal sharpness only partially recovers: `|Δprob|=0.005077`, still far below old `0.009323` and only modestly above cp015 `0.003717`.
- Basetrain free-run does not improve on cp015 overall: `all_ex_root=6.732179` vs cp015 `5.647744` (worse), though `leg=10.475876` vs cp015 `10.826995` (slightly better).
- Stage6 does improve over cp015 (`0.331054` vs `0.431377` all_ex_root; `0.880223` vs `1.167171` leg), but it does not recover to old (`0.313279` / `0.874230`) and is still well behind `cp015_with_old_planstack` (`0.295533` / `0.740703`).
- So `w_contact_plan` is clearly causal and helpful, but this single-variable restore is not sufficient by itself to reproduce the full old-plan benefit.
