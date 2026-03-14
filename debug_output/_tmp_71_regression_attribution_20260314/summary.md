# 71 regression attribution

- raw `70b` is diagnostic-only; it is not the operational handoff in this lane
- the comparison here is `current 70R -> 71` vs `candidate lowdrift 70R -> 71`
- eval contract: model-source only; strict was not needed for the conclusion

## Short conclusion

- candidate starts `71` from a better `70R` on `all_ex_root/leg`, but the `71` stage gives it far less additional gain than the current lane does
- the candidate gap is already visible during early `71` replay snapshots, so this is not an inherited-start-only story
- the candidate still improves `calf_r@SIC2-4` and `foot_l/ball_l@SIC12-15`, but broad regressions on other leg joints/windows outweigh those hotspot wins
- best next step: `shorter_71_or_early_stop` (candidate replay improves through mid-training but never catches current 71; snapshot selection helps a bit, but not enough by itself)

## Replay snapshots

| lane_snapshot | DirectGeoLocalDeg | all_ex_root | leg | nonleg | arm | legs_main | arms_main | foot_l/ball_l@SIC12-15 | calf_r@SIC2-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_s000 | 0.158235 | 0.158235 | 0.556049 | 0.072222 | 0.082665 | 0.556049 | 0.082665 | 1.118483 | 0.613849 |
| current_s020 | 0.122295 | 0.122295 | 0.353883 | 0.072222 | 0.082665 | 0.353883 | 0.082665 | 0.473506 | 0.408613 |
| current_s060 | 0.124932 | 0.124932 | 0.368718 | 0.072222 | 0.082665 | 0.368718 | 0.082665 | 0.952248 | 0.320382 |
| current_s120 | 0.112269 | 0.112269 | 0.297488 | 0.072222 | 0.082665 | 0.297488 | 0.082665 | 0.433427 | 0.535532 |
| current_s180 | 0.111911 | 0.111911 | 0.295473 | 0.072222 | 0.082665 | 0.295473 | 0.082665 | 0.599272 | 0.440912 |
| candidate_s000 | 0.130926 | 0.130926 | 0.349263 | 0.083717 | 0.091849 | 0.349263 | 0.091849 | 0.860095 | 0.393019 |
| candidate_s020 | 0.135698 | 0.135698 | 0.376105 | 0.083717 | 0.091849 | 0.376105 | 0.091849 | 0.783757 | 0.428210 |
| candidate_s060 | 0.134537 | 0.134537 | 0.369575 | 0.083717 | 0.091849 | 0.369575 | 0.091849 | 0.722985 | 0.198369 |
| candidate_s120 | 0.126309 | 0.126309 | 0.323293 | 0.083717 | 0.091849 | 0.323293 | 0.091849 | 0.536233 | 0.136361 |
| candidate_s180 | 0.127787 | 0.127787 | 0.331611 | 0.083717 | 0.091849 | 0.331611 | 0.091849 | 0.540575 | 0.295644 |

## Reference 70R -> 71 gain decomposition

| metric | inherited (candidate70R-current70R) | 71 gain gap ((cand71-cand70R)-(cur71-cur70R)) | final gap (candidate71-current71) |
|---|---:|---:|---:|
| DirectGeoLocalDeg | -0.027310 | 0.043187 | 0.015877 |
| all_ex_root | -0.027310 | 0.043187 | 0.015877 |
| leg | -0.206786 | 0.242925 | 0.036138 |
| nonleg | 0.011496 | 0.000000 | 0.011496 |
| arm | 0.009184 | 0.000000 | 0.009184 |
| legs_main | -0.206786 | 0.242925 | 0.036138 |
| arms_main | 0.009184 | 0.000000 | 0.009184 |
| foot_l_ball_l_SIC12_15 | -0.258388 | 0.199691 | -0.058697 |
| calf_r_SIC2_4 | -0.220830 | 0.075562 | -0.145269 |

## Final-stage biggest candidate regressions

| leg_joint | delta(candidate71-current71) | current71 | candidate71 |
|---|---:|---:|---:|
| calf_l | 0.176958 | 0.274648 | 0.451606 |
| ball_r | 0.056374 | 0.214300 | 0.270674 |
| foot_l | 0.055202 | 0.408709 | 0.463911 |
| foot_r | 0.053937 | 0.291310 | 0.345247 |
| thigh_r | 0.006593 | 0.239627 | 0.246221 |
| ball_l | 0.003101 | 0.276364 | 0.279465 |
| calf_r | -0.031371 | 0.287997 | 0.256626 |
| thigh_l | -0.031688 | 0.370829 | 0.339140 |

| leg_SIC | delta(candidate71-current71) | current71 | candidate71 |
|---|---:|---:|---:|
| SIC26 | 0.239443 | 0.091224 | 0.330667 |
| SIC03 | 0.229487 | 0.298561 | 0.528049 |
| SIC12 | 0.222226 | 0.398464 | 0.620689 |
| SIC24 | 0.208647 | 0.190522 | 0.399169 |
| SIC25 | 0.192599 | 0.214588 | 0.407187 |
| SIC13 | 0.178620 | 0.280130 | 0.458750 |
| SIC18 | 0.172368 | 0.155506 | 0.327874 |
| SIC09 | 0.157123 | 0.292204 | 0.449327 |
| SIC34 | 0.149427 | 0.267361 | 0.416788 |
| SIC08 | 0.138680 | 0.492036 | 0.630716 |

