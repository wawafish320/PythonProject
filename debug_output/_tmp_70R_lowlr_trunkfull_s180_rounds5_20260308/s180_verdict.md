# low-LR trunkfull s180 rounds5 verdict

## Key outcome

- The `low-LR trunkfull` candidate **still beats `new70R` at 180 step + rounds=5** on upper-body regression: overall `arms_main`, A (52-59), and B (76-80) are all negative vs `new70R`.
- Relative to `new70R`, `legs_main` is also better overall and in A; B is effectively neutral (`+0.003865`), so this is **not** an arms-for-legs trade in aggregate.
- Relative to `current70R`, the `s180` candidate still improves both `arms_main` and `legs_main` overall, and both problematic arm windows remain negative.
- Compared with the stronger `s60` checkpoint, `s180` is mixed but stable: it improves overall legs and B-arm, but gives back part of the A-window arm gain. So the fix still holds, though the best upper-body sweet spot may be earlier than 180 for some windows.

## Compare: current70R -> candidate

- overall: `legs_main=-0.317204`, `arms_main=-0.066846`, `left_arm_main=-0.084986`, `right_arm_main=-0.048707`
- A window (52-59): `legs_main=-0.136320`, `arms_main=-0.040796`
- B window (76-80): `legs_main=-0.688290`, `arms_main=-0.172084`

## Compare: new70R -> candidate

- overall: `legs_main=-0.054554`, `arms_main=-0.087329`, `left_arm_main=-0.076603`, `right_arm_main=-0.098056`
- A window (52-59): `legs_main=-0.070562`, `arms_main=-0.240041`
- B window (76-80): `legs_main=+0.003865`, `arms_main=-0.220233`

## Reference: current70R -> new70R

- overall: `legs_main=-0.262650`, `arms_main=+0.020483`, `left_arm_main=-0.008383`, `right_arm_main=+0.049349`
- A window (52-59): `legs_main=-0.065758`, `arms_main=+0.199245`
- B window (76-80): `legs_main=-0.692155`, `arms_main=+0.048149`

## Candidate drift: s60 -> s180

- overall: `legs_main=-0.022891`, `arms_main=+0.007171`, `left_arm_main=-0.003732`, `right_arm_main=+0.018074`
- A window (52-59): `legs_main=-0.007442`, `arms_main=+0.074934`
- B window (76-80): `legs_main=+0.012935`, `arms_main=-0.021263`

## Interpretation

- Sign convention is `candidate - baseline`; negative means lower direct geolocal error and is better.
- The original claim survives the longer run: `new70R` is not the best reachable solution under the new representation geometry; `low LR + trunk mobility` still finds a better upper-body solution at 180 steps.
- The `s180` candidate is strong enough to replace the current `new70R` for the next chained validation (`71 -> 72 -> lambda_final`).
- The only caution is that A-window arm gain weakens relative to `s60`, and B-window `legs_main` vs `new70R` is nearly flat rather than clearly negative. That suggests watching window-level drift in the downstream chain, not just overall means.

## Recommended next step

- Promote `models/__tmp_70R_new_lowlr_trunkfull_s180_20260308/ckpt_last_WalkF_stage7_70R_new_lowlr_trunkfull_s180_20260308.pth` as the new `70R` candidate and rerun `71 -> 72 -> lambda_final` with the same rounds=5 freerun lane and the same `legs_main/arms_main/A/B` summary checks.
