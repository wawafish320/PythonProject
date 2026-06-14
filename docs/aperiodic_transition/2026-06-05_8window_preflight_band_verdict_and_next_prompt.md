> TRIAGE: KEEP-AS-PROVENANCE. Historical evidence / anti-relitigation note; preserve to explain why this path was rejected or bounded. Do not use as a live plan unless a new decision record explicitly reopens it.

# 8-Window Preflight Block — Reviewer Verdict, Direction, and Next Execution Prompt

Date: 2026-06-05
Role: reviewer / direction-setter (not executor). Read-only verification of the
execution dialogue's `2026-06-05_8window_train_fit_review.md`.

Artifacts independently re-read:
- `debug_output/_tmp_action_handoff_8window_train_fit_beststage1_20260605/summary.json`
- `.../per_metric.csv` (stage1_supervised_fit_8 rows, 328 rows / 8 windows / 41 metrics)
- `.../heading_exact_gt_vs_stage1_audit.json`
- `tools/run_action_handoff_inbetween_8window_train_fit_debug.py`
  (preflight gate L1071–1101; `--heading-tolerance-rad` default 1e-5 L1728;
  heading floor applied only to command gate term L924; classification L1259–1267)
- `2026-06-04_band_audit.md`, `2026-06-04_status_synthesis.md`,
  `2026-06-01_middle_generator_acceptance_contract.md`

---

## 1. Verdict (per load-bearing claim)

| # | Execution claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Stage1 supervised-fit-8 reached the low-MSE GT basin | **确认** | `flat_standardized_mse=5.43e-10`, `state_mse=1.43e-10`, `max_abs_delta_stage1_true_raw=3.40e-4`. 8 windows expressible/memorizable; warm-start crutch still works at 8. |
| 2 | Exact GT passes heading; stage1-pred heading is 3–6e-5 rad | **确认** | heading audit JSON: GT `heading_error_p95` 2.58e-8..4.75e-8 (pass 8/8); stage1-pred 2.995e-5..6.17e-5 (fail 8/8). Recomputed from artifact, matches. |
| 3 | The 1e-5 command-heading tolerance is an over-tight, no-baseline value | **确认** | `--heading-tolerance-rad=1e-5` is a fixed floor; band audit shows GT heading baseline ~2–4e-8 (250× below 1e-5). 1e-5 rad ≈ 0.0006° — numerical-noise tolerance, not a physical heading threshold. |
| 4 | Negative controls still fail under this artifact | **确认** | shortcut + command-demotion both fail; failures land on `rate_budget` / `pose_continuity` / `endpoint_bridgeability` / `command_response` / `support_honesty`. Heading-band relaxation does **not** rescue them (they fail on other families). |
| 5 | **Stage2 was blocked _purely_ by a heading-band 假象; the fix is "relabel heading, rerun Stage2"** | **推翻 (材料不足 → 结论错)** | Per-metric fail set at stage1-best shows **three** zero-slack pathologies; heading is the largest-margin one but only **2/8** windows fail on heading alone. Relabeling heading only would leave **6/8** windows still preflight-blocked. See §2. |
| 6 | "No multimodality claimed from this artifact" | **确认** | Correct restraint; deterministic fixed-schedule run cannot exclude contract-width/optimization. Hold the line. |

The §4 stall-classification table in the execution doc labels **all 8** windows
"heading-band preflight block." That label is produced by
`heading_block = any("heading_error" in m for m in fail)` (L1259) — it fires when
heading is *among* the failures, and **does not assert heading is the only
failure**. The doc reads it as "heading is the sole blocker." It is not.

---

## 2. Corrected diagnosis — three zero-slack classes, not one

Preflight (`stage1_fit_ok`, L1071–1074) requires **every** metric in **every**
window to pass the accepted-p99 band. At the near-GT stage1-best output
(residual ~3.4e-4 state), the failing metrics decompose into three classes:

### Class A — realized-vs-commanded heading (the headline, but two metrics)
| metric | kind | band | stage1 raw | slack |
|---|---|---|---|---|
| `heading_error_p95_rad` | upper | **1e-5 floor** | 3.0–6.2e-5 | −2.0 … −5.2 |
| `support_side.heading_error_p95_rad` | **interval, GT-exact [~1.5e-8, 5e-8], NO floor** | ~4e-8 | 3.0–6.2e-5 | **−619 … −1673** |

The catastrophic blocker is `support_side.heading_error_p95_rad`: it is scored
against a GT-exact interval (~4e-8) and the `--heading-tolerance-rad` floor is
applied **only** to the command gate term (L924), **not** to this support-side
metric. Wiring asymmetry. Both are the same physical quantity (realized heading
vs command), both amplified from the 3.4e-4 state residual.

### Class B — interval bands whose edge **is** the window's own GT value (zero-slack interval edges)
GT sits exactly on the per-clip min/max envelope edge; the 3.4e-4 residual tips
microscopically outside. All `support_side_correctness` family:
| metric | interval | stage1 raw | slack |
|---|---|---|---|
| `support_side.root_lateral_mean` | [−0.4409, −0.2173] | −0.4409 | −1.8e-5 |
| `support_side.right_rel_z_mean` | [0.03064, 0.1254] | 0.03064 | −3.1e-6 |
| `support_side.right_rel_norm_p95` | [0.2881, 0.4653] | 0.2881 | −2.6e-7 |
| `support_side.right_rel_y_mean` / `right_rel_x_mean` | (lower edge) | ≈GT | ~−1e-6 |
| `support_side.claimed_support_slip_p95_mps` | [1.096, 2.302] | 1.095 | −6.6e-4 |
| `support_side.claimed_support_slip_mean_mps` | (lower edge) | ≈GT | −8.9e-5 |

### Class C — per-clip upper bands evaluated on **sibling** windows (cross-window zero-slack)
The band was kept at the single-window value; a *different* window of the same
clip has near-GT prediction at/above that band:
| metric | band | stage1 raw | slack | window |
|---|---|---|---|---|
| `yaw_rate_step_abs_p95` | 0.13711 | 0.13712 | −7.4e-5 | Walk_L_To_L:3-18 |
| `angvel_component_p95_p95` | 0.59697 | 0.59697 | −1.5e-5 | Walk_R_To_R:37-52 |

### Per-window blocker map (why "relabel heading only" fails)
| window | heading only? | also blocked by |
|---|---|---|
| Walk_L_To_L:6-21 | ✅ heading-only | — |
| Walk_L_To_L:17-32 | ✅ heading-only | — |
| Walk_L_To_L:3-18 | ❌ | yaw_rate_step (C), right_rel_z + right_rel_norm (B) |
| Walk_R_To_L:0-15 | ❌ | right_rel_y, root_lateral (B) |
| Walk_R_To_L:46-61 | ❌ | right_rel_z, root_lateral (B) |
| Walk_R_To_L:70-85 | ❌ | right_rel_x (B) |
| Walk_R_To_R:37-52 | ❌ | angvel_component (C) |
| Walk_R_To_R:77-92 | ❌ | claimed_support_slip ×2, root_lateral (B) |

→ Relabeling Class A alone unblocks **2/8**. The other 6 stay preflight-blocked
and Stage2 would skip again. **The execution's "next decision" would burn a cycle.**

### Unifying read
This is the **same** pathology the 2026-06-04 band audit fixed for 6 upper metrics
(rootvel / bone_angvel / foot_slip), now surfacing across **more metrics and more
band kinds** because we moved from 1 window to 8: every band that was pinned to a
single window's GT has **zero slack** for (a) sibling windows' GT and (b) a near-GT
prediction's O(3e-4) residual. The band audit was **incomplete** — it only
relabeled the metrics that were zero-slack on the one window. It never touched
interval bands or the heading metrics.

This is also the **first genuine cross-window signal**: Class C proves a per-clip
band calibrated on one window does not envelope a sibling window's GT. That is a
band-generalization fact, surfaced before any minimax/model-generalization fact.

---

## 3. Direction

Do **not** rerun Stage2 yet. The discriminating step is a **full-family GT/decoder
exactness preflight on all 8 windows**, then a complete band relabel across all
three classes, each guarded. Only then is Stage2 a real test.

Map to the project's three band classes (status §discipline 9):
- **Class B + C → "GT-exact zero-slack" → convert to baseline-percentile.** Relabel
  using the *pooled cross-window continuous baseline* (p1/p99 interval envelope for
  interval metrics; p99 upper for upper metrics), giving GT genuine slack — **not**
  "just enough to clear stage1's residual."
- **Class A (heading) → "no-baseline principled tolerance" (third class).** Set one
  tolerance in the **seam between {GT / decoder-replay} and {shortcut + command-
  demotion negative controls}**, applied to **both** `heading_error_p95_rad` and
  `support_side.heading_error_p95_rad` (fix the floor asymmetry).

Guard every relabel: GT passes, decoder-replay passes, **all** negative controls
**still fail**, guard identity holds. The discriminative load is carried by the
negative controls failing on their own families — confirm that per-control, don't
assume it.

---

## 4. Next execution prompt (self-contained — paste to executor)

> **Role / constraints.** Debug-only. Do **not** modify production trainer / runtime /
> gate / checkpoint. Do **not** reopen representation / entanglement / lifting /
> diffusion / yaw-prediction. Every band change must be re-guarded (GT pass, decoder
> pass, **all** negative controls still fail, guard identity holds). Read the doc and
> the numbers below before acting; if any number you reproduce disagrees with what is
> stated here, **stop and report the discrepancy first** — do not "fix" silently.
>
> **Background.** 8-window debug train-fit (flat `state281`, deterministic decoder,
> oracle support/contact schedule, 3 causal items, adjusted guard + 2026-06-04 band
> relabels). Stage1 supervised-fit-8 reached the GT basin
> (`flat_standardized_mse=5.43e-10`, `max_abs_delta_to_GT=3.4e-4`) but Stage2 minimax
> was **skipped** because preflight requires every metric in every window to pass the
> accepted-p99 band (`stage1_fit_ok`, tool L1071–1074).
>
> **Read first:**
> - `docs/aperiodic_transition/2026-06-05_8window_preflight_band_verdict_and_next_prompt.md` (this verdict — §2 has the per-metric fail table)
> - `docs/aperiodic_transition/2026-06-05_8window_train_fit_review.md`
> - `docs/aperiodic_transition/2026-06-04_band_audit.md`
> - `docs/aperiodic_transition/2026-06-01_middle_generator_acceptance_contract.md` (§C support honesty, §D command response)
> - `tools/run_action_handoff_inbetween_8window_train_fit_debug.py`
> - artifact dir `debug_output/_tmp_action_handoff_8window_train_fit_beststage1_20260605/`
>
> **Established decision (do not re-litigate).** The preflight block is **not**
> heading-only. At stage1-best, 6/8 windows also fail non-heading near-boundary
> metrics. Three classes (see §2 table):
> - **A. heading** — `heading_error_p95_rad` (upper, 1e-5 floor, slack −2…−5) and
>   `support_side.heading_error_p95_rad` (interval, GT-exact ~4e-8, **no floor**,
>   slack −619…−1673).
> - **B. interval edges = window GT** — `support_side.root_lateral_mean`,
>   `right_rel_{x,y,z}_mean`, `right_rel_norm_p95`, `claimed_support_slip_{p95,mean}_mps`
>   (slacks −1e-7…−1e-3).
> - **C. per-clip upper on sibling window** — `yaw_rate_step_abs_p95` (Walk_L_To_L:3-18),
>   `angvel_component_p95_p95` (Walk_R_To_R:37-52) (slacks −1e-5…−7e-5).
>
> **Discriminative experiments (in order):**
>
> **(E1) Full-family GT + decoder-replay preflight on all 8 windows.** Score exact
> `true_raw` and decoder-replay-from-GT for all 8 windows under the accepted bands,
> **all 41 metrics** (not just heading). Output per-window per-metric: raw value,
> band (kind + edges), normalized slack, pass. Confirm GT passes (boundary-inclusive)
> and report, for every Class-B/C metric, the **GT slack** (how much headroom the band
> gives GT). Decompose each stage1-best fail into:
> - **(i) band-too-tight-for-cross-window-GT** — GT itself fails, or sits exactly on a
>   zero-slack edge.
> - **(ii) fit-residual amplification** — GT passes with margin, stage1-pred fails.
> Report the count in each bucket per metric.
>
> **(E2) Class B/C relabel via pooled cross-window baseline.** For every Class-B/C
> metric, recompute the band from the **pooled continuous Walk_F + turn baseline**
> (same source the 2026-06-04 band audit used): p99 for upper metrics, p1/p99 envelope
> for interval metrics — so GT gets genuine slack from a real distribution, **not** a
> margin tuned to clear stage1's 3.4e-4 residual. Tabulate old band → new band → basis,
> exactly like band_audit Step 2. Reject any relabel that does not also keep negative
> controls failing.
>
> **(E3) Class A heading principled tolerance.** Compute `heading_error_p95_rad` and
> `support_side.heading_error_p95_rad` for: GT (`true_raw`), decoder-replay-from-GT,
> and **each** negative control (shortcut family + command-demotion family). Report the
> full sorted list. Set **one** principled tolerance strictly in the seam:
> `max(decoder-side heading) < tol < min(shortcut/command-demotion heading)`. Apply it
> as a floor to **both** the command and the support-side heading metrics (fix the
> L924 asymmetry — support-side must receive the same floor). If no clean seam exists
> (some negative control's heading ≤ decoder's), **report it and stop** — that means
> heading is non-discriminative and must be documented as a passenger acceptance check
> (GT-pass + gross-violation catch), not a relabel-to-pass.
>
> **(E4) Re-guard, then rerun Stage2 for real.** With A+B+C relabels applied:
> re-verify one-window full-family pass (regression), GT pass 8/8, decoder pass 8/8,
> **all** negative controls still fail (report per-control failed families — confirm
> the relabels did not flip any control to pass), guard identity `max_abs_seq_delta=0`.
> Then run Stage2 minimax-8. Report per window: dual band status (**p99 accepted pass
> + p95 shadow**), final normalized slack per metric, and **stall classification**:
> optimization/capacity (low-MSE basin reachable, slack improves monotonically) vs
> conflict-window / candidate-multimodal (minimax oscillates, two windows pull opposite
> signs on a shared metric). Do **not** escalate to sampling/multimodality from a
> deterministic fixed-schedule run — flag candidates only.
>
> **Pass/guard criteria (every step):** GT 8/8 pass · decoder 8/8 pass · one-window
> regression pass · **all** negative controls still fail (per-control evidence) · guard
> identity `max_abs_seq_delta=0` · no band tightened, no band widened past its pooled
> baseline percentile · heading tolerance inside the decoder↔shortcut seam.
>
> **Outputs:** artifact dir `debug_output/_tmp_action_handoff_8window_preflight_bandfix_<date>/`
> with `gt_decoder_fullfamily_preflight.{csv,json}` (E1), `band_relabel_classBC.csv`
> (E2), `heading_seam_audit.{csv,json}` (E3), Stage2 `per_window.csv` / `per_metric.csv`
> / `stage2_minimax_step_log.csv` / `stall_classification.csv`, and `summary.{json,md}`.
> Doc: `docs/aperiodic_transition/<date>_8window_preflight_bandfix_review.md` reporting
> E1 buckets, A/B/C relabel tables with guard columns, Stage2 per-window dual-band +
> stall classification, and an explicit statement of whether the 8-window minimax
> generalization question is now **answered** or still deferred.

---

## 5. Standing-discipline check on this verdict

- **表征怀疑放最后** ✅ — diagnosis is band/wiring, not representation.
- **可行性 ≠ 泛化** ✅ — Stage1 GT-basin is memorization; not banked as generalization.
- **每个 band 改动保持负控 fail** ✅ — required per-control in E2/E3/E4.
- **判定对全 6 family + 负控,不挑子集** ✅ — caught exactly the subset-hollow read
  (heading-only) the execution doc fell into; E1 forces full-family.
- **不 bank 空心 pass** ✅ — bands relabeled from pooled baseline, not from stage1
  residual; heading tol in a real seam.
- **症状 vs 机制** — heading/support metrics are symptoms; the mechanism is zero-slack
  per-window band edges + a residual-amplifying derived metric + a missing floor.
- **执行 verdict 先独立复推** ✅ — recomputed every load-bearing number from
  `per_metric.csv` / heading audit JSON; overturned claim #5.
- **band 分三类** ✅ — B/C → baseline-percentile, A → no-baseline principled tolerance.
