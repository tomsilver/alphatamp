# Pre-registration — restock3d_v3 coverage-bug fix-only retrain

**Written before the retrain result lands** (analytic-only pipeline; no real MP). Recorded so the
outcome cannot be reinterpreted post-hoc. Folds into the notebook entry once results are in.

---

## RESULT (2026-08-21, retrain complete — predictions CONFIRMED)

Test n=100, 3 seeds, identical recipe, only the canonicalize bug fixed:

| arm | adaptive | static | adaptive − static (paired) |
|---|---|---|---|
| evidence-knockout (buggy, coverage≡0) | 11.11 ± 0.98 | 11.05 ± 0.88 | +0.06 [−0.11, +0.25] |
| **fix-only (coverage revived)** | 12.18 ± 0.33 | 12.27 ± 0.13 | **−0.09 [−0.32, +0.12]** |

- **Prediction (a) CONFIRMED — fix-only recovers ≈0 adaptivity.** adaptive − static = −0.09, CI
  includes 0 (still inert, like the knockout's +0.06). Per-stratum: only n=9 hints (−0.50
  [−1.29, +0.19]), none CI-clean.
- **Prediction (b) CONFIRMED — the revived signal is live-but-misleading and slightly HURTS.**
  fix-only vs knockout: adaptive **+1.07 [+0.25, +1.92]**, static +1.22 [+0.40, +2.11] — both
  CI-clean *worse*. The wrong-language F2 coverage (ordering of an assignment decision) is the
  ~1 %-value channel, and feeding it live degrades rather than helps (DD2D-s1 lesson).
- **Not overfitting.** Both arms' selected val_fp ≈ 7.6–8.2 (buggy 7.63/8.17/7.57, fixed
  7.83/7.85/7.63); the ~4-FP val↔test gap is a split-difficulty artifact present in both, and the
  buggy model reaching the same val with coverage≡0 proves coverage contributes ≈nothing to v3.
- **The fix is still correct** (a real bug: coverage must not be silently zeroed) and is a
  prerequisite where coverage IS the right language (DD2D). On v3 it is inert-to-slightly-harmful,
  which is exactly why the lever is `repeat` (F3, the 74 % coverage can never see), not the fix.

## Setup being tested

The canonicalize `None`→`[]` coercion silently re-typed v3's class-1 culprit records (F2/F4,
`dev_added=None`) as class-2-with-empty-deviation, emptying the culprit pool `K` → **coverage/waste
were identically 0** on v3. Fixed (`_rename_atoms` preserves `None`). Now retraining the **deployed
recipe with only the bug fixed**, 3 seeds, identical to the buggy arm otherwise. `repeat`/`regroup`
are **not** added yet — this rung isolates the coverage revival.

**Control (evidence-knockout arm):** the buggy checkpoints = SPECTRE adaptive trained *and* evaluated
with the scalar evidence forcibly zero → **adaptive 11.11 ± 0.98 ≈ static 11.05 ± 0.88 (Δ +0.06
[−0.11, +0.25])**. This is the clean "adaptive pathway contributes nothing when the scalars are
blank" control; kept as a named arm.

**Ceiling (P2 oracle, 0 soundness violations):** FP_static 11.05 → FP_oracle-strict **2.81** (ALL),
headroom **8.24 FP (75%)**; per-stratum harvest 38 / 60 / 80 / 76 % at n=6/7/8/9.

## Pre-registered predictions (fix-only, coverage revived, waste still vacuous)

**(a) Harvested fraction is SMALL.** Post-fix, coverage varies across candidates in only ~10–20 % of
F2-bearing contexts (constant elsewhere; P1), and it asks an *ordering* question ("discharge the
resident before re-placing the target") of an *assignment* decision — worse, `place_tall`/
`place_short` have identical abstract effects so the level bit is invisible. **Expected: adaptive
moves only modestly below static; harvested fraction `(FP_static − FP_adaptive)/8.24` < ~0.25.**
*Falsifier:* if fix-only recovers **≥ half** the headroom (≥ 4.1 FP; adaptive ≤ ~7), coverage carries
more than expected → `repeat`/`regroup`'s marginal value shrinks and the ablation ladder should expect
smaller deltas.

**(b) The negative is LIVE and directional.** Post-fix coverage is a *real* signal but plausibly
**wrong-polarity on F2**: "discharge the resident earlier" is not the crowding fix (you want *fewer*
objects on the level, not a resident handled sooner). By the DD2D-s1 precedent, live-but-misleading
evidence can *hurt*. **Pre-registered reading: if fix-only HURTS at some strata (adaptive CI-clean
worse than static there), that counts as EVIDENCE FOR the language-mismatch thesis** (coverage speaks
ordering; v3 needs grouping), **not** as evidence against failure-conditioned evidence in general.
**Read the paired comparison PER-STRATUM, not only ALL.** Stated now to forbid post-hoc scrambling.

## P2 decomposition by certificate type (measured, pre-retrain) — apportions repeat vs regroup

Rerunning the oracle with one certificate family at a time (0 soundness violations for both):

| pruning | ALL | n=6 | n=7 | n=8 | n=9 | harvested |
|---|---|---|---|---|---|---|
| static | 11.05 | 1.77 | 3.44 | 12.69 | 26.29 | — |
| **F3-only (repeat)** | **2.86** | 1.13 | 1.37 | 2.63 | 6.29 | **74 %** |
| F2-only (regroup) | 10.99 | 1.73 | 3.44 | 12.63 | 26.15 | **1 %** |
| both (oracle) | 2.81 | 1.09 | 1.37 | 2.59 | 6.19 | 75 % |

- **`repeat` (F3) carries ~the entire ceiling (74 % alone, at every stratum).** F3 is a per-block
  property (a tall block can't go short) that recurs across many pool candidates → constant pruning.
- **`regroup` (F2) is worth ~1 % everywhere** — the over-packed grouping is near-unique per candidate,
  so it rarely recurs. **Not** concentrated at n=8/9. `regroup`'s pre-registered target is therefore
  ~1 FP; do not expect it to move the headline.
- **F2-as-exact-step → 263 killed successes** — empirically proves `repeat` must NOT fire on F2.

**Consequence for the fix-only prediction:** the coverage bug fix revives the F2/culprit channel
(coverage/waste), which the oracle prices at **~1 %**. F3 (the 74 %) is blameless — coverage can
*never* see it. So **fix-only is expected to recover ≈ nothing** (≤ ~1 FP), and the real lever is
`repeat` (a new F3 column), not the coverage fix. This tightens prediction (a): harvested fraction
≈ 0, not merely "small."

## Cross-env scope correction (measured, pre-build) — `repeat` is NOT "provable ∧ culprit-free"

Blame-structure census (blame==∅ = intrinsic certificate candidate):

| env | provable ∧ empty-culprits | repeat-eligible (blame==∅) | regroup (class-1) | cov/waste (class-2) |
|---|---|---|---|---|
| restock3d_v3 | 78 % | 78 % (=F3) | 22 % | 0 % |
| dd2d_v4 | 92 % | **92 %** | 7 % | 0 % |
| stickbutton2d | 75 % | 38 % | 0 % | 43 % |

"provable ∧ culprit-free" is **not env-safe**: DD2D's 92 % blameless-provable records are
*means-failures* (an exhausted pick/place sample), not intrinsic step certificates; scoping `repeat`
there would vote against candidates containing feasible steps (wrong polarity). SB2D's "GOTCHA"
(empty `culprits` but blame≠∅ via deviation) shows the predicate must be `blame==∅`, not empty-
`culprits`. The discriminator between a certificate and a means-failure is `proof_tier`
(monotone ∧ local ∧ exact).

**Env-safe scope: `repeat = proof_tier(schema) ∧ provable ∧ blame==∅`.** v3 F3 qualifies **only if
v3 declares a DomainSpec** marking `place` height-failure proof-tier (currently `EMPTY_SPEC`) — a
required Stage-1 task, with care that it not spuriously fire the `dead`/demotion path on F2 (gate that
with `blame==∅` too, or use a repeat-dedicated axiom flag). DD2D → only blameless `retrieve` (~1 %,
already proof-tier) qualifies → `repeat` ≈ inert (side-hypothesis holds). SB2D `EMPTY_SPEC` → `repeat`
never fires → inert.

## Decision rule (independent of the fix-only outcome)

Proceed to build `repeat` (**`proof_tier ∧ provable ∧ blame==∅`**, F3 — the 74 % workhorse, needs a
v3 DomainSpec) as the primary lever; `regroup` (culprit-bearing F2 chart) is a ~1 % secondary, kept
for cross-env vocabulary completeness, not the headline. Fix-only is a lower rung of the ablation
ladder. The evidence-knockout control (buggy arm) anchors the bottom.
