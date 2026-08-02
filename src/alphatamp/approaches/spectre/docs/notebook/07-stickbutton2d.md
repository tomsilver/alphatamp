# SPECTRE Notebook — StickButton2D as a second environment

3 entries, 2026-08-01 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

---
<a id="2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage"></a>
## 2026-08-01 — DD2D compare cache rebuilt to the unified coverage/waste definition (7.44 to 5.78)

<!--strip-->
> **id** `2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage` · **status** active
> · **tracks** evaluation, env-dd2d
<!--/strip-->

**What.** The DD2D method-comparison notebook was still reporting SPECTREv3-adaptive at
**7.44** — the pre-unification coverage/waste definition — even though the deployed
checkpoint has been `checkpoints_v3_unified` (unified coverage/waste) since 2026-07-31 and
`spectre_score_v3` already reported **5.78 ± 0.10** for it. The gap was a stale cache, not
a disagreement: `_V3_ARMS["spectre3"]` was repointed to the unified dir on 2026-07-31, but
`precompute_dd2d_cache._dir_complete` skips any full directory, so the pre-unified
`spectre3_{static,adaptive}` compare-cache rows were never overwritten.

**Result.** Rebuilt with
`precompute_dd2d_cache.py --env-variant dd2d_v4 --methods spectre3 --no-ablations --force`
(CPU, LM Studio holding the GPU). The notebook headline is now:

| | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| SPECTREv3-adaptive | **5.78 ± 0.10** | 0.00 | 3.44 | 10.49 | 9.19 |
| SPECTREv3-static | 21.10 ± 2.11 | 0.00 | 14.21 | 27.76 | 42.44 |

matching the `spectre_score_v3` figures exactly (adaptive was 7.44, static 20.66 under the
old definition). The 3× win over PIGINet (17.27) and the ~11.5 FP margin over v2.2 that the
score instrument already showed are now what the notebook renders too.

**Takeaway.** No new science — this only propagates the 2026-07-31 definition
([`decisions/06`](../decisions/06-v3-performance.md#2026-07-31-unified-coverage-waste-is-the-deployed-definition))
to the cache the notebook reads, so a future reader does not see two different v3 numbers
depending on whether they ran the score tool or opened the notebook. **The §4 ablation
arms were deliberately *not* rebuilt** (`--no-ablations`): they predate the unification and
score under the old definition by design, as a matched-settings seed-0 study. That makes
§4's `deployed` row (now unified, ~5.78) not directly comparable to its matched
`cov+waste, tokens` arm (~7.90, old) — the note in §4 and the `DD2D` registry caveat both
now say so. The standing lesson stands: **re-cache with `--force` whenever an arm is
repointed**, because `_dir_complete` keeps a stale full directory.

---

<a id="2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar"></a>
## 2026-08-01 — SB2D VLMPlan-32B: capable but tail-limited; beats astar, loses to learned methods

<!--strip-->
> **id** `2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar` · **status**
> active · **tracks** baselines, evaluation, env-stickbutton2d
<!--/strip-->

**What.** Add the VLMPlan-32B row (the zero-training-data corner) to the SB2D four-method
comparison. Scored on a **stratified 40-problem subset** (10/stratum, test split) rather
than the full 100 — a compute choice, since b3/b5 problems VLMPlan cannot self-solve run
to the ~10-round stall cap. `qwen3-vl-32b-instruct`, corrected prompt (domain grounding +
effector-chaining rule), stop-at-first-success on. Label-agreement gate **1.000** (36
samples).

**Result — the four-method table.** Mean rollout FP, test; VLMPlan n=40 (10/stratum, so
its stratum-weighted ALL is comparable to the n=100 rows), others n=100, 3 seeds.

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| SPECTREv3-adaptive | **1.69** | 0.08 | 0.24 | 1.13 | 5.29 |
| SPECTREv3-static | 1.98 | 0.08 | 0.32 | 1.52 | 5.99 |
| PIGINet | 2.02 | 0.08 | 0.32 | 1.31 | 6.39 |
| **VLMPlan-32B** | **13.18** | 0.70 | 1.30 | 6.20 | 44.50 |
| astar-dist | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |

Generation: **35/40 solved from VLMPlan's own proposals** (5 fell back to published
order), **0 censored**. Per-problem FP is heavily right-skewed — 22/40 problems are
**FP=0** (the first proposal refines):

| stratum | self-solve | per-problem FP (sorted) |
|---|---|---|
| b1 | 10/10 | 0,0,0,0,0,0,0,0,0,7 |
| b2 | 10/10 | 0,0,0,0,0,0,0,2,4,7 |
| b3 | 10/10 | 0,0,0,0,1,5,5,15,15,21 |
| b5 | 5/10 | 0,0,8,13,23,32,62,66,91,150 |

**Takeaway — VLMPlan-32B is a genuine planner here, sitting between astar and the learned
methods, and the pilot badly mis-estimated it.** Three points, the first a correction.

1. **The 2-problem pilot was wrong, and this is why 10/stratum was the right call.** The
   pilot drew train problems 500000 (b3) and 750000 (b5), both in the hard tail: 0
   self-solves, FP 34 and a censored 200. From those two I told the summary VLMPlan
   "loses to astar on b3/b5, censored on b5." The stratified test sample overturns it:
   0 censored anywhere, VLMPlan self-solves the *median* problem in one proposal, and it
   **beats astar-dist overall** (13.18 vs 16.29). An earlier registry caveat asserting
   the pilot reading was corrected in the same commit. Two hard problems are not a row.

2. **It beats the naive planner order but only via b5, and loses to it everywhere else.**
   VLMPlan is worse than astar on b1/b2/b3 (0.70 vs 0.08, 1.30 vs 0.56, 6.20 vs 2.96):
   its off-pool proposals are refined for real and charged as attempts, so its
   charged-but-failed guesses cost it exactly where the pool order is already near-
   optimal. It wins on b5 (44.5 vs 61.6) only because astar's *default* order is
   pathological there (61.56). The overall win is a b5 artefact of a weak baseline, not
   broad superiority.

3. **The representation gap is the headline, and it is wide.** VLMPlan-32B (13.18) trails
   SPECTREv3 (1.69) and PIGINet (2.02) by ~7×. The zero-data corner is a real, competent
   point — 35/40 self-solved — that the trained abstract-first and low-level predictors
   both dominate. That is exactly the framing [`proposal.md`](../proposal.md) §0 wants:
   VLMPlan answers "did you try just asking a VLM?" on the record, as a corner of the
   data axis, not a defeated straw man.

**Next.** The row is n=40; the full 100 would tighten b3/b5 (their tails are what the mean
rides on) but cannot move the ~7× representation gap or the qualitative ordering. Left as
a deliberate stopping point unless the paper needs n=100 parity on this row.

---

<a id="2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed"></a>
## 2026-08-01 — SB2D v3 ablation arms: training is not reproducible from the seed alone

<!--strip-->
> **id** `2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed` ·
> **status** active · **tracks** evaluation, baselines, env-stickbutton2d
<!--/strip-->

**What.** Train the six v3 component arms on StickButton2D so §4 of the comparison
notebook — the coverage × record-tokens 2×2 plus the single-column split — has something
to render on the second environment. Same flags as DD2D's arms, `spectre_sweep.py --preset
sb2dabl`, one seed each (the project's 1-seed dev convention), cached via
`precompute_dd2d_cache.py --env-variant stickbutton2d_v1`.

The demotion pair was deliberately omitted: SB2D resolves to `EMPTY_SPEC`, so
`licenses_demotion` is always false and the two caches would be bit-identical. Vacuous
here, not overlooked.

**Result — the arms, seed 0, mean rollout FP on the 100-problem test split.**

| arm | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| deployed (`spectre3`) | **1.76** | 0.08 | 0.32 | 1.00 | 5.64 |
| cov+waste, no tokens | 1.77 | 0.08 | 0.36 | 1.20 | 5.44 |
| waste column only | 1.92 | 0.08 | 0.32 | 1.32 | 5.96 |
| coverage column only | 2.13 | 0.08 | 0.40 | 1.28 | 6.76 |
| no cov/waste, tokens | 2.53 | 0.08 | 0.48 | 1.60 | 7.96 |
| **cov+waste, tokens** (`abl_cov_rec`) | **2.78** | 0.08 | 0.24 | 2.92 | 7.88 |
| neither (no cols, no tokens) | 2.89 | 0.08 | 0.40 | 1.40 | 9.68 |

**Result — the finding, which is about the instrument and not the arms.** `deployed` and
`abl_cov_rec` are **the same flags at the same seeds**. They were trained twice by
accident — the deployed arm from the sweep, the ablation arm from the `sb2dabl` preset —
and they read **1.76 vs 2.78** at seed 0, a gap of **1.02 FP**. Over three seeds the pair
reads 1.69 ± 0.26 vs 2.00 ± 0.28.

Every ablation gap in the table above is smaller than 1.02.

**Takeaway — SB2D's §4 does not separate, and the table must be read against run-to-run
noise rather than against the seed sd.** Three things follow.

1. **No arm ordering in the SB2D 2×2 should be quoted.** The accidental duplicate is a
   free null-effect control: it measures what the pipeline reports for a contrast that is
   *known* to be zero, and it reports 1.02 FP. The largest real contrast here (`neither` −
   `deployed` = 1.13) barely clears its own noise floor.
2. **The seed sd understates the uncertainty.** ±0.26 across three seeds is the spread of
   *one* training run per seed; it does not contain the between-run variance at fixed seed,
   which is roughly four times larger. This is a sharper version of the standing rule that
   a load-bearing per-stratum margin is compared to the seed sd
   ([`decisions/06`](../decisions/06-v3-performance.md#2026-07-27-margin-must-be-compared-to-seed-sd)):
   on this environment even the seed sd is the wrong yardstick.
3. **Training is not reproducible from the seed alone.** Not diagnosed further — likely
   CUDA nondeterminism in the tensorization/backward path, which the project has already
   seen at ~2e-4 on inference scores. What is established is the *magnitude of its
   consequence* on a low-FP environment: where DD2D's means sit near 6–17 FP, SB2D's sit
   near 2, so the same absolute jitter is an order of magnitude more of the signal.

The finding is recorded in `compare_envs.SB2D.caveats`, so it renders under §1 of the
notebook rather than living only here.

**Next.** The DD2D §4 numbers are *not* retroactively suspect — DD2D's contrasts run
1–5 FP against means near 15, and its own arms were never duplicated so no null control
exists there. Establishing whether the same jitter applies at DD2D's scale would need one
deliberate duplicate run, which is ~17 min and is the cheapest thing that would firm up
every 1-seed ablation the project has published.

---

<a id="2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity"></a>
## 2026-08-01 — SB2D comparison: v3 and PIGINet are indistinguishable; adaptivity is the only separation

<!--strip-->
> **id** `2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity` ·
> **status** active · **tracks** baselines, evaluation, method, env-stickbutton2d
<!--/strip-->

**What.** Stand PIGINet up on StickButton2D and reproduce the DD2D comparison notebook
there — the representation contrast (low-level predictor vs abstract-first re-ranker) on
the second environment. Three seeds each, BCE arm, AUPRC-selected, same 267/90 train/val
and same 100-problem test split as SPECTRE v3, same labels.

**Result — the comparison table.** Mean rollout FP on the test split (n = 100), uncensored;
`sd` is across the three seeds.

| method | ALL | sd | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|---|
| astar-dist (planner order) | 16.29 | — | 0.08 | 0.56 | 2.96 | 61.56 |
| PIGINet | 2.02 | 0.19 | 0.08 | 0.32 | **1.31** | 6.39 |
| SPECTREv3-static | 1.98 | 0.28 | 0.08 | 0.32 | 1.52 | 5.99 |
| SPECTREv3-adaptive | **1.69** | 0.26 | 0.08 | 0.24 | 1.13 | **5.29** |

Paired bootstrap over the 100 test problems (seed-averaged per problem, 10 000 resamples):

| comparison | Δ | 95% CI | separates |
|---|---|---|---|
| v3-adaptive − PIGINet | −0.337 | [−0.723, +0.053] | **no** |
| v3-static − PIGINet | −0.047 | [−0.437, +0.353] | **no** |
| v3-adaptive − v3-static | −0.290 | [−0.517, −0.073] | **yes** |
| PIGINet − astar-dist | −14.267 | [−21.383, −7.970] | **yes** |

**Takeaway — on StickButton2D the representation advantage does not reproduce; adaptivity
is the only thing that separates.** Two readings, and the second is the load-bearing one.

1. Both learned methods crush the planner order (−14.3 FP for PIGINet alone). The
   feasibility-prediction problem is real here and learning solves a lot of it.
2. **SPECTRE v3 and PIGINet are statistically indistinguishable**, in *both* deployment
   modes. The abstract-first representation buys nothing measurable over the low-level one
   on this environment (v3-static − PIGINet = −0.05, CI spanning zero). What *is*
   significant is the adaptive increment within SPECTRE: −0.29 FP, CI excluding zero.

That inverts DD2D's attribution. There the static representation carried ~73% of the margin
and adaptivity ~27% (`notebook/01` 2026-06-06). Here the static representation carries
**none** of it and adaptivity carries all of it. The pivot's framing — "abstract-first is
the leading candidate, adaptivity is a secondary composable increment" — survives DD2D and
does not survive this environment unchanged.

**Three caveats, all of which cut in PIGINet's favour and none of which rescue the claim.**

- **PIGINet's image channel is degenerate here by construction.** Every unpressed button is
  the same red disc, so CLIP separates only {button, stick, robot} — information the type
  literals already carry. PIGINet matches v3 *despite* getting nothing from pixels; its
  pose/shape channels are doing the work. An environment with informative perception would,
  if anything, favour it more.
- **`at-pose` literals are synthesised by our adapter**, not stored. SB2D's abstract initial
  state names no positions, so a low-level predictor would otherwise receive none. This is
  a deliberate construction to make PIGINet a fair comparator rather than a strawman; it is
  also the single largest discretionary choice in the port.
- **b5's train split is 17 episodes** for both methods. Its column is substantially a
  generalisation number. Both share the split, so neither is advantaged — but the b5 gap
  (5.29 vs 6.39) is the least trustworthy cell in the table.

**Sample size, stated rather than glossed.** v3-adaptive − PIGINet is −0.337 with CI
[−0.723, +0.053] — *nearly* separating. This is "indistinguishable at n = 100 and 3 seeds",
not "equal". A larger test split or more seeds could resolve it either way, and that is the
cheapest experiment that would sharpen this row.

**Takeaway/next.** The honest cross-environment statement today is: *the abstract
representation wins on DD2D and ties on StickButton2D, while the adaptive increment is
positive on both.* Before that goes in a paper: (1) finish the b3/b5 train splits so b5 is
not a 17-episode extrapolation; (2) decide whether the near-miss CI warrants more seeds;
(3) note that DD2D's own PIGINet row and this one now come from the same code, verified
unchanged at FP 17.0500.

---

<a id="2026-08-01-sb2d-collection-b1-b5-bracket-v3-1"></a>
## 2026-08-01 — SB2D collection, B1-B5 bracket, and v3 at 1.69 FP

<!--strip-->
> **id** `2026-08-01-sb2d-collection-b1-b5-bracket-v3-1` · **status** active ·
> **tracks** method, evaluation, baselines, env-stickbutton2d
<!--/strip-->

**What.** The `stickbutton2d_v1` collection, its B1–B5 baseline bracket, the coverage
re-ranking gate on the full test split, and v3 trained on it.

**Result — the collection, as collected.** Rejection is DD2D's `reason="unsolved"`
convention (drop a problem with no feasible skeleton, redraw).

| | train | val | test | mean pool | rejected (train/val/test) | CPU-h |
|---|---|---|---|---|---|---|
| b1 | 100 | 25 | 25 | 1.5 | 5 / 3 / 2 | 0.3 |
| b2 | 100 | 25 | 25 | 18.0 | 11 / 1 / 2 | 6.6 |
| b3 | 50 | 20 | 25 | 200 | 7 / 3 / 2 | 36.6 |
| b5 | 17 | 20 | 25 | 200 | — (job stopped at cutoff) | ~40 |

**Test is complete at 25 per stratum**; train and val for b3/b5 are short of the planned
100/25. Measured throughput was 18.6 (b3) and 11 (b5) keepers/h on 12 workers, so the
original targets were an 8 h and a 13.6 h job. Targets were re-budgeted per split at 00:31
with test held at full size, since test sizes the headline. Rejection rates are low
everywhere (≈10% at b2/b3), so the solvable-scene bias this introduces is small.

**Result — B1–B5 on the test split, uncensored.** Mean failed attempts before first
success:

| | B1 random | B2 default | B3 static-hist | B4 adaptive-hist | B5 oracle |
|---|---|---|---|---|---|
| ALL | 21.04 | 16.29 | **6.41** | 22.56 | 0.00 |
| b1 | 0.24 | 0.08 | 0.08 | 0.08 | 0.00 |
| b2 | 5.22 | 0.56 | 0.36 | 0.24 | 0.00 |
| b3 | 47.79 | **2.96** | 9.84 | 26.88 | 0.00 |
| b5 | 30.90 | 61.56 | **15.36** | 63.04 | 0.00 |

Two results here are worth more than the method comparison they were collected for.

**B4 is worse than random on this environment** (22.56 vs 21.04 overall; 63.04 vs 30.90 at
b5). The Naive-Bayes adaptive baseline is SPECTRE's headline comparison on RT2D and DD2D
because it is the strongest non-learned adaptive ranker there. It is *actively harmful*
here. **The strongest baseline on SB2D is B3, static-historical, at 6.41** — so the bar a
learned method has to clear is a *static* one, and the "adaptivity premium" framing that
motivated the original RT2D design does not transfer.

**At b5 the planner's own enumeration order is worse than shuffling** (B2 61.56 vs B1
30.90). Measuring where the feasible plans sit in the pool explains it: at b3 the first
success is at **1.5%** of the pool and all successes average **12.9%** — A* order is
genuinely informative. At b5 all successes average **49.9%** (i.e. uniform) while the
first arrives only at **30.9%**. So the order is not merely uninformative at b5, it is
anti-correlated, and a random permutation finds a feasible plan sooner.

**Result — Gate A (does coverage rank?): PASS at b5, marginal at b3.** Non-learned
re-ranking of the remaining pool as failures accrue, 100 test episodes:

| | n | static | coverage+waste | coverage only | waste only | oracle |
|---|---|---|---|---|---|---|
| b1 | 25 | 0.08 | 0.08 | 0.08 | 0.08 | 0.00 |
| b2 | 25 | 0.56 | 0.56 | 0.56 | 0.56 | 0.00 |
| b3 | 25 | 2.96 | 4.44 | **2.88** | 5.52 | 0.00 |
| b5 | 25 | 61.56 | **25.56** | **25.56** | 61.56 | 0.00 |

Coverage alone cuts b5 by **58%** (61.56 → 25.56), better on 20/25 problems paired, worse
on 1. At b3 it is a wash on the mean (2.96 → 2.88) while winning 12/25 and losing 1 — the
mean is dragged by a few large regressions.

**`waste` is not neutral on SB2D, it is inert or harmful.** At b5 `waste_only` reproduces
`static` to the last digit (61.56 both) — completely inert, because every b5 plan has the
same length and therefore the same superfluous-step set. At b3 it *hurts*: 5.52 alone
against static's 2.96, and adding it to coverage as a tie-break degrades coverage from 2.88
to 4.44. This sharpens the registered cross-env dominance flip from "waste carries less
here" to "waste carries nothing here, and can carry negative".

Note the ceiling this sets up: the plain coverage re-ranker at b5 (25.56) is **worse than
B3 static-historical (15.36)**. Coverage is a strong *adaptive* signal and a weak static
one; a method that beats B3 has to combine both, which is what v3 is.

**Takeaway/next.** `waste` earning its place is now an open question on this environment
rather than an assumed yes, so v3 was trained in two arms — the deployed `coverage+waste`
and `--coverage-mode coverage`. Reported below.

**Result — v3 on SB2D, 3 seeds, uncensored test split.**

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| B1 random | 21.04 | 0.24 | 5.22 | 47.79 | 30.90 |
| B2 default (A*) | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |
| B3 static-historical | 6.41 | 0.08 | 0.36 | 9.84 | 15.36 |
| B4 adaptive-historical | 22.56 | 0.08 | 0.24 | 26.88 | 63.04 |
| coverage re-rank (not learned) | — | 0.08 | 0.56 | 2.88 | 25.56 |
| **SPECTRE v3 (deployed)** | **1.69 ± 0.26** | 0.08 ± 0.00 | 0.24 ± 0.08 | **1.13 ± 0.12** | **5.29 ± 1.04** |
| v3, waste column zeroed | 2.04 ± 0.52 | 0.08 ± 0.00 | 0.29 ± 0.12 | 1.85 ± 1.10 | 5.95 ± 1.54 |
| B5 oracle | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

v3 beats the strongest baseline by **3.8x** overall (6.41 → 1.69) and by **2.9x** at b5
(15.36 → 5.29). It is also the only method that is good on *both* hard strata: B2 is the
best baseline at b3 and the worst at b5; B3 is the reverse.

**The waste ablation reverses Gate A's verdict, and that is the interesting part.** As a
hand-coded tie-break waste is harmful at b3 (4.44 against 2.88 for coverage alone). As a
*learned* feature it is worth **+0.36 FP, CI [+0.08, +0.67]**, concentrated at b3 (1.13 vs
1.85). The same column, on the same episodes, helps a model and hurts a rule. The lesson
generalises past this feature: a non-learned probe measures whether a signal is
*monotonically* usable in the direction we guessed, not whether it carries information —
so a failed re-ranking probe is not grounds for dropping a column, and the pass/fail
framing of Gate A was too strong on that point.

**Caveat on b5.** Only **17** b5 episodes are in the training split (the collection was
cut at a wall-clock budget), so b5's 5.29 is largely a *generalisation* result — a model
trained on b1/b2/b3 pools transferring to 5-button pools it barely saw. That is a stronger
claim than the number alone suggests, and also a less stable one; it should be re-measured
on a full b5 train split before it is quoted as a like-for-like stratum result.

**Takeaway/next.** Three things worth doing before this becomes a paper row: (1) finish
the b3/b5 train splits so b5 is not a 17-episode extrapolation; (2) re-run the waste
ablation there, since its value showed up at b3 where data is plentiful; (3) B4 being worse
than random means the "adaptivity premium over B4" framing that motivated RT2D needs
restating for a cross-environment claim — on SB2D the bar is B3, a static ranker.

---

<a id="2026-08-01-stickbutton2d-stood-up-pool-shape-evidence"></a>
## 2026-08-01 — StickButton2D stood up: pool shape, evidence classes, and the two gates

<!--strip-->
> **id** `2026-08-01-stickbutton2d-stood-up-pool-shape-evidence` · **status** active ·
> **tracks** method, data, env-stickbutton2d, evaluation
<!--/strip-->

**What.** Stand up StickButton2D as SPECTRE's second environment end to end: pool filter,
`scene_geometry`, class-2 evidence, pooled `stickbutton2d_v1` variant, and the
400/100/100 collection. Two gates before trusting anything: does the pipeline actually
produce a checkpoint (B), and does coverage still rank on the filtered pools (A).

**Result — pool shape.** Acyclic fraction of a 200-candidate draw, 6 seeds per variant:

| | b1 | b2 | b3 | b5 |
|---|---|---|---|---|
| acyclic / 200 raw | 1–2 | 6–34 | 73–101 | 193–200 |
| acyclic, raw budget 5000 | 1–2 | 6–34 | 200 (≈640 raw) | 200 (200 raw) |
| deployed pool size | ≈2 | 6–34 | 200 | 200 |

Raising the raw budget from 20000 to 5000 changed no pool and cut b1's pool-draw time from
20–61 s to 1–4 s; the 20000 draws were spent enumerating ever-longer padded plans.

**Result — b5's feature degeneracy is structural.** Every b5 acyclic plan has length 6
(1160/1188 skeletons; the rest 7), i.e. 5 presses plus one stick pickup. So
`manipulated = args \ goal_objects = {robot, stick}` for *every* candidate, which pins
`jaccard` and `dead` constant across the pool and collapses the within-length PL loss to a
single bucket. At b5 the only features that can discriminate are `coverage`/`waste` and
the operator/argument token structure. This is the sharpest possible statement of why the
unified definitions were a prerequisite rather than an improvement: on the deployed
`S(c)` formula there would have been *no* usable candidate feature at b5 at all.

**Result — the evidence features do partition a 200-deep pool.** Tensorizing collected
b3/b5 test episodes at `|F| = 3` through the real `build_v3_example`:

| | pool | successes | coverage > 0 | distinct coverage | distinct jaccard |
|---|---|---|---|---|---|
| b3 | 200 | 3–15 | 65–98 | 2 | 1–2 |
| b5 | 200 | 2–9 | 40–69 | 2–4 | **1** |

Three things worth keeping. **Positives are genuinely sparse** — 1–7% of candidates
refine, which is the regime ranking is for. **`jaccard` is constant across the b5 pool**,
confirming the degeneracy above from the data rather than from the argument. And
**coverage is coarse**: with `|K|` of 1–3 it takes 2–4 distinct values, i.e. it is close to
a binary "does this candidate discharge the culprit" rather than a graded score.
Tensorization is not a bottleneck (<0.05 s per 200-candidate episode), so the selector's
per-epoch cost is forward passes, not feature construction.

One episode in six had **all three records blameless** (pure means-failure, no collateral),
giving coverage 0 across the whole pool. That is the case the blameless-record decision
exists to make harmless, and it is not rare.

**Result — Gate B (pipeline produces a checkpoint): PASS.** On 80 train / 20 val b1+b2
episodes, `train_v3` reports `n_train=43 n_val=13` and writes `best.pt`, val_fp improving
1.77 → 1.23 over two epochs. The gap between 80 and 43 is `_trainable`: about half of b1's
episodes have pool size 1. Tensorizing collected episodes through the real
`build_v3_example` gives non-zero, *varying* coverage — 11/15 candidates covered on one
b2 episode, max 1.0 — and record tokens carrying object tags via `dev_blame`. Contexts
whose records are pure means-failure correctly produce coverage 0.

**Result — DD2D is untouched.** `checkpoints_v3_unified` re-scores at **5.78 ± 0.10**
(s0 0.00, s1 3.44 ± 1.36, s2 10.49 ± 0.77, s3 9.19 ± 0.76), identical to the pre-change
figure per stratum as well as overall, after edits to `dataset_v3`, `canonicalize`,
`unified_evidence` and `failure_record`.

**Result — instrumentation is observation-only.** Same-seed differential over b2 and b3,
3 problems × 8 candidates each: `RecordingSampler` and upstream's
`ParameterizedControllerTrajectorySampler` return identical labels, and the recorder
demonstrably captures records (guard against a vacuous pass).

**Takeaway/next.** The porting contract in `porting_guide.md` was incomplete in two ways
that only a real transfer would have found: it assumed a refiner can *name* what it failed
on, and it listed `scene_geometry` as required without saying that its absence produces a
successful-looking training run with no checkpoint. Both are now written up (§2b, §4).

---

