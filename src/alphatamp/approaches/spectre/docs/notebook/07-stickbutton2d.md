# SPECTRE Notebook — StickButton2D as a second environment

2 entries, 2026-08-01 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

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

