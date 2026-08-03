# SPECTRE Notebook — StickButton2D as a second environment

3 entries, 2026-08-01 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

---
<a id="2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate"></a>
## 2026-08-02 — DD2D s1 wall-clock blow-up diagnosed; per-candidate refinement cap

<!--strip-->
> **id** `2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate` · **status**
> active · **tracks** method, evaluation, env-dd2d
<!--/strip-->

**What.** Investigated why §2b's DD2D wall-clock showed SPECTREv3-adaptive *slower* than the
naive planner order overall (5.89 vs 4.94 s ALL), with the whole gap at s1 (11.99 ± 7.81 vs
0.26 s) — suspicious because v3 wins every other stratum. Then added a per-candidate
refinement cap and re-measured. Protocol/decision in
[decisions/07 2026-08-02](../decisions/07-stickbutton2d.md#2026-08-02-per-candidate-refinement-cap-deployed-wall-clock-configuration).

**Result — the s1 blow-up is real, not a bug, but the number is noisy.**
- Timing math verified; the table reproduces from the cache.
- v3 is genuinely (modestly) worse than astar at **s1 on FP** — 3.44 vs 2.24 (astar's *worst*
  s1 problem is only FP=5). The planner-cost order already ranks s1's short/cheap feasible
  plans well.
- The ~1.2-attempt FP gap becomes a ~46× wall-clock gap because of *which* candidates fail.
  Across all test episodes feasible refinements are uniformly cheap (s1 mean 0.32 s, p95
  0.44 s) while near-feasible infeasible candidates burn the full 20 s budget. astar s1 refine
  = 0.13 s (cheap dead-ends); v3 = 11.2 s (expensive traps). **Worked case pid 1250023**: pool
  200 with **29 feasible @ 0.17 s each**, the model ranked **15 of the 20 s traps ahead of all
  29** → 240 s, FP=15 where random ≈ 6 (*worse than random* on that problem).
- The 12.00 ± 7.80 is a heavy-tailed 3-seed mean dominated by ~4–5 recurring hard s1 problems
  (pids 1250011/1250015/1250023…) whose FP swings 2→15 across seeds.

**Result — a per-candidate cap fixes it, cheaply and safely.**
- Safety: per-candidate (not per-problem), so a problem is lost only if *every* feasible
  candidate exceeds the cap. Min-feasible refine time per problem is mean 0.103 s, **max
  0.243 s** → **0/100** problems censored at any cap ≥ 1 s; the median problem keeps ~20 sub-2 s
  feasible candidates.
- Under a **2 s** cap (deployed), ALL wall-clock to first success (3 seeds):

  | method | ALL | s0 | s1 | s2 | s3 | uncapped ALL |
  |---|---|---|---|---|---|---|
  | **SPECTREv3-adaptive** | **1.79 ± 0.44** | 0.43 | 2.40 | 1.88 | 2.45 | 5.89 |
  | SPECTREv3-static | 2.53 ± 0.71 | 0.44 | 2.04 | 3.14 | 4.50 | 7.99 |
  | astar-dist | 2.96 | 0.40 | 0.26 | 1.35 | 9.81 | 4.94 |
  | PIGINet | 3.14 ± 0.39 | 0.71 | 2.01 | 3.88 | 5.98 | 8.35 |

- v3-adaptive becomes the **fastest** method; its s1 collapses 11.99 → 2.40. **FP cost of the
  cap** (ALL): adaptive +0.05 (5.78 → 5.83), astar +0.00 (failures already sub-cap), PIGINet
  +0.23, static +0.26 — a faithful re-run (the adaptive order diverges on 6/300 cells), not a
  `min(t, cap)` accounting.

**Takeaway — next.** The uncapped wall-clock over-punishes the learned ranker: its few failures
are the *expensive* near-feasible candidates a good ranker still tries, so bounding per-skeleton
refinement (which the cap does) is what lets the "try few candidates" advantage show in seconds.
Do not read an uncapped wall-clock as v3's deployed cost. The **residual** is s1, where v3 still
trails astar (2.40 vs 0.26) — the modest s1 FP deficit — a candidate for the model-side R1
cost/enumeration-index feature (give the ranker the planner-cost order it currently cannot see).

---

<a id="2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s"></a>
## 2026-08-02 — StickButton2D PIGINet crops re-sourced from kinder's renderer (stickbutton2d_v1_kinder)

<!--strip-->
> **id** `2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s` · **status**
> active · **tracks** baselines, env-stickbutton2d, data
<!--/strip-->

**What.** Re-sourced the PIGINet baseline's SB2D image crops from **kinder's own renderer**
instead of the schematic rasteriser (`SB2DDomain.crops`, which drew each object as a lone
polygon on a blank background). Delivered as a new env_variant `stickbutton2d_v1_kinder`,
built by a converter (`experiments/spectre/sb2d_render_convert.py`) that copies every record
verbatim and only re-renders the pixels by resetting the env from the stored seed. Per
problem it materialises per-object crops (`render_2dstate` windows, world side 1.4 m, 300 dpi
→ 420²) plus a full `scene.png`. The reader is a thin `SB2DKinderDomain(SB2DDomain)` selected
by `make_sb2d_domain`. Rationale and the five load-bearing choices are in
[decisions/07 2026-08-02](../decisions/07-stickbutton2d.md#2026-08-02-kinder-rendered-piginet-crops-stickbutton2d-via-new).

**Result — conversion + validation.**
- The kinder crops render correctly and carry **real scene context** — a button crop shows
  the table band, the wall, and the stick tip, not a lone disc. On a multi-button (b5) scene
  the per-button crops are **not** pixel-identical (they differ by position/context), the
  direct contrast to the schematic where every unpressed button is the same red disc.
- Records are copied **byte-identical** (geometry, skeleton pool, outcomes, object registry,
  goal all `==` the v1 source; only `provenance.env_variant` differs), which is what licenses
  grafting SPECTRE from v1. Vocab is identical to v1's; `spectre_check_pipeline` passes.
- `env.reset(seed=pid)` + `render_2dstate` is **deterministic** (re-render reproduces
  identical pixels). All seven unit tests pass in ~1 s.

**Result — the comparison (ALL FP, test n=100, 3 seeds; PIGINet retrained on kinder crops).**

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| SPECTREv3-adaptive | **1.69 ± 0.26** | 0.08 | 0.24 | 1.13 | 5.29 |
| SPECTREv3-static | 1.98 ± 0.28 | 0.08 | 0.32 | 1.52 | 5.99 |
| **PIGINet (kinder)** | **2.28 ± 0.29** | 0.07 | 0.35 | 1.17 | 7.55 |
| astar-dist | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |

Paired bootstrap over the 100 problems (negative = v3 better): v3-static − PIGINet = **−0.31,
CI [−0.95, +0.36]**; v3-adaptive − PIGINet = **−0.60, CI [−1.24, +0.08]** — **neither
separates**. The adaptive increment does: adaptive − static = **−0.29, CI [−0.51, −0.08]**.

**Takeaway — the valid pixels did not overturn the finding; they reinforced it.** With real
kinder crops PIGINet is if anything *slightly worse* than with the schematic (2.28 vs the
prior 2.02, the drop entirely at b5: 7.55 vs 6.39), so the representation advantage **still
does not separate on SB2D**. The honest cross-environment statement is unchanged: the abstract
representation wins on DD2D, ties on SB2D; the adaptive increment is positive on both (−0.29
here, matching the schematic's −0.29). The pre-registered caveat held — the crop's added
context is positional, and since two unpressed buttons are identical discs in the real env,
that context is net-neutral-to-mild-distractor, not new signal. **Validity was the point, not
a better number: PIGINet now reads the environment's own pixels, and the tie survives it.**

---

<a id="2026-08-02-dd2d-wall-clock-first-success-fp-flatters"></a>
## 2026-08-02 — DD2D wall-clock to first success: FP flatters the learned ranker (its failures are the expensive ones)

<!--strip-->
> **id** `2026-08-02-dd2d-wall-clock-first-success-fp-flatters` · **status** active ·
> **tracks** evaluation, env-dd2d, tooling
<!--/strip-->

**What.** Added a **wall-clock-to-first-success** section to `compare_methods.py` (DD2D):
per method, seconds to the first successful refinement = abstract-plan-generation + inference +
refinement, summed over the candidates each tries until the first feasible. FP counts failed
attempts; this weighs each by its real cost (a failed refinement runs ~15 ms to ~20 s) and adds
inference — to answer whether the learned ranker's inference is worth it in practice. Refinement
reuses the stored per-candidate `refinement_wall_clock_s` (every method sums the *same* times over
its own order); inference measured on GPU (~22 ms/step, tensorization-dominated); plan-gen a
per-stratum shared constant. All cached in the compare cache. FP table byte-identical after the
`--force` rebuild (timing fields are additive).

**Result (dd2d_v4 test, n=100, 3 seeds). Breakdown of ALL, seconds:**

| method | plan-gen | inference | refinement | **total** | (FP) |
|---|---|---|---|---|---|
| astar-dist | 0.22 | 0.00 | 4.72 | **4.94** | 34.5 |
| SPECTREv3-adaptive | 0.22 | 0.51 | 5.17 | **5.90** | 5.8 |
| SPECTREv3-static | 0.22 | 0.03 | 7.72 | **7.97** | 21.1 |
| PIGINet | 0.22 | 0.27 | 7.86 | **8.35** | 17.3 |

Per-stratum total (s): astar 0.40/0.26/1.35/**17.77**; v3-adaptive 0.44/**12.00**/2.92/**8.25**;
v3-static 0.42/9.01/7.94/14.49; PIGINet 0.71/8.47/9.65/14.57.

**Takeaway — FP flatters the learned ranker.** SPECTREv3-adaptive has **6× lower FP** than astar
(5.8 vs 34.5) yet is **not faster in wall-clock** (5.90 vs 4.94 s). The reason is the whole point
of measuring time: astar's many failures are **cheap dead-ends** (~0.14 s each — 34.5 × 0.14 ≈
4.7 s), while SPECTRE's few failures are the **expensive near-feasible** candidates it correctly
ranks high, which the refiner burns time trying to refute (~0.89 s each). So a better ranking
surfaces the costlier failures, and the FP win does not carry to wall-clock. Robust sub-findings:

- **Inference is small** — v3-adaptive 0.51 s (per-step × steps), v3-static 0.03 s (one pass),
  PIGINet 0.27 s (BCE head; CLIP features cached, so this undercounts a from-scratch encode).
  Refinement dominates every method; plan-gen ~0.22 s is a shared constant.
- **The win is concentrated at s3**, where astar's *volume* of failures wins out: v3-adaptive
  8.25 s vs astar 17.77 s. At s1/s2 the learned ranker is slower (expensive failures + inference):
  s2 2.92 vs 1.35, s1 12.00 vs 0.26.
- **The ALL "adaptive slightly slower" is s1-sensitive and noisy** — s1 reads 12.00 ± 7.80, a few
  problems where the ranker picked a candidate that refined to the ~20 s budget before failing.
  Read the headline as *"no clear wall-clock win overall despite 6× fewer attempts,"* not a precise
  loss. What is robust across strata is the per-failure cost gap (astar cheap, SPECTRE expensive)
  and that inference is the small term.

**Caveats.** The refine times are a within-collection *relative* measure (8-way worker
parallelism, `time_budget=20 s` per candidate) — fair across methods since each sums the same
per-candidate times, but not an isolated single-core benchmark. Plan-gen is a regenerated
per-stratum proxy (PYTHONHASHSEED-dependent). ADR: [decisions/07
2026-08-02](../decisions/07-stickbutton2d.md#2026-08-02-wall-clock-to-first-success-added-compare-methods-reuses-stored).

---

<a id="2026-08-02-s2-ood-degradation-pool-composition-artifact-model"></a>
## 2026-08-02 — s2 OOD degradation is a pool-composition artifact, not model or generator failure

<!--strip-->
> **id** `2026-08-02-s2-ood-degradation-pool-composition-artifact-model` · **status**
> active · **tracks** env-dd2d, evaluation, method
<!--/strip-->

**What.** Root-caused the s2 column of the [2026-08-01 DD2D generalization
result](#2026-08-01-dd2d-generalization-v3-vs-astar-unseen), where v3's FP jumped 10.49 → 30.23
under the unseen-count shift (and dominates the ALL mean). Prompted by the objection that s2
(clear 2) cannot be harder than s3 (clear 3) by construction. All read-only probes on the
collected episodes + the seed-0 checkpoint; no new collection/training/scoring.

**Result.** The intrinsic difficulty ladder is intact and the generator is sound — what shifts is
the *pool's feasible composition*.

- **Not a generator bug.** s2 labels are 100% correct (every s2 problem has a real feasible
  2-subset, none shorter — pool-implied mfs matches the label 10/10 OOD, 25/25 in-dist). Execution
  difficulty is monotone as expected: astar-dist FP **s3 167 ≫ s2 28**; generation keep-rate
  **s3 20% ≪ s2 91%**. s3 is genuinely harder to execute; only the *model's* FP inverts.
- **s2 is genuinely clear-2 but has ~1.5 unique solutions.** 99% of feasible triples are redundant
  supersets of a feasible pair (in-dist 567/575; OOD 8/8); genuine-3 solutions (no feasible pair
  inside) ≈ 0. The circular target admits 18 diametric grasp axes and an axis opens only when its
  antipodal blocker pair is cleared; `crowd=5` is odd → no antipodal pair → ~1.5 feasible pairs.
- **The degradation is dominantly a pool-composition artifact.** Per-length feasibility in the
  k=200 pool:

  | s2 pool, per length | in-dist `dd2d_v4` | unseen count |
  |---|---|---|
  | 2-subset (len 5): candidates / feasible | 96.6 / 2.84 | 172.2 / 1.80 |
  | 3-subset (len 7): candidates / feasible | 92.2 / **23.0** | **18.4 / 1.14** |
  | total feasible | 25.8 | 2.9 |

  In-distribution the feasibility mass is ~23 (redundant) triples; the feasible-**pair** count is
  ~stable OOD (2.84 → 1.80). What collapses is the triples, because at 14 blockers C(14,2)=91
  pairs flood the short-first k=200 cap (→172 pair candidates) and crowd the triples out (92 → 18
  enumerated). The pool covers ~100% of possible pairs but almost none of the triples. So the
  in-dist FP=3 was *flattered* by redundant-triple padding; the shift strips it, exposing the
  problems' true ~1.5-solution difficulty. Model s2 FP corr(feasible count) = **−0.82**, median
  FP 3 → 44 (systematic, not outliers).

**Takeaway-next.** The s2 OOD number (and the ALL mean it dominates) is **confounded by pool
composition, not a clean model-generalization signal.** Read the generalization claim at **s3**
(unaffected — s3 was already feasible-scarce in training, so OOD s3 is in-regime; v3 s3 improves
9.19 → 4.87 while astar s3 stays pathological) plus the s2 caveat, not the s2 point estimate.

A generator redesign to give s2 *substantive* feasible-pair diversity was explored and **rejected
as geometrically blocked**: even collar count (the obvious lever) does not raise diversity
(generator sweep: crowd 5/6/8/10 → ~1.5 feasible pairs) and just pushes problems to mfs=3, because
blocking a circular target from all 18 diametric axes (to keep mfs≥2) fights clean single-pair
openings. Decision: **characterize, do not regen** (regen would also imply re-collecting
train/val/test + retraining, re-baselining every SPECTRE result). ADR:
[decisions/07 2026-08-02](../decisions/07-stickbutton2d.md#2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact).

---

<a id="2026-08-01-dd2d-generalization-v3-vs-astar-unseen"></a>
## 2026-08-01 — DD2D generalization: v3 vs astar on unseen count and unseen shapes

<!--strip-->
> **id** `2026-08-01-dd2d-generalization-v3-vs-astar-unseen` · **status** active ·
> **tracks** env-dd2d, evaluation, method
<!--/strip-->

**What.** First OOD generalization test of the dd2d_v4-trained SPECTRE v3 checkpoint on DD2D
itself — train-old / test-new, no retraining. Two held-out sets, 40 problems each, stratified
s0–s3 (10 each): `dd2d_v4gen_count` (14–16 items = 13–15 blockers vs the trained 9–12, old
shapes) and `dd2d_v4gen_shape` (same unseen count + a new `tee` and `cross` concave family,
≥1 of each forced per scene). Scored v3 vs astar-dist, uncensored deployed FP, 3 seeds, paired
bootstrap (`spectre_score_v3.py --test-variant … --astar-baseline`). Protocol ADR:
[decisions/07 2026-08-01](../decisions/07-stickbutton2d.md#2026-08-01-dd2d-generalization-test-unseen-count-unseen).

**Result.** In-distribution v3 reproduced 5.78 ± 0.10 exactly (instrument check). Scoring ran
clean — **no OOV and no position-index error** on the longer skeletons from denser scenes,
confirming the vocab/config are count- and shape-invariant.

| set | v3 ALL | v3 vs astar (paired) | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|---|
| in-dist `dd2d_v4` (n=100) | 5.78 ± 0.10 | −28.74 [−39.6,−18.8]* | 0.00 | 3.44 | 10.49 | 9.19 |
| unseen count (n=40) | 9.40 ± 2.62 | −39.95 [−64.0,−18.1]* | 0.00 | 2.50 | 30.23 | 4.87 |
| unseen count+shape (n=40) | 11.26 ± 3.44 | −21.89 [−42.6,−3.8]* | 0.00 | 2.40 | 31.97 | 10.67 |

astar-dist ALL: 34.52 / 49.35 / 33.15; astar s3 is pathological: 118.76 / 166.80 / 108.60.
(* CI excludes 0.)

**Takeaway-next.** v3's advantage over the naive planner order **survives OOD — it still wins
overall on both sets (CI excludes 0)** to unseen counts and unseen shapes. But three caveats,
to quote together:
- **Absolute FP degrades ~1.6–1.9×** (5.78 → 9.40 → 11.26); generalization is not free, and the
  shape set is harder than count-only.
- **The ALL-level win is carried by s3**, where astar's default order is pathological
  (108–167 FP) and v3 stays 5–11. Balanced strata, but the s3 astar catastrophe dominates the
  mean — do not read ALL as a uniform advantage.
- **At s2 v3's advantage collapses under the shift**: from clearly beating astar in-distribution
  (10.49 vs 17.08) to tying/slightly trailing OOD (30.23 vs 28.30 count; 31.97 vs 22.00 shape,
  both within the ±9 seed spread). This amplifies v3's already-characterized in-distribution s2
  deficit; s2 seed variance (±9–10) is high at n=10/stratum, so read it as "advantage lost,"
  not a precise loss. The count-set s3 improving to 4.87 (below in-dist 9.19, low variance) is
  the mirror image — more blockers give more feasible 3-subsets, which the ranker exploits.

Consistent with §0 wishlist property #4 (object-count / identity generalization): the abstract
representation transfers across counts and novel geometries well enough to keep beating the
planner order, while degrading where the harder within-length s2 discrimination already bit it.

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

