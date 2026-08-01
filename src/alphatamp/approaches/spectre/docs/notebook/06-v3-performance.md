# SPECTRE Notebook — v3 performance push

9 entries, 2026-07-27 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-07-31-unified-coverage-waste-ab-5-83"></a>
## 2026-07-31 — Unified coverage/waste beats the deployed definition on DD2D: 5.83 vs 7.44

<!--strip-->
> **id** `2026-07-31-unified-coverage-waste-ab-5-83` · **status** active ·
> **tracks** method, evaluation, env-dd2d
<!--/strip-->

- **What:** wired the unified definitions into v3 as a config-gated arm (`--unified-coverage`,
  `dataset_v3` → checkpoint cfg → `inference_v3`) and ran the pre-registered 3-seed A/B on
  dd2d_v4 against the authoritative 7.44. Same flags as the deployed arm plus the new one;
  same two overlap columns, same tensor shape — only the scalars change.

- **Result (dd2d_v4 test, n=100, 3 seeds, uncensored, demotion off):**

  | arm | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | **v3 unified cov/waste** | **5.83 ± 0.11** | 0.00 | 3.67 ± 0.99 | 10.65 ± 0.71 | 9.01 ± 0.77 |
  | v3 deployed | 7.44 ± 0.76 | 0.00 | 3.96 ± 2.50 | 13.15 ± 0.34 | 12.64 ± 1.95 |

  **−1.60 FP, 95% CI [−2.67, −0.65]** (paired bootstrap, excludes 0). Per-seed ALL
  **5.88 / 5.91 / 5.71** against deployed's 6.56 / 7.84 / 7.91 — **complete separation, every
  unified seed beats every deployed seed.** Nothing regresses at any stratum; the gain is
  concentrated at s2 (−2.50) and s3 (−3.63), which is where multiple distinct blockers make
  culprit identity load-bearing.

- **Seed variance collapses 7×: ±0.11 vs ±0.76.** s1's spread falls from ±2.50 to ±0.99. That
  is consistent with the mechanism predicted from the offline probe rather than a surprise:
  the deployed feature is *completely flat* in 8% of contexts (all culprits were `__wall__` /
  `target`, neither ever in `S(c)`), so those episodes were ranked on the other columns alone
  and contributed most of the seed-to-seed slack.

- **Why it wins, restated:** the Actionable filter removes `__wall__` — the single most
  frequent reported culprit on dd2d_v4 — from the denominator, where the deployed formula
  keeps it permanently uncovered. Mean \|K\| 1.99 → 1.67, coverage spread across the pool
  +48%, dead contexts 8% → 0%. The offline probe predicted "at least as good, plausibly
  better" from exactly these three numbers.

- **Verification, in the order it was done:** (1) the frozen baseline was re-scored under the
  new code first and reproduced 7.44 ± 0.76 digit-for-digit; (2) with the flag **off**,
  `dead`/`jaccard` are bit-identical and overlap shape is unchanged, so D-8 exact-absence
  holds; (3) the flag rides the **checkpoint cfg**, not the CLI, so a model trained on unified
  features cannot be scored against deployed ones — the round-trip test that asserts the
  deploy-kwargs set caught this and was updated deliberately; (4) all three runs terminated
  cleanly (30/30 epochs logged, `.owner` released), checked because a killed run leaves a
  complete-looking `best.pt`.

- **Cost, and the memoization that followed.** `covered()`/`_justified()` recomputed
  `matched_steps` per (culprit × record), `touch` per (culprit × record × superfluous step),
  and `blame`/`collateral` on every one of those, for all 200 candidates on every example.
  Hoisting them into a per-(candidate, record) `_Memo` shared by both columns
  (`coverage_and_waste`) took steady-state training from **~112 to ~78 s/epoch (−30%)**.

  Two verifications, because a "pure speedup" that changes a number is the worst kind:
  1. **Re-scoring the unchanged checkpoints under the memoized code is bit-identical** —
     5.83 ± 0.11 / s1 3.67 / s2 10.65 / s3 9.01 / delta −1.60 CI [−2.67, −0.65], every digit.
     This holds the weights fixed, so it isolates the feature computation exactly.
  2. **A full 3-seed retrain reproduces the result**: 5.78 ± 0.10, delta −1.66 CI
     [−2.71, −0.71] against 5.83 ± 0.10 / −1.60 before. The 0.05 FP gap is inside the seed
     sd; epoch-1 train losses match to 4 dp while `val_fp` drifts slightly, which is
     low-bit CUDA nondeterminism in training, not the features.
  `test_memoized_matches_naive_recomputation` pins the equivalence candidate-by-candidate
  on real dd2d_v4 episodes.

  **Correction:** the "283 s/epoch" quoted when this arm first launched was the *epoch-1*
  estimate, which includes warmup. Steady state before memoization was ~112 s/epoch, so the
  honest speedup is 30%, not the order of magnitude that figure implied. The remainder is
  irreducible per-candidate work — `predicted_states`, `superfluous_steps`, and building the
  memo — each O(L) across 200 candidates. Going below it needs restructuring (e.g. hoisting
  `records_from_failure_records` out of `build_v3_example`), not more caching.

- **Takeaway / next:** the unified definition is better on DD2D, not merely compatible, so
  the earlier worry about inheriting checkpoints is moot — a retrain is warranted on its own
  merits. Against the published v2.2 yardstick (17.27, which keeps its own demotion) the
  margin moves from −9.83 to ≈ **−11.4**; note this run's v2.2 row reads 17.36 because
  `spectre_score_v3` scores demotion-off by default, and the paired CI for that pair was not
  computed here.

---

<a id="2026-07-31-unified-coverage-waste-probes"></a>
## 2026-07-31 — Unified coverage/waste: the gate clears on SB2D, and two DD2D compat claims are falsified

<!--strip-->
> **id** `2026-07-31-unified-coverage-waste-probes` · **status** active ·
> **tracks** method, evaluation, env-stickbutton2d
<!--/strip-->

- **What:** implemented `docs/unified_culprits_coverage_waste.md` as `unified_evidence.py`
  (env-agnostic, **not** wired into `dataset_v3` — validation only) plus a SB2D instrumented
  refiner, and ran the three pre-registered probes via `experiments/spectre/unified_probe.py`.
  Both worked examples (§7 DD2D, §8 SB2D) are pinned as tests against the **real** kinder and
  DD2D operator schemas and reproduce number-for-number.

- **Result — P1, the gate: PASSES.** Failure-record classes over 30 problems (k=60):

  | variant | records | means-only | collateral-add | collateral-del | mean \|K\| |
  |---|---|---|---|---|---|
  | b3 | 808 | 40% | 59% | 1% | 2.3 |
  | b5 | 881 | 26% | 74% | 0% | 2.9 |

  Collateral dominates, so `K` genuinely populates. Also confirms §10's registered
  prediction that incidental presses grow with button count (59% → 74%).

- **Result — P3: coverage works, and carries all of it.** Mean failed attempts before first
  success, paired per problem:

  | variant | n | static | coverage+waste | coverage only | waste only | oracle |
  |---|---|---|---|---|---|---|
  | b3 | 15 | 7.20 | **5.87** | 5.87 | 6.93 | 0.00 |
  | b5 | 5 | 24.20 | **17.00** | 17.00 | 24.20 | 0.00 |

  b3 −1.33 attempts (better on 7 problems, worse on 2, tied 6); b5 −7.20 (better 3, worse 1,
  tied 1). `coverage_only` equals `coverage+waste` **exactly** and `waste_only` ≈ static, so
  **waste contributes nothing on SB2D** — §10's registered cross-environment dominance flip
  (waste→DD2D, coverage→SB2D), confirmed. Caveat: b5 n=5, since only 5 of 15 problems had a
  feasible candidate within k=60.

- **Result — P2 falsifies two DD2D backward-compatibility claims.** Over 25 dd2d_v4 test
  episodes / 93 singleton contexts / 60000 (candidate, culprit) pairs:
  - **"Terminal contexts dominate" is false.** All **114** culprit-bearing records are
    `pick` (non-terminal); **zero** are `retrieve`. §7's worked example and the whole compat
    argument assume a collision on the terminal extraction.
  - **"Bit-identical by construction" is false.** Coverage vectors differ on **100%** of
    contexts and the full induced ranking differs on **100%**. Disagreement is 7.17% of
    coverage pairs, 2.17% of waste candidates.
  - **But the decision survives: top-1 pick is identical on 93/93.** Cause breakdown:
    **9.02%** of pairs are goal objects (`target`) entering `K` — barred from deployed
    `S(c) = args \ goal_objects`, uniformly covered under unified, hence ranking-inert by
    the doc's own lemma; **1.58%** are the genuine causal correction §4 anticipated.

- **Result — the DD2D change is directional, not destructive, and the new feature is
  *sharper*.** Over the same 93 contexts:

  | measure | deployed | unified |
  |---|---|---|
  | mean \|K\| | 1.99 | **1.67** |
  | mean spread of coverage across the pool | 0.2315 | **0.3435** (+48%) |
  | contexts where coverage is completely flat (no ranking info) | 7/93 (8%) | **0/93 (0%)** |
  | ordered candidate pairs, sign agreement (non-tied) | — | **97.68%** (8322 vs 198) |
  | top-1 pick | — | identical 93/93 |

  The reason is `__wall__`: it is the **most frequent reported culprit** on dd2d_v4 (1321
  mentions vs 939 for `target`), it is not an abstract object, and the deployed formula keeps
  it in the denominator where it is permanently uncovered. The Actionable filter removes it.
  §1's claim that actionability is "honestly idle on both current environments" was therefore
  **wrong for DD2D** and has been corrected in the design doc.

- **Takeaway / next:** the construction earns its place on SB2D, and on DD2D it is
  *directionally* the same feature (97.7% of ordered pairs, 100% of top-1 picks) with ~48%
  more resolution and no dead contexts. But it is **not** value-compatible, so existing DD2D
  checkpoints cannot be inherited — adopting it there needs a 3-seed retrain A/B against the
  published 7.44 FP. Do *not* patch the goal-object gap by excluding goal objects from `K`:
  on SB2D the culprits **are** goal objects (buttons), which is exactly the blindness the
  redesign removes.

---

<a id="2026-07-30-demotion-cut-authoritative-v3-7-44"></a>
## 2026-07-30 — Demotion cut: the authoritative v3 is 7.44 FP, purely learned

<!--strip-->
> **id** `2026-07-30-demotion-cut-authoritative-v3-7-44` · **status** active ·
> **tracks** evaluation, method
<!--/strip-->

- **What:** proof-tier demotion was **cut from the deployed method** (user call, for story
  coherence: v3's claim is one canonical record consumed by *learned* components, and an
  external hand-declared deduction acting on the ranking was the one thing that did not fit).
  `apply_demotion=False` is now the default everywhere. The authoritative v3 model is the
  demotion-free one. ADR: [`decisions.md`](../decisions/README.md) 2026-07-30.

- **The headline, re-measured** (dd2d_v4 test, n=100, 3 seeds, uncensored):

  | method | seeds | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|---|
  | **SPECTREv3-adaptive** (no demotion) | 3 | **7.44 ± 0.76** | 0.00 | 3.96 ± 2.50 | 13.15 ± 0.34 | 12.64 ± 1.95 |
  | SPECTREv2-adaptive (keeps demotion) | 3 | 17.27 ± 3.02 | 0.00 | 13.67 ± 14.20 | 23.45 ± 2.76 | 31.95 ± 5.62 |

  **−9.83 FP, CI [−12.57, −7.36]** (was −10.06 with demotion). Per-seed ALL: 6.56 / 7.84 /
  7.91. Every other row in the comparison is unchanged — PIGINet 17.27 ± 0.19, astar 34.52.

- **The comparison is now asymmetric, and it handicaps v3.** v2.2 keeps its own observed
  demotion, because 17.27 is the number published for it throughout the project and
  re-scoring the baseline to match a v3 design choice would be moving the goalposts. So the
  margin is measured against a *stronger* baseline than v3 gives itself.

- **What it cost:** 0.23 FP, measured and significant — the [previous
  entry](#2026-07-30-proof-demotion-priced-0-23-fp-deployed) has the full ablation, and its
  measurements all stand. Only its concluding recommendation ("demotion stays") is reversed.

- **Verification worth recording, because two things could have gone silently wrong:**
  1. **The cache had to be rebuilt with `--force`.** `spectre3_adaptive` held the
     demotion-ON rollout; without `--force`, `_dir_complete` would have kept it and §1
     would have gone on reporting 7.20 while the code said otherwise.
  2. **The D-8 equivalence oracle had to be pinned to `apply_demotion=True`.** It compares
     v3-in-compat-mode against the v2.2 rollout, and v2.2 always demotes — so the new
     default made it compare two different *policies*. It still passed, which is the
     problem: it was passing only where the offset happened not to change an argmax. Same
     for the stale-cache test, whose cache was written with demotion on.

- **Takeaway / next:** the deployed system is now one mechanism — model scores, nothing
  else touching the order. The follow-up this creates is the 1.3% of records (all
  `retrieve`) that the tier split still holds out of the token path and that no longer have
  a proof consumer; routing them in needs a retrain.

---

<a id="2026-07-30-proof-demotion-priced-0-23-fp-deployed"></a>
## 2026-07-30 — Proof-demotion priced: 0.23 FP on the deployed model, 1.09 without the learned components

<!--strip-->
> **id** `2026-07-30-proof-demotion-priced-0-23-fp-deployed` · **status**
> partially-superseded · **tracks** evaluation, method · **superseded by**
> 2026-07-30-demotion-cut-authoritative-v3-7-44
>
> ⚠️ **PARTIALLY SUPERSEDED** — every measurement here stands; only the closing
> recommendation ("demotion stays") is reversed. Demotion was cut from the method.
<!--/strip-->

- **What:** withheld the outside-the-net proof-demotion offset
  (`deployed_rollout_v3_traced(apply_demotion=False)`) on two arms and compared against
  their demotion-ON twins. Pure eval — same weights, same seeds, same episodes, proof state
  still advanced; only the finite `1e6` ranking offset is withheld, so the pairs are exactly
  paired. dd2d_v4 test, n=100, uncensored, `strict`.

- **Result:**

  | arm | seeds | ALL | s0 | s1 | s2 | s3 | Δ (off − on) |
  |---|---|---|---|---|---|---|---|
  | deployed v3 · ON | 3 | 7.20 | 0.00 | 3.96 | 12.61 | 12.24 | |
  | deployed v3 · OFF | 3 | 7.44 | 0.00 | 3.96 | 13.15 | 12.64 | **+0.23 [+0.08, +0.43]** * |
  | floor (jaccard only) · ON | 1 | 15.47 | 0.96 | 4.20 | 29.64 | 27.08 | |
  | floor (jaccard only) · OFF | 1 | 16.56 | 0.96 | 4.60 | 30.92 | 29.76 | **+1.09 [+0.65, +1.68]** * |

  Paired bootstrap over problems on the seed-mean; both CIs exclude 0. The floor arm is
  `abl_nocov_norec` — jaccard overlap only, no `coverage`/`waste`, no record tokens — and
  seed 0 is the only seed it has.

- **The sound rule is worth 4.7× more once the learned components are removed** (1.09 vs
  0.23). Put the other way: the learned adaptive features absorb ~79% of what proof-demotion
  would otherwise be contributing. This is the same shape G7 measured on the pre-delta model
  (0.13 with the learned overlap column on, 1.82 with it off) and it survives on the deployed
  state-delta model.

- **A sharper statement than the FP delta: demotion barely *fires* on the deployed model.**
  It changes the realized attempt order on **18/300** deployed (problem, seed) pairs — 6% —
  against **55/100** on the floor arm. The learned features have already ordered the pool so
  that the proof usually has nothing left to correct; where it does still fire, it is right.

- **All of the deployed value is at s2/s3.** s0 and s1 are *bit-identical* with and without
  the offset (s1 3.96 → 3.96 across all three seeds, and 7.68 → 7.68 at 6 seeds). Demotion
  needs a subset-containment proof, which needs multi-object stagings; s1 candidates stage
  one object, so nothing is ever provably dead there.

- **6-seed corroboration** (deployed only, from the CLI, not cached — caching v3 at 6 seeds
  would move §1's headline and break the cross-method 3-seed protocol): 8.23 ± 1.36 ON vs
  8.54 ± 1.43 OFF, Δ +0.31. Same conclusion at twice the seeds.

- **Takeaway / next:** demotion stays, and the reason is unchanged and is *not* this number —
  it is soundness (0 demoted-but-feasible under `strict`; the learned signal is a correlate,
  the offset is a proof, C5/P-E). What this adds is the price: on the current model it buys
  0.23 FP, so a port to an environment where the `local` axiom cannot be declared loses
  little on DD2D-like distributions — "learning is the floor" is cheap here. The 6%-firing
  figure is the number to re-check on env-2, since a domain where proofs fire more often
  would shift the balance back.

- **Superseded:** an uncommitted working-tree note in `compare_dd2d_methods.py` put the
  deployed cost at 0.19 (7.50 → 7.69). That was seed 0 of the **pre-delta** checkpoint and
  never reached a durable file; the floor arm's 1.09 (15.47 → 16.56) reproduces exactly.

---

<a id="2026-07-29-stickbutton2d-b5-reaches-75"></a>
## 2026-07-29 — StickButton2D b5 reaches 75% on a heuristic change alone; b10 stays at 0%

<!--strip-->
> **id** `2026-07-29-stickbutton2d-b5-reaches-75` · **status** active · **tracks**
> env-stickbutton2d
<!--/strip-->

Autonomous session (decisions in
[`autonomous_stickbutton_session.md`](../autonomous_stickbutton_session.md)). Constraint: **no
changes to kinder's refiner or trajectory sampler** — abstract-planning level only.

- **What:** two changes to the A* skeleton generator. (1) The heuristic gains a
  *distance-to-the-nearest-unpressed-button* term. (2) `RobotPressButton*` is no longer
  **grounded** on buttons past the robot's reach.

- **Why the distance term:** the robot presses whatever it drives over, so a plan fails when
  it crosses a button it has not reached yet. If you always go to the *nearest* remaining
  button, nothing unpressed can be on the way — anything on that segment would have been
  nearer. "Walk to the nearest one" and "never press out of order" are the same preference.
  It is load-bearing because every press ordering has the *same length*, so without it A*
  rates all 120 (b5) / 3.6M (b10) orderings as tied.

- **Result — stock sampler, 200 attempts, 20 s each, 20 problems/variant:**

  | variant | ≥1 success | mean #successes | before |
  |---|---|---|---|
  | b3 | **100%** | 15.4 | 100% (first success 14–16 → **2–10**) |
  | b5 | **75%** | 4.0 | **0/8** |
  | b10 | **0%** | 0.0 | 0/4 |

  **b5 clears the 50% bar; b10 does not.**

- **Result — the count term must be weighted just above 1.** Each press adds 1 to `g` and
  removes 1 from the count, so at weight exactly 1 the A* score is depth-invariant and the
  search plateaus: b10 returns an **empty pool** after 30 s. At 1.05 it returns 200 plans in
  1.4 s. Larger weights cost opening diversity (distinct first press / first three, of 200,
  on b5): 1.05 → 5/32 (same as 1.0), 1.5 → 2/7, 2.0 → 1/2. Deployed **1.05**.

- **Why b10 fails, and why no heuristic fixes it.** Failures land at step 0–1, and at b10
  **all 200 candidates share the same first three presses** at every workable weight — so the
  whole budget goes to variations of one bad opening. A single A* run yields goals in `f`
  order, so alternative openings surface only after their subtree is exhausted. Fixing this
  needs prefix-diverse plan generation, a generator change, not a better `h`.

- **Negative — quantising the distance term does nothing.** The idea was that rounding would
  make openings tie and let the generator's RNG diversify them. b10 stays at 1 distinct
  opening for rounding of 0.1/0.25/0.5/1.0 world units. Do not retry.

- **Negative — nearest-first is a good prior, not a guarantee.** Refining the single explicit
  nearest-first plan succeeds for b3 55% / b5 25% / b10 5%: the robot has a body (radius 0.1)
  and the stick is 1.25 long, so even the nearest hop can sweep a button beside the corridor.
  This is why the pool needs 200 attempts rather than 1.

- **Takeaway / next:** adopt b1/b2/b3/**b5**; drop b10. b5 at 4 positives per 200 is a good
  ranking problem — scarce positives are the regime SPECTRE is for. Cost: ~900–2700 s per b5
  problem, so 400/100/100 ≈ 5 h at 30-way parallelism.

---

<a id="2026-07-28-stickbutton2d-feasibility-b1-b3"></a>
## 2026-07-28 — StickButton2D feasibility: b1–b3 are collectable, b5/b10 are not (yet)

<!--strip-->
> **id** `2026-07-28-stickbutton2d-feasibility-b1-b3` · **status**
> partially-superseded · **tracks** env-stickbutton2d, data · **superseded by**
> 2026-07-29-stickbutton2d-b5-reaches-75
>
> ⚠️ **PARTIALLY SUPERSEDED** — b5 reaches 75% after the heuristic change.
<!--/strip-->

- **What:** mapped the kinder / kinder-bilevel-planning substrate for StickButton2D (SPECTRE's
  candidate second environment; map in
  [`kinder_stickbutton2d_map.md`](../kinder_stickbutton2d_map.md)) and measured whether it can
  yield a dataset that is not all-negative. `dataset.py` drops episodes with
  `num_success == 0`, so this gates everything downstream. New harness
  `experiments/spectre/stickbutton_feasibility.py` (parallel, two modes: cheap per-button
  `probe` and ground-truth `full`).

- **Result — stock pipeline is unusable above b3.** hff generator + `BacktrackingRefiner`,
  `samples=5`, `horizon=200`, `timeout=20`:

  | variant | pool | skeletons refined |
  |---|---|---|
  | b1 | 12×3 seeds | 15/36 (42%) |
  | b2 | 12×3 seeds | 9/36 (25%) |
  | b3 | 12×3 seeds | 0/36 |
  | b5 | **200** | **0**, 379 s |
  | b10 | 40 | **0** |

  Refinement fails in 0.2–0.8 s per skeleton, so this is not a timeout problem.

- **Result — two independent causes, both isolated.** (1) The generator is geometry-blind:
  `heuristic_name` is silently ignored (`heuristic_search_plan_generator.py:198` hardcodes
  hff), and kinder's `RobotPressButton*` applies to any button, including ones past the
  robot's reach limit of **1.405** (derived from the env config). Bare-robot plans on
  unreachable table buttons are symbolically shortest and crowd out the pool. (2) The goal
  demands *all* N buttons pressed, so one unpressable button voids every skeleton and episode
  feasibility falls like `Π q_i`.

- **Result — the geometry-aware heuristic helps where reach binds, and is not sufficient.**
  `h = |unpressed| + 1[table button remains ∧ hand empty] + 1[robot-only button remains ∧
  holding]`. b3/seed0 first success **29 → 16**; b3/seed1 (no table buttons) **14 → 14**, i.e.
  it correctly degenerates to hff. b5/b10 unchanged at zero.

- **Result — ground truth, `full` mode** (geometry-aware generator, stock acceptance,
  `k_max=60`, 8 problems/variant, every candidate refined):

  | variant | has ≥1 success | median first-success idx | mean #success / 60 | s/problem |
  |---|---|---|---|---|
  | b1 | **100%** | 0 | 24.2 | 465 |
  | b2 | **100%** | 3 | 9.1 | 313 |
  | b3 | **100%** | 14 | 2.8 | 280 |
  | b5 | **0%** | — | 0 | 133 |

  b1–b3 clear the 80–90% bar outright. Positives thin fast with button count, which is what
  makes b3 the interesting ranking problem and b5 the cliff.

- **Correction — the per-button probe is not a bound in either direction.** It was first
  recorded as an upper bound; it is not. Probe (exact) said b2 55% / b3 35% where truth is
  100% (**under**), and probe (superset) said b5 75% where truth is 0% (**over**). It
  under-estimates because it tries one route per button from `x0`, while a real skeleton may
  reach a button from a different predecessor or press it incidentally; it over-estimates
  because real skeletons must chain presses. Keep it for *failure attribution*, size
  collections from `full` mode.

- **Negative — more sampling does not help.** `PickStickFromNothing` failure is the blocker
  that kills all stick-dependent buttons at once, and it is the one controller with real
  sampled parameters. Over 10 b5 scenes, `num_sampling_attempts_per_step` 5 → 25 → 100 gave
  **7/10 every time with the identical three scenes failing** (0.4 → 1.5 → 5.3 s). Some scenes
  place the stick where it cannot be grasped; filtering, not budget, is the lever.

- **Negative — relaxing acceptance helps per-button but not end-to-end.** Swapping the
  sampler's exact abstract-state equality for `planned ⊆ achieved` (sound for goal
  achievement) lifts the *probe* to b2 90 / b3 85 / b5 75 / b10 55%, because `extra_atoms`
  (incidental presses) is the largest blocker. But on real skeletons it changes nothing:
  **b3 8/8 either way, b5 0/8 either way.** Divergence from the symbolic plan breaks later
  steps. Kept as an off-by-default option (`sampler.py`); **stock semantics retained.**

- **Verified end-to-end:** `spectre_collect.py env=stickbutton2d_b3` → `spectre_build_vocab.py`
  → `spectre_check_pipeline.py` with **0 episodes filtered**; per-episode first-success indices
  (16, 14) match the harness exactly, confirming the collector uses the same generator. Vocab:
  6 operators / 6 predicates / 3 types, val+test OOV-clean.

- **Correction — why b5/b10 fail is NOT obstacle avoidance.** First written up that way;
  wrong. Buttons are `ZOrder.NONE` and never block motion, and the only barrier is the table,
  which a correct plan never drives at. Step attribution: **102/120 skeletons fail at step 0**
  (so not length-compounding either), from two independent causes — (i) out-of-reach plans
  (*missing* atoms), because the heuristic's constant `+1` never makes an impossible action
  look bad; fix is to not **ground** `RobotPressButton*` on stick-only buttons; and (ii)
  **incidental presses** (*extra* atoms) — with grounding pruned, failures are `missing=()`,
  `extra=(Pressed buttonX, ...)`.

- **The mechanism, precisely: "ahead of schedule", not "press each button once."** `Pressed`
  is never deleted, so re-pressing a button is not even observable. Over b5 seeds 0–3, of the
  extra `Pressed` atoms, **41/41 were buttons the remaining plan suffix still intended to
  press and 0 were strays**. The exact-equality check
  (`parameterized_controller_sampler.py:89`) demands each intermediate state match the plan,
  so an early press of a *future* target rejects a trajectory whose final state would be
  correct.

- **Each fix alone measures as useless because the other masks it** — pruning alone: b5 still
  0; superset alone: b5 still 0. Together: **b5 0/4 → 2/4 seeds, b10 0/4 → 0/4.** Real
  movement, still far short of the bar; residual is genuine controller failure (`PickStick`
  on ungraspable-stick scenes). Testing the two separately is how this was first mis-written
  as "needs controller work".

- **Takeaway / next:** b1/b2/b3 are collectable today and clear the target; b5/b10 stay out.
  Cost anchor
  for the real collection: `K_max=200` ≈ 3× the k=60 cost, so 600 problems ≈ 6–8 h at ~30-way
  parallelism. Per-variant mix for the pooled dataset is a user decision pending these
  numbers.

---

<a id="2026-07-28-dd2d-comparison-3-seeds"></a>
## 2026-07-28 — the DD2D comparison at **3 seeds per learned method**, all native to dd2d_v4

<!--strip-->
> **id** `2026-07-28-dd2d-comparison-3-seeds` · **status** active · **tracks**
> evaluation, baselines
<!--/strip-->

- **What:** promoted the state-delta config to deployed, retrained **PIGINet on dd2d_v4 at
  3 seeds** (it had no v4 artifacts and no `--seed` flag at all), re-cached SPECTRE v2.2 and
  v3 at seeds 0–2, and converted `compare_dd2d_methods.py` from a 1-seed table to a 3-seed
  one whose `±` is the spread **across seeds** rather than across problems. VLMPlan stays
  grafted from dd2d_v3 at 1 seed.

- **Result — uncensored deployed FP, dd2d_v4 test, n=100:**

  | method | seeds | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|---|
  | astar-dist | - | 34.52 | 0.00 | 2.24 | 17.08 | 118.76 |
  | PIGINet | 3 | 17.27 ± 0.19 | 0.05 ± 0.02 | 5.04 ± 1.49 | 18.77 ± 1.58 | 45.20 ± 0.84 |
  | SPECTREv2-adaptive | 3 | 17.27 ± 3.02 | 0.00 | 13.67 ± 14.20 | 23.45 ± 2.76 | 31.95 ± 5.62 |
  | SPECTREv2-static | 3 | 20.86 ± 1.96 | 0.00 | 15.08 ± 14.68 | 28.03 ± 3.53 | 40.35 ± 3.43 |
  | **SPECTREv3-adaptive** | 3 | **7.20 ± 0.62** | 0.00 | **3.96 ± 2.50** | **12.61 ± 0.44** | **12.24 ± 1.82** |
  | SPECTREv3-static | 3 | 20.66 ± 1.53 | 0.00 | 15.48 ± 11.73 | 28.69 ± 6.47 | 38.48 ± 3.25 |
  | VLMPlan-8B | 1 | 29.86 | 4.24 | 2.88 | 16.04 | 96.28 |
  | VLMPlan-32B | 1 | 23.55 | 6.76 | 5.04 | 13.16 | 69.24 |

  v3 vs v2.2: **−10.06 FP, CI [−12.83, −7.59]**. Per-seed ALL: v3 6.49 / 7.51 / 7.61;
  v2.2 14.66 / 16.57 / 20.57; PIGINet 17.05 / 17.33 / 17.42.

- **PIGINet and v2.2 tie on the mean at 17.27 and are nothing alike.** The spreads are
  **±0.19 vs ±3.02** — a 16× difference — and the strata run opposite ways: PIGINet is far
  better at s1 (5.04 vs 13.67) and s2 (18.77 vs 23.45), far worse at s3 (45.20 vs 31.95).
  The identical means are a coincidence and should never be reported without the spread
  beside them. It is also a clean illustration of why the `±` had to become across-seed:
  the old across-problem number would have made these two rows look equally (un)reliable.

- **PIGINet is the most seed-stable learned method here** (±0.19 on ALL, ±0.84 at s3),
  which is worth knowing before attributing any small PIGINet difference to noise. Its v4
  result (17.27) is close to its v3 one (18.67), so the re-collection did not move it.

- **The state delta is now deployed** and the headline is 3-seed: 7.20 ± 0.62 vs
  17.27 ± 3.02. **Two disclosures ride with that** (`as_built_v3.md` §7.1): v3 has **6**
  trained seeds and over all six reads **8.23 ± 1.36**; and the yardstick moved from v2.2's
  *best* seed (14.66) to its 3-seed mean (17.27), which is ~2.6 FP of the −10.06.

- **A near-miss worth recording: I read a cache mid-write and got a wrong number.** Querying
  the table while `precompute_dd2d_cache` was still filling `spectre2_adaptive/seed_2` gave
  v2.2 = 12.68 ± 5.17, with seed 2 reading **15.21** instead of 20.57. A half-filled
  `seed_N` directory loads without error — `_dir_complete` is only consulted by the *writer*,
  not the reader — so the mean is silently taken over however many problems happen to exist.
  Caught only because seeds 0 and 1 matched the published values exactly and seed 2 did not.
  **Never read a compare cache while a build is running.**

- **Takeaway / next:** the notebook is now a 3-seed comparison end-to-end and its §1 comes
  from `dd2d_compare.build_table`, the same function behind `spectre_v3_table.py`, so the
  notebook and the CLI reporter cannot drift. Ablations (§4) remain a frozen seed-0
  pre-delta study; re-running them at 3 seeds is the obvious next increment if any of those
  contrasts becomes load-bearing.

---

<a id="2026-07-28-state-delta-ties-6-seeds"></a>
## 2026-07-28 — §6.1's `s_j` built as a state delta on the record token: **a tie** (6 seeds)

<!--strip-->
> **id** `2026-07-28-state-delta-ties-6-seeds` · **status** active · **tracks**
> method, evaluation
<!--/strip-->

- **What:** filled the one unimplemented field of the proposal's `FailureRecord` schema.
  Each record token now also carries the abstract state at its failing step, as the **delta
  from `s_0`** (which atoms the prefix added, which it deleted) — pure STRIPS progression
  over the candidate's own plan, encoded domain-agnostically from the vocab's predicate
  table. `--state-delta`, off by default. Arm = `v3final`'s flags plus that one.

- **Cheap signal checks first (A13's rule), on the real collection:**
  - the delta's *object* set is **exactly `all_objects − unmoved`** on **946,063/946,063**
    records across train+val+test. `unmoved` is already on every record and is read only by
    `proof_demotion_v3`; it has never reached a tensor. So on DD2D the delta's only content
    beyond it is the **predicate label** per object (`on-buffer` vs `holding`).
  - **`corr(j/L, |staged|) = 0.940`** — the delta's *size* is nearly determined by a scalar
    the token already carries. So no count feature was derived from it; that would be `dead`
    again. What it adds is object **identity**.
  - identity is genuinely new: given `(problem, schema, args, j/L)` — everything the model
    already sees about a record — there are **2.65 distinct staged sets** on average, **>1 in
    53.6%** of groups (3.01 / 61.7% on the non-empty half).
  - but under `--aggregate-records`, which the deployed config uses, **47.8% of tokens carry
    an empty delta**, rising to **54.9% at s2/s3**. (The 4.8% figure is *un*-aggregated —
    aggregation collapses the deep re-sampled records and leaves the shallow `j=0` ones
    proportionally dominant.) **The strata that need moving have the least coverage.**

- **Step 0, before training anything:** re-scored the frozen 6-seed baseline under the new
  code. It reproduced **7.90 ± 0.61 / 0.00 / 5.60 / 13.03 / 12.96** digit-for-digit, so
  flag-off really is exact absence and the comparison is against the same instrument.

- **Result — a tie, which was the pre-registered acceptable outcome.** 6 seeds each:

  | | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | v3 + state delta | 8.23 ± 1.36 | 0.00 ± 0.00 | 7.68 ± 5.35 | **12.69 ± 0.80** | **12.57 ± 1.23** |
  | v3 deployed | **7.90 ± 0.61** | 0.00 ± 0.00 | **5.60 ± 3.06** | 13.03 ± 1.52 | 12.96 ± 2.46 |

  Paired bootstrap over problems: **+0.34 FP, 95% CI [−0.30, +1.07]** — includes 0. Per seed
  the sign splits **3–3** (delta 6.49 / 7.51 / 7.61 / 8.61 / 10.43 / 8.76 against 7.50 / 7.63
  / 7.19 / 8.05 / 8.08 / 8.94).

- **The feature was used, not ignored.** `delta_proj` starts at exactly zero and trains away
  from it in every seed, and the deployed checkpoint reports `state_delta=True` with all
  other deploy kwargs identical to the baseline's. So this is a tie *with* the signal
  consumed, not a tie because the branch stayed inert — a distinction A8/A11 showed matters.

- **One observation, offered as an observation only.** At s2 and s3 the delta arm has a
  slightly lower mean and roughly **half the between-seed spread** (0.80 vs 1.52; 1.23 vs
  2.46), while s1 gets noisier (5.35 vs 3.06) and drives the overall sd up. At n=6 the sd of
  an sd is not worth much, so this is **not** claimed as a variance reduction — recorded
  because it is the only structure in the numbers and would be the thing to check first if
  anyone revisits this.

- **Takeaway / next:** the bar was beat-or-tie and it ties, so per the standing instruction
  the arm is **not** pursued further and no attribution arm was run (that was gated on a
  win). Kept in the tree, default off. Two DD2D-specific reasons the ceiling was low are
  measured above and are *not* properties of the mechanism: on a domain where a prefix's
  effects are richer than "one object leaves the drawer", or where failures are deeper so
  fewer tokens sit at `j=0`, the delta carries strictly more. It needs **no new
  instrumentation** in a new environment, so it costs nothing to leave available.

---

<a id="2026-07-27-comparison-retargeted-two-stale-bugs"></a>
## 2026-07-27 — Comparison notebook retargeted to v3; **two stale-checkpoint / stale-cache bugs found while doing it**

<!--strip-->
> **id** `2026-07-27-comparison-retargeted-two-stale-bugs` · **status** active ·
> **tracks** tooling, evaluation
<!--/strip-->

- **What:** rebuilt `experiments/spectre/compare_dd2d_methods.py` around SPECTRE v3 on
  `dd2d_v4`, added a `coverage`/`waste` x record-token ablation, and cached v3 so nothing
  runs inference at notebook load. Retired analyses moved to
  `compare_dd2d_methods_archive.py` (still reads `dd2d_v3`). 1 seed per method throughout
  (seed 0) -- the multi-seed pass is deferred.

- **v3 caches faithfully [verified].** The new `cache_spectre3` path and
  `spectre_score_v3.py` are independent code paths over the same episodes and agree
  exactly on seed 0: **7.50 / s0 0.00 / s1 1.16 / s2 15.80 / s3 13.04**. v2.2 still
  reproduces its published **14.66**. (The 6-seed headline remains 7.90 +- 0.61.)

- **v3-static is 20.96 -- *worse* than v2.2-static (20.08)** while v3-adaptive is 7.50.
  `coverage`/`waste` are identically 0 at `|F|=0`, so v3 has no static advantage by
  construction; the entire margin is adaptive. This is the cleanest direct evidence for
  that claim so far, and it comes free with the static row.

- **BUG 1 -- `p8_cov_final_s{0,1,2}` are epoch-5 stubs, not "the clean 3-seed re-run".**
  `autorun_decisions.md` A15 names them as the reportable number for the jaccard+coverage
  config. All three training logs stop at **epoch 5 of 30**. Scored: **26.97 ALL, s0
  36.64** -- against ~8 for the finished config, and s0 is 0.00 for every other arm
  (target already graspable). Both the cache path and `spectre_score_v3.py` return 26.97,
  so this is the checkpoint, not the reader. Retrained at identical flags as
  `abl_cov_rec`. **A15's recommendation to prefer p8 over `p5_jac_cov` is withdrawn.**

- **BUG 2 -- the v1 comparison rows were double-canonicalized.** `cache_spectre` fed
  `eda.load_split_episodes` (already canonicalized) into `spectre_evaluate_traced`, which
  calls `init_inference_state`, which canonicalizes again -- the same defect that retracted
  the dd2d_v3 13.68. Measured on dd2d_v3: **21.41 raw vs 22.93 double**, differing on
  **39/100** problems. `cache_spectre` and `cache_lenctx` now load through `_RawSplit`;
  v1 rows rebuilt (`SPECTRE-adaptive` 22.93 -> **21.41**, `-static` 25.25 -> **22.98**).

- **Closed: the dd2d_v3 13.68 is corrected to 14.50.** [`decisions.md` 2026-07-26](../decisions/README.md) left this
  open — the cached figure was computed under double canonicalization and "the rebuild will
  settle whether the corrected figure is 14.50". Rebuilt with `--force`: **14.50** (s0 0.00,
  s1 4.64, s2 25.68, s3 27.68), matching the live-run prediction exactly. **13.68 is
  superseded; quote 14.50.** `SPECTREv2-static` on that collection moves 19.12 -> 20.37.

- **Fixing one cache and not its partner briefly inverted a published conclusion.** After
  the v1 rebuild, the archive notebook's T1 read **Δ +1.53, "identity matters"** — a fixed
  adaptive row against a still-stale `lenctx` row. Rebuilding `lenctx` too restores
  **Δ −0.003, CI [−0.020, +0.010], "IDENTITY UNUSED — H2 confirmed"**, a *tighter* interval
  than the original. So T1 survives, but the episode is the lesson: **a paired comparison
  must have both arms rebuilt in the same pass**, since a one-sided fix is worse than no fix.

- **Two guards added, because both bugs were invisible to every existing check.** A killed
  run leaves a complete-looking `best.pt` that loads and scores like a finished model:
  `_warn_if_undertrained` reads the training log (the only record of epochs *reached*) and
  flags a mismatch; `_is_mid_training` reads `train_v3`'s `.owner` pid marker and *skips*
  rather than warns, since a warning in a buffered log is not a guard. `_assert_same_selector`
  additionally refuses to cache G6-generation (censored-selector) arms alongside later ones.

- **The ablation [1 seed, matched: jaccard overlap, no agg/attn; Δ = paired bootstrap vs
  the v2.2 yardstick].**

  | arm | ALL | s0 | s1 | s2 | s3 | Δ vs v2.2 |
  |---|---|---|---|---|---|---|
  | neither | 15.47 | 0.96 | 4.20 | 29.64 | 27.08 | +0.81 [−1.05, +2.94] |
  | tokens only | 16.86 | 0.00 | 4.84 | 27.20 | 35.40 | +2.20 [−0.98, +5.84] |
  | coverage only | 7.82 | 0.00 | 3.48 | 12.28 | 15.52 | **−6.84** [−9.37, −4.55] |
  | coverage + tokens | 7.71 | 0.00 | 1.28 | 14.24 | 15.32 | **−6.95** [−9.47, −4.62] |
  | — col: coverage only | 10.63 | 0.00 | 1.24 | 18.32 | 22.96 | **−4.03** [−6.20, −1.93] |
  | — col: waste only | 7.81 | 0.00 | 1.36 | 14.24 | 15.64 | **−6.85** [−9.24, −4.62] |
  | deployed (+agg +attn) | 7.50 | 0.00 | 1.16 | 15.80 | 13.04 | **−7.16** [−9.56, −5.00] |
  | deployed, records suppressed | 7.33 | 0.00 | 1.20 | 16.36 | 11.76 | **−7.33** [−9.63, −5.32] |
  | v2.2 yardstick | 14.66 | 0.00 | 6.20 | 26.00 | 26.44 | — |

- **Coverage is necessary and very nearly sufficient.** Both coverage-free arms **tie**
  v2.2 (CIs include 0) and both coverage-bearing arms win by ~7 FP. Record tokens *without*
  coverage buy nothing — 16.86 vs 15.47, i.e. slightly worse than neither.

- **`waste` is the load-bearing column, not `coverage` — the opposite of what the naming
  suggests.** waste-only **7.81** is statistically indistinguishable from both-columns
  (7.71), while coverage-only is **10.63**, a further 2.8 FP away. So the operative signal
  is "this candidate removes objects that were *never* reported as blocking" (a
  cheap-and-wrong detector), not "it removes the ones that were". Worth stating carefully in
  any writeup: the mechanism section describes `coverage`, and the measurement says `waste`.
  **1 seed — this one needs replication before it is load-bearing.**

- **The deployed model does not read its record tokens at inference.** Emptying the evidence
  memory at every step costs **nothing**: 7.33 suppressed vs 7.50 as-trained (marginally
  *better*). This extends A8's 0.23 FP finding — measured there on the pre-coverage G6b
  checkpoint — to the actually-deployed model. Reconciles with A17's 6-seed 1.28 FP token
  contribution: **training on tokens shapes the weights, but the trained model does not
  consume them at deploy.** Those are different claims and both hold.

- **Takeaway / next:** any figure quoting `p8_cov_final` must be re-derived. Highest-value
  follow-up is a 3-seed replication of the **waste-vs-coverage** split — it is a 1-seed
  result that changes how the contribution is described. Deferred: multi-seed caching
  (`--seeds 0 1 2`), PIGINet/VLMPlan on dd2d_v4. Deferred: multi-seed caching (the machinery
  is seed-generic, it only needs `--seeds 0 1 2`), PIGINet/VLMPlan on dd2d_v4.

---

<a id="2026-07-27-p5-observed-coverage-waste"></a>
## 2026-07-27 — **P5: observed coverage/waste — v3 weakly dominates v2.2 at every stratum**

<!--strip-->
> **id** `2026-07-27-p5-observed-coverage-waste` · **status** active · **tracks**
> method, evaluation
<!--/strip-->

- **What:** two scalars appended to `cand_overlap`, computed from the culprits the refiner
  *reported* while failing the candidates already tried:

  ```
  coverage = |S(c) ∩ culprits| / |culprits|      waste = |S(c) \ culprits| / |S(c)|
  ```

  These are exactly §5.1's necessity features, with one substitution: the per-object
  necessity `p_i` is **observed** rather than **predicted**. Necessity conditioning was cut
  ([`decisions.md` 2026-07-26](../decisions/README.md)) because its head would have had to predict `p_i` from geometry.
  Once the refiner reports culprits, the same two features need no head at all — and become
  *more* C2-legal, since nothing is inferred by us. Paired with `--overlap-mode jaccard`,
  which drops the `dead` length proxy that G8 showed was wrong at s1.
- **Headline, 6 seeds** (mean ± std *across seeds* of the per-stratum mean), deployed
  config `--overlap-mode jaccard --coverage-feats --aggregate-records --evidence-attn`:

  | | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | **v3 deployed** | **7.90 ± 0.61** | 0.00 ± 0.00 | 5.60 ± 3.06 | **13.03 ± 1.52** | **12.96 ± 2.46** |
  | *v2.2 yardstick* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* |

  **−6.76 FP, 95% CI [−9.43, −4.40]** (paired bootstrap on the seed-mean per problem).
  **Weak dominance holds — nothing regresses — but the strata are not equally won:** s0 and
  s1 **tie**, s2 and s3 **win by ~2×**.

  ⚠ **s1 is a tie, not a win, and 3 seeds said otherwise.** 5.60 ± 3.06 vs 6.20 is a +0.60
  margin against a 3.06 seed sd (0.20 sd), and only **2 of 6 seeds** beat 6.20 — per-seed
  1.16 / 2.72 / 7.48 / 6.68 / 6.28 / 9.28. At 3 seeds s1 read **3.79** and looked like a
  clear win. This is the one number a 3-seed report would have got wrong, and the reason
  the extra seeds were run. Overall FP is by contrast stable across seeds
  (7.50 / 7.63 / 7.19 / 8.05 / 8.08 / 8.94).

- **Baseline seed count.** The yardstick is v2.2 seed 0 — its *best* of three (14.66 /
  16.57 / 20.57; mean 17.27 ± 3.02, s1 sd ±14.20 because seed 2's `relrank` picked an epoch
  scoring 30.04 at s1). Against the 3-seed mean v3's margin would be −9.37 and it would win
  every stratum. **Reported against seed 0** as the conservative choice.

- **Ablations, 1-seed dev, same yardstick:**

  | arm | ALL | s0 | s1 | s2 | s3 | Δ vs v2.2 |
  |---|---|---|---|---|---|---|
  | deployed (seed 0) | 7.50 | 0.00 | 1.16 | 15.80 | 13.04 | **−7.16** ✱ |
  | coverage + `dead` kept | 7.76 | 0.00 | 2.56 | 12.60 | 15.88 | **−6.90** ✱ |
  | coverage, **no record tokens** | 7.82 | 0.00 | 3.48 | 12.28 | 15.52 | **−6.84** ✱ |
  | **coverage alone (jaccard)** | **8.39** | **0.00** | **2.72** | **12.64** | **18.20** | **−6.27** ✱ |
  | rollout-aligned context, no coverage | 14.34 | 0.00 | 8.04 | 17.48 | 31.84 | −0.32 |
  | evidence-attention alone | 14.92 | 0.00 | 3.56 | 27.48 | 28.64 | +0.26 |
  | no records at all | 15.34 | 0.00 | 4.64 | 26.24 | 30.48 | +0.68 |
  | evattn + aggregate | 15.74 | 0.00 | 4.36 | 24.40 | 34.20 | +1.08 |
  | object-evidence alone | 16.12 | 0.00 | 20.84 | 17.80 | 25.84 | +1.46 |
  | *v2.2 yardstick* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* | — |

  ✱ = CI excludes 0. **Every coverage-bearing arm wins; nothing else does.** The result is
  robust to the exact combination but depends entirely on `coverage`/`waste`.

- **This is goal 1 of the v3 proposal**, and the coverage arms are the first all night to
  beat the yardstick at all rather than tie it.
- **P-v3-1's target is met by a different mechanism than predicted.** The pre-registered
  bar was s2 ≤ astar-dist's 17.08 *via necessity conditioning*; s2 lands at **13.03 ± 1.52**
  via observed coverage. The prediction's *number* is beaten; its *mechanism* was withdrawn.
  Worth stating both ways round rather than claiming P-v3-1 succeeded.
- **This is records driving adaptiveness, which was the point.** `coverage`/`waste` read
  `FailureRecord.culprits` — nothing else in the system has access to which object the
  refiner's own collision check found blocking. At |F|=0 both features are identically zero,
  so the first attempt is still purely static; the signal accrues as the rollout observes.
- **The gain is entirely adaptive, and this is the cleanest demonstration of it.** Because
  `coverage`/`waste` are identically zero at |F|=0, the *first* attempt is a purely static
  decision — and both models make it equally well:

  | | solved on attempt 1 | mean FP among the rest |
  |---|---|---|
  | v3 deployed | **25%** | **10.00** |
  | v2.2 yardstick | **25%** | 19.55 |

  **The whole −7 FP appears after the first failure is observed.** Per stratum, among
  episodes not solved immediately: s1 1.16 vs 6.20, s2 15.80 vs 26.00, s3 13.04 vs 26.44.

  Precise reading, since the coincidence invites over-claiming: that 25% is *exactly* the 25
  s0 episodes, for both models — neither solves a single s1–s3 episode on the first attempt.
  So the honest statement is not "the two have equally good static rankers" but "**the first
  attempt separates them not at all, and every attempt after it does**". Which is still the
  operational meaning of "records drive adaptiveness": the model is not a better static
  ranker, it is a better *re*-ranker. It also corroborates the leakage audit independently —
  a feature leaking feasibility would have lifted the first pick too.
- **Leakage audit (0 violations)**, run before trusting the number: features are exactly
  zero at |F|=0; the culprit set is built only from candidates in the failure context, all
  of which are failures; and the deployment loop breaks on success before a successful
  candidate could ever enter the context.
- **Reading the ablation ladder.** Object-evidence *does* fix s3 on its own (25.84, beating
  26.44) while wrecking s1 (20.84) — the same length-calibration tension G8 found, which is
  why it is built but **not deployed**. `dead` stops being harmful once coverage is present
  (7.76 with it vs 8.39 without): its harm was a *symptom* of the missing count signal, not
  intrinsic. And record *tokens* are worth **1.28 FP** at 6 seeds each (7.90 ± 0.61 with,
  9.18 ± 1.41 without) — concentrated **entirely at s1** (5.60 vs 10.78; without them the
  model is worse than v2.2 there) while s2/s3 tie, and they halve the overall variance. A
  1-seed comparison had put this at 0.26 FP; see `autorun_decisions.md` A17 for the
  correction and why I should have caught it.
- **Process note.** Two `p5_jac_cov` processes raced on one checkpoint path after a
  crash-relaunch (it scored 8.57, then 8.39 as the second overwrote it). Same config, so
  nothing here changed, but `train_v3` now refuses to start on a directory a live run owns.
- **Takeaway / next:** the deployed config is recorded as preset `v3final`. Open: **s1**,
  which is a tie and the widest-spread stratum — the one place v3 does not improve on v2.2; a
  frequency-weighted `coverage` (the observed culprit set over-covers ~2.5×, A12); and
  env-2, without which generality stays architectural.

<a id="2026-07-27-p2-missing-g6-cell"></a>
## 2026-07-27 — P2: the missing G6 cell — record *tokens* cost during training even though the deployed model ignores them

<!--strip-->
> **id** `2026-07-27-p2-missing-g6-cell` · **status** active · **tracks** method,
> evaluation
<!--/strip-->

- **What:** the cell the G6 ablation never ran — **records OFF, overlap ON** — which is the
  closest v3 analogue of the v2.2 configuration. Plus `aggregate_records` (one token per
  failing *query* rather than per failed *sample*). Both 1-seed dev on dd2d_v4.
- **Result — uncensored deployed FP, dd2d_v4 test, n=100:**

  | arm | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | p2 `norec` (overlap ON) | **15.34** | 0.00 | **4.64** | 26.24 | 30.48 |
  | p2 `agg` (records, aggregated) | 15.80 | 0.00 | 5.96 | 27.88 | **29.36** |
  | G6b rec+ov (raw records) | 16.17 | 0.00 | 8.56 | **22.00** | 34.12 |
  | *v2.2 yardstick* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* |

- **Two facts that look contradictory and are both true.** `suppress_records` (running the
  G6b checkpoint with its evidence memory emptied at every step) moves it only
  **16.17 → 16.40**, so the *deployed* model barely reads its records. Yet training with
  those same records costs **−0.83 FP overall and −3.9 at s1** against the no-records cell.
  A token stream the model learns to discard is **not free**: it is ignored at inference but
  still shapes the weights while training.
- **The G6 headline is therefore mis-attributed.** Its −3.37 "record increment" was measured
  against a bar with overlap *also* removed. Against the honest bar, records are negative.
  G7's −5.07 for overlap was the real effect all along.
- **Aggregation is a genuine fix, not a wash:** −0.37 vs raw records, s1 8.56 → 5.96, and
  s3 34.12 → **29.36** — the best s3 of any arm to date. So the token flood (mean 226 tokens
  at |F|=30, max 2045, against v2.2's ~40 facts) was a real defect.
- **No single configuration holds every stratum.** s1 belongs to no-records (4.64, beating
  v2.2's 6.20), s2 to *raw* records (22.00, beating 26.00), s3 to *aggregated* records
  (29.36, still short of 26.44). s3 is the only stratum where nothing v3 does wins.
- **Takeaway / next:** two structural suspects, both now measured rather than guessed.
  (i) The scorer concatenates scene, global and record tokens into **one** attention memory,
  so ~10 scene tokens compete with up to 2045 record tokens and discarding evidence is
  loss-minimizing — hence a separate evidence channel. (ii) With v2.2's inherited context
  sampling, **53.7% of training examples carry no evidence at all**, while a deployed s3
  rollout is at |F|=0 for only 3.2% of its decisions — so the model is trained as a static
  ranker most of the time.

<a id="2026-07-27-g8-dropping-dead-fixes-s1"></a>
## 2026-07-27 — G8: dropping the `dead` feature fixes s1 outright; s3 is the last blocker

<!--strip-->
> **id** `2026-07-27-g8-dropping-dead-fixes-s1` · **status** active · **tracks**
> method
<!--/strip-->

- **What:** three arms on dd2d_v4 [1-seed dev], testing the two hypotheses from the G6b
  post-mortem. `jac` drops the `dead` column from the net's `cand_overlap` (keeping the
  sound demotion outside the net); `tailF` spreads half the training |F| mass out to 40,
  because v2.2's inherited cap of 8 never shows the model the regime an s3 rollout visits.
- **Result — uncensored deployed FP, dd2d_v4 test, n=100:**

  | arm | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | g8 `jac` | 16.86 | 0.00 | **4.84** | 27.20 | 35.40 |
  | g8 `tailF` | 17.38 | 0.00 | 6.80 | 27.92 | 34.80 |
  | g8 `jac+tailF` | 18.31 | 0.00 | 6.04 | 25.64 | 41.56 |
  | G6b rec+ov (reference) | **16.17** | 0.00 | 8.56 | **22.00** | **34.12** |
  | *v2.2 yardstick* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* |

- **The s1 diagnosis was right, and the fix is decisive.** Removing `dead` from the net
  takes s1 from 8.56 to **4.84** — past v2.2's 6.20. It was a disguised shortness cue
  (corr(dead, |S|) = −0.284, mean |S| 1.38 dead vs 2.39 alive), sound as an outside-the-net
  offset but, as a *feature*, a free-running "short ⇒ bad" correlate. s1 is exactly the
  stratum where short is correct, so it took the damage. This is L4 reappearing as a
  feature rather than as a token.
- **But it is a trade, not a free win.** The same shortness bias is *correct* at s2/s3, so
  dropping it costs s2 (22.00 → 27.20) and s3 (34.12 → 35.40), and ALL gets slightly worse.
  Length calibration is being carried by a feature that is right at one end of the stratum
  range and wrong at the other.
- **`tailF` does not fix s3** (34.12 → 34.80), so the train/deploy |F| mismatch is real but
  is *not* what limits s3. Recorded as a negative result; the knob stays available.
- **Where this leaves weak dominance.** Per stratum the best v3 arms already have s1 (4.84
  vs 6.20) and s2 (22.00 vs 26.00). **s3 is the only stratum still losing** (34.12 vs
  26.44), and it alone decides the headline: a model with s1 4.84 / s2 22.00 / s3 26.44
  would average **13.32**, beating the yardstick's 14.66.
- **Takeaway / next:** stop tuning the length proxy and give the model the signal it is
  proxying for. The record fields already contain it — how many *distinct* objects have been
  observed to block the target is a direct statement of how many removals are needed — which
  is what the P3 object-evidence column and the P4 evidence-attention channel are for.

