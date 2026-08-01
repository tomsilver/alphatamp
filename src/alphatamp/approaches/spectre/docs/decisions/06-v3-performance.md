# SPECTRE Decisions — v3 performance push

11 entries, 2026-07-27 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-07-31-unified-coverage-waste-is-the-deployed-definition"></a>
## 2026-07-31 — Unified coverage/waste becomes the deployed definition; the checkpoint decides, not the default

<!--strip-->
> **id** `2026-07-31-unified-coverage-waste-is-the-deployed-definition` · **status**
> active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**Context.** v3's deployed `coverage`/`waste` were computed against
`S(c) = args \ goal_objects` — "discretionary work = touching non-goal objects". That is
true on DD2D and false wherever tools exist: on StickButton2D every candidate has
`S(c) = {stick}`, so coverage is identically 0 (the culprit buttons *are* goal objects,
structurally barred from `S`) and waste is identically 1 for every stick-using plan,
including the one that responds perfectly to the evidence. Blind and anti-signed
respectively. [`unified_culprits_coverage_waste.md`](../unified_culprits_coverage_waste.md)
replaces both with definitions derived from the operator schemas alone.

**Decision: the unified definitions are the default for new training runs.**
`TrainV3Config.unified_coverage = True`; `--legacy-coverage` opts out and writes to a
`_legacycov` directory so the two can never overwrite each other. The deployed comparison
arm (`_V3_ARMS["spectre3"]`) points at `checkpoints_v3_unified`.

**Justified by measurement, not by the argument.** dd2d_v4 test, n=100, 3 seeds,
uncensored, demotion off: **5.78 ± 0.10 against the deployed 7.44 ± 0.76 — −1.66 FP, 95% CI
[−2.71, −0.71]**, with **every unified seed beating every deployed seed** and nothing
regressing at any stratum. Seed variance falls 7×. Numbers and the mechanism in
[`notebook.md`](../notebook/06-v3-performance.md) 2026-07-31.

**Why it wins is not the reason it was proposed.** The design argued the two definitions
would be near-identical on DD2D. They are not: coverage values differ on 100% of contexts.
The gain comes from a fact nobody had looked at — `__wall__` is the single most frequent
reported culprit on dd2d_v4 (1321 mentions vs 939 for `target`), it is not an abstract
object, and the deployed formula keeps it in the denominator where it is permanently
uncovered. The Actionable filter removes it: mean |K| 1.99 → 1.67, coverage spread +48%,
and the 8% of contexts where the deployed feature was *completely flat* drop to 0%. The
design doc's claim that actionability is "honestly idle on both current environments" was
wrong for DD2D and has been corrected.

**Consequences.**

- **The checkpoint decides which definition is used, not the current default.**
  `inference_v3._emit_kwargs` reads `cfg.get("unified_coverage")`, so a checkpoint trained
  before today has no key, resolves to `False`, and keeps being scored on the features it
  was trained on. This is the property that makes flipping a default safe at all, and it is
  asserted by the deploy-kwargs round-trip test.
- **`build_v3_example`'s own default stays `False`.** Policy lives in `TrainV3Config` and
  the checkpoint; the tensorizer does what it is told. Two callers do not pass the flag —
  `precompute_dd2d_cache` (which correctly forwards `**deploy` from the checkpoint) and
  `spectre_d2_s2` (a frozen diagnostic that must keep reproducing its old numbers). Flipping
  the primitive would have silently changed the second.
- **Every pre-2026-07-31 v3 number stands as measured** and is not retracted — those models
  were trained and scored on a consistent definition. What changes is which arm is deployed.
  The ablation arms (`abl_*`, `abl_with_demotion`, `abl_suppress_records`) deliberately keep
  their old checkpoints; they are a frozen component study.
- **Re-cache with `--force`.** `_dir_complete` skips a full directory, so without it the
  comparison row silently keeps the previous definition's rollout — the same failure mode as
  the 2026-07-30 demotion switch.
- **Cost:** ~112 → ~78 s/epoch after memoizing the hoistable work (`matched_steps`, `touch`,
  `blame`, `collateral` into a per-(candidate, record) `_Memo`). Verified output-identical
  two ways: re-scoring unchanged checkpoints is bit-identical, and a full retrain lands at
  5.78 vs 5.83, inside seed noise.
- **This is also what unblocks StickButton2D**, which is the reason the redesign happened:
  the same definition is now what wins on DD2D *and* what makes coverage non-vacuous on
  SB2D (−1.33 attempts at b3, −7.20 at b5 in the offline re-rank probe).

---

<a id="2026-07-30-proof-tier-demotion-cut-deployed-method-v3"></a>
## 2026-07-30 — Proof-tier demotion is cut from the deployed method; v3 is a purely learned ranker

<!--strip-->
> **id** `2026-07-30-proof-tier-demotion-cut-deployed-method-v3` · **status** active ·
> **tracks** method, evaluation
<!--/strip-->

**Context.** v3's contribution is *one canonical record, consumed by learned components*.
Proof-tier demotion sat outside that story: an external, hand-declared deduction acting
directly on the ranking. It brought its own axiom registry, its own `strict`/`permissive`
semantics and its own soundness argument (C5/P-E). The 2026-07-30 ablation
([`notebook.md`](../notebook/README.md)) priced it at **0.23 FP** (7.20 → 7.44, 3 seeds, CI
[+0.08, +0.43]) — real and significant, and small.

**Decision (user call): cut proof-tier demotion from the deployed method.** The
authoritative v3 model is the demotion-free one at **7.44 ± 0.76**, and
`apply_demotion=False` is now the default *everywhere* — `deployed_rollout_v3_traced`,
`cache_spectre3`, `train_v3.deployed_val_fp`, and `spectre_score_v3.py` (whose
`--no-demotion` became an opt-in `--with-demotion`).

**The machinery is kept, not deleted.** `ProofStateV3`, the axiom registry and both
demotion modes remain, tested, one flag away. The deduction is *sound* — 0
demoted-but-feasible under `strict` — and on a domain whose proofs fire more often than
DD2D's 6% the trade would go the other way.

**Why the price is affordable, measured on DD2D:**
- it **barely fires** — changes the realized attempt order on **18/300 (6%)** of deployed
  (problem, seed) pairs, against **55/100 (55%)** on the stripped floor arm;
- the learned components **absorb ~79%** of it — the same switch costs **1.09 FP** on the
  floor arm (jaccard only, no coverage/waste, no record tokens);
- it **only ever acted at s2/s3** — s0 and s1 are bit-identical with and without it, at
  both 3 and 6 seeds, because demotion needs a subset-containment proof and an s1
  candidate stages one object.

**Consequences.**
- **The headline moves and is now conservative in a new way.** 7.20 → **7.44 ± 0.76**; vs
  the v2.2 yardstick **−9.83 FP, CI [−12.57, −7.36]** (was −10.06). The yardstick **keeps
  its own demotion** — 17.27 is the number published for v2.2 throughout this project, and
  re-scoring the baseline to match a v3 design choice would be moving the goalposts. So
  the margin is measured against a *stronger* baseline than v3 gives itself.
- **"Learning is the floor" stopped being the fallback and became the configuration.** The
  `DomainSpec` is still read for `manipulated` / `goal_objects` / `length_key`, but its
  **axiom declarations are no longer required at deploy**. A new environment now needs a
  converter and refiner instrumentation; the axiom block is a tuning knob.
- **1.3% of records are now consumed by nothing.** The tier split still holds proof-tier
  records out of the token path — measured 391/29054 on dd2d_v4, all `retrieve` — but
  there is no longer a proof consumer for them. Routing them into the token path is the
  obvious follow-up and needs a retrain, so it was not done. *(This also corrects
  `autorun_decisions.md` A19, which guessed from one episode that "most DD2D failures are
  `retrieve`" and therefore proof-tier; across the test split it is 1.3%.)*
- **The selector moved with the method.** `deployed_val_fp` now passes
  `apply_demotion=False` **explicitly** rather than inheriting it, so a future change to
  the library default cannot shift the selection criterion silently. Existing checkpoints
  were selected under the old criterion; the difference is ~0.2 FP and they were not
  retrained.
- **The D-8 equivalence oracle had to be pinned.** `test_v3_equivalence` compares v3 in
  compat mode against the v2.2 rollout, and v2.2 *always* demotes — so the new default
  made it compare two different policies and pass only where the offset happened not to
  change an argmax. It now forces `apply_demotion=True`, as does the stale-cache test,
  whose cache was written when demotion was part of the method. An oracle that quietly
  stops testing what it claims is worse than no oracle.
- **The comparison notebook's §4.3 inverted**: the deployed row is the demotion-free one
  and switching it back on is the ablation (`abl_with_demotion`, `abl_floor_with_demotion`
  in `_V3_DEMOTION_ARMS`). The inspector's `demoted@t` column is now descriptive for v3 —
  the deduction is still computed and recorded, it just no longer moves the ordering — and
  still causal for v2.2.
- **Superseded:** G7's "keep both anyway, and the reason is soundness not FP" stands as a
  statement about *soundness*, but its practical conclusion is reversed for the deployed
  configuration. The soundness argument did not change; the weight given to story
  coherence over 0.23 FP did.

---

<a id="2026-07-29-stickbutton2d-heuristic-distance-term"></a>
## 2026-07-29 — StickButton2D's heuristic gains a distance term and a weight above 1; b5 joins the collectable set, b10 is dropped

<!--strip-->
> **id** `2026-07-29-stickbutton2d-heuristic-distance-term` · **status** active ·
> **tracks** env-stickbutton2d, data · **supersedes**
> 2026-07-28-stickbutton2d-subclass-plan-generator · **see also**
> `autonomous_stickbutton_session.md`, `kinder_stickbutton2d_map.md`
<!--/strip-->

Autonomous session under an explicit brief: reach ≥50% of problems solvable at b5/b10 within
200 attempts × 20 s, **without touching kinder's refiner or sampler**, by improving the A*
heuristic — and keep the heuristic simple enough to tell a story about. Full decision record
in [`autonomous_stickbutton_session.md`](../autonomous_stickbutton_session.md); numbers in
[`notebook.md` 2026-07-29](../notebook/06-v3-performance.md#2026-07-29-stickbutton2d-b5-reaches-75).

**Context.** Under exact-equality acceptance a skeleton refines only if no leg of its
trajectory sweeps a still-unpressed button (the env presses whatever the robot drives over).
Every press ordering has the *same plan length*, so a count-based heuristic rates all of them
tied — 120 orderings at b5, ~3.6M at b10 — and the pool is filled arbitrarily.

**(a) The heuristic gains one term: distance to the nearest unpressed button.** Normalised by
the world diagonal so it stays in [0, 1] and can only break ties between equal-length plans.
The justification is a two-line argument rather than a tuned penalty: if you always go to the
*nearest* remaining button, nothing unpressed can lie on the way, because anything on that
segment would have been nearer and would have been the target instead. So "walk to the
nearest one" and "never press out of order" are the same preference.

Rejected on measurement: **remaining-tour length**, the other obvious formulation. Because the
counting part is constant along every path, ranking by what is *left* rewards clearing a far
outlier early — the exact inverse of nearest-first. First success on b5/seed5 came at candidate
145 with the tour, 78 with distance-to-nearest.

**(b) Reach becomes a grounding restriction, not a heuristic surcharge.** `RobotPressButton*`
is not grounded on a button past `robot_reach_max_y`. The `+1` surcharge cannot do this job: it
is a constant, and pressing an out-of-reach button still lowers `|unpressed|`, so A* keeps
rating those plans optimal.

**(c) The count term is weighted 1.05, and the reason is structural.** Each press adds 1 to `g`
and removes 1 from the count, so at weight exactly 1 the score is **depth-invariant** and the
search never descends — b10 returns an empty pool after 30 s. It is only *just* above 1 because
the quantity that matters is how much the 200 candidates differ in their **opening** moves
(failures land at step 0–1, so a shared prefix fails as a block): on b5, distinct first press /
first three of 200 candidates is 5/32 at 1.05 — identical to weight 1.0 — but 2/7 at 1.5 and
1/2 at 2.0. Both bounds are pinned by test.

**Consequences.**
- **b5 goes 0/8 → 15/20 (75%) and joins the collectable set**; b3 stays 100% and its first
  refinable candidate moves from index 14–16 to 2–10. **b10 stays 0/20 and is dropped.**
- **b10's blocker is not the heuristic and should not be attacked as one.** All 200 candidates
  share the same first *three* presses at every workable weight, because a single A* run yields
  goals in `f` order and never revisits an opening. Fixing it needs prefix-diverse plan
  generation (forbid-loop / top-k, as DD2D's enumerator does) — a generator change, scoped
  separately. Quantising the distance term to force RNG tie-breaks was tried and does nothing.
- **Nearest-first is a prior, not a guarantee**: the single explicit nearest-first plan refines
  for only b3 55% / b5 25% / b10 5%, because the robot has a body and the stick is 1.25 long.
  This is why the method needs 200 attempts rather than 1, and it bounds how much any ordering
  heuristic can achieve here.
- **Cost**: ~900–2700 s per b5 problem (every candidate refined), so a 400/100/100 b5
  collection is ≈5 h at 30-way parallelism.
- The `acceptance="superset"` sampler option from the previous day remains **off and unused** —
  it was outside this brief and is not part of this result.

---

<a id="2026-07-28-stickbutton2d-subclass-plan-generator"></a>
## 2026-07-28 — StickButton2D: subclass the plan generator (upstream ignores `heuristic_name`), keep stock refinement semantics, gate the dataset on measured feasibility

<!--strip-->
> **id** `2026-07-28-stickbutton2d-subclass-plan-generator` · **status**
> partially-superseded · **tracks** env-stickbutton2d, data · **superseded by**
> 2026-07-29-stickbutton2d-heuristic-distance-term
>
> ⚠️ **PARTIALLY SUPERSEDED** — b5 is collectable after the 2026-07-29 heuristic
> change (0% → 75%). The substrate map, the stock-refinement decision and the
> probe-is-not-a-bound rule all stand; only b5's verdict moved. b10 remains dropped.
<!--/strip-->

**Context.** SPECTRE's method comparison exists on one environment (DD2D); the paper needs a
second. StickButton2D was chosen because the env, predicates, operators, controllers and
refiner all ship upstream (`kindergarden` / `kinder-bilevel-planning` / `bilevel-planning`),
so nothing about the environment has to be re-implemented — and the generic collector
(`collect.py`) already drives kinder envs end-to-end, with `conf/env/stickbutton2d_b5.yaml`
and its submit script already present and correct.

The plumbing was never the risk. **Label sparsity was**, and it was severe: under the stock
pipeline b5 produced **0 successes out of 200 candidates** and b10 zero out of 40, which
`dataset.py` turns into an empty dataset (`num_success == 0` episodes are dropped). Numbers
in [`notebook.md` 2026-07-28](../notebook/README.md); substrate map in
[`kinder_stickbutton2d_map.md`](../kinder_stickbutton2d_map.md).

**(a) The plan generator is subclassed, not configured — because the config knob is a lie.**
`RelationalHeuristicSearchAbstractPlanGenerator` accepts `heuristic_name`, stores it, and then
hardcodes `create_pyperplan_heuristic("hff", ...)` at line 198. Passing `"hadd"` or anything
else has no effect. So the only extension point is the *base*
`HeuristicSearchAbstractPlanGenerator` plus our own `heuristic_factory`
(`envs/stickbutton2d/heuristic.py`).

The heuristic itself supplies the one fact kinder's symbolic model omits: the robot's base has
`ZOrder.ALL` and so cannot drive onto the `ZOrder.FLOOR` table, giving a reach limit of
**1.405** *derived from `StickButton2DEnvConfig`*, not hardcoded. Buttons past it need the
stick. hff cannot see this, so it ranks bare-robot plans on unreachable buttons first — they
are symbolically shortest — and they crowd out the pool. `h = |unpressed| + 1[table button
remains ∧ hand empty] + 1[robot-only button remains ∧ holding stick]`; both extra terms count
a genuinely unavoidable action, so it stays admissible, and on a scene with no table buttons
it degenerates to hff's ordering exactly (verified: b3/seed1 first success at index 14 both
ways; b3/seed0 improves 29 → 16).

**(b) Stock kinder refinement semantics are kept, after measuring the alternative.** The
sampler accepts a step only on *exact* abstract-state equality
(`parameterized_controller_sampler.py:89`). Relaxing that to `planned ⊆ achieved` is sound for
goal achievement — all preconditions are positive and `Pressed` is never deleted, so the final
achieved state contains the goal — and it dramatically improves the *per-button* probe
(b5 0% → 75%, b10 0% → 55%), because incidental presses are the largest single blocker there.

**It buys nothing on real skeletons: b3 is 8/8 with or without it, b5 is 0/8 either way.**
Once an incidental press makes the world diverge from the symbolic plan, later steps are
checked against a plan that no longer describes reality. So the deviation does not pay for
itself and is **not** adopted; it survives as `sampler.py`'s `acceptance="superset"`, off by
default, so the measurement is reproducible rather than folklore.

**(c) Raising sampling budgets is not a lever, and this is worth recording so it is not
retried.** `PickStick*` is the *only* StickButton2D controller with non-degenerate sampled
parameters — every press/place controller's `sample_parameters` returns a constant, so
`num_sampling_attempts_per_step > 1` is a literal no-op there. And even where it applies it
does not help: over 10 b5 scenes, 5 → 25 → 100 attempts gave **7/10 every time with the same
three scenes failing**, at 13× the cost. Stick-graspability is a property of the scene, not of
the search.

**(d) The cheap per-button probe is a failure-attribution tool, not a feasibility predictor —
it is not a bound in either direction.** Because the goal demands all N buttons, one
unpressable button voids every skeleton, so probing each button's robot-route and stick-route
independently costs `2N` refinements instead of `K_max` (≈380 s/problem at b5). It was
initially recorded as an upper bound. **That was wrong**, and measuring against `full` mode
showed it errs both ways: it under-called b2/b3 (55%/35% against a true 100%) because probing
from `x0` tries one route per button while a real skeleton can reach the same button from a
different predecessor or press it incidentally, and it over-called b5 under `superset` (75%
against 0%) because real skeletons must chain presses.

**Consequence, stated as a rule: a variant's collectability is decided in `full` mode, never
from the probe.** The probe's job is saying *why* refinement fails — out-of-reach vs stuck
controller vs extra atoms — which is what produced (a)–(c). This is the same class of error as
the censored-selector episode (2026-07-26): a cheap proxy that tracks the expensive quantity in
the easy regime and diverges exactly where the decision gets made.

**Consequences.**
- New package `envs/stickbutton2d/` (`geometry`, `heuristic`, `sampler`, `diagnostics`), new
  harness `experiments/spectre/stickbutton_feasibility.py`, env configs for b1/b2/b3/b10,
  explicit `_TYPE_AUG_POLICIES` entries, 13 unit tests.
- `collect.py::_make_plan_generator` gains a fourth dispatch branch; `plan_generator=
  "heuristic_search"` opts back into stock hff ordering for an apples-to-apples comparison.
- **b1/b2/b3 are collectable — `full` mode gives 100% of problems ≥1 success at `k_max=60`,
  8 problems each, clearing the 80–90% bar. b5/b10 give 0%.** Positives thin with button
  count (24.2 / 9.1 / 2.8 successes per 60 candidates at b1/b2/b3), so b3 is the interesting
  ranking problem and b5 is the cliff. The pooled dataset's variant mix is the user's call.
- **Correction to an earlier reading of *why* b5/b10 fail.** This entry first attributed them
  to the controllers lacking obstacle avoidance. That is wrong, and nothing in the
  environment supports it: buttons are `ZOrder.NONE` and never block motion, and the only
  barrier is the table, which a correct plan never drives at. Step-level attribution
  (102/120 skeletons fail at **step 0**, so not length-compounding either) shows two
  independent causes — out-of-reach plans failing on *missing* atoms, and incidental presses
  failing on *extra* atoms. Detail in `kinder_stickbutton2d_map.md` §7.
- **The conclusion survives the correction, the reasoning does not.** Pruning impossible
  groundings and relaxing acceptance each measure as useless *alone* because the other masks
  them; together they take b5 from 0/4 seeds to 2/4 and leave b10 at 0/4 — real movement,
  still far short of the bar. The residual is genuine controller failure (`PickStick` on
  scenes where the stick is not graspable). So b5/b10 stay out, for a different and
  better-evidenced reason.
- Cost anchor: `K_max=200` is ~3× the `k_max=60` cost, so 400/100/100 ≈ **6–8 h at ~30-way
  parallelism**. Collection refines every candidate by design.
- Verified end-to-end on b3: collect → `spectre_build_vocab` → `spectre_check_pipeline` with
  **0 episodes filtered**, and per-episode first-success indices matching the harness exactly.
- Not addressed, and blocking a v3 run on this env: the kinder collection path leaves
  `scene_geometry=None` and `refiner_metadata={}`, so the v3 `FailureRecord` pathway is inert
  and `domain.spec_for` falls back to `EMPTY_SPEC`. Refiner instrumentation is a separate
  piece of work.

---

<a id="2026-07-28-state-delta-deployed-3-seed-protocol"></a>
## 2026-07-28 — v3+state-delta is the deployed model; the DD2D comparison becomes 3-seed and PIGINet gains a seed axis

<!--strip-->
> **id** `2026-07-28-state-delta-deployed-3-seed-protocol` · **status** active ·
> **tracks** method, evaluation, baselines · **ratifies** `autorun_decisions.md` A18 ·
> **see also** `as_built_v3.md` §7.1
<!--/strip-->

Follows the entry below, which built the state delta and measured it as a tie. Three
decisions, each with a consequence that outlives it.

**(a) The state-delta configuration is deployed, on a tie.** `spectre_sweep`'s `v3final`
preset gains `--state-delta`, `_V3_ARMS["spectre3"]` repoints to
`checkpoints_v3_v3delta_s{seed}`, and the separate `v3delta` preset is folded away — two
names for the "current" config is how two current configs end up on disk.

The justification is **not** the number, and the doc says so. Over 6 seeds the delta arm is
nominally *behind* (8.23 ± 1.36 vs 7.90 ± 0.61, +0.34 FP CI [−0.30, +1.07], sign splitting
3–3); over the 3 seeds the comparison protocol uses it is nominally *ahead* (7.20 ± 0.62 vs
7.44 ± 0.23). Both are ties. What decides it is that `s_j` is the last unimplemented field of
§6.1's record schema and it costs a new environment **nothing** — it is derived from `s_0`
plus the candidate's own plan, both of which the converter already supplies, so no
instrumentation, no vocabulary, no per-env routine. A record type that advertises a field it
does not carry is a liability in a paper whose contribution is *one canonical record*;
paying nothing to close that is worth more than ±0.3 FP of noise.

**Consequence, stated because it is the uncomfortable half:** the reported v3 headline moves,
and which direction depends on the seed count. Every place the 3-seed figure appears now
carries the 6-seed one beside it (`as_built_v3.md` §7.1, the notebook header). That is the
A18 discipline applied to ourselves rather than to the baseline.

**(b) The comparison is 3 seeds, because that is what every method has — and v3 has 6.**
v2.2 was trained at exactly 3 seeds; PIGINet gets 3 (below). So 3 is a protocol set by the
weakest-covered method, not a subset chosen after seeing results. But it *is* the
better-looking half of v3's six, and pretending otherwise would be exactly the selective
reporting this log has criticised elsewhere, so the 6-seed number is disclosed everywhere the
3-seed one is quoted.

A second, subtler basis change comes with it: the yardstick moves from v2.2 **seed 0**
(14.66, its *best*) to its 3-seed mean (17.27 ± 3.02), because with both sides at 3 seeds the
like-for-like comparison is mean-to-mean. That accounts for roughly 2.6 of the −10.06 FP
headline; against seed 0 the margin is ≈ −7.5. Recorded so nobody reads the headline growing
from −6.76 to −10.06 as v3 having improved by 3 FP.

**(c) PIGINet gets a real seed axis, and the cache layout is detected rather than assumed.**
PIGINet had no `--seed` flag: its only randomness is negative subsampling and torch init, so
three "seeds" would have been three identical runs and any ± a fabrication. `train.py` now
seeds torch, numpy and `PIGINetDataset` together, and records the seed in
`train_metrics.json`. Verified it bites: epoch-0 train loss differs across the three runs
(0.4822 / 0.5003 / 0.4861).

That makes two PIGINet cache layouts live at once — flat `piginet/<pid>.json` on dd2d_v2/v3
(genuinely one deterministic run each) and `piginet/seed_<n>/` on dd2d_v4. The reader
**detects** which it is looking at rather than being told. Both failure modes are real and
silent: fabricating a `seed_0` for a flat cache reports a one-sample spread for something
never sampled, and collapsing a seeded cache to `seed=None` discards the spread we paid to
measure. `build_table` already renders the two differently (`-` and a bare mean vs a count
and a `±`), so the distinction has to survive loading for that to mean anything.

**Consequences.**
- **The notebook's every `±` changed meaning.** It reported the across-*problem* spread of a
  single seed, which reads like a stability claim and is not one. §1/§2 now come from
  `build_table`'s across-seed spread — the same function behind `spectre_v3_table.py`, so the
  notebook and the CLI reporter cannot drift.
- **§4 is pinned to seed 0 explicitly**, and not only because its arms are 1-seed: `_abl_row`
  pairs on `problem_id`, and `.loc[common]` over a multi-seed frame returns one row per
  (seed, problem), so a multi-seed frame silently feeds the paired bootstrap mismatched
  arrays. The v2.2 baseline it measures Δ against is pinned to seed 0 for the same reason.
- **§3 averages per-seed survival curves** rather than pooling all (seed, problem) attempts,
  which would fold seed spread into what reads as a distribution over problems and make a
  3-seed method look smoother than a 1-seed one for no real reason.
- **Re-caching `spectre3` requires `--force`.** `_dir_complete` skips a full directory, so
  without it seed 0 stays pre-delta while seeds 1–2 are the delta model — one method row
  silently mixing two generations. Same class of bug as the double-canonicalization cache.
- The six `abl_*` arms keep their pre-delta checkpoints and are labelled as a frozen seed-0
  component study; re-running them at 3 seeds was out of scope.
- VLMPlan stays on dd2d_v3 at one seed — two model arms × 100 problems is ~10.5 h of
  generation, and it is the only method still grafted (marked † in §2's chart).
- **A reader can be wrong without erroring: never read a compare cache mid-build.**
  Querying the table while `spectre2_adaptive/seed_2` was still filling returned v2.2 =
  12.68 ± 5.17 (seed 2 reading 15.21 instead of 20.57). `_dir_complete` is consulted by the
  *writer* only, so a half-filled `seed_N` directory loads cleanly and the mean is taken
  over however many problems exist so far. Caught only because seeds 0 and 1 matched
  published values exactly and seed 2 did not — i.e. by the cross-check, not by any guard.
  Recorded rather than fixed: the fix would be a completion marker per seed dir, which is
  worth doing if this ever bites unattended.
- **The comparison gained a second, unplanned result**: PIGINet and v2.2 land on the *same*
  mean (17.27) with 16× different seed spreads (±0.19 vs ±3.02) and opposite per-stratum
  profiles. It is a coincidence, and it is the clearest possible argument for the
  across-seed `±` this change introduced — under the old across-problem number the two rows
  would have looked equally reliable.

---

<a id="2026-07-28-state-delta-on-record-ties"></a>
## 2026-07-28 — §6.1's `s_j` is built, as a delta on the record rather than a state; it ties, and ties are the outcome we keep

<!--strip-->
> **id** `2026-07-28-state-delta-on-record-ties` · **status** active · **tracks**
> method · **resolves** D-8 · **see also** `SPECTRE_v3_proposal.md` §6.1
<!--/strip-->

**Context.** `SPECTRE_v3_proposal.md` §6.1 defines the canonical `FailureRecord` as seven
things. Six were built. `s_j` — *the abstract state at step j, symbolic simulation of the
prefix, pure STRIPS, no geometry* — had no field, was never computed, and the module
docstring's own field table silently omitted the row. It was the schema's one unimplemented
slot, and a record type that advertises a field it does not carry is a liability in a paper
whose contribution is *one canonical record*.

**Decision (a): build it, and carry the delta rather than the state.** `s_0` already reaches
the scorer through the scene tokens, so a full state would re-send what the model has and
bury the new part in it. `StateDelta(added, deleted)` carries `s_j \ s_0` and `s_0 \ s_j` as
sorted `(predicate, args)` name pairs. Config-gated `--state-delta`, **off by default**.

**Decision (b): it lives on `FailureRecord`, and that turned out to be the clean option
rather than the expensive one.** The concern was canonicalization: `records_for_candidate`
is called on canonical episodes by the tensorizer and on *raw* ones by the proof state, so a
new object-naming field looks like it needs a fourth `_remap_refiner_metadata` role and a
backfill rule. It needs neither. **`FailureRecord`s are built on demand and never
serialized**, so a derived field inherits whatever namespace its episode is in — exactly as
`unmoved` already does — and the delta is derived from `initial_abstract_state` and
`operator_seq`, both of which `canonicalize_episode` already remaps. Computed *after*
aggregation, "the state at the furthest point this query reached" also falls out for free,
because `_aggregate_per_query` already keeps the deepest record.

**Decision (c): the branch is additive and zero-initialized, not a widened projection.**
Widening `RecordEncoder.proj[0]` from in-width 100 to 164 re-randomizes **every** weight in
that layer (measured: 0.177 max shift on the shared block, against a kaiming bound of 0.100)
— the same init confound `V3Config` already warns about for `n_prior_feats`, where enabling
the prior also zero-inits the head and made every historical prior on/off delta
uninterpretable. Instead `delta_proj` is summed in with zero weight and bias, and the new
submodules are constructed **last**. Consequence: a flag-on model is *functionally identical*
to flag-off at step 0, pinned by test, so the measured difference is the feature and not the
draw. **This is the pattern to reuse for any future v3 feature**, and it generalizes D-8 one
level down: with the flag off the *deployed v3* state dict is unchanged, so
`checkpoints_v3_v3final_s*` still load `strict=True`.

**Decision (d): re-score the frozen baseline under the new code before training anything.**
It reproduced 7.90 ± 0.61 and every stratum digit-for-digit. Cheap, and it is the check whose
absence produced the 13.68-vs-14.50 episode: without it, "exact absence" is a claim about
code rather than a measurement.

**Outcome: a tie, at 6 seeds.** 8.23 ± 1.36 vs 7.90 ± 0.61; paired bootstrap **+0.34 FP, 95%
CI [−0.30, +1.07]**, sign splitting 3–3 across seeds. The pre-registered bar was beat-or-tie
and no configuration search was run — one arm, `v3final`'s flags plus one. Numbers and the
per-stratum table in [`notebook.md` 2026-07-28](../notebook/README.md).

**Consequences.**
- **Kept, default off, zero porting cost.** The delta needs *no new instrumentation*: it is
  derived from data the converter already supplies, and its predicate vocabulary comes from
  the same `train_vocab.json` everything else reads. `porting_guide.md` says so explicitly,
  because "you do not emit this" is the kind of thing a future porter would otherwise
  re-derive.
- **The tie is informative because the feature was consumed.** `delta_proj` trains away from
  its zero init in all six seeds and the checkpoint deploys with `state_delta=True`. That
  distinguishes this from A8's finding (a trained model that had learned to *ignore* its
  record tokens) — the signal was read and did not help. Do not cite the two as the same
  phenomenon; conflating them is how A6/A8 went wrong.
- **Two measured DD2D properties bound what this could have shown here, and neither is a
  property of the mechanism.** (i) The delta's object set is exactly `all_objects − unmoved`
  on 946,063/946,063 records, so its only content beyond an already-present (and still
  un-tensorized) field is the predicate label. (ii) Under `--aggregate-records` 47.8% of
  tokens carry an *empty* delta, 54.9% at s2/s3 — the strata that needed moving had the least
  coverage. On a domain with richer prefix effects or deeper failures the delta carries
  strictly more. Recorded in `as_built_v3.md` §4 so they are not re-derived.
- **A lead recorded, not taken:** `unmoved` still never reaches a tensor. On DD2D it is
  *equivalent* to the delta's object set, so this experiment is also weak evidence that
  tensorizing it alone would not have helped — but it was never run as its own arm, and it
  would be the cheaper of the two.
- **No attribution arm.** It was gated on a win (to separate "predicate structure matters"
  from "we finally tensorized `unmoved`"), and there is no win to attribute.
- **The proposal is updated in place**: §6.1 no longer describes `s_j` as pending.

---

<a id="2026-07-27-cross-collection-grafting-coverage-mode"></a>
## 2026-07-27 — Cross-collection grafting in the comparison notebook; `--coverage-mode`; a checkpoint is not finished until its log says so

<!--strip-->
> **id** `2026-07-27-cross-collection-grafting-coverage-mode` · **status** active ·
> **tracks** tooling, evaluation, env-dd2d · **resolves** A15 · **see also** closes
> the dd2d_v3 **13.68 → 14.50** correction; withdraws A15's `p8_cov_final` preference
<!--/strip-->

Three decisions from retargeting `compare_dd2d_methods.py` to SPECTRE v3. Numbers in
[`notebook.md` 2026-07-27](../notebook/README.md).

**(a) The comparison reads two collections, and the load-bearing row never crosses them.**
v3 exists only on `dd2d_v4` — its `coverage`/`waste` need the culprits only the
instrumented refiner reports — while PIGINet, VLMPlan and SPECTRE v1 exist only on
`dd2d_v3` (retraining them means a fresh CLIP cache and ~10.5 h of VLM generation). Rather
than drop three comparators or re-run them, the notebook reads **v4 as primary** and grafts
**only the methods with no v4 row**, tagging every record with its collection
(`dd2d_compare.merge_collections`; a name present in both resolves to primary).

Licensed by measurement, not convenience: the two test splits have **identical problem-id
sets** (100/100) and agree on 99.2% of scenes and 94.6% of full 200-candidate label vectors
(0.08% of labels differ, 2026-07-26 entry). So the join is well defined and grafted rows are
accurate well within their own seed noise. **v3-vs-v2.2 is native-to-native on both sides**,
which is the point: the claim the table exists to support is exactly paired, and only the
context around it is approximate. The rule generalizes — graft only what cannot be
re-derived, never a method that has a row on the newer collection.

Consequence for the planner inspector: it is **restricted to v4-native methods**. A grafted
method's scores index the v3 pool while the inspector renders the v4 episode, so its rank
column would be wrong on the ~5% of problems whose pools differ. Excluded with a note rather
than shown subtly wrong.

**(b) `--coverage-mode {both,coverage,waste}`.** `--coverage-feats` added both columns as a
pair, so which one carries v3's effect was unmeasured. The flag **zeroes** the unwanted
column instead of narrowing the tensor — the idiom `--overlap-mode` already uses — so the
state-dict shape is untouched and the D-8 exact-absence oracle keeps loading. Default `both`
reproduces the pre-flag behaviour byte for byte (pinned by test).

Also closed here: the **dd2d_v3 13.68**, which the 2026-07-26 entry left pending a
`--force` rebuild, is **14.50** — the figure that entry predicted. 13.68 is superseded
wherever it appears (`as_built_v2.2` §3.7 and two `notebook.md` entries).

**What the ablation it enabled says (1 seed, numbers in `notebook.md`).** `coverage` is
necessary and nearly sufficient — both coverage-free arms *tie* v2.2, both coverage-bearing
arms win ~7 FP, and record tokens without coverage buy nothing. Two results warrant care in
any writeup:

- **`waste` alone (7.81) matches both columns (7.71); `coverage` alone is 10.63.** The
  load-bearing signal is "removes objects never reported as blocking", not "removes the ones
  that were" — the opposite of what §5.1's framing implies. 1 seed; **replicate before
  rewriting the mechanism story.**
- **The deployed model does not read its record tokens at inference** — suppressing them
  costs nothing (7.33 vs 7.50). This does *not* contradict A17's 6-seed 1.28 FP token
  contribution: training on tokens shapes the weights, the trained model does not consume
  them at deploy. Keep the two claims separate; conflating them is how A6/A8 went wrong.

**(c) A checkpoint is not a result until its training log says it finished — and the code
now checks.** Two bugs, both invisible to every existing guard, both found only because a
number looked impossible:

- **`p8_cov_final_s{0,1,2}` are epoch-5-of-30 stubs.** `autorun_decisions.md` A15 named them
  "the clean 3-seed re-run" and recommended them over the race-tainted `p5_jac_cov`. A killed
  run leaves a complete-looking `best.pt` that loads, scores, and fills a cache directory;
  nothing distinguishes it from a finished model, because the checkpoint records the
  *configured* epoch count, not the reached one. It scores 26.97 with **s0 = 36.64**, where
  every other arm gets 0.00 — the tell was a stratum whose correct answer is known.
  **A15's preference for p8 is withdrawn**; anything quoting it must be re-derived.
- **The v1 comparison rows were double-canonicalized** — `cache_spectre` fed already-
  canonicalized episodes into `init_inference_state`, which canonicalizes again. The *same*
  defect that retracted dd2d_v3's 13.68, still live in a sibling function eight months on.
  Worth 1.52 FP and 39/100 problems. Every model cache function now loads through
  `_RawSplit`, and its docstring says so as a rule rather than a local fix.

Three guards, chosen so each fails loudly at the point of use: `_warn_if_undertrained`
(reads the log, the only record of epochs actually reached), `_is_mid_training` (reads
`train_v3`'s `.owner` pid marker and **skips**, because a warning in a buffered log is not a
guard — this run wrote 72 files from a live checkpoint before it was caught), and
`_assert_same_selector` (refuses to mix G6-generation censored-selector arms with later
ones, per 2026-07-26). The standing rule: **a metric computed from a checkpoint is only as
trustworthy as the evidence that the run producing it terminated.**

---

<a id="2026-07-27-necessity-observed-not-predicted"></a>
## 2026-07-27 — Necessity is **observed, not predicted**: `coverage`/`waste` from reported culprits; v3 weakly dominates v2.2

<!--strip-->
> **id** `2026-07-27-necessity-observed-not-predicted` · **status** active ·
> **tracks** method, evaluation, env-dd2d · **defines** G8 · **resolves** G7, P-v3-3,
> C2, C5, L2, R7 · **ratifies** `autorun_decisions.md` A13 · **see also**
> `autorun_decisions.md` A1–A13 for the full overnight chain
<!--/strip-->

Made during an autonomous overnight run (see [`autorun_decisions.md`](../autorun_decisions.md)
for the full A1–A13 chain and everything that did *not* work). This entry records the part
with lasting consequences for the method.

**Context.** After G6b, v3 *matched* deployed v2.2 but did not beat it (16.17 vs 14.66, CI
including 0), and the per-stratum picture was a see-saw: whichever configuration won s1 lost
s3. G8 diagnosed why. The `dead` column of `cand_overlap` is a **length proxy** —
`corr(dead, |S|) = −0.284`, mean `|S|` 1.38 when dead vs 2.39 when not — so it is *correct*
at s3, where long plans are needed, and *wrong* at s1, where short ones are. Dropping it took
s1 from 8.56 to 4.84 and gave s2/s3 back. Every attempt to tune the proxy traded one stratum
for another, because a length preference is the wrong shape for the underlying quantity: at
s3, three specific objects block, and the right candidate removes **those three**. That is a
*count over identified objects*, not a length.

**Decision.** Stop proxying and state it, from data the refiner already reports. `cand_overlap`
gains two columns computed over the culprits observed while failing the candidates already
tried:

```
coverage = |S(c) ∩ culprits| / |culprits|        waste = |S(c) \ culprits| / |S(c)|
```

**These are §5.1's necessity features with `p_i` observed instead of predicted.** Necessity
conditioning was cut on 2026-07-26 because its head would have had to predict per-object
necessity from geometry — an unbudgeted investigation that D2 had shown addressed the wrong
deficit. Once the refiner *reports* culprits, the same two candidate features fall out of the
record with **no head, no second loss, and no geometry routine**. The cut stands; what is
reinstated is the *feature*, on a sounder footing than the version that was cut.

**Why this is legal where `clears` was not (L2/C2).** `clears` was rejected for being a
per-environment geometric routine *we* ran. This is the refiner reporting a collision check
it had already performed — the same legality class as `failure_action`, and precisely what
§6.1 lists `culprits` for. Nothing is inferred by us, which makes it *more* C2-compliant than
a predicted head, not less. It also does not touch C5: the *deduction* (this subset is dead)
still acts only outside the net as demotion; what the net sees is the *observation*.

**Consequences.**
- **Weak dominance over deployed v2.2, the proposal's goal 1.** Over **6 seeds**:
  **7.90 ± 0.61 vs 14.66**, **−6.76 FP, 95% CI [−9.43, −4.40]**. Per stratum (mean ± std
  across seeds): s0 0.00 ± 0.00, s1 5.60 ± 3.06, s2 13.03 ± 1.52, s3 12.96 ± 2.46, against
  the yardstick's 0.00 / 6.20 / 26.00 / 26.44. Nothing regresses; s2 and s3 win by ~2×.
  Deployed config is `--overlap-mode jaccard --coverage-feats --aggregate-records
  --evidence-attn` (preset `v3final`).
- **s1 is a TIE, not a win — and three seeds said otherwise.** 5.60 ± 3.06 vs 6.20 is a
  +0.60 margin against a 3.06 seed sd (0.20 sd), with only **2 of 6 seeds** beating 6.20
  (1.16 / 2.72 / 7.48 / 6.68 / 6.28 / 9.28). At 3 seeds it read 3.79 and looked like a clear
  win. **Consequence for the project's seed discipline:** ≥3 seeds is the stated bar, and on
  the widest-spread stratum 3 was not enough to avoid an over-claim. Where a per-stratum
  claim is load-bearing, check the margin against the seed sd, not just the sign.
- **P-v3-1's number is met, its mechanism is not** — and the bar is cross-collection
  (17.08 was measured on dd2d_v3, ours on dd2d_v4; ~0.08% of labels differ). The bar was
  s2 ≤ 17.08 *via necessity
  conditioning*; s2 lands at 15.88 via observed coverage. Report both halves — the target was
  right, the proposed mechanism was unnecessary.
- **Adaptivity is genuinely record-driven.** Both features are exactly zero until a failure
  is observed, so the first attempt remains purely static (P-D intact) and the signal accrues
  as the rollout proceeds. Nothing else in the system can see which object blocked.
- **A leakage audit is now part of accepting a large effect** — 0 violations here (zero at
  |F|=0; culprits only from candidates in the failure context, all failures; the deploy loop
  breaks on success before a successful candidate could enter the context).
- **Three supporting changes, each individually motivated and individually small**: record
  aggregation to one token per failing *query* (−88.7% tokens), a separate cross-attention
  channel for evidence (the shared softmax made ignoring records loss-minimizing), and
  dropping `dead` from the net while keeping the sound demotion outside it (C5 hygiene).
  Alone, each is roughly a tie with v2.2; the coverage features carry the result.
- **Superseded:** G6's arm levels (censored selector) and G6's "−3.37 record increment",
  which was `cand_overlap` — its bar removed both. **P-v3-3 falsified** (G7): removing
  `cand_overlap` costs −5.07 FP, CI [−8.56, −1.78]; reinstated per R7's escape clause.
- **The result does not depend on the exact combination.** Every coverage-bearing arm beats
  the yardstick significantly (−6.3 to −7.2 FP, all CIs excluding 0); the one arm without
  coverage (rollout-aligned context mass) is a tie at −0.32.
- **The record is consumed two ways and both are load-bearing.** At 6 seeds each, dropping
  the per-failure token stream costs **1.28 FP** (7.90 ± 0.61 → 9.18 ± 1.41) and the loss is
  **entirely at s1** (5.60 → 10.78, i.e. worse than v2.2's 6.20) while s2/s3 tie; tokens also
  halve the overall variance. So: compact features carry s2/s3, tokens carry s1 and
  stability. An earlier 1-seed reading put the token contribution at 0.26 FP and is
  superseded (`autorun_decisions.md` A17).
- **The yardstick is v2.2's best seed, and we report against it anyway.** Two further v2.2
  seeds were trained with its own recipe: per-seed 14.66 / 16.57 / 20.57, mean
  **17.27 ± 3.02**. Against that mean the margin is −9.37 and v3 wins every stratum
  including s1; against seed 0 it is −6.76 with s1 a tie. Reporting the latter is the
  conservative choice and avoids selecting the framing that flatters v3 after seeing both.
  v2.2's s1 spread is ±14.20 — seed 2 lands at 30.04 because `relrank` selected a bad epoch,
  which is the miscalibration R8 replaced.
- **The margin is not the selector.** Every v3 arm *without* coverage ties v2.2 (14.34 /
  14.92 / 15.34 vs 14.66) despite all of them using v3's uncensored deployed-val-FP
  selection. If R8 were carrying the result, those arms would show it.
- **Caveats.** Ablations are 1-seed; the deployed config is 6-seed and the yardstick 3-seed.
  Env-2 remains un-attempted, so generality is architectural (`porting_guide.md`), not
  demonstrated.

---

<a id="2026-07-27-dead-is-a-length-proxy"></a>
## 2026-07-27 — `dead` is a length proxy, so it leaves the representation and stays in the ranking (C5)

<!--strip-->
> **id** `2026-07-27-dead-is-a-length-proxy` · **status** active · **tracks** method ·
> **defines** L4 · **ratifies** `autorun_decisions.md` A3 · **see also** `notebook.md`
> [2026-07-27-g8-dropping-dead-fixes-s1](../notebook/06-v3-performance.md#2026-07-27-g8-dropping-dead-fixes-s1)
<!--/strip-->

Ratified from [`autorun_decisions.md`](../autorun_decisions.md) A3, decided during the
autonomous run of 2026-07-26/27 and adopted here on 2026-07-29. The autorun entry keeps the
full measurement narrative; this records the decision and what follows from it.

**Context.** v3's s1 regressed against the v2.2 yardstick (8.56 vs 6.20) while v3's own
*no-evidence* bar sat at 3.64 — better than either. So the regression was caused by
something v3 added, and the suspect was the `dead` column of `cand_overlap`. Measured on
dd2d_v4 train over 4600 candidate/context pairs:

```
corr(dead, |S|)      = -0.284
mean |S| | dead=1    = 1.38      mean |S| | dead=0 = 2.39
P(feasible | dead=1) = 0.0000    <- the deduction itself is sound
```

**Decision. `dead` is dropped from the net's features while the demotion offset stays
outside it** (`--overlap-mode jaccard`). The proof is sound and keeps acting on the ranking;
what is removed is its use as a *representation*.

**This is a C5 argument, not a tuning knob**, and that distinction is why it is an ADR.
`dead=1` predominantly marks *short* candidates, so as a net input it is a free-running
correlate the scorer fits as "short ⇒ bad" and then applies everywhere — including s1, where
short is *correct*. That is **L4 reappearing as a feature**: the same failure that made
`blocked-at-contents` a prefer-longer cue costing +13.5 FP at s1. C5 says deductions act on
the ranking, not the representation; feeding a proof in as a feature violates that.

**Consequences.**
- s1 goes 8.56 → 4.84 and s2/s3 are given back; this is the change that unblocked the
  performance push that [2026-07-27-necessity-observed-not-predicted]
  (#2026-07-27-necessity-observed-not-predicted) completes.
- **A standing lesson: a proxy is right in the regime it was fitted on and wrong elsewhere.**
  Every attempt to *tune* `dead` traded one stratum for another, because a length preference
  is the wrong shape for the underlying quantity. At s3 three specific objects block and the
  right candidate removes *those three* — a count over identified objects, not a length.
  Give the model the count it is proxying for.
- Safe because G7 measured the offset at only 0.13 FP with overlap on, so the net was not
  relying on the feature to cover for a weak offset.

---

<a id="2026-07-27-record-tokens-are-ignored-at-inference"></a>
## 2026-07-27 — The trained model ignores its record tokens; v3 was matching v2.2, not regressing

<!--strip-->
> **id** `2026-07-27-record-tokens-are-ignored-at-inference` · **status** active ·
> **tracks** method, evaluation · **resolves** G6, A6 · **ratifies**
> `autorun_decisions.md` A8, A11 · **see also** `notebook.md`
> [2026-07-27-p2-missing-g6-cell](../notebook/06-v3-performance.md#2026-07-27-p2-missing-g6-cell)
<!--/strip-->

Ratified from [`autorun_decisions.md`](../autorun_decisions.md) A8 and A11, decided during
the autonomous run of 2026-07-26/27 and adopted here on 2026-07-29.

**Context.** v3 appeared to *underperform* the v2.2 yardstick (16.17 vs 14.66), which was
read as a regression to repair. `suppress_records` — a diagnostic, never a deployment mode —
ran the G6b checkpoint with its evidence memory emptied at every step:

| deploy | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| records ON (as trained) | 16.17 | 0.00 | 8.56 | 22.00 | 34.12 |
| records SUPPRESSED | 16.40 | 0.00 | 8.56 | 22.64 | 34.40 |

**0.23 FP.** The deployed model barely reads its own record tokens.

**Decision — three findings are adopted as the project's reading, replacing earlier ones.**

1. **G6's −3.37 "record increment" was `cand_overlap`, not records.** It agrees with G7's
   independent −5.07 for overlap. The G6 headline was mis-attributed and must not be quoted
   as evidence that record tokens help.
2. **Records are inert at inference but not free in training.** The missing G6 cell
   (`p2_norec`: records off, overlap on) reads **15.34** against 15.80 aggregated and 16.17
   raw — so the token stream *costs* 0.83 FP overall and 3.9 FP at s1 while being ignored at
   deploy. Both are true at once: the stream shapes the weights during training and is
   discarded at inference. **A stream the model learns to ignore is not free.**
3. **v3 matched v2.2; it never underperformed it.** On both splits: v2.2 val 17.30 / test
   14.66, v3 val **17.09** / test 16.17 — v3 is slightly *better* on val, and the test paired
   bootstrap CI was [−2.29, +5.72], including zero. 100-episode splits do not resolve ~1.5 FP.

**Consequences.**
- **The problem was reframed, and that is the decision's real content:** beating v2.2
  required *adding signal*, not repairing a regression. Everything after this — dropping
  `dead`, the second attention channel, observed coverage — followed from that reframing.
- **Do not conflate this with the state-delta tie.** [2026-07-28-state-delta-on-record-ties]
  (#2026-07-28-state-delta-on-record-ties) reports a feature that *was* consumed
  (`delta_proj` trains away from its zero init) and still did not help. This entry reports a
  stream that was **not consumed**. They are different phenomena; conflating them is how the
  A6/A8 reading went wrong the first time.
- Supersedes A6's inference that records were net-harmful by ~1.5 FP: they were not harmful,
  they were inert. The dd2d_v4 fact-inertness finding itself stands.
- Later corrected in magnitude: at 6 seeds the token stream is worth **1.28 FP**, not the
  0.26 first reported — see A17 and [2026-07-27-necessity-observed-not-predicted]
  (#2026-07-27-necessity-observed-not-predicted).

---

<a id="2026-07-27-evidence-needs-its-own-attention-channel"></a>
## 2026-07-27 — Evidence competed with geometry in one softmax; it gets its own cross-attention channel

<!--strip-->
> **id** `2026-07-27-evidence-needs-its-own-attention-channel` · **status** active ·
> **tracks** method · **ratifies** `autorun_decisions.md` A10 · **see also**
> `porting_guide.md` — domain-agnostic, carries to any instrumented env
<!--/strip-->

Ratified from [`autorun_decisions.md`](../autorun_decisions.md) A10, decided during the
autonomous run of 2026-07-26/27 and adopted here on 2026-07-29.

**Context.** [2026-07-27-record-tokens-are-ignored-at-inference]
(#2026-07-27-record-tokens-are-ignored-at-inference) established that the model discards its
record tokens, but not *why*. `CrossAttentionScorer` builds one memory and runs one
cross-attention over it:

```python
memory = torch.cat([scene_tok, glob, fact_tok], dim=1)   # (B, N + 1 + F, D)
attended, _ = self.attn(cand_emb, memory, memory, key_padding_mask=key_pad)
```

With ~10 scene tokens against up to 2045 record tokens, a single softmax divides one fixed
attention budget between the geometry that determines feasibility and the evidence. Geometry
is reliably useful; evidence is noisy. **Discarding evidence is therefore the loss-minimizing
policy**, and the model duly learned it. Aggregation alone does not fix it — it lowers the
ratio to ~3:1, still growing with |F|.

**Decision. Evidence gets its own cross-attention channel** (`CrossAttentionScorerV3`,
`--evidence-attn`): a second, independent attention over the evidence memory, with the head
seeing both attended vectors (`2*D_MODEL → 3*D_MODEL`). Evidence can now be read **without
giving up geometry**, so a useful record no longer has to out-compete the scene to be seen.

**Consequences.**
- **This is an architecture defect, not a fact about evidence** — the framing matters,
  because the same failure set presented as two compact per-candidate scalars
  (`cand_overlap`) is worth 5 FP. Those bypass the attention entirely and concatenate
  straight at the head. Evidence was never useless; the *shape* was wrong.
- **Domain-agnostic**: it changes how tokens are consumed, not what they are, so it carries
  to any environment with an instrumented refiner. Recorded in
  [`porting_guide.md`](../porting_guide.md) for that reason.
- Ships in the deployed `v3final` preset.
- **Implementation trap worth carrying:** a batch row with *no* records yields an all-True
  key-padding mask, and `nn.MultiheadAttention` returns **NaN** for such a row rather than an
  empty result. Guarded by attending under a mask that always leaves one key live and zeroing
  those rows afterwards — the same guard `model.py` already uses. Verified NaN-free with and
  without records.

---

<a id="2026-07-27-margin-must-be-compared-to-seed-sd"></a>
## 2026-07-27 — A per-stratum margin is compared to the seed sd, not just to the baseline; s1 is a tie

<!--strip-->
> **id** `2026-07-27-margin-must-be-compared-to-seed-sd` · **status** active ·
> **tracks** evaluation, process · **ratifies** `autorun_decisions.md` A16 · **see
> also** `as_built_v3.md` §7.1
<!--/strip-->

Ratified from [`autorun_decisions.md`](../autorun_decisions.md) A16, measured during the
autonomous run of 2026-07-26/27 and adopted here on 2026-07-29. This one changes a
**convention**, which is why it is an ADR rather than a notebook entry.

**Context.** The deployed v3 arm was written up at three seeds, where s1 read 3.79 ± 3.29 and
looked like a clear win over the v2.2 yardstick's 6.20. Running three more seeds:

| seed | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| 0 | 7.50 | 0.00 | 1.16 | 15.80 | 13.04 |
| 1 | 7.63 | 0.00 | 2.72 | 12.24 | 15.56 |
| 2 | 7.19 | 0.00 | 7.48 | 12.96 | 8.32 |
| 3 | 8.05 | 0.00 | 6.68 | 11.24 | 14.28 |
| 4 | 8.08 | 0.00 | 6.28 | 12.88 | 13.16 |
| 5 | 8.94 | 0.00 | 9.28 | 13.08 | 13.40 |
| **mean ± sd** | **7.90 ± 0.61** | 0.00 ± 0.00 | **5.60 ± 3.06** | **13.03 ± 1.52** | **12.96 ± 2.46** |
| *v2.2* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* |

**s1 is a tie, not a win.** A +0.60 margin against a 3.06 seed sd is 0.20 sd, and only **2 of
6 seeds** beat 6.20. s2 and s3 are genuine ~2× wins and stable (sd 1.52, 2.46).

**Decision — the reporting convention gains a second test.** "≥3 seeds to report" stays the
bar, but where a **per-stratum claim is load-bearing, the margin is compared to the seed sd,
not merely to the baseline**. A sign is not a result.

**Consequences.**
- **The stated bar was met and the claim was still wrong**, which is the point: three seeds
  was not unlucky, it was that nobody checked the margin-to-spread ratio (0.60 against 3.06).
  A rule that only counts seeds cannot catch this.
- s1 is reported as a **tie** wherever the per-stratum table appears.
- **What did *not* move is equally instructive:** overall FP was 7.44 ± 0.23 at three seeds
  and 7.90 ± 0.61 at six. The headline was never in doubt — only the per-stratum breakdown.
  So this is an argument for more seeds *on wide-spread strata*, not everywhere.
- s1 is now flagged in [`CLAUDE.md`](../../CLAUDE.md) as the stratum where a small-seed report
  is least trustworthy.

---

