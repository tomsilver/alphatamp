# SPECTRE Decisions — StickButton2D as a second environment

3 entries, 2026-08-01 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

---
<a id="2026-08-02-per-candidate-refinement-cap-deployed-wall-clock-configuration"></a>
## 2026-08-02 — Per-candidate refinement cap is the deployed wall-clock configuration

<!--strip-->
> **id** `2026-08-02-per-candidate-refinement-cap-deployed-wall-clock-configuration` ·
> **status** active · **tracks** method, evaluation, env-dd2d
<!--/strip-->

**Context.** The §2b DD2D wall-clock table showed SPECTREv3-adaptive *slower* overall than
the naive planner order (5.89 vs 4.94 s ALL to first success), with the entire gap at s1
(11.99 ± 7.81 vs astar 0.26 s). The diagnosis
([notebook/07 2026-08-02](../notebook/07-stickbutton2d.md#2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate))
is real, not a measurement bug: v3's s1 FP (3.44) is modestly *worse* than astar's (2.24) —
the planner-cost order already ranks s1's short/cheap feasible plans well — and that ~1.2-
attempt FP gap becomes a ~46× wall-clock gap because of *which* candidates each method fails
on. Feasible refinements finish in <0.5 s (p95 0.44 s); the waste is entirely near-feasible
infeasible candidates that burn the full **20 s** refinement budget. astar's s1 failures are
cheap dead-ends (~0.06 s); v3's few extra failures are the 20 s traps. This is "FP flatters
the learned ranker" running against v3 — FP alone hides it.

**Decision.** The **deployed wall-clock configuration is a per-candidate refinement-
abandonment cap** `REFINE_CAP_S = 2 s`: each skeleton is refined for at most C seconds before
the deployment moves to the next in the ranked order; a candidate not refined within C is
abandoned and treated as a failure. Load-bearing choices:

- **Per-candidate, never per-problem.** A per-problem total budget can starve a solvable
  problem (spend it on traps, never reach the feasible skeleton); a per-candidate cap only
  skips the slow *skeleton*. A problem is lost only if *every* feasible candidate exceeds C —
  measured **0/100** on dd2d_v4 (min-feasible refine time per problem: mean 0.103 s, **max
  0.243 s**). Precompute logs this at-risk count (`_feasibility_at_risk`) so a future
  collection where it is non-zero is caught, not silently censored. Provably lossless with an
  iterative-deepening fallback (exhaust the pool at C; if nothing refines, retry uncapped) for
  any domain with slow-feasible plans; it never fires on dd2d_v4.
- **C = 2 s** ≈ 4.5× the feasible p95, so only genuine near-feasible outliers are cut.
- **The cap faithfully shifts FP, so it is re-run, not accounted.** A slow-feasible candidate
  ranked first is abandoned (FP + 1), and for the *adaptive* rollout it enters the failure
  context and re-ranks the rest — so the order diverges (0/100 astar, 6/300 piginet, 4/300
  static, 6/300 adaptive at C=2 s). `deployed_rollout_v3_traced(refine_cap_s=…)` redefines the
  stopping-success set as `outcome==success and time ≤ C` and re-runs; the fixed-order methods
  derive capped FP/refine on their score-order (`_fp_and_refine_capped`). `min(t, C)` on the
  uncapped stored sums would be silently optimistic on exactly those cells.
- **The published FP headline (§1/§2) stays uncapped** at the pool-cap budget — the metric of
  ranking quality is unchanged. The cap is a *wall-clock* deployment configuration; §2b owns
  it and prints the capped-FP delta beside the table.
- The cap applies to **all four pool-ranking methods** (astar-dist, PIGINet, SPECTREv3-static,
  SPECTREv3-adaptive) — a shared-refiner policy, so fairness requires it. It is a test-time
  accounting change: **no retraining**, checkpoints reused as-is.

**Consequences.** Under the 2 s cap, SPECTREv3-adaptive is the **fastest** method — 1.79 ± 0.44 s
ALL vs astar 2.96, PIGINet 3.14, v3-static 2.53 — its s1 collapses 11.99 → 2.40 and it wins s2
(1.88) and s3 (2.45) decisively. The cap's **FP cost is tiny**: adaptive +0.05 (5.78 → 5.83),
astar +0.00 (failures already sub-cap), PIGINet +0.23, static +0.26 — while cutting adaptive's
wall-clock 3.3×. This is the honest resolution of "FP flatters the learned ranker": the ranker's
value (try few candidates) shows in wall-clock only once each failed try is bounded, because the
cap targets exactly the expensive failures the ranker still makes. **DD2D-only** (SB2D's kinder
`BacktrackingRefiner` records no per-candidate times; `EnvSpec.has_timing` gates the section).
Reproduce with `precompute_dd2d_cache.py --env-variant dd2d_v4 --force` (writes
`refine_s_capped`/`fp_capped` per record + `refine_cap_s` in `meta.json`) then read §2b. New
code: `REFINE_CAP_S` + `_fp_and_refine_capped` + `_feasibility_at_risk` in
`precompute_dd2d_cache.py`; `refine_cap_s` + `V3Trace.refine_capped_seconds` in `inference_v3.py`;
`load_refine_cap_s` + `build_time_table(use_capped=…)` in `compare.py`; `test_refine_cap.py`.
The **residual s1 gap** (v3 2.40 vs astar 0.26) is the modest s1 FP deficit and is a candidate
for the model-side R1 cost/enumeration-index feature.

---

<a id="2026-08-02-kinder-rendered-piginet-crops-stickbutton2d-via-new"></a>
## 2026-08-02 — Kinder-rendered PIGINet crops for StickButton2D via a new env_variant

<!--strip-->
> **id** `2026-08-02-kinder-rendered-piginet-crops-stickbutton2d-via-new` · **status**
> active · **tracks** baselines, evaluation, env-stickbutton2d, data, tooling
<!--/strip-->

**Context.** For the representation contrast to be fair, the pixel input a *model* consumes
should come from the environment's own renderer, not an approximation. On SB2D the only
model reading pixels is **PIGINet** (SPECTRE is image-free: its `SceneEncoder` consumes
vector `scene_geometry` — boundary polygons + poses — read from kinder's
`object_to_multibody2d`, so it is already kinder-native). PIGINet's SB2D crops, though, were
produced by a **schematic** rasteriser (`SB2DDomain.crops`): each object drawn as a lone
polygon on a blank background, with no scene context. DD2D is unaffected — it is not a
kinder env and already renders PIGINet crops from its own env renderer. This is SB2D-only.

**Decision.** Route PIGINet's SB2D pixels through **kinder's built-in renderer**, delivered
as a new env_variant **`stickbutton2d_v1_kinder`** built by *converting*
`stickbutton2d_v1`, not re-collecting it. Five choices are load-bearing:
- **Reconstruct, never regenerate — with the sanctioned exception.** The converter
  (`experiments/spectre/sb2d_render_convert.py`) copies every record **verbatim** (plans,
  timings, outcomes, geometry) and only re-renders the pixels, by resetting the env from the
  stored seed (`env.reset(seed=problem_id)`). That reset is the one sanctioned exception to
  the rule (it is deterministic on SB2D; the same reset backs `vlmplan/sb2d_label.py`). Only
  `provenance.env_variant` changes in the record.
- **Per-object crops from the true scene, not a whole-scene embedding.** Each crop is a
  native `render_2dstate` window (world side = the adapter's `_CROP_WORLD`) centred on the
  stored object pose, so it keeps PIGINet's per-object CLIP channel *and* now carries real
  local context (neighbours, stick, table band, wall) that the schematic discarded. A full
  `scene.png` is materialised alongside for possible future use (no consumer wired).
- **No schema change.** Crops live at `raw/<variant>/<split>/images/<pid>/<obj>.png`, a path
  the reader reconstructs from the pid — so `EpisodeRecord` gains no image field and needs no
  migration shim. The reader is a thin `SB2DKinderDomain(SB2DDomain)` overriding only
  `crops()`; `make_sb2d_domain(data_root, variant)` dispatches on variant, keeping the
  schematic as the documented secondary.
- **SPECTRE is grafted, not retrained.** Because the records are byte-identical and SPECTRE
  is image-free, its numbers cannot differ; the comparison notebook grafts SPECTRE (and
  VLMPlan) from `stickbutton2d_v1` via `EnvSpec.legacy_only`, and only PIGINet (+ the cheap
  deterministic astar) is native to the kinder cache. Retraining would add training noise,
  not signal.
- **Kinder does not manufacture signal it cannot have.** Two unpressed buttons are identical
  red discs in the real env too, so the image channel stays partly degenerate; the win, if
  any, is the positional context the crop now carries, not disc appearance.

**Consequences.** The seam turned out to be one function — `domain.crops` — so model, loss,
tokenizer and CLIP cache are untouched; the change is a converter + a subclass + a variant
row. **The re-run reinforced the standing finding rather than overturning it.** PIGINet
retrained on kinder crops (3 seeds, same weighted-bce/40-epoch recipe) reads **2.28 ± 0.29 FP
ALL** — *slightly worse* than the schematic's 2.02, the whole drop at b5 (7.55 vs 6.39). The
paired bootstrap still does not separate: v3-static − PIGINet = −0.31, CI [−0.95, +0.36];
v3-adaptive − PIGINet = −0.60, CI [−1.24, +0.08]; the adaptive increment holds (−0.29, CI
[−0.51, −0.08]). So "the representation advantage does not reproduce on SB2D" survives the
validity fix, and the pre-registered caveat held — the crop's added context is positional and,
since unpressed buttons are identical discs in the real env, net-neutral-to-mild-distractor.
Full numbers in [notebook/07 2026-08-02](../notebook/07-stickbutton2d.md#2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s).
The schematic `stickbutton2d_v1` stays as the secondary/baseline, so the two are never
silently mixed. One kinder-internal coupling was accepted
(`env.unwrapped._object_centric_env._current_state`, mirroring `base_env.render()`), with a
public fallback and a determinism test guarding it.

---

<a id="2026-08-02-wall-clock-to-first-success-added-compare-methods-reuses-stored"></a>
## 2026-08-02 — Wall-clock-to-first-success added to compare_methods; reuses stored refine times

<!--strip-->
> **id** `2026-08-02-wall-clock-to-first-success-added-compare-methods-reuses-stored`
> · **status** active · **tracks** evaluation, tooling, env-dd2d
<!--/strip-->

**Context.** `compare_methods.py` reported only FP (failed attempts before first success). FP
treats every failed attempt as equal cost, but a DD2D failed refinement ranges ~15 ms (a dead-end)
to ~20 s (budget-exhausted), so FP cannot say whether a method's inference cost is *worth it* in
wall-clock. We added a wall-clock-to-first-success metric = abstract-plan-generation + inference +
refinement.

**Decision.** A new **complementary** metric (FP stays the headline), computed so the cross-method
comparison is fair and the result is durable:
- **Refinement time is reused, not re-run.** The dd2d_v3/v4 refiner stores per-candidate
  `refinement_wall_clock_s`; each method's refine-to-first-success is that summed along its own
  attempt order (adaptive = the cached `order`; static = `argsort(-scores)`). Every method sums the
  *same* per-candidate times over its own ordered subset, so the comparison is fair even though the
  absolute seconds are a within-collection relative measure (collector 8-way parallelism, 20 s
  budget).
- **Inference is measured on GPU** (the deployment-realistic path; `~22 ms/step`, CPU-tensorize +
  GPU-forward, tensorization-dominated), via an `infer_seconds` field on `V3Trace`.
- **Plan-gen is a per-stratum shared constant** (identical pool for all four pool-ranking methods),
  measured by regenerating a few problems per stratum and timing the astar top-k enumeration.
- **All three are persisted** in the compare cache (`refine_s`/`infer_s` per record; per-stratum
  `plan_gen_s` in `meta.json`) — measured once at `--force` cache build, reused at render, never
  recomputed. Scope: the four pool-ranking methods (astar-dist, PIGINet, SPECTREv3-static/adaptive)
  on DD2D; gated by `EnvSpec.has_timing` (SB2D's refiner stores no per-candidate times). The FP
  table is byte-identical after the rebuild (timing fields are additive; scores/FP deterministic).

**Consequences.** The headline finding is that **FP flatters the learned ranker**: SPECTREv3-adaptive
has 6× lower FP than astar (5.8 vs 34.5) but is not faster in wall-clock (5.90 vs 4.94 s ALL),
because astar's many failures are cheap dead-ends (~0.14 s) while SPECTRE's few failures are the
expensive *near-feasible* candidates it correctly ranks high (~0.89 s) — a better ranking surfaces
the costlier failures. Inference is the small term (0.03–0.51 s); the learned ranker's wall-clock
advantage is concentrated at s3 (astar's failure *volume*) and is net-negative at s1/s2. Numbers +
per-stratum breakdown + noise caveats in [notebook/07
2026-08-02](../notebook/07-stickbutton2d.md#2026-08-02-dd2d-wall-clock-first-success-fp-flatters).
**Standing implication:** an FP margin on DD2D should not be read as a proportional wall-clock win;
quote the wall-clock section alongside it.

---

<a id="2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact"></a>
## 2026-08-02 — s2 generalization degradation characterized as pool-composition artifact; regen for pair-diversity rejected

<!--strip-->
> **id**
> `2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact` ·
> **status** active · **tracks** env-dd2d, evaluation, method, data
<!--/strip-->

**Context.** The [2026-08-01 generalization test](#2026-08-01-dd2d-generalization-test-unseen-count-unseen)
reported v3's s2 FP degrading 10.49 → 30.23 under the unseen-count shift, framed in that entry's
consequences as v3's "already characterized in-distribution s2 weakness." An objection — s2 (clear
2) cannot be intrinsically harder than s3 (clear 3) — prompted a read-only diagnosis
([notebook/07 2026-08-02](../notebook/07-stickbutton2d.md#2026-08-02-s2-ood-degradation-pool-composition-artifact-model)).
The objection is correct: intrinsic/execution difficulty is monotone (astar-dist FP s3 167 ≫ s2
28; generation keep-rate s3 20% ≪ s2 91%; s2 labels 100% sound). Only the *model's* FP inverts,
and it does so for a reason that is neither a model-generalization failure nor a generator bug.

**Decision.** **Root cause = a pool-composition artifact sitting on top of low s2 solution
diversity; characterize it, do not re-engineer.**
- s2 problems have only **~1.5 unique feasible solutions** (feasible pairs). 99% of feasible
  triples are redundant supersets of those pairs (genuine-3 ≈ 0). The circular target admits 18
  diametric grasp axes; an axis opens only when its antipodal blocker pair is cleared, and
  `crowd=5` (odd) yields no antipodal pair.
- In-distribution, the k=200 pool pads those ~1.5 solutions with ~23 redundant feasible triples
  (92 triples enumerated) → 26 feasible → the ranker finds one in ~3 tries. At 14 blockers,
  C(14,2)=91 pairs flood the short-first cap (→172 pair candidates) and crowd the triples out
  (→18 enumerated, 1.1 feasible) → ~2.9 feasible → FP ~30. So the OOD number exposes the true
  low-diversity difficulty that pool padding hid in-distribution (model FP corr(feasible count)
  = −0.82).
- **A generator redesign for substantive feasible-pair diversity was explored and rejected as
  geometrically blocked.** The obvious lever — even collar count so antipodal pairs each open an
  axis — does not work empirically (generator sweep: crowd 5/6/8/10 → ~1.5 feasible pairs) and
  pushes problems to mfs=3: keeping mfs≥2 requires blocking the circular target from all 18 axes,
  which is exactly the coverage that prevents a single removed pair from cleanly opening one axis.
  Any real regen would also imply re-collecting train/val/test + retraining, re-baselining every
  existing SPECTRE result — a large cost against an uncertain geometric payoff.

**Consequences.** The s2 column of the generalization table — and the ALL mean it dominates — is
**confounded by pool composition, not a clean model-generalization signal**, and is recorded as
such (this entry, the notebook entry, the `CLAUDE.md` DD2D-generalization section, and
`proposal.md` §6). The **s3 column is the clean signal**: s3 was already feasible-scarce in
training, so OOD s3 is in-regime and v3 improves there (9.19 → 4.87) while astar stays pathological
— i.e. v3's advantage over the planner order does generalize where the feasible regime is stable.
This entry **refines** the s2 interpretation in the
[2026-08-01 generalization ADR](#2026-08-01-dd2d-generalization-test-unseen-count-unseen) (which
attributed s2 to model weakness); the numbers there are unchanged, the attribution is corrected
here. No code or data changed.

---

<a id="2026-08-01-dd2d-generalization-test-unseen-count-unseen"></a>
## 2026-08-01 — DD2D generalization test — unseen count and unseen shapes

<!--strip-->
> **id** `2026-08-01-dd2d-generalization-test-unseen-count-unseen` · **status** active
> · **tracks** env-dd2d, method, evaluation, data
<!--/strip-->

**Context.** The dd2d_v4-trained SPECTRE v3 checkpoint had only ever been evaluated
in-distribution (9–12 blockers, the base 7 shape families). The proposal's §6 object-count /
compositional-generalization question and §0 wishlist property #4 were *asserted, never
tested*. We wanted a direct OOD test on DD2D along two axes the model never saw: **more
blockers** and **novel shape figures**, scored train-old / test-new against the existing
checkpoint (no retraining).

**Decision.** Three sub-decisions, each load-bearing.

1. **New shapes ride the geometry-general grasp model — no per-shape code.** `dd2d/grasps.py`
   derives both the global-envelope grasp and the internal/concave grasp purely from
   `shape.polygon` (supporting-line contact runs + a scan-line antipodal search), with no
   branch on family anywhere. So a `tee` (bar+stem) and a `cross` (symmetric plus), both
   **concave**, were added to `dd2d/shapes.py` alone (`_build` + `_CONCAVE_FAMILIES`; kept OUT
   of `_FAMILY_WEIGHTS` so the base sampler never draws them, and sized to the finger/aperture
   constants like `horseshoe`). Verified: 0 floating grasps over 30 seeds each, and the real
   refiner certifies scenes containing them at collection — the grasp model carries over to
   the new shapes and their concave regions, exactly as hypothesised.

2. **Held-out collection = fresh band + unseen count with a *realized-count floor* + forced
   families.** Two test-only sets, 40 problems each, stratified s0–s3 (10 each):
   `dd2d_v4gen_count` (14–16 items = 13–15 blockers, old shapes; isolates count) and
   `dd2d_v4gen_shape` (same count + tee/cross in the pool with **≥1 of each forced** per
   scene). New collector flags (all default-preserving): `--seed-band-base` (base 3 = `[3M,4M)`
   for count, base 4 = `[4M,5M)` for shape — disjoint from train/val/test, `--band=1_000_000`
   kept so `compare.stratum_of` stays valid), `--n-items-min/max`, `--shape-set augmented`,
   `--require-families`, `--fill-max`. The **realized-count floor** was the non-obvious
   necessity: a fill-cap sweep showed 12–22% of scenes truncate below 14 items even at
   `fill_max=0.85` (a small sampled drawer can't fit 15), and such a scene falls *back into the
   seen range* — silently defeating the test. Cranking `fill_max` never closes the tail, so the
   generator now rejects and resamples any scene realizing fewer than `min_items`, which
   *guarantees* every kept problem is genuinely unseen-count. `fill_max=0.72` keeps the
   resample rate low.

3. **Score train-old / test-new via `--test-variant`, reusing the dd2d_v4 vocab.**
   `spectre_score_v3.py`'s new `--test-variant` overrides only the episode dir; vocab, model
   config and checkpoints stay from `--env-variant`. Valid with **no OOV and no retraining**
   because the DD2D vocab / `config_hash` are over the fixed operator/predicate/type sets only
   — a shape family is geometry metadata, not a vocab token, and more blockers only add generic
   objects handled by positional local-ids. The domain spec is shared across `dd2d_*` variants
   (registered in `domain.DOMAINS`) and stratum recovery is pid arithmetic. `--astar-baseline`
   computes astar-dist (default order, score = −plan_idx) off each episode's stored outcomes via
   the shared `rollout_fp`, so v3-vs-astar is one instrument, uncensored, paired bootstrap.

**Consequences.** The scoring ran clean (no OOV, no position-index error on the longer
skeletons from denser scenes) — confirming the count/shape invariance and that the position
encoding tolerates the longer plans. In-distribution v3 reproduced **5.78 ± 0.10** exactly,
validating the instrument. Result (v3 ALL FP, 3 seeds; paired vs astar-dist):

| set | v3 ALL | vs astar | s2 | s3 |
|---|---|---|---|---|
| in-dist `dd2d_v4` (n=100) | 5.78 ± 0.10 | −28.74 [−39.6,−18.8] | 10.49 | 9.19 |
| unseen count (n=40) | 9.40 ± 2.62 | −39.95 [−64.0,−18.1] | 30.23 | 4.87 |
| unseen count+shape (n=40) | 11.26 ± 3.44 | −21.89 [−42.6,−3.8] | 31.97 | 10.67 |

**v3 still wins overall on both held-out sets (CI excludes 0), so its advantage over the naive
planner order survives OOD** — but absolute FP degrades ~1.6–1.9× (5.78 → 9.40 → 11.26), and
the honest stratum reading is that **the win is carried by s3** (astar's default order is
pathological there, 108–167 FP), while **at s2 v3's advantage collapses under the count shift**
(30.23 vs astar 28.30; 31.97 vs 22.00 — within the ±9 seed spread), amplifying v3's already
characterized in-distribution s2 weakness. *(⚠️ s2 root cause refined 2026-08-02: this collapse is
dominantly a **pool-composition artifact** — the k=200 pool crowds out the redundant feasible
triples that padded s2 in-distribution — not model weakness; see
[2026-08-02](#2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact).)*
The shape set is harder than count-only, as expected. Numbers and caveats live in [notebook 07](../notebook/07-stickbutton2d.md)
2026-08-01. The held-out raw dirs are archived and authoritative (DD2D generation is
PYTHONHASHSEED-dependent, so a re-run yields a fresh sample).

---

<a id="2026-08-01-vlmplan-stops-generating-first-feasible-plan"></a>
## 2026-08-01 — VLMPlan stops generating at the first feasible plan

<!--strip-->
> **id** `2026-08-01-vlmplan-stops-generating-first-feasible-plan` · **status** active
> · **tracks** baselines, evaluation
<!--/strip-->

**Context.** VLMPlan's generation loop ran until it stalled or hit its round cap, then
scoring walked the proposals to the first success. The 200-plan budget was read as a
target to approach; it is a **hard ceiling for the case where proposals keep failing**,
which is a different thing.

The two stages are deliberately split — only generation needs a model, so a re-score is
cheap ([2026-07-24](04-comparison.md#2026-07-24-vlmplan-baseline-protocol)) — and the
side effect was that generation had no labels and therefore no way to know it was done.
It kept proposing after the answer had already been found.

The cost became visible once the b5 grounding bug was fixed
([2026-08-01](07-stickbutton2d.md#2026-08-01-off-pool-proposals-grounded-against-domain-filtered)):
b5 problems went from stalling out at 0 plans to running all 10 rounds for 27 plans at
~884 s each, pushing the 100-problem run from ~9 h to ~14 h — to generate proposals the
scorer would never reach.

**Decision.** `generate_sequence` takes a `stop_check`, called after each round, and
stops at the first proposal known to refine. The runner supplies it; `max_plans` remains
the ceiling for the all-failing case.

**FP is unchanged, and that is the whole argument.** The metric is failures before the
first success, so the rollout never looks past that success — proposals after it are
wall-clock and nothing else. Pinned by
`test_vlmplan.py::test_stop_at_first_success_preserves_fp`.

Labels come from `label_step_sequence`, newly extracted as the **single** definition of
how a proposal is labelled (stored outcome if it matches a pooled candidate, live refine
otherwise) and now called by both the scorer and the stop check. Two copies would drift,
and the symptom would be a run stopping on a "success" the scorer then calls a failure.
They share the on-disk memo, so the refinement work is *moved earlier*, not duplicated,
and `vlmplan_score.py` still runs standalone.

**Consequences.**

- **`n_proposed` changes meaning, and §6 of the comparison notebook reports it.** With a
  stop check the count is censored at the first success — "plans needed", not "plans the
  model can produce". **The DD2D rows were generated without it, so that column is not
  comparable across the two environments.** FP is. `stop_at_first_success: false`
  reproduces the old behaviour, and `stopped_on_success` is recorded per problem so a
  short proposal list is never mistaken for a model that ran out of ideas.
- Wall-clock on the SB2D test run drops by roughly the margin above; the exact saving
  depends on how early the first success lands, which is itself the thing being measured.
- The stop check is *conservative by construction*: it can only fire on a plan the
  scorer would also label a success, because it is the same function.

---

<a id="2026-08-01-off-pool-proposals-grounded-against-domain-filtered"></a>
## 2026-08-01 — Off-pool proposals are grounded against the domain, not the filtered pool

<!--strip-->
> **id** `2026-08-01-off-pool-proposals-grounded-against-domain-filtered` · **status**
> active · **tracks** baselines, method, env-stickbutton2d
<!--/strip-->

**Context.** VLMPlan's protocol says a proposal is held to *exactly* the standard a
planner-emitted skeleton meets — no more, no less
([2026-07-24](04-comparison.md#2026-07-24-vlmplan-baseline-protocol)). The SB2D adapter
implemented "exactly" by recovering lifted operators from the episode's own
`skeleton_pool`, which needs no environment and keeps operator identity aligned with
`pool_index`.

That is stricter than intended, and the gap is not hypothetical. The acyclic pool filter
([2026-08-01](07-stickbutton2d.md#2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1))
drops every skeleton containing a `PickStick`/`PlaceStick` cycle, so on b5 **no pooled
plan mentions `PlaceStick`** — while the domain has it and the prompt advertises it. Any
proposal using it died on a `KeyError` and was recorded as inapplicable.

Compounding it, the chaining rule the prompt stated was **false for mixed plans**. It
said "the first press is `...FromNothing`, every later press is `...FromButton`", which
holds only within one uninterrupted run of presses by the same effector. `PlaceStick` and
`PickStickFromButton` both re-add `(AboveNoButton)`; arm presses track `RobotAboveButton`
while stick presses track `StickAboveButton`, so the two never chain into each other.

Together these made the *entire* stick-then-arm strategy unrepresentable — the strategy
the model writes down unprompted ("we must place stick first to use bare arm"). Both b5
pilots returned **0 usable plans**; b5 problem 750000 round 0 was 21 blocks, 19 parsed,
**19 inapplicable**.

**Decision.** Ground against the **domain**, not the filtered pool: `_lifted_by_name`
takes the pool's operators first (env-free, identity-preserving for `pool_index`) and
fills any missing ones from kinder's own `create_bilevel_planning_models`. And correct
the prompt's chaining rule to state effector separation and the two reset actions.

The general rule, which is the part worth carrying to the next environment: **a
pool-generation heuristic is not a legality constraint.** The acyclic filter exists to
stop the pool filling with padding; an off-pool proposal is refined for real and must be
judged against what the domain permits.

**Consequences.**

- Pinned by `test_vlmplan_sb2d.py`: a `PlaceStick` plan grounds, the mixed
  stick→place→arm plan grounds, and — the guard that keeps the other two honest —
  `...FromButton` immediately after `PlaceStick` is still **rejected**, so the tests
  cannot pass on an adapter that simply stopped checking preconditions.
- **The full 100-problem test run was stopped ~8 problems in and restarted.** Its b5
  column would have been near-entirely published-order fallback, and b5 is one of the two
  strata that carry the SB2D result — the other 92 problems were not worth the ~9 hours
  to produce a column known in advance to be an artifact.
- **A wrong disclosure is worse than none.** Deviation 7/8's whole justification is that
  stating a precondition *removes a handicap* every other method gets from the domain for
  free. That argument only holds if the statement is true; the model obeys it either way.
  The corrected note is in `prompts/PROVENANCE.md` deviation 8, with the old text and why
  it was wrong.
- **An unset LLM endpoint is now a hard error.** During the re-pilot the
  `OPENAI_BASE_URL` export was missing and the OpenAI SDK silently fell back to
  `api.openai.com`; 5 requests went to the public API and were rejected 401, and nothing
  was processed only because no valid key was present. A machine with one would have
  completed the run off-box and billed for it. `make_model` now refuses an unconfigured
  endpoint and names the fix, `SPECTRE_VLMPLAN_ALLOW_REMOTE=1` is the deliberate opt-in,
  and `vlmplan_sb2d_32b.yaml` states `base_url` rather than relying on an export.

---

<a id="2026-08-01-comparison-notebook-parameterised-env-registry"></a>
## 2026-08-01 — Comparison notebook parameterised by an env registry

<!--strip-->
> **id** `2026-08-01-comparison-notebook-parameterised-env-registry` · **status**
> active · **tracks** tooling, evaluation
<!--/strip-->

**Context.** `compare_dd2d_methods.py` is where every method comparison is read. Standing
StickButton2D up alongside DD2D was done the obvious way first — copy the file to
`compare_sb2d_methods.py` and edit the constants — which produces two 1400-line notebooks
that share six sections of analysis and drift apart on the first fix applied to one of
them. The project already has a rule against exactly this shape of duplication for
environments (`domain.DomainSpec`, `piginet.PIGINetDomain`, `vlmplan.EnvAdapter`); the
notebook was the last place still forking.

The DD2D assumptions were not concentrated anywhere. They were a hardcoded `env_variant`,
a `primary_name="dd2d_v4"` string tagging every loaded row, strata whose labels mean
min-feasible-subset size, a method list carrying two SPECTRE-v1 rows, `dd2d_*.csv` export
names, a scene renderer imported from `envs/dd2d`, an `n=100` in a chart title, and an
`f"s{k}"` stratum label formatter — a dozen small places, each individually too minor to
notice.

**Decision.** Three files replace the fork:

- `spectre/compare.py` — `dd2d_compare.py` renamed (139 references across 14 files).
- `spectre/compare_envs.py` — the registry: one `EnvSpec` per environment carrying
  variant, legacy graft, stratum labels, axis label, which sections apply, an optional
  scene renderer, and its caveats. **A third environment is one entry.**
- `experiments/spectre/compare_methods.py` — the single notebook, environment chosen by
  an `mo.ui.dropdown`.

`stratum_labels` is the important field. SB2D's button count is recovered by DD2D's
seed-band arithmetic *only because the problem ids were chosen to make that true* — a
coincidence that was implicit in a formula named for DD2D seeds and is now a declaration.

**Caveats live in the registry, beside the number.** `EnvSpec.caveats` renders under §1's
summary table. A reader quoting a figure sees what bounds it in the same view, rather than
in a document they would have to know to open.

**Consequences.**

- **Verified by rendering both environments, not by inspection.** The notebook takes
  `SPECTRE_COMPARE_ENV` for its initial selection specifically so marimo's script mode can
  execute it headlessly for *every* registry entry — otherwise only the entry that sorts
  first is ever smoke-tested. DD2D re-renders unchanged after the rename (7.44 / 17.27 /
  17.27 / 20.66 / 20.86 / 23.55 / 29.86 / 34.52), which is the check that mattered: 139
  mechanical edits is exactly where a silent mis-edit hides.
- Three bugs the fork would have kept: the `collection` column labelled SB2D rows
  `dd2d_v4`; the CSV export wrote `dd2d_method_*.csv` for both environments, so rendering
  the second silently overwrote the first; and §4.3 crashed on an empty frame instead of
  reporting that demotion arms are inapplicable.
- **§4.3 is inapplicable rather than missing on SB2D**, and now says so. Proof-tier
  demotion needs provable query axioms; SB2D resolves to `EMPTY_SPEC`, so the demotion-on
  and demotion-off caches would be bit-identical. Rendering that as an ablation with a
  0.00 Δ would be the worst outcome — a measurement of nothing that looks like a
  measurement.
- Deleted: `compare_dd2d_methods.py`, `compare_sb2d_methods.py`.

---

<a id="2026-08-01-vlmplan-made-env-agnostic-via-labeler-protocol"></a>
## 2026-08-01 — VLMPlan made env-agnostic via a Labeler protocol

<!--strip-->
> **id** `2026-08-01-vlmplan-made-env-agnostic-via-labeler-protocol` · **status**
> active · **tracks** baselines, tooling, env-stickbutton2d
<!--/strip-->

**Context.** VLMPlan is the **zero-training-data corner** of the data × perception grid
([`proposal.md`](../proposal.md) §0), so a second environment without it is a grid missing
a column, not merely a missing row.

`vlmplan/score.py` was already env-agnostic in the parts that matter — budget accounting,
the published-order fill, `label_agreement` — but it reached past its own abstraction in
one place: it imported `DD2DRefiner`, `staging_skeleton` and `reconstruct_scene` directly,
and `REFINER_PRESETS` was keyed by DD2D variant. That import is what makes an *off-pool*
proposal refinable at all, and it is precisely the thing that differs per environment.

The setting of that refiner is not a detail. VLMPlan's score mixes labels from two
sources: stored labels for proposals that match a pool candidate, and live refinement for
the ones that do not. If the live refiner runs at different settings than the collection
did, the two halves of the same row are drawn from different distributions — off-pool
proposals get systematically easier or harder labels than in-pool ones, and the arm's
number moves for a reason that has nothing to do with the model.

**Decision.** Introduce a **`Labeler`** ABC (`vlmplan/adapter.py`) — *given an episode and
a proposed step sequence, return feasible/infeasible* — with `n_refines` and `flush()`.
`score_sequence` and `label_agreement` take one as a parameter. DD2D's implementation
wraps `DD2DRefiner`; SB2D's (`vlmplan/sb2d_label.py`) wraps kinder's `BacktrackingRefiner`
**at the collection's own settings** — `num_sampling_attempts_per_step=5`,
`refinement_timeout_s=20`, `max_trajectory_steps=200` — using the collection's
per-candidate seed rule. `vlmplan/registry.py` dispatches both adapter and labeler on
`env_variant`.

Memoization moved up into a shared `MemoizingLabeler` base keyed on the canonical step
tuple, so both environments get it and neither implements it.

**Consequences.**

- The **label-agreement gate is now the acceptance test for a new environment's labeler,
  not a diagnostic printed after the fact.** SB2D reads **1.000** (35 samples), against
  DD2D's 0.982. It earned that status by catching three real bugs during bring-up, all of
  which presented identically — as stored-success → live-fail, i.e. exactly like env
  drift — at an agreement of 0.571:
  1. the off-pool derived seed was used for plans that *were* in the pool (fixed by
     matching against `pool_index` first);
  2. canonical episode names (`circle_0`) were handed to an env that knows `button0`;
  3. operators were grounded over env objects but the trajectory was progressed from the
     *canonical* initial state.

  None would have been visible in the resulting number. All three were visible in the
  gate.
- Off-pool seeds derive via `hashlib.blake2b`, not `hash()`. Python's `hash()` is
  `PYTHONHASHSEED`-salted, so a re-score in a different process would have drawn different
  labels for the same proposal — the same class of irreproducibility already recorded for
  the DD2D generator ([2026-07-26](05-v3-migration.md#2026-07-26-dd2d-generator-pythonhashseed-dependent)).
- **Deviation 8** added to `vlmplan/prompts/PROVENANCE.md`: `_CONTROLLER_NOTE` states the
  chaining rule (`…FromNothing` vs `…FromButton` depends on where the robot already is).
  Without it the 32B model used `RobotPressButtonFromNothing` for every press and produced
  **11/11 precondition violations**; with it, 5/5 valid. This is the same failure the DD2D
  run hit 28/28 times on a different near-synonymous skill pair, so the mitigation is now
  a documented part of the template rather than a per-environment rediscovery.

---

<a id="2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters"></a>
## 2026-08-01 — PIGINet lifted to an env-agnostic package with per-env adapters

<!--strip-->
> **id** `2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters` ·
> **status** active · **tracks** baselines, tooling, env-stickbutton2d
<!--/strip-->

**Context.** The DD2D comparison notebook's headline is SPECTRE v3 against **PIGINet** —
the low-level predictor over concrete state. That row is the whole representation
question: "what should a feasibility predictor represent skeletons and problems over?"
StickButton2D had SPECTRE v3 and the B1–B5 bracket but no PIGINet, so the second
environment could not answer the question the project exists to ask.

PIGINet lived at `envs/dd2d/piginet/` and was DD2D-specific in five places: a gloss table
imported at module scope, `_SHAPE_MAX` in centimetres, a `drawer_wh` key read out of
`provenance`, a `dd2d_*` directory glob, and its paths in the cache driver. Individually
reasonable; together they make a second environment a rewrite.

**Decision.** Lift the package to `spectre/piginet/` behind a `PIGINetDomain` protocol,
with one adapter per environment — the shape `vlmplan/` already established here, and the
same move `domain.DomainSpec` made for SPECTRE v3 itself.

- **The normalisers become domain state, not module constants.** This is the reason the
  abstraction is a class rather than two more imports. PIGINet divides poses by a frame
  extent and shapes by per-field maxima so both land in `[-1, 1]`. DD2D's are centimetres
  over a ~50×40 drawer; StickButton2D is metres over 3.5×2.5 with objects two orders of
  magnitude smaller. Measured: SB2D shape features read `|mean| 0.372` against their own
  divisors and **`|mean| 0.0061`, max 0.05** against DD2D's — a channel 60× flatter, i.e.
  effectively dead. The conclusion "the low-level predictor loses on StickButton2D" was
  available as a *unit bug* wearing a result's clothes, and nothing would have raised.
- `PIGINetExample` / `ImageRef` move to `piginet/record.py`; DD2D's `record.py` keeps its
  builders and re-exports them, so every existing import resolves.
- `SB2DDomain` builds examples from the **same `EpisodeRecord` pickles SPECTRE trains on**
  — so the two methods' labels are identical by construction, not by agreement — and
  rasterises crops from stored `scene_geometry` (*reconstruct, never regenerate*).
- The cache driver's `--env-variant` choices came from `_V2_CKPT_SUBDIR`, i.e. "collections
  with a SPECTRE v2.2 checkpoint". StickButton2D deliberately has none, so it was rejected
  at the CLI despite having PIGINet and v3 rows. Now the union of the method maps, with a
  missing method failing on its own rather than blocking the driver.

**Consequences.**

- **DD2D is unmoved, verified on the metric rather than on bytes.** Re-running the dd2d_v4
  PIGINet cache gives rollout FP **17.0500 before and after**, per-problem identical on all
  100 problems, with labels and rank order identical. Scores drift by ≤2.3e-4 — CUDA float
  nondeterminism in CLIP inference. The plan's stated bar was "byte-identical", and that
  bar was **wrong for a GPU inference path**: it cannot be met by any re-run, refactor or
  not. The right criterion for this class of change is identical labels, identical rank
  order and an identical derived metric.
- **`at-pose` literals are synthesised for StickButton2D.** Its abstract initial state is
  two atoms and names no positions, so a faithful port had to add one pose literal per
  object, exactly as DD2D's records carry natively. Without it PIGINet receives object
  identities with no coordinates — it would stop being a *low-level* predictor, which is
  the only reason it is in the comparison. This is our construction, not stored data.
- **The image channel is degenerate on StickButton2D and stays in anyway.** Every unpressed
  button is the same red disc, so CLIP separates only {button, stick, robot} — which the
  type literals already give. Crops share one fixed world window so relative scale at least
  survives (the stick renders as a bar, a button as a dot). Reported as a bound on what
  this environment's PIGINet row can be claimed to show, not silently absorbed.
- The lifted package keeps its mypy exclusion. It was covered by the vendored-DD2D
  exclusion for its whole life; moving a file is not the moment to impose strict typing on
  it. `domain.py`, `record.py` and the adapters are ours and stay checked.

---

<a id="2026-08-01-both-evidence-classes-stay-wired-stickbutton2d"></a>
## 2026-08-01 — Both evidence classes stay wired; StickButton2D has only class 2

<!--strip-->
> **id** `2026-08-01-both-evidence-classes-stay-wired-stickbutton2d` · **status**
> active · **tracks** method, data, env-stickbutton2d
<!--/strip-->

**Context.** The unified coverage/waste definitions (2026-07-31) are computed over
*records*, and `records_from_failure_records` built them from one field: `culprits`, the
objects the refiner's own validity check named. That is §2's **class 1**, and it is all
DD2D produces.

StickButton2D produces **none of it**. kinder's motion model rejects a colliding
transition by silently declining to move, and its collision predicate returns a bool
without naming anything, so there is no object-naming check to instrument. Every SB2D
failure is §2's **class 2**: the sample executes and the trace check finds observed ≠
predicted. Nothing serialized that. The failure mode was not an error — it was
`coverage ≡ 0`, `waste ≡ 0`, and v3 silently degrading to a static ranker while reporting
a clean run. The same shape as the `S(c) = args \ goal_objects` problem the unified
definitions were introduced to fix, one level down.

A second, smaller thing surfaced with it: `records_from_failure_records` *dropped* any
record with no culprits. On SB2D that would have been every record.

**Decision.** One path, both classes, always wired; emptiness is data, not a branch.

- **Class 2 is serialized** into `refiner_metadata["failures"]` as `dev_added` /
  `dev_deleted` — `(predicate, [arg, ...])` **name pairs**, not `GroundAtom`s, because
  they have to survive `canonicalize_episode`'s renaming. `unified_evidence` rebuilds real
  ground atoms from a per-episode predicate table at read time, since every consumer
  compares them by identity against operator effects.
- **The class-1 slot is emitted anyway, empty**, and vice versa on DD2D. No consumer
  branches on the environment.
- **Blameless records are kept** rather than filtered. A failure that names nobody is
  still an observation that this step failed, and the record-token stream reads it.
- **`waste` abstains on an empty culprit pool** (returns 0.0). This is the one place
  keeping blameless records was *not* already inert: with `K = ∅` nothing justifies any
  idle step, so the ratio would return a maximally confident 1.0 derived from zero
  evidence — and only on contexts that named nobody, i.e. as noise correlated with having
  no information.
- **Deviation-derived blame is stored separately**, as `dev_blame`, and feeds the record
  token's culprit tag slot only where `culprits` is empty. A culprit was named by the
  environment; this was inferred by us from the trace. Collapsing them would let a model
  trained where the signal is observed be deployed where it is inferred with nothing
  recording the difference.

**Consequences.**

- Inertness of the empty channel is a **proof, not a measurement**: a blameless record
  contributes nothing to `K`, `covered` skips it for every object, `_justified` never
  consults it, and `waste` now abstains. Pinned by
  `test_blameless_records_do_not_change_coverage_or_waste`. DD2D re-scores at
  **5.78 ± 0.10** — identical to the pre-change figure, per stratum as well as overall —
  which is what discharges the standing "re-score the frozen baseline under new code
  before training anything" rule.
- Two traps this exposed, both of which produce no symptom:
  - **Nested names must be remapped.** `_remap_refiner_metadata` renamed `args` /
    `culprits` / `unmoved`; the object names *inside* `dev_added` / `dev_deleted` are one
    level deeper. Missing them makes every record's tags fail to resolve and the whole
    stream degenerate to "some failure of some schema".
  - **Positional pairing must filter both sides.** `records_for_candidate` silently drops
    entries missing `schema`/`step_index`; pairing its output against the *unfiltered*
    metadata list shifts every later deviation onto the wrong record, with both sides
    still well-formed.
- SB2D collection runs through `RecordingSampler`, which **re-implements** upstream's
  sampler loop rather than subclassing a hook — upstream computes the achieved abstract
  state to decide accept-or-reject and then discards it behind a payload-free
  `TrajectorySamplingFailure`. That is the one place this port does not simply wrap
  kinder. It is a same-seed differential measurement, not a claim:
  `test_stickbutton2d_observational.py` refines the same pools through both samplers and
  requires identical labels (b2 and b3, 3 problems × 8 candidates each). A prior docstring
  asserted such a test existed; it did not, and writing it is what makes this safe.

---

<a id="2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1"></a>
## 2026-08-01 — Acyclic pool filter and the pooled stickbutton2d_v1 variant

<!--strip-->
> **id** `2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1` · **status** active
> · **tracks** method, data, env-stickbutton2d
<!--/strip-->

**Context.** Standing up StickButton2D as SPECTRE's second environment needed a pool, and
the pool the substrate produces is not usable as-is.
`HeuristicSearchAbstractPlanGenerator` deliberately allows revisiting abstract states —
"that's important because we need to generate multiple abstract plans"
(`heuristic_search_plan_generator.py`) — which on this domain licenses padding any plan
with `PickStickFromNothing` / `PlaceStick` pairs. Those return to `s_0` *exactly*, so A*
enumerates them in `f` order and they fill the pool.

Measured acyclic fraction of a 200-candidate draw, over 6 seeds per variant:

| | b1 | b2 | b3 | b5 |
|---|---|---|---|---|
| acyclic / 200 raw draws | **1–2** | 6–34 | 73–101 | 193–200 |
| acyclic, raw budget 5000 | 1–2 | 6–34 | **200** (≈640 raw) | 200 (200 raw) |

At b1 all 200 candidates are the same plan with 0–199 pickup/putdown cycles prepended,
running to 400 operators. A ranker asked to order that is being asked a question about
padding, not about feasibility.

Separately, the four button counts had to become one dataset. They differ by two orders of
magnitude in pool size, which is a difficulty axis rather than four separate problems.

**Decision.** Two things, both env-agnostic.

1. **Filter cyclic skeletons out of the pool** (`AcyclicPlanGenerator`): reject a skeleton
   if `s_i == s_j` for any `i < j`, identity being the atom set. Applied uniformly to
   every variant, with a `raw_cap` of 5000 draws as the stop rule for variants whose
   acyclic set is genuinely finite. It reads only the abstract state sequence, so it would
   apply unchanged to any environment whose generator revisits states.
2. **Pool b1/b2/b3/b5 into one `env_variant`, `stickbutton2d_v1`**, with button count as
   the stratum, encoded arithmetically into the problem id
   (`envs/stickbutton2d/strata.py`): `pid = split_band·10⁶ + slot·250000 + index`, chosen
   so the existing `dd2d_compare.stratum_of` returns the slot exactly. b10 is dropped —
   0/20 problems solvable within the budget, and the cause is pool prefix homogeneity that
   needs diverse plan *generation*, not a better heuristic
   (`autonomous_stickbutton_session.md` D5).

**Consequences.**

- The filter is near-inert exactly where the ranking problem is real (b5: removes 0–7 of
  200) and removes the degeneracy where it is not. b3 gains: 200 *real* candidates instead
  of ~90 real + ~110 padded, which also makes b3 roughly twice as expensive to collect as
  the pre-filter measurement implied.
- **This is a benchmark-definition choice, not a free simplification, and the caveat is
  real**: a padded plan can be *genuinely* more refinable than its acyclic core, because
  `PlaceStick` puts the stick down somewhere new and re-picking it changes the geometry.
  What is claimed is that a pool of near-duplicates is the wrong ranking problem — not
  that the dropped plans are infeasible. A domain where tool re-placement is the point
  would want this off.
- Strata 0 and 1 are anchors, not contests. With pools of ≈2 and 6–34, b1 reads 0.07 mean
  failed attempts under the *static* order and every method ties it — the same shape as
  DD2D's `s0 = 0.00`. About half of b1's episodes have pool size 1 and are dropped by
  `train_v3._trainable` (`len(skeleton_pool) >= 2`). b3 and b5 carry the result, and a
  pooled "ALL" mean over unbalanced strata should not be read as a method comparison.
- The pid encoding is arithmetic and therefore silently breakable, so it is pinned by a
  unit test against `stratum_of` and each episode independently records
  `provenance.gen_params["stratum"]` as an audit trail. Strata occupy contiguous pid
  bands, which makes **stride, never truncate** load-bearing here: `paths[:N]` returns b1
  only.

---

