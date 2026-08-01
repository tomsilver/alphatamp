# SPECTRE Decisions — v3 migration

4 entries, 2026-07-26 .. 2026-07-26 (closed). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-07-26-selection-metric-never-censored"></a>
## 2026-07-26 — A selection metric may never be censored below the tail that separates the models; v3's selector is uncensored over the whole val split

<!--strip-->
> **id** `2026-07-26-selection-metric-never-censored` · **status** active · **tracks**
> evaluation, method · **defines** G6b, R8 · **resolves** G6 · **see also** retracts
> G6's arm levels 18.59 / 19.15 / 20.95
<!--/strip-->

**Context.** v3 selects checkpoints on *deployed val FP* — the quantity actually reported —
rather than v2.2's `relrank`, which was miscalibrated on dd2d_v3. But it was computed with
two economies added for speed, before `spectre_sweep.py` made parallel arms possible: a
30-attempt budget, and a 50-episode strided subsample. G6 then produced a table in which
**every** v3 arm, including the no-records bar, underperformed the v2.2 yardstick — which
located the regression in the shared training path rather than in the increment under test.

**What was measured.** Scoring the *same* G6 checkpoint both ways: **11.66** censored@30 on
50 episodes versus **19.72** uncensored on 100. And scoring v2.2 on v3's own censored
selection metric gives **11.12** against v3's selected **11.40** — indistinguishable, while
the two models are 4+ FP apart uncensored on test. The per-epoch dynamic range tells the
story directly: censored, `val_fp` spans ≈[11.1, 17.5] across 30 epochs; uncensored,
≈[17.1, 32.0]. DD2D s2/s3 episodes routinely need 30–40+ attempts, so the budget clipped
every one of them to the same number.

**Decision.** `TrainV3Config.select_budget` → `None` (run to the pool cap, the uncensored
convention reporting already uses, [`decisions.md` 2026-06-07](../decisions/README.md)) and `val_episodes` → 100.
Stated as a standing rule, because the failure mode is not specific to this metric: **a
selection statistic must not be censored below the region where the candidates differ.** A
budget is a ceiling; if the models separate above it, selection is reading noise. Note this
is *not* the same failure as a noisy selector — the G6 curves were stable and picked
sensible mid-training epochs (13/10/12). They were blind, not jittery, which is why the
usual smell test (unstable curves, early-epoch selection) did not fire.

**Consequences.**
- **G6's arm levels are retracted** (18.59/19.15/20.95 → 16.17/21.24/19.54). Two of its
  conclusions do not survive: v3 is *not* worse than v2.2 (**+1.51 FP, CI [−2.29, +5.72]**,
  includes 0, against **+3.93, CI [+0.37, +7.95]** under censoring), and records-*only* does
  not beat the bar (**+1.70**, n.s., where censored it read −1.80). The record increment
  itself survives and strengthens: **−3.37, CI [−6.16, −0.64]**.
- **Cost is real but affordable**: 51 s/epoch versus 17 s, ~50 min for a 3-arm parallel
  sweep. Both knobs are CLI-exposed (`--select-budget`, `--val-episodes`) so the cheap
  recoveries stay open — select every K epochs, or uncensored on a stride — but *not*
  censoring, which is the one economy that cost more than it saved.
- **The yardstick is now scored by the same instrument.** `spectre_score_v3.py --v2-arm`
  loads a `train_v2` checkpoint in D-8 compat mode, so v2.2 and v3 rows come from one code
  path over one set of episodes and can be compared by paired bootstrap instead of by
  juxtaposing two separately-produced numbers. Verified: it reproduces the published 14.66
  exactly, and `permissive` ≡ `strict` on dd2d_v4, so demotion mode is not a confound.
- **Known residual, deliberately not fixed:** the 3-epoch moving average compares epoch 1
  (a single sample) against later 3-epoch means, which mildly favours early epochs. It did
  not bite in G6 or G6b (selections landed at 12/13/23), so requiring a full window is left
  as a change to make when something depends on it, rather than mid-gate.

---

<a id="2026-07-26-necessity-conditioning-cut"></a>
## 2026-07-26 — Necessity conditioning is cut from v3; s2 becomes a characterized limitation, and the adaptive consolidation is the contribution

<!--strip-->
> **id** `2026-07-26-necessity-conditioning-cut` · **status** active · **tracks**
> method · **resolves** P-v3-1, D2, D4
<!--/strip-->

**Context.** `SPECTRE_v3_proposal.md` §5 made necessity-conditioned scoring the headline
revision: a per-object head predicting p_i, aggregated to a difficulty estimate
`d_hat = sum p_i`, whose `mismatch`/`coverage`/`waste` features would gate the ranker's
length preference per episode. Pre-registered prediction P-v3-1: dd2d_v3 **s2 ≤ 17.08**.
Its gate (D2) was explicitly designed to be able to *falsify the premise before the build*.

**What D2 found.** The fork answered **within-length at every stratum**, not just s2
([`notebook.md` 2026-07-26](../notebook/README.md)). Every row is a rollout over the same pool restriction, so
nothing mixes static-vs-adaptive with restricted-vs-full:

| stratum | v2.2 full | v2.2 length-oracle | astar full | astar length-oracle |
|---|---|---|---|---|
| s1 | 6.20 | 5.20 | 2.24 | **1.24** |
| s2 | 26.00 | **33.92** | 17.08 | **5.80** |
| s3 | 26.44 | 30.76 | 118.76 | **30.36** |

Handing the model the correct plan length makes it **worse** at s2 and s3, while the same
restriction makes plain planner order dramatically better. At s3 the model beats astar 4.5×
on the full pool but *ties* it under the length oracle. So **v2.2's entire measured
advantage is length calibration; within a plan length it is effectively random at s2 and
merely par at s3** — which also sits badly with the v2.2 claim that the within-length PL
loss "forces the geometry signal at every stratum".

**Decision (user call): cut necessity conditioning from v3.** Necessity conditioning
improves *length* calibration — precisely the thing v2.2 already does well — so it targets
the wrong deficit. Its residual hope was the *within*-length component (`coverage`/`waste`
separating same-size subsets), which would first require the head to predict p_i accurately
from geometry, an unbudgeted investigation. The project's interest is the **consolidation of
the adaptive component** (one canonical `FailureRecord`, a declarative axiom registry, and
role-separated record tokens replacing five bespoke fact types), which is where v3's
cleanliness and generality claims actually live.

**Consequences.**
- **P-v3-1 is withdrawn.** s2 is reported as a *characterized limitation* with the D2
  decomposition as evidence, not as a fixed number. That is a stronger scientific position
  than a mechanism that misses: the decomposition says exactly which capability is missing
  (within-length discrimination) and shows a non-learned baseline achieving 5.80 there.
- **Kept as measured groundwork, unwired:** `necessity.py` and its tests, plus the D4
  finding that `d_hat == stratum` exactly while s3 subset-lattice coverage is only 0.171 —
  which is itself a caveat any future attempt must carry. `V3Config.use_necessity` remains
  an explicit `NotImplementedError` so the gap is visible rather than silently absent.
- **A lead is recorded, not taken:** enumeration order is a strong within-length signal
  (astar length-oracle 5.80 at s2) that the deployed model cannot see, because R1 removed
  the prior *wholesale* when only its short-first column was implicated in the dd2d_v3 s3
  collapse. An index-only prior was never separately ablated. Left as future work.
- v3's remaining scope: G5 (FailureRecord + axiom registry), G6 (record tokens), G7
  (overlap 2×2), G9 (length generalization), G10 (geometry interface), G11 (consolidation).

---

<a id="2026-07-26-dd2d-generator-pythonhashseed-dependent"></a>
## 2026-07-26 — DD2D's problem generator is `PYTHONHASHSEED`-dependent; `dd2d_v4` ships with the divergence documented rather than fixed

<!--strip-->
> **id** `2026-07-26-dd2d-generator-pythonhashseed-dependent` · **status** active ·
> **tracks** env-dd2d, data · **amends** 2026-07-26-v3-migration-g0-g2 · **see also**
> amends G0's acceptance criterion
<!--/strip-->

Addendum to the same-day G0–G2 ADR. This one is a *finding about DD2D*, not about v3.

**Context.** G0's acceptance was that `dd2d_v4` — a re-collection whose only intended change
was observation-only refiner instrumentation — would reproduce `dd2d_v3`'s labels
candidate-for-candidate. If it did, the whole comparison landscape (PIGINet, v1, astar and
both VLMPlan arms, ~10.5 h of generation) would carry over untouched. It did not.

**What was measured.** Over the 597 problems present in both collections:

| | identical |
|---|---|
| scene (object poses) | 592 (99.2%) |
| labels, all 200 candidates | 565 (94.6%) |
| task plans | 539 (90.3%) |
| fully identical | 519 (86.9%) |

98 of 119400 candidate labels differ (0.08%), plus 3 problems where the collector kept a
different seed at a stratum boundary.

**Root cause — not the instrumentation.** `generate_dd2d_problem` is deterministic *within*
a process but not *across* processes: with everything else fixed, `PYTHONHASHSEED=0` yields
one scene for seed 500039 and `PYTHONHASHSEED=1` yields a different one, each reproducibly.
Python randomises the hash seed per process by default, so **every DD2D collection ever made
(v2, v3, v4) is a valid sample but not a reproducible one**, and the same seed can even yield
a different `n_items`. The leak is set/dict iteration order somewhere in the generation +
rejection-sampling path; the most likely site is `enumerate.py`'s
`present = set(scene.item_names()) - {scene.target}` feeding `_obstacles`. This is almost
certainly the same class of problem behind the neighbouring finding that the dd2d_v3
comparison cache is no longer reproducible from the code on disk.

The instrumentation itself is clean and was verified separately: replaying stored candidates
at their stored seeds through the instrumented refiner reproduces `label`, `steps_bound`,
`plan_length` and `failure_action` on 290/290.

**Decision (user call): accept `dd2d_v4` as collected and document the divergence; do not fix
the generator or re-collect now.** Rejected alternatives: (a) fix the hash-order leak, add a
cross-process determinism test and re-collect (~2 h — makes DD2D reproducible for every later
gate, but v4 still would not match v3, so the comparison rows need re-running either way);
(b) fix and additionally re-run every comparison row including both VLMPlan arms (~10.5 h of
VLM generation to move a row whose qualitative shape cannot plausibly flip on a 0.08% label
change).

**Consequences.**
- G0's acceptance criterion is **amended**: label identity with v3 is unachievable in
  principle, so what is asserted instead is (i) the instrumentation is observation-only
  (verified differentially, 290/290) and (ii) the v3-vs-v4 divergence is measured, bounded
  and recorded above.
- `dd2d_v4` is a **separate env variant**, never averaged with v3 into one row. Numbers are
  comparable at the ~0.1%-label level, far below seed noise, but they are not the same data.
- Existing PIGINet / v1 / astar / VLMPlan rows carry over **with this caveat attached**; any
  claim that turns on a <1% effect must be re-run rather than inherited.
- **Known issue, deliberately left open:** DD2D generation is not reproducible. Anyone
  re-collecting should expect a fresh sample, and a future fix should sort the offending
  iteration and add a cross-process regression test (two `PYTHONHASHSEED` values, one scene).
  Recorded here so it is a known limitation rather than a recurring surprise.

<a id="2026-07-26-v3-migration-g0-g2"></a>
## 2026-07-26 — v3 migration G0–G2: instrument rather than reconstruct; one domain contract; the equivalence oracle is a live run, not a cached artifact

<!--strip-->
> **id** `2026-07-26-v3-migration-g0-g2` · **status** amended · **tracks** method,
> data, env-dd2d · **defines** G0, G1, G2, D-7, D-8 · **resolves** R1, R2, R9
>
> ⚠️ **AMENDED** — the 13.68-vs-14.50 oracle discrepancy recorded here was double
> canonicalization, and the corrected figure is **14.50**, settled in
> [2026-07-27-cross-collection-grafting-coverage-mode](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode).
> G0's acceptance criterion is amended by
> [2026-07-26-dd2d-generator-pythonhashseed-dependent](05-v3-migration.md#2026-07-26-dd2d-generator-pythonhashseed-dependent).
> **Do not quote 13.68.**
<!--/strip-->

Opens the SPECTRE v3 migration (`docs/SPECTRE_v3_proposal.md`, Phases 0–3, DD2D-complete;
Phase 4 / env-2 explicitly descoped). Five decisions, each with a consequence that outlives it.

**(a) Re-collect `dd2d_v4` with an instrumented refiner, rather than backfilling.** The v3
`FailureRecord` needs `culprits` — which objects the failed samples actually collided with —
and that is the one field no stored collection carries: `has_grasp` returns `Grasp | None` and
`drawer_obstacles()` returns *unnamed* polygons, so nothing downstream can name a blocker.
Backfill covers everything else exactly (`j` = `steps_bound`, verified == the index of
`failure_action` on 2528/2528; `q`/`args` = `task_plan[j]`; `U(σ,j)` by set arithmetic), but
leaving `culprits` empty would keep the record schema permanently hypothetical and push
unproven instrumentation onto env-2. Collection is only ~1.6 h, so the cost is small. Rejected:
keeping the analytic `grasp_witness_after_removing` as a stand-in — that is the per-env geometry
routine R4 exists to delete.

**(b) Instrumentation is observation-only, and that is a hard invariant, not a style note.**
`n_attempts` *is* `counter.calls`, so one extra stream call shifts it and cascades into every
label. `grasp_cfree` was therefore refactored to `grasp_blocker(...) < 0` — the culprit is the
witness the short-circuit already computed — and only the *first* blocker per grasp cell is
recorded, because enumerating all of them would mean doing extra intersection work. Verified
differentially against the pre-instrumentation collection: `label`/`steps_bound`/`plan_length`/
`failure_action` identical on 290/290 replayed candidates.

**(c) `n_attempts` is contaminated where the wall-clock budget binds, so the identity gate is
split.** Measured p99 elapsed 20.0046 s against a 20 s budget. Off-budget, `n_attempts`
reproduces 286/286; on-budget (1.4% of candidates) it reproduces **0/4** and systematically
~2× high, because this box is faster than the one that collected v3. So `n_attempts` there
measures host CPU speed. Consequences: the v4-vs-v3 identity gate asserts it only off-budget;
the v3 record names it `n_total` and masks it where `budget_exhausted`; and the `3s+1` exactness
witness is unaffected (budget-bound candidates are nowhere near the floor). Rejected: removing
`time_budget` to make the refiner a pure function of its seed — it would change labels and force
re-scoring PIGINet, v1 and both VLMPlan arms (~10.5 h of generation).

**(d) The `exactness` axiom, and why v2.2's demotion was unsound.** `refine()` loops
`while idx < n and not exhausted()` and then reports `failure_action = plan[best_reached]` — the
deepest step *reached*, not necessarily one that was tested. On a budget exit it still names
`retrieve(target)` though the retrieve never ran. That is the confirmed cause of all **12/18694**
dd2d_v2 demotion violations (all one candidate, `n_attempts=2406`); dd2d_v3 has **0/19547**. So
the leaking axiom is **exactness**, not locality — closing the "cause unknown" the proposal left
open. `QueryAxioms` splits it correctly: the *domain* declares "a completed run of this query is
exhaustive"; the *observation* says whether it actually ran. A fourth emission site now records
the budget-exit case explicitly, so "no evidence" is distinguishable from "lost in conversion".

**(e) One `DomainSpec` replaces eleven DD2D literals; `soundness.py`'s per-fact-type registry is
superseded.** v2.2 hard-coded `place-buffer` (to derive a candidate's manipulated set and plan
length) and `retrieve` (to license demotion) inside supposedly domain-agnostic modules. v3's
`domain.py` derives all three from the operator schema plus a per-query axiom declaration —
DD2D's entire environment-specific content is **three lines and zero geometry**. Licensed by
proof, not hope: `args(σ) \ goal_objects` equals the `place-buffer` filter on **120000/120000**
skeletons, and `len(operator_seq) == 2·|staged|+1` on the same 120000, so the within-length loss
partitions the pool identically. Unknown env variants degrade to `EMPTY_SPEC` (everything
hint-tier) rather than raising — "learning is the floor" becomes the default, not a special case.

**Also settled here.** **R1** (short-first prior) — removed as a scorer feature; only the plan-length
bucket key survives, now `domain.length_key`. Note any historical prior on/off delta is confounded:
enabling it *also* zero-inits the scorer head (`model_v2.py:274-276`). **R2** (computed/geometry
demotion) — not ported. **R9** (`exclude_marginal`) — not ported; it was inert twice over (DD2D only
writes `status ∈ {feasible, infeasible}`, and `collate_v2:384` folded the resulting `None` back to
`False`), so reinstating it needs a real label mask, not a flag. **D-8, the exact-absence
invariant:** every v3 feature is config-gated and compat mode builds the *v2.2 submodule classes*,
so a v2.2 checkpoint loads `strict=True` and the oracle survives until the position encoding is
replaced.

**(f) The equivalence oracle is a live v2.2 run, because the cached one turned out not to be
reproducible.** The intended oracle was the dd2d_v3 comparison cache (it stores the exact attempt
order, per-step demotion state and logits of the deployed policy). Replaying it showed the
*current* v2.2 code and the *current* deployed checkpoint no longer reproduce it: **mean FP 14.50
vs the cached/published 13.68**, per-problem FP identical on 61/100, attempt order on 55/100.
Ruled out: not v3 (v2 and v3 agree bit-for-bit), not dropout (identical in eval), not device (CPU
and CUDA agree), not nondeterminism (identical across processes under `PYTHONHASHSEED=random`),
not the other checkpoint. **Cause identified the same day and recorded in a follow-up entry:
`canonicalize_episode` is not idempotent and the cache builder applied it twice (episodes came
from `eda.load_split_episodes`, already canonicalized, and `build_v2_example` canonicalized
again), so cached numbers were computed on a different object->tag binding than training uses.
Not code staleness -- that diagnosis was wrong. The oracle was therefore
re-pointed to a side-by-side v2.2/v3 run — strictly stronger (bit-identical, no 4-dp tolerance),
needs no stored artifact, and cannot rot. **Consequence: 13.68 must not be quoted again until the
cache is rebuilt with `--force`** (it appears in `as_built_v2.2` §3.7 and two `notebook.md`
entries); the rebuild will settle whether the corrected figure is 14.50. Nothing in the v3 arc
depends on it — the yardstick is a fresh 3-seed v2.2 run on dd2d_v4.

**Consequences.** New: `domain.py`, `model_v3.py`, `dataset_v3.py`, `inference_v3.py`, and
`FailureObservation` + four emission sites in the DD2D refiner; `CONVERTER_VERSION` →
`dd2d_convert_v3`; `ProvenanceBlock.gen_params` (trailing-nullable + `io._migrate` shim, the
v2.2.1 pattern) carries the generator/refiner arguments as an **audit trail only** — it holds
`stratum`, which is the answer, and nothing in the dataset path reads `ProvenanceBlock`.
`refinement_wall_clock_s` is finally populated (it was hardcoded 0.0 because `record.build_example`
dropped `elapsed`), closing the gap where DD2D could not report the proposal's stated *primary*
metric. Tests +27 (`test_dd2d_grasp_witness`, `test_instrumentation_is_observational`,
`test_v3_equivalence`, `test_domain`); spectre suite 385 green, dd2d env suite 123 green.
[`notebook.md` 2026-07-26](../notebook/README.md) has the numbers. **Open:** the necessity labeller (G8) — no collection has
ever populated `aux_labels`, so v2.2's aux head has never received a gradient and §5 is a
build-from-scratch, not a promotion.

