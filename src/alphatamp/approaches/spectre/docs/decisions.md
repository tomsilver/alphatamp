# SPECTRE — Decision Log (ADR style)

One entry per consequential decision. Newest first. Format: context → decision
→ consequences. Refactor-era entries record what moved, what deliberately did
not, and why.

---

## 2026-07-26 — v3 migration G0–G2: instrument rather than reconstruct; one domain contract; the equivalence oracle is a live run, not a cached artifact

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
not the other checkpoint. `model_v2.py`/`dataset_v2.py` are clean at HEAD and the episodes predate
the cache, so the cache was written by a code state no longer on disk. The oracle was therefore
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
`notebook.md` 2026-07-26 has the numbers. **Open:** the necessity labeller (G8) — no collection has
ever populated `aux_labels`, so v2.2's aux head has never received a gradient and §5 is a
build-from-scratch, not a promotion.

## 2026-07-25 — VLMPlan on the v3 test split: two model arms, output caps were binding, one cache dir per arm

Addendum to the 2026-07-24 VLMPlan ADR; the protocol there is unchanged.

**Context.** VLMPlan had only been smoke-tested on 16 **train** problems with one 8B model.
The v3 re-collection has since completed (100 test problems, exactly 25/stratum) and every
other method was retrained and re-scored on it, so the remaining work was the VLMPlan row on
the real held-out split. Three things forced decisions.

**(a) Both arms are Qwen3-VL Instruct; a Gemma arm was tried and rejected.** With the
Qwen3-VL-32B download stalled at 6.95 GB of ~19 GB, `google/gemma-4-31b-qat` was tried as the
large arm — complete on disk, ships its own `mmproj`, genuinely multimodal. It was rejected
because it is a **reasoning model**: a trivial "which item is red?" answer cost 229 completion
tokens of which **222 were `reasoning_tokens`** (~95%), and those count against `max_tokens`
but are stripped before the text the parser sees, so the plan budget goes to invisible
thinking. On the real 10-plan prompt it exceeded 2 minutes without finishing 3 rounds (vs
~10 s/round for the 8B). **The arm axis that matters is therefore not size but model class:
instruct-tuned, non-reasoning, same family.** The large arm is `Qwen3-VL-32B-Instruct`
(resume the stalled download; `mmproj` already complete), and Qwen3-VL **Thinking** variants
are excluded for the same reason. Holding the family fixed also upgrades the pair from a
confounded size+family contrast into a clean **scale** comparison — a strictly better
property than the lineage diversity Gemma would have bought.

**(b) The output cap was silently truncating, and is now telemetered.** Read out of the
run's own response cache: `completion_tokens` hit **exactly 4096** (the cap) on **16/104**
responses and `total_tokens` hit **exactly 8192** (the served window) — both limits were
clamping generation, so the final plan block of those rounds was cut mid-line and dropped by
the parser. Rounds with long responses had the worst malformed rate (0.58 vs 0.26 mid-band),
consistent with that. Fixes: `max_tokens` 4096 → **8192**, models served at **32768**
context, and `RoundLog` now records `prompt_tokens`/`completion_tokens`/`truncated` with
`completion_tokens >= max_tokens` as the exact signal — no guessing from response length and
no upstream change (the backend already returns usage). `n_truncated` rides through the
sequences file onto the compare-cache record and into the notebook's §9, with a warning
banner when nonzero, because a truncated round is a **config fault that under-reports the
model**, not a model result. Consequence: the 2026-07-24 smoke numbers were produced under
truncation and are superseded; and because `max_tokens` is part of the `prpl_llm_utils`
cache key, raising it re-generates rather than replays.

**(c) One cache subdir per arm** — `vlmplan_qwen8b`, `vlmplan_qwen32b` (was a single
`vlmplan`). A cache dir *is* one method row, so two models sharing one would average into a
row that is neither; the existing `assert_single_run` guard catches the same class of error
for two runs. `load_vlmplan_diagnostics` lost its `subdir` default for the same reason —
with two arms, a default makes it easy to read the wrong one.

**Everything else stays frozen** — τ=0.2, `stall_rounds`=2, `plans_per_round`=10,
`max_rounds`=12, `max_plans`=200, temperature 1.0 — so the two arms differ only by model.
The caps change is a bug fix, not a re-tune.

**Consequences.** `SEQUENCE_METHODS` holds both arms and `METHOD_ORDER` picks both up; the
notebook reads **dd2d_v3** and renders one §9 block per arm (skipping an absent one), and its
§5/§6 T0/T1 prose is banner-flagged as written against v2 while the figures beside it render
v3 — re-deriving those conclusions on v3 is deliberately out of scope.
`notebook.md` 2026-07-25 has the numbers.

---

## 2026-07-25 — The dd2d_v3 headline reversal was a training artifact (short-first prior); v2 drops `--use-prior` on v3, restoring SPECTREv2-adaptive as best

**Context.** The 2026-07-24 entry below reported the dd2d_v3 comparison "reversing" — PIGINet best,
SPECTREv2-adaptive collapsing at s3 (85.52, *worse than v1*) — and read it as a real packing
negative-control crossover. The user challenged it: the grasp fix *adds* feasibility, so methods
should improve, not reverse, and v2 (strictly more information than v1) should never do worse than
v1 at s3. A read-only investigation showed the reading was wrong.

**Findings (evidence).** (a) The original dd2d_v2 v2 checkpoint *survives* and, rescored through the
current pipeline, reproduces the published **17.09** exactly → scoring/cache is faithful. (b) Recipe
and training code are byte-identical to that checkpoint (git archaeology). (c) v3 is *easier* than
v2 at every stratum, so "harder data" is ruled out. (d) On v3 the v2 training *diverges* into a
short-first length bias (val_relrank 0.99@e4 → 2.1 while val_loss keeps dropping); the noisy relrank
selector then picks the underfit epoch-4 fluke. Ablation pinned the cause to the **short-first
`--use-prior`** (`[−index,−length]`), which over-biases cross-length ordering on the easier v3 data
and buries the (long) s3 feasibles.

**Decisions.**

(a) **Retract the 2026-07-24 "PIGINet decisively wins / DD2D negative control confirmed" framing** —
it was artifact-contaminated. The pipeline and method were faithful; only the trained v2 checkpoint
was bad.

(b) **Drop `--use-prior` for dd2d_v3** (val-justified: no-prior val ALL 16.9 vs with-prior 29.9),
keeping evidence + overlap + within-length PL + observed proof-demotion. The short-first prior is
now a **data-dependent knob** — it helped on the harder dd2d_v2 / RT2D distributions but hurts the
easier grasp-fixed v3 — *not* a fixed part of the deployed recipe. Corrected result:
**SPECTREv2-adaptive best overall (13.68**, beats PIGINet 18.67 and the v2-data 17.09), s3 fixed
(23.92 vs 85.52), same qualitative shape as v2 (v2 best, strong s3, weaker s2). Dropping the prior
also *restores training convergence* (val deployed-FP stable ~13–16, epochs 12–29).

(c) **`precompute_dd2d_cache.py` picks the v2 checkpoint dir per variant** (`_V2_CKPT_SUBDIR`:
dd2d_v2 → `checkpoints_v2_evidence_prior_ov`, dd2d_v3 → `checkpoints_v2_evidence_ov`), so the cache
reads the config each collection actually uses.

(d) **`train_v2.py` gains `--lr`** (peak LR, for the anti-divergence sweep) and an env-gated
per-epoch checkpoint dump (`SPECTRE_SAVE_ALL_EPOCHS`, diagnostic; harmless when unset). The wl-weight
and lr sweeps did *not* fix the collapse (higher wl-weight made s3 *worse*, refuting "length shortcut
in the loss" as the sole cause) — dropping the prior did.

**Consequences.** Selection caveat: relrank is *miscalibrated* on v3 (never <1) yet safe once the
destabilizing prior is removed (the model converges, so it picks a good epoch); a deployed-val-FP
cross-check (epoch 14 → 15.88) confirms robustness. Only the v3 spectre2 cache changed;
v1/PIGINet/astar/lenctx are untouched. `notebook.md` 2026-07-25 has the corrected table. **Open
(carried):** 3-seed reproduction — the prior's v2-helps/v3-hurts data-dependence and the residual s2
gap (v2 26.20 vs PIGINet 18.60) both need ≥3 seeds; wiring deployed-val-FP selection into `train_v2`
for robustness is a natural follow-up.

## 2026-07-24 — DD2D comparison retargeted to grasp-fixed dd2d_v3 (all 3 models retrained); precompute parameterized by `--env-variant`; the headline flips to PIGINet

> ⚠️ **CORRECTED 2026-07-25 (see the entry above).** The "headline flips to PIGINet" conclusion here
> was a **training artifact** — the short-first `--use-prior` collapsing v2 at s3 on the easier v3
> data — not a real crossover. After dropping the prior, SPECTREv2-adaptive is best (13.68) and s3 is
> fixed (23.92). The retarget-to-v3, `--env-variant` parameterization, and VLMPlan-pilot handling
> below all stand; only the *result/consequences* (PIGINet winning) are superseded.

**Context.** The same-day grasp-model fixes (contact-run material contact, blocky `horseshoe`,
internal concave grasps) and the exact-count collector shifted DD2D feasibility labels, so the
dd2d_v2 SPECTRE-v1 / SPECTRE-v2.2 / PIGINet checkpoints and the whole `compare_dd2d_methods.py`
cache are **stale** (both ADRs that day flagged re-collection → retrain → recompare as required).
The user re-collected a clean dataset at repo-root `data/dd2d/raw_v3/` (100/100/100/100 train, 100
val, 100 test; λ=0.8). This ADR records retraining the comparison on it (VLMPlan out of scope for
this pass; its pilot is train-band only).

**Decisions.**

(a) **Retarget the DD2D comparison to dd2d_v3; the dd2d_v2 numbers are retired.** All three models
were retrained on dd2d_v3 at their established recipes — v1 `spectre_train.py env=dd2d_v3`; v2.2
`train_v2 --evidence --use-prior --use-overlap` (observed demotion, `_ov` dir); PIGINet
`piginet.train --arm weighted_bce --select auprc` on the **native raw_v3 JSON** (`glob(data_root/
split/*/[0-9]*.json)` — no separate collection, just a fresh `clip_cache_v3`). The notebook's
`ENV_VARIANT` is now `dd2d_v3` (n 142→100). Old v2 comparison numbers must not be reported.

(b) **`precompute_dd2d_cache.py` is parameterized by `--env-variant`** (default `dd2d_v2`,
behaviour-preserving). It repoints the test split, vocab, v1/v2 checkpoints, and cache dir by a
string swap; `N_PROBLEMS` is now derived from the real test-split episode count (v3 = 100, was a
hardcoded 140). **PIGINet's paths are pinned per variant** in a `_PIGINET_PATHS` dict — its data
root, CLIP cache, and BCE checkpoint were operator-chosen (v2 lives under the `src/…/envs/dd2d`
tree, v3's data at the repo root), so they can't be derived from the variant by a swap. `meta.json`
now records `env_variant`.

(c) **VLMPlan's 16-problem pilot scoring cache is set aside** (`compare_cache/vlmplan` →
`vlmplan_pilot_bak`, reversible). Its problems are **train-band**, not the test split, so loading it
would inject a spurious partial VLMPlan row into the 6-method test comparison. The notebook §9 then
renders its documented "no VLMPlan cache" placeholder. Restore when a full test-split VLMPlan run
exists. **1-seed dev** parity (SEEDS=[0]) — no multi-seed change.

**Consequences — the headline flips.** On grasp-fixed dd2d_v3 the result **reverses** vs dd2d_v2:
**PIGINet (low-level) is best overall (FP 18.67** vs its old-data 29.70; val AUPRC 0.256→0.429),
and **SPECTREv2-adaptive — the dd2d_v2 winner (17.09) — drops to 24.96 with an s3 collapse
(85.52)**. SPECTREv2 still wins the relational mid-stratum s2 (8.56) but s3 (3-blocker) feasibility
now hinges on packing geometry the abstract representation cannot see, so the low-level predictor
regains its edge — the **packing negative-control / crossover** the pivot predicts (`proposal.md`
§0/§6, and the §6 "DD2D as the packing / negative-control testbed" item). This is a **data-driven**
result: `train_v2`/model/loss are unchanged; v2's `val_relrank` selection is now near-random and
noisy on this data. No model/loss/pipeline change. **Open (blocking any DD2D SPECTRE claim):**
3-seed reproduction — the 1-seed v2 selection is noisy — carried over from 2026-07-20/23.
`notebook.md` 2026-07-24 has the full table + T0 fit.

## 2026-07-24 — VLMPlan baseline: static hard line, off-pool proposals live-refined, generation split from scoring

**Context.** The data × perception grid in the pivot (`proposal.md` §0) has PIGINet as the trained
low-level predictor and SPECTRE as the trained abstract-first one, but the
**zero-training-data / generic-perception corner is empty** — and "did you just try asking a VLM?"
is the reviewer-obvious question. `docs/vlmplan_dd2d_implementation_plan.md` designed a zero-shot
LLMPlan/VLMPlan baseline in the KinDER convention (Huang et al. 2026); nothing was built. Scope was
trimmed by the user to a functional VLMPlan arm (no ±image LLMPlan arm, no §8 probes), on a local
open-weight model for development with a frontier API arm later.

**Decisions.**

(a) **VLMPlan is a *sequence* method, not a re-ranker, and it is scored on the shared metric.**
Every other row reorders the fixed 200-candidate pool; VLMPlan must *produce* its ordered attempt
list. It is still scored on failed refinement attempts before the first success, so the rows are
comparable — but see (c).

(b) **Static hard line.** Between generation rounds the model sees only its own previously proposed
plans (for de-duplication), never a refinement outcome. Any outcome feedback would make it an
adaptive method and a different table row; VLMPlan exists to occupy the *data* axis, not to compete
with SPECTRE's adaptivity.

(c) **Off-pool proposals are live-refined and cost an attempt** (user call over pool-restriction).
The pool holds every 1-blocker and most *ordered* 2-blocker stagings but only ~4% of the 3-blocker
orderings — exactly where stratum 3 lives — so dropping off-pool proposals for free would hand
VLMPlan free attempts and flatter it at the hardest stratum. Labels come from
`reconstruct_scene` + `staging_skeleton` + `DD2DRefiner` (reconstruct-never-regenerate, 2026-07-19),
memoised to disk. **In-pool proposals are never re-refined** — their label is read from the stored
`OutcomeRecord`, so VLMPlan sees byte-identical labels to every other method. Consequence: VLMPlan's
FP counts attempts outside the pool, so it can reach plans no other method can while paying for
every wrong guess. Disclosed in the notebook header rather than buried, and it is why VLMPlan is
absent from the T0 length sections (those need a score per pool candidate).

(d) **`label_agreement` is a first-class gate, because mixing two label sources needs proof.**
Re-label stored pool plans live and compare. Measured n=168: **dd2d_v3 0.982** (fresh) vs
**dd2d_v2 0.917** (collected before that day's grasp changes), the v2 gap appearing in *both*
directions — the signature of one monotone-harder change (contact-run fix) and one monotone-easier
(internal grasps). So the live refiner tracks current env code and v2's gap is the staleness, not a
bug: the refiner is deterministic at v2 settings (live-vs-live 60/60) and the 2026-07-19
reconstruction invariant still holds (0/1624). `vlmplan_score.py` warns below 0.95.

(e) **Refiner settings are a per-collection preset, not a constant.** v2 collected at
`time_budget=4.0`, v3 at `20.0`; a live off-pool label must match its collection's budget or it is
drawn from a different distribution than the stored labels. `REFINER_PRESETS` keyed by env_variant,
`KeyError` on an unknown one rather than a silent default.

(f) **Generation and scoring are separate entry points.** `vlmplan_run.py` queries the model and
writes proposal sequences + transcripts; `vlmplan_score.py` labels them into a compare-cache record.
So a re-collection, or any metric change, re-runs only the free local half, and nothing runs
inference at notebook load (the standing constraint from 2026-07-23).

(g) **Two prompt deviations were necessary, not cosmetic** (full list in
`vlmplan/prompts/PROVENANCE.md`; the KinDER template is vendored byte-identical from
kinder-baselines `4c731dc`, MIT, because `kinder_vlm_planning` is not an installed dep and `kb/` is
gitignored):
- **Per-skill semantics.** The template lists controller signatures only. `pick` and `retrieve` are
  not self-distinguishing in DD2D, and the local model ended **28/28** otherwise-valid plans with
  `pick(target)`. Every other method reads those preconditions/effects from the PDDL domain, so
  stating them in words **removes a handicap**; it says nothing about which subset to stage.
- **Three counted parser leniencies** (markdown decoration, omitted `:type`, omitted `[]` when the
  params box is empty). Under the strict parser 31/31 plan blocks in a round were rejected purely
  for writing `pick(item_2)`, which would make the headline a measure of format compliance rather
  than planning. The omitted type is resolved from the object registry and checked identically, so
  validation is not weakened; a *wrong* stated type is still rejected.

(h) **Temperature 1.0, not 0.0**, plus a per-round `seed`. At temperature 0 the only cross-round
variation is the repeat-suppression block, so a round yielding nothing leaves the next round with a
byte-identical prompt. KinDER's published runs also use temperature 1.

**Consequences.** New package `vlmplan/` (env-agnostic core + `dd2d_adapter.py` as the only
env-aware module) + `envs/dd2d/spectre_render.py` (the labelled Set-of-Mark render, promoted out of
the comparison notebook, which had the only hole-honouring, legibly-labelled DD2D renderer) +
`experiments/spectre/vlmplan_{run,score}.py` + `conf/vlmplan.yaml` + `test_vlmplan.py` (38 offline
tests). `dd2d_compare` gains `SEQUENCE_METHODS` (skipped when absent, so the notebook loads without
a VLM cache) and `load_vlmplan_diagnostics`; the notebook gains a §9 diagnostics section (including
the pre-registered **trivial-mimicry null**) and len-aware bar offsets. **Two incidental fixes with
wider reach:** `stratum_of` was test-split-only (it returned *negative* strata on train — surfaced
as `s-4`) and is now split-agnostic via `seed % 1M`, verified bit-identical on the test band so no
published number moves; and `dd2d_v3` is registered as an env variant (registry + conf) so the fresh
collection is usable without overwriting v2. Deferred: full test-split run on the fresh collection,
the `Qwen3-VL-32B` and GPT-5.x arms, the ±image LLMPlan arm, the §8 probes.
`notebook.md` 2026-07-24 has the numbers.

---

## 2026-07-24 — DD2D collector guarantees EXACT per-stratum counts (in-flight cap + truncation)

**Context.** The parallel collector (`envs/dd2d/dd2d/collect.py` `collect_split`) balances strata
via per-stratum sub-targets (`--target-train 400` → `[100,100,100,100]`). `next_task` already
diverts workers off a filled stratum (round-robin skip on `kept >= target`), but tasks *already
in-flight* for a stratum keep completing after it hits target → **overshoot** (bounded by the
in-flight count, ~all workers for the last-remaining stratum), so a run could end with >400 and
uneven strata. User wants exactly the sub-target per stratum, with freed workers diverted (not
wasted on overshoot).

**Decision.** Two additive changes, no CLI/param change (the existing `--target-*` still expresses
"N per stratum" as `4·N` total):
(a) **In-flight cap** — track `in_flight` per stratum; gate `next_task` on `kept + in_flight >=
target` (was `kept >= target`), increment on submit, decrement on completion. Maintains
`kept + in_flight <= target` at all times ⇒ **no overshoot, zero wasted refines**, self-corrects for
drops (an in-flight drop frees a slot and the stratum refills), and preserves diversion. The
`workers<=1` serial path is unchanged (`in_flight` stays 0 ⇒ old behavior; already exact).
(b) **`_truncate_to_targets` at finalization** — deletes any residual overshoot from disk (keeps the
first `sub_target` by seed, lowest = collected first) and re-derives the manifest tallies. A no-op on
a fresh run (the cap prevents overshoot); it exists as the tested correctness anchor and to make a
`--resume` over a split a *prior (pre-cap)* run overshot exact.

**Rejected.** Overshoot-then-truncate *without* the cap (keeps full tail parallelism but wastes
~`workers` truncated refines on the last stratum). Chose the cap: exact-by-construction and no wasted
compute, at the cost of the final ~`workers` keeps of the last stratum running under-parallelized
(~one extra wave — negligible vs a multi-hour run).

**Consequences.** `--target-train 400` now yields **exactly** 100/100/100/100 (verified: a real
`target 8, workers 4` run ended 2/2/2/2, 8 total, manifest == disk, no overshoot). Tests:
`test_truncate_to_targets` (pure, idempotent) + `test_collect_split_truncates_overshoot_to_exact`
(resume); full `test_dd2d_collect.py` green (14). No change to `collect_problem`, the model, loss, or
the rest of the pipeline. Note: the parallel cap can't be unit-tested through the `monkeypatch`
fake-task harness (it doesn't cross into worker subprocesses) — its correctness rests on the
`kept+in_flight<=target` invariant, the truncation safety net, and the real smoke.

---

## 2026-07-24 — Grasp model extended to internal (concave-region) grasps: grip a sub-feature where the fingers fit

**Context.** After the same-day contact-run fix + blocky horseshoe, the demo videos showed the model
still only ever grips the **outer envelope** — no grasp reaches into a concave region. A real
parallel-jaw gripper can hold the middle bar of a dumbbell or reach into a C; the user asked for
that. Root cause: `direction_admissible` only ever emits the **global x-extreme** supporting lines
(`rot.bounds`), so both fingers always land on the outer envelope. The `Grasp` dataclass already
carries arbitrary `xmin`/`xmax` and the fingers approach from *outside* `[xmin,xmax]` — exactly what
a waist/opening grasp needs — so the limitation was **purely in enumeration**.

**Decision.** Add `grasps._internal_grasps`: a **scan-line antipodal** enumerator that, per
direction, admits any **strictly-internal** material segment `[a,b]` (aperture-valid) as a grasp iff
(a) **finger-fit** — the finger rects clear the item's *own* material ("the grippers fit" in the
concavity) — and (b) **full-face flat contact** — each finger inner face lies on the boundary for
≥ `_FULL_FACE_FRAC`·`FINGER_WIDTH`. `grasp_cells` = the existing global-envelope grasps **+** internal
grasps (deduped; global tried first in `has_grasp`). Only **validated** scan slides are emitted
(never interpolated) and the **exact** segment endpoints are used as `xmin`/`xmax` (rounding is for
grouping only) — both were required to keep finger-fit and contact exact.

**Rejected.** (a) *Vertical-edge detection* to find internal faces — a 90° rotation perturbs edge
x-values past any tight flatness tol, so it silently misses faces (the scan-line approach is
rotation-robust). (b) *Partial-contact internal grasps* — allowing a finger to touch material over
only part of its face admits **curved-shape sliver grasps** (pinching a circle near its top, normals
not antipodal). Requiring **full-face** contact restricts internal grasps to flat features (dumbbell
bar, horseshoe spine/prong, shoe arm) and leaves circles with only their global tangent grasp, so
`can`/`bowl`/`box`/`pillcase` gain **0** internal grasps — verified.

**Consequences.**

- **Realism achieved + demonstrated.** dumbbell gripped at the **bar** (sep ≈ 1.76 cm), horseshoe
  finger **inside the C-opening** (spine, sep ≈ 2.31 cm) + both prongs, shoe finger in the **L-corner**;
  all full-face (2.50/2.5 cm). Finger gap and finger∩item overlap are **0** for every family.
- **Monotone-EASIER, and it flips the earlier-today direction.** Adding grasp cells can only make
  `has_grasp` succeed *more*, so feasible-candidate sets can only grow and `min_feasible_subset` can
  only drop — **partially offsetting** the no-air-grasp change (monotone-harder). Both are
  realism-driven, not difficulty-tuning. DD2D labels shift again (extraction only; `certificate.py`
  packing is unaffected), so the v2.2 collection stays stale and **re-collection → vocab → retrain →
  recompare remains required and deferred** (unchanged scope: code + tests + demos only).
- No change to the SPECTRE model/loss/pipeline. New tests: `test_internal_grasp_on_dumbbell_waist`,
  `test_internal_grasp_on_horseshoe_spine`, `test_fingers_fit_in_isolation`,
  `test_convex_families_have_no_internal_grasp`; existing grasp invariants unchanged. Demo tags an
  "INTERNAL GRASP" cell and `select_cells` always shows one. `notebook.md` 2026-07-24 has the numbers.

---

## 2026-07-24 — Grasp model contacts material (slides on true contact runs, not the hull); `banana`→blocky `horseshoe`

**Context.** The 2026-07-23 concave-grasp probe showed the DD2D two-rectangle gripper closing
onto **air** on concave items: `direction_admissible` kept only the y-**hull** of each supporting
line's contact set and `_slide_positions` drew the slide from the middle of that hull, so on a
disconnected contact set (a C-opening / waist) a finger landed opposite a gap. On the curved
`banana`, *every* grasp was either a gap-closure or a single tangent point — no grasp put finger
area on material. Ahead of adding a VLM-planning baseline (a collection boundary), the user
directed fixing the grasp model for realism (so a reviewer cannot object that the gripper does not
perform as described) and replacing the banana with a blocky shape that admits genuine full contact.

**Decisions.**

(a) **Slides are drawn from the intersection of the two lines' *actual* contact runs, not the
hull** (`grasps.py`: new `_contact_runs_on_line` returns every disconnected run incl. degenerate
tangent points; `_intersect_runs` overlaps the left/right run lists; `direction_admissible`
returns `list[tuple[float,float]]` of valid slide sub-intervals; `_slide_positions` distributes
slides across them). **Contact rule = both finger centres on material (gap = 0).** A single
tangent point counts, so **circles keep their valid grasp**; the stricter "full 2.5 cm finger face
within a run" variant was **rejected** — a circle is point-contact, so it kills 100% of `can`/`bowl`
(the two most-weighted families; measured 2026-07-23). Self-contained to `grasps.py`:
`isolation_graspable` and all callers use only the boolean.

(b) **`banana` → `horseshoe`** (`shapes.py` `_build`): a blocky, right-angled C — vertical spine +
two **equal-length** prongs, opening +x, **symmetric about y=0**, one 8-vertex rectilinear polygon;
prong thickness ≥ `FINGER_WIDTH` (2.5 cm) so a flat finger makes **full-face** contact. Renamed in
`_CONCAVE_FAMILIES`, `_FAMILY_WEIGHTS`, `piginet/glosses.py`, the `certificate.py` symmetry comment,
`demo_grasp_concave.py`, and the tests. Family is **not** model-load-bearing (the net sees geometry
+ the `concave` flag, not the name); the rename only affects PIGINet's text gloss and labels.

(c) **Scope = code + tests + demos only.** Re-collection / vocab / retrain / recompare is required
but **deferred** to the user (see consequences). The demo (`demo_grasp_concave.py`) now reuses
`grasps._contact_runs_on_line` and reads as the fixed-model proof (0/N floating everywhere).

**Consequences.**

- **Monotone-harder, and it invalidates the v2.2 DD2D artifacts.** The fix only ever *removes*
  grasp cells (max finger gap 0.0000 across all 7 families; dumbbell 5.6→4.9 cells/shape), so
  feasible-candidate sets can only shrink and `min_feasible_subset` (the stratum) can only rise.
  The current DD2D **collection + checkpoints + comparison numbers** are therefore stale and must
  not be reported; DD2D must be **re-collected → vocab → retrain → recompare** before any
  label-dependent DD2D number is trusted. Existing collected JSON is stale on **both** axes (family
  name **and** old-model grasp labels).
- **No model/loss/pipeline change**; the SPECTRE architecture, PL loss, F-subset discipline, and
  rollout-based selection are untouched. `certificate.py` handles the rectilinear horseshoe via the
  same exact triangulated decomposition (a blocky C is a simple polygon; if anything easier).
- **Tests:** `test_every_grasp_cell_makes_contact` (all families, gap ≤ tol) +
  `test_horseshoe_grasp_is_full_face` added; the old test asserting floating cells *exist* (the bug)
  inverted; concave-set literal + banana test renamed. Full dd2d + spectre suites **444 pass**, no
  label-count assertion shifted. **Deliverables** (vision-inspected): `out_dd2d/shape_families.png`
  and `out_dd2d/grasp_demos/*.mp4` show the symmetric horseshoe and full-face concave grasps.
  `notebook.md` 2026-07-24 has the numbers.

---

## 2026-07-23 — Adaptive rollout traces persist per-step raw scores + the demoted set

**Context.** The §7 planner inspector needs to show, per candidate, what the *adaptive*
ranker thought versus its static twin — which plans adaptivity promoted or demoted. An
adaptive method re-scores the whole pool after every failure, so there is no single
cached score; the cache held only `{fp, order}`. Three ways to get one: substitute rank
for score (free), run inference live in the notebook, or persist the per-step scores.
Live inference was ruled out by a standing user constraint — **the notebook must never
run inference at load**. Rank-only was viable (an offline `ProofState` replay recovers
the demotions in 0.031 s for all 142 problems, measured), but the user chose the richer
option: rebuild once, read files forever.

**Decisions.**

(a) **Persist the per-step matrix.** `spectre*_adaptive` records gain `step_scores`
(one raw `(K,)` logit row per attempt) and `step_dead` (the provably-dead indices in
force at that step). `fp`/`order` are untouched, so **every published FP number is
unchanged** — verified byte-identical across all 851 `(method, problem)` rows.

(b) **Store raw logits, not the effective row.** The effective row carries a `-1e9`
tried-mask and a `-1e6` demotion offset; those sentinels swamp a rendered score column
and make it unsortable. Raw + `step_dead` is strictly more informative and the effective
row is exactly reconstructible. Demotion is surfaced as its own `demoted@t` column.

(c) **`null` for the model's own mask.** The v2 model masks its failure context, so a
raw row is `-inf` at every already-attempted candidate — not strict JSON. Serialised as
`null`, read back as `NaN`. The non-finite set is exactly `order[:t]` (pinned by test).

(d) **Store `step_dead` rather than replaying it.** The replay is cheap and provably
matches (0 mismatches vs an independent offline `ProofState` run over 142 problems), but
storing removes any chance of the notebook's copy of the rule drifting from the rollout's.

(e) **`score_pool` / `argmax_in_pool` split, leaving `select_next_skeleton` alone.**
The v1 trace needs the score row from the *same* forward pass. `select_next_skeleton` is
public API on the training hot path (`eda.spectre_evaluate`), so its signature is
unchanged and it is now the composition of the two new helpers. `score_pool` returns the
**unmasked** row (`-inf` is not JSON-representable); the mask moves into
`argmax_in_pool`. Behaviour-preservation is pinned by
`test_select_next_skeleton_matches_score_pool_argmax`, not asserted.

(f) **`deployed_rollout_traced` supersedes the `return_trace` flag** (added 2026-07-20,
one caller), mirroring the `spectre_evaluate` / `spectre_evaluate_traced` precedent so
`deployed_rollout` keeps a clean `-> int` contract. `_observed_blocked` is now the public
`observed_blocked`.

(g) **`ad.score` = score at the step the candidate was picked**, not at the final step.
Under (c) the final row is blank for every attempted candidate, which would empty the
most interesting rows; score-at-pick is the opinion the rollout acted on and is available
for 6032/6032 attempted candidates. This also removed the need for a step slider.

**Consequences.** +11 MB cache, ~1 min rebuild per family. `AdaptiveTrace` +
`load_adaptive_trace` / `load_static_scores` are back-compatible single-problem
accessors (legacy records → `step_scores=None`, and the notebook degrades to rank-only).
§7 gained prev/next problem navigation (`mo.state`, split cells so the buttons and the
dropdown stay in sync), an all-methods overview independent of the method dropdown, and
an in-notebook scene renderer — the vendored `render.py` bakes 6 pt labels at 100 dpi
*and* drops polygon holes (so the wall-band frame paints over the drawer), neither of
which a wider raster can fix. Tests +8 (331 green); no new pylint violations
(`evidence.py` 9 → 6). `notebook.md` 2026-07-23 has the numbers. Open: the **3-seed
reproduction** carried over from 2026-07-20.

---

## 2026-07-20 — DD2D comparison notebook: PIGINet-BCE, linear-R² T0, v1+v2 method set (analysis conventions)

**Context.** The DD2D method-comparison notebook (`compare_dd2d_methods.py` + its
`precompute_dd2d_cache.py` / `dd2d_compare.py` backend) was a 4-method, SPECTRE-v1-only
artifact whose cache no longer existed on disk. We recomputed it to include SPECTRE v2.2 and
to make the length analysis interpretable, which forced several analysis-convention choices.

**Decisions.**

(a) **PIGINet is retrained with BCE (paper baseline) and AUPRC-selected**, replacing the prior
PL-listwise `PIGINet_v3`. `weighted_bce` already existed as the paper loss; a new
`train.py --select {rollout_fp,auprc}` flag lets a full run early-stop/checkpoint on **AUPRC**
(paper-faithful classification selection) instead of the deployment rollout-FP arbiter. The
listwise-PL "no BCE" invariant is a **SPECTRE-model** rule, not a PIGINet rule — PIGINet is the
low-level comparator and is trained as its own paper prescribes. Display name is `PIGINet`
(dropped the `_v3`). BCE run: val AUPRC 0.256 / AUROC 0.658 / rollout-FP 24.87.

(b) **T0 leads with the linear length-R² (= pearson²) on per-episode logits; η² retained as a
secondary.** The user asked for R² over η² for interpretability. A per-episode Pearson r is
affine-invariant, so it already *is* the within-problem-normalized correlation (no z-scoring
needed at that granularity). **Load-bearing caveat, verified on this data:** v1-static has
η²=0.998 but linear R²=0.041 — a *non-monotone* length lookup that a straight-line R² badly
understates. So the notebook reports R² **alongside** η² and the learned-length-curve plot,
never R² alone; the takeaway explains the gap. The §8 length-bias scatter z-scores logits
**per problem** before pooling (that is where cross-problem scale would otherwise confound).

(c) **Comparison set = 6 methods, v1 **and** v2, all 1-seed.** Only `seed_0` checkpoints exist
for both the v1 re-ranker and the v2 `_ov` (observed-demotion) deployed model, so every learned
row is explicitly labelled **[1-seed dev]** — for iteration, not writeup-reportable. Static and
adaptive are two deployment modes of one checkpoint (static = empty-context logits; adaptive =
`evidence.deployed_rollout(..., demotion_source="observed")`). `deployed_rollout` gained a
`return_trace=True` returning `(attempts, order)` so the adaptive realized order is cacheable.
*(⚠️ 2026-07-23: `return_trace` was superseded by `deployed_rollout_traced`, which also
returns the per-step scores and demoted sets — see that entry.)*

(d) **The reader tolerates partial caches.** `dd2d_compare` now iterates SPECTRE *families*
(v1, v2) and **skips a family whose dirs are absent** (a v1-only or v2-only cache still loads);
the base `astar`/`piginet` dirs still raise the helpful "build the cache" error.

**Consequences.** New notebook sections **§7 planner inspector** (scene render via
`spectre_geometry.reconstruct_scene` → `dd2d/render.render_scene`, + a paginated ordered-plan
table) and **§8 length-bias explorer**. `length_fit` exposes `pearson`/`r2` (added to
`_FIT_KEYS`); cache subdirs `spectre2_static`/`spectre2_adaptive` + `piginet` (was `piginet_v3`);
`SEEDS=[0]`. Tests updated + added (`test_dd2d_compare.py` r2/pearson + v2-family + graceful-skip;
`test_facts_evidence.py` `deployed_rollout(return_trace=True)`); full spectre suite green (323).
PIGINet deps (`open_clip_torch`, `torchvision` cu130, `scikit-learn`) installed; torch stays
`2.13.0+cu130`. `notebook.md` 2026-07-20 has the numbers. Open: **3-seed reproduction**.

---

## 2026-07-19 — Demotion signal is a flag (default observed); geometry predicate is opt-in, quantified at ~14%

**Context.** The proof-demotion `dead` feature (`candidate ⊆ an observed-blocked set`) needs a
"blocked" signal. It was sourced from the harvested `blocked-at-contents` fact, which runs a
DD2D grasp predicate (`target_blocked_after_removing`) — the same geometry as the rejected
`clears`. Question: can this be made hard-coding-free? First-principles decomposition: the rule
needs (1) *is S blocked?* — domain-specific — and (2) *is C ⊆ S?* — a domain-agnostic set op.
Only (1) is the issue, and it can be **observed** from the refiner (`failure_action="retrieve"`
= all removals ran, target still ungraspable), no geometry. The net cannot *learn* the sound
rule itself (set-containment is a universal-AND attention approximates poorly, and soundness
needs the exact test — empirically the net given blocked *tokens* learned crude "prefer
longer"), so (2) stays a computed domain-agnostic feature the net learns to weight.

**Decision.** `demotion_source` is a **flag, default `observed`**:
- `observed` (default, generalizable): blocked ⇐ the refiner's own `failure_action`; only per-env
  knowledge is a one-line declaration that the retrieval precondition is removal-monotone. Runs
  on any env with a failure-reporting refiner.
- `computed` (opt-in): blocked ⇐ the harvested geometry fact. Adds **counterfactual** demotions
  (subsets whose plan failed earlier, e.g. at extraction, that observation can't recover).

Threaded through `build_v2_example` / `SpectreV2Dataset` / `TrainV2Config` / the `--demotion-
source` CLI (checkpoint dir suffix `_comp`); `evidence.deployed_rollout(...,
demotion_source=...)` is the reusable deployed ranker (model scores + proof-demotion).

**Consequences.** Quantified the geometry predicate's worth: **~14%** (1-seed deployed: observed
18.22 vs computed 15.99; both beat hand-rule 23 / default 34, tie s1, win s3). So the predicate
is load-bearing (counterfactual reasoning), not merely convenient — but the method does **not**
require it: the observed default is essentially sound (1/6376 edge case) and hard-coding-free.
Two checkpoints kept (`checkpoints_v2_evidence_prior_ov` observed / `…_ov_comp` computed), each
recording `demotion_source` in its cfg. `notebook.md` 2026-07-19 has the table. Verdict: the
generalization claim is now backed by a number on this env, before touching a second one.

---

## 2026-07-19 — v2 ranker: fix the length bias generalizably; no hand-crafted per-env predicate; consume proofs structurally not as tokens

**Context.** The in-distribution main table exposed two problems with the v2 learned rankers:
they *lost easy strata* to default-order/hand-rule while winning s3, and the typed-evidence
pathway *harmed* s1. Root cause: the static ranker uses **plan length as a feasibility proxy**
(corr(logit,length)=+0.42; within-length AUROC ≈ chance on s1/s2, good on s3) — correct on s3
(long plans needed), wrong on easy strata — because the hard s3 episodes dominate the PL
gradient. A `clears` predicate (does removing subset S unblock the target) fixes it decisively
but is a hand-crafted per-environment geometry predicate.

**Decisions.**

(a) **Reject `clears` as a model input/foundation.** Performance must not depend on finding a
bespoke predicate per environment (user directive) — that is the opposite of a generalizable
method. `candidate_clears` and the clears-first baseline were removed. (The a-priori clears
*heuristic* is a striking finding — clears-then-index gets 7.4 overall on DD2D — but it stays
out of the method.)

(b) **Default-order / short-first prior is allowed and generalizable.** `[−index/K,
−len/max_len]` are domain-agnostic planner signals present in any TAMP problem (enumeration
order, plan length), fed as an **additive residual with init-toward-prior** (an untrained
prior-model ranks exactly as default-order; corr(logit,index)=−1.0). Distinct from `clears`,
which requires env geometry.

(c) **Within-length PL loss (`loss.within_length_pl_loss`).** The global top-1 PL is
minimizable by a length shortcut; restricting the listwise objective to same-length buckets
removes length as a cue and forces geometry (within-length AUROC → 0.66/0.75 on s1/s2). Plan
length is universal, so this is domain-agnostic. Additive to — not a replacement for — the
global PL (the load-bearing PL invariant holds; within-length is a *bucketed* PL, still
listwise, no BCE).

(d) **Rollout-based, difficulty-normalized checkpoint selection.** Select by mean
`first-feasible-rank / random-baseline-rank` on val at t=0 (the §5-mandated rollout selection
the v2 training had dropped for val PL loss). Per-episode normalization stops the many-attempt
hard episodes from dominating selection — that domination is what let the length-shortcut
checkpoint win. No stratum, no per-env feature.

(e) **Consume proofs structurally (demotion feature), hints as tokens — the fix for "evidence
harms".** Clean facts-on/off showed `blocked-at-contents` was consumed *crudely* as "prefer
longer" (helps s2/s3, destroys s1: +13.5 attempts). Fix: a **`dead` overlap feature** — a
candidate whose action-set ⊆ an observed-blocked set is provably also-blocked (sound
proof-demotion, a domain-agnostic *set relation*, not a geometry predicate) — plus a mild
Jaccard-with-failed hint; and **route proof-tier facts out of the learned fact-token pathway**
(`_fact_arrays` keeps hint-tier only), keeping the unsound "blocked ⊊ subset ⇒ prefer longer"
cue **out**. After this, evidence helps at every stratum (s1 −0.30, s2 −4.66, s3 −6.71).
`N_OVERLAP=2`, `N_PRIOR=2`; `use_prior`/`use_overlap`/`within_length_weight` config flags;
checkpoints under `checkpoints_v2*_prior[_ov]`.

**Consequences.** The learned model is the best in-distribution method overall (17.4 vs
default 34.1 / hand-rule 23.0, 1-seed), ties s1, wins s3 handily; evidence is now a genuine
help everywhere. Loss invariant amended: "listwise PL only" now reads "listwise PL (global +
within-length buckets) only; no pointwise BCE." Open: **s2** still lags default (36.9 vs 18.4)
— residual cross-length length bias where over-removal fails packing — and the 3-seed
validation (these are 1-seed dev numbers per the fast-iteration directive). `notebook.md`
2026-07-19 has the table + per-stratum facts-on/off.

---

## 2026-07-19 — Step-11 typed evidence: offline geometry-grounded harvest + metadata hints; the learned pathway is a composable increment

**Context.** Step 11 needs typed post-mortem facts on the records for the learned evidence
pathway (fact tokens in the scorer). The §6.2 harvest (`harvest.py`) computes facts from a
failed refinement's `RefineResult.bound_plan`, which the definitive collection deliberately
did **not** persist (2026-07-19 decoupling). Two ways to recover them offline: re-refine each
fail (needs a faithfully-regenerated scene — the exact bug the reconstruct-not-regenerate rule
forbids), or reconstruct facts from stored geometry + stored metadata. Also open: what counts
as a *hint* the net can learn from at λ=0.8, where grasp-witness (universal-blocker
intersection) turned out rare.

**Decisions.**

(a) **Harvest geometry-grounded facts by reconstruction, per failed skeleton, not by
re-refinement** (`envs/dd2d/spectre_harvest.py`, built on `spectre_geometry.reconstruct_scene`):
blocked-at-contents (proof, `target_blocked_after_removing`), grasp-witness (hint,
`grasp_witness_after_removing`), pack-impossible (proof, certificate). All are pure functions
of `(stored geometry, staged subset)`, so they inherit the 0/6622 label-consistency of the
reconstructor and need no scene regeneration.

(b) **Recover the abundant hint from the *stored* `refiner_metadata`, not from re-refinement.**
The collection already persisted `failure_action` (e.g. `pick(o11)`), `n_attempts`, `steps_bound`
per fail. So `extraction-failed(item)` (hint: which item's extraction stalled) and
`pack-exhausted(subset)` are read straight off the record — no `bound_plan` needed. At λ=0.8,
`failure_action` is ~93% `pick` → extraction-failed is the dominant, information-rich hint;
grasp-witness fires ~5% (universal blocker rare for a pincered target). This is what makes P5
non-trivial: without the metadata hints the hint tier would be near-empty on-distribution. The
refiner-trace-only constructive facts (extracted-ok / packed-ok) that truly need `bound_plan`
are deferred — the pathway rides entirely on reconstruction + stored metadata (P-F: these are
observations of genuine attempts, not exact computations relabeled as features).

(c) **Certificate off in the definitive harvest.** At λ=0.8 it proves 0 pack-impossibles
(extraction-dominated, 2026-07-18) and costs ~0.5 s/fail, so `spectre_harvest.py` defaults it
off; `--run-certificate` re-enables it at tight λ.

(d) **canonicalize remaps `post_mortem` fact args.** Facts carry object *identity* (their args),
which must bind to the same episode-local tags as the scene/candidate tokens; `canonicalize_episode`
now renames fact args alongside `scene_geometry`/`aux_labels` (outcomes were previously passed
through unchanged — safe only before facts referenced objects).

(e) **Fact tokens are additive; the static path is byte-identical.** `SpectreV2Batch` gains
trailing-`None` `fact_*` + `avail_mask`; with no facts the scorer/loss are the Step-9 static
model exactly. Training samples an F-context (heavy at |F|=0) with **evidence dropout** so the
static pathway must stand alone (P-D). Model selection stays the **static** val PL loss (t=0 is
the deployment start); the **live scramble gauge** (identity-scramble logit sensitivity) is the
facts-are-used detector, recomputed on `best.pt` at eval (the training-log gauge is the
final-epoch model).

**Consequences.** P5 PASSES (`notebook.md` 2026-07-19): scramble gauge 0.091±0.100 (nonzero),
evidence increment +6.22 CI (4.15, 8.43), v2-evidence beats untyped LAZY by +31.57 CI (14.15,
48.74) on strata≥2. The margin decomposes LAZY 71.1 → static 45.6 → evidence 39.5 — the
representation dominates, typed evidence is the **secondary composable increment** the pivot
predicted. New modules `facts.py`, `evidence.py`, `envs/dd2d/spectre_harvest.py`,
`experiments/spectre/{spectre_harvest,spectre_eval_p5}.py` + `test_facts_evidence.py`,
`test_spectre_harvest.py`. The registered *shift* test (larger increment under held-out shape
families) is Step 12.

---

## 2026-07-19 — Post-hoc geometric proofs reconstruct from stored geometry, never regenerate the scene

**Context.** Step 10's hand-rule P4 gate needs a `blocked-at-contents` grasp proof for
*counterfactual* candidates — subsets not attempted in the collector's own rollout order —
so the proof cannot come from the persisted post-mortems alone; it must be *computed*. The
first implementation regenerated the `DrawerScene` from its stored `problem_seed` and ran
the env's grasp check on it. But the DD2D generator is parameterized (`crowd`,
`require_subset`, `min_subset`, `unblocked_target`, `lam`, `n_items`) and those params were
**not** stored per-episode, so the eval *inferred* them from the outcome. Any inference miss
makes the generator's rejection-sampling path diverge → a different scene with the *same
object names* but *different poses*. It passed the `item_names` guard yet its grasp geometry
disagreed with the collected feasibility labels, so a sound proof-demotion (which can only
ever help — it demotes provably-blocked candidates, never the feasible one) produced a
**spurious negative ΔFP** and "P4=no". The bug was invisible because the name-check looked
like a soundness guard but only checked identity, not geometry (`notebook.md` 2026-07-19).

**Decision.** Every post-hoc geometric query over a collected episode is computed by
**reconstructing** the geometry from the record's stored `scene_geometry`, never by
regenerating the scene. New module `envs/dd2d/spectre_geometry.py`
(`target_blocked_after_removing`, `reconstruct_wall_band`): item footprints are
`place_polygon(stored_boundary_i, stored_pose_i)`, the wall band is the fixed
`WALL_BAND`=1.5 cm frame rebuilt from the stored drawer `W`/`D`, and the grasp test is the
env's own `has_grasp`/`grasp_cells` (a pure function of `shape.polygon` = the stored
item-frame ring). This is exact up to the 4-decimal storage rounding — far below
grasp-clearance tolerances. Rejected: (a) regenerate-and-guess-params (the bug); (b) persist
the full generation params to enable faithful regeneration (still redundant work, and the
poses are *already on the record*); (c) cache regenerated `DrawerScene` pickles (caches the
wrong thing — a re-derived scene, not the labeled one). The record already **is** the cache.

**Why it is correct, not just cheaper.** Because the reconstruction runs on the *same poses
the labeler used*, a proof can never contradict a label. Acid test: over all 284 val+test
episodes, **0/6622** feasible subsets reconstruct as `blocked` (a feasible subset opens the
target by definition, so it must never be flagged blocked). The proof-demotion soundness
telemetry (fraction of demoted candidates that later succeed) is therefore 0 *by
construction*, not merely empirically.

**Consequences.** `spectre_handrule_p4.py` rewritten off the reconstructor: all 142 test
episodes usable (was 126 after name-mismatch drops), and P4 flips from a spurious "no"
(ΔFP<0) to a decisive **PASS** (ALL ΔFP +11.08 CI (7.77,14.73); strata≥2 +23.83 CI
(17.80,30.06)). New `test_spectre_geometry.py` pins reconstruction == the live grasp check
across seeds/subsets and the wall-band equality. **This is the standing rule for Step 11's
typed-evidence harvest** and any future post-hoc proof/hint computation — reuse
`spectre_geometry`, do not regenerate. (Renamed the Step-7 `tests/.../test_tags.py` →
`test_object_tags.py` to clear a pytest basename collision with the RT2D
`envs/routedtransport2d/test_tags.py` surfaced when the full suite collects both dirs.)

---

## 2026-07-19 — Correct λ* to 0.8 (in the designed operating range); 0.5 was off-design

**Context.** The 2026-07-18 G0 entry selected λ*=0.5 by maximizing the oracle−GBDT_wl gap
over a sweep {0.8, 0.65, 0.5, 0.4}. But DD2D's **designed operating range is λ ≈ 0.7–0.95**
(the loose, naturalistic regime); the sweep ran *below* it and the selection rule had no
range constraint, so it picked an off-design λ=0.5. The symptom surfaced in the definitive
collection: at λ=0.5 **stratum-3 is nearly ungenerable** (a min-feasible-subset-3 that also
*packs* into a too-tight buffer is rare → ~18 h for a balanced 125), because λ=0.5 is
tighter than the environment was built for.

**Decisions.** (a) **λ* is constrained to the operating range** (`choose_lambda_star` gains
`operating_range=(0.7, 0.95)`; a tighter λ is rejected even if it maximizes the gap — with a
regression test). (b) **Re-swept in-range** {0.7, 0.8, 0.9, 0.95} (80/40 scenes): G0 passes
across the range — within-length GBDT AUROC 0.68 / **0.539** / 0.580 / 0.534, oracle
0.97/1.0/1.0/1.0. (c) **λ* = 0.8.** The auto-rule (max gap) picked 0.95 by 0.005 — noise at
40 val scenes. λ=0.8 is chosen instead: it is the **design default where the original
balanced dataset (incl. stratum-3) was collected** (so s3 generability is confirmed), has
the **highest feasibility rate (31.4%)** and a little more packing structure than the
near-empty λ=0.95 (marg 1.7% vs 0.4%), while its within-length degradation (0.539) ties
0.95. (d) The definitive collection is re-launched at **λ=0.8, tb=4** (tb=4 for consistency
with the G0 determination; at λ=0.8 packing is easy so tb barely affects labels).

**Consequences.** The λ=0.5 collection was killed (kept 0 useful before the diagnosis; s0/1/2
were filling but s3 stuck). λ*=0.8 resolves the s3 rarity and keeps the benchmark in-design.
The G0 *finding* (cheap stats fail within-length while the oracle solves ⇒ a rich
representation has headroom; size-control mandatory) is unchanged — it holds across the whole
in-range sweep. `notebook.md` 2026-07-19 has the in-range table.

---

## 2026-07-19 — Decouple post-mortem harvest + label-hygiene from the collection (Step 5)

**Context.** Step 5 as planned bundled the typed-fact harvest, the §6.5 certificate
label-hygiene, and the definitive 500/100/100 collection into one pass (collect once with
everything). But the harvest's `pack-impossible` fact and the label-hygiene both call the
§8.4 certificate per failed-skeleton subset — up to 5 s each — and the collector refines
k=200 skeletons per problem over 700 problems. Running the certificate in that hot path
would add many hours and risk the multi-hour collection.

**Decision.** Decouple. (a) The **harvest** (`harvest.py`) and **soundness registry**
(`soundness.py`) are built and unit-tested now, but are **not** wired into the collector.
(b) The **definitive collection** persists geometry (via `record_ext`/converter, Step 3)
+ refiner binary labels — exactly what Steps 6–10 need — with **no certificate in the
loop**. (c) The certificate **label-hygiene** (stamp each fail as proven-infeasible vs
marginal) and the **post-mortem harvest** run as a **controlled offline pass** before the
Step-9 training numbers / Step-11 evidence pathway: a DD2D scene is deterministic in its
seed, so the pass regenerates the scene, re-refines each fail (recovering `bound_plan`),
and harvests — with a per-*subset* certificate cache and a tunable budget, none of it on
the collection's critical path. This is not a second collection (no re-generation of the
dataset); it augments the existing records.

**Consequences.** The multi-hour collection stays fast and is launched now (Step 5b);
`post_mortem`/`aux_labels` on the records are populated by the offline pass (built just
before Step 11). Rationale recorded so the split between "collected" and "harvested"
artifacts is not mistaken for missing data. No change to the harvest algorithm or the
schema. `notebook.md` 2026-07-19.

---

## 2026-07-18 — Gate G0 passes at λ*=0.5; size-control is mandatory (v2.2.1 Task 1)

**Context.** G0 (§10.2) is the pre-model off-ramp: does DD2D have a λ where cheap
statistics degrade but the oracle solves? The cheap probes are slack ordering and a
pairwise-features GBDT. First pass measured *overall* per-candidate feasibility AUROC and
found the GBDT reaching 0.90 at tight λ — which reads as an off-ramp (cheap stats explain
feasibility). But DD2D feasibility is length/count-dominated (the v1 snapshot's central
finding), so an overall AUROC is inflated by |S|.

**Decisions.** (a) **G0 is judged on the within-length (size-conditional) AUROC**, not the
overall AUROC. Controlling for |S|, the GBDT is near chance (within-length AUROC 0.58–0.65
at λ∈{0.5,0.65,0.8}) with its top permutation-importance always a size correlate
(`pair_area_complementarity`/`sum_area`) — cheap stats capture length/area, not subset
identity. This is the "area is the new length" trap (§10.3) surfacing at the G0 stage;
size-control is the fix and is now the standing rule for any "representation beats cheap
stats" claim on DD2D. (b) **λ* = 0.5**, chosen to maximize the oracle−GBDT_wl gap among λ
that both degrade (GBDT_wl < 0.65) and solve (oracle ≥ 0.5): at λ=0.5, oracle 1.00 −
GBDT_wl 0.578 = 0.422 (the largest), packing binds (15.7% marginals, certificate-proven
infeasibles present), and it is not so tight that area-stats win (unlike λ=0.4, GBDT_wl
0.803). (c) **G0 uses on-the-fly generation, no persisted collection**; the definitive
500/100/100 collection at λ*=0.5 is folded into Step 5 (which also adds post-mortems), so
DD2D is collected once, not twice.

**Consequences.** G0 PASSES → proceed to the model steps. `g0.py` (probes + within-length
AUROC + λ* rule) + `spectre_g0.py` (parallel sweep) + `test_g0.py` (7). slack ordering
fails everywhere (AUROC ≈ 0.5), so the §10.3 ladder's "beat slack" bar is trivial; the
real acceptance bar is the within-length residual. `notebook.md` 2026-07-18 has the table.

---

## 2026-07-18 — v2.2.1 schema geometry/evidence layer: additive + migration shim

**Context.** v2.2.1 needs ground-truth object geometry (for the geometry-aware model) and
typed post-mortem evidence carried on the episode records, without breaking the many
existing RT2D/kinder `EpisodeRecord` pickles or the abstract-first pipeline.

**Decisions.** (a) **Additive, trailing, nullable fields** — `EpisodeRecord.{scene_geometry,
aux_labels}`, `OutcomeRecord.post_mortem`, plus new frozen dataclasses (`SceneGeometry`/
`ObjectGeometry`/`ContainerGeometry`/`Fact`/`PostMortemRecord`/`AuxLabels`) — all default
`None`/empty, so every existing construction site and RT2D/kinder record round-trips
unchanged. New invariants I5/I6 are **guarded** (fire only when the field is present).
(b) **Load-time migration shim over global regeneration.** Frozen-dataclass pickles restore
via `__dict__` and skip `__init__`/`__post_init__`, so pre-v2.2.1 pickles lack the new
attrs (→ `AttributeError`). `io.load_episode` fills the defaults via `object.__setattr__`.
Chosen over bumping a schema version + regenerating all corpora: RT2D data lives only on
other machines and need not be re-collected; DD2D is re-collected regardless (for
post-mortems). (c) **Geometry is a converter/`record_ext` change, not a raw re-collection**
for the *geometry* part — the DD2D JSON already had pose/shape; `record_ext` now also writes
the item-frame `boundary` ring and `spectre_convert` reads it (`CONVERTER_VERSION`
`v1→v2`). A dir predating `boundary` yields `scene_geometry=None` (abstract-only). The
**abstract STRIPS state stays x0-free**; geometry rides on `scene_geometry`, never in the
atoms.

**Consequences.** On-disk format grew (optional) fields; `test_schema_v2_geometry.py`
covers round-trip + I5/I6 + legacy-pickle migration; the `dd2d_convert` version-pin test
updated to v2. No change to the model/loss/training yet — this is the data-layer
foundation Steps 5 (post-mortem population) and 8 (geometry model) build on. `notebook.md`
2026-07-18.

---

## 2026-07-18 — DD2D arrangement-complete negative packing certificate (v2.2.1 Task 0)

**Context.** v2.2.1 makes completing the arrangement-complete negative certificate the
**blocking Task 0**: until packing-infeasibility can be *proven* (not left provisional
`marginal(budget)`), no label-dependent DD2D number is trustworthy. The spec is
`dd2d_spec.md` §8.4 + P16/P19. The repo already contained an NFP/IFP/nesting packing
substrate (`ttd/ttd_core/`), but the user directed that **ttd is scrapped** (still in the
tree only because it hasn't been removed) and must not be reused.

**Decisions.**

(a) **Build the certificate from scratch on Shapely** (`envs/dd2d/dd2d/certificate.py`),
not on `ttd_core`. Rejected reusing `ttd_core.nesting`: user call, and independently its
`INFEASIBLE` is discretization-relative (fixed 1°/5° grid, not the Lipschitz grid) and its
`packs()` folds `TIMEOUT` into `False` — unsound as a proof.

(b) **Exact convex decomposition via `shapely.constrained_delaunay_triangles`** (Shapely
2.1+). Each triangle ⊆ the shape and the triangles exactly cover it → exact NFP for the 3
**concave** families (`banana`/`shoe`/`dumbbell`). Rejected the plain-Delaunay-of-vertices
fallback (what sank ttd's concave path): it need not respect a reflex boundary → NFP wrong
→ possible false infeasible. `convex_parts` verifies exact cover and refuses (→ marginal)
otherwise.

(c) **All placement orders, not a fixed order.** A single fixed-order sequential DFS is
**not** sound: the first item's free region is the whole IFP, whose only vertices are the
container corners, so interior-only packings are unreachable → false infeasible.
Soundness argument used instead: bottom-left-compact any packing; the most-bottom-left
item is pinned into a container corner (an IFP vertex) and inductively each item in BL
order lands on a free-region vertex — so the BL order (∈ all orders) reaches it. We only
attempt the full all-orders exhaustion for `|S| ≤ MAX_ORDER_ITEMS = 5`; larger subsets no
area bound settles fall to `marginal`, never a partial-search `infeasible`.

(d) **Remove the Brunn–Minkowski area term; keep H1 on exact deflated areas.** A
`Σ(√Aᵢ − (δ/2)√π)²` bound was added then removed: for fixed original area the disk
*maximises* eroded area (isoperimetric), so that expression is an *upper* bound on the
deflated area → it overestimates packed area and **fabricates infeasibilities on tight
buffers**. Since the DFS already computes the exact δ/2-deflated polygons, H1 on their
exact areas is the tightest sound area bound. Process lesson recorded: the
zero-false-infeasible battery **must include tight/near-threshold cases** — a loose-only
battery hid this bug.

(e) **INFEASIBLE only on full exhaustion; timeout ⇒ `None` (marginal, reason=budget),
never infeasible.** Budget = P19 (5 s / 1e5 EGEs). A degenerate δ/2-deflation (thin shape
vanishes) also ⇒ `None`. The verdict is three-valued: `True` (proven infeasible), `False`
(a packing was found ⇒ not infeasible), `None` (undecided → stays marginal).

(f) **Integrate behind a `use_certificate` flag, default off.** `label.py`'s
`label_candidate`/`label_all` gain the flag; it is off inside `generate_dd2d_problem`'s
rejection-sampling loop (where the certificate is called hundreds of times and only the
feasible labels — unaffected by it — drive strata/F3) and on only for authoritative
once-per-candidate labeling. On-by-default hung the DD2D suite. On a `True` verdict the
`marginal(budget)` becomes proven `infeasible(packing)`; on `False` (a packing exists) it
is reclassified `marginal(inaccessible)`; `None` stays `marginal(budget)`.

**Consequences.** Sound: 0 false-infeasible over ~730 constructed-feasible packings (loose
+ tight, concave + circles, |S|=2–4); 16 new `test_certificate.py` tests + 49 DD2D + 259
spectre tests green. At λ=0.8 the certificate proves 0 packing-infeasibles (infeasibility
is extraction-dominated at loose λ) and reclassifies all budget-marginals it saw to
`inaccessible` (they pack). The real-scene *tight-λ* proven-infeasible characterization is
deferred to **Step 4**'s λ-sweep (generation at tight λ is slow). No change to the SPECTRE
model/loss/pipeline; the certificate is a labeler-side soundness upgrade. Applying it to
stamp the SPECTRE training labels (refiner-`fail` outcomes) at collection time is wired in
**Step 5**. `notebook.md` 2026-07-18 has the numbers.

---

## 2026-07-18 — Modernize + pin the substrate deps so a fresh machine resolves

**Context.** Development moved from a MacBook M3 Pro (CPU/MPS) to a new Ubuntu
26.04 workstation (RTX 5090, Ryzen 9 9950X, 64 GB) for GPU training. A fresh
`uv pip install -e ".[develop,ttd]"` on the new box **failed to resolve** — the
root `pyproject.toml` pinned `kindergarden[kinematic2d]==0.0.8` but left the
`kinder-baselines` and `bilevel-planning` git sources **unpinned** (no `rev=`).
Both pins were introduced together on 2026-03-22 (`62d3784`) when compatible, but
upstream then drifted: kinder-baselines bumped to `kindergarden>=0.1.0`
(2026-04-29) and later `bilevel-planning>=0.1.4`, and **dropped the
`kinematic2d` extra** entirely (kindergarden 0.2.0 has no such extra). With no
lockfile, a fresh resolve pulls the drifted HEAD and conflicts; the MacBook only
still works because its venv was resolved months ago and cached — never
re-resolved. User chose "modernize + pin" over reproducing the MacBook's exact
(unpinnable, un-frozen) set or pinning to the ~4-month-old compatible commit.

**Decisions.**

(a) **Bump the whole prpl-mono substrate to one coherent commit `e215d1fc`**
(was `df145d5c` for `relational_structs`/`prpl_utils`/`prpl_llm_utils`/
`tomsgeoms2d`; `bilevel-planning` was previously an *unpinned* prpl-mono source).
prpl-mono is one monorepo — mixing commits across its subpackages is what causes
API breakage — so all five move together. `e215d1fc` provides
`bilevel-planning==0.1.4` (satisfies kinder-baselines HEAD). Added an explicit
`rev=` to the `bilevel-planning` source so it can no longer drift against the
pinned `relational_structs`.

(b) **Bump `kindergarden` 0.0.8 → 0.2.0 and drop the `[kinematic2d]` extra.**
The extra no longer exists (kinder-baselines PR #77 "drop dead kindergarden
extras"); kinder packages now depend on bare `kindergarden>=0.1.0`. `pymunk` (the
kinematic2d substrate) is already a direct alphatamp dep, so dropping the extra
loses nothing.

(c) **Pin both kinder-baselines sources to HEAD `4c731dc8`** (was unpinned) —
`kinder-bilevel-planning` and `kinder-models`, for reproducibility.

**Consequences.** Fresh resolve succeeds; **all spectre tests pass** (254 incl.
slow), spectre mypy clean, spectre pylint 10.00/10 — i.e. the substrate bump did
not break spectre. torch is the cu130 build (`2.13.0+cu130`, see the spectre
`CLAUDE.md` compute-resources note), GPU-verified on the RTX 5090 (sm_120). The
`pyproject.toml` `torch` requirement is left **unpinned** on purpose (the cu130
index is applied at install time, not baked in, so SLURM/other machines are
unaffected). **Reproducibility caveat:** prior spectre results/checkpoints were
produced on the older MacBook substrate (kindergarden 0.0.8 / prpl-mono
`df145d5c`); numbers regenerated on this box use the newer substrate, so
re-verify before comparing across the boundary.

**Follow-up (2026-07-18) — restore `run_ci_checks.sh`.** Two further repo-wide
fixes (not spectre-specific) so the CI script runs on a fresh machine:
(1) **capped `pytest>=7.2.2,<8`** in the `develop` extra — the fresh resolve pulls
`pytest 9.1.1`, but the latest `pytest-pylint` (0.21.0) uses the `path` collect
hook removed in pytest 8.0, INTERNALERRORing `pytest . --pylint`; `<8` (→ 7.4.4)
is the working bound (the upstream kinder-baselines `<9.1` cap is *not* enough).
(2) **excluded the untracked `kb/` sibling checkout** (a local knowledge-base clone
of kinder-baselines & friends, with its own `.git`) from git and every CI tool —
`.gitignore /kb/`, `run_autoformat.sh` docformatter `--exclude`, `[tool.isort]
skip_glob`, `[tool.mypy] exclude ^kb/`, `[tool.pytest] norecursedirs`, `.pylintrc
ignore` — mirroring how `.venv/`/`archive/`/vendored-dd2d are handled; otherwise
`black .`/`isort .`/`mypy .`/`pytest . --pylint` descend into it. After these,
autoformat + pylint (217 pass) are clean and spectre stays fully green.

**Sibling-project failures surfaced by the modern toolchain/CUDA (NOT spectre) —
resolved least-invasively to get full CI green (user call: skip, don't deep-fix
other projects' internals):** (1) the **pre-existing** `mypy` error in
`experiments/collect_data.py:67` — `render()`'s return type widened under newer
gymnasium/mypy — fixed by annotating the local `frame: Any` (matches the method's
own `-> Any`). (2) 4 `simfree_param_policy` tests raise a `cuda:0`-vs-`cpu`
mismatch on a GPU box (pass CPU-only) — marked `skipif(torch.cuda.is_available())`
via a shared `_SKIP_ON_CUDA` marker; the real fix is to thread a device through
that approach. (3) `practice_makes_perfect` fails **device-independently** (CPU
too) with `AbstractPlanGenerationError` under the new substrate — marked
`xfail(strict=False)` with that reason; needs a genuine sibling-project fix, not a
skip. **Autoformat churn:** the newer `docformatter` rewraps ~8 tracked docstrings
(committed so the tree stays autoformat-clean; re-verified idempotent). After all
this, `./run_ci_checks.sh` is green end-to-end (mypy 0 / pylint 217 pass / pytest
269 pass, 11 skipped, 1 xfailed) and spectre itself is untouched by these
sibling-only changes.

---

## 2026-07-12 — DD2D integration: JSON→EpisodeRecord converter, not a native env

**Context.** DD2D (Drawer Decluttering 2D) was migrated in-package under
`envs/dd2d/` with an already-collected PIGINet-style dataset
(`data/dd2d/raw_v2/{train,val,test}`, 425/120/124 problems) and its own
generation pipeline (`envs/dd2d/dd2d/collect.py`). Goal: make DD2D usable as a
SPECTRE problem/dataset, keep the ability to generate more, and start training
SPECTRE on it. Key structural fact: SPECTRE's training path
(`dataset.py`→`vocab.py`→`train.py`) consumes *only* serialized `EpisodeRecord`
pickles — the `SesameModels`/gym/refiner machinery exists solely so `collect.py`
can *generate* episodes from a live sim. And each DD2D problem directory (200
`NNN.json` candidate skeletons over a shared objects/init/goal, each with a
feasibility `label`) already *is* a SPECTRE episode.

**Decisions.**

(a) **Converter, not a native SPECTRE env.** Wire DD2D by converting its JSON to
`EpisodeRecord` (`envs/dd2d/spectre_convert.py`), reusing the entire downstream
pipeline unchanged. Rejected building `create_dd2d_models`/gym stub/closed-form
generator/refiner adapter + `collect.py` dispatch branches: DD2D's refiner is a
geometric packing solver that does not fit the controller-sampler contract, and
fresh generation is already served by `envs/dd2d/dd2d/collect.py`. New data =
run that collector → re-run the converter. Far less code, no re-derivation of
DD2D geometry into the substrate.

(b) **Abstract-only for v1 (x₀-free).** The converter keeps only the six drawer
STRIPS predicates and drops the DD2D `at-pose`/geometry literals — SPECTRE is
deliberately x₀-free. Continuous poses/shapes/sizes remain in the source JSON for
a future x₀-conditioned comparator (proposal §6), not wired now. Consequence:
DD2D is expected to be a **negative control** — abstract-first drops exactly the
packing signal feasibility depends on. Confirmed at epoch 0: AUROC(3) < AUROC(0)
(`notebook.md` 2026-07-12).

(c) **One variant `dd2d_v2` spanning all item counts.** DD2D problems mix
n∈{10..13} within a split; the architecture factors across object counts (typed
local ids, set pooling), so a single variant is natural. The single object type
`item` is fully augmentable (target marked by the `target` predicate, not
identity): `env_registry._TYPE_AUG_POLICIES["dd2d_v2"] = {"item": True}`, no
static-tag stream.

(d) **Label caveat (blocking for research numbers, not for training).** DD2D's
Day-1 labeler marks non-area-proven negatives as *marginal*, not
proven-infeasible (`MIGRATION_DD2D.md` §4); the converter maps `label==false`→
`"fail"` for training, but no label-dependent SPECTRE number is reportable until
the arrangement-complete negative certificate lands.

**Consequences.** New: `envs/dd2d/spectre_operators.py`,
`envs/dd2d/spectre_convert.py`, `experiments/spectre/dd2d_convert.py`,
`conf/dd2d_convert.yaml`, `conf/env/dd2d_v2.yaml`,
`tests/approaches/spectre/test_dd2d_convert.py`; one `env_registry.py` entry.
No change to the model, loss, F-subset discipline, or rollout-based selection —
DD2D flows through the exact same `EpisodeRecord` schema as RT2D/kinder. Verified
end-to-end: 669 episodes converted (0 failures), vocab (3 ops/6 preds/1 type,
OOV-clean), pipeline check + 1-epoch train run pass.

The **vendored DD2D env code** (everything under `envs/dd2d/` except the
`spectre_*` adapter files) is excluded from strict `mypy` (`pyproject.toml`
`[tool.mypy] exclude`) and `pylint` (`.pylintrc` `ignore-paths`) via a
`(?!spectre_)` negative lookahead — it arrived from `envsearch` with 100+
pre-existing type errors and is treated like `lib/` vendoring, while SPECTRE's
own adapter stays fully checked. Open follow-up (not decided here): whether to
let `run_ci_checks.sh`'s repo-wide `black .`/`isort .` normalize the vendored
tree once, or exclude it from formatting too.

---

## 2026-06-25 — Direction pivot: from adaptive reordering to a representation question

**Context.** The project's headline had been *adaptive test-time reordering* (the
SPECTRE re-ranker), evaluated on the bespoke RT2D env against the adaptive
baseline B4. Two results undercut that as the lead: the Ψ-ablation (`notebook.md`
2026-06-06) attributes only **~27%** of SPECTRE's margin over B4 to
failure-conditioning — the **static** Φ+σ representation carries ~73%; and the B6
DP-on-counts sweep (`notebook.md` 2026-06-11) showed lookahead over the count
model is **small, fragile, and saturated**, i.e. not the missing ingredient. A
structural reading of RT2D explains why, and motivates a reframe. Each decision
below is recorded with rationale.

**Decisions.**

(a) **Reframe adaptivity-primary → representation-primary.** The contribution is
now a *representation question for plan-feasibility prediction in
fully-observable (FO), deterministic bilevel TAMP*: what should a feasibility
predictor represent skeletons/problems over? Rationale: the empirics put the
margin in the static representation, not the failure-conditioning. (See
`proposal.md` §0.)

(b) **Demote SPECTRE/reordering to a secondary, composable increment.**
Within-episode failures carry free instance-specific signal, but it is a minority
of the margin; treat the re-ranker as orthogonal to — and combinable with —
whichever representation wins, not as the headline.

(c) **RT2D was effectively partially observable *to the policy* — a mislabeling.**
RT2D was described as FO+deterministic, but the policy π was denied x₀ and the toy
three-gate refiner had **privileged access to the latent z**. To the policy the
problem was therefore effectively partially observable, and the discrete gating
latent had to be **manufactured** — which is why RT2D felt contrived. Rationale:
record so we do not re-derive the bespoke env as if it were a faithful FO TAMP
instance.

(d) **The no-x₀ design was a handicap *in RT2D* — but the nuance matters.**
PIGINet's own ablation shows x₀ carries real signal **in their kitchen
problems**, and PIGINet already works at **150–600 problems**, so the
*data-efficiency* rationale for dropping x₀ does not hold *universally*. This does
**not** establish that x₀ must always be included: whether dropping low-level
state is a helpful abstraction is **domain-dependent**, and there may be problems
where it helps. We are **not committed** either way — the x₀ stance is
experiment-driven. Rationale: avoid over-correcting from "drop x₀ always" to "keep
x₀ always"; both are empirical questions.

(e) **The FO information-ceiling bounds the adaptive component's value.** In
FO+deterministic TAMP, the within-episode refinement failures add **no
information beyond x₀** at the predictor's ceiling (the outcome of every skeleton
is a deterministic function of x₀). This is the structural reason the adaptive
signal is small here, and the structural reason for the pivot. Rationale: it
makes the ~27% finding expected, not a defect of Ψ.

(f) **Reinterpret the 27% finding as "the static representation does the work."**
The ablation is now read as positive evidence for the representation thesis: most
achievable gain is captured by the static ranking, with online updating a small
add-on — consistent with (e).

(g) **Adopt the efficiency/representation framing + crossover prediction +
negative control.** The claim is **efficiency / perception-lightness, not
information access**: under FO+determinism no representation beats an ideal
low-level predictor on information grounds. **Falsifiable prediction (a
hypothesis, not a result):** a *crossover* — in the low-data / weak-perception
regime a well-chosen (richer-than-pixels, cheaper-than-full-state) representation
matches or beats a low-level PIGINet-style predictor on downstream planning
efficiency, while the low-level predictor regains its edge with abundant data +
strong perception. **Negative control:** dense-packing / fine-continuous-fit
domains, where any compressed representation is expected to lose, bound the
claim. *Abstract-first* is the current leading candidate but only one point in a
design space (learned latents, object-centric/graph features, intermediate
symbolic+coarse-geometric states, invented predicates), and may prove too lossy.

(h) **Prefer pre-existing environments that meet a hypothesized-advantage
property wishlist; keep bespoke in scope.** We prefer pre-existing envs *only if*
they exhibit properties we expect to favor a relational/abstract representation,
and keep **bespoke, hand-crafted** envs in scope where they better expose the
advantage. The (open, evolving) property list: (1) feasibility governed by
relational structure the abstraction captures; (2) low-level state
high-dimensional/distracting or hard to extract relational structure from; (3)
perception genuinely limited or costly; (4) object-count/identity generalization;
(5) long horizon / large diverse pool. Planned homes: PIGINet kitchens with
degraded perception, and Khodeir clutter/distractor domains augmented with a
low-level baseline, swept over perception-degradation × training-set size.
Primary metric time-to-first-success; secondary time-to-k. Rationale: the
property combination needed for the *adaptive* claim (shared, refinement-decidable,
instance-specific gating) is rare in pre-existing benchmarks, but the
*representation* claim has real pre-existing homes.

(i) **Freeze the April writeup.** `archive/SPECTRE_WRITEUP_APR_2026.md` is frozen
with a banner (2026-06-25); it reflects the adaptive-reordering framing and is
retained as historical record. Rationale: the living docs (`proposal.md` §0,
this log, `notebook.md`) are the source of truth and must not defer to the frozen
snapshot.

**Consequences.** `proposal.md` now leads with §0 (representation-first), with the
original §1–§6 retained byte-unchanged under "Superseded framing (April 2026)";
`research_lit.md` reframes PIGINet as the low-level static predictor we compare
against and adds a representation lens; `notebook.md` 2026-06-25 records the
reinterpretation and the forward sweep. The RT2D env, the SPECTRE model, and the
B1–B6 baselines/code are **unchanged** — this is a framing pivot, not a code
change. No planner/refiner/abstraction change. What survives intact: the
rollout-based model-selection discipline, the PL loss, the F-subset discipline.

---

## 2026-06-11 — B6 higher horizons: incremental scoring, top-m pruning, no capping

**Context.** B6's per-decision cost is `O(K^{h+1})` (the `O(K^{h−1})` backup tree
× the `O(K²)` re-conditioning leaf), and every RT2D-n3 pool is exactly K=30, so
the exact search was ~12 min at `h=3` and intractable at `h≥4`. The goal was to
reach higher horizons (does the lookahead premium keep growing, or saturate?)
without distorting the evaluated problem.

**Decisions.**

1. **Reject pool capping (train or eval) as the tractability lever.** Capping
   the candidate pool to the first `K_cap` planner-ordered skeletons was
   considered and rejected. *Eval-capping* changes the evaluated problem: the new
   `solvability_at_cap` diagnostic shows RT2D-n3 successes sit at **every**
   planner depth (test solvable@15 ≈ 0.46, @20 ≈ 0.60, reaching ~1.0 only at
   k=30), so capping below 30 censors real successes. *Train-capping* (a briefly
   adopted "symmetric cap" idea) deletes estimator observations for no benefit —
   the per-key/pairwise NB estimands are properties of the data distribution, not
   of a skeleton's pool position, so the rising OOV it induced was self-inflicted.
   **The q-model is always fit on the full train pools, and eval always uses the
   full K=30** (uncensored-eval discipline, 2026-06-07). A `candidate_cap` knob
   was *not* added.

2. **Incremental Naive-Bayes scoring is the real lever.** The leaf was secretly
   `O(K³)`: each re-conditioning rollout step appends a failure, so every
   `(candidate, F)` score was a fresh `Σ_{k'∈F}` recompute. The search now threads
   a scoring context (`dp_on_counts._Ctx`) that extends per-candidate
   `S_succ`/`S_fail` by **one pairwise term per failure edge** (`O(K)`), turning
   the leaf into `O(K²)`. Measured on RT2D-n3 (exact, unpruned): h=2 **93 s → 9 s**,
   h=3 **740 s → 86 s**, h=4 from intractable to **~minutes** — all reproducing the
   exact attempt counts bitwise. Scores use `np.log` so the incremental `S_succ`
   equals `_adaptive_score` bitwise, preserving the `h=1 ≡ B4` identity. Synthetic
   test models without the primitives fall back to the recompute closures; an
   equivalence test pins the two paths together.

3. **Top-m pruning kept as an optional knob, off by default.** With incremental
   scoring making the *exact* search tractable through h=4, `dp_on_counts_baseline`
   now **defaults to `m=None` (exact)** — a deviation from the in-flight plan's
   `m=12`, justified because exact is now affordable and pruning is lossy (m=12
   cost ~0.09 attempts at h=3: 9.24 vs 9.15, for only ~2×). Pruning remains
   available (`m=12`) to push `h≥5`: it restricts the `min` at each **internal**
   lookahead node to the top-m candidates by greedy index; the **root argmin and
   the leaf walk are never pruned**, so which skeleton may actually be attempted
   is unrestricted and `h=1` is untouched. Guarded by an `m≥K`-equals-unpruned
   exactness test.

**Consequences.** New `eda.solvability_at_cap` (+ notebook figure gating any
capping). `dp_on_counts.py` rewritten around `_Ctx` (incremental + closure
backends) with a `m` pruning width; `eda._build_dp_model` supplies the
incremental primitives (`log_succ`/`log_fail`/`delta` with a shared delta cache).
Exact h-sweep numbers and paired stats: see `notebook.md` 2026-06-11. Belief-MDP
-over-z and any planner/refiner/abstraction change remain out of scope.

---

## 2026-06-08 — DP-on-counts (B6): lookahead skeleton-selection baseline

**Context.** B4 (Adaptive Historical) is the headline non-learned adaptive
baseline, but it is *myopic*: at each step it picks the single skeleton with the
highest Naive-Bayes success score and never reasons about what the resulting
failure set leaves for later. We wanted a fair, count-only baseline that shares
B4's estimator but adds multi-step lookahead — to bracket how much of any
SPECTRE-vs-B4 gap is "B4 is myopic" vs "B4's count model is weak". The method is
a receding-horizon expectimax over the cost-to-first-success recursion
`V(F) = min_{σ∈R}[c(σ) + q(σ|F)·V(F∪{σ})]`, solved online to depth `h`. It is a
**baseline** (B6), not SPECTRE, and touches no planner/abstraction/refiner.
Several modelling choices were load-bearing and non-obvious.

**Decisions.**

1. **Base-policy depth indexing (h=1 ≡ B4 exactly).** `h=1` is the no-lookahead
   base greedy policy `argmin index(σ|F)`; `h≥2` is
   `argmin_σ[c(σ) + q(σ|F)·W_{h−2}(F∪{σ})]`. The literal one-step backup
   `argmin[c + q·V̂_0]` is a *policy-improvement* of B4, not B4 (counterexample:
   a candidate with marginally lower fail-prob but a much worse continuation gets
   displaced) — so the only way to honour "h=1 reproduces B4 exactly" is to make
   `h=1` the base policy and have `h` count improvement steps above it. Default
   `h=2` = one real lookahead level.

2. **Calibrated two-class NB posterior for `q` (reject `1−clip(exp(score))`).**
   B4's score `S_succ = log p̂(k) + Σ_{k'∈F} log[p̂(k|k')/p̂(k)]` exponentiates to
   an *unnormalized* NB score that exceeds 1 for `|F|≥2`; `1−clip(exp(S_succ))`
   would force `q=0` (a guaranteed success) precisely when conditioning is most
   informative, confounding the h≥2 regime the baseline exists to probe. Instead
   `q(σ|F) = σ(S_fail − S_succ)` with the complementary
   `S_fail = log(1−p̂(k)) + Σ log[(1−p̂(k|k'))/(1−p̂(k))]` — a proper posterior in
   `(0,1)`, no clip. B4's *ranking* still uses the raw `S_succ`, so this does not
   change B4 or the B6 `h=1` selection.

3. **Re-conditioning greedy leaf `W_0 = V^base` (reject the frozen `Σ c·Π q`).**
   The leaf is the true re-conditioning value of the base policy — a
   stationary-greedy rollout that re-selects `σ*` and re-evaluates `q` at each
   step. A frozen leaf (`q` pinned at `F`) is not the value of any policy under
   the re-conditioning dynamics, so the modeled-value monotonicity
   `W_0 ≥ W_1 ≥ W_2` is *not* guaranteed and in fact breaks under positive
   co-failure correlation — which RT2D is engineered to have. With `W_0 = V^base`,
   `W_{ℓ+1} = TW_ℓ ≤ W_ℓ` by policy improvement (`TV^π ≤ V^π` for any stationary
   `π`); since the bound is index-agnostic, the leaf is ordered by the *same*
   index `h=1` uses, keeping leaf-base ≡ `h=1`-base exactly (no nesting wrinkle).
   Cost: the leaf is an `O(K²)` rollout, mitigated by an episode-independent
   `q`/`S_succ` cache plus a call-scoped `W` memo; per-decision is `O(K^{h−1}·K²)`,
   tractable for `h≤3, K≤30`.

4. **`time` objective cost = train per-key mean refine time.** `attempts` uses
   `c≡1`; `time` uses the mean `refinement_wall_clock_s` per canonical key fit on
   train (OOV keys → global mean). Per-skeleton times are logged on
   `OutcomeRecord` but never pre-aggregated, so `_fit_refine_costs` aggregates
   them (mirrors `_fit_marginals`).

5. **Default eval budget = 30 (uncensored).** `dp_on_counts_baseline` defaults
   `attempt_budget=30` (= RT2D-n3 pool cap, the uncensored standard, this log
   2026-06-07) rather than the B1–B5 legacy default of 20, so a direct caller
   does not silently reintroduce censoring. Model selection's
   `val_rollout_attempts` budget (20) is unaffected.

**Monotonicity is a property of the modeled value, not realized rollouts.** The
unit test asserts `W_0 ≥ W_1 ≥ W_2` on the *modeled* value (including a
positive-correlation instance). Realized held-out attempt means need **not** be
monotone in `h` — finite-count `q` and sparser conditioning at large `|F|` make
deeper lookahead optimize against a less reliable model — so the realized
h-curve is reported as a result, never used as a pass/fail gate.

**Out of scope (future work).** A belief-MDP-over-latent-`z` variant: it needs
the scene latent `z` logged per training problem and is an oracle-structured
reference, not a fair count baseline. No refiner/planner/abstraction changes.

**Consequences.** New module `dp_on_counts.py` (env-free search) +
`eda.dp_on_counts_baseline` (B6); B4's NB scorer extracted to the shared
`_adaptive_score` (B4 output unchanged, guarded by existing tests). B6 registered
in `analyze_spectre.py` (h∈{1,2,3} sweep, comparison table/figures, extreme-q
diagnostic). The baseline roster is now B1–B6 (B6 = DP-on-counts; B4 is its
`h=1` special case). Run numbers: see `notebook.md` 2026-06-08.

---

## 2026-06-07 — Analysis notebook converted to marimo (`.py`)

**Context.** The analysis notebook was a Jupyter `.ipynb`
(`experiments/spectre/analyze_spectre.ipynb`, gitignored as scratch). Jupyter's
JSON-on-disk format is opaque to Claude Code — cell outputs are elided and edits
are clumsy — which made iterating on the EDA/comparison notebook with CC
painful.

**Decision.** Convert the notebook to a **marimo** notebook
(`experiments/spectre/analyze_spectre.py`): a pure-Python, text-first format
(cells are `@app.cell` functions) that CC can read and edit directly. The new
`.py` is the canonical analysis notebook and is **tracked**. Both files are kept
for now — the `.ipynb` stays gitignored alongside it — but forward-looking
"how to run the analysis" references point to the `.py`. Behaviour is preserved:
the marimo notebook reproduces every number and artifact (verified — pool-cap
1.000, overlap 0.973/OVERLAPPING, SPECTRE mean-attempts ≈ 5.67, §6 verdict
FAIL on 3.4, plus the SVG/PDF exports). Data root now resolves relative to the
notebook file (`mo.notebook_dir()`), so it runs from any launch directory rather
than only from `experiments/`.

**Keeping it out of CI.** A marimo `.py` does not satisfy `mypy`/`pylint`, so it
is excluded: `[tool.mypy] exclude` regex and `.pylintrc ignore-patterns` both
skip `analyze_spectre.py`; `black`/`isort`/`docformatter` still format it
(marimo files are black-compatible). `marimo` added to the `develop` extra so
the tracked notebook is runnable from a dev install. `run_ci_checks.sh` is
unaffected (verified: mypy reports no issues, pylint skips the file).

**Consequences.** marimo's single-definition dataflow rule forced a few local
renames during conversion (shared plotting temporaries `_`-prefixed; the
`_color_tag` helper promoted to a shared `color_tag`; the styled-table cell split
so its Styler renders, replacing Jupyter's `display(...)`). Historical
references to the `.ipynb` in past records (this log's 2026-06-07 uncensored-eval
entry; `archive/README.md`) are left as-is — they accurately describe figures
generated by the `.ipynb` before this conversion.

---

## 2026-06-07 — Report uncensored evaluation results (attempt budget = pool cap)

**Context.** The RT2D-n3 headline table and figures
(`experiments/spectre/analyze_spectre.ipynb`) were generated with
`ATTEMPT_BUDGET = 30` — equal to the candidate-pool cap — while the living
docs and the writeup described the evaluation attempt budget as 20. At budget
20, ~2–4% of episodes hit the cap and are censored to 21 (budget + 1); at
budget 30 the budget never binds (pool ≤ 30), so every episode runs to its
true first-success attempt and nothing is censored. The frozen-context
ablation (notebook 2026-06-06) surfaced the discrepancy: the full-variant
mean only reproduced the headline at budget 30, not 20.

**Decision.** Headline / reported evaluation metrics use the **uncensored**
budget — attempt budget = candidate-pool cap (30 for RT2D-n3) — so reported
attempt counts are the true time-to-first-success with no censoring. An
uncensored distribution is more informative than a censored one, especially
in the tail: it shows where any method (SPECTRE included) does badly rather
than collapsing those episodes to a single censored value. This is the
standard for SPECTRE evaluation tables going forward; if a future env's pool
cap differs, the eval budget tracks that cap.

**Scope — what this does NOT change.** Model selection and early stopping stay
on `val_rollout_attempts` at its own rollout budget (20) — a separate knob
from evaluation reporting (selection picks the checkpoint; this decision
governs how the chosen checkpoint is reported). The rollout-metric discipline
(proposal.md §5) is untouched.

**Consequences.** Writeup §Training and `archive/README.md` corrected 20 → 30
(the writeup's headline numbers were always budget-30; only the stated budget
was wrong). Pending reconciliation in the same commit: the "attempt budget 20"
phrasing in `proposal.md` §1 and the spectre `CLAUDE.md` headline line, which
refer to the evaluation/reporting budget and should read 30 (uncensored) — the
`val_rollout_attempts` mentions (budget 20, model selection) stay as-is.

---

## 2026-06-06 — Documentation discipline codified in CLAUDE.md

**Context.** The instruction to keep the living docs updated was a single
passive bullet ("Record run outcomes in notebook.md; lasting decisions in
decisions.md; method changes in proposal.md") — where but never when. It
demonstrably failed: `notebook.md` stayed empty for ~2 months of training
runs, every pre-refactor ADR below was reconstructed retroactively, and the
stale AUROC-as-key-metric claims survived ~6 weeks after the rollout-metric
change.

**Decision.** The spectre `CLAUDE.md` gains a "Documentation discipline"
section: a change-type → doc → format routing table (run numbers — including
negative results — → `notebook.md`; lasting choices → `decisions.md` ADR;
method/pipeline/protocol changes → `proposal.md` in place + §6 reconcile), a
before-commit rule (the doc entry ships in the same commit as the change),
a materiality threshold (mechanical refactors/formatting/typos exempt), and
a litmus test ("in 3 months, will we know this happened and why?").

**Alternative rejected.** Mechanical enforcement via a Claude Code hook:
project-level `.claude/settings.json` is committed and would fire for every
user of the shared monorepo, not just spectre development. A personal
`settings.local.json` hook remains an option if instructions alone prove
insufficient.

**Consequences.** Doc updates are part of the definition of done for any
non-trivial spectre commit. This entry is the first written under the rule.

---

## 2026-06-06 — Dated writeup snapshots in `docs/archive/`

**Context.** A high-quality paper-style writeup of the full project state was
deposited as `archive/SPECTRE_WRITEUP_APR_2026.md` (dated 2026-04-27 — two
days after the move to rollout-based model selection, whose checkpoints its
results use). It is a valuable reference but will go stale; the living docs
must not defer to it.

**Decision.** Writeups are dated, frozen, narrative exports named
`SPECTRE_WRITEUP_<MON>_<YYYY>.md` under `docs/archive/`, catalogued in the
"Snapshots" section of `archive/README.md`. The living docs (`proposal.md` /
`decisions.md` / `notebook.md`) remain the source of truth and win on
disagreement. At deposit time: reconcile any divergence into the living docs
first (headline results → a dated `notebook.md` entry; new limitations /
future-work items → `proposal.md` §6), then freeze. After freezing, snapshots
are not edited — staleness annotations go in `archive/README.md`. (One
documented exception: the 2026-06-06 fix of the writeup's pool-cap-30 /
attempt-budget-20 conflation.)

**Consequences.** `notebook.md` seeded with the 2026-04-27 results entry;
writeup-only limitations (data efficiency, Ψ fixed-size summary,
compositional generalization, x₀-conditioned prior) merged into
`proposal.md` §6. Next snapshot due when multi-seed RT2D results land.

---

## 2026-06-04 — Silo refactor: scope and placement

**Context.** Spectre files were scattered across a shared monorepo (root spec
docs, mixed `experiments/`, spectre edits to shared configs). Refactor executed
on branch `spectre-refactor`; safety/reversibility prioritized over tidiness.

**Decisions.**

1. **Docs home = `src/alphatamp/approaches/spectre/docs/`.** Original specs
   moved byte-unchanged to `docs/archive/` (historical notes live in
   `archive/README.md`, not in the files, to keep them unchanged);
   consolidated living proposal in `docs/proposal.md`; this log; `notebook.md`
   for running EDA notes; `RESEARCH_LIT.md` → `docs/research_lit.md`.
2. **Hydra configs live in `experiments/spectre/conf/`, not
   `src/.../spectre/conf/` + `pkg://`.** All five spectre Hydra entry points
   are scripts under `experiments/`; moving scripts and configs *together*
   keeps every `@hydra.main(config_path="conf")` byte-identical, requires no
   `__init__.py` in config dirs, and no package-data additions to the shared
   `pyproject.toml`. The `pkg://` route works under the editable install but
   is strictly more moving parts for zero extra siloing.
3. **All spectre experiment files moved into `experiments/spectre/`**: the 5
   `.py` entry points, 2 `.slurm`, 3 submit/collect `.sh`, the analysis
   notebook and its output artifacts. Shared `experiments/conf/` now contains
   only other-project configs.
4. **The 3 env configs (`clutteredstorage2d_b5`, `routedtransport2d_n3_v1`,
   `stickbutton2d_b5`) moved with spectre's conf tree, not deleted.** They
   were believed unused/historical but are *live*: the first two are composed
   as `defaults` by `spectre_collect`/`spectre_build_vocab`/`spectre_train`;
   the third is selected via CLI override in `submit_spectre_stickbutton2d_b5.sh`.
   Grep-proven that no other project references them.
5. **Shared `experiments/conf/hydra/launcher/slurm.yaml` restored to `main`'s
   values (4 cpus / 16 GB).** Spectre work had bumped it to 8/32 in place — a
   contamination of a shared config also referenced by `collect_data.py`.
   Spectre keeps its tuning in its own copy at
   `experiments/spectre/conf/hydra/launcher/slurm.yaml`, which resolves via
   spectre's config_path.
6. **`.gitignore` `archive/` rule anchored to `/archive/`.** The unanchored
   rule (meant for the root archive of old experiment results) silently
   ignored the new `docs/archive/`. Verified only two `archive` dirs exist in
   the repo, so anchoring is behavior-preserving for everything else.
7. **Deliberately left in place:** `src/alphatamp/approaches/spectre/`
   (IS the importable package), `tests/approaches/spectre/` (import paths),
   `data/spectre/` (the `data_root: "data/spectre"` convention in configs and
   shell scripts is unchanged), `experiments/slurm_outputs/` (shared scratch,
   gitignored — spectre keeps writing there rather than adding new ignore
   rules for a private dir), `experiments/__init__.py` (shared), all
   other-project files (bandit/BOX, sim-free param policy, LLM
   cluttered-storage), `tests/datasets/*.pkl` (other-project fixtures; 1-byte
   pickle churn was `git restore`d, not committed).
8. **Pre-refactor cleanup commits:** `eda.py` (+3-line `set_name` helper)
   committed; `.gitignore` merge-conflict markers fixed and scratch/data
   ignores added (`.data/`, `.sandbox-*`, `data/spectre/{raw,checkpoints,configs,derived}/`,
   `*.ipynb`).

**Consequences / follow-ups.**

- Anyone with muscle-memory paths (`python experiments/spectre_train.py …`)
  must add the `spectre/` segment.
- `main`'s `.gitignore` may still carry the merge-conflict markers — fix worth
  upstreaming separately.
- The slimmed root `CLAUDE.md` and the launcher revert only exist on this
  branch until merged.

---

## Pre-refactor decisions worth remembering (imported from specs/history)

- **2026-04 — Listwise PL loss over pointwise BCE.** Attempt 2 failed because
  BCE is not rollout-aligned; PL `−log P(argmax ∈ SUCC)` is. Load-bearing.
- **2026-04 — F contains failures only.** Test-time F can never contain a
  success; training F ⊆ FAIL_e strictly. Violation was an Attempt-2 root cause.
- **2026-04 — RT2D over kinder kinematic2d.** Lookup-table baseline (B3) is
  near-oracle on kinder envs → no research gap; RT2D engineered so beating B4
  requires relational tag binding (see `archive/SYNTHETIC_ENVIRONMENT.md`).
  *(⚠️ revisited 2026-06-25 → see the 2026-06-25 pivot entry: RT2D was
  effectively partially observable to the policy, and the evaluation now prefers
  pre-existing envs meeting the representation-advantage property wishlist.)*
- **2026-04 — Layer 2 (parquet) collapsed in the data pipeline.** At
  500/100/100-episode scale, globbing + loading raw episodes is fast enough;
  EDA operates in memory (`archive/SPECTRE_TRAINING_PIPELINE_AS_BUILT.md` §3.1
  has the migration-back checklist).
- **2026-04 — Live frozen-dataclass schema instead of plain dicts.** Pickle
  stability insurance hasn't been needed; live objects let downstream code
  call substrate APIs directly (as-built §3.2).
- **2026-04 — Set-Transformer atom pooling, per-type augmentation policy,
  vocab-driven arity sizing, rollout-aligned F-mix, F-sample multiplier** —
  RT2D fixes 1–5 (`archive/SPECTRE_RT2D_METHOD_SPEC.md` §9).
- **2026-04 — AUROC(3) is the offline diagnostic that tracks test attempts;
  atom-sensitivity probes (D.1/D.2) are red herrings.** Never optimize for the
  probes. *Superseded for model selection (2026-04-25): checkpointing and
  early stopping use rollout-based `val_rollout_attempts` (see the
  overfitting-response entry below); AUROC(3) remains a secondary diagnostic.*
- **2026-04/05 — Overfitting response sequence:** diagnose → extra dropout →
  rollout-based validation/checkpoint selection (`checkpoint_metric =
  "val_rollout_attempts"` in `train.py`, used for both checkpointing and early
  stopping — aligned with the rollout-based test-time objective) → heuristic
  (FF z-score) prior as warm start (`train.prior_type`). Evaluation of prior
  choice pending.
