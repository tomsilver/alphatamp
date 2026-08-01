# SPECTRE Decisions — Method comparison and VLMPlan

8 entries, 2026-07-23 .. 2026-07-25 (closed). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-07-25-vlmplan-v3-test-split-two-arms"></a>
## 2026-07-25 — VLMPlan on the v3 test split: two model arms, output caps were binding, one cache dir per arm

<!--strip-->
> **id** `2026-07-25-vlmplan-v3-test-split-two-arms` · **status** active · **tracks**
> baselines, tooling · **supersedes** 2026-07-24-vlmplan-baseline-protocol
<!--/strip-->

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
[`notebook.md` 2026-07-25](../notebook/README.md) has the numbers.

---

<a id="2026-07-25-v3-headline-reversal-was-training-artifact"></a>
## 2026-07-25 — The dd2d_v3 headline reversal was a training artifact (short-first prior); v2 drops `--use-prior` on v3, restoring SPECTREv2-adaptive as best

<!--strip-->
> **id** `2026-07-25-v3-headline-reversal-was-training-artifact` · **status** active ·
> **tracks** method, baselines, env-dd2d · **supersedes**
> 2026-07-24-dd2d-comparison-retargeted-v3
<!--/strip-->

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
v1/PIGINet/astar/lenctx are untouched. [`notebook.md` 2026-07-25](../notebook/README.md) has the corrected table. **Open
(carried):** 3-seed reproduction — the prior's v2-helps/v3-hurts data-dependence and the residual s2
gap (v2 26.20 vs PIGINet 18.60) both need ≥3 seeds; wiring deployed-val-FP selection into `train_v2`
for robustness is a natural follow-up.

<a id="2026-07-24-dd2d-comparison-retargeted-v3"></a>
## 2026-07-24 — DD2D comparison retargeted to grasp-fixed dd2d_v3 (all 3 models retrained); precompute parameterized by `--env-variant`; the headline flips to PIGINet

<!--strip-->
> **id** `2026-07-24-dd2d-comparison-retargeted-v3` · **status** partially-superseded
> · **tracks** baselines, evaluation, env-dd2d · **superseded by**
> 2026-07-25-v3-headline-reversal-was-training-artifact
>
> ⚠️ **PARTIALLY SUPERSEDED** — the "headline flips to PIGINet" result was a training
> artifact (the short-first `--use-prior`), not a crossover. The retarget to dd2d_v3,
> the `--env-variant` parameterization and the VLMPlan pilot handling all stand. **Do
> not quote PIGINet 18.67 as the winner.**
<!--/strip-->

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
[`notebook.md` 2026-07-24](../notebook/README.md) has the full table + T0 fit.

<a id="2026-07-24-vlmplan-baseline-protocol"></a>
## 2026-07-24 — VLMPlan baseline: static hard line, off-pool proposals live-refined, generation split from scoring

<!--strip-->
> **id** `2026-07-24-vlmplan-baseline-protocol` · **status** partially-superseded ·
> **tracks** baselines, env-dd2d · **superseded by**
> 2026-07-25-vlmplan-v3-test-split-two-arms
>
> ⚠️ **PARTIALLY SUPERSEDED** — the smoke numbers here were produced under a binding
> `max_tokens` cap and are superseded. The protocol (static hard line, off-pool
> proposals live-refined, generation split from scoring) stands.
<!--/strip-->

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
[`notebook.md` 2026-07-24](../notebook/README.md) has the numbers.

---

<a id="2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts"></a>
## 2026-07-24 — DD2D collector guarantees EXACT per-stratum counts (in-flight cap + truncation)

<!--strip-->
> **id** `2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts` · **status**
> active · **tracks** data, env-dd2d
<!--/strip-->

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

<a id="2026-07-24-grasp-internal-concave-grasps"></a>
## 2026-07-24 — Grasp model extended to internal (concave-region) grasps: grip a sub-feature where the fingers fit

<!--strip-->
> **id** `2026-07-24-grasp-internal-concave-grasps` · **status** active · **tracks**
> env-dd2d
<!--/strip-->

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
  "INTERNAL GRASP" cell and `select_cells` always shows one. [`notebook.md` 2026-07-24](../notebook/README.md) has the numbers.

---

<a id="2026-07-24-grasp-model-contacts-material"></a>
## 2026-07-24 — Grasp model contacts material (slides on true contact runs, not the hull); `banana`→blocky `horseshoe`

<!--strip-->
> **id** `2026-07-24-grasp-model-contacts-material` · **status** active · **tracks**
> env-dd2d
<!--/strip-->

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
  [`notebook.md` 2026-07-24](../notebook/README.md) has the numbers.

---

<a id="2026-07-23-adaptive-traces-persist-step-scores"></a>
## 2026-07-23 — Adaptive rollout traces persist per-step raw scores + the demoted set

<!--strip-->
> **id** `2026-07-23-adaptive-traces-persist-step-scores` · **status** active ·
> **tracks** tooling · **supersedes** 2026-07-20-dd2d-comparison-notebook-piginet-bce
<!--/strip-->

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
(`evidence.py` 9 → 6). [`notebook.md` 2026-07-23](../notebook/README.md) has the numbers. Open: the **3-seed
reproduction** carried over from 2026-07-20.

---

