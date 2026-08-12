# SPECTRE v3 — As Built

Companion to [`as_built_v2.2.md`](as_built_v2.2.md): what v3 *is*, as implemented, with
the evidence for each choice. The design intent lives in
[`SPECTRE_v3_proposal.md`](SPECTRE_v3_proposal.md); where this document and the proposal
disagree, **this one describes the code** and the proposal describes what was planned.
Numbers cite [`notebook.md`](notebook/README.md); decisions cite [`decisions.md`](decisions/README.md)
and, for the 2026-07-26/27 autonomous run, [`autorun_decisions.md`](autorun_decisions.md).

> **⚠️ Update (2026-08-09).** The deployed input surface was made **domain-agnostic** —
> `obj_is_target`→`obj_is_goal`, `obj_rel` 8→3 (`d_rel=3`), `concave` cut
> ([`decisions/07` 2026-08-08](decisions/07-stickbutton2d.md#2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target)).
> That regressed the mean as optimization *variance* (not information loss), recovered by a
> training-side lever, **`--select-window 5`**, giving **DD2D 5.92 ± 0.29 (s1 4.84), SB2D
> 1.84 ± 0.26** — both tie the frozen target-anchored yardstick below (5.78 / 1.69), paired CI
> includes 0. EMA weight-averaging was built and tested but is inert on both envs
> ([`decisions/07` 2026-08-09](decisions/07-stickbutton2d.md#2026-08-09-narrowed-input-variance-selector-noise-fixed-wider)).
> The **5.78** below is the frozen yardstick, not the current deployed number.
>
> **Status (2026-07-31).** All three v3 goals are met on DD2D. Performance: over **3 seeds
> each**, v3 **5.78 ± 0.10** vs deployed v2.2 **17.27 ± 3.02**, winning every stratum — §7.
> Cleanliness and generality: §1, §3, and [`porting_guide.md`](porting_guide.md).
>
> **Coverage/waste moved to the unified definitions on 2026-07-31**
> ([`unified_culprits_coverage_waste.md`](unified_culprits_coverage_waste.md),
> `decisions.md` that date), which took v3 from 7.44 ± 0.76 to **5.78 ± 0.10** —
> **−1.66 FP, CI [−2.71, −0.71]**, every seed beating every baseline seed. Sections below
> that quote 7.44 describe the *previous* definition and are correct for it; they are not
> retracted, because those models were trained and scored consistently. The deployed arm is
> now `checkpoints_v3_unified`.
>
> **v3 is a purely learned ranker. Proof-tier demotion was cut from the method on
> 2026-07-30** (`decisions.md`): nothing outside the network touches the ordering any
> more. It cost a measured **0.23 FP** (7.20 → 7.44) and bought a system with one kind of
> component in it — see §4.1 for the trade and §8 for what the axiom registry is still
> for. The machinery is kept, tested and one flag away; it is *off*, not deleted.
>
> **The deployed model gained the record's abstract-state delta (§6.1's `s_j`) on
> 2026-07-28.** It is a *tie* with the previous configuration, not an improvement, and is
> deployed because it completes the record schema at no porting cost — §7.1 gives both
> readings, including the 6-seed one in which it is nominally behind.
>
> **Caveats:** the headline is 3 seeds, chosen as the count every method has; the ablations
> in §7.2 are 1-seed and predate both the delta and the demotion cut; and the generality
> claim is architectural, not demonstrated by a transfer to a second environment. §9 lists
> the limitations.

---

## 1. What v3 changed, in one table

| | v2.2 | v3 |
|---|---|---|
| Per-environment knowledge | 11 DD2D literals across "domain-agnostic" modules | **one `DomainSpec`**: 3 lines, 0 geometry |
| Evidence schema | 5 bespoke fact types + `FactEncoder` type vocabulary | **one `FailureRecord`** + role-separated tokens over the domain's own operator schemas |
| Sound demotion | `failure_action.startswith("retrieve")`, DD2D-specific, always on | **cut from the method** (R10). The declarative `QueryAxioms(monotone, local, exact)` machinery remains, off by default |
| What acts on the ranking | model scores **+ an external proof offset** | **model scores only** — one kind of component |
| Evidence source | offline harvest pass reconstructing facts from stored geometry | **refiner instrumentation**, observation-only (verified 290/290) |
| Prior | `[-index, -length]` data-dependent hand switch | removed (R1); length survives only as the within-length loss bucket key |
| Checkpoint selection | `relrank`, miscalibrated on dd2d_v3 | **uncensored deployed-val-FP** over the whole val split |

The generality claim is concrete and checkable: porting to a new environment needs a
converter and refiner instrumentation. The `DomainSpec` (§3) is still read — it derives
`manipulated`, `goal_objects` and `length_key` from the operator schema — but its **axiom
declarations are no longer required at deploy**, because nothing consumes a proof.

---

## 2. Architecture

Unchanged from v2.2 except where stated; shared primitives (SAB/PMA, tags, PL losses) are
*imported*, never copied, because they are survivors rather than v2-specific.

- **`SceneEncoder`** — per-object [tag emb; 32-point boundary descriptor via PMA; pose;
  `obj_rel`; `obj_is_goal`] → SAB×2. **Narrowed for domain-agnosticism (2026-08-08,
  [ADR](decisions/07-stickbutton2d.md#2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target)):**
  `obj_rel` is the width-3 anchor-free triple `[area, sinθ, cosθ]` (`D_REL_V3 = 3`), not v2.2's
  width-8 target-anchored vector; `obj_is_goal` (any goal-named object) replaces `obj_is_target`
  (the single DD2D target). The encoder takes `d_rel` per instance, so a compat-mode load of a
  v2.2 checkpoint still builds the width-8 scene. An inference probe priced the removal at Δ 0.00
  FP on both deployed models.
- **`CandidateEncoder`** — per-step [op emb + position + projected arg tags] → PMA.
  `CandidateEncoderV3` optionally replaces the learned absolute position table with a
  sinusoidal encoding (§6).
- **`RecordEncoder`** (new) — one observed failure → one token:
  `Linear([schema emb ; pooled arg-tags ; pooled culprit-tags ; scalars])`.
  **Role separation is load-bearing**: in v2.2 "argument of the failed query" and "object
  implicated as a blocker" were distinguished *by fact type*; pooling both into one slot
  would destroy that distinction. Scalars are `[j/L, log1p(effort)/10, exhausted,
  effort_is_total]` — v2.2 harvested `Fact.scalars` and then dropped them in the tensorizer.
- **`CrossAttentionScorer`** — candidates attend over scene tokens and the evidence memory;
  `[dead, jaccard]` overlap features concatenated at the head.
- **`CrossAttentionScorerV3`** (new, gated) — the same, but with a **separate attention
  channel for evidence**. v2.2 concatenates scene, global and evidence into one memory, so a
  single softmax must divide its mass between ~10 scene tokens and up to 2045 record tokens;
  since geometry is reliably useful and evidence noisy, discarding evidence is
  loss-minimizing, and the model duly learned to. Two channels remove the competition.
- **`SceneEncoderV3`** (new, gated) — scene tokens gain a 5-scalar per-object summary of the
  observed failures, so evidence enters through the **tag join** the architecture is built
  around rather than as free-floating tokens. Column 5 carries proof-tier *culprits*: the
  identity of an object the refiner reported as blocking, which is the observed counterpart
  of the `clears` predicate L2 rejected — rejected for being a routine *we* ran, not for the
  information it carried.
- **Observed `coverage` / `waste`** (new, gated) — `cand_overlap` widens to 4. These are
  §5.1's necessity features computed from reported culprits instead of a predicted per-object
  head; necessity conditioning was cut because the head would have had to predict `p_i` from
  geometry, and once the refiner reports culprits the same two features need no head at all.
- **`AuxHead`** — present, **never trained**. No collection has ever populated
  `aux_labels`, so v2.2's masked BCE contributed exactly zero gradient. v3 says so rather
  than implying an aux loss exists. (`as_built_v2.2` §2.4's claim to the contrary is
  incorrect and is corrected here.)

**Which of these the deployed config actually enables** — the gated components are not all
on, and it matters for reading §7's ablations:

| component | in deployed config? |
|---|---|
| `RecordEncoder` (record tokens), aggregated per query | **yes** |
| `CrossAttentionScorerV3` (separate evidence channel) | **yes** |
| observed `coverage`/`waste`, `dead` dropped from the net | **yes** — this is what carries the result |
| record `state_delta` (§6.1's `s_j`, as a delta from `s_0`) | **yes** — a tie on DD2D; deployed to complete the record schema (§7.1) |
| **proof-tier demotion** (the external offset) | **no** — cut 2026-07-30 (§4.1); `apply_demotion=False` is the default everywhere |
| `SceneEncoderV3` (per-object evidence) | **no** — built and tested; hurt s1 badly on its own (20.84) |
| `CandidateEncoderV3` (sinusoidal positions) | **no** — built and tested; G9 descoped. The D-8 oracle's *forward*/structure half is live; its *rollout* half retired with the 2026-08-08 scene narrowing |
| necessity head | **no** — cut; `use_necessity` raises |

**Loss** — listwise Plackett–Luce, global + within-length buckets. No pointwise BCE on the
ranker. The bucket key is `domain.length_key`, verified to induce the identical partition
to v2.2's DD2D-specific key on 120000/120000 skeletons.

**D-8, exact absence.** Every v3 *feature* is config-gated on `V3Config`, and with all flags
off the model is built from the *same submodule classes under the same attribute names* as
v2.2 — so a v2.2 checkpoint loads `strict=True` in compat mode. **The rollout-equivalence half
of the oracle retired 2026-08-08**: deployed v3 narrows the scene (`d_rel = 3`) where v2.2 is
width-8, so a *deployed* rollout is no longer bit-identical to v2.2 by design. What
`test_v3_equivalence.py` still enforces, and is what made the data-path rewrites safe: v2.2
loads into a **compat-mode** (`d_rel = 8`) `SpectreV3Model`, the submodule structure matches at
that width, and a **forward pass over the same width-8 batch stays bit-identical**. (The
scene-narrowing is a *removal*, so unlike the config-gated features it is not additive; it is
the deliberate exception noted in the ADR.)

---

## 3. The domain contract — the whole per-environment surface

```python
_DD2D = DomainSpec(
    axioms={
        "retrieve":     QueryAxioms(monotone=True, local=True, exact=True),
        "pick":         QueryAxioms(),
        "place-buffer": QueryAxioms(),
    },
    min_calls_per_schema={"pick": 1, "place-buffer": 2, "retrieve": 1},
)
```

Everything else is derived from the operator schema: `goal_objects` from the goal literals,
`manipulated` as `args(σ) \ goal_objects` (equals v2.2's `place-buffer` filter on
120000/120000 skeletons), `length_key` as the operator count.

Declaring an axiom has the epistemic status of writing the PDDL domain file — it is
*specification*, not a learned or inferred routine. An unknown environment degrades to
`EMPTY_SPEC` rather than raising.

**Since 2026-07-30 the axioms are optional at deploy.** With demotion cut (§4.1) nothing
consumes a proof, so the three `QueryAxioms` lines above affect only (a) which records are
held out of the token path as proof-tier — 1.3% of them on dd2d_v4, all `retrieve` — and
(b) the opt-in `apply_demotion=True` path. **"Learning is the floor" stopped being the
fallback and became the deployed configuration.** A new environment supplies a converter
and refiner instrumentation; the axiom block is now a tuning knob rather than a
requirement.

*(Retained because it still governs the opt-in path:)* **the `exact` axiom is not
decoration.** `refine()` reports the deepest step *reached*, which on a wall-clock exit was
never tested — it will name `retrieve(target)` though the retrieve never ran. That is the
confirmed cause of all 12/18694 dd2d_v2 demotion violations, and splitting "the domain says
this query type is exhaustive when it completes" from "the observation says it actually
ran" is what makes `strict` mode sound: 0 demoted-but-feasible.

---

## 4. Failures as observations

`FailureRecord(candidate_idx, step_index, schema, args, culprits, unmoved, n_step,
exhausted, budget_exhausted, effort_is_total, instrumented, state_delta)`.

The records were consumed in two tiers, and **the proof tier was cut on 2026-07-30**
(§4.1). What remains:

- **Hint tier** — a record becomes a learned token, and the observed culprits it carries
  become the per-candidate `coverage`/`waste` columns. This is now the *only* consumer.
- **Proof tier** — where the domain declares monotone + local + exact *and* the observation
  proves the query ran, the consequence *could* be applied outside the network as a finite
  demotion offset (never pool removal, P-E). Retained, tested, `apply_demotion=True`, and
  **not deployed**.

The tier split itself still runs: proof-tier records are held out of the token path exactly
as before, so the trained checkpoints see the input distribution they trained on. On
dd2d_v4 that is **391 / 29054 records (1.3%)**, all `retrieve` — see §4.1 for why that
number matters.

### 4.1 Why the proof tier was cut

**The story is the reason, and the price is small and measured.** v3's claim is one
canonical record consumed by learned components. An external, hand-declared deduction
acting on the ranking is a second kind of thing in the system: it needs the axiom registry,
it needs `strict`/`permissive` semantics, and it needs a soundness argument (C5/P-E) that
is a paragraph of the paper. Removing it makes the deployed system **one mechanism**, and
makes "learning is the floor" the configuration rather than the fallback.

What it costs, measured (`notebook.md` 2026-07-30, notebook §4.3): **0.23 FP**, 7.20 →
7.44 over 3 seeds, paired-bootstrap CI [+0.08, +0.43]. Real, significant, and small.

Three measurements say it is a cheap thing to give up *on this domain*:

- **It barely fires.** Demotion changed the realized attempt order on **18/300 (6%)** of
  deployed (problem, seed) pairs, against **55/100 (55%)** on the stripped floor arm. The
  learned features have already ordered the pool so the proof usually has nothing left to
  correct.
- **The learned components absorb ~79% of it.** On the floor arm — jaccard only, no
  `coverage`/`waste`, no record tokens — withholding the offset costs **1.09 FP** (15.47 →
  16.56), 4.7× the deployed cost.
- **It only ever acted at s2/s3.** s0 and s1 are *bit-identical* with and without it, at
  both 3 and 6 seeds. Demotion needs a subset-containment proof, which needs multi-object
  stagings; an s1 candidate stages one object, so nothing there is ever provably dead.

**What is honestly given up.** The offset was *sound* — 0 demoted-but-feasible under
`strict` — and a learned correlate is not a proof. On a domain whose proofs fire far more
often than DD2D's 6%, this trade would go the other way, and the flag exists for exactly
that. Also: with no proof consumer, the 1.3% of records the tier split holds back are now
used by **nothing**. Routing them into the token path is the obvious follow-up and needs a
retrain, so it was not done here (§9).

*(This also corrects an unverified note in `autorun_decisions.md` A19, which guessed from a
single episode that "most DD2D failures are `retrieve`" and therefore proof-tier. Measured
across the test split it is 1.3%.)*

**`effort_is_total` exists because a re-collection would otherwise silently redefine a
column**: backfilled records report whole-attempt effort, instrumented ones report per-step.

**`state_delta` fills §6.1's last empty slot** (`--state-delta`, **in the deployed
configuration** since 2026-07-28). Every other field of the proposal's record was built;
`s_j` — the abstract state at the failing step — was not. What is carried is the *delta*
from `s_0` (which atoms the prefix added, which it deleted), since `s_0` already reaches the
scorer through the scene tokens. It is pure STRIPS progression over the candidate's own
plan, so it needs **no new instrumentation**: any environment that can supply an
`EpisodeRecord` at all gets it for free. That is why it is deployed — it is a **tie** on
DD2D, not a win (§7.1), and completing the schema for free is the whole of the case for it.

Encoding, with two choices that are load-bearing rather than incidental: each delta atom is
projected **before** the pool over atoms (so `{on-buffer(o1), holding(o2)}` and
`{on-buffer(o2), holding(o1)}` differ), and an atom's argument slots are **concatenated
positionally** rather than pooled (so `p(a,b)` and `p(b,a)` differ — invisible on all-unary
DD2D, which is why a synthetic test pins it). The branch is additive and **zero-initialized**
so that turning the flag on changes no other parameter's initialization; widening the record
projection instead would have re-randomized all of it and made the arm an init lottery rather
than an ablation.

Two DD2D-specific caveats bound what it can show *here* and are recorded so they are not
re-derived:

- the delta's *object* set is exactly `all_objects − unmoved` on **946,063/946,063** records,
  so on this domain its only content beyond the already-populated (and never-tensorized)
  `unmoved` field is the **predicate label** on each object;
- under `aggregate_records`, **47.8%** of tokens carry an *empty* delta (54.9% at s2/s3),
  because aggregation collapses the deep re-sampled records and leaves the shallow `j = 0`
  ones proportionally dominant.

Its **size** is nearly determined by the `j/L` scalar the token already carries
(corr 0.940), so no count feature is derived from it — that would be `dead` again. What it
adds is object *identity*: given everything the model already sees about a record
(`problem, schema, args, j/L`) there are 2.65 distinct staged sets on average, >1 in 53.6%
of groups.

**Instrumentation is observation-only, and that is an invariant.** `n_attempts` *is*
`counter.calls`, so one extra stream call shifts it and cascades into every label.
`grasp_cfree` was therefore refactored to `grasp_blocker(...) < 0` — the culprit is the
witness the short-circuit already computed. Verified differentially: `label`, `steps_bound`,
`plan_length`, `failure_action` identical on 290/290 replayed candidates.

**Aggregation.** The refiner emits one observation per failed *sample*; §6.1 defines a record
per failing *query*. Left raw, a candidate whose `place-buffer(o)` was retried across many
poses contributes hundreds of near-identical tokens (mean 2.2 per candidate but max 290; at
|F|=30, mean 226 tokens and max 2045, against v2.2's ~40 facts). `aggregate_records` collapses
to one per `(schema, args)` — deepest step, summed effort, unioned culprits — for −88.7%
tokens with nothing the token *encodes* lost.

---

## 5. Training and selection

Recipe: AdamW, lr 3e-4, cosine with 2 warmup epochs, 30 epochs, batch 8, dropout 0.1,
weight decay 5e-4, within-length weight 1.0, tag-permutation augmentation on. Identical to
v2.2's deployed recipe — verified field-by-field against the stored checkpoint cfgs, so
v3-vs-v2.2 is not a recipe comparison.

Deployed feature flags (the `v3final` sweep preset): `--overlap-mode jaccard
--coverage-feats --aggregate-records --evidence-attn --state-delta`.

**Selection is uncensored deployed-val-FP** over the whole 100-episode val split, on a
3-epoch moving average, with the demotion rule pinned to `permissive` so a change to the
rule cannot move the selector underneath a comparison. Three guards, each from a specific
failure — see `train_v3.py`'s module docstring. The censoring lesson generalizes and is
worth restating: **a selection statistic must never be censored below the region where the
candidates differ**, and *stable curves are not evidence of a good selector* — the censored
selector's curves were stable and picked sensible mid-training epochs while spanning ≈6 FP
where the uncensored one spans ≈15.

**Failure-context sampling.** `sample_context` keeps ~35% of mass at `|F| = 0` (the
deployment start, which the static pathway must own alone) and applies evidence dropout.
`tail_max_f` optionally spreads half the non-empty mass out to |F| ≈ 40, because v2.2's
inherited cap of 8 never shows the model the regime an s3 rollout actually spends most of
its attempts in.

---

## 6. Position encoding

`CandidateEncoderV3` replaces the learned absolute `nn.Embedding(64, D_MODEL)` with a
sinusoidal encoding, subclassed rather than edited in place because D-7 freezes v2 modules.
`pos_emb` is *deleted*, so it leaves the state dict — which is precisely why enabling it
retires the D-8 oracle.

**Honest scope note:** the motivating OOV problem does **not** occur on DD2D. Measured, s0–s2
candidate pools already contain 9-operator plans (max step index 8) while s3 needs only 6, so
under the "train s0–s2 / deploy s3" protocol the absolute table is never queried out of range.
The sinusoidal encoder is future-proofing for longer-horizon domains and a generality argument
— not a fix for a live DD2D defect. Claiming otherwise would be unsupported.

---

## 7. Results

**Uncensored deployed FP on dd2d_v4 test, n=100.** Lower is better.

**Headline: 3 seeds each** (mean ± std *across seeds* of the per-stratum mean):

| method | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| **v3 deployed** (no demotion) | **7.44 ± 0.76** | 0.00 ± 0.00 | **3.96 ± 2.50** | **13.15 ± 0.34** | **12.64 ± 1.95** |
| **v2.2 yardstick** (with demotion) | 17.27 ± 3.02 | 0.00 ± 0.00 | 13.67 ± 14.20 | 23.45 ± 2.76 | 31.95 ± 5.62 |

**−9.83 FP, 95% CI [−12.57, −7.36]** (paired bootstrap over problems on the seed-mean).
v3 wins ALL, s2 and s3; s0 is a tie at 0.00 (every method solves those on the first
attempt); s1 is a win on the means but against a ±14.20 baseline spread, so read it as
"not worse" rather than as a 3× improvement. Per-seed ALL: 6.56 / 7.84 / 7.91.

**The comparison is deliberately asymmetric and it handicaps v3.** v3 runs without
proof-demotion (§4.1); the v2.2 yardstick keeps its own observed demotion, because 17.27 is
the number published for it throughout this project and re-scoring the baseline to match a
v3 design choice would be moving the goalposts. So the margin is measured against a
*stronger* baseline than v3 gives itself.

**Three seeds because that is what every method has** — v2.2 was trained at exactly 3, and
PIGINet gained a seed axis for the comparison notebook. It is a protocol set by the
weakest-covered method. §7.1 gives v3's 6-seed reading, which is *worse*, and it is stated
there rather than omitted.

**Against the other comparators** (same split, same protocol; full table in
[`notebook.md` 2026-07-28](notebook/README.md)):

| method | seeds | ALL | s1 | s2 | s3 |
|---|---|---|---|---|---|
| **SPECTREv3-adaptive** | 3 | **7.44 ± 0.76** | 3.96 ± 2.50 | 13.15 ± 0.34 | 12.64 ± 1.95 |
| PIGINet (low-level, BCE) | 3 | 17.27 ± 0.19 | 5.04 ± 1.49 | 18.77 ± 1.58 | 45.20 ± 0.84 |
| SPECTREv2-adaptive | 3 | 17.27 ± 3.02 | 13.67 ± 14.20 | 23.45 ± 2.76 | 31.95 ± 5.62 |
| VLMPlan-32B (zero-shot) | 1 † | 23.55 | 5.04 | 13.16 | 69.24 |
| astar-dist | - | 34.52 | 2.24 | 17.08 | 118.76 |

† grafted from dd2d_v3. **PIGINet and v2.2 tie on the mean at 17.27 and are nothing alike** —
±0.19 vs ±3.02, and PIGINet is much better at s1/s2 and much worse at s3. Never quote either
mean without its spread; the coincidence is exactly the kind that reads as a finding.

Reproduce:

```bash
python experiments/spectre/spectre_sweep.py --preset v3final --seeds 0 1 2
# demotion is OFF by default; `--with-demotion` re-enables it (the §4.1 ablation).
# NOTE the `--v2-arm` yardstick is scored through the same switch, so this command
# reports v2.2 WITHOUT its demotion; the 17.27 above is v2.2 as published, read from
# the compare cache, whose `cache_spectre2` uses v2's own always-demoting rollout.
python experiments/spectre/spectre_score_v3.py --env-variant dd2d_v4 \
    --arm "v3 deployed:checkpoints_v3_v3delta_s{seed}" --seeds 0 1 2
```

**The baseline is now averaged, not best-cased, and that inflates the margin.** Earlier
versions of this document compared against v2.2 **seed 0** (14.66) because v3 had 6 seeds
and v2.2 had 1, and seed 0 is v2.2's *best* (per-seed 14.66 / 16.57 / 20.57). With both
sides at 3 seeds the like-for-like comparison is mean-to-mean, so the yardstick moves to
17.27 — a weaker baseline than the one previously quoted, and roughly 2.6 FP of the −9.83
is that change of basis rather than a change in v3. Against seed 0 the margin is ≈ −7.2.
v2.2's s1 spread of ±14.20 (seed 2 lands at 30.04, `relrank` selecting a bad epoch) is
itself the miscalibration R8 replaced.

### 7.1 Seed counts, and the reading that disagrees

v3 has **6** trained seeds; the headline uses 3. The two readings disagree about whether
the state delta helped, and both are ties by paired bootstrap:

| configuration | 3 seeds (0–2) | 6 seeds |
|---|---|---|
| **v3 deployed** (state delta, no demotion) | **7.44 ± 0.76** | 8.54 ± 1.43 |
| v3 pre-delta, *with* demotion (`checkpoints_v3_v3final_s*`) | 7.44 ± 0.23 | 7.90 ± 0.61 |

Delta-vs-pre-delta at 6 seeds is **+0.34 FP, CI [−0.30, +1.07]** — a tie, with the sign
splitting 3–3 across seeds (6.49 / 7.51 / 7.61 / 8.61 / 10.43 / 8.76 against 7.50 / 7.63 /
7.19 / 8.05 / 8.08 / 8.94). So **the state delta is deployed on the strength of completing
the record schema for free, not on the strength of the number.** Anyone quoting 7.44 should
know that the same model over all six trained seeds reads 8.54, and that 3 seeds is the
better-looking half of what exists — it is the count every *method* has, which is a real
constraint, but it is not the count v3 has.

*(Both columns are now demotion-free for the deployed row; the delta-vs-pre-delta
comparison quoted above was measured with demotion on, before it was cut. The tie stands
either way — the demotion cut moves both configurations by ~0.2-0.3 FP.)*

**s1 is where a 3-seed report is least trustworthy**, and this has bitten before: at 3 seeds
the pre-delta arm's s1 read 3.79 ± 3.29 and looked like a clear win; at 6 it moved to
5.60 ± 3.06, with four of six seeds *worse* than the v2.2 seed-0 figure. s1 has the smallest
FP and the largest relative spread. Overall FP is by contrast stable across seeds.

### 7.2 Ablations (1 seed, pre-delta)

All against v2.2 seed 0. These arms **predate the state delta** and were not re-run, so they
decompose the pre-delta configuration; the deployed row differs from them by the delta as
well as by record aggregation and evidence-attention.

| method | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| v3 pre-delta (seed 0) | 7.50 | 0.00 | 1.16 | 15.80 | 13.04 |
| v3, coverage + `dead` kept | 7.76 | 0.00 | 2.56 | 12.60 | 15.88 |
| v3, coverage, **no record tokens** | 7.82 | 0.00 | 3.48 | 12.28 | 15.52 |
| v3, coverage only (no aggregation/attention) | 8.39 | 0.00 | 2.72 | 12.64 | 18.20 |
| v3, rollout-aligned context, no coverage | 14.34 | 0.00 | 8.04 | 17.48 | 31.84 |
| v3, evidence-attention only | 14.92 | 0.00 | 3.56 | 27.48 | 28.64 |
| v3, no records at all | 15.34 | 0.00 | 4.64 | 26.24 | 30.48 |
| v3 as of G6b (record tokens, shared attention) | 16.17 | 0.00 | 8.56 | 22.00 | 34.12 |
| **v2.2 yardstick** | 14.66 | 0.00 | 6.20 | 26.00 | 26.44 |

**Every coverage-bearing arm beats the yardstick significantly** (−6.3 to −7.2 FP, all CIs
excluding 0), so the result does not depend on the exact combination. What it depends on is
`coverage`/`waste`: the one arm without them (rollout-aligned context) is a tie at −0.32.

**Both consumptions of the record are load-bearing**, measured at 6 seeds each:

| | ALL | s1 | s2 | s3 |
|---|---|---|---|---|
| deployed (**with** record tokens) | **7.90 ± 0.61** | **5.60 ± 3.06** | 13.03 ± 1.52 | 12.96 ± 2.46 |
| coverage only (**no** tokens) | 9.18 ± 1.41 | 10.78 ± 6.47 | 12.91 ± 0.84 | 13.03 ± 2.00 |

The tokens are worth **1.28 FP**, and their contribution is concentrated entirely at **s1**
(5.60 vs 10.78 — without them the model is *worse than v2.2* there) while s2 and s3 are
ties. They also **halve the variance** (overall sd 0.61 vs 1.41). So the method is one
canonical record consumed *two* ways: compact per-candidate features carrying s2/s3, and a
per-failure token stream carrying s1 and the stability. Neither alone is the method.

*(An earlier 1-seed comparison put the token contribution at 0.26 FP and is superseded —
`autorun_decisions.md` A17.)*

**The deployed configuration** is `--overlap-mode jaccard --coverage-feats
--aggregate-records --evidence-attn`, i.e. four changes to the G6b model, each motivated by
a measurement rather than a sweep:

| change | why | measured before training |
|---|---|---|
| drop `dead` from the net | it is a *length* proxy: correct at s3, wrong at s1 | corr(dead, \|S\|) = −0.284 |
| observed `coverage`/`waste` | states the *count* `dead` was proxying for | 2.45× separation at s3 |
| aggregate records per query | one token per failing *query*, not per failed sample | −88.7% tokens, max 2045 → 37 |
| separate evidence attention | evidence competed with geometry in one softmax | `suppress_records`: 16.17 → 16.40 |

**On P-v3-1.** The pre-registered bar was s2 ≤ 17.08 *via necessity conditioning*, and note
the bar was measured on **dd2d_v3** while ours is **dd2d_v4** (~0.08% of labels differ, so
comparable, but not the same benchmark). s2 lands
at 15.88 (12.64 in the coverage-only arm), so the **number** is beaten — but by observed
culprits, not by the predicted necessity head, which was withdrawn. Both halves are worth
stating: the target was right, the proposed mechanism was not needed.

**On P-v3-3.** Falsified, and reported as such: removing `cand_overlap` costs −5.07 FP,
CI [−8.56, −1.78]. It is reinstated per R7's own escape clause.

### 7.3 It is not the selector

v3 replaced v2.2's `relrank` checkpoint selection with uncensored deployed-val-FP (R8), and
`relrank` is known-miscalibrated — so an obvious question is how much of the margin is just
better selection rather than the representation. **The ablations answer it directly: every
v3 arm *without* coverage ties v2.2, despite all of them using the v3 selector.**

| arm (all use the v3 selector) | ALL | vs v2.2 |
|---|---|---|
| rollout-aligned context, no coverage | 14.34 | −0.32, n.s. |
| evidence-attention only | 14.92 | +0.26, n.s. |
| no records at all | 15.34 | +0.68, n.s. |
| *v2.2 (relrank selector)* | *14.66* | — |

If the selector were carrying the result, those arms would already show it. They do not, so
the ~7 FP belongs to `coverage`/`waste`, not to R8.

### 7.4 What the win rests on

The whole gain traces to one substitution. §5.1 wanted a per-object necessity `p_i`
*predicted* by a head; v3 gets the same two candidate features from culprits the refiner
*reported*:

```
coverage = |S(c) ∩ culprits| / |culprits|      waste = |S(c) \ culprits| / |S(c)|
```

Both are exactly zero until a failure has been observed, so the first attempt is still
purely static and the signal accrues as the rollout proceeds.

**Measured, and it is the sharpest statement of the contribution:** v3 and v2.2 solve the
*same* 25% of episodes on attempt 1, while among the episodes needing a second attempt v3
averages **10.00** FP against v2.2's **19.55**. The entire −7 FP appears *after the first
observed failure*.

Stated precisely, because the round number invites over-claiming: that 25% is exactly the 25
s0 episodes, for both models — neither solves any s1–s3 episode immediately. So the claim is
not "the static rankers are equally good in some nuanced sense" but the blunter and stronger
**"the first attempt separates the two methods not at all; every attempt after it does."**
v3 is not a better static ranker, it is a better **re**-ranker — which is what the adaptive
component is supposed to buy, and an independent corroboration of the leakage audit, since a
feature leaking feasibility would have lifted the first pick as well. A leakage audit
(features zero at |F|=0; culprits only from candidates in the failure context, all of which
are failures; the deploy loop breaks on success before a successful candidate can enter the
context) returned 0 violations.

---

## 8. What was removed, and what came back

| # | Component | Disposition | Evidence |
|---|---|---|---|
| R1 | short-first prior | **removed** | data-dependent; diverged training on dd2d_v3 (L3) |
| R2 | computed demotion source | **not ported** | last per-env geometry routine in the deployment story |
| R3 | packing certificate in the method | **not ported** | inert: 0 proofs at λ=0.8 |
| R4 | analytic `grasp-witness` | **replaced** by observed culprits | C1/C2 |
| R5 | 5 fact types + `FactEncoder` vocab | **replaced** by one record | §4 |
| R6 | global token | **kept** — container tokens (its replacement) are G10 work, which was not reached; removing it first would have deleted information with nothing to carry it | |
| R7 | `cand_overlap` | **kept — P-v3-3 falsified** | removal costs −5.07 FP, CI [−8.56, −1.78] |
| R8 | `relrank` selection | **replaced** | §5 |
| R9 | `exclude_marginal` | **not ported** | inert twice over; reinstating needs a real label mask |
| R10 | **proof-tier demotion** (the external offset) | **cut from the method 2026-07-30**; machinery kept, `apply_demotion=False` everywhere | costs 0.23 FP, CI [+0.08, +0.43]; fires on 6% of deployed rollouts vs 55% on the floor arm (§4.1) |

Necessity conditioning (proposal §5) was **cut** — D2 showed the s2 deficit is
*within-length*, which necessity conditioning does not address. `necessity.py` remains built
and tested but unwired, and `V3Config.use_necessity` raises rather than silently doing
nothing.

---

## 9. Known limitations

1. **The headline is 3 seeds, and v3 has 6.** 3 is the count every *method* has (v2.2 was
   trained at exactly 3), so it is the like-for-like protocol — but it is also the
   better-looking half of v3's seeds: over all 6 the deployed model reads **8.54 ± 1.43**
   rather than 7.44 ± 0.76. §7.1 carries both. The **ablations in §7.2 are 1 seed** and
   predate the state delta, accepted by paired bootstrap over problems.
2. **The state delta is deployed on a tie.** It does not improve the number on DD2D
   (§7.1); it is in the deployed config because it completes §6.1's record schema at zero
   porting cost. Two measured DD2D properties bound what it could show here (§4) and
   neither is a property of the mechanism.
3. **Cutting demotion leaves 1.3% of records consumed by nothing.** The tier split still
   holds proof-tier records out of the token path (391/29054 on dd2d_v4, all `retrieve`),
   but there is no longer a proof consumer to hand them to. Routing them into the token
   path is the obvious follow-up and needs a retrain, so it was not done here (§4.1).
4. **DD2D generation is `PYTHONHASHSEED`-dependent**, so no collection is reproducible
   across processes. dd2d_v4 differs from dd2d_v3 on 0.08% of candidate labels.
5. **dd2d_v4 carries no harvested post-mortem facts**, so any v2.2 checkpoint trained on it
   has an inert evidence pathway. This is a property of the *collection*, not of v2.2.
6. **Env-2 has not been attempted.** The generality claim is therefore *architectural* — the
   contract is 3 lines and the fallback is measured — not yet *demonstrated by transfer*.
