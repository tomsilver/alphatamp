# SPECTRE v3 — As Built

Companion to [`as_built_v2.2.md`](as_built_v2.2.md): what v3 *is*, as implemented, with
the evidence for each choice. The design intent lives in
[`SPECTRE_v3_proposal.md`](SPECTRE_v3_proposal.md); where this document and the proposal
disagree, **this one describes the code** and the proposal describes what was planned.
Numbers cite [`notebook.md`](notebook.md); decisions cite [`decisions.md`](decisions.md)
and, for the 2026-07-26/27 autonomous run, [`autorun_decisions.md`](autorun_decisions.md).

> **Status (2026-07-27).** All three v3 goals are met on DD2D. Performance: v3 **weakly
> dominates deployed v2.2** — 7.90 ± 0.61 vs 14.66 over 6 seeds (−6.76 FP, CI
> [−9.43, −4.40]); s2 and s3 win by ~2×, s0 and s1 tie — §7. Cleanliness and generality: §1, §3, and
> [`porting_guide.md`](porting_guide.md). **Caveats: the deployed config is 6-seed; every ablation is
> 1-seed and so is the v2.2 yardstick**; and the generality claim is architectural, not yet
> demonstrated by a transfer to a second environment. §9 lists the limitations.

---

## 1. What v3 changed, in one table

| | v2.2 | v3 |
|---|---|---|
| Per-environment knowledge | 11 DD2D literals across "domain-agnostic" modules | **one `DomainSpec`**: 3 lines, 0 geometry |
| Evidence schema | 5 bespoke fact types + `FactEncoder` type vocabulary | **one `FailureRecord`** + role-separated tokens over the domain's own operator schemas |
| Sound demotion | `failure_action.startswith("retrieve")`, DD2D-specific | **declarative `QueryAxioms(monotone, local, exact)`** per query type |
| Evidence source | offline harvest pass reconstructing facts from stored geometry | **refiner instrumentation**, observation-only (verified 290/290) |
| Prior | `[-index, -length]` data-dependent hand switch | removed (R1); length survives only as the within-length loss bucket key |
| Checkpoint selection | `relrank`, miscalibrated on dd2d_v3 | **uncensored deployed-val-FP** over the whole val split |
| Demotion soundness | 12/3289 demoted candidates were feasible on dd2d_v2 | **0** under `strict` mode |

The generality claim is concrete and checkable: porting to a new environment needs a
converter, refiner instrumentation, and a `DomainSpec`. DD2D's is reproduced in full in §3.

---

## 2. Architecture

Unchanged from v2.2 except where stated; shared primitives (SAB/PMA, tags, PL losses) are
*imported*, never copied, because they are survivors rather than v2-specific.

- **`SceneEncoder`** — per-object [tag emb; 32-point boundary descriptor via PMA; pose;
  relation-to-target; is-target] → SAB×2. Unchanged.
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
| `SceneEncoderV3` (per-object evidence) | **no** — built and tested; hurt s1 badly on its own (20.84) |
| `CandidateEncoderV3` (sinusoidal positions) | **no** — built and tested; G9 descoped, so the D-8 oracle is still live |
| necessity head | **no** — cut; `use_necessity` raises |

**Loss** — listwise Plackett–Luce, global + within-length buckets. No pointwise BCE on the
ranker. The bucket key is `domain.length_key`, verified to induce the identical partition
to v2.2's DD2D-specific key on 120000/120000 skeletons.

**D-8, exact absence.** Every v3 feature is config-gated on `V3Config`, and with all flags
off the model is built from the *same submodule classes under the same attribute names* as
v2.2 — so a v2.2 checkpoint loads `strict=True` and `test_v3_equivalence.py` demands
identical decisions through the v3 code path. That oracle is what made the data-path
rewrites safe. It retires only when `sinusoidal_pos` is enabled, since `cands.pos_emb`
then leaves the state dict.

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
`EMPTY_SPEC` (everything hint-tier) rather than raising, so **"learning is the floor" is
the default path, not a special case**.

**The `exact` axiom is not decoration.** `refine()` reports the deepest step *reached*, which
on a wall-clock exit was never tested — it will name `retrieve(target)` though the retrieve
never ran. That is the confirmed cause of all 12/18694 dd2d_v2 demotion violations. Splitting
"the domain says this query type is exhaustive when it completes" from "the observation says
it actually ran" is what makes `strict` mode sound: 0 demoted-but-feasible.

---

## 4. Failures as observations

`FailureRecord(candidate_idx, step_index, schema, args, culprits, unmoved, n_step,
exhausted, budget_exhausted, effort_is_total, instrumented)`.

Two tiers, and the split is the lesson of L4:

- **Proof tier** — where the domain declares monotone + local + exact *and* the observation
  proves the query ran, the consequence is applied **outside the network** as a finite
  demotion offset. Never pool removal (P-E): a wrong proof costs attempts, it cannot lose
  the feasible plan.
- **Hint tier** — everything else becomes a learned token.

**`effort_is_total` exists because a re-collection would otherwise silently redefine a
column**: backfilled records report whole-attempt effort, instrumented ones report per-step.

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

**Headline, 6 seeds** (mean ± std *across seeds* of the per-stratum mean):

| method | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| **v3 deployed** (6 seeds) | **7.90 ± 0.61** | 0.00 ± 0.00 | 5.60 ± 3.06 | **13.03 ± 1.52** | **12.96 ± 2.46** |
| **v2.2 yardstick** (1 seed) | 14.66 | 0.00 | 6.20 | 26.00 | 26.44 |

**−6.76 FP, 95% CI [−9.43, −4.40]** (paired bootstrap over problems on the seed-mean).

Reproduce:

```bash
python experiments/spectre/spectre_sweep.py --preset v3final --seeds 0 1 2 3 4 5
python experiments/spectre/spectre_score_v3.py \
    --arm "v3 deployed:checkpoints_v3_v3final_s{seed}" --seeds 0 1 2 3 4 5 \
    --baseline "v2.2 yardstick:checkpoints_v2_evidence_ov"
```

**Weak dominance holds — no stratum regresses — but the strata are not equally won**, and
the honest reading is stratum by stratum:

| stratum | verdict |
|---|---|
| s0 | **tie** at 0.00 — every method solves these on the first attempt |
| s1 | **tie, not a win.** 5.60 ± 3.06 vs 6.20 is a +0.60 margin against a 3.06 seed sd (0.20 sd), and only **2 of 6 seeds** beat 6.20 — per-seed 1.16 / 2.72 / 7.48 / 6.68 / 6.28 / 9.28 |
| s2 | **win** — 13.03 ± 1.52 vs 26.00, ~2× |
| s3 | **win** — 12.96 ± 2.46 vs 26.44, ~2× |

**The extra seeds earned their keep, and this is the one number a 3-seed report would have
got wrong.** At 3 seeds s1 read 3.79 ± 3.29 and looked like a clear win; three more seeds
moved it to 5.60 and revealed that four of six seeds are *worse* than the yardstick there.
s1 has the smallest FP and so the largest relative spread. Overall FP is by contrast stable
(per-seed 7.50 / 7.63 / 7.19 / 8.05 / 8.08 / 8.94).

**Ablations, 1-seed dev**, all against the same yardstick:

| method | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| v3 deployed (seed 0) | 7.50 | 0.00 | 1.16 | 15.80 | 13.04 |
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

**On P-v3-1.** The pre-registered bar was s2 ≤ 17.08 *via necessity conditioning*. s2 lands
at 15.88 (12.64 in the coverage-only arm), so the **number** is beaten — but by observed
culprits, not by the predicted necessity head, which was withdrawn. Both halves are worth
stating: the target was right, the proposed mechanism was not needed.

**On P-v3-3.** Falsified, and reported as such: removing `cand_overlap` costs −5.07 FP,
CI [−8.56, −1.78]. It is reinstated per R7's own escape clause.

### 7.1 It is not the selector

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

### 7.2 What the win rests on

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

Necessity conditioning (proposal §5) was **cut** — D2 showed the s2 deficit is
*within-length*, which necessity conditioning does not address. `necessity.py` remains built
and tested but unwired, and `V3Config.use_necessity` raises rather than silently doing
nothing.

---

## 9. Known limitations

1. **1-seed development.** Every v3 number is 1 seed, accepted by paired bootstrap over
   problems. Paper numbers need ≥3 seeds.
3. **DD2D generation is `PYTHONHASHSEED`-dependent**, so no collection is reproducible
   across processes. dd2d_v4 differs from dd2d_v3 on 0.08% of candidate labels.
4. **dd2d_v4 carries no harvested post-mortem facts**, so any v2.2 checkpoint trained on it
   has an inert evidence pathway. This is a property of the *collection*, not of v2.2.
5. **Env-2 has not been attempted.** The generality claim is therefore *architectural* — the
   contract is 3 lines and the fallback is measured — not yet *demonstrated by transfer*.
