# Failure-record learned pathway — as-built retrospective

**Status:** as-built, 2026-08-24. This is the *retrospective* companion to the probe plan
[`failed_records_fix.md`](failed_records_fix.md): what we actually followed, deviated from,
decided, learned, and shipped. Where the two disagree, **this document is the record of what
happened**; `failed_records_fix.md` remains the frozen original plan.

**One-line result:** the deployed *scalars-on* method is unchanged; a scalar-free,
domain-agnostic **pre-pooling step-join** (F-B2) recovers **~25% of the scalar gap** on DD2D
(the workstream's one positive lever, now baked into the deployed recipe as future-proofing),
while content enrichment, match-hint biases, auxiliary supervision, and a rollout-aligned
large-|F| curriculum were each **tried and rejected**. The honest headline is
*architecture (how evidence is read), not content/hints/supervision/curriculum, is the lever —
and it is substitutive with the scalars, not additive*.

---

## 1. The objection this workstream answers

SPECTRE's adaptive wins come from hand-compiled scalar programs (`coverage`, `waste`,
`repeat`, `regroup`, `dead`, `jaccard`) that run over the failure records at tensorize time.
A reviewer can call that hand-engineering. The workstream asks the falsifiable question: **can
the network learn, from the raw failure-record tokens, what those compiled programs extract?**
If yes, the typed scalars demote from *need-to-have* to scaffolding/ceiling. If no, that is
itself a publishable finding (*compiled relational evidence programs beat induced ones at robot
data scales*), and the deployed method stands unchanged either way.

**Scope guard (held throughout):** every change is **additive, flag-gated, zero-init**; flag-off
adds no `state_dict` keys and old checkpoints load `strict=True`; the deployed pipeline was never
confounded. DD2D (`dd2d_v4`) is primary (largest measured gap). Development ran **1 seed** with a
**paired bootstrap over problems** as the gate; headline numbers are **3 seeds**.

## 2. Plan coverage — what we followed, deviated, or dropped

| Plan item (`failed_records_fix.md`) | Outcome | What actually happened |
|---|---|---|
| **P-0** architecture audit | **Followed** | Confirmed the evidence cross-attention query is the **pooled** candidate embedding (`model.py`), so step-level joins are architecturally impossible without a pre-pooling interaction. This is the pivotal finding — it made F-B2 mandatory, not optional. |
| **P-1** certificate-record holdout-off | **Followed → negative** | Added the `--no-record-holdout` flag + census test. The holdout is **inert**: disabling it did not recover FP. Ruled out hypothesis **C4a**. |
| **P-2** sufficiency audit | **Followed → refined** | `coverage`/`waste`/`repeat` are **recoverable** from the token stream (content hypothesis **C1 ruled out for the FP-relevant scalars**). `regroup` is the genuine exception (its establishing-step schema is dropped), but `regroup` is ~0–1% and FP-irrelevant, so it does not motivate enrichment. |
| **P-3** decoding probe | **Deviated (folded into P-2/step-join)** | Not run as a standalone linear-probe experiment; the sufficiency audit + the step-join result together answered "content is present, architecture is the bottleneck," which is what P-3 was meant to disambiguate. |
| **P-4** teachability curve | **Dropped (design decision)** | This was the entry point to **F-C1 auxiliary supervision**, which we cut on principle (§3). Without F-C1 in scope, the teachability curve had no consumer, so it was removed. |
| **F-A** content enrichment (rung-1 evidence steps) | **Built, then cut** | The `--record-mode steps` pathway (`RecordStepEncoder`, `build_evidence_steps`) was fully built and tested. **Empirically inert alone** (`fr_steps` −0.04) and **harmful combined** with the step-join (attention dilution). Kept flag-gated off per the build-then-disable convention; **not pursued**. |
| **F-B2** pre-pooling step-join | **Built → the lever** | `StepJoin` inserts a candidate-step × evidence cross-attention **before** the PMA pool. Scalar-free, over the **summary** record tokens (not enriched steps). **The only arm that moved FP** — recovers ~25% of the scalar gap. **Now in the deployed recipe.** |
| **F-B1** match-primitive edge biases | **Built → rejected** | `--step-join-match-bias` adds exact candidate-step × record match indicators as attention biases. **Hurts** (`fr_join_mb` +1.37 vs `fr_join`) — the hard hints degrade the soft learned join. Kept flag-gated off; a dead end. |
| **F-C1** auxiliary scalar supervision | **Dropped (design decision)** | Cut on the user's principle: anyone able to *supervise* with the compiled scalars could just *feed them in* — it is redundant and undercuts the "learned from raw evidence" claim. Never built. |
| **F-C2** rollout-aligned large-|F| curriculum | **Built → negative** | `--context-mode rollout` + `fc2_build_phi.py`. Reshaped the training |F| to each episode's deployment visit distribution `Uniform{0..φₑ}`. **Decisively worse** (§4). |
| **Rung 2** full flight recorder | **Not built** | Descoped once rung-1 content was shown inert; the ≥25% gate for the deferred rungs was cleared only by the *architecture* arm, not content. |

## 3. Design decisions (with rationale)

- **Shared-encoder reuse (load-bearing).** Evidence steps and candidate steps are encoded by the
  **same** `CandidateEncoder` weights (`encode_steps`/`pool_steps`, a byte-identical refactor of the
  old `CandidateEncoder.forward`). A failed `place_short(b)` and the current `place_short(b)` become
  near-identical vectors *by construction*, turning the relational join into a similarity in a shared
  space — the one primitive attention is good at — at near-zero new parameters.
- **Additive zero-init branches, never widened `Linear`s.** Every new submodule (`RecordStepEncoder`,
  `StepJoin`) is built *last* and conditionally, so a flag-off model keeps its exact init draws and is
  zero-init-identical at step 0. This preserved the D-8 exact-absence equivalence oracle (old
  checkpoints load `strict=True`).
- **Step-join over *summary* tokens, not enriched steps.** The winning arm (`fr_join`) sets
  `--step-join` **without** `--record-mode steps`. So the content is the plain rung-0 summary record
  tokens; only the *architecture* (pre-pooling join) changed. This is what isolates "architecture is
  the lever, content is not."
- **Cut C1/F-A/F-B1/F-C1 rather than stack them.** Each "add more information/hints/supervision" fix was
  inert or harmful; stacking them diluted the join. The kept machinery stays one flag away (build-then-
  disable), but the deployed direction is the single clean lever.
- **F-C2 = per-episode `Uniform{0..φₑ}` with a 30% `|F|=0` floor.** φₑ is a reference policy's deployed
  FP on the episode, so `Uniform{0..φₑ}` *is* that rollout's per-episode |F| visit distribution; the 30%
  static floor (user directive) preserves the |F|=0 training mass the static ranker needs.

## 4. What was learned (numbers)

All DD2D `dd2d_v4`, test n=100, 3 seeds, uncensored deployed FP. Matched anchors:
**floor `abl_floor` 15.73 ± 0.68** (jaccard backbone, no adaptive features) →
**ceiling `abl_all` 7.45 ± 0.93** (all scalars). Gap = **8.28**. gap-closure = (floor − arm)/gap.

| arm | scalars | ALL FP | s1 | gap-closure | note |
|---|---|---|---|---|---|
| `fr_summary` (records, pooled query) | none | 15.13 ± 1.42 | 16.20 ± 5.59 | **7%** | raw records ≈ inert |
| **`fr_join`** (step-join, F-B2) | **none** | **13.66 ± 1.90** | 17.05 ± 7.02 | **~25%** | −2.08 [−4.48, +0.02] vs floor (grazes 0); **the lever** |
| `fr_join_mb` (+ match-bias, F-B1) | none | 15.02 ± 2.35 | 18.15 ± 8.53 | 9% | +1.37 vs `fr_join` — **hurts** |
| `fr_all_join` (join + all scalars) | all | 7.23 ± 0.75 | 10.09 ± 2.14 | ~103% | ≈ ceiling → **substitutive** |
| `abl_all` (scalars, ceiling) | all | 7.45 ± 0.93 | 10.12 ± 2.46 | 100% | — |
| `fr_join_fc2` (F-C2 curriculum) | none | **16.39 ± 3.05** | 20.20 ± 8.74 | negative | **+2.73 [+1.18, +4.37] vs `fr_join` — worse** |

**Four things learned:**
1. **Architecture, not content, is the lever.** Only the pre-pooling step-join moved FP; enriched
   content (C1), match-hints (F-B1), and supervision (F-C1) were inert, harmful, or cut.
2. **~25% recovery, ~3× raw records** — but the CI grazes 0, so it is real-but-s1-variance-marginal at
   3 seeds. (An earlier 2-seed "−2.38 confirmed" did **not** survive to 3 seeds; corrected.)
3. **Substitutive, not additive.** Stacking the join on the full scalars changes nothing
   (`fr_all_join` 7.23 ≈ `abl_all` 7.45) — it learns a *subset* of the scalars' signal, never new signal.
4. **The scalars stabilize s1**, exactly where the raw-evidence join is noisiest (`fr_join` s1 ±7.02 vs
   `abl_all` s1 ±2.46). That is where hand-computation earns its keep.

### 4.1 F-C2 negative, in detail (2026-08-23)

The rollout-aligned curriculum made `fr_join` **worse in both mean and variance** (ALL 13.66→16.39,
+2.73 CI excludes 0; s1 std 7.02→8.74). Two causes, both plausible:

- **(A) conceptual — visit-aligned ≠ FP-value-aligned.** FP is time-to-*first*-success: the small-|F|
  decisions (who to try first) dominate the metric because the first success ends the episode. The
  visit histogram weights every step equally, so the curriculum reallocated capacity *away* from the
  decisive small-|F| regime (compounded by lowering `p_empty` 0.35→0.30).
- **(B) implementation — an untamed tail.** The realized training |F| stayed small in the bulk (p50 0,
  p90 14) but grew a **fat tail** (p99 79, max 184, ~5% of examples > |F|=30, s2 to 184) that plausibly
  destabilized training — consistent with the *variance* rising. A φ-capped variant (cap at ~p90≈49 or
  tighter) would disambiguate A vs B and is the more guardrail-faithful design; **not yet run**.

**Standing:** F-C2 as-built is a clean negative; the capped disambiguation is the one open thread.

## 5. What was ultimately implemented (flags + modules)

**Flags** (all default to current behavior; the deployed recipe opts in only to `--step-join`):

| flag (`train.py`) | `TrainConfig` field | default | in deployed recipe? |
|---|---|---|---|
| `--no-record-holdout` | `record_holdout` (inverted) | True (holdout on) | no (inert) |
| `--record-mode {summary,steps}` | `record_mode` | `summary` | no (content inert) |
| `--step-join` | `use_step_join` | False | **YES (2026-08-23)** |
| `--step-join-match-bias` | `step_join_match_bias` | False | no (hurts) |
| `--context-mode {uniform,rollout}` + `--phi-path` | `context_mode`/`phi_path` | `uniform`/`""` | no (F-C2 negative) |

**Modules:**
- `dataset.py`: `build_record_arrays(hold_out_proof_tier=…)`, `build_evidence_steps(…)` (rung-1 arrays),
  `sample_context(…, phi=…)` (F-C2 draw).
- `encoders.py`: `CandidateEncoder.encode_steps` / `pool_steps` (byte-identical split of the old forward);
  `SpectreExample`/`SpectreBatch` `rec_step_*` fields.
- `model.py`: `RecordStepEncoder` (rung-1 evidence-step tokens), `StepJoin` (pre-pooling join, optional
  match-bias), `step_match_indicators`/`_arg_bitmask` (F-B1 exact indicators).
- `experiments/spectre/fc2_build_phi.py`: reference-rollout → `{problem_id: φₑ}` JSON.
- `experiments/spectre/spectre_sweep.py`: presets `failed_records` (fr_summary/fr_join/fr_join_mb),
  `failed_records_restock`, `failed_records_fc2`; `--step-join` added to `v3final`.
- Tests: `test_record_holdout.py`, `test_rung1_steps.py`, `test_fc2_context.py`.

**Deployed recipe change (2026-08-23):** `--step-join` added to `v3final` (spectre_sweep.py),
`sb2d_finalize.sh`, `restock3d_v3_train.sh`, and the CLAUDE.md command. It is scalar-free and
off-byte-identical / on-zero-init-additive; on DD2D-scalars-on it is a **within-noise tie** (substitutive),
so this is **future-proofing, not a measured deployed FP win**. **Current deployed checkpoints predate it**
and pick it up on their next full retrain; the SB2D/restock3d adoptions are **unmeasured** on those envs
(measured only on DD2D) and should be re-measured on their next retrain.

## 6. Current architecture — how the evidence pathway works now

Per candidate, the scorer combines three things: the candidate skeleton embedding, the (optional)
failure-**evidence** memory, and the compiled scalar columns. The 2026-08 changes affect only how the
candidate embedding is formed and what feeds the evidence memory:

1. **Evidence memory** `fact_tok` is, in precedence order: rung-1 evidence-**step** tokens
   (`RecordStepEncoder`, if `record_mode="steps"`) → **summary** record tokens (`RecordEncoder`, the
   deployed path) → legacy hint facts. Summary vs step tokens are never both built (they encode the same
   failures).
2. **Candidate embedding.** *Without* step-join: the candidate's per-step tokens are pooled by the PMA
   (`CandidateEncoder`) into one vector — this is the **pooled query** P-0 flagged, so evidence joins can
   only happen at the pooled level. *With* `--step-join` (deployed): the candidate's per-step tokens
   **cross-attend over `fact_tok` before pooling** (`StepJoin`, zero-init residual), so per-step
   candidate × evidence joins are representable; the result is then pooled.
3. **Scorer.** The pooled (now evidence-aware) candidate embedding is the **query** into the original
   post-pooling evidence cross-attention (`EvidenceCrossAttentionScorer`), then fused with the scalar
   columns at the head. So with step-join on there are **two** evidence-interaction points — the new
   pre-pooling join and the original post-pooling attention.

The compiled scalars (`coverage`/`waste`/`repeat`/`dead`/`jaccard`, plus `--state-delta`) are **unchanged**
and remain the deployed method's headline signal; the step-join is a scalar-free architectural addition that
learns a *subset* of the same signal from the raw tokens.

## 7. Status & open items

- **Shipped & standing:** step-join in the deployed recipe (future-proofing); the diagnostic arc
  (C4a out, C1 out for FP-relevant scalars, C2 confirmed → step-join fixes it); all machinery flag-gated,
  512 fast tests green, deployed method byte-unchanged.
- **Open:** the **F-C2 φ-capped** disambiguation (A conceptual vs B tail-instability); an s1-variance
  settle for `fr_join`'s marginal −2.08 CI (more seeds / wider selector); re-measuring `--step-join` on
  **SB2D and restock3d_v3** (adopted but unmeasured there).
- **Cut / not pursued:** F-A content enrichment (`record_mode=steps`), F-B1 match-bias, F-C1 aux
  supervision, rung-2 flight recorder, `regroup`'s establishing-step schema.

**Citations:** notebook/decisions `07-stickbutton2d` entries 2026-08-22 (step-join lever, holdout inert,
adaptive-feature isolation) and 2026-08-23 (learned-pathway final results); proposal §6 "Learned pathway
from raw failure evidence"; the frozen plan [`failed_records_fix.md`](failed_records_fix.md).
