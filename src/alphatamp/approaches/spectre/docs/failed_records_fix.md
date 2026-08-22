# Failure-record tokens — inertness diagnosis, enrichment ladder, and learned-pathway probes

**Status:** probe plan + enrichment spec, 2026-08-22. Nothing built except where marked.
**Epistemic tags:** `[M]` measured · `[D]` by-construction · `[P]` registered prediction · `[U]` unverified.
**Authoritative sources:** `as_built.md` (§3 architecture, §5 records/aggregation, §10.5 wall-clock), `dataset.py` (certificate-record token holdout — defined behaviorally in the §2 C4a box; dead/blocked; overlap vector — line refs :710/:976/:999 as of 2026-08, locate by behavior if drifted), `encoders.py` (RecordEncoder, CandidateEncoder), `model.py` (EvidenceCrossAttentionScorer), stored pools + `refiner_metadata` (no re-collection anywhere in this plan).
**Scope guard:** this is the **learned-pathway workstream**. The deployed method (scalars on) is untouched and remains the paper's headline pipeline; the repeat/regroup + fixed-coverage workstream proceeds in parallel. Coordinate arms so the two workstreams never share a confounded comparison.

---

## 0. Framing (read first — it defines what "fixed" means)

The scalar features (dead, jaccard, coverage, waste, repeat, regroup) consume **nothing but the failure records + the candidate skeletons**. So the question is not "records vs features" — it is *where the relational program runs*: compiled by us at tensorize time, or induced by the network from examples. The target claim is: *a model can learn to extract from raw failure evidence what the compiled programs extract, given the right input representation, primitives, and supervision.* Every probe below exists to make that claim falsifiable — including falsifiable in the negative direction, which is also a publishable finding (§8).

## 1. The result being explained

| arm | DD2D FP | SB2D | v3 |
|---|---|---|---|
| SPECTRE-static | 19.80 `[M]` | 2.06 `[M]` | 11.05 `[M]` |
| + records-only (tokens, no scalar evidence feats) | ≈ 17.8 (−2) `[M]` | ≈ inert `[M]` | ≈ inert `[M]` |
| + full scalar pathway | ≈ 6.3–6.9 (−13) `[M]` | 1.75 `[M]` | (retrain in flight) |

Tokens alone capture ~15% of what the compiled programs capture on DD2D, ~0% elsewhere.

## 2. Hypotheses for inertness (compatible; probes assign weight)

- **C1 — content gap** `[D]`: required inputs are structurally absent from the token stream. Established over the token-anatomy analysis: tokens are attempt-anonymous (no candidate-id → co-failures within one plan cannot be grouped), carry no establishing steps (the seating chart is absent), and state deltas are level-blind (`Stored` carries no section). Consequence: **regroup is provably not computable from the current stream**; coverage is in-principle computable (recurrence-by-effects + touch-sets + culprit tags all derivable). No architecture or training change can fix a missing input.
- **C2 — primitive gap** `[U]`: the required computation is an exact relational join (equality tests on episode-local tags whose global meaning is deliberately stripped by permutation augmentation; order comparisons; universal quantification across records) — a program, not a similarity. Soft dot-product attention is weak at near-exact matching. Sharper sub-suspect: if the evidence cross-attention query is the **pooled** candidate embedding rather than per-step tokens, step-level joins are architecturally impossible regardless of training.
- **C3 — sample inefficiency** `[U]`: the join is representable but not learnable from ~500 episodes through a listwise loss emitting one scalar per candidate (brutal credit assignment for a multi-step relational program).
- **C4 — plumbing handicaps** `[D, partially]`: (a) **the certificate-record token holdout** (see box below): at tensorize time, records that are both proof-tier and provable are filtered out of the evidence token stream before the model sees them. On DD2D that is the certificate-grade family — the tokens-only arm never saw the records whose signal the `dead` column carries, so the measured −2 is an underestimate of what tokens can learn; (b) scalars enjoy late fusion at the head (short gradient path) while evidence attention is buried; (c) the training-time failure-context sampler may under-represent large-|F| contexts where evidence matters most.

> **The certificate-record token holdout, self-contained** (do not rely on line numbers — locate it by behavior). During failure-context tensorization, any record satisfying **proof-tier ∧ `proves_failure()`** is dropped from the token stream: it never becomes a `RecordEncoder` token. *Proof-tier* = the record's schema is declared monotone ∧ local ∧ exact via `QueryAxioms` (the same predicate that feeds the `dead`/`blocked` computation); *provable* = `exhausted and not budget_exhausted`. **To find it:** search `dataset.py` for the failure-context → token loop and look for a filter condition combining the proof-tier check (historically via a `licenses_demotion()`-style call — a misleading name; it computes eligibility, it does not apply demotion) with `proves_failure()`; it was at `dataset.py:710` as of 2026-08, with a stale comment along the lines of "handled structurally by demotion, not learned." **Why it exists:** when the outside-the-net proof demotion was live, holding these records out of the learned pathway was a de-duplication ("the symbolic machinery already handles them"). The demotion was cut 2026-07-30 (`apply_demotion=False`), but the holdout remained; in the deployed config it is near-harmless because the same signal reaches the model through the `dead` scalar — **but in a tokens-only arm (scalars off), these records reach the model through no channel at all.** Note the filter gates per *schema*: this is also why v3's `step_certificate` flag was created instead of declaring v3 place schemas proof-tier (which would have yanked F2 crowding records out of the stream too). Verify empirically before P-1: count tokens per environment with the filter on vs. off — DD2D should show a large delta (its census: ~92% blameless-provable records), SB2D/v3 whatever their proof-tier declarations imply; record the counts in the notebook entry.

## 3. Probes (no re-collection; ordered by dependency)

**P-0 — architecture code audit** (minutes). Read `model.py`: is the evidence cross-attention query the pooled candidate embedding or per-step candidate tokens? Pooled → C2's sharp form is confirmed and F-B2 (pre-pooling interaction) becomes mandatory, not optional. Also confirm which record fields actually reach the token (vs. exist in metadata only).

**P-1 — holdout-off rerun** (one cheap DD2D retrain, 3 seeds). Retrain the tokens-only arm with the **certificate-record token holdout disabled** (the C4a box: the tensorize-time filter that drops proof-tier ∧ provable records from the evidence token stream), so that for the first time the tokens-only model actually *sees* the certificate-grade records. Concretely: locate the filter by its behavior (proof-tier check ∧ `proves_failure()` inside the failure-context tokenization loop — at `dataset.py:710` as of 2026-08, but trust the condition, not the line), put it behind a flag defaulting to current behavior, run the arm with the flag off, and first sanity-check the flag with the token-count delta from the C4a box (DD2D large, others per their declarations). This corrects the baseline before any conclusion about learnability — the current −2 number partly measures records withheld, not learning failure. *Registered:* recovers ≥1 FP of the gap `[P]`. All later arms build on the corrected baseline. Leave the deployed (scalars-on) config untouched: there the holdout is near-harmless de-duplication, and changing it would confound the parallel workstream.

**P-2 — sufficiency audit** (symbolic + empirical, hours). Per scalar: derive from definitions whether it is a function of (token-stream content + current-candidate inputs). Then verify empirically: hash (token-bag, candidate) across the dataset; hunt for collisions with different scalar values — any collision is a proof of insufficiency. *Registered:* regroup fails; coverage and waste pass `[P]`. Rerun after each enrichment rung; **acceptance test for "not GIGO" = all four scalars pass.**

**P-3 — decoding probe** (a day, frozen checkpoints). Train a linear/small-MLP probe from the evidence-channel internal representation to predict scalar values. Decodable but unused → C3 (training/credit-assignment). Not decodable → C2 (never got in). Run on the corrected baseline and again on each enrichment rung.

**P-4 — teachability with free labels** (the decisive one, ~a day). The scalars are computable ⇒ supervised training data for them is **unlimited and free**: synthesize (candidate, failure-context) → coverage/repeat/regroup targets from stored pools, millions of examples, no simulator. Train the token pathway alone on this task at increasing data scale; plot the learning curve.
- Fails even at abundant data → **C2 architectural** → F-B fixes.
- Succeeds at scale, but the curve says ~500 episodes is hopeless → **C3** → F-C1 (aux supervision) is the bridge.
- Succeeds at small scale → inertness was C1/C4 all along.

**P-5 — attention-mass audit** (optional diagnostic, hours). On enriched streams, measure where evidence attention lands (failed steps vs establishing steps vs padding). Signal-to-noise instrumentation for the rung-1-vs-rung-2 comparison; not a gate.

**Decision point A** after P-0..P-4: assign weights to C1–C4; pick the fix set. Expected outcome `[P]`: C1 ∧ C4 confirmed, C2 partially (pooled-query risk), C3 real but bridgeable.

## 4. Fix F-A — token enrichment ladder (fixes C1)

**Design principle:** *record what happened; don't compute what it implies.* Attempt-ids, step identities, statuses, sample counts are facts. Coverage is an implication. Enrichment adds facts only. One disclosed exception below.

All evidence steps are encoded by the **shared CandidateEncoder** (same weights as the current candidate) — this is load-bearing: the current candidate's `place_short(b)` and a failed skeleton's `place_short(b)` become near-identical vectors *by construction*, converting the relational join into similarity in a shared embedding space — the one primitive attention is actually good at. Near-zero new parameters (one attempt-id embedding, status/outcome slots).

**Rung 0 — summary-only** (status quo, holdout-off): one token per aggregated (schema, args) record. The corrected baseline.

**Rung 1 — compressed relevant-subset** (the recommended target): per failed attempt, tokens for
- the **failed step** (schema, args, failure marker, exhausted/effort scalars, culprit tags, per-culprit sample counts — see below);
- the **establishing step of each culprit** (the successful step that seated each blamer), status = succeeded-and-blamed;
- a shared **attempt-id** on all of them.
Drop clean successes of non-culprit steps; drop unreached suffixes; transient (non-exhausted) records behind a flag (F-A3). Token budget: blameless failure = **1 token** (identical to rung 0); culprit-bearing = 1 + |culprits|. Episode total ≈ |F|…2|F| — the old budget, with the seating chart, attempt grouping, and level bits restored. *Disclosure:* relevance-filtering is a selection judgment (which facts to keep), sitting between raw recording and feature engineering — defended under the same principle as the existing aggregation; say so in the writeup.

**Rung 2 — full flight recorder**: every step of every failed skeleton, status ∈ {succeeded, effortful-success, failed-here, unreached} (unreached must be un-confusable with succeeded). Budget |F|×L (~300–750 tokens at DD2D |F|=30) — attention dilution is the real cost, not deployment wall-clock (§7). Context cap by **dropping whole attempts** (recent / deepest-reaching first); never pool an attempt into one vector — pooling re-creates the anonymity rung 1 exists to remove.

**F-A2 — un-union culprit multiplicity** (rungs 1–2): keep per-culprit failed-sample counts (log-squashed), recovered from pre-aggregation records in `refiner_metadata` `[verify present in storage — aggregation is a tensorize-time flag]`. "9 of 10 samples hit b" ≠ "5 hit b, 5 hit c". v3 analytic records are single-sample; field degenerates gracefully.

**F-A3 — transient-record flag** (DD2D-only ablation): include/exclude non-exhausted (backjump-resolved) records. *Registered:* small effect `[P]`; decide by measurement.

**Assumption to assert in code:** annotations key by (schema, args), which equals step identity only because no skeleton repeats a ground action (no un-store). Assert it; key by step index if a future domain violates it.

## 5. Fix F-B — architecture (fixes C2; gated on P-0/P-4)

**F-B1 — match-primitive edge biases:** compute exact indicators at tensorize time — (candidate step's args ∩ record's culprits ≠ ∅), (candidate step == failed step, schema+canonical args), (candidate step == an establishing step), (before/after relative position) — and feed them as attention biases on candidate-step × evidence-token pairs (relative-position-bias style). *Boundary disclosure for the paper:* these compute **equality only** — domain-agnostic, content-free; "the model is told what matches and learns what matching means." Flag the choice explicitly rather than letting a reviewer find the line.

**F-B2 — pre-pooling evidence interaction** (mandatory if P-0 finds pooled query): candidate *step tokens* cross-attend over evidence tokens before pooling, so step-level joins are representable at all.

## 6. Fix F-C — training (fixes C3)

**F-C1 — auxiliary supervision / self-distillation:** during training, the evidence pathway additionally predicts the compiled scalars (coverage, waste, repeat, regroup) from tokens — labels free and unlimited (P-4 machinery). **Evaluate tokens-only at inference.** If it works, the paper sentence writes itself: the typed features become training-time scaffolding, absent at test time.

**F-C2 — context-sampling curriculum** (only if P-3 says "decodable but unused"): oversample large-|F| and mixed-family contexts, rollout-aligned.

## 7. Wall-clock note (pre-empting the objection)

A re-rank is one forward pass of a ~324k-param model — milliseconds on the 5090; a refinement attempt costs seconds (abandonment caps 2 s / 10 s). One avoided attempt pays for ~10³ re-ranks; §10.5's measured result is SPECTRE-adaptive **fastest under the cap** on both envs, re-ranking included. |F|×L costs training memory and signal-to-noise, not deployment time. Report eval wall-clock per arm anyway (it is the honest check).

## 8. Arms, gates, and registered predictions

Matched settings, 3 checksum-distinct seeds, one variable per rung, DD2D primary (largest measured gap), SB2D/v3 as generality confirms. Report per arm: FP and **fraction of scalar-gap closed** = (FP_static − FP_arm) / (FP_static − FP_scalars-on).

| rung | arm (all tokens-only at eval unless noted) |
|---|---|
| 0 | status-quo tokens-only, holdout OFF (P-1) — corrected baseline |
| 1 | + rung-1 compressed enrichment (F-A, F-A2) |
| 2 | + match biases / pre-pooling (F-B) |
| 3 | + auxiliary supervision (F-C1) |
| 4 | rung-2 flight recorder swap-in for rung 1 (context-capped) |
| ref | deployed scalars-on arm — the ceiling reference |
| opt | tokens(enriched) + scalars combined — interaction rung, last |

**Registered predictions** `[P]`: P-2 regroup insufficient at rung 0, all-sufficient at rung 1+. Rung 1 ≈ rung 4 on FP within CI at ~10× fewer tokens (compressed carries the signal). Rungs 1–3 cumulative on DD2D; the largest single delta from F-C1. **Headline gate:** tokens-only-at-eval closes ≥50% of the scalar gap on DD2D and is CI-positive on ≥1 other env → the "model learns to use raw failure evidence, typed features as scaffolding/ceiling" paper. Below that → the honest alternative headline: *compiled relational evidence programs beat induced ones at robot data scales, by a measured margin* — a real finding, and the deployed scalars-on method stands unchanged either way.

**Abort criteria:** P-4 flat at abundant data after F-B fixes → stop chasing parity; write the negative result with the curve as the figure. ICRA-clock guard: P-0..P-4 are bounded (≈ a week with retrains); rungs 2–4 proceed only if rung 1 + corrected baseline show ≥25% gap-closure — otherwise this workstream freezes at "diagnosed, negative, disclosed" and effort returns to the deployed pipeline.

## 9. Order of operations

1. P-0 code audit; P-1 holdout-off retrain (launch first — everything reads against it).
2. P-2 sufficiency audit at rung 0; implement rung 1 enrichment; P-2 again (acceptance: all four scalars sufficient).
3. P-4 teachability curves on rung-1 tokens (± F-B variants as arms of the curve) → Decision point A.
4. Rung ladder retrains per §8; P-3/P-5 diagnostics on each.
5. `decisions.md` entry: enrichment principle + relevance-filtering disclosure + (schema,args) assumption; `notebook.md` entries per rung with FP + gap-closure fraction.
