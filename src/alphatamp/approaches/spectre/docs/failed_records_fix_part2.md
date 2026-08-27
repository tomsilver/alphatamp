# Tokens-only pathway, round 2 — truncation, composition, interference

**Status:** probe plan + fix menu, 2026-08-26. Supersedes the fix sections of `failure_records_fix.md` for this workstream; its probe results and hygiene conventions still stand.
**Epistemic tags:** `[M]` measured · `[D]` by-construction · `[P]` registered prediction · `[U]` unverified.
**Scope guard:** the deployed pipeline (scalars on: repeat + coverage + waste + jaccard) is untouched by everything below. All context edits are **tensorize/eval-time on the token stream only** — the scalar inputs (K, coverage, waste, repeat) keep their current aggregation, because 19.8 → ~6.3 was measured with it and silently changing a working pipeline's inputs is the worst place to break the single-variable rule.

---

## 0. Where we stand

Isolated +records arm (tokens as the *only* adaptivity source, jaccard off), DD2D `[M]`:

| arm | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| static | 18.35 | 0.00 | 21.84 | 21.09 | 30.45 |
| +records (StepJoin) | 19.68 | 0.00 | **19.72** | 24.79 | 34.20 |

Pattern (mirrored on SB2D: b2 helps, b3/b5 hurt): tokens help where contexts are small and one record names the answer; hurt where many records must be composed. ALL is a wash of a real s1 gain against an s2/s3 penalty.

**Standing corrections to record in the notebook before proceeding:**
- The fr_join "23% → 43% gap-closure" entry was **confounded** — jaccard was in the floor; the step-join delta was s1-localized, substitutive, CI-grazing (fr_join − fr_summary = −1.47 [−2.97, +0.11]). Strike out and re-tag; do not cite in the paper.
- Earlier +records ≈ 13 FP result was a config slip (+jaccard left on). The isolated number is the table above.

**Ruled out — do not resurrect:**
- **F-A content enrichment** (rung-1/rung-2 token enrichment): measured inert + attention dilution; regroup unused. Machinery stays flag-gated off (build-then-disable), no source removed.
- **F-C1 auxiliary scalar supervision:** ruled out on deployment logic — if the scalars are computable, input them; supervising toward them while withholding them is a demo, not a method.
- **F-C2 context-sampling curriculum:** measured to hurt.
- **regroup:** unbuilt; probes showed repeat carries the F3 signal, regroup mostly inert (chart-rebuild starvation in split-diverse pools `[U]` — optional one-script incidence count if the vocabulary table needs the distinction).

## 1. Hygiene preconditions (before interpreting anything)

- **H-0:** the existing static and +records arms above are multi-seed — attach their paired bootstrap CIs so the s2/s3 penalties (+3.7, +3.8) are stated with intervals; these CIs are also the calibration for what 1-seed screening deltas can and can't distinguish. `[U]`
- **H-1:** audit the certificate-record token holdout state in the isolated +records arm. If it was ON, the DD2D tokens-only model saw only the culprit-bearing slice (~8% of records) and every row below means something different. Record the flag state in the arm's config block.
- **H-2:** verify the +records arm's evidence path matches the deployed StepJoin variant (with/without F-B1 equality biases) and note which.

## 2. Probes — all eval-time on existing checkpoints, ~an afternoon each, no training

Sequencing: **W3 first**, together with W1's step-0 histogram (checkpoint-independent). Rationale for W3's priority: its fix (X2) is *foundational* rather than additive — if interference is real, W1/W2 gains measured on the damaged checkpoint are unreliable and must be re-measured on the X2 model anyway, whereas W1/W2 stacked on a protected substrate are durable. Probing W1/W2 first risks doing them twice; probing W3 first never does. If W3 comes back clean, run W1 step-1 and the W2 sweep on the existing checkpoints as planned.

### W3 — interference: is the damage in the weights? (priority 1; cheapest, most discriminating, foundational fix)

**Idea.** The +records arm is a *jointly retrained model*, not static-plus-a-channel. If its evidence pathway degraded the shared candidate/scene representations during training, it will underperform static **even with nothing to attend to**.

- **Procedure:** eval the +records checkpoints with evidence blanked (F = ∅ behavior — the leakage-invariant path) at all strata; compare to the static arm.
- *Registered (low confidence):* blanked ≈ static within CI — interference is not the primary mechanism `[P]`.
- **If blanked < static at s2/s3 → build fix X2 first** (§3), then re-run W1/W2 probes on the X2 checkpoint — on the damaged checkpoint their readings are discounted; the context isn't the patient, the weights are.
- **If blanked ≈ static:** interference exonerated; X2 is *not* built preemptively (fixing a non-problem adds a config dimension), but it stays in the menu as a substrate option if later token arms show instability — with the caveat that adopting it changes the reference arm for every subsequent comparison.

### W1 — deepest-record truncation (priority 2; probe now, candidate treatment later)

**Idea.** DD2D/SB2D emit records per failed query and aggregate by (schema, args) — a backjumping candidate can contribute several tokens, including records for *earlier* steps that were subsequently resolved by re-binding. A resolved record is a prefix-conditional fact whose premises the search revised: "under bindings the plan abandoned, this query found nothing." The terminal fact is the deepest rejection. Truncating to one record per failed candidate (the deepest) removes revised-premise statements and shrinks the bag at exactly the strata where it drowns the model. **Consistency bonus:** v3 already does this on both paths (`failure_metadata()` deepest-rejection; analytic first-violation) — this aligns DD2D/SB2D token emission with v3's semantics: one emission rule everywhere.

- **Step 0 — sizing histogram** (minutes, storage only): records-per-failed-attempt by stratum, DD2D + SB2D. *Gate:* if multiplicity at s2/s3 is ≈1.1, truncation cannot explain a 3–4 FP penalty — demote W1 to consistency-cleanup and move on. *Registered:* multiplicity ≥ 1.5 at s2/s3 `[P]`.
- **Step 1 — eval-time truncation** on the existing +records checkpoints: filter each attempt's records to the deepest at context-build time. Train/eval mismatch handicaps this arm (model trained on full bags), so improvement *despite* the handicap is strong evidence.
- **Step 2 — one retrained arm** with truncated contexts, 1 seed (screen; multi-seed only if it graduates).
- **Proof-tier exemption variant:** monotone proof-tier records are prefix-robust — valid despite backtracking. In holdout-off arms they're present; run truncation both with and without exempting them (one flag). In holdout-on arms the question is moot.
- **Fragility fold-in (if shipped as treatment):** keep "step 2 burned 9 samples before succeeding" as scalars on the surviving token (total attempt effort via `effort_is_total`, backjump count, # distinct rejected steps) — raw facts, zero extra tokens. Side effect: one token per failed attempt dissolves attempt-anonymity by construction.
- *Registered:* truncation helps s2/s3, ~neutral s1 (s1 contexts mostly single-record already) `[P]`. If the s2/s3 penalty **survives** full truncation, bag *size within attempts* was never the problem — composition across attempts was (→ W2's fix).

### W2 — composition lesion: per-record reads vs cross-record aggregation (priority 3; instrument, never a treatment)

**Idea.** To an ideal reasoner more evidence never hurts (the compiled scalars quantify over the full bag and win by 13 FP — the bag's information is real). If the learned model does *worse* with the full bag than with one record, the failure is model-side: it can read a single incident report but drowns composing thirty.

- **Procedure:** eval the +records checkpoints at s2/s3 with contexts truncated to the k most recent records, k ∈ {1, 2, 4, 8, full}; plot FP vs k. "Most recent" = the failure of the best surviving candidate — the most locally relevant single item; the sweep, not the endpoint, is the finding.
- **Combined cell:** rerun the sweep on deepest-only contexts (W1 × W2 grid) — separates within-attempt multiplicity from cross-attempt composition.
- Train/eval mismatch again cuts *against* small-k arms.
- *Registered:* FP at s2/s3 is non-monotone in k — small-k beats full `[P]`.
- **If confirmed → fix X1** (compiled aggregation, §3): learn the per-(step, record) adjustment, compile the aggregation (fixed sum/max over learned pair scores) — engineered *structure*, learned *content*; one level up the same ladder as StepJoin.

## 3. Fix menu — try in decision-tree order; 1-seed screening, multi-seed only for what survives

- **T1 — ship deepest-only truncation** (if W1 positive): tensorize-time, token stream only, flag-gated, one retrain rung per env. The only item here that is both fix and semantic cleanup; it can ship even at small positive effect on the consistency argument.
- **X1 — compiled aggregation** (if W2's signature confirmed and W1 insufficient): per-(candidate-step, record) learned adjustment; aggregation across records/steps fixed by hand (sum or max, chosen once, disclosed). Rationale: the quantifier is what the compiled scalars provide and the model demonstrably can't induce; give it the quantifier, keep the reads learned.
- **X2 — zero-initialized gated residual + stop-gradient** (if W3 confirms interference): `score = static_score + g·adjustment`, g (or final layer) initialized to 0 so step-0 model ≡ static; g may condition on |F| ("shut up in large contexts"); stop-gradient (or frozen static trunk) so the evidence branch reads shared representations without writing to them. Property, not hope: attaching the channel cannot make the model worse than static.
- **Freeze** (if all probes null, or no fix shows a directional win at screening): the workstream ends at "diagnosed, measured, disclosed."

## 4. Decision tree (joint reading; screening mode)

**Seed policy for this phase:** 1 seed per arm, matched settings. A 1-seed run is a *screen*, not a result: DD2D seed noise is ~±1 FP, so treat |Δ| ≲ 2 FP as "undecided," not "null" — park undecided arms for the multi-seed pass rather than discarding them. Directional wins graduate to the final-numbers protocol (3 checksum-distinct seeds + paired bootstrap CIs); nothing enters the paper or `as_built.md` §10 on 1 seed.

1. **H-1 finds holdout ON** → rerun the isolated arm holdout-off before anything else; all rows below assume the corrected arm.
2. **W3: blanked < static at s2/s3** → interference → try **X2** first. Re-run W1/W2 eval probes on the X2 model — their readings on the damaged checkpoint are discounted.
3. **W3 clean, W1 step-1 removes the s2/s3 penalty** → within-attempt multiplicity was the noise → **T1** confirm-retrain.
4. **W3 clean, W1 insufficient, W2 small-k > full** → cross-attempt composition failure → **X1**, on deepest-only contexts if W1 was directionally positive.
5. **W3 clean, W1 null, W2 monotone (full ≥ all k)** → size and composition both exonerated → re-check hygiene (H-0/H-2); if clean, **freeze**.

Fixes are cheap to try in sequence with Claude Code; combinations (e.g. T1 + X2) are allowed *after* each shows a directional win alone — the single-variable rule applies to attribution, not to how many rungs get attempted.

## 5. Paper phrasing per branch (write the honest sentence now)

- Any fix lands CI-clean: "raw records contribute measurably once [truncated to terminal facts / given a compiled quantifier / gated into a protected score]; compiled programs still carry the majority — measured margin X."
- Freeze branch: "records + step-join contribute at low strata but are net-inert in aggregate without a scalar anchor; end-to-end induction of the relational programs fails at robot data scale — the compiled, environment-agnostic vocabulary is the operative mechanism." The deployed 19.8 → ~6.3 result **is** the failure-evidence claim; tokens-only parity was never load-bearing.

## 6. Order of operations

1. H-0/H-1/H-2 + notebook strike-outs (hours).
2. W3 blanked eval + W1 step-0 histogram (checkpoint-independent) — first batch.
3. Branch: W3 dirty → build X2, then run W1 step-1 + W2 sweep *on the X2 checkpoint*. W3 clean → run W1 step-1 + W2 sweep on the existing checkpoints.
4. Decision tree → try remaining fixes in tree order, 1 seed each; park |Δ| ≲ 2 FP arms as undecided.
5. Graduation pass: every surviving fix (and every parked-undecided arm worth deciding) reruns at 3 seeds + paired bootstrap before any number reaches the paper, `as_built.md`, or a cross-arm comparison.
6. `notebook.md` per probe (tagged 1-seed where applicable); `decisions.md` entry for what shipped and what froze, including branches not taken and why.
