# Why does SPECTRE work on DD2D? — Hypotheses & Diagnostic Plan (v2)

*Updated 2026-07-14 after the implementation audit (`spectre_audits_07-14.md`).
Supersedes v1. Changes: audit questions resolved (old T0), id-leakage hypothesis
retired, the length-recalibration hypothesis upgraded to near-forced-by-elimination,
one new cheap test added, and two intervention tests corrected for the actual
input representation `[s0, ops, sL]`.*

---

## 0. Facts established by the audit (no longer hypotheses)

- **Metric & bookkeeping are clean.** The reported number is rollout-FP
  (= attempts-to-first-success − 1). All 124 test problems have solvable 200-skeleton
  pools; no censoring or exclusion path ever fired; all four methods scored the
  identical pools, differing only in ordering/scoring. Every mean in the table is a
  true uncensored mean.
- **Canonicalization is per-episode**, with one object-renumbering shared across the
  candidate pool and the failed set, augmentation ON, canonical order alphabetical.
  Consequences: (a) within an episode, object identity **is** trackable between failed
  and remaining skeletons — identity-based adaptation was architecturally possible;
  (b) across training examples, id *values* were randomized, so the model cannot have
  attached meaning to any particular id.
- **Input representation is `[STATE_0, OP_1 … OP_L, STATE_L]`** — initial state,
  operators, final state; no intermediate states. In DD2D, s_L re-encodes the staged
  subset as `on-buffer` atoms, so subset identity and plan length each enter e(s)
  through **two channels** (operator arguments and s_L atoms).
- **π = 0** for both SPECTRE arms (no prior), and **both PIGINet and SPECTRE use a
  listwise rollout-aligned loss**. The SPECTRE-vs-PIGINet comparison therefore
  isolates two axes only: input substrate, and (for the adaptive arm) adaptivity.
- **Retired: H6 (id-ordering leakage).** Both leakage conditions fail (augmentation
  on; canonical order uncorrelated with planner/geometry).

Reference table (mean FP, per stratum; s0 = target open … s3 = 3+ blockers):

| method | s0 | s1 | s2 | s3 | ALL |
|---|---|---|---|---|---|
| astar-dist | 0.00 | 1.81 | 16.07 | 122.76 | 33.01 |
| PIGINet_v3 | 4.34 | 12.32 | 17.83 | 49.28 | 20.39 |
| SPECTRE-adaptive | 2.57 | 5.95 | 30.69 | 41.36 | 19.23 |
| SPECTRE-static | 16.14 | 14.97 | 27.14 | 39.18 | 23.75 |

What the three comparisons isolate:
adaptive-vs-static = the context module c_t (gains only on s0/s1: +13.6, +9.0;
zero-to-negative on s2/s3); static-vs-astar = the learned one-shot ranking (the big
s3 win, 123 → 39, lives here, so it is **not** adaptation); SPECTRE-vs-PIGINet =
substrate + adaptivity (small overall gap, sign flips by stratum).

---

## 1. The elimination argument (new, and central)

For SPECTRE-static's one-shot ranking, list everything that can differ between two
candidates in the same episode:

1. s₀ — shared by all candidates: zero discriminative signal.
2. π — zero by configuration.
3. Object ids — exchangeable by augmentation: id values carry no meaning.
4. s_L — the staged subset again: meaningless ids + an `on-buffer` count (≈ length).
5. Plan length / staged-item count, and whether an operator touches the
   target-flagged object (the target is distinguished by a predicate, not an id).

Items 1–4 contribute nothing or reduce to item 5. So the **only usable
cross-candidate signal for the static ranking is plan length (plus the target
flag)** — there is no channel through which SPECTRE-static could know *which*
same-size subset is correct. The adaptive model has exactly one extra resource:
within-episode identity linkage to the failed set.

---

## 2. Hypotheses

Tags: **[near-forced]** (true unless the code contradicts the audit),
**[supported]** (consistent with the observed pattern, direct test pending),
**[first-principles]** (argued from structure/math, with a measurable check),
**[conjecture]**.

**H1 — The s3 win is length recalibration, not subset knowledge. [near-forced]**
astar-dist charges per action, so on s3 problems it wades through most singleton and
pair plans (~70–80 attempts of the ~84-attempt gap, toy arithmetic) before reaching
deep into the triples. A static model that learned "prefer longer staging plans"
skips that prefix while knowing nothing about which triple is right. By the
elimination argument, length is also the only signal it *could* use. One bias
explains the whole static profile: catastrophic s0 (16.1 on a stratum solved in 0),
bad s1, neutral s2, excellent s3.

**H2 — c_t does size escalation, not subset identification. [supported]**
The context module's gains sit exactly where length-regime mistakes live (s0/s1) and
vanish where same-size subset choice is the problem (s2/s3). Size statistics of the
failed set ("short plans keep failing → escalate") are the easiest thing for a
64-dim pooled vector to carry and are unambiguous evidence regardless of *which*
objects were involved.

**H3 — The adaptive gain is mostly repair of a self-inflicted wound. [supported]**
SPECTRE-static ranks retrieve-only ~17th on s0. The adaptive model starts from the
same ranking (c₀ at step 1), fails once or twice, then escapes. Measured against the
best *static* order rather than SPECTRE-static, adaptive SPECTRE still loses on
s0/s1 (2.6 vs 0.0; 6.0 vs 1.8).

**H4 — Outcome-only identity reasoning has a low ceiling. [first-principles]**
A failure near the truth ({o0,o1} when the answer is {o0,o1,o2}) and a failure
nowhere near it ({o5}) present identically as FAIL — nothing in the inputs breaks
the symmetry. Toy result: with one true blocking triple and *uniformly* chosen
failed pairs, "promote supersets of failed sets" yields exactly the base-rate hit
probability — zero gain. A nonzero version exists only insofar as attempted-and-
failed sets are pre-enriched in true blockers by whatever ranked them. Post-audit
nuance: the pool is astar-dist's top-200, which is proximity-enriched by
construction, so enrichment may be non-trivial — measure it (T5) rather than assume
the ceiling is zero. Since the audit confirmed identity reasoning was
architecturally possible, a null result on s2/s3 has exactly two remaining
explanations: the ceiling was too low (this hypothesis), or the model never learned
to use identity (T1 separates them).

**H5 — Shared training-mix length bias. [hypothesis]**
With balanced training strata, ~75% of problems require staging, so both learned
rankers plausibly absorbed "deprioritize short plans." PIGINet's geometry partially
rescues it on s0 (4.3 vs 16.1); SPECTRE has no rescue.

**H6 — PIGINet's image pathway contributes little on DD2D. [conjecture]**
A listwise-trained model with images and poses needs 12.3 attempts on s1 — a stratum
a hand-coded proximity heuristic nearly solves (1.8). Candidate cause: frozen CLIP
ViT-B/32 features of top-down polygon renders carry little grasp-corridor /
packing signal. If true, "SPECTRE beats PIGINet" partly means "PIGINet failed to
convert geometry into ranking value on this benchmark."

**Unified account (conjunction of H1 + H2, pending T0–T2):** SPECTRE on DD2D ≈ a
learned plan-length prior (static) plus a length-escalation ladder (adaptive) —
both reproducible in a few lines without learning — and its headline win over
PIGINet reflects that this length policy beats PIGINet's geometric ranking on this
stratum mix.

---

## 3. Tests

All run on the recorded pools — no new refinement compute. Ordered by cost. The
audit's own reproduction snippets (`spectre_audits_07-14.md` §Verification) are the
template for the data reads.

### T0 — Length-R² check (new; ~10 lines; do first)
Per episode, predict SPECTRE-static's scores from plan length alone (R² and rank
correlation; optionally add the target-flag feature). H1 predicts near-monotone
ranking in length, residuals ≈ noise. If R² is high, H1 is confirmed in one pass.
→ **H1**.

### T1 — Context surgery at eval time (decisive for H2 vs. "identity unused"; no retraining)
Rerun SPECTRE-adaptive with modified failure contexts:
- **Length-only:** replace each failed skeleton with a synthetic one of the same
  length but random object ids. Unchanged performance ⇒ identity content unused ⇒ H2.
- **Identity-scrambled:** re-encode failed skeletons under a different renumbering
  than the remaining candidates.
- **Cross-episode swap:** context from another episode with matched |F|
  (matched-stratum vs mismatched-stratum variants).

**Implementation requirement (from Finding 2):** identity enters e(s) through both
operator arguments *and* s_L's `on-buffer` atoms. Every intervention must modify
both channels consistently (synthetic contexts need consistent synthetic s_L's;
scrambles must hit ops and s_L atoms together), or the intervention is broken and
will produce a false "identity unused" conclusion. → **H2, H4**.

### T2 — Rank decomposition on s3 (confirmation check for H1, with preregistered prediction)
Split the rank of the first feasible plan into (# size-≤2 plans tried before it) +
(# wrong size-≥3 plans tried before it). Preregistered prediction under H1 +
elimination argument: SPECTRE-static's gain over astar-dist sits ~entirely in term 1,
and term 2 matches the random-within-size null. A term-2 result *below* the null
would contradict the elimination argument and demand a code-level explanation.
→ **H1**.

### T3 — Hand-coded nulls (each ~50 lines; become paper baselines)
- **Size-3-first re-sort** of the astar-dist pool (ties by original order).
  Preregistered prediction: reproduces SPECTRE-static's whole per-stratum profile
  (good s3, terrible s0/s1) with zero learning.
- **Stratum ladder:** retrieve-only first, then size-1 (slack order), then size-2, …
  escalating on failure — pure H2 mechanism, no learning. This is now the critical
  null for the adaptive arm: if it matches SPECTRE-adaptive overall, the learned
  method on DD2D is a two-line heuristic.
- **Naive-Bayes over object ids** (RT2D's B4): if it beats SPECTRE-adaptive on
  s2/s3, identity headroom exists that the learned model missed.
- **Random order** row, for context on every mean.
→ **H1, H2, H3**.

### T4 — Rank-of-truth trajectories + case studies (qualitative core)
Per stratum × method, plot the rank of (a) retrieve-only and (b) the eventually-
successful plan vs attempt step t. Prediction: on s0, static parks retrieve-only
~rank 17; adaptive catapults it into the top 3 after failure #1. Pair with
`render_scene` on a few s2/s3 instances (blockers highlighted), printing each
method's top-5 subsets at t = 1, 2, 3. Also record where PIGINet ranks retrieve-only
on s0 (separates H5 from H6). → **H2, H3, H5**.

### T5 — Enrichment measurement (prices H4's ceiling; now more interesting post-audit)
Among attempted-and-failed size-2 subsets on s3 problems, measure the fraction p
that are subsets of a true minimal blocking set (exact labels from the enumerator),
against the pool base rate. Because the pool is astar-dist-enriched, p may be
non-trivial. Interpretation matrix with T1:
- p ≈ base rate → identity mechanism was dead on arrival in DD2D (two-line paper
  argument for v2.1).
- p ≫ base rate but T1 shows identity unused → usable signal existed at a modest
  ceiling; the model didn't learn it (learnability, not information — phrase
  carefully).
→ **H4**.

### T6 — Superset-promotion statistic (quantifies the H4 mechanism directly)
After each failure of subset S, compare mean rank change of candidates ⊋ S vs
size-matched low-overlap candidates. H2 predicts no difference beyond size.
→ **H2 vs H4-mechanism**.

### T7 — Linear probes on c_t (interpretability)
From frozen c_t, linearly predict: |F_t|, mean/max failed length (expected to
dominate), stratum, failed-object-id indicator vector, true-blocker indicator
vector. H2 predicts the first three probe well, the last two poorly. → **H2**.

### T8 — PIGINet image ablation
Eval- or retrain-ablate the image channel. Unchanged performance ⇒ images earn
nothing on DD2D (H6), and the substrate comparison collapses to
architecture-on-abstract-features. → **H6 vs H5**.

### T9 — Paired statistics for any headline claim
Per-problem paired differences with bootstrap CIs, plus medians. The 1.16-FP
overall SPECTRE-vs-PIGINet gap is uninterpretable from marginal ±20–30 stds.
Be consistent about seed-averaging vs pooling (the adaptive numbers are
seed-averaged per problem — fractional max FP 145.67). → guards everything above.

---

## 4. Accounting caveats (carry into any writeup)

- **Tie handling differs by arm:** static FP awards 0.5 credit on exact score ties;
  the adaptive rollout breaks ties by argmax. Static-vs-adaptive deltas are not
  identical definitions at sub-attempt resolution — one more reason not to interpret
  the sign of the small negative s2/s3 "premiums" (−3.6, −2.2).
- **Latent asymmetric-censoring risk (future runs only):** static methods have no
  sequential budget; the adaptive rollout right-censors at FP = 200. Harmless here
  only because pool size = budget = 200 and every pool is solvable. Any future
  dataset breaking either property re-opens this.
- **Stratum labels** rely on the Day-1 labeler (`marginal(budget)` fallback; negative
  certificate deferred), so stratum-level claims inherit the snapshot's honesty gate.
- The H4 toy assumes a unique minimal blocking set and success ⇔ coverage (ignores
  packing, multiple minimal sets); T5 measures the real quantity, so no conclusion
  rests on the idealization.

---

## 5. Expected story if the leading hypotheses hold

Outcome-only failure evidence supports **coarse, size-level** adaptation well (real,
but matchable by a trivial ladder baseline) and **fine, subset-level** adaptation
only weakly (low ceiling — the {o5} vs {o0,o1} failure symmetry; ceiling priced by
T5). The s3 win over astar-dist is a *static* length-recalibration effect,
reproducible by a one-line re-sort — and by the elimination argument, length was the
only signal available to the static ranker in the first place. Together this is a
first-principles + empirical case that breaking the failure symmetry requires typed,
diagnostic evidence — e.g. "at the deepest reached state, o2 still collided with the
target grasp" — i.e., exactly the SPECTREv2.1 evidence-record design. The confusing
win becomes the motivating diagnostic for the next method.

**Minimal first pass:** T0 (length-R²) → T3 re-sort + ladder baselines → T2
decomposition. All three are analysis over existing records; together they confirm
or kill the unified account in a day or two.
