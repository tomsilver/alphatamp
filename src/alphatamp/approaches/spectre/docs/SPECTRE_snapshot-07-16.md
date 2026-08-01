# SPECTRE — Project Snapshot (2026-07-16)

*A standalone, self-contained overview written for a reader with no prior exposure to
this project. It explains what SPECTRE is, exactly how it is currently deployed on the
DD2D environment, and what has been learned and ruled out. It folds in the 2026-07-14
implementation audit and the running lab notebook / decision log, plus the newest
(mid-July 2026) diagnostic results.*

*Status: a **living dated snapshot**, not a frozen archive export. Where it disagrees
with the living docs (`proposal.md`, `decisions.md`, `notebook.md`), those win. A
[glossary](#8--glossary) at the end defines every term; terms are also defined on first
use.*

---

## 0. TL;DR

**SPECTRE is a learned re-ranker for task-and-motion planning (TAMP).** When a robot
planner has a *pool* of candidate high-level plans and must try them one at a time
(each try is expensive), SPECTRE decides **which plan to try next** — from an
*abstract, geometry-free* description of each plan plus *which plans have already
failed*.

Two things are worth stating up front because they shape everything below:

1. **The project's framing changed (2026-06-25).** SPECTRE began as a story about
   *adaptive re-ordering* ("condition on failures to pick the next plan"). It is now
   framed as a **representation question**: *what should a plan-feasibility predictor
   represent plans over* — pixels/low-level state, or a cheaper relational abstraction?
   The adaptive re-ordering is now a secondary, optional component.

2. **On DD2D, SPECTRE is deliberately a "negative control," and the newest diagnostics
   show it behaves as a pure *plan-length* ranker** — it uses essentially no object
   identity or geometry. Surprisingly it still ties a low-level geometry model
   (PIGINet). The resolution (Section 5) is that on DD2D, *plan length is the dominant
   axis of feasibility*, so a length-only policy is competitive even though it "knows"
   almost nothing.

---

## 1. Problem setting (in plain language)

**Bilevel TAMP.** A robot task is solved in two levels:

- **Symbolic level.** A classical planner enumerates a *pool* of candidate
  **skeletons** — sequences of symbolic actions (operators) that reach the goal *on
  paper* (e.g., "pick block A, place it aside, then grasp the target"). Many different
  skeletons can reach the same goal.
- **Geometric level.** A **refiner** tries to turn a skeleton into an actual collision-
  free motion plan with concrete positions. This can **fail** even though the skeleton
  is symbolically valid — the geometry may not work out (no collision-free grasp, no
  room on the buffer, etc.). A skeleton the refiner can realize is **feasible**;
  otherwise **infeasible**.

Because refinement is expensive, you refine skeletons **one at a time until one
succeeds**. This trial sequence is a **rollout**. The **order** you try them in
dominates wall-clock: try a feasible skeleton early and you finish fast; keep picking
infeasible ones and you waste attempts.

**What SPECTRE does.** It learns a scoring function that orders the pool, and can
**re-order the remaining pool after each failure**. The goal is to reach a feasible
skeleton in as few attempts as possible.

---

## 2. SPECTRE as written (the method)

*(Full detail: `proposal.md` §3 and the audit `spectre_audits_07-14.md` Findings 1–2.)*

### 2.1 What SPECTRE looks at — the representation (deliberately "x₀-free")

SPECTRE **does not** look at the low-level scene (`x₀` — camera images, object poses,
sizes). It represents each skeleton by an **abstract token sequence**:

```
[ STATE_0 , OP_1 , OP_2 , … , OP_L , STATE_L ]
```

— the initial abstract state, the L operators of the plan, and the **final** abstract
state. Intermediate states are *not* encoded (an internal design choice called
"Substage A"). "Abstract state" here means only the symbolic facts (e.g. which items
are in the drawer), not geometry.

Object names are replaced with **typed local ids** ("canonicalization"): the concrete
name `o5` becomes `item_3`, so the model reasons about *roles*, not specific names.
Two facts about this matter later:

- The renumbering is done **per episode** (one relabeling shared across the whole
  candidate pool *and* the failed set), so the model *can* track "the same object
  appears in this failed plan and this candidate."
- At training time the ids are **randomly permuted** (augmentation), so the model
  cannot attach meaning to any specific id value.

### 2.2 The three learned modules (~185k parameters, width d = 64)

- **Φ (skeleton encoder).** A small transformer over the token sequence above, with a
  **Set-Transformer** pooling the symbolic facts inside each state token. Produces one
  embedding `e(s)` per skeleton.
- **Ψ (context encoder).** A permutation-invariant set encoder over the embeddings of
  the skeletons that **have already failed** in this episode. Produces a context vector
  `c_t`. For the empty history (no failures yet) it emits a learned vector `c₀`.
- **σ (scorer).** A small MLP over `[ e(s) ; c_t ; prior ]` → a scalar score (logit)
  for each remaining skeleton. The next skeleton tried is the highest-scoring one.

A plug-in **prior π** can warm-start σ; on DD2D the prior is **zero** (π ≡ 0).

### 2.3 Loss and training discipline (hard-won)

- **Listwise Plackett-Luce loss** — rewards putting *a feasible skeleton first*, which
  aligns with the rollout objective. (Pointwise binary cross-entropy was tried and
  killed; it is not rollout-aligned.)
- **F-subset rule.** Each training example conditions on a failed set F that contains
  **only genuine failures** (never a success), matching what happens at test time.

### 2.4 Static vs adaptive — one flag

A single switch (`freeze_context`) gives the two deployment modes compared later:

- **SPECTRE-static** — score the pool **once** at `c₀` and never re-rank. A fixed
  one-shot ranking; the strict apples-to-apples comparator to any one-shot model.
- **SPECTRE-adaptive** — re-encode `c_t` from the growing failed set and **re-rank the
  remaining pool after every failure**. This is SPECTRE's headline deployment mode.

### 2.5 Metric discipline

- **Model selection is rollout-based** (`val_rollout_attempts` — simulated
  attempts-to-first-success on the validation set). AUROC (a classification diagnostic)
  is *secondary*, never the selection criterion.
- **Evaluation is uncensored**: the attempt budget is set equal to the candidate-pool
  size, so the budget never binds and every reported number is a true
  attempts-to-first-success.

### 2.6 Baselines (SPECTRE is the *candidate*, not a baseline)

The project brackets SPECTRE against B1–B6: **B1** random order, **B2** default planner
order, **B3** static-historical (success-rate lookup), **B4** adaptive-historical
(failure-conditioned counts), **B5** oracle, **B6** DP-on-counts (lookahead over B4's
model). "Baseline" always means B1–B6 — never SPECTRE-specific code.

---

## 3. How SPECTRE is deployed on DD2D

*(Sources: [`decisions.md` 2026-07-12](decisions/03-dd2d-v2.2.md#2026-07-12-dd2d-integration-converter-not-native-env); audit Findings 3–4; [`notebook.md` 2026-07-12](notebook/03-dd2d-v2.2.md#2026-07-12-dd2d-wired-via-converter)/13;
combinatorics computed 2026-07-14.)*

### 3.1 The DD2D domain

**DD2D = Drawer Declutter 2D.** A target item sits in a drawer surrounded by other
items ("blockers"). The goal is `extracted(target)` — get the target out. To reach it
you may need to first move some blockers onto a "buffer." The abstract STRIPS model is
tiny:

- **1 object type** (`item` — the target is flagged by a `target` predicate, not by
  type), **6 predicates**, and **3 operators**: `pick`, `place-buffer`, `retrieve`.
- Removing one blocker is a forced two-action unit (`pick` then `place-buffer`), and
  `retrieve(target)` is always last. So **a plan that stages `m` blockers has length
  `2m + 1`.** (m = 0 is the plan `retrieve(target)` alone.)

### 3.2 Integration = a converter, not a native environment

DD2D shipped with its own dataset of pre-refined, pre-labeled candidate plans. Rather
than re-implement its geometry, the project **converts** each DD2D problem directory
(one JSON per candidate plan, all sharing the same objects/goal) into SPECTRE's
`EpisodeRecord` format (`spectre_convert.py`). Key consequences:

- **Abstract-only / x₀-free.** The converter keeps the 6 drawer predicates and
  **drops** the continuous geometry (`at-pose` literals). SPECTRE therefore sees none
  of the packing geometry — by design.
- **One variant, `dd2d_v2`**, spanning 10–13 items per problem.
- **669 problems** total (425 train / 120 val / 124 test), **pools of exactly 200
  candidate skeletons** each, with an overall skeleton success (feasibility) rate of
  **~0.11**.
- **Label caveat (important).** DD2D's Day-1 labeler marks not-yet-certified negatives
  as *marginal* rather than proven-infeasible. So all DD2D numbers here are
  **diagnostic, not yet publishable**, until an arrangement-complete negative
  certificate lands.

### 3.3 Why plan length carries so much structure (pool combinatorics)

Because every plan is an ordered choice of which blockers to stage, the pool is
organized by length. With `B` blockers (`B = items − 1`), the number of distinct plans
that stage exactly `m` blockers is the permutation count `P(B, m) = B·(B−1)···(B−m+1)`:

| items n | B | m=0 (len 1) | m=1 (len 3) | m=2 (len 5) | m=3 (len 7) |
|---|---|---|---|---|---|
| 10 | 9  | 1 | 9  | 72  | 504 |
| 13 | 12 | 1 | 12 | 132 | 1320 |

The 200-skeleton cap keeps **all** of the short plans (m ≤ 2) plus a **truncated** slice
of the m = 3 plans — no pool ever reaches m = 4. The upshot: on DD2D, "how many blockers
to stage" (≈ plan length) is a first-class, low-dimensional axis, and the residual
question is *which specific* blockers (object identity). This split — length vs identity
— is the theme of Section 5.

### 3.4 The four-method comparison

All four methods rank the **same 200-skeleton pool per problem**; they differ *only* in
the ordering function:

| Method | What it sees / how it ranks | Order type |
|---|---|---|
| **astar-dist** | planner enumeration order (`score = −plan_idx`) | fixed, non-learned |
| **PIGINet_v3** | a low-level predictor: CLIP image features + literals over the concrete scene | fixed, learned one-shot |
| **SPECTRE-static** | SPECTRE scored once at `c₀` | fixed, learned one-shot |
| **SPECTRE-adaptive** | SPECTRE re-ranked after every failure | adaptive |

**Metric: rollout false-positives (rollout-FP)** = number of infeasible skeletons tried
before the first feasible one (i.e. attempts-to-first-success − 1). Lower is better.

**Problems are stratified s0–s3** by the *minimum number of blockers that must be
removed* for feasibility (s0 = target already free, s3 = ≥3 required).

**Headline results (DD2D test, 124 problems, mean rollout-FP):**

| method | s0 | s1 | s2 | s3 | ALL |
|---|---|---|---|---|---|
| astar-dist | 0.0 | 1.8 | 16.1 | 122.8 | 33.0 |
| PIGINet_v3 | 4.3 | 12.3 | 17.8 | 49.3 | 20.4 |
| SPECTRE-adaptive | 2.6 | 5.9 | 30.7 | 41.4 | **19.2** |
| SPECTRE-static | 16.1 | 15.0 | 27.1 | 39.2 | 23.8 |

At face value SPECTRE-adaptive edges PIGINet on the mean (19.2 vs 20.4). Section 5
explains why that comparison is subtler than it looks.

---

## 4. Implementation audit (2026-07-14), condensed

The audit (`spectre_audits_07-14.md`) verified four things about the DD2D run, each
grounded in code and data:

1. **Canonicalization is per-episode**, with one object relabeling shared across the
   candidate pool and the failed set, augmentation ON, canonical order alphabetical.
   → Object identity *is* trackable within an episode (the precondition an
   identity-based mechanism would need), and an id-ordering-leakage hypothesis ("H6")
   is **ruled out** (augmentation randomizes ids; order is not geometry-correlated).
2. **The representation is `[STATE_0, OP…, STATE_L]`** — initial state, operators, final
   state; no intermediate states; no low-level geometry.
3. **The metric is rollout-FP, and its censoring/exclusion code paths are inert** on
   this run: all 124 pools are solvable (≥1 feasible skeleton) and the budget equals the
   pool size, so nothing is censored or dropped. Reported means are true uncensored
   values. (One latent, currently-harmless asymmetry is noted for future datasets.)
4. **The candidate pool is identical across all four methods** — they differ only in
   ordering/scoring, never in which skeletons exist.

---

## 5. What we have learned and ruled out

### 5.1 From the living docs (RT2D era + DD2D wiring)

- **RT2D headline (2026-04-27).** On the bespoke RoutedTransport2D environment, SPECTRE
  cut attempts-to-first-success by **41–62%** vs the baselines. (RT2D is a hand-built
  environment designed so that beating the adaptive baseline B4 requires relational
  reasoning.)
- **Ψ-ablation (2026-06-06).** Freezing the context (removing failure-conditioning)
  costs ~1 attempt. Conclusion: **failure-conditioning accounts for only ~27% of
  SPECTRE's margin; the static representation carries ~73%.** This is what motivated
  demoting adaptivity.
- **Fully-observable (FO) information ceiling (2026-06-25).** In fully-observable,
  deterministic TAMP, every skeleton's outcome is a deterministic function of the
  initial low-level state `x₀`, so *within-episode failures add no information beyond
  `x₀`.* This structurally **bounds** how much adaptivity can ever help — and it
  triggered the **direction pivot** from "adaptive re-ordering" to the "representation
  question."
- **B6 lookahead (2026-06-11).** Multi-step lookahead over the count model is small,
  fragile, and saturated — **not** the missing ingredient.
- **DD2D negative-control confirmed (2026-07-12/13).** On the first DD2D training runs,
  SPECTRE ≈ B2 (it barely beats trivial shortest-plan-first order), AUROC ≈ 0.55
  (barely above chance), and crucially **AUROC(3) < AUROC(0)** — i.e. conditioning on
  failures does *not* help. This is the expected fingerprint of a **too-lossy
  abstraction**: dropping poses/sizes removes exactly the packing signal DD2D
  feasibility depends on.
- **Process lessons.** Uncensored evaluation (budget = pool cap) is the reporting
  standard; the PL loss and F-subset rules are load-bearing; and a **seed-forwarding
  bug** meant some earlier "multi-seed" runs were duplicate checkpoints (no valid ≥3-seed
  zero-prior RT2D run exists yet).

### 5.2 From the July-2026 diagnostic battery (this session — dated, not yet in the living docs)

These sharpen *why* SPECTRE behaves as it does on DD2D. They are reproducible from
`experiments/spectre/compare_dd2d_methods.py` and the comparison cache.

**T0 — "Is SPECTRE just ranking by plan length?"** For each method, regress its
per-skeleton scores on plan length (η² = fraction of score variance explained by
length; "within-length" = the remainder, i.e. genuine same-length discrimination):

| method | η²(length) | within-length signal | uses length? |
|---|---|---|---|
| SPECTRE-static | **1.00** | ~0.00 | it *is* a length ranking |
| astar-dist | 0.75 | 0.25 | short-first |
| PIGINet_v3 | 0.21 | **0.79** | mostly *not* length; discriminates within length |

→ **SPECTRE-static's score is a pure function of plan length** (to ~4 significant
figures); it cannot tell two same-length plans apart. Its learned length curve is
*non-monotone* (it front-loads 1- and 3-blocker plans and buries 2-blocker plans).
PIGINet is the opposite: it genuinely discriminates *within* a length using geometry
(within-length AUROC ≈ **0.67**, vs SPECTRE-static's ≈ **0.49** = random).

**T1 — "Does the adaptive context use *which* objects failed, or only *how many*?"**
Rerun SPECTRE-adaptive but replace every failed skeleton in the context with a random
*same-length* skeleton (correct length, random identity). Result:

- Rollout-FP is **unchanged**: Δ = **+0.00** FP, 95% CI **[−0.02, +0.03]**, in every
  stratum. (The context embedding does shift by ~0.005, but never enough to change a
  selection.)

→ **The context module Ψ ignores failed-skeleton identity; it uses only size/length.**
Combined with T0, **SPECTRE on DD2D uses plan length and essentially nothing
identity-specific** — neither in the static ranker nor in the adaptive context.

**Why does a length-only SPECTRE tie PIGINet, which really does use geometry?** Three
findings resolve the apparent paradox:

1. **Length is DD2D's dominant feasibility axis — a hard floor.** Feasibility rate by
   (stratum, m = blockers staged):

   | stratum | m0 | m1 | m2 | m3 |
   |---|---|---|---|---|
   | s0 | 1.00 | 0.44 | 0.22 | 0.14 |
   | s1 | 0.00 | 0.11 | 0.11 | 0.16 |
   | s2 | 0.00 | 0.00 | 0.02 | 0.12 |
   | s3 | 0.00 | 0.00 | 0.00 | 0.04 |

   Staging *fewer* than the required number of blockers is feasible **0%** of the time.
   So plan length alone eliminates the large majority of candidates (on s3, *every*
   length-≤5 plan — all of m ≤ 2, roughly 80–145 of the 200 depending on item count — is
   guaranteed-infeasible). Getting the *length regime* right captures most of the
   achievable gain; *which* subset (identity/geometry) is a weaker, residual battle.
2. **PIGINet's geometry is real but weak and mis-deployed.** It discriminates
   within-length (AUROC 0.67) but is *flat in length* (it does not prioritize the right
   length band), so it wastes attempts on guaranteed-infeasible wrong-length plans.
   SPECTRE does the mirror image: nails the length regime, ignores the residual.
3. **The aggregate "tie" hides opposite per-stratum profiles.** Head-to-head
   (SPECTRE-adaptive vs PIGINet, per-problem wins):

   | stratum | SPECTRE mean | PIGINet mean | SPECTRE wins | PIGINet wins |
   |---|---|---|---|---|
   | s0 | 2.6 | 4.3 | 6 | 21 |
   | s1 | 5.9 | 12.3 | 16 | 21 |
   | s2 | 30.7 | 17.8 | 6 | 23 |
   | s3 | 41.4 | 49.3 | 15 | 14 |
   | ALL | 19.2 | 20.4 | **43** | **79** |

   **PIGINet wins more individual problems (79 vs 43).** SPECTRE's lower *mean* is
   carried by the high-FP **s3 tail**, where getting the length regime right (prefer
   m=3) pays off most. The mean is a poor summary here.

**Bottom line of 5.2.** "SPECTRE matches PIGINet on DD2D" is **not** evidence that an
abstract representation beats geometry. It reflects that (a) DD2D difficulty is largely
*count*-governed, which the length abstraction captures for free, making DD2D a
*softer* negative control than intended, and (b) PIGINet under-converts its geometry.
Neither method solves the within-length (identity) problem — SPECTRE ignores it,
PIGINet attempts it weakly.

---

## 6. Current status and open questions

- **Label certificate pending.** DD2D negatives are *marginal*; no label-dependent DD2D
  number is publishable until the arrangement-complete negative certificate lands.
- **No low-level / x₀-conditioned SPECTRE comparator yet.** The headline "representation
  crossover" prediction (abstraction wins in low-data/weak-perception; low-level wins
  with abundant data/strong perception) **cannot be tested** until a SPECTRE variant
  that also sees `x₀` is built. DD2D retains the poses/sizes needed for this.
- **DD2D is a softer negative control than intended.** Its difficulty is count-governed,
  so the length abstraction has real purchase; a cleaner separator would hold the
  required-blocker *count* fixed and vary *which* subset is correct.
- **PIGINet's geometry may be under-converted (H6/T8).** Whether PIGINet's flat-in-length
  ranking is a frozen-CLIP image-pathway failure is untested — an image-ablation (T8)
  would settle it.
- **≥3-seed rigor.** No valid multi-seed zero-prior RT2D run exists yet (seed-forwarding
  bug); DD2D runs use 3 seeds but inherit the label caveat.

---

## 7. Where to look next (living documents)

- `proposal.md` — the current method and framing (source of truth; §0 is the pivot).
- `decisions.md` — dated decision log (ADR style), newest first.
- `notebook.md` — dated run/EDA/ablation log.
- `spectre_audits_07-14.md` — the implementation audit with file:line citations.
- `experiments/spectre/compare_dd2d_methods.py` — the marimo notebook that produces the
  DD2D comparison and the T0/T1 analyses in Section 5.2.

---

## 8. Glossary

- **Skeleton** — a symbolic action sequence that reaches the goal on paper; the unit
  SPECTRE ranks.
- **Refinement** — turning a skeleton into a concrete collision-free motion plan; may
  fail geometrically. **Feasible** = refinement succeeds.
- **Pool** — the set of candidate skeletons for one problem (200 on DD2D).
- **Rollout** — trying skeletons one at a time until one is feasible.
- **Rollout-FP** — infeasible skeletons tried before the first feasible one
  (= attempts-to-first-success − 1); the DD2D metric. Lower is better.
- **Stratum (s0–s3)** — a DD2D problem's *minimum required blockers to remove* (s0 =
  none, s3 = ≥3).
- **m / blockers staged** — how many blockers a plan removes; plan length = `2m + 1`.
- **Φ / Ψ / σ** — SPECTRE's skeleton encoder / failed-set context encoder / scorer.
- **c₀, c_t** — the context vector for empty history / after t failures.
- **x₀** — the low-level scene (images, poses, sizes). SPECTRE is "x₀-free."
- **B1–B6** — the non-learned/heuristic baselines (Section 2.6). SPECTRE is the
  *candidate*, never a baseline.
- **PIGINet** — a low-level feasibility predictor (CLIP image features + literals over
  the concrete scene); the "geometry-using" comparator on DD2D.
- **RT2D / DD2D** — RoutedTransport2D (bespoke env, RT2D-era results) / Drawer
  Declutter 2D (the current geometric-packing negative-control env).
- **η² (T0)** — fraction of a method's score variance explained by plan length; η² ≈ 1
  means "the ranking is a length ranking."
