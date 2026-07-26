# SPECTRE v3 — Proposal (v3.0.0, 2026-07-26)

A standalone proposal for moving from **SPECTRE v2.2** (documented in
[`as_built_v2.2.md`](as_built_v2.2.md)) to **SPECTRE v3**. It is written to the same
conventions as the as-built doc and is intended to become the v3 design-of-record: the
migration plan (§7) is a sequence of **gated increments**, each with an acceptance
criterion and a pre-registered prediction, and the v2.2 code is **preserved untouched**
throughout (§7.1).

**The three v3 goals.**
1. **Performance** — retain v2.2's per-stratum results and fix the weak stratum:
   target **weak per-stratum dominance** over deployed v2.2 on dd2d_v3, with **s2 ≤
   astar-dist (17.08)** as the headline improvement. No stratum may regress beyond seed
   noise.
2. **Cleanliness / story** — reduce the system to **three ideas** (representation,
   objective + learned conditioning, failures-as-observations), remove vestigial
   components, and organize all ablations along one decomposition ladder.
3. **Generality** — a 2D/3D-shared geometry interface and a domain-independent adaptive
   pathway, so that a second environment (target: Khodeir-style 3D sorting on
   drake-tamp) needs only: a converter, refiner instrumentation, and a per-query-type
   axiom declaration (~1 bit each). No per-environment predicates, features, or fact
   vocabularies.

**Epistemic conventions** (carried from v2.2 and used throughout):
- **[established]** — measured and gated in v2.2 docs; **[diagnosed]** — a mechanism
  identified on 1-seed or on dd2d_v2, not yet re-verified; **[hypothesis]** — proposed
  here, untested, with a pre-registered prediction.
- **Seed discipline:** [1-seed dev] to iterate, **≥3 checksum-distinct seeds** to
  report. No [1-seed dev] number appears in a writeup.
- The dd2d_v3 reference numbers quoted below (v2.2-adaptive **13.68** overall; s0 0.00 /
  s1 4.60 / **s2 26.20** / s3 23.92; astar-dist s2 17.08; PIGINet 18.67) are currently
  **[1-seed dev]** — 3-seeding them is Gate 0 and precedes everything else.

---

## 1. Orientation — what v3 is

Same job as v2.2: a **learned listwise re-ranker** for bilevel TAMP that, given a pool
of candidate skeletons, orders them so refinement tries a feasible one first, and
re-ranks as refinement failures are observed. Metrics unchanged: rollout **FP** (failed
attempts before first success) and attempts-to-first-success, uncensored.

What changes is the *shape* of the system. v2.2 accreted as a sequence of patches — a
data-dependent prior knob, two overlap features, five bespoke fact types, a packing
certificate, a global feature vector with zeroed slots, two demotion sources. v3
re-derives the survivors from three principles:

1. **Representation** — a relational, tag-joined, object-centric geometric encoding,
   generalized to a single 2D/3D interface (§4).
2. **Objective + learned conditioning** — listwise Plackett–Luce (global +
   within-length), with the manual prior knob replaced by **necessity-conditioned
   scoring**: a learned, a-priori-legal, per-object difficulty estimate that gates the
   length preference per episode (§5).
3. **Failures as observations** — one canonical, domain-independent **FailureRecord**
   emitted by refiner instrumentation, consumed in two tiers: sound demotion via
   declared axioms (outside the net), learned hint tokens (inside) (§6).

Everything not derivable from these three is removed (§3) or demoted to harness.

---

## 2. Inherited law — constraints and learnings carried from v1/v2.2

These are the v2.2 §5 constraints plus the lessons of the v1→v2.2 arc, restated as
binding rules for every v3 design choice. Full provenance:
`as_built_v2.2.md` §5/§7, `decisions.md`, `consolidation_2026-07-19.md`.

### 2.1 The constraints

- **C1 — Generalizability.** No hand-crafted per-environment predicate or feature may
  be *load-bearing*. v3 strengthens this: the **only** domain inputs the method accepts
  are (a) the domain's own operator/stream schemas (the same specification every TAMP
  method consumes) and (b) a **declarative axiom registry** (§6.4) stating, per query
  type, whether observed failures generalize soundly. With an empty registry the system
  must still work ("learning is the floor").
- **C2 — Realism / a-priori-ness.** Only two legitimate *input* sources: **a-priori**
  (plan length, enumeration order, ground-truth initial geometry, goal literals) and
  **observed** (what the refiner reports while attempting plans, recorded — never
  re-derived by our own domain computation). Feasibility labels / oracle / stratum are
  **the answer**: off-limits as inputs or test-time gates. Labels remain legitimate as
  *training supervision* (that is what labels are for — `success_mask` and the necessity
  targets both live here).
- **C3 — Per-dataset validation of optional components** — and, stronger for v3:
  **prefer eliminating manual knobs by learning them.** The prior on/off switch (which
  helped dd2d_v2 and *diverged training* on dd2d_v3) is the cautionary tale; necessity
  conditioning (§5) is its learned replacement. Checkpoint selection is itself a
  component to validate (relrank miscalibrated on dd2d_v3 → deployed-val-FP selection).
- **C4 — Reconstruct, don't regenerate.** Any post-hoc geometry is a pure function of
  stored poses, never re-derived from the seed. (In v3 this constraint mostly retires:
  the adaptive pathway records live refiner observations instead of reconstructing
  facts — but it still governs dataset tooling.)
- **C5 — The net never sees proofs; deductions act on the ranking, not the
  representation.** Sound consequences are enforced outside the network
  (finite-offset demotion, never pool removal — completeness invariant P-E). A wrong
  network weight can never override a proof; a wrong proof can only reorder.
- **C6 — Listwise PL only** (global + within-length buckets); no pointwise BCE on the
  ranker (the necessity head's small masked BCE is auxiliary supervision, not ranking).
- **C7 — Seed discipline** (above), and **diagnostic-first ordering**: cheapest probe
  before implementation; pre-register predictions; every build stage a publishable
  fallback.

### 2.2 The learnings

- **L1 — Length collapse is a representation failure** [established]. v1's ranking was
  a pure length function (η² = 1.00) because anonymous ids + no geometry made
  same-length skeletons staging different subsets identical inputs. Episode-local tags
  + object geometry fixed it (η² → 0.23). Resampling/weighting alone cannot fix length
  bias when the model cannot read blockedness from geometry.
- **L2 — Hand-coded predicates are a trap** [established]. The `clears` predicate
  "unlocked" performance and was rejected: performance must not depend on finding a
  bespoke geometric routine per environment. (Its a-priori heuristic value — 7.4 FP —
  stays out of the method.)
- **L3 — The prior is data-dependent and can diverge training** [established]. The
  short-first prior was load-bearing on dd2d_v2 and *caused* the s3 collapse + training
  divergence on the easier dd2d_v3. A component that helps one distribution can hurt
  another; a manual switch is the wrong mechanism (→ §5).
- **L4 — Un-split evidence harms** [established]. Consumed crudely, `blocked-at-…`
  became a "prefer longer" cue (+13.5 FP on s1). The proof/hint split — proofs
  structural, hints learned — made evidence help at every stratum.
- **L5 — Adaptivity's ceiling** [established, analysis]. In deterministic
  fully-observable TAMP, I(R; F | x₀) = 0: the failure stream adds no *information*
  beyond x₀. Adaptivity is an **efficiency claim** — a cheaper route to what x₀
  contains — and its gain is parasitic on static-predictor weakness. Paper wording must
  respect this.
- **L6 — Proof-demotion's leverage is axiom-shaped, not universal** [established,
  analysis; this chat]. The subset rule is sound in DD2D because of (A) **obstruction
  monotonicity** (universal in collision-based feasibility) and (B) **locality**
  (staged objects leave the query's region — a DD2D world-layout property). (B) fails
  in, e.g., same-surface declutter. Hence the axiom registry: the *mechanism* is
  domain-independent, the *axioms* are domain-declared, and the system degrades to
  hints where they are not declared. Related work: this is conflict generalization
  (Ortiz-Haro et al. 2022; Khodeir et al. feedback) with a declarative soundness
  interface and a learned fallback — position it there.
- **L7 — The decomposition is the story** [established]. LAZY 71.1 → static 45.6 →
  +evidence 39.5 (dd2d_v2): the representation does the bulk; typed evidence is a
  composable increment; proofs a structural bonus. The paper survives a reviewer
  discounting any single rung.
- **L8 — s2 is a needle-hunt with a suspected cross-length cause** [diagnosed,
  dd2d_v2]. ~2% of size-2 plans feasible; the diagnosed failure mode is *over-removal
  that fails packing* (residual cross-length bias). Not yet re-verified on dd2d_v3 —
  Gate D2 decides the s2 strategy.
- **L9 — Benchmark integrity.** DD2D *requires* concave footprints so cheap heuristics
  fail; do not "clean up" shapes. Always size-control when claiming a representation
  beats cheap statistics.

---

## 3. What v3 removes (vestigial components)

Removals are performed **subtractively and gated** (§7.3): one at a time, per-stratum
checked. A removal that breaks dominance is reinstated — and recorded as a finding.

| # | Component | Status in v2.2 | v3 disposition | Rationale |
|---|---|---|---|---|
| R1 | **Default-order / short-first prior** (`cand_prior`, `prior_gate`) | Data-dependent knob; already **off** in deployed dd2d_v3 | **Remove.** | L3. Subsumed by necessity conditioning (§5), which makes the length preference scene-conditional and learned. |
| R2 | **Computed demotion source** (`demotion_source="computed"`, `blocked-at-contents` geometry reconstruction, `spectre_geometry` proof path) | Opt-in; measured ~14% increment [1-seed] | **Remove from method and paper** (user decision: vestigial). Code stays in v2 modules, unported. | Observed demotion is essentially sound (1/6376) and hard-coding-free; the computed path is the last per-env geometry routine in the deployment story. |
| R3 | **`pack-impossible` fact + packing certificate in the method** | Inert: certificate off in runner, 0 proofs at λ=0.8, demotion path never triggered | **Remove from the method.** Certificate survives as *dataset tooling* only (labeling/audits). | A documented-but-dead mechanism is a reviewer liability. |
| R4 | **Analytic `grasp-witness`** (`grasp_witness_after_removing` via env `has_grasp`) | Hint-tier, rare at λ=0.8; a per-env geometry routine feeding the learned pathway | **Remove**; replaced by *observed* culprit instrumentation (§6.2). | C1/C2: record what the refiner's own collision checks touched; don't compute it ourselves. |
| R5 | **Five bespoke fact types + `FactEncoder` type vocabulary** | `blocked-at-contents / extraction-failed / grasp-witness / pack-exhausted / pack-impossible` | **Replace** with one `FailureRecord` schema (§6.1); the old types become lossy projections of it. | One record, one embedding, one soundness interface. |
| R6 | **Global token / `glob_feats`** | 3 of 6 slots hardcoded 0; buffer geometry reaches scorer only via container token anyway | **Remove**; containers become first-class scene tokens (§4.4). | Deletes an admitted incompleteness instead of patching it. |
| R7 | **`cand_overlap` = [dead, jaccard]** | Bundled into P5's +6.22; never isolated | **Candidate removal, gated** (Gate S4). `dead` duplicates outside-demotion; `jaccard` is a hand-built similarity the fact-token attention should learn via the tag join. | If the evidence increment survives, the net's boundary is maximally clean (C5). If not, `jaccard` returns as one honest feature — itself a finding. |
| R8 | **`relrank` checkpoint selection** | Miscalibrated on dd2d_v3 (never < 1) | **Replace** with deployed-val-FP selection wired into `train_v3` (currently offline-only). | C3; selector noise otherwise contaminates every v3 comparison. |
| R9 | **`exclude_marginal` (inert)** | Marginal fails silently labeled `False` | Fix or delete the flag; document the label-hygiene choice either way. | Small bias; known incompleteness. |

Not removed (survivors, with their evidence): episode-local tags [P-A], geometry +
SAB scene encoder [P1], listwise PL + within-length [L1/L4 arc], observed
proof-demotion outside the net [P4: +11.08 FP], hint-evidence pathway [P5: +6.22, as a
bundle], aux head — **promoted**, not vestigial (§5).

---

## 4. Geometry substrate — one 2D/3D interface

The v2.2 encoders are already point-set networks; v3 makes the interface
dimension-generic so **one architecture trains on 2D and 3D data**, with 2D as an
embedded special case. Every item below carries a **DD2D regression gate**: the
refactored encoder must reproduce deployed-v2.2 per-stratum numbers within seed noise
before any 3D data exists.

### 4.1 Object footprints → surface point sets in ℝ³

- **Interface:** an object is a **sampled surface point set in the item frame**, in ℝ³.
  2D objects embed as (x, y, 0) (the current 32-point boundary ring, zero-padded).
- **Encoder:** unchanged in form — shared per-point MLP (input width 3) → PMA → 32-d
  descriptor. Order- and start-vertex-invariant; a point *set*, not a rasterization —
  the concave-safe argument is preserved and worth keeping for the paper.
- **3D budget:** expect 128–256 surface samples per object (vs 32); benchmark tensor
  memory at pool sizes before Phase 4.
- **Rejected alternatives:** voxel/SDF grids (resolution-bound, heavy, kills the
  exactness story); image crops as primary substrate (reintroduces the perception
  confound the P2 parity 2×2 exists to isolate — crops remain a *parity arm*, not the
  method).

### 4.2 Pose → shared SE(2) ⊂ SE(3) encoding

Position (x, y, z; z = 0 in 2D), normalized by workspace scale, + a continuous
rotation representation (6D rotation-matrix parameterization; degenerates cleanly to
sinθ/cosθ in-plane). Scalar shape stats generalize directly: area→volume,
concavity→`1 − vol(shape)/vol(convex hull)`.

### 4.3 Goal-anchored relations replace `is_target` / `obj_rel`

The v2.2 relation-to-target block is C2-legal (raw a-priori relative geometry, no
domain computation) but **schema-brittle**: it hard-wires a single distinguished goal
object. Per-environment feature switching is exactly the C1 anti-pattern, so v3 uses
one definition that *reduces to* the current one:

- Per-object **`goal_role`** embedding derived from the goal literals (object mentioned
  in goal / not) — computable in any PDDL problem; replaces `obj_is_target`.
- The relative-geometry scalar block is computed **per (object, goal-object) pair**
  with masked aggregation over goal objects. DD2D has exactly one goal object, so this
  reduces to the current 8-d `obj_rel` — the regression gate can verify behavioral
  equivalence.

### 4.4 Containers as first-class scene tokens

Buffer / drawer / bins / shelf regions each become a scene token: own surface point
set + a `container` role flag, entering the same SAB relational join as objects. This
replaces the zeroed `glob_feats` buffer slots (R6) and generalizes to arbitrary
regions in 3D with no new mechanism.

### 4.5 Candidate encoder: extrapolating positions; per-step queries only if indicted

- **Keep** the single 64-d PMA-pooled candidate vector by default. Capacity is not the
  concern (subset identity over M ≤ 14 objects ≪ 64-d; the anti-collapse test
  [established] proves distinguishability; within-length AUROC 0.585–0.673
  [established] proves usable signal; order information enters via per-step position
  embeddings *before* pooling).
- **Required change:** replace the learned absolute `pos_emb` table with a
  **length-extrapolating encoding** (sinusoidal or relative). An absolute table is OOV
  beyond the training max length and silently breaks the train-s0–s2 / deploy-s3
  experiment (§8).
- **Conditional change (Gate D5):** if the Phase-0 probe (within-length ranking
  quality as a function of plan length L) shows degradation in L, switch the scorer to
  **per-step queries** (candidate = set of step tokens; each step cross-attends over
  the scene; pool attended step features at the end). Expected necessary for
  longer-horizon 3D sorting, likely unnecessary for DD2D — decide on the probe, not on
  taste.

---

## 5. Necessity-conditioned scoring (the headline revision) [hypothesis]

**Promotion, not addition:** v2.2's aux head (per-object *necessary/relevant* logits,
weight-0.2 BCE, training-only) becomes a deployment-consumed **necessity head**.

### 5.1 Mechanism

- Per object i, the head predicts pᵢ = σ(logitᵢ) ≈ P(object i must be manipulated),
  trained on the existing `necessary` labels (derived from pool feasibility labels —
  C2-legal as *training supervision*; verify the exact label definition in
  `dataset_v2` during Phase 0).
- **Difficulty estimate** d̂ = Σᵢ pᵢ — expected number of required manipulations, an
  **aggregate of per-object predictions, not a categorical length classifier**.
  Because it is a sum over objects it extrapolates to scenes harder than any training
  episode; a length classifier structurally cannot ("complication prediction" framing;
  length is a derived statistic of obstruction domains).
- **Candidate features** for candidate c with staged/manipulated set S(c), computed
  from **detached** pᵢ (the head trains only on its own BCE; the ranker gradient must
  not warp it):
  - `mismatch = | |S(c)| − d̂ |`
  - `coverage = Σ_{i∈S(c)} pᵢ`
  - `waste    = Σ_{i∈S(c)} (1 − pᵢ)`
  These enter the scorer head where `cand_prior` used to sit.
- **Retires the prior (R1):** the short-first/long-first decision is made per episode
  by predicted difficulty, not per dataset by hand — the learned replacement C3 asks
  for, and the mechanism that removes L3's failure mode.

### 5.2 Toy sanity check

Three blockers A, B, C; ground truth: {A, B} minimal feasible; any set containing the
oversized C fails packing. A decent head: p ≈ (0.90, 0.85, 0.10), d̂ ≈ 1.85.
Candidate {A}: mismatch 0.85 — penalized. {A,B}: mismatch 0.15, waste 0 — best.
{A,B,C}: waste 0.9 — penalized. This is exactly the diagnosed s2 failure (over-removal
into pack failures) being priced correctly; on an s3 scene with three high-necessity
objects d̂ ≈ 3 flips the preference automatically; a 5-blocker scene yields d̂ ≈ 5
even if training never saw length 5.

### 5.3 §5-compliance defense (write this into the paper)

Necessity features look like a cousin of the rejected `clears`; the distinction is
sharp and must be stated: `clears` was a **hand-coded per-environment geometric
computation supplied as an input**; necessity is **learned from domain-agnostic
supervision** (which objects appear in feasible skeletons — derivable from any pool +
labels in any TAMP domain), with **zero environment geometry routines at inference**.
d̂ is *predicted*, never given — stratum remains off-limits (C2).

### 5.4 Known risks (log now)

- Necessity labels inherit pool-truncation noise ("every feasible skeleton" is really
  "every *observed* feasible skeleton").
- In domains where nearly all objects are goal-referenced (block stacking), `necessary`
  saturates and d̂ degenerates toward object count — the features become uninformative
  rather than harmful, and the unconditioned path stays as an ablation. Do not oversell
  universality.
- If the head is inaccurate, conditioning can hurt: keep the ungated model as the
  fallback arm in every Phase-2 comparison.

---

## 6. The adaptive component, cleaned: one canonical FailureRecord

First-principles question: *when a refinement attempt fails, what is legally
observable with no domain computation?* A refinement grounds steps of σ = (a₁ … a_L)
until step j fails on some continuous query. Standard PDDLStream-style refiner
instrumentation — recording computations that already ran (C2's "things we would see
anyway") — yields:

### 6.1 The record

```
FailureRecord(σ):
  j            — failing step index (depth reached)
  q, args      — failing query schema + object arguments
                 (in PDDLStream: the failed stream instance — reported generically)
  s_j          — abstract state at step j (symbolic simulation of the prefix;
                 pure STRIPS, no geometry)
  U(σ, j)      — objects unmoved by the prefix before j (derivable from s_j)
  n, exhausted — sampler effort + exhausted-vs-timeout flag
  culprits     — objects implicated in the failed samples' collision checks
```

`culprits` legality note: every collision checker computes *pairs*; recording which
objects the failed samples collided with is instrumentation of an existing
computation — the same legality class as `failure_action` — **not** an analytic
predicate like the removed `grasp_witness_after_removing` (R4). Whether the current
refiner *exposes* it is an engineering task (Phase 2), not a design question.

### 6.2 The old fact types are projections of this record

Information accounting per fact type: **projection** = a fixed function g(R) of the
record, no information added; **approximation** = per-record weaker; **outside** = not
expressible as an observation of a computation that ran.

| v2.2 fact type | Relation to FailureRecord | Accounting |
|---|---|---|
| `blocked-at-contents` (**observed** mode) | **Exact projection**: j = retrieve step ⟹ prefix completed ⟹ q = grasp(target), U = all ∖ staged set | Demotion needs exactly (q, args, U); Gate A1's reduction test verifies candidate-for-candidate equivalence. Zero loss vs deployed v2.2. |
| `blocked-at-contents` (**computed** mode) | **Outside — by design.** The predicate evaluated blockedness *counterfactually* on plans that died at extraction; no query ran ⟹ no observation ⟹ no record | The measured ~14% increment [established, 1-seed]. Not a schema failure: C2 (observed-only) *defines* it out, and R2 cuts it per project decision. |
| `extraction-failed` | **Projection, strictly enriched**: q = grasp(blocker), args = that blocker, + depth j, effort n | Resolves the old "does it carry object args?" uncertainty by construction. (Enrichment is deployment/instrumented-collection only — see §6.6.) |
| `pack-exhausted` | **Projection, strictly enriched**: q = place(·, container), exhausted = True, + placed object, depth, culprits | Same caveat as above. |
| `grasp-witness` | **Approximation** via the observed `culprits` field | Per-record weaker than the analytic version (sampled grasps vs exhaustive `has_grasp`), but: it was hint-tier, so exactness was never structurally exploited — the learned consumer is the right consumer for noisy evidence; it was rare at λ=0.8 [established], so likely not load-bearing (D3's leave-one-out gates this); and culprits are emitted on *every* collision-mediated failure, so corpus-wide information plausibly rises even as per-record precision falls [hypothesis — A2's bar tests it]. **Never promoted to proof tier**: "all sampled grasps hit x + exhausted" is not an axiom we can declare. |
| `pack-impossible` | **Outside** — an analytic certificate about a counterfactual, structurally inexpressible as an observation | Loses nothing today (0 proofs at λ=0.8, demotion path never triggered [established]). If a tighter-λ regime ever made packing proofs valuable, they would re-enter as a separate opt-in proof source — which this paper has chosen not to have (R3). |

**Framing for the doc and the paper:** relative to *deployed* v2.2 (observed demotion
+ hint tokens), records ⊇ facts with strict enrichment. The only two losses —
computed-mode `blocked-at-contents` and `pack-impossible` — are **chosen removals
(R2, R3), not casualties of the unification**. State it that way, so no reader
concludes the record schema is silently lossy.

### 6.3 Hint-tier consumption (inside the net)

One record → one token, with **role-separated tag slots** (required, not optional):

```
record token = Linear([ query-schema emb ; pooled arg-tags ; pooled culprit-tags ; scalars (j/L, n, exhausted) ])
                                              ▲ distinct projections per role ▲
```

- **Why role separation is load-bearing:** in v2.2, `grasp-witness` was a *separate
  token* from the failure that spawned it, so "argument of the failed query" and
  "object implicated as a blocker" were distinguished *by fact type*. A naive encoding
  that mean-pools arg-tags and culprit-tags into one slot destroys that role
  distinction — the net would see "objects associated with this failure" without
  knowing the picked object from the objects blocking it. With separated slots, every
  old fact type is recoverable as a function of record fields: `extraction-failed` ≡
  (q = grasp, arg-role = blocker); `pack-exhausted` ≡ (q = place, exhausted);
  witness content ≡ the culprit slot.
- The **query-schema vocabulary** is inherited from the domain/stream file — not
  hand-designed — replacing `FactEncoder`'s bespoke type vocabulary; the scalars slot
  finally consumes what the v2.2 tensorizer drops. The v2.2 **tier embedding is
  dropped without loss**: only hint-tier facts ever entered the net, so it was a
  constant.
- Encoding rules unchanged: categorical ids are embedding lookups (never raw
  continuous scalars); genuinely ordinal quantities (depth, effort counts) are
  normalized scalars.

### 6.4 Proof-tier consumption (outside the net): the certificate rule + axiom registry

Generalizes v2.2's subset demotion via the L6 decomposition:

- **Registry:** per query type, the domain declares (or doesn't) two axioms:
  **monotone** (failure against occupancy O ⟹ failure against any O′ ⊇ O — universal
  physics in collision-based feasibility) and **local** (the staged/moved objects leave
  the query's relevant region). Declaring an axiom has the same epistemic status as
  writing the PDDL domain file: it is *specification*, not a per-env inference routine.
- **Rule:** demote candidate σ′ iff it issues the **same query on the same args** at
  some step j′ with **U(σ′, j′) ⊇ U(σ, j)**, and the registry declares that query type
  monotone + local. Demotion remains a finite offset on the ranking — never pool
  removal (C5 / P-E).
- **Reduction test (Gate A1):** on DD2D this reduces exactly to the v2.2 subset rule
  (U = all ∖ staged ⟹ U′ ⊇ U ⟺ staged′ ⊆ staged). The new code path must reproduce
  v2.2's demotion decisions **candidate-for-candidate** before it may change anything
  else.
- **Degradation:** empty registry ⟹ records flow only through hint tokens; "learning
  is the floor" becomes a *measured* claim on env-2 (§8, prediction P-v3-4).
- **Paper wording (respecting L5/L6):** *where a domain admits monotonicity +
  locality, one observed failure soundly prunes exponentially many candidates; where
  it doesn't, evidence degrades to learned hints and the ranker still works.* Never
  claim "the demotion rule generalizes" unconditionally.

### 6.5 Phase-0 autopsy

The v2.2 "effectively sound (1/6376)" observed-demotion edge case is empirical
evidence that some assumption leaks even in DD2D. Find that one case (≈1 hour); it
identifies precisely which axiom needs a guard. [Cause currently unknown — do not
speculate in writing until inspected.]

### 6.6 Data availability: backfill vs. instrumentation

"Records replace facts" holds at the schema level; data availability adds one
sequencing constraint. **Deployment-time** records are free — the refiner runs live
and instrumentation observes everything. **Training-time** records for the existing
dd2d_v3 collection can only be **backfilled** from stored metadata (`failure_action` +
whatever args it carries); re-refining to obtain richer records is forbidden (C4).
Consequences:

1. **Parity is testable now:** backfilled records on dd2d_v3 are exact images of the
   stored v2.2 facts, so Gates A1/A2a certify no-regression without any new
   collection.
2. **Enrichment (args everywhere, culprits, depth) requires an instrumented
   collection** — and a model trained on degraded backfilled records has never seen
   the rich fields populated, so **do not deploy rich records against a
   backfill-trained model**. The train-time and deploy-time record distributions must
   match per checkpoint.
3. Cleanest sequencing (reflected in §7.4): **A2a** = backfill parity on dd2d_v3;
   **A2b** = full instrumented records, folded into the DAgger/dd2d_v4 round on DD2D
   *or* simply native on env-2 (where instrumentation exists from day one), making
   DD2D enrichment optional rather than blocking.

---

## 7. Migration plan — gated increments, v2.2 preserved

### 7.1 Code preservation (hard requirement)

Follow the pattern that preserved v1 through the v2 build: **v3 lands as parallel
modules, v2.2 is never edited in place.**

- New: `model_v3.py`, `dataset_v3.py`, `failure_record.py` (schema + refiner
  instrumentation adapters), `axiom_registry.py`, `train_v3.py` (with deployed-val-FP
  selection built in, R8), `proof_demotion_v3.py`. Selected by flags, exactly as
  `model_v2` was.
- v1 and v2 modules (`model.py`, `model_v2.py`, `dataset_v2.py`, `proof_demotion.py`,
  `spectre_geometry.py` — including the retired computed-demotion path, R2) remain on
  disk, untouched, loadable, so every v2.2 number stays reproducible.
- Checkpoints in fresh dirs (`checkpoints_v3*/dd2d_v3/...`); the `--env-variant` /
  checkpoint-map machinery extends, not mutates.
- Shared primitives (SAB/PMA, tags, PL losses) are imported, not copied — they are
  survivors, not v2-specific.

### 7.2 Phase 0 — hygiene + diagnostics (days; gates everything)

| Gate | What | Acceptance / decision |
|---|---|---|
| **G0** | **3-seed the deployed v2.2 dd2d_v3 numbers** (and PIGINet/astar rows as needed) | The dominance reference exists. No v3 comparison before this. |
| **D1** | Wire **deployed-val-FP selection** into training (R8) | Selector noise eliminated before any v3 run. |
| **D2** | **s2 loss decomposition** on existing checkpoints: per s2 rollout, classify failed attempts cross-length vs within-length; histogram failure modes (pack / extraction / still-blocked) from `refiner_metadata`; compute the **oracle-length bound** (model restricted to correct-length candidates — stratum as a *measured diagnostic bound*, never a method input) | **The fork:** oracle-length ≲ 17.08 ⟹ s2 is cross-length calibration ⟹ §5 is the right fix. Otherwise the bottleneck is within-length discrimination and §5 alone won't save s2 — say so before building. |
| **D3** | **Fact battery** on v2.2: per-type emission histograms; leave-one-type-out on hint tokens; split scramble gauge into type-scramble vs args-scramble; verify `extraction-failed` args; wire `Fact.scalars` into the v2 tensorizer (one line) and measure | Which evidence actually carries the +6.22; informs §6 token design. |
| **D4** | Verify the aux `necessary/relevant` **label definition** in `dataset_v2`; audit for any deployment-time label path (expect none) | §5 stands on verified ground. |
| **D5** | **Bottleneck probe:** within-length ranking quality vs plan length L; `pos_emb` OOV check | Flat in L ⟹ keep single-vector candidates (DD2D); degrading ⟹ per-step queries (§4.5). |
| **D6** | **1/6376 autopsy** (§6.5) | Names the axiom guard. |

### 7.3 Phase 1 — subtractive (v2.2 → "v2.2-lean"), one removal per gate

Order of expected safety; acceptance rule for every step: **no stratum degrades beyond
seed noise on dd2d_v3** [1-seed dev per step; the surviving lean config gets 3 seeds].
A removal that fails is reinstated and recorded as a finding.

S1 global token (R6, container tokens already carry the geometry) → S2 pack-impossible
path + certificate-out-of-method (R3) → S3 confirm prior removal end-to-end (R1;
already off in the deployed config) → **S4 `cand_overlap`** (R7 — the genuinely
uncertain one; P5 measured the bundle, so this needs its own gate). The subtractive
study is lab-notebook material; the paper carries only the resulting ladder.

### 7.4 Phase 2 — additive on DD2D

| Gate | What | Acceptance / prediction |
|---|---|---|
| **A1** | FailureRecord schema + certificate rule + registry (§6.1, §6.4), records **backfilled** from stored dd2d_v3 metadata (§6.6) | **Reduction test:** reproduces v2.2 demotion decisions candidate-for-candidate on DD2D before anything else changes. Then: per-stratum parity with v2.2-lean. |
| **A2a** | Role-separated record tokens (§6.3) replace FactEncoder, trained on **backfilled** records (parity-class information) | Evidence increment ≥ the D3-measured v2.2 increment; args-scramble gauge > 0. Certifies no-regression with no new collection. |
| **A2b** | **Instrumented records** (args everywhere, culprits, depth) — refiner instrumentation + a fresh collection: folded into the DAgger/dd2d_v4 round, *or* deferred to env-2 where instrumentation is native (§6.6) | Rich-record model ≥ A2a model per stratum. Train/deploy record distributions must match per checkpoint (never rich-at-deploy on a backfill-trained model). Optional for DD2D, mandatory for env-2. |
| **A3** | **Necessity conditioning** (§5) | **Pre-registered (P-v3-1):** s2 ≤ astar-dist 17.08 with s0/s1/s3 within noise — *conditional on D2 having said "cross-length"*. If D2 said cross-length and A3 fails, the mechanism is falsified; write that down now. |
| **A4** | **Length generalization:** train s0–s2, deploy s3 (needs §4.5 pos fix) | **Pre-registered (P-v3-2):** beats the same-protocol v2.2 model; d̂ correlates with true stratum while never receiving it. |

### 7.5 Phase 3 — geometry interface (still DD2D-only)

Point-sets-in-ℝ³ (2D at z = 0), SE(3)-shared pose, goal roles + goal-anchored
relations, containers-as-tokens (§4.1–4.4). **Gate G-geo:** reproduces the Phase-2
model's per-stratum numbers within seed noise. No new learning ideas in this phase —
interface work only.

### 7.6 Phase 4 — second environment (3D sorting, drake-tamp)

Setup budget (this *is* the generality claim — keep it small and report it): a
converter (meshes → surface point sets; bins → container tokens; goal literals → goal
roles), refiner instrumentation for FailureRecords, and axiom declarations (~1 bit per
query type; where locality is doubtful, don't declare it).
**Pre-registered (P-v3-4):** with an **empty registry**, the v3 ranker beats its env-2
baselines; declared axioms are a measured increment on top (as on DD2D: P4 +11.08). If
the increment turns out to be where all env-2 performance lives, that falsifies
"learning is the floor" — better said in advance. Note for framing: sorting's
feasibility bottleneck (reachability/collision) is a *different* dominant statistic
than DD2D's joint packing — exactly what a generalist-representation claim wants.

### 7.7 Deferred-but-promised (unchanged from v2.2)

P2 information-parity 2×2 (v3-with-crops / PIGINet-with-polygons — the fairness
answer), P3 shape-family shift, DAgger re-collection round.

---

## 8. Paper story & ablation plan

**Method = the three ideas of §1**, one subsection + one mechanism figure each. The
main experimental figure is the **decomposition ladder**, per stratum:

```
LAZY (untyped adaptive) → static representation → + hint records → + axiom demotion → + necessity conditioning
        71.1                     45.6                  39.5             (P4-class)            (v3, A3)
```

(dd2d_v2 reference values shown; the paper reports the dd2d_v3 3-seed ladder.) Every
kept component sits on a rung; component-level ablations (aux increment, goal-anchored
relations vs none, per-query-type LOTO, S4's overlap finding) go to an **appendix
table** — findable, not narrated. Baselines: astar-dist, PIGINet (retrained,
BCE/AUPRC), VLMPlan (zero-shot data-axis endpoint), v1, hand-rule. Headline claims:
per-stratum weak dominance on DD2D; length generalization (A4); env-2 transfer with
declared setup budget (P-v3-4); the parity 2×2 as the fairness answer.

**Defense-risk register** (preempt in the text): necessity-vs-`clears` distinction
(§5.3); necessity label noise & stacking saturation (§5.4); adaptivity-ceiling wording
(L5 — efficiency, not information); demotion generality wording (L6/§6.4 — axioms
declared, not assumed); polygon-privilege fairness (parity 2×2); size-control on any
"beats cheap statistics" claim (L9).

---

## 9. Consolidated pre-registered predictions

| ID | Prediction | Falsified if |
|---|---|---|
| P-v3-1 | With necessity conditioning, dd2d_v3 **s2 ≤ 17.08** (astar-dist), s0/s1/s3 within seed noise of v2.2 | s2 stays > astar-dist despite D2 = "cross-length" |
| P-v3-2 | Train s0–s2 / deploy s3 beats same-protocol v2.2; d̂ tracks stratum without receiving it | no improvement, or d̂ uncorrelated |
| P-v3-3 | The S4 overlap removal is performance-neutral (attention learns soft set-overlap via tags) | evidence increment collapses without `jaccard` — reinstate as one feature, report |
| P-v3-4 | Env-2, **empty registry**: v3 beats env-2 baselines; declared axioms a measured further increment | learning-is-the-floor fails on env-2 |
| P-v3-5 | Geometry-interface refactor (Phase 3) is a per-stratum no-op on DD2D | any stratum shifts beyond noise |

---

## 10. Open questions / risks

1. **D2's fork is real:** if s2 is within-length-limited, §5 doesn't fix it and the
   honest fallback is "s2 approaches the ~17 all-methods ceiling only via better
   within-length discrimination" — a representation problem, and possibly a paper
   *finding* rather than a fix.
2. **Refiner instrumentation coverage:** `culprits` and per-step query reporting may
   need drake-tamp/refiner plumbing. Per the A2a/A2b split (§6.6, §7.4) this no longer
   blocks Phase 2 on DD2D (backfill suffices for parity), but it is **mandatory for
   env-2** — scope the plumbing during Phase 2 so Phase 4 isn't gated on it.
3. **3D point budgets** (memory at 128–256 points × pool sizes) — benchmark before
   Phase 4 collection.
4. **Headroom honesty:** s2 is ~¼ of episodes; fixing it moves the overall dd2d_v3
   number by only ~2–3 FP. v3's value is claim quality — per-stratum dominance, length
   generalization, env-2 transfer — not a dramatic headline-number jump. Resist
   capacity surgery (D_MODEL, depth) unless a Phase-0 diagnostic indicts capacity;
   nothing measured so far does.
