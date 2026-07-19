# SPECTRE Proposal — living document

**Current version: v2.2.1** (2026-07-18).
Skeleton ranking for bilevel TAMP from geometry, typed post-mortem evidence, and sound deductions.
Standalone; assumes no prior project exposure. §13 records the revision history and maps every critique from each review round to its resolution. Future revisions edit this file in place.

---

## 1. Background: the problem in plain terms

A task-and-motion-planning (TAMP) robot solves problems like "get the buried scissors out of a cluttered drawer" in two levels. The **task level** proposes a *skeleton* — a symbolic action sequence such as `pick(o3); place-on-buffer(o3); pick(o7); place-on-buffer(o7); retrieve(target)` — saying *what* to do but not *where exactly*. The **motion level** then tries to *refine* the skeleton: fill in concrete grasps, placements, and paths by calling geometric samplers and collision checkers. Refinement can fail — the chosen objects may not clear a path, the staged objects may not fit on the counter — and the planner moves to the next skeleton.

On hard problems there are hundreds of candidates and refinement is expensive, so **the order of attempts dominates planning time**. Metrics: **rollout FP** (failed attempts before first success) and, primarily, **wall-clock**, because FP is sensitive to what one chooses to call an "attempt" (§10.1).

Our testbed is **DD2D** (Drawer Decluttering 2D): a top-down 2D drawer with 9–14 rigid items; the target starts un-graspable; the robot must stage a *subset* of blockers onto a size-limited buffer so a two-finger grasp corridor opens. The hard decision is *which subset* — it must both clear the target and jointly fit on the buffer.

**SPECTREv1** (a learned re-ranker over skeletons and their failure history) was diagnosed precisely: its ranking was a pure function of plan *length* (η²(length)=1.00, architecture-forced — anonymous object ids and no geometry meant same-length skeletons staging different subsets were identical inputs), and its failure context carried nothing but length (identity-scrambling changed behavior bitwise not at all). Meanwhile an image-conditioned low-level predictor (PIGINet-style) achieved 79% within-length discrimination — the geometric signal exists and is learnable — and a zero-training distance heuristic was unbeatable on easy strata, hopeless on hard ones. v2.x exists to dominate that frontier with mechanisms that transfer beyond DD2D.

---

## 2. Deployment realism (scope note)

Call information *privileged* if the deployed system would not have it. A deployed TAMP robot necessarily maintains a **geometric belief** — object poses and shapes derived from perception — because grasp sampling, IK, and collision checking cannot run without one. Conditioning a ranker on that belief, or on arithmetic over it, therefore requires no perception modules the system does not already need: the method's inputs are **deployable in kind**.

What our experiments do idealize is **exactness**: the ranker conditions on the same geometry the refiner plans in, with no model error. This is the standard idealization of the learned-TAMP literature (feasibility predictors are trained and evaluated against simulator state throughout the PIGINet/LAZY line), and we disclose it rather than attack it here. The setting where the idealization is dropped — perception noise, under which the full-observability ceiling (§3, P-D) dissolves and failure evidence acquires an *access* claim rather than an efficiency claim — is deliberately out of scope and recorded as future work (§12): it is the natural home of the project's perception-degradation crossover hypothesis, and pursuing it would add an entire experimental axis that this paper's headline question does not need.

What remains load-bearing from the realism discussion is not a sensor model but an accounting principle, promoted to P-F below: the adaptive pathway consumes only *observations of attempts the system actually made*, and no computation may be relabeled to move its cost off the books.

---

## 3. Design principles

**P-A. Identity must point at content.** An object's name carries no information; its geometry and role do. Wherever an object is mentioned — scene, skeleton, failure report — the representation must let the model look up its content. (Violated by v1; cause of the length-only collapse.)

**P-B. Never make the network learn what the planner logs for free.** The refiner knows which action failed, how deep it got, which bodies collided, which sampler exhausted. Microsecond-computable quantities (set intersections, area sums) enter as computed features, not as things attention must rediscover.

**P-C. Logic handles the provable part; learning handles the noisy part.** Sound deductions are consumed by a symbolic filter the network cannot override; suggestive evidence is consumed by the learned scorer.

**P-D. The static pathway owns the ceiling.** In fully-observable deterministic TAMP the correct ranking is a function of x₀ alone; within-episode failures add no *information* beyond x₀, only a cheaper route to it. The evidence pathway is therefore a **recovery mechanism for static mistakes**, its measurable value proportional to static error, and the system is built and evaluated static-first; ideal behavior is FP = 0 with evidence never needed. Evaluation corollary: the evidence increment must be measured where the static pathway errs *honestly* — distribution shift, a second environment — never against an artificially weakened static model.

**P-E. Degrade gracefully; never lose completeness.** No learned or deduced component may remove a candidate from the pool. Everything is a re-ordering; if every rule and prediction is wrong, the system reduces to attempting candidates in some order.

**P-F. Evidence is observation of attempts, accounted honestly.** The adaptive pathway's inputs are exactly the observations generated by attempts the system actually made — post-mortems, never pre-mortems (§5) — and no exact geometric computation may be smuggled into the static pathway under the label "feature" while its cost goes unreported. FP is ledger-sensitive; wall-clock with full cost accounting is primary. This is a principle about method identity and honest bookkeeping, not about sensors.

---

## 4. Terminology: proof facts and hint facts

("Certified" is avoided; PDDLStream already uses it for stream-asserted positive facts.)

- **Proof fact** — a statement following with certainty from an exact computation **performed by the refiner** plus declared domain assumptions (§6.3). Example: "with drawer contents C, the target has no collision-free grasp," established by the refiner exhaustively checking the finite grasp set with exact collision tests.
- **Hint fact** — evidence, not proof. Example: "the sampler failed to pack subset S in 150 tries," or "objects {o3} intersected every corridor at the failure state" (removing *other* objects can open corridors o3 doesn't touch).

Proof facts are proofs about the model the refiner plans in — in our experiments, the environment's true geometry. On hardware, where even the refiner's model is imperfect, proofs are model-relative; assumption 0 in §6.3 discloses this, and the telemetry there detects its violation.

Tier assignment is a small per-domain declaration, never learned. An empty declaration runs everything as hints; the proof tier is an optimization, not a requirement.

---

## 5. What the system does at deployment

Per episode: **(1)** encode the scene and the candidate pool; score all candidates with the empty fact set. **(2)** Apply the **proof-demotion filter** over accumulated proof facts: provably-dead candidates move to the back of the ranking — demoted, never deleted (P-E). **(3)** Attempt the top candidate. **(4)** On failure, run the **post-mortem harvest** at the deepest reached state (definition §6.2): read the refiner's failing check, its contact pairs, its sampler-exhaustion log, and the successful prefix, as typed facts. **(5)** Rescore and repeat.

**There is no pre-mortem probe, and the reasons are recorded** because an earlier revision contained one and the removal is a substantive decision, not an omission. First, on DD2D the probe is *provably* a post-mortem of the direct plan in disguise: the shortest plan is the single goal action, refining it consists of exactly the probe's check, so probe-first and direct-attempt-first are the same computation with different bookkeeping. Second, P-F: a probe is exact geometric computation relabeled as a zeroth non-attempt — its cost belongs on the attempt ledger where it is visible, and admitting it would blur the method's defining claim that adaptivity consumes only observations of genuine attempts. What is knowingly given up: the guarantee of exactly-zero FP on trivially-free targets — the model must *learn* target-freeness from scene geometry, and the pre-registered claim weakens accordingly (P2). In domains whose shortest plan is multi-step, a cost-based case for x₀-side checks may reopen; recorded as future work outside the method.

Consequences worth naming: t = 0 conditioning is purely static, so the training-time "no facts" dropout case matches deployment exactly; and no reporting split is needed to keep the PIGINet comparison fair — with no probe, the information-matched comparison is the default (§9).

---

## 6. Data layer

### 6.1 Scene records

Per problem, an object-centric record: per object, pose and a **footprint descriptor** — a fixed-size set of boundary sample points encoded by a small point-set network (explicitly not a radial profile: three of DD2D's seven shape families are concave, where radial functions are multi-valued and would silently corrupt exactly the interesting shapes). Plus global tokens for containers and **free-space/buffer dimensions**. The descriptor slot is the perception interface: polygons here, CLIP crop features in image domains, point-cloud features in 3D.

### 6.2 Post-mortem records and the harvest state

Per refinement **attempt** (skeleton × seed), logged from the refiner's own trace:

| Field | Content |
|---|---|
| `attempt` | skeleton id, seed |
| `failed_step` | index ℓ*+1, action schema, argument objects |
| `harvest_state` | see below |
| `facts` | typed facts (§6.4) read from the refiner's checks at the harvest state |
| `harvest_cost` | wall-clock of fact extraction (always logged) |

Under a backjumping refiner, "deepest reached state" is ambiguous. Definition: over all partial bindings explored, take the maximal bound-prefix length ℓ* (ties → most recent); record that prefix's binding, the world state produced by executing it, and the check that failed at ℓ*+1 under that binding. Collection writes the harvest state as an artifact; a unit test replays the bound prefix and asserts the state hash matches.

### 6.3 The soundness registry (assumptions, declared per domain)

A negative deduction is proof-tier only under explicit assumptions. For the removal-monotone rule used throughout ("blocked at contents C ⟹ blocked at every C′ ⊇ C"), DD2D declares:

0. **Model fidelity** — the fact was computed by the refiner against the model it plans in; the proof is about that model. In our experiments the model is the environment's true geometry; on hardware this assumption is where physical-model error enters, and it is monitored, not assumed silently (telemetry below).
1. **Exactness** — the failing check enumerated a finite set exactly (the discrete grasp set, exact polygon collision), not a sampler giving up.
2. **Removal-monotonicity / quasi-statics** — removing an object leaves other poses unchanged (true in DD2D by construction; false in settling 3D piles).
3. **Locality** — actions outside the container don't change collision status inside it.

Every domain ships its registry (a few lines), reported verbatim. Where an assumption cannot be declared, those checks emit hints — sharpness lost, correctness kept. **Soundness telemetry:** the fraction of proof-demoted candidates that later succeed is logged; under correct assumptions it is 0, and any nonzero value is a live alarm that a declaration — including assumption 0 on hardware — is wrong, caught in the wasted-attempts regime rather than the unsolvable regime (P-E).

### 6.4 The fact vocabulary (generated, not authored)

Facts are `(type, argument objects, tier, scalars)`. The vocabulary is **generated from the domain's stream/action schema**, not hand-written per problem: any stream-based substrate already types its failures (which stream instance exhausted, with which arguments, after how many samples), any mainstream geometry backend reports contact pairs on failed checks, and the bound prefix identifies which stream instances *succeeded*. Generated types = {one per stream type} ∪ {contact-witness} ∪ {prefix-success}. DD2D's "corridor witness" is not bespoke — it is what contact-pair readout *means* when the failing check is a 2D grasp test. What is hand-written per domain is only the registry above. Interface generality is not signal generality: contact pairs are always extractable, informative only where failures are relational (rich in DD2D; thin in kinematics-dominated domains) — measured per domain as the *evidence increment*, and reported rather than presumed.

| Fact type | Example (DD2D) | Tier | Source |
|---|---|---|---|
| blocked-at-contents | target has no clear grasp with contents C | proof (registry) | refiner's exact clearing check at harvest state |
| grasp-witness | {o3} intersects every open corridor | hint | contact pairs of that same check |
| pack-exhausted | subset S: 150 placement samples failed | hint | sampler exhaustion log |
| pack-impossible | Σ deflated areas of S > buffer area | proof | sound area bound |
| extracted-ok | o1 extractable under contents C (witness grasp) | proof (constructive) | successful prefix step |
| packed-ok | {o1,o2} packs (witness placements) | proof (constructive) | successful prefix step |

The positive rows matter: an attempt that staged {o1,o2} and died at `retrieve` *proved* o1 and o2 extractable and {o1,o2} packable — half of each attempt's information, previously discarded. By removal-monotonicity, `extracted-ok(o | C)` stays valid under any C′ ⊆ C.

**Worked example** (true minimal clearing set {o1,o2,o3}). Attempt A stages {o1,o2}, dies at retrieve. Harvest: proof `blocked-at-contents(C₀∖{o1,o2})` — by monotonicity every candidate staging a subset of {o1,o2} is provably dead → demoted, no learning involved; hint `grasp-witness{o3}` → soft evidence for o3-covering candidates; proofs `extracted-ok(o1)`, `extracted-ok(o2)`, `packed-ok({o1,o2})` → sound credit toward candidates containing them. Attempt B stages only irrelevant o5, dies at retrieve: near-empty proof demotion but the *richer* witness {o1,o2,o3}. A geometrically "far" failure produced the better diagnosis — no near/far scalar is needed, and the feared pathology (recursively chasing an irrelevant object's blockers) cannot occur: **facts constrain and hint; the prior proposes.** A hint about o5's blockers touches only candidates that already move o5, and whether those rank highly is the static pathway's job.

### 6.5 Label hygiene (critical path)

DD2D's labeler currently lacks the *arrangement-complete negative certificate*; many packing infeasibilities are `marginal(budget)` rather than proven. Training on marginals as negatives injects **correlated** noise concentrated at tight packings — the interesting boundary — teaching spurious area-conservatism invisibly. **Task 0 of the build is completing the certificate.** Fallback if it slips: exclude `marginal(budget)` from the loss; report the excluded count and stratum distribution.

### 6.6 Collection protocol

Per training problem: refine the candidate pool non-short-circuitingly for success labels; sample training failure-contexts with the rollout-aligned size mix (matching deployment's visit distribution); write every attempt's post-mortem record. **Mask-aware contexts:** the proof-demotion implied by a context's facts is applied when constructing that context, so the scorer trains on the pool distribution it will see. Pools of hundreds of candidates are capped (all short + a sampled tail), disclosed as data cost. One DAgger-style re-collection round (contexts under the trained policy's realized orders) is budgeted, with measured sensitivity.

---

## 7. Architecture

Everything binds through **episode-local object tags**: each object gets a tag embedding, randomly assigned per episode, permuted across epochs so no tag accumulates global meaning; the same tag appears wherever the object appears (scene, skeleton, fact). This discharges P-A and provably removes the v1 collapse mechanism (which was forced by identical inputs).

**Token families.** *Scene tokens*: [tag; footprint descriptor; pose; relation-to-target scalars]. *Candidate tokens*: operator schema + position + argument slots holding tags — a skeleton is a *program over the scene*. *Fact tokens* (one per fact): [fact-type; tier; failing schema; argument tags; scalars (depth, samples)]. *Global tokens*: container/buffer geometry, pool statistics.

**Computed overlap features (P-B).** Per candidate, against the current fact set: witness-overlap counts |S_c ∩ W_f| (max/mean over facts), coverage flags [S_c ⊇ W_f], proven-dead flag, proven-prefix credit (fraction of staged set covered by extracted-ok/packed-ok). Set intersection is microseconds; attention should not have to rediscover it.

**Static-side scalar features: harness-side by default.** An earlier revision fed hand-picked arithmetic scalars (area slack, centroid distances, corridor-overlap counts) into candidate embeddings. v2.2 keeps them out of the model's inputs, on two grounds that survive independently of any perception argument. First, corridor-overlap counts require the same collision machinery as the deleted probe — exact computation smuggled in as features, violating P-F's accounting rule. Second, a fixed list like {slack, centroid distance} is DD2D-flavored hand engineering whose proper role is *adversarial*: these are the null models the learned encoder must **beat** (§10.3), and moving them into the inputs would let the model pass its own exam by copying the answer key. Default expectation, stated as such: the encoder should discover slack-like statistics itself — that is what representation learning is for — and a "+scalars" input ablation measures what discovery costs. Schema-generic scalars (pairwise distances) may be auto-generated as inputs; anything domain-flavored is harness-side only.

**Scorer.** Per-candidate cross-attention (candidate as query over scene + fact memory), concatenated with overlap features → one logit. Linear in pool size; hundreds of candidates at d = 64 is milliseconds. This solves the *computational* large-pool problem only; the *statistical* one — picking one of hundreds at t = 0 from limited data — is addressed only indirectly (geometry, auxiliary head, sublist-sampled listwise training) and is named as an open pressure point, not claimed solved.

**Proof-demotion filter (outside the network).** Proof facts compile to demotion rules on the *ranking*, never the pool: proven-dead candidates go to the back; the pool never empties; if everything is proven dead, demoted candidates are attempted anyway in order. The network cannot override a proof; a wrong proof only reorders (P-E).

---

## 8. Training

**Loss.** Listwise Plackett–Luce over (remaining pool, its successes) — v1's hardest-won correct decision (pointwise BCE is demonstrably mis-aligned with time-to-first-success). Large pools: sampled sublists containing ≥ 1 success.

**Contexts.** Teacher-forced from real attempt traces with their real fact records, rollout-aligned size mix, mask-aware. Fact identity is never synthesized or scrambled at training time.

**Evidence dropout, weighted at t = 0** — exactly the deployment distribution, since t = 0 truly has no facts. The static pathway must stand alone (P-D).

**Auxiliary head (well-posed).** Minimal feasible subsets are routinely non-unique, so "in *the* solution" has no label. Supervise **necessary(o)** (in every minimal feasible subset) and **relevant(o)** (in at least one), both computable from the enumerator; small weight, imbalance-corrected, ablatable. Design bet: direct gradient into the geometry pathway in low data.

**Live evidence-usage instrumentation.** Shared parameters + t = 0-heavy dropout create a path of least resistance: ignore facts, minimize loss statically — invisible in the loss curve. So the v1 scramble diagnostic becomes a training-time gauge: at every validation checkpoint, Δ(rollout FP, scrambled vs real fact identities) on the val split, watched across training.

---

## 9. Baselines and comparisons

All methods condition on the same scene; image baselines receive renders of it. Main table:

1. **astar-dist** — zero-training geometric distance heuristic (the easy-strata bar).
2. **slack ordering** — the strongest known one-scalar ranking (the shortcut to beat, and the elimination-ladder null).
3. **Hand-rule stack** — slack (or astar-dist) + proof demotion + witness-overlap boost; **zero learned parameters**. The most dangerous baseline in the paper, included because of that: if it lands near the full model, that is the finding, and we want it in week two.
4. **PIGINet-style** — the learned low-level static comparator (rendered crops).
5. **LAZY-style untyped adaptive** — static prior − β·(action-overlap with failed skeletons), β tuned: untyped failure conditioning, exactly what the typed pathway claims to beat. Its absence would make the adaptive comparison a strawman against our own v1.
6. **SPECTREv1** — regression reference.
7. **Oracle** — labeler-ordered floor.

**Information parity.** With no probe, v2.2-static and PIGINet condition on the same scene at t = 0 by construction. The remaining representational confound — polygons vs pixels of the same scene — is handled by the committed 2×2 cell **v2.2-with-crops** (CLIP features in the descriptor slot), with PIGINet-with-polygons as a stretch cell.

**Statistics.** Paired per-problem differences vs the strongest baseline, stratified bootstrap CIs; ≥ 3 genuinely distinct seeds with a CI test asserting distinct checkpoint checksums (this project has shipped silently duplicated "seeds" before); wall-clock end-to-end including harvest costs, reported beside FP — wall-clock is primary wherever bookkeeping could differ.

---

## 10. Experiments, gates, and pre-registered predictions

### 10.1 Metric discipline

FP is ledger-sensitive: near-free computations can be labeled "attempt" or "feature" and move the number. Wall-clock with full cost accounting (harvest costs, inference costs) is therefore the primary metric; FP is reported as the interpretable secondary.

### 10.2 Gate G0 — does the benchmark test its own thesis? (before any model code)

Run the slack ordering and a pairwise-features GBDT probe (per-object hand features → P(relevant), aggregated per skeleton) through the full harness across the buffer-tightness dial λ. Choose λ* where cheap statistics degrade but the oracle still solves — the regime where subset-coupled feasibility binds. Pre-registered off-ramp: if no such λ exists, DD2D as configured cannot support the subset-coupling claim, and the honest next step is benchmark work, not model work.

### 10.3 The elimination ladder (anti-shortcut acceptance test)

"η²(length) < 1" is far too weak — a model learning *only* area slack passes it while remaining subset-blind (area is the new length). Acceptance is a nested variance decomposition of v2.2-static's scores — length → +slack → +pairwise proximity → **residual (true subset identity)** — with the bar: *v2.2-static beats the slack ordering by a paired margin, CI excluding zero, on strata ≥ 2 at λ**. The general lesson, applied prospectively: know the cheapest statistic that could explain your ranking before training.

### 10.4 Pre-registered predictions

- **P1** (near-certain; the fix is mechanical): v2.2-static's η²(length) ≪ 1 with substantial within-length variance — and it clears the §10.3 ladder, which is the real bar.
- **P2**: v2.2-static ≥ PIGINet per stratum on FP and wall-clock. Honest weakening relative to the probe-bearing revision: no exactly-zero s0 claim — the claim is s0 ≤ PIGINet, and astar-dist may remain champion on easy strata; if it does, the distance prior enters as a feature (the v1 heuristic-prior precedent), and "dominates the frontier" is claimed only where the numbers support it.
- **P3** (evidence is used, and matters under shift): the typed-evidence increment — real vs scrambled fact identities — is > 0 with CI clearance under held-out shape families, and larger there than on-distribution. Registered under shift because on-distribution a *strong static model legitimately leaves evidence little to recover* (P-D); planting the falsifier there would punish success.
- **P4**: the non-learned hand-rule stack cuts FP over its own static base, CI excluding zero.
- **P5**: v2.2 ≥ the LAZY-style untyped baseline wherever the evidence increment is nonzero — typing must beat failure-counting or the typed machinery is unearned.

### 10.5 Second environment

One pre-existing, not-designed-by-us domain (Khodeir-style clutter/sorting or a kitchen domain): descriptor slot swapped to that domain's perception, a written soundness registry (or the documented statement that nothing is declarable and all facts run as hints), reported domain-specific line count, and the P3 evidence-increment measurement — the priority use, because hand-written demotion rules exist for DD2D but not there, which is exactly where a *learned* consumer of typed facts must earn its place.

---

## 11. Generalization contract

v2.2 ports to a new domain given four interfaces: (i) **object-centric geometry access** — poses plus any shape representation for the descriptor slot; (ii) **refiner introspection** — deepest bound prefix, failing check, contact pairs, sampler-exhaustion logs (standard in mainstream backends and stream planners); (iii) **typed action schemas** — any PDDL-family TAMP; (iv) **a soundness registry** — a few declared lines per check type, *or nothing*, in which case all facts are hints and proof demotion is simply inactive. Nothing references drawers, buffers, or corridors. Per domain, the paper reports: registry contents, domain-specific line count, harvest wall-clock, soundness telemetry, and the measured evidence increment.

---

## 12. Positioning, scope, and build order

**Novelty, honestly located.** Deducing constraints from failed refinement has prior art, cited first, not last: IDTMP adds negated constraints to an SMT solver on motion failure (our proof demotion is skeleton-level clause learning); Srivastava et al.'s interface layer feeds refinement errors to the task level; Lagriffoul's culprit detection extracts responsible constraints; Stilman-style NAMO backward chaining and Krontiris–Bekris dependency graphs perform blocking-subset deduction natively. Claimed contribution: the combination none of them has — a schema-generated, domain-portable **typed-fact interface** with an explicit **proof/hint split** (sound deductions consumed symbolically, suggestive ones by a learned ranker), bound at object level into a geometry-aware scorer, with declared soundness assumptions and runtime telemetry.

**Scope: one headline.** The ICRA headline is the **static conditioning study** — what should a feasibility ranker condition on (elimination ladder, information-parity comparison, the certificate-features ablation). The typed-evidence pathway is the secondary contribution, *leading* only if the second environment's P3 result lands in time. Pre-registered fallback: if the evidence increment is null on-distribution and under shift, the paper is the conditioning study plus the proof tier as a non-learned planner-side contribution — the hand-rule stack's numbers stand regardless.

**Future work (recorded, out of scope).** Under perception noise the full-observability ceiling dissolves — the policy conditions on an estimate while refinement outcomes reflect the model planned in — and failure evidence acquires an *access* claim rather than an efficiency claim. That regime is the natural home of the project's perception-degradation crossover hypothesis and the principled successor study to this paper; it is excluded here because it requires a belief/model evaluation split and noise-model machinery orthogonal to the headline question. Likewise, domains whose shortest plan is multi-step may reopen a cost-based case for x₀-side exact checks (§5).

**Build order** (each stage a publishable fallback): **Task 0** — arrangement-complete negative certificate (fallback: marginals out of the loss, counts reported). **Task 1** — gate G0 (λ*); hard off-ramp if it fails. **Task 2** — harness: LAZY baseline, hand-rule stack, paired-bootstrap statistics, seed checksums, wall-clock accounting. **Task 3** — v2.2-static, judged by the ladder and P1/P2. **Task 4** — proof demotion + hand-rule stack in the main table (P4). **Task 5** — learned typed-evidence pathway with the live scramble gauge (P3/P5). **Task 6** — shape-family shift experiments and the second environment.

---

## 13. Revision history and critique maps

**v2 → v2.1** (advisor/reviewer round; full map retained from the v2.1 document): soft demotion + soundness registry replacing deletion masks (completeness risk); necessary/relevant aux targets (ill-posedness under non-unique minima); boundary-point descriptor (radial fails concave shapes); computed overlap features (attention re-deriving set algebra); positive prefix facts (discarded successful prefixes); harvest-state definition + replay test (backjumping ambiguity); marginal-label exclusion / certificate on critical path; mask-aware contexts + DAgger round; live scramble gauge; elimination ladder (area is the new length); information 2×2 (input confound); LAZY baseline (missing untyped-adaptive comparator); hand-rule stack promoted; paired statistics, seed checksums, wall-clock; proof/hint renaming ("certified" collision); CDCL/IDTMP prior-art positioning; shift-robustness hypothesis made explicit; one-headline scope + G0 gate.

**v2.1 → v2.2** (project-discussion round): pre-mortem probe removed (provably a post-mortem of the direct plan in disguise on DD2D; ledger honesty); static-side hand-picked scalars removed as inputs and re-scoped as harness-side null models; deployment-realism analysis added, including a percept/reality fidelity split with a noise axis and an access-claim prediction.

**v2.2 → v2.2.1** (this revision):

| Critique | Resolution |
|---|---|
| The percept/fidelity layer over-literalizes the realism critique and adds an experimental axis the headline does not need | Two-copy design, σ axis, σ-conditioning, and the σ-growth prediction removed throughout; realism reduced to a scope note (§2): inputs are deployable-in-kind, exactness is disclosed as the field-standard idealization, and the noise/access direction is parked as recorded future work (§12) |
| What survives that round, on independent grounds | Pre-mortem stays removed (the disguise argument never depended on perception); P-F retained, reframed as an accounting/identity principle; the scalar disposition retained on hand-engineering / null-model grounds alone (§7); P-D restored to its full-observability form with the evaluation-placement corollary |

---

## 14. Epistemic status

| Claim | Status |
|---|---|
| v1's length-only ranking was architecture-forced | Demonstrated |
| Tag binding removes the collapse mechanism | Near-certain (removes the forcing; residual risk is optimization) |
| Pre-mortem ≡ direct-plan post-mortem on DD2D | Derivable from the domain definition |
| Proof demotion sound under the DD2D registry | Provable given the registry; the registry is the assumption; telemetry monitors it |
| Conditioning inputs are deployable-in-kind (no modules beyond the belief the refiner needs) | Definitional under §2's definition of privileged; exactness idealization disclosed |
| Typed facts shift-robust vs learned geometry (P3) | Conjecture — the research bet of the evidence half |
| L0 encoder rediscovers slack-like statistics | Design bet; the +scalars ablation measures it |
| DD2D at λ* exercises subset coupling | Conditional on G0; off-ramp pre-registered |
| Aux head helps in low data | Design bet; ablatable |

The honest closing tension: at full observability the static and evidence pathways compete for the same headroom — the better the static ranker, the less evidence can show on-distribution. v2.2.1 resolves this the way v2.1 did, by *where it evaluates* (shape-family shift, the second environment), and records the stronger resolution — a perception-noise regime where evidence's claim upgrades from efficiency to access — as the successor study rather than carrying its machinery into this paper. Whether the evidence increment is CI-clean under shift is exactly what P3 and Tasks 5–6 exist to find out; the fallback if it is not is written down in advance.
