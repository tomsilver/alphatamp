# Unified culprits, coverage, and waste — consolidated definitions

*Consolidates the 2026-07-31 design discussion.*

> **Status: IMPLEMENTED AND DEPLOYED (2026-07-31).** `unified_evidence.py` implements this
> spec and is the **sole, unconditional** coverage/waste path — there is no toggle. (The old
> `unified_coverage` flag on the then-`TrainV3Config` was retired when the legacy
> `S(c) = args \ goal_objects` path was removed in the 2026-08-12 publication refactor, commit
> `37b477c`; `TrainV3Config` is now `TrainConfig`.) Measured on dd2d_v4 (n=100,
> 3 seeds, uncensored): **5.78 ± 0.10 against the previous deployed 7.44 ± 0.76 — −1.66 FP,
> 95% CI [−2.71, −0.71]**, every seed beating every baseline seed. ADR:
> [`decisions.md` 2026-07-31](decisions/06-v3-performance.md#2026-07-31-unified-coverage-waste-is-the-deployed-definition).
>
> **Two claims in this document were falsified by the probes and are corrected inline
> below:** DD2D contexts are *not* dominated by the terminal `retrieve` (all 114
> culprit-bearing records are `pick`), and the definitions are *not* bit-identical to the
> deployed ones on DD2D (coverage differs on 100% of contexts). The construction is better
> anyway, for a reason the design did not anticipate — see §1 on `__wall__`.

---

## 0. Setting and notation

Fix a grounded problem: object set `O`, goal atoms `G`, and the set of ground operator instances of the domain. A candidate skeleton is `c = ⟨s_1, …, s_L⟩`, each step `s_j` a ground operator with preconditions `pre(s_j)`, add-effects `A⁺(s_j)`, and delete-effects `A⁻(s_j)`. We say step `s` **touches** object `k` iff `k` is mentioned in `A⁺(s) ∪ A⁻(s)`; `touch(c, k)` is the set of indices of steps of `c` touching `k`.

`ŝ_j(c)` denotes the candidate's own **predicted abstract state immediately before step `s_j`**, computed by STRIPS progression from `abs(x₀)` along `c`'s prefix. This is the identical machinery the `state_delta` field already runs; no new instrumentation.

The failure context `F` is the set of failure records observed so far in the episode. A record `r` carries (at least): its class (§2), its failed step `s_r` with that step's ground effects, its blamed objects `blame(r)`, and — for class 2 — the observed deviation `Δ_r = (A_r, D_r)`: atoms unexpectedly added, atoms unexpectedly missing.

**Deployment-start convention (leakage invariant, per as_built_v3 §7.4):** both features below are defined as exactly `0` whenever `F = ∅`. The first attempt is purely static; everything below applies from the first observed failure onward. All ratios use the convention `0/0 := 0`.

---

## 1. Shared machinery: two filters and a lemma

Two filters are used by everything downstream. Both are computed from the grounded domain alone — the same epistemic status as writing the operator file, per the §3 domain contract. Neither mentions any environment by name.

**Universal objects and anchored atoms.** An object is **universal** iff it appears in the argument list of *every* ground operator instance of the domain (behaviorally: the robot). An atom is **anchored** iff it mentions at least one non-universal object; nullary atoms are unanchored by definition. `anch(X)` is the anchored subset of atom set `X`. On SB2D this admits `Pressed(b)`, `RobotAboveButton(robot, b)`, `StickAboveButton(stick, b)`, `Grasped(robot, stick)` (the stick is non-universal — it is absent from the robot-press operators) and rejects `HandEmpty(robot)` and `AboveNoButton()`. The filter exists because bookkeeping atoms otherwise thread everything together: unfiltered, `HandEmpty` chains every DD2D staging pair into the causal spine (breaking waste's backward compatibility completely) and creates spurious context matches between unrelated steps.

**Actionable, non-universal objects.** Object `k` is **actionable** iff some ground operator instance's effect atoms mention `k`. The culprit pool is restricted to actionable **non-universal** objects — universality applies to `K` itself, not only to atoms. The universality exclusion is the filter doing real work: the robot *is* actionable (mentioned in `RobotAboveButton` effects) and can be named by class-2 deviations, but blame on an object touched by every step of every candidate is non-discriminative for coverage and actively corrupting for waste, where a robot-touching superfluous step (`PlaceStick`) would count as spuriously justified. Actionability is **not** idle on DD2D, contrary to an earlier draft of this paragraph: measured over dd2d_v4, `__wall__` is the *most frequent* reported culprit (1321 mentions across 3182 culprit-bearing records, against 939 for `target` and a few hundred each for the movable items). It is not an abstract object at all, so it is filtered here — and its removal is the main reason the unified culprit pool is *smaller* than the deployed one (mean |K| 1.99 → 1.67) and the coverage feature correspondingly sharper. On SB2D the filter is genuinely idle, since the table and walls never enter the abstract object universe.

**Ranking-inertness lemma (coverage only).** Any object that is uniformly covered, or uniformly uncovered, across the entire candidate pool cannot reorder candidates: coverage is `(a_c + u)/(n + m)` with `u, m` constant across the pool and is strictly monotone in the candidate-varying count `a_c`. Consequence: a filter misfire (a junk object slipping through) degrades feature magnitude but provably cannot change the coverage-induced ranking. The lemma does **not** extend to waste, whose denominator is per-candidate — and that asymmetry is load-bearing: an earlier draft leaned on the lemma to tolerate the robot inside `K`, which is exactly the unprotected case, since waste's justification is a per-step existential over `K` and one universal object in the pool spuriously justifies every early superfluous step touching it. Universal objects are therefore excluded from `K` outright rather than tolerated as inert.

---

## 2. Culprits

A failed refinement sample dies in exactly one of two ways, and each way already computes the objects to blame — culprit extraction reads an answer the refiner produced anyway, preserving the observation-only invariant (no extra stream calls).

> **Class 1 — constraint rejection.** A validity check (collision, bounds) rejects the sample before a successor state exists. `blame(r)` = the objects named by the violated check.
>
> **Class 2 — effect mismatch.** The sample executes, and exact-trace checking finds the observed abstract state ≠ the predicted one. The raw deviation is `Δ_r = (A_r, D_r)` — atoms unexpectedly added, atoms unexpectedly missing. The record's coverage-bearing content is the **collateral deviation**: the raw deviation minus the failed step's own declared effects,
>
> ```
> Δ̃_r = (Ã_r, D̃_r) = ( A_r \ A⁻(s_r),  D_r \ A⁺(s_r) )
> ```
>
> `blame(r)` = the objects mentioned in the atoms of `Ã_r ∪ D̃_r`. The stripped remainder — the step's own adds that never materialized (`D_r ∩ A⁺(s_r)`) and its own deletes that never took (`A_r ∩ A⁻(s_r)`) — is **means failure**: the statement that this query, as attempted, does not produce its effects. It generates no culprits, and it is carried by the burned-query token (schema, args, exhausted) — the channel this design has assigned to reachability/modality evidence throughout.

The **culprit pool** at any point in the rollout is

```
K = (Actionable \ Universal) ∩ ⋃_{r ∈ F} blame(r)
```

*Plain language:* every record splits into two kinds of information. What the step *failed to do* — couldn't reach, couldn't grasp its way in, effects never took — is means failure, and it rides the token channel as a burned query. What the failure *did to bystanders* — the blocker standing in the way, the button pressed by accident, the stick knocked out of the grasp — is actionable blame, and only that enters the culprit pool. So DD2D's grasp-collision blockers enter verbatim (class 1); SB2D's incidentally pressed buttons enter via `Ã_r` (class 2); and an out-of-reach robot-press generates *no culprits at all* — its entire deviation is its own effects failing to take, `Δ̃_r = (∅, ∅)`, and the reachability signal is exactly the burned query. A table-collision record likewise contributes no culprits and survives as a token. In one sentence: **culprits are collateral damage; burned queries are failed means.**

---

## 3. Context matching: where a candidate would re-enter the failed situation

The record's **matching signature** is the anchored, *signed* effect signature of its failed step: `sig⁺(r) = anch(A⁺(s_r))`, `sig⁻(r) = anch(A⁻(s_r))`. The **matched steps** of candidate `c` for record `r` are

```
M_c(r) = { j : (A⁺(s_j) ∩ sig⁺(r)) ∪ (A⁻(s_j) ∩ sig⁻(r)) ≠ ∅ }
```

*Plain language:* a step of `c` re-enters `r`'s situation iff it tries to accomplish the same thing the failed step was trying to accomplish — matched by what the step *does*, not what it is called, so a stick-press of `b2` re-enters the context of a failed robot-press of `b2`. Sign is respected (adds match adds, deletes match deletes): a step that adds what the failed step deleted is the opposite of a re-attempt, not an instance of one. The anchor filter prevents bookkeeping atoms (`AboveNoButton`, `HandEmpty`) from matching unrelated steps to each other; without it, signed matching would introduce a `HandEmpty` collision between `pick(o)` and `pick(o′)` on DD2D that the deployed feature never had.

---

## 4. Coverage

The covered-test is **class-dependent**, because the abstraction can state class-2 hazards but not class-1 hazards. Blockedness is not a predicate (the information-poverty principle), so class 1 gets an index proxy; an unpredicted atom *is* a predicate, so class 2 gets an exact state test.

> **Class-1 test (index precedence, per record `r` blaming `k`):**
> if `M_c(r) ≠ ∅`: some `j ∈ touch(c, k)` satisfies `j < min M_c(r)`
> (equivalently, before *every* matched step, since min is earliest);
> if `M_c(r) = ∅`: fallback to bare membership, `touch(c, k) ≠ ∅`.
>
> **Class-2 test (state entailment, per record `r` and blamed object `k`):**
> writing `X|_k` for the atoms of `X` mentioning `k`,
> for every `j ∈ M_c(r)`:  `Ã_r|_k ⊆ ŝ_j(c)`  and  `D̃_r|_k ∩ ŝ_j(c) = ∅`.
> (Vacuously true if `M_c(r) = ∅` — the deviation cannot recur at any recognized recurrence.)
>
> **Covered, conjunctive across records:**  `covered(k | c) ⟺` every `r ∈ F` with `k ∈ blame(r)` passes its class's test.
>
> ```
> coverage(c) = |{ k ∈ K : covered(k | c) }| / |K|
> ```

*Plain language.* Each record is a short story: "I was trying to accomplish X, and `k` ruined it." Coverage is recall over those stories. For a collision story, the candidate gets credit if it deals with `k` before the first point where it would retry X — and if it never retries X, dealing with `k` at all suffices (that fallback *is* the old DD2D semantics, not a patch on it). For an accident story, the candidate gets credit exactly when replaying the recorded accident (its collateral deviation, §2) against the candidate's own predictions is a no-op everywhere X recurs — the accidental effect has been made intentional. The state test is what survives multi-touch plans: a plan that presses a switch and later un-presses it before the danger point is correctly denied credit, which no first-index or last-index test can decide, because polarity is a state question wearing an index costume.

**Why the collateral restriction is in the test.** Run the *unrestricted* test on an out-of-reach record — `A_r = {AboveNoButton}`, `D_r = {Pressed(b1), RobotAboveButton(robot, b1)}` — and both halves misbehave. The `D_r` half is satisfied by construction at any matched step (a step about to add those atoms has them absent from `ŝ_j`), so a candidate retrying the identical doomed robot-press earns full credit. The `A_r` half is worse — anti-signed: `AboveNoButton ∈ ŝ_j` holds before `FromNothing` matched steps (it is that operator's precondition) and fails before `FromButton` ones, so a correct stick fix mid-chain can be *denied* the credit the doomed retry receives. Both pathologies live entirely in the means components and vanish under `Δ̃_r`; the collateral components are the ones the test was built for (`Pressed(b1)` entailed before the recurrence) and genuinely discriminates on (`Grasped(robot, stick)` absent from a robot-only plan's predictions, present in a stick retry's).

**Why per-object.** Without `X|_k`, every object in `blame(r)` shares one bit and coverage becomes recall weighted by `|blame(r)|`: a stick sweep incidentally pressing two buttons (`Ã_r = {Pressed(b2), Pressed(b4)}`) would deny a candidate credit for the `b2` it discharged because of the `b4` it did not. The restriction makes the per-story recall of §6 literal. It is invisible whenever the collateral deviation names a single object — the common case — and an atom mentioning two blamed objects rides in both restrictions.

**Exactness conditions, stated rather than implied.** The class-1 index test is exact when discharge is monotone within the candidate — nothing un-touches `k` before the recurrence. This holds on both current environments for structural reasons: SB2D has no operator with `Pressed` in its delete effects (checked against `stickbutton2d_bilevel.py`), and DD2D candidates touch each staged object exactly once. On future domains with put-it-back-later structure the class-1 proxy under-determines, and the residual belongs to the record tokens and geometry cross-attention. The class-2 state test needs no such caveat and is pinned by a synthetic reversible-toggle test where index and state tests disagree.

**Backward compatibility on DD2D.** With terminal or absent contexts, the class-1 test reduces to `k ∈ S(c)` — bit-identical to the deployed feature by construction. The two can disagree only on records with a *non-terminal* context (a blocked intermediate `pick(o)`) scored against candidates that stage both `k` and `o` in inverted order — and on that intersection the new test is a causal correction (the plan really would re-hit the collision), not noise. **MEASURED 2026-07-31, and the prediction was wrong.** Disagreement is **7.17%** of (candidate, culprit) pairs, not "well under 1%", and the premise fails too: extraction-side terminal contexts do **not** dominate — all 114 culprit-bearing dd2d_v4 records are `pick`, none are `retrieve`. Breakdown: **9.02%** of pairs are goal objects (`target`) entering `K`, which the deployed `S(c)` bars structurally and which are uniformly covered here, hence ranking-inert by the lemma; **1.58%** are the genuine causal correction described above. Coverage *vectors* differ on 100% of contexts, so existing checkpoints do **not** stand untouched — but top-1 pick is identical on 93/93 contexts, and a 3-seed retrain measures the unified definition **better** (5.78 vs 7.44), so the migration is a gain rather than a cost.

---

## 5. Waste

Waste's numerator was never the problem; its denominator was. `S(c) = args \ goal_objects` hard-codes "discretionary = touches non-goal objects," which is true on DD2D and false wherever tools exist. The unified denominator computes discretion from the candidate's own causal structure: **discretionary work is work the abstraction cannot explain.**

> **Superfluous steps (anchored backward relevance).** Initialize the needed set `N ← anch(G)` and walk `c` back-to-front. Step `s_j` is **live** iff `A⁺(s_j) ∩ N ≠ ∅`; if live, update `N ← (N \ A⁺(s_j)) ∪ anch(pre(s_j))`; otherwise `s_j` is **superfluous**. `W(c)` = the superfluous indices. (Machinery from the plan-justification literature — Fink & Yang's well-justified plans is the entry point; verify the citation before it enters `research_lit.md`.)
>
> The pass never consults `A⁻`, so it detects irrelevance but not **threats** — a step that deletes an atom a later step needs is not flagged. That is sound here only because candidates are STRIPS-valid sequential plans by construction, so no such step can exist; the guarantee lapses for partial-order or parallel plans, where the denominator would need a genuine causal-link analysis rather than a backward sweep.
>
> **Justified.** A superfluous step `s_j` is justified iff it touches some `k ∈ K` at a position satisfying the class-1 timing for `k` (before every matched step of every record blaming `k`; bare touch when no record's context is matched). Justification attribution stays index-level even for class-2 blame — a recorded simplification carrying the same monotonicity caveat as the class-1 test.
>
> ```
> waste(c) = |{ j ∈ W(c) : ¬justified(j) }| / |W(c)|
> ```

*Plain language.* Waste is precision over unexplained work: of the steps the abstract model says you didn't need, what fraction answer to nothing the evidence has named? Live steps — the causal spine that actually produces the goal, including tool acquisition — are off the books entirely. That is what dissolves the SB2D anti-signal rather than suppressing it: `PickStick` feeds `Grasped` feeds `StickPress` feeds `Pressed(b4)`, so it is live and never enters the denominator, with no per-environment switch anywhere. What remains in the denominator is genuinely pointless work (a place-repick cycle, a trailing `PlaceStick`), which the old object-level `S(c)` could not even see — the stick was one object whether used well or fiddled with.

**Backward compatibility on DD2D.** Under the anchor filter, each staging pair is a causal dead-end (`on-buffer(o)` feeds no anchored need; `handempty` is filtered), so `W(c)` is exactly the staging steps — and since every staged object contributes the same two steps, the step-ratio equals the deployed object-ratio identically, with the same minority-class precedence correction as coverage. One shared offline probe covers both features.

**Known false positive, characterized.** A step whose only causal contribution is an unanchored atom is marked superfluous — on SB2D, a mid-plan `PlaceStick` that frees the hand for robot presses contributes only `HandEmpty`/`AboveNoButton` and is mislabeled as unexplained work. Bounded (one step per modality switch, only when `stick ∉ K`); the unfiltered alternative is a total compatibility break rather than a cosmetic one.

---

## 6. The whole construction in three sentences

Failures explain themselves: every rejection the refiner produces already names the objects to blame, whether by a violated collision check or by the collateral atoms of a trace mismatch — while what a step failed to do about its own effects rides the burned-query tokens. **Coverage** is recall over that blame — a candidate is credited for a named object only if it discharges it before re-entering the situation that named it, where "discharge" is state-entailment when the abstraction can express the hazard and touch-precedence when it cannot. **Waste** is precision over unexplained work — of the steps the abstraction's own causal chain cannot justify, the fraction that answer to no named object.

Nothing in that paragraph names a drawer, a button, or a stick. The per-domain input is the operator schemas themselves.

---

## 7. Worked example — DD2D

Target `t` behind blockers `o1, o2, o3`; candidates stage a subset to the buffer, then extract. Schematic effect names (`holding(t)` etc. stand in for the exact DD2D atoms). Pool:

```
c1 = ⟨pick(o1), pb(o1), extract(t)⟩                    stages {o1}
c2 = ⟨pick(o2), pb(o2), extract(t)⟩                    stages {o2}
c3 = ⟨pick(o1), pb(o1), pick(o2), pb(o2), extract(t)⟩  stages {o1, o2}
c4 = ⟨pick(o3), pb(o3), extract(t)⟩                    stages {o3}
```

**Attempt 1:** the static ranker tries `c4`. Its extraction grasp collides; the check names `o1`. Record `r1`: class 1, failed step `extract(t)`, `sig⁺ = {holding(t)}` (schematic), `blame = {o1}`. Now `K = {o1}`, and for every candidate the matched step is its own terminal extraction — the dominant DD2D case, where precedence is automatic and every number below coincides bit-for-bit with the deployed `|S∩K|/|K|`, `|S\K|/|S|`.

Relevance pass (waste denominator): each staging pair dead-ends — `pb(o)` adds `on-buffer(o)`, which no anchored need consumes; `pick(o)`'s `holding(o)` therefore never enters `N`; `handempty` is filtered. So `W` = the staging steps, and the extraction chain is live.

| candidate | coverage | waste | reading |
|---|---|---|---|
| `c1` | 1/1 = **1.0** | 0/2 = **0.0** | responds to exactly the evidence |
| `c3` | 1/1 = **1.0** | 2/4 = **0.5** | responds, plus unexplained work on `o2` |
| `c2` | 0/1 = **0.0** | 2/2 = **1.0** | all work unexplained, culprit ignored |

**Attempt 2** tries `c1`. Suppose the scene in fact needs both `o1` and `o2` cleared: `c1`'s extraction fails naming `o2`, so `K = {o1, o2}`. Recompute: `c3` → coverage 2/2 = 1.0, waste 0/4 = 0.0; `c2` → coverage 1/2 = 0.5, waste 0/2 = 0.0. **Attempt 3** tries `c3` and succeeds. FP = 2, each failure converted into exactly one step up the escalation ladder — the evidence-proportional behavior the deployed features already exhibit, reproduced here because on terminal contexts the unified definitions *are* the deployed ones.

---

## 8. Worked example — SB2D

Four buttons. `b1` sits on the corridor to `b2` (any direct approach to `b2` brushes `b1`); `b4` is out of robot reach. `rp` = robot-press, `sp` = stick-press (the From-Nothing/From-Button variants as the chain requires). Pool:

```
c_A = ⟨rp(b2), rp(b1), rp(b3), PickStick, sp(b4)⟩
c_B = ⟨rp(b1), rp(b2), rp(b3), PickStick, sp(b4)⟩
c_C = ⟨rp(b3), rp(b2), rp(b1), PickStick, sp(b4)⟩
c_D = ⟨PickStick, sp(b4), sp(b2), sp(b1), sp(b3)⟩
c_E = ⟨rp(b1), rp(b2), rp(b3), PickStick, PlaceStick, PickStick, sp(b4)⟩
```

**Attempt 1:** `c_A`. Every sampled approach to `b2` brushes `b1`; each sample executes and the trace check finds `Pressed(b1)` unpredicted; the query exhausts. Record `r`: class 2, failed step `RobotPressButtonFromNothing(robot, b2)`, raw deviation `Δ_r = ({Pressed(b1)}, ∅)` — the press of `b2` itself completed, so the only deviation is the brush. `Pressed(b1)` is not among the failed step's own deletes, so it is fully collateral: `Δ̃_r = ({Pressed(b1)}, ∅)`, `blame = {b1}`. Signature: `sig⁺ = {Pressed(b2), RobotAboveButton(robot, b2)}`, `sig⁻ = anch({AboveNoButton}) = ∅`. `K = {b1}`.

Old formulas, for contrast: `S(c) = {stick}` for every candidate here, so old coverage ≡ 0 (`b1` is a goal object, structurally barred from `S`) and old waste = 1 for every stick-using plan — including the plan that responds perfectly. The features are blind and anti-signed respectively.

Unified coverage (class-2 state test; matched step = the step adding `Pressed(b2)`, whatever its modality):

| candidate | matched step | `ŝ` before it contains `Pressed(b1)`? | coverage |
|---|---|---|---|
| `c_B` | step 2, `rp(b2)` | yes — pressed at step 1 | **1.0** |
| `c_C` | step 2, `rp(b2)` | no — only `Pressed(b3)` | **0.0** |
| `c_D` | step 3, `sp(b2)` | no — only `Pressed(b4)`, `Grasped` | **0.0** |
| `c_E` | step 2, `rp(b2)` | yes | **1.0** |

Note `c_D` is matched through a *stick*-press — effect-based matching working across modality — and correctly denied credit: its predictions do not entail the accident at the recurrence.

Unified waste: in `c_B` every step is live (`Pressed` chain; `PickStick` supplies `Grasped` for `sp(b4)`), so `W = ∅`, waste **0.0** — the anti-signal is gone by definition, not by switch. In `c_E`, the backward pass marks the place-repick cycle: the later pick supplies `Grasped`, so the earlier pick and the `PlaceStick` dead-end (their only contributions are `Grasped`-already-resupplied and unanchored bookkeeping). `W(c_E)` = those 2 steps, both touching only `stick ∉ K` → waste **1.0**. (Which of the two picks survives is an artifact of the backward pass and immaterial.) Step-level resolution the old object-level feature structurally lacked: `c_B` and `c_E` had identical `S(c)`.

**Re-rank:** `c_B` (coverage 1, waste 0) over `c_E` (1, 1) over `c_C`/`c_D` (0, –). **Attempt 2** runs `c_B`: after step 1, `Pressed(b1)` is in its own predictions; the identical physical brush matches the prediction; refinement succeeds. FP = 1, where a static ranker pays roughly one attempt per misordered-pair it prefers — the accident made intentional.

**The record that contributes no culprits.** Suppose the pool also contained `c_F = ⟨rp(b4), …⟩` and it was tried first: every sampled approach falls short of the unreachable `b4`, and each execution deviates by `A_r = {AboveNoButton}` (the step's own delete that failed to take) and `D_r = {Pressed(b4), RobotAboveButton(robot, b4)}` (the step's own adds that never materialized). Everything is means failure: `Δ̃_r = (∅, ∅)`, no culprits, `K` unchanged. The record survives as a burned-query token — `RobotPressButtonFromNothing(robot, b4)`, exhausted — and the modality response (rank stick-press-`b4` candidates up) is the token channel's job, the division of labor claimed for it throughout this design. Under the unrestricted class-2 test, this same record would instead have handed full coverage credit to any candidate retrying the identical doomed press, and could have denied it to a mid-chain stick fix.

---

## 9. Conventions and open decision points

Chosen and load-bearing, invisible on both current environments, to be recorded in `decisions.md`: conjunctive semantics across records blaming the same culprit (discharged everywhere it was blamed, or not at all); class-2 vacuous coverage when no step of the candidate matches the context (implied by the ∀ over an empty set — the causally correct reading, since the witnessed deviation cannot recur at any recognized recurrence); waste justification held at index level for class-2 blame; the anchor-filter false positive of §5; and the empty-pool behavior (`F ≠ ∅` with `K = ∅` leaves waste's formula active — matching deployed DD2D semantics — while `F = ∅` gates both features to 0). Added in the same-day revision after review, all three no-ops on DD2D: universal objects excluded from `K` itself — **a no-op there by construction, not by measurement**, since no DD2D object appears in the argument list of every ground operator instance (`pick(o₁)` does not mention `o₂`), so `Universal = ∅` and the exclusion cannot bind; the collateral restriction `Δ̃_r` (a no-op wherever class 2 is empty, i.e., DD2D); and the per-object restriction `X|_k` (a no-op whenever the collateral deviation names a single object). The latter two are asserted by the offline regression; the first needs no probe.

## 10. Epistemic status ledger

| claim | status |
|---|---|
| Deployed DD2D coverage/waste formulas, numbers, waste-dominance (7.81 vs 10.63) | established (as_built_v3 §7) |
| SB2D exact-trace semantics; incidental presses are a real class-2 family | **established by measurement** — 59%/74% of b3/b5 records carry collateral (2026-07-31) |
| `Pressed` never deleted by any SB2D operator (class-1 monotonicity there) | established (code check) |
| Ranking-inertness lemma | by construction (§1) |
| Class-1 ⇔ deployed on terminal/absent contexts | proof holds, but **the antecedent does not**: all dd2d_v4 culprit-bearing contexts are `pick`, not terminal (2026-07-31) |
| U-old vs U-new disagreement rate ≈ 0 on dd2d_v4 | **FALSIFIED** — 7.17% of coverage pairs, 2.17% of waste candidates (2026-07-31) |
| Incidental-press frequency grows with button count | **CONFIRMED** — collateral records 59% at b3, 74% at b5 (2026-07-31) |
| Coverage/waste dominance flips across environments (waste→DD2D, coverage→SB2D) | **CONFIRMED on SB2D** — coverage-only equals coverage+waste exactly, waste-only ≈ static (2026-07-31) |
| Class-2 state test beats index tests on reversible domains | to be pinned by synthetic toggle test (no current-env evidence possible) |
| Universal-exclusion is a no-op on DD2D (`Universal = ∅`: no object is in every ground operator instance) | by construction |
| Collateral and per-object restrictions are no-ops on DD2D's observed records | expected; asserted by the same offline regression as the disagreement probe |
| Backward pass detects irrelevance but not threats; sound only for STRIPS-valid sequential plans | by construction (§5) |
| Out-of-reach robot-press failures manifest as pure means-failure class-2 records (`Δ̃_r = (∅, ∅)`) | derived from the operator schemas; confirm shape on first SB2D collection |
| Fink & Yang as the well-justified-plans citation | unverified; check before `research_lit.md` |
| Unified beats deployed on DD2D | **established** — 5.78 ± 0.10 vs 7.44 ± 0.76, −1.66 FP, CI [−2.71, −0.71], 3 seeds (2026-07-31) |
| `__wall__` is the most frequent dd2d_v4 culprit and drives the gain | **established** — 1321 mentions; mean \|K\| 1.99→1.67, spread +48%, flat contexts 8%→0% |
| Memoization is output-identical | **established** — re-scoring unchanged checkpoints is bit-identical; retrain 5.78 vs 5.83 (inside seed sd) |
