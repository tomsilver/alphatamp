# SPECTRE — As Built

What SPECTRE *is*, as implemented, with the evidence for each choice. This document describes the
**code** (the authoritative source); where a design note or the living [`proposal.md`](proposal.md)
disagrees, the code wins. Numbers are the current [`compare_methods.py`](../../../../experiments/spectre/compare_methods.py)
caches; decisions cite [`decisions/`](decisions/README.md), runs cite [`notebook/`](notebook/README.md).

*(Historical note: SPECTRE was built iteratively (v1 → v2.2 → v3) and unified into one method in the
2026-08-12 publication refactor. The "vN" comparisons that motivated individual choices live in the
[`decisions/`](decisions/README.md) log and the frozen [`docs/archive/`](archive/); this document
describes only the current, single method.)*

---

## 1. The problem, and the thesis

Bilevel TAMP planners enumerate a pool `S = {s₁ … s_K}` of goal-reaching abstract skeletons, then
refine them one at a time until one succeeds. On hard problems the **order of refinement attempts
dominates wall-clock**, so the value is in a good re-ranker.

**The thesis is failure-information utilization.** Every refinement failure observed *within an
episode* carries instance-specific information about which remaining skeletons to try next — which
object blocked, which query burned, how deep the refinement got. Existing re-rankers underuse it:

- static historical rankers and the default planner order (astar/FF) fix the order before the first
  attempt and never update — they ignore within-episode failures entirely;
- **PIGINet** scores each skeleton once from the concrete low-level initial state — a *static*
  feasibility predictor, blind to the failures that accrue during the episode;
- **VLMPlan** proposes plans zero-shot from a scene image — no failure feedback at all;
- even the other *learned adaptive* baseline, **LAZY** (Khodeir et al.), consumes failures only as a
  scalar online feasibility statistic.

SPECTRE turns **each observed failure into structured evidence** and re-scores the whole pool against
the accumulated failure context on every attempt: the *culprits* a failure blamed become
per-candidate **coverage/waste** features (§6); each failure becomes a **record token** carrying its
burned query and arguments (§4–§5); and each record carries the abstract-state delta at its failing
step. The claim is that this *stronger utilization* of failure information — not a new representation
per se — is what buys the ordering.

**The results are literal about this** (§10.4): SPECTRE and the static rankers solve the *same*
episodes on the first attempt (the first attempt separates them not at all); the entire margin
appears **after the first observed failure**. SPECTRE beats both the failure-blind static baselines
and the other adaptive learned baseline (LAZY) on the DD2D packing domain. It is evaluated on two
environments — **DD2D** (Drawer-Declutter 2D, a packing/retrieval domain) and **StickButton2D**
(SB2D, a tool-use button-pressing domain) — with a 3-line per-environment contract (§7).

*(Secondary, not the thesis: SPECTRE reads an abstract, object-centric representation rather than
low-level pixels — which is how PIGINet enters as the low-level comparator. On DD2D the abstract
ranker wins decisively; on SB2D the learned methods do not separate — §10.)*

---

## 2. The method in brief

Given a pool of `K` candidate skeletons and the set `F` of failures observed so far this episode,
SPECTRE scores every candidate and the deployed rollout tries them in descending score, updating `F`
after each failure. Three modules, trained jointly (d = 64 throughout):

- **Φ, per-candidate encoding** — a `SceneEncoder` over the initial abstract state's objects (§4) and
  a `CandidateEncoder` over the skeleton's operator sequence.
- **Ψ, failure evidence** — a `RecordEncoder` turns each observed failure into a token; the culprits
  those failures blame become the `coverage`/`waste` overlap features (§6).
- **σ, scorer** — an `EvidenceCrossAttentionScorer`: candidates cross-attend over the scene tokens and,
  in a *separate* attention channel, over the evidence tokens, with the overlap features concatenated
  at the head → one scalar logit per candidate.

**Loss is listwise Plackett–Luce**, global plus within-length buckets (`plackett_luce_loss` +
`within_length_pl_loss`) — rollout-aligned with time-to-first-success. Pointwise BCE is not used on
the ranker (it is not rollout-aligned). The within-length bucket key is `DomainSpec.length_key`
(the operator count).

---

## 3. Architecture (as implemented)

`SpectreModel(SpectreConfig)` (`model.py`) composes, over a `SpectreBatch` (`encoders.py`):

- **`SceneEncoder`** (`self.scene`) — per object `[tag embedding; 32-point boundary descriptor pooled
  by PMA; pose; obj_rel; obj_is_goal]` → SAB×2 (§4). Point-set boundary encoding (`FootprintEncoder`)
  is concave-safe.
- **`CandidateEncoder`** (`self.cands`) — per step `[operator embedding + learned position + projected
  argument tags]` → PMA.
- **`RecordEncoder`** (`self.records`, built when `use_records`) — one observed failure → one token,
  `Linear([schema embedding ; pooled arg-tags ; pooled culprit-tags ; scalars])`. **Role separation is
  load-bearing**: the objects the failed query was *about* (`rec_arg_tags`) and the objects observed to
  *block* it (`rec_culprit_tags`) go in different slots; pooling both into one slot would say "these
  objects are associated with this failure" without saying which was the target and which the obstacle.
  Scalars are `[j/L, log1p(effort)/10, exhausted, effort_is_total]`. Each token also carries the
  record's **state delta** `s_j − s_0` (added/deleted atoms, kept on separate role axes).
- **`EvidenceCrossAttentionScorer`** (`self.scorer`) — candidates attend over scene tokens and, in a
  *second* `MultiheadAttention` channel, over the evidence memory, with `cand_overlap` features
  concatenated at the head. The separate channel is deployed because a single softmax over ~10 scene
  tokens and up to hundreds of record tokens makes evidence compete with geometry for attention mass —
  geometry is reliably useful and the model learned to discard evidence; two channels remove the
  competition (§9). (The base single-memory `CrossAttentionScorer` remains as its superclass, selected
  only when `evidence_attn` is off — not the deployed path.)
- **`FactEncoder`** (`self.facts`) and **`AuxHead`** (`self.aux`) — both **built but not trained**: they
  are survivors of the earlier fact-based evidence path and the aux head, kept because their parameters
  are in the deployed checkpoint's `state_dict`, and left inert (the training loss is Plackett–Luce
  only; `run_training` discards the aux logits). `use_necessity` raises — necessity conditioning was cut
  (§9).

**Which components the deployed config enables** (the `v3final` recipe, §8):

| component | deployed? |
|---|---|
| `RecordEncoder` record tokens, aggregated per query | **yes** |
| `EvidenceCrossAttentionScorer` (separate evidence channel) | **yes** |
| observed `coverage`/`waste` on `cand_overlap`; `dead` dropped from the net | **yes** — carries the result (§6, §9) |
| record `state_delta` (`s_j − s_0`) | **yes** — a tie on DD2D, deployed to complete the record schema at no porting cost (§10.5) |
| `FactEncoder`, `AuxHead` | built, **untrained** |

The deployed model is **324311 parameters**; `inference.load_checkpoint` rebuilds it and returns the
deploy-time switches that change what `dataset.build_example` *emits* (`overlap_mode`,
`aggregate_records`, `coverage_feats`, `coverage_mode`, `state_delta`) — read off the checkpoint, never
from the caller, so a model is never scored on a feature it was not trained on.

---

## 4. The input surface — domain-agnostic by design

The scene inputs were **narrowed to domain-agnostic columns** so the same encoder serves both
environments without target-specific privilege ([`decisions/07` 2026-08-08](decisions/07-stickbutton2d.md#2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target)).
Per object, `SceneEncoder` reads `[obj_tags, boundary(32-pt ring), pose(x/scale, y/scale, θ),
obj_rel, obj_is_goal]`, where:

- **`obj_rel` = the width-3 anchor-free triple `[area, sin θ, cos θ]`** (`D_REL = 3`), not a
  target-anchored vector. The DD2D-specific offsets **`(dx, dy, distance-to-target, area-ratio-to-target)`**
  and the privileged **`concave`** flag were **dropped** — they are meaningless on SB2D (there is no
  single "target" object; a button is not "near the drawer target"). Absolute position survives in
  `pose`.
- **`obj_is_goal`** (1.0 for any object named in the goal literals) **replaces** the old single-target
  `is_target`.

An inference-time probe priced the removal at **Δ 0.00 FP** on both deployed models — the dropped
columns were inference-inert, not information the ranker used. The narrowing raised across-seed
variance, recovered by widening the selection window (§8).

The **`cand_overlap`** feature block is width **4**: `[dead, jaccard, coverage, waste]` (`N_OVERLAP_COV`).
`dead` and `jaccard` are cheap set-overlap signals against the failed set; `coverage`/`waste` are the
evidence-grounded features of §6. (`overlap_mode`/`coverage_mode` zero unwanted columns rather than
resize, so the tensor shape is fixed.)

---

## 5. Failures as observations

Instrumentation is **observation-only** — an invariant. `n_attempts` *is* `counter.calls`, so one
extra stream call would shift every label; culprit extraction therefore reads answers the refiner
produced anyway (e.g. DD2D's `grasp_cfree` was refactored to expose the blocker witness the
short-circuit already computed). Verified differentially: label, steps-bound, plan-length,
failure-action identical on 290/290 replayed candidates.

Each failure is one `FailureRecord` (`failure_record.py`):
`(candidate_idx, step_index, schema, args, culprits, unmoved, n_step, exhausted, budget_exhausted,
effort_is_total, instrumented, dev_blame, state_delta)`. The two evidence channels:

- **`culprits`** — **class-1** blame: the objects a validity check (collision, bounds) named when it
  rejected a sample before a successor state existed. DD2D's grasp collisions are class 1.
- **`dev_blame`** — **class-2** blame: the objects named by the *collateral* effect-mismatch deviation,
  for environments (like SB2D) whose refiner returns only a bool from its collision check and so has no
  class-1 channel. SB2D's incidental button presses are class 2.

`state_delta` is `s_j − s_0` by STRIPS progression over the candidate's own plan — pure abstract
bookkeeping, no new instrumentation; `None` (not computed) is distinguished from `StateDelta()`
(computed-but-empty). `proves_failure()` = `exhausted and not budget_exhausted` (a query that ran to
exhaustion, not one cut off by the attempt budget). `effort_is_total` distinguishes backfilled
whole-attempt effort from per-step instrumented effort, so a re-collection cannot silently redefine a
scalar.

**Aggregation.** The refiner emits one record per failed *sample*; the deployed config aggregates to
one per `(schema, args)` (`--aggregate-records`) — deepest step, summed effort, unioned culprits —
because a candidate whose `place-buffer(o)` was retried across many poses would otherwise contribute
hundreds of near-identical tokens (up to ~2045 at `|F|=30`). Aggregation is −88.7% tokens with
nothing the token *encodes* lost.

---

## 6. Coverage and waste — evidence-grounded, from the operator schema alone

Two per-candidate scalars appended to `cand_overlap`, computed only from failures the refiner already
reported by `unified_evidence.coverage_and_waste(...)` — the **sole** coverage/waste path (the old
`coverage = |S(c) ∩ culprits| / |culprits|`, `waste = |S(c) \ culprits| / |S(c)|` with
`S(c) = args \ goal_objects` was **removed** in the 2026-08-12 refactor; there is no toggle). Both are
exactly `0` when `F = ∅` (`if not records or not pool: return 0.0`) — the leakage invariant, so the
first attempt is purely static and the signal accrues as the rollout proceeds.

**Notation.** A candidate `c = ⟨s_1 … s_L⟩`; `touch(c, k)` = the step indices whose add/delete effects
mention object `k`; `ŝ_j(c)` = the candidate's predicted abstract state *before* step `s_j` by STRIPS
progression (`predicted_states`, the same machinery `state_delta` runs).

**Two domain-only filters** (`scene_filters`, from the grounded operator set — no environment named):
an object is **universal** iff it appears in the argument list of *every* ground operator instance
(behaviourally the robot); an atom is **anchored** iff it mentions a non-universal object; an object is
**actionable** iff some operator's effects mention it. On DD2D `universal = ∅`; on SB2D it is the robot.

**Culprits** (`blame`, `culprit_pool`). A failed sample dies in one of two classes, each of which
already named the objects to blame:
- **Class 1:** `blame(r)` = the objects the violated check named (`culprits`).
- **Class 2:** the raw deviation is `(added, deleted)`; blame reads the **collateral** deviation — the
  raw deviation minus the failed step's *own* declared effects, `collateral(r) = (added − A⁻(s_r),
  deleted − A⁺(s_r))` — and `blame(r)` = the objects those collateral atoms mention. The stripped
  remainder (the step's own effects that failed to take) is **means failure** — it names no culprit and
  instead rides the burned-query record token. So an out-of-reach robot-press blames nobody
  (`collateral = (∅, ∅)`). *In one line: culprits are collateral damage; burned queries are failed
  means.*

The **culprit pool** is `K = (Actionable \ Universal) ∩ ⋃_{r∈F} blame(r)`. Universal objects are
excluded from `K` itself — not merely from anchoring — because waste's justification is a per-step
existential over `K`, and one universal object (the robot, touched by every step) would spuriously
justify every superfluous step.

**Context matching** (`matched_steps`). A step `s_j` of `c` *matches* record `r` iff its effects
intersect the anchored, *signed* effect signature of `r`'s failed step (`A⁺(s_j) ∩ anch(A⁺(s_r))` or
`A⁻(s_j) ∩ anch(A⁻(s_r))` non-empty). Matching is by what a step *does*, not what it is named — a
stick-press of `b2` re-enters the context of a failed robot-press of `b2` — with sign respected
(adding what the failed step deleted is the opposite of a re-attempt). `M_c(r)` = the matched indices.

**Coverage** (`covered`, `coverage`) — recall over the failures' stories, **conjunctive** across every
record blaming `k`, and **class-dependent** (the abstraction can state a class-2 hazard as a predicate
but not a class-1 "blockedness"):
- *Class-1 test (index precedence):* if `M_c(r) ≠ ∅`, some `j ∈ touch(c, k)` with `j < min M_c(r)`
  (deal with `k` before the earliest recurrence); if `M_c(r) = ∅`, bare membership `touch(c, k) ≠ ∅`.
- *Class-2 test (state entailment):* for every `j ∈ M_c(r)`, restricting the collateral deviation to
  atoms mentioning `k`, the needed adds must hold and the forbidden deletes must be absent in `ŝ_j(c)`
  (`need ⊆ ŝ_j` and `forbid ∩ ŝ_j = ∅`) — replaying the recorded accident against the candidate's own
  predictions is a no-op wherever the situation recurs (the accident has been made intentional).
- `coverage(c) = |{ k ∈ K : covered(k | c) }| / |K|`.

**Waste** (`superfluous_steps`, `_justified`, `waste`) — precision over unexplained work:
- *Superfluous steps (backward relevance):* initialise `needed ← anch(goal)`, walk `c` back-to-front; a
  step is *live* iff it adds something needed (then `needed ← (needed − adds) ∪ anch(pre)`), else
  *superfluous*. The live causal spine — including tool acquisition, `PickStick` feeding `Grasped` —
  never enters the denominator, which is what dissolves SB2D's tool anti-signal with *no per-environment
  switch*. The pass never consults deletes (detects irrelevance, not threats) — sound because candidates
  are STRIPS-valid sequential plans.
- *Justified:* a superfluous step is justified iff it touches some `k ∈ K` at an index before every
  matched step of every record blaming `k`.
- `waste(c) = |{ j superfluous : ¬justified(j) }| / |{ superfluous }|`, and **abstains** (returns 0) on
  an empty culprit pool — otherwise every candidate would read 1.0 from zero evidence.

**Intuition.** On DD2D a grasp collision names a blocker (class 1); a candidate that stages the blocker
before extracting is covered, and staging an *unnamed* object is waste. On SB2D an incidental press
names a button (class 2); a candidate whose own predictions already contain that press at the
recurrence is covered, and a stick-fiddle that feeds no goal atom is waste. Nothing in these definitions
names a drawer, a button, or a stick — the per-domain input is the operator schemas themselves.

---

## 7. The domain contract — the whole per-environment surface

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
`manipulated` as `args(σ) \ goal_objects`, `length_key` as the operator count. Declaring an axiom has
the epistemic status of writing the PDDL domain file — specification, not a learned routine. `spec_for`
maps an env-variant to its spec; **`EMPTY_SPEC`** (declares nothing) is the fallback, and **is what SB2D
uses** — SB2D registers no axioms, so its per-environment code is entirely the converter + refiner
instrumentation under `envs/stickbutton2d/`.

Since the proof-tier demotion was cut (§9), **nothing at deploy time consumes a proof**, so the
`QueryAxioms` are *optional*: their only remaining effect is (a) which failures set the `dead` overlap
column and (b) which proof-tier records are held out of the learned token stream (1.3% of dd2d_v4
records, all `retrieve`). "Learning is the floor" is the deployed configuration, not a fallback — an
`EMPTY_SPEC` environment (like SB2D) runs the full learned method.

---

## 8. Training and selection

`run_training` (`train.py`): AdamW, lr 3e-4, cosine schedule with 2 warmup epochs, 30 epochs, batch 8,
dropout 0.1, weight decay 5e-4, within-length weight 1.0, tag-permutation augmentation on.

**Deployed recipe** (`spectre_sweep.py --preset v3final`): `--overlap-mode jaccard --coverage-feats
--aggregate-records --evidence-attn --state-delta --select-window 5`.

**Selection is uncensored deployed-val-FP** over the whole 100-episode val split, on a moving average
of the last **5** epochs (`--select-window 5`). The window was widened from the default 3 because the
domain-agnostic narrowing (§4) raised across-seed variance and a 3-epoch window locked onto unlucky val
epochs; ma5 recovers parity ([`decisions/07` 2026-08-09](decisions/07-stickbutton2d.md#2026-08-09-narrowed-input-variance-selector-noise-fixed-wider)).
The hard-won lesson generalizes: **a selection statistic must never be censored below the region where
the candidates differ**, and *stable curves are not evidence of a good selector*.

**EMA weight-averaging** (`--weight-avg ema`) is built and tested but **off** — it was inert on both
environments. Checkpoints land in `data/spectre/checkpoints_spectre*` (the DD2D deployed model is
`checkpoints_spectre_unified`, SB2D is `checkpoints_spectre`).

---

## 9. What was removed, and why

Each of these was built, measured, and cut; the reasoning is the record of why the deployed method is
one learned mechanism rather than a stack of parts.

| component | disposition | evidence |
|---|---|---|
| **proof-tier demotion** (an external, hand-declared score offset on provably-dead candidates) | **cut** 2026-07-30; machinery removed | costs **0.23 FP** (7.20 → 7.44 at the time), fired on only **6%** of deployed rollouts (vs 55% on a stripped floor arm), and the learned features absorbed ~79% of its value — so on this domain it barely acts, and cutting it makes the deployed system *one mechanism* (model scores only, pure argmax). It was *sound* (0 demoted-but-feasible); on a domain whose proofs fire far more often the trade reverses. |
| **legacy coverage/waste** (`S(c) = args \ goal_objects`) | **removed** 2026-08-12 | the unified definitions (§6) are the only path; they are domain-agnostic where the legacy denominator hard-coded "discretionary = touches a non-goal object," true on DD2D and false wherever tools exist. |
| **per-object obj-evidence** (a scene-token evidence summary) | **removed** | hurt s1 badly on its own; evidence enters through the record tokens + separate attention channel instead. |
| **sinusoidal positions** (`CandidateEncoderV3`) | **removed** | the motivating OOV never occurs on DD2D (s0–s2 pools already contain longer plans than s3 needs); it was future-proofing, not a fix. |
| **`tail_max_f`** (`|F|` sampling out to ~40) | **removed** | an unused training knob. |
| **necessity head** (proposal §5.1, a predicted per-object `p_i`) | **cut**; `use_necessity` raises | the s2 deficit it targeted is *within-length*, which it does not address — and once the refiner *reports* culprits the same two features (coverage/waste) need no predicted head at all. |

The single substitution the win rests on: §5.1 wanted a per-object necessity `p_i` *predicted* by a
head; SPECTRE gets the same two candidate features from culprits the refiner *reported* — no head, no
second loss, no geometry routine, and strictly more C2-legal (nothing is inferred by us). A leakage
audit returned 0 violations (features zero at `|F|=0`; culprits only from candidates already in the
failure context; the deploy loop breaks on success before a successful candidate can enter the context).

---

## 10. Results

Uncensored deployed FP, test n=100. Lower is better; mean ± across-seed std (3 seeds for the learned
methods; a single deterministic run — no ± — for astar and VLMPlan). Numbers are the current
`compare_methods.py` caches; render any environment with `SPECTRE_COMPARE_ENV=<key> python
experiments/spectre/compare_methods.py`.

### 10.1 DD2D — `dd2d_v4` (strata s0…s3 = size of the minimum feasible subset)

| method | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| **SPECTRE-adaptive** | **6.29 ± 0.31** | 0.00 | 6.67 ± 1.67 | 9.75 ± 0.21 | 8.73 ± 1.01 |
| SPECTRE-static | 19.80 ± 2.12 | 0.00 | 25.64 | 20.39 | 33.16 |
| PIGINet (low-level, BCE) | 17.27 ± 0.19 | 0.05 | 5.04 | 18.77 | 45.20 |
| LAZY-adaptive (Khodeir et al.) | 23.26 ± 0.50 | 0.36 | 9.59 | 24.44 | 58.65 |
| astar-dist | 34.52 | 0.00 | 2.24 | 17.08 | 118.76 |
| VLMPlan-GPT5.6 (terra, n=40) | 35.23 | 26.9 | 26.7 | 28.0 | 59.3 |
| VLMPlan-32B (local Qwen, n=40) | 23.55 | 6.76 | 5.04 | 13.16 | 69.24 |

SPECTRE-adaptive beats every comparator — the failure-blind static rankers and PIGINet, the other
adaptive learned ranker LAZY, and both VLMPlan arms. *(The deployed number reads **6.29 ± 0.31** in the
current caches, retrained under the 2026-08-12 refactor code; it ties the frozen yardsticks **5.92**
(domain-agnostic) / **5.78** (target-anchored) within ~1.3 seed-sd — the small delta is fresh-run GPU
non-determinism, confirmed by a byte-identical same-checkpoint rollout at 324311 params.)*

### 10.2 StickButton2D — `stickbutton2d_v1_kinder` (strata b1/b2/b3/b5 = button count)

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| **SPECTRE-adaptive** | **1.75 ± 0.19** | 0.08 | 0.31 | 1.13 | 5.49 |
| SPECTRE-static | 2.06 ± 0.21 | 0.08 | 0.36 | 1.57 | 6.21 |
| PIGINet (kinder crops) | 2.28 ± 0.29 | 0.07 | 0.35 | 1.17 | 7.55 |
| LAZY-adaptive | 1.85 ± 0.02 | 0.08 | 0.36 | 2.32 | 4.63 |
| astar-dist | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |
| VLMPlan-GPT5.6 (terra, n=40) | 6.43 | 0.00 | 2.4 | 0.9 | 22.4 |
| VLMPlan-32B (local Qwen, n=40) | 13.18 | 0.70 | 1.30 | 6.20 | 44.50 |

**On SB2D the learned methods do not separate.** Paired bootstrap: SPECTRE-adaptive − PIGINet =
−0.60, CI [−1.24, +0.08]; SPECTRE-adaptive − LAZY = −0.01, CI [−0.72, +0.72]. The *adaptive increment*
within SPECTRE is positive on both environments; the *representation* advantage over the low-level
predictor is DD2D-only. The failure-information thesis holds (adaptivity helps on both); the
abstract-vs-low-level contrast does not transfer.

### 10.3 Generalization and held-out strata

- **Unseen shapes** (`dd2d_v4gen_shapeonly_sz07`, concave tee/cross figures): SPECTRE-adaptive
  **2.79 ± 0.36** vs PIGINet 22.68 — shape generalization is essentially free, and adaptivity does the
  lifting (SPECTRE-static 15.00).
- **Held-out stratum, DD2D** (train s0–s2, evaluate the never-trained s3): SPECTRE-adaptive s3 **9.97**
  ≈ its in-distribution s3, while PIGINet s3 **85.89** (~9× worse) — the low-level predictor collapses
  out-of-distribution while the abstract ranker generalizes.
- **Held-out stratum, SB2D** (train b1/b2/b3, evaluate never-trained b5): PIGINet **5.36** ≈ SPECTRE
  **6.87** — the SB2D non-separation reproduces out-of-distribution.

### 10.4 What the win rests on — the after-first-failure decomposition

SPECTRE and the static rankers solve the **same** episodes on attempt 1 (on DD2D, exactly the s0
episodes; neither solves any s1–s3 episode immediately), so the first attempt separates the two methods
**not at all**. The entire margin appears *after* the first observed failure. SPECTRE is not a better
*static* ranker — it is a better *re*-ranker, which is exactly what a failure-conditioned method should
buy, and an independent corroboration of the leakage audit (a feature leaking feasibility would have
lifted the first pick too).

### 10.5 Wall-clock (§2b), and the state-delta tie

Under a per-candidate refinement-abandonment cap (a deployment knob; DD2D 2 s, SB2D 10 s), SPECTRE-adaptive is the **fastest** method to first success on both environments — DD2D 1.79 s ALL (vs astar
2.96), SB2D 11.17 s (vs static 12.64, PIGINet 15.15, astar 97.40). Uncapped on DD2D its wall-clock is
~equal to astar's despite 6× fewer failed attempts (its few failures are the *expensive* near-feasible
traps), which the cap targets directly. — The **record state-delta** is deployed as a *tie* on DD2D (it
completes the record schema at zero porting cost, needing no new instrumentation), not because it moved
the number.

### 10.6 Reproduce

```bash
python experiments/spectre/spectre_sweep.py --preset v3final --seeds 0 1 2      # DD2D, 3 seeds
bash experiments/spectre/sb2d_finalize.sh                                       # SB2D pipeline
python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v4 --force --methods spectre3 --no-ablations
SPECTRE_COMPARE_ENV=dd2d python experiments/spectre/compare_methods.py          # render the table
```

*Caveat on the notebook's §4 ablation grid (DD2D only):* those arms were trained under the
**pre-2026-07-31** coverage/waste definition and are read internally against their own floor — they are
matched-settings and not comparable to the §10.1 headline (which uses the unified definition). Do not
cross-quote them.

---

## 11. Known limitations

1. **The representation advantage is DD2D-only.** On SB2D the learned methods (SPECTRE, PIGINet, LAZY)
   do not separate (§10.2); the abstract-vs-low-level contrast does not transfer, even though the
   failure-conditioned adaptive increment is positive on both. The generality claim for SB2D rests on
   adaptivity, not representation.
2. **The headline is 3 seeds.** It is the count every *method* has (the comparators were trained at 3);
   SPECTRE itself has 6 trained seeds, over which the DD2D deployed model reads ~8.5 rather than 6.3.
   The current cache retrain reads **6.29 ± 0.31** vs the frozen **5.92** yardstick — within ~1.3
   seed-sd; the definitive behavior check is the byte-identical same-checkpoint rollout, not the retrain
   mean.
3. **The state delta is deployed on a tie** (§10.5) — it completes the record schema at zero porting
   cost, not because it improved DD2D FP.
4. **SB2D's b5 train split is small** (17 episodes, a wall-clock-budget cut), so the b5 column is
   substantially a generalization number rather than a like-for-like stratum result.
5. **DD2D generation is `PYTHONHASHSEED`-dependent**, so no collection is bit-reproducible across
   processes.
