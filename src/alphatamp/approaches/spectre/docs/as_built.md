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
and the other adaptive learned baseline (LAZY) on the DD2D packing domain. It is evaluated on
**DD2D** (Drawer-Declutter 2D, a packing/retrieval domain) and **StickButton2D** (SB2D, a tool-use
button-pressing domain), with a 3-line per-environment contract (§7); a **third environment,
Restock3D** (a 3D / kinematic-PyBullet shelf-restocking domain, §7b), exercises the same contract with
a 3D point-cloud representation and now has a full comparison on a **synthetic (analytic-label)**
dataset (§10.7) — where the abstract ranker's gap over the low-level predictor is widest, read as an
upper bound pending a real-refiner audit.

*(Secondary, not the thesis: SPECTRE reads an abstract, object-centric representation rather than
low-level pixels — which is how PIGINet enters as the low-level comparator. On DD2D the abstract
ranker wins decisively; on SB2D the learned methods do not separate — §10.)*

---

## 2. The method in brief

Given a pool of `K` candidate skeletons and the set `F` of failures observed so far this episode,
SPECTRE scores every candidate and the deployed rollout tries them in descending score, updating `F`
after each failure. Three modules over a shared `d = 64` width, trained in **two stages** — a
pure-geometry static trunk first, then the failure-conditioned channel as a frozen-trunk residual (§8):

- **Φ, per-candidate encoding** — a `SceneEncoder` over the initial abstract state's objects (§4;
  each object read through a point-set descriptor — a 2D boundary ring or a 3D point cloud — plus
  its **initial-state and goal atom profiles**) and a `CandidateEncoder` over the skeleton's operator
  sequence.
- **Ψ, failure evidence** — a `RecordEncoder` turns each observed failure into a token; the culprits
  those failures blame become the `coverage`/`waste` overlap features (§6).
- **σ, scorer** — a `ResidualEvidenceScorer` around an `EvidenceCrossAttentionScorer`: a *static logit*
  from candidates cross-attending the scene tokens (frozen after stage 1) plus a *separate*, |F|-gated
  residual that cross-attends the evidence tokens with the overlap features and is added on top → one
  scalar logit per candidate (§3, §8).

**Loss is listwise Plackett–Luce**, global plus within-length buckets (`plackett_luce_loss` +
`within_length_pl_loss`) — rollout-aligned with time-to-first-success. Pointwise BCE is not used on
the ranker (it is not rollout-aligned). The within-length bucket key is `DomainSpec.length_key`
(the operator count).

---

## 3. Architecture (as implemented)

`SpectreModel(SpectreConfig)` (`model.py`) composes, over a `SpectreBatch` (`encoders.py`):

- **`SceneEncoder`** (`self.scene`) — per object `[tag embedding; point-set descriptor; pose; obj_rel;
  obj_is_goal]` → SAB×2 (§4). The point-set descriptor comes from one of two interchangeable modules,
  chosen by config (`use_pointset = use_pca_feats or use_edgeconv or use_point_sab or pma_seeds > 1`):
  the v1 **`FootprintEncoder`** (a PMA over per-point coordinate embeddings, concave-safe) when every
  switch is off, or the upgraded **`PointSetEncoder`** (deployed, §4) — `lift(C_pt→32→64)` → one DGCNN
  **EdgeConv** interaction layer → a point-set **SAB** → **multi-seed PMA** (4 seeds) →
  `Linear(64·seeds→32)`. Exactly one submodule is built, so config-off adds no `state_dict` keys and
  old checkpoints load `strict=True`. The descriptor width `D_DESCRIPTOR = 32` is frozen for both. The
  encoder is dimension-generic — the same code path serves a 2D boundary ring (`point_dim = 2`) and a
  3D surface cloud (`point_dim = 3`, Restock3D). When `atom_mode = "profiles"` the per-object atom
  profile from the `AtomProfileEncoder` is **added into** each scene token before the SABs (§4).
- **`AtomProfileEncoder`** (`self.atoms`, built when `atom_mode == "profiles"`) — turns the **initial
  abstract state** and the **goal atoms** into per-object *atom profiles*. For each object it
  scatter-**sums** `pred_emb(pred_id) + slot_emb(arg_slot)` (`D_ATOM = 32`) over every atom naming that
  object at that argument position; the init and goal profiles are computed **separately** ("true now"
  vs "wanted") and concatenated. Nullary atoms (e.g. `handempty`) pool to a **global** term added to
  the scorer's global memory token. It is order-invariant (a sum, not attention — a per-atom-token
  attention channel is the reserved-but-unbuilt `atom_mode = "tokens"`, "Rung B"). Both output
  projections are **zero-initialized**, so a freshly built `"profiles"` model is functionally identical
  to `"off"` at step 0. It is injected into the scene tokens (not a new token stream or attention
  channel), so it is complementary to `obj_is_goal` — carrying the goal's *predicate identity and
  argument roles* (`On(a,b) ≠ On(b,a)`) and the *entire initial abstract state*, which the binary flag
  discards.
- **`CandidateEncoder`** (`self.cands`) — per step `[operator embedding + learned position + projected
  argument tags]` → PMA.
- **`RecordEncoder`** (`self.records`, built when `use_records`) — one observed failure → one token,
  `Linear([schema embedding ; pooled arg-tags ; pooled culprit-tags ; scalars])`. **Role separation is
  load-bearing**: the objects the failed query was *about* (`rec_arg_tags`) and the objects observed to
  *block* it (`rec_culprit_tags`) go in different slots; pooling both into one slot would say "these
  objects are associated with this failure" without saying which was the target and which the obstacle.
  Scalars are `[j/L, log1p(effort)/10, exhausted, effort_is_total]`. Each token also carries the
  record's **state delta** `s_j − s_0` (added/deleted atoms, kept on separate role axes).
- **`ResidualEvidenceScorer`** (`self.scorer`, the deployed scorer when `residual_adaptive` is on; a
  subclass of `CrossAttentionScorer`) — computes `logit = static_logit + g(|F|)·adjustment`:
  - **static_logit** is the pure-geometry path: candidates attend over `[scene tokens; global]` and the
    head reads `[cand_emb; attended]`, with `n_overlap = 0` on the static head so it is shape-identical
    to a pure-static checkpoint. After stage 1 these submodules (`attn`/`glob_proj`/`head`) are
    **frozen** (§8).
  - **adjustment** = `adaptive_head([ev; overlap])`, where `ev` is the evidence read through a
    *separate* `evid_attn` cross-attention over the record memory (the built-but-off `compiled_agg`
    sum/max is the X1 variant, chosen by `evidence_agg`), and the `cand_overlap` scalar features (§4,
    §6) enter **here, not the static head**. Its output `Linear` is **zero-initialized**.
  - **gate** `g = sigmoid(Linear(1,16) → GELU → Linear(16,1))` over `log1p(context_size)`
    (`context_size = ((~avail) & pool_mask).sum(dim=-1)` = the number of failures in context), also
    zero-initialized → `g = 0.5` flat at init. So at step 0 `adjustment ≡ 0` and `logit ≡ static_logit`
    exactly — an *initialization* identity, not a runtime floor.

  The separate evidence channel exists because a single softmax over ~10 scene tokens and up to hundreds
  of record tokens makes evidence compete with geometry for attention mass — geometry is reliably useful
  and a jointly-trained model learned to discard evidence; two channels plus the frozen trunk remove the
  competition (§9). The base single-memory `CrossAttentionScorer` is the superclass, selected only when
  `residual_adaptive`/`evidence_attn` are off — not the deployed path.
- **`FactEncoder`** (`self.facts`) and **`AuxHead`** (`self.aux`) — both **built but not trained**: they
  are survivors of the earlier fact-based evidence path and the aux head, kept because their parameters
  are in the deployed checkpoint's `state_dict`, and left inert (the training loss is Plackett–Luce
  only; `run_training` discards the aux logits). `use_necessity` raises — necessity conditioning was cut
  (§9).

**Which components the deployed config enables** (the two-stage residual-adaptive recipe, §8):

| component | deployed? |
|---|---|
| `RecordEncoder` record tokens, aggregated per query | **yes** |
| `ResidualEvidenceScorer` (frozen static trunk + |F|-gated evidence residual) | **yes** — the deployed scorer (§3, §8) |
| observed `coverage`/`waste`/`repeat` on the width-6 `cand_overlap`; `dead`/`regroup` held at 0 | **yes** — carries the result (§4, §6, §9) |
| record `state_delta` (`s_j − s_0`) | **yes** — a tie on DD2D, deployed to complete the record schema at no porting cost (§10.5) |
| `PointSetEncoder` (`use_pca_feats` + `use_edgeconv` + `use_point_sab` + `pma_seeds = 4`) | **yes** — trained since the 2026-08-27 refresh |
| `AtomProfileEncoder` (`atom_mode = "profiles"`, init + goal atoms) | **yes** — trained since the 2026-08-27 refresh |
| `FactEncoder`, `AuxHead` | built, **untrained** |

The deployed model (PointSetEncoder + AtomProfileEncoder + the residual scorer) is larger than the
earlier 324311-parameter footprint. In the residual **stage 2**, `--freeze-static` freezes the **117
warm-started static tensors** (`scene`/`cands`/`facts`/`atoms`/`aux`/`scorer.{attn,glob_proj,head}`)
and trains only the **23 residual tensors** (`records.*`, `scorer.{evid_attn, adaptive_head, gate}`);
the optimizer filters on `requires_grad` (§8). `inference.load_checkpoint` rebuilds the model from the
saved config and reads back every architecture switch off the checkpoint (never from the caller): the
point-set switches (`scene_3d` → `point_dim`/`pose_dim`, `use_pca_feats`, `use_edgeconv`,
`use_point_sab`, `pma_seeds`, `edgeconv_k`), the atom switches (`atom_mode`, `use_init_atoms`,
`use_goal_atoms`), and the residual switches (`residual_adaptive`, `evidence_agg`) are baked into
`SpectreConfig`; the emission-only switches that change what
`dataset.build_example` *emits* (`overlap_mode`, `aggregate_records`, `coverage_feats`,
`coverage_mode`, `state_delta`) are returned alongside. Old checkpoints lacking any of these keys
resolve to all-off / 2D via the `.get` defaults, so a model is never scored on a feature it was not
trained on.

---

## 4. The input surface — domain-agnostic by design

The scene inputs were **narrowed to domain-agnostic columns** so the same encoder serves every
environment without target-specific privilege ([`decisions/07` 2026-08-08](decisions/07-stickbutton2d.md#2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target)).
Per object, `SceneEncoder` reads `[obj_tags, point-set descriptor, pose(x/scale, y/scale, θ),
obj_rel, obj_is_goal]` (plus, when `atom_mode = "profiles"`, an added atom profile — below), where:

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

**The point-set descriptor (deployed upgrade, `PointSetEncoder`).** Each object's outline is no longer
a bag of raw coordinates. The tensorizer (`compute_point_feats`) builds a per-point feature vector
`[coords ; oriented normal ; bending ; flatness]` from the point set alone — a Euclidean kNN graph
(`knn_idx`, k = 4 in 2D / 6 in 3D), a local-PCA frame giving the surface normal (sign-disambiguated
outward by an inside test — Shapely `contains` in 2D, an away-from-origin box test in 3D), 2D **signed
curvature** `κ̂ = tanh(κ·h̄)` (convex `> 0`, reflex `< 0`) or 3D **surface variation**
`λ₃/Σλ`. `C_pt` is **6 in 2D** `[x, y, nₓ, n_y, κ̂, f]` and **8 in 3D** `[x, y, z, nₓ, n_y, n_z, f, 0]`.
The `PointSetEncoder` lifts these per-point, runs one **EdgeConv** interaction layer over the fixed kNN
graph (`msg = mlp[hᵢ ; hⱼ − hᵢ]`, max-aggregated, zero-init residual), a point **SAB**, and a **4-seed
PMA** to the 32-d descriptor. Two new batch fields carry it — `point_feats (B,N,P,C_pt)` and
`knn_idx (B,N,P,k)` — both trailing-nullable (older pickles load with them `None` and take the v1
`FootprintEncoder` path). *(Rationale + the design deviations from `docs/pointset_encoder_upgrade.md` —
default `pma_seeds`, 3D `edgeconv_k = 6`, box-test orientation — are in
[`decisions/07` 2026-08-18](decisions/07-stickbutton2d.md#2026-08-18-pointsetencoder-upgrade-per-point-differential-features-edgeconv).)*

**The 3D point cloud (Restock3D).** A cube and a tall block share a 2D footprint and differ only in
**height** — the F3 axis (§7b) a footprint is blind to — so 3D objects are represented by a point
cloud, not a boundary ring. `ObjectGeometry` gained optional `point_cloud` / `pose_z` / `height` (all
`None` on the 2D envs, which therefore pickle byte-unchanged); `scene_geometry.object_point_cloud`
samples an analytic 32-point axis-aligned-box surface whose z-extent scales with the object's height.
The 3D cloud reuses the existing `obj_boundary` tensor at shape `(B,N,P,3)` (there is no separate
`point_cloud` batch field), and the encoder widens through `point_dim 2→3` / `pose_dim 3→4`, derived at
load time from the checkpoint's `scene_3d` flag. Collection always writes the 3D geometry for
Restock3D; `--scene-3d` is a **training** flag that decides whether to consume the third dimension.

**Atom profiles (deployed, `AtomProfileEncoder`).** With `atom_mode = "profiles"` the model reads the
**initial abstract state atoms** and the **goal atoms** (§3): per object, `pred_emb + slot_emb`
scatter-summed over the atoms naming it (init and goal pooled separately, `D_ATOM = 32`), added into the
scene token; nullary atoms become a global term. Four trailing-nullable batch fields carry them
(`init_atom_pred` / `init_atom_arg_tags` / `goal_atom_pred` / `goal_atom_arg_tags`), gated independently
by `use_init_atoms` / `use_goal_atoms`. This corrects the earlier design where the abstract `s₀` was not
tokenized at all: `s₀` and `g` now reach the ranker both through the per-object `obj_is_goal` flag and
through these profiles, which additionally carry predicate identity and argument-slot order.

The **`cand_overlap`** feature block is width **6** in the deployed recipe:
`[dead, jaccard, coverage, waste, repeat, regroup]` (`N_OVERLAP_COV = 4` is the base coverage width;
`--repeat-feats`/`--regroup-feats` append the last two — `dataset.py` sizes it
`n_ov = 2 + 2·want_cov + 2·want_rr`). `dead` and `jaccard` are cheap set-overlap signals against the
failed set (`dead = 0` under `--overlap-mode jaccard`); `coverage`/`waste`/`repeat` are the
evidence-grounded features of §6 (`regroup` is deprecated and held at 0). These columns feed the
scorer's residual `adjustment`, not the static head (§3). (`overlap_mode`/`coverage_mode` zero unwanted
columns rather than resize a fixed span, so the shape is stable.)

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

**Repeat** (`repeat`, the exact-step failure certificate) — a third evidence-grounded column, but a
*veto* rather than a graded score. `repeat(c) = 1` iff `c`'s skeleton contains a step whose exact
`(schema, canonical-args)` identity matches a **blameless, exhausted** failure already in context —
`not culprits and not dev_blame` (an intrinsic dead step: no collateral culprit and no deviation
witness) and `proves_failure()` (run to exhaustion, not budget-cut) — of a schema declaring
`QueryAxioms.step_certificate = True` (`dataset.py` `_rr_repeat_steps`). It encodes "this exact step was
already *proven* to fail on its own merits, so any candidate repeating it fails too." The
`step_certificate` flag is **dedicated** to this feature (`domain.py`) and independent of `proof_tier`,
so it never touches `dead`, demotion, or the record-token holdout. **Only Restock3D-v3's
`place_tall`/`place_short` declare it** (an over-tall `place_short` is a certifiable F3, §7b), so
`repeat` is **identically 0 on DD2D and SB2D** — the recipe passes `--repeat-feats` uniformly and the
feature degrades gracefully where no schema declares the certificate. Its `regroup` companion (gated by
`grouping_certificate`) is **deprecated and off**: sound on Restock3D-v3 but wrong-polarity on DD2D
(there the culprits are blockers you *want* to stage).

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

## 7b. Restock3D — the third environment (3D, in progress)

Restock3D is the third evaluation environment and the first **3D** one: a kinematic-PyBullet shelf
domain where a mobile-base arm must store floor-staged **small cubes** and **tall blocks** onto a shelf
with a **tall/bottom section** (high ceiling) and a **short/top section** (low ceiling). Feasibility is
**real PyBullet collision** — the refiner runs the actual pick/place motion planners and fails when no
collision-free solution exists; the instrumentation is observation-only, probing a rejected sample
purely to *attribute* it (§5's invariant, here `getClosestPoints` witnesses). Three failure modes make
the refinement order matter — exactly the property SPECTRE needs:

- **F3 (height mismatch):** an upright tall block placed into the *short* section collides the shelf
  board above with no movable involved — culprit-free and exhausted, so it `proves_failure()`. This is
  the F3 axis a 2D footprint cannot see, and the reason 3D objects carry a **point cloud** (§4).
- **F2 (crowding):** the held object at its resting pose collides a resident the plan already stored on
  that band; the residents are class-1 self-inflicted culprits (a candidate that over-packs a section
  fails once it runs out of free x).
- **Reach-over ("far is harder"):** the front grasp reaches **north** over anything nearer, so a
  *southern* object obstructs the diagonal approach to a *northern* one even when the northern grasp is
  reachable at its final config. The base slides laterally in a clear southern corridor for free; only
  northward reach is costly, so the naive order fails and a **south-to-north** store order clears it.

The deployed **v3** variant makes block **selection** the hard problem: each object has a **sampled
width** (`U[0.02, 0.08]`) and a **sampled height** straddling the section cutoffs (short-eligible
`≤ 0.12 m`, tall-only up to `0.17 m`), on a **re-balanced shelf** (section clearances `(0.27, 0.22)`;
the short section is no longer cube-only). Which subset of blocks goes on which level is a genuine
packing/assignment decision the abstraction cannot see. Placement is continuous — two operators
`place_tall`/`place_short` with **identical abstract effects** (`add {HandEmpty, Stored}`), the section
choice a symbolic token validated only by real collision (`place_short(tall)` → F3). Difficulty is
**4 strata by object count** `n = 6/7/8/9`. Collection labels come from an **analytic refinability
classifier** (`feasibility_v3.classify_skeleton`, pure geometry over a capacity/height/split model), so
the deployed `restock3d_v3` dataset is **synthetic** (no motion planning; wall-clock synthesized), with
a real-PyBullet refiner kept as the eval instrument and a `restock3d_v3_real` hybrid-prune collection in
progress. The full environment snapshot — geometry, generator, operators, refiner, strata, and
collection state — is [`restock3d_proposal.md`](restock3d_proposal.md). Restock3D is the 3D
representation testbed the point-cloud path and the atom profiles were built for, and the **only
environment where the `repeat` certificate (§6) is active**.

**Status: synthetic comparison in hand, real audit pending.** `restock3d_v3` is trained (SPECTRE,
PIGINet, LAZY, 3 seeds) and wired into `compare_methods.py` (§10.7). Its numbers are an **upper bound**
— a real-refiner pilot found large analytic↔real label disagreement — and the `restock3d_v3_real`
collection that would price it is still running; a VLMPlan arm is deferred.

---

## 8. Training and selection

`run_training` (`train.py`): AdamW, lr 3e-4, cosine schedule with 2 warmup epochs, 30 epochs, batch 8,
dropout 0.1, weight decay 5e-4, within-length weight 1.0, tag-permutation augmentation on.

**Deployed recipe — two-stage residual-adaptive** (`experiments/spectre/refresh_dd2d_sb2d_train.sh`).
The failure-conditioned channel is trained as a residual on a frozen static trunk, which is what makes
the failure records net-positive (§9):

- **Backbone** `BB = --use-pca-feats --use-edgeconv --use-point-sab --pma-seeds 4 --atom-mode profiles
  --select-window 5` (the PointSetEncoder upgrade + init/goal atom profiles). The class defaults for
  these switches stay off, so config-off equivalence tests and old-checkpoint loads remain valid.
- **Stage 1 — pure-geometry static trunk:** `BB --no-overlap --no-records --out-suffix _abl_static` →
  `checkpoints_spectre_norec_noov_atoms_abl_static/<env>/seed_{seed}/best.pt` (reused across refreshes
  rather than retrained each time).
- **Stage 2 — residual on the frozen trunk:** `BB --overlap-mode jaccard --coverage-feats --repeat-feats
  --aggregate-records --evidence-attn --state-delta --residual-adaptive --freeze-static
  --init-static-from <trunk>/seed_{seed}/best.pt --out-suffix _resid_full` →
  **`checkpoints_spectre_atoms_resid_full`** (one dir holds all 3 seeds; the SB2D twin runs from the
  same script). `--freeze-static` freezes the 117 warm-started static tensors and trains the 23 residual
  tensors; `{seed}` is substituted into the trunk path per seed.

**`--step-join` was dropped** from the deployed recipe (it perturbs the frozen static input, §9); the
`spectre_sweep.py --preset v3final` recipe that still carries it is the **superseded joint baseline**
(frozen in `docs/joint_refresh_snapshot.md`). Restock3D-v3 is trained by its own script
(`restock3d_v3_train.sh`) — the **jointly-trained** recipe (not the residual) with `--scene-3d
--coverage-mode both --repeat-feats --step-join` → `checkpoints_spectre_atoms_repeat` — where the
`repeat` certificate, not a residual, carries the adaptive win.

**Selection is uncensored deployed-val-FP** over the whole 100-episode val split, on a moving average
of the last **5** epochs (`--select-window 5`). The window was widened from the default 3 because the
domain-agnostic narrowing (§4) raised across-seed variance and a 3-epoch window locked onto unlucky val
epochs; ma5 recovers parity ([`decisions/07` 2026-08-09](decisions/07-stickbutton2d.md#2026-08-09-narrowed-input-variance-selector-noise-fixed-wider)).
The hard-won lesson generalizes: **a selection statistic must never be censored below the region where
the candidates differ**, and *stable curves are not evidence of a good selector*.

**EMA weight-averaging** (`--weight-avg ema`) is built and tested but **off** — it was inert on both
environments. Checkpoints land in `data/spectre/checkpoints_spectre*`; the DD2D and SB2D deployed models
are both **`checkpoints_spectre_atoms_resid_full`**, and Restock3D-v3 is `checkpoints_spectre_atoms_repeat`.

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
| **`--step-join`** (a per-step candidate×evidence join before pooling) | **cut** from the deployed DD2D/SB2D recipe 2026-08-27 | it perturbs the frozen static input the residual depends on, and was independently inert on DD2D/SB2D; retained only in the superseded joint baseline and the Restock3D-v3 script. |
| **`regroup`** (the F2 seating-chart certificate, `grouping_certificate`) | **deprecated, off** | sound on Restock3D-v3 but wrong-polarity on DD2D (culprits there are blockers you *want* to stage), and it adds nothing over `repeat`. |

**Why the failure records are trained as a frozen-trunk residual** (§3, §8). Jointly training the
record/evidence channel with the geometry trunk made `+records` *net-negative* on DD2D — it helped s1
but interfered with the shared candidate/scene representation at s2/s3 (the "W3 interference"). Freezing
the stage-1 trunk and adding the records as a zero-init |F|-gated residual removes that interference and
flips `+records` to net-positive/neutral (§10.4), with the deployed full model at least as good as the
joint model at every stratum.

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
| **SPECTRE-adaptive** | **6.42 ± 1.28** | 0.00 | 6.96 | 8.53 | 10.19 |
| SPECTRE-static | 18.35 ± 2.15 | 0.00 | 21.84 | 21.09 | 30.45 |
| PIGINet (low-level, BCE) | 17.27 ± 0.19 | 0.05 | 5.04 | 18.77 | 45.20 |
| LAZY-adaptive (Khodeir et al.) | 23.26 ± 0.50 | 0.36 | 9.59 | 24.44 | 58.65 |
| astar-dist | 34.52 | 0.00 | 2.24 | 17.08 | 118.76 |
| VLMPlan-GPT5.6 (terra, n=40) | 35.23 | 26.9 | 26.7 | 28.0 | 59.3 |
| VLMPlan-32B (local Qwen, n=40) | 23.55 | 6.76 | 5.04 | 13.16 | 69.24 |

SPECTRE-adaptive beats every comparator — the failure-blind static rankers and PIGINet, the other
adaptive learned ranker LAZY, and both VLMPlan arms. The deployed number is **6.42 ± 1.28** (3 seeds;
the residual-adaptive `checkpoints_spectre_atoms_resid_full`), superseding the jointly-trained 7.11 and
the pre-refactor 5.78 / 5.92 yardsticks; the wide seed sd reflects the residual's s1 variance, so read
the mean rather than a single seed.

### 10.2 StickButton2D — `stickbutton2d_v1_kinder` (strata b1/b2/b3/b5 = button count)

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| **SPECTRE-adaptive** | **1.98 ± 0.45** | 0.08 | 0.31 | 1.13 | 6.21 |
| SPECTRE-static | 2.22 ± 0.36 | 0.08 | 0.36 | 1.57 | 6.92 |
| PIGINet (kinder crops) | 2.28 ± 0.29 | 0.07 | 0.35 | 1.17 | 7.55 |
| LAZY-adaptive | 1.85 ± 0.02 | 0.08 | 0.36 | 2.32 | 4.63 |
| astar-dist | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |
| VLMPlan-GPT5.6 (terra, n=40) | 6.43 | 0.00 | 2.4 | 0.9 | 22.4 |
| VLMPlan-32B (local Qwen, n=40) | 13.18 | 0.70 | 1.30 | 6.20 | 44.50 |

**On SB2D the learned methods do not separate.** SPECTRE-adaptive (1.98), PIGINet (2.28) and LAZY
(1.85) sit within seed spread of one another (all ≫ astar 16.29). The *adaptive increment* within
SPECTRE is positive (static 2.22 → adaptive 1.98), as on DD2D; the *representation* advantage over the
low-level predictor is DD2D-only. The failure-information thesis holds (adaptivity helps on both); the
abstract-vs-low-level contrast does not transfer. (b1–b3 are near-anchor strata every method ties on;
the SB2D `repeat` column is inert here — §6.)

### 10.3 Generalization and held-out strata

- **Unseen shapes** (`dd2d_v4gen_shapeonly_sz07`, concave tee/cross figures): SPECTRE-adaptive
  **3.97 ± 1.04** vs PIGINet 22.68 — shape generalization is essentially free, and adaptivity does the
  lifting (SPECTRE-static 14.31).
- **Held-out stratum, DD2D** (train s0–s2, evaluate the never-trained s3, `dd2d_v4_holdout_s3`):
  SPECTRE-adaptive **5.07 ± 0.87** vs PIGINet **27.88** (SPECTRE-static 26.32) — the low-level predictor
  collapses out-of-distribution while the abstract ranker generalizes.
- **Held-out stratum, SB2D** (train b1/b2/b3, evaluate never-trained b5): SPECTRE-adaptive **1.81 ± 0.26**
  ≈ PIGINet **1.68** — the SB2D non-separation reproduces out-of-distribution.

### 10.4 What the win rests on — after-first-failure, and the residual decomposition

SPECTRE and the static rankers solve the **same** episodes on attempt 1 (on DD2D, exactly the s0
episodes; neither solves any s1–s3 episode immediately), so the first attempt separates the two methods
**not at all**. The entire margin appears *after* the first observed failure. SPECTRE is not a better
*static* ranker — it is a better *re*-ranker, which is exactly what a failure-conditioned method should
buy, and an independent corroboration of the leakage audit (a feature leaking feasibility would have
lifted the first pick too).

**The residual decomposition** (§8; Δ vs the frozen static trunk, paired bootstrap; the superseded
jointly-trained value in brackets):

| arm | DD2D ALL | Δ vs static | SB2D ALL | Δ vs static |
|---|---|---|---|---|
| static (frozen trunk) | 18.35 | — | 2.22 | — |
| + records | 17.87 | −0.48 [−1.23, +0.18] (joint +1.33) | 2.19 | −0.03 [−0.09, +0.01] (joint +0.14) |
| + scalars | 6.63 | −11.72 [−15.61, −8.30] | 1.90 | −0.32 [−0.52, −0.16] |
| **full (deployed)** | **6.42** | −11.93 [−15.79, −8.54] | **1.98** | −0.24 [−0.49, −0.01] |

Two readings: the compiled **scalars (`coverage`/`waste`) carry the bulk** of the adaptive win (−11.7
FP on DD2D), and **freezing the trunk is what makes the raw failure records net-useful** — jointly
trained, the same `+records` channel was net-*negative* (+1.33), interfering with the shared
representation at s2/s3; as a frozen-trunk residual it is net-positive-to-neutral, and the deployed full
model beats the joint model at every stratum (DD2D 6.42 < joint 7.11).

### 10.5 Wall-clock (§2b), and the state-delta tie

Under a per-candidate refinement-abandonment cap (a deployment knob; DD2D 2 s, SB2D 10 s), SPECTRE-adaptive is the **fastest** method to first success on both environments — DD2D 1.79 s ALL (vs astar
2.96), SB2D 11.17 s (vs static 12.64, PIGINet 15.15, astar 97.40). Uncapped on DD2D its wall-clock is
~equal to astar's despite 6× fewer failed attempts (its few failures are the *expensive* near-feasible
traps), which the cap targets directly. — The **record state-delta** is deployed as a *tie* on DD2D (it
completes the record schema at zero porting cost, needing no new instrumentation), not because it moved
the number.

### 10.6 Reproduce

```bash
bash experiments/spectre/refresh_dd2d_sb2d_train.sh                             # DD2D + SB2D, 3 seeds, two-stage residual
python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v4 --force --methods spectre3
SPECTRE_COMPARE_ENV=dd2d python experiments/spectre/compare_methods.py          # render the table
```

The notebook's §4 ablation grid is the **residual 4-arm decomposition** (§10.4) — static / +records /
+scalars / full — trained under the deployed definitions and read against the frozen static trunk, so
it *is* comparable to the §10.1 headline.

### 10.7 Restock3D-v3 — synthetic comparison (upper bound)

Uncensored FP, `restock3d_v3` test n=100, 3 seeds, strata by object count `n = 6/7/8/9`. **Labels are
analytic (synthetic)** — the geometric refinability classifier, no motion planning — so read this as an
*upper bound* on the representation gap, not a real-refiner result (§7b).

| method | ALL | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|
| **SPECTRE-adaptive** (`repeat`) | **3.13 ± 0.11** | 1.04 | 1.37 | 2.99 | 7.11 |
| SPECTRE-static | 12.01 ± 0.30 | — | — | — | — |
| LAZY-adaptive | 11.79 ± 0.10 | — | — | — | — |
| PIGINet (low-level) | 38.11 ± 1.23 | — | — | — | — |
| astar-dist | 38.41 | 5.48 | 14.72 | 49.64 | 83.8 |

This is where the §0 **representation crossover appears most sharply**: the low-level predictor PIGINet
(**38.11**) is no better than the naive planner order (astar **38.41**), while both abstract rankers beat
them ~3× (SPECTRE, LAZY at ~12 without the certificate). SPECTRE's further edge is entirely the
**`repeat` F3 certificate** (§6) — it drops SPECTRE-adaptive from ~12 (a tie with LAZY) to **3.13**, and
since SPECTRE-static is 12.01 the whole gap is adaptive. **⚠️ Synthetic caveat:** a real-refiner pilot
found the analytic classifier disagrees materially with real PyBullet refinement (~58% false-positive on
analytic-feasible candidates, ~13% false-negative), so a `restock3d_v3_real` collection is under way to
price how much of the gap survives; no real comparison exists yet. Restock3D is deployed with the
jointly-trained `--scene-3d --atom-mode profiles --repeat-feats` recipe
(`checkpoints_spectre_atoms_repeat`), not the DD2D/SB2D residual.

---

## 11. Known limitations

1. **The representation advantage is DD2D-only.** On SB2D the learned methods (SPECTRE, PIGINet, LAZY)
   do not separate (§10.2); the abstract-vs-low-level contrast does not transfer, even though the
   failure-conditioned adaptive increment is positive on both. The generality claim for SB2D rests on
   adaptivity, not representation.
2. **The headline is 3 seeds** — the count every comparator has. The DD2D deployed sd (**± 1.28**) is
   wide, driven by the residual's s1 variance, so the mean rather than any single seed is the result;
   the deployed 6.42 supersedes the jointly-trained 7.11 and the pre-refactor 5.78 / 5.92 yardsticks.
3. **The state delta is deployed on a tie** (§10.5) — it completes the record schema at zero porting
   cost, not because it improved DD2D FP.
4. **SB2D's b5 train split is small** (17 episodes, a wall-clock-budget cut), so the b5 column is
   substantially a generalization number rather than a like-for-like stratum result.
5. **DD2D generation is `PYTHONHASHSEED`-dependent**, so no collection is bit-reproducible across
   processes.
6. **Restock3D-v3's numbers are synthetic** (analytic labels, §7b, §10.7) — an upper bound on the
   representation gap, not a real-refiner result. A real-refiner pilot found the analytic classifier and
   real PyBullet refinement disagree materially (~58% FP / ~13% FN), and the `restock3d_v3_real` audit
   collection that would price how much of the gap survives is still in progress.
