# Restock3D — Implementation Proposal (v0.1, 2026-08-13)

The third SPECTRE evaluation environment: 3D, TidyBot-compatible, built on the existing Shelf3D /
KinDER infrastructure, designed to be hard for the baselines (astar, PIGINet, VLMPlan, LAZY) while
feeding SPECTRE's evidence channels — and deployable on the physical lab shelf without altering its
exterior dimensions.

**Epistemic conventions** (per the project ledger discipline). Every load-bearing claim below is
tagged:
- **[E]** established — read from code/docs/assets or already demonstrated in operation;
- **[D]** derived — follows from [E] facts by construction (e.g., the coverage-polarity analysis);
- **[P]** registered prediction — falsifiable, with the probe that tests it named;
- **[?]** unverified — an estimate or an assumption a probe must confirm before anything depends
  on it.

This document is deliberately **stage-gated**: no phase begins until its gate's probes pass, and
each gate names its fallback. The design is a hypothesis, not a commitment.

> **Build status (2026-08-15, autonomous).** The kinematic **no-clutter v1** (F2 over-assignment +
> F3 tall-into-short; F1/clutter deferred) is built and Stage-0 approved. Since then: the
> **eager-validity heuristic** (`astar_eager`, `envs/restock3d/eager_search.py` +
> `eager_tables.py`), the **oracle solver** (`envs/restock3d/oracle.py`), and a real per-candidate
> cap (`refine_cap.py`) are built; per-stratum **timeout** and **K_max** are calibrated. Numbers +
> the eager-vs-plain pool-order finding:
> [`decisions/07` 2026-08-15](decisions/07-stickbutton2d.md#2026-08-15-restock3d-eager-validity-heuristic-oracle-solver-budget)
> / [`notebook/07` 2026-08-15](notebook/07-stickbutton2d.md#2026-08-15-restock3d-eager-heuristic-oracle-calibration-timeout).
> The guide docs `restock3d_eager_heuristic_guide.md` / `restock3d_oracle_solver.md` describe the
> full-clutter design; v1 realised their F2+F3 subset (regions single-object, `blockers=∅`, Pick
> penalty inert, no relocation phase).
> **F1 clutter re-added 2026-08-15** ([`decisions/07`](decisions/07-stickbutton2d.md#2026-08-15-restock3d-f1-clutter-re-added-relocation-buffer)):
> movable clutter beside a cube goal (+y, gap ~0.07 m) obstructs its top-down grasp, relocated via a
> `PlaceBuffer`→`OnBuffer` floor buffer; the eager table gets a T5 penalty + order-aware feasibility and
> the oracle a relocation phase. Deployed on **r1 only** — F1 composes with r1's F2 but the F1+F3
> relocate-first search on r3 does not enumerate within budget (oracle certifies it, no planner surfaces
> it), so r3 stays F2+F3. Base collision is best-effort (planner-level, wide base vs dense floor);
> coverage/waste verified non-degenerate on F1 (env-agnostic). Still deferred: the full relocation-aware
> **collection + training**, **r3 F1** (relocation-aware pool generator), step-time base enforcement
> (navigable floor), learned baselines, the `compare_envs` EnvSpec.

---

## 1. Background and rationale

### 1.1 Why a third environment

SPECTRE's thesis is **failure-information utilization**: each within-episode refinement failure is
turned into structured evidence (record tokens, culprit-derived coverage/waste, jaccard/dead
overlaps) and the skeleton pool is re-scored against the accumulated failure context on every
attempt. On DD2D and SB2D the adaptive increment is positive on both, and the entire margin over
static rankers appears after the first observed failure [E, as_built §10.4].

The third environment must do three jobs the current pair cannot:

1. **3D + real robot.** A MuJoCo task matched to the physical TidyBot, closing the
   real → sim → real story the paper needs.
2. **A distinct structural stress.** Per the environment-selection criterion settled earlier:
   *reuse the existing failure classes, but stress a structural property neither current
   environment exercises.* The property chosen here is **self-inflicted culprits** — failures whose
   blamed objects are in a hazardous configuration *because the plan itself put them there* —
   which forces order-awareness and gives the record `state_delta` (deployed on a tie in 2D
   [E, as_built §10.5]) a chance to be load-bearing.
3. **A baseline↔oracle gap concentrated at the abstract level.** The task planner must produce
   many goal-reaching skeletons that fail refinement for reasons an oracle with geometric
   knowledge avoids — a large task space, not a hard sampling problem.

### 1.2 Two designs already rejected, and why

**Shelf-pick declutter** (blockers start on shelf goal regions; move them, then place targets).
Requires picking *from* the shelf. The pick skills are top-down / ground-style with an
`OnGround` precondition [E, capabilities §2/§3]; a front-facing grasp was attempted and abandoned
after substantial unfruitful engineering. Rejected on cost. (Its *spirit* survives as an optional
lever — §4.6 — with zero grasp engineering.)

**Pure height/capacity twist** (short/tall cells, short/tall objects, nothing else). This is a
fine hardness mechanism but **starves coverage and waste**, for two separable reasons, both
checkable against `as_built.md` §6 [D]:

- *Culprit degeneracy.* A `Place(tall, short_cell)` failure's collision witness is the shelf —
  one monolithic pose-only body in both implementations [E]. Blaming it is formally admissible
  (the shelf appears in Place's add effect, so it passes the actionable filter) but carries no
  discrimination: `touch(c, shelf)` is every Place step of every candidate, so class-1 precedence
  saturates. Routed instead as a **culprit-free exhausted record** ("burned queries are failed
  means" [E]) the failure is legitimate evidence — but for the record/jaccard/dead channels only,
  never coverage/waste.
- *Empty waste denominator.* With a goal of "every object stored," backward relevance marks every
  pick and place live; the superfluous set is empty and waste reads 0 on every candidate, always.
  Waste — the stronger of the two features on DD2D [E] — would be structurally dead.

The fix is compositional, not a feature redefinition: add **movable culprits** (pick-side clutter)
and **discretionary steps** (relocations whose necessity is instance-dependent).

---

## 2. The environment

### 2.1 Scene and task

A floor **staging area** (in the `lab2` scene, as in `Shelf3D-o2.json`) holds:

- **N goal objects**: small cubes plus 0–3 **tall blocks**;
- **M clutter blocks** (not named in the goal), some spawned adjacent to goal objects so that a
  top-down grasp of the goal object collides with them.

The **cupboard** (fixed exterior; §3) has its intermediate levels repositioned to create **short
cells and one tall cell**, each layer split by partitions into 2–3 **compartments**. Each
compartment's front strip is a named **region** — an abstract object with a pose and extent, so
the `SceneEncoder`'s footprint path handles it like any other object [E — regions/partitions are
existing `Cupboard` + task-JSON machinery; capabilities §3].

**Goal:** `Stored(o)` for every goal object, where `Place(robot, o, region)` adds
`{InRegion(o, region), Stored(o)}`. Assignment of objects to regions is free — that freedom is
where the combinatorial task space lives. Heights and floor capacities are geometry, invisible
above the abstraction line; that invisibility is where the false positives come from.

**Operators.** `Pick(robot, obj)` unchanged (top-down, ground precondition — never violated by
design, since nothing is ever retrieved from the shelf). `Place(robot, obj, region)` replaces the
region-blind `Place(·, ·, shelf)`. No new skill *families*; the place controller is
region-parameterized (§5), not re-invented.

### 2.2 The three failure families and what each feeds

| family | trigger | blame channel | feeds |
|---|---|---|---|
| **F1 grasp obstruction** (pick side) | top-down grasp sweep collides with adjacent clutter | class-1, movable, *pre-existing* culprits | coverage + waste, verbatim DD2D semantics |
| **F2 crowding** (place side) | `Place(o, R)` collides with objects the *same plan* placed in R earlier | class-1, movable, **self-inflicted** culprits | record tokens, `state_delta`, jaccard — the novelty carrier |
| **F3 height mismatch** | tall block under a short cell's ceiling — no valid sample exists | culprit-free exhausted query | record tokens, jaccard/dead |

F1 exists to give coverage/waste their food with correct polarity: relocating blamed clutter is
touch-before-match (covered), relocating unblamed clutter is an unjustified superfluous step
(waste). F2 is the structurally new stress. F3 is the clean, sampler-exhaustible infeasibility
that makes `proves_failure()` semantics available. In the dynamic phase, MuJoCo physics adds a
free **class-2** family (a placement bumping a neighbor out of its region is a collateral
deviation blaming the neighbor) — a bonus, explicitly not load-bearing [?].

### 2.3 Registered design risk R1 — coverage polarity inverts on F2

Derived directly from the §6 definitions [D]: for a crowding record (failed `Place(T, A)`,
culprits = smalls the plan placed in A), class-1 index-precedence reads the **doomed order as
covered** (the smalls are touched before the re-attempt — by the very placements that create the
hazard) and the **tall-first fix as uncovered** (T's placement precedes any touch of the smalls).
Reassignments read covered trivially via the bare-membership branch (every goal object is touched
by every plan). Net: on the F2 family, coverage is expected to be an **anti-signal**, the same
shape as the pre-fix SB2D tool anti-signal on waste.

Position: **do not redesign coverage preemptively.** Coverage was the weaker feature on DD2D;
F2's discrimination is carried by jaccard + culprit tags + order-carrying position embeddings +
`state_delta`; F1 keeps coverage/waste correct where their polarity holds; and per-environment
checkpoints mean a learned head can absorb a consistent within-environment sign. Probe P4
measures the polarity per family. The contingency, if P4 demands it, has a principled
domain-agnostic shape (a U3): *when a class-1 record's culprits appear in its own delta-added
atoms, test entailment on those atoms instead of precedence* — which yields the correct polarity
on the toy walk-through, and which must first pass a frozen-pool diff showing it is (or is
acceptably not) a no-op on DD2D/SB2D. Gate, don't build.

---

## 3. Geometry and sizing

All numbers here are chosen against the **real** constraint set, not the kinematic marker
abstraction — GRIPPERS.md is explicit that the kinematic grasp ignores width, so anything meant
for hardware must be sized to the 2F-85 [E].

### 3.1 Fixed quantities (the lab shelf)

From `Shelf3D-o2.json` and the capabilities doc [E]:

| quantity | value |
|---|---|
| interior length (width) | 0.602 m |
| interior depth | 0.254 m |
| Σ shelf_heights (vertical budget, exterior fixed) | 0.762 m |
| board thickness | 0.0127 m |
| current layout | `[0.254, 0.254, 0.254]`, surfaces ≈ z 0.01 / 0.28 / 0.54 / 0.80 |
| cell clearance under current layout | ≈ 0.241 m |
| **proven placement reach** | insertion at z ≈ 0.59 over the 0.54 surface — i.e., only the *top* interior cell is demonstrated [E] |

The last row matters more than it looks: **every insertion height other than ≈0.59 is unverified
territory** [?], which is why probe P2 (a reach/insertion map) gates the cell layout.

### 3.2 The constraint system

Let `c_s`, `c_t` be short/tall cell clearances (`clearance = shelf_height − 0.0127`), `H_s`,
`H_t` object heights, `m_hand` the vertical margin the hand needs above a held object during
horizontal insertion **[? — the single most important unknown; bounded by P2]**, and
`m_inf ≈ 0.05` the infeasibility margin.

1. `Σ shelf_heights = 0.762` (exterior fixed — the one thing we may not change).
2. `H_s + m_hand ≤ c_s` — short objects placeable in short cells.
3. `H_t ≥ c_s + m_inf` — talls **robustly** excluded from short cells. Written against the
   *object's own height*, not the hand, so the infeasibility survives regardless of how
   faithfully any simulator models the gripper [D]. This is what makes F3 clean.
4. `H_t + m_hand ≤ c_t` — talls placeable in the tall cell.
5. grasp-axis of every object ∈ [0.03, 0.05] m (proven sweet spot; hard cap ~0.07) [E].

Existing operation gives one free bound: the current skill places 0.04 cubes into 0.241-clearance
cells, so **`m_hand ≤ ~0.20` is established** [E]. Anything tighter needs P2.

### 3.3 Two candidate cell layouts

**Config B — two levels (safe under existing evidence).** Remove one intermediate board;
`shelf_heights = [0.508, 0.254]` — tall cell at the bottom, short cell on top.

- Clearances: `c_t ≈ 0.495`, `c_s ≈ 0.241`.
- The short cell *is* the currently proven placement geometry (same clearance, insertion
  z ≈ 0.59 ≈ the demonstrated pose) [E]. The tall cell's insertions sit at z ≈ 0.30–0.45,
  nearer the floor than proven but with enormous headroom.
- `H_t = 0.29` satisfies (3) with margin 0.05 and (4) with headroom 0.205 even at the known
  worst-case `m_hand = 0.20`. **Config B is feasible using only established bounds** [D].
- Cost: 2 layers → 4 regions (2 tall + 2 short compartments). Requires confirming a board may
  physically be removed (G0-a).

**Config A — three levels (more regions, gated on P2).** `shelf_heights = [0.356, 0.203, 0.203]`
— tall cell bottom, two short cells above; surfaces ≈ z 0.01 / 0.37 / 0.57 / 0.77.

- Clearances: `c_t ≈ 0.343`, `c_s ≈ 0.190`. `H_t = 0.24` (margin 0.05 over `c_s`).
- Needs `m_hand ≤ 0.15` for the short cells and `≤ 0.103` for the tall cell — **both unverified**
  [?]. Yields 6 regions.

**Decision rule:** run P2 first. If P2 bounds `m_hand ≤ 0.10`, take Config A (more regions →
larger assignment space). Otherwise take Config B. If P2 shows only the proven top cell is
reachable at all, the height-mismatch family (F3) is dropped and difficulty falls back to
partitions + capacity + clutter alone (§7, fallback FB-2).

Both configs must snap to the physical shelf's actual pin positions — measured in G0-a, then
mirrored into the JSON so sim and hardware match.

### 3.4 Object specification

| object | dims (m) | grasp axis | mass (kg) | role | notes |
|---|---|---|---|---|---|
| small cube | 0.04³ | 0.04 | 0.01–0.02 | goal | the proven MuJoCo size (`"size": 0.02` half-extent) [E] |
| tall block | 0.05 × 0.05 × `H_t` | 0.05 | 0.03–0.06 | goal | `H_t` = 0.29 (B) / 0.24 (A); pads grip the top ~37.5 mm [E]; aspect ≤ 6:1 — if MuJoCo toppling is a problem, widen footprint to 0.06 (grasp 0.06 ≤ 0.07 cap) [?] |
| tall clutter | 0.05 × 0.05 × 0.20 | 0.05 | 0.02 | blocker | deliberately the same *family* as tall goals so appearance alone doesn't reveal role |
| wide box (optional) | 0.09 × 0.045 × 0.045 | 0.045 | 0.02 | goal | footprint-pressure knob for capacity tuning; grasps the short axis like the elongated kinematic blocks [E] |

Carrying a 0.24–0.29 m block below the pinch site requires the retract/home trajectory to clear
the floor — cheap check inside P2 [?].

### 3.5 Blocking geometry (F1), from the 2F-85 numbers

From GRIPPERS.md [E]: stroke 85 mm pad-to-pad → open pads' inner faces at ±42.5 mm from the
grasp center, outer faces ≈ ±50 mm (pad thickness 6.4–8 mm); pad faces 22 × 37.5 mm; the
knuckle/wrist body above the pads is wider still.

Derived design rules [D over [E], envelopes to be calibrated in P1 against the actual collision
meshes]:

- an **equal-height neighbor** blocks a top-down grasp when its face is within ≈ **3 cm** of the
  target's face along the grasp axis (it enters the descending finger sweep) [?];
- **tall clutter** (≥ 0.15 m) blocks from much farther — within ≈ **8–10 cm** — by intersecting
  the hand/wrist volume rather than the fingers [?]. Tall clutter is therefore the *primary*
  blocker: robust to collision-mesh fidelity, and cheap to author (the `Shelf3D-o2.json` pattern
  of tight per-object init regions already places objects adjacently [E]).

**Implementation, kinematic phase:** the kinematic grasp is a 2×2×7 cm marker test that ignores
all of this [E], so F1 is implemented as an explicit sampler-level validity check —
`grasp_cfree_3d(target, others)`: a swept-volume box above and around the target, sized from the
2F-85 envelope; intersecting objects → reject **and name**. This mirrors DD2D's `grasp_cfree`
exactly (a validity check that already computes its witnesses — the same observation-only
refactor pattern as before [E]), and it makes kinematic blocking match real geometry *by
construction* rather than by physics.

### 3.6 Capacity model

Regions are single-row **front strips** (JSON `ranges` confine depth to ~0.12 of the 0.254 m) —
depth-wise packing is excluded in v1 to keep strata labels clean (it becomes a difficulty
escalator later, at the price of near-feasible traps). Slots per strip ≈
`floor(W_strip / (obj_width + 0.025 sampler margin))`: a 0.29 m strip holds ~4 small cubes; a
0.19 m strip (3-way partition) holds ~2–3. Partition widths are the free tightness knob — pure
JSON config [E].

Abstract feasibility of an assignment [D]: with `S_t`/`S_s` total slots in tall/short regions and
`N_t`/`N_s` tall/small goal objects — feasible iff `N_t ≤ S_t` and `N_s ≤ S_s + (S_t − N_t)`,
*and* orderable so that talls reach tall slots before smalls consume them. The valid-assignment
fraction (measured in P3) is the astar-FP dial.

---

## 4. Difficulty strata

Four strata, mirroring the DD2D discipline (s0–s3 = minimum feasible subset size). The
per-instance difficulty statistic is `d = (k, σ)` where `k` = minimum number of clutter
relocations (oracle-computable from the blocking graph) and `σ = S_t − N_t` = tall-slot slack —
both crisp, both enumerable at generation time, so strata are *labels on generated instances*,
not hopes.

| stratum | recipe (nominal, tuned by P3/P5) | active families | expected behavior |
|---|---|---|---|
| **r0** | `N_s`=3, `N_t`=0, `M`=0, slack everywhere (`k`=0, σ large) | none | every method ≈ 0 FP — the s0/b1 analog and the sanity floor |
| **r1** | `N_s`=3–4, `N_t`=0, `M`=2–4 with `k` ∈ {1,2}; capacity slack | F1 | shortest skeletons move no clutter → astar defers exactly the plans blocked instances need; coverage/waste live |
| **r2** | `N_s`=4–5, `N_t`=1–2, `M`=0, σ ∈ {0,1} | F2 + F3 | equal-length skeletons differ only in assignment/order → astar's ordering is tie-breaking noise over a small valid fraction |
| **r3** | `N_s`=5–6, `N_t`=2, `M`=2–3, σ ≤ 1, `k` ≥ 1 | all | families compose; the headline stratum |

The compositional structure doubles as attribution: per-stratum results show which failure family
drives which method's failures — a figure DD2D/SB2D cannot produce.

**Why the baselines struggle** [P, tested by P5 and the eventual comparison]:
- **astar-dist**: length-blind to assignment (all clutter-free skeletons tie), and
  systematically prefers relocation-free plans on r1/r3 — FP scales with the inverse
  valid-assignment fraction.
- **PIGINet**: must predict packing feasibility from the static initial state; the slack-holdout
  stratum (train slack-rich, test σ ≤ 0 tight) is the analog of the DD2D s3 holdout where it
  degraded ~9× [E for DD2D; P for transfer].
- **LAZY**: a scalar online statistic cannot localize which (object, region) pair died.
- **VLMPlan**: zero-shot combinatorial packing from an image, with heights partially occluded by
  the cupboard frame.
- **Static SPECTRE vs adaptive**: after `Place(T, R_short)` exhausts, only the adaptive path can
  down-rank every candidate sharing the dead query → the after-first-failure decomposition
  should reproduce [P].

---

## 5. Implementation plan (stage-gated)

### G0 — measurements and infra checks (≤ 1 day, mostly no code)

- **G0-a** Measure the physical shelf: pin positions/increments; whether an intermediate board
  can be *removed* (Config B legality). Snap §3.3 numbers to reality.
- **G0-b** Confirm `primitive_objects.py` supports 3-axis box dims (tall blocks), not just cubic
  `size` [?].
- **G0-c** Confirm `shelf_partitions` JSON semantics end-to-end (builder → regions → 
  `sample_pose_in_region`/`check_in_region`), using `ConstrainedCupboard3D` as the working
  reference [E that the pieces exist; ? that they compose for this use].
- **G0-d** (optional lever, §4.6→§5 note) Check whether grounding is reachability-pruned — if
  so, immovable pre-placed shelf residents drop out of the actionable set, hence out of the
  culprit pool `K`, by construction rather than special-casing [?].

### Gate A — geometric feasibility probes (kinematic PyBullet, days)

- **P1 — grasp-sweep blocking.** Implement `grasp_cfree_3d`; calibrate its envelope against the
  gripper collision meshes; author jammed scenes (tall clutter at 3/6/9 cm gaps; equal-height at
  1/2/3 cm) and verify rejection + correct witnesses. *Abort criterion:* if no density that the
  instance generator can realistically produce yields blocking, F1 dies → **FB-1** (§7).
- **P2 — reach/insertion map.** IK sweep of place insertions across candidate surface heights ×
  object heights (0.04 / 0.24 / 0.29), including retract-with-tall-block floor clearance.
  Output: an empirical `m_hand(z)` bound. *Decision:* Config A vs B (§3.3). *Abort:* only the
  proven top cell reachable → **FB-2**.

### Phase K — kinematic build (the fast-iteration environment)

- **K1** Heterogeneous layer spacing in `create_pybullet_shelf` (per-layer list instead of scalar;
  the per-layer `_shelf_surface_ids` already exist [E]).
- **K2** Region objects (borrowing `obstruction3d`'s `target_region` + `is_inside` [E]) with
  poses/extents exposed to the converter; `InRegion` predicate.
- **K3** Operators: `Place(robot, obj, region)` adding `{InRegion, Stored}`; goal deriver over
  `Stored`.
- **K4** Region-parameterized place sampler (Transport3D's per-target offset + z template [E]),
  with resident-collision rejection **naming witnesses** (F2) and clean exhaustion on height
  mismatch (F3, so `proves_failure()` holds).
- **K5** Instance generator targeting `d = (k, σ)` per stratum; `PYTHONHASHSEED` note carried
  over from DD2D.
- **K6** SPECTRE converter + `FailureRecord` instrumentation under `envs/restock3d/`;
  `EMPTY_SPEC` domain contract (the SB2D precedent: per-environment code = converter + refiner
  instrumentation, nothing else [E]).
- **K7** Wire into the dd2d_v4-style pool-enumeration / labeling / caching protocol unchanged.

### Gate B — pool-shape and feature-polarity probes

- **P3 — pool shape** (~50 instances/stratum): valid-assignment fraction; order-sensitivity
  fraction (same multiset, different order, different refinement outcome); superfluous-step
  prevalence (the waste denominator must be non-empty *in practice*); pool diversity per the
  standard probe. *Tuning levers:* partition widths, `N`/`M`, σ. *Abort:* if r2 pools are
  order-insensitive after tuning, F2 is not doing its job → re-examine strip widths before
  anything downstream.
- **P4 — per-family feature polarity** (offline, no training): correlation of coverage/waste
  with eventual candidate success, split by which family produced the evidence. Tests R1's
  inversion prediction [P]. *Decision rule:* pick-side correct + place-side inverted → ship
  as-is (learned head absorbs it), record the finding; only if coverage is net-destructive does
  the U3 contingency open, gated on the frozen-2D-pool diff.

### Gate C — baseline-gap probe, then training

- **P5 — astar/oracle gap** on r2/r3: target the DD2D-like regime (astar mean FP in the tens
  against a small oracle FP). *Abort:* gap < ~10 FP after tuning → the environment does not
  earn its slot; stop before spending training compute.
- Then: data collection (matched protocol), SPECTRE training per the v3final recipe, **PIGINet
  on a matched collection** (the standing rule: no cross-collection head-to-head numbers), the
  VLMPlan adapter as one isolated env module per the established design.

### Phase D — dynamic MuJoCo (the real-robot-matched task)

- **D1** Task JSONs `Restock3D-r{0..3}.json`: `Cupboard` with the chosen `shelf_heights` +
  `shelf_partitions`; regions; per-object tight init regions for adjacency (the o2 pattern);
  object specs from §3.4.
- **D2** Extend the bilevel `goal_deriver` to read JSON region goals (a gap the capabilities doc
  itself names [E]).
- **D3** Generalize `PlaceShelfController` from its single hardcoded EE pose to a
  region-parameterized target — **the main engineering line item** of the whole plan; note it is
  a *place* generalization, an order of magnitude simpler than the abandoned front *grasp* (no
  closure geometry, reach-in and release, and the kinematic controller already does the motion
  shape [E]).
- **D4** Pick controller unchanged. Class-2 (knock-over) records observed for free; logged, not
  load-bearing.

### Phase R — real robot

Levels pinned per G0-a; tall blocks fabricated light (3D print / foam, ≤ ~60 g) per §3.4;
privileged-state deployment per the established `OracleAdapter` path (the adaptive pathway is
perception-invariant [E]); demo = an r3-style instance end-to-end. Object perception remains the
known frontier and is out of scope here [E, TidyBot doc §4].

---

## 6. Registered predictions

| id | prediction | tested by |
|---|---|---|
| RP-1 | after-first-failure decomposition reproduces (static ≈ adaptive on attempt 1; margin entirely post-failure) | final comparison |
| RP-2 | `state_delta` ablation is *positive* on Restock3D (vs the DD2D tie) — F2 makes it load-bearing | ablation grid |
| RP-3 | coverage polarity inverts on F2 evidence, is correct on F1 | P4 |
| RP-4 | waste correlates with success on r1/r3 via clutter justification | P4 |
| RP-5 | PIGINet degrades disproportionately on the σ-tight holdout stratum | holdout eval |
| RP-6 | astar FP on r2/r3 scales ~inversely with valid-assignment fraction | P3 × P5 |

## 7. Fallbacks

- **FB-1** (P1 fails — no realistic blocking): F1 is the coverage/waste carrier; without it the
  design as stated starves those features. Fallbacks, in order: (a) exploit the kinematic
  ≥2-in-marker grasp rule with near-contact packing (witness = the second marker object — free,
  but tighter than real 2F-85 blocking, so sim/real blocking thresholds diverge); (b) accept a
  record-token-only environment and re-scope the paper's coverage/waste claims to the 2D pair.
  (b) is honest but weakens the unified-features story — hence P1 runs first.
- **FB-2** (P2 fails — only the proven cell reachable): drop F3; differentiate difficulty by
  partitions + capacity + clutter only (single cell height). F2 and F1 survive intact.
- **FB-3** (P3 fails — order-insensitive pools after tuning): F2 degrades to pure capacity; the
  environment still separates adaptive from static via F1 + F3 evidence, but the self-inflicted
  novelty claim is withdrawn.

## 8. Optional levers (pocketed, not in v1)

- **Immovable residents**: pre-place non-goal cubes inside cells (JSON `initial_state` supports
  it [E]) to vary per-region capacity per instance — the rescued spirit of the shelf-pick idea,
  zero grasp engineering. Contingent on G0-d (reachability pruning keeps them out of `K`).
- **Depth-wise regions**: two-row strips reintroduce reach-past-front-row failures — more
  hardness, noisier strata.
- **Cupboard top surface** (z ≈ 0.80, top-down reachable): an always-feasible overflow region —
  *reduces* hardness, so excluded; noted only so nobody adds it by accident.

## 9. Open questions for the lab

1. May a board be physically removed (Config B), or only repositioned (constrains to Config A
   geometry with P2 risk)? — G0-a.
2. Confirm the pin increments so §3.3 snaps to real positions. — G0-a.
3. Is the ~0.19-clearance short cell of Config A acceptable on the real shelf for human loading
   /reset ergonomics during demos?
4. Object fabrication: 3D print vs foam for the tall blocks (mass and friction targets in §3.4).
