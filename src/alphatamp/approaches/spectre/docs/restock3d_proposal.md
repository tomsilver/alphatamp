# Restock3D — Environment Snapshot (`restock3d_v2`, current 2026-08-20)

The third SPECTRE evaluation environment: a **3D, kinematic-PyBullet continuous-packing** task
in which a mobile-base arm stores floor objects onto a two-section shelf. This file is a
**current-state snapshot of the deployed `restock3d_v2` variant**, written against the code
(`envs/restock3d/`), not against the older design docs — the implementation diverged
substantially from the original 2026-08-13 proposal and again when v2 replaced the discrete
region model.

> **What this supersedes.** The original stage-gated proposal (MuJoCo/TidyBot substrate,
> top-down grasp, discrete multi-slot shelf regions, F1 grasp-obstruction clutter) and the
> intermediate **v1** as-built (kinematic PyBullet, single-object *discrete* regions,
> `Place(obj,region)`/`InRegion`) are **history**. v1 still exists in the tree, frozen, and
> coexists with v2 (`models.py`/`oracle.py`/`place_controller.py` untouched); but **v2 is what
> is collected, trained, and evaluated.** The design/rationale history and the physical-shelf /
> real-robot sizing live in the ledger and archive — see *History* at the end. Where this
> snapshot and any older doc disagree, this snapshot wins for v2.

Full ledger: [`decisions/07`](decisions/07-stickbutton2d.md) / [`notebook/07`](notebook/07-stickbutton2d.md),
entries 2026-08-14 → 2026-08-20.

---

## 1. What it is, and why

The robot must **store every goal object** — small cubes and tall blocks — onto a shelf split
into a **tall bottom section** and a **short top section**. The abstraction says only *"pick it,
place it on a shelf section"*; **whether a plan actually refines is decided by real PyBullet
collision**, and it hinges on structure the abstraction cannot see:

- **F3 (height mismatch)** — a tall block placed into the *short* section overhangs and hits the
  capping board. The planner is free to emit `place_short(tall_block)`; refinement rejects it.
- **F2 (continuous crowding / packing)** — a section is one continuous strip, not a set of slots.
  Placing another object where the strip is already occupied collides a resident; a *full* strip
  cannot fit another object at all. Capacity is emergent geometry, invisible above the abstraction.
- **Reach-over (F4, depth)** — the front grasp reaches *north over* anything nearer than the
  target, so a goal south of another (within a lateral corridor, with a tall block involved)
  blocks the farther goal's pick until the nearer one is cleared. The naive pick order fails; the
  **south-to-north (nearest-first)** order succeeds — "far is harder."

So a task planner produces many goal-reaching skeletons that fail refinement for reasons an
oracle with geometric knowledge avoids — a large *abstract* task space, not a hard sampling
problem — which is exactly the regime SPECTRE's failure-conditioning targets. It also stresses a
property DD2D/SB2D do not: **self-inflicted / order-dependent culprits** (a crowding or reach-over
failure is blamed on objects the plan itself placed or left in the way).

---

## 2. Substrate and feasibility

- **Kinematic PyBullet** (the MuJoCo/TidyBot substrate of the original proposal was superseded
  2026-08-14; its dynamics were soft-collision/teleport). A holonomic SE2 mobile base
  (~0.55×0.51 m) + arm; motion is planned by BiRRT.
- **Feasibility = real collision, never a symbolic gate.** The base env reverts colliding moves;
  the pick/place motion planners raise `TrajectorySamplingFailure` when no collision-free
  solution exists. `Restock3DEnvConfig.check_base_collisions = True` — a boxed-in base **fails**
  (an intended refinement failure) rather than phasing through.

---

## 3. Scene layout — fully-lateral, three disjoint x-bands

Left→right along world x, three disjoint bands keep the base out of the object field
(`kinematic_env.py`):

| band | extent | role |
|---|---|---|
| **buffer** | x ≈ −1.1 | relocation staging — **present but inert** (clutter = 0; §12) |
| **object** | x ∈ [−0.80, −0.20], y ∈ [0.60, 1.20] (~0.6×0.6 m) | where goal objects spawn on the floor |
| **shelf** | centre (0.4, 1.4); width 0.60198 m, depth 0.254 m | the storage target |

The base parks ~0.72 m **south** of a front-grasp target, so it stays south (y ≲ 0.55) of every
object (y ≳ 0.60): it slides laterally through a clear **southern corridor** and **never crosses
the object field** — which is what makes the base collision-free and lets `check_base_collisions`
stay on. (This fully-lateral 3-band arrangement replaced an earlier shelf-north layout whose base
phase-through was the bug it fixed.)

---

## 4. The shelf — two sections (tall bottom, short top)

The shelf is built from solid board bodies at cumulative heights (`section_geometry.py`,
`region_geometry.section_surfaces`, config in `kinematic_env.py`). Only the **boards** are
collision bodies (`shelf_structure_ids()`); the ±x side walls and +y back panel render but are
cosmetic (non-collision), so F3 is a *ceiling-board* collision, not a wall effect.

| section | index | surface z | clearance (gap to ceiling board) |
|---|---|---|---|
| **tall** (bottom) → `place_tall` | `section_0` | 0.29 | **0.34** |
| **short** (top) → `place_short` | `section_1` | 0.6427 (= 0.29 + 0.34 + 0.0127) | **0.15** |

A tall block is **0.24 m** tall (half-extent z = 0.12). It fits the 0.34 tall gap but overhangs
the 0.15 short gap by **~0.09 m** → collides the short section's capping board → **F3**.

**Placement band (the continuous free space).** Each section is one **wide continuous strip**,
not discrete slots. Object-centre x spans `shelf_width/2 − 0.04` per side → the analytic band
**x ∈ [0.139, 0.661]** (board extent minus a 0.04 m per-side end margin), centred at x = 0.4, so
the band half-width is ≈ 0.261 m (~0.52 m wide). The front strip is at **y = 1.35** (= shelf y −
0.05 offset), with ±0.01 m y-jitter. Both sections share this band.

---

## 5. Objects

| object | half-extents (m) | full height (m) | family | role |
|---|---|---|---|---|
| small cube | (0.025, 0.025, 0.025) | 0.05 | `cube` | goal (`cube_goal*`) |
| tall block | (0.025, 0.025, 0.12) | 0.24 | `tall` | goal (`block_goal*`) — the F3 driver |
| clutter block | (0.025, 0.025, 0.05) | 0.10 | `clutter` | distractor — **inert** (§12) |

A cube and a tall block share a 2D footprint and differ **only in height** — the F3 axis. That is
the load-bearing fact behind the SPECTRE scene representation (§10).

---

## 6. Generation — region-rejection spawn (no floor grid)

Objects are **not** placed on a fixed grid. `generator._sample_positions` is a
**region-rejection (Poisson-disk-style) sampler**: each object's xy is drawn uniformly in the
object band, and a candidate is **rejected if it lies within an exclusion radius of any
already-placed object**:

- band x ∈ [−0.80, −0.20], y ∈ [0.60, 1.20];
- **exclusion radius 0.12 m** (min centre-to-centre) — front-grasp lateral clearance only;
- **xy only** → poses stay axis-aligned;
- **200 rejection attempts per object**, then the *whole layout* is reseeded; up to **64 reseeds**
  before `build_spec` raises;
- **deterministic in the env seed** — a tiny LCG (`_Rng`) seeded from `(seed, stratum, attempt)`,
  so the layout reproduces per seed (with a `reseed-on-failure` cushion).

**Object typing is random but deterministic:** the sampler places positions type-agnostically,
then a Fisher-Yates shuffle designates `n_tall` of the spots as tall blocks and the rest cubes.
Reach-over difficulty is *not* baked into the sampler — it emerges from geometry and is resolved
by pick **order** (south-to-north), not by spawn placement.

---

## 7. Abstract model — operators and predicates (`models_v2.py`)

- **Types:** `robot` (`Kinematic3DRobot`), `cube` (`Kinematic3DCuboid`). **No `region` type** (v1's
  `RegionType` is dropped).
- **Predicates:** `HandEmpty(robot)`, `Holding(robot,cube)`, `OnFloor(cube)`, `Stored(cube)`, plus an
  **inert `OnBuffer(cube)`**. **`InRegion` is dropped** — there is no region abstraction. `Stored`
  is purely geometric (object underside near a section surface AND xy on the shelf band); **no
  per-section capacity is represented**, which is precisely why over-full sections are invisible to
  the planner (the continuous-packing false positives).
- **Operators:**
  - `pick(robot, target)` — pre `{HandEmpty, OnFloor}`; add `{Holding}`; del `{HandEmpty, OnFloor}`.
  - **`place_tall(robot, target)`** and **`place_short(robot, target)`** — **identical abstract
    signatures and effects**: pre `{Holding}` *only* (no capacity, no height, no `Clear`); add
    `{HandEmpty, Stored}`; del `{Holding}`. The **tall/short choice is a symbolic token**: it binds
    the controller to `section_0` vs `section_1` and is validated *geometrically at refinement*
    (`place_short(tall_block)` → F3), never abstractly. Nothing hard-codes tall→bottom.
  - `place_buffer(robot, target)` — inert relocation op (add `{HandEmpty, OnBuffer}`).
- **Goal:** `Stored(o)` for **every** goal object (all `cube_goal*` + `block_goal*`). Assignment of
  objects to sections is free — that freedom, plus the invisible capacity/height, is where the
  false positives come from.

---

## 8. Placement — continuous uniform packing (`place_controller_v2.py`)

`SectionFrontPlaceController` (subclass of v1's front place) overrides only the sampler:

- **x = `uniform(−band, band)`** across the whole section strip (`band ≈ 0.261`) — **no discrete
  slots**;
- **y = front strip + `uniform(−0.01, 0.01)`** jitter;
- **z = section surface + half-height + 3 mm pad**; the object is placed **upright** (translate-only:
  the EE is derived from the recorded front grasp, so an object keeps its axis-aligned floor
  orientation — a tall block stays upright → F3, a cube lands flat).

**Capacity / crowding is emergent** — there is no slot check. A sampled x that overlaps a resident
collides during the real motion plan → `TrajectorySamplingFailure` → resample. A full strip
exhausts its retries. Placements are ~1-in-6 reliable per sample (BiRRT flakiness), so the rollout
retries generously: **~18 attempts/step** in the oracle/demo path, `num_sampling_attempts_per_step`
(default 10) on the collection path. The controller reuses v1's translate-only front place verbatim
by synthesising the section as a hidden internal `__section_N` region object, so `kinematic_env.py`
and the inherited `step` are unchanged.

---

## 9. Failure taxonomy and evidence (`instrumented_refiner.py`)

The refiner **does not gate** — it runs the real controllers via the real transition function and a
candidate fails only by real collision / MP failure. On each rejection an **observation-only** probe
attributes the cause (it sets/reads poses then restores them; it never changes the accept/reject
decision). Deepest-rejection metadata is serialized as `refiner_metadata["failures"]` with fields
`{step_index, schema, args, culprits, n_step, exhausted, budget_exhausted, dev_added, dev_deleted}`.

- **F3 (tall-into-short)** — `_probe_place_v2` lifts the held object to the band centre just off the
  surface and runs real collision detection against the shelf boards. A hit ⇒ F3, **culprit-free**,
  so it `proves_failure()`. (Unchanged from v1: the ceiling board spans the whole band, so a single
  centre probe still catches a too-tall block.)
- **F2 (continuous crowding)** — if the object fits height-wise but the place still failed, it is
  attributed to the section's **residents**: objects the prefix already stored on that surface
  (underside within 0.05 m of the section surface_z). This is *continuous section-capacity
  attribution* — the objects that crowded the placement out — not v1's discrete slot check. Class-1
  culprits.
- **Reach-over (F4)** — `_probe_pick` first checks F1 grasp obstruction (`grasp_blockers`; wired but
  retired under the unified front grasp, so effectively inert), then **`reach_over_culprits`**: the
  un-cleared **south** floor objects that block the target's front-pick reach corridor (a geometric
  rule: A south of B by ≥ 0.03 m, lateral |Δx| < 0.12 m, with a tall block involved). Class-1,
  actionable — a south-to-north order clears them first.
- **C2 (class-2 deviation)** — anything else carries the abstract-state deviation
  (`dev_added`/`dev_deleted`) instead of culprits.

**Coverage/waste.** Reach-over revives **coverage with the correct polarity**: a south-to-north
candidate *stores the south blockers before re-picking the target* → it **covers** them (coverage
1.00 vs 0.00 for a talls-first / naive candidate). This is the opposite of F2, where "touching" a
culprit *creates* the hazard. **Waste stays degenerate** under reach-over-only (goal-necessary
reordering has an empty superfluous set); reviving it needs a non-goal approach-corridor clutter —
the inert buffer machinery is exactly that lever (§12).

---

## 10. SPECTRE scene representation — full 3D point cloud (`scene_geometry.py`)

Because a cube and a tall block share a 2D footprint and differ only in height (the F3 axis), a
2D-footprint scene would be blind to the decisive dimension. So each object emits a **full 3D
analytic point cloud** (from ground-truth half-extents, not sensed):

- **32-point analytic box surface** (`object_point_cloud`) — 8 corners + 6 face-quadrant centres,
  scaled by half-extents so the z-extent scales with height;
- alongside the **2D boundary ring**, **pose_z**, and **height** (for 2D consumers and the height
  scalar);
- families `tall` / `cube` / `clutter` / `robot`; the shelf recorded as a `ContainerGeometry`
  (`kind="shelf"`), not a registry object; `frame_h` carries the z extent for the 3D path.

The deployed SPECTRE model consumes this via `--scene-3d` (point_dim 3, pose_dim 4) + the
PointSetEncoder, and also ingests the initial abstract state + goal atoms (`--atom-mode profiles`).

---

## 11. Difficulty strata (`generator.STRATA_V2_PILOT`, `strata_v2.py`)

Strata are **(n_tall × n_short)** section configs (tall blocks × short cubes). Two numbering
systems, deliberately separated: the **banding stratum** (difficulty index 0..4, encoded in
`problem_id` via `V2_STRATUM_BAND = SPLIT_BAND // 5`) and the committed **recipe key** (a
`generator.STRATA` entry pinning object counts; recipe tuple = `(n_small, n_tall, n_tall_regions,
n_short_regions)`, **no clutter field** — clutter lives in the separate all-zero
`_CLUTTER_PER_STRATUM`).

| banding stratum | recipe key | n_tall × n_short | objects | collection size (train/val/test) |
|---|---|---|---|---|
| 0 | 11 | 2 × 2 | 4 | 50 / 15 / 15 |
| 1 | 12 | 3 × 3 | 6 | 50 / 15 / 15 |
| 2 | 14 | 3 × 4 | 7 | 25 / 10 / 10 |
| 3 | 15 | 4 × 3 | 7 | 25 / 10 / 10 |
| 4 | 13 | 4 × 4 | 8 | 25 / 10 / 10 |

Full target = **175 / 60 / 60 = 295**. Per-stratum collection budgets `(K_max, r_cap s)` =
`{0:(20,40), 1:(40,70), 2:(75,80), 3:(75,80), 4:(75,90)}`; collected one single-stratum job at a
time in `SEQUENTIAL_ORDER = (0, 1, 3, 2, 4)` (light first, 4×3 before 3×4). Gym ids
`spectre/Restock3Dv2-r{key}-v0`.

---

## 12. Distractor blockers — supported, currently inert

The environment has **full end-to-end support for non-goal distractor blockers, but every stratum
runs with zero of them.** The machinery is one flag away:

- **Generation:** `generator._sample_blockers` would place `n_clutter` cubes each ~0.09 m off a
  target's ±x face (inside the target's exclusion radius, clear of everything else). It **early-
  returns `[]`** because `_CLUTTER_PER_STRATUM` is **0 for every key**.
- **Model:** the `OnBuffer` predicate and `place_buffer` operator exist; goals never reference them.
- **Controllers:** `BufferPlaceController` / `in_buffer_zone` (the buffer x-band, §3) exist.
- **Oracle:** `oracle.py` has the relocation phase (`Pick(clutter) + PlaceBuffer(clutter)`) that only
  runs when clutter is present.

**To enable:** raise `_CLUTTER_PER_STRATUM[key] > 0` in `generator.py` **and** the matching
`kinematic_env.CLUTTER_PER_STRATUM[key]` (generator drives positions, kinematic_env drives specs —
the two must agree). Distractor clutter is the path to a **non-goal approach-corridor blocker** that
must be relocated to the buffer — the missing carrier that would revive **waste** (§9).

---

## 13. Oracle and plan-generation prior

- **Oracle (`oracle_v2.py`)** builds a feasible skeleton directly and certifies it — no collection
  pipeline: tall blocks → the tall section, cubes load-balanced across sections; a **south-to-north
  (nearest-first)** pick order; then a manual rollout-with-resampling certifier (18 attempts/step,
  fresh RNG per attempt, optional `max_seconds` cap). Section choice is validated by real collision
  (`place_short(tall)` never certifies). It certifies sampled scenes across the strata.
- **Geometry-informed plan-generation prior (`plan_generator_v2.py`).**
  `GeometryGuidedRestockPlanGenerator` subclasses the stock hff generator and replaces unit operator
  cost with a nearest-first **pick cost** `c(pick(o)) = 1 + λ·|{o′ unpicked OnFloor : d(o′) < d(o)}|`
  (λ=1, `d(o)` = object y = northward reach), so a plan's extra penalty is its Kendall-tau inversion
  count vs the south-to-north oracle order. It generates the oracle plan in **~15–26 attempts / 100%**
  vs geometry-blind hff's **~4000 / 50–80%** (≈200×). This is the **default pool generator** for v2
  collection. It is a *plan-generation* prior, distinct from the (deferred) eager section-capacity
  heuristic.

---

## 14. Collection, training, and comparison state (current)

- **Collection** (`experiments/spectre/restock3d_v2_{collect,run_all}.py`) follows the DD2D/SB2D
  protocol — **no oracle in the loop**: the geometry-guided prior emits the candidate pool, each
  skeleton is refined non-short-circuiting by `BacktrackingRefiner`, and a problem is kept iff ≥1
  candidate refines. Full `EpisodeRecord`s carry the pool, per-candidate outcomes + wall-clock, the
  3D scene geometry, and F2/F3 instrumented failures. Env_variant **`restock3d_v2`**. Sequential
  per-stratum jobs; RAM-sized worker counts.
- **Status:** strata **2×2, 3×3, 4×3 collected**; **3×4 and 4×4 still collecting** (as of 2026-08-20).
- **Comparison** (`compare_methods.py`, env key `restock3d`): SPECTRE (3D point-set + atoms), PIGINet
  (oblique height-visible crops), and LAZY (9-dim height graph) are trained + eval-cached vs the naive
  planner order, over the collected strata (currently {2×2, 3×3, 4×3}, 3 seeds). Headline so far: all
  learned methods crush the naive order; **LAZY dominates**, **SPECTRE edges PIGINet at the crowded
  4×3** (the first hint of the representation advantage, CI still includes 0), **adaptivity is inert**
  on these strata. Numbers + CIs in [`notebook/07` 2026-08-20](notebook/07-stickbutton2d.md#2026-08-20-restock3d-4x3-stratum-added-3-strata).

---

## 15. Deferred / open

- **Remaining strata** — 3×4 and 4×4 collection; then re-train/re-cache (the crowding + asymmetric
  3×4 are where the SPECTRE > PIGINet edge may reach significance).
- **Waste revival** — still degenerate under reach-over-only; needs the non-goal approach-corridor
  clutter (the inert buffer machinery, §12).
- **Eager section-capacity heuristic** — a refinement-order heuristic folding continuous
  section-capacity into A* costs (distinct from the plan-generation prior in §13).
- **VLMPlan** adapter + labeler for restock3D.
- **≥3-seed / full-strata paper numbers**; the dynamic-MuJoCo and real-robot phases.

---

## History

The original stage-gated proposal (v0.1, 2026-08-13: MuJoCo/TidyBot, top-down grasp, discrete
multi-slot regions with `InRegion`, F1 grasp-obstruction clutter, the two candidate cell layouts,
and the physical-shelf / 2F-85 gripper sizing needed for the eventual real-robot phase) and the v1
as-built (kinematic PyBullet, single-object **discrete** regions) are retained as design
rationale/history in the ADR/notebook ledger:
[`decisions/07`](decisions/07-stickbutton2d.md) / [`notebook/07`](notebook/07-stickbutton2d.md),
2026-08-13 → 2026-08-18 (env origin, the fully-lateral rebuild, the v2 continuous-packing variant,
the geometry-informed prior, the 3D point-cloud representation, and the 5-stratum full-collection
protocol). The real-robot geometry constraints (interior 0.602 × 0.254 m, Σ shelf heights 0.762 m
fixed exterior, board thickness 0.0127 m, grasp axis 0.03–0.05 m) still apply to the deferred
hardware phase.
