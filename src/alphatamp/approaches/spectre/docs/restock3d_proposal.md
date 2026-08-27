# Restock3D — Environment Snapshot (`restock3d_v3`, current 2026-08-27)

The third SPECTRE evaluation environment: a **3D, kinematic-PyBullet continuous-packing** task in
which a mobile-base arm stores floor objects onto a two-section shelf. This file is a **current-state
snapshot of the deployed `restock3d_v3` variant**, written against the code (`envs/restock3d/`), not
the older design docs — the implementation has moved on twice (the v2 continuous-packing rebuild, then
the v3 per-object-dimensions difficulty pass).

> **What v3 is, and what it supersedes.** v3 keeps v2's continuous-packing substrate but makes block
> **selection** the hard problem: every object now has a **sampled width** and a **sampled height**
> straddling the section cutoffs, on a **re-balanced shelf**, and problems are drawn by a **new analytic
> generator** that guarantees difficulty. **v2 is retained frozen as a negative control** (an easier
> env where the learned rankers all sit near-oracle; a second `RESTOCK3D` EnvSpec still points at
> `restock3d_v2`). v1 (discrete regions) and the original 2026-08-13 proposal are history. Where this
> snapshot and any older doc disagree, this snapshot wins for v3.

Full ledger: [`decisions/07`](decisions/07-stickbutton2d.md) / [`notebook/07`](notebook/07-stickbutton2d.md),
entries 2026-08-14 → 2026-08-27. Single source of truth for the geometric constants:
[`envs/restock3d/feasibility_v3.py`](../envs/restock3d/feasibility_v3.py).

---

## 1. What it is, and why

The robot must **store every goal object** onto a shelf split into a **tall bottom section** and a
**short top section**. The abstraction says only *"pick it, place it on a shelf section"*; **whether a
plan refines is decided by real geometry**, and it hinges on structure the abstraction cannot see:

- **Block selection (the v3 difficulty axis).** Each block has a per-object width and height. Which
  subset of blocks goes on which level — and in what order — is a genuine packing/assignment problem:
  heights near the short/tall cutoff make "can this block go short?" a real decision, and widths near
  the section capacity make "do these blocks all fit on one level?" a real decision.
- **F3 (height mismatch).** A block too tall for a section cannot be threaded under its ceiling board
  (the gripper needs ~0.10 m of headroom below the board). The planner is free to emit
  `place_short(tall_block)`; refinement rejects it.
- **F2 (continuous crowding / packing).** A section is one continuous strip. Placing another block where
  the strip is already occupied collides a resident; a *full* strip fits nothing more. Capacity is
  emergent geometry, invisible above the abstraction.

So a task planner produces many goal-reaching skeletons that fail refinement for reasons an oracle with
geometric knowledge avoids — a large *abstract* task space, not a hard sampling problem — exactly the
regime SPECTRE's failure-conditioning targets. It also stresses a property DD2D/SB2D do not:
**self-inflicted / order-dependent culprits** (a crowding failure is blamed on blocks the plan itself
placed) and a **blameless height certificate** (F3), which is what the `repeat` feature (§9) reads.

---

## 2. Substrate and feasibility

- **Kinematic PyBullet.** A holonomic SE2 mobile base (~0.55×0.51 m) + arm; motion planned by BiRRT.
- **Feasibility = real collision, never a symbolic gate** *for the eval instrument*. The pick/place
  motion planners raise `TrajectorySamplingFailure` when no collision-free solution exists;
  `Restock3DEnvConfig.check_base_collisions = True` (a boxed-in base **fails** rather than phasing
  through).
- **Collection labels are analytic.** The deployed `restock3d_v3` dataset is labelled by a pure-geometry
  **analytic refinability classifier** (`feasibility_v3.classify_skeleton`, §6) — no motion planning —
  which is byte-compatible with the real refiner's failure schema. The real PyBullet refiner is kept as
  the **eval instrument** and the audit reference (§14). A real-MP-labelled collection
  (`restock3d_v3_real`) is in progress.

---

## 3. Scene layout — fully-lateral, three disjoint x-bands

Left→right along world x, three disjoint bands keep the base out of the object field (unchanged from
v2, `kinematic_env.py`):

| band | role |
|---|---|
| **buffer** (x ≈ −1.1) | relocation staging — **present but inert** (clutter = 0; §12) |
| **object** (x ∈ [−0.80, −0.20], y ∈ [0.60, 1.20]) | where goal objects spawn on the floor |
| **shelf** (centre ≈ (0.4, 1.4)) | the storage target |

The base parks ~0.72 m **south** of a front-grasp target, so it stays south of every object: it slides
laterally through a clear southern corridor and **never crosses the object field**, which is what lets
`check_base_collisions` stay on.

---

## 4. The shelf — two sections, re-balanced for v3

Solid board bodies at cumulative heights (`region_geometry.section_surfaces`; config
`generator_v3.v3_config()` → `Restock3DEnvConfig(section_clearances = feasibility_v3.SECTION_CLEARANCES)`).
Only the boards are collision bodies; the side/back panels render but are cosmetic, so F3 is a
*ceiling-board* effect. Board thickness `_SHELF_HEIGHT = 0.0127`; `bottom_surface_z = 0.29`.

| section | index | surface z | clearance (gap to ceiling board) |
|---|---|---|---|
| **tall** (bottom) → `place_tall` | `section_0` | 0.29 | **0.27** |
| **short** (top) → `place_short` | `section_1` | **0.5727** (= 0.29 + 0.27 + 0.0127) | **0.22** |

**v3 re-balances the partition to `SECTION_CLEARANCES = (0.27, 0.22)`** (from v2's `(0.34, 0.15)`) —
same total shelf height (0.49) but the short section is much taller. This is why **the short section is
no longer cube-only**: with a 0.22 m clearance and a 0.12 m short cutoff, short-eligible blocks span
0.05–0.12 m rather than only the 0.05 m cube. Each section is one **wide continuous packing band**
(§8), not per-object cells. Lateral band constants: `GAP = 0.06`, `USABLE = 0.50`, `END_MARGIN = 0.04`;
physical shelf width 0.60198.

---

## 5. Objects — per-object sampled width and height (the v3 change)

v3 drops v2's type-keyed constant dims (fixed 0.05 m cube / 0.24 m tall block) and **samples every
object's width and height** (`generator_v3`, depth fixed at half-extent 0.025):

- **Width** `U[0.02, 0.08]` (`feasibility_v3.WIDTH_MIN/WIDTH_MAX`), per object, rounded to 4 dp — full
  x-width, so section capacity (§6) genuinely depends on which blocks land there.
- **Height**, sampled in three role bands (`generator_v3._sample_heights`, then shuffled so role order ≠
  sample order):
  - **forced/tall-only** (`n_forced` blocks): `U[0.121, 0.17]` — must go to the tall section;
  - **near-threshold** (`n_near` blocks): `U[0.09, 0.15]` — straddles the short cutoff (the genuine
    decisions);
  - **free/short-eligible** (the rest): `U[0.05, 0.12]`.
- **Cutoffs** `SHORT_CUTOFF = 0.12`, `TALL_CUTOFF = 0.17` (`CUTOFF = {"tall": 0.17, "short": 0.12}`).
  `height_eligible(h, section)` = `h ≤ CUTOFF[section] + eps` — the F3 arm-insertion rule (~0.10 m
  gripper headroom below each board).
- **Colour cue** (`_rgba`): reddish if `h > SHORT_CUTOFF` (tall-only), greenish if short-eligible.

A block's 2D footprint no longer determines its class — width and height are independent — which is the
load-bearing fact behind the SPECTRE 3D point-cloud scene representation (§10).

---

## 6. Generation — the new analytic generator (`generator_v3.py`, `feasibility_v3.py`)

`build_spec_v3(seed, stratum)` (LRU-cached) samples widths + role-banded heights, applies an acceptance
filter, then draws floor XY with v2's region-rejection sampler and shuffles spot↔role. It loops up to
`_MAX_RESEED_V3 = 600` before raising.

**Capacity formula.** `level_used(widths) = Σw + GAP·(n−1) + 2·END_MARGIN = Σw + 0.06·(n−1) + 0.08`;
`level_fits` iff `level_used ≤ USABLE + eps = 0.50` (empty level → 0).

**Splits + acceptance.** A "split" assigns each block to `{tall, short}`; `split_is_feasible` requires
every block **height-eligible** for its section AND both levels pass the capacity formula.
`enumerate_feasible_splits` walks all `2^n` masks; `feasible_ratio → (n_feasible, 2^n, rho)`;
`min_fill_over_feasible` is the loosest feasible packing's fill fraction. `_accept(blocks, p)` requires
**all** of: `n_feasible ≥ 1`; `rho` within the stratum's `rho_band`; `min_fill` within
`FILL_BAND = (0.55, 0.995)`; and — if `require_crack` — **both** greedy hand-rules pick an *infeasible*
split ("no universal rule").

**Greedy hand-rules** (`HAND_RULES`): `greedy_widest_best_fit` (widest-first, best-fit into the
height-eligible level with least leftover slack) and `greedy_send_shortest_up` (shortest-first, fill
short while it fits, rest to tall). A hard stratum is one that defeats both.

**`classify_skeleton`** — the analytic refinability classifier used for collection labels. It walks a
candidate skeleton in order (`pick`/`place_tall`/`place_short`) and returns `None` if feasible, else the
**first-violation** failure dict in the exact `instrumented_refiner.failure_metadata` shape:
- **F3 height** (tested first): a place of a block taller than its section cutoff → **culprit-free**
  (`dev_added/dev_deleted = []`, `proves_failure()`).
- **F2 crowding**: a place that overflows `level_fits` → culprits = the residents already on that level.
- **F4 reach-over**: a pick whose south corridor still holds uncleared blockers (shared `_blocks_reach`)
  → **culprit-free/"dead"** (parity with the real `_probe_pick`; still a class-2 infeasibility, only
  culprit attribution is dropped). **v3 tracks only F2 (crowding) culprits.**

---

## 7. Abstract model (`models_v3.py`) — reuses v2's abstraction

`create_restock3d_v3_models` calls v2's `build_restock3d_v2_models` with
`lifted_controllers_factory = create_lifted_controllers_v3`; only the sim body (per-seed dims) and the
place controllers differ.

- **Predicates:** `{HandEmpty, Holding, OnFloor, Stored}` (+ inert `OnBuffer`). **No `region` type, no
  capacity, no height** — `Stored` is purely geometric section membership, so over-full sections and
  wrong-height placements are invisible to the planner (the intended false-positive source).
- **Operators:** `pick`; `place_tall`/`place_short` with **identical abstract effects**
  (pre `{Holding}`; add `{HandEmpty, Stored}`; del `{Holding}`) — the tall/short choice is a symbolic
  token, bound to `section_0`/`section_1` and validated only by real geometry (`place_short(tall)` → F3);
  `place_buffer` (inert relocation).
- **Goal:** `Stored(o)` for every goal object; section assignment is free.

---

## 8. Placement — left-to-right continuous packing (`place_controller_v3.py`)

`LeftToRightSectionPlaceController` (subclass of v2's `SectionFrontPlaceController`) overrides only
`sample_parameters`: it reads the section's current residents from state (`_resident_right_edges`,
filtered to bodies within `_RESIDENT_Z_TOL = 0.05` of the section surface) and packs the held block at
the **leftmost free slot** (`leftmost_slot_center`: `left_face = max(right_edges) + GAP` if residents
else `cx − USABLE/2 + END_MARGIN`), plus `±0.01` jitter. It is **consistent-by-construction with
`feasibility_v3.level_fits`**, so a full strip is unpackable both analytically and in sim. Objects land
upright (translate-only front place from the recorded grasp).

---

## 9. Failure taxonomy and evidence (`instrumented_refiner.py`)

The refiner does not gate — a candidate fails only by real collision / MP failure. On each rejection an
**observation-only** probe attributes the cause. The v3 behaviour is gated by `section_height_cutoffs`
(None ⇒ v2 byte-identical); `collect.py` passes `{section_0: 0.17, section_1: 0.12}` for the v3 model.

- **F3 (tall-into-short)** — the v3 **arm-insertion parity probe** in `_probe_place_v2`: if the held
  block's full height `> section cutoff + 1e-9`, return culprit-free `"F3"` *before* the PyBullet
  block-vs-board test (the board clearance sits ~0.10 m above the arm-insertion cutoff, so a block in
  `(cutoff, clearance]` fits under the board but the arm cannot thread it in). Culprit-free, provable.
- **F2 (continuous crowding)** — a place that fits height-wise but still fails is attributed to the
  section's **residents** (blocks the prefix already stored there, within `_RESIDENT_Z_TOL`). Class-1
  self-inflicted culprits. **This is the only culprit class v3 tracks.**
- **Pick-side (F1 grasp / F4 reach-over): DISABLED** — `_probe_pick` returns `((), "C2")`. F1 is retired
  under the unified front grasp; F4 was ~0.03% of real failures (`decisions/07` 2026-08-25). The
  `grasp_blockers`/`reach_over_culprits` geometry remains defined (for the generator and coverage) but
  is not consulted. The analytic classifier's F4-dead / F2-only rule mirrors this exactly.

**Coverage / waste / repeat.** Coverage/waste are computed as elsewhere (from the F2 residents).
**`repeat` is the load-bearing adaptive signal here**: Restock3D-v3's `place_tall`/`place_short` are the
**only** schemas that declare `QueryAxioms.step_certificate = True`, so an F3 failure (blameless,
exhausted) certifies its exact `(schema, args)` step, and any candidate repeating that step is vetoed
(`repeat = 1`). This is the F3 mass that coverage — an *ordering* signal — cannot see; it is what makes
adaptivity load-bearing on this env (§14). `regroup` (`grouping_certificate`) is declared but deprecated
and off.

---

## 10. SPECTRE scene representation — full 3D point cloud + atoms (`scene_geometry.py`)

Because width and height are independent per object (§5), a 2D-footprint scene would be blind to the
decisive dimensions. Each object emits a **32-point analytic box-surface point cloud** (from ground-truth
half-extents, z-extent scaling with height) alongside the 2D boundary ring, `pose_z`, and `height`. The
deployed SPECTRE model consumes this via `--scene-3d` (`point_dim 3`, `pose_dim 4`) + the PointSetEncoder,
and also ingests the initial abstract state + goal atoms via `--atom-mode profiles`.

---

## 11. Difficulty strata (`strata_v3.py`)

Four strata, **one block count each** — `n = 6/7/8/9` — riding the **shared 4-stratum band**
(`STRATUM_BAND = SPLIT_BAND // 4`), so `compare.stratum_of` needs no routing edit. `FILL_BAND = (0.55,
0.995)`. Gym ids `spectre/Restock3Dv3-r{stratum}-v0`.

| stratum | n | rho_band | n_forced | n_near | require_crack | K_max | r_cap (s) | sizes (train/val/test) |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 | (0.08, 0.55)  | 1 | 2 | False | 40  | 50  | 100 / 25 / 25 |
| 1 | 7 | (0.02, 0.30)  | 1 | 2 | False | 60  | 70  | 100 / 25 / 25 |
| 2 | 8 | (0.005, 0.15) | 2 | 2 | True  | 150 | 90  | 100 / 25 / 25 |
| 3 | 9 | (0.002, 0.06) | 2 | 3 | True  | 200 | 110 | 100 / 25 / 25 |

Full target = **400 / 100 / 100 = 600**.

---

## 12. Distractor blockers — supported, currently inert

The environment retains v2's full end-to-end support for non-goal distractor blockers (the buffer band,
`OnBuffer`/`place_buffer`, `BufferPlaceController`, the oracle relocation phase), but **every stratum
runs with zero of them** (clutter = 0). Distractor clutter is the deferred lever that would revive
**waste** (which is degenerate when all reordering is goal-necessary, §9).

---

## 13. Oracle and plan-generation prior

- **Oracle** builds a feasible skeleton directly and certifies it by real rollout-with-resampling — no
  collection pipeline. Section choice is validated by real collision.
- **Geometry-informed plan-generation prior** (`GeometryGuidedRestockPlanGenerator`): the stock hFF
  generator with a nearest-first pick cost `c(pick(o)) = 1 + λ·|{o′ unpicked OnFloor : d(o′) < d(o)}|`
  (λ=1, `d(o)` = northward reach `y`), so a plan's penalty is its Kendall-τ inversion count vs the
  south-to-north order. This is the **default pool generator** for v3 collection (a *plan-generation*
  prior, distinct from the deferred eager section-capacity heuristic).

---

## 14. Collection, training, and comparison state (current)

Two collection variants exist, both registered in `env_registry.py`; the geometry-guided prior emits the
candidate pool, each skeleton is labelled non-short-circuiting, and a problem is kept iff ≥1 candidate
is feasible. Full `EpisodeRecord`s carry the pool, per-candidate outcomes + wall-clock, the 3D scene
geometry, and F2/F3 failures.

- **`restock3d_v3` — SYNTHETIC (analytic labels).** `CollectionConfig.refiner_mode = "analytic"`:
  `collect._restock3d_analytic_outcome` labels via `classify_skeleton` (no motion planning) and
  **synthesizes wall-clock** (fail = full `r_cap`; success = `U[0.6, 0.8]·r_cap`). Collected by
  `restock3d_v3_collect.py` / `restock3d_v3_run_all.sh` (one stratum/process, workers 16/12/6/4),
  **400 / 100 / 100**. **This is what is trained and wired into the comparison.**
- **`restock3d_v3_real` — REAL (hybrid-prune).** Same env/strata/generator, labelled by real PyBullet
  MP: **TRAIN → `hybrid_prune`** (analytic classifier prunes the K_max pool, real MP labels the
  analytic-feasible candidates + a deterministic 25% audit of the analytic-infeasible, the rest trust
  the analytic label; each `OutcomeRecord.label_source ∈ {real, analytic}`); **VAL/TEST → fully real**.
  Motivated by a pilot showing the analytic classifier is a poor proxy (**~58% false-positive** on
  analytic-feasible, **~13% false-negative**). Scripts `restock3d_v3_real_{pilot,run_all}.sh`; raw
  episodes exist but there is **no derived cache, no EnvSpec, and no comparison table yet**.

**Training + comparison (synthetic).** SPECTRE deploy recipe (`restock3d_v3_train.sh`, jointly-trained —
not the DD2D/SB2D residual): `--scene-3d --atom-mode profiles --coverage-mode both --repeat-feats
--step-join --out-suffix _repeat` → `checkpoints_spectre_atoms_repeat`. Comparison via the
`RESTOCK3D_V3` `EnvSpec` (`compare_envs.py`, key `restock3d_v3`, strata n=6..9 on the shared 4-band,
`has_timing`, isolation ablation arms; `precompute_dd2d_cache._REFINE_CAP_S["restock3d_v3"] = 90`).
PIGINet (`piginet_s{seed}`, height-visible oblique crops) and LAZY (9-dim height graph) are trained + eval-cached; **no VLMPlan** (deferred). Result (test n=100, 3 seeds; **synthetic → upper bound**):

| method | ALL |
|---|---|
| **SPECTRE-adaptive** (`repeat`) | **3.13 ± 0.11** (n6 1.04 / n7 1.37 / n8 2.99 / n9 7.11) |
| SPECTRE-static | 12.01 ± 0.30 |
| LAZY-adaptive | 11.79 ± 0.10 |
| PIGINet (low-level) | 38.11 ± 1.23 |
| astar-dist | 38.41 |

The §0 **representation crossover appears most sharply here**: PIGINet ties the naive planner order
(38.11 ≈ astar 38.41), both abstract rankers beat them ~3×, and the **`repeat` F3 certificate** drops
SPECTRE from ~12 (a tie with LAZY) to **3.13** (SPECTRE-static is 12.01, so the whole gap is adaptive).
**⚠️ These are analytic-synthetic labels** — an upper bound favouring the geometry-encoding
representation; the `restock3d_v3_real` audit is what will price how much survives real MP noise.

---

## 15. Deferred / open

- **Real-refiner comparison** — the `restock3d_v3_real` collection (hybrid-prune train / real val+test),
  its derived cache + EnvSpec, and a real-vs-synthetic audit slice.
- **VLMPlan** adapter + labeler for Restock3D.
- **Eager section-capacity heuristic** — folding continuous section-capacity into A* costs (distinct
  from the plan-generation prior in §13).
- **Waste revival** — still degenerate under F2/F3-only; needs the non-goal approach-corridor clutter
  (the inert buffer machinery, §12).
- **≥3-seed / real paper numbers**; the dynamic-MuJoCo and real-robot phases.

---

## History

v2 (`restock3d_v2`, continuous-packing, fixed 0.05 cube / 0.24 tall block, 5 banding strata
n_tall×n_short) is **retained frozen as the easier negative control** — the learned rankers all sit
near-oracle there, so it cannot separate methods; a second `RESTOCK3D` EnvSpec still points at it.
v1 (kinematic PyBullet, single-object **discrete** regions with `InRegion`) and the original stage-gated
proposal (v0.1, 2026-08-13: MuJoCo/TidyBot, top-down grasp, discrete multi-slot regions, F1
grasp-obstruction clutter, the real-robot shelf/gripper sizing) are retained as design rationale in the
ADR/notebook ledger ([`decisions/07`](decisions/07-stickbutton2d.md) /
[`notebook/07`](notebook/07-stickbutton2d.md), 2026-08-13 → 2026-08-27). The real-robot geometry
constraints (interior 0.602 × 0.254 m, Σ shelf heights 0.762 m, board thickness 0.0127 m) still apply to
the deferred hardware phase.
