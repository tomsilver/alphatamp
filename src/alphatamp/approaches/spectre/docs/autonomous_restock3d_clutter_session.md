# Autonomous session — Restock3D F1 clutter (relocation), base-collision fix, re-calibration, coverage/waste

**2026-08-15, unsupervised overnight run.** No human in the loop: the user approved a gated plan to
re-add F1 grasp-obstruction clutter per `docs/restock3d_proposal.md` §2–4 (deferred in v1), fix the
base-drives-through-blocks bug first, re-run calibration with clutter, and wire coverage/waste — asking
that design choices be made by best judgment, recorded here + in an ADR, and reported at the end. This
narrative is **unratified** (judged without a human); the ratified decision is
[`decisions/07` 2026-08-15](decisions/07-stickbutton2d.md#2026-08-15-restock3d-f1-clutter-re-added-relocation-buffer),
numbers in [`notebook/07` 2026-08-15](notebook/07-stickbutton2d.md#2026-08-15-restock3d-f1-clutter-build-mechanism-calibration).

Gates ran 0→1→2→3→{4,5,6}→7; every gate passed. The non-trivial judgment calls, each with rationale:

## Design choices (best-judgment, no human)

- **DC-B0 — base collision is best-effort at the planner, not enforced.** Root cause: the mobile base
  footprint is **0.55×0.51 m** (measured AABB over all base-body links) vs floor-object spacing
  **~0.30 m**. Enabling `check_base_collisions=True` + floor movables in the base-nav set collapsed
  oracle certification to **r0 50% / r1 0% / r2 0% / r3 0%** (was 100%) — the wide base is geometrically
  boxed by the dense floor. Fix: `_base_nav_collision_ids` adds floor movables (minus the
  approached/carried target) to all four `get_base_plan` sites; `get_base_plan` falls back to a
  shelf-only straight path when boxed (so a pick never hard-fails); `check_base_collisions` stays
  **False** (step-time reversion would re-break the fallback). The base avoids floor blocks where a
  collision-free path exists (rare on the short pick hops — the base-nav demo scan shows primary
  avoidance refusing on every short-hop placement), else reverts to the pre-fix teleport. Full
  enforcement needs a navigable floor layout; deferred. Honest limitation: in dense scenes the base
  still teleports, so the user's cosmetic concern persists until the floor is made navigable.

- **DC-B1 — F1 targets CUBE goals; clutter +y (toward the shelf), gap ~0.07 m.** Deterministic Gate-1
  sweep: a cube's top-down grasp is obstructed for a +y clutter at 0.05–0.10 m (named culprit, clutter
  itself pickable, no deadlock cycle); +x/−x never block a top-down grasp; a tall block's *front* grasp
  is not blockable by side clutter (and close clutter is itself blocked by the tall block → a cycle).
  `grasp_blockers(sim, obj, state)` was factored out of `_probe_pick` as the single source of truth so
  the refiner probe, the eager blockers table and the oracle agree.

- **DC-B2 — relocation via DD2D-style `OnBuffer`/`PlaceBuffer` + a floor buffer zone; the floor had to
  be registered as a placement surface.** Buffers are controller-side spots (`BUFFER_SPOTS`/
  `in_buffer_zone`), not abstract regions (a floor region at surface_z≈0 would surface-z-match and
  wrongly emit `Stored`). `BufferPlaceController` mirrors the local `RegionPlaceController` (top-down
  place; `lift()` already returns the empty arm to HOME, so a custom retract was tried and **removed**).
  The load-bearing bug: the base env only releases a grasped object onto a **registered surface** (the
  shelf boards), so a floor buffer place never detached (`grasped` stuck, `finger 0.29`, next pick's arm
  MP failed) until `ObjectCentricRestock3DEnv._get_surfaces_supporting_object` counted the floor
  (underside within `min_placement_dist` of z=0). Also fixed an SE2 base-plan smoothing ±π assertion by
  falling back to raw waypoints. The oracle got a relocation phase; the eager table a T5 penalty +
  order-aware feasibility.

- **DC-B3 — deployed recipe is r1 clutter only; r3 stays F2+F3.** The Gate-3 sweep found the crux:
  F1's relocate-first skeleton is longer and *off the hff gradient*, so the plain hff pool is
  **censored** on every cluttered problem (0 feasible in top-200 — a catastrophic naive FP, the intended
  difficulty) while the oracle certifies 100%. Only the eager T5 penalty surfaces the feasible, and only
  on r1 (first-feasible 0); on r3 the F1+F3+relocation search does not enumerate within budget (eager
  times out with 0 candidates). So F1 **composes with F2 (r1) but not with F3 (r3)** at the
  pool-generation level. r1 gets clutter (deployable pool = eager); r3 stays no-clutter. The
  relocation-aware pool generator for r3 (and the full collection) is deferred — this is the DC1/DC3
  pool-composition tension amplified.

- **DC-B4 — coverage/waste verified non-degenerate on F1 with no new compute code.** An F1 record names
  the clutter (class-1 culprit via `grasp_blockers`); the clutter is actionable (`PlaceBuffer` adds
  `OnBuffer`) so it enters the culprit pool; a relocate-first candidate covers it (coverage 1.00 vs 0.00
  direct, RP-3) and relocating an unblamed clutter is unjustified waste (1.00 vs 0.50, RP-4).
  `coverage_feats` was already plumbed through `TrainConfig`/`dataset`/`model`; training deferred.

## CI scoping (judgment)

Full `./run_ci_checks.sh` was **not** run: the working tree carries the user's uncommitted ShelfObstruct3D
WIP (separate effort), which pollutes a repo-wide `mypy .` / `pytest --pylint` regardless of this work,
matching the v1 calibration session. Verified scoped: autoformat (black/docformatter/isort) clean; the
restock3d fast test suite (29 tests) + the new clutter tests (5 fast, 3 slow) pass; the new one-off
diagnostic/demo/sweep/probe scripts were added to the mypy exclude + pylint `ignore-patterns` alongside
the existing restock3d harness precedent, and `pybullet` (no stubs) was added to the mypy
`ignore_missing_imports` override. Residual, pre-existing and shared with v1: the restock3d package
carries substrate-typing mypy noise (`ObjectCentricState` vs `Kinematic3DObjectCentricState` attrs,
untyped substrate calls) and comment/docstring line-length pylint debt (the v1 "residual line-length
debt"); the two code-level pylint issues this work introduced (an unused var, a missing inner-class
docstring) were fixed.

## Deferred (unchanged + new)

The full relocation-aware collection + SPECTRE training on the cluttered env; **r3 F1** (needs a
relocation-aware pool generator); **step-time base-collision enforcement** (needs a navigable floor
layout); learned baselines (PIGINet / VLMPlan / LAZY), the `compare_envs.py` `EnvSpec`, and the
physical-robot phase.
